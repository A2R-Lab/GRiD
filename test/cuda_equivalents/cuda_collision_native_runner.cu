// Gate for NATIVE capsule-row collision (Bundle 1a). Four properties, one exe:
//  (1) VERDICT: the broad(sphere)->fine(capsule) config_free must equal an independent
//      fine-only capsule check on EVERY (config, obstacle) pair — this is the covering
//      property of the DERIVED broad spheres (built from the rows at bake time) plus the
//      grid_cc_config_free_capsule mask logic, end-to-end.
//  (2) NARROWING: on a sparse probe set the mask must show a PARTIAL flag (fine pass ran
//      but skipped rows) — non-vacuous, same policy as the sphere two-tier gate.
//  (3) GRADIENT: collision_distance_gradient (envelope-theorem composition over BOTH
//      endpoint gradients, weighted (1-t*)/t*) vs central finite differences of
//      collision_distance in q. T=double, h=1e-6, max rel err < 1e-4.
//  (4) PAIRS: with a single obstacle, collision_distance_pairs must equal the reduced
//      collision_distance bitwise (same SDF call, no argmin ambiguity).
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using T = double;
namespace gc = grid_collision;
constexpr int NQ = grid::NUM_POS;
constexpr int NV = grid::NUM_VEL;
constexpr int NB = gc::NUM_COLLISION_SPHERES_BROAD;
constexpr int NR = gc::NUM_COLLISION_ROWS;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

// (1)+(2): two-tier verdict vs independent fine-only capsule check, per obstacle.
__global__ void gate_kernel(const T *q0, const grid::robotModel<T> *m,
                            const gc::Sphere<T> *obs, int nobs, int *two_out, int *fine_out, int *rechk_out) {
    __shared__ T s_q[NQ], s_bpos[3*NB], s_br[NB], s_seg[6*NR], s_rr[NR];
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) s_q[i] = q0[i];
    __syncthreads();
    for (int o = 0; o < nobs; ++o) {
        gc::Environment<T> env{ &obs[o], 1, nullptr, 0, nullptr, 0 };
        bool two = gc::config_free<T>(s_q, m, env, s_bpos, s_br, s_seg, s_rr, nullptr);
        __syncthreads();  // s_seg/s_rr now hold this config's fine capsule batch
        bool fine = true;
        if (gc::grid_cc_self_collision_capsules<T>(s_seg, s_rr, gc::g_collision_self_cc_ranges, gc::NUM_COLLISION_SELF_CC_RANGES))
            fine = false;
        else
            for (int i = 0; i < NR; ++i)
                if (gc::grid_cc_capsule_in_environment<T>(env, gc::grid_cc_row_capsule<T>(s_seg, s_rr, i))) { fine = false; break; }
        // narrowing witness with self ranges disabled (env-only), same policy as the sphere gate
        int rc = NR;
        gc::grid_cc_config_free_capsule<T>(env,
            s_bpos, s_br, gc::g_collision_self_cc_ranges_broad, 0, NB, gc::g_collision_sphere_link_broad,
            s_seg, s_rr, gc::g_collision_self_cc_ranges, 0, NR, gc::g_collision_row_link, &rc);
        if (threadIdx.x == 0) { two_out[o] = two ? 1 : 0; fine_out[o] = fine ? 1 : 0; rechk_out[o] = rc; }
        __syncthreads();
    }
}

// (3)+(4): analytic clearance Jacobian vs central FD + pairs/reduced bitwise consistency.
__global__ void fd_kernel(const T *q0, const grid::robotModel<T> *m, const gc::Sphere<T> *obs1,
                          T *maxrel_out, int *pairs_ok_out) {
    __shared__ T s_q[NQ], s_seg[6*NR], s_rr[NR], s_nrm[3*NR], s_t[NR];
    __shared__ T s_dist[NR], s_ddist[NR*NV], s_pg[3*NV*2*NR];
    __shared__ T s_dp[NR], s_dm[NR], s_pdist[NR], s_pnrm[3*NR], s_pt[NR];
    __shared__ T s_maxrel; __shared__ int s_pok;
    gc::Environment<T> env{ obs1, 1, nullptr, 0, nullptr, 0 };
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) s_q[i] = q0[i];
    if (threadIdx.x == 0) { s_maxrel = static_cast<T>(0); s_pok = 1; }
    __syncthreads();
    gc::collision_distance_gradient<T>(s_dist, s_ddist, s_q, m, env, s_seg, s_rr, s_nrm, s_t, s_pg, nullptr);
    __syncthreads();
    // (4) pairs vs reduced: n_obs == 1 -> pair (i,0) must equal the reduced value bitwise
    gc::collision_distance_pairs<T>(s_pdist, s_pnrm, s_pt, s_q, m, env, s_seg, s_rr, nullptr);
    __syncthreads();
    if (threadIdx.x == 0)
        for (int i = 0; i < NR; ++i)
            if (s_pdist[i] != s_dist[i]) s_pok = 0;
    __syncthreads();
    // (3) central FD in each q dof (thread 0 perturbs; the distance calls are block-wide)
    const T h = static_cast<T>(1e-6);
    for (int vi = 0; vi < NV; ++vi) {
        if (threadIdx.x == 0) s_q[vi] += h;
        __syncthreads();
        gc::collision_distance<T>(s_dp, s_pnrm, s_pt, s_q, m, env, s_seg, s_rr, nullptr);
        __syncthreads();
        if (threadIdx.x == 0) s_q[vi] -= static_cast<T>(2) * h;
        __syncthreads();
        gc::collision_distance<T>(s_dm, s_pnrm, s_pt, s_q, m, env, s_seg, s_rr, nullptr);
        __syncthreads();
        if (threadIdx.x == 0) {
            s_q[vi] += h;
            for (int i = 0; i < NR; ++i) {
                T fd = (s_dp[i] - s_dm[i]) / (static_cast<T>(2) * h);
                T an = s_ddist[i*NV + vi];
                T den = fabs(an) > static_cast<T>(1) ? fabs(an) : static_cast<T>(1);
                T rel = fabs(fd - an) / den;
                if (rel > s_maxrel) s_maxrel = rel;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) { *maxrel_out = s_maxrel; *pairs_ok_out = s_pok; }
}

int main(int argc, char **argv) {
    const bool quick = (argc > 1);   // sanitizer runs: coarse obstacle grid + fewer configs
    const double ostep = quick ? 0.4 : 0.15;
    const grid::robotModel<T> *m = grid::init_robotModel<T>();
    size_t sb = grid::MULTI_TARGET_POSITION_BROAD_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t sf = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t sg = grid::MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t smem = sb; if (sf > smem) smem = sf; if (sg > smem) smem = sg;
    cudaFuncSetAttribute(gate_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaFuncSetAttribute(fd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    // DENSE verdict sweep + SPARSE narrowing probe (same policy as the sphere two-tier gate)
    std::vector<gc::Sphere<T>> obs;
    for (double x=-0.6; x<=0.6+1e-9; x+=ostep)
      for (double y=-0.6; y<=0.6+1e-9; y+=ostep)
        for (double z=0.0; z<=1.1+1e-9; z+=ostep)
          obs.push_back(gc::Sphere<T>{x,y,z,0.08});
    int nobs = (int)obs.size();
    std::vector<gc::Sphere<T>> probe;
    for (double x=-0.7; x<=0.7+1e-9; x+=0.35)
      for (double y=-0.7; y<=0.7+1e-9; y+=0.35)
        for (double z=0.1; z<=1.2+1e-9; z+=0.35)
          probe.push_back(gc::Sphere<T>{x,y,z,0.04});
    int nprobe = (int)probe.size();

    gc::Sphere<T> *d_obs; CK(cudaMalloc(&d_obs, nobs*sizeof(gc::Sphere<T>)));
    CK(cudaMemcpy(d_obs, obs.data(), nobs*sizeof(gc::Sphere<T>), cudaMemcpyHostToDevice));
    gc::Sphere<T> *d_probe; CK(cudaMalloc(&d_probe, nprobe*sizeof(gc::Sphere<T>)));
    CK(cudaMemcpy(d_probe, probe.data(), nprobe*sizeof(gc::Sphere<T>), cudaMemcpyHostToDevice));
    int maxo = nobs > nprobe ? nobs : nprobe;
    int *d_two,*d_fine,*d_rechk; CK(cudaMalloc(&d_two,maxo*sizeof(int))); CK(cudaMalloc(&d_fine,maxo*sizeof(int))); CK(cudaMalloc(&d_rechk,maxo*sizeof(int)));
    T *d_q; CK(cudaMalloc(&d_q, NQ*sizeof(T)));

    std::vector<std::vector<T>> configs;
    configs.push_back(std::vector<T>(NQ, 0.0));
    int ncfg = quick ? 1 : 6;
    for (int c=1;c<=ncfg;++c){ std::vector<T> q(NQ); for(int i=0;i<NQ;++i) q[i]=0.4*sin(0.7*i+c)+0.15*c; configs.push_back(q); }

    long total=0, mismatch=0; int first_c=-1,first_o=-1,first_two=-1,first_fine=-1;
    long partial=0, fully_narrowed=0; int min_rechk=NR+1; long probe_collided=0;
    std::vector<int> htwo(maxo), hfine(maxo), hrechk(maxo);
    struct Sweep { gc::Sphere<T> *d; int n; bool witness; };
    Sweep sweeps[2] = { {d_obs, nobs, false}, {d_probe, nprobe, true} };
    for (int s=0;s<2;++s){
      for (int c=0;c<(int)configs.size();++c){
        CK(cudaMemcpy(d_q, configs[c].data(), NQ*sizeof(T), cudaMemcpyHostToDevice));
        gate_kernel<<<1,128,smem>>>(d_q, m, sweeps[s].d, sweeps[s].n, d_two, d_fine, d_rechk);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(htwo.data(), d_two, sweeps[s].n*sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hfine.data(), d_fine, sweeps[s].n*sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hrechk.data(), d_rechk, sweeps[s].n*sizeof(int), cudaMemcpyDeviceToHost));
        for (int o=0;o<sweeps[s].n;++o){
            ++total;
            if(htwo[o]!=hfine[o]){ if(!mismatch){first_c=c;first_o=o;first_two=htwo[o];first_fine=hfine[o];} ++mismatch; }
            if(!sweeps[s].witness) continue;
            int rc=hrechk[o];
            if(rc<min_rechk) min_rechk=rc;
            if(rc==0) ++fully_narrowed;
            else if(rc<NR) ++partial;
            if(hfine[o]==0) ++probe_collided;
        }
      }
    }

    // (3)+(4): gradient FD + pairs consistency, one generic single-obstacle env per config
    gc::Sphere<T> obs1{0.3, 0.2, 0.6, 0.05};
    gc::Sphere<T> *d_obs1; CK(cudaMalloc(&d_obs1, sizeof(gc::Sphere<T>)));
    CK(cudaMemcpy(d_obs1, &obs1, sizeof(gc::Sphere<T>), cudaMemcpyHostToDevice));
    T *d_maxrel; int *d_pok; CK(cudaMalloc(&d_maxrel,sizeof(T))); CK(cudaMalloc(&d_pok,sizeof(int)));
    double grad_maxrel = 0.0; int pairs_ok = 1;
    for (int c=0;c<(int)configs.size();++c){
        CK(cudaMemcpy(d_q, configs[c].data(), NQ*sizeof(T), cudaMemcpyHostToDevice));
        fd_kernel<<<1,128,smem>>>(d_q, m, d_obs1, d_maxrel, d_pok);
        CK(cudaDeviceSynchronize());
        T mr; int pk;
        CK(cudaMemcpy(&mr, d_maxrel, sizeof(T), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&pk, d_pok, sizeof(int), cudaMemcpyDeviceToHost));
        if (mr > grad_maxrel) grad_maxrel = mr;
        if (!pk) pairs_ok = 0;
    }

    printf("NB=%d NR=%d configs=%zu dense=%d sparse=%d pairs=%ld mismatches=%ld (probe collisions=%ld)\n",
           NB, NR, configs.size(), nobs, nprobe, total, mismatch, probe_collided);
    printf("link_CC narrowing (probe): min_rechecked=%d/%d partial(0<rc<NR)=%ld fully_narrowed(rc==0)=%ld\n",
           (min_rechk<=NR?min_rechk:-1), NR, partial, fully_narrowed);
    printf("gradient: fd_maxrel=%.3e  pairs_bitwise_ok=%d\n", grad_maxrel, pairs_ok);
    if (mismatch) printf("first mismatch: config=%d obstacle=%d two=%d fine=%d\n", first_c, first_o, first_two, first_fine);
    bool verdict_ok = (mismatch == 0);
    bool narrowing_nonvacuous = (partial > 0);
    bool grad_ok = (grad_maxrel < 1e-4);
    if (!narrowing_nonvacuous)
        printf("NON-VACUOUS FAIL: no partial link flag observed (mask never skipped a subset) — widen the probe grid.\n");
    if (!grad_ok)
        printf("GRADIENT FAIL: fd_maxrel=%.3e >= 1e-4\n", grad_maxrel);
    bool ok = verdict_ok && narrowing_nonvacuous && grad_ok && pairs_ok;
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 3;
}

// Gate for W3 Increment 2: the broad->fine two-tier config_free must return the SAME verdict as a
// fine-only check on EVERY (configuration, obstacle) pair. The two-tier driver short-circuits the
// FREE case on the coarse tier; correctness requires the broad cover to be conservative (a coarser
// covering encloses the finer one), so "broad clear => definitely free" can never miss a collision
// the fine tier would catch. This sweeps a 3D grid of obstacles across several bent configs and
// asserts two-tier == fine-only bit-for-bit. (T=double for a clean check; production is fp32.)
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using T = double;
namespace gc = grid_collision;
constexpr int NQ = grid::NUM_POS;
constexpr int NB = gc::NUM_COLLISION_SPHERES_BROAD;  // coarse broad-phase tier
constexpr int NF = gc::NUM_COLLISION_SPHERES;         // fine / public tier

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

// One kernel: for each obstacle, run the two-tier config_free AND an independent fine-only check on
// the fine spheres config_free already populated. Emit both verdicts for the host to compare.
__global__ void gate_kernel(const T *q0, const grid::robotModel<T> *m,
                            const gc::Sphere<T> *obs, int nobs, int *two_out, int *fine_out, int *rechk_out) {
    __shared__ T s_q[NQ], s_bpos[3*NB], s_br[NB], s_fpos[3*NF], s_fr[NF];
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) s_q[i] = q0[i];
    __syncthreads();
    for (int o = 0; o < nobs; ++o) {
        gc::Environment<T> env{ &obs[o], 1, nullptr, 0, nullptr, 0 };
        bool two = gc::config_free<T>(s_q, m, env, s_bpos, s_br, s_fpos, s_fr, nullptr);
        __syncthreads();  // s_bpos/s_br/s_fpos/s_fr now hold this config's broad+fine batches
        // Independent fine-only verdict == what a single-tier config_free returns.
        bool fine = true;
        if (gc::grid_cc_self_collision<T>(s_fpos, s_fr, gc::g_collision_self_cc_ranges, gc::NUM_COLLISION_SELF_CC_RANGES))
            fine = false;
        else
            for (int i = 0; i < NF; ++i)
                if (gc::grid_cc_sphere_in_environment<T>(env, s_fpos[3*i], s_fpos[3*i+1], s_fpos[3*i+2], s_fr[i])) { fine = false; break; }
        // link_CC narrowing WITNESS (env-only): this iiwa spherization self-collides at rest (fat
        // overlapping spheres), which would flood the hit-mask on every config and hide the ENV
        // narrowing — the main perf win (the fine ENV loop is the costly one). So measure narrowing
        // on the environment path alone by re-running the driver with self ranges disabled (0), reusing
        // the positions config_free just computed. rc is a thread-LOCAL so the dbg write is race-free.
        int rc = NF;
        gc::grid_cc_config_free<T>(env,
            s_bpos, s_br, gc::g_collision_self_cc_ranges_broad, 0, NB, gc::g_collision_sphere_link_broad,
            s_fpos, s_fr, gc::g_collision_self_cc_ranges, 0, NF, gc::g_collision_sphere_link, &rc);
        if (threadIdx.x == 0) { two_out[o] = two ? 1 : 0; fine_out[o] = fine ? 1 : 0; rechk_out[o] = rc; }
        __syncthreads();
    }
}

int main(int argc, char **argv) {
    const bool quick = (argc > 1);   // sanitizer runs: coarse obstacle grid + 2 configs
    const double ostep = quick ? 0.4 : 0.15;
    const grid::robotModel<T> *m = grid::init_robotModel<T>();
    size_t sb = grid::MULTI_TARGET_POSITION_BROAD_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t sf = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t smem = sb > sf ? sb : sf;
    cudaFuncSetAttribute(gate_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    // DENSE grid spanning the iiwa envelope (VERDICT sweep): radius/spacing chosen so it is nearly
    // space-filling at the collision scale -> most links flag -> stresses two-tier==fine-only, but by
    // design gives NO link_CC narrowing (every broad sphere hits some obstacle).
    std::vector<gc::Sphere<T>> obs;
    for (double x=-0.6; x<=0.6+1e-9; x+=ostep)
      for (double y=-0.6; y<=0.6+1e-9; y+=ostep)
        for (double z=0.0; z<=1.1+1e-9; z+=ostep)
          obs.push_back(gc::Sphere<T>{x,y,z,0.08});
    int nobs = (int)obs.size();
    // SPARSE probe set (NARROWING witness): small, widely-spaced obstacles so each touches at most a
    // local cluster of links -> partial link flags (0<rc<NF) and clear-far ones (rc==0). Same
    // covering-property correctness (verdict still checked below), but exercises the mask skip-logic.
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

    // A battery of configs: zero, and several bent poses.
    std::vector<std::vector<T>> configs;
    configs.push_back(std::vector<T>(NQ, 0.0));
    int ncfg = quick ? 1 : 6;
    for (int c=1;c<=ncfg;++c){ std::vector<T> q(NQ); for(int i=0;i<NQ;++i) q[i]=0.4*sin(0.7*i+c)+0.15*c; configs.push_back(q); }

    // Verdict sweep (DENSE) + narrowing witness (SPARSE probe). Both assert two-tier==fine-only; the
    // probe additionally must show a PARTIAL flag (0<rc<NF) — the fine pass ran AND skipped spheres.
    long total=0, mismatch=0; int first_c=-1,first_o=-1,first_two=-1,first_fine=-1;
    long partial=0, fully_narrowed=0; int min_rechk=NF+1; long last_collided=0;
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
            if(!sweeps[s].witness) continue;          // narrowing stats only from the sparse probe
            int rc=hrechk[o];
            if(rc<min_rechk) min_rechk=rc;
            if(rc==0) ++fully_narrowed;               // broad clear: fine pass fully skipped
            else if(rc<NF) ++partial;                 // some links flagged, others skipped
            if(hfine[o]==0) ++last_collided;
        }
      }
    }

    printf("NB=%d NF=%d  configs=%zu  dense=%d sparse=%d  pairs=%ld  mismatches=%ld  (probe collisions=%ld)\n",
           NB, NF, configs.size(), nobs, nprobe, total, mismatch, last_collided);
    printf("link_CC narrowing (probe): min_rechecked=%d/%d  partial(0<rc<NF)=%ld  fully_narrowed(rc==0)=%ld\n",
           (min_rechk<=NF?min_rechk:-1), NF, partial, fully_narrowed);
    if (mismatch) printf("first mismatch: config=%d obstacle=%d two=%d fine=%d\n", first_c, first_o, first_two, first_fine);
    // PASS requires BOTH: (1) two-tier verdict == fine-only on every pair (correctness), and (2) the
    // narrowing is non-vacuous — at least one PARTIAL flag where the fine pass ran but skipped spheres.
    bool verdict_ok = (mismatch == 0);
    bool narrowing_nonvacuous = (partial > 0);
    if (!narrowing_nonvacuous)
        printf("NON-VACUOUS FAIL: no partial link flag observed (mask never skipped a subset) — widen the probe grid.\n");
    bool ok = verdict_ok && narrowing_nonvacuous;
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 3;
}

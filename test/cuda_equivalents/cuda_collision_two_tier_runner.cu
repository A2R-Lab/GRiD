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
                            const gc::Sphere<T> *obs, int nobs, int *two_out, int *fine_out) {
    __shared__ T s_q[NQ], s_bpos[3*NB], s_br[NB], s_fpos[3*NF], s_fr[NF];
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) s_q[i] = q0[i];
    __syncthreads();
    for (int o = 0; o < nobs; ++o) {
        gc::Environment<T> env{ &obs[o], 1, nullptr, 0, nullptr, 0 };
        bool two = gc::config_free<T>(s_q, m, env, s_bpos, s_br, s_fpos, s_fr, nullptr);
        __syncthreads();  // s_fpos/s_fr hold the fine batch config_free computed
        // Independent fine-only verdict == what a single-tier config_free returns.
        bool fine = true;
        if (gc::grid_cc_self_collision<T>(s_fpos, s_fr, gc::g_collision_self_cc_ranges, gc::NUM_COLLISION_SELF_CC_RANGES))
            fine = false;
        else
            for (int i = 0; i < NF; ++i)
                if (gc::grid_cc_sphere_in_environment<T>(env, s_fpos[3*i], s_fpos[3*i+1], s_fpos[3*i+2], s_fr[i])) { fine = false; break; }
        if (threadIdx.x == 0) { two_out[o] = two ? 1 : 0; fine_out[o] = fine ? 1 : 0; }
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

    // Obstacle grid spanning the iiwa envelope; radius chosen so many but not all configs collide.
    std::vector<gc::Sphere<T>> obs;
    for (double x=-0.6; x<=0.6+1e-9; x+=ostep)
      for (double y=-0.6; y<=0.6+1e-9; y+=ostep)
        for (double z=0.0; z<=1.1+1e-9; z+=ostep)
          obs.push_back(gc::Sphere<T>{x,y,z,0.08});
    int nobs = (int)obs.size();

    gc::Sphere<T> *d_obs; CK(cudaMalloc(&d_obs, nobs*sizeof(gc::Sphere<T>)));
    CK(cudaMemcpy(d_obs, obs.data(), nobs*sizeof(gc::Sphere<T>), cudaMemcpyHostToDevice));
    int *d_two,*d_fine; CK(cudaMalloc(&d_two,nobs*sizeof(int))); CK(cudaMalloc(&d_fine,nobs*sizeof(int)));
    T *d_q; CK(cudaMalloc(&d_q, NQ*sizeof(T)));

    // A battery of configs: zero, and several bent poses.
    std::vector<std::vector<T>> configs;
    configs.push_back(std::vector<T>(NQ, 0.0));
    int ncfg = quick ? 1 : 6;
    for (int c=1;c<=ncfg;++c){ std::vector<T> q(NQ); for(int i=0;i<NQ;++i) q[i]=0.4*sin(0.7*i+c)+0.15*c; configs.push_back(q); }

    long total=0, mismatch=0; int first_c=-1,first_o=-1,first_two=-1,first_fine=-1;
    std::vector<int> htwo(nobs), hfine(nobs);
    for (int c=0;c<(int)configs.size();++c){
        CK(cudaMemcpy(d_q, configs[c].data(), NQ*sizeof(T), cudaMemcpyHostToDevice));
        gate_kernel<<<1,128,smem>>>(d_q, m, d_obs, nobs, d_two, d_fine);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(htwo.data(), d_two, nobs*sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hfine.data(), d_fine, nobs*sizeof(int), cudaMemcpyDeviceToHost));
        for (int o=0;o<nobs;++o){ ++total; if(htwo[o]!=hfine[o]){ if(!mismatch){first_c=c;first_o=o;first_two=htwo[o];first_fine=hfine[o];} ++mismatch; } }
    }

    // Count collisions on the LAST config's sweep so the gate is demonstrably non-vacuous.
    long last_collided=0; for(int o=0;o<nobs;++o) if(hfine[o]==0) ++last_collided;
    printf("NB=%d NF=%d  configs=%zu obstacles=%d  pairs=%ld  mismatches=%ld  (last-config collisions=%ld)\n",
           NB, NF, configs.size(), nobs, total, mismatch, last_collided);
    if (mismatch) printf("first mismatch: config=%d obstacle=%d two=%d fine=%d\n", first_c, first_o, first_two, first_fine);
    bool ok = (mismatch == 0);
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 3;
}

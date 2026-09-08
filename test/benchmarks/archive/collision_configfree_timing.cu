// config_free latency cell (night queue, Bundle 1a follow-up): times the broad->fine
// config_free over a batch of random configs against a fixed obstacle set. Compiled twice
// per A/B arm — once against the SPHERIZED header (default) and once against the NATIVE
// capsule-row header (-DGRID_CC_NATIVE, which only switches the fine-tier scratch sizes;
// both config_free overloads share the same 4-scratch call shape). Interleaving of the
// two exes across reps happens at the script level (same-run pairs, per A/B policy).
// Output: one line "config_free_us_per_config=<v> free_frac=<f> B=<B> iters=<I>".
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

using T = float;   // production collision precision
namespace gc = grid_collision;
constexpr int NQ = grid::NUM_POS;
constexpr int NB_ = gc::NUM_COLLISION_SPHERES_BROAD;
#ifdef GRID_CC_NATIVE
constexpr int FINE_POS_SZ = 6 * gc::NUM_COLLISION_ROWS;
constexpr int FINE_R_SZ   = gc::NUM_COLLISION_ROWS;
#else
constexpr int FINE_POS_SZ = 3 * gc::NUM_COLLISION_SPHERES;
constexpr int FINE_R_SZ   = gc::NUM_COLLISION_SPHERES;
#endif

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(2);} }while(0)

__global__ void time_kernel(const T *d_qs, int B, const grid::robotModel<T> *m,
                            const gc::Sphere<T> *obs, int nobs, const gc::Plane<T> *pl,
                            int *d_free_count) {
    __shared__ T s_q[NQ], s_bpos[3*NB_], s_br[NB_], s_fpos[FINE_POS_SZ], s_fr[FINE_R_SZ];
    __shared__ int s_free;
    if (threadIdx.x == 0) s_free = 0;
    __syncthreads();
    gc::Environment<T> env{ obs, nobs, nullptr, 0, nullptr, 0, pl, 1 };
    for (int b = blockIdx.x; b < B; b += gridDim.x) {
        for (int i = threadIdx.x; i < NQ; i += blockDim.x) s_q[i] = d_qs[b*NQ + i];
        __syncthreads();
        bool f = gc::config_free<T>(s_q, m, env, s_bpos, s_br, s_fpos, s_fr, nullptr);
        if (threadIdx.x == 0 && f) atomicAdd(&s_free, 1);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(d_free_count, s_free);
}

int main(int argc, char **argv) {
    const int B = argc > 1 ? atoi(argv[1]) : 1024;
    const int ITERS = argc > 2 ? atoi(argv[2]) : 50;
    const int threads = argc > 3 ? atoi(argv[3]) : 128;
    const grid::robotModel<T> *m = grid::init_robotModel<T>();
    size_t sb = grid::MULTI_TARGET_POSITION_BROAD_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t sf = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t smem = sb > sf ? sb : sf;
    cudaFuncSetAttribute(time_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    // fixed obstacle set: 8 spheres ringing the workspace + the ground plane
    std::vector<gc::Sphere<T>> obs;
    for (int k = 0; k < 8; ++k) {
        float a = 0.785398f * k;
        obs.push_back(gc::Sphere<T>{0.55f*cosf(a), 0.55f*sinf(a), 0.35f + 0.08f*k, 0.06f});
    }
    gc::Plane<T> pl{0.f, 0.f, 1.f, -0.05f};
    gc::Sphere<T> *d_obs; CK(cudaMalloc(&d_obs, obs.size()*sizeof(gc::Sphere<T>)));
    CK(cudaMemcpy(d_obs, obs.data(), obs.size()*sizeof(gc::Sphere<T>), cudaMemcpyHostToDevice));
    gc::Plane<T> *d_pl; CK(cudaMalloc(&d_pl, sizeof(pl)));
    CK(cudaMemcpy(d_pl, &pl, sizeof(pl), cudaMemcpyHostToDevice));

    // deterministic LCG configs (same seed both arms -> identical work)
    std::vector<T> qs(B * NQ);
    unsigned s = 12345u;
    for (auto &v : qs) { s = s * 1664525u + 1013904223u; v = ((s >> 8) & 0xFFFF) / 65535.0f * 4.0f - 2.0f; }
    T *d_qs; CK(cudaMalloc(&d_qs, qs.size()*sizeof(T)));
    CK(cudaMemcpy(d_qs, qs.data(), qs.size()*sizeof(T), cudaMemcpyHostToDevice));
    int *d_free; CK(cudaMalloc(&d_free, sizeof(int)));

    // warmup
    CK(cudaMemset(d_free, 0, sizeof(int)));
    time_kernel<<<64, threads, smem>>>(d_qs, B, m, d_obs, (int)obs.size(), d_pl, d_free);
    CK(cudaDeviceSynchronize());
    int free_n; CK(cudaMemcpy(&free_n, d_free, sizeof(int), cudaMemcpyDeviceToHost));

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    for (int it = 0; it < ITERS; ++it)
        time_kernel<<<64, threads, smem>>>(d_qs, B, m, d_obs, (int)obs.size(), d_pl, d_free);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("config_free_us_per_config=%.5f free_frac=%.4f B=%d iters=%d threads=%d\n",
           1000.0f * ms / (float)(ITERS * B), (float)free_n / (float)B, B, ITERS, threads);
    return 0;
}

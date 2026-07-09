// Validation for W3 Step 3 grid_collision::config_free.
//
// End-to-end wiring test: sphere WORLD positions via the W1b batched extractor
// (grid::multi_target_position_device) -> environment SDF checks (grid_cc_sphere_in_environment)
// -> collision-free verdict. Self-consistent, no external oracle:
//   * empty/far environment (+ tiny radii) => config_free == TRUE  (free)
//   * an obstacle sphere placed exactly ON sphere 0's world position => config_free == FALSE
// The SDF primitives + baked-range self-collision are unit-tested separately
// (test_cuda_collision_geometry.py); this runner certifies the extractor->config_free binding.
//
// fp32 (collision change-of-record). Correctness only, no timing.
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using T = float;
namespace gc = grid_collision;
constexpr int NQ = grid::NUM_POS;
constexpr int NS = gc::NUM_COLLISION_SPHERES;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

// Run config_free against a caller-provided obstacle list; export the verdict + sphere positions.
__global__ void cf_kernel(const T *d_q, const grid::robotModel<T> *m,
                          const gc::Sphere<T> *d_obst, int n_obst, int *d_free, T *d_pos) {
    __shared__ T s_pos[3*NS];
    __shared__ T s_r[NS];
    gc::Environment<T> env{ d_obst, n_obst, nullptr, 0, nullptr, 0 };
    bool is_free = gc::config_free<T>(d_q, m, env, s_pos, s_r, nullptr);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        *d_free = is_free ? 1 : 0;
        for (int i = 0; i < 3*NS; ++i) d_pos[i] = s_pos[i];
    }
}

int main(){
    const grid::robotModel<T> *d_m = grid::init_robotModel<T>();
    size_t smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();

    std::vector<T> hq(NQ); for(int i=0;i<NQ;++i) hq[i]=0.2f*sinf(0.7f*i)+0.1f;
    T *d_q, *d_pos; int *d_free; gc::Sphere<T> *d_obst;
    CK(cudaMalloc(&d_q, NQ*sizeof(T)));
    CK(cudaMalloc(&d_pos, 3*NS*sizeof(T)));
    CK(cudaMalloc(&d_free, sizeof(int)));
    CK(cudaMalloc(&d_obst, sizeof(gc::Sphere<T>)));
    CK(cudaMemcpy(d_q, hq.data(), NQ*sizeof(T), cudaMemcpyHostToDevice));
    cudaFuncSetAttribute(cf_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    // ---- 1. no obstacles: expect FREE (tiny radii => no self-collision) ----
    int free_empty; std::vector<T> pos(3*NS);
    cf_kernel<<<1,256,smem>>>(d_q, d_m, nullptr, 0, d_free, d_pos); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&free_empty, d_free, sizeof(int), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(pos.data(), d_pos, 3*NS*sizeof(T), cudaMemcpyDeviceToHost));
    printf("EMPTY free=%d  sphere0=(% .6f % .6f % .6f)\n", free_empty, pos[0], pos[1], pos[2]);

    // ---- 2. obstacle ON sphere 0: expect COLLISION (not free) ----
    gc::Sphere<T> hit{ pos[0], pos[1], pos[2], 0.5f };
    CK(cudaMemcpy(d_obst, &hit, sizeof(hit), cudaMemcpyHostToDevice));
    int free_hit;
    cf_kernel<<<1,256,smem>>>(d_q, d_m, d_obst, 1, d_free, d_pos); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&free_hit, d_free, sizeof(int), cudaMemcpyDeviceToHost));
    printf("ONHIT free=%d\n", free_hit);

    // ---- 3. obstacle far away: expect FREE ----
    gc::Sphere<T> far{ 1.0e3f, 1.0e3f, 1.0e3f, 0.01f };
    CK(cudaMemcpy(d_obst, &far, sizeof(far), cudaMemcpyHostToDevice));
    int free_far;
    cf_kernel<<<1,256,smem>>>(d_q, d_m, d_obst, 1, d_free, d_pos); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&free_far, d_free, sizeof(int), cudaMemcpyDeviceToHost));
    printf("FAR free=%d\n", free_far);

    printf("NS=%d NRANGES=%d\n", NS, gc::NUM_COLLISION_SELF_CC_RANGES);
    bool ok = (free_empty==1) && (free_hit==0) && (free_far==1);
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 3;
}

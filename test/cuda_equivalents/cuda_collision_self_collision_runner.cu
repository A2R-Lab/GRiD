// Validation for W3 Increment 0: the self-collision path THROUGH grid_collision::config_free.
//
// config_free runs grid_cc_self_collision over the baked g_collision_self_cc_ranges before the
// environment checks. The SDF + range table are unit-tested standalone (test_cuda_collision_geometry.py
// / build_self_cc_ranges), but the config_free -> self_collision binding was previously only exercised
// with tiny radii (never actually self-colliding). This runner drives config_free with an EMPTY
// environment so the ONLY thing that can flip the verdict is the self-collision range loop, then
// exports the verdict for the Python gate to assert against the crafted spec:
//   * huge radii on a NON-ADJACENT sphere pair => self-collision => config_free == FALSE
//   * huge radii on an ADJACENT-only pair (excluded from the ranges) => config_free == TRUE (free)
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

// config_free with an EMPTY environment: verdict is decided purely by grid_cc_self_collision.
__global__ void selfcc_kernel(const T *d_q, const grid::robotModel<T> *m, int *d_free) {
    __shared__ T s_pos[3*NS];
    __shared__ T s_r[NS];
    gc::Environment<T> env{ nullptr, 0, nullptr, 0, nullptr, 0 };
    bool is_free = gc::config_free<T>(d_q, m, env, s_pos, s_r, nullptr);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) *d_free = is_free ? 1 : 0;
}

int main(){
    const grid::robotModel<T> *d_m = grid::init_robotModel<T>();
    size_t smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();

    std::vector<T> hq(NQ); for(int i=0;i<NQ;++i) hq[i]=0.2f*sinf(0.7f*i)+0.1f;
    T *d_q; int *d_free;
    CK(cudaMalloc(&d_q, NQ*sizeof(T)));
    CK(cudaMalloc(&d_free, sizeof(int)));
    CK(cudaMemcpy(d_q, hq.data(), NQ*sizeof(T), cudaMemcpyHostToDevice));
    cudaFuncSetAttribute(selfcc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    int empty_free;
    selfcc_kernel<<<1,256,smem>>>(d_q, d_m, d_free); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&empty_free, d_free, sizeof(int), cudaMemcpyDeviceToHost));

    printf("SELFCC empty_free=%d NS=%d NRANGES=%d\n", empty_free, NS, gc::NUM_COLLISION_SELF_CC_RANGES);
    printf("RESULT: PASS\n");   // "PASS" == the runner executed cleanly; the Python gate asserts empty_free.
    return 0;
}

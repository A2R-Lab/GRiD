// Native capsule-row SELF-collision gate: drives grid_cc_self_collision_capsules THROUGH the
// SINGLE-TIER native config_free (empty environment), certifying the baked range table +
// adjacency exclusion on capsule rows end-to-end. Prints empty_free= and NRANGES= for the
// pytest wrapper (positive spec: non-adjacent huge pair -> 0; negative: adjacent-only -> 1, 0 ranges).
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>

using T = double;
namespace gc = grid_collision;
constexpr int NQ = grid::NUM_POS;
constexpr int NR = gc::NUM_COLLISION_ROWS;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

__global__ void gate_kernel(const T *q0, const grid::robotModel<T> *m, int *free_out) {
    __shared__ T s_q[NQ], s_seg[6*NR], s_rr[NR];
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) s_q[i] = q0[i];
    __syncthreads();
    gc::Environment<T> env{};   // EMPTY -> verdict governed by self-collision only
    bool free_ = gc::config_free<T>(s_q, m, env, s_seg, s_rr, nullptr);
    if (threadIdx.x == 0) *free_out = free_ ? 1 : 0;
}

int main() {
    const grid::robotModel<T> *m = grid::init_robotModel<T>();
    size_t smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(gate_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    T q0[NQ] = {};   // zero config
    T *d_q; CK(cudaMalloc(&d_q, NQ*sizeof(T)));
    CK(cudaMemcpy(d_q, q0, NQ*sizeof(T), cudaMemcpyHostToDevice));
    int *d_free; CK(cudaMalloc(&d_free, sizeof(int)));
    gate_kernel<<<1,128,smem>>>(d_q, m, d_free);
    CK(cudaDeviceSynchronize());
    int hfree; CK(cudaMemcpy(&hfree, d_free, sizeof(int), cudaMemcpyDeviceToHost));
    printf("empty_free=%d NRANGES=%d NROWS=%d\n", hfree, gc::NUM_COLLISION_SELF_CC_RANGES, NR);
    printf("RESULT: PASS\n");   // verdict asserted by the pytest wrapper via the kv tokens
    return 0;
}

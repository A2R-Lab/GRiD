// Robot-general validation for W2a grid::multi_target_position_gradient_device.
//
// Prints d(world pos)/dv for every baked target (3 x NV per target, row-fastest) and the
// end_effector_pose_gradient position rows (rows 0..2 per ee) so the Python test can
// compare against a central-difference FD oracle and assert the offset==0 targets equal
// the corresponding end_effector_pose_gradient rows (bit-identical: same Jv fill).
//
// Also self-checks THREAD-INVARIANCE at 1 / 32 / 256 threads. Correctness only, no timing.
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using T = double;
constexpr int NQ  = grid::NUM_POS;
constexpr int NV  = grid::NUM_VEL;
constexpr int NT  = grid::NUM_MULTI_TARGETS;
constexpr int NEE = grid::NUM_EES;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

__global__ void mtg_kernel(T *d_out, const T *d_q, const grid::robotModel<T> *m) {
    __shared__ T s_out[3*NV*NT];
    grid::multi_target_position_gradient_device<T>(s_out, d_q, m);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
        for (int i = 0; i < 3*NV*NT; ++i) d_out[i] = s_out[i];
}

// Forced-spill twin: TIER_MINIMAL routes the Jacobian scratch (Xworld|Jv|Jw|ro) to
// d_workspace. Output must be BIT-identical to TIER_SHARED (whole-arena spill only relocates).
__global__ void mtg_kernel_spill(T *d_out, const T *d_q, const grid::robotModel<T> *m, T *d_ws) {
    __shared__ T s_out[3*NV*NT];
    grid::multi_target_position_gradient_device<T, grid::TIER_MINIMAL>(s_out, d_q, m, d_ws);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
        for (int i = 0; i < 3*NV*NT; ++i) d_out[i] = s_out[i];
}

__global__ void eeg_kernel(T *d_g, const T *d_q, const grid::robotModel<T> *m) {
    __shared__ T s_g[6*NV*NEE];
    grid::end_effector_pose_gradient_device<T>(s_g, d_q, m);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
        for (int i = 0; i < 6*NV*NEE; ++i) d_g[i] = s_g[i];
}

int main(){
    const grid::robotModel<T> *d_m = grid::init_robotModel<T>();
    size_t smem = std::max(grid::MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(),
                           grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>());

    std::vector<T> hq(NQ); for(int i=0;i<NQ;++i) hq[i]=0.2*sin(0.7*i)+0.1;
    T *d_q,*d_out,*d_g;
    CK(cudaMalloc(&d_q,NQ*sizeof(T)));
    CK(cudaMalloc(&d_out,3*NV*NT*sizeof(T)));
    CK(cudaMalloc(&d_g,6*NV*NEE*sizeof(T)));
    CK(cudaMemcpy(d_q,hq.data(),NQ*sizeof(T),cudaMemcpyHostToDevice));

    cudaFuncSetAttribute(mtg_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    cudaFuncSetAttribute(eeg_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);

    // ---- thread-invariance: identical output at 1 / 32 / 256 threads ----
    const int tc[3] = {1, 32, 256};
    std::vector<std::vector<T>> res(3, std::vector<T>(3*NV*NT));
    for (int k=0;k<3;++k){
        mtg_kernel<<<1,tc[k],smem>>>(d_out,d_q,d_m); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(res[k].data(),d_out,3*NV*NT*sizeof(T),cudaMemcpyDeviceToHost));
    }
    double tinv=0; for(int k=1;k<3;++k) for(int i=0;i<3*NV*NT;++i) tinv=std::max(tinv,fabs(res[k][i]-res[0][i]));
    printf("THREADINV maxdiff=%.3e\n", tinv);

    // ---- forced-spill: TIER_MINIMAL scratch->d_workspace must be BIT-identical ----
    size_t smem_spill = grid::MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::TIER_MINIMAL>();
    size_t ws_bytes   = grid::MULTI_TARGET_POSITION_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES<T, grid::TIER_MINIMAL>();
    T *d_ws=nullptr; if (ws_bytes) CK(cudaMalloc(&d_ws, ws_bytes));
    cudaFuncSetAttribute(mtg_kernel_spill, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_spill);
    std::vector<T> res_spill(3*NV*NT);
    mtg_kernel_spill<<<1,256,smem_spill>>>(d_out,d_q,d_m,d_ws); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(res_spill.data(),d_out,3*NV*NT*sizeof(T),cudaMemcpyDeviceToHost));
    double spilldiff=0; for(int i=0;i<3*NV*NT;++i) spilldiff=std::max(spilldiff,fabs(res_spill[i]-res[2][i]));
    printf("SPILLDIFF maxdiff=%.3e (smem %zu->%zu, ws %zu B)\n",
           spilldiff, grid::MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(), smem_spill, ws_bytes);

    std::vector<T> hg(6*NV*NEE);
    eeg_kernel<<<1,256,smem>>>(d_g,d_q,d_m); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hg.data(),d_g,6*NV*NEE*sizeof(T),cudaMemcpyDeviceToHost));

    for(int t=0;t<NT;++t) for(int vi=0;vi<NV;++vi){ int ob=3*(NV*t+vi);
        printf("MTG %d %d % .17g % .17g % .17g\n", t, vi, res[2][ob+0], res[2][ob+1], res[2][ob+2]); }
    for(int e=0;e<NEE;++e) for(int vi=0;vi<NV;++vi){ int gb=6*(NV*e+vi);
        printf("EEG %d %d % .17g % .17g % .17g\n", e, vi, hg[gb+0], hg[gb+1], hg[gb+2]); }

    if (tinv > 1e-9) { printf("RESULT: FAIL (thread-variance %.3e)\n", tinv); return 3; }
    if (spilldiff != 0.0) { printf("RESULT: FAIL (spill non-bit-identical %.3e)\n", spilldiff); return 4; }
    printf("RESULT: PASS\n");
    return 0;
}

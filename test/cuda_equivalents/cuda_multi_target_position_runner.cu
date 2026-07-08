// Robot-general validation for W1b grid::multi_target_position_device.
//
// Prints each baked target's world position (from the batched extraction) and each
// end-effector's world position (from end_effector_pose_device) so the Python test
// can compare against a NumPy world-FK oracle (Xw[anchor] @ [offset,1]) and assert
// the offset==0 targets equal the corresponding end_effector_pose positions.
//
// Also self-checks THREAD-INVARIANCE: the single-block kernel must produce identical
// output at 1 / 32 / 256 threads (a core GRiD invariant). Correctness only, no timing.
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using T = double;
constexpr int NQ  = grid::NUM_POS;
constexpr int NT  = grid::NUM_MULTI_TARGETS;
constexpr int NEE = grid::NUM_EES;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

// Batched multi-target positions -> d_out (3*NT).
__global__ void mt_kernel(T *d_out, const T *d_q, const grid::robotModel<T> *m) {
    __shared__ T s_out[3*NT];
    grid::multi_target_position_device<T>(s_out, d_q, m);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
        for (int i = 0; i < 3*NT; ++i) d_out[i] = s_out[i];
}

// End-effector poses (6 per ee) -> d_pose (6*NEE); rows 0..2 per ee are the xyz.
__global__ void ee_kernel(T *d_pose, const T *d_q, const grid::robotModel<T> *m) {
    __shared__ T s_pose[6*NEE];
    grid::end_effector_pose_device<T>(s_pose, d_q, m);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
        for (int i = 0; i < 6*NEE; ++i) d_pose[i] = s_pose[i];
}

int main(){
    const grid::robotModel<T> *d_m = grid::init_robotModel<T>();
    size_t smem = std::max(grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>(),
                           grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>());

    std::vector<T> hq(NQ); for(int i=0;i<NQ;++i) hq[i]=0.2*sin(0.7*i)+0.1;
    T *d_q,*d_out,*d_pose;
    CK(cudaMalloc(&d_q,NQ*sizeof(T)));
    CK(cudaMalloc(&d_out,3*NT*sizeof(T)));
    CK(cudaMalloc(&d_pose,6*NEE*sizeof(T)));
    CK(cudaMemcpy(d_q,hq.data(),NQ*sizeof(T),cudaMemcpyHostToDevice));

    cudaFuncSetAttribute(mt_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    cudaFuncSetAttribute(ee_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);

    // ---- thread-invariance: identical output at 1 / 32 / 256 threads ----
    const int threadCounts[3] = {1, 32, 256};
    std::vector<std::vector<T>> res(3, std::vector<T>(3*NT));
    for (int k=0;k<3;++k){
        mt_kernel<<<1,threadCounts[k],smem>>>(d_out,d_q,d_m); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(res[k].data(),d_out,3*NT*sizeof(T),cudaMemcpyDeviceToHost));
    }
    double tinv=0; for(int k=1;k<3;++k) for(int i=0;i<3*NT;++i) tinv=std::max(tinv,fabs(res[k][i]-res[0][i]));
    printf("THREADINV maxdiff=%.3e\n", tinv);

    // ee poses (256 threads)
    std::vector<T> hpose(6*NEE);
    ee_kernel<<<1,256,smem>>>(d_pose,d_q,d_m); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hpose.data(),d_pose,6*NEE*sizeof(T),cudaMemcpyDeviceToHost));

    // dump (use the 256-thread multi-target result)
    for(int t=0;t<NT;++t) printf("MT %d % .17g % .17g % .17g\n", t, res[2][3*t+0], res[2][3*t+1], res[2][3*t+2]);
    for(int e=0;e<NEE;++e) printf("EE %d % .17g % .17g % .17g\n", e, hpose[6*e+0], hpose[6*e+1], hpose[6*e+2]);

    if (tinv > 1e-9) { printf("RESULT: FAIL (thread-variance %.3e)\n", tinv); return 3; }
    printf("RESULT: PASS\n");
    return 0;
}

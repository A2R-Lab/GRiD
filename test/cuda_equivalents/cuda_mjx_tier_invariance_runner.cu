// mjx second-order tier-invariance runner (B3-narrow).
//
// The MuJoCo-convention ("mjx") twins of the second-order kernels (idsva_so /
// fdsva_so) run a post-pass epilogue that reuses the SO-temp region of
// d_workspace as `d_mjx_scratch`. At a SPILLED resource tier that same region
// ALSO holds the pin inner's spilled s_temp arena (d_temp_spill) -- they alias
// by design (time-disjoint: the inner arena is dead once the epilogue starts).
// This runner proves the aliasing is safe: it launches the mjx kernel at
// TIER_SHARED (nothing spilled) and at TIER_MINIMAL (whole inner arena spilled,
// so d_mjx_scratch overlaps the just-freed arena) on IDENTICAL input and asserts
// the two outputs are BIT-identical, across a thread-count sweep (also a
// thread-invariance check). Any read of the still-live spilled arena, or a
// missing write-before-read barrier around the reuse, breaks bit-identity.
//
// Build: nvcc -arch=sm_120 -O2 -DMJX_ALGO_FDSVA={0,1} runner.cu (needs a mjx
// header: enable_mujoco_kernels=True, floating go2). Prints "RESULT: PASS/FAIL".
#include <cstdio>
#include <cmath>
#include <vector>
#include "grid.cuh"
using namespace grid;
using T = float;

#ifndef MJX_ALGO_FDSVA
#define MJX_ALGO_FDSVA 0
#endif

#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s @%d: %s\nRESULT: FAIL\n",#x,__LINE__,cudaGetErrorString(e)); return 2;}}while(0)

template<int TIER>
static void launch(gridData<T>* hd, const robotModel<T>* rm, int threads){
#if MJX_ALGO_FDSVA
    size_t smem = FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T,TIER>();
    cudaFuncSetAttribute(fdsva_so_kernel<T,TIER,true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    fdsva_so_kernel<T,TIER,true><<<dim3(1),dim3(threads),smem>>>(
        hd->d_df2, hd->d_workspace, hd->d_q_qd_u, Q_QD_U_STRIDE, hd->d_idsva_so, rm, (T)-9.81, 1);
#else
    size_t smem = IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES<T,TIER>();
    cudaFuncSetAttribute(idsva_so_world_frame_kernel<T,TIER,true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    idsva_so_world_frame_kernel<T,TIER,true><<<dim3(1),dim3(threads),smem>>>(
        hd->d_idsva_so, hd->d_workspace, hd->d_q_qd_u, Q_QD_U_STRIDE, rm, (T)-9.81, 1);
#endif
}

// The comparison buffer: fdsva writes d_df2, idsva writes d_idsva_so.
static inline T* out_ptr(gridData<T>* hd){
#if MJX_ALGO_FDSVA
    return hd->d_df2;
#else
    return hd->d_idsva_so;
#endif
}

int main(int /*argc*/, char** /*argv*/){
    auto* hd = init_gridData<T,1>();
    auto* rm = init_robotModel<T>();
    // deterministic input; a normalized base quaternion at q[3..6] (xyzw).
    std::vector<T> in(Q_QD_U_STRIDE);
    for(int i=0;i<Q_QD_U_STRIDE;i++) in[i]=(T)(0.3*std::sin(0.7*i+1.0)+0.11*i*0.01);
    { T x=0.1f,y=0.2f,z=0.3f,w=1.0f; T n=std::sqrt(x*x+y*y+z*z+w*w); in[3]=x/n;in[4]=y/n;in[5]=z/n;in[6]=w/n; }
    CK(cudaMemcpy(hd->d_q_qd_u, in.data(), Q_QD_U_STRIDE*sizeof(T), cudaMemcpyHostToDevice));

    const int N = SECOND_ORDER_TENSOR_SIZE;
    std::vector<T> A(N), B(N);
    const int threadsets[] = {32, 128, 256};
    double worst = 0; int worstT = 0, worst_nnan = 0;
    for(int ti=0; ti<3; ti++){
        int th = threadsets[ti];
        CK(cudaMemset(out_ptr(hd), 0, N*sizeof(T)));
        launch<TIER_SHARED>(hd, rm, th);  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(A.data(), out_ptr(hd), N*sizeof(T), cudaMemcpyDeviceToHost));
        CK(cudaMemset(out_ptr(hd), 0, N*sizeof(T)));
        launch<TIER_MINIMAL>(hd, rm, th); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(B.data(), out_ptr(hd), N*sizeof(T), cudaMemcpyDeviceToHost));
        double md=0; int nnan=0; double amax=0;
        for(int i=0;i<N;i++){ if(std::isnan(A[i])||std::isnan(B[i])) nnan++;
            double d=std::fabs((double)A[i]-(double)B[i]); if(d>md)md=d;
            double a=std::fabs((double)A[i]); if(a>amax)amax=a; }
        printf("threads=%3d  SHARED vs MINIMAL(spilled)  max|d|=%.3e  nnan=%d  |A|max=%.3e\n", th, md, nnan, amax);
        if(md>worst){worst=md;worstT=th;} worst_nnan += nnan;
    }
    bool pass = (worst == 0.0) && (worst_nnan == 0);
    printf("WORST tier-invariance delta = %.3e @threads=%d  nnan=%d\n", worst, worstT, worst_nnan);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

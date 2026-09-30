// GATO nit 2 (2026-09-24): grid_plant::multi_target_position[_gradient] raw evaluators must equal the
// generated multi_target_position[_gradient]_device fns (same checks as the contact-frame runner:
// PLANTDIFF / THREADINV / SPILLDIFF). Derived from cuda_contact_frame_positions_runner.cu by renaming.
// Validation for the multi-target position family (GATO ask 2026-09-20):
//   grid::multi_target_position_device            -> 3*NUM_MULTI_TARGETS world positions of the
//                                                     baked contact ORIGINS (the f_ext_body wrench points)
//   grid::multi_target_position_gradient_device   -> 3*NUM_VEL per frame tangent Jacobian
//                                                     [3*NUM_VEL*f + 3*vi + row], tangent [v_lin; omega; joints]
//   grid_plant::multi_target_position[_gradient]  -> the caller-scratch wrappers (must equal the device fns)
// Reads q (NUM_POS doubles, one per line) from ./q.txt so the Python test controls the pose
// (a floating base needs a UNIT quaternion). Prints P/J rows, plus THREADINV / SPILLDIFF /
// PLANTDIFF self-checks. Correctness only, no timing.
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using T = double;
constexpr int NQ = grid::NUM_POS;
constexpr int NV = grid::NUM_VEL;
constexpr int NF = grid::NUM_MULTI_TARGETS;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

__global__ void pos_kernel(T *d_pos, const T *d_q, const grid::robotModel<T> *m) {
    __shared__ T s_pos[3*NF];
    grid::multi_target_position_device<T>(s_pos, d_q, m);
    __syncthreads();
    if (threadIdx.x == 0) for (int i = 0; i < 3*NF; ++i) d_pos[i] = s_pos[i];
}
__global__ void pos_kernel_spill(T *d_pos, const T *d_q, const grid::robotModel<T> *m, T *d_ws) {
    __shared__ T s_pos[3*NF];
    grid::multi_target_position_device<T, grid::TIER_MINIMAL>(s_pos, d_q, m, d_ws);
    __syncthreads();
    if (threadIdx.x == 0) for (int i = 0; i < 3*NF; ++i) d_pos[i] = s_pos[i];
}
__global__ void grad_kernel(T *d_grad, const T *d_q, const grid::robotModel<T> *m) {
    __shared__ T s_grad[3*NV*NF];
    grid::multi_target_position_gradient_device<T>(s_grad, d_q, m);
    __syncthreads();
    if (threadIdx.x == 0) for (int i = 0; i < 3*NV*NF; ++i) d_grad[i] = s_grad[i];
}
__global__ void plant_kernel(T *d_pos, T *d_grad, const T *d_q, const grid::robotModel<T> *m) {
    __shared__ T s_pos[3*NF];
    __shared__ T s_grad[3*NV*NF];
    __shared__ __align__(16) T s_scratch[(grid::MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() / sizeof(T))];
    grid_plant::multi_target_position_gradient<T>(s_pos, s_grad, d_q, s_scratch, m);
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < 3*NF; ++i) d_pos[i] = s_pos[i];
        for (int i = 0; i < 3*NV*NF; ++i) d_grad[i] = s_grad[i];
    }
}
__global__ void plant_pos_kernel(T *d_pos, const T *d_q, const grid::robotModel<T> *m) {
    __shared__ T s_pos[3*NF];
    __shared__ __align__(16) T s_scratch[(grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>() / sizeof(T))];
    grid_plant::multi_target_position<T>(s_pos, d_q, s_scratch, m);
    __syncthreads();
    if (threadIdx.x == 0) for (int i = 0; i < 3*NF; ++i) d_pos[i] = s_pos[i];
}

int main(){
    std::vector<T> hq(NQ);
    { FILE *f = fopen("q.txt", "r"); if (!f) { printf("no q.txt\n"); return 3; }
      for (int i = 0; i < NQ; ++i) if (fscanf(f, "%lf", &hq[i]) != 1) { printf("short q.txt\n"); return 3; }
      fclose(f); }
    const grid::robotModel<T> *d_m = grid::init_robotModel<T>();
    size_t smem_p = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t smem_g = grid::MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    T *d_q,*d_pos,*d_grad,*d_pos2,*d_grad2;
    CK(cudaMalloc(&d_q,NQ*sizeof(T))); CK(cudaMalloc(&d_pos,3*NF*sizeof(T))); CK(cudaMalloc(&d_pos2,3*NF*sizeof(T)));
    CK(cudaMalloc(&d_grad,3*NV*NF*sizeof(T))); CK(cudaMalloc(&d_grad2,3*NV*NF*sizeof(T)));
    CK(cudaMemcpy(d_q,hq.data(),NQ*sizeof(T),cudaMemcpyHostToDevice));
    cudaFuncSetAttribute(pos_kernel,  cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_p);
    cudaFuncSetAttribute(grad_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_g);

    // thread invariance (positions + Jacobian) at 1 / 32 / 256 threads
    const int tc[3] = {1, 32, 256};
    std::vector<std::vector<T>> rp(3, std::vector<T>(3*NF)), rg(3, std::vector<T>(3*NV*NF));
    for (int k=0;k<3;++k){
        pos_kernel<<<1,tc[k],smem_p>>>(d_pos,d_q,d_m); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(rp[k].data(),d_pos,3*NF*sizeof(T),cudaMemcpyDeviceToHost));
        grad_kernel<<<1,tc[k],smem_g>>>(d_grad,d_q,d_m); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(rg[k].data(),d_grad,3*NV*NF*sizeof(T),cudaMemcpyDeviceToHost));
    }
    double tinv=0;
    for(int k=1;k<3;++k){ for(int i=0;i<3*NF;++i) tinv=std::max(tinv,fabs(rp[k][i]-rp[0][i]));
                          for(int i=0;i<3*NV*NF;++i) tinv=std::max(tinv,fabs(rg[k][i]-rg[0][i])); }
    printf("THREADINV maxdiff=%.3e\n", tinv);

    // forced spill (TIER_MINIMAL scratch -> d_workspace) must be bit-identical
    size_t smem_spill = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T, grid::TIER_MINIMAL>();
    size_t ws_bytes   = grid::MULTI_TARGET_POSITION_DEVICE_INLINE_WORKSPACE_BYTES<T, grid::TIER_MINIMAL>();
    T *d_ws=nullptr; if (ws_bytes) CK(cudaMalloc(&d_ws, ws_bytes));
    cudaFuncSetAttribute(pos_kernel_spill, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_spill);
    std::vector<T> rs(3*NF);
    pos_kernel_spill<<<1,256,smem_spill>>>(d_pos,d_q,d_m,d_ws); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(rs.data(),d_pos,3*NF*sizeof(T),cudaMemcpyDeviceToHost));
    double sd=0; for(int i=0;i<3*NF;++i) sd=std::max(sd,fabs(rs[i]-rp[2][i]));
    printf("SPILLDIFF maxdiff=%.3e (ws %zu B)\n", sd, ws_bytes);

    // grid_plant caller-scratch wrappers must equal the device fns
    std::vector<T> pp(3*NF), pg(3*NV*NF), pp2(3*NF);
    plant_kernel<<<1,256>>>(d_pos2,d_grad2,d_q,d_m); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(pp.data(),d_pos2,3*NF*sizeof(T),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(pg.data(),d_grad2,3*NV*NF*sizeof(T),cudaMemcpyDeviceToHost));
    plant_pos_kernel<<<1,256>>>(d_pos2,d_q,d_m); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(pp2.data(),d_pos2,3*NF*sizeof(T),cudaMemcpyDeviceToHost));
    double pd=0; for(int i=0;i<3*NF;++i){ pd=std::max(pd,fabs(pp[i]-rp[2][i])); pd=std::max(pd,fabs(pp2[i]-rp[2][i])); }
    for(int i=0;i<3*NV*NF;++i) pd=std::max(pd,fabs(pg[i]-rg[2][i]));
    printf("PLANTDIFF maxdiff=%.3e\n", pd);

    for(int f=0;f<NF;++f) printf("P %d % .17g % .17g % .17g\n", f, rp[2][3*f+0], rp[2][3*f+1], rp[2][3*f+2]);
    for(int f=0;f<NF;++f) for(int vi=0;vi<NV;++vi)
        printf("J %d %d % .17g % .17g % .17g\n", f, vi, rg[2][3*NV*f+3*vi+0], rg[2][3*NV*f+3*vi+1], rg[2][3*NV*f+3*vi+2]);
    return (tinv == 0.0 && sd == 0.0 && pd == 0.0) ? 0 : 1;
}

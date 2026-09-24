// Racecheck gate for the FLOATING-base composition (GATO nit 3, 2026-09-24): GATO saw warp-level WAW
// warnings in its merit kernel (loaders + integrator_inner); GRiD's own composition must stay at 0 hazards.
// Kernels: fd device wrapper, ee_pose device wrapper (XmatsHom), integrator device wrapper. argv[1] = threads.
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <vector>
#include <cstdlib>
using T = float;
constexpr int NQ = grid::NUM_POS; constexpr int NV = grid::NUM_VEL;
#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)
__global__ void fd_k(T *out, const T *q, const T *qd, const T *u, const grid::robotModel<T> *m) {
    __shared__ T s_q[NQ], s_qd[NQ], s_u[NQ], s_qdd[NV];
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) { s_q[i] = q[i]; s_qd[i] = qd[i]; s_u[i] = u[i]; }
    __syncthreads();
    grid::forward_dynamics_device<T>(s_qdd, s_q, s_qd, s_u, m, nullptr, (T)-9.81);
    __syncthreads();
    if (threadIdx.x == 0) for (int i = 0; i < NV; ++i) out[i] = s_qdd[i];
}
__global__ void ee_k(T *out, const T *q, const grid::robotModel<T> *m) {
    __shared__ T s_q[NQ], s_ee[6 * grid::NUM_EES];
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) s_q[i] = q[i];
    __syncthreads();
    grid::end_effector_pose_device<T>(s_ee, s_q, m);
    __syncthreads();
    if (threadIdx.x == 0) for (int i = 0; i < 6 * grid::NUM_EES; ++i) out[i] = s_ee[i];
}
__global__ void integ_k(T *out, const T *q, const T *qd, const T *u, const grid::robotModel<T> *m) {
    __shared__ T s_q[NQ], s_qd[NQ], s_u[NQ], s_x[NQ + NV];
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) { s_q[i] = q[i]; s_qd[i] = qd[i]; s_u[i] = u[i]; }
    __syncthreads();
    grid::integrator_device<T>(s_x, s_q, s_qd, s_u, m, nullptr, (T)-9.81, (T)0.01);
    __syncthreads();
    if (threadIdx.x == 0) for (int i = 0; i < NQ + NV; ++i) out[i] = s_x[i];
}
int main(int argc, char **argv) { int NT = argc > 1 ? atoi(argv[1]) : 256; printf("THREADS %d\n", NT);
    std::vector<T> hq(NQ, 0.1f), hqd(NQ, 0.2f), hu(NQ, 0.3f);
    hq[3] = 0.1f; hq[4] = 0.2f; hq[5] = 0.3f; hq[6] = 0.927f; hqd[NV] = 0; hu[NV] = 0;  // unit-ish quaternion xyzw, pads
    const grid::robotModel<T> *m = grid::init_robotModel<T>();
    T *dq, *dqd, *du, *dout; CK(cudaMalloc(&dq, NQ * 4)); CK(cudaMalloc(&dqd, NQ * 4)); CK(cudaMalloc(&du, NQ * 4)); CK(cudaMalloc(&dout, 4096));
    CK(cudaMemcpy(dq, hq.data(), NQ * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dqd, hqd.data(), NQ * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(du, hu.data(), NQ * 4, cudaMemcpyHostToDevice));
    size_t s_fd = grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), s_ee = grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>(), s_in = grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>();
    CK(cudaFuncSetAttribute(fd_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)s_fd));
    CK(cudaFuncSetAttribute(ee_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)s_ee));
    CK(cudaFuncSetAttribute(integ_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)s_in));
    printf("KERNEL fd\n"); fd_k<<<1, NT, s_fd>>>(dout, dq, dqd, du, m); CK(cudaDeviceSynchronize());
    printf("KERNEL ee\n"); ee_k<<<1, NT, s_ee>>>(dout, dq, m); CK(cudaDeviceSynchronize());
    printf("KERNEL integ\n"); integ_k<<<1, NT, s_in>>>(dout, dq, dqd, du, m); CK(cudaDeviceSynchronize());
    printf("OK\n"); return 0;
}

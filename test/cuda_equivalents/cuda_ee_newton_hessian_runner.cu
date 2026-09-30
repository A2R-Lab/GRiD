// Permanent regression gate for the full-Newton EE-position cost Hessian
// (grid_plant::ee_pos_cost_hessian, PR #19). Self-checking, no external oracle:
// the true Hessian is verified against a central finite-difference of the
// analytic gradient (grid_plant::ee_pos_cost_gradient), and the Gauss-Newton
// variant is checked to differ from it by EXACTLY the residual-weighted
// curvature term it drops.
//
//   1. hess<GAUSS_NEWTON=false>  ==  FD(ee_pos_cost_gradient)     (true Hessian)
//   2. hess<GAUSS_NEWTON=true>   !=  FD  (must differ ~ curvature)
//   3. |gn_vs_fd - |Hn - Hgn||  ~  0   (the difference IS the curvature)
//
// Fixed-base only (NUM_POS == NUM_VEL), so a q-perturbation FD is well defined
// (no quaternion tangent handling). fp64 throughout to expose the curvature well
// above float32 cancellation. Prints "PASS"/"FAIL" and the three metrics; exit
// code 0 on PASS. Robot-general via the grid:: dimension constants.
#include <cstdio>
#include <cmath>
#include <vector>

#define GRID_HEADER
#include "grid.cuh"   // self-contained: vendors barrier.cuh + all glass ops into grid::glass

using T = double;
constexpr int NQ = grid::NUM_POS;
constexpr int NV = grid::NUM_VEL;
constexpr int NEE = grid::NUM_EES;   // the inner fills pose/grad/hess for ALL end-effectors
constexpr int NX = NQ + NV;
constexpr int EE = 0;

// The ee_pos_cost_{gradient,hessian} internal arena (s_scratch) is robot-sized —
// it scales with the joint count, so it goes in DYNAMIC shared memory, sized at
// launch from the generated grid:: macros and opted-in via cudaFuncSetAttribute
// (mirrors cuda_plant_smoke_runner.cu). The small fixed I/O buffers stay static.
extern __shared__ __align__(16) T s_dyn[];

__global__ void grad_kernel(T *d_grad, const T *d_q, const T *d_pdes, const T *d_W,
                            const grid::robotModel<T> *d_rm) {
    __shared__ T s_q[NQ], s_pdes[3], s_W[3], s_pose[6 * NEE], s_grad_out[NX];
    __shared__ T s_dee[6 * NV * NEE];
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) s_q[i] = d_q[i];
    for (int i = threadIdx.x; i < 3;  i += blockDim.x) { s_pdes[i] = d_pdes[i]; s_W[i] = d_W[i]; }
    __syncthreads();
    grid_plant::ee_pos_cost_gradient<T, EE, false>(s_grad_out, s_q, s_pdes, s_W, s_pose, s_dee, s_dyn, d_rm);
    __syncthreads();
    for (int i = threadIdx.x; i < NX; i += blockDim.x) d_grad[i] = s_grad_out[i];
}

template <bool GN>
__global__ void hess_kernel(T *d_hess, const T *d_q, const T *d_pdes, const T *d_W,
                            const grid::robotModel<T> *d_rm) {
    __shared__ T s_q[NQ], s_pdes[3], s_W[3], s_pose[6 * NEE];
    __shared__ T s_dee[6 * NV * NEE], s_d2ee[6 * NV * NV * NEE], s_hess[NX * NX];
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) s_q[i] = d_q[i];
    for (int i = threadIdx.x; i < 3;  i += blockDim.x) { s_pdes[i] = d_pdes[i]; s_W[i] = d_W[i]; }
    __syncthreads();
    grid_plant::ee_pos_cost_hessian<T, EE, false, GN>(s_hess, s_q, s_pdes, s_W,
                                                      s_pose, s_dee, s_d2ee, s_dyn, d_rm);
    __syncthreads();
    for (int i = threadIdx.x; i < NX * NX; i += blockDim.x) d_hess[i] = s_hess[i];
}

static size_t g_grad_dyn = 0, g_hess_dyn = 0;

static void grad_at(const std::vector<T> &q, T *d_buf, T *d_q, T *d_pdes, T *d_W,
                    const grid::robotModel<T> *d_rm, std::vector<T> &out) {
    cudaMemcpy(d_q, q.data(), NQ * sizeof(T), cudaMemcpyHostToDevice);
    grad_kernel<<<1, 64, g_grad_dyn>>>(d_buf, d_q, d_pdes, d_W, d_rm);
    cudaMemcpy(out.data(), d_buf, NX * sizeof(T), cudaMemcpyDeviceToHost);
}

int main() {
    static_assert(NQ == NV, "cuda_ee_newton_hessian_runner is fixed-base only (NUM_POS == NUM_VEL)");
    grid::robotModel<T> *d_rm = grid::init_robotModel<T>();

    // Size dynamic smem to each kernel's robot-dependent arena; the Newton hess
    // path needs the hessian arena, the grad kernel the gradient arena. Raise the
    // opt-in cap so big-joint-count robots fit.
    g_grad_dyn = grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    g_hess_dyn = grid::END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grad_kernel,         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g_grad_dyn);
    cudaFuncSetAttribute(hess_kernel<false>,  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g_hess_dyn);
    cudaFuncSetAttribute(hess_kernel<true>,   cudaFuncAttributeMaxDynamicSharedMemorySize, (int)g_hess_dyn);

    // Deterministic mid-range configuration + a FAR target so the residual (and
    // thus the curvature term) is large and unambiguous.
    std::vector<T> q(NQ);
    for (int i = 0; i < NQ; ++i) q[i] = 0.3 * ((i % 2) ? -1.0 : 1.0) * (1.0 + 0.15 * i);
    T pdes[3] = {0.9, -0.6, 1.2};
    T W[3]    = {2.0, 1.5, 3.0};

    T *d_q, *d_pdes, *d_W, *d_grad, *d_hess_n, *d_hess_gn;
    cudaMalloc(&d_q, NQ * sizeof(T)); cudaMalloc(&d_pdes, 3 * sizeof(T)); cudaMalloc(&d_W, 3 * sizeof(T));
    cudaMalloc(&d_grad, NX * sizeof(T));
    cudaMalloc(&d_hess_n, NX * NX * sizeof(T)); cudaMalloc(&d_hess_gn, NX * NX * sizeof(T));
    cudaMemcpy(d_pdes, pdes, 3 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, 3 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), NQ * sizeof(T), cudaMemcpyHostToDevice);

    hess_kernel<false><<<1, 64, g_hess_dyn>>>(d_hess_n, d_q, d_pdes, d_W, d_rm);   // Newton (default)
    hess_kernel<true ><<<1, 64, g_hess_dyn>>>(d_hess_gn, d_q, d_pdes, d_W, d_rm);  // Gauss-Newton
    std::vector<T> Hn(NX * NX), Hgn(NX * NX);
    cudaMemcpy(Hn.data(),  d_hess_n,  NX * NX * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(Hgn.data(), d_hess_gn, NX * NX * sizeof(T), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 2; }

    // FD Hessian from the analytic gradient (central differences over the q dims).
    const T eps = 1e-6;
    std::vector<T> Hfd(NX * NX, 0.0), gp(NX), gm(NX);
    for (int j = 0; j < NQ; ++j) {
        std::vector<T> qp = q, qm = q; qp[j] += eps; qm[j] -= eps;
        grad_at(qp, d_grad, d_q, d_pdes, d_W, d_rm, gp);
        grad_at(qm, d_grad, d_q, d_pdes, d_W, d_rm, gm);
        for (int i = 0; i < NX; ++i) Hfd[j * NX + i] = (gp[i] - gm[i]) / (2 * eps);  // col-major col j
    }

    auto max_abs_diff = [&](const std::vector<T> &A, const std::vector<T> &B) {
        T m = 0; for (int i = 0; i < NX * NX; ++i) m = fmax(m, fabs(A[i] - B[i])); return m;
    };
    T newton_vs_fd = max_abs_diff(Hn, Hfd);
    T gn_vs_fd     = max_abs_diff(Hgn, Hfd);
    T curv_mag = 0;  // ||Hn - Hgn||_max = the curvature term magnitude (must be >> 0 here)
    for (int i = 0; i < NX * NX; ++i) curv_mag = fmax(curv_mag, fabs(Hn[i] - Hgn[i]));

    printf("newton_vs_fd  max|diff| = %.3e   (PASS if < 1e-5)\n", newton_vs_fd);
    printf("gn_vs_fd      max|diff| = %.3e   (must be LARGE ~ curvature)\n", gn_vs_fd);
    printf("curvature magnitude     = %.3e   (must be > 1e-2 at this residual)\n", curv_mag);
    bool pass = (newton_vs_fd < 1e-5) && (curv_mag > 1e-2) && (fabs(gn_vs_fd - curv_mag) < 1e-5);
    printf(pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}

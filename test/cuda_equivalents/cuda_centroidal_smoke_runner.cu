// CUDA smoke runner for the generated centroidal / energy device kernels and
// the grid_plant CoM / centroidal-momentum costs (D1b + D1c).
//
// Input on stdin (whitespace-separated floats):
//   q (NUM_POS) qd (NUM_VEL)
//
// The runner drives each centroidal device function from a single-block kernel
// and prints the results in BEGIN/END framed blocks (column-major), to be
// cross-checked against the RBDReference numpy oracles (which match Pinocchio
// to ~1e-14).
//
// Emitted blocks (D1b: the 5 centroidal device kernels):
//   gen_gravity     1 x NUM_VEL                 generalized_gravity_device(g)
//   nonlinear       1 x NUM_VEL                 nonlinear_effects_device(c)
//   com             1 x 3                       com_device CoM position
//   jcom            3 x NUM_VEL                 com_device CoM Jacobian (col-major)
//   ccrba_A         6 x NUM_VEL                 ccrba_device CMM A (col-major)
//   ccrba_h         1 x 6                       ccrba_device momentum h
//   energy          1 x 3                       energy_device [KE, PE, mechanical]
//
// Emitted blocks (D1c: the grid_plant CoM / momentum costs):
//   com_cost_value  1 x 1
//   com_cost_grad   1 x NX                      [J_com^T W r ; 0]   (NX = NQ+NV)
//   com_cost_hess   NX x NX                     J_com^T W J_com (q-block only)
//   mom_cost_value  1 x 1
//   mom_cost_grad   1 x NX                      [0 ; A^T W r]
//   mom_cost_hess   NX x NX                     A^T W A (qd-block only)
//
// The deterministic cost setup (p_des / h_des / weights) is mirrored exactly in
// the Python test.
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "grid.cuh"

template <typename T>
void read_vector(T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        double value;
        if (!(std::cin >> value)) { std::cerr << "read fail " << i << "\n"; std::exit(2); }
        dst[i] = static_cast<T>(value);
    }
}

template <typename T>
void print_matrix_col_major(const std::string &name, const T *data, int rows, int cols) {
    std::cout << "BEGIN " << name << " " << rows << " " << cols << "\n";
    std::cout << std::setprecision(10);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (col) std::cout << " ";
            std::cout << static_cast<double>(data[row + rows * col]);
        }
        std::cout << "\n";
    }
    std::cout << "END " << name << "\n";
}
template <typename T>
void print_vector(const std::string &name, const T *data, int count) {
    print_matrix_col_major(name, data, 1, count);
}

constexpr int NQ = grid::NUM_POS;
constexpr int NV = grid::NUM_VEL;
constexpr int NX = NQ + NV;

// ---- deterministic cost setup (MUST match the Python test exactly) ----
template <typename T> __host__ __device__ T comW_val(int r)  { return static_cast<T>(5.0) + r; }
template <typename T> __host__ __device__ T momW_val(int r)  { return static_cast<T>(2.0) + static_cast<T>(0.5) * r; }
// p_des / h_des are offset from the realized value by a fixed per-index delta so
// the residual is non-trivial; the Python test applies the same offset to the
// double-precision oracle CoM / momentum.
template <typename T> __host__ __device__ T comDes_off(int r) { return static_cast<T>(0.05) * (r + 1); }
template <typename T> __host__ __device__ T momDes_off(int r) { return static_cast<T>(0.1)  * (r + 1); }

// ----- D1b: drive the 5 centroidal device functions -----
template <typename T>
__global__ void centroidal_kernel(const T *g_q, const T *g_qd,
                                  const grid::robotModel<T> *d_robotModel, T gravity,
                                  T *o_grav, T *o_nle, T *o_com, T *o_jcom,
                                  T *o_A, T *o_h, T *o_energy) {
    __shared__ T s_q[NQ], s_qd[NV];
    __shared__ T s_grav[NV], s_nle[NV];
    __shared__ T s_com[3 + 3 * NV];
    __shared__ T s_ccrba[6 * NV + 6];
    __shared__ T s_energy[3];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    for (int i = tid; i < NV; i += nth) s_qd[i] = g_qd[i];
    __syncthreads();

    grid::generalized_gravity_device<T>(s_grav, s_q, s_qd, d_robotModel, gravity);
    __syncthreads();
    grid::nonlinear_effects_device<T>(s_nle, s_q, s_qd, d_robotModel, gravity);
    __syncthreads();
    grid::com_device<T>(s_com, s_q, d_robotModel);
    __syncthreads();
    grid::ccrba_device<T>(s_ccrba, s_q, s_qd, d_robotModel);
    __syncthreads();
    grid::energy_device<T>(s_energy, s_q, s_qd, d_robotModel, gravity);
    __syncthreads();

    for (int i = tid; i < NV; i += nth) { o_grav[i] = s_grav[i]; o_nle[i] = s_nle[i]; }
    for (int r = tid; r < 3; r += nth) o_com[r] = s_com[r];
    for (int i = tid; i < 3 * NV; i += nth) o_jcom[i] = s_com[3 + i];   // 3 x NV col-major
    for (int i = tid; i < 6 * NV; i += nth) o_A[i] = s_ccrba[i];        // 6 x NV col-major
    for (int r = tid; r < 6; r += nth) o_h[r] = s_ccrba[6 * NV + r];
    for (int r = tid; r < 3; r += nth) o_energy[r] = s_energy[r];
    __syncthreads();
}

// ----- D1c: drive the grid_plant CoM / momentum costs -----
template <typename T>
__global__ void cost_kernel(const T *g_q, const T *g_qd,
                            const grid::robotModel<T> *d_robotModel,
                            T *o_cv, T *o_cg, T *o_ch,
                            T *o_mv, T *o_mg, T *o_mh) {
    __shared__ T s_q[NQ], s_qd[NV];
    __shared__ T s_pdes[3], s_cW[3], s_hdes[6], s_mW[6];
    __shared__ T s_out[1], s_grad[NX], s_hess[NX * NX];
    __shared__ T s_com[3 + 3 * NV];
    __shared__ T s_ccrba[6 * NV + 6];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    for (int i = tid; i < NV; i += nth) s_qd[i] = g_qd[i];
    for (int r = tid; r < 3; r += nth) s_cW[r] = comW_val<T>(r);
    for (int r = tid; r < 6; r += nth) s_mW[r] = momW_val<T>(r);
    __syncthreads();

    // Realize the CoM / momentum so p_des / h_des are (realized + fixed offset).
    grid::com_device<T>(s_com, s_q, d_robotModel);
    grid::ccrba_device<T>(s_ccrba, s_q, s_qd, d_robotModel);
    __syncthreads();
    if (tid == 0) {
        for (int r = 0; r < 3; ++r) s_pdes[r] = s_com[r] + comDes_off<T>(r);
        for (int r = 0; r < 6; ++r) s_hdes[r] = s_ccrba[6 * NV + r] + momDes_off<T>(r);
    }
    __syncthreads();

    // ---- com cost ----
    grid_plant::com_cost<T>(s_out, s_q, s_pdes, s_cW, s_com, d_robotModel);
    __syncthreads(); if (tid == 0) o_cv[0] = s_out[0]; __syncthreads();
    // Zero s_grad first: com_cost_gradient writes the tangent block [0,NV) and
    // the qd block [NQ,NX), but for a floating base the surplus position slots
    // [NV,NQ) are not written, so initialize them to zero deterministically.
    for (int i = tid; i < NX; i += nth) s_grad[i] = static_cast<T>(0);
    __syncthreads();
    grid_plant::com_cost_gradient<T, false>(s_grad, s_q, s_pdes, s_cW, s_com, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_cg[i] = s_grad[i];
    __syncthreads();
    grid_plant::com_cost_hessian<T, false>(s_hess, s_q, s_cW, s_com, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX * NX; i += nth) o_ch[i] = s_hess[i];
    __syncthreads();

    // ---- momentum cost ----
    grid_plant::momentum_cost<T>(s_out, s_q, s_qd, s_hdes, s_mW, s_ccrba, d_robotModel);
    __syncthreads(); if (tid == 0) o_mv[0] = s_out[0]; __syncthreads();
    for (int i = tid; i < NX; i += nth) s_grad[i] = static_cast<T>(0);
    __syncthreads();
    grid_plant::momentum_cost_gradient<T, false>(s_grad, s_q, s_qd, s_hdes, s_mW, s_ccrba, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_mg[i] = s_grad[i];
    __syncthreads();
    grid_plant::momentum_cost_hessian<T, false>(s_hess, s_q, s_qd, s_mW, s_ccrba, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX * NX; i += nth) o_mh[i] = s_hess[i];
    __syncthreads();
}

template <typename T>
T *dmalloc(int count) { T *p; cudaMalloc(&p, count * sizeof(T)); return p; }
template <typename T>
void dcopy_out(const std::string &name, T *dptr, int rows, int cols) {
    std::vector<T> h(rows * cols);
    cudaMemcpy(h.data(), dptr, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);
    print_matrix_col_major(name, h.data(), rows, cols);
}

template <typename T>
void run() {
    const T gravity = static_cast<T>(9.81);
    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    std::vector<T> h_q(NQ), h_qd(NV);
    read_vector(h_q.data(), NQ);
    read_vector(h_qd.data(), NV);

    print_vector("input_q", h_q.data(), NQ);
    print_vector("input_qd", h_qd.data(), NV);

    T *g_q = dmalloc<T>(NQ), *g_qd = dmalloc<T>(NV);
    cudaMemcpy(g_q, h_q.data(), NQ * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_qd, h_qd.data(), NV * sizeof(T), cudaMemcpyHostToDevice);

    T *o_grav = dmalloc<T>(NV), *o_nle = dmalloc<T>(NV);
    T *o_com = dmalloc<T>(3), *o_jcom = dmalloc<T>(3 * NV);
    T *o_A = dmalloc<T>(6 * NV), *o_h = dmalloc<T>(6), *o_energy = dmalloc<T>(3);
    T *o_cv = dmalloc<T>(1), *o_cg = dmalloc<T>(NX), *o_ch = dmalloc<T>(NX * NX);
    T *o_mv = dmalloc<T>(1), *o_mg = dmalloc<T>(NX), *o_mh = dmalloc<T>(NX * NX);

    const int nthreads = grid::MAX_PERF_LEVEL_THREADS;
    // The centroidal device functions compose the grid:: XmatsHom kinematics
    // + EE-Jacobian arena via auto-allocating wrappers (extern __shared__); size
    // each kernel's dynamic smem to the max device requirement it composes.
    // generalized_gravity / nonlinear_effects share the ID_BIAS arena; com /
    // ccrba / energy each have their own. Size to the max over all five.
    size_t dyn = grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>();
    if (grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>() > dyn) dyn = grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>();
    if (grid::ENERGY_DYNAMIC_SHARED_MEM_BYTES<T>() > dyn) dyn = grid::ENERGY_DYNAMIC_SHARED_MEM_BYTES<T>();
    if (grid::ID_BIAS_DYNAMIC_SHARED_MEM_BYTES<T>() > dyn) dyn = grid::ID_BIAS_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(centroidal_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
    cudaFuncSetAttribute(cost_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);

    centroidal_kernel<T><<<1, nthreads, dyn>>>(g_q, g_qd, d_robotModel, gravity,
        o_grav, o_nle, o_com, o_jcom, o_A, o_h, o_energy);
    cudaDeviceSynchronize();

    cost_kernel<T><<<1, nthreads, dyn>>>(g_q, g_qd, d_robotModel,
        o_cv, o_cg, o_ch, o_mv, o_mg, o_mh);
    cudaDeviceSynchronize();

    // D1b blocks
    dcopy_out("gen_gravity", o_grav, 1, NV);
    dcopy_out("nonlinear", o_nle, 1, NV);
    dcopy_out("com", o_com, 1, 3);
    dcopy_out("jcom", o_jcom, 3, NV);
    dcopy_out("ccrba_A", o_A, 6, NV);
    dcopy_out("ccrba_h", o_h, 1, 6);
    dcopy_out("energy", o_energy, 1, 3);
    // D1c blocks
    dcopy_out("com_cost_value", o_cv, 1, 1);
    dcopy_out("com_cost_grad", o_cg, 1, NX);
    dcopy_out("com_cost_hess", o_ch, NX, NX);
    dcopy_out("mom_cost_value", o_mv, 1, 1);
    dcopy_out("mom_cost_grad", o_mg, 1, NX);
    dcopy_out("mom_cost_hess", o_mh, NX, NX);

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main() {
    run<float>();
    return 0;
}

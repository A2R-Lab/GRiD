// Mimic-SAFE CUDA smoke runner for the centroidal device kernels.
//
// This is a MINIMAL companion to cuda_centroidal_smoke_runner.cu. It drives the
// mimic-supported centroidal DEVICE fns:
//   grid::generalized_gravity_device  (g(q)   = RNEA(q, 0, 0))
//   grid::nonlinear_effects_device    (c(q,qd)= RNEA(q, qd, 0))
//   grid::com_device                  (p_com (3) + J_com (3 x NV))
//   grid::ccrba_device                (A (6 x NV) + h (6))
//   grid::energy_device               ({KE, PE, mechanical})
//
// The com/ccrba/energy kinematics-domain device fns are now ALPHA-FOLDED for
// mimic robots (centroidal_inner's per-body world Jacobian carries the mimic
// multiplier), so they emit + validate for mimic robots. The FULL centroidal
// runner additionally drives the grid_plant com_cost / momentum_cost costs,
// which are emitted for NON-MIMIC robots only (_plant.py) — that is what keeps
// the full runner structurally non-mimic-only. This runner only references the
// centroidal device fns, so it links + runs on a mimic robot (and on the
// non-mimic control). Each kinematics device fn declares its OWN extern-shared
// arena, so we launch one kernel per fn (sized to that fn's *_DYNAMIC macro).
//
// Input on stdin (whitespace-separated floats):
//   q (NUM_POS) qd (NUM_VEL)
//
// Emitted blocks (column-major, BEGIN/END framed):
//   gen_gravity   1 x NUM_VEL   generalized_gravity_device(g)
//   nonlinear     1 x NUM_VEL   nonlinear_effects_device(c)
//   com           1 x 3         com_device p_com
//   jcom          3 x NV        com_device J_com (column-major)
//   ccrba_A       6 x NV        ccrba_device A (column-major)
//   ccrba_h       1 x 6         ccrba_device h
//   energy        1 x 3         energy_device {KE, PE, mechanical}
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

// Drive ONLY the two mimic-supported id-bias device functions.
template <typename T>
__global__ void centroidal_bias_kernel(const T *g_q, const T *g_qd,
                                       const grid::robotModel<T> *d_robotModel, T gravity,
                                       T *o_grav, T *o_nle) {
    __shared__ T s_q[NQ], s_qd[NV];
    __shared__ T s_grav[NV], s_nle[NV];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    for (int i = tid; i < NV; i += nth) s_qd[i] = g_qd[i];
    __syncthreads();

    grid::generalized_gravity_device<T>(s_grav, s_q, s_qd, d_robotModel, gravity);
    __syncthreads();
    grid::nonlinear_effects_device<T>(s_nle, s_q, s_qd, d_robotModel, gravity);
    __syncthreads();

    for (int i = tid; i < NV; i += nth) { o_grav[i] = s_grav[i]; o_nle[i] = s_nle[i]; }
    __syncthreads();
}

// com_device: s_out = [p_com (3); J_com (3 x NV col-major)] = 3 + 3*NV.
template <typename T>
__global__ void com_kernel(const T *g_q, const grid::robotModel<T> *d_robotModel, T *o_com) {
    __shared__ T s_q[NQ];
    __shared__ T s_out[3 + 3 * NV];
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    __syncthreads();
    grid::com_device<T>(s_out, s_q, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 3 + 3 * NV; i += nth) o_com[i] = s_out[i];
    __syncthreads();
}

// ccrba_device: s_out = [A (6 x NV col-major); h (6)] = 6*NV + 6.
template <typename T>
__global__ void ccrba_kernel(const T *g_q, const T *g_qd,
                             const grid::robotModel<T> *d_robotModel, T *o_ccrba) {
    __shared__ T s_q[NQ], s_qd[NV];
    __shared__ T s_out[6 * NV + 6];
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    for (int i = tid; i < NV; i += nth) s_qd[i] = g_qd[i];
    __syncthreads();
    grid::ccrba_device<T>(s_out, s_q, s_qd, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 6 * NV + 6; i += nth) o_ccrba[i] = s_out[i];
    __syncthreads();
}

// energy_device: s_out = [KE, PE, mechanical] = 3.
template <typename T>
__global__ void energy_kernel(const T *g_q, const T *g_qd,
                              const grid::robotModel<T> *d_robotModel, T gravity, T *o_energy) {
    __shared__ T s_q[NQ], s_qd[NV];
    __shared__ T s_out[3];
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    for (int i = tid; i < NV; i += nth) s_qd[i] = g_qd[i];
    __syncthreads();
    grid::energy_device<T>(s_out, s_q, s_qd, d_robotModel, gravity);
    __syncthreads();
    for (int i = tid; i < 3; i += nth) o_energy[i] = s_out[i];
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
    const T gravity = static_cast<T>(-9.81);
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

    const int nthreads = grid::MAX_PERF_LEVEL_THREADS;
    // generalized_gravity / nonlinear_effects compose the RNEA id-bias arena via
    // the auto-allocating extern __shared__ wrapper; size to the ID_BIAS macro.
    size_t dyn = grid::INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(centroidal_bias_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);

    centroidal_bias_kernel<T><<<1, nthreads, dyn>>>(g_q, g_qd, d_robotModel, gravity, o_grav, o_nle);
    // Fail loudly on a bad launch: an unchecked launch failure leaves the (zeroed)
    // outputs untouched, which then masquerades as a real (wrong) result the Python
    // oracle would silently diff against. gpuErrchkKernel() (from grid.cuh) does
    // cudaPeekAtLastError() + cudaDeviceSynchronize() and aborts on any error.
    gpuErrchkKernel();

    dcopy_out("gen_gravity", o_grav, 1, NV);
    dcopy_out("nonlinear", o_nle, 1, NV);

    // ---- com / ccrba / energy (alpha-folded for mimic) ----
    // Each kinematics device fn declares its OWN extern-shared arena, so launch
    // one kernel per fn, sized to that fn's *_DYNAMIC_SHARED_MEM_BYTES macro.
    T *o_com = dmalloc<T>(3 + 3 * NV);
    T *o_ccrba = dmalloc<T>(6 * NV + 6);
    T *o_energy = dmalloc<T>(3);

    size_t dyn_com = grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(com_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_com);
    com_kernel<T><<<1, nthreads, dyn_com>>>(g_q, d_robotModel, o_com);
    gpuErrchkKernel();

    size_t dyn_ccrba = grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(ccrba_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_ccrba);
    ccrba_kernel<T><<<1, nthreads, dyn_ccrba>>>(g_q, g_qd, d_robotModel, o_ccrba);
    gpuErrchkKernel();

    size_t dyn_energy = grid::ENERGY_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(energy_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_energy);
    energy_kernel<T><<<1, nthreads, dyn_energy>>>(g_q, g_qd, d_robotModel, gravity, o_energy);
    gpuErrchkKernel();

    // p_com (first 3) + J_com (next 3*NV, col-major 3 x NV)
    {
        std::vector<T> h(3 + 3 * NV);
        cudaMemcpy(h.data(), o_com, (3 + 3 * NV) * sizeof(T), cudaMemcpyDeviceToHost);
        print_matrix_col_major("com", h.data(), 1, 3);
        print_matrix_col_major("jcom", h.data() + 3, 3, NV);
    }
    // A (6 x NV col-major) + h (6)
    {
        std::vector<T> h(6 * NV + 6);
        cudaMemcpy(h.data(), o_ccrba, (6 * NV + 6) * sizeof(T), cudaMemcpyDeviceToHost);
        print_matrix_col_major("ccrba_A", h.data(), 6, NV);
        print_matrix_col_major("ccrba_h", h.data() + 6 * NV, 1, 6);
    }
    dcopy_out("energy", o_energy, 1, 3);

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main() {
    run<float>();
    return 0;
}

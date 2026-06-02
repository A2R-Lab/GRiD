// Mimic-SAFE CUDA smoke runner for the centroidal id-bias device kernels.
//
// This is a deliberately MINIMAL companion to cuda_centroidal_smoke_runner.cu.
// It drives ONLY the two mimic-supported centroidal primitives:
//   grid::generalized_gravity_device  (g(q)   = RNEA(q, 0, 0))
//   grid::nonlinear_effects_device    (c(q,qd)= RNEA(q, qd, 0))
//
// The full centroidal runner additionally drives com / ccrba / energy device
// fns and the grid_plant com_cost / momentum_cost costs, ALL of which codegen
// emits for NON-MIMIC robots only (_centroidal.py kin path + _plant.py). That
// makes the full runner structurally non-mimic-only: a mimic robot's header
// simply does not define those symbols, so it cannot link. This runner only
// references generalized_gravity / nonlinear_effects, which ARE emitted for
// mimic robots (RNEA bias wrappers, gated on `id`), so it links + runs on a
// mimic robot. That gives the missing RUNTIME check of the s_vaf=18*NB
// NB-sizing fix in _centroidal.py for fixed-base mimic robots.
//
// Input on stdin (whitespace-separated floats):
//   q (NUM_POS) qd (NUM_VEL)
//
// Emitted blocks (column-major, BEGIN/END framed):
//   gen_gravity   1 x NUM_VEL   generalized_gravity_device(g)
//   nonlinear     1 x NUM_VEL   nonlinear_effects_device(c)
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

    const int nthreads = grid::MAX_PERF_LEVEL_THREADS;
    // generalized_gravity / nonlinear_effects compose the RNEA id-bias arena via
    // the auto-allocating extern __shared__ wrapper; size to the ID_BIAS macro.
    size_t dyn = grid::ID_BIAS_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(centroidal_bias_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);

    centroidal_bias_kernel<T><<<1, nthreads, dyn>>>(g_q, g_qd, d_robotModel, gravity, o_grav, o_nle);
    // Fail loudly on a bad launch: an unchecked launch failure leaves the (zeroed)
    // outputs untouched, which then masquerades as a real (wrong) result the Python
    // oracle would silently diff against. gpuErrchkKernel() (from grid.cuh) does
    // cudaPeekAtLastError() + cudaDeviceSynchronize() and aborts on any error.
    gpuErrchkKernel();

    dcopy_out("gen_gravity", o_grav, 1, NV);
    dcopy_out("nonlinear", o_nle, 1, NV);

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main() {
    run<float>();
    return 0;
}

// CUDA smoke runner for the generated general-frame geometric Jacobian device
// kernel (E2). Drives grid::frame_jacobian_device from a single-block kernel for
// the three pinocchio reference frames and prints the results in BEGIN/END
// framed blocks (column-major), to be cross-checked against the RBDReference
// numpy oracle (which matches pinocchio's getFrameJacobian/getJointJacobian to
// ~1e-14).
//
// Input on stdin (whitespace-separated):
//   target_jid (int)        joint id of the frame
//   q (NUM_POS floats)
//
// Emitted blocks (each 6 x NUM_VEL, column-major, rows [linear(3); angular(3)]):
//   J_local     reference_frame = 0  (LOCAL)
//   J_world     reference_frame = 1  (WORLD)
//   J_lwa       reference_frame = 2  (LOCAL_WORLD_ALIGNED)
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

constexpr int NQ = grid::NUM_POS;
constexpr int NV = grid::NUM_VEL;

template <typename T>
__global__ void frame_jac_kernel(const T *g_q, const int target_jid,
                                 const grid::robotModel<T> *d_robotModel,
                                 T *o_local, T *o_world, T *o_lwa) {
    __shared__ T s_q[NQ];
    __shared__ T s_J[6 * NV];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    __syncthreads();

    grid::frame_jacobian_device<T>(s_J, target_jid, 0, s_q, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 6 * NV; i += nth) o_local[i] = s_J[i];
    __syncthreads();

    grid::frame_jacobian_device<T>(s_J, target_jid, 1, s_q, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 6 * NV; i += nth) o_world[i] = s_J[i];
    __syncthreads();

    grid::frame_jacobian_device<T>(s_J, target_jid, 2, s_q, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 6 * NV; i += nth) o_lwa[i] = s_J[i];
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
    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    int target_jid;
    if (!(std::cin >> target_jid)) { std::cerr << "read fail target_jid\n"; std::exit(2); }
    std::vector<T> h_q(NQ);
    read_vector(h_q.data(), NQ);

    print_matrix_col_major("input_q", h_q.data(), 1, NQ);

    T *g_q = dmalloc<T>(NQ);
    cudaMemcpy(g_q, h_q.data(), NQ * sizeof(T), cudaMemcpyHostToDevice);

    T *o_local = dmalloc<T>(6 * NV), *o_world = dmalloc<T>(6 * NV), *o_lwa = dmalloc<T>(6 * NV);

    const int nthreads = grid::MAX_PERF_LEVEL_THREADS;
    size_t dyn = grid::FRAME_JACOBIAN_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(frame_jac_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);

    frame_jac_kernel<T><<<1, nthreads, dyn>>>(g_q, target_jid, d_robotModel,
        o_local, o_world, o_lwa);
    cudaDeviceSynchronize();

    dcopy_out("J_local", o_local, 6, NV);
    dcopy_out("J_world", o_world, 6, NV);
    dcopy_out("J_lwa", o_lwa, 6, NV);

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main() {
    run<float>();
    return 0;
}

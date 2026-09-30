// Host-surface CUDA runner for the general-frame Jacobian family (S1).
//
// Drives the BATCHED host launchers (grid::frame_jacobian, and — when emitted —
// grid::frame_jacobian_dot / grid::osc_inertia) end-to-end through the gridData
// output buffers (hd_data->d_frame_jacobian / d_frame_jacobian_dot / d_osc_inertia,
// copied back into the matching h_* buffers). The launchable surface bakes a fixed
// frame target (the leaf-EE joint) and the LOCAL_WORLD_ALIGNED reference frame, so
// the test cross-checks ONLY that (target, frame) pair against the RBDReference
// numpy oracle.
//
// Input on stdin (whitespace-separated):
//   q  (NUM_POS floats)
//   qd (NUM_VEL floats)
//
// Emitted blocks (column-major): FJ (6 x NUM_VEL), FJD (6 x NUM_VEL), LAM (6 x 6).
// LAM is omitted for mimic headers that did not select osc_inertia
// (GRID_FRAME_JAC_MIMIC); the test skips the Lambda check when it is absent.
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

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
void run() {
    const dim3 block_dimms(1, 1, 1);
    const int nthreads = grid::MAX_PERF_LEVEL_THREADS;
    const dim3 thread_dimms(nthreads, 1, 1);

    constexpr int NQ = grid::NUM_POS;
    constexpr int NV = grid::NUM_VEL;

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    // q|qd|qdd into the q_qd_u host buffer (qdd unused but keeps the layout uniform).
    read_vector(hd_data->h_q_qd_u, NQ);
    read_vector(&hd_data->h_q_qd_u[NQ], NV);
    for (int i = 0; i < NV; ++i) hd_data->h_q_qd_u[NQ + NV + i] = static_cast<T>(0);

    // frame_jacobian: batched host surface -> hd_data->h_frame_jacobian (6 x NV).
    grid::frame_jacobian<T>(hd_data, d_robotModel, 1, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    print_matrix_col_major("FJ", hd_data->h_frame_jacobian, 6, NV);

#ifdef GRID_HAS_FRAME_JACOBIAN_DOT
    grid::frame_jacobian_dot<T>(hd_data, d_robotModel, 1, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    print_matrix_col_major("FJD", hd_data->h_frame_jacobian_dot, 6, NV);
#endif

#ifdef GRID_HAS_OSC_INERTIA
    grid::osc_inertia<T>(hd_data, d_robotModel, 1, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    print_matrix_col_major("LAM", hd_data->h_osc_inertia, 6, 6);
#endif

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main() {
    run<float>();
    return 0;
}

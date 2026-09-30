// Batched host-surface CUDA runner — regression guard for the floating nq-vs-nv
// matrix-buffer stride bug class (see project_grid_floating_nqnv_bug_class).
//
// The standard equivalence runner uses init_gridData<T,1> and calls kernels with
// its OWN device buffers, so it never exercises the gridData h_* host-wrapper copy
// NOR the batch>1 (NUM_TIMESTEPS>1) per-timestep stride. This runner drives the
// BATCHED HOST WRAPPERS (grid::minv / grid::crba) end-to-end through hd_data's h_*
// output buffers at NUM_TIMESTEPS = GRID_BATCH, so a per-timestep stride mismatch
// (kernel writes nv*nv, host copy/malloc uses nq*nq) corrupts slot k>0 and the test
// catches it. Byte-identical for a fixed base (nq == nv).
//
// Input on stdin: GRID_BATCH consecutive q vectors (NUM_POS floats each).
// Output: per-timestep MINV (NV x NV) and M (NV x NV), column-major.
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "grid.cuh"

#ifndef GRID_BATCH
#define GRID_BATCH 3
#endif

template <typename T>
void read_vector(T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        double value;
        if (!(std::cin >> value)) { std::cerr << "read fail " << i << "\n"; std::exit(2); }
        dst[i] = static_cast<T>(value);
    }
}

template <typename T>
void print_matrix_col_major(const std::string &name, int slot, const T *data, int rows, int cols) {
    const std::string tag = name + std::to_string(slot);  // MINV0, MINV1, ... one matrix per batch slot
    std::cout << "BEGIN " << tag << " " << rows << " " << cols << "\n";
    std::cout << std::setprecision(10);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (col) std::cout << " ";
            std::cout << static_cast<double>(data[row + rows * col]);
        }
        std::cout << "\n";
    }
    std::cout << "END " << tag << "\n";
}

template <typename T>
void run() {
    const dim3 block_dimms(1, 1, 1);
    const int nthreads = grid::MAX_PERF_LEVEL_THREADS;
    const dim3 thread_dimms(nthreads, 1, 1);
    constexpr int NQ = grid::NUM_POS;
    constexpr int NV = grid::NUM_VEL;
    constexpr int B = GRID_BATCH;

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, B>();

    // The non-compressed host wrappers read q from h_q_qd_u at stride 3*NUM_JOINTS.
    const int stride = 3 * grid::NUM_JOINTS;
    for (int k = 0; k < B; ++k) {
        read_vector(&hd_data->h_q_qd_u[k * stride], NQ);
        for (int i = NQ; i < stride; ++i) hd_data->h_q_qd_u[k * stride + i] = static_cast<T>(0);
        // mirror into h_q (some wrappers read it) for completeness
        for (int i = 0; i < NQ; ++i) hd_data->h_q[k * grid::NUM_JOINTS + i] = hd_data->h_q_qd_u[k * stride + i];
    }

    grid::minv<T>(hd_data, d_robotModel, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k)
        print_matrix_col_major("MINV", k, &hd_data->h_Minv[k * NV * NV], NV, NV);

#ifdef GRID_HAS_CRBA
    grid::crba<T>(hd_data, d_robotModel, static_cast<T>(0), B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k)
        print_matrix_col_major("M", k, &hd_data->h_M[k * NV * NV], NV, NV);
#endif

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main() {
    run<float>();
    return 0;
}

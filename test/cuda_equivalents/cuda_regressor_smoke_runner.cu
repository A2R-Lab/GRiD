// Test runner for the CUDA joint-torque regressor kernel.
//
// Mirrors `cuda_idsva_so_world_frame_smoke_runner.cu` but invokes
// `inverse_dynamics_regressor` (host launcher) with an explicit caller-allocated
// d_Y output buffer (the regressor output is nv x 10*NUM_BODIES and is NOT a
// gridData field). Used by `test_cuda_regressor` to validate the CUDA emission
// against `RBDReference.inverse_dynamics_regressor` (the verified numpy reference)
// and the structural identity Y @ pi == inverse_dynamics(q,qd,qdd).

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "grid.cuh"

#ifndef GRID_CUDA_REGRESSOR_TEST_THREADS
#define GRID_CUDA_REGRESSOR_TEST_THREADS 64
#endif

template <typename T>
void read_vector(T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        double value;
        if (!(std::cin >> value)) {
            std::cerr << "Failed to read input value " << i << std::endl;
            std::exit(2);
        }
        dst[i] = static_cast<T>(value);
    }
}

template <typename T>
void print_flat(const std::string &name, const T *data, int rows, int cols) {
    std::cout << "BEGIN " << name << " " << rows << " " << cols << "\n";
    std::cout << std::setprecision(10);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (c) std::cout << " ";
            std::cout << static_cast<double>(data[r * cols + c]);
        }
        std::cout << "\n";
    }
    std::cout << "END " << name << "\n";
}

template <typename T>
int run() {
    // Unified gravity convention: GRiD and the RBDReference oracle both use -9.81.
    const T gravity = static_cast<T>(-9.81);
    const dim3 block_dimms(1, 1, 1);
    const int _req_threads = GRID_CUDA_REGRESSOR_TEST_THREADS;
    const int _nthreads = _req_threads < grid::MAX_PERF_LEVEL_THREADS ? _req_threads : grid::MAX_PERF_LEVEL_THREADS;
    const dim3 thread_dimms(_nthreads, 1, 1);

    const int nv = grid::NUM_VEL;
    const int nb = grid::NUM_BODIES;
    const int Y_rows = nv;
    const int Y_cols = 10 * nb;
    const int Y_size = Y_rows * Y_cols;

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    // q|qd|qdd into the q_qd_u host buffer, floating-aware layout:
    //   q  (NUM_POS) | qd (NUM_VEL) | qdd (NUM_VEL), total Q_QD_U_STRIDE.
    read_vector(hd_data->h_q_qd_u, grid::NUM_POS);
    read_vector(&hd_data->h_q_qd_u[grid::NUM_POS], grid::NUM_VEL);
    read_vector(&hd_data->h_q_qd_u[grid::NUM_POS + grid::NUM_VEL], grid::NUM_VEL);

    // caller-allocated output buffer for the regressor.
    T *d_Y = nullptr;
    gpuErrchk(cudaMalloc((void **)&d_Y, Y_size * sizeof(T)));
    T *h_Y = (T *)malloc(Y_size * sizeof(T));

    grid::inverse_dynamics_regressor<T>(
        hd_data, d_Y, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(h_Y, d_Y, Y_size * sizeof(T), cudaMemcpyDeviceToHost));

    int first_bad = -1;
    for (int i = 0; i < Y_size; ++i) {
        if (first_bad < 0 && !std::isfinite(static_cast<double>(h_Y[i]))) first_bad = i;
    }
    if (first_bad >= 0) {
        std::cerr << "Non-finite regressor output at " << first_bad << std::endl;
    }

    T config[5];
    config[0] = static_cast<T>(grid::NUM_POS);
    config[1] = static_cast<T>(grid::NUM_VEL);
    config[2] = static_cast<T>(grid::NUM_BODIES);
    config[3] = static_cast<T>(grid::NUM_JOINTS);
    config[4] = static_cast<T>(grid::INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>());

    print_flat("regressor_config", config, 1, 5);
    print_flat("regressor", h_Y, Y_rows, Y_cols);

    free(h_Y);
    gpuErrchk(cudaFree(d_Y));
    grid::close_grid<T>(streams, d_robot_model, hd_data);
    return 0;
}

int main() {
    return run<float>();
}

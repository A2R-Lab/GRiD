// Test runner for the CUDA FD parameter-gradient kernel (dqdd/dpi = -Minv . Y).
//
// Mirrors `cuda_regressor_smoke_runner.cu` but invokes
// `fd_parameter_gradient` (host launcher) with an explicit caller-allocated
// d_dqdd_dpi output buffer (the output is nv x 10*NUM_BODIES and is NOT a
// gridData field). Used by `test_cuda_fd_parameter_gradient` to validate the
// CUDA emission against `RBDReference.fd_parameter_gradient` (numpy reference).
//
// Input block is q|qd|u (positions, velocities, torques) packed into the
// q_qd_u host buffer in the standard floating-aware layout.

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "grid.cuh"

#ifndef GRID_CUDA_FPG_TEST_THREADS
#define GRID_CUDA_FPG_TEST_THREADS 64
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
    const T gravity = static_cast<T>(9.81);
    const dim3 block_dimms(1, 1, 1);
    const int _req_threads = GRID_CUDA_FPG_TEST_THREADS;
    const int _nthreads = _req_threads < grid::MAX_PERF_LEVEL_THREADS ? _req_threads : grid::MAX_PERF_LEVEL_THREADS;
    const dim3 thread_dimms(_nthreads, 1, 1);

    const int nv = grid::NUM_VEL;
    const int nb = grid::NUM_BODIES;
    const int out_rows = nv;
    const int out_cols = 10 * nb;
    const int out_size = out_rows * out_cols;

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    // q|qd|u into the q_qd_u host buffer, floating-aware layout:
    //   q (NUM_POS) | qd (NUM_VEL) | u (NUM_VEL), total Q_QD_U_STRIDE.
    read_vector(hd_data->h_q_qd_u, grid::NUM_POS);
    read_vector(&hd_data->h_q_qd_u[grid::NUM_POS], grid::NUM_VEL);
    read_vector(&hd_data->h_q_qd_u[grid::NUM_POS + grid::NUM_VEL], grid::NUM_VEL);

    T *d_out = nullptr;
    gpuErrchk(cudaMalloc((void **)&d_out, out_size * sizeof(T)));
    T *h_out = (T *)malloc(out_size * sizeof(T));

    grid::forward_dynamics_parameter_gradient<T>(
        hd_data, d_out, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpy(h_out, d_out, out_size * sizeof(T), cudaMemcpyDeviceToHost));

    int first_bad = -1;
    for (int i = 0; i < out_size; ++i) {
        if (first_bad < 0 && !std::isfinite(static_cast<double>(h_out[i]))) first_bad = i;
    }
    if (first_bad >= 0) {
        std::cerr << "Non-finite FD param-grad output at " << first_bad << std::endl;
    }

    T config[5];
    config[0] = static_cast<T>(grid::NUM_POS);
    config[1] = static_cast<T>(grid::NUM_VEL);
    config[2] = static_cast<T>(grid::NUM_BODIES);
    config[3] = static_cast<T>(grid::NUM_JOINTS);
    config[4] = static_cast<T>(grid::FD_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>());

    print_flat("fpg_config", config, 1, 5);
    print_flat("forward_dynamics_parameter_gradient", h_out, out_rows, out_cols);

    free(h_out);
    gpuErrchk(cudaFree(d_out));
    grid::close_grid<T>(streams, d_robot_model, hd_data);
    return 0;
}

int main() {
    return run<float>();
}

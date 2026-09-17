// Test runner for the CUDA joint-torque-regressor STATE derivative (dY/dx, B.0).
//
// Mirrors `cuda_regressor_smoke_runner.cu` and invokes
// `inverse_dynamics_regressor_gradient` (host launcher). The output d_dY_dx is a
// gridData field (2*nv*nv x 10*NUM_BODIES per timestep: dq block then dqd block,
// each direction c a row-major nv x 10NB matrix); the host copies it back into
// hd_data->h_dY_dx, which this runner reads directly. Used by
// `test_cuda_regressor_gradient` to validate the CUDA emission against
// `RBDReference.inverse_dynamics_regressor_gradient` (the verified numpy oracle)
// and the B.0 identity dY_dx[c] @ pi == dtau_dx[:, c].

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "grid.cuh"

#ifndef GRID_CUDA_REGRESSOR_GRADIENT_TEST_THREADS
#define GRID_CUDA_REGRESSOR_GRADIENT_TEST_THREADS 64
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
    const T gravity = static_cast<T>(-9.81);
    const dim3 block_dimms(1, 1, 1);
    const int _req_threads = GRID_CUDA_REGRESSOR_GRADIENT_TEST_THREADS;
    const int _nthreads = _req_threads < grid::MAX_PERF_LEVEL_THREADS ? _req_threads : grid::MAX_PERF_LEVEL_THREADS;
    const dim3 thread_dimms(_nthreads, 1, 1);

    const int nv = grid::NUM_VEL;
    const int nb = grid::NUM_BODIES;
    const int cols = 10 * nb;
    const int rows = 2 * nv * nv;   // dq half then dqd half, nv directions x nv rows each

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    // q|qd|qdd canonical layout (Q_QD_U_STRIDE == 3*NUM_POS): q@0, qd@NUM_POS,
    // qdd@2*NUM_POS (matches the kernel's s_q_qd_qdd slots).
    read_vector(hd_data->h_q_qd_u, grid::NUM_POS);
    read_vector(&hd_data->h_q_qd_u[grid::NUM_POS], grid::NUM_VEL);
    read_vector(&hd_data->h_q_qd_u[2 * grid::NUM_POS], grid::NUM_VEL);

    grid::inverse_dynamics_regressor_gradient<T>(
        hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    const T *h_dY_dx = hd_data->h_dY_dx;

    int first_bad = -1;
    for (int i = 0; i < rows * cols; ++i) {
        if (first_bad < 0 && !std::isfinite(static_cast<double>(h_dY_dx[i]))) first_bad = i;
    }
    if (first_bad >= 0) {
        std::cerr << "Non-finite dY/dx output at " << first_bad << std::endl;
    }

    T config[5];
    config[0] = static_cast<T>(grid::NUM_POS);
    config[1] = static_cast<T>(grid::NUM_VEL);
    config[2] = static_cast<T>(grid::NUM_BODIES);
    config[3] = static_cast<T>(grid::NUM_JOINTS);
    config[4] = static_cast<T>(grid::INVERSE_DYNAMICS_REGRESSOR_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>());

    print_flat("regressor_gradient_config", config, 1, 5);
    print_flat("regressor_gradient", h_dY_dx, rows, cols);

    grid::close_grid<T>(streams, d_robot_model, hd_data);
    return 0;
}

int main() {
    return run<float>();
}

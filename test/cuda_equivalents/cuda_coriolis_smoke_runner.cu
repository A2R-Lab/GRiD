// Test runner for the CUDA full Coriolis matrix C(q, qd).
//
// Mirrors cuda_energy_regressor_smoke_runner.cu and invokes the host launcher
// `coriolis_matrix`. The output lives in gridData (hd_data->d_coriolis, nv*nv);
// the host copies it back into hd_data->h_coriolis, which this runner reads
// directly. Used by `test_cuda_coriolis` to validate the CUDA emission against
// RBDReference._EnergyMixin.coriolis_matrix and the identity C qd + g == nle.
//
// Accepts an optional argv[1] = block thread count (for the thread-invariance
// sweep). Defaults to the compile-time GRID_CUDA_CORIOLIS_TEST_THREADS. Both are
// clamped to MAX_PERF_LEVEL_THREADS (the kernel __launch_bounds__ cap).

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "grid.cuh"

#ifndef GRID_CUDA_CORIOLIS_TEST_THREADS
#define GRID_CUDA_CORIOLIS_TEST_THREADS 64
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
int run(int req_threads) {
    // Unified gravity convention: GRiD and the RBDReference oracle both use -9.81.
    const T gravity = static_cast<T>(-9.81);
    const dim3 block_dimms(1, 1, 1);
    const int _nthreads = req_threads < grid::MAX_PERF_LEVEL_THREADS ? req_threads : grid::MAX_PERF_LEVEL_THREADS;
    const dim3 thread_dimms(_nthreads, 1, 1);

    const int nv = grid::NUM_VEL;
    const int C_size = nv * nv;

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    // q|qd|qdd into the q_qd_u host buffer, floating-aware layout:
    //   q (NUM_POS) | qd (NUM_VEL) | qdd (NUM_VEL), total Q_QD_U_STRIDE.
    read_vector(hd_data->h_q_qd_u, grid::NUM_POS);
    read_vector(&hd_data->h_q_qd_u[grid::NUM_POS], grid::NUM_VEL);
    read_vector(&hd_data->h_q_qd_u[grid::NUM_POS + grid::NUM_VEL], grid::NUM_VEL);
    // The coriolis host reads the compressed buffers hd_data->h_q_qd (non-compressed
    // branch uses h_q_qd_u). Mirror q|qd into h_q_qd so both code paths agree.
    for (int i = 0; i < grid::NUM_JOINTS; ++i) hd_data->h_q_qd[i] = hd_data->h_q_qd_u[i];
    for (int i = 0; i < grid::NUM_VEL; ++i) hd_data->h_q_qd[grid::NUM_JOINTS + i] = hd_data->h_q_qd_u[grid::NUM_POS + i];

    grid::coriolis_matrix<T>(
        hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    const T *h_C = hd_data->h_coriolis;

    int first_bad = -1;
    for (int i = 0; i < C_size; ++i) {
        if (first_bad < 0 && !std::isfinite(static_cast<double>(h_C[i]))) first_bad = i;
    }
    if (first_bad >= 0) {
        std::cerr << "Non-finite coriolis output at " << first_bad << std::endl;
    }

    T config[4];
    config[0] = static_cast<T>(grid::NUM_POS);
    config[1] = static_cast<T>(grid::NUM_VEL);
    config[2] = static_cast<T>(grid::NUM_BODIES);
    config[3] = static_cast<T>(grid::NUM_JOINTS);

    print_flat("coriolis_config", config, 1, 4);
    print_flat("coriolis", h_C, nv, nv);

    grid::close_grid<T>(streams, d_robot_model, hd_data);
    return 0;
}

int main(int argc, char **argv) {
    int req_threads = GRID_CUDA_CORIOLIS_TEST_THREADS;
    if (argc > 1) req_threads = std::atoi(argv[1]);
    return run<float>(req_threads);
}

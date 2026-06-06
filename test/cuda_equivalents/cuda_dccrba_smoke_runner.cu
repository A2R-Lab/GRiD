// Test runner for the analytic dCCRBA surfaces:
//   dccrba             dA_dq[:,k,m]  (6*NV*NV, layout dA[row + 6*k + 6*NV*m])
//   cmm_time_variation Adot          (6*NV, column-major [linear;angular] @ CoM)
//
// Runs in DOUBLE (the analytic tensor is validated to the TIGHT value bucket; an
// fp64 kernel avoids float32 conditioning noise on big robots). Reads both outputs
// from the gridData host buffers. Accepts argv[1] = block thread count for the
// thread-invariance sweep (clamped to the kernel __launch_bounds__ cap).

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "grid.cuh"

#ifndef GRID_CUDA_DCCRBA_TEST_THREADS
#define GRID_CUDA_DCCRBA_TEST_THREADS 64
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
    std::cout << std::setprecision(16);
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
    const dim3 block_dimms(1, 1, 1);
    const int _nthreads = req_threads < grid::MAX_PERF_LEVEL_THREADS ? req_threads : grid::MAX_PERF_LEVEL_THREADS;
    const dim3 thread_dimms(_nthreads, 1, 1);

    const int nv = grid::NUM_VEL;
    const int dccrba_size = 6 * nv * nv;
    const int adot_size = 6 * nv;

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    // q|qd|qdd into the q_qd_u host buffer, floating-aware layout.
    read_vector(hd_data->h_q_qd_u, grid::NUM_POS);
    read_vector(&hd_data->h_q_qd_u[grid::NUM_POS], grid::NUM_VEL);
    read_vector(&hd_data->h_q_qd_u[grid::NUM_POS + grid::NUM_VEL], grid::NUM_VEL);
    // q-only buffer (dccrba reads d_q) and q|qd buffer (cmm reads d_q_qd_u).
    for (int i = 0; i < grid::NUM_JOINTS; ++i) hd_data->h_q[i] = hd_data->h_q_qd_u[i];
    for (int i = 0; i < grid::NUM_JOINTS; ++i) hd_data->h_q_qd[i] = hd_data->h_q_qd_u[i];
    for (int i = 0; i < grid::NUM_VEL; ++i) hd_data->h_q_qd[grid::NUM_JOINTS + i] = hd_data->h_q_qd_u[grid::NUM_POS + i];

    grid::cmm_time_variation<T>(hd_data, d_robot_model, 1, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    grid::dccrba<T>(hd_data, d_robot_model, 1, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    const T *h_dccrba = hd_data->h_dccrba;
    const T *h_adot = hd_data->h_cmm_time_variation;

    int first_bad = -1;
    for (int i = 0; i < dccrba_size; ++i)
        if (first_bad < 0 && !std::isfinite(static_cast<double>(h_dccrba[i]))) first_bad = i;
    for (int i = 0; i < adot_size; ++i)
        if (first_bad < 0 && !std::isfinite(static_cast<double>(h_adot[i]))) first_bad = i;
    if (first_bad >= 0) std::cerr << "Non-finite dccrba/adot output at " << first_bad << std::endl;

    T config[4];
    config[0] = static_cast<T>(grid::NUM_POS);
    config[1] = static_cast<T>(grid::NUM_VEL);
    config[2] = static_cast<T>(grid::NUM_BODIES);
    config[3] = static_cast<T>(grid::NUM_JOINTS);

    print_flat("dccrba_config", config, 1, 4);
    print_flat("dccrba", h_dccrba, dccrba_size, 1);
    print_flat("cmm_time_variation", h_adot, adot_size, 1);

    grid::close_grid<T>(streams, d_robot_model, hd_data);
    return 0;
}

int main(int argc, char **argv) {
    int req_threads = GRID_CUDA_DCCRBA_TEST_THREADS;
    if (argc > 1) req_threads = std::atoi(argv[1]);
    return run<double>(req_threads);
}

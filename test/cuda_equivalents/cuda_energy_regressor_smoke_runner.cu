// Test runner for the CUDA energy regressors (kinetic + potential).
//
// Mirrors `cuda_regressor_smoke_runner.cu` and invokes the host launchers
// `kinetic_energy_regressor` / `potential_energy_regressor`. Both outputs live in
// gridData (hd_data->d_ke_regressor / d_pe_regressor, each length 10*NUM_BODIES);
// the hosts copy them back into hd_data->h_ke_regressor / h_pe_regressor, which
// this runner reads directly. Used by `test_cuda_energy_regressor` to validate the
// CUDA emission against RBDReference.kinetic_energy_regressor /
// potential_energy_regressor and the identities y_KE.pi == kinetic_energy,
// y_PE.pi == potential_energy.

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

    const int nb = grid::NUM_BODIES;
    const int Y_size = 10 * nb;

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    // q|qd|qdd into the q_qd_u host buffer. Canonical layout (Q_QD_U_STRIDE == 3*NUM_POS):
    // each field gets a NUM_POS(=nq)-wide slot -> q@0, qd@nq, qdd@2*nq (kernels read the
    // 3rd field at 2*NUM_POS). KE/PE do not consume qdd so this is harmless here, but use
    // the canonical nq-based offset for consistency with the other q|qd|* runners.
    read_vector(hd_data->h_q_qd_u, grid::NUM_POS);
    read_vector(&hd_data->h_q_qd_u[grid::NUM_POS], grid::NUM_VEL);
    read_vector(&hd_data->h_q_qd_u[2 * grid::NUM_POS], grid::NUM_VEL);
    // The KE / PE hosts read from the compressed buffers hd_data->h_q_qd / h_q
    // (non-compressed branch uses h_q_qd_u for KE and h_q for PE). Mirror q (and
    // q|qd) into those host buffers so both code paths see consistent inputs.
    for (int i = 0; i < grid::NUM_JOINTS; ++i) hd_data->h_q[i] = hd_data->h_q_qd_u[i];
    for (int i = 0; i < grid::NUM_JOINTS; ++i) hd_data->h_q_qd[i] = hd_data->h_q_qd_u[i];
    for (int i = 0; i < grid::NUM_VEL; ++i) hd_data->h_q_qd[grid::NUM_JOINTS + i] = hd_data->h_q_qd_u[grid::NUM_POS + i];

    grid::kinetic_energy_regressor<T>(
        hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    grid::potential_energy_regressor<T>(
        hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());

    const T *h_ke = hd_data->h_ke_regressor;
    const T *h_pe = hd_data->h_pe_regressor;

    int first_bad = -1;
    for (int i = 0; i < Y_size; ++i) {
        if (first_bad < 0 && !std::isfinite(static_cast<double>(h_ke[i]))) first_bad = i;
        if (first_bad < 0 && !std::isfinite(static_cast<double>(h_pe[i]))) first_bad = i;
    }
    if (first_bad >= 0) {
        std::cerr << "Non-finite energy regressor output at " << first_bad << std::endl;
    }

    T config[4];
    config[0] = static_cast<T>(grid::NUM_POS);
    config[1] = static_cast<T>(grid::NUM_VEL);
    config[2] = static_cast<T>(grid::NUM_BODIES);
    config[3] = static_cast<T>(grid::NUM_JOINTS);

    print_flat("regressor_config", config, 1, 4);
    print_flat("ke_regressor", h_ke, 1, Y_size);
    print_flat("pe_regressor", h_pe, 1, Y_size);

    grid::close_grid<T>(streams, d_robot_model, hd_data);
    return 0;
}

int main() {
    return run<float>();
}

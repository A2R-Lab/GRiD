#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "grid.cuh"

#ifndef GRID_CUDA_SECOND_ORDER_TEST_THREADS
#define GRID_CUDA_SECOND_ORDER_TEST_THREADS 64
#endif

#ifndef GRID_CUDA_SECOND_ORDER_ENABLE_FDSVA
#define GRID_CUDA_SECOND_ORDER_ENABLE_FDSVA 1
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
void print_flat(const std::string &name, const T *data, int count) {
    std::cout << "BEGIN " << name << " 1 " << count << "\n";
    std::cout << std::setprecision(10);
    for (int i = 0; i < count; ++i) {
        if (i) {
            std::cout << " ";
        }
        std::cout << static_cast<double>(data[i]);
    }
    std::cout << "\nEND " << name << "\n";
}

template <typename T>
int run() {
    const T gravity = static_cast<T>(9.81);
    const dim3 block_dimms(1, 1, 1);
    // Clamp to the robot's MAX_PERF_LEVEL_THREADS (the kernels' __launch_bounds__ cap,
    // resolved dynamically from the generated header) so a swept count above the
    // bound doesn't fail with cudaErrorInvalidValue.
    const int _req_threads = GRID_CUDA_SECOND_ORDER_TEST_THREADS;
    const int _nthreads = _req_threads < grid::MAX_PERF_LEVEL_THREADS ? _req_threads : grid::MAX_PERF_LEVEL_THREADS;
    const dim3 thread_dimms(_nthreads, 1, 1);

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    read_vector(hd_data->h_q_qd_u, grid::NUM_POS);
    read_vector(&hd_data->h_q_qd_u[grid::NUM_POS], grid::NUM_VEL);
    read_vector(&hd_data->h_q_qd_u[grid::NUM_POS + grid::NUM_VEL], grid::NUM_VEL);

    grid::idsva_so_body_frame<T>(
        hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());

#if GRID_CUDA_SECOND_ORDER_ENABLE_FDSVA
    grid::fdsva_so<T>(
        hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
#endif

    const int tensor_count = grid::SECOND_ORDER_TENSOR_SIZE;
#if !GRID_CUDA_SECOND_ORDER_ENABLE_FDSVA
    for (int i = 0; i < tensor_count; ++i) {
        hd_data->h_df2[i] = static_cast<T>(0);
    }
#endif
    int first_bad_idsva = -1;
    int first_bad_fdsva = -1;
    for (int i = 0; i < tensor_count; ++i) {
        if (first_bad_idsva < 0 &&
            !std::isfinite(static_cast<double>(hd_data->h_idsva_so[i]))) {
            first_bad_idsva = i;
        }
        if (first_bad_fdsva < 0 &&
            !std::isfinite(static_cast<double>(hd_data->h_df2[i]))) {
            first_bad_fdsva = i;
        }
    }
    if (first_bad_idsva >= 0) {
        std::cerr << "Non-finite idsva_so output at "
                  << first_bad_idsva << std::endl;
    }
    if (first_bad_fdsva >= 0) {
        std::cerr << "Non-finite fdsva_so output at "
                  << first_bad_fdsva << std::endl;
    }

    T config[12];
    config[0] = static_cast<T>(grid::IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T>());
    config[1] = static_cast<T>(grid::FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T>());
    config[2] = static_cast<T>(grid::GRID_IDSVA_SO_USES_GLOBAL_OUTPUT);
    config[3] = static_cast<T>(grid::GRID_FDSVA_SO_USES_GLOBAL_TENSORS);
    config[4] = static_cast<T>(grid::GRID_FDSVA_SO_USES_WORKSPACE_TEMP);
    config[5] = static_cast<T>(grid::GRID_GENERATES_IDSVA_SO_BODY_FRAME);
    config[6] = static_cast<T>(grid::GRID_GENERATES_FDSVA_SO);
    config[7] = static_cast<T>(grid::NUM_POS);
    config[8] = static_cast<T>(grid::NUM_VEL);
    config[9] = static_cast<T>(grid::NUM_BODIES);
    config[10] = static_cast<T>(grid::Q_QD_U_STRIDE);
    config[11] = static_cast<T>(tensor_count);

    print_flat("second_order_config", config, 12);
    print_flat("idsva_so_body_frame", hd_data->h_idsva_so, tensor_count);
    print_flat("fdsva_so", hd_data->h_df2, tensor_count);

    grid::close_grid<T>(streams, d_robot_model, hd_data);
    return 0;
}

int main() {
    return run<float>();
}

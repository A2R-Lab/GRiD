// Four distinct states/forces; one or two blocks must stride through the batch.
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include "grid.cuh"

template <typename T>
void read_values(T *out, int n) {
    for (int i = 0; i < n; ++i) {
        double v;
        if (!(std::cin >> v)) std::exit(2);
        out[i] = static_cast<T>(v);
    }
}

template <typename T>
void print_matrix(const std::string &name, const T *data, int rows, int cols) {
    std::cout << "BEGIN " << name << " " << rows << " " << cols << "\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c)
            std::cout << std::setprecision(10) << data[r + rows*c] << " ";
        std::cout << "\n";
    }
    std::cout << "END " << name << "\n";
}

template <typename T, grid::IntegratorType IT>
void evaluate(const std::string &name, grid::gridData<T> *data,
              grid::robotModel<T> *model, cudaStream_t *streams,
              dim3 blocks, dim3 threads, T dt) {
    constexpr int B = 4, NX = grid::NUM_POS + grid::NUM_VEL;
    constexpr int NV = grid::NUM_VEL, NG = 6 * NV * NV;
    grid::integrator<T, IT>(data, model, T(-9.81), dt, B, blocks, threads, streams);
    for (int k = 0; k < B; ++k)
        print_matrix(name + "_value_" + std::to_string(k), data->h_x_kp1 + k*NX, 1, NX);
    grid::integrator_gradient<T, IT>(data, model, T(-9.81), dt, B, blocks, threads, streams);
    for (int k = 0; k < B; ++k)
        print_matrix(name + "_gradient_" + std::to_string(k), data->h_dAB + k*NG, 2*NV, 3*NV);
    grid::integrator_with_gradient<T, IT>(data, model, T(-9.81), dt, B, blocks, threads, streams);
    for (int k = 0; k < B; ++k) {
        print_matrix(name + "_both_value_" + std::to_string(k), data->h_x_kp1 + k*NX, 1, NX);
        print_matrix(name + "_both_gradient_" + std::to_string(k), data->h_dAB + k*NG, 2*NV, 3*NV);
    }
}

template <typename T>
void run(int nthreads, int nblocks, bool null_forces) {
    constexpr int B = 4, NQ = grid::NUM_POS, NV = grid::NUM_VEL, NF = 6*grid::NUM_BODIES;
    auto *streams = grid::init_grid_streams<T>();
    gpuErrchk(grid::init_grid_kernel_attr_integrator_checked<T>());
    gpuErrchk(grid::init_grid_kernel_attr_integrator_gradient_checked<T>());
    gpuErrchk(grid::init_grid_kernel_attr_integrator_with_gradient_checked<T>());
    auto *model = grid::init_robotModel<T>();
    auto *data = grid::init_gridData<T, B>();
    for (int k = 0; k < B; ++k) {
        T *row = data->h_q_qd_u + k*3*NQ;
        for (int j = 0; j < 3*NQ; ++j) row[j] = T(0);
        read_values(row, NQ);
        read_values(row + NQ, NV);
        read_values(row + 2*NQ, NV);
    }
    T dt;
    read_values(&dt, 1);
    read_values(data->h_f_ext, B*NF);
    gpuErrchk(cudaMemcpy(data->d_f_ext, data->h_f_ext, B*NF*sizeof(T), cudaMemcpyHostToDevice));
    T *saved = data->d_f_ext;
    if (null_forces) data->d_f_ext = nullptr;
    dim3 blocks(nblocks), threads(nthreads);
    evaluate<T, grid::IntegratorType::EULER>("euler", data, model, streams, blocks, threads, dt);
    evaluate<T, grid::IntegratorType::SEMI_IMPLICIT_EULER>("semi_implicit_euler", data, model, streams, blocks, threads, dt);
    evaluate<T, grid::IntegratorType::MIDPOINT>("midpoint", data, model, streams, blocks, threads, dt);
    evaluate<T, grid::IntegratorType::RK4>("rk4", data, model, streams, blocks, threads, dt);
    evaluate<T, grid::IntegratorType::TRAPEZOIDAL>("trapezoidal", data, model, streams, blocks, threads, dt);
    evaluate<T, grid::IntegratorType::CONSTANT_ACCELERATION>("constant_acceleration", data, model, streams, blocks, threads, dt);
    data->d_f_ext = saved;
    grid::close_grid<T>(streams, model, data);
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    run<float>(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]) != 0);
}

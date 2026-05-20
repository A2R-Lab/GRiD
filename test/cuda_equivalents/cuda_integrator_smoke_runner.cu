// CUDA smoke runner for the generated integrator kernels.
//
// Input on stdin (whitespace-separated floats):
//   q (NUM_JOINTS) qd (NUM_VEL) u (NUM_VEL) dt
//
// Output (BEGIN/END framed blocks, column-major print):
//   integrator_euler_x_kp1                 -> 1 x (2*NUM_VEL)
//   integrator_euler_dAB                   -> (2*NUM_VEL) x (3*NUM_VEL)
//   integrator_euler_dAB_with_x_kp1        -> (2*NUM_VEL) x (3*NUM_VEL)
//   integrator_euler_x_kp1_with_dAB        -> 1 x (2*NUM_VEL)
//   integrator_si_euler_x_kp1              -> 1 x (2*NUM_VEL)
//   integrator_si_euler_dAB                -> (2*NUM_VEL) x (3*NUM_VEL)
//   integrator_si_euler_dAB_with_x_kp1     -> (2*NUM_VEL) x (3*NUM_VEL)
//   integrator_si_euler_x_kp1_with_dAB     -> 1 x (2*NUM_VEL)
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "grid.cuh"

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
void print_vector(const std::string &name, const T *data, int count) {
    print_matrix_col_major(name, data, 1, count);
}

template <typename T, grid::IntegratorType IT>
void run_value_only(const std::string &prefix,
                    grid::gridData<T> *hd_data,
                    grid::robotModel<T> *d_robotModel,
                    cudaStream_t *streams,
                    const dim3 &block_dimms,
                    const dim3 &thread_dimms,
                    const T *original_q_qd_u,
                    T gravity,
                    T dt) {
    const int N = grid::NUM_JOINTS;
    std::memcpy(hd_data->h_q_qd_u, original_q_qd_u, 3 * N * sizeof(T));
    grid::integrator<T, IT>(hd_data, d_robotModel, gravity, dt, 1, block_dimms, thread_dimms, streams);
    print_vector(prefix + "_x_kp1", hd_data->h_x_kp1, 2 * N);
}

template <typename T, grid::IntegratorType IT>
void run_one(const std::string &prefix,
             grid::gridData<T> *hd_data,
             grid::robotModel<T> *d_robotModel,
             cudaStream_t *streams,
             const dim3 &block_dimms,
             const dim3 &thread_dimms,
             const T *original_q_qd_u,
             T gravity,
             T dt) {
    const int N = grid::NUM_JOINTS;
    // restore inputs in case the previous run touched them
    std::memcpy(hd_data->h_q_qd_u, original_q_qd_u, 3 * N * sizeof(T));

    // value-only
    grid::integrator<T, IT>(hd_data, d_robotModel, gravity, dt, 1, block_dimms, thread_dimms, streams);
    print_vector(prefix + "_x_kp1", hd_data->h_x_kp1, 2 * N);

    // gradient-only
    grid::integrator_gradient<T, IT>(hd_data, d_robotModel, gravity, dt, 1, block_dimms, thread_dimms, streams);
    print_matrix_col_major(prefix + "_dAB", hd_data->h_dAB, 2 * N, 3 * N);

    // both-at-once
    grid::integrator_gradient_with_x_kp1<T, IT>(hd_data, d_robotModel, gravity, dt, 1, block_dimms, thread_dimms, streams);
    print_vector(prefix + "_x_kp1_with_dAB", hd_data->h_x_kp1, 2 * N);
    print_matrix_col_major(prefix + "_dAB_with_x_kp1", hd_data->h_dAB, 2 * N, 3 * N);
}

template <typename T>
void run() {
    const T gravity = static_cast<T>(9.81);
    const dim3 block_dimms(1, 1, 1);
    const dim3 thread_dimms(grid::SUGGESTED_THREADS, 1, 1);

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    const int N = grid::NUM_JOINTS;
    std::vector<T> h_q(N), h_qd(N), h_u(N);
    read_vector(h_q.data(), N);
    read_vector(h_qd.data(), N);
    read_vector(h_u.data(), N);
    double dt_double;
    if (!(std::cin >> dt_double)) {
        std::cerr << "Failed to read dt" << std::endl;
        std::exit(2);
    }
    const T dt = static_cast<T>(dt_double);

    // Pack into h_q_qd_u
    std::vector<T> original(3 * N);
    for (int i = 0; i < N; ++i) {
        original[i]          = h_q[i];
        original[N + i]      = h_qd[i];
        original[2 * N + i]  = h_u[i];
    }
    std::memcpy(hd_data->h_q_qd_u, original.data(), 3 * N * sizeof(T));

    print_vector("input_q",  h_q.data(),  N);
    print_vector("input_qd", h_qd.data(), N);
    print_vector("input_u",  h_u.data(),  N);
    {
        const double dt_v = static_cast<double>(dt);
        std::cout << "BEGIN input_dt 1 1\n";
        std::cout << std::setprecision(10) << dt_v << "\n";
        std::cout << "END input_dt\n";
    }

    run_one<T, grid::IntegratorType::EULER>("integrator_euler",
        hd_data, d_robotModel, streams, block_dimms, thread_dimms, original.data(), gravity, dt);
    run_one<T, grid::IntegratorType::SEMI_IMPLICIT_EULER>("integrator_si_euler",
        hd_data, d_robotModel, streams, block_dimms, thread_dimms, original.data(), gravity, dt);

    // Midpoint / RK3 / RK4: full path (value + gradient + both).
    run_one<T, grid::IntegratorType::MIDPOINT>("integrator_midpoint",
        hd_data, d_robotModel, streams, block_dimms, thread_dimms, original.data(), gravity, dt);
    run_one<T, grid::IntegratorType::RK3>("integrator_rk3",
        hd_data, d_robotModel, streams, block_dimms, thread_dimms, original.data(), gravity, dt);
    run_one<T, grid::IntegratorType::RK4>("integrator_rk4",
        hd_data, d_robotModel, streams, block_dimms, thread_dimms, original.data(), gravity, dt);

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main() {
    run<float>();
    return 0;
}

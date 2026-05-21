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

// Whether this generated header is for a floating-base robot. The integrator
// gradient kernels are emitted only for fixed-base (nq == nv); floating-base
// supports the value path only.
static constexpr bool GRID_INTEGRATOR_FLOATING = (grid::NUM_POS != grid::NUM_VEL);

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
    const int input_count = grid::NUM_POS + 2 * grid::NUM_VEL;
    const int x_kp1_count = grid::NUM_POS + grid::NUM_VEL;
    std::memcpy(hd_data->h_q_qd_u, original_q_qd_u, input_count * sizeof(T));
    grid::integrator<T, IT>(hd_data, d_robotModel, gravity, dt, 1, block_dimms, thread_dimms, streams);
    print_vector(prefix + "_x_kp1", hd_data->h_x_kp1, x_kp1_count);
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
    const int input_count = grid::NUM_POS + 2 * grid::NUM_VEL;
    const int x_kp1_count = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    // value-only (always)
    std::memcpy(hd_data->h_q_qd_u, original_q_qd_u, input_count * sizeof(T));
    grid::integrator<T, IT>(hd_data, d_robotModel, gravity, dt, 1, block_dimms, thread_dimms, streams);
    print_vector(prefix + "_x_kp1", hd_data->h_x_kp1, x_kp1_count);

    // gradient + both-at-once. The gradient kernels are emitted whenever
    // integrator_gradient is requested; we must preprocessor-guard (not just
    // if-constexpr) because a discarded if-constexpr branch is still
    // name-looked-up and the symbols are absent in a value-only build.
    //
    // Floating-base supports the gradient only for EULER right now (SI-Euler /
    // Midpoint / RK3 / RK4 floating gradients static_assert in the kernel). The
    // inner `if constexpr` keeps those kernels from being instantiated for
    // floating-base, so their static_asserts never fire.
#if GRID_HAS_INTEGRATOR_GRADIENT
    (void) nv;
    if constexpr (!GRID_INTEGRATOR_FLOATING || IT == grid::IntegratorType::EULER) {
        grid::integrator_gradient<T, IT>(hd_data, d_robotModel, gravity, dt, 1, block_dimms, thread_dimms, streams);
        print_matrix_col_major(prefix + "_dAB", hd_data->h_dAB, 2 * nv, 3 * nv);

        grid::integrator_gradient_with_x_kp1<T, IT>(hd_data, d_robotModel, gravity, dt, 1, block_dimms, thread_dimms, streams);
        print_vector(prefix + "_x_kp1_with_dAB", hd_data->h_x_kp1, x_kp1_count);
        print_matrix_col_major(prefix + "_dAB_with_x_kp1", hd_data->h_dAB, 2 * nv, 3 * nv);
    }
#else
    (void) nv;
#endif
}

template <typename T>
void run() {
    const T gravity = static_cast<T>(9.81);
    const dim3 block_dimms(1, 1, 1);
    const dim3 thread_dimms(grid::SUGGESTED_THREADS, 1, 1);

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    const int nq = grid::NUM_POS;
    const int nv = grid::NUM_VEL;
    std::vector<T> h_q(nq), h_qd(nv), h_u(nv);
    read_vector(h_q.data(), nq);
    read_vector(h_qd.data(), nv);
    read_vector(h_u.data(), nv);
    double dt_double;
    if (!(std::cin >> dt_double)) {
        std::cerr << "Failed to read dt" << std::endl;
        std::exit(2);
    }
    const T dt = static_cast<T>(dt_double);

    // Pack into h_q_qd_u as [q (nq) | qd (nv) | u (nv)].
    const int input_count = nq + 2 * nv;
    std::vector<T> original(input_count);
    for (int i = 0; i < nq; ++i) original[i] = h_q[i];
    for (int i = 0; i < nv; ++i) {
        original[nq + i]      = h_qd[i];
        original[nq + nv + i] = h_u[i];
    }
    std::memcpy(hd_data->h_q_qd_u, original.data(), input_count * sizeof(T));

    print_vector("input_q",  h_q.data(),  nq);
    print_vector("input_qd", h_qd.data(), nv);
    print_vector("input_u",  h_u.data(),  nv);
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

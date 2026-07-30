// CUDA equivalence runner for the SPHERICAL (ball) joint integrator GRADIENT.
// Mirrors cuda_integrator_smoke_runner.cu's gradient sections but instantiates
// ONLY the single-stage IntegratorTypes (EULER / SEMI_IMPLICIT_EULER /
// TRAPEZOIDAL) — spherical multi-stage RK gradients are a follow-on slice and
// static_assert out, so an RK instantiation would fail this TU's compile.
//
// Input on stdin: q (NUM_POS) qd (NUM_VEL) u (NUM_VEL) dt
// Output blocks per IT prefix:
//   <prefix>_dAB               -> (2*NUM_VEL) x (3*NUM_VEL), column-major
//   <prefix>_x_kp1_with_dAB    -> 1 x (NUM_POS + NUM_VEL)
//   <prefix>_dAB_with_x_kp1    -> (2*NUM_VEL) x (3*NUM_VEL)
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "grid.cuh"

int g_num_threads = 0;

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
    std::cout << std::setprecision(12);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (c) std::cout << " ";
            std::cout << static_cast<double>(data[c * rows + r]);
        }
        std::cout << "\n";
    }
    std::cout << "END " << name << "\n";
}

template <typename T>
void print_vector(const std::string &name, const T *data, int count) {
    std::cout << "BEGIN " << name << " 1 " << count << "\n";
    std::cout << std::setprecision(12);
    for (int i = 0; i < count; ++i) {
        if (i) std::cout << " ";
        std::cout << static_cast<double>(data[i]);
    }
    std::cout << "\nEND " << name << "\n";
}

template <typename T, grid::IntegratorType IT>
void run_one(const std::string &prefix,
             grid::gridData<T> *hd_data,
             grid::robotModel<T> *d_robotModel,
             cudaStream_t *streams,
             const dim3 &block_dimms,
             const dim3 &thread_dimms,
             const T *original_q_qd_u,
             T gravity, T dt) {
    const int input_count = 3 * grid::NUM_POS;   // canonical nq-wide slots
    const int x_kp1_count = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;

    std::memcpy(hd_data->h_q_qd_u, original_q_qd_u, input_count * sizeof(T));
    grid::integrator_gradient<T, IT>(hd_data, d_robotModel, gravity, dt, 1, block_dimms, thread_dimms, streams);
    print_matrix_col_major(prefix + "_dAB", hd_data->h_dAB, 2 * nv, 3 * nv);

    std::memcpy(hd_data->h_q_qd_u, original_q_qd_u, input_count * sizeof(T));
    grid::integrator_with_gradient<T, IT>(hd_data, d_robotModel, gravity, dt, 1, block_dimms, thread_dimms, streams);
    print_vector(prefix + "_x_kp1_with_dAB", hd_data->h_x_kp1, x_kp1_count);
    print_matrix_col_major(prefix + "_dAB_with_x_kp1", hd_data->h_dAB, 2 * nv, 3 * nv);
}

template <typename T>
void run() {
    const T gravity = static_cast<T>(-9.81);
    const dim3 block_dimms(1, 1, 1);
    const int requested = g_num_threads > 0 ? g_num_threads : grid::MAX_PERF_LEVEL_THREADS;
    const int nthreads = requested < grid::MAX_PERF_LEVEL_THREADS ? requested : grid::MAX_PERF_LEVEL_THREADS;
    const dim3 thread_dimms(nthreads, 1, 1);

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
    print_vector("input_q", h_q.data(), nq);
    print_vector("input_qd", h_qd.data(), nv);
    print_vector("input_u", h_u.data(), nv);

    // canonical nq-wide slots [q | qd @ nq | u @ 2*nq]; unused tails zeroed.
    const int input_count = 3 * nq;
    std::vector<T> original(input_count, T(0));
    for (int i = 0; i < nq; ++i) original[i] = h_q[i];
    for (int i = 0; i < nv; ++i) {
        original[nq + i] = h_qd[i];
        original[2 * nq + i] = h_u[i];
    }

    run_one<T, grid::IntegratorType::EULER>("integrator_euler", hd_data, d_robotModel, streams, block_dimms, thread_dimms, original.data(), gravity, dt);
    run_one<T, grid::IntegratorType::SEMI_IMPLICIT_EULER>("integrator_si_euler", hd_data, d_robotModel, streams, block_dimms, thread_dimms, original.data(), gravity, dt);
    run_one<T, grid::IntegratorType::TRAPEZOIDAL>("integrator_trapezoidal", hd_data, d_robotModel, streams, block_dimms, thread_dimms, original.data(), gravity, dt);

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main(int argc, char **argv) {
    if (argc > 1) {
        int requested = std::atoi(argv[1]);
        g_num_threads = requested > 0 ? requested : 0;
    }
    const char *equiv_t = std::getenv("GRID_EQUIV_T");
    if (equiv_t != nullptr && std::string(equiv_t) == "double") {
        run<double>();
    } else {
        run<float>();
    }
    return 0;
}

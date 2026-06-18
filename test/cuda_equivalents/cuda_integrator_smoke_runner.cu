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

// Block thread count for all integrator kernel launches. 0 => use
// grid::MAX_PERF_LEVEL_THREADS. Overridable via argv[1] so the test harness can
// sweep warp counts to catch thread-count-dependent races.
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
    const int input_count = 3 * grid::NUM_POS;  // gridData canonical [q|qd@nq|u@2nq] (nq-wide slots)
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
    const int input_count = 3 * grid::NUM_POS;  // gridData canonical [q|qd@nq|u@2nq] (nq-wide slots)
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
    // All five integrator gradients are now emitted for both fixed- and
    // floating-base (the floating SI-Euler / Midpoint / RK3 / RK4 gradients add
    // the SE(3) dIntegrate chain-rule wiring), so no per-IT guard is needed.
#if GRID_HAS_INTEGRATOR_GRADIENT
    (void) nv;
    grid::integrator_gradient<T, IT>(hd_data, d_robotModel, gravity, dt, 1, block_dimms, thread_dimms, streams);
    print_matrix_col_major(prefix + "_dAB", hd_data->h_dAB, 2 * nv, 3 * nv);

    grid::integrator_with_gradient<T, IT>(hd_data, d_robotModel, gravity, dt, 1, block_dimms, thread_dimms, streams);
    print_vector(prefix + "_x_kp1_with_dAB", hd_data->h_x_kp1, x_kp1_count);
    print_matrix_col_major(prefix + "_dAB_with_x_kp1", hd_data->h_dAB, 2 * nv, 3 * nv);
#else
    (void) nv;
#endif
}

template <typename T>
void run() {
    const T gravity = static_cast<T>(-9.81);
    const dim3 block_dimms(1, 1, 1);
    // The integrator kernels are compiled with
    // __launch_bounds__(tier_max_threads<TIER>()) (= MAX_PERF_LEVEL_THREADS at
    // TIER_SHARED). Launching with MORE threads than that bound fails with
    // cudaErrorInvalidValue, so a swept count above the bound (e.g. 448 on a
    // small robot whose MAX_PERF_LEVEL_THREADS is 352) must be clamped down. The
    // clamped value is still multi-warp, so thread-count race coverage holds.
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

    // Pack into h_q_qd_u as the gridData canonical layout [q (nq) | qd @ nq | u @ 2*nq]
    // — nq-WIDE slots (stride nq), matching what the integrator host wrapper copies
    // (stride_q = 3*NUM_JOINTS) and what forward_dynamics reads (s_u = &s_q_qd_u[2*nq]).
    // NOTE: u must sit at 2*nq, NOT nq+nv. For a FLOATING base nq != nv, so packing u at
    // nq+nv misaligns it by (nq-nv) and feeds forward_dynamics a garbage torque -> wrong
    // qdd everywhere (fixed-base nq==nv hid this). The qd/u tails (indices [nq+nv, 2*nq)
    // and [2*nq+nv, 3*nq)) are unused padding; zero them for determinism.
    const int input_count = 3 * nq;
    std::vector<T> original(input_count, T(0));
    for (int i = 0; i < nq; ++i) original[i] = h_q[i];
    for (int i = 0; i < nv; ++i) {
        original[nq + i]       = h_qd[i];
        original[2 * nq + i]   = h_u[i];
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

    // External forces (opt-in via GRID_RUNNER_FEXT): read 6*NUM_BODIES body-major
    // local-frame [angular; linear] values (next on stdin after dt) into
    // hd_data->h_f_ext and copy to d_f_ext. The integrator host wrapper reads
    // hd_data->d_f_ext, so every integrator launch below evaluates FD at the
    // f_ext-perturbed operating point. Default (env unset): d_f_ext stays zeroed,
    // byte-identical to the no-fext path.
    if (std::getenv("GRID_RUNNER_FEXT") != nullptr) {
        read_vector(hd_data->h_f_ext, 6 * grid::NUM_BODIES);
        gpuErrchk(cudaMemcpy(hd_data->d_f_ext, hd_data->h_f_ext,
                             6 * grid::NUM_BODIES * sizeof(T), cudaMemcpyHostToDevice));
        print_vector("input_f_ext", hd_data->h_f_ext, 6 * grid::NUM_BODIES);
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

    // Trapezoidal: single-stage. Value = combined-tangent retract
    // q_new = integrate(q, dt*qd + 0.5*dt^2*qdd); v_new = qd + dt*qdd. Gradient now
    // emitted for BOTH fixed- and floating-base (the floating top rows carry the
    // SE(3) dIntegrate chain-rule wiring at the combined tangent w), so it runs the
    // full value+gradient+both path like the other single-stage integrators.
    run_one<T, grid::IntegratorType::TRAPEZOIDAL>("integrator_trapezoidal",
        hd_data, d_robotModel, streams, block_dimms, thread_dimms, original.data(), gravity, dt);

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main(int argc, char **argv) {
    if (argc > 1) {
        int requested = std::atoi(argv[1]);
        if (requested > 0) { g_num_threads = requested; }
    }
    run<float>();
    return 0;
}

// CUDA runner for the DEDICATED f_ext_gradient algorithm family (section A of
// the differentiability extensions plan), distinct from the f_ext-as-parameter
// runner (cuda_equivalence_runner.cu + GRID_RUNNER_FEXT). It drives the emitted
// host wrappers and prints the f_ext-gradient outputs:
//
//   dtau_dfext      = -J^T          (nv x 6*NB)            [A.1]
//   dqdd_dfext      =  M^{-1} J^T   (nv x 6*NB)            [A.2]
//   did_du_dfext_dq = -dJ^T/dq      (nv x 6*NB x nv)       [A.3, both base modes]
//
// All three are q-only (f_ext enters RNEA additively & linearly), so the runner
// reads ONLY q on stdin (NUM_JOINTS values, project layout). The A.3 block is
// emitted for both base modes; it is printed iff GRID_HAS_F_EXT_GRADIENT_DQ.
//
// Output: BEGIN/END framed, column-major print (row + rows*col), matching the
// other cuda_equivalents runners so test_cuda_executable_equivalence._parse_runner_output
// can read it.
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "grid.cuh"

// The A.3 (-dJ^T/dq) host wrapper is emitted for BOTH base modes as the ANALYTIC
// closed form (the free-flyer root is subsumed by the per-column S loop, no FD).
#ifndef GRID_CUDA_FLOATING_BASE
#define GRID_CUDA_FLOATING_BASE 0
#endif
#define GRID_HAS_F_EXT_GRADIENT_DQ 1

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
void run() {
    const dim3 block_dimms(1, 1, 1);
    const dim3 thread_dimms(grid::MAX_PERF_LEVEL_THREADS, 1, 1);

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    const int nq = grid::NUM_JOINTS;   // project q layout length (== NUM_JOINTS)
    const int nv = grid::NUM_VEL;
    const int nb = grid::NUM_BODIES;
    const int out_each = nv * 6 * nb;

    std::vector<T> h_q(nq);
    read_vector(h_q.data(), nq);

    // The f_ext_gradient host wrappers read q from h_q_qd_u (uncompressed) under
    // the [q | qd | u] layout; only q is consumed (the gradients are q-only).
    for (int i = 0; i < nq; ++i) hd_data->h_q_qd_u[i] = h_q[i];
    for (int i = nq; i < 3 * nq; ++i) hd_data->h_q_qd_u[i] = static_cast<T>(0);

    print_matrix_col_major("input_q", h_q.data(), 1, nq);

    // First-order: dtau/dfext (-J^T) and dqdd/dfext (M^-1 J^T).
    grid::f_ext_gradient<T>(hd_data, d_robotModel, 1, block_dimms, thread_dimms, streams);
    print_matrix_col_major("f_ext_gradient_dtau_dfext", hd_data->h_dtau_dfext, nv, 6 * nb);
    print_matrix_col_major("f_ext_gradient_dqdd_dfext", hd_data->h_dqdd_dfext, nv, 6 * nb);

#if GRID_HAS_F_EXT_GRADIENT_DQ
    // A.3: -dJ^T/dq, size nv x 6NB x nv. Printed as a (nv*6NB) x nv matrix with
    // the q-coordinate as the column (matches the kernel layout
    // [ (row v_j) + nv*(6NB col) + nv*6NB*qi ]).
    grid::f_ext_gradient_dq<T>(hd_data, d_robotModel, 1, block_dimms, thread_dimms, streams);
    print_matrix_col_major("f_ext_gradient_did_du_dfext_dq", hd_data->h_f_ext_gradient_dq, out_each, nv);
#else
    (void) out_each;
#endif

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main() {
    run<float>();
    return 0;
}

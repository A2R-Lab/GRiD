// Minimal fixed-base runner for the BRANCHING-skew fixture: exercises ONLY the
// aba / minv / crba host wrappers (the surfaces whose Tier-B dense-S emit had
// the single-joint-per-BFS-level restriction lifted in _aba.py / _minv.py).
//
// It deliberately avoids inverse_dynamics / forward_dynamics so the header can
// be generated as algorithm_list="aba,minv,crba" without pulling in algorithms
// whose Tier-B branching support lands separately. Reads q / qd / u from stdin
// (one whitespace row each) and prints BEGIN/END blocks parsed by the test.
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "grid.cuh"

static int g_num_threads = 64;

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

template <typename T>
void read_vector(T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        double v = 0.0;
        std::cin >> v;
        dst[i] = static_cast<T>(v);
    }
}

template <typename T>
void run() {
    const T gravity = static_cast<T>(-9.81);
    const dim3 block_dimms(1, 1, 1);
    const dim3 thread_dimms(g_num_threads, 1, 1);

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    read_vector(hd_data->h_q, grid::NUM_JOINTS);
    read_vector(&hd_data->h_q_qd[grid::NUM_JOINTS], grid::NUM_JOINTS);
    read_vector(&hd_data->h_q_qd_u[2 * grid::NUM_JOINTS], grid::NUM_JOINTS);

    for (int i = 0; i < grid::NUM_JOINTS; ++i) {
        hd_data->h_q_qd[i] = hd_data->h_q[i];
        hd_data->h_q_qd_u[i] = hd_data->h_q[i];
        hd_data->h_q_qd_u[i + grid::NUM_JOINTS] = hd_data->h_q_qd[i + grid::NUM_JOINTS];
        hd_data->h_qdd[i] = static_cast<T>(0);
    }

    print_vector("input_q", hd_data->h_q, grid::NUM_JOINTS);
    print_vector("input_qd", &hd_data->h_q_qd[grid::NUM_JOINTS], grid::NUM_JOINTS);
    print_vector("input_u", &hd_data->h_q_qd_u[2 * grid::NUM_JOINTS], grid::NUM_JOINTS);

    grid::minv<T, true>(hd_data, d_robot_model, 1, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    print_matrix_col_major("minv", hd_data->h_Minv, grid::NUM_JOINTS, grid::NUM_JOINTS);

    grid::aba<T>(hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    print_vector("aba", hd_data->h_qdd, grid::NUM_JOINTS);

    grid::crba<T, true>(hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    print_matrix_col_major("crba", hd_data->h_M, grid::NUM_JOINTS, grid::NUM_JOINTS);

    grid::close_grid<T>(streams, d_robot_model, hd_data);
}

int main() {
    run<float>();
    return 0;
}

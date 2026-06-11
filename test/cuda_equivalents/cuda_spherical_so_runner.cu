// CUDA equivalence runner for SPHERICAL (ball) joint idsva_so (2nd-order inverse-
// dynamics derivatives), Tier-C. Spherical fixed-base robots route to the
// WORLD-frame inner (the body-frame inner's single-DoF contractions are wrong for
// a 6x3 ball motion subspace); the dispatching `grid::idsva_so<T>` wrapper picks
// the world frame at codegen time for any robot with a spherical joint.
//
// nq != nv: a spherical joint carries a 4-wide unit-quaternion q-block + 3 v-slots,
// so the per-timestep INPUT slot is NQ(=grid::NUM_POS)-wide (the canonical
// 3*NUM_POS pack: q@0, qd@nq, qdd@2*nq). The SO output is the four nv^3 tensors
// [d2tau_dq2 | d2tau_dqd2 | d2tau_dvdq | dM_dq] = SECOND_ORDER_TENSOR_SIZE total.
//
// Drives the dispatching HOST batch wrapper `idsva_so<T, GRID_DATA_ALL>` over a
// B-timestep trajectory of IDENTICAL inputs (the §1e per-timestep nq-stride path
// the bindings use). It emits BOTH the first batch row ("idsva_so", the canonical
// single result) AND every batch row ("idsva_so_batch_k"); the test asserts every
// row matches the WORLD oracle and the rows are mutually identical (§1e
// self-consistency), and sweeps argv[1] thread counts for invariance.
//
// (The standalone `idsva_so_device` inline entry carries a pre-existing latent
// const-qualifier mismatch on s_qdd -- the inner takes non-const s_qdd for the mjx
// accel reframe -- so this runner uses the host kernel path the bindings actually
// use, not that inline entry.)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "grid.cuh"

int g_num_threads = 32;

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
void print_vector(const std::string &name, const T *data, int count) {
    std::cout << "BEGIN " << name << " 1 " << count << "\n";
    std::cout << std::setprecision(10);
    for (int i = 0; i < count; ++i) {
        if (i) std::cout << " ";
        std::cout << static_cast<double>(data[i]);
    }
    std::cout << "\nEND " << name << "\n";
}

template <typename T>
void run() {
    const T gravity = static_cast<T>(-9.81);

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();

    const int nq = grid::NUM_JOINTS;   // == nq for this codegen
    const int nv = grid::NUM_VEL;
    const int so_len = grid::SECOND_ORDER_TENSOR_SIZE;   // 4*nv^3

    // ----- read inputs (q is nq-wide, qd / qdd are nv-wide) -----
    std::vector<T> h_q(nq);
    std::vector<T> h_qd(nv);
    std::vector<T> h_qdd(nv);
    read_vector(h_q.data(), nq);
    read_vector(h_qd.data(), nv);
    read_vector(h_qdd.data(), nv);
    print_vector("input_q", h_q.data(), nq);
    print_vector("input_qd", h_qd.data(), nv);
    print_vector("input_qdd", h_qdd.data(), nv);

    // ----- dispatching HOST batch wrapper over B IDENTICAL timesteps -----
    // The per-timestep slot is NQ-wide for q, qd AND qdd (the canonical 3*NUM_POS
    // pack: q@[0,nq), qd@[nq,2nq), qdd@[2nq,3nq)). grid::idsva_so dispatches to
    // idsva_so_world_frame for spherical (the predicate _idsva_so_use_world_frame).
    const int B = 4;
    grid::gridData<T> *hd_data = grid::init_gridData<T, B>();
    for (int k = 0; k < B; ++k) {
        for (int i = 0; i < nq; ++i)
            hd_data->h_q_qd_u[k * 3 * nq + i] = h_q[i];                   // q @ [0, nq)
        for (int i = 0; i < nq; ++i)
            hd_data->h_q_qd_u[k * 3 * nq + nq + i] =                       // qd @ [nq, 2nq)
                (i < nv) ? h_qd[i] : static_cast<T>(0);
        for (int i = 0; i < nq; ++i)
            hd_data->h_q_qd_u[k * 3 * nq + 2 * nq + i] =                   // qdd @ [2nq, 3nq)
                (i < nv) ? h_qdd[i] : static_cast<T>(0);
    }
    const dim3 block_dimms(1, 1, 1);
    const dim3 thread_dimms(g_num_threads, 1, 1);
    grid::idsva_so<T, grid::GRID_DATA_ALL>(
        hd_data, d_robot_model, gravity, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());

    // Emit the first row as the canonical single result, then every batch row.
    {
        std::vector<T> row0(so_len);
        for (int i = 0; i < so_len; ++i) row0[i] = hd_data->h_idsva_so[i];
        print_vector("idsva_so", row0.data(), so_len);
    }
    for (int k = 0; k < B; ++k) {
        std::vector<T> row(so_len);
        for (int i = 0; i < so_len; ++i) row[i] = hd_data->h_idsva_so[k * so_len + i];
        print_vector("idsva_so_batch_" + std::to_string(k), row.data(), so_len);
    }

    grid::close_grid<T>(streams, d_robot_model, hd_data);
}

int main(int argc, char **argv) {
    if (argc > 1) {
        int requested = std::atoi(argv[1]);
        g_num_threads = requested > 0 ? requested : 0;
    }
    if (g_num_threads <= 0 || g_num_threads > grid::MAX_PERF_LEVEL_THREADS) {
        g_num_threads = grid::MAX_PERF_LEVEL_THREADS;
    }
    const char *equiv_t = std::getenv("GRID_EQUIV_T");
    if (equiv_t != nullptr && std::string(equiv_t) == "double") {
        run<double>();
    } else {
        run<float>();
    }
    return 0;
}

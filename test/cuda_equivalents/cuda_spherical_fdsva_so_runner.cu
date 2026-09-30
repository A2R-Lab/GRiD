// CUDA equivalence runner for SPHERICAL (ball) joint fdsva_so (2nd-order FORWARD-
// dynamics derivatives). Sibling of cuda_spherical_so_runner.cu (idsva_so), exercising
// the fdsva_so host batch wrapper instead. Spherical fixed-base robots route the
// COMPOSED idsva_so inner to the WORLD frame (the body-frame inner's single-DoF S
// contractions are wrong for a 6x3 ball motion subspace); fdsva_so_device picks the
// world inner for any robot with a spherical joint (predicate _fdsva_so_use_world_idsva).
//
// nq != nv: a spherical joint carries a 4-wide unit-quaternion q-block + 3 v-slots,
// so the per-timestep INPUT slot is NQ(=grid::NUM_POS)-wide (the canonical 3*NUM_POS
// pack: q@0, qd@nq, u@2*nq). The fdsva_so output is the four nv^3 tensors
// [daba_dqdq | daba_dvdq | daba_dvdv | daba_dtdq] = SECOND_ORDER_TENSOR_SIZE total
// (the RBDReference.fdsva_so return order; NOT the idsva_so order).
//
// Drives the dispatching HOST batch wrapper `fdsva_so<T, GRID_DATA_ALL>` over a
// B-timestep trajectory of IDENTICAL inputs (the §1e per-timestep nq-stride path the
// bindings use). It emits BOTH the first batch row ("fdsva_so", the canonical single
// result) AND every batch row ("fdsva_so_batch_k"); the test asserts every row matches
// the WORLD-routed oracle and the rows are mutually identical (§1e self-consistency),
// and sweeps argv[1] thread counts for invariance.
//
// A4 (surgical cold spill): GRID_CUDA_TARGET_SHARED_MEM_BYTES forces select_shared_
// tier_3way to pick the idsva_cold rung even on this small robot, so the runner can
// exercise the COLD_IN_SMEM=false world-inner path (cold trio Xdown/v_w/a_w spilled to
// the SO-temp d_workspace region) without a big-robot compile.
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

    // ----- read inputs (q is nq-wide, qd / u are nv-wide) -----
    std::vector<T> h_q(nq);
    std::vector<T> h_qd(nv);
    std::vector<T> h_u(nv);
    read_vector(h_q.data(), nq);
    read_vector(h_qd.data(), nv);
    read_vector(h_u.data(), nv);
    print_vector("input_q", h_q.data(), nq);
    print_vector("input_qd", h_qd.data(), nv);
    print_vector("input_u", h_u.data(), nv);

    // ----- dispatching HOST batch wrapper over B IDENTICAL timesteps -----
    // The per-timestep slot is NQ-wide for q, qd AND u (the canonical 3*NUM_POS pack:
    // q@[0,nq), qd@[nq,2nq), u@[2nq,3nq)). grid::fdsva_so composes idsva_so_world_frame
    // for spherical (the predicate _fdsva_so_use_world_idsva).
    const int B = 4;
    grid::gridData<T> *hd_data = grid::init_gridData<T, B>();
    for (int k = 0; k < B; ++k) {
        for (int i = 0; i < nq; ++i)
            hd_data->h_q_qd_u[k * 3 * nq + i] = h_q[i];                   // q @ [0, nq)
        for (int i = 0; i < nq; ++i)
            hd_data->h_q_qd_u[k * 3 * nq + nq + i] =                       // qd @ [nq, 2nq)
                (i < nv) ? h_qd[i] : static_cast<T>(0);
        for (int i = 0; i < nq; ++i)
            hd_data->h_q_qd_u[k * 3 * nq + 2 * nq + i] =                   // u @ [2nq, 3nq)
                (i < nv) ? h_u[i] : static_cast<T>(0);
    }
    const dim3 block_dimms(1, 1, 1);
    const dim3 thread_dimms(g_num_threads, 1, 1);
    grid::fdsva_so<T, grid::GRID_DATA_ALL>(
        hd_data, d_robot_model, gravity, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());

    // Emit the first row as the canonical single result, then every batch row.
    {
        std::vector<T> row0(so_len);
        for (int i = 0; i < so_len; ++i) row0[i] = hd_data->h_df2[i];
        print_vector("fdsva_so", row0.data(), so_len);
    }
    for (int k = 0; k < B; ++k) {
        std::vector<T> row(so_len);
        for (int i = 0; i < so_len; ++i) row[i] = hd_data->h_df2[k * so_len + i];
        print_vector("fdsva_so_batch_" + std::to_string(k), row.data(), so_len);
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

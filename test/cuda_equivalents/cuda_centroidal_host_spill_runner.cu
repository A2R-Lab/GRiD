// Centroidal HOST-WRAPPER CUDA runner — validates the com/ccrba/energy KERNEL spill
// path (DE-GATE #2): centroidal_inner<T,false> with the Jw band carved out of s_temp
// into a separate tier-routed buffer (smem tail at the J-in-smem tier, d_workspace at
// the J-spilled tier). The existing centroidal smoke runner drives the *_device fns
// (which keep J in smem, centroidal_inner<T,true>); this one drives the host wrappers
// grid::com / grid::ccrba / grid::energy end-to-end so the previously-dead <T,false>
// inner branch + the energy KE reach-back no-J offsets + the d_workspace repoint are
// exercised. Build with a forced-low GRID_CUDA_TARGET_SHARED_MEM_BYTES at CODEGEN so
// the chosen rung is the J-spilled one even on a small robot.
//
// Input on stdin: GRID_BATCH consecutive [q (NUM_POS); qd (NUM_VEL)] vectors.
// Output: per-slot COM (3), JCOM (3 x NV col-major), CCRBA_A (6 x NV col-major),
//         CCRBA_H (6), ENERGY (3 = [KE, PE, mechanical]).
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "grid.cuh"

#ifndef GRID_BATCH
#define GRID_BATCH 2
#endif

template <typename T>
void read_vector(T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        double value;
        if (!(std::cin >> value)) { std::cerr << "read fail " << i << "\n"; std::exit(2); }
        dst[i] = static_cast<T>(value);
    }
}

template <typename T>
void print_vec(const std::string &name, int slot, const T *data, int n) {
    const std::string tag = name + std::to_string(slot);
    std::cout << "BEGIN " << tag << " 1 " << n << "\n" << std::setprecision(10);
    for (int i = 0; i < n; ++i) { if (i) std::cout << " "; std::cout << static_cast<double>(data[i]); }
    std::cout << "\nEND " << tag << "\n";
}

template <typename T>
void run() {
    const dim3 block_dimms(1, 1, 1);
    int nthreads = grid::MAX_PERF_LEVEL_THREADS;
    if (const char *e = std::getenv("GRID_NTHREADS")) { int v = std::atoi(e); if (v > 0) nthreads = v; }
    const dim3 thread_dimms(nthreads, 1, 1);
    constexpr int NQ = grid::NUM_POS;
    constexpr int NV = grid::NUM_VEL;
    constexpr int NJ = grid::NUM_JOINTS;
    constexpr int B  = GRID_BATCH;
    const T gravity = static_cast<T>(-9.81);

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, B>();

    // Uncompressed host wrappers read [q;qd] from h_q_qd_u at stride 3*NUM_JOINTS;
    // the kernel loads 2*NQ floats then splits s_q=[0:NQ], s_qd=[NQ:NQ+NV].
    const int stride = 3 * NJ;
    for (int k = 0; k < B; ++k) {
        T *slot = &hd_data->h_q_qd_u[k * stride];
        for (int i = 0; i < stride; ++i) slot[i] = static_cast<T>(0);
        read_vector(slot, NQ);            // q  (ccrba/energy read [q;qd] from h_q_qd_u)
        read_vector(slot + NQ, NV);       // qd
        // com is q-only and reads from h_q at stride NUM_JOINTS (== NUM_POS here).
        for (int i = 0; i < NQ; ++i) hd_data->h_q[k * NJ + i] = slot[i];
    }

    grid::com<T>(hd_data, d_robotModel, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k) {
        const T *c = &hd_data->h_com[k * (3 + 3 * NV)];
        print_vec("COM", k, c, 3);
        print_vec("JCOM", k, c + 3, 3 * NV);
    }

    grid::ccrba<T>(hd_data, d_robotModel, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k) {
        const T *a = &hd_data->h_ccrba[k * (6 * NV + 6)];
        print_vec("CCRBA_A", k, a, 6 * NV);
        print_vec("CCRBA_H", k, a + 6 * NV, 6);
    }

    grid::energy<T>(hd_data, d_robotModel, gravity, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k)
        print_vec("ENERGY", k, &hd_data->h_energy[k * 3], 3);

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main() {
    run<float>();
    return 0;
}

// Workspace-chunk bit-identity runner — gate for the chunked (two-phase) batch
// SO seam (GRID_WORKSPACE_CHUNK). The chunk-aware host wrappers (idsva_so body /
// world, fdsva_so, f_ext_gradient_dq) sweep the batch in C-sized launches that
// reuse a chunk-sized workspace arena while outputs stay full-N on device. Since
// timesteps are independent and buffer addresses never enter the arithmetic, a
// chunked run must produce outputs BIT-IDENTICAL to an unchunked run.
//
// The pytest compiles this file TWICE from the SAME chunk-capable header — once
// with -DGRID_WORKSPACE_CHUNK=<C> and once with -DGRID_WORKSPACE_CHUNK=0 — using
// the exact bench flag composition (GRID_ALLOC_GATE + per-algo GRID_ALLOC_*), and
// asserts the two stdouts are equal. Outputs are printed as raw float bit
// patterns (hex), so the comparison is exact, not tolerance-based.
//
// Inputs are generated in-process by a fixed LCG (identical in both arms; no
// stdin). For a floating base the root quaternion block q[3..6] is normalized so
// the states are valid; everything downstream is deterministic either way.
#include <cmath>
#include <cstdio>
#include <cstring>

#include "grid.cuh"

#ifndef GRID_BATCH
#define GRID_BATCH 32
#endif

using T = float;

static void dump_bits(const char *name, const T *data, size_t count) {
    printf("BEGIN %s %zu\n", name, count);
    for (size_t i = 0; i < count; ++i) {
        unsigned bits;
        memcpy(&bits, &data[i], sizeof(bits));
        printf("%08x%c", bits, ((i & 15) == 15 || i + 1 == count) ? '\n' : ' ');
    }
    printf("END %s\n", name);
}

int main() {
    const dim3 block_dimms(1, 1, 1);
    const dim3 thread_dimms(grid::MAX_PERF_LEVEL_THREADS, 1, 1);
    constexpr int B = GRID_BATCH;
    constexpr int NQ = grid::NUM_POS;
    const T gravity = static_cast<T>(9.81);

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, B>();

    // fixed LCG q/qd/u in [-1, 1]; DISTINCT per batch slot so a chunk-boundary
    // off-by-one (wrong slot read/written) cannot alias into a bit-identical pass
    const int stride = 3 * grid::NUM_JOINTS;
    unsigned s = 42u;
    for (int k = 0; k < B; ++k) {
        for (int i = 0; i < stride; ++i) {
            s = s * 1664525u + 1013904223u;
            hd_data->h_q_qd_u[k * stride + i] = static_cast<T>(((s >> 8) & 0xFFFF) / 65535.0f * 2.0f - 1.0f);
        }
        if (NQ != grid::NUM_VEL) {  // floating base: normalize the root quaternion q[3..6]
            T *q = &hd_data->h_q_qd_u[k * stride];
            T n = std::sqrt(q[3]*q[3] + q[4]*q[4] + q[5]*q[5] + q[6]*q[6]);
            for (int i = 3; i < 7; ++i) q[i] /= n;
        }
        for (int i = 0; i < NQ; ++i)
            hd_data->h_q[k * grid::NUM_JOINTS + i] = hd_data->h_q_qd_u[k * stride + i];
    }

    // idsva_so FIRST: fills hd_data->d_idsva_so on device, which fdsva_so then
    // consumes as a (chunk-offset) INPUT — so the fdsva_so leg also proves the
    // input-side offsetting.
    grid::idsva_so<T>(hd_data, d_robotModel, gravity, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    dump_bits("IDSVA_SO", hd_data->h_idsva_so, (size_t)grid::SECOND_ORDER_TENSOR_SIZE * B);

    grid::fdsva_so<T>(hd_data, d_robotModel, gravity, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    dump_bits("FDSVA_SO", hd_data->h_df2, (size_t)grid::SECOND_ORDER_TENSOR_SIZE * B);

    grid::f_ext_gradient_dq<T>(hd_data, d_robotModel, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    dump_bits("F_EXT_GRADIENT_DQ", hd_data->h_f_ext_gradient_dq,
              (size_t)grid::NUM_VEL * 6 * grid::NUM_BODIES * grid::NUM_VEL * B);

    grid::close_grid<T>(streams, d_robotModel, hd_data);
    return 0;
}

// Workspace-slots bit-identity runner — gate for the runtime workspace-slot
// seam. init_gridData auto-fits gridData.workspace_timestep_slots to remaining
// device memory (GRID_WORKSPACE_TIMESTEP_SLOTS env overrides), kernels index the
// workspace arena per-BLOCK (grid_workspace_slot()), and every workspace-using
// host wrapper clamps its launch grid to the slot count. Since timesteps are
// independent and per-timestep computation never depends on which block (or how
// many blocks) executes it, a slot-clamped run must produce outputs BIT-IDENTICAL
// to an unclamped one.
//
// The pytest compiles this file ONCE and runs it three times — env unset
// (slots == GRID_BATCH, clamp is a no-op), GRID_WORKSPACE_TIMESTEP_SLOTS=3
// (grid clamped 32 -> 3 blocks; 3 deliberately does not divide the batch), and
// =1 (fully serialized single-slot extreme) — asserting all three stdouts are
// equal. Outputs are raw float bit patterns (hex), so the comparison is exact.
// The exe launches with a MULTI-block grid (one block per timestep unclamped),
// so the clamp path genuinely engages under the forced-slot arms.
//
// Inputs are generated in-process by a fixed LCG (identical in all arms; no
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
    constexpr int B = GRID_BATCH;
    // multi-block grid: one block per timestep when unclamped, so the forced-slot
    // arms exercise the wrapper's grid clamp + per-block slot indexing for real
    const dim3 block_dimms(B, 1, 1);
    const dim3 thread_dimms(grid::MAX_PERF_LEVEL_THREADS, 1, 1);
    constexpr int NQ = grid::NUM_POS;
    const T gravity = static_cast<T>(9.81);

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, B>();
    // stderr on purpose: the arms' slot counts DIFFER, and the test compares stdout
    fprintf(stderr, "workspace_timestep_slots=%d\n", hd_data->workspace_timestep_slots);

    // fixed LCG q/qd/u in [-1, 1]; DISTINCT per batch slot so a slot-mapping
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
    // consumes as a full-N INPUT — outputs never shrink under slot clamping.
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

// =============================================================================
// GRiD CUDA usage example: a HEAVY (second-order) kernel via the _host surface.
//
// Companion to inverse_dynamics_kernel_example.cu. That one shows the low-level
// `_inner`/`_device` surfaces you write your OWN kernel against. This one shows
// the OTHER end of the layering: for a heavy algorithm you just want to *call*,
// reach for the generated `_host` wrapper -- it owns the gridData bookkeeping,
// the host<->device memcpy, the L2-pinned d_workspace, and the launch. This is
// the recommended way to run idsva_so (the analytical second-order inverse
// dynamics: d2tau/dq, d2tau/dqd, d2tau/dvdq, dM/dq).
//
// Generate the header (note the body-frame token for fixed base):
//   GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
//       algorithm_list=["idsva_so_body_frame"], output_path="grid.cuh")
//
// The flat output tensor grid::SECOND_ORDER_TENSOR_SIZE (= 4*NV^3) packs the four
// NVxNVxNV blocks back-to-back in this order (C/row-major within each block):
//   [ d2tau_dq | d2tau_dqd | d2tau_dvdq | dM_dq ]
// validate_so.py reconstructs exactly this fold from RBDReference.
// =============================================================================
#include <cstdio>
#include <vector>

#include "grid_so.cuh"   // generated with algorithm_list=["idsva_so_body_frame"]

template <typename T>
static void run() {
    const T gravity = static_cast<T>(-9.81);
    const dim3 block_dimms(1, 1, 1);
    int threads = 32;
    if (threads > grid::MAX_PERF_LEVEL_THREADS) threads = grid::MAX_PERF_LEVEL_THREADS;
    const dim3 thread_dimms(threads, 1, 1);

    // The _host lifecycle: streams + model (compile-time constants) + a gridData
    // that bundles every host/device buffer this algorithm reads or writes,
    // including the L2-pinned d_workspace the second-order kernel spills to.
    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    // Fill the HOST inputs (q, qd, qdd) -- the _host wrapper copies them to the
    // device for us. Deterministic, matching examples/cuda/validate_so.py.
    const int np = grid::NUM_POS, nv = grid::NUM_VEL;
    for (int i = 0; i < np; ++i) hd_data->h_q_qd_u[i] = static_cast<T>(0.1 * (i + 1));
    for (int i = 0; i < nv; ++i) {
        hd_data->h_q_qd_u[np + i]      = static_cast<T>(0.01 * (i + 1));   // qd
        hd_data->h_q_qd_u[np + nv + i] = static_cast<T>(0.02 * (i + 1));   // qdd
    }

    // One call. The wrapper memcpys inputs in, sizes + reserves the dynamic
    // shared memory, manages d_workspace, launches idsva_so_body_frame_kernel
    // (one block here), and memcpys the result into hd_data->h_idsva_so.
    grid::idsva_so_body_frame<T>(
        hd_data, d_robotModel, gravity, /*num_timesteps=*/1,
        block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    const int n = grid::SECOND_ORDER_TENSOR_SIZE;
    printf("BEGIN idsva_so_body_frame %d\n", n);
    for (int i = 0; i < n; ++i)
        printf("%.10g%s", static_cast<double>(hd_data->h_idsva_so[i]),
               (i + 1) % 8 == 0 || i + 1 == n ? "\n" : " ");
    printf("END idsva_so_body_frame\n");

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main() {
    run<float>();
    return 0;
}

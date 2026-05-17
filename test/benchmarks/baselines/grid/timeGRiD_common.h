/***
 * Shared setup for the split GRiD timing binaries.
 *
 * Why split?
 * ----------
 * `_single_timing` kernels need `__noinline__ grid_licm_barrier` (calls
 * across a separate-compilation boundary so nvcc cannot LICM-elide the rep
 * loop), which only works when the TU is compiled with `-rdc=true`. But
 * `-rdc=true` disables aggressive cross-function inlining for ALL kernels
 * in the TU, including the batch ones — which we measured at up to 3×
 * slowdown on chain-heavy algos (iiwa14 ee_pose_gradient N=16 was the
 * worst offender). Splitting batch vs single timings into separate TUs
 * lets each compile with its own optimal flags:
 *
 *   timeGRiD_single.cu  →  -rdc=true   (anti-LICM correctness)
 *   timeGRiD_batch.cu   →  no -rdc     (aggressive inlining, fast batch)
 *
 * This header carries only the shared host-side scaffolding (init / load
 * inputs / warmup / close). Each .cu defines its own `measure_*` template
 * helpers, its own `test<>()` dispatcher, and its own `main()`.
 ***/
#pragma once

#ifndef GRID_HEADER_FILE
#include "../../../../grid.cuh"
#else
#include GRID_HEADER_FILE
#endif
#include "../util/experiment_helpers.h"

#define GRAVITY 9.81

inline dim3 grid_timing_dimms() { return dim3(grid::SUGGESTED_THREADS, 1, 1); }

// True when the kernel's requested dynamic shared memory exceeds the device's
// per-block cap (i.e. not even cudaFuncSetAttribute could open enough). Used
// by measure_* helpers to skip kernels that can't possibly run on this device
// (e.g. fdsva_so on g1_floating wants ~197 KB but sm_120 caps at ~100 KB).
inline bool grid_kernel_fits_device(size_t requested_bytes) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    int max_bytes = 0;
    if (cudaDeviceGetAttribute(&max_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess) return false;
    return requested_bytes <= static_cast<size_t>(max_bytes);
}

// ---------------------------------------------------------------------------
// run_all_tests<>: shared init / load / warmup / close skeleton.
// `do_timings` is the per-TU dispatcher (single or batch) — passes
// streams + device pointers + max timesteps so the dispatcher can pick
// whichever NUM_TIMESTEPS values it cares about.
// ---------------------------------------------------------------------------
template <typename T, int MAX_TIMESTEPS, typename DispatcherFn>
__host__ void run_all_tests(bool floating_base, DispatcherFn do_timings){
    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T,MAX_TIMESTEPS>();

    // load q,qd,u — codegen's NUM_JOINTS already accounts for floating-base position dim;
    // strides match init_gridData allocs (NUM_JOINTS, 2*NUM_JOINTS, 3*NUM_JOINTS). The
    // floating_base flag is informational here; do not inflate strides on top of it.
    (void)floating_base;
    for(int k = 0; k < MAX_TIMESTEPS; k++){
        for (int ind = 0; ind < grid::NUM_JOINTS; ind++) {
            T val = getRand<double>();
            hd_data->h_q_qd_u[k*(3*grid::NUM_JOINTS) + ind] = val;
            hd_data->h_q_qd[k*(2*grid::NUM_JOINTS) + ind] = val;
            hd_data->h_q[k*(grid::NUM_JOINTS) + ind] = val;
        }
        for(int ind = 0; ind < grid::NUM_JOINTS; ind++){
            T val2 = getRand<double>(); T val3 = getRand<double>();
            hd_data->h_q_qd_u[k*(3*grid::NUM_JOINTS) + grid::NUM_JOINTS + ind] = val2;
            hd_data->h_q_qd_u[k*(3*grid::NUM_JOINTS) + 2*grid::NUM_JOINTS + ind] = val3;
            hd_data->h_q_qd[k*(2*grid::NUM_JOINTS) + grid::NUM_JOINTS + ind] = val2;
        }
    }
    gpuErrchk(cudaMemcpy(hd_data->d_q_qd_u,hd_data->h_q_qd_u,3*grid::NUM_JOINTS*MAX_TIMESTEPS*sizeof(T),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(hd_data->d_q_qd,hd_data->h_q_qd,2*grid::NUM_JOINTS*MAX_TIMESTEPS*sizeof(T),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(hd_data->d_q,hd_data->h_q,grid::NUM_JOINTS*MAX_TIMESTEPS*sizeof(T),cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    // GPU warmup: run several ID batches and discard before timing
    dim3 dimms = grid_timing_dimms();
    for(int w = 0; w < 5; w++){
        grid::inverse_dynamics<T,false,true>(hd_data,d_robotModel,GRAVITY,MAX_TIMESTEPS,dim3(MAX_TIMESTEPS,1,1),dimms,streams);
    }
    gpuErrchk(cudaDeviceSynchronize());

    do_timings(streams, d_robotModel, hd_data);

    grid::close_grid<T>(streams,d_robotModel,hd_data);
}

inline bool parse_floating_base_arg(int argc, const char **argv){
    bool floating_base = false;
    if (argc > 1 && argv[1][0] == 'T') {floating_base = true; printf("Floating Base = True\n");}
    else {printf("Floating Base = False\n");}
    return floating_base;
}

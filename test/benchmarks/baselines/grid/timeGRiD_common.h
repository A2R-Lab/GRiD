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
 * This header carries the shared host-side scaffolding (init / load
 * inputs / warmup / close) and the `measure_batch_pair` template loop
 * used by every batch wrapper. Each .cu file defines its own
 * `measure_*_entry` host function, and (for dispatcher mains) its own
 * `int main()`.
 *
 * Per-algo TU layout (P6-7b, default):
 *   timeGRiD_single_<algo>.cu  → measure_<algo>_single_entry()      (compiled with -rdc=true)
 *   timeGRiD_batch_<algo>.cu   → measure_<algo>_batch_entry()        (compiled WITHOUT -rdc=true)
 *   timeGRiD_single_main.cu    → main() that calls each *_single_entry
 *   timeGRiD_batch_main.cu     → main() that calls each *_batch_entry, looped over N
 *
 * Monolithic fallback (--no-per-algo-tus):
 *   timeGRiD_single.cu (this entire single-call binary in one TU)
 *   timeGRiD_batch.cu  (this entire batch binary in one TU)
 ***/
#pragma once

#ifndef GRID_HEADER_FILE
#include "../../../../grid.cuh"
#else
#include GRID_HEADER_FILE
#endif
#include "../util/experiment_helpers.h"

#define GRAVITY 9.81

inline dim3 grid_timing_dimms() { return dim3(grid::MAX_PERF_LEVEL_THREADS, 1, 1); }

// ---------------------------------------------------------------------------
// Shared timing loop for one (with-memory, compute-only) batch pair. Takes
// the invocations as lambdas — a function-style macro chokes on the commas
// inside `dim3(N,1,1)`.
//
// Hoisted into the common header so the per-algo batch TUs
// (timeGRiD_batch_<algo>.cu) and the monolithic timeGRiD_batch.cu can both
// share it. Must stay a template so each TU instantiates it locally
// (no .o symbol leakage across TUs).
// ---------------------------------------------------------------------------
template <int TEST_ITERS, typename WMFn, typename COFn>
__host__ void measure_batch_pair(const char *label, int NUM_TIMESTEPS, WMFn with_mem, COFn compute_only){
    struct timespec start, end;
    std::vector<double> times;
    times.reserve(TEST_ITERS);
    for(int iter = 0; iter < TEST_ITERS; iter++){
        clock_gettime(CLOCK_MONOTONIC,&start);
        with_mem();
        clock_gettime(CLOCK_MONOTONIC,&end);
        times.push_back(time_delta_us_timespec(start,end));
    }
    printf("[N:%d]: %s WITH MEMORY: ",NUM_TIMESTEPS,label); printStats(&times); times.clear();
    for(int iter = 0; iter < TEST_ITERS; iter++){
        clock_gettime(CLOCK_MONOTONIC,&start);
        compute_only();
        clock_gettime(CLOCK_MONOTONIC,&end);
        times.push_back(time_delta_us_timespec(start,end));
    }
    printf("[N:%d]: %s COMPUTE ONLY: ",NUM_TIMESTEPS,label); printStats(&times); times.clear();
}

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

// One-liner skip used at the top of each measure_<algo>_{single,batch}.
// Prints a parseable "<LABEL> SKIPPED" line (matches timing_parser.py null
// handling) and returns from the enclosing function when the kernel's
// requested smem exceeds the device cap. The smem-bytes constexpr name is
// passed as TOK so callers stay one line.
#define GRID_SKIP_IF_KERNEL_TOO_BIG(LABEL, TOK)                                          \
    do {                                                                                 \
        if (!grid_kernel_fits_device(grid::TOK<T>())) {                                  \
            printf("Single Call " LABEL " SKIPPED (kernel needs %zu bytes shared mem, " \
                   "exceeds device cap)\n", grid::TOK<T>());                             \
            return;                                                                      \
        }                                                                                \
    } while (0)
#define GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG(LABEL, N, TOK)                                 \
    do {                                                                                 \
        if (!grid_kernel_fits_device(grid::TOK<T>())) {                                  \
            printf("[N:%d]: " LABEL " SKIPPED (kernel needs %zu bytes shared mem, "      \
                   "exceeds device cap)\n", (N), grid::TOK<T>());                        \
            return;                                                                      \
        }                                                                                \
    } while (0)

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

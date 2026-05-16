/***
 * Batch timing binary. Compiled WITHOUT `-rdc=true` so nvcc can inline
 * the inner SIMT helpers (`dot_prod`, `grid_xhom_or_dxhom_ptr`, etc.)
 * aggressively across function boundaries. This recovers the ~3× perf
 * we saw HEAD lose vs pre_glass on chain-heavy algos at small batch
 * sizes (iiwa14 ee_pose_gradient N=16, etc.).
 *
 * The `_single_timing` kernels are intentionally NOT instantiated here.
 * Without `-rdc=true` their anti-LICM `__noinline__ grid_licm_barrier`
 * would still link (as a regular non-relocatable inline) but the LICM
 * pass would happily elide the rep loop — wrong timings. Single-call
 * lives in timeGRiD_single.cu.
 *
 * Output format matches the original timeGRiD.cu so test/benchmarks/
 * timing_parser.py picks up "[N:X]: <ALGO> COMPUTE ONLY: ..." lines
 * unchanged.
 ***/
#include "timeGRiD_common.h"

// ---------------------------------------------------------------------------
// Shared timing loop for one (with-memory, compute-only) batch pair. Takes
// the invocations as lambdas — a function-style macro chokes on the commas
// inside `dim3(N,1,1)`.
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

template <typename T, int TEST_ITERS>
__host__ void measure_id_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("ID", N,
        [&]{ grid::inverse_dynamics<T,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::inverse_dynamics_compute_only<T,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_minv_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("Minv", N,
        [&]{ grid::direct_minv<T,true>(d,m,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::direct_minv_compute_only<T,true>(d,m,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_fd_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("FD", N,
        [&]{ grid::forward_dynamics<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::forward_dynamics_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_aba_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("ABA", N,
        [&]{ grid::aba<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::aba_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_crba_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("CRBA", N,
        [&]{ grid::crba<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::crba_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_id_du_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("ID_DU", N,
        [&]{ grid::inverse_dynamics_gradient<T,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::inverse_dynamics_gradient_compute_only<T,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_fd_du_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("FD_DU", N,
        [&]{ grid::forward_dynamics_gradient<T,false>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::forward_dynamics_gradient_compute_only<T,false>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("EE_POSE", N,
        [&]{ grid::end_effector_pose<T>(d,m,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::end_effector_pose_compute_only<T>(d,m,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_gradient_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("EE_POSE_GRADIENT", N,
        [&]{ grid::end_effector_pose_gradient<T>(d,m,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::end_effector_pose_gradient_compute_only<T>(d,m,N,dim3(N,1,1),dimms); });
}
#if GRID_HAS_IDSVA_SO
template <typename T, int TEST_ITERS>
__host__ void measure_idsva_so_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("IDSVA_SO", N,
        [&]{ grid::idsva_so_host<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::idsva_so_host_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_IDSVA_SO_SPATIAL_V2
template <typename T, int TEST_ITERS>
__host__ void measure_idsva_so_sv2_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("IDSVA_SO_SV2", N,
        [&]{ grid::idsva_so_spatial_v2_host<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::idsva_so_spatial_v2_host_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_FDSVA_SO
template <typename T, int TEST_ITERS>
__host__ void measure_fdsva_so_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("FDSVA_SO", N,
        [&]{ grid::fdsva_so<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::fdsva_so_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif

template <typename T, int TEST_ITERS>
__host__ void run_batch_at(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    measure_id_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_minv_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_fd_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_aba_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_crba_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_id_du_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_fd_du_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_ee_pose_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_ee_pose_gradient_batch<T,TEST_ITERS>(N, streams, m, d);
#if GRID_HAS_IDSVA_SO
    measure_idsva_so_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_IDSVA_SO_SPATIAL_V2
    measure_idsva_so_sv2_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_FDSVA_SO
    measure_fdsva_so_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
}

template <typename T, int TEST_ITERS>
__host__ void run_batch_timings(cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
#if !TEST_FOR_EQUIVALENCE
    run_batch_at<T,TEST_ITERS>(16, streams, m, d);
    run_batch_at<T,TEST_ITERS>(32, streams, m, d);
    run_batch_at<T,TEST_ITERS>(64, streams, m, d);
    run_batch_at<T,TEST_ITERS>(128, streams, m, d);
    run_batch_at<T,TEST_ITERS>(256, streams, m, d);
#endif
}

int main(int argc, const char **argv){
    bool floating_base = parse_floating_base_arg(argc, argv);
    run_all_tests<float, 256>(floating_base, [&](cudaStream_t *streams, grid::robotModel<float> *m, grid::gridData<float> *d){
        run_batch_timings<float, TEST_ITERS_GLOBAL>(streams, m, d);
    });
    return 0;
}

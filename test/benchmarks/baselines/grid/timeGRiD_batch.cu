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

// measure_batch_pair() now lives in timeGRiD_common.h so the per-algo TU
// split (timeGRiD_batch_<algo>.cu) can share it.

template <typename T, int TEST_ITERS>
__host__ void measure_id_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("ID", N, ID_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("ID", N,
        [&]{ grid::inverse_dynamics<T,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::inverse_dynamics_compute_only<T,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_minv_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("Minv", N, MINV_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("Minv", N,
        [&]{ grid::direct_minv<T,true>(d,m,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::direct_minv_compute_only<T,true>(d,m,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_fd_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("FD", N, FD_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("FD", N,
        [&]{ grid::forward_dynamics<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::forward_dynamics_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_aba_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("ABA", N, ABA_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("ABA", N,
        [&]{ grid::aba<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::aba_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_crba_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("CRBA", N, CRBA_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("CRBA", N,
        [&]{ grid::crba<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::crba_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_id_du_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("ID_DU", N, ID_DU_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("ID_DU", N,
        [&]{ grid::inverse_dynamics_gradient<T,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::inverse_dynamics_gradient_compute_only<T,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_fd_du_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("FD_DU", N, FD_DU_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("FD_DU", N,
        [&]{ grid::forward_dynamics_gradient<T,false>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::forward_dynamics_gradient_compute_only<T,false>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("EE_POSE", N, EE_POS_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("EE_POSE", N,
        [&]{ grid::end_effector_pose<T>(d,m,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::end_effector_pose_compute_only<T>(d,m,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_gradient_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("EE_POSE_GRADIENT", N, DEE_POS_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("EE_POSE_GRADIENT", N,
        [&]{ grid::end_effector_pose_gradient<T>(d,m,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::end_effector_pose_gradient_compute_only<T>(d,m,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_hessian_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("EE_POSE_HESSIAN", N, D2EE_POS_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("EE_POSE_HESSIAN", N,
        [&]{ grid::end_effector_pose_gradient_hessian<T>(d,m,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::end_effector_pose_gradient_hessian_compute_only<T>(d,m,N,dim3(N,1,1),dimms); });
}
#if GRID_HAS_IDSVA_SO_BODY_FRAME
template <typename T, int TEST_ITERS>
__host__ void measure_idsva_so_body_frame_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("IDSVA_SO_BODY_FRAME", N, IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("IDSVA_SO_BODY_FRAME", N,
        [&]{ grid::idsva_so_body_frame_host<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::idsva_so_body_frame_host_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_IDSVA_SO_WORLD_FRAME
template <typename T, int TEST_ITERS>
__host__ void measure_idsva_so_world_frame_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("IDSVA_SO_WORLD_FRAME", N, IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("IDSVA_SO_WORLD_FRAME", N,
        [&]{ grid::idsva_so_world_frame_host<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::idsva_so_world_frame_host_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_FDSVA_SO
template <typename T, int TEST_ITERS>
__host__ void measure_fdsva_so_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("FDSVA_SO", N, FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("FDSVA_SO", N,
        [&]{ grid::fdsva_so<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::fdsva_so_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_IDSVA_SO
template <typename T, int TEST_ITERS>
__host__ void measure_idsva_so_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("IDSVA_SO", N, IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("IDSVA_SO", N,
        [&]{ grid::idsva_so<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::idsva_so_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif
// Time integrators (host signatures take an extra dt; IntegratorType defaults to EULER).
#if GRID_HAS_INTEGRATOR
template <typename T, int TEST_ITERS>
__host__ void measure_integrator_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("INTEGRATOR", N, INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("INTEGRATOR", N,
        [&]{ grid::integrator<T>(d,m,GRAVITY,static_cast<T>(0.01),N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::integrator_compute_only<T>(d,m,GRAVITY,static_cast<T>(0.01),N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_INTEGRATOR_GRADIENT
template <typename T, int TEST_ITERS>
__host__ void measure_integrator_gradient_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("INTEGRATOR_GRADIENT", N, INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("INTEGRATOR_GRADIENT", N,
        [&]{ grid::integrator_gradient<T>(d,m,GRAVITY,static_cast<T>(0.01),N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::integrator_gradient_compute_only<T>(d,m,GRAVITY,static_cast<T>(0.01),N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_integrator_with_gradient_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("INTEGRATOR_WITH_GRADIENT", N, INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("INTEGRATOR_WITH_GRADIENT", N,
        [&]{ grid::integrator_gradient_with_x_kp1<T>(d,m,GRAVITY,static_cast<T>(0.01),N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::integrator_gradient_with_x_kp1_compute_only<T>(d,m,GRAVITY,static_cast<T>(0.01),N,dim3(N,1,1),dimms); });
}
#endif

template <typename T, int TEST_ITERS>
__host__ void run_batch_at(bool floating_base, int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
#if defined(GRID_BENCH_D2EE_ONLY) && GRID_BENCH_D2EE_ONLY
    // d2ee-only fast bench: only measure ee_pose_hessian to keep compile+run short.
    (void)floating_base;
    measure_ee_pose_hessian_batch<T,TEST_ITERS>(N, streams, m, d);
#else
    measure_id_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_minv_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_fd_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_aba_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_crba_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_id_du_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_fd_du_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_ee_pose_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_ee_pose_gradient_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_ee_pose_hessian_batch<T,TEST_ITERS>(N, streams, m, d);
#if GRID_HAS_IDSVA_SO
    measure_idsva_so_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_IDSVA_SO_BODY_FRAME
    measure_idsva_so_body_frame_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_IDSVA_SO_WORLD_FRAME
    measure_idsva_so_world_frame_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_FDSVA_SO
    measure_fdsva_so_batch<T,TEST_ITERS>(N, streams, m, d);
    (void)floating_base;
#endif
#if GRID_HAS_INTEGRATOR
    measure_integrator_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_INTEGRATOR_GRADIENT
    measure_integrator_gradient_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_integrator_with_gradient_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#endif
}

template <typename T, int TEST_ITERS>
__host__ void run_batch_timings(bool floating_base, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
#if !TEST_FOR_EQUIVALENCE
    run_batch_at<T,TEST_ITERS>(floating_base, 16, streams, m, d);
    run_batch_at<T,TEST_ITERS>(floating_base, 32, streams, m, d);
    run_batch_at<T,TEST_ITERS>(floating_base, 64, streams, m, d);
    run_batch_at<T,TEST_ITERS>(floating_base, 128, streams, m, d);
    run_batch_at<T,TEST_ITERS>(floating_base, 256, streams, m, d);
#endif
}

int main(int argc, const char **argv){
    bool floating_base = parse_floating_base_arg(argc, argv);
    run_all_tests<float, 256>(floating_base, [&](cudaStream_t *streams, grid::robotModel<float> *m, grid::gridData<float> *d){
        run_batch_timings<float, TEST_ITERS_GLOBAL>(floating_base, streams, m, d);
    });
    return 0;
}

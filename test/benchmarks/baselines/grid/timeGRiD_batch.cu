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

#if GRID_HAS_INVERSE_DYNAMICS
template <typename T, int TEST_ITERS>
__host__ void measure_id_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("INVERSE_DYNAMICS", N, INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("INVERSE_DYNAMICS", N,
        [&]{ grid::inverse_dynamics<T,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::inverse_dynamics_compute_only<T,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_MINV
template <typename T, int TEST_ITERS>
__host__ void measure_minv_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("Minv", N, MINV_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("Minv", N,
        [&]{ grid::minv<T,true>(d,m,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::minv_compute_only<T,true>(d,m,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_FORWARD_DYNAMICS
template <typename T, int TEST_ITERS>
__host__ void measure_fd_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("FORWARD_DYNAMICS", N, FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("FORWARD_DYNAMICS", N,
        [&]{ grid::forward_dynamics<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::forward_dynamics_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_ABA
template <typename T, int TEST_ITERS>
__host__ void measure_aba_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("ABA", N, ABA_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("ABA", N,
        [&]{ grid::aba<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::aba_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_CRBA
template <typename T, int TEST_ITERS>
__host__ void measure_crba_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("CRBA", N, CRBA_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("CRBA", N,
        [&]{ grid::crba<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::crba_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
template <typename T, int TEST_ITERS>
__host__ void measure_id_du_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("INVERSE_DYNAMICS_GRADIENT", N, INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("INVERSE_DYNAMICS_GRADIENT", N,
        [&]{ grid::inverse_dynamics_gradient<T,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::inverse_dynamics_gradient_compute_only<T,false,true>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
template <typename T, int TEST_ITERS>
__host__ void measure_fd_du_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("FORWARD_DYNAMICS_GRADIENT", N, FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("FORWARD_DYNAMICS_GRADIENT", N,
        [&]{ grid::forward_dynamics_gradient<T,false>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::forward_dynamics_gradient_compute_only<T,false>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_END_EFFECTOR_POSE
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("END_EFFECTOR_POSE", N, END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("END_EFFECTOR_POSE", N,
        [&]{ grid::end_effector_pose<T>(d,m,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::end_effector_pose_compute_only<T>(d,m,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_gradient_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("END_EFFECTOR_POSE_GRADIENT", N, END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("END_EFFECTOR_POSE_GRADIENT", N,
        [&]{ grid::end_effector_pose_gradient<T>(d,m,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::end_effector_pose_gradient_compute_only<T>(d,m,N,dim3(N,1,1),dimms); });
}
#endif
// multi_target_position{,_gradient}: an OPT-IN family -- these blocks only exist when the robot was
// generated WITH a target batch (--multi-target-from-collision), so a default robot compiles them out
// and is unaffected. ⚠ THE COST SCALES WITH THE BATCH SIZE (NUM_MULTI_TARGETS): these µs are only
// meaningful next to the target count. No competitor has a counterpart -> capability-lead, not a W/L.
#if GRID_HAS_MULTI_TARGET_POSITION
template <typename T, int TEST_ITERS>
__host__ void measure_multi_target_position_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("MULTI_TARGET_POSITION", N, MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("MULTI_TARGET_POSITION", N,
        [&]{ grid::multi_target_position<T>(d,m,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::multi_target_position_compute_only<T>(d,m,N,dim3(N,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_multi_target_position_gradient_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("MULTI_TARGET_POSITION_GRADIENT", N, MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("MULTI_TARGET_POSITION_GRADIENT", N,
        [&]{ grid::multi_target_position_gradient<T>(d,m,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::multi_target_position_gradient_compute_only<T>(d,m,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN || (defined(GRID_BENCH_D2EE_ONLY) && GRID_BENCH_D2EE_ONLY)
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_hessian_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("END_EFFECTOR_POSE_HESSIAN", N, END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("END_EFFECTOR_POSE_HESSIAN", N,
        [&]{ grid::end_effector_pose_hessian<T>(d,m,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::end_effector_pose_hessian_compute_only<T>(d,m,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_IDSVA_SO_BODY_FRAME
template <typename T, int TEST_ITERS>
__host__ void measure_idsva_so_body_frame_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("IDSVA_SO_BODY_FRAME", N, IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("IDSVA_SO_BODY_FRAME", N,
        [&]{ grid::idsva_so_body_frame<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::idsva_so_body_frame_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
}
#endif
#if GRID_HAS_IDSVA_SO_WORLD_FRAME
template <typename T, int TEST_ITERS>
__host__ void measure_idsva_so_world_frame_batch(int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
    GRID_SKIP_BATCH_IF_KERNEL_TOO_BIG("IDSVA_SO_WORLD_FRAME", N, IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES);
    dim3 dimms = grid_timing_dimms();
    measure_batch_pair<TEST_ITERS>("IDSVA_SO_WORLD_FRAME", N,
        [&]{ grid::idsva_so_world_frame<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::idsva_so_world_frame_compute_only<T>(d,m,GRAVITY,N,dim3(N,1,1),dimms); });
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
        [&]{ grid::integrator_with_gradient<T>(d,m,GRAVITY,static_cast<T>(0.01),N,dim3(N,1,1),dimms,streams); },
        [&]{ grid::integrator_with_gradient_compute_only<T>(d,m,GRAVITY,static_cast<T>(0.01),N,dim3(N,1,1),dimms); });
}
#endif

template <typename T, int TEST_ITERS>
__host__ void run_batch_at(bool floating_base, int N, cudaStream_t *streams, grid::robotModel<T> *m, grid::gridData<T> *d){
#if defined(GRID_BENCH_D2EE_ONLY) && GRID_BENCH_D2EE_ONLY
    // d2ee-only fast bench: only measure ee_pose_hessian to keep compile+run short.
    (void)floating_base;
    measure_ee_pose_hessian_batch<T,TEST_ITERS>(N, streams, m, d);
#else
    // Core measures gated on GRID_HAS_* so a subset build (GRID_BENCH_ALGORITHM_LIST)
    // compiles — the measure_* templates are only instantiated when called, so gating
    // the call is sufficient (mirrors the SO/integrator block below).
#if GRID_HAS_INVERSE_DYNAMICS
    measure_id_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_MINV
    measure_minv_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_FORWARD_DYNAMICS
    measure_fd_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_ABA
    measure_aba_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_CRBA
    measure_crba_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
    measure_id_du_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
    measure_fd_du_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_END_EFFECTOR_POSE
    measure_ee_pose_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
    measure_ee_pose_gradient_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_MULTI_TARGET_POSITION
    measure_multi_target_position_batch<T,TEST_ITERS>(N, streams, m, d);
    measure_multi_target_position_gradient_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN
    measure_ee_pose_hessian_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_IDSVA_SO
    measure_idsva_so_batch<T,TEST_ITERS>(N, streams, m, d);
#endif
#if GRID_HAS_IDSVA_SO_BODY_FRAME
    // body_frame is the PRODUCTION second-order path only for fixed-base robots.
    // For floating-base the dispatcher routes SO -> world_frame, so the body_frame
    // kernel is non-production there and must NOT pollute the W/L tally; gate it off.
    if (!floating_base) {
        measure_idsva_so_body_frame_batch<T,TEST_ITERS>(N, streams, m, d);
    }
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
    run_batch_at<T,TEST_ITERS>(floating_base, 1024, streams, m, d);
#endif
}

int main(int argc, const char **argv){
    bool floating_base = parse_floating_base_arg(argc, argv);
    // MAX_TIMESTEPS sizes the device buffers (d_q_qd_u etc. + gridData) — must be
    // >= the largest batch N timed below (now 1024), or N=1024 launches read/write
    // out of bounds.
    run_all_tests<float, 1024>(floating_base, [&](cudaStream_t *streams, grid::robotModel<float> *m, grid::gridData<float> *d){
        run_batch_timings<float, TEST_ITERS_GLOBAL>(floating_base, streams, m, d);
    });
    return 0;
}

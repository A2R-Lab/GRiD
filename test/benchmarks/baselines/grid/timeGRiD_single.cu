/***
 * Single-call timing binary. Compiled with `-rdc=true` so the anti-LICM
 * `__noinline__ grid_licm_barrier` actually defeats LICM in the
 * `_single_timing` rep loops. Only the `_single_timing` kernel variants
 * are instantiated here — the batch counterparts live in timeGRiD_batch.cu.
 *
 * Output format matches the original timeGRiD.cu so test/benchmarks/
 * timing_parser.py picks up "Single Call <ALGO> Xus" lines unchanged.
 ***/
#include "timeGRiD_common.h"

// ---------------------------------------------------------------------------
// Per-algo single-call timing helpers. Each wraps the corresponding
// grid::<algo>_single_timing<...>() entrypoint.
// ---------------------------------------------------------------------------
template <typename T, int TEST_ITERS>
__host__ void measure_id_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("ID", ID_DYNAMIC_SHARED_MEM_BYTES);
    grid::inverse_dynamics_single_timing<T,false,true>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_minv_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("Minv", MINV_DYNAMIC_SHARED_MEM_BYTES);
    grid::minv_single_timing<T,true>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_fd_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("FD", FD_DYNAMIC_SHARED_MEM_BYTES);
    grid::forward_dynamics_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_aba_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("ABA", ABA_DYNAMIC_SHARED_MEM_BYTES);
    grid::aba_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_crba_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("CRBA", CRBA_DYNAMIC_SHARED_MEM_BYTES);
    grid::crba_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_id_du_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("ID_DU", ID_DU_DYNAMIC_SHARED_MEM_BYTES);
    grid::inverse_dynamics_gradient_single_timing<T,false,true>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_fd_du_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("FD_DU", FD_DU_DYNAMIC_SHARED_MEM_BYTES);
    grid::forward_dynamics_gradient_single_timing<T,false>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("EE_POSE", EE_POS_DYNAMIC_SHARED_MEM_BYTES);
    grid::end_effector_pose_single_timing<T>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_gradient_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("EE_POSE_GRADIENT", DEE_POS_DYNAMIC_SHARED_MEM_BYTES);
    grid::end_effector_pose_gradient_single_timing<T>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_hessian_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("EE_POSE_HESSIAN", D2EE_POS_DYNAMIC_SHARED_MEM_BYTES);
    grid::end_effector_pose_hessian_single_timing<T>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
#if GRID_HAS_IDSVA_SO_BODY_FRAME
template <typename T, int TEST_ITERS>
__host__ void measure_idsva_so_body_frame_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("IDSVA_SO_BODY_FRAME", IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES);
    grid::idsva_so_body_frame_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
#endif
#if GRID_HAS_IDSVA_SO_WORLD_FRAME
template <typename T, int TEST_ITERS>
__host__ void measure_idsva_so_world_frame_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("IDSVA_SO_WORLD_FRAME", IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES);
    grid::idsva_so_world_frame_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
#endif
#if GRID_HAS_FDSVA_SO
template <typename T, int TEST_ITERS>
__host__ void measure_fdsva_so_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("FDSVA_SO", FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES);
    grid::fdsva_so_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
#endif
#if GRID_HAS_IDSVA_SO
template <typename T, int TEST_ITERS>
__host__ void measure_idsva_so_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("IDSVA_SO", IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES);
    grid::idsva_so_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
#endif
// Time integrators. Host signatures take an extra dt (const T) before num_timesteps;
// IntegratorType defaults to EULER. A fixed bench dt is used (it doesn't affect timing).
#if GRID_HAS_INTEGRATOR
template <typename T, int TEST_ITERS>
__host__ void measure_integrator_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("INTEGRATOR", INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES);
    grid::integrator_single_timing<T>(hd_data,d_robotModel,GRAVITY,static_cast<T>(0.01),TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
#endif
#if GRID_HAS_INTEGRATOR_GRADIENT
template <typename T, int TEST_ITERS>
__host__ void measure_integrator_gradient_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("INTEGRATOR_GRADIENT", INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES);
    grid::integrator_gradient_single_timing<T>(hd_data,d_robotModel,GRAVITY,static_cast<T>(0.01),TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_integrator_with_gradient_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    GRID_SKIP_IF_KERNEL_TOO_BIG("INTEGRATOR_WITH_GRADIENT", INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES);
    grid::integrator_with_gradient_single_timing<T>(hd_data,d_robotModel,GRAVITY,static_cast<T>(0.01),TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
#endif

template <typename T, int TEST_ITERS>
__host__ void run_single_timings(bool floating_base, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
#if !TEST_FOR_EQUIVALENCE
    measure_id_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    measure_minv_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    measure_fd_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    measure_aba_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    measure_crba_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    measure_id_du_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    measure_fd_du_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    measure_ee_pose_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    measure_ee_pose_gradient_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    measure_ee_pose_hessian_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    #if GRID_HAS_IDSVA_SO
    measure_idsva_so_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    #endif
    #if GRID_HAS_IDSVA_SO_BODY_FRAME
    measure_idsva_so_body_frame_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    #endif
    #if GRID_HAS_IDSVA_SO_WORLD_FRAME
    measure_idsva_so_world_frame_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    #endif
    #if GRID_HAS_FDSVA_SO
    measure_fdsva_so_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    (void)floating_base;
    #endif
    #if GRID_HAS_INTEGRATOR
    measure_integrator_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    #endif
    #if GRID_HAS_INTEGRATOR_GRADIENT
    measure_integrator_gradient_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    measure_integrator_with_gradient_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    #endif
#endif
}

int main(int argc, const char **argv){
    bool floating_base = parse_floating_base_arg(argc, argv);
    run_all_tests<float, 256>(floating_base, [&](cudaStream_t *streams, grid::robotModel<float> *m, grid::gridData<float> *d){
        run_single_timings<float, SINGLE_CALL_ITERS_GLOBAL>(floating_base, streams, m, d);
    });
    return 0;
}

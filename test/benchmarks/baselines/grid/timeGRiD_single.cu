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
    grid::inverse_dynamics_single_timing<T,false,true>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_minv_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::direct_minv_single_timing<T,true>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_fd_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::forward_dynamics_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_aba_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::aba_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_crba_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::crba_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_id_du_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::inverse_dynamics_gradient_single_timing<T,false,true>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_fd_du_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::forward_dynamics_gradient_single_timing<T,false>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::end_effector_pose_single_timing<T>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_gradient_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::end_effector_pose_gradient_single_timing<T>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
#if GRID_HAS_IDSVA_SO
template <typename T, int TEST_ITERS>
__host__ void measure_idsva_so_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::idsva_so_host_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
#endif
#if GRID_HAS_FDSVA_SO
template <typename T, int TEST_ITERS>
__host__ void measure_fdsva_so_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::fdsva_so_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),grid_timing_dimms(),streams);
}
#endif

template <typename T, int TEST_ITERS>
__host__ void run_single_timings(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
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
    #if GRID_HAS_IDSVA_SO
    measure_idsva_so_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    #endif
    #if GRID_HAS_FDSVA_SO
    measure_fdsva_so_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
    #endif
#endif
}

int main(int argc, const char **argv){
    bool floating_base = parse_floating_base_arg(argc, argv);
    run_all_tests<float, 256>(floating_base, [&](cudaStream_t *streams, grid::robotModel<float> *m, grid::gridData<float> *d){
        run_single_timings<float, SINGLE_CALL_ITERS_GLOBAL>(streams, m, d);
    });
    return 0;
}

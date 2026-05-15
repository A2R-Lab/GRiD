/***
nvcc -std=c++11 -o timeGRiD.exe timeGRiD.cu -gencode arch=compute_86,code=sm_86 -O3 -ftz=true -prec-div=false -prec-sqrt=false

PHASE 7B step 1 of 3: factored per-algo measurement blocks out of the giant
test<>() function into separate `measure_<algo>_single<>()` and
`measure_<algo>_batch<>()` template functions. Same compile time + behavior
as the original; intermediate step toward step 2 (move each algo to its own
.cu file) and step 3 (parallel nvcc compile in grid/run.py).

Each measure function does ONE algo's WITH MEMORY + COMPUTE ONLY pair (for
batch) or just the single_timing call (for single). The dispatcher
test<>() iterates over the algo list. Splitting cleanly into per-TU files
in step 2 just means moving each measure_<algo> function to its own .cu;
no behavioral changes needed.
***/

#ifndef GRID_HEADER_FILE
#include "../../../../grid.cuh"
#else
#include GRID_HEADER_FILE
#endif
#include "../util/experiment_helpers.h"

dim3 dimms(grid::SUGGESTED_THREADS,1,1);
#define GRAVITY 9.81

// ---------------------------------------------------------------------------
// Per-algo single-call timing helpers. Called once for NUM_TIMESTEPS==1.
// Each just wraps grid::<algo>_single_timing<...>().
// ---------------------------------------------------------------------------
template <typename T, int TEST_ITERS>
__host__ void measure_id_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::inverse_dynamics_single_timing<T,false,true>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_minv_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::direct_minv_single_timing<T,true>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),dimms,streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_fd_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::forward_dynamics_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_aba_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::aba_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_crba_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::crba_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_id_du_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::inverse_dynamics_gradient_single_timing<T,false,true>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_fd_du_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::forward_dynamics_gradient_single_timing<T,false>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::end_effector_pose_single_timing<T>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),dimms,streams);
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_gradient_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::end_effector_pose_gradient_single_timing<T>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),dimms,streams);
}
#if GRID_GENERATES_IDSVA_SO
template <typename T, int TEST_ITERS>
__host__ void measure_idsva_so_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::idsva_so_host_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);
}
#endif
#if GRID_GENERATES_FDSVA_SO
template <typename T, int TEST_ITERS>
__host__ void measure_fdsva_so_single(cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    grid::fdsva_so_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);
}
#endif

// ---------------------------------------------------------------------------
// Per-algo batch timing helpers. Each runs TEST_ITERS reps of the WITH MEMORY
// and COMPUTE ONLY variants for one algo at NUM_TIMESTEPS samples and prints
// the result. The shared timing loop is a template that takes the two
// invocations as callables (lambdas) — a function-style macro chokes on the
// commas inside `dim3(N,1,1)`.
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
__host__ void measure_id_batch(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    measure_batch_pair<TEST_ITERS>("ID", NUM_TIMESTEPS,
        [&]{ grid::inverse_dynamics<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams); },
        [&]{ grid::inverse_dynamics_compute_only<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_minv_batch(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    measure_batch_pair<TEST_ITERS>("Minv", NUM_TIMESTEPS,
        [&]{ grid::direct_minv<T,true>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams); },
        [&]{ grid::direct_minv_compute_only<T,true>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_fd_batch(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    measure_batch_pair<TEST_ITERS>("FD", NUM_TIMESTEPS,
        [&]{ grid::forward_dynamics<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams); },
        [&]{ grid::forward_dynamics_compute_only<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_aba_batch(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    measure_batch_pair<TEST_ITERS>("ABA", NUM_TIMESTEPS,
        [&]{ grid::aba<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams); },
        [&]{ grid::aba_compute_only<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_crba_batch(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    measure_batch_pair<TEST_ITERS>("CRBA", NUM_TIMESTEPS,
        [&]{ grid::crba<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams); },
        [&]{ grid::crba_compute_only<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_id_du_batch(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    measure_batch_pair<TEST_ITERS>("ID_DU", NUM_TIMESTEPS,
        [&]{ grid::inverse_dynamics_gradient<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams); },
        [&]{ grid::inverse_dynamics_gradient_compute_only<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_fd_du_batch(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    measure_batch_pair<TEST_ITERS>("FD_DU", NUM_TIMESTEPS,
        [&]{ grid::forward_dynamics_gradient<T,false>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams); },
        [&]{ grid::forward_dynamics_gradient_compute_only<T,false>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_batch(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    measure_batch_pair<TEST_ITERS>("EE_POSE", NUM_TIMESTEPS,
        [&]{ grid::end_effector_pose<T>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams); },
        [&]{ grid::end_effector_pose_compute_only<T>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms); });
}
template <typename T, int TEST_ITERS>
__host__ void measure_ee_pose_gradient_batch(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    measure_batch_pair<TEST_ITERS>("EE_POSE_GRADIENT", NUM_TIMESTEPS,
        [&]{ grid::end_effector_pose_gradient<T>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams); },
        [&]{ grid::end_effector_pose_gradient_compute_only<T>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms); });
}
#if GRID_GENERATES_IDSVA_SO
template <typename T, int TEST_ITERS>
__host__ void measure_idsva_so_batch(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    measure_batch_pair<TEST_ITERS>("IDSVA_SO", NUM_TIMESTEPS,
        [&]{ grid::idsva_so_host<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams); },
        [&]{ grid::idsva_so_host_compute_only<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms); });
}
#endif
#if GRID_GENERATES_FDSVA_SO
template <typename T, int TEST_ITERS>
__host__ void measure_fdsva_so_batch(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
    measure_batch_pair<TEST_ITERS>("FDSVA_SO", NUM_TIMESTEPS,
        [&]{ grid::fdsva_so<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams); },
        [&]{ grid::fdsva_so_compute_only<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms); });
}
#endif

// ---------------------------------------------------------------------------
// Dispatcher: invokes every algo's measure function for the given
// NUM_TIMESTEPS. Single-call path uses SINGLE_CALL_ITERS_GLOBAL; batch path
// uses TEST_ITERS_GLOBAL.
// ---------------------------------------------------------------------------
template <typename T, int TEST_ITERS>
__host__
void test(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
   	#if TEST_FOR_EQUIVALENCE
		printf("q,qd,u\n");
   	#else
		if(NUM_TIMESTEPS == 1){
			measure_id_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
			measure_minv_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
			measure_fd_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
			measure_aba_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
			measure_crba_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
			measure_id_du_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
			measure_fd_du_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
			measure_ee_pose_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
			measure_ee_pose_gradient_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
			#if GRID_GENERATES_IDSVA_SO
			measure_idsva_so_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
			#endif
			#if GRID_GENERATES_FDSVA_SO
			measure_fdsva_so_single<T,TEST_ITERS>(streams, d_robotModel, hd_data);
			#endif
		}
		else{
			measure_id_batch<T,TEST_ITERS>(NUM_TIMESTEPS, streams, d_robotModel, hd_data);
			measure_minv_batch<T,TEST_ITERS>(NUM_TIMESTEPS, streams, d_robotModel, hd_data);
			measure_fd_batch<T,TEST_ITERS>(NUM_TIMESTEPS, streams, d_robotModel, hd_data);
			measure_aba_batch<T,TEST_ITERS>(NUM_TIMESTEPS, streams, d_robotModel, hd_data);
			measure_crba_batch<T,TEST_ITERS>(NUM_TIMESTEPS, streams, d_robotModel, hd_data);
			measure_id_du_batch<T,TEST_ITERS>(NUM_TIMESTEPS, streams, d_robotModel, hd_data);
			measure_fd_du_batch<T,TEST_ITERS>(NUM_TIMESTEPS, streams, d_robotModel, hd_data);
			measure_ee_pose_batch<T,TEST_ITERS>(NUM_TIMESTEPS, streams, d_robotModel, hd_data);
			measure_ee_pose_gradient_batch<T,TEST_ITERS>(NUM_TIMESTEPS, streams, d_robotModel, hd_data);
			#if GRID_GENERATES_IDSVA_SO
			measure_idsva_so_batch<T,TEST_ITERS>(NUM_TIMESTEPS, streams, d_robotModel, hd_data);
			#endif
			#if GRID_GENERATES_FDSVA_SO
			measure_fdsva_so_batch<T,TEST_ITERS>(NUM_TIMESTEPS, streams, d_robotModel, hd_data);
			#endif
		}
	#endif
}

template<typename T, int TEST_ITERS>
void run_all_tests(bool floating_base){
	const int MAX_TIMESTEPS = 256;
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
	for(int w = 0; w < 5; w++){
		grid::inverse_dynamics<T,false,true>(hd_data,d_robotModel,GRAVITY,MAX_TIMESTEPS,dim3(MAX_TIMESTEPS,1,1),dimms,streams);
	}
	gpuErrchk(cudaDeviceSynchronize());

	test<T,SINGLE_CALL_ITERS_GLOBAL>(1,streams,d_robotModel,hd_data);
	#if !TEST_FOR_EQUIVALENCE
		test<T,TEST_ITERS>(16,streams,d_robotModel,hd_data);
		test<T,TEST_ITERS>(32,streams,d_robotModel,hd_data);
		test<T,TEST_ITERS>(64,streams,d_robotModel,hd_data);
		test<T,TEST_ITERS>(128,streams,d_robotModel,hd_data);
		test<T,TEST_ITERS>(256,streams,d_robotModel,hd_data);
	#endif

	grid::close_grid<T>(streams,d_robotModel,hd_data);
}

int main(int argc, const char **argv){
	bool floating_base = false;
	if (argc > 1 && argv[1][0] == 'T') {floating_base = true; printf("Floating Base = True\n");}
	else {printf("Floating Base = False\n");}
	run_all_tests<float,TEST_ITERS_GLOBAL>(floating_base); return 0;
}

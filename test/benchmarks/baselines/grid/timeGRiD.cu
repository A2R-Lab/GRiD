/***
nvcc -std=c++11 -o timeGRiD.exe timeGRiD.cu -gencode arch=compute_86,code=sm_86 -O3 -ftz=true -prec-div=false -prec-sqrt=false
***/

#ifndef GRID_HEADER_FILE
#include "../../../../grid.cuh"
#else
#include GRID_HEADER_FILE
#endif
#include "../util/experiment_helpers.h"

dim3 dimms(grid::SUGGESTED_THREADS,1,1);
#define GRAVITY 9.81

template <typename T, int TEST_ITERS>
__host__
void test(int NUM_TIMESTEPS, cudaStream_t *streams, grid::robotModel<T> *d_robotModel, grid::gridData<T> *hd_data){
   	#if TEST_FOR_EQUIVALENCE
		printf("q,qd,u\n");
   	#else
		struct timespec start, end;
	   	std::vector<double> times = {};

		if(NUM_TIMESTEPS == 1){
    		grid::inverse_dynamics_single_timing<T,false,true>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);

    		grid::direct_minv_single_timing<T,true>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),dimms,streams);

    		grid::forward_dynamics_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);

    		grid::aba_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);

    		grid::crba_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);

    		grid::inverse_dynamics_gradient_single_timing<T,false,true>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);

    		grid::forward_dynamics_gradient_single_timing<T,false>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);

    		grid::end_effector_pose_single_timing<T>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),dimms,streams);

    		grid::end_effector_pose_gradient_single_timing<T>(hd_data,d_robotModel,TEST_ITERS,dim3(1,1,1),dimms,streams);

    		#if GRID_GENERATES_IDSVA_SO
    		grid::idsva_so_host_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);
    		#endif

    		#if GRID_GENERATES_FDSVA_SO
    		grid::fdsva_so_single_timing<T>(hd_data,d_robotModel,GRAVITY,TEST_ITERS,dim3(1,1,1),dimms,streams);
    		#endif
		}
		else{
			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::inverse_dynamics<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: ID WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::inverse_dynamics_compute_only<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: ID COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::direct_minv<T,true>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: Minv WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::direct_minv_compute_only<T,true>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: Minv COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::forward_dynamics<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: FD WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::forward_dynamics_compute_only<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: FD COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::aba<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: ABA WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::aba_compute_only<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: ABA COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::crba<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: CRBA WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::crba_compute_only<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: CRBA COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::inverse_dynamics_gradient<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: ID_DU WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::inverse_dynamics_gradient_compute_only<T,false,true>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: ID_DU COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::forward_dynamics_gradient<T,false>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: FD_DU WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::forward_dynamics_gradient_compute_only<T,false>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: FD_DU COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::end_effector_pose<T>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: EE_POSE WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::end_effector_pose_compute_only<T>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: EE_POSE COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::end_effector_pose_gradient<T>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: EE_POSE_GRADIENT WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::end_effector_pose_gradient_compute_only<T>(hd_data,d_robotModel,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: EE_POSE_GRADIENT COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			#if GRID_GENERATES_IDSVA_SO
			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::idsva_so_host<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: IDSVA_SO WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::idsva_so_host_compute_only<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: IDSVA_SO COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();
			#endif

			#if GRID_GENERATES_FDSVA_SO
			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::fdsva_so<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms,streams);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: FDSVA_SO WITH MEMORY: ",NUM_TIMESTEPS); printStats(&times); times.clear();

			for(int iter = 0; iter < TEST_ITERS; iter++){
				clock_gettime(CLOCK_MONOTONIC,&start);
				grid::fdsva_so_compute_only<T>(hd_data,d_robotModel,GRAVITY,NUM_TIMESTEPS,dim3(NUM_TIMESTEPS,1,1),dimms);
				clock_gettime(CLOCK_MONOTONIC,&end);
				times.push_back(time_delta_us_timespec(start,end));
			}
			printf("[N:%d]: FDSVA_SO COMPUTE ONLY: ",NUM_TIMESTEPS); printStats(&times); times.clear();
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

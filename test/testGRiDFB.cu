/***
nvcc -std=c++11 -o testGRiD.exe testGRiD.cu -gencode arch=compute_86,code=sm_86
***/

#include <iostream>
#include <algorithm>
#include <type_traits>
#include "../grid.cuh"

template <typename T>
__host__
void test(){
	T gravity = static_cast<T>(9.81);
	dim3 dimms(grid::SUGGESTED_THREADS,1,1);
	cudaStream_t *streams = grid::init_grid<T>();
	grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
	grid::gridData<T> *hd_data = grid::init_gridData<T,1>();

	// initialize values based on floating base or not
	T random_fb[90] = {0.300623, -1.427442, 0.047334, 0.2675568650090604, -0.6452699620984409, -0.001724583948895716, -0.7155676249036168, -1.226503, -0.619695, 0.973148, -0.750689, -0.253769, 0.493305, -0.695605, 0.425334, 0.340006, -0.178834, -0.013169, -2.349815, 0.405039, -2.266609, -0.424634, 1.034167, -0.270165, -0.18414, -1.111512, 0.659046, 0.183907, 0.944741, 0.579223, 0.497338, 0.870245, 1.098656, 1.553845, -1.160813, -2.30901, 0.501948, 1.172242, 0.451889, 0.883051, -0.662848, 0.038682, 0.814782, 1.139002, 0.2817, -1.699318, 0.72425, 0.503564, 0.78011, -0.424718, 0.736483, -1.500795, 0.636129, -0.351871, 0.029238, -1.177703, 0.329867, 0.684543, 0.223669, 1.556482, -0.477746, 2.010085, 0.26853, 1.4253, 1.747454, -0.317835, 0.336185, 0.752943, -0.506264, -2.587783, -0.356798, 0.154351, 2.536409, -0.547202, -1.094094, 0.600488, 0.473008, -0.033037, 0.095979, -1.173089, 0.04475, -1.920187, 0.656968, -0.625342, 0.762751, 1.943894, 1.846422, 0.207588, -0.233651, -0.57805};
	T random_nonfb[90] = {0.300623, -1.427442, 0.047334, -0.512040, -1.437442, 0.500384, -0.881586, -1.226503, -0.619695, 0.973148, -0.750689, -0.253769, 0.493305, -0.695605, 0.425334, 0.340006, -0.178834, -0.013169, -2.349815, 0.405039, -2.266609, -0.424634, 1.034167, -0.270165, -0.184140, -1.111512, 0.659046, 0.183907, 0.944741, 0.579223, 0.497338, 0.870245, 1.098656, 1.553845, -1.160813, -2.309010, 0.501948, 1.172242, 0.451889, 0.883051, -0.662848, 0.038682, 0.814782, 1.139002, 0.281700, -1.699318, 0.724250, 0.503564, 0.780110, -0.424718, 0.736483, -1.500795, 0.636129, -0.351871, 0.029238, -1.177703, 0.329867, 0.684543, 0.223669, 1.556482, -0.477746, 2.010085, 0.268530, 1.425300, 1.747454, -0.317835, 0.336185, 0.752943, -0.506264, -2.587783, -0.356798, 0.154351, 2.536409, -0.547202, -1.094094, 0.600488, 0.473008, -0.033037, 0.095979, -1.173089, 0.044750, -1.920187, 0.656968, -0.625342, 0.762751, 1.943894, 1.846422, 0.207588, -0.233651, -0.578050};
	T random[90];
	if (grid::NUM_JOINTS != grid::NUM_VEL) {
		for(int i = 0; i < 90; i++) random[i] = random_fb[i];
	}
	else {
		for(int i = 0; i < 90; i++) random[i] = random_nonfb[i];
	}
	if (grid::NUM_JOINTS > 30) {
		printf("GRiD does not support robots with > 30 dof"); 
		return;
	} 

	for (int i = 0; i < grid::NUM_JOINTS; i++) hd_data->h_q_qd_u[i] = random[i];
	for (int i = 0; i < grid::NUM_VEL; i++) hd_data->h_q_qd_u[grid::NUM_JOINTS+i] = random[i+grid::NUM_JOINTS];
	for (int i = 0; i < grid::NUM_VEL; i++) hd_data->h_q_qd_u[grid::NUM_JOINTS+grid::NUM_VEL+i] = static_cast<T>(0);


	gpuErrchk(cudaMemcpy(hd_data->d_q_qd_u,hd_data->h_q_qd_u,3*grid::NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());

	// q,qd,u
	printMat<T,1,grid::NUM_JOINTS>(hd_data->h_q_qd_u,1);
	printf("\n");
	printMat<T,1,grid::NUM_VEL>(&hd_data->h_q_qd_u[grid::NUM_JOINTS],1);
    printf("\n");
	printMat<T,1,grid::NUM_VEL>(&hd_data->h_q_qd_u[grid::NUM_JOINTS+grid::NUM_VEL],1);

	grid::inverse_dynamics_gradient<T,false,false>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dim3(32,1,1),streams);
	printf("\n");
	printMat<T,grid::NUM_VEL,grid::NUM_VEL>(hd_data->h_dc_du,grid::NUM_VEL);
	printf("\n");
	printMat<T,grid::NUM_VEL,grid::NUM_VEL>(&hd_data->h_dc_du[grid::NUM_VEL*grid::NUM_VEL],grid::NUM_VEL);

	printf("\n");
	grid::direct_minv<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
	printMat<T,grid::NUM_VEL,grid::NUM_VEL>(hd_data->h_Minv,grid::NUM_VEL);

	printf("\n");
	grid::forward_dynamics<T>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
	printMat<T,1,grid::NUM_VEL>(hd_data->h_qdd,1);

	printf("\n");
	grid::inverse_dynamics<T,false,false>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
	printMat<T,1,grid::NUM_VEL>(hd_data->h_c,1);

	grid::forward_dynamics_gradient<T,false>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dim3(32,1,1),streams);
	printf("\n");
	printMat<T,grid::NUM_VEL,grid::NUM_VEL>(hd_data->h_df_du,grid::NUM_VEL);
	printf("\n");
	printMat<T,grid::NUM_VEL,grid::NUM_VEL>(&hd_data->h_df_du[grid::NUM_VEL*grid::NUM_VEL],grid::NUM_VEL);
	

		// printf("\n");
		// grid::end_effector_positions<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
		// printMat<T,1,6*grid::NUM_EES>(hd_data->h_eePos,1);

		// printf("\n");
		// grid::end_effector_positions_gradient<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
		// printMat<T,6,grid::NUM_EES*grid::NUM_JOINTS>(hd_data->h_deePos,6);

		// grid::aba<T>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
		// printf("\n");
		// printMat<T,1,grid::NUM_JOINTS>(hd_data->h_qdd,1);

		// grid::crba<T>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
		// printf("\n");
		// printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(hd_data->h_M,grid::NUM_JOINTS);
	

	grid::close_grid<T>(streams,d_robotModel,hd_data);
}

int main(){
	test<float>(); return 0;
}
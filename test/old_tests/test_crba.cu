#include "grid.cuh" 
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <cuda_runtime.h>
//run CodeGen to get grid.cuh

int main() {

    printf("buckle up and enjoy the ride \n");

    grid::gridData<float> *hd_data = grid::init_gridData<float,1>();
    grid::robotModel<float> *d_robotModel = grid::init_robotModel<float>();
    const int num_timesteps = 1;
    float gravity = static_cast<float>(9.81);
    //printf("gravity =  %f", gravity);
    dim3 dimms(grid::SUGGESTED_THREADS,1,1);
    cudaStream_t *streams = grid::init_grid<float>();

    hd_data->h_q_qd_u[0] = 1.24;
    hd_data->h_q_qd_u[1] = 0.13;
    hd_data->h_q_qd_u[2] = -0.17;
    hd_data->h_q_qd_u[3] = 1.33;
    hd_data->h_q_qd_u[4] = 0.22;
    hd_data->h_q_qd_u[5] = -0.56;
    hd_data->h_q_qd_u[6] = 0.99;

    hd_data->h_q_qd_u[7] = 0;
    hd_data->h_q_qd_u[8] = 0;
    hd_data->h_q_qd_u[9] = 0;
    hd_data->h_q_qd_u[10] = 0;
    hd_data->h_q_qd_u[11] = 0;
    hd_data->h_q_qd_u[12] = 0;
    hd_data->h_q_qd_u[13] = 0;

    hd_data->h_q_qd_u[14] = 0;
    hd_data->h_q_qd_u[15] = 0;
    hd_data->h_q_qd_u[16] = 0;
    hd_data->h_q_qd_u[17] = 0;
    hd_data->h_q_qd_u[18] = 0;
    hd_data->h_q_qd_u[19] = 0;
    hd_data->h_q_qd_u[20] = 0;

    gpuErrchk(cudaMemcpy(hd_data->d_q_qd_u,hd_data->h_q_qd_u,3*grid::NUM_JOINTS*sizeof(float),cudaMemcpyHostToDevice));
	gpuErrchk(cudaDeviceSynchronize());

    //printf("testing....");

	//printf("q,qd,u\n");
	//printMat<float,1,grid::NUM_JOINTS>(hd_data->h_q_qd_u,1);
	//printMat<float,1,grid::NUM_JOINTS>(&hd_data->h_q_qd_u[grid::NUM_JOINTS],1);
	//printMat<float,1,grid::NUM_JOINTS>(&hd_data->h_q_qd_u[2*grid::NUM_JOINTS],1);
    
    grid::crba<float>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
	printMat<float,1,grid::NUM_JOINTS>(hd_data->h_H,1);
    printf("byeee \n");
    // print H matrix so for loops;

    return 0; 

}


// run it with nvcc test_crba_cu -std=c++11 --> run on ssh or old laptop bc GPU 
// genCode = compute_??? arch__??? --> need to google to see what type of GPU
//nvidia-smi will spit out lots of info about GPU
//-o text.exe changes the name to test.exe 
#include "grid.cuh"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

int main() {
    grid::gridData<float> *hd_data = grid::init_gridData<float,1>();
    grid::robotModel<float> *d_robotModel = grid::init_robotModel<float>();;
    const int num_timesteps = 1;
    float gravity = static_cast<float>(9.81);
    dim3 dimms(grid::SUGGESTED_THREADS,1,1);
    cudaStream_t *streams = grid::init_grid<float>();
    hd_data->h_q_qd_u[0] = 0.8;
    hd_data->h_q_qd_u[1] = 0.3;
    hd_data->h_q_qd_u[2] = 1;
    hd_data->h_q_qd_u[3] = 0.2;
    hd_data->h_q_qd_u[4] = 0.7;
    hd_data->h_q_qd_u[5] = 0.6;
    hd_data->h_q_qd_u[6] = 0.4;

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

    printf("q,qd,u\n");
    printMat<float,1,grid::NUM_JOINTS>(hd_data->h_q_qd_u,1);
    printMat<float,1,grid::NUM_JOINTS>(&hd_data->h_q_qd_u[grid::NUM_JOINTS],1);
    printMat<float,1,grid::NUM_JOINTS>(&hd_data->h_q_qd_u[2*grid::NUM_JOINTS],1);

    printf("aba\n");
    grid::aba<float>(hd_data, d_robotModel, gravity, 1, dim3(1,1,1), dimms, streams);
    printMat<float,1,grid::NUM_JOINTS>(hd_data->h_qdd,1);
    return 0;
}
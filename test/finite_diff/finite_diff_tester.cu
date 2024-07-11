#include <iostream>
#include "finite_diff.cuh"
// #include "/home/a2rlab4/GRiDBenchmarks/GRiD/grid.cuh"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

using namespace grid; 

int example_func(float *input, float *func_output, const grid::robotModel<float> *d_robotModel, const float gravity) {

    // grid::aba<float>(func_output, va, input, &input[7], &input[14], &input[21], temp, gravity); 
    dim3 dimms(grid::SUGGESTED_THREADS,1,1);
    cudaStream_t *streams = grid::init_grid<float>();
    grid::gridData<float> *hd_data = grid::init_gridData<float,1>();

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

    grid::aba<float>(hd_data, d_robotModel, gravity, 1, dim3(1,1,1), dimms, streams);
    return 0;

}

template <typename T, int O_R, int O_C, int I_N>
__global__ void fd_kernel(T *d_q_qd_u, T *d_fdoutput, T eps, const robotModel<T> *d_robotModel, const T gravity) {
    __shared__ T s_fdoutput[49*7*4];
    // __shared__ T s_fdoutput[98];
    __shared__ T s_qdd_plus[O_R];
    __shared__ T s_qdd_minus[O_R];
    __shared__ T s_q_qd_tau[3*O_R]; 

    for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 3*O_R; ind += blockDim.x*blockDim.y){
        s_q_qd_tau[ind] = d_q_qd_u[ind];
    }
    // finite_diff_aba<T,O_R,O_C,I_N>(s_q_qd_tau, s_fdoutput, eps, d_robotModel, gravity);
    finite_diff_for_ddp<T,O_R,O_C,I_N>(s_q_qd_tau, s_fdoutput, eps, d_robotModel, gravity);
    // finite_diff<T,O_R,O_C,I_N>(input, output, example_func(input, func_output, d_robotModel, gravity), eps);

    for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49*7*4; ind += blockDim.x*blockDim.y){
        d_fdoutput[ind] = s_fdoutput[ind];
    }
     __syncthreads();
}

int main() {   
    grid::gridData<float> *hd_data = grid::init_gridData<float,1>();
    grid::robotModel<float> *d_robotModel = grid::init_robotModel<float>();
    float gravity = static_cast<float>(9.81);
    // dim3 dimms(grid::SUGGESTED_THREADS,1,1);
    cudaStream_t *streams = grid::init_grid<float>();
    
    // q
    hd_data->h_q_qd_u[0] = 1.24;
    hd_data->h_q_qd_u[1] = 0.13;
    hd_data->h_q_qd_u[2] = -0.17;
    hd_data->h_q_qd_u[3] = 1.33;
    hd_data->h_q_qd_u[4] = 0.22;
    hd_data->h_q_qd_u[5] = -0.56;
    hd_data->h_q_qd_u[6] = 0.99;
    //qd
    hd_data->h_q_qd_u[7] = 0.0;
    hd_data->h_q_qd_u[8] = 0.0;
    hd_data->h_q_qd_u[9] = 0.0;
    hd_data->h_q_qd_u[10] = 0.0;
    hd_data->h_q_qd_u[11] = 0.0;
    hd_data->h_q_qd_u[12] = 0.0;
    hd_data->h_q_qd_u[13] = 0.0;

    //tau
    hd_data->h_q_qd_u[14] = 0.0;
    hd_data->h_q_qd_u[15] = 0.0;
    hd_data->h_q_qd_u[16] = 0.0;
    hd_data->h_q_qd_u[17] = 0.0;
    hd_data->h_q_qd_u[18] = 0.0;
    hd_data->h_q_qd_u[19] = 0.0;
    hd_data->h_q_qd_u[20] = 0.0;

    gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,21*sizeof(float),cudaMemcpyHostToDevice,streams[0]));
    gpuErrchk(cudaDeviceSynchronize());

    float eps = 0.1; 
    // T *input, T *output, T (*func)(T*), T eps, const robotModel<T> *d_robotModel, const float gravity, const int NUM_TIMESTEPS)
    // fd_kernel<float,7,7,7><<<1,1>>>(input, func_output, output, eps, d_robotModel, gravity);
    // printf("in main to kernel\n"); 
    // fd_kernel<float,7,7,7><<<1,1,grid::ABA_DYNAMIC_SHARED_MEM_COUNT*sizeof(float)>>>(hd_data->d_q_qd_u, hd_data->d_fdoutput, eps, d_robotModel, gravity); 
    fd_kernel<float,7,7,7><<<1,1,grid:: ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(float)>>>(hd_data->d_q_qd_u, hd_data->d_fdoutput, eps, d_robotModel, gravity); 
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(hd_data->h_fdoutput,hd_data->d_fdoutput,49*7*4*sizeof(float),cudaMemcpyDeviceToHost));
    gpuErrchk(cudaDeviceSynchronize());


    gpuErrchk(cudaFree(d_robotModel));
    gpuErrchk(cudaFree(hd_data->d_q_qd_u));
    gpuErrchk(cudaFree(hd_data->d_fdoutput));
    free(hd_data->h_fdoutput);
    free(hd_data->h_q_qd_u);
    for(int i=0; i<3; i++){gpuErrchk(cudaStreamDestroy(streams[i]));} free(streams);

    return 0;

};
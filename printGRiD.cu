/***
nvcc -std=c++11 -o printGRiD.exe printGRiD.cu -gencode arch=compute_86,code=sm_86
***/

#include <random>
#include <algorithm>
#include "grid.cuh"
#define RANDOM_MEAN 0
#define RANDOM_STDEV 1
std::default_random_engine randEng(1337); // fixed seed
std::normal_distribution<double> randDist(RANDOM_MEAN, RANDOM_STDEV); //mean followed by stdiv
template <typename T>
T getRand(){return static_cast<T>(randDist(randEng));}

template <typename T>
__host__
void test(){
    T gravity = static_cast<T>(9.81);
    dim3 dimms(grid::SUGGESTED_THREADS,1,1);
    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T,1>();
    
    // load q,qd,u
    for(int j = 0; j < grid::NUM_JOINTS; j++){
        hd_data->h_q_qd_u[j] = getRand<double>(); 
        hd_data->h_q_qd_u[j+grid::NUM_JOINTS] = getRand<double>(); 
        hd_data->h_q_qd_u[j+2*grid::NUM_JOINTS] = getRand<double>();
    }
    gpuErrchk(cudaMemcpy(hd_data->d_q_qd_u,hd_data->h_q_qd_u,3*grid::NUM_JOINTS*sizeof(T),cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());

    printf("q,qd,u\n");
    printMat<T,1,grid::NUM_JOINTS>(hd_data->h_q_qd_u,1);
    printMat<T,1,grid::NUM_JOINTS>(&hd_data->h_q_qd_u[grid::NUM_JOINTS],1);
    printMat<T,1,grid::NUM_JOINTS>(&hd_data->h_q_qd_u[2*grid::NUM_JOINTS],1);

    printf("c via inverse dynamics\n");
    grid::inverse_dynamics<T,false,false>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
    printMat<T,1,grid::NUM_JOINTS>(hd_data->h_c,1);

    printf("Minv via direct minv\n");
    grid::direct_minv<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
    printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(hd_data->h_Minv,grid::NUM_JOINTS);

    printf("qdd via forward dynamics\n");
    grid::forward_dynamics<T>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
    printMat<T,1,grid::NUM_JOINTS>(hd_data->h_qdd,1);

    printf("qdd via aba\n");
    grid::aba<T>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
    printMat<T,1,grid::NUM_JOINTS>(hd_data->h_qdd,1);

    printf("M via crba\n");
    grid::crba<T>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
    printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(hd_data->h_M,grid::NUM_JOINTS);

    grid::inverse_dynamics_gradient<T,true,false>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
    printf("dc_dq\n");
    printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(hd_data->h_dc_du,grid::NUM_JOINTS);
    printf("dc_dqd\n");
    printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(&hd_data->h_dc_du[grid::NUM_JOINTS*grid::NUM_JOINTS],grid::NUM_JOINTS);

    grid::forward_dynamics_gradient<T,false>(hd_data,d_robotModel,gravity,1,dim3(1,1,1),dimms,streams);
    printf("df_dq\n");
    printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(hd_data->h_df_du,grid::NUM_JOINTS);
    printf("df_dqd\n");
    printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(&hd_data->h_df_du[grid::NUM_JOINTS*grid::NUM_JOINTS],grid::NUM_JOINTS);

    printf("eePos\n");
    grid::end_effector_pose<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
    printMat<T,1,6*grid::NUM_EES>(hd_data->h_eePos,1);

    // printf("eePos - for panda_grasptarget_hand\n");
    // grid::end_effector_pose_panda_grasptarget_hand<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
    // printMat<T,1,6*grid::NUM_EES>(hd_data->h_eePos,1);

    printf("deePos\n");
    grid::end_effector_pose_gradient<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
    for(int ee=0; ee < grid::NUM_EES; ee++){
        printf("deePos[%d]\n",ee);
        printMat<T,6,grid::NUM_JOINTS>(&hd_data->h_deePos[ee*6*grid::NUM_JOINTS],6);
    }

    // printf("deePos - for panda_grasptarget_hand\n");
    // grid::end_effector_pose_gradient_panda_grasptarget_hand<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
    // printMat<T,6,grid::NUM_JOINTS>(&hd_data->h_deePos[0],6);

    printf("d2eePos\n");
    grid::end_effector_pose_gradient_hessian<T,false>(hd_data,d_robotModel,1,dim3(1,1,1),dimms,streams);
    for(int ee=0; ee < grid::NUM_EES; ee++){
        printf("deePos[%d]\n",ee);
        printMat<T,6,grid::NUM_JOINTS>(&hd_data->h_deePos[ee*6*grid::NUM_JOINTS],6);
        printf("d2eePos[%d]\n",ee);
        for (int i=0; i < 6; i++){
            int offset = ee*6*grid::NUM_JOINTS*grid::NUM_JOINTS + i*grid::NUM_JOINTS*grid::NUM_JOINTS;
            printf("[%d]\n",i); printMat<T,grid::NUM_JOINTS,grid::NUM_JOINTS>(&hd_data->h_d2eePos[offset],grid::NUM_JOINTS);
        }
    }
    grid::close_grid<T>(streams,d_robotModel,hd_data);
}

int main(void){
    test<float>(); return 0;
}
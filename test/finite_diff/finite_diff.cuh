// #include "grid.cuh"
#include "/home/a2rlab4/GRiDBenchmarks/GRiD/grid.cuh"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

using namespace grid; 

template <typename T, int O_R, int O_C, int I_N>
 __device__ void finite_diff(T *input, T *output, int (*func)(T* input, T* output, const grid::robotModel<T> *d_robotMode, const T gravity), T eps, const grid::robotModel<T> *d_robotModel, const T gravity){
// void finite_diff(T *input, T *output, int (*func)(T* input, T* output, const grid::robotModel<T> *d_robotModel*, const T gravity), T eps){
    for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < I_N; ind += blockDim.x*blockDim.y){
        T *val_plus[O_R * O_C];
        T *val_minus[O_R * O_C];

        unsigned input_ind = ind/I_N;

        // T *input_plus = input;
        // T *input_minus = input;
        // input_plus[input_ind] += eps;
        // input_minus[input_ind] -= eps;

        input->h_q_qd_u[input_ind] += eps;
        val_plus[0] = func(input, output, d_robotModel,gravity);
        input->h_q_qd_u[input_ind] -= (2*eps);
        val_minus[0] = func(input, output, d_robotModel,gravity);

        for(int rc = 0; rc < (O_R*O_C); rc++){
            output[(ind*O_C*O_R) + rc] = (val_plus[rc] - val_minus[rc])/ (2* eps);
        }
        printMat<T,7,7>(&output[0],7);
    }
}

template <typename T, int O_R, int O_C, int I_N>
__device__ void finite_diff_aba(T *s_q_qd_tau, T *s_fdoutput, T eps, const grid::robotModel<T> *d_robotModel, const T gravity){
    __shared__ T s_qdd_plus[O_R];
    __shared__ T s_qdd_minus[O_R];
    T *s_tau = &s_q_qd_tau[2 * O_R]; 
    T *qqd_plus = s_q_qd_tau;
    T *qqd_minus = s_q_qd_tau;
    
    __shared__ T s_va[12*O_R];
    extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
    // load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
    for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < I_N*2; ind += blockDim.x*blockDim.y){
  
        if(threadIdx.x == 0 && blockIdx.x == 0){
            qqd_plus[ind] += eps;
        }

        grid::load_update_XImats_helpers<T>(s_XImats, &qqd_plus[0], d_robotModel, s_temp);
        __syncthreads();
        grid::aba_inner<T>(s_qdd_plus, s_va, &qqd_plus[0], &qqd_plus[7], s_tau, s_XImats, s_temp, gravity);
        __syncthreads();

        if(threadIdx.x == 0 && blockIdx.x == 0){
            qqd_minus[ind] -= (2*eps);
        } 
        // __syncthreads();
        grid::load_update_XImats_helpers<T>(s_XImats, &qqd_minus[0], d_robotModel, s_temp);
        __syncthreads();
        grid::aba_inner<T>(s_qdd_minus, s_va, &qqd_minus[0], &qqd_minus[7], s_tau, s_XImats, s_temp, gravity);
        __syncthreads();

       
        if(threadIdx.x == 0 && blockIdx.x == 0){
            qqd_minus[ind] += (eps);
        }
         __syncthreads();

        for(int i = 0; i < (O_R); i++){
            s_fdoutput[(ind*O_R) + i] = (s_qdd_plus[i] - s_qdd_minus[i])/ (2* eps);
        }
         __syncthreads();

    }

       __syncthreads();
         if(threadIdx.x == 0 && blockIdx.x == 0){
            printf("s_fdoutput \n");
            printMat<T,7,14>(&s_fdoutput[0],7);
         }
        __syncthreads();

}

template <typename T, int O_R, int O_C, int I_N>
__device__ void finite_diff_for_ddp(T *s_q_qd_tau, T *s_fdoutput, T eps, const grid::robotModel<T> *d_robotModel, const T gravity){
    __shared__ T s_df_du_plus[O_R*O_R*3*21];
    __shared__ T s_df_du_minus[O_R*O_R*3*21];
    __shared__ T s_qdd_plus[O_R];
    __shared__ T s_qdd_minus[O_R];
    T *s_tau = &s_q_qd_tau[2 * O_R]; 
    T *qqd_plus = s_q_qd_tau;
    T *qqd_minus = s_q_qd_tau;
    
    __shared__ T s_vaf[12*O_R];
    __shared__ T s_va[12*O_R];
    extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];
    // load_update_XImats_helpers<T>(s_XImats, s_q, d_robotModel, s_temp);
    for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < I_N*3; ind += blockDim.x*blockDim.y){
        //      __syncthreads();
        //  if(threadIdx.x == 0 && blockIdx.x == 0){
        //     printf("ind %d \n", ind);
        //  }
        // __syncthreads();
        
        qqd_plus[ind] += eps;
        // if(threadIdx.x == 0 && blockIdx.x == 0){
        //     // qqd_plus[ind] += eps;
        //     printf("qqd_plus[ind] %f \n", qqd_plus[ind]);
        //     printMat<T,1,7>(&qqd_plus[0],1);
        // }

        grid::load_update_XImats_helpers<T>(s_XImats, &qqd_plus[0], d_robotModel, s_temp);
        __syncthreads();
        //  void inverse_dynamics_gradient_inner(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_vaf, T *s_XImats, T *s_temp, const T gravity)
        // grid::inverse_dynamics_gradient_inner<T>(s_dc_du_plus, &qqd_plus[0], &qqd_plus[O_R], s_vaf, s_XImats, s_temp, gravity);
        grid::aba_inner<T>(s_qdd_plus, s_va, &qqd_plus[0], &qqd_plus[7], &qqd_plus[14], s_XImats, s_temp, gravity);
        // if(threadIdx.x == 0 && blockIdx.x == 0){
        //     // qqd_plus[ind] += eps;
        //     printf("s_qdd_plus\n");
        //     printMat<T,1,7>(&s_qdd_plus[0],1);
        // }
        // grid::fdsva_device(&qqd_plus[0],  &qqd_plus[7], s_qdd_plus, const T *s_tau,const robotModel<T> *d_robotModel, const T gravity)
        grid::fdsva_inner<T>(&s_df_du_plus[21*7*ind], &qqd_plus[0], &qqd_plus[7], s_qdd_plus, &qqd_plus[14], s_XImats, s_temp, gravity);
        // (T *s_fddq_fddqd_fddt, T *s_q, T *s_qd, const T *s_qdd, const T *s_tau, T *s_XImats, T *s_temp, const T gravity) 
        //   __syncthreads();
        //  if(threadIdx.x == 0 && blockIdx.x == 0){
        //     // if(ind < 7){ 
        //     printf("s_df_du_plus %d \n", ind);
        //     printMat<T,O_R,O_R>(&s_df_du_plus[21*7*ind],O_R);
        //     // }
        //  }
        // __syncthreads();
        // __syncthreads();

        qqd_minus[ind] -= (2*eps);
        
        // if(threadIdx.x == 0 && blockIdx.x == 0){
        //     // if(ind < 7){ 
        //     // qqd_minus[ind] -= (2*eps);
        //     printf("qqd_minus\n");
        //     printMat<T,1,7>(&qqd_minus[0],1);
        //     // }
        // } 
        __syncthreads();
        grid::load_update_XImats_helpers<T>(s_XImats, &qqd_minus[0], d_robotModel, s_temp);
        __syncthreads();
        // grid::inverse_dynamics_gradient_inner<T>(s_dc_du_minus, &qqd_minus[0], &qqd_minus[O_R], s_vaf, s_XImats, s_temp, gravity);
        grid::aba_inner<T>(s_qdd_minus, s_va, &qqd_minus[0], &qqd_minus[7], &qqd_minus[14], s_XImats, s_temp, gravity);
        //    if(threadIdx.x == 0 && blockIdx.x == 0){
        //     // qqd_plus[ind] += eps;
        //     printf("s_qdd_minus\n");
        //     printMat<T,1,7>(&s_qdd_minus[0],1);
        // }
        grid::fdsva_inner<T>(&s_df_du_minus[21*7*ind], &qqd_minus[0], &qqd_minus[7], s_qdd_minus, &qqd_minus[14], s_XImats, s_temp, gravity);
         
        //  if(threadIdx.x == 0 && blockIdx.x == 0){
        //     // if(ind < 7){ 

        //     printf("s_df_du_minus %d \n", ind);
        //     printMat<T,O_R,O_R>(&s_df_du_minus[21*7*ind],O_R);
        //     // }
        // }
        // __syncthreads();

        // if(threadIdx.x == 0 && blockIdx.x == 0){
            qqd_minus[ind] += eps;
        // }
         __syncthreads();

    }
    for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R*3*21; ind += blockDim.x*blockDim.y){
        s_fdoutput[ind] = (s_df_du_plus[ind] - s_df_du_minus[ind])/ (2* eps);

    }

         __syncthreads();
        if(threadIdx.x == 0 && blockIdx.x == 0){
            for(int i=0; i<21; i++){
                printf("s_fdoutput %d\n", i);
                printMat<T,O_R,21>(&s_fdoutput[21*7*i],O_R);
            }
         }
        __syncthreads();

}
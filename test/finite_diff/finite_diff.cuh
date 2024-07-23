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
__device__ void finite_diff_fdsva(T *s_q_qd_tau, T *s_fdoutput, T eps, const grid::robotModel<T> *d_robotModel, const T gravity){
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

template <typename T, int O_R, int O_C, int I_N>
__device__ void finite_diff_for_ddp(T *s_q_qd_tau, T *s_df2, T eps, const grid::robotModel<T> *d_robotModel, const T gravity){
        __shared__ T s_Minv[49];
        __shared__ T s_qdd[49];
        __shared__ T s_dc_du_plus[98*7];
        __shared__ T s_vaf_plus[84];
        __shared__ T s_qdd1_plus[7];
        __shared__ T s_M_plus[49*7];
        __shared__ T s_dc_du_minus[98*7];
        __shared__ T s_vaf_minus[84];
        __shared__ T s_qdd1_minus[7];
        __shared__ T s_M_minus[49*7];
        __shared__ T s_di_du[1372];
        __shared__ T s_df_du[147];
        __shared__ T s_XImats_plus[504];
        __shared__ T s_XImats_minus[504];
        extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];

        // T *s_di_dqq = s_di_du; T *s_di_dqdqd = &s_di_du[343]; T *s_di_dqqd = &s_di_du[686]; T *s_dm_dq = &s_di_du[1029];
        // grid::load_update_XImats_helpers<T>(s_XImats, &s_q_qd_tau[0], d_robotModel, s_temp);
        // T *s_df_dq = s_df_du; T *s_df_dqd = &s_df_du[49]; T *s_df_tau = &s_df_du[98];
        // grid::direct_minv_inner<T>(s_Minv, &s_q_qd_tau[0], s_XImats, s_temp);
        // grid::forward_dynamics_finish<T>(s_qdd, &s_q_qd_tau[14], s_temp, s_Minv);
        // grid::fdsva_inner<T>(s_df_du, &s_q_qd_tau[0], &s_q_qd_tau[7], s_qdd, &s_q_qd_tau[14], s_XImats, s_temp, gravity);
  
        T *s_tau = &s_q_qd_tau[2 * O_R]; 
        T *qqd_plus = s_q_qd_tau;
        T *qqd_minus = s_q_qd_tau;

        // solves for di_dq
       for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < I_N; ind += blockDim.x*blockDim.y){
        qqd_plus[ind] += eps; 

    //   if(threadIdx.x == 0 && blockIdx.x == 0){
    //     printf("qqd_plus[%d] = \n", ind);
    //     printMat<T,1,7>(&qqd_plus[0],1);

    //   }

        load_update_XImats_helpers<T>(s_XImats_plus, &qqd_plus[0], d_robotModel, s_temp);
        __syncthreads();

        inverse_dynamics_inner<T>(s_temp, s_vaf_plus, &qqd_plus[0], &qqd_plus[7], s_XImats_plus, s_temp, gravity);
        // direct_minv_inner<T>(s_Minv, &qqd_plus[0], s_XImats, s_temp);
        forward_dynamics_finish<T>(s_qdd1_plus, s_tau, s_temp, s_Minv);
        //   if(threadIdx.x == 0 && blockIdx.x == 0){
        //     printf("s_Minv \n");
        //     printMat<T,7,7>(&s_Minv[0],7);

        //  }
        inverse_dynamics_inner_vaf<T>(s_vaf_plus, &qqd_plus[0], &qqd_plus[7], s_qdd1_minus, s_XImats_plus, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(&s_dc_du_plus[98*ind], &qqd_plus[0], &qqd_plus[7], s_vaf_plus, s_XImats_plus, s_temp, gravity);
        // if(threadIdx.x == 0 && blockIdx.x == 0){
        //     printf("s_dc_du_plus[%d] = \n", ind);
        //     printMat<T,7,7>(&s_dc_du_plus[ind*98],7);
        //     printMat<T,7,7>(&s_dc_du_plus[ind*98 + 49],7);

        // }

        qqd_minus[ind] -= (2*eps);
    //     if(threadIdx.x == 0 && blockIdx.x == 0){
    //     printf("qqd_plus[%d] = \n", ind);
    //     printMat<T,1,7>(&qqd_plus[0],1);

    //   }
        load_update_XImats_helpers<T>(s_XImats_minus, &qqd_minus[0], d_robotModel, s_temp);
        __syncthreads();

        inverse_dynamics_inner<T>(s_temp, s_vaf_plus, &qqd_minus[0], &qqd_minus[7], s_XImats_minus, s_temp, gravity);
        // direct_minv_inner<T>(s_Minv, &qqd_plus[0], s_XImats, s_temp);
        forward_dynamics_finish<T>(s_qdd1_minus, s_tau, s_temp, s_Minv);
        inverse_dynamics_inner_vaf<T>(s_vaf_minus, &qqd_minus[0], &qqd_minus[7], s_qdd1_minus, s_XImats_minus, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(&s_dc_du_minus[98*ind], &qqd_minus[0], &qqd_minus[7], s_vaf_minus, s_XImats_minus, s_temp, gravity);

    //       if(threadIdx.x == 0 && blockIdx.x == 0){
    //         printf("s_dc_du_minus[%d] = \n", ind);
    //         printMat<T,7,7>(&s_dc_du_minus[ind*98],7);
    //         printMat<T,7,7>(&s_dc_du_minus[ind*98 + 49],7);

    //   }
        qqd_minus[ind] += eps;
        // if(threadIdx.x == 0 && blockIdx.x == 0){
        // printf("qqd_minus[%d] = \n", ind);
        // printMat<T,1,7>(&qqd_minus[0],1);
        // printf("qqd_plus[%d] = \n", ind);
        // printMat<T,1,7>(&qqd_plus[0],1);

        // }


       }
       __syncthreads();
    //              __syncthreads();
    //   if(threadIdx.x == 0 && blockIdx.x == 0){

    //     for(int i=0; i<7; i++){
    //          printf("s_dc_du_plus[%d] = \n", i*49);
    //         printMat<T,7,7>(&s_dc_du_plus[i*49],7);
    //         printf("s_dc_du_minus[%d] = \n", i*49);
    //         printMat<T,7,7>(&s_dc_du_minus[i*49],7);
    //     }
    //   }

        for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[ind] = (s_dc_du_plus[ind] - s_dc_du_minus[ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[49 + ind] = (s_dc_du_plus[98 +ind] - s_dc_du_minus[98 +ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[98+ ind] = (s_dc_du_plus[196+ind] - s_dc_du_minus[196+ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[147 + ind] = (s_dc_du_plus[294 +ind] - s_dc_du_minus[294 +ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[196 + ind] = (s_dc_du_plus[392 +ind] - s_dc_du_minus[392 +ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[245 + ind] = (s_dc_du_plus[490 +ind] - s_dc_du_minus[490 +ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[294+ ind] = (s_dc_du_plus[588+ind] - s_dc_du_minus[ 588+ind])/ (eps* 2);
            // }
        }

           __syncthreads();
        // solves for di_dqd
        for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < I_N; ind += blockDim.x*blockDim.y){
        qqd_plus[7 + ind] += eps; 

        load_update_XImats_helpers<T>(s_XImats_plus, &qqd_plus[0], d_robotModel, s_temp);
        __syncthreads();

        inverse_dynamics_inner<T>(s_temp, s_vaf_plus, &qqd_plus[0], &qqd_plus[7], s_XImats_plus, &s_temp[7], gravity);
        // direct_minv_inner<T>(s_Minv, &qqd_plus[0], s_XImats, s_temp);
        forward_dynamics_finish<T>(s_qdd1_plus, s_tau, s_temp, s_Minv);
        inverse_dynamics_inner_vaf<T>(s_vaf_plus, &qqd_plus[0], &qqd_plus[7], s_qdd1_minus, s_XImats_plus, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(&s_dc_du_plus[98*ind], &qqd_plus[0], &qqd_plus[7], s_vaf_plus, s_XImats_plus, s_temp, gravity);

        qqd_minus[7 + ind] -= (2*eps);
        load_update_XImats_helpers<T>(s_XImats_minus, &qqd_minus[0], d_robotModel, s_temp);
        __syncthreads();

        inverse_dynamics_inner<T>(s_temp, s_vaf_plus, &qqd_minus[0], &qqd_minus[7], s_XImats_minus, &s_temp[7], gravity);
        // direct_minv_inner<T>(s_Minv, &qqd_plus[0], s_XImats, s_temp);
        forward_dynamics_finish<T>(s_qdd1_minus, s_tau, s_temp, s_Minv);
        inverse_dynamics_inner_vaf<T>(s_vaf_minus, &qqd_minus[0], &qqd_minus[7], s_qdd1_minus, s_XImats_minus, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(&s_dc_du_minus[98*ind], &qqd_minus[0], &qqd_minus[7], s_vaf_minus, s_XImats_minus, s_temp, gravity);


        qqd_minus[7 + ind] += eps;

       }
       __syncthreads();
                      __syncthreads();
    //   if(threadIdx.x == 0 && blockIdx.x == 0){

    //     for(int i=0; i<7; i++){
    //         printf("s_dc_du_plus[%d] = \n", i*49);
    //         printMat<T,7,7>(&s_dc_du_plus[i*49],7);
    //         printf("s_dc_du_minus[%d] = \n", i*49);
    //         printMat<T,7,7>(&s_dc_du_minus[i*49],7);
    //     }
    //   }

    //     for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R*O_R; ind += blockDim.x*blockDim.y){
    // //   if(threadIdx.x == 0 && blockIdx.x == 0){
    // //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
    // //   }

    //       s_di_du[343 + ind] = (s_dc_du_plus[ind] - s_dc_du_minus[ind])/ (2* eps);
    //     }
     for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[343+ ind] = (s_dc_du_plus[49 +ind] - s_dc_du_minus[49+ ind])/ (eps* 2);
            // s_di_du[343+ ind] = 3;
            
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[343+ 49 + ind] = (s_dc_du_plus[147 +ind] - s_dc_du_minus[147 +ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[343+ 98+ ind] = (s_dc_du_plus[245+ind] - s_dc_du_minus[245+ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[343+ 147 + ind] = (s_dc_du_plus[343 +ind] - s_dc_du_minus[343 +ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[343+ 196 + ind] = (s_dc_du_plus[441 +ind] - s_dc_du_minus[441 +ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[343+ 245 + ind] = (s_dc_du_plus[539 +ind] - s_dc_du_minus[539 +ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[343+ 294+ ind] = (s_dc_du_plus[637+ind] - s_dc_du_minus[ 637+ind])/ (eps* 2);
            // }
        }
       __syncthreads();

        // solves for di_dqd_dq

        for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < I_N; ind += blockDim.x*blockDim.y){
        qqd_plus[ind] += eps; 
        // qqd_plus[7 + ind] += eps; 

        load_update_XImats_helpers<T>(s_XImats_plus, &qqd_plus[0], d_robotModel, s_temp);
        __syncthreads();

        inverse_dynamics_inner<T>(s_temp, s_vaf_plus, &qqd_plus[0], &qqd_plus[7], s_XImats_plus, &s_temp[7], gravity);
        // direct_minv_inner<T>(s_Minv, &qqd_plus[0], s_XImats, s_temp);
        forward_dynamics_finish<T>(s_qdd1_plus, s_tau, s_temp, s_Minv);
        inverse_dynamics_inner_vaf<T>(s_vaf_plus, &qqd_plus[0], &qqd_plus[7], s_qdd1_minus, s_XImats_plus, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(&s_dc_du_plus[98*ind], &qqd_plus[0], &qqd_plus[7], s_vaf_plus, s_XImats_plus, s_temp, gravity);

        qqd_minus[ind] -= (2*eps);
        // qqd_minus[7 + ind] -= (2*eps);
        load_update_XImats_helpers<T>(s_XImats_minus, &qqd_minus[0], d_robotModel, s_temp);
        __syncthreads();

        inverse_dynamics_inner<T>(s_temp, s_vaf_plus, &qqd_minus[0], &qqd_minus[7], s_XImats_minus, &s_temp[7], gravity);
        // direct_minv_inner<T>(s_Minv, &qqd_plus[0], s_XImats, s_temp);
        forward_dynamics_finish<T>(s_qdd1_minus, s_tau, s_temp, s_Minv);
        inverse_dynamics_inner_vaf<T>(s_vaf_minus, &qqd_minus[0], &qqd_minus[7], s_qdd1_minus, s_XImats_minus, s_temp, gravity);
        inverse_dynamics_gradient_inner<T>(&s_dc_du_minus[98*ind], &qqd_minus[0], &qqd_minus[7], s_vaf_minus, s_XImats_minus, s_temp, gravity);


        qqd_minus[ind] += eps;
        // qqd_minus[7 + ind] += eps;

       }
       __syncthreads();
    //                   __syncthreads();
    //   if(threadIdx.x == 0 && blockIdx.x == 0){

    //     for(int i=0; i<7; i++){
    //          printf("s_dc_du_plus[%d] = \n", i*49);
    //         printMat<T,7,7>(&s_dc_du_plus[i*49],7);
    //         printf("s_dc_du_minus[%d] = \n", i*49);
    //         printMat<T,7,7>(&s_dc_du_minus[i*49],7);
    //     }
    //   }
        __syncthreads();

        // for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R*O_R; ind += blockDim.x*blockDim.y){
        //   s_di_du[686 + ind] = (s_dc_du_plus[ind] - s_dc_du_minus[ind])/ (2* eps);
        // }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[686 +ind] = (s_dc_du_plus[49 +ind] - s_dc_du_minus[49 +ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[686 +49 + ind] = (s_dc_du_plus[147 +ind] - s_dc_du_minus[147 +ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[686 +98+ ind] = (s_dc_du_plus[245+ind] - s_dc_du_minus[245+ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[686 +147 + ind] = (s_dc_du_plus[343 +ind] - s_dc_du_minus[343 +ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[686 +196 + ind] = (s_dc_du_plus[441 +ind] - s_dc_du_minus[441 +ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[686 +245 + ind] = (s_dc_du_plus[539 +ind] - s_dc_du_minus[539 +ind])/ (eps* 2);
            // }
        }
          for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R; ind += blockDim.x*blockDim.y){
            // if (ind % 7 != 0){
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("ind = %d and s_dc_du_plus[ind] = %f and s_dc_du_minus[ind] = %f \n", ind, s_dc_du_plus[ind], s_dc_du_minus[ind]);
            // }
            s_di_du[686 +294+ ind] = (s_dc_du_plus[637+ind] - s_dc_du_minus[ 637+ind])/ (eps* 2);
            // }
        }

           __syncthreads();
       __syncthreads();
    
        // solves for dM_dq
        for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < I_N; ind += blockDim.x*blockDim.y){
        qqd_plus[ind] += eps; 

        load_update_XImats_helpers<T>(s_XImats_plus, &qqd_plus[0], d_robotModel, s_temp);
        // __syncthreads();

        // grid::direct_minv_inner<T>(&s_Minv_plus[ind*49], &qqd_plus[0], s_XImats, s_temp);
        crba_inner(&s_M_plus[ind*49], &qqd_plus[0], &qqd_plus[7], s_XImats_plus, s_temp, gravity);
        // __syncthreads();


        qqd_minus[ind] -= (2*eps);
        load_update_XImats_helpers<T>(s_XImats_minus, &qqd_minus[0], d_robotModel, s_temp);
        // __syncthreads();

        // grid::direct_minv_inner<T>(&s_Minv_minus[ind*49], &qqd_plus[0], s_XImats, s_temp);
        crba_inner(&s_M_minus[ind*49], &qqd_minus[0], &qqd_minus[7], s_XImats_minus, s_temp, gravity);
        // __syncthreads();


        qqd_minus[ind] += eps;

       }
        __syncthreads();

    //    __syncthreads();
    //       if(threadIdx.x == 0 && blockIdx.x == 0){

    //     for(int i=0; i<7; i++){
    //          printf("s_M_plus[%d] = \n", i*49);
    //         printMat<T,7,7>(&s_M_plus[i*49],7);
    //         printf("s_M_minus[%d] = \n", i*49);
    //         printMat<T,7,7>(&s_M_minus[i*49],7);
    //     }
    //   }
    //                   __syncthreads();


        for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R*O_R; ind += blockDim.x*blockDim.y){
          s_di_du[1029 + ind] = (s_M_plus[ind] - s_M_minus[ind])/ (2* eps);
            // if(threadIdx.x == 0 && blockIdx.x == 0){
            //     printf("s_di_du[1029 + ind]%d  = %f bc s_M_plus[ind] = %f and s_M_minus[ind] = %f\n", ind,s_di_du[1029 +ind], s_M_plus[ind], s_M_minus[ind]);
            // }

        }

        if(threadIdx.x == 0 && blockIdx.x == 0){
            for(int i=0; i<28; i++){
                printf("s_di_du %d\n", i);
                printMat<T,O_R,7>(&s_di_du[49*i],7);
            }
         }
        __syncthreads();

        load_update_XImats_helpers<T>(s_XImats, &s_q_qd_tau[0], d_robotModel, s_temp);
        T *s_df_dq = s_df_du; T *s_df_dqd = &s_df_du[49]; T *s_df_tau = &s_df_du[98];
        direct_minv_inner<T>(s_Minv, &s_q_qd_tau[0], s_XImats, s_temp);
        forward_dynamics_finish<T>(s_qdd, &s_q_qd_tau[14], s_temp, s_Minv);
        fdsva_inner<T>(s_df_du, &s_q_qd_tau[0], &s_q_qd_tau[7], s_qdd, &s_q_qd_tau[14], s_XImats, s_temp, gravity);

         __syncthreads();
        T *s_dm_dq = &s_di_du[1029];


        T *dM_dqxfd_dq = s_temp;
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 343; ind += blockDim.x*blockDim.y){
            int page = ind / 49;
            int row = ind % 7;
            int col = ind % 49 / 7;
            dM_dqxfd_dq[ind] = dot_prod<T,7,7,1>(&s_dm_dq[49*page + row], &s_df_dq[7*col]);
        }
        __syncthreads();
        T *rot_dM_dqxfd_dqd = &s_temp[0];
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
            int page = 0;
            int row = ind % 7;
            int col = ind / 7;
            rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
            int page = 1;
            int row = ind % 7;
            int col = ind / 7;
            rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
            int page = 2;
            int row = ind % 7;
            int col = ind / 7;
            rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
            int page = 3;
            int row = ind % 7;
            int col = ind / 7;
            rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
            int page = 4;
            int row = ind % 7;
            int col = ind / 7;
            rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
            int page = 5;
            int row = ind % 7;
            int col = ind / 7;
            rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
            int page = 6;
            int row = ind % 7;
            int col = ind / 7;
            rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 343; ind += blockDim.x*blockDim.y){
            s_df2[ind] = s_di_du[ind] + dM_dqxfd_dq[ind] + rot_dM_dqxfd_dqd[ind];
        }
        __syncthreads();
        T *dM_dqxfd_dqd = s_temp;
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 343; ind += blockDim.x*blockDim.y){
            int page = ind / 49;
            int row = ind % 7;
            int col = ind % 49 / 7;
            dM_dqxfd_dqd[ind] = dot_prod<T,7,7,1>(&s_dm_dq[49*page + row], &s_df_dqd[7*col]);
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 343; ind += blockDim.x*blockDim.y){
            s_df2[343+ ind] = s_di_du[343 + ind] + dM_dqxfd_dqd[ind];
            s_df2[686+ ind] = s_di_du[686 + ind];
        }
        __syncthreads();
        s_Minv[1] = s_Minv[7];
        s_Minv[2] = s_Minv[14];
        s_Minv[9] = s_Minv[15];
        s_Minv[3] = s_Minv[21];
        s_Minv[10] = s_Minv[22];
        s_Minv[17] = s_Minv[23];
        s_Minv[4] = s_Minv[28];
        s_Minv[11] = s_Minv[29];
        s_Minv[18] = s_Minv[30];
        s_Minv[25] = s_Minv[31];
        s_Minv[5] = s_Minv[35];
        s_Minv[12] = s_Minv[36];
        s_Minv[19] = s_Minv[37];
        s_Minv[26] = s_Minv[38];
        s_Minv[33] = s_Minv[39];
        s_Minv[6] = s_Minv[42];
        s_Minv[13] = s_Minv[43];
        s_Minv[20] = s_Minv[44];
        s_Minv[27] = s_Minv[45];
        s_Minv[34] = s_Minv[46];
        s_Minv[41] = s_Minv[47];
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 343; ind += blockDim.x*blockDim.y){
            int page = ind / 49;
            int row = ind % 7;
            int col = ind % 49 / 7;
            s_df2[1029+ ind] = dot_prod<T,7,7,1>(&s_dm_dq[49*page + row], &s_Minv[7*col]);
        }
        __syncthreads();
        for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1372; ind += blockDim.x*blockDim.y){
            int page = ind / 49;
            int row = ind % 7;
            int col = ind % 49 / 7;
            s_df2[ind] = dot_prod<T,7,7,1>(&s_Minv[row], &s_df2[49*page + 7*col]);
            s_df2[ind] *= (-1);
        }
        __syncthreads();

  
             __syncthreads();
      if(threadIdx.x == 0 && blockIdx.x == 0){

        for(int i=0; i<28; i++){
            if (i == 0){
                printf("solving for fd_dq \n");
                            }
            if (i == 7){
                printf("solving for fd_dqd \n");
            }
            if (i == 14){
                printf("solving for fd_dqd_dq \n");
            }
            if (i == 21){
                printf("solving for fd_dtau_dq \n");
            }
            printf("s_df2[%d] = \n", i*49);
            printMat<T,7,7>(&s_df2[i*49],7);
        }
      }

}

// template <typename T, int O_R, int O_C, int I_N>
// __device__ void finite_diff_for_ddp(T *s_q_qd_tau, T *s_df2, T eps, const grid::robotModel<T> *d_robotModel, const T gravity){
//         __shared__ T s_Minv[49];
//         __shared__ T s_qdd[49];
//         __shared__ T s_dc_du_plus[98*7];
//         __shared__ T s_vaf_plus[84];
//         __shared__ T s_qdd1_plus[7];
//         __shared__ T s_M_plus[49*7];
//         __shared__ T s_dc_du_minus[98*7];
//         __shared__ T s_vaf_minus[84];
//         __shared__ T s_qdd1_minus[7];
//         __shared__ T s_M_minus[49*7];
//         __shared__ T s_di_du[1372];
//         __shared__ T s_df_du[147];
//         extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[504];

//         // T *s_di_dqq = s_di_du; T *s_di_dqdqd = &s_di_du[343]; T *s_di_dqqd = &s_di_du[686]; T *s_dm_dq = &s_di_du[1029];
//         // grid::load_update_XImats_helpers<T>(s_XImats, &s_q_qd_tau[0], d_robotModel, s_temp);
//         // T *s_df_dq = s_df_du; T *s_df_dqd = &s_df_du[49]; T *s_df_tau = &s_df_du[98];
//         // grid::direct_minv_inner<T>(s_Minv, &s_q_qd_tau[0], s_XImats, s_temp);
//         // grid::forward_dynamics_finish<T>(s_qdd, &s_q_qd_tau[14], s_temp, s_Minv);
//         // grid::fdsva_inner<T>(s_df_du, &s_q_qd_tau[0], &s_q_qd_tau[7], s_qdd, &s_q_qd_tau[14], s_XImats, s_temp, gravity);
  
//         // fd of idsva
//         T *s_tau = &s_q_qd_tau[2 * O_R]; 
//         T *qqd_plus = s_q_qd_tau;
//         T *qqd_minus = s_q_qd_tau;

//         // solves for di_dq
//        for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < I_N; ind += blockDim.x*blockDim.y){
//         qqd_plus[ind] += eps; 

//         grid::load_update_XImats_helpers<T>(s_XImats, &qqd_plus[0], d_robotModel, s_temp);
//         __syncthreads();

//         inverse_dynamics_inner<T>(s_temp, s_vaf_plus, &qqd_plus[0], &qqd_plus[7], s_XImats, &s_temp[7], gravity);
//         // direct_minv_inner<T>(s_Minv, &qqd_plus[0], s_XImats, s_temp);
//         forward_dynamics_finish<T>(s_qdd1_plus, s_tau, s_temp, s_Minv);
//         //   if(threadIdx.x == 0 && blockIdx.x == 0){
//         //     printf("s_Minv \n");
//         //     printMat<T,7,7>(&s_Minv[0],7);

//         //  }
//         inverse_dynamics_inner_vaf<T>(s_vaf_plus, &qqd_plus[0], &qqd_plus[7], s_qdd1_minus, s_XImats, s_temp, gravity);
//         inverse_dynamics_gradient_inner<T>(&s_dc_du_plus[49*ind], &qqd_plus[0], &qqd_plus[7], s_vaf_plus, s_XImats, s_temp, gravity);

//         qqd_minus[ind] -= (2*eps);
//         grid::load_update_XImats_helpers<T>(s_XImats, &qqd_minus[0], d_robotModel, s_temp);
//         __syncthreads();

//         inverse_dynamics_inner<T>(s_temp, s_vaf_plus, &qqd_minus[0], &qqd_minus[7], s_XImats, &s_temp[7], gravity);
//         // direct_minv_inner<T>(s_Minv, &qqd_plus[0], s_XImats, s_temp);
//         forward_dynamics_finish<T>(s_qdd1_minus, s_tau, s_temp, s_Minv);
//         inverse_dynamics_inner_vaf<T>(s_vaf_minus, &qqd_minus[0], &qqd_minus[7], s_qdd1_minus, s_XImats, s_temp, gravity);
//         inverse_dynamics_gradient_inner<T>(&s_dc_du_minus[49*ind], &qqd_minus[0], &qqd_minus[7], s_vaf_minus, s_XImats, s_temp, gravity);


//         qqd_minus[ind] += eps;

//        }
//        __syncthreads();
//                  __syncthreads();
//       if(threadIdx.x == 0 && blockIdx.x == 0){

//         for(int i=0; i<7; i++){
//              printf("s_dc_du_plus[%d] = \n", i*49);
//             printMat<T,7,7>(&s_dc_du_plus[i*49],7);
//             printf("s_dc_du_minus[%d] = \n", i*49);
//             printMat<T,7,7>(&s_dc_du_minus[i*49],7);
//         }
//       }

//         for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R*O_R; ind += blockDim.x*blockDim.y){
//           s_di_du[ind] = (s_dc_du_plus[ind] - s_dc_du_minus[ind])/ (2* eps);
//         }

//            __syncthreads();
//         // solves for di_dqd
//         for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < I_N; ind += blockDim.x*blockDim.y){
//         qqd_plus[7 + ind] += eps; 

//         grid::load_update_XImats_helpers<T>(s_XImats, &qqd_plus[0], d_robotModel, s_temp);
//         __syncthreads();

//         inverse_dynamics_inner<T>(s_temp, s_vaf_plus, &qqd_plus[0], &qqd_plus[7], s_XImats, &s_temp[7], gravity);
//         // direct_minv_inner<T>(s_Minv, &qqd_plus[0], s_XImats, s_temp);
//         forward_dynamics_finish<T>(s_qdd1_plus, s_tau, s_temp, s_Minv);
//         inverse_dynamics_inner_vaf<T>(s_vaf_plus, &qqd_plus[0], &qqd_plus[7], s_qdd1_minus, s_XImats, s_temp, gravity);
//         inverse_dynamics_gradient_inner<T>(&s_dc_du_plus[49*ind], &qqd_plus[0], &qqd_plus[7], s_vaf_plus, s_XImats, s_temp, gravity);

//         qqd_minus[7 + ind] -= (2*eps);
//         grid::load_update_XImats_helpers<T>(s_XImats, &qqd_minus[0], d_robotModel, s_temp);
//         __syncthreads();

//         inverse_dynamics_inner<T>(s_temp, s_vaf_plus, &qqd_minus[0], &qqd_minus[7], s_XImats, &s_temp[7], gravity);
//         // direct_minv_inner<T>(s_Minv, &qqd_plus[0], s_XImats, s_temp);
//         forward_dynamics_finish<T>(s_qdd1_minus, s_tau, s_temp, s_Minv);
//         inverse_dynamics_inner_vaf<T>(s_vaf_minus, &qqd_minus[0], &qqd_minus[7], s_qdd1_minus, s_XImats, s_temp, gravity);
//         inverse_dynamics_gradient_inner<T>(&s_dc_du_minus[49*ind], &qqd_minus[0], &qqd_minus[7], s_vaf_minus, s_XImats, s_temp, gravity);


//         qqd_minus[7 + ind] += eps;

//        }
//        __syncthreads();
//                       __syncthreads();
//     //   if(threadIdx.x == 0 && blockIdx.x == 0){

//     //     for(int i=0; i<7; i++){
//     //         printf("s_dc_du_plus[%d] = \n", i*49);
//     //         printMat<T,7,7>(&s_dc_du_plus[i*49],7);
//     //         printf("s_dc_du_minus[%d] = \n", i*49);
//     //         printMat<T,7,7>(&s_dc_du_minus[i*49],7);
//     //     }
//     //   }

//         for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R*O_R; ind += blockDim.x*blockDim.y){
//           s_di_du[343 + ind] = (s_dc_du_plus[ind] - s_dc_du_minus[ind])/ (2* eps);
//         }
//        __syncthreads();

//         // solves for di_dq_dqd

//         for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < I_N; ind += blockDim.x*blockDim.y){
//         qqd_plus[ind] += eps; 
//         qqd_plus[7 + ind] += eps; 

//         grid::load_update_XImats_helpers<T>(s_XImats, &qqd_plus[0], d_robotModel, s_temp);
//         __syncthreads();

//         inverse_dynamics_inner<T>(s_temp, s_vaf_plus, &qqd_plus[0], &qqd_plus[7], s_XImats, &s_temp[7], gravity);
//         // direct_minv_inner<T>(s_Minv, &qqd_plus[0], s_XImats, s_temp);
//         forward_dynamics_finish<T>(s_qdd1_plus, s_tau, s_temp, s_Minv);
//         inverse_dynamics_inner_vaf<T>(s_vaf_plus, &qqd_plus[0], &qqd_plus[7], s_qdd1_minus, s_XImats, s_temp, gravity);
//         inverse_dynamics_gradient_inner<T>(&s_dc_du_plus[49*ind], &qqd_plus[0], &qqd_plus[7], s_vaf_plus, s_XImats, s_temp, gravity);

//         qqd_minus[ind] -= (2*eps);
//         qqd_minus[7 + ind] -= (2*eps);
//         grid::load_update_XImats_helpers<T>(s_XImats, &qqd_minus[0], d_robotModel, s_temp);
//         __syncthreads();

//         inverse_dynamics_inner<T>(s_temp, s_vaf_plus, &qqd_minus[0], &qqd_minus[7], s_XImats, &s_temp[7], gravity);
//         // direct_minv_inner<T>(s_Minv, &qqd_plus[0], s_XImats, s_temp);
//         forward_dynamics_finish<T>(s_qdd1_minus, s_tau, s_temp, s_Minv);
//         inverse_dynamics_inner_vaf<T>(s_vaf_minus, &qqd_minus[0], &qqd_minus[7], s_qdd1_minus, s_XImats, s_temp, gravity);
//         inverse_dynamics_gradient_inner<T>(&s_dc_du_minus[49*ind], &qqd_minus[0], &qqd_minus[7], s_vaf_minus, s_XImats, s_temp, gravity);


//         qqd_minus[ind] += eps;
//         qqd_minus[7 + ind] += eps;

//        }
//        __syncthreads();
//     //                   __syncthreads();
//     //   if(threadIdx.x == 0 && blockIdx.x == 0){

//     //     for(int i=0; i<7; i++){
//     //          printf("s_dc_du_plus[%d] = \n", i*49);
//     //         printMat<T,7,7>(&s_dc_du_plus[i*49],7);
//     //         printf("s_dc_du_minus[%d] = \n", i*49);
//     //         printMat<T,7,7>(&s_dc_du_minus[i*49],7);
//     //     }
//     //   }
//         __syncthreads();

//         for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R*O_R; ind += blockDim.x*blockDim.y){
//           s_di_du[686 + ind] = (s_dc_du_plus[ind] - s_dc_du_minus[ind])/ (2* eps);
//         }
//        __syncthreads();
    
//         // solves for dM_dq
//         for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < I_N; ind += blockDim.x*blockDim.y){
//         qqd_plus[ind] += eps; 

//         grid::load_update_XImats_helpers<T>(s_XImats, &qqd_plus[0], d_robotModel, s_temp);
//         __syncthreads();

//         // grid::direct_minv_inner<T>(&s_Minv_plus[ind*49], &qqd_plus[0], s_XImats, s_temp);
//         crba_inner(&s_M_plus[ind*49], &qqd_plus[0], &qqd_plus[7], s_XImats, s_temp, gravity);
//         __syncthreads();


//         qqd_minus[ind] -= (2*eps);
//         grid::load_update_XImats_helpers<T>(s_XImats, &qqd_minus[0], d_robotModel, s_temp);
//         __syncthreads();

//         // grid::direct_minv_inner<T>(&s_Minv_minus[ind*49], &qqd_plus[0], s_XImats, s_temp);
//         crba_inner(&s_M_minus[ind*49], &qqd_minus[0], &qqd_minus[7], s_XImats, s_temp, gravity);
//         __syncthreads();


//         qqd_minus[ind] += eps;

//        }
//        __syncthreads();
//           if(threadIdx.x == 0 && blockIdx.x == 0){

//         for(int i=0; i<7; i++){
//              printf("s_M_plus[%d] = \n", i*49);
//             printMat<T,7,7>(&s_M_plus[i*49],7);
//             printf("s_M_minus[%d] = \n", i*49);
//             printMat<T,7,7>(&s_M_minus[i*49],7);
//         }
//       }
//                       __syncthreads();


//         for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < O_R*O_R*O_R; ind += blockDim.x*blockDim.y){
//           s_di_du[1029 + ind] = (s_M_plus[ind] - s_M_minus[ind])/ (2* eps);
//             if(threadIdx.x == 0 && blockIdx.x == 0){
//                 printf("s_di_du[1029 + ind]%d  = %f bc s_M_plus[ind] = %f and s_M_minus[ind] = %f\n", ind,s_di_du[1029 +ind], s_M_plus[ind], s_M_minus[ind]);

//             }

//         }

//         if(threadIdx.x == 0 && blockIdx.x == 0){
//             for(int i=0; i<28; i++){
//                 printf("s_di_du %d\n", i);
//                 printMat<T,O_R,7>(&s_di_du[49*i],7);
//             }
//          }
//         __syncthreads();

//         load_update_XImats_helpers<T>(s_XImats, &s_q_qd_tau[0], d_robotModel, s_temp);
//         T *s_df_dq = s_df_du; T *s_df_dqd = &s_df_du[49]; T *s_df_tau = &s_df_du[98];
//         direct_minv_inner<T>(s_Minv, &s_q_qd_tau[0], s_XImats, s_temp);
//         forward_dynamics_finish<T>(s_qdd, &s_q_qd_tau[14], s_temp, s_Minv);
//         fdsva_inner<T>(s_df_du, &s_q_qd_tau[0], &s_q_qd_tau[7], s_qdd, &s_q_qd_tau[14], s_XImats, s_temp, gravity);

//          __syncthreads();
//         T *s_dm_dq = &s_di_du[1029];


//         T *dM_dqxfd_dq = s_temp;
//         for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 343; ind += blockDim.x*blockDim.y){
//             int page = ind / 49;
//             int row = ind % 7;
//             int col = ind % 49 / 7;
//             dM_dqxfd_dq[ind] = dot_prod<T,7,7,1>(&s_dm_dq[49*page + row], &s_df_dq[7*col]);
//         }
//         __syncthreads();
//         T *rot_dM_dqxfd_dqd = &s_temp[0];
//         for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
//             int page = 0;
//             int row = ind % 7;
//             int col = ind / 7;
//             rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
//         }
//         __syncthreads();
//         for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
//             int page = 1;
//             int row = ind % 7;
//             int col = ind / 7;
//             rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
//         }
//         __syncthreads();
//         for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
//             int page = 2;
//             int row = ind % 7;
//             int col = ind / 7;
//             rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
//         }
//         __syncthreads();
//         for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
//             int page = 3;
//             int row = ind % 7;
//             int col = ind / 7;
//             rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
//         }
//         __syncthreads();
//         for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
//             int page = 4;
//             int row = ind % 7;
//             int col = ind / 7;
//             rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
//         }
//         __syncthreads();
//         for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
//             int page = 5;
//             int row = ind % 7;
//             int col = ind / 7;
//             rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
//         }
//         __syncthreads();
//         for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 49; ind += blockDim.x*blockDim.y){
//             int page = 6;
//             int row = ind % 7;
//             int col = ind / 7;
//             rot_dM_dqxfd_dqd[49*col + row + 7*page] = dM_dqxfd_dq[49*page + ind];
//         }
//         __syncthreads();
//         for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 343; ind += blockDim.x*blockDim.y){
//             s_df2[ind] = s_di_du[ind] + dM_dqxfd_dq[ind] + rot_dM_dqxfd_dqd[ind];
//         }
//         __syncthreads();
//         T *dM_dqxfd_dqd = s_temp;
//         for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 343; ind += blockDim.x*blockDim.y){
//             int page = ind / 49;
//             int row = ind % 7;
//             int col = ind % 49 / 7;
//             dM_dqxfd_dqd[ind] = dot_prod<T,7,7,1>(&s_dm_dq[49*page + row], &s_df_dqd[7*col]);
//         }
//         __syncthreads();
//         for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 343; ind += blockDim.x*blockDim.y){
//             s_df2[343+ ind] = s_di_du[343 + ind] + dM_dqxfd_dqd[ind];
//             s_df2[686+ ind] = s_di_du[686 + ind];
//         }
//         __syncthreads();
//         s_Minv[1] = s_Minv[7];
//         s_Minv[2] = s_Minv[14];
//         s_Minv[9] = s_Minv[15];
//         s_Minv[3] = s_Minv[21];
//         s_Minv[10] = s_Minv[22];
//         s_Minv[17] = s_Minv[23];
//         s_Minv[4] = s_Minv[28];
//         s_Minv[11] = s_Minv[29];
//         s_Minv[18] = s_Minv[30];
//         s_Minv[25] = s_Minv[31];
//         s_Minv[5] = s_Minv[35];
//         s_Minv[12] = s_Minv[36];
//         s_Minv[19] = s_Minv[37];
//         s_Minv[26] = s_Minv[38];
//         s_Minv[33] = s_Minv[39];
//         s_Minv[6] = s_Minv[42];
//         s_Minv[13] = s_Minv[43];
//         s_Minv[20] = s_Minv[44];
//         s_Minv[27] = s_Minv[45];
//         s_Minv[34] = s_Minv[46];
//         s_Minv[41] = s_Minv[47];
//         for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 343; ind += blockDim.x*blockDim.y){
//             int page = ind / 49;
//             int row = ind % 7;
//             int col = ind % 49 / 7;
//             s_df2[1029+ ind] = dot_prod<T,7,7,1>(&s_dm_dq[49*page + row], &s_Minv[7*col]);
//         }
//         __syncthreads();
//         for(int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < 1372; ind += blockDim.x*blockDim.y){
//             int page = ind / 49;
//             int row = ind % 7;
//             int col = ind % 49 / 7;
//             s_df2[ind] = dot_prod<T,7,7,1>(&s_Minv[row], &s_df2[49*page + 7*col]);
//             s_df2[ind] *= (-1);
//         }
//         __syncthreads();

  
//             //  __syncthreads();
//     //   if(threadIdx.x == 0 && blockIdx.x == 0){

//     //     for(int i=0; i<28; i++){
//     //         printf("s_df2[%d] = \n", i*49);
//     //         printMat<T,7,7>(&s_df2[i*49],7);
//     //     }
//     //   }

// }
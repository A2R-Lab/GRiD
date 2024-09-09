// This file tests the direct_minv function in the generated grid.cuh file
// It outputs the inverse of mass matrix computed in grid.cuh
// Change the q, qd, qdd arrays, FLOATING_BASE and the NUM_DOF variable based on the robot
// grid.cuh comes from generategrid.py

#include <iostream>
#include "grid.cuh"
#include <random>

int main() {
  using T = float;
  const int NUM_TIMESTEPS = 1;
  const int NUM_DOF = 12;
  const bool FLOATING_BASE = false;

  grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
  cudaStream_t *streams = grid::init_grid<T>();
  grid::gridData<T> *hd_data = grid::init_gridData<T,NUM_TIMESTEPS>(); 
 
  // Init q, qd, qdd for Minv
T q[NUM_DOF+FLOATING_BASE] = {-0.336899, 1.29662, -0.677475, -1.42182, -0.706676, -0.134981, -1.14953, -0.296646, 2.13845, 2.00956, 1.55163, 2.2893};
T qd[NUM_DOF] = {0.43302, -0.421561, -0.645439, -1.86055, -0.0130938, -0.458284, 0.741174, 1.76642, 0.898011, -1.85675, 1.62223, 0.709379};
T qdd[NUM_DOF] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  for (int i = 0; i < NUM_DOF+FLOATING_BASE; i++) {
    hd_data->h_q[i] = q[i];    // Initialize joint positions
    hd_data->h_q_qd[i] = q[i];               // Initialize joint positions
    hd_data->h_q_qd_u[i] = q[i];                     // Initialize joint positions

  }

  for (int i = 0; i < NUM_DOF; i++) { 
    hd_data->h_q_qd[NUM_DOF + FLOATING_BASE + i] = qd[i]; // Initialize joint velocities 

    hd_data->h_q_qd_u[NUM_DOF + FLOATING_BASE + i] = qd[i]; // Initialize joint velocities 
    hd_data->h_q_qd_u[2 * NUM_DOF + FLOATING_BASE + i] = qdd[i]; // Initialize joint accelerations
  } 

  dim3 block_dimms = dim3(256, 1, 1);
  dim3 thread_dimms = dim3(32,1,1);
  
  grid::direct_minv(hd_data, d_robotModel, NUM_TIMESTEPS, block_dimms, thread_dimms, streams);

  for (int i = 0; i < NUM_DOF; i++) {
    for (int j = 0; j < NUM_DOF; j++) {
      printf("Minv[%i]: %f  ", j * NUM_DOF + i, hd_data->h_Minv[j * NUM_DOF + i]);
    }
    printf("\n");
  }

  
  
  grid::close_grid<T>(streams, d_robotModel, hd_data);
    
}
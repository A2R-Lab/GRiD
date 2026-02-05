# PyGRiD: Python Bindings for GRiD

This package provides Python bindings for the CUDA-based GRiD (**G**PU-Accelerated **Ri**gid Body **D**ynamics) library using pybind11.

GRiD is a library of functions that computes physics quantities needed for trajectory planning algorithms by generating CUDA kernels for your specific robot. Read more about this in the repository's main [README](../README.md).

**Example GRiD Usage: Trajectory Planning**
1. A target pose is defined using joint angles or end-effector positions and converted into a series of waypoints from the current pose.
2. A motor control algorithm is used to calculate the torques needed to move the robot to the next waypoint.
3. GRiD is invoked and `forward_dynamics` is called to convert the calculated torques into accelerations
4. An integrator is used to convert the accelerations into updated velocities and positions.
5. The robot moves to the next waypoint, and steps 2-4 is repeated until the final target position is reached.

An example of this use case can be found in the implementation of KKT systems solvers in [MPCGPU](https://github.com/A2R-Lab/MPCGPU/tree/0efde8c63c38465bba630ba569c4f8a30c1b009c).

## Installation
**Option 1: Install using pip**

    pip3 install .

For editable install, add the `-e` flag to the command above.

**Option 2: Manual build and install (for developers)**

    cmake -B build
    cmake --build build -j
    pip3 install -e .


## Quick Start
To use PyGRiD, you will first need a grid.cuh file generated from your URDF file. This is a one-time generation for your robot configuration. See the [GRiD README](../README.md) for more information about what this does and how this works.

    generateGRiD.py PATH_TO_URDF

The grid.cuh file contains optimized CUDA C++ code for parallel RBD calculation computations. The Python bindings invoke the CUDA kernels defined in grid.cuh to maintain the efficiency of the computations.

Example Python API usage:
```python
import numpy as np
import gridCuda

# generate random joint positions, velocities, and controls as input
np.random.seed(0)
q = np.random.normal(0, 1, gridCuda.NUM_JOINTS).astype(np.float32) # positions
qd = np.random.normal(0, 1, gridCuda.NUM_JOINTS).astype(np.float32) # velocities
u = np.random.normal(0, 1, gridCuda.NUM_JOINTS).astype(np.float32) # torques

# Create a GRiD instance with default gravity (9.81)
grid = gridCuda.GRiDDataFloat(q, qd, u) # or GRidDataDouble for double precision

# Calculate inverse dynamics
c = grid.inverse_dynamics()
print(c)
```
See [test_grid_cuda.py](tests/test_grid_cuda.py) for a more detailed example.

## API Reference
[Full API Reference and Documentation](#api-reference)

### Classes
- `GRiDDataFloat(q, qd, u)`: Single-precision (float) implementation of RBD functions
- Note that the double implementation caused errors

### Functions
- `load_joint_info(q, qd, u)`: Update the input parameters for RBD calculations
- `get_end_effector_positions()`: Calculates end-effector poses
- `get_end_effector_position_gradients()`: Calculates end-effector pose gradients
- `inverse_dynamics()`: Calculates the RNEA torque vector
- `minv()`: Calculates the inverse of the mass matrix
- `forward_dynamics()`: Calculates joint accelerations
- `inverse_dynamics_gradient()`: Calculates Jacobian of inverse dynamics w.r.t *q* and *qd*
- `forward_dynamics_gradient()`: Calculates Jacobian of forward dynamics w.r.t *q* and *qd*

### Variables
- `NUM_JOINTS`: Number of joints defined in the URDF
- `NUM_EES`: Number of end-effectors based on the URDF specification


## Requirements

    C++11 compatible compiler
    CUDA Toolkit >= 11.1 (compatible with compute capability 8.6)
    CMake >= 3.10
    Python >= 3.6
    pybind11

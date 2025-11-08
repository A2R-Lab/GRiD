PyGrid: Python Bindings for CUDA Grid Dynamics

This package provides Python bindings for the CUDA-based Grid Dynamics library using pybind11.
Requirements

    C++11 compatible compiler
    CUDA Toolkit (compatible with compute capability 86)
    CMake >= 3.10
    Python >= 3.6
    pybind11

Installation
Option 1: Install using pip

bash

pip install .

Option 2: Manual build

python3 setup.py build


Usage (python)n

import numpy as np
import gridCuda

# Create a grid instance with default gravity (9.81)
grid = gridCuda.GRidDataFloat()  # or GRidDataDouble for double precision

# Set joint positions, velocities, and control inputs
q = np.random.normal(0, 1, gridCuda.NUM_JOINTS).astype(np.float32)
qd = np.random.normal(0, 1, gridCuda.NUM_JOINTS).astype(np.float32)
u = np.random.normal(0, 1, gridCuda.NUM_JOINTS).astype(np.float32)

grid.load_joint_info(q, qd, u)

# Calculate inverse dynamics
c = grid.inverse_dynamics()


See example_pygrid.py for a more detailed example.
API Reference
Classes

    GridDataFloat: Single-precision (float) implementation.
    Note that the double implementation caused errors. 





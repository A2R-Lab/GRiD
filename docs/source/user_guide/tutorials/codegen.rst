Generate CUDA for a robot
==============================

GRiD's code generator (``grid_codegen/`` in this repository) produces a
robot-specific ``grid.cuh`` header; generating the header does not compile or
run CUDA. The ``grid-generate`` CLI ships in the ``grid-rbd`` wheel, so
``pip install grid-rbd`` (:doc:`installation <../getting_started/installation>`)
is enough; only a source checkout additionally needs its Git submodules
populated.

Generate a header
-----------------

From the repository root (the sample URDFs under ``grid_assets/robots/`` are
in the checkout, not the wheel — substitute your own URDF path):

.. code-block:: shell

   grid-generate grid_assets/robots/iiwa14.urdf --algorithm-list dynamics -o /tmp/iiwa_grid.cuh

``--algorithm-list`` restricts emission to named algorithms or profiles.
Omit it to use the full profile. Add ``-f`` for a floating base; use
``--no-mujoco-kernels`` to omit the additional MuJoCo-convention kernels.
For collision geometry, see :doc:`collisions`.

Generate from Python
--------------------

.. code-block:: python

   from URDFParser import URDFParser
   from grid_codegen import GRiDCodeGenerator

   robot = URDFParser().parse("grid_assets/robots/iiwa14.urdf")
   generator = GRiDCodeGenerator(robot)
   generator.gen_all_code(
       algorithm_list=["dynamics"],
       output_path="/tmp/iiwa_grid.cuh",
   )

``GRiDCodeGenerator`` is the Python class name, not a separate repository
or installable module. GRiD's code generator vendors GLASS device-side
linear algebra into the header by default. Set ``vendor_glass=False`` when
using an external ``glass.cuh`` on your C++ include path.

Use the generated code
----------------------

* For Python applications, :doc:`python_wrappers` handles generation,
  compilation and caching through ``grid_rbd.load_robot(...)``.
* For CUDA applications, include the header in your own translation unit
  (next section). See :doc:`../../api_reference/gridcodegen` for the
  entry-point layers and :doc:`../concepts/library_safe_initialization` for
  allocation and cleanup.
* Use :doc:`../getting_started/fast_robot_setup` to reduce large-robot build
  time and memory, and :doc:`cuda_validation` to check generated results.

Include it in your kernel
-------------------------

This is the power-user entry point: the header is self-contained (GLASS
device-side linear algebra is vendored in by default), so a kernel that
``#include``\ s it needs only ``nvcc``. Every algorithm has a ``_device``
routine that runs inside your block on shared-memory operands, which is how
you fuse GRiD dynamics into an MPC, RL or controls kernel:

.. code-block:: cuda

   #include "grid.cuh"   // generated: grid-generate robot.urdf -o grid.cuh

   template <typename T>
   __global__ void my_kernel(T *d_tau, const T *d_q, const T *d_qd, const T *d_qdd,
                             const grid::robotModel<T> *model, T gravity) {
       __shared__ T s_q[grid::NUM_JOINTS], s_qd[grid::NUM_JOINTS],
                    s_qdd[grid::NUM_JOINTS], s_tau[grid::NUM_JOINTS];
       for (int i = threadIdx.x; i < grid::NUM_JOINTS; i += blockDim.x) {
           s_q[i] = d_q[i]; s_qd[i] = d_qd[i]; s_qdd[i] = d_qdd[i];
       }
       __syncthreads();
       grid::inverse_dynamics_device<T>(s_tau, s_q, s_qd, s_qdd, model,
                                        /*d_f_ext=*/nullptr, gravity);
       __syncthreads();   // fuse your own math on s_tau here
       for (int i = threadIdx.x; i < grid::NUM_JOINTS; i += blockDim.x) d_tau[i] = s_tau[i];
   }
   // host: grid::init_grid<float>(); auto *model = grid::init_robotModel<float>();
   // my_kernel<float><<<1, 128, grid::INVERSE_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES<float>()>>>(...);

.. code-block:: shell

   nvcc -arch=sm_90 -std=c++17 my_kernel.cu -o my_kernel   # pick your GPU's arch

``examples/cuda/`` holds the full version of this kernel (the easy ``_device``
call and the full-control ``_inner`` call with explicit workspace), plus
``build_and_validate.sh``, which generates the header, builds, runs and checks
the output against the reference implementation; the notebook
``examples/notebooks/07_inline_cuda.ipynb`` is the same walk-through run
inline. Each ``_device`` routine's dynamic shared-memory requirement is
exposed as ``grid::<ALGO>_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>()``; the
:doc:`resource tiers <../concepts/resource_tier_system>` page covers fitting a
routine into a tight outer kernel.

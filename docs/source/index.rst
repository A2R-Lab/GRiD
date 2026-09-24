GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients
===================================================================

`GRiD <https://github.com/A2R-Lab/GRiD>`_ turns a URDF into optimized,
per-robot CUDA C++ for rigid-body dynamics, kinematics, their **analytical
first- and second-order derivatives**, and a trajectory-optimization plant
layer — then hands you that code three ways: a numpy handle, a
``jax.jit``-able FFI surface, or ``torch.autograd``-aware ops, all backed by
one content-addressed ``.so`` cache. It is the dynamics layer underneath
`GATO <http://a2r-lab.org/GATO/>`_, `MPCGPU <https://a2r-lab.org/publication/mpcgpu/>`_,
`HJCD-IK <https://a2r-lab.org/publication/hjcdik/>`_ and other A2R Lab GPU
solvers, built on `GLASS <http://a2r-lab.org/GLASS/>`_ (block-local linear
algebra), `RBDReference <https://github.com/A2R-Lab/RBDReference>`_ (the
Pinocchio-validated numpy oracle every kernel is tested against) and
`URDFParser <https://github.com/A2R-Lab/URDFParser>`_.

**One block per problem, batched.** Every algorithm runs as a single CUDA
block per sample with in-block parallelism, so a batch of 16 or 4096 states
is the same kernel design on a Jetson or an RTX 5090 — no multi-block
reductions, no cooperative groups, bit-deterministic and thread-count
invariant. Artifacts are built per target architecture (``sm_XX``); the
model, the API and the generated code carry over, the binary does not.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: numpy
      :link: user_guide/tutorials/python_wrappers
      :link-type: doc

      .. code-block:: python

         import grid_rbd
         r = grid_rbd.load_robot("iiwa14.urdf")
         qdd = r.forward_dynamics(q, qd, u)
         dqdd = r.forward_dynamics_gradient(q, qd, u)

      20+ batched methods; inputs and outputs are host arrays.

   .. grid-item-card:: JAX
      :link: jax-ffi-quickstart
      :link-type: ref

      .. code-block:: python

         r = grid_rbd.load_robot("go2.urdf", backend="jax",
                                 floating_base=True)
         loss = lambda q: r.forward_dynamics(q, qd, u).sum()
         g = jax.jit(jax.grad(loss))(q)   # analytic VJP, on device

      Device-in / device-out FFI targets that compose under ``jit``,
      ``vmap`` and ``scan``; ``jax.grad`` runs the analytical gradient
      kernels, never finite differences.

   .. grid-item-card:: torch
      :link: user_guide/tutorials/python_wrappers
      :link-type: doc

      .. code-block:: python

         r = grid_rbd.load_robot("iiwa14.urdf", backend="torch")
         qdd = r.forward_dynamics(q, qd, u)   # CUDA tensors in/out
         qdd.sum().backward()                 # analytic backward
         step = r.capture("forward_dynamics", q, qd, u)  # CUDA graph

      Autograd-aware ops on the current CUDA stream, CUDA-graph capture
      for fixed-batch replay.

What you get per robot
----------------------

* **Dynamics**: inverse dynamics (RNEA), forward dynamics (Minv-based and
  ABA), the joint-space inertia matrix (CRBA) and its inverse, gravity,
  non-linear effects, the Coriolis matrix, external body wrenches and tool
  loads.
* **Derivatives**: analytical gradients of inverse and forward dynamics
  (IDSVA / FDSVA), their **second-order** tensors, inverse-dynamics and
  forward-dynamics gradients with respect to the inertial parameters, and
  end-effector pose Jacobians and Hessians.
* **Kinematics and centroidal quantities**: end-effector poses (baked or
  runtime targets), frame Jacobians and their time derivatives, centroidal
  momentum matrix, its time variation and the operational-space inertia.
* **Plant layer**: integrators with gradients, quadratic and barrier costs
  with gradients and Hessians, ready for a trajectory optimizer.
* **Conventions**: Pinocchio by default; MuJoCo/mjx-convention twins of
  values and derivatives on floating-base robots (``handle.mujoco.<op>``).

Every algorithm exists on two surfaces that are tested for numerical
agreement — the numpy oracle (validated against Pinocchio) and the generated
CUDA — and the GPU outcomes are captured in a signed receipt that CPU-only CI
verifies against the committed test fingerprints (the receipt-verify job goes
red when fingerprinted tests change without a refreshed receipt; a release
requires one fresh full receipt at the release tip under the release policy).

Measured
--------

GPU-resident (the state already on the device, the design point for MPC
rollouts and RL sampling), GRiD wins every comparable cell against the five
GPU and CPU baselines in the competitive sweep; the largest second-order
tensor (a 29-DoF humanoid at batch 256) stays on the GPU for the next
pipeline stage at wall-time parity with a 24-thread CPU codegen. The
:doc:`benchmarks page <user_guide/tutorials/benchmarks>` has the dated,
per-cell numbers, the with-memory caveats and the harness to reproduce them
on your GPU.

Portable by construction
------------------------

The generated code adapts to the device it runs on rather than to the one it
was tuned on: a per-algorithm **resource tier** chooses how much of the
working set lives in shared memory (``SHARED`` for peak performance,
``LITE`` and ``MINIMAL`` rungs that spill scratch to global memory so the
largest humanoids still fit a 48 KB budget), launch configurations are
autotuned per robot and architecture, and a **runtime context** carries the device arena, streams
and model tables — several isolated pipelines can share one GPU, one handle
can swap inertial parameters at run time, and autograd refuses to
differentiate a model that changed under it. GRiD targets a single GPU, from
embedded Jetson-class devices to desktop cards, one artifact per
architecture; the tested deployment platform of this release is Linux x86_64
(see :doc:`compatibility and known limitations <user_guide/getting_started/compatibility>`).

30-second quickstart
--------------------

.. code-block:: shell

   git clone --recursive https://github.com/A2R-Lab/GRiD && cd GRiD
   bash install/base_install.sh && source .venv/bin/activate   # + pip install -e ".[jax]" / ".[torch]"

.. code-block:: python

   import numpy as np, grid_rbd
   r = grid_rbd.load_robot("config/robot_assets/iiwa14.urdf")   # compiles once, cached forever
   q, qd, u = (np.zeros((8, r.nq), np.float32) for _ in range(3))
   print(r.forward_dynamics(q, qd, u).shape)                    # (8, 7)

The first ``load_robot`` of a robot generates and compiles its ``.so``
(minutes; see :doc:`fast robot setup <user_guide/getting_started/fast_robot_setup>`
for the cold/warm numbers and how to keep a humanoid build in RAM); every
later load is seconds.

Go deeper
---------

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: How do I…?
      :link: how_do_i
      :link-type: doc

      The task router: calling, generating, testing, benchmarking,
      debugging — one table.

   .. grid-item-card:: Concepts
      :link: user_guide/concepts/index
      :link-type: doc

      Design principles, the codegen architecture, resource tiers, runtime
      contexts, operand validation, the I/O ABI, mjx conventions.

   .. grid-item-card:: From raw CUDA
      :link: user_guide/tutorials/codegen
      :link-type: doc

      ``grid-generate robot.urdf`` emits a self-contained ``grid.cuh`` to
      ``#include`` in your own kernels.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: API Reference
      :link: api_reference/index
      :link-type: doc

      The Python bindings, the URDF parser, the reference algorithms and
      the code generator.

   .. grid-item-card:: Benchmarks
      :link: user_guide/tutorials/benchmarks
      :link-type: doc

      Competitive and historical sweeps, the with-memory story, and how to
      re-measure on your hardware.

   .. grid-item-card:: Validation and CI
      :link: user_guide/tutorials/cuda_validation
      :link-type: doc

      The CUDA-vs-numpy equivalence suite, the signed GPU receipt and the
      split test driver.

Citation
--------

If you use GRiD in your research, please cite:

.. code-block:: text

   @inproceedings{plancher2022grid,
     title={GRiD: GPU-Accelerated Rigid Body Dynamics with Analytical Gradients},
     author={Brian Plancher and Sabrina M. Neuman and Radhika Ghosal and Scott Kuindersma and Vijay Janapa Reddi},
     booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
     year={2022},
     month={May}
   }

.. toctree::
   :hidden:

   how_do_i

.. toctree::
   :hidden:
   :caption: User Guide

   user_guide/getting_started/installation
   user_guide/getting_started/fast_robot_setup
   user_guide/getting_started/library_overview
   user_guide/getting_started/docker_setup
   user_guide/getting_started/compatibility
   user_guide/glossary

.. toctree::
   :hidden:
   :caption: Tutorials

   user_guide/tutorials/index

.. toctree::
   :hidden:
   :caption: Concepts

   user_guide/concepts/index

.. toctree::
   :hidden:
   :caption: API Reference

   api_reference/index

.. toctree::
   :hidden:
   :caption: Project info

   contribution_guidelines
   sphinx_edit_guide

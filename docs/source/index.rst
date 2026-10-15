GRiD documentation
==================

`GRiD <https://github.com/A2R-Lab/GRiD>`_ generates optimized, robot-specific
CUDA C++ from a URDF: dynamics, kinematics and collisions, their analytical
derivatives and Hessians, and a plant layer for trajectory optimization. The
Python interfaces are a numpy handle, a ``jax.jit``-able FFI surface and
``torch.autograd``-aware ops, all backed by one content-addressed ``.so``
cache. GRiD is the dynamics layer underneath
`GATO <http://a2r-lab.org/GATO/>`_, `MPCGPU <https://a2r-lab.org/publication/mpcgpu/>`_,
`HJCD-IK <https://a2r-lab.org/publication/hjcdik/>`_ and other A2R Lab GPU
solvers, built on `GLASS <http://a2r-lab.org/GLASS/>`_ (block-local linear
algebra), `RBDReference <https://github.com/A2R-Lab/RBDReference>`_ (the
Pinocchio-validated numpy oracle every kernel is tested against) and
`URDFParser <https://github.com/A2R-Lab/URDFParser>`_.

Collision routines use the generated CUDA interface; see
:doc:`the collision workflow <user_guide/tutorials/collisions>` for geometry
coverage and integration. Start with the installation guide and the examples
below, or visit the `project homepage <../>`_ for an overview.

Every algorithm runs as a single CUDA block per sample with in-block
parallelism, so a batch of 16 or of 4096 states uses the same kernel design on
a Jetson or on an RTX 5090. There are no multi-block reductions and no
cooperative groups; results are bit-deterministic and independent of the
thread count. Artifacts are built per target architecture (``sm_XX``): the
model, the API and the generated code carry over, the binary does not.

.. grid:: 1 2 4 4
   :gutter: 3

   .. grid-item-card:: numpy
      :link: user_guide/tutorials/python_wrappers
      :link-type: doc

      ``load_robot(urdf)`` returns a handle with more than 20 batched
      methods; inputs and outputs are host arrays batched on axis 0.

   .. grid-item-card:: JAX
      :link: jax-ffi-quickstart
      :link-type: ref

      ``backend="jax"`` gives device-in, device-out FFI targets that compose
      under ``jit``, ``vmap`` and ``scan``; ``jax.grad`` runs the analytical
      gradient kernels rather than finite differences.

   .. grid-item-card:: torch
      :link: user_guide/tutorials/python_wrappers
      :link-type: doc

      ``backend="torch"`` gives autograd-aware ops on the current CUDA
      stream, plus CUDA-graph capture for fixed-batch replay.

   .. grid-item-card:: header-only CUDA
      :link: user_guide/tutorials/codegen
      :link-type: doc

      ``grid-generate robot.urdf`` writes one self-contained ``grid.cuh``
      to ``#include`` in your own kernels. Only ``nvcc`` is needed and no
      Python runs at runtime.

Complete, copy-paste examples for the Python surfaces are in the
:ref:`quickstart <landing-quickstart>` below; the header-only path is on
:doc:`user_guide/tutorials/codegen`.

What you get per robot
----------------------

* Dynamics: inverse dynamics (RNEA), forward dynamics (Minv-based and
  ABA), the joint-space inertia matrix (CRBA) and its inverse, gravity,
  non-linear effects, the Coriolis matrix, external body wrenches and tool
  loads.
* Derivatives: analytical gradients of inverse and forward dynamics
  (IDSVA / FDSVA), their second-order tensors, inverse-dynamics and
  forward-dynamics gradients with respect to the inertial parameters, and
  end-effector pose Jacobians and Hessians.
* Kinematics and centroidal quantities: end-effector poses (baked or
  runtime targets), frame Jacobians and their time derivatives, centroidal
  momentum matrix, its time variation and the operational-space inertia.
* Plant layer: integrators with gradients, quadratic and barrier costs
  with gradients and Hessians, ready for a trajectory optimizer.
* Collisions: self- and environment-collision checks with covering-sphere
  or native-primitive representations and coarse-to-fine evaluation in CUDA.
* Conventions: Pinocchio by default; MuJoCo/mjx-convention twins of
  values and derivatives on floating-base robots (``handle.mujoco.<op>``).

The test suites compare generated CUDA with CPU references and record GPU
results in signed receipts. See :doc:`validation <user_guide/tutorials/cuda_validation>`
for the test workflow and :doc:`backend coverage <user_guide/tutorials/backend_coverage>`
for operation-specific support.

Performance and release measurements
------------------------------------------------------------

The :doc:`release measurements <release_measurements>` page holds one audited
collection: three robots, fifteen operations and batch sizes 16 to 1024, with
GRiD through every surface beside Pinocchio, MJX, MuJoCo Warp, MuJoCo CPU, BARD
and Frax. It states the protocol, every cell's validation status and the cases
GRiD loses.
The :doc:`benchmarks page <user_guide/tutorials/benchmarks>` retains dated
development experiments and harness guidance; those results are not release
evidence.

Portable by construction
------------------------

The generated code adapts to the device it runs on rather than to the one it
was tuned on. A per-algorithm resource tier chooses how much of the working set
lives in shared memory: ``SHARED`` for peak performance, and ``LITE`` and
``MINIMAL`` rungs that spill scratch to global memory so the largest humanoids
still fit a 48 KB budget. Launch configurations are autotuned per robot and
architecture. A runtime context carries the device arena, the streams and the
model tables, so several isolated pipelines can share one GPU, one handle can
swap inertial parameters at run time, and autograd refuses to differentiate a
model that changed under it. GRiD targets a single GPU, from embedded
Jetson-class devices to desktop cards, with one artifact per architecture. The
tested deployment platform of this release is Linux x86_64; see
:doc:`compatibility and known limitations <user_guide/getting_started/compatibility>`.

.. _landing-quickstart:

Quickstart
----------

Install from PyPI. The extras select the jax and torch packages but not their
CUDA builds, so add the CUDA wheels yourself. A recursive clone with an
editable install is the developer path; see
:doc:`installation <user_guide/getting_started/installation>`.

.. code-block:: shell

   pip install "grid-rbd[jax,torch]" "jax[cuda12]"   # extras optional: the JAX and torch surfaces

   # developer checkout instead:
   git clone --recursive https://github.com/A2R-Lab/GRiD && cd GRiD
   bash install/base_install.sh && source .venv/bin/activate
   pip install -e ".[jax,torch]" "jax[cuda12]"

Each example below runs as written from a repository checkout; the URDFs live
under ``grid_assets/robots/``. The first ``load_robot`` of a robot generates and
compiles its ``.so``: about 100 s for the 7-DoF iiwa14 on an RTX 5090 and about
6 min for the g1 humanoid built without the mjx kernel twins. The cold and warm
build table and the RAM-safe subset builds are on
:doc:`fast robot setup <user_guide/getting_started/fast_robot_setup>`. Every
later load takes seconds, because the artifact is cached by content key and
rebuilt only when the URDF, the options, the GRiD version or the toolchain
change.

.. tab-set::

   .. tab-item:: numpy

      .. code-block:: python

         import numpy as np
         import grid_rbd

         r = grid_rbd.load_robot("grid_assets/robots/iiwa14.urdf")
         q, qd, u = (np.zeros((8, r.nq), np.float32) for _ in range(3))
         qdd = r.forward_dynamics(q, qd, u)                 # (8, 7)
         dqdd = r.forward_dynamics_gradient(q, qd, u)       # (8, 7, 14) = [d/dq | d/dqd]
         print(qdd.shape, dqdd.shape)

   .. tab-item:: JAX

      .. code-block:: python

         import jax
         import jax.numpy as jnp
         import grid_rbd

         r = grid_rbd.load_robot("grid_assets/robots/go2.urdf", backend="jax",
                                 floating_base=True)
         q = jnp.zeros((8, r.nq), jnp.float32).at[:, 6].set(1.0)   # unit quaternion (x, y, z, w)
         qd = jnp.zeros((8, r.nv), jnp.float32)                    # nv-wide: the tangent width
         u = jnp.zeros((8, r.nv), jnp.float32)
         loss = lambda a: r.forward_dynamics(a, qd, u).sum()
         g = jax.jit(jax.grad(loss))(q)                           # analytic VJP, stays on the device
         print(g.shape)

   .. tab-item:: torch

      .. code-block:: python

         import torch
         import grid_rbd

         r = grid_rbd.load_robot("grid_assets/robots/iiwa14.urdf", backend="torch")
         q = torch.zeros(8, r.nq, device="cuda", requires_grad=True)
         qd = torch.zeros(8, r.nq, device="cuda")
         u = torch.zeros(8, r.nq, device="cuda")
         qdd = r.forward_dynamics(q, qd, u)                 # CUDA tensors in and out
         qdd.sum().backward()                               # analytic backward
         step = r.capture("forward_dynamics", q.detach(), qd, u)   # CUDA-graph replay
         print(q.grad.shape, step.replay().shape)

Go deeper
---------

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: How do I…?
      :link: how_do_i
      :link-type: doc

      One table that routes every task: calling, generating, testing,
      benchmarking and debugging.

   .. grid-item-card:: Concepts
      :link: user_guide/concepts/index
      :link-type: doc

      Design principles, the codegen architecture, resource tiers, runtime
      contexts, operand validation, the I/O ABI, mjx conventions.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: API Reference
      :link: api_reference/index
      :link-type: doc

      The Python bindings, the URDF parser, the reference algorithms and
      the code generator.

   .. grid-item-card:: Benchmarks
      :link: user_guide/tutorials/benchmarks
      :link-type: doc

      Competitive and historical sweeps, the with-memory comparison, and
      how to re-measure on your hardware.

   .. grid-item-card:: Validation and CI
      :link: user_guide/tutorials/cuda_validation
      :link-type: doc

      The CUDA-vs-numpy equivalence suite, the signed GPU receipt and the
      split test driver.

Citation
--------

The original ICRA 2022 paper describes the implementation preserved at
`robot-acceleration/GRiD <https://github.com/robot-acceleration/GRiD>`_.
Ongoing development lives at
`A2R-Lab/GRiD <https://github.com/A2R-Lab/GRiD>`_. The paper does not describe
all current features or establish their performance. If you use GRiD in your
research, cite the original paper and record the software commit or release:

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
   :caption: Get started

   user_guide/getting_started/installation
   user_guide/getting_started/fast_robot_setup
   user_guide/getting_started/library_overview
   user_guide/getting_started/compatibility
   user_guide/glossary

.. toctree::
   :hidden:
   :caption: Examples

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
   :caption: Performance and validation

   release_measurements

.. toctree::
   :hidden:
   :caption: Project info

   contribution_guidelines

User Guide Overview
===================

GRiD generates optimized CUDA C++ implementations of rigid-body dynamics,
kinematics, and analytical-gradient algorithms from URDF robot models. The
package combines three Python components:

* :doc:`URDFParser <tutorials/urdf_parser>` for reading robot models.
* :doc:`RBDReference <tutorials/python_algorithms>` for CPU reference
  algorithms.
* :doc:`GRiDCodeGenerator <tutorials/codegen>` for CUDA header generation.

.. figure:: imgs/GRiD.png
   :alt: GRiD library ecosystem
   :width: 80%
   :align: center

   A URDF model is parsed, checked against Python reference algorithms, and
   lowered into generated CUDA code that can be validated and benchmarked.

Where to start
--------------

* **Install + first CUDA header**:
  :doc:`getting_started/installation`.
* **Call GRiD from Python**:
  :doc:`tutorials/python_wrappers` covers the ``grid-rbd`` package and
  the JAX FFI bridge.
* **Call GRiD from raw CUDA**:
  :doc:`tutorials/codegen` walks through the ``grid-generate`` CLI and
  the generated header API.
* **Architecture deep-dive**:
  :doc:`concepts/codegen_architecture` documents the four emission
  layers (``_inner`` / ``_device`` / ``_kernel`` / host) and the
  thread-count constraints.
* **Validation + performance**:
  :doc:`tutorials/cuda_validation` and :doc:`tutorials/benchmarks`.

Current support
---------------

GRiD supports revolute, prismatic, and fixed-joint robot models without
closed kinematic loops.

Implemented CUDA algorithm families include:

* Inverse dynamics / RNEA.
* Direct inverse mass matrix.
* Forward dynamics via Minv + RNEA.
* ABA and CRBA.
* Inverse- and forward-dynamics gradients.
* End-effector pose, gradient, and Hessian.
* Second-order inverse dynamics (IDSVA-SO: body-frame for fixed-base,
  world-frame for floating-base, with a codegen-time dispatcher) and
  second-order forward dynamics (FDSVA-SO) on both fixed and floating
  bases.

See :doc:`tutorials/cuda_support_status` for the current fixed/floating
support matrix and known caveats.

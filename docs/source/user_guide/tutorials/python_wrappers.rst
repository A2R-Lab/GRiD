Python Wrappers (``grid-rbd``)
==============================

The ``grid-rbd`` package wraps GRiD's per-robot CUDA codegen behind a
two-tier Python API: a slow one-time ``register_robot()`` step that
generates and compiles a per-robot ``.so``, and fast subsequent
algorithm calls on the returned ``RobotHandle``.

Source: ``python/`` in the GRiD repo.

Install (editable, from a GRiD checkout)
----------------------------------------

.. code-block:: shell

   cd path/to/GRiD
   pip install -e python/

This builds a small pybind11 extension (``grid_rbd._core``) at install
time. ``nvcc`` is **not** required for the install — only for
:py:func:`grid_rbd.register_robot`, which compiles a per-robot
``.so`` on first call.

Register-then-run UX
--------------------

.. code-block:: python

   import grid_rbd

   # One-time per (robot, options, GRiD version, CUDA arch).
   # ~30-60 s for iiwa14; cached under ~/.cache/grid-rbd/.
   handle = grid_rbd.register_robot(
       name="iiwa14",
       urdf_path="path/to/iiwa.urdf",
       floating_base=False,
       max_batch_size=256,
   )

   # Fast inference. All methods are 2D-batched on axis 0.
   import numpy as np
   q  = np.random.randn(64, handle.num_joints).astype(np.float32)
   qd = np.random.randn(64, handle.num_joints).astype(np.float32)
   u  = np.random.randn(64, handle.num_joints).astype(np.float32)

   qdd = handle.forward_dynamics(q, qd, u)   # shape (64, NJ)
   M   = handle.crba(q)                      # shape (64, NJ, NJ)

A complete walkthrough exercising every bound method is at
``python/examples/quickstart_iiwa14.py``.

Method surface
--------------

All methods take and return 2D ``float32`` arrays where axis 0 is the
batch. Gravity is passed as a **positive magnitude** (default 9.81),
following GRiD's internal convention; this differs from
``RBDReference.rnea(..., GRAVITY=-9.81)`` so use ``gravity=9.81`` for
cross-validation.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Method
     - Returns
     - Notes
   * - ``rnea(q, qd, qdd=None)``
     - ``(B, NJ)``
     - Inverse dynamics bias.
   * - ``minv(q)``
     - ``(B, NJ, NJ)``
     - Direct mass-matrix inverse.
   * - ``forward_dynamics(q, qd, u)``
     - ``(B, NJ)``
     - ``M⁻¹·(τ − c)``.
   * - ``aba(q, qd, u)``
     - ``(B, NJ)``
     - Articulated body algorithm FD.
   * - ``crba(q)``
     - ``(B, NJ, NJ)``
     - Composite-rigid-body mass matrix.
   * - ``end_effector_pose(q)``
     - ``(B, 6*NUM_EES)``
     - ``[xyz, rpy]`` per EE.
   * - ``end_effector_pose_gradient(q)``
     - ``(B, 6*NUM_EES, NJ)``
     - EE pose Jacobian.
   * - ``end_effector_pose_hessian(q)``
     - ``(B, 6*NUM_EES, NJ, NJ)``
     - EE pose Hessian (∂²ee/∂q²).
   * - ``rnea_grad(q, qd, qdd=None)``
     - ``(B, NJ, 2*NJ)``
     - ``[dc_dq | dc_dqd]``.
   * - ``forward_dynamics_grad(q, qd, u)``
     - ``(B, NJ, 2*NJ)``
     - ``[df_dq | df_dqd]``.
   * - ``idsva_so(q, qd, qdd=None)``
     - tuple of 4 ``(B, NV, NV, NV)``
     - Second-order ID (dispatched).
   * - ``fdsva_so(q, qd, u)``
     - tuple of 4 ``(B, NV, NV, NV)``
     - Second-order FD.

Validation against ``RBDReference`` lives at
``test/python_wrappers/test_iiwa14_smoke.py`` (16 tests, all numerical
methods pass at float32 precision).

Cache layout
------------

.. code-block:: text

   ~/.cache/grid-rbd/
   ├── manifest.json              # name -> cache_key
   └── store/<cache_key>/
       ├── grid.cuh
       ├── wrapper.cu
       ├── robot.so
       ├── meta.json
       └── robot.build.log

Cache key = SHA-256 of ``urdf_bytes + canonical_json(options) +
grid_rbd_version + cuda_arch``. CUDA arch in the key means a roaming
home directory (e.g. NFS-mounted between a laptop and a desktop) safely
keeps separate ``.so`` files per machine.

Override the cache root with ``$GRID_RBD_CACHE_DIR`` or
``cache_dir=...`` on ``register_robot``.

End-effector target selection
-----------------------------

By default ``register_robot`` uses GRiD's default EE choice (all leaf
nodes). Pass ``ee_joint_names=["iiwa_joint_7"]`` to bake a specific
fixed-joint target into the codegen:

.. code-block:: python

   handle = grid_rbd.register_robot(
       name="iiwa14_wrist",
       urdf_path="iiwa.urdf",
       ee_joint_names=["iiwa_joint_7"],
   )

A different ``ee_joint_names`` value lands in a separate cache entry —
both targets can coexist in the cache.

.. _jax-ffi-quickstart:

JAX FFI (``grid_rbd[jax]``)
---------------------------

Install with ``pip install grid-rbd[jax]`` to get the JAX bridge.
The same per-robot ``.so`` is shared with the plain wrapper — no
recompile on first ``grid_rbd.jax.register_robot``:

.. code-block:: python

   import grid_rbd.jax as grid_jax
   import jax

   handle = grid_jax.register_robot(name="iiwa14", urdf_path="iiwa.urdf")

   @jax.jit
   def step(q, qd, u):
       return handle.forward_dynamics(q, qd, u)

Methods slot into the JAX FFI machinery as ``ffi_call`` targets
running on JAX-supplied CUDA streams. Inputs may be numpy or
``jax.Array`` — JAX moves data to device transparently before the
handler runs, and outputs stay device-resident.

v0.3 JAX surface (parity with the plain ``RobotHandle``): all 12
methods listed in the table above are bound via FFI and JIT-compatible.
The SO methods (``idsva_so``, ``fdsva_so``) follow the plain wrapper's
tuple-of-four convention.

Coming next
-----------

* Floating-base JAX FFI for ``idsva_so`` (currently routes to the
  body-frame kernel; world-frame fallback for floating-base needs the
  codegen to emit a preprocessor-visible dispatcher).
* Any-thread-count library functions for CUDA-inline users (see
  :doc:`../concepts/cublasdx_removal_design`).
* CLI shortcut: ``grid-rbd register iiwa.urdf --name iiwa14``.

See also
--------

* :doc:`benchmarks` — bench harness and what the methods cost.
* :doc:`../concepts/algorithms/index` — algorithm-level docs.
* ``docs/python_wrappers_plan.md`` (repo root) — design rationale.

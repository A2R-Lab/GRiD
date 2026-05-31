Python Wrappers (``grid-rbd``)
==============================

The ``grid-rbd`` package wraps GRiD's per-robot CUDA codegen behind a
two-tier Python API: a slow one-time ``register_robot()`` step that
generates and compiles a per-robot ``.so``, and fast subsequent
algorithm calls on the returned handle.

``register_robot`` accepts a ``backend=`` argument — ``"numpy"`` (the
default, returning a ``RobotHandle``), ``"jax"`` (a ``JaxRobotHandle``),
or ``"torch"`` (a ``TorchRobotHandle``) — and a ``urdf_string=`` argument
to register from inline URDF text instead of a file on disk. All three
backends share the same content-addressed ``.so`` cache.

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
     - ``(B, 6*NUM_EES, NV)``
     - EE pose Jacobian ``d(pose)/dv`` in **tangent space** (matches pinocchio).
       Fixed-base ``NV == NJ``; floating-base ``NV = 6 + n_joints`` (spatial
       twist, ``[omega; v]``) rather than the older quaternion-derivative columns.
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

PyTorch backend (``backend="torch"``)
-------------------------------------

``register_robot(..., backend="torch")`` returns a ``TorchRobotHandle``
whose methods return ``torch.Tensor``. The four differentiable
algorithms (``rnea`` / ``forward_dynamics`` / ``aba`` / ``integrator``)
are autograd-aware — their backward passes are analytic, reusing the
existing ``*_gradient`` kernels — while the remaining methods are
forward-only ops. The ``.so`` is shared with the numpy/JAX surfaces; the
torch op block is compiled in under ``-DGRID_RBD_WITH_TORCH`` when torch
is present at register time.

.. code-block:: python

   import grid_rbd, torch

   h = grid_rbd.register_robot("iiwa14", urdf_path="iiwa.urdf", backend="torch")

   q  = torch.randn(64, h.num_joints, device="cuda", requires_grad=True)
   qd = torch.randn(64, h.num_joints, device="cuda", requires_grad=True)
   u  = torch.randn(64, h.num_joints, device="cuda", requires_grad=True)

   qdd = h.forward_dynamics(q, qd, u)   # autograd-aware torch.Tensor
   qdd.sum().backward()                 # gradients flow to q, qd, u

For fixed-batch, low-launch-overhead replay (MPC / training),
``handle.capture(method, *example_inputs, **kwargs)`` returns a
``GraphCallable`` backed by a CUDA graph. A mandatory off-graph warmup
runs the one-time >48 KB dynamic-smem opt-in (illegal during capture)
before the graph is recorded:

.. code-block:: python

   g = h.capture("forward_dynamics", q, qd, u)   # warmup + capture
   qdd = g(q_new, qd_new, u_new)                 # copy_ + replay

.. note::

   The backward VJP contractions run torch's own CUDA kernels, so the
   installed torch build must support the GPU's compute capability. On an
   RTX 5090 (sm_120) you need a torch **cu128** (or newer) build — a
   cu124 wheel (max sm_90) cannot launch on sm_120. The ``grid`` /
   ``grid_plant`` kernels themselves are always nvcc-built for the
   detected arch and are unaffected.

``grid_plant`` cost / barrier / plant-step methods
--------------------------------------------------

The handle also exposes the generated ``grid_plant`` trajectory-
optimization surface (validated against ``RBDReference._PlantMixin``).
All take/return 2D arrays with axis 0 = batch; cost methods return
``(value, grad, hess)`` and barriers return ``(value, grad, hess_diag)``.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Method
     - Returns
   * - ``quadratic_state_cost(x, x_des, Q)``
     - ``value (B,)``, ``grad (B, NX)``, ``hess (B, NX, NX)``
   * - ``quadratic_input_cost(u, u_des, R)``
     - ``value (B,)``, ``grad (B, NV)``, ``hess (B, NV, NV)``
   * - ``ee_pos_cost(q, p_des, W)``
     - ``value (B,)``, ``grad (B, NX)``, Gauss-Newton ``hess (B, NX, NX)``
   * - ``joint_position_barrier(var, lower, upper, mu)``
     - ``value (B,)``, ``grad (B, NP)``, ``hess_diag (B, NP)``
   * - ``joint_velocity_barrier(var, lower, upper, mu)``
     - as above over ``NV``
   * - ``joint_torque_barrier(var, lower, upper, mu)``
     - as above over ``NV``
   * - ``plant_step(x, u, dt, integrator_type="euler")``
     - ``(B, NX)`` next state

External forces (``f_ext``)
---------------------------

Per-body external forces are an opt-in feature of the underlying CUDA
codegen and the ``RBDReference`` oracle (body-local frame, subtracted
from the per-body force; an empty/``None`` value reproduces the no-force
path). The generated host wrappers carry the ``d_f_ext`` argument; an
``f_ext=`` kwarg on the ``RobotHandle`` algorithm methods is on the
roadmap.

Coming next
-----------

* Floating-base JAX FFI for ``idsva_so`` (currently routes to the
  body-frame kernel; world-frame fallback for floating-base needs the
  codegen to emit a preprocessor-visible dispatcher).
* An ``f_ext=`` kwarg on the Python handle algorithm methods (the CUDA
  codegen already threads ``d_f_ext``).
* CLI shortcut: ``grid-rbd register iiwa.urdf --name iiwa14``.

See also
--------

* :doc:`benchmarks` — bench harness and what the methods cost.
* :doc:`../concepts/algorithms/index` — algorithm-level docs.
* ``docs/python_wrappers_plan.md`` (repo root) — design rationale.

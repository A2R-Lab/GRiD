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

Source: ``bindings/`` in the GRiD repo.

.. seealso::

   In a hurry? :doc:`../getting_started/fast_robot_setup` is the quick
   canonical path — one-call ``load_robot``, cache anatomy, and warm-up
   recipes.

Install (editable, from a GRiD checkout)
----------------------------------------

.. code-block:: shell

   cd path/to/GRiD
   pip install -e .

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
``bindings/examples/quickstart_iiwa14.py``.

Method surface
--------------

All methods take and return 2D ``float32`` arrays where axis 0 is the
batch. ``gravity`` is the **signed gravitational acceleration**, default
``-9.81`` (standard downward gravity) — the same convention as
``RBDReference.inverse_dynamics(..., GRAVITY=-9.81)`` and pinocchio, so
pass the same value to both for cross-validation.

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Method
     - Returns
     - Notes
   * - ``inverse_dynamics(q, qd, qdd=None)``
     - ``(B, NJ)``
     - Inverse dynamics bias (RNEA); alias ``rnea``. ``qdd=None`` ⇒ bias
       ``c = h − g``; a nonzero ``qdd`` adds the ``M·qdd`` term.
   * - ``minv(q)``
     - ``(B, NJ, NJ)``
     - Direct mass-matrix inverse.
   * - ``forward_dynamics(q, qd, u)``
     - ``(B, NJ)``
     - ``M⁻¹·(τ − c)``; alias ``fd``.
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
   * - ``inverse_dynamics_gradient(q, qd, qdd=None)``
     - ``(B, NJ, 2*NJ)``
     - ``[dc_dq | dc_dqd]``.
   * - ``forward_dynamics_gradient(q, qd, u)``
     - ``(B, NJ, 2*NJ)``
     - ``[df_dq | df_dqd]``.
   * - ``idsva_so(q, qd, qdd=None)``
     - tuple of 4 ``(B, NV, NV, NV)``
     - Second-order ID (dispatched).
   * - ``fdsva_so(q, qd, u)``
     - tuple of 4 ``(B, NV, NV, NV)``
     - Second-order FD.
   * - ``com(q)`` / ``ccrba(q, qd)`` / ``energy(q, qd)``
     - varies
     - Centroidal kinematics (CoM + CoM Jacobian; CMM ``A`` + momentum ``h``;
       KE/PE/mechanical energy).
   * - ``coriolis_matrix(q, qd)``
     - ``(B, NV, NV)``
     - Coriolis matrix ``C(q,q̇)`` (``C·q̇ + g = nonlinear_effects``).
   * - ``kinetic_energy_regressor(q, qd)`` / ``potential_energy_regressor(q)``
     - ``(B, 10*NB)``
     - Inertial-parameter energy regressors (``E = y·π``).
   * - ``dccrba(q)``
     - ``(B, 6, NV, NV)``
     - ∂A/∂q tensor (centroidal-momentum-matrix derivative).
   * - ``cmm_time_variation(q, qd)``
     - ``(B, 6, NV)``
     - Ȧ = ``Σ_i (∂A/∂q_i)·q̇_i``.
   * - ``frame_jacobian(q)`` / ``frame_jacobian_dot(q, qd)`` / ``osc_inertia(q)``
     - ``(B, 6, NV)`` / ``(B, 6, NV)`` / ``(B, 6, 6)``
     - General-frame J / J̇ (runtime ``target_jid`` + ``reference_frame``) and
       operational-space inertia Λ.
   * - ``end_effector_pose_runtime(q, ee_joint_names=None, ee_offsets=None)``
     - ``(B, 6*NUM_EES)``
     - Runtime arbitrary multi-EE pose (runtime target joints + per-target
       offset); ``end_effector_pose_gradient_runtime`` returns its Jacobian.

The numeric methods also accept ``allow_fp64=True`` at ``register_robot`` for
an fp64-in/fp64-out convenience cast (compute stays fp32).

Validation against ``RBDReference`` lives at
``test/python_wrappers/test_iiwa14_smoke.py`` (16 tests, all numerical
methods pass at float32 precision).

Build cost on large floating-base robots
----------------------------------------

On a floating-base, non-mimic robot GRiD emits **two** variants of each
kernel: the Pinocchio-convention ("pin") kernel and a MuJoCo-convention
("mjx") twin applying the ``G = blockdiag(R, I)`` output basis change.
That convention change is cheap in principle but was expensive in generated
code; the second-order mjx epilogues have since been block-parallelized, which
cut them substantially. Current mjx SASS relative to pin (measured
**go2-floating**, ``nv=18``):

.. list-table::
   :header-rows: 1

   * - Kernel
     - mjx / pin
   * - ``idsva_so_world_frame``
     - **2.42x** (was 5.5x rolled / 28x raw)
   * - ``fdsva_so``
     - **1.41x** (was 3.0x)
   * - ``inverse_dynamics_gradient``, ``forward_dynamics``, ``minv``, ``crba``
     - ~1.0x

The second-order mjx twins are still the largest kernels in a humanoid build,
so if you do not need the MuJoCo convention, pin-only is lighter to compile.

If you do not need the MuJoCo output convention, build pin-only:

.. code-block:: python

   handle = grid_rbd.register_robot(
       name="g1", urdf_path="g1.urdf", floating_base=True,
       enable_mujoco_kernels=False,
   )

On g1-floating that is the difference between not building at all and a
~33 min build at ~11 GB peak. Fixed-base and mimic robots (e.g. ``h1_2``)
never get mjx twins, so the flag is a no-op there. It is mutually
exclusive with ``output_convention="mujoco"``, and participates in the
``.so`` cache key only when ``False``, so existing caches stay valid.
The generator emits a warning naming this flag when it detects a large
floating-base non-mimic robot.

For test suites and codegen sessions, ``GRID_ENABLE_MUJOCO_KERNELS=0``
makes pin-only the default for every ``gen_all_code`` call that does not
pass the argument explicitly (an explicit argument always wins). It does
**not** affect ``register_robot``/``precompile``, whose ``.so`` is cached
under the option dict — an env var that changed the build without changing
the cache key would return a stale ``.so``.

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
grid_rbd_version + cuda_arch`` **plus** the wrapper-template hash, the
codegen-source hash (every ``grid_codegen/``/``URDFParser`` file and
``_compile.py``), and the torch/jax ABI tags — editing the codegen or the
wrapper rotates every key, so rebuilds happen automatically and
``force_rebuild`` is never needed for that. CUDA arch in the key means a roaming
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

Install with ``pip install -e ".[jax]"`` to get the JAX bridge.
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

JAX surface: the core dynamics / kinematics / SO methods are bound via
FFI and JIT-compatible (the SO methods ``idsva_so`` / ``fdsva_so`` follow
the plain wrapper's tuple-of-four convention), with autograd-aware
``inverse_dynamics`` / ``forward_dynamics`` (qdd-aware), ``end_effector_pose``,
``f_ext`` parity, and the inertial-parameter (π) regressor VJP path. The
centroidal / kinematics value ops (``generalized_gravity``,
``nonlinear_effects``, ``energy``, ``com``, ``ccrba``, ``dccrba``,
``cmm_time_variation``, ``coriolis_matrix``, ``frame_jacobian`` /
``frame_jacobian_dot``, ``osc_inertia``, and the KE/PE regressors) are now
exposed on the jax and torch surfaces too (forward-only, no autograd).

PyTorch backend (``backend="torch"``)
-------------------------------------

``register_robot(..., backend="torch")`` returns a ``TorchRobotHandle``
whose methods return ``torch.Tensor``. The four differentiable
algorithms (``inverse_dynamics`` / ``forward_dynamics`` / ``aba`` / ``integrator``)
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
   * - ``com_cost(q, p_des, W)``
     - ``value (B,)``, ``grad (B, NX)``, Gauss-Newton ``hess (B, NX, NX)`` (CoM tracking)
   * - ``momentum_cost(q, qd, h_des, W)``
     - ``value (B,)``, ``grad (B, NX)``, Gauss-Newton ``hess (B, NX, NX)`` (centroidal-momentum tracking)
   * - ``joint_position_barrier(var, lower, upper, mu)``
     - ``value (B,)``, ``grad (B, NP)``, ``hess_diag (B, NP)``
   * - ``joint_velocity_barrier(var, lower, upper, mu)``
     - as above over ``NV``
   * - ``joint_torque_barrier(var, lower, upper, mu)``
     - as above over ``NV``
   * - ``plant_step(x, u, dt, integrator_type="euler")``
     - ``(B, NX)`` next state
   * - ``plant_step_gradient(x, u, dt, integrator_type="euler")``
     - ``(B, 2*NV, 3*NV)`` ``[A|B]`` = ``d x_{k+1}/d(x,u)``
   * - ``plant_step_hessian(x, u, dt, integrator_type="euler")``
     - ``(B, 2*NV, 3*NV, 3*NV)`` second-order sensitivity ``d²x_{k+1}/d(x,u)²``
       (fixed- and floating-base, euler/semi-implicit-euler; RK deferred)

External forces (``f_ext``)
---------------------------

Per-body external forces are an opt-in feature of the underlying CUDA
codegen and the ``RBDReference`` oracle (body-local frame, subtracted
from the per-body force; an empty/``None`` value reproduces the no-force
path). The generated host wrappers carry the ``d_f_ext`` argument, and an
``f_ext=`` kwarg is exposed on the ``RobotHandle`` algorithm methods that
support it (``inverse_dynamics`` / ``forward_dynamics`` / ``aba`` and
their gradients); the default ``None`` reproduces the no-force path.

Coming next
-----------

* Floating-base JAX FFI for ``idsva_so`` (currently routes to the
  body-frame kernel; world-frame fallback for floating-base needs the
  codegen to emit a preprocessor-visible dispatcher).

See also
--------

* :doc:`benchmarks` — bench harness and what the methods cost.
* :doc:`../concepts/algorithms/index` — algorithm-level docs.
* ``docs/open-tasks/archive/python_wrappers_plan.md`` — design rationale (archived).

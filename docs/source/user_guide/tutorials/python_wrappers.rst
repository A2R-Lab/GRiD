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
   * - ``inverse_dynamics_regressor(q, qd, qdd=None)``
     - ``(B, NV, 10*NB)``
     - Joint-torque inertial-parameter regressor ``Y`` (``tau = Y·π``);
       bound on all three backends.
   * - ``inverse_dynamics_wrt_params(q, qd, params)`` /
       ``forward_dynamics_wrt_params(q, qd, u, params)``
     - ``(B, NJ)``
     - jax/torch only: ID / FD as differentiable-in-π ops — autograd flows
       the analytic ``∂c/∂π = Y`` / ``∂q̈/∂π = −M⁻¹Y`` to ``params``.
   * - ``forward_dynamics_parameter_gradient(q, qd, u)``
     - ``(B, NV, 10*NB)``
     - jax/torch only: the FD inertial-parameter gradient ``∂q̈/∂π = −M⁻¹Y``.
   * - ``attach_tool(joint, mass=..., ...)`` / ``detach_tool()`` /
       ``tool_fext(q, wrench)``
     - varies / — / ``(B, 6*NB)``
     - Runtime welded tool/payload (``enable_tool=True``): numpy computes
       the composed payload inertia + tool-tip frame and delegates to the
       runtime tables; ``tool_fext`` maps a world-aligned tool-tip wrench to
       joint-local ``f_ext``.
   * - ``contact_fext(q, f_c)`` (+ the ``contact_frames`` property)
     - ``(B, 6*NB)``
     - Multi-contact: maps per-contact-frame world-aligned wrenches to
       joint-local ``f_ext``; needs a ``register_robot(contact_frames=[...])``
       build.
   * - ``set_inertia_params(t)`` / ``set_transform_params(t)`` /
       ``set_joint_dynamics(damping=, friction=)``
     - —
     - Runtime-mutable model tables (flag-gated builds; no recompile).
   * - ``fk_batched(q)``
     - ``(B, 7)``
     - Large-batch forward kinematics for the leaf EE frame
       (``[xyz, quat wxyz]``); ``use_warp=True`` selects the warp-cooperative
       variant.

The numeric methods also accept ``allow_fp64=True`` at ``register_robot`` for
an fp64-in/fp64-out convenience cast (compute stays fp32).

Validation against ``RBDReference`` lives at
``test/python_wrappers/test_iiwa14_smoke.py`` (28 tests, all numerical
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
   ├── manifest.json              # name -> content key (writers take manifest.lock)
   ├── bykey/<input_key>          # stage-1 pointer -> content key
   └── store/<content_key>/
       ├── grid.cuh
       ├── wrapper.cu
       ├── robot.so
       ├── meta.json
       ├── build_inputs.json      # the build identity this entry was built under
       └── robot.build.log

Two keys. The **stage-1 input key** = SHA-256 of ``urdf_bytes +
canonical_json(options) + grid_rbd_version`` **plus** the codegen-source
hash (every ``grid_codegen/``/``URDFParser`` file) and the *build identity*:
CUDA arch, ``nvcc`` path + version, host C++ compiler, the content of the
vendored GLASS headers, ``_compile.py``, the wrapper template, the torch/jax
ABI tags and the generation-time env knobs — the one list in
``grid_codegen/env_knobs.py`` (``GRID_CUDA_TARGET_SHARED_MEM_BYTES``,
``GRID_CUDA_TARGET_LITE_SHARED_MEM_BYTES``, ``GRID_CUDA_SHARED_MEM_TYPE_SIZE_BYTES``,
``GRID_NO_LICM_BARRIER``, ``GRID_FDSVA_SO_MINV_TILE``, ``GRID_GLASS_REVISION``; the
benchmark and equivalence header caches key the same list, and a test keeps it equal
to the reads in the tree). The **stage-2 content key** = SHA-256 of exactly
what ``nvcc`` compiled (generated bytes + flag drivers + toolchain); the
``.so`` lives under it. A stage-1 pointer is honoured only when the entry's
``build_inputs.json`` equals the current identity, so a toolkit upgrade or a
dirty GLASS checkout can never return a stale ``.so``; a codegen edit whose
emitted bytes are unchanged re-runs generation only (no ``nvcc``), and
``force_rebuild`` is never needed for correctness. CUDA arch and toolchain in
the key mean a roaming home directory (e.g. NFS-mounted between a laptop and
a desktop) safely keeps separate ``.so`` files per machine.

Override the cache root with ``$GRID_RBD_CACHE_DIR`` or
``cache_dir=...`` on ``register_robot``.

Dimensions, layouts and differentiability
-----------------------------------------

**Dimension names.** Every handle exposes three read-only widths (the legacy
names remain as aliases):

.. list-table::
   :header-rows: 1
   :widths: 14 22 64

   * - Name
     - Legacy alias
     - Meaning
   * - ``nq``
     - ``num_joints``
     - Configuration width: ``q`` is ``(B, nq)``. Fixed base: the joint
       count. Floating base: ``7 + joints`` — ``[p(3), quat xyzw(4), joints]``.
   * - ``nv``
     - ``num_vel``
     - Tangent width: matrix and gradient **outputs** are ``nv``-wide
       (``minv`` is ``(B, nv*nv)``, ``*_gradient`` blocks are ``nv`` columns).
       Floating base: ``6 + joints`` — ``[v_lin(3), omega(3), joints]`` in the
       pinocchio LOCAL convention (``output_convention="mujoco"`` flips the
       root block, see :doc:`../concepts/mjx_convention`).
   * - ``nb``
     - ``num_bodies``
     - Body count (base included on a floating base): ``f_ext`` is
       ``(B, 6*nb)``, one spatial force per body in the local body frame.

**Velocity-space inputs are nq-wide.** ``qd``, ``qdd`` and ``u`` are passed
at the ``nq`` stride: on a floating base the tangent occupies the first
``nv`` slots and the trailing slot is a 0 pad (the raw transport layout the
kernels read). An ``nv``-wide input raises a clear ``ValueError`` rather than
being padded silently; outputs never carry the pad. Accepting natural
``nv``-wide velocity inputs at the high-level API is a registered design
item (audit W11) — not done, so as not to change existing shapes quietly.

**What is differentiable.** ``jax`` and ``torch`` handles attach an analytic
backward (a batched VJP through the ``*_gradient`` kernels) to exactly these
methods; everything else is forward-only on every backend.

.. list-table::
   :header-rows: 1
   :widths: 30 26 44

   * - Method
     - Differentiable inputs
     - Notes
   * - ``forward_dynamics`` / ``aba``
     - ``q``, ``qd``, ``u``
     - ``f_ext`` is a residual, its cotangent is zero. ``∂qdd/∂u = M⁻¹``.
   * - ``inverse_dynamics``
     - ``q``, ``qd``
     - ``qdd`` and ``f_ext`` are residuals (the derivative is taken AT the
       saved acceleration/force); their cotangents are zero.
   * - ``end_effector_pose``
     - ``q``
     - the derivative of the returned coordinates ``[xyz, rpy]`` (6 per EE),
       i.e. a coordinate, not a geometric, orientation derivative.
       ``fk_batched`` is the 7-coordinate position + quaternion surface and is
       forward-only.
   * - ``integrator``
     - ``q``, ``qd``, ``u``
     - **fixed-base only** (the SE(3)-chart VJP is not implemented); through
       the baked ``integrator_with_gradient`` kernel (the same fused step +
       Jacobian that ``integrator_gradient`` / ``plant_step_gradient`` return);
       ``dt``/steps are static.
   * - ``inverse_dynamics_wrt_params``
     - ``params``, ``q``, ``qd``
     - local parameter sensitivity (``∂τ/∂π`` via the regressor at
       ``qdd = 0``) PLUS the ordinary ``q``/``qd`` cotangents; see below.
   * - ``forward_dynamics_wrt_params``
     - ``params``, ``q``, ``qd``, ``u``
     - local parameter sensitivity (``∂qdd/∂π = -M⁻¹Y``) PLUS the ordinary
       ``q``/``qd``/``u`` cotangents; see below.

Each of these is a **reverse-mode VJP only**: there is no forward-mode
(``jax.jvp`` / ``jax.jacfwd`` through them fails), no higher-order autograd
(differentiating the backward again is not supported — use the explicit
second-order kernels ``idsva_so`` / ``fdsva_so`` / ``end_effector_pose_hessian``,
which are forward-only outputs), and ``jax.vmap`` composes only along the
batch axis. The ``q`` cotangent on a floating base is the exact ambient
pullback through the kernel's quaternion normalization (pinocchio handles)
or the on-manifold pullback (``mjx`` handles); see *Gradient semantics* under
the JAX section.

**``*_wrt_params`` are local sensitivities, not parameterized functions.**
``inverse_dynamics_wrt_params(q, qd, params)`` returns the SAME torque the
baked (or current runtime) inertial table gives, whatever ``params`` you pass;
the operand exists so that the backward can supply ``∂τ/∂π`` through the
regressor VJP. Use them as the linearization point for sysID / calibration
(the gradient with respect to ``π`` at the current model), not as a function
you can evaluate at a different ``π`` — to actually change the model, call
``set_inertia_params`` (``runtime_inertia=True`` builds). A functional
parameterized forward is a registered design decision (audit W11, option B),
not something the current API pretends to be.

**Runtime tables reach dynamics, not baked kinematics.** ``runtime_inertia``,
``runtime_transform`` and ``runtime_joint_dynamics`` rebuild the dynamics
tables from a mutable device table once per launch, so ``inverse_dynamics``,
``forward_dynamics``, their gradients and the second-order kernels follow
``set_*_params`` with no recompile. The homogeneous-transform path used by
``end_effector_pose`` / ``end_effector_pose_gradient`` / ``_hessian`` /
``frame_jacobian`` / ``fk_batched`` reads the BAKED joint origins: after
``set_transform_params`` the kinematics still report the URDF geometry. That
is a documented partial capability, not a bug you can work around from
Python; consistent kinematic updates are a separate feature (register item).

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

Contact frames and welded tools
-------------------------------

``register_robot(contact_frames=[...])`` takes a list of **fixed-joint
names** and bakes the contact family into the build: the handle gains the
``contact_fext`` method (per-contact-frame world-aligned wrenches →
joint-local ``f_ext``) and a ``contact_frames`` property. The kwarg
re-keys the ``.so`` cache, so contact and non-contact builds coexist. It
is accepted by every backend's ``register_robot`` (NumPy, JAX, PyTorch)
and composes with a subset ``algorithm_list`` — e.g.
``algorithm_list=["inverse_dynamics", "forward_dynamics"]`` plus
``contact_frames=[...]`` builds only the dynamics cores and the contact
family (the ``bindings/examples/multi_contact_fext.py`` recipe).

The baked contact set also exposes its ORIGINS to CUDA/plant consumers
(GATO ask 2026-09-20): ``grid::contact_frame_positions_device`` (the
``3*NUM_CONTACT_FRAMES`` world positions of the points ``f_ext_body`` takes
the wrench about, registration order) and
``grid::contact_frame_positions_gradient_device`` (``3 x NUM_VEL`` per
frame, layout ``[3*NUM_VEL*f + 3*vi + row]``, floating tangent
``[v_lin; omega; joints]``), with the caller-scratch wrappers
``grid_plant::contact_frame_positions[_gradient](…, s_scratch, …)`` sized by
``CONTACT_FRAME_POSITIONS[_GRADIENT]_DYNAMIC_SHARED_MEM_COUNT``. They ride
the multi-target emitters, so a dynamics-only ``algorithm_list`` still gets
them. (No Python method yet — the device/plant layer is the consumer.)
The same caller-scratch pair exists for the DEFAULT multi-target batch (the
``multi_target_batch`` option / collision spheres):
``grid_plant::multi_target_position[_gradient](…, s_scratch, …)`` with
``s_scratch`` sized by the constexpr
``MULTI_TARGET_POSITION[_GRADIENT]_DYNAMIC_SHARED_MEM_BYTES<T>()`` — the raw
evaluator a consumer's own FK carve would otherwise have to compose.

``register_robot(enable_tool=True)`` enables the runtime welded-tool
surface — ``attach_tool`` / ``detach_tool`` / ``tool_fext`` — by turning
on the runtime inertia table + runtime contact surface, all with no
recompile at attach time. See
``bindings/examples/AGENT_INTEGRATION_GUIDE.md`` and
``bindings/examples/tool_use.py`` for the full recipes.

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
whose methods return ``torch.Tensor``. The differentiable
algorithms (``inverse_dynamics`` / ``forward_dynamics`` / ``aba`` /
``end_effector_pose`` / ``integrator``) are autograd-aware — their backward
passes are analytic, reusing the existing ``*_gradient`` kernels — while the
remaining methods are forward-only ops. The ``.so`` is shared with the numpy/JAX surfaces; the
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

**What the gradients mean (both backends, audit 2026-09-19).**

* **Floating base, w.r.t. ``q``.** The analytic kernels differentiate in the
  Pinocchio free-flyer *tangent* chart (``[v_lin local, ω local, joints]``,
  nv-wide; that is what ``inverse_dynamics_gradient`` etc. return). The
  public ``q`` is ``[pos(3), quat_xyzw(4), joints]`` (nq = nv+1), and
  ``jax.grad`` / ``q.grad`` return the exact pullback to THAT layout: the
  kernels evaluate ``R(p/|p|)``, so the value is ``f ∘ normalize`` on the
  ambient quaternion and the gradient is its true ambient derivative
  (central differences over any ``q`` component agree, including for a
  non-unit quaternion; the radial direction is a null direction). Under
  ``output_convention="mujoco"`` the twins use the mjx free-joint chart
  (linear world, angular local, ``quat_wxyz``) and expect a UNIT quaternion
  (they do not renormalize): the returned ``q`` cotangent is the on-manifold
  pullback (zero radial component). Need the tangent-space gradient instead?
  Contract your cotangent with the ``*_gradient`` Jacobian yourself.
  Spherical joints have their own four-position/three-tangent quaternion
  blocks. Their pullbacks use the model's per-joint coordinate offsets, not
  a floating-root assumption. Old spherical artifacts without this layout
  metadata must be re-registered before using framework autodiff.
* **External forces.** ``f_ext`` is a non-differentiated input, but the
  ``q``/``qd`` gradients are taken *at* the given force (a fixed body-local
  wrench has q-dependent joint torques). ``f_ext`` must be ``(B, 6*num_bodies)``
  in the body-local frame; on the jax surface ``(6*num_bodies,)`` and
  ``(1, 6*num_bodies)`` broadcast forms are materialized to the batch, and any
  other leading shape is rejected before the FFI call (the native handlers
  reject a batch mismatch too).

**Shared runtime lifetime.** Backend handles referring to the same compiled
artifact share its model and arena. Closing or collecting one handle does not
reset the others. Framework-allocated arena buffers are retained by the shared
native owner until the last Runner closes, and outstanding device work is
completed before that buffer is released. A fresh runtime allocates lazily on
its first operation or parameter update, so CUDA allocation errors can surface
there rather than during registration. A framework view opened after a NumPy
arena is already live reuses it; it does not replace the allocator or erase
inertia, transform, or tool updates. Isolation is opt-in: ``handle.context()``
(numpy, jax and torch handles alike) opens a **runtime context** of its own on
the same artifact — its own arena, tables, streams and launch overrides — see
the *Runtime contexts* concepts page. Concurrent operations on ONE context's
scratch are still not independently safe (per-call leases are a later
increment). Runtime-parameter mutations (``set_inertia_params``,
``attach_tool`` …) are serialized against every in-flight call and bump
``handle.model_version``; a torch or JAX backward whose forward ran under an
older version raises (``model mutated between forward and backward``) instead
of differentiating the new model — recompute the forward after mutating.

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

See also
--------

* :doc:`benchmarks` — bench harness and what the methods cost.
* :doc:`../concepts/algorithms/index` — algorithm-level docs.
* ``docs/open-tasks/archive/python_wrappers_plan.md`` — design rationale (archived).

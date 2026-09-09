"""Shared machinery for the numpy/jax/torch robot-handle surfaces (C3 fold).

Three mixins, each previously hand-copied per surface:

- :class:`MujocoViewBase` — the ``handle.mujoco`` view's shared core: MuJoCo
  parameter names, the mjx output convention applied PER CALL (an explicit
  ``_convention="mujoco"`` forwarded to the owning handle, never mutating its
  shared ``output_convention``), thread-safe next to pinocchio-convention calls.
  The five value/dynamics methods every surface exposes live here.
- :class:`MujocoDerivativeViewMixin` — the derivative / kinematics / integrator
  / plant-cost view methods (all THREE views compose it since the A4 roster
  unification, 2026-09-09 — the numpy view used to expose only values).
- :class:`BaseDelegateMixin` — jax/torch handle methods that pure-delegate to
  the underlying numpy :class:`~grid_rbd._handle.RobotHandle` (``self._base``):
  tool welding, runtime parameter tables, thread-count control, metadata.

Every method body is the exact code that lived in the per-surface copies; only
surface-specific docstring wording was neutralized.
"""


class MujocoViewBase:
    """Shared core of the MuJoCo-native ``handle.mujoco`` view: MuJoCo parameter
    names (``qpos``/``qvel``/``qacc``/``qfrc``), mjx output convention applied per
    call via ``_convention="mujoco"`` (thread-safe; never mutates the handle's
    shared default). On a fixed base the convention is a no-op (no free-flyer)."""

    __slots__ = ("_h",)

    def __init__(self, handle) -> None:
        self._h = handle

    # ── value / dynamics ──────────────────────────────────────────────────
    def inverse_dynamics(self, qpos, qvel, qacc=None, *, gravity: float = -9.81, f_ext=None):
        """RNEA in MuJoCo convention: τ = id(qpos, qvel, qacc). Returns mjx-frame τ."""
        return self._h.inverse_dynamics(qpos, qvel, qacc, gravity=gravity, f_ext=f_ext,
                                        _convention="mujoco")

    def forward_dynamics(self, qpos, qvel, qfrc, *, gravity: float = -9.81, f_ext=None):
        """Forward dynamics in MuJoCo convention: qacc = fd(qpos, qvel, qfrc)."""
        return self._h.forward_dynamics(qpos, qvel, qfrc, gravity=gravity, f_ext=f_ext,
                                        _convention="mujoco")

    def aba(self, qpos, qvel, qfrc, *, gravity: float = -9.81, f_ext=None):
        """Articulated-body forward dynamics in MuJoCo convention."""
        return self._h.aba(qpos, qvel, qfrc, gravity=gravity, f_ext=f_ext, _convention="mujoco")

    def crba(self, qpos, *, gravity: float = -9.81):
        """Mass matrix M(qpos) in the mjx frame (G M G^T)."""
        return self._h.crba(qpos, gravity=gravity, _convention="mujoco")

    def minv(self, qpos):
        """Inverse mass matrix Minv(qpos) in the mjx frame (G^-T Minv G^-1)."""
        return self._h.minv(qpos, _convention="mujoco")

    def __repr__(self) -> str:
        return f"<mujoco view of {self._h!r}>"


    # ── centroidal / energy (A4 roster unification, 2026-09-09: previously
    # numpy-view-only; all three handles carry these with _convention=) ──
    def com(self, qpos):
        """CoM position (invariant) + CoM Jacobian (reframed) in the mjx frame."""
        return self._h.com(qpos, _convention="mujoco")

    def ccrba(self, qpos, qvel):
        """Centroidal momentum matrix (reframed) + momentum h (invariant), mjx frame."""
        return self._h.ccrba(qpos, qvel, _convention="mujoco")

    def energy(self, qpos, qvel, *, gravity: float = -9.81):
        """Kinetic / potential / mechanical energy (frame-invariant) from mjx inputs."""
        return self._h.energy(qpos, qvel, gravity=gravity, _convention="mujoco")

    def kinetic_energy_regressor(self, qpos, qvel, *, gravity: float = -9.81):
        """Kinetic-energy regressor (frame-invariant) from mjx inputs."""
        return self._h.kinetic_energy_regressor(qpos, qvel, gravity=gravity, _convention="mujoco")

    def potential_energy_regressor(self, qpos, *, gravity: float = -9.81):
        """Potential-energy regressor (frame-invariant) from mjx inputs."""
        return self._h.potential_energy_regressor(qpos, gravity=gravity, _convention="mujoco")


class MujocoDerivativeViewMixin:
    """The jax/torch-shared derivative / kinematics / integrator / plant-cost
    methods of the ``handle.mujoco`` view (compose with :class:`MujocoViewBase`)."""

    __slots__ = ()

    # ── kinematics / regressor ────────────────────────────────────────────
    def end_effector_pose(self, qpos):
        """End-effector pose from mjx-convention qpos."""
        return self._h.end_effector_pose(qpos, _convention="mujoco")

    def end_effector_pose_gradient(self, qpos):
        """EE pose Jacobian reframed to the mjx free-joint tangent (J·G^-1)."""
        return self._h.end_effector_pose_gradient(qpos, _convention="mujoco")

    def end_effector_pose_hessian(self, qpos):
        """EE pose Hessian in the mjx convention."""
        return self._h.end_effector_pose_hessian(qpos, _convention="mujoco")

    def inverse_dynamics_regressor(self, qpos, qvel, qacc=None, *, gravity: float = -9.81):
        """Joint-torque regressor with base-linear rows in the mjx frame."""
        return self._h.inverse_dynamics_regressor(qpos, qvel, qacc, gravity=gravity,
                                                  _convention="mujoco")

    # ── first / second-order derivatives ──────────────────────────────────
    def inverse_dynamics_gradient(self, qpos, qvel, qacc=None, *, gravity: float = -9.81):
        """∂τ/∂(q,qd) in the mjx convention."""
        return self._h.inverse_dynamics_gradient(qpos, qvel, qacc, gravity=gravity,
                                                 _convention="mujoco")

    def forward_dynamics_gradient(self, qpos, qvel, qfrc, *, gravity: float = -9.81):
        """∂qacc/∂(q,qd) in the mjx convention."""
        return self._h.forward_dynamics_gradient(qpos, qvel, qfrc, gravity=gravity,
                                                 _convention="mujoco")

    def idsva_so(self, qpos, qvel, qacc=None, *, gravity: float = -9.81):
        """Second-order inverse dynamics (4 tensors) in the mjx convention."""
        return self._h.idsva_so(qpos, qvel, qacc, gravity=gravity, _convention="mujoco")

    def fdsva_so(self, qpos, qvel, qfrc, *, gravity: float = -9.81):
        """Second-order forward dynamics (4 tensors) in the mjx convention."""
        return self._h.fdsva_so(qpos, qvel, qfrc, gravity=gravity, _convention="mujoco")

    # ── integrator / plant ────────────────────────────────────────────────
    def integrator(self, qpos, qvel, qfrc, dt, *, integrator_type: str = "euler",
                   gravity: float = -9.81):
        """One integration step in the mjx convention (global-additive retract)."""
        return self._h.integrator(qpos, qvel, qfrc, dt, integrator_type=integrator_type,
                                  gravity=gravity, _convention="mujoco")

    def integrator_gradient(self, qpos, qvel, qfrc, dt, *, integrator_type: str = "euler",
                            gravity: float = -9.81):
        """Integrator state-transition Jacobian in the mjx convention."""
        return self._h.integrator_gradient(qpos, qvel, qfrc, dt, integrator_type=integrator_type,
                                           gravity=gravity, _convention="mujoco")

    def plant_step(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """Plant step x_{k+1} in the mjx convention."""
        return self._h.plant_step(x, u, dt, integrator_type=integrator_type, gravity=gravity,
                                  _convention="mujoco")

    def plant_step_gradient(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """Plant-step state-transition Jacobian in the mjx convention."""
        return self._h.plant_step_gradient(x, u, dt, integrator_type=integrator_type,
                                           gravity=gravity, _convention="mujoco")

    def quadratic_state_cost(self, x, x_des, Q):
        """Quadratic state cost (value/grad/GN-hess) in the mjx convention."""
        return self._h.quadratic_state_cost(x, x_des, Q, _convention="mujoco")

    def ee_pos_cost(self, qpos, p_des, W):
        """End-effector position tracking cost in the mjx convention."""
        return self._h.ee_pos_cost(qpos, p_des, W, _convention="mujoco")

    def com_cost(self, qpos, p_des, W):
        """Center-of-mass tracking cost in the mjx convention."""
        return self._h.com_cost(qpos, p_des, W, _convention="mujoco")

    def momentum_cost(self, qpos, qvel, h_des, W):
        """Centroidal-momentum tracking cost in the mjx convention."""
        return self._h.momentum_cost(qpos, qvel, h_des, W, _convention="mujoco")


class BaseDelegateMixin:
    """jax/torch handle surface that pure-delegates to the underlying numpy
    ``RobotHandle`` (``self._base``): tool welding, runtime parameter tables,
    thread-count control, metadata, and the shared output-convention contract."""

    @property
    def output_convention(self) -> str:
        """Default IO convention for this handle: ``"pinocchio"`` or ``"mujoco"``.
        Settable. ``"mujoco"`` on a FIXED base is a deliberate no-op (no
        free-flyer, so mjx == pinocchio — same uniform-interface semantics as the
        numpy handle); on a FLOATING base it needs a .so built with the mjx
        kernel twins. Per-call overrides use the thread-safe ``.mujoco`` view."""
        return self._output_convention

    @output_convention.setter
    def output_convention(self, value: str) -> None:
        if value not in ("pinocchio", "mujoco"):
            raise ValueError(
                f"output_convention must be 'pinocchio' or 'mujoco'; got {value!r}")
        if (value == "mujoco" and self.floating_base
                and not getattr(self._base._runner, "has_inverse_dynamics_mujoco", False)):
            raise ValueError(
                "output_convention='mujoco' needs a floating-base .so built with "
                "the mjx kernel twins (this one was built with "
                "enable_mujoco_kernels=False, or the robot is mimic/skew) — "
                "re-register with enable_mujoco_kernels=True.")
        self._output_convention = value

    def _mjx_active(self, convention=None) -> bool:
        """True when mjx-convention IO should actually engage: the resolved
        convention is ``"mujoco"`` AND the robot is floating-base (a fixed base
        has no free-flyer, so mjx coincides with pinocchio and the flag is a
        no-op — mirrors the numpy handle's ``_mjx_active``)."""
        return self._resolve_convention(convention) == "mujoco" and self.floating_base

    def _refuse_mjx_f_ext(self, key: str, f_ext, convention=None) -> None:
        """Refuse a caller ``f_ext`` under the ACTIVE mjx convention for every
        method whose spec row carries ``mjx_rejects_f_ext`` (the mjx kernel
        twins do not reframe external wrenches — dispatching them with f_ext
        returns silently wrong torques/Jacobians; 2026-09-09 layout audit).
        One table-driven implementation of the six-plus hand refusals."""
        if f_ext is None or not self._mjx_active(convention):
            return
        from grid_codegen.abi_specs import ABI_SPECS, MJX_F_EXT_REFUSAL
        if ABI_SPECS[key].mjx_rejects_f_ext:
            raise NotImplementedError(MJX_F_EXT_REFUSAL.format(name=key))

    def attach_tool(self, joint, *, mass, com=(0.0, 0.0, 0.0), inertia=None,
                    tip_transform=None):
        """Weld a rigid tool/payload at runtime (no recompile). Delegates to the base
        handle: composes the payload inertia into ``joint``'s child link and stores an
        optional SE(3) ``tip_transform``. See :py:meth:`grid_rbd.RobotHandle.attach_tool`.
        The inertia change is seen by every surface (numpy/jax/torch share the .so); for
        the tool-tip frame, pass ``ee_offsets=[tip_transform]`` to
        :py:meth:`end_effector_pose_runtime` (or use the numpy handle's stateful default)."""
        return self._base.attach_tool(joint, mass=mass, com=com, inertia=inertia,
                                      tip_transform=tip_transform)

    def detach_tool(self):
        """Remove the attached tool (restore baked inertia). See
        :py:meth:`grid_rbd.RobotHandle.detach_tool`."""
        self._base.detach_tool()

    @property
    def tool(self):
        """The currently attached tool dict or ``None``."""
        return self._base.tool

    def tool_fext(self, q, wrench, *, joint=None, offset=None):
        """World-aligned tool-tip wrench -> joint-local f_ext ``(B, 6*num_bodies)``,
        ready to pass as ``f_ext=`` to the dynamics. Delegates to the base handle
        (returns a numpy array). See :py:meth:`grid_rbd.RobotHandle.tool_fext`."""
        return self._base.tool_fext(q, wrench, joint=joint, offset=offset)

    def close(self) -> None:
        """Release the underlying .so handle (delegates to the base RobotHandle).
        Idempotent. Process-global backend registrations (JAX FFI targets / the
        torch op library) are unaffected."""
        self._base.close()

    @property
    def dtype(self) -> str:
        """Compute precision of the underlying .so (``"float32"`` / ``"float64"``).
        Arrays/tensors passed to the methods follow this dtype (jax fp64 needs
        x64 mode; the torch ops check it)."""
        return self._base.dtype

    def kernel_max_threads(self, algo: str) -> int:
        """Real compiled ``__launch_bounds__`` ceiling of the baked kernel for the
        short autotune key (``id``, ``fd``, ``minv``, ``id_du``, ``ee_pose``, …) —
        ``cudaFuncGetAttributes().maxThreadsPerBlock`` at the kernel's baked
        ``launch_cfg<ALGO>::TIER``. Returns ``-1`` when the key is unknown/not-built
        (the FFI autotune then infers the tier from the swept ceiling)."""
        return self._base.kernel_max_threads(algo)

    @property
    def inertia_params(self):
        """The BAKED 10-param-per-body inertia table, shape ``(num_bodies, 10)``
        (``[m, h(3), I_O(6)]`` per body). Fetch, mutate, and pass to
        :py:meth:`set_inertia_params`. Only on a ``runtime_inertia`` build."""
        return self._base.inertia_params

    @property
    def joint_damping(self):
        """Baked per-v-slot viscous damping (length nv). Only on a
        ``runtime_joint_dynamics`` build."""
        return self._base.joint_damping

    @property
    def joint_friction(self):
        """Baked per-v-slot Coulomb friction (length nv). Only on a
        ``runtime_joint_dynamics`` build."""
        return self._base.joint_friction

    @property
    def max_perf_level_threads(self) -> int:
        """Codegen-time thread-count hint (DOF-aware, warp-rounded); the default
        per-block thread count for kernel launches."""
        return self._base.max_perf_level_threads

    @property
    def runtime_inertia(self) -> bool:
        """True if this robot was registered with ``runtime_inertia=True`` (the
        .so carries a mutable inertia table + :py:meth:`set_inertia_params`)."""
        return self._base.runtime_inertia

    @property
    def runtime_joint_dynamics(self) -> bool:
        """True if registered with ``runtime_joint_dynamics=True`` (mutable
        damping/friction table + :py:meth:`set_joint_dynamics`). The backend kernels read
        the same device-resident table, so a poke through any surface is seen here."""
        return self._base.runtime_joint_dynamics

    @property
    def runtime_transform(self) -> bool:
        """True if registered with ``runtime_transform=True`` (mutable joint-origin
        table + :py:meth:`set_transform_params`). The backend kernels read the same
        device-resident table, so a poke through any surface is seen here."""
        return self._base.runtime_transform

    def set_inertia_params(self, params) -> None:
        """Update the device-resident inertia table at runtime (no recompile).
        ``params`` is the ``(num_bodies, 10)`` (or flat ``10*num_bodies``) table in
        the same basis as :py:attr:`inertia_params`. Only valid on a robot
        registered with ``runtime_inertia=True``."""
        self._base.set_inertia_params(params)

    def set_joint_dynamics(self, damping=None, friction=None) -> None:
        """Update the device-resident damping/friction table at runtime (no
        recompile). ``damping``/``friction`` are length-nv (v-slot indexed); an
        omitted side keeps its baked value. Only on a ``runtime_joint_dynamics``
        build."""
        self._base.set_joint_dynamics(damping=damping, friction=friction)

    def set_threads_per_block(self, n: int) -> None:
        """Force a single global per-block thread count for all subsequent kernel
        launches issued through the underlying .so. By default each algorithm uses
        its own autotuned per-algo thread count; ``n >= 1`` overrides that."""
        self._base.set_threads_per_block(n)

    def set_transform_params(self, params) -> None:
        """Update the device-resident joint-origin table at runtime (no recompile).
        ``params`` is the ``(num_joints, 6)`` (or flat ``6*num_joints``) table in the
        same basis as :py:attr:`transform_params`. Only valid on a robot registered
        with ``runtime_transform=True``."""
        self._base.set_transform_params(params)

    @property
    def threads_per_block(self) -> int:
        """Active global threads-per-block override (``-1`` = per-algo autotuned
        default baked into ``grid.cuh``; ``>= 1`` = a forced global override)."""
        return self._base.threads_per_block

    @property
    def transform_params(self):
        """The BAKED per-joint origin table, shape ``(num_joints, 6)``
        (``[x, y, z, r, p, y]`` per joint). Fetch, mutate, and pass to
        :py:meth:`set_transform_params`. Only on a ``runtime_transform`` build."""
        return self._base.transform_params

    # ── parity trio (N2.6, 2026-09-08): URDF-limit metadata + launch-config
    # overlays, forwarded so jax/torch handles match the numpy surface. ──

    @property
    def joint_pos_limits(self):
        """Per-joint position limits ``[lower, upper]`` (index == joint id) from
        the URDF; ``None`` where unspecified. Metadata only."""
        return self._base.joint_pos_limits

    @property
    def joint_vel_limits(self):
        """Per-joint velocity limits (index == joint id) from the URDF;
        ``None`` where unspecified. Metadata only."""
        return self._base.joint_vel_limits

    @property
    def joint_effort_limits(self):
        """Per-joint effort (torque) limits (index == joint id) from the URDF;
        ``None`` where unspecified. Metadata only."""
        return self._base.joint_effort_limits

    def apply_profile_overlay(self, profile: str) -> int:
        """Apply the tuned per-algo threads/tier overlay block ``profile`` from
        ``config/launch_configs/<robot>/<gpu>.json`` to the shared underlying
        ``.so`` (all surfaces over it see the change). Returns the number of
        algos overlaid (0 = no config for this robot/GPU/profile)."""
        return self._base.apply_profile_overlay(profile)

    def apply_batch_overlay(self, profile: str = "ffi") -> int:
        """Apply the tuned small-batch switch table (threshold → n_small
        threads per algo) for ``profile`` to the shared underlying ``.so``.
        Returns the number of algos configured."""
        return self._base.apply_batch_overlay(profile)


# ── Deliberate backend asymmetries (documented, not bugs — N2.6) ─────────────
# The jax/torch surfaces deliberately DO NOT mirror these numpy-handle members:
#
#   member                  where       why it stays single-surface
#   ----------------------  ----------  ------------------------------------------
#   fk_batched              numpy only  G2 batched FK helper for host-side IK
#                                       seeding; rc=3 on floating/mimic. Framework
#                                       users differentiate ee_pose instead.
#   plant_step_hessian      numpy only  DDP consumers (GATO/PDDP) drive it from
#                                       host pipelines; no vjp story on jax/torch.
#   *_wrt_params            jax+torch   regressor-basis parameter gradients exist
#                                       to be DIFFERENTIATED (custom_vjp/autograd
#                                       sysID ops); numpy exposes the regressor
#                                       itself (inverse_dynamics_regressor).
#   capture()               torch only  CUDA-graph capture is a torch-runtime
#                                       feature (stream capture of the op chain).
#   vjp/autograd wiring     jax/torch   custom_vjp / autograd.Function only mean
#                                       something on a framework surface.

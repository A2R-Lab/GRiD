"""RobotHandle — Python-facing wrapper around a compiled per-robot .so.

The handle is what users actually interact with after register_robot()
returns. It bridges the pybind11 Runner (which dlopens the .so and calls
its C ABI) to numpy/Python conventions, and adds shape validation +
helpful error messages.

All algorithm methods take and return 2D arrays where axis 0 is the
batch dimension. Per the wrapper plan, single-call semantics are not
exposed — batch=1 covers it with negligible overhead.

Gravity convention
------------------
`gravity` is the **signed** gravitational acceleration along world +z, default
``-9.81`` (standard downward gravity) — the same convention as pinocchio and
``RBDReference`` (``GRAVITY=-9.81``). Pass the same value to both for matching
results; the default already matches.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


# ─── structured second-order return types (shared across numpy/jax/torch) ─────
#
# idsva_so / fdsva_so return four rank-3 tensors each shape (B, NV, NV, NV).
# A NamedTuple gives them names while staying a plain tuple — positional
# unpacking (`a, b, c, d = h.idsva_so(...)`) and indexing still work, so this
# is backward-compatible. The component names match RBDReference's
# ``idsva_so_*`` return order (d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq).


class SecondOrderID(NamedTuple):
    """Second-order inverse-dynamics tensors, each shape ``(B, NV, NV, NV)``.

    Matches ``RBDReference.idsva_so_body_frame`` / ``idsva_so_world_frame``:
    ``(d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq)``.
    """
    d2tau_dq: Any
    d2tau_dqd: Any
    d2tau_cross: Any
    dM_dq: Any


class SecondOrderFD(NamedTuple):
    """Second-order forward-dynamics tensors, each shape ``(B, NV, NV, NV)``:
    ``(d2qdd_dq, d2qdd_dqd, d2qdd_cross, d2qdd_du)`` (the FD analogue of
    :class:`SecondOrderID`; consult ``RBDReference.fdsva_so`` for the exact
    Singh/Wensing tensor semantics)."""
    d2qdd_dq: Any
    d2qdd_dqd: Any
    d2qdd_cross: Any
    d2qdd_du: Any


# Integrator-type name -> the int code the C ABI dispatches onto IntegratorType.
_INTEGRATOR_CODES = {
    "euler": 0,
    "semi_implicit_euler": 1,
    "si_euler": 1,
    "midpoint": 2,
    "rk3": 3,
    "rk4": 4,
}


def _integrator_code(integrator_type: str) -> int:
    try:
        return _INTEGRATOR_CODES[integrator_type.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"unknown integrator_type {integrator_type!r}; expected one of "
            f"{sorted(set(_INTEGRATOR_CODES))}"
        )


# Pinocchio reference-frame ordering (matches RBDReference / the CUDA enum):
# LOCAL=0, WORLD=1, LOCAL_WORLD_ALIGNED=2.
_REFERENCE_FRAME_CODES = {"local": 0, "world": 1, "local_world_aligned": 2}


def _frame_args(target_jid, reference_frame):
    """Normalize the frame_jacobian[_dot] runtime frame kwargs to the C ABI's
    (int target_jid, int reference_frame), where -1 means "use the codegen
    leaf-EE / LWA default baked into the host wrapper". ``reference_frame`` may
    be an int (0/1/2) or one of LOCAL / WORLD / LOCAL_WORLD_ALIGNED."""
    tj = -1 if target_jid is None else int(target_jid)
    if reference_frame is None:
        rf = -1
    elif isinstance(reference_frame, str):
        key = reference_frame.lower()
        if key not in _REFERENCE_FRAME_CODES:
            raise ValueError(
                f"unknown reference_frame {reference_frame!r}; expected one of "
                "LOCAL / WORLD / LOCAL_WORLD_ALIGNED (or 0/1/2)"
            )
        rf = _REFERENCE_FRAME_CODES[key]
    else:
        rf = int(reference_frame)
    return tj, rf


class _MujocoView:
    """MuJoCo-native view over a :class:`RobotHandle` (``handle.mujoco``).

    Exposes only the convention-supported VALUE methods, with MuJoCo parameter
    names (``qpos``/``qvel``/``qacc``/``qfrc``) and the mjx output convention applied
    PER CALL — it forwards an explicit ``_convention="mujoco"`` rather than mutating
    the handle's shared ``output_convention``, so it is thread-safe and safe to use
    concurrently with pinocchio-convention calls on the same handle. On a fixed base
    the convention is a no-op (no free-flyer), so the view simply matches pinocchio.

    Derivative / second-order / other convention-sensitive surfaces are deliberately
    NOT exposed here while their mjx codegen fusion is in progress (calling them in
    mujoco mode raises); they appear on this view as each one lands."""

    __slots__ = ("_h",)

    def __init__(self, handle: "RobotHandle") -> None:
        self._h = handle

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
        """Inverse mass matrix Minv(qpos) in the mjx frame (G Minv G^T)."""
        return self._h.minv(qpos, _convention="mujoco")

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

    def __repr__(self) -> str:
        return f"<mujoco view of {self._h!r}>"


class RobotHandle:
    """Opaque handle to a compiled per-robot GRiD library.

    Created by `grid_rbd.register_robot(...)` and `grid_rbd.get_robot(...)`.
    Don't construct directly; the constructor wires up the pybind11 Runner
    plus the metadata loaded from the cache's meta.json.

    Precision: float32 by default — methods cast inputs to ``float32`` and
    compute in single precision. A **true fp64 tier** (Phase 8) is available by
    registering with ``dtype="float64"``: the handle then drives a double-
    precision .so (``_core.RunnerF64``), casts inputs to ``float64``, and returns
    ``float64`` arrays computed end-to-end in double precision (``handle.dtype ==
    "float64"``). The legacy ``allow_fp64=True`` is only an fp32-compute upcast
    convenience (compute in fp32, cast i/o to fp64, single-precision accuracy
    caveat); it is off by default and ignored for a true-fp64 handle. The
    jax/torch handles are strictly fp32.

    Method index (all take/return ``(B, …)`` arrays, batch axis first)::

        dynamics    inverse_dynamics (rnea) · forward_dynamics (fd) · aba ·
                    crba · minv · generalized_gravity · nonlinear_effects
        gradients   inverse_dynamics_gradient · forward_dynamics_gradient
        2nd-order   idsva_so → SecondOrderID · fdsva_so → SecondOrderFD
        kinematics  end_effector_pose[_gradient|_hessian] · fk_batched ·
                    frame_jacobian[_dot] · com · ccrba · osc_inertia ·
                    dccrba · cmm_time_variation
        energy      energy · coriolis_matrix ·
                    kinetic_energy_regressor · potential_energy_regressor
        integration integrator[_gradient]
        plant/cost  plant_step[_gradient|_hessian] · quadratic_state_cost ·
                    quadratic_input_cost · ee_pos_cost · com_cost ·
                    momentum_cost · joint_{position,velocity,torque}_barrier

    Short aliases: ``rnea`` → :py:meth:`inverse_dynamics`,
    ``fd`` → :py:meth:`forward_dynamics` (``aba`` / ``crba`` / ``minv`` already
    use their field-standard names).
    """

    def __init__(self, name: str, so_path: str, meta: dict[str, Any],
                 *, allow_fp64: bool = False) -> None:
        from . import _core  # pybind11 extension; built at pip install time

        self._name = name
        self._meta = dict(meta)
        # fp64 (Phase 8): a .so built with dtype="float64" has a double-precision
        # C ABI; it must be driven through RunnerF64 (which declares its buffers
        # + fn-pointers as double) and fed/returned float64 numpy arrays. fp32
        # (default / pre-Phase-8 meta lacking a dtype field) uses Runner. _dt is
        # the host-side numpy element dtype every method casts inputs to.
        self._dtype = str(meta.get("dtype", "float32"))
        if self._dtype == "float64":
            self._dt = np.float64
            self._runner = _core.RunnerF64(so_path)
        else:
            self._dt = np.float32
            self._runner = _core.Runner(so_path)
        # fp64-in / fp64-out convenience (compute stays fp32). Off by default.
        # Ignored for a true-fp64 build (outputs are already float64).
        self.allow_fp64 = bool(allow_fp64) and self._dtype != "float64"
        # Output convention: "pinocchio" (default, native) or "mujoco" (mjx parity).
        # Only affects FLOATING-base robots; a no-op (byte-identical) on fixed base.
        self._output_convention = "pinocchio"

        # Sanity-check that the .so's reported constants match meta.json.
        # A mismatch implies the cache is corrupted.
        for key, runner_val in [
            ("num_joints", self._runner.num_joints),
            ("num_vel", self._runner.num_vel),
            ("num_ees", self._runner.num_ees),
        ]:
            cached_val = meta.get(key)
            if cached_val is not None and cached_val != runner_val:
                raise RuntimeError(
                    f"Cache inconsistency: meta.json says {key}={cached_val} "
                    f"but the .so reports {runner_val}. Re-register with "
                    f"force_rebuild=True."
                )

    # ─── metadata ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_joints(self) -> int:
        return self._runner.num_joints

    @property
    def num_vel(self) -> int:
        return self._runner.num_vel

    @property
    def num_ees(self) -> int:
        return self._runner.num_ees

    @property
    def dtype(self) -> str:
        """Compute precision of this handle's .so: ``"float32"`` (default) or
        ``"float64"`` (Phase 8 true fp64 tier). Inputs are cast to / outputs are
        returned in this numpy dtype."""
        return self._dtype

    @property
    def floating_base(self) -> bool:
        return bool(self._meta.get("floating_base", False))

    @property
    def output_convention(self) -> str:
        """Output/IO convention: ``"pinocchio"`` (default, GRiD-native — xyzw quat,
        spatial-local free-joint velocity) or ``"mujoco"`` (mjx parity — wxyz quat,
        global-linear free-joint velocity). Setting ``"mujoco"`` makes the value
        methods take and return MuJoCo-convention ``q``/``qd``/``qdd``/``M``/... for
        a FLOATING base; it is a byte-identical no-op for a fixed base. See
        ``RBDReference/equivalents/mujoco_convention.md`` for the exact transforms.
        Currently applies to the VALUE methods (inverse_dynamics, forward_dynamics,
        aba, crba, minv); gradient/second-order surfaces stay pinocchio-convention."""
        return self._output_convention

    @output_convention.setter
    def output_convention(self, value: str) -> None:
        if value not in ("pinocchio", "mujoco"):
            raise ValueError(
                f"output_convention must be 'pinocchio' or 'mujoco', got {value!r}")
        self._output_convention = value

    @property
    def mujoco(self) -> "_MujocoView":
        """MuJoCo-native view (``handle.mujoco.inverse_dynamics(qpos, qvel, qacc)``):
        the supported value methods with MuJoCo parameter names and the mjx output
        convention applied PER CALL — thread-safe and independent of the handle's
        ``output_convention`` default (it never mutates shared state). Cached."""
        view = getattr(self, "_mujoco_view", None)
        if view is None:
            view = _MujocoView(self)
            self._mujoco_view = view
        return view

    def _resolve_convention(self, convention=None) -> str:
        """Per-call output convention: an explicit ``convention`` override (used by
        the thread-safe :py:attr:`mujoco` view) wins over the handle's mutable
        ``output_convention`` default. Validated."""
        conv = self._output_convention if convention is None else convention
        if conv not in ("pinocchio", "mujoco"):
            raise ValueError(f"convention must be 'pinocchio' or 'mujoco', got {conv!r}")
        return conv

    def _mjx_active(self, convention=None) -> bool:
        """True when mjx-convention transforms should actually run (mujoco AND a
        floating base — fixed base has no free-flyer, so the flag is a no-op).
        ``convention`` is an optional per-call override (thread-safe; the
        :py:attr:`mujoco` view passes it instead of mutating shared state)."""
        return self._resolve_convention(convention) == "mujoco" and self.floating_base

    def _mjx_guard_unsupported(self, method: str) -> None:
        """Interim scaffold (removed as the codegen mjx fusion lands per algorithm):
        RAISE on a floating-base derivative/second-order method while
        ``output_convention="mujoco"`` rather than silently returning a PIN-frame
        result. The value methods (id/fd/aba/crba/minv) already transform correctly;
        the gradient / Hessian / second-order surfaces do NOT yet, so fail loudly.
        The mjx transforms for these are validated in
        ``RBDReference/equivalents/mujoco_convention.py`` and are being baked into the
        kernels (see docs/open-tasks/mjx_codegen_fusion_master_plan.md)."""
        if self._mjx_active():
            raise NotImplementedError(
                f"{method}() does not yet support output_convention='mujoco' on a "
                "floating base (it would silently return a pinocchio-frame result). "
                "Use output_convention='pinocchio' and transform with "
                "RBDReference.equivalents.mujoco_convention, or wait for the mjx "
                "codegen fusion. The value methods (inverse_dynamics, forward_dynamics, "
                "crba, minv, aba) DO support mujoco mode.")

    # ─── runtime-mutable inertia (D.4 / Phase 5) ─────────────────────────────

    @property
    def runtime_inertia(self) -> bool:
        """True if this robot was registered with ``runtime_inertia=True`` (the
        .so carries a mutable inertia table + :py:meth:`set_inertia_params`)."""
        return bool(self._meta.get("runtime_inertia", False))

    @property
    def inertia_params(self):
        """The BAKED 10-param-per-body inertia table, shape ``(num_bodies, 10)``.

        Each row is ``[m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]`` (mass,
        first moment ``h = m*c``, then the 6 upper-triangle entries of the
        link's inertia about its frame origin) in the frozen GRiD/URDF regressor
        basis — body-indexed, bodies 1..N (the synthetic world-frame link is
        dropped, mirroring the device table layout). For a FIXED base
        ``num_bodies == num_joints``; for a FLOATING base the floating trunk is
        row 0 and ``num_bodies < num_joints (== num_pos)``. Fetch this, mutate
        it, and pass it to
        :py:meth:`set_inertia_params`. Only available on a ``runtime_inertia``
        build (raises otherwise; the values aren't persisted for a baked .so).
        """
        params = self._meta.get("inertia_params")
        if params is None:
            raise RuntimeError(
                "inertia_params is only available on a robot registered with "
                "runtime_inertia=True. Re-register with "
                "register_robot(..., runtime_inertia=True, force_rebuild=True).")
        return np.asarray(params, dtype=self._dt)

    def set_inertia_params(self, params) -> None:
        """Update the device-resident inertia table at runtime (no recompile).

        ``params`` is the 10-param-per-body table — either flat
        ``(10*num_bodies,)`` or ``(num_bodies, 10)`` — in the same layout /
        basis as :py:attr:`inertia_params` (bodies 1..N, each ``[m, h(3),
        I_O(6)]``). All subsequent algorithm calls (inverse_dynamics, crba, …)
        reconstruct the per-link spatial inertia from the updated table. The
        sysID / domain-randomization / payload entry point.

        The table is body-indexed by ``num_bodies`` (the inertia-body count, ==
        :py:attr:`inertia_params` rows), NOT ``num_joints``: for a FIXED base
        they coincide, but for a FLOATING base (or a mimic robot)
        ``num_joints == num_pos > num_bodies`` and the device table is
        ``10*num_bodies`` long.

        Only valid on a robot registered with ``runtime_inertia=True``; raises a
        clear error otherwise. Passing the baked :py:attr:`inertia_params` back
        reproduces the baked result.
        """
        if not self.runtime_inertia:
            raise RuntimeError(
                "set_inertia_params requires a robot registered with "
                "runtime_inertia=True. Re-register with "
                "register_robot(..., runtime_inertia=True, force_rebuild=True).")
        nb = self.num_bodies
        arr = np.ascontiguousarray(params, dtype=self._dt)
        if arr.shape == (nb, 10):
            arr = arr.reshape(-1)
        elif arr.shape != (10 * nb,):
            raise ValueError(
                f"params must be ({nb}, 10) or ({10 * nb},) = 10*num_bodies "
                f"(bodies 1..N, [m, h(3), I_O(6)] each); got shape {arr.shape}.")
        arr = np.ascontiguousarray(arr, dtype=self._dt)
        self._runner.set_inertia_params(arr)

    @property
    def max_batch(self) -> int:
        return self._runner.max_batch

    @property
    def max_perf_level_threads(self) -> int:
        """Codegen-time thread-count hint (DOF-aware, warp-rounded).

        The default block size for kernel launches. Since v2.0 it is a
        recommendation, not an enforced floor — callers can override
        via :py:meth:`set_threads_per_block`.
        """
        return self._runner.max_perf_level_threads

    @property
    def threads_per_block(self) -> int:
        """Current per-block thread count used by kernel launches."""
        return self._runner.threads_per_block

    def set_threads_per_block(self, n: int) -> None:
        """Override the per-block thread count for all subsequent kernel
        launches issued through this handle.

        The codegen does block-cooperative compute: each block handles one
        timestep with its threads cooperating via block-stride loops.
        Batching across timesteps is grid-stride at the block level. Any
        block size ``n >= 1`` (up to the per-block max, 1024 on current
        GPUs) is valid; smaller sizes are correct but slower.

        Default: :py:attr:`max_perf_level_threads`.
        """
        self._runner.set_threads_per_block(int(n))

    # ─── algorithms ──────────────────────────────────────────────────────────
    #
    # All methods take 2D float32 arrays of shape (B, num_joints) for the q/qd/qdd
    # /u inputs (GRiD's kernels consume q, qd, qdd and u all at the NUM_JOINTS
    # (== num_pos == nq) stride — for a floating base the 6-dof base velocity
    # occupies the first slots and the +1 quaternion offset is a padded slot).
    # VALUE vector outputs (torque c, qdd) are likewise (B, num_joints).
    # MATRIX / Jacobian outputs are tangent-space (pinocchio convention) and are
    # nv-dimensioned: crba/minv -> (B, num_vel, num_vel); the dynamics gradients
    # -> (B, num_vel, 2*num_vel). For a FIXED base num_vel == num_joints so all
    # shapes coincide; for a FLOATING base num_vel < num_joints.

    @property
    def num_bodies(self) -> int:
        """Number of bodies/links (incl. the base for floating-base). The
        external-force array ``f_ext`` is shaped ``(B, 6*num_bodies)``."""
        return self._runner.num_bodies

    def _prep_f_ext(self, f_ext):
        """Validate + coerce the optional external-force argument.

        ``f_ext`` is ``(B, 6*num_bodies)`` float32, body-major, each per-body
        wrench ordered ``[angular(3); linear(3)]`` in that link's LOCAL frame.
        This matches ``RBDReference.apply_external_forces`` (which subtracts the
        local wrench from the per-body force, ``f[:, i] -= f_ext[i]``) and the
        GATO/CUDA ``f -= f_ext`` convention. Returns None (no-op) if f_ext is
        None, keeping the no-f_ext path identical to before.
        """
        if f_ext is None:
            return None
        fe = np.ascontiguousarray(f_ext, dtype=self._dt)
        nb = self.num_bodies
        if fe.ndim != 2 or fe.shape[1] != 6 * nb:
            raise ValueError(
                f"f_ext must be (batch, 6*num_bodies) = (batch, {6 * nb}); "
                f"got shape {fe.shape}. Layout is body-major, each body a "
                f"length-6 [angular; linear] wrench in the body's local frame."
            )
        return fe

    def _cast_out(self, *arrays):
        """fp64-out convenience: upcast results to float64 when ``allow_fp64``
        is set (compute already ran in fp32; this is a pure host-side cast with
        the obvious precision caveat). A no-op otherwise. Returns a single array
        for one input, else a tuple — mirroring the wrapped method's return."""
        if not self.allow_fp64:
            return arrays[0] if len(arrays) == 1 else arrays
        out = tuple(np.asarray(a, dtype=np.float64) for a in arrays)
        return out[0] if len(out) == 1 else out

    # ─── mjx (MuJoCo) output-convention transforms (floating base only) ──────
    #
    # When ``output_convention="mujoco"`` the value methods accept and return
    # MuJoCo-convention quantities. Inputs are converted mjx->pin before the
    # kernel, outputs pin->mjx after. The transforms touch only the free-flyer
    # block (quat reorder + the G=blockdiag(R,I) root basis change + the omega x v
    # acceleration term); internal joints are untouched. See `_mujoco.py` and
    # `RBDReference/equivalents/mujoco_convention.md`. Velocity-space inputs are
    # nq-wide (tangent in the first NV slots), so the slice-based transforms apply
    # unchanged. Done in float64 then cast back to the handle dtype.

    def _mjx_inputs(self, q, qd=None, qdd=None, u=None):
        from . import _mujoco
        q_pin = _mujoco.q_mjx_to_pin(np.asarray(q, dtype=np.float64), True)
        R = _mujoco.base_rotation(q_pin)
        qd_pin = None if qd is None else _mujoco.v_mjx_to_pin(np.asarray(qd, np.float64), R, True)
        qdd_pin = None if qdd is None else _mujoco.accel_mjx_to_pin(np.asarray(qdd, np.float64), qd_pin, R, True)
        u_pin = None if u is None else _mujoco.force_mjx_to_pin(np.asarray(u, np.float64), R, True)
        return q_pin, qd_pin, qdd_pin, u_pin, R

    def inverse_dynamics(self, q, qd, qdd=None, *, gravity: float = -9.81, f_ext=None, _convention=None):
        """Inverse dynamics (RNEA): τ = M(q)·qdd + h(q,qd) − g(q). Returns ``(B, NJ)``.

        With ``qdd=None`` (default) this is the **bias** c = h(q,qd) − g(q)
        (= ``RBDReference.inverse_dynamics(q, qd, qdd=0)``). Pass a nonzero
        ``qdd`` to get the full RNEA torque including the inertial term M·qdd
        — the acceleration is now plumbed through (USE_QDD_FLAG=true).

        ``f_ext`` (optional): per-body external forces, shape
        ``(B, 6*num_bodies)``, body-major, each ``[angular; linear]`` in the
        body's local frame (subtracted from the per-body force, matching
        ``RBDReference.inverse_dynamics(..., f_ext=...)``). Default None ⇒ no external force.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd``/``qdd`` are
        MuJoCo-convention and the returned ``τ`` is in the mjx frame.
        """
        # mjx (MuJoCo output convention): prefer the NATIVE mjx kernel when available.
        # It bakes the convention transform into the kernel (raw mjx inputs in, mjx
        # tau out) — no host-side input convert or tau rotation. Requires an explicit
        # qdd (the qdd=0 bias path is nonlinear_effects) and currently no f_ext (the
        # kernel input-convert doesn't reframe external wrenches yet); otherwise fall
        # back to the validated pin-kernel + _mujoco.py post-process path below.
        if (self._mjx_active(_convention) and qdd is not None and f_ext is None
                and getattr(self._runner, "has_inverse_dynamics_mujoco", False)):
            q   = np.ascontiguousarray(q,   dtype=self._dt)
            qd  = np.ascontiguousarray(qd,  dtype=self._dt)
            qdd_arr = np.ascontiguousarray(qdd, dtype=self._dt)
            c = self._runner.inverse_dynamics_mujoco(q, qd, qdd_arr, gravity, None)
            return self._cast_out(c)

        R = None
        if self._mjx_active(_convention):
            q, qd, qdd, _, R = self._mjx_inputs(q, qd, qdd)
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        qdd_arr = None
        if qdd is not None:
            qdd_arr = np.ascontiguousarray(qdd, dtype=self._dt)
        c = self._runner.inverse_dynamics(q, qd, qdd_arr, gravity, self._prep_f_ext(f_ext))
        if R is not None:
            from . import _mujoco
            c = _mujoco.id_tau_pin_to_mjx(np.asarray(c, np.float64), R, True).astype(self._dt)
        return self._cast_out(c)

    def minv(self, q, *, _convention=None):
        """Direct mass-matrix inverse Minv(q). Returns shape (B, NV, NV).

        Minv is the tangent-space (pinocchio-convention) inverse mass matrix:
        ``NV x NV``. For a FIXED base ``NV == NJ`` (== num_pos) so the shape is
        unchanged; for a FLOATING base ``NV = 6 + n_joints < NJ = 7 + n_joints``
        (the +1 is the quaternion offset in q only). GRiD's `minv` kernel writes
        only the upper triangle (lower zero); we symmetrize on the host before
        returning so the matrix matches `RBDReference.minv(..., output_dense=True)`.
        The symmetrization is a single numpy op per call — negligible cost.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned ``Minv`` is the mjx-frame inverse mass matrix.
        """
        # mjx: prefer the native kernel (raw mjx q in, full DENSE SYMMETRIC mjx Minv
        # out — the G^-T Minv G^-1 congruence is baked into the kernel). This path
        # SKIPS both the host symmetrize and the minv_pin_to_mjx post-process.
        if self._mjx_active(_convention) and getattr(self._runner, "has_minv_mujoco", False):
            q = np.ascontiguousarray(q, dtype=self._dt)
            return self._cast_out(self._runner.minv_mujoco(q))

        R = None
        if self._mjx_active(_convention):
            q, _, _, _, R = self._mjx_inputs(q)
        q = np.ascontiguousarray(q, dtype=self._dt)
        m = self._runner.minv(q)
        # Symmetrize: M = L + L^T - diag(L)  where L is the lower triangle.
        m_full = m + m.swapaxes(-1, -2)
        diag_idx = np.arange(m.shape[-1])
        m_full[:, diag_idx, diag_idx] -= np.diagonal(m, axis1=-2, axis2=-1)
        if R is not None:
            from . import _mujoco
            m_full = _mujoco.minv_pin_to_mjx(np.asarray(m_full, np.float64), R, True).astype(self._dt)
        return self._cast_out(m_full)

    def forward_dynamics(self, q, qd, u, *, gravity: float = -9.81, f_ext=None, _convention=None):
        """Forward dynamics qdd = M⁻¹·(τ − c). Returns shape (B, NJ).

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``,
        body-major, ``[angular; linear]`` local-frame (see :py:meth:`inverse_dynamics`).
        With ``output_convention="mujoco"`` (floating base) ``q``/``qd``/``u`` are
        MuJoCo-convention and the returned ``qdd`` is in the mjx frame."""
        # mjx: prefer the native kernel (raw mjx in, mjx qdd out — accel_out baked in);
        # fall back to the validated host path. f_ext isn't reframed by the kernel.
        if (self._mjx_active(_convention) and f_ext is None
                and getattr(self._runner, "has_forward_dynamics_mujoco", False)):
            q  = np.ascontiguousarray(q,  dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            u  = np.ascontiguousarray(u,  dtype=self._dt)
            return self._cast_out(self._runner.forward_dynamics_mujoco(q, qd, u, gravity, None))

        R = None; qd_pin = None
        if self._mjx_active(_convention):
            q, qd_pin, _, u, R = self._mjx_inputs(q, qd, u=u); qd = qd_pin
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        u  = np.ascontiguousarray(u,  dtype=self._dt)
        acc = self._runner.forward_dynamics(q, qd, u, gravity, self._prep_f_ext(f_ext))
        if R is not None:
            from . import _mujoco
            acc = _mujoco.fd_qdd_pin_to_mjx(np.asarray(acc, np.float64), qd_pin, R, True).astype(self._dt)
        return self._cast_out(acc)

    def aba(self, q, qd, u, *, gravity: float = -9.81, f_ext=None, _convention=None):
        """Recursive forward dynamics via Articulated Body Algorithm.
        Returns shape (B, NJ). Alternative to forward_dynamics() with the
        same output but a different implementation.

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``,
        body-major, ``[angular; linear]`` local-frame (see :py:meth:`inverse_dynamics`).
        With ``output_convention="mujoco"`` (floating base) the IO is mjx-convention."""
        if (self._mjx_active(_convention) and f_ext is None
                and getattr(self._runner, "has_aba_mujoco", False)):
            q  = np.ascontiguousarray(q,  dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            u  = np.ascontiguousarray(u,  dtype=self._dt)
            return self._cast_out(self._runner.aba_mujoco(q, qd, u, gravity, None))

        R = None; qd_pin = None
        if self._mjx_active(_convention):
            q, qd_pin, _, u, R = self._mjx_inputs(q, qd, u=u); qd = qd_pin
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        u  = np.ascontiguousarray(u,  dtype=self._dt)
        acc = self._runner.aba(q, qd, u, gravity, self._prep_f_ext(f_ext))
        if R is not None:
            from . import _mujoco
            acc = _mujoco.fd_qdd_pin_to_mjx(np.asarray(acc, np.float64), qd_pin, R, True).astype(self._dt)
        return self._cast_out(acc)

    def crba(self, q, *, gravity: float = -9.81, _convention=None):
        """Joint-space mass matrix M(q) via Composite Rigid Body Algorithm.
        Returns shape (B, NV, NV) — the tangent-space (pinocchio-convention)
        mass matrix. FIXED base: NV == NJ (unchanged); FLOATING base: NV < NJ
        (the kernel writes NUM_VEL x NUM_VEL). Pass `gravity` only because the
        host wrapper takes it; the result doesn't depend on gravity.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned ``M`` is the mjx-frame mass matrix."""
        # mjx: prefer the native kernel (raw mjx q in, mjx M out — the G M G^T
        # congruence is baked into the kernel); fall back to the validated host path.
        if (self._mjx_active(_convention)
                and getattr(self._runner, "has_crba_mujoco", False)):
            q = np.ascontiguousarray(q, dtype=self._dt)
            return self._cast_out(self._runner.crba_mujoco(q, gravity))

        R = None
        if self._mjx_active(_convention):
            q, _, _, _, R = self._mjx_inputs(q)
        q = np.ascontiguousarray(q, dtype=self._dt)
        M = self._runner.crba(q, gravity)
        if R is not None:
            from . import _mujoco
            M = _mujoco.mass_matrix_pin_to_mjx(np.asarray(M, np.float64), R, True).astype(self._dt)
        return self._cast_out(M)

    def end_effector_pose(self, q, *, _convention=None):
        """End-effector pose [xyz, rpy] per EE. Returns shape (B, 6*NUM_EES).
        For multi-EE robots, reshape to (B, NUM_EES, 6) at the caller side.

        With ``output_convention="mujoco"`` (floating base) ``q`` is
        MuJoCo-convention; the pose itself is frame-INVARIANT, but the native
        kernel routes ``q`` through the mjx quaternion reorder (so the output
        equals feeding the pin kernel the pin-converted ``q``)."""
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_end_effector_pose_mujoco", False):
                raise NotImplementedError(
                    "end_effector_pose(output_convention='mujoco') needs a "
                    "floating-base .so built with the mjx kernel — re-register with "
                    "force_rebuild=True.")
            q = np.ascontiguousarray(q, dtype=self._dt)
            return self._runner.end_effector_pose_mujoco(q)
        q = np.ascontiguousarray(q, dtype=self._dt)
        return self._runner.end_effector_pose(q)

    def fk_batched(self, q, *, use_warp: bool = False):
        """Large-batch forward kinematics, one block (thread variant) or warp
        (warp variant) per sample.

        Input  q:     (B, NUM_POS)  joint positions (batch-major).
        Output pose7: (B, 7) = [tx, ty, tz, qw, qx, qy, qz] for the leaf EE
        frame, where the last four are the unit quaternion (w, x, y, z).

        `use_warp=True` runs the warp-cooperative per-sample inner; both
        variants return identical poses. Only available for fixed-base,
        non-mimic robots (raises otherwise)."""
        q = np.ascontiguousarray(q, dtype=self._dt)
        return self._runner.fk_batched(q, use_warp)

    def end_effector_pose_gradient(self, q, *, _convention=None):
        """End-effector pose Jacobian d/dv (TANGENT, pinocchio convention).

        Returns shape (B, 6*NUM_EES, NV). Floating-base produces the
        spatial Jacobian (omega; v) base block, not the older non-standard
        quaternion-derivative columns. Fixed-base shape unchanged (NV == NJ).

        GRiD's `h_end_effector_pose_gradient` is stored column-major as (6, NUM_EES*NV) per
        timestep; we re-orient to (6*NUM_EES, NV) per timestep.

        With ``output_convention="mujoco"`` (floating base) ``q`` is
        MuJoCo-convention and the returned Jacobian has its base-linear columns
        reframed into the mjx frame (computed natively in the kernel).
        """
        NEE = self.num_ees
        NV = self.num_vel
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_end_effector_pose_gradient_mujoco", False):
                raise NotImplementedError(
                    "end_effector_pose_gradient(output_convention='mujoco') needs a "
                    "floating-base .so built with the mjx kernel — re-register with "
                    "force_rebuild=True.")
            q = np.ascontiguousarray(q, dtype=self._dt)
            raw = self._runner.end_effector_pose_gradient_mujoco(q)
            B = raw.shape[0]
            return raw.reshape(B, NEE, NV, 6).transpose(0, 1, 3, 2).reshape(B, 6 * NEE, NV)
        q = np.ascontiguousarray(q, dtype=self._dt)
        raw = self._runner.end_effector_pose_gradient(q)
        B = raw.shape[0]
        return raw.reshape(B, NEE, NV, 6).transpose(0, 1, 3, 2).reshape(B, 6 * NEE, NV)

    def inverse_dynamics_gradient(self, q, qd, qdd=None, *, gravity: float = -9.81, f_ext=None, _convention=None):
        """∂τ/∂(q, qd). Returns shape (B, NV, 2*NV) — concatenated
        [dc_dq | dc_dqd], tangent-space (pinocchio) convention. Slice with
        `[..., :NV]` / `[..., NV:]`. FIXED base: NV == NJ (unchanged); FLOATING
        base: NV < NJ (the kernel writes nv x 2nv).

        ``qdd`` (optional): joint acceleration. The gradient depends on it
        (through the M·qdd term); ``qdd=None`` (default) ⇒ the bias gradient at
        qdd=0. Now plumbed through (USE_QDD_FLAG=true) matching
        :py:meth:`inverse_dynamics`.

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``.
        f_ext enters RNEA affinely, so for a CONSTANT f_ext the Jacobian
        ∂c/∂(q,qd) is unchanged; the kwarg is for consistency with inverse_dynamics().

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd``/``qdd`` are
        MuJoCo-convention and the returned gradient is in the mjx frame (the full
        convention transform — reframe + base-row rotate + ω×v couplings — is baked
        into the kernel). Requires an explicit ``qdd`` (and no ``f_ext``)."""
        NV = self.num_vel
        if (self._mjx_active(_convention) and qdd is not None and f_ext is None
                and getattr(self._runner, "has_inverse_dynamics_gradient_mujoco", False)):
            q   = np.ascontiguousarray(q,   dtype=self._dt)
            qd  = np.ascontiguousarray(qd,  dtype=self._dt)
            qdd_arr = np.ascontiguousarray(qdd, dtype=self._dt)
            raw = self._runner.inverse_dynamics_gradient_mujoco(q, qd, qdd_arr, gravity, None)
            B = raw.shape[0]
            blocks = raw.reshape(B, 2, NV, NV).transpose(0, 1, 3, 2)
            return np.concatenate([blocks[:, 0], blocks[:, 1]], axis=-1)
        if self._mjx_active(_convention):
            raise NotImplementedError(
                "inverse_dynamics_gradient(output_convention='mujoco') needs an explicit "
                "qdd, no f_ext, and a floating-base .so built with the mjx kernel "
                "(re-register with force_rebuild=True).")
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        qdd_arr = None
        if qdd is not None:
            qdd_arr = np.ascontiguousarray(qdd, dtype=self._dt)
        raw = self._runner.inverse_dynamics_gradient(q, qd, qdd_arr, gravity, self._prep_f_ext(f_ext))
        # GRiD's dc_du = [dc_dq (NV×NV col-major), dc_dqd (NV×NV col-major)]
        # per timestep, total 2*NV² floats. Reshape to (B, 2, NV, NV) col-major,
        # transpose each block, hstack to match RBDReference's (NV, 2*NV).
        # FIXED base: NV == NJ (unchanged); FLOATING base: NV < NJ.
        B = raw.shape[0]
        NV = self.num_vel
        blocks = raw.reshape(B, 2, NV, NV).transpose(0, 1, 3, 2)  # row-major now
        return np.concatenate([blocks[:, 0], blocks[:, 1]], axis=-1)

    def forward_dynamics_gradient(self, q, qd, u, *, gravity: float = -9.81, f_ext=None, _convention=None):
        """∂qdd/∂(q, qd). Returns shape (B, NV, 2*NV), tangent-space (pinocchio)
        convention. FIXED base: NV == NJ (unchanged); FLOATING base: NV < NJ.

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``;
        affine in f_ext so a constant f_ext leaves this Jacobian unchanged.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd``/``u`` are
        MuJoCo-convention and the returned gradient is in the mjx frame (the full
        convention transform — reframe + base-row rotate + ω×v couplings — is baked
        into the kernel). Requires no ``f_ext``."""
        NV = self.num_vel
        if (self._mjx_active(_convention) and f_ext is None
                and getattr(self._runner, "has_forward_dynamics_gradient_mujoco", False)):
            q  = np.ascontiguousarray(q,  dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            u  = np.ascontiguousarray(u,  dtype=self._dt)
            raw = self._runner.forward_dynamics_gradient_mujoco(q, qd, u, gravity, None)
            B = raw.shape[0]
            blocks = raw.reshape(B, 2, NV, NV).transpose(0, 1, 3, 2)
            return np.concatenate([blocks[:, 0], blocks[:, 1]], axis=-1)
        if self._mjx_active(_convention):
            raise NotImplementedError(
                "forward_dynamics_gradient(output_convention='mujoco') needs no f_ext "
                "and a floating-base .so built with the mjx kernel "
                "(re-register with force_rebuild=True).")
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        u  = np.ascontiguousarray(u,  dtype=self._dt)
        raw = self._runner.forward_dynamics_gradient(q, qd, u, gravity, self._prep_f_ext(f_ext))
        # Same layout as dc_du: [df_dq, df_dqd] NV×NV col-major blocks.
        B = raw.shape[0]
        blocks = raw.reshape(B, 2, NV, NV).transpose(0, 1, 3, 2)
        return np.concatenate([blocks[:, 0], blocks[:, 1]], axis=-1)

    def end_effector_pose_hessian(self, q):
        """End-effector pose Hessian ∂²(pose)/∂v² (tangent-space, pinocchio convention).
        Returns shape (B, 6*NUM_EES, NV, NV). For fixed-base NV == NJ; for
        floating-base the (NV, NV) block indexes spatial twist components."""
        self._mjx_guard_unsupported("end_effector_pose_hessian")
        q = np.ascontiguousarray(q, dtype=self._dt)
        return self._runner.end_effector_pose_hessian(q)

    def idsva_so(self, q, qd, qdd=None, *, gravity: float = -9.81):
        """Second-order inverse dynamics. Returns a :class:`SecondOrderID`
        NamedTuple ``(d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq)``, each tensor
        shape ``(B, NV, NV, NV)``. (NamedTuple is a plain tuple — positional
        unpacking and indexing still work.)

        Uses the codegen-time dispatcher: body-frame for fixed-base,
        world-frame for floating-base.
        """
        self._mjx_guard_unsupported("idsva_so")
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        # qdd is packed into the device acceleration slot; pass explicit zeros for
        # the default (qdd=None ⇒ zero acceleration) so the result never depends on
        # a stale device buffer from a previous call.
        qdd_in = qdd if qdd is not None else np.zeros_like(q)
        qdd_arr = np.ascontiguousarray(qdd_in, dtype=self._dt)
        NV = self.num_vel
        flat = self._runner.idsva_so(q, qd, qdd_arr, 4 * NV ** 3, gravity)
        # Slice the 4 NV^3 blocks. Each block is stored as raw column/row
        # depending on the kernel; we return them as (B, NV, NV, NV)
        # without further reshape — callers wanting tensor-axis semantics
        # should consult RBDReference's idsva_so docs.
        B = flat.shape[0]
        blocks = [flat[:, i*NV**3:(i+1)*NV**3].reshape(B, NV, NV, NV) for i in range(4)]
        return SecondOrderID(*self._cast_out(*blocks))

    def fdsva_so(self, q, qd, u, *, gravity: float = -9.81):
        """Second-order forward dynamics. Returns a :class:`SecondOrderFD`
        NamedTuple of 4 tensors each shape ``(B, NV, NV, NV)`` (a plain tuple,
        so positional unpacking / indexing still work)."""
        self._mjx_guard_unsupported("fdsva_so")
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        u  = np.ascontiguousarray(u,  dtype=self._dt)
        NV = self.num_vel
        flat = self._runner.fdsva_so(q, qd, u, 4 * NV ** 3, gravity)
        B = flat.shape[0]
        blocks = [flat[:, i*NV**3:(i+1)*NV**3].reshape(B, NV, NV, NV) for i in range(4)]
        return SecondOrderFD(*self._cast_out(*blocks))

    def integrator(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81, _convention=None):
        """One integration step x_{k+1} = integrator(x_k, u, dt).

        Returns shape (B, NUM_POS + NUM_VEL) — concatenated [q_new, v_new].
        `dt` is the runtime timestep; gravity is the signed gravitational acceleration (default -9.81).
        `integrator_type` is one of euler / semi_implicit_euler / midpoint /
        rk3 / rk4.

        With ``output_convention="mujoco"`` (floating base) the free-joint base
        position takes a GLOBAL additive step (the MuJoCo retract) rather than
        pinocchio's SE(3) update; ``q``/``qd`` are MuJoCo-convention and the returned
        ``q_new`` is in the mjx frame (quaternion wxyz). Baked into the kernel."""
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        u  = np.ascontiguousarray(u,  dtype=self._dt)
        it = _integrator_code(integrator_type)
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_integrator_mujoco", False):
                raise NotImplementedError(
                    "integrator(output_convention='mujoco') needs a floating-base .so "
                    "built with the mjx kernel — re-register with force_rebuild=True.")
            return self._runner.integrator_mujoco(q, qd, u, float(dt), it, gravity=float(gravity))
        return self._runner.integrator(q, qd, u, float(dt), it, gravity=float(gravity))

    def integrator_gradient(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """Gradient of the integrator step. Returns shape (B, 2*NV, 3*NV) —
        column blocks [d/dq | d/dqd | d/du] in tangent space.

        `dt` is the runtime timestep; gravity is the signed gravitational acceleration (default -9.81)."""
        self._mjx_guard_unsupported("integrator_gradient")
        q  = np.ascontiguousarray(q,  dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        u  = np.ascontiguousarray(u,  dtype=self._dt)
        it = _integrator_code(integrator_type)
        raw = self._runner.integrator_gradient(q, qd, u, float(dt), it, gravity=float(gravity))
        # h_dAB is (2*NV x 3*NV) column-major per timestep; recover row-major.
        B = raw.shape[0]
        NV = self.num_vel
        return raw.reshape(B, 3 * NV, 2 * NV).transpose(0, 2, 1)

    # ─── grid_plant surface (cost / barrier / plant-step) ────────────────────
    #
    # Composed over the grid:: device surface (integrator + EE pose/Jacobian).
    # Validated against RBDReference._PlantMixin. All take/return 2D arrays with
    # axis 0 = batch. Cost methods return (value, grad, hess); barriers return
    # (value, grad, hess_diag). Conventions mirror RBDReference/_plant.py.

    def quadratic_state_cost(self, x, x_des, Q):
        """1/2 * sum_i Q_i (x_i - x_des_i)^2 over the full state x = [q; qd].

        x / x_des / Q are (B, NUM_POS + NUM_VEL). Returns:
          value (B,), grad (B, NX), hess = diag(Q) (B, NX, NX).
        """
        self._mjx_guard_unsupported("quadratic_state_cost")
        x = np.ascontiguousarray(x, dtype=self._dt)
        x_des = np.ascontiguousarray(x_des, dtype=self._dt)
        Q = np.ascontiguousarray(Q, dtype=self._dt)
        return self._runner.quadratic_state_cost(x, x_des, Q)

    def quadratic_input_cost(self, u, u_des, R):
        """1/2 * sum_i R_i (u_i - u_des_i)^2 over the input u (size NUM_VEL).

        u / u_des / R are (B, NUM_VEL). Returns:
          value (B,), grad (B, NV), hess = diag(R) (B, NV, NV).
        """
        u = np.ascontiguousarray(u, dtype=self._dt)
        u_des = np.ascontiguousarray(u_des, dtype=self._dt)
        R = np.ascontiguousarray(R, dtype=self._dt)
        return self._runner.quadratic_input_cost(u, u_des, R)

    def ee_pos_cost(self, q, p_des, W):
        """End-effector position cost over the 3 position axes (EE 0).

        q is (B, NUM_POS); p_des / W are (B, 3). Returns:
          value (B,), grad_x (B, NX) = [J_p^T (W·r); 0], GN hess_x (B, NX, NX)
          with the top-left NV×NV q-block = J_p^T diag(W) J_p.
        The hessian is returned in the kernel's column-major layout; since the
        GN hessian J_p^T W J_p is symmetric the row/col-major distinction is
        immaterial.
        """
        self._mjx_guard_unsupported("ee_pos_cost")
        q = np.ascontiguousarray(q, dtype=self._dt)
        p_des = np.ascontiguousarray(p_des, dtype=self._dt)
        W = np.ascontiguousarray(W, dtype=self._dt)
        return self._runner.ee_pos_cost(q, p_des, W)

    def joint_position_barrier(self, var, lower, upper, mu):
        """Log-barrier b = -mu·(log(x-lo)+log(hi-x)) over NUM_POS positions.

        var / lower / upper are (B, NUM_POS); an ±inf bound contributes zero.
        Returns (value (B,), grad (B, NUM_POS), hess_diag (B, NUM_POS)).
        """
        return self._barrier("joint_position_barrier", var, lower, upper, mu)

    def joint_velocity_barrier(self, var, lower, upper, mu):
        """Log-barrier over NUM_VEL velocities. See joint_position_barrier."""
        return self._barrier("joint_velocity_barrier", var, lower, upper, mu)

    def joint_torque_barrier(self, var, lower, upper, mu):
        """Log-barrier over NUM_VEL torques. See joint_position_barrier."""
        return self._barrier("joint_torque_barrier", var, lower, upper, mu)

    def _barrier(self, method, var, lower, upper, mu):
        var = np.ascontiguousarray(var, dtype=self._dt)
        lower = np.ascontiguousarray(lower, dtype=self._dt)
        upper = np.ascontiguousarray(upper, dtype=self._dt)
        return getattr(self._runner, method)(var, lower, upper, float(mu))

    def plant_step(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """x_{k+1} = integrator(x_k, u_k, dt). Thin wrapper over grid::integrator.

        x is (B, NUM_POS + NUM_VEL); u is (B, NUM_VEL). Returns (B, NX).
        `integrator_type` is one of euler / semi_implicit_euler / midpoint /
        rk3 / rk4 (same codes as :py:meth:`integrator`).
        """
        self._mjx_guard_unsupported("plant_step")
        x = np.ascontiguousarray(x, dtype=self._dt)
        u = np.ascontiguousarray(u, dtype=self._dt)
        it = _integrator_code(integrator_type)
        return self._runner.plant_step(x, u, float(dt), it, float(gravity))

    def plant_step_gradient(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """[A | B] = d x_{k+1}/d(x,u) = the integrator-gradient s_dAB surface.

        x is (B, NUM_POS + NUM_VEL); u is (B, NUM_VEL). Returns (B, 2*NV, 3*NV)
        with column blocks [d/dq | d/dqd | d/du] in tangent space. Pass-through
        to grid::integrator_gradient (the value is byte-identical to
        :py:meth:`integrator_gradient`). Matches ``RBDReference.plant_step_gradient``
        (= ``integrator_gradient``). ``integrator_type`` is one of euler /
        semi_implicit_euler / midpoint / rk3 / rk4.
        """
        self._mjx_guard_unsupported("plant_step_gradient")
        x = np.ascontiguousarray(x, dtype=self._dt)
        u = np.ascontiguousarray(u, dtype=self._dt)
        it = _integrator_code(integrator_type)
        raw = self._runner.plant_step_gradient(x, u, float(dt), it, float(gravity))
        # raw is filled with the (2*NV x 3*NV) column-major dAB; recover row-major.
        B = raw.shape[0]
        NV = self.num_vel
        return raw.reshape(B, 3 * NV, 2 * NV).transpose(0, 2, 1)

    def plant_step_hessian(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """Second-order sensitivity of the integrator step x_{k+1} = [q; v].

        x is (B, NUM_POS + NUM_VEL); u is (B, NUM_VEL). Returns the s_d2AB
        surface of shape (B, 2*NV, 3*NV, 3*NV):

            H[b, o, a, b'] = d^2 x_{k+1}[o] / dz[a] dz[b'],  z = [dq; dqd; du]

        with output rows split as [position-tangent (NV); velocity (NV)].
        Matches ``RBDReference.plant_step_hessian``. Pass-through to
        grid::integrator_hessian_device (composes fdsva_so + dt-scaled assembly).

        Scope (first landing): euler / semi_implicit_euler on a FIXED base.
        Floating-base and multi-stage RK are deferred (the C-ABI returns rc=3 /
        raises for any other ``integrator_type``).
        """
        self._mjx_guard_unsupported("plant_step_hessian")
        x = np.ascontiguousarray(x, dtype=self._dt)
        u = np.ascontiguousarray(u, dtype=self._dt)
        it = _integrator_code(integrator_type)
        raw = self._runner.plant_step_hessian(x, u, float(dt), it, float(gravity))
        # raw is row-major (2*NV, 3*NV*3*NV) per timestep — reshape the trailing
        # 9*NV^2 into (3*NV, 3*NV) (C-order, no transpose: H is already row-major).
        B = raw.shape[0]
        NV = self.num_vel
        return raw.reshape(B, 2 * NV, 3 * NV, 3 * NV)

    def com_cost(self, q, p_des, W):
        """Center-of-mass tracking cost over the 3 CoM axes.

        q is (B, NUM_POS); p_des / W are (B, 3). Returns:
          value (B,), grad_x (B, NX) = [J_com^T (W·r); 0], GN hess_x (B, NX, NX)
          with the top-left NV×NV q-block = J_com^T diag(W) J_com.
        Matches ``RBDReference.com_cost(q, p_des, W)``.
        """
        self._mjx_guard_unsupported("com_cost")
        q = np.ascontiguousarray(q, dtype=self._dt)
        p_des = np.ascontiguousarray(p_des, dtype=self._dt)
        W = np.ascontiguousarray(W, dtype=self._dt)
        return self._runner.com_cost(q, p_des, W)

    def momentum_cost(self, q, qd, h_des, W):
        """Centroidal-momentum tracking cost over the 6 momentum components.

        q is (B, NUM_POS); qd is (B, NUM_VEL); h_des / W are (B, 6). Returns:
          value (B,), grad_x (B, NX) = [0; A^T (W·r)], GN hess_x (B, NX, NX)
          with the bottom-right NV×NV qd-block = A^T diag(W) A.
        Matches ``RBDReference.momentum_cost(q, qd, h_des, W)``.
        """
        self._mjx_guard_unsupported("momentum_cost")
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        h_des = np.ascontiguousarray(h_des, dtype=self._dt)
        W = np.ascontiguousarray(W, dtype=self._dt)
        return self._runner.momentum_cost(q, qd, h_des, W)

    # ─── centroidal / energy / general-frame kinematics (F2) ─────────────────
    #
    # Convenience compositions over the grid:: kinematics/dynamics surface,
    # validated against RBDReference's centroidal / energy / frame mixins. All
    # take 2D float32 (B, NUM_JOINTS) inputs. The frame_jacobian family targets a
    # frame fixed at codegen time (the leaf end-effector joint,
    # LOCAL_WORLD_ALIGNED reference frame); a runtime frame/reference_frame kwarg
    # is not yet supported on the GPU surface (the host/kernel bake the target).

    def com(self, q, *, _convention=None):
        """Center-of-mass world position p_com (3,) and CoM Jacobian J_com.

        Returns ``(p_com, J_com)`` where ``p_com`` is ``(B, 3)`` and ``J_com``
        is ``(B, 3, NV)`` = ``d(p_com)/dv``. Matches ``RBDReference.com(q)``
        (= ``p_com``) and ``RBDReference.jacobian_com(q)`` (= ``J_com``).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        ``p_com`` is invariant and the ``J_com`` columns are reframed (computed
        natively in the kernel)."""
        NV = self.num_vel
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_com_mujoco", False):
                raise NotImplementedError(
                    "com(output_convention='mujoco') needs a floating-base .so built "
                    "with the mjx kernel — re-register with force_rebuild=True.")
            q = np.ascontiguousarray(q, dtype=self._dt)
            raw = self._runner.com_mujoco(q)
        else:
            q = np.ascontiguousarray(q, dtype=self._dt)
            raw = self._runner.com(q)  # (B, 3 + 3*NV): [p_com(3); J_com(3 x NV col-major)]
        B = raw.shape[0]
        p_com = raw[:, :3]
        # J_com stored column-major (3 x NV): J[r + 3*c]; recover (B, 3, NV).
        j_com = raw[:, 3:].reshape(B, NV, 3).transpose(0, 2, 1)
        return self._cast_out(p_com), self._cast_out(j_com)

    def ccrba(self, q, qd, *, _convention=None):
        """Centroidal momentum matrix A (6 x NV) and momentum h = A·qd (6,).

        Returns ``(A, h)`` where ``A`` is ``(B, 6, NV)`` and ``h`` is ``(B, 6)``,
        in the Pinocchio convention (``[linear; angular]`` at the CoM, world
        aligned). Matches ``RBDReference.ccrba(q, qd)``.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; ``h`` is invariant and the ``A`` columns are reframed
        (computed natively in the kernel)."""
        NV = self.num_vel
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_ccrba_mujoco", False):
                raise NotImplementedError(
                    "ccrba(output_convention='mujoco') needs a floating-base .so built "
                    "with the mjx kernel — re-register with force_rebuild=True.")
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            raw = self._runner.ccrba_mujoco(q, qd)
        else:
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            raw = self._runner.ccrba(q, qd)  # (B, 6*NV + 6): [A(6 x NV col-major); h(6)]
        B = raw.shape[0]
        A = raw[:, : 6 * NV].reshape(B, NV, 6).transpose(0, 2, 1)
        h = raw[:, 6 * NV:]
        return self._cast_out(A), self._cast_out(h)

    def energy(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Kinetic / potential / mechanical energy. Returns ``(B, 3)`` =
        ``[KE, PE, KE+PE]``. Matches ``RBDReference.kinetic_energy`` /
        ``potential_energy`` / ``mechanical_energy`` (PE uses ``gravity``).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; the energies are frame-INVARIANT, so the native kernel
        only converts the inputs and the output equals the pin result."""
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_energy_mujoco", False):
                raise NotImplementedError(
                    "energy(output_convention='mujoco') needs a floating-base .so built "
                    "with the mjx kernel — re-register with force_rebuild=True.")
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            return self._cast_out(self._runner.energy_mujoco(q, qd, float(gravity)))
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        return self._runner.energy(q, qd, float(gravity))

    def generalized_gravity(self, q, *, gravity: float = -9.81, _convention=None):
        """Generalized gravity torque g(q) = RNEA(q, 0, 0). Returns ``(B, NV)``.
        Matches ``RBDReference.generalized_gravity(q, GRAVITY=gravity)``.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned g is in the mjx frame (the base rows are rotated in-kernel).
        Output shape is invariant ``(B, NV)``."""
        if self._mjx_active(_convention) and getattr(self._runner, "has_generalized_gravity_mujoco", False):
            q = np.ascontiguousarray(q, dtype=self._dt)
            return self._runner.generalized_gravity_mujoco(q, float(gravity))
        if self._mjx_active(_convention):
            raise NotImplementedError(
                "generalized_gravity(output_convention='mujoco') needs a floating-base "
                ".so built with the mjx kernel (re-register with force_rebuild=True).")
        q = np.ascontiguousarray(q, dtype=self._dt)
        return self._runner.generalized_gravity(q, float(gravity))

    def nonlinear_effects(self, q, qd, *, gravity: float = -9.81):
        """Nonlinear (bias) effects c(q,qd) = RNEA(q, qd, 0) = C(q,qd)·qd + g(q).
        Returns ``(B, NV)``. Matches ``RBDReference.nonlinear_effects(q, qd,
        GRAVITY=gravity)``."""
        self._mjx_guard_unsupported("nonlinear_effects")
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        return self._runner.nonlinear_effects(q, qd, float(gravity))

    def coriolis_matrix(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Coriolis matrix C(q,qd). Returns ``(B, NV, NV)`` row-major, with
        ``C·qd + g(q) = nonlinear_effects(q, qd)``. Matches
        ``RBDReference.coriolis_matrix(q, qd)`` (gravity is unused by C; the
        kwarg mirrors the host wrapper signature).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and the returned ``C`` is the mjx-frame Coriolis matrix
        (congruence ``G C Gᵀ``, computed natively in the kernel)."""
        NV = self.num_vel
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_coriolis_matrix_mujoco", False):
                raise NotImplementedError(
                    "coriolis_matrix(output_convention='mujoco') needs a floating-base "
                    ".so built with the mjx kernel — re-register with force_rebuild=True.")
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            raw = self._runner.coriolis_matrix_mujoco(q, qd, float(gravity))
            return self._cast_out(raw.reshape(raw.shape[0], NV, NV))
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        raw = self._runner.coriolis_matrix(q, qd, float(gravity))  # (B, NV*NV) row-major
        return self._cast_out(raw.reshape(raw.shape[0], NV, NV))

    def kinetic_energy_regressor(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Kinetic-energy regressor y_KE, length ``10*num_bodies``, with
        ``KE = y_KE · π`` (π = stacked per-link inertial parameters, body-major,
        10 params/body). Returns ``(B, 10*num_bodies)``. Matches
        ``RBDReference.kinetic_energy_regressor(q, qd)`` (gravity unused).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; the regressor is frame-INVARIANT, so the native kernel
        only converts the inputs and the output equals the pin result."""
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_kinetic_energy_regressor_mujoco", False):
                raise NotImplementedError(
                    "kinetic_energy_regressor(output_convention='mujoco') needs a "
                    "floating-base .so built with the mjx kernel — re-register with "
                    "force_rebuild=True.")
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            return self._cast_out(self._runner.kinetic_energy_regressor_mujoco(q, qd, float(gravity)))
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        return self._cast_out(self._runner.kinetic_energy_regressor(q, qd, float(gravity)))

    def potential_energy_regressor(self, q, *, gravity: float = -9.81, _convention=None):
        """Potential-energy regressor y_PE, length ``10*num_bodies``, with
        ``PE = y_PE · π``. Returns ``(B, 10*num_bodies)``. Matches
        ``RBDReference.potential_energy_regressor(q, GRAVITY=gravity)`` (PE uses
        ``gravity``; only the mass + first-moment columns are nonzero).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        the regressor is frame-INVARIANT, so the native kernel only converts the
        input and the output equals the pin result."""
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_potential_energy_regressor_mujoco", False):
                raise NotImplementedError(
                    "potential_energy_regressor(output_convention='mujoco') needs a "
                    "floating-base .so built with the mjx kernel — re-register with "
                    "force_rebuild=True.")
            q = np.ascontiguousarray(q, dtype=self._dt)
            return self._cast_out(self._runner.potential_energy_regressor_mujoco(q, float(gravity)))
        q = np.ascontiguousarray(q, dtype=self._dt)
        return self._cast_out(self._runner.potential_energy_regressor(q, float(gravity)))

    def dccrba(self, q):
        """dCCRBA tensor ∂A/∂q, shape ``(B, 6, NV, NV)`` indexed
        ``[:, :, k, i] = ∂A[:, k]/∂q_i`` (Pinocchio centroidal convention,
        ``[linear; angular]`` at the CoM, world-aligned). Matches
        ``RBDReference.dccrba(q)`` (which returns ``(6, NV, NV)`` per sample).

        Runs on all robots — including mimic and big floating-base (the
        centroidal sweep pool spills to global memory at the spilled tiers). A
        clear ``RuntimeError`` is raised only in the rare case where even the
        most-spilled tier's centroidal pool exceeds this GPU's shared-memory cap.
        """
        self._mjx_guard_unsupported("dccrba")
        q = np.ascontiguousarray(q, dtype=self._dt)
        raw = self._runner.dccrba(q)  # (B, 6*NV*NV) flat, dA[row + 6*k + 6*NV*m]
        B = raw.shape[0]
        NV = self.num_vel
        # flat layout dA[row + 6*k + 6*NV*m] -> (B, m, k, row) then -> (B, row, k, m).
        return self._cast_out(raw.reshape(B, NV, NV, 6).transpose(0, 3, 2, 1))

    def cmm_time_variation(self, q, qd, *, _convention=None):
        """Centroidal-momentum-matrix time variation Ȧ = dA(q(t))/dt, shape
        ``(B, 6, NV)`` (Pinocchio convention, ``[linear; angular]`` at the CoM,
        world-aligned) = ``Σ_i (∂A/∂q_i)·qd_i``. Matches
        ``RBDReference.cmm_time_variation(q, qd)``.

        Runs on all robots (mimic + big floating-base via centroidal-pool spill);
        raises a clear ``RuntimeError`` only on the rare oversized-pool case (see
        :py:meth:`dccrba`).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and the returned Ȧ has its columns reframed into the mjx
        frame (computed natively in the kernel)."""
        NV = self.num_vel
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_cmm_time_variation_mujoco", False):
                raise NotImplementedError(
                    "cmm_time_variation(output_convention='mujoco') needs a "
                    "floating-base .so built with the mjx kernel — re-register with "
                    "force_rebuild=True.")
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            raw = self._runner.cmm_time_variation_mujoco(q, qd)  # (B, 6*NV) col-major A[r + 6*c]
            B = raw.shape[0]
            return self._cast_out(raw.reshape(B, NV, 6).transpose(0, 2, 1))
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        raw = self._runner.cmm_time_variation(q, qd)  # (B, 6*NV) col-major A[r + 6*c]
        B = raw.shape[0]
        return self._cast_out(raw.reshape(B, NV, 6).transpose(0, 2, 1))

    def frame_jacobian(self, q, *, target_jid=None, reference_frame=None, _convention=None):
        """Geometric Jacobian (6 x NV, ``[linear; angular]``) of a frame.
        Returns ``(B, 6, NV)``. Matches ``RBDReference.frame_jacobian(q,
        frame_name, reference_frame)``.

        ``target_jid`` selects the frame's joint id (default: the leaf
        end-effector joint baked at codegen time). ``reference_frame`` is
        ``'LOCAL'`` (0), ``'WORLD'`` (1), or ``'LOCAL_WORLD_ALIGNED'`` (2, the
        default), or the equivalent int. Both are now RUNTIME parameters of the
        GPU surface.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned Jacobian is column-reframed ``J G⁻¹`` (mjx frame)."""
        NV = self.num_vel
        tj, rf = _frame_args(target_jid, reference_frame)
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_frame_jacobian_mujoco", False):
                raise NotImplementedError(
                    "frame_jacobian(output_convention='mujoco') needs a floating-base "
                    ".so built with the mjx kernel — re-register with force_rebuild=True.")
            q = np.ascontiguousarray(q, dtype=self._dt)
            raw = self._runner.frame_jacobian_mujoco(q, tj, rf)
            return raw.reshape(raw.shape[0], NV, 6).transpose(0, 2, 1)
        q = np.ascontiguousarray(q, dtype=self._dt)
        raw = self._runner.frame_jacobian(q, tj, rf)  # (B, 6*NV) col-major: J[r + 6*c]
        return raw.reshape(raw.shape[0], NV, 6).transpose(0, 2, 1)

    def frame_jacobian_dot(self, q, qd, *, target_jid=None, reference_frame=None, _convention=None):
        """Time derivative Jdot of :py:meth:`frame_jacobian` along v = qd
        (6 x NV, ``[linear; angular]``). Returns ``(B, 6, NV)``. Matches
        ``RBDReference.frame_jacobian_dot(q, qd, frame_name, reference_frame)``.

        ``target_jid`` / ``reference_frame`` are RUNTIME parameters (default:
        leaf-EE joint / ``LOCAL_WORLD_ALIGNED``); see :py:meth:`frame_jacobian`.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and Jdot is column-reframed (mjx frame)."""
        NV = self.num_vel
        tj, rf = _frame_args(target_jid, reference_frame)
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_frame_jacobian_dot_mujoco", False):
                raise NotImplementedError(
                    "frame_jacobian_dot(output_convention='mujoco') needs a floating-base "
                    ".so built with the mjx kernel — re-register with force_rebuild=True.")
            q = np.ascontiguousarray(q, dtype=self._dt)
            qd = np.ascontiguousarray(qd, dtype=self._dt)
            raw = self._runner.frame_jacobian_dot_mujoco(q, qd, tj, rf)
            return raw.reshape(raw.shape[0], NV, 6).transpose(0, 2, 1)
        q = np.ascontiguousarray(q, dtype=self._dt)
        qd = np.ascontiguousarray(qd, dtype=self._dt)
        raw = self._runner.frame_jacobian_dot(q, qd, tj, rf)  # (B, 6*NV) col-major
        return raw.reshape(raw.shape[0], NV, 6).transpose(0, 2, 1)

    def osc_inertia(self, q, *, _convention=None):
        """Operational-space (task) inertia Lambda = (J·M⁻¹·Jᵀ)⁻¹ (6 x 6) for
        the leaf-EE frame (LWA). Returns ``(B, 6, 6)``. Matches
        ``RBDReference.osc_inertia(q)``.

        Lambda is frame-INVARIANT, but with ``output_convention="mujoco"`` the
        MuJoCo ``q`` (wxyz quaternion) must be reordered before the kinematics
        build — the mjx kernel does that, so mjx ``q`` routes through it."""
        if self._mjx_active(_convention):
            if not getattr(self._runner, "has_osc_inertia_mujoco", False):
                raise NotImplementedError(
                    "osc_inertia(output_convention='mujoco') needs a floating-base "
                    ".so built with the mjx kernel — re-register with force_rebuild=True.")
            q = np.ascontiguousarray(q, dtype=self._dt)
            raw = self._runner.osc_inertia_mujoco(q)
            return raw.reshape(raw.shape[0], 6, 6)
        q = np.ascontiguousarray(q, dtype=self._dt)
        raw = self._runner.osc_inertia(q)  # (B, 36) row/col-major (symmetric)
        return raw.reshape(raw.shape[0], 6, 6)

    # ─── runtime arbitrary multi-EE pose / pose-gradient (target + offset) ─────

    def _resolve_ee_jids(self, ee_joint_names):
        """Resolve ee_joint_names -> joint ids, mirroring
        RBDReference.select_end_effector_joints: ``None`` => all leaf joints; a
        str / list of names => their joint ids (from the cached joint_names map)."""
        names = self._meta.get("joint_names")
        if ee_joint_names is None:
            jids = self._meta.get("leaf_jids")
            if jids is None:
                raise RuntimeError(
                    "this .so predates the runtime-target API (no leaf_jids in "
                    "meta.json); re-register with force_rebuild=True")
            return list(jids)
        if names is None:
            raise RuntimeError(
                "this .so predates the runtime-target API (no joint_names in "
                "meta.json); re-register with force_rebuild=True")
        if isinstance(ee_joint_names, str):
            ee_joint_names = [ee_joint_names]
        jids = []
        for name in ee_joint_names:
            try:
                jids.append(names.index(name))
            except ValueError:
                raise ValueError(f"Could not find joint named: {name}")
        return jids

    @staticmethod
    def _normalize_ee_offsets(ee_offsets, num_ees):
        """Normalize ee_offsets to a list of length-3 [x,y,z] (one per EE).
        ``None`` => zero offset (frame origin); a single offset is applied to all
        EEs (matches the oracle's first-offset broadcast). Accepts [x,y,z] or
        homogeneous [x,y,z,1]."""
        if ee_offsets is None:
            return [np.zeros(3, dtype=self._dt)] * num_ees
        offs = [np.asarray(o, dtype=self._dt).reshape(-1)[:3] for o in ee_offsets]
        if len(offs) == 1:
            offs = offs * num_ees
        if len(offs) != num_ees:
            raise ValueError(
                f"ee_offsets length {len(offs)} != number of EEs {num_ees}")
        return offs

    def end_effector_pose_runtime(self, q, ee_joint_names=None, ee_offsets=None):
        """Runtime-target end-effector pose ``[xyz; rpy]`` at an offset point.

        Mirrors ``RBDReference.end_effector_pose(q, ee_joint_names, ee_offsets)``:
        ``ee_joint_names`` (None => all leaf joints, or a str / list of joint
        names) selects the EE frames, ``ee_offsets`` (None => frame origin, or one
        ``[x,y,z]`` / ``[x,y,z,1]`` per EE) shifts the measurement point. The
        single-target GPU kernel is looped over the resolved jid list and the
        results stacked.

        Returns ``(B, NUM_EE, 6)`` where each row is ``[x, y, z, roll, pitch, yaw]``.
        """
        q = np.ascontiguousarray(q, dtype=self._dt)
        jids = self._resolve_ee_jids(ee_joint_names)
        offsets = self._normalize_ee_offsets(ee_offsets, len(jids))
        per_ee = []
        for jid, off in zip(jids, offsets):
            raw = self._runner.end_effector_pose_runtime(
                q, int(jid), np.ascontiguousarray(off, dtype=self._dt))  # (B, 6)
            per_ee.append(raw)
        return self._cast_out(np.stack(per_ee, axis=1))  # (B, NUM_EE, 6)

    def end_effector_pose_gradient_runtime(self, q, ee_joint_names=None, ee_offsets=None):
        """Runtime-target end-effector pose gradient ``d[xyz; rpy]/dv`` (6 x NV)
        at an offset point. Same ``ee_joint_names`` / ``ee_offsets`` semantics as
        :py:meth:`end_effector_pose_runtime`; mirrors
        ``RBDReference.end_effector_pose_gradient(q, ee_joint_names, ee_offsets)``.

        Returns ``(B, NUM_EE, 6, NV)``.
        """
        self._mjx_guard_unsupported("end_effector_pose_gradient_runtime")
        q = np.ascontiguousarray(q, dtype=self._dt)
        NV = self.num_vel
        jids = self._resolve_ee_jids(ee_joint_names)
        offsets = self._normalize_ee_offsets(ee_offsets, len(jids))
        per_ee = []
        for jid, off in zip(jids, offsets):
            raw = self._runner.end_effector_pose_gradient_runtime(
                q, int(jid), np.ascontiguousarray(off, dtype=self._dt))  # (B, 6*NV) col-major
            B = raw.shape[0]
            per_ee.append(raw.reshape(B, NV, 6).transpose(0, 2, 1))  # (B, 6, NV)
        return self._cast_out(np.stack(per_ee, axis=1))  # (B, NUM_EE, 6, NV)

    # ─── field-standard short aliases ────────────────────────────────────────
    # `rnea`/`fd` are the names roboticists (pinocchio / frax / bard) reach for;
    # bind them to the long-named methods (aba / crba / minv already match).
    rnea = inverse_dynamics
    fd = forward_dynamics

    # ─── lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the underlying .so handle. After close(), method calls
        will fail. Idempotent."""
        if self._runner is not None:
            del self._runner
            self._runner = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"RobotHandle(name={self._name!r}, num_joints={self.num_joints}, "
            f"num_vel={self.num_vel}, num_ees={self.num_ees}, "
            f"floating_base={self.floating_base})"
        )

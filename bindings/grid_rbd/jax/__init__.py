"""grid_rbd.jax — JAX FFI integration.

Mirrors the standard grid_rbd API but returns a :py:class:`JaxRobotHandle`
whose methods are JAX-callable, `jax.jit`-compatible, and run on
JAX-managed CUDA streams.

Usage:

    import grid_rbd.jax as grid_jax
    import jax

    handle = grid_jax.register_robot(name="iiwa14", urdf_path="iiwa.urdf")

    @jax.jit
    def step(q, qd):
        return handle.inverse_dynamics(q, qd)

The underlying ``.so`` is shared with the plain ``grid_rbd.register_robot``
cache — registering the same name from both APIs uses the same compiled
library and doesn't trigger a recompile.

Surface (parity with the plain ``RobotHandle``):
  ``inverse_dynamics``, ``minv``, ``forward_dynamics``, ``aba``, ``crba``,
  ``generalized_gravity``, ``nonlinear_effects``, ``energy``, ``com``, ``ccrba``,
  ``dccrba``, ``cmm_time_variation``, ``coriolis_matrix``,
  ``kinetic_energy_regressor``, ``potential_energy_regressor``,
  ``frame_jacobian``, ``frame_jacobian_dot``, ``osc_inertia``,
  ``end_effector_pose``, ``end_effector_pose_gradient``,
  ``end_effector_pose_hessian``, ``end_effector_pose_runtime``,
  ``end_effector_pose_gradient_runtime``,
  ``inverse_dynamics_gradient``, ``forward_dynamics_gradient``,
  ``idsva_so``, ``fdsva_so``, plus the grid_plant cost / barrier / plant-step
  surface (``plant_step``, ``plant_step_gradient``, ``quadratic_state_cost``,
  ``quadratic_input_cost``, ``ee_pos_cost``, ``joint_position_barrier``,
  ``joint_velocity_barrier``, ``joint_torque_barrier``, ``com_cost``,
  ``momentum_cost``). All run device-resident on JAX-supplied streams. The
  cost/barrier/step ops that require a gated kernel (plant_step[_gradient],
  ee/com/momentum cost) are only available when the per-robot ``.so`` was built
  with that kernel (raises a clear "symbol missing" error otherwise).
"""
from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import Any

import numpy as np

import grid_rbd as _grid_rbd
from grid_rbd._handle import RobotHandle, SecondOrderID, SecondOrderFD


# ─── handler registry ───────────────────────────────────────────────────────

# Maps (cache_key, method_name) → bool indicating whether the FFI target has
# been registered with JAX. Registration is process-global (JAX maintains the
# target table), so we only need to do it once per (cache_key, method).
_REGISTERED: dict[tuple[str, str], bool] = {}
_LOCK = threading.Lock()


def _require_jax():
    """Import + return the ``jax`` module, or raise a clear, actionable error.

    Routed through the surface entry points (register_robot /
    _register_method_target) so a missing optional dep gives install guidance
    instead of a bare ``ModuleNotFoundError`` from deep inside a method."""
    try:
        import jax
    except ImportError as e:
        raise ImportError(
            "grid_rbd.jax requires JAX, which isn't installed. Install the "
            "optional extra:  pip install -e 'grid-rbd[jax]'  (or "
            "`pip install jax[cuda12]`). The numpy and torch backends do not "
            "need JAX."
        ) from e
    return jax


def _ffi_target_name(cache_key: str, method: str) -> str:
    """JAX FFI target names are global. Key by cache_key (.so identity) so
    different robots → different targets, and so re-registering the same
    robot doesn't collide."""
    return f"grid_rbd_{method}_{cache_key[:12]}"


def _register_method_target(
    so_path: Path,
    cache_key: str,
    method: str,
    symbol: str,
) -> str:
    """Register one FFI target with JAX, returning the target name."""
    jax = _require_jax()
    target_name = _ffi_target_name(cache_key, method)
    with _LOCK:
        if _REGISTERED.get((cache_key, method)):
            return target_name
        # Load the .so and grab the handler symbol as a void*.
        lib = ctypes.CDLL(str(so_path))
        try:
            fn_ptr = getattr(lib, symbol)
        except AttributeError as e:
            raise RuntimeError(
                f"Symbol {symbol!r} missing from {so_path}; was the .so "
                f"compiled with GRID_RBD_WITH_JAX? (Reinstall jax + "
                f"force_rebuild=True at register_robot.)"
            ) from e
        # ctypes function objects are valid void*; wrap as a PyCapsule that
        # JAX accepts. ``jax.ffi.pycapsule`` builds the correct capsule from a
        # raw function pointer.
        capsule = jax.ffi.pycapsule(ctypes.cast(fn_ptr, ctypes.c_void_p).value)
        jax.ffi.register_ffi_target(target_name, capsule, platform="CUDA")
        _REGISTERED[(cache_key, method)] = True
    return target_name


# ─── mjx view ────────────────────────────────────────────────────────────────


class _JaxMujocoView:
    """MuJoCo-native, differentiable view over a :class:`JaxRobotHandle`
    (``handle.mujoco``). Mirrors the numpy handle's ``.mujoco`` view: MuJoCo
    parameter names, the mjx output convention applied PER CALL (forwards an
    explicit ``_convention="mujoco"``, never mutating the shared default), so it
    is safe alongside pinocchio-convention calls on the same handle. Each method
    stays ``jax.grad``/``jax.vmap``-able (the VJP uses the mjx-convention analytic
    Jacobian). Methods appear here as each one's mjx FFI target lands."""

    __slots__ = ("_h",)

    def __init__(self, handle: "JaxRobotHandle") -> None:
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

    # ── kinematics / regressor ────────────────────────────────────────────
    def end_effector_pose(self, qpos):
        """End-effector pose from mjx-convention qpos (differentiable)."""
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

    def __repr__(self) -> str:
        return f"<mujoco view of {self._h!r}>"


# ─── JaxRobotHandle ─────────────────────────────────────────────────────────


class JaxRobotHandle:
    """JAX-flavored wrapper. Methods return ``jax.Array`` and are jittable.

    Wraps an underlying :py:class:`grid_rbd.RobotHandle` (which dlopens the
    same .so used by the plain Python wrapper) plus per-method JAX FFI
    target registrations.
    """

    def __init__(self, base: RobotHandle, cache_key: str, so_path: str,
                 output_convention: str = "pinocchio"):
        self._base = base
        self._cache_key = cache_key
        self._so_path = Path(so_path)
        if output_convention not in ("pinocchio", "mujoco"):
            raise ValueError(
                f"output_convention must be 'pinocchio' or 'mujoco'; got {output_convention!r}")
        self._output_convention = output_convention
        self._mjx_view = None

    # ─── metadata (delegated) ────────────────────────────────────────────
    @property
    def name(self) -> str:        return self._base.name
    @property
    def num_joints(self) -> int:  return self._base.num_joints
    @property
    def num_vel(self) -> int:     return self._base.num_vel
    @property
    def num_ees(self) -> int:     return self._base.num_ees
    @property
    def num_bodies(self) -> int:  return self._base.num_bodies
    @property
    def floating_base(self) -> bool: return self._base.floating_base
    @property
    def max_batch(self) -> int:   return self._base.max_batch
    @property
    def dtype(self) -> str:
        """Compute precision of the underlying .so (``"float32"`` / ``"float64"``).
        The jax surface is strictly fp32; this mirrors the base handle's report."""
        return self._base.dtype

    # ─── runtime-mutable inertia (D.4 / sysID) ───────────────────────────
    @property
    def runtime_inertia(self) -> bool:
        """True if this robot was registered with ``runtime_inertia=True`` (the
        .so carries a mutable inertia table + :py:meth:`set_inertia_params`)."""
        return self._base.runtime_inertia

    @property
    def inertia_params(self):
        """The BAKED 10-param-per-body inertia table, shape ``(num_bodies, 10)``
        (``[m, h(3), I_O(6)]`` per body). Fetch, mutate, and pass to
        :py:meth:`set_inertia_params`. Only on a ``runtime_inertia`` build."""
        return self._base.inertia_params

    def set_inertia_params(self, params) -> None:
        """Update the device-resident inertia table at runtime (no recompile).
        ``params`` is the ``(num_bodies, 10)`` (or flat ``10*num_bodies``) table in
        the same basis as :py:attr:`inertia_params`. Only valid on a robot
        registered with ``runtime_inertia=True``."""
        self._base.set_inertia_params(params)

    # ─── kernel launch configuration ─────────────────────────────────────
    @property
    def max_perf_level_threads(self) -> int:
        """Codegen-time thread-count hint (DOF-aware, warp-rounded); the default
        per-block thread count for kernel launches."""
        return self._base.max_perf_level_threads

    @property
    def threads_per_block(self) -> int:
        """Current per-block thread count used by kernel launches."""
        return self._base.threads_per_block

    def set_threads_per_block(self, n: int) -> None:
        """Override the per-block thread count for all subsequent kernel launches
        issued through the underlying .so. Default: :py:attr:`max_perf_level_threads`."""
        self._base.set_threads_per_block(n)

    # ─── output convention (mjx parity) ──────────────────────────────────
    @property
    def output_convention(self) -> str:
        """Default IO convention for this handle: ``"pinocchio"`` or ``"mujoco"``.
        Settable. ``"mujoco"`` requires a floating base (mjx and pinocchio coincide
        on a fixed base). Per-call overrides use the thread-safe ``.mujoco`` view."""
        return self._output_convention

    @output_convention.setter
    def output_convention(self, value: str) -> None:
        if value not in ("pinocchio", "mujoco"):
            raise ValueError(
                f"output_convention must be 'pinocchio' or 'mujoco'; got {value!r}")
        if value == "mujoco" and not self.floating_base:
            raise ValueError(
                "output_convention='mujoco' requires a floating-base robot "
                f"({self.name} is fixed-base)")
        self._output_convention = value

    def _resolve_convention(self, convention):
        """None → the handle default; else the explicit per-call convention."""
        return self._output_convention if convention is None else convention

    def _mt(self, convention, method, symbol):
        """Register (and return) the FFI target for a DIRECT (non-custom_vjp) method,
        dispatching to the ``_mujoco`` variant when the resolved convention is mujoco.
        mjx requires a floating base (the _mujoco symbol is #ifdef'd out of fixed .so)."""
        if self._resolve_convention(convention) == "mujoco":
            if not self.floating_base:
                raise ValueError(
                    f"output_convention='mujoco' requires a floating-base robot "
                    f"({self.name} is fixed-base)")
            method, symbol = method + "_mujoco", symbol + "_mujoco"
        return _register_method_target(self._so_path, self._cache_key, method, symbol)

    @property
    def mujoco(self) -> "_JaxMujocoView":
        """MuJoCo-native view (``handle.mujoco.inverse_dynamics(qpos, qvel, qacc)``):
        forwards a per-call ``_convention="mujoco"`` WITHOUT mutating the shared
        ``output_convention`` default, so it is safe alongside pinocchio calls."""
        v = self._mjx_view
        if v is None:
            v = self._mjx_view = _JaxMujocoView(self)
        return v

    # ─── algorithm methods ───────────────────────────────────────────────

    # ─── small helpers ───────────────────────────────────────────────────

    def _prep_2d(self, name: str, *arrays):
        """Cast to float32 jax arrays, validate (B, NJ), enforce same batch.

        Also accepts the per-sample ``(NJ,)`` (1D) shape so the ops compose
        with ``jax.vmap``: under a vmap the mapped slice is 1D, and the FFI
        calls carry ``vmap_method="broadcast_all"`` which re-adds the mapped
        axis and dispatches a single native (B, NJ) kernel. For 1D inputs we
        return ``B = None`` (the batch only materializes inside the vmap).
        """
        import jax.numpy as jnp
        cast = [jnp.asarray(a, dtype=jnp.float32) for a in arrays]
        for i, a in enumerate(cast):
            if a.ndim not in (1, 2) or a.shape[-1] != self.num_joints:
                raise ValueError(
                    f"{name}: arg{i} must be (B, {self.num_joints}) or "
                    f"({self.num_joints},) under vmap; got {a.shape}")
        ndim0 = cast[0].ndim
        for i, a in enumerate(cast[1:], start=1):
            if a.ndim != ndim0:
                raise ValueError(
                    f"{name}: arg{i} ndim={a.ndim} != arg0 ndim={ndim0}")
        if ndim0 == 1:
            return cast, None  # per-sample (vmap) — batch handled by broadcast_all
        B = cast[0].shape[0]
        for i, a in enumerate(cast[1:], start=1):
            if a.shape[0] != B:
                raise ValueError(
                    f"{name}: arg{i} batch={a.shape[0]} != arg0 batch={B}")
        if B > self.max_batch:
            raise ValueError(
                f"{name}: batch={B} > max_batch={self.max_batch}")
        return cast, B

    @staticmethod
    def _out(lead_from, *trailing):
        """Build a ShapeDtypeStruct whose leading dims mirror ``lead_from``
        (the prepped input) and whose trailing dims are ``trailing``.

        For a normal ``(B, NJ)`` input the leading dim is ``(B,)``; under a
        ``jax.vmap`` the prepped slice is ``(NJ,)`` so the leading dim is
        empty — ``vmap_method="broadcast_all"`` re-adds the mapped axis.
        """
        import jax
        import jax.numpy as jnp
        lead = lead_from.shape[:-1]  # drop the NJ axis
        return jax.ShapeDtypeStruct(lead + tuple(trailing), jnp.float32)

    # ─── differentiable-op registry (lazy, per-handle) ───────────────────
    #
    # GRiD emits ANALYTIC gradients, so there is no autodiff tape — the VJP of
    # a forward op is a matvec of the cotangent with the analytic Jacobian
    # (itself an FFI call). We wrap the core forwards with ``jax.custom_vjp``
    # whose ``bwd`` contracts the cotangent with the matching analytic-gradient
    # FFI op. ``vmap_method="broadcast_all"`` on every ffi_call makes the same
    # ops compose with ``jax.vmap`` (a no-op for direct (B,NJ) calls; under a
    # vmap it re-adds the mapped axis and dispatches one native batched call).
    #
    # ``gravity`` is a static (non-differentiated) scalar attribute, declared
    # ``nondiff_argnums=(0,)`` so it is threaded through as a plain Python
    # float (never a JAX tracer) and baked into the FFI attribute. The bwd
    # therefore returns no cotangent for it.
    def _differentiable(self, convention: str = "pinocchio"):
        # Per-convention registry of custom_vjp ops. ``convention="mujoco"`` builds
        # the SAME closures but every FFI target is the ``_mujoco`` variant (kernel
        # launched with MUJOCO_OUTPUT=true): the forward returns mjx-convention
        # outputs and the analytic-gradient VJP is the mjx-convention Jacobian, so
        # jax.grad through an mjx forward stays self-consistent. mjx is FLOATING-base
        # only, non-mimic/non-skew (the _mujoco symbols are #ifdef GRID_RBD_WITH_MUJOCO
        # out of fixed/mimic/skew .so).
        if convention not in ("pinocchio", "mujoco"):
            raise ValueError(
                f"output_convention must be 'pinocchio' or 'mujoco'; got {convention!r}")
        mjx = (convention == "mujoco")
        if mjx and not self.floating_base:
            raise ValueError(
                "output_convention='mujoco' requires a floating-base robot "
                f"({self.name} is fixed-base; mjx and pinocchio coincide there)")
        cache = getattr(self, "_diff_cache", None)
        if cache is None:
            cache = self._diff_cache = {}
        if convention in cache:
            return cache[convention]
        import functools
        import jax
        import jax.numpy as jnp
        import numpy as np
        nj, nv, nee = self.num_joints, self.num_vel, self.num_ees

        def _t(method, symbol):
            # In mjx mode dispatch to the _mujoco-suffixed target/symbol.
            if mjx:
                method, symbol = method + "_mujoco", symbol + "_mujoco"
            return _register_method_target(self._so_path, self._cache_key, method, symbol)

        VM = "broadcast_all"

        # nq↔nv bridge for the VJPs. The dynamics VALUE outputs (c / qdd) are
        # nj-wide (so their cotangents are nj-wide), but the analytic Jacobians
        # are nv x nv (tangent space). For a FLOATING base nv < nj: take the
        # leading nv of the value cotangent (the meaningful tangent rows; the
        # trailing nj-vs-nv slot is the quaternion-padding of the value buffer)
        # before contracting, and pad the nv-wide input cotangent back to nj for
        # the nj-wide q/qd/u inputs. FIXED base nv == nj → both are no-ops.
        def _slice_nv(ct):
            return ct if nv == nj else ct[..., :nv]

        def _pad_nj(g):
            if nv == nj:
                return g
            return jnp.pad(g, [(0, 0)] * (g.ndim - 1) + [(0, nj - nv)])

        # ── forward_dynamics: qdd = f(q,qd,u);  ∂qdd/∂q,∂qdd/∂qd via the
        #    analytic gradient FFI, ∂qdd/∂u = M⁻¹. ───────────────────────────
        @functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
        def fd(gravity, q, qd, u, f_ext):
            t = _t("forward_dynamics", "grid_rbd_jax_forward_dynamics")
            return jax.ffi.ffi_call(t, self._out(q, nj), vmap_method=VM)(
                q, qd, u, f_ext, gravity=np.float32(gravity))

        def fd_fwd(gravity, q, qd, u, f_ext):
            return fd(gravity, q, qd, u, f_ext), (q, qd, u)

        def fd_bwd(gravity, res, ct):
            q, qd, u = res
            tg = _t("forward_dynamics_gradient", "grid_rbd_jax_forward_dynamics_gradient")
            tm = _t("minv", "grid_rbd_jax_minv")
            # The Jacobian / Minv matrices are nv x nv (tangent space); the qdd
            # VALUE (and its cotangent ct) stays nj-wide. For a floating base
            # nv < nj: contract the leading nv of ct (the meaningful tangent rows)
            # and pad the resulting nv-wide input cotangent back to nj. FIXED base
            # nv == nj so _slice_nv / _pad_nj are byte-identical no-ops.
            flat = jax.ffi.ffi_call(tg, self._out(q, 2 * nv * nv), vmap_method=VM)(
                q, qd, u, gravity=np.float32(gravity))
            # GRiD writes (2, NV, NV) column-major; transpose to row-major (out, in).
            blocks = flat.reshape(q.shape[:-1] + (2, nv, nv)).swapaxes(-2, -1)
            df_dq, df_dqd = blocks[..., 0, :, :], blocks[..., 1, :, :]
            mflat = jax.ffi.ffi_call(tm, self._out(q, nv * nv), vmap_method=VM)(
                q, gravity=np.float32(gravity))
            m = mflat.reshape(q.shape[:-1] + (nv, nv))
            # ∂qdd/∂u = Minv. pin minv writes the lower triangle (symmetrize);
            # the mjx minv_mujoco kernel writes a FULL DENSE symmetric matrix (the
            # G^-T Minv G^-1 congruence baked in) so it is used as-is.
            if mjx:
                minv = m
            else:
                eye = jnp.eye(nv, dtype=m.dtype)
                minv = m + jnp.swapaxes(m, -1, -2) - m * eye
            ctv = _slice_nv(ct)
            gq = _pad_nj(jnp.einsum('...o,...oi->...i', ctv, df_dq))
            gqd = _pad_nj(jnp.einsum('...o,...oi->...i', ctv, df_dqd))
            gu = _pad_nj(jnp.einsum('...o,...oi->...i', ctv, minv))
            # cotangents for (q, qd, u, f_ext); f_ext is non-diff.
            return (gq, gqd, gu, None)

        fd.defvjp(fd_fwd, fd_bwd)

        # ── inverse_dynamics (RNEA): τ = M·qdd + h(q,qd) − g(q);  ∂τ/∂(q,qd) via
        #    the analytic gradient FFI. qdd and f_ext are non-differentiated
        #    explicit buffers (the FFI has no optional-buffer support, so the
        #    public method passes zeros when omitted — mirroring idsva_so). The
        #    qdd VALUE is threaded into the analytic grad FFI (which now takes an
        #    explicit qdd buffer), so the q/qd VJP includes ∂(M·qdd)/∂q for a
        #    nonzero-qdd call (a zero qdd reduces to the bias gradient).
        #    qdd/f_ext receive no cotangent. ────────────────────────────────────
        @functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
        def idyn(gravity, q, qd, qdd, f_ext):
            t = _t("inverse_dynamics", "grid_rbd_jax_inverse_dynamics")
            return jax.ffi.ffi_call(t, self._out(q, nj), vmap_method=VM)(
                q, qd, qdd, f_ext, gravity=np.float32(gravity))

        def id_fwd(gravity, q, qd, qdd, f_ext):
            return idyn(gravity, q, qd, qdd, f_ext), (q, qd, qdd)

        def id_bwd(gravity, res, ct):
            q, qd, qdd = res
            tg = _t("inverse_dynamics_gradient", "grid_rbd_jax_inverse_dynamics_gradient")
            # Jacobian is nv x 2nv (tangent); the torque VALUE/cotangent is nj-wide
            # — slice leading nv, contract, pad back to nj (no-op for fixed base).
            flat = jax.ffi.ffi_call(tg, self._out(q, 2 * nv * nv), vmap_method=VM)(
                q, qd, qdd, gravity=np.float32(gravity))
            blocks = flat.reshape(q.shape[:-1] + (2, nv, nv)).swapaxes(-2, -1)
            dc_dq, dc_dqd = blocks[..., 0, :, :], blocks[..., 1, :, :]
            ctv = _slice_nv(ct)
            gq = _pad_nj(jnp.einsum('...o,...oi->...i', ctv, dc_dq))
            gqd = _pad_nj(jnp.einsum('...o,...oi->...i', ctv, dc_dqd))
            # cotangents for (q, qd, qdd, f_ext); qdd/f_ext are non-diff.
            return (gq, gqd, None, None)

        idyn.defvjp(id_fwd, id_bwd)

        # ── end_effector_pose: pose(q) (6*NEE);  ∂pose/∂q via the EE-pose
        #    gradient FFI (TANGENT-space Jacobian, pinocchio convention). ─────
        @jax.custom_vjp
        def eepose(q):
            t = _t("end_effector_pose", "grid_rbd_jax_end_effector_pose")
            return jax.ffi.ffi_call(t, self._out(q, 6 * nee), vmap_method=VM)(q)

        def ee_fwd(q):
            return eepose(q), (q,)

        def ee_bwd(res, ct):
            (q,) = res
            tg = _t("end_effector_pose_gradient", "grid_rbd_jax_end_effector_pose_gradient")
            raw = jax.ffi.ffi_call(tg, self._out(q, 6 * nee * nv), vmap_method=VM)(q)
            # mirror the public reshape: (NEE, NV, 6) → (6*NEE, NV) row-major.
            J = (raw.reshape(q.shape[:-1] + (nee, nv, 6))
                    .swapaxes(-2, -1)
                    .reshape(q.shape[:-1] + (6 * nee, nv)))
            # J cols index NV (tangent); the q input is nj-wide → pad the nv-wide
            # input cotangent back to nj (no-op for fixed base, nv == nj).
            gq = _pad_nj(jnp.einsum('...o,...oi->...i', ct, J))
            return (gq,)

        eepose.defvjp(ee_fwd, ee_bwd)

        nb = self._base.num_bodies
        npar = 10 * nb  # 10 standard inertial params per link

        # ── inverse_dynamics w.r.t. inertial params pi (sysID): the forward op
        #    is the bias c = ID(q,qd,qdd=0), which is AFFINE in pi with Jacobian
        #    the joint-torque regressor Y(q,qd,qdd=0). pi enters as a
        #    differentiable input whose VALUE the forward pass ignores (the .so
        #    carries the baked-in inertia); its cotangent is Yᵀ·ct. This is the
        #    linearization around the compiled model — exactly the outer-loop
        #    sysID gradient. q/qd cotangents still flow via id_gradient. ───────
        @functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
        def idyn_pi(gravity, q, qd, params):
            # forward output is independent of `params` value (baked-in inertia);
            # multiply-add by 0 keeps `params` in the trace for custom_vjp.
            del params
            t = _t("inverse_dynamics", "grid_rbd_jax_inverse_dynamics")
            # ID FFI now takes explicit qdd + f_ext buffers; sysID is the bias
            # (qdd=0) with no external force → pass zeros for both.
            z = jnp.zeros_like(q)
            zfe = jnp.zeros(q.shape[:-1] + (6 * nb,), dtype=q.dtype)
            return jax.ffi.ffi_call(t, self._out(q, nj), vmap_method=VM)(
                q, qd, z, zfe, gravity=np.float32(gravity))

        def id_pi_fwd(gravity, q, qd, params):
            return idyn_pi(gravity, q, qd, params), (q, qd)

        def id_pi_bwd(gravity, res, ct):
            q, qd = res
            # q/qd cotangents (analytic id_gradient). sysID is the bias (qdd=0);
            # the grad FFI now takes an explicit qdd buffer → pass zeros.
            tg = _t("inverse_dynamics_gradient", "grid_rbd_jax_inverse_dynamics_gradient")
            zq = jnp.zeros_like(q)
            # Jacobian + regressor rows are nv-wide; the c cotangent is nj-wide.
            flat = jax.ffi.ffi_call(tg, self._out(q, 2 * nv * nv), vmap_method=VM)(
                q, qd, zq, gravity=np.float32(gravity))
            blocks = flat.reshape(q.shape[:-1] + (2, nv, nv)).swapaxes(-2, -1)
            dc_dq, dc_dqd = blocks[..., 0, :, :], blocks[..., 1, :, :]
            ctv = _slice_nv(ct)
            gq = _pad_nj(jnp.einsum('...o,...oi->...i', ctv, dc_dq))
            gqd = _pad_nj(jnp.einsum('...o,...oi->...i', ctv, dc_dqd))
            # pi cotangent: ctv · Y, Y = ∂c/∂pi (NV x 10*NB) at qdd=0.
            tr = _t("inverse_dynamics_regressor", "grid_rbd_jax_inverse_dynamics_regressor")
            qdd0 = jnp.zeros_like(q)
            Yflat = jax.ffi.ffi_call(tr, self._out(q, nv * npar), vmap_method=VM)(
                q, qd, qdd0, gravity=np.float32(gravity))
            Y = Yflat.reshape(q.shape[:-1] + (nv, npar))  # row-major (NV, 10NB)
            gpi = jnp.einsum('...o,...op->...p', ctv, Y)
            return (gq, gqd, gpi)

        idyn_pi.defvjp(id_pi_fwd, id_pi_bwd)

        # ── forward_dynamics w.r.t. inertial params pi: qdd = FD(q,qd,u);
        #    ∂qdd/∂pi = -Minv · Y(q,qd,qdd_actual) (the analytic
        #    forward_dynamics_parameter_gradient kernel). q/qd/u cotangents flow
        #    exactly as the plain fd VJP. ───────────────────────────────────────
        @functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
        def fd_pi(gravity, q, qd, u, params):
            del params
            t = _t("forward_dynamics", "grid_rbd_jax_forward_dynamics")
            # FD FFI now takes an explicit f_ext buffer; sysID has no external
            # force → pass zeros.
            zfe = jnp.zeros(q.shape[:-1] + (6 * nb,), dtype=q.dtype)
            return jax.ffi.ffi_call(t, self._out(q, nj), vmap_method=VM)(
                q, qd, u, zfe, gravity=np.float32(gravity))

        def fd_pi_fwd(gravity, q, qd, u, params):
            return fd_pi(gravity, q, qd, u, params), (q, qd, u)

        def fd_pi_bwd(gravity, res, ct):
            q, qd, u = res
            tg = _t("forward_dynamics_gradient", "grid_rbd_jax_forward_dynamics_gradient")
            tm = _t("minv", "grid_rbd_jax_minv")
            # Jacobian / Minv / param-gradient rows are nv-wide; qdd cotangent nj-wide.
            flat = jax.ffi.ffi_call(tg, self._out(q, 2 * nv * nv), vmap_method=VM)(
                q, qd, u, gravity=np.float32(gravity))
            blocks = flat.reshape(q.shape[:-1] + (2, nv, nv)).swapaxes(-2, -1)
            df_dq, df_dqd = blocks[..., 0, :, :], blocks[..., 1, :, :]
            mflat = jax.ffi.ffi_call(tm, self._out(q, nv * nv), vmap_method=VM)(
                q, gravity=np.float32(gravity))
            m = mflat.reshape(q.shape[:-1] + (nv, nv))
            eye = jnp.eye(nv, dtype=m.dtype)
            minv = m + jnp.swapaxes(m, -1, -2) - m * eye
            ctv = _slice_nv(ct)
            gq = _pad_nj(jnp.einsum('...o,...oi->...i', ctv, df_dq))
            gqd = _pad_nj(jnp.einsum('...o,...oi->...i', ctv, df_dqd))
            gu = _pad_nj(jnp.einsum('...o,...oi->...i', ctv, minv))
            # pi cotangent: ctv · (∂qdd/∂pi), ∂qdd/∂pi = -Minv·Y (NV x 10*NB).
            tp = _t("forward_dynamics_parameter_gradient",
                    "grid_rbd_jax_forward_dynamics_parameter_gradient")
            Gflat = jax.ffi.ffi_call(tp, self._out(q, nv * npar), vmap_method=VM)(
                q, qd, u, gravity=np.float32(gravity))
            G = Gflat.reshape(q.shape[:-1] + (nv, npar))  # row-major (NV, 10NB)
            gpi = jnp.einsum('...o,...op->...p', ctv, G)
            return (gq, gqd, gu, gpi)

        fd_pi.defvjp(fd_pi_fwd, fd_pi_bwd)

        d = {"forward_dynamics": fd, "inverse_dynamics": idyn, "end_effector_pose": eepose,
             "inverse_dynamics_wrt_params": idyn_pi, "forward_dynamics_wrt_params": fd_pi}
        cache[convention] = d
        return d

    # ─── algorithm methods ───────────────────────────────────────────────

    def _f_ext_or_zeros(self, like, f_ext):
        """Materialize an f_ext buffer (B/…, 6*NUM_BODIES) — JAX FFI has no
        optional-buffer support, so an absent f_ext is passed as explicit zeros
        (the no-f_ext path is then byte-identical). ``like`` is a prepped (…, NJ)
        input whose leading dims + dtype the buffer mirrors."""
        import jax.numpy as jnp
        n = 6 * self.num_bodies
        if f_ext is None:
            return jnp.zeros(like.shape[:-1] + (n,), dtype=like.dtype)
        fe = jnp.asarray(f_ext, dtype=jnp.float32)
        if fe.shape[-1] != n:
            raise ValueError(f"f_ext last dim must be 6*num_bodies = {n}; got {fe.shape}")
        return fe

    def inverse_dynamics(self, q, qd, qdd=None, *, gravity: float = -9.81, f_ext=None,
                         _convention=None):
        """Inverse dynamics (RNEA): τ = M(q)·qdd + h(q,qd) − g(q).

        ``q``, ``qd``: jax.Array shape (B, NJ), dtype float32. With ``qdd=None``
        (default) returns the bias c = h − g; pass a nonzero ``qdd`` for the full
        RNEA torque (the acceleration is plumbed through). Returns shape (B, NJ).

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)`` (see
        the numpy handle); JAX has no optional buffers, so ``None`` is passed as
        explicit zeros internally.

        Differentiable (``jax.grad`` / ``jax.jacobian`` / ``jax.vjp`` w.r.t.
        ``q``, ``qd``) via GRiD's analytic ``inverse_dynamics_gradient`` (the
        backward threads the actual ``qdd`` through, so the Jacobian includes
        ∂(M·qdd)/∂q for a nonzero-qdd call; qdd/f_ext are not differentiated), and
        ``jax.vmap``-able over the leading batch axis.
        """
        import jax.numpy as jnp
        if qdd is None:
            (q, qd), B = self._prep_2d("inverse_dynamics", q, qd)
            qdd_b = jnp.zeros_like(q)
        else:
            (q, qd, qdd_b), B = self._prep_2d("inverse_dynamics", q, qd, qdd)
        fe = self._f_ext_or_zeros(q, f_ext)
        return self._differentiable(self._resolve_convention(_convention))["inverse_dynamics"](
            gravity, q, qd, qdd_b, fe)

    def minv(self, q, *, _convention=None):
        """Direct mass-matrix inverse Minv(q). Returns (B, NV, NV).

        Minv is the tangent-space (pinocchio-convention) inverse mass matrix:
        NV x NV. FIXED base: NV == NJ (shape unchanged); FLOATING base: NV < NJ
        (the kernel writes NUM_VEL*NUM_VEL). The pin kernel writes the lower
        triangle; we symmetrize inside the JAX graph so callers see a full SPD
        matrix. (The plain numpy wrapper does the same.)

        With ``output_convention="mujoco"`` (floating base) the returned Minv is the
        mjx-frame inverse mass matrix (G^-T Minv G^-1); the ``minv_mujoco`` kernel
        writes it FULL DENSE so no host symmetrize is applied.
        """
        import jax
        import jax.numpy as jnp
        conv = self._resolve_convention(_convention)
        target = self._mt(conv, "minv", "grid_rbd_jax_minv")
        (q,), B = self._prep_2d("minv", q)
        nv = self.num_vel
        flat = jax.ffi.ffi_call(target, self._out(q, nv * nv), vmap_method="broadcast_all")(q)
        m = flat.reshape(q.shape[:-1] + (nv, nv))
        if conv == "mujoco":
            return m  # mjx kernel writes a full dense symmetric matrix
        # pin kernel fills the lower triangle; symmetrize as M + Mᵀ − diag(M).
        eye = jnp.eye(nv, dtype=m.dtype)
        return m + jnp.swapaxes(m, -1, -2) - m * eye

    def forward_dynamics(self, q, qd, u, *, gravity: float = -9.81, f_ext=None,
                         _convention=None):
        """qdd = forward_dynamics(q, qd, u). Returns (B, NJ).

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)`` (see
        the numpy handle); ``None`` is passed as explicit zeros internally.

        Differentiable (``jax.grad`` / ``jax.jacobian`` / ``jax.vjp`` w.r.t.
        ``q``, ``qd``, ``u``) via GRiD's analytic ``forward_dynamics_gradient``
        (for ∂qdd/∂q, ∂qdd/∂qd) and ``minv`` (∂qdd/∂u = M⁻¹), and
        ``jax.vmap``-able over the leading batch axis. ``f_ext`` is not differentiated.
        """
        (q, qd, u), B = self._prep_2d("forward_dynamics", q, qd, u)
        fe = self._f_ext_or_zeros(q, f_ext)
        return self._differentiable(self._resolve_convention(_convention))["forward_dynamics"](
            gravity, q, qd, u, fe)

    def inverse_dynamics_wrt_params(self, q, qd, params, *, gravity: float = -9.81):
        """Inverse-dynamics bias c = ID(q, qd, qdd=0), differentiable w.r.t. the
        per-link inertial parameters ``params`` (π) as well as ``q``/``qd``.

        ``params``: (B, 10*NUM_BODIES) — per-link [m, m*c(3), I_O(6)] in the
        parser's origin-frame basis (same as ``RBDReference._regressor`` and the
        ``inverse_dynamics_regressor`` Y). The FORWARD value is independent of
        ``params`` (the compiled ``.so`` carries the baked-in inertia); the op
        exists so ``jax.grad``/``jax.vjp`` can flow the analytic
        ``∂c/∂π = Y(q,qd,qdd=0)`` (regressor) to ``params``. This is the
        linearization of the bias around the compiled model — the outer-loop
        system-ID gradient. ``q``/``qd`` gradients are unchanged.

        Returns (B, NJ). ``jax.vmap``-able over the leading batch axis.
        """
        (q, qd), B = self._prep_2d("inverse_dynamics_wrt_params", q, qd)
        import jax.numpy as jnp
        params = jnp.asarray(params, dtype=jnp.float32)
        return self._differentiable()["inverse_dynamics_wrt_params"](gravity, q, qd, params)

    def forward_dynamics_wrt_params(self, q, qd, u, params, *, gravity: float = -9.81):
        """Forward dynamics qdd = FD(q, qd, u), differentiable w.r.t. the per-link
        inertial parameters ``params`` (π) as well as ``q``/``qd``/``u``.

        ``params``: (B, 10*NUM_BODIES) — see :meth:`inverse_dynamics_wrt_params`.
        The forward value is independent of ``params`` (baked-in inertia); the
        VJP flows the analytic ``∂qdd/∂π = -M⁻¹·Y`` (the
        ``forward_dynamics_parameter_gradient`` kernel) to ``params``.
        ``q``/``qd``/``u`` gradients are unchanged.

        Returns (B, NJ). ``jax.vmap``-able over the leading batch axis.
        """
        (q, qd, u), B = self._prep_2d("forward_dynamics_wrt_params", q, qd, u)
        import jax.numpy as jnp
        params = jnp.asarray(params, dtype=jnp.float32)
        return self._differentiable()["forward_dynamics_wrt_params"](gravity, q, qd, u, params)

    def inverse_dynamics_regressor(self, q, qd, qdd=None, *, gravity: float = -9.81,
                                   _convention=None):
        """Joint-torque regressor Y with τ = Y·π (∂τ/∂π). Returns
        (B, NV, 10*NUM_BODIES). ``qdd=None`` ⇒ zeros (the bias regressor used by
        :meth:`inverse_dynamics_wrt_params`). Row-major (NV, 10*NB) per sample;
        the per-link basis is [m, m*c(3), I_O(6)].

        With ``output_convention="mujoco"`` (floating base) the base-linear rows are
        rotated to the mjx frame (same covector transform as the τ value)."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = self._mt(_convention,
            "inverse_dynamics_regressor", "grid_rbd_jax_inverse_dynamics_regressor")
        if qdd is None:
            qdd = jnp.zeros_like(jnp.asarray(q, dtype=jnp.float32))
        (q, qd, qdd), B = self._prep_2d("inverse_dynamics_regressor", q, qd, qdd)
        nv, npar = self.num_vel, 10 * self.num_bodies
        flat = jax.ffi.ffi_call(target, self._out(q, nv * npar), vmap_method="broadcast_all")(
            q, qd, qdd, gravity=np.float32(gravity))
        return flat.reshape(q.shape[:-1] + (nv, npar))

    def forward_dynamics_parameter_gradient(self, q, qd, u, *, gravity: float = -9.81):
        """FD inertial-parameter gradient ∂qdd/∂π = -M⁻¹·Y. Returns
        (B, NV, 10*NUM_BODIES), row-major (NV, 10*NB) per sample."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = _register_method_target(
            self._so_path, self._cache_key,
            "forward_dynamics_parameter_gradient",
            "grid_rbd_jax_forward_dynamics_parameter_gradient")
        (q, qd, u), B = self._prep_2d("forward_dynamics_parameter_gradient", q, qd, u)
        nv, npar = self.num_vel, 10 * self.num_bodies
        flat = jax.ffi.ffi_call(target, self._out(q, nv * npar), vmap_method="broadcast_all")(
            q, qd, u, gravity=np.float32(gravity))
        return flat.reshape(q.shape[:-1] + (nv, npar))

    def aba(self, q, qd, u, *, gravity: float = -9.81, f_ext=None, _convention=None):
        """qdd = aba(q, qd, u) via the articulated body algorithm. Returns (B, NJ).

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)`` (see
        the numpy handle); ``None`` is passed as explicit zeros internally.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd``/``u`` are
        MuJoCo-convention and the returned ``qdd`` is in the mjx frame."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = self._mt(_convention, "aba", "grid_rbd_jax_aba")
        (q, qd, u), B = self._prep_2d("aba", q, qd, u)
        fe = self._f_ext_or_zeros(q, f_ext)
        out_type = self._out(q, self.num_joints)
        return jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(
            q, qd, u, fe, gravity=np.float32(gravity))

    def crba(self, q, *, gravity: float = -9.81, _convention=None):
        """Mass matrix M(q) via composite rigid body algorithm. Returns (B, NV, NV).

        Tangent-space (pinocchio-convention) mass matrix. FIXED base: NV == NJ
        (shape unchanged); FLOATING base: NV < NJ (the kernel writes NUM_VEL*NUM_VEL).

        With ``output_convention="mujoco"`` (floating base) the returned M is the
        mjx-frame mass matrix (G M G^T congruence, written full dense)."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = self._mt(_convention, "crba", "grid_rbd_jax_crba")
        (q,), B = self._prep_2d("crba", q)
        nv = self.num_vel
        flat = jax.ffi.ffi_call(target, self._out(q, nv * nv), vmap_method="broadcast_all")(
            q, gravity=np.float32(gravity))
        return flat.reshape(q.shape[:-1] + (nv, nv))

    # ─── centroidal / energy / kinematics value methods (numpy-handle parity) ──
    #
    # Mirror the numpy RobotHandle's centroidal / energy / regressor / frame
    # methods exactly (same FFI symbols `grid_rbd_jax_<method>`, same
    # gravity/int attrs, same output reshape). Forward-only (no custom_vjp),
    # matching the numpy surface. The reshapes are copied VERBATIM from
    # `_handle.py` (the oracle) with the leading batch dim taken from
    # ``q.shape[:-1]`` so they also compose under ``jax.vmap``.

    def generalized_gravity(self, q, *, gravity: float = -9.81, _convention=None):
        """Generalized gravity torque g(q) = RNEA(q, 0, 0). Returns (B, NV).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned g is in the mjx frame (base rows rotated in-kernel)."""
        import jax
        import numpy as np
        target = self._mt(_convention, "generalized_gravity", "grid_rbd_jax_generalized_gravity")
        (q,), B = self._prep_2d("generalized_gravity", q)
        nv = self.num_vel
        return jax.ffi.ffi_call(target, self._out(q, nv), vmap_method="broadcast_all")(
            q, gravity=np.float32(gravity))

    def nonlinear_effects(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Nonlinear (bias) effects c(q,qd) = RNEA(q, qd, 0). Returns (B, NV).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and the returned bias matches MuJoCo's ``qfrc_bias``."""
        import jax
        import numpy as np
        target = self._mt(_convention, "nonlinear_effects", "grid_rbd_jax_nonlinear_effects")
        (q, qd), B = self._prep_2d("nonlinear_effects", q, qd)
        nv = self.num_vel
        return jax.ffi.ffi_call(target, self._out(q, nv), vmap_method="broadcast_all")(
            q, qd, gravity=np.float32(gravity))

    def energy(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Kinetic / potential / mechanical energy. Returns (B, 3) = [KE, PE, KE+PE].

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; the energies are frame-INVARIANT (inputs converted in-kernel)."""
        import jax
        import numpy as np
        target = self._mt(_convention, "energy", "grid_rbd_jax_energy")
        (q, qd), B = self._prep_2d("energy", q, qd)
        return jax.ffi.ffi_call(target, self._out(q, 3), vmap_method="broadcast_all")(
            q, qd, gravity=np.float32(gravity))

    def com(self, q, *, _convention=None):
        """Center-of-mass world position p_com and CoM Jacobian J_com.

        Returns ``(p_com, J_com)`` where ``p_com`` is ``(B, 3)`` and ``J_com``
        is ``(B, 3, NV)`` = ``d(p_com)/dv``. The FFI returns one flat ``(B, 3 + 3*NV)``
        buffer ``[p_com(3); J_com(3 x NV col-major)]`` (split mirrors the numpy handle).

        With ``output_convention="mujoco"`` (floating base) ``p_com`` is invariant
        and the ``J_com`` columns are reframed (computed in-kernel)."""
        import jax
        target = self._mt(_convention, "com", "grid_rbd_jax_com")
        (q,), B = self._prep_2d("com", q)
        nv = self.num_vel
        raw = jax.ffi.ffi_call(target, self._out(q, 3 + 3 * nv), vmap_method="broadcast_all")(q)
        lead = q.shape[:-1]
        p_com = raw[..., :3]
        # J_com stored column-major (3 x NV): J[r + 3*c]; recover (..., 3, NV).
        j_com = raw[..., 3:].reshape(lead + (nv, 3)).swapaxes(-2, -1)
        return p_com, j_com

    def ccrba(self, q, qd, *, _convention=None):
        """Centroidal momentum matrix A (6 x NV) and momentum h = A·qd (6,).

        Returns ``(A, h)`` where ``A`` is ``(B, 6, NV)`` and ``h`` is ``(B, 6)``.
        The FFI returns one flat ``(B, 6*NV + 6)`` buffer ``[A(6 x NV col-major); h(6)]``.

        With ``output_convention="mujoco"`` (floating base) ``h`` is invariant and
        the ``A`` columns are reframed (computed in-kernel)."""
        import jax
        target = self._mt(_convention, "ccrba", "grid_rbd_jax_ccrba")
        (q, qd), B = self._prep_2d("ccrba", q, qd)
        nv = self.num_vel
        raw = jax.ffi.ffi_call(target, self._out(q, 6 * nv + 6), vmap_method="broadcast_all")(q, qd)
        lead = q.shape[:-1]
        A = raw[..., : 6 * nv].reshape(lead + (nv, 6)).swapaxes(-2, -1)
        h = raw[..., 6 * nv:]
        return A, h

    def dccrba(self, q, *, _convention=None):
        """dCCRBA tensor ∂A/∂q, shape ``(B, 6, NV, NV)`` indexed
        ``[:, :, k, i] = ∂A[:, k]/∂q_i`` (Pinocchio centroidal convention).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned tensor is the dA/dq of the mjx CMM (same layout)."""
        import jax
        target = self._mt(_convention, "dccrba", "grid_rbd_jax_dccrba")
        (q,), B = self._prep_2d("dccrba", q)
        nv = self.num_vel
        raw = jax.ffi.ffi_call(target, self._out(q, 6 * nv * nv), vmap_method="broadcast_all")(q)
        lead = q.shape[:-1]
        # flat layout dA[row + 6*k + 6*NV*m] -> (..., m, k, row) then -> (..., row, k, m).
        return raw.reshape(lead + (nv, nv, 6)).swapaxes(-3, -1)

    def cmm_time_variation(self, q, qd, *, _convention=None):
        """Centroidal-momentum-matrix time variation Ȧ = dA(q(t))/dt, shape
        ``(B, 6, NV)`` = ``Σ_i (∂A/∂q_i)·qd_i``.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and Ȧ has its columns reframed (computed in-kernel)."""
        import jax
        target = self._mt(_convention, "cmm_time_variation", "grid_rbd_jax_cmm_time_variation")
        (q, qd), B = self._prep_2d("cmm_time_variation", q, qd)
        nv = self.num_vel
        raw = jax.ffi.ffi_call(target, self._out(q, 6 * nv), vmap_method="broadcast_all")(q, qd)
        # (B, 6*NV) col-major A[r + 6*c] -> (..., NV, 6) -> (..., 6, NV).
        return raw.reshape(q.shape[:-1] + (nv, 6)).swapaxes(-2, -1)

    def coriolis_matrix(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Coriolis matrix C(q,qd). Returns ``(B, NV, NV)`` row-major, with
        ``C·qd + g(q) = nonlinear_effects(q, qd)`` (gravity unused by C; the
        kwarg mirrors the host wrapper signature).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and the returned ``C`` is the mjx-frame Coriolis matrix."""
        import jax
        import numpy as np
        target = self._mt(_convention, "coriolis_matrix", "grid_rbd_jax_coriolis_matrix")
        (q, qd), B = self._prep_2d("coriolis_matrix", q, qd)
        nv = self.num_vel
        raw = jax.ffi.ffi_call(target, self._out(q, nv * nv), vmap_method="broadcast_all")(
            q, qd, gravity=np.float32(gravity))
        return raw.reshape(q.shape[:-1] + (nv, nv))  # row-major

    def kinetic_energy_regressor(self, q, qd, *, gravity: float = -9.81, _convention=None):
        """Kinetic-energy regressor y_KE, length ``10*num_bodies``, with
        ``KE = y_KE · π``. Returns ``(B, 10*num_bodies)`` (gravity unused).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; the regressor is frame-INVARIANT (inputs converted in-kernel)."""
        import jax
        import numpy as np
        target = self._mt(_convention,
            "kinetic_energy_regressor", "grid_rbd_jax_kinetic_energy_regressor")
        (q, qd), B = self._prep_2d("kinetic_energy_regressor", q, qd)
        npar = 10 * self.num_bodies
        return jax.ffi.ffi_call(target, self._out(q, npar), vmap_method="broadcast_all")(
            q, qd, gravity=np.float32(gravity))

    def potential_energy_regressor(self, q, *, gravity: float = -9.81, _convention=None):
        """Potential-energy regressor y_PE, length ``10*num_bodies``, with
        ``PE = y_PE · π`` (PE uses ``gravity``). Returns ``(B, 10*num_bodies)``.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        the regressor is frame-INVARIANT (input converted in-kernel)."""
        import jax
        import numpy as np
        target = self._mt(_convention,
            "potential_energy_regressor", "grid_rbd_jax_potential_energy_regressor")
        (q,), B = self._prep_2d("potential_energy_regressor", q)
        npar = 10 * self.num_bodies
        return jax.ffi.ffi_call(target, self._out(q, npar), vmap_method="broadcast_all")(
            q, gravity=np.float32(gravity))

    def frame_jacobian(self, q, *, target_jid=None, reference_frame=None, _convention=None):
        """Geometric Jacobian (6 x NV, ``[linear; angular]``) of a frame.
        Returns ``(B, 6, NV)``.

        ``target_jid`` selects the frame's joint id (default: the leaf
        end-effector joint baked at codegen time). ``reference_frame`` is
        ``'LOCAL'`` (0), ``'WORLD'`` (1), or ``'LOCAL_WORLD_ALIGNED'`` (2, the
        default), or the equivalent int — passed as int64 FFI attrs.

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention
        and the returned Jacobian is column-reframed ``J G⁻¹`` (mjx frame)."""
        import jax
        import numpy as np
        from .._handle import _resolve_frame_args
        target = self._mt(_convention, "frame_jacobian", "grid_rbd_jax_frame_jacobian")
        tj, rf = _resolve_frame_args(self._base._meta, target_jid, reference_frame)
        (q,), B = self._prep_2d("frame_jacobian", q)
        nv = self.num_vel
        raw = jax.ffi.ffi_call(target, self._out(q, 6 * nv), vmap_method="broadcast_all")(
            q, target_jid=np.int64(tj), reference_frame=np.int64(rf))
        # (B, 6*NV) col-major J[r + 6*c] -> (..., NV, 6) -> (..., 6, NV).
        return raw.reshape(q.shape[:-1] + (nv, 6)).swapaxes(-2, -1)

    def frame_jacobian_dot(self, q, qd, *, target_jid=None, reference_frame=None, _convention=None):
        """Time derivative Jdot of :py:meth:`frame_jacobian` along v = qd
        (6 x NV, ``[linear; angular]``). Returns ``(B, 6, NV)``.

        ``target_jid`` / ``reference_frame`` are runtime int64 FFI attrs (default:
        leaf-EE joint / ``LOCAL_WORLD_ALIGNED``); see :py:meth:`frame_jacobian`.

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention and Jdot is column-reframed (mjx frame)."""
        import jax
        import numpy as np
        from .._handle import _resolve_frame_args
        target = self._mt(_convention, "frame_jacobian_dot", "grid_rbd_jax_frame_jacobian_dot")
        tj, rf = _resolve_frame_args(self._base._meta, target_jid, reference_frame)
        (q, qd), B = self._prep_2d("frame_jacobian_dot", q, qd)
        nv = self.num_vel
        raw = jax.ffi.ffi_call(target, self._out(q, 6 * nv), vmap_method="broadcast_all")(
            q, qd, target_jid=np.int64(tj), reference_frame=np.int64(rf))
        return raw.reshape(q.shape[:-1] + (nv, 6)).swapaxes(-2, -1)

    def osc_inertia(self, q, *, _convention=None):
        """Operational-space (task) inertia Lambda = (J·M⁻¹·Jᵀ)⁻¹ (6 x 6) for
        the leaf-EE frame (LWA). Returns ``(B, 6, 6)``.

        With ``output_convention="mujoco"`` the MuJoCo ``q`` is reordered before the
        kinematics build (Lambda is otherwise frame-INVARIANT)."""
        import jax
        target = self._mt(_convention, "osc_inertia", "grid_rbd_jax_osc_inertia")
        (q,), B = self._prep_2d("osc_inertia", q)
        raw = jax.ffi.ffi_call(target, self._out(q, 36), vmap_method="broadcast_all")(q)
        return raw.reshape(q.shape[:-1] + (6, 6))

    def end_effector_pose(self, q, *, _convention=None):
        """End-effector pose [xyz, rpy] per EE. Returns (B, 6*NUM_EES).

        Differentiable (``jax.grad`` / ``jax.jacobian`` / ``jax.vjp`` w.r.t.
        ``q``) via GRiD's analytic ``end_effector_pose_gradient`` (tangent-space
        Jacobian), and ``jax.vmap``-able over the leading batch axis.
        """
        (q,), B = self._prep_2d("end_effector_pose", q)
        return self._differentiable(self._resolve_convention(_convention))["end_effector_pose"](q)

    def end_effector_pose_gradient(self, q, *, _convention=None):
        """End-effector pose Jacobian d/dv (TANGENT space, pinocchio convention).

        Returns (B, 6*NUM_EES, NV). Floating-base now produces the spatial
        Jacobian (omega; v) base block, not the older non-standard quaternion
        derivative columns. Fixed-base shape is unchanged (NV == NJ).

        The kernel writes a column-major (6, NEE*NV) buffer per timestep;
        we mirror the plain wrapper's reshape/transpose to the row-major
        (6*NEE, NV) convention.

        With ``output_convention="mujoco"`` (floating base) the base-velocity
        Jacobian columns are reframed to the mjx free-joint tangent (J·G^-1)."""
        import jax
        import jax.numpy as jnp
        target = self._mt(_convention,
            "end_effector_pose_gradient", "grid_rbd_jax_end_effector_pose_gradient")
        (q,), B = self._prep_2d("end_effector_pose_gradient", q)
        nee = self.num_ees
        nv = self.num_vel
        out_type = self._out(q, 6 * nee * nv)
        raw = jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(q)
        lead = q.shape[:-1]
        return (raw.reshape(lead + (nee, nv, 6))
                   .swapaxes(-2, -1)
                   .reshape(lead + (6 * nee, nv)))

    def end_effector_pose_hessian(self, q, *, _convention=None):
        """End-effector pose Hessian d^2(pose)/dv^2 (tangent space, pinocchio convention).
        Returns (B, 6*NUM_EES, NV, NV). For fixed-base NV == NJ; for floating-base
        the (NV, NV) block indexes spatial twist components.

        With ``output_convention="mujoco"`` (floating base) the base-tangent indices
        are reframed to the mjx free-joint convention."""
        import jax
        import jax.numpy as jnp
        target = self._mt(_convention,
            "end_effector_pose_hessian", "grid_rbd_jax_end_effector_pose_hessian")
        (q,), B = self._prep_2d("end_effector_pose_hessian", q)
        nee = self.num_ees
        nv = self.num_vel
        out_type = self._out(q, 6 * nee * nv * nv)
        flat = jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(q)
        return flat.reshape(q.shape[:-1] + (6 * nee, nv, nv))

    def end_effector_pose_runtime(self, q, ee_joint_names=None, ee_offsets=None,
                                  *, _convention=None):
        """Runtime-target multi-EE pose ``[xyz; rpy]`` at an offset point.

        Mirrors :py:meth:`grid_rbd.RobotHandle.end_effector_pose_runtime`:
        ``ee_joint_names`` (None => all leaf joints, or a str / list of joint
        names) selects the EE frames; ``ee_offsets`` (None => frame origin, or one
        ``[x,y,z]`` / ``[x,y,z,1]`` per EE) shifts the measurement point. The
        single-target FFI op is looped over the resolved jid list (``target_jid``
        + ``off{x,y,z}`` are static FFI attrs) and stacked.

        Returns ``(..., NUM_EE, 6)``. Forward-only (no autograd).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        the world pose VALUE is frame-invariant."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = self._mt(_convention,
            "end_effector_pose_runtime", "grid_rbd_jax_end_effector_pose_runtime")
        jids = self._base._resolve_ee_jids(ee_joint_names)
        offsets = self._base._normalize_ee_offsets(ee_offsets, len(jids))
        (q,), B = self._prep_2d("end_effector_pose_runtime", q)
        out_type = self._out(q, 6)
        per_ee = []
        for jid, off in zip(jids, offsets):
            off = np.asarray(off, dtype=np.float32).reshape(-1)
            raw = jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(
                q, target_jid=np.int64(int(jid)),
                offx=np.float32(off[0]), offy=np.float32(off[1]), offz=np.float32(off[2]))
            per_ee.append(raw)  # (..., 6)
        return jnp.stack(per_ee, axis=-2)  # (..., NUM_EE, 6)

    def end_effector_pose_gradient_runtime(self, q, ee_joint_names=None, ee_offsets=None,
                                           *, _convention=None):
        """Runtime-target multi-EE pose gradient ``d[xyz; rpy]/dv`` (6 x NV) at an
        offset point. Same ``ee_joint_names`` / ``ee_offsets`` semantics as
        :py:meth:`end_effector_pose_runtime`; mirrors
        :py:meth:`grid_rbd.RobotHandle.end_effector_pose_gradient_runtime`.

        The kernel writes a column-major ``(B, 6*NV)`` per target; we reshape
        ``(..., NV, 6)`` then transpose to ``(..., 6, NV)`` and stack.

        Returns ``(..., NUM_EE, 6, NV)``. Forward-only (no autograd).

        With ``output_convention="mujoco"`` (floating base) the base-linear columns
        are reframed by R^T in-kernel (column-reframe class)."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = self._mt(_convention, "end_effector_pose_gradient_runtime",
            "grid_rbd_jax_end_effector_pose_gradient_runtime")
        jids = self._base._resolve_ee_jids(ee_joint_names)
        offsets = self._base._normalize_ee_offsets(ee_offsets, len(jids))
        (q,), B = self._prep_2d("end_effector_pose_gradient_runtime", q)
        nv = self.num_vel
        out_type = self._out(q, 6 * nv)
        per_ee = []
        for jid, off in zip(jids, offsets):
            off = np.asarray(off, dtype=np.float32).reshape(-1)
            raw = jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(
                q, target_jid=np.int64(int(jid)),
                offx=np.float32(off[0]), offy=np.float32(off[1]), offz=np.float32(off[2]))
            # (..., 6*NV) col-major -> (..., NV, 6) -> (..., 6, NV)
            per_ee.append(raw.reshape(q.shape[:-1] + (nv, 6)).swapaxes(-2, -1))
        return jnp.stack(per_ee, axis=-3)  # (..., NUM_EE, 6, NV)

    def inverse_dynamics_gradient(self, q, qd, qdd=None, *, gravity: float = -9.81,
                                  _convention=None):
        """∂c/∂(q, qd) — concatenated [dc_dq | dc_dqd]. Returns (B, NV, 2*NV),
        tangent-space (pinocchio) convention. FIXED base: NV == NJ (unchanged);
        FLOATING base: NV < NJ (the kernel writes nv x 2nv).

        Matches the plain wrapper layout: GRiD writes (2, NV, NV) column-major
        blocks; we reshape/transpose/concat to row-major (NV, 2*NV).

        ``qdd=None`` (default) ⇒ the bias gradient ∂(h−g)/∂(q,qd); pass a nonzero
        ``qdd`` to include ∂(M·qdd)/∂q. JAX has no optional buffers, so ``None``
        is passed as explicit zeros internally (byte-identical to the bias path).

        With ``output_convention="mujoco"`` (floating base) the gradient is the
        mjx-convention Jacobian (rows base-rotated, columns base-reframed)."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = self._mt(_convention,
            "inverse_dynamics_gradient", "grid_rbd_jax_inverse_dynamics_gradient")
        if qdd is None:
            (q, qd), B = self._prep_2d("inverse_dynamics_gradient", q, qd)
            qdd_b = jnp.zeros_like(q)
        else:
            (q, qd, qdd_b), B = self._prep_2d("inverse_dynamics_gradient", q, qd, qdd)
        nv = self.num_vel
        out_type = self._out(q, 2 * nv * nv)
        raw = jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(
            q, qd, qdd_b, gravity=np.float32(gravity))
        blocks = raw.reshape(q.shape[:-1] + (2, nv, nv)).swapaxes(-2, -1)
        return jnp.concatenate([blocks[..., 0, :, :], blocks[..., 1, :, :]], axis=-1)

    def forward_dynamics_gradient(self, q, qd, u, *, gravity: float = -9.81,
                                  _convention=None):
        """∂qdd/∂(q, qd) — concatenated [df_dq | df_dqd]. Returns (B, NV, 2*NV),
        tangent-space (pinocchio) convention. FIXED base: NV == NJ (unchanged);
        FLOATING base: NV < NJ (the kernel writes nv x 2nv).

        With ``output_convention="mujoco"`` (floating base) the gradient is the
        mjx-convention Jacobian."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = self._mt(_convention,
            "forward_dynamics_gradient", "grid_rbd_jax_forward_dynamics_gradient")
        (q, qd, u), B = self._prep_2d("forward_dynamics_gradient", q, qd, u)
        nv = self.num_vel
        out_type = self._out(q, 2 * nv * nv)
        raw = jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(
            q, qd, u, gravity=np.float32(gravity))
        blocks = raw.reshape(q.shape[:-1] + (2, nv, nv)).swapaxes(-2, -1)
        return jnp.concatenate([blocks[..., 0, :, :], blocks[..., 1, :, :]], axis=-1)

    def idsva_so(self, q, qd, qdd=None, *, gravity: float = -9.81, _convention=None):
        """Second-order inverse dynamics at joint acceleration ``qdd``.

        Returns a :class:`grid_rbd.SecondOrderID` NamedTuple of 4 jax.Arrays
        each shape (B, NV, NV, NV): (d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq).
        (A plain tuple — positional unpacking / indexing still work.) Uses the
        codegen-time dispatcher (body-frame fixed-base, world-frame floating-base).

        ``qdd=None`` ⇒ zero acceleration (explicit zeros are passed so the
        result never depends on a stale device buffer from a prior call).

        With ``output_convention="mujoco"`` (floating base) all four tensors are in
        the mjx convention (the kernel transforms every slab).
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = self._mt(_convention, "idsva_so", "grid_rbd_jax_idsva_so")
        if qdd is None:
            qdd = jnp.zeros_like(jnp.asarray(q, dtype=jnp.float32))
        (q, qd, qdd), B = self._prep_2d("idsva_so", q, qd, qdd)
        nv = self.num_vel
        out_type = self._out(q, 4 * nv ** 3)
        flat = jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(
            q, qd, qdd, gravity=np.float32(gravity))
        lead = q.shape[:-1]
        return SecondOrderID(*(
            flat[..., i * nv ** 3:(i + 1) * nv ** 3].reshape(lead + (nv, nv, nv))
            for i in range(4)
        ))

    def fdsva_so(self, q, qd, u, *, gravity: float = -9.81, _convention=None):
        """Second-order forward dynamics.

        Returns a :class:`grid_rbd.SecondOrderFD` NamedTuple of 4 jax.Arrays
        each shape (B, NV, NV, NV) (a plain tuple, so positional unpacking /
        indexing still work). Uses the same scratch buffer (``d_idsva_so``) as
        ``idsva_so``, so the two methods cannot run concurrently on the same handle.

        With ``output_convention="mujoco"`` (floating base) all four tensors are in
        the mjx convention.
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        # mjx fdsva_so LANDED: _mt routes to grid_rbd_jax_fdsva_so_mujoco (the
        # MUJOCO_OUTPUT=true kernel; epilogue recomputes Minv/qdd/dqdd_du fresh into a
        # disjoint scratch band — the §1g/§1h liveness bug is fixed).
        target = self._mt(_convention, "fdsva_so", "grid_rbd_jax_fdsva_so")
        (q, qd, u), B = self._prep_2d("fdsva_so", q, qd, u)
        nv = self.num_vel
        out_type = self._out(q, 4 * nv ** 3)
        flat = jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(
            q, qd, u, gravity=np.float32(gravity))
        lead = q.shape[:-1]
        return SecondOrderFD(*(
            flat[..., i * nv ** 3:(i + 1) * nv ** 3].reshape(lead + (nv, nv, nv))
            for i in range(4)
        ))

    def integrator(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81,
                   _convention=None):
        """One integration step. Returns (B, NUM_POS + NUM_VEL).

        ``dt`` and the integrator type are passed as FFI attributes (runtime
        scalars); gravity is the signed gravitational acceleration (default -9.81).

        With ``output_convention="mujoco"`` (floating base) inputs/outputs are
        MuJoCo-convention (global-additive base retract + reframed base velocity).
        """
        import numpy as np
        import jax
        import jax.numpy as jnp
        from .._handle import _integrator_code
        target = self._mt(_convention, "integrator", "grid_rbd_jax_integrator")
        (q, qd, u), B = self._prep_2d("integrator", q, qd, u)
        out_type = jax.ShapeDtypeStruct((B, self.num_joints + self.num_vel), jnp.float32)
        return jax.ffi.ffi_call(target, out_type)(
            q, qd, u, dt=np.float32(dt), it=np.int64(_integrator_code(integrator_type)),
            gravity=np.float32(gravity))

    def integrator_gradient(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81,
                            _convention=None):
        """Gradient of the integrator step. Returns (B, 2*NV, 3*NV) — column
        blocks [d/dq | d/dqd | d/du] in tangent space.

        With ``output_convention="mujoco"`` (floating base) the state-transition
        Jacobian is in the mjx convention (retract tangent + reframed velocity)."""
        import numpy as np
        import jax
        import jax.numpy as jnp
        from .._handle import _integrator_code
        target = self._mt(_convention,
            "integrator_gradient", "grid_rbd_jax_integrator_gradient")
        (q, qd, u), B = self._prep_2d("integrator_gradient", q, qd, u)
        nv = self.num_vel
        out_type = jax.ShapeDtypeStruct((B, 2 * nv * 3 * nv), jnp.float32)
        flat = jax.ffi.ffi_call(target, out_type)(
            q, qd, u, dt=np.float32(dt), it=np.int64(_integrator_code(integrator_type)),
            gravity=np.float32(gravity))
        # h_dAB is (2*NV x 3*NV) column-major per timestep; recover row-major.
        return flat.reshape(B, 3 * nv, 2 * nv).transpose(0, 2, 1)

    # ─── field-standard short aliases ────────────────────────────────────────
    # `rnea`/`fd` are the names roboticists reach for (pinocchio / frax / bard);
    # bind them to the long-named methods (aba / crba / minv already match).
    rnea = inverse_dynamics
    fd = forward_dynamics

    # ─── grid_plant surface (cost / barrier / plant-step) ────────────────────
    #
    # Mirror the numpy RobotHandle plant methods exactly (shapes / fields /
    # gravity / integrator_type). Cost methods return (value, grad, hess);
    # barriers return (value, grad, hess_diag). value is squeezed to (B,) to
    # match the numpy surface.

    def _prep_plant(self, name, **arrays):
        """Cast to float32 jax arrays, enforce 2D + same batch + max_batch.
        Unlike _prep_2d this allows arbitrary last dims (the plant inputs are
        not all (B, NJ))."""
        import jax.numpy as jnp
        cast = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in arrays.items()}
        first = next(iter(cast.values()))
        if first.ndim != 2:
            raise ValueError(f"{name}: inputs must be 2D (B, N)")
        B = first.shape[0]
        for k, a in cast.items():
            if a.ndim != 2 or a.shape[0] != B:
                raise ValueError(f"{name}: {k} must be 2D with batch={B}; got {a.shape}")
        if B > self.max_batch:
            raise ValueError(f"{name}: batch={B} > max_batch={self.max_batch}")
        return cast, B

    def _cost_out_types(self, B, n_grad, n_hess):
        import jax
        import jax.numpy as jnp
        return (
            jax.ShapeDtypeStruct((B, 1), jnp.float32),
            jax.ShapeDtypeStruct((B, n_grad), jnp.float32),
            jax.ShapeDtypeStruct((B, n_hess), jnp.float32),
        )

    def quadratic_state_cost(self, x, x_des, Q, *, _convention=None):
        """1/2 sum_i Q_i (x_i - x_des_i)^2 over x=[q;qd]. Returns
        (value (B,), grad (B, NX), hess=diag(Q) (B, NX, NX)).

        With ``output_convention="mujoco"`` (floating base) ``x`` is MuJoCo-convention;
        the value is invariant, the grad base-rotates (covector) and the GN hess is
        the mjx congruence."""
        import jax
        target = self._mt(_convention,
            "plant_quadratic_state_cost", "grid_rbd_jax_plant_quadratic_state_cost")
        cast, B = self._prep_plant("quadratic_state_cost", x=x, x_des=x_des, Q=Q)
        nx = self.num_joints + self.num_vel
        out, grad, hess = jax.ffi.ffi_call(target, self._cost_out_types(B, nx, nx * nx))(
            cast["x"], cast["x_des"], cast["Q"])
        return out[:, 0], grad, hess.reshape(B, nx, nx)

    def quadratic_input_cost(self, u, u_des, R):
        """1/2 sum_i R_i (u_i - u_des_i)^2 over u (NV). Returns
        (value (B,), grad (B, NV), hess=diag(R) (B, NV, NV))."""
        import jax
        target = _register_method_target(
            self._so_path, self._cache_key,
            "plant_quadratic_input_cost", "grid_rbd_jax_plant_quadratic_input_cost")
        cast, B = self._prep_plant("quadratic_input_cost", u=u, u_des=u_des, R=R)
        nv = self.num_vel
        out, grad, hess = jax.ffi.ffi_call(target, self._cost_out_types(B, nv, nv * nv))(
            cast["u"], cast["u_des"], cast["R"])
        return out[:, 0], grad, hess.reshape(B, nv, nv)

    def _barrier(self, name, symbol, var, lower, upper, mu, n):
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = _register_method_target(self._so_path, self._cache_key, name, symbol)
        cast, B = self._prep_plant(name, var=var, lower=lower, upper=upper)
        out_types = (
            jax.ShapeDtypeStruct((B, 1), jnp.float32),
            jax.ShapeDtypeStruct((B, n), jnp.float32),
            jax.ShapeDtypeStruct((B, n), jnp.float32),
        )
        out, grad, hdiag = jax.ffi.ffi_call(target, out_types)(
            cast["var"], cast["lower"], cast["upper"], mu=np.float32(mu))
        return out[:, 0], grad, hdiag

    def joint_position_barrier(self, var, lower, upper, mu):
        """Log-barrier over NUM_POS positions. Returns
        (value (B,), grad (B, NUM_POS), hess_diag (B, NUM_POS))."""
        return self._barrier("plant_joint_position_barrier",
                             "grid_rbd_jax_plant_joint_position_barrier",
                             var, lower, upper, mu, self.num_joints)

    def joint_velocity_barrier(self, var, lower, upper, mu):
        """Log-barrier over NUM_VEL velocities. See joint_position_barrier."""
        return self._barrier("plant_joint_velocity_barrier",
                             "grid_rbd_jax_plant_joint_velocity_barrier",
                             var, lower, upper, mu, self.num_vel)

    def joint_torque_barrier(self, var, lower, upper, mu):
        """Log-barrier over NUM_VEL torques. See joint_position_barrier."""
        return self._barrier("plant_joint_torque_barrier",
                             "grid_rbd_jax_plant_joint_torque_barrier",
                             var, lower, upper, mu, self.num_vel)

    def plant_step(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81,
                   _convention=None):
        """x_{k+1} = integrator(x_k, u_k, dt). x (B, NX); u (B, NV). Returns (B, NX).

        With ``output_convention="mujoco"`` (floating base) ``x`` is MuJoCo-convention
        and the returned next state is mjx-convention (global-additive base retract)."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from .._handle import _integrator_code
        target = self._mt(_convention, "plant_step", "grid_rbd_jax_plant_step")
        cast, B = self._prep_plant("plant_step", x=x, u=u)
        nx = self.num_joints + self.num_vel
        out_type = jax.ShapeDtypeStruct((B, nx), jnp.float32)
        return jax.ffi.ffi_call(target, out_type)(
            cast["x"], cast["u"], dt=np.float32(dt),
            it=np.int64(_integrator_code(integrator_type)), gravity=np.float32(gravity))

    def plant_step_gradient(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81,
                            _convention=None):
        """[A|B] = d x_{k+1}/d(x,u). x (B, NX); u (B, NV). Returns (B, 2*NV, 3*NV)
        with column blocks [d/dq | d/dqd | d/du] (tangent space).

        With ``output_convention="mujoco"`` (floating base) the state-transition
        Jacobian is in the mjx convention."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from .._handle import _integrator_code
        target = self._mt(_convention,
            "plant_step_gradient", "grid_rbd_jax_plant_step_gradient")
        cast, B = self._prep_plant("plant_step_gradient", x=x, u=u)
        nv = self.num_vel
        out_type = jax.ShapeDtypeStruct((B, 2 * nv * 3 * nv), jnp.float32)
        flat = jax.ffi.ffi_call(target, out_type)(
            cast["x"], cast["u"], dt=np.float32(dt),
            it=np.int64(_integrator_code(integrator_type)), gravity=np.float32(gravity))
        # (2*NV x 3*NV) column-major per timestep; recover row-major.
        return flat.reshape(B, 3 * nv, 2 * nv).transpose(0, 2, 1)

    def ee_pos_cost(self, q, p_des, W, *, _convention=None):
        """End-effector position cost (EE 0). q (B, NQ); p_des/W (B, 3). Returns
        (value (B,), grad_x (B, NX), GN hess_x (B, NX, NX)).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        value invariant, grad covector-rotated, GN hess the mjx congruence."""
        import jax
        target = self._mt(_convention, "plant_ee_pos_cost", "grid_rbd_jax_plant_ee_pos_cost")
        cast, B = self._prep_plant("ee_pos_cost", q=q, p_des=p_des, W=W)
        nx = self.num_joints + self.num_vel
        out, grad, hess = jax.ffi.ffi_call(target, self._cost_out_types(B, nx, nx * nx))(
            cast["q"], cast["p_des"], cast["W"])
        return out[:, 0], grad, hess.reshape(B, nx, nx)

    def com_cost(self, q, p_des, W, *, _convention=None):
        """Center-of-mass tracking cost. q (B, NQ); p_des/W (B, 3). Returns
        (value (B,), grad_x (B, NX), GN hess_x (B, NX, NX)).

        With ``output_convention="mujoco"`` (floating base) ``q`` is MuJoCo-convention;
        value invariant, grad covector-rotated, GN hess the mjx congruence."""
        import jax
        target = self._mt(_convention, "plant_com_cost", "grid_rbd_jax_plant_com_cost")
        cast, B = self._prep_plant("com_cost", q=q, p_des=p_des, W=W)
        nx = self.num_joints + self.num_vel
        out, grad, hess = jax.ffi.ffi_call(target, self._cost_out_types(B, nx, nx * nx))(
            cast["q"], cast["p_des"], cast["W"])
        return out[:, 0], grad, hess.reshape(B, nx, nx)

    def momentum_cost(self, q, qd, h_des, W, *, _convention=None):
        """Centroidal-momentum tracking cost. q (B, NQ); qd (B, NV); h_des/W (B, 6).
        Returns (value (B,), grad_x (B, NX), GN hess_x (B, NX, NX)).

        With ``output_convention="mujoco"`` (floating base) ``q``/``qd`` are
        MuJoCo-convention; value invariant, grad covector-rotated, GN hess congruence."""
        import jax
        target = self._mt(_convention, "plant_momentum_cost", "grid_rbd_jax_plant_momentum_cost")
        cast, B = self._prep_plant("momentum_cost", q=q, qd=qd, h_des=h_des, W=W)
        nx = self.num_joints + self.num_vel
        out, grad, hess = jax.ffi.ffi_call(target, self._cost_out_types(B, nx, nx * nx))(
            cast["q"], cast["qd"], cast["h_des"], cast["W"])
        return out[:, 0], grad, hess.reshape(B, nx, nx)

    # ─── lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the underlying .so handle (delegates to the base RobotHandle).
        Idempotent. The process-global JAX FFI target registrations are unaffected."""
        self._base.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ─── public API ─────────────────────────────────────────────────────────────


def register_robot(
    name: str,
    urdf_path: str | None = None,
    *,
    urdf_string: str | None = None,
    floating_base: bool = False,
    ee_joint_names: list[str] | tuple[str, ...] | None = None,
    max_batch_size: int = 256,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
    cuda_arch: int | None = None,
    output_convention: str = "pinocchio",
) -> JaxRobotHandle:
    """Register a robot for use with JAX.

    Compiles + caches the same per-robot ``.so`` that
    :py:func:`grid_rbd.register_robot` uses (cache hit if already
    compiled). Additionally registers JAX FFI targets so the methods
    are callable inside ``jax.jit``.

    Returns a :py:class:`JaxRobotHandle`.
    """
    _require_jax()  # fail early with install guidance if jax is missing
    base = _grid_rbd.register_robot(
        name=name,
        urdf_path=urdf_path,
        urdf_string=urdf_string,
        floating_base=floating_base,
        ee_joint_names=ee_joint_names,
        max_batch_size=max_batch_size,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
        cuda_arch=cuda_arch,
    )
    # Pull the cache_key + .so path from the manifest so we can dlopen
    # to register JAX FFI symbols.
    from grid_rbd._cache import default_cache_dir, manifest_lookup, store_dir
    cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    entry = manifest_lookup(cache_dir, name)
    if entry is None:
        raise RuntimeError(
            f"register_robot returned but {name!r} isn't in manifest; "
            f"cache may be corrupted")
    so_path = store_dir(cache_dir, entry["cache_key"]) / "robot.so"
    return JaxRobotHandle(base, entry["cache_key"], str(so_path),
                          output_convention=output_convention)


def get_robot(
    name: str,
    cache_dir: str | Path | None = None,
    *,
    output_convention: str = "pinocchio",
) -> JaxRobotHandle:
    """Look up a previously-registered robot. Same cache as
    :py:func:`grid_rbd.get_robot`. ``output_convention`` ('pinocchio' or
    'mujoco') mirrors :py:func:`register_robot` and can also be set later
    via the handle's ``output_convention`` property."""
    _require_jax()  # fail early with install guidance if jax is missing
    base = _grid_rbd.get_robot(name, cache_dir=cache_dir)
    from grid_rbd._cache import default_cache_dir, manifest_lookup, store_dir
    cd = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    entry = manifest_lookup(cd, name)
    so_path = store_dir(cd, entry["cache_key"]) / "robot.so"
    return JaxRobotHandle(base, entry["cache_key"], str(so_path),
                          output_convention=output_convention)


__all__ = ["JaxRobotHandle", "register_robot", "get_robot"]

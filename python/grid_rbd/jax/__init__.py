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
  ``end_effector_pose``, ``end_effector_pose_gradient``,
  ``end_effector_pose_hessian``, ``inverse_dynamics_gradient``, ``forward_dynamics_gradient``,
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
from grid_rbd._handle import RobotHandle


# ─── handler registry ───────────────────────────────────────────────────────

# Maps (cache_key, method_name) → bool indicating whether the FFI target has
# been registered with JAX. Registration is process-global (JAX maintains the
# target table), so we only need to do it once per (cache_key, method).
_REGISTERED: dict[tuple[str, str], bool] = {}
_LOCK = threading.Lock()


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
    import jax
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


# ─── JaxRobotHandle ─────────────────────────────────────────────────────────


class JaxRobotHandle:
    """JAX-flavored wrapper. Methods return ``jax.Array`` and are jittable.

    Wraps an underlying :py:class:`grid_rbd.RobotHandle` (which dlopens the
    same .so used by the plain Python wrapper) plus per-method JAX FFI
    target registrations.
    """

    def __init__(self, base: RobotHandle, cache_key: str, so_path: str):
        self._base = base
        self._cache_key = cache_key
        self._so_path = Path(so_path)

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
    def floating_base(self) -> bool: return self._base.floating_base
    @property
    def max_batch(self) -> int:   return self._base.max_batch

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
    def _differentiable(self):
        d = getattr(self, "_diff_cache", None)
        if d is not None:
            return d
        import functools
        import jax
        import jax.numpy as jnp
        import numpy as np
        nj, nv, nee = self.num_joints, self.num_vel, self.num_ees

        def _t(method, symbol):
            return _register_method_target(self._so_path, self._cache_key, method, symbol)

        VM = "broadcast_all"

        # ── forward_dynamics: qdd = f(q,qd,u);  ∂qdd/∂q,∂qdd/∂qd via the
        #    analytic gradient FFI, ∂qdd/∂u = M⁻¹. ───────────────────────────
        @functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
        def fd(gravity, q, qd, u):
            t = _t("forward_dynamics", "grid_rbd_jax_forward_dynamics")
            return jax.ffi.ffi_call(t, self._out(q, nj), vmap_method=VM)(
                q, qd, u, gravity=np.float32(gravity))

        def fd_fwd(gravity, q, qd, u):
            return fd(gravity, q, qd, u), (q, qd, u)

        def fd_bwd(gravity, res, ct):
            q, qd, u = res
            tg = _t("forward_dynamics_gradient", "grid_rbd_jax_forward_dynamics_gradient")
            tm = _t("minv", "grid_rbd_jax_minv")
            flat = jax.ffi.ffi_call(tg, self._out(q, 2 * nj * nj), vmap_method=VM)(
                q, qd, u, gravity=np.float32(gravity))
            # GRiD writes (2, NJ, NJ) column-major; transpose to row-major (out, in).
            blocks = flat.reshape(q.shape[:-1] + (2, nj, nj)).swapaxes(-2, -1)
            df_dq, df_dqd = blocks[..., 0, :, :], blocks[..., 1, :, :]
            mflat = jax.ffi.ffi_call(tm, self._out(q, nj * nj), vmap_method=VM)(
                q, gravity=np.float32(gravity))
            m = mflat.reshape(q.shape[:-1] + (nj, nj))
            eye = jnp.eye(nj, dtype=m.dtype)
            minv = m + jnp.swapaxes(m, -1, -2) - m * eye  # ∂qdd/∂u
            gq = jnp.einsum('...o,...oi->...i', ct, df_dq)
            gqd = jnp.einsum('...o,...oi->...i', ct, df_dqd)
            gu = jnp.einsum('...o,...oi->...i', ct, minv)
            return (gq, gqd, gu)

        fd.defvjp(fd_fwd, fd_bwd)

        # ── inverse_dynamics (bias): c = h(q,qd) − g(q);  ∂c/∂(q,qd) via the
        #    analytic gradient FFI. ──────────────────────────────────────────
        @functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
        def idyn(gravity, q, qd):
            t = _t("inverse_dynamics", "grid_rbd_jax_inverse_dynamics")
            return jax.ffi.ffi_call(t, self._out(q, nj), vmap_method=VM)(
                q, qd, gravity=np.float32(gravity))

        def id_fwd(gravity, q, qd):
            return idyn(gravity, q, qd), (q, qd)

        def id_bwd(gravity, res, ct):
            q, qd = res
            tg = _t("inverse_dynamics_gradient", "grid_rbd_jax_inverse_dynamics_gradient")
            flat = jax.ffi.ffi_call(tg, self._out(q, 2 * nj * nj), vmap_method=VM)(
                q, qd, gravity=np.float32(gravity))
            blocks = flat.reshape(q.shape[:-1] + (2, nj, nj)).swapaxes(-2, -1)
            dc_dq, dc_dqd = blocks[..., 0, :, :], blocks[..., 1, :, :]
            gq = jnp.einsum('...o,...oi->...i', ct, dc_dq)
            gqd = jnp.einsum('...o,...oi->...i', ct, dc_dqd)
            return (gq, gqd)

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
            gq = jnp.einsum('...o,...oi->...i', ct, J)  # cols index NV(=NJ fixed-base)
            return (gq,)

        eepose.defvjp(ee_fwd, ee_bwd)

        d = {"forward_dynamics": fd, "inverse_dynamics": idyn, "end_effector_pose": eepose}
        self._diff_cache = d
        return d

    # ─── algorithm methods ───────────────────────────────────────────────

    def inverse_dynamics(self, q, qd, *, gravity: float = -9.81):
        """Inverse dynamics: c = M(q)·qdd_zero + h(q,qd) − g(q).

        ``q``, ``qd``: jax.Array shape (B, NJ), dtype float32.
        Returns shape (B, NJ).

        Differentiable (``jax.grad`` / ``jax.jacobian`` / ``jax.vjp`` w.r.t.
        ``q``, ``qd``) via GRiD's analytic ``inverse_dynamics_gradient``, and
        ``jax.vmap``-able over the leading batch axis.
        """
        (q, qd), B = self._prep_2d("inverse_dynamics", q, qd)
        return self._differentiable()["inverse_dynamics"](gravity, q, qd)

    def minv(self, q):
        """Direct mass-matrix inverse Minv(q). Returns (B, NJ, NJ).

        The kernel writes the lower triangle; we symmetrize inside the JAX
        graph so callers see a full SPD matrix. (The plain wrapper does the
        same in numpy.)
        """
        import jax
        import jax.numpy as jnp
        target = _register_method_target(
            self._so_path, self._cache_key, "minv", "grid_rbd_jax_minv")
        (q,), B = self._prep_2d("minv", q)
        nj = self.num_joints
        flat = jax.ffi.ffi_call(target, self._out(q, nj * nj), vmap_method="broadcast_all")(q)
        m = flat.reshape(q.shape[:-1] + (nj, nj))
        # Kernel fills the lower triangle; symmetrize as M + Mᵀ − diag(M).
        eye = jnp.eye(nj, dtype=m.dtype)
        return m + jnp.swapaxes(m, -1, -2) - m * eye

    def forward_dynamics(self, q, qd, u, *, gravity: float = -9.81):
        """qdd = forward_dynamics(q, qd, u). Returns (B, NJ).

        Differentiable (``jax.grad`` / ``jax.jacobian`` / ``jax.vjp`` w.r.t.
        ``q``, ``qd``, ``u``) via GRiD's analytic ``forward_dynamics_gradient``
        (for ∂qdd/∂q, ∂qdd/∂qd) and ``minv`` (∂qdd/∂u = M⁻¹), and
        ``jax.vmap``-able over the leading batch axis.
        """
        (q, qd, u), B = self._prep_2d("forward_dynamics", q, qd, u)
        return self._differentiable()["forward_dynamics"](gravity, q, qd, u)

    def aba(self, q, qd, u, *, gravity: float = -9.81):
        """qdd = aba(q, qd, u) via the articulated body algorithm. Returns (B, NJ)."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = _register_method_target(
            self._so_path, self._cache_key, "aba", "grid_rbd_jax_aba")
        (q, qd, u), B = self._prep_2d("aba", q, qd, u)
        out_type = self._out(q, self.num_joints)
        return jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(
            q, qd, u, gravity=np.float32(gravity))

    def crba(self, q, *, gravity: float = -9.81):
        """Mass matrix M(q) via composite rigid body algorithm. Returns (B, NJ, NJ)."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = _register_method_target(
            self._so_path, self._cache_key, "crba", "grid_rbd_jax_crba")
        (q,), B = self._prep_2d("crba", q)
        nj = self.num_joints
        flat = jax.ffi.ffi_call(target, self._out(q, nj * nj), vmap_method="broadcast_all")(
            q, gravity=np.float32(gravity))
        return flat.reshape(q.shape[:-1] + (nj, nj))

    def end_effector_pose(self, q):
        """End-effector pose [xyz, rpy] per EE. Returns (B, 6*NUM_EES).

        Differentiable (``jax.grad`` / ``jax.jacobian`` / ``jax.vjp`` w.r.t.
        ``q``) via GRiD's analytic ``end_effector_pose_gradient`` (tangent-space
        Jacobian), and ``jax.vmap``-able over the leading batch axis.
        """
        (q,), B = self._prep_2d("end_effector_pose", q)
        return self._differentiable()["end_effector_pose"](q)

    def end_effector_pose_gradient(self, q):
        """End-effector pose Jacobian d/dv (TANGENT space, pinocchio convention).

        Returns (B, 6*NUM_EES, NV). Floating-base now produces the spatial
        Jacobian (omega; v) base block, not the older non-standard quaternion
        derivative columns. Fixed-base shape is unchanged (NV == NJ).

        The kernel writes a column-major (6, NEE*NV) buffer per timestep;
        we mirror the plain wrapper's reshape/transpose to the row-major
        (6*NEE, NV) convention.
        """
        import jax
        import jax.numpy as jnp
        target = _register_method_target(
            self._so_path, self._cache_key,
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

    def end_effector_pose_hessian(self, q):
        """End-effector pose Hessian d^2(pose)/dv^2 (tangent space, pinocchio convention).
        Returns (B, 6*NUM_EES, NV, NV). For fixed-base NV == NJ; for floating-base
        the (NV, NV) block indexes spatial twist components."""
        import jax
        import jax.numpy as jnp
        target = _register_method_target(
            self._so_path, self._cache_key,
            "end_effector_pose_hessian", "grid_rbd_jax_end_effector_pose_hessian")
        (q,), B = self._prep_2d("end_effector_pose_hessian", q)
        nee = self.num_ees
        nv = self.num_vel
        out_type = self._out(q, 6 * nee * nv * nv)
        flat = jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(q)
        return flat.reshape(q.shape[:-1] + (6 * nee, nv, nv))

    def inverse_dynamics_gradient(self, q, qd, *, gravity: float = -9.81):
        """∂c/∂(q, qd) — concatenated [dc_dq | dc_dqd]. Returns (B, NJ, 2*NJ).

        Matches the plain wrapper layout: GRiD writes (2, NJ, NJ) column-major
        blocks; we reshape/transpose/concat to row-major (NJ, 2*NJ).
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = _register_method_target(
            self._so_path, self._cache_key,
            "inverse_dynamics_gradient", "grid_rbd_jax_inverse_dynamics_gradient")
        (q, qd), B = self._prep_2d("inverse_dynamics_gradient", q, qd)
        nj = self.num_joints
        out_type = self._out(q, 2 * nj * nj)
        raw = jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(
            q, qd, gravity=np.float32(gravity))
        blocks = raw.reshape(q.shape[:-1] + (2, nj, nj)).swapaxes(-2, -1)
        return jnp.concatenate([blocks[..., 0, :, :], blocks[..., 1, :, :]], axis=-1)

    def forward_dynamics_gradient(self, q, qd, u, *, gravity: float = -9.81):
        """∂qdd/∂(q, qd) — concatenated [df_dq | df_dqd]. Returns (B, NJ, 2*NJ)."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = _register_method_target(
            self._so_path, self._cache_key,
            "forward_dynamics_gradient", "grid_rbd_jax_forward_dynamics_gradient")
        (q, qd, u), B = self._prep_2d("forward_dynamics_gradient", q, qd, u)
        nj = self.num_joints
        out_type = self._out(q, 2 * nj * nj)
        raw = jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(
            q, qd, u, gravity=np.float32(gravity))
        blocks = raw.reshape(q.shape[:-1] + (2, nj, nj)).swapaxes(-2, -1)
        return jnp.concatenate([blocks[..., 0, :, :], blocks[..., 1, :, :]], axis=-1)

    def idsva_so(self, q, qd, qdd=None, *, gravity: float = -9.81):
        """Second-order inverse dynamics at joint acceleration ``qdd``.

        Returns a tuple of 4 jax.Arrays each shape (B, NV, NV, NV):
        (d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq). Uses the codegen-time
        dispatcher (body-frame for fixed-base, world-frame for floating-base).

        ``qdd=None`` ⇒ zero acceleration (explicit zeros are passed so the
        result never depends on a stale device buffer from a prior call).
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = _register_method_target(
            self._so_path, self._cache_key,
            "idsva_so", "grid_rbd_jax_idsva_so")
        if qdd is None:
            qdd = jnp.zeros_like(jnp.asarray(q, dtype=jnp.float32))
        (q, qd, qdd), B = self._prep_2d("idsva_so", q, qd, qdd)
        nv = self.num_vel
        out_type = self._out(q, 4 * nv ** 3)
        flat = jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(
            q, qd, qdd, gravity=np.float32(gravity))
        lead = q.shape[:-1]
        return tuple(
            flat[..., i * nv ** 3:(i + 1) * nv ** 3].reshape(lead + (nv, nv, nv))
            for i in range(4)
        )

    def fdsva_so(self, q, qd, u, *, gravity: float = -9.81):
        """Second-order forward dynamics.

        Returns a tuple of 4 jax.Arrays each shape (B, NV, NV, NV). Uses
        the same scratch buffer (``d_idsva_so``) as ``idsva_so``, so the
        two methods cannot run concurrently on the same handle.
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        target = _register_method_target(
            self._so_path, self._cache_key,
            "fdsva_so", "grid_rbd_jax_fdsva_so")
        (q, qd, u), B = self._prep_2d("fdsva_so", q, qd, u)
        nv = self.num_vel
        out_type = self._out(q, 4 * nv ** 3)
        flat = jax.ffi.ffi_call(target, out_type, vmap_method="broadcast_all")(
            q, qd, u, gravity=np.float32(gravity))
        lead = q.shape[:-1]
        return tuple(
            flat[..., i * nv ** 3:(i + 1) * nv ** 3].reshape(lead + (nv, nv, nv))
            for i in range(4)
        )

    def integrator(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """One integration step. Returns (B, NUM_POS + NUM_VEL).

        ``dt`` and the integrator type are passed as FFI attributes (runtime
        scalars); gravity is the signed gravitational acceleration (default -9.81).
        """
        import numpy as np
        import jax
        import jax.numpy as jnp
        from .._handle import _integrator_code
        target = _register_method_target(
            self._so_path, self._cache_key, "integrator", "grid_rbd_jax_integrator")
        (q, qd, u), B = self._prep_2d("integrator", q, qd, u)
        out_type = jax.ShapeDtypeStruct((B, self.num_joints + self.num_vel), jnp.float32)
        return jax.ffi.ffi_call(target, out_type)(
            q, qd, u, dt=np.float32(dt), it=np.int64(_integrator_code(integrator_type)),
            gravity=np.float32(gravity))

    def integrator_gradient(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """Gradient of the integrator step. Returns (B, 2*NV, 3*NV) — column
        blocks [d/dq | d/dqd | d/du] in tangent space."""
        import numpy as np
        import jax
        import jax.numpy as jnp
        from .._handle import _integrator_code
        target = _register_method_target(
            self._so_path, self._cache_key,
            "integrator_gradient", "grid_rbd_jax_integrator_gradient")
        (q, qd, u), B = self._prep_2d("integrator_gradient", q, qd, u)
        nv = self.num_vel
        out_type = jax.ShapeDtypeStruct((B, 2 * nv * 3 * nv), jnp.float32)
        flat = jax.ffi.ffi_call(target, out_type)(
            q, qd, u, dt=np.float32(dt), it=np.int64(_integrator_code(integrator_type)),
            gravity=np.float32(gravity))
        # h_dAB is (2*NV x 3*NV) column-major per timestep; recover row-major.
        return flat.reshape(B, 3 * nv, 2 * nv).transpose(0, 2, 1)

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

    def quadratic_state_cost(self, x, x_des, Q):
        """1/2 sum_i Q_i (x_i - x_des_i)^2 over x=[q;qd]. Returns
        (value (B,), grad (B, NX), hess=diag(Q) (B, NX, NX))."""
        import jax
        target = _register_method_target(
            self._so_path, self._cache_key,
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

    def plant_step(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """x_{k+1} = integrator(x_k, u_k, dt). x (B, NX); u (B, NV). Returns (B, NX)."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from .._handle import _integrator_code
        target = _register_method_target(
            self._so_path, self._cache_key, "plant_step", "grid_rbd_jax_plant_step")
        cast, B = self._prep_plant("plant_step", x=x, u=u)
        nx = self.num_joints + self.num_vel
        out_type = jax.ShapeDtypeStruct((B, nx), jnp.float32)
        return jax.ffi.ffi_call(target, out_type)(
            cast["x"], cast["u"], dt=np.float32(dt),
            it=np.int64(_integrator_code(integrator_type)), gravity=np.float32(gravity))

    def plant_step_gradient(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """[A|B] = d x_{k+1}/d(x,u). x (B, NX); u (B, NV). Returns (B, 2*NV, 3*NV)
        with column blocks [d/dq | d/dqd | d/du] (tangent space)."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from .._handle import _integrator_code
        target = _register_method_target(
            self._so_path, self._cache_key,
            "plant_step_gradient", "grid_rbd_jax_plant_step_gradient")
        cast, B = self._prep_plant("plant_step_gradient", x=x, u=u)
        nv = self.num_vel
        out_type = jax.ShapeDtypeStruct((B, 2 * nv * 3 * nv), jnp.float32)
        flat = jax.ffi.ffi_call(target, out_type)(
            cast["x"], cast["u"], dt=np.float32(dt),
            it=np.int64(_integrator_code(integrator_type)), gravity=np.float32(gravity))
        # (2*NV x 3*NV) column-major per timestep; recover row-major.
        return flat.reshape(B, 3 * nv, 2 * nv).transpose(0, 2, 1)

    def ee_pos_cost(self, q, p_des, W):
        """End-effector position cost (EE 0). q (B, NQ); p_des/W (B, 3). Returns
        (value (B,), grad_x (B, NX), GN hess_x (B, NX, NX))."""
        import jax
        target = _register_method_target(
            self._so_path, self._cache_key, "plant_ee_pos_cost", "grid_rbd_jax_plant_ee_pos_cost")
        cast, B = self._prep_plant("ee_pos_cost", q=q, p_des=p_des, W=W)
        nx = self.num_joints + self.num_vel
        out, grad, hess = jax.ffi.ffi_call(target, self._cost_out_types(B, nx, nx * nx))(
            cast["q"], cast["p_des"], cast["W"])
        return out[:, 0], grad, hess.reshape(B, nx, nx)

    def com_cost(self, q, p_des, W):
        """Center-of-mass tracking cost. q (B, NQ); p_des/W (B, 3). Returns
        (value (B,), grad_x (B, NX), GN hess_x (B, NX, NX))."""
        import jax
        target = _register_method_target(
            self._so_path, self._cache_key, "plant_com_cost", "grid_rbd_jax_plant_com_cost")
        cast, B = self._prep_plant("com_cost", q=q, p_des=p_des, W=W)
        nx = self.num_joints + self.num_vel
        out, grad, hess = jax.ffi.ffi_call(target, self._cost_out_types(B, nx, nx * nx))(
            cast["q"], cast["p_des"], cast["W"])
        return out[:, 0], grad, hess.reshape(B, nx, nx)

    def momentum_cost(self, q, qd, h_des, W):
        """Centroidal-momentum tracking cost. q (B, NQ); qd (B, NV); h_des/W (B, 6).
        Returns (value (B,), grad_x (B, NX), GN hess_x (B, NX, NX))."""
        import jax
        target = _register_method_target(
            self._so_path, self._cache_key, "plant_momentum_cost", "grid_rbd_jax_plant_momentum_cost")
        cast, B = self._prep_plant("momentum_cost", q=q, qd=qd, h_des=h_des, W=W)
        nx = self.num_joints + self.num_vel
        out, grad, hess = jax.ffi.ffi_call(target, self._cost_out_types(B, nx, nx * nx))(
            cast["q"], cast["qd"], cast["h_des"], cast["W"])
        return out[:, 0], grad, hess.reshape(B, nx, nx)


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
) -> JaxRobotHandle:
    """Register a robot for use with JAX.

    Compiles + caches the same per-robot ``.so`` that
    :py:func:`grid_rbd.register_robot` uses (cache hit if already
    compiled). Additionally registers JAX FFI targets so the methods
    are callable inside ``jax.jit``.

    Returns a :py:class:`JaxRobotHandle`.
    """
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
    return JaxRobotHandle(base, entry["cache_key"], str(so_path))


def get_robot(
    name: str,
    cache_dir: str | Path | None = None,
) -> JaxRobotHandle:
    """Look up a previously-registered robot. Same cache as
    :py:func:`grid_rbd.get_robot`."""
    base = _grid_rbd.get_robot(name, cache_dir=cache_dir)
    from grid_rbd._cache import default_cache_dir, manifest_lookup, store_dir
    cd = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    entry = manifest_lookup(cd, name)
    so_path = store_dir(cd, entry["cache_key"]) / "robot.so"
    return JaxRobotHandle(base, entry["cache_key"], str(so_path))


__all__ = ["JaxRobotHandle", "register_robot", "get_robot"]

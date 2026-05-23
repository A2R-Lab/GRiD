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
This wrapper follows GRiD's internal convention: `gravity` is the
**magnitude** (positive, default 9.81). The downward direction is
applied inside the kernel. If you're cross-checking against
``RBDReference.rnea(..., GRAVITY=-9.81)``, pass ``gravity=9.81`` here
for matching results.
"""
from __future__ import annotations

from typing import Any

import numpy as np


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


class RobotHandle:
    """Opaque handle to a compiled per-robot GRiD library.

    Created by `grid_rbd.register_robot(...)` and `grid_rbd.get_robot(...)`.
    Don't construct directly; the constructor wires up the pybind11 Runner
    plus the metadata loaded from the cache's meta.json.
    """

    def __init__(self, name: str, so_path: str, meta: dict[str, Any]) -> None:
        from . import _core  # pybind11 extension; built at pip install time

        self._name = name
        self._meta = dict(meta)
        self._runner = _core.Runner(so_path)

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
    def floating_base(self) -> bool:
        return bool(self._meta.get("floating_base", False))

    @property
    def max_batch(self) -> int:
        return self._runner.max_batch

    @property
    def suggested_threads(self) -> int:
        """Codegen-time thread-count hint (DOF-aware, warp-rounded).

        The default block size for kernel launches. Since v2.0 it is a
        recommendation, not an enforced floor — callers can override
        via :py:meth:`set_threads_per_block`.
        """
        return self._runner.suggested_threads

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

        Default: :py:attr:`suggested_threads`.
        """
        self._runner.set_threads_per_block(int(n))

    # ─── algorithms ──────────────────────────────────────────────────────────
    #
    # All methods take 2D float32 arrays of shape (B, num_joints) for the
    # inputs. They return either (B, num_joints) or (B, num_joints, num_joints)
    # depending on the algorithm.

    def rnea(self, q, qd, qdd=None, *, gravity: float = 9.81):
        """Inverse dynamics (RNEA). Returns the bias term c = M·qdd_zero + h − g.

        Currently `qdd` is accepted for API stability but ignored (the
        underlying wrapper uses USE_QDD_FLAG=false). Future v2 will plumb
        through user-provided qdd.
        """
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        qdd_arr = None
        if qdd is not None:
            qdd_arr = np.ascontiguousarray(qdd, dtype=np.float32)
        return self._runner.rnea(q, qd, qdd_arr, gravity)

    def minv(self, q):
        """Direct mass-matrix inverse Minv(q). Returns shape (B, NJ, NJ).

        GRiD's `direct_minv` kernel writes only the lower triangle (upper
        zero); we symmetrize on the host before returning so the matrix
        matches `RBDReference.minv(..., output_dense=True)`. The
        symmetrization is a single numpy op per call — negligible cost.
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        m = self._runner.minv(q)
        # Symmetrize: M = L + L^T - diag(L)  where L is the lower triangle.
        m_full = m + m.swapaxes(-1, -2)
        diag_idx = np.arange(m.shape[-1])
        m_full[:, diag_idx, diag_idx] -= np.diagonal(m, axis1=-2, axis2=-1)
        return m_full

    def forward_dynamics(self, q, qd, u, *, gravity: float = 9.81):
        """Forward dynamics qdd = M⁻¹·(τ − c). Returns shape (B, NJ)."""
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        u  = np.ascontiguousarray(u,  dtype=np.float32)
        return self._runner.forward_dynamics(q, qd, u, gravity)

    def aba(self, q, qd, u, *, gravity: float = 9.81):
        """Recursive forward dynamics via Articulated Body Algorithm.
        Returns shape (B, NJ). Alternative to forward_dynamics() with the
        same output but a different implementation."""
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        u  = np.ascontiguousarray(u,  dtype=np.float32)
        return self._runner.aba(q, qd, u, gravity)

    def crba(self, q, *, gravity: float = 9.81):
        """Joint-space mass matrix M(q) via Composite Rigid Body Algorithm.
        Returns shape (B, NJ, NJ). Pass `gravity` only because the host
        wrapper takes it; the result doesn't depend on gravity."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        return self._runner.crba(q, gravity)

    def end_effector_pose(self, q):
        """End-effector pose [xyz, rpy] per EE. Returns shape (B, 6*NUM_EES).
        For multi-EE robots, reshape to (B, NUM_EES, 6) at the caller side."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        return self._runner.end_effector_pose(q)

    def end_effector_pose_gradient(self, q):
        """End-effector pose Jacobian. Returns shape (B, 6*NUM_EES, NJ).

        GRiD's `h_deePos` is stored column-major as (6, NUM_EES*NJ) per
        timestep. We re-orient to match RBDReference's per-EE
        (6, NJ) convention; for a multi-EE robot the output stacks
        them as (6*NUM_EES, NJ) along axis -2.
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        raw = self._runner.end_effector_pose_gradient(q)
        B = raw.shape[0]
        NEE = self.num_ees
        NJ = self.num_joints
        # raw is (B, 6*NEE, NJ) row-major over the flat buffer. The flat
        # buffer is (6, NEE*NJ) column-major, i.e. raw_flat[r + 6*c] where
        # c = ee*NJ + j, r = output index 0..5. Reinterpret:
        return raw.reshape(B, NEE, NJ, 6).transpose(0, 1, 3, 2).reshape(B, 6 * NEE, NJ)

    def rnea_grad(self, q, qd, qdd=None, *, gravity: float = 9.81):
        """∂c/∂(q, qd). Returns shape (B, NJ, 2*NJ) — concatenated
        [dc_dq | dc_dqd]. Slice with `[..., :NJ]` / `[..., NJ:]`."""
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        qdd_arr = None
        if qdd is not None:
            qdd_arr = np.ascontiguousarray(qdd, dtype=np.float32)
        raw = self._runner.rnea_grad(q, qd, qdd_arr, gravity)
        # GRiD's h_dc_du = [dc_dq (NJ×NJ col-major), dc_dqd (NJ×NJ col-major)]
        # per timestep, total 2*NJ² floats. Reshape to (B, 2, NJ, NJ) col-major,
        # transpose each block, hstack to match RBDReference's (NJ, 2*NJ).
        B = raw.shape[0]
        NJ = self.num_joints
        blocks = raw.reshape(B, 2, NJ, NJ).transpose(0, 1, 3, 2)  # row-major now
        return np.concatenate([blocks[:, 0], blocks[:, 1]], axis=-1)

    def forward_dynamics_grad(self, q, qd, u, *, gravity: float = 9.81):
        """∂qdd/∂(q, qd). Returns shape (B, NJ, 2*NJ)."""
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        u  = np.ascontiguousarray(u,  dtype=np.float32)
        raw = self._runner.forward_dynamics_grad(q, qd, u, gravity)
        # Same layout as h_dc_du: [df_dq, df_dqd] col-major blocks.
        B = raw.shape[0]
        NJ = self.num_joints
        blocks = raw.reshape(B, 2, NJ, NJ).transpose(0, 1, 3, 2)
        return np.concatenate([blocks[:, 0], blocks[:, 1]], axis=-1)

    def end_effector_pose_hessian(self, q):
        """End-effector pose Hessian ∂²ee/∂q². Returns shape (B, 6*NUM_EES, NJ, NJ)."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        return self._runner.end_effector_pose_hessian(q)

    def idsva_so(self, q, qd, qdd=None, *, gravity: float = 9.81):
        """Second-order inverse dynamics. Returns a tuple of 4 tensors:
        (d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq), each shape (B, NV, NV, NV).

        Uses the codegen-time dispatcher: body-frame for fixed-base,
        world-frame for floating-base.
        """
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        qdd_arr = np.ascontiguousarray(qdd, dtype=np.float32) if qdd is not None else None
        NV = self.num_vel
        flat = self._runner.idsva_so(q, qd, qdd_arr, 4 * NV ** 3, gravity)
        # Slice the 4 NV^3 blocks. Each block is stored as raw column/row
        # depending on the kernel; we return them as (B, NV, NV, NV)
        # without further reshape — callers wanting tensor-axis semantics
        # should consult RBDReference's idsva_so docs.
        B = flat.shape[0]
        return tuple(flat[:, i*NV**3:(i+1)*NV**3].reshape(B, NV, NV, NV) for i in range(4))

    def fdsva_so(self, q, qd, u, *, gravity: float = 9.81):
        """Second-order forward dynamics. Returns shape (B, 4*NV^3) as a flat
        view of the four output tensors; slice [..., i*NV^3:(i+1)*NV^3] for
        each component."""
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        u  = np.ascontiguousarray(u,  dtype=np.float32)
        NV = self.num_vel
        flat = self._runner.fdsva_so(q, qd, u, 4 * NV ** 3, gravity)
        B = flat.shape[0]
        return tuple(flat[:, i*NV**3:(i+1)*NV**3].reshape(B, NV, NV, NV) for i in range(4))

    def integrator(self, q, qd, u, dt, *, integrator_type: str = "euler"):
        """One integration step x_{k+1} = integrator(x_k, u, dt).

        Returns shape (B, NUM_POS + NUM_VEL) — concatenated [q_new, v_new].
        `dt` is the runtime timestep; gravity is the standard 9.81 constant.
        `integrator_type` is one of euler / semi_implicit_euler / midpoint /
        rk3 / rk4."""
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        u  = np.ascontiguousarray(u,  dtype=np.float32)
        it = _integrator_code(integrator_type)
        return self._runner.integrator(q, qd, u, float(dt), it)

    def integrator_gradient(self, q, qd, u, dt, *, integrator_type: str = "euler"):
        """Gradient of the integrator step. Returns shape (B, 2*NV, 3*NV) —
        column blocks [d/dq | d/dqd | d/du] in tangent space.

        `dt` is the runtime timestep; gravity is the standard 9.81 constant."""
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        u  = np.ascontiguousarray(u,  dtype=np.float32)
        it = _integrator_code(integrator_type)
        raw = self._runner.integrator_gradient(q, qd, u, float(dt), it)
        # h_dAB is (2*NV x 3*NV) column-major per timestep; recover row-major.
        B = raw.shape[0]
        NV = self.num_vel
        return raw.reshape(B, 3 * NV, 2 * NV).transpose(0, 2, 1)

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

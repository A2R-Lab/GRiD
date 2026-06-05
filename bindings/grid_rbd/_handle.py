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
    # All methods take 2D float32 arrays of shape (B, num_joints) for the
    # inputs. They return either (B, num_joints) or (B, num_joints, num_joints)
    # depending on the algorithm.

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
        fe = np.ascontiguousarray(f_ext, dtype=np.float32)
        nb = self.num_bodies
        if fe.ndim != 2 or fe.shape[1] != 6 * nb:
            raise ValueError(
                f"f_ext must be (batch, 6*num_bodies) = (batch, {6 * nb}); "
                f"got shape {fe.shape}. Layout is body-major, each body a "
                f"length-6 [angular; linear] wrench in the body's local frame."
            )
        return fe

    def inverse_dynamics(self, q, qd, qdd=None, *, gravity: float = -9.81, f_ext=None):
        """Inverse dynamics (RNEA). Returns the bias term c = M·qdd_zero + h − g.

        Currently `qdd` is accepted for API stability but ignored (the
        underlying wrapper uses USE_QDD_FLAG=false). Future v2 will plumb
        through user-provided qdd.

        ``f_ext`` (optional): per-body external forces, shape
        ``(B, 6*num_bodies)``, body-major, each ``[angular; linear]`` in the
        body's local frame (subtracted from the per-body force, matching
        ``RBDReference.inverse_dynamics(..., f_ext=...)``). Default None ⇒ no external force.
        """
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        qdd_arr = None
        if qdd is not None:
            qdd_arr = np.ascontiguousarray(qdd, dtype=np.float32)
        return self._runner.inverse_dynamics(q, qd, qdd_arr, gravity, self._prep_f_ext(f_ext))

    def minv(self, q):
        """Direct mass-matrix inverse Minv(q). Returns shape (B, NJ, NJ).

        GRiD's `minv` kernel writes only the lower triangle (upper
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

    def forward_dynamics(self, q, qd, u, *, gravity: float = -9.81, f_ext=None):
        """Forward dynamics qdd = M⁻¹·(τ − c). Returns shape (B, NJ).

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``,
        body-major, ``[angular; linear]`` local-frame (see :py:meth:`inverse_dynamics`)."""
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        u  = np.ascontiguousarray(u,  dtype=np.float32)
        return self._runner.forward_dynamics(q, qd, u, gravity, self._prep_f_ext(f_ext))

    def aba(self, q, qd, u, *, gravity: float = -9.81, f_ext=None):
        """Recursive forward dynamics via Articulated Body Algorithm.
        Returns shape (B, NJ). Alternative to forward_dynamics() with the
        same output but a different implementation.

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``,
        body-major, ``[angular; linear]`` local-frame (see :py:meth:`inverse_dynamics`)."""
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        u  = np.ascontiguousarray(u,  dtype=np.float32)
        return self._runner.aba(q, qd, u, gravity, self._prep_f_ext(f_ext))

    def crba(self, q, *, gravity: float = -9.81):
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

    def fk_batched(self, q, *, use_warp: bool = False):
        """Large-batch forward kinematics, one block (thread variant) or warp
        (warp variant) per sample.

        Input  q:     (B, NUM_POS)  joint positions (batch-major).
        Output pose7: (B, 7) = [tx, ty, tz, qw, qx, qy, qz] for the leaf EE
        frame, where the last four are the unit quaternion (w, x, y, z).

        `use_warp=True` runs the warp-cooperative per-sample inner; both
        variants return identical poses. Only available for fixed-base,
        non-mimic robots (raises otherwise)."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        return self._runner.fk_batched(q, use_warp)

    def end_effector_pose_gradient(self, q):
        """End-effector pose Jacobian d/dv (TANGENT, pinocchio convention).

        Returns shape (B, 6*NUM_EES, NV). Floating-base produces the
        spatial Jacobian (omega; v) base block, not the older non-standard
        quaternion-derivative columns. Fixed-base shape unchanged (NV == NJ).

        GRiD's `h_end_effector_pose_gradient` is stored column-major as (6, NUM_EES*NV) per
        timestep; we re-orient to (6*NUM_EES, NV) per timestep.
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        raw = self._runner.end_effector_pose_gradient(q)
        B = raw.shape[0]
        NEE = self.num_ees
        NV = self.num_vel
        return raw.reshape(B, NEE, NV, 6).transpose(0, 1, 3, 2).reshape(B, 6 * NEE, NV)

    def inverse_dynamics_gradient(self, q, qd, qdd=None, *, gravity: float = -9.81, f_ext=None):
        """∂c/∂(q, qd). Returns shape (B, NJ, 2*NJ) — concatenated
        [dc_dq | dc_dqd]. Slice with `[..., :NJ]` / `[..., NJ:]`.

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``.
        f_ext enters RNEA affinely, so for a CONSTANT f_ext the Jacobian
        ∂c/∂(q,qd) is unchanged; the kwarg is for consistency with inverse_dynamics()."""
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        qdd_arr = None
        if qdd is not None:
            qdd_arr = np.ascontiguousarray(qdd, dtype=np.float32)
        raw = self._runner.inverse_dynamics_gradient(q, qd, qdd_arr, gravity, self._prep_f_ext(f_ext))
        # GRiD's h_dc_du = [dc_dq (NJ×NJ col-major), dc_dqd (NJ×NJ col-major)]
        # per timestep, total 2*NJ² floats. Reshape to (B, 2, NJ, NJ) col-major,
        # transpose each block, hstack to match RBDReference's (NJ, 2*NJ).
        B = raw.shape[0]
        NJ = self.num_joints
        blocks = raw.reshape(B, 2, NJ, NJ).transpose(0, 1, 3, 2)  # row-major now
        return np.concatenate([blocks[:, 0], blocks[:, 1]], axis=-1)

    def forward_dynamics_gradient(self, q, qd, u, *, gravity: float = -9.81, f_ext=None):
        """∂qdd/∂(q, qd). Returns shape (B, NJ, 2*NJ).

        ``f_ext`` (optional): per-body external forces ``(B, 6*num_bodies)``;
        affine in f_ext so a constant f_ext leaves this Jacobian unchanged."""
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        u  = np.ascontiguousarray(u,  dtype=np.float32)
        raw = self._runner.forward_dynamics_gradient(q, qd, u, gravity, self._prep_f_ext(f_ext))
        # Same layout as h_dc_du: [df_dq, df_dqd] col-major blocks.
        B = raw.shape[0]
        NJ = self.num_joints
        blocks = raw.reshape(B, 2, NJ, NJ).transpose(0, 1, 3, 2)
        return np.concatenate([blocks[:, 0], blocks[:, 1]], axis=-1)

    def end_effector_pose_hessian(self, q):
        """End-effector pose Hessian ∂²(pose)/∂v² (tangent-space, pinocchio convention).
        Returns shape (B, 6*NUM_EES, NV, NV). For fixed-base NV == NJ; for
        floating-base the (NV, NV) block indexes spatial twist components."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        return self._runner.end_effector_pose_hessian(q)

    def idsva_so(self, q, qd, qdd=None, *, gravity: float = -9.81):
        """Second-order inverse dynamics. Returns a tuple of 4 tensors:
        (d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq), each shape (B, NV, NV, NV).

        Uses the codegen-time dispatcher: body-frame for fixed-base,
        world-frame for floating-base.
        """
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        # qdd is packed into the device acceleration slot; pass explicit zeros for
        # the default (qdd=None ⇒ zero acceleration) so the result never depends on
        # a stale device buffer from a previous call.
        qdd_in = qdd if qdd is not None else np.zeros_like(q)
        qdd_arr = np.ascontiguousarray(qdd_in, dtype=np.float32)
        NV = self.num_vel
        flat = self._runner.idsva_so(q, qd, qdd_arr, 4 * NV ** 3, gravity)
        # Slice the 4 NV^3 blocks. Each block is stored as raw column/row
        # depending on the kernel; we return them as (B, NV, NV, NV)
        # without further reshape — callers wanting tensor-axis semantics
        # should consult RBDReference's idsva_so docs.
        B = flat.shape[0]
        return tuple(flat[:, i*NV**3:(i+1)*NV**3].reshape(B, NV, NV, NV) for i in range(4))

    def fdsva_so(self, q, qd, u, *, gravity: float = -9.81):
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

    def integrator(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """One integration step x_{k+1} = integrator(x_k, u, dt).

        Returns shape (B, NUM_POS + NUM_VEL) — concatenated [q_new, v_new].
        `dt` is the runtime timestep; gravity is the signed gravitational acceleration (default -9.81).
        `integrator_type` is one of euler / semi_implicit_euler / midpoint /
        rk3 / rk4."""
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        u  = np.ascontiguousarray(u,  dtype=np.float32)
        it = _integrator_code(integrator_type)
        return self._runner.integrator(q, qd, u, float(dt), it, gravity=float(gravity))

    def integrator_gradient(self, q, qd, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """Gradient of the integrator step. Returns shape (B, 2*NV, 3*NV) —
        column blocks [d/dq | d/dqd | d/du] in tangent space.

        `dt` is the runtime timestep; gravity is the signed gravitational acceleration (default -9.81)."""
        q  = np.ascontiguousarray(q,  dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        u  = np.ascontiguousarray(u,  dtype=np.float32)
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
        x = np.ascontiguousarray(x, dtype=np.float32)
        x_des = np.ascontiguousarray(x_des, dtype=np.float32)
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        return self._runner.quadratic_state_cost(x, x_des, Q)

    def quadratic_input_cost(self, u, u_des, R):
        """1/2 * sum_i R_i (u_i - u_des_i)^2 over the input u (size NUM_VEL).

        u / u_des / R are (B, NUM_VEL). Returns:
          value (B,), grad (B, NV), hess = diag(R) (B, NV, NV).
        """
        u = np.ascontiguousarray(u, dtype=np.float32)
        u_des = np.ascontiguousarray(u_des, dtype=np.float32)
        R = np.ascontiguousarray(R, dtype=np.float32)
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
        q = np.ascontiguousarray(q, dtype=np.float32)
        p_des = np.ascontiguousarray(p_des, dtype=np.float32)
        W = np.ascontiguousarray(W, dtype=np.float32)
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
        var = np.ascontiguousarray(var, dtype=np.float32)
        lower = np.ascontiguousarray(lower, dtype=np.float32)
        upper = np.ascontiguousarray(upper, dtype=np.float32)
        return getattr(self._runner, method)(var, lower, upper, float(mu))

    def plant_step(self, x, u, dt, *, integrator_type: str = "euler", gravity: float = -9.81):
        """x_{k+1} = integrator(x_k, u_k, dt). Thin wrapper over grid::integrator.

        x is (B, NUM_POS + NUM_VEL); u is (B, NUM_VEL). Returns (B, NX).
        `integrator_type` is one of euler / semi_implicit_euler / midpoint /
        rk3 / rk4 (same codes as :py:meth:`integrator`).
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        u = np.ascontiguousarray(u, dtype=np.float32)
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
        x = np.ascontiguousarray(x, dtype=np.float32)
        u = np.ascontiguousarray(u, dtype=np.float32)
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
        x = np.ascontiguousarray(x, dtype=np.float32)
        u = np.ascontiguousarray(u, dtype=np.float32)
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
        q = np.ascontiguousarray(q, dtype=np.float32)
        p_des = np.ascontiguousarray(p_des, dtype=np.float32)
        W = np.ascontiguousarray(W, dtype=np.float32)
        return self._runner.com_cost(q, p_des, W)

    def momentum_cost(self, q, qd, h_des, W):
        """Centroidal-momentum tracking cost over the 6 momentum components.

        q is (B, NUM_POS); qd is (B, NUM_VEL); h_des / W are (B, 6). Returns:
          value (B,), grad_x (B, NX) = [0; A^T (W·r)], GN hess_x (B, NX, NX)
          with the bottom-right NV×NV qd-block = A^T diag(W) A.
        Matches ``RBDReference.momentum_cost(q, qd, h_des, W)``.
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        h_des = np.ascontiguousarray(h_des, dtype=np.float32)
        W = np.ascontiguousarray(W, dtype=np.float32)
        return self._runner.momentum_cost(q, qd, h_des, W)

    # ─── centroidal / energy / general-frame kinematics (F2) ─────────────────
    #
    # Convenience compositions over the grid:: kinematics/dynamics surface,
    # validated against RBDReference's centroidal / energy / frame mixins. All
    # take 2D float32 (B, NUM_JOINTS) inputs. The frame_jacobian family targets a
    # frame fixed at codegen time (the leaf end-effector joint,
    # LOCAL_WORLD_ALIGNED reference frame); a runtime frame/reference_frame kwarg
    # is not yet supported on the GPU surface (the host/kernel bake the target).

    def com(self, q):
        """Center-of-mass world position p_com (3,) and CoM Jacobian J_com.

        Returns ``(p_com, J_com)`` where ``p_com`` is ``(B, 3)`` and ``J_com``
        is ``(B, 3, NV)`` = ``d(p_com)/dv``. Matches ``RBDReference.com(q)``
        (= ``p_com``) and ``RBDReference.jacobian_com(q)`` (= ``J_com``).
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        raw = self._runner.com(q)  # (B, 3 + 3*NV): [p_com(3); J_com(3 x NV col-major)]
        B = raw.shape[0]
        NV = self.num_vel
        p_com = raw[:, :3]
        # J_com stored column-major (3 x NV): J[r + 3*c]; recover (B, 3, NV).
        j_com = raw[:, 3:].reshape(B, NV, 3).transpose(0, 2, 1)
        return p_com, j_com

    def ccrba(self, q, qd):
        """Centroidal momentum matrix A (6 x NV) and momentum h = A·qd (6,).

        Returns ``(A, h)`` where ``A`` is ``(B, 6, NV)`` and ``h`` is ``(B, 6)``,
        in the Pinocchio convention (``[linear; angular]`` at the CoM, world
        aligned). Matches ``RBDReference.ccrba(q, qd)``.
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        raw = self._runner.ccrba(q, qd)  # (B, 6*NV + 6): [A(6 x NV col-major); h(6)]
        B = raw.shape[0]
        NV = self.num_vel
        A = raw[:, : 6 * NV].reshape(B, NV, 6).transpose(0, 2, 1)
        h = raw[:, 6 * NV:]
        return A, h

    def energy(self, q, qd, *, gravity: float = -9.81):
        """Kinetic / potential / mechanical energy. Returns ``(B, 3)`` =
        ``[KE, PE, KE+PE]``. Matches ``RBDReference.kinetic_energy`` /
        ``potential_energy`` / ``mechanical_energy`` (PE uses ``gravity``).
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        return self._runner.energy(q, qd, float(gravity))

    def generalized_gravity(self, q, *, gravity: float = -9.81):
        """Generalized gravity torque g(q) = RNEA(q, 0, 0). Returns ``(B, NV)``.
        Matches ``RBDReference.generalized_gravity(q, GRAVITY=gravity)``."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        return self._runner.generalized_gravity(q, float(gravity))

    def nonlinear_effects(self, q, qd, *, gravity: float = -9.81):
        """Nonlinear (bias) effects c(q,qd) = RNEA(q, qd, 0) = C(q,qd)·qd + g(q).
        Returns ``(B, NV)``. Matches ``RBDReference.nonlinear_effects(q, qd,
        GRAVITY=gravity)``."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        return self._runner.nonlinear_effects(q, qd, float(gravity))

    def frame_jacobian(self, q, *, target_jid=None, reference_frame=None):
        """Geometric Jacobian (6 x NV, ``[linear; angular]``) of a frame.
        Returns ``(B, 6, NV)``. Matches ``RBDReference.frame_jacobian(q,
        frame_name, reference_frame)``.

        ``target_jid`` selects the frame's joint id (default: the leaf
        end-effector joint baked at codegen time). ``reference_frame`` is
        ``'LOCAL'`` (0), ``'WORLD'`` (1), or ``'LOCAL_WORLD_ALIGNED'`` (2, the
        default), or the equivalent int. Both are now RUNTIME parameters of the
        GPU surface.
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        tj, rf = _frame_args(target_jid, reference_frame)
        raw = self._runner.frame_jacobian(q, tj, rf)  # (B, 6*NV) col-major: J[r + 6*c]
        B = raw.shape[0]
        NV = self.num_vel
        return raw.reshape(B, NV, 6).transpose(0, 2, 1)

    def frame_jacobian_dot(self, q, qd, *, target_jid=None, reference_frame=None):
        """Time derivative Jdot of :py:meth:`frame_jacobian` along v = qd
        (6 x NV, ``[linear; angular]``). Returns ``(B, 6, NV)``. Matches
        ``RBDReference.frame_jacobian_dot(q, qd, frame_name, reference_frame)``.

        ``target_jid`` / ``reference_frame`` are RUNTIME parameters (default:
        leaf-EE joint / ``LOCAL_WORLD_ALIGNED``); see :py:meth:`frame_jacobian`.
        """
        q = np.ascontiguousarray(q, dtype=np.float32)
        qd = np.ascontiguousarray(qd, dtype=np.float32)
        tj, rf = _frame_args(target_jid, reference_frame)
        raw = self._runner.frame_jacobian_dot(q, qd, tj, rf)  # (B, 6*NV) col-major
        B = raw.shape[0]
        NV = self.num_vel
        return raw.reshape(B, NV, 6).transpose(0, 2, 1)

    def osc_inertia(self, q):
        """Operational-space (task) inertia Lambda = (J·M⁻¹·Jᵀ)⁻¹ (6 x 6) for
        the leaf-EE frame (LWA). Returns ``(B, 6, 6)``. Matches
        ``RBDReference.osc_inertia(q)``."""
        q = np.ascontiguousarray(q, dtype=np.float32)
        raw = self._runner.osc_inertia(q)  # (B, 36) row/col-major (symmetric)
        return raw.reshape(raw.shape[0], 6, 6)

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

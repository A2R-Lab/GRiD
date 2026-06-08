"""MuJoCo / mjx output-convention transforms (binding-side mirror).

A self-contained copy of the floating-base pin<->mjx value transforms validated
in ``RBDReference/equivalents/mujoco_convention.py`` (which the binding cannot
import). Used by ``RobotHandle`` when ``output_convention="mujoco"``.

GRiD is natively pinocchio-convention. For a free-floating base MuJoCo differs in
two ways: the free-flyer quaternion is wxyz (pin/GRiD xyzw), and the free-joint
velocity is ``[v_lin GLOBAL ; omega LOCAL]`` (pin spatial twist is fully LOCAL),
related by ``G(q) = blockdiag(R, I_3)`` on the leading 6 tangent DOF. ``G`` is
orthogonal, so ``G^{-1}=G^T`` and ``G^{-T}=G``. Acceleration is NOT a plain
rotation: ``a_mjx_lin = R(a_pin_lin + omega x v_local)``.

Every function is a no-op for a fixed base (``floating=False``) — the mjx flag is
a provable, byte-identical no-op there. Batched: leading axis is the batch.

Tangent layout assumed (matches the binding's floating output ordering): the
free-flyer root occupies q ``[pos(3), quat_xyzw(4)]`` and tangent
``[v_lin(3), omega(3)]`` at the front; everything after is internal joints.
"""
from __future__ import annotations

import numpy as np

_POS = slice(0, 3)
_QUAT = slice(3, 7)
_LIN = slice(0, 3)
_ANG = slice(3, 6)


def _R_from_quat_xyzw(q4):
    """Batched 3x3 rotation(s) from xyzw quaternion(s); q4 shape (..., 4)."""
    x, y, z, w = q4[..., 0], q4[..., 1], q4[..., 2], q4[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.empty(q4.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (yy + zz); R[..., 0, 1] = 2 * (xy - wz);     R[..., 0, 2] = 2 * (xz + wy)
    R[..., 1, 0] = 2 * (xy + wz);     R[..., 1, 1] = 1 - 2 * (xx + zz); R[..., 1, 2] = 2 * (yz - wx)
    R[..., 2, 0] = 2 * (xz - wy);     R[..., 2, 1] = 2 * (yz + wx);     R[..., 2, 2] = 1 - 2 * (xx + yy)
    return R


def base_rotation(q):
    """(B,3,3) base rotation(s) from a batched floating-base config ``q`` (B,nq)."""
    return _R_from_quat_xyzw(np.asarray(q, dtype=np.float64)[:, _QUAT])


# --- configuration / input relabels -------------------------------------------

def q_pin_to_mjx(q, floating):
    q = np.array(q, dtype=np.float64, copy=True)
    if floating:
        quat = q[:, _QUAT].copy()
        q[:, 3] = quat[:, 3]; q[:, 4] = quat[:, 0]; q[:, 5] = quat[:, 1]; q[:, 6] = quat[:, 2]
    return q


def q_mjx_to_pin(q, floating):
    q = np.array(q, dtype=np.float64, copy=True)
    if floating:
        wxyz = q[:, _QUAT].copy()
        q[:, 3] = wxyz[:, 1]; q[:, 4] = wxyz[:, 2]; q[:, 5] = wxyz[:, 3]; q[:, 6] = wxyz[:, 0]
    return q


def v_mjx_to_pin(v, R, floating):
    v = np.array(v, dtype=np.float64, copy=True)
    if floating:
        v[:, _LIN] = np.einsum('bji,bj->bi', R, v[:, _LIN])   # R^T @ v_lin
    return v


def v_pin_to_mjx(v, R, floating):
    v = np.array(v, dtype=np.float64, copy=True)
    if floating:
        v[:, _LIN] = np.einsum('bij,bj->bi', R, v[:, _LIN])   # R @ v_lin
    return v


force_mjx_to_pin = v_mjx_to_pin     # covector: G^{-T}=G, so input uses R^T as well


def accel_mjx_to_pin(a, v_pin, R, floating):
    """a_pin_lin = R^T a_mjx_lin - omega x v_local (v_pin already in pin frame)."""
    a = np.array(a, dtype=np.float64, copy=True)
    if floating:
        v_pin = np.asarray(v_pin, dtype=np.float64)
        omega = v_pin[:, _ANG]; v_lin = v_pin[:, _LIN]
        a[:, _LIN] = np.einsum('bji,bj->bi', R, a[:, _LIN]) - np.cross(omega, v_lin)
    return a


def accel_pin_to_mjx(a, v_pin, R, floating):
    """a_mjx_lin = R(a_pin_lin + omega x v_local)."""
    a = np.array(a, dtype=np.float64, copy=True)
    if floating:
        v_pin = np.asarray(v_pin, dtype=np.float64)
        omega = v_pin[:, _ANG]; v_lin = v_pin[:, _LIN]
        a[:, _LIN] = np.einsum('bij,bj->bi', R, a[:, _LIN] + np.cross(omega, v_lin))
    return a


# --- value-output transforms pin->mjx -----------------------------------------

def id_tau_pin_to_mjx(tau, R, floating):
    tau = np.array(tau, dtype=np.float64, copy=True)
    if floating:
        tau[:, _LIN] = np.einsum('bij,bj->bi', R, tau[:, _LIN])
    return tau


def fd_qdd_pin_to_mjx(qdd, qd_pin, R, floating):
    return accel_pin_to_mjx(qdd, qd_pin, R, floating)


def _congruence(M, R, floating):
    """G M G^T for batched M (B,nv,nv); G = I except top-left 3x3 = R."""
    if not floating:
        return np.asarray(M, dtype=np.float64)
    M = np.array(M, dtype=np.float64, copy=True)
    # left-multiply rows 0:3 by R, then right-multiply cols 0:3 by R^T.
    M[:, _LIN, :] = np.einsum('bij,bjk->bik', R, M[:, _LIN, :])
    M[:, :, _LIN] = np.einsum('bik,bjk->bij', M[:, :, _LIN], R)
    return M


mass_matrix_pin_to_mjx = _congruence    # M_mjx = G M G^T (G^{-T}=G)
minv_pin_to_mjx = _congruence           # Minv_mjx = G Minv G^T

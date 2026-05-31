"""grid_plant smoke tests for the `grid-rbd` package (G1 binding layer).

Registers iiwa14 (fixed-base), exercises the grid_plant surface exposed on
the RobotHandle (plant_step, quadratic state/input cost, ee_pos_cost, and the
joint position/velocity/torque log-barriers), and asserts numerical agreement
with the `RBDReference._PlantMixin` numpy reference at float32 precision.

Skip conditions mirror the other python-wrapper smokes (grid_rbd importable,
nvcc on PATH, iiwa14 URDF fixture present).

Run with:
    pytest test/python_wrappers/test_iiwa14_plant_smoke.py -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed (pip install python/)")

# In-repo URDF (always present) so this smoke is self-contained.
_URDF = _REPO_ROOT / "robot_assets" / "iiwa14.urdf"
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF fixture not present at {_URDF}", allow_module_level=True)
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)


pytestmark = pytest.mark.python_wrappers

_TOL = 5e-3


@pytest.fixture(scope="module")
def handle():
    return _grid_rbd.register_robot(
        name="iiwa14_plant_smoke",
        urdf_path=str(_URDF),
        floating_base=False,
        max_batch_size=8,
    )


@pytest.fixture(scope="module")
def ref():
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    return RBDReference(URDFParser().parse(str(_URDF), floating_base=False))


@pytest.fixture(scope="module")
def samples(handle):
    rng = np.random.default_rng(0)
    NQ, NV = handle.num_joints, handle.num_vel
    B = 4
    q = rng.standard_normal((B, NQ)).astype(np.float32)
    qd = rng.standard_normal((B, NV)).astype(np.float32)
    u = rng.standard_normal((B, NV)).astype(np.float32)
    return {"q": q, "qd": qd, "u": u, "x": np.concatenate([q, qd], axis=1).astype(np.float32), "B": B}


def _rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-3)))


def test_quadratic_state_cost(handle, ref, samples):
    x, B = samples["x"], samples["B"]
    NX = handle.num_joints + handle.num_vel
    rng = np.random.default_rng(1)
    x_des = rng.standard_normal((B, NX)).astype(np.float32)
    Q = (np.abs(rng.standard_normal((B, NX))) + 0.5).astype(np.float32)
    val, grad, hess = handle.quadratic_state_cost(x, x_des, Q)
    for b in range(B):
        rv, rg, rh = ref.quadratic_state_cost(x[b], x_des[b], Q[b])
        assert abs(val[b] - rv) < _TOL
        assert _rel(grad[b], rg) < _TOL
        assert _rel(hess[b], rh) < _TOL


def test_quadratic_input_cost(handle, ref, samples):
    u, B = samples["u"], samples["B"]
    NV = handle.num_vel
    rng = np.random.default_rng(2)
    u_des = rng.standard_normal((B, NV)).astype(np.float32)
    R = (np.abs(rng.standard_normal((B, NV))) + 0.5).astype(np.float32)
    val, grad, hess = handle.quadratic_input_cost(u, u_des, R)
    for b in range(B):
        rv, rg, rh = ref.quadratic_input_cost(u[b], u_des[b], R[b])
        assert abs(val[b] - rv) < _TOL
        assert _rel(grad[b], rg) < _TOL
        assert _rel(hess[b], rh) < _TOL


def test_plant_step(handle, ref, samples):
    x, u, B = samples["x"], samples["u"], samples["B"]
    NQ = handle.num_joints
    dt = 0.01
    out = handle.plant_step(x, u, dt, integrator_type="euler")
    for b in range(B):
        r = ref.plant_step(x[b, :NQ], x[b, NQ:], u[b], dt, integrator_type="euler")
        assert _rel(out[b], r) < _TOL


def test_ee_pos_cost(handle, ref, samples):
    q, B = samples["q"], samples["B"]
    rng = np.random.default_rng(3)
    p_des = rng.standard_normal((B, 3)).astype(np.float32)
    W = (np.abs(rng.standard_normal((B, 3))) + 0.5).astype(np.float32)
    val, grad, hess = handle.ee_pos_cost(q, p_des, W)
    for b in range(B):
        rv, rg, rh = ref.ee_pos_cost(q[b], p_des[b], W[b], ee=0)
        assert abs(val[b] - rv) < _TOL
        assert _rel(grad[b], rg) < _TOL
        assert _rel(hess[b], rh) < _TOL


def test_joint_position_barrier(handle, ref, samples):
    q, B = samples["q"], samples["B"]
    NQ = handle.num_joints
    lo = (q.min(0, keepdims=True) - 1.0).repeat(B, 0).astype(np.float32)
    hi = (q.max(0, keepdims=True) + 1.0).repeat(B, 0).astype(np.float32)
    mu = 0.1
    val, grad, hd = handle.joint_position_barrier(q, lo, hi, mu)
    for b in range(B):
        rv, rg, rh = ref.joint_position_barrier(q[b], lo[b], hi[b], mu)
        assert abs(val[b] - rv) < _TOL
        assert _rel(grad[b], rg) < _TOL
        assert _rel(hd[b], rh) < _TOL


def test_joint_torque_barrier_inf_bound(handle, ref, samples):
    """One +inf bound must contribute exactly zero (isfinite guard)."""
    u, B = samples["u"], samples["B"]
    NV = handle.num_vel
    lo = (-3.0 * np.ones((B, NV))).astype(np.float32)
    hi = (3.0 * np.ones((B, NV))).astype(np.float32)
    hi[:, 0] = np.inf
    mu = 0.1
    val, grad, hd = handle.joint_torque_barrier(u, lo, hi, mu)
    for b in range(B):
        rv, rg, rh = ref.joint_torque_barrier(u[b], lo[b], hi[b], mu)
        assert abs(val[b] - rv) < _TOL
        assert _rel(grad[b], rg) < _TOL
        assert _rel(hd[b], rh) < _TOL

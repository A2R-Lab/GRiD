"""Python-wrapper smoke tests for the `grid-rbd` package.

Registers iiwa14 (fixed-base) at session scope, exercises every bound
method, and asserts numerical agreement with `RBDReference` at float32
precision. The session-scoped registration takes ~30-60s for the
first run; subsequent runs hit the cache and start in <1s.

Skip conditions:
  * `grid_rbd` not importable (pip install python/ skipped).
  * `nvcc` not on PATH (would fail at register time anyway).
  * iiwa14 URDF fixture not present.

Run with:
    pytest test/python_wrappers/ -m python_wrappers -v
or as part of the full suite:
    pytest -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


# Repo root is parent of `test/`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


# ─── skip preconditions ─────────────────────────────────────────────────────

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed (pip install python/)")

_URDF = (
    Path.home()
    / ".cache/robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF fixture not present at {_URDF}", allow_module_level=True)

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)


pytestmark = pytest.mark.python_wrappers


_TOL = 5e-3   # float32 vs float64 cross-precision; some algos drift ~1e-4


# ─── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def handle():
    return _grid_rbd.register_robot(
        name="iiwa14_pytest_smoke",
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
    NJ = handle.num_joints
    B = 4
    return {
        "q":  rng.standard_normal((B, NJ)).astype(np.float32),
        "qd": rng.standard_normal((B, NJ)).astype(np.float32),
        "u":  rng.standard_normal((B, NJ)).astype(np.float32),
    }


def _max_err(out_grid, out_ref):
    return float(np.max(np.abs(out_grid - out_ref)))


# ─── tests ──────────────────────────────────────────────────────────────────


def test_metadata(handle):
    assert handle.num_joints == 7
    assert handle.num_vel == 7
    assert handle.num_ees == 1
    assert handle.floating_base is False
    assert handle.max_batch == 8


def test_rnea(handle, ref, samples):
    grid = handle.rnea(samples["q"], samples["qd"])
    for i, (q, qd) in enumerate(zip(samples["q"], samples["qd"])):
        c_ref, *_ = ref.rnea(q.astype(np.float64), qd.astype(np.float64), GRAVITY=-9.81)
        assert _max_err(grid[i], c_ref) < _TOL


def test_minv(handle, ref, samples):
    grid = handle.minv(samples["q"])
    for i, q in enumerate(samples["q"]):
        assert _max_err(grid[i], ref.minv(q.astype(np.float64))) < _TOL


def test_forward_dynamics(handle, ref, samples):
    grid = handle.forward_dynamics(samples["q"], samples["qd"], samples["u"])
    for i, (q, qd, u) in enumerate(zip(samples["q"], samples["qd"], samples["u"])):
        ref_qdd = ref.forward_dynamics(q.astype(np.float64), qd.astype(np.float64), u.astype(np.float64))
        assert _max_err(grid[i], ref_qdd) < _TOL


def test_aba(handle, ref, samples):
    grid = handle.aba(samples["q"], samples["qd"], samples["u"])
    for i, (q, qd, u) in enumerate(zip(samples["q"], samples["qd"], samples["u"])):
        ref_qdd = ref.aba(q.astype(np.float64), qd.astype(np.float64), u.astype(np.float64), GRAVITY=-9.81)
        assert _max_err(grid[i], ref_qdd) < _TOL


def test_crba(handle, ref, samples):
    grid = handle.crba(samples["q"])
    for i, q in enumerate(samples["q"]):
        assert _max_err(grid[i], ref.crba(q.astype(np.float64))) < _TOL


def test_end_effector_pose(handle, ref, samples):
    grid = handle.end_effector_pose(samples["q"])
    for i, q in enumerate(samples["q"]):
        ee_ref = ref.end_effector_pose(q.astype(np.float64))[0].flatten()
        assert _max_err(grid[i][: 6 * handle.num_ees], ee_ref) < _TOL


def test_end_effector_pose_gradient(handle, ref, samples):
    grid = handle.end_effector_pose_gradient(samples["q"])
    for i, q in enumerate(samples["q"]):
        dee_ref = ref.end_effector_pose_gradient(q.astype(np.float64))[0]
        assert _max_err(grid[i], dee_ref) < _TOL


def test_rnea_grad(handle, ref, samples):
    grid = handle.rnea_grad(samples["q"], samples["qd"])
    for i, (q, qd) in enumerate(zip(samples["q"], samples["qd"])):
        dc_ref = ref.rnea_grad(q.astype(np.float64), qd.astype(np.float64), GRAVITY=-9.81)
        assert _max_err(grid[i], dc_ref) < _TOL


def test_forward_dynamics_grad(handle, ref, samples):
    grid = handle.forward_dynamics_grad(samples["q"], samples["qd"], samples["u"])
    NJ = handle.num_joints
    for i, (q, qd, u) in enumerate(zip(samples["q"], samples["qd"], samples["u"])):
        dq, dqd = ref.forward_dynamics_grad(q.astype(np.float64), qd.astype(np.float64), u.astype(np.float64))
        assert _max_err(grid[i][:, :NJ], dq)  < _TOL
        assert _max_err(grid[i][:, NJ:], dqd) < _TOL


def test_register_idempotent(handle):
    """Re-registering the same robot reuses the cache (cache hit ⇒ fast)."""
    import time
    t0 = time.time()
    h2 = _grid_rbd.register_robot(
        name="iiwa14_pytest_smoke",
        urdf_path=str(_URDF),
        floating_base=False,
        max_batch_size=8,
    )
    elapsed = time.time() - t0
    assert h2.num_joints == handle.num_joints
    # Cache hit should be well under a second (no nvcc invocation).
    assert elapsed < 5.0, f"cache hit took {elapsed:.1f}s — should be <1s"


def test_get_robot_roundtrip(handle):
    h2 = _grid_rbd.get_robot("iiwa14_pytest_smoke")
    assert h2.num_joints == handle.num_joints


def test_get_robot_missing_raises():
    with pytest.raises(_grid_rbd.RobotNotRegisteredError):
        _grid_rbd.get_robot("does_not_exist_xyz")


# ─── Phase-C extension: hessian + SO ───────────────────────────────────────

def test_end_effector_pose_hessian_shape(handle, samples):
    """Smoke: hessian returns the right shape. Numerical agreement requires
    a Pinocchio reference; RBDReference's d2ee_pose isn't a 1:1 layout match,
    so we only assert shape + finiteness here.
    """
    d2ee = handle.end_effector_pose_hessian(samples["q"])
    NJ = handle.num_joints
    assert d2ee.shape == (samples["q"].shape[0], 6 * handle.num_ees, NJ, NJ)
    assert np.all(np.isfinite(d2ee))


def test_idsva_so_shape(handle, samples):
    """Smoke: idsva_so returns 4 tensors of shape (B, NV, NV, NV). Numerical
    agreement vs RBDReference's idsva_so is covered by the existing CUDA
    equivalence suite (test_cuda_idsva_so_*)."""
    out = handle.idsva_so(samples["q"], samples["qd"])
    assert isinstance(out, tuple) and len(out) == 4
    NV = handle.num_vel
    for t in out:
        assert t.shape == (samples["q"].shape[0], NV, NV, NV)
        assert np.all(np.isfinite(t))


def test_fdsva_so_shape(handle, samples):
    """Smoke: fdsva_so returns 4 tensors of shape (B, NV, NV, NV)."""
    out = handle.fdsva_so(samples["q"], samples["qd"], samples["u"])
    assert isinstance(out, tuple) and len(out) == 4
    NV = handle.num_vel
    for t in out:
        assert t.shape == (samples["q"].shape[0], NV, NV, NV)
        assert np.all(np.isfinite(t))

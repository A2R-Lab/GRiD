"""Joint viscous damping + Coulomb friction (opt-in) on the grid_rbd handle.

`register_robot(..., use_joint_dynamics=True)` emits the joint-local bias
`tau -= damping*qd + friction*sign(qd)` in the inverse_dynamics / forward_dynamics
/ aba VALUE paths (gated; default-OFF is byte-identical). This validates the opt-in
path against `RBDReference(use_joint_dynamics=True)` (the authoritative oracle —
pinocchio ignores model.damping/friction in its value path), and confirms the
DEFAULT (off) build matches the bare (no-damping) oracle.

Robots: iiwa14 (damping 0.5×7, no friction), fr3 (damping+friction, mimic).
Skips if grid_rbd / nvcc / URDF unavailable.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers
_TOL = 1e-3
_IIWA = _REPO_ROOT / "robot_assets" / "iiwa14.urdf"
_FR3 = _REPO_ROOT / "robot_assets" / "fr3.urdf"
if not (_IIWA.exists() and _FR3.exists()):
    pytest.skip("iiwa14/fr3 URDF not present", allow_module_level=True)


def _oracle(urdf, use_jd):
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    return RBDReference(URDFParser().parse(str(urdf), floating_base=False), use_joint_dynamics=use_jd)


def _samples(nj, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((4, nj)).astype(np.float32),
            rng.standard_normal((4, nj)).astype(np.float32),
            rng.standard_normal((4, nj)).astype(np.float32))


def _max(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


@pytest.fixture(scope="module", params=[("iiwa14", _IIWA), ("fr3", _FR3)], ids=["iiwa14", "fr3"])
def robot(request):
    name, urdf = request.param
    h_on = _grid_rbd.register_robot(name=f"{name}_jd_on_pytest", urdf_path=str(urdf),
                                    floating_base=False, use_joint_dynamics=True, max_batch_size=8)
    h_off = _grid_rbd.register_robot(name=f"{name}_jd_off_pytest", urdf_path=str(urdf),
                                     floating_base=False, max_batch_size=8)
    return h_on, h_off, urdf


def test_damping_friction_value_path(robot):
    """id/fd/aba with use_joint_dynamics=True match RBDReference(use_joint_dynamics=True)."""
    h_on, _, urdf = robot
    nj = h_on.num_joints
    q, qd, u = _samples(nj)
    ref = _oracle(urdf, use_jd=True)
    id_g = h_on.inverse_dynamics(q, qd)
    fd_g = h_on.forward_dynamics(q, qd, u)
    aba_g = h_on.aba(q, qd, u)
    for i, (qi, qdi, ui) in enumerate(zip(q, qd, u)):
        qi, qdi, ui = qi.astype(np.float64), qdi.astype(np.float64), ui.astype(np.float64)
        c_ref, *_ = ref.inverse_dynamics(qi, qdi, GRAVITY=-9.81)
        assert _max(id_g[i], c_ref) < _TOL, f"id[{i}]: {_max(id_g[i], c_ref):.2e}"
        assert _max(fd_g[i], ref.forward_dynamics(qi, qdi, ui)) < _TOL
        assert _max(aba_g[i], ref.aba(qi, qdi, ui, GRAVITY=-9.81)) < _TOL


def test_default_off_matches_bare_oracle_and_differs_from_on(robot):
    """Default (no use_joint_dynamics) matches the BARE oracle, and the damping
    bias is non-trivial (on != off) — proves the flag actually does something."""
    h_on, h_off, urdf = robot
    nj = h_off.num_joints
    q, qd, _ = _samples(nj, seed=1)
    ref_bare = _oracle(urdf, use_jd=False)
    id_off = h_off.inverse_dynamics(q, qd)
    for i, (qi, qdi) in enumerate(zip(q, qd)):
        c_ref, *_ = ref_bare.inverse_dynamics(qi.astype(np.float64), qdi.astype(np.float64), GRAVITY=-9.81)
        assert _max(id_off[i], c_ref) < _TOL
    # on vs off must differ (damping bias present)
    assert _max(h_on.inverse_dynamics(q, qd), id_off) > 1e-2

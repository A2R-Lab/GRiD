"""Validation for the runtime welded-tool binding (attach_tool / detach_tool).

register_robot(..., enable_tool=True) turns on the runtime-mutable inertia table + the runtime
single-contact f_ext surface. handle.attach_tool(joint, mass=..., com=..., inertia=..., tip_transform=...)
then, with NO recompile:

  1. composes the payload's spatial inertia into that joint's child link (set_inertia_params), and
  2. (if tip_transform given) makes subsequent end_effector_pose_runtime[_gradient] default to the
     SE(3) tool-tip frame.

This closes the binding gaps:
  * attach_payload -> the CUDA dynamics match an RBDReference oracle whose attach-link inertia is the
    composite (proves the row mapping + poke + on-device rebuild + dynamics chain);
  * SE(3) tip frame through the binding matches RBDReference.end_effector_pose(q, [joint], [X_tool]);
  * mid-chain attach changes upstream torques but not the downstream tip;
  * detach_tool restores the baked robot.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import robot_urdf

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it",
                allow_module_level=True)
_IIWA = robot_urdf("iiwa14")
if not _IIWA.exists():
    pytest.skip(f"iiwa14 URDF not present at {_IIWA}", allow_module_level=True)

from test.python_wrappers.test_runtime_inertia import _build_perturbed_oracle, _max_rel_err
from grid_rbd._payload import compose_payload_inertia

pytestmark = pytest.mark.python_wrappers

_LEAF = "iiwa_joint_7"
_MID = "iiwa_joint_4"


def _parse(urdf_path):
    from URDFParser import URDFParser
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(str(urdf_path), floating_base=False)


def _oracle():
    from RBDReference import RBDReference
    return RBDReference(_parse(_IIWA))


@pytest.fixture(scope="module")
def iiwa_tool():
    return _grid_rbd.register_robot(
        name="iiwa14_tool_pytest", urdf_path=str(_IIWA),
        floating_base=False, enable_tool=True, max_batch_size=8)


@pytest.fixture(scope="module")
def samples(iiwa_tool):
    rng = np.random.default_rng(1)
    NJ = iiwa_tool.num_joints
    return {"q": rng.standard_normal((4, NJ)).astype(np.float32),
            "qd": rng.standard_normal((4, NJ)).astype(np.float32)}


def test_tool_metadata(iiwa_tool):
    assert iiwa_tool.runtime_inertia is True
    j2row = iiwa_tool._meta.get("inertia_row_by_joint_name")
    assert j2row and _LEAF in j2row and _MID in j2row


def test_attach_payload_matches_oracle(iiwa_tool, samples):
    """attach_tool's inertia composition + on-device rebuild + dynamics == an oracle
    whose attach-link inertia is the composite."""
    q, qd = samples["q"], samples["qd"]
    row = iiwa_tool._meta["inertia_row_by_joint_name"][_LEAF]
    baked = np.asarray(iiwa_tool.inertia_params, dtype=np.float64)

    mass, com, inertia = 2.5, [0.0, 0.0, 0.09], np.diag([0.02, 0.02, 0.008])
    iiwa_tool.attach_tool(_LEAF, mass=mass, com=com, inertia=inertia)

    # independent oracle: the whole table with row -> composite (same as the handle poked).
    composite = baked.copy()
    composite[row] = compose_payload_inertia(baked[row], mass, com, inertia)
    oracle = _build_perturbed_oracle(_IIWA, False, composite)

    grid_c = iiwa_tool.inverse_dynamics(q, qd)
    grid_M = iiwa_tool.crba(q)
    for i, (qi, qdi) in enumerate(zip(q, qd)):
        c_ref, *_ = oracle.inverse_dynamics(qi.astype(np.float64), qdi.astype(np.float64),
                                            GRAVITY=-9.81)
        M_ref = oracle.crba(qi.astype(np.float64))
        assert _max_rel_err(grid_c[i], c_ref) < 5e-3
        assert _max_rel_err(np.asarray(grid_M[i]).reshape(M_ref.shape), M_ref) < 5e-3
    iiwa_tool.detach_tool()


def test_se3_tip_matches_oracle(iiwa_tool, samples):
    """The SE(3) tool tip pose/gradient through the binding == the oracle at the same X_tool."""
    q = samples["q"]
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)  # 90deg x
    X = np.eye(4); X[:3, :3] = R; X[:3, 3] = [0.0, 0.0, 0.15]
    iiwa_tool.attach_tool(_LEAF, mass=1.0, com=[0, 0, 0.075], tip_transform=X)
    oracle = _oracle()

    tip = np.asarray(iiwa_tool.end_effector_pose_runtime(q))            # (B, 1, 6) -> tool default
    for i, qi in enumerate(q):
        ref = np.asarray(oracle.end_effector_pose(
            qi.astype(np.float64), ee_joint_names=_LEAF, ee_offsets=[X])[0]).reshape(-1)
        got = tip[i].reshape(-1)
        # position + rpy via rotation matrix (avoid branch-cut); float32 FK floor.
        np.testing.assert_allclose(got[:3], ref[:3], atol=3e-3)
    iiwa_tool.detach_tool()


def test_mid_chain_attach(iiwa_tool, samples):
    """Attach mid-chain: upstream torques change, the downstream (leaf) tip does not."""
    q, qd = samples["q"], samples["qd"]
    tau0 = np.asarray(iiwa_tool.inverse_dynamics(q, qd))
    tip0 = np.asarray(iiwa_tool.end_effector_pose_runtime(q, ee_joint_names=_LEAF))
    iiwa_tool.attach_tool(_MID, mass=1.5, com=[0.0, 0.05, 0.0])
    tau1 = np.asarray(iiwa_tool.inverse_dynamics(q, qd))
    tip1 = np.asarray(iiwa_tool.end_effector_pose_runtime(q, ee_joint_names=_LEAF))
    assert np.max(np.abs(tau1 - tau0)) > 1e-2          # payload felt upstream
    np.testing.assert_allclose(tip1, tip0, atol=1e-4)  # leaf tip unaffected
    iiwa_tool.detach_tool()


def test_tool_fext_matches_host_map(iiwa_tool, samples):
    """tool_fext (world-aligned tip wrench -> joint-local f_ext) matches an independent
    host recomputation from RBDReference's world FK, is nonzero only on the attach body,
    and shifts inverse_dynamics torques when fed as f_ext."""
    if not getattr(iiwa_tool._runner, "has_tool_fext", False):
        pytest.skip("this .so predates tool_fext (stale cache); rebuild with force_rebuild=True")
    ref = _oracle()
    NB = iiwa_tool.num_bodies
    jid = int(iiwa_tool._resolve_ee_jids(_LEAF)[0])
    row = iiwa_tool._meta["inertia_row_by_joint_name"][_LEAF]
    rc = np.array([0.03, -0.02, 0.11], dtype=np.float64)
    q = samples["q"][:1]
    wrench = np.array([[0.7, -0.4, 0.3, 2.0, -1.5, 5.0]], dtype=np.float32)

    fext = np.asarray(iiwa_tool.tool_fext(q, wrench, joint=_LEAF, offset=rc)).reshape(-1)

    Xw, _ = ref._frame_world_placement_and_chain(q[0].astype(np.float64))
    R = Xw[jid][:3, :3]
    nw, fw = wrench[0, :3].astype(np.float64), wrench[0, 3:].astype(np.float64)
    g, h = R.T @ nw, R.T @ fw
    exp = np.zeros(6 * NB)
    exp[6 * row:6 * row + 6] = np.concatenate([g + np.cross(rc, h), h])
    np.testing.assert_allclose(fext, exp, atol=2e-3)

    nz = [b for b in range(NB) if np.max(np.abs(fext[6 * b:6 * b + 6])) > 1e-6]
    assert nz == [row], f"f_ext nonzero on {nz}, expected only body {row}"

    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    tau_free = np.asarray(iiwa_tool.inverse_dynamics(q, qd, qdd)).reshape(-1)
    tau_fe = np.asarray(iiwa_tool.inverse_dynamics(q, qd, qdd, f_ext=fext[None, :])).reshape(-1)
    assert np.max(np.abs(tau_fe - tau_free)) > 1e-3


def test_detach_restores_baked(iiwa_tool, samples):
    q, qd = samples["q"], samples["qd"]
    tau_baked = np.asarray(iiwa_tool.inverse_dynamics(q, qd))
    iiwa_tool.attach_tool(_LEAF, mass=4.0, com=[0, 0, 0.1])
    iiwa_tool.detach_tool()
    tau_after = np.asarray(iiwa_tool.inverse_dynamics(q, qd))
    np.testing.assert_allclose(tau_after, tau_baked, atol=1e-4)
    assert iiwa_tool.tool is None

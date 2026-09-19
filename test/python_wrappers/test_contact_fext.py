"""Validation for the multi-contact f_ext binding (register_robot(contact_frames=[...])).

register_robot(..., contact_frames=[fixed-joint names]) bakes the f_ext_body
contact family and exposes handle.contact_fext(q, f_c): per registered frame a
world-aligned [n_w; f_w] wrench (moment about the frame origin,
LOCAL_WORLD_ALIGNED) -> joint-local (B, 6*num_bodies) f_ext, ready to pass as
f_ext= to the dynamics ops.

iiwa14 carries TWO fixed frames on the same leaf body (iiwa_joint_ee +
tool0_joint, both children of iiwa_joint_7), so the test also proves the
per-body SUM of two contacts (the baked deterministic per-body fold).
Checks: host-map oracle (RBDReference world FK), zero-wrench -> zero f_ext,
support confined to the contact bodies, and the torques actually shift when
the f_ext is fed to inverse_dynamics.
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

pytestmark = pytest.mark.python_wrappers

_FRAMES = ["iiwa_joint_ee", "tool0_joint"]


def _parse(urdf_path):
    from URDFParser import URDFParser
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(str(urdf_path), floating_base=False)


@pytest.fixture(scope="module")
def iiwa_contact():
    handle = _grid_rbd.register_robot(
        "iiwa14_contact_fext_test", str(_IIWA), floating_base=False,
        contact_frames=_FRAMES,
    )
    yield handle
    handle.close()


def test_contact_fext_matches_host_map(iiwa_contact):
    """contact_fext matches an independent host recomputation from RBDReference's
    world FK; two frames on the same body SUM; support is exactly the contact
    bodies; zero wrenches give exactly zero."""
    from RBDReference import RBDReference
    h = iiwa_contact
    frames = h.contact_frames
    assert [f["name"] for f in frames] == _FRAMES
    NB = h.num_bodies
    NC = len(frames)
    rng = np.random.default_rng(20260917)
    q = rng.uniform(-1.0, 1.0, size=(1, h.num_joints)).astype(np.float32)
    f_c = rng.uniform(-5.0, 5.0, size=(1, 6 * NC)).astype(np.float32)

    fext = np.asarray(h.contact_fext(q, f_c)).reshape(-1)

    ref = RBDReference(_parse(_IIWA))
    Xw, _ = ref._frame_world_placement_and_chain(q[0].astype(np.float64))
    exp = np.zeros(6 * NB)
    for c, fr in enumerate(frames):
        jid, rc = int(fr["jid"]), np.asarray(fr["offset"], dtype=np.float64)
        R = Xw[jid][:3, :3]
        nw = f_c[0, 6 * c:6 * c + 3].astype(np.float64)
        fw = f_c[0, 6 * c + 3:6 * c + 6].astype(np.float64)
        g, hh = R.T @ nw, R.T @ fw
        exp[6 * jid:6 * jid + 6] += np.concatenate([g + np.cross(rc, hh), hh])
    np.testing.assert_allclose(fext, exp, atol=2e-3)

    contact_bodies = sorted({int(fr["jid"]) for fr in frames})
    nz = [b for b in range(NB) if np.max(np.abs(fext[6 * b:6 * b + 6])) > 1e-6]
    assert nz == contact_bodies, f"f_ext nonzero on {nz}, expected {contact_bodies}"

    zero = np.asarray(h.contact_fext(q, np.zeros_like(f_c)))
    assert np.max(np.abs(zero)) == 0.0

    # and the torques actually feel it
    qd = np.zeros((1, h.num_vel), dtype=np.float32)
    qdd = np.zeros((1, h.num_vel), dtype=np.float32)
    tau0 = np.asarray(h.inverse_dynamics(q, qd, qdd))
    tau1 = np.asarray(h.inverse_dynamics(q, qd, qdd, f_ext=fext.reshape(1, -1)))
    assert np.max(np.abs(tau1 - tau0)) > 1e-3


# ---------------------------------------------------------------------------
# Branching FLOATING robot on a DYNAMICS-ONLY subset build (audit W07/W08,
# 2026-09-19). contact_frames= is an opt-in input, not an algorithm: without
# any kinematics algorithm in algorithm_list the contact family must still
# (a) be emitted at all (it used to sit inside the kinematics block),
# (b) get the homogeneous-transform block appended to the XImats table (its
#     XmatsHom loader copies from d_XImats[baseXI_size + ind] — a missing block
#     compiled AND ran, reading past the table: NaN world transform for go2's
#     last BFS joint, and 0*NaN then poisoned that body's row on EVERY call),
# (c) size its launcher smem on its own F_EXT_CONTACT_DYNAMIC_SHARED_MEM_BYTES.
# iiwa14 (serial chain) never exposed (b); go2 does.
# ---------------------------------------------------------------------------
_GO2 = robot_urdf("go2")
_GO2_FEET = ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]


def _parse_floating(urdf_path):
    from URDFParser import URDFParser
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(str(urdf_path), floating_base=True)


@pytest.fixture(scope="module")
def go2_subset_contact():
    if not _GO2.exists():
        pytest.skip(f"go2 URDF not present at {_GO2}")
    handle = _grid_rbd.register_robot(
        "go2_contact_fext_subset_test", str(_GO2), floating_base=True,
        contact_frames=_GO2_FEET,
        algorithm_list=["inverse_dynamics", "forward_dynamics"],   # NO kinematics algorithm
        enable_mujoco_kernels=False,
    )
    yield handle
    handle.close()


def test_contact_fext_floating_dynamics_only_subset_matches_host_map(go2_subset_contact):
    h = go2_subset_contact
    frames = h.contact_frames
    assert [f["name"] for f in frames] == _GO2_FEET
    assert h._runner.num_contact_frames == len(_GO2_FEET)
    NB, NC = h.num_bodies, len(frames)
    rng = np.random.default_rng(20260919)
    q = np.zeros((1, h.num_joints), dtype=np.float32)      # [pos3, quat_xyzw, joints]
    q[0, :3] = rng.uniform(-1.0, 1.0, size=3)
    quat = rng.standard_normal(4)
    q[0, 3:7] = (quat / np.linalg.norm(quat)).astype(np.float32)
    q[0, 7:] = rng.uniform(-1.0, 1.0, size=h.num_joints - 7)
    f_c = rng.uniform(-5.0, 5.0, size=(1, 6 * NC)).astype(np.float32)

    fext = np.asarray(h.contact_fext(q, f_c)).reshape(-1)
    assert np.isfinite(fext).all(), "non-finite f_ext (the missing-XmatsHom-block class)"

    from RBDReference import RBDReference
    ref = RBDReference(_parse_floating(_GO2))
    Xw, _ = ref._frame_world_placement_and_chain(q[0].astype(np.float64))
    exp = np.zeros(6 * NB)
    for c, fr in enumerate(frames):
        jid, rc = int(fr["jid"]), np.asarray(fr["offset"], dtype=np.float64)
        R = Xw[jid][:3, :3]
        nw = f_c[0, 6 * c:6 * c + 3].astype(np.float64)
        fw = f_c[0, 6 * c + 3:6 * c + 6].astype(np.float64)
        g, hh = R.T @ nw, R.T @ fw
        exp[6 * jid:6 * jid + 6] += np.concatenate([g + np.cross(rc, hh), hh])
    np.testing.assert_allclose(fext, exp, atol=2e-3)

    contact_bodies = sorted({int(fr["jid"]) for fr in frames})
    nz = [b for b in range(NB) if np.max(np.abs(fext[6 * b:6 * b + 6])) > 1e-6]
    assert nz == contact_bodies, f"f_ext nonzero on {nz}, expected {contact_bodies}"

"""Pinocchio as the oracle for every robot GRiD ships and every parser fixture:
each body's spatial inertia as the parser builds it, and each joint frame's
world placement at random configurations. CPU only, seconds per robot. This is
the check the 2026-10-16 UR5e finding needed: a rotated <inertial><origin> and
a prismatic joint with a rotated origin were both taken unrotated, and the ten
receipt robots happen to contain neither in a form the receipt could see."""
import contextlib
import glob
import io
import os
from pathlib import Path

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")
from URDFParser import URDFParser  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = sorted(glob.glob(str(ROOT / "grid_assets/robots/*.urdf")))
FIXTURES = sorted(glob.glob(str(ROOT / "external/URDFParser/tests/fixtures/*.urdf")))
# Parser extension joint types pinocchio's URDF reader cannot build (their
# equivalence lives in the parser's and RBDReference's programmatic tests).
NOT_IN_PINOCCHIO_URDF = {"helical_arm.urdf", "spherical_arm.urdf", "mixed_spherical_arm.urdf", "translation_arm.urdf"}
# pinocchio's URDF reader ignores a planar joint's <axis>; the skew-planar
# kinematics keeps its programmatic oracle, its inertias are checked here.
NO_URDF_KINEMATICS_ORACLE = {"skew_planar_arm.urdf"}
SCALAR_JOINTS = {"revolute", "continuous", "prismatic"}

URDFS = [p for p in ASSETS + FIXTURES if os.path.basename(p) not in NOT_IN_PINOCCHIO_URDF]
IDS = [os.path.basename(p)[:-5] for p in URDFS]


def _parse(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(path, floating_base=False)


def _pin_joint_of_body(model, link_name):
    return model.frames[model.getFrameId(link_name)].parentJoint


def _snapped_pin_model(path):
    """pinocchio built from the URDF with every <origin rpy> snapped to the
    quarter-turn grid the parser snaps joint origins to (Joint._snap_to_pi_grid,
    tolerance 1e-5): a URDF that writes 3.14159 describes a robot 2.7e-6 rad
    away from the one the parser models, and the oracle must model the parser's
    robot. RBDReference's pinocchio backend snaps the same way."""
    from bs4 import BeautifulSoup
    from URDFParser.Joint import _snap_to_pi_grid
    soup = BeautifulSoup(open(path).read(), "xml")
    for origin in soup.find_all("origin"):
        if origin.has_attr("rpy") and len(origin["rpy"].split()) == 3:
            origin["rpy"] = " ".join(repr(float(_snap_to_pi_grid(float(v)))) for v in origin["rpy"].split())
    return pin.buildModelFromXML(str(soup))


@pytest.mark.parametrize("path", URDFS, ids=IDS)
def test_every_body_inertia_matches_pinocchio(path):
    robot = _parse(path)
    model = _snapped_pin_model(path)
    checked = 0
    for joint in robot.get_joints_ordered_by_id():
        link = robot.get_link_by_name(joint.child)
        if link.is_dummy_link():
            continue
        S = np.asarray(link.get_spatial_inertia(), dtype=float)          # angular first
        P = model.inertias[_pin_joint_of_body(model, link.name)].matrix()  # linear first
        scale = max(np.abs(P).max(), 1.0)
        np.testing.assert_allclose(S[3, 3], P[0, 0], rtol=0, atol=1e-9 * scale, err_msg=f"{link.name} mass")
        np.testing.assert_allclose(S[:3, 3:], P[3:, :3], rtol=0, atol=1e-9 * scale, err_msg=f"{link.name} first moment")
        np.testing.assert_allclose(S[:3, :3], P[3:, 3:], rtol=0, atol=1e-9 * scale, err_msg=f"{link.name} inertia about the joint origin")
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("path", [p for p in URDFS if os.path.basename(p) not in NO_URDF_KINEMATICS_ORACLE],
                         ids=[i for p, i in zip(URDFS, IDS) if os.path.basename(p) not in NO_URDF_KINEMATICS_ORACLE])
def test_joint_placements_match_pinocchio_at_random_configurations(path):
    robot = _parse(path)
    joints = robot.get_joints_ordered_by_id()
    model = _snapped_pin_model(path)
    data = model.createData()
    assert all(j.jtype in SCALAR_JOINTS for j in joints), "non-scalar parser joint"
    rng = np.random.default_rng(20261016)
    for _ in range(3):
        q = rng.uniform(-1.0, 1.0, len(joints))
        q_pin = np.zeros(model.nq)
        pending = []
        for k, joint in enumerate(joints):
            if model.existJointName(joint.name):
                jm = model.joints[model.getJointId(joint.name)]
                if jm.nq == 1:
                    q_pin[jm.idx_q] = q[k]
                else:   # a continuous joint: pinocchio's unbounded revolute is (cos, sin)
                    assert jm.nq == 2 and joint.jtype == "continuous", f"{joint.name}: pinocchio nq {jm.nq}"
                    q_pin[jm.idx_q:jm.idx_q + 2] = [np.cos(q[k]), np.sin(q[k])]
                continue
            pending.append(q[k])
            link = robot.get_link_by_name(joint.child)
            if link.is_dummy_link():
                continue
            jm = model.joints[_pin_joint_of_body(model, link.name)]
            assert jm.nq == 4 and len(pending) == 3, f"{joint.name}: unexpected decomposition"
            x, y, yaw = pending
            q_pin[jm.idx_q:jm.idx_q + 4] = [x, y, np.cos(yaw), np.sin(yaw)]
            pending = []
        pin.forwardKinematics(model, data, q_pin)
        world = {-1: np.eye(4)}
        checked = 0
        for k, joint in enumerate(joints):
            X = np.asarray(joint.get_transformation_matrix_hom_function()(q[k]), dtype=float).reshape(4, 4)
            world[joint.get_id()] = world[robot.get_parent_id(joint.get_id())] @ X
            link = robot.get_link_by_name(joint.child)
            if link.is_dummy_link():
                continue
            expected = data.oMi[_pin_joint_of_body(model, link.name)].homogeneous
            np.testing.assert_allclose(world[joint.get_id()], expected, rtol=0, atol=1e-9, err_msg=joint.name)
            checked += 1
        assert checked > 0

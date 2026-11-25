"""Root translation joints on roots the stock robots do not have.

test_floating_base_offset_invariance.py covers a floating base and a planar
base whose prismatic joints lie along world axes at the world origin, with
nothing attached to the intermediate links. The generator treats any prismatic
joint joined to the world through prismatic joints only as a base position and
evaluates the dynamics with it at zero. Two synthetic robots exercise what the
stock ones leave out:

``skew_rail``     a rail along a skew axis, mounted at an offset and a rotation
                  from the world, carrying a cart with mass; a second prismatic
                  joint on the cart; then two revolute joints.
``branching``     a track carrying a trunk with mass; the trunk carries a
                  prismatic lift with an arm and, on a second branch, a
                  revolute arm.

Three claims are checked.

Dynamics do not depend on any root translation coordinate, and the generated
code returns the same bits wherever those coordinates are. That holds with a
body-local external wrench applied to every body.

Centroidal quantities are taken about the centre of mass. They do not depend
on the outermost coordinate, which moves the whole robot. They do depend on an
inner one when mass stays behind on the intermediate link or on another
branch, and there the generated code must follow the reference.

A result that stops changing is only right if it was right to begin with, so
the first-order dynamics are compared with the numpy reference at a state far
from the origin.

Run with:
    pytest test/python_wrappers/test_root_translation_offset_invariance.py -m python_wrappers -v
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers

_B = 8
_TOL = 5e-3   # float32 against the float64 reference
_FAR = ((30.0, -20.0), (-400.0, 250.0))   # (outer, inner) root coordinates
_NEAR_INNER = 2.5                          # an inner offset the reference comparison can resolve
_ALGORITHMS = [
    "inverse_dynamics", "forward_dynamics", "aba", "crba", "minv",
    "inverse_dynamics_gradient", "forward_dynamics_gradient",
    "idsva_so_body_frame", "fdsva_so", "coriolis_matrix",
    "nonlinear_effects", "generalized_gravity",
    "ccrba", "dccrba", "cmm_time_variation", "end_effector_pose", "com",
]
_INERTIA = 'ixx="0.3" iyy="0.4" izz="0.5" ixy="0.02" ixz="-0.01" iyz="0.03"'


def _link(name, mass, com):
    return (f'<link name="{name}"><inertial><origin xyz="{com}" rpy="0 0 0"/>'
            f'<mass value="{mass}"/><inertia {_INERTIA}/></inertial></link>')


def _joint(name, kind, parent, child, xyz, rpy, axis):
    return (f'<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
            f'<origin xyz="{xyz}" rpy="{rpy}"/><axis xyz="{axis}"/>'
            '<limit lower="-2000" upper="2000" effort="100" velocity="100"/></joint>')


_SKEW_RAIL = (
    '<robot name="skew_rail"><link name="world"/>'
    + _link("cart", 4.0, "0.05 -0.02 0.1") + _link("slider", 2.5, "-0.03 0.04 0.02")
    + _link("upper", 1.5, "0.0 0.02 0.15") + _link("lower", 1.0, "0.01 0.0 0.12")
    + '<link name="tip"/>'
    + _joint("rail", "prismatic", "world", "cart", "0.3 -0.2 0.5", "0.4 -0.3 0.7", "0.6 0 0.8")
    + _joint("slide", "prismatic", "cart", "slider", "0.1 0.05 0.2", "0.2 0.1 -0.5", "0 1 0")
    + _joint("shoulder", "revolute", "slider", "upper", "0.0 0.1 0.1", "0.3 0 0.2", "0 0 1")
    + _joint("elbow", "revolute", "upper", "lower", "0.05 0 0.3", "0 0.4 0", "0 1 0")
    + '<joint name="tip_fixed" type="fixed"><parent link="lower"/><child link="tip"/>'
      '<origin xyz="0 0 0.25"/></joint></robot>')

_BRANCHING = (
    '<robot name="branching_rail"><link name="world"/>'
    + _link("trunk", 5.0, "0.02 0.01 0.2") + _link("mast", 2.0, "0.0 -0.02 0.3")
    + _link("arm_a", 1.2, "0.1 0 0.02") + _link("arm_b", 1.4, "0 0.12 0.01")
    + _link("hand_b", 0.6, "0.02 0.05 0")
    + _joint("track", "prismatic", "world", "trunk", "-0.4 0.3 0.1", "0 0 0.5", "1 0 0")
    + _joint("lift", "prismatic", "trunk", "mast", "0.1 0 0.2", "0.1 -0.2 0", "0 0 1")
    + _joint("swing_a", "revolute", "mast", "arm_a", "0 0 0.4", "0.2 0 0", "0 1 0")
    + _joint("swing_b", "revolute", "trunk", "arm_b", "0 0.2 0.3", "0 0.3 0.1", "1 0 0")
    + _joint("wrist_b", "revolute", "arm_b", "hand_b", "0 0.25 0", "0.1 0 0", "0 0 1")
    + '</robot>')

# name -> (URDF text, outermost root translation joint, inner root translation joint)
_ROBOTS = {
    "skew_rail": (_SKEW_RAIL, "rail", "slide"),
    "branching": (_BRANCHING, "track", "lift"),
}


def _flat(value):
    if dataclasses.is_dataclass(value):
        parts = [getattr(value, f.name) for f in dataclasses.fields(value)]
    elif isinstance(value, dict):
        parts = list(value.values())
    elif isinstance(value, (tuple, list)):
        parts = list(value)
    else:
        parts = [value]
    return np.concatenate([np.asarray(p).reshape(_B, -1) for p in parts], axis=1)


@dataclasses.dataclass
class _Model:
    handle: object
    ref: object
    outer: int
    inner: int
    state: dict

    def q(self, outer=0.0, inner=0.0):
        q = self.state["q"].copy()
        q[:, self.outer] = outer
        q[:, self.inner] = inner
        return q.astype(np.float32)


@pytest.fixture(scope="module", params=sorted(_ROBOTS))
def model(request, tmp_path_factory):
    from RBDReference import RBDReference
    from URDFParser import URDFParser
    from grid_codegen.helpers._topology_helpers import _root_translation_joints

    text, outer_name, inner_name = _ROBOTS[request.param]
    urdf = tmp_path_factory.mktemp(request.param) / "robot.urdf"
    urdf.write_text(text)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(urdf))
    count = len(robot.get_joints_ordered_by_id())
    names = [robot.get_joint_by_id(i).get_name() for i in range(count)]
    # The generator and this test must mean the same joints.
    assert {names[i] for i in _root_translation_joints(robot, count)} == {outer_name, inner_name}
    handle = grid_rbd.register_robot(
        f"{request.param}_root_translation_pytest", str(urdf), max_batch_size=_B,
        algorithm_list=_ALGORITHMS, enable_mujoco_kernels=False)
    assert handle.nq == handle.nv == count
    rng = np.random.default_rng(11)
    state = dict(
        q=rng.uniform(-1.0, 1.0, (_B, count)),
        qd=rng.uniform(-1.0, 1.0, (_B, count)).astype(np.float32),
        qdd=rng.uniform(-1.0, 1.0, (_B, count)).astype(np.float32),
        u=rng.uniform(-5.0, 5.0, (_B, count)).astype(np.float32),
        f_ext=rng.uniform(-5.0, 5.0, (_B, 6 * handle.num_bodies)).astype(np.float32))
    return _Model(handle, RBDReference(robot), names.index(outer_name), names.index(inner_name), state)


_DYNAMICS = {
    "inverse_dynamics": lambda h, q, s: h.inverse_dynamics(q, s["qd"], s["qdd"]),
    "forward_dynamics": lambda h, q, s: h.forward_dynamics(q, s["qd"], s["u"]),
    "aba": lambda h, q, s: h.aba(q, s["qd"], s["u"]),
    "crba": lambda h, q, s: h.crba(q),
    "minv": lambda h, q, s: h.minv(q),
    "inverse_dynamics_gradient": lambda h, q, s: h.inverse_dynamics_gradient(q, s["qd"], s["qdd"]),
    "forward_dynamics_gradient": lambda h, q, s: h.forward_dynamics_gradient(q, s["qd"], s["u"]),
    "idsva_so": lambda h, q, s: h.idsva_so(q, s["qd"], s["qdd"]),
    "fdsva_so": lambda h, q, s: h.fdsva_so(q, s["qd"], s["u"]),
    "coriolis_matrix": lambda h, q, s: h.coriolis_matrix(q, s["qd"]),
    "nonlinear_effects": lambda h, q, s: h.nonlinear_effects(q, s["qd"]),
    "generalized_gravity": lambda h, q, s: h.generalized_gravity(q),
}
_WITH_WRENCH = {
    "inverse_dynamics": lambda h, q, s: h.inverse_dynamics(q, s["qd"], s["qdd"], f_ext=s["f_ext"]),
    "forward_dynamics": lambda h, q, s: h.forward_dynamics(q, s["qd"], s["u"], f_ext=s["f_ext"]),
    "aba": lambda h, q, s: h.aba(q, s["qd"], s["u"], f_ext=s["f_ext"]),
    "inverse_dynamics_gradient": lambda h, q, s: h.inverse_dynamics_gradient(
        q, s["qd"], s["qdd"], f_ext=s["f_ext"]),
    "forward_dynamics_gradient": lambda h, q, s: h.forward_dynamics_gradient(
        q, s["qd"], s["u"], f_ext=s["f_ext"]),
}
_CENTROIDAL = {
    "ccrba": lambda h, q, s: h.ccrba(q, s["qd"]),
    "dccrba": lambda h, q, s: h.dccrba(q),
    "cmm_time_variation": lambda h, q, s: h.cmm_time_variation(q, s["qd"]),
}


def _assert_same_bits(model, call, positions, what):
    at_origin = _flat(call(model.handle, model.q(), model.state))
    assert np.all(np.isfinite(at_origin))
    for outer, inner in positions:
        moved = _flat(call(model.handle, model.q(outer, inner), model.state))
        assert np.array_equal(moved, at_origin), (
            f"{what} changed with the root coordinates at ({outer}, {inner}): max abs difference "
            f"{np.max(np.abs(moved.astype(np.float64) - at_origin.astype(np.float64))):.3e}")
    return at_origin


@pytest.mark.parametrize("kernel", sorted(_DYNAMICS))
def test_dynamics_output_is_bit_identical_at_any_root_position(model, kernel):
    positions = [*_FAR, (_FAR[1][0], 0.0), (0.0, _FAR[1][1])]   # together, then each alone
    _assert_same_bits(model, _DYNAMICS[kernel], positions, kernel)


@pytest.mark.parametrize("kernel", sorted(_WITH_WRENCH))
def test_dynamics_with_a_local_wrench_is_bit_identical_at_any_root_position(model, kernel):
    with_wrench = _assert_same_bits(model, _WITH_WRENCH[kernel], _FAR, f"{kernel} with f_ext")
    without = _flat(_DYNAMICS[kernel](model.handle, model.q(), model.state))
    assert np.max(np.abs(with_wrench - without)) > 1e-2   # the wrench is not being dropped


@pytest.mark.parametrize("kernel", sorted(_CENTROIDAL))
def test_centroidal_output_is_bit_identical_along_the_outermost_coordinate(model, kernel):
    _assert_same_bits(model, _CENTROIDAL[kernel], [(outer, 0.0) for outer, _ in _FAR], kernel)


def _rows(model, q, fn):
    return np.stack([np.asarray(fn(row.astype(np.float64), i), dtype=np.float64).reshape(-1)
                     for i, row in enumerate(q)])


def test_centroidal_output_follows_the_reference_along_the_inner_coordinate(model):
    """Mass stays on the intermediate link (and on the other branch), so moving
    the inner joint changes the mass distribution and the centroidal results."""
    handle, ref, s = model.handle, model.ref, model.state
    qd = s["qd"].astype(np.float64)
    home = model.q(_FAR[1][0], 0.0)
    q = model.q(_FAR[1][0], _NEAR_INNER)
    A, h = handle.ccrba(q, s["qd"])
    A_home, _ = handle.ccrba(home, s["qd"])
    assert np.max(np.abs(np.asarray(A) - np.asarray(A_home))) > 0.1   # it does depend on it
    centre = model.q(0.0, _NEAR_INNER)   # the reference at the origin: it is exact either way
    A_ref = _rows(model, centre, lambda row, i: ref.ccrba(row, qd[i])[0])
    h_ref = _rows(model, centre, lambda row, i: ref.ccrba(row, qd[i])[1])
    np.testing.assert_allclose(np.asarray(A).reshape(_B, -1), A_ref, rtol=0, atol=_TOL)
    np.testing.assert_allclose(np.asarray(h).reshape(_B, -1), h_ref, rtol=0, atol=_TOL)
    dA_ref = _rows(model, centre, lambda row, i: ref.cmm_time_variation(row, qd[i]))
    np.testing.assert_allclose(
        np.asarray(handle.cmm_time_variation(q, s["qd"])).reshape(_B, -1), dA_ref, rtol=0, atol=_TOL)


def test_first_order_dynamics_follow_the_reference_far_from_the_origin(model):
    handle, ref, s = model.handle, model.ref, model.state
    q = model.q(*_FAR[1])
    qd, qdd, u = (s[k].astype(np.float64) for k in ("qd", "qdd", "u"))
    count = handle.nv
    wrench = s["f_ext"].astype(np.float64).reshape(_B, -1, 6)
    tau_ref = _rows(model, q, lambda row, i: ref.inverse_dynamics(row, qd[i], qdd[i])[0])
    np.testing.assert_allclose(
        np.asarray(handle.inverse_dynamics(q, s["qd"], s["qdd"])).reshape(_B, -1), tau_ref,
        rtol=0, atol=_TOL)
    tau_wrench_ref = _rows(model, q, lambda row, i: ref.inverse_dynamics(
        row, qd[i], qdd[i], f_ext=list(wrench[i]))[0])
    np.testing.assert_allclose(
        np.asarray(handle.inverse_dynamics(q, s["qd"], s["qdd"], f_ext=s["f_ext"])).reshape(_B, -1),
        tau_wrench_ref, rtol=0, atol=_TOL)
    M_ref = _rows(model, q, lambda row, i: ref.crba(row))
    np.testing.assert_allclose(
        np.asarray(handle.crba(q)).reshape(_B, count * count), M_ref, rtol=0, atol=_TOL)
    qdd_ref = _rows(model, q, lambda row, i: ref.forward_dynamics(row, qd[i], u[i]))
    np.testing.assert_allclose(
        np.asarray(handle.forward_dynamics(q, s["qd"], s["u"])).reshape(_B, -1), qdd_ref,
        rtol=0, atol=_TOL)


def test_world_quantities_still_follow_the_root(model):
    """The centre of mass and the pose are world quantities. They must move by
    the joint's own displacement: the skew rail moves along its mounted axis."""
    handle, ref = model.handle, model.ref
    outer = _FAR[0][0]
    com_origin = np.asarray(handle.com(model.q())[0]).reshape(_B, 3)
    com_moved = np.asarray(handle.com(model.q(outer, 0.0))[0]).reshape(_B, 3)
    expected = _rows(model, model.q(outer, 0.0), lambda row, i: ref.com(row)) \
        - _rows(model, model.q(), lambda row, i: ref.com(row))
    assert np.max(np.abs(expected)) > 1.0
    np.testing.assert_allclose(com_moved - com_origin, expected, rtol=0, atol=2e-4)
    pose_origin = _flat(handle.end_effector_pose(model.q()))
    pose_moved = _flat(handle.end_effector_pose(model.q(outer, 0.0)))
    assert np.max(np.abs(pose_moved - pose_origin)) > 1.0

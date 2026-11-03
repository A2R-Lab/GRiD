"""CPU contract checks for fixed-base armature code generation.

The numerical check defines the intended mass-equation/derivative oracle; it
does not execute the generated CUDA. GPU equivalence is a separate check.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pinocchio as pin
import pytest

from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_codegen import GRiDCodeGenerator


ARMATURE = np.array([0.4, 0.2])
ALGORITHMS = ["inverse_dynamics", "inverse_dynamics_gradient", "minv", "crba",
              "aba", "forward_dynamics", "forward_dynamics_gradient"]


def robot_xml(axis="0 0 -1", kind="revolute", mimic=False):
    return f"""<robot name="armature_contract">
      <link name="base"/>
      <link name="one"><inertial><origin xyz="0.2 0.03 0"/>
        <mass value="1.3"/><inertia ixx="0.06" iyy="0.08" izz="0.1" ixy="0" ixz="0" iyz="0"/>
      </inertial></link>
      <link name="two"><inertial><origin xyz="0.15 0.02 0.04"/>
        <mass value="0.7"/><inertia ixx="0.025" iyy="0.03" izz="0.04" ixy="0" ixz="0" iyz="0"/>
      </inertial></link>
      <joint name="first" type="revolute"><parent link="base"/><child link="one"/>
        <origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
        <limit lower="-2" upper="2" velocity="3" effort="20"/>
        <dynamics damping="0.2" friction="0.7"/></joint>
      <joint name="second" type="{kind}"><parent link="one"/><child link="two"/>
        <origin xyz="0.4 0 0.1"/><axis xyz="{axis}"/>
        <limit lower="-2" upper="2" velocity="3" effort="20"/>
        <dynamics damping="0.3" friction="0.4"/>
        {'<mimic joint="first" multiplier="1" offset="0"/>' if mimic else ''}
      </joint></robot>"""


def parse_robot(path, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(str(path), **kwargs)


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "two.urdf"
    path.write_text(robot_xml())
    return path


def emit(robot, path, **kwargs):
    generator = GRiDCodeGenerator(robot, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        generator.gen_all_code(output_path=str(path), algorithm_list=ALGORITHMS,
                               enable_mujoco_kernels=False)
    return path.read_text()


@pytest.mark.parametrize("bad", [[1.0], [-1.0, 0.0], [float("nan"), 0.0],
                                  [float("inf"), 0.0], [float("-inf"), 0.0]])
def test_armature_rejects_invalid_vector(model_path, bad):
    with pytest.raises(ValueError, match="JOINT_ARMATURE"):
        GRiDCodeGenerator(parse_robot(model_path), JOINT_ARMATURE=bad)


@pytest.mark.parametrize("options", [{"floating_base": True},
                                     {"axis": "0.6 0.8 0"},
                                     {"kind": "prismatic"}, {"mimic": True}])
def test_armature_rejects_unsupported_topology(tmp_path, options):
    options = dict(options)
    floating = options.pop("floating_base", False)
    path = tmp_path / "unsupported.urdf"
    path.write_text(robot_xml(**options))
    robot = parse_robot(path, floating_base=floating)
    with pytest.raises(NotImplementedError, match="JOINT_ARMATURE"):
        GRiDCodeGenerator(robot, JOINT_ARMATURE=[0.1] * robot.get_num_joints())


def test_none_and_zero_are_byte_neutral_with_tanh(model_path, tmp_path):
    robot = parse_robot(model_path)
    kwargs = dict(USE_JOINT_DYNAMICS=True, FRICTION_MODEL="tanh", FRICTION_WIDTH=0.2)
    omitted = emit(robot, tmp_path / "omitted.cuh", **kwargs)
    none = emit(robot, tmp_path / "none.cuh", JOINT_ARMATURE=None, **kwargs)
    zero = emit(robot, tmp_path / "zero.cuh", JOINT_ARMATURE=[0.0, 0.0], **kwargs)
    assert omitted == none == zero
    assert "s_armature" not in zero


def test_nonzero_emission_preserves_smooth_friction(model_path, tmp_path):
    source = emit(parse_robot(model_path), tmp_path / "armature.cuh",
                  JOINT_ARMATURE=ARMATURE, USE_JOINT_DYNAMICS=True,
                  FRICTION_MODEL="tanh", FRICTION_WIDTH=[0.2, 0.25])
    assert "s_c[dof_id] += s_armature[dof_id] * s_qdd[dof_id];" in source
    assert "s_M[jid+jid*2]" in source
    assert source.count(" + s_armature[") >= 3  # ABA, Minv, CRBA
    assert "tanh(" in source
    assert "sech" in source  # Existing analytical smooth-friction derivative
    assert "s_linalg_smem" not in source  # Removed in the 0.5.5 ABI


def test_mass_equation_and_first_derivatives_with_tanh(model_path):
    """Cross-check the intended coupled model using independent Pinocchio ABA.

    RBDReference stays unchanged and supplies only the link-body quantities.
    The armature is added explicitly in this oracle, never hidden in the peer.
    """
    robot = parse_robot(model_path)
    ref = RBDReference(robot, use_joint_dynamics=True, friction_model="tanh", friction_width=0.2)
    model = pin.buildModelFromUrdf(str(model_path))
    model.armature = ARMATURE.copy()
    coulomb, damping = np.array([0.7, 0.4]), np.array([0.2, 0.3])
    rng = np.random.default_rng(20261026)
    velocities = [np.zeros(2), np.array([0.01, -0.02]), *rng.uniform(-2, 2, (8, 2))]
    for velocity in velocities:
        q, torque = rng.uniform(-1, 1, 2), rng.uniform(-3, 3, 2)
        mass = np.asarray(ref.crba(q)) + np.diag(ARMATURE)
        bias = np.asarray(ref.inverse_dynamics(q, velocity)[0])
        acceleration = np.linalg.solve(mass, torque - bias)
        passive = damping * velocity + coulomb * np.tanh(velocity / 0.2)
        data = model.createData()
        expected = pin.aba(model, data, q, velocity, torque - passive).copy()
        np.testing.assert_allclose(acceleration, expected, rtol=2e-6, atol=2e-6)
        id_gradient = ref.inverse_dynamics_gradient(q, velocity, acceleration)
        dq, dv = np.hsplit(-np.linalg.solve(mass, id_gradient), 2)
        pin.computeABADerivatives(model, data, q, velocity, torque - passive)
        slope = damping + (coulomb / 0.2) * (1 - np.tanh(velocity / 0.2)**2)
        np.testing.assert_allclose(dq, data.ddq_dq, rtol=4e-6, atol=4e-6)
        np.testing.assert_allclose(dv, data.ddq_dv - np.linalg.solve(mass, np.diag(slope)),
                                   rtol=4e-6, atol=4e-6)
        for axis in range(2):
            step = np.eye(2)[axis] * 1e-6
            def value(v):
                return np.linalg.solve(mass, torque - np.asarray(ref.inverse_dynamics(q, v)[0]))
            finite_difference = (value(velocity + step) - value(velocity - step)) / 2e-6
            np.testing.assert_allclose(dv[:, axis], finite_difference, rtol=1e-5, atol=1e-6)

"""The floating base quaternion is normalised in one place.

Every generated use of the base quaternion goes through `grid_quat_norm2` (the
squared norm) or `grid_base_rotation` (which calls it). No kernel carries its
own copy of the rotation or of the squared norm, so `assume_unit_quaternions` removes the
normalisation from the whole header by changing one function body.
"""
from __future__ import annotations

import contextlib
import os
import re

import pytest
import sympy as sp

from grid_assets import robot_urdf
from external.URDFParser.URDFParser import URDFParser
from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator
from grid_codegen.helpers._expr_printer import expr_T

HELPER = re.compile(r"T grid_quat_norm2\(T qx, T qy, T qz, T qw\) \{\n(.*?)\n\s*\}", re.S)
# A hand-written copy would read the quaternion slots into scalars or divide by their squares.
COPIES = re.compile(r"T qx = \w+\[3\]|/\s*\(+\s*\(?s_q\[3\]\)?\s*\*")


def header(tmp_path, robot_id, floating, **options):
    path = tmp_path / f"{robot_id}.cuh"
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        robot = URDFParser().parse(str(robot_urdf(robot_id)), floating_base=floating)
        GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(output_path=str(path), **options)
    return path.read_text()


def test_printer_folds_the_squared_norm():
    q1, q2, q3, q4 = sp.symbols("q1_fb q2_fb q3_fb q4_fb")
    n2 = q1**2 + q2**2 + q3**2 + q4**2
    text = expr_T(2 * q1 * q2 / n2 + q3 / n2**2)
    assert "q1_fb)*(q1_fb" not in text
    assert text.count("grid_quat_norm2<T>(q1_fb, q2_fb, q3_fb, q4_fb)") == 3
    with pytest.raises(ValueError, match="squared norm"):
        expr_T(q1 / (q1**2 + q2**2))


def test_norm_substitution_keeps_the_grouping_and_refuses_what_it_cannot_place():
    """The substitution works on printed text. These are the forms it must keep right."""
    for tag in ("fb", "sph"):
        q1, q2, q3, q4 = sp.symbols(f"q1_{tag} q2_{tag} q3_{tag} q4_{tag}")
        n2 = q1**2 + q2**2 + q3**2 + q4**2
        call = f"grid_quat_norm2<T>(q1_{tag}, q2_{tag}, q3_{tag}, q4_{tag})"
        assert expr_T(q1 / n2) == f"q1_{tag}/({call})"
        assert expr_T(sp.sqrt(n2)) == f"sqrt({call})"
        assert expr_T(q1 / sp.sqrt(n2)) == f"q1_{tag}/sqrt({call})"
        assert expr_T((n2 + 1) / n2) == f"({call} + 1)/({call})"
        assert expr_T(q1 / n2**5) == f"q1_{tag}/pow({call}, static_cast<T>(5))"
        # a sum that is not the squared norm is left alone
        assert "grid_quat_norm2" not in expr_T(q1**2 + q2**2 + q3**2 + 2 * q4**2)
        # a divisor that is not the squared norm itself is refused, not printed
        for divisor in (2 * n2, n2 + 1, q1**2 + q2**2, q1):
            with pytest.raises(ValueError, match="squared norm"):
                expr_T(q2 / divisor)


@pytest.mark.parametrize("unit", [False, True])
def test_floating_header_normalises_in_one_place(tmp_path, unit):
    source = header(tmp_path, "go2", True, enable_mujoco_kernels=True, assume_unit_quaternions=unit)
    bodies = HELPER.findall(source)
    assert len(bodies) == 1
    assert ("return static_cast<T>(1);" in bodies[0]) == unit
    assert ("return fma(qx, qx, fma(qy, qy, fma(qz, qz, qw*qw)));" in bodies[0]) == (not unit)
    assert source.count("void grid_base_rotation(const T *q, T *R) {") == 1
    assert source.count("grid_base_rotation<T>(&") > 100
    assert source.count("grid_quat_norm2<T>(s_q[3], s_q[4], s_q[5], s_q[6])") > 100
    assert not COPIES.search(source)


def test_spherical_joint_quaternion_uses_the_same_helper(tmp_path):
    from pathlib import Path
    import external.URDFParser as parser_package
    urdf = Path(parser_package.__file__).parent / "tests/fixtures/spherical_arm.urdf"
    algorithms = ["inverse_dynamics", "forward_dynamics", "end_effector_pose", "inverse_dynamics_gradient"]
    for unit in (False, True):
        path = tmp_path / f"spherical_{unit}.cuh"
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            robot = URDFParser().parse(str(urdf), floating_base=False)
            GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
                output_path=str(path), algorithm_list=algorithms, assume_unit_quaternions=unit)
        source = path.read_text()
        bodies = HELPER.findall(source)
        assert len(bodies) == 1 and ("return static_cast<T>(1);" in bodies[0]) == unit
        assert source.count("grid_quat_norm2<T>(s_q[0], s_q[1], s_q[2], s_q[3])") > 10
        assert not re.search(r"/\s*\(+\s*\(?s_q\[0\]\)?\s*\*", source)


def _robot(joint):
    return f'<?xml version="1.0"?><robot name="r"><link name="a"/><link name="b"/>{joint}</robot>'.encode()


def _joint(kind):
    return f'<joint name="j" type="{kind}"><parent link="a"/><child link="b"/></joint>'


def test_quaternion_detection_reads_the_joints_as_the_parser_does():
    from grid_rbd._cache import has_quaternion_coordinates
    assert has_quaternion_coordinates(_robot(_joint("revolute")), True)
    assert has_quaternion_coordinates(_robot(_joint("spherical")), False)
    assert has_quaternion_coordinates(_robot(_joint("spher&#105;cal")), False)       # an encoded attribute
    assert has_quaternion_coordinates(_robot(_joint("spherical").replace('"', "'")), False)
    assert not has_quaternion_coordinates(_robot(_joint("revolute")), False)
    assert not has_quaternion_coordinates(_robot("<!-- " + _joint("spherical") + " -->" + _joint("revolute")), False)
    assert has_quaternion_coordinates(b"not a urdf", False)                          # unreadable: keep the option


def test_option_enters_the_code_options_only_when_true_on_a_model_with_a_quaternion(tmp_path):
    """Both routes (the handle and the artifact builder) build their options the same way."""
    from grid_rbd import _artifacts
    spherical, encoded, revolute = (tmp_path / name for name in ("s.urdf", "e.urdf", "r.urdf"))
    spherical.write_bytes(_robot(_joint("spherical")))
    encoded.write_bytes(_robot(_joint("spher&#105;cal")))
    revolute.write_bytes(_robot(_joint("revolute")))

    def options(path, floating, unit):
        return _artifacts._code_options(
            path, 120, floating_base=floating, ee_joint_names=None, dtype="float32", algorithm_list=None,
            enable_mujoco_kernels=True, cost_frames=None, contact_frames=None, use_joint_dynamics=False,
            friction_model="sign", friction_width=None, runtime_inertia=False, runtime_transform=False,
            runtime_joint_dynamics=False, emit_plant=False, collision=False, collision_res=None,
            collision_native=False, spherized_urdf=None, strict_collision=True, launch_config_profile=None,
            assume_unit_quaternions=unit)

    for path, floating in ((spherical, False), (encoded, False), (revolute, True)):
        assert options(path, floating, True).get("assume_unit_quaternions") is True
        assert "assume_unit_quaternions" not in options(path, floating, False)
        assert options(path, floating, True) != options(path, floating, False)
    assert "assume_unit_quaternions" not in options(revolute, False, True)
    assert options(revolute, False, True) == options(revolute, False, False)


def test_fixed_base_header_has_no_quaternion_helper(tmp_path):
    source = header(tmp_path, "iiwa14", False)
    assert "grid_quat_norm2" not in source and "grid_base_rotation" not in source

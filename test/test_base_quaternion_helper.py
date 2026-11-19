"""The floating base quaternion is normalised in one place.

Every generated use of the base quaternion goes through `grid_quat_inv_norm2` (the
reciprocal squared norm) or `grid_base_rotation` (which calls it). No kernel carries its
own copy of the rotation or of the division, so `assume_unit_quaternions` removes the
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

HELPER = re.compile(r"T grid_quat_inv_norm2\(T qx, T qy, T qz, T qw\) \{\n(.*?)\n\s*\}", re.S)
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
    assert "/" not in text
    assert text.count("grid_quat_inv_norm2<T>(q1_fb, q2_fb, q3_fb, q4_fb)") == 3
    with pytest.raises(ValueError, match="squared norm"):
        expr_T(q1 / (q1**2 + q2**2))


@pytest.mark.parametrize("unit", [False, True])
def test_floating_header_normalises_in_one_place(tmp_path, unit):
    source = header(tmp_path, "go2", True, enable_mujoco_kernels=True, assume_unit_quaternions=unit)
    bodies = HELPER.findall(source)
    assert len(bodies) == 1
    assert ("return static_cast<T>(1);" in bodies[0]) == unit
    assert ("/(qx*qx + qy*qy + qz*qz + qw*qw)" in bodies[0]) == (not unit)
    assert source.count("void grid_base_rotation(const T *q, T *R) {") == 1
    assert source.count("grid_base_rotation<T>(&") > 100
    assert source.count("grid_quat_inv_norm2<T>(s_q[3], s_q[4], s_q[5], s_q[6])") > 100
    assert not COPIES.search(source)


def test_fixed_base_header_has_no_quaternion_helper(tmp_path):
    source = header(tmp_path, "iiwa14", False)
    assert "grid_quat_inv_norm2" not in source and "grid_base_rotation" not in source

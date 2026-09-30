"""CPU-only guard for the recursive floating ABA's root gravity (2026-09-29).

The second forward pass needs a0 = -X0^{-1}[:,5] * g. It used to invert the root
transform X0 with invert_matrix, which does not pivot: at axis-permutation
orientations (quat xyzw 0.5*(1,1,1,1)) X0's leading diagonal entry is exactly 0
and every qdd came back NaN, on every non-mimic floating robot. For a spatial
transform that column is [0; X(5,3:6)^T], the closed form inverse_dynamics
already emits, so the kernel must use it and never invert X0.
"""
from __future__ import annotations

import contextlib
import os

import pytest

from config import robot_urdf
from external.URDFParser.URDFParser import URDFParser
from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator

_ROOT_GRAVITY = "(row < 3 ? static_cast<T>(0) : -s_XImats[6*row + 5] * gravity)"


@pytest.mark.parametrize("robot_id", ["iiwa14", "go2", "g1"])
def test_floating_aba_root_gravity_is_closed_form(robot_id, tmp_path):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(robot_urdf(robot_id)), floating_base=True)
        generator = GRiDCodeGenerator(robot, FILE_NAMESPACE="grid")
        assert not generator.robot_has_mimic_joints()
        header = tmp_path / "grid.cuh"
        generator.gen_all_code(output_path=str(header),
                               algorithm_list=["inverse_dynamics", "minv", "forward_dynamics", "aba"])
    source = header.read_text()
    start = source.index("void aba_inner(")
    body = source[start:source.index("\n    }\n", start)]
    assert "Recursive floating ABA" in body
    assert "= s_XImats[ind];" not in body  # no copy of X0 into an inversion buffer
    # the only 6x6 inversion left is the SPD root articulated inertia D
    assert body.count("invert_matrix(6,") == 1
    assert body.count(_ROOT_GRAVITY) == 1

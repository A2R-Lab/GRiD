"""A fixed-joint target whose name is not a C identifier (UR5e's hyphenated
flange) must still name valid emitted symbols. The raw name keys the robot-tree
lookup; only the suffix pasted into function, kernel and macro names changes."""
import contextlib
import io
import re
from pathlib import Path

import pytest

from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator
from grid_codegen.helpers._code_generation_helpers import ident_suffix
from URDFParser import URDFParser

FIXTURE = Path(__file__).resolve().parents[1] / "external/URDFParser/tests/fixtures/rotated_inertial_arm.urdf"


def test_ident_suffix_rules():
    assert ident_suffix("") == "" and ident_suffix(None) == ""
    assert ident_suffix("tool0") == "_tool0"
    assert ident_suffix("tool0-flange") == "_tool0_flange"
    assert ident_suffix("ee.link 1") == "_ee_link_1"
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", "end_effector_pose" + ident_suffix("tool0-flange"))


def test_hyphenated_fixed_target_emits_valid_identifiers(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(FIXTURE), floating_base=False)
    assert robot.get_fixed_joint_by_name("tool0-flange") is not None
    out = tmp_path / "grid.cuh"
    with contextlib.redirect_stdout(io.StringIO()):
        GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
            algorithm_list=["end_effector_pose"], fixed_target_name="tool0-flange", output_path=str(out))
    text = out.read_text()
    assert "end_effector_pose_tool0_flange(" in text
    assert "end_effector_pose_kernel_tool0_flange" in text
    assert "#define GRID_RBD_EE_POSE_FN end_effector_pose_tool0_flange" in text
    # The raw name may appear in a comment and in the string macro that
    # records the target; it must not appear in code outside those.
    code = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    code = re.sub(r"//[^\n]*", "", code)
    code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', code)
    assert "tool0-flange" not in code
    assert '#define GRID_EE_FIXED_TARGET_NAME "tool0-flange"' in text

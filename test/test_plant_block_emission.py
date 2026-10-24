"""E1 (HJCD-IK, 2026-10-23): the grid_plant block rides on a plant request.

A header generated for kinematics alone carries no plant block (the barrier families, the
activations and the menu presets were compile-time weight a kinematics consumer never
instantiated); the default 'all' profile, any request with a plant algorithm, baked cost or
contact frames, a multi-target batch, or an explicit emit_plant=True keep it. The bindings
always pass emit_plant=True because the wrapper units compile against the block.
"""
import contextlib
import io
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from grid_codegen import GRiDCodeGenerator  # noqa: E402
from grid_codegen.GRiDCodeGenerator import PLANT_TRIGGER_ALGORITHMS, plant_block_requested  # noqa: E402
from grid_assets import robot_urdf  # noqa: E402
from URDFParser import URDFParser  # noqa: E402

PLANT_OPEN = "namespace grid_plant {"
PLANT_SKIPPED = "[grid_plant] namespace not emitted"


def _header(tmp_path, name, **kwargs):
    robot = URDFParser().parse(str(robot_urdf("iiwa14")))
    gen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid")
    out = tmp_path / (name + ".cuh")
    with contextlib.redirect_stdout(io.StringIO()):
        gen.gen_all_code(output_path=str(out), enable_mujoco_kernels=False, **kwargs)
    return out.read_text()


def test_kinematics_profile_drops_the_plant_block(tmp_path):
    h = _header(tmp_path, "kin", codegen_profile="kinematics")
    assert PLANT_OPEN not in h and PLANT_SKIPPED in h
    assert "GRID_PLANT_HAS_" not in h and "activation" not in h.lower().split("namespace grid_collision")[0][-2000:]


def test_explicit_option_and_plant_requests_keep_the_block(tmp_path):
    assert PLANT_OPEN in _header(tmp_path, "kin_forced", codegen_profile="kinematics", emit_plant=True)
    assert PLANT_OPEN in _header(tmp_path, "kin_deriv", codegen_profile="kinematics-derivatives")
    assert PLANT_OPEN in _header(tmp_path, "cost_frames", codegen_profile="kinematics",
                                 cost_frames=["iiwa_joint_4"])
    assert PLANT_OPEN in _header(tmp_path, "integrators", algorithm_list=["integrator"])
    assert PLANT_OPEN not in _header(tmp_path, "all_off", emit_plant=False)


def test_default_profile_keeps_the_block(tmp_path):
    h = _header(tmp_path, "all")
    assert PLANT_OPEN in h and PLANT_SKIPPED not in h


@pytest.mark.parametrize("profile,expected", [
    ("all", True), ("kinematics", False), ("kinematics-derivatives", True), ("dynamics-core", False),
    ("integrators", True), ("second-order", True)])
def test_decision_rule_per_profile(profile, expected):
    from grid_codegen._algo_profiles import normalize_codegen_algorithms
    robot = URDFParser().parse(str(robot_urdf("iiwa14")))
    gen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid")
    algorithms = normalize_codegen_algorithms(gen, profile, None)
    assert plant_block_requested(None, profile, None, algorithms, None, None, None) is expected
    assert plant_block_requested(True, profile, None, algorithms, None, None, None) is True
    assert plant_block_requested(False, profile, None, algorithms, None, None, None) is False


def test_trigger_set_names_only_plant_compositions():
    assert "end_effector_pose" not in PLANT_TRIGGER_ALGORITHMS
    assert {"integrator", "fdsva_so", "end_effector_pose_gradient", "com", "ccrba"} <= PLANT_TRIGGER_ALGORITHMS

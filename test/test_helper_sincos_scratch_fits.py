"""The XmatsHom / XImats helpers write sin(q) and cos(q) (or the mimic per-body
fold) into the caller's ``s_temp`` BEFORE any inner runs. The arena carve must
therefore reserve at least that much scratch even when the inner itself needs
less — guide §7.z28: the one-target ``end_effector_pose`` inner asks for 2x16
floats, so on g1 (36 positions) the helper's cos block overran into the
topology helpers and raced (NaN pose rows, nondeterministic per block).

Parses the emitted carve comment of each kernel that calls a sin/cos helper
and compares it with the floor. Pure Python — no nvcc, no GPU.
"""
from __future__ import annotations
import contextlib
import os
import re
from pathlib import Path

import pytest

from test.test_shared_arena_covers_carve import robot_urdf

KERNEL_RE = re.compile(r"^\s*void\s+(\w+)\(")
TEMP_RE = re.compile(r"//\s+T s_temp\[(\d+)\]")
HELPER_RE = re.compile(r"load_update_(?:XmatsHom|XImats)_helpers<T>\(")


def _generate(robot_id, floating, target, out):
    from URDFParser import URDFParser
    from grid_codegen import GRiDCodeGenerator
    urdf = robot_urdf(robot_id)
    if not urdf.exists():
        pytest.skip(f"{robot_id}.urdf not found")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(urdf), floating_base=floating)
        if robot is None:
            pytest.skip(f"{robot_id} URDF parse failed")
        gen = GRiDCodeGenerator(robot, FILE_NAMESPACE="grid")
        gen.gen_all_code(codegen_profile="all", output_path=str(out), algorithm_list=["end_effector_pose"],
                         fixed_target_name=target, enable_mujoco_kernels=False)
    return gen, out.read_text()


def _kernel_temps(text):
    """{kernel: s_temp count} for every kernel whose body calls a sin/cos helper."""
    lines = text.splitlines()
    # `__global__` and `__launch_bounds__(...)` sit on their own lines above the signature.
    starts = [(i, m.group(1)) for i, line in enumerate(lines) for m in [KERNEL_RE.match(line)]
              if m and any("__global__" in prev for prev in lines[max(0, i - 3):i])]
    out = {}
    for n, (i, name) in enumerate(starts):
        stop = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        body = "\n".join(lines[i:stop])
        temp = TEMP_RE.search(body)
        if temp and HELPER_RE.search(body):
            out[name] = int(temp.group(1))
    return out


@pytest.mark.parametrize("robot_id, floating, target", [
    ("g1", True, "right_hand_palm_joint"),
    ("go2", True, "FR_foot_joint"),
    ("iiwa14", False, "iiwa_joint_ee"),
    # fr3's leaves are the mimic finger pair, which a single fixed target cannot
    # name; "" emits the all-leaves kernel (two targets), still one carve to check.
    ("fr3", False, ""),
])
def test_one_target_end_effector_pose_carve_holds_the_helper_sincos_table(robot_id, floating, target, tmp_path):
    gen, text = _generate(robot_id, floating, target, tmp_path / f"{robot_id}.cuh")
    # The independent part is the parse of the EMITTED carve; the floor is the
    # helper's documented scratch size (2*num_pos, 3*NB with mimic joints).
    floor = (3 * gen.robot.get_num_joints()) if gen.robot_has_mimic_joints() else 2 * gen.robot.get_num_pos()
    temps = _kernel_temps(text)
    ee = {k: v for k, v in temps.items() if k.startswith("end_effector_pose_kernel")}
    assert ee, "no end_effector_pose kernel calling a sin/cos helper was emitted"
    short = {k: v for k, v in temps.items() if v < floor}
    assert not short, f"{robot_id}: s_temp smaller than the helper's sin/cos table ({floor}): {short}"


def test_floor_matches_helper_documented_size():
    """The floor is the base of the helper's own documented scratch size."""
    from grid_codegen.helpers._topology_helpers import _helpers_sincos_temp_floor
    from URDFParser import URDFParser
    from grid_codegen import GRiDCodeGenerator
    urdf = robot_urdf("go2")
    if not urdf.exists():
        pytest.skip("go2.urdf not found")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        gen = GRiDCodeGenerator(URDFParser().parse(str(urdf), floating_base=True), FILE_NAMESPACE="grid")
    assert _helpers_sincos_temp_floor(gen) == 2 * gen.robot.get_num_pos() == gen.gen_load_update_XImats_helpers_temp_mem_size()

"""CPU-only guards for floating-base mimic codegen (2026-09-29, h1_2 floating).

Two generator reads were wrong once a mimic joint appears on a floating base:

* the signed S index was read at ``s_topology_helpers[nv + jid]``, but the table
  holds parent_inds (NJ) and then S_inds with six floating-root entries, so body
  jid sits at ``NJ + 5 + jid``. The two agree only when nv == NJ + 5, i.e. with
  no mimic joints (mimic joints are bodies without a velocity slot);
* the inverse-dynamics ``a += (v x S) qd`` term read ``s_qd[jid + 5]``, which is
  the body's velocity slot only until the first mimic joint.

Both produced velocity-dependent bias errors (h1_2: qdd off by up to 6.6e4 at
a hand joint) that 1% norm-relative guards had hidden.
"""
from __future__ import annotations

import contextlib
import os
import re

import pytest

from config import ROBOT_ASSETS_DIR, robot_urdf
from external.URDFParser.URDFParser import URDFParser
from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator

_ROBOTS = sorted(p.stem for p in ROBOT_ASSETS_DIR.glob("*.urdf"))


def _generator(robot_id, floating):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(robot_urdf(robot_id)), floating_base=floating)
        return GRiDCodeGenerator(robot, FILE_NAMESPACE="grid")


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
@pytest.mark.parametrize("robot_id", _ROBOTS)
def test_s_inds_stride_points_at_each_bodys_axis(robot_id, floating):
    generator = _generator(robot_id, floating)
    robot = generator.robot
    num_joints = robot.get_num_joints()
    # The table exactly as gen_init_topology_helpers lays it out.
    table = [robot.get_parent_id(jid) for jid in range(num_joints)]
    table += [int(entry) for entry in robot.get_S_inds(num_joints)]
    stride = generator._s_inds_stride()
    first = 1 if floating else 0
    for jid in range(first, num_joints):
        if not robot.S_is_cardinal_by_id(jid):
            continue
        assert table[stride + jid] == robot.get_signed_S_index_by_id(jid), (robot_id, jid)
    if not generator.robot_has_mimic_joints():
        assert stride == robot.get_num_vel()  # byte-identical for every non-mimic robot


@pytest.mark.parametrize("robot_id", ["fr3"])
def test_floating_mimic_velocity_product_reads_the_mimic_slot(robot_id, tmp_path):
    generator = _generator(robot_id, True)
    assert generator.robot_has_mimic_joints()
    header = tmp_path / "grid.cuh"
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        generator.gen_all_code(output_path=str(header),
                               algorithm_list=["inverse_dynamics", "minv", "forward_dynamics"])
    source = header.read_text()
    stride = generator._s_inds_stride()
    for name in ("inverse_dynamics_inner(", "inverse_dynamics_inner_vaf("):
        starts = [m.start() for m in re.finditer(r"void " + re.escape(name), source)]
        assert starts, name
        for start in starts:
            body = source[start:source.index("\n    }\n", start)]
            for line in body.splitlines():
                if "peq_scaled" not in line:
                    continue
                assert "s_qd[jid + 5]" not in line, line.strip()
                assert not re.search(r"s_qd\[\d+\]", line), line.strip()
                for offset in re.findall(r"s_topology_helpers\[(\d+) \+ jid\]", line):
                    assert int(offset) == stride, line.strip()


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
def test_mimic_aba_uses_the_decomposition_with_its_whole_arena(floating, tmp_path):
    """ABA's recursion cannot fold mimic joints; mimic robots must take the
    Minv * (tau - c) decomposition on BOTH bases (floating used to take the
    recursion), and the decomposition has no cold band, so every spill rung that
    keeps the arena in smem must size it in full (TIER_LITE used the recursion's
    hot size: h1_2 fixed 4998 of 14001 floats)."""
    from grid_codegen.algorithms._aba import _aba_surgical_inner_smem_size

    generator = _generator("fr3", floating)
    assert generator.robot_has_mimic_joints()
    assert _aba_surgical_inner_smem_size(generator) == generator.gen_aba_inner_temp_mem_size()
    header = tmp_path / "grid.cuh"
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        generator.gen_all_code(output_path=str(header),
                               algorithm_list=["inverse_dynamics", "minv", "forward_dynamics", "aba"])
    source = header.read_text()
    start = source.index("void aba_inner(")
    body = source[start:source.index("\n    }\n", start)]
    assert "mimic ABA: qdd = Minv * (tau - rnea(q,qd,0))" in body
    assert "Recursive floating ABA" not in body


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
@pytest.mark.parametrize("robot_id", ["iiwa14", "go2", "fr3", "h1_2"])
def test_aba_launch_arena_matches_the_kernel_layout(robot_id, floating, tmp_path):
    """The launch smem (algo_registry ArenaCtx) and the kernel's surgical-rung
    layout (_aba_surgical_inner_smem_size) are computed separately; they must
    agree or the kernel runs past its dynamic smem (h1_2 fixed TIER_LITE:
    illegal memory access, 2026-09-29)."""
    from grid_codegen.algo_registry import arena_ctx_from_codegen
    from grid_codegen.algorithms._aba import _aba_surgical_inner_smem_size

    generator = _generator(robot_id, floating)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        generator.gen_all_code(output_path=str(tmp_path / "grid.cuh"),
                               algorithm_list=["inverse_dynamics", "minv", "forward_dynamics", "aba"])
    assert arena_ctx_from_codegen(generator).aba_surgical_inner == _aba_surgical_inner_smem_size(generator)

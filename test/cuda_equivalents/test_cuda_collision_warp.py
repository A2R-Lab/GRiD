"""Gates for the G1/G2/G3 collision asks (HJCD-IK -> GRiD, 2026-10-04).

G1  the sphere batch exposed as device data (grid_collision::sphere_anchor / sphere_offset /
    sphere_radius, per tier) must equal the baked batch the FK extractor uses (mt_anchor /
    mt_offset inside multi_target_position_inner and g_collision_sphere_r) — HJCD's
    `test_kernel_sidecar_tables_match_the_baked_collision_batch`, ported.
G2  grid_collision::warp::config_free / collision_distance from caller-held joint world
    transforms (ee_pose_inner_warp) must give the block path's verdict and clearances on random
    configurations x random environments, from warp 0 of a multi-warp block, at several block
    sizes (cuda_collision_warp_runner.cu).
G3  with a 2-tier spherizer spec the warp verdict runs the broad->fine cascade; the verdict must
    still equal the block (fine-equivalent) one.
"""
from __future__ import annotations
import contextlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._collision import build_self_cc_ranges
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.test_cuda_collision_config_free import _robot, _gen_header

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import robot_urdf  # noqa: E402

COLLISION_INCLUDE = REPO_ROOT / "grid_codegen" / "collision"
RUNNER_SOURCE = Path(__file__).with_name("cuda_collision_warp_runner.cu")


def _mixed_spec(robot):
    """Two spheres per movable joint with radii large enough that random configurations hit the
    environment and each other often (the verdict must be exercised on both branches)."""
    anchors, offset, radius = [], [], []
    for k, j in enumerate(robot.get_joints_ordered_by_id()):
        jid = int(j.get_id())
        anchors += [jid, jid]
        offset += [0.02 + 0.005 * k, -0.01, 0.03, -0.03, 0.02, 0.09]
        radius += [0.05, 0.04]
    return {"anchor": anchors, "offset": offset, "radius": radius,
            "self_cc_ranges": build_self_cc_ranges(robot, anchors)}


def _ints(text):
    return [int(v) for v in re.findall(r"-?\d+", text)]


def _floats(text):
    return [float(v.rstrip("f")) for v in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?f?", text)]


def _baked(header_text, name):
    # `__constant__ int x[N] = {..};` (G1 tables) or `static const T x[] = { static_cast<T>(..), .. };` (mt_*)
    m = re.search(r"\b" + re.escape(name) + r"\[\d*\] = \{([^}]*)\};", header_text)
    assert m, name
    return m.group(1).replace("static_cast<T>", "")


def _compile(build_dir):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / (RUNNER_SOURCE.stem + ".exe")
    cmd = [nvcc, "-std=c++17", "-O2", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           "-I", str(build_dir), "-I", str(COLLISION_INCLUDE), "-o", str(exe), str(runner_copy)]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"{RUNNER_SOURCE.name} compile FAILED.\ncmd: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return exe


def _run(exe, threads, n_cfg, seed, amplitude=None):
    args = [str(exe), str(threads), str(n_cfg), str(seed)] + ([str(amplitude)] if amplitude is not None else [])
    run = subprocess.run(args, capture_output=True, text=True)
    assert run.returncode == 0, f"warp runner FAILED (threads={threads}):\nstdout:\n{run.stdout[-3000:]}\nstderr:\n{run.stderr}"
    summary = [l for l in run.stdout.splitlines() if l.startswith("SUMMARY")][0]
    hits = int(re.search(r"hits=(\d+)", summary).group(1))
    configs = int(re.search(r"configs=(\d+)", summary).group(1))
    # the batch must exercise both branches of the verdict
    assert 0 < hits < configs, summary
    return summary


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_sphere_tables_match_the_baked_collision_batch(tmp_path):
    """G1: sphere_anchor/offset/radius == mt_anchor/mt_offset/g_collision_sphere_r (same order)."""
    robot = _robot()
    spec = _mixed_spec(robot)
    header = _gen_header(robot, tmp_path / "tables", spec).read_text()
    assert _ints(_baked(header, "sphere_anchor")) == spec["anchor"] == _ints(_baked(header, "mt_anchor"))
    assert _floats(_baked(header, "sphere_offset")) == pytest.approx(spec["offset"], abs=1e-7)
    assert _floats(_baked(header, "mt_offset")) == pytest.approx(spec["offset"], abs=1e-7)
    assert _floats(_baked(header, "sphere_radius")) == pytest.approx(spec["radius"], abs=1e-7)
    assert _floats(_baked(header, "g_collision_sphere_r")) == pytest.approx(spec["radius"], abs=1e-7)
    assert re.search(r"constexpr int NUM_SPHERES = NUM_COLLISION_SPHERES;", header)
    assert "namespace warp {" in header


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("threads", [32, 64, 256])
def test_warp_matches_block_single_tier(tmp_path, threads):
    """G2: warp verdict + clearances == block path on 300 random configs x environments."""
    robot = _robot()
    build_dir = tmp_path / "warp_single"
    _gen_header(robot, build_dir, _mixed_spec(robot))
    exe = _compile(build_dir)
    print(_run(exe, threads, 300, 7))


def _two_tier_spec(robot):
    """Hand-built broad->fine spec: fine = _mixed_spec; broad = ONE covering sphere per joint
    (centre at the joint origin, radius = max over that joint's fine spheres of |offset| + r,
    plus margin) — the covering property the cascade relies on holds by construction. (The
    spherizer's dense iiwa14 model self-collides at rest, so it cannot exercise the
    broad-clear branch.)"""
    import math
    fine = _mixed_spec(robot)
    cover = {}
    for a, k in zip(fine["anchor"], range(len(fine["anchor"]))):
        ox, oy, oz = fine["offset"][3 * k:3 * k + 3]
        cover[a] = max(cover.get(a, 0.0), math.sqrt(ox * ox + oy * oy + oz * oz) + fine["radius"][k] + 0.005)
    b_anchor = sorted(cover)
    broad = {"name": "broad", "anchor": b_anchor, "offset": [0.0] * (3 * len(b_anchor)),
             "radius": [cover[a] for a in b_anchor], "self_cc_ranges": build_self_cc_ranges(robot, b_anchor)}
    return {"tiers": [broad, {"name": "fine", **fine}]}


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("threads", [32, 128])
def test_warp_matches_block_two_tier_cascade(tmp_path, threads):
    """G3: broad->fine cascade in the warp path (2 tiers) == block verdict, both branches hit."""
    robot = _robot()
    spec = _two_tier_spec(robot)
    build_dir = tmp_path / "warp_tiers"
    header = _gen_header(robot, build_dir, spec).read_text()
    assert "sphere_anchor_broad" in header and "w_broad" in header, "expected the 2-tier cascade emission"
    assert _ints(_baked(header, "sphere_anchor_broad")) == spec["tiers"][0]["anchor"]
    exe = _compile(build_dir)
    print(_run(exe, threads, 300, 11))

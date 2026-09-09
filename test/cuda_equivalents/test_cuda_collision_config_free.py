"""CUDA gate for W3 `grid_collision::config_free` (the namespace emitter).

Certifies the END-TO-END binding: the codegen emits a `grid_collision` namespace whose
`config_free` runs the W1b batched extractor (grid::multi_target_position_device) then the
static SDF checks (grid_collision_geometry.cuh). Self-consistent (no external oracle):
  * empty / far environment + tiny radii  => config_free == free
  * obstacle placed ON sphere 0           => config_free == in-collision
  * self-collision path (Increment 0): huge radii on a NON-ADJACENT sphere pair, empty env
    => config_free == in-collision (via grid_cc_self_collision); an ADJACENT-only pair
    (excluded from the baked ranges) => free, proving the adjacency exclusion through config_free.
The SDF math + baked-range self-collision are unit-tested by test_cuda_collision_geometry.py;
this gate covers the generated data tables + the extractor->config_free wiring.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._collision import build_self_cc_ranges
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from RBDReference.tests.model_sources import resolve_robot_spec, iter_robot_cases
from RBDReference.tests import MANIFEST_PATH
from RBDReference.equivalents.reference_backend import build_project_adapter

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import robot_urdf
COLLISION_INCLUDE = REPO_ROOT / "grid_codegen" / "collision"
RUNNER_SOURCE = Path(__file__).with_name("cuda_collision_config_free_runner.cu")
SELFCC_RUNNER_SOURCE = Path(__file__).with_name("cuda_collision_self_collision_runner.cu")


def _robot(robot_id="iiwa14", base_mode="fixed"):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            spec = case["spec"]
            break
    else:
        pytest.skip(f"{robot_id}-{base_mode} not in manifest")
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"could not resolve {robot_id}: {exc}")
    return build_project_adapter(spec, resolved, base_mode=base_mode).robot


def _collision_spec(robot):
    """One tiny sphere per movable joint frame (anchor = jid), small nonzero offset so the
    extractor's offset epilogue is exercised; radii tiny so no self-collision at the test q."""
    # get_joints_ordered_by_id() returns MOVABLE joints only (fixed joints are removed by
    # remove_fixed_joints); one sphere per movable frame.
    anchors = [int(j.get_id()) for j in robot.get_joints_ordered_by_id()]
    offset, radius = [], []
    for k, _a in enumerate(anchors):
        offset.extend([0.02 + 0.005 * k, -0.01, 0.03])
        radius.append(0.01)
    return {"anchor": anchors, "offset": offset, "radius": radius,
            "self_cc_ranges": build_self_cc_ranges(robot, anchors)}


def _two_sphere_spec(robot, anchor_a, anchor_b, radius):
    """A minimal two-sphere spec anchored on the two given movable joints (huge radius so the
    verdict is governed only by whether the pair is in the baked self_cc_ranges). Used to probe
    the self-collision path in isolation: a NON-adjacent pair collides; an ADJACENT pair is
    excluded from the ranges and stays free."""
    anchors = [int(anchor_a), int(anchor_b)]
    offset = [0.02, -0.01, 0.03, -0.02, 0.01, -0.03]
    return {"anchor": anchors, "offset": offset, "radius": [float(radius), float(radius)],
            "self_cc_ranges": build_self_cc_ranges(robot, anchors)}


def _gen_header(robot, build_dir, spec):
    build_dir.mkdir(parents=True, exist_ok=True)
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(robot, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        # SPLIT codegen: collision emission is collision_spec-driven; the list only
        # needs one ee key to satisfy the include_any_kinematics gate.
        codegen.gen_all_code(codegen_profile="kinematics", output_path=str(header), collision_spec=spec)
    return header


def _compile_and_run(build_dir, runner_source, extra_args=None):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / runner_source.name
    shutil.copyfile(runner_source, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / (runner_source.stem + ".exe")
    cmd = [nvcc, "-std=c++17", "-O2", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           "-I", str(build_dir), "-I", str(COLLISION_INCLUDE), "-o", str(exe), str(runner_copy)]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"{runner_source.name} compile FAILED.\ncmd: {' '.join(cmd)}\n"
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    run = subprocess.run([str(exe)] + [str(a) for a in (extra_args or [])], capture_output=True, text=True)
    assert run.returncode == 0, f"{runner_source.name} FAILED:\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    assert run.stdout.strip().endswith("RESULT: PASS"), run.stdout
    return run.stdout


def _parse_kv(stdout, key):
    for tok in stdout.split():
        if tok.startswith(key + "="):
            return int(tok.split("=", 1)[1])
    raise AssertionError(f"'{key}=' not found in runner stdout:\n{stdout}")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_collision_config_free(tmp_path):
    robot = _robot()
    spec = _collision_spec(robot)
    build_dir = tmp_path / "collision_config_free"
    _gen_header(robot, build_dir, spec)
    print(_compile_and_run(build_dir, RUNNER_SOURCE))


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_collision_config_free_real_robot(tmp_path):
    """Full AUTOMATED flow on a real, fully-covered robot: the custom spherizer converts go2's
    URDF collision geometry -> covering spheres -> collision_spec_from_urdf -> grid_collision::
    config_free that compiles and runs. go2 at its home config (q=0) is self-collision-free, so
    the config_free verdict is governed by the environment: empty/far => free, obstacle-on-sphere
    => in-collision. This is the end-to-end certification of the `--collision` pipeline."""
    from URDFParser import URDFParser
    from grid_codegen.algorithms._collision import collision_spec_from_urdf
    urdf = robot_urdf("go2")
    if not urdf.exists():
        pytest.skip("go2.urdf not found")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(urdf), floating_base=False)
        if robot is None:
            pytest.skip("go2 URDF parse failed")
        spec = collision_spec_from_urdf(robot, str(urdf), resolution=0.05)
    nq = len(robot.get_joints_ordered_by_id())  # movable dof (fixed-base) == NUM_POS
    build_dir = tmp_path / "collision_real_go2"
    _gen_header(robot, build_dir, spec)
    # home config (q=0) is self-collision-free -> the config_free runner's EMPTY/ONHIT/FAR gate holds.
    out = _compile_and_run(build_dir, RUNNER_SOURCE, extra_args=["0"] * nq)
    assert _parse_kv(out, "free") == 1  # first "free=" token is the EMPTY verdict
    print(out)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_collision_self_collision(tmp_path):
    """Increment 0: drive grid_cc_self_collision THROUGH config_free (empty env). A non-adjacent
    huge-radius pair must be flagged (config_free false); an adjacent-only pair is excluded from
    the baked ranges and stays free -- the range table's adjacency exclusion, proven end-to-end."""
    robot = _robot()

    # POSITIVE: full per-joint spec with huge radii on a NON-adjacent pair (iiwa14 is a serial
    # chain, so anchors 0 and 2 are non-adjacent) -> self-collision -> config_free == in-collision.
    pos_spec = _collision_spec(robot)
    pos_spec["radius"][0] = 1.0e3
    pos_spec["radius"][2] = 1.0e3
    assert pos_spec["self_cc_ranges"], "expected non-empty self_cc_ranges for the serial chain"
    pos_dir = tmp_path / "selfcc_positive"
    _gen_header(robot, pos_dir, pos_spec)
    pos_out = _compile_and_run(pos_dir, SELFCC_RUNNER_SOURCE)
    assert _parse_kv(pos_out, "empty_free") == 0, f"non-adjacent huge pair should self-collide:\n{pos_out}"

    # NEGATIVE (exclusion): two spheres on ADJACENT frames (anchors 0 and 1) with huge radii. The
    # only possible pair is adjacent -> build_self_cc_ranges emits ZERO ranges -> config_free free.
    neg_spec = _two_sphere_spec(robot, 0, 1, radius=1.0e3)
    assert not neg_spec["self_cc_ranges"], "adjacent-only pair must yield empty self_cc_ranges"
    neg_dir = tmp_path / "selfcc_negative"
    _gen_header(robot, neg_dir, neg_spec)
    neg_out = _compile_and_run(neg_dir, SELFCC_RUNNER_SOURCE)
    assert _parse_kv(neg_out, "empty_free") == 1, f"adjacent-excluded pair should stay free:\n{neg_out}"
    assert _parse_kv(neg_out, "NRANGES") == 0, f"adjacent-only pair must bake 0 ranges:\n{neg_out}"
    print(pos_out + neg_out)

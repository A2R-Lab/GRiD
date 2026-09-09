"""CUDA FD-oracle gate for the SELF-collision rows (GATO ask 2026-08-01).

`grid_cc_config_free` was boolean-only for robot-vs-robot; these emits are the self-pair
analogue of the env collision_distance[_gradient] family so a solver can bind self-collision
to its constraint-row mechanisms: `self_collision_distance[_gradient]` (per-sphere reduced-min
over the baked adjacency-excluded pair set + argmin-partner freeze seam) and
`self_collision_distance_pairs[_gradient]` (un-reduced, compile-time NUM_SELF_COLLISION_PAIRS).

Certifies on iiwa14 against central FD:
  1. the reduced value is BIT-EXACTLY the min over that sphere's un-reduced pair rows, and the
     argmin partner indexes a pair achieving it;
  2. the per-pair Jacobian d(d_p)/dq matches central FD (both endpoints move);
  3. the reduced gradient row is BIT-IDENTICAL to its argmin pair's row (orientation-invariant);
  4. the baked pair set is non-empty and well-formed.
See cuda_collision_selfcc_rows_runner.cu; SSOT = docs/open-tasks/gato_ask_self_collision_rows_2026-08-01.md.
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
from grid_codegen.algorithms._collision import collision_spec_from_urdf
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import robot_urdf
COLLISION_INCLUDE = REPO_ROOT / "grid_codegen" / "collision"
RUNNER_SOURCE = Path(__file__).with_name("cuda_collision_selfcc_rows_runner.cu")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_self_collision_rows_fd(tmp_path):
    from URDFParser import URDFParser
    urdf = robot_urdf("iiwa14")
    if not urdf.exists():
        pytest.skip("iiwa14.urdf not found")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(urdf), floating_base=False)
        if robot is None:
            pytest.skip("iiwa14 URDF parse failed")
        spec = collision_spec_from_urdf(robot, str(urdf), resolution=0.06)
        build_dir = tmp_path / "collision_selfcc_rows"
        build_dir.mkdir()
        header = build_dir / "grid.cuh"
        # SPLIT codegen: collision emission is collision_spec-driven; the list only
        # needs one ee key to satisfy the include_any_kinematics gate.
        GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
            codegen_profile="kinematics", output_path=str(header), collision_spec=spec)

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_collision_selfcc_rows_runner.exe"
    cmd = [nvcc, "-std=c++17", "-O2", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           "-I", str(build_dir), "-I", str(COLLISION_INCLUDE), "-I", str(REPO_ROOT),
           "-o", str(exe), str(runner_copy)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, f"nvcc build failed:\n{build.stderr}"

    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    assert "RESULT: PASS" in run.stdout, f"runner failed:\n{run.stdout}\n{run.stderr}"
    assert run.returncode == 0, f"runner exit {run.returncode}:\n{run.stdout}"

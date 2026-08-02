"""CUDA gate for W3 Increment 2: the broad->fine two-tier config_free.

Certifies that a multi-density collision codegen (broad + fine sphere tiers baked into ONE header)
produces a config_free whose verdict is BIT-IDENTICAL to a fine-only check on every (configuration,
obstacle) pair -- i.e. the coarse broad-phase reject is conservative (covering spheres enclose the
finer tier), so it can never miss a collision the fine tier would catch. Also exercises the
suffix-threaded multi_target emitter (multi_target_position_broad_* coexisting with the unsuffixed
fine batch). See cuda_collision_two_tier_runner.cu.
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
from grid_codegen.algorithms._collision import multi_tier_collision_spec_from_urdf
from test.cuda_equivalents.test_cuda_executable_equivalence import _detect_cuda_arch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import robot_urdf
COLLISION_INCLUDE = REPO_ROOT / "grid_codegen" / "collision"
RUNNER_SOURCE = Path(__file__).with_name("cuda_collision_two_tier_runner.cu")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_collision_two_tier_matches_fine(tmp_path):
    from URDFParser import URDFParser
    urdf = robot_urdf("iiwa14")
    if not urdf.exists():
        pytest.skip("iiwa14.urdf not found")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(urdf), floating_base=False)
        if robot is None:
            pytest.skip("iiwa14 URDF parse failed")
        # Two densities -> broad (0.10) + fine (0.06); config_free rejects on broad, confirms on fine.
        spec = multi_tier_collision_spec_from_urdf(robot, str(urdf), [0.10, 0.06])
        assert "tiers" in spec and len(spec["tiers"]) == 2, spec
        build_dir = tmp_path / "collision_two_tier"
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
    exe = build_dir / "cuda_collision_two_tier_runner.exe"
    cmd = [nvcc, "-std=c++17", "-O2", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           "-I", str(build_dir), "-I", str(COLLISION_INCLUDE), "-o", str(exe), str(runner_copy)]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"two-tier runner compile FAILED.\ncmd: {' '.join(cmd)}\nstderr:\n{result.stderr}")
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, f"two-tier runner FAILED:\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    assert run.stdout.strip().endswith("RESULT: PASS"), run.stdout
    print(run.stdout)

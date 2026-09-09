"""CUDA FD-oracle gate for the GATO Ask-3 collision surface: the Plane primitive + per-pair rows.

`collision_distance` min-reduces over the environment, and that argmin is non-smooth exactly where
the nearest obstacle switches -- so a solver wanting one smooth constraint row per (sphere, obstacle)
pair cannot use it. `collision_distance_pairs{,_gradient}` drop the reduction and emit the full
NUM_COLLISION_SPHERES x n_obs block; `Plane<T>` adds the half-space (ground) primitive.

Certifies on iiwa14 against central FD, with an environment holding ALL FOUR primitive kinds
(2 spheres + 1 capsule + 1 cuboid + 1 plane -> n_obs=5, every branch of the flattened obstacle index):
  1. min over a pair row == collision_distance's reduced value, BIT-EXACTLY (the un-reduced rows are a
     strict refinement of the existing API, not a drifting reimplementation);
  2. the per-pair Jacobian d(d_io)/dq matches central FD;
  3. the plane row is exactly n.p - d - r with the plane's own constant unit normal;
  4. the boolean (squared-gap) and signed plane SDFs agree in sign across the surface -- the
     clamp-to-excess step that keeps a center BELOW the plane from squaring away its sign.
See cuda_collision_pairs_runner.cu.
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
RUNNER_SOURCE = Path(__file__).with_name("cuda_collision_pairs_runner.cu")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_collision_distance_pairs_fd(tmp_path):
    from URDFParser import URDFParser
    urdf = robot_urdf("iiwa14")
    if not urdf.exists():
        pytest.skip("iiwa14.urdf not found")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(urdf), floating_base=False)
        if robot is None:
            pytest.skip("iiwa14 URDF parse failed")
        spec = collision_spec_from_urdf(robot, str(urdf), resolution=0.06)
        build_dir = tmp_path / "collision_pairs"
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
    exe = build_dir / "cuda_collision_pairs_runner.exe"
    cmd = [nvcc, "-std=c++17", "-O2", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           "-I", str(build_dir), "-I", str(COLLISION_INCLUDE), "-I", str(REPO_ROOT),
           "-o", str(exe), str(runner_copy)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, f"nvcc build failed:\n{build.stderr}"

    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    assert "RESULT: PASS" in run.stdout, f"runner failed:\n{run.stdout}\n{run.stderr}"
    assert run.returncode == 0, f"runner exit {run.returncode}:\n{run.stdout}"

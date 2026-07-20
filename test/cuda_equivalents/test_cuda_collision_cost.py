"""CUDA FD-oracle gate for W3 Increment 3: the differentiable collision path.

Certifies, against central finite differences on a real spherized robot (iiwa14), that the exposed
raw primitive `collision_distance_gradient` (per-sphere clearance Jacobian d(d_i)/dq) and the
assembled `collision_cost_gradient` match numerical derivatives, and that the GN
`collision_cost_hessian` is symmetric PSD. The SDF surface normals are unit-tested separately
(test_cuda_collision_geometry.py); this gate covers the normal x batched-position-gradient
composition + the cost assembly. See cuda_collision_cost_runner.cu.
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
from test.cuda_equivalents.test_cuda_executable_equivalence import _detect_cuda_arch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import robot_urdf
COLLISION_INCLUDE = REPO_ROOT / "grid_codegen" / "collision"
RUNNER_SOURCE = Path(__file__).with_name("cuda_collision_cost_runner.cu")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_collision_cost_fd(tmp_path):
    from URDFParser import URDFParser
    urdf = robot_urdf("iiwa14")
    if not urdf.exists():
        pytest.skip("iiwa14.urdf not found")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(urdf), floating_base=False)
        if robot is None:
            pytest.skip("iiwa14 URDF parse failed")
        spec = collision_spec_from_urdf(robot, str(urdf), resolution=0.06)
        build_dir = tmp_path / "collision_cost"
        build_dir.mkdir()
        header = build_dir / "grid.cuh"
        GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
            codegen_profile="all", output_path=str(header), collision_spec=spec)

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_collision_cost_runner.exe"
    cmd = [nvcc, "-std=c++17", "-O2", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           "-I", str(build_dir), "-I", str(COLLISION_INCLUDE), "-o", str(exe), str(runner_copy)]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"cost runner compile FAILED.\ncmd: {' '.join(cmd)}\nstderr:\n{result.stderr}")
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, f"cost runner FAILED:\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    assert run.stdout.strip().endswith("RESULT: PASS"), run.stdout
    print(run.stdout)

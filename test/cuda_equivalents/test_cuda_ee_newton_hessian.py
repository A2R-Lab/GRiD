"""Permanent regression gate for the full-Newton EE-position cost Hessian (PR #19).

The default `grid_plant::ee_pos_cost_hessian` is the TRUE analytic Hessian
(J_p^T W J_p + the residual-weighted EE-curvature term), with `GAUSS_NEWTON=true`
opting into the PSD J_p^T W J_p approximation. The plant-equivalence suite already
checks the GN path against the ratified J_p^T W J_p oracle; this cell covers the
NEW default path, which no NumPy oracle computes today, via a CUDA-side
self-check: the Newton Hessian must equal a finite-difference of the analytic
gradient, and GN must differ from it by exactly the dropped curvature term.

Fixed-base robots only (the runner FDs a q-perturbation, which needs
NUM_POS == NUM_VEL). Correctness only — no timing. Override the robot set with
GRID_CUDA_NEWTON_ROBOTS.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from GRiDCodeGenerator import GRiDCodeGenerator
from test.cuda_equivalents.test_cuda_executable_equivalence import _detect_cuda_arch
from RBDReference.tests.model_sources import resolve_robot_spec, iter_robot_cases
from RBDReference.tests import MANIFEST_PATH
from RBDReference.equivalents.reference_backend import build_project_adapter


RUNNER_SOURCE = Path(__file__).with_name("cuda_ee_newton_hessian_runner.cu")


def _robot_cells():
    override = os.environ.get("GRID_CUDA_NEWTON_ROBOTS")
    if override:
        return [(r.strip(), "fixed") for r in override.split(",") if r.strip()]
    # Small fixed-base robots — cheap, and enough to guard the emit. go2 is a
    # quadruped whose fixed-base model is still NUM_POS == NUM_VEL.
    return [("iiwa14", "fixed"), ("go2", "fixed")]


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not in manifest")


def _generate_header(project_model, build_dir):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(project_model.robot, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(codegen_profile="all", output_path=str(header))
    return header


def _compile_runner(build_dir):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_ee_newton_hessian_runner.exe"
    cmd = [
        nvcc, "-std=c++17", "-O2",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-I", str(build_dir), "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "cuda_ee_newton_hessian_runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("robot_id,base_mode", _robot_cells(), ids=lambda v: str(v))
def test_ee_pos_cost_hessian_newton_matches_fd(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)

    build_dir = tmp_path / f"{robot_id}_{base_mode}_ee_newton"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    executable = _compile_runner(build_dir)

    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, (
        "Newton EE-cost Hessian self-check FAILED.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # Surface the metrics in -s runs; the pass/fail is the runner's exit code.
    assert "PASS" in result.stdout, result.stdout

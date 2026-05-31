"""CUDA equivalence test for the generated general-frame geometric Jacobian
device kernel (E2).

Validates grid::frame_jacobian_device against the RBDReference numpy oracle
(`RBDReference.frame_jacobian`), which itself matches pinocchio's
getFrameJacobian / getJointJacobian to ~1e-14, for the three pinocchio
reference frames (LOCAL / WORLD / LOCAL_WORLD_ALIGNED).

The CUDA path is float32, so the comparison uses a float32-scale tolerance like
the other CUDA smoke tests. The frame target is the leaf joint id of each robot
(the project joint id passed straight through to the device as target_jid; the
numpy oracle is queried by the same joint's name).

Robots: iiwa14-fixed + go2-floating (override with
GRID_CUDA_FRAME_JAC_ROBOTS="iiwa14:fixed,go2:floating").
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from GRiDCodeGenerator import GRiDCodeGenerator
from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _build_cuda_samples,
    _detect_cuda_arch,
    _parse_runner_output,
    _run_runner,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter


RUNNER_SOURCE = Path(__file__).with_name("cuda_frame_jacobian_smoke_runner.cu")
_REF_FRAMES = (("J_local", "LOCAL"), ("J_world", "WORLD"), ("J_lwa", "LOCAL_WORLD_ALIGNED"))


def _robot_modes():
    raw = os.environ.get("GRID_CUDA_FRAME_JAC_ROBOTS", "iiwa14:fixed,go2:floating")
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        rid, _, mode = tok.partition(":")
        out.append((rid.strip(), (mode.strip() or "fixed")))
    return out


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not in manifest")


def _generate_header(project_model, build_dir):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(project_model.robot, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(algorithm_list=["frame_jacobian"], output_path=str(header))
    return header


def _compile_runner(build_dir):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists() and shutil.which("nvcc") is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_frame_jacobian_smoke_runner.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "GLASS" / "include"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-I{glass_inc}", "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA frame_jacobian smoke runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


def _stdin(target_jid, q):
    rows = [str(int(target_jid)),
            " ".join(f"{v:.9g}" for v in np.asarray(q, dtype=np.float32))]
    return "\n".join(rows) + "\n"


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_frame_jacobian_matches_reference(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_frame_jac"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    executable, cmd = _compile_runner(build_dir)

    robot = project_model.robot
    leaf_id = robot.get_leaf_nodes()[0]
    leaf_name = robot.get_joint_by_id(leaf_id).get_name()
    nv = project_model.nv

    samples = _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)
    rtol, atol = 2e-3, 2e-3

    def close(actual, expected, msg):
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64), expected,
            rtol=rtol, atol=max(atol, rtol * scale), err_msg=msg,
        )

    for sample in samples:
        q = np.asarray(sample.q, np.float64)
        out = _parse_runner_output(_run_runner(executable, _stdin(leaf_id, q), cmd))
        for block, ref_frame in _REF_FRAMES:
            J_ref = np.asarray(
                project_model.frame_jacobian(q, leaf_name, ref_frame), dtype=np.float64)
            J_cuda = out[block].reshape(6, nv, order="F")
            close(J_cuda, J_ref, f"{robot_id}-{base_mode} @ {sample.name} {ref_frame}")

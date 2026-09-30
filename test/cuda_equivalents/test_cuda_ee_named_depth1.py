"""BUG7 regression gate (GATO 2026-08-11): NAMED fixed-target EE pose whose
chain reaches the root EARLY (depth << n_bfs_levels).

go2-floating with fixed_target_name='imu_joint' is depth 1 of 4: the old
ping-pong chain walk in gen_end_effector_pose_inner `continue`d rooted columns
without carrying the live buffer across the swap, and the extractor read one
global final parity — so the imu pose came out BASE-RELATIVE (constant offset,
J == 0 downstream; the entire go2 EE tracking cost was a no-op). The gradient /
hessian inners were already immune (single-buffer s_Xworld BFS), which is
exactly why the existing named-target wrapper test — gradient/hessian only —
never fired.

This test drives grid::end_effector_pose_target_device on the device and
compares the [xyz; rpy] pose against the RBDReference oracle at random floating
configurations. With the pre-fix codegen the xyz rows are the constant
base-relative imu offset and fail immediately.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import (
    _build_cuda_samples,
    _detect_cuda_arch,
    _parse_runner_output,
    _run_runner,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter


RUNNER_SOURCE = Path(__file__).with_name("cuda_ee_named_depth1_runner.cu")

# (robot, base_mode, named fixed target with an early-rooted chain)
_CASES = [("go2", "floating", "imu_joint")]


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not in the robot manifest")


@pytest.mark.cuda_equivalence
@pytest.mark.floating_base
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode", "target_name"), _CASES,
                         ids=[f"{r}-{b}-{t}" for r, b, t in _CASES])
def test_cuda_ee_named_depth1_pose_matches_reference(tmp_path, robot_id, base_mode, target_name):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)

    build_dir = tmp_path / f"{robot_id}_{base_mode}_{target_name}"
    build_dir.mkdir()
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(project_model.robot, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(algorithm_list=["end_effector_pose"],
                             fixed_target_name=target_name,
                             output_path=str(header))

    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists() and shutil.which("nvcc") is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_ee_named_depth1_runner.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-I{glass_inc}", "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA ee_named_depth1 runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    samples = _build_cuda_samples(project_model, random_count=4, include_corner_samples=True)
    for sample in samples:
        q = np.asarray(sample.q, np.float64)
        stdin = " ".join(f"{v:.17g}" for v in q) + "\n"
        out = _parse_runner_output(_run_runner(executable, stdin, cmd))
        pose = out["pose"].reshape(-1)
        ref = np.asarray(
            project_model.end_effector_pose(q, target_name, None), dtype=np.float64
        ).reshape(-1)
        tag = f"{robot_id}-{base_mode} tgt={target_name} @ {sample.name}"
        # xyz: float32 chain-up on a depth-1 chain — tight.
        np.testing.assert_allclose(pose[:3], ref[:3], rtol=2e-4, atol=2e-4,
                                   err_msg=f"{tag} xyz")
        # rpy: skip near gimbal lock (|pitch| ~ pi/2), same policy as the
        # runtime-EE test; otherwise compare wrapped angle differences.
        if abs(abs(ref[4]) - np.pi / 2) > 1e-2:
            d = np.mod(pose[3:6] - ref[3:6] + np.pi, 2 * np.pi) - np.pi
            np.testing.assert_allclose(d, np.zeros(3), atol=5e-4,
                                       err_msg=f"{tag} rpy")

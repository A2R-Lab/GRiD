"""CUDA equivalence test for the generated frame velocity and acceleration
(item C of 0.8, 2026-11-03), against the RBDReference oracle that pinocchio's
getFrameVelocity / getFrameAcceleration / getFrameClassicalAcceleration verify
(RBDReference/tests/test_frame_velocity_acceleration_equivalence.py):

  * grid::frame_velocity_device      = J qd            (LOCAL / WORLD / LWA)
  * grid::frame_acceleration_device  = J qdd + Jdot qd (spatial in LOCAL / WORLD,
                                        classical in LOCAL_WORLD_ALIGNED)

The target is each robot's leaf joint; the thread count is swept and the
outputs must be bit-identical across counts. Robots default to iiwa14, go2,
gen3 and the mimic fr3 on both bases; narrow with
GRID_CUDA_FRAME_KIN_ROBOTS="iiwa14:fixed,go2:floating".
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
from test.cuda_equivalents.executable_cache import cached_nvcc_executable
from test.cuda_equivalents.thread_invariance import thread_sweep
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter

RUNNER_SOURCE = Path(__file__).with_name("cuda_frame_kinematics_smoke_runner.cu")
_REF_FRAMES = (("local", "LOCAL"), ("world", "WORLD"), ("lwa", "LOCAL_WORLD_ALIGNED"))
_ALGO_KEYS = ["frame_velocity", "frame_acceleration"]
_DEFAULT_ROBOTS = "iiwa14:fixed,iiwa14:floating,go2:fixed,go2:floating,gen3:fixed,gen3:floating,fr3:fixed,fr3:floating"


def _robot_modes():
    raw = os.environ.get("GRID_CUDA_FRAME_KIN_ROBOTS", _DEFAULT_ROBOTS)
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            rid, _, mode = tok.partition(":")
            out.append((rid.strip(), mode.strip() or "fixed"))
    return out


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    raise RuntimeError(f"{robot_id}-{base_mode} not in manifest")


def _generate_header(project_model, build_dir):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(project_model.robot, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(algorithm_list=_ALGO_KEYS, output_path=str(header), enable_mujoco_kernels=False)
    return header


def _stdin(target_jid, q, qd, qdd):
    rows = [str(int(target_jid))] + [" ".join(f"{v:.9g}" for v in np.asarray(x, dtype=np.float32)) for x in (q, qd, qdd)]
    return "\n".join(rows) + "\n"


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _robot_modes(), ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_frame_kinematics_matches_reference(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    resolved = resolve_robot_spec(spec)
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_frame_kin"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    arch = _detect_cuda_arch()
    exe, cmd = cached_nvcc_executable(
        sources=[RUNNER_SOURCE, build_dir / "grid.cuh"],
        flags=["-std=c++17", "-O0", "-gencode", f"arch=compute_{arch},code=sm_{arch}", f"-I{glass_inc}"],
        exe_name="cuda_frame_kinematics_smoke_runner", fallback_dir=build_dir, include_dirs=[build_dir],
        what=f"frame kinematics runner {robot_id}-{base_mode}")

    robot = project_model.robot
    leaf_id = robot.get_leaf_nodes()[0]
    leaf_name = robot.get_joint_by_id(leaf_id).get_name()
    samples = _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)

    def close(actual, expected, msg, rtol, atol):
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), expected,
                                   rtol=rtol, atol=max(atol, rtol * scale), err_msg=msg)

    for sample in samples:
        q, qd, qdd = (np.asarray(x, np.float64) for x in (sample.q, sample.qd, sample.qdd))
        stdin = _stdin(leaf_id, q, qd, qdd)
        expected = {}
        for blk, ref_frame in _REF_FRAMES:
            expected["V_" + blk] = np.asarray(project_model.frame_velocity(q, qd, leaf_name, ref_frame), np.float64)
            expected["A_" + blk] = np.asarray(project_model.frame_acceleration(q, qd, qdd, leaf_name, ref_frame), np.float64)

        def run(threads):
            out = _parse_runner_output(_run_runner(exe, stdin, cmd, num_threads=threads))
            return np.concatenate([out[k].reshape(-1) for k in sorted(expected)])

        def check(values, threads):
            offset = 0
            for k in sorted(expected):
                got = values[offset:offset + 6]; offset += 6
                tol = 2e-3 if k.startswith("V_") else 5e-3   # float32 kernels; the acceleration sums two products
                close(got, expected[k], f"{k} {robot_id}-{base_mode} @ {sample.name} threads={threads}", tol, tol)

        thread_sweep((1, 32, 128, 256), run, check)

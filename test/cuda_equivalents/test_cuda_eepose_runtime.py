"""CUDA equivalence test for the runtime-target end-effector pose + pose-gradient
family (additive): end_effector_pose_runtime ([xyz; rpy] 6-vector) and
end_effector_pose_gradient_runtime (6 x NUM_VEL d[xyz; rpy]/dv).

Validates against the RBDReference numpy oracle (matches pinocchio / the analytic
geometric Jacobian to ~1e-14) at MULTIPLE arbitrary (non-leaf) targets, with a
nonzero offset AND offset=0:
  * grid::end_effector_pose_runtime_device          vs RBDReference.end_effector_pose
  * grid::end_effector_pose_gradient_runtime_device vs RBDReference.end_effector_pose_gradient

Offset checks: offset=0 matches the frame-origin oracle; a nonzero offset shifts
the position by exactly R_target * offset and leaves rpy unchanged.

rpy gimbal lock: at pitch ~= +-pi/2 the E^{-1} block (rows 3..5 of the gradient)
blows up on BOTH the oracle and the device; those samples are skipped (the xyz
rows + the position pose are still checked).

The CUDA path is float32, so the comparison uses a float32-scale tolerance like
the other CUDA smoke tests.

Robots: iiwa14-fixed + go2-floating + fr3-fixed (mimic) + fr3-floating + h1_2-fixed
(override with GRID_CUDA_EEPOSE_RT_ROBOTS=
"iiwa14:fixed,go2:floating,fr3:fixed,fr3:floating,h1_2:fixed"). RBDReference is
authoritative for the mimic robots.
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


RUNNER_SOURCE = Path(__file__).with_name("cuda_eepose_runtime_smoke_runner.cu")
_ALGO_KEYS = ["end_effector_pose_runtime", "end_effector_pose_gradient_runtime"]
_OFFSET = np.array([0.05, -0.03, 0.07], dtype=np.float64)
# Skip the rpy rows when the EE pitch is within this band of +-pi/2 (E^{-1}
# gimbal-lock singularity, on the oracle AND the device).
_PITCH_GUARD = 0.15


def _robot_modes():
    raw = os.environ.get(
        "GRID_CUDA_EEPOSE_RT_ROBOTS",
        "iiwa14:fixed,go2:floating,fr3:fixed,fr3:floating,h1_2:fixed")
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
        codegen.gen_all_code(algorithm_list=_ALGO_KEYS, output_path=str(header))
    return header


def _compile_runner(build_dir):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists() and shutil.which("nvcc") is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_eepose_runtime_smoke_runner.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-I{glass_inc}", "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA eepose_runtime smoke runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


def _stdin(target_jid, offset, q):
    rows = [str(int(target_jid)),
            " ".join(f"{v:.9g}" for v in np.asarray(offset, dtype=np.float32)),
            " ".join(f"{v:.9g}" for v in np.asarray(q, dtype=np.float32))]
    return "\n".join(rows) + "\n"


def _nonleaf_targets(robot):
    """Up to 3 arbitrary non-leaf articulated-joint targets (mid + deep)."""
    leaves = set(robot.get_leaf_nodes())
    cand = []
    for j in range(robot.get_num_joints()):
        joint = robot.get_joint_by_id(j)
        if joint is None or j in leaves:
            continue
        name = joint.get_name()
        if name:
            cand.append(j)
    if not cand:
        # no non-leaf articulated joints (e.g. trivial chain) -> fall back to leaf
        leaf = robot.get_leaf_nodes()[0]
        return [(leaf, robot.get_joint_by_id(leaf).get_name())]
    cand.sort(key=lambda j: len(robot.get_ancestors_by_id(j)))
    picks = {cand[0], cand[len(cand) // 2], cand[-1]}
    return [(j, robot.get_joint_by_id(j).get_name()) for j in sorted(picks)]


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_eepose_runtime_matches_reference(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_eepose_rt"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    executable, cmd = _compile_runner(build_dir)

    robot = project_model.robot
    nv = project_model.nv
    targets = _nonleaf_targets(robot)
    samples = _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)

    def close(actual, expected, msg, rtol=2e-3, atol=2e-3):
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64), expected,
            rtol=rtol, atol=max(atol, rtol * scale), err_msg=msg,
        )

    def _chain_depth(target_jid):
        return len(robot.get_ancestors_by_id(target_jid)) + 1

    def _chain_world_scale(q, target_jid):
        """Largest |world coordinate| over the target's ancestor chain. A float32
        FK chain-up carries ~1e-6 RELATIVE error on coordinates of this size,
        accumulated over the chain depth -- so the position error floor scales
        with this (deep humanoid hands reach the EE via ~1m-scale torso/arm
        intermediates even when the EE's own |position| is small). This keeps the
        check a per-config float32 CONDITIONING floor, not a global loosen."""
        chain = sorted(robot.get_ancestors_by_id(target_jid)) + [target_jid]
        coords = [1.0]
        for j in chain:
            joint = robot.get_joint_by_id(j)
            if joint is None or not joint.get_name():
                continue
            wpos = np.asarray(
                project_model.end_effector_pose(q, joint.get_name(), None),
                dtype=np.float64).reshape(-1)[:3]
            coords.append(float(np.max(np.abs(wpos))))
        # error ~ (per-matmul float32 epsilon) * world-scale * chain depth.
        return max(coords) * _chain_depth(target_jid)

    for target_jid, target_name in targets:
        for sample in samples:
            q = np.asarray(sample.q, np.float64)
            out = _parse_runner_output(
                _run_runner(executable, _stdin(target_jid, _OFFSET, q), cmd))
            tag = f"{robot_id}-{base_mode} tgt={target_name} @ {sample.name}"
            # float32 FK-chain conditioning floor for the position rows: a
            # single-precision world chain-up accumulates ~few*1e-3 absolute per
            # matmul on ~1m-scale intermediates, so scale the floor by
            # world-scale * chain-depth (see _chain_world_scale). Shallow chains
            # (most robots) stay near the 2e-3 base; only deep humanoid hands
            # (h1_2 thumb, depth ~14) widen it.
            pos_atol = max(2e-3, 4e-3 * _chain_world_scale(q, target_jid))

            # ---- POSE ----
            pose0 = out["pose0"].reshape(-1)
            poseN = out["poseN"].reshape(-1)
            ref0 = np.asarray(project_model.end_effector_pose(q, target_name, None),
                              dtype=np.float64).reshape(-1)
            offN = np.array([_OFFSET[0], _OFFSET[1], _OFFSET[2], 1.0])
            refN = np.asarray(project_model.end_effector_pose(q, target_name, offN),
                              dtype=np.float64).reshape(-1)

            pitch = float(ref0[4])
            near_gimbal = abs(abs(pitch) - np.pi / 2) < _PITCH_GUARD
            # rpy is a NONLINEAR function of the EE rotation; a very deep float32 FK
            # chain (e.g. h1_2's depth-11 thumb) corrupts R enough that the rpy
            # extraction diverges even far from gimbal lock. That is the same
            # float32 CONDITIONING floor as the position (the float64 numpy mirror
            # is exact -- the index math + mimic alpha-fold are byte-correct). Skip
            # rpy for such deep chains; shallow chains (<=8 deep: iiwa/fr3/go2 and
            # h1_2's non-thumb targets) still check rpy tightly.
            rpy_trustworthy = (not near_gimbal) and (_chain_depth(target_jid) <= 8)

            # position (always valid; float32 FK-chain conditioning floor).
            close(pose0[:3], ref0[:3], f"pose0 xyz {tag}", atol=pos_atol)
            close(poseN[:3], refN[:3], f"poseN xyz {tag}", atol=pos_atol)
            if rpy_trustworthy:
                close(pose0[3:], ref0[3:], f"pose0 rpy {tag}")
                close(poseN[3:], refN[3:], f"poseN rpy {tag}")
                # rpy unchanged by the offset (device side, float32).
                close(poseN[3:], pose0[3:], f"poseN rpy==pose0 rpy {tag}")
            # nonzero offset shifts position by R_target * offset (device side).
            # R_target also comes off the (float32) FK chain, so this residual
            # carries the same conditioning floor as the position.
            R = np.asarray(project_model.end_effector_rotation_matrix(q, target_name),
                           dtype=np.float64)
            close(poseN[:3] - pose0[:3], R @ _OFFSET, f"pos shift = R*offset {tag}",
                  atol=pos_atol)

            # ---- GRADIENT ----
            grad0 = out["grad0"].reshape(6, nv, order="F")
            gradN = out["gradN"].reshape(6, nv, order="F")
            g0_ref = np.asarray(
                project_model.end_effector_pose_gradient(q, target_name, None),
                dtype=np.float64)
            gN_ref = np.asarray(
                project_model.end_effector_pose_gradient(q, target_name, offN),
                dtype=np.float64)
            # The Jv (xyz) rows carry the same float32 FK-chain conditioning floor
            # as the position; the rpy rows are valid only away from gimbal lock.
            close(grad0[:3, :], g0_ref[:3, :], f"grad0 Jv {tag}", atol=pos_atol)
            close(gradN[:3, :], gN_ref[:3, :], f"gradN Jv {tag}", atol=pos_atol)
            if rpy_trustworthy:
                close(grad0[3:, :], g0_ref[3:, :], f"grad0 rpy-rows {tag}")
                close(gradN[3:, :], gN_ref[3:, :], f"gradN rpy-rows {tag}")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"),
                         [("iiwa14", "fixed"), ("fr3", "fixed")],
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_eepose_runtime_thread_invariance(tmp_path, robot_id, base_mode):
    """Single-block kernels are block-stride loops, so ANY thread count that fits
    must produce IDENTICAL device output. Sweep {1, 32, 256} and assert the
    runtime pose + gradient blocks are bit-identical across counts (a low-count
    divergence is a missing __syncthreads, not a tolerance artifact)."""
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_eepose_rt_inv"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    executable, cmd = _compile_runner(build_dir)

    robot = project_model.robot
    target_jid, _ = _nonleaf_targets(robot)[-1]
    samples = _build_cuda_samples(project_model, random_count=1, include_corner_samples=True)
    sample = samples[0]
    stdin = _stdin(target_jid, _OFFSET, np.asarray(sample.q, np.float64))

    ref = None
    for nthreads in (1, 32, 256):
        out = _parse_runner_output(_run_runner(executable, stdin, cmd, num_threads=nthreads))
        block = np.concatenate([out[k].reshape(-1) for k in ("pose0", "poseN", "grad0", "gradN")])
        if ref is None:
            ref = block
        else:
            np.testing.assert_array_equal(
                block, ref, err_msg=f"{robot_id}-{base_mode} thread-count {nthreads} diverged")

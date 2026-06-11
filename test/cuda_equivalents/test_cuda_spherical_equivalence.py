"""End-to-end CUDA verification of the SPHERICAL (ball) joint inverse_dynamics
(Phase-6 Tier-C, first slice).

A spherical joint is a 3-DoF manifold joint: NV=3 (body-frame angular velocity),
NQ=4 (unit quaternion xyzw, pinocchio JointModelSpherical convention), so NQ!=NV
exactly like the floating-base free-flyer's rotation sub-block. This test codegens
``inverse_dynamics`` for the spherical fixtures (the ONLY algorithm ported for
spherical so far), drives the dedicated ``cuda_spherical_runner.cu``, and asserts
the CUDA RNEA (gravity + Coriolis, qdd=0) matches the verified RBDReference numpy
reference (itself validated against pinocchio in
``URDFParser/tests/test_spherical_joint*``).

It exercises BOTH CUDA surfaces:
  * the device function ``inverse_dynamics_device`` (explicit nq-wide s_q /
    nv-wide s_qd buffers), at thread counts {1, 32, 256} for invariance; and
  * the HOST batch wrapper ``inverse_dynamics<T,false,true>`` over a 4-timestep
    trajectory (the per-timestep NQ-wide input-slot path the bindings use — the
    §1e nq-stride check: every batch row must equal the single-call device row).

Fixtures: spherical_arm (root spherical + revolute) and mixed_spherical_arm
(revolute -> spherical -> revolute, the mid-chain case that shifts every
downstream q/v offset).
"""
import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from URDFParser import URDFParser
from RBDReference import RBDReference
from GRiDCodeGenerator import GRiDCodeGenerator
from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _detect_cuda_arch,
    _parse_runner_output,
    GPU_UNAVAILABLE_PATTERNS,
)

RUNNER_SOURCE = Path(__file__).with_name("cuda_spherical_runner.cu")
FIXDIR = Path(__file__).resolve().parents[2] / "URDFParser" / "tests" / "fixtures"

# Each fixture's quaternion-block start index in q (the spherical joint's first
# q slot). spherical_arm: ball at jid0 -> q[0:4]. mixed_spherical_arm: revolute
# then ball at jid1 -> q[1:5].
QUAT_START = {"spherical_arm.urdf": 0, "mixed_spherical_arm.urdf": 1}


def _parse(name):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return URDFParser().parse(str(FIXDIR / name), floating_base=False)


def _generate_header(robot, build_dir):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid"
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        # inverse_dynamics + crba are the ported algorithms for spherical (Tier-C);
        # requesting any other raises NotImplementedError by design.
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header),
            algorithm_list=["inverse_dynamics", "crba"],
        )
    return header


def _compile_runner(build_dir):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_spherical_runner.exe"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-DGRID_CUDA_FLOATING_BASE=0",
        "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-gencode", f"arch=compute_{arch},code=compute_{arch}",
        "-o", str(exe), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA spherical runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return exe


def _run(exe, q, qd, threads=32):
    def row(v):
        return " ".join(f"{x:.9g}" for x in np.asarray(v, dtype=np.float64))
    stdin = "\n".join([row(q), row(qd)]) + "\n"
    result = subprocess.run(
        [str(exe), str(threads)], input=stdin, cwd=exe.parent,
        capture_output=True, text=True
    )
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0:
        if any(p in combined for p in GPU_UNAVAILABLE_PATTERNS):
            pytest.skip("CUDA runtime unavailable.")
        pytest.fail(f"runner failed:\n{result.stdout}\n{result.stderr}")
    return _parse_runner_output(result.stdout)


def _random_q(robot, fixture, rng):
    """Random configuration with a UNIT quaternion in the spherical q-block."""
    nq = robot.get_num_pos()
    q = rng.uniform(-1.0, 1.0, nq)
    qs = QUAT_START[fixture]
    quat = rng.uniform(-1.0, 1.0, 4)
    quat /= np.linalg.norm(quat)
    q[qs:qs + 4] = quat
    return q


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_inverse_dynamics_matches_reference(tmp_path, fixture):
    """CUDA spherical inverse_dynamics must match the RBDReference numpy oracle
    (gravity+Coriolis, qdd=0) on both the device function and the host batch
    wrapper, and be thread-count invariant + batch self-consistent (§1e)."""
    robot = _parse(fixture)
    assert robot is not None
    ref = RBDReference(robot)
    nv = robot.get_num_vel()

    exe = _compile_runner(tmp_path) if _generate_header(robot, tmp_path) else None

    rng = np.random.default_rng(7)
    zeros = np.zeros(nv, dtype=np.float64)
    failures = []
    for trial in range(4):
        q = _random_q(robot, fixture, rng)
        qd = rng.uniform(-0.8, 0.8, nv)
        out = _run(exe, q, qd)
        tag = f"{fixture} trial {trial}"

        ref_c = np.asarray(
            ref.inverse_dynamics(q, qd, zeros, GRAVITY=-9.81)[0], dtype=np.float64
        ).reshape(-1)

        cuda_dev = np.asarray(out["inverse_dynamics"], dtype=np.float64).reshape(-1)
        if cuda_dev.shape != ref_c.shape:
            failures.append(f"{tag} device: shape {cuda_dev.shape} != {ref_c.shape}")
        elif not np.allclose(cuda_dev, ref_c, atol=1e-4, rtol=1e-4):
            failures.append(
                f"{tag} device: max|d|={np.max(np.abs(cuda_dev - ref_c)):.3e}\n"
                f"  cuda={cuda_dev}\n  ref ={ref_c}")

        # Host batch wrapper: every timestep row must equal the oracle AND the
        # device single-call result (catches the §1e nq-stride bug).
        for k in range(4):
            row = np.asarray(out[f"inverse_dynamics_batch_{k}"], dtype=np.float64).reshape(-1)
            if not np.allclose(row, ref_c, atol=1e-4, rtol=1e-4):
                failures.append(
                    f"{tag} batch[{k}] vs ref: max|d|={np.max(np.abs(row - ref_c)):.3e}")
            if not np.allclose(row, cuda_dev, atol=1e-5, rtol=1e-5):
                failures.append(
                    f"{tag} batch[{k}] vs device: max|d|={np.max(np.abs(row - cuda_dev)):.3e}")

        # --- crba: mass matrix M (NV x NV) vs RBDReference oracle ---
        ref_M = np.asarray(ref.crba(q), dtype=np.float64)
        assert ref_M.shape == (nv, nv), f"{tag} oracle M shape {ref_M.shape} != {(nv, nv)}"
        cuda_M = np.asarray(out["crba"], dtype=np.float64)
        if cuda_M.size != nv * nv:
            failures.append(f"{tag} crba device: size {cuda_M.size} != {nv*nv}")
        else:
            # CUDA M is column-major NV x NV; reshape to compare with row-major oracle.
            cuda_M = cuda_M.reshape(nv, nv, order="F")
            if not np.allclose(cuda_M, ref_M, atol=1e-4, rtol=1e-4):
                failures.append(
                    f"{tag} crba device: max|d|={np.max(np.abs(cuda_M - ref_M)):.3e}\n"
                    f"  cuda=\n{cuda_M}\n  ref =\n{ref_M}")
            for k in range(4):
                blk = np.asarray(out[f"crba_batch_{k}"], dtype=np.float64).reshape(nv, nv, order="F")
                if not np.allclose(blk, ref_M, atol=1e-4, rtol=1e-4):
                    failures.append(
                        f"{tag} crba batch[{k}] vs ref: max|d|={np.max(np.abs(blk - ref_M)):.3e}")
                if not np.allclose(blk, cuda_M, atol=1e-5, rtol=1e-5):
                    failures.append(
                        f"{tag} crba batch[{k}] vs device: max|d|={np.max(np.abs(blk - cuda_M)):.3e}")

    assert not failures, "spherical CUDA equivalence failures:\n" + "\n".join(failures)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_thread_invariant(tmp_path, fixture):
    """The single-block spherical inverse_dynamics kernel must be exactly
    thread-count invariant (bit-identical at 1 / 32 / 256 threads)."""
    robot = _parse(fixture)
    assert robot is not None
    exe = _compile_runner(tmp_path) if _generate_header(robot, tmp_path) else None

    rng = np.random.default_rng(11)
    q = _random_q(robot, fixture, rng)
    qd = rng.uniform(-0.8, 0.8, robot.get_num_vel())

    base_out = _run(exe, q, qd, threads=32)
    base = np.asarray(base_out["inverse_dynamics"], dtype=np.float64)
    base_M = np.asarray(base_out["crba"], dtype=np.float64)
    failures = []
    for threads in (1, 32, 256):
        out = _run(exe, q, qd, threads=threads)
        dev = np.asarray(out["inverse_dynamics"], dtype=np.float64)
        if not np.array_equal(dev, base):
            failures.append(
                f"{fixture} threads={threads}: device differs from threads=32 "
                f"(max|d|={np.max(np.abs(dev - base)):.3e})")
        # batch rows must also equal the device single-call result at every count
        for k in range(4):
            row = np.asarray(out[f"inverse_dynamics_batch_{k}"], dtype=np.float64)
            if not np.array_equal(row, dev):
                failures.append(
                    f"{fixture} threads={threads}: batch[{k}] != device single-call")
        # crba mass matrix must be bit-identical across thread counts + batch rows
        devM = np.asarray(out["crba"], dtype=np.float64)
        if not np.array_equal(devM, base_M):
            failures.append(
                f"{fixture} threads={threads}: crba device differs from threads=32 "
                f"(max|d|={np.max(np.abs(devM - base_M)):.3e})")
        for k in range(4):
            blk = np.asarray(out[f"crba_batch_{k}"], dtype=np.float64)
            if not np.array_equal(blk, devM):
                failures.append(
                    f"{fixture} threads={threads}: crba batch[{k}] != device single-call")

    assert not failures, "spherical thread-invariance failures:\n" + "\n".join(failures)

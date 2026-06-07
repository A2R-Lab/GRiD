"""End-to-end CUDA verification of BRANCHING-skew inverse_dynamics / aba / minv / crba.

joint-1b ported aba/minv/inverse_dynamics to Tier-B (dense 6-vector S) for skew
robots but their Tier-B branches asserted a single joint per BFS level (serial),
so a skew robot
with a BRANCHING topology (2+ joints sharing a parent / at the same BFS level)
hit a codegen AssertionError. That restriction is now lifted: the Tier-B dense-S
emit loops over every joint in the level (each with its own dense S), mirroring
the cardinal multi-joint-per-level structure.

This test codegens the dedicated Y-shaped branching-skew fixture (joint_2 and
joint_3 share parent link1 -> same BFS level, all four joints skew), drives a
minimal aba/minv/crba runner, and asserts the CUDA outputs match the
RBDReference numpy oracle -- which is itself validated against pinocchio's
JointModelRevoluteUnaligned / JointModelPrismaticUnaligned in GRiD joint order
(test_branching_skew_axis.py / the helical+skew pin equivalence pattern).

The serial skew fixture (skew_axis_arm.urdf) is also generated through the same
path as a no-regression guard on the single-joint-per-level case.
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

RUNNER_SOURCE = Path(__file__).with_name("cuda_branching_skew_runner.cu")
FIXDIR = Path(__file__).resolve().parents[2] / "URDFParser" / "tests" / "fixtures"
GLASS_INCLUDE = Path(__file__).resolve().parents[2] / "GLASS" / "include"

pytestmark = [
    pytest.mark.cuda_equivalence,
    pytest.mark.developer_only,
    pytest.mark.robot_smoke,
]


def _parse(name):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return URDFParser().parse(str(FIXDIR / name), floating_base=False)


def _generate_header(robot, build_dir):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid"
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header),
            algorithm_list="inverse_dynamics,aba,minv,crba",
        )
    return header


def _compile_runner(build_dir):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_branching_skew_runner.exe"
    cmd = [
        nvcc, "-std=c++11", "-O0",
        "-DGRID_CUDA_FLOATING_BASE=0",
        "-DGRID_RUNNER_SKIP_GRADIENTS=1",
        "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
        "-I", str(GLASS_INCLUDE),
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-gencode", f"arch=compute_{arch},code=compute_{arch}",
        "-o", str(exe), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA branching-skew runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return exe


def _run(exe, q, qd, u, qdd):
    def row(v):
        return " ".join(f"{x:.9g}" for x in np.asarray(v, dtype=np.float32))
    stdin = "\n".join([row(q), row(qd), row(u), row(qdd)]) + "\n"
    result = subprocess.run(
        [str(exe)], input=stdin, cwd=exe.parent, capture_output=True, text=True
    )
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0:
        if any(p in combined for p in GPU_UNAVAILABLE_PATTERNS):
            pytest.skip("CUDA runtime unavailable.")
        pytest.fail(f"runner failed:\n{result.stdout}\n{result.stderr}")
    return _parse_runner_output(result.stdout)


def _check(failures, label, cuda, ref, atol=2e-4, rtol=2e-4):
    cuda = np.asarray(cuda, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    if cuda.shape != ref.shape:
        failures.append(f"{label}: shape {cuda.shape} != {ref.shape}")
        return
    if not np.allclose(cuda, ref, atol=atol, rtol=rtol):
        failures.append(
            f"{label}: max|d|={np.max(np.abs(cuda - ref)):.3e}\n  cuda={cuda}\n  ref ={ref}")


@pytest.mark.parametrize(
    "fixture",
    ["branching_skew_arm.urdf", "skew_axis_arm.urdf"],
    ids=["branching", "serial"],
)
def test_cuda_skew_aba_minv_matches_reference(tmp_path, fixture):
    """CUDA aba / minv (and crba) for a skew robot must match the RBDReference
    numpy oracle. ``branching_skew_arm`` exercises the lifted multi-joint-per-BFS
    -level Tier-B emit; ``skew_axis_arm`` is the serial-skew no-regression guard.
    """
    robot = _parse(fixture)
    assert robot is not None
    assert robot.robot_has_skew_axis(), f"{fixture} should have skew axes"
    ref = RBDReference(robot)
    n = robot.get_num_vel()

    _generate_header(robot, tmp_path)
    exe = _compile_runner(tmp_path)

    rng = np.random.default_rng(7)
    failures = []
    for trial in range(4):
        q = rng.uniform(-0.5, 0.5, n)
        qd = rng.uniform(-0.8, 0.8, n)
        tau = rng.uniform(-0.5, 0.5, n)
        qdd = rng.uniform(-0.6, 0.6, n)
        out = _run(exe, q, qd, tau, qdd)
        tag = f"{fixture} trial {trial}"

        # inverse_dynamics: tau = ID(q, qd, qdd) -- exercises the lifted Tier-B
        # forward (S*qd/S*qdd) + mxS branches on the branching topology.
        cuda_id = np.asarray(out["inverse_dynamics"], dtype=np.float64).reshape(-1)
        ref_id = np.asarray(
            ref.inverse_dynamics(q, qd, qdd)[0], dtype=np.float64).reshape(-1)
        _check(failures, f"{tag} inverse_dynamics", cuda_id, ref_id)

        # aba: qdd = ABA(q, qd, tau)
        cuda_aba = np.asarray(out["aba"], dtype=np.float64).reshape(-1)
        ref_aba = np.asarray(ref.aba(q, qd, tau), dtype=np.float64).reshape(-1)
        _check(failures, f"{tag} aba", cuda_aba, ref_aba)

        # minv: the kernel fills the upper triangle; compare upper triangles.
        cuda_minv = np.asarray(out["minv"], dtype=np.float64).reshape(n, n, order="F")
        ref_minv = np.asarray(ref.minv(q), dtype=np.float64)
        _check(failures, f"{tag} minv(upper)",
               np.triu(cuda_minv), np.triu(ref_minv))

        # crba: M (sanity / cross-check; crba Tier-B already branching-safe)
        cuda_m = np.asarray(out["crba"], dtype=np.float64).reshape(n, n, order="F")
        ref_m = np.asarray(ref.crba(q), dtype=np.float64)
        _check(failures, f"{tag} crba", cuda_m, ref_m)

    assert not failures, "branching-skew CUDA equivalence failures:\n" + "\n".join(failures)

"""CUDA equivalence for the SPHERICAL (ball) joint integrator GRADIENT.

The 2026-07-30 slice: `integrator_gradient` / `integrator_with_gradient` for
fixed-base spherical robots. The q-block top rows of dAB carry per-spherical-
joint 3x3 SO(3) dIntegrate blocks (ARG_q = exp(-phi), ARG_v = J_r(phi)) —
the omega-only restriction of the floating free-flyer machinery — precomputed
row-major into the repurposed s_dInt_*_6x6 buffers and consumed block-
diagonally by the dAB assembly. Single-stage IntegratorTypes only (EULER /
SEMI_IMPLICIT_EULER / TRAPEZOIDAL; multi-stage RK static_asserts — follow-on).

Oracle: RBDReference.integrator_gradient, which routes every q-block through
dIntegrate (spherical branch pinned against pinocchio's ARG0/ARG1 semantics by
the RBDReference test suite). Compared at fp32 + fp64, thread counts
{1, 32, 256} (thread-count invariance), on both spherical fixtures.
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
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import (
    _detect_cuda_arch,
    _parse_runner_output,
    GPU_UNAVAILABLE_PATTERNS,
)

pytestmark = [
    pytest.mark.cuda_equivalence,
    pytest.mark.developer_only,
    pytest.mark.robot_smoke,
]

RUNNER_SOURCE = Path(__file__).with_name("cuda_spherical_integrator_gradient_runner.cu")
FIXDIR = Path(__file__).resolve().parents[2] / "external" / "URDFParser" / "tests" / "fixtures"

QUAT_START = {"spherical_arm.urdf": 0, "mixed_spherical_arm.urdf": 1}
DT = 0.01

INTEGRATORS = (
    ("integrator_euler", "euler"),
    ("integrator_si_euler", "semi_implicit_euler"),
    ("integrator_trapezoidal", "trapezoidal"),
)


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
            algorithm_list=["integrator", "integrator_gradient", "integrator_with_gradient"],
        )
    return header


def _compile_runner(build_dir, equiv_t):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / f"cuda_spherical_integrator_gradient_runner_{equiv_t}.exe"
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
            "CUDA spherical integrator gradient runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return exe


def _run(exe, q, qd, u, dt, threads=32, equiv_t="float"):
    def row(v):
        return " ".join(f"{x:.12g}" for x in np.asarray(v, dtype=np.float64))
    stdin = "\n".join([row(q), row(qd), row(u), f"{dt:.12g}"]) + "\n"
    env = dict(os.environ)
    env["GRID_EQUIV_T"] = equiv_t
    result = subprocess.run(
        [str(exe), str(threads)], input=stdin, cwd=exe.parent,
        capture_output=True, text=True, env=env,
    )
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0:
        if any(p in combined for p in GPU_UNAVAILABLE_PATTERNS):
            pytest.skip("CUDA runtime unavailable.")
        pytest.fail(f"runner failed:\n{result.stdout}\n{result.stderr}")
    return _parse_runner_output(result.stdout)


def _random_q(robot, fixture, rng):
    nq = robot.get_num_pos()
    q = rng.uniform(-1.0, 1.0, nq)
    qs = QUAT_START[fixture]
    quat = rng.uniform(-1.0, 1.0, 4)
    quat /= np.linalg.norm(quat)
    q[qs:qs + 4] = quat
    return q


@pytest.fixture(scope="module", params=sorted(QUAT_START), ids=lambda f: f.split(".")[0])
def spherical_case(request, tmp_path_factory):
    fixture = request.param
    robot = _parse(fixture)
    assert robot is not None
    build_dir = tmp_path_factory.mktemp(f"sph_intgrad_{fixture.split('.')[0]}")
    _generate_header(robot, build_dir)
    exes = {t: _compile_runner(build_dir, t) for t in ("float", "double")}
    return fixture, robot, exes


@pytest.mark.parametrize("equiv_t", ["float", "double"])
def test_spherical_integrator_gradient_matches_reference(spherical_case, equiv_t):
    fixture, robot, exes = spherical_case
    ref = RBDReference(robot)
    nv = robot.get_num_vel()
    rng = np.random.default_rng(42)
    tol = 5e-3 if equiv_t == "float" else 1e-8

    for trial in range(3):
        q = _random_q(robot, fixture, rng)
        qd = rng.uniform(-1.0, 1.0, nv)
        u = rng.uniform(-1.0, 1.0, nv)
        out = _run(exes[equiv_t], q, qd, u, DT, threads=32, equiv_t=equiv_t)
        for tag, it_name in INTEGRATORS:
            expected = np.asarray(
                ref.integrator_gradient(q, qd, u, DT, integrator_type=it_name),
                dtype=np.float64,
            )  # (2nv, 3nv)
            for suffix in ("_dAB", "_dAB_with_x_kp1"):
                got = np.asarray(out[tag + suffix], dtype=np.float64)
                assert got.shape == expected.shape, (tag + suffix, got.shape, expected.shape)
                err = float(np.max(np.abs(got - expected)))
                assert err < tol, (
                    f"{fixture} {tag}{suffix} trial {trial} ({equiv_t}): "
                    f"max|d|={err:.3e} vs oracle (tol {tol})")


def test_spherical_integrator_gradient_thread_invariance(spherical_case):
    """Identical dAB at thread counts 1 / 32 / 256 (fp64, exact match)."""
    fixture, robot, exes = spherical_case
    nv = robot.get_num_vel()
    rng = np.random.default_rng(7)
    q = _random_q(robot, fixture, rng)
    qd = rng.uniform(-1.0, 1.0, nv)
    u = rng.uniform(-1.0, 1.0, nv)
    baseline = None
    for threads in (1, 32, 256):
        out = _run(exes["double"], q, qd, u, DT, threads=threads, equiv_t="double")
        stacked = np.concatenate([
            np.asarray(out[tag + "_dAB"], dtype=np.float64).reshape(-1)
            for tag, _ in INTEGRATORS
        ])
        if baseline is None:
            baseline = stacked
        else:
            np.testing.assert_array_equal(
                stacked, baseline,
                err_msg=f"{fixture}: dAB differs at {threads} threads (thread-count variance)")

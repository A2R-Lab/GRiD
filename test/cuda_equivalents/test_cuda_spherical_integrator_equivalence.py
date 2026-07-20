"""CUDA equivalence test for the SPHERICAL (ball) joint time INTEGRATOR.

Validates `grid::integrator<T, IT>` (value surface x_{k+1} = integrator(x_k, u, dt))
for all five IntegratorTypes (EULER / SEMI_IMPLICIT_EULER / MIDPOINT / RK3 / RK4)
on the spherical fixtures, against a Python reference composed from the verified
RBDReference primitives:
  * the q-side is a Lie-group retract  q_{k+1} = ref.integrate(q, dt * src_v)
    (SO(3) quaternion exp per ball joint + plain add for the downstream-shifted
    revolute slots — the §1e case); and
  * the qd-side is the TrajoptPlant Runge-Kutta weighting of stage accelerations
    qd_{k+1} = qd + dt * sum_i b_i * qdd_i, with each stage qdd_i evaluated via the
    pinocchio-validated ref.aba at the stage configuration.

The numpy ABA-recursion does run for the qdd VALUE on a ball joint (ref.aba is the
pinocchio-validated FD oracle used by the dynamics-value spherical test); only the
scalar-per-body ABA *minv* recursion raises on spherical, which is irrelevant here.

It exercises BOTH CUDA surfaces, fp32 + fp64, at thread counts {1, 32, 256}:
  * the device function ``integrator_device<T, IT>`` (explicit nq-wide s_q /
    nv-wide s_qd,s_u buffers); and
  * the HOST batch wrapper ``integrator<T, IT, GRID_DATA_ALL>`` over a 4-timestep
    trajectory (the per-timestep NQ-wide input-slot path the bindings use — the
    §1e nq-stride check: every batch row must equal the single-call device row).

Asserts additionally that the spherical q-block of x_{k+1} stays UNIT-NORM and that
the downstream revolute q-slots equal the plain Euler add q + dt*qd.

Fixtures: spherical_arm (root spherical + revolute) and mixed_spherical_arm
(revolute -> spherical -> revolute, the mid-chain §1e case).
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
from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _detect_cuda_arch,
    _parse_runner_output,
    GPU_UNAVAILABLE_PATTERNS,
)

RUNNER_SOURCE = Path(__file__).with_name("cuda_spherical_integrator_runner.cu")
FIXDIR = Path(__file__).resolve().parents[2] / "external" / "URDFParser" / "tests" / "fixtures"

# (tag-prefix, integrator-type-name) for the five emitted kernels.
INTEGRATORS = (
    ("integrator_euler", "euler"),
    ("integrator_si_euler", "semi_implicit_euler"),
    ("integrator_midpoint", "midpoint"),
    ("integrator_rk3", "rk3"),
    ("integrator_rk4", "rk4"),
)

# Each fixture's quaternion-block start index in q (the spherical joint's first
# q slot). spherical_arm: ball at jid0 -> q[0:4]. mixed_spherical_arm: revolute
# then ball at jid1 -> q[1:5].
QUAT_START = {"spherical_arm.urdf": 0, "mixed_spherical_arm.urdf": 1}
DT = 0.01


def _parse(name):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return URDFParser().parse(str(FIXDIR / name), floating_base=False)


def _generate_header(robot, build_dir):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid"
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        # `integrator` is the requested algorithm; it transitively pulls in
        # inverse_dynamics + minv + forward_dynamics (+ crba for the spherical
        # inv(M) minv path). Requesting any non-ported algorithm would raise.
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header),
            algorithm_list=["integrator"],
        )
    return header


def _compile_runner(build_dir, equiv_t):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / f"cuda_spherical_integrator_runner_{equiv_t}.exe"
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
            "CUDA spherical integrator runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return exe


def _run(exe, q, qd, u, threads=32, equiv_t="float"):
    def row(v):
        return " ".join(f"{x:.12g}" for x in np.asarray(v, dtype=np.float64))
    stdin = "\n".join([row(q), row(qd), row(u)]) + "\n"
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
    """Random configuration with a UNIT quaternion in the spherical q-block."""
    nq = robot.get_num_pos()
    q = rng.uniform(-1.0, 1.0, nq)
    qs = QUAT_START[fixture]
    quat = rng.uniform(-1.0, 1.0, 4)
    quat /= np.linalg.norm(quat)
    q[qs:qs + 4] = quat
    return q


def _ref_integrator(ref, robot, q, qd, u, dt, integrator_type):
    """Python reference x_{k+1} = [q_{k+1}; qd_{k+1}] mirroring the codegen.

    q-side: ref.integrate(q, dt*src_v) (SO(3) retract + add). src_v = qd for
    EULER and the multi-stage TrajoptPlant variants (the q-source is always the
    ORIGINAL qd); for SEMI_IMPLICIT_EULER it is the freshly-updated v_{k+1}.
    qd-side: qd + dt*sum_i b_i*qdd_i with each stage qdd_i = ref.aba at the
    stage config (the TrajoptPlant xdot_i = [qd; qdd_i] convention: every stage
    uses the ORIGINAL qd for the q-offset)."""
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    g = -9.81
    qdd1 = np.asarray(ref.aba(q, qd, u, GRAVITY=g), dtype=np.float64).reshape(-1)

    if integrator_type in ("euler", "semi_implicit_euler"):
        qd_new = qd + dt * qdd1
        src_v = qd_new if integrator_type == "semi_implicit_euler" else qd
        q_new = ref.integrate(q, dt * src_v)
        return np.concatenate([q_new, qd_new])

    # Multi-stage. Stage 2 config: p1 = integrate(q, c1*dt*qd), p1_qd = qd + c1*dt*qdd1.
    c1 = 0.5
    p1_q = ref.integrate(q, c1 * dt * qd)
    p1_qd = qd + c1 * dt * qdd1
    qdd2 = np.asarray(ref.aba(p1_q, p1_qd, u, GRAVITY=g), dtype=np.float64).reshape(-1)

    if integrator_type == "midpoint":
        accel = qdd2
    else:
        # Stage 3 config.
        c2 = 0.75 if integrator_type == "rk3" else 0.5
        p2_q = ref.integrate(q, c2 * dt * qd)
        p2_qd = qd + c2 * dt * qdd2
        qdd3 = np.asarray(ref.aba(p2_q, p2_qd, u, GRAVITY=g), dtype=np.float64).reshape(-1)
        if integrator_type == "rk3":
            accel = (2.0 / 9.0) * qdd1 + (3.0 / 9.0) * qdd2 + (4.0 / 9.0) * qdd3
        else:  # rk4
            c3 = 1.0
            p3_q = ref.integrate(q, c3 * dt * qd)
            p3_qd = qd + c3 * dt * qdd3
            qdd4 = np.asarray(ref.aba(p3_q, p3_qd, u, GRAVITY=g), dtype=np.float64).reshape(-1)
            accel = (1.0 / 6.0) * qdd1 + (2.0 / 6.0) * qdd2 + (2.0 / 6.0) * qdd3 + (1.0 / 6.0) * qdd4

    qd_new = qd + dt * accel
    q_new = ref.integrate(q, dt * qd)  # q-source always original qd (TrajoptPlant)
    return np.concatenate([q_new, qd_new])


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("equiv_t,atol,rtol", [("float", 1e-4, 1e-4), ("double", 1e-10, 1e-10)])
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_integrator_matches_reference(tmp_path, fixture, equiv_t, atol, rtol):
    """CUDA spherical integrator (all 5 ITs, device + host-batch) must match the
    RBDReference oracle, be batch self-consistent (§1e), keep the ball q-block
    unit-norm, and integrate downstream revolute slots as plain Euler q+dt*qd."""
    robot = _parse(fixture)
    assert robot is not None
    ref = RBDReference(robot)
    nq = robot.get_num_pos()
    nv = robot.get_num_vel()
    qs = QUAT_START[fixture]
    unit_tol = 1e-6 if equiv_t == "float" else 1e-12

    exe = _compile_runner(tmp_path, equiv_t) if _generate_header(robot, tmp_path) else None

    rng = np.random.default_rng(13)
    failures = []
    for trial in range(4):
        q = _random_q(robot, fixture, rng)
        qd = rng.uniform(-0.8, 0.8, nv)
        u = rng.uniform(-0.5, 0.5, nv)
        out = _run(exe, q, qd, u, threads=32, equiv_t=equiv_t)
        tag = f"{fixture} {equiv_t} trial {trial}"

        for prefix, it_name in INTEGRATORS:
            ref_x = _ref_integrator(ref, robot, q, qd, u, DT, it_name)
            assert ref_x.shape == (nq + nv,)

            dev = np.asarray(out[prefix + "_x_kp1"], dtype=np.float64).reshape(-1)
            if dev.shape != ref_x.shape:
                failures.append(f"{tag} {prefix} device: shape {dev.shape} != {ref_x.shape}")
                continue
            if not np.allclose(dev, ref_x, atol=atol, rtol=rtol):
                failures.append(
                    f"{tag} {prefix} device: max|d|={np.max(np.abs(dev - ref_x)):.3e}\n"
                    f"  cuda={dev}\n  ref ={ref_x}")

            # spherical q-block must stay UNIT-NORM.
            quat = dev[qs:qs + 4]
            qnorm = float(np.linalg.norm(quat))
            if abs(qnorm - 1.0) > unit_tol:
                failures.append(f"{tag} {prefix} ball quat norm {qnorm:.12f} != 1 (tol {unit_tol})")

            # downstream revolute slots integrate as plain Euler q + dt*qd.
            for jid in range(robot.get_num_joints()):
                jtype = getattr(robot.get_joint_by_id(jid), "jtype", None)
                if jtype == "spherical":
                    continue
                iq = robot.get_joint_index_q(jid)
                iv = robot.get_joint_index_v(jid)
                iq = iq if isinstance(iq, (list, tuple)) else [iq]
                iv = iv if isinstance(iv, (list, tuple)) else [iv]
                src_v = qd  # q-source is original qd for euler + multi-stage
                if it_name == "semi_implicit_euler":
                    src_v = dev[nq:nq + nv]  # SI-Euler q-source is v_{k+1}
                for qi, vi in zip(iq, iv):
                    expect = q[qi] + DT * src_v[vi]
                    if not np.isclose(dev[qi], expect, atol=max(atol, 1e-5), rtol=rtol):
                        failures.append(
                            f"{tag} {prefix} revolute q[{qi}] {dev[qi]:.6e} != q+dt*qd {expect:.6e}")

            # host batch: every row == oracle AND == device single-call (§1e).
            for k in range(4):
                row = np.asarray(out[f"{prefix}_x_kp1_batch_{k}"], dtype=np.float64).reshape(-1)
                if not np.allclose(row, ref_x, atol=atol, rtol=rtol):
                    failures.append(
                        f"{tag} {prefix} batch[{k}] vs ref: max|d|={np.max(np.abs(row - ref_x)):.3e}")
                if not np.allclose(row, dev, atol=1e-6, rtol=1e-6):
                    failures.append(
                        f"{tag} {prefix} batch[{k}] vs device: max|d|={np.max(np.abs(row - dev)):.3e}")

    assert not failures, "spherical integrator CUDA equivalence failures:\n" + "\n".join(failures)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_integrator_thread_invariant(tmp_path, fixture):
    """The single-block spherical integrator kernels (all 5 ITs) must be exactly
    thread-count invariant (bit-identical at 1/32/256) on device + host-batch."""
    robot = _parse(fixture)
    assert robot is not None
    exe = _compile_runner(tmp_path, "float") if _generate_header(robot, tmp_path) else None

    rng = np.random.default_rng(17)
    q = _random_q(robot, fixture, rng)
    qd = rng.uniform(-0.8, 0.8, robot.get_num_vel())
    u = rng.uniform(-0.5, 0.5, robot.get_num_vel())

    base_out = _run(exe, q, qd, u, threads=32)
    baselines = {p: np.asarray(base_out[p + "_x_kp1"], dtype=np.float64) for p, _ in INTEGRATORS}
    failures = []
    for threads in (1, 32, 256):
        out = _run(exe, q, qd, u, threads=threads)
        for prefix, _ in INTEGRATORS:
            dev = np.asarray(out[prefix + "_x_kp1"], dtype=np.float64)
            if not np.array_equal(dev, baselines[prefix]):
                failures.append(
                    f"{fixture} threads={threads}: {prefix} device differs from threads=32 "
                    f"(max|d|={np.max(np.abs(dev - baselines[prefix])):.3e})")
            for k in range(4):
                row = np.asarray(out[f"{prefix}_x_kp1_batch_{k}"], dtype=np.float64)
                if not np.array_equal(row, dev):
                    failures.append(
                        f"{fixture} threads={threads}: {prefix} batch[{k}] != device single-call")

    assert not failures, "spherical integrator thread-invariance failures:\n" + "\n".join(failures)

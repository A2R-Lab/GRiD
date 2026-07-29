"""End-to-end CUDA verification of the SPHERICAL (ball) joint dynamics suite
(Phase-6 Tier-C): inverse_dynamics, crba, minv, and forward_dynamics.

forward_dynamics routes through minv (= inv(CRBA(q))) + the compute_c RNEA, not
the standalone ABA. The Tier-C CUDA minv therefore mirrors the mimic path
(crba_inner + invert_matrix), since the per-body scalar ABA minv recursion does
not generalize to a 3-DoF ball joint. The numpy ABA-recursion minv/forward_dynamics
oracles RAISE on spherical for the same reason, so the verified oracles are
inv(ref.crba(q)) for minv and the pinocchio-validated ref.aba(q,qd,u) for FD.

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
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _detect_cuda_arch,
    _parse_runner_output,
    GPU_UNAVAILABLE_PATTERNS,
)

RUNNER_SOURCE = Path(__file__).with_name("cuda_spherical_runner.cu")
FIXDIR = Path(__file__).resolve().parents[2] / "external" / "URDFParser" / "tests" / "fixtures"

# Each fixture's quaternion-block start index in q (the spherical joint's first
# q slot). spherical_arm: ball at jid0 -> q[0:4]. mixed_spherical_arm: revolute
# then ball at jid1 -> q[1:5].
QUAT_START = {"spherical_arm.urdf": 0, "mixed_spherical_arm.urdf": 1}


def _parse(name):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return URDFParser().parse(str(FIXDIR / name), floating_base=False)


# The eight ported spherical (Tier-C) algorithms. Requesting any other raises
# NotImplementedError by design. This is the FULL set the thread-invariance test
# (which reads every block) needs; the correctness tests each SPLIT off a subset.
_SPHERICAL_ALL_ALGOS = [
    "inverse_dynamics", "crba", "minv", "forward_dynamics",
    "inverse_dynamics_gradient", "forward_dynamics_gradient",
    "aba", "fdsva_so",
]

# Per-concern SPLIT cells: (codegen_algorithm_list, run_tokens). Each correctness test
# codegens only its cell's algos (a broken/omitted OTHER algo can't void it — the Bug-A
# fix) and compiles the runner gated to only its RUN token(s). The value test bundles the
# five value algos it cross-checks; the gradient/second-order tests isolate to one. Groups
# include the inner deps the kernel calls (FD → minv+id; gradients → crba+id; parity with
# the flagship split's probe-confirmed dependency groups). run_tokens=None (thread-invariance)
# builds the full all-block runner (GRID_RUN_DEFAULT=1) since it reads every output block.
_SPHERICAL_VALUE_ALGOS = ["inverse_dynamics", "crba", "minv", "forward_dynamics", "aba"]
_SPHERICAL_VALUE_TOKENS = frozenset({
    "RUN_INVERSE_DYNAMICS", "RUN_CRBA", "RUN_MINV", "RUN_FORWARD_DYNAMICS", "RUN_ABA",
})
_SPHERICAL_IDG_ALGOS = ["inverse_dynamics_gradient", "inverse_dynamics", "crba", "minv", "forward_dynamics"]
_SPHERICAL_FDG_ALGOS = ["forward_dynamics_gradient", "inverse_dynamics_gradient",
                        "inverse_dynamics", "crba", "minv", "forward_dynamics"]
# fdsva_so composes the id/fd gradient + Minv machinery, so its header pulls the full
# dynamics-gradient dep set; keep it a tight-but-complete group (excludes nothing it needs).
_SPHERICAL_FDSVA_ALGOS = _SPHERICAL_ALL_ALGOS


def _generate_header(robot, build_dir, algos=None):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid"
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header),
            algorithm_list=list(algos) if algos is not None else list(_SPHERICAL_ALL_ALGOS),
        )
    return header


# Shared per-algorithm COMPILE selector (see grid_runner_select.cuh). The runner
# #includes it; copy it next to the runner so the isolated-dir compile resolves it.
_SELECT_HEADER = Path(__file__).with_name("grid_runner_select.cuh")


def _compile_runner(build_dir, run_tokens=None):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    shutil.copyfile(_SELECT_HEADER, build_dir / _SELECT_HEADER.name)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_spherical_runner.exe"
    # Split mode: gate the runner to only this cell's RUN token(s) so a build break in
    # another algo can't void it (Bug-A isolation). Without run_tokens the runner builds
    # every block (back-compat all-in-one, used by the thread-invariance test).
    split_defines = []
    if run_tokens:
        split_defines.append("-DGRID_RUN_SPLIT")
        split_defines.extend(f"-D{tok}=1" for tok in sorted(run_tokens))
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-DGRID_CUDA_FLOATING_BASE=0",
        "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
        *split_defines,
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


def _run(exe, q, qd, u, threads=32, dtype="float"):
    def row(v):
        return " ".join(f"{x:.9g}" for x in np.asarray(v, dtype=np.float64))
    stdin = "\n".join([row(q), row(qd), row(u)]) + "\n"
    env = dict(os.environ)
    env["GRID_EQUIV_T"] = dtype  # "float" (fp32) or "double" (fp64)
    result = subprocess.run(
        [str(exe), str(threads)], input=stdin, cwd=exe.parent,
        capture_output=True, text=True, env=env
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
    """CUDA spherical inverse_dynamics + crba + minv + forward_dynamics must match
    the RBDReference numpy oracle on both the device function and the host batch
    wrapper, and be batch self-consistent (§1e).

    The numpy ABA-recursion minv/forward_dynamics RAISE on a 3-DoF ball joint
    (scalar per-body U/Dinv), so the verified oracles here are inv(crba(q)) for
    minv and the (pinocchio-validated) ref.aba(q,qd,u) for forward_dynamics."""
    robot = _parse(fixture)
    assert robot is not None
    ref = RBDReference(robot)
    nv = robot.get_num_vel()

    # SPLIT: value cell — codegen + gate only the five value algos this test cross-checks.
    exe = (_compile_runner(tmp_path, run_tokens=_SPHERICAL_VALUE_TOKENS)
           if _generate_header(robot, tmp_path, _SPHERICAL_VALUE_ALGOS) else None)

    rng = np.random.default_rng(7)
    zeros = np.zeros(nv, dtype=np.float64)
    failures = []
    for trial in range(4):
        q = _random_q(robot, fixture, rng)
        qd = rng.uniform(-0.8, 0.8, nv)
        u = rng.uniform(-0.5, 0.5, nv)
        out = _run(exe, q, qd, u)
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

        # --- minv: Minv = inv(CRBA(q)) (NV x NV) vs inv(oracle M) ---
        # The CUDA Tier-C minv routes through crba_inner + invert_matrix; the only
        # spherical-correct numpy oracle is inv(ref.crba(q)) (ABA minv raises).
        ref_Minv = np.linalg.inv(ref_M)
        cuda_Minv = np.asarray(out["minv"], dtype=np.float64)
        if cuda_Minv.size != nv * nv:
            failures.append(f"{tag} minv device: size {cuda_Minv.size} != {nv*nv}")
        else:
            # CUDA stores Minv SYMMETRIC_UPPER column-major; mirror upper->lower for
            # the dense compare (the FD finish reads only the upper triangle).
            cuda_Minv = cuda_Minv.reshape(nv, nv, order="F")
            cuda_Minv = np.triu(cuda_Minv) + np.triu(cuda_Minv, 1).T
            if not np.allclose(cuda_Minv, ref_Minv, atol=1e-4, rtol=1e-4):
                failures.append(
                    f"{tag} minv device: max|d|={np.max(np.abs(cuda_Minv - ref_Minv)):.3e}\n"
                    f"  cuda=\n{cuda_Minv}\n  ref =\n{ref_Minv}")
            for k in range(4):
                blk = np.asarray(out[f"minv_batch_{k}"], dtype=np.float64).reshape(nv, nv, order="F")
                blk = np.triu(blk) + np.triu(blk, 1).T
                if not np.allclose(blk, ref_Minv, atol=1e-4, rtol=1e-4):
                    failures.append(
                        f"{tag} minv batch[{k}] vs ref: max|d|={np.max(np.abs(blk - ref_Minv)):.3e}")
                if not np.allclose(blk, cuda_Minv, atol=1e-5, rtol=1e-5):
                    failures.append(
                        f"{tag} minv batch[{k}] vs device: max|d|={np.max(np.abs(blk - cuda_Minv)):.3e}")

        # --- forward_dynamics: qdd = Minv*(u-c) vs pinocchio-validated ref.aba ---
        ref_qdd = np.asarray(ref.aba(q, qd, u, GRAVITY=-9.81), dtype=np.float64).reshape(-1)
        cuda_qdd = np.asarray(out["forward_dynamics"], dtype=np.float64).reshape(-1)
        if cuda_qdd.shape != ref_qdd.shape:
            failures.append(f"{tag} fd device: shape {cuda_qdd.shape} != {ref_qdd.shape}")
        elif not np.allclose(cuda_qdd, ref_qdd, atol=1e-4, rtol=1e-4):
            failures.append(
                f"{tag} fd device: max|d|={np.max(np.abs(cuda_qdd - ref_qdd)):.3e}\n"
                f"  cuda={cuda_qdd}\n  ref ={ref_qdd}")
        for k in range(4):
            row = np.asarray(out[f"forward_dynamics_batch_{k}"], dtype=np.float64).reshape(-1)
            if not np.allclose(row, ref_qdd, atol=1e-4, rtol=1e-4):
                failures.append(
                    f"{tag} fd batch[{k}] vs ref: max|d|={np.max(np.abs(row - ref_qdd)):.3e}")
            if not np.allclose(row, cuda_qdd, atol=1e-5, rtol=1e-5):
                failures.append(
                    f"{tag} fd batch[{k}] vs device: max|d|={np.max(np.abs(row - cuda_qdd)):.3e}")

        # --- standalone aba: DIRECT 3x3-D recursion qdd vs ref.aba AND vs cuda FD ---
        # NOTE the oracle ref.aba SHORT-CIRCUITS to the Minv-compose path on a
        # spherical robot (the scalar ABA recursion cannot do a 3-DoF ball joint),
        # so it validates the qdd VALUE, not the 3x3-D recursion STRUCTURE. The
        # device-vs-forward_dynamics cross-check therefore matters: it confirms the
        # standalone DIRECT-recursion CUDA aba agrees with the (independently-
        # validated) Minv-compose CUDA forward_dynamics on the same input.
        cuda_aba = np.asarray(out["aba"], dtype=np.float64).reshape(-1)
        if cuda_aba.shape != ref_qdd.shape:
            failures.append(f"{tag} aba device: shape {cuda_aba.shape} != {ref_qdd.shape}")
        else:
            if not np.allclose(cuda_aba, ref_qdd, atol=1e-4, rtol=1e-4):
                failures.append(
                    f"{tag} aba device vs ref: max|d|={np.max(np.abs(cuda_aba - ref_qdd)):.3e}\n"
                    f"  cuda={cuda_aba}\n  ref ={ref_qdd}")
            # cross-check: standalone aba == compose-path forward_dynamics.
            if not np.allclose(cuda_aba, cuda_qdd, atol=1e-4, rtol=1e-4):
                failures.append(
                    f"{tag} aba device vs cuda forward_dynamics: "
                    f"max|d|={np.max(np.abs(cuda_aba - cuda_qdd)):.3e}")
        for k in range(4):
            row = np.asarray(out[f"aba_batch_{k}"], dtype=np.float64).reshape(-1)
            if not np.allclose(row, ref_qdd, atol=1e-4, rtol=1e-4):
                failures.append(
                    f"{tag} aba batch[{k}] vs ref: max|d|={np.max(np.abs(row - ref_qdd)):.3e}")
            if not np.allclose(row, cuda_aba, atol=1e-5, rtol=1e-5):
                failures.append(
                    f"{tag} aba batch[{k}] vs device: max|d|={np.max(np.abs(row - cuda_aba)):.3e}")

    assert not failures, "spherical CUDA equivalence failures:\n" + "\n".join(failures)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_thread_invariant(tmp_path, fixture):
    """The single-block spherical inverse_dynamics / crba / minv / forward_dynamics
    kernels must be exactly thread-count invariant (bit-identical at 1/32/256)."""
    robot = _parse(fixture)
    assert robot is not None
    # Thread-invariance reads EVERY output block, so it needs the full all-algo runner
    # (run_tokens=None → GRID_RUN_DEFAULT builds all blocks) against the full header.
    exe = _compile_runner(tmp_path) if _generate_header(robot, tmp_path) else None

    rng = np.random.default_rng(11)
    q = _random_q(robot, fixture, rng)
    qd = rng.uniform(-0.8, 0.8, robot.get_num_vel())
    u = rng.uniform(-0.5, 0.5, robot.get_num_vel())

    base_out = _run(exe, q, qd, u, threads=32)
    base = np.asarray(base_out["inverse_dynamics"], dtype=np.float64)
    base_M = np.asarray(base_out["crba"], dtype=np.float64)
    base_Minv = np.asarray(base_out["minv"], dtype=np.float64)
    base_qdd = np.asarray(base_out["forward_dynamics"], dtype=np.float64)
    base_aba = np.asarray(base_out["aba"], dtype=np.float64)
    base_idg = np.asarray(base_out["inverse_dynamics_gradient"], dtype=np.float64)
    base_fdg = np.asarray(base_out["forward_dynamics_gradient"], dtype=np.float64)
    # fdsva_so has no device single-call surface in the runner (the host wrapper is
    # the canonical §1e path); use its batch row 0 as the invariance baseline.
    base_so = np.asarray(base_out["fdsva_so_batch_0"], dtype=np.float64)
    failures = []
    # (algo output key, device baseline, batch key prefix) tuples to sweep. For
    # fdsva_so the "device single-call" baseline IS its own batch row 0.
    cells = [
        ("inverse_dynamics", base, "inverse_dynamics_batch_"),
        ("crba", base_M, "crba_batch_"),
        ("minv", base_Minv, "minv_batch_"),
        ("forward_dynamics", base_qdd, "forward_dynamics_batch_"),
        ("aba", base_aba, "aba_batch_"),
        ("inverse_dynamics_gradient", base_idg, "inverse_dynamics_gradient_batch_"),
        ("forward_dynamics_gradient", base_fdg, "forward_dynamics_gradient_batch_"),
        ("fdsva_so_batch_0", base_so, "fdsva_so_batch_"),
    ]
    for threads in (1, 32, 256):
        out = _run(exe, q, qd, u, threads=threads)
        for key, baseline, batch_prefix in cells:
            dev = np.asarray(out[key], dtype=np.float64)
            if not np.array_equal(dev, baseline):
                failures.append(
                    f"{fixture} threads={threads}: {key} device differs from threads=32 "
                    f"(max|d|={np.max(np.abs(dev - baseline)):.3e})")
            for k in range(4):
                row = np.asarray(out[f"{batch_prefix}{k}"], dtype=np.float64)
                if not np.array_equal(row, dev):
                    failures.append(
                        f"{fixture} threads={threads}: {key} batch[{k}] != device single-call")

    assert not failures, "spherical thread-invariance failures:\n" + "\n".join(failures)


# Per-(fixture, precision) absolute/relative tolerance buckets for the
# inverse_dynamics_gradient cell. fp64 is near machine-exact vs the float64
# oracle; fp32 carries the kernel's single-precision round-off (the dense
# reduced-space fold does ~NB X^T/I matvecs). NEVER loosen a global tolerance —
# bucket per cell (debug-guide §6).
_IDG_TOL = {
    ("spherical_arm.urdf", "float"): dict(atol=2e-4, rtol=2e-4),
    ("mixed_spherical_arm.urdf", "float"): dict(atol=2e-4, rtol=2e-4),
    ("spherical_arm.urdf", "double"): dict(atol=1e-8, rtol=1e-8),
    ("mixed_spherical_arm.urdf", "double"): dict(atol=1e-8, rtol=1e-8),
}


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("dtype", ["float", "double"])
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_inverse_dynamics_gradient_matches_reference(tmp_path, fixture, dtype):
    """CUDA spherical inverse_dynamics_gradient (dc_du = [dc_dq | dc_dqd], 2*NV*NV)
    must match the fixed RBDReference oracle on BOTH the with-qdd kernel single-call
    and the host batch wrapper, at fp32 and fp64, and be batch self-consistent (§1e).

    Routes through the DENSE serial reduced-space inner: a mid-chain (or fixed-root)
    3-DoF ball joint owns a 3-wide v-block (NJ != NV) that the sparse single-DoF band
    cannot represent. The runner uses the SAME nonzero accel for the kernel's qdd and
    the wrapper's h_qdd slot, so the oracle is inverse_dynamics_gradient(q, qd, qdd=u).
    mixed_spherical_arm (revolute -> ball -> revolute) is the decisive §1e case
    (every downstream q/v slot shifts by the ball's nq=4 / nv=3 widths)."""
    robot = _parse(fixture)
    assert robot is not None
    ref = RBDReference(robot)
    nv = robot.get_num_vel()
    tol = _IDG_TOL[(fixture, dtype)]

    # SPLIT: inverse_dynamics_gradient cell (excludes fd_gradient / fdsva_so).
    exe = (_compile_runner(tmp_path, run_tokens=frozenset({"RUN_INVERSE_DYNAMICS_GRADIENT"}))
           if _generate_header(robot, tmp_path, _SPHERICAL_IDG_ALGOS) else None)

    rng = np.random.default_rng(23)
    failures = []
    for trial in range(4):
        q = _random_q(robot, fixture, rng)
        qd = rng.uniform(-0.8, 0.8, nv)
        qdd = rng.uniform(-0.5, 0.5, nv)   # runner reads this slot (h_u) as qdd
        out = _run(exe, q, qd, qdd, dtype=dtype)
        tag = f"{fixture} [{dtype}] trial {trial}"

        # Oracle dc_du in reduced v-space: [dc_dq (nv x nv) | dc_dqd (nv x nv)].
        dc_du = ref.inverse_dynamics_gradient(
            q.copy(), qd.copy(), qdd.copy(), GRAVITY=-9.81,
            public_output=False, normalize_input=True)
        ref_dq, ref_dqd = np.hsplit(np.asarray(dc_du, dtype=np.float64), [nv])

        dev = np.asarray(out["inverse_dynamics_gradient"], dtype=np.float64).reshape(-1)
        if dev.size != 2 * nv * nv:
            failures.append(f"{tag} device: size {dev.size} != {2*nv*nv}")
            continue
        # column-major nv x nv per half (matches the kernel's [c*nv + v_i] store).
        dev_dq = dev[:nv * nv].reshape(nv, nv, order="F")
        dev_dqd = dev[nv * nv:].reshape(nv, nv, order="F")
        if not np.allclose(dev_dq, ref_dq, **tol):
            failures.append(
                f"{tag} dc_dq device: max|d|={np.max(np.abs(dev_dq - ref_dq)):.3e}\n"
                f"  cuda=\n{dev_dq}\n  ref =\n{ref_dq}")
        if not np.allclose(dev_dqd, ref_dqd, **tol):
            failures.append(
                f"{tag} dc_dqd device: max|d|={np.max(np.abs(dev_dqd - ref_dqd)):.3e}\n"
                f"  cuda=\n{dev_dqd}\n  ref =\n{ref_dqd}")

        # Host batch wrapper: every timestep row == oracle AND == device single-call.
        for k in range(4):
            blk = np.asarray(out[f"inverse_dynamics_gradient_batch_{k}"], dtype=np.float64).reshape(-1)
            blk_dq = blk[:nv * nv].reshape(nv, nv, order="F")
            blk_dqd = blk[nv * nv:].reshape(nv, nv, order="F")
            if not (np.allclose(blk_dq, ref_dq, **tol) and np.allclose(blk_dqd, ref_dqd, **tol)):
                failures.append(
                    f"{tag} batch[{k}] vs ref: max|d|="
                    f"{max(np.max(np.abs(blk_dq - ref_dq)), np.max(np.abs(blk_dqd - ref_dqd))):.3e}")
            # batched == standalone (the §1e nq-stride self-consistency check).
            if not np.allclose(blk, dev, atol=1e-9 if dtype == "double" else 1e-5,
                               rtol=1e-9 if dtype == "double" else 1e-5):
                failures.append(
                    f"{tag} batch[{k}] vs device: max|d|={np.max(np.abs(blk - dev)):.3e}")

    assert not failures, "spherical id-gradient equivalence failures:\n" + "\n".join(failures)


# Per-(fixture, precision) tolerance buckets for the forward_dynamics_gradient
# cell. fd_gradient = -Minv * id_gradient, so it inherits the dense id-gradient
# fold round-off AND the float32 |Minv|-conditioning floor (the inverse mass
# matrix amplifies the single-precision error). fp64 is near machine-exact.
# NEVER loosen a global tolerance — bucket per cell (debug-guide §6).
_FDG_TOL = {
    ("spherical_arm.urdf", "float"): dict(atol=5e-4, rtol=5e-4),
    ("mixed_spherical_arm.urdf", "float"): dict(atol=5e-4, rtol=5e-4),
    ("spherical_arm.urdf", "double"): dict(atol=1e-8, rtol=1e-8),
    # mixed_spherical_arm fp64: the -Minv @ dc_du finish amplifies the dense
    # id-gradient accumulation by |Minv| on entries of magnitude ~25, so the
    # ABSOLUTE residual floor lands at a few e-8 (relative ~1e-9, near machine
    # exact). The CUDA value and the composed numpy oracle do the SAME matmul in a
    # different fma/accumulation order -> a few-ulp-scaled-by-|Minv| absolute gap.
    # Bucket the atol up to clear it; rtol stays machine-tight (debug-guide §6,
    # the float32 |Minv|-conditioning floor — here surfacing at fp64 on the
    # larger-magnitude mid-chain robot). NEVER loosen a global tolerance.
    ("mixed_spherical_arm.urdf", "double"): dict(atol=1e-7, rtol=1e-8),
}


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("dtype", ["float", "double"])
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_forward_dynamics_gradient_matches_reference(tmp_path, fixture, dtype):
    """CUDA spherical forward_dynamics_gradient (df_du = [df_dq | df_dqd], 2*NV*NV)
    must match the RBDReference oracle on BOTH the u-input kernel single-call and
    the host batch wrapper, at fp32 and fp64, and be batch self-consistent (§1e).

    forward_dynamics_gradient is a pure orchestrator: df_du = -Minv * id_gradient(
    q, qd, qdd=forward_dynamics(q,qd,u)). For spherical it composes the already-
    spherical-aware sub-inners (minv via crba, the dense id-gradient) and finishes
    with a dimension-agnostic nv x 2nv tangent-space matmul (NO joint-id indexing,
    NO nq dependence). The CUDA -Minv*dc/du apply reads Minv through its
    SYMMETRIC_UPPER mirror index, so the emitted df_du is FULLY dense (both
    triangles) and directly comparable to the dense oracle (no triangle fix-up).
    mixed_spherical_arm (revolute -> ball -> revolute) is the decisive §1e case
    (every downstream q/v slot shifts by the ball's nq=4 / nv=3 widths).

    NOTE the oracle is COMPOSED from spherical-safe pieces, NOT ref.forward_
    dynamics_gradient: that method internally calls ref.minv, whose ABA-recursion
    RAISES on a 3-DoF ball joint (the same reason the CUDA minv routes through
    inv(CRBA)). So the verified oracle mirrors the CUDA composition exactly:
    qdd = ref.aba(q,qd,u) (pinocchio-validated), dc_du = ref.inverse_dynamics_
    gradient(q,qd,qdd), Minv = inv(ref.crba(q)), df_du = -Minv @ dc_du."""
    robot = _parse(fixture)
    assert robot is not None
    ref = RBDReference(robot)
    nv = robot.get_num_vel()
    tol = _FDG_TOL[(fixture, dtype)]

    # SPLIT: forward_dynamics_gradient cell (excludes fdsva_so).
    exe = (_compile_runner(tmp_path, run_tokens=frozenset({"RUN_FORWARD_DYNAMICS_GRADIENT"}))
           if _generate_header(robot, tmp_path, _SPHERICAL_FDG_ALGOS) else None)

    rng = np.random.default_rng(29)
    failures = []
    for trial in range(4):
        q = _random_q(robot, fixture, rng)
        qd = rng.uniform(-0.8, 0.8, nv)
        u = rng.uniform(-0.5, 0.5, nv)   # runner reads this slot as the input torque u
        out = _run(exe, q, qd, u, dtype=dtype)
        tag = f"{fixture} [{dtype}] trial {trial}"

        # Oracle df_du = -Minv @ [dc_dq | dc_dqd], composed from spherical-safe
        # pieces (ref.minv raises on a ball joint; see docstring). qdd from the
        # pinocchio-validated ABA; dc_du from the (spherical-aware) dense id-grad;
        # Minv = inv(CRBA(q)) — exactly the CUDA orchestration.
        qdd = np.asarray(ref.aba(q.copy(), qd.copy(), u.copy(), GRAVITY=-9.81),
                         dtype=np.float64).reshape(-1)
        dc_du = ref.inverse_dynamics_gradient(
            q.copy(), qd.copy(), qdd.copy(), GRAVITY=-9.81,
            public_output=False, normalize_input=True)
        dc_dq, dc_dqd = np.hsplit(np.asarray(dc_du, dtype=np.float64), [nv])
        minv = np.linalg.inv(np.asarray(ref.crba(q.copy()), dtype=np.float64))
        ref_dq = -minv @ dc_dq
        ref_dqd = -minv @ dc_dqd

        dev = np.asarray(out["forward_dynamics_gradient"], dtype=np.float64).reshape(-1)
        if dev.size != 2 * nv * nv:
            failures.append(f"{tag} device: size {dev.size} != {2*nv*nv}")
            continue
        # column-major nv x nv per half (matches the kernel's [c*nv + v_i] store).
        dev_dq = dev[:nv * nv].reshape(nv, nv, order="F")
        dev_dqd = dev[nv * nv:].reshape(nv, nv, order="F")
        if not np.allclose(dev_dq, ref_dq, **tol):
            failures.append(
                f"{tag} df_dq device: max|d|={np.max(np.abs(dev_dq - ref_dq)):.3e}\n"
                f"  cuda=\n{dev_dq}\n  ref =\n{ref_dq}")
        if not np.allclose(dev_dqd, ref_dqd, **tol):
            failures.append(
                f"{tag} df_dqd device: max|d|={np.max(np.abs(dev_dqd - ref_dqd)):.3e}\n"
                f"  cuda=\n{dev_dqd}\n  ref =\n{ref_dqd}")

        # Host batch wrapper: every timestep row == oracle AND == device single-call.
        for k in range(4):
            blk = np.asarray(out[f"forward_dynamics_gradient_batch_{k}"], dtype=np.float64).reshape(-1)
            blk_dq = blk[:nv * nv].reshape(nv, nv, order="F")
            blk_dqd = blk[nv * nv:].reshape(nv, nv, order="F")
            if not (np.allclose(blk_dq, ref_dq, **tol) and np.allclose(blk_dqd, ref_dqd, **tol)):
                failures.append(
                    f"{tag} batch[{k}] vs ref: max|d|="
                    f"{max(np.max(np.abs(blk_dq - ref_dq)), np.max(np.abs(blk_dqd - ref_dqd))):.3e}")
            # batched == standalone (the §1e nq-stride self-consistency check).
            if not np.allclose(blk, dev, atol=1e-9 if dtype == "double" else 1e-5,
                               rtol=1e-9 if dtype == "double" else 1e-5):
                failures.append(
                    f"{tag} batch[{k}] vs device: max|d|={np.max(np.abs(blk - dev)):.3e}")

    assert not failures, "spherical fd-gradient equivalence failures:\n" + "\n".join(failures)


# Per-(fixture, precision) tolerance buckets for the fdsva_so cell. fdsva_so is the
# 2nd derivative of GRiD's OWN forward_dynamics surface (-Minv contractions over the
# idsva_so tensors + the dM_dq*fd_grad cross terms); the CUDA kernel and the numpy
# oracle do the SAME einsum chain in a different fma/accumulation order. fp64 is
# near machine-exact; fp32 carries the SO accumulation round-off, amplified by the
# |Minv| conditioning of the final -Minv reduction (mixed_spherical_arm has larger-
# magnitude entries). NEVER loosen a global tolerance — bucket per cell (§6).
_FDSVA_SO_TOL = {
    ("spherical_arm.urdf", "float"): dict(atol=2e-3, rtol=2e-3),
    ("mixed_spherical_arm.urdf", "float"): dict(atol=2e-3, rtol=2e-3),
    ("spherical_arm.urdf", "double"): dict(atol=1e-7, rtol=1e-8),
    ("mixed_spherical_arm.urdf", "double"): dict(atol=1e-7, rtol=1e-8),
}

# daba_dvdv (qd-qd block) for a MID-CHAIN ball joint is validated against a
# forward_dynamics VALUE finite-difference rather than ref.fdsva_so. The numpy
# ref.idsva_so's d2tau_dqd carries a spurious cross term inside the mid-chain ball
# velocity sub-block (robust ~0.5-1.2 abs across seeds; see the RBDReference
# test_spherical_fdsva_so_matches_value_finite_difference note). The CUDA world-
# frame idsva_so inner computes it CORRECTLY: the CUDA daba_dvdv matches the
# value-FD to ~7e-5 (FD floor) at BOTH fp32 and fp64, while ref.fdsva_so disagrees
# with the value-FD by ~1.0. So for daba_dvdv the value-FD is the ground-truth
# oracle (CUDA is right, the numpy oracle is the one to fix in idsva_so). The
# root-ball fixture's ref.fdsva_so dvdv IS value-FD-correct, so it uses ref there.
_FDSVA_SO_VALUE_FD_TOL = {
    "float": dict(atol=3e-3, rtol=0),
    "double": dict(atol=3e-3, rtol=0),  # FD truncation floor (~7e-5) dominates
}


def _fdsva_so_value_fd_dvdv(ref, q, qd, u, nv, h=1e-5):
    """Central 4-point finite difference of forward_dynamics wrt (qd_j, qd_k):
    daba_dvdv[i,j,k] = d^2 qdd_i / dqd_j dqd_k. The qd-qd block touches no manifold
    (pure flat velocity), so the value-FD is the exact ground truth."""
    def fwd(qq, vv):
        return np.asarray(ref.forward_dynamics(qq.copy(), vv.copy(), u.copy()),
                          dtype=np.float64).reshape(-1)
    fd = np.zeros((nv, nv, nv))
    for j in range(nv):
        for k in range(nv):
            def g(sj, sk):
                v = qd.copy(); v[j] += sj * h; v[k] += sk * h
                return fwd(q, v)
            fd[:, j, k] = (g(1, 1) - g(1, -1) - g(-1, 1) + g(-1, -1)) / (4 * h * h)
    return fd


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("dtype", ["float", "double"])
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_fdsva_so_matches_reference(tmp_path, fixture, dtype):
    """CUDA spherical fdsva_so (4*NV^3 = [daba_dqdq | daba_dvdq | daba_dvdv |
    daba_dtdq], each NV x NV x NV row-major) must match its ground-truth oracle on
    the host batch wrapper, at fp32 and fp64, and be batch self-consistent (§1e).

    fdsva_so's contract is PURE nv-tangent (nv^3 tensors, nv x nv Minv) -- dimension-
    agnostic, no per-body S iteration, no nq arithmetic -- so spherical needs zero
    algorithm-specific code; it only routes the composed idsva_so inner through the
    WORLD frame (the body-frame single-DoF S contractions are wrong for a 3-DoF ball
    joint, exactly as idsva_so does for spherical). ref.fdsva_so runs end-to-end on a
    ball joint via the committed minv widening (its internal minv = inv(CRBA)).
    mixed_spherical_arm (revolute -> ball -> revolute) is the decisive §1e case.

    ORACLE PER BLOCK: daba_dqdq / daba_dvdq / daba_dtdq are validated vs
    ref.fdsva_so. daba_dvdv is validated vs a forward_dynamics VALUE finite
    difference: ref.idsva_so's numpy d2tau_dqd is defective for a MID-CHAIN ball's
    velocity sub-block (so ref.fdsva_so dvdv is wrong there by ~1.0), but the CUDA
    world-frame idsva_so inner computes it correctly (matches the value-FD to the FD
    floor). The CUDA value is the correct one; the numpy idsva_so dvdv is the bug.
    """
    robot = _parse(fixture)
    assert robot is not None
    ref = RBDReference(robot)
    nv = robot.get_num_vel()
    tol = _FDSVA_SO_TOL[(fixture, dtype)]
    vfd_tol = _FDSVA_SO_VALUE_FD_TOL[dtype]
    nv3 = nv * nv * nv

    # SPLIT: fdsva_so cell (gate to only the second-order block; header pulls its
    # id/fd-gradient + Minv deps).
    exe = (_compile_runner(tmp_path, run_tokens=frozenset({"RUN_FDSVA_SO"}))
           if _generate_header(robot, tmp_path, _SPHERICAL_FDSVA_ALGOS) else None)

    rng = np.random.default_rng(31)
    failures = []
    for trial in range(4):
        q = _random_q(robot, fixture, rng)
        qd = rng.uniform(-0.8, 0.8, nv)
        u = rng.uniform(-0.5, 0.5, nv)   # runner reads this slot as the input torque u
        out = _run(exe, q, qd, u, dtype=dtype)
        tag = f"{fixture} [{dtype}] trial {trial}"

        # ref.fdsva_so for the dqdq / dvdq / dtdq blocks (it is correct for those).
        daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq = ref.fdsva_so(
            q.copy(), qd.copy(), u.copy(), GRAVITY=-9.81)
        ref_blocks = [np.asarray(t, dtype=np.float64) for t in
                      (daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq)]
        names = ["daba_dqdq", "daba_dvdq", "daba_dvdv", "daba_dtdq"]
        # daba_dvdv ground-truth = forward_dynamics value-FD (CUDA is correct here).
        vfd_dvdv = _fdsva_so_value_fd_dvdv(ref, q, qd, u, nv)

        def _oracle(b):
            return (vfd_dvdv, vfd_tol) if b == 2 else (ref_blocks[b], tol)

        # Row 0 is the canonical result; rows 1..3 are batch self-consistency.
        dev_full = np.asarray(out["fdsva_so_batch_0"], dtype=np.float64).reshape(-1)
        if dev_full.size != 4 * nv3:
            failures.append(f"{tag} device: size {dev_full.size} != {4*nv3}")
            continue
        # Each nv^3 block is stored row-major [(i*nv+j)*nv+k] -> C-order reshape.
        for b in range(4):
            oracle_t, oracle_tol = _oracle(b)
            dev_t = dev_full[b * nv3:(b + 1) * nv3].reshape(nv, nv, nv)
            if not np.allclose(dev_t, oracle_t, **oracle_tol):
                failures.append(
                    f"{tag} {names[b]}: max|d|={np.max(np.abs(dev_t - oracle_t)):.3e}")

        # Host batch wrapper: every timestep row == oracle AND == row 0 (§1e).
        for k in range(4):
            blk = np.asarray(out[f"fdsva_so_batch_{k}"], dtype=np.float64).reshape(-1)
            for b in range(4):
                oracle_t, oracle_tol = _oracle(b)
                blk_t = blk[b * nv3:(b + 1) * nv3].reshape(nv, nv, nv)
                if not np.allclose(blk_t, oracle_t, **oracle_tol):
                    failures.append(
                        f"{tag} batch[{k}] {names[b]} vs oracle: "
                        f"max|d|={np.max(np.abs(blk_t - oracle_t)):.3e}")
            if not np.allclose(blk, dev_full, atol=1e-9 if dtype == "double" else 1e-5,
                               rtol=1e-9 if dtype == "double" else 1e-5):
                failures.append(
                    f"{tag} batch[{k}] vs row0: max|d|={np.max(np.abs(blk - dev_full)):.3e}")

    assert not failures, "spherical fdsva_so equivalence failures:\n" + "\n".join(failures)

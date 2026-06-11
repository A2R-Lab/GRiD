"""End-to-end CUDA verification of SPHERICAL (ball) joint idsva_so (Tier-C):
the four second-order inverse-dynamics derivative tensors
[d2tau_dq2 | d2tau_dqd2 | d2tau_dvdq | dM_dq], each NV x NV x NV.

ROUTING: spherical fixed-base robots dispatch to the WORLD-frame idsva_so inner
(predicate `_idsva_so_use_world_frame`). The body-frame inner uses single-DoF
contractions that are WRONG for a 6x3 spherical motion subspace; the world-frame
inner is already per-velocity-column multi-DoF-aware (built for floating/mimic),
keyed on `wf_body_v_start` / `wf_body_v_index`. For a (non-mimic) spherical robot
that velocity metadata is the identity per-column map (alpha == 1, n_int == NV), so
the existing non-mimic world path is exactly correct -- no internal slab / fold.

ORACLE: `RBDReference.idsva_so_world_frame` (NOT the body frame, which is buggy for
mid-chain spherical -- it diffs ~0.01 from the world frame on mixed_spherical_arm).
The world oracle is pinocchio-validated on cardinal robots and matches the body
frame to 1e-16 on the root-ball fixture (where body happens to be correct).

A spherical joint is a 3-DoF manifold joint: NV=3 (body-frame angular velocity),
NQ=4 (unit quaternion xyzw, pinocchio JointModelSpherical), so NQ != NV exactly
like the floating free-flyer's rotation sub-block. q is consumed ONLY by
load_update_XImats (already spherical-aware via the quaternion substitution); the
SO inner never indexes s_q for a joint angle. qd/qdd are nv-tangent; the output
tensors are nv^3 tangent.

It drives the dispatching HOST batch wrapper `idsva_so<T, GRID_DATA_ALL>` over a
4-timestep trajectory of IDENTICAL inputs (the per-timestep NQ-wide input-slot
path the bindings use) -- this is the path the spherical dispatcher actually
routes, so it validates the routing decision. Every batch row must equal the WORLD
oracle AND the rows must be mutually identical (the §1e nq-stride self-consistency
check), swept over thread counts {1, 32, 256} for invariance.

(The standalone `idsva_so_device` inline entry carries a pre-existing latent
const-qualifier mismatch on s_qdd, so the runner uses the host kernel path the
bindings actually use, not that inline entry -- see the runner header comment.)

Fixtures: spherical_arm (root spherical + revolute, NV=4) and mixed_spherical_arm
(revolute -> spherical -> revolute, NV=5 -- the mid-chain §1e case that shifts
every downstream q/v offset AND is where the body frame is wrong).

NOTE: a SEPARATE runner (cuda_spherical_so_runner.cu) from the dynamics-suite
cuda_spherical_runner.cu, to avoid SO-smem entanglement with that runner.
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

RUNNER_SOURCE = Path(__file__).with_name("cuda_spherical_so_runner.cu")
FIXDIR = Path(__file__).resolve().parents[2] / "URDFParser" / "tests" / "fixtures"

# Each fixture's quaternion-block start index in q (the spherical joint's first q
# slot). spherical_arm: ball at jid0 -> q[0:4]. mixed_spherical_arm: revolute then
# ball at jid1 -> q[1:5].
QUAT_START = {"spherical_arm.urdf": 0, "mixed_spherical_arm.urdf": 1}

# The CUDA s_idsva_so packs the four nv^3 blocks in this order (matches the world
# oracle's (d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq) return order).
SO_BLOCK_NAMES = ("d2tau_dq2", "d2tau_dqd2", "d2tau_dvdq", "dM_dq")


def _parse(name):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return URDFParser().parse(str(FIXDIR / name), floating_base=False)


def _generate_header(robot, build_dir):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid"
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        # Request idsva_so_world_frame: this (a) emits the world-frame inner +
        # kernel + host (the dispatch target for spherical), AND (b) emits the
        # dispatching `idsva_so` host wrapper, which routes spherical -> world via
        # `_idsva_so_use_world_frame`. (Requesting idsva_so_body_frame alone would
        # NOT emit the world frame on a fixed-base robot, leaving the spherical
        # dispatcher's world target undefined -> link error.)
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header),
            algorithm_list=["idsva_so_world_frame"],
        )
    return header


def _compile_runner(build_dir):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_spherical_so_runner.exe"
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
            "CUDA spherical SO runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return exe


def _run(exe, q, qd, qdd, threads=32, dtype="float"):
    def row(v):
        return " ".join(f"{x:.9g}" for x in np.asarray(v, dtype=np.float64))
    stdin = "\n".join([row(q), row(qd), row(qdd)]) + "\n"
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


def _oracle_so(ref, q, qd, qdd):
    """World-frame SO oracle, flattened to [d2tau_dq2 | d2tau_dqd2 | d2tau_dvdq |
    dM_dq] (each nv^3, C-order) to match the CUDA s_idsva_so pack."""
    tensors = ref.idsva_so_world_frame(q.copy(), qd.copy(), qdd.copy(), GRAVITY=-9.81)
    return [np.asarray(t, dtype=np.float64) for t in tensors]


# Per-(fixture, precision) tolerance buckets. fp32 carries the world inner's
# single-precision round-off (the triple-ancestor walk does ~NB X^T/I/crm
# contractions). fp64: the CUDA value and the numpy world oracle do the SAME
# contractions in a different fma/accumulation order, so on output cells of
# magnitude ~tens the ABSOLUTE residual floors at a few-ulp-scaled-by-magnitude
# ~2e-9 (measured 1.7e-9 spherical / 1.9e-9 mixed); the RELATIVE error stays
# machine-tight. Bucket the atol up to clear it; keep rtol near machine-exact --
# NEVER loosen a global tolerance (debug-guide §6, same pattern as the spherical
# fd-gradient fp64 bucket).
_SO_TOL = {
    ("spherical_arm.urdf", "float"): dict(atol=2e-3, rtol=2e-3),
    ("mixed_spherical_arm.urdf", "float"): dict(atol=2e-3, rtol=2e-3),
    ("spherical_arm.urdf", "double"): dict(atol=1e-7, rtol=1e-9),
    ("mixed_spherical_arm.urdf", "double"): dict(atol=1e-7, rtol=1e-9),
}


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("dtype", ["float", "double"])
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_idsva_so_matches_world_reference(tmp_path, fixture, dtype):
    """CUDA spherical idsva_so (4 nv^3 tensors) must match RBDReference.idsva_so_
    WORLD_frame on the dispatching host batch wrapper, at fp32 and fp64, and be
    batch self-consistent (§1e: every timestep row identical).

    mixed_spherical_arm (revolute -> ball -> revolute) is the decisive case: it is
    BOTH the §1e mid-chain shift AND where the body frame is wrong (world differs
    from body by ~0.01), so matching the world oracle there proves the routing."""
    robot = _parse(fixture)
    assert robot is not None
    ref = RBDReference(robot)
    nv = robot.get_num_vel()
    tol = _SO_TOL[(fixture, dtype)]
    block = nv ** 3

    exe = _compile_runner(tmp_path) if _generate_header(robot, tmp_path) else None

    rng = np.random.default_rng(31)
    failures = []
    for trial in range(4):
        q = _random_q(robot, fixture, rng)
        qd = rng.uniform(-0.8, 0.8, nv)
        qdd = rng.uniform(-0.5, 0.5, nv)
        out = _run(exe, q, qd, qdd, dtype=dtype)
        tag = f"{fixture} [{dtype}] trial {trial}"

        ref_blocks = _oracle_so(ref, q, qd, qdd)

        single = np.asarray(out["idsva_so"], dtype=np.float64).reshape(-1)
        if single.size != 4 * block:
            failures.append(f"{tag} single: size {single.size} != {4*block}")
            continue
        for bi, bname in enumerate(SO_BLOCK_NAMES):
            cuda_b = single[bi * block:(bi + 1) * block].reshape(nv, nv, nv)
            ref_b = ref_blocks[bi]
            if not np.allclose(cuda_b, ref_b, **tol):
                failures.append(
                    f"{tag} {bname} single: max|d|="
                    f"{np.max(np.abs(cuda_b - ref_b)):.3e}")

        # §1e: every timestep row == oracle AND == the single (row-0) result.
        for k in range(4):
            row = np.asarray(out[f"idsva_so_batch_{k}"], dtype=np.float64).reshape(-1)
            for bi, bname in enumerate(SO_BLOCK_NAMES):
                blk_b = row[bi * block:(bi + 1) * block].reshape(nv, nv, nv)
                if not np.allclose(blk_b, ref_blocks[bi], **tol):
                    failures.append(
                        f"{tag} {bname} batch[{k}] vs ref: max|d|="
                        f"{np.max(np.abs(blk_b - ref_blocks[bi])):.3e}")
            if not np.allclose(row, single, atol=1e-9 if dtype == "double" else 1e-5,
                               rtol=1e-9 if dtype == "double" else 1e-5):
                failures.append(
                    f"{tag} batch[{k}] vs row0: max|d|={np.max(np.abs(row - single)):.3e}")

    assert not failures, "spherical idsva_so equivalence failures:\n" + "\n".join(failures)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_idsva_so_thread_invariant(tmp_path, fixture):
    """The single-block spherical idsva_so (world-frame) kernel must be exactly
    thread-count invariant (bit-identical at 1/32/256), AND every host-batch row
    must bit-match the row-0 result.

    §1j watch: a thread-count-dependent discrepancy that vanishes when isolated is
    almost always an uninitialized/under-initialized shared-scratch read (the
    mixed_spherical_arm flake-watch). Run under concurrent GPU load to surface it.
    The world inner uses NO GLASS beta=0 GEMM (only hand-rolled accum + dot_prod),
    so the specific CRBA beta=0 hazard is absent; this guards any remaining cold
    scratch slot."""
    robot = _parse(fixture)
    assert robot is not None
    exe = _compile_runner(tmp_path) if _generate_header(robot, tmp_path) else None

    rng = np.random.default_rng(13)
    q = _random_q(robot, fixture, rng)
    qd = rng.uniform(-0.8, 0.8, robot.get_num_vel())
    qdd = rng.uniform(-0.5, 0.5, robot.get_num_vel())

    base = np.asarray(_run(exe, q, qd, qdd, threads=32)["idsva_so"], dtype=np.float64)
    failures = []
    for threads in (1, 32, 256):
        out = _run(exe, q, qd, qdd, threads=threads)
        single = np.asarray(out["idsva_so"], dtype=np.float64)
        if not np.array_equal(single, base):
            failures.append(
                f"{fixture} threads={threads}: idsva_so differs from threads=32 "
                f"(max|d|={np.max(np.abs(single - base)):.3e})")
        for k in range(4):
            row = np.asarray(out[f"idsva_so_batch_{k}"], dtype=np.float64)
            if not np.array_equal(row, single):
                failures.append(
                    f"{fixture} threads={threads}: idsva_so batch[{k}] != row-0 result")

    assert not failures, "spherical idsva_so thread-invariance failures:\n" + "\n".join(failures)

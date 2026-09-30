"""End-to-end CUDA verification of SPHERICAL (ball) joint fdsva_so (Tier-C): the four
second-order FORWARD-dynamics derivative tensors
[daba_dqdq | daba_dvdq | daba_dvdv | daba_dtdq], each NV x NV x NV.

Sibling of test_cuda_spherical_so_equivalence.py (idsva_so), exercising fdsva_so.

ROUTING: spherical fixed-base robots compose the WORLD-frame idsva_so inner inside
fdsva_so_device (predicate `_fdsva_so_use_world_idsva` — the body-frame inner's
single-DoF contractions are wrong for a 6x3 spherical motion subspace). So the
oracle must assemble fdsva_so from the WORLD-frame second-order tensors, not the
body frame (RBDReference.fdsva_so dispatches by base type -> body for fixed-base,
which is buggy for mid-chain ball). We replicate RBDReference.fdsva_so's einsum
assembly here but substitute `idsva_so_world_frame` for the inner SO tensors; the
remaining ingredients (minv, forward_dynamics, forward_dynamics_gradient) are all
spherical-safe (the same ones the spherical fd-gradient test uses).

A4 (this PR): fdsva_so gained a surgical cold rung that routes ONLY the composed
WORLD idsva_so inner's cold trio (Xdown/v_w/a_w) to the SO-temp d_workspace region
(COLD_IN_SMEM=false) while the hot pool stays in smem, sitting between the
'outputs->global' rung and the all-or-nothing pool->global rung. This test forces
that rung on these small robots via GRID_CUDA_TARGET_SHARED_MEM_BYTES (a ~576-byte
window where global_tensors no longer fits but idsva_cold does) so the spilled-
cold-trio code path is exercised without a 20-min big-robot compile.

A spherical joint is a 3-DoF manifold joint: NV=3 (body-frame angular velocity),
NQ=4 (unit quaternion xyzw), so NQ != NV; q is consumed ONLY by load_update_XImats
(spherical-aware via the quaternion substitution). qd/u are nv-tangent; the outputs
are nv^3 tangent.

It drives the dispatching HOST batch wrapper `fdsva_so<T, GRID_DATA_ALL>` over a
4-timestep trajectory of IDENTICAL inputs (the per-timestep NQ-wide input-slot path
the bindings use). Every batch row must equal the WORLD-routed oracle AND the rows
must be mutually identical (the §1e nq-stride self-consistency check), swept over
thread counts {1, 32, 256} for invariance.

Fixtures: spherical_arm (root spherical + revolute, NV=4) and mixed_spherical_arm
(revolute -> spherical -> revolute, NV=5 -- the mid-chain §1e case that shifts every
downstream q/v offset AND is where the body frame is wrong).
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

RUNNER_SOURCE = Path(__file__).with_name("cuda_spherical_fdsva_so_runner.cu")
FIXDIR = Path(__file__).resolve().parents[2] / "external" / "URDFParser" / "tests" / "fixtures"

# Each fixture's quaternion-block start index in q (the spherical joint's first q
# slot). spherical_arm: ball at jid0 -> q[0:4]. mixed_spherical_arm: revolute then
# ball at jid1 -> q[1:5].
QUAT_START = {"spherical_arm.urdf": 0, "mixed_spherical_arm.urdf": 1}

# The CUDA s_df2 packs the four nv^3 blocks in RBDReference.fdsva_so return order.
FDSVA_BLOCK_NAMES = ("daba_dqdq", "daba_dvdq", "daba_dvdv", "daba_dtdq")

# A4 forced-cold-rung targets: GRID_CUDA_TARGET_SHARED_MEM_BYTES where
# select_shared_tier_3way picks the idsva_cold rung (PERF=LITE=idsva_cold). Found
# empirically (the ~576-byte window between global_tensors and idsva_cold arenas).
# None -> default PERF (full smem, no cold spill).
COLD_RUNG_TARGET = {"spherical_arm.urdf": "7400", "mixed_spherical_arm.urdf": "9600"}


def _parse(name):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return URDFParser().parse(str(FIXDIR / name), floating_base=False)


def _generate_header(robot, build_dir, shared_mem_target=None):
    header = build_dir / "grid.cuh"
    # The shared-mem target is read by GRiDCodeGenerator.__init__ from the env at
    # construction time, so set it around the construct + emit.
    prev = os.environ.get("GRID_CUDA_TARGET_SHARED_MEM_BYTES")
    if shared_mem_target is not None:
        os.environ["GRID_CUDA_TARGET_SHARED_MEM_BYTES"] = shared_mem_target
    try:
        codegen = GRiDCodeGenerator(
            robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid"
        )
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            # Request fdsva_so: spherical codegen supports it (+ its idsva_so_world
            # composition). The dispatching `fdsva_so` host wrapper composes the world
            # inner for spherical via `_fdsva_so_use_world_idsva`.
            codegen.gen_all_code(
                include_homogenous_transforms=True,
                output_path=str(header),
                algorithm_list=["fdsva_so"],
            )
    finally:
        if prev is None:
            os.environ.pop("GRID_CUDA_TARGET_SHARED_MEM_BYTES", None)
        else:
            os.environ["GRID_CUDA_TARGET_SHARED_MEM_BYTES"] = prev
    return header


def _compile_runner(build_dir):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_spherical_fdsva_so_runner.exe"
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
            "CUDA spherical fdsva_so runner compilation failed.\n"
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


def _oracle_fdsva_so_world(ref, q, qd, u, GRAVITY=-9.81):
    """WORLD-routed fdsva_so oracle: RBDReference.fdsva_so's einsum assembly with the
    inner second-order tensors taken from idsva_so_WORLD_frame (the frame the spherical
    fdsva_so_device composes), flattened to the 4 nv^3 blocks in the CUDA s_df2 order
    [daba_dqdq | daba_dvdq | daba_dvdv | daba_dtdq]."""
    Minv = ref.minv(q.copy())
    qdd = ref.forward_dynamics(q.copy(), qd.copy(), u.copy())
    di2_dq, di2_dqd, di2_dvdq, dm_dq = ref.idsva_so_world_frame(
        q.copy(), qd.copy(), qdd.copy(), GRAVITY)
    fd_dq, fd_dqd = ref.forward_dynamics_gradient(q.copy(), qd.copy(), u.copy())

    daba_dqdq = -np.einsum('il,ljk->ijk', Minv,
                           di2_dq + np.einsum('ilk,lj->ijk', dm_dq, fd_dq)
                           + np.einsum('ilk,lj->ikj', dm_dq, fd_dq))
    daba_dvdq = -np.einsum('il,ljk->ijk', Minv,
                           di2_dvdq + np.einsum('ilk,lj->ijk', dm_dq, fd_dqd))
    daba_dvdv = -np.einsum('il,ljk->ijk', Minv, di2_dqd)
    daba_dtdq = -np.einsum('il,ljk->ijk', Minv,
                           np.einsum('ilk,lj->ijk', dm_dq, Minv))
    return [np.asarray(t, dtype=np.float64) for t in
            (daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq)]


# Per-(fixture, precision) tolerance buckets. fp32 carries the world inner's single-
# precision round-off compounded through the fdsva_so einsum assembly (Minv contraction
# + the SO tensors). fp64: CUDA vs the numpy assembly do the SAME contractions in a
# different fma order so the absolute residual floors at a few ulp-scaled-by-magnitude;
# bucket the atol up to clear it, keep rtol near machine-exact. NEVER loosen a global
# tolerance (debug-guide §6). Mirrors the spherical idsva_so + fd-gradient buckets.
_FDSVA_TOL = {
    ("spherical_arm.urdf", "float"): dict(atol=3e-3, rtol=3e-3),
    ("mixed_spherical_arm.urdf", "float"): dict(atol=3e-3, rtol=3e-3),
    ("spherical_arm.urdf", "double"): dict(atol=1e-7, rtol=1e-9),
    ("mixed_spherical_arm.urdf", "double"): dict(atol=1e-7, rtol=1e-9),
}


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("dtype", ["float", "double"])
@pytest.mark.parametrize("tier", ["perf", "cold"])
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_fdsva_so_matches_world_reference(tmp_path, fixture, tier, dtype):
    """CUDA spherical fdsva_so (4 nv^3 tensors) must match the WORLD-routed fdsva_so
    oracle on the dispatching host batch wrapper, at fp32 AND fp64, at PERF AND the
    A4 forced cold-spill tier, and be batch self-consistent (§1e: every timestep row
    identical).

    The `cold` tier forces select_shared_tier_3way to pick the A4 idsva_cold rung
    (the composed world inner spills its cold trio Xdown/v_w/a_w to d_workspace), so
    the value MUST be byte-equivalent to PERF -- the cold trio is dead before the hot
    triple-walk, so spilling it changes occupancy, never numerics."""
    robot = _parse(fixture)
    assert robot is not None
    ref = RBDReference(robot)
    nv = robot.get_num_vel()
    tol = _FDSVA_TOL[(fixture, dtype)]
    block = nv ** 3
    target = COLD_RUNG_TARGET[fixture] if tier == "cold" else None

    exe = _compile_runner(tmp_path) if _generate_header(robot, tmp_path, target) else None

    rng = np.random.default_rng(31)
    failures = []
    for trial in range(4):
        q = _random_q(robot, fixture, rng)
        qd = rng.uniform(-0.8, 0.8, nv)
        u = rng.uniform(-0.5, 0.5, nv)
        out = _run(exe, q, qd, u, dtype=dtype)
        tag = f"{fixture} [{tier}/{dtype}] trial {trial}"

        ref_blocks = _oracle_fdsva_so_world(ref, q, qd, u)

        single = np.asarray(out["fdsva_so"], dtype=np.float64).reshape(-1)
        if single.size != 4 * block:
            failures.append(f"{tag} single: size {single.size} != {4*block}")
            continue
        for bi, bname in enumerate(FDSVA_BLOCK_NAMES):
            cuda_b = single[bi * block:(bi + 1) * block].reshape(nv, nv, nv)
            ref_b = ref_blocks[bi]
            if not np.allclose(cuda_b, ref_b, **tol):
                failures.append(
                    f"{tag} {bname} single: max|d|="
                    f"{np.max(np.abs(cuda_b - ref_b)):.3e}")

        # §1e: every timestep row == oracle AND == the single (row-0) result.
        for k in range(4):
            row = np.asarray(out[f"fdsva_so_batch_{k}"], dtype=np.float64).reshape(-1)
            for bi, bname in enumerate(FDSVA_BLOCK_NAMES):
                blk_b = row[bi * block:(bi + 1) * block].reshape(nv, nv, nv)
                if not np.allclose(blk_b, ref_blocks[bi], **tol):
                    failures.append(
                        f"{tag} {bname} batch[{k}] vs ref: max|d|="
                        f"{np.max(np.abs(blk_b - ref_blocks[bi])):.3e}")
            if not np.allclose(row, single, atol=1e-9 if dtype == "double" else 1e-5,
                               rtol=1e-9 if dtype == "double" else 1e-5):
                failures.append(
                    f"{tag} batch[{k}] vs row0: max|d|={np.max(np.abs(row - single)):.3e}")

    assert not failures, "spherical fdsva_so equivalence failures:\n" + "\n".join(failures)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("tier", ["perf", "cold"])
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_fdsva_so_thread_invariant(tmp_path, fixture, tier):
    """The single-block spherical fdsva_so (world-frame composed) kernel must be exactly
    thread-count invariant (bit-identical at 1/32/256), AND every host-batch row must
    bit-match the row-0 result -- at PERF AND the A4 forced cold-spill tier.

    §1j watch: a thread-count-dependent discrepancy that vanishes when isolated is
    almost always an uninitialized/under-initialized shared-scratch read. The A4 cold
    rung spills the WORLD inner's cold trio (Xdown/v_w/a_w) to d_workspace -- those must
    be WRITTEN-before-READ under the existing __syncthreads() ordering. Run under
    concurrent GPU load to surface a cold-scratch flake."""
    robot = _parse(fixture)
    assert robot is not None
    target = COLD_RUNG_TARGET[fixture] if tier == "cold" else None
    exe = _compile_runner(tmp_path) if _generate_header(robot, tmp_path, target) else None

    rng = np.random.default_rng(13)
    q = _random_q(robot, fixture, rng)
    qd = rng.uniform(-0.8, 0.8, robot.get_num_vel())
    u = rng.uniform(-0.5, 0.5, robot.get_num_vel())

    base = np.asarray(_run(exe, q, qd, u, threads=32)["fdsva_so"], dtype=np.float64)
    failures = []
    for threads in (1, 32, 256):
        out = _run(exe, q, qd, u, threads=threads)
        single = np.asarray(out["fdsva_so"], dtype=np.float64)
        if not np.array_equal(single, base):
            failures.append(
                f"{fixture} [{tier}] threads={threads}: fdsva_so differs from threads=32 "
                f"(max|d|={np.max(np.abs(single - base)):.3e})")
        for k in range(4):
            row = np.asarray(out[f"fdsva_so_batch_{k}"], dtype=np.float64)
            if not np.array_equal(row, single):
                failures.append(
                    f"{fixture} [{tier}] threads={threads}: fdsva_so batch[{k}] != row-0 result")

    assert not failures, "spherical fdsva_so thread-invariance failures:\n" + "\n".join(failures)

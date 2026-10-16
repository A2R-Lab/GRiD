"""CUDA equivalence test for the generated general-frame geometric Jacobian
family (E2): J, its time derivative Jdot, and the operational-space inertia
Lambda.

Validates against the RBDReference numpy oracle, which itself matches
pinocchio to ~1e-14:
  * grid::frame_jacobian_device      vs RBDReference.frame_jacobian
                                        (getFrameJacobian / getJointJacobian)
  * grid::frame_jacobian_dot_device  vs RBDReference.frame_jacobian_dot
                                        (computeJointJacobiansTimeVariation)
  * grid::osc_inertia_device         vs RBDReference.osc_inertia
                                        (inv(J Minv J^T))
for the three pinocchio reference frames (LOCAL / WORLD / LOCAL_WORLD_ALIGNED).

Lambda is self-contained: grid::osc_inertia_device composes Minv on device
(via minv_inner, F-region spilled to a shared s_F buffer) and densifies
the SYMMETRIC_UPPER output internally during the J*Minv*J^T contraction — the
runner feeds it q alone, no external Minv.

The CUDA path is float32, so the comparison uses a float32-scale tolerance like
the other CUDA smoke tests. Lambda, a float32 inverse, is held to the normwise
backward error at every sample and entrywise only below cond(A) 1e5 (see
check_osc_inertia; guide 7.z48). The frame target is the leaf joint id of each robot
(the project joint id passed straight through to the device as target_jid; the
numpy oracle is queried by the same joint's name).

Robots: every manifest robot on both bases (widened 2026-10-15; narrow with
GRID_CUDA_FRAME_JAC_ROBOTS="iiwa14:fixed,go2:floating,fr3:fixed").

fr3 (mimic) now exercises Lambda too: osc_inertia composes Minv on device via
minv_inner -> crba_inner -> invert (== RBDReference.minv's mimic fast
path inv(CRBA(q))), matching the numpy oracle to float32 on both bases x all 3
reference frames.

fr3 is a MIMIC robot (fr3_finger_joint2 mimics fr3_finger_joint1, multiplier 1.0,
sharing one velocity coordinate). It exercises the geometric-Jacobian mimic fold:
each chain joint accumulates its column contribution scaled by its mimic
multiplier onto the shared reduced v-slot (mirrors the ee_pose_gradient Step 3b
alpha-accumulate and RBDReference.frame_jacobian's `scale = mimic_scale(j)`).
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
    CUDA_DEFAULT_TOLERANCE,
    _build_cuda_samples,
    _detect_cuda_arch,
    _parse_runner_output,
    _run_runner,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter


RUNNER_SOURCE = Path(__file__).with_name("cuda_frame_jacobian_smoke_runner.cu")
# (J block, Jdot block, Lambda block, pinocchio reference frame).
_REF_FRAMES = (
    ("J_local", "Jd_local", "L_local", "LOCAL"),
    ("J_world", "Jd_world", "L_world", "WORLD"),
    ("J_lwa", "Jd_lwa", "L_lwa", "LOCAL_WORLD_ALIGNED"),
)
_ALGO_KEYS = ["frame_jacobian", "frame_jacobian_dot", "osc_inertia"]

# Lambda = inv(A) with A = J Minv J^T, compared in two tiers (guide 7.z48).
# A float32 inverse carries an entrywise relative error of about u*cond(A),
# u = 2^-24, so the 5e-3 entrywise class is meaningful below cond(A) ~1e5 and
# ill-posed above it. The raw residual |Lambda A - I| is bounded by the same
# u*cond and is no fix (a CPU float32 inverse measures 8e-2 at cond 1.2e7).
# The check that holds at every conditioning is the normwise backward error
# of each column x_j of Lambda_cuda as the solution of A_ref x = e_j
# (Rigal and Gaches 1967):
#     eta_j = ||A_ref x_j - e_j||_2 / (||A_ref||_2 ||x_j||_2),
# the smallest relative perturbation of A_ref for which x_j is exact. So
# eta <= float32 class states that Lambda_cuda is the exact inverse of a
# matrix within that class of A_ref, whatever cond(A_ref) is. Measured eta is
# 1e-11..9e-8 on every well-conditioned sample of every robot and <= 3e-6 at
# the exactly singular corner samples (cond ~1e14); the class is the harness
# default 2e-4. Below cond 1e5 the entrywise comparison is kept as well. When
# A_ref is singular at float32 working precision (cond >= 2^24 = 1/u) the
# float32 inverse need not exist: the q=0 corner of iiwa14 (cond inf) and of
# h1_2 (cond 6e18) gives a non-finite Lambda on the device and from a CPU
# float32 inverse alike, so a non-finite Lambda is accepted there and only
# there; a finite Lambda is always held to the backward-error check.
OSC_INERTIA_ENTRYWISE_COND = 1e5
OSC_INERTIA_BACKWARD_TOL = CUDA_DEFAULT_TOLERANCE["rtol"]
OSC_INERTIA_FLOAT32_SINGULAR_COND = 2.0 ** 24


def check_osc_inertia(L_cuda, A_ref, osc_inertia_ref, tag, *, rtol=5e-3, atol=5e-3):
    """Assert Lambda_cuda inverts A_ref to the float32 class; return (cond, eta, mode).

    `osc_inertia_ref` is a callable returning the float64 oracle Lambda; it is
    only evaluated in the entrywise tier (cond(A_ref) < 1e5), so an exactly
    singular A_ref never reaches the oracle's inverse.
    """
    L = np.asarray(L_cuda, dtype=np.float64).reshape(6, 6)
    A = np.asarray(A_ref, dtype=np.float64)
    cond = float(np.linalg.cond(A))
    mode = ("entrywise+backward-error" if cond < OSC_INERTIA_ENTRYWISE_COND
            else "backward-error")
    if not np.isfinite(L).all():
        assert cond >= OSC_INERTIA_FLOAT32_SINGULAR_COND, (
            f"Lambda {tag}: non-finite entries although A_ref is nonsingular at "
            f"float32 precision (cond(A_ref)={cond:.3e} < 2^24, check={mode})")
        return cond, float("nan"), "singular-at-float32 (non-finite inverse)"
    norm_a = float(np.linalg.norm(A, 2))
    resid = A @ L - np.eye(6)
    tiny = np.finfo(np.float64).tiny
    eta = max(
        float(np.linalg.norm(resid[:, j])) / (norm_a * max(float(np.linalg.norm(L[:, j])), tiny))
        for j in range(6)
    )
    assert eta <= OSC_INERTIA_BACKWARD_TOL, (
        f"Lambda {tag}: normwise backward error {eta:.3e} > {OSC_INERTIA_BACKWARD_TOL:.0e} "
        f"(cond(A_ref)={cond:.3e}, check={mode})")
    if cond < OSC_INERTIA_ENTRYWISE_COND:
        L_ref = np.asarray(osc_inertia_ref(), dtype=np.float64)
        scale = float(np.max(np.abs(L_ref)))
        np.testing.assert_allclose(
            L, L_ref, rtol=rtol, atol=max(atol, rtol * scale),
            err_msg=f"Lambda {tag} (cond(A_ref)={cond:.3e}, check={mode})")
    return cond, eta, mode


def osc_inertia_summary(prefix, records):
    """One line per cell: cond range, how many samples took which check, max eta."""
    if not records:
        return f"[osc_inertia] {prefix}: no Lambda checks"
    conds = [r[0] for r in records]
    both = sum(1 for r in records if r[2].startswith("entrywise"))
    singular = sum(1 for r in records if r[2].startswith("singular"))
    etas = [r[1] for r in records if np.isfinite(r[1])]
    max_eta = f"{max(etas):.1e}" if etas else "n/a"
    return (f"[osc_inertia] {prefix}: {len(records)} Lambda checks, cond(A_ref) "
            f"{min(conds):.1e}..{max(conds):.1e}, entrywise+backward-error {both}, "
            f"backward-error only {len(records) - both - singular}, "
            f"singular-at-float32 non-finite {singular}, max eta {max_eta}")


_ALL_ROBOTS = ("iiwa14", "go2", "g1", "h1_2", "fr3", "rizon4", "gen3", "fetch", "baxter")
_DEFAULT_ROBOTS = ",".join(f"{r}:{b}" for b in ("fixed", "floating") for r in _ALL_ROBOTS)


def _robot_modes():
    raw = os.environ.get("GRID_CUDA_FRAME_JAC_ROBOTS", _DEFAULT_ROBOTS)
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
    executable = build_dir / "cuda_frame_jacobian_smoke_runner.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-I{glass_inc}", "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA frame_jacobian smoke runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


def _stdin(target_jid, q, qd):
    rows = [str(int(target_jid)),
            " ".join(f"{v:.9g}" for v in np.asarray(q, dtype=np.float32)),
            " ".join(f"{v:.9g}" for v in np.asarray(qd, dtype=np.float32))]
    return "\n".join(rows) + "\n"


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_frame_jacobian_matches_reference(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_frame_jac"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    executable, cmd = _compile_runner(build_dir)

    robot = project_model.robot
    leaf_id = robot.get_leaf_nodes()[0]
    leaf_name = robot.get_joint_by_id(leaf_id).get_name()
    nv = project_model.nv

    samples = _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)

    def close(actual, expected, msg, rtol=2e-3, atol=2e-3):
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64), expected,
            rtol=rtol, atol=max(atol, rtol * scale), err_msg=msg,
        )

    lambda_records = []
    for sample in samples:
        q = np.asarray(sample.q, np.float64)
        qd = np.asarray(sample.qd, np.float64)
        out = _parse_runner_output(_run_runner(executable, _stdin(leaf_id, q, qd), cmd))
        minv_ref = np.asarray(project_model.minv(q), dtype=np.float64)
        for jblk, dblk, lblk, ref_frame in _REF_FRAMES:
            tag = f"{robot_id}-{base_mode} @ {sample.name} {ref_frame}"
            # J: analytic vs analytic (tight).
            J_ref = np.asarray(
                project_model.frame_jacobian(q, leaf_name, ref_frame), dtype=np.float64)
            close(out[jblk].reshape(6, nv, order="F"), J_ref, f"J {tag}")
            # Jdot: both sides are finite differences of the same analytic J
            # (numpy oracle h=1e-6, device h=1e-4) -> looser float32-FD tolerance.
            Jd_ref = np.asarray(
                project_model.frame_jacobian_dot(q, qd, leaf_name, ref_frame),
                dtype=np.float64)
            close(out[dblk].reshape(6, nv, order="F"), Jd_ref, f"Jdot {tag}",
                  rtol=5e-2, atol=5e-2)
            # Lambda = (J Minv J^T)^-1: 6x6. Emitted for mimic robots too
            # (e.g. fr3): osc_inertia composes Minv on device via
            # minv_inner -> crba_inner -> invert, which the fr3-fixed CUDA
            # crba/minv equivalence already proves correct. The L_* keys are
            # absent only if osc_inertia was not selected at all (then skip).
            if lblk not in out:
                continue
            # Every sample is checked, singular corners included; see
            # check_osc_inertia for the two-tier basis.
            lambda_records.append(check_osc_inertia(
                out[lblk].reshape(6, 6, order="F"), J_ref @ minv_ref @ J_ref.T,
                lambda: project_model.osc_inertia(q, leaf_name, ref_frame),
                tag))
    print(osc_inertia_summary(f"{robot_id}-{base_mode}", lambda_records))

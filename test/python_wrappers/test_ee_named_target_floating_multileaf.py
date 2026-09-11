"""C4: EE named-target (floating-base) + multi-leaf (NUM_EES > 1) validation.

The end_effector_pose GRADIENT and HESSIAN named-fixed-target codegen gap was
closed on **iiwa14-fixed only** (grad + hessian bit-exact vs the pinocchio
oracle, thread-invariant). This harness extends that validation to the two
unvalidated regimes:

  1. NAMED fixed-target, FLOATING base (NUM_EES == 1).
     register_robot(floating_base=True, ee_joint_names=["<fixed_joint>"]) bakes
     fixed_target_name into gen_all_code (bindings/grid_rbd/_compile.py:183-186),
     so the codegen emits the `_<name>` gradient/hessian kernels and the handle's
     end_effector_pose_gradient(q) / _hessian(q) report THAT named frame. This
     exercises the floating root's 6 base-velocity columns through the named-target
     chain (never validated — only iiwa14-FIXED was).

  2. MULTI-LEAF (NUM_EES > 1), FLOATING base.
     register_robot(floating_base=True, ee_joint_names=None) targets all leaf
     nodes (the DEFAULT codegen). For a multi-leaf robot (go2: 4 calf leaves,
     baxter: 3) the gradient/hessian output carries a per-leaf OUTPUT STRIDE
     (handle reshapes (B, NEE, NV, 6)). This exercises the 6*NEE*NV gradient /
     6*NEE*NV*NV hessian stride that a single-leaf manipulator never hits.

Both halves are checked against the INDEPENDENT pinocchio backend (the C++
authority — NOT GRiD's own RBDReference, which would mask shared bugs; this is
why the iiwa14 smoke test only shape-checks the hessian). Plus a thread-
invariance sweep (1 / 32 / 100 / max_perf) on every regime: single-block
kernels are block-stride loops, so any thread count that fits MUST produce
bit-identical output.

The CUDA path is float32; rpy rows blow up at gimbal lock (pitch ~= +-pi/2) on
BOTH oracle and device, so those rows are skipped per (leaf, sample). The xyz
(position-Jacobian) rows are always checked.

Run on the serial GPU queue (the first register per robot compiles a .so,
~1-5 min each; subsequent runs hit the grid-rbd cache):

    .venv/bin/python -m pytest \
        test/python_wrappers/test_ee_named_target_floating_multileaf.py \
        -m python_wrappers -v

Robot/target choices are overridable; see _CASES below.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


# ─── skip preconditions ─────────────────────────────────────────────────────

_grid_rbd = pytest.importorskip(
    "grid_rbd", reason="grid-rbd not installed (pip install bindings/)"
)

if shutil.which("nvcc") is None:
    pytest.skip(
        "nvcc not on PATH; grid-rbd register_robot requires it",
        allow_module_level=True,
    )

# Shared oracle / model plumbing (mirrors test_cuda_eepose_runtime.py exactly).
from RBDReference.tests import MANIFEST_PATH  # noqa: E402
from RBDReference.tests.model_sources import (  # noqa: E402
    iter_robot_cases,
    resolve_robot_spec,
)
from RBDReference.equivalents.reference_backend import build_project_adapter  # noqa: E402
from RBDReference.equivalents import build_adapter, resolve_backend  # noqa: E402

# _build_cuda_samples gives correct floating-quat sample layouts (q[0:3]=xyz,
# q[3:7]=quat_xyzw normalized, q[7:]=joints) plus deterministic corner samples.
from test.cuda_equivalents.cuda_harness import (  # noqa: E402
    _build_cuda_samples,
)


pytestmark = [pytest.mark.python_wrappers, pytest.mark.floating_base]


# ─── case matrix ────────────────────────────────────────────────────────────
#
# Each case: (robot_id, base_mode, ee_mode, ee_joint_name).
#   ee_mode == "named"     -> register with ee_joint_names=[ee_joint_name]
#                             (NUM_EES == 1; the named-target codegen path).
#   ee_mode == "multileaf" -> register with ee_joint_names=None
#                             (NUM_EES == #leaves; the output-stride path).
#                             ee_joint_name is ignored (set to None).
#
# Chosen from a live URDFParser introspection (2026-06-15):
#   go2-floating  : 4 calf leaves; 'imu_joint' is a valid fixed-joint target.
#   baxter-floating: 3 arm leaves; 'right_hand_camera_axis' is a fixed target.
#   iiwa14-floating: 1 leaf; 'iiwa_joint_ee' is the flange target. Single-leaf
#                    on FLOATING base = the named-target floating canary that the
#                    iiwa14-FIXED validation never reached (floating root columns).
#
# Override the whole matrix with GRID_EE_NAMED_CASES, a ';'-separated list of
# 'robot:base:mode[:joint]' tokens (e.g. "go2:floating:named:imu_joint;
# baxter:floating:multileaf").
#
# KNOWN CODEGEN GAP (flagged 2026-06-15, see docs/open-tasks/C4_ee_named_target_validation.md):
# the NAMED fixed-target codegen on a FLOATING **branched** (non-serial-chain)
# robot CRASHES at codegen time — gen_end_effector_pose_inner's branched parent-
# walk (_eepose_gradient_hessian.py) lacked the serial path's `parent==-1` root
# guard, so the fixed-target extra-bfs-level (+1) walked past the root link and
# called get_parent_id on a None link (Robot.py:288).
# FIXED (C4, 2026-06-15): gen_end_effector_pose_inner's branched parent-walk now
# clamps a column to the -1 sentinel once its jid no longer resolves to a link
# (_parent_or_root), so the over-walk stops at the root and the existing runtime
# `if(parent_jid==-1){continue;}` guard skips the compose — exactly as the all-leaf
# path already does. go2/baxter floating named now codegen cleanly; these cases are
# HARD passes (no xfail). Byte-identical for every previously-OK case.
_DEFAULT_CASES = [
    # NAMED fixed-target, FLOATING base, SERIAL chain (NUM_EES == 1) — WORKS:
    ("iiwa14", "floating", "named", "iiwa_joint_ee"),
    # NAMED fixed-target, FLOATING base, BRANCHED (NUM_EES == 1) — WORKS (C4 fix):
    ("go2", "floating", "named", "imu_joint"),
    ("baxter", "floating", "named", "right_hand_camera_axis"),
    # MULTI-LEAF (NUM_EES > 1), FLOATING base (output-stride) — WORKS:
    ("go2", "floating", "multileaf", None),
    ("baxter", "floating", "multileaf", None),
    # Cross-check: the iiwa14-FIXED named case that the original validation
    # already closed — a guard against regressing the path this work extends.
    ("iiwa14", "fixed", "named", "iiwa_joint_ee"),
]


def _parse_cases():
    raw = os.environ.get("GRID_EE_NAMED_CASES")
    if not raw:
        return _DEFAULT_CASES
    out = []
    for tok in raw.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        rid, base, mode = parts[0], parts[1], parts[2]
        joint = parts[3] if len(parts) > 3 and parts[3] else None
        out.append((rid.strip(), base.strip(), mode.strip(), joint))
    return out


_CASES = _parse_cases()


def _case_id(case) -> str:
    rid, base, mode, joint = case
    suffix = f"-{joint}" if joint else ""
    return f"{rid}-{base}-{mode}{suffix}"


def _case_params():
    """pytest.param per case. All cases are hard pass/fail (the named-floating-
    branched codegen crash was fixed 2026-06-15; no known-gap xfails remain)."""
    return [pytest.param(case, id=_case_id(case)) for case in _CASES]


# Float32 cross-precision tolerance (matches the other CUDA smoke tests). The
# per-leaf, per-sample comparison additionally scales the atol by the reference
# magnitude so near-zero entries do not trip an absolute floor.
_RTOL = 2e-3
_ATOL = 2e-3
# rpy rows are skipped within this band of +-pi/2 (E^{-1} gimbal-lock blowup,
# on BOTH the oracle and the device).
_PITCH_GUARD = 0.15
# Thread counts swept for the bit-invariance check. 0 == the per-algo autotuned
# default; we also probe max_perf explicitly. 100 = a non-multiple-of-32 partial
# trailing warp.
_THREAD_SWEEP = (1, 32, 100)


# ─── helpers ────────────────────────────────────────────────────────────────


def _resolve_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not in manifest")


def _urdf_path(resolved):
    p = Path(resolved.urdf_path)
    if not p.exists():
        pytest.skip(f"URDF asset not present at {p}; run ./install/developer_install.sh")
    return p


def _close(actual, expected, msg, *, rtol=_RTOL, atol=_ATOL):
    expected = np.asarray(expected, dtype=np.float64)
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected,
        rtol=rtol,
        atol=max(atol, rtol * scale),
        err_msg=msg,
    )


def _leaf_target_names(robot, ee_mode, ee_joint_name):
    """The ORACLE target name(s), one per emitted EE, in NUM_EES order.

    For the named-target build NUM_EES == 1 and the single target is the named
    fixed joint. For the multi-leaf build NUM_EES == #leaves, in get_leaf_nodes()
    order (the SAME order the codegen iterates and the handle reshape preserves).
    """
    if ee_mode == "named":
        return [ee_joint_name]
    return [robot.get_joint_by_id(j).get_name() for j in robot.get_leaf_nodes()]


def _register(robot_id, base_mode, ee_mode, ee_joint_name, urdf, batch):
    ee_names = [ee_joint_name] if ee_mode == "named" else None
    name = f"c4_{robot_id}_{base_mode}_{ee_mode}_{ee_joint_name or 'leaves'}"
    return _grid_rbd.register_robot(
        name=name,
        urdf_path=str(urdf),
        floating_base=(base_mode == "floating"),
        ee_joint_names=ee_names,
        # EE-pose (kinematic) test -> no mjx twins needed; drop them for a fast compile.
        enable_mujoco_kernels=False,
        max_batch_size=max(batch, 8),
    )


# ─── tests ──────────────────────────────────────────────────────────────────


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("case", _case_params())
def test_ee_named_multileaf_gradient_hessian(case, request):
    """Named-target (floating) and multi-leaf gradient + hessian match the
    pinocchio oracle, per EE leaf and per sample."""
    robot_id, base_mode, ee_mode, ee_joint_name = case
    spec = _resolve_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {robot_id}-{base_mode}: {exc}")
    urdf = _urdf_path(resolved)

    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    robot = project_model.robot
    nv = project_model.nv

    # Validate the case is well-formed against the actual model BEFORE compiling.
    if ee_mode == "named":
        fixed_names = {fj.name for fj in robot.fixed_joints}
        if ee_joint_name not in fixed_names:
            pytest.skip(
                f"{robot_id}-{base_mode}: '{ee_joint_name}' is not a fixed joint "
                f"(available: {sorted(fixed_names)[:8]} ...)"
            )
    else:
        if len(robot.get_leaf_nodes()) < 2:
            pytest.skip(
                f"{robot_id}-{base_mode}: only {len(robot.get_leaf_nodes())} leaf — "
                "multi-leaf stride needs NUM_EES > 1."
            )

    # INDEPENDENT oracle = pinocchio (the C++ authority). RBDReference's own
    # d2ee layout is not a 1:1 match — using it would mask shared bugs.
    backend = resolve_backend(os.environ.get("GRID_REFERENCE_BACKEND", "pinocchio"))
    reference_model = (
        project_model
        if backend == "reference"
        else build_adapter(spec, resolved, base_mode=base_mode, backend=backend)
    )

    samples = _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)
    batch = len(samples)
    handle = _register(robot_id, base_mode, ee_mode, ee_joint_name, urdf, batch)

    expected_nee = 1 if ee_mode == "named" else len(robot.get_leaf_nodes())
    assert handle.num_ees == expected_nee, (
        f"{_case_id(case)}: handle.num_ees={handle.num_ees} != expected {expected_nee} "
        "(named-target build must collapse to NUM_EES==1; multi-leaf must keep all leaves)."
    )
    assert handle.num_vel == nv

    targets = _leaf_target_names(robot, ee_mode, ee_joint_name)
    assert len(targets) == handle.num_ees

    # Batched device call once; compare each (sample, leaf) against the oracle.
    q_batch = np.stack([np.asarray(s.q, dtype=np.float32) for s in samples], axis=0)
    grad = handle.end_effector_pose_gradient(q_batch)   # (B, 6*NEE, NV)
    hess = handle.end_effector_pose_hessian(q_batch)     # (B, 6*NEE, NV, NV)
    assert grad.shape == (batch, 6 * handle.num_ees, nv)
    assert hess.shape == (batch, 6 * handle.num_ees, nv, nv)
    assert np.all(np.isfinite(grad)) and np.all(np.isfinite(hess))

    grad = grad.reshape(batch, handle.num_ees, 6, nv)
    hess = hess.reshape(batch, handle.num_ees, 6, nv, nv)

    for si, sample in enumerate(samples):
        q = np.asarray(sample.q, dtype=np.float64)
        for li, tgt in enumerate(targets):
            tag = f"{_case_id(case)} leaf={tgt} @ {sample.name}"

            g_ref = np.asarray(
                reference_model.end_effector_pose_gradient(q, tgt), dtype=np.float64
            )  # (6, NV)
            h_ref = np.asarray(
                reference_model.end_effector_pose_hessian(q, tgt), dtype=np.float64
            )  # (6, NV, NV)
            assert g_ref.shape == (6, nv)
            assert h_ref.shape == (6, nv, nv)

            # Gimbal-lock guard: at pitch ~= +-pi/2 the rpy rows (3..5) diverge on
            # both oracle and device. Detect from the oracle pose pitch.
            pose_ref = np.asarray(
                reference_model.end_effector_pose(q, tgt), dtype=np.float64
            ).reshape(-1)
            pitch = float(pose_ref[4])
            near_gimbal = abs(abs(pitch) - np.pi / 2.0) < _PITCH_GUARD
            rows = slice(0, 3) if near_gimbal else slice(0, 6)

            # ---- GRADIENT (position rows always; rpy rows away from gimbal) ----
            _close(grad[si, li, rows, :], g_ref[rows, :], f"grad {tag}")
            # ---- HESSIAN ----
            _close(hess[si, li, rows, :, :], h_ref[rows, :, :], f"hess {tag}")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("case", _case_params())
def test_ee_named_multileaf_thread_invariance(case, request):
    """Single-block block-stride kernels must be thread-count-invariant: the
    named-target / multi-leaf gradient + hessian must be BIT-identical across
    {1, 32, 100, max_perf} threads. A low-count divergence is a missing
    __syncthreads, not a tolerance artifact."""
    robot_id, base_mode, ee_mode, ee_joint_name = case
    spec = _resolve_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {robot_id}-{base_mode}: {exc}")
    urdf = _urdf_path(resolved)

    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    robot = project_model.robot

    if ee_mode == "named":
        if ee_joint_name not in {fj.name for fj in robot.fixed_joints}:
            pytest.skip(f"{robot_id}-{base_mode}: '{ee_joint_name}' not a fixed joint")
    elif len(robot.get_leaf_nodes()) < 2:
        pytest.skip(f"{robot_id}-{base_mode}: needs NUM_EES > 1 for the multi-leaf stride")

    samples = _build_cuda_samples(project_model, random_count=1, include_corner_samples=True)
    q_batch = np.stack([np.asarray(s.q, dtype=np.float32) for s in samples], axis=0)
    handle = _register(robot_id, base_mode, ee_mode, ee_joint_name, urdf, len(samples))

    # Probe {1, 32, 100} plus the autotuned max_perf level (clamped at the
    # per-block cap by set_threads_per_block / the kernel launch).
    thread_counts = list(_THREAD_SWEEP)
    max_perf = int(handle.max_perf_level_threads)
    if max_perf not in thread_counts:
        thread_counts.append(max_perf)

    ref_grad = ref_hess = None
    for n in thread_counts:
        handle.set_threads_per_block(int(n))
        g = np.asarray(handle.end_effector_pose_gradient(q_batch))
        h = np.asarray(handle.end_effector_pose_hessian(q_batch))
        if ref_grad is None:
            ref_grad, ref_hess = g, h
        else:
            np.testing.assert_array_equal(
                g, ref_grad,
                err_msg=f"{_case_id(case)}: ee_pose_gradient diverged at threads={n}",
            )
            np.testing.assert_array_equal(
                h, ref_hess,
                err_msg=f"{_case_id(case)}: ee_pose_hessian diverged at threads={n}",
            )


@pytest.mark.cuda_equivalence
@pytest.mark.robot_smoke
def test_ee_named_root_attached_zero_gradient_hessian(request):
    """A named fixed target welded to the world ROOT on a fixed base (go2-fixed
    'imu_joint': trunk is the root link) has a constant pose, so the gradient and
    hessian are IDENTICALLY ZERO. Codegen used to refuse this config with
    NotImplementedError; it now emits a zero-fill inner (2026-08-26). Gates:
    registration succeeds, pose still matches the oracle (build sanity), and the
    gradient/hessian outputs are exactly 0.0 (not merely small)."""
    robot_id, base_mode, ee_joint_name = "go2", "fixed", "imu_joint"
    spec = _resolve_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {robot_id}-{base_mode}: {exc}")
    urdf = _urdf_path(resolved)

    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    robot = project_model.robot
    nv = project_model.nv
    fixed_names = {fj.name for fj in robot.fixed_joints}
    if ee_joint_name not in fixed_names:
        pytest.skip(f"{robot_id}-{base_mode}: '{ee_joint_name}' is not a fixed joint")
    # The case is only the root-attached one if the target's parent has no
    # movable joint — guard the fixture assumption rather than silently testing
    # a different (chain-bearing) configuration.
    fj = next(f for f in robot.fixed_joints if f.name == ee_joint_name)
    fj_parent = fj.get_parent()
    parent_joint = robot.get_joint_by_name(fj_parent) if fj_parent not in ("", "-1") else None
    if parent_joint is not None:
        pytest.skip(f"{ee_joint_name} has movable parent {fj_parent!r} on {base_mode} base — "
                    "not the root-attached case this test pins")

    samples = _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)
    batch = len(samples)
    handle = _register(robot_id, base_mode, "named", ee_joint_name, urdf, batch)
    assert handle.num_ees == 1

    q_batch = np.stack([np.asarray(s.q, dtype=np.float32) for s in samples], axis=0)

    # Build sanity: a root-attached pose is CONSTANT — identical (bitwise) across
    # every sample regardless of q, and finite. This is the oracle-free twin of
    # the zero-gradient claim (d pose/dv == 0 <=> pose independent of q).
    pose = np.asarray(handle.end_effector_pose(q_batch)).reshape(batch, 6)
    assert np.all(np.isfinite(pose))
    for si in range(1, batch):
        np.testing.assert_array_equal(
            pose[si], pose[0],
            err_msg="root-attached named-target pose must be q-independent")

    grad = np.asarray(handle.end_effector_pose_gradient(q_batch))
    hess = np.asarray(handle.end_effector_pose_hessian(q_batch))
    assert grad.shape == (batch, 6, nv)
    assert hess.shape == (batch, 6, nv, nv)
    np.testing.assert_array_equal(
        grad, np.zeros_like(grad),
        err_msg="root-attached named-target gradient must be identically zero")
    np.testing.assert_array_equal(
        hess, np.zeros_like(hess),
        err_msg="root-attached named-target hessian must be identically zero")

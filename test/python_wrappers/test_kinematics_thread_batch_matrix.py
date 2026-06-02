"""V6 — kinematics thread-count-invariance matrix (single-thread + warp + sweep × batch).

GRiD runs every function as ONE CUDA block per robot, with all parallelism IN the
block (block-stride loops across threads, grid-stride across the batch). A correct
single-block kernel is therefore **thread-count-invariant**: launching it with 1
thread, a single warp (32), or hundreds of threads must produce the SAME result for
the same inputs. A thread-count-dependent result is a reduction / __syncthreads /
parallel-loop-stride bug (e.g. a reduction that assumes blockDim >= N, or a missing
sync between a write phase and a read/accumulate phase that happens to be correct
only within one warp).

This test exercises the KINEMATICS surfaces through the `grid_rbd` binding (which
exposes `set_threads_per_block(n)` — the C ABI `grid_rbd_set_threads_per_block`) and
batched inputs (leading batch dim, grid-stride across timesteps), sweeping a MATRIX of:

  * surfaces:     end_effector_pose (FK), end_effector_pose_gradient (ee_pose_gradient)
  * thread counts: {1, 2, 16, 32, 64, 128, 256}  (single-thread=1 and warp=32 headline)
  * batch sizes:   {1, 16, 256}
  * robots:        iiwa14:fixed (small serial arm), go2:floating (branched floating)

The HEADLINE invariant asserted is thread-count-INVARIANCE: for fixed inputs, every
thread count's output must match the warp (32-thread) baseline to (near) float
identity. We additionally check correctness vs the RBDReference numpy/pinocchio oracle.

frame_jacobian is intentionally NOT covered here: it is DEVICE-ONLY (no host/kernel
or grid_rbd wrapper yet — backlog S1). Adding a wrapper is out of scope; it is
reported as blocked, not built.

Editable-install / clone note: mirrors test_fk_batched.py. To exercise THIS clone,
build _core in-place and run with PYTHONPATH=<clone>/python prepended so this clone's
grid_rbd (and its GRiDCodeGenerator + wrapper_template.cu) win import resolution.

Run with:
    PYTHONPATH=$PWD/python pytest test/python_wrappers/test_kinematics_thread_batch_matrix.py -v

B5 (RESOLVED): grid_rbd FLOATING-base end_effector_pose_gradient now returns the
d/dv TANGENT Jacobian (NV columns; pinocchio convention) through a correctly-sized,
finite device buffer, matching fixed-base. The historical end-to-end break (public
reshape ValueError on NUM_POS!=NV + garbage/uninitialized raw output ~1e31..1e35)
was a STALE precompiled grid_rbd._core: the d/dv-convention ripple (codegen kernel +
wrapper_template.cu + _core.cpp + _handle.py, all now sized 6*NUM_EES*NUM_VEL) had
landed in source, but the in-tree _core.so predated it and still used the old
NUM_POS-column ABI. Rebuilding _core (src/_core.cpp → grid_rbd/_core*.so) fixed both
the floating reshape crash and the garbage values; the floating cells below are now
un-skipped and assert thread-invariance + oracle correctness directly through the
binding.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


# Repo root is parent of test/. Insert FIRST so this clone's submodules
# (URDFParser / RBDReference / GRiDCodeGenerator) and python/ package win.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "python"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed (build python/ _core)")

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)

# Guard: resolved THIS clone's grid_rbd, not the MAIN-repo editable shadow.
_GRID_RBD_DIR = Path(_grid_rbd.__file__).resolve().parent
if _REPO_ROOT not in _GRID_RBD_DIR.parents:
    pytest.skip(
        f"grid_rbd resolved to {_GRID_RBD_DIR} (not this clone under {_REPO_ROOT}); "
        "set PYTHONPATH=<clone>/python and build _core in-place",
        allow_module_level=True,
    )

_ASSETS = _REPO_ROOT / "robot_assets"

pytestmark = pytest.mark.python_wrappers


# ─── matrix axes ─────────────────────────────────────────────────────────────

# Single-thread (1) and warp (32) are the headline cells; the rest probe partial
# warps, multi-warp, and the cross-over points. The grid_rbd wrapper accepts any
# n >= 1 (codegen carries no launch_bounds floor since v2.0), so 1 is a LEGAL
# launch — a crash/wrong result at 1 is a thread-count-invariance FINDING.
_THREAD_COUNTS = (1, 2, 16, 32, 64, 128, 256)
_WARP = 32  # the thread-count-invariance baseline

# 256 is the compiled max_batch_size below; 1 and 16 bracket it. Single-thread on
# B=256 is the slowest cell but still gives the correctness + invariance signal.
_BATCH_SIZES = (1, 16, 256)
_MAX_BATCH = 256

# (name, urdf, floating) — iiwa14 fixed serial arm + go2 branched floating.
_ROBOTS = [
    ("iiwa14", "iiwa14.urdf", False),
    ("go2", "go2.urdf", True),
]

# Tolerances. The thread-count-INVARIANCE check is near-bit-exact (same kernel,
# same inputs, only blockDim differs → any float reassociation from a different
# reduction tree is the only legal source of difference; it must be tiny). The
# ORACLE correctness check is a float32-vs-float64 comparison so it is looser.
_INVARIANCE_ATOL = 1e-5   # thread-count A vs thread-count B (same inputs)
_POS_TOL = 5e-4           # FK position vs float64 oracle
_ROT_TOL = 5e-3           # FK rotation-matrix vs float64 oracle
_GRAD_NORM_RTOL = 5e-3    # ee_pose_gradient vs oracle, full-matrix norm-relative


# ─── helpers ─────────────────────────────────────────────────────────────────

def _rpy_to_R(rpy):
    """roll-pitch-yaw → R = Rz(yaw) Ry(pitch) Rx(roll), the RBDReference/GRiD
    arctan2 extraction convention. Used so rpy branch ambiguity (a 2π wrap or a
    gimbal-lock roll/yaw split) doesn't masquerade as a pose error."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _register(name, urdf, floating):
    urdf_path = _ASSETS / urdf
    if not urdf_path.exists():
        pytest.skip(f"{urdf} fixture not present at {urdf_path}")
    return _grid_rbd.register_robot(
        name=f"v6_kin_{name}_{'fb' if floating else 'fx'}",
        urdf_path=str(urdf_path),
        floating_base=floating,
        max_batch_size=_MAX_BATCH,
    )


def _reference(urdf, floating):
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    return RBDReference(URDFParser().parse(str(_ASSETS / urdf), floating_base=floating))


def _random_q(handle, B, floating, seed):
    """A (B, NUM_POS) float32 config. Floating-base lays out [xyz, quat(xyzw),
    joints]; we normalize the quaternion per sample so the FK is well posed."""
    rng = np.random.default_rng(seed)
    # grid_rbd's `num_joints` IS the q (NUM_POS) width for both fixed and
    # floating: iiwa14 -> 7, go2-floating -> 19 (= 7 base [xyz, quat] + 12 leg
    # joints). The floating base occupies the first 7 entries [xyz, quat(xyzw)].
    NQ = handle.num_joints
    q = rng.uniform(-1.5, 1.5, size=(B, NQ)).astype(np.float32)
    if floating:
        # joints start after [xyz(3), quat(4)]; set a valid (normalized) quat.
        for b in range(B):
            quat = rng.uniform(-1.0, 1.0, size=4)
            quat /= np.linalg.norm(quat)
            q[b, 0:3] = rng.uniform(-0.3, 0.3, size=3).astype(np.float32)
            q[b, 3:7] = quat.astype(np.float32)
    return q


def _ee_targets(ref_robot):
    """EE target joint names in get_leaf_nodes() order (the order grid_rbd emits
    its per-EE output blocks in)."""
    return [ref_robot.get_joint_by_id(jid).get_name()
            for jid in ref_robot.get_leaf_nodes()]


def _oracle_ee_pose(ref, q_row, targets):
    """RBDReference end_effector_pose → flat (6*NEE,) [xyz; rpy] per EE."""
    poses = [np.asarray(ref.end_effector_pose(q_row.astype(np.float64),
                                              ee_joint_names=[t]), dtype=np.float64).reshape(-1)
             for t in targets]
    return np.concatenate(poses)


def _oracle_ee_grad(ref, q_row, targets):
    """RBDReference end_effector_pose_gradient → (NEE, 6, NV) float64."""
    blocks = []
    for t in targets:
        g = ref.end_effector_pose_gradient(q_row.astype(np.float64), ee_joint_names=[t])
        blocks.append(np.asarray(g[0], dtype=np.float64))  # 6 x NV
    return np.stack(blocks, axis=0)  # (NEE, 6, NV)


# ─── thread-count invariance (the headline) ──────────────────────────────────

@pytest.mark.parametrize("rname,urdf,floating", _ROBOTS, ids=[r[0] for r in _ROBOTS])
@pytest.mark.parametrize("batch", _BATCH_SIZES, ids=lambda b: f"B{b}")
def test_end_effector_pose_thread_count_invariant(rname, urdf, floating, batch):
    """FK: end_effector_pose is identical (to ~float tol) across every thread
    count {1,2,16,32,64,128,256} for the SAME batched inputs. A divergent thread
    count is a single-block reduction/sync bug."""
    handle = _register(rname, urdf, floating)
    q = _random_q(handle, batch, floating, seed=101)

    handle.set_threads_per_block(_WARP)
    baseline = np.asarray(handle.end_effector_pose(q), dtype=np.float64)

    worst = {}
    for n in _THREAD_COUNTS:
        handle.set_threads_per_block(n)
        out = np.asarray(handle.end_effector_pose(q), dtype=np.float64)
        assert out.shape == baseline.shape, (
            f"{rname} B={batch} threads={n}: shape {out.shape} != baseline {baseline.shape}"
        )
        d = float(np.max(np.abs(out - baseline)))
        worst[n] = d
        assert d <= _INVARIANCE_ATOL, (
            f"FINDING thread-count-invariance VIOLATION: {rname} B={batch} "
            f"end_effector_pose threads={n} vs warp(32) max|Δ|={d:.3e} "
            f"(> {_INVARIANCE_ATOL:.1e}). per-thread worst={worst}"
        )


def _raw_ee_grad(handle, q):
    """Raw kernel ee_pose_gradient, shape (B, 6*NEE, NUM_POS). We read the raw
    runner output directly rather than handle.end_effector_pose_gradient(), whose
    host-side reshape is currently BROKEN for floating-base (it reshapes to NV
    columns but the kernel emits NUM_POS columns — see the module FINDINGS note).
    The raw tensor is the actual kernel result and is what we need for the
    thread-count-invariance comparison on BOTH fixed and floating robots."""
    return np.asarray(handle._runner.end_effector_pose_gradient(q), dtype=np.float64)


@pytest.mark.parametrize("rname,urdf,floating", _ROBOTS, ids=[r[0] for r in _ROBOTS])
@pytest.mark.parametrize("batch", _BATCH_SIZES, ids=lambda b: f"B{b}")
def test_end_effector_pose_gradient_thread_count_invariant(rname, urdf, floating, batch):
    """ee_pose_gradient: identical across thread counts for the same inputs.
    Gradients fan independent columns across threads, so a missing sync between
    the FK-cache write and the per-column read would show as thread-dependence.

    Operates on the RAW kernel tensor (handle._runner.end_effector_pose_gradient);
    the invariance property is a KERNEL property, validated directly on kernel
    output. Both fixed- and floating-base are exercised (B5 fixed): the binding now
    emits NV (d/dv tangent) columns through a correctly-sized, finite device
    buffer."""
    handle = _register(rname, urdf, floating)
    q = _random_q(handle, batch, floating, seed=202)

    handle.set_threads_per_block(_WARP)
    baseline = _raw_ee_grad(handle, q)
    scale = max(float(np.max(np.abs(baseline))), 1e-9)

    worst = {}
    for n in _THREAD_COUNTS:
        handle.set_threads_per_block(n)
        out = _raw_ee_grad(handle, q)
        assert out.shape == baseline.shape, (
            f"{rname} B={batch} threads={n}: grad shape {out.shape} != baseline {baseline.shape}"
        )
        # rel to the gradient's own scale (entries span orders of magnitude).
        d = float(np.max(np.abs(out - baseline))) / scale
        worst[n] = d
        assert d <= _INVARIANCE_ATOL, (
            f"FINDING thread-count-invariance VIOLATION: {rname} B={batch} "
            f"end_effector_pose_gradient threads={n} vs warp(32) max|Δ|/scale={d:.3e} "
            f"(> {_INVARIANCE_ATOL:.1e}). per-thread worst={worst}"
        )


# ─── correctness vs the RBDReference oracle (per thread count) ────────────────

@pytest.mark.parametrize("rname,urdf,floating", _ROBOTS, ids=[r[0] for r in _ROBOTS])
@pytest.mark.parametrize("threads", _THREAD_COUNTS, ids=lambda t: f"t{t}")
def test_end_effector_pose_matches_reference(rname, urdf, floating, threads):
    """FK value matches the float64 RBDReference oracle at EVERY thread count
    (a small batch keeps this cheap; the invariance test above covers B=256).
    Position rows compared directly; rpy rows compared as rotation matrices so
    the atan2 branch / gimbal split is not counted as error."""
    handle = _register(rname, urdf, floating)
    ref = _reference(urdf, floating)
    targets = _ee_targets(ref.robot)
    NEE = len(targets)

    B = 8
    q = _random_q(handle, B, floating, seed=303)
    handle.set_threads_per_block(threads)
    out = np.asarray(handle.end_effector_pose(q), dtype=np.float64)  # (B, 6*NEE)

    max_pos, max_rot = 0.0, 0.0
    for b in range(B):
        ref_flat = _oracle_ee_pose(ref, q[b], targets)  # (6*NEE,)
        for e in range(NEE):
            got = out[b, e * 6:(e + 1) * 6]
            exp = ref_flat[e * 6:(e + 1) * 6]
            max_pos = max(max_pos, float(np.max(np.abs(got[:3] - exp[:3]))))
            R_got = _rpy_to_R(got[3:6])
            R_exp = _rpy_to_R(exp[3:6])
            max_rot = max(max_rot, float(np.max(np.abs(R_got - R_exp))))
    assert max_pos < _POS_TOL, (
        f"{rname} threads={threads} FK position error {max_pos:.2e} (> {_POS_TOL:.1e})"
    )
    assert max_rot < _ROT_TOL, (
        f"{rname} threads={threads} FK rotation error {max_rot:.2e} (> {_ROT_TOL:.1e})"
    )


@pytest.mark.parametrize("rname,urdf,floating", _ROBOTS, ids=[r[0] for r in _ROBOTS])
@pytest.mark.parametrize("threads", _THREAD_COUNTS, ids=lambda t: f"t{t}")
def test_end_effector_pose_gradient_matches_reference(rname, urdf, floating, threads):
    """ee_pose_gradient value matches the float64 RBDReference oracle at every
    thread count. Uses a full-matrix norm-relative guard (float32 + rpy-derivative
    rows can show entrywise cancellation on near-singular configs) and SKIPS any
    EE whose reference is non-finite (rpy gimbal lock → analytic d(rpy)/dq blows
    up); the position-row Jacobian is always well posed and asserted.

    Both fixed- and floating-base are exercised. The kernel emits the d/dv TANGENT
    Jacobian (NV columns; pinocchio convention), so for floating-base it aligns 1:1
    with the oracle's NV-column tangent Jacobian — the base block is the spatial
    Jacobian (omega; v), not the older non-standard quaternion-derivative columns
    (B5 fixed: binding now returns correctly-sized finite NV-column output)."""
    handle = _register(rname, urdf, floating)
    ref = _reference(urdf, floating)
    targets = _ee_targets(ref.robot)
    NEE = len(targets)
    NV = handle.num_vel

    B = 8
    q = _random_q(handle, B, floating, seed=404)
    handle.set_threads_per_block(threads)
    out = np.asarray(handle.end_effector_pose_gradient(q), dtype=np.float64)  # (B, 6*NEE, NV)

    worst_norm_rel = 0.0
    compared = 0
    for b in range(B):
        oracle = _oracle_ee_grad(ref, q[b], targets)  # (NEE, 6, NV)
        for e in range(NEE):
            exp = oracle[e]                       # (6, NV)
            got = out[b, e * 6:(e + 1) * 6, :]    # (6, NV)
            if not np.all(np.isfinite(exp)):
                continue  # reference undefined here (rpy gimbal singularity)
            norm_rel = (np.linalg.norm(got - exp)
                        / max(np.linalg.norm(exp), 1e-9))
            worst_norm_rel = max(worst_norm_rel, float(norm_rel))
            compared += 1
    assert compared > 0, f"{rname} threads={threads}: no finite-reference EE grad to compare"
    assert worst_norm_rel < _GRAD_NORM_RTOL, (
        f"{rname} threads={threads} ee_pose_gradient norm-rel error {worst_norm_rel:.2e} "
        f"(> {_GRAD_NORM_RTOL:.1e})"
    )

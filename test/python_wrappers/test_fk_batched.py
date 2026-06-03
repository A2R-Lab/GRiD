"""CUDA-equivalence test for the large-batch FK path (`handle.fk_batched`).

This is the committed coverage for the warp/thread batched-FK surface
(`ee_pose_inner_{thread,warp}` + `ee_pose_fk_batched_kernel`/`_host` +
`grid_rbd.fk_batched`). It was previously validated only ad-hoc by the
implementing agent with no committed test; this closes that gap.

What it covers
--------------
  * Both per-sample variants: `use_warp=False` (one block/thread per sample)
    and `use_warp=True` (one warp per sample), confirming they agree.
  * B > 1 (B = 64) distinct random configurations, batch-major layout
    (`q[b*NUM_POS + j]` in → `pose7[b*7 + k]` out).
  * A serial 7R arm (iiwa14) AND a non-iiwa14 robot — gen3 (a different 7R
    serial arm) and go2 (a branched, 12-DoF quadruped) — to lock the
    de-hardcoded-from-iiwa14 generalization (serial + branched trees).
  * Per-sample agreement of the returned pos+quat pose against the
    `RBDReference.end_effector_pose` oracle (rotation matrix compared, so
    the quaternion sign convention is irrelevant; position compared directly).
  * Floating-base / mimic robots: the batched inner is intentionally absent
    (the codegen skips it), so `fk_batched` must *raise* rather than crash —
    those robots route through `end_effector_pose`.

Editable-install workaround
---------------------------
The venv's editable `grid_rbd` shadows to the MAIN repo. To exercise THIS
clone instead, build the `_core` extension in-place here
(`python bindings/setup.py build_ext --inplace`) and run with
`PYTHONPATH=<clone>/bindings` prepended so this clone's `grid_rbd` package (and,
via `grid_rbd._compile.repo_root()`, this clone's GRiDCodeGenerator +
wrapper_template.cu) win import resolution.

Run with:
    PYTHONPATH=$PWD/bindings pytest test/python_wrappers/test_fk_batched.py -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


# Repo root is parent of `test/`. Insert FIRST so this clone's submodules
# (URDFParser / RBDReference / GRiDCodeGenerator) and `bindings/` package win.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "bindings"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─── skip preconditions ─────────────────────────────────────────────────────

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed (build bindings/ _core)")

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)

# Guard: make sure we resolved THIS clone's grid_rbd, not the MAIN-repo shadow.
# (If a sibling agent's editable install wins, fk_batched coverage would be
# meaningless for this branch.)
_GRID_RBD_DIR = Path(_grid_rbd.__file__).resolve().parent
if _REPO_ROOT not in _GRID_RBD_DIR.parents:
    pytest.skip(
        f"grid_rbd resolved to {_GRID_RBD_DIR} (not this clone under {_REPO_ROOT}); "
        "set PYTHONPATH=<clone>/bindings and build _core in-place",
        allow_module_level=True,
    )

_ASSETS = _REPO_ROOT / "robot_assets"


pytestmark = pytest.mark.python_wrappers


_B = 64                       # batch size (B > 1)
_POS_TOL = 5e-4               # float32 FK position tolerance
_ROT_TOL = 5e-3              # float32 rotation-matrix tolerance
_VARIANT_TOL = 1e-5          # warp vs thread variant should match closely


# ─── small rotation helpers ─────────────────────────────────────────────────

def _quat_wxyz_to_R(q):
    """Unit quaternion (w, x, y, z) → 3x3 rotation matrix."""
    w, x, y, z = q
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _rpy_to_R(rpy):
    """roll-pitch-yaw → R = Rz(yaw) Ry(pitch) Rx(roll), matching the
    RBDReference's arctan2 extraction convention."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ─── fixtures ───────────────────────────────────────────────────────────────

# (name, urdf, expected num_pos, serial?) — gen3/go2 lock generalization
# beyond the iiwa14 hardcoding (different 7R serial arm + branched quadruped).
_ROBOTS = [
    ("iiwa14", "iiwa14.urdf", 7, True),
    ("gen3", "gen3.urdf", 7, True),
    ("go2", "go2.urdf", 12, False),
]


def _register(name, urdf):
    urdf_path = _ASSETS / urdf
    if not urdf_path.exists():
        pytest.skip(f"{urdf} fixture not present at {urdf_path}")
    return _grid_rbd.register_robot(
        name=f"fk_batched_pytest_{name}",
        urdf_path=str(urdf_path),
        floating_base=False,
        max_batch_size=_B,
    )


def _reference(urdf):
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    return RBDReference(URDFParser().parse(str(_ASSETS / urdf), floating_base=False))


# ─── tests ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,urdf,n_pos,serial", _ROBOTS,
                         ids=[r[0] for r in _ROBOTS])
def test_fk_batched_matches_reference(name, urdf, n_pos, serial):
    """Both variants on B=64 distinct configs match the RBDReference oracle,
    and the warp/thread variants agree with each other."""
    handle = _register(name, urdf)
    ref = _reference(urdf)

    assert handle.num_joints == n_pos
    NJ = handle.num_joints

    rng = np.random.default_rng(7)
    q = rng.uniform(-2.5, 2.5, size=(_B, NJ)).astype(np.float32)

    pose7_thread = handle.fk_batched(q, use_warp=False)
    pose7_warp = handle.fk_batched(q, use_warp=True)

    # Layout: (B, 7) = [tx,ty,tz, qw,qx,qy,qz].
    assert pose7_thread.shape == (_B, 7)
    assert pose7_warp.shape == (_B, 7)

    # The two per-sample mappings must produce identical poses (account for
    # quaternion double-cover by comparing rotation matrices + position).
    for b in range(_B):
        assert np.allclose(pose7_thread[b, :3], pose7_warp[b, :3], atol=_VARIANT_TOL), \
            f"{name}: thread/warp position mismatch at sample {b}"
        Rt = _quat_wxyz_to_R(pose7_thread[b, 3:])
        Rw = _quat_wxyz_to_R(pose7_warp[b, 3:])
        assert np.max(np.abs(Rt - Rw)) < _VARIANT_TOL, \
            f"{name}: thread/warp rotation mismatch at sample {b}"

    # Oracle: RBDReference.end_effector_pose returns a list of [xyz; rpy],
    # ordered by get_leaf_nodes(); entry 0 is the leaf the batched kernel
    # targets (default_ee = get_leaf_nodes()[0]).
    max_pos_err = 0.0
    max_rot_err = 0.0
    for b in range(_B):
        ee_ref = ref.end_effector_pose(q[b].astype(np.float64))[0].flatten()
        pos_ref = ee_ref[:3]
        R_ref = _rpy_to_R(ee_ref[3:6])
        for pose7 in (pose7_thread, pose7_warp):
            pos = pose7[b, :3]
            R = _quat_wxyz_to_R(pose7[b, 3:])
            max_pos_err = max(max_pos_err, float(np.max(np.abs(pos - pos_ref))))
            max_rot_err = max(max_rot_err, float(np.max(np.abs(R - R_ref))))

    assert max_pos_err < _POS_TOL, f"{name}: max position error {max_pos_err:.2e}"
    assert max_rot_err < _ROT_TOL, f"{name}: max rotation error {max_rot_err:.2e}"


@pytest.mark.parametrize("use_warp", [False, True], ids=["thread", "warp"])
def test_fk_batched_variant_invariant_to_block_threads(use_warp):
    """The dedicated restricted-execution FK variants use a FIXED slice of each
    block — the single-thread variant (use_warp=False) only thread 0, the
    single-warp variant (use_warp=True) only warp 0 — so their output MUST be
    invariant to the block thread count set via set_threads_per_block. (Reference
    correctness of each variant is covered by test_fk_batched_matches_reference;
    this isolates the one-thread / one-warp restriction property.) The warp variant
    needs >=32 threads (a full warp); the thread variant works at any >=1."""
    handle = _register("iiwa14", "iiwa14.urdf")
    NJ = handle.num_joints
    q = np.random.default_rng(13).uniform(-2.5, 2.5, size=(_B, NJ)).astype(np.float32)
    thread_counts = (32, 64, 128, 256) if use_warp else (1, 2, 32, 128, 256)

    handle.set_threads_per_block(thread_counts[0])
    baseline = np.asarray(handle.fk_batched(q, use_warp=use_warp), dtype=np.float64)
    for n in thread_counts:
        handle.set_threads_per_block(n)
        out = np.asarray(handle.fk_batched(q, use_warp=use_warp), dtype=np.float64)
        assert out.shape == baseline.shape
        d = float(np.max(np.abs(out - baseline)))
        assert d < _VARIANT_TOL, (
            f"{'warp' if use_warp else 'thread'} FK variant NOT invariant to block "
            f"size: threads={n} differs from threads={thread_counts[0]} by {d:.2e} "
            f"(the variant must use only thread-0 / warp-0 regardless of block size)"
        )


def test_fk_batched_layout_distinct_per_sample():
    """Confirms the batch-major `q[b*N]` / `pose7[b*7]` layout actually carries
    per-sample data: distinct configs must yield distinct poses, and a shuffled
    batch must produce the correspondingly shuffled poses."""
    handle = _register("iiwa14", "iiwa14.urdf")
    NJ = handle.num_joints
    rng = np.random.default_rng(11)
    q = rng.uniform(-2.0, 2.0, size=(_B, NJ)).astype(np.float32)

    pose7 = handle.fk_batched(q, use_warp=False)
    # distinct random configs ⇒ distinct poses (no row is accidentally shared)
    assert len({tuple(np.round(p, 5)) for p in pose7}) == _B

    # Permute the batch; each output row must follow its input row.
    perm = rng.permutation(_B)
    pose7_perm = handle.fk_batched(q[perm], use_warp=False)
    assert np.allclose(pose7_perm, pose7[perm], atol=1e-5)


def test_fk_batched_absent_for_floating_base():
    """Floating-base robots route through end_effector_pose; the batched FK
    inner is intentionally absent, so fk_batched must raise (rc=3) — not crash
    or silently return garbage. end_effector_pose must still work."""
    urdf_path = _ASSETS / "iiwa14.urdf"
    if not urdf_path.exists():
        pytest.skip("iiwa14 fixture not present")
    handle = _grid_rbd.register_robot(
        name="fk_batched_pytest_iiwa14_floating",
        urdf_path=str(urdf_path),
        floating_base=True,
        max_batch_size=_B,
    )
    assert handle.floating_base is True
    NJ = handle.num_joints
    q = np.zeros((4, NJ), dtype=np.float32)
    with pytest.raises(Exception):
        handle.fk_batched(q, use_warp=False)
    # Sanity: the standard EE-pose path is present for floating-base.
    out = handle.end_effector_pose(q)
    assert out.shape[0] == 4

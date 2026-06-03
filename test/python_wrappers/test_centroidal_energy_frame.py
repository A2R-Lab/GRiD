"""F2 binding coverage: centroidal / energy / general-frame kinematics.

Exercises the grid_rbd handle methods newly bound in F2 and asserts numerical
agreement with the RBDReference numpy oracle at float32 precision:

  com / ccrba / energy / generalized_gravity / nonlinear_effects
  frame_jacobian / frame_jacobian_dot / osc_inertia

Robots: iiwa14 (7-DoF serial arm) + go2 (12-DoF branched quadruped, fixed
base) — kept small so the register/compile stays fast. Both are non-mimic.

Run with:
    PYTHONPATH=$PWD/bindings pytest test/python_wrappers/test_centroidal_energy_frame.py -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


# Repo root is parent of `test/`. Insert FIRST so this clone's submodules
# (URDFParser / RBDReference / GRiDCodeGenerator) and `bindings/` win import.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "bindings"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─── skip preconditions ─────────────────────────────────────────────────────

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed (build bindings/ _core)")

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)

_GRID_RBD_DIR = Path(_grid_rbd.__file__).resolve().parent
if _REPO_ROOT not in _GRID_RBD_DIR.parents:
    pytest.skip(
        f"grid_rbd resolved to {_GRID_RBD_DIR} (not this clone under {_REPO_ROOT}); "
        "set PYTHONPATH=<clone>/bindings and build _core in-place",
        allow_module_level=True,
    )

_ASSETS = _REPO_ROOT / "robot_assets"

pytestmark = pytest.mark.python_wrappers


_TOL = 5e-3   # float32 vs float64 cross-precision; some compositions drift ~1e-4
_B = 4        # batch


# (name, urdf) — small non-mimic robots: a serial arm + a branched quadruped.
_ROBOTS = [
    ("iiwa14", "iiwa14.urdf"),
    ("go2", "go2.urdf"),
]


def _register(name, urdf):
    urdf_path = _ASSETS / urdf
    if not urdf_path.exists():
        pytest.skip(f"{urdf} fixture not present at {urdf_path}")
    return _grid_rbd.register_robot(
        name=f"{name}_f2_centroidal_pytest",
        urdf_path=str(urdf_path),
        floating_base=False,
        max_batch_size=8,
    )


def _reference(urdf):
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    return RBDReference(URDFParser().parse(str(_ASSETS / urdf), floating_base=False))


def _leaf_frame_name(ref):
    """Name of the joint the codegen bakes as the frame_jacobian target (the
    first leaf node, matching `gen_frame_jacobian_kernel`'s default_tjid). The
    RBDReference frame_* methods require an explicit frame name (no leaf
    default), so we resolve it here to match the GPU surface's fixed target."""
    robot = ref.robot
    leaf_id = robot.get_leaf_nodes()[0]
    return robot.get_joint_by_id(leaf_id).get_name()


def _samples(handle):
    rng = np.random.default_rng(0)
    NJ = handle.num_joints
    return {
        "q":  rng.standard_normal((_B, NJ)).astype(np.float32),
        "qd": rng.standard_normal((_B, NJ)).astype(np.float32),
    }


def _max_err(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


@pytest.fixture(scope="module", params=_ROBOTS, ids=[r[0] for r in _ROBOTS])
def robot(request):
    name, urdf = request.param
    handle = _register(name, urdf)
    ref = _reference(urdf)
    return handle, ref, _samples(handle)


# ─── centroidal ──────────────────────────────────────────────────────────────

def test_com(robot):
    handle, ref, s = robot
    p_com, j_com = handle.com(s["q"])
    NV = handle.num_vel
    assert p_com.shape == (_B, 3)
    assert j_com.shape == (_B, 3, NV)
    for i, q in enumerate(s["q"]):
        assert _max_err(p_com[i], ref.com(q.astype(np.float64))) < _TOL
        assert _max_err(j_com[i], ref.jacobian_com(q.astype(np.float64))) < _TOL


def test_ccrba(robot):
    handle, ref, s = robot
    A, h = handle.ccrba(s["q"], s["qd"])
    NV = handle.num_vel
    assert A.shape == (_B, 6, NV)
    assert h.shape == (_B, 6)
    for i, (q, qd) in enumerate(zip(s["q"], s["qd"])):
        A_ref, h_ref = ref.ccrba(q.astype(np.float64), qd.astype(np.float64))
        assert _max_err(A[i], A_ref) < _TOL
        assert _max_err(h[i], h_ref) < _TOL


def test_energy(robot):
    handle, ref, s = robot
    en = handle.energy(s["q"], s["qd"], gravity=-9.81)
    assert en.shape == (_B, 3)
    for i, (q, qd) in enumerate(zip(s["q"], s["qd"])):
        ke = ref.kinetic_energy(q.astype(np.float64), qd.astype(np.float64))
        pe = ref.potential_energy(q.astype(np.float64), GRAVITY=-9.81)
        assert _max_err(en[i, 0], ke) < _TOL
        assert _max_err(en[i, 1], pe) < _TOL
        assert _max_err(en[i, 2], ke + pe) < _TOL


# ─── energy / gravity / bias ─────────────────────────────────────────────────

def test_generalized_gravity(robot):
    handle, ref, s = robot
    gg = handle.generalized_gravity(s["q"], gravity=-9.81)
    assert gg.shape == (_B, handle.num_vel)
    for i, q in enumerate(s["q"]):
        assert _max_err(gg[i], ref.generalized_gravity(q.astype(np.float64), GRAVITY=-9.81)) < _TOL


def test_nonlinear_effects(robot):
    handle, ref, s = robot
    nle = handle.nonlinear_effects(s["q"], s["qd"], gravity=-9.81)
    assert nle.shape == (_B, handle.num_vel)
    for i, (q, qd) in enumerate(zip(s["q"], s["qd"])):
        ref_nle = ref.nonlinear_effects(q.astype(np.float64), qd.astype(np.float64), GRAVITY=-9.81)
        assert _max_err(nle[i], ref_nle) < _TOL


# ─── general-frame kinematics ────────────────────────────────────────────────

def test_frame_jacobian(robot):
    handle, ref, s = robot
    fn = _leaf_frame_name(ref)
    J = handle.frame_jacobian(s["q"])
    NV = handle.num_vel
    assert J.shape == (_B, 6, NV)
    for i, q in enumerate(s["q"]):
        assert _max_err(J[i], ref.frame_jacobian(q.astype(np.float64), fn)) < _TOL


def test_frame_jacobian_nondefault_frame(robot):
    """Part B: a RUNTIME non-leaf target_jid + non-LWA reference_frame. Picks a
    mid-chain joint (not the baked leaf-EE) and the WORLD reference frame, and
    checks the GPU surface matches the RBDReference for that exact frame."""
    handle, ref, s = robot
    robot_obj = ref.robot
    leaf_id = robot_obj.get_leaf_nodes()[0]
    # A non-leaf ancestor of the leaf (so the chain has DOFs); fall back to leaf
    # if the chain is trivial.
    ancestors = sorted(robot_obj.get_ancestors_by_id(leaf_id))
    target_id = ancestors[len(ancestors) // 2] if ancestors else leaf_id
    target_name = robot_obj.get_joint_by_id(target_id).get_name()
    NV = handle.num_vel
    for rf_name, rf_code in (("WORLD", 1), ("LOCAL", 0)):
        J = handle.frame_jacobian(s["q"], target_jid=target_id, reference_frame=rf_code)
        assert J.shape == (_B, 6, NV)
        for i, q in enumerate(s["q"]):
            ref_J = ref.frame_jacobian(q.astype(np.float64), target_name, rf_name)
            assert _max_err(J[i], ref_J) < _TOL, f"{rf_name} target={target_id}"
    # And confirm the string reference_frame form works identically.
    J_str = handle.frame_jacobian(s["q"], target_jid=target_id, reference_frame="WORLD")
    J_int = handle.frame_jacobian(s["q"], target_jid=target_id, reference_frame=1)
    assert _max_err(J_str, J_int) == 0.0


def test_frame_jacobian_default_target_explicit_frame(robot):
    """Regression: an explicit reference_frame with the DEFAULT (leaf-EE) target
    must be honored. The C-ABI once gated on `target_jid < 0 || reference_frame
    < 0`, so a default target (sentinel -1) silently dropped the explicit frame
    and returned LWA. Each arg must resolve INDEPENDENTLY."""
    handle, ref, s = robot
    fn = _leaf_frame_name(ref)
    for rf_name, rf_code in (("WORLD", 1), ("LOCAL", 0)):
        # default target (omitted) + explicit non-LWA frame
        J = handle.frame_jacobian(s["q"], reference_frame=rf_code)
        for i, q in enumerate(s["q"]):
            ref_J = ref.frame_jacobian(q.astype(np.float64), fn, rf_name)
            assert _max_err(J[i], ref_J) < _TOL, f"default-target {rf_name}"
    # The explicit-frame result must actually DIFFER from the LWA default
    # (else the frame was dropped) — unless the leaf frame is trivially aligned.
    J_lwa = handle.frame_jacobian(s["q"])
    J_world = handle.frame_jacobian(s["q"], reference_frame="WORLD")
    assert _max_err(J_lwa, J_world) > 1e-3, "explicit WORLD frame was ignored on default target"


def test_frame_jacobian_dot(robot):
    handle, ref, s = robot
    fn = _leaf_frame_name(ref)
    Jd = handle.frame_jacobian_dot(s["q"], s["qd"])
    NV = handle.num_vel
    assert Jd.shape == (_B, 6, NV)
    # Jdot is finite-differenced (1e-4 step) on the GPU and central-differenced
    # in the oracle; float32 FD drift is looser than the value surfaces.
    for i, (q, qd) in enumerate(zip(s["q"], s["qd"])):
        ref_Jd = ref.frame_jacobian_dot(q.astype(np.float64), qd.astype(np.float64), fn)
        assert _max_err(Jd[i], ref_Jd) < 5e-2


def test_osc_inertia(robot):
    handle, ref, s = robot
    fn = _leaf_frame_name(ref)
    L = handle.osc_inertia(s["q"])
    assert L.shape == (_B, 6, 6)
    # Lambda = (J Minv J^T)^-1 only exists when the task matrix is full-rank.
    # A branched robot whose leaf chain has < 6 DoF (e.g. go2's 3-DoF leg) gives
    # a rank-deficient task matrix → the 6x6 inverse blows up identically-but-
    # differently in float32 GRiD vs float64 oracle. Gate on conditioning
    # (mirrors the S1 host test's cond < 1e8 guard); only well-conditioned
    # samples carry a meaningful comparison. float32 round-trip through the 6x6
    # inverse is looser than the value surfaces (5e-3, per the S1 gate).
    checked = 0
    for i, q in enumerate(s["q"]):
        q64 = q.astype(np.float64)
        J = ref.frame_jacobian(q64, fn)
        task = J @ ref.minv(q64) @ J.T
        if np.linalg.cond(task) >= 1e8:
            continue
        checked += 1
        L_ref = ref.osc_inertia(q64, fn)
        # scale-relative gate (mirrors S1 host test atol = max(5e-3, 5e-3*scale)):
        # Lambda entries can be O(1e2), so a flat 5e-3 absolute tol is too tight.
        scale = float(np.max(np.abs(L_ref)))
        assert _max_err(L[i], L_ref) < max(5e-3, 5e-3 * scale)
    if checked == 0:
        pytest.skip("no well-conditioned task-inertia sample (leaf chain rank-deficient)")

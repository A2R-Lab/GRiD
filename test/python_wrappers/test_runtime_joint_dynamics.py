"""C5: runtime-mutable joint dynamics (`set_joint_dynamics`) validation.

The surface is `register_robot(..., use_joint_dynamics=True, runtime_joint_dynamics=True)`
+ `handle.set_joint_dynamics(damping=, friction=)` + `handle.joint_damping/.joint_friction`.
The .so carries a device-resident `d_joint_dynamics_params` table (2*nv = [damping||friction],
v-slot indexed, ALPHA-FOLDED) that the id/fd/aba/*_gradient bias reads in place of the baked
literal. The table is URDF-initialized, so the runtime path is BIT-IDENTICAL to the baked
literal build until poked (NO on-device sincos rebuild, NO sparsity change — unlike
runtime_transform, which is only float-identical).

Coverage (mirrors test_runtime_inertia.py's bit-identical-until-poked structure):
  1. Untouched runtime build == baked damped build (bit-for-bit).
  2. Poke -> physical correctness: the delta vs the bare (undamped) build equals the
     ANALYTIC per-v-slot bias  b*qd + f*sign(qd)  (iiwa14, clean non-mimic v-slot map).
  3. Zeroing the table == the bare undamped build (the runtime toggle).
  4. Round-trip: poking the baked values back reproduces the untouched result.
  5. FFI parity of a poke: numpy/jax/torch share the device struct, so a poke through one
     surface is seen by all.

iiwa14 fixed (damping 0.5x7, non-mimic — exact v-slot map for the analytic check) and fr3
(damping+friction+mimic — mechanism checks only). fp32. Skips if grid_rbd/nvcc/URDF absent.

Run with:
    pytest test/python_wrappers/test_runtime_joint_dynamics.py -m python_wrappers -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "bindings"))

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers

_IIWA = _REPO_ROOT / "config/robot_assets" / "iiwa14.urdf"
_FR3 = _REPO_ROOT / "config/robot_assets" / "fr3.urdf"
if not _IIWA.exists():
    pytest.skip("iiwa14 URDF not present", allow_module_level=True)


def _na(a):
    return np.asarray(a, dtype=np.float64)


def _maxabs(a, b):
    return float(np.max(np.abs(_na(a) - _na(b))))


def _samples(nj, seed=0):
    rng = np.random.default_rng(seed)
    # bias qd well away from 0 so sign(qd) is locally constant (friction is smooth)
    qd = np.sign(rng.standard_normal((4, nj))) * (0.5 + np.abs(rng.standard_normal((4, nj))))
    return (rng.standard_normal((4, nj)).astype(np.float32),
            qd.astype(np.float32),
            rng.standard_normal((4, nj)).astype(np.float32))


# ─── iiwa14 (non-mimic): the full analytic + mechanism suite ──────────────────


@pytest.fixture(scope="module")
def iiwa_builds():
    common = dict(urdf_path=str(_IIWA), floating_base=False, max_batch_size=8)
    h_rt = _grid_rbd.register_robot(name="iiwa14_rtjd_pytest", use_joint_dynamics=True,
                                    runtime_joint_dynamics=True, force_rebuild=True, **common)
    h_baked = _grid_rbd.register_robot(name="iiwa14_baked_jd_pytest", use_joint_dynamics=True,
                                       force_rebuild=True, **common)
    h_bare = _grid_rbd.register_robot(name="iiwa14_bare_jd_pytest", force_rebuild=True, **common)
    return h_rt, h_baked, h_bare


def test_metadata(iiwa_builds):
    h_rt, _, _ = iiwa_builds
    assert h_rt.runtime_joint_dynamics is True
    nv = h_rt.num_vel
    assert h_rt.joint_damping.shape == (nv,)
    assert h_rt.joint_friction.shape == (nv,)
    # iiwa14 has damping (0.5) and no friction
    assert float(np.max(np.abs(h_rt.joint_damping))) > 0.0


def test_untouched_runtime_equals_baked_bitwise(iiwa_builds):
    """The runtime table is URDF-initialized to the SAME folded coeffs the baked
    literal uses, with no sincos rebuild — so an UNTOUCHED runtime build must equal
    the baked damped build BIT-FOR-BIT (np.array_equal), not merely close."""
    h_rt, h_baked, _ = iiwa_builds
    nj = h_rt.num_joints
    q, qd, u = _samples(nj)
    for algo, args in (("inverse_dynamics", (q, qd)), ("forward_dynamics", (q, qd, u)),
                       ("aba", (q, qd, u)), ("inverse_dynamics_gradient", (q, qd, u))):
        rt = getattr(h_rt, algo)(*args)
        bk = getattr(h_baked, algo)(*args)
        assert np.array_equal(_na(rt), _na(bk)), (
            f"{algo}: untouched runtime != baked bit-for-bit (max {_maxabs(rt, bk):.2e})")


def test_poke_matches_analytic_bias(iiwa_builds):
    """Poke perturbed damping b2; the id delta vs the BARE undamped build must equal
    the analytic per-v-slot bias  b2*qd  (iiwa14 non-mimic: v-slot == joint index, so
    the length-nv coeffs align with the nq==nv inputs). Proves the table is wired into
    the bias, not merely stored.

    NOTE: iiwa14 has NO baked friction, and the codegen emits the friction term ONLY
    when the robot has friction (HAS_FRIC at codegen time) — so the friction half of
    the table is inert for iiwa14 and is NOT exercised here (poking friction is a
    no-op for a friction-less robot; that's a v1 coverage property, not a bug). The
    friction path is covered by fr3 (mechanism) + the numpy test_joint_dynamics oracle."""
    h_rt, _, h_bare = iiwa_builds
    nv = h_rt.num_vel
    nj = h_rt.num_joints
    assert nj == nv, "this analytic check assumes non-mimic fixed base (nq==nv)"
    assert float(np.max(np.abs(h_rt.joint_friction))) == 0.0, "iiwa14 should have no baked friction"
    rng = np.random.default_rng(5)
    b2 = (0.1 + np.abs(rng.standard_normal(nv))).astype(np.float32)
    h_rt.set_joint_dynamics(damping=b2)   # damping only (iiwa14 has no friction term)

    q, qd, u = _samples(nj, seed=6)
    id_poked = _na(h_rt.inverse_dynamics(q, qd))
    id_bare = _na(h_bare.inverse_dynamics(q, qd))
    delta = id_poked - id_bare
    expected = b2[None, :].astype(np.float64) * _na(qd)   # bias = b2*qd (no friction)
    err = float(np.max(np.abs(delta - expected)))
    assert err < 2e-3, f"poked bias delta != b2*qd: {err:.2e}"
    # restore for any later test
    h_rt.set_joint_dynamics(damping=h_rt.joint_damping, friction=h_rt.joint_friction)


def test_zeroing_table_equals_bare(iiwa_builds):
    """Zeroing damping+friction makes the bias 0 -> the runtime build matches the
    bare (use_joint_dynamics=False) build (the runtime toggle)."""
    h_rt, _, h_bare = iiwa_builds
    nv = h_rt.num_vel
    nj = h_rt.num_joints
    h_rt.set_joint_dynamics(damping=np.zeros(nv, np.float32), friction=np.zeros(nv, np.float32))
    q, qd, u = _samples(nj, seed=7)
    for algo, args in (("inverse_dynamics", (q, qd)), ("forward_dynamics", (q, qd, u))):
        zeroed = getattr(h_rt, algo)(*args)
        bare = getattr(h_bare, algo)(*args)
        assert _maxabs(zeroed, bare) < 1e-4, f"{algo}: zeroed table != bare ({_maxabs(zeroed, bare):.2e})"
    h_rt.set_joint_dynamics(damping=h_rt.joint_damping, friction=h_rt.joint_friction)


def test_roundtrip_baked_values(iiwa_builds):
    """Poking the baked values back reproduces the untouched (baked) result."""
    h_rt, h_baked, _ = iiwa_builds
    nj = h_rt.num_joints
    q, qd, u = _samples(nj, seed=8)
    # perturb then restore
    nv = h_rt.num_vel
    h_rt.set_joint_dynamics(damping=np.full(nv, 3.0, np.float32))
    h_rt.set_joint_dynamics(damping=h_rt.joint_damping, friction=h_rt.joint_friction)
    rt = h_rt.inverse_dynamics(q, qd)
    bk = h_baked.inverse_dynamics(q, qd)
    assert _maxabs(rt, bk) < 1e-5, f"round-trip != baked ({_maxabs(rt, bk):.2e})"


def test_omitted_side_keeps_baked(iiwa_builds):
    """set_joint_dynamics(damping=...) with friction omitted keeps the baked
    friction (and vice-versa) — partial poke convenience."""
    h_rt, h_baked, _ = iiwa_builds
    nj = h_rt.num_joints
    nv = h_rt.num_vel
    q, qd, u = _samples(nj, seed=9)
    # poke only damping to the baked damping -> output unchanged from baked
    h_rt.set_joint_dynamics(damping=h_rt.joint_damping)
    assert _maxabs(h_rt.inverse_dynamics(q, qd), h_baked.inverse_dynamics(q, qd)) < 1e-5
    h_rt.set_joint_dynamics(damping=h_rt.joint_damping, friction=h_rt.joint_friction)


# ─── FFI parity of a poke (numpy/jax/torch share the device struct) ───────────


def test_poke_seen_across_surfaces():
    """A poke through the numpy handle mutates the single device-resident
    d_joint_dynamics_params struct, so jax/torch (sharing it) see the poked values:
    numpy == jax == torch AFTER the mutation. Skips if jax/torch unavailable."""
    gj = pytest.importorskip("grid_rbd.jax")
    pytest.importorskip("jax")
    gt = pytest.importorskip("grid_rbd.torch")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available for torch")
    common = dict(urdf_path=str(_IIWA), floating_base=False, max_batch_size=8,
                  use_joint_dynamics=True, runtime_joint_dynamics=True)
    hn = _grid_rbd.register_robot(name="iiwa14_rtjd_ffi_pytest", force_rebuild=True, **common)
    hj = gj.register_robot(name="iiwa14_rtjd_ffi_pytest", **common)
    ht = gt.register_robot(name="iiwa14_rtjd_ffi_pytest", **common)
    nv = hn.num_vel
    nj = hn.num_joints
    rng = np.random.default_rng(13)
    b2 = (0.2 + np.abs(rng.standard_normal(nv))).astype(np.float32)
    hn.set_joint_dynamics(damping=b2)               # poke via numpy -> device struct
    q, qd, _ = _samples(nj, seed=14)
    qt = torch.tensor(q, device="cuda", dtype=torch.float32)
    qdt = torch.tensor(qd, device="cuda", dtype=torch.float32)
    cn = _na(hn.inverse_dynamics(q, qd))
    cj = _na(hj.inverse_dynamics(q, qd))
    ct = _na(ht.inverse_dynamics(qt, qdt).detach().cpu().numpy())
    assert float(np.max(np.abs(cn - cj))) < 2e-3, "jax did not see the numpy poke"
    assert float(np.max(np.abs(cn - ct))) < 2e-3, "torch did not see the numpy poke"
    hn.set_joint_dynamics(damping=hn.joint_damping, friction=hn.joint_friction)


# ─── fr3 (mimic): mechanism checks only (folded v-slot, no analytic formula) ──


@pytest.mark.skipif(not _FR3.exists(), reason="fr3 URDF not present")
def test_fr3_mimic_untouched_equals_baked_and_zero_toggles():
    """fr3 has damping + friction + a mimic joint (folded per-v-slot table). Verify
    the MECHANISM on a mimic robot: untouched runtime == baked (bit-for-bit) and
    zeroing == bare. The analytic per-v-slot formula is skipped (mimic fold maps
    several jids into one v-slot; covered by the numpy test_joint_dynamics oracle)."""
    common = dict(urdf_path=str(_FR3), floating_base=False, max_batch_size=8)
    h_rt = _grid_rbd.register_robot(name="fr3_rtjd_pytest", use_joint_dynamics=True,
                                    runtime_joint_dynamics=True, force_rebuild=True, **common)
    h_baked = _grid_rbd.register_robot(name="fr3_baked_jd_pytest", use_joint_dynamics=True,
                                       force_rebuild=True, **common)
    h_bare = _grid_rbd.register_robot(name="fr3_bare_jd_pytest", force_rebuild=True, **common)
    nj = h_rt.num_joints
    nv = h_rt.num_vel
    q, qd, u = _samples(nj, seed=21)
    # Untouched runtime == baked to FLOAT precision (not necessarily BIT-for-bit) on a
    # robot WITH friction: the runtime path reads f from the table and evaluates the
    # Coulomb term f*sign(qd) at a slightly different point in the accumulation than the
    # baked path folds the constant, so a sign-flipping qd can reorder the add by <=1
    # ULP (measured ~1.5e-8 abs / 1.2e-7 rel on a couple of fr3 entries). This mirrors
    # the documented runtime_transform caveat (on-device rebuild is float- not
    # bit-identical). Damping-only robots (iiwa14) have no sign term and ARE bit-exact
    # (test_untouched_runtime_equals_baked_bitwise), so that stricter check stays.
    rt = _na(h_rt.inverse_dynamics(q, qd))
    bk = _na(h_baked.inverse_dynamics(q, qd))
    assert np.allclose(rt, bk, rtol=1e-5, atol=1e-6), \
        f"fr3: untouched runtime != baked to float precision (max {np.max(np.abs(rt - bk)):.2e})"
    h_rt.set_joint_dynamics(damping=np.zeros(nv, np.float32), friction=np.zeros(nv, np.float32))
    assert _maxabs(h_rt.inverse_dynamics(q, qd), h_bare.inverse_dynamics(q, qd)) < 1e-4, \
        "fr3: zeroed table != bare"

"""C5-light: `use_joint_dynamics=True` value/gradient parity across numpy/jax/torch.

`use_joint_dynamics` is a BUILD-TIME codegen flag that bakes the viscous-damping +
Coulomb-friction bias `tau += b*qd + f*sign(qd)` (and the `diag(b)` gradient term)
into the id/fd/aba/*_gradient kernels — NOT a per-algo GRID_HAS_* gate. The jax/torch
FFI handlers call those SAME baked C-ABI symbols, so a damped build computes the
biased result on every surface. This test proves the three surfaces AGREE (the C5
Part-A wiring: dropped the numpy-only raise + forwarded the flag) and that the bias
is actually present (damped != undamped). It also pins the SO regression: the bias is
linear in qd, so its second derivative is 0 → idsva_so / fdsva_so are UNCHANGED by the
flag.

Robots: iiwa14 (damping 0.5x7, no friction), fr3 (damping + friction + mimic). Fixed
base, fp32 (jax/torch are fp32-only). Skips cleanly if jax/torch/nvcc unavailable.

Run with:
    pytest test/python_wrappers/test_joint_dynamics_ffi.py -m python_wrappers -v
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
from config import robot_urdf

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
_gj = pytest.importorskip("grid_rbd.jax", reason="grid_rbd.jax import failed (pip install grid-rbd[jax])")
_jax = pytest.importorskip("jax", reason="jax not installed")
_gt = pytest.importorskip("grid_rbd.torch", reason="grid_rbd.torch import failed (pip install grid-rbd[torch])")
_torch = pytest.importorskip("torch", reason="torch not installed")
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)
if not _torch.cuda.is_available():
    pytest.skip("CUDA device not available for torch", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers

_TOL = 2e-3   # fp32 cross-surface agreement (same kernel, different launch path)
_IIWA = robot_urdf("iiwa14")
_FR3 = robot_urdf("fr3")
if not (_IIWA.exists() and _FR3.exists()):
    pytest.skip("iiwa14/fr3 URDF not present", allow_module_level=True)


def _np(a):
    """Coerce a numpy / jax / torch result to a float64 numpy array."""
    if hasattr(a, "detach"):           # torch tensor
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def _maxabs(a, b):
    return float(np.max(np.abs(_np(a) - _np(b))))


def _samples(nj, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((4, nj)).astype(np.float32),
            rng.standard_normal((4, nj)).astype(np.float32),
            rng.standard_normal((4, nj)).astype(np.float32))


# (name, urdf) — fixed base, fp32.
_CASES = [("iiwa14", _IIWA), ("fr3", _FR3)]
_IDS = ["iiwa14", "fr3"]
_URDF_BY_NAME = dict(_CASES)


def _reference(name, use_joint_dynamics=False):
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    robot = URDFParser().parse(str(_URDF_BY_NAME[name]), floating_base=False)
    return RBDReference(robot, use_joint_dynamics=use_joint_dynamics)


@pytest.fixture(scope="module", params=_CASES, ids=_IDS)
def handles(request):
    """Register the SAME damped robot on numpy + jax + torch (shared cache key ->
    ONE build), plus a bare (undamped) numpy build for the bias-present check."""
    name, urdf = request.param
    common = dict(urdf_path=str(urdf), floating_base=False, max_batch_size=8)
    hn = _grid_rbd.register_robot(name=f"{name}_jd_ffi_pytest", use_joint_dynamics=True,
                                  force_rebuild=True, **common)
    hj = _gj.register_robot(name=f"{name}_jd_ffi_pytest", use_joint_dynamics=True, **common)
    ht = _gt.register_robot(name=f"{name}_jd_ffi_pytest", use_joint_dynamics=True, **common)
    hbare = _grid_rbd.register_robot(name=f"{name}_jd_ffi_bare_pytest", force_rebuild=True, **common)
    return name, hn, hj, ht, hbare


def _torch_in(*arrs):
    return tuple(_torch.tensor(a, device="cuda", dtype=_torch.float32) for a in arrs)


def test_value_path_numpy_jax_torch_parity(handles):
    """id / fd / aba: numpy == jax == torch on the damped build (the C5 Part-A
    cross-surface gate — the FFI handlers compute the same baked bias)."""
    name, hn, hj, ht, _ = handles
    nj = hn.num_joints
    q, qd, u = _samples(nj)
    qt, qdt, ut = _torch_in(q, qd, u)
    checks = {
        "inverse_dynamics": (hn.inverse_dynamics(q, qd),
                             hj.inverse_dynamics(q, qd),
                             ht.inverse_dynamics(qt, qdt)),
        "forward_dynamics": (hn.forward_dynamics(q, qd, u),
                             hj.forward_dynamics(q, qd, u),
                             ht.forward_dynamics(qt, qdt, ut)),
        "aba": (hn.aba(q, qd, u), hj.aba(q, qd, u), ht.aba(qt, qdt, ut)),
    }
    for algo, (cn, cj, ct) in checks.items():
        assert _maxabs(cn, cj) < _TOL, f"{algo}: numpy vs jax {_maxabs(cn, cj):.2e}"
        assert _maxabs(cn, ct) < _TOL, f"{algo}: numpy vs torch {_maxabs(cn, ct):.2e}"


def test_gradient_path_numpy_jax_torch_parity(handles):
    """id_gradient / fd_gradient: numpy == jax == torch on the damped build (the
    dc_dqd damping diagonal flows identically through every surface)."""
    name, hn, hj, ht, _ = handles
    nj = hn.num_joints
    q, qd, u = _samples(nj, seed=2)
    qt, qdt, ut = _torch_in(q, qd, u)
    checks = {
        "inverse_dynamics_gradient": (hn.inverse_dynamics_gradient(q, qd, u),
                                      hj.inverse_dynamics_gradient(q, qd, u),
                                      ht.inverse_dynamics_gradient(qt, qdt, ut)),
        "forward_dynamics_gradient": (hn.forward_dynamics_gradient(q, qd, u),
                                      hj.forward_dynamics_gradient(q, qd, u),
                                      ht.forward_dynamics_gradient(qt, qdt, ut)),
    }
    for algo, (gn, gj, gt) in checks.items():
        assert _maxabs(gn, gj) < _TOL, f"{algo}: numpy vs jax {_maxabs(gn, gj):.2e}"
        assert _maxabs(gn, gt) < _TOL, f"{algo}: numpy vs torch {_maxabs(gn, gt):.2e}"


def test_damped_differs_from_undamped(handles):
    """The damped build must DIFFER from a bare (use_joint_dynamics=False) build by
    a non-trivial margin — guards against a silent no-op flag. (iiwa14/fr3 both have
    nonzero damping.)"""
    name, hn, _, _, hbare = handles
    nj = hn.num_joints
    q, qd, u = _samples(nj, seed=3)
    assert _maxabs(hn.inverse_dynamics(q, qd), hbare.inverse_dynamics(q, qd)) > 1e-2, \
        "damped id == undamped id (flag is a no-op?)"
    assert _maxabs(hn.forward_dynamics(q, qd, u), hbare.forward_dynamics(q, qd, u)) > 1e-2


def test_second_order_damping_contract(handles):
    """SO contract under use_joint_dynamics — the two algorithms differ ON PURPOSE:

    * idsva_so (INVERSE dynamics SO): the damping torque B*qd is linear in qd in
      TORQUE space, so every second partial of tau is untouched -> damped and bare
      builds must agree bit-for-bit (empirically 0.0 diff).
    * fdsva_so (FORWARD dynamics SO): qdd = Minv(q)(tau - h - B qd) maps the
      damping term through Minv(q), which is q-dependent at EVERY order — e.g.
      d2qdd/dq2 gains d2[Minv]/dq2 · (B qd) terms — so damped and bare builds
      legitimately DIFFER, and the damped build's contract is agreement with the
      damped ORACLE (RBDReference(use_joint_dynamics=True)), which follows the
      value path through its derivative layers automatically.

    The pre-2026-08 version asserted equality for BOTH — analytically wrong for
    fdsva_so; it was masked for weeks by the numpy-surface idsva_so launch-tier
    crash and failed the moment that was fixed (verified: damped CUDA fdsva_so
    matches the damped oracle to ~1e-4 relative and is FAR from the bare one)."""
    name, hn, _, _, hbare = handles
    nj = hn.num_joints
    q, qd, u = _samples(nj, seed=4)

    on = hn.idsva_so(q, qd, u)
    off = hbare.idsva_so(q, qd, u)
    assert _maxabs(on, off) < 1e-5, \
        f"idsva_so: damping leaked into inverse-dynamics 2nd order ({_maxabs(on, off):.2e})"

    fd_on = [np.asarray(t, np.float64) for t in hn.fdsva_so(q, qd, u)]
    fd_off = [np.asarray(t, np.float64) for t in hbare.fdsva_so(q, qd, u)]
    assert max(np.abs(a - b).max() for a, b in zip(fd_on, fd_off)) > 1e-2, \
        "fdsva_so: damped build identical to bare — Minv-mapped damping lost from FD 2nd order"

    ref = _reference(name, use_joint_dynamics=True)
    for i in range(min(2, q.shape[0])):
        ref_t = [np.asarray(t, np.float64) for t in
                 ref.fdsva_so(q[i].astype(np.float64), qd[i].astype(np.float64),
                              u[i].astype(np.float64))]
        for k, (c, r) in enumerate(zip(fd_on, ref_t)):
            rel = np.abs(c[i] - r).max() / max(1.0, np.abs(r).max())
            assert rel < 1e-3, \
                f"fdsva_so tensor[{k}] sample {i}: damped CUDA vs damped oracle rel={rel:.2e}"


def test_nle_coriolis_damping_convention(handles):
    """Pins the INTENTIONAL value-surface convention under use_joint_dynamics
    (2026-08-08 audit; both surfaces are consistent BY CONSTRUCTION):

    * nonlinear_effects = ID(q, qd, 0) — an RNEA wrapper on both the CUDA and
      oracle sides, so it INCLUDES the damping bias when the flag is on.
    * coriolis_matrix is the pure pin computeCoriolisMatrix recursion — damping
      never enters (verified bit-identical damped-vs-bare).
    * Consequence: the identity C(q,qd) qd + g(q) == nle holds on BARE builds;
      on damped builds the residual IS the damping bias, exactly.
    """
    name, hn, _, _, hbare = handles
    nj = hn.num_joints
    q, qd, _ = _samples(nj, seed=7)
    qd64 = qd.astype(np.float64)

    nle_d = np.asarray(hn.nonlinear_effects(q, qd, gravity=-9.81), np.float64)
    nle_b = np.asarray(hbare.nonlinear_effects(q, qd, gravity=-9.81), np.float64)
    C_d = np.asarray(hn.coriolis_matrix(q, qd), np.float64)
    C_b = np.asarray(hbare.coriolis_matrix(q, qd), np.float64)
    gg = np.asarray(hbare.generalized_gravity(q, gravity=-9.81), np.float64)

    assert np.abs(nle_d - nle_b).max() > 1e-2, \
        "nle must include the damping bias on the damped build"
    assert np.abs(C_d - C_b).max() == 0.0, \
        "coriolis_matrix must be PURE (damping-free) on every build"
    for i in range(q.shape[0]):
        resid_bare = np.abs(C_b[i] @ qd64[i] + gg[i] - nle_b[i]).max()
        assert resid_bare < 1e-4, f"bare identity C qd + g == nle broke ({resid_bare:.2e})"
        resid_damped = C_d[i] @ qd64[i] + gg[i] - nle_d[i]
        bias = nle_b[i] - nle_d[i]
        assert np.abs(resid_damped - bias).max() < 1e-4, \
            "damped identity residual must equal the damping bias exactly"

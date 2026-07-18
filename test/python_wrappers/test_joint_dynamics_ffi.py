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


def test_second_order_unchanged_by_damping(handles):
    """SO regression: the bias is LINEAR in qd, so its 2nd derivative is 0 ->
    idsva_so / fdsva_so are byte-for-byte unaffected by use_joint_dynamics. Compares
    the damped build's SO tensors against the bare build's (same q,qd,qdd)."""
    name, hn, _, _, hbare = handles
    nj = hn.num_joints
    q, qd, u = _samples(nj, seed=4)
    for algo in ("idsva_so", "fdsva_so"):
        fn_on = getattr(hn, algo, None)
        fn_off = getattr(hbare, algo, None)
        if fn_on is None or fn_off is None:
            pytest.skip(f"{algo} not on the numpy surface for {name}")
        # idsva_so(q,qd,qdd); fdsva_so(q,qd,qdd,tau) — both take (q,qd,u)-style triples
        on = fn_on(q, qd, u)
        off = fn_off(q, qd, u)
        assert _maxabs(on, off) < 1e-5, f"{algo}: damping leaked into 2nd order ({_maxabs(on, off):.2e})"

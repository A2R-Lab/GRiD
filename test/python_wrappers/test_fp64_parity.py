"""fp64 (double-precision) tight-tolerance parity for the grid_rbd handle.

The fp64 tier (`register_robot(..., dtype="float64")`, -DGRID_WRAPPER_T_DOUBLE)
exists to deliver materially tighter numerics than fp32. This guards that:
  (1) an fp64 handle round-trips float64 and matches the float64 RBDReference
      oracle to ~machine precision (~1e-16, vs fp32's ~1e-7) across the value +
      first-order surfaces — INCLUDING the gravity-using id/fd paths now that the
      fp64 gravity-arg precision cap is fixed, and
  (2) fp64 is ORDERS tighter than fp32 on the same robot/sample — the assertion
      that proves the tier earns its keep (a regression that silently degraded
      fp64 to fp32-quality would otherwise pass unnoticed).

Skips if grid_rbd / nvcc / the URDF fixture are unavailable.

Run with:  pytest test/python_wrappers/test_fp64_parity.py -m python_wrappers -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from config import robot_urdf

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
_URDF = robot_urdf("iiwa14")
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF not present at {_URDF}", allow_module_level=True)
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers

_FP64_TOL = 1e-12   # fp64 vs float64 oracle: value + first-order surfaces on iiwa14


@pytest.fixture(scope="module")
def h64():
    return _grid_rbd.register_robot(
        name="iiwa14_fp64_pytest", urdf_path=str(_URDF),
        floating_base=False, dtype="float64", max_batch_size=8)


@pytest.fixture(scope="module")
def h32():
    return _grid_rbd.register_robot(
        name="iiwa14_fp32_pytest", urdf_path=str(_URDF),
        floating_base=False, dtype="float32", max_batch_size=8)


@pytest.fixture(scope="module")
def ref():
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    return RBDReference(URDFParser().parse(str(_URDF), floating_base=False))


@pytest.fixture(scope="module")
def samples64(h64):
    rng = np.random.default_rng(0)
    NJ = h64.num_joints
    B = 4
    return {
        "q":  rng.standard_normal((B, NJ)).astype(np.float64),
        "qd": rng.standard_normal((B, NJ)).astype(np.float64),
        "u":  rng.standard_normal((B, NJ)).astype(np.float64),
    }


def _rel(a, b):
    # relative max-error (magnitude-scaled) — the right metric for outputs whose
    # magnitude varies a lot (fd torques reach ~1e3, so a fixed ABS tol is wrong).
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))


# Per-algo fp64 relative tolerances, set from MEASURED iiwa14 errors (not guessed).
# With the fp64 gravity-arg fix (the C-ABI / pybind11 `gravity` param is now the
# robot scalar `T`/`CT`, not float32), ALL of id/fd/crba/minv collapse to ~machine
# precision: measured rel errors id 4.1e-16, fd 8.3e-17, crba 9.1e-17, minv 1.4e-17.
# Tolerances are set ~1e4x above the measured floor (still ~1e8x tighter than fp32
# ~1e-7) so they're robust to RNG/driver jitter while the contrast test below proves
# the orders-of-magnitude gap.
_FP64_REL = {"inverse_dynamics": 1e-12, "crba": 1e-12, "minv": 1e-12, "forward_dynamics": 1e-12}


def test_dtype_roundtrip(h64, samples64):
    assert h64.dtype == "float64"
    out = h64.inverse_dynamics(samples64["q"], samples64["qd"])
    assert np.asarray(out).dtype == np.float64


def test_fp64_inverse_dynamics_tight(h64, ref, samples64):
    grid = h64.inverse_dynamics(samples64["q"], samples64["qd"])
    for i, (q, qd) in enumerate(zip(samples64["q"], samples64["qd"])):
        c_ref, *_ = ref.inverse_dynamics(q, qd, GRAVITY=-9.81)
        assert _rel(grid[i], c_ref) < _FP64_REL["inverse_dynamics"], f"id[{i}]: {_rel(grid[i], c_ref):.2e}"


def test_fp64_crba_tight(h64, ref, samples64):
    grid = h64.crba(samples64["q"])
    for i, q in enumerate(samples64["q"]):
        assert _rel(grid[i], ref.crba(q)) < _FP64_REL["crba"], f"crba[{i}]: {_rel(grid[i], ref.crba(q)):.2e}"


def test_fp64_minv_tight(h64, ref, samples64):
    grid = h64.minv(samples64["q"])
    for i, q in enumerate(samples64["q"]):
        assert _rel(grid[i], ref.minv(q)) < _FP64_REL["minv"], f"minv[{i}]: {_rel(grid[i], ref.minv(q)):.2e}"


def test_fp64_forward_dynamics_tight(h64, ref, samples64):
    grid = h64.forward_dynamics(samples64["q"], samples64["qd"], samples64["u"])
    for i, (q, qd, u) in enumerate(zip(samples64["q"], samples64["qd"], samples64["u"])):
        assert _rel(grid[i], ref.forward_dynamics(q, qd, u)) < _FP64_REL["forward_dynamics"], \
            f"fd[{i}]: {_rel(grid[i], ref.forward_dynamics(q, qd, u)):.2e}"


def test_fp64_materially_tighter_than_fp32(h64, h32, ref, samples64):
    """The load-bearing assertion: on the same robot/sample, fp64's relative error
    vs the float64 oracle is >=100x below fp32's, across crba AND the gravity-using
    id/fd paths. The fp64 gravity-arg precision cap is FIXED: the C-ABI / pybind11
    `gravity` param is now the robot scalar type (CT == double for an fp64 .so), so a
    Python float -9.81 reaches the kernel in full double precision instead of being
    rounded to fp32 (-9.81000041...). id/fd consequently collapse to true fp64
    (~1e-16 rel) and now clear the same >=100x bar that crba (the no-gravity probe)
    always did — measured ratios are ~1e8x. A regression that silently re-introduced
    the fp32-gravity cap (or any fp32 degradation) would fail this test."""
    q64 = samples64["q"]
    q32 = q64.astype(np.float32)
    qd64, u64 = samples64["qd"], samples64["u"]
    qd32, u32 = qd64.astype(np.float32), u64.astype(np.float32)

    def contrast(name, out64, out32, oracle):
        e64 = max(_rel(out64[i], oracle(i)) for i in range(len(q64)))
        e32 = max(_rel(out32[i], oracle(i)) for i in range(len(q64)))
        assert e64 < 1e-9, f"fp64 {name} not tight: {e64:.2e}"
        assert e64 < e32 * 1e-2, \
            f"{name}: fp64 {e64:.2e} not >=100x tighter than fp32 {e32:.2e}"

    # crba — the no-gravity composite-inertia probe (always reached true fp64).
    contrast("crba", h64.crba(q64), h32.crba(q32), lambda i: ref.crba(q64[i]))
    # inverse_dynamics — exercises the gravity arg; the path the fix targets.
    contrast("inverse_dynamics",
             h64.inverse_dynamics(q64, qd64), h32.inverse_dynamics(q32, qd32),
             lambda i: ref.inverse_dynamics(q64[i], qd64[i], GRAVITY=-9.81)[0])
    # forward_dynamics — also gravity-dependent.
    contrast("forward_dynamics",
             h64.forward_dynamics(q64, qd64, u64), h32.forward_dynamics(q32, qd32, u32),
             lambda i: ref.forward_dynamics(q64[i], qd64[i], u64[i]))


# ─── Wave 2a: fp64 on the torch / jax surfaces ──────────────────────────────
# The fp64 .so now carries fp64 jax (GRID_FFI_T=F64, .Attr<T> gravity/dt) and
# torch (GRID_TORCH_DTYPE=kFloat64, data_ptr<T>) surfaces. Gate BOTH the same
# way the numpy tier is gated: tight vs the float64 oracle AND materially
# tighter than fp32 — the jax contrast test specifically catches the
# fp32-attr gravity rounding cap (_core.cpp:49-57 class of bug).


@pytest.fixture(scope="module")
def h64_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch reports no CUDA device")
    import grid_rbd
    return grid_rbd.register_robot(
        name="iiwa14_fp64_torch_pytest", urdf_path=str(_URDF),
        floating_base=False, dtype="float64", max_batch_size=8, backend="torch")


@pytest.fixture(scope="module")
def h64_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import grid_rbd
    return grid_rbd.register_robot(
        name="iiwa14_fp64_jax_pytest", urdf_path=str(_URDF),
        floating_base=False, dtype="float64", max_batch_size=8, backend="jax")


def test_fp64_torch_dtype_and_tight(h64_torch, ref, samples64):
    import torch
    q64, qd64, u64 = samples64["q"], samples64["qd"], samples64["u"]
    assert h64_torch.dtype == "float64"
    q = torch.tensor(q64, dtype=torch.float64, device="cuda")
    qd = torch.tensor(qd64, dtype=torch.float64, device="cuda")
    u = torch.tensor(u64, dtype=torch.float64, device="cuda")
    out = h64_torch.inverse_dynamics(q, qd, gravity=-9.81)
    assert out.dtype == torch.float64
    got = out.cpu().numpy()
    for i in range(len(q64)):
        c_ref, *_ = ref.inverse_dynamics(q64[i], qd64[i], GRAVITY=-9.81)
        assert _rel(got[i], c_ref) < _FP64_REL["inverse_dynamics"], \
            f"torch fp64 id[{i}]: rel {_rel(got[i], c_ref):.2e}"
    # fp32 tensors must be REJECTED by the op dtype check (no silent downcast).
    with pytest.raises(RuntimeError, match="float64"):
        h64_torch.inverse_dynamics(q.float(), qd.float(), gravity=-9.81)
    fd = h64_torch.forward_dynamics(q, qd, u, gravity=-9.81)
    assert fd.dtype == torch.float64
    got_fd = fd.cpu().numpy()
    for i in range(len(q64)):
        exp = np.asarray(ref.forward_dynamics(q64[i], qd64[i], u64[i]),
                         dtype=np.float64).reshape(-1)
        assert _rel(got_fd[i], exp) < _FP64_REL["forward_dynamics"], \
            f"torch fp64 fd[{i}]: rel {_rel(got_fd[i], exp):.2e}"


def test_fp64_jax_dtype_and_tight(h64_jax, ref, samples64):
    import jax.numpy as jnp
    q64, qd64, u64 = samples64["q"], samples64["qd"], samples64["u"]
    assert h64_jax.dtype == "float64"
    q, qd, u = jnp.asarray(q64), jnp.asarray(qd64), jnp.asarray(u64)  # x64 on -> float64
    out = np.asarray(h64_jax.inverse_dynamics(q, qd, gravity=-9.81))
    assert out.dtype == np.float64
    for i in range(len(q64)):
        c_ref, *_ = ref.inverse_dynamics(q64[i], qd64[i], GRAVITY=-9.81)
        # This is the assertion that catches an fp32 gravity attr (caps at ~4e-8).
        assert _rel(out[i], c_ref) < _FP64_REL["inverse_dynamics"], \
            f"jax fp64 id[{i}]: rel {_rel(out[i], c_ref):.2e}"
    fd = np.asarray(h64_jax.forward_dynamics(q, qd, u, gravity=-9.81))
    assert fd.dtype == np.float64
    for i in range(len(q64)):
        exp = np.asarray(ref.forward_dynamics(q64[i], qd64[i], u64[i]),
                         dtype=np.float64).reshape(-1)
        assert _rel(fd[i], exp) < _FP64_REL["forward_dynamics"], \
            f"jax fp64 fd[{i}]: rel {_rel(fd[i], exp):.2e}"

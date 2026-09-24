"""W03 malformed-input contract, driven through all three surfaces (2026-09-24).

Every row of the operand-validation table (docs: concepts/operand_validation)
is exercised with a malformed input on numpy, torch and JAX — including the
NATIVE JAX handler with the Python pre-checks bypassed, which is the boundary a
traced program or a direct handler call reaches. Subset artifacts (shared with
test_runtime_contexts.py, same names → cache hits).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
grid_rbd = pytest.importorskip("grid_rbd")

pytestmark = pytest.mark.python_wrappers
MAX_BATCH = 16


def _register(name, urdf, floating, algos, runtime_inertia=False):
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not on PATH")
    return grid_rbd.register_robot(name, str(_REPO / "config/robot_assets" / urdf), floating_base=floating,
                                   max_batch_size=MAX_BATCH, algorithm_list=algos, enable_mujoco_kernels=False,
                                   runtime_inertia=runtime_inertia)


@pytest.fixture(scope="module")
def iiwa():
    h = _register("ctx_pytest_iiwa14", "iiwa14.urdf", False,
                  ["forward_dynamics", "inverse_dynamics", "forward_dynamics_gradient", "minv"], runtime_inertia=True)
    yield h
    h.close()


@pytest.fixture(scope="module")
def go2():
    h = _register("ctx_pytest_go2", "go2.urdf", True, ["forward_dynamics", "inverse_dynamics"])
    yield h
    h.close()


def _cache_key(handle):
    return grid_rbd.manifest_lookup(grid_rbd.default_cache_dir(), handle._name)["cache_key"]


def _state(h, B=4):
    rng = np.random.default_rng(0)
    q = 0.3 * rng.standard_normal((B, h.nq)).astype(np.float32)
    if h.floating_base:
        quat = rng.standard_normal((B, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True); q[:, 3:7] = quat
    qd = 0.3 * rng.standard_normal((B, h.nq)).astype(np.float32); u = 0.3 * rng.standard_normal((B, h.nq)).astype(np.float32)
    if h.floating_base:
        qd[:, h.nv:] = 0; u[:, h.nv:] = 0
    return q, qd, u


# ─── numpy ───────────────────────────────────────────────────────────────────

def test_numpy_rejects_zero_batch_and_shape_faults(iiwa):
    q, qd, u = _state(iiwa)
    with pytest.raises((ValueError, RuntimeError), match="batch must be >= 1"):
        iiwa.forward_dynamics(q[:0], qd[:0], u[:0])
    with pytest.raises((ValueError, RuntimeError), match="batch"):   # pybind: matching batch dim
        iiwa.forward_dynamics(q, qd[:3], u)
    with pytest.raises((ValueError, RuntimeError), match="must be"):
        iiwa.forward_dynamics(q, np.zeros((4, iiwa.nq + 1), np.float32), u)
    # a per-sample 1D input is a documented convenience (batch of one, 1D result)
    assert iiwa.forward_dynamics(q[0], qd[0], u[0]).shape == (iiwa.nq,)
    big = np.zeros((MAX_BATCH + 1, iiwa.nq), np.float32)
    with pytest.raises((ValueError, RuntimeError), match="max_batch"):
        iiwa.forward_dynamics(big, big, big)


def test_numpy_coerces_layout_and_dtype(iiwa):
    q, qd, u = _state(iiwa)
    ref = iiwa.forward_dynamics(q, qd, u)
    # a non-contiguous view and a float64 array are copied/cast, not refused
    qT = np.asfortranarray(q)
    assert not qT.flags["C_CONTIGUOUS"]
    assert np.allclose(iiwa.forward_dynamics(qT, qd.astype(np.float64), u), ref, atol=1e-6)


def test_numpy_refuses_a_broadcast_force_and_the_nv_footgun(iiwa, go2):
    q, qd, u = _state(iiwa)
    fe1 = np.zeros((1, 6 * iiwa.num_bodies), np.float32)
    with pytest.raises((ValueError, RuntimeError)):
        iiwa.forward_dynamics(q, qd, u, f_ext=fe1)
    q, qd, u = _state(go2)
    with pytest.raises(ValueError, match="nv=|num_vel"):
        go2.forward_dynamics(q, qd[:, :go2.nv], u)


# ─── torch ───────────────────────────────────────────────────────────────────

def test_torch_native_checks(iiwa):
    torch = pytest.importorskip("torch")
    import grid_rbd.torch as gt
    tv = gt.TorchRobotHandle(iiwa, _cache_key(iiwa), iiwa._so_path)
    q, qd, u = (torch.as_tensor(x, device="cuda") for x in _state(iiwa))
    with pytest.raises(RuntimeError, match="batch must be >= 1"):
        tv.forward_dynamics(q[:0], qd[:0], u[:0])
    with pytest.raises(RuntimeError, match="must equal the leading operand's batch"):
        tv.forward_dynamics(q, qd[:3], u)
    with pytest.raises(RuntimeError, match="must be a CUDA tensor"):
        tv.forward_dynamics(q.cpu(), qd, u)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        tv.forward_dynamics(q.t().contiguous().t(), qd, u)
    with pytest.raises(RuntimeError, match="must be float32"):
        tv.forward_dynamics(q.double(), qd, u)
    with pytest.raises(RuntimeError, match="last dim"):
        tv.forward_dynamics(torch.zeros(4, iiwa.nq + 1, device="cuda"), qd, u)
    big = torch.zeros(MAX_BATCH + 1, iiwa.nq, device="cuda")
    with pytest.raises(RuntimeError, match="max_batch"):
        tv.forward_dynamics(big, big, big)
    with pytest.raises(RuntimeError, match="f_ext: batch 1 must equal"):
        tv.forward_dynamics(q, qd, u, f_ext=torch.zeros(1, 6 * iiwa.num_bodies, device="cuda"))


# ─── JAX: Python pre-checks, then the NATIVE handler with them bypassed ───────

def test_jax_python_checks(iiwa, go2):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp, grid_rbd.jax as gj
    jv = gj.JaxRobotHandle(iiwa, _cache_key(iiwa), iiwa._so_path)
    q, qd, u = (jnp.asarray(x) for x in _state(iiwa))
    with pytest.raises(ValueError, match="batch"):
        jv.forward_dynamics(q, qd[:3], u)
    with pytest.raises(ValueError, match="must be"):
        jv.forward_dynamics(q, jnp.zeros((4, iiwa.nq + 1)), u)
    with pytest.raises(ValueError, match="max_batch"):
        big = jnp.zeros((MAX_BATCH + 1, iiwa.nq)); jv.forward_dynamics(big, big, big)
    gv = gj.JaxRobotHandle(go2, _cache_key(go2), go2._so_path)
    q, qd, u = (jnp.asarray(x) for x in _state(go2))
    with pytest.raises(ValueError, match="FLOATING-base"):
        gv.forward_dynamics(q, qd[:, :go2.nv], u)
    # an empty batch: XLA elides a zero-sized custom call, so the native check
    # is never reached and the result is simply empty (documented behaviour)
    z = jnp.zeros((0, iiwa.nq))
    assert jax.block_until_ready(jv.forward_dynamics(z, z, z)).shape == (0, iiwa.nq)


def test_jax_native_handler_checks_every_operand(iiwa):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp, grid_rbd.jax as gj
    key = _cache_key(iiwa)
    target = gj._register_method_target(iiwa._so_path, key, "forward_dynamics", "grid_rbd_jax_forward_dynamics")
    nj, nb = iiwa.nq, iiwa.num_bodies
    q, qd, u = (jnp.asarray(x) for x in _state(iiwa))
    fe = jnp.zeros((4, 6 * nb), jnp.float32)
    attrs = dict(gravity=np.float32(-9.81), ctx_id=np.int64(iiwa.ctx_id))

    def call(*ops, B=4):
        out = jax.ShapeDtypeStruct((B, nj), jnp.float32)
        return jax.block_until_ready(jax.ffi.ffi_call(target, out, vmap_method="broadcast_all")(*ops, **attrs))

    call(q, qd, u, fe)                                                    # well-formed
    with pytest.raises(Exception, match="qd batch must equal the leading operand's batch"):
        call(q, qd[:3], u, fe)
    with pytest.raises(Exception, match="u: must be 2D"):
        call(q, qd, u[0], fe)
    with pytest.raises(Exception, match="f_ext batch must equal the q batch"):
        call(q, qd, u, fe[:1])
    assert call(q[:0], qd[:0], u[:0], fe[:0], B=0).shape == (0, nj)   # XLA elides the zero-sized call
    with pytest.raises(Exception, match="batch > max_batch"):
        z = jnp.zeros((MAX_BATCH + 1, nj), jnp.float32)
        call(z, z, z, jnp.zeros((MAX_BATCH + 1, 6 * nb), jnp.float32), B=MAX_BATCH + 1)

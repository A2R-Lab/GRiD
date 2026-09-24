"""codex R1 acceptance (2026-09-24): a FLOATING-base artifact built WITH the MuJoCo
kernels loads and runs its `_mujoco` twins on numpy, torch and JAX — value ops,
forward/backward through the mjx-convention VJPs, and on an explicit context.

The fixed-base smokes #ifdef the twins out, which is how 30 twins once lacked
their leading `ctx_id` and still passed every green gate. Subset artifact so the
module stays minutes.
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
ALGOS = ["forward_dynamics", "inverse_dynamics", "forward_dynamics_gradient", "minv"]


@pytest.fixture(scope="module")
def go2():
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not on PATH")
    h = grid_rbd.register_robot("ctx_pytest_go2_mjx", str(_REPO / "config/robot_assets/go2.urdf"),
                                floating_base=True, max_batch_size=16, algorithm_list=ALGOS,
                                enable_mujoco_kernels=True)
    yield h
    h.close()


def _cache_key(handle):
    return grid_rbd.manifest_lookup(grid_rbd.default_cache_dir(), handle._name)["cache_key"]


def _state(h, B=4, seed=0):
    rng = np.random.default_rng(seed)
    q = 0.3 * rng.standard_normal((B, h.nq)).astype(np.float32)
    quat = rng.standard_normal((B, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True); q[:, 3:7] = quat
    qd = 0.3 * rng.standard_normal((B, h.nq)).astype(np.float32); u = 0.3 * rng.standard_normal((B, h.nq)).astype(np.float32)
    qd[:, h.nv:] = 0; u[:, h.nv:] = 0
    return q, qd, u


def test_numpy_mujoco_twins_run_and_differ_from_pinocchio(go2):
    q, qd, u = _state(go2)
    pin = go2.forward_dynamics(q, qd, u)
    mjx = go2.mujoco.forward_dynamics(q, qd, u)
    assert mjx.shape == pin.shape and np.isfinite(mjx).all()
    assert not np.allclose(mjx, pin, atol=1e-5), "mjx twin returned the pinocchio-convention value"
    assert np.isfinite(go2.mujoco.inverse_dynamics(q, qd, np.zeros_like(q))).all()
    assert np.isfinite(go2.mujoco.minv(q)).all()


def test_numpy_mujoco_twins_on_an_explicit_context(go2):
    q, qd, u = _state(go2)
    ref = go2.mujoco.forward_dynamics(q, qd, u)
    ctx = go2.context()
    try:
        out = ctx.mujoco.forward_dynamics(q, qd, u)
        assert ctx.ctx_id != go2.ctx_id and np.allclose(out, ref, atol=1e-5)
    finally:
        ctx.close()


def test_torch_mujoco_forward_and_backward(go2):
    torch = pytest.importorskip("torch")
    import grid_rbd.torch as gt
    tv = gt.TorchRobotHandle(go2, _cache_key(go2), go2._so_path)
    q, qd, u = _state(go2)
    tq = torch.as_tensor(q, device="cuda").requires_grad_(True)
    tqd, tu = (torch.as_tensor(x, device="cuda") for x in (qd, u))
    out = tv.mujoco.forward_dynamics(tq, tqd, tu)
    assert np.allclose(out.detach().cpu().numpy(), go2.mujoco.forward_dynamics(q, qd, u), atol=1e-5)
    out.sum().backward()
    assert tq.grad is not None and torch.isfinite(tq.grad).all()


def test_jax_mujoco_forward_and_grad(go2):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp, grid_rbd.jax as gj
    jv = gj.JaxRobotHandle(go2, _cache_key(go2), go2._so_path)
    q, qd, u = _state(go2)
    jq, jqd, ju = (jnp.asarray(x) for x in (q, qd, u))
    out = np.asarray(jax.block_until_ready(jv.mujoco.forward_dynamics(jq, jqd, ju)))
    assert np.allclose(out, go2.mujoco.forward_dynamics(q, qd, u), atol=1e-5)
    g = jax.block_until_ready(jax.jit(jax.grad(lambda a: jv.mujoco.forward_dynamics(a, jqd, ju).sum()))(jq))
    assert np.isfinite(np.asarray(g)).all()

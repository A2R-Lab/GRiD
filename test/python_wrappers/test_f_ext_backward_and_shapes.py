"""External-force contracts on the autodiff surfaces (audit W02 + W03, 2026-09-19).

W02: the JAX custom VJPs of inverse/forward dynamics dropped ``f_ext`` from
their residuals and the gradient FFI handlers took no force buffer, so a q/qd
gradient under a nonzero body-local wrench was the ZERO-force gradient. A fixed
body-local wrench still has q-dependent joint torques (its world direction
rotates with the links), so the gradient must be taken AT the force. Oracle:
central differences of the public function (fp64 build), and torch (which
always threaded ``ctx.f_ext``) as the cross-backend referee.

W03: the FFI copies ``batch*6*NUM_BODIES`` elements sized by the STATE batch. A
(1, 6nb) force for a batch of 8 used to pass the Python check and read past its
buffer. Now: broadcast forms are MATERIALIZED on the jax surface, wrong widths
are rejected, and the native torch/jax boundaries reject a batch mismatch.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import robot_urdf  # noqa: E402

grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH", allow_module_level=True)
_IIWA = robot_urdf("iiwa14")
if not _IIWA.exists():
    pytest.skip(f"iiwa14 URDF not present at {_IIWA}", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers
_ALGOS = ["inverse_dynamics", "inverse_dynamics_gradient", "forward_dynamics",
          "forward_dynamics_gradient", "minv"]
_EPS = 1e-6


def _sample(nq, nb, seed=20260919, batch=1):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1.0, 1.0, (batch, nq))
    qd = rng.uniform(-1.0, 1.0, (batch, nq))
    u = rng.uniform(-1.0, 1.0, (batch, nq))
    f_ext = rng.uniform(-5.0, 5.0, (batch, 6 * nb))       # body-local wrenches on EVERY body
    return q, qd, u, f_ext


def _central_fd(f, x, eps=_EPS):
    g = np.zeros_like(x)
    for i in range(x.size):
        d = np.zeros_like(x); d.flat[i] = eps
        g.flat[i] = (f(x + d) - f(x - d)) / (2 * eps)
    return g


@pytest.fixture(scope="module")
def jax_iiwa():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    h = grid_rbd.register_robot("w02_iiwa14_jax_fp64", str(_IIWA), backend="jax",
                                algorithm_list=_ALGOS, dtype="float64")
    yield h
    h.close()


@pytest.fixture(scope="module")
def torch_iiwa():
    pytest.importorskip("torch")
    h = grid_rbd.register_robot("w02_iiwa14_torch_fp64", str(_IIWA), backend="torch",
                                algorithm_list=_ALGOS, dtype="float64")
    yield h
    h.close()


@pytest.mark.parametrize("method", ["inverse_dynamics", "forward_dynamics"])
def test_jax_gradient_is_taken_at_the_external_force(jax_iiwa, method):
    import jax, jax.numpy as jnp
    h = jax_iiwa
    q, qd, u, f_ext = _sample(h.num_joints, h.num_bodies)
    w = np.linspace(0.5, 1.5, h.num_joints)

    def loss(qq, qqd, fe):
        if method == "inverse_dynamics":
            out = h.inverse_dynamics(qq, qqd, f_ext=fe)
        else:
            out = h.forward_dynamics(qq, qqd, u, f_ext=fe)
        return jnp.sum(jnp.asarray(w) * out)

    for name, argnum, x in (("q", 0, q), ("qd", 1, qd)):
        g = np.asarray(jax.grad(loss, argnums=argnum)(q, qd, f_ext))
        args = [q, qd]
        def f_of(xx, _i=argnum):
            a = list(args); a[_i] = xx
            return float(loss(a[0], a[1], f_ext))
        fd = _central_fd(f_of, x)
        scale = max(1.0, np.abs(fd).max())
        np.testing.assert_allclose(g, fd, rtol=2e-5, atol=2e-6 * scale,
                                   err_msg=f"{method}: d/d{name} at nonzero f_ext vs central differences")
    # the force-conditioned q-gradient must DIFFER from the zero-force one (the old behaviour)
    g_force = np.asarray(jax.grad(loss, argnums=0)(q, qd, f_ext))
    g_zero = np.asarray(jax.grad(loss, argnums=0)(q, qd, np.zeros_like(f_ext)))
    assert np.abs(g_force - g_zero).max() > 1e-3 * max(1.0, np.abs(g_force).max())


def test_jax_jit_does_not_capture_the_force_as_a_constant(jax_iiwa):
    import jax, jax.numpy as jnp
    h = jax_iiwa
    q, qd, u, f1 = _sample(h.num_joints, h.num_bodies, seed=1)
    _, _, _, f2 = _sample(h.num_joints, h.num_bodies, seed=2)
    w = np.linspace(0.5, 1.5, h.num_joints)
    grad_fn = jax.jit(jax.grad(lambda qq, fe: jnp.sum(jnp.asarray(w) * h.inverse_dynamics(qq, qd, f_ext=fe))))
    g1, g2 = np.asarray(grad_fn(q, f1)), np.asarray(grad_fn(q, f2))
    e1 = np.asarray(jax.grad(lambda qq: jnp.sum(jnp.asarray(w) * h.inverse_dynamics(qq, qd, f_ext=f1)))(q))
    e2 = np.asarray(jax.grad(lambda qq: jnp.sum(jnp.asarray(w) * h.inverse_dynamics(qq, qd, f_ext=f2)))(q))
    np.testing.assert_allclose(g1, e1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(g2, e2, rtol=1e-10, atol=1e-12)
    assert np.abs(g1 - g2).max() > 1e-6


def test_torch_and_jax_force_conditioned_gradients_agree(jax_iiwa, torch_iiwa):
    import jax, jax.numpy as jnp, torch
    q, qd, u, f_ext = _sample(jax_iiwa.num_joints, jax_iiwa.num_bodies, seed=3)
    w = np.linspace(0.5, 1.5, jax_iiwa.num_joints)
    gj = np.asarray(jax.grad(lambda qq: jnp.sum(jnp.asarray(w) * jax_iiwa.inverse_dynamics(qq, qd, f_ext=f_ext)))(q))
    T = lambda a: torch.as_tensor(a, dtype=torch.float64, device="cuda")
    qt = T(q).requires_grad_(True)
    (T(w) * torch_iiwa.inverse_dynamics(qt, T(qd), f_ext=T(f_ext))).sum().backward()
    np.testing.assert_allclose(gj, qt.grad.cpu().numpy(), rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------- W03 shapes
def test_jax_f_ext_broadcast_is_materialized_and_widths_rejected(jax_iiwa):
    h = jax_iiwa
    q, qd, u, f_ext = _sample(h.num_joints, h.num_bodies, seed=4, batch=8)
    one = f_ext[:1]                                                   # (1, 6nb) for a batch of 8
    per_row = np.asarray(h.inverse_dynamics(q, qd, f_ext=np.repeat(one, 8, axis=0)))
    np.testing.assert_allclose(np.asarray(h.inverse_dynamics(q, qd, f_ext=one)), per_row, rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(h.inverse_dynamics(q, qd, f_ext=one[0])), per_row, rtol=0, atol=0)
    with pytest.raises(ValueError, match="last dim"):
        h.inverse_dynamics(q, qd, f_ext=f_ext[:, :-6])
    with pytest.raises(ValueError, match="leading dims"):
        h.inverse_dynamics(q, qd, f_ext=f_ext[:3])                     # 3 rows neither match 8 nor broadcast


def test_torch_native_rejects_f_ext_batch_mismatch(torch_iiwa):
    import torch
    h = torch_iiwa
    q, qd, u, f_ext = _sample(h.num_joints, h.num_bodies, seed=5, batch=8)
    T = lambda a: torch.as_tensor(a, dtype=torch.float64, device="cuda")
    with pytest.raises(RuntimeError, match="batch"):
        h.inverse_dynamics(T(q), T(qd), f_ext=T(f_ext[:1]))
    with pytest.raises(RuntimeError, match="last dim"):
        h.inverse_dynamics(T(q), T(qd), f_ext=T(f_ext[:, :-6]))
    ok = h.inverse_dynamics(T(q), T(qd), f_ext=T(f_ext))
    assert tuple(ok.shape) == (8, h.num_joints)

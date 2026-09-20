"""Floating-base CONFIGURATION cotangents through the public jax/torch autodiff
(audit W01, 2026-09-19).

The analytic gradient kernels differentiate in the Pinocchio free-flyer tangent
chart ``[v_lin LOCAL, ω LOCAL, joints]`` (nv-wide). The public ``q`` is
``[pos(3), quat_xyzw(4), joints]`` (nq = nv+1). Before this fix the shared VJP
driver tail-padded the tangent cotangent into the q slot: joints landed one
slot early, the LAST joint's gradient was silently zero, and the quaternion
slots carried raw ω cotangents. The kernels evaluate R(p/|p|), so the public
function is ``f ∘ normalize`` on the ambient quaternion and its exact
derivative is what central differences over EVERY ambient q component measure
— that is the oracle here (fp64 build, both a unit and a NON-unit quaternion).
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
_GO2 = robot_urdf("go2")
if not _GO2.exists():
    pytest.skip(f"go2 URDF not present at {_GO2}", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers

_ALGOS = ["inverse_dynamics", "inverse_dynamics_gradient", "forward_dynamics",
          "forward_dynamics_gradient", "minv", "end_effector_pose", "end_effector_pose_gradient"]
_EPS = 1e-6


def _sample(nq, nv, unit_quat: bool, seed=20260919):
    rng = np.random.default_rng(seed)
    q = np.zeros(nq)
    q[:3] = rng.uniform(-1.0, 1.0, 3)
    quat = rng.standard_normal(4)
    quat /= np.linalg.norm(quat)
    q[3:7] = quat if unit_quat else 1.3 * quat        # non-unit exercises the normalizing extension
    q[7:] = rng.uniform(-1.0, 1.0, nq - 7)
    qd = np.zeros(nq); qd[:nv] = rng.uniform(-1.0, 1.0, nv)
    u = np.zeros(nq); u[:nv] = rng.uniform(-1.0, 1.0, nv)
    return q, qd, u


def _central_fd(f, x, eps=_EPS):
    g = np.zeros_like(x)
    for i in range(x.size):
        d = np.zeros_like(x); d[i] = eps
        g[i] = (f(x + d) - f(x - d)) / (2 * eps)
    return g


# ---------------------------------------------------------------- jax
@pytest.fixture(scope="module")
def jax_go2():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    h = grid_rbd.register_robot("w01_go2_jax_fp64", str(_GO2), floating_base=True, backend="jax",
                                algorithm_list=_ALGOS, enable_mujoco_kernels=False,
                                dtype="float64")
    yield h
    h.close()


def _jax_cases(h):
    import jax.numpy as jnp
    nv = h.num_vel
    w_tau = np.linspace(0.5, 1.5, h.num_joints)
    w_pose = np.linspace(-1.0, 1.0, 6 * h.num_ees)
    return {
        "inverse_dynamics": lambda q, qd, u: jnp.sum(jnp.asarray(w_tau) * h.inverse_dynamics(q[None], qd[None])[0]),
        "forward_dynamics": lambda q, qd, u: jnp.sum(jnp.asarray(w_tau) * h.forward_dynamics(q[None], qd[None], u[None])[0]),
        "end_effector_pose": lambda q, qd, u: jnp.sum(jnp.asarray(w_pose) * h.end_effector_pose(q[None]).reshape(-1)),
    }


@pytest.mark.parametrize("unit_quat", [True, False], ids=["unit-quat", "non-unit-quat"])
@pytest.mark.parametrize("method", ["inverse_dynamics", "forward_dynamics", "end_effector_pose"])
def test_jax_q_gradient_matches_ambient_finite_differences(jax_go2, method, unit_quat):
    import jax
    h = jax_go2
    q, qd, u = _sample(h.num_joints, h.num_vel, unit_quat)
    loss = _jax_cases(h)[method]
    g_q = np.asarray(jax.grad(loss, argnums=0)(q, qd, u))
    fd_q = _central_fd(lambda x: float(loss(x, qd, u)), q)
    scale = max(1.0, np.abs(fd_q).max())
    np.testing.assert_allclose(g_q, fd_q, rtol=2e-5, atol=2e-6 * scale,
                               err_msg=f"{method}: d/dq (pos|quat|joints) vs ambient central differences")
    # the historically dropped LAST joint slot and the quaternion slots must be live
    assert abs(g_q[-1]) > 1e-6 and np.abs(g_q[3:7]).max() > 1e-6
    if method != "end_effector_pose":
        g_qd = np.asarray(jax.grad(loss, argnums=1)(q, qd, u))
        fd_qd = _central_fd(lambda x: float(loss(q, x, u)), qd)
        np.testing.assert_allclose(g_qd, fd_qd, rtol=2e-5, atol=2e-6 * max(1.0, np.abs(fd_qd).max()))


# ---------------------------------------------------------------- torch
@pytest.fixture(scope="module")
def torch_go2():
    pytest.importorskip("torch")
    h = grid_rbd.register_robot("w01_go2_torch_fp64", str(_GO2), floating_base=True, backend="torch",
                                algorithm_list=_ALGOS, enable_mujoco_kernels=False,
                                dtype="float64")
    yield h
    h.close()


@pytest.mark.parametrize("unit_quat", [True, False], ids=["unit-quat", "non-unit-quat"])
@pytest.mark.parametrize("method", ["inverse_dynamics", "forward_dynamics", "end_effector_pose"])
def test_torch_q_gradient_matches_ambient_finite_differences(torch_go2, method, unit_quat):
    import torch
    h = torch_go2
    q, qd, u = _sample(h.num_joints, h.num_vel, unit_quat)
    w_tau = torch.linspace(0.5, 1.5, h.num_joints, dtype=torch.float64, device="cuda")
    w_pose = torch.linspace(-1.0, 1.0, 6 * h.num_ees, dtype=torch.float64, device="cuda")
    T = lambda a: torch.as_tensor(a, dtype=torch.float64, device="cuda")

    def loss(qt, qdt, ut):
        if method == "inverse_dynamics":
            return (w_tau * h.inverse_dynamics(qt[None], qdt[None])[0]).sum()
        if method == "forward_dynamics":
            return (w_tau * h.forward_dynamics(qt[None], qdt[None], ut[None])[0]).sum()
        return (w_pose * h.end_effector_pose(qt[None]).reshape(-1)).sum()

    qt = T(q).requires_grad_(True)
    loss(qt, T(qd), T(u)).backward()
    g_q = qt.grad.detach().cpu().numpy()
    fd_q = _central_fd(lambda x: float(loss(T(x), T(qd), T(u)).item()), q)
    scale = max(1.0, np.abs(fd_q).max())
    np.testing.assert_allclose(g_q, fd_q, rtol=2e-5, atol=2e-6 * scale,
                               err_msg=f"{method}: torch d/dq vs ambient central differences")
    assert abs(g_q[-1]) > 1e-6 and np.abs(g_q[3:7]).max() > 1e-6


# ---------------------------------------------------------------- mujoco convention (numpy driver)
# The mjx twins differentiate in the mjx free-joint chart (linear WORLD, angular
# LOCAL) with q = [pos, quat_wxyz, joints], and — unlike the pin kernels — do NOT
# renormalize the quaternion (MuJoCo expects a unit one). The q cotangent GRiD
# returns under output_convention="mujoco" is therefore the on-manifold pullback:
# it agrees with ambient central differences on the TANGENTIAL subspace and has
# zero radial component. Exercised through the shared driver with numpy arrays
# (the twins' fp64 build with jax/torch glue is a 10-minute compile; the
# framework-agnostic transform is the same code path both surfaces call).
@pytest.fixture(scope="module")
def numpy_go2_mjx():
    h = grid_rbd.register_robot("w01_probe_go2_mjx", str(_GO2), floating_base=True,
                                algorithm_list=["inverse_dynamics", "inverse_dynamics_gradient",
                                                "end_effector_pose", "end_effector_pose_gradient"],
                                enable_mujoco_kernels=True, output_convention="mujoco",
                                dtype="float64", allow_fp64=True)
    yield h
    h.close()


@pytest.mark.parametrize("method", ["inverse_dynamics", "end_effector_pose"])
def test_mujoco_convention_q_cotangent_is_the_on_manifold_pullback(numpy_go2_mjx, method):
    from grid_rbd._vjp_common import _configuration_cotangent
    h = numpy_go2_mjx
    nq, nv = h.num_joints, h.num_vel
    rng = np.random.default_rng(5)
    q = np.zeros(nq); q[:3] = rng.uniform(-1, 1, 3)
    quat = rng.standard_normal(4); q[3:7] = quat / np.linalg.norm(quat)          # wxyz, UNIT
    q[7:] = rng.uniform(-1, 1, nq - 7)
    qd = np.zeros(nq); qd[:nv] = rng.uniform(-1, 1, nv); qdd = np.zeros(nq)
    if method == "inverse_dynamics":
        w = np.linspace(0.5, 1.5, nv)
        f = lambda qq: float(w @ np.asarray(h.inverse_dynamics(qq[None], qd[None], qdd[None]), dtype=np.float64)[0, :nv])
        G = np.asarray(h.inverse_dynamics_gradient(q[None], qd[None], qdd[None]), dtype=np.float64)[0][:, :nv]
    else:
        w = np.linspace(-1.0, 1.0, 6 * h.num_ees)
        f = lambda qq: float(w @ np.asarray(h.end_effector_pose(qq[None]), dtype=np.float64).reshape(-1))
        G = np.asarray(h.end_effector_pose_gradient(q[None]), dtype=np.float64)[0].reshape(-1, nv)
    g_tan = (w @ G)                                                               # (nv,) tangent cotangent
    g_q = _configuration_cotangent(g_tan[None], q[None], mjx=True, layout=h.configuration_layout)[0]
    fd = _central_fd(f, q)
    P = np.eye(4) - np.outer(q[3:7], q[3:7])                                      # tangential projector
    scale = max(1.0, np.abs(fd).max())
    np.testing.assert_allclose(g_q[:3], fd[:3], rtol=2e-5, atol=2e-6 * scale)         # world-linear
    np.testing.assert_allclose(g_q[3:7] @ P, fd[3:7] @ P, rtol=2e-5, atol=2e-6 * scale)  # tangential quaternion
    assert abs(g_q[3:7] @ q[3:7]) < 1e-9 * scale                                        # zero radial component
    np.testing.assert_allclose(g_q[7:], fd[7:], rtol=2e-5, atol=2e-6 * scale)          # joints, shifted by one


# These are the generator's existing spherical CUDA fixtures (fixed base).
# Floating+spherical pullbacks are covered by test_configuration_cotangent;
# that combined model currently fails earlier in RNEA code generation's
# single-axis topology helper, independently of the Python pullback.
@pytest.fixture(scope="module", params=[("spherical_arm", False), ("mixed_spherical_arm", False)])
def spherical_handle(request):
    name, floating = request.param
    urdf = REPO_ROOT / "external/URDFParser/tests/fixtures" / f"{name}.urdf"
    h = grid_rbd.register_robot(f"w01_{name}_{floating}_fp64", str(urdf),
                                floating_base=floating, algorithm_list=_ALGOS,
                                enable_mujoco_kernels=False, max_batch_size=8, dtype="float64")
    yield h
    h.close()


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("method", ["inverse_dynamics", "forward_dynamics", "end_effector_pose"])
def test_spherical_public_q_gradient(spherical_handle, backend, method):
    framework = pytest.importorskip(backend)
    if backend == "jax": framework.config.update("jax_enable_x64", True)
    base = spherical_handle
    h = grid_rbd.get_robot(base.name, backend=backend)
    rng = np.random.default_rng(81)
    q = rng.uniform(-.5, .5, base.num_joints)
    for kind, qi, vi, nq, nv in base.configuration_layout:
        if kind != "euclidean":
            start = qi + (3 if kind == "floating" else 0)
            q[start:start+4] *= 1.2 / np.linalg.norm(q[start:start+4])
    qd = np.zeros_like(q); qd[:base.num_vel] = rng.uniform(-.2, .2, base.num_vel)
    u = np.zeros_like(q); u[:base.num_vel] = rng.uniform(-.3, .3, base.num_vel)
    try:
        if backend == "jax":
            loss = _jax_cases(h)[method]
            f = lambda x: loss(x, qd, u)
            actual = np.asarray(framework.jit(framework.grad(f))(q))
            numeric = _central_fd(lambda x: float(f(x)), q)
        else:
            T = lambda x: framework.as_tensor(x, dtype=framework.float64, device="cuda")
            def f(x):
                if method == "end_effector_pose": out = h.end_effector_pose(x[None])
                elif method == "inverse_dynamics": out = h.inverse_dynamics(x[None], T(qd)[None])
                else: out = h.forward_dynamics(x[None], T(qd)[None], T(u)[None])
                w = framework.linspace(.5, 1.5, out.numel(), dtype=out.dtype, device=out.device)
                return (out.reshape(-1)*w).sum()
            qt = T(q).requires_grad_(True)
            f(qt).backward()
            actual = qt.grad.cpu().numpy()
            numeric = _central_fd(lambda x: float(f(T(x)).item()), q)
        assert actual.shape == q.shape
        np.testing.assert_allclose(actual, numeric, rtol=2e-5,
                                   atol=2e-6*max(1., np.abs(numeric).max()))
    finally:
        h.close()


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("method", ["inverse_dynamics", "end_effector_pose"])
def test_mujoco_public_autodiff_with_explicit_normalization(numpy_go2_mjx, backend, method):
    framework = pytest.importorskip(backend)
    if backend == "jax": framework.config.update("jax_enable_x64", True)
    h = grid_rbd.get_robot(numpy_go2_mjx.name, backend=backend, output_convention="mujoco")
    q, qd, _ = _sample(h.num_joints, h.num_vel, False, seed=31)
    try:
        if backend == "jax":
            import jax.numpy as xp
            def loss(x):
                normalized = xp.concatenate((x[:3], x[3:7]/xp.linalg.norm(x[3:7]), x[7:]))
                if method == "inverse_dynamics": out = h.inverse_dynamics(normalized[None], qd[None])
                else: out = h.end_effector_pose(normalized[None])
                return (out.reshape(-1)*xp.linspace(.5, 1.5, out.size)).sum()
            actual = np.asarray(framework.jit(framework.grad(loss))(q))
            numeric = _central_fd(lambda x: float(loss(x)), q)
        else:
            T = lambda x: framework.as_tensor(x, dtype=framework.float64, device="cuda")
            def loss(x):
                normalized = framework.cat((x[:3], x[3:7]/framework.linalg.vector_norm(x[3:7]), x[7:]))
                if method == "inverse_dynamics": out = h.inverse_dynamics(normalized[None], T(qd)[None])
                else: out = h.end_effector_pose(normalized[None])
                w = framework.linspace(.5, 1.5, out.numel(), dtype=out.dtype, device=out.device)
                return (out.reshape(-1)*w).sum()
            qt = T(q).requires_grad_(True)
            loss(qt).backward()
            actual = qt.grad.cpu().numpy()
            numeric = _central_fd(lambda x: float(loss(T(x)).item()), q)
        np.testing.assert_allclose(actual, numeric, rtol=2e-5,
                                   atol=2e-6*max(1., np.abs(numeric).max()))
    finally:
        h.close()

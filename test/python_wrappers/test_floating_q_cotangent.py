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

Both precisions (2026-10-15). The jax, torch and spherical tests run over
``dtype``: the ``float32`` variant is the receipt's (GRiD runs in fp32 on the
GPU), the ``float64`` variant is the fp64 alpha lane (``fp64_alpha``,
``make test-fp64``) and keeps the oracle and tolerances above. A central
difference of an fp32 public function measures float32 rounding rather than
the kernel (docs/agent_debugging_guide.md 7.z31), so the fp32 variants take
their ambient central differences over the float64 ``RBDReference`` evaluated
as ``f(normalize(q))``: the same public layout (``[x, y, z, qx, qy, qz, qw,
joints]``, nv-wide LOCAL velocities, leaf end effectors in ``get_leaf_nodes()``
order) with every quaternion block normalized explicitly, which is the function
the kernels compute. The two mujoco-convention tests have no float64 CPU oracle
(``RBDReference`` speaks the pinocchio convention only), so they stay fp64 and
carry ``fp64_alpha`` without an fp32 sibling.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from grid_assets import robot_urdf  # noqa: E402

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
_NP = {"float32": np.float32, "float64": np.float64}
# Gradient-vs-oracle tolerances as (rtol, atol per unit of the oracle's scale).
# float64: the fp64 build against central differences of its own public function.
# float32: the fp32 build against the float64 RBDReference oracle, at the CUDA
# equivalence harness float class (cuda_harness.CUDA_DEFAULT_TOLERANCE: rtol 2e-4
# with the atol floored at rtol * max|expected|). Never looser than that class.
_GRAD_TOL = {"float64": (2e-5, 2e-6), "float32": (2e-4, 2e-4)}


def _assert_grad_close(actual, expected, dtype, err_msg=""):
    rtol, atol = _GRAD_TOL[dtype]
    scale = max(1.0, float(np.abs(expected).max()))
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol * scale, err_msg=err_msg)


@pytest.fixture(scope="module", params=["float32", pytest.param("float64", marks=pytest.mark.fp64_alpha)])
def dtype(request):
    """The build precision; float32 is the receipt's, float64 the fp64 alpha lane."""
    return request.param


def _sample(nq, nv, unit_quat: bool, seed=20260919):
    rng = np.random.default_rng(seed)
    q = np.zeros(nq)
    q[:3] = rng.uniform(-1.0, 1.0, 3)
    quat = rng.standard_normal(4)
    quat /= np.linalg.norm(quat)
    q[3:7] = quat if unit_quat else 1.3 * quat        # non-unit exercises the normalizing extension
    q[7:] = rng.uniform(-1.0, 1.0, nq - 7)
    qd = rng.uniform(-1.0, 1.0, nv)
    u = rng.uniform(-1.0, 1.0, nv)
    return q, qd, u


def _central_fd(f, x, eps=_EPS):
    g = np.zeros_like(x)
    for i in range(x.size):
        d = np.zeros_like(x); d[i] = eps
        g[i] = (f(x + d) - f(x - d)) / (2 * eps)
    return g


# ---------------------------------------------------------------- float64 oracle (fp32 variants)
def _reference(urdf, floating):
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(urdf), floating_base=floating)
    return RBDReference(robot)


def _normalized_q(q, layout):
    """``normalize`` of the public q: every quaternion block (the floating root's
    ``[3:7]`` slice, a spherical joint's own 4) scaled to unit norm, as the
    kernels do before evaluating R(p/|p|)."""
    q = np.asarray(q, dtype=np.float64).copy()
    for kind, qi, _vi, _nq, _nv in layout:
        if kind == "euclidean":
            continue
        start = qi + (3 if kind == "floating" else 0)
        q[start:start + 4] /= np.linalg.norm(q[start:start + 4])
    return q


def _reference_cases(ref, h):
    """The float64 oracle of ``_jax_cases``: the same scalar losses evaluated by
    RBDReference at ``normalize(q)``, as functions of ambient (q, qd, u).
    ``h`` is any backend handle; the quaternion block layout is read from the
    numpy handle of the same registration (only it exposes
    ``configuration_layout``)."""
    layout = grid_rbd.get_robot(h.name).configuration_layout
    w_tau = np.linspace(0.5, 1.5, h.num_vel)
    w_pose = np.linspace(-1.0, 1.0, 6 * h.num_ees)
    leaves = [ref.robot.get_joint_by_id(jid).get_name() for jid in ref.robot.get_leaf_nodes()]
    assert len(leaves) == h.num_ees, (leaves, h.num_ees)

    def inverse_dynamics(q, qd, u):
        c, *_ = ref.inverse_dynamics(_normalized_q(q, layout), np.asarray(qd, np.float64), GRAVITY=-9.81)
        return float(w_tau @ np.asarray(c, np.float64).reshape(-1))

    def forward_dynamics(q, qd, u):
        qdd = ref.forward_dynamics(_normalized_q(q, layout), np.asarray(qd, np.float64),
                                   np.asarray(u, np.float64))
        return float(w_tau @ np.asarray(qdd, np.float64).reshape(-1))

    def end_effector_pose(q, qd, u):
        qn = _normalized_q(q, layout)
        pose = np.concatenate([np.asarray(ref.end_effector_pose(qn, ee_joint_names=[leaf]),
                                          np.float64).reshape(-1) for leaf in leaves])
        return float(w_pose @ pose)

    return {"inverse_dynamics": inverse_dynamics, "forward_dynamics": forward_dynamics,
            "end_effector_pose": end_effector_pose}


@pytest.fixture(scope="module")
def ref_go2():
    return _reference(_GO2, True)


# ---------------------------------------------------------------- jax
@pytest.fixture(scope="module")
def jax_go2(dtype):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    h = grid_rbd.register_robot(f"w01_go2_jax_{dtype}", str(_GO2), floating_base=True, backend="jax",
                                algorithm_list=_ALGOS, enable_mujoco_kernels=False,
                                dtype=dtype)
    yield h
    h.close()


def _jax_cases(h):
    import jax.numpy as jnp
    nv = h.num_vel
    w_tau = np.linspace(0.5, 1.5, h.num_vel)
    w_pose = np.linspace(-1.0, 1.0, 6 * h.num_ees)
    return {
        "inverse_dynamics": lambda q, qd, u: jnp.sum(jnp.asarray(w_tau) * h.inverse_dynamics(q[None], qd[None])[0]),
        "forward_dynamics": lambda q, qd, u: jnp.sum(jnp.asarray(w_tau) * h.forward_dynamics(q[None], qd[None], u[None])[0]),
        "end_effector_pose": lambda q, qd, u: jnp.sum(jnp.asarray(w_pose) * h.end_effector_pose(q[None]).reshape(-1)),
    }


@pytest.mark.parametrize("unit_quat", [True, False], ids=["unit-quat", "non-unit-quat"])
@pytest.mark.parametrize("method", ["inverse_dynamics", "forward_dynamics", "end_effector_pose"])
def test_jax_q_gradient_matches_ambient_finite_differences(jax_go2, ref_go2, dtype, method, unit_quat):
    import jax
    h = jax_go2
    q, qd, u = (np.asarray(a, dtype=_NP[dtype]) for a in _sample(h.num_joints, h.num_vel, unit_quat))
    q64, qd64, u64 = (np.asarray(a, dtype=np.float64) for a in (q, qd, u))
    loss = _jax_cases(h)[method]
    if dtype == "float64":
        oracle = lambda qq, qqd, uu: float(loss(qq, qqd, uu))      # the fp64 public function
    else:
        oracle = _reference_cases(ref_go2, h)[method]               # the float64 RBDReference (7.z31)
    g_q = np.asarray(jax.grad(loss, argnums=0)(q, qd, u))
    fd_q = _central_fd(lambda x: oracle(x, qd64, u64), q64)
    _assert_grad_close(g_q, fd_q, dtype,
                       err_msg=f"{method}: d/dq (pos|quat|joints) vs ambient central differences ({dtype})")
    # the historically dropped LAST joint slot and the quaternion slots must be live
    assert abs(g_q[-1]) > 1e-6 and np.abs(g_q[3:7]).max() > 1e-6
    if method != "end_effector_pose":
        g_qd = np.asarray(jax.grad(loss, argnums=1)(q, qd, u))
        fd_qd = _central_fd(lambda x: oracle(q64, x, u64), qd64)
        _assert_grad_close(g_qd, fd_qd, dtype, err_msg=f"{method}: d/dqd vs central differences ({dtype})")


# ---------------------------------------------------------------- torch
@pytest.fixture(scope="module")
def torch_go2(dtype):
    pytest.importorskip("torch")
    h = grid_rbd.register_robot(f"w01_go2_torch_{dtype}", str(_GO2), floating_base=True, backend="torch",
                                algorithm_list=_ALGOS, enable_mujoco_kernels=False,
                                dtype=dtype)
    yield h
    h.close()


@pytest.mark.parametrize("unit_quat", [True, False], ids=["unit-quat", "non-unit-quat"])
@pytest.mark.parametrize("method", ["inverse_dynamics", "forward_dynamics", "end_effector_pose"])
def test_torch_q_gradient_matches_ambient_finite_differences(torch_go2, ref_go2, dtype, method, unit_quat):
    import torch
    h = torch_go2
    tdt = getattr(torch, dtype)
    q, qd, u = (np.asarray(a, dtype=_NP[dtype]) for a in _sample(h.num_joints, h.num_vel, unit_quat))
    q64, qd64, u64 = (np.asarray(a, dtype=np.float64) for a in (q, qd, u))
    w_tau = torch.linspace(0.5, 1.5, h.num_vel, dtype=tdt, device="cuda")
    w_pose = torch.linspace(-1.0, 1.0, 6 * h.num_ees, dtype=tdt, device="cuda")
    T = lambda a: torch.as_tensor(a, dtype=tdt, device="cuda")

    def loss(qt, qdt, ut):
        if method == "inverse_dynamics":
            return (w_tau * h.inverse_dynamics(qt[None], qdt[None])[0]).sum()
        if method == "forward_dynamics":
            return (w_tau * h.forward_dynamics(qt[None], qdt[None], ut[None])[0]).sum()
        return (w_pose * h.end_effector_pose(qt[None]).reshape(-1)).sum()

    if dtype == "float64":
        oracle = lambda x: float(loss(T(x), T(qd), T(u)).item())   # the fp64 public function
    else:
        case = _reference_cases(ref_go2, h)[method]                 # the float64 RBDReference (7.z31)
        oracle = lambda x: case(x, qd64, u64)
    qt = T(q).requires_grad_(True)
    loss(qt, T(qd), T(u)).backward()
    g_q = qt.grad.detach().cpu().numpy()
    fd_q = _central_fd(oracle, q64)
    _assert_grad_close(g_q, fd_q, dtype,
                       err_msg=f"{method}: torch d/dq vs ambient central differences ({dtype})")
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
# fp64 only (fp64_alpha, no fp32 sibling): the oracle is a central difference of
# the public function, which needs the fp64 build, and RBDReference offers no
# mujoco-convention evaluation to stand in for it at fp32.
@pytest.fixture(scope="module")
def numpy_go2_mjx():
    h = grid_rbd.register_robot("w01_probe_go2_mjx", str(_GO2), floating_base=True,
                                algorithm_list=["inverse_dynamics", "inverse_dynamics_gradient",
                                                "end_effector_pose", "end_effector_pose_gradient"],
                                enable_mujoco_kernels=True, output_convention="mujoco",
                                dtype="float64")
    yield h
    h.close()


@pytest.mark.fp64_alpha
@pytest.mark.parametrize("method", ["inverse_dynamics", "end_effector_pose"])
def test_mujoco_convention_q_cotangent_is_the_on_manifold_pullback(numpy_go2_mjx, method):
    from grid_rbd._vjp_common import _configuration_cotangent
    h = numpy_go2_mjx
    nq, nv = h.num_joints, h.num_vel
    rng = np.random.default_rng(5)
    q = np.zeros(nq); q[:3] = rng.uniform(-1, 1, 3)
    quat = rng.standard_normal(4); q[3:7] = quat / np.linalg.norm(quat)          # wxyz, UNIT
    q[7:] = rng.uniform(-1, 1, nq - 7)
    qd = rng.uniform(-1, 1, nv); qdd = np.zeros(nv)
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
# The fixture yields (handle, float64 RBDReference of the same URDF).
@pytest.fixture(scope="module", params=[("spherical_arm", False), ("mixed_spherical_arm", False)])
def spherical_handle(request, dtype):
    name, floating = request.param
    urdf = REPO_ROOT / "external/URDFParser/tests/fixtures" / f"{name}.urdf"
    h = grid_rbd.register_robot(f"w01_{name}_{floating}_{dtype}", str(urdf),
                                floating_base=floating, algorithm_list=_ALGOS,
                                enable_mujoco_kernels=False, max_batch_size=8, dtype=dtype)
    yield h, _reference(urdf, floating)
    h.close()


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("method", ["inverse_dynamics", "forward_dynamics", "end_effector_pose"])
def test_spherical_public_q_gradient(spherical_handle, dtype, backend, method):
    framework = pytest.importorskip(backend)
    if backend == "jax": framework.config.update("jax_enable_x64", True)
    base, ref = spherical_handle
    h = grid_rbd.get_robot(base.name, backend=backend)
    rng = np.random.default_rng(81)
    q = rng.uniform(-.5, .5, base.num_joints)
    for kind, qi, vi, nq, nv in base.configuration_layout:
        if kind != "euclidean":
            start = qi + (3 if kind == "floating" else 0)
            q[start:start+4] *= 1.2 / np.linalg.norm(q[start:start+4])
    qd = rng.uniform(-.2, .2, base.num_vel)
    u = rng.uniform(-.3, .3, base.num_vel)
    q, qd, u = (np.asarray(a, dtype=_NP[dtype]) for a in (q, qd, u))
    q64, qd64, u64 = (np.asarray(a, dtype=np.float64) for a in (q, qd, u))
    # fp32: the float64 RBDReference at normalize(q) is the finite-difference oracle (7.z31)
    reference = None if dtype == "float64" else _reference_cases(ref, base)[method]
    try:
        if backend == "jax":
            loss = _jax_cases(h)[method]
            f = lambda x: loss(x, qd, u)
            actual = np.asarray(framework.jit(framework.grad(f))(q))
            oracle = (lambda x: float(f(x))) if reference is None else (lambda x: reference(x, qd64, u64))
            numeric = _central_fd(oracle, q64)
        else:
            tdt = getattr(framework, dtype)
            T = lambda x: framework.as_tensor(x, dtype=tdt, device="cuda")
            def f(x):
                if method == "end_effector_pose": out = h.end_effector_pose(x[None])
                elif method == "inverse_dynamics": out = h.inverse_dynamics(x[None], T(qd)[None])
                else: out = h.forward_dynamics(x[None], T(qd)[None], T(u)[None])
                w = framework.linspace(.5, 1.5, out.numel(), dtype=out.dtype, device=out.device)
                return (out.reshape(-1)*w).sum()
            qt = T(q).requires_grad_(True)
            f(qt).backward()
            actual = qt.grad.cpu().numpy()
            if reference is None:
                oracle = lambda x: float(f(T(x)).item())
            else:
                # the torch loss weights every output by linspace(.5, 1.5, n), unlike _jax_cases
                def oracle(x):
                    return _torch_weighted_reference(ref, base, method, x, qd64, u64)
            numeric = _central_fd(oracle, q64)
        assert actual.shape == q.shape
        _assert_grad_close(actual, numeric, dtype,
                           err_msg=f"{method}/{backend}: public d/dq vs central differences ({dtype})")
    finally:
        h.close()


def _torch_weighted_reference(ref, h, method, q, qd, u):
    """The float64 oracle of the torch spherical loss: every output entry weighted
    by linspace(.5, 1.5, n), evaluated by RBDReference at normalize(q)."""
    qn = _normalized_q(q, h.configuration_layout)
    if method == "end_effector_pose":
        leaves = [ref.robot.get_joint_by_id(jid).get_name() for jid in ref.robot.get_leaf_nodes()]
        out = np.concatenate([np.asarray(ref.end_effector_pose(qn, ee_joint_names=[leaf]),
                                         np.float64).reshape(-1) for leaf in leaves])
    elif method == "inverse_dynamics":
        out, *_ = ref.inverse_dynamics(qn, qd, GRAVITY=-9.81)
    else:
        out = ref.forward_dynamics(qn, qd, u)
    out = np.asarray(out, np.float64).reshape(-1)
    return float(out @ np.linspace(.5, 1.5, out.size))


@pytest.mark.fp64_alpha
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

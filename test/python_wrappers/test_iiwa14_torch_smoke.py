"""PyTorch-backend smoke tests for the `grid-rbd` package (D.3).

Registers iiwa14 (fixed-base) with the torch backend, exercises every method,
and asserts:
  1. forward parity vs the numpy `RobotHandle` (same .so/cache),
  2. analytic backward (autograd) vs central-difference VJP for the 4
     differentiable ops (inverse_dynamics / forward_dynamics / aba / integrator),
  3. CUDA-Graphs capture/replay equivalence vs eager.

Skips when torch / CUDA / nvcc are unavailable.

Run with:
    pytest test/python_wrappers/test_iiwa14_torch_smoke.py -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
torch = pytest.importorskip("torch", reason="torch not installed")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH", allow_module_level=True)

_URDF = _REPO_ROOT / "robot_assets" / "iiwa14.urdf"
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF fixture not present at {_URDF}", allow_module_level=True)


pytestmark = pytest.mark.python_wrappers

_TOL = 5e-3
_GTOL = 2e-2  # loose float32 finite-difference VJP tolerance


@pytest.fixture(scope="module")
def th():
    import grid_rbd.torch as gt
    return gt.register_robot(
        name="iiwa14_torch_smoke", urdf_path=str(_URDF),
        floating_base=False, max_batch_size=64)


@pytest.fixture(scope="module")
def nh():
    # numpy handle on the same robot (same .so/cache) for parity reference.
    return _grid_rbd.get_robot("iiwa14_torch_smoke")


@pytest.fixture(scope="module")
def samples(th):
    rng = np.random.default_rng(0)
    NJ = th.num_joints
    B = 8
    return {
        "qn":  rng.standard_normal((B, NJ)).astype(np.float32),
        "qdn": rng.standard_normal((B, NJ)).astype(np.float32),
        "un":  rng.standard_normal((B, NJ)).astype(np.float32),
        "B": B,
    }


def _t(x):
    return torch.tensor(x, device="cuda", dtype=torch.float32)


def _rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-3)))


# ─── (1) forward parity vs numpy ────────────────────────────────────────────


def test_forward_parity(th, nh, samples):
    qn, qdn, un = samples["qn"], samples["qdn"], samples["un"]
    q, qd, u = _t(qn), _t(qdn), _t(un)
    checks = {
        "inverse_dynamics": (th.inverse_dynamics(q, qd).cpu().numpy(), nh.inverse_dynamics(qn, qdn)),
        "forward_dynamics": (th.forward_dynamics(q, qd, u).cpu().numpy(), nh.forward_dynamics(qn, qdn, un)),
        "aba": (th.aba(q, qd, u).cpu().numpy(), nh.aba(qn, qdn, un)),
        "minv": (th.minv(q).cpu().numpy(), nh.minv(qn)),
        "crba": (th.crba(q).cpu().numpy(), nh.crba(qn)),
        "end_effector_pose": (th.end_effector_pose(q).cpu().numpy(), nh.end_effector_pose(qn)),
        "ee_pose_gradient": (th.end_effector_pose_gradient(q).cpu().numpy(), nh.end_effector_pose_gradient(qn)),
        "ee_pose_hessian": (th.end_effector_pose_hessian(q).cpu().numpy(), nh.end_effector_pose_hessian(qn)),
        "inverse_dynamics_gradient": (th.inverse_dynamics_gradient(q, qd).cpu().numpy(), nh.inverse_dynamics_gradient(qn, qdn)),
        "forward_dynamics_gradient": (th.forward_dynamics_gradient(q, qd, u).cpu().numpy(), nh.forward_dynamics_gradient(qn, qdn, un)),
        "integrator": (th.integrator(q, qd, u, 0.01).detach().cpu().numpy(), nh.integrator(qn, qdn, un, 0.01)),
        "integrator_grad": (th.integrator_gradient(q, qd, u, 0.01).cpu().numpy(), nh.integrator_gradient(qn, qdn, un, 0.01)),
    }
    for name, (a, b) in checks.items():
        assert _rel(a, b) < _TOL, f"{name}: {_rel(a, b):.3e}"


def test_idsva_so_parity(th, nh, samples):
    q, qd = _t(samples["qn"]), _t(samples["qdn"])
    tso = th.idsva_so(q, qd)
    nso = nh.idsva_so(samples["qn"], samples["qdn"])
    for i in range(4):
        assert _rel(tso[i].cpu().numpy(), nso[i]) < _TOL


def test_inverse_dynamics_honors_qdd(th, nh, samples):
    """torch inverse_dynamics must USE qdd: full RNEA τ at a nonzero qdd matches
    the numpy handle; qdd=None == bias; nonzero qdd shifts τ."""
    qn, qdn, an = samples["qn"], samples["qdn"], samples["un"]
    q, qd, a = _t(qn), _t(qdn), _t(an)
    tau = th.inverse_dynamics(q, qd, a).cpu().numpy()
    assert _rel(tau, nh.inverse_dynamics(qn, qdn, an)) < _TOL, "torch RNEA(qdd) != numpy"
    bias = th.inverse_dynamics(q, qd, None).cpu().numpy()
    assert _rel(bias, nh.inverse_dynamics(qn, qdn)) < _TOL
    assert float(np.max(np.abs(tau - bias))) > 1e-2, "torch inverse_dynamics ignored qdd"


# ─── (1b) inertial-parameter (sysID) ops: forward parity vs the JAX surface ──
# The numpy RobotHandle doesn't expose the regressor / param-gradient family;
# the JAX surface is the task's named algorithmic reference, so we cross-check
# the torch regressor + FD-parameter-gradient forward values against it.


@pytest.fixture(scope="module")
def jh():
    jax = pytest.importorskip("jax", reason="jax not installed")
    import grid_rbd.jax as gj
    return gj.get_robot("iiwa14_torch_smoke")


def test_param_op_forward_parity_vs_jax(th, jh, samples):
    import jax.numpy as jnp
    qn, qdn, un = samples["qn"], samples["qdn"], samples["un"]
    q, qd, u = _t(qn), _t(qdn), _t(un)
    # regressor Y = ∂c/∂π at qdd=0  (B, NJ, 10*NB)
    Yt = th.inverse_dynamics_regressor(q, qd).cpu().numpy()
    Yj = np.asarray(jh.inverse_dynamics_regressor(jnp.asarray(qn), jnp.asarray(qdn)))
    assert _rel(Yt, Yj) < _TOL, f"regressor: {_rel(Yt, Yj):.3e}"
    # FD parameter gradient G = ∂qdd/∂π = -M⁻¹·Y  (B, NJ, 10*NB)
    Gt = th.forward_dynamics_parameter_gradient(q, qd, u).cpu().numpy()
    Gj = np.asarray(jh.forward_dynamics_parameter_gradient(
        jnp.asarray(qn), jnp.asarray(qdn), jnp.asarray(un)))
    assert _rel(Gt, Gj) < _TOL, f"fd_param_grad: {_rel(Gt, Gj):.3e}"


# ─── (2) autograd: analytic backward vs central-difference VJP ──────────────


def _fd_vjp_err(fn_apply, args_np, eps=1e-3):
    leafs = [_t(a).requires_grad_(True) for a in args_np]
    out = fn_apply(*leafs)
    gout = torch.randn_like(out)
    (out * gout).sum().backward()
    ana = [l.grad.detach().cpu().numpy().copy() for l in leafs]
    gout_np = gout.cpu().numpy()
    errs = []
    for ai, a in enumerate(args_np):
        ga = np.zeros_like(a)
        for idx in range(a.size):
            ap = [x.copy() for x in args_np]; am = [x.copy() for x in args_np]
            ap[ai].reshape(-1)[idx] += eps; am[ai].reshape(-1)[idx] -= eps
            with torch.no_grad():
                fp = fn_apply(*[_t(x) for x in ap]).cpu().numpy()
                fm = fn_apply(*[_t(x) for x in am]).cpu().numpy()
            ga.reshape(-1)[idx] = np.sum(gout_np * (fp - fm) / (2 * eps))
        # globally-normalized error: max|ana-fd| / (max|fd| + atol). A per-element
        # relative metric blows up on near-zero gradient entries (float32 FD noise).
        errs.append(float(np.max(np.abs(ana[ai] - ga)) / (np.max(np.abs(ga)) + 1e-3)))
    return max(errs)


def test_autograd_inverse_dynamics(th, samples):
    b = 2
    err = _fd_vjp_err(lambda a, c: th.inverse_dynamics(a, c),
                      [samples["qn"][:b], samples["qdn"][:b]])
    assert err < _GTOL, f"inverse_dynamics VJP err {err:.3e}"


def test_autograd_forward_dynamics(th, samples):
    b = 2
    err = _fd_vjp_err(lambda a, c, d: th.forward_dynamics(a, c, d),
                      [samples["qn"][:b], samples["qdn"][:b], samples["un"][:b]])
    assert err < _GTOL, f"fd VJP err {err:.3e}"


def test_autograd_aba(th, samples):
    b = 2
    err = _fd_vjp_err(lambda a, c, d: th.aba(a, c, d),
                      [samples["qn"][:b], samples["qdn"][:b], samples["un"][:b]])
    assert err < _GTOL, f"aba VJP err {err:.3e}"


def test_autograd_integrator(th, samples):
    b = 2
    err = _fd_vjp_err(lambda a, c, d: th.integrator(a, c, d, 0.01),
                      [samples["qn"][:b], samples["qdn"][:b], samples["un"][:b]])
    assert err < _GTOL, f"integrator VJP err {err:.3e}"


# ─── (2b) inertial-parameter (sysID) VJP: torch autograd vs JAX custom_vjp ──
# Analytic-vs-analytic: torch's π-gradient (and q/qd[/u] gradients) from the
# wrt_params ops must match the JAX custom_vjp surface for the SAME scalar loss
# (cotangent), and equal the closed-form contraction grad·Y / grad·G.


def test_autograd_inverse_dynamics_wrt_params(th, jh, samples):
    import jax
    import jax.numpy as jnp
    b = 2
    qn, qdn = samples["qn"][:b], samples["qdn"][:b]
    NB = th.num_bodies
    rng = np.random.default_rng(7)
    pi_n = rng.standard_normal((b, 10 * NB)).astype(np.float32)
    cot = rng.standard_normal((b, th.num_joints)).astype(np.float32)  # shared cotangent

    # torch: scalar loss = <cot, c(q,qd;π)>; backprop to (q, qd, π).
    q = _t(qn).requires_grad_(True)
    qd = _t(qdn).requires_grad_(True)
    pi = _t(pi_n).requires_grad_(True)
    c = th.inverse_dynamics_wrt_params(q, qd, pi)
    (c * _t(cot)).sum().backward()
    gq_t, gqd_t, gpi_t = (x.grad.cpu().numpy() for x in (q, qd, pi))

    # closed form: gpi = cot · Y, with Y the torch regressor at qdd=0.
    Y = th.inverse_dynamics_regressor(_t(qn), _t(qdn)).cpu().numpy()
    gpi_cf = np.einsum("bo,bop->bp", cot, Y)
    assert _rel(gpi_t, gpi_cf) < _TOL, f"id π-grad vs closed-form: {_rel(gpi_t, gpi_cf):.3e}"

    # JAX custom_vjp reference for (q, qd, π).
    def loss(qj, qdj, pij):
        out = jh.inverse_dynamics_wrt_params(qj, qdj, pij)
        return jnp.sum(out * jnp.asarray(cot))
    gq_j, gqd_j, gpi_j = jax.grad(loss, argnums=(0, 1, 2))(
        jnp.asarray(qn), jnp.asarray(qdn), jnp.asarray(pi_n))
    assert _rel(gq_t, np.asarray(gq_j)) < _TOL, f"id grad_q vs jax: {_rel(gq_t, np.asarray(gq_j)):.3e}"
    assert _rel(gqd_t, np.asarray(gqd_j)) < _TOL, f"id grad_qd vs jax: {_rel(gqd_t, np.asarray(gqd_j)):.3e}"
    assert _rel(gpi_t, np.asarray(gpi_j)) < _TOL, f"id grad_π vs jax: {_rel(gpi_t, np.asarray(gpi_j)):.3e}"


def test_autograd_forward_dynamics_wrt_params(th, jh, samples):
    import jax
    import jax.numpy as jnp
    b = 2
    qn, qdn, un = samples["qn"][:b], samples["qdn"][:b], samples["un"][:b]
    NB = th.num_bodies
    rng = np.random.default_rng(11)
    pi_n = rng.standard_normal((b, 10 * NB)).astype(np.float32)
    cot = rng.standard_normal((b, th.num_joints)).astype(np.float32)

    q = _t(qn).requires_grad_(True)
    qd = _t(qdn).requires_grad_(True)
    u = _t(un).requires_grad_(True)
    pi = _t(pi_n).requires_grad_(True)
    qdd = th.forward_dynamics_wrt_params(q, qd, u, pi)
    (qdd * _t(cot)).sum().backward()
    gq_t, gqd_t, gu_t, gpi_t = (x.grad.cpu().numpy() for x in (q, qd, u, pi))

    # closed form: gpi = cot · G, G = ∂qdd/∂π = -M⁻¹·Y.
    G = th.forward_dynamics_parameter_gradient(_t(qn), _t(qdn), _t(un)).cpu().numpy()
    gpi_cf = np.einsum("bo,bop->bp", cot, G)
    assert _rel(gpi_t, gpi_cf) < _TOL, f"fd π-grad vs closed-form: {_rel(gpi_t, gpi_cf):.3e}"

    def loss(qj, qdj, uj, pij):
        out = jh.forward_dynamics_wrt_params(qj, qdj, uj, pij)
        return jnp.sum(out * jnp.asarray(cot))
    gq_j, gqd_j, gu_j, gpi_j = jax.grad(loss, argnums=(0, 1, 2, 3))(
        jnp.asarray(qn), jnp.asarray(qdn), jnp.asarray(un), jnp.asarray(pi_n))
    assert _rel(gq_t, np.asarray(gq_j)) < _TOL, f"fd grad_q vs jax: {_rel(gq_t, np.asarray(gq_j)):.3e}"
    assert _rel(gqd_t, np.asarray(gqd_j)) < _TOL, f"fd grad_qd vs jax: {_rel(gqd_t, np.asarray(gqd_j)):.3e}"
    assert _rel(gu_t, np.asarray(gu_j)) < _TOL, f"fd grad_u vs jax: {_rel(gu_t, np.asarray(gu_j)):.3e}"
    assert _rel(gpi_t, np.asarray(gpi_j)) < _TOL, f"fd grad_π vs jax: {_rel(gpi_t, np.asarray(gpi_j)):.3e}"


# ─── (3) CUDA-Graphs capture/replay ─────────────────────────────────────────


def test_cuda_graph_replay(th, samples):
    q, qd, u = _t(samples["qn"]), _t(samples["qdn"]), _t(samples["un"])
    g = th.capture("forward_dynamics", q, qd, u)
    rng = np.random.default_rng(1)
    B, NJ = samples["B"], th.num_joints
    q2 = _t(rng.standard_normal((B, NJ)).astype(np.float32))
    qd2 = _t(rng.standard_normal((B, NJ)).astype(np.float32))
    u2 = _t(rng.standard_normal((B, NJ)).astype(np.float32))
    captured = g(q2, qd2, u2).clone()
    eager = th.forward_dynamics(q2, qd2, u2)
    # same kernel, same config → identical (bit-for-bit within tight tol).
    assert _rel(captured.cpu().numpy(), eager.detach().cpu().numpy()) < 1e-5
    # a second replay with new inputs must give new correct outputs.
    q3 = _t(rng.standard_normal((B, NJ)).astype(np.float32))
    out3 = g(q3, qd2, u2).clone()
    eager3 = th.forward_dynamics(q3, qd2, u2)
    assert _rel(out3.cpu().numpy(), eager3.detach().cpu().numpy()) < 1e-5


def test_capture_over_max_batch_raises(th):
    NJ = th.num_joints
    big = _t(np.zeros((th.max_batch + 1, NJ), np.float32))
    with pytest.raises(Exception):
        th.forward_dynamics(big, big, big)


# ─── P-tier1: centroidal / kinematics family parity (torch vs numpy oracle) ───

_PTIER1_ARRAY = [
    ("generalized_gravity",        ("qn",)),
    ("nonlinear_effects",          ("qn", "qdn")),
    ("coriolis_matrix",            ("qn", "qdn")),
    ("kinetic_energy_regressor",   ("qn", "qdn")),
    ("potential_energy_regressor", ("qn",)),
    ("energy",                     ("qn", "qdn")),
    ("cmm_time_variation",         ("qn", "qdn")),
    ("dccrba",                     ("qn",)),
    ("frame_jacobian",             ("qn",)),
    ("frame_jacobian_dot",         ("qn", "qdn")),
    ("osc_inertia",                ("qn",)),
]


@pytest.mark.parametrize("method,argnames", _PTIER1_ARRAY, ids=[m for m, _ in _PTIER1_ARRAY])
def test_ptier1_array_matches_numpy(th, nh, samples, method, argnames):
    """Each centroidal/kinematics value op on the torch surface must match the
    plain numpy handle (pinocchio-validated oracle) — same .so, same kernel."""
    nargs = [samples[a] for a in argnames]
    targs = [_t(samples[a]) for a in argnames]
    out_torch = getattr(th, method)(*targs).detach().cpu().numpy()
    out_numpy = np.asarray(getattr(nh, method)(*nargs))
    assert out_torch.shape == out_numpy.shape, f"{method}: {out_torch.shape} vs {out_numpy.shape}"
    assert _rel(out_torch, out_numpy) < _TOL, f"{method}: rel {_rel(out_torch, out_numpy):.3e}"


def test_ptier1_com_ccrba_tuple_matches_numpy(th, nh, samples):
    """com / ccrba return (a, b) tuples — both elements must match the oracle."""
    qn, qdn = samples["qn"], samples["qdn"]
    for name, nargs, targs in (("com", (qn,), (_t(qn),)), ("ccrba", (qn, qdn), (_t(qn), _t(qdn)))):
        tt = getattr(th, name)(*targs)
        nn = getattr(nh, name)(*nargs)
        for i, (a, b) in enumerate(zip(tt, nn)):
            a = a.detach().cpu().numpy(); b = np.asarray(b)
            assert a.shape == b.shape, f"{name}[{i}]: {a.shape} vs {b.shape}"
            assert _rel(a, b) < _TOL, f"{name}[{i}]: rel {_rel(a, b):.3e}"

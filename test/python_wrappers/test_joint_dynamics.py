"""Joint viscous damping + Coulomb friction (opt-in) on the grid_rbd handle.

`register_robot(..., use_joint_dynamics=True)` emits the joint-local bias
`tau += damping*qd + friction*sign(qd)` in the inverse_dynamics / forward_dynamics
/ aba VALUE paths AND its derivative in the id/fd GRADIENT paths (the dc_dqd
diagonal gains `damping`; friction's subgradient is 0 so it contributes nothing).
Everything is gated; the default-OFF build is byte-identical. This validates the
opt-in value AND gradient paths against `RBDReference(use_joint_dynamics=True)`
(the authoritative oracle — pinocchio ignores model.damping/friction), confirms
the DEFAULT (off) build matches the bare (no-damping) oracle, and checks that
FRICTION produces ZERO gradient delta (damping-only sensitivity).

Robots: iiwa14 (damping 0.5×7, no friction), fr3 (damping+friction, mimic).
fp32 + fp64 (numpy backend). Skips if grid_rbd / nvcc / URDF unavailable.

friction_model="tanh" (fr3-tanh row): the smooth f*tanh(qd/w) term replaces
f*sign(qd); its slope (f/w) sech^2(qd/w) enters the dc_dqd diagonal, so the
on-minus-off gradient delta is the oracle's per-sample joint-dynamics slope and
the samples include qd = 0 and |qd| <= w, where the sign model has no slope.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from grid_assets import robot_urdf

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers
_TOL = 1e-3
_IIWA = robot_urdf("iiwa14")
_FR3 = robot_urdf("fr3")
if not (_IIWA.exists() and _FR3.exists()):
    pytest.skip("iiwa14/fr3 URDF not present", allow_module_level=True)


_WIDTH = 0.03


def _oracle(urdf, use_jd, model="sign"):
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    return RBDReference(URDFParser().parse(str(urdf), floating_base=False),
                        use_joint_dynamics=use_jd, friction_model=model, friction_width=_WIDTH)


def _samples(nj, seed=0, dt=np.float32, model="sign"):
    """Four (q, qd, u) rows. Under the tanh model row 0 is at rest and row 1 has
    every velocity inside the width, the regimes the sign model cannot differentiate."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((4, nj)).astype(dt)
    qd = rng.standard_normal((4, nj))
    if model == "tanh":
        qd[0] = 0.0
        qd[1] = rng.uniform(-0.5, 0.5, nj) * _WIDTH
    return q, qd.astype(dt), rng.standard_normal((4, nj)).astype(dt)


def _max(a, b):
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


def _damping_diag(urdf, nv):
    """The expected dc_dqd diagonal contribution from joint damping (no friction):
    sum_jid alpha(jid)*damping(jid) into each reduced v-slot, floating root skipped.
    Mirrors RBDReference.inverse_dynamics_gradient_bpass_dqd / the codegen emit."""
    from URDFParser import URDFParser
    r = URDFParser().parse(str(urdf), floating_base=False)
    diag = np.zeros(nv)
    if not r.robot_has_joint_damping():
        return diag
    for jid in range(r.get_num_bodies()):
        b = float(r.get_damping_by_id(jid))
        if b == 0.0:
            continue
        idx = r.get_joint_index_v(jid)
        j = r.get_joint_by_id(jid)
        alpha = float(j.get_mimic_multiplier()) if getattr(j, "is_mimic", False) else 1.0
        for k in (idx if isinstance(idx, (list, tuple, np.ndarray)) else [idx]):
            diag[k] += alpha * b
    return diag


# (name, urdf, dtype, friction_model) — fp32 + fp64 (numpy backend) for both robots,
# plus the fr3 tanh friction build (fp32). The float64 rows are the fp64 alpha tier
# (outside the receipt; make test-fp64).
_PARAMS = [("iiwa14", _IIWA, "float32", "sign"),
           pytest.param(("iiwa14", _IIWA, "float64", "sign"), marks=pytest.mark.fp64_alpha),
           ("fr3", _FR3, "float32", "sign"),
           pytest.param(("fr3", _FR3, "float64", "sign"), marks=pytest.mark.fp64_alpha),
           ("fr3", _FR3, "float32", "tanh")]
_IDS = ["iiwa14-fp32", "iiwa14-fp64", "fr3-fp32", "fr3-fp64", "fr3-tanh-fp32"]


@pytest.fixture(scope="module", params=_PARAMS, ids=_IDS)
def robot(request):
    name, urdf, dtype, model = request.param
    suf = ("" if dtype == "float32" else "_f64") + ("" if model == "sign" else "_tanh")
    # force_rebuild: the grid-rbd cache key is content-addressed on the BUILD
    # INPUTS (urdf, flags, arch), NOT on the generated CUDA source, so a robot
    # registered before a codegen change keeps its stale .cuh/.so. This test
    # validates the inverse_dynamics_gradient damping EMIT (codegen), so it must
    # regenerate from the CURRENT codegen rather than trust a possibly-stale store
    # (a pre-helper store has the value-path damping but no gradient-path diagonal
    # -> on==off and the assertions silently regress). Rebuild both handles.
    h_on = _grid_rbd.register_robot(name=f"{name}_jd_on{suf}_pytest", urdf_path=str(urdf),
                                    floating_base=False, use_joint_dynamics=True,
                                    friction_model=model,
                                    friction_width=(_WIDTH if model == "tanh" else None),
                                    dtype=dtype, max_batch_size=8, force_rebuild=True)
    h_off = _grid_rbd.register_robot(name=f"{name}_jd_off{suf}_pytest", urdf_path=str(urdf),
                                     floating_base=False, dtype=dtype, max_batch_size=8,
                                     force_rebuild=True)
    np_dt = np.float64 if dtype == "float64" else np.float32
    # fp64 reaches ~1e-9; fp32 ~1e-3.
    tol = 1e-7 if dtype == "float64" else _TOL
    assert h_on.friction_model == model and h_off.friction_model == "sign"
    return h_on, h_off, urdf, np_dt, tol, model


def test_damping_friction_value_path(robot):
    """id/fd/aba with use_joint_dynamics=True match RBDReference(use_joint_dynamics=True)."""
    h_on, _, urdf, dt, tol, model = robot
    nj = h_on.num_joints
    q, qd, u = _samples(nj, dt=dt, model=model)
    ref = _oracle(urdf, use_jd=True, model=model)
    id_g = h_on.inverse_dynamics(q, qd)
    fd_g = h_on.forward_dynamics(q, qd, u)
    aba_g = h_on.aba(q, qd, u)
    for i, (qi, qdi, ui) in enumerate(zip(q, qd, u)):
        qi, qdi, ui = qi.astype(np.float64), qdi.astype(np.float64), ui.astype(np.float64)
        c_ref, *_ = ref.inverse_dynamics(qi, qdi, GRAVITY=-9.81)
        assert _max(id_g[i], c_ref) < tol, f"id[{i}]: {_max(id_g[i], c_ref):.2e}"
        assert _max(fd_g[i], ref.forward_dynamics(qi, qdi, ui)) < tol
        assert _max(aba_g[i], ref.aba(qi, qdi, ui, GRAVITY=-9.81)) < tol


def test_damping_gradient_path(robot):
    """id_gradient + fd_gradient with use_joint_dynamics=True match
    RBDReference(use_joint_dynamics=True) — the damping diagonal flows through
    dc_dqd (id) and qdd_dqd = -Minv*dc_dqd (fd, no separate emit)."""
    h_on, _, urdf, dt, tol, model = robot
    nj = h_on.num_joints
    q, qd, u = _samples(nj, seed=2, dt=dt, model=model)
    ref = _oracle(urdf, use_jd=True, model=model)
    idg = h_on.inverse_dynamics_gradient(q, qd, u)        # (B, NV, 2*NV) = [dc_dq|dc_dqd]
    fdg = h_on.forward_dynamics_gradient(q, qd, u)        # (B, NV, 2*NV) = [qdd_dq|qdd_dqd]
    for i, (qi, qdi, ui) in enumerate(zip(q, qd, u)):
        qi, qdi, ui = qi.astype(np.float64), qdi.astype(np.float64), ui.astype(np.float64)
        idg_ref = np.asarray(ref.inverse_dynamics_gradient(qi, qdi, ui, GRAVITY=-9.81))  # (NV, 2*NV)
        assert _max(idg[i], idg_ref) < tol, f"id_grad[{i}]: {_max(idg[i], idg_ref):.2e}"
        qdd_dq_ref, qdd_dqd_ref = ref.forward_dynamics_gradient(qi, qdi, ui)
        fdg_ref = np.concatenate([qdd_dq_ref, qdd_dqd_ref], axis=-1)
        assert _max(fdg[i], fdg_ref) < tol, f"fd_grad[{i}]: {_max(fdg[i], fdg_ref):.2e}"


def test_gradient_delta_is_damping_diag_only_no_friction(robot):
    """on - off gradient delta is EXACTLY the damping diagonal on dc_dqd; the
    dc_dq half and the off-diagonal dc_dqd are unchanged, and FRICTION (fr3)
    contributes ZERO — proves the friction subgradient is correctly dropped.
    Under the tanh model the delta is the oracle's per-sample slope diagonal
    b + (f/w) sech^2(qd/w), which is largest at rest (row 0)."""
    h_on, h_off, urdf, dt, tol, model = robot
    nj = h_on.num_joints
    nv = h_on.num_vel
    q, qd, u = _samples(nj, seed=3, dt=dt, model=model)
    idg_on = h_on.inverse_dynamics_gradient(q, qd, u)
    idg_off = h_off.inverse_dynamics_gradient(q, qd, u)
    delta = np.asarray(idg_on, np.float64) - np.asarray(idg_off, np.float64)
    ref = _oracle(urdf, use_jd=True, model=model)
    for i in range(delta.shape[0]):
        expected = np.zeros((nv, 2 * nv))
        if model == "tanh":
            slope = ref._joint_dynamics_derivative_diag(qd[i].astype(np.float64), 1)
            expected[:, nv:] = np.diag(slope)
            # the friction slope is present at rest; the sign model would show damping only
            if i == 0:
                assert float(np.max(slope - _damping_diag(urdf, nv))) > 1.0
        else:
            expected[:, nv:] = np.diag(_damping_diag(urdf, nv))   # only dc_dqd diagonal
        scale = max(1.0, float(np.max(np.abs(expected))))
        assert _max(delta[i], expected) / scale < tol, (
            f"grad delta[{i}] != joint-dynamics slope diagonal: {_max(delta[i], expected):.2e}")
    # sanity: the delta is non-trivial (damping is actually present)
    assert float(np.max(np.abs(_damping_diag(urdf, nv)))) > 1e-3


def test_default_off_matches_bare_oracle_and_differs_from_on(robot):
    """Default (no use_joint_dynamics) matches the BARE oracle in BOTH value and
    gradient, and the damping term is non-trivial (on != off) — proves the flag
    actually does something and the default build's gradients are unchanged."""
    h_on, h_off, urdf, dt, tol, model = robot
    nj = h_off.num_joints
    q, qd, u = _samples(nj, seed=1, dt=dt, model=model)
    ref_bare = _oracle(urdf, use_jd=False)
    id_off = h_off.inverse_dynamics(q, qd)
    idg_off = h_off.inverse_dynamics_gradient(q, qd, u)
    for i, (qi, qdi, ui) in enumerate(zip(q, qd, u)):
        qi, qdi, ui = qi.astype(np.float64), qdi.astype(np.float64), ui.astype(np.float64)
        c_ref, *_ = ref_bare.inverse_dynamics(qi, qdi, GRAVITY=-9.81)
        assert _max(id_off[i], c_ref) < tol
        idg_ref = np.asarray(ref_bare.inverse_dynamics_gradient(qi, qdi, ui, GRAVITY=-9.81))
        assert _max(idg_off[i], idg_ref) < tol
    # on vs off must differ (damping present in value + gradient)
    assert _max(h_on.inverse_dynamics(q, qd), id_off) > 1e-2
    assert _max(h_on.inverse_dynamics_gradient(q, qd, u), idg_off) > 1e-3


@pytest.mark.parametrize("urdf,model", [(_IIWA, "sign"), (_FR3, "sign"), (_FR3, "tanh")],
                         ids=["iiwa14", "fr3", "fr3-tanh"])
def test_rbdreference_oracle_dqd_block_matches_fd(urdf, model):
    """RBDReference self-test (no GPU): the analytic inverse_dynamics_gradient
    dc_dqd block (use_joint_dynamics=True) matches a central finite-difference of
    inverse_dynamics w.r.t. qd. Validates the oracle damping fix directly.

    Under the sign model the qd samples are kept WELL away from 0 so the friction
    term f*sign(qd) is locally constant (sign(qd±h)==sign(qd)) -> its
    central-difference is exactly 0, matching the analytic gradient which
    (correctly) drops friction. Under the tanh model the samples sit inside the
    width, where the slope (f/w) sech^2(qd/w) is largest."""
    from URDFParser import URDFParser
    from RBDReference import RBDReference
    rp = URDFParser().parse(str(urdf), floating_base=False)
    nv = rp.get_num_vel()
    ref = RBDReference(rp, use_joint_dynamics=True, friction_model=model, friction_width=_WIDTH)
    rng = np.random.default_rng(7)
    q = rng.standard_normal(nv)
    if model == "tanh":
        qd = rng.uniform(-0.5, 0.5, nv) * _WIDTH
    else:
        # bias velocity sign strongly away from 0 (|qd| >= 0.5) so friction is smooth.
        qd = np.sign(rng.standard_normal(nv)) * (0.5 + np.abs(rng.standard_normal(nv)))
    qdd = rng.standard_normal(nv)
    grad = np.asarray(ref.inverse_dynamics_gradient(q, qd, qdd, GRAVITY=-9.81))
    dqd_analytic = grad[:, nv:]   # dc_dqd half of the hstacked [dc_dq | dc_dqd]
    h = 1e-6
    dqd_fd = np.zeros((nv, nv))
    for k in range(nv):
        qd_p = qd.copy(); qd_p[k] += h
        qd_m = qd.copy(); qd_m[k] -= h
        c_p, *_ = ref.inverse_dynamics(q, qd_p, qdd, GRAVITY=-9.81)
        c_m, *_ = ref.inverse_dynamics(q, qd_m, qdd, GRAVITY=-9.81)
        dqd_fd[:, k] = (np.asarray(c_p) - np.asarray(c_m)) / (2 * h)
    err = float(np.max(np.abs(dqd_analytic - dqd_fd)))
    assert err < 1e-4, f"dc_dqd analytic vs FD mismatch: {err:.2e}"

"""CUDA-free unit tests for the binding-side MuJoCo-convention transforms
(``grid_rbd._mujoco``). The transforms mirror the validated oracle in
``RBDReference/equivalents/mujoco_convention.py``; here we check the batched
slice behaviour, the round-trips, and — crucially — the fixed-base no-op
(the regression guarantee: ``output_convention="mujoco"`` never changes a
fixed-base result).

These need only numpy + the binding package source (no compiled .so, no GPU),
so they run anywhere. An optional check cross-validates against real MuJoCo on a
minimal matched model when ``mujoco`` is importable.
"""
import importlib.util

import numpy as np
import pytest

from grid_rbd import _mujoco as bm


def _state(rng, nq=19, nv=18):
    q = rng.standard_normal(nq); q[3:7] /= np.linalg.norm(q[3:7])
    return q[None], rng.standard_normal(nv)[None], rng.standard_normal(nv)[None]


def test_quat_round_trip():
    rng = np.random.default_rng(0); q, _, _ = _state(rng)
    assert np.allclose(bm.q_mjx_to_pin(bm.q_pin_to_mjx(q, True), True), q, atol=1e-14)
    # the reorder really moves the scalar: pin xyzw[3] -> mjx wxyz[0]
    qm = bm.q_pin_to_mjx(q, True)
    assert qm[0, 3] == q[0, 6] and qm[0, 4] == q[0, 3]


def test_velocity_accel_round_trip():
    rng = np.random.default_rng(1); q, qd, qdd = _state(rng)
    R = bm.base_rotation(q)
    assert np.allclose(bm.v_mjx_to_pin(bm.v_pin_to_mjx(qd, R, True), R, True), qd, atol=1e-13)
    a_mjx = bm.accel_pin_to_mjx(qdd, qd, R, True)
    assert np.allclose(bm.accel_mjx_to_pin(a_mjx, qd, R, True), qdd, atol=1e-13)


def test_mass_matrix_congruence_is_spd_preserving():
    rng = np.random.default_rng(2); q, _, _ = _state(rng)
    R = bm.base_rotation(q)
    A = rng.standard_normal((1, 18, 18)); M = A @ A.transpose(0, 2, 1) + np.eye(18)
    Mm = bm.mass_matrix_pin_to_mjx(M, R, True)
    assert np.allclose(Mm, Mm.transpose(0, 2, 1), atol=1e-10)        # still symmetric
    assert np.linalg.eigvalsh(Mm[0]).min() > 0                       # still SPD
    # kinetic energy is frame-invariant: 1/2 v_mjx^T M_mjx v_mjx == 1/2 v_pin^T M_pin v_pin
    v_pin = rng.standard_normal((1, 18)); v_mjx = bm.v_pin_to_mjx(v_pin, R, True)
    ke_pin = v_pin[0] @ M[0] @ v_pin[0]; ke_mjx = v_mjx[0] @ Mm[0] @ v_mjx[0]
    assert abs(ke_pin - ke_mjx) < 1e-9


@pytest.mark.parametrize("fn,args", [
    ("q_pin_to_mjx", ()), ("v_pin_to_mjx", ("R",)), ("id_tau_pin_to_mjx", ("R",)),
])
def test_fixed_base_is_identity(fn, args):
    rng = np.random.default_rng(3); q, qd, _ = _state(rng, nq=18, nv=18)
    R = bm.base_rotation(np.concatenate([q[:, :3], [[0, 0, 0, 1]], q[:, 3:]], axis=1)) \
        if "R" in args else None
    x = q if fn == "q_pin_to_mjx" else qd
    call = getattr(bm, fn)
    out = call(x, R, False) if "R" in args else call(x, False)
    assert np.array_equal(out, x)            # byte-identical no-op on fixed base


def test_mass_matrix_fixed_base_identity():
    rng = np.random.default_rng(4)
    M = rng.standard_normal((2, 5, 5))
    assert np.array_equal(bm.mass_matrix_pin_to_mjx(M, None, False), M)
    assert np.array_equal(bm.minv_pin_to_mjx(M, None, False), M)


@pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="needs mujoco")
def test_against_real_mujoco_mass_matrix():
    """G·M·G^T (from a known SPD M) reproduces mj_fullM's free-joint block rotation
    on a minimal free body — confirms the G direction and quat order."""
    import mujoco
    xml = """<mujoco><worldbody><body><freejoint/>
      <inertial pos="0 0 0" mass="2" diaginertia="0.1 0.2 0.3"/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    rng = np.random.default_rng(5)
    quat = rng.standard_normal(4); quat /= np.linalg.norm(quat)
    q_pin = np.concatenate([rng.standard_normal(3), quat])[None]
    R = bm.base_rotation(q_pin)
    d.qpos[:] = bm.q_pin_to_mjx(q_pin, True)[0]; mujoco.mj_forward(m, d)
    M_mj = np.zeros((6, 6)); mujoco.mj_fullM(m, M_mj, d.qM)
    # pin mass matrix of a single free body at origin = diag(m,m,m,Ix,Iy,Iz)
    M_pin = np.diag([2., 2., 2., 0.1, 0.2, 0.3])[None]
    M_grid = bm.mass_matrix_pin_to_mjx(M_pin, R, True)[0]
    assert np.abs(M_grid - M_mj).max() < 1e-9

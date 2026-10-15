"""CPU-only gates for the Frax floating-base transports (test/benchmarks/release/frax_floating.py).

The gradient transports are checked against finite differences of the VALUE
transports on a synthetic smooth dynamics model: the test needs no Frax, no
oracle and no GPU, only numpy. The synthetic model has the same structure as a
rigid-body system (an SPD, configuration-dependent mass matrix and a bias that
depends on configuration and velocity), evaluated on the pin configuration
through its local retraction."""
from __future__ import annotations
import numpy as np
import pytest

from test.benchmarks.release import frax_floating as ff

NV = 10          # 6 base + 4 joints
H = 1e-6


def _exp_so3(w):
    angle = np.linalg.norm(w)
    K = ff._skew(w)
    if angle < 1e-12:
        return np.eye(3) + K
    return np.eye(3) + np.sin(angle) / angle * K + (1 - np.cos(angle)) / angle**2 * K @ K


def _quat_xyzw(R):
    w = np.sqrt(max(0., 1 + np.trace(R))) / 2
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([x, y, z, w])


class Synthetic:
    """tau = M(c) a + bias(c, v) on the pin configuration c = (p, R, joints)."""

    def __init__(self, rng):
        self.W = rng.normal(size=(NV * NV, NV + 6)) * .3
        self.b = rng.normal(size=NV * NV)
        self.W2 = rng.normal(size=(NV, NV + 6)) * .3
        self.C = rng.normal(size=(NV, NV, NV)) * .1

    @staticmethod
    def features(p, R, joints):
        return np.concatenate((p, R.ravel(), joints))

    def mass(self, p, R, joints):
        S = np.sin(self.W @ self.features(p, R, joints) + self.b).reshape(NV, NV)
        return S @ S.T + NV * np.eye(NV)

    def bias(self, p, R, joints, v):
        g = np.sin(self.W2 @ self.features(p, R, joints))
        return g * v + np.einsum("ijk,j,k->i", self.C, v, v) + g

    def tau(self, p, R, joints, v, a):
        return self.mass(p, R, joints) @ a + self.bias(p, R, joints, v)

    def accel(self, p, R, joints, v, tau):
        return np.linalg.solve(self.mass(p, R, joints), tau - self.bias(p, R, joints, v))

    # oracle-style tangent-space derivatives (local retraction) by central differences
    def retract(self, p, R, joints, delta):
        return p + R @ delta[:3], R @ _exp_so3(delta[3:6]), joints + delta[6:]

    def tangent_jacobian(self, fn, p, R, joints):
        cols = []
        for k in range(NV):
            e = np.zeros(NV); e[k] = H
            cols.append((fn(*self.retract(p, R, joints, e)) - fn(*self.retract(p, R, joints, -e))) / (2 * H))
        return np.stack(cols, axis=1)

    def jacobian(self, fn, x):
        cols = []
        for k in range(x.size):
            e = np.zeros(x.size); e[k] = H
            cols.append((fn(x + e) - fn(x - e)) / (2 * H))
        return np.stack(cols, axis=1)


def _sample(rng):
    R = _exp_so3(rng.normal(size=3) * .6)
    q_pin = np.concatenate((rng.normal(size=3), _quat_xyzw(R), rng.normal(size=NV - 6)))
    return q_pin, rng.normal(size=NV), rng.normal(size=NV)


def _pin_state(q_frax):
    """The pin configuration (p, R, joints) of a Frax coordinate vector."""
    return q_frax[:3], ff.rotation_from_euler_xyz(q_frax[3:6]), q_frax[6:]


def test_module_self_check_and_round_trips():
    assert ff.self_check()
    rng = np.random.default_rng(3)
    for _ in range(5):
        q_pin, v, a = _sample(rng)
        f = ff.FloatingFrame(q_pin, NV)
        assert np.allclose(ff.rotation_from_euler_xyz(f.theta), f.R, atol=1e-12)
        qd, qdd = f.qdot_frax(v), f.qddot_frax(v, a)
        assert np.allclose(f.G @ qd, v, atol=1e-12)
        assert np.allclose(f.G @ qdd + ff.gdot_times_qdot(f.theta, qd, NV), a, atol=1e-12)
        tau = rng.normal(size=NV)
        assert np.allclose(f.Ginv.T @ f.force_frax(tau), tau, atol=1e-12)
        M = np.eye(NV) + np.outer(tau, tau)
        assert np.allclose(f.minv_frax(np.linalg.inv(M)) @ f.mass_matrix_frax(M), np.eye(NV), atol=1e-9)


def test_closed_form_velocity_map_derivatives_match_complex_step():
    rng = np.random.default_rng(4)
    for _ in range(5):
        theta, qd = rng.normal(size=3), rng.normal(size=NV)
        dG = ff.velocity_map_jacobian(theta, NV)
        for k in range(3):
            shifted = theta.astype(complex); shifted[k] += 1e-20j
            np.testing.assert_allclose(ff.velocity_map(shifted, NV).imag / 1e-20, dG[k], atol=1e-12)
        gdot = ff.gdot_times_qdot(theta, qd, NV)
        fd = (ff.velocity_map(theta + H * qd[3:6], NV) - ff.velocity_map(theta - H * qd[3:6], NV)) / (2 * H) @ qd
        np.testing.assert_allclose(gdot, fd, atol=1e-7)
        J = ff.gdot_times_qdot_jacobian(theta, qd, NV)
        for k in range(3):
            e = np.zeros(3); e[k] = H
            fd = (ff.gdot_times_qdot(theta + e, qd, NV) - ff.gdot_times_qdot(theta - e, qd, NV)) / (2 * H)
            np.testing.assert_allclose(J[:, k], fd, atol=1e-7)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_inverse_dynamics_gradient_transport_matches_finite_differences(seed):
    rng = np.random.default_rng(seed)
    model = Synthetic(rng)
    q_pin, v, a = _sample(rng)
    f = ff.FloatingFrame(q_pin, NV)
    p, R, joints = f.p, f.R, f.joints
    tau_pin = model.tau(p, R, joints, v, a)
    M = model.mass(p, R, joints)
    dtau_dq = model.tangent_jacobian(lambda p_, R_, j_: model.tau(p_, R_, j_, v, a), p, R, joints)
    dtau_dv = model.jacobian(lambda v_: model.tau(p, R, joints, v_, a), v)
    np.testing.assert_allclose(model.jacobian(lambda a_: model.tau(p, R, joints, v, a_), a), M, atol=1e-6)
    dq, dv = f.id_gradient_frax(dtau_dq, dtau_dv, M, tau_pin, v, a)
    # Frax's function: tau_frax(q_frax, qd_frax, qdd_frax) through the VALUE transports.
    qd, qdd = f.qdot_frax(v), f.qddot_frax(v, a)
    def tau_frax(q_frax, qd_frax, qdd_frax):
        p_, R_, j_ = _pin_state(q_frax)
        G = ff.velocity_map(q_frax[3:6], NV)
        v_pin = G @ qd_frax
        a_pin = G @ qdd_frax + ff.gdot_times_qdot(q_frax[3:6], qd_frax, NV)
        return G.T @ model.tau(p_, R_, j_, v_pin, a_pin)
    np.testing.assert_allclose(tau_frax(f.q_frax(), qd, qdd), f.force_frax(tau_pin), atol=1e-10)
    fd_q = model.jacobian(lambda x: tau_frax(x, qd, qdd), f.q_frax())
    fd_v = model.jacobian(lambda x: tau_frax(f.q_frax(), x, qdd), qd)
    np.testing.assert_allclose(dq, fd_q, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(dv, fd_v, atol=1e-5, rtol=1e-6)
    # the joint block is untouched by the base change of coordinates
    np.testing.assert_allclose(dq[6:, 6:], dtau_dq[6:, 6:], atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_forward_dynamics_gradient_transport_matches_finite_differences(seed):
    rng = np.random.default_rng(10 + seed)
    model = Synthetic(rng)
    q_pin, v, tau_pin = _sample(rng)
    f = ff.FloatingFrame(q_pin, NV)
    p, R, joints = f.p, f.R, f.joints
    a_pin = model.accel(p, R, joints, v, tau_pin)
    Minv = np.linalg.inv(model.mass(p, R, joints))
    da_dq = model.tangent_jacobian(lambda p_, R_, j_: model.accel(p_, R_, j_, v, tau_pin), p, R, joints)
    da_dv = model.jacobian(lambda v_: model.accel(p, R, joints, v_, tau_pin), v)
    dq, dv = f.fd_gradient_frax(da_dq, da_dv, Minv, a_pin, v, tau_pin)
    qd, tau_frax = f.qdot_frax(v), f.force_frax(tau_pin)
    def qdd_frax(q_frax, qd_frax, tau_frax_):
        p_, R_, j_ = _pin_state(q_frax)
        G = ff.velocity_map(q_frax[3:6], NV)
        Gi = ff.velocity_map_inverse(q_frax[3:6], NV)
        a = model.accel(p_, R_, j_, G @ qd_frax, Gi.T @ tau_frax_)
        return Gi @ (a - ff.gdot_times_qdot(q_frax[3:6], qd_frax, NV))
    np.testing.assert_allclose(qdd_frax(f.q_frax(), qd, tau_frax), f.qddot_frax(v, a_pin), atol=1e-10)
    fd_q = model.jacobian(lambda x: qdd_frax(x, qd, tau_frax), f.q_frax())
    fd_v = model.jacobian(lambda x: qdd_frax(f.q_frax(), x, tau_frax), qd)
    np.testing.assert_allclose(dq, fd_q, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(dv, fd_v, atol=1e-5, rtol=1e-6)

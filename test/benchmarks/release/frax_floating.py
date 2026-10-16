"""Frax floating-base coordinates <-> the shared Pinocchio-convention fixture.

Frax (`Robot(..., add_floating_base=True)`) prepends SIX scalar joints to the
URDF chain -- x/y/z prismatic along the world axes, then roll/pitch/yaw revolute
joints about the x, y and z axes of the preceding frame -- so its base pose is
``p = q[0:3]`` and ``R = Rx(q[3]) Ry(q[4]) Rz(q[5])`` (intrinsic XYZ Euler) and
``nq == nv``. The fixture is Pinocchio's free flyer: ``q = [p, quat_xyzw, joints]``,
tangent ``v = [v_lin_local, omega_local, joint_rates]``.

Both describe the same physical base frame, so every quantity transports through
the velocity map ``v_pin = G(theta) qdot_frax`` with

    G = blockdiag([[R^T, 0], [0, R^T E(theta)]], I)      E(theta) = d omega_world / d theta

(``E`` columns: x_hat, Rx(a) y_hat, Rx(a) Ry(b) z_hat). Consequences used below:

    qdot_frax  = G^{-1} v_pin
    qddot_frax = G^{-1} (a_pin - Gdot qdot_frax)         (a_pin = d v_pin / dt, Pinocchio's joint acceleration)
    tau_frax   = G^T tau_pin                              (virtual work)
    M_frax     = G^T M_pin G,     Minv_frax = G^{-1} Minv_pin G^{-T}
    nle_frax   = G^T (nle_pin + M_pin Gdot qdot_frax)     (Frax's bias is at qddot_frax = 0, i.e. a_pin = Gdot qdot)
    J_frax     = J_pin G                                  (a pose Jacobian w.r.t. qdot_frax instead of the pin tangent)

``Gdot qdot = sum_k thetadot_k (dG/dtheta_k) qdot`` uses the closed-form
``dG/dtheta_k`` (``dR/da = [x]_x R``, ``dR/db = Rx [y]_x Ry Rz``, ``dR/dc = R [z]_x``
and the matching columns of ``dE``); every transport runs in float64 outside the
timed window. Nothing here touches Frax's own arithmetic -- the timed call is
Frax's unmodified function on Frax's native coordinates.

Gradients. A configuration perturbation ``delta`` of ``q_frax`` moves the pin
configuration along the same map, ``delta_pin = G(theta) delta`` (the local
translation ``R^T delta_p`` and the local rotation ``R^T E delta_theta``), so the
oracle's tangent-space derivatives ``dtau/dq_pin`` compose with ``G`` exactly as
the velocities do. Frax's autodiff gradient holds ``qdot_frax`` and
``qddot_frax`` fixed while ``theta`` moves, which drags ``v_pin = G qdot`` and
``a_pin = G qddot + Gdot qdot`` along. Writing ``dG_k = dG/dtheta_k`` and
``Gdot = sum_k qdot_{3+k} dG_k``:

    Kv[:, 3+k] = dG_k qdot                                (d v_pin / d theta_k)
    Kq[:, 3+k] = dG_k qddot + d(Gdot qdot)/d theta_k      (d a_pin / d theta_k)
    Ka         = Gdot + [0 | dG_k qdot | 0]               (d a_pin / d qdot_frax, extra columns 3:6)
    P[:, 3+k]  = dG_k^T tau_pin                           (the output prefactor's own derivative)

    dtau_frax/dq_frax  = P + G^T (dtau/dq_pin G + dtau/dv_pin Kv + M Kq)
    dtau_frax/dqd_frax = G^T (dtau/dv_pin G + M Ka)

For forward dynamics ``qddot_frax = G^{-1} (a_pin(q, v_pin, tau_pin) - Gdot qdot)``
with ``tau_pin = G^{-T} tau_frax`` held fixed in Frax coordinates, so
``d tau_pin / d theta_k = -G^{-T} dG_k^T tau_pin`` and
``d G^{-1} / d theta_k = -G^{-1} dG_k G^{-1}``:

    dqdd_frax/dq_frax [:, 3+k] += -G^{-1} dG_k qddot_frax
                                + G^{-1} (da/dv_pin dG_k qdot - Minv G^{-T} dG_k^T tau_pin - d(Gdot qdot)/d theta_k)
    dqdd_frax/dq_frax          += G^{-1} da/dq_pin G
    dqdd_frax/dqd_frax          = G^{-1} (da/dv_pin G - Ka)

``d(Gdot qdot)/d theta_k`` is a complex step on the closed-form ``Gdot qdot``.
Second derivatives are not transported: the Euler-to-tangent map's Hessian would
enter the comparison and the cell would no longer be like-for-like with the
analytical second-order kernels.

Chart conditioning. The intrinsic XYZ Euler chart is singular at pitch +-pi/2:
``E(theta)`` loses rank there, so ``G`` and every transported quantity scale
like ``1/cos(pitch)`` and the mass matrix Frax factorises, ``M_frax = G^T M G``,
has condition number of order ``1/cos^2(pitch)`` times the tangent-space one
(1e6 at a few degrees from the singularity). Frax's fp32 Cholesky solve of that
matrix on the GPU then carries 1e-3 to 1e-2 relative error, against 1e-4 on the
rest of the fixture; the transport itself reproduces the oracle to 1e-12 in
float64 at the same samples. ``chart_condition`` is ``1/|cos(pitch)|``, and the
samples above ``CHART_CONDITION_MAX`` are left out of the validation of Frax's
floating-base cells (never out of the timing); the collector records the count.
"""

from __future__ import annotations
import numpy as np

_EX, _EY, _EZ = np.eye(3)

# Validation scope for Frax's floating-base cells: samples closer than about
# 5.7 degrees to the Euler pitch singularity (chart condition above 10) are not
# validated. Chosen from the 2026-10-16 captures: the retained samples sit in the
# fp32 class (relative L2 below 3e-4 on go2 and G1 at every batch) and about one
# sample in a hundred of the fixture is excluded.
CHART_CONDITION_MAX = 10.0


def chart_condition(theta):
    """1/|cos(pitch)|: the growth factor of the Euler chart's velocity map."""
    return 1.0 / max(abs(float(np.cos(theta[1]))), np.finfo(float).tiny)


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_from_quat_xyzw(quat):
    x, y, z, w = quat / np.linalg.norm(quat)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def euler_xyz_from_rotation(R):
    """theta with R == Rx(theta0) Ry(theta1) Rz(theta2); pitch kept in (-pi/2, pi/2)."""
    b = np.arctan2(R[0, 2], np.hypot(R[0, 0], R[0, 1]))
    a = np.arctan2(-R[1, 2], R[2, 2])
    c = np.arctan2(-R[0, 1], R[0, 0])
    return np.array([a, b, c])


def rotation_from_euler_xyz(theta):
    return _rot_x(theta[0]) @ _rot_y(theta[1]) @ _rot_z(theta[2])


def euler_rate_map(theta):
    """E(theta): omega_world = E thetadot for the intrinsic XYZ chain."""
    rx = _rot_x(theta[0])
    return np.stack([_EX, rx @ _EY, rx @ _rot_y(theta[1]) @ _EZ], axis=1)


def velocity_map(theta, nv):
    """G(theta): v_pin = G qdot_frax (complex-step safe)."""
    R = rotation_from_euler_xyz(theta)
    G = np.eye(nv, dtype=np.result_type(theta, float))
    G[0:3, 0:3] = R.T
    G[3:6, 3:6] = R.T @ euler_rate_map(theta)
    return G


def velocity_map_inverse(theta, nv):
    R = rotation_from_euler_xyz(theta)
    Gi = np.eye(nv)
    Gi[0:3, 0:3] = R
    Gi[3:6, 3:6] = np.linalg.solve(euler_rate_map(theta), R)
    return Gi


def velocity_map_jacobian(theta, nv):
    """dG/dtheta as an array (3, nv, nv), closed form and complex-step safe."""
    rx, ry, rz = _rot_x(theta[0]), _rot_y(theta[1]), _rot_z(theta[2])
    R = rx @ ry @ rz
    dR = (_skew(_EX) @ R, rx @ _skew(_EY) @ ry @ rz, R @ _skew(_EZ))
    E = euler_rate_map(theta)
    zero = np.zeros(3)
    dE = (np.stack([zero, _skew(_EX) @ rx @ _EY, _skew(_EX) @ rx @ ry @ _EZ], axis=1),
          np.stack([zero, zero, rx @ _skew(_EY) @ ry @ _EZ], axis=1),
          np.zeros((3, 3)))
    out = np.zeros((3, nv, nv), dtype=np.result_type(theta, float))
    for k in range(3):
        out[k, 0:3, 0:3] = dR[k].T
        out[k, 3:6, 3:6] = dR[k].T @ E + R.T @ dE[k]
    return out


def gdot_matrix(theta, qdot, nv):
    """Gdot = sum_k thetadot_k dG/dtheta_k for thetadot = qdot[3:6]."""
    dG = velocity_map_jacobian(theta, nv)
    return np.tensordot(np.asarray(qdot)[3:6], dG, axes=(0, 0))


def gdot_times_qdot(theta, qdot, nv):
    """(d/dt G(theta(t))) qdot, closed form (complex-step safe in theta)."""
    return gdot_matrix(theta, qdot, nv) @ np.asarray(qdot)


def gdot_times_qdot_jacobian(theta, qdot, nv, step=1e-20):
    """d(Gdot qdot)/d theta as (nv, 3), by complex step on the closed form."""
    out = np.zeros((nv, 3))
    for k in range(3):
        shifted = np.asarray(theta, dtype=complex).copy()
        shifted[k] += 1j * step
        out[:, k] = gdot_times_qdot(shifted, np.asarray(qdot, np.float64), nv).imag / step
    return out


class FloatingFrame:
    """All transports for one fixture sample, built from the pin configuration."""

    def __init__(self, q_pin, nv):
        self.nv = nv
        q_pin = np.asarray(q_pin, np.float64)
        self.p = q_pin[0:3]
        self.R = rotation_from_quat_xyzw(q_pin[3:7])
        self.theta = euler_xyz_from_rotation(self.R)
        self.chart_condition = chart_condition(self.theta)
        self.joints = q_pin[7:]
        self.G = velocity_map(self.theta, nv)
        self.Ginv = velocity_map_inverse(self.theta, nv)
        self.dG = velocity_map_jacobian(self.theta, nv)

    def q_frax(self):
        return np.concatenate((self.p, self.theta, self.joints))

    def qdot_frax(self, v_pin):
        return self.Ginv @ np.asarray(v_pin, np.float64)

    def qddot_frax(self, v_pin, a_pin):
        qd = self.qdot_frax(v_pin)
        return self.Ginv @ (np.asarray(a_pin, np.float64) - gdot_times_qdot(self.theta, qd, self.nv))

    def force_frax(self, tau_pin):
        return self.G.T @ np.asarray(tau_pin, np.float64)

    def mass_matrix_frax(self, M_pin):
        return self.G.T @ np.asarray(M_pin, np.float64) @ self.G

    def minv_frax(self, Minv_pin):
        return self.Ginv @ np.asarray(Minv_pin, np.float64) @ self.Ginv.T

    def bias_frax(self, nle_pin, M_pin, v_pin):
        qd = self.qdot_frax(v_pin)
        a_pin = gdot_times_qdot(self.theta, qd, self.nv)
        return self.G.T @ (np.asarray(nle_pin, np.float64) + np.asarray(M_pin, np.float64) @ a_pin)

    def pose_jacobian_frax(self, J_pin):
        return np.asarray(J_pin, np.float64) @ self.G

    # ── gradient transports (first order only; see the module docstring) ──
    def _coupling(self, qd, qdd):
        """(Kv, Kq, Ka, dGd): the theta- and qdot-dependence of v_pin and a_pin at
        fixed Frax inputs; dGd[:, k] = d(Gdot qdot)/d theta_k."""
        nv = self.nv
        Kv, Kq = np.zeros((nv, nv)), np.zeros((nv, nv))
        Ka = gdot_matrix(self.theta, qd, nv)
        dGd = gdot_times_qdot_jacobian(self.theta, qd, nv)
        for k in range(3):
            Kv[:, 3 + k] = self.dG[k] @ qd
            Kq[:, 3 + k] = self.dG[k] @ qdd + dGd[:, k]
            Ka[:, 3 + k] += self.dG[k] @ qd
        return Kv, Kq, Ka, dGd

    def id_gradient_frax(self, dtau_dq, dtau_dv, M_pin, tau_pin, v_pin, a_pin):
        """(dtau_frax/dq_frax, dtau_frax/dqd_frax) from the oracle's tangent-space
        RNEA derivatives, with qdot_frax and qddot_frax held fixed."""
        nv = self.nv
        dtau_dq, dtau_dv, M = (np.asarray(x, np.float64) for x in (dtau_dq, dtau_dv, M_pin))
        tau_pin = np.asarray(tau_pin, np.float64)
        qd, qdd = self.qdot_frax(v_pin), self.qddot_frax(v_pin, a_pin)
        Kv, Kq, Ka, _ = self._coupling(qd, qdd)
        P = np.zeros((nv, nv))
        for k in range(3):
            P[:, 3 + k] = self.dG[k].T @ tau_pin
        dq = P + self.G.T @ (dtau_dq @ self.G + dtau_dv @ Kv + M @ Kq)
        dv = self.G.T @ (dtau_dv @ self.G + M @ Ka)
        return dq, dv

    def fd_gradient_frax(self, da_dq, da_dv, Minv_pin, a_pin, v_pin, tau_pin):
        """(dqdd_frax/dq_frax, dqdd_frax/dqd_frax) from the oracle's tangent-space
        forward-dynamics derivatives, with qdot_frax and tau_frax held fixed."""
        nv = self.nv
        da_dq, da_dv, Minv = (np.asarray(x, np.float64) for x in (da_dq, da_dv, Minv_pin))
        tau_pin = np.asarray(tau_pin, np.float64)
        qd, qdd = self.qdot_frax(v_pin), self.qddot_frax(v_pin, a_pin)
        Kv, _, Ka, dGd = self._coupling(qd, qdd)
        dq = self.Ginv @ da_dq @ self.G
        for k in range(3):
            dtau_k = -self.Ginv.T @ self.dG[k].T @ tau_pin
            dq[:, 3 + k] += self.Ginv @ (-self.dG[k] @ qdd + da_dv @ Kv[:, 3 + k]
                                         + Minv @ dtau_k - dGd[:, k])
        dv = self.Ginv @ (da_dv @ self.G - Ka)
        return dq, dv


def self_check(rng=np.random.default_rng(0), nv=9):
    """Round-trip and derivative identities, independent of Frax (CPU, fast)."""
    quat = rng.normal(size=4)
    q = np.concatenate((rng.normal(size=3), quat / np.linalg.norm(quat), rng.normal(size=nv - 6)))
    f = FloatingFrame(q, nv)
    assert np.allclose(rotation_from_euler_xyz(f.theta), f.R, atol=1e-12)
    assert np.allclose(f.G @ f.Ginv, np.eye(nv), atol=1e-12)
    # Closed-form dG/dtheta against a complex step of G.
    for k in range(3):
        shifted = f.theta.astype(complex)
        shifted[k] += 1e-20j
        assert np.allclose(velocity_map(shifted, nv).imag / 1e-20, f.dG[k], atol=1e-12)
    # Finite-difference check of Gdot qdot against the closed form.
    qd = rng.normal(size=nv)
    h = 1e-6
    fd = (velocity_map(f.theta + h*qd[3:6], nv) - velocity_map(f.theta - h*qd[3:6], nv)) / (2*h) @ qd
    assert np.allclose(fd, gdot_times_qdot(f.theta, qd, nv), atol=1e-7)
    # E maps Euler rates to the world angular velocity: R^T Rdot = [omega_body]_x.
    thetadot = qd[3:6]
    Rc = rotation_from_euler_xyz(f.theta.astype(complex) + 1j*1e-20*thetadot)
    Rdot = Rc.imag / 1e-20
    omega_body_skew = f.R.T @ Rdot
    omega_body = np.array([omega_body_skew[2, 1], omega_body_skew[0, 2], omega_body_skew[1, 0]])
    assert np.allclose(omega_body, f.R.T @ euler_rate_map(f.theta) @ thetadot, atol=1e-12)
    return True


if __name__ == "__main__":
    print("frax_floating self-check:", self_check())

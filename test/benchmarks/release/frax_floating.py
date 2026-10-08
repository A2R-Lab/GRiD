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

``Gdot qdot`` is evaluated by complex step on theta, exact to rounding; every
transport runs in float64 outside the timed window. Nothing here touches Frax's
own arithmetic -- the timed call is Frax's unmodified function on Frax's native
coordinates.
"""
from __future__ import annotations
import numpy as np


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
    return np.stack([np.array([1, 0, 0]), rx @ np.array([0, 1, 0]),
                     rx @ _rot_y(theta[1]) @ np.array([0, 0, 1])], axis=1)


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


def gdot_times_qdot(theta, qdot, nv, step=1e-20):
    """(d/dt G(theta(t))) qdot = Im[G(theta + i h thetadot)] / h  qdot."""
    thetadot = qdot[3:6]
    Gc = velocity_map(theta.astype(complex) + 1j * step * thetadot, nv)
    return (Gc.imag / step) @ qdot


class FloatingFrame:
    """All transports for one fixture sample, built from the pin configuration."""

    def __init__(self, q_pin, nv):
        self.nv = nv
        q_pin = np.asarray(q_pin, np.float64)
        self.p = q_pin[0:3]
        self.R = rotation_from_quat_xyzw(q_pin[3:7])
        self.theta = euler_xyz_from_rotation(self.R)
        self.joints = q_pin[7:]
        self.G = velocity_map(self.theta, nv)
        self.Ginv = velocity_map_inverse(self.theta, nv)

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


def self_check(rng=np.random.default_rng(0), nv=9):
    """Round-trip and derivative identities, independent of Frax (CPU, fast)."""
    quat = rng.normal(size=4)
    q = np.concatenate((rng.normal(size=3), quat / np.linalg.norm(quat), rng.normal(size=nv - 6)))
    f = FloatingFrame(q, nv)
    assert np.allclose(rotation_from_euler_xyz(f.theta), f.R, atol=1e-12)
    assert np.allclose(f.G @ f.Ginv, np.eye(nv), atol=1e-12)
    # Finite-difference check of Gdot qdot against the complex step.
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

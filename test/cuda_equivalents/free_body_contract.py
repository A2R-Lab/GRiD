"""Independent Newton/Euler and SE(3) oracle for the base-force contract.

No GRiD or RBDReference mathematics is called here. The fixture's constants are
deliberately explicit rather than obtained through production model metadata.
"""
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

URDF = Path(__file__).with_suffix('.urdf')
MASS = 2.3
COM = np.array([.13, -.07, .09])
INERTIA = np.diag([.31, .43, .57])


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def acceleration(q, v, force, gravity=-9.81):
    """Body-origin component derivatives, with force=[f_O, torque_O]."""
    linear, omega = np.asarray(v[:3]), np.asarray(v[3:])
    f, torque = np.asarray(force[:3]), np.asarray(force[3:])
    angular = np.linalg.solve(INERTIA, torque - np.cross(COM, f)
                              - np.cross(omega, INERTIA @ omega))
    local_g = Rotation.from_quat(q[3:7]).inv().apply([0., 0., gravity])
    dv = (f / MASS + local_g - np.cross(omega, linear)
          - np.cross(angular, COM) - np.cross(omega, np.cross(omega, COM)))
    return np.r_[dv, angular]


def integrate(q, delta):
    twist = np.zeros((4, 4))
    twist[:3, :3], twist[:3, 3] = skew(delta[3:]), delta[:3]
    inc = expm(twist)
    rotation = Rotation.from_quat(q[3:7])
    return np.r_[q[:3] + rotation.apply(inc[:3, 3]),
                 (rotation * Rotation.from_matrix(inc[:3, :3])).as_quat()]


def difference(a, b):
    ra = Rotation.from_quat(a[3:7])
    phi = (ra.inv() * Rotation.from_quat(b[3:7])).as_rotvec()
    # The top-right block of exp([[phi^, I], [0, 0]]) is the SO(3) left Jacobian.
    augmented = np.zeros((6, 6))
    augmented[:3, :3], augmented[:3, 3:] = skew(phi), np.eye(3)
    jac = expm(augmented)[:3, 3:]
    rho = np.linalg.solve(jac, ra.inv().apply(b[:3] - a[:3]))
    return np.r_[rho, phi]


def step(q, v, force, dt, semi=False):
    updated = v + dt * acceleration(q, v, force)
    return np.r_[integrate(q, dt * (updated if semi else v)), updated]


def step_jacobian(q, v, force, dt, semi=False, epsilon=1e-5):
    nominal = step(q, v, force, dt, semi)
    result = np.empty((12, 18))
    for col in range(18):
        d = np.eye(18)[col] * epsilon
        values = []
        for sign in (1, -1):
            changed = step(integrate(q, sign*d[:6]), v+sign*d[6:12],
                           force+sign*d[12:], dt, semi)
            values.append(np.r_[difference(nominal[:7], changed[:7]), changed[7:]-nominal[7:]])
        result[:, col] = (values[0]-values[1]) / (2*epsilon)
    return result


def rotor_map():
    """Four +body-z thrusts in N, moment arms relative to the body origin."""
    arms = np.array([[.24, .19, .02], [-.24, .19, .02],
                     [-.24, -.19, .02], [.24, -.19, .02]])
    directions = np.tile([0., 0., 1.], (4, 1))
    torques = np.cross(arms, directions)
    torques[:, 2] = [.018, -.018, .018, -.018]
    return np.vstack([directions.T, torques.T])


def cases():
    q = np.r_[[.4, -.2, .8], Rotation.from_rotvec([.31, -.23, .17]).as_quat()]
    v = np.array([.24, -.33, .19, .7, -.4, .5])
    forces = [np.zeros(6), *np.diag([1.7, -2.1, 3.2, -.4, .6, -.8]),
              rotor_map() @ np.array([3.2, 4.1, 2.9, 3.7])]
    return [(q.copy(), v.copy(), np.asarray(f)) for f in forces]

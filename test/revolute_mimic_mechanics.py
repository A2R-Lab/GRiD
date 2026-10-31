"""Closed-form mechanics of the skew revolute mimic fixture, independent of every spatial-algebra library.

The fixture ``revolute_mimic_fixed_tool`` has two bodies on one skew axis line: the
rotor at angle q and the follower, a revolute mimic with multiplier -0.5 and offset
0.17, at absolute angle 0.5 q + 0.17. Both inertia tensors are 0.01 times the
identity about their centres of mass, so each body's moment about the axis is
0.01 plus mass times the squared distance of its centre of mass from the axis.
The generalized mass is constant, so the single coordinate has no velocity
terms; inverse dynamics is M qdd plus the gravity term, the derivative of the
potential with respect to q. Everything below uses only numpy and scipy.
"""
import numpy as np
from scipy.spatial.transform import Rotation

AXIS = np.array([.6, 0., .8])
ORIGIN_R = Rotation.from_euler("xyz", [.3, .2, .1]).as_matrix()
BODIES = (  # (mass, centre of mass in the shared joint frame, angle slope, angle offset)
    (1.0, np.array([.03, .02, -.01]), 1.0, 0.0),
    (0.7, np.array([.05, -.02, .03]), 0.5, 0.17),
)
GRAVITY = 9.81


def generalized_mass():
    total = 0.
    for mass, com, slope, _ in BODIES:
        total += slope ** 2 * (0.01 + mass * np.sum(np.cross(com, AXIS) ** 2))
    return total


def gravity_torque(q):
    """d/dq of the potential sum of m g z, in joint units."""
    total = 0.
    for mass, com, slope, offset in BODIES:
        theta = slope * q + offset
        rotated = Rotation.from_rotvec(AXIS * theta).apply(com)
        d_rotated = np.cross(AXIS, rotated)  # d/dtheta of Rot(axis, theta) com
        total += mass * GRAVITY * (ORIGIN_R @ d_rotated)[2] * slope
    return total


def inverse_dynamics(q, qdd):
    return generalized_mass() * qdd + gravity_torque(q)


def forward_dynamics(q, tau):
    return (tau - gravity_torque(q)) / generalized_mass()

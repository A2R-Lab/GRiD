"""Test helper utilities (formerly in util/util.py)."""
from __future__ import annotations

import random

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=100)




def rand3_to_quat(u, v, w):
    u = abs(u)
    v = abs(v)
    w = abs(w)
    while u > 1:
        u -= 1
    while v > 1:
        v -= 1
    while w > 1:
        w -= 1
    sqrtomu = np.sqrt(1 - u)
    sqrtu = np.sqrt(u)
    PI = 3.14159
    a = sqrtomu * np.sin(2 * PI * v)
    b = sqrtomu * np.cos(2 * PI * v)
    c = sqrtu * np.sin(2 * PI * w)
    d = sqrtu * np.cos(2 * PI * w)
    return (a, b, c, d)


def initializeValues(robot, MATCH_CPP_RANDOM=False):
    n = robot.get_num_pos()
    m = robot.get_num_vel()
    q = np.zeros((n))
    qd = np.zeros((m))
    u = np.zeros((m))

    if MATCH_CPP_RANDOM:
        cpp_random = [-0.3369, 1.2966, -0.6775, -1.4218, -0.7067, -0.1350, -1.1495, 0.4330, -0.4216, -0.6454, -1.8605, -0.0131, -0.4583, 0.7412, 0.7418, 1.9284, -0.9039, 0.0334, 1.1799, -1.9460, 0.3287]
        for i in range(n):
            q[i] = cpp_random[i]
        for i in range(m):
            qd[i] = cpp_random[i + n]
        for i in range(m):
            u[i] = cpp_random[i + n + m]
    else:
        for i in range(n):
            q[i] = random.random()
            if i < m:
                qd[i] = random.random()
                u[i] = random.random()

    if robot.floating_base and robot.using_quaternion:
        (a, b, c, d) = rand3_to_quat(q[3], q[4], q[5])
        q[3] = a
        q[4] = b
        q[5] = c
        q[6] = d

    return q, qd, u, n

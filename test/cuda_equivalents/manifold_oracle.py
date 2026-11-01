"""Reference models, admitted states and the fourth-order stencil of the manifold tests.

A helper module shared by the CPU manifold tests and the GPU primitive and
spherical plant Hessian suites; the split driver fingerprints it into their shards.
"""
from pathlib import Path

import numpy as np
from RBDReference import RBDReference
from URDFParser import URDFParser

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / 'external/URDFParser/tests/fixtures'


def reference(name='mixed_spherical_arm', floating=False):
    return RBDReference(URDFParser().parse(str(FIX / (name + '.urdf')), floating_base=floating))


def state(ref, seed=430):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-.5, .5, ref.robot.get_num_pos())
    for _, iq, _ in ref._joint_retract_specs():
        q[iq[-4:]] /= np.linalg.norm(q[iq[-4:]])
    return q, rng.uniform(-.6, .6, ref.robot.get_num_vel())


def derivative(f, width, h=2e-4):
    if width == 0:
        return np.empty(np.shape(f(np.empty(0))) + (0,))
    columns = []
    for k in range(width):
        e = np.eye(width)[k] * h
        columns.append((8*(f(e)-f(-e)) - (f(2*e)-f(-2*e)))/(12*h))
    return np.stack(columns, axis=-1)

"""Skew revolute mimic dynamics in the numpy oracle against closed-form mechanics.

Pinocchio has no revolute mimic with a nonunit multiplier, so this fixture's
inverse dynamics and mass matrix are pinned to the independent closed form in
``test/cuda_equivalents/revolute_mimic_mechanics.py`` instead (placements and inertias are still
checked against Pinocchio by ``test_urdf_oracle``). The CUDA twin is
``test/cuda_equivalents/test_cuda_revolute_mimic_mechanics.py``.
"""
import numpy as np
import pytest
from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_assets import robot_urdf
from test.cuda_equivalents import revolute_mimic_mechanics as mech

CASES = ((0., 0., 0.), (.13, -.4, 2.), (.5, .7, -3.), (-1.2, .25, .4))


@pytest.fixture(scope="module")
def reference():
    robot = URDFParser().parse(str(robot_urdf("revolute_mimic_fixed_tool")))
    assert any(getattr(j, "is_mimic", False) for j in robot.get_joints_ordered_by_id())
    return RBDReference(robot)


def test_generalized_mass_is_constant_and_matches_closed_form(reference):
    expected = mech.generalized_mass()
    for q, _, _ in CASES:
        np.testing.assert_allclose(reference.crba(np.array([q])), [[expected]], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(reference.minv(np.array([q])), [[1. / expected]], atol=1e-10, rtol=1e-10)


def test_inverse_dynamics_matches_closed_form(reference):
    for q, qd, qdd in CASES:
        tau = reference.inverse_dynamics(np.array([q]), np.array([qd]), np.array([qdd]))[0]
        np.testing.assert_allclose(np.asarray(tau).reshape(-1), [mech.inverse_dynamics(q, qdd)],
                                   atol=1e-12, rtol=1e-12)


def test_forward_dynamics_and_aba_match_closed_form(reference):
    for q, qd, tau in CASES:
        expected = [mech.forward_dynamics(q, tau)]
        np.testing.assert_allclose(np.asarray(reference.forward_dynamics(np.array([q]), np.array([qd]),
                                                                         np.array([tau]))).reshape(-1),
                                   expected, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(reference.aba(np.array([q]), np.array([qd]),
                                                            np.array([tau]))).reshape(-1),
                                   expected, atol=1e-10, rtol=1e-10)


def test_gravity_torque_is_the_potential_derivative():
    for q, _, _ in CASES:
        h = 1e-6
        def potential(x):
            from scipy.spatial.transform import Rotation
            total = 0.
            for mass, com, slope, offset in mech.BODIES:
                p = mech.ORIGIN_R @ Rotation.from_rotvec(mech.AXIS * (slope * x + offset)).apply(com)
                total += mass * mech.GRAVITY * p[2]
            return total
        np.testing.assert_allclose(mech.gravity_torque(q), (potential(q + h) - potential(q - h)) / (2 * h),
                                   atol=1e-8, rtol=1e-8)

"""CPU physics/oracle check for the native six-row base-force contract."""
import numpy as np
import pytest
from RBDReference import RBDReference
from URDFParser import URDFParser
from test.cuda_equivalents.free_body_contract import (URDF, MASS, COM, INERTIA, acceleration,
                                     cases, rotor_map, step, step_jacobian)


def test_free_body_force_values_and_power():
    robot = URDFParser().parse(str(URDF), floating_base=True)
    assert (robot.get_num_pos(), robot.get_num_vel()) == (7, 6)
    ref = RBDReference(robot)
    for q, v, force in cases():
        expected = acceleration(q, v, force)
        bias = ref.inverse_dynamics(q, v)[0]
        actual = np.linalg.solve(ref.crba(q), force - bias)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
        # Wrench at O and wrench at COM give identical mechanical power.
        at_com = force[3:] - np.cross(COM, force[:3])
        v_com = v[:3] + np.cross(v[3:], COM)
        np.testing.assert_allclose(force @ v, force[:3] @ v_com + at_com @ v[3:], atol=1e-14)
    # Torque-free angular momentum equation is independent of gravity.
    q, v, _ = cases()[0]
    a = acceleration(q, v, np.zeros(6))
    np.testing.assert_allclose(INERTIA @ a[3:] + np.cross(v[3:], INERTIA @ v[3:]), 0, atol=1e-15)
    assert MASS > 0 and np.max(np.diag(INERTIA)) < np.sum(np.diag(INERTIA))/2
    g = rotor_map()
    np.testing.assert_array_equal(g[:2], 0)
    np.testing.assert_array_equal(g[2], 1)
    np.testing.assert_allclose(g[3:5], [[.19,.19,-.19,-.19],[-.24,.24,.24,-.24]])
    assert np.linalg.matrix_rank(g) == 4


@pytest.mark.parametrize('semi', [False, True], ids=['euler', 'semi-implicit'])
def test_free_body_discrete_first_order(semi):
    ref = RBDReference(URDFParser().parse(str(URDF), floating_base=True))
    kind = 'semi_implicit_euler' if semi else 'euler'
    for q, v, force in cases():
        for dt in (.003, .02):
            np.testing.assert_allclose(ref.plant_step(q,v,force,dt,kind), step(q,v,force,dt,semi), rtol=2e-12,atol=2e-12)
            actual = ref.plant_step_gradient(q,v,force,dt,kind)
            coarse = step_jacobian(q,v,force,dt,semi,epsilon=3e-4)
            fine = step_jacobian(q,v,force,dt,semi,epsilon=3e-5)
            np.testing.assert_allclose(actual, fine, rtol=2e-7, atol=2e-9)
            assert np.max(abs(fine-coarse)) < 3e-8

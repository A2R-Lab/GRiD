"""The floating base quaternion is normalised inside the generated code by default.

`assume_unit_quaternions=True` compiles the normalisation out. The two builds agree on a
unit quaternion. On a scaled quaternion the default build returns what it returns for the
unit one, and the build that assumes unit input does not.
"""
import numpy as np
import pytest

import grid_rbd
from grid_assets import robot_urdf

pytestmark = pytest.mark.python_wrappers

ALGORITHMS = ["inverse_dynamics", "forward_dynamics", "inverse_dynamics_gradient"]


def register(unit):
    return grid_rbd.register_robot(
        f"unit_quaternion_{unit}", str(robot_urdf("go2")), floating_base=True,
        algorithm_list=ALGORITHMS, enable_mujoco_kernels=False, max_batch_size=4,
        assume_unit_quaternions=unit)


def outputs(handle, q, v, u):
    return (handle.inverse_dynamics(q, v, u), handle.forward_dynamics(q, v, u),
            handle.inverse_dynamics_gradient(q, v, u))


def test_normalisation_is_inside_unless_the_caller_assumes_unit_input():
    default, assumed = register(False), register(True)
    assert not default.assume_unit_quaternions and assumed.assume_unit_quaternions
    assert default._so_path != assumed._so_path
    rng = np.random.default_rng(1119)
    q = rng.uniform(-.5, .5, (4, default.num_joints)).astype(np.float32)
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    v, u = [rng.uniform(-.5, .5, (4, default.num_vel)).astype(np.float32) for _ in range(2)]
    scaled = q.copy()
    scaled[:, 3:7] *= np.float32(1.5)

    unit = outputs(default, q, v, u)
    for actual, expected in zip(outputs(assumed, q, v, u), unit):
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    for actual, expected in zip(outputs(default, scaled, v, u), unit):
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    drift = max(np.max(np.abs(np.asarray(a) - np.asarray(e)))
                for a, e in zip(outputs(assumed, scaled, v, u), unit))
    assert drift > 1e-1

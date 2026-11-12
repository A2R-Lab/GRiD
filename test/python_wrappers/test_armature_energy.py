"""Total energy includes rotor inertia; the inertial regressor stays 10/body."""
import contextlib
import io

import numpy as np
import pytest

import grid_rbd
from grid_assets import robot_urdf
from URDFParser import URDFParser
from RBDReference import RBDReference

pytestmark = pytest.mark.python_wrappers


@pytest.fixture(scope="module", params=[("iiwa14", False), ("go2", True), ("fr3", False)],
                ids=["iiwa14-fixed", "go2-floating", "fr3-mimic"])
def model(request):
    name, floating = request.param
    path = str(robot_urdf(name))
    with contextlib.redirect_stdout(io.StringIO()):
        ref = RBDReference(URDFParser().parse(path, floating_base=floating))
    h = grid_rbd.register_robot(
        name=f"armature_energy_{name}", urdf_path=path, floating_base=floating,
        runtime_armature=True, algorithm_list=["energy", "kinetic_energy_regressor"],
        enable_mujoco_kernels=False, max_batch_size=3,
    )
    yield h, ref, floating
    h.close()


@pytest.mark.parametrize("threads", [1, 32, 128])
def test_total_energy_and_inertial_regressor_offset(model, threads):
    h, ref, floating = model
    h.set_threads_per_block(threads)
    rng = np.random.default_rng(17)
    q = rng.uniform(-.6, .6, (3, h.num_joints)).astype(np.float32)
    v = rng.uniform(-1.0, 1.0, (3, h.num_vel)).astype(np.float32)
    if floating:
        q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    a = np.zeros(h.num_vel, dtype=np.float32)
    slots = h._meta["armature_slots"]
    a[slots] = np.linspace(.2, .6, len(slots))
    zero = np.zeros_like(a)
    h.set_armature(zero)
    plain_energy = h.energy(q, v)
    plain_y = h.kinetic_energy_regressor(q, v)
    h.set_armature(a)
    ref.set_armature(a)
    try:
        actual = h.energy(q, v)
        y = h.kinetic_energy_regressor(q, v)
        assert y.shape == (3, 10 * h.num_bodies)
        np.testing.assert_array_equal(y, plain_y)
        rotor = .5 * np.sum(a.astype(np.float64) * v.astype(np.float64)**2, axis=1)
        np.testing.assert_allclose(actual[:, 0] - plain_energy[:, 0], rotor, rtol=2e-4, atol=2e-4)
        np.testing.assert_array_equal(actual[:, 1], plain_energy[:, 1])
        pi = np.asarray(ref.robot.get_inertia_params_ordered_by_id()[1:]).reshape(-1)
        assert pi.shape == (10 * h.num_bodies,)
        for i in range(3):
            ke = ref.kinetic_energy(q[i].astype(float), v[i].astype(float))
            pe = ref.potential_energy(q[i].astype(float))
            expected = np.array([ke, pe, ke + pe])
            np.testing.assert_allclose(actual[i], expected, rtol=2e-4,
                                       atol=2e-4 * max(1.0, np.abs(expected).max()))
            np.testing.assert_allclose(y[i] @ pi + rotor[i], ke, rtol=2e-4, atol=2e-4)
    finally:
        h.set_armature(zero)
        ref.set_armature(zero)

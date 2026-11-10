"""Armature on the public floating/default-registry path, including MJX twins."""
import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

import grid_rbd
from grid_assets import robot_urdf
from grid_codegen.armature import supported_slots
from RBDReference import RBDReference
from URDFParser import URDFParser

pytestmark = pytest.mark.python_wrappers


@pytest.mark.parametrize("runtime", [False, True], ids=["baked", "runtime"])
def test_floating_default_registration(runtime):
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(robot_urdf("go2")), floating_base=True)
    armature = np.zeros(robot.get_num_vel(), dtype=np.float32)
    armature[supported_slots(robot)] = .25
    ref = RBDReference(robot)
    ref.set_armature(armature)
    # Deliberately omit algorithm_list and enable_mujoco_kernels: this is the
    # registration most users call, not a hand-selected compile subset.
    h = grid_rbd.register_robot(
        f"armature_default_{runtime}", str(robot_urdf("go2")),
        floating_base=True, armature=armature, runtime_armature=runtime,
        max_batch_size=1,
    )
    try:
        rng = np.random.default_rng(194)
        q = rng.uniform(-.2, .2, (1, h.num_joints)).astype(np.float32)
        q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
        v, u = [rng.uniform(-.2, .2, (1, h.num_vel)).astype(np.float32)
                for _ in range(2)]
        np.testing.assert_array_equal(h.armature, armature)
        actual = h.forward_dynamics(q, v, u)[0]
        expected = ref.forward_dynamics(*(x[0].astype(float) for x in (q, v, u)))
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
        header = (Path(h._so_path).parent / "grid.cuh").read_text()
        assert "integrator_mjx" in header
        # All three overloaded native launches used to bypass the clamp.
        # Exercise both qdd overloads and the direct JAX/Torch entry points.
        def outputs(handle, convert):
            qr, vr, ur = [convert(x) for x in (q, v, u)]
            return (handle.inverse_dynamics(qr, vr),
                    handle.inverse_dynamics(qr, vr, ur),
                    handle.inverse_dynamics_gradient(qr, vr),
                    handle.inverse_dynamics_gradient(qr, vr, ur),
                    handle.forward_dynamics_gradient(qr, vr, ur))
        h.set_threads_per_block(32)
        baseline = outputs(h, lambda x: x)
        h.set_threads_per_block(1024)
        for actual, expected in zip(outputs(h, lambda x: x), baseline):
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
        import jax.numpy as jnp
        import torch
        from grid_rbd.jax import JaxRobotHandle
        from grid_rbd.torch import TorchRobotHandle
        key = Path(h._so_path).parent.name
        handles = [(JaxRobotHandle(h, key, h._so_path), jnp.asarray, np.asarray),
                   (TorchRobotHandle(h, key, h._so_path),
                    lambda x: torch.as_tensor(x, device="cuda"),
                    lambda x: x.cpu().numpy())]
        for handle, convert, download in handles:
            for actual, expected in zip(outputs(handle, convert), baseline):
                np.testing.assert_allclose(download(actual), expected, rtol=2e-4, atol=2e-4)
    finally:
        h.close()

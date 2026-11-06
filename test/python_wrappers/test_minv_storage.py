"""CRBA-inverse paths must preserve the public symmetric-upper raw contract."""
import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

import grid_rbd
from grid_assets import robot_urdf
from URDFParser import URDFParser
from RBDReference import RBDReference

pytestmark = pytest.mark.python_wrappers


@pytest.mark.parametrize("model", ["fr3", "mixed_spherical_arm"])
@pytest.mark.parametrize("dtype", ["float32", pytest.param("float64", marks=pytest.mark.fp64_alpha)])
def test_native_and_three_backend_inverse_mass(model, dtype):
    import jax
    import jax.numpy as jnp
    import torch
    from grid_rbd.jax import JaxRobotHandle
    from grid_rbd.torch import TorchRobotHandle
    path = (robot_urdf(model) if model == "fr3" else
            Path(__file__).resolve().parents[2] / "external/URDFParser/tests/fixtures/mixed_spherical_arm.urdf")
    with contextlib.redirect_stdout(io.StringIO()):
        ref = RBDReference(URDFParser().parse(str(path)))
    h = grid_rbd.register_robot(f"minv_storage_{model}_{dtype}", str(path), dtype=dtype,
                               algorithm_list=["minv", "crba"],
                               enable_mujoco_kernels=False, max_batch_size=3)
    previous_x64 = jax.config.x64_enabled
    try:
        jax.config.update("jax_enable_x64", dtype == "float64")
        rng = np.random.default_rng(31)
        q = rng.uniform(-.5, .5, (3, h.num_joints)).astype(dtype)
        if model == "mixed_spherical_arm":
            q[:, 1:5] /= np.linalg.norm(q[:, 1:5], axis=1, keepdims=True)
        mass = np.stack([ref.crba(row.astype(float)) for row in q])
        expected = np.linalg.inv(mass)
        tol = 3e-4 if dtype == "float32" else 1e-10
        key = Path(h._so_path).parent.name
        th = TorchRobotHandle(h, key, h._so_path)
        jh = JaxRobotHandle(h, key, h._so_path)
        # Direct FFI launches honor explicit overrides; unlike the native host
        # wrapper they do not clamp them. Exercise the actual compiled ceiling.
        ceiling = h.kernel_max_threads("minv")
        assert ceiling >= 32
        for threads in sorted({1, 32, min(128, ceiling)}):
            h.set_threads_per_block(threads)
            raw = h._runner.minv(q).reshape(3, h.num_vel, h.num_vel).swapaxes(-1, -2)
            np.testing.assert_array_equal(np.tril(raw, -1), 0)
            np.testing.assert_allclose(raw, np.triu(expected), rtol=tol, atol=tol)
            results = [h.minv(q), np.asarray(jh.minv(jnp.asarray(q))),
                       th.minv(torch.as_tensor(q, device="cuda")).cpu().numpy()]
            for actual in results:
                np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
                np.testing.assert_allclose(mass @ actual, np.broadcast_to(np.eye(h.num_vel), actual.shape),
                                           rtol=tol, atol=tol)
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
        h.close()

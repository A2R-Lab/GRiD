"""Deferred GPU regression for the hand-written NQ/NV JAX packers.

Distinct rows, nonzero qdd and B>1 expose the old NQ-wide source pitch. Compare
against the NumPy C ABI, whose packing already respects NV. This file is part
of the GPU receipt; do not run during another project's timing window.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.python_wrappers


@pytest.fixture(scope="module", params=["go2", "g1"])
def handles(request):
    import grid_rbd
    from config import robot_urdf
    options = dict(floating_base=True, enable_mujoco_kernels=False, max_batch_size=8,
                   algorithm_list=["idsva_so_body_frame", "integrator", "integrator_gradient"])
    name = "nv_packing_" + request.param
    plain = grid_rbd.register_robot(name + "_numpy", str(robot_urdf(request.param)), **options)
    try:
        jax = grid_rbd.register_robot(name + "_jax", str(robot_urdf(request.param)), backend="jax", **options)
        try:
            yield plain, jax
        finally:
            jax.close()
    finally:
        plain.close()


def inputs(handle, batch):
    nq, nv = handle.num_joints, handle.num_vel
    assert nq == nv + 1
    rng = np.random.default_rng(927)
    q = rng.uniform(-.1, .1, (batch, nq)).astype(np.float32)
    q[:, 6] += 1
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    v = rng.uniform(-.1, .1, (batch, nv)).astype(np.float32)
    a = rng.uniform(-.2, .2, (batch, nv)).astype(np.float32)
    return q, v, a


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_floating_idsva_so_pack_matches_numpy(handles, batch, compiled):
    import jax
    import jax.numpy as jnp
    plain, device = handles
    args = inputs(plain, batch)
    fn = jax.jit(device.idsva_so) if compiled else device.idsva_so
    actual = fn(*(jnp.asarray(x) for x in args))
    expected = plain.idsva_so(*args)
    assert len(actual) == len(expected) == 4
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(a), b, rtol=2e-4, atol=1e-3)


@pytest.mark.parametrize("operation", ["integrator", "integrator_gradient"])
@pytest.mark.parametrize("scheme", ["euler", "semi_implicit_euler", "midpoint",
                                    "rk4", "trapezoidal", "constant_acceleration"])
def test_floating_integrator_pack_matches_numpy(handles, operation, scheme):
    import jax
    import jax.numpy as jnp
    plain, device = handles
    args = inputs(plain, 3)
    expected = getattr(plain, operation)(*args, .001, integrator_type=scheme)
    fn = jax.jit(lambda q, v, u: getattr(device, operation)(q, v, u, .001, integrator_type=scheme))
    actual = fn(*(jnp.asarray(x) for x in args))
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-4, atol=1e-3)

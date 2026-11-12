"""Standalone ABA with a skew scalar joint after a spherical block."""
import contextlib
import io

import numpy as np
import pytest

import grid_rbd
from RBDReference import RBDReference
from URDFParser import URDFParser
from test.test_armature_mixed_spherical import mixed_skew_path

pytestmark = pytest.mark.python_wrappers


@pytest.mark.parametrize("runtime", [False, True], ids=["baked", "runtime"])
def test_aba_mixed_spherical_skew_matches_reference(tmp_path, runtime):
    path = mixed_skew_path(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        ref = RBDReference(URDFParser().parse(str(path)))
    armature = np.array([.1, 0, 0, 0, .4], dtype=np.float32)
    ref.set_armature(armature)
    h = grid_rbd.register_robot(
        f"aba_mixed_skew_{runtime}", str(path), armature=armature,
        runtime_armature=runtime, algorithm_list=["aba"],
        enable_mujoco_kernels=False, max_batch_size=4,
    )
    try:
        rng = np.random.default_rng(1914)
        q = rng.uniform(-.4, .4, (4, h.num_joints)).astype(np.float32)
        q[:, 1:5] /= np.linalg.norm(q[:, 1:5], axis=1, keepdims=True)
        v, u = [rng.uniform(-.4, .4, (4, h.num_vel)).astype(np.float32)
                for _ in range(2)]
        expected = np.stack([ref.aba(qr.astype(float), vr.astype(float), ur.astype(float))
                             for qr, vr, ur in zip(q, v, u)])
        for threads in (1, 32, 128):
            h.set_threads_per_block(threads)
            np.testing.assert_allclose(h.aba(q, v, u), expected, rtol=2e-4, atol=2e-4)
    finally:
        h.close()

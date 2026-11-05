"""Native context ownership and rejected-update atomicity for armature."""
import numpy as np
import pytest

import grid_rbd
from grid_assets import robot_urdf

pytestmark = pytest.mark.python_wrappers


@pytest.fixture(scope="module", params=[False, True], ids=["fixed", "floating"])
def owner(request):
    h = grid_rbd.register_robot(
        name=f"armature_owner_{request.param}", urdf_path=str(robot_urdf("iiwa14")),
        floating_base=request.param, runtime_armature=True,
        algorithm_list=["minv"], enable_mujoco_kernels=False, max_batch_size=2,
    )
    yield h
    h.close()


def test_independent_contexts_and_shared_alias(owner):
    a = np.zeros(owner.num_vel, dtype=np.float32)
    a[owner._meta["armature_slots"]] = .2
    zero = np.zeros_like(a)
    one, two = owner.context(), owner.context()
    alias = grid_rbd.get_robot(owner.name)
    try:
        owner.set_armature(a)
        np.testing.assert_array_equal(alias.armature, a)
        np.testing.assert_array_equal(one.armature, zero)
        one.set_armature(a * 2)
        np.testing.assert_array_equal(one.armature, a * 2)
        np.testing.assert_array_equal(two.armature, zero)
        # Bypass the public Python setter: readback must still observe native data.
        owner._runner.set_armature_params(a * 3)
        np.testing.assert_array_equal(alias.armature, a * 3)
        created_after_mutation = owner.context()
        try:
            np.testing.assert_array_equal(created_after_mutation.armature, zero)
        finally:
            created_after_mutation.close()
        # Real backend aliases share the native owner despite distinct handles.
        for backend in ("jax", "torch"):
            surface = grid_rbd.register_robot(
                owner.name, str(robot_urdf("iiwa14")), backend=backend,
                floating_base=owner.floating_base, runtime_armature=True,
                algorithm_list=["minv"], enable_mujoco_kernels=False, max_batch_size=2,
            )
            try:
                np.testing.assert_array_equal(surface.armature, a * 3)
                surface.set_armature(a * 4)
                np.testing.assert_array_equal(alias.armature, a * 4)
                surface.set_armature(a * 3)
            finally:
                surface.close()
    finally:
        owner.set_armature(zero)
        alias.close()
        one.close()
        two.close()
    for closed in (one, two):
        with pytest.raises(RuntimeError, match="closed"):
            _ = closed.armature


def test_invalid_native_and_python_updates_preserve_epoch(owner):
    good = np.zeros(owner.num_vel, dtype=np.float32)
    slot = owner._meta["armature_slots"][0]
    good[slot] = .25
    owner.set_armature(good)
    try:
        version = owner.model_version
        for badval in (-.1, -1e-100, np.nan, np.inf, 1e100):
            bad = good.astype(np.float64)
            bad[slot] = badval
            for setter in (owner.set_armature, owner._runner.set_armature_params):
                with pytest.raises((ValueError, RuntimeError)):
                    setter(bad)
                assert owner.model_version == version
                np.testing.assert_array_equal(owner.armature, good)
        if owner.num_vel > len(owner._meta["armature_slots"]):
            bad = good.astype(np.float64)
            bad[0] = 1e-100  # must not vanish before the unsupported-slot check
            for setter in (owner.set_armature, owner._runner.set_armature_params):
                with pytest.raises((ValueError, RuntimeError)):
                    setter(bad)
                assert owner.model_version == version
                np.testing.assert_array_equal(owner.armature, good)
        owner.set_armature(good * 2)
        assert owner.model_version > version
    finally:
        owner.set_armature(np.zeros_like(good))


def test_parameter_update_changes_only_own_dynamics(owner):
    one, two = owner.context(), owner.context()
    q = np.zeros((1, owner.num_joints), dtype=np.float32)
    if owner.floating_base:
        q[0, 6] = 1  # xyzw unit quaternion
    try:
        baseline = two.minv(q)
        np.testing.assert_array_equal(one.minv(q), baseline)
        a = np.zeros(owner.num_vel, dtype=np.float32)
        a[owner._meta["armature_slots"]] = .25
        one.set_armature(a)
        changed = one.minv(q)
        assert not np.array_equal(changed, baseline)
        np.testing.assert_array_equal(two.minv(q), baseline)
        expected = np.linalg.inv(np.linalg.inv(baseline[0].astype(np.float64)) + np.diag(a))
        np.testing.assert_allclose(changed[0], expected, rtol=2e-4,
                                   atol=2e-4 * max(1.0, np.abs(expected).max()))
    finally:
        one.close()
        two.close()

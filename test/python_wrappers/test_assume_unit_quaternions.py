"""A quaternion is normalised inside the generated code by default.

`assume_unit_quaternions=True` compiles the normalisation out. The two builds agree on a
unit quaternion. On a scaled quaternion the default build returns what it returns for the
unit one, and the build that assumes unit input does not. This holds for the floating
base in both output conventions and for a spherical joint, through a registered handle
and through a consumer artifact.
"""
import ctypes
from pathlib import Path

import numpy as np
import pytest

import grid_rbd
import external.URDFParser as parser_package
from grid_assets import robot_urdf

pytestmark = pytest.mark.python_wrappers

SPHERICAL_URDF = Path(parser_package.__file__).parent / "tests/fixtures/spherical_arm.urdf"
# name: (urdf, floating base, MuJoCo twins, first quaternion coordinate, algorithms)
CASES = {
    "floating": (robot_urdf("go2"), True, False, 3,
                 ["inverse_dynamics", "forward_dynamics", "inverse_dynamics_gradient"]),
    "floating_mujoco": (robot_urdf("go2"), True, True, 3, ["inverse_dynamics", "forward_dynamics"]),
    "spherical": (SPHERICAL_URDF, False, False, 0, ["inverse_dynamics", "forward_dynamics"]),
}


def register(case, unit):
    urdf, floating, mujoco, _, algorithms = CASES[case]
    return grid_rbd.register_robot(
        f"unit_quaternion_{case}_{unit}", str(urdf), floating_base=floating,
        algorithm_list=algorithms, enable_mujoco_kernels=mujoco, max_batch_size=4,
        assume_unit_quaternions=unit)


def outputs(handle, case, q, v, u):
    surface = handle.mujoco if CASES[case][2] else handle
    return [getattr(surface, name)(q, v, u) for name in CASES[case][4]]


@pytest.mark.parametrize("case", list(CASES))
def test_normalisation_is_inside_unless_the_caller_assumes_unit_input(case):
    default, assumed = register(case, False), register(case, True)
    try:
        assert not default.assume_unit_quaternions and assumed.assume_unit_quaternions
        assert default._so_path != assumed._so_path
        first = CASES[case][3]
        rng = np.random.default_rng(1119)
        q = rng.uniform(-.5, .5, (4, default.num_joints)).astype(np.float32)
        q[:, first:first + 4] /= np.linalg.norm(q[:, first:first + 4], axis=1, keepdims=True)
        v, u = [rng.uniform(-.5, .5, (4, default.num_vel)).astype(np.float32) for _ in range(2)]
        scaled = q.copy()
        scaled[:, first:first + 4] *= np.float32(1.5)

        unit = outputs(default, case, q, v, u)
        for actual, expected in zip(outputs(assumed, case, q, v, u), unit):
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
        for actual, expected in zip(outputs(default, case, scaled, v, u), unit):
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
        drift = max(np.max(np.abs(np.asarray(a) - np.asarray(e)))
                    for a, e in zip(outputs(assumed, case, scaled, v, u), unit))
        assert drift > 1e-1
    finally:
        default.close()
        assumed.close()


@pytest.mark.parametrize("encoded", [False, True])
def test_artifact_route_carries_the_option_for_a_spherical_joint(tmp_path, encoded):
    """The option reaches a consumer artifact and separates its key, also when the joint
    type is written with a character reference, which a search of the raw text misses."""
    from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
    text = SPHERICAL_URDF.read_text()
    if encoded:
        assert 'type="spherical"' in text
        text = text.replace('type="spherical"', 'type="spher&#105;cal"')
    urdf = tmp_path / "model.urdf"
    urdf.write_text(text)
    source = tmp_path / "norm.cu"
    source.write_text('#include "grid.cuh"\n'
                      'extern "C" __attribute__((visibility("default"))) float squared_norm() '
                      '{ return grid::grid_quat_norm2<float>(0.f, 0.f, 0.f, 2.f); }\n')
    options = dict(sources=[source], cache_dir=tmp_path / "cache", cuda_arch=int(_detect_cuda_arch()),
                   algorithm_list=["end_effector_pose"], enable_mujoco_kernels=False)
    built = {unit: grid_rbd.build_artifact("unit", f"probe_{unit}", urdf, assume_unit_quaternions=unit, **options)
             for unit in (False, True)}
    assert built[False].content_key != built[True].content_key
    for unit, expected in ((False, 4.0), (True, 1.0)):
        library = ctypes.CDLL(str(built[unit].path))
        library.squared_norm.restype = ctypes.c_float
        assert library.squared_norm() == expected

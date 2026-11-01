"""Constant follower FK/Jacobian: block, thread and warp, both precisions."""
from pathlib import Path
import subprocess
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from URDFParser import URDFParser
from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


@pytest.mark.parametrize('kind', ['prismatic', 'revolute'])
@pytest.mark.parametrize('precision', ['float', pytest.param('double', marks=pytest.mark.fp64_alpha)])
def test_constant_mimic_kinematics(tmp_path, kind, precision):
    urdf = tmp_path / 'constant.urdf'
    urdf.write_text(robot_urdf('slider_mimic_fixed_tool').read_text()
                    .replace('type="prismatic"', f'type="{kind}"')
                    .replace('mimic joint="slide"', 'mimic joint="tool"'))
    robot = URDFParser().parse(str(urdf))
    gen = GRiDCodeGenerator(robot, False, False, 'grid')
    gen.gen_all_code(output_path=str(tmp_path / 'grid.cuh'),
                     algorithm_list=['end_effector_pose', 'end_effector_pose_gradient'],
                     include_homogenous_transforms=True, fixed_target_name='tool',
                     enable_mujoco_kernels=False)
    runner = Path(__file__).with_name('cuda_prismatic_fixed_target_runner.cu')
    flags = ['-std=c++17', '-O2', f'-arch=sm_{_detect_cuda_arch()}'] + (['-DUSE_DOUBLE'] if precision == 'double' else [])
    exe, _ = cached_nvcc_executable([runner], flags, exe_name='constant.exe', fallback_dir=tmp_path,
                                    include_dirs=[tmp_path], what='constant mimic runner')
    axis = np.array([.6, 0., .8])
    origin = np.eye(4)
    origin[:3, :3] = Rotation.from_euler('xyz', [.3, .2, .1]).as_matrix()
    origin[:3, 3] = [.04, .01, .02]

    def motion(angle):
        T = np.eye(4)
        if kind == 'prismatic':
            T[:3, 3] = axis * angle
        else:
            T[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        return T

    def expected(q):
        body = origin @ motion(q) @ motion(.17)
        tool = body @ origin
        pose = np.r_[tool[:3, 3], Rotation.from_matrix(tool[:3, :3]).as_euler('xyz')]
        quat = Rotation.from_matrix(body[:3, :3]).as_quat()
        return pose, np.r_[body[:3, 3], quat[3], quat[:3]]

    tol = 2e-5 if precision == 'float' else 1e-9
    for q in (0., .13, .5):
        pose, frame = expected(q)
        # Independent transform finite difference, not the reference/generator.
        h = 1e-6
        jac = (expected(q+h)[0] - expected(q-h)[0])/(2*h)
        baseline = None
        for threads in (1, 17, 32, 64, 256):
            result = subprocess.run([str(exe), str(threads), str(q)], capture_output=True, text=True)
            assert result.returncode == 0, result.stdout + result.stderr
            values = np.fromstring(result.stdout.strip().splitlines()[-1], sep=' ')
            assert values.size == (26 if threads >= 32 else 19)
            np.testing.assert_allclose(values[:12], np.r_[pose, jac], atol=tol, rtol=tol)
            np.testing.assert_allclose(values[12:19], frame, atol=tol, rtol=tol)
            if threads >= 32:
                np.testing.assert_allclose(values[19:26], frame, atol=tol, rtol=tol)
            if baseline is not None:
                np.testing.assert_array_equal(values[:19], baseline)
            baseline = values[:19]

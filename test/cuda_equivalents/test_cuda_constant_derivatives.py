"""Constant-joint derivatives, including genuinely empty coordinate buffers."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


def constant_model(directory, zero=False, locked=False, branched=False):
    inertial = ('<inertial><origin xyz=".03 -.02 .07"/><mass value="1"/>'
                '<inertia ixx=".3" iyy=".4" izz=".5" ixy="0" ixz="0" iyz="0"/></inertial>')
    links = ''.join(f'<link name="{name}">{inertial}</link>'
                    for name in ('base', 'mount', 'arm', 'follower'))
    drive = ('<joint name="drive" type="fixed"><parent link="mount"/><child link="arm"/>'
             '<origin xyz=".1 0 .2"/></joint>' if zero else
             '<joint name="drive" type="revolute"><parent link="mount"/><child link="arm"/>'
             '<origin xyz=".1 0 .2"/><axis xyz="0 1 0"/>'
             '<limit lower="-3" upper="3" effort="10" velocity="10"/></joint>')
    follower = ('<joint name="constant" type="fixed"><parent link="arm"/><child link="follower"/>'
                '<origin xyz=".2 0 .3" rpy="0 0 .17"/></joint>' if locked else
                '<joint name="constant" type="revolute"><parent link="arm"/><child link="follower"/>'
                '<origin xyz=".2 0 .3"/><axis xyz="0 0 1"/>'
                '<limit lower="-3" upper="3" effort="10" velocity="10"/>'
                '<mimic joint="fixed" multiplier="2" offset=".17"/></joint>')
    if branched:
        follower = follower.replace('<parent link="arm"/>', '<parent link="mount"/>')
    path = directory / f'model-{zero}-{locked}-{branched}.urdf'
    path.write_text(f'<robot name="constant">{links}'
                   '<joint name="fixed" type="fixed"><parent link="base"/><child link="mount"/></joint>'
                   f'{drive}{follower}</robot>')
    return URDFParser().parse(str(path))


@pytest.fixture(scope='module', params=[(z, p) for z in ('chain', 'zero', 'branch') for p in ('float', 'double')])
def derivative_probe(request, tmp_path_factory):
    layout, precision = request.param
    zero = layout == 'zero'
    path = tmp_path_factory.mktemp(f'constant-derivatives-{layout}-{precision}')
    robot = constant_model(path, zero=zero, branched=layout == 'branch')
    gen = GRiDCodeGenerator(robot, dtype=precision)
    gen.gen_all_code(output_path=str(path / 'grid.cuh'), emit_plant=False,
                     algorithm_list=['inverse_dynamics_gradient', 'frame_jacobian', 'frame_jacobian_dot'],
                     include_homogenous_transforms=True, enable_mujoco_kernels=False)
    source = Path(__file__).with_name('cuda_constant_derivatives_runner.cu')
    exe = path / 'probe'
    nvcc = shutil.which('nvcc')
    assert nvcc, 'CUDA toolkit required'
    result = subprocess.run([nvcc, '-std=c++17', '-O2', f'-arch=sm_{_detect_cuda_arch()}',
                             f'-I{path}', f'-DDTYPE={precision}',
                             f'-DID_TEMP={gen.gen_inverse_dynamics_gradient_inner_temp_mem_size()}',
                             str(source), '-o', str(exe)], capture_output=True, text=True)
    (path / 'compile.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    return robot, RBDReference(robot), path, exe, precision, layout


@pytest.mark.parametrize('mode', [0, 1, 2], ids=['id-gradient', 'frame-jacobian', 'frame-jacobian-dot'])
def test_constant_derivatives(derivative_probe, mode):
    robot, ref, path, exe, precision, layout = derivative_probe
    zero = layout == 'zero'
    q = np.full(robot.get_num_pos(), .35)
    v, a = np.full(robot.get_num_vel(), .23), np.full(robot.get_num_vel(), -.31)
    if mode == 0:
        jac = ref.inverse_dynamics_gradient(q, v, a)
        expected = np.r_[jac[:, :len(v)].ravel(order='F'), jac[:, len(v):].ravel(order='F')]
        # A separately parsed rigidly locked model also checks force transmission.
        if not zero:
            locked = RBDReference(constant_model(path, locked=True, branched=layout == 'branch'))
            np.testing.assert_allclose(jac, locked.inverse_dynamics_gradient(q, v, a), atol=1e-11, rtol=1e-10)
    else:
        method = ref.frame_jacobian if mode == 1 else ref.frame_jacobian_dot
        args = (q,) if mode == 1 else (q, v)
        expected = method(*args, 'constant', reference_frame='LOCAL_WORLD_ALIGNED').ravel(order='F')
    tol = 3e-5 if precision == 'float' else 1e-11
    baseline = None
    for threads in (1, 17, 32, 64, 256):
        result = subprocess.run([str(exe), str(threads), str(mode)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        actual = np.fromstring(result.stdout, sep=' ')
        assert actual.size == expected.size + 2
        np.testing.assert_array_equal(actual[[0, -1]], [123, 456])
        np.testing.assert_allclose(actual[1:-1], expected, atol=tol, rtol=tol)
        if baseline is not None:
            np.testing.assert_array_equal(actual, baseline)
        baseline = actual
    if layout == 'branch':
        for threads in (1, 17, 32, 64, 256):
            # Internal gradient bands of a different, constant root branch must
            # stay exactly zero when the independent branch's qd/qdd are NaN.
            poison_mode = 3 if mode == 0 else mode
            result = subprocess.run([str(exe), str(threads), str(poison_mode), 'poison'],
                                    capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            values = np.fromstring(result.stdout, sep=' ')
            np.testing.assert_array_equal(values[[0, -1]], [123, 456])
            if mode == 0:
                bands = values[1:-1].reshape(6, robot.get_num_bodies(), 6)
                constant = robot.get_joint_by_name('constant').jid
                np.testing.assert_array_equal(bands[:, constant], 0)
                drive = robot.get_joint_by_name('drive').jid
                assert np.isnan(bands[:, drive]).any(), 'Do not sanitize real NaN inputs'
            else:
                np.testing.assert_array_equal(values[1:-1], 0)

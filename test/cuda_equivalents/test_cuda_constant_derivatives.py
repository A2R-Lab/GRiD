"""Constant-joint derivatives, including genuinely empty coordinate buffers."""
from pathlib import Path
import subprocess

import numpy as np
import pytest
from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.constant_model import constant_model
from test.cuda_equivalents.executable_cache import cached_nvcc_executable
from test.cuda_equivalents.thread_invariance import thread_sweep

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


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
    exe, _ = cached_nvcc_executable([source], ['-std=c++17', '-O2', f'-arch=sm_{_detect_cuda_arch()}',
                                              f'-DDTYPE={precision}',
                                              f'-DID_TEMP={gen.gen_inverse_dynamics_gradient_inner_temp_mem_size()}'],
                                    exe_name='probe', fallback_dir=path, include_dirs=[path],
                                    what='constant derivatives runner')
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
    def run(threads):
        result = subprocess.run([str(exe), str(threads), str(mode)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return np.fromstring(result.stdout, sep=' ')

    def check(actual, _):
        assert actual.size == expected.size + 2
        np.testing.assert_array_equal(actual[[0, -1]], [123, 456])
        np.testing.assert_allclose(actual[1:-1], expected, atol=tol, rtol=tol)

    thread_sweep((1, 17, 32, 64, 256), run, check)
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

"""Constant-body second-order folds and Coriolis projection, including nv=0."""
from pathlib import Path
import subprocess

import numpy as np
import pytest
from RBDReference import RBDReference
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable
from test.cuda_equivalents.constant_model import constant_model

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


@pytest.fixture(scope='module', params=[(z, p) for z in (False, True) for p in ('float', 'double')])
def second_probe(request, tmp_path_factory):
    zero, precision = request.param
    path = tmp_path_factory.mktemp(f'constant-second-{zero}-{precision}')
    robot = constant_model(path, zero=zero)
    gen = GRiDCodeGenerator(robot, dtype=precision)
    gen.gen_all_code(output_path=str(path / 'grid.cuh'), emit_plant=False,
                     algorithm_list=['idsva_so_body_frame', 'idsva_so_world_frame', 'coriolis_matrix'],
                     enable_idsva_so_world_frame=True, enable_idsva_so_body_frame=True,
                     enable_mujoco_kernels=False)
    exe, _ = cached_nvcc_executable([Path(__file__).with_name('cuda_constant_second_order_runner.cu')], [
        '-std=c++17', '-O2', f'-arch=sm_{_detect_cuda_arch()}', f'-DDTYPE={precision}',
        f'-DBODY_TEMP={gen.gen_idsva_so_body_frame_inner_temp_mem_size()}',
        f'-DWORLD_TEMP={gen.gen_idsva_so_world_frame_temp_mem_size()}'],
        exe_name='probe', fallback_dir=path, include_dirs=[path], what='constant second order runner')
    return robot, path, exe, precision, zero


@pytest.mark.parametrize('mode', [0, 1, 2], ids=['body', 'world', 'coriolis'])
def test_constant_second_order(second_probe, mode):
    robot, path, exe, precision, zero = second_probe
    ref = RBDReference(robot)
    q, v, a = [np.full(robot.get_num_vel(), x) for x in (.35, .23, -.31)]
    method = ['idsva_so_body_frame', 'idsva_so_world_frame', 'coriolis_matrix'][mode]
    args = (q, v) if mode == 2 else (q, v, a)
    parts = getattr(ref, method)(*args)
    expected = np.asarray(parts).ravel()
    if not zero:
        locked = RBDReference(constant_model(path, locked=True))
        np.testing.assert_allclose(parts, getattr(locked, method)(*args), atol=1e-11, rtol=1e-10)
    tol = 5e-5 if precision == 'float' else 1e-10
    baseline = None
    for threads in (1, 17, 32, 64, 256):
        result = subprocess.run([str(exe), str(threads), str(mode)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        actual = np.fromstring(result.stdout, sep=' ')
        assert actual.size == expected.size+2
        np.testing.assert_array_equal(actual[[0, -1]], [123, 456])
        np.testing.assert_allclose(actual[1:-1], expected, atol=tol, rtol=tol)
        if baseline is not None:
            np.testing.assert_array_equal(actual, baseline)
        baseline = actual

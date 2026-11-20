"""Compile registration on the CPU; separately authorized CUDA correctness.

The compile test never loads CUDA or runs the resulting executable. Both public
registration surfaces instantiate ordinary and bench-only twins for float/double.
"""
import contextlib
import io
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pytest
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.test_cuda_contact_frame_positions import _robot, _q_for, _retract
from RBDReference import RBDReference

RUNNER = Path(__file__).with_name('cuda_multitarget_registration_runner.cu')


def build(directory, scalar, *, compile_only=False):
    robot = _robot('go2', True)
    targets = [dict(anchor_jid=int(j), offset=(0.03, -0.02, 0.05)) for j in robot.get_leaf_nodes()]
    with contextlib.redirect_stdout(io.StringIO()):
        GRiDCodeGenerator(robot, FILE_NAMESPACE='grid').gen_all_code(
            algorithm_list=['end_effector_pose'], multi_target_batch=targets,
            enable_mujoco_kernels=False, output_path=str(directory / 'grid.cuh'))
    exe = directory / 'registration'
    # The standalone compiler check never queries a GPU; runtime coverage uses
    # the same optimization and native architecture as its multi-target sibling.
    compiler = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    if compile_only:
        flags = ['-O1', '-arch=compute_80']
    else:
        from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
        arch = _detect_cuda_arch()
        flags = ['-O2', '-gencode', f'arch=compute_{arch},code=sm_{arch}']
    command = [compiler, '-std=c++17', *flags,
               '-DSCALAR=' + scalar, '-I' + str(directory), str(RUNNER), '-o', str(exe)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return robot, targets, exe


@pytest.mark.cuda_equivalence
@pytest.mark.parametrize('scalar', ['float', 'double'])
def test_floating_registered_kernels_with_workspace(tmp_path, scalar):
    robot, targets, exe = build(tmp_path, scalar)
    q = _q_for(robot, True).astype(np.float32 if scalar == 'float' else np.float64)
    result = subprocess.run([str(exe)], input='\n'.join(map(str, q)), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = np.array([list(map(float, line.split())) for line in result.stdout.splitlines()])
    ref = RBDReference(robot)
    def positions(configuration):
        world, _ = ref._frame_world_placement_and_chain(configuration)
        return np.array([(world[t['anchor_jid']] @ np.array([*t['offset'], 1]))[:3] for t in targets])
    nq = robot.get_num_vel()
    expected = np.empty((len(targets), nq + 1, 3))
    expected[:, 0] = positions(q.astype(float))
    for k in range(nq):
        delta = np.zeros(nq); delta[k] = 1e-6
        expected[:, k+1] = (positions(_retract(q.astype(float), delta, True)) -
                            positions(_retract(q.astype(float), -delta, True))) / 2e-6
    # New float registration coverage; preserve the stricter existing double
    # multi-target position and gradient gates separately.
    actual = actual.reshape(len(targets), nq+1, 3)
    np.testing.assert_allclose(actual[:, 0], expected[:, 0],
                               rtol=2e-4 if scalar == 'float' else 1e-10,
                               atol=1e-5 if scalar == 'float' else 1e-10)
    np.testing.assert_allclose(actual[:, 1:], expected[:, 1:],
                               rtol=2e-4 if scalar == 'float' else 1e-5,
                               atol=1e-5 if scalar == 'float' else 1e-6)


if __name__ == '__main__':
    # Explicit opt-in compiler regression, kept outside pytest's compile-free
    # CPU lane. It never executes or loads the compiled CUDA program.
    import argparse
    import tempfile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-only', action='store_true', required=True)
    parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='grid-registration-compile-') as root:
        for scalar in ('float', 'double'):
            directory = Path(root)/scalar
            directory.mkdir()
            build(directory, scalar, compile_only=True)
            print(scalar + ': checked and unchecked registration compiled; not executed')

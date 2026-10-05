"""Complete bundled meshes -> bounded spheres -> explicit CLI -> CUDA vs CPU FK."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from config import robot_urdf
from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_codegen.algorithms._spherize import spherize_urdf
from grid_codegen.algorithms._collision import build_sphere_tiers
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable


@pytest.fixture(scope='module')
def complete_model(tmp_path_factory):
    from grid_codegen.cli import main
    directory = tmp_path_factory.mktemp('complete_collision')
    source = robot_urdf('iiwa14')
    spheres = spherize_urdf(source, .06, directory / 'spheres.urdf', mesh_mode='bounded-bulge')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, 'argv', ['grid-generate', str(source), '--spherized-urdf',
            str(spheres), '--algorithm-list', 'kinematics', '-o', str(directory / 'grid.cuh')])
        main()
    robot = URDFParser().parse(str(source))
    spec = build_sphere_tiers(robot, {'fine': spheres})['fine']
    arch = _detect_cuda_arch()
    root = Path(__file__).resolve().parents[2]
    exe, _ = cached_nvcc_executable(
        [Path(__file__).with_name('cuda_collision_complete_runner.cu'), directory / 'grid.cuh'],
        ['-std=c++17', '-O2', '-gencode', f'arch=compute_{arch},code=sm_{arch}'],
        exe_name='complete_collision.exe', fallback_dir=directory,
        include_dirs=[root / 'grid_codegen/collision'], what='complete collision model')
    return exe, robot, RBDReference(robot), spec


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize('threads', [1, 32, 100])
def test_complete_collision_model_matches_cpu(complete_model, threads):
    exe, robot, ref, spec = complete_model
    rng = np.random.default_rng(20261005)
    radius = np.asarray(spec['radius'])
    offsets = np.asarray(spec['offset']).reshape(-1, 3)
    for q in [np.zeros(robot.get_num_pos()), *rng.uniform(-2, 2, (5, robot.get_num_pos()))]:
        q = q.astype(np.float32).astype(float)
        world, _ = ref._frame_world_placement_and_chain(q)
        expected = np.array([(world[a] @ np.r_[p, 1])[:3]
                             for a, p in zip(spec['anchor'], offsets)])
        self_hit = any(np.sum((expected[i] - expected[j])**2) < (radius[i]+radius[j])**2
            for i, start, end in spec['self_cc_ranges'] for j in range(start, end+1))
        for obstacle in [np.array([1000., 1000., 1000., .1]), np.r_[expected[0], .1]]:
            obstacle = obstacle.astype(np.float32).astype(float)
            env_hit = np.any(np.sum((expected-obstacle[:3])**2, axis=1) < (radius+obstacle[3])**2)
            result = subprocess.run([str(exe), str(threads)], input=' '.join(map(str, np.r_[q, obstacle])),
                                    capture_output=True, text=True, check=True)
            lines = result.stdout.strip().splitlines()
            actual = np.fromstring(lines[-1], sep=' ').reshape(-1, 3)
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
            assert int(lines[-2]) == int(not (self_hit or env_hit))

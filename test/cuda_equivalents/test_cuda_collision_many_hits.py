"""Collision flags/debug output: many writers, both tiers, repeated clear/hit calls."""
import subprocess
from pathlib import Path

import pytest

from URDFParser import URDFParser
from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._collision import build_self_cc_ranges
from grid_codegen.resources import collision_include_dir
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable


@pytest.fixture(scope='module', params=[(False, False), (False, True), (True, False), (True, True)],
                ids=['single-env', 'single-self', 'two-env', 'two-self'])
def many_hits_executable(request, tmp_path_factory):
    two, self_hits = request.param
    directory = tmp_path_factory.mktemp('collision_many_hits')
    urdf = directory / 'tiny.urdf'
    links = ''.join(f'<link name="l{i}"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>' for i in range(4))
    joints = ''.join(f'<joint name="j{i}" type="revolute"><parent link="l{i}"/><child link="l{i+1}"/><origin xyz="0 0 0.3"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="10"/></joint>' for i in range(3))
    urdf.write_text(f'<robot name="many_hits">{links}{joints}</robot>')
    robot = URDFParser().parse(str(urdf))
    anchors = [i % 3 for i in range(72)]
    fine = dict(anchor=anchors, offset=[v for i in range(72) for v in (float(i), 0., 0.)],
                radius=[1000. if self_hits else .001]*72,
                self_cc_ranges=build_self_cc_ranges(robot, anchors))
    spec = {'tiers': [dict(fine, name='broad', radius=[r*2 for r in fine['radius']]),
                      dict(fine, name='fine')]} if two else fine
    header = directory / 'grid.cuh'
    GRiDCodeGenerator(robot, FILE_NAMESPACE='grid').gen_all_code(
        codegen_profile='kinematics', collision_spec=spec, output_path=str(header))
    arch = _detect_cuda_arch()
    exe, _ = cached_nvcc_executable(
        [Path(__file__).with_name('cuda_collision_many_hits_runner.cu'), header],
        ['-std=c++17', '-O2', '-gencode', f'arch=compute_{arch},code=sm_{arch}',
         f'-DTWO_TIERS={int(two)}', f'-DSELF_HITS={int(self_hits)}'],
        exe_name='many_hits.exe', fallback_dir=directory,
        include_dirs=[collision_include_dir()], what='adversarial collision flags')
    return exe


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize('threads', [1, 32, 100])
def test_collision_many_hits(many_hits_executable, threads):
    result = subprocess.run([str(many_hits_executable), str(threads)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'RESULT: PASS' in result.stdout

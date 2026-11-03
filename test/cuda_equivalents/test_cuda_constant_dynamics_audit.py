"""Constant-body transport and floating-coordinate ownership against locked models."""
from pathlib import Path
import subprocess

import numpy as np
import pytest
from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._idsva_so import gen_floating_gravity_d2tau_dq_spill_count
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable
from test.cuda_equivalents.constant_model import constant_model
from test.cuda_equivalents.thread_invariance import thread_sweep

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


def audit_model(directory, *, floating, descendant, locked=False):
    constant_model(directory, locked=locked)
    source = directory / f'model-False-{locked}-False.urdf'
    text = source.read_text()
    if descendant:
        text = text.replace('</robot>', '''
          <link name="tip"><inertial><origin xyz=".04 .03 -.02"/><mass value=".7"/>
            <inertia ixx=".2" iyy=".25" izz=".3" ixy=".01" ixz="0" iyz="0"/>
          </inertial></link>
          <joint name="tip" type="revolute"><parent link="follower"/><child link="tip"/>
            <origin xyz=".2 -.1 .15" rpy=".1 -.2 .3"/><axis xyz="1 0 0"/>
            <limit lower="-3" upper="3" effort="10" velocity="10"/>
          </joint></robot>''')
    source.write_text(text)
    return URDFParser().parse(str(source), floating_base=floating)


@pytest.fixture(scope='module', params=[(f, d, p) for f, d in
    [(False, True), (True, False), (True, True)] for p in ('float', 'double')])
def dynamics_probe(request, tmp_path_factory):
    floating, descendant, precision = request.param
    path = tmp_path_factory.mktemp(f'constant-dynamics-{floating}-{descendant}-{precision}')
    robot = audit_model(path, floating=floating, descendant=descendant)
    locked = audit_model(path, floating=floating, descendant=descendant, locked=True)
    gen = GRiDCodeGenerator(robot, dtype=precision)
    gen.gen_all_code(output_path=str(path / 'grid.cuh'), emit_plant=False,
                     algorithm_list=['crba', 'inverse_dynamics_gradient',
                                     'idsva_so_body_frame', 'idsva_so_world_frame'],
                     enable_idsva_so_body_frame=True, enable_idsva_so_world_frame=True,
                     enable_mujoco_kernels=False)
    # Use the same public-output epilogue as the generated body-frame kernel;
    # its fixed-base inner intentionally has a different mixed-axis layout.
    gen.code_str = ''
    gen.gen_idsva_so_body_frame_public_dvdq_layout_repair()
    runner = path / 'probe.cu'
    runner.write_text(Path(__file__).with_name('cuda_constant_dynamics_audit_runner.cu')
                      .read_text().replace('BODY_OUTPUT_REPAIR', gen.code_str))
    exe, _ = cached_nvcc_executable([runner], [
        '-std=c++17', '-O2', '-lineinfo', f'-arch=sm_{_detect_cuda_arch()}', f'-DDTYPE={precision}',
        f'-DID_TEMP={gen.gen_inverse_dynamics_gradient_inner_temp_mem_size()}',
        f'-DCRBA_TEMP={gen.gen_crba_inner_temp_mem_size()}',
        f'-DBODY_TEMP={gen.gen_idsva_so_body_frame_inner_temp_mem_size()}',
        f'-DWORLD_TEMP={gen.gen_idsva_so_world_frame_temp_mem_size()}',
        f'-DGRAV_TEMP={gen_floating_gravity_d2tau_dq_spill_count(gen)}'],
        exe_name='probe', fallback_dir=path, include_dirs=[path], what='constant dynamics audit runner')
    q = np.linspace(.13, .37, robot.get_num_pos())
    if floating:
        q[3:7] = np.array([.1, -.2, .3, .9]) / np.linalg.norm([.1, -.2, .3, .9])
    v = np.linspace(-.23, .19, robot.get_num_vel())
    a = np.linspace(.31, -.11, robot.get_num_vel())
    np.savetxt(path / 'inputs.txt', np.r_[q, v, a], fmt='%.17g')
    return robot, locked, path, exe, precision, q, v, a


@pytest.mark.parametrize('mode', [0, 1, 2, 3], ids=['crba', 'id-gradient', 'body-so', 'world-so'])
def test_constant_dynamics_audit(dynamics_probe, mode):
    robot, locked, path, exe, precision, q, v, a = dynamics_probe
    method = ['crba', 'inverse_dynamics_gradient', 'idsva_so_body_frame', 'idsva_so_world_frame'][mode]
    args = (q,) if mode == 0 else (q, v, a)
    reference = np.asarray(getattr(RBDReference(robot), method)(*args))
    independent = np.asarray(getattr(RBDReference(locked), method)(*args))
    np.testing.assert_allclose(reference, independent, atol=2e-10, rtol=2e-10)
    expected = reference.ravel(order='F' if mode < 2 else 'C')
    tol = 5e-5 if precision == 'float' else 2e-10
    def run(threads):
        ran = subprocess.run([str(exe), str(threads), str(mode), str(path / 'inputs.txt')],
                             capture_output=True, text=True)
        assert ran.returncode == 0, ran.stderr
        return np.fromstring(ran.stdout, sep=' ')

    def check(actual, _):
        assert actual.size == expected.size + 2
        np.testing.assert_array_equal(actual[[0, -1]], [123, 456])
        np.testing.assert_allclose(actual[1:-1], expected, atol=tol, rtol=tol)

    thread_sweep((1, 17, 32, 64, 256), run, check)

"""CPU guards for the plant gradient's distinct full-shared contract."""
from pathlib import Path
import re
import pytest
from URDFParser import URDFParser
from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._plant import _plant_step_gradient_extra_t_buffers


def test_spherical_plant_gradient_scratch(tmp_path):
    inertial = ('<inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" '
                'ixy="0" ixz="0" iyz="0"/></inertial>')
    links = ''.join(f'<link name="b{i}">{inertial}</link>' for i in range(6))
    joints = ''.join(f'<joint name="j{i}" type="spherical"><parent link="b{i}"/>'
                     f'<child link="b{i+1}"/></joint>' for i in range(5))
    path = tmp_path / 'spherical.urdf'
    path.write_text(f'<robot name="five_balls">{links}{joints}</robot>')
    robot = URDFParser().parse(str(path))
    assert robot.get_num_pos() == 20 and robot.get_num_vel() == 15
    slots = dict(_plant_step_gradient_extra_t_buffers(GRiDCodeGenerator(robot)))
    assert slots['s_x'] == 35
    assert slots['s_q_orig'] == 20
    assert slots['s_dInt_q_6x6'] == slots['s_dInt_v_6x6'] == 45


def test_every_plant_gradient_wrapper_uses_full_shared_reservation():
    source = (Path(__file__).parents[1] / 'bindings/grid_rbd/wrapper_template.cu').read_text()
    for function in ['launch_plant_step_gradient', 'launch_plant_step_gradient_mujoco',
                     'launch_plant_step_gradient_jax', 'torch_launch_plant_step_gradient']:
        body = source.split(f'void {function}(', 1)[1].split('\n}', 1)[0]
        assert 'grid_plant::PLANT_STEP_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()' in body
        assert 'grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES' not in body
    assert source.count('plant_step_gradient_check_shared_memory<T>()') == 4


@pytest.mark.parametrize('suffix', ['', '_gradient', '_hessian'])
@pytest.mark.parametrize('convention', ['', '_mujoco'])
def test_plant_step_cabi_scalars_match_precision(suffix, convention):
    root = Path(__file__).parents[1]
    wrapper = (root / 'bindings/grid_rbd/wrapper_template.cu').read_text()
    name = f'grid_plant_step{suffix}{convention}'
    signature = re.search(r'extern "C" int ' + name + r'\(([^)]*)\)', wrapper).group(1)
    assert 'T gravity, T dt' in signature
    core = (root / 'bindings/src/_core.cpp').read_text()
    pointer = core.split('using fn_plant_step_t', 1)[1].split(';', 1)[0]
    assert 'int, CT, CT, int)' in pointer

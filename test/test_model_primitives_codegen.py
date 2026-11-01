"""CPU generation boundaries for the additive model primitive surface."""
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
from grid_codegen import GRiDCodeGenerator
from URDFParser import URDFParser
from test.cuda_equivalents.manifold_oracle import reference


def test_kinematics_primitives_do_not_pull_in_plant(tmp_path):
    gen=GRiDCodeGenerator(reference('mixed_spherical_arm').robot)
    header=tmp_path/'grid.cuh'
    gen.gen_all_code(output_path=str(header),codegen_profile='kinematics',
        emit_plant=False,enable_mujoco_kernels=False)
    text=header.read_text()
    assert 'bool joint_transform(' in text
    assert 'bool joint_transform_scalar(' in text
    assert 'configuration_difference_jacobians(' not in text
    assert 'void integrator_hessian_device(' not in text


@pytest.mark.parametrize('joints',[1,2])
def test_spherical_topology_queries_joints_not_configuration(tmp_path,joints):
    path=Path(__file__).resolve().parents[1]/'external/URDFParser/tests/fixtures/spherical_arm.urdf'
    tree=ET.parse(path);root=tree.getroot()
    if joints==1:
        root.remove(root.find("joint[@name='joint_2']"));root.remove(root.find("link[@name='link2']"))
    else:
        root.find("joint[@name='joint_2']").set('type','spherical')
    tree.write(tmp_path/'single.urdf')
    robot=URDFParser().parse(str(tmp_path/'single.urdf'))
    assert (robot.get_num_joints(),robot.get_num_pos(),robot.get_num_vel())==(joints,4*joints,3*joints)
    gen=GRiDCodeGenerator(robot)
    assert gen.gen_topology_helpers_size()==0
    gen.gen_all_code(output_path=str(tmp_path/'grid.cuh'),
        algorithm_list=['fdsva_so','integrator_gradient'],emit_plant=True,
        enable_mujoco_kernels=False)
    # The smallest supported model also needs the serial retract scratch floor.
    from grid_codegen.algorithms._plant import _plant_hessian_pool_size
    from grid_codegen.algorithms._integrator_gradient import SPHERICAL_HESSIAN_SCRATCH
    assert _plant_hessian_pool_size(gen)>=SPHERICAL_HESSIAN_SCRATCH

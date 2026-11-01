"""Constant mimic expressions must not read a missing or unrelated q slot."""
import pytest
from URDFParser import URDFParser
from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from grid_codegen.helpers._topology_helpers import _emit_mimic_q_fold
from grid_codegen.algorithms._eepose_gradient_hessian import _ee_pose_inner_angle_expr


@pytest.mark.parametrize('kind', ['prismatic', 'revolute'])
@pytest.mark.parametrize('constant_only', [False, True])
def test_constant_fold_has_no_coordinate_read(tmp_path, kind, constant_only):
    text = robot_urdf('slider_mimic_fixed_tool').read_text()
    text = text.replace('type="prismatic"', f'type="{kind}"')
    text = text.replace('mimic joint="slide"', 'mimic joint="tool"')
    if constant_only:
        text = text.replace(f'<joint name="slide" type="{kind}">', '<joint name="slide" type="fixed">')
    path = tmp_path / 'constant.urdf'
    path.write_text(text)
    robot = URDFParser().parse(str(path))
    joint = robot.get_joint_by_name('follower')
    gen = GRiDCodeGenerator(robot, False, False, 'grid')
    expression = _ee_pose_inner_angle_expr(gen, joint.jid)
    assert expression == 'static_cast<T>(0.17)'
    _emit_mimic_q_fold(gen)
    emitted = gen.code_str
    assert f's_temp[{joint.jid}] = static_cast<T>(0.17);' in emitted
    if constant_only:
        assert 's_q[' not in emitted

"""Plant cost menu codegen referees (CPU): cost-frame baking, the emitted menu
surface and its gates, and the frame_eval arena offset the velocity term reads.

iiwa14 fixed: the default frame list is the end-effector joint alone; a named
articulated joint and a fixed joint (with its offset transform) bake as frames 1
and 2; an unknown name is refused; cost_frames pulls the frame Jacobian family
in; the menu preset, the velocity term and the gravity inner are emitted on the
fixed base; on the floating go2 the velocity term is emitted, and the tangent
state term and the preset appear once the Lie helper bundle (the integrator) is
in the algorithm list, in the tangent layout with its constants.
"""
from __future__ import annotations

import contextlib
import os
import re

import numpy as np
import pytest

from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._plant_menu import resolve_cost_frames, MENU_TERMS
from grid_assets import robot_urdf
from URDFParser import URDFParser


def _robot(name, floating=False):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        return URDFParser().parse(str(robot_urdf(name)), floating_base=floating)


def _gen(tmp_path, robot, **kw):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        gen = GRiDCodeGenerator(robot, False, False, "grid")
        out = tmp_path / "grid.cuh"
        gen.gen_all_code(output_path=str(out), enable_mujoco_kernels=False, **kw)
    return gen, out.read_text()


def test_cost_frame_resolution_and_baking(tmp_path):
    robot = _robot("iiwa14")
    gen = GRiDCodeGenerator(robot, False, False, "grid")
    gen._ee_target_name = ""
    ee = robot.get_joint_by_id(robot.get_leaf_nodes()[0]).get_name()
    assert [f[0] for f in resolve_cost_frames(gen, None)] == [ee]
    fixed = robot.get_fixed_joint_by_name("iiwa_joint_ee")
    assert fixed is not None
    frames = resolve_cost_frames(gen, [ee, "iiwa_joint_4", "iiwa_joint_ee"])
    assert [f[0] for f in frames] == [ee, "iiwa_joint_4", "iiwa_joint_ee"]
    assert frames[1][1] == robot.get_joint_by_name("iiwa_joint_4").get_id()
    assert np.allclose(np.asarray(frames[1][2]).reshape(4, 4, order="F"), np.eye(4))
    assert frames[2][1] == robot.get_joint_by_name(fixed.get_parent()).get_id()
    X = np.asarray(frames[2][2]).reshape(4, 4, order="F")
    assert np.allclose(X, np.asarray(fixed.get_transformation_matrix_hom())) and not np.allclose(X, np.eye(4))
    with pytest.raises(ValueError):
        resolve_cost_frames(gen, ["no_such_frame"])
    # a frame given twice, or the end-effector repeated, bakes once
    assert len(resolve_cost_frames(gen, ["iiwa_joint_4", "iiwa_joint_4", ee])) == 2

    gen, text = _gen(tmp_path, robot, cost_frames=["iiwa_joint_4", "iiwa_joint_ee"], algorithm_list=["inverse_dynamics"])
    assert "frame_jacobian" in gen.generated_algorithms
    assert "const int GRID_PLANT_NUM_COST_FRAMES = 3;" in text
    assert re.search(r"static const int cf_jid\[\] = \{ 6, 3, 6 \};", text)
    assert "void frame_eval(" in text and "void frame_position_cost(" in text
    assert "void frame_orientation_cost_hessian(" in text and "void frame_velocity_eval(" in text
    assert "GRID_PLANT_HAS_COST_MENU 1" in text and "GRID_PLANT_HAS_GRAVITY_TORQUE 1" in text
    for i, t in enumerate(MENU_TERMS):
        assert f"const int GRID_PLANT_MENU_{t} = {i};" in text
    # the velocity derivative reads the world transforms at the frame_eval carve's s_temp offset
    i = text.index("void frame_eval(")
    carve = re.findall(r"T \*(s_\w+) = grid_arena_ptr<T>\(s_arena, s_arena_offset\);\n\s*s_arena_offset \+= sizeof\(T\) \* static_cast<size_t>\((\d+)\);", text[i:i + 6000])
    names = [c[0] for c in carve]
    before_temp = sum(int(sz) for nm, sz in carve[:names.index("s_temp")])
    assert gen._frame_eval_xworld_offset == before_temp
    assert f"reinterpret_cast<const T *>(s_scratch) + {before_temp};" in text


def test_frame_terms_gates(tmp_path):
    # without frame_jacobian and without cost_frames the frame terms are skipped with a note
    robot = _robot("iiwa14")
    _, text = _gen(tmp_path, robot, algorithm_list=["inverse_dynamics", "end_effector_pose", "end_effector_pose_gradient"])
    assert "void frame_eval(" not in text and "frame_eval / frame_*_cost / tracking_cost_menu skipped" in text
    assert "void state_cost(" in text and "void linear_feature_cost(" in text
    # one capability constant per menu piece (the adapter static_asserts on these, not on comments)
    for name in ("ACTIVATIONS", "STATE_COST", "INPUT_COST", "LINEAR_FEATURE_COST", "GRAVITY_TORQUE"):
        assert "#define GRID_PLANT_HAS_" + name + " 1" in text, name
    # floating go2 without the Lie helper bundle: the frame terms including the twist are emitted,
    # the tangent state term and the preset are skipped with the reason
    go2 = _robot("go2", True)
    _, text = _gen(tmp_path, go2, cost_frames=[go2.get_joint_by_id(1).get_name()], algorithm_list=["inverse_dynamics"])
    assert "void frame_orientation_cost(" in text and "void frame_position_cost(" in text and "void frame_velocity_cost(" in text
    for name in ("FRAME_EVAL", "FRAME_POSITION_COST", "FRAME_ORIENTATION_COST", "FRAME_VELOCITY_COST"):
        assert "#define GRID_PLANT_HAS_" + name + " 1" in text, name
    assert "GRID_PLANT_HAS_COST_MENU" not in text and "void state_cost_tangent(" not in text
    assert "state_cost_tangent / tracking_cost_menu skipped: the tangent state residual needs the grid:: Lie helper bundle" in text
    # floating go2 with the integrator: the tangent state term and the preset in the tangent layout
    _, text = _gen(tmp_path, go2, cost_frames=[go2.get_joint_by_id(1).get_name()], algorithm_list=["inverse_dynamics", "integrator"])
    nv = go2.get_num_vel()
    assert "#define GRID_PLANT_HAS_STATE_COST_TANGENT 1" in text and "#define GRID_PLANT_HAS_COST_MENU 1" in text
    assert "void state_cost_tangent(" in text and "void tracking_cost_menu(" in text
    assert "const int GRID_PLANT_MENU_TANGENT = 1;" in text
    assert "const int GRID_PLANT_MENU_NU = " + str(nv - 6) + ";" in text
    assert "const int GRID_PLANT_MENU_NLX = " + str(2 * nv - 6) + ";" in text
    assert "state_cost_tangent<T, true>(s_out, s_x, p.x_des, p.Q" in text
    assert "joint_position_barrier<T, " + str(go2.get_num_pos() - 7) + ", 7>(s_out, s_x" in text
    # the fixed iiwa14 preset states the x layout
    _, text = _gen(tmp_path, robot, cost_frames=[robot.get_joint_by_id(1).get_name()], algorithm_list=["inverse_dynamics", "end_effector_pose", "end_effector_pose_gradient", "frame_jacobian"])
    assert "const int GRID_PLANT_MENU_TANGENT = 0;" in text and "const int GRID_PLANT_MENU_NU = " + str(robot.get_num_vel()) + ";" in text


def test_activation_ids_and_term_order_match_the_oracle():
    from RBDReference import RBDReference
    from grid_codegen.algorithms._plant_menu import ACTIVATIONS
    ids = dict(ACTIVATIONS)
    assert ids["GRID_PLANT_ACTIVATION_QUADRATIC"] == RBDReference.ACTIVATION_QUADRATIC
    assert ids["GRID_PLANT_ACTIVATION_PSEUDO_HUBER"] == RBDReference.ACTIVATION_PSEUDO_HUBER
    assert ids["GRID_PLANT_ACTIVATION_RELAXED_LOG_BARRIER"] == RBDReference.ACTIVATION_RELAXED_LOG_BARRIER
    assert ids["GRID_PLANT_ACTIVATION_QUADRATIC_BARRIER"] == RBDReference.ACTIVATION_QUADRATIC_BARRIER
    assert ids["GRID_PLANT_ACTIVATION_SMOOTH_HINGE"] == RBDReference.ACTIVATION_SMOOTH_HINGE
    assert ids["GRID_PLANT_ACTIVATION_NONE"] == -1


def test_root_attached_named_target_bakes_no_default_frame(tmp_path):
    """go2's imu_joint is a fixed joint on the base (the world root for the frame
    chain): as the baked named target it yields no default cost frame and the
    frame terms are skipped with a note, the build itself succeeding (the receipt
    cell test_ee_named_root_attached_zero_gradient_hessian); naming it explicitly
    as a cost frame is refused."""
    go2 = _robot("go2", False)
    gen, text = _gen(tmp_path, go2, fixed_target_name="imu_joint", algorithm_list=["end_effector_pose", "frame_jacobian"])
    assert gen._cost_frames == []
    assert "void frame_eval(" not in text and "attached to the world root" in text
    gen2 = GRiDCodeGenerator(go2, False, False, "grid")
    gen2._ee_target_name = "imu_joint"
    assert resolve_cost_frames(gen2, None) == []
    assert [f[0] for f in resolve_cost_frames(gen2, [go2.get_joint_by_id(1).get_name()])] == [go2.get_joint_by_id(1).get_name()]
    with pytest.raises(ValueError):
        resolve_cost_frames(gen2, ["imu_joint"])
    # on the floating go2 the same frame hangs off the floating-base joint and bakes as frame 0 on joint 0
    go2f = _robot("go2", True)
    gen3 = GRiDCodeGenerator(go2f, False, False, "grid")
    gen3._ee_target_name = "imu_joint"
    frames = resolve_cost_frames(gen3, None)
    assert [f[0] for f in frames] == ["imu_joint"] and frames[0][1] == 0

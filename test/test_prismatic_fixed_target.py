"""G13: all-prismatic kinematics with a nonidentity fixed tool."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from URDFParser import URDFParser


def test_single_dof_contraction_uses_one_tile():
    from grid_codegen.algorithms._fdsva_so import _fdsva_so_minv_tile
    assert _fdsva_so_minv_tile(1) == 1
    for n in range(2, 65):
        width = _fdsva_so_minv_tile(n)
        assert 1 <= width <= 8 and n % width == 0 and n // width >= 2


@pytest.mark.parametrize("profile,axis", [("kinematics", "0.6 0 0.8"), ("all", "0 0 1")])
@pytest.mark.parametrize("model", ["slider_fixed_tool", "slider_mimic_fixed_tool"])
def test_prismatic_fixed_tool_generation(tmp_path, profile, axis, model):
    # General-axis second-order dynamics are not supported by the all profile.
    # Exercise the reported skew kinematics case and cardinal full generation.
    urdf = tmp_path / "slider.urdf"
    urdf.write_text(robot_urdf(model).read_text().replace("0.6 0 0.8", axis))
    robot = URDFParser().parse(str(urdf))
    output = tmp_path / "grid.cuh"
    gen = GRiDCodeGenerator(robot, False, False, "grid")
    gen.gen_all_code(output_path=str(output), codegen_profile=profile,
                     include_homogenous_transforms=True, fixed_target_name="tool",
                     enable_mujoco_kernels=False)
    text = output.read_text()
    assert "load_update_XmatsHom_helpers" in text
    # q-dependent translation must still be updated when there is no sin/cos.
    helper = text.split("void load_update_XmatsHom_helpers(", 1)[1].split("\n    }", 1)[0]
    assert "s_q[0]" in helper
    assert "cgrps::memcpy_async(tgrp" not in text
    if "mimic" in model:
        for loader in ("load_update_XmatsHom_helpers", "load_update_XImats_helpers"):
            body = text.split(f"void {loader}(", 1)[1].split("\n    }", 1)[0]
            assert "static_cast<T>(-0.5) * s_q[0] + static_cast<T>(0.17)" in body


@pytest.mark.parametrize("q", [0., 0.13, 0.5])
@pytest.mark.parametrize("mimic", [False, True])
def test_slider_oracle_matches_analytic_transform_and_jacobian(q, mimic):
    from RBDReference import RBDReference
    robot = URDFParser().parse(str(robot_urdf("slider_mimic_fixed_tool" if mimic else "slider_fixed_tool")))
    ref = RBDReference(robot)
    r = Rotation.from_euler("xyz", [0.3, 0.2, 0.1]).as_matrix()
    p = np.array([0.04, 0.01, 0.02])
    axis = np.array([0.6, 0., 0.8])
    displacement = 0.5 * q + 0.17 if mimic else q
    expected = np.r_[p + r @ (axis * displacement + p), Rotation.from_matrix(r @ r).as_euler("xyz")]
    actual = np.asarray(ref.end_effector_pose(np.array([q]), ee_joint_names=["tool"])).reshape(6)
    jac = np.asarray(ref.end_effector_pose_gradient(np.array([q]), ee_joint_names=["tool"])).reshape(6)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(jac, np.r_[(0.5 if mimic else 1.) * (r @ axis), np.zeros(3)],
                               atol=1e-12, rtol=1e-12)

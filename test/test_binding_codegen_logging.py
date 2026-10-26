"""Library code generation is quiet on stdout, diagnostic via the package logger."""
import logging

from grid_assets import robot_urdf
from grid_rbd._compile import generate_grid_cuh


def test_codegen_reports_joint_order_only_to_logger(tmp_path, capsys, caplog):
    with caplog.at_level(logging.INFO, logger="grid_rbd"):
        generate_grid_cuh(robot_urdf("slider_fixed_tool"),
                          {"algorithm_list": ["end_effector_pose"],
                           "ee_joint_names": ["tool"], "enable_mujoco_kernels": False},
                          tmp_path / "grid.cuh")
    assert capsys.readouterr().out == ""
    assert "joint order: slide (nq=1, nv=1)" in caplog.text

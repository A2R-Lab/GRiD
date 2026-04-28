import numpy as np
import pytest

from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.model_sources import iter_robot_cases
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples
from test.pinocchio_equivalents.utils.comparators import assert_close


def pose_vector_to_rotation_matrix(pose_vector):
    roll, pitch, yaw = pose_vector[3:]
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, sr], [0.0, -sr, cr]])
    ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    rz = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rx @ ry @ rz


def assert_pose_close(actual, expected):
    assert_close(actual[:3], expected[:3], algorithm="rnea")
    actual_rot = pose_vector_to_rotation_matrix(actual)
    expected_rot = pose_vector_to_rotation_matrix(expected)
    assert_close(actual_rot, expected_rot, algorithm="rnea")


def build_iiwa_fixed_case_params():
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="fixed"):
        spec = case["spec"]
        if spec.robot_id != "iiwa14":
            continue
        params.append(
            pytest.param(
                spec,
                "fixed",
                id="iiwa14-fixed",
                marks=[
                    pytest.mark.pinocchio_equivalence,
                    pytest.mark.developer_only,
                    pytest.mark.robot_smoke,
                ],
            )
        )
    return params


@pytest.mark.parametrize(("spec", "base_mode"), build_iiwa_fixed_case_params())
@pytest.mark.parametrize(
    ("target_name", "offset"),
    [
        ("iiwa_joint_7", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        ("iiwa_joint_ee", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        ("tool0_joint", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        ("tool0_joint", np.array([0.02, 0.0, 0.01, 1.0], dtype=np.float64)),
    ],
)
def test_iiwa_fixed_base_pose_targets_match_pinocchio(
    spec, base_mode, target_name, offset, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        actual = project_model.end_effector_pose(sample.q, target_name, offset=offset)
        expected = pinocchio_model.end_effector_pose(sample.q, target_name, offset=offset)
        assert_pose_close(actual, expected)

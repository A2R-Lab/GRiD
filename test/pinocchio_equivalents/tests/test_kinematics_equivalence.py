import numpy as np
import pytest

from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.model_sources import iter_robot_cases
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples
from test.pinocchio_equivalents.utils.comparators import assert_close

POSE_TARGETS = {
    "iiwa14": [
        ("iiwa_joint_7", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        ("iiwa_joint_ee", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        ("tool0_joint", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        ("tool0_joint", np.array([0.02, 0.0, 0.01, 1.0], dtype=np.float64)),
    ],
    "go2": [
        ("FL_calf_joint", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        ("FL_foot_joint", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        ("RR_foot_joint", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        ("FL_foot_joint", np.array([0.01, 0.0, -0.015, 1.0], dtype=np.float64)),
    ],
    "g1": [
        ("left_ankle_roll_joint", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        ("right_ankle_roll_joint", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        ("left_hand_palm_joint", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
        ("pelvis_contour_joint", np.array([0.01, -0.01, 0.0, 1.0], dtype=np.float64)),
    ],
}


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


def assert_pose_close(actual, expected, robot_id: str):
    assert_close(actual[:3], expected[:3], algorithm="rnea", robot_id=robot_id)
    actual_rot = pose_vector_to_rotation_matrix(actual)
    expected_rot = pose_vector_to_rotation_matrix(expected)
    assert_close(actual_rot, expected_rot, algorithm="rnea", robot_id=robot_id)


def build_fixed_pose_case_params():
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="fixed"):
        spec = case["spec"]
        if spec.robot_id not in POSE_TARGETS:
            continue
        for target_name, offset in POSE_TARGETS[spec.robot_id]:
            params.append(
                pytest.param(
                    spec,
                    "fixed",
                    target_name,
                    offset,
                    id=f"{spec.robot_id}-fixed-{target_name}",
                    marks=[
                        pytest.mark.pinocchio_equivalence,
                        pytest.mark.developer_only,
                        pytest.mark.robot_smoke,
                    ],
                )
            )
    return params


@pytest.mark.parametrize(
    ("spec", "base_mode", "target_name", "offset"), build_fixed_pose_case_params()
)
def test_fixed_base_pose_targets_match_pinocchio(
    spec, base_mode, target_name, offset, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        actual = project_model.end_effector_pose(sample.q, target_name, offset=offset)
        expected = pinocchio_model.end_effector_pose(sample.q, target_name, offset=offset)
        assert_pose_close(actual, expected, robot_id=spec.robot_id)

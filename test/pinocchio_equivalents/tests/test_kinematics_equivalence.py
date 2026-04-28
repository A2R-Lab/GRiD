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


def assert_pose_close(actual, expected, robot_id: str):
    assert_close(actual[:3], expected[:3], algorithm="rnea", robot_id=robot_id)
    actual_rot = pose_vector_to_rotation_matrix(actual)
    expected_rot = pose_vector_to_rotation_matrix(expected)
    assert_close(actual_rot, expected_rot, algorithm="rnea", robot_id=robot_id)


def assert_pose_and_rotation_close(
    actual_pose,
    expected_pose,
    actual_rot,
    expected_rot,
    robot_id: str,
):
    assert_close(actual_pose[:3], expected_pose[:3], algorithm="rnea", robot_id=robot_id)
    assert_close(actual_rot, expected_rot, algorithm="rnea", robot_id=robot_id)


def build_fixed_pose_case_params():
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="fixed"):
        spec = case["spec"]
        params.append(
            pytest.param(
                spec,
                "fixed",
                id=f"{spec.robot_id}-fixed",
                marks=[
                    pytest.mark.pinocchio_equivalence,
                    pytest.mark.developer_only,
                    pytest.mark.robot_smoke,
                ],
            )
        )
    return params


def select_pose_targets(project_model, pinocchio_model):
    targets = []
    mimic_joint_names = set(getattr(pinocchio_model, "mimic_joint_names", []))
    if project_model.joint_names:
        targets.append((project_model.joint_names[0], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)))
        if len(project_model.joint_names) > 1:
            targets.append((project_model.joint_names[-1], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)))
    if project_model.fixed_joint_names:
        targets.append((project_model.fixed_joint_names[0], np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)))

    deduped = []
    seen = set()
    for target_name, offset in targets:
        if target_name in seen:
            continue
        if target_name in mimic_joint_names:
            continue
        if target_name not in pinocchio_model.frame_names:
            continue
        deduped.append((target_name, offset))
        seen.add(target_name)

    if deduped:
        offset_target = deduped[-1][0]
        deduped.append(
            (offset_target, np.array([0.01, 0.0, 0.01, 1.0], dtype=np.float64))
        )
    return deduped


@pytest.mark.parametrize(("spec", "base_mode"), build_fixed_pose_case_params())
def test_fixed_base_pose_targets_match_pinocchio(spec, base_mode, project_model, pinocchio_model):
    targets = select_pose_targets(project_model, pinocchio_model)
    if not targets:
        pytest.skip(f"No common pose targets available for {spec.robot_id}")
    for target_name, offset in targets:
        for sample in build_dynamics_samples(project_model):
            actual = project_model.end_effector_pose(sample.q, target_name, offset=offset)
            expected = pinocchio_model.end_effector_pose(sample.q, target_name, offset=offset)
            actual_rot = project_model.end_effector_rotation_matrix(sample.q, target_name)
            expected_rot = pinocchio_model.end_effector_rotation_matrix(sample.q, target_name)
            assert_pose_and_rotation_close(
                actual,
                expected,
                actual_rot,
                expected_rot,
                robot_id=spec.robot_id,
            )

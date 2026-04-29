import numpy as np
import pytest

from test.pinocchio_equivalents.conftest import MANIFEST_PATH, build_case_params
from test.pinocchio_equivalents.utils.comparators import assert_close
from test.pinocchio_equivalents.utils.model_sources import iter_robot_cases
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples


def select_derivative_target(project_model, pinocchio_model):
    mimic_joint_names = set(getattr(pinocchio_model, "mimic_joint_names", []))
    leaf_ids = project_model.robot.get_leaf_nodes()
    leaf_names = [
        project_model.robot.get_joint_by_id(leaf_id).get_name()
        for leaf_id in leaf_ids
    ]
    for joint_name in reversed(leaf_names):
        if joint_name in mimic_joint_names:
            continue
        if joint_name not in pinocchio_model.frame_names:
            continue
        return joint_name, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return None


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="floating"))
def test_floating_root_derivative_interface_is_well_formed(spec, base_mode, project_model):
    root_joint = project_model.robot.get_joint_by_id(0)
    root_indices = project_model.robot.get_joint_index_q(0)
    assert root_joint.get_local_q_dim() == 7
    assert list(root_indices) == [0, 1, 2, 3, 4, 5, 6]

    sample = build_dynamics_samples(project_model)[1]
    q_block = np.asarray(sample.q[root_indices], dtype=np.float64)
    xmat = np.asarray(
        project_model.robot.get_Xmat_hom_Func_by_id(0)(q_block),
        dtype=np.float64,
    )
    assert xmat.shape == (4, 4)
    assert np.isfinite(xmat).all()

    for local_index in range(root_joint.get_local_q_dim()):
        dX = np.asarray(
            project_model.robot.get_dXmat_hom_local_Func_by_id(0, local_index)(q_block),
            dtype=np.float64,
        )
        assert dX.shape == (4, 4)
        assert np.isfinite(dX).all()

    d2X = np.asarray(
        project_model.robot.get_d2Xmat_hom_local_Func_by_id(0, 3, 6)(q_block),
        dtype=np.float64,
    )
    assert d2X.shape == (4, 4)
    assert np.isfinite(d2X).all()


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="fixed"))
def test_fixed_base_pose_gradient_matches_pinocchio(
    spec, base_mode, project_model, pinocchio_model
):
    selection = select_derivative_target(project_model, pinocchio_model)
    if selection is None:
        pytest.skip(f"No common articulated leaf target available for {spec.robot_id}")
    target_name, offset = selection
    sample = build_dynamics_samples(project_model)[1]
    actual = project_model.end_effector_pose_gradient(sample.q, target_name, offset=offset)
    expected = pinocchio_model.end_effector_pose_gradient(sample.q, target_name, offset=offset)
    assert_close(actual, expected, algorithm="pose_gradient", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="floating"))
def test_floating_base_pose_gradient_matches_pinocchio(
    spec, base_mode, project_model, pinocchio_model
):
    selection = select_derivative_target(project_model, pinocchio_model)
    if selection is None:
        pytest.skip(f"No common articulated leaf target available for {spec.robot_id}")
    target_name, offset = selection
    sample = build_dynamics_samples(project_model)[1]
    actual = project_model.end_effector_pose_gradient(sample.q, target_name, offset=offset)
    expected = pinocchio_model.end_effector_pose_gradient(sample.q, target_name, offset=offset)
    assert_close(actual, expected, algorithm="pose_gradient", robot_id=spec.robot_id)


def build_hessian_case_params():
    params = []
    for case in iter_robot_cases(MANIFEST_PATH):
        spec = case["spec"]
        base_mode = case["base_mode"]
        if spec.robot_id != "iiwa14":
            continue
        marks = [
            pytest.mark.pinocchio_equivalence,
            pytest.mark.developer_only,
            getattr(pytest.mark, f"robot_{spec.tier}"),
        ]
        if base_mode == "floating":
            marks.append(pytest.mark.floating_base)
        params.append(pytest.param(spec, base_mode, id=f"{spec.robot_id}-{base_mode}", marks=marks))
    return params


@pytest.mark.parametrize(("spec", "base_mode"), build_hessian_case_params())
def test_pose_hessian_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    selection = select_derivative_target(project_model, pinocchio_model)
    if selection is None:
        pytest.skip(f"No common articulated leaf target available for {spec.robot_id}")
    target_name, offset = selection
    sample = build_dynamics_samples(project_model)[1]
    actual = project_model.end_effector_pose_hessian(sample.q, target_name, offset=offset)
    expected = pinocchio_model.end_effector_pose_hessian(sample.q, target_name, offset=offset)
    assert_close(actual, expected, algorithm="pose_hessian", robot_id=spec.robot_id)

import pytest

from test.pinocchio_equivalents.conftest import build_case_params


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params())
def test_model_dimensions_are_positive(spec, base_mode, project_model, pinocchio_model):
    assert project_model.nq > 0
    assert project_model.nv > 0
    assert pinocchio_model.nq > 0
    assert pinocchio_model.nv > 0
    assert project_model.nv == pinocchio_model.nv
    continuous_count = sum(
        1 for joint_type in project_model.joint_types_by_id.values() if joint_type == "continuous"
    )
    assert pinocchio_model.nq == project_model.nq + continuous_count


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params())
def test_joint_names_are_unique(spec, base_mode, project_model, pinocchio_model):
    assert len(project_model.joint_names) == len(set(project_model.joint_names))
    assert len(pinocchio_model.joint_names) == len(set(pinocchio_model.joint_names))


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params())
def test_actuated_joint_sets_match(spec, base_mode, project_model, pinocchio_model):
    project_names = project_model.actuated_joint_names
    pin_names = pinocchio_model.actuated_joint_names
    assert project_names, "Expected at least one movable or actuated joint on the GRiD side."
    assert pin_names, "Expected at least one movable or actuated joint on the Pinocchio side."
    assert set(project_names) == set(pin_names), (
        f"Actuated joint name sets differ for {spec.robot_id} ({base_mode}). "
        f"GRiD={project_names}, Pinocchio={pin_names}"
    )

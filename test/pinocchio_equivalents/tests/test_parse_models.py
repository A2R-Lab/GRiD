from pathlib import Path

import pytest

from test.pinocchio_equivalents.conftest import build_case_params


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params())
def test_robot_resolution_succeeds(spec, base_mode, resolved_robot_spec, source_lock_entry):
    assert resolved_robot_spec.robot_id == spec.robot_id
    assert Path(resolved_robot_spec.urdf_path).is_file(), resolved_robot_spec.urdf_path
    assert source_lock_entry["resolved_urdf_path"] == resolved_robot_spec.urdf_path


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params())
def test_grid_strict_parse_succeeds(
    spec, base_mode, resolved_robot_spec, project_model_attempt
):
    project_model, error = project_model_attempt
    assert error is None, (
        "GRiD strict parse failed for "
        f"robot_id={spec.robot_id}, embodiment={spec.embodiment}, "
        f"source_kind={spec.source_kind}, urdf={resolved_robot_spec.urdf_path}, "
        f"base_mode={base_mode}: {error}"
    )
    assert project_model.robot is not None
    assert project_model.parse_output is not None


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params())
def test_pinocchio_load_succeeds(spec, base_mode, pinocchio_model):
    assert pinocchio_model.model is not None
    assert pinocchio_model.data is not None

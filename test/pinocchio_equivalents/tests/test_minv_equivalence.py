import pytest

from test.pinocchio_equivalents.utils.comparators import assert_close
from test.pinocchio_equivalents.conftest import build_case_params
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="fixed"))
def test_fixed_base_minv_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    for sample in build_dynamics_samples(project_model):
        if not pinocchio_model.has_invertible_mass_matrix(sample.q):
            pytest.skip(
                f"{spec.robot_id} fixed-base mass matrix is singular for the resolved source model, so minv equivalence is not well-defined."
            )
        actual = project_model.minv(sample.q)
        expected = pinocchio_model.minv(sample.q)
        assert_close(actual, expected, algorithm="minv", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="floating"))
def test_floating_base_minv_matches_pinocchio(
    spec, base_mode, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        if not pinocchio_model.has_invertible_mass_matrix(sample.q):
            pytest.skip(
                f"{spec.robot_id} floating-base mass matrix is singular for the resolved source model, so minv equivalence is not well-defined."
            )
        actual = project_model.minv(sample.q)
        expected = pinocchio_model.minv(sample.q)
        assert_close(actual, expected, algorithm="minv", robot_id=spec.robot_id)

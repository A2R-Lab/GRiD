import pytest

from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.model_sources import iter_robot_cases
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples
from test.pinocchio_equivalents.utils.comparators import assert_close


def build_fixed_case_params():
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


def build_floating_case_params():
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="floating"):
        spec = case["spec"]
        params.append(
            pytest.param(
                spec,
                "floating",
                id=f"{spec.robot_id}-floating",
                marks=[
                    pytest.mark.pinocchio_equivalence,
                    pytest.mark.developer_only,
                    pytest.mark.robot_smoke,
                    pytest.mark.floating_base,
                ],
            )
        )
    return params


@pytest.mark.parametrize(("spec", "base_mode"), build_fixed_case_params())
def test_fixed_base_crba_matches_pinocchio(
    spec, base_mode, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        actual = project_model.crba(sample.q)
        expected = pinocchio_model.crba(sample.q)
        assert_close(actual, expected, algorithm="minv", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_floating_case_params())
def test_floating_base_crba_matches_pinocchio(
    spec, base_mode, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        if not pinocchio_model.has_invertible_mass_matrix(sample.q):
            pytest.skip(
                f"{spec.robot_id} floating-base mass matrix is singular for the resolved source model, so CRBA equivalence is not well-defined."
            )
        actual = project_model.crba(sample.q)
        expected = pinocchio_model.crba(sample.q)
        assert_close(actual, expected, algorithm="minv", robot_id=spec.robot_id)

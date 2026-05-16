import pytest

from test.pinocchio_equivalents.conftest import build_case_params
from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.comparators import assert_close
from test.pinocchio_equivalents.utils.model_sources import iter_robot_cases
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples


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


@pytest.mark.parametrize(("spec", "base_mode"), build_fixed_case_params())
def test_fixed_base_forward_dynamics_grad_matches_pinocchio_aba_derivatives(
    spec, base_mode, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        if not pinocchio_model.has_invertible_mass_matrix(sample.q):
            pytest.skip(
                f"{spec.robot_id} fixed-base mass matrix is singular for the resolved source model, so forward-dynamics derivatives are not well-defined."
            )
        actual_dq, actual_dqd = project_model.forward_dynamics_grad(
            sample.q, sample.qd, sample.qdd
        )
        expected_dq, expected_dqd = pinocchio_model.forward_dynamics_grad(
            sample.q, sample.qd, sample.qdd
        )
        assert_close(actual_dq, expected_dq, algorithm="rnea", robot_id=spec.robot_id)
        assert_close(actual_dqd, expected_dqd, algorithm="rnea", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="floating"))
def test_floating_base_forward_dynamics_grad_matches_pinocchio(
    spec, base_mode, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        if not pinocchio_model.has_invertible_mass_matrix(sample.q):
            pytest.skip(
                f"{spec.robot_id} floating-base mass matrix is singular for the resolved source model, so forward-dynamics derivatives are not well-defined."
            )
        actual_dq, actual_dqd = project_model.forward_dynamics_grad(
            sample.q, sample.qd, sample.qdd
        )
        expected_dq, expected_dqd = pinocchio_model.forward_dynamics_grad(
            sample.q, sample.qd, sample.qdd
        )
        assert_close(actual_dq, expected_dq, algorithm="rnea", robot_id=spec.robot_id)
        assert_close(actual_dqd, expected_dqd, algorithm="rnea", robot_id=spec.robot_id)

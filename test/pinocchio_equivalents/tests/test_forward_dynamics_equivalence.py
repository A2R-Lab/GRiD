import pytest

from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.comparators import assert_close
from test.pinocchio_equivalents.utils.model_sources import iter_robot_cases
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples


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
def test_iiwa_fixed_base_forward_dynamics_matches_pinocchio_aba(
    spec, base_mode, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        actual = project_model.forward_dynamics(sample.q, sample.qd, sample.qdd)
        expected = pinocchio_model.forward_dynamics(sample.q, sample.qd, sample.qdd)
        assert_close(actual, expected, algorithm="rnea")


@pytest.mark.parametrize(("spec", "base_mode"), build_iiwa_fixed_case_params())
def test_iiwa_fixed_base_forward_dynamics_matches_project_aba(
    spec, base_mode, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        actual = project_model.forward_dynamics(sample.q, sample.qd, sample.qdd)
        expected = project_model.aba(sample.q, sample.qd, sample.qdd)
        assert_close(actual, expected, algorithm="rnea")

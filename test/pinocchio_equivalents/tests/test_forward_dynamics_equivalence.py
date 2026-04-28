import pytest

from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.comparators import assert_close
from test.pinocchio_equivalents.utils.model_sources import iter_robot_cases
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples


SUPPORTED_FIXED_ROBOTS = {"iiwa14", "go2", "g1"}


def build_fixed_case_params():
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="fixed"):
        spec = case["spec"]
        if spec.robot_id not in SUPPORTED_FIXED_ROBOTS:
            continue
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
def test_fixed_base_forward_dynamics_matches_pinocchio_aba(
    spec, base_mode, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        actual = project_model.forward_dynamics(sample.q, sample.qd, sample.qdd)
        expected = pinocchio_model.forward_dynamics(sample.q, sample.qd, sample.qdd)
        assert_close(actual, expected, algorithm="rnea", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_fixed_case_params())
def test_fixed_base_forward_dynamics_matches_project_aba(
    spec, base_mode, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        actual = project_model.forward_dynamics(sample.q, sample.qd, sample.qdd)
        expected = project_model.aba(sample.q, sample.qd, sample.qdd)
        assert_close(actual, expected, algorithm="rnea", robot_id=spec.robot_id)

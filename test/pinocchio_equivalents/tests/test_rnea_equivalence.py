import pytest

from test.pinocchio_equivalents.utils.capability_matrix import get_capability
from test.pinocchio_equivalents.utils.comparators import assert_close
from test.pinocchio_equivalents.conftest import build_case_params
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="fixed"))
def test_fixed_base_rnea_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    for sample in build_dynamics_samples(project_model):
        actual = project_model.rnea(sample.q, sample.qd, sample.qdd)
        expected = pinocchio_model.rnea(sample.q, sample.qd, sample.qdd)
        assert_close(actual, expected, algorithm="rnea")


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="floating"))
def test_floating_base_rnea_matches_pinocchio_when_supported(
    spec, base_mode, project_model, pinocchio_model
):
    capability = get_capability(base_mode, "rnea")
    if not capability["supported"]:
        pytest.xfail(capability["reason"])
    for sample in build_dynamics_samples(project_model):
        actual = project_model.rnea(sample.q, sample.qd, sample.qdd)
        expected = pinocchio_model.rnea(sample.q, sample.qd, sample.qdd)
        assert_close(actual, expected, algorithm="rnea")

"""Pinocchio-grounded equivalence tests for second-order dynamics derivatives.

These tests compare GRiD's analytic second-order paths against Pinocchio's bound
C++ `ComputeRNEASecondOrderDerivatives` (for `idsva_so` / `idsva_so_spatial_v2`)
and a Pinocchio-grounded analytic composition (for `fdsva_so`). The Pinocchio
second-order RNEA is exposed via the `pin_so_ext` pybind11 extension under
`test/pinocchio_equivalents/pin_so_ext/`; the loader builds it on demand.

These supersede the naive q-component finite-difference oracles in
`test_second_order_equivalence.py` (now removed for floating-base), which
perturbed quaternion components directly rather than in the body-frame Lie
tangent that the analytic path uses.
"""
import numpy as np
import pytest

from test.pinocchio_equivalents.conftest import build_case_params
from test.pinocchio_equivalents.utils.comparators import assert_close
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples


_IDSVA_TENSOR_NAMES = ("d2tau_dq", "d2tau_dqd", "d2tau_dvdq", "dM_dq")
_FDSVA_TENSOR_NAMES = ("daba_dqdq", "daba_dvdq", "daba_dvdv", "daba_dtdq")


def _has_invertible_mass_matrix(pinocchio_model, sample):
    return pinocchio_model.has_invertible_mass_matrix(sample.q)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params())
def test_idsva_so_matches_pinocchio_second_order(
    spec, base_mode, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        actual = project_model.idsva_so(sample.q, sample.qd, sample.qdd)
        expected = pinocchio_model.idsva_so(sample.q, sample.qd, sample.qdd)
        for name, a, e in zip(_IDSVA_TENSOR_NAMES, actual, expected):
            assert_close(
                np.asarray(a),
                np.asarray(e),
                algorithm="idsva_so",
                robot_id=spec.robot_id,
            )


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params())
def test_idsva_so_spatial_v2_matches_pinocchio_second_order(
    spec, base_mode, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        actual = project_model.reference.idsva_so_spatial_v2(
            sample.q, sample.qd, sample.qdd
        )
        expected = pinocchio_model.idsva_so(sample.q, sample.qd, sample.qdd)
        for name, a, e in zip(_IDSVA_TENSOR_NAMES, actual, expected):
            assert_close(
                np.asarray(a),
                np.asarray(e),
                algorithm="idsva_so",
                robot_id=spec.robot_id,
            )


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params())
def test_fdsva_so_matches_pinocchio_composition(
    spec, base_mode, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        if not _has_invertible_mass_matrix(pinocchio_model, sample):
            pytest.skip(
                f"{spec.robot_id}-{base_mode} mass matrix is singular for the "
                f"resolved source model at sample={sample.name}; fdsva_so is not "
                f"well-defined."
            )
        actual = project_model.fdsva_so(sample.q, sample.qd, sample.qdd)
        expected = pinocchio_model.fdsva_so(sample.q, sample.qd, sample.qdd)
        for name, a, e in zip(_FDSVA_TENSOR_NAMES, actual, expected):
            assert_close(
                np.asarray(a),
                np.asarray(e),
                algorithm="second_order_fdsva",
                robot_id=spec.robot_id,
            )

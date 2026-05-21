"""Pinocchio equivalence test for the RBDReference integrator.

Cross-checks `ProjectModelAdapter.integrator(_gradient)` (which delegates to
`RBDReference.integrator` / `.integrator_grad`) against
`PinocchioModelAdapter.integrator(_gradient)` (which calls `pin.integrate`,
`pin.dIntegrate`, and `pin.aba` directly). This is the strongest claim we
make for floating-base: Pinocchio's `integrate` IS the canonical Lie-group
retract, so a successful equivalence test means RBDReference's
hand-implemented SE(3) exp / right-Jacobian agree with Pinocchio bit-perfectly
(modulo float64 rounding).

For fixed-base both implementations reduce to the same algebraic identity,
so the test is a tight consistency check rather than an independent claim.
"""

from __future__ import annotations

import pytest

from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.comparators import assert_close
from test.pinocchio_equivalents.utils.model_sources import iter_robot_cases
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples


import numpy as np

_INTEGRATORS = ("euler", "semi_implicit_euler", "midpoint", "rk3", "rk4")
_DEFAULT_DT = 0.01


def _assert_integrator_state_close(actual_x, expected_x, project_model, pinocchio_model, spec):
    """Compare an integrator state x = [q_new, v_new] across the two adapters.

    The velocity block is in tangent space and compared directly. The
    configuration block is compared in the tangent space via `pin.difference`
    so the check is representation-agnostic (scalar-angle vs [cos,sin]
    continuous joints) and wrap-safe (2*pi), rather than comparing raw nq
    coordinates."""
    nq = project_model.nq
    nv = project_model.nv
    actual_x = np.asarray(actual_x, dtype=np.float64)
    expected_x = np.asarray(expected_x, dtype=np.float64)
    q_a, v_a = actual_x[:nq], actual_x[nq:]
    q_e, v_e = expected_x[:nq], expected_x[nq:]
    assert_close(v_a, v_e, algorithm="rnea", robot_id=spec.robot_id)
    q_residual = pinocchio_model.q_tangent_residual(q_a, q_e)
    assert_close(q_residual, np.zeros(nv), algorithm="rnea", robot_id=spec.robot_id)


def build_case_params(base_mode: str):
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        spec = case["spec"]
        marks = [
            pytest.mark.pinocchio_equivalence,
            pytest.mark.developer_only,
            pytest.mark.robot_smoke,
        ]
        if base_mode == "floating":
            marks.append(pytest.mark.floating_base)
        params.append(pytest.param(spec, base_mode, id=f"{spec.robot_id}-{base_mode}", marks=marks))
    return params


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("fixed"))
@pytest.mark.parametrize("integrator_type", _INTEGRATORS)
def test_fixed_base_integrator_matches_pinocchio(
    spec, base_mode, integrator_type, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        if not pinocchio_model.has_invertible_mass_matrix(sample.q):
            pytest.skip(
                f"{spec.robot_id} fixed-base mass matrix is singular for the resolved source model."
            )
        u = sample.qdd  # convention: 3rd input vector is the control torque
        actual_x = project_model.integrator(sample.q, sample.qd, u, _DEFAULT_DT, integrator_type=integrator_type)
        expected_x = pinocchio_model.integrator(sample.q, sample.qd, u, _DEFAULT_DT, integrator_type=integrator_type)
        _assert_integrator_state_close(actual_x, expected_x, project_model, pinocchio_model, spec)
        actual_dAB = project_model.integrator_gradient(sample.q, sample.qd, u, _DEFAULT_DT, integrator_type=integrator_type)
        expected_dAB = pinocchio_model.integrator_gradient(sample.q, sample.qd, u, _DEFAULT_DT, integrator_type=integrator_type)
        assert_close(actual_dAB, expected_dAB, algorithm="rnea", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params("floating"))
@pytest.mark.parametrize("integrator_type", _INTEGRATORS)
def test_floating_base_integrator_matches_pinocchio(
    spec, base_mode, integrator_type, project_model, pinocchio_model
):
    for sample in build_dynamics_samples(project_model):
        if not pinocchio_model.has_invertible_mass_matrix(sample.q):
            pytest.skip(
                f"{spec.robot_id} floating-base mass matrix is singular for the resolved source model."
            )
        u = sample.qdd
        actual_x = project_model.integrator(sample.q, sample.qd, u, _DEFAULT_DT, integrator_type=integrator_type)
        expected_x = pinocchio_model.integrator(sample.q, sample.qd, u, _DEFAULT_DT, integrator_type=integrator_type)
        _assert_integrator_state_close(actual_x, expected_x, project_model, pinocchio_model, spec)
        actual_dAB = project_model.integrator_gradient(sample.q, sample.qd, u, _DEFAULT_DT, integrator_type=integrator_type)
        expected_dAB = pinocchio_model.integrator_gradient(sample.q, sample.qd, u, _DEFAULT_DT, integrator_type=integrator_type)
        assert_close(actual_dAB, expected_dAB, algorithm="rnea", robot_id=spec.robot_id)

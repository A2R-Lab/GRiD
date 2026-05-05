import numpy as np
import pytest

from test.pinocchio_equivalents.conftest import build_case_params
from test.pinocchio_equivalents.utils.comparators import assert_close
from test.pinocchio_equivalents.utils.project_adapter import build_project_adapter
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples


def _build_models(spec, resolved_robot_spec):
    pinocchio_order = build_project_adapter(
        spec,
        resolved_robot_spec,
        base_mode="floating",
        floating_base_convention="pinocchio",
    )
    legacy_order = build_project_adapter(
        spec,
        resolved_robot_spec,
        base_mode="floating",
        floating_base_convention="legacy",
    )
    return pinocchio_order, legacy_order


def _convert_sample_to_legacy(legacy_model, sample):
    q = legacy_model.robot.denormalize_floating_base_q_output(sample.q)
    qd = legacy_model.robot.denormalize_floating_base_v_output(sample.qd)
    qdd = legacy_model.robot.denormalize_floating_base_v_output(sample.qdd)
    return q, qd, qdd


def _canonicalize_vector(legacy_model, vector):
    return legacy_model.robot.normalize_floating_base_v_input(vector)


def _canonicalize_vv_matrix(legacy_model, matrix):
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    permutation = legacy_model.robot.get_floating_base_v_permutation_to_internal()
    if permutation is None:
        return matrix
    matrix[: len(permutation), :] = matrix[permutation, :]
    matrix[:, : len(permutation)] = matrix[:, permutation]
    return matrix


def _canonicalize_v_reduced_q_matrix(legacy_model, matrix):
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    permutation = legacy_model.robot.get_floating_base_v_permutation_to_internal()
    if permutation is None:
        return matrix
    matrix[: len(permutation), :] = matrix[permutation, :]
    return matrix


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="floating"))
def test_floating_base_public_vector_outputs_match_across_conventions(
    spec, base_mode, developer_environment, resolved_robot_spec
):
    pin_model, legacy_model = _build_models(spec, resolved_robot_spec)

    for sample in build_dynamics_samples(pin_model):
        q_legacy, qd_legacy, qdd_legacy = _convert_sample_to_legacy(legacy_model, sample)

        assert_close(
            pin_model.rnea(sample.q, sample.qd, sample.qdd),
            _canonicalize_vector(
                legacy_model,
                legacy_model.reference.rnea(q_legacy, qd_legacy, qdd_legacy)[0],
            ),
            algorithm="rnea",
            robot_id=spec.robot_id,
        )
        assert_close(
            pin_model.aba(sample.q, sample.qd, sample.qdd),
            _canonicalize_vector(
                legacy_model,
                legacy_model.reference.aba(q_legacy, qd_legacy, qdd_legacy),
            ),
            algorithm="aba",
            robot_id=spec.robot_id,
        )
        assert_close(
            pin_model.forward_dynamics(sample.q, sample.qd, sample.qdd),
            _canonicalize_vector(
                legacy_model,
                legacy_model.reference.forward_dynamics(q_legacy, qd_legacy, qdd_legacy),
            ),
            algorithm="forward_dynamics",
            robot_id=spec.robot_id,
        )


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="floating"))
def test_floating_base_public_matrix_outputs_match_across_conventions(
    spec, base_mode, developer_environment, resolved_robot_spec
):
    pin_model, legacy_model = _build_models(spec, resolved_robot_spec)

    for sample in build_dynamics_samples(pin_model):
        q_legacy, _qd_legacy, _qdd_legacy = _convert_sample_to_legacy(legacy_model, sample)

        assert_close(
            pin_model.minv(sample.q),
            _canonicalize_vv_matrix(legacy_model, legacy_model.reference.minv(q_legacy)),
            algorithm="minv",
            robot_id=spec.robot_id,
        )
        assert_close(
            pin_model.crba(sample.q),
            _canonicalize_vv_matrix(legacy_model, legacy_model.reference.crba(q_legacy)),
            algorithm="minv",
            robot_id=spec.robot_id,
        )


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="floating"))
def test_floating_base_public_gradient_outputs_match_across_conventions(
    spec, base_mode, developer_environment, resolved_robot_spec
):
    pin_model, legacy_model = _build_models(spec, resolved_robot_spec)

    for sample in build_dynamics_samples(pin_model):
        q_legacy, qd_legacy, qdd_legacy = _convert_sample_to_legacy(legacy_model, sample)

        pin_dq, pin_dqd = pin_model.rnea_grad(sample.q, sample.qd, sample.qdd)
        legacy_du = legacy_model.reference.rnea_grad(q_legacy, qd_legacy, qdd_legacy)
        legacy_dq = _canonicalize_v_reduced_q_matrix(
            legacy_model, legacy_du[:, : pin_model.nv]
        )
        legacy_dqd = _canonicalize_vv_matrix(
            legacy_model, legacy_du[:, pin_model.nv :]
        )
        assert_close(pin_dq, legacy_dq, algorithm="rnea", robot_id=spec.robot_id)
        assert_close(pin_dqd, legacy_dqd, algorithm="rnea", robot_id=spec.robot_id)

        pin_fd_dq, pin_fd_dqd = pin_model.forward_dynamics_grad(
            sample.q, sample.qd, sample.qdd
        )
        legacy_fd_dq, legacy_fd_dqd = legacy_model.reference.forward_dynamics_grad(
            q_legacy, qd_legacy, qdd_legacy
        )
        assert_close(
            pin_fd_dq,
            _canonicalize_v_reduced_q_matrix(legacy_model, legacy_fd_dq),
            algorithm="forward_dynamics",
            robot_id=spec.robot_id,
        )
        assert_close(
            pin_fd_dqd,
            _canonicalize_vv_matrix(legacy_model, legacy_fd_dqd),
            algorithm="forward_dynamics",
            robot_id=spec.robot_id,
        )

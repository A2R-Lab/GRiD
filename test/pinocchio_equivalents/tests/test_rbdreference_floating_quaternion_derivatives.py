import numpy as np
import pytest

from RBDReference import RBDReference
from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from test.pinocchio_equivalents.utils.project_adapter import build_project_adapter
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples


def _floating_smoke_params():
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="floating"):
        spec = case["spec"]
        if spec.robot_id not in {"iiwa14", "go2"}:
            continue
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


def _build_project_model(spec, base_mode):
    return build_project_adapter(spec, resolve_robot_spec(spec), base_mode=base_mode)


def _perturb_reduced_quaternion(quat, col, step):
    perturbed = np.asarray(quat, dtype=np.float64).copy()
    perturbed[col] += step
    return perturbed


def test_reduced_quaternion_normalization_derivatives_match_finite_difference():
    quat = np.array([0.25, -0.4, 0.15, 0.87], dtype=np.float64)
    step = 1e-6

    actual_jac = RBDReference._reduced_quaternion_normalization_jacobian(quat)
    expected_jac = np.zeros((4, 3), dtype=np.float64)
    for col in range(3):
        q_pos = _perturb_reduced_quaternion(quat, col, step)
        q_neg = _perturb_reduced_quaternion(quat, col, -step)
        expected_jac[:, col] = (
            RBDReference._normalize_xyzw_quaternion(q_pos)
            - RBDReference._normalize_xyzw_quaternion(q_neg)
        ) / (2.0 * step)
    np.testing.assert_allclose(actual_jac, expected_jac, atol=1e-10, rtol=1e-10)

    actual_hess = RBDReference._reduced_quaternion_normalization_hessian(quat)
    expected_hess = np.zeros((4, 3, 3), dtype=np.float64)
    for col in range(3):
        q_pos = _perturb_reduced_quaternion(quat, col, step)
        q_neg = _perturb_reduced_quaternion(quat, col, -step)
        expected_hess[:, :, col] = (
            RBDReference._reduced_quaternion_normalization_jacobian(q_pos)
            - RBDReference._reduced_quaternion_normalization_jacobian(q_neg)
        ) / (2.0 * step)
    np.testing.assert_allclose(actual_hess, expected_hess, atol=1e-9, rtol=1e-9)


def test_reduced_quaternion_rotation_derivatives_match_finite_difference():
    quat = np.array([-0.35, 0.2, -0.5, 0.75], dtype=np.float64)
    step = 1e-6

    actual_jac = RBDReference._reduced_quaternion_rotation_jacobian(quat)
    expected_jac = np.zeros((3, 3, 3), dtype=np.float64)
    for col in range(3):
        q_pos = _perturb_reduced_quaternion(quat, col, step)
        q_neg = _perturb_reduced_quaternion(quat, col, -step)
        expected_jac[:, :, col] = (
            RBDReference._quat_xyzw_to_rotation_matrix(q_pos)
            - RBDReference._quat_xyzw_to_rotation_matrix(q_neg)
        ) / (2.0 * step)
    np.testing.assert_allclose(actual_jac, expected_jac, atol=1e-9, rtol=1e-9)

    actual_hess = RBDReference._reduced_quaternion_rotation_hessian(quat)
    expected_hess = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for col in range(3):
        q_pos = _perturb_reduced_quaternion(quat, col, step)
        q_neg = _perturb_reduced_quaternion(quat, col, -step)
        expected_hess[:, :, :, col] = (
            RBDReference._reduced_quaternion_rotation_jacobian(q_pos)
            - RBDReference._reduced_quaternion_rotation_jacobian(q_neg)
        ) / (2.0 * step)
    np.testing.assert_allclose(actual_hess, expected_hess, atol=2e-8, rtol=2e-8)


def test_reduced_quaternion_angular_map_derivatives_match_finite_difference():
    quat = np.array([-0.35, 0.2, -0.5, 0.75], dtype=np.float64)
    step = 1e-6

    for jacobian_fn, hessian_fn in (
        (
            RBDReference._reduced_quaternion_to_world_angular_jacobian,
            RBDReference._reduced_quaternion_to_world_angular_hessian,
        ),
        (
            RBDReference._reduced_quaternion_to_body_angular_jacobian,
            RBDReference._reduced_quaternion_to_body_angular_hessian,
        ),
    ):
        actual_jac = jacobian_fn(quat)
        actual_hess = hessian_fn(quat)
        expected_hess = np.zeros((3, 3, 3), dtype=np.float64)
        for col in range(3):
            q_pos = _perturb_reduced_quaternion(quat, col, step)
            q_neg = _perturb_reduced_quaternion(quat, col, -step)
            expected_hess[:, :, col] = (
                jacobian_fn(q_pos) - jacobian_fn(q_neg)
            ) / (2.0 * step)
        np.testing.assert_allclose(actual_hess, expected_hess, atol=2e-8, rtol=2e-8)

    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    np.testing.assert_allclose(
        RBDReference._reduced_quaternion_to_world_angular_jacobian(identity),
        2.0 * np.eye(3),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        RBDReference._reduced_quaternion_to_body_angular_jacobian(identity),
        2.0 * np.eye(3),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize(("spec", "base_mode"), _floating_smoke_params())
def test_floating_idsva_so_finite_diff_mode_matches_default(
    spec,
    base_mode,
):
    project_model = _build_project_model(spec, base_mode)
    sample = build_dynamics_samples(project_model)[0]
    default = project_model.reference.idsva_so(sample.q, sample.qd, sample.qdd)
    explicit = project_model.reference.idsva_so(
        sample.q,
        sample.qd,
        sample.qdd,
        floating_d2tau_dq_mode="finite_diff",
    )
    for default_tensor, explicit_tensor in zip(default, explicit):
        np.testing.assert_allclose(default_tensor, explicit_tensor, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(("spec", "base_mode"), _floating_smoke_params())
def test_floating_idsva_so_compare_mode_records_q_side_report(
    spec,
    base_mode,
):
    project_model = _build_project_model(spec, base_mode)
    for sample in build_dynamics_samples(project_model):
        project_model.reference.idsva_so(
            sample.q,
            sample.qd,
            sample.qdd,
            floating_d2tau_dq_mode="compare",
        )
        report = project_model.reference.last_floating_idsva_d2tau_dq_compare
        assert set(report) == {"bad_count", "max_abs", "rel_norm", "first_mismatches"}
        assert report["max_abs"] <= 1e-6
        assert report["rel_norm"] <= 1e-9

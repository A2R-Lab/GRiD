import numpy as np
import pytest

from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.comparators import assert_close
from test.pinocchio_equivalents.utils.model_sources import iter_robot_cases
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples


SMOKE_FLOATING_ROBOTS = {"iiwa14", "go2", "g1"}
FIXED_REGRESSION_ROBOTS = {"iiwa14"}


def build_fixed_case_params():
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="fixed"):
        spec = case["spec"]
        if spec.robot_id not in FIXED_REGRESSION_ROBOTS:
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


def build_floating_case_params():
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="floating"):
        spec = case["spec"]
        if spec.robot_id not in SMOKE_FLOATING_ROBOTS:
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


def _normalize_project_q(q, base_mode):
    q = np.asarray(q, dtype=np.float64).copy()
    if base_mode == "floating":
        quat = q[3:7]
        quat_norm = np.linalg.norm(quat)
        if quat_norm == 0.0:
            raise ValueError("Floating-base quaternion norm was zero during second-order normalization.")
        q[3:7] = quat / quat_norm
    return q


def _perturb_reduced_dynamics_q(q, q_ind, step, base_mode):
    q_pert = np.asarray(q, dtype=np.float64).copy()
    if base_mode == "floating":
        if q_ind < 6:
            q_pert[q_ind] += step
        else:
            q_pert[q_ind + 1] += step
        return _normalize_project_q(q_pert, base_mode)
    q_pert[q_ind] += step
    return q_pert


def _second_order_samples(project_model, base_mode):
    samples = build_dynamics_samples(project_model)
    if base_mode == "floating":
        return [samples[-1]]
    return samples


def _finite_difference_idsva_so(project_model, sample, step=1e-6):
    n = project_model.nv
    q = _normalize_project_q(sample.q, project_model.base_mode)
    qd = np.asarray(sample.qd, dtype=np.float64)
    qdd = np.asarray(sample.qdd, dtype=np.float64)

    d2tau_dq = np.zeros((n, n, n), dtype=np.float64)
    d2tau_dqd = np.zeros((n, n, n), dtype=np.float64)
    d2tau_dvdq = np.zeros((n, n, n), dtype=np.float64)
    dM_dq = np.zeros((n, n, n), dtype=np.float64)

    for dind in range(n):
        q_pos = _perturb_reduced_dynamics_q(q, dind, step, project_model.base_mode)
        q_neg = _perturb_reduced_dynamics_q(q, dind, -step, project_model.base_mode)

        dc_dq_pos, dc_dqd_pos = project_model.rnea_grad(q_pos, qd, qdd)
        dc_dq_neg, dc_dqd_neg = project_model.rnea_grad(q_neg, qd, qdd)
        d2tau_dq[:, :, dind] = (dc_dq_pos - dc_dq_neg) / (2.0 * step)
        d2tau_dvdq[:, :, dind] = (dc_dqd_pos - dc_dqd_neg) / (2.0 * step)
        dM_dq[:, :, dind] = (project_model.crba(q_pos) - project_model.crba(q_neg)) / (2.0 * step)

        qd_pos = qd.copy()
        qd_neg = qd.copy()
        qd_pos[dind] += step
        qd_neg[dind] -= step
        _dc_dq_tmp, dc_dqd_pos = project_model.rnea_grad(q, qd_pos, qdd)
        _dc_dq_tmp, dc_dqd_neg = project_model.rnea_grad(q, qd_neg, qdd)
        d2tau_dqd[:, :, dind] = (dc_dqd_pos - dc_dqd_neg) / (2.0 * step)

    return d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq


def _finite_difference_fdsva_so(project_model, sample, step=1e-6):
    n = project_model.nv
    q = _normalize_project_q(sample.q, project_model.base_mode)
    qd = np.asarray(sample.qd, dtype=np.float64)
    u = np.asarray(sample.qdd, dtype=np.float64)

    daba_dqdq = np.zeros((n, n, n), dtype=np.float64)
    daba_dvdq = np.zeros((n, n, n), dtype=np.float64)
    daba_dvdv = np.zeros((n, n, n), dtype=np.float64)
    daba_dtdq = np.zeros((n, n, n), dtype=np.float64)

    for dind in range(n):
        q_pos = _perturb_reduced_dynamics_q(q, dind, step, project_model.base_mode)
        q_neg = _perturb_reduced_dynamics_q(q, dind, -step, project_model.base_mode)

        fd_dq_pos, fd_dqd_pos = project_model.forward_dynamics_grad(q_pos, qd, u)
        fd_dq_neg, fd_dqd_neg = project_model.forward_dynamics_grad(q_neg, qd, u)
        daba_dqdq[:, :, dind] = (fd_dq_pos - fd_dq_neg) / (2.0 * step)
        daba_dvdq[:, :, dind] = (fd_dqd_pos - fd_dqd_neg) / (2.0 * step)
        daba_dtdq[:, :, dind] = (project_model.minv(q_pos) - project_model.minv(q_neg)) / (2.0 * step)

        qd_pos = qd.copy()
        qd_neg = qd.copy()
        qd_pos[dind] += step
        qd_neg[dind] -= step
        _fd_dq_tmp, fd_dqd_pos = project_model.forward_dynamics_grad(q, qd_pos, u)
        _fd_dq_tmp, fd_dqd_neg = project_model.forward_dynamics_grad(q, qd_neg, u)
        daba_dvdv[:, :, dind] = (fd_dqd_pos - fd_dqd_neg) / (2.0 * step)

    return daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq


@pytest.mark.parametrize(("spec", "base_mode"), build_fixed_case_params())
def test_fixed_base_idsva_so_matches_finite_difference(
    spec, base_mode, project_model
):
    for sample in _second_order_samples(project_model, base_mode):
        actual = project_model.idsva_so(sample.q, sample.qd, sample.qdd)
        expected = _finite_difference_idsva_so(project_model, sample)
        for actual_tensor, expected_tensor in zip(actual, expected):
            assert_close(
                actual_tensor,
                expected_tensor,
                algorithm="idsva_so",
                robot_id=spec.robot_id,
            )


@pytest.mark.parametrize(("spec", "base_mode"), build_floating_case_params())
def test_floating_base_idsva_so_matches_finite_difference(
    spec, base_mode, project_model
):
    for sample in _second_order_samples(project_model, base_mode):
        actual = project_model.idsva_so(sample.q, sample.qd, sample.qdd)
        expected = _finite_difference_idsva_so(project_model, sample)
        for actual_tensor, expected_tensor in zip(actual, expected):
            assert_close(
                actual_tensor,
                expected_tensor,
                algorithm="idsva_so",
                robot_id=spec.robot_id,
            )


@pytest.mark.parametrize(("spec", "base_mode"), build_fixed_case_params())
def test_fixed_base_fdsva_so_matches_finite_difference(
    spec, base_mode, project_model
):
    for sample in _second_order_samples(project_model, base_mode):
        actual = project_model.fdsva_so(sample.q, sample.qd, sample.qdd)
        expected = _finite_difference_fdsva_so(project_model, sample)
        for actual_tensor, expected_tensor in zip(actual, expected):
            assert_close(
                actual_tensor,
                expected_tensor,
                algorithm="second_order_fdsva",
                robot_id=spec.robot_id,
            )


@pytest.mark.parametrize(("spec", "base_mode"), build_floating_case_params())
def test_floating_base_fdsva_so_matches_finite_difference(
    spec, base_mode, project_model
):
    for sample in _second_order_samples(project_model, base_mode):
        actual = project_model.fdsva_so(sample.q, sample.qd, sample.qdd)
        expected = _finite_difference_fdsva_so(project_model, sample)
        for actual_tensor, expected_tensor in zip(actual, expected):
            assert_close(
                actual_tensor,
                expected_tensor,
                algorithm="second_order_fdsva",
                robot_id=spec.robot_id,
            )

import numpy as np
import pytest

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


def _set_deterministic_joint_positions(project_model, q):
    joint_count = project_model.nq - 7
    if joint_count:
        q[7:] = np.linspace(-0.15, 0.15, joint_count, dtype=np.float64)


def _make_lie_oracle_sample(project_model, pose_name):
    if pose_name == "random_nonidentity":
        sample = build_dynamics_samples(project_model)[1]
        return sample.q, sample.qd, sample.qdd

    q = np.zeros(project_model.nq, dtype=np.float64)
    qd = np.linspace(-0.35, 0.35, project_model.nv, dtype=np.float64)
    qdd = np.linspace(0.25, -0.25, project_model.nv, dtype=np.float64)
    _set_deterministic_joint_positions(project_model, q)

    if pose_name == "identity_nonzero":
        q[6] = 1.0
    elif pose_name == "fixed_nonidentity":
        axis = np.array([0.3, -0.4, 0.5], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(30.0)
        q[0:3] = np.array([0.05, -0.04, 0.03], dtype=np.float64)
        q[3:6] = axis * np.sin(0.5 * angle)
        q[6] = np.cos(0.5 * angle)
    else:
        raise ValueError(f"Unknown Lie-oracle sample: {pose_name}")
    return q, qd, qdd




@pytest.mark.parametrize(("spec", "base_mode"), _floating_smoke_params())
@pytest.mark.parametrize(
    "pose_name",
    ("identity_nonzero", "fixed_nonidentity", "random_nonidentity"),
)
def test_floating_idsva_so_matches_lie_finite_difference(
    spec,
    base_mode,
    pose_name,
):
    project_model = _build_project_model(spec, base_mode)
    q, qd, qdd = _make_lie_oracle_sample(project_model, pose_name)
    d2tau_dq = project_model.reference.idsva_so(q, qd, qdd)[0]
    lie_oracle = project_model.reference._floating_idsva_d2tau_dq_lie_finite_diff(
        q,
        qd,
        qdd,
    )
    np.testing.assert_allclose(d2tau_dq, lie_oracle, atol=1e-5, rtol=1e-7)



@pytest.mark.parametrize(("spec", "base_mode"), _floating_smoke_params())
def test_floating_gravity_direct_lie_d2tau_dq_matches_lie_finite_difference(
    spec,
    base_mode,
):
    project_model = _build_project_model(spec, base_mode)
    q, _qd, _qdd = _make_lie_oracle_sample(project_model, "fixed_nonidentity")
    zeros = np.zeros(project_model.nv, dtype=np.float64)
    direct = project_model.reference._floating_gravity_d2tau_dq_lie_direct(q)
    lie_oracle = project_model.reference._floating_idsva_d2tau_dq_lie_finite_diff(
        q,
        zeros,
        zeros,
    )
    np.testing.assert_allclose(direct, lie_oracle, atol=1e-6, rtol=1e-8)



"""Self-consistency tests for the RBDReference floating-base gravity-Hessian helper.

The primary floating-base second-order correctness checks live in
`test_second_order_pinocchio_equivalence.py`, which compares `idsva_so` /
`idsva_so_spatial_v2` / `fdsva_so` against Pinocchio's bound C++
`ComputeRNEASecondOrderDerivatives`. The single remaining test in this file
exercises the internal `_floating_gravity_d2tau_dq_lie_direct` helper against
RBDReference's own Lie-tangent finite-difference reference, which is the
oracle the helper is designed against. The helper has no Pinocchio
counterpart, so the self-consistency check stays useful.
"""
import numpy as np
import pytest

from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from test.pinocchio_equivalents.utils.project_adapter import build_project_adapter


def _floating_smoke_params():
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="floating"):
        spec = case["spec"]
        if spec.robot_id not in {"iiwa14", "go2"}:
            continue
        params.append(pytest.param(
            spec, "floating",
            id=f"{spec.robot_id}-floating",
            marks=[pytest.mark.pinocchio_equivalence,
                   pytest.mark.developer_only,
                   pytest.mark.robot_smoke,
                   pytest.mark.floating_base],
        ))
    return params


def _build_project_model(spec, base_mode):
    return build_project_adapter(spec, resolve_robot_spec(spec), base_mode=base_mode)


def _fixed_nonidentity_q(project_model):
    q = np.zeros(project_model.nq, dtype=np.float64)
    joint_count = project_model.nq - 7
    if joint_count:
        q[7:] = np.linspace(-0.15, 0.15, joint_count, dtype=np.float64)
    axis = np.array([0.3, -0.4, 0.5], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(30.0)
    q[0:3] = np.array([0.05, -0.04, 0.03], dtype=np.float64)
    q[3:6] = axis * np.sin(0.5 * angle)
    q[6] = np.cos(0.5 * angle)
    return q


@pytest.mark.parametrize(("spec", "base_mode"), _floating_smoke_params())
def test_floating_gravity_direct_lie_d2tau_dq_matches_lie_finite_difference(
    spec, base_mode,
):
    project_model = _build_project_model(spec, base_mode)
    q = _fixed_nonidentity_q(project_model)
    zeros = np.zeros(project_model.nv, dtype=np.float64)
    direct = project_model.reference._floating_gravity_d2tau_dq_lie_direct(q)
    lie_oracle = project_model.reference._floating_idsva_d2tau_dq_lie_finite_diff(
        q, zeros, zeros,
    )
    np.testing.assert_allclose(direct, lie_oracle, atol=1e-6, rtol=1e-8)

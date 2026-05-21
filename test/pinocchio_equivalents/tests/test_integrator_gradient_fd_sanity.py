"""Finite-difference sanity check for the analytical integrator gradient
(fixed-base only).

Compares `ProjectModelAdapter.integrator_gradient(...)` against a centered
finite difference of `ProjectModelAdapter.integrator(...)` for each
(integrator_type, sample) pair. This is independent of the CUDA codegen
and catches bugs in the hand-derived analytical gradient formulas — the same
Python reference that the `test_cuda_integrator_equivalence` suite diffs
the CUDA kernel against.

**Floating-base scope:** the floating-base analytical gradient lives in
the nv-tangent space and a numerically-precise FD comparison requires a
proper SE(3) log to extract tangent-space differences from the SE(3)
output state. Rather than maintain that machinery here we lean on the
dedicated `test_integrator_pinocchio_equivalence.py` suite, which diffs
our analytical floating-base gradient against `pin.dIntegrate` directly —
that's a tighter and more direct check.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.model_sources import iter_robot_cases, resolve_robot_spec
from test.pinocchio_equivalents.utils.project_adapter import build_project_adapter


_INTEGRATORS = ("euler", "semi_implicit_euler", "midpoint", "rk3", "rk4")


def _robot_ids():
    raw = os.environ.get("GRID_INTEGRATOR_FD_ROBOTS", "iiwa14,go2")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _robot_spec(robot_id: str, base_mode: str):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not in manifest")


def _samples(nv: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    out = []
    out.append((np.zeros(nv), np.zeros(nv), np.zeros(nv), "zero"))
    for k in range(2):
        out.append((
            rng.uniform(-1.0, 1.0, size=nv),
            rng.uniform(-1.0, 1.0, size=nv),
            rng.uniform(-0.5, 0.5, size=nv),
            f"random_{k}",
        ))
    return out


@pytest.mark.developer_only
@pytest.mark.parametrize("robot_id", _robot_ids(), ids=lambda r: f"{r}-fd-fixed")
@pytest.mark.parametrize("integrator_type", _INTEGRATORS)
def test_integrator_gradient_matches_finite_difference(robot_id, integrator_type):
    spec = _robot_spec(robot_id, "fixed")
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    pm = build_project_adapter(spec, resolved, base_mode="fixed")
    nv = pm.nv
    dt = 0.01
    eps = 1e-6

    for q, qd, u, name in _samples(nv):
        analytical = pm.integrator_gradient(q, qd, u, dt, integrator_type=integrator_type)
        inputs = np.concatenate([q, qd, u])
        fd = np.zeros((2 * nv, 3 * nv))
        for j in range(3 * nv):
            ip = inputs.copy(); ip[j] += eps
            im = inputs.copy(); im[j] -= eps
            qp, qdp, up = ip[:nv], ip[nv:2 * nv], ip[2 * nv:]
            qm, qdm, um = im[:nv], im[nv:2 * nv], im[2 * nv:]
            xp = pm.integrator(qp, qdp, up, dt, integrator_type=integrator_type)
            xm = pm.integrator(qm, qdm, um, dt, integrator_type=integrator_type)
            fd[:, j] = (xp - xm) / (2 * eps)
        np.testing.assert_allclose(
            analytical, fd, rtol=1e-4, atol=1e-5,
            err_msg=f"{robot_id} {integrator_type} analytical vs FD gradient @ {name}",
        )

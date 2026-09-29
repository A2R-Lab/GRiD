"""CPU-only guard: the CUDA integrator equivalence samples stay float32-meaningful.

The CUDA test compares float32 kernels against the float64 reference at rtol 5e-4.
That comparison measures the kernel only while the step itself is well-conditioned:
if relative noise of order a float32 forward-dynamics error, injected into every
stage evaluation of the float64 reference, already moves x_kp1 past the tolerance,
a failing cell says nothing about the kernel. Feeding DynamicsSample.qdd in as the
torque made energetic dt=0.1 RK4 steps diverge (G1: stage-4 qdd ~2e22) and failed
exactly that way; _torque_driven drives with u = ID(q, qd, qdd) instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from test.cuda_equivalents.test_cuda_integrator_equivalence import _samples, _torque_driven

# Relative noise per forward-dynamics evaluation. Fitted to the logged G1 RK4 failures
# (2026-09-28), the GPU behaved like 3e-7..1e-6, so 1e-6 is a conservative float32 model.
_FD_NOISE = 1.0e-6
_RTOL = _ATOL = 5.0e-4  # the CUDA test's tolerance
_DT = 0.1  # the largest step the CUDA test takes
_MARGIN = 0.25  # noise may use at most a quarter of the tolerance
_ROBOTS = [("iiwa14", "fixed"), ("go2", "floating"), ("g1", "fixed"), ("g1", "floating")]


def _project_model(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            spec = case["spec"]
            return build_project_adapter(spec, resolve_robot_spec(spec), base_mode=base_mode)
    pytest.skip(f"{robot_id}-{base_mode} not in manifest")


def _noise_to_tolerance(project_model, sample, u, monkeypatch):
    """max over entries of |x_noisy - x| / tolerance for one RK4 step (1.0 = the bound)."""
    reference = project_model.reference
    exact = project_model.integrator(sample.q, sample.qd, u, _DT, integrator_type="rk4")
    clean_fd = reference.forward_dynamics
    rng = np.random.default_rng(20260928)

    def noisy_fd(q, qd, tau, f_ext=None):
        out = np.asarray(clean_fd(q, qd, tau, f_ext=f_ext)).reshape(-1)
        return out * (1.0 + _FD_NOISE * rng.standard_normal(out.shape))

    monkeypatch.setattr(reference, "forward_dynamics", noisy_fd)
    noisy = project_model.integrator(sample.q, sample.qd, u, _DT, integrator_type="rk4")
    monkeypatch.setattr(reference, "forward_dynamics", clean_fd)
    tolerance = max(_ATOL, _RTOL * np.abs(exact).max()) + _RTOL * np.abs(exact)
    return float(np.max(np.abs(noisy - exact) / tolerance))


@pytest.mark.parametrize("robot_id,base_mode", _ROBOTS)
def test_torque_driven_samples_reach_their_acceleration(robot_id, base_mode):
    project_model = _project_model(robot_id, base_mode)
    for sample in _samples(project_model):
        driven = _torque_driven(project_model, sample)
        qdd = np.asarray(project_model.forward_dynamics(sample.q, sample.qd, driven.qdd)).reshape(-1)
        np.testing.assert_allclose(qdd, sample.qdd, rtol=1e-9, atol=1e-9, err_msg=sample.name)


@pytest.mark.parametrize("robot_id,base_mode", _ROBOTS)
def test_rk4_samples_are_float32_meaningful(robot_id, base_mode, monkeypatch):
    project_model = _project_model(robot_id, base_mode)
    for sample in _samples(project_model):
        driven = _torque_driven(project_model, sample)
        ratio = _noise_to_tolerance(project_model, sample, driven.qdd, monkeypatch)
        assert ratio < _MARGIN, f"{robot_id}-{base_mode} {sample.name}: noise/tolerance {ratio:.3g}"


def test_acceleration_as_torque_was_not_float32_meaningful(monkeypatch):
    """Negative control: the old convention fails the same check (G1: ~40x the tolerance)."""
    project_model = _project_model("g1", "fixed")
    sample = next(s for s in _samples(project_model) if s.name == "high_acceleration")
    assert _noise_to_tolerance(project_model, sample, sample.qdd, monkeypatch) > 1.0

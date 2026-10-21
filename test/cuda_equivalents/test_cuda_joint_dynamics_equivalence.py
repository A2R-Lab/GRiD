"""CUDA equivalence for USE_JOINT_DYNAMICS on a FLOATING base (root-skip path).

The damping/friction feature is validated fixed-base end-to-end by
`test/python_wrappers/test_joint_dynamics.py` (value + gradient vs the damped
oracle), but the FLOATING root-skip path (the 6 root velocity slots carry no
damping) had no CUDA gate — and the grid_rbd binding can't provide one (its
floating q packing is NUM_POS-wide vs the oracle's NUM_VEL tangent space). This
gate goes through the CUDA-equivalence runner instead, which reconciles the
floating layout: codegen iiwa14-FLOATING with USE_JOINT_DYNAMICS=True and
compare id/fd/aba values + id/fd gradients against
``RBDReference(use_joint_dynamics=True)``.

The oracle is the damped ProjectModelAdapter (NOT pinocchio — pin.rnea ignores
model.damping/friction, so it cannot oracle this feature). Independence is
retained by the python_wrappers FD self-check
(test_rbdreference_oracle_dqd_block_matches_fd) that pins the damped oracle
against its own finite differences.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

from grid_codegen import GRiDCodeGenerator
from RBDReference import RBDReference
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter
from test.cuda_equivalents.cuda_harness import (
    _build_cuda_samples,
    _compile_runner,
    _expected_output,
    _hash_file,
    _parse_runner_output,
    _run_runner,
    _sample_to_stdin,
)

pytestmark = [
    pytest.mark.cuda_equivalence,
    pytest.mark.developer_only,
    pytest.mark.robot_smoke,
    pytest.mark.floating_base,
]

_ALGOS = (
    "inverse_dynamics", "minv", "forward_dynamics", "aba", "crba",
    "inverse_dynamics_gradient", "forward_dynamics_gradient",
)
_RUN_TOKENS = frozenset({
    "RUN_INVERSE_DYNAMICS", "RUN_FORWARD_DYNAMICS", "RUN_ABA",
    "RUN_INVERSE_DYNAMICS_GRADIENT", "RUN_FORWARD_DYNAMICS_GRADIENT",
})
# (compared output, norm-relative tolerance): fp32 kernels vs the float64 damped
# oracle. NORM-relative (|cuda-ref| / |ref|), not absolute — the energetic
# samples produce Minv-amplified outputs in the 1e3-1e4 range where fp32 leaves
# proportional residuals (same rationale as the flagship suite's norm guards).
_COMPARES = (
    ("inverse_dynamics", 1e-4),
    ("forward_dynamics", 1e-3),
    ("aba", 1e-3),
    ("inverse_dynamics_gradient_q", 1e-3),
    ("inverse_dynamics_gradient_qd", 1e-3),
    ("forward_dynamics_gradient_q", 1e-2),
    ("forward_dynamics_gradient_qd", 1e-2),
)


def _floating_spec(robot_id):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="floating"):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-floating not in the robot manifest.")


def test_floating_damped_cuda_matches_damped_oracle(tmp_path):
    spec = _floating_spec("iiwa14")
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"could not resolve iiwa14 manifest: {exc}")

    project_model = build_project_adapter(spec, resolved, base_mode="floating")
    if not project_model.robot.robot_has_joint_damping():
        pytest.fail("iiwa14 manifest URDF no longer declares joint damping — "
                    "pick a damped robot for this gate.")
    damped = dataclasses.replace(
        project_model,
        reference=RBDReference(project_model.robot, use_joint_dynamics=True),
    )

    # damped codegen (USE_JOINT_DYNAMICS=True): id/fd/aba bias + id-gradient diag
    header = tmp_path / "grid.cuh"
    codegen = GRiDCodeGenerator(
        project_model.robot, DEBUG_MODE=False, NEED_PRINT_MAT=True,
        FILE_NAMESPACE="grid", USE_JOINT_DYNAMICS=True,
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            algorithm_list=list(_ALGOS),
            output_path=str(header),
        )
    executable, compile_cmd = _compile_runner(
        tmp_path,
        floating_base=True,
        # The runner key is content-hashed from tmp_path/grid.cuh, so the
        # damped header can never collide with the undamped flagship cache
        # entries for the same robot (different bytes, different key).
        run_tokens=_RUN_TOKENS,
    )

    samples = _build_cuda_samples(project_model, random_count=3)
    checked = 0
    for sample in samples:
        stdout = _run_runner(executable, _sample_to_stdin(sample), compile_cmd)
        outputs = _parse_runner_output(stdout)
        for name, tol in _COMPARES:
            if name not in outputs:
                pytest.fail(f"runner printed no '{name}' output for {sample.name}")
            expected = np.asarray(
                _expected_output(damped, damped, sample, name), dtype=np.float64
            ).reshape(-1)
            actual = np.asarray(outputs[name], dtype=np.float64).reshape(-1)
            assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
            scale = max(float(np.linalg.norm(expected)), 1.0)
            err = float(np.linalg.norm(actual - expected)) / scale
            assert err < tol, (
                f"{sample.name} {name}: damped CUDA vs damped oracle relnorm={err:.3e} (tol {tol})")
        # the damping term must be ACTIVE: the damped CUDA id must differ from the
        # UNDAMPED oracle by ~|damping*qd| (guards against a silently-ignored flag).
        bare = np.asarray(
            _expected_output(project_model, project_model, sample, "inverse_dynamics"),
            dtype=np.float64,
        ).reshape(-1)
        damped_id = np.asarray(outputs["inverse_dynamics"], dtype=np.float64).reshape(-1)
        if float(np.max(np.abs(np.asarray(sample.qd)))) > 0.1:
            assert float(np.max(np.abs(damped_id - bare))) > 1e-3, (
                f"{sample.name}: damped build matches the UNDAMPED oracle — "
                "USE_JOINT_DYNAMICS appears inactive")
        checked += 1
    assert checked >= 3


# ---- smooth (tanh) friction: derivatives at rest and inside the width ----------
_TANH_WIDTH = 0.03


def _rest_and_in_width_samples(samples, width):
    """Derive from each sample one at rest (qd = 0) and one with every velocity
    inside the tanh width, the two regimes where the sign model has no slope."""
    rng = np.random.default_rng(20261021)
    derived = []
    for s in samples:
        derived.append(dataclasses.replace(s, name=s.name + "_rest", qd=np.zeros_like(s.qd)))
        derived.append(dataclasses.replace(
            s, name=s.name + "_inwidth",
            qd=(rng.uniform(-0.5, 0.5, size=s.qd.shape) * width).astype(np.float64)))
    return derived


def test_floating_tanh_friction_cuda_matches_tanh_oracle(tmp_path):
    """fr3 FLOATING (friction + mimic fold + root skip) with FRICTION_MODEL="tanh":
    id/fd/aba values and the id/fd gradients match RBDReference(use_joint_dynamics=True,
    friction_model="tanh") at random velocities, at rest and inside the width. At
    rest the smooth gradient carries the friction slope (f/w) sech^2(0) = f/w per
    slot, which the sign-model oracle does not; the test pins that difference."""
    spec = _floating_spec("fr3")
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"could not resolve fr3 manifest: {exc}")

    project_model = build_project_adapter(spec, resolved, base_mode="floating")
    if not project_model.robot.robot_has_joint_friction():
        pytest.fail("fr3 manifest URDF no longer declares joint friction — "
                    "pick a robot with friction for this gate.")
    smooth = dataclasses.replace(
        project_model,
        reference=RBDReference(project_model.robot, use_joint_dynamics=True,
                               friction_model="tanh", friction_width=_TANH_WIDTH),
    )
    sign = dataclasses.replace(
        project_model,
        reference=RBDReference(project_model.robot, use_joint_dynamics=True),
    )

    header = tmp_path / "grid.cuh"
    codegen = GRiDCodeGenerator(
        project_model.robot, DEBUG_MODE=False, NEED_PRINT_MAT=True,
        FILE_NAMESPACE="grid", USE_JOINT_DYNAMICS=True,
        FRICTION_MODEL="tanh", FRICTION_WIDTH=_TANH_WIDTH,
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            algorithm_list=list(_ALGOS),
            output_path=str(header),
        )
    executable, compile_cmd = _compile_runner(
        tmp_path, floating_base=True, run_tokens=_RUN_TOKENS,
    )

    base = _build_cuda_samples(project_model, random_count=2)
    samples = base + _rest_and_in_width_samples(base, _TANH_WIDTH)
    nv = project_model.nv
    checked = 0
    slope_pinned = 0
    for sample in samples:
        stdout = _run_runner(executable, _sample_to_stdin(sample), compile_cmd)
        outputs = _parse_runner_output(stdout)
        for name, tol in _COMPARES:
            if name not in outputs:
                pytest.fail(f"runner printed no '{name}' output for {sample.name}")
            expected = np.asarray(
                _expected_output(smooth, smooth, sample, name), dtype=np.float64
            ).reshape(-1)
            actual = np.asarray(outputs[name], dtype=np.float64).reshape(-1)
            assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
            scale = max(float(np.linalg.norm(expected)), 1.0)
            err = float(np.linalg.norm(actual - expected)) / scale
            assert err < tol, (
                f"{sample.name} {name}: tanh CUDA vs tanh oracle relnorm={err:.3e} (tol {tol})")
        if sample.name.endswith("_rest"):
            # the smooth model's velocity gradient differs from the sign model's by
            # the friction slope on the diagonal of the actuated slots (f/w per slot)
            sign_grad = np.asarray(
                _expected_output(sign, sign, sample, "inverse_dynamics_gradient_qd"),
                dtype=np.float64).reshape(nv, nv)
            cuda_grad = np.asarray(outputs["inverse_dynamics_gradient_qd"], dtype=np.float64).reshape(nv, nv)
            slope = np.asarray(smooth.reference._joint_dynamics_derivative_diag(np.zeros(nv), 1)) \
                - np.asarray(sign.reference._joint_dynamics_derivative_diag(np.zeros(nv), 1))
            assert float(np.max(slope)) > 1.0, "fr3 friction slope at rest should be large (f/w)"
            delta = cuda_grad - sign_grad
            assert np.allclose(delta, np.diag(slope), atol=2e-2 * float(np.max(slope))), (
                f"{sample.name}: CUDA tanh gradient minus sign oracle is not the friction slope diagonal")
            slope_pinned += 1
        checked += 1
    assert checked >= 6 and slope_pinned >= 2

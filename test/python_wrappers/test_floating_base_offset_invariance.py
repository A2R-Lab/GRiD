"""Floating-base dynamics do not depend on where the base is.

Velocities, forces and their derivatives are expressed in body frames, so every
dynamics kernel returns the same result wherever the base sits in the world.
The generated code must also return the same bits: a kernel that carries the
base position through its recursion works about the world origin and loses
float32 accuracy with the square of the distance (the second-order kernels did,
before the floating root transform was evaluated with the position at zero).

Each kernel is called with the base at the origin and at two distant positions
and the outputs are compared for equality. Kernels whose result is a world
quantity are checked for the dependence they should have.

Run with:
    pytest test/python_wrappers/test_floating_base_offset_invariance.py -m python_wrappers -v
"""
from __future__ import annotations

import dataclasses
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
from grid_assets import robot_urdf  # noqa: E402

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers

_B = 8
_OFFSETS = ((30.0, -20.0, 5.0), (-400.0, 250.0, 1000.0))
_ALGORITHMS = [
    "inverse_dynamics", "forward_dynamics", "aba", "crba", "minv",
    "inverse_dynamics_gradient", "forward_dynamics_gradient",
    "idsva_so_body_frame", "fdsva_so", "coriolis_matrix",
    "nonlinear_effects", "generalized_gravity", "integrator_gradient",
    "end_effector_pose", "com",
]


def _flat(value):
    if dataclasses.is_dataclass(value):
        parts = [getattr(value, f.name) for f in dataclasses.fields(value)]
    elif isinstance(value, dict):
        parts = list(value.values())
    elif isinstance(value, (tuple, list)):
        parts = list(value)
    else:
        parts = [value]
    return np.concatenate([np.asarray(p).reshape(_B, -1) for p in parts], axis=1)


@pytest.fixture(scope="module", params=["go2", "iiwa14"])
def model(request):
    robot = request.param
    handle = grid_rbd.register_robot(
        f"{robot}_offset_invariance_pytest", str(robot_urdf(robot)), floating_base=True,
        max_batch_size=_B, algorithm_list=_ALGORITHMS, enable_mujoco_kernels=False)
    rng = np.random.default_rng(5)
    quat = rng.normal(size=(_B, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    state = dict(
        quat=quat, joints=rng.uniform(-1.0, 1.0, (_B, handle.nq - 7)),
        qd=rng.uniform(-1.0, 1.0, (_B, handle.nv)).astype(np.float32),
        qdd=rng.uniform(-1.0, 1.0, (_B, handle.nv)).astype(np.float32),
        u=rng.uniform(-5.0, 5.0, (_B, handle.nv)).astype(np.float32))
    return handle, state


def _q(state, position):
    base = np.tile(np.asarray(position, dtype=np.float64), (_B, 1))
    return np.concatenate([base, state["quat"], state["joints"]], axis=1).astype(np.float32)


_INVARIANT = {
    "inverse_dynamics": lambda h, q, s: h.inverse_dynamics(q, s["qd"], s["qdd"]),
    "forward_dynamics": lambda h, q, s: h.forward_dynamics(q, s["qd"], s["u"]),
    "aba": lambda h, q, s: h.aba(q, s["qd"], s["u"]),
    "crba": lambda h, q, s: h.crba(q),
    "minv": lambda h, q, s: h.minv(q),
    "inverse_dynamics_gradient": lambda h, q, s: h.inverse_dynamics_gradient(q, s["qd"], s["qdd"]),
    "forward_dynamics_gradient": lambda h, q, s: h.forward_dynamics_gradient(q, s["qd"], s["u"]),
    "idsva_so": lambda h, q, s: h.idsva_so(q, s["qd"], s["qdd"]),
    "fdsva_so": lambda h, q, s: h.fdsva_so(q, s["qd"], s["u"]),
    "coriolis_matrix": lambda h, q, s: h.coriolis_matrix(q, s["qd"]),
    "nonlinear_effects": lambda h, q, s: h.nonlinear_effects(q, s["qd"]),
    "generalized_gravity": lambda h, q, s: h.generalized_gravity(q),
    "integrator_gradient": lambda h, q, s: h.integrator_gradient(q, s["qd"], s["u"], 0.01),
}


@pytest.mark.parametrize("kernel", sorted(_INVARIANT))
def test_dynamics_output_is_bit_identical_at_any_base_position(model, kernel):
    handle, state = model
    call = _INVARIANT[kernel]
    at_origin = _flat(call(handle, _q(state, (0.0, 0.0, 0.0)), state))
    assert np.all(np.isfinite(at_origin))
    for position in _OFFSETS:
        moved = _flat(call(handle, _q(state, position), state))
        assert np.array_equal(moved, at_origin), (
            f"{kernel} changed with the base at {position}: "
            f"max abs difference {np.max(np.abs(moved.astype(np.float64) - at_origin.astype(np.float64))):.3e}")


def test_world_quantities_still_follow_the_base(model):
    """The pose and the centre of mass are world quantities and must move with
    the base; the dynamics change must not have zeroed the position for them."""
    handle, state = model
    position = np.asarray(_OFFSETS[0])
    com_origin = np.asarray(handle.com(_q(state, (0.0, 0.0, 0.0)))[0]).reshape(_B, 3)
    com_moved = np.asarray(handle.com(_q(state, position))[0]).reshape(_B, 3)
    np.testing.assert_allclose(com_moved - com_origin, np.tile(position, (_B, 1)), rtol=0, atol=2e-4)
    pose_origin = _flat(handle.end_effector_pose(_q(state, (0.0, 0.0, 0.0))))
    pose_moved = _flat(handle.end_effector_pose(_q(state, position)))
    assert np.max(np.abs(pose_moved - pose_origin)) > 1.0


def test_integrator_carries_the_true_position(model):
    handle, state = model
    if not hasattr(handle, "integrator"):
        pytest.skip("handle has no integrator method")
    position = np.asarray(_OFFSETS[0])
    q0 = _q(state, (0.0, 0.0, 0.0))
    q1 = _q(state, position)
    out0 = handle.integrator(q0, state["qd"], state["u"], 0.01)
    out1 = handle.integrator(q1, state["qd"], state["u"], 0.01)
    next0 = np.asarray(out0[0] if isinstance(out0, (tuple, list)) else out0).reshape(_B, -1)
    next1 = np.asarray(out1[0] if isinstance(out1, (tuple, list)) else out1).reshape(_B, -1)
    np.testing.assert_allclose(next1[:, :3] - next0[:, :3], np.tile(position, (_B, 1)), rtol=0, atol=2e-4)
    np.testing.assert_array_equal(next1[:, 3:handle.nq], next0[:, 3:handle.nq])

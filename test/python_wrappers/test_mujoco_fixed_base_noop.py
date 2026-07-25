"""`output_convention="mujoco"` must be a uniform, no-op interface on a FIXED base.

MuJoCo and Pinocchio conventions differ ONLY in the free-flyer (floating-base)
representation -- wxyz quat + global-linear base velocity. A fixed base has no free
joint, so mjx == pinocchio there, byte-for-byte. For ease of use we want the SAME
interface on every robot: generic code that does `register_robot(..., output_convention=
"mujoco")` (or calls `handle.mujoco.<method>`) should work on fixed AND floating robots
alike, returning the (identical) pinocchio values on a fixed base rather than raising.

Regression: register_robot used to REJECT output_convention="mujoco" on a fixed base with
a ValueError, breaking that uniformity. This asserts it is now accepted and behaves as a
no-op (and composes freely with enable_mujoco_kernels=False, since a fixed base emits no
mjx twins to drop).
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "bindings"))
from config import robot_urdf  # noqa: E402

_IIWA = robot_urdf("iiwa14")
pytestmark = [pytest.mark.python_wrappers, pytest.mark.developer_only]


def _has_cuda():
    return (shutil.which("nvcc") or Path("/usr/local/cuda/bin/nvcc").exists()) and shutil.which("nvidia-smi")


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _IIWA.exists(), reason="iiwa14 urdf missing")
def test_fixed_base_mujoco_convention_is_noop():
    from grid_rbd import register_robot
    # The subset build keeps this fast; the point is the API + no-op numerics, not coverage.
    common = dict(urdf_path=str(_IIWA), floating_base=False,
                  algorithm_list=["inverse_dynamics"], max_batch_size=8)
    # (1) accepted (no ValueError) even paired with enable_mujoco_kernels=False on a fixed base
    h_mjx = register_robot(name="iiwa14_fixed_mjx_noop", output_convention="mujoco",
                           enable_mujoco_kernels=False, force_rebuild=True, **common)
    assert h_mjx.output_convention == "mujoco"
    assert not h_mjx.floating_base

    rng = np.random.default_rng(0)
    q = rng.standard_normal((4, h_mjx.num_joints)).astype(np.float32)
    qd = rng.standard_normal((4, h_mjx.num_joints)).astype(np.float32)

    # (2) the mjx-convention output equals the pinocchio output byte-for-byte (no free-flyer)
    tau_mjx_default = np.asarray(h_mjx.inverse_dynamics(q, qd))     # handle default = mujoco
    tau_view = np.asarray(h_mjx.mujoco.inverse_dynamics(q, qd))     # explicit mujoco view
    tau_pin = np.asarray(h_mjx.inverse_dynamics(q, qd, _convention="pinocchio"))
    assert np.array_equal(tau_mjx_default, tau_pin), "fixed-base mujoco default != pinocchio"
    assert np.array_equal(tau_view, tau_pin), "fixed-base handle.mujoco view != pinocchio"

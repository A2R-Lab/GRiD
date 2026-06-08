"""GPU regression test for the NATIVE MuJoCo-convention kernel path.

The binding can produce mjx-convention outputs two ways:
  1. host post-process: run the pin kernel, then rotate on the host (`_mujoco.py`).
  2. NATIVE kernel: a `MUJOCO_OUTPUT=true` template instantiation of the kernel that
     bakes the convention transform in (raw mjx inputs in, mjx outputs out) — the
     `grid_rbd_*_mujoco` C-ABI entries, dispatched by `handle.mujoco.<method>`.

Path (1) is validated against real MuJoCo in `test_mujoco_transforms.py`. This test
guards path (2) — the inverse_dynamics REFERENCE for the codegen-fusion sweep — by
checking the native kernel agrees with the validated host oracle on a floating robot,
batched, and that the transform is actually non-trivial (base rows differ from pin).

Needs nvcc + a CUDA GPU (register_robot compiles a per-robot .so), so it is
developer_only and skipped where the toolchain/GPU is absent.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_GO2 = _REPO / "robot_assets" / "go2.urdf"

pytestmark = pytest.mark.developer_only


def _has_cuda() -> bool:
    if shutil.which("nvcc") is None and not Path("/usr/local/cuda/bin/nvcc").exists():
        return False
    return shutil.which("nvidia-smi") is not None


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_inverse_dynamics_matches_host_oracle():
    from grid_rbd import register_robot
    from grid_rbd import _mujoco as bm

    h = register_robot("go2_mjx_kernel_test", str(_GO2),
                       floating_base=True, force_rebuild=True)
    assert h.floating_base
    # The native mjx ID symbol must be present in a floating-base .so.
    assert h._runner.has_inverse_dynamics_mujoco, \
        "floating-base .so is missing grid_rbd_inverse_dynamics_mujoco"

    nq, nv = h.num_joints, h.num_vel
    rng = np.random.default_rng(0)
    for B in (1, 4):
        qpos = rng.standard_normal((B, nq))
        qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)  # wxyz quat
        qvel = rng.standard_normal((B, nq)); qvel[:, nv:] = 0.0  # pad past the tangent
        qacc = rng.standard_normal((B, nq)); qacc[:, nv:] = 0.0

        native = np.asarray(h.mujoco.inverse_dynamics(qpos, qvel, qacc), dtype=np.float64)

        q_pin, qd_pin, qdd_pin, _, R = h._mjx_inputs(qpos, qvel, qacc)
        pin_c = np.asarray(h.inverse_dynamics(q_pin, qd_pin, qdd_pin), dtype=np.float64)
        expected = bm.id_tau_pin_to_mjx(pin_c, R, True)

        assert np.allclose(native, expected, rtol=2e-3, atol=2e-2), \
            f"native mjx ID != host oracle (B={B}): max|d|={np.abs(native-expected).max():.3e}"

    # Non-triviality: the mjx base-linear rows must differ from the raw pin frame,
    # else a silently-broken (no-op) transform would pass the oracle check vacuously.
    raw_pin = np.asarray(h.inverse_dynamics(qpos, qvel, qacc), dtype=np.float64)
    assert np.abs(native[0, :3] - raw_pin[0, :3]).max() > 1e-2

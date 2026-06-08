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


@pytest.fixture(scope="module")
def go2_floating():
    """Register go2-floating ONCE (force_rebuild to exercise freshly-generated
    codegen, not a stale cache) and share it across the mjx-kernel checks."""
    from grid_rbd import register_robot
    return register_robot("go2_mjx_kernel_test", str(_GO2),
                          floating_base=True, force_rebuild=True)


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_inverse_dynamics_matches_host_oracle(go2_floating):
    from grid_rbd import _mujoco as bm
    h = go2_floating
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


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_crba_matches_host_oracle(go2_floating):
    """CONGRUENCE-class reference: M_mjx = G M_pin G^T baked into the kernel."""
    from grid_rbd import _mujoco as bm
    h = go2_floating
    assert h._runner.has_crba_mujoco, \
        "floating-base .so is missing grid_rbd_crba_mujoco"

    nq, nv = h.num_joints, h.num_vel
    rng = np.random.default_rng(1)
    for B in (1, 4):
        qpos = rng.standard_normal((B, nq))
        qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)  # wxyz quat

        native = np.asarray(h.mujoco.crba(qpos), dtype=np.float64)         # (B, nv, nv)

        q_pin, _, _, _, R = h._mjx_inputs(qpos)
        pin_M = np.asarray(h.crba(q_pin), dtype=np.float64)
        expected = bm.mass_matrix_pin_to_mjx(pin_M, R, True)

        assert np.allclose(native, expected, rtol=2e-3, atol=2e-2), \
            f"native mjx CRBA != host oracle (B={B}): max|d|={np.abs(native-expected).max():.3e}"

    # Non-triviality + SPD preservation (the congruence must really act on the base block).
    raw_pin = np.asarray(h.crba(qpos), dtype=np.float64)
    assert np.abs(native[0, :3, :] - raw_pin[0, :3, :]).max() > 1e-2
    assert np.linalg.eigvalsh(native[0]).min() > 0


def _rand_state(h, rng, B, with_qd=True, with_u=False):
    nq, nv = h.num_joints, h.num_vel
    qpos = rng.standard_normal((B, nq)); qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    qvel = rng.standard_normal((B, nq)); qvel[:, nv:] = 0.0
    u = rng.standard_normal((B, nq))
    return qpos, (qvel if with_qd else None), (u if with_u else None)


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
@pytest.mark.parametrize("method", ["forward_dynamics", "aba"])
def test_native_mjx_accel_out_matches_host_oracle(go2_floating, method):
    """accel_out class: qdd_mjx[0:3] = R(qdd_pin + omega x v)."""
    from grid_rbd import _mujoco as bm
    h = go2_floating
    assert getattr(h._runner, f"has_{method}_mujoco")
    rng = np.random.default_rng(2)
    for B in (1, 4):
        qpos, qvel, u = _rand_state(h, rng, B, with_qd=True, with_u=True)
        native = np.asarray(getattr(h, method)(qpos, qvel, u, _convention="mujoco"), dtype=np.float64)
        q_pin, qd_pin, _, u_pin, R = h._mjx_inputs(qpos, qvel, u=u)
        pin = np.asarray(getattr(h, method)(q_pin, qd_pin, u_pin), dtype=np.float64)
        expected = bm.fd_qdd_pin_to_mjx(pin, qd_pin, R, True)
        assert np.allclose(native, expected, rtol=2e-3, atol=2e-2), \
            f"{method} mjx != oracle (B={B}): max|d|={np.abs(native-expected).max():.3e}"
    raw_pin = np.asarray(getattr(h, method)(qpos, qvel, u), dtype=np.float64)
    assert np.abs(native[0, :3] - raw_pin[0, :3]).max() > 1e-2


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_coriolis_matches_host_oracle(go2_floating):
    """congruence class with qd input: C_mjx = G C_pin G^T."""
    from grid_rbd import _mujoco as bm
    h = go2_floating
    assert h._runner.has_coriolis_matrix_mujoco
    rng = np.random.default_rng(3)
    for B in (1, 4):
        qpos, qvel, _ = _rand_state(h, rng, B, with_qd=True)
        native = np.asarray(h.coriolis_matrix(qpos, qvel, _convention="mujoco"), dtype=np.float64)
        q_pin, qd_pin, _, _, R = h._mjx_inputs(qpos, qvel)
        pin_C = np.asarray(h.coriolis_matrix(q_pin, qd_pin), dtype=np.float64)
        expected = bm.coriolis_matrix_pin_to_mjx(pin_C, R, True)
        assert np.allclose(native, expected, rtol=2e-3, atol=2e-2), \
            f"coriolis mjx != oracle (B={B}): max|d|={np.abs(native-expected).max():.3e}"
    raw_pin = np.asarray(h.coriolis_matrix(qpos, qvel), dtype=np.float64)
    assert np.abs(native[0, :3, :] - raw_pin[0, :3, :]).max() > 1e-2


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_frame_jacobian_matches_host_oracle(go2_floating):
    """column-reframe class: J_mjx = J_pin G^{-1} (base-linear cols)."""
    from grid_rbd import _mujoco as bm
    h = go2_floating
    assert h._runner.has_frame_jacobian_mujoco and h._runner.has_frame_jacobian_dot_mujoco
    rng = np.random.default_rng(4)
    for B in (1, 4):
        qpos, qvel, _ = _rand_state(h, rng, B, with_qd=True)
        q_pin, qd_pin, _, _, R = h._mjx_inputs(qpos, qvel)
        # frame_jacobian (q only)
        nat_J = np.asarray(h.frame_jacobian(qpos, _convention="mujoco"), dtype=np.float64)
        exp_J = bm.jacobian_pin_to_mjx(np.asarray(h.frame_jacobian(q_pin), np.float64), R, True)
        assert np.allclose(nat_J, exp_J, rtol=2e-3, atol=2e-2), \
            f"frame_jacobian mjx != oracle (B={B}): max|d|={np.abs(nat_J-exp_J).max():.3e}"
        # frame_jacobian_dot (q, qd)
        nat_Jd = np.asarray(h.frame_jacobian_dot(qpos, qvel, _convention="mujoco"), dtype=np.float64)
        exp_Jd = bm.jacobian_pin_to_mjx(np.asarray(h.frame_jacobian_dot(q_pin, qd_pin), np.float64), R, True)
        assert np.allclose(nat_Jd, exp_Jd, rtol=2e-3, atol=2e-2), \
            f"frame_jacobian_dot mjx != oracle (B={B}): max|d|={np.abs(nat_Jd-exp_Jd).max():.3e}"
    # non-triviality: base-linear columns differ from the raw pin Jacobian
    raw_J = np.asarray(h.frame_jacobian(qpos), dtype=np.float64)
    assert np.abs(nat_J[0, :, :3] - raw_J[0, :, :3]).max() > 1e-2


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_osc_inertia_invariant_but_quat_reordered(go2_floating):
    """osc_inertia value is frame-INVARIANT, but the mjx q (wxyz) must be reordered
    before the kinematics build — the mjx kernel does that. So native(q_mjx) equals
    pin(q_pin), and BOTH differ from feeding the raw mjx q to the pin kernel."""
    h = go2_floating
    assert h._runner.has_osc_inertia_mujoco
    rng = np.random.default_rng(5)
    for B in (1, 4):
        qpos, _, _ = _rand_state(h, rng, B, with_qd=False)
        q_pin, _, _, _, _ = h._mjx_inputs(qpos)
        native = np.asarray(h.osc_inertia(qpos, _convention="mujoco"), dtype=np.float64)
        pin = np.asarray(h.osc_inertia(q_pin), dtype=np.float64)   # invariant -> equal
        assert np.allclose(native, pin, rtol=2e-3, atol=2e-2), \
            f"osc_inertia mjx != pin-invariant (B={B}): max|d|={np.abs(native-pin).max():.3e}"
    # feeding the raw mjx (wxyz) q to the PIN kernel mis-builds the kinematics -> differs,
    # confirming the quaternion reorder is load-bearing.
    raw_wrong = np.asarray(h.osc_inertia(qpos), dtype=np.float64)
    assert np.abs(native[0] - raw_wrong[0]).max() > 1e-3

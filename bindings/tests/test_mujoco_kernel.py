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


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_minv_matches_host_oracle(go2_floating):
    """CONGRUENCE-class: Minv_mjx = G Minv_pin G^T baked into the kernel (the native
    kernel returns a FULL DENSE SYMMETRIC mjx Minv — no host symmetrize/post-process)."""
    from grid_rbd import _mujoco as bm
    h = go2_floating
    assert h._runner.has_minv_mujoco, \
        "floating-base .so is missing grid_rbd_minv_mujoco"
    rng = np.random.default_rng(6)
    for B in (1, 4):
        qpos, _, _ = _rand_state(h, rng, B, with_qd=False)
        native = np.asarray(h.mujoco.minv(qpos), dtype=np.float64)         # (B, nv, nv)
        q_pin, _, _, _, R = h._mjx_inputs(qpos)
        pin_Minv = np.asarray(h.minv(q_pin), dtype=np.float64)
        expected = bm.minv_pin_to_mjx(pin_Minv, R, True)
        assert np.allclose(native, expected, rtol=2e-3, atol=2e-2), \
            f"native mjx Minv != host oracle (B={B}): max|d|={np.abs(native-expected).max():.3e}"
    # full symmetric output + non-trivial base-block reframe.
    assert np.allclose(native[0], native[0].T, rtol=2e-3, atol=2e-2)
    raw_pin = np.asarray(h.minv(qpos), dtype=np.float64)
    assert np.abs(native[0, :3, :] - raw_pin[0, :3, :]).max() > 1e-2


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_energy_invariant_but_inputs_converted(go2_floating):
    """energy is frame-INVARIANT, but the mjx q/qd MUST be converted (quat reorder +
    qd reframe) for a correct build — the mjx kernel does that. native(q_mjx) equals
    pin(q_pin), and feeding the raw mjx q/qd to the pin kernel differs."""
    h = go2_floating
    assert h._runner.has_energy_mujoco, \
        "floating-base .so is missing grid_rbd_energy_mujoco"
    rng = np.random.default_rng(7)
    for B in (1, 4):
        qpos, qvel, _ = _rand_state(h, rng, B, with_qd=True)
        q_pin, qd_pin, _, _, _ = h._mjx_inputs(qpos, qvel)
        native = np.asarray(h.mujoco.energy(qpos, qvel), dtype=np.float64)
        pin = np.asarray(h.energy(q_pin, qd_pin), dtype=np.float64)        # invariant -> equal
        assert np.allclose(native, pin, rtol=2e-3, atol=2e-2), \
            f"energy mjx != pin-invariant (B={B}): max|d|={np.abs(native-pin).max():.3e}"
    # feeding raw mjx q/qd to the pin kernel mis-builds -> differs (input convert is load-bearing).
    raw_wrong = np.asarray(h.energy(qpos, qvel), dtype=np.float64)
    assert np.abs(native[0] - raw_wrong[0]).max() > 1e-3


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_com_matches_host_oracle(go2_floating):
    """com: p_com is frame-INVARIANT; J_com is column-REFRAMED (J_mjx = J_pin G^{-1})."""
    from grid_rbd import _mujoco as bm
    h = go2_floating
    assert h._runner.has_com_mujoco, \
        "floating-base .so is missing grid_rbd_com_mujoco"
    rng = np.random.default_rng(8)
    for B in (1, 4):
        qpos, _, _ = _rand_state(h, rng, B, with_qd=False)
        q_pin, _, _, _, R = h._mjx_inputs(qpos)
        nat_p, nat_J = h.mujoco.com(qpos)
        nat_p = np.asarray(nat_p, np.float64); nat_J = np.asarray(nat_J, np.float64)
        pin_p, pin_J = h.com(q_pin)
        exp_p = np.asarray(pin_p, np.float64)                              # invariant
        exp_J = bm.jacobian_pin_to_mjx(np.asarray(pin_J, np.float64), R, True)  # reframed
        assert np.allclose(nat_p, exp_p, rtol=2e-3, atol=2e-2), \
            f"com p_com mjx != invariant (B={B}): max|d|={np.abs(nat_p-exp_p).max():.3e}"
        assert np.allclose(nat_J, exp_J, rtol=2e-3, atol=2e-2), \
            f"com J_com mjx != oracle (B={B}): max|d|={np.abs(nat_J-exp_J).max():.3e}"
    # non-triviality: J_com base-linear columns differ from the raw pin frame.
    _, raw_J = h.com(qpos)
    assert np.abs(nat_J[0, :, :3] - np.asarray(raw_J, np.float64)[0, :, :3]).max() > 1e-2


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_ccrba_matches_host_oracle(go2_floating):
    """ccrba: A is column-REFRAMED (A_mjx = A_pin G^{-1}); h = A·qd is frame-INVARIANT."""
    from grid_rbd import _mujoco as bm
    h = go2_floating
    assert h._runner.has_ccrba_mujoco, \
        "floating-base .so is missing grid_rbd_ccrba_mujoco"
    rng = np.random.default_rng(9)
    for B in (1, 4):
        qpos, qvel, _ = _rand_state(h, rng, B, with_qd=True)
        q_pin, qd_pin, _, _, R = h._mjx_inputs(qpos, qvel)
        nat_A, nat_h = h.mujoco.ccrba(qpos, qvel)
        nat_A = np.asarray(nat_A, np.float64); nat_h = np.asarray(nat_h, np.float64)
        pin_A, pin_h = h.ccrba(q_pin, qd_pin)
        exp_A = bm.jacobian_pin_to_mjx(np.asarray(pin_A, np.float64), R, True)  # reframed
        exp_h = np.asarray(pin_h, np.float64)                                   # invariant
        assert np.allclose(nat_A, exp_A, rtol=2e-3, atol=2e-2), \
            f"ccrba A mjx != oracle (B={B}): max|d|={np.abs(nat_A-exp_A).max():.3e}"
        assert np.allclose(nat_h, exp_h, rtol=2e-3, atol=2e-2), \
            f"ccrba h mjx != invariant (B={B}): max|d|={np.abs(nat_h-exp_h).max():.3e}"
    # non-triviality: A base-linear columns differ from the raw pin frame.
    raw_A, _ = h.ccrba(qpos, qvel)
    assert np.abs(nat_A[0, :, :3] - np.asarray(raw_A, np.float64)[0, :, :3]).max() > 1e-2


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
@pytest.mark.parametrize("method", ["kinetic_energy_regressor", "potential_energy_regressor"])
def test_native_mjx_energy_regressor_invariant_but_inputs_converted(go2_floating, method):
    """energy regressors are frame-INVARIANT, but the mjx inputs MUST be converted
    (quat reorder + qd reframe). native(mjx) == pin(pin); raw-mjx-into-pin differs."""
    h = go2_floating
    assert getattr(h._runner, f"has_{method}_mujoco"), \
        f"floating-base .so is missing grid_rbd_{method}_mujoco"
    rng = np.random.default_rng(10)
    with_qd = method == "kinetic_energy_regressor"
    for B in (1, 4):
        qpos, qvel, _ = _rand_state(h, rng, B, with_qd=with_qd)
        q_pin, qd_pin, _, _, _ = h._mjx_inputs(qpos, qvel)
        if with_qd:
            native = np.asarray(getattr(h.mujoco, method)(qpos, qvel), np.float64)
            pin = np.asarray(getattr(h, method)(q_pin, qd_pin), np.float64)
            raw_wrong = np.asarray(getattr(h, method)(qpos, qvel), np.float64)
        else:
            native = np.asarray(getattr(h.mujoco, method)(qpos), np.float64)
            pin = np.asarray(getattr(h, method)(q_pin), np.float64)
            raw_wrong = np.asarray(getattr(h, method)(qpos), np.float64)
        assert np.allclose(native, pin, rtol=2e-3, atol=2e-2), \
            f"{method} mjx != pin-invariant (B={B}): max|d|={np.abs(native-pin).max():.3e}"
    # input convert is load-bearing: raw mjx q (wxyz) into the pin kernel mis-builds.
    assert np.abs(native[0] - raw_wrong[0]).max() > 1e-3


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_cmm_time_variation_matches_host_oracle(go2_floating):
    """column-reframe class with qd input: Adot_mjx = Adot_pin G^{-1} (base cols)."""
    from grid_rbd import _mujoco as bm
    h = go2_floating
    assert h._runner.has_cmm_time_variation_mujoco, \
        "floating-base .so is missing grid_rbd_cmm_time_variation_mujoco"
    rng = np.random.default_rng(11)
    for B in (1, 4):
        qpos, qvel, _ = _rand_state(h, rng, B, with_qd=True)
        q_pin, qd_pin, _, _, R = h._mjx_inputs(qpos, qvel)
        native = np.asarray(h.cmm_time_variation(qpos, qvel, _convention="mujoco"), np.float64)  # (B,6,NV)
        pin_Ad = np.asarray(h.cmm_time_variation(q_pin, qd_pin), np.float64)
        expected = bm.jacobian_pin_to_mjx(pin_Ad, R, True)
        assert np.allclose(native, expected, rtol=2e-3, atol=2e-2), \
            f"cmm_time_variation mjx != oracle (B={B}): max|d|={np.abs(native-expected).max():.3e}"
    # non-triviality: base-linear columns differ from the raw pin frame.
    raw_Ad = np.asarray(h.cmm_time_variation(qpos, qvel), np.float64)
    assert np.abs(native[0, :, :3] - raw_Ad[0, :, :3]).max() > 1e-2


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_end_effector_pose_invariant_but_quat_reordered(go2_floating):
    """end_effector_pose value is frame-INVARIANT, but the mjx q (wxyz) must be
    reordered before the kinematics build (latent-bug path like osc_inertia). So
    native(q_mjx) equals pin(q_pin), and BOTH differ from feeding the raw mjx q
    to the pin kernel."""
    h = go2_floating
    assert h._runner.has_end_effector_pose_mujoco, \
        "floating-base .so is missing grid_rbd_end_effector_pose_mujoco"
    rng = np.random.default_rng(12)
    for B in (1, 4):
        qpos, _, _ = _rand_state(h, rng, B, with_qd=False)
        q_pin, _, _, _, _ = h._mjx_inputs(qpos)
        native = np.asarray(h.end_effector_pose(qpos, _convention="mujoco"), np.float64)
        pin = np.asarray(h.end_effector_pose(q_pin), np.float64)   # invariant -> equal
        assert np.allclose(native, pin, rtol=2e-3, atol=2e-2), \
            f"end_effector_pose mjx != pin-invariant (B={B}): max|d|={np.abs(native-pin).max():.3e}"
    # feeding the raw mjx (wxyz) q to the PIN kernel mis-builds the kinematics -> differs,
    # confirming the quaternion reorder is load-bearing.
    raw_wrong = np.asarray(h.end_effector_pose(qpos), np.float64)
    assert np.abs(native[0] - raw_wrong[0]).max() > 1e-3


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_end_effector_pose_gradient_matches_host_oracle(go2_floating):
    """column-reframe class: J_pose_mjx = J_pose_pin G^{-1} (base-linear cols)."""
    from grid_rbd import _mujoco as bm
    h = go2_floating
    assert h._runner.has_end_effector_pose_gradient_mujoco, \
        "floating-base .so is missing grid_rbd_end_effector_pose_gradient_mujoco"
    rng = np.random.default_rng(13)
    for B in (1, 4):
        qpos, _, _ = _rand_state(h, rng, B, with_qd=False)
        q_pin, _, _, _, R = h._mjx_inputs(qpos)
        native = np.asarray(h.end_effector_pose_gradient(qpos, _convention="mujoco"), np.float64)  # (B,6*NEE,NV)
        pin_J = np.asarray(h.end_effector_pose_gradient(q_pin), np.float64)
        expected = bm.jacobian_pin_to_mjx(pin_J, R, True)
        assert np.allclose(native, expected, rtol=2e-3, atol=2e-2), \
            f"ee_pose_gradient mjx != oracle (B={B}): max|d|={np.abs(native-expected).max():.3e}"
    # non-triviality: base-linear columns differ from the raw pin frame.
    raw_J = np.asarray(h.end_effector_pose_gradient(qpos), np.float64)
    assert np.abs(native[0, :, :3] - raw_J[0, :, :3]).max() > 1e-2


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_integrator_retract(go2_floating):
    """RETRACT class: the MuJoCo free-joint integrator takes a GLOBAL additive base
    position step (pos += dt*v) instead of pinocchio's SE(3) update; the joints
    integrate identically. Validates the defining mjx property + joints-match-pin +
    non-triviality vs the pin SE(3) base step."""
    h = go2_floating
    assert h._runner.has_integrator_mujoco
    nq, nv = h.num_joints, h.num_vel
    # dt large enough that the mjx GLOBAL base step is distinguishable from pin's
    # SE(3) step (they agree to O(dt^2), ~1e-2 here) — so assertion (1) genuinely
    # confirms the global-add rule rather than coincidentally matching SE(3).
    dt = 0.1
    rng = np.random.default_rng(6)
    for B in (1, 4):
        qpos, qvel, u = _rand_state(h, rng, B, with_qd=True, with_u=True)
        x_next = np.asarray(h.integrator(qpos, qvel, u, dt, _convention="mujoco"), np.float64)
        q_next = x_next[:, :nq]
        # (1) defining mjx retract: base position is a GLOBAL additive step.
        exp_pos = qpos[:, :3] + dt * qvel[:, :3]
        assert np.allclose(q_next[:, :3], exp_pos, rtol=2e-3, atol=2e-3), \
            f"integrator base pos != global add (B={B}): max|d|={np.abs(q_next[:, :3]-exp_pos).max():.3e}"
        # (2) the base quaternion stays a unit quaternion (wxyz).
        assert np.allclose(np.linalg.norm(q_next[:, 3:7], axis=1), 1.0, atol=1e-4)
        # (3) joints integrate identically to the pin integrator on pin-converted inputs.
        q_pin, qd_pin, _, u_pin, _ = h._mjx_inputs(qpos, qvel, u=u)
        pin_x = np.asarray(h.integrator(q_pin, qd_pin, u_pin, dt), np.float64)
        assert np.allclose(q_next[:, 7:], pin_x[:, 7:nq], rtol=2e-3, atol=2e-3), \
            f"integrator joints != pin (B={B}): max|d|={np.abs(q_next[:, 7:]-pin_x[:, 7:nq]).max():.3e}"
        # (4) non-triviality: the mjx base position differs from pin's SE(3) base step.
        assert np.abs(q_next[0, :3] - pin_x[0, :3]).max() > 1e-4


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_inverse_dynamics_gradient_matches_oracle(go2_floating):
    """FULL-GRADIENT class: dc/d(q,qd) mjx = reframe + base-row rotate + ω×v
    couplings (M from in-kernel crba reuse), validated vs the RBDReference oracle."""
    from RBDReference.equivalents.mujoco_convention import (
        id_gradient_pin_to_mjx, FloatingRootLayout)
    h = go2_floating
    assert h._runner.has_inverse_dynamics_gradient_mujoco
    nq, nv = h.num_joints, h.num_vel
    layout = FloatingRootLayout()
    rng = np.random.default_rng(7)
    for B in (1, 4):
        qpos, qvel, _ = _rand_state(h, rng, B, with_qd=True)
        qacc = rng.standard_normal((B, nq)); qacc[:, nv:] = 0.0
        native = np.asarray(h.inverse_dynamics_gradient(qpos, qvel, qacc, _convention="mujoco"),
                            np.float64)  # (B, NV, 2NV)
        q_pin, qd_pin, qdd_pin, _, R = h._mjx_inputs(qpos, qvel, qacc)
        pin_g = np.asarray(h.inverse_dynamics_gradient(q_pin, qd_pin, qdd_pin), np.float64)
        M = np.asarray(h.crba(q_pin), np.float64)                       # (B, NV, NV)
        tau = np.asarray(h.inverse_dynamics(q_pin, qd_pin, qdd_pin), np.float64)  # (B, NJ)
        exp = np.empty_like(native)
        for b in range(B):
            dq = pin_g[b, :, :nv]; dqd = pin_g[b, :, nv:]
            o_dq, o_dqd = id_gradient_pin_to_mjx(
                dq, dqd, M[b], tau[b, :nv], qd_pin[b, :nv], qdd_pin[b, :nv], R[b], layout)
            exp[b] = np.concatenate([o_dq, o_dqd], axis=-1)
        assert np.allclose(native, exp, rtol=5e-3, atol=5e-2), \
            f"id-grad mjx != oracle (B={B}): max|d|={np.abs(native-exp).max():.3e}"
    # non-triviality: the mjx gradient differs from feeding raw mjx q to the pin grad.
    raw = np.asarray(h.inverse_dynamics_gradient(qpos, qvel, qacc), np.float64)
    assert np.abs(native - raw).max() > 1e-2


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_forward_dynamics_gradient_matches_oracle(go2_floating):
    """FULL-GRADIENT class: dqdd/d(q,qd) mjx = reframe + base-row rotate + ω×v
    couplings (qdd computed in-kernel), validated vs the RBDReference oracle."""
    from RBDReference.equivalents.mujoco_convention import (
        fd_gradient_pin_to_mjx, FloatingRootLayout)
    h = go2_floating
    assert h._runner.has_forward_dynamics_gradient_mujoco
    nq, nv = h.num_joints, h.num_vel
    layout = FloatingRootLayout()
    rng = np.random.default_rng(11)
    for B in (1, 4):
        qpos, qvel, u = _rand_state(h, rng, B, with_qd=True, with_u=True)
        native = np.asarray(h.forward_dynamics_gradient(qpos, qvel, u, _convention="mujoco"),
                            np.float64)  # (B, NV, 2NV)
        q_pin, qd_pin, _, u_pin, R = h._mjx_inputs(qpos, qvel, u=u)
        pin_g = np.asarray(h.forward_dynamics_gradient(q_pin, qd_pin, u_pin), np.float64)
        Minv = np.asarray(h.minv(q_pin), np.float64)                    # (B, NV, NV)
        qdd = np.asarray(h.forward_dynamics(q_pin, qd_pin, u_pin), np.float64)  # (B, NJ)
        exp = np.empty_like(native)
        for b in range(B):
            dq = pin_g[b, :, :nv]; dqd = pin_g[b, :, nv:]
            o_dq, o_dqd = fd_gradient_pin_to_mjx(
                dq, dqd, Minv[b], qdd[b, :nv], qd_pin[b, :nv], u_pin[b, :nv], R[b], layout)
            exp[b] = np.concatenate([o_dq, o_dqd], axis=-1)
        assert np.allclose(native, exp, rtol=5e-3, atol=5e-2), \
            f"fd-grad mjx != oracle (B={B}): max|d|={np.abs(native-exp).max():.3e}"
    # non-triviality: the mjx gradient differs from feeding raw mjx q to the pin grad.
    raw = np.asarray(h.forward_dynamics_gradient(qpos, qvel, u), np.float64)
    assert np.abs(native - raw).max() > 1e-2


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_generalized_gravity_matches_oracle(go2_floating):
    """g(q) transforms like a generalized force (tau): the base rows are rotated
    into the mjx frame; output shape is invariant (B, NV). Validated vs the
    RBDReference id-force oracle."""
    from grid_rbd import _mujoco as bm
    h = go2_floating
    assert h._runner.has_generalized_gravity_mujoco
    rng = np.random.default_rng(13)
    for B in (1, 4):
        qpos, _, _ = _rand_state(h, rng, B, with_qd=False)
        native = np.asarray(h.generalized_gravity(qpos, _convention="mujoco"), np.float64)  # (B, NV)
        q_pin, _, _, _, R = h._mjx_inputs(qpos)
        g_pin = np.asarray(h.generalized_gravity(q_pin), np.float64)
        exp = bm.id_tau_pin_to_mjx(g_pin, R, True)  # g transforms like a generalized force
        assert np.allclose(native, exp, rtol=5e-3, atol=5e-2), \
            f"gravity mjx != oracle (B={B}): max|d|={np.abs(native-exp).max():.3e}"
    # non-triviality: the mjx base rows differ from the raw pin gravity.
    assert np.abs(native[:, :6] - g_pin[:, :6]).max() > 1e-2


@pytest.mark.skipif(not _has_cuda(), reason="needs nvcc + CUDA GPU")
@pytest.mark.skipif(not _GO2.exists(), reason="go2.urdf asset missing")
def test_native_mjx_nonlinear_effects_matches_oracle(go2_floating):
    """ACCEL-COUPLE class: mjx qfrc_bias = base-rotate(nle_pin + M·δa), δa carries the
    floating-root −ω×v on the base-linear block (injected via a zeroed s_qdd in-kernel).
    Validated vs the RBDReference oracle. Also asserts the floating PIN bias value did
    NOT regress when the codegen switched it to the qdd-input path: cross-checked vs
    C(q,qd)·qd + g(q) from the INDEPENDENT coriolis_matrix + generalized_gravity
    kernels (neither is the bias kernel)."""
    from RBDReference.equivalents.mujoco_convention import (
        nonlinear_effects_pin_to_mjx, FloatingRootLayout)
    h = go2_floating
    assert h._runner.has_nonlinear_effects_mujoco
    nv = h.num_vel
    layout = FloatingRootLayout()
    rng = np.random.default_rng(17)
    for B in (1, 4):
        qpos, qvel, _ = _rand_state(h, rng, B, with_qd=True)
        native = np.asarray(h.nonlinear_effects(qpos, qvel, _convention="mujoco"), np.float64)  # (B, NV)
        q_pin, qd_pin, _, _, R = h._mjx_inputs(qpos, qvel)
        nle_pin = np.asarray(h.nonlinear_effects(q_pin, qd_pin), np.float64)   # (B, NV)
        M = np.asarray(h.crba(q_pin), np.float64)                             # (B, NV, NV)
        # PIN-VALUE REGRESSION GUARD: nle_pin == C·qd + g via independent kernels.
        C = np.asarray(h.coriolis_matrix(q_pin, qd_pin), np.float64)          # (B, NV, NV)
        g = np.asarray(h.generalized_gravity(q_pin), np.float64)             # (B, NV)
        cqd_g = np.einsum("bij,bj->bi", C, qd_pin[:, :nv]) + g
        assert np.allclose(nle_pin, cqd_g, rtol=5e-3, atol=5e-2), \
            f"floating PIN nle regressed (B={B}): max|d|={np.abs(nle_pin-cqd_g).max():.3e}"
        exp = np.empty_like(native)
        for b in range(B):
            exp[b] = nonlinear_effects_pin_to_mjx(nle_pin[b], M[b], qd_pin[b, :nv], R[b], layout)
        assert np.allclose(native, exp, rtol=5e-3, atol=5e-2), \
            f"nle mjx != oracle (B={B}): max|d|={np.abs(native-exp).max():.3e}"
    # non-triviality: the mjx bias differs from feeding raw mjx q/qd to the pin bias.
    raw = np.asarray(h.nonlinear_effects(qpos, qvel), np.float64)
    assert np.abs(native - raw).max() > 1e-2

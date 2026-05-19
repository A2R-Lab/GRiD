"""Any-thread-count correctness tests for the v2.0 codegen.

After the v2.0 cuBLASDx removal, kernels no longer carry
``__launch_bounds__(SUGGESTED_THREADS)`` and the SIMT GLASS helpers
use block-stride loops, so any block size that fits per-block (≤1024
on current GPUs) should produce numerically identical results.

This test sweeps block sizes {64, 128, 256, SUGGESTED_THREADS, 512}
and verifies every bound RobotHandle method on iiwa14_fixed produces
results within float32 tolerance of the SUGGESTED_THREADS reference.

Block size 32 (single warp) is not included because the EE-pose-Hessian
emission expects at least 2 warps for the 4*NUM_EES tensor write
parallelism. Block sizes >512 are skipped to keep the test fast.

Run with:
    pytest test/python_wrappers/test_any_thread_count.py -m python_wrappers -v
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


# ─── skip preconditions ─────────────────────────────────────────────────────

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")

_URDF = (
    Path.home()
    / ".cache/robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF fixture not present at {_URDF}", allow_module_level=True)

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH; grid-rbd register_robot requires it", allow_module_level=True)


pytestmark = pytest.mark.python_wrappers

# Tolerance vs the SUGGESTED_THREADS reference. Same algorithms, same
# float32 precision; differences are limited to FP-summation ordering in
# the block-stride loops when blockDim changes. Empirically ≤1e-5 on
# iiwa14 across all bound methods.
_TOL = 5e-5


# ─── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def handle():
    return _grid_rbd.register_robot(
        name="iiwa14_any_thread_count",
        urdf_path=str(_URDF),
        floating_base=False,
        max_batch_size=8,
    )


@pytest.fixture(scope="module")
def samples(handle):
    rng = np.random.default_rng(0)
    NJ = handle.num_joints
    B = 4
    return {
        "q":  rng.standard_normal((B, NJ)).astype(np.float32),
        "qd": rng.standard_normal((B, NJ)).astype(np.float32),
        "u":  rng.standard_normal((B, NJ)).astype(np.float32),
    }


@pytest.fixture(scope="module")
def reference(handle, samples):
    """Run every method once at SUGGESTED_THREADS (the codegen-time default).

    This is our reference oracle; the parametrized tests below compare
    against these values at varying block sizes.
    """
    # Reset to default explicitly in case a prior test left a different setting.
    handle.set_threads_per_block(handle.suggested_threads)
    q, qd, u = samples["q"], samples["qd"], samples["u"]
    return {
        "rnea":                          handle.rnea(q, qd),
        "minv":                          handle.minv(q),
        "forward_dynamics":              handle.forward_dynamics(q, qd, u),
        "aba":                           handle.aba(q, qd, u),
        "crba":                          handle.crba(q),
        "end_effector_pose":             handle.end_effector_pose(q),
        "end_effector_pose_gradient":    handle.end_effector_pose_gradient(q),
        "end_effector_pose_hessian":     handle.end_effector_pose_hessian(q),
        "rnea_grad":                     handle.rnea_grad(q, qd),
        "forward_dynamics_grad":         handle.forward_dynamics_grad(q, qd, u),
        "idsva_so":                      handle.idsva_so(q, qd),
        "fdsva_so":                      handle.fdsva_so(q, qd, u),
    }


# ─── tests ──────────────────────────────────────────────────────────────────


_BLOCK_SIZES = [64, 128, 256, 512]


def _call_method(handle, method, samples):
    q, qd, u = samples["q"], samples["qd"], samples["u"]
    return {
        "rnea":                          lambda: handle.rnea(q, qd),
        "minv":                          lambda: handle.minv(q),
        "forward_dynamics":              lambda: handle.forward_dynamics(q, qd, u),
        "aba":                           lambda: handle.aba(q, qd, u),
        "crba":                          lambda: handle.crba(q),
        "end_effector_pose":             lambda: handle.end_effector_pose(q),
        "end_effector_pose_gradient":    lambda: handle.end_effector_pose_gradient(q),
        "end_effector_pose_hessian":     lambda: handle.end_effector_pose_hessian(q),
        "rnea_grad":                     lambda: handle.rnea_grad(q, qd),
        "forward_dynamics_grad":         lambda: handle.forward_dynamics_grad(q, qd, u),
        "idsva_so":                      lambda: handle.idsva_so(q, qd),
        "fdsva_so":                      lambda: handle.fdsva_so(q, qd, u),
    }[method]()


def _max_abs_err(a, b):
    """Handle scalar-output, single-array, and tuple-of-arrays cases."""
    if isinstance(a, tuple):
        return max(np.max(np.abs(ai - bi)) for ai, bi in zip(a, b))
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def test_set_threads_per_block_default(handle):
    """Default threads_per_block should equal suggested_threads."""
    handle.set_threads_per_block(handle.suggested_threads)
    assert handle.threads_per_block == handle.suggested_threads


def test_set_threads_per_block_rejects_zero(handle):
    with pytest.raises((ValueError, RuntimeError)):
        handle.set_threads_per_block(0)


def test_set_threads_per_block_rejects_negative(handle):
    with pytest.raises((ValueError, RuntimeError)):
        handle.set_threads_per_block(-1)


@pytest.mark.parametrize("threads", _BLOCK_SIZES)
@pytest.mark.parametrize("method", [
    "rnea", "minv", "forward_dynamics", "aba", "crba",
    "end_effector_pose", "end_effector_pose_gradient", "end_effector_pose_hessian",
    "rnea_grad", "forward_dynamics_grad",
    "idsva_so", "fdsva_so",
])
def test_method_at_block_size(handle, samples, reference, threads, method):
    """Every method must produce results within float32 tolerance of the
    SUGGESTED_THREADS reference at every block size in the sweep."""
    handle.set_threads_per_block(threads)
    try:
        actual = _call_method(handle, method, samples)
        err = _max_abs_err(actual, reference[method])
        assert err < _TOL, (
            f"{method} at threads={threads}: max_abs_err={err:.3e} > {_TOL:.3e}"
        )
    finally:
        # Restore default so subsequent tests see a clean state.
        handle.set_threads_per_block(handle.suggested_threads)

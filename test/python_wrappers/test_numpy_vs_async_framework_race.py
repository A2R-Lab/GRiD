"""numpy path vs in-flight framework work on the SAME artifact (audit W04-B class,
found by the 2026-09-19 receipt run: test_jax_f_ext_parity_vs_numpy failed under
the compile-pool load with a dense mismatch — the numpy inverse_dynamics kernel
ran against a d_f_ext the jax handler's trailing async memset had just zeroed).

The jax/torch handlers enqueue H2D copies, the kernel and the f_ext reset on
their own (non-blocking) streams into the shared g_data buffers; the numpy path
stages synchronously on the legacy default stream. Without draining in-flight
framework work first, interleaving an un-materialized jax call with a numpy call
is a race. Load-dependent (2/20 failures under two concurrent codegen jobs before
the fix), so the loop below is long and interleaves aggressively; it is a
correctness net, not a timing benchmark.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import robot_urdf  # noqa: E402

grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH", allow_module_level=True)
_IIWA = robot_urdf("iiwa14")
if not _IIWA.exists():
    pytest.skip(f"iiwa14 URDF not present at {_IIWA}", allow_module_level=True)
pytestmark = pytest.mark.python_wrappers
_ALGOS = ["inverse_dynamics", "forward_dynamics"]


@pytest.fixture(scope="module")
def pair():
    jax = pytest.importorskip("jax")
    np_h = grid_rbd.register_robot("w04b_race_np", str(_IIWA), algorithm_list=_ALGOS)
    jx_h = grid_rbd.register_robot("w04b_race_jax", str(_IIWA), backend="jax", algorithm_list=_ALGOS)
    yield np_h, jx_h
    jx_h.close(); np_h.close()


def test_numpy_call_is_not_clobbered_by_in_flight_jax_work(pair):
    np_h, jx_h = pair
    rng = np.random.default_rng(20260920)
    B, nq, nb = 16, np_h.num_joints, np_h.num_bodies
    q = rng.uniform(-1, 1, (B, nq)).astype(np.float32)
    qd = rng.uniform(-1, 1, (B, nq)).astype(np.float32)
    f_ext = (0.5 * rng.standard_normal((B, 6 * nb))).astype(np.float32)
    ref = np.asarray(np_h.inverse_dynamics(q, qd, f_ext=f_ext))         # quiet-box oracle
    ref0 = np.asarray(np_h.inverse_dynamics(q, qd))
    worst = 0.0
    for i in range(200):
        pending = [jx_h.inverse_dynamics(q, qd, f_ext=f_ext) for _ in range(4)]   # enqueued, NOT materialized
        out = np.asarray(np_h.inverse_dynamics(q, qd, f_ext=f_ext))               # numpy path must drain first
        worst = max(worst, float(np.abs(out - ref).max()))
        assert np.abs(out - ref).max() < 5e-3, f"iteration {i}: numpy ID clobbered by in-flight jax work"
        out0 = np.asarray(np_h.inverse_dynamics(q, qd))
        assert np.abs(out0 - ref0).max() < 5e-3, f"iteration {i}: zero-force numpy ID clobbered"
        for p in pending:                                                          # and jax stays right too
            assert np.abs(np.asarray(p) - ref).max() < 5e-3, f"iteration {i}: jax ID wrong"
    assert worst < 5e-3

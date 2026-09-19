"""Shared-runtime ownership across handles of ONE artifact (audit W04 increment A,
2026-09-19).

Every Runner dlopens the robot .so; the .so owns ONE runtime (g_data / g_robot /
streams / the runtime parameter tables). Two handles on the same artifact share
it. Before the fix each Runner's destructor called grid_rbd_close()
unconditionally, so closing handle B freed the runtime handle A was using: A's
next call silently re-initialized (baked defaults back, live inertia updates
lost). The runtime is now reference-counted per .so path inside the extension:
it closes when the LAST owner goes away, close() is idempotent, and a fresh
registration after everyone closed re-initializes cleanly.
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
_KW = dict(floating_base=False, runtime_inertia=True, algorithm_list=["inverse_dynamics"])


def _register(name):
    return grid_rbd.register_robot(name, str(_IIWA), **_KW)


def _tau(h, q, qd):
    return np.asarray(h.inverse_dynamics(q, qd))


def test_sibling_close_does_not_reset_the_shared_runtime():
    a = _register("w04_lifetime_a")
    b = _register("w04_lifetime_b")          # same artifact -> same .so -> same runtime
    rng = np.random.default_rng(20260919)
    q = rng.uniform(-1, 1, (2, a.num_joints)).astype(np.float32)
    qd = rng.uniform(-1, 1, (2, a.num_joints)).astype(np.float32)
    baked = _tau(a, q, qd)
    params = np.array(a.inertia_params, dtype=np.float64)
    params[:, 0] *= 2.0                      # double every mass
    a.set_inertia_params(params)
    doubled = _tau(a, q, qd)
    assert np.abs(doubled - baked).max() > 1e-3
    np.testing.assert_allclose(_tau(b, q, qd), doubled)   # shared table: B sees A's update

    b.close()                                # sibling goes away ...
    b.close()                                # ... idempotently
    np.testing.assert_allclose(_tau(a, q, qd), doubled,   # ... and A keeps its live runtime
                               err_msg="closing a sibling handle reset the shared runtime")
    a.close()
    # everyone closed: a fresh registration re-initializes with BAKED params
    c = _register("w04_lifetime_c")
    np.testing.assert_allclose(_tau(c, q, qd), baked)
    c.close()


def test_reverse_close_order_and_gc():
    import gc
    a = _register("w04_lifetime_a2")
    b = _register("w04_lifetime_b2")
    q = np.zeros((1, a.num_joints), np.float32); qd = np.zeros_like(q)
    ref = _tau(a, q, qd)
    a.close()                                # the FIRST-opened handle closes first
    np.testing.assert_allclose(_tau(b, q, qd), ref)
    del a; gc.collect()
    np.testing.assert_allclose(_tau(b, q, qd), ref)
    b.close()

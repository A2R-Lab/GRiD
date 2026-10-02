"""Allocate-once host round trip on the numpy surface (2026-10-01).

`handle.pinned_empty(shape)` + `out=` on the first-order gradients and idsva_so /
fdsva_so: the C ABI retargets the
generated host wrapper's D2H copy at the caller's buffer (GridMirrorRetarget), so no
host-side memcpy follows and a page-locked buffer receives the copy at the PCIe rate.
Contract pinned here: bit-identical to the default call, the returned tensors are VIEWS
of `out`, `out` works pinned or pageable, the mirror is restored after the call (a later
default call is unaffected), and bad `out` arguments are refused with clear messages.
Skips when grid_rbd / CUDA / nvcc are unavailable.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from config import robot_urdf  # noqa: E402

_grid_rbd = pytest.importorskip("grid_rbd", reason="grid-rbd not installed")
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH", allow_module_level=True)
_URDF = robot_urdf("iiwa14")
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF fixture not present at {_URDF}", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers


@pytest.fixture(scope="module")
def h():
    return _grid_rbd.register_robot(name="iiwa14_host_transfer_numpy", urdf_path=str(_URDF),
                                    floating_base=False, max_batch_size=8)


@pytest.fixture(scope="module")
def inputs(h):
    rng = np.random.default_rng(3)
    B = 8
    return (rng.standard_normal((B, h.num_joints)).astype(np.float32),
            rng.standard_normal((B, h.num_vel)).astype(np.float32),
            rng.standard_normal((B, h.num_vel)).astype(np.float32))


def _flat(tensors):
    return np.concatenate([np.asarray(t).ravel() for t in tensors])


def test_pinned_empty_is_page_locked_and_in_the_compute_dtype(h):
    buf = h.pinned_empty((8, 4 * h.num_vel ** 3))
    assert buf.shape == (8, 4 * h.num_vel ** 3) and buf.dtype == np.float32
    assert h.is_pinned(buf) and not h.is_pinned(np.empty(4, np.float32))
    with pytest.raises(ValueError, match="computes in float32"):
        h.pinned_empty((2, 2), dtype=np.float64)


@pytest.mark.parametrize("method", ["idsva_so", "fdsva_so"])
@pytest.mark.parametrize("pinned", [True, False], ids=["pinned", "pageable"])
def test_out_receives_the_result_as_views_and_matches_default(h, inputs, method, pinned):
    q, qd, x = inputs
    B, n = q.shape[0], 4 * h.num_vel ** 3
    fn = getattr(h, method)
    ref = _flat(fn(q, qd, x))
    out = h.pinned_empty((B, n)) if pinned else np.empty((B, n), np.float32)
    got = fn(q, qd, x, out=out)
    assert np.array_equal(_flat(got), ref)
    # the tensors are VIEWS of out (per-item layout: 4 slabs of nv^3 per batch row)
    assert all(np.shares_memory(t, out) for t in got), "returned tensors must be views of out"
    nv = h.num_vel
    assert all(np.array_equal(out.reshape(B, 4, nv, nv, nv)[:, k], np.asarray(got[k])) for k in range(4))
    # the context's mirror is restored: a default call afterwards is unaffected
    assert np.array_equal(_flat(fn(q * 0.5, qd, x)), _flat(fn(q * 0.5, qd, x, out=out)))


@pytest.mark.parametrize("method", ["inverse_dynamics_gradient", "forward_dynamics_gradient"])
@pytest.mark.parametrize("pinned", [True, False], ids=["pinned", "pageable"])
def test_gradient_out_is_a_view_in_the_public_layout_and_matches_default(h, inputs, method, pinned):
    q, qd, x = inputs
    B, nv = q.shape[0], h.num_vel
    fn = getattr(h, method)
    ref = fn(q, qd, x)
    out = h.pinned_empty((B, 2 * nv * nv)) if pinned else np.empty((B, 2 * nv * nv), np.float32)
    out[:] = np.nan                                             # every element must be written
    got = fn(q, qd, x, out=out)
    assert got.shape == ref.shape == (B, nv, 2 * nv)
    assert np.array_equal(got, ref) and np.shares_memory(got, out)
    # raw layout of `out`: one column-major nv x 2nv matrix per item
    assert np.array_equal(out.reshape(B, 2 * nv, nv).transpose(0, 2, 1), ref)
    # mirror restored; reusing the buffer overwrites the view in place
    assert np.array_equal(fn(q * 0.5, qd, x), fn(q * 0.5, qd, x, out=out))
    assert np.array_equal(got, fn(q * 0.5, qd, x)), "the view tracks the reused buffer"


def test_direct_download_ops_fill_every_element(h, inputs):
    """minv / crba / the gradients lost their second device download (the wrapper's own
    D2H is retargeted at the result array): the default calls still return full, finite,
    repeatable results."""
    q, qd, x = inputs
    for call in (lambda: h.minv(q), lambda: h.crba(q), lambda: h.inverse_dynamics_gradient(q, qd, x),
                 lambda: h.forward_dynamics_gradient(q, qd, x)):
        a, b = call(), call()
        assert np.isfinite(a).all() and np.array_equal(a, b)
    M, Minv = h.crba(q).astype(np.float64), h.minv(q).astype(np.float64)
    assert np.allclose(M @ Minv, np.eye(h.num_vel), atol=5e-3)


def test_bad_out_is_refused(h, inputs):
    q, qd, x = inputs
    B, n = q.shape[0], 4 * h.num_vel ** 3
    with pytest.raises(ValueError, match="shape"):
        h.idsva_so(q, qd, x, out=np.empty((B, n - 1), np.float32))
    with pytest.raises(ValueError, match="dtype"):
        h.idsva_so(q, qd, x, out=np.empty((B, n), np.float64))
    with pytest.raises(ValueError, match="C-contiguous"):
        h.idsva_so(q, qd, x, out=np.empty((n, B), np.float32).T)
    ro = np.empty((B, n), np.float32); ro.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        h.idsva_so(q, qd, x, out=ro)
    nv = h.num_vel
    with pytest.raises(ValueError, match="shape"):               # the public shape is NOT the buffer shape
        h.inverse_dynamics_gradient(q, qd, x, out=np.empty((B, nv, 2 * nv), np.float32))

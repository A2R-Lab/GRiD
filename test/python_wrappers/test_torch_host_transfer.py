"""Allocate-once host round trips on the torch surface (2026-10-01).

`pinned_host_like` / `copy_to_host` / `GraphCallable.replay_into` are the documented
way to get an op's output back to the CPU at the pinned D2H rate instead of `.cpu()`'s
pageable staged copy (measured 6x on 700 MB outputs). These tests pin the CONTRACT:
bit-identical values vs the eager/.cpu() path, page-locked mirrors, tuple outputs, shape
mismatch refused, and the mirrors staying valid across repeated replays.
Skips when torch / CUDA / nvcc are unavailable.
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
torch = pytest.importorskip("torch", reason="torch not installed")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)
if shutil.which("nvcc") is None:
    pytest.skip("nvcc not on PATH", allow_module_level=True)
_URDF = robot_urdf("iiwa14")
if not _URDF.exists():
    pytest.skip(f"iiwa14 URDF fixture not present at {_URDF}", allow_module_level=True)

pytestmark = pytest.mark.python_wrappers

import grid_rbd.torch as gt  # noqa: E402


@pytest.fixture(scope="module")
def th():
    return gt.register_robot(name="iiwa14_host_transfer_torch", urdf_path=str(_URDF),
                             floating_base=False, max_batch_size=8)


@pytest.fixture(scope="module")
def inputs(th):
    rng = np.random.default_rng(1)
    B, nq, nv = 8, th.num_joints, th.num_vel
    mk = lambda w: torch.from_numpy(rng.standard_normal((B, w)).astype(np.float32)).cuda()
    return mk(nq), mk(nv), mk(nv)


def test_pinned_mirror_matches_cpu_for_single_and_tuple_outputs(th, inputs):
    q, qd, u = inputs
    out = th.inverse_dynamics_gradient(q, qd, u)              # single tensor
    so = th.idsva_so(q, qd, u)                                 # tuple of 4
    for dev in (out, so):
        host = gt.pinned_host_like(dev)
        mirrors = host if isinstance(host, tuple) else (host,)
        assert all(m.is_pinned() and m.device.type == "cpu" for m in mirrors)
        got = gt.copy_to_host(host, dev)
        assert got is host
        ref = [t.cpu() for t in (dev if isinstance(dev, tuple) else (dev,))]
        assert all(torch.equal(m, r) for m, r in zip(mirrors, ref))


def test_replay_into_matches_eager_and_reuses_the_mirror(th, inputs):
    q, qd, u = inputs
    g = th.capture("inverse_dynamics_gradient", q, qd, u)
    host = gt.pinned_host_like(g.static_out)
    first = g.replay_into(host)
    assert first is host and torch.equal(host, th.inverse_dynamics_gradient(q, qd, u).cpu())
    # new inputs through the captured graph: the SAME mirror receives the new value
    q2 = q * 0.5
    g.static_in[0].copy_(q2)
    g.replay_into(host)
    assert torch.equal(host, th.inverse_dynamics_gradient(q2, qd, u).cpu())


def test_copy_to_host_refuses_mismatched_tuple(th, inputs):
    q, qd, u = inputs
    so = th.idsva_so(q, qd, u)
    host = gt.pinned_host_like(so[:2])
    with pytest.raises(ValueError, match="mirrors"):
        gt.copy_to_host(host, so)

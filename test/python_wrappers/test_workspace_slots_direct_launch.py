"""Reduced-workspace-slots A/B for the DIRECT-LAUNCH surfaces (jax FFI + torch ops).

test_cuda_workspace_slots.py proves the slot clamp for the standalone CUDA exe
path (generated host wrappers). This is its bindings twin: the jax/torch
handlers in wrapper_template.cu launch kernels DIRECTLY (no host wrapper), so
they carry their own grid clamp (grid_rbd_grid_for). Before 2026-08-09 they
launched dim3(batch) unclamped — when init_gridData fits fewer slots than the
batch (forced here via GRID_WORKSPACE_TIMESTEP_SLOTS), blocks aliased live
workspace and produced silent wrong results.

Each arm runs in a SUBPROCESS because workspace_timestep_slots is fixed at
grid_rbd_init time (env read inside init_gridData). Outputs must be
BIT-IDENTICAL between the full-slots and slots=2 arms: per-timestep work is
independent and fixed-order, so the clamped grid (grid-stride over timesteps)
changes scheduling, not arithmetic.

Covers a workspace-heavy algorithm (idsva_so) and a light one
(inverse_dynamics) on both backends, batch=8 >> slots=2.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
URDF = REPO_ROOT / "config" / "robot_assets" / "iiwa14.urdf"

_CHILD = r"""
import os, sys
import numpy as np

backend = sys.argv[1]
out_path = sys.argv[2]
URDF_PATH = sys.argv[3]
# GRID_WORKSPACE_TIMESTEP_SLOTS is inherited from the parent env (or absent).

import grid_rbd
h = grid_rbd.register_robot(
    "iiwa14_slots_ab", URDF_PATH, backend=backend)

rng = np.random.default_rng(1234)
B = 8
q = rng.uniform(-1.0, 1.0, (B, 7)).astype(np.float32)
qd = rng.uniform(-1.0, 1.0, (B, 7)).astype(np.float32)
u = rng.uniform(-1.0, 1.0, (B, 7)).astype(np.float32)

if backend == "jax":
    import jax.numpy as jnp
    qj, qdj, uj = jnp.asarray(q), jnp.asarray(qd), jnp.asarray(u)
    so = tuple(np.asarray(t) for t in h.idsva_so(qj, qdj))
    idv = np.asarray(h.inverse_dynamics(qj, qdj, uj))
else:
    import torch
    qt = torch.as_tensor(q, device="cuda")
    qdt = torch.as_tensor(qd, device="cuda")
    ut = torch.as_tensor(u, device="cuda")
    so = tuple(t.cpu().numpy() for t in h.idsva_so(qt, qdt))
    idv = h.inverse_dynamics(qt, qdt, ut).cpu().numpy()

arrays = {f"so{i}": a for i, a in enumerate(so)}
arrays["id"] = idv
np.savez(out_path, **arrays)
h.close()
"""


def _run_arm(backend: str, slots: str | None, out: Path) -> None:
    env = dict(os.environ)
    env.pop("GRID_WORKSPACE_TIMESTEP_SLOTS", None)
    if slots is not None:
        env["GRID_WORKSPACE_TIMESTEP_SLOTS"] = slots
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, backend, str(out), str(URDF)],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, (
        f"{backend} arm (slots={slots}) failed rc={proc.returncode}\n"
        f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )


@pytest.mark.python_wrappers
@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_direct_launch_bit_identical_under_reduced_slots(backend):
    with tempfile.TemporaryDirectory() as td:
        full = Path(td) / "full.npz"
        few = Path(td) / "few.npz"
        _run_arm(backend, None, full)
        _run_arm(backend, "2", few)
        a, b = np.load(full), np.load(few)
        assert sorted(a.files) == sorted(b.files)
        for key in a.files:
            assert a[key].shape == b[key].shape
            assert np.array_equal(a[key], b[key]), (
                f"{backend} {key}: outputs differ between full-slots and "
                f"slots=2 (max abs diff "
                f"{np.max(np.abs(a[key] - b[key])):.3e}) — the direct-launch "
                "grid clamp is broken (workspace aliasing)"
            )

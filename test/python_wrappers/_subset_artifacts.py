"""Shared helpers for the small-subset wrapper modules (runtime contexts, operand
validation, MuJoCo twins): one registration spelling, one state sampler, one
manifest lookup — so the three modules cannot drift (hygiene, 2026-09-24)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "config/robot_assets"


def register_subset(name, urdf, *, floating, algos, max_batch=16, runtime_inertia=False, mujoco=False):
    """A subset artifact keyed by (name, options); identical calls across modules hit the cache."""
    import grid_rbd
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not on PATH")
    return grid_rbd.register_robot(name, str(ASSETS / urdf), floating_base=floating, max_batch_size=max_batch,
                                   algorithm_list=list(algos), enable_mujoco_kernels=mujoco,
                                   runtime_inertia=runtime_inertia)


def cache_key(handle):
    """The manifest's content key (what the jax/torch views key their registrations on)."""
    import grid_rbd
    return grid_rbd.manifest_lookup(grid_rbd.default_cache_dir(), handle._name)["cache_key"]


def random_state(h, B=4, seed=0):
    """(q, qd, u) at the nq stride; floating base gets a unit quaternion and zero pads."""
    rng = np.random.default_rng(seed)
    q = 0.3 * rng.standard_normal((B, h.nq)).astype(np.float32)
    if h.floating_base:
        quat = rng.standard_normal((B, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True); q[:, 3:7] = quat
    qd = 0.3 * rng.standard_normal((B, h.nq)).astype(np.float32); u = 0.3 * rng.standard_normal((B, h.nq)).astype(np.float32)
    if h.floating_base:
        qd[:, h.nv:] = 0; u[:, h.nv:] = 0
    return q, qd, u

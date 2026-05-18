"""grid-rbd — GPU-accelerated rigid body dynamics with a register-then-run UX.

Two-tier user model:

  # One-time per (robot, options, GRiD version, CUDA arch):
  handle = grid_rbd.register_robot(name="iiwa14", urdf_path="iiwa.urdf")

  # Many times after, fast:
  qdd = handle.forward_dynamics(q, qd, u)   # (B, NJ) → (B, NJ)

Registration parses the URDF, generates a per-robot grid.cuh via GRiDCodeGenerator,
compiles a small .so wrapping the generated kernels, and caches the .so under
~/.cache/grid-rbd/. Subsequent registrations of the same robot reuse the cache.

The handle is what you call algorithm methods on. All methods take and return
2D arrays where axis 0 is the batch dimension; batch=1 is fine for single-call
use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ._cache import (
    compute_cache_key,
    default_cache_dir,
    detect_cuda_arch,
    list_registered as _list_registered,
    manifest_lookup,
    manifest_register,
    store_dir,
)
from ._compile import generate_and_compile
from ._handle import RobotHandle


__version__ = "0.1.0"


class RobotNotRegisteredError(KeyError):
    """Raised by get_robot() when no robot under that name has been registered."""

    def __init__(self, name: str):
        super().__init__(
            f"No robot registered under name {name!r}. Call "
            f"grid_rbd.register_robot(name={name!r}, urdf_path=...) first."
        )
        self.name = name


# ─── public API ──────────────────────────────────────────────────────────────


def register_robot(
    name: str,
    urdf_path: str,
    *,
    floating_base: bool = False,
    max_batch_size: int = 256,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
    cuda_arch: int | None = None,
) -> RobotHandle:
    """Register a robot for fast subsequent calls.

    Generates grid.cuh from the URDF, compiles a per-robot .so, and caches
    it under cache_dir (default ~/.cache/grid-rbd/). Idempotent: if a cache
    entry matching (urdf, options, grid_rbd version, cuda_arch) already
    exists, the existing .so is reused — no recompile.

    Parameters
    ----------
    name : str
        Human-friendly handle name. Re-registering under the same name with
        a different URDF overwrites the binding (the old .so lingers in the
        cache for manual GC).
    urdf_path : str
        Path to the robot's URDF file.
    floating_base : bool, optional
        Treat the robot as floating-base. Default False (fixed-base).
    max_batch_size : int, optional
        Compile-time max batch size. Calls with batch <= this run on a
        single launch; larger batches must be chunked by the caller (a
        helper will be added in v2).
    cache_dir : str | Path | None, optional
        Override the cache root. Default uses platformdirs / $GRID_RBD_CACHE_DIR.
    force_rebuild : bool, optional
        Skip the cache hit check and regenerate + recompile.
    cuda_arch : int | None, optional
        Compute capability as int (e.g. 120 for sm_120). Default detects
        via nvidia-smi.

    Returns
    -------
    RobotHandle
        Ready for forward_dynamics / rnea / minv / etc.
    """
    cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    urdf_p = Path(urdf_path).expanduser().resolve()
    if not urdf_p.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_p}")
    urdf_bytes = urdf_p.read_bytes()

    if cuda_arch is None:
        cuda_arch = detect_cuda_arch()
    if cuda_arch == 0:
        raise RuntimeError(
            "Could not detect CUDA arch via nvidia-smi. "
            "Pass cuda_arch=<int> explicitly (e.g. 120 for sm_120)."
        )

    # Only options that affect generated code go into the cache key.
    code_options = {
        "floating_base": bool(floating_base),
        "max_batch": int(max_batch_size),
    }
    cache_key = compute_cache_key(urdf_bytes, code_options, cuda_arch)
    entry_dir = store_dir(cache_dir, cache_key)
    so_path = entry_dir / "robot.so"

    if force_rebuild or not so_path.exists():
        meta = generate_and_compile(
            urdf_p, code_options, entry_dir,
            cuda_arch=cuda_arch, max_batch=max_batch_size,
        )
    else:
        import json
        meta = json.loads((entry_dir / "meta.json").read_text())

    manifest_register(cache_dir, name, cache_key, meta)
    return RobotHandle(name, str(so_path), meta)


def get_robot(name: str, cache_dir: str | Path | None = None) -> RobotHandle:
    """Look up a previously-registered robot by name.

    Raises RobotNotRegisteredError if `name` isn't in the manifest.
    """
    cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    entry = manifest_lookup(cache_dir, name)
    if entry is None:
        raise RobotNotRegisteredError(name)
    cache_key = entry["cache_key"]
    so_path = store_dir(cache_dir, cache_key) / "robot.so"
    if not so_path.exists():
        raise RuntimeError(
            f"Manifest entry for {name!r} points at {so_path}, but the file "
            f"is missing. Cache is corrupted; re-register with force_rebuild=True."
        )
    return RobotHandle(name, str(so_path), entry)


def list_registered(cache_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Return manifest entries for all registered robots."""
    cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    return _list_registered(cache_dir)


__all__ = [
    "RobotHandle",
    "RobotNotRegisteredError",
    "register_robot",
    "get_robot",
    "list_registered",
    "default_cache_dir",
    "__version__",
]

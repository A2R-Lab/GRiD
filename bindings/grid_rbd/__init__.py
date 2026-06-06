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
from ._handle import RobotHandle, SecondOrderID, SecondOrderFD


__version__ = "0.4.0"


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
    urdf_path: str | None = None,
    *,
    urdf_string: str | None = None,
    floating_base: bool = False,
    ee_joint_names: list[str] | tuple[str, ...] | None = None,
    max_batch_size: int = 256,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
    cuda_arch: int | None = None,
    backend: str = "numpy",
    allow_fp64: bool = False,
) -> RobotHandle:
    """Register a robot for fast subsequent calls.

    Generates grid.cuh from the URDF, compiles a per-robot .so, and caches
    it under cache_dir (default ~/.cache/grid-rbd/). Idempotent: if a cache
    entry matching (urdf, options, grid_rbd version, cuda_arch) already
    exists, the existing .so is reused — no recompile.

    Precision: **all backends compute in float32 only** (no ``dtype=`` knob;
    a true fp64 tier is a future codegen item). For the numpy backend you may
    pass ``allow_fp64=True`` for an fp64-in / fp64-out *convenience* cast on the
    returned handle — compute still runs in fp32 and results are upcast, so the
    single-precision accuracy caveat applies. ``allow_fp64`` is ignored for the
    jax/torch backends (they are strictly fp32).

    Parameters
    ----------
    name : str
        Human-friendly handle name. Re-registering under the same name with
        a different URDF overwrites the binding (the old .so lingers in the
        cache for manual GC).
    urdf_path : str | None
        Path to the robot's URDF file. Mutually exclusive with urdf_string.
    urdf_string : str | None, optional
        Inline URDF text (no file on disk). Mutually exclusive with urdf_path.
        The cache key hashes the URDF bytes, so an inline string and the
        equivalent file dedupe to the same compiled .so. The string is
        persisted as entry_dir/robot.urdf for re-runs / debugging.
    backend : str, optional
        "numpy" (default) → a numpy RobotHandle; "jax" → a JaxRobotHandle
        (grid_rbd.jax); "torch" → a TorchRobotHandle (grid_rbd.torch). The
        jax/torch backends forward to their submodule's register_robot.
    floating_base : bool, optional
        Treat the robot as floating-base. Default False (fixed-base).
    ee_joint_names : list[str] | None, optional
        Names of fixed joints to treat as end-effector targets. Default
        None ⇒ codegen uses all leaf nodes. Currently only the first
        name is honored (single-target codegen); multi-target support is
        a v2 concern. Passing a different list changes the cache key,
        so different target choices land in separate cache entries.
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
        Ready for forward_dynamics / inverse_dynamics / minv / etc.
    """
    if backend not in ("numpy", "jax", "torch"):
        raise ValueError(f"backend must be 'numpy', 'jax', or 'torch'; got {backend!r}")
    if backend == "jax":
        from . import jax as _jax_backend
        return _jax_backend.register_robot(
            name, urdf_path, urdf_string=urdf_string, floating_base=floating_base,
            ee_joint_names=ee_joint_names, max_batch_size=max_batch_size,
            cache_dir=cache_dir, force_rebuild=force_rebuild, cuda_arch=cuda_arch)
    if backend == "torch":
        from . import torch as _torch_backend
        return _torch_backend.register_robot(
            name, urdf_path, urdf_string=urdf_string, floating_base=floating_base,
            ee_joint_names=ee_joint_names, max_batch_size=max_batch_size,
            cache_dir=cache_dir, force_rebuild=force_rebuild, cuda_arch=cuda_arch)

    cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Source the URDF bytes from either an inline string or a file. The cache
    # key hashes these bytes (compute_cache_key), so an inline string and the
    # equivalent file dedupe to the same .so automatically.
    if (urdf_path is None) == (urdf_string is None):
        raise ValueError("pass exactly one of urdf_path= or urdf_string=")
    urdf_p: Path | None = None
    if urdf_string is not None:
        urdf_bytes = urdf_string.encode("utf-8")
    else:
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
        "ee_joint_names": list(ee_joint_names) if ee_joint_names else [],
    }
    cache_key = compute_cache_key(urdf_bytes, code_options, cuda_arch)
    entry_dir = store_dir(cache_dir, cache_key)
    so_path = entry_dir / "robot.so"

    if force_rebuild or not so_path.exists():
        # generate_and_compile takes a Path. For an inline URDF, persist the
        # string under entry_dir/robot.urdf (visible for re-runs / debugging)
        # and pass that path. The cache key is computed from the string bytes,
        # not the path, so the path doesn't leak into the key.
        gen_urdf_path = urdf_p
        if urdf_string is not None:
            entry_dir.mkdir(parents=True, exist_ok=True)
            gen_urdf_path = entry_dir / "robot.urdf"
            gen_urdf_path.write_text(urdf_string)
        meta = generate_and_compile(
            gen_urdf_path, code_options, entry_dir,
            cuda_arch=cuda_arch, max_batch=max_batch_size,
        )
    else:
        import json
        meta = json.loads((entry_dir / "meta.json").read_text())

    manifest_register(cache_dir, name, cache_key, meta)
    return RobotHandle(name, str(so_path), meta, allow_fp64=allow_fp64)


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


def precompile(
    name: str,
    urdf_path: str | None = None,
    *,
    urdf_string: str | None = None,
    tiers: Iterable[dict[str, Any]] | None = None,
    floating_base: bool = False,
    ee_joint_names: list[str] | tuple[str, ...] | None = None,
    max_batch_size: int = 256,
    backends: Iterable[str] = ("numpy",),
    cache_dir: str | Path | None = None,
    cuda_arch: int | None = None,
) -> list[dict[str, Any]]:
    """Ahead-of-time: build + populate the persistent cache for a robot so every
    later ``register_robot`` / ``get_robot`` / ``jax.jit`` is an instant cache hit.

    Each entry in ``tiers`` is a dict of codegen-affecting overrides applied on
    top of the defaults (``floating_base`` / ``ee_joint_names`` /
    ``max_batch_size``) — e.g. ``tiers=[{}, {"floating_base": True}]`` prebuilds
    both the fixed- and floating-base ``.so``. ``tiers=None`` builds the single
    default tier. Each requested ``backend`` ("numpy"/"jax"/"torch") warms that
    surface's artifacts on the same cached ``.so``.

    This is a thin, idempotent driver over :py:func:`register_robot`: a tier
    already in the cache is a no-op (no nvcc); a missing tier compiles once and
    populates the cache. Returns the manifest entry for each (tier, backend)
    built, in order. Build offline once, ship/keep the cache dir, and every
    later run starts in well under a second.
    """
    if tiers is None:
        tiers = [{}]
    else:
        tiers = list(tiers)
    backends = list(backends)
    if not backends:
        raise ValueError("backends must name at least one of 'numpy'/'jax'/'torch'")

    results: list[dict[str, Any]] = []
    cd = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    for i, tier in enumerate(tiers):
        opts = {
            "floating_base": floating_base,
            "ee_joint_names": ee_joint_names,
            "max_batch_size": max_batch_size,
            **dict(tier),
        }
        # Distinct manifest name per tier so multiple tiers under one logical
        # robot don't clobber each other's name binding. Single-tier keeps the
        # plain name so a follow-up get_robot(name) just works.
        tier_name = name if len(tiers) == 1 else f"{name}__tier{i}"
        for backend in backends:
            register_robot(
                name=tier_name,
                urdf_path=urdf_path,
                urdf_string=urdf_string,
                backend=backend,
                cache_dir=cache_dir,
                force_rebuild=False,
                cuda_arch=cuda_arch,
                **opts,
            )
            entry = manifest_lookup(cd, tier_name)
            results.append({"name": tier_name, "backend": backend, **(entry or {})})
    return results


__all__ = [
    "RobotHandle",
    "SecondOrderID",
    "SecondOrderFD",
    "RobotNotRegisteredError",
    "register_robot",
    "get_robot",
    "list_registered",
    "precompile",
    "default_cache_dir",
    "__version__",
]

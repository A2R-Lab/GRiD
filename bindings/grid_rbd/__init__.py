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
    dtype: str = "float32",
    runtime_inertia: bool = False,
    use_joint_dynamics: bool = False,
    output_convention: str = "pinocchio",
    algorithm_list: list[str] | tuple[str, ...] | str | None = None,
) -> RobotHandle:
    """Register a robot for fast subsequent calls.

    Generates grid.cuh from the URDF, compiles a per-robot .so, and caches
    it under cache_dir (default ~/.cache/grid-rbd/). Idempotent: if a cache
    entry matching (urdf, options, grid_rbd version, cuda_arch) already
    exists, the existing .so is reused — no recompile.

    Precision (Phase 8): the numpy backend supports a true fp64 compute tier via
    ``dtype="float64"`` — it builds a SEPARATE .so (``-DGRID_WRAPPER_T_DOUBLE`` +
    the matching codegen knob that re-derives the shared-mem spill tiers at 2×
    bytes) and the handle takes/returns float64 numpy arrays computed end-to-end
    in double precision. fp32 (``dtype="float32"``, default) is unchanged and
    byte-identical. The fp32 and fp64 .so coexist in the cache (dtype is in the
    cache key). NOTE: fp64 doubles every arena's smem footprint, lowering
    occupancy; some big-robot second-order kernels that already max-spill at fp32
    may not fit the device opt-in cap at fp64 — those kernels are left
    unregistered and raise a clear runtime error when called (no new gating).
    The jax/torch backends are strictly fp32 (``dtype="float64"`` is rejected for
    them). ``allow_fp64`` is the LEGACY fp32-compute upcast convenience (compute
    in fp32, cast i/o to fp64); prefer ``dtype="float64"`` for real double
    precision. ``allow_fp64`` is ignored when ``dtype="float64"``.

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
    runtime_inertia : bool, optional
        Build the robot with a runtime-mutable inertia table (D.4 / Phase 5,
        numpy backend only). Default False ⇒ the per-link spatial inertia is
        baked into the .so (byte-identical to a plain build, same cache key).
        When True, the codegen emits a ``d_inertia_params`` table + on-device
        6x6 rebuild + a host mutator, and the handle gains
        :py:meth:`RobotHandle.set_inertia_params` (mutate inertia at runtime, no
        recompile) and :py:attr:`RobotHandle.inertia_params` (the baked values to
        fetch-then-mutate). Re-keys the cache (the runtime-inertia .so coexists
        with the baked one). With the baked values it reproduces the baked
        result; mutate to do sysID / domain randomization / payload changes.
    use_joint_dynamics : bool, optional
        Model joint-local viscous damping + Coulomb friction in the value paths
        (inverse_dynamics / forward_dynamics / aba): ``tau -= damping*qd +
        friction*sign(qd)``, per joint, using the damping/friction declared in the
        URDF. Default False ⇒ the historical no-op build (byte-identical header, same
        cache key, and consistent with the bare-Pinocchio oracle which ignores
        damping/friction). When True the bias is emitted ONLY for robots that declare
        nonzero damping/friction; it re-keys the cache (the damped .so coexists with
        the baked no-op one). Match against ``RBDReference(..., use_joint_dynamics=True)``.
    algorithm_list : list[str] | str | None, optional
        Build only a SUBSET of algorithms into the per-robot ``.so`` instead of the
        full default profile. Default ``None`` ⇒ the historical full build (every
        method available; byte-identical header, same cache key, reuses the existing
        ``.so``). When set (e.g. ``["inverse_dynamics", "forward_dynamics"]``), only
        the named algorithms — plus their transitive dependencies, which
        GRiDCodeGenerator expands automatically (e.g. ``forward_dynamics_gradient``
        pulls in ``minv`` / ``inverse_dynamics`` / ``inverse_dynamics_gradient``) —
        are codegen'd and compiled. This cuts nvcc wall
        time, peak RAM, and ``.so`` size dramatically for big robots with heavy
        second-order kernels (e.g. ``fdsva_so`` on a mid-chain spherical robot is
        20+ min / 7 GB). Methods that were NOT built raise a clear runtime error
        naming the algorithm to add and rebuild — not a segfault. Re-keys the cache
        (a subset ``.so`` coexists with the full build). Recognized names mirror the
        codegen keys: ``inverse_dynamics``, ``minv``, ``forward_dynamics``, ``aba``,
        ``crba``, ``inverse_dynamics_gradient``, ``forward_dynamics_gradient``,
        ``idsva_so_body_frame``, ``fdsva_so``, ``end_effector_pose``[``_gradient``/
        ``_hessian``], ``integrator``, ``integrator_gradient``, plus curated profile
        sets like ``"dynamics-core"``. Supported on ALL backends (numpy / jax /
        torch): the JAX/torch FFI handlers are per-CORE-algo gated, so a subset
        ``.so`` builds only the requested cores on those surfaces too. A method that
        was NOT built raises the same clean "not built into this robot .so — add to
        algorithm_list and rebuild" error on jax/torch as it does on numpy.
    output_convention : str, optional
        Default IO convention for the returned handle: ``"pinocchio"`` (default,
        GRiD-native) or ``"mujoco"`` (mjx parity — wxyz quat, global-linear free-joint
        velocity). A runtime setting (NOT in the cache key — the .so is identical); it
        is a byte-identical no-op on a fixed base. Currently the VALUE methods
        (id/fd/aba/crba/minv) honor it; the derivative/second-order surfaces raise in
        mujoco mode until their codegen fusion lands. Equivalent to setting
        ``handle.output_convention`` after registration, or using the per-call thread-safe
        ``handle.mujoco`` view. numpy backend only for now.

    Returns
    -------
    RobotHandle
        Ready for forward_dynamics / inverse_dynamics / minv / etc.
    """
    if backend not in ("numpy", "jax", "torch"):
        raise ValueError(f"backend must be 'numpy', 'jax', or 'torch'; got {backend!r}")
    if output_convention not in ("pinocchio", "mujoco"):
        raise ValueError(
            f"output_convention must be 'pinocchio' or 'mujoco'; got {output_convention!r}")
    if output_convention == "mujoco" and not floating_base:
        # mjx and pinocchio coincide on a fixed base (no free-flyer); the _mujoco
        # native symbols are floating-base only, so reject early with a clear message.
        raise ValueError(
            "output_convention='mujoco' requires floating_base=True "
            "(mjx and pinocchio coincide on a fixed base).")
    if dtype not in ("float32", "float64"):
        raise ValueError(f"dtype must be 'float32' or 'float64'; got {dtype!r}")
    if dtype == "float64" and backend != "numpy":
        raise ValueError(
            f"dtype='float64' is only supported for the numpy backend; the "
            f"{backend!r} backend is strictly fp32 (Phase 8). Use backend='numpy'.")
    if runtime_inertia and backend != "numpy":
        raise ValueError(
            f"runtime_inertia=True is only supported for the numpy backend; the "
            f"{backend!r} backend does not yet thread the mutable inertia table. "
            f"Use backend='numpy'.")
    if use_joint_dynamics and backend != "numpy":
        raise ValueError(
            f"use_joint_dynamics=True is only supported for the numpy backend; the "
            f"{backend!r} backend does not yet thread the joint-dynamics flag. "
            f"Use backend='numpy'.")
    # Subset build (algorithm_list) is now supported on ALL backends: the jax/torch
    # FFI handlers are per-CORE-algo gated (#if GRID_HAS_<ALGO>), so a reduced profile
    # builds only the requested cores on those surfaces too, and the backend wrappers
    # map a missing-symbol AttributeError to the same clean subset error numpy raises.
    if backend == "jax":
        from . import jax as _jax_backend
        return _jax_backend.register_robot(
            name, urdf_path, urdf_string=urdf_string, floating_base=floating_base,
            ee_joint_names=ee_joint_names, max_batch_size=max_batch_size,
            cache_dir=cache_dir, force_rebuild=force_rebuild, cuda_arch=cuda_arch,
            output_convention=output_convention, algorithm_list=algorithm_list)
    if backend == "torch":
        from . import torch as _torch_backend
        return _torch_backend.register_robot(
            name, urdf_path, urdf_string=urdf_string, floating_base=floating_base,
            ee_joint_names=ee_joint_names, max_batch_size=max_batch_size,
            cache_dir=cache_dir, force_rebuild=force_rebuild, cuda_arch=cuda_arch,
            output_convention=output_convention, algorithm_list=algorithm_list)

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
    # fp64 (Phase 8): only inject dtype into the cache key for the fp64 build so
    # existing fp32 cache entries (keyed without a dtype field) stay valid — an
    # fp32 register_robot is byte-identical to pre-Phase-8 and reuses its .so.
    if dtype == "float64":
        code_options["dtype"] = "float64"
    # D.4 / Phase 5: runtime-mutable inertia. Only inject the flag (and thus re-key
    # the cache) when True, so a default register_robot is byte-identical to before
    # and reuses its existing fp32 .so. A runtime_inertia .so lands in its own entry.
    if runtime_inertia:
        code_options["runtime_inertia"] = True
    # Joint dynamics (viscous damping + Coulomb friction). Only inject the flag (and
    # thus re-key the cache) when True, so a default register_robot is byte-identical
    # to before and reuses its existing .so. A use_joint_dynamics .so lands in its own
    # entry — a damped build never collides with the historical no-op build.
    if use_joint_dynamics:
        code_options["use_joint_dynamics"] = True
    # Subset-build: only inject the algorithm_list into the cache key (and thus
    # re-key the cache) when the caller requests a non-default subset, so a default
    # register_robot is byte-identical to before and reuses its existing full .so.
    # A subset .so lands in its own entry, keyed by the (normalized) requested set.
    # Normalize to a canonical list-of-strings so equivalent spellings (comma string
    # vs list, ordering) dedupe to the same cache entry.
    if algorithm_list is not None:
        if isinstance(algorithm_list, str):
            algos = [a.strip() for a in algorithm_list.replace(";", ",").split(",") if a.strip()]
        else:
            algos = [str(a).strip() for a in algorithm_list if str(a).strip()]
        if not algos:
            raise ValueError("algorithm_list must name at least one algorithm or profile")
        code_options["algorithm_list"] = sorted(set(algos))
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
    handle = RobotHandle(name, str(so_path), meta, allow_fp64=allow_fp64)
    # output_convention is a runtime IO setting (no effect on the cached .so), so it
    # is applied to the handle rather than the cache key. mjx is a no-op on fixed base.
    handle.output_convention = output_convention
    return handle


def get_robot(name: str, cache_dir: str | Path | None = None, *,
              output_convention: str = "pinocchio") -> RobotHandle:
    """Look up a previously-registered robot by name.

    Raises RobotNotRegisteredError if `name` isn't in the manifest.
    ``output_convention`` ('pinocchio' or 'mujoco') is a runtime IO setting
    mirroring :py:func:`register_robot`; it can also be set later via
    ``handle.output_convention``.
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
    handle = RobotHandle(name, str(so_path), entry)
    handle.output_convention = output_convention
    return handle


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

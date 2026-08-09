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

import hashlib
import warnings
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


# Single-source the version from the installed distribution metadata so it can
# never drift from pyproject. Falls back to "unknown" when running from an
# uninstalled tree (e.g. a bare source checkout with no editable install).
try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("grid-rbd")
    del _pkg_version
except Exception:  # pragma: no cover - only hit when not installed
    __version__ = "unknown"


class RobotNotRegisteredError(KeyError):
    """Raised by get_robot() when no robot under that name has been registered."""

    def __init__(self, name: str):
        super().__init__(
            f"No robot registered under name {name!r}. Call "
            f"grid_rbd.register_robot(name={name!r}, urdf_path=...) first."
        )
        self.name = name


# Robots we've already warned about lacking an FFI-autotune entry — keyed on
# (launch_config_robot, floating_base) so the no-autotune warning fires at most
# once per robot/base per process, not once per register_robot/load_robot call.
_FFI_AUTOTUNE_WARNED: set[tuple[str, bool]] = set()


def _warn_if_no_ffi_autotune(robot_ident: str, floating_base: bool) -> None:
    """Emit a one-time-per-(robot,base) warning when a robot resolves to NO
    FFI-autotuned launch_config and will fall back to a conservative bake.

    The jax/torch FFI launch path is acutely sensitive to the per-block thread
    count: without a baked per-algo ``ffi_bases`` entry the robot lands on the
    conservative (TIER_SHARED, MAX_PERF_LEVEL_THREADS) default, which can be
    100-150x off the fast regime ([[project_grid_jax_ffi_thread_pathology]]).
    Mirrors the resolve in :py:func:`register_robot` (``profile="ffi"``); a no-op
    (no warning) whenever an entry exists or the config can't be resolved at all
    (no codegen package → nothing to warn about). Best-effort: never raises."""
    key = (str(robot_ident), bool(floating_base))
    if key in _FFI_AUTOTUNE_WARNED:
        return
    try:
        from grid_codegen.GRiDCodeGenerator import load_launch_config
    except Exception:
        return  # no codegen package (sdist install) → can't assess; stay silent
    try:
        lc = load_launch_config(robot_ident, bool(floating_base), profile="ffi")
    except Exception:
        return
    if lc:
        return  # has an FFI-autotuned config → fast regime, nothing to warn about
    _FFI_AUTOTUNE_WARNED.add(key)
    warnings.warn(
        f"grid_rbd: no FFI-autotuned thread config for {robot_ident!r} "
        f"(floating_base={bool(floating_base)}); using a conservative thread "
        f"bake. The jax/torch FFI launch path may run far below peak throughput. "
        f"Run `python test/benchmarks/autotune_ffi.py --robot {robot_ident} "
        f"--base {'floating' if floating_base else 'fixed'}` to tune it, or set "
        f"handle.set_threads_per_block(n) explicitly.",
        stacklevel=3,
    )


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
    runtime_transform: bool = False,
    runtime_joint_dynamics: bool = False,
    use_joint_dynamics: bool = False,
    enable_tool: bool = False,
    output_convention: str = "pinocchio",
    algorithm_list: list[str] | tuple[str, ...] | str | None = None,
    enable_mujoco_kernels: bool = True,
    _profile_overlay: str | None = "pybind",
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
        is a byte-identical no-op on a fixed base (mjx≡pinocchio with no free-flyer), so
        it is accepted on fixed AND floating robots alike for a uniform interface. The
        value methods (id/fd/aba/crba/minv) and the native-mjx derivative/second-order
        surfaces (id_gradient / fd_gradient / idsva_so / fdsva_so, floating base) honor
        it. Equivalent to setting ``handle.output_convention`` after registration, or
        using the per-call thread-safe ``handle.mujoco`` view.
    enable_mujoco_kernels : bool, optional
        Default ``True``. Set ``False`` to build a PIN-ONLY ``.so``: the mjx
        (MuJoCo-convention) kernel twins are not instantiated and their C-ABI entry
        points return rc=3 ("not built into this .so"). On a large floating-base
        non-mimic robot this is the difference between building and running out of
        memory — the second-order mjx twins are the largest kernels (``idsva_so_world_frame``
        was 28x its pin kernel raw; block-parallelizing the epilogue cut it to 2.42x pin on
        go2-floating), so pin-only builds g1 in ~33 min at ~11 GB peak. Use it if you
        do not need the MuJoCo output convention (GATO / PDDP second-order DDP, or
        anything reading Pinocchio-convention derivatives). No-op on fixed-base and
        mimic robots, which never get mjx twins. On a FLOATING base it is mutually
        exclusive with ``output_convention="mujoco"`` (that needs the twins); on a fixed
        base ``output_convention="mujoco"`` is itself a no-op, so the two compose freely.
        Codegen-affecting: it participates in the ``.so`` cache key only when ``False``,
        so existing caches stay valid.

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
    # mjx and pinocchio COINCIDE on a fixed base (no free-flyer): output_convention="mujoco"
    # is a provable byte-identical no-op there (handle._mjx_active gates on floating_base), so
    # accept it silently -- generic code that registers every robot with
    # output_convention="mujoco" then works on fixed AND floating robots alike (same interface).
    # The mjx kernels/symbols are only needed (and only emitted) on a FLOATING base, so the
    # enable_mujoco_kernels compatibility guard below applies only there.
    if output_convention == "mujoco" and floating_base and not enable_mujoco_kernels:
        # Caught here rather than at call time: on a floating base enable_mujoco_kernels=False
        # drops the mjx kernels from the .so entirely, so a 'mujoco' handle over that .so would
        # build fine and then fail on every derivative call with a bare rc=3.
        raise ValueError(
            "output_convention='mujoco' is incompatible with "
            "enable_mujoco_kernels=False (the mjx kernels are not built into the .so). "
            "Pass enable_mujoco_kernels=True, or use output_convention='pinocchio'.")
    if dtype not in ("float32", "float64"):
        raise ValueError(f"dtype must be 'float32' or 'float64'; got {dtype!r}")
    if dtype == "float64" and backend != "numpy":
        raise ValueError(
            f"dtype='float64' is only supported for the numpy backend; the "
            f"{backend!r} backend is strictly fp32 (Phase 8). Use backend='numpy'.")
    # runtime_inertia / runtime_transform are supported on ALL backends: the jax
    # FFI + torch custom-op kernels read the SAME device-resident mutable table the
    # numpy runner pokes (a single dlopen'd .so → one d_inertia_params /
    # d_transform_params __device__ global), exactly like runtime_joint_dynamics
    # (test_runtime_joint_dynamics::test_poke_seen_across_surfaces). A poke through
    # any surface is seen by all three.
    # use_joint_dynamics is a BUILD-TIME codegen flag baked into the id/fd/aba/*_gradient
    # kernels (not a per-algo GRID_HAS_* gate); the jax/torch FFI handlers call those same
    # baked symbols. So all three backends support it — it just re-keys the cache. (C5.)
    # FFI-autotune coverage warning (Friction 9). All three backends launch their
    # fast (jax/torch) path through the FFI thread-config; a robot with no baked
    # ffi_bases entry falls back to a conservative thread count that can be far off
    # the fast regime. Warn once per (robot, base) BEFORE dispatch so it covers
    # numpy / jax / torch uniformly. Best-effort + one-shot (never raises).
    try:
        from grid_rbd._compile import _resolve_launch_config_robot
        _ident = _resolve_launch_config_robot(
            urdf_path if urdf_path is not None else name)
        _warn_if_no_ffi_autotune(_ident, floating_base)
    except Exception:
        pass

    # Subset build (algorithm_list) is now supported on ALL backends: the jax/torch
    # FFI handlers are per-CORE-algo gated (#if GRID_HAS_<ALGO>), so a reduced profile
    # builds only the requested cores on those surfaces too, and the backend wrappers
    # map a missing-symbol AttributeError to the same clean subset error numpy raises.
    if backend in ("jax", "torch") and allow_fp64:
        # allow_fp64 is the legacy numpy-surface fp32-compute/fp64-io upcast; the
        # jax/torch surfaces take framework arrays whose dtype the caller controls,
        # so silently ignoring it would misreport precision. Real fp64 for these
        # backends arrives with the dtype='float64' template un-gating.
        raise NotImplementedError(
            f"allow_fp64 is not supported on the {backend!r} backend (numpy-only "
            "legacy upcast); pass fp32 arrays, or use backend='numpy'.")
    if backend == "jax":
        from . import jax as _jax_backend
        return _jax_backend.register_robot(
            name, urdf_path, urdf_string=urdf_string, floating_base=floating_base,
            ee_joint_names=ee_joint_names, max_batch_size=max_batch_size,
            cache_dir=cache_dir, force_rebuild=force_rebuild, cuda_arch=cuda_arch,
            output_convention=output_convention, algorithm_list=algorithm_list,
            use_joint_dynamics=use_joint_dynamics,
            runtime_joint_dynamics=runtime_joint_dynamics,
            runtime_inertia=runtime_inertia,
            runtime_transform=runtime_transform, enable_tool=enable_tool,
            enable_mujoco_kernels=enable_mujoco_kernels)
    if backend == "torch":
        from . import torch as _torch_backend
        return _torch_backend.register_robot(
            name, urdf_path, urdf_string=urdf_string, floating_base=floating_base,
            ee_joint_names=ee_joint_names, max_batch_size=max_batch_size,
            cache_dir=cache_dir, force_rebuild=force_rebuild, cuda_arch=cuda_arch,
            output_convention=output_convention, algorithm_list=algorithm_list,
            use_joint_dynamics=use_joint_dynamics,
            runtime_joint_dynamics=runtime_joint_dynamics,
            runtime_inertia=runtime_inertia,
            runtime_transform=runtime_transform, enable_tool=enable_tool,
            enable_mujoco_kernels=enable_mujoco_kernels)

    cache_key, so_path, meta = warm_robot(
        name, urdf_path, urdf_string=urdf_string, floating_base=floating_base,
        ee_joint_names=ee_joint_names, max_batch_size=max_batch_size,
        cache_dir=cache_dir, force_rebuild=force_rebuild, cuda_arch=cuda_arch,
        dtype=dtype, runtime_inertia=runtime_inertia,
        runtime_transform=runtime_transform,
        runtime_joint_dynamics=runtime_joint_dynamics,
        use_joint_dynamics=use_joint_dynamics, enable_tool=enable_tool,
        algorithm_list=algorithm_list, enable_mujoco_kernels=enable_mujoco_kernels,
    )
    handle = RobotHandle(name, str(so_path), meta, allow_fp64=allow_fp64)
    # output_convention is a runtime IO setting (no effect on the cached .so), so it
    # is applied to the handle rather than the cache key. mjx is a no-op on fixed base.
    handle.output_convention = output_convention
    # E6 per-algo threads overlay: the numpy/pybind surface defaults to "pybind"; the
    # jax/torch surfaces pass _profile_overlay=None/"torch" (jax's baked default IS
    # ffi). No-op until the robot JSON carries a <profile>_bases block.
    if _profile_overlay:
        handle.apply_profile_overlay(_profile_overlay)
    return handle


def warm_robot(
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
    dtype: str = "float32",
    runtime_inertia: bool = False,
    runtime_transform: bool = False,
    runtime_joint_dynamics: bool = False,
    use_joint_dynamics: bool = False,
    enable_tool: bool = False,
    algorithm_list: list[str] | tuple[str, ...] | str | None = None,
    enable_mujoco_kernels: bool = True,
) -> tuple[str, Path, dict]:
    """Generate + compile (or cache-hit) a robot's ``.so`` WITHOUT constructing
    a handle — no dlopen, no eager ``grid_rbd_init``, no device allocations.

    This is exactly :py:func:`register_robot`'s codegen/cache half (same cache
    key, same manifest entry); register_robot delegates here and then builds the
    handle. Returns ``(cache_key, so_path, meta)``. Used by the split-suite
    driver's compile-warm phase; needs only nvcc (pass ``cuda_arch=`` explicitly
    to skip the nvidia-smi arch probe).
    """
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
    # enable_tool (welded tool / payload): a convenience that turns on the pieces
    # attach_tool needs in one flag — the runtime-mutable inertia table (to compose
    # the payload) AND the runtime single-contact f_ext surface (tip forces). The
    # runtime EE pose/gradient surfaces are already in the default build. Only inject
    # the extra flag (re-keying the cache) when set, so a default build is unchanged.
    if enable_tool:
        runtime_inertia = True
        code_options["enable_contact_runtime"] = True
    if runtime_inertia:
        code_options["runtime_inertia"] = True
    # runtime_transform (mirror of runtime_inertia): runtime-mutable joint-frame
    # <origin>. Only inject the flag (re-keying the cache) when True so a default
    # register_robot is byte-identical and reuses its existing .so; a
    # runtime_transform .so lands in its own cache entry.
    if runtime_transform:
        code_options["runtime_transform"] = True
    # Joint dynamics (viscous damping + Coulomb friction). Only inject the flag (and
    # thus re-key the cache) when True, so a default register_robot is byte-identical
    # to before and reuses its existing .so. A use_joint_dynamics .so lands in its own
    # entry — a damped build never collides with the historical no-op build.
    if use_joint_dynamics:
        code_options["use_joint_dynamics"] = True
    # runtime_joint_dynamics (C5, mirror of runtime_inertia): runtime-mutable
    # damping/friction table (set_joint_dynamics). Inject the flag (re-keying the
    # cache) only when True, so a default register_robot is byte-identical and reuses
    # its existing .so; a runtime_joint_dynamics .so lands in its own cache entry.
    if runtime_joint_dynamics:
        code_options["runtime_joint_dynamics"] = True
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
    # Pin-only build: drop the mjx (MuJoCo output-convention) kernel twins. On a big
    # floating-base non-mimic robot the second-order mjx twins are the largest kernels
    # (idsva_so_world_frame's twin was 28x its pin kernel raw, cut to 2.42x by block-
    # parallelizing the epilogue), still the bulk of a humanoid build. Inject-only-when-False so
    # every existing cache entry (keyed without this field) stays valid and a default
    # register_robot still reuses its .so. No-op on fixed-base and mimic robots, which
    # never get mjx twins -- but still re-keys, so it is only injected when asked for.
    if not enable_mujoco_kernels:
        code_options["enable_mujoco_kernels"] = False
    # Launch-config bake (A1b + FFI autotune): the binding launches via the jax/torch
    # FFI path, so it bakes the "ffi" profile (ffi_bases) by default — see
    # _compile.generate_grid_cuh + GRiDCodeGenerator.load_launch_config. The per-algo
    # {tier,threads} that get baked are NOT derivable from urdf_bytes, so without these
    # the cache would NOT invalidate when config/launch_configs/ changes (e.g. after re-running
    # autotune_ffi.py) or when the profile differs. Fold BOTH the profile and the
    # RESOLVED config values into the cache key so a re-autotune rebuilds the .so.
    code_options["launch_config_profile"] = "ffi"
    try:
        from grid_codegen.GRiDCodeGenerator import load_launch_config
        from grid_rbd._compile import _resolve_launch_config_robot
        _lc_robot = _resolve_launch_config_robot(str(urdf_p) if urdf_path is not None else name)
        _lc = load_launch_config(_lc_robot, bool(floating_base), profile="ffi")
        if _lc:
            # canonical, json-safe fingerprint of the baked per-algo {tier,threads}
            code_options["launch_config"] = {k: dict(v) for k, v in sorted(_lc.items())}
    except Exception:
        pass  # un-resolvable config -> conservative fallback bakes; key stays urdf-derived
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
    return cache_key, so_path, meta


def get_robot(name: str, cache_dir: str | Path | None = None, *,
              backend: str = "numpy",
              output_convention: str = "pinocchio",
              _profile_overlay: str | None = "pybind") -> RobotHandle:
    """Look up a previously-registered robot by name.

    Raises RobotNotRegisteredError if `name` isn't in the manifest.
    ``backend`` ('numpy' / 'jax' / 'torch') mirrors :py:func:`register_robot`:
    it returns the matching backend handle for the same cached ``.so`` (the
    cache is shared across backends). Default 'numpy' keeps the historical
    return type.
    ``output_convention`` ('pinocchio' or 'mujoco') is a runtime IO setting
    mirroring :py:func:`register_robot`; it can also be set later via
    ``handle.output_convention``.
    """
    if backend not in ("numpy", "jax", "torch"):
        raise ValueError(f"backend must be 'numpy', 'jax', or 'torch'; got {backend!r}")
    if backend == "jax":
        from . import jax as _jax_backend
        return _jax_backend.get_robot(name, cache_dir, output_convention=output_convention)
    if backend == "torch":
        from . import torch as _torch_backend
        return _torch_backend.get_robot(name, cache_dir, output_convention=output_convention)
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
    if _profile_overlay:
        handle.apply_profile_overlay(_profile_overlay)  # E6 (no-op until tuned); jax/torch override
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


def load_robot(
    urdf_path: str | None = None,
    *,
    backend: str = "numpy",
    floating_base: bool = False,
    urdf_string: str | None = None,
    name: str | None = None,
    cache_dir: str | Path | None = None,
    **opts: Any,
):
    """One-call convenience: load a URDF and return a ready-to-use handle.

    This is the frictionless entry point — no name ceremony, no two-call
    precompile→get_robot dance. It is a THIN wrapper over
    :py:func:`register_robot`:

      1. Derives a stable, content-addressed ``name`` from the URDF bytes (so
         the caller never types a name, and re-loading the SAME urdf returns the
         SAME cached robot — no recompile). Override with ``name=`` if you want a
         human-friendly handle.
      2. Calls ``register_robot`` (which is itself idempotent on the cache key —
         a matching ``.so`` is reused, no nvcc).
      3. Returns a handle on the requested ``backend``: ``"numpy"`` → the pybind
         :class:`RobotHandle`, ``"jax"`` → a ``grid_rbd.jax`` handle, ``"torch"``
         → a ``grid_rbd.torch`` handle.

    Parameters
    ----------
    urdf_path : str | None
        Path to the robot URDF. Mutually exclusive with ``urdf_string``.
    backend : str
        ``"numpy"`` (default) / ``"jax"`` / ``"torch"``.
    floating_base : bool
        Treat the robot as floating-base (default fixed). Folded into the
        derived name, so the fixed and floating loads of one URDF get distinct,
        stable handles (they already compile to distinct cache entries).
    urdf_string : str | None
        Inline URDF text instead of a file. Mutually exclusive with ``urdf_path``.
    name : str | None
        Override the auto-derived handle name. Default ``None`` ⇒
        ``f"{stem}_{floating|fixed}_{sha256(urdf)[:12]}"`` (collision-resistant:
        a 48-bit content hash + the base flag).
    cache_dir : str | Path | None
        Cache root override (see :py:func:`register_robot`).
    **opts
        Any other :py:func:`register_robot` keyword (``ee_joint_names``,
        ``max_batch_size``, ``runtime_inertia``, ``runtime_transform``,
        ``output_convention``, ``algorithm_list``, ``dtype``, ``force_rebuild``,
        ``cuda_arch``, …) — passed straight through. ``backend``-incompatible
        options raise the same clear error ``register_robot`` already gives.

    Returns
    -------
    RobotHandle | JaxRobotHandle | TorchRobotHandle
        Ready for forward_dynamics / inverse_dynamics / minv / etc.
    """
    if backend not in ("numpy", "jax", "torch"):
        raise ValueError(f"backend must be 'numpy', 'jax', or 'torch'; got {backend!r}")
    if (urdf_path is None) == (urdf_string is None):
        raise ValueError("pass exactly one of urdf_path= or urdf_string=")

    # Stable, content-addressed default name: hash the URDF bytes so reusing the
    # SAME urdf reuses the cache + manifest binding, and two DIFFERENT urdfs map
    # to different names. sha256[:12] = 48 bits, collision-resistant for a robot
    # library. Fold floating_base into the name so the fixed and floating loads
    # of one urdf don't rebind each other's manifest entry (their cache keys
    # already differ). NOTE: the auto-name is for cache reuse only; the .so cache
    # key hashes the FULL urdf bytes + options, so a hash here never causes a
    # wrong-robot reuse.
    if name is None:
        if urdf_string is not None:
            urdf_bytes = urdf_string.encode("utf-8")
            stem = "inline"
        else:
            up = Path(urdf_path).expanduser()
            urdf_bytes = up.read_bytes()
            stem = up.stem or "robot"
        digest = hashlib.sha256(urdf_bytes).hexdigest()[:12]
        base_tag = "floating" if floating_base else "fixed"
        name = f"{stem}_{base_tag}_{digest}"

    # register_robot is idempotent on the cache; for the fast backends it both
    # builds-or-reuses and returns the backend handle directly, so a single call
    # is all we need (no separate get_robot).
    return register_robot(
        name,
        urdf_path,
        urdf_string=urdf_string,
        backend=backend,
        floating_base=floating_base,
        cache_dir=cache_dir,
        **opts,
    )


__all__ = [
    "RobotHandle",
    "SecondOrderID",
    "SecondOrderFD",
    "RobotNotRegisteredError",
    "register_robot",
    "load_robot",
    "get_robot",
    "list_registered",
    "precompile",
    "default_cache_dir",
    "__version__",
]

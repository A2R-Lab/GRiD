"""grid_rbd.jax — JAX FFI integration.

Mirrors the standard grid_rbd API but returns a :py:class:`JaxRobotHandle`
whose methods are JAX-callable, `jax.jit`-compatible, and run on
JAX-managed CUDA streams.

Usage:

    import grid_rbd.jax as grid_jax
    import jax

    handle = grid_jax.register_robot(name="iiwa14", urdf_path="iiwa.urdf")

    @jax.jit
    def step(q, qd):
        return handle.rnea(q, qd)

The underlying ``.so`` is shared with the plain ``grid_rbd.register_robot``
cache — registering the same name from both APIs uses the same compiled
library and doesn't trigger a recompile.

v0.1 surface:
  * ``rnea`` only. Adding the other methods is mechanical — each one is
    a 30-line C++ FFI handler + 10-line Python registration.

v0.1 limitations:
  * Handlers do device→host round-trips for inputs and outputs. Slow but
    correct. v0.2 will keep the data on device end-to-end.
"""
from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import Any

import numpy as np

import grid_rbd as _grid_rbd
from grid_rbd._handle import RobotHandle


# ─── handler registry ───────────────────────────────────────────────────────

# Maps (cache_key, method_name) → bool indicating whether the FFI target has
# been registered with JAX. Registration is process-global (JAX maintains the
# target table), so we only need to do it once per (cache_key, method).
_REGISTERED: dict[tuple[str, str], bool] = {}
_LOCK = threading.Lock()


def _ffi_target_name(cache_key: str, method: str) -> str:
    """JAX FFI target names are global. Key by cache_key (.so identity) so
    different robots → different targets, and so re-registering the same
    robot doesn't collide."""
    return f"grid_rbd_{method}_{cache_key[:12]}"


def _register_method_target(
    so_path: Path,
    cache_key: str,
    method: str,
    symbol: str,
) -> str:
    """Register one FFI target with JAX, returning the target name."""
    import jax
    target_name = _ffi_target_name(cache_key, method)
    with _LOCK:
        if _REGISTERED.get((cache_key, method)):
            return target_name
        # Load the .so and grab the handler symbol as a void*.
        lib = ctypes.CDLL(str(so_path))
        try:
            fn_ptr = getattr(lib, symbol)
        except AttributeError as e:
            raise RuntimeError(
                f"Symbol {symbol!r} missing from {so_path}; was the .so "
                f"compiled with GRID_RBD_WITH_JAX? (Reinstall jax + "
                f"force_rebuild=True at register_robot.)"
            ) from e
        # ctypes function objects are valid void*; wrap as a PyCapsule that
        # JAX accepts. ``jax.ffi.pycapsule`` builds the correct capsule from a
        # raw function pointer.
        capsule = jax.ffi.pycapsule(ctypes.cast(fn_ptr, ctypes.c_void_p).value)
        jax.ffi.register_ffi_target(target_name, capsule, platform="CUDA")
        _REGISTERED[(cache_key, method)] = True
    return target_name


# ─── JaxRobotHandle ─────────────────────────────────────────────────────────


class JaxRobotHandle:
    """JAX-flavored wrapper. Methods return ``jax.Array`` and are jittable.

    Wraps an underlying :py:class:`grid_rbd.RobotHandle` (which dlopens the
    same .so used by the plain Python wrapper) plus per-method JAX FFI
    target registrations.
    """

    def __init__(self, base: RobotHandle, cache_key: str, so_path: str):
        self._base = base
        self._cache_key = cache_key
        self._so_path = Path(so_path)

    # ─── metadata (delegated) ────────────────────────────────────────────
    @property
    def name(self) -> str:        return self._base.name
    @property
    def num_joints(self) -> int:  return self._base.num_joints
    @property
    def num_vel(self) -> int:     return self._base.num_vel
    @property
    def num_ees(self) -> int:     return self._base.num_ees
    @property
    def floating_base(self) -> bool: return self._base.floating_base
    @property
    def max_batch(self) -> int:   return self._base.max_batch

    # ─── algorithm methods ───────────────────────────────────────────────

    # ─── small helpers ───────────────────────────────────────────────────

    def _prep_2d(self, name: str, *arrays):
        """Cast to float32 jax arrays, validate (B, NJ), enforce same batch."""
        import jax.numpy as jnp
        cast = [jnp.asarray(a, dtype=jnp.float32) for a in arrays]
        for i, a in enumerate(cast):
            if a.ndim != 2 or a.shape[1] != self.num_joints:
                raise ValueError(
                    f"{name}: arg{i} must be (B, {self.num_joints}); got {a.shape}")
        B = cast[0].shape[0]
        for i, a in enumerate(cast[1:], start=1):
            if a.shape[0] != B:
                raise ValueError(
                    f"{name}: arg{i} batch={a.shape[0]} != arg0 batch={B}")
        if B > self.max_batch:
            raise ValueError(
                f"{name}: batch={B} > max_batch={self.max_batch}")
        return cast, B

    # ─── algorithm methods ───────────────────────────────────────────────

    def rnea(self, q, qd):
        """Inverse dynamics: c = M(q)·qdd_zero + h(q,qd) − g(q).

        ``q``, ``qd``: jax.Array shape (B, NJ), dtype float32.
        Returns shape (B, NJ).
        """
        import jax
        import jax.numpy as jnp
        target = _register_method_target(
            self._so_path, self._cache_key, "rnea", "grid_rbd_jax_rnea")
        (q, qd), B = self._prep_2d("rnea", q, qd)
        out_type = jax.ShapeDtypeStruct(q.shape, jnp.float32)
        return jax.ffi.ffi_call(target, out_type)(q, qd)

    def minv(self, q):
        """Direct mass-matrix inverse Minv(q). Returns (B, NJ, NJ).

        The kernel writes the lower triangle; we symmetrize inside the JAX
        graph so callers see a full SPD matrix. (The plain wrapper does the
        same in numpy.)
        """
        import jax
        import jax.numpy as jnp
        target = _register_method_target(
            self._so_path, self._cache_key, "minv", "grid_rbd_jax_minv")
        (q,), B = self._prep_2d("minv", q)
        nj = self.num_joints
        out_type = jax.ShapeDtypeStruct((B, nj, nj), jnp.float32)
        m = jax.ffi.ffi_call(target, out_type)(q)
        # Kernel fills the lower triangle; symmetrize as M + Mᵀ − diag(M).
        eye = jnp.eye(nj, dtype=m.dtype)
        return m + jnp.swapaxes(m, -1, -2) - m * eye

    def forward_dynamics(self, q, qd, u):
        """qdd = forward_dynamics(q, qd, u). Returns (B, NJ)."""
        import jax
        import jax.numpy as jnp
        target = _register_method_target(
            self._so_path, self._cache_key,
            "forward_dynamics", "grid_rbd_jax_forward_dynamics")
        (q, qd, u), B = self._prep_2d("forward_dynamics", q, qd, u)
        out_type = jax.ShapeDtypeStruct(q.shape, jnp.float32)
        return jax.ffi.ffi_call(target, out_type)(q, qd, u)

    def aba(self, q, qd, u):
        """qdd = aba(q, qd, u) via the articulated body algorithm. Returns (B, NJ)."""
        import jax
        import jax.numpy as jnp
        target = _register_method_target(
            self._so_path, self._cache_key, "aba", "grid_rbd_jax_aba")
        (q, qd, u), B = self._prep_2d("aba", q, qd, u)
        out_type = jax.ShapeDtypeStruct(q.shape, jnp.float32)
        return jax.ffi.ffi_call(target, out_type)(q, qd, u)

    def crba(self, q):
        """Mass matrix M(q) via composite rigid body algorithm. Returns (B, NJ, NJ)."""
        import jax
        import jax.numpy as jnp
        target = _register_method_target(
            self._so_path, self._cache_key, "crba", "grid_rbd_jax_crba")
        (q,), B = self._prep_2d("crba", q)
        nj = self.num_joints
        out_type = jax.ShapeDtypeStruct((B, nj, nj), jnp.float32)
        return jax.ffi.ffi_call(target, out_type)(q)


# ─── public API ─────────────────────────────────────────────────────────────


def register_robot(
    name: str,
    urdf_path: str,
    *,
    floating_base: bool = False,
    ee_joint_names: list[str] | tuple[str, ...] | None = None,
    max_batch_size: int = 256,
    cache_dir: str | Path | None = None,
    force_rebuild: bool = False,
    cuda_arch: int | None = None,
) -> JaxRobotHandle:
    """Register a robot for use with JAX.

    Compiles + caches the same per-robot ``.so`` that
    :py:func:`grid_rbd.register_robot` uses (cache hit if already
    compiled). Additionally registers JAX FFI targets so the methods
    are callable inside ``jax.jit``.

    Returns a :py:class:`JaxRobotHandle`.
    """
    base = _grid_rbd.register_robot(
        name=name,
        urdf_path=urdf_path,
        floating_base=floating_base,
        ee_joint_names=ee_joint_names,
        max_batch_size=max_batch_size,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
        cuda_arch=cuda_arch,
    )
    # Pull the cache_key + .so path from the manifest so we can dlopen
    # to register JAX FFI symbols.
    from grid_rbd._cache import default_cache_dir, manifest_lookup, store_dir
    cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    entry = manifest_lookup(cache_dir, name)
    if entry is None:
        raise RuntimeError(
            f"register_robot returned but {name!r} isn't in manifest; "
            f"cache may be corrupted")
    so_path = store_dir(cache_dir, entry["cache_key"]) / "robot.so"
    return JaxRobotHandle(base, entry["cache_key"], str(so_path))


def get_robot(
    name: str,
    cache_dir: str | Path | None = None,
) -> JaxRobotHandle:
    """Look up a previously-registered robot. Same cache as
    :py:func:`grid_rbd.get_robot`."""
    base = _grid_rbd.get_robot(name, cache_dir=cache_dir)
    from grid_rbd._cache import default_cache_dir, manifest_lookup, store_dir
    cd = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    entry = manifest_lookup(cd, name)
    so_path = store_dir(cd, entry["cache_key"]) / "robot.so"
    return JaxRobotHandle(base, entry["cache_key"], str(so_path))


__all__ = ["JaxRobotHandle", "register_robot", "get_robot"]

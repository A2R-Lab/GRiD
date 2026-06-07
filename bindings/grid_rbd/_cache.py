"""Per-host cache for compiled per-robot .so files.

Layout:
    ~/.cache/grid-rbd/
    ├── manifest.json              # name -> cache_key registry
    └── store/
        ├── <cache_key>/
        │   ├── grid.cuh           # generated header
        │   ├── wrapper.cu         # boilerplate that exposes C ABI
        │   ├── robot.so           # compiled per-robot library
        │   ├── meta.json          # NUM_JOINTS / NUM_VEL / NUM_EES / options
        │   └── build.log
        └── ...

Cache key = sha256(urdf_bytes + canonical_json(options) + grid_rbd_version + cuda_arch).
CUDA arch is part of the key so a multi-GPU user keeps separate .so files.

The manifest binds a human-friendly `name` to a cache key. Re-registering the
same name with a different URDF or options overwrites the binding (the old
.so file lingers in store/ for manual GC; future v2 will add `grid-rbd gc`).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import platformdirs
    _DEFAULT_DIR = Path(platformdirs.user_cache_dir("grid-rbd"))
except ImportError:
    _DEFAULT_DIR = Path.home() / ".cache" / "grid-rbd"


_SCHEMA_VERSION = 1


def default_cache_dir() -> Path:
    """Return the default cache directory; honors $GRID_RBD_CACHE_DIR."""
    override = os.environ.get("GRID_RBD_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return _DEFAULT_DIR


def detect_cuda_arch() -> int:
    """Detect the compute capability of the first installed GPU (as int, e.g. 120 for sm_120).

    Returns 0 if no GPU is detectable; callers should error out then.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().split("\n")[0].strip()
        major, minor = out.split(".")
        return int(major) * 10 + int(minor)
    except Exception:
        return 0


def package_version() -> str:
    """Version of the grid-rbd package; used as a cache-key input so generated
    code from older versions doesn't get reused after upgrade.

    Reads the package's own ``__version__`` (deterministic) rather than
    ``importlib.metadata.version("grid-rbd")``, which is ambiguous when a stale
    root ``GRiD-RBD`` dist also normalizes to ``grid-rbd`` (it would
    nondeterministically return 1.0.0 vs 0.1.0 by sys.path order, silently
    re-keying the compile cache and causing spurious recompiles)."""
    try:
        from . import __version__
        return __version__
    except Exception:
        return "0.0.0-dev"


def _wrapper_template_hash() -> str:
    """sha256 of the bundled wrapper_template.cu so editing the wrapper
    invalidates the cache. The package version alone isn't enough for editable
    dev installs where the version doesn't bump on every edit."""
    path = Path(__file__).parent / "wrapper_template.cu"
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return ""


def _torch_abi_tag() -> str:
    """Torch version + CXX11-ABI tag, mixed into the cache key so a torch-aware
    .so isn't reused across incompatible torch ABIs. Empty when torch is absent
    (a no-torch build is valid and shouldn't carry a torch tag)."""
    try:
        import torch
        return f"torch={torch.__version__},cxxabi={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"
    except Exception:
        return ""


def _jax_ffi_tag() -> str:
    """JAX-FFI availability + version tag, mixed into the cache key.

    `compile_so` emits the JAX FFI handler block (and links jax's FFI headers)
    only when jax is importable at compile time. Whether those `grid_rbd_jax_*`
    symbols exist therefore changes with jax's presence/version, so it must be
    part of the key: a .so built with NO jax installed lacks the FFI symbols,
    and a later jax-enabled session must NOT reuse it (the FFI dlsym would fail
    with a confusing 'symbol missing' error). Empty when jax is absent so a
    no-jax build doesn't carry a jax tag (mirrors `_torch_abi_tag`)."""
    try:
        import jax
        from jax import ffi as _jax_ffi  # noqa: F401  (probe FFI availability)
        return f"jaxffi={jax.__version__}"
    except Exception:
        return ""


def canonical_options(options: dict[str, Any]) -> str:
    """Canonical JSON serialization of compile options for hashing.

    Only options that affect generated code are included; cosmetic stuff
    (cache_dir, force_rebuild, etc.) is filtered by the caller.
    """
    return json.dumps(options, sort_keys=True, separators=(",", ":"))


def compute_cache_key(urdf_bytes: bytes, options: dict[str, Any], cuda_arch: int) -> str:
    """Compute the content-addressable cache key for a (urdf, options, arch)
    combination. The package version is mixed in so upgrades invalidate
    correctly."""
    h = hashlib.sha256()
    h.update(urdf_bytes)
    h.update(canonical_options(options).encode())
    h.update(f"arch={cuda_arch}".encode())
    h.update(f"grid_rbd={package_version()}".encode())
    h.update(f"wrapper={_wrapper_template_hash()}".encode())
    # Mix in the torch ABI tag so a torch-aware build isn't reused under an
    # incompatible torch version (the TORCH_LIBRARY symbols bake in the ABI).
    # Empty string when torch is absent → no effect on no-torch builds.
    h.update(f"{_torch_abi_tag()}".encode())
    # Mix in the JAX-FFI tag so a .so built without jax (no FFI symbols) isn't
    # reused by a later jax-enabled session, and a jax-version bump that changes
    # the FFI ABI re-keys. Empty when jax is absent → no effect on no-jax builds.
    h.update(f"{_jax_ffi_tag()}".encode())
    return h.hexdigest()


def manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def store_dir(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / "store" / cache_key


def load_manifest(cache_dir: Path) -> dict[str, Any]:
    """Load the manifest JSON, returning an empty schema-conformant dict if
    none exists yet. Caller does not need to handle FileNotFoundError."""
    path = manifest_path(cache_dir)
    if not path.exists():
        return {"schema_version": _SCHEMA_VERSION, "robots": {}}
    with path.open() as f:
        data = json.load(f)
    if data.get("schema_version") != _SCHEMA_VERSION:
        # Future: handle migrations. For now, refuse to use older manifests.
        raise RuntimeError(
            f"manifest at {path} has schema_version "
            f"{data.get('schema_version')!r}, expected {_SCHEMA_VERSION}. "
            "Delete the cache to start fresh."
        )
    return data


def save_manifest(cache_dir: Path, manifest: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(cache_dir)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(path)  # atomic on POSIX


def manifest_register(
    cache_dir: Path,
    name: str,
    cache_key: str,
    meta: dict[str, Any],
) -> None:
    """Bind `name` to `cache_key` in the manifest, overwriting any prior
    binding under the same name."""
    manifest = load_manifest(cache_dir)
    manifest["robots"][name] = {
        "cache_key": cache_key,
        **meta,
    }
    save_manifest(cache_dir, manifest)


def manifest_lookup(cache_dir: Path, name: str) -> dict[str, Any] | None:
    """Return manifest entry for `name`, or None if not registered."""
    manifest = load_manifest(cache_dir)
    return manifest.get("robots", {}).get(name)


def list_registered(cache_dir: Path) -> list[dict[str, Any]]:
    """List all registered robots with their metadata."""
    manifest = load_manifest(cache_dir)
    return [
        {"name": name, **entry}
        for name, entry in manifest.get("robots", {}).items()
    ]

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
    code from older versions doesn't get reused after upgrade."""
    try:
        from importlib.metadata import version
        return version("grid-rbd")
    except Exception:
        return "0.0.0-dev"


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

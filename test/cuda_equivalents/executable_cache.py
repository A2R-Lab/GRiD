"""Content-keyed cache for the CUDA test executables that tests compile themselves.

The flagship runner has its own cache in cuda_harness. The integrator, plant and
second-order runners compile one large generated header each (up to hours for
the humanoids), so they share this one: a per-key lock, a private staging build,
one atomic publication with a manifest, and never a cached failure. Kept out of
cuda_harness so that editing the cache stales only the shards whose modules
import it (see run_split_suite._shard_fingerprint_paths).
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from test.cuda_equivalents.cuda_harness import (
    _cache_enabled,
    _cache_root,
    _hash_file,
    _nvcc_version_text,
    _sha256_bytes,
    _stable_json_hash,
)

EXECUTABLE_CACHE_SCHEMA = 1
_QUOTED_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"]+)"', re.MULTILINE)


def _resolve_include(name: str, including: Path, include_dirs) -> Path | None:
    for base in (including.parent, *include_dirs):
        candidate = Path(base) / name
        if candidate.is_file():
            return candidate
    return None


def _source_closure(sources, include_dirs) -> list[tuple[str, str]]:
    """(logical name, sha256) for every source and every quoted include they reach.

    Logical names are basenames for the sources and the include spelling for the
    rest, so the key does not depend on the (temporary) directory a test used.
    System <...> headers are covered by the toolkit identity instead.
    """
    entries, seen = [], set()
    stack = [(Path(src).name, Path(src)) for src in sources]
    while stack:
        name, path = stack.pop()
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        data = path.read_bytes()
        entries.append((name, _sha256_bytes(data)))
        for include in _QUOTED_INCLUDE.findall(data.decode("utf-8", errors="replace")):
            found = _resolve_include(include, path, include_dirs)
            if found is not None:
                stack.append((include, found))
    return sorted(entries)


def _toolkit_identity(nvcc: str) -> dict:
    return {"nvcc": os.path.realpath(nvcc), "version": _nvcc_version_text()}


def executable_cache_key(sources, flags, include_dirs=(), *, nvcc: str, variant=None) -> tuple[str, dict]:
    """Content key for one nvcc build: every input byte, the toolkit, flags, variant."""
    payload = {
        "schema": EXECUTABLE_CACHE_SCHEMA,
        "kind": "cuda_test_executable",
        "sources": _source_closure(sources, include_dirs),
        "toolkit": _toolkit_identity(nvcc),
        "flags": list(flags),
        "variant": variant or {},
    }
    return _stable_json_hash(payload), payload


def cached_nvcc_executable(sources, flags, *, exe_name: str, fallback_dir: Path,
                           include_dirs=(), variant=None, what: str = "CUDA runner"):
    """Build ``sources[0]`` with nvcc once per content key and return (exe, cmd).

    ``sources`` are copied side by side into a private build directory (so quoted
    includes such as grid.cuh resolve locally); ``flags`` are every nvcc flag except
    -I, -o and the source path. The key covers all source bytes plus the quoted
    includes they reach, the toolkit identity, the flags and ``variant``.

    Concurrency and failure: a per-key flock serializes builders, a build happens
    in a temporary directory and is published by one atomic rename together with
    its manifest, and a failed build is deleted, never cached. A hit re-checks the
    executable's sha256 against the manifest. GRID_CUDA_DISABLE_CACHE=1 builds in
    ``fallback_dir`` instead.
    """
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc was not found; install CUDA Toolkit to run CUDA tests.")
    sources = [Path(src) for src in sources]
    include_dirs = [Path(d) for d in include_dirs]

    def _command(build_dir: Path) -> list[str]:
        return [nvcc, *flags, *(f"-I{d}" for d in include_dirs),
                "-o", str(build_dir / exe_name), str(build_dir / sources[0].name)]

    def _build(build_dir: Path) -> list[str]:
        for src in sources:
            if src.resolve() != (build_dir / src.name).resolve():
                shutil.copyfile(src, build_dir / src.name)
        cmd = _command(build_dir)
        result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
        if result.returncode != 0:
            pytest.fail(f"{what} compilation failed.\nCommand: {' '.join(cmd)}\n"
                        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        return cmd

    if not _cache_enabled():
        fallback_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build(fallback_dir)
        return fallback_dir / exe_name, cmd

    key, payload = executable_cache_key(sources, flags, include_dirs, nvcc=nvcc, variant=variant)
    root = _cache_root() / "executables"
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / key
    manifest_path = final_dir / "manifest.json"
    with open(root / f"{key}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            exe = final_dir / exe_name
            if (manifest.get("key") == key and exe.is_file()
                    and _hash_file(exe) == manifest.get("executable_sha256")):
                print(f"[cuda-cache] executable hit {what} key={key[:12]}", flush=True)
                return exe, _command(final_dir)
        if final_dir.exists():  # incomplete or corrupt entry: rebuild it
            shutil.rmtree(final_dir)
        staging = Path(tempfile.mkdtemp(prefix=f".build-{key[:12]}-", dir=root))
        try:
            print(f"[cuda-cache] executable miss {what} key={key[:12]}", flush=True)
            started = time.time()
            cmd = _build(staging)
            (staging / "manifest.json").write_text(json.dumps({
                "key": key,
                "key_payload": payload,
                "command": cmd,
                "executable": exe_name,
                "executable_sha256": _hash_file(staging / exe_name),
                "build_seconds": round(time.time() - started, 1),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }, indent=2, sort_keys=True) + "\n")
            os.rename(staging, final_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    return final_dir / exe_name, _command(final_dir)

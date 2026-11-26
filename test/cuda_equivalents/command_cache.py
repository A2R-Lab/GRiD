"""Content-keyed cache for the nvcc commands the native tests run themselves.

Most native test modules build one small program per case with a direct
``subprocess.run([nvcc, ...])``. With no cache, each case pays its compile on
every run, on the one GPU shard chain. ``cached_nvcc_run`` takes the same
command and returns the same kind of result, so a module changes one call.

The key holds every byte the compiler reads that the repository controls: each
source on the command line, every header it reaches through ``-I`` directories
and through the including file's own directory, the flags in order, and the
toolkit identity. Paths are left out (the source's basename, an include by its
spelling), so a build made in one temporary directory is found from another.
The toolkit's own headers are covered by the toolkit identity.

A hit copies the stored output to the requested ``-o`` path and returns the
compiler's stored messages, so a test that reads the warnings sees the same
text. A miss runs the command as given and stores the output when it succeeds. A failed command is
never stored, so a test that expects a compile error runs the compiler each
time. A command this module cannot key (no ``-o``, no source) runs unchanged.
GRID_CUDA_DISABLE_CACHE=1 runs every command unchanged.
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

from test.cuda_equivalents.cuda_harness import (
    _cache_enabled,
    _cache_root,
    _hash_file,
    _nvcc_version_text,
    _sha256_bytes,
    _stable_json_hash,
)

COMMAND_CACHE_SCHEMA = 1
_STORED_NAME = "output"
_SOURCE_SUFFIXES = (".cu", ".cpp", ".cc", ".cxx", ".c")
_BINARY_INPUT_SUFFIXES = (".o", ".a", ".so")
_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*(["<])([^">]+)[">]', re.MULTILINE)


def _split_command(cmd, cwd: Path):
    """(flags, include_dirs, sources, binary_inputs, output) of one nvcc command.

    ``flags`` keeps every argument that is not a path the key replaces: the
    output, the ``-I`` directories and the positional input files.
    """
    def path(arg):
        candidate = Path(arg)
        return candidate if candidate.is_absolute() else cwd / candidate

    flags, include_dirs, sources, binaries, forced, output = [], [], [], [], [], None
    args = iter(str(a) for a in cmd[1:])
    for arg in args:
        if arg == "-o":
            output = path(next(args))
        elif arg == "-I":
            include_dirs.append(path(next(args)))
        elif arg.startswith("-I"):
            include_dirs.append(path(arg[2:]))
        elif arg == "-include":
            forced.append(path(next(args)))
            flags.append(arg)
        elif not arg.startswith("-") and arg.endswith(_SOURCE_SUFFIXES):
            sources.append(path(arg))
        elif not arg.startswith("-") and arg.endswith(_BINARY_INPUT_SUFFIXES):
            binaries.append(path(arg))
        else:
            flags.append(arg)
    return flags, include_dirs, sources + forced, binaries, output


def _include_closure(sources, include_dirs) -> list[tuple[str, str]]:
    """(logical name, sha256) of the sources and of every header they reach."""
    entries, seen = [], set()
    stack = [(Path(src).name, Path(src)) for src in sources]
    while stack:
        name, file = stack.pop()
        resolved = file.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        data = file.read_bytes()
        entries.append((name, _sha256_bytes(data)))
        for style, include in _INCLUDE.findall(data.decode("utf-8", errors="replace")):
            bases = ([file.parent] if style == '"' else []) + list(include_dirs)
            for base in bases:
                candidate = Path(base) / include
                if candidate.is_file():
                    stack.append((include, candidate))
                    break
    return sorted(entries)


def command_cache_key(cmd, cwd=None) -> tuple[str, dict, Path] | None:
    """(key, payload, output path), or None when the command cannot be keyed."""
    cwd = Path(cwd) if cwd is not None else Path.cwd()
    flags, include_dirs, sources, binaries, output = _split_command(cmd, cwd)
    if output is None or not sources or not all(src.is_file() for src in sources + binaries):
        return None
    compiler = shutil.which(str(cmd[0])) or str(cmd[0])
    payload = {
        "schema": COMMAND_CACHE_SCHEMA,
        "kind": "cuda_test_command",
        "sources": _include_closure(sources, include_dirs),
        "binary_inputs": sorted((b.name, _hash_file(b)) for b in binaries),
        "toolkit": {"nvcc": os.path.realpath(compiler), "version": _nvcc_version_text()},
        "flags": flags,
        "output_suffix": output.suffix,
    }
    return _stable_json_hash(payload), payload, output


def _valid_entry(final_dir: Path, key: str) -> dict | None:
    """The entry's manifest when it is complete and intact, else None."""
    try:
        manifest = json.loads((final_dir / "manifest.json").read_text())
        stored = final_dir / _STORED_NAME
        intact = (manifest.get("key") == key and stored.is_file()
                  and _hash_file(stored) == manifest.get("output_sha256")
                  and isinstance(manifest.get("stdout"), str) and isinstance(manifest.get("stderr"), str))
        return manifest if intact else None
    except (OSError, ValueError):
        return None


def cached_nvcc_run(cmd, *, cwd=None, what: str = "CUDA test program", **run_kwargs):
    """``subprocess.run(cmd, cwd=cwd, ...)`` for an nvcc build, with a content cache.

    Returns a ``CompletedProcess``. On a hit it has return code 0 and the stored
    compiler messages, and the stored file is at the command's ``-o`` path. Only
    the captured text form is cached (the default); any other form runs unchanged.
    """
    run_kwargs.setdefault("capture_output", True)
    run_kwargs.setdefault("text", True)
    cacheable = (_cache_enabled() and run_kwargs["capture_output"] is True and run_kwargs["text"] is True
                 and not set(run_kwargs) - {"capture_output", "text", "timeout"})
    keyed = command_cache_key(cmd, cwd) if cacheable else None
    if keyed is None:
        return subprocess.run(cmd, cwd=cwd, **run_kwargs)
    key, payload, output = keyed
    root = _cache_root() / "commands"
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / key
    with open(root / f"{key}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = _valid_entry(final_dir, key)
        if manifest is not None:
            print(f"[cuda-cache] command hit {what} key={key[:12]}", flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_dir / _STORED_NAME, output)
            return subprocess.CompletedProcess(list(cmd), 0, manifest["stdout"], manifest["stderr"])
        if final_dir.exists():  # incomplete or corrupt entry: rebuild it
            shutil.rmtree(final_dir)
        print(f"[cuda-cache] command miss {what} key={key[:12]}", flush=True)
        started = time.time()
        result = subprocess.run(cmd, cwd=cwd, **run_kwargs)
        if result.returncode != 0 or not output.is_file():
            return result
        staging = Path(tempfile.mkdtemp(prefix=f".build-{key[:12]}-", dir=root))
        try:
            shutil.copy2(output, staging / _STORED_NAME)
            (staging / "manifest.json").write_text(json.dumps({
                "key": key,
                "key_payload": payload,
                "command": [str(a) for a in cmd],
                "output_sha256": _hash_file(staging / _STORED_NAME),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "build_seconds": round(time.time() - started, 1),
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }, indent=2, sort_keys=True) + "\n")
            os.rename(staging, final_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return result

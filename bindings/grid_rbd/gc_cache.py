"""Garbage-collect the per-host robot cache (2026-10-10, user-ratified policy).

    python -m grid_rbd.gc_cache [--cache-dir DIR] [--execute] [--objects-days N]
    grid_rbd.gc_cache()                      # dry run, returns the report
    grid_rbd.gc_cache(execute=True)          # delete

Garbage = store/<content_key>/ entries that NO manifest name points to, plus
bykey/ pointers whose entry is gone (or is being removed), plus dead `.gen_*`
staging dirs. Nothing else: an entry a name still binds is kept however old
it is, and per-unit objects (objects/) are kept forever — `--objects-days N`
opts in to dropping objects not built or linked from in N days. Dry run is
the default; `--execute` / `execute=True` deletes. The scan + delete runs
under the manifest lock so a concurrent register_robot cannot bind a name to
an entry while it is being removed.

Why names drift: re-registering a name after any codegen / bindings / toolchain
change binds it to a NEW content key and leaves the old entry behind (the
store is content-addressed and never overwrites). A test suite does this for
hundreds of names, so most of a dev box's store is unreferenced.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from ._cache import default_cache_dir, load_manifest


def _tree_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


def gc_cache(cache_dir: str | Path | None = None, *, execute: bool = False,
             objects_days: float | None = None) -> dict[str, Any]:
    """Scan (and with ``execute=True`` delete) the cache garbage described in
    the module docstring. Returns a report dict; sizes in bytes."""
    cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    report: dict[str, Any] = {
        "cache_dir": str(cache_dir), "executed": execute,
        "entries_kept": 0, "entries_removed": [], "entries_bytes": 0,
        "pointers_removed": 0, "staging_removed": 0,
        "objects_removed": 0, "objects_bytes": 0,
    }
    store = cache_dir / "store"
    if not store.is_dir():
        return report
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "manifest.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            referenced = {e.get("cache_key") for e in load_manifest(cache_dir).get("robots", {}).values()}
            entries = [d for d in store.iterdir() if d.is_dir() and not d.name.startswith(".")]
            garbage = [d for d in entries if d.name not in referenced]
            report["entries_kept"] = len(entries) - len(garbage)
            for d in garbage:
                report["entries_bytes"] += _tree_bytes(d)
                report["entries_removed"].append(d.name)
                if execute:
                    shutil.rmtree(d, ignore_errors=True)
            gone = {d.name for d in garbage}
            bykey = cache_dir / "bykey"
            if bykey.is_dir():
                for p in bykey.iterdir():
                    try:
                        target = p.read_text().strip()
                    except OSError:
                        continue
                    if target in gone or not (store / target / "robot.so").exists():
                        report["pointers_removed"] += 1
                        if execute:
                            p.unlink(missing_ok=True)
            from . import _sweep_dead_staging_dirs
            staging = list(store.glob(".gen_*_*"))
            report["staging_removed"] = _sweep_dead_staging_dirs(store) if execute else len(staging)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    objects = cache_dir / "objects"
    if objects_days is not None and objects_days > 0 and objects.is_dir():
        cutoff = time.time() - objects_days * 86400
        for o in objects.glob("wrapper_*.o"):
            try:
                st = o.stat()
            except OSError:
                continue
            if st.st_mtime < cutoff:
                report["objects_removed"] += 1
                report["objects_bytes"] += st.st_size
                if execute:
                    o.unlink(missing_ok=True)
    return report


def format_report(r: dict[str, Any]) -> str:
    verb = "removed" if r["executed"] else "would remove"
    gib = lambda b: f"{b / 2**30:.2f} GiB"
    lines = [f"grid_rbd cache {r['cache_dir']}: {verb}",
             f"  {len(r['entries_removed'])} store entries no name binds ({gib(r['entries_bytes'])}); {r['entries_kept']} kept",
             f"  {r['pointers_removed']} dangling bykey pointers, {r['staging_removed']} dead staging dirs"]
    if r["objects_removed"]:
        lines.append(f"  {r['objects_removed']} cached unit objects ({gib(r['objects_bytes'])})")
    if not r["executed"]:
        lines.append("  (dry run — pass --execute to delete)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m grid_rbd.gc_cache", description=__doc__.split("\n\n")[1])
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--execute", action="store_true", help="delete (default: dry run)")
    ap.add_argument("--objects-days", type=float, default=None,
                    help="also drop per-unit objects not built/linked from in N days (default: keep forever)")
    a = ap.parse_args(argv)
    print(format_report(gc_cache(a.cache_dir, execute=a.execute, objects_days=a.objects_days)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

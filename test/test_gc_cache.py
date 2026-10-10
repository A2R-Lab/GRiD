"""grid_rbd.gc_cache: unreferenced store entries + dangling pointers + dead staging dirs; dry run default."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import importlib
gcmod = importlib.import_module("grid_rbd.gc_cache")   # the package exports the FUNCTION under this name
from grid_rbd._cache import save_manifest, keymap_record


def _cache(tmp_path: Path) -> Path:
    cd = tmp_path / "grid-rbd"
    store = cd / "store"
    for key in ("aaa", "bbb", "ccc"):
        (store / key).mkdir(parents=True)
        (store / key / "robot.so").write_bytes(b"\x7fELF" + b"x" * 1000)
        (store / key / "meta.json").write_text("{}")
    (store / "ddd").mkdir()                       # incomplete entry (no robot.so), unreferenced
    (store / ".gen_999999_dead").mkdir()          # staging dir of a dead pid
    save_manifest(cd, {"schema_version": 1, "robots": {"r1": {"cache_key": "aaa"}, "r2": {"cache_key": "bbb"}}})
    keymap_record(cd, "in-aaa", "aaa")
    keymap_record(cd, "in-ccc", "ccc")           # points at garbage
    keymap_record(cd, "in-gone", "zzz")          # points at nothing
    objs = cd / "objects"; objs.mkdir()
    (objs / "wrapper_core-old.o").write_bytes(b"o" * 10)
    old = time.time() - 40 * 86400
    os.utime(objs / "wrapper_core-old.o", (old, old))
    (objs / "wrapper_core-new.o").write_bytes(b"o" * 10)
    return cd


def test_dry_run_reports_and_deletes_nothing(tmp_path):
    cd = _cache(tmp_path)
    r = gcmod.gc_cache(cd)
    assert sorted(r["entries_removed"]) == ["ccc", "ddd"] and r["entries_kept"] == 2
    assert r["pointers_removed"] == 2 and r["staging_removed"] == 1 and r["objects_removed"] == 0
    assert (cd / "store" / "ccc").exists() and (cd / "bykey" / "in-ccc").exists()
    assert "dry run" in gcmod.format_report(r)


def test_execute_removes_only_garbage(tmp_path):
    cd = _cache(tmp_path)
    r = gcmod.gc_cache(cd, execute=True)
    store = cd / "store"
    assert sorted(p.name for p in store.iterdir()) == ["aaa", "bbb"]
    assert sorted(p.name for p in (cd / "bykey").iterdir()) == ["in-aaa"]
    assert r["entries_bytes"] > 1000
    assert len(list((cd / "objects").iterdir())) == 2, "objects kept forever by default"
    # idempotent
    r2 = gcmod.gc_cache(cd, execute=True)
    assert r2["entries_removed"] == [] and r2["pointers_removed"] == 0


def test_objects_days_opt_in(tmp_path):
    cd = _cache(tmp_path)
    r = gcmod.gc_cache(cd, execute=True, objects_days=30)
    assert r["objects_removed"] == 1
    assert sorted(p.name for p in (cd / "objects").iterdir()) == ["wrapper_core-new.o"]


def test_cli_dry_run(tmp_path, capsys):
    cd = _cache(tmp_path)
    assert gcmod.main(["--cache-dir", str(cd)]) == 0
    out = capsys.readouterr().out
    assert "would remove" in out and "2 store entries" in out
    assert (cd / "store" / "ccc").exists()


def test_missing_cache_is_a_noop(tmp_path):
    r = gcmod.gc_cache(tmp_path / "nothing")
    assert r["entries_removed"] == [] and not r["executed"]

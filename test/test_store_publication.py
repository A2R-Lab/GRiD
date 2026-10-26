"""Publication failures and synchronized same-process cache writers, CPU only."""
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import multiprocessing
import threading

import pytest

from grid_rbd import _store


def _stage(root, value):
    stage = _store.staging_dir(root, "same-key")
    (stage / "grid.cuh").write_text("header")
    (stage / "robot.so").write_text(value)
    (stage / "meta.json").write_text(json.dumps({"value": value}))
    return stage


def test_same_process_staging_is_private(tmp_path):
    a = _stage(tmp_path, "a")
    b = _stage(tmp_path, "b")
    assert a != b and (a / "robot.so").read_text() == "a"


@pytest.mark.parametrize("existing", ["missing", "partial", "bad-json"])
def test_concurrent_repair_adopts_only_one_complete_winner(tmp_path, existing):
    entry = tmp_path / "key"
    if existing != "missing":
        entry.mkdir()
        (entry / "meta.json").write_text("{" if existing == "bad-json" else "{}")
    a, b = _stage(tmp_path, "a"), _stage(tmp_path, "b")
    barrier = threading.Barrier(2)

    def publish(stage):
        barrier.wait(timeout=10)
        return _store.publish_entry(stage, entry, "robot.so")

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(publish, [a, b]))
    assert results[0] == results[1] == _store.entry_metadata(entry, "robot.so")
    assert (entry / "robot.so").read_text() == results[0]["value"]
    assert not list(tmp_path.glob(".replaced_*"))


def test_publish_io_failure_restores_previous_entry(tmp_path, monkeypatch):
    entry = _stage(tmp_path, "old")
    stage = _stage(tmp_path, "new")
    original = _store.os.replace

    def fail(src, dst):
        if src == stage:
            raise PermissionError(errno.EACCES, "injected failure")
        return original(src, dst)

    monkeypatch.setattr(_store.os, "replace", fail)
    with pytest.raises(PermissionError, match="injected"):
        _store.publish_entry(stage, entry, "robot.so", force=True)
    assert _store.entry_metadata(entry, "robot.so") == {"value": "old"}
    assert (stage / "robot.so").read_text() == "new"


def test_object_publish_uses_unique_thread_temporaries(tmp_path, monkeypatch):
    a, b = tmp_path / "a.o", tmp_path / "b.o"
    a.write_bytes(b"object")
    b.write_bytes(b"object")
    cached = tmp_path / "cached.o"
    original = _store.os.replace
    barrier = threading.Barrier(2)
    names = []

    def synchronized(src, dst):
        names.append(src)
        barrier.wait(timeout=10)
        return original(src, dst)

    monkeypatch.setattr(_store.os, "replace", synchronized)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda obj: _store.publish_object(obj, cached), [a, b]))
    assert len(set(names)) == 2
    assert cached.read_bytes() == b"object" and not list(tmp_path.glob("*.tmp"))


def _process_publish(root, value, barrier, results):
    """Top-level target also exercises separately imported stores under spawn."""
    stage = _stage(root, value)
    barrier.wait(timeout=20)
    meta = _store.publish_entry(stage, root / "key", "robot.so")
    results.put(meta)


@pytest.mark.parametrize("existing", ["missing", "partial", "bad-json"])
def test_process_repair_shares_the_external_lock_inode(tmp_path, existing):
    entry = tmp_path / "key"
    if existing != "missing":
        entry.mkdir()
        (entry / "meta.json").write_text("{" if existing == "bad-json" else "{}")
    ctx = multiprocessing.get_context("spawn")
    barrier, results = ctx.Barrier(2), ctx.Queue()
    workers = [ctx.Process(target=_process_publish, args=(tmp_path, value, barrier, results))
               for value in ("a", "b")]
    try:
        for worker in workers:
            worker.start()
        values = [results.get(timeout=30) for _ in workers]
        for worker in workers:
            worker.join(timeout=20)
            assert worker.exitcode == 0
        assert values[0] == values[1] == _store.entry_metadata(entry, "robot.so")
        assert (entry / "robot.so").read_text() == values[0]["value"]
        assert not list(tmp_path.glob(".replaced_*"))
    finally:
        for worker in workers:
            if worker.pid is not None and worker.is_alive():
                worker.terminate()  # only this test's own child
                worker.join(timeout=10)
        results.close()


@pytest.mark.parametrize("empty", ["robot.so", "grid.cuh"])
def test_empty_runtime_file_is_not_a_complete_entry(tmp_path, empty):
    stage = _stage(tmp_path, "valid")
    (stage / empty).write_bytes(b"")
    assert _store.entry_metadata(stage, "robot.so") is None
    with pytest.raises(RuntimeError, match="incomplete artifact"):
        _store.publish_entry(stage, tmp_path / "key", "robot.so")
    assert not (tmp_path / "key").exists()

"""Publication failures and synchronized same-process cache writers, CPU only."""
from concurrent.futures import ThreadPoolExecutor
import errno
import json
import multiprocessing
import os
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


def _interrupted_publish(root, checkpoint):
    """Abrupt death while holding the entry lock: no Python finally/rollback."""
    entry = root / "key"
    staged = _stage(root, "interrupted")
    replace = _store.os.replace

    def interrupt(src, dst):
        if src == staged and checkpoint == "before-install":
            os._exit(73)
        replace(src, dst)
        if src == staged and checkpoint == "after-install":
            os._exit(73)

    _store.os.replace = interrupt
    _store.publish_entry(staged, entry, "robot.so", force=True)


@pytest.mark.parametrize("checkpoint", ["before-install", "after-install"])
@pytest.mark.parametrize("existing", ["complete", "damaged"])
def test_retry_after_process_death_releases_lock_and_preserves_objects(tmp_path, checkpoint, existing):
    entry = tmp_path / "key"
    _stage(tmp_path, "old").rename(entry)
    if existing == "damaged":
        (entry / "robot.so").unlink()
    objects = tmp_path / "objects"
    objects.mkdir()
    cached = objects / "retained.o"
    cached.write_bytes(b"reusable object")
    ctx = multiprocessing.get_context("spawn")
    worker = ctx.Process(target=_interrupted_publish, args=(tmp_path, checkpoint))
    try:
        worker.start()
        worker.join(timeout=20)
        assert worker.exitcode == 73
    finally:
        if worker.pid is not None and worker.is_alive():
            worker.terminate()
            worker.join(timeout=10)
    # A crash can leave a backup and a temporarily absent entry. It must not
    # advertise a partial result or strand the OS lock. Retrying repairs it;
    # no scavenging/deletion of another process's backup is required.
    observed = _store.entry_metadata(entry, "robot.so")
    assert observed == ({"value": "interrupted"} if checkpoint == "after-install" else None)
    results, barrier = ctx.Queue(), ctx.Barrier(1)
    retry = ctx.Process(target=_process_publish, args=(tmp_path, "retry", barrier, results))
    try:
        retry.start()
        winner = results.get(timeout=30)
        retry.join(timeout=20)
        assert retry.exitcode == 0
        assert winner == {"value": "interrupted" if checkpoint == "after-install" else "retry"}
        assert (entry / "robot.so").read_text() == winner["value"]
        assert cached.read_bytes() == b"reusable object"
        assert len(list(tmp_path.glob(".replaced_*"))) == 1
    finally:
        if retry.pid is not None and retry.is_alive():
            retry.terminate()
            retry.join(timeout=10)
        results.close()

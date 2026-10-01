"""CPU-only guard: the cuda harness header cache is one-writer-per-key.

Until 2026-10-01 the harness caches had no writer locking, so the split driver
ran GPU shards strictly one at a time. `_cache_key_lock` (a per-key flock, the
executable_cache idiom) is what makes GRID_SPLIT_SHARD_JOBS>1 sound: two shards
arriving at the same missing key must produce ONE generation, and the second
must read the first's result. The test drives the real `_generate_grid_header`
with the generator stubbed out (no nvcc, no robot) and two contending threads
(flock is per open file description, so threads contend like processes).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from test.cuda_equivalents import cuda_harness


def test_header_cache_generates_once_under_contention(tmp_path, monkeypatch):
    monkeypatch.setenv("GRID_CUDA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GRID_CUDA_PROGRESS", "0")
    monkeypatch.delenv("GRID_CUDA_DISABLE_CACHE", raising=False)
    generations = []

    def fake_gen(codegen, project_model, header_path, include_homogenous_transforms,
                 codegen_algorithm_list=None):
        generations.append(threading.get_ident())
        time.sleep(0.4)                         # long enough for the second thread to arrive
        Path(header_path).write_text("// stub header\n")

    monkeypatch.setattr(cuda_harness, "_run_gen_all_code", fake_gen)
    monkeypatch.setattr(cuda_harness, "GRiDCodeGenerator", lambda *a, **k: object())
    monkeypatch.setattr(cuda_harness, "_header_cache_key", lambda *a, **k: "0" * 64)
    model = SimpleNamespace(spec=SimpleNamespace(robot_id="stub"), base_mode="fixed", robot=None)

    results = []

    def worker(i):
        build_dir = tmp_path / f"build{i}"
        build_dir.mkdir()
        results.append(cuda_harness._generate_grid_header(model, None, build_dir, None))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(generations) == 1, "two contenders generated the same key twice (lock not held)"
    assert len(results) == 2
    assert all(path.read_text() == "// stub header\n" for path, _key in results)
    assert (tmp_path / "cache" / "headers" / ("0" * 64) / "grid.cuh").exists()
    assert (tmp_path / "cache" / "headers" / ("0" * 64 + ".lock")).exists()


def test_cache_key_lock_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("GRID_CUDA_CACHE_DIR", str(tmp_path / "cache"))
    order = []

    def holder():
        with cuda_harness._cache_key_lock("runners", "abc"):
            order.append("a-in")
            time.sleep(0.3)
            order.append("a-out")

    def waiter():
        time.sleep(0.05)
        with cuda_harness._cache_key_lock("runners", "abc"):
            order.append("b-in")

    a, b = threading.Thread(target=holder), threading.Thread(target=waiter)
    a.start(); b.start(); a.join(); b.join()
    assert order == ["a-in", "a-out", "b-in"]

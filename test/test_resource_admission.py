"""Controlled pressure/failure tests; never exhaust the real machine's RAM."""
import json
import sys
import threading

import pytest

from test.resource_admission import MemoryAdmission
from test import compile_sched


def budget(free, resident=None, log=None):
    return MemoryAdmission(floor_kb=20, available=lambda: free[0],
        rss=lambda pid: (resident or {}).get(pid, 0), log_path=log)


def test_reservations_are_atomic_and_do_not_double_count_rss(tmp_path):
    free, rss = [100], {}
    b = budget(free, rss, tmp_path / "memory.jsonl")
    assert b.reserve("pool", 50)
    b.attach("pool", 123)
    assert not b.reserve("shard", 40)
    # 30 KiB of the pool's allowance is now resident, already removed from free.
    free[0], rss[123] = 70, 30
    assert b.reserve("shard", 30)  # 70 - (50-30) - 30 == 20
    assert not b.reserve("extra", 1)
    assert b.release("pool") == 30
    b.release("shard")
    assert not b.active
    events = [json.loads(x) for x in (tmp_path / "memory.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["admit", "admit", "release", "release"]


def test_competing_admissions_cannot_spend_the_same_headroom():
    b = budget([100])
    barrier = threading.Barrier(3)
    results = []
    def reserve(name):
        barrier.wait()
        results.append(b.reserve(name, 60))
    threads = [threading.Thread(target=reserve, args=(str(i),)) for i in range(2)]
    for t in threads: t.start()
    barrier.wait()
    for t in threads: t.join()
    assert sorted(results) == [False, True]


def test_pressure_latches_drain_only_and_unknown_is_not_zero():
    free = [100]
    b = budget(free)
    with pytest.raises(ValueError): b.reserve("unknown", 0)
    assert b.reserve("running", 50)
    free[0] = 19
    assert b.pressure()
    free[0] = 100
    assert not b.reserve("later", 1)
    b.release("running")
    assert not b.active and b.blocked


def test_zero_headroom_remains_in_minimum_after_recovery():
    free = [0]
    b = budget(free)
    assert b.pressure()
    free[0] = 100
    b.pressure()
    assert b.min_available_kb == 0


@pytest.mark.parametrize("returncode", [0, 7])
def test_scheduler_releases_success_and_failure(tmp_path, monkeypatch, returncode):
    monkeypatch.setattr(compile_sched, "mem_available_kb", lambda: 100)
    b = budget([100])
    pool = compile_sched.RamScheduler(compile_sched.Ledger(tmp_path / "rss.json"),
        max_jobs=2, floor_kb=20, margin=1, default_peak_kb=30,
        strict_memory=True, admission=b)
    jobs = [compile_sched.Job(str(i), [sys.executable, "-c", f"raise SystemExit({returncode})"],
            str(i), log_path=str(tmp_path / f"{i}.log")) for i in range(2)]
    results = pool.run(jobs)
    assert len(results) == 2 and {r.rc for r in results.values()} == {returncode}
    assert not b.active


def test_no_admission_is_a_bounded_refusal_not_a_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(compile_sched, "mem_available_kb", lambda: 25)
    b = budget([25])
    pool = compile_sched.RamScheduler(compile_sched.Ledger(tmp_path / "rss.json"),
        floor_kb=20, margin=1, default_peak_kb=30, strict_memory=True, admission=b)
    result = pool.run([compile_sched.Job("never", ["must-not-run"], "never")])
    assert result["never"].rc == -1 and pool.resource_blocked and not b.active


def test_sampler_includes_new_session_descendants(tmp_path):
    child = "import time; b=bytearray(32*1024*1024); time.sleep(2)"
    parent = ("import subprocess,sys; p=subprocess.Popen([sys.executable,'-c',%r],"
              "start_new_session=True); p.wait()" % child)
    pool = compile_sched.RamScheduler(compile_sched.Ledger(tmp_path / "rss.json"),
        max_jobs=1, floor_kb=0, default_peak_kb=1)
    result = pool.run([compile_sched.Job("nested", [sys.executable,"-c",parent],
        "nested", log_path=str(tmp_path / "nested.log"))])["nested"]
    assert result.rc == 0 and result.peak_kb > 36 * 1024


def test_stopped_scheduler_releases_owned_running_reservations(tmp_path, monkeypatch):
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: 100)
    b = budget([100])
    pool = compile_sched.RamScheduler(compile_sched.Ledger(tmp_path / 'rss.json'),
        max_jobs=1, floor_kb=20, margin=1, default_peak_kb=30, strict_memory=True, admission=b)
    entered = threading.Event()
    original = pool._launch
    def launch(*args):
        launched = original(*args)
        entered.set()
        return launched
    monkeypatch.setattr(pool, '_launch', launch)
    jobs = [compile_sched.Job(str(i), [sys.executable, '-c', 'import time; time.sleep(.2)'],
            str(i), log_path=str(tmp_path / f'{i}.log')) for i in range(2)]
    thread = pool.run_async(jobs)
    assert entered.wait(5)
    pool.stop()
    thread.join(5)
    assert not thread.is_alive() and not b.active
    assert pool.results['0'].rc == 0 and pool.results['1'].rc == -1


@pytest.mark.parametrize('failure', ['launch', 'collect'])
def test_scheduler_exception_drains_owned_jobs_and_reservations(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: 100)
    b = budget([100])
    pool = compile_sched.RamScheduler(compile_sched.Ledger(tmp_path / 'rss.json'),
        max_jobs=2, floor_kb=20, margin=1, default_peak_kb=30, strict_memory=True, admission=b)
    original = pool._launch
    def launch(job, *args):
        if job.name == '1' and failure == 'launch':
            raise OSError('controlled launch failure')
        return original(job, *args)
    monkeypatch.setattr(pool, '_launch', launch)
    if failure == 'collect':
        def collect(*args):
            raise OSError('controlled collect failure')
        monkeypatch.setattr(pool, '_collect', collect)
    jobs = [compile_sched.Job(str(i), [sys.executable, '-c', 'import time; time.sleep(.2)'],
            str(i), log_path=str(tmp_path / f'{i}.log')) for i in range(2)]
    pool.done_events = {job.name: threading.Event() for job in jobs}
    with pytest.raises(OSError, match='controlled'):
        pool._run_pool(jobs)
    assert b.blocked and not b.active and not pool.running_pids()
    assert all(event.is_set() for event in pool.done_events.values())
    assert {result.rc for result in pool.results.values()} == {-1}

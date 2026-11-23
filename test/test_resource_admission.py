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


def test_memory_cap_is_an_extra_refusal_and_never_relaxes_the_reserve():
    free, rss = [1000], {}
    capped = MemoryAdmission(floor_kb=20, available=lambda: free[0],
                             rss=lambda pid: rss.get(pid, 0), cap_kb=100)
    assert capped.reserve("a", 60)
    capped.attach("a", 1)
    assert not capped.reserve("b", 41) and "memory cap 100 KiB" in capped.refusal
    assert capped.reserve("b", 40)
    capped.attach("b", 2)
    # Resident pages still count against the cap; growth past an allowance counts in full.
    free[0], rss[1], rss[2] = 900, 60, 40
    assert not capped.reserve("c", 1)
    rss[1] = 80
    capped.release("b")
    assert not capped.reserve("c", 21) and capped.reserve("c", 20)
    capped.release("a"); capped.release("c")
    # A cap above the live headroom changes nothing: the reserve still refuses.
    free[0] = 100
    loose = MemoryAdmission(floor_kb=20, available=lambda: free[0], rss=lambda pid: 0, cap_kb=10_000)
    plain = budget(free)
    for b in (loose, plain):
        assert not b.reserve("big", 81) and "reserve 20 KiB" in b.refusal
        assert b.reserve("fits", 80) and not b.reserve("extra", 1)
    assert capped.blocked is None and loose.blocked is None


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


# --- adaptive profile: pressure is a wait, not a stop -----------------------

def test_unlatched_pressure_refuses_then_resumes_when_headroom_returns(tmp_path):
    free, now = [100], [0.0]
    b = MemoryAdmission(floor_kb=20, available=lambda: free[0], rss=lambda pid: 0,
                        log_path=tmp_path / "m.jsonl", latch=False, patience_s=60, clock=lambda: now[0])
    free[0] = 10                               # another user took the memory
    assert not b.pressure() and b.blocked is None
    assert not b.reserve("a", 5) and not b.give_up()
    now[0] = 59
    assert not b.give_up()
    free[0] = 100                              # it came back
    assert b.reserve("a", 50)
    b.release("a")
    free[0] = 10
    assert not b.reserve("b", 5) and not b.give_up()   # the clock restarts after an admission
    now[0] = 59 + 61
    assert b.give_up() and b.waits == 2
    events = [json.loads(x)["event"] for x in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert events == ["wait", "admit", "release", "wait"]


def test_unlatched_never_gives_up_while_it_holds_work_and_halt_still_stops():
    free, now = [100], [0.0]
    b = MemoryAdmission(floor_kb=20, available=lambda: free[0], rss=lambda pid: 0,
                        latch=False, patience_s=1, clock=lambda: now[0])
    assert b.reserve("a", 70) and not b.reserve("b", 20)
    now[0] = 100
    assert not b.give_up()
    b.halt("scheduler failure")
    assert b.pressure() and b.give_up()


def test_latched_profiles_give_up_at_once_as_before():
    b = budget([10])
    assert b.pressure() and not b.reserve("a", 1) and b.give_up()


def _job(name, tmp_path, **kw):
    return compile_sched.Job(name, [sys.executable, "-c", "print('compiling runner')"], name,
                             log_path=str(tmp_path / f"{name}.log"), **kw)


class _History:
    shows_build = staticmethod(lambda text: "compiling runner" in text)

    def __init__(self, peaks):
        self.peaks, self.recorded = peaks, []

    def predict_kb(self, kind, robot, name, units, floor_kb):
        return self.peaks.get(name, floor_kb)

    def record(self, *row):
        self.recorded.append(row[:4])


def test_scheduler_charges_the_measured_class_and_records_real_builds(tmp_path, monkeypatch):
    monkeypatch.setattr(compile_sched, "mem_available_kb", lambda: 1000)
    history = _History({"seen": 100})
    admission = MemoryAdmission(floor_kb=20, available=lambda: 1000, rss=lambda pid: 0,
                                log_path=tmp_path / "m.jsonl", latch=False)
    pool = compile_sched.RamScheduler(compile_sched.Ledger(tmp_path / "l.json"), max_jobs=1,
        floor_kb=20, margin=1.5, default_peak_kb=400, strict_memory=True,
        admission=admission, class_history=history)
    pool.run([_job("a", tmp_path, klass=("native", "go2", "seen", 1), peak_floor_kb=400),
              _job("b", tmp_path, klass=("native", "go2", "new", 1), peak_floor_kb=400),
              _job("c", tmp_path, peak_floor_kb=300)])
    admits = {e["token"]: e["predicted_kb"] for e in map(json.loads,
              (tmp_path / "m.jsonl").read_text().splitlines()) if e["event"] == "admit"}
    assert admits == {"compile:a": 150, "compile:b": 600, "compile:c": 450}
    assert history.recorded == [("native", "go2", "seen", 1), ("native", "go2", "new", 1)]
    assert all(r.rc == 0 for r in pool.results.values())


def test_scheduler_waits_out_pressure_instead_of_stopping(tmp_path, monkeypatch):
    free = [10]
    monkeypatch.setattr(compile_sched, "mem_available_kb", lambda: free[0])
    monkeypatch.setattr(compile_sched.time, "sleep", lambda s: free.__setitem__(0, 1000))
    admission = MemoryAdmission(floor_kb=20, available=lambda: free[0], rss=lambda pid: 0,
                                latch=False, patience_s=3600)
    pool = compile_sched.RamScheduler(compile_sched.Ledger(tmp_path / "l.json"), max_jobs=2,
        floor_kb=20, default_peak_kb=100, strict_memory=True, admission=admission)
    pool.run([_job("a", tmp_path, peak_floor_kb=100)])
    assert pool.results["a"].rc == 0 and pool.resource_blocked is None and admission.waits == 1


def test_scheduler_pauses_when_headroom_never_returns(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(compile_sched, "mem_available_kb", lambda: 10)
    monkeypatch.setattr(compile_sched.time, "sleep", lambda s: now.__setitem__(0, now[0] + 40))
    admission = MemoryAdmission(floor_kb=20, available=lambda: 10, rss=lambda pid: 0,
                                latch=False, patience_s=60, clock=lambda: now[0])
    pool = compile_sched.RamScheduler(compile_sched.Ledger(tmp_path / "l.json"), max_jobs=2,
        floor_kb=20, default_peak_kb=100, strict_memory=True, admission=admission)
    pool.run([_job("a", tmp_path, peak_floor_kb=100)])
    assert pool.results["a"].rc == -1 and "reserve 20 KiB" in pool.resource_blocked


def test_cpu_budget_counts_units_and_other_load(tmp_path, monkeypatch):
    monkeypatch.setattr(compile_sched, "mem_available_kb", lambda: 10 ** 9)
    hold = tmp_path / "hold"
    hold.touch()
    wait = [sys.executable, "-c",
            f"import os,time\nwhile os.path.exists({str(hold)!r}): time.sleep(.05)"]
    load = [0.0]
    pool = compile_sched.RamScheduler(compile_sched.Ledger(tmp_path / "l.json"), max_jobs=8,
        floor_kb=1, default_peak_kb=1, cpu_budget=8, loadavg=lambda: load[0])
    jobs = [compile_sched.Job(f"w{i}", wait, f"w{i}", cpus=4) for i in range(3)]
    load[0] = 6.0                      # six cores busy elsewhere: one job still runs
    pool.run_async(jobs)
    deadline = compile_sched.time.monotonic() + 20
    while len(pool.running_pids()) < 1 and compile_sched.time.monotonic() < deadline:
        compile_sched.time.sleep(.05)
    compile_sched.time.sleep(1.5)
    assert len(pool.running_pids()) == 1
    load[0] = 4.0                      # only our own job is loading the machine now
    while len(pool.running_pids()) < 2 and compile_sched.time.monotonic() < deadline:
        compile_sched.time.sleep(.05)
    compile_sched.time.sleep(1.5)
    assert len(pool.running_pids()) == 2   # 4 + 4 fill 8 cores; the third waits
    hold.unlink()
    pool.wait([j.name for j in jobs])
    assert all(pool.results[j.name].rc == 0 for j in jobs)

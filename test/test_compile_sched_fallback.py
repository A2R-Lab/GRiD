"""The compile scheduler on a box without GNU time, and a pool thread that
cannot launch: both must be loud, neither may hang or silently do nothing
(the first Orin receipt, 2026-10-16, compiled nothing through the pool for
hours because the missing /usr/bin/time raised inside the pool thread)."""
import sys
import time

sys.path.insert(0, "test")
import compile_sched  # noqa: E402


def _sched(tmp_path):
    return compile_sched.RamScheduler(compile_sched.Ledger(tmp_path / "rss.json"),
                                      max_jobs=1, floor_kb=0, default_peak_kb=1)


def test_jobs_run_without_gnu_time(tmp_path, monkeypatch):
    monkeypatch.setattr(compile_sched, "_HAVE_TIME_BIN", False)
    sched = _sched(tmp_path)
    job = compile_sched.Job(name="hello", argv=[sys.executable, "-c", "print('ran')"],
                            ledger_key="hello", log_path=str(tmp_path / "hello.log"))
    results = sched.run([job])
    assert results["hello"].rc == 0
    assert "ran" in (tmp_path / "hello.log").read_text()


def test_launch_failure_resolves_every_job_and_returns(tmp_path, capsys, monkeypatch):
    # Without the GNU time wrapper a missing program raises at Popen (with the
    # wrapper, time itself reports rc 127 and the pool carries on).
    monkeypatch.setattr(compile_sched, "_HAVE_TIME_BIN", False)
    sched = _sched(tmp_path)
    jobs = [compile_sched.Job(name=f"j{i}", argv=["/nonexistent/compiler", "x"], ledger_key=f"j{i}",
                              log_path=str(tmp_path / f"j{i}.log")) for i in range(3)]
    thread = sched.run_async(jobs)
    thread.join(timeout=20)
    assert not thread.is_alive()
    for j in jobs:
        assert sched.done_events[j.name].is_set()
        assert sched.results[j.name].rc == -1
        assert "scheduler stopped" in sched.results[j.name].stdout_tail
    assert "FATAL: scheduler stopped" in capsys.readouterr().out

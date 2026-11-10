"""Receipt resource admission must not bypass memory or lose pending tests."""
import sys
from types import SimpleNamespace

import pytest

from test import run_split_suite as driver

sys.path.insert(0, 'test')
import compile_sched


def test_first_measurement_creates_missing_ledger_directory(tmp_path):
    path = tmp_path/'new-checkout'/'.split_suite'/'compile_rss.json'
    ledger = compile_sched.Ledger(path)
    assert not path.parent.exists()
    ledger.record('cold', 200, secs=3)
    loaded = compile_sched.Ledger(path)
    assert loaded.predict_kb('cold', 100, 1) == 200
    assert loaded.predict_secs('cold') == 3
    loaded.record('cold', 50, secs=1)
    assert compile_sched.Ledger(path).predict_kb('cold', 100, 1) == 200


@pytest.mark.parametrize('available', [0, 100, 119])
def test_strict_first_job_refuses_without_launch(tmp_path, monkeypatch, available):
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: available)
    monkeypatch.setattr(compile_sched.subprocess, 'Popen',
                        lambda *a, **k: pytest.fail('insufficient memory must not launch'))
    scheduler = compile_sched.RamScheduler(compile_sched.Ledger(tmp_path/'rss.json'),
        max_jobs=1, floor_kb=20, default_peak_kb=100, margin=1, strict_memory=True)
    jobs = [compile_sched.Job(n, ['never'], n) for n in ('a', 'b')]
    results = scheduler.run(jobs)
    assert scheduler.resource_blocked
    assert set(results) == {'a', 'b'}
    assert all(row.rc == -1 and 'resource blocked' in row.stdout_tail for row in results.values())
    assert all(event.is_set() for event in scheduler.done_events.values())


def test_strict_floor_cannot_be_reduced_by_small_cached_peak(tmp_path, monkeypatch):
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: 119)
    ledger = compile_sched.Ledger(tmp_path/'rss.json')
    ledger.record('small', 1)
    scheduler = compile_sched.RamScheduler(ledger, floor_kb=20, default_peak_kb=100,
                                           margin=1, strict_memory=True)
    result = scheduler.run([compile_sched.Job('small', ['never'], 'small')])
    assert result['small'].rc == -1 and 'predicted 100 KiB' in scheduler.resource_blocked


def test_strict_admission_at_boundary_and_after_refusal(tmp_path, monkeypatch):
    memory = [119]
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: memory[0])
    ledger = compile_sched.Ledger(tmp_path/'rss.json')
    jobs = [compile_sched.Job(n, [sys.executable, '-c', 'print("executed")'], n,
                              log_path=str(tmp_path/f'{n}.log')) for n in ('a', 'b')]
    def fresh():
        return compile_sched.RamScheduler(ledger, max_jobs=1, floor_kb=20,
                                          default_peak_kb=100, margin=1, strict_memory=True)
    assert all(r.rc == -1 for r in fresh().run(jobs).values())
    assert not (tmp_path/'a.log').exists()
    memory[0] = 120
    scheduler = fresh()
    assert all(r.rc == 0 for r in scheduler.run(jobs).values())
    assert scheduler.resource_blocked is None
    assert all((tmp_path/f'{n}.log').read_text().count('executed') == 1 for n in ('a', 'b'))


def test_strict_admission_retains_larger_recorded_peak(tmp_path, monkeypatch):
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: 219)
    monkeypatch.setattr(compile_sched.subprocess, 'Popen',
                        lambda *a, **k: pytest.fail('recorded peak must not be capped'))
    ledger = compile_sched.Ledger(tmp_path/'rss.json')
    ledger.record('large', 200)
    scheduler = compile_sched.RamScheduler(ledger, floor_kb=20, default_peak_kb=100,
                                           margin=1, strict_memory=True)
    result = scheduler.run([compile_sched.Job('large', ['never'], 'large')])
    assert result['large'].rc == -1
    assert 'predicted 200 KiB' in scheduler.resource_blocked


def fake_pool(events, out, *, pause=False, blocked=None):
    class Thread:
        alive = True
        def is_alive(self):
            return self.alive
        def join(self, timeout):
            assert timeout <= 1
            events.append('compile-finished')
            self.alive = False
    class Pool:
        resource_blocked = blocked
        def run_async(self, jobs):
            events.append('compile-started')
            if pause:
                (out/'PAUSE').touch()
            return Thread()
        def stop(self):
            events.append('compile-stop-requested')
    return Pool()


def test_serial_phases_finish_pool_before_all_shards(tmp_path, monkeypatch):
    monkeypatch.setenv('GRID_SPLIT_RESOURCE_PROFILE', 'serial')
    events = []
    prior, shards = [{'shard': 'already-clean'}], ['a', 'b', 'c']
    def run(actual, out, receipts, extra, saved, **kw):
        assert events == ['compile-started', 'compile-finished']
        assert actual == shards and saved == prior
        events.append('all-shards')
        return saved + [{'shard': s} for s in actual], False
    monkeypatch.setattr(driver, 'phase_run', run)
    result, paused = driver.run_phases(shards, tmp_path, True, [], prior,
        fake_pool(events, tmp_path), ['warm'], {'a': ['warm']})
    assert not paused and [r['shard'] for r in result] == ['already-clean', 'a', 'b', 'c']


@pytest.mark.parametrize('blocked', [None, 'first job exceeds allowance'])
def test_compile_phase_pause_preserves_prior_without_dispatch(tmp_path, monkeypatch, blocked):
    monkeypatch.setenv('GRID_SPLIT_RESOURCE_PROFILE', 'serial')
    monkeypatch.setattr(driver, 'phase_run', lambda *a, **k: pytest.fail('must not launch shard'))
    events, prior = [], [{'shard': 'already-clean'}]
    result, paused = driver.run_phases(['pending'], tmp_path, True, [], prior,
        fake_pool(events, tmp_path, pause=blocked is None, blocked=blocked), ['warm'], {})
    assert paused and result == prior and (tmp_path/'PAUSE').exists()
    if blocked:
        assert blocked in (tmp_path/'resource-blocked.json').read_text()
    else:
        assert 'compile-stop-requested' in events


def test_existing_pause_never_starts_pool(tmp_path, monkeypatch):
    monkeypatch.setenv('GRID_SPLIT_RESOURCE_PROFILE', 'serial')
    (tmp_path/'PAUSE').touch()
    events = []
    assert driver.run_phases(['pending'], tmp_path, False, [], [],
        fake_pool(events, tmp_path), ['warm'], {}) == ([], True)
    assert events == []
    assert (tmp_path/'results.json').read_text().strip() == '[]'


def test_wrappers_only_prewarm_pause_has_resume_anchor(tmp_path, monkeypatch):
    monkeypatch.setenv('GRID_SPLIT_RESOURCE_PROFILE', 'serial')
    monkeypatch.setattr(driver, 'phase_run', lambda *a, **k: pytest.fail('no shard yet'))
    results, paused = driver.run_phases(['wrapper'], tmp_path, False, [], [],
        fake_pool([], tmp_path, pause=True), ['warm'], {})
    assert paused and results == []
    assert not (tmp_path/'partition.json').exists()
    assert driver.load_prior_results(tmp_path, {'wrapper'}) == []
    assert (tmp_path/'results.json').is_file()


def test_shard_memory_refusal_then_resume_preserves_scope(tmp_path, monkeypatch):
    monkeypatch.setenv('GRID_SPLIT_RESOURCE_PROFILE', 'serial')
    monkeypatch.setenv('GRID_SPLIT_SHARD_JOBS', '9')
    assert driver.shard_jobs() == 1
    required = int(driver.SERIAL_PEAK_KB * driver.SERIAL_MARGIN) + driver.GPU_PHASE_FLOOR_KB
    memory = [required - 1]
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: memory[0])
    shards = [SimpleNamespace(name=n) for n in ('a', 'b')]
    launched = []
    def launch(spec, *a):
        launched.append(spec.name)
        # Once first shard finishes, new external pressure must block the next.
        if spec.name == 'a':
            memory[0] = required - 1
        return SimpleNamespace(spec=spec, proc=SimpleNamespace(pid=1), poll=lambda: 0, close=lambda: None)
    def finish(run, rc, results, out):
        results.append({'shard': run.spec.name, 'rc': rc})
    monkeypatch.setattr(driver, '_launch_shard', launch)
    monkeypatch.setattr(driver, '_finish_shard', finish)
    results, paused = driver.phase_run(shards, tmp_path, False, [], [])
    assert paused and results == [] and launched == []
    (tmp_path/'PAUSE').unlink()
    memory[0] = required
    results, paused = driver.phase_run(shards, tmp_path, False, [], results)
    assert paused and launched == ['a'] and results == [{'shard': 'a', 'rc': 0}]
    (tmp_path/'PAUSE').unlink()
    memory[0] = required
    results, paused = driver.phase_run(shards[1:], tmp_path, False, [], results)
    assert not paused and launched == ['a', 'b']
    assert [row['shard'] for row in results] == ['a', 'b']


def test_unknown_profile_is_rejected(monkeypatch):
    monkeypatch.setenv('GRID_SPLIT_RESOURCE_PROFILE', 'typo')
    with pytest.raises(SystemExit, match='Unknown'):
        driver.shard_jobs()


def test_direct_serial_profile_caps_worker_environment(monkeypatch):
    import os
    monkeypatch.setenv('GRID_SPLIT_RESOURCE_PROFILE', 'serial')
    keys = ('GRID_RBD_BUILD_JOBS', 'GRID_SPLIT_COMPILE_JOBS', 'GRID_SPLIT_SHARD_JOBS')
    for key in keys:
        monkeypatch.setenv(key, '9')
    observed = []
    monkeypatch.setattr(sys, 'argv', ['run_split_suite', '--warm-one', 'not-executed.json'])
    monkeypatch.setattr(driver, 'warm_one_worker', lambda path: observed.append(
        tuple(os.environ[key] for key in keys)) or 0)
    assert driver.main() == 0
    assert observed == [('2', '1', '1')]


def test_legacy_overlap_does_not_drain_pool_first(tmp_path, monkeypatch):
    monkeypatch.setenv('GRID_SPLIT_RESOURCE_PROFILE', 'overlap')
    events = []
    def run(*args, **kwargs):
        assert events == ['compile-started']
        return [], False
    monkeypatch.setattr(driver, 'phase_run', run)
    assert driver.run_phases(['a'], tmp_path, False, [], [],
        fake_pool(events, tmp_path), ['warm'], {}) == ([], False)

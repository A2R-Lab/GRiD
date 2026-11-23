"""Receipt resource admission must not bypass memory or lose pending tests."""
import sys
from types import SimpleNamespace

import pytest

from test import run_split_suite as driver

sys.path.insert(0, 'test')
import compile_sched


@pytest.fixture(autouse=True)
def synthetic_resource_keys(monkeypatch, tmp_path):
    # These state-machine tests do not load a GPU or hash the real checkout.
    monkeypatch.setattr(driver, 'shard_resource_key', lambda spec, args: 'synthetic:' + spec.name)
    monkeypatch.setattr(driver, 'REPO_ROOT', tmp_path)


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


@pytest.mark.parametrize('profile', ['serial', 'bounded-overlap'])
def test_profiles_execute_the_same_scope_once(tmp_path, monkeypatch, profile):
    from test.resource_admission import MemoryAdmission
    monkeypatch.setenv('GRID_SPLIT_RESOURCE_PROFILE', profile)
    monkeypatch.setenv('GRID_SPLIT_SHARD_JOBS', '99')
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: 100_000_000)
    assert driver.shard_jobs() == 1
    shards = [SimpleNamespace(name=n, targets=['synthetic']) for n in ('a', 'b')]
    launched = []
    def launch(spec, *args):
        launched.append(spec.name)
        return SimpleNamespace(spec=spec, proc=SimpleNamespace(pid=1), poll=lambda:0, close=lambda:None, t0=0)
    monkeypatch.setattr(driver, '_launch_shard', launch)
    monkeypatch.setattr(driver, '_finish_shard',
        lambda run, rc, results, out: results.append({'shard':run.spec.name, 'rc':rc}))
    admission = MemoryAdmission(floor_kb=driver.GPU_PHASE_FLOOR_KB,
        available=lambda:100_000_000, rss=lambda pid:0) if profile == 'bounded-overlap' else None
    result, paused = driver.phase_run(shards, tmp_path, False, [], [{'shard':'prior'}], admission=admission)
    assert not paused and launched == ['a', 'b']
    assert [r['shard'] for r in result] == ['prior','a','b']
    assert admission is None or not admission.active


def test_bounded_shard_pressure_preserves_completed_results(tmp_path, monkeypatch):
    from test.resource_admission import MemoryAdmission
    monkeypatch.setenv('GRID_SPLIT_RESOURCE_PROFILE', 'bounded-overlap')
    free = [100_000_000]
    b = MemoryAdmission(floor_kb=driver.GPU_PHASE_FLOOR_KB, available=lambda:free[0], rss=lambda pid:0)
    shards = [SimpleNamespace(name=n, targets=['synthetic']) for n in ('a','b')]
    def launch(spec, *args):
        def poll():
            free[0] = 1
            return 0
        return SimpleNamespace(spec=spec, proc=SimpleNamespace(pid=1), poll=poll, close=lambda:None, t0=0)
    monkeypatch.setattr(driver, '_launch_shard', launch)
    monkeypatch.setattr(driver, '_finish_shard',
        lambda run, rc, results, out: results.append({'shard':run.spec.name,'rc':rc}))
    result, paused = driver.phase_run(shards, tmp_path, False, [], [], admission=b)
    assert paused and result == [{'shard':'a','rc':0}] and not b.active
    assert (tmp_path/'PAUSE').exists()


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
    from test.resource_history import shard_floor_kb
    required = int(shard_floor_kb(['synthetic']) * driver.SERIAL_MARGIN) + driver.GPU_PHASE_FLOOR_KB
    memory = [required - 1]
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: memory[0])
    shards = [SimpleNamespace(name=n, targets=['synthetic']) for n in ('a', 'b')]
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


@pytest.mark.parametrize('cap, expected', [(None, 180), (500, 180), (180, 180), (150, 150)])
def test_memory_cap_lowers_but_never_raises_the_pool_budget(tmp_path, monkeypatch, cap, expected):
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: 200)
    scheduler = compile_sched.RamScheduler(compile_sched.Ledger(tmp_path/'rss.json'),
        floor_kb=20, default_peak_kb=100, margin=1, strict_memory=True, cap_kb=cap)
    assert scheduler.budget_kb == expected and scheduler.floor_kb == 20
    assert compile_sched.pool_budget_kb(200, 20, 100, cap) == expected


def test_capped_budget_refuses_what_the_uncapped_budget_admits(tmp_path, monkeypatch):
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: 200)
    ledger = compile_sched.Ledger(tmp_path/'rss.json')
    def run(cap):
        scheduler = compile_sched.RamScheduler(ledger, max_jobs=1, floor_kb=20,
            default_peak_kb=100, margin=1, strict_memory=True, cap_kb=cap)
        job = compile_sched.Job('a', [sys.executable, '-c', 'pass'], 'a', log_path=str(tmp_path/'a.log'))
        return scheduler, scheduler.run([job])['a'].rc
    scheduler, rc = run(99)
    assert rc == -1 and 'budget 99 KiB' in scheduler.resource_blocked
    assert not (tmp_path/'a.log').exists()
    assert run(100)[1] == 0 and run(None)[1] == 0


def test_cap_does_not_lift_the_live_backstop(tmp_path, monkeypatch):
    # 119 available cannot hold a 100 KiB job above the 20 KiB floor, whatever the cap.
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: 119)
    monkeypatch.setattr(compile_sched.subprocess, 'Popen',
                        lambda *a, **k: pytest.fail('a cap must not admit below the floor'))
    scheduler = compile_sched.RamScheduler(compile_sched.Ledger(tmp_path/'rss.json'),
        floor_kb=20, default_peak_kb=100, margin=1, strict_memory=True, cap_kb=10_000)
    assert scheduler.budget_kb == 100
    assert scheduler.run([compile_sched.Job('a', ['never'], 'a')])['a'].rc == -1


def direct_environment(monkeypatch, profile, **env):
    import os
    monkeypatch.setenv('GRID_SPLIT_RESOURCE_PROFILE', profile)
    keys = ('GRID_RBD_BUILD_JOBS', 'GRID_SPLIT_COMPILE_JOBS', 'GRID_SPLIT_SHARD_JOBS',
            'GRID_SPLIT_MEMORY_CAP_GIB')
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    observed = []
    monkeypatch.setattr(sys, 'argv', ['run_split_suite', '--warm-one', 'not-executed.json'])
    monkeypatch.setattr(driver, 'warm_one_worker', lambda path: observed.append(
        tuple(os.environ.get(key) for key in keys)) or 0)
    assert driver.main() == 0
    return observed[0]


def test_bounded_profile_defaults_and_stated_tuning_reach_workers(monkeypatch):
    assert direct_environment(monkeypatch, 'bounded-overlap') == ('2', '2', '1', None)
    assert direct_environment(monkeypatch, 'bounded-overlap', GRID_SPLIT_SHARD_JOBS='9',
        GRID_SPLIT_COMPILE_JOBS='4', GRID_RBD_BUILD_JOBS='3',
        GRID_SPLIT_MEMORY_CAP_GIB='40') == ('3', '4', '1', '40')
    assert driver.memory_cap_kb() == 40 * 1024 * 1024


def test_serial_profile_drops_a_stray_memory_cap(monkeypatch):
    assert direct_environment(monkeypatch, 'serial', GRID_SPLIT_COMPILE_JOBS='4',
        GRID_SPLIT_MEMORY_CAP_GIB='40') == ('2', '1', '1', None)
    assert driver.memory_cap_kb() is None


@pytest.mark.parametrize('key, value', [
    ('GRID_SPLIT_COMPILE_JOBS', '0'), ('GRID_SPLIT_COMPILE_JOBS', 'many'),
    ('GRID_RBD_BUILD_JOBS', '-1'), ('GRID_SPLIT_MEMORY_CAP_GIB', '0'),
    ('GRID_SPLIT_MEMORY_CAP_GIB', 'nan'), ('GRID_SPLIT_MEMORY_CAP_GIB', 'inf'),
    ('GRID_SPLIT_MEMORY_CAP_GIB', 'lots')])
def test_bounded_profile_rejects_malformed_tuning_before_work(monkeypatch, key, value):
    monkeypatch.setattr(driver, 'warm_one_worker', lambda path: pytest.fail('must not start work'))
    with pytest.raises(SystemExit, match=key):
        direct_environment(monkeypatch, 'bounded-overlap', **{key: value})


@pytest.mark.parametrize('profile, cap', [('bounded-overlap', 40 * 1024 * 1024), ('serial', None)])
def test_phases_give_the_cap_to_the_one_admission_authority(tmp_path, monkeypatch, capsys, profile, cap):
    monkeypatch.setenv('GRID_SPLIT_RESOURCE_PROFILE', profile)
    monkeypatch.setenv('GRID_SPLIT_MEMORY_CAP_GIB', '40')
    monkeypatch.setenv('GRID_SPLIT_COMPILE_JOBS', '4')
    monkeypatch.setenv('GRID_RBD_BUILD_JOBS', '3')
    seen = []
    monkeypatch.setattr(driver, 'phase_run', lambda *a, **k: seen.append(k['admission']) or ([], False))
    class Pool(fake_pool([], tmp_path).__class__):
        def run_async(self, jobs):
            return SimpleNamespace(join=lambda timeout=None: None, is_alive=lambda: False)
    pool = Pool()
    driver.run_phases(['a'], tmp_path, False, [], [], pool, ['warm'], {})
    assert seen[0].cap_kb == cap and pool.admission is seen[0]
    assert seen[0].floor_kb == driver.GPU_PHASE_FLOOR_KB
    out = capsys.readouterr().out
    if cap:
        assert 'at most 4 pool jobs / 1 shard / 3 units; memory cap 40 GiB' in out
    else:
        assert 'memory cap' not in out


def test_bounded_profile_line_is_unchanged_without_tuning(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('GRID_SPLIT_RESOURCE_PROFILE', 'bounded-overlap')
    for key in ('GRID_SPLIT_MEMORY_CAP_GIB', 'GRID_SPLIT_COMPILE_JOBS', 'GRID_RBD_BUILD_JOBS'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(driver, 'phase_run', lambda *a, **k: ([], False))
    driver.run_phases(['a'], tmp_path, False, [], [], None, [], {})
    assert ('Resource profile: bounded-overlap; at most 2 pool jobs / 1 shard / 2 units; '
            'one reservation authority includes inline builds') in capsys.readouterr().out


def test_resource_report_reads_meminfo_and_the_ledger_only(tmp_path, monkeypatch):
    gib = 1024 * 1024
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: 56 * gib)
    monkeypatch.setattr(compile_sched, 'meminfo_kb', lambda field: {'MemTotal': 62 * gib}[field])
    monkeypatch.setattr(driver.os, 'cpu_count', lambda: 24)
    monkeypatch.setattr(driver.subprocess, 'Popen', lambda *a, **k: pytest.fail('no process'))
    monkeypatch.setattr(driver.subprocess, 'run', lambda *a, **k: pytest.fail('no process'))
    report = driver.resource_report().splitlines()
    assert report[:5] == ['cores: 24', 'MemTotal: 62.0 GiB', 'MemAvailable: 56.0 GiB',
                          'host reserve (floor): 14.0 GiB', 'pool budget: 42.0 GiB']
    assert report[5].startswith('largest single allowance: 20.8 GiB') and '2 fit at once now' in report[5]
    assert report[6].startswith('largest recorded compile peak: none recorded')
    assert report[7] == 'serial: 1 compile slot, 2 units per build, 1 GPU shard'
    assert report[8].startswith('bounded-overlap: 2 compile slots, 2 units per build, 1 GPU shard')
    ledger = compile_sched.Ledger(tmp_path/'test/.split_suite/compile_rss.json')
    ledger.record('small', gib // 2)
    ledger.record('large', int(4.84 * gib))
    before = sorted(p for p in tmp_path.rglob('*'))
    capped = driver.resource_report(30 * gib).splitlines()
    assert 'pool budget with --memory-cap 30: 30.0 GiB' in capped
    assert any('1 fit at once now' in line for line in capped)
    assert 'largest recorded compile peak: 4.84 GiB (2 ledger entries)' in capped
    assert 'pool budget with --memory-cap 99: 42.0 GiB' in driver.resource_report(99 * gib)
    assert sorted(p for p in tmp_path.rglob('*')) == before
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: 30 * gib)
    assert 'would be refused now' in driver.resource_report()

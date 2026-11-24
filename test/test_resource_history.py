"""Resource measurements are not portable across changed build identities."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import compile_sched
import resource_history as history


def test_driver_preserves_complete_existing_build_identity(tmp_path, monkeypatch):
    import run_split_suite as driver
    from grid_rbd import _cache
    payload = {'generation_env': {'GRID_FDSVA_SO_MINV_TILE': '8'},
               'cuda_archs': [89, 120], 'torch_abi': 'example',
               'future_identity_field': 'must-not-be-dropped'}
    monkeypatch.delattr(driver.resource_environment, '_snapshot', raising=False)
    monkeypatch.setattr(_cache, 'detect_cuda_arch', lambda: 120)
    monkeypatch.setattr(_cache, 'build_identity', lambda arch: dict(payload))
    monkeypatch.setattr(_cache, '_codegen_source_hash', lambda: 'generator')
    monkeypatch.setattr(_cache, 'package_version', lambda: 'test')
    monkeypatch.setattr(driver, '_nvcc_version', lambda: 'nvcc-test')
    monkeypatch.setattr(driver.subprocess, 'check_output', lambda *a, **kw: 'test\n')
    try:
        env = driver.resource_environment()
        assert env['build_identity'] == payload
        before = history.identity(tmp_path, [], {}, env)
        del driver.resource_environment._snapshot
        payload['generation_env'] = {'GRID_FDSVA_SO_MINV_TILE': '16'}
        assert history.identity(tmp_path, [], {}, driver.resource_environment()) != before
    finally:
        if hasattr(driver.resource_environment, '_snapshot'):
            del driver.resource_environment._snapshot


def test_identity_tracks_bytes_recipe_and_environment_not_location(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    for root in (a, b):
        root.mkdir()
        (root / 'source.py').write_text('old')
    def key(root=a, recipe=None, env=None):
        return history.identity(root, ['source.py'], recipe or {'robot': 'arm'},
                                env or {'units': 2, 'nvcc': 'toolchain1'})
    before = key()
    assert before == key(b)
    assert before != key(recipe={'robot': 'other'})
    assert before != key(env={'units': 4, 'nvcc': 'toolchain1'})
    assert before != key(env={'units': 2, 'nvcc': 'toolchain2'})
    (a / 'source.py').write_text('new')
    assert before != key()


def test_class_floors_keep_unmeasured_large_and_wider_builds_conservative():
    gib = history.GIB_KB
    assert history.compile_floor_kb('native', 'h1_2', 2) == 6*gib
    assert history.compile_floor_kb('wrapper', 'iiwa14', 2) == 8*gib
    assert history.compile_floor_kb('wrapper', 'go2', 2, {'floating_base': True}) == 12*gib
    assert history.compile_floor_kb('wrapper', 'go2', 2,
        {'floating_base': True, 'enable_mujoco_kernels': False}) == 8*gib
    assert history.compile_floor_kb('wrapper', 'h1_2', 2) == 16*gib
    assert history.compile_floor_kb('native', 'iiwa14', 4) == 16*gib
    assert history.compile_floor_kb('unknown', 'iiwa14', 2) == 16*gib
    assert history.shard_floor_kb(['test_armature_default_registration.py']) == 12*gib
    assert history.shard_floor_kb(['test.py[h1_2]']) == 16*gib


def test_completed_worker_persists_identity_peak_before_pool_report(tmp_path, monkeypatch):
    ledger = compile_sched.Ledger(tmp_path / 'rss.json')
    sched = compile_sched.RamScheduler(ledger, max_jobs=1, floor_kb=0,
                                     default_peak_kb=1, margin=1, strict_memory=True)
    job = compile_sched.Job('one', [sys.executable, '-c', 'print("done")'],
        'resource-v1:test', log_path=str(tmp_path/'one.log'), peak_floor_kb=1,
        record_history=True)
    result = sched.run([job])['one']
    assert result.rc == 0 and result.peak_kb > 0
    assert compile_sched.Ledger(ledger.path).data[job.ledger_key]['peak_kb'] == result.peak_kb
    ledger.record(job.ledger_key, 1)
    assert ledger.data[job.ledger_key]['peak_kb'] == result.peak_kb


def test_class_floor_replaces_flat_default_but_not_larger_observation(tmp_path, monkeypatch):
    monkeypatch.setattr(compile_sched, 'mem_available_kb', lambda: 1000)
    ledger = compile_sched.Ledger(tmp_path/'rss.json')
    ledger.record('resource-v1:test', 50)
    sched = compile_sched.RamScheduler(ledger, default_peak_kb=900,
                                      floor_kb=100, margin=1.3, strict_memory=True)
    seen = []
    def launch(job, pred, *args):
        seen.append(pred)
        sched.stop()
        return False
    monkeypatch.setattr(sched, '_launch', launch)
    sched.run([compile_sched.Job('one', [], 'resource-v1:test', peak_floor_kb=30)])
    assert seen == [65]


# --- measured classes (adaptive profile) ------------------------------------
import json

resource_history = history
GIB = history.GIB_KB

def test_class_history_charges_the_measurement_and_bounds_other_widths(tmp_path):
    h = resource_history.ClassHistory(tmp_path / 'class.json')
    floor = 16 * GIB
    assert h.predict_kb('wrapper', 'go2', 'a', 4, floor) == floor   # nothing measured
    h.record('wrapper', 'go2', 'a', 2, 4 * GIB, proc_kb=int(1.7 * GIB), secs=100)
    assert h.predict_kb('wrapper', 'go2', 'a', 2, floor) == 4 * GIB
    assert h.predict_kb('wrapper', 'go2', 'a', 1, floor) == 4 * GIB  # narrower cannot need more
    # Wider and unmeasured: one driver plus four compilers of the largest size.
    assert h.predict_kb('wrapper', 'go2', 'a', 4, floor) == 5 * int(1.7 * GIB)
    h.record('wrapper', 'go2', 'a', 4, 7 * GIB, proc_kb=2 * GIB)
    assert h.predict_kb('wrapper', 'go2', 'a', 4, floor) == 7 * GIB
    # A measurement above the old floor is charged in full; a bound is not.
    h.record('wrapper', 'go2', 'big', 2, 9 * GIB, proc_kb=7 * GIB)
    assert h.predict_kb('wrapper', 'go2', 'big', 4, floor) == floor
    h.record('wrapper', 'go2', 'big', 4, 20 * GIB)
    assert h.predict_kb('wrapper', 'go2', 'big', 4, floor) == 20 * GIB
    # Max over history, persisted; a smaller later sample changes nothing.
    h.record('wrapper', 'go2', 'a', 4, 1 * GIB)
    assert resource_history.ClassHistory(tmp_path / 'class.json').rows == h.rows
    assert h.predict_kb('wrapper', 'go2', 'a', 4, floor) == 7 * GIB
    h.record('native', 'go2', 'small', 1, GIB // 4)
    assert h.predict_kb('native', 'go2', 'small', 1, floor) == GIB   # never under 1 GiB


def test_unseen_class_uses_the_largest_peer_then_the_floor(tmp_path):
    h = resource_history.ClassHistory(tmp_path / 'class.json')
    h.record('wrapper', 'go2', 'a', 4, 7 * GIB)
    h.record('wrapper', 'go2', 'b', 4, 9 * GIB)
    assert h.predict_kb('wrapper', 'go2', 'new', 4, 16 * GIB) == 9 * GIB
    assert h.predict_kb('wrapper', 'h1_2', 'new', 4, 16 * GIB) == 16 * GIB
    assert h.predict_kb('native', 'go2', 'new', 4, 16 * GIB) == 16 * GIB
    # A peer with no bound at this width leaves the floor in place.
    h.record('wrapper', 'go2', 'c', 2, 3 * GIB)   # no process size recorded
    assert h.predict_kb('wrapper', 'go2', 'new', 4, 16 * GIB) == 16 * GIB


def test_only_real_builds_count_as_measurements():
    assert resource_history.log_shows_build('[cuda-equivalence] generating header cache miss key=1')
    assert resource_history.log_shows_build("BUILT go2,floating_base=True abc")
    assert not resource_history.log_shows_build('HIT go2,floating_base=True abc\nWARMED x (1s)')


def test_reserve_and_width_scale_with_the_machine(tmp_path):
    reserve = resource_history.adaptive_reserve_kb
    assert reserve(8 * GIB) == 4 * GIB and reserve(16 * GIB) == 4 * GIB
    assert reserve(62 * GIB) == int(0.15 * 62 * GIB) and reserve(256 * GIB) == int(0.15 * 256 * GIB)
    h = resource_history.ClassHistory(tmp_path / 'class.json')
    h.record('wrapper', 'go2', 'a', 4, 7 * GIB, proc_kb=2 * GIB)
    h.record('wrapper', 'go2', 'a', 2, 4 * GIB, proc_kb=2 * GIB)
    h.record('wrapper', 'go2', 'a', 1, int(2.5 * GIB), proc_kb=2 * GIB)
    plan = lambda budget, cores, **k: resource_history.plan_width(
        budget_kb=budget * GIB, cores=cores, history=h, margin=1.3, **k)
    assert plan(50, 24) == (24, 4, int(7 * GIB * 1.3))        # large box: full width
    assert plan(12, 8) == (8, 4, int(7 * GIB * 1.3))          # 16 GiB box: still fits
    assert plan(6, 6) == (6, 2, int(4 * GIB * 1.3))           # 8 GiB Jetson: two units
    assert plan(4, 4)[1] == 1                                 # smaller still: one unit
    assert plan(50, 24, units=2, slots=3)[:2] == (3, 2)       # stated values are upper bounds
    assert plan(6, 6, units=8)[1] == 2


def test_import_seeds_cold_builds_from_a_finished_run(tmp_path):
    run = tmp_path / 'compile'
    run.mkdir()
    ident = lambda kind, robot, key: dict(key=key, kind=kind, recipe=dict(robot=robot),
                                          environment=dict(GRID_RBD_BUILD_JOBS='2'))
    (run / 'resource-identities.json').write_text(json.dumps(dict(
        warm00_go2=ident('wrapper', 'go2', 'k1'), warm01_iiwa14=ident('wrapper', 'iiwa14', 'k2'),
        prewarm_go2=ident('native', 'go2', 'k3'))))
    (run / 'warm00_go2.log').write_text('BUILT go2 abc\n')
    (run / 'warm00_go2.time').write_text('\tMaximum resident set size (kbytes): 1000\n')
    (run / 'warm01_iiwa14.log').write_text('HIT iiwa14 abc\n')            # cache hit: skipped
    (run / 'prewarm_go2.log').write_text('compiling runner arch=sm_120\n')
    ledger = dict(k1=dict(peak_kb=4000, secs=9.0), k2=dict(peak_kb=10, secs=1.0),
                  k3=dict(peak_kb=300, secs=5.0))
    h = resource_history.ClassHistory(tmp_path / 'class.json')
    assert resource_history.import_run(h, run, ledger) == 2
    name = resource_history.recipe_name(dict(robot='go2'))
    assert h.rows[f'wrapper|go2|{name}'] == {'2': dict(tree_kb=4000, proc_kb=1000, secs=9.0)}
    assert list(h.rows[f'native|go2|{name}']) == ['1']
    assert not any(key.startswith('wrapper|iiwa14') for key in h.rows)


def test_two_threads_recording_at_once_both_persist(tmp_path, monkeypatch):
    # The pool thread and the shard loop share one object. Both writes used to
    # go through one temporary name, and the second rename failed.
    import threading
    path = tmp_path / 'class.json'
    h = resource_history.ClassHistory(path)
    names = [f'n{i}' for i in range(16)]
    start, errors = threading.Barrier(len(names)), []

    def record(name):
        start.wait(5)
        try:
            h.record('wrapper', 'go2', name, 2, 3 * GIB, proc_kb=GIB)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=record, args=(n,)) for n in names]
    [t.start() for t in threads], [t.join() for t in threads]
    assert not errors
    stored = resource_history.ClassHistory(path).rows
    assert set(stored) == {f'wrapper|go2|{n}' for n in names} and stored == h.rows
    assert not list(tmp_path.glob('*.tmp'))


def test_a_stale_writer_does_not_lower_a_stored_peak(tmp_path):
    path = tmp_path / 'class.json'
    a, b = resource_history.ClassHistory(path), resource_history.ClassHistory(path)
    a.record('wrapper', 'go2', 'r', 2, 9 * GIB, proc_kb=4 * GIB)
    b.record('wrapper', 'go2', 'r', 2, 2 * GIB, proc_kb=GIB)     # read before a wrote
    b.record('wrapper', 'go2', 'other', 2, 3 * GIB)
    for h in (b, resource_history.ClassHistory(path)):
        assert h.predict_kb('wrapper', 'go2', 'r', 2, 16 * GIB) == 9 * GIB
        assert h.predict_kb('wrapper', 'go2', 'other', 2, 16 * GIB) == 3 * GIB


def test_a_history_that_cannot_be_stored_does_not_fail_the_job(tmp_path, capsys):
    blocker = tmp_path / 'file'
    blocker.write_text('')
    h = resource_history.ClassHistory(blocker / 'class.json')    # parent is a file
    h.record('wrapper', 'go2', 'r', 2, 3 * GIB)
    assert h.predict_kb('wrapper', 'go2', 'r', 2, 16 * GIB) == 3 * GIB
    assert 'not stored' in capsys.readouterr().err


@pytest.mark.parametrize('classes', [
    ['bad'],
    {'wrapper|go2|r': ['bad']},
    {'wrapper|go2|r': {'2': {'tree_kb': 'bad'}}},
    {'wrapper|go2|r': {'2': {'tree_kb': -GIB, 'proc_kb': 0, 'secs': 0}}},
    {'wrapper|go2|r': {'two': {'tree_kb': GIB}}},
    {'wrapper|go2|r': {'2': {'tree_kb': 4 * GIB}, '4': {'tree_kb': True}}},
    {'wrapper|go2|r': {'2': {'tree_kb': float('inf')}}},
    {'wrapper|go2|r': {'2': {'tree_kb': 4 * GIB, 'proc_kb': float('inf')}}},
    {'wrapper|go2|r': {'2': {'tree_kb': 4 * GIB, 'secs': float('nan')}}},
    {'wrapper|go2|r': {'\u00b2': {'tree_kb': 4 * GIB}}},
    {'wrapper|go2|r': {'4': {'tree_kb': 20 * GIB}, '04': {'tree_kb': 30 * GIB}}},
    {'wrapper|go2|r': {'0': {'tree_kb': 4 * GIB}}},
    {'wrapper|go2|r': {'2': {'tree_kb': 10 ** 400}}},
    {'wrapper|go2|r': {'2': {'tree_kb': 4 * GIB, 'secs': 10 ** 400}}},
    {'wrapper|go2|r': {'2': {'tree_kb': 4 * GIB}, '-4': {'tree_kb': 4 * GIB}}},
])
def test_a_malformed_class_is_charged_as_unmeasured(tmp_path, capsys, classes):
    path = tmp_path / 'class.json'
    # 1e309 is how a file spells a number that decodes to infinity.
    path.write_text(json.dumps(dict(schema=1, classes=classes)).replace('Infinity', '1e309'))
    h = resource_history.ClassHistory(path)
    assert h.rows == {} and capsys.readouterr().err
    assert h.predict_kb('wrapper', 'go2', 'r', 2, 16 * GIB) == 16 * GIB
    assert h.largest_kb(2, 16 * GIB) == 16 * GIB


def test_a_malformed_class_does_not_discard_the_others(tmp_path):
    path = tmp_path / 'class.json'
    path.write_text(json.dumps(dict(schema=1, classes={
        'wrapper|go2|bad': {'2': {'tree_kb': -1}},
        'wrapper|iiwa14|ok': {'2': {'tree_kb': 3 * GIB, 'proc_kb': GIB, 'secs': 5}}})))
    h = resource_history.ClassHistory(path)
    assert list(h.rows) == ['wrapper|iiwa14|ok']
    assert h.predict_kb('wrapper', 'iiwa14', 'ok', 2, 16 * GIB) == 3 * GIB


def test_the_planned_width_is_one_admission_accepts(tmp_path):
    # A measurement above the unmeasured-class floor is charged in full by
    # admission, so the plan has to count it in full as well.
    from resource_admission import MemoryAdmission
    h = resource_history.ClassHistory(tmp_path / 'class.json')
    h.record('wrapper', 'go2', 'heavy', 4, 20 * GIB, proc_kb=4 * GIB)
    h.record('wrapper', 'go2', 'heavy', 2, 10 * GIB, proc_kb=4 * GIB)
    assert h.largest_kb(4, 16 * GIB) == h.predict_kb('wrapper', 'go2', 'heavy', 4, 16 * GIB) == 20 * GIB
    _slots, units, largest = resource_history.plan_width(
        budget_kb=22 * GIB, cores=8, history=h, margin=1.3)
    assert (units, largest) == (2, int(10 * GIB * 1.3))
    admission = MemoryAdmission(floor_kb=4 * GIB, available=lambda: 30 * GIB, rss=lambda pid: 0,
                                cap_kb=22 * GIB, latch=False)
    assert admission.reserve('job', int(h.predict_kb('wrapper', 'go2', 'heavy', units, 16 * GIB) * 1.3))
    # The smallest charge is the same in both: never under 1 GiB.
    h2 = resource_history.ClassHistory(tmp_path / 'small.json')
    h2.record('native', 'go2', 'tiny', 1, GIB // 4)
    assert h2.largest_kb(1, 16 * GIB) == h2.predict_kb('native', 'go2', 'tiny', 1, 16 * GIB) == GIB
    # A class with no bound at a width is charged the floor there, in both.
    h2.record('wrapper', 'go2', 'narrow', 2, 3 * GIB)            # no process size
    assert h2.largest_kb(4, 16 * GIB) == h2.predict_kb('wrapper', 'go2', 'narrow', 4, 16 * GIB) == 16 * GIB

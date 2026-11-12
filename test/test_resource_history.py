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

import copy
import json
from types import SimpleNamespace

import pytest

from test.receipt_integrity import verify_scope, verify_header_keys
from test.run_split_suite import _locked_compile_dependency


@pytest.mark.parametrize('returncode,changed', [(0, False), (1, True), (128, True)])
def test_oracle_gitlink_guard_fails_closed(monkeypatch, returncode, changed):
    from test import codegen_neutrality as neutrality
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=returncode)
    monkeypatch.setattr(neutrality.subprocess, 'run', run)
    assert neutrality.reference_inputs_changed('ancestor') is changed
    assert calls == [['git', 'diff', '--quiet', 'ancestor', '--', 'external/RBDReference']]


def test_glass_pin_is_a_codegen_input():
    from test import codegen_neutrality as neutrality
    assert 'external/GLASS' in neutrality.CODEGEN_INPUT_PATHS


def test_scope_uses_ids_not_counts():
    scope = {'node_ids': ['a', 'b']}
    verify_scope({'tests': [{'node_id': n} for n in ['a', 'b', 'c']]}, scope)
    with pytest.raises(ValueError, match='omits'):
        verify_scope({'tests': [{'node_id': n} for n in ['a', 'c']]}, scope)
    with pytest.raises(ValueError, match='Duplicate'):
        verify_scope({'tests': [{'node_id': 'a'}, {'node_id': 'a'}]}, scope)


def test_fresh_headers_may_change_count_but_carried_may_not(tmp_path):
    before = {'shards': {'fresh': [{'content_sha256': 'old'}],
                         'carry': [{'content_sha256': 'carry'}]}}
    after = copy.deepcopy(before)
    after['shards']['fresh'] = [{'content_sha256': 'new'}, {'content_sha256': 'new2'}]
    (tmp_path / 'fresh.header_keys.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in after['shards']['fresh']))
    receipt = {'shards': [{'name': 'fresh'}, {'name': 'carry', 'carried': True}]}
    verify_header_keys(before, after, receipt, tmp_path)
    after['shards']['carry'][0]['content_sha256'] = 'changed'
    with pytest.raises(ValueError, match='Carried'):
        verify_header_keys(before, after, receipt, tmp_path)


def test_fresh_missing_sidecar_cannot_reuse_old_rows(tmp_path):
    keys = {'shards': {'fresh': [{'content_sha256': 'old'}]}}
    with pytest.raises(ValueError, match='Fresh'):
        verify_header_keys(keys, keys, {'shards': [{'name': 'fresh'}]}, tmp_path)


@pytest.mark.parametrize('owner_groups,compiler,waiter,expected', [
    ([10], 'cicc', 20, True), ([99], 'cicc', 20, False),
    ([10], 'python', 20, False), ([10], 'cicc', 99, False),
])
def test_stall_exemption_requires_exact_waited_lock(owner_groups, compiler, waiter, expected):
    processes = f'11 10 python\n12 10 {compiler}\n21 {waiter} python\n'
    locks = '7: FLOCK ADVISORY WRITE 11 08:01:555 0 EOF\n7: -> FLOCK ADVISORY WRITE 21 08:01:555 0 EOF\n'
    assert _locked_compile_dependency(20, owner_groups, processes, locks) is expected


def test_shard_resumes_stall_countdown_after_dependency_compile(tmp_path, monkeypatch):
    from test import run_split_suite as runner
    clock = [0.0]
    dependency = [True]
    killed = []
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    monkeypatch.setenv('GRID_SPLIT_STALL_SECS', '10')
    monkeypatch.setattr(runner, '_compiler_child_alive', lambda pid: False)
    monkeypatch.setattr(runner, '_waiting_for_compile', lambda *args: dependency[0])
    monkeypatch.setattr(runner, '_kill_group', lambda proc: killed.append(proc.pid))
    log_path = tmp_path / 'log'
    log_path.touch()
    shard = runner._ShardRun(None, SimpleNamespace(pid=42, poll=lambda: None),
                             None, log_path, tmp_path / 'junit.xml')
    assert shard.poll() is None
    clock[0] = 100
    assert shard.poll() is None  # quiet log but an exact blocking compile dependency
    dependency[0] = False
    clock[0] = 109
    assert shard.poll() is None
    clock[0] = 111
    assert shard.poll() == 'STALL'
    assert killed == [42]


def test_aggregate_does_not_replace_fresh_evidence_with_old(tmp_path, monkeypatch):
    from test import run_split_suite as runner
    keys = tmp_path / 'keys.json'
    before = {'shards': {'fresh': [{'content_sha256': 'old'}]}}
    keys.write_text(json.dumps(before))
    monkeypatch.setattr(runner, 'HEADER_KEYS_PATH', keys)
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps({'shards': [{'name': 'fresh'}]}))
    with pytest.raises(ValueError, match='lost its header'):
        runner.aggregate_header_keys(tmp_path, [{'shard': 'fresh'}], receipt)
    assert json.loads(keys.read_text()) == before
    (tmp_path / 'fresh.header_keys.jsonl').write_text('{"content_sha256": "new"}\n')
    runner.aggregate_header_keys(tmp_path, [{'shard': 'fresh'}], receipt)
    assert json.loads(keys.read_text())['shards']['fresh'] == [{'content_sha256': 'new'}]


def test_validation_environment_cannot_inherit_narrow_test_scope(monkeypatch):
    from test.run_validation import validation_env
    monkeypatch.setenv('PYTEST_ADDOPTS', '-k tiny')
    monkeypatch.setenv('GRID_CUDA_TARGET_SHARED_MEM_BYTES', '123')
    monkeypatch.setenv('SPLIT_DOMAINS', 'wrappers')
    env = validation_env()
    assert 'PYTEST_ADDOPTS' not in env
    assert 'GRID_CUDA_TARGET_SHARED_MEM_BYTES' not in env
    assert 'SPLIT_DOMAINS' not in env
    assert env['OMP_NUM_THREADS'] == '1'


def test_validation_resume_uses_saved_plan_not_replanning(tmp_path):
    from test.run_validation import validation_command
    assert '--refresh-from' in validation_command(tmp_path)
    resumed = validation_command(tmp_path, resume=True)
    assert '--resume' in resumed
    assert '--refresh-from' not in resumed

import copy
import json
from types import SimpleNamespace

import pytest

from test.receipt_integrity import write_manifest, verify_header_keys
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


def test_manifest_is_exact_sorted_and_keeps_its_header(tmp_path):
    path = tmp_path / 'gpu-proof-tests.txt'
    path.write_text('# reviewed scope\nstale::old\n')
    ids = write_manifest({'tests': [{'node_id': n} for n in ['b', 'a']]}, path)
    assert ids == {'a', 'b'}
    assert path.read_text() == '# reviewed scope\na\nb\n'   # removed ids drop out (exact set)
    with pytest.raises(ValueError, match='Duplicate'):
        write_manifest({'tests': [{'node_id': 'a'}, {'node_id': 'a'}]}, path)


def test_committed_manifest_matches_the_committed_receipt():
    """The policy gate is EXACT: the reviewed manifest and the committed
    receipt's collection must agree in both directions."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / 'gpu-proof.json').read_text())
    manifest = {l.strip() for l in (root / 'test/gpu-proof-tests.txt').read_text().splitlines()
                if l.strip() and not l.startswith('#')}
    assert set(receipt['session']['node_ids']) == manifest


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


# ─── receipts by architecture (test/gpu-proof-receipts.yaml) ─────────────────

from pathlib import Path as _Path

from test import receipt_integrity as ri

ROOT = _Path(__file__).resolve().parents[1]


def synthetic_receipt(cap='8.7', *, name='NVIDIA Jetson Orin', memory='65536 MiB',
                      driver='540.4.0', python='3.10.12', commit='abcdef0123456789',
                      ended='2026-10-20T10:00:00Z', gpu=True):
    env = {'python_version': python,
           'platform': 'Linux-5.15.148-tegra-aarch64-with-glibc2.35',
           'gpu_info': {'devices': [{'compute_capability': cap, 'name': name,
                                     'memory': memory, 'driver_version': driver}],
                        'source': 'nvidia-smi'} if gpu else None}
    return {'environment': env,
            'repo': {'commit_sha': commit, 'branch': 'main'},
            'session': {'started_at': '2026-10-20T01:00:00Z', 'ended_at': ended},
            'shards': [{'name': 'a'}, {'name': 'b', 'carried': True}],
            'tests': [{'node_id': 'x::a', 'outcome': 'passed'},
                      {'node_id': 'x::b', 'outcome': 'skipped'}]}


def write_table(path, primary, secondary=None):
    import yaml
    path.write_text(yaml.safe_dump({'primary': primary, 'secondary': secondary or []}))
    return path


def test_committed_table_primary_is_gpu_proof_json_and_matches_the_committed_receipt():
    rows = ri.load_receipt_table()
    primary = rows[0]
    assert primary.primary and primary.receipt == 'gpu-proof.json' and primary.arch == 120
    assert primary.expected_skips == 'test/gpu-proof-expected-skips.txt'
    assert primary.header_keys == 'test/gpu-proof-header-keys.json'
    assert rows[1:], 'the Jetson wave needs at least one secondary row'
    for row in rows[1:]:
        assert not row.primary and row.receipt.startswith('receipts/')
        assert (ROOT / row.expected_skips).is_file()
    assert any(row.arch == 87 for row in rows[1:])   # Jetson Orin
    receipt = json.loads((ROOT / 'gpu-proof.json').read_text())
    assert ri.receipt_arch(receipt) == primary.arch
    assert ri.require_primary_receipt(receipt, rows) is primary


def test_rows_are_selected_by_detected_arch(tmp_path):
    table = ri.load_receipt_table(write_table(
        tmp_path / 't.yaml', {'arch': 120, 'label': 'RTX 5090', 'receipt': 'gpu-proof.json'},
        [{'arch': 87, 'label': 'Jetson Orin', 'receipt': 'receipts/gpu-proof-sm87.json',
          'expected_skips': 'receipts/sm87-skips.txt'}]))
    primary = ri.receipt_row_for_arch(120, table)
    assert primary.primary and primary.receipt == 'gpu-proof.json'
    orin = ri.receipt_row_for_arch(87, table)
    assert not orin.primary
    assert orin.receipt == 'receipts/gpu-proof-sm87.json'
    assert orin.expected_skips == 'receipts/sm87-skips.txt'
    assert orin.header_keys == 'receipts/gpu-proof-sm87-header-keys.json'
    assert orin.sm == 'sm_87'
    with pytest.raises(ValueError, match='add a row.*gpu-proof-receipts.yaml'):
        ri.receipt_row_for_arch(75, table)
    with pytest.raises(ValueError, match='no GPU detected'):
        ri.receipt_row_for_arch(0, table)


@pytest.mark.parametrize('secondary,match', [
    ([{'arch': 87, 'label': 'Orin', 'receipt': 'gpu-proof-sm87.json'}], 'under receipts/'),
    ([{'arch': 87, 'label': 'Orin', 'receipt': 'gpu-proof.json'}], 'under receipts/'),
    ([{'arch': 120, 'label': 'Orin', 'receipt': 'receipts/gpu-proof-sm120.json'}], 'duplicate arch'),
    ([{'arch': 87, 'label': 'Orin', 'receipt': 'receipts/a.json'},
      {'arch': 72, 'label': 'Xavier', 'receipt': 'receipts/a.json'}], 'duplicate receipt'),
    ([{'arch': '87', 'label': 'Orin', 'receipt': 'receipts/a.json'}], 'positive integer'),
    ([{'arch': 87, 'receipt': 'receipts/a.json'}], 'lacks'),
    ([{'arch': 87, 'label': 'Orin', 'receipt': 'receipts/a.json', 'gpu': 'x'}], 'unknown key'),
])
def test_table_rejects_rows_that_could_alias_or_shadow_the_primary(tmp_path, secondary, match):
    path = write_table(tmp_path / 't.yaml',
                       {'arch': 120, 'label': 'RTX 5090', 'receipt': 'gpu-proof.json'}, secondary)
    with pytest.raises(ValueError, match=match):
        ri.load_receipt_table(path)


def test_table_requires_a_primary_row(tmp_path):
    path = tmp_path / 't.yaml'
    path.write_text('secondary: []\n')
    with pytest.raises(ValueError, match='primary'):
        ri.load_receipt_table(path)


def test_receipt_arch_reads_the_recorded_compute_capability():
    assert ri.receipt_arch(synthetic_receipt('12.0')) == 120
    assert ri.receipt_arch(synthetic_receipt('8.7')) == 87
    assert ri.receipt_arch(synthetic_receipt(gpu=False)) is None
    assert ri.receipt_arch({}) is None


def test_manifest_is_written_from_the_primary_receipt_only(tmp_path):
    manifest = tmp_path / 'gpu-proof-tests.txt'
    manifest.write_text('# reviewed\nold\n')
    secondary = tmp_path / 'sm87.json'
    secondary.write_text(json.dumps(synthetic_receipt('8.7')))
    with pytest.raises(ValueError, match='secondary receipt.*sm_87.*Jetson Orin.*primary receipt'):
        ri.main(['--receipt', str(secondary), '--write-manifest', str(manifest)])
    assert manifest.read_text() == '# reviewed\nold\n'
    no_gpu = tmp_path / 'nogpu.json'
    no_gpu.write_text(json.dumps(synthetic_receipt(gpu=False)))
    with pytest.raises(ValueError, match='records no GPU'):
        ri.main(['--receipt', str(no_gpu), '--write-manifest', str(manifest)])
    assert manifest.read_text() == '# reviewed\nold\n'
    primary = tmp_path / 'primary.json'
    primary.write_text(json.dumps(synthetic_receipt('12.0', name='NVIDIA GeForce RTX 5090')))
    assert ri.main(['--receipt', str(primary), '--write-manifest', str(manifest)]) == 0
    assert manifest.read_text() == '# reviewed\nx::a\nx::b\n'


def test_main_takes_exactly_one_action(tmp_path):
    with pytest.raises(SystemExit):
        ri.main([])
    with pytest.raises(SystemExit):
        ri.main(['--verify-all', '--table'])
    with pytest.raises(SystemExit):
        ri.main(['--write-manifest', str(tmp_path / 'm.txt')])   # no --receipt


def compatibility_columns():
    """Header cells of the "What we ran" list-table on the compatibility page."""
    text = (ROOT / 'docs/source/user_guide/getting_started/compatibility.rst').read_text()
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('   * - GPU architecture'))
    cells = [lines[start][len('   * - '):]]
    for line in lines[start + 1:]:
        if not line.startswith('     - '):
            break
        cells.append(line[len('     - '):])
    return cells


def test_reporter_rows_follow_the_compatibility_table_columns():
    columns = compatibility_columns()
    assert columns[:2] == ['GPU architecture', 'GPU'] and columns[-1] == 'Date'
    rows = ri.rst_table_rows([synthetic_receipt('8.7')])
    assert rows.startswith('   * - ') and rows.endswith('\n')
    cells = rows.rstrip('\n')[len('   * - '):].split('\n     - ')
    assert len(cells) == len(columns)
    by_name = dict(zip(columns, cells))
    assert by_name['GPU architecture'] == 'sm_87 (compute capability 8.7)'
    assert by_name['GPU'] == 'NVIDIA Jetson Orin, 64 GB'
    assert by_name['Driver'] == '540.4.0'
    assert by_name['Python'] == 'CPython 3.10.12'
    assert by_name['OS'] == 'Linux-5.15.148-tegra-aarch64-with-glibc2.35'
    assert by_name['CUDA toolkit (nvcc)'] == ri.NOT_RECORDED
    assert by_name['torch wheel'] == ri.NOT_RECORDED
    assert by_name['Date'] == 'receipt 2026-10-20'
    ran = by_name['What ran']
    assert '2 tests: 1 passed, 1 skipped' in ran and '1 of 2 shards carried' in ran
    assert 'commit abcdef0' in ran
    assert ri.rst_table_rows([]) == ''


def test_reporter_cli_prints_every_receipt_on_disk(tmp_path, capsys, monkeypatch):
    receipt = tmp_path / 'r.json'
    receipt.write_text(json.dumps(synthetic_receipt('7.2', name='NVIDIA Xavier', driver=None)))
    assert ri.main(['--table', '--receipt', str(receipt)]) == 0
    out = capsys.readouterr().out
    assert 'sm_72 (compute capability 7.2)' in out and f'- {ri.NOT_RECORDED}\n' in out
    assert out.count('   * - ') == 1


def test_verify_all_runs_each_receipt_with_its_own_skips_and_reports_missing_ones(tmp_path, capsys):
    root = tmp_path
    (root / 'receipts').mkdir()
    (root / 'gpu-proof.json').write_text(json.dumps(synthetic_receipt(
        '12.0', name='NVIDIA GeForce RTX 5090', memory='32607 MiB', commit='1234567abc')))
    (root / 'receipts/gpu-proof-sm87.json').write_text(json.dumps(synthetic_receipt('8.7')))
    table = ri.load_receipt_table(write_table(
        root / 't.yaml', {'arch': 120, 'label': 'RTX 5090', 'receipt': 'gpu-proof.json'},
        [{'arch': 87, 'label': 'Jetson Orin', 'receipt': 'receipts/gpu-proof-sm87.json',
          'expected_skips': 'receipts/sm87-skips.txt'},
         {'arch': 72, 'label': 'Jetson Xavier', 'receipt': 'receipts/gpu-proof-sm72.json'}]))
    calls = []
    from types import SimpleNamespace
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    now = ri.datetime(2026, 10, 25, tzinfo=ri.timezone.utc)
    assert ri.verify_all(table, root=root, run=run, now=now) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert 'NVIDIA GeForce RTX 5090, 32 GB; commit 1234567; 4 d old; 1 passed, 1 skipped; ok' in lines[0]
    assert lines[0].startswith('sm_120') and 'gpu-proof.json' in lines[0]
    assert lines[1].startswith('sm_87') and lines[1].endswith('ok')
    assert lines[2].startswith('sm_72') and 'not recorded (secondary receipt missing)' in lines[2]
    assert len(calls) == 2
    assert calls[0][calls[0].index('--expected-skips') + 1] == str(root / 'test/gpu-proof-expected-skips.txt')
    assert calls[1][calls[1].index('--expected-skips') + 1] == str(root / 'receipts/sm87-skips.txt')
    assert all(c[c.index('--policy') + 1] == str(root / 'test/gpu-proof-policy.yaml') for c in calls)
    # a failed verification is the exit code, and the verifier's output is shown
    def failing(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout='[gpu-proof] FAIL: stale\n', stderr='')
    assert ri.verify_all(table, root=root, run=failing, now=now) == 1
    out = capsys.readouterr().out
    assert '[gpu-proof] FAIL: stale' in out and 'FAIL (rc=1)' in out
    # a missing primary receipt is an error, a missing secondary is not
    (root / 'gpu-proof.json').unlink()
    assert ri.verify_all(table, root=root, run=run, now=now) == 1
    assert 'primary receipt missing' in capsys.readouterr().out

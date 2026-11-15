"""No CUDA or signing required: adversarial scope and author-bound verifier tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import subprocess
from test import provisional_receipt as p


def receipt(ids=('smoke', 'extra')):
    return dict(repo={'commit_sha': 'a' * 40}, session={'node_ids': list(ids)},
                tests=[{'node_id': n, 'outcome': 'passed'} for n in ids],
                environment={'gpu_info': {'devices': [{'compute_capability': '8.9'}]}},
                shards=[{'name': 'unit'}])


def test_smoke_is_real_full_scope_and_optional_units_do_not_overlap():
    smoke = p.manifest(p.ROOT / 'test/gpu-proof-provisional-smoke.txt')
    full = p.manifest(p.ROOT / 'test/gpu-proof-tests.txt')
    assert smoke <= full
    extra = [n for unit in p.optional_units(full, smoke) for n in unit]
    assert not set(extra) & smoke
    assert len(extra) == len(set(extra)) and set(extra) <= full
    assert all(any(f'[{r}-' in n for r in ('iiwa14', 'go2', 'fr3')) for n in extra)


def test_unknown_arch_valid_scope_is_not_full_receipt():
    p.check_scope(receipt(), {'smoke'}, {'smoke', 'extra', 'unused'})


def test_small_native_class_allowance_and_unknown_fallback():
    from test.resource_admission import MemoryAdmission
    gib = 1024**2
    smoke = p.manifest(p.ROOT / 'test/gpu-proof-provisional-smoke.txt')
    full = p.manifest(p.ROOT / 'test/gpu-proof-tests.txt')
    optional = p.optional_units(full, smoke)
    allowance = p.unit_peak_kb(smoke, smoke, optional)
    assert allowance == int(8 * 1.3 * gib)
    for group in optional:
        assert p.unit_peak_kb(group, smoke, optional) == allowance
    for group in (set(), smoke | {'unreviewed'}, {'test/new.py::unknown'}):
        assert p.unit_peak_kb(group, group, []) == int(16 * 1.3 * gib)
    # Admission depends on available memory, not installed capacity. The
    # current contributor profile does NOT yet claim the proposed 8 GiB lane.
    for available, expected in ((28, True), (24, False), (8, False)):
        admission = MemoryAdmission(floor_kb=14*gib,
                                    available=lambda: available*gib, rss=lambda pid: 0)
        assert admission.reserve('smoke', allowance) is expected


@pytest.mark.parametrize('kind', ['missing', 'extra', 'duplicate', 'unfinished', 'failed',
                                  'skipped', 'carried', 'no_gpu'])
def test_scope_fails_closed(kind):
    r = receipt()
    if kind == 'missing': r = receipt(('extra',))
    if kind == 'extra': r = receipt(('smoke', 'outside'))
    if kind == 'duplicate': r['tests'].append(r['tests'][0])
    if kind == 'unfinished': r['session']['node_ids'].append('not-run')
    if kind in ('failed', 'skipped'): r['tests'][1]['outcome'] = kind
    if kind == 'carried': r['shards'][0]['carried'] = True
    if kind == 'no_gpu': r['environment'] = {}
    with pytest.raises(ValueError): p.check_scope(r, {'smoke'}, {'smoke', 'extra'})


def test_verification_uses_trusted_author_and_policy(tmp_path, monkeypatch):
    candidate = tmp_path / 'candidate'; candidate.mkdir()
    (candidate / 'receipts').mkdir()
    path = candidate / 'receipts/provisional.json'
    # Use trusted real IDs; fake subprocess only, never bypass in production.
    r = receipt(sorted(p.manifest(p.ROOT / 'test/gpu-proof-provisional-smoke.txt')))
    path.write_text(json.dumps(r))
    commands = []
    def output(cmd, **kwargs):
        if cmd[1] == 'rev-parse': return 'b' * 40 + '\n'
        return 'receipts/provisional.json\n'
    def run(cmd, **kwargs):
        commands.append(cmd)
        if 'verify' in cmd:
            rules = json.loads(Path(cmd[cmd.index('--policy') + 1]).read_text())
            assert rules['allowed_signers'] == ['contributor']
            assert rules['signer_mode'] == 'restricted'
            assert rules['allow_carried'] is False
            assert 'grid_codegen' in rules['required_fingerprint_paths']
            assert cmd[cmd.index('--github-user') + 1] == 'contributor'
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(p.subprocess, 'check_output', output)
    monkeypatch.setattr(p.subprocess, 'run', run)
    p.verify(path, root=candidate, author='contributor', head='b' * 40)
    assert any('verify' in cmd for cmd in commands)
    monkeypatch.setattr(p.subprocess, 'check_output', lambda cmd, **kw:
                        'b' * 40 if cmd[1] == 'rev-parse' else 'grid_codegen/changed.py')
    with pytest.raises(ValueError, match='source changed'):
        p.verify(path, root=candidate, author='contributor', head='b' * 40)


def test_workflow_executes_only_base_tools():
    source = (p.ROOT / '.github/workflows/contributor-evidence.yml').read_text()
    assert 'pull_request_target:' in source
    assert 'python -I trusted/test/' in source
    assert 'pip install -e' not in source
    assert 'run: python candidate' not in source
    assert 'persist-credentials: false' in source
    assert 'contents: read' in source


@pytest.mark.parametrize('mode', ['budget', 'failed_optional', 'pressure', 'no_memory'])
def test_record_budget_and_failures_never_hide_outcomes(tmp_path, monkeypatch, mode):
    from test import resource_admission, run_validation
    root = tmp_path / 'candidate'; root.mkdir()
    (root / 'test').mkdir()
    (root / 'test/gpu-proof-provisional-smoke.txt').write_text('smoke\n')
    (root / 'test/gpu-proof-tests.txt').write_text('extra\nsmoke\n')
    (root / 'gpu-proof.json').write_text(json.dumps({'tests': []}))
    monkeypatch.setattr(p, 'ROOT', root)
    monkeypatch.setattr(p, 'optional_units', lambda *args: [['extra']])
    monkeypatch.setattr(run_validation, 'validation_env', lambda: {})
    monkeypatch.setattr(p.subprocess, 'check_output', lambda cmd, **kw:
                        '' if cmd[1] == 'status' else 'a' * 40)
    times = iter([0, 1900 if mode == 'budget' else 0])
    monkeypatch.setattr(p.time, 'monotonic', lambda: next(times))
    class Admission:
        def __init__(self, **kw): self.blocked = None
        def reserve(self, *args): return mode != 'no_memory'
        def attach(self, *args): pass
        def pressure(self):
            if mode == 'pressure': self.blocked = 'reserve exhausted'
        def release(self, *args): pass
    monkeypatch.setattr(resource_admission, 'MemoryAdmission', Admission)
    launched, merges = [], []
    class Process:
        pid = 123
        def __init__(self, command, **kw):
            assert kw['start_new_session'] is True
            node = command[-1]; launched.append(node)
            self.returncode = 1 if node == 'extra' and mode == 'failed_optional' else 0
            self.polls = 0
            path = Path(next(x.split('=', 1)[1] for x in command if x.startswith('--gpu-proof-out=')))
            data = receipt((node,))
            if self.returncode: data['tests'][0]['outcome'] = 'failed'
            path.write_text(json.dumps(data))
        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else self.returncode
        def wait(self, **kw): return self.returncode
    monkeypatch.setattr(p.subprocess, 'Popen', Process)
    def merge(command, **kw):
        merges.append(command)
        dest = Path(command[command.index('--out') + 1])
        dest.write_text(json.dumps(receipt(tuple(launched))))
    monkeypatch.setattr(p.subprocess, 'run', merge)
    out = tmp_path / 'evidence'
    if mode == 'budget':
        p.record(out, 30, 'contributor')
        assert launched == ['smoke'] and len(merges) == 1
        assert (out / 'provisional.json').exists()
    else:
        with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
            p.record(out, 30, 'contributor')
        assert not merges and not (out / 'provisional.json').exists()
        if mode == 'failed_optional': assert launched == ['smoke', 'extra']
        if mode == 'no_memory': assert not launched
    assert not (root / 'receipts').exists()

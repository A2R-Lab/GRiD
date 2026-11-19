"""No CUDA or signing required: adversarial scope and author-bound verifier tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import subprocess
import shutil
import sys
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


def test_minimal_optional_grouping_matches_recording_driver():
    from test.run_split_suite import _atom_key
    smoke = p.manifest(p.ROOT / 'test/gpu-proof-provisional-smoke.txt')
    full = p.manifest(p.ROOT / 'test/gpu-proof-tests.txt')
    expected = {}
    for node in sorted(full - smoke):
        if '/test_cuda_executable_equivalence.py::' in node:
            key = _atom_key(node)
            if len(key) > 1 and key[1] in {'iiwa14', 'go2', 'fr3'}:
                expected.setdefault(key, []).append(node)
    assert p.optional_units(full, smoke) == list(expected.values())


@pytest.mark.parametrize('case', ['dispatch', 'full', 'symlink', 'dangling'])
def test_isolated_ci_without_submodules(tmp_path, case):
    """Real isolated interpreter + git; signature service is a stated test double.

    -S excludes editable-install hooks that could otherwise hide missing CI
    dependencies. Load only installed PyYAML; no robotics package is needed.
    This tests trusted dispatch, NOT a genuine signed receipt acceptance.
    """
    import yaml
    trusted = tmp_path / 'trusted'
    (trusted / 'test').mkdir(parents=True)
    for name in ('provisional_receipt.py', 'receipt_integrity.py',
                 'gpu-proof-provisional-smoke.txt', 'gpu-proof-tests.txt',
                 'gpu-proof-policy-provisional.yaml', 'gpu-proof-policy-release.yaml',
                 'gpu-proof-provisional-skips.txt'):
        shutil.copy(p.ROOT / 'test' / name, trusted / 'test' / name)
    poison = "raise RuntimeError('PLANTED_CODE_EXECUTED')\n"
    # Package initialization and the full recording driver must never run.
    for name in ('__init__.py', 'run_split_suite.py'):
        (trusted / 'test' / name).write_text(poison)
    subprocess.run(['git', 'init', '-q', str(trusted)], check=True)
    subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-qm', 'trusted', '--allow-empty'], cwd=trusted, check=True)
    candidate = tmp_path / 'candidate'
    (candidate / 'receipts').mkdir(parents=True)
    for name in ('yaml.py', 'json.py', 'sitecustomize.py', 'receipt_integrity.py'):
        (candidate / name).write_text(poison)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=candidate, text=True).strip()
    git('init', '-q')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-qm', 'fixture', '--allow-empty')
    head = git('rev-parse', 'HEAD')
    smoke = p.manifest(trusted / 'test/gpu-proof-provisional-smoke.txt')
    ids = p.manifest(trusted / 'test/gpu-proof-tests.txt') if case == 'full' else smoke
    data = receipt(sorted(ids)); data['repo']['commit_sha'] = head
    path = candidate / 'receipts/provisional.json'
    secret = tmp_path / 'outside.txt'
    secret.write_text('PRIVATE_SENTINEL_DO_NOT_READ')
    if case in ('symlink', 'dangling'):
        path.symlink_to(secret if case == 'symlink' else tmp_path / 'missing')
    else:
        path.write_text(json.dumps(data))
    git('add', 'receipts/provisional.json')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
        'commit', '-qm', 'evidence')
    head = git('rev-parse', 'HEAD')
    bootstrap = '''
import sys, runpy
sys.path.insert(0, sys.argv[1])
import yaml
sys.path.pop(0)
module = runpy.run_path(sys.argv[2], run_name='trusted_verifier')
verify = module['verify']
def signature_double(command, **kwargs):
    if 'verify' in command:
        print('TRUSTED_SIGNATURE_DISPATCH')
    else:
        return original_run(command, **kwargs)
original_run = verify.__globals__['subprocess'].run
verify.__globals__['subprocess'].run = signature_double
verify.__globals__['_gpu_proof_bin'] = lambda: 'trusted-signature-test-double'
from pathlib import Path
verify(Path(sys.argv[3]), root=Path(sys.argv[4]), trusted=Path(sys.argv[2]).parents[1], author='contributor', head=sys.argv[5])
'''
    result = subprocess.run([sys.executable, '-I', '-S', '-c', bootstrap,
                             str(Path(yaml.__file__).parent.parent),
                             str(trusted / 'test/provisional_receipt.py'),
                             str(path), str(candidate), head], cwd=candidate,
                            capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert 'ModuleNotFoundError' not in output
    assert 'PLANTED_CODE_EXECUTED' not in output
    assert 'PRIVATE_SENTINEL_DO_NOT_READ' not in output
    if case == 'dispatch':
        assert result.returncode == 0, output
        assert 'TRUSTED_SIGNATURE_DISPATCH' in output
    else:
        assert result.returncode != 0
        assert 'TRUSTED_SIGNATURE_DISPATCH' not in output
        assert ('outside the bounded contributor robot set' if case == 'full'
                else 'must not be a symlink') in output


def test_unknown_arch_valid_scope_is_not_full_receipt():
    p.check_scope(receipt(), {'smoke'}, {'smoke', 'extra', 'unused'})


def test_small_native_class_allowance_and_unknown_fallback():
    from test.resource_admission import MemoryAdmission
    gib = 1024**2
    smoke = p.manifest(p.ROOT / 'test/gpu-proof-provisional-smoke.txt')
    full = p.manifest(p.ROOT / 'test/gpu-proof-tests.txt')
    optional = p.optional_units(full, smoke)
    allowance = p.unit_peak_kb(smoke, smoke, optional)
    assert allowance == gib  # 2 * 0.39 GiB + 0.22 GiB headroom
    for group in optional:
        assert p.unit_peak_kb(group, smoke, optional) == int(8 * 1.3 * gib)
    for group in (set(), smoke | {'unreviewed'}, {'test/new.py::unknown'}):
        assert p.unit_peak_kb(group, group, []) == int(16 * 1.3 * gib)
    # The contributor floor admits measured smoke on an 8 GiB available lane;
    # it does not admit unmeasured optional groups on the same machine.
    for available, expected in ((16, True), (8, True), (3, True), (2.9, False)):
        admission = MemoryAdmission(floor_kb=p.CONTRIBUTOR_RESERVE_KB,
                                    available=lambda: available*gib, rss=lambda pid: 0)
        assert admission.reserve('smoke', allowance) is expected
    admission = MemoryAdmission(floor_kb=p.CONTRIBUTOR_RESERVE_KB,
                                available=lambda: 8*gib, rss=lambda pid: 0)
    assert not admission.reserve('optional', p.unit_peak_kb(optional[0], smoke, optional))


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
        if cmd[1] == 'ls-tree':
            return ('100644 blob ' + 'c' * 40 + '\treceipts/provisional.json\0'
                    if kwargs['cwd'] == candidate else '')
        if cmd[1] == 'cat-file': return json.dumps(r)
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
                        'grid_codegen/changed.py' if cmd[1] == 'diff' else output(cmd, **kw))
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
    assert '! -L candidate/receipts/provisional.json' in source
    import yaml
    workflow = yaml.load(source, Loader=yaml.BaseLoader)
    steps = workflow['jobs']['evidence']['steps']
    trusted = steps[0]['with']
    assert trusted['repository'] == '${{ github.event.pull_request.base.repo.full_name }}'
    assert trusted['ref'] == 'refs/heads/${{ github.event.pull_request.base.ref }}'
    assert 'base.sha' not in source
    assert steps[1]['working-directory'] == 'trusted'
    assert 'git rev-parse --verify HEAD' in steps[1]['run']
    assert '$GITHUB_STEP_SUMMARY' in steps[1]['run']


@pytest.mark.parametrize('descendant', ['docs/guide.md', 'grid_codegen/source.py'])
def test_inherited_evidence_is_neutral_on_descendants(tmp_path, monkeypatch, descendant):
    trusted, candidate = tmp_path / 'trusted', tmp_path / 'candidate'
    trusted.mkdir()
    def git(root, *args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    def commit(root):
        git(root, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
            'commit', '-qm', 'fixture')
    git(trusted, 'init', '-q')
    (trusted / 'receipts').mkdir()
    # Deliberately not JSON: inherited content must never reach the verifier.
    (trusted / 'receipts/provisional.json').write_text('historical evidence')
    git(trusted, 'add', 'receipts/provisional.json'); commit(trusted)
    subprocess.run(['git', 'clone', '-q', str(trusted), str(candidate)], check=True)
    changed = candidate / descendant
    changed.parent.mkdir(parents=True); changed.write_text('new source')
    git(candidate, 'add', descendant); commit(candidate)
    path = candidate / 'receipts/provisional.json'
    p.verify(path, root=candidate, trusted=trusted, author='another-contributor',
             head=git(candidate, 'rev-parse', 'HEAD'))
    # Changing evidence cannot borrow the inherited exemption.
    path.write_text('changed invalid evidence')
    git(candidate, 'add', 'receipts/provisional.json'); commit(candidate)
    with pytest.raises(json.JSONDecodeError):
        p.verify(path, root=candidate, trusted=trusted, author='another-contributor',
                 head=git(candidate, 'rev-parse', 'HEAD'))


def test_inherited_git_symlink_is_rejected_before_content(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'receipts').mkdir()
    path = tmp_path / 'receipts/provisional.json'
    path.symlink_to('/private/sentinel')
    subprocess.run(['git', 'add', 'receipts/provisional.json'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-qm', 'symlink'], cwd=tmp_path, check=True)
    with pytest.raises(ValueError, match='regular Git blob'):
        p.evidence_blob(tmp_path, 'HEAD')


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
        def __init__(self, **kw):
            from test.host_memory import effective_available_kb
            assert kw['floor_kb'] == p.CONTRIBUTOR_RESERVE_KB
            assert kw['available'] is effective_available_kb
            self.blocked = None
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
            assert kw['env']['GRID_RBD_BUILD_JOBS'] == '1'
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

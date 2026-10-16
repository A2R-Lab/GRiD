"""Acceptance matrix for the launcher; all driver/verifier calls are simulated."""
import json
from types import SimpleNamespace

import pytest

from test import run_validation as launcher


@pytest.fixture
def launch(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    (root / 'test').mkdir(parents=True)
    original = {'tests': [{'node_id': 'required'}],
                'shards': [{'name': 'fresh'}], 'identity': 'original'}
    before = {'shards': {'fresh': [{'content_sha256': 'old'}]}}
    (root / 'gpu-proof.json').write_text(json.dumps(original))
    (root / 'test/gpu-proof-header-keys.json').write_text(json.dumps(before))
    (root / 'test/gpu-proof-tests.txt').write_text('required\n')
    monkeypatch.setattr(launcher, 'ROOT', root)
    monkeypatch.setattr(launcher, 'LOCK_PATH', tmp_path / 'lock')
    state = SimpleNamespace(root=root, out=tmp_path / 'run', calls=[],
                            dirty='', head='candidate', outcome='success')

    def git(command, **kwargs):
        return state.dirty if command[1] == 'status' else state.head

    def run(command, **kwargs):
        state.calls.append(command)
        if command[1] == '-u':
            if state.outcome == 'interrupt':
                raise KeyboardInterrupt
            if state.outcome == 'driver-fail':
                return SimpleNamespace(returncode=3)
            if state.outcome == 'noop':
                return SimpleNamespace(returncode=0)
            shards = launcher.Path(command[-1])
            (shards / 'receipts').mkdir(parents=True, exist_ok=True)
            rows = [{'content_sha256': 'new'}, {'content_sha256': 'new2'}]
            (shards / 'receipts/fresh.header_keys.jsonl').write_text(
                '\n'.join(json.dumps(row) for row in rows))
            (root / 'test/gpu-proof-header-keys.json').write_text(
                json.dumps({'shards': {'fresh': rows}}))
            receipt = dict(original, identity='new')
            if state.outcome == 'missing-scope':
                receipt['tests'] = []
            if state.outcome == 'bad-headers':
                (shards / 'receipts/fresh.header_keys.jsonl').unlink()
            if state.outcome == 'head-moved':
                state.head = 'another-candidate'
            (shards / 'gpu-proof.json').write_text(json.dumps(receipt))
        elif state.outcome == 'verify-fail':
            raise launcher.subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(launcher.subprocess, 'check_output', git)
    monkeypatch.setattr(launcher.subprocess, 'run', run)
    monkeypatch.setattr(launcher, 'run_driver', lambda command, **kwargs: run(command))

    def invoke(*extra):
        monkeypatch.setattr('sys.argv', ['run_validation', '--out', str(state.out), *extra])
        launcher.main()

    state.invoke = invoke
    state.receipt = lambda: json.loads((root / 'gpu-proof.json').read_text())
    state.status = lambda: json.loads((state.out / 'state.json').read_text())
    return state


def test_dry_launcher_has_no_output_or_lock_side_effects(launch):
    launch.invoke()
    assert not launch.calls and not launch.out.exists()
    assert not launcher.LOCK_PATH.exists()


def test_real_sigterm_forwards_to_owned_driver_and_waits(tmp_path):
    """Real CPU subprocesses, no receipt or GPU; parent SIGTERM leaves no child."""
    import os
    import signal
    import subprocess
    import sys
    import time
    marker = tmp_path / 'child.pid'
    child = f'import os,time; open({str(marker)!r}, "w").write(str(os.getpid())); time.sleep(60)'
    script = ('from test.run_validation import run_driver\n'
              f'import sys\nwith open({str(tmp_path / "lock")!r}, "a") as lock, '
              f'open({str(tmp_path / "log")!r}, "w") as log:\n'
              f' run_driver([sys.executable,"-c",{child!r}],log=log,lock=lock)\n')
    parent = subprocess.Popen([sys.executable, '-c', script], cwd=launcher.ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert marker.exists(), 'child did not start'
        pid = int(marker.read_text())
        parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=10) != 0
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if parent.poll() is None:
            parent.kill(); parent.wait()


def test_dirty_candidate_is_rejected_before_creating_evidence(launch):
    launch.dirty = ' M source.py'
    with pytest.raises(SystemExit, match='clean'):
        launch.invoke('--execute')
    assert not launch.calls and not launch.out.exists()


def test_recording_preserves_fresh_headers_and_awaits_evidence_commit(launch):
    launch.invoke('--execute')
    assert launch.receipt()['identity'] == 'new'
    assert launch.status()['status'] == 'awaiting-evidence-commit'
    keys = json.loads((launch.root / 'test/gpu-proof-header-keys.json').read_text())
    assert len(keys['shards']['fresh']) == 2
    assert len(launch.calls) == 1  # no dirty-tree verifier or hidden retry
    assert set(launch.status()['evidence_hashes']) == {'gpu-proof.json', 'test/gpu-proof-header-keys.json'}


@pytest.mark.parametrize('outcome', [
    'driver-fail', 'interrupt', 'bad-headers', 'head-moved',
])
def test_failure_retains_evidence_without_promoting_receipt(launch, outcome):
    launch.outcome = outcome
    with pytest.raises((RuntimeError, ValueError, KeyboardInterrupt,
                        launcher.subprocess.CalledProcessError)):
        launch.invoke('--execute')
    assert launch.receipt()['identity'] == 'original'
    assert launch.status()['status'] == 'failed'
    assert (launch.out / 'header-keys.before.json').exists()


def test_noop_refresh_verifies_existing_receipt(launch):
    launch.outcome = 'noop'
    launch.invoke('--execute')
    assert launch.status()['status'] == 'verified'
    assert launch.receipt()['identity'] == 'original'


def test_busy_lock_never_starts_driver(launch):
    with launcher.LOCK_PATH.open('a') as lock:
        launcher.fcntl.flock(lock, launcher.fcntl.LOCK_EX | launcher.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            launch.invoke('--execute')
    assert not launch.calls and launch.status()['status'] == 'failed'


@pytest.mark.parametrize('candidate', [None, 'wrong', 'candidate'])
def test_resume_requires_original_candidate_and_saved_plan(launch, candidate):
    original = launch.out.parent / 'original'
    shards = original / 'shards'
    shards.mkdir(parents=True)
    (shards / 'refresh_plan.json').write_text('{}')
    if candidate is not None:
        (original / 'state.json').write_text(json.dumps({'candidate': candidate}))
    if candidate != 'candidate':
        with pytest.raises(SystemExit, match='state.json|candidate'):
            launch.invoke('--execute', '--resume', str(shards))
        assert not launch.calls
    else:
        launch.invoke('--execute', '--resume', str(shards))
        assert '--resume' in launch.calls[0]
        assert '--refresh-from' not in launch.calls[0]
        assert launch.status()['status'] == 'awaiting-evidence-commit'


def test_full_command_never_carries():
    command = launcher.validation_command('/tmp/test-full', full=True)
    assert '--refresh-from' not in command
    assert '--receipts' in command


def test_finalize_real_git_evidence_transition(tmp_path, monkeypatch):
    """Real clean/dirty Git states; only GPU driver and signature CLI are stubbed."""
    import subprocess
    root = tmp_path / 'repo'
    (root / 'test').mkdir(parents=True)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    git('init', '-q')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'Launcher Test')
    (root / 'source.py').write_text('original\n')
    (root / 'gpu-proof.json').write_text('{}')
    (root / 'test/gpu-proof-header-keys.json').write_text('{"shards": {}}')
    (root / 'test/gpu-proof-tests.txt').write_text('required\n')
    git('add', 'source.py', 'gpu-proof.json', 'test/gpu-proof-header-keys.json', 'test/gpu-proof-tests.txt')
    git('commit', '-qm', 'Candidate')
    out = tmp_path / 'run'
    monkeypatch.setattr(launcher, 'ROOT', root)
    monkeypatch.setattr(launcher, 'LOCK_PATH', tmp_path / 'lock')
    calls = []
    def driver(command, **kwargs):
        calls.append('driver')
        assert '--refresh-from' not in command
        shards = out / 'shards'
        (shards / 'receipts').mkdir(parents=True)
        receipt = {'tests': [{'node_id': 'required'}], 'shards': [{'name': 'fresh'}]}
        row = {'content_sha256': 'new'}
        (shards / 'receipts/fresh.header_keys.jsonl').write_text(json.dumps(row) + '\n')
        (root / 'test/gpu-proof-header-keys.json').write_text(json.dumps({'shards': {'fresh': [row]}}))
        (shards / 'gpu-proof.json').write_text(json.dumps(receipt))
        return SimpleNamespace(returncode=0)
    def verify(receipt, directory, *, full):
        assert not git('status', '--porcelain')
        assert full
        calls.append('verify')
    monkeypatch.setattr(launcher, 'run_driver', driver)
    monkeypatch.setattr(launcher, 'verify_receipt', verify)
    monkeypatch.setattr('sys.argv', ['run_validation', '--out', str(out), '--execute', '--full'])
    launcher.main()
    assert json.loads((out / 'state.json').read_text())['status'] == 'awaiting-evidence-commit'
    with pytest.raises(ValueError, match='clean'):
        launcher.finalize(out)
    git('add', 'gpu-proof.json', 'test/gpu-proof-header-keys.json')
    git('commit', '-qm', 'Evidence')
    launcher.finalize(out)
    assert calls == ['driver', 'verify']
    assert json.loads((out / 'state.json').read_text())['status'] == 'verified'
    # A clean source edit still must not pass the finalization gate.
    (root / 'source.py').write_text('changed\n')
    git('add', 'source.py'); git('commit', '-qm', 'Changed source')
    with pytest.raises(ValueError, match='outside evidence'):
        launcher.finalize(out)
    assert calls == ['driver', 'verify']
    (root / 'source.py').write_text('original\n')
    git('add', 'source.py'); git('commit', '-qm', 'Restore candidate source')
    (root / 'gpu-proof.json').write_text('{"tampered": true}')
    git('add', 'gpu-proof.json'); git('commit', '-qm', 'Tamper evidence')
    with pytest.raises(ValueError, match='evidence changed'):
        launcher.finalize(out)
    assert calls == ['driver', 'verify']


def test_validation_env_strips_overrides_and_sets_receipt_knobs(monkeypatch):
    """The launcher owns the driver's env: stray GRID_*/SPLIT*/pytest overrides
    are dropped; the receipt knobs (compile pool 3, two GPU shards side by
    side — W1 2026-10-10 — and 4 nvcc units per bindings build) are set here,
    not inherited."""
    monkeypatch.setenv('GRID_SPLIT_SHARD_JOBS', '9')
    monkeypatch.setenv('GRID_RBD_BUILD_JOBS', '99')
    monkeypatch.setenv('GRID_CUDA_KIN_THREADS', '1')
    monkeypatch.setenv('PYTEST_ADDOPTS', '-x')
    env = launcher.validation_env()
    assert env['GRID_SPLIT_SHARD_JOBS'] == '2'
    assert env['GRID_SPLIT_COMPILE_JOBS'] == '3'
    assert env['GRID_RBD_BUILD_JOBS'] == '4'
    assert 'GRID_CUDA_KIN_THREADS' not in env and 'PYTEST_ADDOPTS' not in env


# ─── receipts by architecture: the driver picks the path from the table ──────

from pathlib import Path as _Path

from test import run_split_suite as runner

ROOT = _Path(__file__).resolve().parents[1]


def _receipt_on(cap):
    return {'environment': {'gpu_info': {'devices': [{'compute_capability': cap}]}},
            'repo': {'commit_sha': 'f' * 40}, 'shards': [], 'tests': []}


@pytest.mark.parametrize('arch,role,path', [
    (120, 'primary', 'gpu-proof.json'),
    (87, 'secondary', 'receipts/gpu-proof-sm87.json'),
])
def test_driver_prints_the_receipt_target_for_the_detected_arch(monkeypatch, capsys, arch, role, path):
    monkeypatch.setattr(runner, 'detect_cuda_arch', lambda: arch)
    monkeypatch.setattr('sys.argv', ['run_split_suite', '--print-receipt-target'])
    assert runner.main() == 0
    fields = capsys.readouterr().out.rstrip('\n').split('\t')
    assert fields[:3] == [role, str(arch), path] and fields[3]


@pytest.mark.parametrize('arch,match', [(75, 'add a row'), (0, 'no GPU detected')])
def test_driver_refuses_an_arch_without_a_row_before_any_work(monkeypatch, capsys, arch, match):
    monkeypatch.setattr(runner, 'detect_cuda_arch', lambda: arch)
    monkeypatch.setattr(runner.subprocess, 'Popen',
                        lambda *a, **k: pytest.fail('a shard was launched'))
    monkeypatch.setattr('sys.argv', ['run_split_suite', '--print-receipt-target'])
    assert runner.main() == 2
    err = capsys.readouterr().err
    assert err.startswith('FATAL:') and match in err
    if arch:
        assert 'test/gpu-proof-receipts.yaml' in err


@pytest.mark.parametrize('flag', ['--refresh-from', '--receipts-carry-from'])
def test_driver_refuses_to_carry_across_architectures(tmp_path, monkeypatch, capsys, flag):
    """A Jetson must not refresh from (or carry) the RTX 5090's receipt."""
    old = tmp_path / 'gpu-proof.json'
    old.write_text(json.dumps(_receipt_on('12.0')))
    monkeypatch.setattr(runner, 'detect_cuda_arch', lambda: 87)
    monkeypatch.setattr(runner, 'collect_cuda_node_ids',
                        lambda *a, **k: pytest.fail('planning started before the arch guard'))
    monkeypatch.setattr(runner.subprocess, 'Popen',
                        lambda *a, **k: pytest.fail('a shard was launched'))
    before = runner.HEADER_KEYS_PATH
    monkeypatch.setattr('sys.argv', ['run_split_suite', '--receipts', flag, str(old),
                                     '--out', str(tmp_path / 'out')])
    assert runner.main() == 2
    err = capsys.readouterr().err
    assert 'recorded on sm_120' in err and 'sm_87' in err and 'receipts/gpu-proof-sm87.json' in err
    assert runner.HEADER_KEYS_PATH == before


def test_driver_refuses_a_rowless_arch_when_recording(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, 'detect_cuda_arch', lambda: 75)
    monkeypatch.setattr(runner.subprocess, 'Popen',
                        lambda *a, **k: pytest.fail('a shard was launched'))
    monkeypatch.setattr('sys.argv', ['run_split_suite', '--receipts', '--skip-warm',
                                     '--out', str(tmp_path / 'out')])
    assert runner.main() == 2
    assert 'add a row' in capsys.readouterr().err


def test_run_gpu_proof_sh_writes_to_the_table_path():
    """The wrapper asks the driver for the row and never hard-codes the
    primary path as a destination (a secondary run leaves gpu-proof.json alone)."""
    text = (ROOT / 'test/run_gpu_proof.sh').read_text()
    assert 'run_split_suite.py --print-receipt-target' in text
    assert '--gpu-proof-out="$RECEIPT_PATH"' in text
    assert 'cp "$OUT_DIR/gpu-proof.json" "$RECEIPT_PATH"' in text
    assert 'REFRESH_SRC="$RECEIPT_PATH"' in text
    code = [line for line in text.splitlines() if not line.lstrip().startswith('#')]
    assert not any('--gpu-proof-out=gpu-proof.json' in line or
                   'REFRESH_SRC="gpu-proof.json"' in line or
                   'cp "$OUT_DIR/gpu-proof.json" gpu-proof.json' in line for line in code)


def test_launcher_is_primary_only():
    """run_validation.py refreshes from and promotes to gpu-proof.json; a
    secondary architecture records through run_gpu_proof.sh instead."""
    command = launcher.validation_command('/tmp/test-refresh')
    assert command[command.index('--refresh-from') + 1] == str(launcher.ROOT / 'gpu-proof.json')

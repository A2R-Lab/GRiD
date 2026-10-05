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
    (root / 'test/gpu-proof-scope.json').write_text(json.dumps({'node_ids': ['required']}))
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


def test_verified_refresh_preserves_fresh_headers_and_promotes_receipt(launch):
    launch.invoke('--execute')
    assert launch.receipt()['identity'] == 'new'
    assert launch.status()['status'] == 'verified'
    keys = json.loads((launch.root / 'test/gpu-proof-header-keys.json').read_text())
    assert len(keys['shards']['fresh']) == 2
    assert '--expected-skips' in launch.calls[-1]


@pytest.mark.parametrize('outcome', [
    'driver-fail', 'interrupt', 'missing-scope', 'bad-headers', 'head-moved', 'verify-fail',
])
def test_failure_retains_evidence_without_promoting_receipt(launch, outcome):
    launch.outcome = outcome
    with pytest.raises((RuntimeError, ValueError, KeyboardInterrupt,
                        launcher.subprocess.CalledProcessError)):
        launch.invoke('--execute')
    assert launch.receipt()['identity'] == 'original'
    assert launch.status()['status'] == 'failed'
    assert (launch.out / 'header-keys.before.json').exists()
    if outcome == 'verify-fail':
        keys = json.loads((launch.root / 'test/gpu-proof-header-keys.json').read_text())
        assert keys['shards']['fresh'][0]['content_sha256'] == 'new'


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
        assert launch.status()['status'] == 'verified'

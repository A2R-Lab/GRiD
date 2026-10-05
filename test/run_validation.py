"""One tracked launcher for a quiet-window everyday receipt refresh.

Dry by default. Use --execute only in a user-scheduled GPU/compile window.
Holds the shared timing lock throughout, retains logs on failure, and never
restores stale header-key evidence. No clock/governor changes and no timing.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = Path('/tmp/a2rlab-timing.lock')
sys.path.insert(0, str(ROOT))
from test.receipt_integrity import verify_header_keys, verify_scope


def validation_env():
    env = dict(os.environ)
    # Scope/numerical/cache overrides belong in a reviewed test configuration,
    # not inherited accidentally from an unrelated experiment.
    for key in list(env):
        if key.startswith(('GRID_', 'SPLIT')) or key in ('PYTEST_ARGS', 'PYTEST_ADDOPTS', 'SCOPE'):
            env.pop(key)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', MAX_JOBS='1',
               GRID_SPLIT_COMPILE_JOBS='3', GRID_SPLIT_SHARD_JOBS='1')
    return env


def validation_command(shard_dir, resume=False):
    command = [sys.executable, '-u', str(ROOT / 'test/run_split_suite.py'),
               '--receipts', '--domains', 'wrappers,cuda']
    if resume:
        # The driver restores refresh-from and the exact saved partition.
        # Passing refresh-from here would re-plan instead of resuming it.
        return command + ['--resume', str(shard_dir)]
    return command + ['--refresh-from', str(ROOT / 'gpu-proof.json'), '--out', str(shard_dir)]


def run_driver(command, *, log, lock):
    """Forward cancellation to our isolated driver, keeping its lock until exit."""
    previous = signal.getsignal(signal.SIGTERM)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    process = None
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=validation_env(),
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(lock.fileno(),))
        return subprocess.CompletedProcess(command, process.wait())
    except BaseException:
        # The split driver's handler shuts down its separately grouped compilers
        # and pytest workers. Do not release the shared lock before it has exited.
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New evidence directory')
    parser.add_argument('--resume', type=Path, help='Existing split-driver directory to resume')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    shard_dir = args.resume.resolve() if args.resume else out / 'shards'
    command = validation_command(shard_dir, resume=bool(args.resume))
    print(' '.join(command), flush=True)
    if not args.execute:
        print('Dry plan only. No collection, compiles, locks or GPU calls started.')
        return
    if args.resume and not (shard_dir / 'refresh_plan.json').is_file():
        raise SystemExit('Resume requires an existing refresh_plan.json from this launcher.')
    git = lambda *a: subprocess.check_output(['git', *a], cwd=ROOT, text=True).strip()
    if git('status', '--porcelain'):
        raise SystemExit('Commit correctness inputs before recording; tree must be clean.')
    candidate = git('rev-parse', 'HEAD')
    if args.resume:
        prior_state_path = shard_dir.parent / 'state.json'
        if not prior_state_path.is_file():
            raise SystemExit('Resume requires the original launcher state.json beside the shard directory.')
        prior_state = json.loads(prior_state_path.read_text())
        if prior_state.get('candidate') != candidate:
            raise SystemExit('Resume candidate differs from the original run; freeze and plan a new refresh.')
    out.mkdir(parents=True, exist_ok=False)
    keys_path = ROOT / 'test/gpu-proof-header-keys.json'
    before = json.loads(keys_path.read_text())
    (out / 'header-keys.before.json').write_text(json.dumps(before, indent=2) + '\n')
    state = {'candidate': candidate, 'command': command, 'shard_dir': str(shard_dir),
             'status': 'waiting-for-lock'}
    def save():
        (out / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        with LOCK_PATH.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state['status'] = 'running'; save()
            with (out / 'driver.log').open('x') as log:
                result = run_driver(command, log=log, lock=lock)
            state['driver_rc'] = result.returncode
            if result.returncode:
                raise RuntimeError(f'Driver rc={result.returncode}; evidence retained in {shard_dir}')
            if git('rev-parse', 'HEAD') != candidate:
                raise RuntimeError('HEAD changed during recording')
            merged = shard_dir / 'gpu-proof.json'
            receipt = json.loads(merged.read_text()) if merged.exists() else json.loads((ROOT / 'gpu-proof.json').read_text())
            verify_scope(receipt, json.loads((ROOT / 'test/gpu-proof-scope.json').read_text()))
            if merged.exists():
                verify_header_keys(before, json.loads(keys_path.read_text()), receipt, shard_dir / 'receipts')
            elif json.loads(keys_path.read_text()) != before:
                raise RuntimeError('No-op refresh unexpectedly changed header keys')
            verify = [str(Path(sys.executable).with_name('gpu-proof')), 'verify',
                      '--receipt', str(merged if merged.exists() else ROOT / 'gpu-proof.json'),
                      '--repo', str(ROOT), '--policy', 'test/gpu-proof-policy.yaml',
                      '--expected-skips', 'test/gpu-proof-expected-skips.txt']
            with (out / 'verify.log').open('x') as log:
                subprocess.run(verify, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            if merged.exists():
                import shutil
                shutil.copyfile(merged, ROOT / 'gpu-proof.json')
            state['status'] = 'verified'; save()
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); save()
        raise


if __name__ == '__main__':
    main()

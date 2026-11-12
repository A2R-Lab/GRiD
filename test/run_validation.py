"""Tracked launcher for everyday refreshes and fresh full release receipts.

Dry by default. Use --execute only in a user-scheduled GPU/compile window.
Holds the shared timing lock throughout, retains logs on failure, and never
restores stale header-key evidence. No clock/governor changes and no timing.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import shutil
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = Path('/tmp/a2rlab-timing.lock')
sys.path.insert(0, str(ROOT))
from test.receipt_integrity import verify_header_keys, verify_manifest_scope


def validation_env(profile='serial'):
    if profile not in ('serial', 'bounded-overlap'):
        raise ValueError(f'Unsupported receipt profile: {profile}')
    env = dict(os.environ)
    # Scope/numerical/cache overrides belong in a reviewed test configuration,
    # not inherited accidentally from an unrelated experiment.
    for key in list(env):
        if key.startswith(('GRID_', 'SPLIT')) or key in ('PYTEST_ARGS', 'PYTEST_ADDOPTS', 'SCOPE'):
            env.pop(key)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', MAX_JOBS='1',
               # Match correctness/pilot allocation: do not reserve most of
               # VRAM before the other backend in a shard has initialized.
               XLA_PYTHON_CLIENT_PREALLOCATE='false',
               # Serial drains prewarm first; bounded overlap reserves the
               # whole shard, including inline compilers, with the pool.
               # Both enforce admission even for the first job.
               GRID_SPLIT_RESOURCE_PROFILE=profile,
               GRID_SPLIT_COMPILE_JOBS='1' if profile == 'serial' else '2', GRID_SPLIT_SHARD_JOBS='1',
               GRID_RBD_BUILD_JOBS='2')
    return env


def validation_command(shard_dir, resume=False, full=False):
    command = [sys.executable, '-u', str(ROOT / 'test/run_split_suite.py'),
               '--receipts', '--domains', 'wrappers,cuda']
    if resume:
        # The driver restores refresh-from and the exact saved partition.
        # Passing refresh-from here would re-plan instead of resuming it.
        return command + ['--resume', str(shard_dir)]
    if not full:
        command += ['--refresh-from', str(ROOT / 'gpu-proof.json')]
    return command + ['--out', str(shard_dir)]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_receipt(receipt_path, out, *, full=False):
    policy = 'test/gpu-proof-policy-release.yaml' if full else 'test/gpu-proof-policy.yaml'
    command = [str(Path(sys.executable).with_name('gpu-proof')), 'verify',
               '--receipt', str(receipt_path), '--repo', str(ROOT), '--policy', policy,
               '--expected-skips', 'test/gpu-proof-expected-skips.txt']
    with (out / 'verify.log').open('a') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def finalize(out):
    """Verify committed evidence only; never launch a driver or change source."""
    out = out.resolve()
    state_path = out / 'state.json'
    state = json.loads(state_path.read_text())
    if state['status'] not in ('awaiting-evidence-commit', 'verified'):
        raise ValueError('Only completed evidence can be finalized')
    if git('status', '--porcelain'):
        raise ValueError('Commit generated evidence before finalizing; tree must be clean')
    subprocess.run(['git', 'merge-base', '--is-ancestor', state['candidate'], 'HEAD'],
                   cwd=ROOT, check=True)
    allowed = {'gpu-proof.json', 'test/gpu-proof-header-keys.json'}
    changed = set(git('diff', '--name-only', state['candidate'], 'HEAD').splitlines())
    if changed - allowed:
        raise ValueError(f'Candidate changed outside evidence: {sorted(changed - allowed)}')
    for relative, expected in state['evidence_hashes'].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f'Generated evidence changed: {relative}')
    receipt = json.loads((ROOT / 'gpu-proof.json').read_text())
    verify_manifest_scope(receipt, ROOT / 'test/gpu-proof-tests.txt')
    verify_header_keys(json.loads((out / 'header-keys.before.json').read_text()),
                       json.loads((ROOT / 'test/gpu-proof-header-keys.json').read_text()),
                       receipt, Path(state['shard_dir']) / 'receipts')
    verify_receipt(ROOT / 'gpu-proof.json', out, full=state.get('full', False))
    state.update(status='verified', verified_commit=git('rev-parse', 'HEAD'))
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    print('Receipt verified; no collection or compilation was started.')


def run_driver(command, *, log, lock, profile='serial'):
    """Forward cancellation to our isolated driver, keeping its lock until exit."""
    previous = signal.getsignal(signal.SIGTERM)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    process = None
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=validation_env(profile),
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
    parser.add_argument('--out', type=Path, help='New evidence directory')
    parser.add_argument('--resume', type=Path, help='Existing split-driver directory to resume')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--full', action='store_true', help='Fresh full release run; no carried shards')
    parser.add_argument('--resource-profile', choices=('serial', 'bounded-overlap'), default='serial')
    parser.add_argument('--fallback-from', type=Path,
                        help='Prior failed/paused launcher directory; start a fresh full SERIAL run with no carries')
    parser.add_argument('--finalize', type=Path, help='Verify an existing run after committing its generated evidence; no GPU work')
    args = parser.parse_args()
    if args.finalize:
        if args.out or args.resume or args.execute or args.full or args.fallback_from or args.resource_profile != 'serial':
            parser.error('--finalize is a verification-only operation; do not combine with launch options')
        with LOCK_PATH.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finalize(args.finalize)
        return
    if args.out is None:
        parser.error('--out is required unless using --finalize')
    out = args.out.resolve()
    shard_dir = args.resume.resolve() if args.resume else out / 'shards'
    command = validation_command(shard_dir, resume=bool(args.resume), full=args.full)
    print(' '.join(command), flush=True)
    if not args.execute:
        print('Dry plan only. No collection, compiles, locks or GPU calls started.')
        return
    if git('status', '--porcelain'):
        raise SystemExit('Commit correctness inputs before recording; tree must be clean.')
    candidate = git('rev-parse', 'HEAD')
    if args.resume:
        prior_state_path = shard_dir.parent / 'state.json'
        if not prior_state_path.is_file():
            raise SystemExit('Resume requires the original launcher state.json beside the shard directory.')
        prior_state = json.loads(prior_state_path.read_text())
        if bool(prior_state.get('full', False)) != args.full:
            raise SystemExit('Resume must preserve the original full/refresh mode')
        if prior_state.get('candidate') != candidate:
            raise SystemExit('Resume candidate differs from the original run; freeze and plan a new refresh.')
        original_profile = prior_state.get('resource_profile', {}).get('GRID_SPLIT_RESOURCE_PROFILE', 'serial')
        if original_profile != args.resource_profile:
            raise SystemExit('Resume must preserve resource profile; use a fresh full serial fallback instead')
    fallback = None
    if args.fallback_from:
        if not args.full or args.resume or args.resource_profile != 'serial':
            raise SystemExit('Fallback requires --full, serial, and no --resume; no prior outcomes are carried')
        fallback = json.loads((args.fallback_from.resolve() / 'state.json').read_text())
        if (fallback.get('candidate') != candidate or not fallback.get('full')
                or fallback.get('status') not in ('failed', 'paused', 'interrupted')):
            raise SystemExit('Fallback source must be a stopped full run of this exact candidate')
    out.mkdir(parents=True, exist_ok=False)
    keys_path = ROOT / 'test/gpu-proof-header-keys.json'
    before = json.loads(keys_path.read_text())
    (out / 'header-keys.before.json').write_text(json.dumps(before, indent=2) + '\n')
    env = validation_env(args.resource_profile)
    state = {'candidate': candidate, 'command': command, 'shard_dir': str(shard_dir), 'full': args.full,
             'resource_profile': {key: env[key] for key in (
                 'GRID_SPLIT_RESOURCE_PROFILE', 'GRID_SPLIT_COMPILE_JOBS',
                 'GRID_SPLIT_SHARD_JOBS', 'GRID_RBD_BUILD_JOBS',
                 'XLA_PYTHON_CLIENT_PREALLOCATE')},
             'status': 'waiting-for-lock'}
    if fallback is not None:
        state['fallback'] = {'from': str(args.fallback_from.resolve()),
            'previous_profile': fallback.get('resource_profile'),
            'strategy': 'fresh-full-no-carries'}
    def save():
        (out / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        with LOCK_PATH.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state['status'] = 'running'; save()
            if args.full:
                with (out / 'collection.log').open('x') as collection_log:
                    subprocess.run(
                        [sys.executable, str(ROOT / 'test/run_split_suite.py'),
                         '--check-manifest', str(ROOT / 'test/gpu-proof-tests.txt')],
                        cwd=ROOT, env=env, stdout=collection_log,
                        stderr=subprocess.STDOUT, check=True)
            with (out / 'driver.log').open('x') as log:
                result = run_driver(command, log=log, lock=lock, profile=args.resource_profile)
            state['driver_rc'] = result.returncode
            if result.returncode:
                if result.returncode == 3 and (shard_dir / 'PAUSE').exists():
                    state['status'] = 'paused'
                raise RuntimeError(f'Driver rc={result.returncode}; evidence retained in {shard_dir}')
            if git('rev-parse', 'HEAD') != candidate:
                raise RuntimeError('HEAD changed during recording')
            merged = shard_dir / 'gpu-proof.json'
            receipt = json.loads(merged.read_text()) if merged.exists() else json.loads((ROOT / 'gpu-proof.json').read_text())
            verify_manifest_scope(receipt, ROOT / 'test/gpu-proof-tests.txt')
            if merged.exists():
                verify_header_keys(before, json.loads(keys_path.read_text()), receipt, shard_dir / 'receipts')
            elif json.loads(keys_path.read_text()) != before:
                raise RuntimeError('No-op refresh unexpectedly changed header keys')
            if merged.exists():
                # Driver emits tracked headers. Stage both evidence files for an
                # explicit maintainer commit, then verify the CLEAN descendant.
                dirty = git('status', '--porcelain').splitlines()
                if any(line.strip()[2:] != 'test/gpu-proof-header-keys.json' for line in dirty):
                    raise RuntimeError('Unexpected dirty input after recording')
                shutil.copyfile(merged, ROOT / 'gpu-proof.json')
                state['evidence_hashes'] = {p: file_hash(ROOT / p) for p in
                    ('gpu-proof.json', 'test/gpu-proof-header-keys.json')}
                state['status'] = 'awaiting-evidence-commit'; save()
                print('Recording complete. Commit gpu-proof.json and test/gpu-proof-header-keys.json, '
                      f'then run {sys.executable} test/run_validation.py --finalize {out}', flush=True)
            else:
                if args.full:
                    raise RuntimeError('Full run did not produce a fresh merged receipt')
                verify_receipt(ROOT / 'gpu-proof.json', out)
                state['status'] = 'verified'; save()
    except BaseException as exc:
        state.update(status='paused' if state['status'] == 'paused' else 'failed', error=repr(exc)); save()
        raise


if __name__ == '__main__':
    main()

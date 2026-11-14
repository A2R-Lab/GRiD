"""Bounded contributor evidence, never a primary or release receipt.

Recording is called only by run_validation while it owns coordination.
Verification executes no candidate code; run this module from the trusted PR
base, with --repo pointing at the untrusted candidate checkout.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test.receipt_integrity import node_ids, receipt_arch, _gpu_proof_bin


def manifest(path):
    lines = [s for s in path.read_text().splitlines() if s and not s.startswith('#')]
    if not lines or lines != sorted(set(lines)):
        raise ValueError(f'noncanonical or empty manifest: {path}')
    return set(lines)


def check_scope(receipt, smoke, allowed):
    actual = node_ids(receipt)
    collected = receipt['session']['node_ids']
    if len(collected) != len(set(collected)) or set(collected) != actual:
        raise ValueError('unfinished or duplicate collection')
    if not smoke <= actual or not actual <= allowed:
        raise ValueError('missing mandatory smoke or out-of-policy extra tests')
    if any(t.get('outcome') != 'passed' for t in receipt['tests']):
        raise ValueError('every provisional outcome must pass (including optional tests)')
    if any(s.get('carried') for s in receipt.get('shards', [])):
        raise ValueError('provisional evidence cannot carry outcomes')
    if receipt_arch(receipt) is None:
        raise ValueError('provisional evidence must identify its GPU')


def verify(path, *, root=ROOT, trusted=ROOT, author, head):
    """Fail closed on unsigned, self-selected scope/signers, or source descendants."""
    import yaml
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}', author):
        raise ValueError('invalid PR author')
    if not re.fullmatch(r'[0-9a-f]{40}', head):
        raise ValueError('expected full PR head SHA')
    actual_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if actual_head != head:
        raise ValueError('verification checkout is not the PR head')
    path = Path(path).resolve()
    if path != (root / 'receipts/provisional.json').resolve():
        raise ValueError('CI only accepts receipts/provisional.json')
    path.relative_to(root.resolve())
    receipt = json.loads(path.read_text())
    smoke = manifest(trusted / 'test/gpu-proof-provisional-smoke.txt')
    allowed = manifest(trusted / 'test/gpu-proof-tests.txt')
    check_scope(receipt, smoke, allowed)
    extra = node_ids(receipt) - smoke
    units = optional_units(allowed, smoke)
    if extra - {n for unit in units for n in unit}:
        raise ValueError('extra tests are outside the bounded contributor robot set')
    if any(extra & set(unit) and not set(unit) <= extra for unit in units):
        raise ValueError('optional compilation atom was only partly recorded')
    commit = receipt['repo']['commit_sha']
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('invalid recorded commit')
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, head], cwd=root, check=True)
    changed = subprocess.check_output(['git', 'diff', '--name-only', commit, head],
                                      cwd=root, text=True).splitlines()
    if set(changed) - {'receipts/provisional.json'}:
        raise ValueError('source changed after recording; only the receipt commit may follow')
    policy = yaml.safe_load((trusted / 'test/gpu-proof-policy-provisional.yaml').read_text())
    release = yaml.safe_load((trusted / 'test/gpu-proof-policy-release.yaml').read_text())
    for key in ('required_fingerprint_paths', 'required_fingerprint_extra_paths',
                'required_fingerprint_excluded_paths'):
        policy[key] = release[key]
    policy.update(signer_mode='restricted', allowed_signers=[author])
    with tempfile.TemporaryDirectory(prefix='grid-provisional-verify-') as directory:
        rules = Path(directory) / 'policy.json'
        rules.write_text(json.dumps(policy))
        subprocess.run([_gpu_proof_bin(), 'verify', '--receipt', str(path), '--repo', str(root),
                        '--policy', str(rules), '--github-user', author, '--require-gpu',
                        '--expected-skips', str(trusted / 'test/gpu-proof-provisional-skips.txt')],
                       cwd=trusted, check=True)
    print(f'Provisionally verified on sm_{receipt_arch(receipt)} by {author}; not release evidence.')


def optional_units(ids, smoke):
    """Small-robot compile atoms; conservative all-stale selection for now.

    No dependency-closure claim. Already-run smoke IDs are removed, while the
    remaining thread variants stay grouped. Exact full manifest bounds scope.
    """
    from test.run_split_suite import _atom_key
    groups = {}
    for node in sorted(ids - smoke):
        if '/test_cuda_executable_equivalence.py::' not in node:
            continue
        key = _atom_key(node)
        if len(key) > 1 and key[1] in {'iiwa14', 'go2', 'fr3'}:
            groups.setdefault(key, []).append(node)
    return list(groups.values())


def record(out, minutes, author):
    """Serial execution; mandatory smoke completes even if it exceeds budget.

    Budget is an admission bound, not an interrupt deadline. Once a unit
    starts, retain all its outcomes. Any failed unit prevents publication.
    """
    from test.run_validation import validation_env
    from test.compile_sched import mem_available_kb
    from test.compile_sched import session_tree_rss_kb
    from test.resource_admission import MemoryAdmission
    smoke = manifest(ROOT / 'test/gpu-proof-provisional-smoke.txt')
    allowed = manifest(ROOT / 'test/gpu-proof-tests.txt')
    if not smoke <= allowed:
        raise ValueError('smoke is not a subset of the full manifest')
    if not 30 <= minutes <= 60:
        raise ValueError('contributor budget must be 30 to 60 minutes')
    if not author or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}', author):
        raise ValueError('recording requires the contributor GitHub username')
    out = out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError('recording output must be outside the candidate checkout')
    out.mkdir(parents=True, exist_ok=False)
    admission = MemoryAdmission(floor_kb=14 * 1024 * 1024, available=mem_available_kb,
                                rss=session_tree_rss_kb, log_path=out / 'memory-admission.jsonl')
    started = time.monotonic()
    receipts = []
    env = validation_env()
    env['GRID_ENABLE_MUJOCO_KERNELS'] = '0'
    baseline = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    prior = json.loads((ROOT / 'gpu-proof.json').read_text())
    costs = {t['node_id']: t.get('duration_s', 0) for t in prior['tests']}
    units = [sorted(smoke)] + optional_units(allowed, smoke)
    for index, unit in enumerate(units):
        estimate = max(600, 1.3 * sum(costs.get(n, 600) for n in unit))
        if index and time.monotonic() - started + estimate > minutes * 60:
            break
        # Conservative unknown whole-shard allowance + 14 GiB reserve.
        # Never starts if memory is unavailable; never kills a peer.
        token = f'provisional-{index:03d}'
        if not admission.reserve(token, int(16 * 1.3 * 1024 * 1024)):
            if not index:
                raise RuntimeError('insufficient memory for mandatory smoke')
            break
        receipt = out / f'unit-{index:03d}.json'
        command = [sys.executable, '-m', 'pytest', '-q', '-m', 'gpu_proof',
                   '--gpu-proof-enable', f'--gpu-proof-out={receipt}',
                   f'--gpu-proof-github-user={author}',
                   f'--gpu-proof-shard=provisional-{index:03d}', *unit]
        with (out / f'unit-{index:03d}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            admission.attach(token, process.pid)
            while process.poll() is None:
                admission.pressure()  # latch; drain owned unit, never kill peers
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            admission.release(token)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
            if admission.blocked:
                raise RuntimeError(f'memory pressure during unit: {admission.blocked}; logs retained')
        data = json.loads(receipt.read_text())
        check_scope(data, set(unit), set(unit))
        receipts.append(receipt)
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != baseline:
            raise RuntimeError('HEAD moved during provisional recording')
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
            raise RuntimeError('checkout changed during provisional recording')
    destination = out / 'provisional.json'
    subprocess.run([_gpu_proof_bin(), 'merge', '--repo', str(ROOT), '--verify-inputs',
                    '--out', str(destination), *map(str, receipts)], cwd=ROOT, check=True)
    check_scope(json.loads(destination.read_text()), smoke, allowed)
    print(f'Provisional evidence: {destination}; copy to receipts/provisional.json and commit only that file.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--author')
    parser.add_argument('--head')
    parser.add_argument('--record', type=Path)
    parser.add_argument('--minutes', type=int, default=30)
    args = parser.parse_args()
    if args.verify and not args.record and args.author and args.head:
        verify(args.verify, root=args.repo.resolve(), author=args.author, head=args.head)
    elif args.record and not args.verify:
        record(args.record, args.minutes, args.author)
    else:
        parser.error('use --verify PATH --author USER --head SHA, or launcher-owned --record DIR')


if __name__ == '__main__':
    main()

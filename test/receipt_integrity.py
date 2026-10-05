"""Structural evidence checks, complementary to gpu-proof signature verification.

The minimum scope is an explicit reviewed node-ID set, not a test-count floor.
Header-key checks distinguish carried evidence from freshly recorded sidecars.
This module never restores an old header-key file over a fresh recording.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def node_ids(receipt):
    ids = [t['node_id'] for t in receipt['tests']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate test node IDs in receipt')
    return set(ids)


def verify_scope(receipt, scope):
    missing = set(scope['node_ids']) - node_ids(receipt)
    if missing:
        raise ValueError(f'Receipt omits {len(missing)} required tests: {sorted(missing)[:5]}')


def read_sidecar(path):
    """Deduplicate complete records; malformed evidence is an error."""
    unique = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get('content_sha256'):
            raise ValueError(f'Invalid header-key record in {path}')
        unique[json.dumps(row, sort_keys=True)] = row
    return list(unique.values())


def verify_header_keys(before, after, receipt, sidecar_dir):
    old, new = before.get('shards', {}), after.get('shards', {})
    canonical = lambda rows: {json.dumps(r, sort_keys=True) for r in rows}
    attested = {s['name'] for s in receipt['shards']}
    if set(new) - attested:
        raise ValueError('Header keys include unattested shards')
    for shard in receipt['shards']:
        name = shard['name']
        if shard.get('carried'):
            if canonical(old.get(name, [])) != canonical(new.get(name, [])):
                raise ValueError(f'Carried header evidence changed: {name}')
        else:
            path = Path(sidecar_dir) / f'{name}.header_keys.jsonl'
            rows = read_sidecar(path) if path.exists() else []
            if canonical(rows) != canonical(new.get(name, [])):
                raise ValueError(f'Fresh header evidence differs from sidecar: {name}')
            if name in old and old[name] and not rows:
                raise ValueError(f'Fresh shard lost its header-key recording: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--extend-scope', action='store_true',
                        help='Explicit maintainer action: add nodes from a verified full receipt; never remove nodes')
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if args.extend_scope:
        old = json.loads(args.scope.read_text()) if args.scope.exists() else {'node_ids': []}
        ids = set(old['node_ids']) | node_ids(receipt)
        args.scope.write_text(json.dumps({'schema': 1, 'node_ids': sorted(ids)}, indent=2) + '\n')
    verify_scope(receipt, json.loads(args.scope.read_text()))
    print(f'Receipt minimum scope satisfied ({len(node_ids(receipt))} recorded tests)')


if __name__ == '__main__':
    main()

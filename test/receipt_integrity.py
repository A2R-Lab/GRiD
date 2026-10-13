"""Structural evidence checks, complementary to gpu-proof signature verification.

The receipt scope itself is enforced by pytest-gpu-proof (policy
required_test_manifest = test/gpu-proof-tests.txt, an EXACT node-ID set); this
module only writes that manifest from a verified receipt. Header-key checks
distinguish carried evidence from freshly recorded sidecars and never restore
an old header-key file over a fresh recording.
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


def write_manifest(receipt, path):
    """Rewrite the reviewed scope manifest from a receipt's collection, keeping
    the file's leading comment block. Explicit maintainer action: the diff
    (added AND removed node IDs) is reviewed in the PR, and the file is a
    receipt fingerprint path, so it must be committed BEFORE recording."""
    ids = node_ids(receipt)
    header = [line for line in path.read_text().splitlines()
              if line.startswith('#')] if path.exists() else []
    path.write_text('\n'.join(header + sorted(ids)) + '\n')
    return ids


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
    parser.add_argument('--write-manifest', type=Path, required=True, metavar='PATH',
                        help='Explicit maintainer action: rewrite the reviewed scope manifest '
                             '(test/gpu-proof-tests.txt) from a verified full-scope receipt')
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    ids = write_manifest(receipt, args.write_manifest)
    print(f'Wrote {len(ids)} node IDs -> {args.write_manifest}')


if __name__ == '__main__':
    main()

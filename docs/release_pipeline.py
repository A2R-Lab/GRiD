"""Rebuild the reviewed capture selection, without timing or publication.

Usage: python docs/release_pipeline.py --output <new-report-directory>
Figure publication remains a separate, explicitly approved operation.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(__file__).with_name('release_dataset.json')


def validate_baseline_selection(table, requirement=None):
    requirement = requirement or json.loads(DATASET.read_text())['required_baseline']
    fields = ('robot', 'operation', 'backend', 'batch', 'repeat')
    expected = set(itertools.product(requirement['robots'], requirement['operations'],
                                    requirement['backends'], requirement['batches'], requirement['repeats']))
    found = set()
    for row in table['raw_records']:
        key = tuple(row[f] for f in fields)
        if key not in expected:
            continue
        if Path(row['capture']).parent.name != requirement['capture'] or row['status'] != 'validated':
            raise ValueError(f'Required baseline capture was replaced or failed validation: {key}')
        if key in found:
            raise ValueError(f'Duplicate baseline repeat: {key}')
        found.add(key)
    if found != expected:
        raise ValueError(f'Required baseline is incomplete: {len(expected - found)} missing repeats')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(DATASET.read_text())
    baseline = next(ROOT / p for p in recipe['captures'] if Path(p).name == recipe['required_baseline']['capture'])
    if hashlib.sha256((baseline / 'manifest.json').read_bytes()).hexdigest() != recipe['required_baseline']['manifest_sha256']:
        raise ValueError('Required baseline manifest changed')
    command = [sys.executable, '-m', 'test.benchmarks.release.report',
               *recipe['captures'], '--output', str(args.output.resolve())]
    for path in recipe['accepted_source_drift']:
        command += ['--accept-source-drift', path]
    subprocess.run(command, cwd=ROOT, check=True)
    table = json.loads((args.output / 'table.json').read_text())
    validate_baseline_selection(table)
    bad = [r for r in table['cells'] if r['status'] in
           ('contract_mismatch', 'incomplete', 'validation_failed', 'not_collected')]
    if bad:
        raise ValueError(f'{len(bad)} incomplete or incompatible report cells')
    subprocess.run([sys.executable, 'docs/plot_release_figures.py', str(args.output.resolve())], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'docs/export_release_tables.py', '--report', str(args.output.resolve()),
                    '--output', str((args.output / 'website-preview').resolve())], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()

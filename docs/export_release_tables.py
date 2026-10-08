"""Export tables and audit metadata from an explicit report; never a stale hardcoded capture.

Run after plot_release_figures.py. The default output is a preview directory;
writing tracked publication assets requires --approve after user review.
"""
import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/source/_static/release'
sys.path.insert(0, str(ROOT))
from docs.release_pipeline import validate_baseline_selection
from test.benchmarks.release import report
from test.benchmarks.release.protocol import digest


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def export(source, output, approved=False):
    table = json.loads((source / 'table.json').read_text())
    validate_baseline_selection(table)
    raw, rows = table['raw_records'], table['cells']
    if any(r['status'] in ('contract_mismatch', 'incomplete', 'validation_failed', 'not_collected') for r in rows):
        raise ValueError('Report contains incompatible or incomplete cells')
    # Re-read hashed manifests and captures before trusting a saved table.
    report.ACCEPTED_SOURCE_DRIFT.update(table['accepted_source_drift'])
    selected = {}
    for path in table['capture_order']:
        for r in report.records(path):
            key = tuple(r[k] for k in ('robot', 'operation', 'backend', 'batch', 'repeat'))
            if key not in selected or selected[key]['status'] == 'not_collected':
                selected[key] = r
    if list(selected.values()) != raw or report.aggregate(raw) != rows:
        raise ValueError('Saved table does not reproduce from its pinned captures')
    cache = {}
    for r in raw:
        if r['status'] != 'validated':
            continue
        path = r['capture']
        if path not in cache:
            cache[path] = json.loads(Path(path).read_text())
        cell = next(c for c in cache[path]['cells'] if c['batch'] == r['batch'])
        for side in ('host_to_host', 'resident'):
            if side in cell:
                t = cell[side]
                if len(t['samples_us']) != 300 or abs(statistics.mean(t['samples_us']) - t['mean_us']) > max(1e-6, t['mean_us'] * 1e-10):
                    raise ValueError(f'Sample count/mean mismatch: {path} {side}')
    lookup = {(r['robot'], r['operation'], r['backend'], r['batch']): r for r in rows if r['status'] == 'validated'}
    comparisons = []
    for (robot, op, backend, batch), g in lookup.items():
        if backend not in ('grid_cuda', 'grid_jax'):
            continue
        for cb in ('pinocchio', 'pinocchio_plain', 'mjx', 'mujoco_warp', 'mujoco_cpu', 'bard', 'frax', 'curobo'):
            c = lookup.get((robot, op, cb, batch))
            if not c:
                continue
            for side in ('host', 'resident'):
                field = side + '_us'
                if not g.get(field) or not c.get(field):
                    continue
                lo = c[side + '_min_us'] / g[side + '_max_us']
                hi = c[side + '_max_us'] / g[side + '_min_us']
                comparisons.append(dict(robot=robot, operation=op, batch=batch, grid=backend, baseline=cb,
                    boundary=side, ratio=c[field]/g[field], observed_lower=lo, observed_upper=hi,
                    result='range_win' if lo > 1 else 'range_loss' if hi < 1 else 'overlap'))
    variability = []
    for backend in sorted({r['backend'] for r in rows}):
        rr = [r for r in rows if r['backend'] == backend and r['status'] == 'validated']
        variability.append(dict(backend=backend, cells=len(rr),
            host_flags=sum(report.unstable(r, 'host_us') for r in rr),
            resident_flags=sum(report.unstable(r, 'resident_us') for r in rr),
            boundary_flags=sum(bool(r['boundary_flag']) for r in rr)))
    audit = dict(publication_approved=approved, workers=len(cache),
        measurements=sum(r['status'] == 'validated' for r in raw),
        status_counts=dict(Counter(r['status'] for r in rows)), variability=variability,
        raw_manifests={p: digest(Path(p) / 'manifest.json') for p in table['capture_order']},
        sample_counts_and_means_verified=True, hashes_and_contracts_verified=True,
        pinocchio_thread_policy='Core: best of 1,2,4,8,16,24 workers capped by batch; secondary tables retain original policy',
        comparison_definition='Ratios of medians of three run means; observed ranges are not confidence intervals')
    output.mkdir(parents=True, exist_ok=True)
    save(output / 'audit.json', audit)
    save(output / 'comparisons.json', dict(cells=comparisons))
    secondary = []
    for name in ('secondary_table.csv', 'secondary_provenance.json'):
        if output.resolve() != ASSETS.resolve():
            shutil.copyfile(ASSETS / name, output / name)
    with (output / 'secondary_table.csv').open() as f:
        secondary = list(csv.DictReader(f))
    sections = []
    cols = ['robot', 'operation', 'backend', 'batch', 'dtype', 'status', 'threads',
            'host_us', 'host_min_us', 'host_max_us',
            'resident_us', 'resident_min_us', 'resident_max_us', 'reason']
    for title, data in [('Core and wrappers — 300 samples per repeat', rows), ('Secondary operations — original 30-sample protocol', secondary)]:
        head = '<tr>' + ''.join(f'<th>{c}</th>' for c in cols) + '</tr>'
        body = ''.join('<tr>' + ''.join('<td>' + html.escape(str(r.get(c, '—'))) + '</td>' for c in cols) + '</tr>' for r in data)
        sections.append(f'<h2>{title}</h2><div class="scroll"><table>{head}{body}</table></div>')
    (output / 'tables.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>GRiD benchmark tables</title><style>body{font:14px system-ui;margin:2rem}.scroll{overflow:auto;max-height:70vh}td,th{padding:.5rem;border:1px solid #ddd}</style><h1>GRiD measurements</h1><p>September/October 2026 Updated Results. Times are microseconds per batch. Secondary results retain their original protocol.</p>' + ''.join(sections) + '<footer>© 2026 A²R Lab</footer></html>\n')
    manifest = json.loads((output / 'manifest.json').read_text())
    manifest['outputs'] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    save(output / 'manifest.json', manifest)
    print(json.dumps(audit, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--approve', action='store_true')
    args = parser.parse_args()
    if args.output.resolve() == ASSETS.resolve() and not args.approve:
        parser.error('Publication requires --approve after user review')
    export(args.report, args.output, args.approve)


if __name__ == '__main__':
    main()

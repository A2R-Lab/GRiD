"""Audit and export the 27 September tables from pinned captures; CPU only.

Run docs/plot_release_figures.py with --approve first. This adds the full
tables and their audit/provenance to the figure manifest without collecting
new measurements or changing capture manifests.
"""
from collections import Counter
import csv
import html
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test.benchmarks.release import report
from test.benchmarks.release.protocol import CORE, digest

RUN = ROOT / 'test/benchmarks/results/release-final-core-20260927-nevcJ2'
OUT = ROOT / 'test/benchmarks/results/release-final-report-20260927'
ASSETS = ROOT / 'docs/source/_static/release'
OLD = ROOT / 'test/benchmarks/results/release-analysis-20260927/table.json'


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def seal_assets():
    manifest = json.loads((ASSETS/'manifest.json').read_text())
    for path in ASSETS.glob('*.csv'):
        path.write_text(path.read_text())
    for path in ASSETS.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    manifest['outputs']={p.name:digest(p) for p in sorted(ASSETS.iterdir()) if p.is_file() and p.name!='manifest.json'}
    save(ASSETS/'manifest.json',manifest)


def main():
    raw = list(report.records(RUN))  # validates every manifest entry and capture hash
    rows = report.aggregate(raw)
    assert len(raw) == 1782 and len(rows) == 594
    assert Counter(r['status'] for r in rows) == {
        'validated': 420, 'excluded_method': 90, 'adapter_pending': 60, 'model_mismatch': 24}
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
                assert len(t['samples_us']) == 300
                assert abs(statistics.mean(t['samples_us'])-t['mean_us']) < max(1e-6, t['mean_us']*1e-10)
    lookup = {(r['robot'], r['operation'], r['backend'], r['batch']): r for r in rows if r['status']=='validated'}
    comparisons = []
    for (robot, op, backend, batch), g in lookup.items():
        if backend not in ('grid_cuda', 'grid_jax'):
            continue
        for cb in ('pinocchio','pinocchio_plain','mjx','mujoco_warp','mujoco_cpu','bard','frax'):
            c = lookup.get((robot,op,cb,batch))
            if not c:
                continue
            for side in ('host','resident'):
                field = side+'_us'
                if not g.get(field) or not c.get(field):
                    continue
                lo = c[side+'_min_us']/g[side+'_max_us']
                hi = c[side+'_max_us']/g[side+'_min_us']
                comparisons.append(dict(robot=robot,operation=op,batch=batch,grid=backend,baseline=cb,
                    boundary=side,ratio=c[field]/g[field],observed_lower=lo,observed_upper=hi,
                    result='range_win' if lo>1 else 'range_loss' if hi<1 else 'overlap'))
    save(OUT/'comparisons.json',dict(definition='Baseline / GRiD; observed min/max process-mean envelopes, not confidence intervals',cells=comparisons))
    variability = []
    for b in sorted({r['backend'] for r in rows}):
        rr = [r for r in rows if r['backend']==b and r['status']=='validated']
        variability.append(dict(backend=b,cells=len(rr),host_flags=sum(report.unstable(r,'host_us') for r in rr),
            resident_flags=sum(report.unstable(r,'resident_us') for r in rr),
            boundary_flags=sum(bool(r['boundary_flag']) for r in rr)))
    audit = dict(publication_approved=True,commit=json.loads((RUN/'plan.json').read_text())['provenance']['commit'],
        raw_manifest_sha256=digest(RUN/'manifest.json'),workers=210,measurements=1260,
        status_counts=dict(Counter(r['status'] for r in rows)),variability=variability,
        sample_counts_and_means_verified=True,hashes_and_contracts_verified=True,
        comparison_definition='Ratios of medians of three run means; envelopes are observed ranges, not confidence intervals')
    save(OUT/'audit.json',audit)
    save(ASSETS/'audit.json',audit)
    shutil.copyfile(OUT/'comparisons.json',ASSETS/'comparisons.json')
    # Existing secondary results stay separate, retaining their earlier protocol.
    old = json.loads(OLD.read_text())
    for capture in old['capture_order']:
        list(report.records(capture))  # validate raw manifests again, no new timing
    secondary_raw = [r for r in old['raw_records'] if r['operation'] not in CORE]
    secondary = report.aggregate(secondary_raw)
    assert not any(r['status'] in ('contract_mismatch','incomplete','validation_failed') for r in secondary)
    for r in secondary_raw:
        assert json.loads(r['contract'])['iterations'] == 30
    with (ASSETS/'secondary_table.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(secondary[0]),lineterminator='\n');w.writeheader();w.writerows(secondary)
    save(ASSETS/'secondary_provenance.json',dict(publication_approved=True,iterations=30,repeats=3,
         source_table_sha256=digest(OLD),capture_order=old['capture_order'],
         accepted_source_drift=old['accepted_source_drift'],
         status_counts=dict(Counter(r['status'] for r in secondary)),
         manifests={str(p):digest(Path(p)/'manifest.json') for p in old['capture_order']}))
    cols=['robot','operation','backend','batch','dtype','status','host_us','host_min_us','host_max_us',
          'resident_us','resident_min_us','resident_max_us','boundary_flag','reason']
    sections=[]
    for title, data in [('Core and wrappers — 300 samples per repeat',rows),('Secondary operations — 30 samples per repeat',secondary)]:
        head='<tr>'+''.join(f'<th>{c}</th>' for c in cols)+'</tr>'
        body=''.join('<tr>'+''.join('<td>'+html.escape(str(round(r[c],3) if isinstance(r.get(c),float) else r.get(c) if r.get(c) is not None else '—'))+'</td>' for c in cols)+'</tr>' for r in data)
        sections.append(f'<h2>{title}</h2><div class="scroll"><table>{head}{body}</table></div>')
    (ASSETS/'tables.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GRiD benchmark tables</title><style>body{font:14px system-ui;margin:2rem}.scroll{overflow:auto;max-height:70vh}table{border-collapse:collapse}td,th{padding:.5rem;border:1px solid #ddd;white-space:nowrap}th{position:sticky;top:0;background:#eee}</style><h1>GRiD measurements</h1><p>Data from the 27 September 2026 collection. Times are microseconds per batch, median and observed range of three process means. Protocols remain separate; missing data is not zero.</p>'+''.join(sections)+'<footer>© 2026 A²R Lab</footer></html>\n')
    seal_assets()
    print(json.dumps(audit,indent=2))
    print('secondary',len(secondary),Counter(r['status'] for r in secondary))


if __name__=='__main__':
    main()

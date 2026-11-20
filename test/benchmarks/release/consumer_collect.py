"""Bounded collector stage: first-slice native consumer contracts only."""
import json
import sys
from .consumer_contract import BATCHES, ROBOTS, OPERATIONS, compatible, summarize_repeats
from .protocol import write_json

BACKENDS = ('grid_cuda', 'pinocchio_plain')


def plan(args):
    robots, backends = args.robots or list(ROBOTS), args.backends or list(BACKENDS)
    operations, batches = args.operations or list(OPERATIONS), args.batches or list(BATCHES)
    if not set(robots) <= set(ROBOTS) or not set(backends) <= set(BACKENDS) or not set(operations) <= set(OPERATIONS):
        raise ValueError('consumers stage only supports iiwa14/go2, grid_cuda/pinocchio_plain, ' + '/'.join(OPERATIONS))
    if not batches or not set(batches) <= set(BATCHES) or len(batches) != len(set(batches)):
        raise ValueError('consumer batches must be unique members of 1,4,16,64,256,1024')
    if args.prepare_only:
        raise ValueError('use --correctness-only for the consumer correctness smoke')
    if args.smoke or args.accuracy_policy != 'strict':
        raise ValueError('first consumer slice uses --correctness-only and the strict accuracy policy')
    repeats = 1 if args.correctness_only else args.repeats
    if min(repeats, args.iterations, args.warmups, args.cpu_threads, args.timeout) < 1 or args.warm_seconds < 0:
        raise ValueError('invalid consumer run counts')
    return dict(schema=1, boundary_contract_version=1,
                purpose='correctness' if args.correctness_only else 'consumer-measurement',
                batches=batches, repeats=repeats,
                cpu_thread_policy='report 1 and ceiling separately; no tuning',
                jobs=[dict(robot=r, backend=b, operation=o, repeat=i, cpu_threads=t) for r in robots
                      for o in operations for b in backends
                      for t in (sorted({1, args.cpu_threads}) if b == 'pinocchio_plain' else [None])
                      for i in range(repeats)])


def summarize(captures, *, repeats):
    groups = {}
    for capture in captures:
        for cell in capture.get('cells', []):
            metadata = cell.get('contract', {})
            key = (capture['robot'], capture['operation'], metadata.get('batch', cell.get('batch')), capture['backend'],
                   capture.get('cpu_threads', 1 if capture['backend'] == 'pinocchio_plain' else None))
            groups.setdefault(key, []).append((capture['repeat'], cell, capture.get('collector_sources')))
    output = []
    for key, values in sorted(groups.items()):
        ids = [i for i, _, _ in values]
        cells = [cell for _, cell, _ in sorted(values, key=lambda pair: pair[0])]
        sources = values[0][2]
        first = cells[0].get('contract', {})
        row = dict(zip(('robot', 'operation', 'batch', 'backend', 'cpu_threads'), key), contract=first,
                   collector_sources=sources, adapter=cells[0].get('adapter'), comparison_eligible=False)
        if (sorted(ids) == list(range(repeats)) and all(c.get('comparison_eligible')
                and c.get('validation', {}).get('comparison_eligible')
                and c['validation'].get('validated_row_ids') == list(range(first.get('batch', 0)))
                and c['validation'].get('fixture_sha256') == first.get('fixture_sha256') for c in cells)
                and sources and all(s == sources for _, _, s in values)
                and row['adapter'] and all(c.get('adapter') == row['adapter'] for c in cells)
                and all(c['validation'].get('accuracy_policy') == cells[0]['validation'].get('accuracy_policy')
                        and c['validation'].get('accuracy_policy_version') == cells[0]['validation'].get('accuracy_policy_version')
                        for c in cells)
                and all(compatible(first, c.get('contract', {})) for c in cells)):
            row.update(comparison_eligible=True,
                       status='accuracy_warning' if any(c['validation']['status']=='accuracy_warning' for c in cells) else 'validated',
                       boundaries={b: summarize_repeats([c['samples_us'][b] for c in cells], expected_repeats=repeats)
                                   for b in ('gpu', 'host')})
            native = [c.get('native_layout') for c in cells]
            if all(n and n.get('comparison_eligible') and n['validation'].get('comparison_eligible')
                   and n['validation'].get('validated_row_ids') == list(range(first['batch']))
                   and n['validation'].get('fixture_sha256') == first['fixture_sha256']
                   and n['contract'] == native[0]['contract'] for n in native):
                row['native_layout'] = dict(contract=native[0]['contract'], comparison_eligible=True,
                    summary=summarize_repeats([n['samples_us'] for n in native], expected_repeats=repeats))
        output.append(row)
    ratios = []
    for row in output:
        if row['backend'] != 'grid_cuda' or not row['comparison_eligible']:
            continue
        peers = [p for p in output if p['backend'] == 'pinocchio_plain'
                 and all(p[k] == row[k] for k in ('robot', 'operation', 'batch'))]
        for peer in peers:
            if (peer['comparison_eligible'] and row['collector_sources'] == peer['collector_sources']
                    and compatible(row['contract'], peer['contract'])):
                ratios.append(dict(robot=row['robot'], operation=row['operation'], batch=row['batch'],
                                   cpu_threads=peer['cpu_threads'], boundary_contract='dense-contract only',
                                   cpu_over_grid={b: peer['boundaries'][b]['median_repeat_mean_us']/row['boundaries'][b]['median_repeat_mean_us']
                                                  for b in ('gpu', 'host')
                                                  if row['boundaries'][b]['median_repeat_mean_us'] > 0}))
    return dict(schema=1, headline_boundary=None,
                boundary_labels={'gpu': 'GPU dense contract', 'native_layout': 'GPU native layout (GRiD only)', 'host': 'host dense contract'},
                cells=output, ratios=ratios)


def collect(args):
    from .collect import run_job, provenance
    payload = plan(args)
    if not args.execute:
        print(json.dumps(payload, indent=2)); return
    if args.output is None:
        raise ValueError('--execute requires a fresh --output directory')
    args.output.mkdir(parents=True, exist_ok=False)
    payload['provenance'] = provenance()
    write_json(args.output/'plan.json', payload)
    failed, captures = False, []
    for job in payload['jobs']:
        name = '{robot}-{operation}-{backend}-t{cpu_threads}-r{repeat}'.format(**job)
        path = args.output/(name+'.json')
        command = [sys.executable, '-m', 'test.benchmarks.release.consumer_worker',
                   '--robot', job['robot'], '--backend', job['backend'], '--operation', job['operation'],
                   '--batches', *map(str, payload['batches']), '--output', str(path),
                   '--warmups', str(args.warmups), '--iterations', str(args.iterations),
                   '--warm-seconds', str(args.warm_seconds), '--cpu-threads', str(job['cpu_threads'] or 1)]
        if args.correctness_only:
            command.append('--correctness-only')
        rc, expired = run_job(command, args.output/(name+'.log'), args.timeout)
        failed |= bool(rc or expired)
        capture = json.loads(path.read_text()) if path.exists() else dict(**job, cells=[], error='missing output')
        batches = [c.get('contract', {}).get('batch') for c in capture.get('cells', [])]
        if sorted(batches, key=str) != sorted(payload['batches'], key=str):
            failed = True
        capture['repeat'] = job['repeat']; captures.append(capture)
        if failed:
            break  # keep evidence, stop a broken campaign
    report = summarize(captures, repeats=payload['repeats'])
    report.update(purpose=payload['purpose'], campaign_complete=not failed and len(captures)==len(payload['jobs']))
    if failed:
        report['ratios'] = []
    write_json(args.output/'consumer-report.json', report)
    if failed:
        raise SystemExit(1)

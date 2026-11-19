"""One first-slice backend/robot/operation; no timings before symmetric checks."""
import argparse
from pathlib import Path
import traceback
from .consumer_contract import contract, accept
from .protocol import write_json, source_fingerprints


def run_cell(adapter, metadata, expected, *, correctness_only, warmups, iterations, warm_seconds):
    rows = list(range(metadata['batch']))
    snapshots = {b: {phase: adapter.once(b) for phase in ('before', 'after', 'repeat')}
                 for b in ('host', 'gpu')}
    preflight = accept(metadata, rows, expected, snapshots)
    cell = dict(contract=metadata, validation=preflight, status=preflight['status'], comparison_eligible=False)
    if not preflight['comparison_eligible']:
        return cell
    if not correctness_only:
        samples = {}
        for boundary in ('host', 'gpu'):
            samples[boundary], snapshots[boundary]['after'] = adapter.measure(boundary, warmups, iterations, warm_seconds)
            snapshots[boundary]['repeat'] = adapter.once(boundary)
        # 'after' is the actual last timed result, not a replacement call.
        cell['samples_us'] = samples
    cell['validation'] = accept(metadata, rows, expected, snapshots)
    cell['comparison_eligible'] = not correctness_only and cell['validation']['comparison_eligible']
    cell['status'] = cell['validation']['status']
    cell['adapter'] = dict(adapter.metadata)
    return cell


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('robot', 'backend', 'operation'):
        ap.add_argument('--'+name, required=True)
    ap.add_argument('--batches', nargs='+', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--iterations', type=int, default=30)
    ap.add_argument('--warmups', type=int, default=5)
    ap.add_argument('--warm-seconds', type=float, default=1.5)
    ap.add_argument('--cpu-threads', type=int, default=1)
    ap.add_argument('--correctness-only', action='store_true')
    args = ap.parse_args()
    from .fixtures import Fixture
    from .consumer_adapter import ConsumerAdapter, fixture_digest
    capture = dict(schema=1, purpose='correctness' if args.correctness_only else 'consumer-measurement',
                   robot=args.robot, backend=args.backend, operation=args.operation,
                   collector_sources=source_fingerprints(), cells=[])
    failed = False
    try:
        fixture = Fixture(args.robot, max(args.batches))
        identity = fixture_digest(fixture)
        capture['fixture'] = fixture.metadata
        for batch in args.batches:
            adapter = None
            try:
                metadata = contract(args.operation, batch, fixture.nq, fixture.nv, identity)
                expected = fixture.expected(args.operation, batch).reshape(batch, metadata['output']['width'])
                adapter = ConsumerAdapter(args.backend, args.operation, fixture, batch, args.output.parent, args.cpu_threads)
                cell = run_cell(adapter, metadata, expected, correctness_only=args.correctness_only,
                                warmups=args.warmups, iterations=args.iterations, warm_seconds=args.warm_seconds)
                failed |= not cell['validation']['comparison_eligible']
            except Exception as error:
                failed = True
                cell = dict(batch=batch, comparison_eligible=False, status='error', reason=str(error))
                traceback.print_exc()
            finally:
                if adapter is not None:
                    try:
                        adapter.close()
                    except Exception as error:
                        failed = True
                        cell.update(comparison_eligible=False, status='error', reason=f'cleanup: {error}')
            capture['cells'].append(cell)
            write_json(args.output, capture)
    except Exception as error:
        failed = True
        capture['setup_error'] = str(error)
        traceback.print_exc()
    write_json(args.output, capture)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

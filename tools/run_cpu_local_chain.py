"""CPU-only end-to-end experiment on saved inputs; no captured outputs reused."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from tools.cpu_local_chain import chain


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(actual, expected, robot, referee):
    actual, expected = np.asarray(actual), np.asarray(expected)
    try:
        referee._assert_allclose_with_optional_norm_guard(actual, expected, rtol=2e-4, atol=2e-4, err_msg='',
                              **referee._fdsva_so_tolerance(robot))
        passed = True
    except AssertionError:
        passed = False
    return dict(gate=passed, max_abs=float(np.max(np.abs(actual-expected))),
                relative=float(np.linalg.norm(actual-expected) / max(np.linalg.norm(expected), 1e-30)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cases', nargs='*')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from test.cuda_equivalents import test_cuda_second_order_fallback as t
    from tools import cpu_local_chain, cpu_qq_study
    paths = sorted(args.captures.glob('*.npz'))
    if args.cases:
        selected = {v.replace('/', '-', 1) for v in args.cases}
        paths = [p for p in paths if p.stem in selected]
        if {p.stem for p in paths} != selected: raise ValueError('missing requested inputs')
    if not paths: raise ValueError('no input cases')
    args.out.mkdir(parents=True, exist_ok=False)
    def git(*cmd): return subprocess.check_output(['git', *cmd], text=True).strip()
    provenance = dict(source_head=git('rev-parse', 'HEAD'), source_status=git('status', '--short'),
                      numpy=np.__version__,
                      scripts={str(p): sha(p) for p in (Path(__file__), Path(cpu_local_chain.__file__),
                               Path(cpu_qq_study.__file__), Path(t.__file__))},
                      cases=[p.stem for p in paths],
                      candidate_inputs=['quantized q/qd/u', 'pinned parsed model constants and symbolic transforms'],
                      excluded=['captured qdd/Minv/grad/tensors', 'reference numerical intermediates',
                                'external forces', 'nonlinear coupling', 'non-root multi-DoF joints'],
                      limitations=['CPU float32 operation order and libm, not CUDA FMA order',
                                   'finite input set, not a support envelope'])
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    models = {}; failures = []
    for path in paths:
        robot, sample = path.stem.split('-', 1)
        if robot not in models:
            spec = t._robot_spec(robot, 'floating')
            models[robot] = t.build_project_adapter(spec, t.resolve_robot_spec(spec), base_mode='floating')
        model = models[robot]; ref = model.reference
        if 'reference' not in provenance:
            reference_path = Path(inspect.getfile(type(ref))).resolve()
            provenance['reference'] = dict(path=str(reference_path), sha256=sha(reference_path))
            provenance['peers'] = {}
            for peer in ('RBDReference', 'URDFParser', 'GLASS'):
                peer_path = reference_path.parent.parent / peer
                if git('-C', str(peer_path), 'status', '--short'):
                    raise ValueError('dirty peer')
                provenance['peers'][peer] = git('-C', str(peer_path), 'rev-parse', 'HEAD')
            (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        with np.load(path) as saved:
            # Read ONLY inputs. No device outputs enter this experiment.
            q, qd, u = [np.asarray(saved[k], dtype=np.float32).astype(np.float64) for k in ('q','qd','u')]
        actual64 = chain(model, q, qd, u, np.float64)
        expected = t._flatten_second_order_tensors(model.fdsva_so(q, qd, u))
        expected_qdd = model.forward_dynamics(q, qd, u)
        exact_tensors = ref.idsva_so(q, qd, actual64['qdd'])
        for name, actual, oracle in [
            ('M', actual64['matrix'], ref.crba(q)),
            ('qdd', actual64['qdd'], expected_qdd),
            ('dc', actual64['dc'], ref.inverse_dynamics_gradient(q, qd, actual64['qdd'])),
            ('tensors', np.asarray((*actual64['tensors'], actual64['dm'])), np.asarray(exact_tensors)),
            ('output', t._flatten_second_order_tensors(actual64['output']), expected),
        ]:
            np.testing.assert_allclose(actual, oracle, atol=2e-8, rtol=2e-9, err_msg=name)
        row = dict(robot=robot, sample=sample, capture_sha256=sha(path), nv=model.nv,
                   acceptance=dict(rtol=2e-4, atol=2e-4, **t._fdsva_so_tolerance(robot)),
                   identity64_max=float(np.max(np.abs(t._flatten_second_order_tensors(actual64['output'])-expected))),
                   results={})
        for order in ('forward', 'reverse'):
            actual = chain(model, q, qd, u, np.float32, order)
            result = metrics(t._flatten_second_order_tensors(actual['output']), expected, robot, t)
            result['qdd_max_abs'] = float(np.max(np.abs(actual['qdd']-expected_qdd)))
            result['qdd_relative'] = float(np.linalg.norm(actual['qdd']-expected_qdd)/max(np.linalg.norm(expected_qdd),1e-30))
            row['results'][order] = result
            if not result['gate']: failures.append(f'{path.stem}/{order}')
        (args.out / f'{path.stem}.json').write_text(json.dumps(row, indent=2)+'\n')
        with (args.out / 'results.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(robot, sample, row['results'], flush=True)
    (args.out / 'summary.json').write_text(json.dumps(dict(cases=len(paths), failures=failures), indent=2)+'\n')
    return 2 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())

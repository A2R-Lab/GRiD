"""Run a bounded CPU-only q-Hessian study against existing saved captures.

Requires the explicitly selected study_a/f/g scripts and captures directory.
Never launches a GPU, compiler, receipt or timing job. Refuses output overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

try:
    from .cpu_qq_study import local_qq, world_qq
except ImportError:
    from cpu_qq_study import local_qq, world_qq


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entry_attribution(error, inverse, limit=12):
    """Linear q-Hessian error propagation; not a full float32 simulation."""
    propagated = -np.einsum('ab,bjk->ajk', inverse, error)
    index = tuple(map(int, np.unravel_index(np.abs(propagated).argmax(), propagated.shape)))
    a, j, k = index
    terms = -inverse[a] * error[:, j, k]
    ranked = np.argsort(np.abs(terms))[::-1][:limit]
    weights = np.max(np.abs(inverse), axis=0)[:, None, None] * np.abs(error)
    top = np.argsort(weights.reshape(-1))[::-1][:limit]
    return dict(worst_output_index=index, worst_output=float(propagated[index]),
                contribution_abs_sum=float(np.abs(terms).sum()),
                cancellation_ratio=float(np.abs(terms).sum() / max(abs(terms.sum()), 1e-300)),
                contributing_torque_rows=[dict(row=int(i), value=float(terms[i]),
                                               tensor_error=float(error[i, j, k])) for i in ranked],
                ranked_tensor_entries=[dict(index=list(map(int, np.unravel_index(int(i), error.shape))),
                                           weighted_error=float(weights.reshape(-1)[i]),
                                           error=float(error.reshape(-1)[i])) for i in top])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cases', nargs='*', help='robot/sample; absent means all saved captures')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(args.study_dir.resolve()))
    G = importlib.import_module('study_g_mass_derivative')
    F, A, t = G.F, G.A, G.t
    captures = sorted((args.study_dir / 'captures').glob('*.npz'))
    if args.cases:
        selected = {c.replace('/', '-', 1) for c in args.cases}
        captures = [p for p in captures if p.stem in selected]
        if {p.stem for p in captures} != selected:
            raise ValueError('requested capture absent')
    if not captures: raise ValueError('no captures')
    args.out.mkdir(parents=True, exist_ok=False)
    def git(*cmd):
        return subprocess.check_output(['git', *cmd], text=True).strip()
    provenance = dict(argv=sys.argv, source_head=git('rev-parse', 'HEAD'),
                      source_status=git('status', '--short'), numpy=np.__version__,
                      scripts={str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name('cpu_qq_study.py'),
                               Path(A.__file__), Path(F.__file__), Path(G.__file__)]},
                      assumptions=['CPU arithmetic, not CUDA instruction order',
                                   'Rounded oracle spatial inputs; not generated transform evaluation',
                                   'Bias RHS and first-order ID derivative remain oracle substitutions',
                                   'Unreplaced device tensor error transplanted to candidate acceleration'])
    (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    models = {}
    for path in captures:
        robot, sample = path.stem.split('-', 1)
        if robot not in models:
            spec = t._robot_spec(robot, 'floating')
            models[robot] = t.build_project_adapter(spec, t.resolve_robot_spec(spec), base_mode='floating')
        model = models[robot]; ref = model.reference; n = model.nv
        if 'reference' not in provenance:
            import inspect
            ref_path = Path(inspect.getfile(type(ref))).resolve()
            provenance['reference'] = dict(path=str(ref_path), sha256=sha(ref_path))
            provenance['source_files'] = {str(p): sha(p) for p in [Path(t.__file__).resolve()]}
            provenance['peers'] = {}
            for peer in ('RBDReference', 'URDFParser', 'GLASS'):
                peer_path = ref_path.parent.parent / peer
                provenance['peers'][peer] = subprocess.check_output(
                    ['git', '-C', str(peer_path), 'rev-parse', 'HEAD'], text=True).strip()
                if subprocess.check_output(['git', '-C', str(peer_path), 'status', '--short'], text=True).strip():
                    raise ValueError('dirty numerical study peer')
            (args.out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        z = np.load(path)
        q, qd, u = [A.r32(z[k]).astype(np.float64) for k in ('q', 'qd', 'u')]
        qdd, _, _, _ = A.exact_chain(model, q, qd, u)
        expected = t._flatten_second_order_tensors(model.fdsva_so(q, qd, u))
        at_dev = np.asarray(ref.idsva_so(q, qd, z['qdd'].astype(np.float64)))
        dev = z['idso'].reshape(4, n, n, n)
        matrix = np.asarray(ref.crba(q))
        tree_m = F.crba32(model, F.tree_data(model, q))
        solve = F.methods(tree_m, A.r32(z['minv']))['equilibrated']
        own = solve(A.r32(matrix @ qdd)[:, None])[:, 0]
        exact = np.asarray(ref.idsva_so(q, qd, own.astype(np.float64)))
        local64 = local_qq(model, q, qd, own, np.float64)
        # A convention/index/sign mismatch is a hard error, not a study result.
        np.testing.assert_allclose(local64, exact[0], atol=2e-9, rtol=2e-10)
        world64 = world_qq(model, q, qd, own, np.float64)
        np.testing.assert_allclose(world64, exact[0], atol=2e-9, rtol=2e-10)
        regroup64 = world_qq(model, q, qd, own, np.float64, reassociate=True)
        np.testing.assert_allclose(regroup64, exact[0], atol=2e-9, rtol=2e-10)
        local = local_qq(model, q, qd, own)
        reverse = local_qq(model, q, qd, own, order='reverse')
        world = world_qq(model, q, qd, own)
        regroup = world_qq(model, q, qd, own, reassociate=True)
        grad = -solve(A.r32(ref.inverse_dynamics_gradient(q, qd, own.astype(np.float64))))
        cols = solve(np.eye(n, dtype=np.float32))
        base = exact + dev - at_dev
        base[3] = G.derivative(model, q)
        def output(qq):
            ts = base.copy(); ts[0] = qq
            values = A.contract32(solve, (grad[:, :n], grad[:, n:]), [A.r32(v) for v in ts] + [cols])
            assert all(v.dtype == np.float32 for v in values)
            return t._flatten_second_order_tensors(values)
        candidates = dict(transplanted_qq=base[0], oracle_qq=exact[0], local_qq=local,
                          local_reverse=reverse, world_cpu32=world, world_reassociated32=regroup)
        row = dict(robot=robot, sample=sample, capture_sha256=sha(path), nv=n,
                   acceptance=dict(rtol=2e-4, atol=2e-4, **t._fdsva_so_tolerance(robot)),
                   expected_max_abs=float(np.max(np.abs(expected))),
                   local64_identity_max=float(np.max(np.abs(local64-exact[0]))),
                   world64_identity_max=float(np.max(np.abs(world64-exact[0]))),
                   reassociated64_identity_max=float(np.max(np.abs(regroup64-exact[0]))),
                   results={name: F.metrics(output(qq), expected, robot) for name, qq in candidates.items()},
                   qq_relative_errors={name: A.err(qq, exact[0]) for name, qq in candidates.items()},
                   attribution=entry_attribution(dev[0]-at_dev[0], np.linalg.inv(tree_m.astype(np.float64))))
        # Single-entry substitutions preserve all remaining captured error.
        row['top_entry_substitutions'] = []
        for item in row['attribution']['ranked_tensor_entries'][:6]:
            index = tuple(item['index']); corrected = base[0].copy(); corrected[index] = exact[0][index]
            row['top_entry_substitutions'].append(dict(index=index, **F.metrics(output(corrected), expected, robot)))
        (args.out / (path.stem + '.json')).write_text(json.dumps(row, indent=2) + '\n')
        with (args.out / 'results.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
        print(robot, sample, {k: v['gate'] for k, v in row['results'].items()}, flush=True)


if __name__ == '__main__':
    main()

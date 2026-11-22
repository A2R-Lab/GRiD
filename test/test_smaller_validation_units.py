"""The saved release manifest is enough to test partition structure on CPU."""
import json
from pathlib import Path
from test import run_split_suite as runner

ROOT = Path(__file__).resolve().parents[1]


def test_full_saved_scope_has_single_owner_and_compile_atoms_stay_whole():
    receipt = json.loads((ROOT / 'gpu-proof.json').read_text())
    ids = [t['node_id'] for t in receipt['tests'] if t['node_id'].startswith('test/cuda_equivalents/')]
    durations = {t['node_id']: t['duration_s'] for t in receipt['tests']}
    shards = runner.pack_cuda_shards(ids, durations, 7200)
    assert sorted(n for s in shards for n in s.targets) == sorted(ids)
    homes = {}
    for shard in shards:
        assert len({n.split('::')[0] for n in shard.targets}) == 1
        for node in shard.targets:
            homes.setdefault(runner._atom_key(node), set()).add(shard.name)
        module = Path(shard.targets[0].split('::')[0]).stem
        if module in runner.ISOLATED_ATOM_MODULES:
            assert len({runner._atom_key(n) for n in shard.targets}) == 1
    assert all(len(value) == 1 for value in homes.values())


def test_integrator_base_and_kinematic_batch_ownership():
    prefix = 'test/cuda_equivalents/'
    a = prefix + 'test_cuda_integrator_equivalence.py::test_x[iiwa14-integrator-floating-shared]'
    b = a.replace('shared', 'lite')
    c = a.replace('floating', 'fixed')
    assert runner._atom_key(a) == runner._atom_key(b)
    assert runner._atom_key(a) != runner._atom_key(c)
    a = prefix + 'test_cuda_kinematics_thread_invariance.py::test_x[go2-floating-batch1]'
    assert runner._atom_key(a) == runner._atom_key(a.replace('batch1', 'batch256'))


def test_fresh_clone_seeds_costs_but_rolling_valid_values_win(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'REPO_ROOT', tmp_path)
    ledger = tmp_path / 'durations.json'
    monkeypatch.setattr(runner, 'DURATIONS_PATH', ledger)
    (tmp_path / 'gpu-proof.json').write_text(json.dumps({'tests': [
        {'node_id': 'a', 'duration_s': 12}, {'node_id': 'b', 'duration_s': 20},
        {'node_id': 'invalid', 'duration_s': -1}]}))
    assert runner.load_durations() == {'a': 12, 'b': 20}
    ledger.write_text(json.dumps({'a': 4, 'b': float('nan'), 'c': 6}))
    assert runner.load_durations() == {'a': 4, 'b': 20, 'c': 6}
    ledger.write_text('invalid json')
    assert runner.load_durations() == {'a': 12, 'b': 20}

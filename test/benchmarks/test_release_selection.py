from pathlib import Path
import pytest

from docs.release_pipeline import validate_baseline_selection
from test.benchmarks.release.report import cross_backend_contract


def test_cross_backend_contract_keeps_affinity_and_arithmetic_policy():
    import json
    a = dict(cpu_threads=8, cpu_power={'affinity': list(range(24))}, arithmetic_policy={'cpu_blas_threads': 1})
    b = dict(a, cpu_threads=24)
    assert cross_backend_contract(json.dumps(a)) == cross_backend_contract(json.dumps(b))
    assert json.dumps(a) != json.dumps(b)  # same-backend repeat contract remains strict
    b['cpu_power'] = {'affinity': [0]}
    assert cross_backend_contract(json.dumps(a)) != cross_backend_contract(json.dumps(b))


def test_new_grid_addendum_cannot_drop_required_cpu_baseline():
    rule = dict(robots=['iiwa14'], operations=['inverse_dynamics_gradient'],
                backends=['pinocchio'], batches=[1024], repeats=[0], capture='pin24')
    row = dict(robot='iiwa14', operation='inverse_dynamics_gradient', backend='pinocchio',
               batch=1024, repeat=0, capture='/capture/pin24/worker.json', status='validated')
    validate_baseline_selection({'raw_records': [row]}, rule)
    with pytest.raises(ValueError, match='incomplete'):
        validate_baseline_selection({'raw_records': []}, rule)
    with pytest.raises(ValueError, match='replaced'):
        validate_baseline_selection({'raw_records': [dict(row, capture='/capture/pin8/worker.json')]}, rule)

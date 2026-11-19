import copy
from types import SimpleNamespace
import numpy as np
import pytest
from test.benchmarks.release.consumer_contract import contract
from test.benchmarks.release.consumer_worker import run_cell
from test.benchmarks.release.consumer_collect import plan, summarize


class Adapter:
    metadata = {'test_double': True}
    def __init__(self, fault=None):
        self.fault, self.measurements = fault, []
    def once(self, boundary):
        return np.zeros((4, 3), np.float32)
    def measure(self, boundary, *args):
        self.measurements.append(boundary)
        result = self.once(boundary)
        if self.fault == boundary:
            result[3, 1] = 1
        return [10., 12.], result


@pytest.mark.parametrize('fault', [None, 'host', 'gpu'])
def test_worker_checks_actual_last_timed_outputs(fault):
    adapter = Adapter(fault)
    meta = contract('inverse_dynamics', 4, 3, 3, 'a'*64)
    result = run_cell(adapter, meta, np.zeros((4,3)), correctness_only=False,
                      warmups=1, iterations=2, warm_seconds=0)
    assert result['comparison_eligible'] is (fault is None)
    assert adapter.measurements == ['host', 'gpu']


def test_preflight_failure_prevents_timing_and_smoke_never_times():
    meta = contract('inverse_dynamics', 4, 3, 3, 'a'*64)
    for correct, expected in ((True, np.zeros((4,3))), (False, np.ones((4,3)))):
        adapter = Adapter()
        result = run_cell(adapter, meta, expected, correctness_only=correct,
                          warmups=1, iterations=2, warm_seconds=0)
        assert not result['comparison_eligible'] and adapter.measurements == []


def test_plan_is_equal_repeats_and_bounded():
    args = SimpleNamespace(robots=None, backends=None, operations=None, batches=None,
                           repeats=3, iterations=2, warmups=1, warm_seconds=0,
                           cpu_threads=2, timeout=30, prepare_only=False,
                           correctness_only=False, smoke=False, accuracy_policy='strict')
    result = plan(args)
    assert result['batches'] == [1,4,16,64,256,1024]
    assert len(result['jobs']) == 24
    args.robots = ['g1']
    with pytest.raises(ValueError): plan(args)


def test_report_rejects_missing_repeats_legacy_and_different_boundaries():
    meta = contract('inverse_dynamics', 4, 3, 3, 'a'*64)
    cell = run_cell(Adapter(), meta, np.zeros((4,3)), correctness_only=False,
                    warmups=1, iterations=2, warm_seconds=0)
    captures = [dict(robot='iiwa14', operation='inverse_dynamics', backend=backend,
                     repeat=i, collector_sources={'collector.py': 'a'*64}, cells=[copy.deepcopy(cell)])
                for backend in ('grid_cuda', 'pinocchio_plain') for i in range(3)]
    report = summarize(captures, repeats=3)
    assert len(report['ratios']) == 1
    assert not summarize(captures[:-1], repeats=3)['ratios']
    different = copy.deepcopy(captures)
    different[-1]['cells'][0]['contract']['completion'] = 'asynchronous enqueue'
    assert not summarize(different, repeats=3)['ratios']
    legacy = copy.deepcopy(captures)
    for capture in legacy:
        del capture['cells'][0]['contract']
    assert not summarize(legacy, repeats=3)['ratios']
    changed = copy.deepcopy(captures)
    changed[-1]['cells'][0]['adapter']['cpu_threads'] = 17
    assert not summarize(changed, repeats=3)['ratios']
    changed = copy.deepcopy(captures)
    changed[-1]['collector_sources'] = {'collector.py': 'b'*64}
    assert not summarize(changed, repeats=3)['ratios']
    captures[-1]['cells'][0]['validation']['status'] = 'accuracy_warning'
    assert summarize(captures, repeats=3)['cells'][-1]['status'] == 'accuracy_warning'

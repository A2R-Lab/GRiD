"""CPU acceptance contract before native boundary integration."""
import copy
import numpy as np
import pytest
from test.benchmarks.release.consumer_contract import (
    contract, accept, summarize_repeats, compatible, build_identity)


def fixture():
    expected = np.arange(48, dtype=float).reshape(16, 3)
    c = contract('inverse_dynamics', 16, 3, 3, 'a'*64)
    snapshots = {b: {phase: expected.astype(np.float32) for phase in ('before', 'after', 'repeat')}
                 for b in ('host', 'gpu')}
    return c, expected, snapshots


def test_complete_symmetric_acceptance():
    c, expected, values = fixture()
    result = accept(c, list(range(16)), expected, values)
    assert result['status'] == 'validated' and result['comparison_eligible']
    assert result['validated_row_ids'] == list(range(16))
    assert result['fixture_sha256'] == 'a'*64


@pytest.mark.parametrize('fault', ['wrong_row', 'prefix', 'nan', 'shape', 'chart', 'host_only', 'gpu_only', 'stateful'])
def test_invalid_evidence_is_never_eligible(fault):
    c, expected, values = fixture()
    ids = list(range(16))
    if fault == 'wrong_row': ids[4] = 9
    if fault == 'prefix': ids = ids[:8]
    if fault == 'nan': values['host']['after'][1, 1] = np.nan
    if fault == 'shape': values['gpu']['before'] = expected[:, :2]
    if fault == 'chart': values['host']['before'] = expected[:, ::-1]
    if fault == 'host_only': del values['gpu']
    if fault == 'gpu_only': del values['host']
    if fault == 'stateful': values['gpu']['repeat'][1, 1] += 1
    assert not accept(c, ids, expected, values)['comparison_eligible']


def test_repeats_and_contract_versions_are_not_silently_combined():
    c, _, _ = fixture()
    assert compatible(c, copy.deepcopy(c))
    assert not compatible(c, {})  # legacy has no boundary contract
    assert not compatible(None, None)
    assert not accept(None, [], [], {})['comparison_eligible']
    other = copy.deepcopy(c); other['input']['bytes'] -= 4
    assert not compatible(other, other)
    other = copy.deepcopy(c); other['output']['representation'] = 'native-sparse'
    assert not compatible(c, other)
    other = copy.deepcopy(c); del other['completion']
    assert not compatible(c, other)
    result = summarize_repeats([[10, 12], [20, 22], [11, 13]], expected_repeats=3)
    assert result['repeat_means_us'] == [11, 21, 12]
    assert result['spread_us'] == 10 and result['noisy']
    with pytest.raises(ValueError): summarize_repeats([[10]], expected_repeats=3)


def test_build_identity_covers_code_dependencies_not_runtime_samples(tmp_path):
    source, header = tmp_path/'bridge.cpp', tmp_path/'nested.h'
    source.write_text('source'); header.write_text('header')
    def key(**kwargs):
        return build_identity([source, header], kwargs.get('flags', ['-O2']),
                              kwargs.get('compiler', 'compiler v1'))
    old = key()
    assert old == key()
    for kw in ({'flags': ['-O3']}, {'compiler': 'v2'}):
        assert key(**kw) != old
    header.write_text('changed')
    assert key() != old


def test_batch_cells_share_maximum_compile_capacity(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from test.benchmarks.release import grid_adapter
    from test.benchmarks.release.consumer_adapter import ConsumerAdapter
    capacities = []
    def stop_before_gpu(backend, operation, fixture, capacity, directory):
        capacities.append(capacity)
        raise RuntimeError('stop before GPU')
    monkeypatch.setattr(grid_adapter, 'GridAdapter', stop_before_gpu)
    fixture = SimpleNamespace(spec=SimpleNamespace(robot_id='iiwa14'), q=np.zeros((4,7)), nv=7)
    for batch in (1,4):
        with pytest.raises(RuntimeError, match='stop before GPU'):
            ConsumerAdapter('grid_cuda', 'inverse_dynamics', fixture, batch, tmp_path)
    assert capacities == [4,4]


def test_cuda_root_follows_selected_nvcc(monkeypatch, tmp_path):
    from grid_rbd import _compile
    from test.benchmarks.release.consumer_adapter import cuda_root, host_compiler
    nvcc = tmp_path/'toolkit/bin/nvcc'; nvcc.parent.mkdir(parents=True); nvcc.touch()
    monkeypatch.setattr(_compile, 'find_nvcc', lambda: str(nvcc))
    assert cuda_root() == nvcc.parent.parent
    monkeypatch.setenv('CXX','missing-compiler-for-test')
    with pytest.raises(FileNotFoundError): host_compiler()


@pytest.mark.parametrize('host,kernel', [
    ('end_effector_pose', 'end_effector_pose_kernel'),
    ('end_effector_pose_FR_foot_joint', 'end_effector_pose_kernel_FR_foot_joint'),
])
def test_consumer_kernel_alias_follows_generated_target(tmp_path, host, kernel):
    from test.benchmarks.release.grid_adapter import consumer_kernel_flags
    header = tmp_path/'grid.cuh'
    header.write_text('    #define GRID_RBD_EE_POSE_FN '+host+'\n')
    assert consumer_kernel_flags(header) == ['-DGRID_CONSUMER_EE_KERNEL='+kernel]


def test_consumer_kernel_alias_absent_or_unrecognized(tmp_path):
    from test.benchmarks.release.grid_adapter import consumer_kernel_flags
    header = tmp_path/'grid.cuh'
    header.write_text('// inverse dynamics only\n')
    assert consumer_kernel_flags(header) == []
    header.write_text('#define GRID_RBD_EE_POSE_FN unexpected_target\n')
    with pytest.raises(ValueError, match='unsupported generated pose alias'):
        consumer_kernel_flags(header)


def test_compiler_discovers_nested_and_conditional_headers(tmp_path):
    from test.benchmarks.release.consumer_adapter import dependencies
    source, nested = tmp_path/'bridge.cpp', tmp_path/'nested.h'
    selected, ignored = tmp_path/'selected.h', tmp_path/'ignored.h'
    source.write_text('#include "nested.h"\n')
    nested.write_text('#if SELECTED\n#include "selected.h"\n#else\n#include "ignored.h"\n#endif\n')
    selected.write_text('// selected'); ignored.write_text('// ignored')
    paths = dependencies('g++', source, ['-DSELECTED=1'])
    assert {source, nested, selected} <= set(paths)
    assert ignored not in paths


def test_forward_dynamics_contract_names_torque_and_acceleration():
    from test.benchmarks.release.consumer_contract import contract, compatible
    forward = contract('forward_dynamics', 16, 19, 18, 'b'*64)
    inverse = contract('inverse_dynamics', 16, 19, 18, 'b'*64)
    assert forward['input']['fields'] == ['q', 'v', 'u'] and inverse['input']['fields'] == ['q', 'v', 'a']
    assert forward['input']['width'] == inverse['input']['width'] == 19 + 2*18
    assert forward['output']['width'] == 18 and 'acceleration' in forward['output']['convention']
    assert compatible(forward, forward) and not compatible(forward, inverse)

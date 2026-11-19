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


def test_build_identity_covers_dependencies_flags_compiler_fixture(tmp_path):
    source, header = tmp_path/'bridge.cpp', tmp_path/'nested.h'
    source.write_text('source'); header.write_text('header')
    def key(**kwargs):
        return build_identity([source, header], kwargs.get('flags', ['-O2']),
                              kwargs.get('compiler', 'compiler v1'), kwargs.get('fixture', 'a'*64))
    old = key()
    assert old == key()
    for kw in ({'flags': ['-O3']}, {'compiler': 'v2'}, {'fixture': 'b'*64}):
        assert key(**kw) != old
    header.write_text('changed')
    assert key() != old


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

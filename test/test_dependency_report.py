"""CPU mutation fixtures: no GPU, generated code execution, cache or receipt reuse."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from tools import dependency_report as d
from tools import id_scope_audit as audit

CONTEXT = dict(architecture='unmeasured', compiler='unmeasured', host_compiler='unmeasured',
               flags=[], abi='prototype', generation_options={}, model='fixture')
FILES = {p: {'digest': d.digest(p), 'mode': '100644'} for p in (
    'grid_codegen/algorithms/_inverse_dynamics.py', 'grid_codegen/algorithms/_other.py',
    'grid_codegen/helpers/spatial.py', 'grid_assets/robot.urdf',
    'bindings/src/wrapper.cu', 'test/oracle.py', 'test/fixture.py',
    'test/tolerances.json', 'test/manifest.txt', 'test/policy.yaml', 'docs/design.md')}


def snapshots():
    return d.make_snapshot(FILES, CONTEXT), copy.deepcopy(FILES), copy.deepcopy(CONTEXT)


@pytest.mark.parametrize('path', list(FILES))
def test_deliberate_source_mutation(path):
    before, files, context = snapshots()
    files[path]['digest'] = d.digest('edited')
    result = d.compare(before, d.make_snapshot(files, context))
    for row in result['exports'].values():
        expected_object = path.startswith(('grid_codegen/', 'bindings/', 'grid_assets/'))
        expected_evidence = not path.startswith('docs/')
        assert row['object_changed'] == expected_object
        assert row['evidence_changed'] == expected_evidence
    assert result['reuse_authorized'] is False and result['narrowing_ready'] is False


@pytest.mark.parametrize('field', sorted(d.CONTEXT_KEYS))
def test_build_configuration_model_feature_and_abi_mutation(field):
    before, files, context = snapshots()
    context[field] = 'changed'
    result = d.compare(before, d.make_snapshot(files, context))
    assert all(r['object_changed'] and r['evidence_changed'] for r in result['exports'].values())


def test_unknown_file_widens_and_missing_input_invalidates_everything():
    before, files, context = snapshots()
    files['new_driver.custom'] = {'digest': d.digest('x'), 'mode': '100644'}
    after = d.make_snapshot(files, context)
    assert after['unclassified'] == ['new_driver.custom']
    assert all(r['object_changed'] for r in d.compare(before, after)['exports'].values())
    missing = d.make_snapshot(FILES, context, missing=['unresolved include'])
    result = d.compare(before, missing)
    assert result['invalidate_broadly'] and all(r['object_changed'] for r in result['exports'].values())


def test_new_deleted_test_only_changes_evidence():
    before, files, context = snapshots()
    files.pop('test/oracle.py')
    files['test/new_test.py'] = {'digest': d.digest('new node'), 'mode': '100644'}
    result = d.compare(before, d.make_snapshot(files, context))
    assert all(not r['object_changed'] and r['evidence_changed'] for r in result['exports'].values())


def test_graph_edges_and_cached_key_fields_are_not_trusted():
    before, _, _ = snapshots()
    after = copy.deepcopy(before)
    after['proposed_object_keys'] = {'fake': 'unchanged'}
    assert not any(r['object_changed'] for r in d.compare(before, after)['exports'].values())
    after['nodes']['object:inverse_dynamics_inner']['deps'].append('missing')
    with pytest.raises(ValueError, match='unresolved'): d.compare(before, after)
    after = copy.deepcopy(before); after['schema'] = 999
    with pytest.raises(ValueError, match='schema'): d.compare(before, after)
    after = copy.deepcopy(before); after['missing'] = ['lost header']
    with pytest.raises(ValueError, match='completeness'): d.compare(before, after)


def test_cycles_are_included_and_graph_topology_affects_identity():
    nodes = {'a': {'deps': ['b']}, 'b': {'deps': ['a']}}
    assert d.closure(nodes, 'a') == ['a', 'b']
    previous = d.key(nodes, 'a')
    nodes['b']['deps'] = []
    assert d.key(nodes, 'a') != previous


@pytest.mark.parametrize('bad', [None, {}, {**CONTEXT, 'compiler': None}, {**CONTEXT, 'unknown': 'x'}])
def test_context_must_be_explicit(bad):
    with pytest.raises(ValueError): d.make_snapshot(FILES, bad)


def test_same_snapshot_never_claims_reuse():
    before, _, _ = snapshots()
    result = d.compare(before, before)
    assert not result['changed_nodes']
    assert not result['reuse_authorized'] and not result['narrowing_ready']


def test_cli_malformed_snapshot_exits_nonzero(tmp_path):
    import sys
    path = tmp_path / 'bad.json'; path.write_text('[]')
    script = Path(d.__file__)
    result = subprocess.run([sys.executable, str(script), 'compare', str(path), str(path)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'invalidate broadly' in result.stderr and not result.stdout


def test_git_snapshot_never_reads_symlink_target_and_notices_new_sources(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args])
    git('init', '-q'); git('config', 'user.name', 'CPU'); git('config', 'user.email', 'cpu@example.invalid')
    (tmp_path / 'source.py').write_text('x = 1')
    (tmp_path / 'link.py').symlink_to('/not/a/file/to/read')
    git('add', 'source.py', 'link.py'); git('commit', '-qm', 'fixture')
    (tmp_path / 'new.py').write_text('new = 1')
    result = d.snapshot(tmp_path, CONTEXT)
    assert result['complete'] is False
    assert result['missing'] == ['link.py', 'untracked:new.py']
    (tmp_path / 'source.py').chmod(0o755)
    assert d.snapshot(tmp_path, CONTEXT)['nodes']['file:source.py']['identity']['mode'] == '100755'


def test_scope_audit_reports_indirect_helpers_without_safety_claim(tmp_path):
    source = tmp_path / 'grid_codegen'; source.mkdir()
    (source / 'emitter.py').write_text('''
def gen_inverse_dynamics_inner(self):
    self.helper()
    self.gen_add_code_line("__shared__ float a[2];")
def gen_inverse_dynamics_device(self):
    self.gen_inverse_dynamics_inner()
def helper(self):
    self.gen_add_sync()
def gen_add_sync(self):
    self.gen_add_code_line("__syncthreads();")
''')
    report = audit.audit(tmp_path)
    assert not report['scope_safe']
    assert {r['emitter'] for r in report['emitters']} >= {'helper', 'gen_add_sync'}
    assert any('gen_add_code_line' in x for x in report['unresolved'])
    assert any(f['kind'] == 'block-shared-storage' for r in report['emitters'] for f in r['findings'])

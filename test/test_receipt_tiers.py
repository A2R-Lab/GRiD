"""CPU-only policy regressions. No signature verification is replaced in production."""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from test import receipt_integrity as ri


@pytest.mark.parametrize('version,wanted', [('0.7.2', [True, False, False]),
                                          ('0.8.0', [True, True, False]),
                                          ('1.0.0', [True, True, True])])
def test_tier_boundaries(version, wanted):
    assert [ri.requires_fresh(r, version) for r in ri.load_receipt_table()] == wanted


@pytest.mark.parametrize('version', ['v1.0.0', '1.0', '1.0.0rc1', '01.0.0', None])
def test_ambiguous_version_rejected(version):
    with pytest.raises(ValueError):
        ri.release_version(version)


@pytest.mark.parametrize('patch', [
    {'tier': 'retired', 'retired_in': '0.7.1'},
    {'tier': 'retired'}, {'tier': 'primary'},
    {'receipt': 'receipts/../../gpu-proof.json'},
    {'expected_skips': '../skips.txt'},
    {'waivers': [{'version': '0.8.0', 'reason': ''}]},
    {'waivers': [{'version': '0.8.0', 'reason': 'offline', 'all': True}]},
])
def test_bad_secondary_policy_rejected(patch):
    row = dict(arch=87, label='Orin', receipt='receipts/orin.json', tier='minor', **{})
    row.update(patch)
    with pytest.raises(ValueError):
        ri._row(row, primary=False, table_path='test')


def test_primary_never_waived():
    with pytest.raises(ValueError, match='primary waivers'):
        ri._row(dict(arch=120, label='GPU', receipt='gpu-proof.json',
                     waivers=[dict(version='0.8.0', reason='offline')]),
                primary=True, table_path='test')


def test_retirement_cannot_apply_retroactively():
    row = replace(ri.load_receipt_table()[1], tier='retired', retired_in='0.8.0')
    assert not ri.requires_fresh(row, '0.8.0')
    with pytest.raises(ValueError):
        ri.requires_fresh(row, '0.7.2')


def test_unknown_gpu_is_only_provisional():
    row = ri.provisional_row(89)
    assert row.tier == 'provisional' and not row.primary
    assert row.receipt != 'gpu-proof.json'
    assert row.expected_skips == 'test/gpu-proof-provisional-skips.txt'
    with pytest.raises(ValueError):
        ri.receipt_row_for_arch(89)


def test_previously_recorded_secondary_cannot_disappear(tmp_path):
    row = ri.load_receipt_table()[1]
    assert ri.verify_all([row], root=tmp_path, required_receipts={row.receipt}) == 1


def test_generated_policy_inventory_is_current():
    root = Path(__file__).resolve().parents[1]
    text = (root / 'docs/source/user_guide/getting_started/receipt_tiers.inc').read_text()
    assert text.split('\n\n', 1)[1] == ri.tier_table()


def test_historical_age_shift_preserves_carry_bound():
    policy = dict(max_age_days=30, carried_max_age_days=5, allow_dirty=False,
                  required_test_manifest='manifest', allow_carried=True)
    receipt = {'session': {'ended_at': '2025-11-14T00:00:00Z'}}
    result = ri.historical_policy(policy, receipt, datetime(2026, 11, 14, tzinfo=timezone.utc))
    assert result['max_age_days'] == 396
    assert result['carried_max_age_days'] == 371
    assert result['required_test_manifest'] == 'manifest'
    assert result['allow_dirty'] is False
    assert policy['max_age_days'] == 30


def test_recorded_tree_ancestry_and_cleanup(tmp_path):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'CPU test')
    git('config', 'user.email', 'test@example.invalid')
    (tmp_path / 'file').write_text('original')
    git('add', 'file')
    git('commit', '-qm', 'Original')
    commit = git('rev-parse', 'HEAD')
    (tmp_path / 'file').write_text('new')
    git('commit', '-qam', 'New')
    with ri.recorded_tree(tmp_path, commit) as (tree, scratch):
        assert (tree / 'file').read_text() == 'original'
        assert tree.exists()
    assert not tree.exists() and not scratch.exists()
    assert len(git('worktree', 'list').splitlines()) == 1
    with pytest.raises(ValueError):
        with ri.recorded_tree(tmp_path, '--bad'):
            pytest.fail('invalid revision was accepted')


def test_release_only_checks_required_freshness_and_exact_waivers(tmp_path, monkeypatch):
    rows = ri.load_receipt_table()
    rows[1] = replace(rows[1], waivers=(('0.8.0', 'Orin unavailable; maintainer reviewed'),))
    (tmp_path / 'gpu-proof.json').write_text(json.dumps({
        'environment': {'gpu_info': {'devices': [{'compute_capability': '12.0'}]}}}))
    history = []
    monkeypatch.setattr(ri, 'verify_all', lambda *a, **kw: history.append(True) or 0)
    calls = []
    def run(cmd, **kw):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    assert ri.verify_release('0.8.0', rows, root=tmp_path, run=run) == 0
    assert history and len(calls) == 1
    assert calls[0][calls[0].index('--policy')+1].endswith('gpu-proof-policy-release.yaml')
    assert ri.verify_release('0.9.0', rows, root=tmp_path, run=run) == 1
    monkeypatch.setattr(ri, 'verify_all', lambda *a, **kw: 1)
    assert ri.verify_release('0.7.2', rows, root=tmp_path, run=run) == 1

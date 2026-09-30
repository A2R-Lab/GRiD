"""CPU-only receipt-scope regressions: code invalidates, website wording does not."""
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from pytest_gpu_proof.fingerprint import compute_fingerprint

ROOT = Path(__file__).resolve().parents[1]
SCOPE = tomllib.loads((ROOT / 'pyproject.toml').read_text())['tool']['gpu_proof']['fingerprint_paths']


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, 'init', '-q')
    git(tmp_path, 'config', 'user.name', 'Receipt scope test')
    git(tmp_path, 'config', 'user.email', 'scope@example.invalid')
    for path in SCOPE:
        target = tmp_path / path
        if not target.suffix:
            target = target / 'input.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('original\n')
    for path in ('docs/landing/index.html', 'docs/source/guide.rst', 'docs/plot_release_figures.py'):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('original\n')
    git(tmp_path, 'add', '--', *SCOPE, 'docs')
    git(tmp_path, 'commit', '-qm', 'Initial test inputs')
    return tmp_path


def fingerprint(root):
    return compute_fingerprint(SCOPE, str(root), exclude_paths=['gpu-proof.json'])['digest']


def test_release_policy_requires_exact_recorded_scope():
    policy = yaml.safe_load((ROOT / 'test/gpu-proof-policy-release.yaml').read_text())
    assert sorted(policy['required_fingerprint_paths']) == sorted(SCOPE)
    assert policy['required_fingerprint_extra_paths'] == []
    assert policy['required_fingerprint_excluded_paths'] == ['gpu-proof.json']
    assert policy['allow_carried'] is False
    assert policy['allow_dirty'] is False
    assert not any(p.startswith('docs') or p == '.' for p in SCOPE)
    for path in SCOPE:
        assert git(ROOT, 'ls-files', '--', path), f'Empty fingerprint input: {path}'


@pytest.mark.parametrize('path', [
    'grid_codegen/input.py', 'bindings/grid_rbd/input.py', 'bindings/src/input.py',
    'test/conftest.py', 'test/cuda_equivalents/input.py', 'config/input.py',
])
def test_correctness_edit_changes_fingerprint(repo, path):
    before = fingerprint(repo)
    (repo / path).write_text('changed\n')
    assert fingerprint(repo) != before


def test_committed_docs_edit_keeps_fingerprint(repo):
    before = fingerprint(repo)
    for path in (repo / 'docs').rglob('*'):
        if path.is_file():
            path.write_text('new wording or figure label\n')
    git(repo, 'add', '--', 'docs')
    git(repo, 'commit', '-qm', 'Update presentation only')
    assert fingerprint(repo) == before


@pytest.mark.parametrize('peer', ['GLASS', 'RBDReference', 'URDFParser'])
def test_uninitialized_peer_gitlink_change_invalidates(repo, peer):
    # CPU CI does not initialize peers in the receipt-verification checkout.
    path = f'external/{peer}'
    git(repo, 'rm', '-qr', '--', path)
    old = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'update-index', '--add', '--cacheinfo', f'160000,{old},{path}')
    before = fingerprint(repo)
    git(repo, 'commit', '-qm', 'Add peer gitlink')
    new = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'update-index', '--cacheinfo', f'160000,{new},{path}')
    assert fingerprint(repo) != before

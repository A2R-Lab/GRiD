"""Fast CPU checks for both installed resources and pinned-source packaging."""
import json
from pathlib import Path

import pytest

from grid_codegen import resources
from _build_resources import source_bundle, stage_resources


def test_checkout_resource_resolution():
    assert (resources.resource_path('GLASS') / 'glass.cuh').is_file()
    assert (resources.resource_path('launch_configs') / 'iiwa14/rtx5090_sm120.json').is_file()
    assert (resources.collision_include_dir() / 'grid_collision_geometry.cuh').is_file()
    assert (resources.package_dir('URDFParser') / 'URDFParser.py').is_file()
    with pytest.raises(ValueError, match='Unknown'):
        resources.resource_path('../outside')


def test_exported_bundle_does_not_require_git(tmp_path, monkeypatch):
    import _build_resources as build
    info, files = source_bundle(build.ROOT)
    root = tmp_path / 'export'
    stage_resources(root / 'grid_codegen/_data')
    for name in info['python_sources']:
        target = root / 'external' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((build.ROOT / 'external' / name).read_bytes())
    monkeypatch.setattr(build.subprocess, 'check_output', lambda *a, **k: pytest.fail('Git must not be called'))
    actual, _ = source_bundle(root)
    assert actual == info
    monkeypatch.setattr(resources, 'data_dir', lambda: root / 'grid_codegen/_data')
    assert resources.checkout_root() is None
    assert resources.bundled_provenance()['peers']['GLASS']['commit'] == info['peers']['GLASS']['commit']
    assert resources.resource_path('GLASS').is_dir()
    source = next(iter(actual['python_sources']))
    (root / 'external' / source).write_text('tampered\n')
    with pytest.raises(RuntimeError, match='peer source hash mismatch'):
        source_bundle(root)


def test_bundle_resource_tampering_is_rejected(tmp_path):
    import _build_resources as build
    stage_resources(tmp_path / 'grid_codegen/_data')
    target = tmp_path / 'grid_codegen/_data/GLASS/glass.cuh'
    target.write_text('tampered\n')
    with pytest.raises(RuntimeError, match='resource hash mismatch'):
        source_bundle(tmp_path)


def test_installed_source_identity_is_location_independent_and_detects_edits(tmp_path, monkeypatch):
    from grid_rbd import _cache
    roots = [tmp_path / 'first-site-packages', tmp_path / 'other-site-packages']
    for root in roots:
        for package in ('grid_codegen', 'URDFParser'):
            (root / package).mkdir(parents=True)
            (root / package / '__init__.py').write_text('# same installed bytes\n')
    monkeypatch.setattr(resources, 'package_dir', lambda name: roots[0] / name)
    original = _cache._codegen_source_hash()
    monkeypatch.setattr(resources, 'package_dir', lambda name: roots[1] / name)
    assert _cache._codegen_source_hash() == original
    (roots[1] / 'URDFParser/__init__.py').write_text('# changed parser bytes\n')
    assert _cache._codegen_source_hash() != original


def test_missing_toolkit_error_explains_registration_requirement(monkeypatch):
    from grid_rbd import _compile
    monkeypatch.setattr(_compile.shutil, 'which', lambda _: None)
    with pytest.raises(RuntimeError, match=r'nvcc not found.*CUDA Toolkit.*register_robot'):
        _compile.find_nvcc()

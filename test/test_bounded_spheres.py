import xml.etree.ElementTree as ET

import numpy as np
import pytest

from grid_codegen.algorithms._spherize import spherize_urdf
from grid_codegen.algorithms._collision import validate_sphere_model, resolve_spherized_urdf
from grid_codegen.algorithms._bounded_spheres import bounded_spheres, sampled_fidelity


def test_missing_mesh_fails_without_publishing_partial_model(tmp_path):
    urdf = tmp_path / 'robot.urdf'
    urdf.write_text('<robot name="r"><link name="base"><collision><geometry><mesh filename="missing.stl"/></geometry></collision></link></robot>')
    out = tmp_path / 'out.urdf'
    with pytest.raises(ValueError, match='could not be resolved'):
        spherize_urdf(urdf, 0.05, out)
    assert not out.exists()


def test_package_mesh_lookup_uses_only_explicit_roots(tmp_path, monkeypatch):
    from grid_codegen.algorithms._spherize import _resolve_mesh_path
    package = tmp_path / 'assets' / 'demo'
    package.mkdir(parents=True)
    mesh = package / 'link.obj'
    mesh.write_text('fixture')
    monkeypatch.delenv('ROS_PACKAGE_PATH', raising=False)
    assert _resolve_mesh_path('package://demo/link.obj', str(tmp_path / 'urdfs')) is None
    for root in (package, package.parent):
        monkeypatch.setenv('ROS_PACKAGE_PATH', str(root))
        assert _resolve_mesh_path('package://demo/link.obj', str(tmp_path / 'urdfs')) == str(mesh)


@pytest.mark.parametrize('geometry', ['<box size="1 -1 1"/>', '<box size="1 1"/>',
                                    '<cylinder radius="-1" length="1"/>', '<sphere radius="nan"/>'])
def test_invalid_primitive_dimensions_fail_before_output(tmp_path, geometry):
    source, out = tmp_path / 'source.urdf', tmp_path / 'out.urdf'
    source.write_text(f'<robot><link name="base"><collision><geometry>{geometry}</geometry></collision></link></robot>')
    with pytest.raises(ValueError, match='invalid'):
        spherize_urdf(source, 0.05, out)
    assert not out.exists()


@pytest.mark.parametrize('geometry', ['<box size="1 -1 1"/>', '<cylinder radius="-1" length="1"/>'])
def test_fidelity_report_rejects_invalid_source_geometry(tmp_path, geometry):
    from grid_codegen.spherize import fidelity_report
    source, spheres = tmp_path / 'source.urdf', tmp_path / 'spheres.urdf'
    source.write_text(f'<robot><link name="base"><collision><geometry>{geometry}</geometry></collision></link></robot>')
    spheres.write_text('<robot><link name="base"><collision><geometry><sphere radius="1"/></geometry></collision></link></robot>')
    with pytest.raises(ValueError, match='invalid'):
        fidelity_report(source, spheres)


@pytest.mark.parametrize('extents', [(0.03, 0.025, 0.02), (0.002, 0.03, 0.03)])
def test_bounded_fit_is_deterministic_and_covers_thin_mesh_samples(extents):
    import trimesh
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation([0.013, -0.007, 0.011])
    spheres = bounded_spheres(mesh)
    assert spheres == bounded_spheres(mesh)
    report = sampled_fidelity(mesh, spheres, samples=300)
    assert report['uncovered_surface_fraction'] == 0
    assert report['bulge_max_m'] < 0.02  # numerical voxel diagnostic, not a global certificate
    assert report['spheres'] > 0


def test_sphere_model_requires_links_and_matching_frames(tmp_path):
    source = tmp_path / 'source.urdf'
    source.write_text('<robot name="r"><link name="base"/><link name="tip"><collision><geometry><sphere radius=".02"/></geometry></collision></link><joint name="j" type="fixed"><parent link="base"/><child link="tip"/><origin xyz="0 0 .1"/></joint></robot>')
    good = tmp_path / 'spheres.urdf'
    spherize_urdf(source, 0.05, good)
    validate_sphere_model(source, good)
    assert resolve_spherized_urdf(str(good), source) == good.resolve()
    tree = ET.parse(good)
    tree.getroot().find('joint/origin').set('xyz', '0 0 .2')
    tree.write(good)
    with pytest.raises(ValueError, match='kinematic frames'):
        validate_sphere_model(source, good)


def test_report_cli_on_existing_spheres(tmp_path, monkeypatch, capsys):
    from grid_codegen.spherize import main
    source = tmp_path / 'source.urdf'
    source.write_text('<robot name="r"><link name="base"><collision><geometry><sphere radius=".02"/></geometry></collision></link></robot>')
    monkeypatch.setattr('sys.argv', ['grid-spherize', str(source), '--report', str(source)])
    main()
    assert 'not a continuous-coverage certificate' in capsys.readouterr().out


def test_codegen_cli_sphere_path_implies_collision_and_rejects_native(tmp_path, monkeypatch):
    from grid_codegen.cli import parseInputs
    source = tmp_path / 'source.urdf'
    source.write_text('<robot name="r"/>')
    argv = ['grid-generate', str(source), '--spherized-urdf', str(source)]
    monkeypatch.setattr('sys.argv', argv)
    args = parseInputs()
    assert args.collision and args.spherized_urdf == str(source)
    monkeypatch.setattr('sys.argv', argv + ['--collision-native'])
    with pytest.raises(SystemExit):
        parseInputs()


def test_bundled_iiwa_collision_meshes_are_complete_and_pinned(tmp_path, monkeypatch):
    import hashlib
    from pathlib import Path
    from grid_assets import robot_urdf
    from grid_codegen.algorithms._spherize import _resolve_mesh_path
    from grid_codegen.algorithms._collision import parse_spherized_urdf
    monkeypatch.delenv('ROS_PACKAGE_PATH', raising=False)
    source = robot_urdf('iiwa14')
    expected = {
        'link_6.obj': 'bfdba14c8462325caec427fe73ace2bbb24e4c604305840371c7b13ed1011ab1',
        'link_7.obj': 'cafe5f56d19435c9858398856ebb7fca04028bdd801b10bc1193fe0355826b4b',
    }
    found = set()
    for mesh in ET.parse(source).getroot().findall('link/collision/geometry/mesh'):
        path = Path(_resolve_mesh_path(mesh.get('filename'), str(source.parent)))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[path.name]
        found.add(path.name)
    assert found == set(expected)
    out = tmp_path / 'complete.urdf'
    spherize_urdf(source, 0.05, out)
    validate_sphere_model(source, out)
    model = parse_spherized_urdf(out)
    for link in ET.parse(source).getroot().findall('link'):
        if link.findall('collision'):
            assert model.get(link.get('name')), link.get('name')


def test_codegen_cli_consumes_supplied_sphere_model(tmp_path, monkeypatch):
    from grid_assets import robot_urdf
    from grid_codegen.cli import main
    source = robot_urdf('iiwa14')
    spheres = tmp_path / 'spheres.urdf'
    spherize_urdf(source, 0.06, spheres)
    header = tmp_path / 'grid.cuh'
    monkeypatch.setattr('sys.argv', ['grid-generate', str(source), '--spherized-urdf',
        str(spheres), '--algorithm-list', 'kinematics', '-o', str(header)])
    main()
    text = header.read_text()
    assert 'namespace grid_collision' in text
    assert 'sphere_anchor' in text
    assert 'config_free' in text


def test_missing_optional_preset_reports_explicit_path(monkeypatch):
    from pathlib import Path
    is_file = Path.is_file
    monkeypatch.setattr(Path, 'is_file', lambda p: False if
        str(p).endswith('external/foam/assets/panda/smaller_panda_spherized.urdf') else is_file(p))
    with pytest.raises(ValueError, match='explicit path'):
        resolve_spherized_urdf('foam', 'unused-until-preset-resolves.urdf')


def _three_link_source(tmp_path):
    """base (fixed to mount) -> mount (revolute to tip) -> tip; base and mount have no movable
    ancestor, tip does. Every link carries collision geometry in the source."""
    source = tmp_path / 'source.urdf'
    source.write_text(
        '<robot name="r">'
        '<link name="base"><collision><geometry><sphere radius=".03"/></geometry></collision></link>'
        '<link name="mount"><collision><geometry><sphere radius=".02"/></geometry></collision></link>'
        '<link name="tip"><collision><geometry><sphere radius=".02"/></geometry></collision></link>'
        '<joint name="weld" type="fixed"><parent link="base"/><child link="mount"/><origin xyz="0 0 .1"/></joint>'
        '<joint name="j1" type="revolute"><parent link="mount"/><child link="tip"/><origin xyz="0 0 .1"/>'
        '<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
        '</robot>')
    return source


def test_sphere_model_tolerates_base_chain_omission_with_a_warning(tmp_path):
    """V2 (HJCD-IK): a link with no movable ancestor may be empty or absent; the validator warns
    and names it. A movable link keeps the strict rule."""
    import warnings
    source = _three_link_source(tmp_path)
    spheres = tmp_path / 'spheres.urdf'
    spherize_urdf(source, 0.05, spheres)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        validate_sphere_model(source, spheres)          # complete model: silent
    tree = ET.parse(spheres)
    root = tree.getroot()
    mount = next(l for l in root.findall('link') if l.get('name') == 'mount')
    for col in mount.findall('collision'):
        mount.remove(col)                               # mount: present, no spheres
    base = next(l for l in root.findall('link') if l.get('name') == 'base')
    root.remove(base)                                   # base: absent, with its fixed joint
    root.remove(next(j for j in root.findall('joint') if j.get('name') == 'weld'))
    tree.write(spheres)
    with pytest.warns(UserWarning) as record:
        validate_sphere_model(source, spheres)
    messages = ' '.join(str(w.message) for w in record)
    assert 'leaves out base' in messages and 'no spheres for mount' in messages
    tip = next(l for l in root.findall('link') if l.get('name') == 'tip')
    for col in tip.findall('collision'):
        tip.remove(col)                                 # tip has a movable ancestor: still an error
    tree.write(spheres)
    with pytest.raises(ValueError, match='omits collision geometry for tip'):
        validate_sphere_model(source, spheres)


def test_sphere_model_still_rejects_a_missing_movable_link(tmp_path):
    source = _three_link_source(tmp_path)
    spheres = tmp_path / 'spheres.urdf'
    spherize_urdf(source, 0.05, spheres)
    tree = ET.parse(spheres)
    root = tree.getroot()
    root.remove(next(l for l in root.findall('link') if l.get('name') == 'tip'))
    root.remove(next(j for j in root.findall('joint') if j.get('name') == 'j1'))
    tree.write(spheres)
    with pytest.raises(ValueError, match='kinematic frames'):
        validate_sphere_model(source, spheres)


def test_base_chain_omission_flows_through_the_spec_builder(tmp_path):
    """The tolerated omission reaches codegen unchanged: the spec builder skips base-chain
    spheres anyway, so a model without its base link yields the same rows as a complete one."""
    import warnings
    from URDFParser import URDFParser
    from grid_codegen.algorithms._collision import build_sphere_tiers
    source = _three_link_source(tmp_path)
    full = tmp_path / 'full.urdf'
    spherize_urdf(source, 0.05, full)
    robot = URDFParser().parse(str(source))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        reference = build_sphere_tiers(robot, {'fine': resolve_spherized_urdf(str(full), source)})['fine']
    trimmed = tmp_path / 'trimmed.urdf'
    tree = ET.parse(full)
    root = tree.getroot()
    root.remove(next(l for l in root.findall('link') if l.get('name') == 'base'))
    root.remove(next(j for j in root.findall('joint') if j.get('name') == 'weld'))
    tree.write(trimmed)
    with pytest.warns(UserWarning, match='leaves out base'):
        path = resolve_spherized_urdf(str(trimmed), source)
    tier = build_sphere_tiers(robot, {'fine': path})['fine']
    assert tier['n'] == reference['n'] > 0
    assert tier['anchor'] == reference['anchor'] and tier['offset'] == reference['offset']
    assert tier['radius'] == reference['radius']

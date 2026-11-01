"""Schema invariants and independent descriptor-only forward kinematics."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_codegen.model_descriptor import (canonical_bytes, descriptor_from_robot,
                                           descriptor_id, validate_descriptor)
from grid_rbd._compile import configuration_layout_from_robot
from grid_codegen.model_metadata_cpp import emit_model_metadata
from grid_rbd import describe_model
from test.cuda_equivalents.constant_model import constant_model
from test.cuda_equivalents.descriptor_oracle import assert_reference, descriptor_fk

ROOT = Path(__file__).resolve().parents[1]


def test_public_cpu_description_and_option_rejection(tmp_path, monkeypatch, capsys):
    robot = constant_model(tmp_path)
    capsys.readouterr()
    source = tmp_path/'model-False-False-False.urdf'
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('CPU inspection started a process'))
    assert describe_model(source) == descriptor_from_robot(robot)
    assert not capsys.readouterr().out
    with pytest.raises(ValueError, match='schema'):
        describe_model(source, schema_version=2)
    with pytest.raises(TypeError):
        describe_model(source, runtime_transform='false')
    with pytest.raises(TypeError):
        describe_model(source, arbitrary_options=True)


@pytest.mark.parametrize('fixture,floating',[
    ('mixed_spherical_arm',False),('branching_skew_arm',False),
    ('constant',False),('constant',True)])
def test_generated_wrapper_metadata_has_one_coordinate_source(tmp_path,fixture,floating):
    from types import SimpleNamespace
    from grid_rbd._compile import generate_grid_cuh
    from grid_rbd._handle import RobotHandle
    if fixture=='constant':
        constant_model(tmp_path)
        source=tmp_path/'model-False-False-False.urdf'
    else:
        source=ROOT/'external/URDFParser/tests/fixtures'/f'{fixture}.urdf'
    out=tmp_path/'grid.cuh'
    meta=generate_grid_cuh(source,dict(floating_base=floating,
        algorithm_list=['end_effector_pose'],emit_plant=False,enable_mujoco_kernels=False),out)
    d=describe_model(source,floating_base=floating)
    assert meta['model_descriptor']==d==json.loads((tmp_path/'model_descriptor.json').read_text())
    assert [meta[k] for k in ('num_joints','num_vel','num_bodies')]==[d[k] for k in ('nq','nv','nb')]
    assert meta['joint_names']==[j['name'] for j in d['joints']]
    handle=object.__new__(RobotHandle)
    handle._meta=meta
    handle._runner=SimpleNamespace(num_joints=d['nq'],num_vel=d['nv'],num_bodies=d['nb'])
    assert handle.model_descriptor==d
    handle.model_descriptor['model_name']='changed copy'
    assert handle.model_descriptor==d
    from grid_rbd._surface_common import BaseDelegateMixin
    # JAX and torch share this metadata delegate; no framework/GPU import needed.
    delegated=BaseDelegateMixin()
    delegated._base=handle
    assert delegated.model_descriptor==d
    delegated.model_descriptor['model_name']='changed delegated copy'
    assert handle.model_descriptor==d
    assert tuple(handle.configuration_layout)==tuple(tuple(b[k] for k in
        ('kind','q_start','v_start','nq','nv')) for b in d['configuration_layout'])


def test_canonical_encoding_golden():
    value = {'z': [None, True, 1, -0.0, 1.5], 'a': 'é'}
    expected = ('["map",[["a",["str","é"]],["z",["list",[["null"],["bool",true],'
                '["int","1"],["f64","0000000000000000"],["f64","3ff8000000000000"]]]]]]').encode()
    assert canonical_bytes(value) == expected
    assert descriptor_id(value) == hashlib.sha256(expected).hexdigest()
    assert canonical_bytes(-0.) == canonical_bytes(0.)
    assert canonical_bytes(1) != canonical_bytes(1.)
    assert canonical_bytes('é') != canonical_bytes('e\u0301')


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), '\ud800', {1: 'key'}, (1, 2)])
def test_canonical_encoding_rejects_bad_values(bad):
    with pytest.raises((ValueError, UnicodeError)):
        canonical_bytes(bad)


@pytest.mark.parametrize('name,floating', [
    ('branching_skew_arm', False), ('mixed_spherical_arm', False),
    ('helical_arm', False), ('rotated_inertial_arm', False),
    ('rotated_inertial_arm', True), ('constant', False), ('constant', True), ('empty', False),
])
def test_descriptor_fk_and_ownership(tmp_path, name, floating):
    if name in ('constant', 'empty'):
        constant_model(tmp_path, zero=name == 'empty')
        source = tmp_path / f'model-{name == "empty"}-False-False.urdf'
    else:
        source = ROOT/'external/URDFParser/tests/fixtures'/f'{name}.urdf'
    robot = URDFParser().parse(str(source), floating_base=floating)
    d = descriptor_from_robot(robot)
    assert validate_descriptor(json.loads(json.dumps(d))) == d
    expected = [tuple(b[k] for k in ('kind', 'q_start', 'v_start', 'nq', 'nv'))
                for b in d['configuration_layout']]
    assert tuple(expected) == configuration_layout_from_robot(robot)
    rng = np.random.default_rng(3010)
    for _ in range(12):
        q = rng.uniform(-.6, .6, d['nq'])
        for block in d['configuration_layout']:
            if block['kind'] in ('floating', 'spherical'):
                end = block['q_start']+block['nq']
                quat = rng.normal(size=4)
                q[end-4:end] = quat/np.linalg.norm(quat)
        assert_reference(d, robot, q)
    assert [j['name'] for j in d['joints']] == [j.name for j in robot.get_joints_ordered_by_id()]


def test_descriptor_hand_anchors_and_negative_controls(tmp_path):
    robot = constant_model(tmp_path)
    d = descriptor_from_robot(robot)
    q = np.array([0.])
    poses = descriptor_fk(d, q)
    constant = next(f for f in d['frames'] if f['name'] == 'constant')
    c, s = np.cos(.17), np.sin(.17)
    expected = np.array([[c, -s, 0, .3], [s, c, 0, 0], [0, 0, 1, .5], [0, 0, 0, 1.]])
    np.testing.assert_allclose(poses[constant['id']], expected, atol=1e-12, rtol=0)
    reversed_transform = copy.deepcopy(d)
    reversed_transform['joints'][0]['T_parent_joint_zero'] = np.linalg.inv(
        reversed_transform['joints'][0]['T_parent_joint_zero']).tolist()
    reversed_transform['descriptor_id'] = descriptor_id(reversed_transform)
    validate_descriptor(reversed_transform)  # Structurally valid, semantically wrong.
    with pytest.raises(AssertionError):
        assert_reference(reversed_transform, robot, q)
    bad = copy.deepcopy(d)
    bad['joints'][1]['q_start'] = 0
    bad['descriptor_id'] = descriptor_id(bad)
    with pytest.raises(ValueError, match='dependent slices'):
        validate_descriptor(bad)
    source = tmp_path/'model-False-False-False.urdf'
    floating = URDFParser().parse(str(source), floating_base=True)
    fd = descriptor_from_robot(floating)
    fq = np.r_[1., 2., 3., 0., 0., np.sqrt(.5), np.sqrt(.5), 0.]
    base = descriptor_fk(fd, fq)[1]
    np.testing.assert_allclose(base, [[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], atol=1e-12)
    swapped = fq.copy()
    swapped[3:7] = np.roll(swapped[3:7], 1)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(descriptor_fk(fd, swapped)[1], base, atol=1e-11)


@pytest.mark.parametrize('mutation', [
    lambda d: d.update(schema_version=2),
    lambda d: d.update(nq=d['nq']+1),
    lambda d: d['configuration_layout'][0].update(nv=2),
    lambda d: d['bodies'][0].update(parent_body=0),
    lambda d: d['frames'][1].update(attached_body=-1),
    lambda d: d['joints'][0].update(axis=[2., 0., 0.]),
    lambda d: d['joints'][1]['mimic'].update(multiplier=1.),
    lambda d: d['joints'][0]['T_parent_joint_zero'][0].__setitem__(0, -1.),
])
def test_rehash_does_not_bless_invalid_metadata(tmp_path, mutation):
    d = descriptor_from_robot(constant_model(tmp_path))
    mutation(d)
    d['descriptor_id'] = descriptor_id(d)
    with pytest.raises(ValueError):
        validate_descriptor(d)


def test_identity_boundaries(tmp_path):
    robot = constant_model(tmp_path)
    d = descriptor_from_robot(robot)
    path = tmp_path/'model-False-False-False.urdf'
    raw = path.read_text()
    path.write_text(raw.replace('mass value="1"', 'mass value="1.7"'))
    changed = descriptor_from_robot(URDFParser().parse(str(path)))
    assert changed == d  # Descriptor is NOT a complete physical-model identity.
    path.write_text(raw.replace('xyz=".2 0 .3"', 'xyz=".2 .1 .3"'))
    assert descriptor_from_robot(URDFParser().parse(str(path)))['descriptor_id'] != d['descriptor_id']
    assert descriptor_from_robot(robot, runtime_transform=True)['descriptor_id'] != d['descriptor_id']
    d['model_name'] += '-changed'
    with pytest.raises(ValueError, match='digest'):
        validate_descriptor(d)


def test_cpp_host_metadata_roundtrip_and_constexpr(tmp_path):
    d = descriptor_from_robot(constant_model(tmp_path))
    d['joints'][0]['name'] = 'quoted"\\é'
    d['frames'][1]['name'] = d['joints'][0]['name']
    d['descriptor_id'] = descriptor_id(d)
    (tmp_path/'metadata.hpp').write_text(emit_model_metadata(d))
    source = tmp_path/'probe.cpp'
    source.write_text('''
#define __host__
#define __device__
#include "metadata.hpp"
#include <cstdio>
constexpr bool check() {
    grid::model::joint_record j{};
    return grid::model::joint(0,j) && j.q_start==0 && j.axis[1]==1.0;
}
static_assert(check());
int main(){
    puts(grid::model::descriptor_json);
    puts(grid::model::joint_name(0));
    for(int i=0;i<grid::model::frame_count;++i){
        grid::model::frame_record f{};
        if(!grid::model::frame(i,f)) return 1;
        printf("%d %d %d", f.id,f.attached_body,f.joint_id);
        for(double x:f.T_body_frame) printf(" %.17g",x);
        puts("");
    }
    grid::model::joint_record j{};
    grid::model::frame_record f{};
    return grid::model::joint(-1,j) || grid::model::joint(grid::model::nj,j)
        || grid::model::frame(-1,f) || grid::model::frame(grid::model::frame_count,f)
        || grid::model::joint_name(-1) != nullptr;
}
''')
    exe = tmp_path/'probe'
    built = subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            str(source), '-o', str(exe)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(exe)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    lines = ran.stdout.splitlines()
    assert json.loads(lines[0]) == d
    assert lines[1] == d['joints'][0]['name']
    for line, f in zip(lines[2:], d['frames'], strict=True):
        expected = [f['id'], f['attached_body'], -1 if f['joint_id'] is None else f['joint_id']]
        expected += np.asarray(f['T_body_frame']).ravel(order='F').tolist()
        np.testing.assert_array_equal(np.fromstring(line, sep=' '), expected)


@pytest.mark.parametrize('path', sorted((ROOT/'grid_assets/robots').rglob('*.urdf')),
                         ids=lambda p: str(p.relative_to(ROOT/'grid_assets/robots')))
def test_bundled_robot_descriptor_wrapper_projection(path):
    robot = URDFParser().parse(str(path))
    d = descriptor_from_robot(robot)
    assert (d['nq'], d['nv'], d['nb']) == (robot.get_num_pos(), robot.get_num_vel(), robot.get_num_bodies())
    blocks = tuple(tuple(b[k] for k in ('kind', 'q_start', 'v_start', 'nq', 'nv'))
                   for b in d['configuration_layout'])
    assert blocks == configuration_layout_from_robot(robot)

"""Public metadata device access and single-pose frame composition, no timing."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from URDFParser import URDFParser
from grid_codegen.model_descriptor import descriptor_from_robot
from grid_codegen.model_metadata_cpp import emit_model_metadata, JOINT_KINDS, FRAME_KINDS
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.test_cuda_constant_derivatives import constant_model

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module', params=['constant', 'floating', 'spherical', 'branch', 'empty', 'runtime'])
def case(request, tmp_path_factory):
    path=tmp_path_factory.mktemp('descriptor_'+request.param)
    if request.param in ('spherical', 'branch'):
        name='mixed_spherical_arm' if request.param=='spherical' else 'branching_skew_arm'
        robot=URDFParser().parse(str(ROOT/'external/URDFParser/tests/fixtures'/f'{name}.urdf'))
    else:
        robot=constant_model(path, zero=request.param=='empty')
        if request.param=='floating':
            robot=URDFParser().parse(str(path/'model-False-False-False.urdf'),floating_base=True)
    descriptor=descriptor_from_robot(robot, runtime_transform=request.param=='runtime')
    (path/'metadata.cuh').write_text(emit_model_metadata(descriptor))
    exe=path/'probe'
    nvcc=shutil.which('nvcc')
    if not nvcc:
        pytest.skip('nvcc not found')
    command=[nvcc,'-std=c++17','-O2','-arch=sm_'+_detect_cuda_arch(),'-I'+str(path),
             str(Path(__file__).with_name('cuda_model_descriptor_runner.cu')),'-o',str(exe)]
    result=subprocess.run(command,capture_output=True,text=True)
    (path/'compile.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    return descriptor,exe


def expected(d):
    values=[]
    for j in d['joints']:
        m=j['mimic']
        pos=j['limits']['position'] or [None,None]
        vel,effort=j['limits']['velocity'],j['limits']['effort']
        values.extend([j['id'],j['parent_body'],j['child_body'],
            -1 if j['q_start'] is None else j['q_start'],
            -1 if j['v_start'] is None else j['v_start'],j['local_nq'],j['local_nv'],
            JOINT_KINDS.index(j['kind']),j['owns_coordinates'],j['axis'] is not None,
            j['pitch'] is not None,m is not None,-1 if not m or m['leader_joint'] is None else m['leader_joint'],
            m['multiplier'] if m else 1.,m['offset'] if m else 0.])
        values.extend(j['axis'] or [0.,0.,0.])
        values.append(j['pitch'] or 0.)
        values.extend(np.asarray(j['T_parent_joint_zero']).ravel(order='F'))
        values.extend([pos[0] is not None,pos[1] is not None,vel is not None,effort is not None,
                       pos[0] or 0.,pos[1] or 0.,vel or 0.,effort or 0.,0,71])
    body=np.array([[0.,-1,0,.3],[1,0,0,-.4],[0,0,1,.2],[0,0,0,1]])
    for f in d['frames']:
        values.extend([f['id'],f['attached_body'],-1 if f['joint_id'] is None else f['joint_id'],
                       FRAME_KINDS.index(f['kind'])])
        values.extend(np.asarray(f['T_body_frame']).ravel(order='F'))
        values.append(not d['runtime_transform'])
        values.extend([91]*16 if d['runtime_transform'] else (body@f['T_body_frame']).ravel(order='F'))
        values.append(0)
    return values


@pytest.mark.parametrize('precision',[0,1],ids=['fp32','fp64'])
def test_device_metadata_and_single_pose_composition(case,precision):
    descriptor,exe=case
    oracle=np.array(expected(descriptor),dtype=np.float64 if precision else np.float32)
    baseline=None
    for threads in (1,17,32,64,256):
        result=subprocess.run([str(exe),str(threads),str(precision)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        actual=np.fromstring(result.stdout,sep='\n')
        np.testing.assert_allclose(actual,oracle,atol=2e-7 if not precision else 2e-15,rtol=0)
        if baseline is not None:
            np.testing.assert_array_equal(actual,baseline)
        baseline=actual

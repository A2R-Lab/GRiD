"""Thread-scope transforms and compact/dense manifold differences."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_codegen import GRiDCodeGenerator
from grid_codegen.model_descriptor import descriptor_from_robot
from test.test_model_descriptor import descriptor_fk
from test.test_manifold_conventions import derivative, state
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.test_cuda_constant_derivatives import constant_model
from grid_assets import robot_urdf

pytestmark=[pytest.mark.cuda_equivalence,pytest.mark.robot_smoke]
ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('name,floating,runtime',[
    ('mixed_spherical_arm',False,False), ('rotated_inertial_arm',True,False),
    ('branching_skew_arm',False,False), ('helical_arm',False,False),
    ('rotated_inertial_arm',False,True), ('constant',False,False),
    ('zero',False,False), ('slider_mimic_fixed_tool',False,False),
    ('fr3',False,False)])
@pytest.mark.parametrize('precision',['float','double'])
def test_model_primitives(tmp_path,name,floating,runtime,precision):
    if name in ('constant','zero'):
        robot=constant_model(tmp_path,zero=name=='zero')
    else:
        path = robot_urdf(name) if name in ('slider_mimic_fixed_tool','fr3') else ROOT/'external/URDFParser/tests/fixtures'/f'{name}.urdf'
        robot=URDFParser().parse(str(path),floating_base=floating)
    ref=RBDReference(robot);d=descriptor_from_robot(robot,runtime_transform=runtime)
    gen=GRiDCodeGenerator(robot,dtype=precision)
    gen.gen_all_code(algorithm_list=['integrator'],emit_plant=False,runtime_transform=runtime,
        include_homogenous_transforms=True,enable_mujoco_kernels=False,output_path=str(tmp_path/'grid.cuh'))
    command=[shutil.which('nvcc'),'-std=c++17','-O2','-lineinfo','-Xptxas=-v',
        '-arch=sm_'+_detect_cuda_arch(),'-DTEST_SCALAR='+precision,'-I'+str(tmp_path),
        str(Path(__file__).with_name('cuda_model_primitives_runner.cu')),'-o',str(tmp_path/'probe')]
    built=subprocess.run(command,capture_output=True,text=True)
    (tmp_path/'compile.log').write_text(built.stdout+built.stderr)
    assert built.returncode==0,built.stderr
    dtype=np.float32 if precision=='float' else np.float64
    inputs=[];expected=[];N=d['nv'];Q=d['nq']
    for s in range(35):
        a,w=state(ref,430+s)
        if s in (0,1): w*=0 if s==0 else 1e-7
        if s in (3,4):
            for _,_,iv in ref._joint_retract_specs():
                axis=np.array([1.,-2.,3.])/np.sqrt(14.)
                w[iv[-3:]]=axis*(np.pi+(-1 if s==3 else 1)*.02)
        b=ref.integrate(a,w)
        if s%2:
            for _,iq,_ in ref._joint_retract_specs(): b[iq[-4:]]*=-1
        a,b=a.astype(dtype),b.astype(dtype);inputs.append(np.r_[a,b])
        a,b=a.astype(float),b.astype(float)
        out=[ref.difference(a,b),ref.dDifference(a,b,'from').ravel(order='F'),ref.dDifference(a,b,'to').ravel(order='F')]
        if s==2:
            for arg in ('from','to'):
                fd=derivative(lambda e: ref.difference(ref.integrate(a,e),b) if arg=='from' else ref.difference(a,ref.integrate(b,e)),N)
                # FD assumes unit input; float-rounding perturbs norms slightly.
                if precision=='double': np.testing.assert_allclose(fd,ref.dDifference(a,b,arg),atol=2e-8,rtol=2e-8)
        frames=descriptor_fk(d,a)
        worlds={f['joint_id']:pose for f,pose in zip(d['frames'],frames) if f['kind']=='joint'}
        for j in d['joints']:
            local=np.linalg.solve(worlds[j['parent_body']],worlds[j['id']]) if j['parent_body']!=-1 else worlds[j['id']]
            out.append(np.full(16,789.) if runtime else local.ravel(order='F'))
            admitted=j['owns_coordinates'] and j['kind'] in ('revolute','continuous','prismatic') and not runtime
            out.append(local.ravel(order='F') if admitted else np.full(16,789.))
        expected.append(np.r_[789.,np.concatenate(out),789.])
    np.asarray(inputs,dtype=dtype).tofile(tmp_path/'input.bin')
    baseline=None
    for threads in (1,17,32,64,128):
        output=tmp_path/f'out-{threads}.bin'
        ran=subprocess.run([str(tmp_path/'probe'),str(tmp_path/'input.bin'),str(output),str(threads)],capture_output=True,text=True)
        (tmp_path/f'run-{threads}.log').write_text(ran.stdout+ran.stderr)
        assert ran.returncode==0,ran.stdout+ran.stderr
        actual=np.fromfile(output,dtype=dtype).reshape(np.shape(expected))
        tol=2e-5 if precision=='float' else 1e-10
        np.testing.assert_allclose(actual,expected,atol=tol,rtol=tol)
        if baseline is not None: np.testing.assert_array_equal(actual,baseline)
        baseline=actual

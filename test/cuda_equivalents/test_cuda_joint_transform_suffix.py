"""Scalar transforms in a composed divergent suffix, not only in isolation."""
from pathlib import Path
import subprocess
import numpy as np
import pytest
from URDFParser import URDFParser
from grid_codegen import GRiDCodeGenerator
from grid_codegen.model_descriptor import descriptor_from_robot
from test.cuda_equivalents.descriptor_oracle import descriptor_fk
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


@pytest.mark.parametrize('n,general', [(7,False),(7,True),(32,True)])
@pytest.mark.parametrize('precision', ['float', pytest.param('double', marks=pytest.mark.fp64_alpha)])
def test_scalar_transform_composed_suffix(tmp_path,n,general,precision):
    inertial='<inertial><mass value="1"/><inertia ixx=".3" iyy=".4" izz=".5" ixy="0" ixz="0" iyz="0"/></inertial>'
    xml='<robot name="suffix"><link name="base"/>'+''.join(f'<link name="b{i}">{inertial}</link>' for i in range(n))+'<link name="tool"/>'
    for i in range(n):
        axis='0 0 1' if not general else ['0 1 0','0 0 -1','1 0 0'][i%3]
        kind='prismatic' if general and i%5==2 else 'revolute'
        parent='base' if i==0 else f'b{i-1}'
        xml+=f'<joint name="j{i}" type="{kind}"><parent link="{parent}"/><child link="b{i}"/><origin xyz=".02 .01 .08" rpy="{.1 if general else 0} 0 .03"/><axis xyz="{axis}"/><limit lower="-3" upper="3" effort="10" velocity="10"/></joint>'
    xml+=f'<joint name="tip" type="fixed"><parent link="b{n-1}"/><child link="tool"/><origin xyz=".01 -.02 .1"/></joint></robot>'
    urdf=tmp_path/'robot.urdf';urdf.write_text(xml)
    robot=URDFParser().parse(str(urdf));d=descriptor_from_robot(robot)
    GRiDCodeGenerator(robot).gen_all_code(output_path=str(tmp_path/'grid.cuh'),
        codegen_profile='kinematics',emit_plant=False,include_homogenous_transforms=True,
        enable_mujoco_kernels=False)
    frame=next(f['id'] for f in d['frames'] if f['name']=='tip' and f['kind']=='fixed_joint')
    inputs=[];expected=[]
    for warp in range(2):
        q=np.linspace(-.15,.2,n);q[warp]+=.1
        poses=descriptor_fk(d,q)
        world={f['joint_id']:p for f,p in zip(d['frames'],poses) if f['kind']=='joint'}
        local=[world[j] if j==0 else np.linalg.solve(world[j-1],world[j]) for j in range(n)]
        inputs.extend(p.ravel(order='F') for p in [*(world[j] for j in range(n)),*local])
        for j in range(32):
            if j>=n:expected.append(np.full(16,789.));continue
            changed=q.copy();changed[j]+=.025*(j+1)
            expected.append(descriptor_fk(d,changed)[frame].ravel(order='F'))
    dtype=np.float32 if precision=='float' else np.float64
    np.asarray(inputs,dtype).tofile(tmp_path/'input.bin')
    for legacy in (True,False):
        name='legacy' if legacy else 'probe'
        flags=['-O3','-std=c++17','-lineinfo','-Xptxas=-v','-arch=sm_'+_detect_cuda_arch(),
            '-DTEST_SCALAR='+precision,'-DTEST_FRAME='+str(frame)]+(['-DTEST_LEGACY'] if legacy else [])
        exe,_=cached_nvcc_executable([Path(__file__).with_name('cuda_joint_transform_suffix_runner.cu')],flags,
            exe_name=name,fallback_dir=tmp_path,include_dirs=[tmp_path],what='joint transform suffix runner')
        run=subprocess.run([str(exe),str(tmp_path/'input.bin'),str(tmp_path/(name+'.bin'))],capture_output=True,text=True)
        (tmp_path/(name+'-run.log')).write_text(run.stdout+run.stderr)
        assert run.returncode==0,run.stderr
        actual=np.fromfile(tmp_path/(name+'.bin'),dtype).reshape(2,64,16)
        tol=2e-5 if precision=='float' else 2e-12
        np.testing.assert_allclose(actual[0],expected,atol=tol,rtol=tol)
        np.testing.assert_array_equal(actual[0],actual[1])

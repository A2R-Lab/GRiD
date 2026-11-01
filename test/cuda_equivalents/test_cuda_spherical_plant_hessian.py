"""Repair audit of already-emitted fixed-spherical plant second derivatives."""
from pathlib import Path
import subprocess

import numpy as np
import pytest
from grid_codegen import GRiDCodeGenerator
from URDFParser import URDFParser
from RBDReference import RBDReference
import xml.etree.ElementTree as ET
from test.cuda_equivalents.manifold_oracle import reference, state, derivative
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable

pytestmark=[pytest.mark.cuda_equivalence,pytest.mark.robot_smoke]


@pytest.mark.parametrize('name',['spherical_arm','mixed_spherical_arm','single_spherical','two_spherical'])
@pytest.mark.parametrize('precision',['float',pytest.param('double', marks=pytest.mark.fp64_alpha)])
@pytest.mark.parametrize('scheme',['euler','semi_implicit_euler'])
@pytest.mark.parametrize('tier',[0,2],ids=['shared','minimal'])
def test_spherical_plant_hessian(tmp_path,name,precision,scheme,tier):
    if name in ('single_spherical','two_spherical'):
        source=Path(__file__).resolve().parents[2]/'external/URDFParser/tests/fixtures/spherical_arm.urdf'
        tree=ET.parse(source);root=tree.getroot()
        if name=='single_spherical':
            root.remove(root.find("joint[@name='joint_2']"))
            root.remove(root.find("link[@name='link2']"))
        else:
            joint=root.find("joint[@name='joint_2']")
            joint.set('type','spherical')
            axis=joint.find('axis')
            if axis is not None:joint.remove(axis)
        tree.write(tmp_path/'single.urdf')
        ref=RBDReference(URDFParser().parse(str(tmp_path/'single.urdf')))
    else:
        ref=reference(name)
    robot=ref.robot; nv=robot.get_num_vel()
    gen=GRiDCodeGenerator(robot,dtype=precision)
    gen.gen_all_code(algorithm_list=['fdsva_so','integrator_gradient'],emit_plant=True,
        enable_mujoco_kernels=False,output_path=str(tmp_path/'grid.cuh'))
    exe,_=cached_nvcc_executable([Path(__file__).with_name('cuda_spherical_plant_hessian_runner.cu')],
        ['-std=c++17','-O2','-lineinfo','-Xptxas=-v','-arch=sm_'+_detect_cuda_arch(),'-DTEST_SCALAR='+precision,
         '-DTEST_INTEGRATOR='+scheme.upper(),'-DTEST_TIER='+str(tier)],
        exe_name='probe',fallback_dir=tmp_path,include_dirs=[tmp_path],what='spherical plant Hessian runner')
    dtype=np.float32 if precision=='float' else np.float64
    q,v=state(ref);u=np.linspace(-.4,.7,nv)
    z=np.r_[q,v,u].astype(dtype);z.tofile(tmp_path/'input.bin')
    nq=len(q); q,v,u=z[:nq].astype(float),z[nq:nq+nv].astype(float),z[nq+nv:].astype(float)
    dt=float(dtype(.02))
    oracle=ref.plant_step_hessian(q,v,u,dt,integrator_type=scheme)
    fd=derivative(lambda d: ref.plant_step_gradient(ref.integrate(q,d[:nv]),
        v+d[nv:2*nv],u+d[2*nv:],dt,integrator_type=scheme),3*nv).transpose(0,2,1)
    np.testing.assert_allclose(oracle,fd,atol=2e-7,rtol=2e-7)
    baseline=None
    # Test the stock shared-tier admission, plus the minimal tier's larger
    # supported blocks. An inadmissible request must refuse before launch.
    for threads in (1,17,32,64,128,256):
        prefix=tmp_path/f'out-{threads}-'
        ran=subprocess.run([str(exe),str(tmp_path/'input.bin'),str(prefix),str(threads)],capture_output=True,text=True)
        (tmp_path/f'run-{threads}.log').write_text(ran.stdout+ran.stderr)
        if ran.returncode==7:
            assert tier==0 and ran.stdout.startswith('ADMISSION_LIMIT=')
            # The n=3 single-spherical stock shared kernel admits one warp;
            # larger tested models admit two. Minimal tier exercises all sizes.
            assert 32<=int(ran.stdout.split('=')[1])<threads
            continue
        assert ran.returncode==0,ran.stdout+ran.stderr
        for repeat in (0,1):
            actual=np.fromfile(str(prefix)+str(repeat),dtype=dtype).reshape(oracle.shape)
            tol=3e-4 if precision=='float' else 1e-9
            np.testing.assert_allclose(actual,oracle,atol=tol,rtol=tol)
            if baseline is not None: np.testing.assert_array_equal(actual,baseline)
            baseline=actual

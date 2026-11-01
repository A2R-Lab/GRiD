"""Bounded public binder acceptance, not timing or downstream acceptance."""
from pathlib import Path
import subprocess

import numpy as np
import pytest
from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


@pytest.fixture(scope='module', params=[('acrobot','float',False,'euler'),
    pytest.param(('acrobot','double',False,'euler'),marks=pytest.mark.fp64_alpha),
    ('iiwa14','float',False,'euler'),
    pytest.param(('iiwa14','double',False,'euler'),marks=pytest.mark.fp64_alpha),
    ('acrobot','float',True,'semi_implicit_euler'),
    pytest.param(('acrobot','double',True,'semi_implicit_euler'),marks=pytest.mark.fp64_alpha),
    ('branching','double',True,'euler')],
    ids=['acrobot-float','acrobot-double','iiwa14-float','iiwa14-double','acrobot-semi-runtime-float',
         'acrobot-semi-runtime-double','branching-runtime-double'])
def workspace_case(request,tmp_path_factory):
    name,scalar,runtime,integrator=request.param
    path=tmp_path_factory.mktemp('workspace_'+name+'_'+scalar)
    if name=='branching':
        urdf=Path(__file__).resolve().parents[2]/'external/URDFParser/tests/fixtures/branching_skew_arm.urdf'
    else:
        urdf = Path(__file__).with_name('consumer_acrobot.urdf') if name=='acrobot' else robot_urdf(name)
    robot=URDFParser().parse(str(urdf))
    gen=GRiDCodeGenerator(robot,dtype=scalar)
    gen.gen_all_code(algorithm_list=['integrator_hessian','integrator_gradient'],emit_plant=True,
                     enable_mujoco_kernels=False,fixed_target_name='iiwa_joint_ee' if name=='iiwa14' else '',
                     runtime_transform=runtime,output_path=str(path/'grid.cuh'))
    exe,_=cached_nvcc_executable([Path(__file__).with_name('cuda_plant_workspace_runner.cu')],
        ['-std=c++17','-O3','-lineinfo','-arch=sm_'+_detect_cuda_arch(),'-Xptxas=-v',
         '-DTEST_INTEGRATOR='+integrator.upper(),'-DTEST_RUNTIME_TRANSFORM='+str(int(runtime)),
         '-DTEST_CROSS_KERNEL_ROUNDOFF='+str(int(name=='branching' and scalar=='double' and runtime)),
         '-DTEST_SCALAR='+scalar],
        exe_name='probe',fallback_dir=path,include_dirs=[path],what='plant workspace runner')
    n=robot.get_num_vel()
    dtype=np.float32 if scalar=='float' else np.float64
    rng=np.random.default_rng(29103)
    z=np.stack([np.r_[rng.uniform(-.6,.6,n),rng.uniform(-.4,.4,n),rng.uniform(-2,2,n)] for _ in range(2)]).astype(dtype)
    z.tofile(path/'input.bin')
    ref=RBDReference(robot)
    expected=[]
    for row in z.astype(float):
        q,v,u=row[:n],row[n:2*n],row[2*n:]
        dt=float(dtype(.01))
        expected.append(np.r_[ref.plant_step_hessian(q,v,u,dt,integrator).ravel(),
                              ref.plant_step_gradient(q,v,u,dt,integrator).ravel(order='F'),
                              ref.plant_step(q,v,u,dt,integrator)])
    np.savez(path/'oracle.npz', inputs=z, expected=np.asarray(expected))
    return path,exe,dtype,np.asarray(expected),robot,z,integrator


@pytest.mark.parametrize('spill',[False,True],ids=['full','spill'])
def test_composed_workspace(workspace_case,spill):
    path,exe,dtype,expected,robot,z,integrator=workspace_case
    n=robot.get_num_vel();nh=18*n**3;nj=6*n*n;nx=2*n
    # Full FP64 is intentionally beyond the physical limit: test its admission
    # separately, never launch it as a positive row. FP32 exercises both shapes.
    if n==7 and dtype==np.float64 and not spill:
        result=subprocess.run([str(exe),str(path/'input.bin'),str(path/'not-written.bin'),'17','0'],capture_output=True,text=True)
        assert result.returncode==8, result.stdout+result.stderr
        return
    baseline=None
    for threads in (1,17,32,64,256):
        output=path/f'out-{spill}-{threads}.bin'
        result=subprocess.run([str(exe),str(path/'input.bin'),str(output),str(threads),str(int(spill))],capture_output=True,text=True)
        (path/f'run-{spill}-{threads}.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stdout+result.stderr+f' exit={result.returncode}'
        actual=np.fromfile(output,dtype=dtype).reshape(expected.shape)
        if path.name.startswith('workspace_branching_'):
            # The approved roundoff allowance is between separately compiled
            # stock/composed kernels, not between our storage layouts.
            other=path/f'cross-storage-{spill}-{threads}.bin'
            compared=subprocess.run([str(exe),str(path/'input.bin'),str(other),str(threads),str(int(not spill))],capture_output=True,text=True)
            (path/f'cross-storage-{spill}-{threads}.log').write_text(compared.stdout+compared.stderr)
            assert compared.returncode==0,compared.stdout+compared.stderr
            np.testing.assert_array_equal(actual,np.fromfile(other,dtype=dtype).reshape(expected.shape))
        (path/f'difference-{spill}-{threads}.txt').write_text('\n'.join(
            f'{tuple(i)}: {actual[tuple(i)]:.17g} vs {expected[tuple(i)]:.17g}'
            for i in np.argwhere(abs(actual-expected)>(2e-5+2e-3*abs(expected)))[:50]))
        if dtype==np.float64:
            np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-9)
        else:
            # Preserve the AGREED consumer policy, pinned DDP a89455b8
            # test_grid_adapter.py: step/J entrywise, Hessian tensor-norm error.
            # The extra entrywise diagnostic above is deliberately retained:
            # iiwa near-zero cancellation exceeds it in the stock kernel too.
            np.testing.assert_allclose(actual[:,nh+nj:],expected[:,nh+nj:],rtol=3e-5,atol=3e-6)
            np.testing.assert_allclose(actual[:,nh:nh+nj],expected[:,nh:nh+nj],rtol=3e-4,atol=3e-6)
        for b in range(2):
            h=actual[b,:nh].reshape(nx,3*n,3*n)
            href=expected[b,:nh].reshape(h.shape)
            budget=(3e-4*max(1.,np.max(abs(href)))) if dtype==np.float32 else 1e-9
            assert np.max(abs(h-href)) < budget
            np.testing.assert_allclose(h,h.swapaxes(1,2),rtol=1e-4 if dtype==np.float32 else 1e-9,
                                       atol=1e-5 if dtype==np.float32 else 1e-9)
            np.testing.assert_array_equal(h[:,nx:,nx:],0)
            if threads==1:
                direction=np.random.default_rng(992+b).normal(size=3*n);direction/=np.linalg.norm(direction)
                ref=RBDReference(robot);errors=[]
                for eps in (1e-3,1e-4,1e-5):
                    plus=z[b].astype(float)+eps*direction;minus=z[b].astype(float)-eps*direction
                    def jac(v):return ref.plant_step_gradient(v[:n],v[n:nx],v[nx:],float(dtype(.01)),integrator)
                    fd=(jac(plus)-jac(minus))/(2*eps)
                    errors.append(np.max(abs(fd-np.einsum('oab,a->ob',h,direction))))
                # The finite difference has a truncation/cancellation floor;
                # analytic FP64 agreement above remains 1e-9 entrywise.
                fd_budget=budget if dtype==np.float32 else 1e-7*max(1.,np.max(abs(href)))
                assert min(errors)<fd_budget,errors
        if baseline is not None: np.testing.assert_array_equal(actual,baseline)
        baseline=actual

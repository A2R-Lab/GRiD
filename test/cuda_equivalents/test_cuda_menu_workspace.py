"""Menu dimensions, alignment, poison and repeated B=2 composition."""
from pathlib import Path
import subprocess
import numpy as np
import pytest
from URDFParser import URDFParser
from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable

pytestmark=[pytest.mark.cuda_equivalence,pytest.mark.robot_smoke]


@pytest.fixture(scope='module',params=[('iiwa14',False,'float'),('go2',True,'float'),
    pytest.param(('iiwa14',False,'double'),marks=pytest.mark.fp64_alpha),('indy7',False,'float')],
    ids=['iiwa14-float','go2-float','iiwa14-double','indy7-float'])
def menu_case(request,tmp_path_factory):
    name,floating,scalar=request.param
    path=tmp_path_factory.mktemp('menu_workspace_'+name+'_'+scalar)
    urdf=Path(__file__).with_name('consumer_indy7.urdf') if name=='indy7' else robot_urdf(name)
    robot=URDFParser().parse(str(urdf),floating_base=floating)
    gen=GRiDCodeGenerator(robot,dtype=scalar)
    gen.gen_all_code(algorithm_list=['integrator','frame_jacobian'],emit_plant=True,
                     enable_mujoco_kernels=False,output_path=str(path/'grid.cuh'))
    exe,_=cached_nvcc_executable([Path(__file__).with_name('cuda_menu_workspace_runner.cu')],
        ['-std=c++17','-O2','-lineinfo','-arch=sm_'+_detect_cuda_arch(),'-Xptxas=-v','-DTEST_SCALAR='+scalar],
        exe_name='probe',fallback_dir=path,include_dirs=[path],what='menu workspace runner')
    return path,exe,robot,np.float32 if scalar=='float' else np.float64


@pytest.mark.parametrize('kind',[0,1,2,3],ids=['zero','default','wide-x','wide-u'])
def test_menu_workspace(menu_case,kind):
    path,exe,robot,dtype=menu_case
    nv,nq=robot.get_num_vel(),robot.get_num_pos();fb=robot.floating_base
    nx=nq+nv;tn=2*nv;nu=nv-6 if fb else nv;nlx=tn-6 if fb else nx
    mx,mu=[(0,0),(nlx,nu),(4097,0),(0,4097)][kind]
    rng=np.random.default_rng(7203)
    ax=rng.uniform(-.03,.03,(mx,nlx)).astype(dtype);au=rng.uniform(-.03,.03,(mu,nu)).astype(dtype)
    bx=rng.uniform(-.1,.1,mx).astype(dtype);bu=rng.uniform(-.1,.1,mu).astype(dtype)
    tx=np.zeros(mx,dtype);tu=np.zeros(mu,dtype)
    wx=rng.uniform(.1,1,mx).astype(dtype);wu=rng.uniform(.1,1,mu).astype(dtype)
    inp=path/f'input-{kind}.bin';np.concatenate([ax.ravel(order='F'),bx,tx,wx,au.ravel(order='F'),bu,tu,wu]).tofile(inp)
    expected=[]
    for b in range(2):
        x=(.01*(np.arange(nx)+1)+.03*b).astype(dtype);u=(-.02*(np.arange(nu)+1)+.04*b).astype(dtype)
        if fb:x[3:7]=[0,0,0,1]
        lx=x[7:] if fb else x
        rx=ax.astype(float)@lx+bx;ru=au.astype(float)@u+bu
        gx=np.zeros(tn);hx=np.zeros((tn,tn));offset=6 if fb else 0
        gx[offset:]=ax.T@(wx*rx);hx[offset:,offset:]=ax.T@(wx[:,None]*ax)
        gu=au.T@(wu*ru);hu=au.T@(wu[:,None]*au)
        expected.append(2*np.r_[.5*(np.dot(wx,rx*rx)+np.dot(wu,ru*ru)),gx,gu,hx.ravel(order='F'),hu.ravel(order='F')])
    expected=np.asarray(expected);baseline=None
    for threads in (1,17,32,64,256):
        output=path/f'out-{kind}-{threads}.bin'
        result=subprocess.run([str(exe),str(inp),str(output),str(threads),str(kind)],capture_output=True,text=True)
        (path/f'run-{kind}-{threads}.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stdout+result.stderr+f' exit={result.returncode}'
        actual=np.fromfile(output,dtype).reshape(expected.shape)
        np.testing.assert_allclose(actual,expected,rtol=1e-4 if dtype==np.float32 else 2e-12,
                                   atol=1e-6 if dtype==np.float32 else 2e-12)
        if baseline is not None:np.testing.assert_array_equal(actual,baseline)
        baseline=actual

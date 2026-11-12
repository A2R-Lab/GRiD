"""Adjacent-matrix hazard audit: both shared world-FK emitters and EE Hessian.

Use actual generated inners, poisoned exact-capacity pools at every alignment
residue, independent oracles and optimized strict sanitizer runs. This covers
serial, branched/floating and mimic topologies, not just the original pose case.
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


@pytest.mark.parametrize('name,floating', [('iiwa14',False),('go2',True),('fr3',False)])
@pytest.mark.parametrize('scalar', ['float',pytest.param('double',marks=pytest.mark.fp64_alpha)])
def test_adjacent_transform_scratch(tmp_path,name,floating,scalar):
    robot=URDFParser().parse(str(robot_urdf(name)),floating_base=floating)
    gen=GRiDCodeGenerator(robot,dtype=scalar)
    gen.gen_all_code(algorithm_list=['end_effector_pose_gradient','end_effector_pose_hessian',
                                    'frame_jacobian','potential_energy_regressor'],
                     enable_mujoco_kernels=False,output_path=str(tmp_path/'grid.cuh'))
    target=robot.get_leaf_nodes()[0]
    exe,_=cached_nvcc_executable([Path(__file__).with_name('cuda_transform_scratch_runner.cu')],
        ['-std=c++17','-O2','-lineinfo','-arch=sm_'+_detect_cuda_arch(),'-DTEST_SCALAR='+scalar,
         f'-DHESS_SCRATCH={gen.gen_end_effector_pose_hessian_inner_temp_mem_size()}',
         f'-DFRAME_SCRATCH={16*robot.get_num_joints()}',f'-DTARGET_JID={target}'],
        exe_name='probe',fallback_dir=tmp_path,include_dirs=[tmp_path],what='adjacent transform scratch')
    dtype=np.float32 if scalar=='float' else np.float64
    q=np.linspace(.07,.23,robot.get_num_pos()).astype(dtype)
    if floating:
        q[3:7]=[.1,-.2,.05,1]
        q[3:7]/=np.linalg.norm(q[3:7])
    q.tofile(tmp_path/'input.bin')
    ref=RBDReference(robot)
    q64=q.astype(float)
    gradient=np.concatenate([j.ravel(order='F') for j in ref.end_effector_pose_gradient(q64)])
    expected=np.r_[gradient,np.asarray(ref.end_effector_pose_hessian_analytic(q64)).ravel(),gradient,
        ref.frame_jacobian(q64,robot.get_joint_by_id(target).get_name()).ravel(order='F'),
        ref.potential_energy_regressor(q64,GRAVITY=float(dtype(-9.81))).ravel()]
    expected=np.tile(expected,16//np.dtype(dtype).itemsize)
    sanitizer=shutil.which('compute-sanitizer')
    assert sanitizer,'CUDA acceptance requires compute-sanitizer on the capable box'
    baseline=None
    for tool,threads in [(None,1),(None,17),(None,32),(None,128),
                          ('racecheck',17),('memcheck',17),('synccheck',17)]:
        label=f'{tool or "run"}-{threads}'
        output=tmp_path/f'{label}.bin'
        cmd=[str(exe),str(tmp_path/'input.bin'),str(output),str(threads)]
        if tool:cmd=[sanitizer,'--tool',tool,'--error-exitcode','86']+cmd
        ran=subprocess.run(cmd,capture_output=True,text=True)
        log=ran.stdout+ran.stderr
        (tmp_path/f'{label}.log').write_text(log)
        assert ran.returncode==0,log
        if tool:
            summary=('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
                     if tool=='racecheck' else 'ERROR SUMMARY: 0 errors')
            assert summary in log,log
        actual=np.fromfile(output,dtype)
        np.testing.assert_allclose(actual,expected,rtol=2e-5 if scalar=='float' else 1e-9,
                                   atol=2e-5 if scalar=='float' else 1e-9)
        if baseline is not None:np.testing.assert_array_equal(actual,baseline)
        baseline=actual

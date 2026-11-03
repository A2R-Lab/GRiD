"""Removing a constant column must preserve a remaining mimic's multiplier."""
import subprocess
import numpy as np
import pytest
from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable
from test.cuda_equivalents.constant_model import constant_model
from test.cuda_equivalents.thread_invariance import thread_sweep

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


@pytest.mark.parametrize('precision', ['float', pytest.param('double', marks=pytest.mark.fp64_alpha)])
def test_constant_branch_keeps_nonunit_mimic_scale(tmp_path, precision):
    constant_model(tmp_path, branched=True)
    path = tmp_path / 'model-False-False-True.urdf'
    text = path.read_text().replace('</robot>', '''
      <link name="tip"><inertial><mass value="1"/>
        <inertia ixx=".3" iyy=".4" izz=".5" ixy="0" ixz="0" iyz="0"/>
      </inertial></link>
      <joint name="tip" type="revolute"><parent link="follower"/><child link="tip"/>
        <origin xyz=".2 .1 0"/><axis xyz="0 0 1"/>
        <limit lower="-3" upper="3" effort="10" velocity="10"/>
        <mimic joint="drive" multiplier="-.5" offset=".1"/>
      </joint></robot>''')
    path.write_text(text)
    robot = URDFParser().parse(str(path))
    gen = GRiDCodeGenerator(robot, dtype=precision)
    gen.gen_all_code(output_path=str(tmp_path / 'grid.cuh'), emit_plant=False,
                     algorithm_list=['end_effector_pose_gradient'],
                     enable_mujoco_kernels=False)
    source = r'''
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
using T=DTYPE;
static_assert(grid::NUM_VEL==1 && grid::NUM_POS==1 && grid::NUM_EES==2);
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 2;}}while(0)
__global__ void probe(T *out,const grid::robotModel<T> *model){
    __shared__ T q[1],jac[12];
    if(threadIdx.x==0)q[0]=T(.35);__syncthreads();
    grid::end_effector_pose_gradient_device<T>(jac,q,model);__syncthreads();
    for(int i=threadIdx.x;i<12;i+=blockDim.x)out[i]=jac[i];
}
int main(int argc,char **argv){
    if(argc!=2)return 1;
    grid::robotModel<T> *model=nullptr;CK(grid::init_robotModel_checked<T>(&model));
    T *out;T h[12];CK(cudaMalloc(&out,sizeof(h)));CK(cudaMemset(out,0xff,sizeof(h)));
    size_t bytes=grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    CK(cudaFuncSetAttribute(probe,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
    probe<<<1,atoi(argv[1]),bytes>>>(out,model);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h,out,sizeof(h),cudaMemcpyDeviceToHost));
    for(auto x:h)printf("%.17g ",double(x));puts("");
    CK(cudaFree(out));CK(grid::free_robotModel_checked<T>(model));
}
'''.replace('DTYPE', precision)
    runner = tmp_path / 'probe.cu'
    runner.write_text(source)
    exe, _ = cached_nvcc_executable([runner], ['-std=c++17', '-O2', '-lineinfo', f'-arch=sm_{_detect_cuda_arch()}'],
                                    exe_name='probe', fallback_dir=tmp_path, include_dirs=[tmp_path],
                                    what='constant branch mimic scale probe')
    expected = np.asarray(RBDReference(robot).end_effector_pose_gradient(np.array([.35]))).ravel()
    # Independent yaw derivative of the tip branch is exactly the mimic scale;
    # the other branch's angular velocity is around y and has zero yaw rate.
    np.testing.assert_array_equal(np.sort(expected.reshape(2, 6)[:, 5]), [-.5, 0.])
    tol = 3e-5 if precision == 'float' else 1e-10

    def run(threads):
        ran = subprocess.run([str(exe), str(threads)], capture_output=True, text=True)
        assert ran.returncode == 0, ran.stderr
        return np.fromstring(ran.stdout, sep=' ')

    def check(actual, _):
        assert actual.size == 12
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)

    thread_sweep((1, 17, 32, 64, 256), run, check)

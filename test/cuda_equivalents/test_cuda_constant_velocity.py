"""A constant branch must not read another branch's velocity placeholder."""
import subprocess

import numpy as np
import pytest
from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


@pytest.mark.parametrize('precision', ['float', pytest.param('double', marks=pytest.mark.fp64_alpha)])
def test_constant_branch_velocity_isolation(tmp_path, precision):
    inertial = ('<inertial><mass value="1"/><origin xyz=".1 .2 .3"/>'
                '<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>')
    links = ''.join(f'<link name="{name}">{inertial}</link>'
                    for name in ('base', 'mount', 'drive', 'constant'))
    urdf = tmp_path / 'branches.urdf'
    urdf.write_text(f'<robot name="branches">{links}'
        '<joint name="fixed" type="fixed"><parent link="base"/><child link="mount"/></joint>'
        '<joint name="drive" type="revolute"><parent link="mount"/><child link="drive"/>'
        '<axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="10"/></joint>'
        '<joint name="constant" type="revolute"><parent link="mount"/><child link="constant"/>'
        '<origin xyz=".2 .1 .3"/><axis xyz="0 0 1"/>'
        '<limit lower="-3" upper="3" effort="10" velocity="10"/>'
        '<mimic joint="fixed" multiplier="2" offset=".17"/></joint></robot>')
    robot = URDFParser().parse(str(urdf))
    gen = GRiDCodeGenerator(robot, dtype=precision)
    gen.gen_all_code(algorithm_list=['inverse_dynamics'], emit_plant=False,
                     enable_mujoco_kernels=False, output_path=str(tmp_path / 'grid.cuh'))
    source = r'''
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <limits>
using T = DTYPE;
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 2;}}while(0)
__global__ void probe(T *out, const T *in, grid::robotModel<T> *model) {
    __shared__ T vaf[18*grid::NUM_BODIES];
    for(int i=threadIdx.x;i<18*grid::NUM_BODIES;i+=blockDim.x) vaf[i]=T(NAN);
    __syncthreads();
    grid::inverse_dynamics_vaf_device<T>(vaf,in,in+1,in+2,model,nullptr,T(-9.81));
    __syncthreads();
    for(int i=threadIdx.x;i<18*grid::NUM_BODIES;i+=blockDim.x) out[i]=vaf[i];
}
int main(int argc,char **argv) {
    if(argc!=3)return 1;
    static_assert(grid::NUM_POS==1 && grid::NUM_VEL==1);
    T in[3]={T(.2),T(0),T(0)};
    if(atoi(argv[2]))in[1]=in[2]=std::numeric_limits<T>::quiet_NaN();
    T *d_in,*out,host[18*grid::NUM_BODIES];
    grid::robotModel<T> *model=nullptr; CK(grid::init_robotModel_checked<T>(&model));
    CK(cudaMalloc(&d_in,sizeof(in)));CK(cudaMalloc(&out,sizeof(host)));
    CK(cudaMemcpy(d_in,in,sizeof(in),cudaMemcpyHostToDevice));
    const size_t bytes=grid::INVERSE_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>();
    CK(cudaFuncSetAttribute(probe,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
    probe<<<1,atoi(argv[1]),bytes>>>(out,d_in,model);
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(host,out,sizeof(host),cudaMemcpyDeviceToHost));
    for(auto x:host)printf("%.17g ",double(x));puts("");
    CK(cudaFree(d_in));CK(cudaFree(out));CK(grid::free_robotModel_checked<T>(model));
}
'''.replace('DTYPE', precision)
    runner = tmp_path / 'probe.cu'
    runner.write_text(source)
    exe, _ = cached_nvcc_executable([runner], ['-std=c++17', '-O2', f'-arch=sm_{_detect_cuda_arch()}'],
                                    exe_name='probe', fallback_dir=tmp_path, include_dirs=[tmp_path],
                                    what='constant velocity probe')
    ref = RBDReference(robot)
    expected = np.asarray(ref.inverse_dynamics_fpass(np.array([.2]), np.zeros(1), np.zeros(1)))
    const = robot.get_joint_by_name('constant').jid
    drive = robot.get_joint_by_name('drive').jid
    baseline = None
    tol = 2e-5 if precision == 'float' else 1e-12
    for threads in (1, 17, 32, 64, 256):
        for poison in (0, 1):
            ran = subprocess.run([str(exe), str(threads), str(poison)], capture_output=True, text=True)
            assert ran.returncode == 0, ran.stderr
            actual = np.fromstring(ran.stdout, sep=' ').reshape(3, robot.get_num_bodies(), 6).transpose(0, 2, 1)
            if poison:
                np.testing.assert_array_equal(actual[:, :, const], baseline[:, :, const])
                assert np.isnan(actual[0, :, drive]).any(), 'Actual NaN inputs must not be sanitized'
            else:
                np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)
                if baseline is not None:
                    np.testing.assert_array_equal(actual, baseline)
                baseline = actual

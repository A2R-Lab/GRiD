"""The cost-menu frame path shares constant-joint column ownership."""
import subprocess
import numpy as np
import pytest
from RBDReference import RBDReference
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable
from test.cuda_equivalents.constant_model import constant_model
from test.cuda_equivalents.thread_invariance import thread_sweep

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


@pytest.mark.parametrize('zero', [False, True])
@pytest.mark.parametrize('precision', ['float', pytest.param('double', marks=pytest.mark.fp64_alpha)])
def test_constant_frame_velocity(tmp_path, zero, precision):
    robot = constant_model(tmp_path, zero=zero)
    gen = GRiDCodeGenerator(robot, dtype=precision)
    gen.gen_all_code(output_path=str(tmp_path / 'grid.cuh'),
                     algorithm_list=['end_effector_pose', 'frame_jacobian'],
                     cost_frames=['constant'], enable_mujoco_kernels=False)
    source = r'''
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
using T=DTYPE;
constexpr int NV=grid::NUM_VEL;
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 2;}}while(0)
__global__ void probe(T *out,const T *in,const grid::robotModel<T> *model) {
    __shared__ T frame[grid_plant::GRID_PLANT_FRAME_SCRATCH_COUNT];
    extern __shared__ __align__(16) unsigned char arena[];
    grid_plant::frame_velocity_eval<T>(0,in,NV ? in+NV : nullptr,frame,reinterpret_cast<T*>(arena),model);
    __syncthreads();
    for(int i=threadIdx.x;i<6;i+=blockDim.x) out[i]=frame[i];
    for(int i=threadIdx.x;i<12*NV;i+=blockDim.x) out[i+6]=frame[16+i];
}
int main(int argc,char **argv) {
    if(argc!=2)return 1;
    T *in=nullptr,*out;T host[6+12*NV];
    grid::robotModel<T> *model=nullptr;CK(grid::init_robotModel_checked<T>(&model));
    if constexpr(NV>0){T h[2]={T(.35),T(.23)};CK(cudaMalloc(&in,sizeof(h)));CK(cudaMemcpy(in,h,sizeof(h),cudaMemcpyHostToDevice));}
    CK(cudaMalloc(&out,sizeof(host)));CK(cudaMemset(out,0xff,sizeof(host)));
    size_t bytes=grid_plant::FRAME_EVAL_DYNAMIC_SHARED_MEM_BYTES<T>();
    CK(cudaFuncSetAttribute(probe,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
    probe<<<1,atoi(argv[1]),bytes>>>(out,in,model);
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(host,out,sizeof(host),cudaMemcpyDeviceToHost));
    for(auto x:host)printf("%.17g ",double(x));puts("");
    CK(cudaFree(in));CK(cudaFree(out));CK(grid::free_robotModel_checked<T>(model));
}
'''.replace('DTYPE', precision)
    runner = tmp_path / 'probe.cu'
    runner.write_text(source)
    exe, _ = cached_nvcc_executable([runner], ['-std=c++17', '-O2', f'-arch=sm_{_detect_cuda_arch()}'],
                                    exe_name='probe', fallback_dir=tmp_path, include_dirs=[tmp_path],
                                    what='constant frame velocity probe')
    ref = RBDReference(robot)
    q, v = np.full(robot.get_num_pos(), .35), np.full(robot.get_num_vel(), .23)
    J = ref.frame_jacobian(q, 'constant', reference_frame='LOCAL_WORLD_ALIGNED')
    if zero:
        dJv = np.empty((6, 0))
    else:
        h = 1e-6
        dJv = ((ref.frame_jacobian(q+h, 'constant', reference_frame='LOCAL_WORLD_ALIGNED')
                - ref.frame_jacobian(q-h, 'constant', reference_frame='LOCAL_WORLD_ALIGNED')) @ v / (2*h)).reshape(6, 1)
    expected = np.r_[J @ v, J.ravel(order='F'), dJv.ravel(order='F')]
    tol = 3e-5 if precision == 'float' else 1e-9

    def run(threads):
        ran = subprocess.run([str(exe), str(threads)], capture_output=True, text=True)
        assert ran.returncode == 0, ran.stderr
        return np.fromstring(ran.stdout, sep=' ')

    thread_sweep((1, 17, 32, 64, 256), run,
                 lambda actual, _: np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol))

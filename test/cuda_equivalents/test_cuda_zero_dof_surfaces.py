"""Empty outputs must not turn constant-body placeholders into device accesses."""
import subprocess

import numpy as np
import pytest
from grid_codegen import GRiDCodeGenerator
from RBDReference import RBDReference
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.executable_cache import cached_nvcc_executable
from test.cuda_equivalents.constant_model import constant_model

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


@pytest.mark.parametrize('precision', ['float', pytest.param('double', marks=pytest.mark.fp64_alpha)])
@pytest.mark.parametrize('surface,zero', [('pose-hessian', True), ('pose-hessian', False),
                                        ('plant-gradient', True), ('plant-gradient', False)])
def test_zero_dof_device_surfaces(tmp_path, precision, surface, zero):
    robot = constant_model(tmp_path, zero=zero)
    gen = GRiDCodeGenerator(robot, dtype=precision)
    gen.gen_all_code(output_path=str(tmp_path / 'grid.cuh'),
                     algorithm_list=['end_effector_pose_hessian' if surface == 'pose-hessian'
                                     else 'integrator_gradient'],
                     emit_plant=surface == 'plant-gradient', enable_mujoco_kernels=False)
    if surface == 'pose-hessian':
        kernel = '''
__global__ void probe(T *out, const T *q, const grid::robotModel<T> *model) {
    grid::end_effector_pose_hessian_device<T>(out+1,out+1+6*NV*NV,q,model);
}
'''
        launch = 'probe<<<1,threads,bytes>>>(out,q,model);'
        resource = 'grid::END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>()'
        attribute = 'probe'
        threads = (1, 17, 32, 64, 256)
    else:
        kernel = ''
        launch = ('grid_plant::plant_step_gradient_kernel<T,grid::IntegratorType::EULER>'
                  '<<<1,threads,bytes>>>(out+1,q,NV ? q+2*NV : nullptr,2*NV,NV,model,T(-9.81),T(.01),1);')
        resource = 'grid_plant::PLANT_STEP_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()'
        attribute = 'grid_plant::plant_step_gradient_kernel<T,grid::IntegratorType::EULER>'
        # The generated zero-DoF plant kernel has a 32-thread launch bound.
        threads = (1, 17, 32)
    source = r'''
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
using T=DTYPE;
constexpr int NV=grid::NUM_VEL;
constexpr int NOUT=OUTPUT_COUNT;
static_assert(grid::NUM_POS==NV && NV<=1);
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 2;}}while(0)
KERNEL
int main(int argc,char **argv){
    if(argc!=2)return 1;
    int threads=atoi(argv[1]);
    grid::robotModel<T> *model=nullptr;CK(grid::init_robotModel_checked<T>(&model));
    T *out,*q=nullptr;T h[NOUT+2];
    for(auto &x:h)x=T(NAN);h[0]=123;h[NOUT+1]=456;
    if constexpr(NV>0){T hq[3]={T(.35),T(.23),T(-.31)};CK(cudaMalloc(&q,sizeof(hq)));CK(cudaMemcpy(q,hq,sizeof(hq),cudaMemcpyHostToDevice));}
    CK(cudaMalloc(&out,sizeof(h)));CK(cudaMemcpy(out,h,sizeof(h),cudaMemcpyHostToDevice));
    size_t bytes=RESOURCE;
    CK((cudaFuncSetAttribute(ATTRIBUTE,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes))));
    LAUNCH
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(h,out,sizeof(h),cudaMemcpyDeviceToHost));
    for(auto x:h)printf("%.17g ",double(x));puts("");
    CK(cudaFree(q));CK(cudaFree(out));CK(grid::free_robotModel_checked<T>(model));
}
'''
    for old, new in [('DTYPE', precision), ('KERNEL', kernel), ('RESOURCE', resource),
                     ('ATTRIBUTE', attribute), ('LAUNCH', launch),
                     ('OUTPUT_COUNT', '6*NV*NV+6*NV' if surface == 'pose-hessian' else '6*NV*NV')]:
        source = source.replace(old, new)
    runner = tmp_path / 'probe.cu'
    runner.write_text(source)
    exe, _ = cached_nvcc_executable([runner], ['-std=c++17', '-O2', '-lineinfo', f'-arch=sm_{_detect_cuda_arch()}'],
                                    exe_name='probe', fallback_dir=tmp_path, include_dirs=[tmp_path],
                                    what='zero dof surfaces probe')
    expected = np.array([123, 456])
    if not zero:
        ref = RBDReference(robot)
        q = np.array([.35])
        if surface == 'pose-hessian':
            hessian = np.asarray(ref.end_effector_pose_hessian_analytic(q)).ravel()
            jacobian = np.asarray(ref.end_effector_pose_gradient(q)).ravel()
            # One coordinate and one EE: both flattenings agree with device layout.
            expected = np.r_[123, hessian, jacobian, 456]
            step = 1e-5
            finite = (np.asarray(ref.end_effector_pose_gradient(q+step))
                      - np.asarray(ref.end_effector_pose_gradient(q-step))) / (2*step)
            np.testing.assert_allclose(hessian, finite.ravel(), atol=1e-8, rtol=1e-8)
        else:
            args = (q, np.array([.23]), np.array([-.31]), .01)
            gradient = ref.plant_step_gradient(*args)
            locked = RBDReference(constant_model(tmp_path, locked=True))
            np.testing.assert_allclose(gradient, locked.plant_step_gradient(*args), atol=1e-11, rtol=1e-10)
            expected = np.r_[123, gradient.ravel(order='F'), 456]
    baseline = None
    for count in threads:
        ran = subprocess.run([str(exe), str(count)], capture_output=True, text=True)
        assert ran.returncode == 0, ran.stderr
        actual = np.fromstring(ran.stdout, sep=' ')
        assert actual.size == expected.size
        np.testing.assert_array_equal(actual[[0, -1]], [123, 456])
        tol = 3e-5 if precision == 'float' else 1e-10
        np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)
        if baseline is not None:
            np.testing.assert_array_equal(actual, baseline)
        baseline = actual

"""Exact target scratch: guarded allocations, independent pose/Jacobian oracle."""
import shutil
import subprocess
import numpy as np
import pytest
from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch

pytestmark=[pytest.mark.cuda_equivalence,pytest.mark.robot_smoke]

SOURCE=r'''
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
using T=TEST_SCALAR;
constexpr int P=grid::EE_TARGET_POSE_INNER_SCRATCH_BYTES<T>()/sizeof(T);
constexpr int G=grid::EE_TARGET_GRAD_INNER_SCRATCH_BYTES<T>()/sizeof(T);
constexpr int OUT=6*grid::NUM_TARGET_EES*(1+grid::NUM_VEL);
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 2;}}while(0)
__global__ void probe(T *out,const T *q,const grid::robotModel<T>*model,int*bad){
    __shared__ T xh[grid::XHOM_T_COUNT],dxh[grid::DXHOM_T_COUNT];
    __shared__ T loader[32*grid::NUM_POS+100],ptmp[P+2],gtmp[G+2];
    __shared__ int topo[grid::TOPOLOGY_HELPERS_COUNT ? grid::TOPOLOGY_HELPERS_COUNT:1];
    const int tid=threadIdx.x,nt=blockDim.x;
    for(int repeat=0;repeat<2;repeat++){
        for(int i=tid;i<P+2;i+=nt)ptmp[i]=NAN;
        for(int i=tid;i<G+2;i+=nt)gtmp[i]=NAN;
        __syncthreads();
        if(tid==0){ptmp[0]=gtmp[0]=123;ptmp[P+1]=gtmp[G+1]=456;}
        __syncthreads();
        grid::load_update_XmatsHom_helpers<T>(xh,dxh,topo,q,model,loader);
        __syncthreads();
        grid::end_effector_pose_target_inner<T>(out,q,xh,topo,ptmp+1,nullptr);
        __syncthreads();
        grid::end_effector_pose_gradient_target_inner<T>(out+6*grid::NUM_TARGET_EES,q,xh,dxh,topo,gtmp+1,nullptr);
        __syncthreads();
        if(tid==0 && (ptmp[0]!=123 || ptmp[P+1]!=456 || gtmp[0]!=123 || gtmp[G+1]!=456))atomicAdd(bad,1);
        __syncthreads();
    }
}
int main(int argc,char**argv){
    if(argc!=4)return 1;
    std::vector<T> q(grid::NUM_POS),out(OUT);
    FILE*f=fopen(argv[1],"rb");if(!f||fread(q.data(),sizeof(T),q.size(),f)!=q.size())return 3;fclose(f);
    T*dq,*dout;int*bad;grid::robotModel<T>*model=nullptr;
    CK(grid::init_robotModel_checked<T>(&model));CK(cudaMalloc(&dq,q.size()*sizeof(T)));
    CK(cudaMalloc(&dout,OUT*sizeof(T)));CK(cudaMalloc(&bad,sizeof(int)));CK(cudaMemset(bad,0,sizeof(int)));
    CK(cudaMemcpy(dq,q.data(),q.size()*sizeof(T),cudaMemcpyHostToDevice));CK(cudaMemset(dout,0xff,OUT*sizeof(T)));
    probe<<<1,atoi(argv[3])>>>(dout,dq,model,bad);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    int errors;CK(cudaMemcpy(&errors,bad,sizeof(int),cudaMemcpyDeviceToHost));if(errors)return 4;
    CK(cudaMemcpy(out.data(),dout,OUT*sizeof(T),cudaMemcpyDeviceToHost));
    f=fopen(argv[2],"wb");if(!f||fwrite(out.data(),sizeof(T),OUT,f)!=OUT)return 5;fclose(f);
    CK(cudaFree(dq));CK(cudaFree(dout));CK(cudaFree(bad));CK(grid::free_robotModel_checked<T>(model));
}
'''


@pytest.mark.parametrize('name,target',[('iiwa14','iiwa_joint_ee'),('go2','FR_foot_joint'),('go2','')])
@pytest.mark.parametrize('scalar',['float','double'])
def test_exact_target_scratch(tmp_path,name,target,scalar):
    robot=URDFParser().parse(str(robot_urdf(name)),floating_base=name=='go2')
    gen=GRiDCodeGenerator(robot,dtype=scalar)
    gen.gen_all_code(algorithm_list=['end_effector_pose','end_effector_pose_gradient'],
                     fixed_target_name=target,include_homogenous_transforms=True,
                     enable_mujoco_kernels=False,output_path=str(tmp_path/'grid.cuh'))
    (tmp_path/'probe.cu').write_text(SOURCE)
    exe=tmp_path/'probe'
    result=subprocess.run([shutil.which('nvcc'),'-std=c++17','-O2','-lineinfo','-arch=sm_'+_detect_cuda_arch(),
        '-DTEST_SCALAR='+scalar,str(tmp_path/'probe.cu'),'-o',str(exe)],capture_output=True,text=True)
    (tmp_path/'compile.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    dtype=np.float32 if scalar=='float' else np.float64
    q=np.linspace(.07,.23,robot.get_num_pos()).astype(dtype)
    if name=='go2':q[3:7]=[0,0,0,1]
    q.tofile(tmp_path/'input.bin')
    ref=RBDReference(robot);names=[target] if target else None
    expected=np.r_[np.asarray(ref.end_effector_pose(q.astype(float),ee_joint_names=names)).ravel(),
        np.concatenate([j.ravel(order='F') for j in ref.end_effector_pose_gradient(q.astype(float),ee_joint_names=names)])]
    baseline=None
    for threads in (1,17,32,64,256):
        output=tmp_path/f'out-{threads}.bin'
        result=subprocess.run([str(exe),str(tmp_path/'input.bin'),str(output),str(threads)],capture_output=True,text=True)
        (tmp_path/f'run-{threads}.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stderr
        actual=np.fromfile(output,dtype)
        np.testing.assert_allclose(actual,expected,rtol=2e-5 if scalar=='float' else 1e-9,
                                   atol=2e-5 if scalar=='float' else 1e-9)
        if baseline is not None:np.testing.assert_array_equal(actual,baseline)
        baseline=actual

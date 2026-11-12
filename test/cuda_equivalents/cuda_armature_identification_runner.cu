// Direct collective helpers, including two-dimensional block-stride indexing.
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include "grid.cuh"
#ifdef TEST_DOUBLE
using Scalar = double;
#else
using Scalar = float;
#endif
#define CK(expr) do { cudaError_t e=(expr); if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<"\n";return 2;} } while(0)
__global__ void probe(Scalar *out, const Scalar *input, const grid::robotModel<Scalar>*model) {
    __shared__ Scalar q[grid::NUM_POS], v[grid::NUM_VEL], u[grid::NUM_VEL];
    const int t=threadIdx.x+blockDim.x*threadIdx.y, stride=blockDim.x*blockDim.y;
    for(int i=t;i<grid::NUM_POS;i+=stride) q[i]=input[i];
    for(int i=t;i<grid::NUM_VEL;i+=stride) {
        v[i]=input[grid::NUM_POS+i]; u[i]=input[grid::NUM_POS+grid::NUM_VEL+i];
    }
    __syncthreads();
    grid::armature_regressor_device(out,input+grid::NUM_POS+2*grid::NUM_VEL);
    __syncthreads();
    grid::forward_dynamics_armature_gradient_device(
        out+grid::NUM_VEL*grid::NUM_ARMATURE_PARAMS,q,v,u,model,
        const_cast<Scalar*>(input+grid::NUM_POS+3*grid::NUM_VEL),Scalar(-9.81));
}
int main(int argc,char**argv) {
    if(argc!=3)return 1;
    const int count=grid::NUM_POS+3*grid::NUM_VEL+6*grid::NUM_BODIES;
    const int output=2*grid::NUM_VEL*grid::NUM_ARMATURE_PARAMS;
    std::vector<Scalar> host(count), result(output);
    for(auto &v:host)if(!(std::cin>>v))return 1;
    Scalar *in=nullptr,*out=nullptr;
    CK(cudaMalloc(&in,count*sizeof(Scalar))); CK(cudaMalloc(&out,(output?output:1)*sizeof(Scalar)));
    CK(cudaMemcpy(in,host.data(),count*sizeof(Scalar),cudaMemcpyHostToDevice));
    grid::robotModel<Scalar>*model=nullptr; CK(grid::init_robotModel_checked<Scalar>(&model));
    constexpr size_t bytes=grid::FORWARD_DYNAMICS_ARMATURE_GRADIENT_DEVICE_SHARED_MEM_BYTES<Scalar>();
    cudaFuncAttributes attr; CK(cudaFuncGetAttributes(&attr,probe));
    int limit; CK(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,0));
    if(bytes+attr.sharedSizeBytes>size_t(limit))return 3;
    CK(cudaFuncSetAttribute(probe,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    probe<<<1,dim3(std::atoi(argv[1]),std::atoi(argv[2])),bytes>>>(out,in,model);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(result.data(),out,output*sizeof(Scalar),cudaMemcpyDeviceToHost));
    std::cout<<std::setprecision(17);
    for(auto v:result)std::cout<<v<<" "; std::cout<<"\n";
    CK(cudaFree(in)); CK(cudaFree(out)); CK(grid::free_robotModel_checked<Scalar>(model));
    return 0;
}

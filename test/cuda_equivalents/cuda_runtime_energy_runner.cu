// Independent device-helper entry, with nonuniform live inertia and armature.
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
__global__ void probe(Scalar *out,const Scalar *input,const grid::robotModel<Scalar>*model) {
    __shared__ Scalar q[grid::NUM_POS],v[grid::NUM_VEL],result[3];
    for(int i=threadIdx.x;i<grid::NUM_POS;i+=blockDim.x) q[i]=input[i];
    for(int i=threadIdx.x;i<grid::NUM_VEL;i+=blockDim.x) v[i]=input[grid::NUM_POS+i];
    __syncthreads();
    grid::energy_device(result,q,v,model,Scalar(-9.81));
    for(int i=threadIdx.x;i<3;i+=blockDim.x) out[i]=result[i];
}
int main() {
    std::vector<Scalar> input(grid::NUM_POS+grid::NUM_VEL),pi(10*grid::NUM_BODIES),a(grid::NUM_VEL);
    for(auto *values:{&input,&pi,&a})for(auto &v:*values)if(!(std::cin>>v))return 1;
    grid::robotModel<Scalar>*model=nullptr;
    CK(grid::init_robotModel_checked<Scalar>(&model));
    grid::set_inertia_params(model,pi.data()); CK(cudaDeviceSynchronize());
    CK(grid::set_armature_params_checked(model,a.data(),a.size()));
    Scalar *in=nullptr,*out=nullptr,result[3];
    CK(cudaMalloc(&in,input.size()*sizeof(Scalar))); CK(cudaMalloc(&out,sizeof(result)));
    CK(cudaMemcpy(in,input.data(),input.size()*sizeof(Scalar),cudaMemcpyHostToDevice));
    constexpr size_t bytes=grid::ENERGY_DYNAMIC_SHARED_MEM_BYTES<Scalar,grid::TIER_SHARED>();
    CK(cudaFuncSetAttribute(probe,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    probe<<<1,32,bytes>>>(out,in,model);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(result,out,sizeof(result),cudaMemcpyDeviceToHost));
    std::cout<<std::setprecision(17);
    for(auto v:result)std::cout<<v<<" "; std::cout<<"\n";
    CK(cudaFree(in)); CK(cudaFree(out)); CK(grid::free_robotModel_checked(model));
}

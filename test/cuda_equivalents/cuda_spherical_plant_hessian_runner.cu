// Existing public stock kernel; no new workspace-binder support implied.
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
using T=TEST_SCALAR;
constexpr int N=grid::NUM_VEL, NX=grid::NUM_POS+N, NH=18*N*N*N;
constexpr auto IT=grid::IntegratorType::TEST_INTEGRATOR;
constexpr int TIER=TEST_TIER;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) {fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);} } while(0)
int main(int argc,char**argv) {
    if(argc!=4) return 1;
    int threads=atoi(argv[3]);
    std::vector<T> z(NX+N),out(NH+2);
    FILE*f=fopen(argv[1],"rb"); if(!f||fread(z.data(),sizeof(T),z.size(),f)!=z.size()) return 3; fclose(f);
    T *input,*output; unsigned char *workspace;
    CK(cudaMalloc(&input,z.size()*sizeof(T))); CK(cudaMalloc(&output,out.size()*sizeof(T)));
    CK(cudaMalloc(&workspace,grid_plant::PLANT_HESSIAN_WORKSPACE_BYTES_PER_TIMESTEP<T>()));
    CK(cudaMemcpy(input,z.data(),z.size()*sizeof(T),cudaMemcpyHostToDevice));
    grid::robotModel<T>*model=nullptr; CK(grid::init_robotModel_checked<T>(&model));
    constexpr size_t sh=grid_plant::INTEGRATOR_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T,TIER>();
    auto kernel=grid_plant::plant_step_hessian_kernel<T,IT,TIER>;
    cudaFuncAttributes attributes{}; CK(cudaFuncGetAttributes(&attributes,kernel));
    if(threads>attributes.maxThreadsPerBlock) {printf("ADMISSION_LIMIT=%d\n",attributes.maxThreadsPerBlock);return 7;}
    CK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sh));
    for(int repeat=0;repeat<2;++repeat) {
        std::fill(out.begin(),out.end(),T(123+repeat));
        CK(cudaMemcpy(output,out.data(),out.size()*sizeof(T),cudaMemcpyHostToDevice));
        CK(cudaMemset(workspace,0xff,grid_plant::PLANT_HESSIAN_WORKSPACE_BYTES_PER_TIMESTEP<T>()));
        grid_plant::plant_step_hessian_kernel<T,IT,TIER><<<1,threads,sh>>>(output+1,workspace,input,input+NX,NX,N,model,T(-9.81),T(.02),1);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(out.data(),output,out.size()*sizeof(T),cudaMemcpyDeviceToHost));
        if(out.front()!=T(123+repeat)||out.back()!=T(123+repeat)) return 4;
        f=fopen((std::string(argv[2])+std::to_string(repeat)).c_str(),"wb");
        if(!f||fwrite(out.data()+1,sizeof(T),NH,f)!=NH) return 5; fclose(f);
    }
    CK(cudaFree(input)); CK(cudaFree(output)); CK(cudaFree(workspace));
    CK(grid::free_robotModel_checked<T>(model));
}

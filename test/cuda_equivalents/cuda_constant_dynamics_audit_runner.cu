#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
using T=DTYPE;
constexpr int NQ=grid::NUM_POS, NV=grid::NUM_VEL, NB=grid::NUM_BODIES;
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));return 2;}}while(0)
namespace grid {
template<class T> __device__ void audit_public_body_layout(T *s_idsva_so){
BODY_OUTPUT_REPAIR
}
}
// The SO inners explicitly permit global output (their normal spill path).
// Use private per-launch output, with guards and NaN-initialized scratch.
template<int MODE> __global__ void probe(T *out,const T *input,T *workspace,const grid::robotModel<T> *model){
    constexpr int TEMP=MODE==0?CRBA_TEMP:MODE==1?ID_TEMP:MODE==2?BODY_TEMP:WORLD_TEMP;
    __shared__ T xi[72*NB],vaf[18*NB],a[NV];
    extern __shared__ T scratch[];
    __shared__ int topology[grid::TOPOLOGY_HELPERS_COUNT?grid::TOPOLOGY_HELPERS_COUNT:1];
    const T *q=input,*v=input+NQ;
    for(int i=threadIdx.x;i<NV;i+=blockDim.x)a[i]=v[NV+i];
    for(int i=threadIdx.x;i<TEMP;i+=blockDim.x)scratch[i]=T(NAN);
    __syncthreads();
    if constexpr(MODE==0){
        grid::load_update_XImats_helpers<T>(xi,q,topology,model,scratch);
        grid::crba_inner<T,true>(out+1,q,v,xi,topology,scratch,nullptr,T(-9.81));
    }
    if constexpr(MODE==1) grid::inverse_dynamics_gradient_device_qdd<T>(out+1,q,v,vaf,a,xi,topology,scratch,nullptr,nullptr,model,nullptr,T(-9.81));
    if constexpr(MODE==2){
        grid::idsva_so_body_frame_inner<T>(out+1,q,v,a,xi,topology,scratch,workspace,model,T(-9.81));
        grid::audit_public_body_layout<T>(out+1);
    }
    if constexpr(MODE==3) grid::idsva_so_world_frame_inner<T>(out+1,q,v,a,xi,topology,scratch,nullptr,model,T(-9.81));
}
int main(int argc,char **argv){
    if(argc!=4)return 1;
    int threads=atoi(argv[1]),mode=atoi(argv[2]);
    std::vector<T> input(NQ+2*NV);
    FILE *f=fopen(argv[3],"r");if(!f)return 3;
    for(auto &x:input){double v;if(fscanf(f,"%lf",&v)!=1)return 4;x=T(v);}fclose(f);
    int count=(mode==0?NV*NV:mode==1?2*NV*NV:4*NV*NV*NV)+2;
    std::vector<T> host(count,T(NAN));host.front()=123;host.back()=456;
    T *in,*out,*workspace=nullptr;grid::robotModel<T> *model=nullptr;CK(grid::init_robotModel_checked<T>(&model));
    if constexpr(GRAV_TEMP>0){CK(cudaMalloc(&workspace,GRAV_TEMP*sizeof(T)));CK(cudaMemset(workspace,0xff,GRAV_TEMP*sizeof(T)));}
    CK(cudaMalloc(&in,input.size()*sizeof(T)));CK(cudaMemcpy(in,input.data(),input.size()*sizeof(T),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&out,count*sizeof(T)));CK(cudaMemcpy(out,host.data(),count*sizeof(T),cudaMemcpyHostToDevice));
    if(mode==0){CK(cudaFuncSetAttribute(probe<0>,cudaFuncAttributeMaxDynamicSharedMemorySize,CRBA_TEMP*sizeof(T)));probe<0><<<1,threads,CRBA_TEMP*sizeof(T)>>>(out,in,workspace,model);}
    if(mode==1){CK(cudaFuncSetAttribute(probe<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,ID_TEMP*sizeof(T)));probe<1><<<1,threads,ID_TEMP*sizeof(T)>>>(out,in,workspace,model);}
    if(mode==2){CK(cudaFuncSetAttribute(probe<2>,cudaFuncAttributeMaxDynamicSharedMemorySize,BODY_TEMP*sizeof(T)));probe<2><<<1,threads,BODY_TEMP*sizeof(T)>>>(out,in,workspace,model);}
    if(mode==3){CK(cudaFuncSetAttribute(probe<3>,cudaFuncAttributeMaxDynamicSharedMemorySize,WORLD_TEMP*sizeof(T)));probe<3><<<1,threads,WORLD_TEMP*sizeof(T)>>>(out,in,workspace,model);}
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(host.data(),out,count*sizeof(T),cudaMemcpyDeviceToHost));
    for(auto x:host)printf("%.17g ",double(x));puts("");
    CK(cudaFree(in));CK(cudaFree(out));CK(cudaFree(workspace));CK(grid::free_robotModel_checked<T>(model));
}

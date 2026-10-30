#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
using T = DTYPE;
constexpr int NV = grid::NUM_VEL;
constexpr int NB = grid::NUM_BODIES;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); return 2; } } while(0)

__global__ void gradient(T *out, const T *in, const grid::robotModel<T> *model, bool internal) {
    __shared__ T vaf[18*NB], xi[72*NB], scratch[ID_TEMP];
    __shared__ int topology[grid::TOPOLOGY_HELPERS_COUNT ? grid::TOPOLOGY_HELPERS_COUNT : 1];
    __shared__ T guarded[2*NV*NV+2];
    for (int i=threadIdx.x;i<ID_TEMP;i+=blockDim.x) scratch[i]=T(NAN);
    for (int i=threadIdx.x;i<2*NV*NV+2;i+=blockDim.x) guarded[i]=T(NAN);
    if(threadIdx.x==0) { guarded[0]=123; guarded[2*NV*NV+1]=456; }
    __syncthreads();
    // Null really means zero coordinates; no fabricated slot zero is allocated.
    const T *v = NV ? in+NV : nullptr, *a = NV ? in+2*NV : nullptr;
    grid::inverse_dynamics_gradient_device_qdd<T>(guarded+1,in,v,vaf,a,xi,topology,scratch,
                                                 nullptr,nullptr,model,nullptr,T(-9.81));
    __syncthreads();
    if(internal) {
        if(threadIdx.x==0) {out[0]=123;out[36*NV*NB+1]=456;}
        for(int i=threadIdx.x;i<36*NV*NB;i+=blockDim.x) out[i+1]=scratch[i];
    } else for(int i=threadIdx.x;i<2*NV*NV+2;i+=blockDim.x) out[i]=guarded[i];
}

__global__ void frame(T *out, const T *in, const grid::robotModel<T> *model, int mode) {
    __shared__ T guarded[6*NV+2];
    for(int i=threadIdx.x;i<6*NV+2;i+=blockDim.x) guarded[i]=T(NAN);
    if(threadIdx.x==0) { guarded[0]=123; guarded[6*NV+1]=456; }
    __syncthreads();
    if(mode==1) grid::frame_jacobian_device<T>(guarded+1,NB-1,2,in,model);
    else grid::frame_jacobian_dot_device<T>(guarded+1,NB-1,2,in,NV ? in+NV : nullptr,model);
    __syncthreads();
    for(int i=threadIdx.x;i<6*NV+2;i+=blockDim.x) out[i]=guarded[i];
}

int main(int argc,char **argv) {
    if(argc!=3 && argc!=4) return 1;
    int threads=atoi(argv[1]), mode=atoi(argv[2]);
    T *in=nullptr,*out=nullptr;
    grid::robotModel<T> *model=nullptr; CK(grid::init_robotModel_checked<T>(&model));
    if constexpr(NV>0) {
        T h[3]={T(.35),T(.23),T(-.31)};
        if(argc==4) h[1]=h[2]=T(NAN);
        CK(cudaMalloc(&in,sizeof(h))); CK(cudaMemcpy(in,h,sizeof(h),cudaMemcpyHostToDevice));
    }
    int count=(mode==3 ? 36*NV*NB : mode==0 ? 2*NV*NV : 6*NV)+2;
    CK(cudaMalloc(&out,count*sizeof(T)));
    if(mode==0 || mode==3) gradient<<<1,threads>>>(out,in,model,mode==3);
    else {
        // Existing public kernel sizer safely covers the nested device carve.
        size_t bytes=grid::FRAME_JACOBIAN_DOT_DYNAMIC_SHARED_MEM_BYTES<T>();
        CK(cudaFuncSetAttribute(frame,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
        frame<<<1,threads,bytes>>>(out,in,model,mode);
    }
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<T> host(count); CK(cudaMemcpy(host.data(),out,count*sizeof(T),cudaMemcpyDeviceToHost));
    for(T x:host) printf("%.17g ",double(x)); puts("");
    CK(cudaFree(in)); CK(cudaFree(out)); CK(grid::free_robotModel_checked<T>(model));
}

#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
using T = DTYPE;
constexpr int NV = grid::NUM_VEL;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); return 2; } } while(0)
// Compose the existing inners without the generated kernels' 64-thread launch
// bound. This tests the device contract at 256 threads, not an invalid launch.
__global__ void probe(T *out, const T *in, const grid::robotModel<T> *model, int mode) {
    __shared__ T xi[72*grid::NUM_BODIES], scratch[BODY_TEMP > WORLD_TEMP ? BODY_TEMP : WORLD_TEMP];
    __shared__ T qdd[NV ? NV : 1];
    __shared__ int topology[grid::TOPOLOGY_HELPERS_COUNT ? grid::TOPOLOGY_HELPERS_COUNT : 1];
    __shared__ T result[4*NV*NV*NV+NV*NV+2];
    int n=mode==2 ? NV*NV : 4*NV*NV*NV;
    for(int i=threadIdx.x;i<n+2;i+=blockDim.x) result[i]=T(NAN);
    if(threadIdx.x==0) {result[0]=123;result[n+1]=456;}
    for(int i=threadIdx.x;i<NV;i+=blockDim.x) qdd[i]=in[2*NV+i];
    __syncthreads();
    const T *v=NV ? in+NV : nullptr;
    if(mode==0) grid::idsva_so_body_frame_inner<T>(result+1,in,v,NV ? qdd : nullptr,xi,topology,scratch,nullptr,model,T(-9.81));
    else if(mode==1) grid::idsva_so_world_frame_inner<T>(result+1,in,v,NV ? qdd : nullptr,xi,topology,scratch,nullptr,model,T(-9.81));
    else grid::coriolis_matrix_device<T>(result+1,in,v,model,T(-9.81));
    __syncthreads();
    for(int i=threadIdx.x;i<n+2;i+=blockDim.x) out[i]=result[i];
}
int main(int argc,char **argv) {
    if(argc!=3) return 1;
    int threads=atoi(argv[1]), mode=atoi(argv[2]);
    T *in=nullptr,*out=nullptr;
    grid::robotModel<T> *model=nullptr; CK(grid::init_robotModel_checked<T>(&model));
    if constexpr(NV>0) {
        T h[3]={T(.35),T(.23),T(-.31)};
        CK(cudaMalloc(&in,sizeof(h))); CK(cudaMemcpy(in,h,sizeof(h),cudaMemcpyHostToDevice));
    }
    int count=(mode==2 ? NV*NV : 4*NV*NV*NV)+2;
    std::vector<T> host(count,T(NAN)); host.front()=123; host.back()=456;
    CK(cudaMalloc(&out,count*sizeof(T)));
    CK(cudaMemcpy(out,host.data(),count*sizeof(T),cudaMemcpyHostToDevice));
    size_t bytes=grid::CORIOLIS_MATRIX_DYNAMIC_SHARED_MEM_BYTES<T>();
    CK(cudaFuncSetAttribute(probe,cudaFuncAttributeMaxDynamicSharedMemorySize,int(bytes)));
    probe<<<1,threads,bytes>>>(out,in,model,mode);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(host.data(),out,count*sizeof(T),cudaMemcpyDeviceToHost));
    for(T x:host) printf("%.17g ",double(x)); puts("");
    CK(cudaFree(in)); CK(cudaFree(out)); CK(grid::free_robotModel_checked<T>(model));
}

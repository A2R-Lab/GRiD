// Optimized real consumers of both world-FK emitters, at every T-aligned residue.
// No extra caller capacity: guards begin immediately after each inner's count.
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
using T = TEST_SCALAR;
constexpr int G = 6*grid::NUM_VEL*grid::NUM_EES;
constexpr int H = G*grid::NUM_VEL;
constexpr int F = 6*grid::NUM_VEL;
constexpr int E = 10*grid::NUM_BODIES;
constexpr int WIDTH = 2*G+H+F+E;
constexpr int RESIDUES = 16/sizeof(T);
constexpr int GS = grid::EE_GRAD_INNER_SMEM_BYTES<T>()/sizeof(T);
constexpr int MAX_S = HESS_SCRATCH > GS ? HESS_SCRATCH : GS;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); return 2; } } while(0)

__device__ void fill_scratch(T *base, int offset, int count) {
    for(int i=threadIdx.x;i<MAX_S+8;i+=blockDim.x)
        base[i]=(i>=offset && i<offset+count)?T(NAN):T(123);
    __syncthreads();
}
__device__ void check_scratch(T *base, int offset, int count, int *bad) {
    __syncthreads();
    for(int i=threadIdx.x;i<MAX_S+8;i+=blockDim.x)
        if((i<offset || i>=offset+count) && base[i]!=T(123))atomicAdd(bad,1);
    __syncthreads();
}
__global__ void probe(T *out,const T *q,const grid::robotModel<T> *model,int *bad) {
    __shared__ __align__(16) T xbuf[grid::XHOM_T_COUNT+8],scratch[MAX_S+8];
    __shared__ T dxh[grid::DXHOM_T_COUNT],loader[32*grid::NUM_POS+100];
    __shared__ int topo[grid::TOPOLOGY_HELPERS_COUNT?grid::TOPOLOGY_HELPERS_COUNT:1];
    for(int offset=1;offset<=RESIDUES;offset++) {
        T *xh=xbuf+offset, *s=scratch+offset, *o=out+(offset-1)*WIDTH;
        for(int i=threadIdx.x;i<grid::XHOM_T_COUNT+8;i+=blockDim.x)xbuf[i]=T(321);
        __syncthreads();
        grid::load_update_XmatsHom_helpers<T>(xh,dxh,topo,q,model,loader);
        __syncthreads();
        fill_scratch(scratch,offset,GS);
        grid::end_effector_pose_gradient_inner<T>(o,q,xh,dxh,topo,s,nullptr);
        check_scratch(scratch,offset,GS,bad);
        fill_scratch(scratch,offset,HESS_SCRATCH);
        grid::end_effector_pose_hessian_inner<T>(o+G,o+G+H,q,xh,topo,s,nullptr,model);
        check_scratch(scratch,offset,HESS_SCRATCH,bad);
        fill_scratch(scratch,offset,FRAME_SCRATCH);
        grid::frame_jacobian_inner<T>(o+2*G+H,TARGET_JID,2,q,xh,model,s);
        check_scratch(scratch,offset,FRAME_SCRATCH,bad);
        fill_scratch(scratch,offset,FRAME_SCRATCH);
        grid::potential_energy_regressor_inner<T>(o+2*G+H+F,q,xh,model,s,T(-9.81));
        check_scratch(scratch,offset,FRAME_SCRATCH,bad);
        for(int i=threadIdx.x;i<grid::XHOM_T_COUNT+8;i+=blockDim.x)
            if((i<offset || i>=offset+grid::XHOM_T_COUNT) && xbuf[i]!=T(321))atomicAdd(bad,1);
        __syncthreads();
    }
}
int main(int argc,char **argv) {
    if(argc!=4)return 1;
    std::vector<T> q(grid::NUM_POS),out(WIDTH*RESIDUES);
    FILE *f=fopen(argv[1],"rb"); if(!f || fread(q.data(),sizeof(T),q.size(),f)!=q.size())return 3; fclose(f);
    T *dq,*dout; int *bad; grid::robotModel<T> *model=nullptr;
    CK(grid::init_robotModel_checked<T>(&model));
    CK(cudaMalloc(&dq,q.size()*sizeof(T))); CK(cudaMalloc(&dout,out.size()*sizeof(T)));
    CK(cudaMalloc(&bad,sizeof(int))); CK(cudaMemset(bad,0,sizeof(int)));
    CK(cudaMemcpy(dq,q.data(),q.size()*sizeof(T),cudaMemcpyHostToDevice));
    CK(cudaMemset(dout,0xff,out.size()*sizeof(T)));
    probe<<<1,atoi(argv[3])>>>(dout,dq,model,bad); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    int errors; CK(cudaMemcpy(&errors,bad,sizeof(int),cudaMemcpyDeviceToHost)); if(errors)return 4;
    CK(cudaMemcpy(out.data(),dout,out.size()*sizeof(T),cudaMemcpyDeviceToHost));
    f=fopen(argv[2],"wb"); if(!f || fwrite(out.data(),sizeof(T),out.size(),f)!=out.size())return 5; fclose(f);
    CK(cudaFree(dq)); CK(cudaFree(dout)); CK(cudaFree(bad)); CK(grid::free_robotModel_checked<T>(model));
}

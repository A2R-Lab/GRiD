#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <vector>
#include <cmath>
using T = SCALAR;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); return 2; } } while(0)

int main() {
    constexpr int nq=grid::NUM_POS, nv=grid::NUM_VEL, nt=grid::NUM_MULTI_TARGETS;
    std::vector<T> q(nq), p(3*nt), j(3*nt*nv), shared_p(3*nt), shared_j(3*nt*nv);
    for (auto &x:q) { double v; if(scanf("%lf",&v)!=1) return 3; x=T(v); }
    // Taking both public registration paths instantiates all four casts,
    // including bench twins; no timing kernel is executed by this runner.
    CK(grid::init_grid_kernel_attr_multi_target_position_checked<T>());
    CK(grid::init_grid_kernel_attr_multi_target_position_gradient_checked<T>());
    grid::init_grid_kernel_attr_multi_target_position<T>();
    grid::init_grid_kernel_attr_multi_target_position_gradient<T>();
    CK(cudaGetLastError());
    auto *model=grid::init_robotModel<T>();
    T *dq=nullptr,*dp=nullptr,*dj=nullptr;
    unsigned char *ws=nullptr;
    const auto bytes=grid::GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>();
    if (!bytes || !grid::MULTI_TARGET_POSITION_GRADIENT_DEVICE_INLINE_WORKSPACE_BYTES<T,grid::TIER_MINIMAL>()) return 4;
    CK(cudaMalloc(&ws,bytes)); CK(cudaMalloc(&dq,nq*sizeof(T)));
    CK(cudaMalloc(&dp,p.size()*sizeof(T))); CK(cudaMalloc(&dj,j.size()*sizeof(T)));
    CK(cudaMemcpy(dq,q.data(),nq*sizeof(T),cudaMemcpyHostToDevice));
    // Register through the repaired public path, then exercise the ordinary
    // production kernels at shared and workspace-using minimal tiers.
    grid::multi_target_position_kernel<T,grid::TIER_SHARED><<<1,32,grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T,grid::TIER_SHARED>()>>>(dp,ws,dq,nq,model,1);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(shared_p.data(),dp,p.size()*sizeof(T),cudaMemcpyDeviceToHost));
    grid::multi_target_position_gradient_kernel<T,grid::TIER_SHARED><<<1,32,grid::MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T,grid::TIER_SHARED>()>>>(dj,ws,dq,nq,model,1);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(shared_j.data(),dj,j.size()*sizeof(T),cudaMemcpyDeviceToHost));
    grid::multi_target_position_kernel<T,grid::TIER_MINIMAL><<<1,32,grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T,grid::TIER_MINIMAL>()>>>(dp,ws,dq,nq,model,1);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(p.data(),dp,p.size()*sizeof(T),cudaMemcpyDeviceToHost));
    grid::multi_target_position_gradient_kernel<T,grid::TIER_MINIMAL><<<1,32,grid::MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T,grid::TIER_MINIMAL>()>>>(dj,ws,dq,nq,model,1);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(j.data(),dj,j.size()*sizeof(T),cudaMemcpyDeviceToHost));
    if(p!=shared_p || j!=shared_j) return 5;
    for(int t=0;t<nt;++t) {
        printf("%.17g %.17g %.17g\n",double(p[3*t]),double(p[3*t+1]),double(p[3*t+2]));
        for(int k=0;k<nv;++k) {
            int i=3*(t*nv+k);
            printf("%.17g %.17g %.17g\n",double(j[i]),double(j[i+1]),double(j[i+2]));
        }
    }
    CK(cudaFree(ws)); CK(cudaFree(dq)); CK(cudaFree(dp)); CK(cudaFree(dj));
    CK(grid::free_robotModel_checked<T>(model));
    return 0;
}

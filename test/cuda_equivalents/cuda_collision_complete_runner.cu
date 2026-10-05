// Complete sphere-model wiring gate. Prints every transformed sphere, not just
// the collision verdict, so an always-colliding approximation cannot hide bad FK.
#define GRID_HEADER
#include "grid.cuh"
#include <iostream>
#include <iomanip>
#include <vector>
namespace gc = grid_collision;
using T = float;
constexpr int NS = gc::NUM_COLLISION_SPHERES;
#define CK(expr) do { auto err=(expr); if(err!=cudaSuccess) { std::cerr<<cudaGetErrorString(err); return 2; } } while(0)

__global__ void evaluate(const T *q, const grid::robotModel<T> *m,
                         const gc::Sphere<T> *obstacle, T *out, int *free_out) {
    __shared__ T pos[3*NS], radii[NS];
    gc::Environment<T> env{obstacle, 1, nullptr, 0, nullptr, 0};
    bool free = gc::config_free<T>(q, m, env, pos, radii, nullptr);
    __syncthreads();
    for(int i=threadIdx.x;i<3*NS;i+=blockDim.x) out[i]=pos[i];
    if(threadIdx.x==0) *free_out=free;
}

int main(int argc, char **argv) {
    int threads = argc>1 ? std::atoi(argv[1]) : 32;
    std::vector<T> q(grid::NUM_POS), pos(3*NS);
    for(auto &x:q) if(!(std::cin>>x)) return 3;
    gc::Sphere<T> obs;
    if(!(std::cin>>obs.x>>obs.y>>obs.z>>obs.r)) return 3;
    auto *model=grid::init_robotModel<T>();
    T *dq,*dp; int *df; gc::Sphere<T> *dobs;
    CK(cudaMalloc(&dq,q.size()*sizeof(T))); CK(cudaMalloc(&dp,pos.size()*sizeof(T)));
    CK(cudaMalloc(&df,sizeof(int))); CK(cudaMalloc(&dobs,sizeof(obs)));
    CK(cudaMemcpy(dq,q.data(),q.size()*sizeof(T),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dobs,&obs,sizeof(obs),cudaMemcpyHostToDevice));
    int smem=grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
    CK(cudaFuncSetAttribute(evaluate,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    evaluate<<<1,threads,smem>>>(dq,model,dobs,dp,df);
    CK(cudaDeviceSynchronize());
    int free;
    CK(cudaMemcpy(&free,df,sizeof(int),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(pos.data(),dp,pos.size()*sizeof(T),cudaMemcpyDeviceToHost));
    std::cout<<free<<'\n'<<std::setprecision(10);
    for(auto x:pos) std::cout<<x<<' ';
    std::cout<<'\n';
    CK(cudaFree(dq)); CK(cudaFree(dp)); CK(cudaFree(df)); CK(cudaFree(dobs));
    CK(cudaFree(model));
    return 0;
}

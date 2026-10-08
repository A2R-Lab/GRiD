// Adversarial same-address collision hits and repeated block-uniform calls.
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
namespace gc = grid_collision;
using T = float;
constexpr int NS = gc::NUM_COLLISION_SPHERES;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); return 2; } } while(0)
__global__ void probe(const grid::robotModel<T>* model, int *errors) {
    __shared__ T q[grid::NUM_POS], pos[3*NS], radii[NS];
    __shared__ gc::Sphere<T> obstacle;
#if TWO_TIERS
    __shared__ T broad[3*NS], broad_r[NS];
    __shared__ int debug_count;
#endif
    for(int i=threadIdx.x;i<grid::NUM_POS;i+=blockDim.x) q[i]=0;
    for(int repeat=0;repeat<32;++repeat) {
        const bool many=(repeat%2)==0;
        if(threadIdx.x==0) {
            obstacle={many?0.0f:10000.0f,0,0,many?1000.0f:0.01f};
#if TWO_TIERS
            debug_count=-1;
#endif
        }
        __syncthreads();
        gc::Environment<T> env{&obstacle,1,nullptr,0,nullptr,0};
#if TWO_TIERS
        bool free=gc::config_free<T>(q,model,env,broad,broad_r,pos,radii,nullptr,&debug_count);
        // Only anchors 0 and 2 are non-adjacent. Self hits flag their 48
        // spheres; the middle link is excluded unless the environment hits it.
        if(debug_count != (many?NS:(SELF_HITS?2*NS/3:0))) atomicAdd(errors,1);
#else
        bool free=gc::config_free<T>(q,model,env,pos,radii,nullptr);
#endif
        if(free != !(many || SELF_HITS)) atomicAdd(errors,1);
        __syncthreads();
    }
}
int main(int argc,char **argv) {
    const int threads=argc>1?std::atoi(argv[1]):32;
    auto *model=grid::init_robotModel<T>();
    int *errors; CK(cudaMalloc(&errors,sizeof(int))); CK(cudaMemset(errors,0,sizeof(int)));
    int smem=grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
#if TWO_TIERS
    int bsmem=grid::MULTI_TARGET_POSITION_BROAD_DYNAMIC_SHARED_MEM_BYTES<T>();
    if(bsmem>smem) smem=bsmem;
#endif
    CK(cudaFuncSetAttribute(probe,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    probe<<<1,threads,smem>>>(model,errors);
    CK(cudaDeviceSynchronize());
    int n; CK(cudaMemcpy(&n,errors,sizeof(int),cudaMemcpyDeviceToHost));
    CK(cudaFree(errors)); CK(cudaFree(model));
    std::printf("errors=%d RESULT: %s\n",n,n?"FAIL":"PASS");
    return n?3:0;
}

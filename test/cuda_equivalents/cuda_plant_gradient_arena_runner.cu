// Standalone exact-size shared arena: no larger Hessian reservation to borrow.
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
using T=float;
using Arena=grid_plant::plant_step_gradient_arena<T>;
constexpr int N=grid::NUM_VEL, NX=grid::NUM_POS+N, COUNT=6*N*N, B=2;
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
__global__ void probe(T*out,const T*z,const grid::robotModel<T>*model,int*bad) {
    extern __shared__ __align__(16) unsigned char raw[];
    const int t=threadIdx.x,nt=blockDim.x,b=blockIdx.x;
    Arena a;
    if(!Arena::bind(raw,Arena::bytes(),a)) {if(t==0) atomicAdd(bad,1);return;}
    // Poison before every invocation so zero-initialized CUDA pages cannot hide
    // missing initialization. The raw gradient contract explicitly owns vaf.
    for(int repeat=0;repeat<2;repeat++) {
        for(int i=t;i<Arena::bytes();i+=nt) raw[i]=0xff;
        __syncthreads();
        for(int i=t;i<NX;i+=nt) a.s_x[i]=z[b*(NX+N)+i];
        for(int i=t;i<N;i+=nt) a.s_u[i]=z[b*(NX+N)+NX+i];
        for(int i=t;i<Arena::s_vaf_count;i+=nt) a.s_vaf[i]=0;
        __syncthreads();
        grid_plant::plant_step_gradient<T,grid::IntegratorType::EULER>(
            a.s_dAB,a.s_x,a.s_u,a.s_df_du,a.s_dc_du,a.s_vaf,a.s_Minv,a.s_qdd,
            a.s_q_orig,a.s_qd_orig,a.s_stage_grad_qdd,a.s_D_qdd_stage,
            a.s_dInt_q_6x6,a.s_dInt_v_6x6,a.s_XImats,a.s_topology_helpers,
            a.s_temp,nullptr,nullptr,model,T(-9.81),T(.01));
        __syncthreads();
        for(int i=t;i<COUNT;i+=nt) {
            if(repeat && out[b*COUNT+i]!=a.s_dAB[i]) atomicAdd(bad,1);
            out[b*COUNT+i]=a.s_dAB[i];
        }
        __syncthreads();
    }
}
int main(int argc,char**argv) {
    if(argc!=4)return 1;
    static_assert(grid::TOPOLOGY_HELPERS_COUNT>0,"fixture must exercise topology storage");
    std::vector<T> z(B*(NX+N)),out(B*COUNT);
    FILE*f=fopen(argv[1],"rb");if(!f||fread(z.data(),sizeof(T),z.size(),f)!=z.size())return 3;fclose(f);
    T*dz,*dout;int*bad;grid::robotModel<T>*model=nullptr;
    CK(cudaMalloc(&dz,z.size()*sizeof(T)));CK(cudaMalloc(&dout,out.size()*sizeof(T)));CK(cudaMalloc(&bad,sizeof(int)));
    CK(cudaMemcpy(dz,z.data(),z.size()*sizeof(T),cudaMemcpyHostToDevice));CK(cudaMemset(bad,0,sizeof(int)));
    CK(grid::init_robotModel_checked<T>(&model));
    CK(grid_plant::plant_step_gradient_check_shared_memory<T>());
    CK(cudaFuncSetAttribute(probe,cudaFuncAttributeMaxDynamicSharedMemorySize,Arena::bytes()));
    probe<<<B,atoi(argv[3]),Arena::bytes()>>>(dout,dz,model,bad);
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    int errors;CK(cudaMemcpy(&errors,bad,sizeof(int),cudaMemcpyDeviceToHost));if(errors)return 4;
    CK(cudaMemcpy(out.data(),dout,out.size()*sizeof(T),cudaMemcpyDeviceToHost));
    f=fopen(argv[2],"wb");if(!f||fwrite(out.data(),sizeof(T),out.size(),f)!=out.size())return 5;fclose(f);
    printf("exact_shared_bytes=%zu topology_ints=%d\n",Arena::bytes(),grid::TOPOLOGY_HELPERS_COUNT);
    CK(cudaFree(dz));CK(cudaFree(dout));CK(cudaFree(bad));CK(grid::free_robotModel_checked<T>(model));
}

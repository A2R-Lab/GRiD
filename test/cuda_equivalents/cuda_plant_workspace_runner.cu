// Composed Hessian and gradient, two invocations, live outer storage, no timing.
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <limits>
using T = TEST_SCALAR;
constexpr int N=grid::NUM_VEL, NX=grid::NUM_POS+N, NH=18*N*N*N, NJ=6*N*N, B=2;
#ifndef TEST_INTEGRATOR
#define TEST_INTEGRATOR EULER
#endif
#ifndef TEST_RUNTIME_TRANSFORM
#define TEST_RUNTIME_TRANSFORM 0
#endif
#ifndef TEST_CROSS_KERNEL_ROUNDOFF
#define TEST_CROSS_KERNEL_ROUNDOFF 0
#endif
// Brian approved this narrow cross-kernel criterion on 2026-10-30. It does
// not alter the independent oracle, repeatability or thread-count checks.
bool stock_matches(T a,T b) {
    if(!std::isfinite(a)||!std::isfinite(b)) return false;
    if(a==b) return true;
    if constexpr(TEST_CROSS_KERNEL_ROUNDOFF && sizeof(T)==sizeof(double))
        return std::abs(a-b)<=T(64)*std::numeric_limits<T>::epsilon()*std::max(std::abs(a),std::abs(b));
    return false;
}
constexpr auto IT=grid::IntegratorType::TEST_INTEGRATOR;
using L=grid_plant::plant_hessian_layout;
template<L layout> using Arena=grid_plant::plant_step_hessian_arena<T,layout>;
constexpr size_t GB=grid_plant::PLANT_STEP_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
template<L layout> constexpr size_t SH=64+(Arena<layout>::shared_bytes()>GB?Arena<layout>::shared_bytes():GB);
template<L layout> constexpr size_t SLOT=Arena<layout>::workspace_bytes()+128;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) {fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);} } while(0)
template<L layout>
__global__ void composed(T *out, unsigned char *workspace, const T *z,
                         const grid::robotModel<T>*model, int *bad, int poison) {
    extern __shared__ __align__(16) unsigned char raw[];
    __shared__ T xi[grid::DYNAMICS_XI_T_COUNT];
    __shared__ T temp[grid::FD_INNER_SMEM_BYTES<T,true>()/sizeof(T)];
    __shared__ T qdd[N];
    volatile T *vx=xi,*vt=temp,*vq=qdd;
    const int tid=threadIdx.x, nt=blockDim.x, b=blockIdx.x;
    const T *x=z+b*(NX+N), *u=x+NX;
    for(int i=tid;i<grid::DYNAMICS_XI_T_COUNT;i+=nt) vx[i]=13+i;
    for(int i=tid;i<sizeof(temp)/sizeof(T);i+=nt) vt[i]=27+i;
    for(int i=tid;i<N;i+=nt) vq[i]=41+i;
    for(int i=tid;i<64;i+=nt) raw[i]=0x5a;
    for(int i=tid;i<(SH<layout>-64)/sizeof(T);i+=nt) reinterpret_cast<T*>(raw+64)[i]=poison?T(NAN):0;
    __syncthreads();
    Arena<layout> a{};
    if(!Arena<layout>::bind(raw+64,SH<layout>-64,workspace+b*SLOT<layout>+64,Arena<layout>::workspace_bytes(),a)) {
        if(tid==0) atomicAdd(bad,1); return;
    }
    for(int i=tid;i<NX;i+=nt) a.s_x[i]=x[i];
    for(int i=tid;i<N;i+=nt) a.s_u[i]=u[i];
    __syncthreads();
    a.template evaluate<IT>(model,T(-9.81),T(.01));
    __syncthreads();
    for(int i=tid;i<NH;i+=nt) out[b*(NH+NJ+NX)+i]=a.s_d2AB[i];
    for(int i=tid;i<grid::DYNAMICS_XI_T_COUNT;i+=nt) if(vx[i]!=13+i) atomicAdd(bad,1);
    for(int i=tid;i<sizeof(temp)/sizeof(T);i+=nt) if(vt[i]!=27+i) atomicAdd(bad,1);
    for(int i=tid;i<N;i+=nt) if(vq[i]!=41+i) atomicAdd(bad,1);
    __syncthreads();
    // Sequential reuse. Raw gradient contract requires initializing its vaf band.
    for(int i=tid;i<GB/sizeof(T);i+=nt) reinterpret_cast<T*>(raw+64)[i]=poison?T(NAN):0;
    __syncthreads();
    grid_plant::plant_step_gradient_arena<T> g;
    if(!decltype(g)::bind(raw+64,GB,g)) {if(tid==0) atomicAdd(bad,1); return;}
    for(int i=tid;i<NX;i+=nt) g.s_x[i]=x[i];
    for(int i=tid;i<N;i+=nt) g.s_u[i]=u[i];
    for(int i=tid;i<decltype(g)::s_vaf_count;i+=nt) g.s_vaf[i]=0;
    __syncthreads();
    grid_plant::plant_step_gradient<T,IT>(g.s_dAB,g.s_x,g.s_u,g.s_df_du,g.s_dc_du,g.s_vaf,g.s_Minv,g.s_qdd,
        g.s_q_orig,g.s_qd_orig,g.s_stage_grad_qdd,g.s_D_qdd_stage,g.s_dInt_q_6x6,g.s_dInt_v_6x6,
        g.s_XImats,g.s_topology_helpers,g.s_temp,nullptr,nullptr,model,T(-9.81),T(.01));
    __syncthreads();
    for(int i=tid;i<NJ;i+=nt) out[b*(NH+NJ+NX)+NH+i]=g.s_dAB[i];
    for(int i=tid;i<64;i+=nt) if(raw[i]!=0x5a) atomicAdd(bad,1);
    __syncthreads();
#if TEST_RUNTIME_TRANSFORM
    grid::load_update_XImats_helpers<T,false,false,true>(xi,x,g.s_topology_helpers,model,temp);
#else
    grid::load_update_XImats_helpers<T>(xi,x,g.s_topology_helpers,model,temp);
#endif
    __syncthreads();
    grid::forward_dynamics_inner<T,true>(qdd,x,x+N,u,xi,g.s_topology_helpers,temp,nullptr,nullptr,T(-9.81));
    __syncthreads();
    for(int i=tid;i<N;i+=nt) {
        out[b*(NH+NJ+NX)+NH+NJ+i]=x[i]+T(.01)*(x[N+i]+(IT==grid::IntegratorType::SEMI_IMPLICIT_EULER?T(.01)*qdd[i]:T(0)));
        out[b*(NH+NJ+NX)+NH+NJ+N+i]=x[N+i]+T(.01)*qdd[i];
    }
}
template<L layout> int run(const char *input,const char *output,int threads) {
    using A=Arena<layout>;
    if(!stock_matches(T(1),T(1)) || stock_matches(T(NAN),T(NAN)) ||
       stock_matches(T(INFINITY),T(INFINITY)) || stock_matches(T(0),T(1e-20)) ||
       stock_matches(T(1),T(1.001))) return 15;
    if constexpr(TEST_CROSS_KERNEL_ROUNDOFF && sizeof(T)==sizeof(double)) {
        if(!stock_matches(T(1),std::nextafter(T(1),T(2))) || stock_matches(T(1),T(1)+T(128)*std::numeric_limits<T>::epsilon())) return 15;
    }
    std::vector<T> z(B*(NX+N)),values(B*(NH+NJ+NX)),first;
    FILE*f=fopen(input,"rb"); if(!f||fread(z.data(),sizeof(T),z.size(),f)!=z.size()) return 3; fclose(f);
    T *dz,*out,*stock,*stockx,*stocku; unsigned char *ws,*stockws; int *bad;
    CK(cudaMalloc(&dz,z.size()*sizeof(T)));CK(cudaMalloc(&out,values.size()*sizeof(T)));
    CK(cudaMalloc(&ws,B*SLOT<layout>));CK(cudaMalloc(&bad,sizeof(int)));
    CK(cudaMemcpy(dz,z.data(),z.size()*sizeof(T),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&stock,B*NH*sizeof(T))); CK(cudaMalloc(&stockx,B*NX*sizeof(T))); CK(cudaMalloc(&stocku,B*N*sizeof(T)));
    constexpr size_t stock_stride=grid_plant::PLANT_HESSIAN_WORKSPACE_BYTES_PER_TIMESTEP<T>();
    constexpr size_t stock_sh=grid_plant::INTEGRATOR_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>();
    CK(cudaMalloc(&stockws,B*stock_stride));
    for(int b=0;b<B;b++) {
        CK(cudaMemcpy(stockx+b*NX,z.data()+b*(NX+N),NX*sizeof(T),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(stocku+b*N,z.data()+b*(NX+N)+NX,N*sizeof(T),cudaMemcpyHostToDevice));
    }
    grid::robotModel<T>*model=nullptr;CK(grid::init_robotModel_checked<T>(&model));
    // Pure host views: exercise refusal without ever launching invalid storage.
    alignas(16) unsigned char host[16];
    A a{}; a.s_x=reinterpret_cast<T*>(host); A original=a;
    if(A::bind(nullptr,A::shared_bytes(),ws,A::workspace_bytes(),a) ||
       A::bind(host,A::shared_bytes()-1,ws,A::workspace_bytes(),a) ||
       A::bind(host+1,A::shared_bytes(),ws,A::workspace_bytes(),a) || a.s_x!=original.s_x) return 4;
    if constexpr(layout==L::deep_spill) {
        if(A::bind(host,A::shared_bytes(),nullptr,A::workspace_bytes(),a) ||
           A::bind(host,A::shared_bytes(),ws,A::workspace_bytes()-1,a) ||
           A::bind(host,A::shared_bytes(),ws+1,A::workspace_bytes(),a)) return 5;
    }
    // Backed host storage checks the complete pointer extent (no fake allocation).
    void *host_arena=nullptr; if(posix_memalign(&host_arena,16,A::shared_bytes())) return 6;
    if(!A::bind(host_arena,A::shared_bytes(),ws,A::workspace_bytes(),a)) return 7;
    free(host_arena);
    cudaFuncAttributes attr;CK(cudaFuncGetAttributes(&attr,composed<layout>));
    int limit;CK(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,0));
    if(SH<layout>+attr.sharedSizeBytes>size_t(limit)) return 8;
    CK(cudaFuncSetAttribute(composed<layout>,cudaFuncAttributeMaxDynamicSharedMemorySize,SH<layout>));
    CK(cudaFuncSetAttribute(grid_plant::plant_step_hessian_kernel<T,IT>,cudaFuncAttributeMaxDynamicSharedMemorySize,stock_sh));
    printf("shared=%zu workspace=%zu outer_dynamic=64 static=%zu regs=%d local=%zu limit=%d\n",
           SH<layout>,A::workspace_bytes(),attr.sharedSizeBytes,attr.numRegs,attr.localSizeBytes,limit);
    for(int repeat=0;repeat<3;repeat++) {
        if(repeat<2) {
            std::vector<unsigned char> bytes(B*SLOT<layout>,0xa5);
            std::vector<T> poison(A::workspace_bytes()/sizeof(T),repeat?T(NAN):T(0));
            for(int b=0;b<B;b++) memcpy(bytes.data()+b*SLOT<layout>+64,poison.data(),A::workspace_bytes());
            CK(cudaMemcpy(ws,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
        }
        CK(cudaMemset(bad,0,sizeof(int)));CK(cudaMemset(out,0xff,values.size()*sizeof(T)));
        composed<layout><<<B,threads,(SH<layout>)>>>(out,ws,dz,model,bad,repeat!=0);
        CK(cudaGetLastError());CK(cudaDeviceSynchronize());
        int errors;CK(cudaMemcpy(&errors,bad,sizeof(int),cudaMemcpyDeviceToHost));if(errors) return 9;
        std::vector<unsigned char> bytes(B*SLOT<layout>);CK(cudaMemcpy(bytes.data(),ws,bytes.size(),cudaMemcpyDeviceToHost));
        for(int b=0;b<B;b++) for(int i=0;i<64;i++) if(bytes[b*SLOT<layout>+i]!=0xa5 || bytes[b*SLOT<layout>+64+A::workspace_bytes()+i]!=0xa5) return 10;
        CK(cudaMemcpy(values.data(),out,values.size()*sizeof(T),cudaMemcpyDeviceToHost));
        // The stock small-robot kernel is launch-bounded (acrobot: 64).
        // The composed caller still executes every requested size, including 256.
        const int stock_threads=threads<grid::MAX_PERF_LEVEL_THREADS?threads:grid::MAX_PERF_LEVEL_THREADS;
        grid_plant::plant_step_hessian_kernel<T,IT><<<B,stock_threads,stock_sh>>>(stock,stockws,stockx,stocku,NX,N,model,T(-9.81),T(.01),B);
        CK(cudaGetLastError());CK(cudaDeviceSynchronize());
        std::vector<T> original(B*NH);CK(cudaMemcpy(original.data(),stock,original.size()*sizeof(T),cudaMemcpyDeviceToHost));
        size_t different=0,rejected=0;double max_difference=0;
        for(T v:original) if(!std::isfinite(v)) return 11;
        for(T v:values) if(!std::isfinite(v)) return 11;
        for(int b=0;b<B;b++) for(int i=0;i<NH;i++) if(original[b*NH+i]!=values[b*(NH+NJ+NX)+i]) {
            if(different==0)fprintf(stderr,"stock mismatch block=%d index=%d stock=%.17g composed=%.17g delta=%.17g\n",b,i,
                double(original[b*NH+i]),double(values[b*(NH+NJ+NX)+i]),double(original[b*NH+i]-values[b*(NH+NJ+NX)+i]));
            different++;max_difference=std::max(max_difference,std::abs(double(original[b*NH+i]-values[b*(NH+NJ+NX)+i])));
            if(!stock_matches(original[b*NH+i],values[b*(NH+NJ+NX)+i])) rejected++;
        }
        if(different){
            fprintf(stderr,"cross-kernel comparison: different=%zu rejected=%zu max_abs=%.17g roundoff_enabled=%d\n",different,rejected,max_difference,TEST_CROSS_KERNEL_ROUNDOFF);
            if(rejected){
                f=fopen(output,"wb");if(f){fwrite(values.data(),sizeof(T),values.size(),f);fclose(f);}
                return 14;
            }
        }
        for(T v:values) if(!std::isfinite(v)) return 11;
        if(repeat && values!=first) return 12; first=values;
    }
    f=fopen(output,"wb");if(!f || fwrite(values.data(),sizeof(T),values.size(),f)!=values.size()) return 13; fclose(f);
    CK(cudaFree(dz));CK(cudaFree(out));CK(cudaFree(ws));CK(cudaFree(bad));CK(grid::free_robotModel_checked<T>(model));
    CK(cudaFree(stock));CK(cudaFree(stockx));CK(cudaFree(stocku));CK(cudaFree(stockws));
    return 0;
}
int main(int argc,char**argv) {
    if(argc!=5) return 1;
    return atoi(argv[4]) ? run<L::deep_spill>(argv[1],argv[2],atoi(argv[3])) : run<L::full_shared>(argv[1],argv[2],atoi(argv[3]));
}

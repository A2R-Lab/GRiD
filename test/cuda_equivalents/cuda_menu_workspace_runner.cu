#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
using T=TEST_SCALAR;
constexpr int NQ=grid::NUM_POS,NV=grid::NUM_VEL,NX=NQ+NV,TN=2*NV;
constexpr int NU=grid_plant::GRID_PLANT_MENU_NU,NLX=grid_plant::GRID_PLANT_MENU_NLX;
constexpr int COUNT=1+TN+NU+TN*TN+NU*NU,B=2;
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);}}while(0)
template<int MX,int MU> using Arena=grid_plant::tracking_cost_menu_arena<T,MX,MU>;
template<int MX,int MU>
__global__ void probe(T *out,int *bad,const T *data,const grid::robotModel<T>*model,int poison) {
    using A=Arena<MX,MU>;
    extern __shared__ __align__(16) unsigned char raw[];
    __shared__ T x[NX],u[NU];
    __shared__ T lower[NX],upper[NX];
    __shared__ int activation[grid_plant::GRID_PLANT_MENU_TERMS];
    __shared__ T eps[grid_plant::GRID_PLANT_MENU_TERMS];
    int tid=threadIdx.x,nt=blockDim.x,b=blockIdx.x;
    for(int i=tid;i<64;i+=nt) raw[i]=raw[64+A::bytes()+i]=0x5a;
    for(int i=tid;i<A::bytes()/sizeof(T);i+=nt) reinterpret_cast<T*>(raw+64)[i]=poison?T(NAN):0;
    for(int i=tid;i<NX;i+=nt) x[i]=T(.01*(i+1)+.03*b);
    for(int i=tid;i<NX;i+=nt) {lower[i]=-INFINITY;upper[i]=INFINITY;}
    __syncthreads();
    if constexpr(NQ!=NV) {if(tid==0) {x[3]=x[4]=x[5]=0;x[6]=1;}}
    for(int i=tid;i<NU;i+=nt) u[i]=T(-.02*(i+1)+.04*b);
    for(int i=tid;i<grid_plant::GRID_PLANT_MENU_TERMS;i+=nt) {activation[i]=-1;eps[i]=0;}
    __syncthreads();
    if(tid==0) {
        if(MX) activation[grid_plant::GRID_PLANT_MENU_LINEAR_X]=grid_plant::GRID_PLANT_ACTIVATION_QUADRATIC;
        if(MU) activation[grid_plant::GRID_PLANT_MENU_LINEAR_U]=grid_plant::GRID_PLANT_ACTIVATION_QUADRATIC;
    }
    __syncthreads();
    A a{}; a.s_frame=reinterpret_cast<T*>(raw);
    if(A::bind(raw+65,A::bytes(),a) || A::bind(raw+64,A::bytes()-1,a) ||
       A::bind(nullptr,A::bytes(),a) || a.s_frame!=reinterpret_cast<T*>(raw)) {if(tid==0) atomicAdd(bad,1);return;}
    if(!A::bind(raw+64,A::bytes(),a)) {if(tid==0) atomicAdd(bad,1);return;}
    const A unchecked=A::carve(raw+64);
    if(a.s_frame!=unchecked.s_frame || a.s_scratch!=unchecked.s_scratch) {if(tid==0) atomicAdd(bad,1);return;}
    grid_plant::cost_menu_params<T> p{};
    p.activation=activation;p.eps=eps;
    p.q_lower=p.qd_lower=p.u_lower=lower;p.q_upper=p.qd_upper=p.u_upper=upper;
    p.Ax=data;p.bx=p.Ax+MX*NLX;p.tx=p.bx+MX;p.Wx=p.tx+MX;
    p.Au=p.Wx+MX;p.bu=p.Au+MU*NU;p.tu=p.bu+MU;p.Wu=p.tu+MU;
    T *v=out+b*COUNT,*gx=v+1,*gu=gx+TN,*hx=gu+NU,*hu=hx+TN*TN;
    grid_plant::tracking_cost_menu<T,MX,MU>(v,x,u,p,a.s_frame,a.s_scratch,model);
    __syncthreads();
    grid_plant::tracking_cost_menu_gradient<T,MX,MU>(gx,gu,x,u,p,a.s_frame,a.s_scratch,model);
    __syncthreads();
    grid_plant::tracking_cost_menu_hessian<T,MX,MU>(hx,hu,x,u,p,a.s_frame,a.s_scratch,model);
    __syncthreads();
    // Reuse with ACCUMULATE=true: result must be exactly twice the first value.
    grid_plant::tracking_cost_menu<T,MX,MU,0,-1,true>(v,x,u,p,a.s_frame,a.s_scratch,model);
    __syncthreads();
    grid_plant::tracking_cost_menu_gradient<T,MX,MU,0,-1,true>(gx,gu,x,u,p,a.s_frame,a.s_scratch,model);
    __syncthreads();
    grid_plant::tracking_cost_menu_hessian<T,MX,MU,0,-1,true>(hx,hu,x,u,p,a.s_frame,a.s_scratch,model);
    __syncthreads();
    for(int i=tid;i<64;i+=nt) if(raw[i]!=0x5a || raw[64+A::bytes()+i]!=0x5a) atomicAdd(bad,1);
}
template<int MX,int MU> int run(const char*input,const char*output,int threads) {
    using A=Arena<MX,MU>;
    constexpr size_t count=MX*(NLX+3)+MU*(NU+3);
    std::vector<T> data(count),result(B*COUNT),first;
    FILE*f=fopen(input,"rb");if(!f || fread(data.data(),sizeof(T),count,f)!=count)return 3;fclose(f);
    T *d,*out;int *bad;
    CK(cudaMalloc(&d,(count?count:1)*sizeof(T)));CK(cudaMalloc(&out,result.size()*sizeof(T)));CK(cudaMalloc(&bad,sizeof(int)));
    if(count) CK(cudaMemcpy(d,data.data(),count*sizeof(T),cudaMemcpyHostToDevice));
    grid::robotModel<T>*model=nullptr;CK(grid::init_robotModel_checked<T>(&model));
    constexpr size_t bytes=A::bytes()+128;
    cudaFuncAttributes attr;CK(cudaFuncGetAttributes(&attr,probe<MX,MU>));
    int cap;CK(cudaDeviceGetAttribute(&cap,cudaDevAttrMaxSharedMemoryPerBlockOptin,0));
    if(bytes+attr.sharedSizeBytes>size_t(cap)) return 4;
    CK(cudaFuncSetAttribute(probe<MX,MU>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    printf("menu=%zu scratch=%zu offset=%zu static=%zu registers=%d local=%zu\n",A::bytes(),A::scratch_bytes(),A::scratch_offset(),attr.sharedSizeBytes,attr.numRegs,attr.localSizeBytes);
    alignas(16) unsigned char dummy[16]; A view{};view.s_frame=reinterpret_cast<T*>(dummy);
    if(A::bind(dummy+1,A::bytes(),view) || A::bind(dummy,A::bytes()-1,view) || A::bind(nullptr,A::bytes(),view) || view.s_frame!=reinterpret_cast<T*>(dummy)) return 5;
    for(int repeat=0;repeat<3;repeat++) {
        CK(cudaMemset(out,0xff,result.size()*sizeof(T)));CK(cudaMemset(bad,0,sizeof(int)));
        probe<MX,MU><<<B,threads,bytes>>>(out,bad,d,model,repeat!=0);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
        int errors;CK(cudaMemcpy(&errors,bad,sizeof(int),cudaMemcpyDeviceToHost));if(errors)return 6;
        CK(cudaMemcpy(result.data(),out,result.size()*sizeof(T),cudaMemcpyDeviceToHost));
        for(T v:result)if(!std::isfinite(v))return 7;
        if(repeat && result!=first)return 8;first=result;
    }
    f=fopen(output,"wb");if(!f||fwrite(result.data(),sizeof(T),result.size(),f)!=result.size())return 9;fclose(f);
    CK(cudaFree(d));CK(cudaFree(out));CK(cudaFree(bad));CK(grid::free_robotModel_checked<T>(model));return 0;
}
int main(int argc,char**argv) {
    if(argc!=5)return 1;
    switch(atoi(argv[4])) {
        case 0:return run<0,0>(argv[1],argv[2],atoi(argv[3]));
        case 1:return run<2*NV-6*(NQ!=NV),grid_plant::GRID_PLANT_MENU_NU>(argv[1],argv[2],atoi(argv[3]));
        case 2:return run<4097,0>(argv[1],argv[2],atoi(argv[3]));
        case 3:return run<0,4097>(argv[1],argv[2],atoi(argv[3]));
        default:return 10;
    }
}

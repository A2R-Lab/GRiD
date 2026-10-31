// Two independent warp anchors, divergent scalar overrides and suffix reuse.
#include "grid.cuh"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
using T=TEST_SCALAR;
constexpr int N=grid::model::nj, FRAME=TEST_FRAME;
#define CK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}while(0)
__device__ __forceinline__ void suffix(T*out,const T*world,const T*local,int changed,T value) {
    T W[16], next[16], joint[16];
    for(int k=0;k<16;++k) W[k]=changed?world[16*(changed-1)+k]:T(k%5==0);
    for(int j=changed;j<N;++j) {
        const T* transform;
        if(j==changed) {
#ifdef TEST_LEGACY
            grid::update_XmatHom_joint<T>(joint,local,j,value);
#else
            if(!grid::model::joint_transform_scalar<T>(j,value,joint)) asm("trap;");
#endif
            transform=joint;
        } else transform=local+16*j;
        grid::glass::thread::gemm<T,4,4,4>(T(1),W,transform,next);
        for(int k=0;k<16;++k) W[k]=next[k];
    }
    if(!grid::model::compose_frame_from_body(FRAME,N-1,W,out)) asm("trap;");
}
__global__ void probe(const T*input,T*output) {
    __shared__ T anchors[2*32*N];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    T* a=anchors+warp*32*N;
    for(int repeat=0;repeat<2;++repeat) {
        if(lane==0)for(int k=0;k<32*N;++k)a[k]=input[warp*32*N+k];
        __syncwarp();
        if(lane<N) {
            T q=T(-.15)+T(.35)*T(lane)/T(N-1);
            if(lane==warp)q+=T(.1);
            q+=T(.025)*T(lane+1);
            T out[16];suffix(out,a,a+16*N,lane,q);
            for(int k=0;k<16;++k)output[(repeat*64+threadIdx.x)*16+k]=out[k];
        }
        __syncwarp(); // readers finish before lane 0 reuses its anchor
    }
}
int main(int argc,char**argv) {
    if(argc!=3)return 1;
    std::vector<T> input(2*32*N),out(2*64*16,789);
    FILE*f=fopen(argv[1],"rb");if(!f||fread(input.data(),sizeof(T),input.size(),f)!=input.size())return 3;fclose(f);
    T*a,*b;CK(cudaMalloc(&a,input.size()*sizeof(T)));CK(cudaMalloc(&b,out.size()*sizeof(T)));
    CK(cudaMemcpy(a,input.data(),input.size()*sizeof(T),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(b,out.data(),out.size()*sizeof(T),cudaMemcpyHostToDevice));
    probe<<<1,64>>>(a,b);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out.data(),b,out.size()*sizeof(T),cudaMemcpyDeviceToHost));
    cudaFuncAttributes attr{};CK(cudaFuncGetAttributes(&attr,probe));
    printf("registers=%d local_bytes=%zu shared_bytes=%zu\n",attr.numRegs,attr.localSizeBytes,attr.sharedSizeBytes);
    f=fopen(argv[2],"wb");if(!f||fwrite(out.data(),sizeof(T),out.size(),f)!=out.size())return 4;fclose(f);
    CK(cudaFree(a));CK(cudaFree(b));
}

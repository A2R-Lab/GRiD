#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
using T=TEST_SCALAR;
constexpr int Q=grid::model::nq,N=grid::model::nv,J=grid::model::nj;
constexpr int WIDTH=N+2*N*N+32*J, SAMPLES=35;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(2);} } while(0)
__global__ void probe(const T *input,T *output,int *errors) {
    // Divergent threads, non-warp-aligned batch, separate blocks/output slots.
    int sample=blockIdx.x*blockDim.x+threadIdx.x;
    if(sample>=SAMPLES) return;
    const T *a=Q?input+sample*2*Q:nullptr,*b=Q?a+Q:nullptr;
    T *out=output+sample*(WIDTH+2)+1;
    grid::configuration_difference(a,b,out);
    grid::configuration_difference_jacobians(a,b,out+N,out+N+N*N);
    for(int k=0;k<grid::model::coordinate_block_count;++k) {
        grid::model::coordinate_block_record rec{}; grid::model::coordinate_block(k,rec);
        T A[38],B[38]; for(int i=0;i<38;++i) A[i]=B[i]=T(321);
        if(grid::configuration_difference_jacobian_block(k,a,b,A+1,B+1,rec.nv*rec.nv-1)) { if(sample==2) printf("short admitted %d\n",k); atomicAdd(errors,1); }
        for(int i=0;i<38;++i) if(A[i]!=T(321)||B[i]!=T(321)) { if(sample==2) printf("short write %d %d\n",k,i); atomicAdd(errors,1); }
        if(!grid::configuration_difference_jacobian_block(k,a,b,A+1,B+1,rec.nv*rec.nv)) { if(sample==2) printf("valid refused %d\n",k); atomicAdd(errors,1); }
        for(int c=0;c<rec.nv;++c) for(int r=0;r<rec.nv;++r) {
            int dense=(rec.v_start+c)*N+rec.v_start+r;
            if(A[1+c*rec.nv+r]!=out[N+dense]||B[1+c*rec.nv+r]!=out[N+N*N+dense]) {
                printf("compact/dense block=%d r=%d c=%d A=%.17g dense=%.17g B=%.17g dense=%.17g\n",
                    k,r,c,double(A[1+c*rec.nv+r]),double(out[N+dense]),
                    double(B[1+c*rec.nv+r]),double(out[N+N*N+dense]));
                atomicAdd(errors,1);
            }
        }
        for(int i=0;i<38;++i) if((i==0||i>rec.nv*rec.nv)&&(A[i]!=T(321)||B[i]!=T(321))) { if(sample==2) printf("overrun %d %d\n",k,i); atomicAdd(errors,1); }
    }
    T invalid[16]; for(int i=0;i<16;++i) invalid[i]=T(321);
    if(grid::configuration_difference_jacobian_block(-1,a,b,invalid,invalid+8,0) ||
       grid::model::joint_transform(-1,a,invalid) ||
       grid::model::joint_transform_scalar(J,T(.2),invalid)) atomicAdd(errors,1);
    for(int i=0;i<16;++i) if(invalid[i]!=T(321)) { if(sample==2) printf("invalid write %d\n",i); atomicAdd(errors,1); }
    for(int slot=0;slot<J;++slot) {
        const int j=(slot+sample)%J; // Distinct runtime IDs across lanes.
        auto dest=out+N+2*N*N+32*j;
        grid::model::joint_record rec{}; grid::model::joint(j,rec);
        bool full=grid::model::joint_transform(j,a,dest);
        bool allowed=!grid::model::runtime_transform && rec.owns_coordinates &&
            (rec.kind==grid::model::joint_kind::revolute || rec.kind==grid::model::joint_kind::continuous ||
             rec.kind==grid::model::joint_kind::prismatic);
        T scalar=allowed?a[rec.q_start]:T(0);
        if(grid::model::joint_transform_scalar(j,scalar,dest+16)!=allowed) { if(sample==2) printf("scalar admission %d\n",j); atomicAdd(errors,1); }
        if(full==grid::model::runtime_transform) { if(sample==2) printf("full admission %d\n",j); atomicAdd(errors,1); }
    }
}
int main(int argc,char**argv) {
    if(argc!=4) return 1;
    int threads=atoi(argv[3]); std::vector<T> input(SAMPLES*2*Q),out(SAMPLES*(WIDTH+2));
    FILE*f=fopen(argv[1],"rb"); if(!f||fread(input.data(),sizeof(T),input.size(),f)!=input.size()) return 3;fclose(f);
    T *din,*dout;int *bad;
    CK(cudaMalloc(&din,std::max(size_t(1),input.size())*sizeof(T))); CK(cudaMalloc(&dout,out.size()*sizeof(T)));CK(cudaMalloc(&bad,sizeof(int)));
    CK(cudaMemcpy(din,input.data(),input.size()*sizeof(T),cudaMemcpyHostToDevice));
    std::vector<T> first;
    for(int repeat=0;repeat<2;++repeat) {
        std::fill(out.begin(),out.end(),T(789));CK(cudaMemcpy(dout,out.data(),out.size()*sizeof(T),cudaMemcpyHostToDevice));CK(cudaMemset(bad,0,sizeof(int)));
        probe<<<(SAMPLES+threads-1)/threads,threads>>>(din,dout,bad);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
        int nbad; CK(cudaMemcpy(&nbad,bad,sizeof(int),cudaMemcpyDeviceToHost));if(nbad){fprintf(stderr,"contract errors: %d\n",nbad);return 4;}
        CK(cudaMemcpy(out.data(),dout,out.size()*sizeof(T),cudaMemcpyDeviceToHost));
        for(int s=0;s<SAMPLES;++s) if(out[s*(WIDTH+2)]!=T(789)||out[(s+1)*(WIDTH+2)-1]!=T(789)) return 5;
        if(repeat && out!=first) return 8;
        first=out;
    }
    f=fopen(argv[2],"wb"); if(!f||fwrite(out.data(),sizeof(T),out.size(),f)!=out.size()) return 6;fclose(f);
    CK(cudaFree(din));CK(cudaFree(dout));CK(cudaFree(bad));
}

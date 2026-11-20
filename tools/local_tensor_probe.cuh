// Private CPU-study companion, not included by production generated headers.
// One block, one directional pair at a time. No atomics, double arithmetic,
// timing events, global scratch arena, or public GRiD layout changes.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

static void checked(cudaError_t error) {
    if (error != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(error)); exit(2); }
}

// Generator supplies NB/NV/NQ, PARENT/SLOT/DIM/ALPHA/S/INERTIA and transforms.
__device__ float crm_entry(const float* v, int r, int c) {
    if (r < 3 && c >= 3) return 0.0f;
    int rr = r%3, cc = c%3;
    if (rr == cc) return 0.0f;
    int offset = (r >= 3 && c < 3) ? 3 : 0;
    if (rr == 0 && cc == 1) return -v[offset+2];
    if (rr == 0 && cc == 2) return v[offset+1];
    if (rr == 1 && cc == 0) return v[offset+2];
    if (rr == 1 && cc == 2) return -v[offset];
    if (rr == 2 && cc == 0) return -v[offset+1];
    return v[offset];
}

__device__ float matrix_row(const float* m, const float* v, int row, bool trans=false) {
    float value = 0.0f;
    for (int k=0;k<6;++k) value += m[trans ? k*6+row : row*6+k]*v[k];
    return value;
}

__device__ float cross_row(const float* v, const float* w, int row, bool dual=false) {
    float value = 0.0f;
    for (int k=0;k<6;++k) value += (dual ? -crm_entry(v,k,row) : crm_entry(v,row,k))*w[k];
    return value;
}

__device__ float transformed(const float* x, const float* dx, const float* ddx,
                            const float* jets, int part, int row,
                            bool dj, bool dk, bool trans=false) {
    float value = matrix_row(x,jets+part*6,row,trans);
    if (part==1 && dj) value += matrix_row(dx,jets,row,trans);
    if (part==2 && dk) value += matrix_row(dx,jets,row,trans);
    if (part==3) {
        if (dj) value += matrix_row(dx,jets+12,row,trans);
        if (dk) value += matrix_row(dx,jets+6,row,trans);
        if (dj && dk) value += matrix_row(ddx,jets,row,trans);
    }
    return value;
}

__device__ float crossed(const float* v, const float* w, int part, int row, bool dual=false) {
    if (part==0) return cross_row(v,w,row,dual);
    if (part==1 || part==2)
        return cross_row(v+part*6,w,row,dual)+cross_row(v,w+part*6,row,dual);
    return ((cross_row(v+18,w,row,dual)+cross_row(v+6,w+12,row,dual))
             +cross_row(v+12,w+6,row,dual))+cross_row(v,w+18,row,dual);
}

__global__ void local_tensor_probe(const float* inputs, float* output) {
    __shared__ float q[NQ], x[NB*36], dx[NB*36], ddx[NB*36];
    __shared__ float v[NB*24], a[NB*24], force[NB*24];
    __shared__ float vj[24], aj[24], momentum[24], gravity[24];
    const float* qd=inputs+NQ; const float* qdd=qd+NV;
    for (int i=threadIdx.x;i<NQ;i+=blockDim.x) q[i]=inputs[i];
    __syncthreads();
    if (threadIdx.x==0) {
        float norm=sqrtf(((q[3]*q[3]+q[4]*q[4])+q[5]*q[5])+q[6]*q[6]);
        for(int i=3;i<7;++i) q[i]/=norm;
    }
    __syncthreads();
    for(int b=0;b<NB;++b) {
        if(threadIdx.x==0) evaluate_transform(b,q,x+36*b);
        __syncthreads();
        for(int z=threadIdx.x;z<36;z+=blockDim.x) {
            int r=z/6,c=z%6; float value=0.0f;
            if(b>0) for(int l=0;l<6;++l)
                value += -ALPHA[b]*crm_entry(S+36*b,r,l)*x[36*b+l*6+c];
            dx[36*b+z]=value;
        }
        __syncthreads();
        for(int z=threadIdx.x;z<36;z+=blockDim.x) {
            int r=z/6,c=z%6; float value=0.0f;
            if(b>0) for(int l=0;l<6;++l)
                value += -ALPHA[b]*crm_entry(S+36*b,r,l)*dx[36*b+l*6+c];
            ddx[36*b+z]=value;
        }
        __syncthreads();
    }
    for(int block=0;block<4;++block) for(int j=0;j<NV;++j) for(int k=0;k<NV;++k) {
        int jk=(block==0 ? 0 : (block==3 ? 2 : 1));
        int kk=(block==1 ? 1 : 0);
        // Root gravity uses the local body orientation, independent of translation.
        if(threadIdx.x==0) {
            for(int z=0;z<24;++z) gravity[z]=0.0f;
            for(int r=0;r<3;++r) gravity[r+3]=9.81f*x[2*6+r];
            float bj[6],bk[6],firstk[6];
            for(int r=0;r<6;++r) {
                bj[r]=(j<6 && jk==0) ? (j<3 ? -1.0f : 1.0f)*S[j*6+r] : 0.0f;
                bk[r]=(k<6 && kk==0) ? (k<3 ? -1.0f : 1.0f)*S[k*6+r] : 0.0f;
            }
            for(int r=0;r<6;++r) {
                gravity[6+r]=-cross_row(bj,gravity,r);
                gravity[12+r]=-cross_row(bk,gravity,r);
                firstk[r]=cross_row(bk,gravity,r);
            }
            for(int r=0;r<6;++r) gravity[18+r]=cross_row(bj,firstk,r);
        }
        __syncthreads();
        for(int b=0;b<NB;++b) {
            int p=PARENT[b];
            for(int z=threadIdx.x;z<24;z+=blockDim.x) {
                int part=z/6,r=z%6; float vv=0.0f,aa=0.0f;
                for(int l=0;l<DIM[b];++l) {
                    int slot=SLOT[b*6+l]; float coef=ALPHA[b]*S[b*36+l*6+r];
                    if(part==0) { vv+=coef*qd[slot]; aa+=coef*qdd[slot]; }
                    if(part==1 && slot==j) { if(jk==1) vv+=coef; if(jk==2) aa+=coef; }
                    if(part==2 && slot==k) { if(kk==1) vv+=coef; if(kk==2) aa+=coef; }
                }
                vj[z]=vv; aj[z]=aa;
            }
            __syncthreads();
            bool dj=p>=0 && jk==0 && j==SLOT[b*6];
            bool dk=p>=0 && kk==0 && k==SLOT[b*6];
            for(int z=threadIdx.x;z<24;z+=blockDim.x) {
                int part=z/6,r=z%6;
                v[b*24+z]=(p<0 ? 0.0f : transformed(x+b*36,dx+b*36,ddx+b*36,v+p*24,part,r,dj,dk))+vj[z];
            }
            __syncthreads();
            for(int z=threadIdx.x;z<24;z+=blockDim.x) {
                int part=z/6,r=z%6;
                float value=p<0 ? gravity[z] : transformed(x+b*36,dx+b*36,ddx+b*36,a+p*24,part,r,dj,dk);
                value+=aj[z];
                if(p>=0) value+=crossed(v+b*24,vj,part,r);
                a[b*24+z]=value;
                momentum[z]=matrix_row(INERTIA+b*36,v+b*24+part*6,r);
            }
            __syncthreads();
            for(int z=threadIdx.x;z<24;z+=blockDim.x)
                force[b*24+z]=matrix_row(INERTIA+b*36,a+b*24+(z/6)*6,z%6)
                               +crossed(v+b*24,momentum,z/6,z%6,true);
            __syncthreads();
        }
        for(int b=NB-1;b>0;--b) {
            int p=PARENT[b];
            bool dj=jk==0 && j==SLOT[b*6], dk=kk==0 && k==SLOT[b*6];
            for(int z=threadIdx.x;z<24;z+=blockDim.x)
                force[p*24+z]+=transformed(x+b*36,dx+b*36,ddx+b*36,force+b*24,z/6,z%6,dj,dk,true);
            __syncthreads();
        }
        for(int row=threadIdx.x;row<NV;row+=blockDim.x) {
            float value=0.0f;
            for(int b=NB-1;b>=0;--b) for(int l=0;l<DIM[b];++l) if(SLOT[b*6+l]==row) {
                float dot=0.0f;
                for(int r=0;r<6;++r) dot+=S[b*36+l*6+r]*force[b*24+18+r];
                value+=ALPHA[b]*dot;
            }
            output[((block*NV+row)*NV+j)*NV+k]=value;
        }
        __syncthreads();
    }
}

int main(int argc,char** argv) {
    if(argc!=4) return 3;
    int threads=atoi(argv[3]); if(threads<1 || threads>128) return 3;
    std::vector<float> input(NQ+2*NV), output(4*NV*NV*NV);
    std::ifstream stream(argv[1],std::ios::binary);
    if(!stream.read(reinterpret_cast<char*>(input.data()),input.size()*sizeof(float))) return 3;
    float *di=nullptr,*dout=nullptr;
    checked(cudaMalloc(&di,input.size()*sizeof(float)));
    checked(cudaMalloc(&dout,output.size()*sizeof(float)));
    checked(cudaMemcpy(di,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice));
    checked(cudaMemset(dout,255,output.size()*sizeof(float)));
    local_tensor_probe<<<1,threads>>>(di,dout);
    checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
    checked(cudaMemcpy(output.data(),dout,output.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(float x: output) if(!std::isfinite(x)) return 4;
    std::ofstream result(argv[2],std::ios::binary);
    if(!result.write(reinterpret_cast<char*>(output.data()),output.size()*sizeof(float))) return 3;
    checked(cudaFree(dout)); checked(cudaFree(di));
    cudaFuncAttributes attributes; checked(cudaFuncGetAttributes(&attributes,local_tensor_probe));
    printf("static_shared_bytes=%zu local_bytes=%zu registers=%d threads=%d\n",
           attributes.sharedSizeBytes,attributes.localSizeBytes,attributes.numRegs,threads);
    return 0;
}

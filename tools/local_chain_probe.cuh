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

// Each thread owns one RHS column; fixed ascending/descending substitution.
__device__ void solve_column(float* rhs, int columns, int column,
                             const float* lower, const float* scale) {
    for(int i=0;i<NV;++i) {
        float v=scale[i]*rhs[i*columns+column];
        for(int k=0;k<i;++k) v-=lower[i*NV+k]*rhs[k*columns+column];
        rhs[i*columns+column]=v/lower[i*NV+i];
    }
    for(int i=NV-1;i>=0;--i) {
        float v=rhs[i*columns+column];
        for(int k=i+1;k<NV;++k) v-=lower[k*NV+i]*rhs[k*columns+column];
        rhs[i*columns+column]=v/lower[i*NV+i];
    }
    for(int i=0;i<NV;++i) rhs[i*columns+column]*=scale[i];
}

__global__ void local_chain_probe(const float* inputs, float* output) {
    __shared__ float q[NQ], x[NB*36], dx[NB*36], ddx[NB*36];
    __shared__ float v[NB*24], a[NB*24], force[NB*24];
    __shared__ float vj[24], aj[24], momentum[24], gravity[24];
    __shared__ float mass[NV*NV], lower[NV*NV], grad[2*NV*NV];
    __shared__ float scale[NV], qdd[NV], bias[NV];
    const float* qd=inputs+NQ; const float* torque=qd+NV;
    for(int i=threadIdx.x;i<NV;i+=blockDim.x) qdd[i]=0.0f;
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
    // -2: mass columns and bias; -1: ID gradient at own acceleration;
    // 0..3: full second-order tensors at that same acceleration.
    for(int block=-2;block<4;++block) {
      int pairs=block==-2 ? NV : (block==-1 ? 2*NV : NV*NV);
      for(int pair=0;pair<pairs;++pair) {
        int j=block<0 ? pair%NV : pair/NV;
        int k=block<0 ? 0 : pair%NV;
        int jk=block==-2 ? 2 : (block==-1 ? (pair<NV ? 0 : 1) : (block==0 ? 0 : (block==3 ? 2 : 1)));
        int kk=block<0 ? 3 : (block==1 ? 1 : 0);
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
                for(int r=0;r<6;++r) dot+=S[b*36+l*6+r]*force[b*24+(block<0 ? 6 : 18)+r];
                value+=ALPHA[b]*dot;
            }
            if(block==-2) {
                mass[row*NV+j]=value+(row==j ? ARMATURE[row] : 0.0f);
                if(j==0) {
                    float base=0.0f;
                    for(int b=NB-1;b>=0;--b) for(int l=0;l<DIM[b];++l) if(SLOT[b*6+l]==row) {
                        float dot=0.0f;
                        for(int r=0;r<6;++r) dot+=S[b*36+l*6+r]*force[b*24+r];
                        base+=ALPHA[b]*dot;
                    }
                    bias[row]=base;
                }
            } else if(block==-1) grad[row*(2*NV)+pair]=-value;
            else output[((block*NV+row)*NV+j)*NV+k]=value;
        }
        __syncthreads();
      }
      if(block==-2) {
        if(threadIdx.x==0) {
            for(int i=0;i<NV;++i) scale[i]=1.0f/sqrtf(mass[i*NV+i]);
            for(int i=0;i<NV;++i) for(int j=0;j<=i;++j) {
                float value=(mass[i*NV+j]*scale[i])*scale[j];
                for(int k=0;k<j;++k) value-=lower[i*NV+k]*lower[j*NV+k];
                lower[i*NV+j]=i==j ? sqrtf(value) : value/lower[j*NV+j];
            }
            for(int i=0;i<NV;++i) qdd[i]=torque[i]-bias[i];
            solve_column(qdd,1,0,lower,scale);
        }
        __syncthreads();
        // The original mass is dead; reuse its private storage for solved I.
        for(int z=threadIdx.x;z<NV*NV;z+=blockDim.x) mass[z]=(z/NV==z%NV ? 1.0f : 0.0f);
        __syncthreads();
        for(int col=threadIdx.x;col<NV;col+=blockDim.x) solve_column(mass,NV,col,lower,scale);
        __syncthreads();
      }
      if(block==-1) {
        for(int col=threadIdx.x;col<2*NV;col+=blockDim.x) solve_column(grad,2*NV,col,lower,scale);
        __syncthreads();
      }
    }
    const float* dm=output+3*NV*NV*NV;
    float* fd=output+4*NV*NV*NV;
    for(int block=0;block<4;++block) {
        for(int col=threadIdx.x;col<NV*NV;col+=blockDim.x) {
            int j=col/NV,k=col%NV;
            for(int i=0;i<NV;++i) {
                int index=(i*NV+j)*NV+k;
                float value=block==0 ? output[index] : (block==1 ? output[2*NV*NV*NV+index] : (block==2 ? output[NV*NV*NV+index] : 0.0f));
                if(block==0) {
                    float term1=0.0f,term2=0.0f;
                    for(int l=0;l<NV;++l) {
                        term1+=dm[(i*NV+l)*NV+k]*grad[l*(2*NV)+j];
                        term2+=dm[(i*NV+l)*NV+j]*grad[l*(2*NV)+k];
                    }
                    value=(value+term1)+term2;
                } else if(block==1 || block==3) {
                    float term=0.0f;
                    for(int l=0;l<NV;++l) term+=dm[(i*NV+l)*NV+k]*(block==1 ? grad[l*(2*NV)+NV+j] : mass[l*NV+j]);
                    value+=term;
                }
                fd[block*NV*NV*NV+index]=-value;
            }
            solve_column(fd+block*NV*NV*NV,NV*NV,col,lower,scale);
        }
        __syncthreads();
    }
}

int main(int argc,char** argv) {
    if(argc!=4) return 3;
    int threads=atoi(argv[3]); if(threads<1 || threads>128) return 3;
    std::vector<float> input(NQ+2*NV), output(8*NV*NV*NV);
    std::ifstream stream(argv[1],std::ios::binary);
    if(!stream.read(reinterpret_cast<char*>(input.data()),input.size()*sizeof(float))) return 3;
    float *di=nullptr,*dout=nullptr;
    checked(cudaMalloc(&di,input.size()*sizeof(float)));
    checked(cudaMalloc(&dout,output.size()*sizeof(float)));
    checked(cudaMemcpy(di,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice));
    checked(cudaMemset(dout,255,output.size()*sizeof(float)));
    local_chain_probe<<<1,threads>>>(di,dout);
    checked(cudaGetLastError()); checked(cudaDeviceSynchronize());
    checked(cudaMemcpy(output.data(),dout,output.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(float x: output) if(!std::isfinite(x)) return 4;
    std::ofstream result(argv[2],std::ios::binary);
    if(!result.write(reinterpret_cast<char*>(output.data()+4*NV*NV*NV),4*NV*NV*NV*sizeof(float))) return 3;
    checked(cudaFree(dout)); checked(cudaFree(di));
    cudaFuncAttributes attributes; checked(cudaFuncGetAttributes(&attributes,local_chain_probe));
    printf("static_shared_bytes=%zu local_bytes=%zu registers=%d threads=%d\n",
           attributes.sharedSizeBytes,attributes.localSizeBytes,attributes.numRegs,threads);
    return 0;
}


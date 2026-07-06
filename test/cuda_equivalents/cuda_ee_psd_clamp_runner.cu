// Robot-general self-check for the PSD_CLAMP option of grid_plant::ee_pos_cost_hessian.
//
// Two guarantees, checked on the NV x NV q-block of the NX x NX cost hessian:
//   (1) SPD GUARANTEE: with a desired EE position FAR from p(q), the residual-weighted
//       Newton curvature makes the unclamped hessian INDEFINITE (min-eig < 0); PSD_CLAMP=true
//       must return a block whose min eigenvalue is >= psd_reg_eps (and still symmetric).
//   (2) NON-CORRUPTION: with the desired position == p(q) (zero residual, Newton == GN), the
//       block is J_p^T W J_p -- PSD but RANK-DEFICIENT (a 3 x NV position Jacobian gives rank <= 3,
//       so NV-3 eigenvalues are exactly zero). eig_clamp lifts ONLY those sub-eps eigenvalues to
//       psd_reg_eps and preserves the rest, so the max entrywise change must be <= eps (it does not
//       perturb the well-conditioned part), and the result must be SPD.
//
// Correctness only (no timing): the GPU is shared. Self-contained (grid.cuh only).
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using T = double;
constexpr int NQ  = grid::NUM_POS;
constexpr int NV  = grid::NUM_VEL;
constexpr int NEE = grid::NUM_EES;
constexpr int NX  = NQ + NV;
static_assert(NQ == NV, "runner assumes fixed-base (NUM_POS == NUM_VEL)");

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

template <bool CLAMP>
__global__ void hess_kernel(T *d_hess, const T *d_q, const T *d_pdes, const T *d_W,
                            T *d_pose, T *d_grad, T *d_d2ee,
                            const grid::robotModel<T> *m, T eps) {
    extern __shared__ __align__(16) T s_arena[];
    // GAUSS_NEWTON=false (full Newton), MUJOCO_OUTPUT=false, PSD_CLAMP=CLAMP.
    grid_plant::ee_pos_cost_hessian<T, 0, false, false, false, CLAMP>(
        d_hess, d_q, d_pdes, d_W, d_pose, d_grad, d_d2ee, s_arena, m, eps);
}

// Emit p(q) (rows 0..2 of the EE pose) so the zero-residual case can target it exactly.
__global__ void pose_kernel(T *d_p, const T *d_q, const grid::robotModel<T> *m) {
    __shared__ T s_pose[6*NEE];
    grid::end_effector_pose_device<T>(s_pose, d_q, m); // self-contained; fills 6*NEE pose
    __syncthreads();
    if (threadIdx.x == 0) for (int r=0;r<3;++r) d_p[r]=s_pose[r];
}

static double min_eig_sym(std::vector<double> M /*NV*NV col-major, by value*/) {
    int n=NV; auto at=[&](int r,int c)->double&{return M[r+n*c];};
    for(int sweep=0;sweep<80;++sweep){
        double off=0; for(int p=0;p<n;++p)for(int q=p+1;q<n;++q)off+=at(p,q)*at(p,q);
        if(off<1e-26)break;
        for(int p=0;p<n;++p)for(int q=p+1;q<n;++q){ double apq=at(p,q); if(fabs(apq)<1e-300)continue;
            double tau=(at(q,q)-at(p,p))/(2*apq);
            double t=(tau>=0?1.0:-1.0)/(fabs(tau)+sqrt(1+tau*tau)); double c=1/sqrt(1+t*t),s=t*c;
            for(int k=0;k<n;++k){double a=at(k,p),b=at(k,q);at(k,p)=c*a-s*b;at(k,q)=s*a+c*b;}
            for(int k=0;k<n;++k){double a=at(p,k),b=at(q,k);at(p,k)=c*a-s*b;at(q,k)=s*a+c*b;}
        }
    }
    double mn=1e300; for(int i=0;i<n;++i)mn=std::min(mn,at(i,i)); return mn;
}

static void extract_qblock(const std::vector<T>& hh, std::vector<double>& qb){
    for(int c=0;c<NV;++c) for(int r=0;r<NV;++r) qb[r+NV*c]=hh[r+NX*c];
}

int main(){
    const grid::robotModel<T> *d_m = grid::init_robotModel<T>();
    const T eps = 1e-6;
    size_t inner  = grid::END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t clampw = (size_t)(NV*NV + (2*NV*NV+2*NV+4))*sizeof(T);
    size_t smem   = std::max(inner, clampw);

    std::vector<T> hq(NQ); for(int i=0;i<NQ;++i) hq[i]=0.2*sin(0.7*i)+0.1;
    std::vector<T> hW(3,1.0);
    T *d_q,*d_pdes,*d_W,*d_hess,*d_pose,*d_grad,*d_d2ee,*d_p;
    CK(cudaMalloc(&d_q,NQ*sizeof(T))); CK(cudaMalloc(&d_pdes,3*sizeof(T))); CK(cudaMalloc(&d_W,3*sizeof(T)));
    CK(cudaMalloc(&d_hess,NX*NX*sizeof(T)));
    CK(cudaMalloc(&d_pose,6*NEE*sizeof(T))); CK(cudaMalloc(&d_grad,6*NV*NEE*sizeof(T))); CK(cudaMalloc(&d_d2ee,6*NV*NV*NEE*sizeof(T)));
    CK(cudaMalloc(&d_p,3*sizeof(T)));
    CK(cudaMemcpy(d_q,hq.data(),NQ*sizeof(T),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_W,hW.data(),3*sizeof(T),cudaMemcpyHostToDevice));

    cudaFuncSetAttribute(hess_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    cudaFuncSetAttribute(hess_kernel<true >, cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);
    cudaFuncSetAttribute(pose_kernel,        cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem);

    std::vector<T> hh(NX*NX); std::vector<double> qb(NV*NV);
    int fails=0;

    // ---- Case 1: FAR target -> indefinite Newton -> clamp must yield SPD ----
    { T far[3]={5.0,5.0,5.0}; CK(cudaMemcpy(d_pdes,far,3*sizeof(T),cudaMemcpyHostToDevice)); }
    hess_kernel<false><<<1,64,smem>>>(d_hess,d_q,d_pdes,d_W,d_pose,d_grad,d_d2ee,d_m,eps); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hh.data(),d_hess,NX*NX*sizeof(T),cudaMemcpyDeviceToHost)); extract_qblock(hh,qb);
    double mn_newton=min_eig_sym(qb);
    hess_kernel<true ><<<1,64,smem>>>(d_hess,d_q,d_pdes,d_W,d_pose,d_grad,d_d2ee,d_m,eps); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hh.data(),d_hess,NX*NX*sizeof(T),cudaMemcpyDeviceToHost)); extract_qblock(hh,qb);
    double mn_clamp=min_eig_sym(qb);
    double asym=0; for(int c=0;c<NV;++c)for(int r=0;r<NV;++r)asym=std::max(asym,fabs(qb[r+NV*c]-qb[c+NV*r]));
    printf("[far ] min-eig newton=% .4e  clamp=% .4e  asym=%.2e (eps=%.1e)\n",mn_newton,mn_clamp,asym,eps);
    if(!(mn_clamp>=eps*(1-1e-6)-1e-9)){printf("  FAIL: clamped block not SPD (min-eig < eps)\n");++fails;}
    else printf("  PASS: clamped block SPD\n");
    if(!(asym<=1e-9)){printf("  FAIL: clamped block not symmetric\n");++fails;}
    if(!(mn_newton<0)) printf("  warn: unclamped Newton not indefinite here (clamp still valid, weaker test)\n");

    // ---- Case 2: target == p(q) -> zero residual -> already-PSD -> clamp ~ identity ----
    pose_kernel<<<1,64,smem>>>(d_p,d_q,d_m); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(d_pdes,d_p,3*sizeof(T),cudaMemcpyDeviceToDevice));
    hess_kernel<false><<<1,64,smem>>>(d_hess,d_q,d_pdes,d_W,d_pose,d_grad,d_d2ee,d_m,eps); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hh.data(),d_hess,NX*NX*sizeof(T),cudaMemcpyDeviceToHost)); std::vector<double> qb_gn(NV*NV); extract_qblock(hh,qb_gn);
    hess_kernel<true ><<<1,64,smem>>>(d_hess,d_q,d_pdes,d_W,d_pose,d_grad,d_d2ee,d_m,eps); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hh.data(),d_hess,NX*NX*sizeof(T),cudaMemcpyDeviceToHost)); std::vector<double> qb_c(NV*NV); extract_qblock(hh,qb_c);
    double maxdiff=0; for(int i=0;i<NV*NV;++i) maxdiff=std::max(maxdiff,fabs(qb_gn[i]-qb_c[i]));
    double mn_c = min_eig_sym(qb_c);
    // Only sub-eps (here: exactly-zero, rank-deficient) eigenvalues are lifted; the lift is bounded
    // by eps, so the entrywise change must not exceed eps (a small slack for Jacobi reconstruction).
    printf("[zero] |clamp-unclamped|_max=%.3e  (bound eps=%.1e)  clamp min-eig=% .4e\n",maxdiff,eps,mn_c);
    if(!(maxdiff<=eps*1.01)){printf("  FAIL: clamp perturbed the well-conditioned part (change > eps)\n");++fails;}
    else printf("  PASS: clamp lifts only the rank-deficient null-space (change <= eps)\n");
    if(!(mn_c>=eps*(1-1e-6)-1e-9)){printf("  FAIL: clamped already-PSD block not SPD\n");++fails;}

    printf(fails? "\nRESULT: FAIL (%d)\n":"\nRESULT: PASS\n", fails);
    return fails?1:0;
}

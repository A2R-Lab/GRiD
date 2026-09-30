// FD-oracle validation for W3 Increment 3: the differentiable collision path.
//
// Validates, against central finite differences, BOTH the exposed raw primitive and the assembled
// cost, on a real robot (iiwa14) with a fixed sphere obstacle placed in range:
//   (1) collision_distance_gradient : s_ddist[i*NV+vi] = d(clearance_i)/dq_vi  vs  FD of
//       collision_distance's s_dist  (single obstacle -> smooth, no argmin switching).
//   (2) collision_cost_gradient      : grad_q  vs  FD of collision_cost (the margin hinge; the TOTAL
//       cost is smooth even though a boundary sphere's activation is not).
// Also checks the GN hessian is symmetric PSD (diag>=0). T=double for clean FD (production is fp32).
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using T = double;
namespace gc = grid_collision;
constexpr int NQ = grid::NUM_POS;
constexpr int NV = grid::NUM_VEL;
constexpr int NS = gc::NUM_COLLISION_SPHERES;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

__constant__ gc::Sphere<T> c_obst;   // single obstacle (in-range), set from host
__device__ T d_margin, d_weight, d_eps;

// analytic: fills d_ddist[NS*NV], d_grad[NV], d_hess[NV*NV] at q0
__global__ void analytic_kernel(const T *q0, const grid::robotModel<T> *m,
                                T *d_ddist, T *d_grad, T *d_hess) {
    __shared__ T s_pos[3*NS], s_r[NS], s_n[3*NS], s_dist[NS], s_ddist[NS*NV], s_pg[3*NV*NS];
    __shared__ T s_grad[NV], s_hess[NV*NV];
    gc::Environment<T> env{ &c_obst, 1, nullptr, 0, nullptr, 0 };
    gc::collision_distance_gradient<T>(s_dist, s_ddist, q0, m, env, s_pos, s_r, s_n, s_pg);
    gc::collision_cost_gradient<T>(s_grad, q0, m, env, d_margin, d_weight, s_pos, s_r, s_n, s_dist, s_ddist, s_pg);
    gc::collision_cost_hessian<T>(s_hess, q0, m, env, d_margin, d_weight, s_pos, s_r, s_n, s_dist, s_ddist, s_pg);
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 0; k < NS*NV; ++k) d_ddist[k] = s_ddist[k];
        for (int k = 0; k < NV; ++k)    d_grad[k]  = s_grad[k];
        for (int k = 0; k < NV*NV; ++k) d_hess[k]  = s_hess[k];
    }
}

// FD: perturb q[vi] +-eps, write per-sphere clearances into d_dist_pm (2 rows) and the scalar cost.
__global__ void fd_kernel(const T *q0, const grid::robotModel<T> *m, int vi, int sign,
                          T *d_dist_out, T *d_cost_out) {
    __shared__ T s_q[NQ], s_pos[3*NS], s_r[NS], s_n[3*NS], s_dist[NS], s_out[1];
    gc::Environment<T> env{ &c_obst, 1, nullptr, 0, nullptr, 0 };
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) s_q[i] = q0[i];
    __syncthreads();
    if (threadIdx.x == 0) s_q[vi] += sign * d_eps;
    __syncthreads();
    gc::collision_distance<T>(s_dist, s_n, s_q, m, env, s_pos, s_r);
    gc::collision_cost<T>(s_out, s_q, m, env, d_margin, d_weight, s_pos, s_r);
    __syncthreads();
    if (threadIdx.x == 0) { for (int i = 0; i < NS; ++i) d_dist_out[i] = s_dist[i]; *d_cost_out = s_out[0]; }
}

int main(){
    const grid::robotModel<T> *m = grid::init_robotModel<T>();
    size_t s1 = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t s2 = grid::MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t smem = s1 > s2 ? s1 : s2;
    cudaFuncSetAttribute(analytic_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaFuncSetAttribute(fd_kernel,       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    // q0: a bent config so distal spheres have nonzero Jacobian; obstacle placed near the arm.
    std::vector<T> hq(NQ); for(int i=0;i<NQ;++i) hq[i]=0.3*sin(0.9*i)+0.2;
    T margin=0.25, weight=5.0, eps=1e-6;
    CK(cudaMemcpyToSymbol(d_margin,&margin,sizeof(T)));
    CK(cudaMemcpyToSymbol(d_weight,&weight,sizeof(T)));
    CK(cudaMemcpyToSymbol(d_eps,&eps,sizeof(T)));

    T *d_q; CK(cudaMalloc(&d_q,NQ*sizeof(T))); CK(cudaMemcpy(d_q,hq.data(),NQ*sizeof(T),cudaMemcpyHostToDevice));

    // Find sphere positions at q0 to place an in-range obstacle (near sphere NS/2).
    T *d_ddist,*d_grad,*d_hess,*d_distpm,*d_cost;
    CK(cudaMalloc(&d_ddist,NS*NV*sizeof(T))); CK(cudaMalloc(&d_grad,NV*sizeof(T)));
    CK(cudaMalloc(&d_hess,NV*NV*sizeof(T)));  CK(cudaMalloc(&d_distpm,NS*sizeof(T))); CK(cudaMalloc(&d_cost,sizeof(T)));
    // obstacle at a fixed world point near the iiwa arm envelope (in range at q0).
    gc::Sphere<T> obst{ 0.15, 0.10, 0.6, 0.12 };
    CK(cudaMemcpyToSymbol(c_obst,&obst,sizeof(obst)));

    // analytic
    analytic_kernel<<<1,128,smem>>>(d_q,m,d_ddist,d_grad,d_hess); CK(cudaDeviceSynchronize());
    std::vector<T> ddist(NS*NV), grad(NV), hess(NV*NV);
    CK(cudaMemcpy(ddist.data(),d_ddist,NS*NV*sizeof(T),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(grad.data(),d_grad,NV*sizeof(T),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hess.data(),d_hess,NV*NV*sizeof(T),cudaMemcpyDeviceToHost));

    // FD over each vi
    std::vector<T> fd_ddist(NS*NV), fd_grad(NV);
    std::vector<T> dp(NS), dm(NS); T cp, cm;
    for (int vi=0; vi<NV; ++vi) {
        fd_kernel<<<1,128,smem>>>(d_q,m,vi,+1,d_distpm,d_cost); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(dp.data(),d_distpm,NS*sizeof(T),cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&cp,d_cost,sizeof(T),cudaMemcpyDeviceToHost));
        fd_kernel<<<1,128,smem>>>(d_q,m,vi,-1,d_distpm,d_cost); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(dm.data(),d_distpm,NS*sizeof(T),cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&cm,d_cost,sizeof(T),cudaMemcpyDeviceToHost));
        for (int i=0;i<NS;++i) fd_ddist[i*NV+vi] = (dp[i]-dm[i])/(2*eps);
        fd_grad[vi] = (cp-cm)/(2*eps);
    }

    // compare
    auto maxerr = [](const std::vector<T>&a, const std::vector<T>&b){
        T e=0; for(size_t k=0;k<a.size();++k){ T d=fabs(a[k]-b[k]); T s=fabs(a[k])+fabs(b[k])+1e-9; e=fmax(e, d/s); } return e; };
    T e_ddist = maxerr(ddist, fd_ddist);
    T e_grad  = maxerr(grad,  fd_grad);
    // active-sphere count (nonzero cost gradient contributions) for a meaningful test
    T gnorm=0; for(int i=0;i<NV;++i) gnorm+=grad[i]*grad[i]; gnorm=sqrt(gnorm);
    // hessian symmetry + PSD diagonal
    T sym=0; for(int r=0;r<NV;++r) for(int c=0;c<NV;++c) sym=fmax(sym,fabs(hess[r+NV*c]-hess[c+NV*r]));
    T mindiag=1e30; for(int i=0;i<NV;++i) mindiag=fmin(mindiag,hess[i+NV*i]);

    printf("NS=%d NV=%d  |grad|=%.4f  ddist_relerr=%.2e  grad_relerr=%.2e  hess_sym=%.2e  hess_mindiag=%.3e\n",
           NS,NV,gnorm,e_ddist,e_grad,sym,mindiag);
    bool ok = (gnorm > 1e-6) && (e_ddist < 1e-6) && (e_grad < 1e-6) && (sym < 1e-9) && (mindiag >= 0.0);
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 3;
}

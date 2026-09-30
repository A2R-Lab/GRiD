// FD-oracle gate for the RUNTIME single-contact f_ext map (welded-tool tip). Robot: go2-FLOATING.
//
// Same map/derivative math as the baked contact family (cuda_f_ext_contact_runner.cu), but the contact
// body `b` and local offset `r_c` are RUNTIME arguments (compiled in here via -DTOOL_JID / -DTOOL_RC{0,1,2}
// from the Python driver, which resolves them from a real go2 foot frame). go2-FLOATING is deliberate:
// this touches the world-FK chain-up + tier arena (the §1s/§1t surface), and go2 is branched AND floating.
//
// CHECKS (all on the RUNTIME device fns): value nonzero; d(f_ext)/d(f_c) vs central FD in f_c + bit-exact
// f_c-independence; d(f_ext)/dq vs central FD in q via an SE(3) RETRACT (grid_integrate_floating_q);
// f_c-linearity; and ZERO rows on every body other than b. T=double for clean FD (production is fp32).
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using T = double;
constexpr int NQ  = grid::NUM_POS;
constexpr int NV  = grid::NUM_VEL;
constexpr int NB  = grid::NUM_BODIES;
constexpr int NR  = 6 * NB;   // f_ext rows
constexpr int NFC = 6;        // single contact wrench
constexpr int TJID = TOOL_JID;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

__device__ T d_eps;

__global__ void analytic_kernel(const T *q0, const T *fc, const T *d_rc, const grid::robotModel<T> *m,
                                T *d_fext, T *d_dfc, T *d_dq) {
    __shared__ T s_fext[NR], s_dfc[NR*NFC], s_dq[NR*NV];
    __shared__ T s_dtau[NV*6*NB], s_dqdd[NV*6*NB];
    __shared__ T s_rc[3];
    if (threadIdx.x < 3 && threadIdx.y == 0) s_rc[threadIdx.x] = d_rc[threadIdx.x];
    // -J^T (its columns ARE the local body Jacobian -> the omega the dq map needs)
    grid::f_ext_gradient_device<T>(s_dtau, s_dqdd, q0, m);
    __syncthreads();
    grid::f_ext_body_runtime_device<T>(s_fext, fc, TJID, s_rc, q0, m);
    grid::f_ext_body_jacobian_dfc_runtime_device<T>(s_dfc, TJID, s_rc, q0, m);
    grid::f_ext_body_jacobian_dq_runtime_device<T>(s_dq, fc, TJID, s_rc, s_dtau, q0, m);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < NR; ++i)      d_fext[i] = s_fext[i];
        for (int i = 0; i < NR*NFC; ++i)  d_dfc[i]  = s_dfc[i];
        for (int i = 0; i < NR*NV; ++i)   d_dq[i]   = s_dq[i];
    }
}

// f_ext value at a perturbed state. mode: 0 = perturb f_c[idx], 1 = SE(3)-retract q along dv[idx].
__global__ void fd_kernel(const T *q0, const T *fc, const T *d_rc, const grid::robotModel<T> *m,
                          int mode, int idx, int sign, T *d_out) {
    __shared__ T s_q[NQ], s_fc[NFC], s_fext[NR], s_dv[NV], s_qpert[NQ], s_rc[3];
    for (int i = threadIdx.x; i < NQ;  i += blockDim.x) s_q[i]  = q0[i];
    for (int i = threadIdx.x; i < NFC; i += blockDim.x) s_fc[i] = fc[i];
    if (threadIdx.x < 3 && threadIdx.y == 0) s_rc[threadIdx.x] = d_rc[threadIdx.x];
    __syncthreads();
    if (mode == 0) {
        if (threadIdx.x == 0) s_fc[idx] += sign * d_eps;
        __syncthreads();
        grid::f_ext_body_runtime_device<T>(s_fext, s_fc, TJID, s_rc, s_q, m);
    } else {
        for (int i = threadIdx.x; i < NV; i += blockDim.x) s_dv[i] = static_cast<T>(0);
        __syncthreads();
        if (threadIdx.x == 0) s_dv[idx] = sign * d_eps;
        __syncthreads();
        grid::grid_integrate_floating_q<T, NQ>(s_q, s_dv, s_qpert);
        __syncthreads();
        grid::f_ext_body_runtime_device<T>(s_fext, s_fc, TJID, s_rc, s_qpert, m);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) for (int i = 0; i < NR; ++i) d_out[i] = s_fext[i];
}

int main(){
    const grid::robotModel<T> *m = grid::init_robotModel<T>();
    size_t smem = grid::F_EXT_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t s2 = grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>();
    if (s2 > smem) smem = s2;
    smem += 4096;
    cudaFuncSetAttribute(analytic_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaFuncSetAttribute(fd_kernel,       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    std::vector<T> hq(NQ, 0.0);
    hq[0]=0.05; hq[1]=-0.03; hq[2]=0.31;
    { T ax=0.3, ay=-0.2, az=0.5, an=sqrt(ax*ax+ay*ay+az*az), th=0.35;
      hq[3]=ax/an*sin(th/2); hq[4]=ay/an*sin(th/2); hq[5]=az/an*sin(th/2); hq[6]=cos(th/2); }
    for (int i = 7; i < NQ; ++i) hq[i] = 0.25*sin(0.8*i) + 0.15;

    // 6D contact wrench (nonzero angular part too, so the R^T n_w term is exercised).
    std::vector<T> hfc(NFC);
    hfc[0]=0.7; hfc[1]=-0.4; hfc[2]=0.3;   hfc[3]=2.0; hfc[4]=-1.5; hfc[5]=40.0;
    T hrc[3] = {TOOL_RC0, TOOL_RC1, TOOL_RC2};
    T eps = 1e-6;
    CK(cudaMemcpyToSymbol(d_eps, &eps, sizeof(T)));

    T *d_q,*d_fc,*d_rc,*d_fext,*d_dfc,*d_dq,*d_tmp;
    CK(cudaMalloc(&d_q, NQ*sizeof(T)));   CK(cudaMemcpy(d_q,hq.data(),NQ*sizeof(T),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_fc, NFC*sizeof(T))); CK(cudaMemcpy(d_fc,hfc.data(),NFC*sizeof(T),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_rc, 3*sizeof(T)));   CK(cudaMemcpy(d_rc,hrc,3*sizeof(T),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_fext, NR*sizeof(T))); CK(cudaMalloc(&d_dfc, (size_t)NR*NFC*sizeof(T)));
    CK(cudaMalloc(&d_dq, (size_t)NR*NV*sizeof(T))); CK(cudaMalloc(&d_tmp, NR*sizeof(T)));

    analytic_kernel<<<1,128,smem>>>(d_q,d_fc,d_rc,m,d_fext,d_dfc,d_dq); CK(cudaDeviceSynchronize());
    std::vector<T> fext(NR), dfc((size_t)NR*NFC), dq((size_t)NR*NV);
    CK(cudaMemcpy(fext.data(), d_fext, NR*sizeof(T), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dfc.data(),  d_dfc,  (size_t)NR*NFC*sizeof(T), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dq.data(),   d_dq,   (size_t)NR*NV*sizeof(T),  cudaMemcpyDeviceToHost));

    T fnorm = 0; for (T v : fext) fnorm += v*v; fnorm = sqrt(fnorm);

    // (3) d(f_ext)/d(f_c) vs central FD in f_c
    std::vector<T> fd_dfc((size_t)NR*NFC), pp(NR), mm(NR);
    for (int j = 0; j < NFC; ++j) {
        fd_kernel<<<1,128,smem>>>(d_q,d_fc,d_rc,m,0,j,+1,d_tmp); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(pp.data(),d_tmp,NR*sizeof(T),cudaMemcpyDeviceToHost));
        fd_kernel<<<1,128,smem>>>(d_q,d_fc,d_rc,m,0,j,-1,d_tmp); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(mm.data(),d_tmp,NR*sizeof(T),cudaMemcpyDeviceToHost));
        for (int r = 0; r < NR; ++r) fd_dfc[(size_t)r + (size_t)NR*j] = (pp[r]-mm[r])/(2*eps);
    }
    T e_dfc = 0;
    for (size_t k = 0; k < dfc.size(); ++k)
        e_dfc = fmax(e_dfc, fabs(dfc[k]-fd_dfc[k])/(fabs(dfc[k])+fabs(fd_dfc[k])+1e-9));

    // (4) d(f_ext)/dq vs central FD in q (SE(3) retract)
    std::vector<T> fd_dq((size_t)NR*NV);
    for (int v = 0; v < NV; ++v) {
        fd_kernel<<<1,128,smem>>>(d_q,d_fc,d_rc,m,1,v,+1,d_tmp); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(pp.data(),d_tmp,NR*sizeof(T),cudaMemcpyDeviceToHost));
        fd_kernel<<<1,128,smem>>>(d_q,d_fc,d_rc,m,1,v,-1,d_tmp); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(mm.data(),d_tmp,NR*sizeof(T),cudaMemcpyDeviceToHost));
        for (int r = 0; r < NR; ++r) fd_dq[(size_t)r + (size_t)NR*v] = (pp[r]-mm[r])/(2*eps);
    }
    T e_dq = 0, dqnorm = 0;
    for (size_t k = 0; k < dq.size(); ++k) {
        dqnorm += dq[k]*dq[k];
        e_dq = fmax(e_dq, fabs(dq[k]-fd_dq[k])/(fabs(dq[k])+fabs(fd_dq[k])+1e-6));
    }
    dqnorm = sqrt(dqnorm);

    // (2) f_c-LINEARITY + f_c-INDEPENDENCE of dfc
    std::vector<T> fc2(NFC); for (int i=0;i<NFC;++i) fc2[i] = 2.0*hfc[i];
    T *d_fc2; CK(cudaMalloc(&d_fc2, NFC*sizeof(T)));
    CK(cudaMemcpy(d_fc2, fc2.data(), NFC*sizeof(T), cudaMemcpyHostToDevice));
    analytic_kernel<<<1,128,smem>>>(d_q,d_fc2,d_rc,m,d_fext,d_dfc,d_dq); CK(cudaDeviceSynchronize());
    std::vector<T> fext2(NR), dfc2((size_t)NR*NFC);
    CK(cudaMemcpy(fext2.data(), d_fext, NR*sizeof(T), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dfc2.data(),  d_dfc,  (size_t)NR*NFC*sizeof(T), cudaMemcpyDeviceToHost));
    T e_lin = 0;
    for (int r = 0; r < NR; ++r) e_lin = fmax(e_lin, fabs(fext2[r] - 2.0*fext[r])/(fabs(fext2[r])+1e-9));
    int dfc_indep_mismatch = 0;
    for (size_t k = 0; k < dfc.size(); ++k) if (dfc[k] != dfc2[k]) ++dfc_indep_mismatch;

    // (5) only body TJID may be nonzero (value + dq rows).
    int nz_bodies = 0, bad_zero = 0;
    for (int b = 0; b < NB; ++b) {
        T s = 0; for (int k = 0; k < 6; ++k) s += fabs(fext[6*b+k]);
        if (s > 0) { ++nz_bodies; if (b != TJID) ++bad_zero; }
    }
    for (int b = 0; b < NB; ++b) if (b != TJID)
        for (int v = 0; v < NV; ++v) for (int k = 0; k < 6; ++k)
            if (dq[(size_t)(6*b+k) + (size_t)NR*v] != 0.0) ++bad_zero;

    printf("go2-FLOATING RUNTIME  NB=%d NV=%d TJID=%d  |f_ext|=%.4f  |df/dq|=%.4f  contacted_bodies=%d\n",
           NB, NV, TJID, fnorm, dqnorm, nz_bodies);
    printf("  dfc_relerr=%.2e   dq_relerr=%.2e\n", e_dfc, e_dq);
    printf("  f_c linearity err=%.2e   dfc f_c-independence mismatches=%d   spurious-nonzero-rows=%d\n",
           e_lin, dfc_indep_mismatch, bad_zero);
    bool ok = (fnorm > 1e-6) && (dqnorm > 1e-6) && (nz_bodies == 1)
              && (e_dfc < 1e-6) && (e_dq < 1e-5)
              && (e_lin < 1e-12) && (dfc_indep_mismatch == 0) && (bad_zero == 0);
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 3;
}

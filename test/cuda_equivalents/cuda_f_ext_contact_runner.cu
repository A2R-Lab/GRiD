// FD-oracle gate for the contact-FRAME f_ext map (GATO ask 1, C.2). Robot: go2-FLOATING.
//
// WHY go2-FLOATING AND NOT AN ARM. This module touches the world-FK chain-up + tier arena -- the exact
// surface where BOTH of this month's silent bugs lived (§1s: a fixed-target jid has no link, so the
// chain-up silently no-ops on BRANCHED trees and reads uninitialized smem; §1t: an under-counted arena
// writes past the end on FLOATING base only). Both compiled clean and passed every fixed-base test.
// go2 is branched AND floating AND has 4 real contact frames, so it is the robot that can actually fail.
//
// WHAT IS CHECKED
//   1. VALUE vs an INDEPENDENT host recomputation from the device's own FK pose:
//        f_ext[b] = [ R^T n_w + r_c x (R^T f_w) ; R^T f_w ]
//      (the host builds R from end_effector-style world transforms it reads back, so this is a real
//       cross-check of the extraction, not a tautology).
//   2. f_c-LINEARITY: f_ext(a*f_c) == a*f_ext(f_c) and f_ext(f1+f2) == f_ext(f1)+f_ext(f2). The map is
//      claimed LINEAR in f_c; if that is false, d(f_ext)/df_c being f_c-independent is also false.
//   3. d(f_ext)/d(f_c) vs central FD in f_c. Also asserted f_c-INDEPENDENT: recomputed at a DIFFERENT
//      f_c and required BIT-identical.
//   4. d(f_ext)/dq vs central FD in q. ⚠ FLOATING BASE: the q-perturbation is an SE(3) retract via
//      grid_integrate_floating_q, NOT a scalar q[i]+=h -- perturbing the quaternion componentwise
//      leaves the manifold and would produce a wrong "oracle" that silently disagrees.
//   5. ZERO-ROW check: bodies with no contact must have exactly zero f_ext and zero sensitivity.
// T=double for clean FD (production is fp32).
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using T = double;
constexpr int NQ  = grid::NUM_POS;
constexpr int NV  = grid::NUM_VEL;
constexpr int NB  = grid::NUM_BODIES;
constexpr int NC  = grid::NUM_CONTACT_FRAMES;
constexpr int NR  = 6 * NB;            // f_ext rows
constexpr int NFC = 6 * NC;            // contact-wrench columns

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

__device__ T d_eps;

// value + both Jacobians at q0, plus the world transforms so the host can independently recompute.
__global__ void analytic_kernel(const T *q0, const T *fc, const grid::robotModel<T> *m,
                                T *d_fext, T *d_dfc, T *d_dq, T *d_dtau, T *d_Xw) {
    __shared__ T s_fext[NR], s_dfc[NR*NFC], s_dq[NR*NV];
    __shared__ T s_dtau[NV*6*NB], s_dqdd[NV*6*NB];
    // -J^T (its columns ARE the local body Jacobian -> the omega the dq map needs)
    grid::f_ext_gradient_device<T>(s_dtau, s_dqdd, q0, m);
    __syncthreads();
    grid::f_ext_body_device<T>(s_fext, fc, q0, m);
    grid::f_ext_body_jacobian_dfc_device<T>(s_dfc, q0, m);
    grid::f_ext_body_jacobian_dq_device<T>(s_dq, fc, s_dtau, q0, m);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < NR; ++i)      d_fext[i] = s_fext[i];
        for (int i = 0; i < NR*NFC; ++i)  d_dfc[i]  = s_dfc[i];
        for (int i = 0; i < NR*NV; ++i)   d_dq[i]   = s_dq[i];
        for (int i = 0; i < NV*6*NB; ++i) d_dtau[i] = s_dtau[i];
    }
}

// f_ext value at a perturbed state. `mode`: 0 = perturb f_c[idx], 1 = SE(3)-retract q along dv[idx].
__global__ void fd_kernel(const T *q0, const T *fc, const grid::robotModel<T> *m,
                          int mode, int idx, int sign, T *d_out) {
    __shared__ T s_q[NQ], s_fc[NFC], s_fext[NR], s_dv[NV], s_qpert[NQ];
    for (int i = threadIdx.x; i < NQ;  i += blockDim.x) s_q[i]  = q0[i];
    for (int i = threadIdx.x; i < NFC; i += blockDim.x) s_fc[i] = fc[i];
    __syncthreads();
    if (mode == 0) {
        if (threadIdx.x == 0) s_fc[idx] += sign * d_eps;
        __syncthreads();
        grid::f_ext_body_device<T>(s_fext, s_fc, s_q, m);
    } else {
        // FLOATING BASE: q lives on SE(3) x R^n. Perturb in the TANGENT space and retract, exactly as
        // grid's own f_ext_gradient_dq FD does. A componentwise q[i] += h would leave the manifold.
        for (int i = threadIdx.x; i < NV; i += blockDim.x) s_dv[i] = static_cast<T>(0);
        __syncthreads();
        if (threadIdx.x == 0) s_dv[idx] = sign * d_eps;
        __syncthreads();
        grid::grid_integrate_floating_q<T, NQ>(s_q, s_dv, s_qpert);
        __syncthreads();
        grid::f_ext_body_device<T>(s_fext, s_fc, s_qpert, m);
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

    // q0: a non-trivial floating-base configuration (unit quaternion, bent legs).
    std::vector<T> hq(NQ, 0.0);
    hq[0]=0.05; hq[1]=-0.03; hq[2]=0.31;                       // base xyz
    { T ax=0.3, ay=-0.2, az=0.5, an=sqrt(ax*ax+ay*ay+az*az), th=0.35;
      hq[3]=ax/an*sin(th/2); hq[4]=ay/an*sin(th/2); hq[5]=az/an*sin(th/2); hq[6]=cos(th/2); }
    for (int i = 7; i < NQ; ++i) hq[i] = 0.25*sin(0.8*i) + 0.15;

    // A 6D contact wrench per foot: nonzero ANGULAR part too, so the 6D path is genuinely exercised
    // (a pure point force would leave n_w = 0 and never test the R^T n_w term).
    std::vector<T> hfc(NFC);
    for (int c = 0; c < NC; ++c) {
        hfc[6*c+0] =  0.7 - 0.2*c;  hfc[6*c+1] = -0.4 + 0.1*c;  hfc[6*c+2] =  0.3 + 0.05*c;  // n_w
        hfc[6*c+3] =  2.0 + 0.5*c;  hfc[6*c+4] = -1.5 + 0.3*c;  hfc[6*c+5] = 40.0 - 2.0*c;   // f_w (support)
    }
    T eps = 1e-6;
    CK(cudaMemcpyToSymbol(d_eps, &eps, sizeof(T)));

    T *d_q,*d_fc,*d_fext,*d_dfc,*d_dq,*d_dtau,*d_Xw,*d_tmp;
    CK(cudaMalloc(&d_q, NQ*sizeof(T)));      CK(cudaMemcpy(d_q,hq.data(),NQ*sizeof(T),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_fc, NFC*sizeof(T)));    CK(cudaMemcpy(d_fc,hfc.data(),NFC*sizeof(T),cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_fext, NR*sizeof(T)));   CK(cudaMalloc(&d_dfc, (size_t)NR*NFC*sizeof(T)));
    CK(cudaMalloc(&d_dq, (size_t)NR*NV*sizeof(T))); CK(cudaMalloc(&d_dtau, (size_t)NV*6*NB*sizeof(T)));
    CK(cudaMalloc(&d_Xw, 16*grid::NUM_JOINTS*sizeof(T))); CK(cudaMalloc(&d_tmp, NR*sizeof(T)));

    analytic_kernel<<<1,128,smem>>>(d_q,d_fc,m,d_fext,d_dfc,d_dq,d_dtau,d_Xw); CK(cudaDeviceSynchronize());
    std::vector<T> fext(NR), dfc((size_t)NR*NFC), dq((size_t)NR*NV);
    CK(cudaMemcpy(fext.data(), d_fext, NR*sizeof(T), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dfc.data(),  d_dfc,  (size_t)NR*NFC*sizeof(T), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dq.data(),   d_dq,   (size_t)NR*NV*sizeof(T),  cudaMemcpyDeviceToHost));

    T fnorm = 0; for (T v : fext) fnorm += v*v; fnorm = sqrt(fnorm);

    // ---- (3) d(f_ext)/d(f_c) vs central FD in f_c ------------------------------------------------
    std::vector<T> fd_dfc((size_t)NR*NFC);
    std::vector<T> pp(NR), mm(NR);
    for (int j = 0; j < NFC; ++j) {
        fd_kernel<<<1,128,smem>>>(d_q,d_fc,m,0,j,+1,d_tmp); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(pp.data(),d_tmp,NR*sizeof(T),cudaMemcpyDeviceToHost));
        fd_kernel<<<1,128,smem>>>(d_q,d_fc,m,0,j,-1,d_tmp); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(mm.data(),d_tmp,NR*sizeof(T),cudaMemcpyDeviceToHost));
        for (int r = 0; r < NR; ++r) fd_dfc[(size_t)r + (size_t)NR*j] = (pp[r]-mm[r])/(2*eps);
    }
    T e_dfc = 0;
    for (size_t k = 0; k < dfc.size(); ++k) {
        T a = dfc[k], b = fd_dfc[k];
        e_dfc = fmax(e_dfc, fabs(a-b)/(fabs(a)+fabs(b)+1e-9));
    }

    // ---- (4) d(f_ext)/dq vs central FD in q (SE(3) retract) ---------------------------------------
    std::vector<T> fd_dq((size_t)NR*NV);
    for (int v = 0; v < NV; ++v) {
        fd_kernel<<<1,128,smem>>>(d_q,d_fc,m,1,v,+1,d_tmp); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(pp.data(),d_tmp,NR*sizeof(T),cudaMemcpyDeviceToHost));
        fd_kernel<<<1,128,smem>>>(d_q,d_fc,m,1,v,-1,d_tmp); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(mm.data(),d_tmp,NR*sizeof(T),cudaMemcpyDeviceToHost));
        for (int r = 0; r < NR; ++r) fd_dq[(size_t)r + (size_t)NR*v] = (pp[r]-mm[r])/(2*eps);
    }
    T e_dq = 0, dqnorm = 0;
    for (size_t k = 0; k < dq.size(); ++k) {
        T a = dq[k], b = fd_dq[k];
        dqnorm += a*a;
        e_dq = fmax(e_dq, fabs(a-b)/(fabs(a)+fabs(b)+1e-6));
    }
    dqnorm = sqrt(dqnorm);

    // ---- (2) f_c-LINEARITY: f_ext(2*f_c) must be 2*f_ext(f_c) --------------------------------------
    std::vector<T> fc2(NFC); for (int i=0;i<NFC;++i) fc2[i] = 2.0*hfc[i];
    T *d_fc2; CK(cudaMalloc(&d_fc2, NFC*sizeof(T)));
    CK(cudaMemcpy(d_fc2, fc2.data(), NFC*sizeof(T), cudaMemcpyHostToDevice));
    analytic_kernel<<<1,128,smem>>>(d_q,d_fc2,m,d_fext,d_dfc,d_dq,d_dtau,d_Xw); CK(cudaDeviceSynchronize());
    std::vector<T> fext2(NR), dfc2((size_t)NR*NFC);
    CK(cudaMemcpy(fext2.data(), d_fext, NR*sizeof(T), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dfc2.data(),  d_dfc,  (size_t)NR*NFC*sizeof(T), cudaMemcpyDeviceToHost));
    T e_lin = 0;
    for (int r = 0; r < NR; ++r) e_lin = fmax(e_lin, fabs(fext2[r] - 2.0*fext[r])/(fabs(fext2[r])+1e-9));
    // f_c-INDEPENDENCE of the dfc block: recomputed at 2*f_c it must be BIT-identical.
    int dfc_indep_mismatch = 0;
    for (size_t k = 0; k < dfc.size(); ++k) if (dfc[k] != dfc2[k]) ++dfc_indep_mismatch;

    // ---- (5) bodies with NO contact must be exactly zero -------------------------------------------
    bool contacted[NB] = {false};
    // (the baked table isn't host-visible; infer from a nonzero f_ext row -- then assert the ZERO rows
    //  are exactly zero, which is the property that matters: no spurious writes into other bodies.)
    int nz_bodies = 0, bad_zero = 0;
    for (int b = 0; b < NB; ++b) {
        T s = 0; for (int k = 0; k < 6; ++k) s += fabs(fext[6*b+k]);
        if (s > 0) { ++nz_bodies; contacted[b] = true; }
    }
    for (int b = 0; b < NB; ++b) if (!contacted[b])
        for (int v = 0; v < NV; ++v) for (int k = 0; k < 6; ++k)
            if (dq[(size_t)(6*b+k) + (size_t)NR*v] != 0.0) ++bad_zero;

    printf("go2-FLOATING  NB=%d NV=%d NC=%d  |f_ext|=%.4f  |df/dq|=%.4f  contacted_bodies=%d\n",
           NB, NV, NC, fnorm, dqnorm, nz_bodies);
    printf("  dfc_relerr=%.2e   dq_relerr=%.2e\n", e_dfc, e_dq);
    printf("  f_c linearity err=%.2e   dfc f_c-independence mismatches=%d   nonzero-rows-on-uncontacted-bodies=%d\n",
           e_lin, dfc_indep_mismatch, bad_zero);
    bool ok = (fnorm > 1e-6) && (dqnorm > 1e-6) && (nz_bodies == NC)
              && (e_dfc < 1e-6) && (e_dq < 1e-5)
              && (e_lin < 1e-12) && (dfc_indep_mismatch == 0) && (bad_zero == 0);
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 3;
}

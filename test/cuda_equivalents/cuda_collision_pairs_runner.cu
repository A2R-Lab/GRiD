// FD-oracle validation for the GATO Ask-3 collision surface: the Plane (half-space) primitive and
// the UN-REDUCED per-(sphere, obstacle) distance rows.
//
// The existing collision_distance min-reduces over the environment, and that argmin is non-smooth
// exactly where the winning obstacle switches. GATO wants one smooth constraint row per pair, so
// collision_distance_pairs{,_gradient} drop the reduction. This runner certifies, on a real spherized
// robot (iiwa14) against an environment holding ALL FOUR primitive kinds (2 spheres, 1 capsule,
// 1 cuboid, 1 plane -> n_obs = 5, exercising every branch of the flattened obstacle index):
//   (1) REDUCTION CONSISTENCY: min over o of pairs[i*n_obs+o] == collision_distance's s_dist[i],
//       BIT-EXACTLY (same SDFs, same order, same strict-< tie-break). The un-reduced rows must be a
//       strict refinement of the reduced API, not a reimplementation that drifts from it.
//   (2) PAIR JACOBIAN: s_ddist[pair*NV+vi] = d(d_io)/dq_vi vs central FD of the pair clearance.
//   (3) PLANE EXACTNESS: the plane's normal is its own (constant, unit) and its clearance is exactly
//       n.p - d - r -- checked against a host recomputation from the extracted sphere positions.
//   (4) BOOLEAN/SIGNED AGREEMENT: sign(grid_cc_sphere_plane) == sign(grid_cc_sphere_plane_signed) over
//       a swept set of centers straddling the surface (the squared-gap form must not lose the sign for
//       a center BELOW the plane -- the clamp-to-excess step that makes it match grid_cc_sphere_cuboid).
// T=double for clean FD (production is fp32).
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
constexpr int NOBS = 5;                 // 2 spheres + 1 capsule + 1 cuboid + 1 plane
constexpr int NPAIR = NS * NOBS;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

__constant__ gc::Sphere<T>  c_sph[2];
__constant__ gc::Capsule<T> c_cap[1];
__constant__ gc::Cuboid<T>  c_box[1];
__constant__ gc::Plane<T>   c_pln[1];
__device__ T d_eps;

__device__ __forceinline__ gc::Environment<T> make_env() {
    gc::Environment<T> env{ c_sph, 2, c_cap, 1, c_box, 1, c_pln, 1 };
    return env;
}

// analytic: per-pair clearances + Jacobian, plus the REDUCED clearance from the existing API and the
// sphere world positions (so the host can recompute the plane row exactly).
__global__ void analytic_kernel(const T *q0, const grid::robotModel<T> *m,
                                T *d_pdist, T *d_pddist, T *d_pnorm, T *d_red, T *d_pos, T *d_r) {
    __shared__ T s_pos[3*NS], s_r[NS], s_pg[3*NV*NS];
    __shared__ T s_pdist[NPAIR], s_pnorm[3*NPAIR], s_pddist[NPAIR*NV];
    __shared__ T s_red[NS], s_rednorm[3*NS];
    gc::Environment<T> env = make_env();
    gc::collision_distance_pairs_gradient<T>(s_pdist, s_pddist, q0, m, env, s_pos, s_r, s_pnorm, s_pg);
    gc::collision_distance<T>(s_red, s_rednorm, q0, m, env, s_pos, s_r);   // the reduced (argmin) API
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 0; k < NPAIR; ++k)    d_pdist[k]  = s_pdist[k];
        for (int k = 0; k < NPAIR*NV; ++k) d_pddist[k] = s_pddist[k];
        for (int k = 0; k < 3*NPAIR; ++k)  d_pnorm[k]  = s_pnorm[k];
        for (int k = 0; k < NS; ++k)       d_red[k]    = s_red[k];
        for (int k = 0; k < 3*NS; ++k)     d_pos[k]    = s_pos[k];
        for (int k = 0; k < NS; ++k)       d_r[k]      = s_r[k];   // baked radii, cast to T on-device
    }
}

// FD: perturb q[vi] += sign*eps, emit the per-pair clearances.
__global__ void fd_kernel(const T *q0, const grid::robotModel<T> *m, int vi, int sign, T *d_pdist) {
    __shared__ T s_q[NQ], s_pos[3*NS], s_r[NS], s_pdist[NPAIR], s_pnorm[3*NPAIR];
    gc::Environment<T> env = make_env();
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) s_q[i] = q0[i];
    __syncthreads();
    if (threadIdx.x == 0) s_q[vi] += sign * d_eps;
    __syncthreads();
    gc::collision_distance_pairs<T>(s_pdist, s_pnorm, s_q, m, env, s_pos, s_r);
    __syncthreads();
    if (threadIdx.x == 0) for (int k = 0; k < NPAIR; ++k) d_pdist[k] = s_pdist[k];
}

int main(){
    const grid::robotModel<T> *m = grid::init_robotModel<T>();
    size_t s1 = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t s2 = grid::MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t smem = s1 > s2 ? s1 : s2;
    cudaFuncSetAttribute(analytic_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaFuncSetAttribute(fd_kernel,       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    // ---- (4) host-side boolean/signed sign agreement for the plane, swept across the surface ----
    // A center BELOW the plane has n.p-d < 0; the squared-gap form must still report collision rather
    // than squaring away the sign. Sweep straddles the surface at a range of radii.
    gc::Plane<T> hp{ 0.0, 0.0, 1.0, 0.0 };   // canonical ground plane z >= 0
    int sign_mismatch = 0;
    for (int k = -40; k <= 40; ++k) {
        T z = 0.01 * k;                       // -0.4 .. 0.4, straddling the surface
        for (int rk = 1; rk <= 5; ++rk) {
            T r = 0.05 * rk;
            T sq = gc::grid_cc_sphere_plane<T>(hp, 0.3, -0.2, z, r);
            T nx, ny, nz;
            T sd = gc::grid_cc_sphere_plane_signed<T>(hp, 0.3, -0.2, z, r, &nx, &ny, &nz);
            if ((sq < 0) != (sd < 0)) ++sign_mismatch;
        }
    }

    // ---- environment: every primitive kind, placed near the iiwa envelope at q0 ----
    std::vector<T> hq(NQ); for(int i=0;i<NQ;++i) hq[i]=0.3*sin(0.9*i)+0.2;
    gc::Sphere<T>  hs[2] = { {0.15, 0.10, 0.60, 0.12}, {-0.20, 0.25, 0.45, 0.08} };
    gc::Capsule<T> hc[1] = { {0.40, -0.30, 0.20,  0.40, -0.30, 0.80,  0.06} };
    gc::Cuboid<T>  hb[1] = { { 0.0, -0.45, 0.55,
                               1.0, 0.0, 0.0, 0.10,
                               0.0, 1.0, 0.0, 0.10,
                               0.0, 0.0, 1.0, 0.15 } };
    // Tilted half-space below the arm (unit normal), so the plane row is non-degenerate but clear.
    T pn[3] = {0.2, -0.1, 1.0};
    T pl = sqrt(pn[0]*pn[0]+pn[1]*pn[1]+pn[2]*pn[2]);
    gc::Plane<T> hpl{ pn[0]/pl, pn[1]/pl, pn[2]/pl, -0.30 };
    T eps = 1e-6;
    CK(cudaMemcpyToSymbol(c_sph, hs,  sizeof(hs)));
    CK(cudaMemcpyToSymbol(c_cap, hc,  sizeof(hc)));
    CK(cudaMemcpyToSymbol(c_box, hb,  sizeof(hb)));
    CK(cudaMemcpyToSymbol(c_pln, &hpl, sizeof(hpl)));
    CK(cudaMemcpyToSymbol(d_eps, &eps, sizeof(T)));

    T *d_q; CK(cudaMalloc(&d_q,NQ*sizeof(T))); CK(cudaMemcpy(d_q,hq.data(),NQ*sizeof(T),cudaMemcpyHostToDevice));
    T *d_pdist,*d_pddist,*d_pnorm,*d_red,*d_pos,*d_r,*d_fd;
    CK(cudaMalloc(&d_pdist, NPAIR*sizeof(T)));      CK(cudaMalloc(&d_pddist, NPAIR*NV*sizeof(T)));
    CK(cudaMalloc(&d_pnorm, 3*NPAIR*sizeof(T)));    CK(cudaMalloc(&d_red,    NS*sizeof(T)));
    CK(cudaMalloc(&d_pos,   3*NS*sizeof(T)));       CK(cudaMalloc(&d_fd,     NPAIR*sizeof(T)));
    CK(cudaMalloc(&d_r,     NS*sizeof(T)));

    analytic_kernel<<<1,128,smem>>>(d_q,m,d_pdist,d_pddist,d_pnorm,d_red,d_pos,d_r); CK(cudaDeviceSynchronize());
    std::vector<T> pdist(NPAIR), pddist(NPAIR*NV), pnorm(3*NPAIR), red(NS), pos(3*NS), rad(NS);
    CK(cudaMemcpy(pdist.data(), d_pdist,  NPAIR*sizeof(T),    cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(pddist.data(),d_pddist, NPAIR*NV*sizeof(T), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(pnorm.data(), d_pnorm,  3*NPAIR*sizeof(T),  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(red.data(),   d_red,    NS*sizeof(T),       cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(pos.data(),   d_pos,    3*NS*sizeof(T),     cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rad.data(),   d_r,      NS*sizeof(T),       cudaMemcpyDeviceToHost));

    // ---- (1) reduction consistency: min over the pair row == the reduced API, BIT-EXACT ----
    int red_mismatch = 0;
    for (int i = 0; i < NS; ++i) {
        T mn = pdist[i*NOBS];
        for (int o = 1; o < NOBS; ++o) if (pdist[i*NOBS+o] < mn) mn = pdist[i*NOBS+o];
        if (mn != red[i]) ++red_mismatch;        // exact: same SDFs, same order, same tie-break
    }

    // ---- (3) plane exactness: row o=4 must be n.p - d - r with the plane's own (constant) normal ----
    // radii come back from the kernel (the baked fp32 table, cast to T exactly as the device does).
    T plane_derr = 0, plane_nerr = 0;
    for (int i = 0; i < NS; ++i) {
        T r = rad[i];
        T expect = hpl.nx*pos[3*i] + hpl.ny*pos[3*i+1] + hpl.nz*pos[3*i+2] - hpl.d - r;
        int pair = i*NOBS + 4;                                     // planes are LAST in the flattened order
        plane_derr = fmax(plane_derr, fabs(pdist[pair] - expect));
        plane_nerr = fmax(plane_nerr, fabs(pnorm[3*pair+0]-hpl.nx) + fabs(pnorm[3*pair+1]-hpl.ny) + fabs(pnorm[3*pair+2]-hpl.nz));
    }

    // ---- (2) pair Jacobian vs central FD ----
    std::vector<T> fd(NPAIR*NV), dp(NPAIR), dm(NPAIR);
    for (int vi=0; vi<NV; ++vi) {
        fd_kernel<<<1,128,smem>>>(d_q,m,vi,+1,d_fd); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(dp.data(),d_fd,NPAIR*sizeof(T),cudaMemcpyDeviceToHost));
        fd_kernel<<<1,128,smem>>>(d_q,m,vi,-1,d_fd); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(dm.data(),d_fd,NPAIR*sizeof(T),cudaMemcpyDeviceToHost));
        for (int p=0;p<NPAIR;++p) fd[p*NV+vi] = (dp[p]-dm[p])/(2*eps);
    }
    T e_jac = 0, jnorm = 0;
    for (int k=0;k<NPAIR*NV;++k) {
        T d = fabs(pddist[k]-fd[k]), s = fabs(pddist[k])+fabs(fd[k])+1e-9;
        e_jac = fmax(e_jac, d/s);
        jnorm += pddist[k]*pddist[k];
    }
    jnorm = sqrt(jnorm);

    printf("NS=%d NOBS=%d NV=%d NPAIR=%d  |J|=%.4f\n", NS, NOBS, NV, NPAIR, jnorm);
    printf("  reduce_mismatch=%d (min over pairs vs collision_distance, bit-exact)\n", red_mismatch);
    printf("  pair_jac_relerr=%.2e   plane_derr=%.2e   plane_nerr=%.2e   plane_sign_mismatch=%d\n",
           e_jac, plane_derr, plane_nerr, sign_mismatch);
    bool ok = (jnorm > 1e-6) && (red_mismatch == 0) && (e_jac < 1e-6)
              && (plane_derr < 1e-12) && (plane_nerr < 1e-15) && (sign_mismatch == 0);
    printf("RESULT: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 3;
}

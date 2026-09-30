// FD-oracle validation for the SELF-collision rows (GATO ask 2026-08-01): the self-pair analogue
// of collision_distance[_gradient] + the pairs twins, over the baked adjacency-excluded pair set.
//
// Certifies on a real spherized robot (iiwa14):
//   (1) REDUCTION CONSISTENCY: self_collision_distance's s_dist[i] == min over sphere i's baked
//       pairs of the un-reduced pair clearance, BIT-EXACTLY, and s_partner[i] indexes a pair
//       achieving that min (-1 iff sphere i has no baked partners).
//   (2) PAIR JACOBIAN: s_ddist[p*NV+vi] = d(d_p)/dq_vi vs central FD of the pair clearance.
//   (3) REDUCED ROW == ARGMIN PAIR ROW: the reduced gradient row of sphere i is BIT-IDENTICAL to
//       the pair-gradient row of its argmin pair. (Orientation-invariant: for pair (j,i) both the
//       normal and the position-gradient difference flip sign, so the product is unchanged.)
//   (4) The baked pair set is non-empty and every pair references valid sphere indices.
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
constexpr int NP = gc::NUM_SELF_COLLISION_PAIRS;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

__device__ T d_eps;

__global__ void analytic_kernel(const T *q0, const grid::robotModel<T> *m,
                                T *d_dist, T *d_norm, int *d_partner, T *d_ddist,
                                T *d_pdist, T *d_pnorm, T *d_pddist) {
    __shared__ T s_pos[3*NS], s_r[NS], s_pg[3*NV*NS];
    __shared__ T s_dist[NS], s_norm[3*NS], s_ddist[NS*NV];
    __shared__ int s_partner[NS];
    __shared__ T s_pdist[NP], s_pnorm[3*NP], s_pddist[NP*NV];
    gc::self_collision_distance_gradient<T>(s_dist, s_ddist, q0, m, s_pos, s_r, s_norm, s_partner, s_pg);
    gc::self_collision_distance_pairs_gradient<T>(s_pdist, s_pddist, q0, m, s_pos, s_r, s_pnorm, s_pg);
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 0; k < NS; ++k)      d_dist[k]    = s_dist[k];
        for (int k = 0; k < 3*NS; ++k)    d_norm[k]    = s_norm[k];
        for (int k = 0; k < NS; ++k)      d_partner[k] = s_partner[k];
        for (int k = 0; k < NS*NV; ++k)   d_ddist[k]   = s_ddist[k];
        for (int k = 0; k < NP; ++k)      d_pdist[k]   = s_pdist[k];
        for (int k = 0; k < 3*NP; ++k)    d_pnorm[k]   = s_pnorm[k];
        for (int k = 0; k < NP*NV; ++k)   d_pddist[k]  = s_pddist[k];
    }
}

// FD: perturb q[vi] += sign*eps, emit the per-pair clearances.
__global__ void fd_kernel(const T *q0, const grid::robotModel<T> *m, int vi, int sign, T *d_pdist) {
    __shared__ T s_q[NQ], s_pos[3*NS], s_r[NS], s_pdist[NP], s_pnorm[3*NP];
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) s_q[i] = q0[i];
    __syncthreads();
    if (threadIdx.x == 0) s_q[vi] += sign * d_eps;
    __syncthreads();
    gc::self_collision_distance_pairs<T>(s_pdist, s_pnorm, s_q, m, s_pos, s_r);
    __syncthreads();
    if (threadIdx.x == 0) for (int k = 0; k < NP; ++k) d_pdist[k] = s_pdist[k];
}

int main(){
    static_assert(NP > 0, "iiwa14 must bake a non-empty self-collision pair set");
    const grid::robotModel<T> *m = grid::init_robotModel<T>();
    size_t s1 = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t s2 = grid::MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t smem = s1 > s2 ? s1 : s2;
    cudaFuncSetAttribute(analytic_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    cudaFuncSetAttribute(fd_kernel,       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);

    std::vector<T> hq(NQ); for(int i=0;i<NQ;++i) hq[i]=0.3*sin(0.9*i)+0.2;
    T eps = 1e-6;
    CK(cudaMemcpyToSymbol(d_eps, &eps, sizeof(T)));

    // baked pair tables (host copies, for the reduction / validity checks)
    std::vector<int> pi(NP), pj(NP);
    CK(cudaMemcpyFromSymbol(pi.data(), gc::g_collision_self_pair_i, NP*sizeof(int)));
    CK(cudaMemcpyFromSymbol(pj.data(), gc::g_collision_self_pair_j, NP*sizeof(int)));

    T *d_q; CK(cudaMalloc(&d_q,NQ*sizeof(T))); CK(cudaMemcpy(d_q,hq.data(),NQ*sizeof(T),cudaMemcpyHostToDevice));
    T *d_dist,*d_norm,*d_ddist,*d_pdist,*d_pnorm,*d_pddist,*d_fdp,*d_fdm; int *d_partner;
    CK(cudaMalloc(&d_dist,  NS*sizeof(T)));       CK(cudaMalloc(&d_norm, 3*NS*sizeof(T)));
    CK(cudaMalloc(&d_partner, NS*sizeof(int)));   CK(cudaMalloc(&d_ddist, NS*NV*sizeof(T)));
    CK(cudaMalloc(&d_pdist, NP*sizeof(T)));       CK(cudaMalloc(&d_pnorm, 3*NP*sizeof(T)));
    CK(cudaMalloc(&d_pddist, NP*NV*sizeof(T)));
    CK(cudaMalloc(&d_fdp, NP*sizeof(T)));         CK(cudaMalloc(&d_fdm, NP*sizeof(T)));

    analytic_kernel<<<1,128,smem>>>(d_q,m,d_dist,d_norm,d_partner,d_ddist,d_pdist,d_pnorm,d_pddist);
    CK(cudaDeviceSynchronize());

    std::vector<T> dist(NS), ddist(NS*NV), pdist(NP), pddist(NP*NV);
    std::vector<int> partner(NS);
    CK(cudaMemcpy(dist.data(),   d_dist,   NS*sizeof(T),    cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(partner.data(),d_partner,NS*sizeof(int),  cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ddist.data(),  d_ddist,  NS*NV*sizeof(T), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(pdist.data(),  d_pdist,  NP*sizeof(T),    cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(pddist.data(), d_pddist, NP*NV*sizeof(T), cudaMemcpyDeviceToHost));

    int fails = 0;

    // (4) pair validity
    for (int p = 0; p < NP; ++p)
        if (pi[p] < 0 || pi[p] >= NS || pj[p] <= pi[p] || pj[p] >= NS) {
            if (++fails <= 5) printf("FAIL pair %d invalid (%d,%d)\n", p, pi[p], pj[p]);
        }

    // (1) reduction consistency, bit-exact, both orientations of each pair
    for (int i = 0; i < NS; ++i) {
        T best = 1e30; int hits = 0;
        for (int p = 0; p < NP; ++p)
            if (pi[p] == i || pj[p] == i) { ++hits; if (pdist[p] < best) best = pdist[p]; }
        if (hits == 0) {
            if (partner[i] != -1 || dist[i] != (T)1e30) {
                if (++fails <= 5) printf("FAIL sphere %d: no pairs but partner=%d dist=%g\n", i, partner[i], (double)dist[i]);
            }
            continue;
        }
        if (dist[i] != best) {
            if (++fails <= 5) printf("FAIL sphere %d: reduced %.17g != pair-min %.17g\n", i, (double)dist[i], (double)best);
        }
        if (partner[i] < 0) { if (++fails <= 5) printf("FAIL sphere %d: partner -1 with %d pairs\n", i, hits); continue; }
        // the (i, partner) pair's clearance must equal the reduced value
        bool found = false;
        for (int p = 0; p < NP; ++p)
            if ((pi[p] == i && pj[p] == partner[i]) || (pj[p] == i && pi[p] == partner[i])) {
                found = true;
                if (pdist[p] != dist[i] && ++fails <= 5)
                    printf("FAIL sphere %d: argmin pair %d dist %.17g != reduced %.17g\n", i, p, (double)pdist[p], (double)dist[i]);
            }
        if (!found && ++fails <= 5) printf("FAIL sphere %d: partner %d not in baked pair set\n", i, partner[i]);
    }

    // (2) pair Jacobian vs central FD
    std::vector<T> fdp(NP), fdm(NP);
    double max_rel = 0.0;
    for (int vi = 0; vi < NV; ++vi) {
        fd_kernel<<<1,128,smem>>>(d_q,m,vi,+1,d_fdp); CK(cudaDeviceSynchronize());
        fd_kernel<<<1,128,smem>>>(d_q,m,vi,-1,d_fdm); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(fdp.data(), d_fdp, NP*sizeof(T), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(fdm.data(), d_fdm, NP*sizeof(T), cudaMemcpyDeviceToHost));
        for (int p = 0; p < NP; ++p) {
            double fd = (double)(fdp[p] - fdm[p]) / (2.0*(double)eps);
            double an = (double)pddist[p*NV + vi];
            double rel = fabs(fd - an) / (1.0 + fabs(fd));
            if (rel > max_rel) max_rel = rel;
            if (rel > 1e-5 && ++fails <= 5)
                printf("FAIL pair %d vi %d: FD %.10g vs analytic %.10g (rel %.3g)\n", p, vi, fd, an, rel);
        }
    }

    // (3) reduced gradient row == argmin pair row, bit-identical
    for (int i = 0; i < NS; ++i) {
        if (partner[i] < 0) {
            for (int vi = 0; vi < NV; ++vi)
                if (ddist[i*NV+vi] != (T)0 && ++fails <= 5)
                    printf("FAIL sphere %d: no-pair row not zero at vi %d\n", i, vi);
            continue;
        }
        for (int p = 0; p < NP; ++p) {
            if (!((pi[p] == i && pj[p] == partner[i]) || (pj[p] == i && pi[p] == partner[i]))) continue;
            for (int vi = 0; vi < NV; ++vi)
                if (ddist[i*NV+vi] != pddist[p*NV+vi] && ++fails <= 5)
                    printf("FAIL sphere %d vi %d: reduced row %.17g != pair row %.17g (pair %d)\n",
                           i, vi, (double)ddist[i*NV+vi], (double)pddist[p*NV+vi], p);
        }
    }

    printf("NS=%d NP=%d max FD rel err %.3g\n", NS, NP, max_rel);
    if (fails == 0) { printf("RESULT: PASS\n"); return 0; }
    printf("RESULT: FAIL (%d)\n", fails);
    return 1;
}

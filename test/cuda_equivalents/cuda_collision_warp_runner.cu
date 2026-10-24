// Validation for grid_collision::warp (G2/G3, HJCD asks 2026-10-04; G7/G10 margin and self toggle,
// 2026-10-23): the warp-scoped verdict and per-sphere clearance computed from caller-held joint
// world transforms must match the block path (grid_collision::config_free / collision_distance) on
// every random configuration x random environment, from warp 0 of a multi-warp block, at every
// block size, and both verdicts must follow the clearance rule: a hit iff any raw clearance
// (environment, plus the self pairs when check_self) is below the margin.
//
// Protocol: argv[1] = block threads (default 64), argv[2] = number of configs (default 200),
// argv[3] = seed, argv[4] = joint amplitude A (default pi), argv[5] = environment x shift
// (1e3 = far away), argv[6] = margin in metres (default 0), argv[7] = check_self (default 1),
// argv[8] = boundary sweep (default 0: off). Each config draws q uniformly in [-A, A] and an
// environment of a few spheres + one capsule + one cuboid placed near the robot (so verdicts mix
// free/hit). The reference verdict comes from the distance surfaces: c = min over spheres of the
// environment clearance (and of the self clearance when check_self); c > margin + eps must be free,
// c < margin - eps must be a hit, and inside the band only block == warp is required (the checkers
// test squared gaps while the distances take a square root, so the two can round apart within a
// few ulps of the threshold). The boundary sweep re-runs every config at margin = c - 1e-4 (must
// be free), c (block == warp; the free/hit split is reported) and c + 1e-4 (must be a hit), and the
// host checks the primitive rule at exactly representable values: contact at the margin is free,
// four ulps of a metre inside is a hit, four outside is free. Prints per-config lines and a final "RESULT PASS|FAIL".
// fp32 (collision change-of-record). Correctness only, no timing.
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>

using T = float;
namespace gc = grid_collision;
constexpr int NQ = grid::NUM_POS;
constexpr int NS = gc::NUM_COLLISION_SPHERES;
constexpr int NX = grid::NUM_JOINTS;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

__global__ void cmp_kernel(const T *d_q, const grid::robotModel<T> *m,
                           const gc::Sphere<T> *d_sph, int n_sph, const gc::Capsule<T> *d_cap, int n_cap,
                           const gc::Cuboid<T> *d_cub, int n_cub, float margin, int check_self,
                           int *d_block_free, int *d_warp_free, T *d_block_dist, float *d_warp_dist,
                           T *d_block_normal, float *d_warp_normal, T *d_self_dist) {
    extern __shared__ T s_dyn[];
    __shared__ T s_pos[3*NS];
    __shared__ T s_r[NS];
    __shared__ T s_dist[NS];
    __shared__ T s_normal[3*NS];
    __shared__ T s_sdist[NS];
    __shared__ T s_snormal[3*NS];
    __shared__ int s_partner[NS];
    __shared__ T s_jointX[16*NX];
    __shared__ T s_XmatsHom[grid::XHOM_T_COUNT];
    __shared__ float w_scratch[gc::warp::W_SCRATCH_FLOATS];
    __shared__ float w_dist[NS];
    __shared__ float w_normal[3*NS];
    gc::Environment<T> env{ d_sph, n_sph, d_cap, n_cap, d_cub, n_cub, nullptr, 0 };
    const bool self_on = check_self != 0;
    // block path (production): extractor + SDF checks (broad->fine signature when 2+ tiers)
#if GRID_COLLISION_NUM_TIERS > 1
    __shared__ T s_bpos[3*gc::NUM_COLLISION_SPHERES_BROAD];
    __shared__ T s_br[gc::NUM_COLLISION_SPHERES_BROAD];
    bool is_free = gc::config_free<T>(d_q, m, env, s_bpos, s_br, s_pos, s_r, nullptr, nullptr, static_cast<T>(margin), self_on);
#else
    bool is_free = gc::config_free<T>(d_q, m, env, s_pos, s_r, nullptr, static_cast<T>(margin), self_on);
#endif
    __syncthreads();
    gc::collision_distance<T>(s_dist, s_normal, d_q, m, env, s_pos, s_r, nullptr);
    __syncthreads();
    gc::self_collision_distance<T>(s_sdist, s_snormal, s_partner, d_q, m, s_pos, s_r, nullptr);
    __syncthreads();
    // warp path: warp 0 builds the joint world transforms (ee_pose_inner_warp) then runs the
    // warp-scoped API; other warps idle (as in a multi-warp IK block)
    grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, nullptr, d_q, m, s_dyn);
    __syncthreads();
    int w_free = -1;
    if (threadIdx.x < 32) {
        grid::ee_pose_inner_warp<T>(s_jointX, s_XmatsHom, d_q, 0);
        __syncwarp();
        w_free = gc::warp::config_free<T>(s_jointX, env, w_scratch, margin, self_on) ? 1 : 0;
        gc::warp::collision_distance<T>(s_jointX, env, w_dist, w_normal, w_scratch);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *d_block_free = is_free ? 1 : 0;
        *d_warp_free = w_free;
        for (int i = 0; i < NS; ++i) { d_block_dist[i] = s_dist[i]; d_warp_dist[i] = w_dist[i]; d_self_dist[i] = s_sdist[i]; }
        for (int i = 0; i < 3*NS; ++i) { d_block_normal[i] = s_normal[i]; d_warp_normal[i] = w_normal[i]; }
    }
}

static unsigned long long g_state = 0x9E3779B97F4A7C15ull;
static float urand() { g_state = g_state * 6364136223846793005ull + 1442695040888963407ull; return (float)((g_state >> 33) & 0xffffff) / (float)0x1000000; }

// The primitive rule at exactly representable values: robot sphere at the origin with radius 0.25
// tested at r + margin against an obstacle sphere of radius 0.25 centred at x = 1, and against the
// plane x = 0 from x = 1. At margin 0.5 (sphere) and 0.75 (plane) the summed radius is exactly 1
// and the clearance equals the margin: free. A margin 2^-21 larger (four ulps of one metre, so the
// sum stays exactly representable) is a hit; 2^-21 smaller is free.
static bool exact_boundary_rule() {
    bool ok = true;
    const float rs = 0.25f, ro = 0.25f, d = ldexpf(1.f, -21);
    const float m_s = 0.5f;
    ok &= !(gc::grid_cc_sphere_sphere<float>(0.f, 0.f, 0.f, rs + m_s, 1.f, 0.f, 0.f, ro) < 0.f);
    ok &=  (gc::grid_cc_sphere_sphere<float>(0.f, 0.f, 0.f, rs + (m_s + d), 1.f, 0.f, 0.f, ro) < 0.f);
    ok &= !(gc::grid_cc_sphere_sphere<float>(0.f, 0.f, 0.f, rs + (m_s - d), 1.f, 0.f, 0.f, ro) < 0.f);
    gc::Plane<float> pl{ 1.f, 0.f, 0.f, 0.f };
    const float m_p = 0.75f;
    ok &= !(gc::grid_cc_sphere_plane<float>(pl, 1.f, 0.f, 0.f, rs + m_p) < 0.f);
    ok &=  (gc::grid_cc_sphere_plane<float>(pl, 1.f, 0.f, 0.f, rs + (m_p + d)) < 0.f);
    ok &= !(gc::grid_cc_sphere_plane<float>(pl, 1.f, 0.f, 0.f, rs + (m_p - d)) < 0.f);
    printf("BOUNDARY_EXACT %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char **argv){
    int threads = argc > 1 ? atoi(argv[1]) : 64;
    int n_cfg = argc > 2 ? atoi(argv[2]) : 200;
    if (argc > 3) g_state ^= (unsigned long long)atoll(argv[3]) * 0x2545F4914F6CDD1Dull;
    const float amp = argc > 4 ? (float)atof(argv[4]) : 3.14159265f;
    const float env_shift = argc > 5 ? (float)atof(argv[5]) : 0.f;   // added to every obstacle x (1e3 = environment far away)
    const float margin = argc > 6 ? (float)atof(argv[6]) : 0.f;
    const int check_self = argc > 7 ? atoi(argv[7]) : 1;
    const int boundary = argc > 8 ? atoi(argv[8]) : 0;
    const float eps = 1e-5f;        // the rounding band between squared-gap tests and square-root distances
    const float step = 1e-4f;       // the boundary sweep's offset on either side of the threshold
    const bool exact_ok = exact_boundary_rule();
    const grid::robotModel<T> *d_m = grid::init_robotModel<T>();
    size_t smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
    T *d_q, *d_bd, *d_bn, *d_sd; float *d_wd, *d_wn; int *d_bf, *d_wf;
    gc::Sphere<T> *d_sph; gc::Capsule<T> *d_cap; gc::Cuboid<T> *d_cub;
    CK(cudaMalloc(&d_q, NQ*sizeof(T))); CK(cudaMalloc(&d_bd, NS*sizeof(T))); CK(cudaMalloc(&d_wd, NS*sizeof(float)));
    CK(cudaMalloc(&d_sd, NS*sizeof(T)));
    CK(cudaMalloc(&d_bn, 3*NS*sizeof(T))); CK(cudaMalloc(&d_wn, 3*NS*sizeof(float)));
    CK(cudaMalloc(&d_bf, sizeof(int))); CK(cudaMalloc(&d_wf, sizeof(int)));
    CK(cudaMalloc(&d_sph, 4*sizeof(gc::Sphere<T>))); CK(cudaMalloc(&d_cap, sizeof(gc::Capsule<T>))); CK(cudaMalloc(&d_cub, sizeof(gc::Cuboid<T>)));
    cudaFuncSetAttribute(cmp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    std::vector<T> hq(NQ), bd(NS), bn(3*NS), sd(NS); std::vector<float> wd(NS), wn(3*NS);
    int n_hit = 0, n_mismatch = 0, n_ref_mismatch = 0, n_band = 0, n_boundary_fail = 0, n_at_free = 0, n_at_hit = 0, n_boundary = 0;
    float worst = 0.f;
    for (int k = 0; k < n_cfg; ++k) {
        for (int i = 0; i < NQ; ++i) hq[i] = (urand() * 2.f - 1.f) * amp;
        gc::Sphere<T> sph[4]; for (int i = 0; i < 4; ++i) sph[i] = { env_shift + (urand()-0.5f)*1.6f, (urand()-0.5f)*1.6f, urand()*1.2f, 0.03f + 0.12f*urand() };
        gc::Capsule<T> cap{ env_shift + (urand()-0.5f)*1.6f, (urand()-0.5f)*1.6f, urand()*1.2f, (urand()-0.5f)*1.6f, (urand()-0.5f)*1.6f, urand()*1.2f, 0.03f + 0.08f*urand() };
        gc::Cuboid<T> cub{ env_shift + (urand()-0.5f)*1.6f, (urand()-0.5f)*1.6f, urand()*1.2f, 1,0,0, 0.05f+0.15f*urand(), 0,1,0, 0.05f+0.15f*urand(), 0,0,1, 0.05f+0.15f*urand() };
        int n_sph = 1 + (int)(urand() * 3.99f);
        CK(cudaMemcpy(d_q, hq.data(), NQ*sizeof(T), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_sph, sph, 4*sizeof(gc::Sphere<T>), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_cap, &cap, sizeof(cap), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_cub, &cub, sizeof(cub), cudaMemcpyHostToDevice));
        int n_cap = urand() < 0.6f ? 1 : 0, n_cub = urand() < 0.6f ? 1 : 0;
        // one launch at a given margin; returns the two verdicts and fills the distance vectors
        auto launch = [&](float mg, int &bf, int &wf) -> int {
            cmp_kernel<<<1, threads, smem>>>(d_q, d_m, d_sph, n_sph, d_cap, n_cap, d_cub, n_cub, mg, check_self, d_bf, d_wf, d_bd, d_wd, d_bn, d_wn, d_sd);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(&bf, d_bf, sizeof(int), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&wf, d_wf, sizeof(int), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(bd.data(), d_bd, NS*sizeof(T), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(wd.data(), d_wd, NS*sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(bn.data(), d_bn, 3*NS*sizeof(T), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(wn.data(), d_wn, 3*NS*sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sd.data(), d_sd, NS*sizeof(T), cudaMemcpyDeviceToHost));
            return 0;
        };
        int bf, wf;
        if (launch(margin, bf, wf)) return 2;
        float dmax = 0.f;
        for (int i = 0; i < NS; ++i) { float d = fabsf((float)bd[i] - wd[i]); if (d > dmax) dmax = d; }
        for (int i = 0; i < 3*NS; ++i) { float d = fabsf((float)bn[i] - wn[i]); if (d > dmax) dmax = d; }
        // reference clearance: environment on every sphere, self pairs only when they are checked
        float c = FLT_MAX;
        for (int i = 0; i < NS; ++i) { c = fminf(c, (float)bd[i]); if (check_self) c = fminf(c, (float)sd[i]); }
        int ref = -1;                                  // -1 = inside the rounding band
        if (c > margin + eps) ref = 1; else if (c < margin - eps) ref = 0;
        if (bf != wf) ++n_mismatch;
        if (ref < 0) ++n_band; else if (bf != ref) ++n_ref_mismatch;
        if (!bf) ++n_hit;
        if (dmax > worst) worst = dmax;
        printf("C %d block=%d warp=%d ref=%d clearance=%.6f dist_maxdiff=%.3e\n", k, bf, wf, ref, c, dmax);
        if (boundary && c > 2.f * step && c < 10.f) {
            ++n_boundary;
            int b1, w1, b2, w2, b3, w3;
            if (launch(c - step, b1, w1)) return 2;    // just below the threshold: free
            if (launch(c, b2, w2)) return 2;           // at the threshold: block == warp, split reported
            if (launch(c + step, b3, w3)) return 2;    // just above the threshold: hit
            bool okb = (b1 == 1 && w1 == 1) && (b2 == w2) && (b3 == 0 && w3 == 0);
            if (!okb) ++n_boundary_fail;
            if (b2) ++n_at_free; else ++n_at_hit;
            printf("B %d below=%d/%d at=%d/%d above=%d/%d %s\n", k, b1, w1, b2, w2, b3, w3, okb ? "ok" : "FAIL");
        }
    }
    bool ok = exact_ok && (n_mismatch == 0) && (n_ref_mismatch == 0) && (n_boundary_fail == 0) && (worst == 0.f);
    printf("SUMMARY configs=%d hits=%d mismatches=%d ref_mismatches=%d band=%d boundary=%d boundary_fail=%d at_free=%d at_hit=%d worst_dist_diff=%.3e margin=%.4f check_self=%d threads=%d NS=%d\n",
           n_cfg, n_hit, n_mismatch, n_ref_mismatch, n_band, n_boundary, n_boundary_fail, n_at_free, n_at_hit, worst, margin, check_self, threads, NS);
    printf("RESULT %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

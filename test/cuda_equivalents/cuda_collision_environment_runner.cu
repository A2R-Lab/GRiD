// Validation for the environment ownership API (HJCD-IK ask G8a, 2026-10-23):
// grid_collision::make_host_environment (arrays -> validated host lists), HostEnvironment::view()
// (host pointers for the host primitives), DeviceEnvironment (deep copy, device recorded, freed
// on destruction, move-only) and grid_cc_cuboid_from_quaternion.
//
// Checks, each printed as "CHECK <name> PASS|FAIL", then "RESULT PASS|FAIL":
//   layout      identity quaternion gives identity axes and half extents; a 90 degree yaw turns
//               axis u into +y and v into -x; a (0,0,2) plane normal with d = 0.2 is stored as
//               (0,0,1) with d = 0.1.
//   rejects     a negative radius, a non-finite centre, a zero quaternion, a zero normal and a
//               count without data each throw std::invalid_argument.
//   parity      the owned device view and a hand-uploaded copy of the same lists give bit-identical
//               config_free verdicts and collision_distance outputs on random configurations.
//   host        grid_cc_sphere_in_environment on the host view equals the device evaluation on the
//               owned view at the same points.
//   lifetime    free device memory after destruction equals the value before construction; a moved
//               from object is empty and the destination keeps the lists.
// argv[1] = block threads (default 64), argv[2] = configs (default 100), argv[3] = seed.
#define GRID_HEADER
#include "grid.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <vector>

using T = float;
namespace gc = grid_collision;
constexpr int NQ = grid::NUM_POS;
constexpr int NS = gc::NUM_COLLISION_SPHERES;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA ERR %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(e)); return 2; } }while(0)

__global__ void eval_kernel(const T *d_q, const grid::robotModel<T> *m, gc::Environment<T> env,
                            int *d_free, T *d_dist, T *d_normal) {
    extern __shared__ T s_dyn[];   // the batched extractor's dynamic arena (MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES)
    (void)s_dyn;
    __shared__ T s_pos[3*NS];
    __shared__ T s_r[NS];
    __shared__ T s_dist[NS];
    __shared__ T s_normal[3*NS];
#if GRID_COLLISION_NUM_TIERS > 1
    __shared__ T s_bpos[3*gc::NUM_COLLISION_SPHERES_BROAD];
    __shared__ T s_br[gc::NUM_COLLISION_SPHERES_BROAD];
    bool is_free = gc::config_free<T>(d_q, m, env, s_bpos, s_br, s_pos, s_r, nullptr);
#else
    bool is_free = gc::config_free<T>(d_q, m, env, s_pos, s_r, nullptr);
#endif
    __syncthreads();
    gc::collision_distance<T>(s_dist, s_normal, d_q, m, env, s_pos, s_r, nullptr);
    __syncthreads();
    if (threadIdx.x == 0) {
        *d_free = is_free ? 1 : 0;
        for (int i = 0; i < NS; ++i) d_dist[i] = s_dist[i];
        for (int i = 0; i < 3*NS; ++i) d_normal[i] = s_normal[i];
    }
}

__global__ void point_kernel(gc::Environment<T> env, const T *d_pts, int n, int *d_hit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d_hit[i] = gc::grid_cc_sphere_in_environment<T>(env, d_pts[4*i], d_pts[4*i+1], d_pts[4*i+2], d_pts[4*i+3]) ? 1 : 0;
}

static unsigned long long g_state = 0x9E3779B97F4A7C15ull;
static float urand() { g_state = g_state * 6364136223846793005ull + 1442695040888963407ull; return (float)((g_state >> 33) & 0xffffff) / (float)0x1000000; }
static bool near(float a, float b, float tol = 1e-6f) { return fabsf(a - b) <= tol; }

template <typename F>
static bool throws_invalid(F f) {
    try { f(); } catch (const std::invalid_argument &) { return true; } catch (...) { return false; }
    return false;
}

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 64;
    int n_cfg = argc > 2 ? atoi(argv[2]) : 100;
    if (argc > 3) g_state ^= (unsigned long long)atoll(argv[3]) * 0x2545F4914F6CDD1Dull;
    bool all_ok = true;
    auto report = [&](const char *name, bool ok) { printf("CHECK %s %s\n", name, ok ? "PASS" : "FAIL"); all_ok &= ok; };

    // ---- layout ----
    {
        gc::Cuboid<T> b = gc::grid_cc_cuboid_from_quaternion<T>(1, 2, 3, 1, 0, 0, 0, 0.2f, 0.4f, 0.6f);
        bool ok = near(b.ux, 1) && near(b.uy, 0) && near(b.uz, 0) && near(b.vx, 0) && near(b.vy, 1) && near(b.vz, 0)
               && near(b.wx, 0) && near(b.wy, 0) && near(b.wz, 1) && near(b.hu, 0.1f) && near(b.hv, 0.2f) && near(b.hw, 0.3f)
               && near(b.cx, 1) && near(b.cy, 2) && near(b.cz, 3);
        const float s2 = sqrtf(0.5f);
        gc::Cuboid<T> y = gc::grid_cc_cuboid_from_quaternion<T>(0, 0, 0, 2*s2, 0, 0, 2*s2, 1, 1, 1);   // un-normalized 90 deg yaw
        ok &= near(y.ux, 0) && near(y.uy, 1) && near(y.uz, 0) && near(y.vx, -1) && near(y.vy, 0) && near(y.vz, 0) && near(y.wz, 1);
        const T plane[4] = { 0, 0, 2, 0.2f };
        gc::HostEnvironment<T> env = gc::make_host_environment<T>(nullptr, 0, nullptr, 0, nullptr, 0, plane, 1);
        ok &= env.planes.size() == 1 && near(env.planes[0].nz, 1) && near(env.planes[0].d, 0.1f) && env.size() == 1;
        report("layout", ok);
    }
    // ---- rejects ----
    {
        const T bad_r[4] = { 0, 0, 0, -0.1f };
        const T nan_c[4] = { nanf(""), 0, 0, 0.1f };
        const T zero_q[10] = { 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 };
        const T zero_n[4] = { 0, 0, 0, 0.1f };
        const T good[4] = { 0, 0, 0, 0.1f };
        bool ok = throws_invalid([&]{ gc::make_host_environment<T>(bad_r, 1, nullptr, 0, nullptr, 0); })
               && throws_invalid([&]{ gc::make_host_environment<T>(nan_c, 1, nullptr, 0, nullptr, 0); })
               && throws_invalid([&]{ gc::make_host_environment<T>(nullptr, 0, nullptr, 0, zero_q, 1); })
               && throws_invalid([&]{ gc::make_host_environment<T>(nullptr, 0, nullptr, 0, nullptr, 0, zero_n, 1); })
               && throws_invalid([&]{ gc::make_host_environment<T>(nullptr, 2, nullptr, 0, nullptr, 0); })
               && !throws_invalid([&]{ gc::make_host_environment<T>(good, 1, nullptr, 0, nullptr, 0); });
        report("rejects", ok);
    }
    // ---- the shared environment for parity, host and lifetime ----
    const grid::robotModel<T> *d_m = grid::init_robotModel<T>();
    const size_t smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(eval_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    std::vector<T> sph, cap, cub, pla;
    for (int i = 0; i < 3; ++i) { sph.insert(sph.end(), { (urand()-0.5f)*1.2f, (urand()-0.5f)*1.2f, urand()*1.0f, 0.05f + 0.1f*urand() }); }
    for (int i = 0; i < 2; ++i) { cap.insert(cap.end(), { (urand()-0.5f)*1.2f, (urand()-0.5f)*1.2f, urand()*1.0f, (urand()-0.5f)*1.2f, (urand()-0.5f)*1.2f, urand()*1.0f, 0.04f + 0.06f*urand() }); }
    for (int i = 0; i < 2; ++i) { float a = urand() * 3.14159f; cub.insert(cub.end(), { (urand()-0.5f)*1.2f, (urand()-0.5f)*1.2f, urand()*1.0f, cosf(a/2), 0, 0, sinf(a/2), 0.1f + 0.2f*urand(), 0.1f + 0.2f*urand(), 0.1f + 0.2f*urand() }); }
    pla.insert(pla.end(), { 0, 0, 1, -0.05f });   // floor just below the base
    size_t free_before = 0, total = 0;
    CK(cudaDeviceSynchronize()); CK(cudaMemGetInfo(&free_before, &total));
    {
        gc::HostEnvironment<T> host = gc::make_host_environment<T>(sph.data(), 3, cap.data(), 2, cub.data(), 2, pla.data(), 1);
        gc::DeviceEnvironment<T> owned(host);
        bool shape_ok = owned.size() == 8 && !owned.empty() && owned.device() >= 0 && owned.view().n_planes == 1;
        // hand upload of the same lists (the pre-G8a pattern)
        gc::Sphere<T> *d_s; gc::Capsule<T> *d_c; gc::Cuboid<T> *d_b; gc::Plane<T> *d_p;
        CK(cudaMalloc(&d_s, host.spheres.size()*sizeof(gc::Sphere<T>))); CK(cudaMemcpy(d_s, host.spheres.data(), host.spheres.size()*sizeof(gc::Sphere<T>), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_c, host.capsules.size()*sizeof(gc::Capsule<T>))); CK(cudaMemcpy(d_c, host.capsules.data(), host.capsules.size()*sizeof(gc::Capsule<T>), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_b, host.cuboids.size()*sizeof(gc::Cuboid<T>))); CK(cudaMemcpy(d_b, host.cuboids.data(), host.cuboids.size()*sizeof(gc::Cuboid<T>), cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_p, host.planes.size()*sizeof(gc::Plane<T>))); CK(cudaMemcpy(d_p, host.planes.data(), host.planes.size()*sizeof(gc::Plane<T>), cudaMemcpyHostToDevice));
        gc::Environment<T> manual{ d_s, 3, d_c, 2, d_b, 2, d_p, 1 };
        T *d_q, *d_dist[2], *d_normal[2]; int *d_free[2];
        CK(cudaMalloc(&d_q, NQ*sizeof(T)));
        for (int k = 0; k < 2; ++k) { CK(cudaMalloc(&d_dist[k], NS*sizeof(T))); CK(cudaMalloc(&d_normal[k], 3*NS*sizeof(T))); CK(cudaMalloc(&d_free[k], sizeof(int))); }
        std::vector<T> hq(NQ), dist0(NS), dist1(NS), nrm0(3*NS), nrm1(3*NS);
        int mismatches = 0, hits = 0;
        for (int c = 0; c < n_cfg; ++c) {
            for (int i = 0; i < NQ; ++i) hq[i] = (urand() * 2.f - 1.f) * 3.14159265f;
            CK(cudaMemcpy(d_q, hq.data(), NQ*sizeof(T), cudaMemcpyHostToDevice));
            eval_kernel<<<1, threads, smem>>>(d_q, d_m, owned.view(), d_free[0], d_dist[0], d_normal[0]);
            eval_kernel<<<1, threads, smem>>>(d_q, d_m, manual, d_free[1], d_dist[1], d_normal[1]);
            CK(cudaDeviceSynchronize());
            int f0, f1; CK(cudaMemcpy(&f0, d_free[0], sizeof(int), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&f1, d_free[1], sizeof(int), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(dist0.data(), d_dist[0], NS*sizeof(T), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(dist1.data(), d_dist[1], NS*sizeof(T), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(nrm0.data(), d_normal[0], 3*NS*sizeof(T), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(nrm1.data(), d_normal[1], 3*NS*sizeof(T), cudaMemcpyDeviceToHost));
            bool same = (f0 == f1);
            for (int i = 0; i < NS; ++i) same &= (dist0[i] == dist1[i]);
            for (int i = 0; i < 3*NS; ++i) same &= (nrm0[i] == nrm1[i]);
            if (!same) ++mismatches;
            if (!f0) ++hits;
        }
        printf("PARITY configs=%d hits=%d mismatches=%d\n", n_cfg, hits, mismatches);
        report("parity", shape_ok && mismatches == 0 && hits > 0 && hits < n_cfg);
        // host primitives on the host view against the device on the owned view
        const int n_pts = 256;
        std::vector<T> pts(4 * n_pts); std::vector<int> host_hit(n_pts), dev_hit(n_pts);
        for (int i = 0; i < n_pts; ++i) { pts[4*i] = (urand()-0.5f)*1.6f; pts[4*i+1] = (urand()-0.5f)*1.6f; pts[4*i+2] = (urand()-0.5f)*0.4f + 0.3f; pts[4*i+3] = 0.02f + 0.1f*urand(); }
        gc::Environment<T> hview = host.view();
        for (int i = 0; i < n_pts; ++i) host_hit[i] = gc::grid_cc_sphere_in_environment<T>(hview, pts[4*i], pts[4*i+1], pts[4*i+2], pts[4*i+3]) ? 1 : 0;
        T *d_pts; int *d_hit;
        CK(cudaMalloc(&d_pts, pts.size()*sizeof(T))); CK(cudaMalloc(&d_hit, n_pts*sizeof(int)));
        CK(cudaMemcpy(d_pts, pts.data(), pts.size()*sizeof(T), cudaMemcpyHostToDevice));
        point_kernel<<<(n_pts + 127) / 128, 128>>>(owned.view(), d_pts, n_pts, d_hit);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(dev_hit.data(), d_hit, n_pts*sizeof(int), cudaMemcpyDeviceToHost));
        int host_hits = 0, diff = 0;
        for (int i = 0; i < n_pts; ++i) { host_hits += host_hit[i]; diff += (host_hit[i] != dev_hit[i]); }
        printf("HOST points=%d hits=%d differences=%d\n", n_pts, host_hits, diff);
        report("host", diff == 0 && host_hits > 0 && host_hits < n_pts);
        // move semantics
        gc::DeviceEnvironment<T> moved(std::move(owned));
        bool move_ok = owned.empty() && owned.view().spheres == nullptr && owned.device() < 0
                    && moved.size() == 8 && moved.view().spheres != nullptr;
        gc::DeviceEnvironment<T> assigned; assigned = std::move(moved);
        move_ok &= moved.empty() && assigned.size() == 8;
        eval_kernel<<<1, threads, smem>>>(d_q, d_m, assigned.view(), d_free[0], d_dist[0], d_normal[0]);
        CK(cudaDeviceSynchronize());
        report("move", move_ok);
        CK(cudaFree(d_s)); CK(cudaFree(d_c)); CK(cudaFree(d_b)); CK(cudaFree(d_p)); CK(cudaFree(d_q)); CK(cudaFree(d_pts)); CK(cudaFree(d_hit));
        for (int k = 0; k < 2; ++k) { CK(cudaFree(d_dist[k])); CK(cudaFree(d_normal[k])); CK(cudaFree(d_free[k])); }
    }   // `assigned` releases here
    size_t free_after = 0;
    CK(cudaDeviceSynchronize()); CK(cudaMemGetInfo(&free_after, &total));
    printf("LIFETIME free_before=%zu free_after=%zu\n", free_before, free_after);
    report("lifetime", free_after == free_before);
    printf("RESULT %s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

// Native capsule margins: generated cascade/single-tier vs independent fine-only
// checks and the consumer's two scalar calls. No timing, one block per configuration.
#define GRID_HEADER
#include "grid.cuh"
#include "margin_config.h"
#include <cstdio>
#include <vector>
namespace gc = grid_collision;
using T = double;
constexpr int NR = gc::NUM_COLLISION_ROWS;
constexpr int NQ = grid::NUM_POS;
#define CK(x) do { if ((x) != cudaSuccess) { printf("CUDA failure line %d\n", __LINE__); return 2; } } while (0)

__global__ void check(const T *q, const grid::robotModel<T> *model,
                      gc::Sphere<T> obstacle, gc::CollisionMargins<T> margins,
                      bool self_on, int *out) {
    extern __shared__ T scratch[];
    __shared__ T pos[6*NR], radii[NR];
    gc::Environment<T> env{}; env.spheres = &obstacle; env.n_spheres = 1;
    gc::Environment<T> empty{};
#if TEST_NATIVE_TWO_TIERS
    __shared__ T broad[3*gc::NUM_COLLISION_SPHERES_BROAD], br[gc::NUM_COLLISION_SPHERES_BROAD];
    #define CONFIG(e, mg, self) gc::config_free<T>(q, model, e, broad, br, pos, radii, nullptr, nullptr, mg, self)
#else
    #define CONFIG(e, mg, self) gc::config_free<T>(q, model, e, pos, radii, nullptr, mg, self)
#endif
    const bool combined = CONFIG(env, margins, self_on);
    __syncthreads();
    bool fine = true;
    if (self_on && gc::grid_cc_self_collision_capsules<T>(pos, radii,
            gc::g_collision_self_cc_ranges, gc::NUM_COLLISION_SELF_CC_RANGES, margins.self_margin)) fine = false;
    for (int i = 0; i < NR; ++i)
        if (gc::grid_cc_capsule_in_environment<T>(env,
                gc::grid_cc_row_capsule<T>(pos, radii, i, margins.environment_margin))) fine = false;
    __syncthreads();
    const bool environment = CONFIG(env, margins.environment_margin, false);
    __syncthreads();
    const bool self = CONFIG(empty, margins.self_margin, self_on);
    __syncthreads();
    out[threadIdx.x] = combined == fine && combined == (environment && self) ? int(combined) : -1;
    #undef CONFIG
}

static unsigned long long state = 771;
static T random_value() { state = state * 6364136223846793005ull + 1; return T((state >> 33) & 0xffffff)/T(0x1000000); }

int main(int argc, char **argv) {
    const int cases = argc > 1 ? 12 : 200; // compact sanitizer variant retains every margin/thread combination
    const auto *model = grid::init_robotModel<T>();
    T *q; int *out; CK(cudaMalloc(&q, NQ*sizeof(T))); CK(cudaMalloc(&out, 128*sizeof(int)));
    size_t smem = grid::MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>();
#if TEST_NATIVE_TWO_TIERS
    smem = std::max(smem, grid::MULTI_TARGET_POSITION_BROAD_DYNAMIC_SHARED_MEM_BYTES<T>());
#endif
    CK(cudaFuncSetAttribute(check, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem)));
    int hits = 0, free = 0, env_changes = 0, self_changes = 0;
    std::vector<T> hostq(NQ); int host[128];
    for (int k = 0; k < cases; ++k) {
        for (auto &v : hostq) v = (random_value()*2-1)*T(3.14);
        CK(cudaMemcpy(q, hostq.data(), NQ*sizeof(T), cudaMemcpyHostToDevice));
        gc::Sphere<T> obs{(random_value()-.5)*1.6, (random_value()-.5)*1.6, random_value()*1.2, .1};
        int verdict[2][4];
        const gc::CollisionMargins<T> margins[] = {{0,0}, {.1,0}, {0,.2}, {.1,.2}};
        for (int self = 0; self <= 1; ++self) for (int mg = 0; mg < 4; ++mg) {
            int baseline = -1;
            for (int threads : {1, 32, 128}) {
                check<<<1, threads, smem>>>(q, model, obs, margins[mg], bool(self), out);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(host, out, threads*sizeof(int), cudaMemcpyDeviceToHost));
                if (threads == 1) baseline = host[0];
                for (int t = 0; t < threads; ++t) if (host[t] < 0 || host[t] != baseline) {
                    printf("mismatch config=%d self=%d margin=%d thread=%d\n", k, self, mg, t); return 1;
                }
            }
            verdict[self][mg] = baseline;
            if (baseline) ++free; else ++hits;
        }
        env_changes += verdict[0][0] != verdict[0][1];
        self_changes += verdict[1][0] != verdict[1][2];
        if (verdict[0][0] != verdict[0][2] || verdict[0][1] != verdict[0][3]) return 1;
    }
    printf("hits=%d free=%d env_changes=%d self_changes=%d\n", hits, free, env_changes, self_changes);
    if (!hits || !free || (argc == 1 && (!env_changes || !self_changes))) return 1;
    CK(cudaFree(q)); CK(cudaFree(out));
    puts("RESULT: PASS");
}

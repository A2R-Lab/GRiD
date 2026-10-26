// Exactly representable independent thresholds for the static sphere/capsule cascades.
#include "grid_collision_geometry.cuh"
#include <cstdio>
#include <limits>
#include <type_traits>
namespace gc = grid_collision;

template<class T>
__host__ __device__ bool thresholds() {
    const T p[] = {0, 0, 1, 1, 0, 1}, r[] = {T(.25), T(.25)};
    const T seg[] = {0, T(-.125), 1, 0, T(.125), 1,
                    1, T(-.125), 1, 1, T(.125), 1};
    const T cover[] = {T(.375), T(.375)};
    const int links[] = {0, 1}, pairs[] = {0, 1, 1};
    const gc::Plane<T> plane{0, 0, 1, 0};
    gc::Environment<T> env{};
    env.planes = &plane; env.n_planes = 1;
    const T step = T(1.0 / 1048576);
    for (int e = -1; e <= 1; ++e) for (int s = -1; s <= 1; ++s) {
        gc::CollisionMargins<T> margins(T(.75) + T(e)*step, T(.5) + T(s)*step);
        for (int check_self = 0; check_self <= 1; ++check_self) {
            const bool expected = e <= 0 && (!check_self || s <= 0);
            // Deliberately infer T: the non-deduced margin parameter must accept both
            // the value object and a scalar without disrupting template deduction.
            const bool sphere = gc::grid_cc_config_free(env, p, r, pairs, 1, 2, links,
                p, r, pairs, 1, 2, links, nullptr, margins, check_self);
            const bool capsule = gc::grid_cc_config_free_capsule(env, p, cover, pairs, 1, 2, links,
                seg, r, pairs, 1, 2, links, nullptr, margins, check_self);
            if (sphere != expected || capsule != expected) return false;
        }
    }
    const T common[] = {T(0), T(.5), T(.5)+step, T(.75), T(1)};
    for (T margin : common) {
        const bool expected = margin <= T(.5);
        if (gc::grid_cc_config_free(env, p, r, pairs, 1, 2, links,
                p, r, pairs, 1, 2, links, nullptr, margin) != expected) return false;
        if (gc::grid_cc_config_free_capsule(env, p, cover, pairs, 1, 2, links,
                seg, r, pairs, 1, 2, links, nullptr, margin) != expected) return false;
    }
    return true;
}

__global__ void check(int *out) { out[threadIdx.x] = thresholds<float>() && thresholds<double>(); }

template<class T> bool factory() {
    auto zero = gc::make_collision_margins<T>();
    auto common = gc::make_collision_margins<T>(T(.1));
    auto env = gc::make_collision_margins<T>(T(.1), T(0));
    auto self = gc::make_collision_margins<T>(T(.1), std::nullopt, T(0));
    auto both = gc::make_collision_margins<T>(T(.1), T(.2), T(.3));
    if (zero.self_margin != 0 || zero.environment_margin != 0 ||
        common.environment_margin != T(.1) || common.self_margin != T(.1) ||
        env.environment_margin != 0 || env.self_margin != T(.1) ||
        self.environment_margin != T(.1) || self.self_margin != 0 ||
        both.environment_margin != T(.2) || both.self_margin != T(.3)) return false;
    for (T bad : {T(-1), std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN()}) {
        for (int side = 0; side < 3; ++side) {
            try {
                (void)gc::make_collision_margins<T>(side == 0 ? bad : T(0),
                    side == 1 ? bad : T(0), side == 2 ? bad : T(0));
                return false;
            } catch (const std::invalid_argument &) {}
        }
    }
    return true;
}

int main() {
    if (!factory<float>() || !factory<double>() || !thresholds<float>() || !thresholds<double>()) return 1;
    int *out; int host[128];
    if (cudaMalloc(&out, sizeof(host)) != cudaSuccess) return 2;
    for (int threads : {1, 32, 128}) {
        check<<<1, threads>>>(out);
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(host, out, threads*sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return 2;
        for (int i = 0; i < threads; ++i) if (host[i] != 1) return 1;
    }
    cudaFree(out);
    puts("MARGINS_FACTORY_BOUNDARY PASS");
    puts("RESULT: PASS");
}

// grid_collision static geometry header (W3 Component E).
//
// Robot-agnostic, hand-written (peer to the GLASS linalg headers), templated on <T> with
// fp32 the expected/default instantiation (geometry precision is ample; halves spill).
// Consumes sphere WORLD positions produced by the batched multi-target extractor
// (grid::multi_target_position) and per-robot baked self_cc_ranges from the generated
// grid_collision namespace (Component D). Ported from the pRRTC collision primitives in
// HJCD-IK-grid-glass csrc/collision/{utils.cuh, environment.hh, prrtc_collision.cuh}.
//
// SDF convention (matches the reference): every primitive returns the SQUARED GAP
//   squared_gap := d2 - r_sum^2 ,   value < 0  <=>  in collision.
// Keeping the squared form avoids a sqrt on the hot path. The true signed clearance is
// sign(g)*sqrt(|g|) when a metric value is needed (differentiable path, Phase 2).
#pragma once
#include <cuda_runtime.h>
#include <cmath>   // sqrtf/sqrt for the differentiable (true-distance + normal) path

namespace grid_collision {

// ------------------------------------------------------------------ shapes
template <typename T>
struct Sphere { T x, y, z, r; };

template <typename T>
struct Capsule { T ax, ay, az, bx, by, bz, r; };   // segment endpoints a,b + radius

template <typename T>
struct Cuboid {                                     // oriented box: center c + 3 axes*half-extent
    T cx, cy, cz;
    T ux, uy, uz, hu;   // axis u (unit) and half-extent hu
    T vx, vy, vz, hv;
    T wx, wy, wz, hw;
};

// Runtime obstacle set (NOT baked — matches the reference Environment<T>). Pointers + counts;
// device upload deep-copies each list then patches these members (Component D upload contract).
template <typename T>
struct Environment {
    const Sphere<T>  *spheres;  int n_spheres;
    const Capsule<T> *capsules; int n_capsules;
    const Cuboid<T>  *cuboids;  int n_cuboids;
};

// ------------------------------------------------------------------ helpers
// Precision-correct |v| for both T=float and T=double, host + device (no <math.h> dependency,
// no float->double promotion that a C-style fabs would introduce for T=float).
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_abs(T v) { return v < static_cast<T>(0) ? -v : v; }

template <typename T>
__host__ __device__ __forceinline__ T grid_cc_clamp01(T t) {
    return t < static_cast<T>(0) ? static_cast<T>(0) : (t > static_cast<T>(1) ? static_cast<T>(1) : t);
}

template <typename T>
__host__ __device__ __forceinline__ T grid_cc_sql2_3(T ax, T ay, T az, T bx, T by, T bz) {
    T dx = ax - bx, dy = ay - by, dz = az - bz;
    return dx * dx + dy * dy + dz * dz;
}

// precision-correct sqrt (float->sqrtf, double->sqrt), host + device, no float->double promotion.
__host__ __device__ __forceinline__ float  grid_cc_sqrt_impl(float v)  { return sqrtf(v); }
__host__ __device__ __forceinline__ double grid_cc_sqrt_impl(double v) { return sqrt(v); }
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_sqrt(T v) { return grid_cc_sqrt_impl(v); }

// ------------------------------------------------------------------ SDFs (squared_gap; <0 = collision)
// sphere vs sphere
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_sphere_sphere(
        T ax, T ay, T az, T ar, T bx, T by, T bz, T br) {
    T rs = ar + br;
    return grid_cc_sql2_3<T>(ax, ay, az, bx, by, bz) - rs * rs;
}

// sphere vs capsule (segment): closest point on segment a->b to the sphere center, t clamped to [0,1].
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_sphere_capsule(const Capsule<T> &c, T x, T y, T z, T r) {
    T abx = c.bx - c.ax, aby = c.by - c.ay, abz = c.bz - c.az;
    T apx = x - c.ax,    apy = y - c.ay,    apz = z - c.az;
    T denom = abx * abx + aby * aby + abz * abz;
    T t = denom > static_cast<T>(0) ? (apx * abx + apy * aby + apz * abz) / denom : static_cast<T>(0);
    t = grid_cc_clamp01<T>(t);
    T qx = c.ax + t * abx, qy = c.ay + t * aby, qz = c.az + t * abz;
    T rs = r + c.r;
    return grid_cc_sql2_3<T>(x, y, z, qx, qy, qz) - rs * rs;
}

// sphere vs oriented cuboid: project sphere-center offset onto each OBB axis, clamp to the slab,
// distance to the box surface is the norm of the outside-slab excess. Center inside box -> excess 0
// on every axis -> returns -r^2 < 0 (collision), which is correct.
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_sphere_cuboid(const Cuboid<T> &b, T x, T y, T z, T r) {
    T dx = x - b.cx, dy = y - b.cy, dz = z - b.cz;
    T pu = dx * b.ux + dy * b.uy + dz * b.uz;
    T pv = dx * b.vx + dy * b.vy + dz * b.vz;
    T pw = dx * b.wx + dy * b.wy + dz * b.wz;
    T eu = grid_cc_abs<T>(pu) - b.hu; eu = eu > static_cast<T>(0) ? eu : static_cast<T>(0);
    T ev = grid_cc_abs<T>(pv) - b.hv; ev = ev > static_cast<T>(0) ? ev : static_cast<T>(0);
    T ew = grid_cc_abs<T>(pw) - b.hw; ew = ew > static_cast<T>(0) ? ew : static_cast<T>(0);
    return (eu * eu + ev * ev + ew * ew) - r * r;
}

// ------------------------------------------------------------------ environment reduction
// One sphere vs ALL obstacle lists; early-out on first collision.
template <typename T>
__host__ __device__ __forceinline__ bool grid_cc_sphere_in_environment(
        const Environment<T> &env, T x, T y, T z, T r) {
    for (int i = 0; i < env.n_spheres; ++i) {
        const Sphere<T> &s = env.spheres[i];
        if (grid_cc_sphere_sphere<T>(x, y, z, r, s.x, s.y, s.z, s.r) < static_cast<T>(0)) return true;
    }
    for (int i = 0; i < env.n_capsules; ++i)
        if (grid_cc_sphere_capsule<T>(env.capsules[i], x, y, z, r) < static_cast<T>(0)) return true;
    for (int i = 0; i < env.n_cuboids; ++i)
        if (grid_cc_sphere_cuboid<T>(env.cuboids[i], x, y, z, r) < static_cast<T>(0)) return true;
    return false;
}

// ------------------------------------------------------------------ self-collision over baked ranges
// self_cc_ranges is the per-robot generated table (grid_collision namespace, Component D):
// each row {sphere_i, start_j, end_j} => check sphere i against spheres [start_j..end_j].
// s_sphere_pos = the batched extractor output (N x 3 world xyz); s_sphere_r = baked radii.
template <typename T>
__host__ __device__ __forceinline__ bool grid_cc_self_collision(
        const T *s_sphere_pos, const T *s_sphere_r,
        const int *self_cc_ranges, int n_ranges) {
    for (int k = 0; k < n_ranges; ++k) {
        int i  = self_cc_ranges[3 * k + 0];
        int j0 = self_cc_ranges[3 * k + 1];
        int j1 = self_cc_ranges[3 * k + 2];
        T ix = s_sphere_pos[3 * i], iy = s_sphere_pos[3 * i + 1], iz = s_sphere_pos[3 * i + 2], ir = s_sphere_r[i];
        for (int j = j0; j <= j1; ++j) {
            if (grid_cc_sphere_sphere<T>(ix, iy, iz, ir,
                    s_sphere_pos[3 * j], s_sphere_pos[3 * j + 1], s_sphere_pos[3 * j + 2], s_sphere_r[j])
                < static_cast<T>(0)) return true;
        }
    }
    return false;
    // TODO(perf, W3-D): block/warp-parallelize the range loop (thread-per-range, warp any-reduce
    // early-bail) as the reference does; keep single-block. A link_CC mask enables the broad->fine
    // narrowing below.
}

// ================================================================== differentiable path
// SIGNED-DISTANCE + NORMAL variants (the differentiable collision cost, W3 Phase 2). Each returns
// the TRUE signed clearance d = dist - r_sum (NOT the squared gap) and writes the unit surface
// normal n = d(d)/d(sphere_center) = the direction from the obstacle toward the sphere center
// (increasing-clearance direction). Composed with the batched position gradient dp/dq (W2a) this
// gives d(d)/dq = n^T (dp/dq). At the degenerate coincident case (dist -> 0) n falls back to a
// fixed unit vector (the cost there is dominated by penetration; the direction is arbitrary).
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_normalize3(T &vx, T &vy, T &vz) {
    T d = grid_cc_sqrt<T>(vx * vx + vy * vy + vz * vz);
    if (d > static_cast<T>(1e-12)) { T inv = static_cast<T>(1) / d; vx *= inv; vy *= inv; vz *= inv; }
    else { vx = static_cast<T>(1); vy = static_cast<T>(0); vz = static_cast<T>(0); }
    return d;
}

template <typename T>
__host__ __device__ __forceinline__ T grid_cc_sphere_sphere_signed(
        T x, T y, T z, T r, T cx, T cy, T cz, T cr, T *nx, T *ny, T *nz) {
    T vx = x - cx, vy = y - cy, vz = z - cz;
    T dist = grid_cc_normalize3<T>(vx, vy, vz);
    *nx = vx; *ny = vy; *nz = vz;
    return dist - (r + cr);
}

template <typename T>
__host__ __device__ __forceinline__ T grid_cc_sphere_capsule_signed(
        const Capsule<T> &c, T x, T y, T z, T r, T *nx, T *ny, T *nz) {
    T abx = c.bx - c.ax, aby = c.by - c.ay, abz = c.bz - c.az;
    T apx = x - c.ax, apy = y - c.ay, apz = z - c.az;
    T denom = abx * abx + aby * aby + abz * abz;
    T t = denom > static_cast<T>(0) ? (apx * abx + apy * aby + apz * abz) / denom : static_cast<T>(0);
    t = grid_cc_clamp01<T>(t);
    T qx = c.ax + t * abx, qy = c.ay + t * aby, qz = c.az + t * abz;
    T vx = x - qx, vy = y - qy, vz = z - qz;
    T dist = grid_cc_normalize3<T>(vx, vy, vz);
    *nx = vx; *ny = vy; *nz = vz;
    return dist - (r + c.r);
}

template <typename T>
__host__ __device__ __forceinline__ T grid_cc_sphere_cuboid_signed(
        const Cuboid<T> &b, T x, T y, T z, T r, T *nx, T *ny, T *nz) {
    T dx = x - b.cx, dy = y - b.cy, dz = z - b.cz;
    T pu = dx * b.ux + dy * b.uy + dz * b.uz;   // sphere-center offset in the box axis frame
    T pv = dx * b.vx + dy * b.vy + dz * b.vz;
    T pw = dx * b.wx + dy * b.wy + dz * b.wz;
    T eu = grid_cc_abs<T>(pu) - b.hu, ev = grid_cc_abs<T>(pv) - b.hv, ew = grid_cc_abs<T>(pw) - b.hw;
    T su = pu < static_cast<T>(0) ? static_cast<T>(-1) : static_cast<T>(1);
    T sv = pv < static_cast<T>(0) ? static_cast<T>(-1) : static_cast<T>(1);
    T sw = pw < static_cast<T>(0) ? static_cast<T>(-1) : static_cast<T>(1);
    if (eu > static_cast<T>(0) || ev > static_cast<T>(0) || ew > static_cast<T>(0)) {
        // OUTSIDE at least one slab: normal = normalized world excess (clamped per axis).
        T ou = eu > static_cast<T>(0) ? eu : static_cast<T>(0);
        T ov = ev > static_cast<T>(0) ? ev : static_cast<T>(0);
        T ow = ew > static_cast<T>(0) ? ew : static_cast<T>(0);
        T au = su * ou, av = sv * ov, aw = sw * ow;   // signed excess along each axis
        T wx = au * b.ux + av * b.vx + aw * b.wx;     // -> world
        T wy = au * b.uy + av * b.vy + aw * b.wy;
        T wz = au * b.uz + av * b.vz + aw * b.wz;
        T dist = grid_cc_normalize3<T>(wx, wy, wz);
        *nx = wx; *ny = wy; *nz = wz;
        return dist - r;
    }
    // INSIDE the box: penetrating. Normal = the axis of LEAST penetration (nearest face).
    T slu = b.hu - grid_cc_abs<T>(pu), slv = b.hv - grid_cc_abs<T>(pv), slw = b.hw - grid_cc_abs<T>(pw);
    T pen; T ax, ay, az; T sgn;
    if (slu <= slv && slu <= slw) { pen = slu; ax = b.ux; ay = b.uy; az = b.uz; sgn = su; }
    else if (slv <= slw)          { pen = slv; ax = b.vx; ay = b.vy; az = b.vz; sgn = sv; }
    else                          { pen = slw; ax = b.wx; ay = b.wy; az = b.wz; sgn = sw; }
    *nx = sgn * ax; *ny = sgn * ay; *nz = sgn * az;   // box axes are unit -> normal already unit
    return -pen - r;
}

// One point (sphere i) vs the WHOLE environment: nearest (most negative) signed distance + its
// surface normal. Returns a large positive sentinel + a fixed normal when the environment is empty.
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_nearest_obstacle(
        const Environment<T> &env, T x, T y, T z, T r, T *nx, T *ny, T *nz) {
    T best = static_cast<T>(1e30);
    T bnx = static_cast<T>(1), bny = static_cast<T>(0), bnz = static_cast<T>(0);
    T tnx, tny, tnz, d;
    for (int i = 0; i < env.n_spheres; ++i) {
        const Sphere<T> &s = env.spheres[i];
        d = grid_cc_sphere_sphere_signed<T>(x, y, z, r, s.x, s.y, s.z, s.r, &tnx, &tny, &tnz);
        if (d < best) { best = d; bnx = tnx; bny = tny; bnz = tnz; }
    }
    for (int i = 0; i < env.n_capsules; ++i) {
        d = grid_cc_sphere_capsule_signed<T>(env.capsules[i], x, y, z, r, &tnx, &tny, &tnz);
        if (d < best) { best = d; bnx = tnx; bny = tny; bnz = tnz; }
    }
    for (int i = 0; i < env.n_cuboids; ++i) {
        d = grid_cc_sphere_cuboid_signed<T>(env.cuboids[i], x, y, z, r, &tnx, &tny, &tnz);
        if (d < best) { best = d; bnx = tnx; bny = tny; bnz = tnz; }
    }
    *nx = bnx; *ny = bny; *nz = bnz;
    return best;
}

// ------------------------------------------------------------------ broad -> fine driver
// grid_cc_config_free: (1) approx (broad) spheres -> approx env+self check; (2) ONLY if the broad
// pass flags a possible collision, run the fine tier + full checks. GRiD supplies both sphere tiers
// as named batches (broad="approx", fine="all") via the batched extractor; positions come from
// grid::multi_target_position over each tier's descriptor.
template <typename T>
__host__ __device__ bool grid_cc_config_free(
        const Environment<T> &env,
        const T *s_broad_pos, const T *s_broad_r, const int *broad_self_ranges, int n_broad_ranges, int n_broad,
        const T *s_fine_pos,  const T *s_fine_r,  const int *fine_self_ranges,  int n_fine_ranges,  int n_fine) {
    bool broad_hit = grid_cc_self_collision<T>(s_broad_pos, s_broad_r, broad_self_ranges, n_broad_ranges);
    for (int i = 0; !broad_hit && i < n_broad; ++i)
        broad_hit = grid_cc_sphere_in_environment<T>(env, s_broad_pos[3*i], s_broad_pos[3*i+1], s_broad_pos[3*i+2], s_broad_r[i]);
    if (!broad_hit) return true;                       // coarse reject -> definitely free
    if (grid_cc_self_collision<T>(s_fine_pos, s_fine_r, fine_self_ranges, n_fine_ranges)) return false;
    for (int i = 0; i < n_fine; ++i)
        if (grid_cc_sphere_in_environment<T>(env, s_fine_pos[3*i], s_fine_pos[3*i+1], s_fine_pos[3*i+2], s_fine_r[i]))
            return false;
    return true;
    // NOTE: broad_hit is coarse (whole-config). The reference narrows by a per-link link_CC mask so
    // the fine pass only re-checks flagged links; add that once the batched extractor exposes per-link
    // sphere ranges. Differentiable path (GATO/PDDP) reuses these SDFs for d(sdf)/dp = surface normal,
    // composed with the W2a batched gradient -> d(min-dist)/dq. See design_W3.
}

}  // namespace grid_collision

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
#include <limits>
#include <stdexcept>   // the host-side environment builders throw on bad geometry
#include <string>
#include <utility>
#include <vector>

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

// Half-space {p : n.p >= d}, with n a UNIT normal pointing into the FREE side (n.p - d is the
// signed distance of p from the surface, > 0 = clear). The canonical ground plane is
// {0,0,1, z_floor}. Unlike every other primitive its signed distance is EXACTLY LINEAR in p,
// so its gradient is constant and globally smooth — no argmin seam, no degenerate normal.
template <typename T>
struct Plane { T nx, ny, nz, d; };

// Runtime obstacle set (NOT baked — matches the reference Environment<T>). Pointers + counts;
// device upload deep-copies each list then patches these members (Component D upload contract).
// The FLATTENED obstacle index space (used by the per-pair rows below) is, in order:
//   [0, n_spheres) spheres | capsules | cuboids | planes.
// Planes are appended LAST so existing positional brace-init of the first 6 members still
// value-initializes them to {nullptr, 0} (an env with no planes).
template <typename T>
struct Environment {
    const Sphere<T>  *spheres  = nullptr; int n_spheres  = 0;
    const Capsule<T> *capsules = nullptr; int n_capsules = 0;
    const Cuboid<T>  *cuboids  = nullptr; int n_cuboids  = 0;
    const Plane<T>   *planes   = nullptr; int n_planes   = 0;
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

// sphere vs half-space: s = n.p - d is the (exact, signed) center distance from the surface. Take the
// outside-excess max(0,s) exactly as the cuboid does, so a center BELOW the plane (s<0) yields -r^2 < 0
// (collision) rather than a spuriously positive s^2. <0 <=> s < r <=> collision.
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_sphere_plane(const Plane<T> &p, T x, T y, T z, T r) {
    T s = p.nx * x + p.ny * y + p.nz * z - p.d;
    T e = s > static_cast<T>(0) ? s : static_cast<T>(0);
    return e * e - r * r;
}

// ------------------------------------------------------------------ capsule-pair SDFs
// Native-primitive robot geometry (capsule links instead of covering spheres) needs the
// capsule-vs-{capsule, cuboid, plane} pairs; capsule-vs-sphere is grid_cc_sphere_capsule
// with the roles already symmetric. Same squared_gap convention (<0 = collision).

// closest points of two segments (Ericson RTCD 5.1.9 — degenerate/parallel safe): writes the
// clamped parameters s (on A's core a->b) and t (on B's) plus both closest points. Shared by the
// boolean squared-gap check below and the signed/differentiable capsule variants (which need s
// for the envelope-theorem composition d(dist)/dq = n^T [(1-s) da/dq + s db/dq]).
template <typename T>
__host__ __device__ __forceinline__ void grid_cc_seg_seg_closest(
        const Capsule<T> &A, const Capsule<T> &B, T *s_out, T *t_out,
        T *px, T *py, T *pz, T *qx, T *qy, T *qz) {
    T d1x = A.bx - A.ax, d1y = A.by - A.ay, d1z = A.bz - A.az;
    T d2x = B.bx - B.ax, d2y = B.by - B.ay, d2z = B.bz - B.az;
    T rx = A.ax - B.ax,  ry = A.ay - B.ay,  rz = A.az - B.az;
    T a = d1x * d1x + d1y * d1y + d1z * d1z;
    T e = d2x * d2x + d2y * d2y + d2z * d2z;
    T f = d2x * rx + d2y * ry + d2z * rz;
    T s, t;
    if (a == static_cast<T>(0) && e == static_cast<T>(0)) {
        s = static_cast<T>(0); t = static_cast<T>(0);
    } else if (a == static_cast<T>(0)) {
        s = static_cast<T>(0); t = grid_cc_clamp01<T>(f / e);
    } else {
        T c = d1x * rx + d1y * ry + d1z * rz;
        if (e == static_cast<T>(0)) {
            t = static_cast<T>(0); s = grid_cc_clamp01<T>(-c / a);
        } else {
            T b = d1x * d2x + d1y * d2y + d1z * d2z;
            T denom = a * e - b * b;
            s = denom != static_cast<T>(0) ? grid_cc_clamp01<T>((b * f - c * e) / denom)
                                           : static_cast<T>(0);
            t = (b * s + f) / e;
            if (t < static_cast<T>(0)) {
                t = static_cast<T>(0); s = grid_cc_clamp01<T>(-c / a);
            } else if (t > static_cast<T>(1)) {
                t = static_cast<T>(1); s = grid_cc_clamp01<T>((b - c) / a);
            }
        }
    }
    *s_out = s; *t_out = t;
    *px = A.ax + s * d1x; *py = A.ay + s * d1y; *pz = A.az + s * d1z;
    *qx = B.ax + t * d2x; *qy = B.ay + t * d2y; *qz = B.az + t * d2z;
}

// capsule vs capsule: closest squared distance between the two core segments minus the
// summed-radius square (segment math = grid_cc_seg_seg_closest above).
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_capsule_capsule(const Capsule<T> &A, const Capsule<T> &B) {
    T s, t, px, py, pz, qx, qy, qz;
    grid_cc_seg_seg_closest<T>(A, B, &s, &t, &px, &py, &pz, &qx, &qy, &qz);
    T rs = A.r + B.r;
    return grid_cc_sql2_3<T>(px, py, pz, qx, qy, qz) - rs * rs;
}

// capsule vs half-space: the core-segment plane distance is LINEAR in the segment parameter,
// so its minimum sits at an endpoint. Outside-excess like grid_cc_sphere_plane (an endpoint
// below the plane must read as collision, not a spuriously positive s^2).
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_capsule_plane(const Plane<T> &p, const Capsule<T> &c) {
    T sa = p.nx * c.ax + p.ny * c.ay + p.nz * c.az - p.d;
    T sb = p.nx * c.bx + p.ny * c.by + p.nz * c.bz - p.d;
    T s = sa < sb ? sa : sb;
    T e = s > static_cast<T>(0) ? s : static_cast<T>(0);
    return e * e - c.r * c.r;
}

// box-frame squared point-box distance at segment parameter t (helper for capsule_cuboid)
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_seg_box_d2(const T pa[3], const T d[3], const T h[3], T t) {
    T acc = static_cast<T>(0);
    for (int k = 0; k < 3; ++k) {
        T p = pa[k] + t * d[k];
        T ex = grid_cc_abs<T>(p) - h[k];
        if (ex > static_cast<T>(0)) acc += ex * ex;
    }
    return acc;
}

// capsule vs oriented cuboid. In the box frame the squared core-segment/box distance
//   D2(t) = sum_k max(0, |p_k(t)| - h_k)^2,   p(t) = pa + t (pb - pa),  t in [0,1]
// is CONVEX piecewise-quadratic: pieces split where a coordinate crosses +-h_k (<= 6
// interior breakpoints). Exact minimum by finite enumeration: evaluate D2 at every
// breakpoint/endpoint and, per interval, at its midpoint and at the clamped stationary
// point of the active-set quadratic (active set read off at the midpoint). Fixed loop
// bounds, no iteration-to-convergence -> deterministic and host/device identical.
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_capsule_cuboid(const Cuboid<T> &b, const Capsule<T> &c) {
    // endpoints into the box frame
    T ax0 = c.ax - b.cx, ay0 = c.ay - b.cy, az0 = c.az - b.cz;
    T bx0 = c.bx - b.cx, by0 = c.by - b.cy, bz0 = c.bz - b.cz;
    T pa[3], pb[3], h[3];
    pa[0] = ax0 * b.ux + ay0 * b.uy + az0 * b.uz;  pb[0] = bx0 * b.ux + by0 * b.uy + bz0 * b.uz;  h[0] = b.hu;
    pa[1] = ax0 * b.vx + ay0 * b.vy + az0 * b.vz;  pb[1] = bx0 * b.vx + by0 * b.vy + bz0 * b.vz;  h[1] = b.hv;
    pa[2] = ax0 * b.wx + ay0 * b.wy + az0 * b.wz;  pb[2] = bx0 * b.wx + by0 * b.wy + bz0 * b.wz;  h[2] = b.hw;
    T d[3] = { pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2] };

    // candidate ts: endpoints + per-axis +-h crossings (clamped set, <= 8)
    T ts[8]; int n = 0;
    ts[n++] = static_cast<T>(0);
    ts[n++] = static_cast<T>(1);
    for (int k = 0; k < 3; ++k) {
        if (d[k] != static_cast<T>(0)) {
            T t1 = (h[k] - pa[k]) / d[k];
            T t2 = (-h[k] - pa[k]) / d[k];
            if (t1 > static_cast<T>(0) && t1 < static_cast<T>(1)) ts[n++] = t1;
            if (t2 > static_cast<T>(0) && t2 < static_cast<T>(1)) ts[n++] = t2;
        }
    }
    // insertion sort (n <= 8; deterministic)
    for (int i = 1; i < n; ++i) {
        T key = ts[i]; int j = i - 1;
        while (j >= 0 && ts[j] > key) { ts[j + 1] = ts[j]; --j; }
        ts[j + 1] = key;
    }
    T best = grid_cc_seg_box_d2<T>(pa, d, h, ts[0]);
    for (int i = 1; i < n; ++i) {
        T v = grid_cc_seg_box_d2<T>(pa, d, h, ts[i]);
        if (v < best) best = v;
    }
    for (int i = 0; i + 1 < n; ++i) {
        T lo = ts[i], hi = ts[i + 1];
        if (!(hi > lo)) continue;
        T tm = (lo + hi) * static_cast<T>(0.5);
        T vm = grid_cc_seg_box_d2<T>(pa, d, h, tm); if (vm < best) best = vm;
        // active-set quadratic sum_k (ck + ek t)^2 on this interval; stationary point
        T sce = static_cast<T>(0), see = static_cast<T>(0);
        for (int k = 0; k < 3; ++k) {
            T p = pa[k] + tm * d[k];
            if (grid_cc_abs<T>(p) > h[k]) {
                T sg = p > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(-1);
                T ck = sg * pa[k] - h[k], ek = sg * d[k];
                sce += ck * ek; see += ek * ek;
            }
        }
        if (see > static_cast<T>(0)) {
            T tstar = -sce / see;
            tstar = tstar < lo ? lo : (tstar > hi ? hi : tstar);
            T v = grid_cc_seg_box_d2<T>(pa, d, h, tstar); if (v < best) best = v;
        }
    }
    return best - c.r * c.r;
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
    for (int i = 0; i < env.n_planes; ++i)
        if (grid_cc_sphere_plane<T>(env.planes[i], x, y, z, r) < static_cast<T>(0)) return true;
    return false;
}

// ------------------------------------------------------------------ self-collision over baked ranges
// self_cc_ranges is the per-robot generated table (grid_collision namespace, Component D):
// each row {sphere_i, start_j, end_j} => check sphere i against spheres [start_j..end_j].
// s_sphere_pos = the batched extractor output (N x 3 world xyz); s_sphere_r = baked radii.
// margin (default 0): a pair is a hit iff its clearance is below the margin; the margin is added to
// sphere i's radius only, so every pair sees it once (G7, 2026-10-23).
template <typename T>
__host__ __device__ __forceinline__ bool grid_cc_self_collision(
        const T *s_sphere_pos, const T *s_sphere_r,
        const int *self_cc_ranges, int n_ranges, T margin = static_cast<T>(0)) {
    for (int k = 0; k < n_ranges; ++k) {
        int i  = self_cc_ranges[3 * k + 0];
        int j0 = self_cc_ranges[3 * k + 1];
        int j1 = self_cc_ranges[3 * k + 2];
        T ix = s_sphere_pos[3 * i], iy = s_sphere_pos[3 * i + 1], iz = s_sphere_pos[3 * i + 2], ir = s_sphere_r[i] + margin;
        for (int j = j0; j <= j1; ++j) {
            if (grid_cc_sphere_sphere<T>(ix, iy, iz, ir,
                    s_sphere_pos[3 * j], s_sphere_pos[3 * j + 1], s_sphere_pos[3 * j + 2], s_sphere_r[j])
                < static_cast<T>(0)) return true;
        }
    }
    return false;
    // TODO(perf, W3-D): block/warp-parallelize the range loop (thread-per-range, warp any-reduce
    // early-bail) as the reference does; keep single-block. (The broad->fine link_CC narrowing is
    // now implemented in grid_cc_config_free below via a per-link uint64 hit-mask.)
}

// ------------------------------------------------------------------ CAPSULE robot rows
// Native-primitive robot geometry: each robot row is a CAPSULE {a, b, r} whose endpoints ride the
// batched extractor as TWO consecutive targets, so s_seg_pos[6i..6i+2] = a_i and [6i+3..6i+5] = b_i
// in world. a == b degenerates to a sphere (seg-seg closest-point math is point-safe), so one row
// type serves spherized and native links alike. Same squared_gap convention throughout.
// inflate (default 0) is added to the row radius: the margin-aware checkers test rows at r + margin.
template <typename T>
__host__ __device__ __forceinline__ Capsule<T> grid_cc_row_capsule(
        const T *s_seg_pos, const T *s_row_r, int i, T inflate = static_cast<T>(0)) {
    return Capsule<T>{ s_seg_pos[6*i],     s_seg_pos[6*i + 1], s_seg_pos[6*i + 2],
                       s_seg_pos[6*i + 3], s_seg_pos[6*i + 4], s_seg_pos[6*i + 5], s_row_r[i] + inflate };
}

// One robot capsule row vs ALL obstacle lists; early-out on first collision.
// (grid_cc_sphere_capsule's roles are symmetric: an env SPHERE vs the robot capsule reuses it.)
template <typename T>
__host__ __device__ __forceinline__ bool grid_cc_capsule_in_environment(
        const Environment<T> &env, const Capsule<T> &c) {
    for (int i = 0; i < env.n_spheres; ++i) {
        const Sphere<T> &s = env.spheres[i];
        if (grid_cc_sphere_capsule<T>(c, s.x, s.y, s.z, s.r) < static_cast<T>(0)) return true;
    }
    for (int i = 0; i < env.n_capsules; ++i)
        if (grid_cc_capsule_capsule<T>(c, env.capsules[i]) < static_cast<T>(0)) return true;
    for (int i = 0; i < env.n_cuboids; ++i)
        if (grid_cc_capsule_cuboid<T>(env.cuboids[i], c) < static_cast<T>(0)) return true;
    for (int i = 0; i < env.n_planes; ++i)
        if (grid_cc_capsule_plane<T>(env.planes[i], c) < static_cast<T>(0)) return true;
    return false;
}

// Self-collision over baked ranges, capsule rows. IDENTICAL range table semantics to the sphere
// form ({row_i, start_j, end_j}, adjacency pre-excluded at bake time) — only the pair SDF changes.
template <typename T>
__host__ __device__ __forceinline__ bool grid_cc_self_collision_capsules(
        const T *s_seg_pos, const T *s_row_r,
        const int *self_cc_ranges, int n_ranges, T margin = static_cast<T>(0)) {
    for (int k = 0; k < n_ranges; ++k) {
        int i  = self_cc_ranges[3 * k + 0];
        int j0 = self_cc_ranges[3 * k + 1];
        int j1 = self_cc_ranges[3 * k + 2];
        Capsule<T> ci = grid_cc_row_capsule<T>(s_seg_pos, s_row_r, i, margin);
        for (int j = j0; j <= j1; ++j)
            if (grid_cc_capsule_capsule<T>(ci, grid_cc_row_capsule<T>(s_seg_pos, s_row_r, j))
                < static_cast<T>(0)) return true;
    }
    return false;
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

// sphere vs half-space, signed. The clearance is EXACT and linear (dist - r, no clamping) and the
// normal is the plane's own — constant, always unit, no degenerate case. This is the primitive GATO's
// ground-contact rows want: g(q) and dg/dq are smooth everywhere, unlike the argmin over a set.
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_sphere_plane_signed(
        const Plane<T> &p, T x, T y, T z, T r, T *nx, T *ny, T *nz) {
    *nx = p.nx; *ny = p.ny; *nz = p.nz;
    return (p.nx * x + p.ny * y + p.nz * z - p.d) - r;
}

// ------------------------------------------------------------------ flattened obstacle index space
// Obstacles are addressed by a single index o in [0, grid_cc_num_obstacles(env)), laid out
// spheres | capsules | cuboids | planes. This is the column index of the PER-PAIR rows
// (grid_collision::collision_distance_pairs) and the iteration order of the argmin below.
template <typename T>
__host__ __device__ __forceinline__ int grid_cc_num_obstacles(const Environment<T> &env) {
    return env.n_spheres + env.n_capsules + env.n_cuboids + env.n_planes;
}

// Sphere vs the o-th obstacle: signed clearance + unit surface normal (increasing-clearance direction).
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_obstacle_signed(
        const Environment<T> &env, int o, T x, T y, T z, T r, T *nx, T *ny, T *nz) {
    if (o < env.n_spheres) {
        const Sphere<T> &s = env.spheres[o];
        return grid_cc_sphere_sphere_signed<T>(x, y, z, r, s.x, s.y, s.z, s.r, nx, ny, nz);
    }
    o -= env.n_spheres;
    if (o < env.n_capsules) return grid_cc_sphere_capsule_signed<T>(env.capsules[o], x, y, z, r, nx, ny, nz);
    o -= env.n_capsules;
    if (o < env.n_cuboids)  return grid_cc_sphere_cuboid_signed<T>(env.cuboids[o], x, y, z, r, nx, ny, nz);
    o -= env.n_cuboids;
    return grid_cc_sphere_plane_signed<T>(env.planes[o], x, y, z, r, nx, ny, nz);
}

// One point (sphere i) vs the WHOLE environment: nearest (most negative) signed distance + its
// surface normal. Returns a large positive sentinel + a fixed normal when the environment is empty.
// The argmin ties break toward the LOWEST flattened obstacle index (strict <), and it is exactly this
// switch of the winning obstacle that makes the reduced distance non-smooth in q — the reason the
// per-pair rows exist alongside it.
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_nearest_obstacle(
        const Environment<T> &env, T x, T y, T z, T r, T *nx, T *ny, T *nz) {
    T best = static_cast<T>(1e30);
    T bnx = static_cast<T>(1), bny = static_cast<T>(0), bnz = static_cast<T>(0);
    T tnx, tny, tnz;
    const int n_obs = grid_cc_num_obstacles<T>(env);
    for (int o = 0; o < n_obs; ++o) {
        T d = grid_cc_obstacle_signed<T>(env, o, x, y, z, r, &tnx, &tny, &tnz);
        if (d < best) { best = d; bnx = tnx; bny = tny; bnz = tnz; }
    }
    *nx = bnx; *ny = bny; *nz = bnz;
    return best;
}

// ------------------------------------------------------------------ broad -> fine driver (link_CC mask)
// grid_cc_config_free: (1) run the broad (approx) tier in FULL, OR-ing every hit link into a uint64
// hit-mask keyed by anchor (frame) id; (2) if no link is flagged, the config is definitely free;
// (3) otherwise run the fine tier but re-check ONLY spheres whose link the broad pass flagged.
// The covering-sphere property (a broad sphere on link L encloses the fine spheres on L) guarantees a
// real fine collision on link L is always broad-flagged, so the mask NEVER drops a true hit -> the
// verdict stays bit-identical to a fine-only check, while the (larger) fine tier skips unflagged links.
// broad_sphere_link[i] / fine_sphere_link[i] = the sphere's anchor id (bit index); NUM_JOINTS<=64 is
// static_asserted at the baked-table site so every id fits a uint64. dbg_fine_rechecked (optional, a
// caller thread-LOCAL to stay race-free) receives the count of fine spheres surviving the mask.
template <typename T>
__host__ __device__ bool grid_cc_config_free(
        const Environment<T> &env,
        const T *s_broad_pos, const T *s_broad_r, const int *broad_self_ranges, int n_broad_ranges, int n_broad,
        const int *broad_sphere_link,
        const T *s_fine_pos,  const T *s_fine_r,  const int *fine_self_ranges,  int n_fine_ranges,  int n_fine,
        const int *fine_sphere_link, int *dbg_fine_rechecked = nullptr,
        T margin = static_cast<T>(0), bool check_self = true) {
    // margin: a hit iff a raw clearance is below it (robot spheres tested at r + margin on both tiers,
    // which keeps the covering argument); check_self = false skips both self-range scans (G7/G10).
    unsigned long long hit_mask = 0ull;
    // Broad self-collision: full scan (NO early-out — the mask needs every hit), flag BOTH endpoints'
    // links of each hit range (a self-pair's two spheres may sit on different links).
    if (check_self) for (int k = 0; k < n_broad_ranges; ++k) {
        int i = broad_self_ranges[3*k], j0 = broad_self_ranges[3*k+1], j1 = broad_self_ranges[3*k+2];
        T ix = s_broad_pos[3*i], iy = s_broad_pos[3*i+1], iz = s_broad_pos[3*i+2], ir = s_broad_r[i] + margin;
        for (int j = j0; j <= j1; ++j)
            if (grid_cc_sphere_sphere<T>(ix, iy, iz, ir,
                    s_broad_pos[3*j], s_broad_pos[3*j+1], s_broad_pos[3*j+2], s_broad_r[j]) < static_cast<T>(0)) {
                hit_mask |= (1ull << broad_sphere_link[i]);
                hit_mask |= (1ull << broad_sphere_link[j]);
            }
    }
    // Broad vs environment: full scan, flag each hit sphere's link.
    for (int i = 0; i < n_broad; ++i)
        if (grid_cc_sphere_in_environment<T>(env, s_broad_pos[3*i], s_broad_pos[3*i+1], s_broad_pos[3*i+2], s_broad_r[i] + margin))
            hit_mask |= (1ull << broad_sphere_link[i]);

    // Narrowing measure (non-vacuous-gate hook): fine spheres surviving the mask. Written before the
    // fine checks so it is set on every non-trivially-free path regardless of an early collision exit.
    if (dbg_fine_rechecked != nullptr) {
        int survive = 0;
        for (int i = 0; i < n_fine; ++i)
            if ((hit_mask >> fine_sphere_link[i]) & 1ull) ++survive;
        *dbg_fine_rechecked = survive;
    }
    if (hit_mask == 0ull) return true;                 // no link flagged -> definitely free

    // Fine pass, NARROWED. self-range {i, j0..j1}: skip the whole range if link(i) is unflagged — by
    // the covering property no j can truly collide with i then (a real i-j hit would have broad-flagged
    // link(i)). If link(i) IS flagged we still scan all its j (conservative, and cheap).
    if (check_self) for (int k = 0; k < n_fine_ranges; ++k) {
        int i = fine_self_ranges[3*k], j0 = fine_self_ranges[3*k+1], j1 = fine_self_ranges[3*k+2];
        if (((hit_mask >> fine_sphere_link[i]) & 1ull) == 0ull) continue;
        T ix = s_fine_pos[3*i], iy = s_fine_pos[3*i+1], iz = s_fine_pos[3*i+2], ir = s_fine_r[i] + margin;
        for (int j = j0; j <= j1; ++j)
            if (grid_cc_sphere_sphere<T>(ix, iy, iz, ir,
                    s_fine_pos[3*j], s_fine_pos[3*j+1], s_fine_pos[3*j+2], s_fine_r[j]) < static_cast<T>(0)) return false;
    }
    // Fine vs environment: skip spheres on unflagged links.
    for (int i = 0; i < n_fine; ++i) {
        if (((hit_mask >> fine_sphere_link[i]) & 1ull) == 0ull) continue;
        if (grid_cc_sphere_in_environment<T>(env, s_fine_pos[3*i], s_fine_pos[3*i+1], s_fine_pos[3*i+2], s_fine_r[i] + margin))
            return false;
    }
    return true;
    // Differentiable path (GATO/PDDP) reuses these SDFs for d(sdf)/dp = surface normal, composed with
    // the W2a batched gradient -> d(min-dist)/dq. See design_W3. Future >64-frame robots: swap the
    // uint64 hit_mask for a bool[NUM_JOINTS] (the static_assert at the baked table site fires first).
}

// Broad -> fine driver, CAPSULE fine tier. The broad tier stays covering SPHERES (one per link,
// derived at bake time to enclose every fine capsule on that link — sphere math keeps the broad
// full-scan cheap and the covering argument identical), the fine tier is native capsule rows
// (s_fine_seg = 6 floats/row). Mask semantics identical to grid_cc_config_free above.
template <typename T>
__host__ __device__ bool grid_cc_config_free_capsule(
        const Environment<T> &env,
        const T *s_broad_pos, const T *s_broad_r, const int *broad_self_ranges, int n_broad_ranges, int n_broad,
        const int *broad_sphere_link,
        const T *s_fine_seg,  const T *s_fine_r,  const int *fine_self_ranges,  int n_fine_ranges,  int n_fine,
        const int *fine_row_link, int *dbg_fine_rechecked = nullptr,
        T margin = static_cast<T>(0), bool check_self = true) {
    unsigned long long hit_mask = 0ull;
    if (check_self) for (int k = 0; k < n_broad_ranges; ++k) {
        int i = broad_self_ranges[3*k], j0 = broad_self_ranges[3*k+1], j1 = broad_self_ranges[3*k+2];
        T ix = s_broad_pos[3*i], iy = s_broad_pos[3*i+1], iz = s_broad_pos[3*i+2], ir = s_broad_r[i] + margin;
        for (int j = j0; j <= j1; ++j)
            if (grid_cc_sphere_sphere<T>(ix, iy, iz, ir,
                    s_broad_pos[3*j], s_broad_pos[3*j+1], s_broad_pos[3*j+2], s_broad_r[j]) < static_cast<T>(0)) {
                hit_mask |= (1ull << broad_sphere_link[i]);
                hit_mask |= (1ull << broad_sphere_link[j]);
            }
    }
    for (int i = 0; i < n_broad; ++i)
        if (grid_cc_sphere_in_environment<T>(env, s_broad_pos[3*i], s_broad_pos[3*i+1], s_broad_pos[3*i+2], s_broad_r[i] + margin))
            hit_mask |= (1ull << broad_sphere_link[i]);

    if (dbg_fine_rechecked != nullptr) {
        int survive = 0;
        for (int i = 0; i < n_fine; ++i)
            if ((hit_mask >> fine_row_link[i]) & 1ull) ++survive;
        *dbg_fine_rechecked = survive;
    }
    if (hit_mask == 0ull) return true;

    if (check_self) for (int k = 0; k < n_fine_ranges; ++k) {
        int i = fine_self_ranges[3*k], j0 = fine_self_ranges[3*k+1], j1 = fine_self_ranges[3*k+2];
        if (((hit_mask >> fine_row_link[i]) & 1ull) == 0ull) continue;
        Capsule<T> ci = grid_cc_row_capsule<T>(s_fine_seg, s_fine_r, i, margin);
        for (int j = j0; j <= j1; ++j)
            if (grid_cc_capsule_capsule<T>(ci, grid_cc_row_capsule<T>(s_fine_seg, s_fine_r, j))
                < static_cast<T>(0)) return false;
    }
    for (int i = 0; i < n_fine; ++i) {
        if (((hit_mask >> fine_row_link[i]) & 1ull) == 0ull) continue;
        if (grid_cc_capsule_in_environment<T>(env, grid_cc_row_capsule<T>(s_fine_seg, s_fine_r, i, margin)))
            return false;
    }
    return true;
}

// ================================================================== differentiable path, CAPSULE rows
// Signed clearance + unit surface normal + the ROBOT-side core-segment parameter t* of the closest
// point, per obstacle type. t* is what makes the row differentiable through FK: by the envelope
// theorem (t* is a minimizer over the segment) the clearance derivative is
//   d(d)/dq = n^T [ (1-t*) da/dq + t* db/dq ]
// with da/dq, db/dq the batched endpoint position gradients (the 2N-target multi_target batch).
// n points from the obstacle toward the robot capsule's closest core point (increasing clearance).
// Conventions at non-smooth spots (documented, deterministic): penetrating-cuboid depth is reported
// at the enumerated core minimizer t*; a segment parallel to a plane reports t* = 0.

template <typename T>
__host__ __device__ __forceinline__ T grid_cc_capsule_sphere_signed(
        const Capsule<T> &c, T sx, T sy, T sz, T sr, T *nx, T *ny, T *nz, T *t_out) {
    T abx = c.bx - c.ax, aby = c.by - c.ay, abz = c.bz - c.az;
    T apx = sx - c.ax, apy = sy - c.ay, apz = sz - c.az;
    T denom = abx * abx + aby * aby + abz * abz;
    T t = denom > static_cast<T>(0) ? (apx * abx + apy * aby + apz * abz) / denom : static_cast<T>(0);
    t = grid_cc_clamp01<T>(t);
    T px = c.ax + t * abx, py = c.ay + t * aby, pz = c.az + t * abz;
    T vx = px - sx, vy = py - sy, vz = pz - sz;              // obstacle -> robot core point
    T dist = grid_cc_normalize3<T>(vx, vy, vz);
    *nx = vx; *ny = vy; *nz = vz; *t_out = t;
    return dist - (c.r + sr);
}

template <typename T>
__host__ __device__ __forceinline__ T grid_cc_capsule_capsule_signed(
        const Capsule<T> &robot, const Capsule<T> &obs, T *nx, T *ny, T *nz, T *t_out) {
    T s, t, px, py, pz, qx, qy, qz;
    grid_cc_seg_seg_closest<T>(robot, obs, &s, &t, &px, &py, &pz, &qx, &qy, &qz);
    T vx = px - qx, vy = py - qy, vz = pz - qz;              // obstacle core -> robot core
    T dist = grid_cc_normalize3<T>(vx, vy, vz);
    *nx = vx; *ny = vy; *nz = vz; *t_out = s;                // s = ROBOT-side parameter
    return dist - (robot.r + obs.r);
}

// robot capsule vs cuboid: find the core-segment minimizer t* by the same exact piecewise-quadratic
// enumeration as grid_cc_capsule_cuboid, then delegate to the sphere-cuboid signed form at p(t*)
// (exact outside the box; penetrating case reports nearest-face depth at t*, see note above).
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_capsule_cuboid_signed(
        const Cuboid<T> &b, const Capsule<T> &c, T *nx, T *ny, T *nz, T *t_out) {
    T ax0 = c.ax - b.cx, ay0 = c.ay - b.cy, az0 = c.az - b.cz;
    T bx0 = c.bx - b.cx, by0 = c.by - b.cy, bz0 = c.bz - b.cz;
    T pa[3], pb[3], h[3];
    pa[0] = ax0 * b.ux + ay0 * b.uy + az0 * b.uz;  pb[0] = bx0 * b.ux + by0 * b.uy + bz0 * b.uz;  h[0] = b.hu;
    pa[1] = ax0 * b.vx + ay0 * b.vy + az0 * b.vz;  pb[1] = bx0 * b.vx + by0 * b.vy + bz0 * b.vz;  h[1] = b.hv;
    pa[2] = ax0 * b.wx + ay0 * b.wy + az0 * b.wz;  pb[2] = bx0 * b.wx + by0 * b.wy + bz0 * b.wz;  h[2] = b.hw;
    T d[3] = { pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2] };
    T ts[8]; int n = 0;
    ts[n++] = static_cast<T>(0);
    ts[n++] = static_cast<T>(1);
    for (int k = 0; k < 3; ++k) {
        if (d[k] != static_cast<T>(0)) {
            T t1 = (h[k] - pa[k]) / d[k];
            T t2 = (-h[k] - pa[k]) / d[k];
            if (t1 > static_cast<T>(0) && t1 < static_cast<T>(1)) ts[n++] = t1;
            if (t2 > static_cast<T>(0) && t2 < static_cast<T>(1)) ts[n++] = t2;
        }
    }
    for (int i = 1; i < n; ++i) {
        T key = ts[i]; int j = i - 1;
        while (j >= 0 && ts[j] > key) { ts[j + 1] = ts[j]; --j; }
        ts[j + 1] = key;
    }
    T tbest = ts[0], best = grid_cc_seg_box_d2<T>(pa, d, h, ts[0]);
    for (int i = 1; i < n; ++i) {
        T v = grid_cc_seg_box_d2<T>(pa, d, h, ts[i]);
        if (v < best) { best = v; tbest = ts[i]; }
    }
    for (int i = 0; i + 1 < n; ++i) {
        T lo = ts[i], hi = ts[i + 1];
        if (!(hi > lo)) continue;
        T tm = (lo + hi) * static_cast<T>(0.5);
        T vm = grid_cc_seg_box_d2<T>(pa, d, h, tm);
        if (vm < best) { best = vm; tbest = tm; }
        T sce = static_cast<T>(0), see = static_cast<T>(0);
        for (int k = 0; k < 3; ++k) {
            T p = pa[k] + tm * d[k];
            if (grid_cc_abs<T>(p) > h[k]) {
                T sg = p > static_cast<T>(0) ? static_cast<T>(1) : static_cast<T>(-1);
                T ck = sg * pa[k] - h[k], ek = sg * d[k];
                sce += ck * ek; see += ek * ek;
            }
        }
        if (see > static_cast<T>(0)) {
            T tstar = -sce / see;
            tstar = tstar < lo ? lo : (tstar > hi ? hi : tstar);
            T v = grid_cc_seg_box_d2<T>(pa, d, h, tstar);
            if (v < best) { best = v; tbest = tstar; }
        }
    }
    *t_out = tbest;
    T px = c.ax + tbest * (c.bx - c.ax);
    T py = c.ay + tbest * (c.by - c.ay);
    T pz = c.az + tbest * (c.bz - c.az);
    return grid_cc_sphere_cuboid_signed<T>(b, px, py, pz, c.r, nx, ny, nz);
}

template <typename T>
__host__ __device__ __forceinline__ T grid_cc_capsule_plane_signed(
        const Plane<T> &p, const Capsule<T> &c, T *nx, T *ny, T *nz, T *t_out) {
    T sa = p.nx * c.ax + p.ny * c.ay + p.nz * c.az - p.d;
    T sb = p.nx * c.bx + p.ny * c.by + p.nz * c.bz - p.d;
    *nx = p.nx; *ny = p.ny; *nz = p.nz;
    *t_out = sa <= sb ? static_cast<T>(0) : static_cast<T>(1);   // min endpoint; parallel -> t=0
    return (sa <= sb ? sa : sb) - c.r;
}

// Robot capsule row vs the o-th flattened obstacle (spheres | capsules | cuboids | planes).
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_capsule_obstacle_signed(
        const Environment<T> &env, int o, const Capsule<T> &c, T *nx, T *ny, T *nz, T *t_out) {
    if (o < env.n_spheres) {
        const Sphere<T> &s = env.spheres[o];
        return grid_cc_capsule_sphere_signed<T>(c, s.x, s.y, s.z, s.r, nx, ny, nz, t_out);
    }
    o -= env.n_spheres;
    if (o < env.n_capsules) return grid_cc_capsule_capsule_signed<T>(c, env.capsules[o], nx, ny, nz, t_out);
    o -= env.n_capsules;
    if (o < env.n_cuboids)  return grid_cc_capsule_cuboid_signed<T>(env.cuboids[o], c, nx, ny, nz, t_out);
    o -= env.n_cuboids;
    return grid_cc_capsule_plane_signed<T>(env.planes[o], c, nx, ny, nz, t_out);
}

// One capsule row vs the WHOLE environment: nearest signed distance + its normal + robot-side t*.
// Empty environment -> large positive sentinel, fixed normal, t* = 0. Ties break to the lowest index.
template <typename T>
__host__ __device__ __forceinline__ T grid_cc_nearest_obstacle_capsule(
        const Environment<T> &env, const Capsule<T> &c, T *nx, T *ny, T *nz, T *t_out) {
    T best = static_cast<T>(1e30);
    T bnx = static_cast<T>(1), bny = static_cast<T>(0), bnz = static_cast<T>(0), bt = static_cast<T>(0);
    T tnx, tny, tnz, tt;
    const int n_obs = grid_cc_num_obstacles<T>(env);
    for (int o = 0; o < n_obs; ++o) {
        T d = grid_cc_capsule_obstacle_signed<T>(env, o, c, &tnx, &tny, &tnz, &tt);
        if (d < best) { best = d; bnx = tnx; bny = tny; bnz = tnz; bt = tt; }
    }
    *nx = bnx; *ny = bny; *nz = bnz; *t_out = bt;
    return best;
}


// ================================================================== environment ownership (G8a, 2026-10-23)
// Host-side builders from plain arrays and an owning device environment, so a consumer does not
// carry its own upload code (HJCD-IK and GATO each had one). Row layouts, one row per obstacle:
//   spheres  (M, 4):  x y z r
//   capsules (M, 7):  ax ay az bx by bz r          (segment endpoints and radius)
//   cuboids  (M, 10): cx cy cz qw qx qy qz dx dy dz (centre, unit quaternion w x y z, FULL dims)
//   planes   (M, 4):  nx ny nz d                    (half-space n.p >= d; n is normalized here)
// make_host_environment validates every row (finite numbers, positive radii and dimensions, a
// non-zero quaternion and normal) and throws std::invalid_argument naming the row; the host
// vectors feed the host primitives through HostEnvironment::view() (host pointers) and the
// device through DeviceEnvironment (deep copy, device recorded, freed on destruction, movable,
// not copyable). DeviceEnvironment::view() is the Environment<T> kernels take by value.

// Oriented box from a centre, a unit quaternion (w, x, y, z; normalized here) and FULL dimensions:
// the box axes are the columns of the rotation matrix, the half-extents half the dimensions.
template <typename T>
__host__ __device__ inline Cuboid<T> grid_cc_cuboid_from_quaternion(
        T cx, T cy, T cz, T qw, T qx, T qy, T qz, T dx, T dy, T dz) {
    const T n = grid_cc_sqrt<T>(qw * qw + qx * qx + qy * qy + qz * qz);
    const T inv = static_cast<T>(1) / n;
    qw *= inv; qx *= inv; qy *= inv; qz *= inv;
    const T two = static_cast<T>(2), one = static_cast<T>(1), half = static_cast<T>(0.5);
    Cuboid<T> b;
    b.cx = cx; b.cy = cy; b.cz = cz;
    b.ux = one - two * (qy * qy + qz * qz); b.uy = two * (qx * qy + qz * qw);       b.uz = two * (qx * qz - qy * qw);       b.hu = half * dx;
    b.vx = two * (qx * qy - qz * qw);       b.vy = one - two * (qx * qx + qz * qz); b.vz = two * (qy * qz + qx * qw);       b.hv = half * dy;
    b.wx = two * (qx * qz + qy * qw);       b.wy = two * (qy * qz - qx * qw);       b.wz = one - two * (qx * qx + qy * qy); b.hw = half * dz;
    return b;
}

template <typename T>
struct HostEnvironment {
    std::vector<Sphere<T>>  spheres;
    std::vector<Capsule<T>> capsules;
    std::vector<Cuboid<T>>  cuboids;
    std::vector<Plane<T>>   planes;
    int size() const { return int(spheres.size() + capsules.size() + cuboids.size() + planes.size()); }
    bool empty() const { return size() == 0; }
    // Host-pointer view for the host instantiations of the primitives (tests, CPU references).
    Environment<T> view() const {
        return Environment<T>{ spheres.data(),  int(spheres.size()),
                               capsules.data(), int(capsules.size()),
                               cuboids.data(),  int(cuboids.size()),
                               planes.data(),   int(planes.size()) };
    }
};

namespace detail {
template <typename T>
inline T grid_cc_checked_finite(T v, const char *what, int row) {
    if (!(v == v) || v > std::numeric_limits<T>::max() || v < -std::numeric_limits<T>::max())
        throw std::invalid_argument(std::string("grid_collision: ") + what + " row " + std::to_string(row) + " is not finite");
    return v;
}
template <typename T>
inline T grid_cc_checked_positive(T v, const char *what, int row) {
    if (!(grid_cc_checked_finite<T>(v, what, row) > static_cast<T>(0)))
        throw std::invalid_argument(std::string("grid_collision: ") + what + " row " + std::to_string(row) + " needs a positive radius or dimension");
    return v;
}
inline void grid_cc_check_rows(const void *p, int n, const char *what) {
    if (n < 0 || (n > 0 && p == nullptr))
        throw std::invalid_argument(std::string("grid_collision: ") + what + " count " + std::to_string(n) + " with no data");
}
}  // namespace detail

template <typename T>
inline HostEnvironment<T> make_host_environment(const T *spheres, int n_spheres,
                                                const T *capsules, int n_capsules,
                                                const T *cuboids, int n_cuboids,
                                                const T *planes = nullptr, int n_planes = 0) {
    using namespace detail;
    grid_cc_check_rows(spheres, n_spheres, "spheres");
    grid_cc_check_rows(capsules, n_capsules, "capsules");
    grid_cc_check_rows(cuboids, n_cuboids, "cuboids");
    grid_cc_check_rows(planes, n_planes, "planes");
    HostEnvironment<T> env;
    env.spheres.reserve(n_spheres); env.capsules.reserve(n_capsules);
    env.cuboids.reserve(n_cuboids); env.planes.reserve(n_planes);
    for (int i = 0; i < n_spheres; ++i) {
        const T *s = spheres + 4 * i;
        env.spheres.push_back(Sphere<T>{ grid_cc_checked_finite<T>(s[0], "spheres", i), grid_cc_checked_finite<T>(s[1], "spheres", i),
                                         grid_cc_checked_finite<T>(s[2], "spheres", i), grid_cc_checked_positive<T>(s[3], "spheres", i) });
    }
    for (int i = 0; i < n_capsules; ++i) {
        const T *c = capsules + 7 * i;
        env.capsules.push_back(Capsule<T>{ grid_cc_checked_finite<T>(c[0], "capsules", i), grid_cc_checked_finite<T>(c[1], "capsules", i),
                                           grid_cc_checked_finite<T>(c[2], "capsules", i), grid_cc_checked_finite<T>(c[3], "capsules", i),
                                           grid_cc_checked_finite<T>(c[4], "capsules", i), grid_cc_checked_finite<T>(c[5], "capsules", i),
                                           grid_cc_checked_positive<T>(c[6], "capsules", i) });
    }
    for (int i = 0; i < n_cuboids; ++i) {
        const T *b = cuboids + 10 * i;
        for (int k = 0; k < 7; ++k) grid_cc_checked_finite<T>(b[k], "cuboids", i);
        if (!(b[3] * b[3] + b[4] * b[4] + b[5] * b[5] + b[6] * b[6] > static_cast<T>(0)))
            throw std::invalid_argument("grid_collision: cuboids row " + std::to_string(i) + " has a zero quaternion");
        env.cuboids.push_back(grid_cc_cuboid_from_quaternion<T>(
            b[0], b[1], b[2], b[3], b[4], b[5], b[6],
            grid_cc_checked_positive<T>(b[7], "cuboids", i), grid_cc_checked_positive<T>(b[8], "cuboids", i),
            grid_cc_checked_positive<T>(b[9], "cuboids", i)));
    }
    for (int i = 0; i < n_planes; ++i) {
        const T *p = planes + 4 * i;
        for (int k = 0; k < 4; ++k) grid_cc_checked_finite<T>(p[k], "planes", i);
        const T n = grid_cc_sqrt<T>(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        if (!(n > static_cast<T>(0)))
            throw std::invalid_argument("grid_collision: planes row " + std::to_string(i) + " has a zero normal");
        env.planes.push_back(Plane<T>{ p[0] / n, p[1] / n, p[2] / n, p[3] / n });
    }
    return env;
}

template <typename T>
class DeviceEnvironment {
public:
    DeviceEnvironment() = default;
    explicit DeviceEnvironment(const HostEnvironment<T> &host) {
        if (cudaGetDevice(&device_) != cudaSuccess)
            throw std::runtime_error("grid_collision: no current CUDA device for the environment upload");
        try {
            d_spheres_  = upload(host.spheres);
            d_capsules_ = upload(host.capsules);
            d_cuboids_  = upload(host.cuboids);
            d_planes_   = upload(host.planes);
        } catch (...) {
            release();
            throw;
        }
        view_ = Environment<T>{ d_spheres_,  int(host.spheres.size()),  d_capsules_, int(host.capsules.size()),
                                d_cuboids_,  int(host.cuboids.size()),  d_planes_,   int(host.planes.size()) };
    }
    ~DeviceEnvironment() { release(); }
    DeviceEnvironment(const DeviceEnvironment &) = delete;
    DeviceEnvironment &operator=(const DeviceEnvironment &) = delete;
    DeviceEnvironment(DeviceEnvironment &&o) noexcept { take(o); }
    DeviceEnvironment &operator=(DeviceEnvironment &&o) noexcept {
        if (this != &o) { release(); take(o); }
        return *this;
    }
    // The device-pointer view kernels take by value; valid while this object lives.
    const Environment<T> &view() const { return view_; }
    int device() const { return device_; }
    int size() const { return view_.n_spheres + view_.n_capsules + view_.n_cuboids + view_.n_planes; }
    bool empty() const { return size() == 0; }
    // Free the device lists on the device that holds them; idempotent, never throws.
    void release() noexcept {
        if (device_ < 0) { view_ = Environment<T>{}; return; }
        int current = -1;
        const bool switch_device = (cudaGetDevice(&current) == cudaSuccess && current != device_);
        if (switch_device) cudaSetDevice(device_);
        if (d_spheres_)  cudaFree(d_spheres_);
        if (d_capsules_) cudaFree(d_capsules_);
        if (d_cuboids_)  cudaFree(d_cuboids_);
        if (d_planes_)   cudaFree(d_planes_);
        if (switch_device) cudaSetDevice(current);
        d_spheres_ = nullptr; d_capsules_ = nullptr; d_cuboids_ = nullptr; d_planes_ = nullptr;
        view_ = Environment<T>{};
        device_ = -1;
    }
private:
    template <typename S>
    static S *upload(const std::vector<S> &v) {
        if (v.empty()) return nullptr;
        S *d = nullptr;
        cudaError_t e = cudaMalloc(&d, sizeof(S) * v.size());
        if (e != cudaSuccess)
            throw std::runtime_error(std::string("grid_collision: environment upload failed: ") + cudaGetErrorString(e));
        e = cudaMemcpy(d, v.data(), sizeof(S) * v.size(), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            cudaFree(d);
            throw std::runtime_error(std::string("grid_collision: environment upload failed: ") + cudaGetErrorString(e));
        }
        return d;
    }
    void take(DeviceEnvironment &o) noexcept {
        d_spheres_ = o.d_spheres_; d_capsules_ = o.d_capsules_; d_cuboids_ = o.d_cuboids_; d_planes_ = o.d_planes_;
        view_ = o.view_; device_ = o.device_;
        o.d_spheres_ = nullptr; o.d_capsules_ = nullptr; o.d_cuboids_ = nullptr; o.d_planes_ = nullptr;
        o.view_ = Environment<T>{}; o.device_ = -1;
    }
    Sphere<T>  *d_spheres_  = nullptr;
    Capsule<T> *d_capsules_ = nullptr;
    Cuboid<T>  *d_cuboids_  = nullptr;
    Plane<T>   *d_planes_   = nullptr;
    Environment<T> view_{};
    int device_ = -1;
};

// One call from arrays to the device: validates, uploads, owns.
template <typename T>
inline DeviceEnvironment<T> make_device_environment(const T *spheres, int n_spheres,
                                                    const T *capsules, int n_capsules,
                                                    const T *cuboids, int n_cuboids,
                                                    const T *planes = nullptr, int n_planes = 0) {
    return DeviceEnvironment<T>(make_host_environment<T>(spheres, n_spheres, capsules, n_capsules,
                                                         cuboids, n_cuboids, planes, n_planes));
}

}  // namespace grid_collision

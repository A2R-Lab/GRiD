// wrapper_template.cu — Robot-agnostic C ABI on top of a generated grid.cuh.
//
// This file is the SAME for every robot — only the included grid.cuh
// (which carries the per-robot codegen output) differs. Compiled by
// grid_rbd._compile.compile_so() into ~/.cache/grid-rbd/store/<key>/robot.so
// at register_robot() time.
//
// The Python side (grid_rbd._runner) dlopens the resulting .so and calls
// these `extern "C"` symbols via ctypes.
//
// Conventions:
//   * All array layouts use the same convention as the generated host
//     wrappers (h_q_qd_u: [q[NJ], qd[NJ], u[NJ]] per timestep, contiguous).
//   * Functions take a `num_timesteps` (the actual batch size) which must be
//     <= GRID_RBD_MAX_BATCH (the compile-time N baked into init_gridData).
//   * Functions return 0 on success, non-zero on a CUDA error (caller can
//     log + reraise as a Python exception).

#include "grid.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>

// C4 EE binding aliases (codegen emits these in grid.cuh whenever kinematics are
// built; see gen_all_code). A single named fixed target routes them to the
// _<name> launchers + a 1-EE count. Fallback to the all-leaf default so a
// kinematics-free subset profile (where grid_rbd_num_ees() still references
// GRID_RBD_NUM_EES unconditionally) compiles; the FN/KERNEL aliases are only used
// under GRID_HAS_END_EFFECTOR_POSE* guards, so they need no fallback here.
#ifndef GRID_RBD_NUM_EES
#define GRID_RBD_NUM_EES grid::NUM_EES
#endif

// fp64 (Phase 8): the buffer/compute element type. Default float (byte-identical
// fp32 ABI). A .so built with -DGRID_WRAPPER_T_DOUBLE uses double — same symbol
// names, doubled smem footprint, re-derived spill tiers (see the matching codegen
// knob GRiDCodeGenerator(dtype="double")). The dtype is a property of WHICH .so
// you dlopen; the Runner (Runner vs RunnerF64) must match.
#ifdef GRID_WRAPPER_T_DOUBLE
using T = double;
#else
using T = float;
#endif

// Compile-time max batch size (overridable via -DGRID_RBD_MAX_BATCH=N).
#ifndef GRID_RBD_MAX_BATCH
#define GRID_RBD_MAX_BATCH 256
#endif
static constexpr int kMaxBatch = GRID_RBD_MAX_BATCH;

// ─── singleton state ─────────────────────────────────────────────────────────
// One Runner-process / one .so / one set of device buffers. Re-entrancy from
// multiple Python threads is the caller's responsibility (CUDA streams are
// fine; we use the single set of streams from init_grid<T>()).

// gridData's second template parameter is grid::gridDataKind, not the
// batch size — the batch is baked into init_gridData<T, N>() but the
// struct type itself doesn't carry it.
static grid::gridData<T>*    g_data   = nullptr;
static grid::robotModel<T>*  g_robot  = nullptr;
static cudaStream_t*         g_streams = nullptr;
// CUDA grid = ONE BLOCK PER BATCH SAMPLE (dim3(batch,1,1)), matching the C++ benchmark
// harness's launch geometry. GRiD kernels are single-block-per-sample (blockIdx.x indexes
// the timestep via a grid-stride over num_timesteps); launching gridDim=1 is CORRECT but
// runs the whole batch on ONE SM (~100x slow, linear in N). Every launch site below uses
// dim3((unsigned)batch,1,1) so jax / torch / pybind all call the kernel the SAME way it was
// autotuned. (Historically this was a stuck dim3(1,1,1) global that ran the whole batch on one
// block — fixed; each sample is now its own block.)
// Threads-per-block: default to the per-algo autotuned launch_cfg<ALGO>::THREADS
// baked into grid.cuh (each algo uses its own autotuned count, not one global default).
// g_threads_override
// lets the caller force one count for ALL algos: -1 = use the autotuned per-algo
// default; >=1 = force that many. Set via grid_rbd_set_threads_per_block().
static int g_threads_override = -1;

// Per-algo threads overlay (E6): the torch/pybind profiles share the baked .so but
// want DIFFERENT per-algo block sizes than the baked ffi default (CUDA-graph replay
// vs raw numpy shift the optimum). -1 = "use launch_cfg<ALGO>::THREADS". The GLOBAL
// override (g_threads_override) still wins when set (explicit user intent via
// set_threads_per_block). Applied at handle init via apply_profile_overlay(profile).
static int g_threads_per_algo[grid::GRID_ALGO_COUNT];
static bool g_threads_per_algo_init = false;
static inline void grid_rbd_threads_per_algo_init() {
    if (g_threads_per_algo_init) return;
    for (int i = 0; i < grid::GRID_ALGO_COUNT; ++i) g_threads_per_algo[i] = -1;
    g_threads_per_algo_init = true;
}

// Per-algo launch threads = the autotuned default unless the user forced an override.
// (GRID_ALGO_COUNT hits the primary launch_cfg template = MAX_PERF_LEVEL_THREADS, i.e.
// the historical default — use it for algos with no baked entry / plant kernels.)
//
// PER-ALGO TIER: the host-wrapper / direct-kernel launches below pass
// grid::launch_cfg<GRID_ALGO_X>::TIER (per-algo autotuned resource tier). A FEW calls
// INTENTIONALLY do NOT pass a tier — do not "fix" them:
//   * grid::idsva_so(...)            — a DISPATCHER wrapper with NO RESOURCE_TIER param
//                                       (it bakes the per-frame tier internally).
//   * the non-mjx grid::integrator / grid::com / grid::ccrba / grid::energy calls —
//     com/ccrba/energy have no baked launch_cfg entry (GRID_ALGO_COUNT == default tier,
//     so a tier would be a no-op); the non-mjx integrator's MUJOCO_OUTPUT gate is plain
//     floating_base (incl. mimic/skew) with no reliable floating-only macro here
//     (tracked: backlog item L / the descriptor-table refactor will unify this).
// All OTHER algos thread the tier through both the jax/torch direct launch AND the
// numpy/pybind C-ABI host-wrapper call (Transform A + the C-ABI tier wiring).
template <int ALGO>
static inline dim3 grid_rbd_launch_threads() {
    // Priority: explicit global override > per-algo profile overlay > baked launch_cfg.
    if (g_threads_override >= 1) return dim3((unsigned)g_threads_override, 1, 1);
    grid_rbd_threads_per_algo_init();
    // ALGO == GRID_ALGO_COUNT (plant / no-entry path) hits the primary launch_cfg
    // template; constexpr-exclude it so the per-algo array is never indexed OOB.
    if constexpr (ALGO >= 0 && ALGO < grid::GRID_ALGO_COUNT) {
        if (g_threads_per_algo[ALGO] >= 1) return dim3((unsigned)g_threads_per_algo[ALGO], 1, 1);
    }
    int n = grid::launch_cfg<ALGO>::THREADS;
    return dim3((unsigned)n, 1, 1);
}

// Clamp a requested thread count to a specific kernel's register-limited maxThreadsPerBlock.
// Register-heavy kernels (e.g. momentum_cost at ~140 regs/thread => max 384) cannot
// launch at the default MAX_PERF_LEVEL_THREADS (448); without this clamp the launch
// fails with cudaErrorInvalidConfiguration ("too many resources requested"), which
// cudaDeviceSynchronize() does NOT report — the kernel silently never runs and the
// output buffers are left stale. All GRID single-block kernels are thread-count-
// invariant (block-stride SIMT loops), so launching with fewer threads is correct.
template <typename KernelPtr>
static dim3 grid_clamp_threads_for(KernelPtr kernel, dim3 requested) {
    cudaFuncAttributes attr;
    if (cudaFuncGetAttributes(&attr, (const void*)kernel) != cudaSuccess) {
        cudaGetLastError();  // swallow — fall back to the requested dims
        return requested;
    }
    unsigned cap = (attr.maxThreadsPerBlock > 0) ? (unsigned)attr.maxThreadsPerBlock : requested.x;
    if (requested.x > cap) requested.x = cap;
    return requested;
}

// Grid dim for a direct batch launch. Kernels index the per-block workspace
// arena by blockIdx and grid-stride over timesteps, so launching more blocks
// than init_gridData actually fitted (workspace_timestep_slots can be FEWER
// than kMaxBatch under device memory pressure) aliases live workspace across
// blocks — silent wrong results. Clamping is exactly what the generated host
// wrappers do (_grid_ws_n); grid-striding makes it correct for every batch
// kernel, workspace-consuming or not.
static inline dim3 grid_rbd_grid_for(int batch) {
    int slots = (g_data && g_data->workspace_timestep_slots > 0)
                    ? g_data->workspace_timestep_slots : batch;
    int blocks = batch < slots ? batch : slots;
    return dim3((unsigned)(blocks > 0 ? blocks : 1), 1, 1);
}

// ─── lifecycle ───────────────────────────────────────────────────────────────

extern "C" int grid_rbd_init() {
    if (g_data) return 0;  // already initialized
    grid_consume_last_error();  // clear stale sticky state (GRID_GPUERRCHK_NO_EXIT builds)
    g_streams = grid::init_grid<T>();
    g_robot   = grid::init_robotModel<T>();
    g_data    = grid::init_gridData<T, kMaxBatch>();
    // Bindings compile with -DGRID_GPUERRCHK_NO_EXIT: a failed cudaMalloc etc. in
    // the generated init path no longer exit()s the embedding interpreter — it
    // lands in the sticky slot. A partially-initialized arena is NOT safe to
    // close (unset device pointers); drop it and report rc = 100 + cudaError.
    cudaError_t e = grid_consume_last_error();
    if (e != cudaSuccess) {
        g_data = nullptr; g_robot = nullptr; g_streams = nullptr;
        return 100 + (int)e;
    }
    return (g_data && g_robot && g_streams) ? 0 : 1;
}

extern "C" int grid_rbd_close() {
    if (!g_data) return 0;
    grid_consume_last_error();
    grid::close_grid<T>(g_streams, g_robot, g_data);
    g_data    = nullptr;
    g_robot   = nullptr;
    g_streams = nullptr;
    cudaError_t e = grid_consume_last_error();
    return (e != cudaSuccess) ? 100 + (int)e : 0;
}

// ─── metadata ────────────────────────────────────────────────────────────────

extern "C" int grid_rbd_num_joints()     { return grid::NUM_JOINTS; }
extern "C" int grid_rbd_num_vel()        { return grid::NUM_VEL; }
extern "C" int grid_rbd_num_ees()        { return GRID_RBD_NUM_EES; }
extern "C" int grid_rbd_num_bodies()     { return grid::NUM_BODIES; }
extern "C" int grid_rbd_max_batch()      { return kMaxBatch; }
extern "C" int grid_rbd_max_perf_level_threads() { return grid::MAX_PERF_LEVEL_THREADS; }
// Returns the active global override: -1 means "use the per-algo autotuned
// default" (launch_cfg<ALGO>::THREADS baked into grid.cuh); a value >=1 means
// the caller forced that thread count for ALL algos via set_threads_per_block.
extern "C" int grid_rbd_threads_per_block() { return g_threads_override; }
extern "C" int grid_rbd_set_threads_per_block(int n) {
    // Control the per-block thread count used for all subsequent kernel launches.
    // The DEFAULT is per-algo autotuned: each call defaults its threads-per-block
    // to that algorithm's launch_cfg<ALGO>::THREADS baked into grid.cuh (per-algo,
    // not one global count). This setter forces a single global override:
    //   * n == 0  -> reset to the per-algo autotuned defaults.
    //   * n >= 1  -> force that many threads for EVERY algo (overrides the autotune).
    //   * n <  0  -> invalid (returns 1, no change).
    // Values larger than a kernel's register-limited max will fail at launch time
    // with cudaErrorInvalidConfiguration; the codegen no longer pins launch_bounds,
    // so any block size with enough threads to cover the parallel work is valid (the
    // SIMT helpers use block-stride loops, so smaller block sizes are correct but slower).
    if (n < 0) return 1;
    g_threads_override = (n == 0) ? -1 : n;
    return 0;
}

// ─── per-algo threads overlay (E6) ───────────────────────────────────────────
// Number of baked algos = the GridAlgo enum size; Python derives the overlay index
// from the SAME descriptor-table launch order (algo_registry.ALGO_DESCRIPTORS) and
// asserts it matches.
extern "C" int grid_rbd_algo_count() { return grid::GRID_ALGO_COUNT; }
// Set a per-algo threads override. algo = the GridAlgo enum index. n==0 -> clear
// (back to launch_cfg<ALGO>::THREADS); n>=1 -> force for that algo only. The global
// override (set_threads_per_block) still takes precedence when set.
extern "C" int grid_rbd_set_threads_for(int algo, int n) {
    if (algo < 0 || algo >= grid::GRID_ALGO_COUNT || n < 0) return 1;
    grid_rbd_threads_per_algo_init();
    g_threads_per_algo[algo] = (n == 0) ? -1 : n;
    return 0;
}

// ─── kernel introspection: real compiled __launch_bounds__ ceiling (E1) ──────
//
// grid_rbd_kernel_max_threads(algo) returns the REAL maxThreadsPerBlock of the
// kernel this binding bakes for `algo` — cudaFuncGetAttributes() on
// grid::<algo>_kernel instantiated at the SAME tier the host launchers use
// (launch_cfg<ALGO>::TIER, forwarded into the kernel's RESOURCE_TIER). That is
// min(__launch_bounds__(tier_max_threads<TIER>()), register-limited max), so the
// FFI autotune can record a SELF-CONSISTENT {tier, threads} instead of guessing a
// host tier and clamping to it (the tier-contract fix; design_autotune_matrix.md
// §1). Returns -1 on a null/unknown key or a CUDA error — the Python side then
// falls back to swept-ceiling inference and never crashes (so a stale .so missing
// this symbol degrades gracefully). Reuses grid_clamp_threads_for's mechanism.
//
// `algo` is the SHORT autotune key (id, minv, fd, aba, crba, id_du, fd_du,
// ee_pose, ee_pose_gradient, ee_pose_hessian, idsva_so, fdsva_so) — same keys the
// sweep passes. Each branch is guarded by the algo's GRID_HAS_* macro so a subset
// .so that didn't emit a kernel returns -1 rather than failing to link. Overloaded
// kernels (id / id_du / fd_du have qdd + no-qdd overloads that SHARE
// __launch_bounds__) are disambiguated with an explicit function-pointer cast to
// one overload's signature — either reports the identical ceiling (mirrors the
// codegen's own static_cast<void(*)(...)> kernel aliases).
static int grid_kernel_ceiling(const void* fp) {
    cudaFuncAttributes attr;
    if (cudaFuncGetAttributes(&attr, fp) != cudaSuccess) {
        cudaGetLastError();  // swallow — report "unknown" so Python falls back
        return -1;
    }
    return (attr.maxThreadsPerBlock > 0) ? attr.maxThreadsPerBlock : -1;
}

// Take the address of grid::KERN<T, launch_cfg<ALGO>::TIER>, cast to SIG (selects
// one overload for the overloaded kernels; a no-op for single-definition ones),
// and read its maxThreadsPerBlock.
#define GRID_KERNEL_CEIL(KERN, ALGO, ...) \
    grid_kernel_ceiling((const void*)static_cast<__VA_ARGS__>( \
        &grid::KERN<T, grid::launch_cfg<grid::ALGO>::TIER>))

extern "C" int grid_rbd_kernel_max_threads(const char* algo) {
    if (!algo) return -1;
    using RM = const grid::robotModel<T>*;
#if GRID_HAS_INVERSE_DYNAMICS
    if (std::strcmp(algo, "id") == 0)
        return GRID_KERNEL_CEIL(inverse_dynamics_kernel, GRID_ALGO_INVERSE_DYNAMICS,
                                void(*)(T*, const T*, const int, T*, RM, const T, const int));
#endif
#if GRID_HAS_MINV
    if (std::strcmp(algo, "minv") == 0)
        return GRID_KERNEL_CEIL(minv_kernel, GRID_ALGO_MINV,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const int));
#endif
#if GRID_HAS_FORWARD_DYNAMICS
    if (std::strcmp(algo, "fd") == 0)
        return GRID_KERNEL_CEIL(forward_dynamics_kernel, GRID_ALGO_FORWARD_DYNAMICS,
                                void(*)(T*, unsigned char*, const T*, const int, T*, RM, const T, const int));
#endif
#if GRID_HAS_ABA
    if (std::strcmp(algo, "aba") == 0)
        return GRID_KERNEL_CEIL(aba_kernel, GRID_ALGO_ABA,
                                void(*)(T*, unsigned char*, const T*, const int, T*, RM, const T, const int));
#endif
#if GRID_HAS_CRBA
    if (std::strcmp(algo, "crba") == 0)
        return GRID_KERNEL_CEIL(crba_kernel, GRID_ALGO_CRBA,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const T, const int));
#endif
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
    if (std::strcmp(algo, "id_du") == 0)
        return GRID_KERNEL_CEIL(inverse_dynamics_gradient_kernel, GRID_ALGO_INVERSE_DYNAMICS_GRADIENT,
                                void(*)(T*, unsigned char*, const T*, const int, T*, RM, const T, const int));
#endif
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
    if (std::strcmp(algo, "fd_du") == 0)
        return GRID_KERNEL_CEIL(forward_dynamics_gradient_kernel, GRID_ALGO_FORWARD_DYNAMICS_GRADIENT,
                                void(*)(T*, unsigned char*, const T*, const int, T*, RM, const T, const int));
#endif
#if GRID_HAS_END_EFFECTOR_POSE
    if (std::strcmp(algo, "ee_pose") == 0)
        return GRID_KERNEL_CEIL(GRID_RBD_EE_POSE_KERNEL, GRID_ALGO_END_EFFECTOR_POSE,
                                void(*)(T*, const T*, const int, RM, const int));
#endif
#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
    if (std::strcmp(algo, "ee_pose_gradient") == 0)
        return GRID_KERNEL_CEIL(GRID_RBD_EE_POSE_GRADIENT_KERNEL, GRID_ALGO_END_EFFECTOR_POSE_GRADIENT,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const int));
#endif
#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN
    if (std::strcmp(algo, "ee_pose_hessian") == 0)
        return GRID_KERNEL_CEIL(GRID_RBD_EE_POSE_HESSIAN_KERNEL, GRID_ALGO_END_EFFECTOR_POSE_HESSIAN,
                                void(*)(T*, T*, unsigned char*, const T*, const int, RM, const int));
#endif
    if (std::strcmp(algo, "idsva_so") == 0) {
        // Dispatcher: the codegen emits EXACTLY ONE concrete frame kernel per robot
        // (world for floating/spherical, body for cardinal fixed). Query whichever
        // variant is present, at its frame-specific tier. Frame-specific ceilings can
        // differ (different register footprints) — correct, we want the one that runs.
#if GRID_HAS_IDSVA_SO_WORLD_FRAME
        return GRID_KERNEL_CEIL(idsva_so_world_frame_kernel, GRID_ALGO_IDSVA_SO_WORLD_FRAME,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const T, const int));
#elif GRID_HAS_IDSVA_SO_BODY_FRAME
        return GRID_KERNEL_CEIL(idsva_so_body_frame_kernel, GRID_ALGO_IDSVA_SO_BODY_FRAME,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const T, const int));
#else
        return -1;
#endif
    }
#if GRID_HAS_FDSVA_SO
    if (std::strcmp(algo, "fdsva_so") == 0)
        return GRID_KERNEL_CEIL(fdsva_so_kernel, GRID_ALGO_FDSVA_SO,
                                void(*)(T*, unsigned char*, const T*, const int, T*, RM, const T, const int));
#endif
    return -1;  // unknown / not-built algo key
}
#undef GRID_KERNEL_CEIL

// ─── runtime-mutable inertia (D.4 / Phase 5) ─────────────────────────────────
//
// Gated on GRID_RBD_RUNTIME_INERTIA, which grid_rbd._compile sets (alongside the
// codegen `runtime_inertia` flag) only when the robot was registered with
// runtime_inertia=True. The generated grid.cuh then exports grid::set_inertia_params
// (a thin cudaMemcpy into the device-resident d_inertia_params table). Without the
// flag the symbol is absent and the Runner's set_inertia_params raises a clear error.
//
// h_params: 10*grid::NUM_BODIES scalars, body-indexed bodies 1..N (the synthetic
// world-frame link is dropped, mirroring init_inertia_params / the I-region of
// d_XImats), each a length-10 [m, h(3)=m*c, I_O(6)] vector in the frozen
// regressor basis. The device table is shared by all subsequent kernel calls (the
// I-region of s_XImats is rebuilt from it on the cold XImats load). Returns 0 on
// success.
//
// NB: the table is sized by grid::NUM_BODIES (the inertia-body count), NOT
// grid::NUM_JOINTS. For a FIXED base these coincide; for a FLOATING base (or a
// mimic robot) NUM_JOINTS == NUM_POS > NUM_BODIES, so using NUM_JOINTS here would
// over-report the table size and reject the only correct (NUM_BODIES, 10) table.
#ifdef GRID_RBD_RUNTIME_INERTIA
extern "C" int grid_rbd_set_inertia_params(const T* h_params) {
    if (!g_robot) { int rc = grid_rbd_init(); if (rc) return rc; }
    grid::set_inertia_params<T>(g_robot, h_params);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = grid_consume_last_error();  // NO_EXIT sticky (launch-time failures)
    return (err == cudaSuccess) ? 0 : (int)err;
}
extern "C" int grid_rbd_inertia_params_size() { return 10 * grid::NUM_BODIES; }
#endif

// ─── runtime-mutable joint-frame transform (runtime_transform) ────────────────
//
// Gated on GRID_RBD_RUNTIME_TRANSFORM (set by grid_rbd._compile alongside the
// codegen `runtime_transform` flag). The generated grid.cuh exports
// grid::set_transform_params (a thin cudaMemcpy into the device-resident
// d_transform_params table). h_params: 6*grid::NUM_JOINTS scalars, joint-indexed
// ALL joints 0..NB-1, each a [x,y,z,roll,pitch,yaw] raw URDF <origin> vector. The
// device rebuilds each joint's constant Xfixed from it once per launch. The table
// is sized by NUM_JOINTS (one origin per joint), NOT NUM_BODIES. Returns 0 on
// success.
#ifdef GRID_RBD_RUNTIME_TRANSFORM
extern "C" int grid_rbd_set_transform_params(const T* h_params) {
    if (!g_robot) { int rc = grid_rbd_init(); if (rc) return rc; }
    grid::set_transform_params<T>(g_robot, h_params);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = grid_consume_last_error();  // NO_EXIT sticky (launch-time failures)
    return (err == cudaSuccess) ? 0 : (int)err;
}
extern "C" int grid_rbd_transform_params_size() { return 6 * grid::NUM_JOINTS; }
#endif

// ─── runtime-mutable joint dynamics (runtime_joint_dynamics) ──────────────────
//
// Gated on GRID_RBD_RUNTIME_JOINT_DYNAMICS (set by grid_rbd._compile alongside the
// codegen runtime_joint_dynamics flag). grid.cuh then exports
// grid::set_joint_dynamics_params (a thin cudaMemcpy into the device-resident
// d_joint_dynamics_params table). h_params: 2*grid::NUM_VEL scalars,
// [damping(nv) || friction(nv)], v-slot indexed and ALPHA-FOLDED (one fused
// coefficient per v-slot, matching init_joint_dynamics_params). The table is
// bit-identical to the baked literal bias until poked (no sincos rebuild, no
// sparsity change). Returns 0 on success.
#ifdef GRID_RBD_RUNTIME_JOINT_DYNAMICS
extern "C" int grid_rbd_set_joint_dynamics_params(const T* h_params) {
    if (!g_robot) { int rc = grid_rbd_init(); if (rc) return rc; }
    grid::set_joint_dynamics_params<T>(g_robot, h_params);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = grid_consume_last_error();  // NO_EXIT sticky (launch-time failures)
    return (err == cudaSuccess) ? 0 : (int)err;
}
extern "C" int grid_rbd_joint_dynamics_params_size() { return 2 * grid::NUM_VEL; }
#endif

// ─── shared input-packing helper ─────────────────────────────────────────────
//
// h_q_qd_u layout (matches generated host wrappers):
//   timestep t: [q[0..NJ-1], qd[0..NJ-1], u[0..NJ-1]]
//   contiguous across t.

static inline void pack_q_qd_u(const T* q, const T* qd, const T* u,
                               int batch, int num_joints)
{
    const int stride = 3 * num_joints;
    for (int t = 0; t < batch; ++t) {
        std::memcpy(&g_data->h_q_qd_u[t * stride + 0],          &q[t * num_joints],
                    num_joints * sizeof(T));
        std::memcpy(&g_data->h_q_qd_u[t * stride + num_joints], &qd[t * num_joints],
                    num_joints * sizeof(T));
        if (u) {
            std::memcpy(&g_data->h_q_qd_u[t * stride + 2 * num_joints], &u[t * num_joints],
                        num_joints * sizeof(T));
        }
    }
}

// ─── external-force helper ───────────────────────────────────────────────────
//
// f_ext layout (caller side): (batch, 6*NUM_BODIES) row-major, body-major per
// timestep, each per-body wrench ordered [angular(3); linear(3)] in the body's
// LOCAL frame — identical to gridData::h_f_ext / d_f_ext and to RBDReference's
// apply_external_forces (the kernel does f -= f_ext). A null f_ext leaves the
// (zeroed) singleton buffer untouched, so the no-f_ext path is byte-identical
// to before this surface existed.
//
// apply_f_ext() copies the user's wrench into the device buffer; reset_f_ext()
// re-zeroes it after the launch so a later no-f_ext call sees a clean buffer
// (the singleton is shared across calls). batch must be <= kMaxBatch.

static inline int apply_f_ext(const T* f_ext, int batch) {
    if (!f_ext) return 0;
    const size_t n = (size_t)6 * grid::NUM_BODIES * batch;
    std::memcpy(g_data->h_f_ext, f_ext, n * sizeof(T));
    if (cudaMemcpy(g_data->d_f_ext, g_data->h_f_ext, n * sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 5;
    return 0;
}

static inline void reset_f_ext(const T* f_ext, int batch) {
    if (!f_ext) return;
    const size_t n = (size_t)6 * grid::NUM_BODIES * batch;
    std::memset(g_data->h_f_ext, 0, n * sizeof(T));
    cudaMemset(g_data->d_f_ext, 0, n * sizeof(T));
}

// ─── algorithms ──────────────────────────────────────────────────────────────

// RNEA: c = M(q)·qdd + h(q,qd) − g(q)  (with qdd defaulting to 0 if null)
// f_ext (optional, may be null): (batch, 6*NUM_BODIES) local-frame body wrenches.
//
// qdd wiring: the generated grid::inverse_dynamics<T, USE_QDD_FLAG> HOST wrapper
// reads the joint acceleration from the SEPARATE gridData buffer hd_data->d_qdd
// (NOT the u-slot of d_q_qd_u) and, when USE_QDD_FLAG=true, copies h_qdd→d_qdd
// itself. So we fill g_data->h_qdd from the caller's qdd and instantiate the
// USE_QDD_FLAG=true overload; a null qdd keeps the (faster) qdd=0 overload.
extern "C" int grid_rbd_inverse_dynamics(
    const T* q, const T* qd, const T* qdd_opt,
    T* c_out,
    int batch, T gravity, const T* f_ext)
{
#if GRID_HAS_INVERSE_DYNAMICS
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;  // caller should chunk

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, nullptr, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    // The non-mjx host wrapper carries a MUJOCO_OUTPUT template param ONLY for a
    // floating non-mimic/skew robot (exactly when GRID_RBD_WITH_MUJOCO is defined),
    // so the RESOURCE_TIER position shifts by one. Gate the trailing template args
    // on that same macro to reach the per-algo autotuned tier on every robot shape.
    if (qdd_opt) {
        // Host wrapper copies h_qdd→d_qdd (NUM_JOINTS per timestep, contiguous).
        std::memcpy(g_data->h_qdd, qdd_opt, (size_t)batch * nj * sizeof(T));
// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_INVERSE_DYNAMICS)
        grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER>(
#else
        grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER>(
#endif
            g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS>(), g_streams);
    } else {
// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_INVERSE_DYNAMICS)
        grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/false, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER>(
#else
        grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/false, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER>(
#endif
            g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS>(), g_streams);
    }

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(c_out, g_data->h_c, batch * nj * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)qdd_opt; (void)c_out; (void)batch; (void)gravity; (void)f_ext;
    return 3;  // inverse_dynamics not built into this .so (subset codegen profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_INVERSE_DYNAMICS
// MuJoCo-convention inverse dynamics (floating base only). Identical signature to
// grid_rbd_inverse_dynamics, but q/qd/qdd are MuJoCo-native (quat wxyz, free-joint
// velocity [v_lin GLOBAL; omega LOCAL]) and the returned tau is in the mjx frame.
//
// The mjx output convention is baked into the KERNEL via the compile-time
// MUJOCO_OUTPUT=true template arg: the kernel converts the inputs mjx->pin on load
// (quat reorder + base velocity/acceleration reframe) and rotates the base-linear
// tau rows back to the mjx frame before saving — so NO host-side pre/post-process is
// needed (this is the fast path that replaces pin-kernel + _mujoco.py rotation).
//
// qdd is REQUIRED: the qdd=0 "bias" path cannot represent mjx (mjx qacc=0 implies a
// nonzero pin acceleration -omega x v — the nonlinear_effects accel-coupling), so a
// null qdd returns rc=4. Callers wanting the mjx bias use nonlinear_effects instead.
extern "C" int grid_rbd_inverse_dynamics_mujoco(
    const T* q, const T* qd, const T* qdd_opt,
    T* c_out,
    int batch, T gravity, const T* f_ext)
{
    if (!qdd_opt) return 4;  // mjx requires an explicit qdd (see note above)
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, nullptr, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    // Host wrapper copies h_qdd->d_qdd (NUM_JOINTS per timestep, contiguous).
    std::memcpy(g_data->h_qdd, qdd_opt, (size_t)batch * nj * sizeof(T));
    grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false,
                           /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(c_out, g_data->h_c, batch * nj * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_INVERSE_DYNAMICS

// Direct mass-matrix inverse: Minv(q)
extern "C" int grid_rbd_minv(
    const T* q,
    T* minv_out,
    int batch)
{
#if GRID_HAS_MINV
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    pack_q_qd_u(q, /*qd=*/q, /*u=*/nullptr, batch, nj);  // qd/u unused by minv

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_MINV)
    grid::minv<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER>(
#else
    grid::minv<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER>(
#endif
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_MINV>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;

    // The minv kernel writes Minv as nv x nv (NUM_VEL*NUM_VEL = 324 for floating)
    // per timestep, but the generated grid::minv host wrapper copies d_Minv->h_Minv
    // with an nj*nj (NUM_JOINTS*NUM_JOINTS) stride, which over-reads each 324-block
    // and corrupts h_Minv[1:] for batch>1. Copy straight from the (correctly
    // nv*nv-strided) device buffer instead, sizing the public output nv x nv.
    // For a FIXED base nv == nj, so this is byte-identical to the old path.
    cudaMemcpy(minv_out, g_data->d_Minv, (size_t)batch * nv * nv * sizeof(T),
               cudaMemcpyDeviceToHost);
    return 0;
#else
    (void)q; (void)minv_out; (void)batch;
    return 3;  // minv not built into this .so (subset codegen profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_MINV
// MuJoCo-convention direct mass-matrix inverse (floating base only): the kernel
// reorders the quaternion and applies the congruence Minv_mjx = G^-T Minv_pin G^-1
// on the base block (MUJOCO_OUTPUT=true). The native kernel writes a FULL DENSE
// SYMMETRIC mjx Minv (both triangles), so NO host symmetrize and NO host
// minv_pin_to_mjx post-process are needed.
extern "C" int grid_rbd_minv_mujoco(
    const T* q,
    T* minv_out,
    int batch)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    pack_q_qd_u(q, /*qd=*/q, /*u=*/nullptr, batch, nj);  // qd/u unused by minv

    grid::minv<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
               /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_MINV>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;

    cudaMemcpy(minv_out, g_data->d_Minv, (size_t)batch * nv * nv * sizeof(T),
               cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_MINV

// Forward dynamics: qdd = Minv(q)·(τ − c(q,qd))
// f_ext (optional, may be null): (batch, 6*NUM_BODIES) local-frame body wrenches.
extern "C" int grid_rbd_forward_dynamics(
    const T* q, const T* qd, const T* u,
    T* qdd_out,
    int batch, T gravity, const T* f_ext)
{
#if GRID_HAS_FORWARD_DYNAMICS
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_FORWARD_DYNAMICS)
    grid::forward_dynamics<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER>(
#else
    grid::forward_dynamics<T, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER>(
#endif
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_FORWARD_DYNAMICS>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(qdd_out, g_data->h_qdd, batch * nj * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)u; (void)qdd_out; (void)batch; (void)gravity; (void)f_ext;
    return 3;  // forward_dynamics not built into this .so (subset codegen profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_FORWARD_DYNAMICS
// MuJoCo-convention forward dynamics (floating base only). q/qd/u are MuJoCo-native;
// the kernel converts inputs mjx->pin on load and maps the output acceleration
// qdd[0:3] = R(qdd_pin + omega x v) back to the mjx frame (MUJOCO_OUTPUT=true) — no
// host pre/post-process. f_ext is not reframed by the kernel input-convert, so the
// _handle dispatch only takes this path when f_ext is null.
extern "C" int grid_rbd_forward_dynamics_mujoco(
    const T* q, const T* qd, const T* u,
    T* qdd_out,
    int batch, T gravity, const T* f_ext)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    grid::forward_dynamics<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_FORWARD_DYNAMICS>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(qdd_out, g_data->h_qdd, batch * nj * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_FORWARD_DYNAMICS

// Articulated body algorithm: qdd = aba(q, qd, u)
// f_ext (optional, may be null): (batch, 6*NUM_BODIES) local-frame body wrenches.
extern "C" int grid_rbd_aba(
    const T* q, const T* qd, const T* u,
    T* qdd_out,
    int batch, T gravity, const T* f_ext)
{
#if GRID_HAS_ABA
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_ABA)
    grid::aba<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER>(
#else
    grid::aba<T, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER>(
#endif
        g_data, g_robot, gravity, batch,
                 dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_ABA>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(qdd_out, g_data->h_qdd, batch * nj * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)u; (void)qdd_out; (void)batch; (void)gravity; (void)f_ext;
    return 3;  // aba not built into this .so (subset codegen profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_ABA
// MuJoCo-convention ABA (floating base only). Same accel_out convention as
// forward_dynamics: q/qd/u raw mjx in, mjx-frame qdd out (MUJOCO_OUTPUT=true). f_ext
// not reframed -> the _handle dispatch only uses this path when f_ext is null.
extern "C" int grid_rbd_aba_mujoco(
    const T* q, const T* qd, const T* u,
    T* qdd_out,
    int batch, T gravity, const T* f_ext)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    grid::aba<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_ABA>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(qdd_out, g_data->h_qdd, batch * nj * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_ABA

// Composite rigid body algorithm: M = crba(q)
extern "C" int grid_rbd_crba(
    const T* q,
    T* m_out,
    int batch, T gravity)
{
#if GRID_HAS_CRBA
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    pack_q_qd_u(q, q, nullptr, batch, nj);  // qd/u unused

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_CRBA)
    grid::crba<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER>(
#else
    grid::crba<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER>(
#endif
        g_data, g_robot, gravity, batch,
                  dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_CRBA>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;

    // The crba kernel writes M as nv x nv (NUM_VEL*NUM_VEL = 324 for floating)
    // per timestep. (The generated grid::crba host wrapper already copies
    // d_M->h_M at the correct nv*nv stride, but the public copy-out used the
    // nj*nj stride, over-reading and corrupting m_out[1:] for batch>1.) Copy
    // straight from the device buffer at nv*nv to be unambiguous and to match
    // the kernel; FIXED base has nv == nj so this is byte-identical.
    cudaMemcpy(m_out, g_data->d_M, (size_t)batch * nv * nv * sizeof(T),
               cudaMemcpyDeviceToHost);
    return 0;
#else
    (void)q; (void)m_out; (void)batch; (void)gravity;
    return 3;  // crba not built into this .so (subset codegen profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_CRBA
// MuJoCo-convention mass matrix (floating base only): M_mjx = G M_pin G^T, the
// congruence baked into the kernel (MUJOCO_OUTPUT=true). q is MuJoCo-native (quat
// wxyz); the kernel reorders the quaternion and applies the congruence on the base
// block before saving, so NO host pre/post-process is needed.
extern "C" int grid_rbd_crba_mujoco(
    const T* q,
    T* m_out,
    int batch, T gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    pack_q_qd_u(q, q, nullptr, batch, nj);  // qd/u unused

    grid::crba<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
               /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_CRBA>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;

    cudaMemcpy(m_out, g_data->d_M, (size_t)batch * nv * nv * sizeof(T),
               cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_CRBA

// End-effector pose: 6×NUM_EES per timestep (xyz + rpy).
extern "C" int grid_rbd_end_effector_pose(
    const T* q,
    T* ee_out,
    int batch)
{
#if GRID_HAS_END_EFFECTOR_POSE
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, q, nullptr, batch, nj);

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_EE_POSE)
    grid::GRID_RBD_EE_POSE_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE>::TIER>(
#else
    grid::GRID_RBD_EE_POSE_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE>::TIER>(
#endif
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_END_EFFECTOR_POSE>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(ee_out, g_data->h_end_effector_pose, batch * 6 * GRID_RBD_NUM_EES * sizeof(T));
    return 0;
#else
    (void)q; (void)ee_out; (void)batch;
    return 3;  // end_effector_pose not built into this .so (subset codegen profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_END_EFFECTOR_POSE
// MuJoCo-convention end_effector_pose(q) -> 6*NUM_EES per timestep. The pose is
// frame-INVARIANT; the kernel (MUJOCO_OUTPUT=true) only converts the mjx-native q
// (quaternion reorder, like osc_inertia). Output byte-equal to feeding the pin
// kernel the pin-converted q.
extern "C" int grid_rbd_end_effector_pose_mujoco(const T* q, T* ee_out, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);
    grid::GRID_RBD_EE_POSE_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                            /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE>::TIER>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_END_EFFECTOR_POSE>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(ee_out, g_data->h_end_effector_pose, (size_t)batch * 6 * GRID_RBD_NUM_EES * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_END_EFFECTOR_POSE

// Batched forward kinematics (large-batch, one block/warp per sample):
//   q layout:     (batch, NUM_POS)            -> q[b*NUM_POS + j]
//   pose7 layout: (batch, 7) = [tx,ty,tz, qw,qx,qy,qz]
// use_warp selects the warp-cooperative inner (1) vs the thread inner (0).
// Only present when the generated header supports the standalone FK inner
// (fixed-base, non-mimic robots); floating-base/mimic robots return rc=3.
extern "C" int grid_rbd_fk_batched(
    const T* q,
    T* pose7_out,
    int batch, int use_warp)
{
#ifdef GRID_HAS_FK_BATCHED
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int n = grid::NUM_POS;
    // device scratch (cached, grown to kMaxBatch on first use)
    static T* d_q_fk = nullptr;
    static T* d_pose7 = nullptr;
    if (!d_q_fk) {
        if (cudaMalloc(&d_q_fk, sizeof(T) * kMaxBatch * n) != cudaSuccess) return 4;
        if (cudaMalloc(&d_pose7, sizeof(T) * kMaxBatch * 7) != cudaSuccess) return 4;
    }
    cudaMemcpy(d_q_fk, q, sizeof(T) * batch * n, cudaMemcpyHostToDevice);

    if (use_warp)
        grid::ee_pose_fk_batched<T, /*USE_WARP=*/true >(d_pose7, d_q_fk, batch, g_robot,
                                                        (int)grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>().x < 32 ? 32 : (int)grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>().x);
    else
        grid::ee_pose_fk_batched<T, /*USE_WARP=*/false>(d_pose7, d_q_fk, batch, g_robot,
                                                        (int)grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>().x);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;

    cudaMemcpy(pose7_out, d_pose7, sizeof(T) * batch * 7, cudaMemcpyDeviceToHost);
    return 0;
#else
    (void)q; (void)pose7_out; (void)batch; (void)use_warp;
    return 3;  // not supported for this robot (floating-base / spherical; mimic supported since 2026-08-01)
#endif
}

// End-effector pose Jacobian (d/dv tangent, pinocchio convention):
// 6×NUM_EES×NUM_VEL per timestep. Floating-base now produces the spatial
// Jacobian columns rather than the older non-standard quaternion-derivative
// columns (the v-tangent dimension is nv = 6 + n_joints vs the old nq = 7 +
// n_joints). Fixed-base shape unchanged (nq == nv).
extern "C" int grid_rbd_end_effector_pose_gradient(
    const T* q,
    T* dee_out,
    int batch)
{
#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    pack_q_qd_u(q, q, nullptr, batch, nj);

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_EE_POSE_GRADIENT)
    grid::GRID_RBD_EE_POSE_GRADIENT_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER>(
#else
    grid::GRID_RBD_EE_POSE_GRADIENT_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER>(
#endif
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(dee_out, g_data->h_end_effector_pose_gradient,
                batch * 6 * GRID_RBD_NUM_EES * nv * sizeof(T));
    return 0;
#else
    (void)q; (void)dee_out; (void)batch;
    return 3;  // end_effector_pose_gradient not built into this .so (subset profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_END_EFFECTOR_POSE_GRADIENT
// MuJoCo-convention end_effector_pose Jacobian (q) -> 6*NUM_EES*NUM_VEL per
// timestep. Column-reframe: q raw mjx in (kernel reorders the quaternion) and the
// kernel (MUJOCO_OUTPUT=true) reframes the base-linear Jacobian columns before
// saving (the column reframe acts on the NV axis cols 0:3).
extern "C" int grid_rbd_end_effector_pose_gradient_mujoco(const T* q, T* dee_out, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);
    grid::GRID_RBD_EE_POSE_GRADIENT_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                                     /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(dee_out, g_data->h_end_effector_pose_gradient,
                (size_t)batch * 6 * GRID_RBD_NUM_EES * grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_END_EFFECTOR_POSE_GRADIENT

// ∂c/∂(q, qd): output shape (batch, NV, 2*NV) — concatenated [dc_dq | dc_dqd]
// (tangent-space; FIXED base NV == NJ, FLOATING base NV < NJ).
// f_ext (optional, may be null): (batch, 6*NUM_BODIES) local-frame body wrenches.
// f_ext enters RNEA additively (affine), so dc/d(q,qd) is unchanged for a
// CONSTANT f_ext; this just keeps the bias consistent with grid_rbd_inverse_dynamics.
extern "C" int grid_rbd_inverse_dynamics_gradient(
    const T* q, const T* qd, const T* qdd_opt,
    T* dc_du_out,
    int batch, T gravity, const T* f_ext)
{
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, nullptr, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    // qdd dependence: ∂c/∂(q,qd) DOES depend on qdd (via the M·qdd term's
    // derivatives). Same separate-d_qdd convention as grid_rbd_inverse_dynamics:
    // the host wrapper copies h_qdd→d_qdd when USE_QDD_FLAG=true. Null qdd keeps
    // the qdd=0 overload.
    if (qdd_opt) {
        std::memcpy(g_data->h_qdd, qdd_opt, (size_t)batch * nj * sizeof(T));
// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_INVERSE_DYNAMICS_GRADIENT)
        grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(
#else
        grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(
#endif
            g_data, g_robot, gravity, batch,
            dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(), g_streams);
    } else {
// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_INVERSE_DYNAMICS_GRADIENT)
        grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/false, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(
#else
        grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/false, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(
#endif
            g_data, g_robot, gravity, batch,
            dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(), g_streams);
    }

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    // dc_du is nv x 2nv (2*NUM_VEL*NUM_VEL = 648 for floating) per timestep — the
    // gradient kernel writes nv-dimensioned rows/cols. The generated host wrapper
    // copies d_dc_du->h_dc_du at an nj*2nj stride (poisoning h_dc_du[1:] for
    // batch>1); copy straight from the device buffer at 2*nv*nv. FIXED base:
    // nv == nj, byte-identical to the old path.
    const int nv = grid::NUM_VEL;
    cudaMemcpy(dc_du_out, g_data->d_dc_du,
               (size_t)batch * 2 * nv * nv * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
#else
    (void)q; (void)qd; (void)qdd_opt; (void)dc_du_out; (void)batch; (void)gravity; (void)f_ext;
    return 3;  // inverse_dynamics_gradient not built into this .so (subset profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_INVERSE_DYNAMICS_GRADIENT
// MuJoCo-convention inverse-dynamics gradient (floating base only). q/qd/qdd are
// MuJoCo-native; the kernel converts inputs mjx->pin on load and applies the full
// gradient convention transform (reframe + base-row rotate + ω×v couplings, with M
// from an in-kernel crba reuse) so the returned dc/d(q,qd) is the mjx-frame gradient.
// REQUIRES qdd (the mjx gradient is the with-qdd surface; a null qdd returns rc=4).
extern "C" int grid_rbd_inverse_dynamics_gradient_mujoco(
    const T* q, const T* qd, const T* qdd_opt,
    T* dc_du_out,
    int batch, T gravity, const T* f_ext)
{
    if (!qdd_opt) return 4;  // mjx gradient requires an explicit qdd
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, nullptr, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    std::memcpy(g_data->h_qdd, qdd_opt, (size_t)batch * nj * sizeof(T));
    grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false,
                                    /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    const int nv = grid::NUM_VEL;
    cudaMemcpy(dc_du_out, g_data->d_dc_du,
               (size_t)batch * 2 * nv * nv * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_INVERSE_DYNAMICS_GRADIENT

// ∂qdd/∂(q, qd): output shape (batch, NV, 2*NV) (tangent-space; FIXED base
// NV == NJ, FLOATING base NV < NJ).
// f_ext (optional, may be null): (batch, 6*NUM_BODIES) local-frame body wrenches.
extern "C" int grid_rbd_forward_dynamics_gradient(
    const T* q, const T* qd, const T* u,
    T* df_du_out,
    int batch, T gravity, const T* f_ext)
{
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_FORWARD_DYNAMICS_GRADIENT)
    grid::forward_dynamics_gradient<T, /*USE_QDD_MINV_FLAG=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER>(
#else
    grid::forward_dynamics_gradient<T, /*USE_QDD_MINV_FLAG=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER>(
#endif
        g_data, g_robot, gravity, batch,
        dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    // df_du is nv x 2nv (2*NUM_VEL*NUM_VEL = 648 for floating) per timestep; same
    // nq-vs-nv stride bug as inverse_dynamics_gradient. Copy straight from the
    // (correctly 2*nv*nv-strided) device buffer. FIXED base: nv == nj, unchanged.
    const int nv = grid::NUM_VEL;
    cudaMemcpy(df_du_out, g_data->d_df_du,
               (size_t)batch * 2 * nv * nv * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
#else
    (void)q; (void)qd; (void)u; (void)df_du_out; (void)batch; (void)gravity; (void)f_ext;
    return 3;  // forward_dynamics_gradient not built into this .so (subset profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_FORWARD_DYNAMICS_GRADIENT
// MuJoCo-convention forward-dynamics gradient (floating base only). q/qd/u are
// MuJoCo-native; qdd is computed internally. The kernel converts inputs mjx->pin on
// load and applies the full gradient convention transform (reframe + base-row rotate
// + ω×v couplings) so the returned df/d(q,qd) is the mjx-frame gradient.
extern "C" int grid_rbd_forward_dynamics_gradient_mujoco(
    const T* q, const T* qd, const T* u,
    T* df_du_out,
    int batch, T gravity, const T* f_ext)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    grid::forward_dynamics_gradient<T, /*USE_QDD_MINV_FLAG=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                                    /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    const int nv = grid::NUM_VEL;
    cudaMemcpy(df_du_out, g_data->d_df_du,
               (size_t)batch * 2 * nv * nv * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_FORWARD_DYNAMICS_GRADIENT

// End-effector pose Hessian: 6×NUM_EES×NV×NV per timestep (d^2/dv^2 tangent).
// Calls grid::end_effector_pose_hessian which fills BOTH end_effector_pose_hessian AND
// end_effector_pose_gradient; we only copy end_effector_pose_hessian out. If the caller wants both they should
// call end_effector_pose_gradient separately (the kernels are fast enough
// that doing the work twice is fine for a small convenience).
extern "C" int grid_rbd_end_effector_pose_hessian(
    const T* q,
    T* d2ee_out,
    int batch)
{
#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    pack_q_qd_u(q, q, nullptr, batch, nj);

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_EE_POSE_HESSIAN)
    grid::GRID_RBD_EE_POSE_HESSIAN_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER>(
#else
    grid::GRID_RBD_EE_POSE_HESSIAN_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER>(
#endif
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(d2ee_out, g_data->h_end_effector_pose_hessian,
                batch * 6 * GRID_RBD_NUM_EES * nv * nv * sizeof(T));
    return 0;
#else
    (void)q; (void)d2ee_out; (void)batch;
    return 3;  // end_effector_pose_hessian not built into this .so (subset profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_END_EFFECTOR_POSE_HESSIAN
// MuJoCo-convention end_effector_pose Hessian (q) -> 6*NUM_EES*NV*NV per timestep.
// q is raw mjx (kernel reorders the quaternion); the kernel (MUJOCO_OUTPUT=true)
// double-column-reframes the Hessian (J·G^{-1} on both tangent indices) and adds the
// symmetrized base-rotation frame term before saving. Output is invariant-shaped.
extern "C" int grid_rbd_end_effector_pose_hessian_mujoco(const T* q, T* d2ee_out, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);
    grid::GRID_RBD_EE_POSE_HESSIAN_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                                    /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(d2ee_out, g_data->h_end_effector_pose_hessian,
                (size_t)batch * 6 * GRID_RBD_NUM_EES * grid::NUM_VEL * grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_END_EFFECTOR_POSE_HESSIAN

// Second-order inverse dynamics. Output is the concatenated SO tensor of
// shape SECOND_ORDER_TENSOR_SIZE = 4 * NV^3 per timestep (four NV^3 blocks:
// d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq). The Python side slices into the
// four named tensors.
extern "C" int grid_rbd_idsva_so(
    const T* q, const T* qd, const T* qdd,
    T* out, int batch, T gravity)
{
#if GRID_HAS_IDSVA_SO
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    // idsva_so reads the joint acceleration from the u-slot of d_q_qd_u (s_qdd);
    // pack qdd there so the second-order tensors use the requested acceleration.
    pack_q_qd_u(q, qd, qdd, batch, nj);

// signature switch (same rule as fdsva_so): floating builds carry MUJOCO_OUTPUT.
// RESOURCE_TIER must match the tier the autotuned thread count was picked for —
// the default-tier instantiation with a LITE-tuned count exceeded the default
// kernel's register-limited thread cap and failed the launch (invalid argument).
#if defined(GRID_RBD_SIG_MJX_IDSVA_SO)
    grid::idsva_so<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO>::TIER>(
#else
    grid::idsva_so<T, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO>::TIER>(
#endif
        g_data, g_robot, gravity, batch,
        dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_IDSVA_SO>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(out, g_data->h_idsva_so,
                batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)qdd; (void)out; (void)batch; (void)gravity;
    return 3;  // idsva_so not built into this .so (subset codegen profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_IDSVA_SO
// MuJoCo-convention idsva_so(q, qd, qdd) -> 4*NV^3 (floating base only). q/qd/qdd are
// raw mjx (kernel input-converts); the kernel (MUJOCO_OUTPUT=true) transforms all four
// 2nd-order tensors to the mjx frame (explicit-analytic SO transform + dM_dq closed
// form), reusing the id/crba/id-grad inners. The mjx kernel is register-heavy; the
// post-launch error check surfaces a silent launch-config failure as rc!=0.
extern "C" int grid_rbd_idsva_so_mujoco(const T* q, const T* qd, const T* qdd, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, qdd, batch, grid::NUM_JOINTS);
    grid::idsva_so<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_IDSVA_SO>(), g_streams);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;  // launch-config failure (e.g. too many registers)
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_idsva_so,
                batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_IDSVA_SO

// inverse_dynamics_regressor(q, qd, qdd) -> Y, row-major NV x (10*NUM_BODIES) per
// timestep. tau = Y . pi (pi = the 10*NUM_BODIES stacked link inertia params), so Y
// is exactly affine in each link's spatial inertia. The host wrapper reads q|qd|qdd
// from d_q_qd_u (qdd in the u-slot, like idsva_so) and writes hd_data->h_Y.
extern "C" int grid_rbd_inverse_dynamics_regressor(
    const T* q, const T* qd, const T* qdd,
    T* out, int batch, T gravity)
{
#if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, qdd, batch, nj);
    grid::inverse_dynamics_regressor<T>(
        g_data, g_robot, gravity, batch,
        dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_Y,
                (size_t)batch * grid::NUM_VEL * 10 * grid::NUM_BODIES * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)qdd; (void)out; (void)batch; (void)gravity;
    return 3;  // inverse_dynamics_regressor not built into this .so (subset profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
// MuJoCo-convention inverse_dynamics_regressor (floating base only). q/qd/qdd are raw
// mjx (kernel input-converts); the regressor ROWS are tangent-indexed generalized
// forces, so the base-LINEAR rows (0:3) rotate by R (Y_mjx[0:3] = R Y_pin[0:3]) -- the
// same base-row rotate as id_tau. Baked via the MUJOCO_OUTPUT=true template flag.
extern "C" int grid_rbd_inverse_dynamics_regressor_mujoco(
    const T* q, const T* qd, const T* qdd,
    T* out, int batch, T gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, qdd, batch, nj);
    grid::inverse_dynamics_regressor<T, /*USE_COMPRESSED_MEM=*/false, grid::GRID_DATA_ALL,
                                     /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, gravity, batch,
        dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_Y,
                (size_t)batch * grid::NUM_VEL * 10 * grid::NUM_BODIES * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_INVERSE_DYNAMICS_REGRESSOR

// Second-order forward dynamics. Output is 4 * NV^3 per timestep
// (d2qdd_dq, d2qdd_dqd, d2qdd_dudq — interpretation per Singh/Wensing).
extern "C" int grid_rbd_fdsva_so(
    const T* q, const T* qd, const T* u,
    T* out, int batch, T gravity)
{
#if GRID_HAS_FDSVA_SO
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_FDSVA_SO)
    grid::fdsva_so<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER>(
#else
    grid::fdsva_so<T, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER>(
#endif
        g_data, g_robot, gravity, batch,
        dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_FDSVA_SO>(), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(out, g_data->h_df2,
                batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)u; (void)out; (void)batch; (void)gravity;
    return 3;  // fdsva_so not built into this .so (subset codegen profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_FDSVA_SO
// MuJoCo-convention fdsva_so(q, qd, u) -> 4*NV^3 (floating base only). q/qd/u raw mjx
// (kernel input-converts); the kernel (MUJOCO_OUTPUT=true) transforms all four
// 2nd-order forward-dynamics tensors to the mjx frame (explicit-analytic SO transform,
// contravector output-map). Register-heavy; post-launch error check surfaces a silent
// launch-config failure as rc!=0.
extern "C" int grid_rbd_fdsva_so_mujoco(const T* q, const T* qd, const T* u, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, u, batch, grid::NUM_JOINTS);
    grid::fdsva_so<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_FDSVA_SO>(), g_streams);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_df2,
                batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_FDSVA_SO


// ────────────────────────────────────────────────────────────────────────────
// Centroidal / energy / general-frame kinematics surface (F2 binding layer)
// ────────────────────────────────────────────────────────────────────────────
//
// Thin host-path wrappers over grid::{com,ccrba,energy,generalized_gravity,
// nonlinear_effects,frame_jacobian,frame_jacobian_dot,osc_inertia}. Each stages
// the (q[,qd]) inputs into the singleton gridData, launches the host wrapper
// (which owns the H->D / D->H staging + its own kernel launch), then copies the
// per-timestep output buffer out. Output layouts (per timestep) mirror the
// gridData buffers documented in GRiDCodeGenerator/algorithms/_centroidal.py:
//   com                 : 3 + 3*NUM_VEL  ([p_com(3); J_com(3 x NV, col-major)])
//   ccrba               : 6*NUM_VEL + 6  ([A(6 x NV, col-major); h(6)], Pinocchio [lin;ang]@CoM)
//   energy              : 3              ([KE, PE, KE+PE])
//   generalized_gravity : NUM_VEL       (g(q) = RNEA(q,0,0))
//   nonlinear_effects   : NUM_VEL       (c(q,qd) = RNEA(q,qd,0))
//   frame_jacobian      : 6*NUM_VEL     (6 x NV col-major, [linear;angular], target frame)
//   frame_jacobian_dot  : 6*NUM_VEL     (time-derivative of frame_jacobian along qd)
//   osc_inertia         : 36            (6x6 task inertia Lambda = (J Minv J^T)^-1)
//
// frame_jacobian / frame_jacobian_dot take the target frame at RUNTIME:
// (int target_jid, int reference_frame) trail the C-ABI signature; pass -1 for
// either to fall back to the codegen leaf-EE / LOCAL_WORLD_ALIGNED default baked
// into the host wrapper. osc_inertia still bakes its frame at codegen time.

// com uses the COMPRESSED input layout (h_q / d_q, stride NUM_JOINTS), unlike
// the other surfaces which read the [q,qd,u]-interleaved h_q_qd_u.
static inline void pack_q(const T* q, int batch, int num_joints) {
    std::memcpy(g_data->h_q, q, (size_t)batch * num_joints * sizeof(T));
}

// com(q) -> [p_com(3); J_com(3 x NV)] per timestep, total 3 + 3*NUM_VEL floats.
// Gated on GRID_HAS_COM: com/ccrba/energy ARE emitted for mimic robots (the
// centroidal_inner Jacobian fold is alpha-reduced; validated on fr3/h1_2 by
// cuda_centroidal_mimic_smoke_runner.cu). rc=3 only when a reduced codegen
// profile didn't generate com for this robot.
extern "C" int grid_rbd_com(const T* q, T* out, int batch) {
#ifdef GRID_HAS_COM
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q(q, batch, grid::NUM_JOINTS);
    grid::com<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_com, (size_t)batch * (3 + 3 * grid::NUM_VEL) * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch;
    return 3;  // com not generated for this robot (reduced codegen profile)
#endif
}

// ccrba(q, qd) -> [A(6 x NV); h(6)] per timestep, total 6*NUM_VEL + 6 floats.
// Gated on GRID_HAS_CCRBA (emitted for mimic too, alpha-folded); rc=3 only when a
// reduced codegen profile didn't generate ccrba for this robot.
extern "C" int grid_rbd_ccrba(const T* q, const T* qd, T* out, int batch) {
#ifdef GRID_HAS_CCRBA
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::ccrba<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1),
        grid_clamp_threads_for(grid::ccrba_kernel<T>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>()), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_ccrba, (size_t)batch * (6 * grid::NUM_VEL + 6) * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch;
    return 3;  // ccrba not generated for this robot (reduced codegen profile)
#endif
}

// energy(q, qd) -> [KE, PE, KE+PE] per timestep, total 3 floats. Takes gravity.
// Gated on GRID_HAS_ENERGY (emitted for mimic too, alpha-folded); rc=3 only when a
// reduced codegen profile didn't generate energy for this robot.
extern "C" int grid_rbd_energy(const T* q, const T* qd, T* out, int batch, T gravity) {
#ifdef GRID_HAS_ENERGY
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::energy<T>(g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_energy, (size_t)batch * 3 * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch;
    return 3;  // energy not generated for this robot (reduced codegen profile)
#endif
}

#if defined(GRID_HAS_COM) && defined(GRID_RBD_WITH_MUJOCO)
// MuJoCo-convention com(q) -> [p_com(3); J_com(3 x NV)] per timestep. p_com is
// INVARIANT; the J_com columns are reframed by the kernel (MUJOCO_OUTPUT=true): q
// is MuJoCo-native (quat wxyz) and the kernel reorders the quaternion + applies the
// column reframe before saving, so NO host pre/post-process is needed.
extern "C" int grid_rbd_com_mujoco(const T* q, T* out, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q(q, batch, grid::NUM_JOINTS);
    grid::com<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
              /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_com, (size_t)batch * (3 + 3 * grid::NUM_VEL) * sizeof(T));
    return 0;
}
#endif  // GRID_HAS_COM && GRID_RBD_WITH_MUJOCO

#if defined(GRID_HAS_CCRBA) && defined(GRID_RBD_WITH_MUJOCO)
// MuJoCo-convention ccrba(q, qd) -> [A(6 x NV); h(6)] per timestep. The centroidal
// momentum h is INVARIANT; the A columns are reframed by the kernel
// (MUJOCO_OUTPUT=true). q/qd raw mjx in (kernel reorders the quaternion + reframes
// qd), so NO host pre/post-process is needed.
extern "C" int grid_rbd_ccrba_mujoco(const T* q, const T* qd, T* out, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::ccrba<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_ccrba, (size_t)batch * (6 * grid::NUM_VEL + 6) * sizeof(T));
    return 0;
}
#endif  // GRID_HAS_CCRBA && GRID_RBD_WITH_MUJOCO

#if defined(GRID_HAS_ENERGY) && defined(GRID_RBD_WITH_MUJOCO)
// MuJoCo-convention energy(q, qd) -> [KE, PE, KE+PE] per timestep. The energies are
// frame-INVARIANT; the kernel (MUJOCO_OUTPUT=true) just converts the mjx-native
// inputs (quaternion reorder + qd reframe) so the energy is built correctly. Output
// is byte-equal to feeding the pin kernel the pin-converted q/qd.
extern "C" int grid_rbd_energy_mujoco(const T* q, const T* qd, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::energy<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                 /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_energy, (size_t)batch * 3 * sizeof(T));
    return 0;
}
#endif  // GRID_HAS_ENERGY && GRID_RBD_WITH_MUJOCO

// generalized_gravity(q) -> g(q) = RNEA(q,0,0) per timestep, NUM_VEL floats. Takes gravity.
extern "C" int grid_rbd_generalized_gravity(const T* q, T* out, int batch, T gravity) {
#if GRID_HAS_GENERALIZED_GRAVITY
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);  // qd unused (zeroed internally)
    grid::generalized_gravity<T>(g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_c, (size_t)batch * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)gravity;
    return 3;  // generalized_gravity not built into this .so (subset profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_GENERALIZED_GRAVITY
// MuJoCo-convention generalized_gravity(q) -> g(q) (floating base only). q is raw
// mjx (kernel reorders the quaternion); the kernel (MUJOCO_OUTPUT=true) base-rotates
// the gravity output so the returned g is mjx-frame. Output is NUM_VEL invariant-shaped.
extern "C" int grid_rbd_generalized_gravity_mujoco(const T* q, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);  // qd unused (zeroed internally)
    grid::generalized_gravity<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                              /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_c, (size_t)batch * grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_GENERALIZED_GRAVITY

// nonlinear_effects(q, qd) -> c(q,qd) = RNEA(q,qd,0) per timestep, NUM_VEL floats. Takes gravity.
extern "C" int grid_rbd_nonlinear_effects(const T* q, const T* qd, T* out, int batch, T gravity) {
#if GRID_HAS_NONLINEAR_EFFECTS
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::nonlinear_effects<T>(g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_c, (size_t)batch * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch; (void)gravity;
    return 3;  // nonlinear_effects not built into this .so (subset profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_NONLINEAR_EFFECTS
// MuJoCo-convention nonlinear_effects(q, qd) -> c(q,qd) (floating base only). q is
// raw mjx (kernel reorders the quaternion). The kernel (MUJOCO_OUTPUT=true) injects
// the accel-couple delta_a (base-linear = -(omega x v_lin)) via a zeroed s_qdd then
// base-rotates the bias output, so the returned c is the mjx-frame qfrc_bias.
extern "C" int grid_rbd_nonlinear_effects_mujoco(const T* q, const T* qd, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::nonlinear_effects<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                            /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_c, (size_t)batch * grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_NONLINEAR_EFFECTS

// coriolis_matrix(q, qd) -> nv x nv Coriolis matrix C(q,qd), row-major
// (C[row*nv + col]). Always emitted with the "all" profile (mimic-safe:
// alpha-folded column assembly), so it is bound UNGATED like com/ccrba.
extern "C" int grid_rbd_coriolis_matrix(const T* q, const T* qd, T* out, int batch, T gravity) {
#if GRID_HAS_CORIOLIS_MATRIX
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::coriolis_matrix<T>(g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_coriolis, (size_t)batch * grid::NUM_VEL * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch; (void)gravity;
    return 3;  // coriolis_matrix not built into this .so (subset profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_CORIOLIS_MATRIX
// MuJoCo-convention Coriolis matrix (floating base only): C_mjx = G C_pin G^T, a
// congruence baked into the kernel (MUJOCO_OUTPUT=true). q/qd raw mjx in (kernel
// reorders the quaternion + reframes qd), mjx-frame C out (nv x nv row-major).
extern "C" int grid_rbd_coriolis_matrix_mujoco(const T* q, const T* qd, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::coriolis_matrix<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                          /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_coriolis, (size_t)batch * grid::NUM_VEL * grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_CORIOLIS_MATRIX

// kinetic_energy_regressor(q, qd) -> length 10*NUM_BODIES regressor y_KE
// (KE = y_KE . pi). Always emitted with the "all" profile (mimic-safe), ungated.
extern "C" int grid_rbd_kinetic_energy_regressor(const T* q, const T* qd, T* out, int batch, T gravity) {
#if GRID_HAS_KINETIC_ENERGY_REGRESSOR
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::kinetic_energy_regressor<T>(g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_ke_regressor, (size_t)batch * 10 * grid::NUM_BODIES * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch; (void)gravity;
    return 3;  // kinetic_energy_regressor not built into this .so (subset profile)
#endif
}

// potential_energy_regressor(q) -> length 10*NUM_BODIES regressor y_PE
// (PE = y_PE . pi). Always emitted with the "all" profile (mimic-safe), ungated.
// Reads the COMPRESSED input layout (h_q / d_q) like com.
extern "C" int grid_rbd_potential_energy_regressor(const T* q, T* out, int batch, T gravity) {
#if GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q(q, batch, grid::NUM_JOINTS);
    grid::potential_energy_regressor<T>(g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_pe_regressor, (size_t)batch * 10 * grid::NUM_BODIES * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)gravity;
    return 3;  // potential_energy_regressor not built into this .so (subset profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_KINETIC_ENERGY_REGRESSOR
// MuJoCo-convention kinetic_energy_regressor(q, qd) -> length 10*NUM_BODIES y_KE.
// The regressor is frame-INVARIANT; the kernel (MUJOCO_OUTPUT=true) only converts
// the mjx-native inputs (quaternion reorder + qd reframe). Output is byte-equal to
// feeding the pin kernel the pin-converted q/qd. (The MUJOCO_OUTPUT instantiation
// only exists for floating, hence the GRID_RBD_WITH_MUJOCO gate.)
extern "C" int grid_rbd_kinetic_energy_regressor_mujoco(const T* q, const T* qd, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::kinetic_energy_regressor<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                                   /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_ke_regressor, (size_t)batch * 10 * grid::NUM_BODIES * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_KINETIC_ENERGY_REGRESSOR

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
// MuJoCo-convention potential_energy_regressor(q) -> length 10*NUM_BODIES y_PE.
// Frame-INVARIANT; the kernel (MUJOCO_OUTPUT=true) only converts the mjx-native q
// (quaternion reorder). Output byte-equal to feeding the pin kernel pin-converted q.
extern "C" int grid_rbd_potential_energy_regressor_mujoco(const T* q, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q(q, batch, grid::NUM_JOINTS);
    grid::potential_energy_regressor<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                                     /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_pe_regressor, (size_t)batch * 10 * grid::NUM_BODIES * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_POTENTIAL_ENERGY_REGRESSOR

// dccrba(q) -> 6*NUM_VEL*NUM_VEL dCCRBA tensor dA/dq (per timestep, as the kernel
// writes it). Reads the COMPRESSED input layout (h_q / d_q). Gated on
// GRID_HAS_DCCRBA: dccrba IS emitted for mimic robots (alpha-folded; validated on
// fr3/h1_2 by test_cuda_dccrba.py), so rc=3 only when a reduced codegen profile
// didn't generate dccrba. For big floating robots whose kernel arena overflows the
// smem cap the host wrapper's grid_check_dynamic_shared_memory_bytes raises a clear
// rc!=0 at launch (the big-floating spill ladder de-gated g1/h1_2; commit c4b3900).
extern "C" int grid_rbd_dccrba(const T* q, T* out, int batch) {
#ifdef GRID_HAS_DCCRBA
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q(q, batch, grid::NUM_JOINTS);
    grid::dccrba<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1),
        grid_clamp_threads_for(grid::dccrba_kernel<T>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>()), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_dccrba, (size_t)batch * 6 * grid::NUM_VEL * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch;
    return 3;  // dccrba not generated for this robot (reduced codegen profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && defined(GRID_HAS_DCCRBA)
// MuJoCo-convention dccrba(q) -> 6*NV*NV dA/dq tensor (floating base only). q is raw
// mjx (kernel reorders the quaternion); the kernel (MUJOCO_OUTPUT=true) double-reframes
// the qd-column and q-tangent indices by G^{-1} and adds the base-rotation frame term
// (using the in-kernel CMM value) before saving.
extern "C" int grid_rbd_dccrba_mujoco(const T* q, T* out, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q(q, batch, grid::NUM_JOINTS);
    grid::dccrba<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                 /*MUJOCO_OUTPUT=*/true>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_dccrba, (size_t)batch * 6 * grid::NUM_VEL * grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_DCCRBA

// cmm_time_variation(q, qd) -> 6*NUM_VEL centroidal-momentum-matrix time
// variation Adot (per timestep). Gated on GRID_HAS_CMM_TIME_VARIATION (emitted for
// mimic too, alpha-folded); returns rc=3 only when a reduced profile didn't generate it.
extern "C" int grid_rbd_cmm_time_variation(const T* q, const T* qd, T* out, int batch) {
#ifdef GRID_HAS_CMM_TIME_VARIATION
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::cmm_time_variation<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1),
        grid_clamp_threads_for(grid::cmm_time_variation_kernel<T>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>()), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_cmm_time_variation, (size_t)batch * 6 * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch;
    return 3;  // cmm_time_variation not generated for this robot (mimic)
#endif
}

#if defined(GRID_HAS_CMM_TIME_VARIATION) && defined(GRID_RBD_WITH_MUJOCO)
// MuJoCo-convention cmm_time_variation(q, qd) -> 6*NUM_VEL Adot (per timestep).
// Column-reframe: q/qd raw mjx in (kernel reorders the quaternion + reframes qd)
// and the kernel (MUJOCO_OUTPUT=true) reframes the Adot columns before saving.
extern "C" int grid_rbd_cmm_time_variation_mujoco(const T* q, const T* qd, T* out, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::cmm_time_variation<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                             /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_cmm_time_variation, (size_t)batch * 6 * grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_HAS_CMM_TIME_VARIATION && GRID_RBD_WITH_MUJOCO

// frame_jacobian(q) -> 6 x NUM_VEL geometric Jacobian (col-major, [linear;angular])
// at the leaf-EE frame, LOCAL_WORLD_ALIGNED. Gated on GRID_HAS_FRAME_JACOBIAN
// (the frame_jacobian family is opt-in codegen; only present when requested).
extern "C" int grid_rbd_frame_jacobian(const T* q, T* out, int batch,
                                       int target_jid, int reference_frame) {
#ifdef GRID_HAS_FRAME_JACOBIAN
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);
    // -1 per arg => "use default" (leaf-EE / LWA); the host resolves each
    // INDEPENDENTLY, so a default target with an explicit frame is honored.
    grid::frame_jacobian<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams,
                            target_jid, reference_frame);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_frame_jacobian, (size_t)batch * 6 * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)target_jid; (void)reference_frame;
    return 3;  // frame_jacobian not generated for this .so
#endif
}

// frame_jacobian_dot(q, qd) -> d/dt of the leaf-EE frame Jacobian along v=qd,
// 6 x NUM_VEL (col-major, [linear;angular]). Gated on GRID_HAS_FRAME_JACOBIAN.
extern "C" int grid_rbd_frame_jacobian_dot(const T* q, const T* qd, T* out, int batch,
                                           int target_jid, int reference_frame) {
#ifdef GRID_HAS_FRAME_JACOBIAN
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    // -1 per arg => "use default" (leaf-EE / LWA), resolved INDEPENDENTLY by the
    // host so a default target with an explicit frame is honored.
    grid::frame_jacobian_dot<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams,
                                target_jid, reference_frame);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_frame_jacobian_dot, (size_t)batch * 6 * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch; (void)target_jid; (void)reference_frame;
    return 3;
#endif
}

// osc_inertia(q) -> 6x6 operational-space (task) inertia Lambda = (J Minv J^T)^-1
// at the leaf-EE frame (LWA), 36 floats per timestep. Gated on GRID_HAS_FRAME_JACOBIAN.
extern "C" int grid_rbd_osc_inertia(const T* q, T* out, int batch) {
#ifdef GRID_HAS_FRAME_JACOBIAN
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);
    grid::osc_inertia<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_osc_inertia, (size_t)batch * 36 * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch;
    return 3;
#endif
}

#if defined(GRID_HAS_FRAME_JACOBIAN) && defined(GRID_RBD_WITH_MUJOCO)
// MuJoCo-convention frame_jacobian family (floating base only). The geometric
// Jacobian / its time-derivative are column-reframed J_mjx = J_pin G^{-1} in the
// kernel (MUJOCO_OUTPUT=true). osc_inertia's value Lambda is frame-INVARIANT, but
// its q input still needs the quaternion reordered (wxyz->xyzw) so the internal
// J/Minv build correctly — the mjx kernel does that, so a raw-mjx q is handled here
// rather than silently mis-built by the pin kernel. All take raw mjx inputs.
extern "C" int grid_rbd_frame_jacobian_mujoco(const T* q, T* out, int batch,
                                              int target_jid, int reference_frame) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);
    grid::frame_jacobian<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                         /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams,
        target_jid, reference_frame);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_frame_jacobian, (size_t)batch * 6 * grid::NUM_VEL * sizeof(T));
    return 0;
}

extern "C" int grid_rbd_frame_jacobian_dot_mujoco(const T* q, const T* qd, T* out, int batch,
                                                  int target_jid, int reference_frame) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::frame_jacobian_dot<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                             /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams,
        target_jid, reference_frame);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_frame_jacobian_dot, (size_t)batch * 6 * grid::NUM_VEL * sizeof(T));
    return 0;
}

extern "C" int grid_rbd_osc_inertia_mujoco(const T* q, T* out, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);
    grid::osc_inertia<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL,
                      /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_osc_inertia, (size_t)batch * 36 * sizeof(T));
    return 0;
}
#endif  // GRID_HAS_FRAME_JACOBIAN && GRID_RBD_WITH_MUJOCO

// end_effector_pose_runtime(q) -> 6-vector [xyz; rpy] of target_jid at a runtime
// offset point in the target frame. target_jid<0 => leaf-EE default; offset may
// be nullptr (=> frame origin {0,0,0}). Gated on GRID_HAS_END_EFFECTOR_POSE_RUNTIME
// (opt-in codegen). Single-target like frame_jacobian; the Python list API loops it.
extern "C" int grid_rbd_end_effector_pose_runtime(const T* q, T* out, int batch,
                                                  int target_jid, const T* offset) {
#ifdef GRID_HAS_END_EFFECTOR_POSE_RUNTIME
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);  // qd/u unused
    // stage the runtime offset (frame origin when offset==nullptr).
    // offset is now the 4x4 col-major SE(3) tool/tip transform (16 floats); identity => frame origin.
    T Xtool[16] = {static_cast<T>(1),0,0,0, 0,static_cast<T>(1),0,0,
                   0,0,static_cast<T>(1),0, 0,0,0,static_cast<T>(1)};
    if (offset) { for (int i = 0; i < 16; ++i) Xtool[i] = offset[i]; }
    if (cudaMemcpy(g_data->d_eepose_runtime_offset, Xtool, 16*sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 101;
    grid::end_effector_pose_runtime<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(),
                                       g_streams, target_jid);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_eePose, (size_t)batch * 6 * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)target_jid; (void)offset;
    return 3;
#endif
}

// end_effector_pose_gradient_runtime(q) -> 6 x NUM_VEL = d[xyz; rpy]/dv of
// target_jid at a runtime offset point (col-major). Same target/offset conventions
// as grid_rbd_end_effector_pose_runtime. Gated on
// GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME.
extern "C" int grid_rbd_end_effector_pose_gradient_runtime(const T* q, T* out, int batch,
                                                           int target_jid, const T* offset) {
#ifdef GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);  // qd/u unused
    // offset is now the 4x4 col-major SE(3) tool/tip transform (16 floats); identity => frame origin.
    T Xtool[16] = {static_cast<T>(1),0,0,0, 0,static_cast<T>(1),0,0,
                   0,0,static_cast<T>(1),0, 0,0,0,static_cast<T>(1)};
    if (offset) { for (int i = 0; i < 16; ++i) Xtool[i] = offset[i]; }
    if (cudaMemcpy(g_data->d_eepose_runtime_offset, Xtool, 16*sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 101;
    grid::end_effector_pose_gradient_runtime<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1),
                                                grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams, target_jid);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_eePoseGrad, (size_t)batch * 6 * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)target_jid; (void)offset;
    return 3;
#endif
}

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention end_effector_pose_runtime (floating base only). The kernel
// input-converts q (quat reorder) on load; the pose VALUE is frame-INVARIANT (the
// 6-vector [xyz; rpy] is the same world frame), so this matches the pin pose with
// the mjx-reordered quaternion. Baked via the MUJOCO_OUTPUT=true host/kernel flag.
extern "C" int grid_rbd_end_effector_pose_runtime_mujoco(const T* q, T* out, int batch,
                                                         int target_jid, const T* offset) {
#ifdef GRID_HAS_END_EFFECTOR_POSE_RUNTIME
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);  // qd/u unused
    // offset is now the 4x4 col-major SE(3) tool/tip transform (16 floats); identity => frame origin.
    T Xtool[16] = {static_cast<T>(1),0,0,0, 0,static_cast<T>(1),0,0,
                   0,0,static_cast<T>(1),0, 0,0,0,static_cast<T>(1)};
    if (offset) { for (int i = 0; i < 16; ++i) Xtool[i] = offset[i]; }
    if (cudaMemcpy(g_data->d_eepose_runtime_offset, Xtool, 16*sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 101;
    grid::end_effector_pose_runtime<T, /*USE_COMPRESSED_MEM=*/false, grid::GRID_DATA_ALL,
                                    /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams, target_jid);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_eePose, (size_t)batch * 6 * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)target_jid; (void)offset;
    return 3;
#endif
}

// MuJoCo-convention end_effector_pose_gradient_runtime (floating base only). The
// pose value is invariant, but the gradient is COLUMN-reframed: the base-linear
// columns reframe by R^T (mjx base-linear velocity is global). Baked via
// MUJOCO_OUTPUT=true. Output 6 x NUM_VEL col-major, like the pin variant.
extern "C" int grid_rbd_end_effector_pose_gradient_runtime_mujoco(const T* q, T* out, int batch,
                                                                  int target_jid, const T* offset) {
#ifdef GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);  // qd/u unused
    // offset is now the 4x4 col-major SE(3) tool/tip transform (16 floats); identity => frame origin.
    T Xtool[16] = {static_cast<T>(1),0,0,0, 0,static_cast<T>(1),0,0,
                   0,0,static_cast<T>(1),0, 0,0,0,static_cast<T>(1)};
    if (offset) { for (int i = 0; i < 16; ++i) Xtool[i] = offset[i]; }
    if (cudaMemcpy(g_data->d_eepose_runtime_offset, Xtool, 16*sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 101;
    grid::end_effector_pose_gradient_runtime<T, /*USE_COMPRESSED_MEM=*/false, grid::GRID_DATA_ALL,
                                             /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), g_streams, target_jid);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_eePoseGrad, (size_t)batch * 6 * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)target_jid; (void)offset;
    return 3;
#endif
}
#endif  // GRID_RBD_WITH_MUJOCO


// ────────────────────────────────────────────────────────────────────────────
// Runtime tool-tip contact wrench -> joint-local f_ext (welded-tool tip forces)
// ────────────────────────────────────────────────────────────────────────────
//
// tool_fext(q, wrench, jid, rc) -> (batch, 6*NUM_BODIES) joint-local f_ext array
// [angular;linear] per body, ready to feed straight to inverse_dynamics(f_ext=...)
// / aba(f_ext=...). `wrench` is a world-aligned [n_w; f_w] 6-vector per timestep at
// the runtime tool tip (body `jid`, local offset `rc`). Gated on GRID_HAS_CONTACT_RUNTIME
// (register with enable_tool=True). The device fn owns the FK arena in dynamic smem;
// we size it from the (>= this arena) EE-pose / f_ext_gradient macros and pass
// d_workspace so it also works at spilling tiers on big robots.
#ifdef GRID_HAS_CONTACT_RUNTIME
__global__ void grid_rbd_tool_fext_kernel(const T* d_q, int stride_q, int jid,
                                          const T* d_rc, const T* d_wrench,
                                          const grid::robotModel<T>* d_robotModel,
                                          T* d_workspace, T* d_out) {
    const int k = blockIdx.x;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    __shared__ T s_fext[6 * grid::NUM_BODIES];
    __shared__ T s_rc[3];
    if (tid < 3) s_rc[tid] = d_rc[tid];
    __syncthreads();
    grid::f_ext_body_runtime_device<T>(s_fext, &d_wrench[k * 6], jid, s_rc,
                                       &d_q[k * stride_q], d_robotModel, d_workspace);
    __syncthreads();
    for (int i = tid; i < 6 * grid::NUM_BODIES; i += nth)
        d_out[k * 6 * grid::NUM_BODIES + i] = s_fext[i];
}

extern "C" int grid_rbd_tool_fext(const T* q, const T* wrench, int jid, const T* rc,
                                  T* out, int batch) {
    if (!g_data) { int rc0 = grid_rbd_init(); if (rc0) return rc0; }
    if (batch > kMaxBatch) return 2;
    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, q, nullptr, batch, nj);   // q at offset 0, stride 3*nj
    const int stride_q = 3 * nj;
    if (cudaMemcpy(g_data->d_q_qd_u, g_data->h_q_qd_u,
                   (size_t)batch * stride_q * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) return 5;
    T *d_wrench = nullptr, *d_rc = nullptr;
    if (cudaMalloc(&d_wrench, (size_t)6 * batch * sizeof(T)) != cudaSuccess) return 6;
    if (cudaMalloc(&d_rc, 3 * sizeof(T)) != cudaSuccess) { cudaFree(d_wrench); return 6; }
    cudaMemcpy(d_wrench, wrench, (size_t)6 * batch * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rc, rc, 3 * sizeof(T), cudaMemcpyHostToDevice);
    size_t smem = grid::F_EXT_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t s2 = grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>();
    if (s2 > smem) smem = s2;
    smem += 4096;
    cudaFuncSetAttribute(grid_rbd_tool_fext_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    grid_rbd_tool_fext_kernel<<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), smem>>>(
        g_data->d_q_qd_u, stride_q, jid, d_rc, d_wrench, g_robot,
        reinterpret_cast<T*>(g_data->d_workspace), g_data->d_f_ext);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    cudaFree(d_wrench); cudaFree(d_rc);
    if (e != cudaSuccess) return 100 + (int)e;
    if (cudaMemcpy(out, g_data->d_f_ext, (size_t)6 * grid::NUM_BODIES * batch * sizeof(T),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    // d_f_ext is the shared f_ext buffer; re-zero it so a later NO-f_ext dynamics call
    // (which does not reset it) is not polluted by the tool wrench we just wrote.
    cudaMemset(g_data->d_f_ext, 0, (size_t)6 * grid::NUM_BODIES * kMaxBatch * sizeof(T));
    return 0;
}
#else
extern "C" int grid_rbd_tool_fext(const T*, const T*, int, const T*, T*, int) { return 3; }
#endif


// ────────────────────────────────────────────────────────────────────────────
// Time integrator (value + gradient)
// ────────────────────────────────────────────────────────────────────────────
//
// dt is a runtime float; gravity is the signed gravitational acceleration (default -9.81). The
// integrator type is selected at call time via an int code (0=EULER,
// 1=SEMI_IMPLICIT_EULER, 2=MIDPOINT, 3=RK3, 4=RK4) dispatched onto the
// compile-time `IntegratorType IT` template. x_kp1 is size (NUM_POS + NUM_VEL)
// per timestep; dAB is (2*NUM_VEL) x (3*NUM_VEL) per timestep (column-major).

// host-path launchers (call the host wrappers, which stage memory + own streams).
// Subset-build: these template BODIES name grid::integrator{,_gradient} directly, so
// they must be `#if`-guarded on the same macro as their caller body — a subset header
// that omits the integrator emits NO grid::integrator symbol at all, and an
// uninstantiated template that references a non-existent qualified name is still a hard
// name-lookup error at parse time (not just a deferred instantiation failure). With the
// guard, a subset .so simply omits the launcher; the caller body is rc=3-stubbed in turn.
#if GRID_HAS_INTEGRATOR
template <grid::IntegratorType IT>
static void launch_integrator_host(int batch, T gravity, T dt) {
// signature switch (same rule as fdsva_so): floating builds carry MUJOCO_OUTPUT.
// Tier must match the per-algo autotuned thread count (see idsva_so above).
#if defined(GRID_RBD_SIG_MJX_INTEGRATOR)
    grid::integrator<T, IT, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER>(
#else
    grid::integrator<T, IT, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER>(
#endif
        g_data, g_robot, /*gravity=*/gravity,
        dt, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_INTEGRATOR>(), g_streams);
}
#ifdef GRID_RBD_WITH_MUJOCO
template <grid::IntegratorType IT>
static void launch_integrator_host_mujoco(int batch, T gravity, T dt) {
    grid::integrator<T, IT, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER>(
        g_data, g_robot, /*gravity=*/gravity,
        dt, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_INTEGRATOR>(), g_streams);
}
#endif
#endif  // GRID_HAS_INTEGRATOR
#if GRID_HAS_INTEGRATOR_GRADIENT
template <grid::IntegratorType IT>
static void launch_integrator_grad_host(int batch, T gravity, T dt) {
    grid::integrator_gradient<T, IT>(g_data, g_robot, /*gravity=*/gravity,
                                     dt, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_INTEGRATOR_GRADIENT>(), g_streams);
}
#ifdef GRID_RBD_WITH_MUJOCO
template <grid::IntegratorType IT>
static void launch_integrator_grad_host_mujoco(int batch, T gravity, T dt) {
    grid::integrator_gradient<T, IT, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR_GRADIENT>::TIER>(
        g_data, g_robot, /*gravity=*/gravity,
        dt, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads<grid::GRID_ALGO_INTEGRATOR_GRADIENT>(), g_streams);
}
#endif
#endif  // GRID_HAS_INTEGRATOR_GRADIENT

#define GRID_RBD_IT_DISPATCH(it_code, FN, ...)                                   \
    switch (it_code) {                                                           \
        case 0: FN<grid::IntegratorType::EULER>(__VA_ARGS__); break;             \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER>(__VA_ARGS__); break;\
        case 2: FN<grid::IntegratorType::MIDPOINT>(__VA_ARGS__); break;          \
        case 3: FN<grid::IntegratorType::RK3>(__VA_ARGS__); break;               \
        case 4: FN<grid::IntegratorType::RK4>(__VA_ARGS__); break;               \
        default: return 3;                                                       \
    }

// plant_step_hessian only supports EULER / SI-EULER (the composed device fn
// static_asserts MIDPOINT/RK out — instantiating those cases would fail to
// COMPILE, so this dispatch never names them). Other IT codes return rc=3.
#define GRID_RBD_IT_DISPATCH_HESSIAN(it_code, FN, ...)                           \
    switch (it_code) {                                                           \
        case 0: FN<grid::IntegratorType::EULER>(__VA_ARGS__); break;             \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER>(__VA_ARGS__); break;\
        default: return 3;                                                       \
    }

// integrator(q, qd, u, dt, it) → x_kp1  (size NUM_POS + NUM_VEL per timestep)
extern "C" int grid_rbd_integrator(
    const T* q, const T* qd, const T* u,
    T* x_kp1_out, int batch, T gravity, T dt, int it)
{
#if GRID_HAS_INTEGRATOR
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);

    GRID_RBD_IT_DISPATCH(it, launch_integrator_host, batch, gravity, dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(x_kp1_out, g_data->h_x_kp1,
                batch * (grid::NUM_POS + grid::NUM_VEL) * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)u; (void)x_kp1_out; (void)batch; (void)gravity; (void)dt; (void)it;
    return 3;  // integrator not built into this .so (subset codegen profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_INTEGRATOR
// MuJoCo-convention integrator (floating base only): the free-joint base POSITION
// takes a GLOBAL additive step (mjx retract) instead of pin's SE(3) V(phi); the
// base quaternion + joints integrate normally. q/qd raw mjx in (q wxyz, qd global),
// x_kp1 raw mjx out (q wxyz). Baked into the kernel (MUJOCO_OUTPUT=true).
extern "C" int grid_rbd_integrator_mujoco(
    const T* q, const T* qd, const T* u,
    T* x_kp1_out, int batch, T gravity, T dt, int it)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);

    GRID_RBD_IT_DISPATCH(it, launch_integrator_host_mujoco, batch, gravity, dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(x_kp1_out, g_data->h_x_kp1,
                batch * (grid::NUM_POS + grid::NUM_VEL) * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_INTEGRATOR

// integrator_gradient(q, qd, u, dt, it) → dAB  (2*NV x 3*NV per timestep)
extern "C" int grid_rbd_integrator_gradient(
    const T* q, const T* qd, const T* u,
    T* dAB_out, int batch, T gravity, T dt, int it)
{
#if GRID_HAS_INTEGRATOR_GRADIENT
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);

    GRID_RBD_IT_DISPATCH(it, launch_integrator_grad_host, batch, gravity, dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }

    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;

    const int nv = grid::NUM_VEL;
    std::memcpy(dAB_out, g_data->h_dAB,
                batch * (2 * nv) * (3 * nv) * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)u; (void)dAB_out; (void)batch; (void)gravity; (void)dt; (void)it;
    return 3;  // integrator_gradient not built into this .so (subset codegen profile)
#endif
}

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_INTEGRATOR_GRADIENT
// MuJoCo-convention integrator_gradient(q, qd, u, dt, it) -> dAB (2NV x 3NV) (floating
// base only). q/qd/u raw mjx (kernel input-converts); the kernel (MUJOCO_OUTPUT=true)
// transforms the discrete state-transition Jacobian to the mjx tangent (global-add
// retract rows + G velocity reframe + input-conversion column couplings). EULER/SI-EULER
// only (multistage static_asserts out). Register-heavy; post-launch error check.
extern "C" int grid_rbd_integrator_gradient_mujoco(
    const T* q, const T* qd, const T* u,
    T* dAB_out, int batch, T gravity, T dt, int it)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, u, batch, grid::NUM_JOINTS);
    GRID_RBD_IT_DISPATCH_HESSIAN(it, launch_integrator_grad_host_mujoco, batch, gravity, dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    const int nv = grid::NUM_VEL;
    std::memcpy(dAB_out, g_data->h_dAB,
                batch * (2 * nv) * (3 * nv) * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_INTEGRATOR_GRADIENT


// ────────────────────────────────────────────────────────────────────────────
// grid_plant C ABI (G1 binding layer)
// ────────────────────────────────────────────────────────────────────────────
//
// Exposes the `grid_plant::` device surface (cost / barrier / plant-step) as
// `extern "C"` host functions. Each launches one block per timestep against the
// per-timestep kernels emitted by GRiDCodeGenerator/algorithms/_plant.py.
//
// Plant-specific in/out buffers (desired states, weights, bounds, scalar
// outputs, dense hessians) are device-allocated lazily here, sized to
// kMaxBatch, and reused across calls (single-robot singleton, like g_data).
// Inputs are staged H->D, outputs copied D->H, with a device sync per call
// (the host-path ABI is synchronous, matching the other algorithms).

namespace {

// Lazily-allocated plant scratch (device). Sized to kMaxBatch * per-timestep.
struct PlantBuffers {
    // generic packed in/out (large enough for the biggest per-call need)
    T* d_in_a   = nullptr;   // var / x / u / q
    T* d_in_b   = nullptr;   // des / lower / p_des / qd / h_des
    T* d_in_c   = nullptr;   // weight / upper / W
    T* d_out    = nullptr;   // scalar cost (1 per timestep)
    T* d_grad   = nullptr;   // gradient / dAB ([A|B], 2*NV*3*NV)
    T* d_hess   = nullptr;   // dense hessian / hess-diagonal
    T* d_d2AB   = nullptr;   // plant_step_hessian s_d2AB (2*NV*3*NV*3*NV) — too
                             // big to share d_grad, so lazily allocated below.
    unsigned char* d_d2AB_workspace = nullptr;  // plant_step_hessian spill band
                             // (per-timestep s_d2AB + fdsva tensors + pool) for
                             // big robots whose arena overflows the smem cap.
    T* d_end_effector_pose  = nullptr;   // ee-pose / com / ccrba scratch (reused per call)
    T* d_end_effector_pose_gradient = nullptr;   // ee-jacobian scratch
    bool allocated = false;
};
static PlantBuffers g_plant;

static int plant_alloc() {
    if (g_plant.allocated) return 0;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    const int nee = grid::NUM_EES;
    const size_t B = (size_t)kMaxBatch;
    // size every buffer for the worst-case per-timestep footprint across calls.
    // >= nv, >= nq, >= 3; floored at 12 so momentum_cost can pack h_des(6)+W(6)
    // contiguously into a single d_in_c buffer even on a small (<12-DOF) robot.
    const size_t vec = std::max((size_t)nx, (size_t)12);
    // dense hessian (nx*nx) OR the plant_step_gradient dAB block (2*nv*3*nv =
    // 6*nv*nv). On a fixed base nx=2nv so nx*nx=4nv^2 < 6nv^2 — size by the max.
    const size_t mat = std::max((size_t)nx * (size_t)nx,
                                (size_t)(2 * nv) * (size_t)(3 * nv));
    // d_grad doubles as the plant_step x_kp1 (nx) reuse AND must NOT be confused
    // with the gradient size; the cost grads are <= nx, so vec covers it.
    // d_end_effector_pose holds the EE pose output (6*nee). The com (3+3*nv) /
    // ccrba (6*nv+6) terms are vestigial — the com/momentum cost kernels now lay
    // their centroidal scratch out of dynamic shared memory, not this buffer — but
    // kept as a conservative floor (harmless over-allocation).
    const size_t kin_scratch = std::max((size_t)(6 * nee),
                                std::max((size_t)(3 + 3 * nv), (size_t)(6 * nv + 6)));
    auto ok = [](cudaError_t e){ return e == cudaSuccess; };
    bool good = true;
    good &= ok(cudaMalloc(&g_plant.d_in_a,  B * vec * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_in_b,  B * vec * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_in_c,  B * vec * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_out,   B * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_grad,  B * mat * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_hess,  B * mat * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_end_effector_pose, B * kin_scratch * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_end_effector_pose_gradient, B * (size_t)(6 * nv * nee) * sizeof(T)));
    if (!good) return 1;
    g_plant.allocated = true;
    return 0;
}

}  // namespace

// quadratic_state_cost / quadratic_input_cost: value + grad + GN-diag hess.
// var/des/weight are (batch, N); out (batch); grad (batch, N); hess (batch, N*N).
template <bool STATE>
static int plant_quadratic_cost_impl(
    const T* var, const T* des, const T* w,
    T* out, T* grad, T* hess, int batch)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int N = STATE ? (grid::NUM_POS + grid::NUM_VEL) : grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, var, batch * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, des, batch * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, w,   batch * N * sizeof(T), cudaMemcpyHostToDevice);
    dim3 grid_dim = grid_rbd_grid_for(batch);
    if (STATE) {
        grid_plant::quadratic_state_cost_kernel<T><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, g_streams[0]>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
        { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    } else {
        grid_plant::quadratic_input_cost_kernel<T><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, g_streams[0]>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
        { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    }
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(out,  g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad, g_plant.d_grad, batch * N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, g_plant.d_hess, batch * N * N * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}

extern "C" int grid_plant_quadratic_state_cost(
    const T* x, const T* x_des, const T* Q, T* out, T* grad, T* hess, int batch) {
    return plant_quadratic_cost_impl<true>(x, x_des, Q, out, grad, hess, batch);
}
extern "C" int grid_plant_quadratic_input_cost(
    const T* u, const T* u_des, const T* R, T* out, T* grad, T* hess, int batch) {
    return plant_quadratic_cost_impl<false>(u, u_des, R, out, grad, hess, batch);
}

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention quadratic_state_cost (floating base only). x = [q(nq); qd(nv)]
// is mjx-native; the kernel input-converts the qd base-linear block (global->local)
// before differencing against the user (mjx-frame) x_des/Q, then reframes the
// qd-block grad (covector) + hess (congruence). The VALUE is convention-DEPENDENT.
// Baked via the MUJOCO_OUTPUT=true template flag. quadratic_state_cost is launched
// at the requested thread count and may be register-capped below it -> clamp + check.
extern "C" int grid_rbd_quadratic_state_cost_mujoco(
    const T* x, const T* x_des, const T* Q, T* out, T* grad, T* hess, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int N = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, x,     batch * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, x_des, batch * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, Q,     batch * N * sizeof(T), cudaMemcpyHostToDevice);
    dim3 grid_dim = grid_rbd_grid_for(batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::quadratic_state_cost_kernel<T, true>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::quadratic_state_cost_kernel<T, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr, 0, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(out,  g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad, g_plant.d_grad, batch * N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, g_plant.d_hess, batch * N * N * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO

// joint_{position,velocity,torque}_barrier: value + grad + hess-diagonal.
// var/lower/upper are (batch, N); out (batch); grad/hess_diag (batch, N).
enum class PlantBarrier { POSITION, VELOCITY, TORQUE };

static int plant_barrier_impl(
    PlantBarrier which,
    const T* var, const T* lower, const T* upper, float mu,
    T* out, T* grad, T* hess_diag, int batch)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int N = (which == PlantBarrier::POSITION) ? grid::NUM_POS : grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, var,   batch * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, lower, batch * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, upper, batch * N * sizeof(T), cudaMemcpyHostToDevice);
    dim3 grid_dim = grid_rbd_grid_for(batch);
    switch (which) {
        case PlantBarrier::POSITION:
            grid_plant::joint_position_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, g_streams[0]>>>(
                g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
                g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch); break;
        case PlantBarrier::VELOCITY:
            grid_plant::joint_velocity_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, g_streams[0]>>>(
                g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
                g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch); break;
        case PlantBarrier::TORQUE:
            grid_plant::joint_torque_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, g_streams[0]>>>(
                g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
                g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch); break;
    }
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(out,       g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad,      g_plant.d_grad, batch * N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess_diag, g_plant.d_hess, batch * N * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}

extern "C" int grid_plant_joint_position_barrier(
    const T* var, const T* lower, const T* upper, float mu,
    T* out, T* grad, T* hess_diag, int batch) {
    return plant_barrier_impl(PlantBarrier::POSITION, var, lower, upper, mu, out, grad, hess_diag, batch);
}
extern "C" int grid_plant_joint_velocity_barrier(
    const T* var, const T* lower, const T* upper, float mu,
    T* out, T* grad, T* hess_diag, int batch) {
    return plant_barrier_impl(PlantBarrier::VELOCITY, var, lower, upper, mu, out, grad, hess_diag, batch);
}
extern "C" int grid_plant_joint_torque_barrier(
    const T* var, const T* lower, const T* upper, float mu,
    T* out, T* grad, T* hess_diag, int batch) {
    return plant_barrier_impl(PlantBarrier::TORQUE, var, lower, upper, mu, out, grad, hess_diag, batch);
}

#ifdef GRID_PLANT_HAS_STEP
// plant_step: x_{k+1} = integrator(x_k, u_k, dt). x (batch, NX); u (batch, NV);
// out (batch, NX). Gated on GRID_PLANT_HAS_STEP (emitted only when the
// integrator algorithm is generated). Integrator type via the same int code.
template <grid::IntegratorType IT>
static void launch_plant_step(int batch, T gravity, T dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(batch);
    // clamp to the kernel's launch cap (register-heavy; else a silent launch failure
    // leaves the stale d_grad buffer -> looks like a no-op step). Mirrors the mjx twin.
    // smem opt-in too (arena can exceed 48 KB on a big floating robot -> rc=201).
    const size_t smem = grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_kernel<T, IT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_kernel<T, IT>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::plant_step_kernel<T, IT><<<grid_dim, thr,
        smem, g_streams[0]>>>(
            g_plant.d_grad /*reuse as d_x_kp1, size NX*/, g_plant.d_in_a, g_plant.d_in_b,
            nx, grid::NUM_VEL, g_robot, gravity, dt, batch);
}

extern "C" int grid_plant_step(
    const T* x, const T* u, T* x_kp1, int batch, float gravity, float dt, int it) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, x, batch * nx * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, u, batch * nv * sizeof(T), cudaMemcpyHostToDevice);
    GRID_RBD_IT_DISPATCH(it, launch_plant_step, batch, (T)gravity, (T)dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(x_kp1, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention plant_step (floating base only). x/u raw mjx; the kernel
// (MUJOCO_OUTPUT=true) input-converts the stacked state + does the global-add retract.
// EULER/SI-EULER only. Register-heavy -> launch-clamp + post-launch error check.
template <grid::IntegratorType IT>
static void launch_plant_step_mujoco(int batch, T gravity, T dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(batch);
    const size_t smem = grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_kernel<T, IT, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_kernel<T, IT, true>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::plant_step_kernel<T, IT, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr,
        smem, g_streams[0]>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, grid::NUM_VEL, g_robot, gravity, dt, batch);
}
extern "C" int grid_plant_step_mujoco(
    const T* x, const T* u, T* x_kp1, int batch, float gravity, float dt, int it) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, x, batch * nx * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, u, batch * grid::NUM_VEL * sizeof(T), cudaMemcpyHostToDevice);
    GRID_RBD_IT_DISPATCH_HESSIAN(it, launch_plant_step_mujoco, batch, (T)gravity, (T)dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(x_kp1, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_PLANT_HAS_STEP

#ifdef GRID_PLANT_HAS_EE_COST
// ee_pos_cost: value + grad over x=[q;qd] + GN hess_x. q (batch, NQ);
// p_des (batch, 3); W (batch, 3); out (batch); grad (batch, NX); hess (batch, NX*NX).
// Gated on GRID_PLANT_HAS_EE_COST (ee_pose + ee_pose_gradient generated).
extern "C" int grid_plant_ee_pos_cost(
    const T* q, const T* p_des, const T* W,
    T* out, T* grad, T* hess, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int nq = grid::NUM_POS;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, q,     batch * nq * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, p_des, batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, W,     batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    // The kernel internally calls ee-pose (value), ee-pose-gradient, and uses
    // the hessian-free GN J^T W J. The dynamic smem must cover the largest of
    // the device fns it invokes (pose-gradient dominates pose).
    size_t smem = grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(batch);
    // clamp to the kernel's launch cap + surface launch errors (else a register-OOR
    // launch is silently rejected -> stale-zero d_grad base block).
    dim3 thr = grid_clamp_threads_for(grid_plant::ee_pos_cost_kernel<T, 0>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::ee_pos_cost_kernel<T, 0><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose, g_plant.d_end_effector_pose_gradient, g_robot, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(out,  g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, g_plant.d_hess, batch * nx * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_PLANT_HAS_EE_COST

#ifdef GRID_PLANT_HAS_COM_COST
// com_cost: value + grad over x=[q;qd] + GN hess_x, CoM-tracking. q (batch, NQ);
// p_des (batch, 3); W (batch, 3); out (batch); grad (batch, NX); hess (batch, NX*NX).
// Gated on GRID_PLANT_HAS_COM_COST (com + ccrba generated, non-mimic).
extern "C" int grid_plant_com_cost(
    const T* q, const T* p_des, const T* W,
    T* out, T* grad, T* hess, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int nq = grid::NUM_POS;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, q,     batch * nq * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, p_des, batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, W,     batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    size_t smem = grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(batch);
    // com_cost_kernel can be register-heavy; clamp to its launch cap (else the
    // launch is silently rejected -> stale-zero d_grad). cudaGetLastError surfaces
    // any launch-config/resource rejection as rc=200+e instead of silent zeros.
    dim3 thr = grid_clamp_threads_for(grid_plant::com_cost_kernel<T, false>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::com_cost_kernel<T><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_robot, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(out,  g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, g_plant.d_hess, batch * nx * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_PLANT_HAS_COM_COST

#ifdef GRID_PLANT_HAS_MOMENTUM_COST
// momentum_cost: value + grad over x=[q;qd] + GN hess_x, centroidal-momentum
// tracking. q (batch, NQ); qd (batch, NV); h_des (batch, 6); W (batch, 6);
// out (batch); grad (batch, NX); hess (batch, NX*NX).
// Gated on GRID_PLANT_HAS_MOMENTUM_COST (com + ccrba generated, non-mimic).
extern "C" int grid_plant_momentum_cost(
    const T* q, const T* qd, const T* h_des, const T* W,
    T* out, T* grad, T* hess, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int nq = grid::NUM_POS;
    const int nv = grid::NUM_VEL;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    // q -> d_in_a, qd -> d_in_b. h_des(6) and W(6) are packed into the two halves
    // of d_in_c (floored to hold >= 12 per timestep in plant_alloc): h_des in the
    // first batch*6 floats, W in the next batch*6 (each read as [k*6 + r]).
    cudaMemcpy(g_plant.d_in_a, q,     batch * nq * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, qd,    batch * nv * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c,                 h_des, batch * 6 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c + (size_t)batch * 6, W, batch * 6 * sizeof(T), cudaMemcpyHostToDevice);
    size_t smem = grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(batch);
    // momentum_cost is register-heavy (~140 regs/thread): clamp to its launch cap.
    dim3 thr = grid_clamp_threads_for(grid_plant::momentum_cost_kernel<T, false>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::momentum_cost_kernel<T><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b,
        g_plant.d_in_c, g_plant.d_in_c + (size_t)batch * 6,
        g_robot, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(out,  g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, g_plant.d_hess, batch * nx * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_PLANT_HAS_MOMENTUM_COST

#if defined(GRID_RBD_WITH_MUJOCO) && defined(GRID_PLANT_HAS_EE_COST)
// MuJoCo-convention ee_pos_cost (floating base only). q is raw mjx; the kernel
// (MUJOCO_OUTPUT=true) input-converts q (quaternion reorder) so the world-frame EE
// value is correct, computes the pin grad/hess, then base-rotates the q-block grad
// (covector) and congruence-reframes the q-block GN hess before saving. Value invariant.
extern "C" int grid_rbd_ee_pos_cost_mujoco(
    const T* q, const T* p_des, const T* W,
    T* out, T* grad, T* hess, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int nq = grid::NUM_POS;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, q,     batch * nq * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, p_des, batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, W,     batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    size_t smem = grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::ee_pos_cost_kernel<T, 0, true>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::ee_pos_cost_kernel<T, /*EE=*/0, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose, g_plant.d_end_effector_pose_gradient, g_robot, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(out,  g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, g_plant.d_hess, batch * nx * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_PLANT_HAS_EE_COST

#if defined(GRID_RBD_WITH_MUJOCO) && defined(GRID_PLANT_HAS_COM_COST)
// MuJoCo-convention com_cost (floating base only). Same transform as ee_pos_cost
// (q-block grad covector + GN hess congruence; q input-converted in-kernel).
extern "C" int grid_rbd_com_cost_mujoco(
    const T* q, const T* p_des, const T* W,
    T* out, T* grad, T* hess, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int nq = grid::NUM_POS;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, q,     batch * nq * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, p_des, batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, W,     batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    size_t smem = grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(batch);
    // clamp to the kernel's launch cap + surface launch errors (mirror the pin path).
    dim3 thr = grid_clamp_threads_for(grid_plant::com_cost_kernel<T, true>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::com_cost_kernel<T, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_robot, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(out,  g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, g_plant.d_hess, batch * nx * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_PLANT_HAS_COM_COST

#if defined(GRID_RBD_WITH_MUJOCO) && defined(GRID_PLANT_HAS_MOMENTUM_COST)
// MuJoCo-convention momentum_cost (floating base only). The kernel input-converts q
// AND qd (qd[0:3] = R^T qd[0:3]) so h = A qd is the correct invariant momentum, then
// reframes the qd-block grad (covector) + GN hess (congruence at offset nq). Value invariant.
extern "C" int grid_rbd_momentum_cost_mujoco(
    const T* q, const T* qd, const T* h_des, const T* W,
    T* out, T* grad, T* hess, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int nq = grid::NUM_POS;
    const int nv = grid::NUM_VEL;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, q,     batch * nq * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, qd,    batch * nv * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c,                 h_des, batch * 6 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c + (size_t)batch * 6, W, batch * 6 * sizeof(T), cudaMemcpyHostToDevice);
    size_t smem = grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(batch);
    // momentum_cost is register-heavy (~140 regs/thread): clamp to its launch cap.
    dim3 thr = grid_clamp_threads_for(grid_plant::momentum_cost_kernel<T, true>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::momentum_cost_kernel<T, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b,
        g_plant.d_in_c, g_plant.d_in_c + (size_t)batch * 6,
        g_robot, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(out,  g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, g_plant.d_hess, batch * nx * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_PLANT_HAS_MOMENTUM_COST

#ifdef GRID_PLANT_HAS_STEP_GRADIENT
// plant_step_gradient: [A|B] = d x_{k+1}/d(x,u) = integrator_gradient([q;qd], u).
// x (batch, NX); u (batch, NV); dAB (batch, 2*NV*3*NV, column-major). The kernel
// owns the FULL FD-grad scratch arena in shared memory (PERF/full-smem), so the
// binding only stages x/u and reads dAB. Gated on GRID_PLANT_HAS_STEP_GRADIENT
// (integrator_gradient generated). IT via the same int code.
template <grid::IntegratorType IT>
static void launch_plant_step_gradient(int batch, T gravity, T dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(batch);
    // clamp to the kernel's launch cap (register-heavy; mirror the mjx twin).
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_gradient_kernel<T, IT>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    // the dAB output band + fdsva-grad scratch arena exceeds the 48 KB static-smem
    // default -> raise the per-kernel dynamic-smem cap (else cudaErrorInvalidValue/rc=201).
    const size_t smem = grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_gradient_kernel<T, IT>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    grid_plant::plant_step_gradient_kernel<T, IT><<<grid_dim, thr,
        smem, g_streams[0]>>>(
            g_plant.d_grad /*reuse as d_dAB, size 2*NV*3*NV*/, g_plant.d_in_a, g_plant.d_in_b,
            nx, nv, g_robot, gravity, dt, batch);
}

extern "C" int grid_plant_step_gradient(
    const T* x, const T* u, T* dAB, int batch, float gravity, float dt, int it) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    const int dab = 2 * nv * 3 * nv;
    cudaMemcpy(g_plant.d_in_a, x, batch * nx * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, u, batch * nv * sizeof(T), cudaMemcpyHostToDevice);
    GRID_RBD_IT_DISPATCH(it, launch_plant_step_gradient, batch, (T)gravity, (T)dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(dAB, g_plant.d_grad, batch * dab * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention plant_step_gradient (floating base only). x/u raw mjx; the kernel
// (MUJOCO_OUTPUT=true) input-converts the stacked state + forwards to the mjx
// integrator_gradient_device (state-transition Jacobian). EULER/SI-EULER only.
template <grid::IntegratorType IT>
static void launch_plant_step_gradient_mujoco(int batch, T gravity, T dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_gradient_kernel<T, IT, true>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    // raise the per-kernel dynamic-smem cap (arena > 48 KB; else rc=201).
    const size_t smem = grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_gradient_kernel<T, IT, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    grid_plant::plant_step_gradient_kernel<T, IT, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr,
        smem, g_streams[0]>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, nv, g_robot, gravity, dt, batch);
}
extern "C" int grid_plant_step_gradient_mujoco(
    const T* x, const T* u, T* dAB, int batch, float gravity, float dt, int it) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, x, batch * nx * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, u, batch * nv * sizeof(T), cudaMemcpyHostToDevice);
    GRID_RBD_IT_DISPATCH_HESSIAN(it, launch_plant_step_gradient_mujoco, batch, (T)gravity, (T)dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(dAB, g_plant.d_grad, batch * (2 * nv * 3 * nv) * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_PLANT_HAS_STEP_GRADIENT

#ifdef GRID_PLANT_HAS_STEP_HESSIAN
// plant_step_hessian: s_d2AB = d^2 x_{k+1}/d z^2, z=[q;qd;u] (the 2nd-order
// sensitivity of the integrator step). x (batch, NX); u (batch, NV); d2AB
// (batch, 2*NV*3*NV*3*NV, row-major H[o*nz*nz + a*nz + b], nz=3*NV). Like
// plant_step_gradient the kernel owns the FULL fdsva_so scratch arena in shared
// memory (SHARED/PERF tier), so the binding only stages x/u and reads d2AB. The
// arena (s_d2AB output band + fdsva_so scratch) exceeds the 48 KB static smem
// default, so we raise the per-kernel cap via cudaFuncSetAttribute before the
// launch. Gated on GRID_PLANT_HAS_STEP_HESSIAN (integrator_hessian generated);
// only EULER / SI-EULER (the composed device fn static_asserts the rest out).
template <grid::IntegratorType IT>
static void launch_plant_step_hessian(int batch, T gravity, T dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    // Tier-aware: at TIER_SHARED the whole arena (s_d2AB output + fdsva scratch)
    // is in smem and d_workspace is nullptr; at LITE/MINIMAL the cold/large bands
    // spill to the per-timestep d_d2AB_workspace. The kernel template's
    // RESOURCE_TIER defaults to GRID_DEFAULT_RESOURCE_TIER (set by codegen from the
    // smem target), so the smem macro + the kernel use the same tier.
    const size_t smem = grid_plant::INTEGRATOR_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_hessian_kernel<T, IT>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid_dim = grid_rbd_grid_for(batch);
    grid_plant::plant_step_hessian_kernel<T, IT><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), smem, g_streams[0]>>>(
        g_plant.d_d2AB, g_plant.d_d2AB_workspace, g_plant.d_in_a, g_plant.d_in_b, nx, nv, g_robot, gravity, dt, batch);
}

extern "C" int grid_plant_step_hessian(
    const T* x, const T* u, T* d2AB, int batch, float gravity, float dt, int it) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    const int nz = 3 * nv;
    const size_t d2ab = (size_t)(2 * nv) * (size_t)nz * (size_t)nz;
    // d_d2AB (18*NV^3 per timestep) is far larger than d_grad's worst-case
    // (6*NV^2), so it gets its own lazily-allocated band (allocated on first use).
    if (g_plant.d_d2AB == nullptr) {
        if (cudaMalloc(&g_plant.d_d2AB, (size_t)kMaxBatch * d2ab * sizeof(T)) != cudaSuccess)
            return 4;
    }
    // Spill workspace: only allocated when some tier spills (big robots). One
    // per-timestep slot per block (k indexes the block). On TIER_SHARED the macro
    // is 0 and the kernel never touches d_workspace (passed but unused).
    if (grid_plant::GRID_PLANT_HESSIAN_USES_WORKSPACE_ANY_TIER && g_plant.d_d2AB_workspace == nullptr) {
        const size_t ws = grid_plant::PLANT_HESSIAN_WORKSPACE_BYTES_PER_TIMESTEP<T>();
        if (cudaMalloc(&g_plant.d_d2AB_workspace, (size_t)kMaxBatch * ws) != cudaSuccess)
            return 4;
    }
    cudaMemcpy(g_plant.d_in_a, x, batch * nx * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, u, batch * nv * sizeof(T), cudaMemcpyHostToDevice);
    GRID_RBD_IT_DISPATCH_HESSIAN(it, launch_plant_step_hessian, batch, (T)gravity, (T)dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(d2AB, g_plant.d_d2AB, batch * d2ab * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention plant_step_hessian (floating base only). x/u raw mjx; the kernel
// (MUJOCO_OUTPUT=true) input-converts the stacked state + transforms the 2nd-order
// state-transition tensor to the mjx tangent (reuses fdsva_so SO tensors + a dedicated
// mjx workspace band carved from d_workspace). EULER/SI-EULER only. Register/smem-heavy.
template <grid::IntegratorType IT>
static void launch_plant_step_hessian_mujoco(int batch, T gravity, T dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    const size_t smem = grid_plant::INTEGRATOR_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_hessian_kernel<T, IT, grid::GRID_DEFAULT_RESOURCE_TIER, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid_dim = grid_rbd_grid_for(batch);
    dim3 thr = grid_clamp_threads_for(
        grid_plant::plant_step_hessian_kernel<T, IT, grid::GRID_DEFAULT_RESOURCE_TIER, true>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::plant_step_hessian_kernel<T, IT, grid::GRID_DEFAULT_RESOURCE_TIER, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_d2AB, g_plant.d_d2AB_workspace, g_plant.d_in_a, g_plant.d_in_b, nx, nv, g_robot, gravity, dt, batch);
}
extern "C" int grid_plant_step_hessian_mujoco(
    const T* x, const T* u, T* d2AB, int batch, float gravity, float dt, int it) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    if (plant_alloc()) return 4;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    const int nz = 3 * nv;
    const size_t d2ab = (size_t)(2 * nv) * (size_t)nz * (size_t)nz;
    if (g_plant.d_d2AB == nullptr) {
        if (cudaMalloc(&g_plant.d_d2AB, (size_t)kMaxBatch * d2ab * sizeof(T)) != cudaSuccess) return 4;
    }
    if (grid_plant::GRID_PLANT_HESSIAN_USES_WORKSPACE_ANY_TIER && g_plant.d_d2AB_workspace == nullptr) {
        const size_t ws = grid_plant::PLANT_HESSIAN_WORKSPACE_BYTES_PER_TIMESTEP<T>();
        if (cudaMalloc(&g_plant.d_d2AB_workspace, (size_t)kMaxBatch * ws) != cudaSuccess) return 4;
    }
    cudaMemcpy(g_plant.d_in_a, x, batch * nx * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, u, batch * nv * sizeof(T), cudaMemcpyHostToDevice);
    GRID_RBD_IT_DISPATCH_HESSIAN(it, launch_plant_step_hessian_mujoco, batch, (T)gravity, (T)dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(d2AB, g_plant.d_d2AB, batch * d2ab * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_PLANT_HAS_STEP_HESSIAN


// ────────────────────────────────────────────────────────────────────────────
// JAX FFI handlers
// ────────────────────────────────────────────────────────────────────────────
//
// Gated on -DGRID_RBD_WITH_JAX (set by _compile.py when JAX is installed at
// register_robot time). Each handler:
//   1. Receives JAX device buffers (q, qd, ...) plus a CUDA stream
//      managed by JAX.
//   2. Repacks the (q, qd[, u]) inputs into the singleton's device-side
//      d_q_qd_u via cudaMemcpy2DAsync D→D (one wide copy per input, no
//      host round-trip).
//   3. Launches the algorithm's kernel DIRECTLY on the JAX stream
//      (bypassing the host wrapper's internal staging logic). All
//      computation stays on the GPU.
//   4. Copies the result from the singleton's device output buffer
//      (d_c, d_qdd, d_M, etc.) into JAX's output buffer via
//      cudaMemcpyAsync D→D on the same stream.
//
// Net per call: 2-3 D→D cudaMemcpy ops + the kernel launch. No host
// involvement, no implicit serialization with non-JAX streams. JAX's
// scheduler owns the work.

// fp64 (Phase 8): JAX/torch surfaces are fp32-only by design (their handlers
// use ffi::F32 / float32 tensors). Suppress the whole block in an fp64 .so so
// it never ABI-mismatches T=double. fp64 is a numpy-handle / inline-CUDA feature.
#if defined(GRID_RBD_WITH_JAX) && !defined(GRID_WRAPPER_T_DOUBLE)

#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

#if GRID_HAS_INVERSE_DYNAMICS
// inverse_dynamics(q, qd, qdd, f_ext) → c   — fully device-resident path.
//
// qdd and f_ext are ALWAYS passed as explicit device buffers from the Python
// surface (JAX FFI has no optional-buffer support, so the wrapper passes zeros
// when the caller omits them — mirroring idsva_so). qdd flows through the
// separate d_qdd buffer + the USE_QDD overload of the kernel (signature
// (d_c, d_q_qd, stride, d_qdd, d_f_ext, ...)); f_ext is copied D→D into d_f_ext.
// MUJOCO templates the kernel's compile-time output-convention flag: MUJOCO=false
// is the pinocchio path (byte-identical to the legacy handler); MUJOCO=true launches
// the same kernel with MUJOCO_OUTPUT=true so the device code converts q/qd mjx->pin on
// load and rotates the base-linear tau rows back to the mjx frame -- no host pre/post.
// The mjx instantiation is FLOATING-base only (gated where the handler is defined).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_inverse_dynamics_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,         // shape (B, NJ), device-resident
    ffi::Buffer<ffi::F32> qd,        // shape (B, NJ), device-resident
    ffi::Buffer<ffi::F32> qdd,       // shape (B, NJ), device-resident
    ffi::Buffer<ffi::F32> f_ext,     // shape (B, 6*NUM_BODIES), device-resident
    ffi::ResultBuffer<ffi::F32> c,   // shape (B, NJ), device-resident
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) {
        return ffi::Error::Internal("grid_rbd_init failed");
    }}
    auto q_shape = q.dimensions();
    if (q_shape.size() != 2) {
        return ffi::Error::InvalidArgument("inverse_dynamics: q must be 2D (B, NJ)");
    }
    int batch = (int)q_shape[0];
    int nj    = (int)q_shape[1];
    if (nj != grid::NUM_JOINTS) {
        return ffi::Error::InvalidArgument("inverse_dynamics: last dim != NUM_JOINTS");
    }
    if (batch > kMaxBatch) {
        return ffi::Error::InvalidArgument(
            "inverse_dynamics: batch exceeds compiled-in max_batch_size; recompile with a larger value");
    }

    // D→D repack: interleave q and qd into the singleton's d_q_qd_u
    // (layout [q[NJ], qd[NJ], u[NJ]] per timestep). cudaMemcpy2DAsync
    // writes a (batch, NJ) src into the (batch, 3*NJ) dst with the right
    // stride in one call per input.
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],       dst_pitch,
                      q.typed_data(),            row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj],      dst_pitch,
                      qd.typed_data(),           row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    // qdd → the separate d_qdd buffer (NJ-contiguous per timestep), consumed
    // by the USE_QDD overload of inverse_dynamics_kernel.
    cudaMemcpyAsync(g_data->d_qdd, qdd.typed_data(),
                    (size_t)batch * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    // f_ext → d_f_ext (the Python surface passes a zero buffer when omitted).
    cudaMemcpyAsync(g_data->d_f_ext, f_ext.typed_data(),
                    (size_t)batch * 6 * grid::NUM_BODIES * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);

    // Launch the qdd overload directly on JAX's stream (the qdd kernel reads
    // the acceleration from d_qdd; signature adds d_qdd after stride).
    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::inverse_dynamics_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS>(),
        grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_c, g_data->d_q_qd_u, stride_q_qd, g_data->d_qdd,
            g_data->d_f_ext, g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("inverse_dynamics_kernel launch failed");

    // D→D copy the result into JAX's output buffer on the same stream.
    cudaMemcpyAsync(c->typed_data(), g_data->d_c,
                    batch * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics,
    grid_rbd_jax_inverse_dynamics_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()  // q
        .Arg<ffi::Buffer<ffi::F32>>()  // qd
        .Arg<ffi::Buffer<ffi::F32>>()  // qdd
        .Arg<ffi::Buffer<ffi::F32>>()  // f_ext
        .Ret<ffi::Buffer<ffi::F32>>()  // c
        .Attr<float>("gravity")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention inverse_dynamics (floating only): identical plumbing, kernel
// launched with MUJOCO_OUTPUT=true. Registered under a separate target name so the
// python jax surface dispatches on output_convention="mujoco".
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics_mujoco,
    grid_rbd_jax_inverse_dynamics_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()  // q
        .Arg<ffi::Buffer<ffi::F32>>()  // qd
        .Arg<ffi::Buffer<ffi::F32>>()  // qdd
        .Arg<ffi::Buffer<ffi::F32>>()  // f_ext
        .Ret<ffi::Buffer<ffi::F32>>()  // c
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_INVERSE_DYNAMICS

// Shared helper: validate (B, NJ) and return batch size, or error.
// Kept inline so each handler stays self-contained for grep-ability.
#define GRID_RBD_FFI_VALIDATE_2D(buf, name, expected_last_dim)                   \
    do {                                                                         \
        auto _dims = (buf).dimensions();                                         \
        if (_dims.size() != 2)                                                   \
            return ffi::Error::InvalidArgument(                                  \
                std::string(name) + ": must be 2D (B, " +                        \
                std::to_string(expected_last_dim) + ")");                        \
        if ((int)_dims[1] != (expected_last_dim))                                \
            return ffi::Error::InvalidArgument(                                  \
                std::string(name) + ": last dim != " +                           \
                std::to_string(expected_last_dim));                              \
    } while (0)


#if GRID_HAS_MINV
// minv(q) → Minv  (kernel writes lower triangle only; symmetrize Python-side)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_minv_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> minv_out)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "minv: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    int nv    = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("minv: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch,
                      q.typed_data(),       row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd_u = 3 * grid::NUM_JOINTS;
    grid::minv_kernel<T, grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_MINV>(),
        grid::MINV_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER>(),
        stream>>>(
            g_data->d_Minv, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u,
            g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("minv_kernel launch failed");

    // Minv is nv x nv per timestep (tangent-space, pinocchio convention) — the
    // kernel writes the (correctly nv*nv-strided) d_Minv. Copy straight from the
    // device buffer at nv*nv (matching the numpy C-ABI minv + the JAX out_shape /
    // VJP, all unified at nv). FIXED base: nv == nj, byte-identical to the old path.
    cudaMemcpyAsync(minv_out->typed_data(), g_data->d_Minv,
                    batch * nv * nv * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_minv,
    grid_rbd_jax_minv_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention minv (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_minv_mujoco,
    grid_rbd_jax_minv_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_MINV


#if GRID_HAS_FORWARD_DYNAMICS
// forward_dynamics(q, qd, u, f_ext) → qdd  (f_ext always passed; zeros if omitted)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_forward_dynamics_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> u,
    ffi::Buffer<ffi::F32> f_ext,
    ffi::ResultBuffer<ffi::F32> qdd_out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "fd: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("fd: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],      dst_pitch,
                      q.typed_data(),            row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj],     dst_pitch,
                      qd.typed_data(),           row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj],   dst_pitch,
                      u.typed_data(),            row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_data->d_f_ext, f_ext.typed_data(),
                    (size_t)batch * 6 * grid::NUM_BODIES * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd_u = 3 * grid::NUM_JOINTS;
    grid::forward_dynamics_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_FORWARD_DYNAMICS>(),
        grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER>(),
        stream>>>(
            g_data->d_qdd, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u,
            g_data->d_f_ext, g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("forward_dynamics_kernel launch failed");

    cudaMemcpyAsync(qdd_out->typed_data(), g_data->d_qdd,
                    batch * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_forward_dynamics,
    grid_rbd_jax_forward_dynamics_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()  // f_ext
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention forward_dynamics (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_forward_dynamics_mujoco,
    grid_rbd_jax_forward_dynamics_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()  // f_ext
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_FORWARD_DYNAMICS


#if GRID_HAS_ABA
// aba(q, qd, u, f_ext) → qdd  — same kernel signature shape as forward_dynamics
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_aba_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> u,
    ffi::Buffer<ffi::F32> f_ext,
    ffi::ResultBuffer<ffi::F32> qdd_out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "aba: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("aba: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],      dst_pitch,
                      q.typed_data(),            row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj],     dst_pitch,
                      qd.typed_data(),           row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj],   dst_pitch,
                      u.typed_data(),            row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_data->d_f_ext, f_ext.typed_data(),
                    (size_t)batch * 6 * grid::NUM_BODIES * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::aba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_ABA>(),
        grid::ABA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER>(),
        stream>>>(
            g_data->d_qdd, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd,
            g_data->d_f_ext, g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("aba_kernel launch failed");

    cudaMemcpyAsync(qdd_out->typed_data(), g_data->d_qdd,
                    batch * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_aba,
    grid_rbd_jax_aba_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()  // f_ext
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention aba (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_aba_mujoco,
    grid_rbd_jax_aba_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()  // f_ext
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_ABA


#if GRID_HAS_CRBA
// crba(q) → M  (kernel writes the full mass matrix; no symmetrize needed)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_crba_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> m_out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "crba: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    int nv    = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("crba: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch,
                      q.typed_data(),       row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::crba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_CRBA>(),
        grid::CRBA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER>(),
        stream>>>(
            g_data->d_M, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd,
            g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("crba_kernel launch failed");

    // M is nv x nv per timestep (tangent-space, pinocchio convention) — the kernel
    // writes the (correctly nv*nv-strided) d_M. Copy straight from the device
    // buffer at nv*nv (matching the numpy C-ABI crba + the JAX out_shape, unified
    // at nv). FIXED base: nv == nj, byte-identical to the old path.
    cudaMemcpyAsync(m_out->typed_data(), g_data->d_M,
                    batch * nv * nv * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_crba,
    grid_rbd_jax_crba_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention crba (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_crba_mujoco,
    grid_rbd_jax_crba_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_CRBA


#if GRID_HAS_END_EFFECTOR_POSE
// end_effector_pose(q) → end_effector_pose  flat (B, 6*NUM_EES)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_end_effector_pose_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> ee_out)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "end_effector_pose: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("end_effector_pose: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch,
                      q.typed_data(),       row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q = 3 * grid::NUM_JOINTS;
    grid::GRID_RBD_EE_POSE_KERNEL<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_END_EFFECTOR_POSE>(),
        grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_end_effector_pose, g_data->d_q_qd_u, stride_q,
            g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("GRID_RBD_EE_POSE_KERNEL launch failed");

    cudaMemcpyAsync(ee_out->typed_data(), g_data->d_end_effector_pose,
                    batch * 6 * GRID_RBD_NUM_EES * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose,
    grid_rbd_jax_end_effector_pose_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention end_effector_pose (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_mujoco,
    grid_rbd_jax_end_effector_pose_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_END_EFFECTOR_POSE


#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
// end_effector_pose_gradient(q) → end_effector_pose_gradient d/dv flat (B, 6*NUM_EES*NV).
// Output convention: d/dv tangent (pinocchio); floating-base shape uses NV
// (= 6 + n_joints) NOT NJ. Python side reshapes/transposes to the
// (B, 6*NUM_EES, NV) row-major convention (see _handle.py).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_end_effector_pose_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> dee_out)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "end_effector_pose_gradient: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    int nv    = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("end_effector_pose_gradient: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch,
                      q.typed_data(),       row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q = 3 * grid::NUM_JOINTS;
    grid::GRID_RBD_EE_POSE_GRADIENT_KERNEL<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>(),
        grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER>(),
        stream>>>(
            g_data->d_end_effector_pose_gradient, g_data->d_workspace, g_data->d_q_qd_u, stride_q,
            g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("GRID_RBD_EE_POSE_GRADIENT_KERNEL launch failed");

    cudaMemcpyAsync(dee_out->typed_data(), g_data->d_end_effector_pose_gradient,
                    batch * 6 * GRID_RBD_NUM_EES * nv * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_gradient,
    grid_rbd_jax_end_effector_pose_gradient_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention end_effector_pose_gradient (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_gradient_mujoco,
    grid_rbd_jax_end_effector_pose_gradient_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_END_EFFECTOR_POSE_GRADIENT


// ─── runtime-target multi-EE pose / pose-gradient (single-target FFI) ─────────
// These mirror the numpy C-ABI grid_rbd_end_effector_pose_runtime: a SINGLE target
// jid + a SINGLE 16-float col-major 4x4 SE(3) tool transform per call (the kernel's
// s_Xtool; identity => frame origin, R_tool=I + t => point offset). The Python
// wrapper resolves names→jids and loops/stacks the EE list (matches _handle.py).
// The transform is passed as a 16-float array .Attr ("xtool") and staged into the
// shared device buffer g_data->d_eepose_runtime_offset via cudaMemcpyAsync on the
// stream — an attr (not a Buffer input) so jax/vmap never tries to batch it.
// target_jid is an int64 attr (already an absolute jid; the Python layer resolves
// names→jids, so no -1 default).
#ifdef GRID_HAS_END_EFFECTOR_POSE_RUNTIME
// end_effector_pose_runtime(q) → (B, 6) [xyz; rpy] of target_jid at offset point.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_end_effector_pose_runtime_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> ee_out,
    int64_t target_jid, ffi::Span<const float> xtool)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "end_effector_pose_runtime: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("end_effector_pose_runtime: batch > max_batch");
    if (xtool.size() != 16) return ffi::Error::InvalidArgument(
        "end_effector_pose_runtime: xtool must be the 16-float col-major 4x4 SE(3) tool transform");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch,
                      q.typed_data(),       row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    // stage the full 16-float X_tool into the shared device buffer (the kernel
    // reads all 16: translation at [12..14], R_tool in the leading columns).
    T xt[16];
    for (int i = 0; i < 16; ++i) xt[i] = static_cast<T>(xtool[i]);
    cudaMemcpyAsync(g_data->d_eepose_runtime_offset, xt, 16 * sizeof(T),
                    cudaMemcpyHostToDevice, stream);

    constexpr int stride_q = 3 * grid::NUM_JOINTS;
    grid::end_effector_pose_runtime_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(),
        grid::END_EFFECTOR_POSE_RUNTIME_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_eePose, g_data->d_q_qd_u, stride_q,
            (int)target_jid, g_data->d_eepose_runtime_offset, g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("end_effector_pose_runtime_kernel launch failed");

    cudaMemcpyAsync(ee_out->typed_data(), g_data->d_eePose,
                    batch * 6 * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_runtime,
    grid_rbd_jax_end_effector_pose_runtime_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("target_jid")
        .Attr<ffi::Span<const float>>("xtool")
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_runtime_mujoco,
    grid_rbd_jax_end_effector_pose_runtime_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("target_jid")
        .Attr<ffi::Span<const float>>("xtool")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_END_EFFECTOR_POSE_RUNTIME


#ifdef GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME
// end_effector_pose_gradient_runtime(q) → (B, 6*NV) col-major d[xyz; rpy]/dv of
// target_jid at the offset point. Python reshapes (B,NV,6)→transpose→(B,6,NV).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_end_effector_pose_gradient_runtime_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> dee_out,
    int64_t target_jid, ffi::Span<const float> xtool)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "end_effector_pose_gradient_runtime: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("end_effector_pose_gradient_runtime: batch > max_batch");
    if (xtool.size() != 16) return ffi::Error::InvalidArgument(
        "end_effector_pose_gradient_runtime: xtool must be the 16-float col-major 4x4 SE(3) tool transform");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch,
                      q.typed_data(),       row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    T xt[16];
    for (int i = 0; i < 16; ++i) xt[i] = static_cast<T>(xtool[i]);
    cudaMemcpyAsync(g_data->d_eepose_runtime_offset, xt, 16 * sizeof(T),
                    cudaMemcpyHostToDevice, stream);

    constexpr int stride_q = 3 * grid::NUM_JOINTS;
    grid::end_effector_pose_gradient_runtime_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(),
        grid::END_EFFECTOR_POSE_GRADIENT_RUNTIME_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_eePoseGrad, g_data->d_q_qd_u, stride_q,
            (int)target_jid, g_data->d_eepose_runtime_offset, g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("end_effector_pose_gradient_runtime_kernel launch failed");

    cudaMemcpyAsync(dee_out->typed_data(), g_data->d_eePoseGrad,
                    batch * 6 * nv * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_gradient_runtime,
    grid_rbd_jax_end_effector_pose_gradient_runtime_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("target_jid")
        .Attr<ffi::Span<const float>>("xtool")
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_gradient_runtime_mujoco,
    grid_rbd_jax_end_effector_pose_gradient_runtime_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("target_jid")
        .Attr<ffi::Span<const float>>("xtool")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME


#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN
// end_effector_pose_hessian(q) → end_effector_pose_hessian  flat (B, 6*NUM_EES*NV*NV)
// The kernel also writes d_end_effector_pose_gradient as a byproduct; we only return d2.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_end_effector_pose_hessian_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> d2ee_out)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "end_effector_pose_hessian: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    int nv    = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("end_effector_pose_hessian: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch,
                      q.typed_data(),       row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q = 3 * grid::NUM_JOINTS;
    grid::GRID_RBD_EE_POSE_HESSIAN_KERNEL<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>(),
        grid::END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER>(),
        stream>>>(
            g_data->d_end_effector_pose_hessian, g_data->d_end_effector_pose_gradient, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q, g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("GRID_RBD_EE_POSE_HESSIAN_KERNEL launch failed");

    cudaMemcpyAsync(d2ee_out->typed_data(), g_data->d_end_effector_pose_hessian,
                    batch * 6 * GRID_RBD_NUM_EES * nv * nv * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_hessian,
    grid_rbd_jax_end_effector_pose_hessian_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention end_effector_pose_hessian (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_hessian_mujoco,
    grid_rbd_jax_end_effector_pose_hessian_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_END_EFFECTOR_POSE_HESSIAN


#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
// inverse_dynamics_gradient(q, qd, qdd) → dc_du  flat (B, 2*NV*NV)
// Python reshapes/transposes to (B, NV, 2*NV) [dc_dq | dc_dqd] (tangent-space;
// FIXED base NV == NJ, FLOATING base NV < NJ).
//
// qdd is ALWAYS passed as an explicit device buffer from the Python surface
// (JAX FFI has no optional-buffer support, so the wrapper passes zeros when the
// caller omits it — mirroring the VALUE inverse_dynamics FFI). ∂c/∂(q,qd)
// depends on qdd via the M·qdd term's derivatives, so qdd flows through the
// separate d_qdd buffer + the USE_QDD overload of the gradient kernel
// (signature adds d_qdd after stride). A zero qdd is byte-identical to the old
// no-qdd behaviour.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_inverse_dynamics_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> qdd,
    ffi::ResultBuffer<ffi::F32> dc_du_out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "inverse_dynamics_gradient: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    int nv    = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("inverse_dynamics_gradient: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],  dst_pitch,
                      q.typed_data(),        row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch,
                      qd.typed_data(),       row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    // qdd → the separate d_qdd buffer (NJ-contiguous per timestep), consumed by
    // the USE_QDD overload of inverse_dynamics_gradient_kernel.
    cudaMemcpyAsync(g_data->d_qdd, qdd.typed_data(),
                    (size_t)batch * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::inverse_dynamics_gradient_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(),
        grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(),
        stream>>>(
            g_data->d_dc_du, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd, g_data->d_qdd,
            g_data->d_f_ext, g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("inverse_dynamics_gradient_kernel launch failed");

    // dc_du is nv x 2nv per timestep (tangent-space) — the gradient kernel writes
    // the (correctly 2*nv*nv-strided) d_dc_du. Copy at 2*nv*nv (matching numpy
    // C-ABI + the JAX out_shape / VJP, unified at nv). FIXED base: nv == nj.
    cudaMemcpyAsync(dc_du_out->typed_data(), g_data->d_dc_du,
                    batch * 2 * nv * nv * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics_gradient,
    grid_rbd_jax_inverse_dynamics_gradient_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()  // q
        .Arg<ffi::Buffer<ffi::F32>>()  // qd
        .Arg<ffi::Buffer<ffi::F32>>()  // qdd
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention inverse_dynamics_gradient (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics_gradient_mujoco,
    grid_rbd_jax_inverse_dynamics_gradient_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()  // q
        .Arg<ffi::Buffer<ffi::F32>>()  // qd
        .Arg<ffi::Buffer<ffi::F32>>()  // qdd
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_INVERSE_DYNAMICS_GRADIENT


#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
// forward_dynamics_gradient(q, qd, u) → df_du  flat (B, 2*NV*NV)
// Python reshapes/transposes to (B, NV, 2*NV) (tangent-space; FIXED base
// NV == NJ, FLOATING base NV < NJ).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_forward_dynamics_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> u,
    ffi::ResultBuffer<ffi::F32> df_du_out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "forward_dynamics_gradient: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    int nv    = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("forward_dynamics_gradient: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],      dst_pitch,
                      q.typed_data(),            row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj],     dst_pitch,
                      qd.typed_data(),           row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj],   dst_pitch,
                      u.typed_data(),            row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd_u = 3 * grid::NUM_JOINTS;
    grid::forward_dynamics_gradient_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>(),
        grid::FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER>(),
        stream>>>(
            g_data->d_df_du, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u,
            g_data->d_f_ext, g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("forward_dynamics_gradient_kernel launch failed");

    // df_du is nv x 2nv per timestep (tangent-space) — the kernel writes the
    // (correctly 2*nv*nv-strided) d_df_du. Copy at 2*nv*nv (matching numpy C-ABI +
    // the JAX out_shape / VJP, unified at nv). FIXED base: nv == nj.
    cudaMemcpyAsync(df_du_out->typed_data(), g_data->d_df_du,
                    batch * 2 * nv * nv * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_forward_dynamics_gradient,
    grid_rbd_jax_forward_dynamics_gradient_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention forward_dynamics_gradient (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_forward_dynamics_gradient_mujoco,
    grid_rbd_jax_forward_dynamics_gradient_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_FORWARD_DYNAMICS_GRADIENT


#if GRID_HAS_IDSVA_SO
// idsva_so(q, qd, qdd) → packed (B, SECOND_ORDER_TENSOR_SIZE)
// The codegen-time dispatcher picks body- vs world-frame; we dispatch here at
// compile time using the GRID_GENERATES_* macros so a per-robot .so calls
// whichever kernel was emitted. qdd is packed into the acceleration (u) slot of
// d_q_qd_u, which the kernel reads as s_qdd — mirroring the numpy
// pack_q_qd_u(q, qd, qdd). The Python surface passes explicit zeros when the
// caller omits qdd, so the result never depends on a stale device buffer.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_idsva_so_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> qdd,
    ffi::ResultBuffer<ffi::F32> out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "idsva_so: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("idsva_so: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],    dst_pitch,
                      q.typed_data(),          row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj],   dst_pitch,
                      qd.typed_data(),         row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj], dst_pitch,
                      qdd.typed_data(),        row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd_u = 3 * grid::NUM_JOINTS;
    // Compile-time frame dispatch: floating-base .so call the world-frame kernel
    // (which carries the MUJOCO_OUTPUT flag); fixed-base .so call the body-frame
    // kernel (pinocchio-only — mjx is floating-base-only, asserted in the #else).
#ifdef GRID_RBD_WITH_MUJOCO
    grid::idsva_so_world_frame_kernel<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>(),
        grid::IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>::TIER>(),
        stream>>>(
            g_data->d_idsva_so, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u,
            g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("idsva_so_world_frame_kernel launch failed");
#else
    static_assert(!MUJOCO, "mjx idsva_so is floating-base only");
    grid::idsva_so_body_frame_kernel<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_BODY_FRAME>::TIER><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_IDSVA_SO_BODY_FRAME>(),
        grid::IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_BODY_FRAME>::TIER>(),
        stream>>>(
            g_data->d_idsva_so, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u,
            g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("idsva_so_body_frame_kernel launch failed");
#endif

    cudaMemcpyAsync(out->typed_data(), g_data->d_idsva_so,
                    batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_idsva_so,
    grid_rbd_jax_idsva_so_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention idsva_so (floating only): world-frame kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_idsva_so_mujoco,
    grid_rbd_jax_idsva_so_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_IDSVA_SO


#if GRID_HAS_FDSVA_SO
// fdsva_so(q, qd, u) → packed (B, SECOND_ORDER_TENSOR_SIZE)
// Uses d_idsva_so as scratch — must not run concurrently with idsva_so.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_fdsva_so_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> u,
    ffi::ResultBuffer<ffi::F32> out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "fdsva_so: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("fdsva_so: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],      dst_pitch,
                      q.typed_data(),            row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj],     dst_pitch,
                      qd.typed_data(),           row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj],   dst_pitch,
                      u.typed_data(),            row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd_u = 3 * grid::NUM_JOINTS;
    // v0.3: assumes fdsva_so was emitted (true for all current robots; iiwa14
    // body-frame fits sm_120's 100 KB shared-mem cap, g1_floating world-frame
    // is the known SKIPPED cell). A compile error here would mean the codegen
    // was invoked with second_order=False, which the grid-rbd wrappers never do.
    grid::fdsva_so_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_FDSVA_SO>(),
        grid::FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER>(),
        stream>>>(
            g_data->d_df2, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u, g_data->d_idsva_so,
            g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("fdsva_so_kernel launch failed");

    cudaMemcpyAsync(out->typed_data(), g_data->d_df2,
                    batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_fdsva_so,
    grid_rbd_jax_fdsva_so_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention fdsva_so (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_fdsva_so_mujoco,
    grid_rbd_jax_fdsva_so_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_FDSVA_SO


// ────────────────────────────────────────────────────────────────────────────
// Inertial-parameter regressor surface (PS2a — sysID autodiff)
// ────────────────────────────────────────────────────────────────────────────
//
// Both outputs are row-major (NV x 10*NUM_BODIES) per timestep, where the
// per-link 10-parameter basis is pi_i = [m, m*c(3), I_O(6)] (the parser's
// origin-frame inertia; matches RBDReference._regressor). These back the
// inertial-parameter VJP: tau = Y . pi so dtau/dpi = Y, and
// dqdd/dpi = -Minv . Y. The Python custom_vjp contracts a cotangent ct (NV)
// with these (NV x 10NB) Jacobians to produce the pi-cotangent (10NB).

#if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
// inverse_dynamics_regressor(q, qd, qdd) → Y  flat (B, NV*10*NUM_BODIES).
// qdd is passed explicitly (the bias regressor used by inverse_dynamics's VJP
// passes zeros). The regressor kernel reads q|qd|qdd from d_q_qd_u (stride
// Q_QD_U_STRIDE), the qdd occupying the u-slot — mirroring idsva_so.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_inverse_dynamics_regressor_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> qdd,
    ffi::ResultBuffer<ffi::F32> Y_out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "inverse_dynamics_regressor: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("inverse_dynamics_regressor: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],    dst_pitch,
                      q.typed_data(),          row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj],   dst_pitch,
                      qd.typed_data(),         row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj], dst_pitch,
                      qdd.typed_data(),        row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd_qdd = 3 * grid::NUM_JOINTS;
    const int out_size = grid::NUM_VEL * 10 * grid::NUM_BODIES;
    grid::inverse_dynamics_regressor_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(),
        grid::INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_Y, g_data->d_workspace, g_data->d_q_qd_u, stride_q_qd_qdd,
            g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("inverse_dynamics_regressor_kernel launch failed");

    cudaMemcpyAsync(Y_out->typed_data(), g_data->d_Y,
                    (size_t)batch * out_size * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics_regressor,
    grid_rbd_jax_inverse_dynamics_regressor_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention inverse_dynamics_regressor (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics_regressor_mujoco,
    grid_rbd_jax_inverse_dynamics_regressor_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_INVERSE_DYNAMICS_REGRESSOR


#if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
// forward_dynamics_parameter_gradient(q, qd, u) → dqdd/dpi = -Minv . Y
// flat (B, NV*10*NUM_BODIES). Internally runs FD at (q,qd,u) and the regressor
// at the resulting qdd, then applies -Minv (mirrors RBDReference). The kernel
// takes d_workspace (the s_Y regressor scratch spills there at LITE/MINIMAL).
static ffi::Error grid_rbd_jax_forward_dynamics_parameter_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> u,
    ffi::ResultBuffer<ffi::F32> dqdd_dpi_out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "forward_dynamics_parameter_gradient: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("forward_dynamics_parameter_gradient: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],    dst_pitch,
                      q.typed_data(),          row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj],   dst_pitch,
                      qd.typed_data(),         row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj], dst_pitch,
                      u.typed_data(),          row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd_u = 3 * grid::NUM_JOINTS;
    const int out_size = grid::NUM_VEL * 10 * grid::NUM_BODIES;
    grid::forward_dynamics_parameter_gradient_kernel<T><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(),
        grid::FORWARD_DYNAMICS_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_dqdd_dpi, g_data->d_workspace, g_data->d_q_qd_u, stride_q_qd_u,
            g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("forward_dynamics_parameter_gradient_kernel launch failed");

    cudaMemcpyAsync(dqdd_dpi_out->typed_data(), g_data->d_dqdd_dpi,
                    (size_t)batch * out_size * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_forward_dynamics_parameter_gradient,
    grid_rbd_jax_forward_dynamics_parameter_gradient_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT


// Integrator. dt + it are FFI attributes (runtime scalars; gravity is the
// standard constant). q/qd/u are packed D→D like aba; the integrator kernels
// are launched directly on the JAX stream.
#if GRID_HAS_INTEGRATOR
template <grid::IntegratorType IT, bool MUJOCO>
static void launch_integrator_kernel_jax(cudaStream_t stream, int batch, float dt, float gravity) {
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::integrator_kernel<T, IT, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_INTEGRATOR>(),
        grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER>(), stream>>>(
            g_data->d_x_kp1, g_data->d_workspace, g_data->d_q_qd_u, stride,
            g_robot, g_data->d_f_ext, /*gravity=*/static_cast<T>(gravity), static_cast<T>(dt), batch);
}
#endif  // GRID_HAS_INTEGRATOR
#if GRID_HAS_INTEGRATOR_GRADIENT
template <grid::IntegratorType IT, bool MUJOCO>
static void launch_integrator_grad_kernel_jax(cudaStream_t stream, int batch, float dt, float gravity) {
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::integrator_gradient_kernel<T, IT, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_INTEGRATOR_GRADIENT>(),
        grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR_GRADIENT>::TIER>(), stream>>>(
            g_data->d_dAB, g_data->d_workspace, g_data->d_q_qd_u, stride,
            g_robot, g_data->d_f_ext, /*gravity=*/static_cast<T>(gravity), static_cast<T>(dt), batch);
}
#endif  // GRID_HAS_INTEGRATOR_GRADIENT

#define GRID_RBD_IT_DISPATCH_FFI(it_code, FN, ...)                                 \
    switch (it_code) {                                                             \
        case 0: FN<grid::IntegratorType::EULER>(__VA_ARGS__); break;               \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER>(__VA_ARGS__); break;  \
        case 2: FN<grid::IntegratorType::MIDPOINT>(__VA_ARGS__); break;            \
        case 3: FN<grid::IntegratorType::RK3>(__VA_ARGS__); break;                 \
        case 4: FN<grid::IntegratorType::RK4>(__VA_ARGS__); break;                 \
        default: return ffi::Error::InvalidArgument("integrator: bad it code");    \
    }

// mjx-aware integrator-type dispatch: instantiates FN<IT_case, MUJOCO_FLAG>. The
// helper's first template arg is supplied by the case; the second (MUJOCO) is
// threaded from the templated impl so the kernel launches with MUJOCO_OUTPUT.
#define GRID_RBD_IT_DISPATCH_FFI_MJX(it_code, FN, MUJOCO_FLAG, ...)                                 \
    switch (it_code) {                                                                              \
        case 0: FN<grid::IntegratorType::EULER, MUJOCO_FLAG>(__VA_ARGS__); break;                   \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER, MUJOCO_FLAG>(__VA_ARGS__); break;     \
        case 2: FN<grid::IntegratorType::MIDPOINT, MUJOCO_FLAG>(__VA_ARGS__); break;                \
        case 3: FN<grid::IntegratorType::RK3, MUJOCO_FLAG>(__VA_ARGS__); break;                     \
        case 4: FN<grid::IntegratorType::RK4, MUJOCO_FLAG>(__VA_ARGS__); break;                     \
        default: return ffi::Error::InvalidArgument("integrator: bad it code");                     \
    }

// Single-stage-only mjx dispatch: the gradient (state-transition Jacobian) mjx epilogue
// is currently Euler / Semi-Implicit-Euler only (multi-stage RK mjx derivatives are
// deferred — the device code static_asserts). Multi-stage codes are rejected at runtime.
#define GRID_RBD_IT_DISPATCH_FFI_MJX_SS(it_code, FN, MUJOCO_FLAG, ...)                              \
    switch (it_code) {                                                                              \
        case 0: FN<grid::IntegratorType::EULER, MUJOCO_FLAG>(__VA_ARGS__); break;                   \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER, MUJOCO_FLAG>(__VA_ARGS__); break;     \
        default: return ffi::Error::InvalidArgument(                                                \
            "mujoco integrator/plant-step gradient supports only euler / semi-implicit-euler");     \
    }

// q/qd/u D→D pack into the singleton's d_q_qd_u (layout [q,qd,u] per timestep).
static void grid_rbd_jax_pack_qqdu(cudaStream_t stream, int batch, int nj,
                                   ffi::Buffer<ffi::F32>& q,
                                   ffi::Buffer<ffi::F32>& qd,
                                   ffi::Buffer<ffi::F32>& u) {
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],     dst_pitch, q.typed_data(),  row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj],    dst_pitch, qd.typed_data(), row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj],  dst_pitch, u.typed_data(),  row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
}

#if GRID_HAS_INTEGRATOR
// integrator(q, qd, u; dt, it) → x_kp1  (B, NUM_POS + NUM_VEL)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_integrator_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q, ffi::Buffer<ffi::F32> qd, ffi::Buffer<ffi::F32> u,
    ffi::ResultBuffer<ffi::F32> x_kp1_out,
    float dt, int64_t it, float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "integrator: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("integrator: batch > max_batch");

    grid_rbd_jax_pack_qqdu(stream, batch, nj, q, qd, u);
    GRID_RBD_IT_DISPATCH_FFI_MJX((int)it, launch_integrator_kernel_jax, MUJOCO, stream, batch, dt, gravity);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("integrator_kernel launch failed");

    cudaMemcpyAsync(x_kp1_out->typed_data(), g_data->d_x_kp1,
                    batch * (grid::NUM_POS + grid::NUM_VEL) * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_integrator,
    grid_rbd_jax_integrator_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("dt").Attr<int64_t>("it").Attr<float>("gravity")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention integrator (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_integrator_mujoco,
    grid_rbd_jax_integrator_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("dt").Attr<int64_t>("it").Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_INTEGRATOR

#if GRID_HAS_INTEGRATOR_GRADIENT
// integrator_gradient(q, qd, u; dt, it) → dAB  (B, 2*NV, 3*NV)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_integrator_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q, ffi::Buffer<ffi::F32> qd, ffi::Buffer<ffi::F32> u,
    ffi::ResultBuffer<ffi::F32> dAB_out,
    float dt, int64_t it, float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "integrator_gradient: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    int nv    = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("integrator_gradient: batch > max_batch");

    grid_rbd_jax_pack_qqdu(stream, batch, nj, q, qd, u);
    // mjx gradient is single-stage (euler/si) only; pin supports all integrator types.
    if constexpr (MUJOCO) {
        GRID_RBD_IT_DISPATCH_FFI_MJX_SS((int)it, launch_integrator_grad_kernel_jax, true, stream, batch, dt, gravity);
    } else {
        GRID_RBD_IT_DISPATCH_FFI_MJX((int)it, launch_integrator_grad_kernel_jax, false, stream, batch, dt, gravity);
    }
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("integrator_grad_kernel launch failed");

    cudaMemcpyAsync(dAB_out->typed_data(), g_data->d_dAB,
                    batch * (2 * nv) * (3 * nv) * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_integrator_gradient,
    grid_rbd_jax_integrator_gradient_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("dt").Attr<int64_t>("it").Attr<float>("gravity")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention integrator_gradient (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_integrator_gradient_mujoco,
    grid_rbd_jax_integrator_gradient_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("dt").Attr<int64_t>("it").Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_INTEGRATOR_GRADIENT


// ────────────────────────────────────────────────────────────────────────────
// JAX FFI: grid_plant surface (cost / barrier / plant-step)
// ────────────────────────────────────────────────────────────────────────────
//
// Mirrors the numpy grid_plant_* C-ABI (above) but device-resident on the JAX
// stream: stage the FFI input buffers D→D into the shared g_plant scratch
// (d_in_a/b/c), launch the SAME grid_plant::*_kernel the host wrapper uses, and
// copy the g_plant outputs (d_out/d_grad/d_hess) D→D into JAX's result buffers.
// The cost/barrier ops return (value, grad, hess[/hess_diag]) as 3 result
// buffers; plant_step / plant_step_gradient return a single buffer. Gated on the
// same GRID_PLANT_HAS_* defines as the C-ABI so a per-robot .so exports only the
// handlers whose kernels were emitted. Python-side reshapes mirror _handle.py.

// quadratic_{state,input}_cost(var, des, w) → (value, grad, hess).
// var/des/w are (B, N); value (B,1); grad (B, N); hess (B, N*N). STATE picks N.
template <bool STATE, bool MUJOCO = false>
static ffi::Error grid_rbd_jax_plant_quadratic_cost_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> var, ffi::Buffer<ffi::F32> des, ffi::Buffer<ffi::F32> w,
    ffi::ResultBuffer<ffi::F32> out, ffi::ResultBuffer<ffi::F32> grad,
    ffi::ResultBuffer<ffi::F32> hess)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    if (plant_alloc()) return ffi::Error::Internal("plant_alloc failed");
    const int N = STATE ? (grid::NUM_POS + grid::NUM_VEL) : grid::NUM_VEL;
    auto dims = var.dimensions();
    if (dims.size() != 2 || (int)dims[1] != N)
        return ffi::Error::InvalidArgument("quadratic_cost: var must be 2D (B, N)");
    int batch = (int)dims[0];
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("quadratic_cost: batch > max_batch");
    cudaMemcpyAsync(g_plant.d_in_a, var.typed_data(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, des.typed_data(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, w.typed_data(),   (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    dim3 grid_dim = grid_rbd_grid_for(batch);
    if (STATE) {
        // mjx (MUJOCO=true) is STATE-only: the state cost reframes (input cost stays pin).
        grid_plant::quadratic_state_cost_kernel<T, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
        if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("quadratic_state_cost_kernel launch failed");
    } else {
        grid_plant::quadratic_input_cost_kernel<T><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
        if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("quadratic_input_cost_kernel launch failed");
    }
    cudaMemcpyAsync(out->typed_data(),  g_plant.d_out,  (size_t)batch * sizeof(T),         cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad->typed_data(), g_plant.d_grad, (size_t)batch * N * sizeof(T),     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess->typed_data(), g_plant.d_hess, (size_t)batch * N * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

static ffi::Error grid_rbd_jax_plant_quadratic_state_cost_impl(
    cudaStream_t stream, ffi::Buffer<ffi::F32> var, ffi::Buffer<ffi::F32> des,
    ffi::Buffer<ffi::F32> w, ffi::ResultBuffer<ffi::F32> out,
    ffi::ResultBuffer<ffi::F32> grad, ffi::ResultBuffer<ffi::F32> hess) {
    return grid_rbd_jax_plant_quadratic_cost_impl<true>(stream, var, des, w, out, grad, hess);
}
static ffi::Error grid_rbd_jax_plant_quadratic_input_cost_impl(
    cudaStream_t stream, ffi::Buffer<ffi::F32> var, ffi::Buffer<ffi::F32> des,
    ffi::Buffer<ffi::F32> w, ffi::ResultBuffer<ffi::F32> out,
    ffi::ResultBuffer<ffi::F32> grad, ffi::ResultBuffer<ffi::F32> hess) {
    return grid_rbd_jax_plant_quadratic_cost_impl<false>(stream, var, des, w, out, grad, hess);
}

#define GRID_RBD_JAX_PLANT_COST_BIND(name, impl)                                  \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(name, impl,                                      \
        ffi::Ffi::Bind()                                                          \
            .Ctx<ffi::PlatformStream<cudaStream_t>>()                            \
            .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
            .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>())

GRID_RBD_JAX_PLANT_COST_BIND(grid_rbd_jax_plant_quadratic_state_cost,
                             grid_rbd_jax_plant_quadratic_state_cost_impl);
GRID_RBD_JAX_PLANT_COST_BIND(grid_rbd_jax_plant_quadratic_input_cost,
                             grid_rbd_jax_plant_quadratic_input_cost_impl);
#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention state cost (floating only): value invariant, grad covector-rotated,
// GN hess congruence (MUJOCO_OUTPUT=true). Input cost has no mjx variant (frame-invariant).
static ffi::Error grid_rbd_jax_plant_quadratic_state_cost_mujoco_impl(
    cudaStream_t stream, ffi::Buffer<ffi::F32> var, ffi::Buffer<ffi::F32> des,
    ffi::Buffer<ffi::F32> w, ffi::ResultBuffer<ffi::F32> out,
    ffi::ResultBuffer<ffi::F32> grad, ffi::ResultBuffer<ffi::F32> hess) {
    return grid_rbd_jax_plant_quadratic_cost_impl<true, true>(stream, var, des, w, out, grad, hess);
}
GRID_RBD_JAX_PLANT_COST_BIND(grid_rbd_jax_plant_quadratic_state_cost_mujoco,
                             grid_rbd_jax_plant_quadratic_state_cost_mujoco_impl);
#endif  // GRID_RBD_WITH_MUJOCO

// joint_{position,velocity,torque}_barrier(var, lower, upper; mu)
// → (value (B,1), grad (B,N), hess_diag (B,N)). POSITION uses NUM_POS else NUM_VEL.
template <int WHICH>  // 0=position, 1=velocity, 2=torque
static ffi::Error grid_rbd_jax_plant_barrier_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> var, ffi::Buffer<ffi::F32> lower, ffi::Buffer<ffi::F32> upper,
    ffi::ResultBuffer<ffi::F32> out, ffi::ResultBuffer<ffi::F32> grad,
    ffi::ResultBuffer<ffi::F32> hess_diag, float mu)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    if (plant_alloc()) return ffi::Error::Internal("plant_alloc failed");
    const int N = (WHICH == 0) ? grid::NUM_POS : grid::NUM_VEL;
    auto dims = var.dimensions();
    if (dims.size() != 2 || (int)dims[1] != N)
        return ffi::Error::InvalidArgument("barrier: var must be 2D (B, N)");
    int batch = (int)dims[0];
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("barrier: batch > max_batch");
    cudaMemcpyAsync(g_plant.d_in_a, var.typed_data(),   (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, lower.typed_data(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, upper.typed_data(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    dim3 grid_dim = grid_rbd_grid_for(batch);
    if (WHICH == 0)
        grid_plant::joint_position_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    else if (WHICH == 1)
        grid_plant::joint_velocity_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    else
        grid_plant::joint_torque_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("joint_barrier_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(),       g_plant.d_out,  (size_t)batch * sizeof(T),     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad->typed_data(),      g_plant.d_grad, (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess_diag->typed_data(), g_plant.d_hess, (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

static ffi::Error grid_rbd_jax_plant_joint_position_barrier_impl(
    cudaStream_t s, ffi::Buffer<ffi::F32> v, ffi::Buffer<ffi::F32> lo, ffi::Buffer<ffi::F32> hi,
    ffi::ResultBuffer<ffi::F32> o, ffi::ResultBuffer<ffi::F32> g, ffi::ResultBuffer<ffi::F32> h, float mu) {
    return grid_rbd_jax_plant_barrier_impl<0>(s, v, lo, hi, o, g, h, mu);
}
static ffi::Error grid_rbd_jax_plant_joint_velocity_barrier_impl(
    cudaStream_t s, ffi::Buffer<ffi::F32> v, ffi::Buffer<ffi::F32> lo, ffi::Buffer<ffi::F32> hi,
    ffi::ResultBuffer<ffi::F32> o, ffi::ResultBuffer<ffi::F32> g, ffi::ResultBuffer<ffi::F32> h, float mu) {
    return grid_rbd_jax_plant_barrier_impl<1>(s, v, lo, hi, o, g, h, mu);
}
static ffi::Error grid_rbd_jax_plant_joint_torque_barrier_impl(
    cudaStream_t s, ffi::Buffer<ffi::F32> v, ffi::Buffer<ffi::F32> lo, ffi::Buffer<ffi::F32> hi,
    ffi::ResultBuffer<ffi::F32> o, ffi::ResultBuffer<ffi::F32> g, ffi::ResultBuffer<ffi::F32> h, float mu) {
    return grid_rbd_jax_plant_barrier_impl<2>(s, v, lo, hi, o, g, h, mu);
}

#define GRID_RBD_JAX_PLANT_BARRIER_BIND(name, impl)                               \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(name, impl,                                      \
        ffi::Ffi::Bind()                                                          \
            .Ctx<ffi::PlatformStream<cudaStream_t>>()                            \
            .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>() \
            .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>() \
            .Attr<float>("mu"))

GRID_RBD_JAX_PLANT_BARRIER_BIND(grid_rbd_jax_plant_joint_position_barrier,
                                grid_rbd_jax_plant_joint_position_barrier_impl);
GRID_RBD_JAX_PLANT_BARRIER_BIND(grid_rbd_jax_plant_joint_velocity_barrier,
                                grid_rbd_jax_plant_joint_velocity_barrier_impl);
GRID_RBD_JAX_PLANT_BARRIER_BIND(grid_rbd_jax_plant_joint_torque_barrier,
                                grid_rbd_jax_plant_joint_torque_barrier_impl);

#ifdef GRID_PLANT_HAS_STEP
// plant_step(x, u; dt, it) → x_kp1  (B, NX). Reuses g_plant.d_grad as x_kp1
// (size NX), matching the C-ABI launch_plant_step.
template <grid::IntegratorType IT, bool MUJOCO>
static void launch_plant_step_jax(cudaStream_t stream, int batch, float gravity, float dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(batch);
    const size_t smem = grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_kernel<T, IT, MUJOCO>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_kernel<T, IT, MUJOCO>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::plant_step_kernel<T, IT, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr,
        smem, stream>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, grid::NUM_VEL, g_robot, (T)gravity, (T)dt, batch);
}

// MUJOCO=true launches the plant_step kernel with MUJOCO_OUTPUT=true (floating only).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_plant_step_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> x, ffi::Buffer<ffi::F32> u,
    ffi::ResultBuffer<ffi::F32> x_kp1,
    float dt, int64_t it, float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    if (plant_alloc()) return ffi::Error::Internal("plant_alloc failed");
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    auto dims = x.dimensions();
    if (dims.size() != 2 || (int)dims[1] != nx)
        return ffi::Error::InvalidArgument("plant_step: x must be 2D (B, NX)");
    int batch = (int)dims[0];
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("plant_step: batch > max_batch");
    cudaMemcpyAsync(g_plant.d_in_a, x.typed_data(), (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, u.typed_data(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    GRID_RBD_IT_DISPATCH_FFI_MJX((int)it, launch_plant_step_jax, MUJOCO, stream, batch, gravity, dt);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("plant_step_kernel launch failed");
    cudaMemcpyAsync(x_kp1->typed_data(), g_plant.d_grad, (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_plant_step,
    grid_rbd_jax_plant_step_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("dt").Attr<int64_t>("it").Attr<float>("gravity")
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_plant_step_mujoco,
    grid_rbd_jax_plant_step_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("dt").Attr<int64_t>("it").Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_PLANT_HAS_STEP

#ifdef GRID_PLANT_HAS_STEP_GRADIENT
// plant_step_gradient(x, u; dt, it) → dAB  (B, 2*NV*3*NV col-major). Reuses
// g_plant.d_grad as the dAB output (size 2*NV*3*NV), matching the C-ABI.
template <grid::IntegratorType IT, bool MUJOCO>
static void launch_plant_step_gradient_jax(cudaStream_t stream, int batch, float gravity, float dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(batch);
    const size_t smem = grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_gradient_kernel<T, IT, MUJOCO>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_gradient_kernel<T, IT, MUJOCO>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::plant_step_gradient_kernel<T, IT, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr,
        smem, stream>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, nv, g_robot, (T)gravity, (T)dt, batch);
}

// MUJOCO=true launches the plant_step_gradient kernel with MUJOCO_OUTPUT=true (floating only).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_plant_step_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> x, ffi::Buffer<ffi::F32> u,
    ffi::ResultBuffer<ffi::F32> dAB,
    float dt, int64_t it, float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    if (plant_alloc()) return ffi::Error::Internal("plant_alloc failed");
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    const int dab = 2 * nv * 3 * nv;
    auto dims = x.dimensions();
    if (dims.size() != 2 || (int)dims[1] != nx)
        return ffi::Error::InvalidArgument("plant_step_gradient: x must be 2D (B, NX)");
    int batch = (int)dims[0];
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("plant_step_gradient: batch > max_batch");
    cudaMemcpyAsync(g_plant.d_in_a, x.typed_data(), (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, u.typed_data(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    // mjx gradient is single-stage (euler/si) only; pin supports all integrator types.
    if constexpr (MUJOCO) {
        GRID_RBD_IT_DISPATCH_FFI_MJX_SS((int)it, launch_plant_step_gradient_jax, true, stream, batch, gravity, dt);
    } else {
        GRID_RBD_IT_DISPATCH_FFI_MJX((int)it, launch_plant_step_gradient_jax, false, stream, batch, gravity, dt);
    }
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("plant_step_gradient_kernel launch failed");
    cudaMemcpyAsync(dAB->typed_data(), g_plant.d_grad, (size_t)batch * dab * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_plant_step_gradient,
    grid_rbd_jax_plant_step_gradient_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("dt").Attr<int64_t>("it").Attr<float>("gravity")
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_plant_step_gradient_mujoco,
    grid_rbd_jax_plant_step_gradient_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("dt").Attr<int64_t>("it").Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_PLANT_HAS_STEP_GRADIENT

#ifdef GRID_PLANT_HAS_EE_COST
// ee_pos_cost(q, p_des, W) → (value (B,1), grad (B,NX), hess (B,NX*NX)).
// MUJOCO=true launches with MUJOCO_OUTPUT=true (floating only).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_plant_ee_pos_cost_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q, ffi::Buffer<ffi::F32> p_des, ffi::Buffer<ffi::F32> W,
    ffi::ResultBuffer<ffi::F32> out, ffi::ResultBuffer<ffi::F32> grad,
    ffi::ResultBuffer<ffi::F32> hess)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    if (plant_alloc()) return ffi::Error::Internal("plant_alloc failed");
    const int nq = grid::NUM_POS;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    auto dims = q.dimensions();
    if (dims.size() != 2 || (int)dims[1] != nq)
        return ffi::Error::InvalidArgument("ee_pos_cost: q must be 2D (B, NQ)");
    int batch = (int)dims[0];
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("ee_pos_cost: batch > max_batch");
    cudaMemcpyAsync(g_plant.d_in_a, q.typed_data(),     (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, p_des.typed_data(), (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, W.typed_data(),     (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    size_t smem = grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::ee_pos_cost_kernel<T, 0, MUJOCO>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::ee_pos_cost_kernel<T, /*EE=*/0, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose, g_plant.d_end_effector_pose_gradient, g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("ee_pos_cost launch failed");
    cudaMemcpyAsync(out->typed_data(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad->typed_data(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess->typed_data(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_PLANT_COST_BIND(grid_rbd_jax_plant_ee_pos_cost,
                             grid_rbd_jax_plant_ee_pos_cost_impl<false>);
#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_PLANT_COST_BIND(grid_rbd_jax_plant_ee_pos_cost_mujoco,
                             grid_rbd_jax_plant_ee_pos_cost_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_PLANT_HAS_EE_COST

#ifdef GRID_PLANT_HAS_COM_COST
// com_cost(q, p_des, W) → (value (B,1), grad (B,NX), hess (B,NX*NX)).
// MUJOCO=true launches with MUJOCO_OUTPUT=true (floating only).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_plant_com_cost_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q, ffi::Buffer<ffi::F32> p_des, ffi::Buffer<ffi::F32> W,
    ffi::ResultBuffer<ffi::F32> out, ffi::ResultBuffer<ffi::F32> grad,
    ffi::ResultBuffer<ffi::F32> hess)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    if (plant_alloc()) return ffi::Error::Internal("plant_alloc failed");
    const int nq = grid::NUM_POS;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    auto dims = q.dimensions();
    if (dims.size() != 2 || (int)dims[1] != nq)
        return ffi::Error::InvalidArgument("com_cost: q must be 2D (B, NQ)");
    int batch = (int)dims[0];
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("com_cost: batch > max_batch");
    cudaMemcpyAsync(g_plant.d_in_a, q.typed_data(),     (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, p_des.typed_data(), (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, W.typed_data(),     (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    size_t smem = grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::com_cost_kernel<T, MUJOCO>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::com_cost_kernel<T, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("com_cost launch failed");
    cudaMemcpyAsync(out->typed_data(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad->typed_data(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess->typed_data(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_PLANT_COST_BIND(grid_rbd_jax_plant_com_cost,
                             grid_rbd_jax_plant_com_cost_impl<false>);
#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_PLANT_COST_BIND(grid_rbd_jax_plant_com_cost_mujoco,
                             grid_rbd_jax_plant_com_cost_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_PLANT_HAS_COM_COST

#ifdef GRID_PLANT_HAS_MOMENTUM_COST
// momentum_cost(q, qd, h_des, W) → (value (B,1), grad (B,NX), hess (B,NX*NX)).
// h_des(6) and W(6) are packed into the two halves of d_in_c, matching the C-ABI.
// MUJOCO=true launches with MUJOCO_OUTPUT=true (floating only).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_plant_momentum_cost_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q, ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> h_des, ffi::Buffer<ffi::F32> W,
    ffi::ResultBuffer<ffi::F32> out, ffi::ResultBuffer<ffi::F32> grad,
    ffi::ResultBuffer<ffi::F32> hess)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    if (plant_alloc()) return ffi::Error::Internal("plant_alloc failed");
    const int nq = grid::NUM_POS;
    const int nv = grid::NUM_VEL;
    const int nx = nq + nv;
    auto dims = q.dimensions();
    if (dims.size() != 2 || (int)dims[1] != nq)
        return ffi::Error::InvalidArgument("momentum_cost: q must be 2D (B, NQ)");
    int batch = (int)dims[0];
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("momentum_cost: batch > max_batch");
    cudaMemcpyAsync(g_plant.d_in_a, q.typed_data(),  (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, qd.typed_data(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c,                  h_des.typed_data(), (size_t)batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c + (size_t)batch * 6, W.typed_data(),  (size_t)batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    size_t smem = grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(batch);
    // momentum_cost is register-heavy (~140 regs/thread): clamp to its launch cap.
    dim3 thr = grid_clamp_threads_for(grid_plant::momentum_cost_kernel<T, MUJOCO>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::momentum_cost_kernel<T, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b,
        g_plant.d_in_c, g_plant.d_in_c + (size_t)batch * 6,
        g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("momentum_cost_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad->typed_data(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess->typed_data(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_plant_momentum_cost,
    grid_rbd_jax_plant_momentum_cost_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_plant_momentum_cost_mujoco,
    grid_rbd_jax_plant_momentum_cost_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_PLANT_HAS_MOMENTUM_COST

// ─── P-tier1: centroidal / energy / kinematics family (jax FFI) ──────────────
//
// These 13 handlers mirror the crba template EXACTLY: stage the device-resident
// input(s) into the singleton's d_q_qd_u buffer (q at offset 0, qd at offset nj)
// via cudaMemcpy2DAsync, launch the per-robot __global__ kernel DIRECTLY on JAX's
// stream (no host round-trip), then D→D copy the flat result into JAX's output.
//
// Input layout (UNIFIED with crba / end_effector_pose): every kernel reads its
// inputs from d_q_qd_u with stride = 3*NUM_JOINTS. The compressed-d_q kernels
// (com/dccrba/pe_regressor/frame_jacobian/osc_inertia) only touch the first
// NUM_JOINTS elements of each strided block, so packing q into the q-slot of
// d_q_qd_u and passing stride=3*NJ feeds them the correct q — identical to how
// the end_effector_pose handler launches the compressed-d_q ee kernel.
//
// d_workspace: passed ONLY for gravity / nle / dccrba / cmm / coriolis (per the
// contract); OMITTED for the others (their kernels have no workspace arg).
//
// R2 (>48KB dynamic smem opt-in): NOT handled per-handler. grid_rbd_init() calls
// grid::init_grid<T>() → init_grid_kernel_attrs<T>(), which issues the
// cudaFuncSetAttribute(MaxDynamicSharedMemorySize) opt-in for EVERY emitted
// algorithm kernel (coriolis/cmm/dccrba/com/ccrba/energy/... all enumerated
// there). The crba/idsva_so handlers rely on the same warmup; these do too.

#if GRID_HAS_GENERALIZED_GRAVITY
// generalized_gravity(q) → g(q), NV  [d_workspace, gravity]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_generalized_gravity_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "generalized_gravity: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("generalized_gravity: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch,
                      q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::generalized_gravity_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_c, g_data->d_workspace, g_data->d_q_qd_u, stride_q_qd, g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("generalized_gravity_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(), g_data->d_c, batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_generalized_gravity,
    grid_rbd_jax_generalized_gravity_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_generalized_gravity_mujoco,
    grid_rbd_jax_generalized_gravity_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_GENERALIZED_GRAVITY


#if GRID_HAS_NONLINEAR_EFFECTS
// nonlinear_effects(q, qd) → c(q,qd), NV  [d_workspace, gravity]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_nonlinear_effects_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::ResultBuffer<ffi::F32> out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "nonlinear_effects: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("nonlinear_effects: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],  dst_pitch, q.typed_data(),  row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::nonlinear_effects_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_c, g_data->d_workspace, g_data->d_q_qd_u, stride_q_qd, g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("nonlinear_effects_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(), g_data->d_c, batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_nonlinear_effects,
    grid_rbd_jax_nonlinear_effects_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_nonlinear_effects_mujoco,
    grid_rbd_jax_nonlinear_effects_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_NONLINEAR_EFFECTS


#if GRID_HAS_CORIOLIS_MATRIX
// coriolis_matrix(q, qd) → C(q,qd), NV*NV row-major  [d_workspace, gravity]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_coriolis_matrix_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::ResultBuffer<ffi::F32> out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "coriolis_matrix: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("coriolis_matrix: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],  dst_pitch, q.typed_data(),  row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::coriolis_matrix_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::CORIOLIS_MATRIX_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_coriolis, g_data->d_workspace, g_data->d_q_qd_u, stride_q_qd, g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("coriolis_matrix_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(), g_data->d_coriolis, batch * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_coriolis_matrix,
    grid_rbd_jax_coriolis_matrix_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_coriolis_matrix_mujoco,
    grid_rbd_jax_coriolis_matrix_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_CORIOLIS_MATRIX


#if GRID_HAS_KINETIC_ENERGY_REGRESSOR
// kinetic_energy_regressor(q, qd) → y_KE, 10*NUM_BODIES  [gravity, no workspace]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_kinetic_energy_regressor_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::ResultBuffer<ffi::F32> out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "kinetic_energy_regressor: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nb = grid::NUM_BODIES;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("kinetic_energy_regressor: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],  dst_pitch, q.typed_data(),  row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::kinetic_energy_regressor_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::KINETIC_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_ke_regressor, g_data->d_q_qd_u, stride_q_qd, g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("kinetic_energy_regressor_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(), g_data->d_ke_regressor, batch * 10 * nb * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_kinetic_energy_regressor,
    grid_rbd_jax_kinetic_energy_regressor_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_kinetic_energy_regressor_mujoco,
    grid_rbd_jax_kinetic_energy_regressor_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_KINETIC_ENERGY_REGRESSOR


#if GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
// potential_energy_regressor(q) → y_PE, 10*NUM_BODIES  [gravity, no workspace]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_potential_energy_regressor_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "potential_energy_regressor: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nb = grid::NUM_BODIES;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("potential_energy_regressor: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride_q = 3 * grid::NUM_JOINTS;
    grid::potential_energy_regressor_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::POTENTIAL_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_pe_regressor, g_data->d_q_qd_u, stride_q, g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("potential_energy_regressor_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(), g_data->d_pe_regressor, batch * 10 * nb * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_potential_energy_regressor,
    grid_rbd_jax_potential_energy_regressor_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_potential_energy_regressor_mujoco,
    grid_rbd_jax_potential_energy_regressor_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_POTENTIAL_ENERGY_REGRESSOR


// ─── Wave 2: gated value ops (energy/com/ccrba/cmm/dccrba) ───────────────────

#ifdef GRID_HAS_ENERGY
// energy(q, qd) → [KE, PE, KE+PE], 3  [gravity, no workspace]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_energy_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::ResultBuffer<ffi::F32> out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "energy: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("energy: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],  dst_pitch, q.typed_data(),  row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::energy_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::ENERGY_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_energy, g_data->d_workspace, g_data->d_q_qd_u, stride_q_qd, g_robot, /*gravity=*/gravity, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("energy_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(), g_data->d_energy, batch * 3 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_energy,
    grid_rbd_jax_energy_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_energy_mujoco,
    grid_rbd_jax_energy_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_ENERGY


#ifdef GRID_HAS_COM
// com(q) → flat [p_com(3); J_com(3*NV)], 3 + 3*NV  [no workspace, no gravity]
// Single flat buffer; the Python layer splits the (p_com, J_com) tuple.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_com_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> out)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "com: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("com: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride_q = 3 * grid::NUM_JOINTS;
    grid::com_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_com, g_data->d_workspace, g_data->d_q_qd_u, stride_q, g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("com_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(), g_data->d_com, batch * (3 + 3 * nv) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_com,
    grid_rbd_jax_com_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_com_mujoco,
    grid_rbd_jax_com_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_COM


#ifdef GRID_HAS_CCRBA
// ccrba(q, qd) → flat [A(6*NV); h(6)], 6*NV + 6  [no workspace, no gravity]
// Single flat buffer; the Python layer splits the (A, h) tuple.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_ccrba_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::ResultBuffer<ffi::F32> out)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "ccrba: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("ccrba: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],  dst_pitch, q.typed_data(),  row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::ccrba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_ccrba, g_data->d_workspace, g_data->d_q_qd_u, stride_q_qd, g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("ccrba_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(), g_data->d_ccrba, batch * (6 * nv + 6) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_ccrba,
    grid_rbd_jax_ccrba_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_ccrba_mujoco,
    grid_rbd_jax_ccrba_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_CCRBA


#ifdef GRID_HAS_CMM_TIME_VARIATION
// cmm_time_variation(q, qd) → Adot, 6*NV  [d_workspace, no gravity]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_cmm_time_variation_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::ResultBuffer<ffi::F32> out)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "cmm_time_variation: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("cmm_time_variation: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],  dst_pitch, q.typed_data(),  row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::cmm_time_variation_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::CMM_TIME_VARIATION_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER>(), stream>>>(
            g_data->d_cmm_time_variation, g_data->d_workspace, g_data->d_q_qd_u, stride_q_qd, g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("cmm_time_variation_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(), g_data->d_cmm_time_variation, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_cmm_time_variation,
    grid_rbd_jax_cmm_time_variation_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_cmm_time_variation_mujoco,
    grid_rbd_jax_cmm_time_variation_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_CMM_TIME_VARIATION


#ifdef GRID_HAS_DCCRBA
// dccrba(q) → dA/dq, 6*NV*NV  [d_workspace, no gravity]  (compressed d_q kernel)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_dccrba_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> out)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "dccrba: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("dccrba: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride_q = 3 * grid::NUM_JOINTS;
    grid::dccrba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::DCCRBA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER>(), stream>>>(
            g_data->d_dccrba, g_data->d_workspace, g_data->d_q_qd_u, stride_q, g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("dccrba_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(), g_data->d_dccrba, batch * 6 * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_dccrba,
    grid_rbd_jax_dccrba_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_dccrba_mujoco,
    grid_rbd_jax_dccrba_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_DCCRBA


// ─── Wave 3: int-attr kinematics (frame_jacobian/dot, osc_inertia) ───────────
//
// frame_jacobian / frame_jacobian_dot take target_jid + reference_frame as
// runtime int64 FFI attrs (cast to int for the kernel). UNLIKE the C-ABI host
// wrapper, these handlers do NOT resolve a negative "use default" sentinel: the
// kernel itself does not interpret target_jid<0 / reference_frame<0, so the
// Python jax surface must pass already-resolved (non-negative) values (it reads
// the leaf-EE / LOCAL_WORLD_ALIGNED defaults from the numpy handle). reference_frame:
// 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED.

#ifdef GRID_HAS_FRAME_JACOBIAN
// frame_jacobian(q) → 6*NV geometric Jacobian  [no workspace, no gravity, int attrs]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_frame_jacobian_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> out,
    int64_t target_jid, int64_t reference_frame)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "frame_jacobian: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("frame_jacobian: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride_q = 3 * grid::NUM_JOINTS;
    grid::frame_jacobian_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::FRAME_JACOBIAN_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_frame_jacobian, g_data->d_q_qd_u, stride_q,
            (int)target_jid, (int)reference_frame, g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("frame_jacobian_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(), g_data->d_frame_jacobian, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_frame_jacobian,
    grid_rbd_jax_frame_jacobian_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("target_jid").Attr<int64_t>("reference_frame")
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_frame_jacobian_mujoco,
    grid_rbd_jax_frame_jacobian_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("target_jid").Attr<int64_t>("reference_frame")
);
#endif  // GRID_RBD_WITH_MUJOCO


// frame_jacobian_dot(q, qd) → 6*NV d/dt Jacobian  [no workspace, no gravity, int attrs]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_frame_jacobian_dot_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::ResultBuffer<ffi::F32> out,
    int64_t target_jid, int64_t reference_frame)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "frame_jacobian_dot: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("frame_jacobian_dot: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],  dst_pitch, q.typed_data(),  row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::frame_jacobian_dot_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::FRAME_JACOBIAN_DOT_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_frame_jacobian_dot, g_data->d_q_qd_u, stride_q_qd,
            (int)target_jid, (int)reference_frame, g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("frame_jacobian_dot_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(), g_data->d_frame_jacobian_dot, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_frame_jacobian_dot,
    grid_rbd_jax_frame_jacobian_dot_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("target_jid").Attr<int64_t>("reference_frame")
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_frame_jacobian_dot_mujoco,
    grid_rbd_jax_frame_jacobian_dot_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("target_jid").Attr<int64_t>("reference_frame")
);
#endif  // GRID_RBD_WITH_MUJOCO


// osc_inertia(q) → 6x6 task inertia Lambda, 36  [d_workspace (tier-spill scratch), no gravity, frame baked at codegen]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_osc_inertia_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> out)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "osc_inertia: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("osc_inertia: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride_q = 3 * grid::NUM_JOINTS;
    grid::osc_inertia_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::OSC_INERTIA_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_osc_inertia, g_data->d_workspace, g_data->d_q_qd_u, stride_q, g_robot, batch);
    if (cudaGetLastError() != cudaSuccess) return ffi::Error::Internal("osc_inertia_kernel launch failed");
    cudaMemcpyAsync(out->typed_data(), g_data->d_osc_inertia, batch * 36 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_osc_inertia,
    grid_rbd_jax_osc_inertia_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_osc_inertia_mujoco,
    grid_rbd_jax_osc_inertia_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_FRAME_JACOBIAN

#endif  // GRID_RBD_WITH_JAX


// ────────────────────────────────────────────────────────────────────────────
// PyTorch custom ops (D.3)
// ────────────────────────────────────────────────────────────────────────────
//
// Gated on -DGRID_RBD_WITH_TORCH (set by _compile.py when torch is installed at
// register_robot time). Structurally identical to the JAX FFI block: each op
//   1. asserts CUDA / contiguous / float32 / (B, NJ),
//   2. grabs the current torch CUDA stream,
//   3. lazily grid_rbd_init() (same guard as the JAX handlers),
//   4. D->D repacks inputs into g_data->d_q_qd_u via the same cudaMemcpy2DAsync
//      pitch trick,
//   5. allocates the output with torch::empty on the same device,
//   6. launches the SAME kernel kernel-direct on the stream (matching smem),
//   7. D->D copies the singleton output buffer into the output tensor,
//   8. returns the tensor with NO host sync — async / stream-ordered so it is
//      CUDA-graph-capturable. (Reshapes to the _handle.py conventions are done
//      Python-side, exactly like the JAX surface.)
//
// Registered under a per-robot op namespace keyed by the cache_key
// (-DGRID_RBD_TORCH_KEY=<hex>) so two robots in one process don't collide.

// fp64: torch surface is fp32-only too (see the JAX note above) — suppress in an fp64 .so.
#if defined(GRID_RBD_WITH_TORCH) && !defined(GRID_WRAPPER_T_DOUBLE)

// Use the Python-free C++ frontend (torch/library.h) rather than
// torch/extension.h, which pulls in <Python.h>. We register ops via the
// TORCH_LIBRARY dispatcher and load them with torch.ops.load_library — no
// pybind/Python C-API needed in the .so.
#include <torch/library.h>
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

namespace {

// ── input validation + D->D pack helpers (mirror the JAX handlers) ──
static inline void grid_torch_check(const torch::Tensor& t, const char* name, int last_dim) {
    TORCH_CHECK(t.is_cuda(), name, ": must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, ": must be contiguous");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, ": must be float32");
    TORCH_CHECK(t.dim() == 2, name, ": must be 2D (B, ", last_dim, ")");
    TORCH_CHECK(t.size(1) == last_dim, name, ": last dim != ", last_dim);
}

static inline int grid_torch_batch(const torch::Tensor& q) {
    int batch = (int)q.size(0);
    TORCH_CHECK(batch <= kMaxBatch, "batch ", batch, " > compiled-in max_batch ", kMaxBatch);
    return batch;
}

static inline torch::Tensor grid_torch_empty(int rows, int cols, const torch::Tensor& like) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(like.device());
    return torch::empty({rows, cols}, opts);
}

// pack q[, qd[, u]] D->D into d_q_qd_u on `stream` (layout [q,qd,u] per ts).
static inline void grid_torch_pack(cudaStream_t stream, int batch, int nj,
                                   const torch::Tensor* q,
                                   const torch::Tensor* qd,
                                   const torch::Tensor* u) {
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    if (q)  cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],      dst_pitch, q->data_ptr<float>(),  row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    if (qd) cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj],     dst_pitch, qd->data_ptr<float>(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    if (u)  cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj],   dst_pitch, u->data_ptr<float>(),  row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
}

static inline void grid_torch_init_or_throw() {
    if (!g_data) { int rc = grid_rbd_init(); TORCH_CHECK(rc == 0, "grid_rbd_init failed"); }
}

// Optional external-force application (stream-ordered, graph-capturable).
// f_ext (if present) is (batch, 6*NUM_BODIES) float32 CUDA, body-major,
// [angular; linear] local-frame — same layout as d_f_ext and the numpy
// surface. Copies D->D into the singleton's d_f_ext on `stream`; pair with
// grid_torch_f_ext_reset() AFTER the kernel launch (also on `stream`) so a
// later no-f_ext call sees the zeroed buffer. A null/absent f_ext is a no-op,
// keeping the no-f_ext path byte-identical and capture-clean.
static inline void grid_torch_f_ext_apply(cudaStream_t stream, int batch,
                                          const c10::optional<torch::Tensor>& f_ext) {
    if (!f_ext.has_value()) return;
    const torch::Tensor& fe = f_ext.value();
    const int row = 6 * grid::NUM_BODIES;
    grid_torch_check(fe, "f_ext", row);
    cudaMemcpyAsync(g_data->d_f_ext, fe.data_ptr<float>(),
                    (size_t)batch * row * sizeof(T), cudaMemcpyDeviceToDevice, stream);
}

static inline void grid_torch_f_ext_reset(cudaStream_t stream, int batch,
                                          const c10::optional<torch::Tensor>& f_ext) {
    if (!f_ext.has_value()) return;
    const int row = 6 * grid::NUM_BODIES;
    cudaMemsetAsync(g_data->d_f_ext, 0, (size_t)batch * row * sizeof(T), stream);
}

// ── forward ops ──

// qdd wiring (torch): when a qdd tensor is provided, copy it D→D into the
// separate d_qdd buffer and launch the USE_QDD overload of the kernel (which
// reads the acceleration from d_qdd; signature adds d_qdd after stride). A null
// qdd keeps the (faster) qdd=0 overload. Mirrors the numpy / JAX ID paths.
#if GRID_HAS_INVERSE_DYNAMICS
template <bool MUJOCO>
torch::Tensor torch_inverse_dynamics(torch::Tensor q, torch::Tensor qd, double gravity,
                         c10::optional<torch::Tensor> qdd,
                         c10::optional<torch::Tensor> f_ext) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "inverse_dynamics: q", nj); grid_torch_check(qd, "inverse_dynamics: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, nullptr);
    grid_torch_f_ext_apply(stream, batch, f_ext);
    auto out = grid_torch_empty(batch, nj, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    if (qdd.has_value()) {
        const torch::Tensor& a = qdd.value();
        grid_torch_check(a, "inverse_dynamics: qdd", nj);
        cudaMemcpyAsync(g_data->d_qdd, a.data_ptr<float>(),
                        (size_t)batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        grid::inverse_dynamics_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS>(), grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_c, g_data->d_q_qd_u, stride, g_data->d_qdd, g_data->d_f_ext, g_robot, (T)gravity, batch);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "inverse_dynamics_kernel launch failed");
    } else if constexpr (MUJOCO) {
        // The bias (no-qdd) kernel overload is pinocchio-only (MUJOCO_OUTPUT lives on
        // the qdd-input overload). For mjx with no qdd, zero d_qdd and use the
        // MUJOCO-capable qdd overload (the kernel converts the mjx qacc=0 input to the
        // correct pin acceleration) — matching the JAX handler, which always passes qdd.
        cudaMemsetAsync(g_data->d_qdd, 0, (size_t)batch * nj * sizeof(T), stream);
        grid::inverse_dynamics_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER, /*MUJOCO_OUTPUT=*/true><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS>(), grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_c, g_data->d_q_qd_u, stride, g_data->d_qdd, g_data->d_f_ext, g_robot, (T)gravity, batch);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "inverse_dynamics_kernel launch failed");
    } else {
        grid::inverse_dynamics_kernel<T><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS>(), grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_c, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "inverse_dynamics_kernel launch failed");
    }
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_c, batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(stream, batch, f_ext);
    return out;
}
#endif  // GRID_HAS_INVERSE_DYNAMICS

#if GRID_HAS_MINV
template <bool MUJOCO>
torch::Tensor torch_minv(torch::Tensor q) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    grid_torch_check(q, "minv: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    // Minv is nv x nv (tangent-space); the kernel writes d_Minv nv*nv-strided.
    // Size the output + copy at nv*nv (unified with numpy/JAX). FIXED base: nv == nj.
    auto out = grid_torch_empty(batch, nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::minv_kernel<T, grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_MINV>(), grid::MINV_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER>(), stream>>>(
        g_data->d_Minv, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "minv_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_Minv, batch * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_MINV

#if GRID_HAS_FORWARD_DYNAMICS
template <bool MUJOCO>
torch::Tensor torch_forward_dynamics(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double gravity,
                                     c10::optional<torch::Tensor> f_ext) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "fd: q", nj); grid_torch_check(qd, "fd: qd", nj); grid_torch_check(u, "fd: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, &u);
    grid_torch_f_ext_apply(stream, batch, f_ext);
    auto out = grid_torch_empty(batch, nj, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::forward_dynamics_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_FORWARD_DYNAMICS>(), grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER>(), stream>>>(
        g_data->d_qdd, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "forward_dynamics_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_qdd, batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(stream, batch, f_ext);
    return out;
}
#endif  // GRID_HAS_FORWARD_DYNAMICS

#if GRID_HAS_ABA
template <bool MUJOCO>
torch::Tensor torch_aba(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double gravity,
                        c10::optional<torch::Tensor> f_ext) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "aba: q", nj); grid_torch_check(qd, "aba: qd", nj); grid_torch_check(u, "aba: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, &u);
    grid_torch_f_ext_apply(stream, batch, f_ext);
    auto out = grid_torch_empty(batch, nj, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::aba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_ABA>(), grid::ABA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER>(), stream>>>(
        g_data->d_qdd, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "aba_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_qdd, batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(stream, batch, f_ext);
    return out;
}
#endif  // GRID_HAS_ABA

#if GRID_HAS_CRBA
template <bool MUJOCO>
torch::Tensor torch_crba(torch::Tensor q, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    grid_torch_check(q, "crba: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    // M is nv x nv (tangent-space); the kernel writes d_M nv*nv-strided. Size the
    // output + copy at nv*nv (unified with numpy/JAX). FIXED base: nv == nj.
    auto out = grid_torch_empty(batch, nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::crba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_CRBA>(), grid::CRBA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER>(), stream>>>(
        g_data->d_M, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "crba_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_M, batch * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_CRBA

#if GRID_HAS_END_EFFECTOR_POSE
template <bool MUJOCO>
torch::Tensor torch_end_effector_pose(torch::Tensor q) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nee = GRID_RBD_NUM_EES;
    grid_torch_check(q, "end_effector_pose: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 6 * nee, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::GRID_RBD_EE_POSE_KERNEL<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_END_EFFECTOR_POSE>(), grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_end_effector_pose, g_data->d_q_qd_u, stride, g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "GRID_RBD_EE_POSE_KERNEL launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_end_effector_pose, batch * 6 * nee * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_END_EFFECTOR_POSE

#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
template <bool MUJOCO>
torch::Tensor torch_end_effector_pose_gradient(torch::Tensor q) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL, nee = GRID_RBD_NUM_EES;
    grid_torch_check(q, "end_effector_pose_gradient: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 6 * nee * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::GRID_RBD_EE_POSE_GRADIENT_KERNEL<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>(), grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER>(), stream>>>(
        g_data->d_end_effector_pose_gradient, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "GRID_RBD_EE_POSE_GRADIENT_KERNEL launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_end_effector_pose_gradient, batch * 6 * nee * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_END_EFFECTOR_POSE_GRADIENT

#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN
template <bool MUJOCO>
torch::Tensor torch_end_effector_pose_hessian(torch::Tensor q) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL, nee = GRID_RBD_NUM_EES;
    grid_torch_check(q, "end_effector_pose_hessian: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 6 * nee * nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::GRID_RBD_EE_POSE_HESSIAN_KERNEL<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>(), grid::END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER>(), stream>>>(
        g_data->d_end_effector_pose_hessian, g_data->d_end_effector_pose_gradient, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "GRID_RBD_EE_POSE_HESSIAN_KERNEL launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_end_effector_pose_hessian, batch * 6 * nee * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_END_EFFECTOR_POSE_HESSIAN

// ─── runtime-target multi-EE pose / pose-gradient (single-target torch ops) ──
// SINGLE target jid + a SINGLE 16-float col-major 4x4 SE(3) tool transform per call
// (the Python wrapper loops the resolved jid list + stacks, mirroring _handle.py).
// The transform is a 16-element CUDA float Tensor; its data_ptr is ALREADY a
// contiguous device pointer to 16 T, so it is handed straight to the kernel's
// d_Xtool (no host staging). target_jid is an absolute jid (the Python layer
// resolves names→jids). Gated like the C-ABI wrappers.
#ifdef GRID_HAS_END_EFFECTOR_POSE_RUNTIME
template <bool MUJOCO>
torch::Tensor torch_end_effector_pose_runtime(torch::Tensor q, int64_t target_jid, torch::Tensor offset) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "end_effector_pose_runtime: q", nj);
    TORCH_CHECK(offset.is_cuda() && offset.numel() == 16,
                "end_effector_pose_runtime: offset must be the 16-element CUDA tensor "
                "holding the col-major 4x4 SE(3) tool transform");
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto off = offset.to(torch::kFloat32).contiguous();
    auto out = grid_torch_empty(batch, 6, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::end_effector_pose_runtime_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::END_EFFECTOR_POSE_RUNTIME_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_eePose, g_data->d_q_qd_u, stride, (int)target_jid, off.data_ptr<float>(), g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "end_effector_pose_runtime_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_eePose, batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_END_EFFECTOR_POSE_RUNTIME

#ifdef GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME
template <bool MUJOCO>
torch::Tensor torch_end_effector_pose_gradient_runtime(torch::Tensor q, int64_t target_jid, torch::Tensor offset) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "end_effector_pose_gradient_runtime: q", nj);
    TORCH_CHECK(offset.is_cuda() && offset.numel() == 16,
                "end_effector_pose_gradient_runtime: offset must be the 16-element CUDA tensor "
                "holding the col-major 4x4 SE(3) tool transform");
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto off = offset.to(torch::kFloat32).contiguous();
    auto out = grid_torch_empty(batch, 6 * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::end_effector_pose_gradient_runtime_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::END_EFFECTOR_POSE_GRADIENT_RUNTIME_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_eePoseGrad, g_data->d_q_qd_u, stride, (int)target_jid, off.data_ptr<float>(), g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "end_effector_pose_gradient_runtime_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_eePoseGrad, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME

// qdd wiring (torch grad): ∂c/∂(q,qd) depends on qdd via the M·qdd term's
// derivatives. When a qdd tensor is provided, copy it D→D into d_qdd and launch
// the USE_QDD overload of the gradient kernel (signature adds d_qdd after
// stride). A null qdd keeps the (faster) qdd=0 overload — byte-identical to the
// prior behaviour. Mirrors the numpy / JAX ID-gradient paths and the order of
// the VALUE torch_inverse_dynamics op (q, qd, gravity, qdd, f_ext).
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
template <bool MUJOCO>
torch::Tensor torch_inverse_dynamics_gradient(torch::Tensor q, torch::Tensor qd, double gravity,
                              c10::optional<torch::Tensor> qdd,
                              c10::optional<torch::Tensor> f_ext) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    grid_torch_check(q, "inverse_dynamics_gradient: q", nj); grid_torch_check(qd, "inverse_dynamics_gradient: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, nullptr);
    grid_torch_f_ext_apply(stream, batch, f_ext);
    // dc_du is nv x 2nv (tangent-space); the kernel writes d_dc_du 2*nv*nv-strided.
    // Size + copy at 2*nv*nv (unified with numpy/JAX). FIXED base: nv == nj.
    auto out = grid_torch_empty(batch, 2 * nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    if (qdd.has_value()) {
        const torch::Tensor& a = qdd.value();
        grid_torch_check(a, "inverse_dynamics_gradient: qdd", nj);
        cudaMemcpyAsync(g_data->d_qdd, a.data_ptr<float>(),
                        (size_t)batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        grid::inverse_dynamics_gradient_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(), grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(), stream>>>(
            g_data->d_dc_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_qdd, g_data->d_f_ext, g_robot, (T)gravity, batch);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "inverse_dynamics_gradient_kernel launch failed");
    } else if constexpr (MUJOCO) {
        // mjx: the bias (no-qdd) gradient overload is pinocchio-only; zero d_qdd and
        // use the MUJOCO-capable qdd overload (matches the JAX handler, which always
        // passes qdd).
        cudaMemsetAsync(g_data->d_qdd, 0, (size_t)batch * nj * sizeof(T), stream);
        grid::inverse_dynamics_gradient_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/true><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(), grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(), stream>>>(
            g_data->d_dc_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_qdd, g_data->d_f_ext, g_robot, (T)gravity, batch);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "inverse_dynamics_gradient_kernel launch failed");
    } else {
        grid::inverse_dynamics_gradient_kernel<T><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(), grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_dc_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "inverse_dynamics_gradient_kernel launch failed");
    }
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_dc_du, batch * 2 * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(stream, batch, f_ext);
    return out;
}
#endif  // GRID_HAS_INVERSE_DYNAMICS_GRADIENT

#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
template <bool MUJOCO>
torch::Tensor torch_forward_dynamics_gradient(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double gravity,
                                          c10::optional<torch::Tensor> f_ext) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    grid_torch_check(q, "fd_grad: q", nj); grid_torch_check(qd, "fd_grad: qd", nj); grid_torch_check(u, "fd_grad: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, &u);
    grid_torch_f_ext_apply(stream, batch, f_ext);
    // df_du is nv x 2nv (tangent-space); the kernel writes d_df_du 2*nv*nv-strided.
    // Size + copy at 2*nv*nv (unified with numpy/JAX). FIXED base: nv == nj.
    auto out = grid_torch_empty(batch, 2 * nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::forward_dynamics_gradient_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>(), grid::FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER>(), stream>>>(
        g_data->d_df_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "forward_dynamics_gradient_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_df_du, batch * 2 * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(stream, batch, f_ext);
    return out;
}
#endif  // GRID_HAS_FORWARD_DYNAMICS_GRADIENT

#if GRID_HAS_IDSVA_SO
template <bool MUJOCO>
torch::Tensor torch_idsva_so(torch::Tensor q, torch::Tensor qd, torch::Tensor qdd, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "idsva_so: q", nj); grid_torch_check(qd, "idsva_so: qd", nj);
    grid_torch_check(qdd, "idsva_so: qdd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // qdd is packed into the acceleration (u) slot, read by the kernel as s_qdd
    // (mirrors numpy pack_q_qd_u(q, qd, qdd)). The Python surface passes explicit
    // zeros for the default so we never read a stale device buffer.
    grid_torch_pack(stream, batch, nj, &q, &qd, &qdd);
    auto out = grid_torch_empty(batch, grid::SECOND_ORDER_TENSOR_SIZE, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    // Compile-time frame dispatch (mirrors the JAX handler): floating-base .so call
    // the world-frame kernel (which carries MUJOCO_OUTPUT); fixed-base .so call the
    // body-frame kernel (pinocchio-only — mjx is floating-base-only).
#ifdef GRID_RBD_WITH_MUJOCO
    grid::idsva_so_world_frame_kernel<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>(), grid::IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>::TIER>(), stream>>>(
        g_data->d_idsva_so, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "idsva_so_world_frame_kernel launch failed");
#else
    static_assert(!MUJOCO, "mjx idsva_so is floating-base only");
    grid::idsva_so_body_frame_kernel<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_BODY_FRAME>::TIER><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_IDSVA_SO_BODY_FRAME>(), grid::IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_BODY_FRAME>::TIER>(), stream>>>(
        g_data->d_idsva_so, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "idsva_so_body_frame_kernel launch failed");
#endif
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_idsva_so, batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_IDSVA_SO

#if GRID_HAS_FDSVA_SO
template <bool MUJOCO>
torch::Tensor torch_fdsva_so(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "fdsva_so: q", nj); grid_torch_check(qd, "fdsva_so: qd", nj); grid_torch_check(u, "fdsva_so: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, &u);
    auto out = grid_torch_empty(batch, grid::SECOND_ORDER_TENSOR_SIZE, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::fdsva_so_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_FDSVA_SO>(), grid::FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER>(), stream>>>(
        g_data->d_df2, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_idsva_so, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "fdsva_so_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_df2, batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_FDSVA_SO

// ── inertial-parameter (sysID) regressor + FD parameter gradient ──
// Mirror the JAX grid_rbd_jax_inverse_dynamics_regressor /
// grid_rbd_jax_forward_dynamics_parameter_gradient handlers: same kernels, same
// d_Y / d_dqdd_dpi / d_workspace scratch, same (B, NV*10*NUM_BODIES) row-major
// output. These back the torch inertial-parameter VJP (tau = Y . pi so
// dtau/dpi = Y, dqdd/dpi = -Minv . Y).

#if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
template <bool MUJOCO>
torch::Tensor torch_inverse_dynamics_regressor(torch::Tensor q, torch::Tensor qd, torch::Tensor qdd, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "inverse_dynamics_regressor: q", nj);
    grid_torch_check(qd, "inverse_dynamics_regressor: qd", nj);
    grid_torch_check(qdd, "inverse_dynamics_regressor: qdd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // qdd occupies the u-slot (read as the acceleration; mirrors the JAX handler).
    grid_torch_pack(stream, batch, nj, &q, &qd, &qdd);
    const int out_size = grid::NUM_VEL * 10 * grid::NUM_BODIES;
    auto out = grid_torch_empty(batch, out_size, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::inverse_dynamics_regressor_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_Y, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "inverse_dynamics_regressor_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_Y, (size_t)batch * out_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_INVERSE_DYNAMICS_REGRESSOR

#if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
torch::Tensor torch_forward_dynamics_parameter_gradient(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "forward_dynamics_parameter_gradient: q", nj);
    grid_torch_check(qd, "forward_dynamics_parameter_gradient: qd", nj);
    grid_torch_check(u, "forward_dynamics_parameter_gradient: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, &u);
    const int out_size = grid::NUM_VEL * 10 * grid::NUM_BODIES;
    auto out = grid_torch_empty(batch, out_size, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::forward_dynamics_parameter_gradient_kernel<T><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::FORWARD_DYNAMICS_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_dqdd_dpi, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "forward_dynamics_parameter_gradient_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_dqdd_dpi, (size_t)batch * out_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT

// torch-local integrator-type dispatch (self-contained; the JAX variant lives
// inside the JAX #ifdef and its default branch returns ffi::Error).
#define GRID_RBD_IT_DISPATCH_TORCH(it_code, FN, ...)                              \
    switch (it_code) {                                                            \
        case 0: FN<grid::IntegratorType::EULER>(__VA_ARGS__); break;              \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER>(__VA_ARGS__); break;\
        case 2: FN<grid::IntegratorType::MIDPOINT>(__VA_ARGS__); break;           \
        case 3: FN<grid::IntegratorType::RK3>(__VA_ARGS__); break;                \
        case 4: FN<grid::IntegratorType::RK4>(__VA_ARGS__); break;                \
        default: TORCH_CHECK(false, "integrator: bad integrator-type code");      \
    }

// mjx-aware torch integrator-type dispatch: instantiates FN<IT_case, MUJOCO_FLAG>.
#define GRID_RBD_IT_DISPATCH_TORCH_MJX(it_code, FN, MUJOCO_FLAG, ...)                           \
    switch (it_code) {                                                                          \
        case 0: FN<grid::IntegratorType::EULER, MUJOCO_FLAG>(__VA_ARGS__); break;               \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER, MUJOCO_FLAG>(__VA_ARGS__); break; \
        case 2: FN<grid::IntegratorType::MIDPOINT, MUJOCO_FLAG>(__VA_ARGS__); break;            \
        case 3: FN<grid::IntegratorType::RK3, MUJOCO_FLAG>(__VA_ARGS__); break;                 \
        case 4: FN<grid::IntegratorType::RK4, MUJOCO_FLAG>(__VA_ARGS__); break;                 \
        default: TORCH_CHECK(false, "integrator: bad integrator-type code");                    \
    }

// Single-stage-only mjx dispatch (gradient): Euler / Semi-Implicit-Euler only.
#define GRID_RBD_IT_DISPATCH_TORCH_MJX_SS(it_code, FN, MUJOCO_FLAG, ...)                        \
    switch (it_code) {                                                                          \
        case 0: FN<grid::IntegratorType::EULER, MUJOCO_FLAG>(__VA_ARGS__); break;               \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER, MUJOCO_FLAG>(__VA_ARGS__); break; \
        default: TORCH_CHECK(false,                                                             \
            "mujoco integrator/plant-step gradient supports only euler / semi-implicit-euler"); \
    }

#if GRID_HAS_INTEGRATOR
template <grid::IntegratorType IT, bool MUJOCO>
static void torch_launch_integrator(cudaStream_t stream, int batch, double dt, double gravity) {
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::integrator_kernel<T, IT, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_INTEGRATOR>(), grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER>(), stream>>>(
        g_data->d_x_kp1, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, g_data->d_f_ext, (T)gravity, (T)dt, batch);
}
#endif  // GRID_HAS_INTEGRATOR
#if GRID_HAS_INTEGRATOR_GRADIENT
template <grid::IntegratorType IT, bool MUJOCO>
static void torch_launch_integrator_grad(cudaStream_t stream, int batch, double dt, double gravity) {
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::integrator_gradient_kernel<T, IT, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_INTEGRATOR_GRADIENT>(), grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR_GRADIENT>::TIER>(), stream>>>(
        g_data->d_dAB, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, g_data->d_f_ext, (T)gravity, (T)dt, batch);
}
#endif  // GRID_HAS_INTEGRATOR_GRADIENT

#if GRID_HAS_INTEGRATOR
template <bool MUJOCO>
torch::Tensor torch_integrator(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double dt, int64_t it, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "integrator: q", nj); grid_torch_check(qd, "integrator: qd", nj); grid_torch_check(u, "integrator: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, &u);
    auto out = grid_torch_empty(batch, grid::NUM_POS + grid::NUM_VEL, q);
    GRID_RBD_IT_DISPATCH_TORCH_MJX((int)it, torch_launch_integrator, MUJOCO, stream, batch, dt, gravity);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "integrator_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_x_kp1, batch * (grid::NUM_POS + grid::NUM_VEL) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_INTEGRATOR

#if GRID_HAS_INTEGRATOR_GRADIENT
template <bool MUJOCO>
torch::Tensor torch_integrator_gradient(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double dt, int64_t it, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "integrator_gradient: q", nj); grid_torch_check(qd, "integrator_gradient: qd", nj); grid_torch_check(u, "integrator_gradient: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, &u);
    auto out = grid_torch_empty(batch, 2 * nv * 3 * nv, q);
    // mjx gradient is single-stage (euler/si) only; pin supports all integrator types.
    if constexpr (MUJOCO) {
        GRID_RBD_IT_DISPATCH_TORCH_MJX_SS((int)it, torch_launch_integrator_grad, true, stream, batch, dt, gravity);
    } else {
        GRID_RBD_IT_DISPATCH_TORCH_MJX((int)it, torch_launch_integrator_grad, false, stream, batch, dt, gravity);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "integrator_grad_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_dAB, batch * (2 * nv) * (3 * nv) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_INTEGRATOR_GRADIENT

// ── grid_plant surface (cost / barrier / plant-step) ──
//
// Mirror the JAX FFI plant handlers: stage the input tensors D→D into the shared
// g_plant scratch, launch the SAME grid_plant::*_kernel, copy the g_plant outputs
// D→D into freshly-allocated output tensors on the same stream. The cost/barrier
// ops return a (value, grad, hess[/hess_diag]) tuple of tensors; plant_step /
// plant_step_gradient return a single tensor. Reshapes to the _handle.py
// conventions are done Python-side. Gated on the same GRID_PLANT_HAS_* defines.

static inline void grid_torch_plant_init() {
    grid_torch_init_or_throw();
    TORCH_CHECK(plant_alloc() == 0, "plant_alloc failed");
}

// q/qd-style check for a (B, N) plant input with an arbitrary last dim.
static inline void grid_torch_check_n(const torch::Tensor& t, const char* name, int n) {
    TORCH_CHECK(t.is_cuda(), name, ": must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, ": must be contiguous");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, ": must be float32");
    TORCH_CHECK(t.dim() == 2, name, ": must be 2D (B, ", n, ")");
    TORCH_CHECK(t.size(1) == n, name, ": last dim != ", n);
}

// quadratic cost (state or input). var/des/w are (B, N). Returns (value, grad, hess).
// STATE=true selects the state kernel (N = NX) else the input kernel (N = NV).
// Non-template on `state` (runtime bool) so the torch::Tensor .data_ptr<float>()
// member-template calls parse unambiguously under nvcc's host pass; MUJOCO is a
// compile-time template flag threaded into the STATE kernel launch (the only mjx
// path — input cost is frame-invariant and always pinocchio).
template <bool MUJOCO>
static std::vector<torch::Tensor> torch_plant_quadratic_cost(
    torch::Tensor var, torch::Tensor des, torch::Tensor w, bool state) {
    grid_torch_plant_init();
    const int N = state ? (grid::NUM_POS + grid::NUM_VEL) : grid::NUM_VEL;
    grid_torch_check_n(var, "quadratic_cost: var", N);
    grid_torch_check_n(des, "quadratic_cost: des", N);
    grid_torch_check_n(w,   "quadratic_cost: w",   N);
    int batch = grid_torch_batch(var);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, var.data_ptr<float>(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, des.data_ptr<float>(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, w.data_ptr<float>(),   (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out  = grid_torch_empty(batch, 1, var);
    auto grad = grid_torch_empty(batch, N, var);
    auto hess = grid_torch_empty(batch, N * N, var);
    dim3 grid_dim = grid_rbd_grid_for(batch);
    if (state)
        grid_plant::quadratic_state_cost_kernel<T, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
    else
        grid_plant::quadratic_input_cost_kernel<T><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "quadratic_cost_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(),  g_plant.d_out,  (size_t)batch * sizeof(T),         cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<float>(), g_plant.d_grad, (size_t)batch * N * sizeof(T),     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess.data_ptr<float>(), g_plant.d_hess, (size_t)batch * N * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hess};
}

template <bool MUJOCO>
std::vector<torch::Tensor> torch_quadratic_state_cost(torch::Tensor x, torch::Tensor x_des, torch::Tensor Q) {
    return torch_plant_quadratic_cost<MUJOCO>(x, x_des, Q, true);
}
std::vector<torch::Tensor> torch_quadratic_input_cost(torch::Tensor u, torch::Tensor u_des, torch::Tensor R) {
    return torch_plant_quadratic_cost<false>(u, u_des, R, false);
}

// barrier (position/velocity/torque). var/lower/upper are (B, N). Returns
// (value, grad, hess_diag). which: 0=position (N=NUM_POS), 1=velocity, 2=torque.
static std::vector<torch::Tensor> torch_plant_barrier(
    torch::Tensor var, torch::Tensor lower, torch::Tensor upper, double mu, int which) {
    grid_torch_plant_init();
    const int N = (which == 0) ? grid::NUM_POS : grid::NUM_VEL;
    grid_torch_check_n(var,   "barrier: var",   N);
    grid_torch_check_n(lower, "barrier: lower", N);
    grid_torch_check_n(upper, "barrier: upper", N);
    int batch = grid_torch_batch(var);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, var.data_ptr<float>(),   (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, lower.data_ptr<float>(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, upper.data_ptr<float>(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out  = grid_torch_empty(batch, 1, var);
    auto grad = grid_torch_empty(batch, N, var);
    auto hdiag = grid_torch_empty(batch, N, var);
    dim3 grid_dim = grid_rbd_grid_for(batch);
    if (which == 0)
        grid_plant::joint_position_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    else if (which == 1)
        grid_plant::joint_velocity_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    else
        grid_plant::joint_torque_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "joint_barrier_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(),   g_plant.d_out,  (size_t)batch * sizeof(T),     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<float>(),  g_plant.d_grad, (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hdiag.data_ptr<float>(), g_plant.d_hess, (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hdiag};
}

std::vector<torch::Tensor> torch_joint_position_barrier(torch::Tensor v, torch::Tensor lo, torch::Tensor hi, double mu) {
    return torch_plant_barrier(v, lo, hi, mu, 0);
}
std::vector<torch::Tensor> torch_joint_velocity_barrier(torch::Tensor v, torch::Tensor lo, torch::Tensor hi, double mu) {
    return torch_plant_barrier(v, lo, hi, mu, 1);
}
std::vector<torch::Tensor> torch_joint_torque_barrier(torch::Tensor v, torch::Tensor lo, torch::Tensor hi, double mu) {
    return torch_plant_barrier(v, lo, hi, mu, 2);
}

#ifdef GRID_PLANT_HAS_STEP
template <grid::IntegratorType IT, bool MUJOCO>
static void torch_launch_plant_step(cudaStream_t stream, int batch, double gravity, double dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(batch);
    const size_t smem = grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_kernel<T, IT, MUJOCO>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_kernel<T, IT, MUJOCO>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::plant_step_kernel<T, IT, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr,
        smem, stream>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, grid::NUM_VEL, g_robot, (T)gravity, (T)dt, batch);
}

template <bool MUJOCO>
torch::Tensor torch_plant_step(torch::Tensor x, torch::Tensor u, double dt, int64_t it, double gravity) {
    grid_torch_plant_init();
    const int nx = grid::NUM_POS + grid::NUM_VEL, nv = grid::NUM_VEL;
    grid_torch_check_n(x, "plant_step: x", nx);
    grid_torch_check_n(u, "plant_step: u", nv);
    int batch = grid_torch_batch(x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, x.data_ptr<float>(), (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, u.data_ptr<float>(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out = grid_torch_empty(batch, nx, x);
    GRID_RBD_IT_DISPATCH_TORCH_MJX((int)it, torch_launch_plant_step, MUJOCO, stream, batch, gravity, dt);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "plant_step_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_plant.d_grad, (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_PLANT_HAS_STEP

#ifdef GRID_PLANT_HAS_STEP_GRADIENT
template <grid::IntegratorType IT, bool MUJOCO>
static void torch_launch_plant_step_gradient(cudaStream_t stream, int batch, double gravity, double dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL, nv = grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(batch);
    const size_t smem = grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_gradient_kernel<T, IT, MUJOCO>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_gradient_kernel<T, IT, MUJOCO>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::plant_step_gradient_kernel<T, IT, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr,
        smem, stream>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, nv, g_robot, (T)gravity, (T)dt, batch);
}

template <bool MUJOCO>
torch::Tensor torch_plant_step_gradient(torch::Tensor x, torch::Tensor u, double dt, int64_t it, double gravity) {
    grid_torch_plant_init();
    const int nx = grid::NUM_POS + grid::NUM_VEL, nv = grid::NUM_VEL;
    const int dab = 2 * nv * 3 * nv;
    grid_torch_check_n(x, "plant_step_gradient: x", nx);
    grid_torch_check_n(u, "plant_step_gradient: u", nv);
    int batch = grid_torch_batch(x);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, x.data_ptr<float>(), (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, u.data_ptr<float>(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out = grid_torch_empty(batch, dab, x);
    // mjx gradient is single-stage (euler/si) only; pin supports all integrator types.
    if constexpr (MUJOCO) {
        GRID_RBD_IT_DISPATCH_TORCH_MJX_SS((int)it, torch_launch_plant_step_gradient, true, stream, batch, gravity, dt);
    } else {
        GRID_RBD_IT_DISPATCH_TORCH_MJX((int)it, torch_launch_plant_step_gradient, false, stream, batch, gravity, dt);
    }
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "plant_step_gradient_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_plant.d_grad, (size_t)batch * dab * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_PLANT_HAS_STEP_GRADIENT

#ifdef GRID_PLANT_HAS_EE_COST
template <bool MUJOCO>
std::vector<torch::Tensor> torch_ee_pos_cost(torch::Tensor q, torch::Tensor p_des, torch::Tensor W) {
    grid_torch_plant_init();
    const int nq = grid::NUM_POS, nx = grid::NUM_POS + grid::NUM_VEL;
    grid_torch_check_n(q, "ee_pos_cost: q", nq);
    grid_torch_check_n(p_des, "ee_pos_cost: p_des", 3);
    grid_torch_check_n(W, "ee_pos_cost: W", 3);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, q.data_ptr<float>(),     (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, p_des.data_ptr<float>(), (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, W.data_ptr<float>(),     (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out  = grid_torch_empty(batch, 1, q);
    auto grad = grid_torch_empty(batch, nx, q);
    auto hess = grid_torch_empty(batch, nx * nx, q);
    size_t smem = grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::ee_pos_cost_kernel<T, 0, MUJOCO>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::ee_pos_cost_kernel<T, 0, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose, g_plant.d_end_effector_pose_gradient, g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "ee_pos_cost launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<float>(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess.data_ptr<float>(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hess};
}
#endif  // GRID_PLANT_HAS_EE_COST

#ifdef GRID_PLANT_HAS_COM_COST
template <bool MUJOCO>
std::vector<torch::Tensor> torch_com_cost(torch::Tensor q, torch::Tensor p_des, torch::Tensor W) {
    grid_torch_plant_init();
    const int nq = grid::NUM_POS, nx = grid::NUM_POS + grid::NUM_VEL;
    grid_torch_check_n(q, "com_cost: q", nq);
    grid_torch_check_n(p_des, "com_cost: p_des", 3);
    grid_torch_check_n(W, "com_cost: W", 3);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, q.data_ptr<float>(),     (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, p_des.data_ptr<float>(), (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, W.data_ptr<float>(),     (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out  = grid_torch_empty(batch, 1, q);
    auto grad = grid_torch_empty(batch, nx, q);
    auto hess = grid_torch_empty(batch, nx * nx, q);
    size_t smem = grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::com_cost_kernel<T, MUJOCO>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::com_cost_kernel<T, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "com_cost launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<float>(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess.data_ptr<float>(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hess};
}
#endif  // GRID_PLANT_HAS_COM_COST

#ifdef GRID_PLANT_HAS_MOMENTUM_COST
template <bool MUJOCO>
std::vector<torch::Tensor> torch_momentum_cost(torch::Tensor q, torch::Tensor qd, torch::Tensor h_des, torch::Tensor W) {
    grid_torch_plant_init();
    const int nq = grid::NUM_POS, nv = grid::NUM_VEL, nx = nq + nv;
    grid_torch_check_n(q, "momentum_cost: q", nq);
    grid_torch_check_n(qd, "momentum_cost: qd", nv);
    grid_torch_check_n(h_des, "momentum_cost: h_des", 6);
    grid_torch_check_n(W, "momentum_cost: W", 6);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, q.data_ptr<float>(),  (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, qd.data_ptr<float>(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c,                  h_des.data_ptr<float>(), (size_t)batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c + (size_t)batch * 6, W.data_ptr<float>(),  (size_t)batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out  = grid_torch_empty(batch, 1, q);
    auto grad = grid_torch_empty(batch, nx, q);
    auto hess = grid_torch_empty(batch, nx * nx, q);
    size_t smem = grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(batch);
    // momentum_cost is register-heavy (~140 regs/thread): clamp to its launch cap.
    dim3 thr = grid_clamp_threads_for(grid_plant::momentum_cost_kernel<T, MUJOCO>, grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>());
    grid_plant::momentum_cost_kernel<T, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b,
        g_plant.d_in_c, g_plant.d_in_c + (size_t)batch * 6, g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "momentum_cost_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<float>(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess.data_ptr<float>(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hess};
}
#endif  // GRID_PLANT_HAS_MOMENTUM_COST

// ─── P-tier1: centroidal / energy / kinematics family (torch ops) ────────────
//
// Mirror torch_crba EXACTLY: grid_torch_pack stages q (+qd) into d_q_qd_u, launch
// the per-robot kernel directly on the current torch CUDA stream, D→D copy the
// flat result into the empty output. Stride is always 3*NUM_JOINTS (the
// compressed-d_q kernels only read the first NUM_JOINTS of each strided block).
// d_workspace passed ONLY where the contract row says YES; gravity only where YES.
// R2 smem opt-in: covered by grid_rbd_init → init_grid_kernel_attrs (see jax note).

#if GRID_HAS_GENERALIZED_GRAVITY
template <bool MUJOCO>
torch::Tensor torch_generalized_gravity(torch::Tensor q, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "generalized_gravity: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::generalized_gravity_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_c, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "generalized_gravity_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_c, batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_GENERALIZED_GRAVITY

#if GRID_HAS_NONLINEAR_EFFECTS
template <bool MUJOCO>
torch::Tensor torch_nonlinear_effects(torch::Tensor q, torch::Tensor qd, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "nonlinear_effects: q", nj); grid_torch_check(qd, "nonlinear_effects: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::nonlinear_effects_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_c, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "nonlinear_effects_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_c, batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_NONLINEAR_EFFECTS

#if GRID_HAS_CORIOLIS_MATRIX
template <bool MUJOCO>
torch::Tensor torch_coriolis_matrix(torch::Tensor q, torch::Tensor qd, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "coriolis_matrix: q", nj); grid_torch_check(qd, "coriolis_matrix: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::coriolis_matrix_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::CORIOLIS_MATRIX_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_coriolis, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "coriolis_matrix_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_coriolis, batch * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_CORIOLIS_MATRIX

#if GRID_HAS_KINETIC_ENERGY_REGRESSOR
template <bool MUJOCO>
torch::Tensor torch_kinetic_energy_regressor(torch::Tensor q, torch::Tensor qd, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nb = grid::NUM_BODIES;
    grid_torch_check(q, "kinetic_energy_regressor: q", nj); grid_torch_check(qd, "kinetic_energy_regressor: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, 10 * nb, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::kinetic_energy_regressor_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::KINETIC_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_ke_regressor, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kinetic_energy_regressor_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_ke_regressor, batch * 10 * nb * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_KINETIC_ENERGY_REGRESSOR

#if GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
template <bool MUJOCO>
torch::Tensor torch_potential_energy_regressor(torch::Tensor q, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nb = grid::NUM_BODIES;
    grid_torch_check(q, "potential_energy_regressor: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 10 * nb, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::potential_energy_regressor_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::POTENTIAL_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_pe_regressor, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "potential_energy_regressor_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_pe_regressor, batch * 10 * nb * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_POTENTIAL_ENERGY_REGRESSOR

#ifdef GRID_HAS_ENERGY
template <bool MUJOCO>
torch::Tensor torch_energy(torch::Tensor q, torch::Tensor qd, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "energy: q", nj); grid_torch_check(qd, "energy: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, 3, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::energy_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::ENERGY_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_energy, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "energy_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_energy, batch * 3 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_ENERGY

#ifdef GRID_HAS_COM
// com → single flat (B, 3 + 3*NV); the Python layer splits the (p_com, J_com) tuple.
template <bool MUJOCO>
torch::Tensor torch_com(torch::Tensor q) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "com: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 3 + 3 * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::com_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_com, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "com_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_com, batch * (3 + 3 * nv) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_COM

#ifdef GRID_HAS_CCRBA
// ccrba → single flat (B, 6*NV + 6); the Python layer splits the (A, h) tuple.
template <bool MUJOCO>
torch::Tensor torch_ccrba(torch::Tensor q, torch::Tensor qd) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "ccrba: q", nj); grid_torch_check(qd, "ccrba: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, 6 * nv + 6, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::ccrba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_ccrba, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "ccrba_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_ccrba, batch * (6 * nv + 6) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_CCRBA

#ifdef GRID_HAS_CMM_TIME_VARIATION
template <bool MUJOCO>
torch::Tensor torch_cmm_time_variation(torch::Tensor q, torch::Tensor qd) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "cmm_time_variation: q", nj); grid_torch_check(qd, "cmm_time_variation: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, 6 * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::cmm_time_variation_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::CMM_TIME_VARIATION_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER>(), stream>>>(
        g_data->d_cmm_time_variation, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "cmm_time_variation_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_cmm_time_variation, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_CMM_TIME_VARIATION

#ifdef GRID_HAS_DCCRBA
template <bool MUJOCO>
torch::Tensor torch_dccrba(torch::Tensor q) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "dccrba: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 6 * nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::dccrba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::DCCRBA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER>(), stream>>>(
        g_data->d_dccrba, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "dccrba_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_dccrba, batch * 6 * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_DCCRBA

#ifdef GRID_HAS_FRAME_JACOBIAN
// frame_jacobian / frame_jacobian_dot: target_jid + reference_frame trail the
// schema as ints, passed straight through to the kernel (NO host -1 default
// resolution — the Python torch surface passes resolved non-negative values).
template <bool MUJOCO>
torch::Tensor torch_frame_jacobian(torch::Tensor q, int64_t target_jid, int64_t reference_frame) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "frame_jacobian: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 6 * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::frame_jacobian_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::FRAME_JACOBIAN_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_frame_jacobian, g_data->d_q_qd_u, stride, (int)target_jid, (int)reference_frame, g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "frame_jacobian_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_frame_jacobian, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}

template <bool MUJOCO>
torch::Tensor torch_frame_jacobian_dot(torch::Tensor q, torch::Tensor qd, int64_t target_jid, int64_t reference_frame) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "frame_jacobian_dot: q", nj); grid_torch_check(qd, "frame_jacobian_dot: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, 6 * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::frame_jacobian_dot_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::FRAME_JACOBIAN_DOT_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_frame_jacobian_dot, g_data->d_q_qd_u, stride, (int)target_jid, (int)reference_frame, g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "frame_jacobian_dot_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_frame_jacobian_dot, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}

template <bool MUJOCO>
torch::Tensor torch_osc_inertia(torch::Tensor q) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "osc_inertia: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 36, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::osc_inertia_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(batch), grid_rbd_launch_threads<grid::GRID_ALGO_COUNT>(), grid::OSC_INERTIA_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_osc_inertia, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "osc_inertia_kernel launch failed");
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_osc_inertia, batch * 36 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_FRAME_JACOBIAN

}  // namespace

// The op library name is keyed by the cache_key so two robots don't collide.
#ifndef GRID_RBD_TORCH_KEY
#define GRID_RBD_TORCH_KEY default
#endif
#define GRID_RBD_TORCH_CONCAT2(a, b) a##b
#define GRID_RBD_TORCH_CONCAT(a, b) GRID_RBD_TORCH_CONCAT2(a, b)
#define GRID_RBD_TORCH_LIB GRID_RBD_TORCH_CONCAT(grid_rbd_torch_, GRID_RBD_TORCH_KEY)

// Indirection so GRID_RBD_TORCH_LIB is fully expanded BEFORE TORCH_LIBRARY
// stringizes/token-pastes it. Without this, TORCH_LIBRARY(GRID_RBD_TORCH_LIB,..)
// registers under the literal token "GRID_RBD_TORCH_LIB" (token-paste suppresses
// expansion), while TORCH_LIBRARY_IMPL's extra macro layer expands it — a
// namespace mismatch that hides every op. The wrapper forces expansion for both.
#define GRID_RBD_TORCH_LIBRARY(ns, m) TORCH_LIBRARY(ns, m)
#define GRID_RBD_TORCH_LIBRARY_IMPL(ns, k, m) TORCH_LIBRARY_IMPL(ns, k, m)

GRID_RBD_TORCH_LIBRARY(GRID_RBD_TORCH_LIB, m) {
#if GRID_HAS_INVERSE_DYNAMICS
    m.def("inverse_dynamics(Tensor q, Tensor qd, float gravity, Tensor? qdd=None, Tensor? f_ext=None) -> Tensor");
#endif  // GRID_HAS_INVERSE_DYNAMICS
#if GRID_HAS_MINV
    m.def("minv(Tensor q) -> Tensor");
#endif  // GRID_HAS_MINV
#if GRID_HAS_FORWARD_DYNAMICS
    m.def("forward_dynamics(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None) -> Tensor");
#endif  // GRID_HAS_FORWARD_DYNAMICS
#if GRID_HAS_ABA
    m.def("aba(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None) -> Tensor");
#endif  // GRID_HAS_ABA
#if GRID_HAS_CRBA
    m.def("crba(Tensor q, float gravity) -> Tensor");
#endif  // GRID_HAS_CRBA
#if GRID_HAS_END_EFFECTOR_POSE
    m.def("end_effector_pose(Tensor q) -> Tensor");
#endif  // GRID_HAS_END_EFFECTOR_POSE
#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
    m.def("end_effector_pose_gradient(Tensor q) -> Tensor");
#endif  // GRID_HAS_END_EFFECTOR_POSE_GRADIENT
#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN
    m.def("end_effector_pose_hessian(Tensor q) -> Tensor");
#endif  // GRID_HAS_END_EFFECTOR_POSE_HESSIAN
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
    m.def("inverse_dynamics_gradient(Tensor q, Tensor qd, float gravity, Tensor? qdd=None, Tensor? f_ext=None) -> Tensor");
#endif  // GRID_HAS_INVERSE_DYNAMICS_GRADIENT
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
    m.def("forward_dynamics_gradient(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None) -> Tensor");
#endif  // GRID_HAS_FORWARD_DYNAMICS_GRADIENT
#if GRID_HAS_IDSVA_SO
    m.def("idsva_so(Tensor q, Tensor qd, Tensor qdd, float gravity) -> Tensor");
#endif  // GRID_HAS_IDSVA_SO
#if GRID_HAS_FDSVA_SO
    m.def("fdsva_so(Tensor q, Tensor qd, Tensor u, float gravity) -> Tensor");
#endif  // GRID_HAS_FDSVA_SO
#if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
    m.def("inverse_dynamics_regressor(Tensor q, Tensor qd, Tensor qdd, float gravity) -> Tensor");
#endif  // GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
#if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
    m.def("forward_dynamics_parameter_gradient(Tensor q, Tensor qd, Tensor u, float gravity) -> Tensor");
#endif  // GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
#if GRID_HAS_INTEGRATOR
    m.def("integrator(Tensor q, Tensor qd, Tensor u, float dt, int it, float gravity) -> Tensor");
#endif  // GRID_HAS_INTEGRATOR
#if GRID_HAS_INTEGRATOR_GRADIENT
    m.def("integrator_gradient(Tensor q, Tensor qd, Tensor u, float dt, int it, float gravity) -> Tensor");
#endif  // GRID_HAS_INTEGRATOR_GRADIENT
    // P-tier1 centroidal/energy/kinematics family (single flat Tensor each; the
    // Python layer reshapes/splits per _handle.py). Gated schemas are always
    // def'd (cheap) but only impl'd under the matching GRID_HAS_* below.
#if GRID_HAS_GENERALIZED_GRAVITY
    m.def("generalized_gravity(Tensor q, float gravity) -> Tensor");
#endif  // GRID_HAS_GENERALIZED_GRAVITY
#if GRID_HAS_NONLINEAR_EFFECTS
    m.def("nonlinear_effects(Tensor q, Tensor qd, float gravity) -> Tensor");
#endif  // GRID_HAS_NONLINEAR_EFFECTS
#if GRID_HAS_CORIOLIS_MATRIX
    m.def("coriolis_matrix(Tensor q, Tensor qd, float gravity) -> Tensor");
#endif  // GRID_HAS_CORIOLIS_MATRIX
#if GRID_HAS_KINETIC_ENERGY_REGRESSOR
    m.def("kinetic_energy_regressor(Tensor q, Tensor qd, float gravity) -> Tensor");
#endif  // GRID_HAS_KINETIC_ENERGY_REGRESSOR
#if GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
    m.def("potential_energy_regressor(Tensor q, float gravity) -> Tensor");
#endif  // GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
    m.def("energy(Tensor q, Tensor qd, float gravity) -> Tensor");
    m.def("com(Tensor q) -> Tensor");
    m.def("ccrba(Tensor q, Tensor qd) -> Tensor");
    m.def("cmm_time_variation(Tensor q, Tensor qd) -> Tensor");
    m.def("dccrba(Tensor q) -> Tensor");
    m.def("frame_jacobian(Tensor q, int target_jid, int reference_frame) -> Tensor");
    m.def("frame_jacobian_dot(Tensor q, Tensor qd, int target_jid, int reference_frame) -> Tensor");
    m.def("osc_inertia(Tensor q) -> Tensor");
    // runtime-target multi-EE pose/gradient (single target jid + 3-vec offset; the
    // Python layer loops the resolved jid list + stacks). Gated schemas always def'd.
    m.def("end_effector_pose_runtime(Tensor q, int target_jid, Tensor offset) -> Tensor");
    m.def("end_effector_pose_gradient_runtime(Tensor q, int target_jid, Tensor offset) -> Tensor");
    // grid_plant surface (cost / barrier always emitted; the rest are gated).
    m.def("quadratic_state_cost(Tensor x, Tensor x_des, Tensor Q) -> Tensor[]");
    m.def("quadratic_input_cost(Tensor u, Tensor u_des, Tensor R) -> Tensor[]");
    m.def("joint_position_barrier(Tensor var, Tensor lower, Tensor upper, float mu) -> Tensor[]");
    m.def("joint_velocity_barrier(Tensor var, Tensor lower, Tensor upper, float mu) -> Tensor[]");
    m.def("joint_torque_barrier(Tensor var, Tensor lower, Tensor upper, float mu) -> Tensor[]");
#ifdef GRID_PLANT_HAS_STEP
    m.def("plant_step(Tensor x, Tensor u, float dt, int it, float gravity) -> Tensor");
#endif
#ifdef GRID_PLANT_HAS_STEP_GRADIENT
    m.def("plant_step_gradient(Tensor x, Tensor u, float dt, int it, float gravity) -> Tensor");
#endif
#ifdef GRID_PLANT_HAS_EE_COST
    m.def("ee_pos_cost(Tensor q, Tensor p_des, Tensor W) -> Tensor[]");
#endif
#ifdef GRID_PLANT_HAS_COM_COST
    m.def("com_cost(Tensor q, Tensor p_des, Tensor W) -> Tensor[]");
#endif
#ifdef GRID_PLANT_HAS_MOMENTUM_COST
    m.def("momentum_cost(Tensor q, Tensor qd, Tensor h_des, Tensor W) -> Tensor[]");
#endif
#ifdef GRID_RBD_WITH_MUJOCO
    // MuJoCo-convention ops (floating only): same schema, name suffixed _mujoco; the
    // CUDA impl launches the kernel with MUJOCO_OUTPUT=true.
#if GRID_HAS_INVERSE_DYNAMICS
    m.def("inverse_dynamics_mujoco(Tensor q, Tensor qd, float gravity, Tensor? qdd=None, Tensor? f_ext=None) -> Tensor");
#endif  // GRID_HAS_INVERSE_DYNAMICS
#if GRID_HAS_MINV
    m.def("minv_mujoco(Tensor q) -> Tensor");
#endif  // GRID_HAS_MINV
#if GRID_HAS_FORWARD_DYNAMICS
    m.def("forward_dynamics_mujoco(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None) -> Tensor");
#endif  // GRID_HAS_FORWARD_DYNAMICS
#if GRID_HAS_ABA
    m.def("aba_mujoco(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None) -> Tensor");
#endif  // GRID_HAS_ABA
#if GRID_HAS_CRBA
    m.def("crba_mujoco(Tensor q, float gravity) -> Tensor");
#endif  // GRID_HAS_CRBA
#if GRID_HAS_END_EFFECTOR_POSE
    m.def("end_effector_pose_mujoco(Tensor q) -> Tensor");
#endif  // GRID_HAS_END_EFFECTOR_POSE
#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
    m.def("end_effector_pose_gradient_mujoco(Tensor q) -> Tensor");
#endif  // GRID_HAS_END_EFFECTOR_POSE_GRADIENT
#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN
    m.def("end_effector_pose_hessian_mujoco(Tensor q) -> Tensor");
#endif  // GRID_HAS_END_EFFECTOR_POSE_HESSIAN
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
    m.def("inverse_dynamics_gradient_mujoco(Tensor q, Tensor qd, float gravity, Tensor? qdd=None, Tensor? f_ext=None) -> Tensor");
#endif  // GRID_HAS_INVERSE_DYNAMICS_GRADIENT
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
    m.def("forward_dynamics_gradient_mujoco(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None) -> Tensor");
#endif  // GRID_HAS_FORWARD_DYNAMICS_GRADIENT
#if GRID_HAS_IDSVA_SO
    m.def("idsva_so_mujoco(Tensor q, Tensor qd, Tensor qdd, float gravity) -> Tensor");
#endif  // GRID_HAS_IDSVA_SO
#if GRID_HAS_FDSVA_SO
    m.def("fdsva_so_mujoco(Tensor q, Tensor qd, Tensor u, float gravity) -> Tensor");
#endif  // GRID_HAS_FDSVA_SO
#if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
    m.def("inverse_dynamics_regressor_mujoco(Tensor q, Tensor qd, Tensor qdd, float gravity) -> Tensor");
#endif  // GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
#if GRID_HAS_INTEGRATOR
    m.def("integrator_mujoco(Tensor q, Tensor qd, Tensor u, float dt, int it, float gravity) -> Tensor");
#endif  // GRID_HAS_INTEGRATOR
#if GRID_HAS_INTEGRATOR_GRADIENT
    m.def("integrator_gradient_mujoco(Tensor q, Tensor qd, Tensor u, float dt, int it, float gravity) -> Tensor");
#endif  // GRID_HAS_INTEGRATOR_GRADIENT
    // P-tier1 mujoco-convention variants (floating only; impl'd under the same GRID_HAS_* below).
#if GRID_HAS_GENERALIZED_GRAVITY
    m.def("generalized_gravity_mujoco(Tensor q, float gravity) -> Tensor");
#endif  // GRID_HAS_GENERALIZED_GRAVITY
#if GRID_HAS_NONLINEAR_EFFECTS
    m.def("nonlinear_effects_mujoco(Tensor q, Tensor qd, float gravity) -> Tensor");
#endif  // GRID_HAS_NONLINEAR_EFFECTS
#if GRID_HAS_CORIOLIS_MATRIX
    m.def("coriolis_matrix_mujoco(Tensor q, Tensor qd, float gravity) -> Tensor");
#endif  // GRID_HAS_CORIOLIS_MATRIX
#if GRID_HAS_KINETIC_ENERGY_REGRESSOR
    m.def("kinetic_energy_regressor_mujoco(Tensor q, Tensor qd, float gravity) -> Tensor");
#endif  // GRID_HAS_KINETIC_ENERGY_REGRESSOR
#if GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
    m.def("potential_energy_regressor_mujoco(Tensor q, float gravity) -> Tensor");
#endif  // GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
    m.def("energy_mujoco(Tensor q, Tensor qd, float gravity) -> Tensor");
    m.def("com_mujoco(Tensor q) -> Tensor");
    m.def("ccrba_mujoco(Tensor q, Tensor qd) -> Tensor");
    m.def("cmm_time_variation_mujoco(Tensor q, Tensor qd) -> Tensor");
    m.def("dccrba_mujoco(Tensor q) -> Tensor");
    m.def("frame_jacobian_mujoco(Tensor q, int target_jid, int reference_frame) -> Tensor");
    m.def("frame_jacobian_dot_mujoco(Tensor q, Tensor qd, int target_jid, int reference_frame) -> Tensor");
    m.def("osc_inertia_mujoco(Tensor q) -> Tensor");
    m.def("end_effector_pose_runtime_mujoco(Tensor q, int target_jid, Tensor offset) -> Tensor");
    m.def("end_effector_pose_gradient_runtime_mujoco(Tensor q, int target_jid, Tensor offset) -> Tensor");
    m.def("quadratic_state_cost_mujoco(Tensor x, Tensor x_des, Tensor Q) -> Tensor[]");
#ifdef GRID_PLANT_HAS_STEP
    m.def("plant_step_mujoco(Tensor x, Tensor u, float dt, int it, float gravity) -> Tensor");
#endif
#ifdef GRID_PLANT_HAS_STEP_GRADIENT
    m.def("plant_step_gradient_mujoco(Tensor x, Tensor u, float dt, int it, float gravity) -> Tensor");
#endif
#ifdef GRID_PLANT_HAS_EE_COST
    m.def("ee_pos_cost_mujoco(Tensor q, Tensor p_des, Tensor W) -> Tensor[]");
#endif
#ifdef GRID_PLANT_HAS_COM_COST
    m.def("com_cost_mujoco(Tensor q, Tensor p_des, Tensor W) -> Tensor[]");
#endif
#ifdef GRID_PLANT_HAS_MOMENTUM_COST
    m.def("momentum_cost_mujoco(Tensor q, Tensor qd, Tensor h_des, Tensor W) -> Tensor[]");
#endif
#endif  // GRID_RBD_WITH_MUJOCO
}

GRID_RBD_TORCH_LIBRARY_IMPL(GRID_RBD_TORCH_LIB, CUDA, m) {
#if GRID_HAS_INVERSE_DYNAMICS
    m.impl("inverse_dynamics", torch_inverse_dynamics<false>);
#endif  // GRID_HAS_INVERSE_DYNAMICS
#if GRID_HAS_MINV
    m.impl("minv", torch_minv<false>);
#endif  // GRID_HAS_MINV
#if GRID_HAS_FORWARD_DYNAMICS
    m.impl("forward_dynamics", torch_forward_dynamics<false>);
#endif  // GRID_HAS_FORWARD_DYNAMICS
#if GRID_HAS_ABA
    m.impl("aba", torch_aba<false>);
#endif  // GRID_HAS_ABA
#if GRID_HAS_CRBA
    m.impl("crba", torch_crba<false>);
#endif  // GRID_HAS_CRBA
#if GRID_HAS_END_EFFECTOR_POSE
    m.impl("end_effector_pose", torch_end_effector_pose<false>);
#endif  // GRID_HAS_END_EFFECTOR_POSE
#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
    m.impl("end_effector_pose_gradient", torch_end_effector_pose_gradient<false>);
#endif  // GRID_HAS_END_EFFECTOR_POSE_GRADIENT
#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN
    m.impl("end_effector_pose_hessian", torch_end_effector_pose_hessian<false>);
#endif  // GRID_HAS_END_EFFECTOR_POSE_HESSIAN
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
    m.impl("inverse_dynamics_gradient", torch_inverse_dynamics_gradient<false>);
#endif  // GRID_HAS_INVERSE_DYNAMICS_GRADIENT
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
    m.impl("forward_dynamics_gradient", torch_forward_dynamics_gradient<false>);
#endif  // GRID_HAS_FORWARD_DYNAMICS_GRADIENT
#if GRID_HAS_IDSVA_SO
    m.impl("idsva_so", torch_idsva_so<false>);
#endif  // GRID_HAS_IDSVA_SO
#if GRID_HAS_FDSVA_SO
    m.impl("fdsva_so", torch_fdsva_so<false>);
#endif  // GRID_HAS_FDSVA_SO
#if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
    m.impl("inverse_dynamics_regressor", torch_inverse_dynamics_regressor<false>);
#endif  // GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
#if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
    m.impl("forward_dynamics_parameter_gradient", torch_forward_dynamics_parameter_gradient);
#endif  // GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
#if GRID_HAS_INTEGRATOR
    m.impl("integrator", torch_integrator<false>);
#endif  // GRID_HAS_INTEGRATOR
#if GRID_HAS_INTEGRATOR_GRADIENT
    m.impl("integrator_gradient", torch_integrator_gradient<false>);
#endif  // GRID_HAS_INTEGRATOR_GRADIENT
    // P-tier1 ungated value ops.
#if GRID_HAS_GENERALIZED_GRAVITY
    m.impl("generalized_gravity", torch_generalized_gravity<false>);
#endif  // GRID_HAS_GENERALIZED_GRAVITY
#if GRID_HAS_NONLINEAR_EFFECTS
    m.impl("nonlinear_effects", torch_nonlinear_effects<false>);
#endif  // GRID_HAS_NONLINEAR_EFFECTS
#if GRID_HAS_CORIOLIS_MATRIX
    m.impl("coriolis_matrix", torch_coriolis_matrix<false>);
#endif  // GRID_HAS_CORIOLIS_MATRIX
#if GRID_HAS_KINETIC_ENERGY_REGRESSOR
    m.impl("kinetic_energy_regressor", torch_kinetic_energy_regressor<false>);
#endif  // GRID_HAS_KINETIC_ENERGY_REGRESSOR
#if GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
    m.impl("potential_energy_regressor", torch_potential_energy_regressor<false>);
#endif  // GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
    // P-tier1 gated value ops + int-attr kinematics.
#ifdef GRID_HAS_ENERGY
    m.impl("energy", torch_energy<false>);
#endif
#ifdef GRID_HAS_COM
    m.impl("com", torch_com<false>);
#endif
#ifdef GRID_HAS_CCRBA
    m.impl("ccrba", torch_ccrba<false>);
#endif
#ifdef GRID_HAS_CMM_TIME_VARIATION
    m.impl("cmm_time_variation", torch_cmm_time_variation<false>);
#endif
#ifdef GRID_HAS_DCCRBA
    m.impl("dccrba", torch_dccrba<false>);
#endif
#ifdef GRID_HAS_FRAME_JACOBIAN
    m.impl("frame_jacobian", torch_frame_jacobian<false>);
    m.impl("frame_jacobian_dot", torch_frame_jacobian_dot<false>);
    m.impl("osc_inertia", torch_osc_inertia<false>);
#endif
#ifdef GRID_HAS_END_EFFECTOR_POSE_RUNTIME
    m.impl("end_effector_pose_runtime", torch_end_effector_pose_runtime<false>);
#endif
#ifdef GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME
    m.impl("end_effector_pose_gradient_runtime", torch_end_effector_pose_gradient_runtime<false>);
#endif
    m.impl("quadratic_state_cost", torch_quadratic_state_cost<false>);
    m.impl("quadratic_input_cost", torch_quadratic_input_cost);
    m.impl("joint_position_barrier", torch_joint_position_barrier);
    m.impl("joint_velocity_barrier", torch_joint_velocity_barrier);
    m.impl("joint_torque_barrier", torch_joint_torque_barrier);
#ifdef GRID_PLANT_HAS_STEP
    m.impl("plant_step", torch_plant_step<false>);
#endif
#ifdef GRID_PLANT_HAS_STEP_GRADIENT
    m.impl("plant_step_gradient", torch_plant_step_gradient<false>);
#endif
#ifdef GRID_PLANT_HAS_EE_COST
    m.impl("ee_pos_cost", torch_ee_pos_cost<false>);
#endif
#ifdef GRID_PLANT_HAS_COM_COST
    m.impl("com_cost", torch_com_cost<false>);
#endif
#ifdef GRID_PLANT_HAS_MOMENTUM_COST
    m.impl("momentum_cost", torch_momentum_cost<false>);
#endif
#ifdef GRID_RBD_WITH_MUJOCO
    // MuJoCo-convention impls (floating only): same op functions instantiated with
    // MUJOCO=true so the kernel launches with MUJOCO_OUTPUT=true.
#if GRID_HAS_INVERSE_DYNAMICS
    m.impl("inverse_dynamics_mujoco", torch_inverse_dynamics<true>);
#endif  // GRID_HAS_INVERSE_DYNAMICS
#if GRID_HAS_MINV
    m.impl("minv_mujoco", torch_minv<true>);
#endif  // GRID_HAS_MINV
#if GRID_HAS_FORWARD_DYNAMICS
    m.impl("forward_dynamics_mujoco", torch_forward_dynamics<true>);
#endif  // GRID_HAS_FORWARD_DYNAMICS
#if GRID_HAS_ABA
    m.impl("aba_mujoco", torch_aba<true>);
#endif  // GRID_HAS_ABA
#if GRID_HAS_CRBA
    m.impl("crba_mujoco", torch_crba<true>);
#endif  // GRID_HAS_CRBA
#if GRID_HAS_END_EFFECTOR_POSE
    m.impl("end_effector_pose_mujoco", torch_end_effector_pose<true>);
#endif  // GRID_HAS_END_EFFECTOR_POSE
#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
    m.impl("end_effector_pose_gradient_mujoco", torch_end_effector_pose_gradient<true>);
#endif  // GRID_HAS_END_EFFECTOR_POSE_GRADIENT
#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN
    m.impl("end_effector_pose_hessian_mujoco", torch_end_effector_pose_hessian<true>);
#endif  // GRID_HAS_END_EFFECTOR_POSE_HESSIAN
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
    m.impl("inverse_dynamics_gradient_mujoco", torch_inverse_dynamics_gradient<true>);
#endif  // GRID_HAS_INVERSE_DYNAMICS_GRADIENT
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
    m.impl("forward_dynamics_gradient_mujoco", torch_forward_dynamics_gradient<true>);
#endif  // GRID_HAS_FORWARD_DYNAMICS_GRADIENT
#if GRID_HAS_IDSVA_SO
    m.impl("idsva_so_mujoco", torch_idsva_so<true>);
#endif  // GRID_HAS_IDSVA_SO
#if GRID_HAS_FDSVA_SO
    m.impl("fdsva_so_mujoco", torch_fdsva_so<true>);
#endif  // GRID_HAS_FDSVA_SO
#if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
    m.impl("inverse_dynamics_regressor_mujoco", torch_inverse_dynamics_regressor<true>);
#endif  // GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
#if GRID_HAS_INTEGRATOR
    m.impl("integrator_mujoco", torch_integrator<true>);
#endif  // GRID_HAS_INTEGRATOR
#if GRID_HAS_INTEGRATOR_GRADIENT
    m.impl("integrator_gradient_mujoco", torch_integrator_gradient<true>);
#endif  // GRID_HAS_INTEGRATOR_GRADIENT
    // P-tier1 mujoco-convention variants.
#if GRID_HAS_GENERALIZED_GRAVITY
    m.impl("generalized_gravity_mujoco", torch_generalized_gravity<true>);
#endif  // GRID_HAS_GENERALIZED_GRAVITY
#if GRID_HAS_NONLINEAR_EFFECTS
    m.impl("nonlinear_effects_mujoco", torch_nonlinear_effects<true>);
#endif  // GRID_HAS_NONLINEAR_EFFECTS
#if GRID_HAS_CORIOLIS_MATRIX
    m.impl("coriolis_matrix_mujoco", torch_coriolis_matrix<true>);
#endif  // GRID_HAS_CORIOLIS_MATRIX
#if GRID_HAS_KINETIC_ENERGY_REGRESSOR
    m.impl("kinetic_energy_regressor_mujoco", torch_kinetic_energy_regressor<true>);
#endif  // GRID_HAS_KINETIC_ENERGY_REGRESSOR
#if GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
    m.impl("potential_energy_regressor_mujoco", torch_potential_energy_regressor<true>);
#endif  // GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
#ifdef GRID_HAS_ENERGY
    m.impl("energy_mujoco", torch_energy<true>);
#endif
#ifdef GRID_HAS_COM
    m.impl("com_mujoco", torch_com<true>);
#endif
#ifdef GRID_HAS_CCRBA
    m.impl("ccrba_mujoco", torch_ccrba<true>);
#endif
#ifdef GRID_HAS_CMM_TIME_VARIATION
    m.impl("cmm_time_variation_mujoco", torch_cmm_time_variation<true>);
#endif
#ifdef GRID_HAS_DCCRBA
    m.impl("dccrba_mujoco", torch_dccrba<true>);
#endif
#ifdef GRID_HAS_FRAME_JACOBIAN
    m.impl("frame_jacobian_mujoco", torch_frame_jacobian<true>);
    m.impl("frame_jacobian_dot_mujoco", torch_frame_jacobian_dot<true>);
    m.impl("osc_inertia_mujoco", torch_osc_inertia<true>);
#endif
#ifdef GRID_HAS_END_EFFECTOR_POSE_RUNTIME
    m.impl("end_effector_pose_runtime_mujoco", torch_end_effector_pose_runtime<true>);
#endif
#ifdef GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME
    m.impl("end_effector_pose_gradient_runtime_mujoco", torch_end_effector_pose_gradient_runtime<true>);
#endif
    m.impl("quadratic_state_cost_mujoco", torch_quadratic_state_cost<true>);
#ifdef GRID_PLANT_HAS_STEP
    m.impl("plant_step_mujoco", torch_plant_step<true>);
#endif
#ifdef GRID_PLANT_HAS_STEP_GRADIENT
    m.impl("plant_step_gradient_mujoco", torch_plant_step_gradient<true>);
#endif
#ifdef GRID_PLANT_HAS_EE_COST
    m.impl("ee_pos_cost_mujoco", torch_ee_pos_cost<true>);
#endif
#ifdef GRID_PLANT_HAS_COM_COST
    m.impl("com_cost_mujoco", torch_com_cost<true>);
#endif
#ifdef GRID_PLANT_HAS_MOMENTUM_COST
    m.impl("momentum_cost_mujoco", torch_momentum_cost<true>);
#endif
#endif  // GRID_RBD_WITH_MUJOCO
}

#endif  // GRID_RBD_WITH_TORCH

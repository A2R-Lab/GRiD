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

// Sync + consume the NO_EXIT sticky error slot (see the comment in
// grid_rbd_sync_consume). Returns 0 on success, 100+cudaError_t on failure.
static inline int grid_rbd_sync_consume() {
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    return (e != cudaSuccess) ? 100 + (int)e : 0;
}

// ─── runtime contexts (W04-B B1, K1+K5; codex constraints C1-C3, 2026-09-24) ─────
// The .so no longer owns ONE set of device buffers. Every call names a context by
// id; a context owns its arena (and the allocator pool it was carved from — K1),
// its robot tables, streams, plant staging, launch overrides and a device-profile
// record. Ids are 64-bit, salted per loaded .so (high 32 bits) so an id minted by
// another artifact is rejected (C1); 0 is an ALIAS for "this artifact's default
// context" (created lazily, closed by grid_rbd_close, re-created on the next
// call — the legacy lifecycle), never a real id; a closed id is tombstoned and
// never resolves again (C1). Lookup takes a strong execution reference under the
// registry mutex atomically with the open check; close refuses new admissions,
// drains admitted calls, completes the device, then frees (C2).
#include <atomic>
#include <chrono>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

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

// release a context's plant staging (always compiled: the close path needs it even on a subset build)
static void plant_free(PlantBuffers *p) {
    if (!p || !p->allocated) return;
    cudaFree(p->d_in_a); cudaFree(p->d_in_b); cudaFree(p->d_in_c); cudaFree(p->d_out); cudaFree(p->d_grad); cudaFree(p->d_hess);
    if (p->d_d2AB) cudaFree(p->d_d2AB);
    if (p->d_d2AB_workspace) cudaFree(p->d_d2AB_workspace);
    if (p->d_end_effector_pose) cudaFree(p->d_end_effector_pose);
    if (p->d_end_effector_pose_gradient) cudaFree(p->d_end_effector_pose_gradient);
    *p = PlantBuffers();
}
struct LaunchOverrides {
    int threads_override = -1;                          // set_threads_per_block: -1 = per-algo autotuned default
    int threads_per_algo[grid::GRID_ALGO_COUNT];        // E6 per-algo overlay (-1 = baked launch_cfg)
    int threads_per_algo_small[grid::GRID_ALGO_COUNT];  // E6 batch-switch small-batch pick
    int batch_threshold_per_algo[grid::GRID_ALGO_COUNT];// E6 batch-switch threshold (0 = off)
    LaunchOverrides() {
        for (int i = 0; i < grid::GRID_ALGO_COUNT; ++i) { threads_per_algo[i] = -1; threads_per_algo_small[i] = -1; batch_threshold_per_algo[i] = 0; }
    }
};
// Recorded at context creation; exposed through grid_rbd_ctx_profile so a
// deployment can assert what it is running on (portability contract, T5).
struct GridDeviceProfile {
    int device_cc;              // compute capability of the device (major*10+minor)
    int artifact_cc;            // the arch this .so was compiled for (GRID_RBD_ARCH), 0 if unknown
    long long total_bytes;      // device memory at creation
    long long free_bytes;       // free device memory at creation (before the arena)
    long long arena_bytes;      // gridData_device_bytes at the fitted slot count
    int workspace_slots;        // the auto-fit (or declared) workspace slot count
    int max_batch;              // kMaxBatch baked into this .so
    long long smem_optin_bytes; // cudaDevAttrMaxSharedMemoryPerBlockOptin
    int slab_installed;         // 1 = the arena was carved from a caller-owned slab
};
struct GridCtx {
    long long id = 0;
    grid::gridData<T>   *data    = nullptr;
    grid::robotModel<T> *robot   = nullptr;
    cudaStream_t        *streams = nullptr;
    grid::grid_device_pool_t pool = {nullptr, 0, 0, 0};   // THIS context's allocator (base=nullptr: cudaMalloc path)
    PlantBuffers *plant = nullptr;                        // plant staging (allocated lazily by plant_alloc)
    LaunchOverrides launch;
    GridDeviceProfile profile = {};
    std::atomic<int> inflight{0};                         // admitted calls (strong execution references)
    bool closing = false;
    // W04-B B2: admission lock + model version. Every compute call holds
    // `admission` SHARED for its whole submission; every MODEL mutator (the
    // runtime-parameter setters — tool attach/detach is one — and the launch
    // overrides) holds it EXCLUSIVE, so a mutation is ordered after every admitted
    // call and before every later one (no torn table reads, no launch-override
    // races). `version` bumps on every model mutation (never on a launch
    // override); autograd forwards stamp it ON DEVICE at execution time and their
    // backwards compare that stamp with their own admission version (K4) — a
    // traced Python int would be trace-time, not execution-time.
    std::shared_mutex admission;
    unsigned long long version = 1;
};
static std::mutex &grid_ctx_mutex() { static std::mutex m; return m; }
static std::unordered_map<long long, GridCtx *> &grid_ctx_registry() { static std::unordered_map<long long, GridCtx *> r; return r; }
static std::unordered_set<long long> &grid_ctx_tombstones() { static std::unordered_set<long long> t; return t; }
static long long grid_ctx_salt() {
    static const long long salt = []() { std::random_device rd; long long v = ((long long)rd() & 0x7fffffffLL); return (v == 0 ? 1 : v) << 32; }();
    return salt;
}
static long long g_ctx_next_serial = 1;     // under the registry mutex
static long long g_ctx_default_id = 0;      // the default context's REAL id (0 = none live)
static grid::grid_device_pool_t g_ctx_pending_default_pool = {nullptr, 0, 0, 0};  // set_device_pool before the default exists
// rc codes (the pybind rc decoder names them): 10 unknown or foreign id, 11 closed id,
// 12 context closing, 13 artifact/device arch mismatch, 14 pool/arena creation failed.
static const char *grid_ctx_rc_message(int rc) {
    switch (rc) {
        case 10: return "unknown context id, or a context of another robot artifact";
        case 11: return "context is closed";
        case 12: return "context is closing";
        case 13: return "this robot artifact was compiled for another GPU architecture";
        case 14: return "context creation failed (arena/pool)";
        case 15: return "model mutated between forward and backward (recompute the forward)";
        default: return "grid_rbd runtime error";
    }
}
static int grid_ctx_create_locked(GridCtx **out, const grid::grid_device_pool_t &pool, bool make_default);
struct GridCtxRef {
    GridCtx *ctx = nullptr; int rc = 0; bool exclusive = false;
    // `exclusive` = a MUTATOR: waits for every admitted call to finish and blocks
    // new admissions until it returns. The registry mutex is released BEFORE the
    // admission lock is taken (lock order: registry -> admission; a long setter
    // must not stall create/close/lookup of other contexts). The strong ref
    // (inflight) is taken under the registry mutex atomically with the open check,
    // so close (which drains inflight to 0) never frees a context we are waiting on.
    explicit GridCtxRef(long long id, bool excl = false) : exclusive(excl) {
        {
            std::lock_guard<std::mutex> lk(grid_ctx_mutex());
            auto &reg = grid_ctx_registry();
            long long real = id;
            if (id == 0) {
                if (g_ctx_default_id == 0) {
                    GridCtx *c = nullptr;
                    int e = grid_ctx_create_locked(&c, g_ctx_pending_default_pool, /*make_default=*/true);
                    if (e != 0) { rc = e; return; }
                }
                real = g_ctx_default_id;
            } else if ((id & ~0xffffffffLL) != grid_ctx_salt()) { rc = 10; return; }
            auto it = reg.find(real);
            if (it == reg.end()) { rc = grid_ctx_tombstones().count(real) ? 11 : 10; return; }
            if (it->second->closing) { rc = 12; return; }
            it->second->inflight.fetch_add(1, std::memory_order_acq_rel);
            ctx = it->second;
        }
        if (exclusive) ctx->admission.lock(); else ctx->admission.lock_shared();
    }
    ~GridCtxRef() {
        if (!ctx) return;
        if (exclusive) ctx->admission.unlock(); else ctx->admission.unlock_shared();
        ctx->inflight.fetch_sub(1, std::memory_order_acq_rel);   // AFTER the unlock: close frees at 0
    }
    GridCtxRef(const GridCtxRef &) = delete; GridCtxRef &operator=(const GridCtxRef &) = delete;
};
// Shadow locals: the bodies below keep the historical names; they now alias the
// resolved context. (void)-casts keep -Wunused quiet where a body uses a subset.
#define GRID_RBD_CTX_LOCALS(ctxptr) \
    GridCtx *g_ctx = (ctxptr); grid::gridData<T> *g_data = g_ctx->data; grid::robotModel<T> *g_robot = g_ctx->robot; \
    cudaStream_t *g_streams = g_ctx->streams; PlantBuffers &g_plant = *g_ctx->plant; \
    (void)g_ctx; (void)g_data; (void)g_robot; (void)g_streams; (void)g_plant
#define GRID_RBD_CTX_OR_RETURN(id) GridCtxRef _cref(id); if (!_cref.ctx) return _cref.rc; GRID_RBD_CTX_LOCALS(_cref.ctx)
// Mutators (B2): exclusive admission — every admitted call has completed its
// submission and no new one is admitted until the mutator returns.
#define GRID_RBD_CTX_MUT_OR_RETURN(id) GridCtxRef _cref(id, /*exclusive=*/true); if (!_cref.ctx) return _cref.rc; GRID_RBD_CTX_LOCALS(_cref.ctx)
#define GRID_RBD_CTX_OR_FFI(id) GridCtxRef _cref(id); if (!_cref.ctx) return ffi::Error::Internal(grid_ctx_rc_message(_cref.rc)); GRID_RBD_CTX_LOCALS(_cref.ctx)
#define GRID_RBD_CTX_OR_THROW(id) GridCtxRef _cref(id); TORCH_CHECK(_cref.ctx != nullptr, "grid_rbd: ", grid_ctx_rc_message(_cref.rc)); GRID_RBD_CTX_LOCALS(_cref.ctx)

// ─── W04-B B2: execution-time model-version stamps (K4) ──────────────────────
// A differentiable FORWARD writes the version it was admitted under into a
// caller-owned int32 device slot, stream-ordered after its own work: under jit /
// graph replay the stamp is a data dependency of the backward, produced when the
// forward EXECUTES. The matching GRADIENT call reads that slot (4-byte D2H + a
// stream sync, inside its own admission scope so no mutation can interleave) and
// refuses to run if the model has moved on (rc 15): backward never silently
// differentiates a model the forward did not see. Version wraps at 2^31 (the
// slot is int32 so JAX needs no x64 mode).
__global__ void grid_rbd_stamp_kernel(int *dst, int v) { if (threadIdx.x == 0) *dst = v; }
static inline int grid_ctx_stamp_value(const GridCtx *c) { return (int)(c->version & 0x7fffffffULL); }
static inline void grid_rbd_stamp_write(GridCtx *ctx, cudaStream_t stream, int *dst) {
    grid_rbd_stamp_kernel<<<1, 1, 0, stream>>>(dst, grid_ctx_stamp_value(ctx));
}
static inline int grid_rbd_stamp_check(GridCtx *ctx, cudaStream_t stream, const int *src, int *seen) {
    int h = 0;
    cudaMemcpyAsync(&h, src, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaError_t e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) return (int)e;
    *seen = h;
    return (h == grid_ctx_stamp_value(ctx)) ? 0 : 15;
}
static std::string grid_ctx_stamp_message(int seen, const GridCtx *ctx) {
    return std::string("model mutated between forward and backward (version ") + std::to_string(seen)
        + " -> " + std::to_string(grid_ctx_stamp_value(ctx)) + "); recompute the forward";
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
static inline dim3 grid_rbd_launch_threads_n(const GridCtx *ctx, int batch) {
    // Priority: explicit override > small-batch regime pick > per-algo profile
    // overlay > baked launch_cfg — all PER CONTEXT (B1). The small-batch pick
    // fires only when this call's batch is <= the algo's threshold.
    const LaunchOverrides &lo = ctx->launch;
    if (lo.threads_override >= 1) return dim3((unsigned)lo.threads_override, 1, 1);
    // ALGO == GRID_ALGO_COUNT (plant / no-entry path) hits the primary launch_cfg
    // template; constexpr-exclude it so the per-algo array is never indexed OOB.
    if constexpr (ALGO >= 0 && ALGO < grid::GRID_ALGO_COUNT) {
        if (lo.batch_threshold_per_algo[ALGO] >= 1 && batch >= 1
            && batch <= lo.batch_threshold_per_algo[ALGO]
            && lo.threads_per_algo_small[ALGO] >= 1) {
            return dim3((unsigned)lo.threads_per_algo_small[ALGO], 1, 1);
        }
        if (lo.threads_per_algo[ALGO] >= 1) return dim3((unsigned)lo.threads_per_algo[ALGO], 1, 1);
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
static inline dim3 grid_rbd_grid_for(const GridCtx *ctx, int batch) {
    int slots = (ctx->data && ctx->data->workspace_timestep_slots > 0)
                    ? ctx->data->workspace_timestep_slots : batch;
    int blocks = batch < slots ? batch : slots;
    return dim3((unsigned)(blocks > 0 ? blocks : 1), 1, 1);
}

// ─── lifecycle ───────────────────────────────────────────────────────────────

// ─── context lifecycle ───────────────────────────────────────────────────────
#ifndef GRID_RBD_ARCH
#define GRID_RBD_ARCH 0
#endif
static int grid_ctx_create_locked(GridCtx **out, const grid::grid_device_pool_t &pool, bool make_default) {
    // registry mutex HELD by the caller. Library-safe construction (parts 1+2):
    // every stage publishes only on success; a later failure rolls the earlier
    // stages back through the checked teardown.
    *out = nullptr;
    grid_consume_last_error();
    int dev = 0; cudaGetDevice(&dev);
    int cc_major = 0, cc_minor = 0;
    cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev);
    const int device_cc = cc_major * 10 + cc_minor;
    if (GRID_RBD_ARCH != 0 && device_cc != GRID_RBD_ARCH) {
        fprintf(stderr, "grid_rbd: this robot artifact was compiled for sm_%d but the device is sm_%d\n", (int)GRID_RBD_ARCH, device_cc);
        return 13;
    }
    GridCtx *c = new GridCtx();
    c->pool = pool; c->pool.used = 0;
    c->plant = new PlantBuffers();
    size_t free_b = 0, total_b = 0; cudaMemGetInfo(&free_b, &total_b);
    const char *failed_op = nullptr;
    cudaError_t e = grid::init_grid_checked<T>(&c->streams, &failed_op);
    if (e == cudaSuccess) e = grid::init_robotModel_checked<T>(&c->robot, &failed_op);
    if (e == cudaSuccess) e = grid::init_gridData_checked<T, kMaxBatch>(&c->data, &failed_op, &c->pool);
    if (e != cudaSuccess) {
        fprintf(stderr, "grid_rbd context create: %s failed: %s\n", failed_op ? failed_op : "init", cudaGetErrorString(e));
        grid::close_grid_checked<T>(c->streams, c->robot, c->data);  // null stages are no-ops
        delete c->plant; delete c;
        return 100 + (int)e;
    }
    c->id = grid_ctx_salt() | (g_ctx_next_serial++);
    c->profile.device_cc = device_cc; c->profile.artifact_cc = (int)GRID_RBD_ARCH;
    c->profile.total_bytes = (long long)total_b; c->profile.free_bytes = (long long)free_b;
    c->profile.workspace_slots = c->data->workspace_timestep_slots;
    c->profile.arena_bytes = (long long)grid::gridData_device_bytes<T, kMaxBatch>(c->profile.workspace_slots);
    c->profile.max_batch = kMaxBatch;
    int optin = 0; cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev); c->profile.smem_optin_bytes = optin;
    c->profile.slab_installed = (c->pool.base != nullptr) ? 1 : 0;
    grid_ctx_registry()[c->id] = c;
    if (make_default) g_ctx_default_id = c->id;
    *out = c;
    return 0;
}
static int grid_ctx_close_by_id(long long real_id) {
    GridCtx *c = nullptr;
    {
        std::lock_guard<std::mutex> lk(grid_ctx_mutex());
        auto &reg = grid_ctx_registry();
        auto it = reg.find(real_id);
        if (it == reg.end()) return grid_ctx_tombstones().count(real_id) ? 11 : 10;
        c = it->second;
        c->closing = true;              // no new admissions (C2)
        reg.erase(it);
        grid_ctx_tombstones().insert(real_id);
        if (g_ctx_default_id == real_id) g_ctx_default_id = 0;
    }
    // drain admitted calls (their guards release on return), then complete the
    // device — framework streams may still reference slab-carved scratch.
    while (c->inflight.load(std::memory_order_acquire) > 0) std::this_thread::sleep_for(std::chrono::microseconds(50));
    grid_consume_last_error();
    cudaError_t pending = cudaDeviceSynchronize();
    const char *failed_op = nullptr;
    cudaError_t e = grid::close_grid_checked<T>(c->streams, c->robot, c->data, &failed_op);
    plant_free(c->plant); delete c->plant; c->plant = nullptr;
    delete c;
    if (e != cudaSuccess) fprintf(stderr, "grid_rbd context close: %s failed: %s\n", failed_op ? failed_op : "close_grid", cudaGetErrorString(e));
    return (pending != cudaSuccess) ? 100 + (int)pending : ((e != cudaSuccess) ? 100 + (int)e : 0);
}
// Legacy lifecycle = the DEFAULT context (alias id 0): init creates it if absent,
// close closes it; the next call re-creates it lazily (historical behavior).
extern "C" int grid_rbd_init() {
    std::lock_guard<std::mutex> lk(grid_ctx_mutex());
    if (g_ctx_default_id != 0) return 0;
    GridCtx *c = nullptr;
    return grid_ctx_create_locked(&c, g_ctx_pending_default_pool, /*make_default=*/true);
}
extern "C" int grid_rbd_close() {
    long long real = 0;
    { std::lock_guard<std::mutex> lk(grid_ctx_mutex()); real = g_ctx_default_id; }
    if (real == 0) return 0;
    return grid_ctx_close_by_id(real);
}
// Explicit contexts (B1): create = full init from an OPTIONAL caller-owned slab
// (base=nullptr → the cudaMalloc path), returning the salted id in *out_id.
extern "C" int grid_rbd_ctx_create(void *pool_base, unsigned long long pool_bytes, int ws_slots, long long *out_id) {
    if (!out_id) return 1;
    *out_id = 0;
    grid::grid_device_pool_t pool = {pool_base, (size_t)pool_bytes, 0, ws_slots < 1 ? kMaxBatch : ws_slots};
    std::lock_guard<std::mutex> lk(grid_ctx_mutex());
    GridCtx *c = nullptr;
    int rc = grid_ctx_create_locked(&c, pool, /*make_default=*/false);
    if (rc == 0) *out_id = c->id;
    return rc;
}
extern "C" int grid_rbd_ctx_close(long long id) {
    if (id == 0) return grid_rbd_close();
    if ((id & ~0xffffffffLL) != grid_ctx_salt()) return 10;
    return grid_ctx_close_by_id(id);
}
// The default context's REAL id (creating it if absent): lets a caller hold the
// default like any other context; 0 stays an alias that follows re-creation.
extern "C" int grid_rbd_ctx_default_id(long long *out_id) {
    if (!out_id) return 1;
    GridCtxRef ref(0);
    if (!ref.ctx) return ref.rc;
    *out_id = ref.ctx->id;
    return 0;
}
extern "C" int grid_rbd_ctx_profile(long long id, GridDeviceProfile *out) {
    if (!out) return 1;
    GridCtxRef ref(id);
    if (!ref.ctx) return ref.rc;
    *out = ref.ctx->profile;
    return 0;
}
// B2: the context's model version (bumps on every runtime-parameter mutation).
extern "C" int grid_rbd_ctx_version(long long id, unsigned long long *out) {
    if (!out) return 1;
    GridCtxRef ref(id);
    if (!ref.ctx) return ref.rc;
    *out = ref.ctx->version;
    return 0;
}
extern "C" int grid_rbd_ctx_count() { std::lock_guard<std::mutex> lk(grid_ctx_mutex()); return (int)grid_ctx_registry().size(); }

// ─── device-pool (slab) install for the DEFAULT context ──────────────────────
// A jax/torch surface hands GRiD a device slab from ITS OWN allocator BEFORE the
// default context exists; its arena is then carved from the slab (grid.cuh
// grid_device_alloc, 256-aligned bump) instead of cudaMalloc-ing, so GRiD's VRAM
// lives inside the framework pool. The slab is CALLER-owned (must outlive the
// context; close never frees it; base=nullptr uninstalls). ws_slots declares the
// workspace slot count the slab was sized for (grid_rbd_device_pool_bytes with
// the same count says how big to make it; <1 means kMaxBatch). Explicit contexts
// take their slab at grid_rbd_ctx_create.
extern "C" long long grid_rbd_device_pool_bytes(int ws_slots) {
    return (long long)grid::gridData_device_bytes<T, kMaxBatch>(
        ws_slots < 1 ? kMaxBatch : ws_slots);
}
extern "C" int grid_rbd_set_device_pool(void *base, unsigned long long bytes, int ws_slots) {
    std::lock_guard<std::mutex> lk(grid_ctx_mutex());
    if (g_ctx_default_id != 0) return 1;  // default context already live: close first
    g_ctx_pending_default_pool.base = base;
    g_ctx_pending_default_pool.bytes = (size_t)bytes;
    g_ctx_pending_default_pool.used = 0;
    g_ctx_pending_default_pool.ws_slots = ws_slots < 1 ? kMaxBatch : ws_slots;
    return 0;
}
extern "C" long long grid_rbd_device_pool_used(void) {
    std::lock_guard<std::mutex> lk(grid_ctx_mutex());
    auto it = grid_ctx_registry().find(g_ctx_default_id);
    return it == grid_ctx_registry().end() ? 0 : (long long)it->second->pool.used;
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
extern "C" int grid_rbd_threads_per_block(long long ctx_id) { GridCtxRef r(ctx_id); return r.ctx ? r.ctx->launch.threads_override : -1; }
extern "C" int grid_rbd_set_threads_per_block(long long ctx_id, int n) {
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
    GRID_RBD_CTX_MUT_OR_RETURN(ctx_id);   // B2: exclusive; launch overrides do not bump the model version
    g_ctx->launch.threads_override = (n == 0) ? -1 : n;
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
extern "C" int grid_rbd_set_threads_for(long long ctx_id, int algo, int n) {
    if (algo < 0 || algo >= grid::GRID_ALGO_COUNT || n < 0) return 1;
    GRID_RBD_CTX_MUT_OR_RETURN(ctx_id);
    g_ctx->launch.threads_per_algo[algo] = (n == 0) ? -1 : n;
    return 0;
}

// Batch-regime overlay setter (E6 batch-switch): when a call's batch is
// <= threshold, that algo launches with n_small threads instead of its
// per-algo overlay / baked default. threshold == 0 clears the switch for
// that algo (n_small ignored). Applied at handle init from the launch-config
// <profile>_bases_by_n block; process-global like the other overlays.
extern "C" int grid_rbd_set_threads_for_n(long long ctx_id, int algo, int threshold, int n_small) {
    if (algo < 0 || algo >= grid::GRID_ALGO_COUNT || threshold < 0) return 1;
    GRID_RBD_CTX_MUT_OR_RETURN(ctx_id);
    if (threshold == 0) {
        g_ctx->launch.batch_threshold_per_algo[algo] = 0;
        g_ctx->launch.threads_per_algo_small[algo] = -1;
        return 0;
    }
    if (n_small < 1) return 1;
    g_ctx->launch.batch_threshold_per_algo[algo] = threshold;
    g_ctx->launch.threads_per_algo_small[algo] = n_small;
    return 0;
}

// Introspection for tests / launch_info: the switch state for one algo.
// Returns 0 and fills (threshold, n_small); threshold 0 = no switch armed.
extern "C" int grid_rbd_get_batch_switch(long long ctx_id, int algo, int *threshold, int *n_small) {
    if (algo < 0 || algo >= grid::GRID_ALGO_COUNT || !threshold || !n_small) return 1;
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    *threshold = g_ctx->launch.batch_threshold_per_algo[algo];
    *n_small = g_ctx->launch.threads_per_algo_small[algo];
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
// ── BEGIN GENERATED KERNEL_MAX_THREADS BRANCHES (grid_codegen/wrapper_body_gen.py — do not hand-edit) ──
// Regenerate: .venv/bin/python -m grid_codegen.wrapper_body_gen
// Rows: CEIL_ROWS (keys crosschecked against the descriptor table's
// autotune_keys by test/test_abi_spec_crosscheck.py).
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
    // ─ batch-2 coverage extension (2026-08-26): keys are the FULL symbol names ─
#if GRID_HAS_F_EXT_GRADIENT
    if (std::strcmp(algo, "f_ext_gradient") == 0)
        return GRID_KERNEL_CEIL(f_ext_gradient_kernel, GRID_ALGO_F_EXT_GRADIENT,
                                void(*)(T*, T*, unsigned char*, const T*, const int, RM, const int));
#endif
#if GRID_HAS_F_EXT_GRADIENT_DQ
    if (std::strcmp(algo, "f_ext_gradient_dq") == 0)
        return GRID_KERNEL_CEIL(f_ext_gradient_dq_kernel, GRID_ALGO_F_EXT_GRADIENT_DQ,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const int));
#endif
#if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
    if (std::strcmp(algo, "inverse_dynamics_regressor") == 0)
        return GRID_KERNEL_CEIL(inverse_dynamics_regressor_kernel, GRID_ALGO_INVERSE_DYNAMICS_REGRESSOR,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const T, const int));
#endif
#if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
    if (std::strcmp(algo, "forward_dynamics_parameter_gradient") == 0)
        return GRID_KERNEL_CEIL(forward_dynamics_parameter_gradient_kernel, GRID_ALGO_FORWARD_DYNAMICS_PARAMETER_GRADIENT,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const T, const int));
#endif
#if GRID_HAS_KINETIC_ENERGY_REGRESSOR
    if (std::strcmp(algo, "kinetic_energy_regressor") == 0)
        return GRID_KERNEL_CEIL(kinetic_energy_regressor_kernel, GRID_ALGO_KINETIC_ENERGY_REGRESSOR,
                                void(*)(T*, const T*, const int, RM, const T, const int));
#endif
#if GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
    if (std::strcmp(algo, "potential_energy_regressor") == 0)
        return GRID_KERNEL_CEIL(potential_energy_regressor_kernel, GRID_ALGO_POTENTIAL_ENERGY_REGRESSOR,
                                void(*)(T*, const T*, const int, RM, const T, const int));
#endif
#if GRID_HAS_FRAME_JACOBIAN
    if (std::strcmp(algo, "frame_jacobian") == 0)
        return GRID_KERNEL_CEIL(frame_jacobian_kernel, GRID_ALGO_FRAME_JACOBIAN,
                                void(*)(T*, const T*, const int, const int, const int, RM, const int));
#endif
#if GRID_HAS_FRAME_JACOBIAN_DOT
    if (std::strcmp(algo, "frame_jacobian_dot") == 0)
        return GRID_KERNEL_CEIL(frame_jacobian_dot_kernel, GRID_ALGO_FRAME_JACOBIAN_DOT,
                                void(*)(T*, const T*, const int, const int, const int, RM, const int));
#endif
#if GRID_HAS_OSC_INERTIA
    if (std::strcmp(algo, "osc_inertia") == 0)
        return GRID_KERNEL_CEIL(osc_inertia_kernel, GRID_ALGO_OSC_INERTIA,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const int));
#endif
#if GRID_HAS_GENERALIZED_GRAVITY
    if (std::strcmp(algo, "generalized_gravity") == 0)
        return GRID_KERNEL_CEIL(generalized_gravity_kernel, GRID_ALGO_GENERALIZED_GRAVITY,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const T, const int));
#endif
#if GRID_HAS_NONLINEAR_EFFECTS
    if (std::strcmp(algo, "nonlinear_effects") == 0)
        return GRID_KERNEL_CEIL(nonlinear_effects_kernel, GRID_ALGO_NONLINEAR_EFFECTS,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const T, const int));
#endif
#if GRID_HAS_ENERGY
    if (std::strcmp(algo, "energy") == 0)
        return GRID_KERNEL_CEIL(energy_kernel, GRID_ALGO_ENERGY,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const T, const int));
#endif
#if GRID_HAS_COM
    if (std::strcmp(algo, "com") == 0)
        return GRID_KERNEL_CEIL(com_kernel, GRID_ALGO_COM,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const int));
#endif
#if GRID_HAS_CCRBA
    if (std::strcmp(algo, "ccrba") == 0)
        return GRID_KERNEL_CEIL(ccrba_kernel, GRID_ALGO_CCRBA,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const int));
#endif
#if GRID_HAS_CORIOLIS_MATRIX
    if (std::strcmp(algo, "coriolis_matrix") == 0)
        return GRID_KERNEL_CEIL(coriolis_matrix_kernel, GRID_ALGO_CORIOLIS_MATRIX,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const T, const int));
#endif
#if GRID_HAS_DCCRBA
    if (std::strcmp(algo, "dccrba") == 0)
        return GRID_KERNEL_CEIL(dccrba_kernel, GRID_ALGO_DCCRBA,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const int));
#endif
#if GRID_HAS_CMM_TIME_VARIATION
    if (std::strcmp(algo, "cmm_time_variation") == 0)
        return GRID_KERNEL_CEIL(cmm_time_variation_kernel, GRID_ALGO_CMM_TIME_VARIATION,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const int));
#endif
// ── END GENERATED KERNEL_MAX_THREADS BRANCHES ──
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
// One gated setter per runtime-mutable device parameter table: exports
// grid_rbd_set_<name>_params (thin sync'd call into the generated
// grid::set_<name>_params cudaMemcpy) + grid_rbd_<name>_params_size.
// Note: unlike the algorithm epilogues these return the RAW cudaError_t (not
// 100+e) on failure — preserved for ABI compatibility.
#define GRID_RBD_RUNTIME_PARAM_SETTER(NAME, SIZE_EXPR)                          \
extern "C" int grid_rbd_set_##NAME##_params(long long ctx_id, const T* h_params) { \
    GRID_RBD_CTX_MUT_OR_RETURN(ctx_id); /* B2: exclusive admission */          \
    ++g_ctx->version;                   /* new version BEFORE any byte moves */ \
    cudaDeviceSynchronize(); /* drain framework streams reading the table */   \
    grid::set_##NAME##_params<T>(g_robot, h_params);                            \
    cudaError_t err = cudaDeviceSynchronize();                                  \
    if (err == cudaSuccess) err = grid_consume_last_error(); /* NO_EXIT sticky */\
    return (err == cudaSuccess) ? 0 : (int)err;                                 \
}                                                                               \
extern "C" int grid_rbd_##NAME##_params_size() { return (SIZE_EXPR); }

#ifdef GRID_RBD_RUNTIME_INERTIA
GRID_RBD_RUNTIME_PARAM_SETTER(inertia, 10 * grid::NUM_BODIES)
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
GRID_RBD_RUNTIME_PARAM_SETTER(transform, 6 * grid::NUM_JOINTS)
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
GRID_RBD_RUNTIME_PARAM_SETTER(joint_dynamics, 2 * grid::NUM_VEL)
#endif

// ─── shared input-packing helper ─────────────────────────────────────────────
//
// h_q_qd_u layout (matches generated host wrappers):
//   timestep t: [q[0..NJ-1], qd[0..NJ-1], u[0..NJ-1]]
//   contiguous across t.

static inline void pack_q_qd_u(GridCtx *ctx, const T* q, const T* qd, const T* u,
                               int batch, int num_joints)
{
    GRID_RBD_CTX_LOCALS(ctx);
    // Drain in-flight framework work first (audit W04-B class, 2026-09-20): the jax/torch
    // handlers enqueue async copies, kernels and the f_ext reset on THEIR (non-blocking)
    // streams into these same g_data buffers; the numpy path stages synchronously on the
    // legacy default stream, so without this a framework memset could land between our
    // staging and our launch (seen: numpy ID computed against a zeroed d_f_ext).
    cudaDeviceSynchronize();
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
// apply_f_ext(g_ctx) copies the user's wrench into the device buffer; reset_f_ext(g_ctx)
// re-zeroes it after the launch so a later no-f_ext call sees a clean buffer
// (the singleton is shared across calls). batch must be <= kMaxBatch.

static inline int apply_f_ext(GridCtx *ctx, const T* f_ext, int batch) {
    GRID_RBD_CTX_LOCALS(ctx);
    if (!f_ext) return 0;
    const size_t n = (size_t)6 * grid::NUM_BODIES * batch;
    std::memcpy(g_data->h_f_ext, f_ext, n * sizeof(T));
    if (cudaMemcpy(g_data->d_f_ext, g_data->h_f_ext, n * sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 5;
    return 0;
}

static inline void reset_f_ext(GridCtx *ctx, const T* f_ext, int batch) {
    GRID_RBD_CTX_LOCALS(ctx);
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

// Direct mass-matrix inverse: Minv(q)

// Forward dynamics: qdd = Minv(q)·(τ − c(q,qd))
// f_ext (optional, may be null): (batch, 6*NUM_BODIES) local-frame body wrenches.

// Articulated body algorithm: qdd = aba(q, qd, u)
// f_ext (optional, may be null): (batch, 6*NUM_BODIES) local-frame body wrenches.

// Composite rigid body algorithm: M = crba(q)

// End-effector pose: 6×NUM_EES per timestep (xyz + rpy).

// Batched forward kinematics (large-batch, one block/warp per sample):
//   q layout:     (batch, NUM_POS)            -> q[b*NUM_POS + j]
//   pose7 layout: (batch, 7) = [tx,ty,tz, qw,qx,qy,qz]
// use_warp selects the warp-cooperative inner (1) vs the thread inner (0).
// Only present when the generated header emits the standalone FK inner —
// fixed AND floating bases, mimic included (registry A2, 2026-08). rc=3 only
// for spherical-joint robots, >32-joint robots (warp inner's lane==jid cap),
// or a reduced codegen profile that skipped the EE-pose family.
extern "C" int grid_rbd_fk_batched(long long ctx_id, 
    const T* q,
    T* pose7_out,
    int batch, int use_warp)
{
#ifdef GRID_HAS_FK_BATCHED
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
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
                                                        (int)grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch).x < 32 ? 32 : (int)grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch).x);
    else
        grid::ee_pose_fk_batched<T, /*USE_WARP=*/false>(d_pose7, d_q_fk, batch, g_robot,
                                                        (int)grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch).x);

    if (int rc = grid_rbd_sync_consume()) return rc;

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

// ∂c/∂(q, qd): output shape (batch, NV, 2*NV) — concatenated [dc_dq | dc_dqd]
// (tangent-space; FIXED base NV == NJ, FLOATING base NV < NJ).
// f_ext (optional, may be null): (batch, 6*NUM_BODIES) local-frame body wrenches.
// The q-Jacobian depends on the supplied body-local force even when that force
// is held constant. Value and gradient calls must use the same f_ext.

// ∂qdd/∂(q, qd): output shape (batch, NV, 2*NV) (tangent-space; FIXED base
// NV == NJ, FLOATING base NV < NJ).
// f_ext (optional, may be null): (batch, 6*NUM_BODIES) local-frame body wrenches.

// End-effector pose Hessian: 6×NUM_EES×NV×NV per timestep (d^2/dv^2 tangent).
// Calls grid::end_effector_pose_hessian which fills BOTH end_effector_pose_hessian AND
// end_effector_pose_gradient; we only copy end_effector_pose_hessian out. If the caller wants both they should
// call end_effector_pose_gradient separately (the kernels are fast enough
// that doing the work twice is fine for a small convenience).

// Second-order inverse dynamics. Output is the concatenated SO tensor of
// shape SECOND_ORDER_TENSOR_SIZE = 4 * NV^3 per timestep (four NV^3 blocks:
// d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq). The Python side slices into the
// four named tensors.

// Second-order forward dynamics. Output is 4 * NV^3 per timestep
// (d2qdd_dq, d2qdd_dqd, d2qdd_dudq — interpretation per Singh/Wensing).


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
static inline void pack_q(GridCtx *ctx, const T* q, int batch, int num_joints) {
    GRID_RBD_CTX_LOCALS(ctx);
    std::memcpy(g_data->h_q, q, (size_t)batch * num_joints * sizeof(T));
}

// com(q) -> [p_com(3); J_com(3 x NV)] per timestep, total 3 + 3*NUM_VEL floats.
// Gated on GRID_HAS_COM: com/ccrba/energy ARE emitted for mimic robots (the
// centroidal_inner Jacobian fold is alpha-reduced; validated on fr3/h1_2 by
// cuda_centroidal_mimic_smoke_runner.cu). rc=3 only when a reduced codegen
// profile didn't generate com for this robot.


// ccrba(q, qd) -> [A(6 x NV); h(6)] per timestep, total 6*NUM_VEL + 6 floats.
// Gated on GRID_HAS_CCRBA (emitted for mimic too, alpha-folded); rc=3 only when a
// reduced codegen profile didn't generate ccrba for this robot.

// energy(q, qd) -> [KE, PE, KE+PE] per timestep, total 3 floats. Takes gravity.
// Gated on GRID_HAS_ENERGY (emitted for mimic too, alpha-folded); rc=3 only when a
// reduced codegen profile didn't generate energy for this robot.

// generalized_gravity(q) -> g(q) = RNEA(q,0,0) per timestep, NUM_VEL floats. Takes gravity.

// nonlinear_effects(q, qd) -> c(q,qd) = RNEA(q,qd,0) per timestep, NUM_VEL floats. Takes gravity.

// coriolis_matrix(q, qd) -> nv x nv Coriolis matrix C(q,qd), row-major
// (C[row*nv + col]). Always emitted with the "all" profile (mimic-safe:
// alpha-folded column assembly), so it is bound UNGATED like com/ccrba.

// kinetic_energy_regressor(q, qd) -> length 10*NUM_BODIES regressor y_KE
// (KE = y_KE . pi). Always emitted with the "all" profile (mimic-safe), ungated.

// potential_energy_regressor(q) -> length 10*NUM_BODIES regressor y_PE
// (PE = y_PE . pi). Always emitted with the "all" profile (mimic-safe), ungated.
// Reads the COMPRESSED input layout (h_q / d_q) like com.

// dccrba(q) -> 6*NUM_VEL*NUM_VEL dCCRBA tensor dA/dq (per timestep, as the kernel
// writes it). Reads the COMPRESSED input layout (h_q / d_q). Gated on
// GRID_HAS_DCCRBA: dccrba IS emitted for mimic robots (alpha-folded; validated on
// fr3/h1_2 by test_cuda_dccrba.py), so rc=3 only when a reduced codegen profile
// didn't generate dccrba. For big floating robots whose kernel arena overflows the
// smem cap the host wrapper's grid_check_dynamic_shared_memory_bytes raises a clear
// rc!=0 at launch (the big-floating spill ladder de-gated g1/h1_2; commit c4b3900).

// cmm_time_variation(q, qd) -> 6*NUM_VEL centroidal-momentum-matrix time
// variation Adot (per timestep). Gated on GRID_HAS_CMM_TIME_VARIATION (emitted for
// mimic too, alpha-folded); returns rc=3 only when a reduced profile didn't generate it.

// frame_jacobian(q) -> 6 x NUM_VEL geometric Jacobian (col-major, [linear;angular])
// at the leaf-EE frame, LOCAL_WORLD_ALIGNED. Gated on GRID_HAS_FRAME_JACOBIAN
// (the frame_jacobian family is opt-in codegen; only present when requested).

// frame_jacobian_dot(q, qd) -> d/dt of the leaf-EE frame Jacobian along v=qd,
// 6 x NUM_VEL (col-major, [linear;angular]). Gated on GRID_HAS_FRAME_JACOBIAN_DOT
// (dot is opt-in on top of frame_jacobian; a frame_jacobian-only build emits no
// dot symbols).

// osc_inertia(q) -> 6x6 operational-space (task) inertia Lambda = (J Minv J^T)^-1
// at the leaf-EE frame (LWA), 36 floats per timestep. Gated on GRID_HAS_OSC_INERTIA
// (osc_inertia is opt-in on top of frame_jacobian; a frame_jacobian-only build
// emits no osc symbols).


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

extern "C" int grid_rbd_tool_fext(long long ctx_id, const T* q, const T* wrench, int jid, const T* rc,
                                  T* out, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(g_ctx, q, q, nullptr, batch, nj);   // q at offset 0, stride 3*nj
    const int stride_q = 3 * nj;
    if (cudaMemcpy(g_data->d_q_qd_u, g_data->h_q_qd_u,
                   (size_t)batch * stride_q * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) return 5;
    T *d_wrench = nullptr, *d_rc = nullptr;
    if (cudaMalloc(&d_wrench, (size_t)6 * batch * sizeof(T)) != cudaSuccess) return 6;
    if (cudaMalloc(&d_rc, 3 * sizeof(T)) != cudaSuccess) { cudaFree(d_wrench); return 6; }
    cudaMemcpy(d_wrench, wrench, (size_t)6 * batch * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rc, rc, 3 * sizeof(T), cudaMemcpyHostToDevice);
    // Sized by the contact family's OWN arena constant (emitted next to
    // GRID_HAS_CONTACT_RUNTIME). It used max(F_EXT_GRADIENT, EE_POSE)+4096 —
    // constants of families a dynamics-only subset does not build.
    size_t smem = grid::F_EXT_CONTACT_RUNTIME_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_rbd_tool_fext_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    grid_rbd_tool_fext_kernel<<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), smem>>>(
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
// Baked multi-contact wrenches -> joint-local f_ext (registered contact frames)
// ────────────────────────────────────────────────────────────────────────────
//
// contact_fext(q, f_c) -> (batch, 6*NUM_BODIES) joint-local f_ext, ready for
// inverse_dynamics(f_ext=...) / forward_dynamics(f_ext=...) / aba(f_ext=...).
// `f_c` is (batch, 6*NUM_CONTACT_FRAMES): per registered frame (registration
// order) a world-aligned [n_w; f_w] 6-vector, moment about the contact-frame
// origin (pinocchio LOCAL_WORLD_ALIGNED — the same convention as tool_fext's
// wrench). Gated on GRID_HAS_CONTACT_FRAMES (register with contact_frames=[...]).
// Mirrors grid_rbd_tool_fext exactly, but calls the BAKED grid::f_ext_body_device
// (per-frame body ids + offsets compiled in; per-body sums in a baked order — no
// atomics, deterministic).
extern "C" int grid_rbd_num_contact_frames() {
#ifdef GRID_HAS_CONTACT_FRAMES
    return grid::NUM_CONTACT_FRAMES;
#else
    return 0;
#endif
}
#ifdef GRID_HAS_CONTACT_FRAMES
__global__ void grid_rbd_contact_fext_kernel(const T* d_q, int stride_q,
                                             const T* d_fc,
                                             const grid::robotModel<T>* d_robotModel,
                                             T* d_workspace, T* d_out) {
    const int k = blockIdx.x;
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    __shared__ T s_fext[6 * grid::NUM_BODIES];
    grid::f_ext_body_device<T>(s_fext, &d_fc[k * 6 * grid::NUM_CONTACT_FRAMES],
                               &d_q[k * stride_q], d_robotModel, d_workspace);
    __syncthreads();
    for (int i = tid; i < 6 * grid::NUM_BODIES; i += nth)
        d_out[k * 6 * grid::NUM_BODIES + i] = s_fext[i];
}

extern "C" int grid_rbd_contact_fext(long long ctx_id, const T* q, const T* f_c, T* out, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    const int nj = grid::NUM_JOINTS;
    const int fc_stride = 6 * grid::NUM_CONTACT_FRAMES;
    pack_q_qd_u(g_ctx, q, q, nullptr, batch, nj);   // q at offset 0, stride 3*nj
    const int stride_q = 3 * nj;
    if (cudaMemcpy(g_data->d_q_qd_u, g_data->h_q_qd_u,
                   (size_t)batch * stride_q * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) return 5;
    T *d_fc = nullptr;
    if (cudaMalloc(&d_fc, (size_t)fc_stride * batch * sizeof(T)) != cudaSuccess) return 6;
    cudaMemcpy(d_fc, f_c, (size_t)fc_stride * batch * sizeof(T), cudaMemcpyHostToDevice);
    // Sized by the contact family's OWN arena constant (emitted next to
    // NUM_CONTACT_FRAMES) — see the note on grid_rbd_tool_fext above.
    size_t smem = grid::F_EXT_CONTACT_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_rbd_contact_fext_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    grid_rbd_contact_fext_kernel<<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), smem>>>(
        g_data->d_q_qd_u, stride_q, d_fc, g_robot,
        reinterpret_cast<T*>(g_data->d_workspace), g_data->d_f_ext);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) e = grid_consume_last_error();
    cudaFree(d_fc);
    if (e != cudaSuccess) return 100 + (int)e;
    if (cudaMemcpy(out, g_data->d_f_ext, (size_t)6 * grid::NUM_BODIES * batch * sizeof(T),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return 5;
    // d_f_ext is the shared f_ext buffer; re-zero it so a later NO-f_ext dynamics
    // call is not polluted by the contact wrenches we just wrote.
    cudaMemset(g_data->d_f_ext, 0, (size_t)6 * grid::NUM_BODIES * kMaxBatch * sizeof(T));
    return 0;
}
#else
extern "C" int grid_rbd_contact_fext(const T*, const T*, T*, int) { return 3; }
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
static void launch_integrator_host(GridCtx *ctx, int batch, T gravity, T dt) {
    GRID_RBD_CTX_LOCALS(ctx);
// signature switch (same rule as fdsva_so): floating builds carry MUJOCO_OUTPUT.
// Tier must match the per-algo autotuned thread count (see idsva_so above).
#if defined(GRID_RBD_SIG_MJX_INTEGRATOR)
    grid::integrator<T, IT, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER>(
#else
    grid::integrator<T, IT, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER>(
#endif
        g_data, g_robot, /*gravity=*/gravity,
        dt, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_INTEGRATOR>(g_ctx, batch), g_streams);
}
#ifdef GRID_RBD_WITH_MUJOCO
template <grid::IntegratorType IT>
static void launch_integrator_host_mujoco(GridCtx *ctx, int batch, T gravity, T dt) {
    GRID_RBD_CTX_LOCALS(ctx);
    grid::integrator<T, IT, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER>(
        g_data, g_robot, /*gravity=*/gravity,
        dt, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_INTEGRATOR>(g_ctx, batch), g_streams);
}
#endif
#endif  // GRID_HAS_INTEGRATOR
#if GRID_HAS_INTEGRATOR_GRADIENT
template <grid::IntegratorType IT>
static void launch_integrator_grad_host(GridCtx *ctx, int batch, T gravity, T dt) {
    GRID_RBD_CTX_LOCALS(ctx);
    grid::integrator_gradient<T, IT>(g_data, g_robot, /*gravity=*/gravity,
                                     dt, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_INTEGRATOR_GRADIENT>(g_ctx, batch), g_streams);
}
#ifdef GRID_RBD_WITH_MUJOCO
template <grid::IntegratorType IT>
static void launch_integrator_grad_host_mujoco(GridCtx *ctx, int batch, T gravity, T dt) {
    GRID_RBD_CTX_LOCALS(ctx);
    grid::integrator_gradient<T, IT, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR_GRADIENT>::TIER>(
        g_data, g_robot, /*gravity=*/gravity,
        dt, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_INTEGRATOR_GRADIENT>(g_ctx, batch), g_streams);
}
#endif
#endif  // GRID_HAS_INTEGRATOR_GRADIENT

#define GRID_RBD_IT_DISPATCH(it_code, FN, ...)                                   \
    switch (it_code) {                                                           \
        case 0: FN<grid::IntegratorType::EULER>(g_ctx, __VA_ARGS__); break;             \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER>(g_ctx, __VA_ARGS__); break;\
        case 2: FN<grid::IntegratorType::MIDPOINT>(g_ctx, __VA_ARGS__); break;          \
        case 3: FN<grid::IntegratorType::RK3>(g_ctx, __VA_ARGS__); break;               \
        case 4: FN<grid::IntegratorType::RK4>(g_ctx, __VA_ARGS__); break;               \
        case 5: FN<grid::IntegratorType::TRAPEZOIDAL>(g_ctx, __VA_ARGS__); break;               \
        default: return 3;                                                       \
    }

// plant_step_hessian only supports EULER / SI-EULER (the composed device fn
// static_asserts MIDPOINT/RK out — instantiating those cases would fail to
// COMPILE, so this dispatch never names them). Other IT codes return rc=3.
#define GRID_RBD_IT_DISPATCH_HESSIAN(it_code, FN, ...)                           \
    switch (it_code) {                                                           \
        case 0: FN<grid::IntegratorType::EULER>(g_ctx, __VA_ARGS__); break;             \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER>(g_ctx, __VA_ARGS__); break;\
        default: return 3;                                                       \
    }

// Shared (IT, MUJOCO_FLAG) two-template-arg switch cores for the jax-FFI /
// torch dispatchers below (which differ only in their bad-code statement).
// ERR_STMT is the whole surface-specific statement (ffi::Error return vs
// TORCH_CHECK). The _SS core is single-stage (EULER / SEMI_IMPLICIT_EULER)
// ONLY — the mjx gradient / plant-step epilogues static_assert multi-stage
// out, so those dispatchers must never name the MIDPOINT/RK/TRAPEZOIDAL cases.
#define GRID_RBD_IT_SWITCH_MJX(it_code, FN, MUJOCO_FLAG, ERR_STMT, ...)                         \
    switch (it_code) {                                                                          \
        case 0: FN<grid::IntegratorType::EULER, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break;               \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break; \
        case 2: FN<grid::IntegratorType::MIDPOINT, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break;            \
        case 3: FN<grid::IntegratorType::RK3, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break;                 \
        case 4: FN<grid::IntegratorType::RK4, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break;                 \
        case 5: FN<grid::IntegratorType::TRAPEZOIDAL, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break;         \
        default: ERR_STMT;                                                                      \
    }
#define GRID_RBD_IT_SWITCH_MJX_SS(it_code, FN, MUJOCO_FLAG, ERR_STMT, ...)                      \
    switch (it_code) {                                                                          \
        case 0: FN<grid::IntegratorType::EULER, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break;               \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break; \
        default: ERR_STMT;                                                                      \
    }


// ── BEGIN GENERATED C-ABI BODIES (grid_codegen/wrapper_body_gen.py — do not hand-edit) ──
// Regenerate: .venv/bin/python -m grid_codegen.wrapper_body_gen
// Table: grid_codegen/abi_specs.py (ABI_SPECS); drift-gated by
// test/test_wrapper_generated_block.py.

extern "C" int grid_rbd_nonlinear_effects(long long ctx_id, const T* q, const T* qd, T* out, int batch, T gravity) {
#if GRID_HAS_NONLINEAR_EFFECTS
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::nonlinear_effects<T>(g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_NONLINEAR_EFFECTS>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_c, (size_t)batch * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch; (void)gravity;
    return 3;  // nonlinear_effects not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_generalized_gravity(long long ctx_id, const T* q, T* out, int batch, T gravity) {
#if GRID_HAS_GENERALIZED_GRAVITY
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    grid::generalized_gravity<T>(g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_GENERALIZED_GRAVITY>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_c, (size_t)batch * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)gravity;
    return 3;  // generalized_gravity not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_coriolis_matrix(long long ctx_id, const T* q, const T* qd, T* out, int batch, T gravity) {
#if GRID_HAS_CORIOLIS_MATRIX
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::coriolis_matrix<T>(g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_CORIOLIS_MATRIX>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_coriolis, (size_t)batch * grid::NUM_VEL*grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch; (void)gravity;
    return 3;  // coriolis_matrix not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_energy(long long ctx_id, const T* q, const T* qd, T* out, int batch, T gravity) {
#ifdef GRID_HAS_ENERGY
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::energy<T>(g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_ENERGY>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_energy, (size_t)batch * 3 * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch; (void)gravity;
    return 3;  // energy not generated for this robot (reduced codegen profile)
#endif
}

extern "C" int grid_rbd_com(long long ctx_id, const T* q, T* out, int batch) {
#ifdef GRID_HAS_COM
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q(g_ctx, q, batch, grid::NUM_JOINTS);
    grid::com<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_COM>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_com, (size_t)batch * (3 + 3 * grid::NUM_VEL) * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch;
    return 3;  // com not generated for this robot (reduced codegen profile)
#endif
}

extern "C" int grid_rbd_ccrba(long long ctx_id, const T* q, const T* qd, T* out, int batch) {
#ifdef GRID_HAS_CCRBA
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::ccrba<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_clamp_threads_for(grid::ccrba_kernel<T>, grid_rbd_launch_threads_n<grid::GRID_ALGO_CCRBA>(g_ctx, batch)), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_ccrba, (size_t)batch * (6 * grid::NUM_VEL + 6) * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch;
    return 3;  // ccrba not generated for this robot (reduced codegen profile)
#endif
}

extern "C" int grid_rbd_dccrba(long long ctx_id, const T* q, T* out, int batch) {
#ifdef GRID_HAS_DCCRBA
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q(g_ctx, q, batch, grid::NUM_JOINTS);
    grid::dccrba<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_clamp_threads_for(grid::dccrba_kernel<T>, grid_rbd_launch_threads_n<grid::GRID_ALGO_DCCRBA>(g_ctx, batch)), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_dccrba, (size_t)batch * 6*grid::NUM_VEL*grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch;
    return 3;  // dccrba not generated for this robot (reduced codegen profile)
#endif
}

extern "C" int grid_rbd_cmm_time_variation(long long ctx_id, const T* q, const T* qd, T* out, int batch) {
#ifdef GRID_HAS_CMM_TIME_VARIATION
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::cmm_time_variation<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_clamp_threads_for(grid::cmm_time_variation_kernel<T>, grid_rbd_launch_threads_n<grid::GRID_ALGO_CMM_TIME_VARIATION>(g_ctx, batch)), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_cmm_time_variation, (size_t)batch * 6*grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch;
    return 3;  // cmm_time_variation not generated for this robot (mimic)
#endif
}

extern "C" int grid_rbd_kinetic_energy_regressor(long long ctx_id, const T* q, const T* qd, T* out, int batch, T gravity) {
#if GRID_HAS_KINETIC_ENERGY_REGRESSOR
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::kinetic_energy_regressor<T>(g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_KINETIC_ENERGY_REGRESSOR>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_ke_regressor, (size_t)batch * 10*grid::NUM_BODIES * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch; (void)gravity;
    return 3;  // kinetic_energy_regressor not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_potential_energy_regressor(long long ctx_id, const T* q, T* out, int batch, T gravity) {
#if GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q(g_ctx, q, batch, grid::NUM_JOINTS);
    grid::potential_energy_regressor<T>(g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_POTENTIAL_ENERGY_REGRESSOR>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_pe_regressor, (size_t)batch * 10*grid::NUM_BODIES * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)gravity;
    return 3;  // potential_energy_regressor not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_frame_jacobian(long long ctx_id, const T* q, T* out, int batch, int target_jid, int reference_frame) {
#ifdef GRID_HAS_FRAME_JACOBIAN
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (target_jid < 0 || target_jid >= grid::NUM_JOINTS) return 1;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    // -1 per arg => "use default" (leaf-EE / LWA); the host resolves each
    // INDEPENDENTLY, so a default target with an explicit frame is honored.
    grid::frame_jacobian<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_FRAME_JACOBIAN>(g_ctx, batch), g_streams, target_jid, reference_frame);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_frame_jacobian, (size_t)batch * 6*grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)target_jid; (void)reference_frame;
    return 3;  // frame_jacobian not generated for this .so
#endif
}

extern "C" int grid_rbd_frame_jacobian_dot(long long ctx_id, const T* q, const T* qd, T* out, int batch, int target_jid, int reference_frame) {
#ifdef GRID_HAS_FRAME_JACOBIAN_DOT
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (target_jid < 0 || target_jid >= grid::NUM_JOINTS) return 1;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    // -1 per arg => "use default" (leaf-EE / LWA); the host resolves each
    // INDEPENDENTLY, so a default target with an explicit frame is honored.
    grid::frame_jacobian_dot<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_FRAME_JACOBIAN_DOT>(g_ctx, batch), g_streams, target_jid, reference_frame);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_frame_jacobian_dot, (size_t)batch * 6*grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch; (void)target_jid; (void)reference_frame;
    return 3;  // frame_jacobian_dot not built
#endif
}

extern "C" int grid_rbd_osc_inertia(long long ctx_id, const T* q, T* out, int batch) {
#ifdef GRID_HAS_OSC_INERTIA
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    grid::osc_inertia<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_OSC_INERTIA>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_osc_inertia, (size_t)batch * 36 * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch;
    return 3;  // osc_inertia not built
#endif
}

extern "C" int grid_rbd_minv(long long ctx_id, const T* q, T* minv_out, int batch) {
#if GRID_HAS_MINV
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_MINV)
    grid::minv<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER>(
#else
    grid::minv<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER>(
#endif
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_MINV>(g_ctx, batch), g_streams);

    if (int rc = grid_rbd_sync_consume()) return rc;

    // Device-direct copy at the nv*nv kernel stride (same nj-stride staging issue
    // as crba).
    cudaMemcpy(minv_out, g_data->d_Minv, (size_t)batch * grid::NUM_VEL*grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
#else
    (void)q; (void)minv_out; (void)batch;
    return 3;  // minv not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_crba(long long ctx_id, const T* q, T* m_out, int batch, T gravity) {
#if GRID_HAS_CRBA
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_CRBA)
    grid::crba<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER>(
#else
    grid::crba<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER>(
#endif
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_CRBA>(g_ctx, batch), g_streams);

    if (int rc = grid_rbd_sync_consume()) return rc;

    // Device-direct copy at the nv*nv kernel stride: the generated host wrapper's
    // h_M staging used the nj*nj stride (over-read for floating; see the nj-stride
    // host-wrapper item). FIXED base has nv == nj so this is byte-identical.
    cudaMemcpy(m_out, g_data->d_M, (size_t)batch * grid::NUM_VEL*grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
#else
    (void)q; (void)m_out; (void)batch; (void)gravity;
    return 3;  // crba not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_end_effector_pose(long long ctx_id, const T* q, T* ee_out, int batch) {
#if GRID_HAS_END_EFFECTOR_POSE
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_EE_POSE)
    grid::GRID_RBD_EE_POSE_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE>::TIER>(
#else
    grid::GRID_RBD_EE_POSE_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE>::TIER>(
#endif
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_END_EFFECTOR_POSE>(g_ctx, batch), g_streams);

    if (int rc = grid_rbd_sync_consume()) return rc;

    std::memcpy(ee_out, g_data->h_end_effector_pose, (size_t)batch * 6*GRID_RBD_NUM_EES * sizeof(T));
    return 0;
#else
    (void)q; (void)ee_out; (void)batch;
    return 3;  // end_effector_pose not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_end_effector_pose_gradient(long long ctx_id, const T* q, T* dee_out, int batch) {
#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_EE_POSE_GRADIENT)
    grid::GRID_RBD_EE_POSE_GRADIENT_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER>(
#else
    grid::GRID_RBD_EE_POSE_GRADIENT_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER>(
#endif
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>(g_ctx, batch), g_streams);

    if (int rc = grid_rbd_sync_consume()) return rc;

    std::memcpy(dee_out, g_data->h_end_effector_pose_gradient, (size_t)batch * 6*GRID_RBD_NUM_EES*grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)dee_out; (void)batch;
    return 3;  // end_effector_pose_gradient not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_end_effector_pose_hessian(long long ctx_id, const T* q, T* d2ee_out, int batch) {
#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_EE_POSE_HESSIAN)
    grid::GRID_RBD_EE_POSE_HESSIAN_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER>(
#else
    grid::GRID_RBD_EE_POSE_HESSIAN_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER>(
#endif
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>(g_ctx, batch), g_streams);

    if (int rc = grid_rbd_sync_consume()) return rc;

    std::memcpy(d2ee_out, g_data->h_end_effector_pose_hessian, (size_t)batch * 6*GRID_RBD_NUM_EES*grid::NUM_VEL*grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)d2ee_out; (void)batch;
    return 3;  // end_effector_pose_hessian not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_idsva_so(long long ctx_id, const T* q, const T* qd, const T* qdd, T* out, int batch, T gravity) {
#if GRID_HAS_IDSVA_SO
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    // idsva_so reads the joint acceleration from the u-slot of d_q_qd_u (s_qdd);
    // pack qdd there so the second-order tensors use the requested acceleration.
    pack_q_qd_u(g_ctx, q, qd, qdd, batch, grid::NUM_JOINTS);

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
// RESOURCE_TIER must match the tier the autotuned thread count was picked for —
// the default-tier instantiation with a LITE-tuned count exceeded the default
// kernel's register-limited thread cap and failed the launch (invalid argument).
#if defined(GRID_RBD_SIG_MJX_IDSVA_SO)
    grid::idsva_so<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO>::TIER>(
#else
    grid::idsva_so<T, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO>::TIER>(
#endif
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_IDSVA_SO>(g_ctx, batch), g_streams);

    if (int rc = grid_rbd_sync_consume()) return rc;

    std::memcpy(out, g_data->h_idsva_so, (size_t)batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)qdd; (void)out; (void)batch; (void)gravity;
    return 3;  // idsva_so not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_fdsva_so(long long ctx_id, const T* q, const T* qd, const T* u, T* out, int batch, T gravity) {
#if GRID_HAS_FDSVA_SO
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, u, batch, grid::NUM_JOINTS);

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
// RESOURCE_TIER must match the autotuned tier (see idsva_so note).
#if defined(GRID_RBD_SIG_MJX_FDSVA_SO)
    grid::fdsva_so<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER>(
#else
    grid::fdsva_so<T, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER>(
#endif
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_FDSVA_SO>(g_ctx, batch), g_streams);

    if (int rc = grid_rbd_sync_consume()) return rc;

    std::memcpy(out, g_data->h_df2, (size_t)batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)u; (void)out; (void)batch; (void)gravity;
    return 3;  // fdsva_so not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_forward_dynamics(long long ctx_id, const T* q, const T* qd, const T* u, T* qdd_out, int batch, T gravity, const T* f_ext) {
#if GRID_HAS_FORWARD_DYNAMICS
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, u, batch, grid::NUM_JOINTS);
    if (int rc = apply_f_ext(g_ctx, f_ext, batch)) return rc;

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_FORWARD_DYNAMICS)
    grid::forward_dynamics<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER>(
#else
    grid::forward_dynamics<T, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER>(
#endif
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_FORWARD_DYNAMICS>(g_ctx, batch), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(g_ctx, f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(qdd_out, g_data->h_qdd, (size_t)batch * grid::NUM_JOINTS * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)u; (void)qdd_out; (void)batch; (void)gravity; (void)f_ext;
    return 3;  // forward_dynamics not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_aba(long long ctx_id, const T* q, const T* qd, const T* u, T* qdd_out, int batch, T gravity, const T* f_ext) {
#if GRID_HAS_ABA
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, u, batch, grid::NUM_JOINTS);
    if (int rc = apply_f_ext(g_ctx, f_ext, batch)) return rc;

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_ABA)
    grid::aba<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER>(
#else
    grid::aba<T, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER>(
#endif
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_ABA>(g_ctx, batch), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(g_ctx, f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(qdd_out, g_data->h_qdd, (size_t)batch * grid::NUM_JOINTS * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)u; (void)qdd_out; (void)batch; (void)gravity; (void)f_ext;
    return 3;  // aba not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_inverse_dynamics(long long ctx_id, const T* q, const T* qd, const T* qdd_opt, T* c_out, int batch, T gravity, const T* f_ext) {
#if GRID_HAS_INVERSE_DYNAMICS
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    if (int rc = apply_f_ext(g_ctx, f_ext, batch)) return rc;

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
    if (qdd_opt) {
        // Host wrapper copies h_qdd->d_qdd (NUM_JOINTS per timestep, contiguous).
        std::memcpy(g_data->h_qdd, qdd_opt, (size_t)batch * grid::NUM_JOINTS * sizeof(T));
#if defined(GRID_RBD_SIG_MJX_INVERSE_DYNAMICS)
        grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER>(
#else
        grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER>(
#endif
            g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS>(g_ctx, batch), g_streams);
    } else {
#if defined(GRID_RBD_SIG_MJX_INVERSE_DYNAMICS)
        grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/false, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER>(
#else
        grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/false, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER>(
#endif
            g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS>(g_ctx, batch), g_streams);
    }

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(g_ctx, f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(c_out, g_data->h_c, (size_t)batch * grid::NUM_JOINTS * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)qdd_opt; (void)c_out; (void)batch; (void)gravity; (void)f_ext;
    return 3;  // inverse_dynamics not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_inverse_dynamics_gradient(long long ctx_id, const T* q, const T* qd, const T* qdd_opt, T* dc_du_out, int batch, T gravity, const T* f_ext) {
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    if (int rc = apply_f_ext(g_ctx, f_ext, batch)) return rc;

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
    if (qdd_opt) {
        // Host wrapper copies h_qdd->d_qdd (NUM_JOINTS per timestep, contiguous).
        std::memcpy(g_data->h_qdd, qdd_opt, (size_t)batch * grid::NUM_JOINTS * sizeof(T));
#if defined(GRID_RBD_SIG_MJX_INVERSE_DYNAMICS_GRADIENT)
        grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(
#else
        grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(
#endif
            g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(g_ctx, batch), g_streams);
    } else {
#if defined(GRID_RBD_SIG_MJX_INVERSE_DYNAMICS_GRADIENT)
        grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/false, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(
#else
        grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/false, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(
#endif
            g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(g_ctx, batch), g_streams);
    }

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(g_ctx, f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    cudaMemcpy(dc_du_out, g_data->d_dc_du, (size_t)batch * 2*grid::NUM_VEL*grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
#else
    (void)q; (void)qd; (void)qdd_opt; (void)dc_du_out; (void)batch; (void)gravity; (void)f_ext;
    return 3;  // inverse_dynamics_gradient not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_forward_dynamics_gradient(long long ctx_id, const T* q, const T* qd, const T* u, T* df_du_out, int batch, T gravity, const T* f_ext) {
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, u, batch, grid::NUM_JOINTS);
    if (int rc = apply_f_ext(g_ctx, f_ext, batch)) return rc;

// signature switch: the host template carries MUJOCO_OUTPUT on floating
// builds regardless of enable_mujoco_kernels — keyed on the per-fn
// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header
// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).
#if defined(GRID_RBD_SIG_MJX_FORWARD_DYNAMICS_GRADIENT)
    grid::forward_dynamics_gradient<T, /*USE_QDD_MINV_FLAG=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/false, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER>(
#else
    grid::forward_dynamics_gradient<T, /*USE_QDD_MINV_FLAG=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER>(
#endif
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>(g_ctx, batch), g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(g_ctx, f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    cudaMemcpy(df_du_out, g_data->d_df_du, (size_t)batch * 2*grid::NUM_VEL*grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
#else
    (void)q; (void)qd; (void)u; (void)df_du_out; (void)batch; (void)gravity; (void)f_ext;
    return 3;  // forward_dynamics_gradient not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_integrator(long long ctx_id, const T* q, const T* qd, const T* u, T* x_kp1_out, int batch, T gravity, T dt, int it) {
#if GRID_HAS_INTEGRATOR
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, u, batch, grid::NUM_JOINTS);
    GRID_RBD_IT_DISPATCH(it, launch_integrator_host, batch, gravity, dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(x_kp1_out, g_data->h_x_kp1, (size_t)batch * (grid::NUM_POS + grid::NUM_VEL) * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)u; (void)x_kp1_out; (void)batch; (void)gravity; (void)dt; (void)it;
    return 3;  // integrator not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_integrator_gradient(long long ctx_id, const T* q, const T* qd, const T* u, T* dAB_out, int batch, T gravity, T dt, int it) {
#if GRID_HAS_INTEGRATOR_GRADIENT
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, u, batch, grid::NUM_JOINTS);
    GRID_RBD_IT_DISPATCH(it, launch_integrator_grad_host, batch, gravity, dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(dAB_out, g_data->h_dAB, (size_t)batch * (2 * grid::NUM_VEL) * (3 * grid::NUM_VEL) * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)u; (void)dAB_out; (void)batch; (void)gravity; (void)dt; (void)it;
    return 3;  // integrator_gradient not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_inverse_dynamics_regressor(long long ctx_id, const T* q, const T* qd, const T* qdd, T* out, int batch, T gravity) {
#if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, qdd, batch, grid::NUM_JOINTS);
    grid::inverse_dynamics_regressor<T>(g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS_REGRESSOR>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_Y, (size_t)batch * grid::NUM_VEL * 10 * grid::NUM_BODIES * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)qdd; (void)out; (void)batch; (void)gravity;
    return 3;  // inverse_dynamics_regressor not built into this .so (subset profile)
#endif
}

extern "C" int grid_rbd_end_effector_pose_runtime(long long ctx_id, const T* q, T* out, int batch, int target_jid, const T* offset) {
#ifdef GRID_HAS_END_EFFECTOR_POSE_RUNTIME
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (target_jid < 0 || target_jid >= grid::NUM_JOINTS) return 1;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    // stage the runtime offset (frame origin when offset==nullptr):
    // offset is the 4x4 col-major SE(3) tool/tip transform (16 floats); identity => frame origin.
    T Xtool[16] = {static_cast<T>(1),0,0,0, 0,static_cast<T>(1),0,0,
                   0,0,static_cast<T>(1),0, 0,0,0,static_cast<T>(1)};
    if (offset) { for (int i = 0; i < 16; ++i) Xtool[i] = offset[i]; }
    if (cudaMemcpy(g_data->d_eepose_runtime_offset, Xtool, 16*sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 101;
    grid::end_effector_pose_runtime<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), g_streams, target_jid);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_eePose, (size_t)batch * 6 * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)target_jid; (void)offset;
    return 3;  // end_effector_pose_runtime not built
#endif
}

extern "C" int grid_rbd_end_effector_pose_gradient_runtime(long long ctx_id, const T* q, T* out, int batch, int target_jid, const T* offset) {
#ifdef GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (target_jid < 0 || target_jid >= grid::NUM_JOINTS) return 1;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    // stage the runtime offset (frame origin when offset==nullptr):
    // offset is the 4x4 col-major SE(3) tool/tip transform (16 floats); identity => frame origin.
    T Xtool[16] = {static_cast<T>(1),0,0,0, 0,static_cast<T>(1),0,0,
                   0,0,static_cast<T>(1),0, 0,0,0,static_cast<T>(1)};
    if (offset) { for (int i = 0; i < 16; ++i) Xtool[i] = offset[i]; }
    if (cudaMemcpy(g_data->d_eepose_runtime_offset, Xtool, 16*sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 101;
    grid::end_effector_pose_gradient_runtime<T>(g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), g_streams, target_jid);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_eePoseGrad, (size_t)batch * 6*grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)target_jid; (void)offset;
    return 3;  // end_effector_pose_gradient_runtime not built
#endif
}

// ── END GENERATED C-ABI BODIES ──

// ── BEGIN GENERATED MJX TWIN BODIES (grid_codegen/wrapper_body_gen.py — do not hand-edit) ──
// Regenerate: .venv/bin/python -m grid_codegen.wrapper_body_gen
// Table: grid_codegen/abi_specs.py (mjx_* fields); docs verbatim from
// grid_codegen/wrapper_mjx_docs.py. The 4 plant cost twins stay hand-written.

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_NONLINEAR_EFFECTS
// MuJoCo-convention nonlinear_effects(q, qd) -> c(q,qd) (floating base only). q is
// raw mjx (kernel reorders the quaternion). The kernel (MUJOCO_OUTPUT=true) injects
// the accel-couple delta_a (base-linear = -(omega x v_lin)) via a zeroed s_qdd then
// base-rotates the bias output, so the returned c is the mjx-frame qfrc_bias.
extern "C" int grid_rbd_nonlinear_effects_mujoco(const T* q, const T* qd, T* out, int batch, T gravity) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::nonlinear_effects<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_NONLINEAR_EFFECTS>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_c, (size_t)batch * grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_NONLINEAR_EFFECTS

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_GENERALIZED_GRAVITY
// MuJoCo-convention generalized_gravity(q) -> g(q) (floating base only). q is raw
// mjx (kernel reorders the quaternion); the kernel (MUJOCO_OUTPUT=true) base-rotates
// the gravity output so the returned g is mjx-frame. Output is NUM_VEL invariant-shaped.
extern "C" int grid_rbd_generalized_gravity_mujoco(const T* q, T* out, int batch, T gravity) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    grid::generalized_gravity<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_GENERALIZED_GRAVITY>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_c, (size_t)batch * grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_GENERALIZED_GRAVITY

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_CORIOLIS_MATRIX
// MuJoCo-convention Coriolis matrix (floating base only): C_mjx = G C_pin G^T, a
// congruence baked into the kernel (MUJOCO_OUTPUT=true). q/qd raw mjx in (kernel
// reorders the quaternion + reframes qd), mjx-frame C out (nv x nv row-major).
extern "C" int grid_rbd_coriolis_matrix_mujoco(const T* q, const T* qd, T* out, int batch, T gravity) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::coriolis_matrix<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_CORIOLIS_MATRIX>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_coriolis, (size_t)batch * grid::NUM_VEL*grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_CORIOLIS_MATRIX

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_ENERGY
// MuJoCo-convention energy(q, qd) -> [KE, PE, KE+PE] per timestep. The energies are
// frame-INVARIANT; the kernel (MUJOCO_OUTPUT=true) just converts the mjx-native
// inputs (quaternion reorder + qd reframe) so the energy is built correctly. Output
// is byte-equal to feeding the pin kernel the pin-converted q/qd.
extern "C" int grid_rbd_energy_mujoco(const T* q, const T* qd, T* out, int batch, T gravity) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::energy<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_ENERGY>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_ENERGY>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_energy, (size_t)batch * 3 * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_ENERGY

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_COM
// MuJoCo-convention com(q) -> [p_com(3); J_com(3 x NV)] per timestep. p_com is
// INVARIANT; the J_com columns are reframed by the kernel (MUJOCO_OUTPUT=true): q
// is MuJoCo-native (quat wxyz) and the kernel reorders the quaternion + applies the
// column reframe before saving, so NO host pre/post-process is needed.
extern "C" int grid_rbd_com_mujoco(const T* q, T* out, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q(g_ctx, q, batch, grid::NUM_JOINTS);
    grid::com<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_COM>::TIER>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_COM>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_com, (size_t)batch * (3 + 3 * grid::NUM_VEL) * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_COM

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_CCRBA
// MuJoCo-convention ccrba(q, qd) -> [A(6 x NV); h(6)] per timestep. The centroidal
// momentum h is INVARIANT; the A columns are reframed by the kernel
// (MUJOCO_OUTPUT=true). q/qd raw mjx in (kernel reorders the quaternion + reframes
// qd), so NO host pre/post-process is needed.
extern "C" int grid_rbd_ccrba_mujoco(const T* q, const T* qd, T* out, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::ccrba<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_CCRBA>::TIER>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_CCRBA>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_ccrba, (size_t)batch * (6 * grid::NUM_VEL + 6) * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_CCRBA

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_DCCRBA
// MuJoCo-convention dccrba(q) -> 6*NV*NV dA/dq tensor (floating base only). q is raw
// mjx (kernel reorders the quaternion); the kernel (MUJOCO_OUTPUT=true) double-reframes
// the qd-column and q-tangent indices by G^{-1} and adds the base-rotation frame term
// (using the in-kernel CMM value) before saving.
extern "C" int grid_rbd_dccrba_mujoco(const T* q, T* out, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q(g_ctx, q, batch, grid::NUM_JOINTS);
    grid::dccrba<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_DCCRBA>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_dccrba, (size_t)batch * 6*grid::NUM_VEL*grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_DCCRBA

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_CMM_TIME_VARIATION
// MuJoCo-convention cmm_time_variation(q, qd) -> 6*NUM_VEL Adot (per timestep).
// Column-reframe: q/qd raw mjx in (kernel reorders the quaternion + reframes qd)
// and the kernel (MUJOCO_OUTPUT=true) reframes the Adot columns before saving.
extern "C" int grid_rbd_cmm_time_variation_mujoco(const T* q, const T* qd, T* out, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::cmm_time_variation<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_CMM_TIME_VARIATION>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_cmm_time_variation, (size_t)batch * 6*grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_CMM_TIME_VARIATION

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_KINETIC_ENERGY_REGRESSOR
// MuJoCo-convention kinetic_energy_regressor(q, qd) -> length 10*NUM_BODIES y_KE.
// The regressor is frame-INVARIANT; the kernel (MUJOCO_OUTPUT=true) only converts
// the mjx-native inputs (quaternion reorder + qd reframe). Output is byte-equal to
// feeding the pin kernel the pin-converted q/qd. (The MUJOCO_OUTPUT instantiation
// only exists for floating, hence the GRID_RBD_WITH_MUJOCO gate.)
extern "C" int grid_rbd_kinetic_energy_regressor_mujoco(const T* q, const T* qd, T* out, int batch, T gravity) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::kinetic_energy_regressor<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_KINETIC_ENERGY_REGRESSOR>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_ke_regressor, (size_t)batch * 10*grid::NUM_BODIES * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_KINETIC_ENERGY_REGRESSOR

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
// MuJoCo-convention potential_energy_regressor(q) -> length 10*NUM_BODIES y_PE.
// Frame-INVARIANT; the kernel (MUJOCO_OUTPUT=true) only converts the mjx-native q
// (quaternion reorder). Output byte-equal to feeding the pin kernel pin-converted q.
extern "C" int grid_rbd_potential_energy_regressor_mujoco(const T* q, T* out, int batch, T gravity) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q(g_ctx, q, batch, grid::NUM_JOINTS);
    grid::potential_energy_regressor<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_POTENTIAL_ENERGY_REGRESSOR>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_pe_regressor, (size_t)batch * 10*grid::NUM_BODIES * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_POTENTIAL_ENERGY_REGRESSOR

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_FRAME_JACOBIAN
// MuJoCo-convention frame_jacobian family (floating base only). The geometric
// Jacobian / its time-derivative are column-reframed J_mjx = J_pin G^{-1} in the
// kernel (MUJOCO_OUTPUT=true). osc_inertia's value Lambda is frame-INVARIANT, but
// its q input still needs the quaternion reordered (wxyz->xyzw) so the internal
// J/Minv build correctly — the mjx kernel does that, so a raw-mjx q is handled here
// rather than silently mis-built by the pin kernel. All take raw mjx inputs.
extern "C" int grid_rbd_frame_jacobian_mujoco(const T* q, T* out, int batch, int target_jid, int reference_frame) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (target_jid < 0 || target_jid >= grid::NUM_JOINTS) return 1;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    grid::frame_jacobian<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_FRAME_JACOBIAN>(g_ctx, batch), g_streams, target_jid, reference_frame);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_frame_jacobian, (size_t)batch * 6*grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_FRAME_JACOBIAN

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_FRAME_JACOBIAN_DOT
extern "C" int grid_rbd_frame_jacobian_dot_mujoco(const T* q, const T* qd, T* out, int batch, int target_jid, int reference_frame) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (target_jid < 0 || target_jid >= grid::NUM_JOINTS) return 1;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::frame_jacobian_dot<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_FRAME_JACOBIAN_DOT>(g_ctx, batch), g_streams, target_jid, reference_frame);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_frame_jacobian_dot, (size_t)batch * 6*grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_FRAME_JACOBIAN_DOT

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_OSC_INERTIA
extern "C" int grid_rbd_osc_inertia_mujoco(const T* q, T* out, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    grid::osc_inertia<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_OSC_INERTIA>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_osc_inertia, (size_t)batch * 36 * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_OSC_INERTIA

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_MINV
// MuJoCo-convention direct mass-matrix inverse (floating base only): the kernel
// reorders the quaternion and applies the congruence Minv_mjx = G^-T Minv_pin G^-1
// on the base block (MUJOCO_OUTPUT=true). The native kernel writes a FULL DENSE
// SYMMETRIC mjx Minv (both triangles), so NO host symmetrize and NO host
// minv_pin_to_mjx post-process are needed.
extern "C" int grid_rbd_minv_mujoco(const T* q, T* minv_out, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    grid::minv<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_MINV>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    cudaMemcpy(minv_out, g_data->d_Minv, (size_t)batch * grid::NUM_VEL*grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_MINV

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_CRBA
// MuJoCo-convention mass matrix (floating base only): M_mjx = G M_pin G^T, the
// congruence baked into the kernel (MUJOCO_OUTPUT=true). q is MuJoCo-native (quat
// wxyz); the kernel reorders the quaternion and applies the congruence on the base
// block before saving, so NO host pre/post-process is needed.
extern "C" int grid_rbd_crba_mujoco(const T* q, T* m_out, int batch, T gravity) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    grid::crba<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_CRBA>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    cudaMemcpy(m_out, g_data->d_M, (size_t)batch * grid::NUM_VEL*grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_CRBA

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_END_EFFECTOR_POSE
// MuJoCo-convention end_effector_pose(q) -> 6*NUM_EES per timestep. The pose is
// frame-INVARIANT; the kernel (MUJOCO_OUTPUT=true) only converts the mjx-native q
// (quaternion reorder, like osc_inertia). Output byte-equal to feeding the pin
// kernel the pin-converted q.
extern "C" int grid_rbd_end_effector_pose_mujoco(const T* q, T* ee_out, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    grid::GRID_RBD_EE_POSE_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE>::TIER>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_END_EFFECTOR_POSE>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(ee_out, g_data->h_end_effector_pose, (size_t)batch * 6*GRID_RBD_NUM_EES * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_END_EFFECTOR_POSE

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_END_EFFECTOR_POSE_GRADIENT
// MuJoCo-convention end_effector_pose Jacobian (q) -> 6*NUM_EES*NUM_VEL per
// timestep. Column-reframe: q raw mjx in (kernel reorders the quaternion) and the
// kernel (MUJOCO_OUTPUT=true) reframes the base-linear Jacobian columns before
// saving (the column reframe acts on the NV axis cols 0:3).
extern "C" int grid_rbd_end_effector_pose_gradient_mujoco(const T* q, T* dee_out, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    grid::GRID_RBD_EE_POSE_GRADIENT_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(dee_out, g_data->h_end_effector_pose_gradient, (size_t)batch * 6*GRID_RBD_NUM_EES*grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_END_EFFECTOR_POSE_GRADIENT

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_END_EFFECTOR_POSE_HESSIAN
// MuJoCo-convention end_effector_pose Hessian (q) -> 6*NUM_EES*NV*NV per timestep.
// q is raw mjx (kernel reorders the quaternion); the kernel (MUJOCO_OUTPUT=true)
// double-column-reframes the Hessian (J·G^{-1} on both tangent indices) and adds the
// symmetrized base-rotation frame term before saving. Output is invariant-shaped.
extern "C" int grid_rbd_end_effector_pose_hessian_mujoco(const T* q, T* d2ee_out, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    grid::GRID_RBD_EE_POSE_HESSIAN_FN<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>(g_ctx, batch), g_streams);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(d2ee_out, g_data->h_end_effector_pose_hessian, (size_t)batch * 6*GRID_RBD_NUM_EES*grid::NUM_VEL*grid::NUM_VEL * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_END_EFFECTOR_POSE_HESSIAN

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_IDSVA_SO
// MuJoCo-convention idsva_so(q, qd, qdd) -> 4*NV^3 (floating base only). q/qd/qdd are
// raw mjx (kernel input-converts); the kernel (MUJOCO_OUTPUT=true) transforms all four
// 2nd-order tensors to the mjx frame (explicit-analytic SO transform + dM_dq closed
// form), reusing the id/crba/id-grad inners. The mjx kernel is register-heavy; the
// post-launch error check surfaces a silent launch-config failure as rc!=0.
extern "C" int grid_rbd_idsva_so_mujoco(const T* q, const T* qd, const T* qdd, T* out, int batch, T gravity) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, qdd, batch, grid::NUM_JOINTS);
    grid::idsva_so<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_IDSVA_SO>(g_ctx, batch), g_streams);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_idsva_so, (size_t)batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_IDSVA_SO

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_FDSVA_SO
// MuJoCo-convention fdsva_so(q, qd, u) -> 4*NV^3 (floating base only). q/qd/u raw mjx
// (kernel input-converts); the kernel (MUJOCO_OUTPUT=true) transforms all four
// 2nd-order forward-dynamics tensors to the mjx frame (explicit-analytic SO transform,
// contravector output-map). Register-heavy; post-launch error check surfaces a silent
// launch-config failure as rc!=0.
extern "C" int grid_rbd_fdsva_so_mujoco(const T* q, const T* qd, const T* u, T* out, int batch, T gravity) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, u, batch, grid::NUM_JOINTS);
    grid::fdsva_so<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_FDSVA_SO>(g_ctx, batch), g_streams);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_df2, (size_t)batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_FDSVA_SO

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_FORWARD_DYNAMICS
// MuJoCo-convention forward dynamics (floating base only). q/qd/u are MuJoCo-native;
// the kernel converts inputs mjx->pin on load and maps the output acceleration
// qdd[0:3] = R(qdd_pin + omega x v) back to the mjx frame (MUJOCO_OUTPUT=true) — no
// host pre/post-process. f_ext is not reframed by the kernel input-convert, so the
// _handle dispatch only takes this path when f_ext is null.
extern "C" int grid_rbd_forward_dynamics_mujoco(const T* q, const T* qd, const T* u, T* qdd_out, int batch, T gravity, const T* f_ext) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, u, batch, grid::NUM_JOINTS);
    if (int rc = apply_f_ext(g_ctx, f_ext, batch)) return rc;
    grid::forward_dynamics<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_FORWARD_DYNAMICS>(g_ctx, batch), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(g_ctx, f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(qdd_out, g_data->h_qdd, (size_t)batch * grid::NUM_JOINTS * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_FORWARD_DYNAMICS

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_ABA
// MuJoCo-convention ABA (floating base only). Same accel_out convention as
// forward_dynamics: q/qd/u raw mjx in, mjx-frame qdd out (MUJOCO_OUTPUT=true). f_ext
// not reframed -> the _handle dispatch only uses this path when f_ext is null.
extern "C" int grid_rbd_aba_mujoco(const T* q, const T* qd, const T* u, T* qdd_out, int batch, T gravity, const T* f_ext) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, u, batch, grid::NUM_JOINTS);
    if (int rc = apply_f_ext(g_ctx, f_ext, batch)) return rc;
    grid::aba<T, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_ABA>(g_ctx, batch), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(g_ctx, f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(qdd_out, g_data->h_qdd, (size_t)batch * grid::NUM_JOINTS * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_ABA

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
extern "C" int grid_rbd_inverse_dynamics_mujoco(const T* q, const T* qd, const T* qdd_opt, T* c_out, int batch, T gravity, const T* f_ext) {
    if (!qdd_opt) return 4;  // mjx requires an explicit qdd
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    if (int rc = apply_f_ext(g_ctx, f_ext, batch)) return rc;
    // Host wrapper copies h_qdd->d_qdd (NUM_JOINTS per timestep, contiguous).
    std::memcpy(g_data->h_qdd, qdd_opt, (size_t)batch * grid::NUM_JOINTS * sizeof(T));
    grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS>(g_ctx, batch), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(g_ctx, f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(c_out, g_data->h_c, (size_t)batch * grid::NUM_JOINTS * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_INVERSE_DYNAMICS

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_INVERSE_DYNAMICS_GRADIENT
// MuJoCo-convention inverse-dynamics gradient (floating base only). q/qd/qdd are
// MuJoCo-native; the kernel converts inputs mjx->pin on load and applies the full
// gradient convention transform (reframe + base-row rotate + ω×v couplings, with M
// from an in-kernel crba reuse) so the returned dc/d(q,qd) is the mjx-frame gradient.
// REQUIRES qdd (the mjx gradient is the with-qdd surface; a null qdd returns rc=4).
extern "C" int grid_rbd_inverse_dynamics_gradient_mujoco(const T* q, const T* qd, const T* qdd_opt, T* dc_du_out, int batch, T gravity, const T* f_ext) {
    if (!qdd_opt) return 4;  // mjx requires an explicit qdd
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);
    if (int rc = apply_f_ext(g_ctx, f_ext, batch)) return rc;
    // Host wrapper copies h_qdd->d_qdd (NUM_JOINTS per timestep, contiguous).
    std::memcpy(g_data->h_qdd, qdd_opt, (size_t)batch * grid::NUM_JOINTS * sizeof(T));
    grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(g_ctx, batch), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(g_ctx, f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(dc_du_out, g_data->d_dc_du, (size_t)batch * 2*grid::NUM_VEL*grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_INVERSE_DYNAMICS_GRADIENT

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_FORWARD_DYNAMICS_GRADIENT
// MuJoCo-convention forward-dynamics gradient (floating base only). q/qd/u are
// MuJoCo-native; qdd is computed internally. The kernel converts inputs mjx->pin on
// load and applies the full gradient convention transform (reframe + base-row rotate
// + ω×v couplings) so the returned df/d(q,qd) is the mjx-frame gradient.
extern "C" int grid_rbd_forward_dynamics_gradient_mujoco(const T* q, const T* qd, const T* u, T* df_du_out, int batch, T gravity, const T* f_ext) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, u, batch, grid::NUM_JOINTS);
    if (int rc = apply_f_ext(g_ctx, f_ext, batch)) return rc;
    grid::forward_dynamics_gradient<T, /*USE_QDD_MINV_FLAG=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true, /*RESOURCE_TIER=*/grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>(g_ctx, batch), g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is
    // recorded in the sticky slot (the stream stays empty, so the sync above
    // returns success — the silent stale-buffer class). Consume it here so the
    // caller gets a loud rc instead of plausible garbage.
    if (e == cudaSuccess) e = grid_consume_last_error();
    reset_f_ext(g_ctx, f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(df_du_out, g_data->d_df_du, (size_t)batch * 2*grid::NUM_VEL*grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_FORWARD_DYNAMICS_GRADIENT

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_INTEGRATOR
// MuJoCo-convention integrator (floating base only): the free-joint base POSITION
// takes a GLOBAL additive step (mjx retract) instead of pin's SE(3) V(phi); the
// base quaternion + joints integrate normally. q/qd raw mjx in (q wxyz, qd global),
// x_kp1 raw mjx out (q wxyz). Baked into the kernel (MUJOCO_OUTPUT=true).
extern "C" int grid_rbd_integrator_mujoco(const T* q, const T* qd, const T* u, T* x_kp1_out, int batch, T gravity, T dt, int it) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, u, batch, grid::NUM_JOINTS);
    GRID_RBD_IT_DISPATCH(it, launch_integrator_host_mujoco, batch, gravity, dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(x_kp1_out, g_data->h_x_kp1, (size_t)batch * (grid::NUM_POS + grid::NUM_VEL) * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_INTEGRATOR

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_INTEGRATOR_GRADIENT
// MuJoCo-convention integrator_gradient(q, qd, u, dt, it) -> dAB (2NV x 3NV) (floating
// base only). q/qd/u raw mjx (kernel input-converts); the kernel (MUJOCO_OUTPUT=true)
// transforms the discrete state-transition Jacobian to the mjx tangent (global-add
// retract rows + G velocity reframe + input-conversion column couplings). EULER/SI-EULER
// only (multistage static_asserts out). Register-heavy; post-launch error check.
extern "C" int grid_rbd_integrator_gradient_mujoco(const T* q, const T* qd, const T* u, T* dAB_out, int batch, T gravity, T dt, int it) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, u, batch, grid::NUM_JOINTS);
    GRID_RBD_IT_DISPATCH_HESSIAN(it, launch_integrator_grad_host_mujoco, batch, gravity, dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(dAB_out, g_data->h_dAB, (size_t)batch * (2 * grid::NUM_VEL) * (3 * grid::NUM_VEL) * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_INTEGRATOR_GRADIENT

#if defined(GRID_RBD_WITH_MUJOCO) && GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
// MuJoCo-convention inverse_dynamics_regressor (floating base only). q/qd/qdd are raw
// mjx (kernel input-converts); the regressor ROWS are tangent-indexed generalized
// forces, so the base-LINEAR rows (0:3) rotate by R (Y_mjx[0:3] = R Y_pin[0:3]) -- the
// same base-row rotate as id_tau. Baked via the MUJOCO_OUTPUT=true template flag.
extern "C" int grid_rbd_inverse_dynamics_regressor_mujoco(const T* q, const T* qd, const T* qdd, T* out, int batch, T gravity) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(g_ctx, q, qd, qdd, batch, grid::NUM_JOINTS);
    grid::inverse_dynamics_regressor<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, gravity, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS_REGRESSOR>(g_ctx, batch), g_streams);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_Y, (size_t)batch * grid::NUM_VEL * 10 * grid::NUM_BODIES * sizeof(T));
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_HAS_INVERSE_DYNAMICS_REGRESSOR

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention end_effector_pose_runtime (floating base only). The kernel
// input-converts q (quat reorder) on load; the pose VALUE is frame-INVARIANT (the
// 6-vector [xyz; rpy] is the same world frame), so this matches the pin pose with
// the mjx-reordered quaternion. Baked via the MUJOCO_OUTPUT=true host/kernel flag.
extern "C" int grid_rbd_end_effector_pose_runtime_mujoco(const T* q, T* out, int batch, int target_jid, const T* offset) {
#ifdef GRID_HAS_END_EFFECTOR_POSE_RUNTIME
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (target_jid < 0 || target_jid >= grid::NUM_JOINTS) return 1;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    // stage the runtime offset (frame origin when offset==nullptr):
    // offset is the 4x4 col-major SE(3) tool/tip transform (16 floats); identity => frame origin.
    T Xtool[16] = {static_cast<T>(1),0,0,0, 0,static_cast<T>(1),0,0,
                   0,0,static_cast<T>(1),0, 0,0,0,static_cast<T>(1)};
    if (offset) { for (int i = 0; i < 16; ++i) Xtool[i] = offset[i]; }
    if (cudaMemcpy(g_data->d_eepose_runtime_offset, Xtool, 16*sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 101;
    grid::end_effector_pose_runtime<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), g_streams, target_jid);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_eePose, (size_t)batch * 6 * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)target_jid; (void)offset;
    return 3;
#endif
}
#endif  // GRID_RBD_WITH_MUJOCO

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention end_effector_pose_gradient_runtime (floating base only). The
// pose value is invariant, but the gradient is COLUMN-reframed: the base-linear
// columns reframe by R^T (mjx base-linear velocity is global). Baked via
// MUJOCO_OUTPUT=true. Output 6 x NUM_VEL col-major, like the pin variant.
extern "C" int grid_rbd_end_effector_pose_gradient_runtime_mujoco(const T* q, T* out, int batch, int target_jid, const T* offset) {
#ifdef GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (target_jid < 0 || target_jid >= grid::NUM_JOINTS) return 1;
    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);
    // stage the runtime offset (frame origin when offset==nullptr):
    // offset is the 4x4 col-major SE(3) tool/tip transform (16 floats); identity => frame origin.
    T Xtool[16] = {static_cast<T>(1),0,0,0, 0,static_cast<T>(1),0,0,
                   0,0,static_cast<T>(1),0, 0,0,0,static_cast<T>(1)};
    if (offset) { for (int i = 0; i < 16; ++i) Xtool[i] = offset[i]; }
    if (cudaMemcpy(g_data->d_eepose_runtime_offset, Xtool, 16*sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 101;
    grid::end_effector_pose_gradient_runtime<T, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL, /*MUJOCO_OUTPUT=*/true>(
        g_data, g_robot, batch, dim3((unsigned)batch, 1, 1), grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), g_streams, target_jid);
    if (int rc = grid_rbd_sync_consume()) return rc;
    std::memcpy(out, g_data->h_eePoseGrad, (size_t)batch * 6*grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)target_jid; (void)offset;
    return 3;
#endif
}
#endif  // GRID_RBD_WITH_MUJOCO

// ── END GENERATED MJX TWIN BODIES ──


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

static int plant_alloc(GridCtx *ctx) {
    GRID_RBD_CTX_LOCALS(ctx);
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
static int plant_quadratic_cost_impl(GridCtx *ctx, 
    const T* var, const T* des, const T* w,
    T* out, T* grad, T* hess, int batch)
{
    GRID_RBD_CTX_LOCALS(ctx);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
    const int N = STATE ? (grid::NUM_POS + grid::NUM_VEL) : grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, var, batch * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, des, batch * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, w,   batch * N * sizeof(T), cudaMemcpyHostToDevice);
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    if (STATE) {
        grid_plant::quadratic_state_cost_kernel<T><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, g_streams[0]>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
        { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    } else {
        grid_plant::quadratic_input_cost_kernel<T><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, g_streams[0]>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
        { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    }
    if (int rc = grid_rbd_sync_consume()) return rc;
    cudaMemcpy(out,  g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad, g_plant.d_grad, batch * N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, g_plant.d_hess, batch * N * N * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}

extern "C" int grid_plant_quadratic_state_cost(long long ctx_id,
    const T* x, const T* x_des, const T* Q, T* out, T* grad, T* hess, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    return plant_quadratic_cost_impl<true>(g_ctx, x, x_des, Q, out, grad, hess, batch);
}
extern "C" int grid_plant_quadratic_input_cost(long long ctx_id,
    const T* u, const T* u_des, const T* R, T* out, T* grad, T* hess, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    return plant_quadratic_cost_impl<false>(g_ctx, u, u_des, R, out, grad, hess, batch);
}

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention quadratic_state_cost (floating base only). x = [q(nq); qd(nv)]
// is mjx-native; the kernel input-converts the qd base-linear block (global->local)
// before differencing against the user (mjx-frame) x_des/Q, then reframes the
// qd-block grad (covector) + hess (congruence). The VALUE is convention-DEPENDENT.
// Baked via the MUJOCO_OUTPUT=true template flag. quadratic_state_cost is launched
// at the requested thread count and may be register-capped below it -> clamp + check.
extern "C" int grid_rbd_quadratic_state_cost_mujoco(long long ctx_id, 
    const T* x, const T* x_des, const T* Q, T* out, T* grad, T* hess, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
    const int N = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, x,     batch * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, x_des, batch * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, Q,     batch * N * sizeof(T), cudaMemcpyHostToDevice);
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::quadratic_state_cost_kernel<T, true>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::quadratic_state_cost_kernel<T, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr, 0, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    if (int rc = grid_rbd_sync_consume()) return rc;
    cudaMemcpy(out,  g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad, g_plant.d_grad, batch * N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, g_plant.d_hess, batch * N * N * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO

// joint_{position,velocity,torque}_barrier: value + grad + hess-diagonal.
// var/lower/upper are (batch, N); out (batch); grad/hess_diag (batch, N).
enum class PlantBarrier { POSITION, VELOCITY, TORQUE };

static int plant_barrier_impl(GridCtx *ctx, 
    PlantBarrier which,
    const T* var, const T* lower, const T* upper, float mu,
    T* out, T* grad, T* hess_diag, int batch)
{
    GRID_RBD_CTX_LOCALS(ctx);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
    const int N = (which == PlantBarrier::POSITION) ? grid::NUM_POS : grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, var,   batch * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, lower, batch * N * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, upper, batch * N * sizeof(T), cudaMemcpyHostToDevice);
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    switch (which) {
        case PlantBarrier::POSITION:
            grid_plant::joint_position_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, g_streams[0]>>>(
                g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
                g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch); break;
        case PlantBarrier::VELOCITY:
            grid_plant::joint_velocity_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, g_streams[0]>>>(
                g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
                g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch); break;
        case PlantBarrier::TORQUE:
            grid_plant::joint_torque_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, g_streams[0]>>>(
                g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
                g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch); break;
    }
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    if (int rc = grid_rbd_sync_consume()) return rc;
    cudaMemcpy(out,       g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad,      g_plant.d_grad, batch * N * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess_diag, g_plant.d_hess, batch * N * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}

extern "C" int grid_plant_joint_position_barrier(long long ctx_id,
    const T* var, const T* lower, const T* upper, float mu,
    T* out, T* grad, T* hess_diag, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    return plant_barrier_impl(g_ctx, PlantBarrier::POSITION, var, lower, upper, mu, out, grad, hess_diag, batch);
}
extern "C" int grid_plant_joint_velocity_barrier(long long ctx_id,
    const T* var, const T* lower, const T* upper, float mu,
    T* out, T* grad, T* hess_diag, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    return plant_barrier_impl(g_ctx, PlantBarrier::VELOCITY, var, lower, upper, mu, out, grad, hess_diag, batch);
}
extern "C" int grid_plant_joint_torque_barrier(long long ctx_id,
    const T* var, const T* lower, const T* upper, float mu,
    T* out, T* grad, T* hess_diag, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    return plant_barrier_impl(g_ctx, PlantBarrier::TORQUE, var, lower, upper, mu, out, grad, hess_diag, batch);
}

#ifdef GRID_PLANT_HAS_STEP
// plant_step: x_{k+1} = integrator(x_k, u_k, dt). x (batch, NX); u (batch, NV);
// out (batch, NX). Gated on GRID_PLANT_HAS_STEP (emitted only when the
// integrator algorithm is generated). Integrator type via the same int code.
template <grid::IntegratorType IT>
static void launch_plant_step(GridCtx *ctx, int batch, T gravity, T dt) {
    GRID_RBD_CTX_LOCALS(ctx);
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    // clamp to the kernel's launch cap (register-heavy; else a silent launch failure
    // leaves the stale d_grad buffer -> looks like a no-op step). Mirrors the mjx twin.
    // smem opt-in too (arena can exceed 48 KB on a big floating robot -> rc=201).
    const size_t smem = grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_kernel<T, IT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_kernel<T, IT>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::plant_step_kernel<T, IT><<<grid_dim, thr,
        smem, g_streams[0]>>>(
            g_plant.d_grad /*reuse as d_x_kp1, size NX*/, g_plant.d_in_a, g_plant.d_in_b,
            nx, grid::NUM_VEL, g_robot, gravity, dt, batch);
}

extern "C" int grid_plant_step(long long ctx_id, 
    const T* x, const T* u, T* x_kp1, int batch, float gravity, float dt, int it) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, x, batch * nx * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, u, batch * nv * sizeof(T), cudaMemcpyHostToDevice);
    GRID_RBD_IT_DISPATCH(it, launch_plant_step, batch, (T)gravity, (T)dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    if (int rc = grid_rbd_sync_consume()) return rc;
    cudaMemcpy(x_kp1, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention plant_step (floating base only). x/u raw mjx; the kernel
// (MUJOCO_OUTPUT=true) input-converts the stacked state + does the global-add retract.
// EULER/SI-EULER only. Register-heavy -> launch-clamp + post-launch error check.
template <grid::IntegratorType IT>
static void launch_plant_step_mujoco(GridCtx *ctx, int batch, T gravity, T dt) {
    GRID_RBD_CTX_LOCALS(ctx);
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    const size_t smem = grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_kernel<T, IT, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_kernel<T, IT, true>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::plant_step_kernel<T, IT, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr,
        smem, g_streams[0]>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, grid::NUM_VEL, g_robot, gravity, dt, batch);
}
extern "C" int grid_plant_step_mujoco(long long ctx_id, 
    const T* x, const T* u, T* x_kp1, int batch, float gravity, float dt, int it) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, x, batch * nx * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, u, batch * grid::NUM_VEL * sizeof(T), cudaMemcpyHostToDevice);
    GRID_RBD_IT_DISPATCH_HESSIAN(it, launch_plant_step_mujoco, batch, (T)gravity, (T)dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    if (int rc = grid_rbd_sync_consume()) return rc;
    cudaMemcpy(x_kp1, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_PLANT_HAS_STEP

#ifdef GRID_PLANT_HAS_EE_COST
// ee_pos_cost: value + grad over x=[q;qd] + GN hess_x. q (batch, NQ);
// p_des (batch, 3); W (batch, 3); out (batch); grad (batch, NX); hess (batch, NX*NX).
// Gated on GRID_PLANT_HAS_EE_COST (ee_pose + ee_pose_gradient generated).
extern "C" int grid_plant_ee_pos_cost(long long ctx_id, 
    const T* q, const T* p_des, const T* W,
    T* out, T* grad, T* hess, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
    const int nq = grid::NUM_POS;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, q,     batch * nq * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, p_des, batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, W,     batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    // The kernel internally calls ee-pose (value), ee-pose-gradient, and uses
    // the hessian-free GN J^T W J. The dynamic smem must cover the largest of
    // the device fns it invokes (pose-gradient dominates pose).
    size_t smem = grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    // clamp to the kernel's launch cap + surface launch errors (else a register-OOR
    // launch is silently rejected -> stale-zero d_grad base block).
    dim3 thr = grid_clamp_threads_for(grid_plant::ee_pos_cost_kernel<T, 0>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::ee_pos_cost_kernel<T, 0><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose, g_plant.d_end_effector_pose_gradient, g_robot, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    if (int rc = grid_rbd_sync_consume()) return rc;
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
extern "C" int grid_plant_com_cost(long long ctx_id, 
    const T* q, const T* p_des, const T* W,
    T* out, T* grad, T* hess, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
    const int nq = grid::NUM_POS;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, q,     batch * nq * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, p_des, batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, W,     batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    size_t smem = grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    // com_cost_kernel can be register-heavy; clamp to its launch cap (else the
    // launch is silently rejected -> stale-zero d_grad). cudaGetLastError surfaces
    // any launch-config/resource rejection as rc=200+e instead of silent zeros.
    dim3 thr = grid_clamp_threads_for(grid_plant::com_cost_kernel<T, false>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::com_cost_kernel<T><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_robot, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    if (int rc = grid_rbd_sync_consume()) return rc;
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
extern "C" int grid_plant_momentum_cost(long long ctx_id, 
    const T* q, const T* qd, const T* h_des, const T* W,
    T* out, T* grad, T* hess, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
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
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    // momentum_cost is register-heavy (~140 regs/thread): clamp to its launch cap.
    dim3 thr = grid_clamp_threads_for(grid_plant::momentum_cost_kernel<T, false>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::momentum_cost_kernel<T><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b,
        g_plant.d_in_c, g_plant.d_in_c + (size_t)batch * 6,
        g_robot, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    if (int rc = grid_rbd_sync_consume()) return rc;
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
extern "C" int grid_rbd_ee_pos_cost_mujoco(long long ctx_id, 
    const T* q, const T* p_des, const T* W,
    T* out, T* grad, T* hess, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
    const int nq = grid::NUM_POS;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, q,     batch * nq * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, p_des, batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, W,     batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    size_t smem = grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::ee_pos_cost_kernel<T, 0, true>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::ee_pos_cost_kernel<T, /*EE=*/0, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose, g_plant.d_end_effector_pose_gradient, g_robot, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    if (int rc = grid_rbd_sync_consume()) return rc;
    cudaMemcpy(out,  g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, g_plant.d_hess, batch * nx * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_RBD_WITH_MUJOCO && GRID_PLANT_HAS_EE_COST

#if defined(GRID_RBD_WITH_MUJOCO) && defined(GRID_PLANT_HAS_COM_COST)
// MuJoCo-convention com_cost (floating base only). Same transform as ee_pos_cost
// (q-block grad covector + GN hess congruence; q input-converted in-kernel).
extern "C" int grid_rbd_com_cost_mujoco(long long ctx_id, 
    const T* q, const T* p_des, const T* W,
    T* out, T* grad, T* hess, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
    const int nq = grid::NUM_POS;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, q,     batch * nq * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, p_des, batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c, W,     batch * 3  * sizeof(T), cudaMemcpyHostToDevice);
    size_t smem = grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    // clamp to the kernel's launch cap + surface launch errors (mirror the pin path).
    dim3 thr = grid_clamp_threads_for(grid_plant::com_cost_kernel<T, true>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::com_cost_kernel<T, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_robot, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    if (int rc = grid_rbd_sync_consume()) return rc;
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
extern "C" int grid_rbd_momentum_cost_mujoco(long long ctx_id, 
    const T* q, const T* qd, const T* h_des, const T* W,
    T* out, T* grad, T* hess, int batch) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
    const int nq = grid::NUM_POS;
    const int nv = grid::NUM_VEL;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, q,     batch * nq * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, qd,    batch * nv * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c,                 h_des, batch * 6 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_c + (size_t)batch * 6, W, batch * 6 * sizeof(T), cudaMemcpyHostToDevice);
    size_t smem = grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    // momentum_cost is register-heavy (~140 regs/thread): clamp to its launch cap.
    dim3 thr = grid_clamp_threads_for(grid_plant::momentum_cost_kernel<T, true>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::momentum_cost_kernel<T, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b,
        g_plant.d_in_c, g_plant.d_in_c + (size_t)batch * 6,
        g_robot, batch);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) return 200 + (int)le;
    if (int rc = grid_rbd_sync_consume()) return rc;
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
static void launch_plant_step_gradient(GridCtx *ctx, int batch, T gravity, T dt) {
    GRID_RBD_CTX_LOCALS(ctx);
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    // clamp to the kernel's launch cap (register-heavy; mirror the mjx twin).
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_gradient_kernel<T, IT>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
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

extern "C" int grid_plant_step_gradient(long long ctx_id, 
    const T* x, const T* u, T* dAB, int batch, float gravity, float dt, int it) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    const int dab = 2 * nv * 3 * nv;
    cudaMemcpy(g_plant.d_in_a, x, batch * nx * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, u, batch * nv * sizeof(T), cudaMemcpyHostToDevice);
    GRID_RBD_IT_DISPATCH(it, launch_plant_step_gradient, batch, (T)gravity, (T)dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    if (int rc = grid_rbd_sync_consume()) return rc;
    cudaMemcpy(dAB, g_plant.d_grad, batch * dab * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention plant_step_gradient (floating base only). x/u raw mjx; the kernel
// (MUJOCO_OUTPUT=true) input-converts the stacked state + forwards to the mjx
// integrator_gradient_device (state-transition Jacobian). EULER/SI-EULER only.
template <grid::IntegratorType IT>
static void launch_plant_step_gradient_mujoco(GridCtx *ctx, int batch, T gravity, T dt) {
    GRID_RBD_CTX_LOCALS(ctx);
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_gradient_kernel<T, IT, true>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    // raise the per-kernel dynamic-smem cap (arena > 48 KB; else rc=201).
    const size_t smem = grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_gradient_kernel<T, IT, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    grid_plant::plant_step_gradient_kernel<T, IT, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr,
        smem, g_streams[0]>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, nv, g_robot, gravity, dt, batch);
}
extern "C" int grid_plant_step_gradient_mujoco(long long ctx_id, 
    const T* x, const T* u, T* dAB, int batch, float gravity, float dt, int it) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    cudaMemcpy(g_plant.d_in_a, x, batch * nx * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_plant.d_in_b, u, batch * nv * sizeof(T), cudaMemcpyHostToDevice);
    GRID_RBD_IT_DISPATCH_HESSIAN(it, launch_plant_step_gradient_mujoco, batch, (T)gravity, (T)dt);
    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }
    if (int rc = grid_rbd_sync_consume()) return rc;
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
static void launch_plant_step_hessian(GridCtx *ctx, int batch, T gravity, T dt) {
    GRID_RBD_CTX_LOCALS(ctx);
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
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    grid_plant::plant_step_hessian_kernel<T, IT><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), smem, g_streams[0]>>>(
        g_plant.d_d2AB, g_plant.d_d2AB_workspace, g_plant.d_in_a, g_plant.d_in_b, nx, nv, g_robot, gravity, dt, batch);
}

extern "C" int grid_plant_step_hessian(long long ctx_id, 
    const T* x, const T* u, T* d2AB, int batch, float gravity, float dt, int it) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
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
    if (int rc = grid_rbd_sync_consume()) return rc;
    cudaMemcpy(d2AB, g_plant.d_d2AB, batch * d2ab * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention plant_step_hessian (floating base only). x/u raw mjx; the kernel
// (MUJOCO_OUTPUT=true) input-converts the stacked state + transforms the 2nd-order
// state-transition tensor to the mjx tangent (reuses fdsva_so SO tensors + a dedicated
// mjx workspace band carved from d_workspace). EULER/SI-EULER only. Register/smem-heavy.
template <grid::IntegratorType IT>
static void launch_plant_step_hessian_mujoco(GridCtx *ctx, int batch, T gravity, T dt) {
    GRID_RBD_CTX_LOCALS(ctx);
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    const size_t smem = grid_plant::INTEGRATOR_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_hessian_kernel<T, IT, grid::GRID_DEFAULT_RESOURCE_TIER, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    dim3 thr = grid_clamp_threads_for(
        grid_plant::plant_step_hessian_kernel<T, IT, grid::GRID_DEFAULT_RESOURCE_TIER, true>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::plant_step_hessian_kernel<T, IT, grid::GRID_DEFAULT_RESOURCE_TIER, /*MUJOCO_OUTPUT=*/true><<<grid_dim, thr, smem, g_streams[0]>>>(
        g_plant.d_d2AB, g_plant.d_d2AB_workspace, g_plant.d_in_a, g_plant.d_in_b, nx, nv, g_robot, gravity, dt, batch);
}
extern "C" int grid_plant_step_hessian_mujoco(long long ctx_id, 
    const T* x, const T* u, T* d2AB, int batch, float gravity, float dt, int it) {
    GRID_RBD_CTX_OR_RETURN(ctx_id);
    if (batch < 1) return 1;
    if (batch > kMaxBatch) return 2;
    if (plant_alloc(g_ctx)) return 4;
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
    if (int rc = grid_rbd_sync_consume()) return rc;
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

// fp64 (Wave 2a): the JAX surface follows the .so's element type T — buffers are
// GRID_FFI_T (F32/F64) and the gravity/dt attrs are T (the fp32-attr rounding cap
// documented in _core.cpp:49-57 applies here too; mu stays float by contract).
#if defined(GRID_RBD_WITH_JAX)

#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

// The FFI buffer element dtype tracks the .so's T (fp64 Wave 2a).
#ifdef GRID_WRAPPER_T_DOUBLE
#define GRID_FFI_T ffi::F64
#else
#define GRID_FFI_T ffi::F32
#endif

// ── shared handler-registration shapes ───────────────────────────────────────
// The bulk of the XLA_FFI_DEFINE_HANDLER_SYMBOL registrations bind one of a
// few (N buffer args, 1 buffer ret, attr tail) shapes; these macros fold the
// repeated Bind() chains. Registrations with unusual attr lists (the
// target_jid/reference_frame ints, the xtool Span, mu, the 4-in/3-out
// momentum cost, the 3-ret plant cost/barrier binds) stay hand-written at
// their definition sites.
#define GRID_RBD_JAX_CTX_  .Ctx<ffi::PlatformStream<cudaStream_t>>()
#define GRID_RBD_JAX_ARG_  .Arg<ffi::Buffer<GRID_FFI_T>>()
#define GRID_RBD_JAX_RET_  .Ret<ffi::Buffer<GRID_FFI_T>>()
// B2 stamps: `_stamped` forward handlers append an int32 stamp RESULT after the
// value; `_checked` gradient handlers take the stamp as their FIRST operand.
#define GRID_RBD_JAX_STAMP_ARG_  .Arg<ffi::Buffer<ffi::S32>>()
#define GRID_RBD_JAX_STAMP_RET_  .Ret<ffi::Buffer<ffi::S32>>()
#define GRID_RBD_FFI_STAMP_CHECK(stamp) \
    { int _seen = 0; int _src = grid_rbd_stamp_check(g_ctx, stream, (stamp).typed_data(), &_seen); \
      if (_src == 15) return ffi::Error::Internal(grid_ctx_stamp_message(_seen, g_ctx)); \
      if (_src != 0) return ffi::Error::Internal("stamp check: cuda error"); }
#define GRID_RBD_JAX_BIND_1IN(name, impl) \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(name, impl, ffi::Ffi::Bind() \
        GRID_RBD_JAX_CTX_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_RET_.Attr<int64_t>("ctx_id"))
#define GRID_RBD_JAX_BIND_2IN(name, impl) \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(name, impl, ffi::Ffi::Bind() \
        GRID_RBD_JAX_CTX_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_RET_.Attr<int64_t>("ctx_id"))
#define GRID_RBD_JAX_BIND_1IN_GRAV(name, impl) \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(name, impl, ffi::Ffi::Bind() \
        GRID_RBD_JAX_CTX_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_RET_ .Attr<T>("gravity").Attr<int64_t>("ctx_id"))
#define GRID_RBD_JAX_BIND_2IN_GRAV(name, impl) \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(name, impl, ffi::Ffi::Bind() \
        GRID_RBD_JAX_CTX_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_RET_ .Attr<T>("gravity").Attr<int64_t>("ctx_id"))
#define GRID_RBD_JAX_BIND_3IN_GRAV(name, impl) \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(name, impl, ffi::Ffi::Bind() \
        GRID_RBD_JAX_CTX_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ \
        GRID_RBD_JAX_RET_ .Attr<T>("gravity").Attr<int64_t>("ctx_id"))
#define GRID_RBD_JAX_BIND_4IN_GRAV(name, impl) \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(name, impl, ffi::Ffi::Bind() \
        GRID_RBD_JAX_CTX_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ \
        GRID_RBD_JAX_RET_ .Attr<T>("gravity").Attr<int64_t>("ctx_id"))
#define GRID_RBD_JAX_BIND_2IN_DT_IT_GRAV(name, impl) \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(name, impl, ffi::Ffi::Bind() \
        GRID_RBD_JAX_CTX_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_RET_ \
        .Attr<T>("dt").Attr<int64_t>("it").Attr<T>("gravity").Attr<int64_t>("ctx_id"))
#define GRID_RBD_JAX_BIND_3IN_DT_IT_GRAV(name, impl) \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(name, impl, ffi::Ffi::Bind() \
        GRID_RBD_JAX_CTX_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_RET_ \
        .Attr<T>("dt").Attr<int64_t>("it").Attr<T>("gravity").Attr<int64_t>("ctx_id"))

// Shared helper: validate (B, NJ) and return batch size, or error.
// Kept inline so each handler stays self-contained for grep-ability.
// W03: every operand after the leading one must also carry the leading batch
// (the copies below are sized by that batch — a shorter operand would be read
// past its end, a longer one silently truncated).
#define GRID_RBD_FFI_VALIDATE_ROWS(buf, name, expected_last_dim, batch)          \
    do {                                                                         \
        GRID_RBD_FFI_VALIDATE_2D(buf, name, expected_last_dim);                  \
        if ((int)(buf).dimensions()[0] != (batch))                               \
            return ffi::Error::InvalidArgument(                                  \
                std::string(name) + " batch must equal the leading operand's batch"); \
    } while (0)
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

// Shared helper: post-launch check. Captures the error ONCE (cudaGetLastError
// clears it) so the cudaError NAME reaches the message — a transient fault
// (cudaErrorMemoryAllocation under VRAM pressure) is distinguishable from a
// config misfit (cudaErrorInvalidConfiguration) post-hoc. ksym must be a
// string literal.
#define GRID_RBD_FFI_CHECK_LAUNCH(ksym)                                          \
    do {                                                                         \
        cudaError_t _le = cudaGetLastError();                                    \
        if (_le != cudaSuccess)                                                  \
            return ffi::Error::Internal(                                         \
                std::string(ksym " launch failed: ") + cudaGetErrorName(_le));   \
    } while (0)

// ── BEGIN GENERATED JAX FFI HANDLERS (grid_codegen/wrapper_body_gen.py — do not hand-edit) ──
// Regenerate: .venv/bin/python -m grid_codegen.wrapper_body_gen
// Table: grid_codegen/abi_specs.py (kernel_args / jax_buffer_inputs
// et al.); docs verbatim from grid_codegen/wrapper_surface_docs.py.

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
static ffi::Error grid_rbd_jax_inverse_dynamics_body(
    GridCtx *ctx, cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> qdd,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity)
{
    GRID_RBD_CTX_LOCALS(ctx);
    GRID_RBD_FFI_VALIDATE_2D(q, "inverse_dynamics: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS;
    if (batch < 1) return ffi::Error::InvalidArgument("inverse_dynamics: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("inverse_dynamics: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "inverse_dynamics: qd", grid::NUM_JOINTS, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(qdd, "inverse_dynamics: qdd", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_data->d_qdd, qdd.typed_data(), batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    GRID_RBD_FFI_VALIDATE_2D(f_ext, "inverse_dynamics: f_ext", 6 * grid::NUM_BODIES);
    if ((int)f_ext.dimensions()[0] != batch) return ffi::Error::InvalidArgument("inverse_dynamics: f_ext batch must equal the q batch");
    cudaMemcpyAsync(g_data->d_f_ext, f_ext.typed_data(), (size_t)batch * 6 * grid::NUM_BODIES * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::inverse_dynamics_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS>(g_ctx, batch), grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_c, g_data->d_q_qd_u, stride, g_data->d_qdd, g_data->d_f_ext, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("inverse_dynamics_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_c, batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(g_data->d_f_ext, 0, (size_t)batch * 6 * grid::NUM_BODIES * sizeof(T), stream);
    return ffi::Error::Success();
}
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_inverse_dynamics_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> qdd,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    return grid_rbd_jax_inverse_dynamics_body<MUJOCO>(g_ctx, stream, q, qd, qdd, f_ext, out, gravity);
}
// B2 `_stamped` twin (the custom_vjp / autograd forward): value + int32 version stamp.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_inverse_dynamics_stamped_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> qdd,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    ffi::ResultBuffer<ffi::S32> stamp,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    ffi::Error e = grid_rbd_jax_inverse_dynamics_body<MUJOCO>(g_ctx, stream, q, qd, qdd, f_ext, out, gravity);
    if (e.failure()) return e;
    grid_rbd_stamp_write(g_ctx, stream, stamp->typed_data());
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_4IN_GRAV(grid_rbd_jax_inverse_dynamics, grid_rbd_jax_inverse_dynamics_impl<false>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics_stamped,
    grid_rbd_jax_inverse_dynamics_stamped_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Attr<T>("gravity")
        .Attr<int64_t>("ctx_id")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention inverse_dynamics (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_4IN_GRAV(grid_rbd_jax_inverse_dynamics_mujoco, grid_rbd_jax_inverse_dynamics_impl<true>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics_mujoco_stamped,
    grid_rbd_jax_inverse_dynamics_stamped_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Attr<T>("gravity")
        .Attr<int64_t>("ctx_id")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_INVERSE_DYNAMICS

#if GRID_HAS_MINV
// minv(q) → Minv  (kernel writes lower triangle only; symmetrize Python-side)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_minv_body(
    GridCtx *ctx, cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out)
{
    GRID_RBD_CTX_LOCALS(ctx);
    GRID_RBD_FFI_VALIDATE_2D(q, "minv: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("minv: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("minv: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::minv_kernel<T, grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_MINV>(g_ctx, batch), grid::MINV_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER>(), stream>>>(
            g_data->d_Minv, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("minv_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_Minv, batch * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_minv_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    return grid_rbd_jax_minv_body<MUJOCO>(g_ctx, stream, q, out);
}
// B2 `_checked` twin (the custom_vjp / autograd backward): refuses a stale forward stamp.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_minv_checked_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> stamp,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_STAMP_CHECK(stamp);
    return grid_rbd_jax_minv_body<MUJOCO>(g_ctx, stream, q, out);
}

GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_minv, grid_rbd_jax_minv_impl<false>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_minv_checked,
    grid_rbd_jax_minv_checked_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("ctx_id")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention minv (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_minv_mujoco, grid_rbd_jax_minv_impl<true>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_minv_mujoco_checked,
    grid_rbd_jax_minv_checked_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("ctx_id")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_MINV

#if GRID_HAS_FORWARD_DYNAMICS
// forward_dynamics(q, qd, u, f_ext) → qdd  (f_ext always passed; zeros if omitted)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_forward_dynamics_body(
    GridCtx *ctx, cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> u,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity)
{
    GRID_RBD_CTX_LOCALS(ctx);
    GRID_RBD_FFI_VALIDATE_2D(q, "forward_dynamics: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS;
    if (batch < 1) return ffi::Error::InvalidArgument("forward_dynamics: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("forward_dynamics: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "forward_dynamics: qd", grid::NUM_JOINTS, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(u, "forward_dynamics: u", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj], dst_pitch, u.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    GRID_RBD_FFI_VALIDATE_2D(f_ext, "forward_dynamics: f_ext", 6 * grid::NUM_BODIES);
    if ((int)f_ext.dimensions()[0] != batch) return ffi::Error::InvalidArgument("forward_dynamics: f_ext batch must equal the q batch");
    cudaMemcpyAsync(g_data->d_f_ext, f_ext.typed_data(), (size_t)batch * 6 * grid::NUM_BODIES * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::forward_dynamics_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_FORWARD_DYNAMICS>(g_ctx, batch), grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER>(), stream>>>(
            g_data->d_qdd, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("forward_dynamics_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_qdd, batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(g_data->d_f_ext, 0, (size_t)batch * 6 * grid::NUM_BODIES * sizeof(T), stream);
    return ffi::Error::Success();
}
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_forward_dynamics_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> u,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    return grid_rbd_jax_forward_dynamics_body<MUJOCO>(g_ctx, stream, q, qd, u, f_ext, out, gravity);
}
// B2 `_stamped` twin (the custom_vjp / autograd forward): value + int32 version stamp.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_forward_dynamics_stamped_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> u,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    ffi::ResultBuffer<ffi::S32> stamp,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    ffi::Error e = grid_rbd_jax_forward_dynamics_body<MUJOCO>(g_ctx, stream, q, qd, u, f_ext, out, gravity);
    if (e.failure()) return e;
    grid_rbd_stamp_write(g_ctx, stream, stamp->typed_data());
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_4IN_GRAV(grid_rbd_jax_forward_dynamics, grid_rbd_jax_forward_dynamics_impl<false>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_forward_dynamics_stamped,
    grid_rbd_jax_forward_dynamics_stamped_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Attr<T>("gravity")
        .Attr<int64_t>("ctx_id")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention forward_dynamics (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_4IN_GRAV(grid_rbd_jax_forward_dynamics_mujoco, grid_rbd_jax_forward_dynamics_impl<true>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_forward_dynamics_mujoco_stamped,
    grid_rbd_jax_forward_dynamics_stamped_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Attr<T>("gravity")
        .Attr<int64_t>("ctx_id")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_FORWARD_DYNAMICS

#if GRID_HAS_ABA
// aba(q, qd, u, f_ext) → qdd  — same kernel signature shape as forward_dynamics
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_aba_body(
    GridCtx *ctx, cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> u,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity)
{
    GRID_RBD_CTX_LOCALS(ctx);
    GRID_RBD_FFI_VALIDATE_2D(q, "aba: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS;
    if (batch < 1) return ffi::Error::InvalidArgument("aba: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("aba: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "aba: qd", grid::NUM_JOINTS, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(u, "aba: u", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj], dst_pitch, u.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    GRID_RBD_FFI_VALIDATE_2D(f_ext, "aba: f_ext", 6 * grid::NUM_BODIES);
    if ((int)f_ext.dimensions()[0] != batch) return ffi::Error::InvalidArgument("aba: f_ext batch must equal the q batch");
    cudaMemcpyAsync(g_data->d_f_ext, f_ext.typed_data(), (size_t)batch * 6 * grid::NUM_BODIES * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::aba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_ABA>(g_ctx, batch), grid::ABA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER>(), stream>>>(
            g_data->d_qdd, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("aba_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_qdd, batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(g_data->d_f_ext, 0, (size_t)batch * 6 * grid::NUM_BODIES * sizeof(T), stream);
    return ffi::Error::Success();
}
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_aba_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> u,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    return grid_rbd_jax_aba_body<MUJOCO>(g_ctx, stream, q, qd, u, f_ext, out, gravity);
}
// B2 `_stamped` twin (the custom_vjp / autograd forward): value + int32 version stamp.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_aba_stamped_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> u,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    ffi::ResultBuffer<ffi::S32> stamp,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    ffi::Error e = grid_rbd_jax_aba_body<MUJOCO>(g_ctx, stream, q, qd, u, f_ext, out, gravity);
    if (e.failure()) return e;
    grid_rbd_stamp_write(g_ctx, stream, stamp->typed_data());
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_4IN_GRAV(grid_rbd_jax_aba, grid_rbd_jax_aba_impl<false>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_aba_stamped,
    grid_rbd_jax_aba_stamped_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Attr<T>("gravity")
        .Attr<int64_t>("ctx_id")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention aba (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_4IN_GRAV(grid_rbd_jax_aba_mujoco, grid_rbd_jax_aba_impl<true>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_aba_mujoco_stamped,
    grid_rbd_jax_aba_stamped_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Attr<T>("gravity")
        .Attr<int64_t>("ctx_id")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_ABA

#if GRID_HAS_CRBA
// crba(q) → M  (kernel writes the full mass matrix; no symmetrize needed)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_crba_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "crba: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("crba: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("crba: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::crba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_CRBA>(g_ctx, batch), grid::CRBA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER>(), stream>>>(
            g_data->d_M, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("crba_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_M, batch * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_1IN_GRAV(grid_rbd_jax_crba, grid_rbd_jax_crba_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention crba (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_1IN_GRAV(grid_rbd_jax_crba_mujoco, grid_rbd_jax_crba_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_CRBA

#if GRID_HAS_END_EFFECTOR_POSE
// end_effector_pose(q) → end_effector_pose  flat (B, 6*NUM_EES)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_end_effector_pose_body(
    GridCtx *ctx, cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out)
{
    GRID_RBD_CTX_LOCALS(ctx);
    GRID_RBD_FFI_VALIDATE_2D(q, "end_effector_pose: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nee = GRID_RBD_NUM_EES;
    if (batch < 1) return ffi::Error::InvalidArgument("end_effector_pose: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("end_effector_pose: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::GRID_RBD_EE_POSE_KERNEL<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_END_EFFECTOR_POSE>(g_ctx, batch), grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_end_effector_pose, g_data->d_q_qd_u, stride, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("GRID_RBD_EE_POSE_KERNEL");
    cudaMemcpyAsync(out->typed_data(), g_data->d_end_effector_pose, batch * 6 * nee * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_end_effector_pose_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    return grid_rbd_jax_end_effector_pose_body<MUJOCO>(g_ctx, stream, q, out);
}
// B2 `_stamped` twin (the custom_vjp / autograd forward): value + int32 version stamp.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_end_effector_pose_stamped_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    ffi::ResultBuffer<ffi::S32> stamp,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    ffi::Error e = grid_rbd_jax_end_effector_pose_body<MUJOCO>(g_ctx, stream, q, out);
    if (e.failure()) return e;
    grid_rbd_stamp_write(g_ctx, stream, stamp->typed_data());
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_end_effector_pose, grid_rbd_jax_end_effector_pose_impl<false>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_stamped,
    grid_rbd_jax_end_effector_pose_stamped_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Attr<int64_t>("ctx_id")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention end_effector_pose (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_end_effector_pose_mujoco, grid_rbd_jax_end_effector_pose_impl<true>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_mujoco_stamped,
    grid_rbd_jax_end_effector_pose_stamped_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<ffi::S32>>()
        .Attr<int64_t>("ctx_id")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_END_EFFECTOR_POSE

#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
// end_effector_pose_gradient(q) → end_effector_pose_gradient d/dv flat (B, 6*NUM_EES*NV).
// Output convention: d/dv tangent (pinocchio); floating-base shape uses NV
// (= 6 + n_joints) NOT NJ. Python side reshapes/transposes to the
// (B, 6*NUM_EES, NV) row-major convention (see _handle.py).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_end_effector_pose_gradient_body(
    GridCtx *ctx, cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out)
{
    GRID_RBD_CTX_LOCALS(ctx);
    GRID_RBD_FFI_VALIDATE_2D(q, "end_effector_pose_gradient: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL, nee = GRID_RBD_NUM_EES;
    if (batch < 1) return ffi::Error::InvalidArgument("end_effector_pose_gradient: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("end_effector_pose_gradient: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::GRID_RBD_EE_POSE_GRADIENT_KERNEL<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>(g_ctx, batch), grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER>(), stream>>>(
            g_data->d_end_effector_pose_gradient, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("GRID_RBD_EE_POSE_GRADIENT_KERNEL");
    cudaMemcpyAsync(out->typed_data(), g_data->d_end_effector_pose_gradient, batch * 6 * nee * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_end_effector_pose_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    return grid_rbd_jax_end_effector_pose_gradient_body<MUJOCO>(g_ctx, stream, q, out);
}
// B2 `_checked` twin (the custom_vjp / autograd backward): refuses a stale forward stamp.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_end_effector_pose_gradient_checked_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> stamp,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_STAMP_CHECK(stamp);
    return grid_rbd_jax_end_effector_pose_gradient_body<MUJOCO>(g_ctx, stream, q, out);
}

GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_end_effector_pose_gradient, grid_rbd_jax_end_effector_pose_gradient_impl<false>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_gradient_checked,
    grid_rbd_jax_end_effector_pose_gradient_checked_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("ctx_id")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention end_effector_pose_gradient (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_end_effector_pose_gradient_mujoco, grid_rbd_jax_end_effector_pose_gradient_impl<true>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_gradient_mujoco_checked,
    grid_rbd_jax_end_effector_pose_gradient_checked_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("ctx_id")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_END_EFFECTOR_POSE_GRADIENT

#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN
// end_effector_pose_hessian(q) → end_effector_pose_hessian  flat (B, 6*NUM_EES*NV*NV)
// The kernel also writes d_end_effector_pose_gradient as a byproduct; we only return d2.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_end_effector_pose_hessian_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "end_effector_pose_hessian: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL, nee = GRID_RBD_NUM_EES;
    if (batch < 1) return ffi::Error::InvalidArgument("end_effector_pose_hessian: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("end_effector_pose_hessian: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::GRID_RBD_EE_POSE_HESSIAN_KERNEL<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>(g_ctx, batch), grid::END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER>(), stream>>>(
            g_data->d_end_effector_pose_hessian, g_data->d_end_effector_pose_gradient, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("GRID_RBD_EE_POSE_HESSIAN_KERNEL");
    cudaMemcpyAsync(out->typed_data(), g_data->d_end_effector_pose_hessian, batch * 6 * nee * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_end_effector_pose_hessian, grid_rbd_jax_end_effector_pose_hessian_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention end_effector_pose_hessian (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_end_effector_pose_hessian_mujoco, grid_rbd_jax_end_effector_pose_hessian_impl<true>);
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
static ffi::Error grid_rbd_jax_inverse_dynamics_gradient_body(
    GridCtx *ctx, cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> qdd,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity)
{
    GRID_RBD_CTX_LOCALS(ctx);
    GRID_RBD_FFI_VALIDATE_2D(q, "inverse_dynamics_gradient: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("inverse_dynamics_gradient: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("inverse_dynamics_gradient: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "inverse_dynamics_gradient: qd", grid::NUM_JOINTS, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(qdd, "inverse_dynamics_gradient: qdd", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_data->d_qdd, qdd.typed_data(), batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    GRID_RBD_FFI_VALIDATE_2D(f_ext, "inverse_dynamics_gradient: f_ext", 6 * grid::NUM_BODIES);
    if ((int)f_ext.dimensions()[0] != batch) return ffi::Error::InvalidArgument("inverse_dynamics_gradient: f_ext batch must equal the q batch");
    cudaMemcpyAsync(g_data->d_f_ext, f_ext.typed_data(), (size_t)batch * 6 * grid::NUM_BODIES * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::inverse_dynamics_gradient_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(g_ctx, batch), grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(), stream>>>(
            g_data->d_dc_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_qdd, g_data->d_f_ext, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("inverse_dynamics_gradient_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_dc_du, batch * 2 * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(g_data->d_f_ext, 0, (size_t)batch * 6 * grid::NUM_BODIES * sizeof(T), stream);
    return ffi::Error::Success();
}
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_inverse_dynamics_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> qdd,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    return grid_rbd_jax_inverse_dynamics_gradient_body<MUJOCO>(g_ctx, stream, q, qd, qdd, f_ext, out, gravity);
}
// B2 `_checked` twin (the custom_vjp / autograd backward): refuses a stale forward stamp.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_inverse_dynamics_gradient_checked_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> stamp,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> qdd,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_STAMP_CHECK(stamp);
    return grid_rbd_jax_inverse_dynamics_gradient_body<MUJOCO>(g_ctx, stream, q, qd, qdd, f_ext, out, gravity);
}

GRID_RBD_JAX_BIND_4IN_GRAV(grid_rbd_jax_inverse_dynamics_gradient, grid_rbd_jax_inverse_dynamics_gradient_impl<false>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics_gradient_checked,
    grid_rbd_jax_inverse_dynamics_gradient_checked_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<T>("gravity")
        .Attr<int64_t>("ctx_id")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention inverse_dynamics_gradient (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_4IN_GRAV(grid_rbd_jax_inverse_dynamics_gradient_mujoco, grid_rbd_jax_inverse_dynamics_gradient_impl<true>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics_gradient_mujoco_checked,
    grid_rbd_jax_inverse_dynamics_gradient_checked_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<T>("gravity")
        .Attr<int64_t>("ctx_id")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_INVERSE_DYNAMICS_GRADIENT

#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
// forward_dynamics_gradient(q, qd, u) → df_du  flat (B, 2*NV*NV)
// Python reshapes/transposes to (B, NV, 2*NV) (tangent-space; FIXED base
// NV == NJ, FLOATING base NV < NJ).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_forward_dynamics_gradient_body(
    GridCtx *ctx, cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> u,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity)
{
    GRID_RBD_CTX_LOCALS(ctx);
    GRID_RBD_FFI_VALIDATE_2D(q, "forward_dynamics_gradient: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("forward_dynamics_gradient: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("forward_dynamics_gradient: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "forward_dynamics_gradient: qd", grid::NUM_JOINTS, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(u, "forward_dynamics_gradient: u", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj], dst_pitch, u.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    GRID_RBD_FFI_VALIDATE_2D(f_ext, "forward_dynamics_gradient: f_ext", 6 * grid::NUM_BODIES);
    if ((int)f_ext.dimensions()[0] != batch) return ffi::Error::InvalidArgument("forward_dynamics_gradient: f_ext batch must equal the q batch");
    cudaMemcpyAsync(g_data->d_f_ext, f_ext.typed_data(), (size_t)batch * 6 * grid::NUM_BODIES * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::forward_dynamics_gradient_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>(g_ctx, batch), grid::FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER>(), stream>>>(
            g_data->d_df_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("forward_dynamics_gradient_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_df_du, batch * 2 * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemsetAsync(g_data->d_f_ext, 0, (size_t)batch * 6 * grid::NUM_BODIES * sizeof(T), stream);
    return ffi::Error::Success();
}
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_forward_dynamics_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> u,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    return grid_rbd_jax_forward_dynamics_gradient_body<MUJOCO>(g_ctx, stream, q, qd, u, f_ext, out, gravity);
}
// B2 `_checked` twin (the custom_vjp / autograd backward): refuses a stale forward stamp.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_forward_dynamics_gradient_checked_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> stamp,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> u,
    ffi::Buffer<GRID_FFI_T> f_ext,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_STAMP_CHECK(stamp);
    return grid_rbd_jax_forward_dynamics_gradient_body<MUJOCO>(g_ctx, stream, q, qd, u, f_ext, out, gravity);
}

GRID_RBD_JAX_BIND_4IN_GRAV(grid_rbd_jax_forward_dynamics_gradient, grid_rbd_jax_forward_dynamics_gradient_impl<false>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_forward_dynamics_gradient_checked,
    grid_rbd_jax_forward_dynamics_gradient_checked_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<T>("gravity")
        .Attr<int64_t>("ctx_id")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention forward_dynamics_gradient (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_4IN_GRAV(grid_rbd_jax_forward_dynamics_gradient_mujoco, grid_rbd_jax_forward_dynamics_gradient_impl<true>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_forward_dynamics_gradient_mujoco_checked,
    grid_rbd_jax_forward_dynamics_gradient_checked_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<T>("gravity")
        .Attr<int64_t>("ctx_id")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_FORWARD_DYNAMICS_GRADIENT

#if GRID_HAS_FDSVA_SO
// fdsva_so(q, qd, u) → packed (B, SECOND_ORDER_TENSOR_SIZE)
// Uses d_idsva_so as scratch — must not run concurrently with idsva_so.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_fdsva_so_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> u,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "fdsva_so: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS;
    if (batch < 1) return ffi::Error::InvalidArgument("fdsva_so: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("fdsva_so: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "fdsva_so: qd", grid::NUM_JOINTS, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(u, "fdsva_so: u", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj], dst_pitch, u.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::fdsva_so_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_FDSVA_SO>(g_ctx, batch), grid::FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER>(), stream>>>(
            g_data->d_df2, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_idsva_so, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("fdsva_so_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_df2, batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_3IN_GRAV(grid_rbd_jax_fdsva_so, grid_rbd_jax_fdsva_so_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention fdsva_so (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_3IN_GRAV(grid_rbd_jax_fdsva_so_mujoco, grid_rbd_jax_fdsva_so_impl<true>);
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
static ffi::Error grid_rbd_jax_inverse_dynamics_regressor_body(
    GridCtx *ctx, cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> qdd,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity)
{
    GRID_RBD_CTX_LOCALS(ctx);
    GRID_RBD_FFI_VALIDATE_2D(q, "inverse_dynamics_regressor: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL, nb = grid::NUM_BODIES;
    if (batch < 1) return ffi::Error::InvalidArgument("inverse_dynamics_regressor: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("inverse_dynamics_regressor: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "inverse_dynamics_regressor: qd", grid::NUM_JOINTS, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(qdd, "inverse_dynamics_regressor: qdd", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj], dst_pitch, qdd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::inverse_dynamics_regressor_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_REGRESSOR>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS_REGRESSOR>(g_ctx, batch), grid::INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_REGRESSOR>::TIER>(), stream>>>(
            g_data->d_Y, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("inverse_dynamics_regressor_kernel");
    const int out_size = grid::NUM_VEL * 10 * grid::NUM_BODIES;
    cudaMemcpyAsync(out->typed_data(), g_data->d_Y, batch * out_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_inverse_dynamics_regressor_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> qdd,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    return grid_rbd_jax_inverse_dynamics_regressor_body<MUJOCO>(g_ctx, stream, q, qd, qdd, out, gravity);
}
// B2 `_checked` twin (the custom_vjp / autograd backward): refuses a stale forward stamp.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_inverse_dynamics_regressor_checked_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> stamp,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> qdd,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_STAMP_CHECK(stamp);
    return grid_rbd_jax_inverse_dynamics_regressor_body<MUJOCO>(g_ctx, stream, q, qd, qdd, out, gravity);
}

GRID_RBD_JAX_BIND_3IN_GRAV(grid_rbd_jax_inverse_dynamics_regressor, grid_rbd_jax_inverse_dynamics_regressor_impl<false>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics_regressor_checked,
    grid_rbd_jax_inverse_dynamics_regressor_checked_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<T>("gravity")
        .Attr<int64_t>("ctx_id")
);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention inverse_dynamics_regressor (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_3IN_GRAV(grid_rbd_jax_inverse_dynamics_regressor_mujoco, grid_rbd_jax_inverse_dynamics_regressor_impl<true>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics_regressor_mujoco_checked,
    grid_rbd_jax_inverse_dynamics_regressor_checked_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<T>("gravity")
        .Attr<int64_t>("ctx_id")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_INVERSE_DYNAMICS_REGRESSOR

#if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
// forward_dynamics_parameter_gradient(q, qd, u) → dqdd/dpi = -Minv . Y
// flat (B, NV*10*NUM_BODIES). Internally runs FD at (q,qd,u) and the regressor
// at the resulting qdd, then applies -Minv (mirrors RBDReference). The kernel
// takes d_workspace (the s_Y regressor scratch spills there at LITE/MINIMAL).
static ffi::Error grid_rbd_jax_forward_dynamics_parameter_gradient_body(
    GridCtx *ctx, cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> u,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity)
{
    GRID_RBD_CTX_LOCALS(ctx);
    GRID_RBD_FFI_VALIDATE_2D(q, "forward_dynamics_parameter_gradient: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL, nb = grid::NUM_BODIES;
    if (batch < 1) return ffi::Error::InvalidArgument("forward_dynamics_parameter_gradient: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("forward_dynamics_parameter_gradient: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "forward_dynamics_parameter_gradient: qd", grid::NUM_JOINTS, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(u, "forward_dynamics_parameter_gradient: u", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj], dst_pitch, u.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::forward_dynamics_parameter_gradient_kernel<T><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_FORWARD_DYNAMICS_PARAMETER_GRADIENT>(g_ctx, batch), grid::FORWARD_DYNAMICS_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_PARAMETER_GRADIENT>::TIER>(), stream>>>(
            g_data->d_dqdd_dpi, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("forward_dynamics_parameter_gradient_kernel");
    const int out_size = grid::NUM_VEL * 10 * grid::NUM_BODIES;
    cudaMemcpyAsync(out->typed_data(), g_data->d_dqdd_dpi, batch * out_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}
static ffi::Error grid_rbd_jax_forward_dynamics_parameter_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> u,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    return grid_rbd_jax_forward_dynamics_parameter_gradient_body(g_ctx, stream, q, qd, u, out, gravity);
}
// B2 `_checked` twin (the custom_vjp / autograd backward): refuses a stale forward stamp.
static ffi::Error grid_rbd_jax_forward_dynamics_parameter_gradient_checked_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> stamp,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> u,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_STAMP_CHECK(stamp);
    return grid_rbd_jax_forward_dynamics_parameter_gradient_body(g_ctx, stream, q, qd, u, out, gravity);
}

GRID_RBD_JAX_BIND_3IN_GRAV(grid_rbd_jax_forward_dynamics_parameter_gradient, grid_rbd_jax_forward_dynamics_parameter_gradient_impl);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_forward_dynamics_parameter_gradient_checked,
    grid_rbd_jax_forward_dynamics_parameter_gradient_checked_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<T>("gravity")
        .Attr<int64_t>("ctx_id")
);
#endif  // GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT

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
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "generalized_gravity: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("generalized_gravity: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("generalized_gravity: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::generalized_gravity_kernel<T, grid::launch_cfg<grid::GRID_ALGO_GENERALIZED_GRAVITY>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_GENERALIZED_GRAVITY>(g_ctx, batch), grid::INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_c, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("generalized_gravity_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_c, batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_1IN_GRAV(grid_rbd_jax_generalized_gravity, grid_rbd_jax_generalized_gravity_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_BIND_1IN_GRAV(grid_rbd_jax_generalized_gravity_mujoco, grid_rbd_jax_generalized_gravity_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_GENERALIZED_GRAVITY

#if GRID_HAS_NONLINEAR_EFFECTS
// nonlinear_effects(q, qd) → c(q,qd), NV  [d_workspace, gravity]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_nonlinear_effects_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "nonlinear_effects: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("nonlinear_effects: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("nonlinear_effects: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "nonlinear_effects: qd", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::nonlinear_effects_kernel<T, grid::launch_cfg<grid::GRID_ALGO_NONLINEAR_EFFECTS>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_NONLINEAR_EFFECTS>(g_ctx, batch), grid::INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_c, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("nonlinear_effects_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_c, batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_2IN_GRAV(grid_rbd_jax_nonlinear_effects, grid_rbd_jax_nonlinear_effects_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_BIND_2IN_GRAV(grid_rbd_jax_nonlinear_effects_mujoco, grid_rbd_jax_nonlinear_effects_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_NONLINEAR_EFFECTS

#if GRID_HAS_CORIOLIS_MATRIX
// coriolis_matrix(q, qd) → C(q,qd), NV*NV row-major  [d_workspace, gravity]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_coriolis_matrix_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "coriolis_matrix: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("coriolis_matrix: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("coriolis_matrix: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "coriolis_matrix: qd", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::coriolis_matrix_kernel<T, grid::launch_cfg<grid::GRID_ALGO_CORIOLIS_MATRIX>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_CORIOLIS_MATRIX>(g_ctx, batch), grid::CORIOLIS_MATRIX_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_CORIOLIS_MATRIX>::TIER>(), stream>>>(
            g_data->d_coriolis, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("coriolis_matrix_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_coriolis, batch * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_2IN_GRAV(grid_rbd_jax_coriolis_matrix, grid_rbd_jax_coriolis_matrix_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_BIND_2IN_GRAV(grid_rbd_jax_coriolis_matrix_mujoco, grid_rbd_jax_coriolis_matrix_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_CORIOLIS_MATRIX

#if GRID_HAS_KINETIC_ENERGY_REGRESSOR
// kinetic_energy_regressor(q, qd) → y_KE, 10*NUM_BODIES  [gravity, no workspace]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_kinetic_energy_regressor_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "kinetic_energy_regressor: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nb = grid::NUM_BODIES;
    if (batch < 1) return ffi::Error::InvalidArgument("kinetic_energy_regressor: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("kinetic_energy_regressor: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "kinetic_energy_regressor: qd", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::kinetic_energy_regressor_kernel<T, grid::launch_cfg<grid::GRID_ALGO_KINETIC_ENERGY_REGRESSOR>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_KINETIC_ENERGY_REGRESSOR>(g_ctx, batch), grid::KINETIC_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_ke_regressor, g_data->d_q_qd_u, stride, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("kinetic_energy_regressor_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_ke_regressor, batch * 10 * nb * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_2IN_GRAV(grid_rbd_jax_kinetic_energy_regressor, grid_rbd_jax_kinetic_energy_regressor_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_BIND_2IN_GRAV(grid_rbd_jax_kinetic_energy_regressor_mujoco, grid_rbd_jax_kinetic_energy_regressor_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_KINETIC_ENERGY_REGRESSOR

#if GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
// potential_energy_regressor(q) → y_PE, 10*NUM_BODIES  [gravity, no workspace]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_potential_energy_regressor_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "potential_energy_regressor: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nb = grid::NUM_BODIES;
    if (batch < 1) return ffi::Error::InvalidArgument("potential_energy_regressor: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("potential_energy_regressor: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::potential_energy_regressor_kernel<T, grid::launch_cfg<grid::GRID_ALGO_POTENTIAL_ENERGY_REGRESSOR>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_POTENTIAL_ENERGY_REGRESSOR>(g_ctx, batch), grid::POTENTIAL_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_pe_regressor, g_data->d_q_qd_u, stride, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("potential_energy_regressor_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_pe_regressor, batch * 10 * nb * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_1IN_GRAV(grid_rbd_jax_potential_energy_regressor, grid_rbd_jax_potential_energy_regressor_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_BIND_1IN_GRAV(grid_rbd_jax_potential_energy_regressor_mujoco, grid_rbd_jax_potential_energy_regressor_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_POTENTIAL_ENERGY_REGRESSOR

// ─── Wave 2: gated value ops (energy/com/ccrba/cmm/dccrba) ───────────────────

#ifdef GRID_HAS_ENERGY
// energy(q, qd) → [KE, PE, KE+PE], 3  [gravity, no workspace]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_energy_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "energy: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS;
    if (batch < 1) return ffi::Error::InvalidArgument("energy: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("energy: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "energy: qd", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::energy_kernel<T, grid::launch_cfg<grid::GRID_ALGO_ENERGY>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_ENERGY>(g_ctx, batch), grid::ENERGY_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_ENERGY>::TIER>(), stream>>>(
            g_data->d_energy, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("energy_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_energy, batch * 3 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_2IN_GRAV(grid_rbd_jax_energy, grid_rbd_jax_energy_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_BIND_2IN_GRAV(grid_rbd_jax_energy_mujoco, grid_rbd_jax_energy_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_ENERGY

#ifdef GRID_HAS_COM
// com(q) → flat [p_com(3); J_com(3*NV)], 3 + 3*NV  [no workspace, no gravity]
// Single flat buffer; the Python layer splits the (p_com, J_com) tuple.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_com_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "com: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("com: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("com: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::com_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COM>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_COM>(g_ctx, batch), grid::COM_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_COM>::TIER>(), stream>>>(
            g_data->d_com, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("com_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_com, batch * (3 + 3 * nv) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_com, grid_rbd_jax_com_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_com_mujoco, grid_rbd_jax_com_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_COM

#ifdef GRID_HAS_CCRBA
// ccrba(q, qd) → flat [A(6*NV); h(6)], 6*NV + 6  [no workspace, no gravity]
// Single flat buffer; the Python layer splits the (A, h) tuple.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_ccrba_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::ResultBuffer<GRID_FFI_T> out,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "ccrba: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("ccrba: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("ccrba: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "ccrba: qd", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::ccrba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_CCRBA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_CCRBA>(g_ctx, batch), grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_CCRBA>::TIER>(), stream>>>(
            g_data->d_ccrba, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("ccrba_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_ccrba, batch * (6 * nv + 6) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_2IN(grid_rbd_jax_ccrba, grid_rbd_jax_ccrba_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_BIND_2IN(grid_rbd_jax_ccrba_mujoco, grid_rbd_jax_ccrba_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_CCRBA

#ifdef GRID_HAS_CMM_TIME_VARIATION
// cmm_time_variation(q, qd) → Adot, 6*NV  [d_workspace, no gravity]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_cmm_time_variation_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::ResultBuffer<GRID_FFI_T> out,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "cmm_time_variation: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("cmm_time_variation: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("cmm_time_variation: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "cmm_time_variation: qd", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::cmm_time_variation_kernel<T, grid::launch_cfg<grid::GRID_ALGO_CMM_TIME_VARIATION>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_CMM_TIME_VARIATION>(g_ctx, batch), grid::CMM_TIME_VARIATION_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_CMM_TIME_VARIATION>::TIER>(), stream>>>(
            g_data->d_cmm_time_variation, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("cmm_time_variation_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_cmm_time_variation, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_2IN(grid_rbd_jax_cmm_time_variation, grid_rbd_jax_cmm_time_variation_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_BIND_2IN(grid_rbd_jax_cmm_time_variation_mujoco, grid_rbd_jax_cmm_time_variation_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_CMM_TIME_VARIATION

#ifdef GRID_HAS_DCCRBA
// dccrba(q) → dA/dq, 6*NV*NV  [d_workspace, no gravity]  (compressed d_q kernel)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_dccrba_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "dccrba: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("dccrba: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("dccrba: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::dccrba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_DCCRBA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_DCCRBA>(g_ctx, batch), grid::DCCRBA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_DCCRBA>::TIER>(), stream>>>(
            g_data->d_dccrba, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("dccrba_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_dccrba, batch * 6 * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_dccrba, grid_rbd_jax_dccrba_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_dccrba_mujoco, grid_rbd_jax_dccrba_impl<true>);
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
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    int64_t target_jid,
    int64_t reference_frame,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "frame_jacobian: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("frame_jacobian: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("frame_jacobian: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::frame_jacobian_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FRAME_JACOBIAN>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_FRAME_JACOBIAN>(g_ctx, batch), grid::FRAME_JACOBIAN_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_frame_jacobian, g_data->d_q_qd_u, stride, (int)target_jid, (int)reference_frame, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("frame_jacobian_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_frame_jacobian, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_frame_jacobian,
    grid_rbd_jax_frame_jacobian_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("target_jid")
        .Attr<int64_t>("reference_frame")
        .Attr<int64_t>("ctx_id")
);

#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_frame_jacobian_mujoco,
    grid_rbd_jax_frame_jacobian_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("target_jid")
        .Attr<int64_t>("reference_frame")
        .Attr<int64_t>("ctx_id")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_FRAME_JACOBIAN

#if defined(GRID_HAS_FRAME_JACOBIAN) && defined(GRID_HAS_FRAME_JACOBIAN_DOT)
// frame_jacobian_dot(q, qd) → 6*NV d/dt Jacobian  [no workspace, no gravity, int attrs]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_frame_jacobian_dot_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::ResultBuffer<GRID_FFI_T> out,
    int64_t target_jid,
    int64_t reference_frame,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "frame_jacobian_dot: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("frame_jacobian_dot: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("frame_jacobian_dot: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "frame_jacobian_dot: qd", grid::NUM_JOINTS, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch, qd.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::frame_jacobian_dot_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FRAME_JACOBIAN_DOT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_FRAME_JACOBIAN_DOT>(g_ctx, batch), grid::FRAME_JACOBIAN_DOT_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_frame_jacobian_dot, g_data->d_q_qd_u, stride, (int)target_jid, (int)reference_frame, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("frame_jacobian_dot_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_frame_jacobian_dot, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_frame_jacobian_dot,
    grid_rbd_jax_frame_jacobian_dot_impl<false>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("target_jid")
        .Attr<int64_t>("reference_frame")
        .Attr<int64_t>("ctx_id")
);

#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_frame_jacobian_dot_mujoco,
    grid_rbd_jax_frame_jacobian_dot_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("target_jid")
        .Attr<int64_t>("reference_frame")
        .Attr<int64_t>("ctx_id")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_FRAME_JACOBIAN && GRID_HAS_FRAME_JACOBIAN_DOT

#if defined(GRID_HAS_FRAME_JACOBIAN) && defined(GRID_HAS_OSC_INERTIA)
// osc_inertia(q) → 6x6 task inertia Lambda, 36  [d_workspace (tier-spill scratch), no gravity, frame baked at codegen]
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_osc_inertia_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> out,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "osc_inertia: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS;
    if (batch < 1) return ffi::Error::InvalidArgument("osc_inertia: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("osc_inertia: batch > max_batch");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch, q.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::osc_inertia_kernel<T, grid::launch_cfg<grid::GRID_ALGO_OSC_INERTIA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_OSC_INERTIA>(g_ctx, batch), grid::OSC_INERTIA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_OSC_INERTIA>::TIER>(), stream>>>(
            g_data->d_osc_inertia, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("osc_inertia_kernel");
    cudaMemcpyAsync(out->typed_data(), g_data->d_osc_inertia, batch * 36 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_osc_inertia, grid_rbd_jax_osc_inertia_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_BIND_1IN(grid_rbd_jax_osc_inertia_mujoco, grid_rbd_jax_osc_inertia_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_FRAME_JACOBIAN && GRID_HAS_OSC_INERTIA

// ── END GENERATED JAX FFI HANDLERS ──


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
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> ee_out,
    int64_t target_jid, ffi::Span<const float> xtool,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "end_effector_pose_runtime: q", grid::NUM_JOINTS);
    if (target_jid < 0 || target_jid >= grid::NUM_JOINTS) return ffi::Error::InvalidArgument("end_effector_pose_runtime: target_jid out of range");
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS;
    if (batch < 1) return ffi::Error::InvalidArgument("end_effector_pose_runtime: batch must be >= 1");
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
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch),
        grid::END_EFFECTOR_POSE_RUNTIME_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_eePose, g_data->d_q_qd_u, stride_q,
            (int)target_jid, g_data->d_eepose_runtime_offset, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("end_effector_pose_runtime_kernel");

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
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("target_jid")
        .Attr<ffi::Span<const float>>("xtool")
.Attr<int64_t>("ctx_id"));
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_runtime_mujoco,
    grid_rbd_jax_end_effector_pose_runtime_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("target_jid")
        .Attr<ffi::Span<const float>>("xtool")
.Attr<int64_t>("ctx_id"));
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_END_EFFECTOR_POSE_RUNTIME


#ifdef GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME
// end_effector_pose_gradient_runtime(q) → (B, 6*NV) col-major d[xyz; rpy]/dv of
// target_jid at the offset point. Python reshapes (B,NV,6)→transpose→(B,6,NV).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_end_effector_pose_gradient_runtime_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::ResultBuffer<GRID_FFI_T> dee_out,
    int64_t target_jid, ffi::Span<const float> xtool,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "end_effector_pose_gradient_runtime: q", grid::NUM_JOINTS);
    if (target_jid < 0 || target_jid >= grid::NUM_JOINTS) return ffi::Error::InvalidArgument("end_effector_pose_gradient_runtime: target_jid out of range");
    int batch = (int)q.dimensions()[0];
    int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("end_effector_pose_gradient_runtime: batch must be >= 1");
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
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch),
        grid::END_EFFECTOR_POSE_GRADIENT_RUNTIME_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_eePoseGrad, g_data->d_q_qd_u, stride_q,
            (int)target_jid, g_data->d_eepose_runtime_offset, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("end_effector_pose_gradient_runtime_kernel");

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
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("target_jid")
        .Attr<ffi::Span<const float>>("xtool")
.Attr<int64_t>("ctx_id"));
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_gradient_runtime_mujoco,
    grid_rbd_jax_end_effector_pose_gradient_runtime_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("target_jid")
        .Attr<ffi::Span<const float>>("xtool")
.Attr<int64_t>("ctx_id"));
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME


#if GRID_HAS_IDSVA_SO
// idsva_so(q, qd, qdd) → packed (B, SECOND_ORDER_TENSOR_SIZE)
// The codegen-time dispatcher picks body- vs world-frame; we mirror that pick
// at compile time via GRID_IDSVA_SO_DISPATCHES_WORLD_FRAME so a per-robot .so
// calls the kernel grid::idsva_so itself would. qdd is packed into the acceleration (u) slot of
// d_q_qd_u, which the kernel reads as s_qdd — mirroring the numpy
// pack_q_qd_u(g_ctx, q, qd, qdd). The Python surface passes explicit zeros when the
// caller omits qdd, so the result never depends on a stale device buffer.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_idsva_so_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q,
    ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> qdd,
    ffi::ResultBuffer<GRID_FFI_T> out,
    T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_VALIDATE_2D(q, "idsva_so: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch < 1) return ffi::Error::InvalidArgument("idsva_so: batch must be >= 1");
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
    // Compile-time frame dispatch — mirrors grid::idsva_so's own codegen rule
    // (GRID_IDSVA_SO_DISPATCHES_WORLD_FRAME: world for floating / spherical /
    // high-DOF fixed, body for cardinal fixed). ⚠NOT the mjx-twins gate: a
    // floating MIMIC robot (h1_2) has no GRID_RBD_WITH_MUJOCO yet still
    // dispatches world — keying this on WITH_MUJOCO launched the 3MB body-frame
    // no-ladder diagnostic there (unlaunchable at any tier; agent guide §1m).
    // The MUJOCO_OUTPUT template param exists only where the host template
    // carries it (GRID_RBD_SIG_MJX_IDSVA_SO — floating bases), so the
    // world-frame spelling forks on that flag.
#if GRID_IDSVA_SO_DISPATCHES_WORLD_FRAME
#ifdef GRID_RBD_SIG_MJX_IDSVA_SO
    grid::idsva_so_world_frame_kernel<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>(g_ctx, batch),
        grid::IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>::TIER>(),
        stream>>>(
            g_data->d_idsva_so, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u,
            g_robot, /*gravity=*/gravity, batch);
#else
    static_assert(!MUJOCO, "mjx idsva_so is floating-base only");
    grid::idsva_so_world_frame_kernel<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>::TIER><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>(g_ctx, batch),
        grid::IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>::TIER>(),
        stream>>>(
            g_data->d_idsva_so, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u,
            g_robot, /*gravity=*/gravity, batch);
#endif
    GRID_RBD_FFI_CHECK_LAUNCH("idsva_so_world_frame_kernel");
#else
    static_assert(!MUJOCO, "mjx idsva_so is floating-base only");
    grid::idsva_so_body_frame_kernel<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_BODY_FRAME>::TIER><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_IDSVA_SO_BODY_FRAME>(g_ctx, batch),
        grid::IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_BODY_FRAME>::TIER>(),
        stream>>>(
            g_data->d_idsva_so, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u,
            g_robot, /*gravity=*/gravity, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("idsva_so_body_frame_kernel");
#endif

    cudaMemcpyAsync(out->typed_data(), g_data->d_idsva_so,
                    batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_3IN_GRAV(grid_rbd_jax_idsva_so, grid_rbd_jax_idsva_so_impl<false>);

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention idsva_so (floating only): world-frame kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_3IN_GRAV(grid_rbd_jax_idsva_so_mujoco, grid_rbd_jax_idsva_so_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_IDSVA_SO


// Integrator. dt + it are FFI attributes (runtime scalars; gravity is the
// standard constant). q/qd/u are packed D→D like aba; the integrator kernels
// are launched directly on the JAX stream.
#if GRID_HAS_INTEGRATOR
template <grid::IntegratorType IT, bool MUJOCO>
static void launch_integrator_kernel_jax(GridCtx *ctx, cudaStream_t stream, int batch, T dt, T gravity) {
    GRID_RBD_CTX_LOCALS(ctx);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::integrator_kernel<T, IT, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INTEGRATOR>(g_ctx, batch),
        grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER>(), stream>>>(
            g_data->d_x_kp1, g_data->d_workspace, g_data->d_q_qd_u, stride,
            g_robot, g_data->d_f_ext, /*gravity=*/static_cast<T>(gravity), static_cast<T>(dt), batch);
}
#endif  // GRID_HAS_INTEGRATOR
#if GRID_HAS_INTEGRATOR_GRADIENT
template <grid::IntegratorType IT, bool MUJOCO>
static void launch_integrator_grad_kernel_jax(GridCtx *ctx, cudaStream_t stream, int batch, T dt, T gravity) {
    GRID_RBD_CTX_LOCALS(ctx);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::integrator_gradient_kernel<T, IT, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<
        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INTEGRATOR_GRADIENT>(g_ctx, batch),
        grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR_GRADIENT>::TIER>(), stream>>>(
            g_data->d_dAB, g_data->d_workspace, g_data->d_q_qd_u, stride,
            g_robot, g_data->d_f_ext, /*gravity=*/static_cast<T>(gravity), static_cast<T>(dt), batch);
}
#endif  // GRID_HAS_INTEGRATOR_GRADIENT

// mjx-aware integrator-type dispatch: instantiates FN<IT_case, MUJOCO_FLAG>. The
// helper's first template arg is supplied by the case; the second (MUJOCO) is
// threaded from the templated impl so the kernel launches with MUJOCO_OUTPUT.
#define GRID_RBD_IT_DISPATCH_FFI_MJX(it_code, FN, MUJOCO_FLAG, ...)                                \
    GRID_RBD_IT_SWITCH_MJX(it_code, FN, MUJOCO_FLAG,                                                \
        return ffi::Error::InvalidArgument("integrator: bad it code"), __VA_ARGS__)

// Single-stage-only mjx dispatch: the gradient (state-transition Jacobian) mjx epilogue
// is currently Euler / Semi-Implicit-Euler only (multi-stage RK mjx derivatives are
// deferred — the device code static_asserts). Multi-stage codes are rejected at runtime.
#define GRID_RBD_IT_DISPATCH_FFI_MJX_SS(it_code, FN, MUJOCO_FLAG, ...)                             \
    GRID_RBD_IT_SWITCH_MJX_SS(it_code, FN, MUJOCO_FLAG,                                             \
        return ffi::Error::InvalidArgument(                                                         \
            "mujoco integrator/plant-step gradient supports only euler / semi-implicit-euler"),     \
        __VA_ARGS__)

// q/qd/u D→D pack into the singleton's d_q_qd_u (layout [q,qd,u] per timestep).
static ffi::Error grid_rbd_jax_pack_qqdu(GridCtx *ctx, cudaStream_t stream, int batch, int nj,
                                   ffi::Buffer<GRID_FFI_T>& q,
                                   ffi::Buffer<GRID_FFI_T>& qd,
                                   ffi::Buffer<GRID_FFI_T>& u) {
    GRID_RBD_CTX_LOCALS(ctx);
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "qd", nj, batch);   // W03: q's batch sizes every copy
    GRID_RBD_FFI_VALIDATE_ROWS(u, "u", nj, batch);
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],     dst_pitch, q.typed_data(),  row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj],    dst_pitch, qd.typed_data(), row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj],  dst_pitch, u.typed_data(),  row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

#if GRID_HAS_INTEGRATOR
// integrator(q, qd, u; dt, it) → x_kp1  (B, NUM_POS + NUM_VEL)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_integrator_body(
    GridCtx *ctx, cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q, ffi::Buffer<GRID_FFI_T> qd, ffi::Buffer<GRID_FFI_T> u,
    ffi::ResultBuffer<GRID_FFI_T> x_kp1_out,
    T dt, int64_t it, T gravity)
{
    GRID_RBD_CTX_LOCALS(ctx);
    GRID_RBD_FFI_VALIDATE_2D(q, "integrator: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch < 1) return ffi::Error::InvalidArgument("integrator: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("integrator: batch > max_batch");

    { ffi::Error _pe = grid_rbd_jax_pack_qqdu(g_ctx, stream, batch, nj, q, qd, u); if (_pe.failure()) return _pe; }
    GRID_RBD_IT_DISPATCH_FFI_MJX((int)it, launch_integrator_kernel_jax, MUJOCO, stream, batch, dt, gravity);
    GRID_RBD_FFI_CHECK_LAUNCH("integrator_kernel");

    cudaMemcpyAsync(x_kp1_out->typed_data(), g_data->d_x_kp1,
                    batch * (grid::NUM_POS + grid::NUM_VEL) * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_integrator_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q, ffi::Buffer<GRID_FFI_T> qd, ffi::Buffer<GRID_FFI_T> u,
    ffi::ResultBuffer<GRID_FFI_T> x_kp1_out,
    T dt, int64_t it, T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    return grid_rbd_jax_integrator_body<MUJOCO>(g_ctx, stream, q, qd, u, x_kp1_out, dt, it, gravity);
}
// B2 `_stamped` twin (the custom_vjp forward): value + int32 version stamp.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_integrator_stamped_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q, ffi::Buffer<GRID_FFI_T> qd, ffi::Buffer<GRID_FFI_T> u,
    ffi::ResultBuffer<GRID_FFI_T> x_kp1_out, ffi::ResultBuffer<ffi::S32> stamp,
    T dt, int64_t it, T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    ffi::Error e = grid_rbd_jax_integrator_body<MUJOCO>(g_ctx, stream, q, qd, u, x_kp1_out, dt, it, gravity);
    if (e.failure()) return e;
    grid_rbd_stamp_write(g_ctx, stream, stamp->typed_data());
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_3IN_DT_IT_GRAV(grid_rbd_jax_integrator, grid_rbd_jax_integrator_impl<false>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(grid_rbd_jax_integrator_stamped, grid_rbd_jax_integrator_stamped_impl<false>, ffi::Ffi::Bind()
    GRID_RBD_JAX_CTX_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_RET_ GRID_RBD_JAX_STAMP_RET_
    .Attr<T>("dt").Attr<int64_t>("it").Attr<T>("gravity").Attr<int64_t>("ctx_id"));

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention integrator (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_3IN_DT_IT_GRAV(grid_rbd_jax_integrator_mujoco, grid_rbd_jax_integrator_impl<true>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(grid_rbd_jax_integrator_mujoco_stamped, grid_rbd_jax_integrator_stamped_impl<true>, ffi::Ffi::Bind()
    GRID_RBD_JAX_CTX_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_RET_ GRID_RBD_JAX_STAMP_RET_
    .Attr<T>("dt").Attr<int64_t>("it").Attr<T>("gravity").Attr<int64_t>("ctx_id"));
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_HAS_INTEGRATOR

#if GRID_HAS_INTEGRATOR_GRADIENT
// integrator_gradient(q, qd, u; dt, it) → dAB  (B, 2*NV, 3*NV)
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_integrator_gradient_body(
    GridCtx *ctx, cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q, ffi::Buffer<GRID_FFI_T> qd, ffi::Buffer<GRID_FFI_T> u,
    ffi::ResultBuffer<GRID_FFI_T> dAB_out,
    T dt, int64_t it, T gravity)
{
    GRID_RBD_CTX_LOCALS(ctx);
    GRID_RBD_FFI_VALIDATE_2D(q, "integrator_gradient: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    int nv    = grid::NUM_VEL;
    if (batch < 1) return ffi::Error::InvalidArgument("integrator_gradient: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("integrator_gradient: batch > max_batch");

    { ffi::Error _pe = grid_rbd_jax_pack_qqdu(g_ctx, stream, batch, nj, q, qd, u); if (_pe.failure()) return _pe; }
    // mjx gradient is single-stage (euler/si) only; pin supports all integrator types.
    if constexpr (MUJOCO) {
        GRID_RBD_IT_DISPATCH_FFI_MJX_SS((int)it, launch_integrator_grad_kernel_jax, true, stream, batch, dt, gravity);
    } else {
        GRID_RBD_IT_DISPATCH_FFI_MJX((int)it, launch_integrator_grad_kernel_jax, false, stream, batch, dt, gravity);
    }
    GRID_RBD_FFI_CHECK_LAUNCH("integrator_grad_kernel");

    cudaMemcpyAsync(dAB_out->typed_data(), g_data->d_dAB,
                    batch * (2 * nv) * (3 * nv) * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_integrator_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q, ffi::Buffer<GRID_FFI_T> qd, ffi::Buffer<GRID_FFI_T> u,
    ffi::ResultBuffer<GRID_FFI_T> dAB_out,
    T dt, int64_t it, T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    return grid_rbd_jax_integrator_gradient_body<MUJOCO>(g_ctx, stream, q, qd, u, dAB_out, dt, it, gravity);
}
// B2 `_checked` twin (the custom_vjp backward): refuses a stale forward stamp.
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_integrator_gradient_checked_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::S32> stamp,
    ffi::Buffer<GRID_FFI_T> q, ffi::Buffer<GRID_FFI_T> qd, ffi::Buffer<GRID_FFI_T> u,
    ffi::ResultBuffer<GRID_FFI_T> dAB_out,
    T dt, int64_t it, T gravity,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    GRID_RBD_FFI_STAMP_CHECK(stamp);
    return grid_rbd_jax_integrator_gradient_body<MUJOCO>(g_ctx, stream, q, qd, u, dAB_out, dt, it, gravity);
}

GRID_RBD_JAX_BIND_3IN_DT_IT_GRAV(grid_rbd_jax_integrator_gradient, grid_rbd_jax_integrator_gradient_impl<false>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(grid_rbd_jax_integrator_gradient_checked, grid_rbd_jax_integrator_gradient_checked_impl<false>, ffi::Ffi::Bind()
    GRID_RBD_JAX_CTX_ GRID_RBD_JAX_STAMP_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_RET_
    .Attr<T>("dt").Attr<int64_t>("it").Attr<T>("gravity").Attr<int64_t>("ctx_id"));

#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention integrator_gradient (floating only): identical plumbing, kernel launched with MUJOCO_OUTPUT=true.
GRID_RBD_JAX_BIND_3IN_DT_IT_GRAV(grid_rbd_jax_integrator_gradient_mujoco, grid_rbd_jax_integrator_gradient_impl<true>);
XLA_FFI_DEFINE_HANDLER_SYMBOL(grid_rbd_jax_integrator_gradient_mujoco_checked, grid_rbd_jax_integrator_gradient_checked_impl<true>, ffi::Ffi::Bind()
    GRID_RBD_JAX_CTX_ GRID_RBD_JAX_STAMP_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_ARG_ GRID_RBD_JAX_RET_
    .Attr<T>("dt").Attr<int64_t>("it").Attr<T>("gravity").Attr<int64_t>("ctx_id"));
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
    ffi::Buffer<GRID_FFI_T> var, ffi::Buffer<GRID_FFI_T> des, ffi::Buffer<GRID_FFI_T> w,
    ffi::ResultBuffer<GRID_FFI_T> out, ffi::ResultBuffer<GRID_FFI_T> grad,
    ffi::ResultBuffer<GRID_FFI_T> hess,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    if (plant_alloc(g_ctx)) return ffi::Error::Internal("plant_alloc failed");
    const int N = STATE ? (grid::NUM_POS + grid::NUM_VEL) : grid::NUM_VEL;
    auto dims = var.dimensions();
    if (dims.size() != 2 || (int)dims[1] != N)
        return ffi::Error::InvalidArgument("quadratic_cost: var must be 2D (B, N)");
    int batch = (int)dims[0];
    if (batch < 1) return ffi::Error::InvalidArgument("quadratic_cost: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("quadratic_cost: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(des, "quadratic_cost: des", N, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(w, "quadratic_cost: w", N, batch);
    cudaMemcpyAsync(g_plant.d_in_a, var.typed_data(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, des.typed_data(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, w.typed_data(),   (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    if (STATE) {
        // mjx (MUJOCO=true) is STATE-only: the state cost reframes (input cost stays pin).
        grid_plant::quadratic_state_cost_kernel<T, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
        GRID_RBD_FFI_CHECK_LAUNCH("quadratic_state_cost_kernel");
    } else {
        grid_plant::quadratic_input_cost_kernel<T><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
        GRID_RBD_FFI_CHECK_LAUNCH("quadratic_input_cost_kernel");
    }
    cudaMemcpyAsync(out->typed_data(),  g_plant.d_out,  (size_t)batch * sizeof(T),         cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad->typed_data(), g_plant.d_grad, (size_t)batch * N * sizeof(T),     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess->typed_data(), g_plant.d_hess, (size_t)batch * N * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

static ffi::Error grid_rbd_jax_plant_quadratic_state_cost_impl(
    cudaStream_t stream, ffi::Buffer<GRID_FFI_T> var, ffi::Buffer<GRID_FFI_T> des,
    ffi::Buffer<GRID_FFI_T> w, ffi::ResultBuffer<GRID_FFI_T> out,
    ffi::ResultBuffer<GRID_FFI_T> grad, ffi::ResultBuffer<GRID_FFI_T> hess, int64_t ctx_id) {
    return grid_rbd_jax_plant_quadratic_cost_impl<true>(stream, var, des, w, out, grad, hess, ctx_id);
}
static ffi::Error grid_rbd_jax_plant_quadratic_input_cost_impl(
    cudaStream_t stream, ffi::Buffer<GRID_FFI_T> var, ffi::Buffer<GRID_FFI_T> des,
    ffi::Buffer<GRID_FFI_T> w, ffi::ResultBuffer<GRID_FFI_T> out,
    ffi::ResultBuffer<GRID_FFI_T> grad, ffi::ResultBuffer<GRID_FFI_T> hess, int64_t ctx_id) {
    return grid_rbd_jax_plant_quadratic_cost_impl<false>(stream, var, des, w, out, grad, hess, ctx_id);
}

#define GRID_RBD_JAX_PLANT_COST_BIND(name, impl)                                  \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(name, impl,                                      \
        ffi::Ffi::Bind()                                                          \
            .Ctx<ffi::PlatformStream<cudaStream_t>>()                            \
            .Arg<ffi::Buffer<GRID_FFI_T>>().Arg<ffi::Buffer<GRID_FFI_T>>().Arg<ffi::Buffer<GRID_FFI_T>>() \
            .Ret<ffi::Buffer<GRID_FFI_T>>().Ret<ffi::Buffer<GRID_FFI_T>>().Ret<ffi::Buffer<GRID_FFI_T>>().Attr<int64_t>("ctx_id"))

GRID_RBD_JAX_PLANT_COST_BIND(grid_rbd_jax_plant_quadratic_state_cost,
                             grid_rbd_jax_plant_quadratic_state_cost_impl);
GRID_RBD_JAX_PLANT_COST_BIND(grid_rbd_jax_plant_quadratic_input_cost,
                             grid_rbd_jax_plant_quadratic_input_cost_impl);
#ifdef GRID_RBD_WITH_MUJOCO
// MuJoCo-convention state cost (floating only): value invariant, grad covector-rotated,
// GN hess congruence (MUJOCO_OUTPUT=true). Input cost has no mjx variant (frame-invariant).
static ffi::Error grid_rbd_jax_plant_quadratic_state_cost_mujoco_impl(
    cudaStream_t stream, ffi::Buffer<GRID_FFI_T> var, ffi::Buffer<GRID_FFI_T> des,
    ffi::Buffer<GRID_FFI_T> w, ffi::ResultBuffer<GRID_FFI_T> out,
    ffi::ResultBuffer<GRID_FFI_T> grad, ffi::ResultBuffer<GRID_FFI_T> hess) {
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
    ffi::Buffer<GRID_FFI_T> var, ffi::Buffer<GRID_FFI_T> lower, ffi::Buffer<GRID_FFI_T> upper,
    ffi::ResultBuffer<GRID_FFI_T> out, ffi::ResultBuffer<GRID_FFI_T> grad,
    ffi::ResultBuffer<GRID_FFI_T> hess_diag, float mu,
    int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    if (plant_alloc(g_ctx)) return ffi::Error::Internal("plant_alloc failed");
    const int N = (WHICH == 0) ? grid::NUM_POS : grid::NUM_VEL;
    auto dims = var.dimensions();
    if (dims.size() != 2 || (int)dims[1] != N)
        return ffi::Error::InvalidArgument("barrier: var must be 2D (B, N)");
    int batch = (int)dims[0];
    if (batch < 1) return ffi::Error::InvalidArgument("barrier: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("barrier: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(lower, "barrier: lower", N, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(upper, "barrier: upper", N, batch);
    cudaMemcpyAsync(g_plant.d_in_a, var.typed_data(),   (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, lower.typed_data(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, upper.typed_data(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    if (WHICH == 0)
        grid_plant::joint_position_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    else if (WHICH == 1)
        grid_plant::joint_velocity_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    else
        grid_plant::joint_torque_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("joint_barrier_kernel");
    cudaMemcpyAsync(out->typed_data(),       g_plant.d_out,  (size_t)batch * sizeof(T),     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad->typed_data(),      g_plant.d_grad, (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess_diag->typed_data(), g_plant.d_hess, (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

static ffi::Error grid_rbd_jax_plant_joint_position_barrier_impl(
    cudaStream_t s, ffi::Buffer<GRID_FFI_T> v, ffi::Buffer<GRID_FFI_T> lo, ffi::Buffer<GRID_FFI_T> hi,
    ffi::ResultBuffer<GRID_FFI_T> o, ffi::ResultBuffer<GRID_FFI_T> g, ffi::ResultBuffer<GRID_FFI_T> h, float mu, int64_t ctx_id) {
    return grid_rbd_jax_plant_barrier_impl<0>(s, v, lo, hi, o, g, h, mu, ctx_id);
}
static ffi::Error grid_rbd_jax_plant_joint_velocity_barrier_impl(
    cudaStream_t s, ffi::Buffer<GRID_FFI_T> v, ffi::Buffer<GRID_FFI_T> lo, ffi::Buffer<GRID_FFI_T> hi,
    ffi::ResultBuffer<GRID_FFI_T> o, ffi::ResultBuffer<GRID_FFI_T> g, ffi::ResultBuffer<GRID_FFI_T> h, float mu, int64_t ctx_id) {
    return grid_rbd_jax_plant_barrier_impl<1>(s, v, lo, hi, o, g, h, mu, ctx_id);
}
static ffi::Error grid_rbd_jax_plant_joint_torque_barrier_impl(
    cudaStream_t s, ffi::Buffer<GRID_FFI_T> v, ffi::Buffer<GRID_FFI_T> lo, ffi::Buffer<GRID_FFI_T> hi,
    ffi::ResultBuffer<GRID_FFI_T> o, ffi::ResultBuffer<GRID_FFI_T> g, ffi::ResultBuffer<GRID_FFI_T> h, float mu, int64_t ctx_id) {
    return grid_rbd_jax_plant_barrier_impl<2>(s, v, lo, hi, o, g, h, mu, ctx_id);
}

#define GRID_RBD_JAX_PLANT_BARRIER_BIND(name, impl)                               \
    XLA_FFI_DEFINE_HANDLER_SYMBOL(name, impl,                                      \
        ffi::Ffi::Bind()                                                          \
            .Ctx<ffi::PlatformStream<cudaStream_t>>()                            \
            .Arg<ffi::Buffer<GRID_FFI_T>>().Arg<ffi::Buffer<GRID_FFI_T>>().Arg<ffi::Buffer<GRID_FFI_T>>() \
            .Ret<ffi::Buffer<GRID_FFI_T>>().Ret<ffi::Buffer<GRID_FFI_T>>().Ret<ffi::Buffer<GRID_FFI_T>>() \
            .Attr<float>("mu").Attr<int64_t>("ctx_id"))

GRID_RBD_JAX_PLANT_BARRIER_BIND(grid_rbd_jax_plant_joint_position_barrier,
                                grid_rbd_jax_plant_joint_position_barrier_impl);
GRID_RBD_JAX_PLANT_BARRIER_BIND(grid_rbd_jax_plant_joint_velocity_barrier,
                                grid_rbd_jax_plant_joint_velocity_barrier_impl);
GRID_RBD_JAX_PLANT_BARRIER_BIND(grid_rbd_jax_plant_joint_torque_barrier,
                                grid_rbd_jax_plant_joint_torque_barrier_impl);

// ── BEGIN GENERATED JAX PLANT TAIL (grid_codegen/wrapper_plant_gen.py — do not hand-edit) ──
#ifdef GRID_PLANT_HAS_STEP
// plant_step(x, u; dt, it) → x_kp1  (B, NX). Reuses g_plant.d_grad as x_kp1
// (size NX), matching the C-ABI launch_plant_step.
template <grid::IntegratorType IT, bool MUJOCO>
static void launch_plant_step_jax(GridCtx *ctx, cudaStream_t stream, int batch, T gravity, T dt) {
    GRID_RBD_CTX_LOCALS(ctx);
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    const size_t smem = grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_kernel<T, IT, MUJOCO>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_kernel<T, IT, MUJOCO>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::plant_step_kernel<T, IT, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr,
        smem, stream>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, grid::NUM_VEL, g_robot, (T)gravity, (T)dt, batch);
}

// MUJOCO=true launches the plant_step kernel with MUJOCO_OUTPUT=true (floating only).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_plant_step_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> x, ffi::Buffer<GRID_FFI_T> u,
    ffi::ResultBuffer<GRID_FFI_T> x_kp1,
    T dt, int64_t it, T gravity, int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    if (plant_alloc(g_ctx)) return ffi::Error::Internal("plant_alloc failed");
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    auto dims = x.dimensions();
    if (dims.size() != 2 || (int)dims[1] != nx)
        return ffi::Error::InvalidArgument("plant_step: x must be 2D (B, NX)");
    int batch = (int)dims[0];
    if (batch < 1) return ffi::Error::InvalidArgument("plant_step: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("plant_step: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(u, "plant_step: u", nv, batch);
    cudaMemcpyAsync(g_plant.d_in_a, x.typed_data(), (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, u.typed_data(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    GRID_RBD_IT_DISPATCH_FFI_MJX((int)it, launch_plant_step_jax, MUJOCO, stream, batch, gravity, dt);
    GRID_RBD_FFI_CHECK_LAUNCH("plant_step_kernel");
    cudaMemcpyAsync(x_kp1->typed_data(), g_plant.d_grad, (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_2IN_DT_IT_GRAV(grid_rbd_jax_plant_step, grid_rbd_jax_plant_step_impl<false>);
#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_BIND_2IN_DT_IT_GRAV(grid_rbd_jax_plant_step_mujoco, grid_rbd_jax_plant_step_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_PLANT_HAS_STEP

#ifdef GRID_PLANT_HAS_STEP_GRADIENT
// plant_step_gradient(x, u; dt, it) → dAB  (B, 2*NV*3*NV col-major). Reuses
// g_plant.d_grad as the dAB output (size 2*NV*3*NV), matching the C-ABI.
template <grid::IntegratorType IT, bool MUJOCO>
static void launch_plant_step_gradient_jax(GridCtx *ctx, cudaStream_t stream, int batch, T gravity, T dt) {
    GRID_RBD_CTX_LOCALS(ctx);
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    const size_t smem = grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_gradient_kernel<T, IT, MUJOCO>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_gradient_kernel<T, IT, MUJOCO>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::plant_step_gradient_kernel<T, IT, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr,
        smem, stream>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, nv, g_robot, (T)gravity, (T)dt, batch);
}

// MUJOCO=true launches the plant_step_gradient kernel with MUJOCO_OUTPUT=true (floating only).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_plant_step_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> x, ffi::Buffer<GRID_FFI_T> u,
    ffi::ResultBuffer<GRID_FFI_T> dAB,
    T dt, int64_t it, T gravity, int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    if (plant_alloc(g_ctx)) return ffi::Error::Internal("plant_alloc failed");
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    const int dab = 2 * nv * 3 * nv;
    auto dims = x.dimensions();
    if (dims.size() != 2 || (int)dims[1] != nx)
        return ffi::Error::InvalidArgument("plant_step_gradient: x must be 2D (B, NX)");
    int batch = (int)dims[0];
    if (batch < 1) return ffi::Error::InvalidArgument("plant_step_gradient: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("plant_step_gradient: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(u, "plant_step_gradient: u", nv, batch);
    cudaMemcpyAsync(g_plant.d_in_a, x.typed_data(), (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, u.typed_data(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    // mjx gradient is single-stage (euler/si) only; pin supports all integrator types.
    if constexpr (MUJOCO) {
        GRID_RBD_IT_DISPATCH_FFI_MJX_SS((int)it, launch_plant_step_gradient_jax, true, stream, batch, gravity, dt);
    } else {
        GRID_RBD_IT_DISPATCH_FFI_MJX((int)it, launch_plant_step_gradient_jax, false, stream, batch, gravity, dt);
    }
    GRID_RBD_FFI_CHECK_LAUNCH("plant_step_gradient_kernel");
    cudaMemcpyAsync(dAB->typed_data(), g_plant.d_grad, (size_t)batch * dab * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_BIND_2IN_DT_IT_GRAV(grid_rbd_jax_plant_step_gradient, grid_rbd_jax_plant_step_gradient_impl<false>);
#ifdef GRID_RBD_WITH_MUJOCO
GRID_RBD_JAX_BIND_2IN_DT_IT_GRAV(grid_rbd_jax_plant_step_gradient_mujoco, grid_rbd_jax_plant_step_gradient_impl<true>);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_PLANT_HAS_STEP_GRADIENT

#ifdef GRID_PLANT_HAS_EE_COST
// ee_pos_cost(q, p_des, W) → (value (B,1), grad (B,NX), hess (B,NX*NX)).
// MUJOCO=true launches with MUJOCO_OUTPUT=true (floating only).
template <bool MUJOCO>
static ffi::Error grid_rbd_jax_plant_ee_pos_cost_impl(
    cudaStream_t stream,
    ffi::Buffer<GRID_FFI_T> q, ffi::Buffer<GRID_FFI_T> p_des, ffi::Buffer<GRID_FFI_T> W,
    ffi::ResultBuffer<GRID_FFI_T> out, ffi::ResultBuffer<GRID_FFI_T> grad,
    ffi::ResultBuffer<GRID_FFI_T> hess, int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    if (plant_alloc(g_ctx)) return ffi::Error::Internal("plant_alloc failed");
    const int nq = grid::NUM_POS;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    auto dims = q.dimensions();
    if (dims.size() != 2 || (int)dims[1] != nq)
        return ffi::Error::InvalidArgument("ee_pos_cost: q must be 2D (B, NQ)");
    int batch = (int)dims[0];
    if (batch < 1) return ffi::Error::InvalidArgument("ee_pos_cost: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("ee_pos_cost: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(p_des, "ee_pos_cost: p_des", 3, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(W, "ee_pos_cost: W", 3, batch);
    cudaMemcpyAsync(g_plant.d_in_a, q.typed_data(),     (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, p_des.typed_data(), (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, W.typed_data(),     (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    size_t smem = grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::ee_pos_cost_kernel<T, 0, MUJOCO>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::ee_pos_cost_kernel<T, /*EE=*/0, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose, g_plant.d_end_effector_pose_gradient, g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("ee_pos_cost");
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
    ffi::Buffer<GRID_FFI_T> q, ffi::Buffer<GRID_FFI_T> p_des, ffi::Buffer<GRID_FFI_T> W,
    ffi::ResultBuffer<GRID_FFI_T> out, ffi::ResultBuffer<GRID_FFI_T> grad,
    ffi::ResultBuffer<GRID_FFI_T> hess, int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    if (plant_alloc(g_ctx)) return ffi::Error::Internal("plant_alloc failed");
    const int nq = grid::NUM_POS;
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    auto dims = q.dimensions();
    if (dims.size() != 2 || (int)dims[1] != nq)
        return ffi::Error::InvalidArgument("com_cost: q must be 2D (B, NQ)");
    int batch = (int)dims[0];
    if (batch < 1) return ffi::Error::InvalidArgument("com_cost: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("com_cost: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(p_des, "com_cost: p_des", 3, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(W, "com_cost: W", 3, batch);
    cudaMemcpyAsync(g_plant.d_in_a, q.typed_data(),     (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, p_des.typed_data(), (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, W.typed_data(),     (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    size_t smem = grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::com_cost_kernel<T, MUJOCO>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::com_cost_kernel<T, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("com_cost");
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
    ffi::Buffer<GRID_FFI_T> q, ffi::Buffer<GRID_FFI_T> qd,
    ffi::Buffer<GRID_FFI_T> h_des, ffi::Buffer<GRID_FFI_T> W,
    ffi::ResultBuffer<GRID_FFI_T> out, ffi::ResultBuffer<GRID_FFI_T> grad,
    ffi::ResultBuffer<GRID_FFI_T> hess, int64_t ctx_id)
{
    GRID_RBD_CTX_OR_FFI(ctx_id);
    if (plant_alloc(g_ctx)) return ffi::Error::Internal("plant_alloc failed");
    const int nq = grid::NUM_POS;
    const int nv = grid::NUM_VEL;
    const int nx = nq + nv;
    auto dims = q.dimensions();
    if (dims.size() != 2 || (int)dims[1] != nq)
        return ffi::Error::InvalidArgument("momentum_cost: q must be 2D (B, NQ)");
    int batch = (int)dims[0];
    if (batch < 1) return ffi::Error::InvalidArgument("momentum_cost: batch must be >= 1");
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("momentum_cost: batch > max_batch");
    GRID_RBD_FFI_VALIDATE_ROWS(qd, "momentum_cost: qd", nv, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(h_des, "momentum_cost: h_des", 6, batch);
    GRID_RBD_FFI_VALIDATE_ROWS(W, "momentum_cost: W", 6, batch);
    cudaMemcpyAsync(g_plant.d_in_a, q.typed_data(),  (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, qd.typed_data(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c,                  h_des.typed_data(), (size_t)batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c + (size_t)batch * 6, W.typed_data(),  (size_t)batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    size_t smem = grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    // momentum_cost is register-heavy (~140 regs/thread): clamp to its launch cap.
    dim3 thr = grid_clamp_threads_for(grid_plant::momentum_cost_kernel<T, MUJOCO>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::momentum_cost_kernel<T, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b,
        g_plant.d_in_c, g_plant.d_in_c + (size_t)batch * 6,
        g_robot, batch);
    GRID_RBD_FFI_CHECK_LAUNCH("momentum_cost_kernel");
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
        .Arg<ffi::Buffer<GRID_FFI_T>>().Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>().Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>().Ret<ffi::Buffer<GRID_FFI_T>>().Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("ctx_id")
);
#ifdef GRID_RBD_WITH_MUJOCO
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_plant_momentum_cost_mujoco,
    grid_rbd_jax_plant_momentum_cost_impl<true>,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>().Arg<ffi::Buffer<GRID_FFI_T>>()
        .Arg<ffi::Buffer<GRID_FFI_T>>().Arg<ffi::Buffer<GRID_FFI_T>>()
        .Ret<ffi::Buffer<GRID_FFI_T>>().Ret<ffi::Buffer<GRID_FFI_T>>().Ret<ffi::Buffer<GRID_FFI_T>>()
        .Attr<int64_t>("ctx_id")
);
#endif  // GRID_RBD_WITH_MUJOCO
#endif  // GRID_PLANT_HAS_MOMENTUM_COST
// ── END GENERATED JAX PLANT TAIL ──

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

// fp64 (Wave 2a): the torch surface follows the .so's element type T — tensor dtype
// checks/allocs use GRID_TORCH_DTYPE and pointers are data_ptr<T>.
#if defined(GRID_RBD_WITH_TORCH)

// The tensor dtype tracks the .so's T (fp64 Wave 2a).
#ifdef GRID_WRAPPER_T_DOUBLE
#define GRID_TORCH_DTYPE torch::kFloat64
#define GRID_TORCH_DTYPE_NAME "float64"
#else
#define GRID_TORCH_DTYPE torch::kFloat32
#define GRID_TORCH_DTYPE_NAME "float32"
#endif

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
    TORCH_CHECK(t.scalar_type() == GRID_TORCH_DTYPE, name, ": must be " GRID_TORCH_DTYPE_NAME " (the robot .so dtype)");
    TORCH_CHECK(t.dim() == 2, name, ": must be 2D (B, ", last_dim, ")");
    TORCH_CHECK(t.size(1) == last_dim, name, ": last dim != ", last_dim);
}

// Post-launch check, torch surface. Captures the error ONCE (cudaGetLastError
// clears it) so the cudaError NAME reaches the exception (see the FFI twin).
static inline void grid_torch_check_launch(const char* ksym) {
    cudaError_t le = cudaGetLastError();
    TORCH_CHECK(le == cudaSuccess, ksym, " launch failed: ", cudaGetErrorName(le));
}

// B2 stamps (K4): `stamp_out` (forward-role ops) receives this admission's model
// version; `stamp_expect` (gradient-role ops) is compared with it — see the
// GridCtx note. Both are optional trailing args, so every existing call site and
// captured graph is unchanged.
static inline void grid_torch_stamp_validate(const torch::Tensor& s, const char* name) {
    TORCH_CHECK(s.is_cuda() && s.scalar_type() == torch::kInt32 && s.numel() >= 1,
                name, ": must be a CUDA int32 tensor with at least one element");
}
static inline void grid_torch_stamp_write(GridCtx *ctx, cudaStream_t stream, const c10::optional<torch::Tensor>& stamp_out) {
    if (!stamp_out.has_value()) return;
    grid_torch_stamp_validate(*stamp_out, "stamp_out");
    grid_rbd_stamp_write(ctx, stream, stamp_out->data_ptr<int32_t>());
}
static inline void grid_torch_stamp_check(GridCtx *ctx, cudaStream_t stream, const c10::optional<torch::Tensor>& stamp_expect) {
    if (!stamp_expect.has_value()) return;
    grid_torch_stamp_validate(*stamp_expect, "stamp_expect");
    int seen = 0;
    int rc = grid_rbd_stamp_check(ctx, stream, stamp_expect->data_ptr<int32_t>(), &seen);
    TORCH_CHECK(rc != 15, "grid_rbd: ", grid_ctx_stamp_message(seen, ctx));
    TORCH_CHECK(rc == 0, "grid_rbd: stamp check failed: ", cudaGetErrorName((cudaError_t)rc));
}

static inline int grid_torch_batch(const torch::Tensor& q) {
    int batch = (int)q.size(0);
    TORCH_CHECK(batch >= 1, "batch must be >= 1 (got ", batch, ")");
    TORCH_CHECK(batch <= kMaxBatch, "batch ", batch, " > compiled-in max_batch ", kMaxBatch);
    return batch;
}
// W03: every operand after the leading one must carry the leading batch (the
// D2D copies are sized by it).
static inline void grid_torch_check_rows(const torch::Tensor& t, int batch, const char* name) {
    TORCH_CHECK(t.size(0) == batch, name, ": batch ", t.size(0), " must equal the leading operand's batch ", batch);
}

static inline torch::Tensor grid_torch_empty(int rows, int cols, const torch::Tensor& like) {
    auto opts = torch::TensorOptions().dtype(GRID_TORCH_DTYPE).device(like.device());
    return torch::empty({rows, cols}, opts);
}

// pack q[, qd[, u]] D->D into d_q_qd_u on `stream` (layout [q,qd,u] per ts).
static inline void grid_torch_pack(GridCtx *ctx, cudaStream_t stream, int batch, int nj,
                                   const torch::Tensor* q,
                                   const torch::Tensor* qd,
                                   const torch::Tensor* u) {
    GRID_RBD_CTX_LOCALS(ctx);
    if (q)  grid_torch_check_rows(*q,  batch, "q");     // W03: q's batch sizes every copy
    if (qd) grid_torch_check_rows(*qd, batch, "qd");
    if (u)  grid_torch_check_rows(*u,  batch, "u");
    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    if (q)  cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],      dst_pitch, q->data_ptr<T>(),  row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    if (qd) cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj],     dst_pitch, qd->data_ptr<T>(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    if (u)  cudaMemcpy2DAsync(&g_data->d_q_qd_u[2*nj],   dst_pitch, u->data_ptr<T>(),  row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
}


// Optional external-force application (stream-ordered, graph-capturable).
// f_ext (if present) is (batch, 6*NUM_BODIES) float32 CUDA, body-major,
// [angular; linear] local-frame — same layout as d_f_ext and the numpy
// surface. Copies D->D into the singleton's d_f_ext on `stream`; pair with
// grid_torch_f_ext_reset(g_ctx) AFTER the kernel launch (also on `stream`) so a
// later no-f_ext call sees the zeroed buffer. A null/absent f_ext is a no-op,
// keeping the no-f_ext path byte-identical and capture-clean.
static inline void grid_torch_f_ext_apply(GridCtx *ctx, cudaStream_t stream, int batch,
                                          const c10::optional<torch::Tensor>& f_ext) {
    GRID_RBD_CTX_LOCALS(ctx);
    if (!f_ext.has_value()) return;
    const torch::Tensor& fe = f_ext.value();
    const int row = 6 * grid::NUM_BODIES;
    grid_torch_check(fe, "f_ext", row);
    // The copy is sized by the STATE batch (audit W03, 2026-09-19): a (1, 6*NB)
    // force for a batch of 8 must be rejected here, not read 7 rows past its end.
    TORCH_CHECK(fe.size(0) == batch, "f_ext: batch ", fe.size(0), " must equal the q batch ", batch);
    cudaMemcpyAsync(g_data->d_f_ext, fe.data_ptr<T>(),
                    (size_t)batch * row * sizeof(T), cudaMemcpyDeviceToDevice, stream);
}

static inline void grid_torch_f_ext_reset(GridCtx *ctx, cudaStream_t stream, int batch,
                                          const c10::optional<torch::Tensor>& f_ext) {
    GRID_RBD_CTX_LOCALS(ctx);
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
                         c10::optional<torch::Tensor> f_ext, int64_t ctx_id,
                         c10::optional<torch::Tensor> stamp_out) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "inverse_dynamics: q", nj); grid_torch_check(qd, "inverse_dynamics: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, nullptr);
    grid_torch_f_ext_apply(g_ctx, stream, batch, f_ext);
    auto out = grid_torch_empty(batch, nj, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    if (qdd.has_value()) {
        const torch::Tensor& a = qdd.value();
        grid_torch_check(a, "inverse_dynamics: qdd", nj);
        grid_torch_check_rows(a, batch, "inverse_dynamics: qdd");
        cudaMemcpyAsync(g_data->d_qdd, a.data_ptr<T>(),
                        (size_t)batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        grid::inverse_dynamics_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS>(g_ctx, batch), grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_c, g_data->d_q_qd_u, stride, g_data->d_qdd, g_data->d_f_ext, g_robot, (T)gravity, batch);
        grid_torch_check_launch("inverse_dynamics_kernel");
    } else if constexpr (MUJOCO) {
        // The bias (no-qdd) kernel overload is pinocchio-only (MUJOCO_OUTPUT lives on
        // the qdd-input overload). For mjx with no qdd, zero d_qdd and use the
        // MUJOCO-capable qdd overload (the kernel converts the mjx qacc=0 input to the
        // correct pin acceleration) — matching the JAX handler, which always passes qdd.
        cudaMemsetAsync(g_data->d_qdd, 0, (size_t)batch * nj * sizeof(T), stream);
        grid::inverse_dynamics_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER, /*MUJOCO_OUTPUT=*/true><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS>(g_ctx, batch), grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_c, g_data->d_q_qd_u, stride, g_data->d_qdd, g_data->d_f_ext, g_robot, (T)gravity, batch);
        grid_torch_check_launch("inverse_dynamics_kernel");
    } else {
        // A3: the bias overload is TIER-templated with a default — instantiate it at
        // the LAUNCH tier so __launch_bounds__ matches the autotuned thread count
        // (default-tier under a LITE-tuned count is the idsva_so launch-failure class).
        grid::inverse_dynamics_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS>(g_ctx, batch), grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_c, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
        grid_torch_check_launch("inverse_dynamics_kernel");
    }
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_c, batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(g_ctx, stream, batch, f_ext);
    grid_torch_stamp_write(g_ctx, stream, stamp_out);
    return out;
}
#endif  // GRID_HAS_INVERSE_DYNAMICS

// ── BEGIN GENERATED TORCH OP BODIES (grid_codegen/wrapper_body_gen.py — do not hand-edit) ──
// Regenerate: .venv/bin/python -m grid_codegen.wrapper_body_gen
// Table: grid_codegen/abi_specs.py (kernel_args et al.); docs verbatim
// from grid_codegen/wrapper_surface_docs.py.

#if GRID_HAS_MINV
template <bool MUJOCO>
torch::Tensor torch_minv(torch::Tensor q, int64_t ctx_id,
                         c10::optional<torch::Tensor> stamp_expect) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "minv: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_stamp_check(g_ctx, stream, stamp_expect);
    grid_torch_pack(g_ctx, stream, batch, nj, &q, nullptr, nullptr);
    // Minv is nv x nv (tangent-space); the kernel writes d_Minv nv*nv-strided.
    // Size the output + copy at nv*nv (unified with numpy/JAX). FIXED base: nv == nj.
    auto out = grid_torch_empty(batch, nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::minv_kernel<T, grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_MINV>(g_ctx, batch), grid::MINV_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER>(), stream>>>(
        g_data->d_Minv, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    grid_torch_check_launch("minv_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_Minv, batch * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_MINV

#if GRID_HAS_FORWARD_DYNAMICS
template <bool MUJOCO>
torch::Tensor torch_forward_dynamics(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double gravity, c10::optional<torch::Tensor> f_ext, int64_t ctx_id,
                                     c10::optional<torch::Tensor> stamp_out) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "forward_dynamics: q", nj);
    grid_torch_check(qd, "forward_dynamics: qd", nj);
    grid_torch_check(u, "forward_dynamics: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, &u);
    grid_torch_f_ext_apply(g_ctx, stream, batch, f_ext);
    auto out = grid_torch_empty(batch, nj, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::forward_dynamics_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_FORWARD_DYNAMICS>(g_ctx, batch), grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER>(), stream>>>(
        g_data->d_qdd, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    grid_torch_check_launch("forward_dynamics_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_qdd, batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(g_ctx, stream, batch, f_ext);
    grid_torch_stamp_write(g_ctx, stream, stamp_out);
    return out;
}
#endif  // GRID_HAS_FORWARD_DYNAMICS

#if GRID_HAS_ABA
template <bool MUJOCO>
torch::Tensor torch_aba(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double gravity, c10::optional<torch::Tensor> f_ext, int64_t ctx_id,
                        c10::optional<torch::Tensor> stamp_out) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "aba: q", nj); grid_torch_check(qd, "aba: qd", nj); grid_torch_check(u, "aba: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, &u);
    grid_torch_f_ext_apply(g_ctx, stream, batch, f_ext);
    auto out = grid_torch_empty(batch, nj, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::aba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_ABA>(g_ctx, batch), grid::ABA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_ABA>::TIER>(), stream>>>(
        g_data->d_qdd, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    grid_torch_check_launch("aba_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_qdd, batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(g_ctx, stream, batch, f_ext);
    grid_torch_stamp_write(g_ctx, stream, stamp_out);
    return out;
}
#endif  // GRID_HAS_ABA

#if GRID_HAS_CRBA
template <bool MUJOCO>
torch::Tensor torch_crba(torch::Tensor q, double gravity, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "crba: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, nullptr, nullptr);
    // M is nv x nv (tangent-space); the kernel writes d_M nv*nv-strided. Size the
    // output + copy at nv*nv (unified with numpy/JAX). FIXED base: nv == nj.
    auto out = grid_torch_empty(batch, nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::crba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_CRBA>(g_ctx, batch), grid::CRBA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER>(), stream>>>(
        g_data->d_M, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    grid_torch_check_launch("crba_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_M, batch * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_CRBA

#if GRID_HAS_END_EFFECTOR_POSE
template <bool MUJOCO>
torch::Tensor torch_end_effector_pose(torch::Tensor q, int64_t ctx_id,
                                      c10::optional<torch::Tensor> stamp_out) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nee = GRID_RBD_NUM_EES;
    grid_torch_check(q, "end_effector_pose: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 6 * nee, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::GRID_RBD_EE_POSE_KERNEL<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_END_EFFECTOR_POSE>(g_ctx, batch), grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_end_effector_pose, g_data->d_q_qd_u, stride, g_robot, batch);
    grid_torch_check_launch("GRID_RBD_EE_POSE_KERNEL");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_end_effector_pose, batch * 6 * nee * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_stamp_write(g_ctx, stream, stamp_out);
    return out;
}
#endif  // GRID_HAS_END_EFFECTOR_POSE

#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
template <bool MUJOCO>
torch::Tensor torch_end_effector_pose_gradient(torch::Tensor q, int64_t ctx_id,
                                               c10::optional<torch::Tensor> stamp_expect) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL, nee = GRID_RBD_NUM_EES;
    grid_torch_check(q, "end_effector_pose_gradient: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_stamp_check(g_ctx, stream, stamp_expect);
    grid_torch_pack(g_ctx, stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 6 * nee * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::GRID_RBD_EE_POSE_GRADIENT_KERNEL<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>(g_ctx, batch), grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_GRADIENT>::TIER>(), stream>>>(
        g_data->d_end_effector_pose_gradient, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    grid_torch_check_launch("GRID_RBD_EE_POSE_GRADIENT_KERNEL");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_end_effector_pose_gradient, batch * 6 * nee * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_END_EFFECTOR_POSE_GRADIENT

#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN
template <bool MUJOCO>
torch::Tensor torch_end_effector_pose_hessian(torch::Tensor q, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL, nee = GRID_RBD_NUM_EES;
    grid_torch_check(q, "end_effector_pose_hessian: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 6 * nee * nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::GRID_RBD_EE_POSE_HESSIAN_KERNEL<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>(g_ctx, batch), grid::END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE_HESSIAN>::TIER>(), stream>>>(
        g_data->d_end_effector_pose_hessian, g_data->d_end_effector_pose_gradient, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    grid_torch_check_launch("GRID_RBD_EE_POSE_HESSIAN_KERNEL");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_end_effector_pose_hessian, batch * 6 * nee * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_END_EFFECTOR_POSE_HESSIAN

#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
template <bool MUJOCO>
torch::Tensor torch_forward_dynamics_gradient(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double gravity, c10::optional<torch::Tensor> f_ext, int64_t ctx_id,
                                              c10::optional<torch::Tensor> stamp_expect) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "forward_dynamics_gradient: q", nj);
    grid_torch_check(qd, "forward_dynamics_gradient: qd", nj);
    grid_torch_check(u, "forward_dynamics_gradient: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_stamp_check(g_ctx, stream, stamp_expect);
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, &u);
    grid_torch_f_ext_apply(g_ctx, stream, batch, f_ext);
    // df_du is nv x 2nv (tangent-space); the kernel writes d_df_du 2*nv*nv-strided.
    // Size + copy at 2*nv*nv (unified with numpy/JAX). FIXED base: nv == nj.
    auto out = grid_torch_empty(batch, 2 * nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::forward_dynamics_gradient_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>(g_ctx, batch), grid::FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER>(), stream>>>(
        g_data->d_df_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    grid_torch_check_launch("forward_dynamics_gradient_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_df_du, batch * 2 * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(g_ctx, stream, batch, f_ext);
    return out;
}
#endif  // GRID_HAS_FORWARD_DYNAMICS_GRADIENT

#if GRID_HAS_FDSVA_SO
template <bool MUJOCO>
torch::Tensor torch_fdsva_so(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double gravity, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "fdsva_so: q", nj); grid_torch_check(qd, "fdsva_so: qd", nj); grid_torch_check(u, "fdsva_so: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, &u);
    auto out = grid_torch_empty(batch, grid::SECOND_ORDER_TENSOR_SIZE, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::fdsva_so_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_FDSVA_SO>(g_ctx, batch), grid::FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER>(), stream>>>(
        g_data->d_df2, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_idsva_so, g_robot, (T)gravity, batch);
    grid_torch_check_launch("fdsva_so_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_df2, batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T), cudaMemcpyDeviceToDevice, stream);
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
torch::Tensor torch_inverse_dynamics_regressor(torch::Tensor q, torch::Tensor qd, torch::Tensor qdd, double gravity, int64_t ctx_id,
                                               c10::optional<torch::Tensor> stamp_expect) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "inverse_dynamics_regressor: q", nj);
    grid_torch_check(qd, "inverse_dynamics_regressor: qd", nj);
    grid_torch_check(qdd, "inverse_dynamics_regressor: qdd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_stamp_check(g_ctx, stream, stamp_expect);
    // qdd occupies the u-slot (read as the acceleration; mirrors the JAX handler).
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, &qdd);
    const int out_size = grid::NUM_VEL * 10 * grid::NUM_BODIES;
    auto out = grid_torch_empty(batch, out_size, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::inverse_dynamics_regressor_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_REGRESSOR>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS_REGRESSOR>(g_ctx, batch), grid::INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_REGRESSOR>::TIER>(), stream>>>(
        g_data->d_Y, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    grid_torch_check_launch("inverse_dynamics_regressor_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_Y, (size_t)batch * out_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_INVERSE_DYNAMICS_REGRESSOR

#if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
torch::Tensor torch_forward_dynamics_parameter_gradient(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double gravity, int64_t ctx_id,
                                                        c10::optional<torch::Tensor> stamp_expect) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "forward_dynamics_parameter_gradient: q", nj);
    grid_torch_check(qd, "forward_dynamics_parameter_gradient: qd", nj);
    grid_torch_check(u, "forward_dynamics_parameter_gradient: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_stamp_check(g_ctx, stream, stamp_expect);
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, &u);
    const int out_size = grid::NUM_VEL * 10 * grid::NUM_BODIES;
    auto out = grid_torch_empty(batch, out_size, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::forward_dynamics_parameter_gradient_kernel<T><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_FORWARD_DYNAMICS_PARAMETER_GRADIENT>(g_ctx, batch), grid::FORWARD_DYNAMICS_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_PARAMETER_GRADIENT>::TIER>(), stream>>>(
        g_data->d_dqdd_dpi, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    grid_torch_check_launch("forward_dynamics_parameter_gradient_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_dqdd_dpi, (size_t)batch * out_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT

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
torch::Tensor torch_generalized_gravity(torch::Tensor q, double gravity, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "generalized_gravity: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::generalized_gravity_kernel<T, grid::launch_cfg<grid::GRID_ALGO_GENERALIZED_GRAVITY>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_GENERALIZED_GRAVITY>(g_ctx, batch), grid::INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_c, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    grid_torch_check_launch("generalized_gravity_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_c, batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_GENERALIZED_GRAVITY

#if GRID_HAS_NONLINEAR_EFFECTS
template <bool MUJOCO>
torch::Tensor torch_nonlinear_effects(torch::Tensor q, torch::Tensor qd, double gravity, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "nonlinear_effects: q", nj); grid_torch_check(qd, "nonlinear_effects: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::nonlinear_effects_kernel<T, grid::launch_cfg<grid::GRID_ALGO_NONLINEAR_EFFECTS>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_NONLINEAR_EFFECTS>(g_ctx, batch), grid::INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_c, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    grid_torch_check_launch("nonlinear_effects_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_c, batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_NONLINEAR_EFFECTS

#if GRID_HAS_CORIOLIS_MATRIX
template <bool MUJOCO>
torch::Tensor torch_coriolis_matrix(torch::Tensor q, torch::Tensor qd, double gravity, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "coriolis_matrix: q", nj); grid_torch_check(qd, "coriolis_matrix: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::coriolis_matrix_kernel<T, grid::launch_cfg<grid::GRID_ALGO_CORIOLIS_MATRIX>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_CORIOLIS_MATRIX>(g_ctx, batch), grid::CORIOLIS_MATRIX_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_CORIOLIS_MATRIX>::TIER>(), stream>>>(
        g_data->d_coriolis, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    grid_torch_check_launch("coriolis_matrix_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_coriolis, batch * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_CORIOLIS_MATRIX

#if GRID_HAS_KINETIC_ENERGY_REGRESSOR
template <bool MUJOCO>
torch::Tensor torch_kinetic_energy_regressor(torch::Tensor q, torch::Tensor qd, double gravity, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nb = grid::NUM_BODIES;
    grid_torch_check(q, "kinetic_energy_regressor: q", nj); grid_torch_check(qd, "kinetic_energy_regressor: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, 10 * nb, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::kinetic_energy_regressor_kernel<T, grid::launch_cfg<grid::GRID_ALGO_KINETIC_ENERGY_REGRESSOR>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_KINETIC_ENERGY_REGRESSOR>(g_ctx, batch), grid::KINETIC_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_ke_regressor, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    grid_torch_check_launch("kinetic_energy_regressor_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_ke_regressor, batch * 10 * nb * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_KINETIC_ENERGY_REGRESSOR

#if GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
template <bool MUJOCO>
torch::Tensor torch_potential_energy_regressor(torch::Tensor q, double gravity, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nb = grid::NUM_BODIES;
    grid_torch_check(q, "potential_energy_regressor: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 10 * nb, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::potential_energy_regressor_kernel<T, grid::launch_cfg<grid::GRID_ALGO_POTENTIAL_ENERGY_REGRESSOR>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_POTENTIAL_ENERGY_REGRESSOR>(g_ctx, batch), grid::POTENTIAL_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_pe_regressor, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    grid_torch_check_launch("potential_energy_regressor_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_pe_regressor, batch * 10 * nb * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_POTENTIAL_ENERGY_REGRESSOR

#ifdef GRID_HAS_ENERGY
template <bool MUJOCO>
torch::Tensor torch_energy(torch::Tensor q, torch::Tensor qd, double gravity, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "energy: q", nj); grid_torch_check(qd, "energy: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, 3, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::energy_kernel<T, grid::launch_cfg<grid::GRID_ALGO_ENERGY>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_ENERGY>(g_ctx, batch), grid::ENERGY_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_ENERGY>::TIER>(), stream>>>(
        g_data->d_energy, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    grid_torch_check_launch("energy_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_energy, batch * 3 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_ENERGY

#ifdef GRID_HAS_COM
// com → single flat (B, 3 + 3*NV); the Python layer splits the (p_com, J_com) tuple.
template <bool MUJOCO>
torch::Tensor torch_com(torch::Tensor q, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "com: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 3 + 3 * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::com_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COM>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_COM>(g_ctx, batch), grid::COM_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_COM>::TIER>(), stream>>>(
        g_data->d_com, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    grid_torch_check_launch("com_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_com, batch * (3 + 3 * nv) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_COM

#ifdef GRID_HAS_CCRBA
// ccrba → single flat (B, 6*NV + 6); the Python layer splits the (A, h) tuple.
template <bool MUJOCO>
torch::Tensor torch_ccrba(torch::Tensor q, torch::Tensor qd, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "ccrba: q", nj); grid_torch_check(qd, "ccrba: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, 6 * nv + 6, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::ccrba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_CCRBA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_CCRBA>(g_ctx, batch), grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_CCRBA>::TIER>(), stream>>>(
        g_data->d_ccrba, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    grid_torch_check_launch("ccrba_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_ccrba, batch * (6 * nv + 6) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_CCRBA

#ifdef GRID_HAS_CMM_TIME_VARIATION
template <bool MUJOCO>
torch::Tensor torch_cmm_time_variation(torch::Tensor q, torch::Tensor qd, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "cmm_time_variation: q", nj); grid_torch_check(qd, "cmm_time_variation: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, 6 * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::cmm_time_variation_kernel<T, grid::launch_cfg<grid::GRID_ALGO_CMM_TIME_VARIATION>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_CMM_TIME_VARIATION>(g_ctx, batch), grid::CMM_TIME_VARIATION_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_CMM_TIME_VARIATION>::TIER>(), stream>>>(
        g_data->d_cmm_time_variation, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    grid_torch_check_launch("cmm_time_variation_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_cmm_time_variation, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_CMM_TIME_VARIATION

#ifdef GRID_HAS_DCCRBA
template <bool MUJOCO>
torch::Tensor torch_dccrba(torch::Tensor q, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "dccrba: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 6 * nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::dccrba_kernel<T, grid::launch_cfg<grid::GRID_ALGO_DCCRBA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_DCCRBA>(g_ctx, batch), grid::DCCRBA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_DCCRBA>::TIER>(), stream>>>(
        g_data->d_dccrba, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    grid_torch_check_launch("dccrba_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_dccrba, batch * 6 * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_DCCRBA

#ifdef GRID_HAS_FRAME_JACOBIAN
// frame_jacobian / frame_jacobian_dot: target_jid + reference_frame trail the
// schema as ints, passed straight through to the kernel (NO host -1 default
// resolution — the Python torch surface passes resolved non-negative values).
template <bool MUJOCO>
torch::Tensor torch_frame_jacobian(torch::Tensor q, int64_t target_jid, int64_t reference_frame, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "frame_jacobian: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 6 * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::frame_jacobian_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FRAME_JACOBIAN>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_FRAME_JACOBIAN>(g_ctx, batch), grid::FRAME_JACOBIAN_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_frame_jacobian, g_data->d_q_qd_u, stride, (int)target_jid, (int)reference_frame, g_robot, batch);
    grid_torch_check_launch("frame_jacobian_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_frame_jacobian, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_FRAME_JACOBIAN

#if defined(GRID_HAS_FRAME_JACOBIAN) && defined(GRID_HAS_FRAME_JACOBIAN_DOT)
template <bool MUJOCO>
torch::Tensor torch_frame_jacobian_dot(torch::Tensor q, torch::Tensor qd, int64_t target_jid, int64_t reference_frame, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "frame_jacobian_dot: q", nj); grid_torch_check(qd, "frame_jacobian_dot: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, 6 * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::frame_jacobian_dot_kernel<T, grid::launch_cfg<grid::GRID_ALGO_FRAME_JACOBIAN_DOT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_FRAME_JACOBIAN_DOT>(g_ctx, batch), grid::FRAME_JACOBIAN_DOT_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_frame_jacobian_dot, g_data->d_q_qd_u, stride, (int)target_jid, (int)reference_frame, g_robot, batch);
    grid_torch_check_launch("frame_jacobian_dot_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_frame_jacobian_dot, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_FRAME_JACOBIAN && GRID_HAS_FRAME_JACOBIAN_DOT

#if defined(GRID_HAS_FRAME_JACOBIAN) && defined(GRID_HAS_OSC_INERTIA)
template <bool MUJOCO>
torch::Tensor torch_osc_inertia(torch::Tensor q, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "osc_inertia: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 36, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::osc_inertia_kernel<T, grid::launch_cfg<grid::GRID_ALGO_OSC_INERTIA>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_OSC_INERTIA>(g_ctx, batch), grid::OSC_INERTIA_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_OSC_INERTIA>::TIER>(), stream>>>(
        g_data->d_osc_inertia, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    grid_torch_check_launch("osc_inertia_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_osc_inertia, batch * 36 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_FRAME_JACOBIAN && GRID_HAS_OSC_INERTIA

// ── END GENERATED TORCH OP BODIES ──

// ─── runtime-target multi-EE pose / pose-gradient (single-target torch ops) ──
// SINGLE target jid + a SINGLE 16-float col-major 4x4 SE(3) tool transform per call
// (the Python wrapper loops the resolved jid list + stacks, mirroring _handle.py).
// The transform is a 16-element CUDA float Tensor; its data_ptr is ALREADY a
// contiguous device pointer to 16 T, so it is handed straight to the kernel's
// d_Xtool (no host staging). target_jid is an absolute jid (the Python layer
// resolves names→jids). Gated like the C-ABI wrappers.
#ifdef GRID_HAS_END_EFFECTOR_POSE_RUNTIME
template <bool MUJOCO>
torch::Tensor torch_end_effector_pose_runtime(torch::Tensor q, int64_t target_jid, torch::Tensor offset, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    TORCH_CHECK(target_jid >= 0 && target_jid < grid::NUM_JOINTS, "end_effector_pose_runtime: target_jid out of range");
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "end_effector_pose_runtime: q", nj);
    TORCH_CHECK(offset.is_cuda() && offset.numel() == 16,
                "end_effector_pose_runtime: offset must be the 16-element CUDA tensor "
                "holding the col-major 4x4 SE(3) tool transform");
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, nullptr, nullptr);
    auto off = offset.to(GRID_TORCH_DTYPE).contiguous();
    auto out = grid_torch_empty(batch, 6, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::end_effector_pose_runtime_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), grid::END_EFFECTOR_POSE_RUNTIME_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_eePose, g_data->d_q_qd_u, stride, (int)target_jid, off.data_ptr<T>(), g_robot, batch);
    grid_torch_check_launch("end_effector_pose_runtime_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_eePose, batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_END_EFFECTOR_POSE_RUNTIME

#ifdef GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME
template <bool MUJOCO>
torch::Tensor torch_end_effector_pose_gradient_runtime(torch::Tensor q, int64_t target_jid, torch::Tensor offset, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    TORCH_CHECK(target_jid >= 0 && target_jid < grid::NUM_JOINTS, "end_effector_pose_gradient_runtime: target_jid out of range");
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "end_effector_pose_gradient_runtime: q", nj);
    TORCH_CHECK(offset.is_cuda() && offset.numel() == 16,
                "end_effector_pose_gradient_runtime: offset must be the 16-element CUDA tensor "
                "holding the col-major 4x4 SE(3) tool transform");
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, nullptr, nullptr);
    auto off = offset.to(GRID_TORCH_DTYPE).contiguous();
    auto out = grid_torch_empty(batch, 6 * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::end_effector_pose_gradient_runtime_kernel<T, grid::launch_cfg<grid::GRID_ALGO_COUNT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), grid::END_EFFECTOR_POSE_GRADIENT_RUNTIME_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_eePoseGrad, g_data->d_q_qd_u, stride, (int)target_jid, off.data_ptr<T>(), g_robot, batch);
    grid_torch_check_launch("end_effector_pose_gradient_runtime_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_eePoseGrad, batch * 6 * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
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
                              c10::optional<torch::Tensor> f_ext, int64_t ctx_id,
                              c10::optional<torch::Tensor> stamp_expect) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_stamp_check(g_ctx, stream, stamp_expect);
    const int nj = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    grid_torch_check(q, "inverse_dynamics_gradient: q", nj); grid_torch_check(qd, "inverse_dynamics_gradient: qd", nj);
    int batch = grid_torch_batch(q);
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, nullptr);
    grid_torch_f_ext_apply(g_ctx, stream, batch, f_ext);
    // dc_du is nv x 2nv (tangent-space); the kernel writes d_dc_du 2*nv*nv-strided.
    // Size + copy at 2*nv*nv (unified with numpy/JAX). FIXED base: nv == nj.
    auto out = grid_torch_empty(batch, 2 * nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    if (qdd.has_value()) {
        const torch::Tensor& a = qdd.value();
        grid_torch_check(a, "inverse_dynamics_gradient: qdd", nj);
        cudaMemcpyAsync(g_data->d_qdd, a.data_ptr<T>(),
                        (size_t)batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        grid::inverse_dynamics_gradient_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(g_ctx, batch), grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(), stream>>>(
            g_data->d_dc_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_qdd, g_data->d_f_ext, g_robot, (T)gravity, batch);
        grid_torch_check_launch("inverse_dynamics_gradient_kernel");
    } else if constexpr (MUJOCO) {
        // mjx: the bias (no-qdd) gradient overload is pinocchio-only; zero d_qdd and
        // use the MUJOCO-capable qdd overload (matches the JAX handler, which always
        // passes qdd).
        cudaMemsetAsync(g_data->d_qdd, 0, (size_t)batch * nj * sizeof(T), stream);
        grid::inverse_dynamics_gradient_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/true><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(g_ctx, batch), grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(), stream>>>(
            g_data->d_dc_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_qdd, g_data->d_f_ext, g_robot, (T)gravity, batch);
        grid_torch_check_launch("inverse_dynamics_gradient_kernel");
    } else {
        // A3: launch-tier instantiation + tier-aware smem bytes (the <T>()/<T> default
        // pair under a TIER-tuned thread count under-sizes smem on divergent-tier
        // robots and trips __launch_bounds__ — the idsva_so launch-failure class).
        grid::inverse_dynamics_gradient_kernel<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>(g_ctx, batch), grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER>(), stream>>>(
            g_data->d_dc_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
        grid_torch_check_launch("inverse_dynamics_gradient_kernel");
    }
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_dc_du, batch * 2 * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(g_ctx, stream, batch, f_ext);
    return out;
}
#endif  // GRID_HAS_INVERSE_DYNAMICS_GRADIENT

#if GRID_HAS_IDSVA_SO
template <bool MUJOCO>
torch::Tensor torch_idsva_so(torch::Tensor q, torch::Tensor qd, torch::Tensor qdd, double gravity, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "idsva_so: q", nj); grid_torch_check(qd, "idsva_so: qd", nj);
    grid_torch_check(qdd, "idsva_so: qdd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // qdd is packed into the acceleration (u) slot, read by the kernel as s_qdd
    // (mirrors numpy pack_q_qd_u(g_ctx, q, qd, qdd)). The Python surface passes explicit
    // zeros for the default so we never read a stale device buffer.
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, &qdd);
    auto out = grid_torch_empty(batch, grid::SECOND_ORDER_TENSOR_SIZE, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    // Compile-time frame dispatch — mirrors the JAX handler / grid::idsva_so:
    // GRID_IDSVA_SO_DISPATCHES_WORLD_FRAME (world for floating / spherical /
    // high-DOF fixed), NOT the mjx-twins gate (a floating MIMIC robot has no
    // GRID_RBD_WITH_MUJOCO yet dispatches world; agent guide §1m). The
    // MUJOCO_OUTPUT param exists only where GRID_RBD_SIG_MJX_IDSVA_SO says so.
#if GRID_IDSVA_SO_DISPATCHES_WORLD_FRAME
#ifdef GRID_RBD_SIG_MJX_IDSVA_SO
    grid::idsva_so_world_frame_kernel<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>(g_ctx, batch), grid::IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>::TIER>(), stream>>>(
        g_data->d_idsva_so, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
#else
    static_assert(!MUJOCO, "mjx idsva_so is floating-base only");
    grid::idsva_so_world_frame_kernel<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>::TIER><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>(g_ctx, batch), grid::IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_WORLD_FRAME>::TIER>(), stream>>>(
        g_data->d_idsva_so, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
#endif
    grid_torch_check_launch("idsva_so_world_frame_kernel");
#else
    static_assert(!MUJOCO, "mjx idsva_so is floating-base only");
    grid::idsva_so_body_frame_kernel<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_BODY_FRAME>::TIER><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_IDSVA_SO_BODY_FRAME>(g_ctx, batch), grid::IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO_BODY_FRAME>::TIER>(), stream>>>(
        g_data->d_idsva_so, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    grid_torch_check_launch("idsva_so_body_frame_kernel");
#endif
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_idsva_so, batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_HAS_IDSVA_SO

// mjx-aware torch integrator-type dispatch: instantiates FN<IT_case, MUJOCO_FLAG>.
#define GRID_RBD_IT_DISPATCH_TORCH_MJX(it_code, FN, MUJOCO_FLAG, ...)                          \
    GRID_RBD_IT_SWITCH_MJX(it_code, FN, MUJOCO_FLAG,                                            \
        TORCH_CHECK(false, "integrator: bad integrator-type code"), __VA_ARGS__)

// Single-stage-only mjx dispatch (gradient): Euler / Semi-Implicit-Euler only.
#define GRID_RBD_IT_DISPATCH_TORCH_MJX_SS(it_code, FN, MUJOCO_FLAG, ...)                       \
    GRID_RBD_IT_SWITCH_MJX_SS(it_code, FN, MUJOCO_FLAG,                                         \
        TORCH_CHECK(false,                                                                      \
            "mujoco integrator/plant-step gradient supports only euler / semi-implicit-euler"), \
        __VA_ARGS__)

#if GRID_HAS_INTEGRATOR
template <grid::IntegratorType IT, bool MUJOCO>
static void torch_launch_integrator(GridCtx *ctx, cudaStream_t stream, int batch, double dt, double gravity) {
    GRID_RBD_CTX_LOCALS(ctx);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::integrator_kernel<T, IT, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INTEGRATOR>(g_ctx, batch), grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR>::TIER>(), stream>>>(
        g_data->d_x_kp1, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, g_data->d_f_ext, (T)gravity, (T)dt, batch);
}
#endif  // GRID_HAS_INTEGRATOR
#if GRID_HAS_INTEGRATOR_GRADIENT
template <grid::IntegratorType IT, bool MUJOCO>
static void torch_launch_integrator_grad(GridCtx *ctx, cudaStream_t stream, int batch, double dt, double gravity) {
    GRID_RBD_CTX_LOCALS(ctx);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::integrator_gradient_kernel<T, IT, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR_GRADIENT>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::GRID_ALGO_INTEGRATOR_GRADIENT>(g_ctx, batch), grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T, grid::launch_cfg<grid::GRID_ALGO_INTEGRATOR_GRADIENT>::TIER>(), stream>>>(
        g_data->d_dAB, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, g_data->d_f_ext, (T)gravity, (T)dt, batch);
}
#endif  // GRID_HAS_INTEGRATOR_GRADIENT

#if GRID_HAS_INTEGRATOR
template <bool MUJOCO>
torch::Tensor torch_integrator(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double dt, int64_t it, double gravity, int64_t ctx_id,
                               c10::optional<torch::Tensor> stamp_out) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "integrator: q", nj); grid_torch_check(qd, "integrator: qd", nj); grid_torch_check(u, "integrator: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, &u);
    auto out = grid_torch_empty(batch, grid::NUM_POS + grid::NUM_VEL, q);
    GRID_RBD_IT_DISPATCH_TORCH_MJX((int)it, torch_launch_integrator, MUJOCO, stream, batch, dt, gravity);
    grid_torch_check_launch("integrator_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_x_kp1, batch * (grid::NUM_POS + grid::NUM_VEL) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_stamp_write(g_ctx, stream, stamp_out);
    return out;
}
#endif  // GRID_HAS_INTEGRATOR

#if GRID_HAS_INTEGRATOR_GRADIENT
template <bool MUJOCO>
torch::Tensor torch_integrator_gradient(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double dt, int64_t it, double gravity, int64_t ctx_id,
                                        c10::optional<torch::Tensor> stamp_expect) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_stamp_check(g_ctx, stream, stamp_expect);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "integrator_gradient: q", nj); grid_torch_check(qd, "integrator_gradient: qd", nj); grid_torch_check(u, "integrator_gradient: u", nj);
    int batch = grid_torch_batch(q);
    grid_torch_pack(g_ctx, stream, batch, nj, &q, &qd, &u);
    auto out = grid_torch_empty(batch, 2 * nv * 3 * nv, q);
    // mjx gradient is single-stage (euler/si) only; pin supports all integrator types.
    if constexpr (MUJOCO) {
        GRID_RBD_IT_DISPATCH_TORCH_MJX_SS((int)it, torch_launch_integrator_grad, true, stream, batch, dt, gravity);
    } else {
        GRID_RBD_IT_DISPATCH_TORCH_MJX((int)it, torch_launch_integrator_grad, false, stream, batch, dt, gravity);
    }
    grid_torch_check_launch("integrator_grad_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_data->d_dAB, batch * (2 * nv) * (3 * nv) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
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

static inline void grid_torch_plant_init(GridCtx *ctx) {
    GRID_RBD_CTX_LOCALS(ctx);
    TORCH_CHECK(plant_alloc(g_ctx) == 0, "plant_alloc failed");
}

// q/qd-style check for a (B, N) plant input with an arbitrary last dim.
static inline void grid_torch_check_n(const torch::Tensor& t, const char* name, int n) {
    TORCH_CHECK(t.is_cuda(), name, ": must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, ": must be contiguous");
    TORCH_CHECK(t.scalar_type() == GRID_TORCH_DTYPE, name, ": must be " GRID_TORCH_DTYPE_NAME " (the robot .so dtype)");
    TORCH_CHECK(t.dim() == 2, name, ": must be 2D (B, ", n, ")");
    TORCH_CHECK(t.size(1) == n, name, ": last dim != ", n);
}

// quadratic cost (state or input). var/des/w are (B, N). Returns (value, grad, hess).
// STATE=true selects the state kernel (N = NX) else the input kernel (N = NV).
// Non-template on `state` (runtime bool) so the torch::Tensor .data_ptr<T>()
// member-template calls parse unambiguously under nvcc's host pass; MUJOCO is a
// compile-time template flag threaded into the STATE kernel launch (the only mjx
// path — input cost is frame-invariant and always pinocchio).
template <bool MUJOCO>
static std::vector<torch::Tensor> torch_plant_quadratic_cost(
    torch::Tensor var, torch::Tensor des, torch::Tensor w, bool state, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    grid_torch_plant_init(g_ctx);
    const int N = state ? (grid::NUM_POS + grid::NUM_VEL) : grid::NUM_VEL;
    grid_torch_check_n(var, "quadratic_cost: var", N);
    grid_torch_check_n(des, "quadratic_cost: des", N);
    grid_torch_check_n(w,   "quadratic_cost: w",   N);
    int batch = grid_torch_batch(var);
    grid_torch_check_rows(des, batch, "quadratic_cost: des");
    grid_torch_check_rows(w,   batch, "quadratic_cost: w");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, var.data_ptr<T>(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, des.data_ptr<T>(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, w.data_ptr<T>(),   (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out  = grid_torch_empty(batch, 1, var);
    auto grad = grid_torch_empty(batch, N, var);
    auto hess = grid_torch_empty(batch, N * N, var);
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    if (state)
        grid_plant::quadratic_state_cost_kernel<T, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
    else
        grid_plant::quadratic_input_cost_kernel<T><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
    grid_torch_check_launch("quadratic_cost_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(),  g_plant.d_out,  (size_t)batch * sizeof(T),         cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<T>(), g_plant.d_grad, (size_t)batch * N * sizeof(T),     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess.data_ptr<T>(), g_plant.d_hess, (size_t)batch * N * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hess};
}

template <bool MUJOCO>
std::vector<torch::Tensor> torch_quadratic_state_cost(torch::Tensor x, torch::Tensor x_des, torch::Tensor Q, int64_t ctx_id) {
    return torch_plant_quadratic_cost<MUJOCO>(x, x_des, Q, true, ctx_id);
}
std::vector<torch::Tensor> torch_quadratic_input_cost(torch::Tensor u, torch::Tensor u_des, torch::Tensor R, int64_t ctx_id) {
    return torch_plant_quadratic_cost<false>(u, u_des, R, false, ctx_id);
}

// barrier (position/velocity/torque). var/lower/upper are (B, N). Returns
// (value, grad, hess_diag). which: 0=position (N=NUM_POS), 1=velocity, 2=torque.
static std::vector<torch::Tensor> torch_plant_barrier(
    torch::Tensor var, torch::Tensor lower, torch::Tensor upper, double mu, int which, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    grid_torch_plant_init(g_ctx);
    const int N = (which == 0) ? grid::NUM_POS : grid::NUM_VEL;
    grid_torch_check_n(var,   "barrier: var",   N);
    grid_torch_check_n(lower, "barrier: lower", N);
    grid_torch_check_n(upper, "barrier: upper", N);
    int batch = grid_torch_batch(var);
    grid_torch_check_rows(lower, batch, "barrier: lower");
    grid_torch_check_rows(upper, batch, "barrier: upper");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, var.data_ptr<T>(),   (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, lower.data_ptr<T>(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, upper.data_ptr<T>(), (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out  = grid_torch_empty(batch, 1, var);
    auto grad = grid_torch_empty(batch, N, var);
    auto hdiag = grid_torch_empty(batch, N, var);
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    if (which == 0)
        grid_plant::joint_position_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    else if (which == 1)
        grid_plant::joint_velocity_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    else
        grid_plant::joint_torque_barrier_kernel<T><<<grid_dim, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch), 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    grid_torch_check_launch("joint_barrier_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(),   g_plant.d_out,  (size_t)batch * sizeof(T),     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<T>(),  g_plant.d_grad, (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hdiag.data_ptr<T>(), g_plant.d_hess, (size_t)batch * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hdiag};
}

std::vector<torch::Tensor> torch_joint_position_barrier(torch::Tensor v, torch::Tensor lo, torch::Tensor hi, double mu, int64_t ctx_id) {
    return torch_plant_barrier(v, lo, hi, mu, 0, ctx_id);
}
std::vector<torch::Tensor> torch_joint_velocity_barrier(torch::Tensor v, torch::Tensor lo, torch::Tensor hi, double mu, int64_t ctx_id) {
    return torch_plant_barrier(v, lo, hi, mu, 1, ctx_id);
}
std::vector<torch::Tensor> torch_joint_torque_barrier(torch::Tensor v, torch::Tensor lo, torch::Tensor hi, double mu, int64_t ctx_id) {
    return torch_plant_barrier(v, lo, hi, mu, 2, ctx_id);
}

// ── BEGIN GENERATED TORCH PLANT TAIL (grid_codegen/wrapper_plant_gen.py — do not hand-edit) ──
#ifdef GRID_PLANT_HAS_STEP
template <grid::IntegratorType IT, bool MUJOCO>
static void torch_launch_plant_step(GridCtx *ctx, cudaStream_t stream, int batch, double gravity, double dt) {
    GRID_RBD_CTX_LOCALS(ctx);
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    const size_t smem = grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_kernel<T, IT, MUJOCO>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_kernel<T, IT, MUJOCO>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::plant_step_kernel<T, IT, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr,
        smem, stream>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, grid::NUM_VEL, g_robot, (T)gravity, (T)dt, batch);
}

template <bool MUJOCO>
torch::Tensor torch_plant_step(torch::Tensor x, torch::Tensor u, double dt, int64_t it, double gravity, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    grid_torch_plant_init(g_ctx);
    const int nx = grid::NUM_POS + grid::NUM_VEL, nv = grid::NUM_VEL;
    grid_torch_check_n(x, "plant_step: x", nx);
    grid_torch_check_n(u, "plant_step: u", nv);
    int batch = grid_torch_batch(x);
    grid_torch_check_rows(u, batch, "plant_step: u");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, x.data_ptr<T>(), (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, u.data_ptr<T>(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out = grid_torch_empty(batch, nx, x);
    GRID_RBD_IT_DISPATCH_TORCH_MJX((int)it, torch_launch_plant_step, MUJOCO, stream, batch, gravity, dt);
    grid_torch_check_launch("plant_step_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_plant.d_grad, (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_PLANT_HAS_STEP

#ifdef GRID_PLANT_HAS_STEP_GRADIENT
template <grid::IntegratorType IT, bool MUJOCO>
static void torch_launch_plant_step_gradient(GridCtx *ctx, cudaStream_t stream, int batch, double gravity, double dt) {
    GRID_RBD_CTX_LOCALS(ctx);
    const int nx = grid::NUM_POS + grid::NUM_VEL, nv = grid::NUM_VEL;
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    const size_t smem = grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grid_plant::plant_step_gradient_kernel<T, IT, MUJOCO>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 thr = grid_clamp_threads_for(grid_plant::plant_step_gradient_kernel<T, IT, MUJOCO>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::plant_step_gradient_kernel<T, IT, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr,
        smem, stream>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, nv, g_robot, (T)gravity, (T)dt, batch);
}

template <bool MUJOCO>
torch::Tensor torch_plant_step_gradient(torch::Tensor x, torch::Tensor u, double dt, int64_t it, double gravity, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    grid_torch_plant_init(g_ctx);
    const int nx = grid::NUM_POS + grid::NUM_VEL, nv = grid::NUM_VEL;
    const int dab = 2 * nv * 3 * nv;
    grid_torch_check_n(x, "plant_step_gradient: x", nx);
    grid_torch_check_n(u, "plant_step_gradient: u", nv);
    int batch = grid_torch_batch(x);
    grid_torch_check_rows(u, batch, "plant_step_gradient: u");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, x.data_ptr<T>(), (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, u.data_ptr<T>(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out = grid_torch_empty(batch, dab, x);
    // mjx gradient is single-stage (euler/si) only; pin supports all integrator types.
    if constexpr (MUJOCO) {
        GRID_RBD_IT_DISPATCH_TORCH_MJX_SS((int)it, torch_launch_plant_step_gradient, true, stream, batch, gravity, dt);
    } else {
        GRID_RBD_IT_DISPATCH_TORCH_MJX((int)it, torch_launch_plant_step_gradient, false, stream, batch, gravity, dt);
    }
    grid_torch_check_launch("plant_step_gradient_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(), g_plant.d_grad, (size_t)batch * dab * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_PLANT_HAS_STEP_GRADIENT

#ifdef GRID_PLANT_HAS_EE_COST
template <bool MUJOCO>
std::vector<torch::Tensor> torch_ee_pos_cost(torch::Tensor q, torch::Tensor p_des, torch::Tensor W, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    grid_torch_plant_init(g_ctx);
    const int nq = grid::NUM_POS, nx = grid::NUM_POS + grid::NUM_VEL;
    grid_torch_check_n(q, "ee_pos_cost: q", nq);
    grid_torch_check_n(p_des, "ee_pos_cost: p_des", 3);
    grid_torch_check_n(W, "ee_pos_cost: W", 3);
    int batch = grid_torch_batch(q);
    grid_torch_check_rows(p_des, batch, "ee_pos_cost: p_des");
    grid_torch_check_rows(W, batch, "ee_pos_cost: W");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, q.data_ptr<T>(),     (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, p_des.data_ptr<T>(), (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, W.data_ptr<T>(),     (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out  = grid_torch_empty(batch, 1, q);
    auto grad = grid_torch_empty(batch, nx, q);
    auto hess = grid_torch_empty(batch, nx * nx, q);
    size_t smem = grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::ee_pos_cost_kernel<T, 0, MUJOCO>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::ee_pos_cost_kernel<T, 0, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose, g_plant.d_end_effector_pose_gradient, g_robot, batch);
    grid_torch_check_launch("ee_pos_cost");
    cudaMemcpyAsync(out.data_ptr<T>(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<T>(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess.data_ptr<T>(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hess};
}
#endif  // GRID_PLANT_HAS_EE_COST

#ifdef GRID_PLANT_HAS_COM_COST
template <bool MUJOCO>
std::vector<torch::Tensor> torch_com_cost(torch::Tensor q, torch::Tensor p_des, torch::Tensor W, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    grid_torch_plant_init(g_ctx);
    const int nq = grid::NUM_POS, nx = grid::NUM_POS + grid::NUM_VEL;
    grid_torch_check_n(q, "com_cost: q", nq);
    grid_torch_check_n(p_des, "com_cost: p_des", 3);
    grid_torch_check_n(W, "com_cost: W", 3);
    int batch = grid_torch_batch(q);
    grid_torch_check_rows(p_des, batch, "com_cost: p_des");
    grid_torch_check_rows(W, batch, "com_cost: W");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, q.data_ptr<T>(),     (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, p_des.data_ptr<T>(), (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c, W.data_ptr<T>(),     (size_t)batch * 3  * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out  = grid_torch_empty(batch, 1, q);
    auto grad = grid_torch_empty(batch, nx, q);
    auto hess = grid_torch_empty(batch, nx * nx, q);
    size_t smem = grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    dim3 thr = grid_clamp_threads_for(grid_plant::com_cost_kernel<T, MUJOCO>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::com_cost_kernel<T, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_robot, batch);
    grid_torch_check_launch("com_cost");
    cudaMemcpyAsync(out.data_ptr<T>(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<T>(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess.data_ptr<T>(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hess};
}
#endif  // GRID_PLANT_HAS_COM_COST

#ifdef GRID_PLANT_HAS_MOMENTUM_COST
template <bool MUJOCO>
std::vector<torch::Tensor> torch_momentum_cost(torch::Tensor q, torch::Tensor qd, torch::Tensor h_des, torch::Tensor W, int64_t ctx_id) {
    GRID_RBD_CTX_OR_THROW(ctx_id);
    grid_torch_plant_init(g_ctx);
    const int nq = grid::NUM_POS, nv = grid::NUM_VEL, nx = nq + nv;
    grid_torch_check_n(q, "momentum_cost: q", nq);
    grid_torch_check_n(qd, "momentum_cost: qd", nv);
    grid_torch_check_n(h_des, "momentum_cost: h_des", 6);
    grid_torch_check_n(W, "momentum_cost: W", 6);
    int batch = grid_torch_batch(q);
    grid_torch_check_rows(qd, batch, "momentum_cost: qd");
    grid_torch_check_rows(h_des, batch, "momentum_cost: h_des");
    grid_torch_check_rows(W, batch, "momentum_cost: W");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaMemcpyAsync(g_plant.d_in_a, q.data_ptr<T>(),  (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_b, qd.data_ptr<T>(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c,                  h_des.data_ptr<T>(), (size_t)batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(g_plant.d_in_c + (size_t)batch * 6, W.data_ptr<T>(),  (size_t)batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    auto out  = grid_torch_empty(batch, 1, q);
    auto grad = grid_torch_empty(batch, nx, q);
    auto hess = grid_torch_empty(batch, nx * nx, q);
    size_t smem = grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);
    // momentum_cost is register-heavy (~140 regs/thread): clamp to its launch cap.
    dim3 thr = grid_clamp_threads_for(grid_plant::momentum_cost_kernel<T, MUJOCO>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));
    grid_plant::momentum_cost_kernel<T, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b,
        g_plant.d_in_c, g_plant.d_in_c + (size_t)batch * 6, g_robot, batch);
    grid_torch_check_launch("momentum_cost_kernel");
    cudaMemcpyAsync(out.data_ptr<T>(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<T>(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess.data_ptr<T>(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hess};
}
#endif  // GRID_PLANT_HAS_MOMENTUM_COST
// ── END GENERATED TORCH PLANT TAIL ──

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

// ── BEGIN GENERATED TORCH OP TABLE (grid_codegen/wrapper_body_gen.py — do not hand-edit) ──
// Regenerate: .venv/bin/python -m grid_codegen.wrapper_body_gen
// ── def/impl op table (X-macro) ──────────────────────────────────────────────
// One row per op that has a <bool MUJOCO> impl template AND (on floating
// builds) a _mujoco twin — i.e. every torch op except the pin-only stragglers
// def'd/impl'd by hand after each table expansion below
// (forward_dynamics_parameter_gradient, quadratic_input_cost, the three
// barriers). Each row macro expands to X(name, sig) when its algorithm gate is
// on and to NOTHING otherwise, so the def / impl / mjx-def / mjx-impl blocks
// share ONE gate per op by construction (a schema def'd without its impl
// throws a confusing error at call time; gating the def makes the op absent
// instead, matching the numpy rc=3 / missing-symbol subset pattern — this also
// fixes the formerly UNGATED energy/com/ccrba/cmm/dccrba/frame-family/
// ee-runtime schema defs). The gate FORM is load-bearing: core-algo GRID_HAS_*
// macros are always defined (to 1/0) -> #if; opt-in ones are defined-or-absent
// -> #ifdef / defined().
#if GRID_HAS_INVERSE_DYNAMICS
#define GRID_TORCH_ROW_INVERSE_DYNAMICS(X) X(inverse_dynamics, "(Tensor q, Tensor qd, float gravity, Tensor? qdd=None, Tensor? f_ext=None, int ctx_id=0, Tensor? stamp_out=None) -> Tensor")
#else
#define GRID_TORCH_ROW_INVERSE_DYNAMICS(X)
#endif
#if GRID_HAS_MINV
#define GRID_TORCH_ROW_MINV(X) X(minv, "(Tensor q, int ctx_id=0, Tensor? stamp_expect=None) -> Tensor")
#else
#define GRID_TORCH_ROW_MINV(X)
#endif
#if GRID_HAS_FORWARD_DYNAMICS
#define GRID_TORCH_ROW_FORWARD_DYNAMICS(X) X(forward_dynamics, "(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None, int ctx_id=0, Tensor? stamp_out=None) -> Tensor")
#else
#define GRID_TORCH_ROW_FORWARD_DYNAMICS(X)
#endif
#if GRID_HAS_ABA
#define GRID_TORCH_ROW_ABA(X) X(aba, "(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None, int ctx_id=0, Tensor? stamp_out=None) -> Tensor")
#else
#define GRID_TORCH_ROW_ABA(X)
#endif
#if GRID_HAS_CRBA
#define GRID_TORCH_ROW_CRBA(X) X(crba, "(Tensor q, float gravity, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_CRBA(X)
#endif
#if GRID_HAS_END_EFFECTOR_POSE
#define GRID_TORCH_ROW_END_EFFECTOR_POSE(X) X(end_effector_pose, "(Tensor q, int ctx_id=0, Tensor? stamp_out=None) -> Tensor")
#else
#define GRID_TORCH_ROW_END_EFFECTOR_POSE(X)
#endif
#if GRID_HAS_END_EFFECTOR_POSE_GRADIENT
#define GRID_TORCH_ROW_END_EFFECTOR_POSE_GRADIENT(X) X(end_effector_pose_gradient, "(Tensor q, int ctx_id=0, Tensor? stamp_expect=None) -> Tensor")
#else
#define GRID_TORCH_ROW_END_EFFECTOR_POSE_GRADIENT(X)
#endif
#if GRID_HAS_END_EFFECTOR_POSE_HESSIAN
#define GRID_TORCH_ROW_END_EFFECTOR_POSE_HESSIAN(X) X(end_effector_pose_hessian, "(Tensor q, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_END_EFFECTOR_POSE_HESSIAN(X)
#endif
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
#define GRID_TORCH_ROW_INVERSE_DYNAMICS_GRADIENT(X) X(inverse_dynamics_gradient, "(Tensor q, Tensor qd, float gravity, Tensor? qdd=None, Tensor? f_ext=None, int ctx_id=0, Tensor? stamp_expect=None) -> Tensor")
#else
#define GRID_TORCH_ROW_INVERSE_DYNAMICS_GRADIENT(X)
#endif
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
#define GRID_TORCH_ROW_FORWARD_DYNAMICS_GRADIENT(X) X(forward_dynamics_gradient, "(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None, int ctx_id=0, Tensor? stamp_expect=None) -> Tensor")
#else
#define GRID_TORCH_ROW_FORWARD_DYNAMICS_GRADIENT(X)
#endif
#if GRID_HAS_IDSVA_SO
#define GRID_TORCH_ROW_IDSVA_SO(X) X(idsva_so, "(Tensor q, Tensor qd, Tensor qdd, float gravity, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_IDSVA_SO(X)
#endif
#if GRID_HAS_FDSVA_SO
#define GRID_TORCH_ROW_FDSVA_SO(X) X(fdsva_so, "(Tensor q, Tensor qd, Tensor u, float gravity, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_FDSVA_SO(X)
#endif
#if GRID_HAS_INVERSE_DYNAMICS_REGRESSOR
#define GRID_TORCH_ROW_INVERSE_DYNAMICS_REGRESSOR(X) X(inverse_dynamics_regressor, "(Tensor q, Tensor qd, Tensor qdd, float gravity, int ctx_id=0, Tensor? stamp_expect=None) -> Tensor")
#else
#define GRID_TORCH_ROW_INVERSE_DYNAMICS_REGRESSOR(X)
#endif
#if GRID_HAS_INTEGRATOR
#define GRID_TORCH_ROW_INTEGRATOR(X) X(integrator, "(Tensor q, Tensor qd, Tensor u, float dt, int it, float gravity, int ctx_id=0, Tensor? stamp_out=None) -> Tensor")
#else
#define GRID_TORCH_ROW_INTEGRATOR(X)
#endif
#if GRID_HAS_INTEGRATOR_GRADIENT
#define GRID_TORCH_ROW_INTEGRATOR_GRADIENT(X) X(integrator_gradient, "(Tensor q, Tensor qd, Tensor u, float dt, int it, float gravity, int ctx_id=0, Tensor? stamp_expect=None) -> Tensor")
#else
#define GRID_TORCH_ROW_INTEGRATOR_GRADIENT(X)
#endif
#if GRID_HAS_GENERALIZED_GRAVITY
#define GRID_TORCH_ROW_GENERALIZED_GRAVITY(X) X(generalized_gravity, "(Tensor q, float gravity, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_GENERALIZED_GRAVITY(X)
#endif
#if GRID_HAS_NONLINEAR_EFFECTS
#define GRID_TORCH_ROW_NONLINEAR_EFFECTS(X) X(nonlinear_effects, "(Tensor q, Tensor qd, float gravity, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_NONLINEAR_EFFECTS(X)
#endif
#if GRID_HAS_CORIOLIS_MATRIX
#define GRID_TORCH_ROW_CORIOLIS_MATRIX(X) X(coriolis_matrix, "(Tensor q, Tensor qd, float gravity, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_CORIOLIS_MATRIX(X)
#endif
#if GRID_HAS_KINETIC_ENERGY_REGRESSOR
#define GRID_TORCH_ROW_KINETIC_ENERGY_REGRESSOR(X) X(kinetic_energy_regressor, "(Tensor q, Tensor qd, float gravity, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_KINETIC_ENERGY_REGRESSOR(X)
#endif
#if GRID_HAS_POTENTIAL_ENERGY_REGRESSOR
#define GRID_TORCH_ROW_POTENTIAL_ENERGY_REGRESSOR(X) X(potential_energy_regressor, "(Tensor q, float gravity, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_POTENTIAL_ENERGY_REGRESSOR(X)
#endif
#ifdef GRID_HAS_ENERGY
#define GRID_TORCH_ROW_ENERGY(X) X(energy, "(Tensor q, Tensor qd, float gravity, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_ENERGY(X)
#endif
#ifdef GRID_HAS_COM
#define GRID_TORCH_ROW_COM(X) X(com, "(Tensor q, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_COM(X)
#endif
#ifdef GRID_HAS_CCRBA
#define GRID_TORCH_ROW_CCRBA(X) X(ccrba, "(Tensor q, Tensor qd, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_CCRBA(X)
#endif
#ifdef GRID_HAS_CMM_TIME_VARIATION
#define GRID_TORCH_ROW_CMM_TIME_VARIATION(X) X(cmm_time_variation, "(Tensor q, Tensor qd, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_CMM_TIME_VARIATION(X)
#endif
#ifdef GRID_HAS_DCCRBA
#define GRID_TORCH_ROW_DCCRBA(X) X(dccrba, "(Tensor q, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_DCCRBA(X)
#endif
#ifdef GRID_HAS_FRAME_JACOBIAN
#define GRID_TORCH_ROW_FRAME_JACOBIAN(X) X(frame_jacobian, "(Tensor q, int target_jid, int reference_frame, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_FRAME_JACOBIAN(X)
#endif
#if defined(GRID_HAS_FRAME_JACOBIAN) && defined(GRID_HAS_FRAME_JACOBIAN_DOT)
#define GRID_TORCH_ROW_FRAME_JACOBIAN_DOT(X) X(frame_jacobian_dot, "(Tensor q, Tensor qd, int target_jid, int reference_frame, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_FRAME_JACOBIAN_DOT(X)
#endif
#if defined(GRID_HAS_FRAME_JACOBIAN) && defined(GRID_HAS_OSC_INERTIA)
#define GRID_TORCH_ROW_OSC_INERTIA(X) X(osc_inertia, "(Tensor q, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_OSC_INERTIA(X)
#endif
#ifdef GRID_HAS_END_EFFECTOR_POSE_RUNTIME
#define GRID_TORCH_ROW_END_EFFECTOR_POSE_RUNTIME(X) X(end_effector_pose_runtime, "(Tensor q, int target_jid, Tensor offset, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_END_EFFECTOR_POSE_RUNTIME(X)
#endif
#ifdef GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME
#define GRID_TORCH_ROW_END_EFFECTOR_POSE_GRADIENT_RUNTIME(X) X(end_effector_pose_gradient_runtime, "(Tensor q, int target_jid, Tensor offset, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_END_EFFECTOR_POSE_GRADIENT_RUNTIME(X)
#endif
#define GRID_TORCH_ROW_QUADRATIC_STATE_COST(X) X(quadratic_state_cost, "(Tensor x, Tensor x_des, Tensor Q, int ctx_id=0) -> Tensor[]")  // always emitted
#ifdef GRID_PLANT_HAS_STEP
#define GRID_TORCH_ROW_PLANT_STEP(X) X(plant_step, "(Tensor x, Tensor u, float dt, int it, float gravity, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_PLANT_STEP(X)
#endif
#ifdef GRID_PLANT_HAS_STEP_GRADIENT
#define GRID_TORCH_ROW_PLANT_STEP_GRADIENT(X) X(plant_step_gradient, "(Tensor x, Tensor u, float dt, int it, float gravity, int ctx_id=0) -> Tensor")
#else
#define GRID_TORCH_ROW_PLANT_STEP_GRADIENT(X)
#endif
#ifdef GRID_PLANT_HAS_EE_COST
#define GRID_TORCH_ROW_EE_POS_COST(X) X(ee_pos_cost, "(Tensor q, Tensor p_des, Tensor W, int ctx_id=0) -> Tensor[]")
#else
#define GRID_TORCH_ROW_EE_POS_COST(X)
#endif
#ifdef GRID_PLANT_HAS_COM_COST
#define GRID_TORCH_ROW_COM_COST(X) X(com_cost, "(Tensor q, Tensor p_des, Tensor W, int ctx_id=0) -> Tensor[]")
#else
#define GRID_TORCH_ROW_COM_COST(X)
#endif
#ifdef GRID_PLANT_HAS_MOMENTUM_COST
#define GRID_TORCH_ROW_MOMENTUM_COST(X) X(momentum_cost, "(Tensor q, Tensor qd, Tensor h_des, Tensor W, int ctx_id=0) -> Tensor[]")
#else
#define GRID_TORCH_ROW_MOMENTUM_COST(X)
#endif

#define GRID_RBD_TORCH_OPS(X) \
    GRID_TORCH_ROW_INVERSE_DYNAMICS(X) \
    GRID_TORCH_ROW_MINV(X) \
    GRID_TORCH_ROW_FORWARD_DYNAMICS(X) \
    GRID_TORCH_ROW_ABA(X) \
    GRID_TORCH_ROW_CRBA(X) \
    GRID_TORCH_ROW_END_EFFECTOR_POSE(X) \
    GRID_TORCH_ROW_END_EFFECTOR_POSE_GRADIENT(X) \
    GRID_TORCH_ROW_END_EFFECTOR_POSE_HESSIAN(X) \
    GRID_TORCH_ROW_INVERSE_DYNAMICS_GRADIENT(X) \
    GRID_TORCH_ROW_FORWARD_DYNAMICS_GRADIENT(X) \
    GRID_TORCH_ROW_IDSVA_SO(X) \
    GRID_TORCH_ROW_FDSVA_SO(X) \
    GRID_TORCH_ROW_INVERSE_DYNAMICS_REGRESSOR(X) \
    GRID_TORCH_ROW_INTEGRATOR(X) \
    GRID_TORCH_ROW_INTEGRATOR_GRADIENT(X) \
    GRID_TORCH_ROW_GENERALIZED_GRAVITY(X) \
    GRID_TORCH_ROW_NONLINEAR_EFFECTS(X) \
    GRID_TORCH_ROW_CORIOLIS_MATRIX(X) \
    GRID_TORCH_ROW_KINETIC_ENERGY_REGRESSOR(X) \
    GRID_TORCH_ROW_POTENTIAL_ENERGY_REGRESSOR(X) \
    GRID_TORCH_ROW_ENERGY(X) \
    GRID_TORCH_ROW_COM(X) \
    GRID_TORCH_ROW_CCRBA(X) \
    GRID_TORCH_ROW_CMM_TIME_VARIATION(X) \
    GRID_TORCH_ROW_DCCRBA(X) \
    GRID_TORCH_ROW_FRAME_JACOBIAN(X) \
    GRID_TORCH_ROW_FRAME_JACOBIAN_DOT(X) \
    GRID_TORCH_ROW_OSC_INERTIA(X) \
    GRID_TORCH_ROW_END_EFFECTOR_POSE_RUNTIME(X) \
    GRID_TORCH_ROW_END_EFFECTOR_POSE_GRADIENT_RUNTIME(X) \
    GRID_TORCH_ROW_QUADRATIC_STATE_COST(X) \
    GRID_TORCH_ROW_PLANT_STEP(X) \
    GRID_TORCH_ROW_PLANT_STEP_GRADIENT(X) \
    GRID_TORCH_ROW_EE_POS_COST(X) \
    GRID_TORCH_ROW_COM_COST(X) \
    GRID_TORCH_ROW_MOMENTUM_COST(X)
// ── END GENERATED TORCH OP TABLE ──

GRID_RBD_TORCH_LIBRARY(GRID_RBD_TORCH_LIB, m) {
    // Table ops: schema = op name + row signature.
#define GRID_TORCH_X_DEF(name, sig) m.def(#name sig);
    GRID_RBD_TORCH_OPS(GRID_TORCH_X_DEF)
#undef GRID_TORCH_X_DEF
    // pin-only stragglers (no mjx twin / non-template impls):
#if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
    m.def("forward_dynamics_parameter_gradient(Tensor q, Tensor qd, Tensor u, float gravity, int ctx_id=0, Tensor? stamp_expect=None) -> Tensor");
#endif  // GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
    m.def("quadratic_input_cost(Tensor u, Tensor u_des, Tensor R, int ctx_id=0) -> Tensor[]");
    m.def("joint_position_barrier(Tensor var, Tensor lower, Tensor upper, float mu, int ctx_id=0) -> Tensor[]");
    m.def("joint_velocity_barrier(Tensor var, Tensor lower, Tensor upper, float mu, int ctx_id=0) -> Tensor[]");
    m.def("joint_torque_barrier(Tensor var, Tensor lower, Tensor upper, float mu, int ctx_id=0) -> Tensor[]");
#ifdef GRID_RBD_WITH_MUJOCO
    // MuJoCo-convention twins (floating only): same schemas, name + "_mujoco";
    // the CUDA impls launch the kernels with MUJOCO_OUTPUT=true.
#define GRID_TORCH_X_DEF_MJX(name, sig) m.def(#name "_mujoco" sig);
    GRID_RBD_TORCH_OPS(GRID_TORCH_X_DEF_MJX)
#undef GRID_TORCH_X_DEF_MJX
#endif  // GRID_RBD_WITH_MUJOCO
}

GRID_RBD_TORCH_LIBRARY_IMPL(GRID_RBD_TORCH_LIB, CUDA, m) {
#define GRID_TORCH_X_IMPL(name, sig) m.impl(#name, torch_##name<false>);
    GRID_RBD_TORCH_OPS(GRID_TORCH_X_IMPL)
#undef GRID_TORCH_X_IMPL
#if GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
    m.impl("forward_dynamics_parameter_gradient", torch_forward_dynamics_parameter_gradient);
#endif  // GRID_HAS_FORWARD_DYNAMICS_PARAMETER_GRADIENT
    m.impl("quadratic_input_cost", torch_quadratic_input_cost);
    m.impl("joint_position_barrier", torch_joint_position_barrier);
    m.impl("joint_velocity_barrier", torch_joint_velocity_barrier);
    m.impl("joint_torque_barrier", torch_joint_torque_barrier);
#ifdef GRID_RBD_WITH_MUJOCO
#define GRID_TORCH_X_IMPL_MJX(name, sig) m.impl(#name "_mujoco", torch_##name<true>);
    GRID_RBD_TORCH_OPS(GRID_TORCH_X_IMPL_MJX)
#undef GRID_TORCH_X_IMPL_MJX
#endif  // GRID_RBD_WITH_MUJOCO
}

#endif  // GRID_RBD_WITH_TORCH

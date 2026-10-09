// wrapper_common.cuh — host-only declarations shared by every translation unit
// of the per-robot robot.so (multi-TU build, 2026-10-09): the element type, the
// context/plant/launch-override types, the context-admission RAII type and its
// macros, the stamp + launch-shape + input-packing helpers. Everything here is a
// type, a macro, a declaration, or a small `static inline` host helper, so a
// family unit that includes it instantiates NO kernels and defines NO exported
// symbol. The registry, lifecycle, metadata and the per-family registration
// aggregator live in the core unit (wrapper_template.cu); the two core entry
// points a unit needs are declared below (grid_rbd_ctx_acquire,
// grid_ctx_rc_message). Hashed into the cache key with wrapper_template.cu.
#pragma once
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

// ─── mirror retarget for the allocate-once numpy round trip (2026-10-01) ──────
// The generated grid::<op> host wrappers copy their output D2H into the pinned
// g_data->h_<out> mirror; the C ABI used to std::memcpy that into the caller's
// array (a second full-size host copy: 28 ms of a 67 ms g1 idsva_so@1024 call).
// A body that constructs this guard aims the mirror at the caller's buffer for
// the duration of the call, so the wrapper's own D2H lands there directly (at the
// PCIe rate when the buffer came from handle.pinned_empty). The destructor
// restores the mirror on EVERY exit path — launch-check / sync early returns
// included — so an error never leaves the context pointing at caller memory.
struct GridMirrorRetarget {
    T **slot; T *saved;
    GridMirrorRetarget(T **s, T *to) : slot(s), saved(*s) { *s = to; }
    ~GridMirrorRetarget() { *slot = saved; }
    GridMirrorRetarget(const GridMirrorRetarget &) = delete;
    GridMirrorRetarget &operator=(const GridMirrorRetarget &) = delete;
};

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
// codex R4 (2026-09-24): safe for PARTIALLY built storage — every non-null pointer is
// released regardless of the success flag (plant_alloc builds into a temporary and
// publishes only on complete success; a failed build is torn down through here).
static void plant_free(PlantBuffers *p) {
    if (!p) return;
    if (p->d_in_a) cudaFree(p->d_in_a);
    if (p->d_in_b) cudaFree(p->d_in_b);
    if (p->d_in_c) cudaFree(p->d_in_c);
    if (p->d_out) cudaFree(p->d_out);
    if (p->d_grad) cudaFree(p->d_grad);
    if (p->d_hess) cudaFree(p->d_hess);
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

// Defined in the core unit (wrapper_template.cu).
const char *grid_ctx_rc_message(int rc);
GridCtx *grid_rbd_ctx_acquire(long long id, int *rc);
int plant_alloc(GridCtx *ctx);   // lazily allocates ctx->plant staging (core)
struct GridCtxRef {
    GridCtx *ctx = nullptr; int rc = 0; bool exclusive = false;
    // `exclusive` = a MUTATOR: waits for every admitted call to finish and blocks
    // new admissions until it returns. The registry lookup + strong execution
    // reference (inflight) live in core (grid_rbd_ctx_acquire: registry mutex held
    // atomically with the open check); the admission lock is taken AFTER the
    // registry mutex is released (lock order: registry -> admission; a long setter
    // must not stall create/close/lookup of other contexts). close drains inflight
    // to 0 before freeing, so it never frees a context we are waiting on.
    explicit GridCtxRef(long long id, bool excl = false) : exclusive(excl) {
        ctx = grid_rbd_ctx_acquire(id, &rc);
        if (!ctx) return;
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
static __global__ void grid_rbd_stamp_kernel(int *dst, int v) { if (threadIdx.x == 0) *dst = v; }
static inline int grid_ctx_stamp_value(const GridCtx *c) { return (int)(c->version & 0x7fffffffULL); }
static inline cudaError_t grid_rbd_stamp_write(GridCtx *ctx, cudaStream_t stream, int *dst) {
    grid_rbd_stamp_kernel<<<1, 1, 0, stream>>>(dst, grid_ctx_stamp_value(ctx));
    return cudaGetLastError();   // launch-configuration errors surface to the caller
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

#ifndef GRID_RBD_ARCH
#define GRID_RBD_ARCH 0
#endif

// ─── shared input-packing helper ─────────────────────────────────────────────
//
// h_q_qd_u layout (matches generated host wrappers):
//   timestep t: [q[0..NJ-1], qd[0..NJ-1], u[0..NJ-1]]   (three NUM_JOINTS-wide slots)
//   contiguous across t.
//
// Width contract (2026-09-26): the CALLER's q rows are num_joints (NUM_POS) wide and the
// qd/u rows are num_vel (NUM_VEL) wide — the physical tangent width, as in Pinocchio and
// MuJoCo. Every kernel indexes its velocity slot by tangent index, so the padded slot is
// filled leading-NUM_VEL-entries-then-zeros for every model (floating or spherical). The
// NUM_JOINTS-pitched staging rows are an internal layout of this .so.

static inline void pack_q_qd_u(GridCtx *ctx, const T* q, const T* qd, const T* u,
                               int batch, int num_joints, int num_vel)
{
    GRID_RBD_CTX_LOCALS(ctx);
    // Drain in-flight framework work first (audit W04-B class, 2026-09-20): the jax/torch
    // handlers enqueue async copies, kernels and the f_ext reset on THEIR (non-blocking)
    // streams into these same g_data buffers; the numpy path stages synchronously on the
    // legacy default stream, so without this a framework memset could land between our
    // staging and our launch (seen: numpy ID computed against a zeroed d_f_ext).
    cudaDeviceSynchronize();
    const int stride = 3 * num_joints;
    const int pad = num_joints - num_vel;
    for (int t = 0; t < batch; ++t) {
        T* row = &g_data->h_q_qd_u[t * stride];
        std::memcpy(&row[0], &q[t * num_joints], num_joints * sizeof(T));
        std::memcpy(&row[num_joints], &qd[t * num_vel], num_vel * sizeof(T));
        if (pad > 0) std::memset(&row[num_joints + num_vel], 0, pad * sizeof(T));
        if (u) {
            std::memcpy(&row[2 * num_joints], &u[t * num_vel], num_vel * sizeof(T));
            if (pad > 0) std::memset(&row[2 * num_joints + num_vel], 0, pad * sizeof(T));
        }
    }
}

// qdd (NUM_VEL-wide caller rows) into the NUM_JOINTS-pitched h_qdd the generated host
// wrappers copy to d_qdd (USE_QDD_FLAG overloads). Same leading-entries-then-zeros rule.
static inline void pack_qdd(GridCtx *ctx, const T* qdd, int batch)
{
    GRID_RBD_CTX_LOCALS(ctx);
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    for (int t = 0; t < batch; ++t) {
        std::memcpy(&g_data->h_qdd[t * nj], &qdd[t * nv], nv * sizeof(T));
        if (nj > nv) std::memset(&g_data->h_qdd[t * nj + nv], 0, (nj - nv) * sizeof(T));
    }
}

// `width` leading entries of each `pitch`-strided row of `src` into the tightly packed
// `dst` (the NUM_VEL-wide caller view of the NUM_JOINTS-pitched h_c / h_qdd buffers).
static inline void unpack_rows(T* dst, const T* src, int batch, int width, int pitch)
{
    for (int t = 0; t < batch; ++t)
        std::memcpy(&dst[t * width], &src[t * pitch], width * sizeof(T));
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

// com uses the COMPRESSED input layout (h_q / d_q, stride NUM_JOINTS), unlike
// the other surfaces which read the [q,qd,u]-interleaved h_q_qd_u.
static inline void pack_q(GridCtx *ctx, const T* q, int batch, int num_joints) {
    GRID_RBD_CTX_LOCALS(ctx);
    std::memcpy(g_data->h_q, q, (size_t)batch * num_joints * sizeof(T));
}

// ─── kernel introspection helper (E1 ceiling; used by every family unit) ─────
static inline int grid_kernel_ceiling(const void* fp) {
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

// ─── integrator-type dispatch (C-ABI surface; the jax/torch twins live in their preludes) ───
#define GRID_RBD_IT_DISPATCH(it_code, FN, ...)                                   \
    switch (it_code) {                                                           \
        case 0: FN<grid::IntegratorType::EULER>(g_ctx, __VA_ARGS__); break;             \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER>(g_ctx, __VA_ARGS__); break;\
        case 2: FN<grid::IntegratorType::MIDPOINT>(g_ctx, __VA_ARGS__); break;          \
        case 3: FN<grid::IntegratorType::RK4>(g_ctx, __VA_ARGS__); break;               \
        case 4: FN<grid::IntegratorType::TRAPEZOIDAL>(g_ctx, __VA_ARGS__); break;       \
        case 5: FN<grid::IntegratorType::CONSTANT_ACCELERATION>(g_ctx, __VA_ARGS__); break; \
        default: return 3;                                                       \
    }

// plant_step_hessian only supports EULER / SI-EULER (the composed device fn
// static_asserts MIDPOINT/RK out — instantiating those cases would fail to
// COMPILE, so this dispatch never names them). Other IT codes return rc=3.
// Single-stage dispatch (EULER / SEMI_IMPLICIT_EULER only): the MuJoCo integration
// values and gradients and the step Hessians support no multi-stage scheme, and the
// generated kernels static_assert those instantiations out — so this switch never
// names them. Other IT codes return rc=3.
#define GRID_RBD_IT_DISPATCH_SS(it_code, FN, ...)                                \
    switch (it_code) {                                                           \
        case 0: FN<grid::IntegratorType::EULER>(g_ctx, __VA_ARGS__); break;             \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER>(g_ctx, __VA_ARGS__); break;\
        default: return 3;                                                       \
    }
#define GRID_RBD_IT_DISPATCH_HESSIAN GRID_RBD_IT_DISPATCH_SS

// Shared (IT, MUJOCO_FLAG) two-template-arg switch cores for the jax-FFI /
// torch dispatchers below (which differ only in their bad-code statement).
// ERR_STMT is the whole surface-specific statement (ffi::Error return vs
// TORCH_CHECK). The _SS core is single-stage (EULER / SEMI_IMPLICIT_EULER)
// ONLY — the mjx gradient / plant-step epilogues static_assert multi-stage
// out, so those dispatchers must never name the multi-stage cases.
#define GRID_RBD_IT_SWITCH_MJX(it_code, FN, MUJOCO_FLAG, ERR_STMT, ...)                         \
    switch (it_code) {                                                                          \
        case 0: FN<grid::IntegratorType::EULER, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break;               \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break; \
        case 2: FN<grid::IntegratorType::MIDPOINT, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break;            \
        case 3: FN<grid::IntegratorType::RK4, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break;                 \
        case 4: FN<grid::IntegratorType::TRAPEZOIDAL, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break;         \
        case 5: FN<grid::IntegratorType::CONSTANT_ACCELERATION, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break; \
        default: ERR_STMT;                                                                      \
    }
#define GRID_RBD_IT_SWITCH_MJX_SS(it_code, FN, MUJOCO_FLAG, ERR_STMT, ...)                      \
    switch (it_code) {                                                                          \
        case 0: FN<grid::IntegratorType::EULER, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break;               \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER, MUJOCO_FLAG>(g_ctx, __VA_ARGS__); break; \
        default: ERR_STMT;                                                                      \
    }

// ─── torch op-library naming (used by the core library definition and every family fragment) ───
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
#define GRID_RBD_TORCH_LIBRARY_FRAGMENT(ns, m) TORCH_LIBRARY_FRAGMENT(ns, m)

// ─── torch surface includes + dtype macros (core defines the op library; family units implement ops) ───
#if defined(GRID_RBD_WITH_TORCH)
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
#endif  // GRID_RBD_WITH_TORCH

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
#include <cstring>

using T = float;

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
static dim3 g_block_dimms = dim3(1, 1, 1);
// g_thread_dimms defaults to the codegen-time MAX_PERF_LEVEL_THREADS hint. After
// the v2.0 cuBLASDx removal there is no hard floor — callers can override
// via grid_rbd_set_threads_per_block() before issuing any kernel calls.
static dim3 g_thread_dimms = dim3(grid::MAX_PERF_LEVEL_THREADS, 1, 1);

// ─── lifecycle ───────────────────────────────────────────────────────────────

extern "C" int grid_rbd_init() {
    if (g_data) return 0;  // already initialized
    g_streams = grid::init_grid<T>();
    g_robot   = grid::init_robotModel<T>();
    g_data    = grid::init_gridData<T, kMaxBatch>();
    return (g_data && g_robot && g_streams) ? 0 : 1;
}

extern "C" int grid_rbd_close() {
    if (!g_data) return 0;
    grid::close_grid<T>(g_streams, g_robot, g_data);
    g_data    = nullptr;
    g_robot   = nullptr;
    g_streams = nullptr;
    return 0;
}

// ─── metadata ────────────────────────────────────────────────────────────────

extern "C" int grid_rbd_num_joints()     { return grid::NUM_JOINTS; }
extern "C" int grid_rbd_num_vel()        { return grid::NUM_VEL; }
extern "C" int grid_rbd_num_ees()        { return grid::NUM_EES; }
extern "C" int grid_rbd_num_bodies()     { return grid::NUM_BODIES; }
extern "C" int grid_rbd_max_batch()      { return kMaxBatch; }
extern "C" int grid_rbd_max_perf_level_threads() { return grid::MAX_PERF_LEVEL_THREADS; }
extern "C" int grid_rbd_threads_per_block() { return (int)g_thread_dimms.x; }
extern "C" int grid_rbd_set_threads_per_block(int n) {
    // Override the per-block thread count used for all subsequent kernel
    // launches. n must be >= 1; values larger than the per-block max
    // (1024 on current GPUs) will fail at launch time with cudaErrorInvalidConfiguration.
    // The default is grid::MAX_PERF_LEVEL_THREADS; the codegen no longer pins
    // launch_bounds, so any block size that has enough threads to cover
    // the algorithm's parallel work is valid (the SIMT helpers use
    // block-stride loops, so smaller block sizes are correct but slower).
    if (n < 1) return 1;
    g_thread_dimms = dim3((unsigned)n, 1, 1);
    return 0;
}

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
extern "C" int grid_rbd_inverse_dynamics(
    const T* q, const T* qd, const T* qdd_opt,
    T* c_out,
    int batch, T gravity, const T* f_ext)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;  // caller should chunk

    // For now, USE_QDD_FLAG=false; qdd defaults to 0 in the kernel. To support
    // user-provided qdd we'd need USE_QDD_FLAG=true and the qdd buffer pushed
    // separately. Document this in the Python layer.
    (void)qdd_opt;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, nullptr, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/false, /*USE_COMPRESSED_MEM=*/false>(
        g_data, g_robot, gravity, batch, g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(c_out, g_data->h_c, batch * nj * sizeof(T));
    return 0;
}

// Direct mass-matrix inverse: Minv(q)
extern "C" int grid_rbd_minv(
    const T* q,
    T* minv_out,
    int batch)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, /*qd=*/q, /*u=*/nullptr, batch, nj);  // qd/u unused by minv

    grid::minv<T, /*USE_COMPRESSED_MEM=*/false>(
        g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(minv_out, g_data->h_Minv, batch * nj * nj * sizeof(T));
    return 0;
}

// Forward dynamics: qdd = Minv(q)·(τ − c(q,qd))
// f_ext (optional, may be null): (batch, 6*NUM_BODIES) local-frame body wrenches.
extern "C" int grid_rbd_forward_dynamics(
    const T* q, const T* qd, const T* u,
    T* qdd_out,
    int batch, T gravity, const T* f_ext)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    grid::forward_dynamics<T>(
        g_data, g_robot, gravity, batch, g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(qdd_out, g_data->h_qdd, batch * nj * sizeof(T));
    return 0;
}

// Articulated body algorithm: qdd = aba(q, qd, u)
// f_ext (optional, may be null): (batch, 6*NUM_BODIES) local-frame body wrenches.
extern "C" int grid_rbd_aba(
    const T* q, const T* qd, const T* u,
    T* qdd_out,
    int batch, T gravity, const T* f_ext)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    grid::aba<T>(g_data, g_robot, gravity, batch,
                 g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(qdd_out, g_data->h_qdd, batch * nj * sizeof(T));
    return 0;
}

// Composite rigid body algorithm: M = crba(q)
extern "C" int grid_rbd_crba(
    const T* q,
    T* m_out,
    int batch, T gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, q, nullptr, batch, nj);  // qd/u unused

    grid::crba<T>(g_data, g_robot, gravity, batch,
                  g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(m_out, g_data->h_M, batch * nj * nj * sizeof(T));
    return 0;
}

// End-effector pose: 6×NUM_EES per timestep (xyz + rpy).
extern "C" int grid_rbd_end_effector_pose(
    const T* q,
    T* ee_out,
    int batch)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, q, nullptr, batch, nj);

    grid::end_effector_pose<T, /*USE_COMPRESSED_MEM=*/false>(
        g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(ee_out, g_data->h_eePos, batch * 6 * grid::NUM_EES * sizeof(T));
    return 0;
}

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
                                                        (int)g_thread_dimms.x < 32 ? 32 : (int)g_thread_dimms.x);
    else
        grid::ee_pose_fk_batched<T, /*USE_WARP=*/false>(d_pose7, d_q_fk, batch, g_robot,
                                                        (int)g_thread_dimms.x);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    cudaMemcpy(pose7_out, d_pose7, sizeof(T) * batch * 7, cudaMemcpyDeviceToHost);
    return 0;
#else
    (void)q; (void)pose7_out; (void)batch; (void)use_warp;
    return 3;  // not supported for this robot (floating-base / mimic)
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
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    pack_q_qd_u(q, q, nullptr, batch, nj);

    grid::end_effector_pose_gradient<T, /*USE_COMPRESSED_MEM=*/false>(
        g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(dee_out, g_data->h_deePos,
                batch * 6 * grid::NUM_EES * nv * sizeof(T));
    return 0;
}

// ∂c/∂(q, qd): output shape (batch, NJ, 2*NJ) — concatenated [dc_dq | dc_dqd].
// f_ext (optional, may be null): (batch, 6*NUM_BODIES) local-frame body wrenches.
// f_ext enters RNEA additively (affine), so dc/d(q,qd) is unchanged for a
// CONSTANT f_ext; this just keeps the bias consistent with grid_rbd_inverse_dynamics.
extern "C" int grid_rbd_inverse_dynamics_gradient(
    const T* q, const T* qd, const T* qdd_opt,
    T* dc_du_out,
    int batch, T gravity, const T* f_ext)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    (void)qdd_opt;  // USE_QDD_FLAG=false currently; future v2

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, nullptr, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/false,
                                       /*USE_COMPRESSED_MEM=*/false>(
        g_data, g_robot, gravity, batch,
        g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(dc_du_out, g_data->h_dc_du,
                batch * nj * 2 * nj * sizeof(T));
    return 0;
}

// ∂qdd/∂(q, qd): output shape (batch, NJ, 2*NJ).
// f_ext (optional, may be null): (batch, 6*NUM_BODIES) local-frame body wrenches.
extern "C" int grid_rbd_forward_dynamics_gradient(
    const T* q, const T* qd, const T* u,
    T* df_du_out,
    int batch, T gravity, const T* f_ext)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    grid::forward_dynamics_gradient<T, /*USE_QDD_MINV_FLAG=*/false>(
        g_data, g_robot, gravity, batch,
        g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    reset_f_ext(f_ext, batch);
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(df_du_out, g_data->h_df_du,
                batch * nj * 2 * nj * sizeof(T));
    return 0;
}

// End-effector pose Hessian: 6×NUM_EES×NV×NV per timestep (d^2/dv^2 tangent).
// Calls grid::end_effector_pose_hessian which fills BOTH d2eePos AND
// deePos; we only copy d2eePos out. If the caller wants both they should
// call end_effector_pose_gradient separately (the kernels are fast enough
// that doing the work twice is fine for a small convenience).
extern "C" int grid_rbd_end_effector_pose_hessian(
    const T* q,
    T* d2ee_out,
    int batch)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    pack_q_qd_u(q, q, nullptr, batch, nj);

    grid::end_effector_pose_hessian<T, /*USE_COMPRESSED_MEM=*/false>(
        g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(d2ee_out, g_data->h_d2eePos,
                batch * 6 * grid::NUM_EES * nv * nv * sizeof(T));
    return 0;
}

// Second-order inverse dynamics. Output is the concatenated SO tensor of
// shape SECOND_ORDER_TENSOR_SIZE = 4 * NV^3 per timestep (four NV^3 blocks:
// d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq). The Python side slices into the
// four named tensors.
extern "C" int grid_rbd_idsva_so(
    const T* q, const T* qd, const T* qdd,
    T* out, int batch, T gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    (void)qdd;  // USE_QDD_FLAG=false for now; future v2

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, nullptr, batch, nj);

    grid::idsva_so<T>(
        g_data, g_robot, gravity, batch,
        g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(out, g_data->h_idsva_so,
                batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T));
    return 0;
}

// Second-order forward dynamics. Output is 4 * NV^3 per timestep
// (d2qdd_dq, d2qdd_dqd, d2qdd_dudq — interpretation per Singh/Wensing).
extern "C" int grid_rbd_fdsva_so(
    const T* q, const T* qd, const T* u,
    T* out, int batch, T gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);

    grid::fdsva_so<T>(
        g_data, g_robot, gravity, batch,
        g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(out, g_data->h_df2,
                batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T));
    return 0;
}


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
//   frame_jacobian      : 6*NUM_VEL     (6 x NV col-major, [linear;angular], leaf-EE / LWA frame)
//   frame_jacobian_dot  : 6*NUM_VEL     (time-derivative of frame_jacobian along qd)
//   osc_inertia         : 36            (6x6 task inertia Lambda = (J Minv J^T)^-1)
//
// target frame for frame_jacobian / frame_jacobian_dot / osc_inertia is baked
// at codegen time (the leaf-EE joint, LOCAL_WORLD_ALIGNED reference frame) — it
// is NOT a runtime parameter of the host/kernel surface, so these methods do
// not take a frame kwarg here.

// com uses the COMPRESSED input layout (h_q / d_q, stride NUM_JOINTS), unlike
// the other surfaces which read the [q,qd,u]-interleaved h_q_qd_u.
static inline void pack_q(const T* q, int batch, int num_joints) {
    std::memcpy(g_data->h_q, q, (size_t)batch * num_joints * sizeof(T));
}

// com(q) -> [p_com(3); J_com(3 x NV)] per timestep, total 3 + 3*NUM_VEL floats.
extern "C" int grid_rbd_com(const T* q, T* out, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q(q, batch, grid::NUM_JOINTS);
    grid::com<T>(g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_com, (size_t)batch * (3 + 3 * grid::NUM_VEL) * sizeof(T));
    return 0;
}

// ccrba(q, qd) -> [A(6 x NV); h(6)] per timestep, total 6*NUM_VEL + 6 floats.
extern "C" int grid_rbd_ccrba(const T* q, const T* qd, T* out, int batch) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::ccrba<T>(g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_ccrba, (size_t)batch * (6 * grid::NUM_VEL + 6) * sizeof(T));
    return 0;
}

// energy(q, qd) -> [KE, PE, KE+PE] per timestep, total 3 floats. Takes gravity.
extern "C" int grid_rbd_energy(const T* q, const T* qd, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::energy<T>(g_data, g_robot, gravity, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_energy, (size_t)batch * 3 * sizeof(T));
    return 0;
}

// generalized_gravity(q) -> g(q) = RNEA(q,0,0) per timestep, NUM_VEL floats. Takes gravity.
extern "C" int grid_rbd_generalized_gravity(const T* q, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);  // qd unused (zeroed internally)
    grid::generalized_gravity<T>(g_data, g_robot, gravity, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_c, (size_t)batch * grid::NUM_VEL * sizeof(T));
    return 0;
}

// nonlinear_effects(q, qd) -> c(q,qd) = RNEA(q,qd,0) per timestep, NUM_VEL floats. Takes gravity.
extern "C" int grid_rbd_nonlinear_effects(const T* q, const T* qd, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::nonlinear_effects<T>(g_data, g_robot, gravity, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_c, (size_t)batch * grid::NUM_VEL * sizeof(T));
    return 0;
}

// frame_jacobian(q) -> 6 x NUM_VEL geometric Jacobian (col-major, [linear;angular])
// at the leaf-EE frame, LOCAL_WORLD_ALIGNED. Gated on GRID_HAS_FRAME_JACOBIAN
// (the frame_jacobian family is opt-in codegen; only present when requested).
extern "C" int grid_rbd_frame_jacobian(const T* q, T* out, int batch) {
#ifdef GRID_HAS_FRAME_JACOBIAN
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, q, nullptr, batch, grid::NUM_JOINTS);
    grid::frame_jacobian<T>(g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_frame_jacobian, (size_t)batch * 6 * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch;
    return 3;  // frame_jacobian not generated for this .so
#endif
}

// frame_jacobian_dot(q, qd) -> d/dt of the leaf-EE frame Jacobian along v=qd,
// 6 x NUM_VEL (col-major, [linear;angular]). Gated on GRID_HAS_FRAME_JACOBIAN.
extern "C" int grid_rbd_frame_jacobian_dot(const T* q, const T* qd, T* out, int batch) {
#ifdef GRID_HAS_FRAME_JACOBIAN
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::frame_jacobian_dot<T>(g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_frame_jacobian_dot, (size_t)batch * 6 * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch;
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
    grid::osc_inertia<T>(g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_osc_inertia, (size_t)batch * 36 * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch;
    return 3;
#endif
}


// ────────────────────────────────────────────────────────────────────────────
// Time integrator (value + gradient)
// ────────────────────────────────────────────────────────────────────────────
//
// dt is a runtime float; gravity is the signed gravitational acceleration (default -9.81). The
// integrator type is selected at call time via an int code (0=EULER,
// 1=SEMI_IMPLICIT_EULER, 2=MIDPOINT, 3=RK3, 4=RK4) dispatched onto the
// compile-time `IntegratorType IT` template. x_kp1 is size (NUM_POS + NUM_VEL)
// per timestep; dAB is (2*NUM_VEL) x (3*NUM_VEL) per timestep (column-major).

// host-path launchers (call the host wrappers, which stage memory + own streams)
template <grid::IntegratorType IT>
static void launch_integrator_host(int batch, T gravity, T dt) {
    grid::integrator<T, IT>(g_data, g_robot, /*gravity=*/gravity,
                            dt, batch, g_block_dimms, g_thread_dimms, g_streams);
}
template <grid::IntegratorType IT>
static void launch_integrator_grad_host(int batch, T gravity, T dt) {
    grid::integrator_gradient<T, IT>(g_data, g_robot, /*gravity=*/gravity,
                                     dt, batch, g_block_dimms, g_thread_dimms, g_streams);
}

#define GRID_RBD_IT_DISPATCH(it_code, FN, ...)                                   \
    switch (it_code) {                                                           \
        case 0: FN<grid::IntegratorType::EULER>(__VA_ARGS__); break;             \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER>(__VA_ARGS__); break;\
        case 2: FN<grid::IntegratorType::MIDPOINT>(__VA_ARGS__); break;          \
        case 3: FN<grid::IntegratorType::RK3>(__VA_ARGS__); break;               \
        case 4: FN<grid::IntegratorType::RK4>(__VA_ARGS__); break;               \
        default: return 3;                                                       \
    }

// integrator(q, qd, u, dt, it) → x_kp1  (size NUM_POS + NUM_VEL per timestep)
extern "C" int grid_rbd_integrator(
    const T* q, const T* qd, const T* u,
    T* x_kp1_out, int batch, T gravity, T dt, int it)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);

    GRID_RBD_IT_DISPATCH(it, launch_integrator_host, batch, gravity, dt);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(x_kp1_out, g_data->h_x_kp1,
                batch * (grid::NUM_POS + grid::NUM_VEL) * sizeof(T));
    return 0;
}

// integrator_gradient(q, qd, u, dt, it) → dAB  (2*NV x 3*NV per timestep)
extern "C" int grid_rbd_integrator_gradient(
    const T* q, const T* qd, const T* u,
    T* dAB_out, int batch, T gravity, T dt, int it)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);

    GRID_RBD_IT_DISPATCH(it, launch_integrator_grad_host, batch, gravity, dt);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    const int nv = grid::NUM_VEL;
    std::memcpy(dAB_out, g_data->h_dAB,
                batch * (2 * nv) * (3 * nv) * sizeof(T));
    return 0;
}


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
    T* d_in_b   = nullptr;   // des / lower / p_des
    T* d_in_c   = nullptr;   // weight / upper
    T* d_out    = nullptr;   // scalar cost (1 per timestep)
    T* d_grad   = nullptr;   // gradient
    T* d_hess   = nullptr;   // dense hessian / hess-diagonal
    T* d_eePos  = nullptr;   // ee-pose scratch
    T* d_deePos = nullptr;   // ee-jacobian scratch
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
    const size_t vec = (size_t)nx;                  // >= nv, >= nq, >= 3
    const size_t mat = (size_t)nx * (size_t)nx;     // dense hessian
    auto ok = [](cudaError_t e){ return e == cudaSuccess; };
    bool good = true;
    good &= ok(cudaMalloc(&g_plant.d_in_a,  B * vec * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_in_b,  B * vec * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_in_c,  B * vec * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_out,   B * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_grad,  B * vec * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_hess,  B * mat * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_eePos, B * (size_t)(6 * nee) * sizeof(T)));
    good &= ok(cudaMalloc(&g_plant.d_deePos, B * (size_t)(6 * nv * nee) * sizeof(T)));
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    if (STATE) {
        grid_plant::quadratic_state_cost_kernel<T><<<grid_dim, g_thread_dimms, 0, g_streams[0]>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
    } else {
        grid_plant::quadratic_input_cost_kernel<T><<<grid_dim, g_thread_dimms, 0, g_streams[0]>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
    }
    cudaError_t e = cudaDeviceSynchronize();
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    switch (which) {
        case PlantBarrier::POSITION:
            grid_plant::joint_position_barrier_kernel<T><<<grid_dim, g_thread_dimms, 0, g_streams[0]>>>(
                g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
                g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch); break;
        case PlantBarrier::VELOCITY:
            grid_plant::joint_velocity_barrier_kernel<T><<<grid_dim, g_thread_dimms, 0, g_streams[0]>>>(
                g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
                g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch); break;
        case PlantBarrier::TORQUE:
            grid_plant::joint_torque_barrier_kernel<T><<<grid_dim, g_thread_dimms, 0, g_streams[0]>>>(
                g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
                g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch); break;
    }
    cudaError_t e = cudaDeviceSynchronize();
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::plant_step_kernel<T, IT><<<grid_dim, g_thread_dimms,
        grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>(), g_streams[0]>>>(
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
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(x_kp1, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
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
    size_t smem = grid::DEE_POS_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::ee_pos_cost_kernel<T, 0><<<grid_dim, g_thread_dimms, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_eePos, g_plant.d_deePos, g_robot, batch);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(out,  g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, g_plant.d_hess, batch * nx * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_PLANT_HAS_EE_COST


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

#ifdef GRID_RBD_WITH_JAX

#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

// inverse_dynamics(q, qd) → c   — fully device-resident path.
static ffi::Error grid_rbd_jax_inverse_dynamics_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,         // shape (B, NJ), device-resident
    ffi::Buffer<ffi::F32> qd,        // shape (B, NJ), device-resident
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

    // Launch kernel directly on JAX's stream (skip the host wrapper's
    // implicit H→D + own-stream dance). RNEA uses USE_QDD_FLAG=false; the
    // kernel name embeds that via overload resolution on the d_q_qd
    // signature without a qdd input.
    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::inverse_dynamics_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::ID_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_c, g_data->d_q_qd_u, stride_q_qd,
            g_data->d_f_ext, g_robot, /*gravity=*/gravity, batch);

    // D→D copy the result into JAX's output buffer on the same stream.
    cudaMemcpyAsync(c->typed_data(), g_data->d_c,
                    batch * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics,
    grid_rbd_jax_inverse_dynamics_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()  // q
        .Arg<ffi::Buffer<ffi::F32>>()  // qd
        .Ret<ffi::Buffer<ffi::F32>>()  // c
        .Attr<float>("gravity")
);

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


// minv(q) → Minv  (kernel writes lower triangle only; symmetrize Python-side)
static ffi::Error grid_rbd_jax_minv_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> minv_out)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "minv: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("minv: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch,
                      q.typed_data(),       row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd_u = 3 * grid::NUM_JOINTS;
    grid::minv_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::MINV_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_Minv, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u,
            g_robot, batch);

    cudaMemcpyAsync(minv_out->typed_data(), g_data->d_Minv,
                    batch * nj * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_minv,
    grid_rbd_jax_minv_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);


// forward_dynamics(q, qd, u) → qdd
static ffi::Error grid_rbd_jax_forward_dynamics_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> u,
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

    constexpr int stride_q_qd_u = 3 * grid::NUM_JOINTS;
    grid::forward_dynamics_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::FD_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_qdd, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u,
            g_data->d_f_ext, g_robot, /*gravity=*/gravity, batch);

    cudaMemcpyAsync(qdd_out->typed_data(), g_data->d_qdd,
                    batch * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_forward_dynamics,
    grid_rbd_jax_forward_dynamics_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);


// aba(q, qd, u) → qdd  — same kernel signature shape as forward_dynamics
static ffi::Error grid_rbd_jax_aba_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> u,
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

    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::aba_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::ABA_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_qdd, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd,
            g_data->d_f_ext, g_robot, /*gravity=*/gravity, batch);

    cudaMemcpyAsync(qdd_out->typed_data(), g_data->d_qdd,
                    batch * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_aba,
    grid_rbd_jax_aba_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);


// crba(q) → M  (kernel writes the full mass matrix; no symmetrize needed)
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
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("crba: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch,
                      q.typed_data(),       row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::crba_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::CRBA_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_M, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd,
            g_robot, /*gravity=*/gravity, batch);

    cudaMemcpyAsync(m_out->typed_data(), g_data->d_M,
                    batch * nj * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_crba,
    grid_rbd_jax_crba_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);


// end_effector_pose(q) → eePos  flat (B, 6*NUM_EES)
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
    grid::end_effector_pose_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::EE_POS_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_eePos, g_data->d_q_qd_u, stride_q,
            g_robot, batch);

    cudaMemcpyAsync(ee_out->typed_data(), g_data->d_eePos,
                    batch * 6 * grid::NUM_EES * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose,
    grid_rbd_jax_end_effector_pose_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);


// end_effector_pose_gradient(q) → deePos d/dv flat (B, 6*NUM_EES*NV).
// Output convention: d/dv tangent (pinocchio); floating-base shape uses NV
// (= 6 + n_joints) NOT NJ. Python side reshapes/transposes to the
// (B, 6*NUM_EES, NV) row-major convention (see _handle.py).
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
    grid::end_effector_pose_gradient_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::DEE_POS_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_deePos, g_data->d_workspace, g_data->d_q_qd_u, stride_q,
            g_robot, batch);

    cudaMemcpyAsync(dee_out->typed_data(), g_data->d_deePos,
                    batch * 6 * grid::NUM_EES * nv * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_gradient,
    grid_rbd_jax_end_effector_pose_gradient_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);


// end_effector_pose_hessian(q) → d2eePos  flat (B, 6*NUM_EES*NV*NV)
// The kernel also writes d_deePos as a byproduct; we only return d2.
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
    grid::end_effector_pose_hessian_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::D2EE_POS_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_d2eePos, g_data->d_deePos, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q, g_robot, batch);

    cudaMemcpyAsync(d2ee_out->typed_data(), g_data->d_d2eePos,
                    batch * 6 * grid::NUM_EES * nv * nv * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_end_effector_pose_hessian,
    grid_rbd_jax_end_effector_pose_hessian_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);


// inverse_dynamics_gradient(q, qd) → dc_du  flat (B, 2*NJ*NJ)
// Python reshapes/transposes to (B, NJ, 2*NJ) [dc_dq | dc_dqd].
// USE_QDD_FLAG=false for now; qdd defaults to 0 in-kernel.
static ffi::Error grid_rbd_jax_inverse_dynamics_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::ResultBuffer<ffi::F32> dc_du_out,
    float gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "inverse_dynamics_gradient: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("inverse_dynamics_gradient: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],  dst_pitch,
                      q.typed_data(),        row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch,
                      qd.typed_data(),       row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd = 3 * grid::NUM_JOINTS;
    grid::inverse_dynamics_gradient_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::ID_DU_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_dc_du, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd,
            g_data->d_f_ext, g_robot, /*gravity=*/gravity, batch);

    cudaMemcpyAsync(dc_du_out->typed_data(), g_data->d_dc_du,
                    batch * nj * 2 * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics_gradient,
    grid_rbd_jax_inverse_dynamics_gradient_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);


// forward_dynamics_gradient(q, qd, u) → df_du  flat (B, 2*NJ*NJ)
// Python reshapes/transposes to (B, NJ, 2*NJ).
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
    grid::forward_dynamics_gradient_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::FD_DU_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_df_du, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u,
            g_data->d_f_ext, g_robot, /*gravity=*/gravity, batch);

    cudaMemcpyAsync(df_du_out->typed_data(), g_data->d_df_du,
                    batch * nj * 2 * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_forward_dynamics_gradient,
    grid_rbd_jax_forward_dynamics_gradient_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);


// idsva_so(q, qd) → packed (B, SECOND_ORDER_TENSOR_SIZE)
// USE_QDD_FLAG=false. The codegen-time dispatcher picks body- vs world-frame;
// we dispatch here at compile time using the GRID_GENERATES_* macros so a
// per-robot .so calls whichever kernel was emitted.
static ffi::Error grid_rbd_jax_idsva_so_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
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
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0],  dst_pitch,
                      q.typed_data(),        row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[nj], dst_pitch,
                      qd.typed_data(),       row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q_qd_u = 3 * grid::NUM_JOINTS;
    // v0.3: hardcoded body-frame kernel call — works for fixed-base robots
    // (which is what the iiwa14 smoke test exercises). Floating-base support
    // needs codegen to emit a #define so this dispatch can branch — tracked
    // as follow-up.
    grid::idsva_so_body_frame_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_idsva_so, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u,
            g_robot, /*gravity=*/gravity, batch);

    cudaMemcpyAsync(out->typed_data(), g_data->d_idsva_so,
                    batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_idsva_so,
    grid_rbd_jax_idsva_so_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);


// fdsva_so(q, qd, u) → packed (B, SECOND_ORDER_TENSOR_SIZE)
// Uses d_idsva_so as scratch — must not run concurrently with idsva_so.
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
    grid::fdsva_so_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_df2, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd_u, g_data->d_idsva_so,
            g_robot, /*gravity=*/gravity, batch);

    cudaMemcpyAsync(out->typed_data(), g_data->d_df2,
                    batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_fdsva_so,
    grid_rbd_jax_fdsva_so_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);


// Integrator. dt + it are FFI attributes (runtime scalars; gravity is the
// standard constant). q/qd/u are packed D→D like aba; the integrator kernels
// are launched directly on the JAX stream.
template <grid::IntegratorType IT>
static void launch_integrator_kernel_jax(cudaStream_t stream, int batch, float dt, float gravity) {
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::integrator_kernel<T, IT><<<
        g_block_dimms, g_thread_dimms,
        grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_x_kp1, g_data->d_workspace, g_data->d_q_qd_u, stride,
            g_robot, /*gravity=*/static_cast<T>(gravity), static_cast<T>(dt), batch);
}
template <grid::IntegratorType IT>
static void launch_integrator_grad_kernel_jax(cudaStream_t stream, int batch, float dt, float gravity) {
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::integrator_gradient_kernel<T, IT><<<
        g_block_dimms, g_thread_dimms,
        grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_dAB, g_data->d_workspace, g_data->d_q_qd_u, stride,
            g_robot, /*gravity=*/static_cast<T>(gravity), static_cast<T>(dt), batch);
}

#define GRID_RBD_IT_DISPATCH_FFI(it_code, FN, ...)                                 \
    switch (it_code) {                                                             \
        case 0: FN<grid::IntegratorType::EULER>(__VA_ARGS__); break;               \
        case 1: FN<grid::IntegratorType::SEMI_IMPLICIT_EULER>(__VA_ARGS__); break;  \
        case 2: FN<grid::IntegratorType::MIDPOINT>(__VA_ARGS__); break;            \
        case 3: FN<grid::IntegratorType::RK3>(__VA_ARGS__); break;                 \
        case 4: FN<grid::IntegratorType::RK4>(__VA_ARGS__); break;                 \
        default: return ffi::Error::InvalidArgument("integrator: bad it code");    \
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

// integrator(q, qd, u; dt, it) → x_kp1  (B, NUM_POS + NUM_VEL)
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
    GRID_RBD_IT_DISPATCH_FFI((int)it, launch_integrator_kernel_jax, stream, batch, dt, gravity);

    cudaMemcpyAsync(x_kp1_out->typed_data(), g_data->d_x_kp1,
                    batch * (grid::NUM_POS + grid::NUM_VEL) * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_integrator,
    grid_rbd_jax_integrator_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("dt").Attr<int64_t>("it").Attr<float>("gravity")
);

// integrator_gradient(q, qd, u; dt, it) → dAB  (B, 2*NV, 3*NV)
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
    GRID_RBD_IT_DISPATCH_FFI((int)it, launch_integrator_grad_kernel_jax, stream, batch, dt, gravity);

    cudaMemcpyAsync(dAB_out->typed_data(), g_data->d_dAB,
                    batch * (2 * nv) * (3 * nv) * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_integrator_gradient,
    grid_rbd_jax_integrator_gradient_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("dt").Attr<int64_t>("it").Attr<float>("gravity")
);

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

#ifdef GRID_RBD_WITH_TORCH

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

torch::Tensor torch_inverse_dynamics(torch::Tensor q, torch::Tensor qd, double gravity,
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
    grid::inverse_dynamics_kernel<T><<<g_block_dimms, g_thread_dimms, grid::ID_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_c, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_c, batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(stream, batch, f_ext);
    return out;
}

torch::Tensor torch_minv(torch::Tensor q) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "minv: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, nj * nj, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::minv_kernel<T><<<g_block_dimms, g_thread_dimms, grid::MINV_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_Minv, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_Minv, batch * nj * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}

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
    grid::forward_dynamics_kernel<T><<<g_block_dimms, g_thread_dimms, grid::FD_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_qdd, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_qdd, batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(stream, batch, f_ext);
    return out;
}

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
    grid::aba_kernel<T><<<g_block_dimms, g_thread_dimms, grid::ABA_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_qdd, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_qdd, batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(stream, batch, f_ext);
    return out;
}

torch::Tensor torch_crba(torch::Tensor q, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "crba: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, nj * nj, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::crba_kernel<T><<<g_block_dimms, g_thread_dimms, grid::CRBA_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_M, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_M, batch * nj * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}

torch::Tensor torch_end_effector_pose(torch::Tensor q) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nee = grid::NUM_EES;
    grid_torch_check(q, "end_effector_pose: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 6 * nee, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::end_effector_pose_kernel<T><<<g_block_dimms, g_thread_dimms, grid::EE_POS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_eePos, g_data->d_q_qd_u, stride, g_robot, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_eePos, batch * 6 * nee * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}

torch::Tensor torch_end_effector_pose_gradient(torch::Tensor q) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL, nee = grid::NUM_EES;
    grid_torch_check(q, "end_effector_pose_gradient: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 6 * nee * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::end_effector_pose_gradient_kernel<T><<<g_block_dimms, g_thread_dimms, grid::DEE_POS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_deePos, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_deePos, batch * 6 * nee * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}

torch::Tensor torch_end_effector_pose_hessian(torch::Tensor q) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL, nee = grid::NUM_EES;
    grid_torch_check(q, "end_effector_pose_hessian: q", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, nullptr, nullptr);
    auto out = grid_torch_empty(batch, 6 * nee * nv * nv, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::end_effector_pose_hessian_kernel<T><<<g_block_dimms, g_thread_dimms, grid::D2EE_POS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_d2eePos, g_data->d_deePos, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_d2eePos, batch * 6 * nee * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}

torch::Tensor torch_inverse_dynamics_gradient(torch::Tensor q, torch::Tensor qd, double gravity,
                              c10::optional<torch::Tensor> f_ext) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "inverse_dynamics_gradient: q", nj); grid_torch_check(qd, "inverse_dynamics_gradient: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, nullptr);
    grid_torch_f_ext_apply(stream, batch, f_ext);
    auto out = grid_torch_empty(batch, nj * 2 * nj, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::inverse_dynamics_gradient_kernel<T><<<g_block_dimms, g_thread_dimms, grid::ID_DU_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_dc_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_dc_du, batch * nj * 2 * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(stream, batch, f_ext);
    return out;
}

torch::Tensor torch_forward_dynamics_gradient(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double gravity,
                                          c10::optional<torch::Tensor> f_ext) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "fd_grad: q", nj); grid_torch_check(qd, "fd_grad: qd", nj); grid_torch_check(u, "fd_grad: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, &u);
    grid_torch_f_ext_apply(stream, batch, f_ext);
    auto out = grid_torch_empty(batch, nj * 2 * nj, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::forward_dynamics_gradient_kernel<T><<<g_block_dimms, g_thread_dimms, grid::FD_DU_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_df_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_df_du, batch * nj * 2 * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(stream, batch, f_ext);
    return out;
}

torch::Tensor torch_idsva_so(torch::Tensor q, torch::Tensor qd, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "idsva_so: q", nj); grid_torch_check(qd, "idsva_so: qd", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, nullptr);
    auto out = grid_torch_empty(batch, grid::SECOND_ORDER_TENSOR_SIZE, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::idsva_so_body_frame_kernel<T><<<g_block_dimms, g_thread_dimms, grid::IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_idsva_so, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_idsva_so, batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}

torch::Tensor torch_fdsva_so(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "fdsva_so: q", nj); grid_torch_check(qd, "fdsva_so: qd", nj); grid_torch_check(u, "fdsva_so: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, &u);
    auto out = grid_torch_empty(batch, grid::SECOND_ORDER_TENSOR_SIZE, q);
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::fdsva_so_kernel<T><<<g_block_dimms, g_thread_dimms, grid::FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_df2, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_idsva_so, g_robot, (T)gravity, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_df2, batch * grid::SECOND_ORDER_TENSOR_SIZE * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}

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

template <grid::IntegratorType IT>
static void torch_launch_integrator(cudaStream_t stream, int batch, double dt, double gravity) {
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::integrator_kernel<T, IT><<<g_block_dimms, g_thread_dimms, grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_x_kp1, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, (T)dt, batch);
}
template <grid::IntegratorType IT>
static void torch_launch_integrator_grad(cudaStream_t stream, int batch, double dt, double gravity) {
    constexpr int stride = 3 * grid::NUM_JOINTS;
    grid::integrator_gradient_kernel<T, IT><<<g_block_dimms, g_thread_dimms, grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_dAB, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, (T)dt, batch);
}

torch::Tensor torch_integrator(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double dt, int64_t it, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS;
    grid_torch_check(q, "integrator: q", nj); grid_torch_check(qd, "integrator: qd", nj); grid_torch_check(u, "integrator: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, &u);
    auto out = grid_torch_empty(batch, grid::NUM_POS + grid::NUM_VEL, q);
    GRID_RBD_IT_DISPATCH_TORCH((int)it, torch_launch_integrator, stream, batch, dt, gravity);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_x_kp1, batch * (grid::NUM_POS + grid::NUM_VEL) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}

torch::Tensor torch_integrator_gradient(torch::Tensor q, torch::Tensor qd, torch::Tensor u, double dt, int64_t it, double gravity) {
    grid_torch_init_or_throw();
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    grid_torch_check(q, "integrator_gradient: q", nj); grid_torch_check(qd, "integrator_gradient: qd", nj); grid_torch_check(u, "integrator_gradient: u", nj);
    int batch = grid_torch_batch(q);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    grid_torch_pack(stream, batch, nj, &q, &qd, &u);
    auto out = grid_torch_empty(batch, 2 * nv * 3 * nv, q);
    GRID_RBD_IT_DISPATCH_TORCH((int)it, torch_launch_integrator_grad, stream, batch, dt, gravity);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_dAB, batch * (2 * nv) * (3 * nv) * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}

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
    m.def("inverse_dynamics(Tensor q, Tensor qd, float gravity, Tensor? f_ext=None) -> Tensor");
    m.def("minv(Tensor q) -> Tensor");
    m.def("forward_dynamics(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None) -> Tensor");
    m.def("aba(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None) -> Tensor");
    m.def("crba(Tensor q, float gravity) -> Tensor");
    m.def("end_effector_pose(Tensor q) -> Tensor");
    m.def("end_effector_pose_gradient(Tensor q) -> Tensor");
    m.def("end_effector_pose_hessian(Tensor q) -> Tensor");
    m.def("inverse_dynamics_gradient(Tensor q, Tensor qd, float gravity, Tensor? f_ext=None) -> Tensor");
    m.def("forward_dynamics_gradient(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None) -> Tensor");
    m.def("idsva_so(Tensor q, Tensor qd, float gravity) -> Tensor");
    m.def("fdsva_so(Tensor q, Tensor qd, Tensor u, float gravity) -> Tensor");
    m.def("integrator(Tensor q, Tensor qd, Tensor u, float dt, int it, float gravity) -> Tensor");
    m.def("integrator_gradient(Tensor q, Tensor qd, Tensor u, float dt, int it, float gravity) -> Tensor");
}

GRID_RBD_TORCH_LIBRARY_IMPL(GRID_RBD_TORCH_LIB, CUDA, m) {
    m.impl("inverse_dynamics", torch_inverse_dynamics);
    m.impl("minv", torch_minv);
    m.impl("forward_dynamics", torch_forward_dynamics);
    m.impl("aba", torch_aba);
    m.impl("crba", torch_crba);
    m.impl("end_effector_pose", torch_end_effector_pose);
    m.impl("end_effector_pose_gradient", torch_end_effector_pose_gradient);
    m.impl("end_effector_pose_hessian", torch_end_effector_pose_hessian);
    m.impl("inverse_dynamics_gradient", torch_inverse_dynamics_gradient);
    m.impl("forward_dynamics_gradient", torch_forward_dynamics_gradient);
    m.impl("idsva_so", torch_idsva_so);
    m.impl("fdsva_so", torch_fdsva_so);
    m.impl("integrator", torch_integrator);
    m.impl("integrator_gradient", torch_integrator_gradient);
}

#endif  // GRID_RBD_WITH_TORCH

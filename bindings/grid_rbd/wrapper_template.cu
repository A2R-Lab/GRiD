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
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;  // caller should chunk

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, nullptr, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    if (qdd_opt) {
        // Host wrapper copies h_qdd→d_qdd (NUM_JOINTS per timestep, contiguous).
        std::memcpy(g_data->h_qdd, qdd_opt, (size_t)batch * nj * sizeof(T));
        grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false>(
            g_data, g_robot, gravity, batch, g_block_dimms, g_thread_dimms, g_streams);
    } else {
        grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/false, /*USE_COMPRESSED_MEM=*/false>(
            g_data, g_robot, gravity, batch, g_block_dimms, g_thread_dimms, g_streams);
    }

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

    std::memcpy(ee_out, g_data->h_end_effector_pose, batch * 6 * grid::NUM_EES * sizeof(T));
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

    std::memcpy(dee_out, g_data->h_end_effector_pose_gradient,
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

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, nullptr, batch, nj);
    if (int rc = apply_f_ext(f_ext, batch)) return rc;

    // qdd dependence: ∂c/∂(q,qd) DOES depend on qdd (via the M·qdd term's
    // derivatives). Same separate-d_qdd convention as grid_rbd_inverse_dynamics:
    // the host wrapper copies h_qdd→d_qdd when USE_QDD_FLAG=true. Null qdd keeps
    // the qdd=0 overload.
    if (qdd_opt) {
        std::memcpy(g_data->h_qdd, qdd_opt, (size_t)batch * nj * sizeof(T));
        grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/true,
                                           /*USE_COMPRESSED_MEM=*/false>(
            g_data, g_robot, gravity, batch,
            g_block_dimms, g_thread_dimms, g_streams);
    } else {
        grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/false,
                                           /*USE_COMPRESSED_MEM=*/false>(
            g_data, g_robot, gravity, batch,
            g_block_dimms, g_thread_dimms, g_streams);
    }

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
// Calls grid::end_effector_pose_hessian which fills BOTH end_effector_pose_hessian AND
// end_effector_pose_gradient; we only copy end_effector_pose_hessian out. If the caller wants both they should
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

    std::memcpy(d2ee_out, g_data->h_end_effector_pose_hessian,
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

    const int nj = grid::NUM_JOINTS;
    // idsva_so reads the joint acceleration from the u-slot of d_q_qd_u (s_qdd);
    // pack qdd there so the second-order tensors use the requested acceleration.
    pack_q_qd_u(q, qd, qdd, batch, nj);

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
// Gated on GRID_HAS_COM: com/ccrba/energy are NOT emitted for mimic robots (the
// per-body Jacobian fold isn't mimic-reduced), so this returns rc=3 there.
extern "C" int grid_rbd_com(const T* q, T* out, int batch) {
#ifdef GRID_HAS_COM
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q(q, batch, grid::NUM_JOINTS);
    grid::com<T>(g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_com, (size_t)batch * (3 + 3 * grid::NUM_VEL) * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch;
    return 3;  // com not generated for this robot (mimic)
#endif
}

// ccrba(q, qd) -> [A(6 x NV); h(6)] per timestep, total 6*NUM_VEL + 6 floats.
// Gated on GRID_HAS_CCRBA (same mimic caveat as com); returns rc=3 otherwise.
extern "C" int grid_rbd_ccrba(const T* q, const T* qd, T* out, int batch) {
#ifdef GRID_HAS_CCRBA
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::ccrba<T>(g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_ccrba, (size_t)batch * (6 * grid::NUM_VEL + 6) * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch;
    return 3;  // ccrba not generated for this robot (mimic)
#endif
}

// energy(q, qd) -> [KE, PE, KE+PE] per timestep, total 3 floats. Takes gravity.
// Gated on GRID_HAS_ENERGY (same mimic caveat as com); returns rc=3 otherwise.
extern "C" int grid_rbd_energy(const T* q, const T* qd, T* out, int batch, T gravity) {
#ifdef GRID_HAS_ENERGY
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::energy<T>(g_data, g_robot, gravity, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_energy, (size_t)batch * 3 * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch;
    return 3;  // energy not generated for this robot (mimic)
#endif
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

// coriolis_matrix(q, qd) -> nv x nv Coriolis matrix C(q,qd), row-major
// (C[row*nv + col]). Always emitted with the "all" profile (mimic-safe:
// alpha-folded column assembly), so it is bound UNGATED like com/ccrba.
extern "C" int grid_rbd_coriolis_matrix(const T* q, const T* qd, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::coriolis_matrix<T>(g_data, g_robot, gravity, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_coriolis, (size_t)batch * grid::NUM_VEL * grid::NUM_VEL * sizeof(T));
    return 0;
}

// kinetic_energy_regressor(q, qd) -> length 10*NUM_BODIES regressor y_KE
// (KE = y_KE . pi). Always emitted with the "all" profile (mimic-safe), ungated.
extern "C" int grid_rbd_kinetic_energy_regressor(const T* q, const T* qd, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::kinetic_energy_regressor<T>(g_data, g_robot, gravity, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_ke_regressor, (size_t)batch * 10 * grid::NUM_BODIES * sizeof(T));
    return 0;
}

// potential_energy_regressor(q) -> length 10*NUM_BODIES regressor y_PE
// (PE = y_PE . pi). Always emitted with the "all" profile (mimic-safe), ungated.
// Reads the COMPRESSED input layout (h_q / d_q) like com.
extern "C" int grid_rbd_potential_energy_regressor(const T* q, T* out, int batch, T gravity) {
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q(q, batch, grid::NUM_JOINTS);
    grid::potential_energy_regressor<T>(g_data, g_robot, gravity, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_pe_regressor, (size_t)batch * 10 * grid::NUM_BODIES * sizeof(T));
    return 0;
}

// dccrba(q) -> 6*NUM_VEL*NUM_VEL dCCRBA tensor dA/dq (per timestep, as the kernel
// writes it). Reads the COMPRESSED input layout (h_q / d_q). Gated on
// GRID_HAS_DCCRBA: dccrba is NOT emitted for mimic robots (per-body Jacobian fold
// isn't mimic-reduced), so this returns rc=3 there. For big floating robots whose
// kernel arena overflows the smem cap the host wrapper's
// grid_check_dynamic_shared_memory_bytes raises a clear rc!=0 at launch.
extern "C" int grid_rbd_dccrba(const T* q, T* out, int batch) {
#ifdef GRID_HAS_DCCRBA
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q(q, batch, grid::NUM_JOINTS);
    grid::dccrba<T>(g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_dccrba, (size_t)batch * 6 * grid::NUM_VEL * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch;
    return 3;  // dccrba not generated for this robot (mimic)
#endif
}

// cmm_time_variation(q, qd) -> 6*NUM_VEL centroidal-momentum-matrix time
// variation Adot (per timestep). Gated on GRID_HAS_CMM_TIME_VARIATION (same
// mimic caveat as dccrba); returns rc=3 when not generated.
extern "C" int grid_rbd_cmm_time_variation(const T* q, const T* qd, T* out, int batch) {
#ifdef GRID_HAS_CMM_TIME_VARIATION
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);
    grid::cmm_time_variation<T>(g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_cmm_time_variation, (size_t)batch * 6 * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)qd; (void)out; (void)batch;
    return 3;  // cmm_time_variation not generated for this robot (mimic)
#endif
}

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
    grid::frame_jacobian<T>(g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams,
                            target_jid, reference_frame);
    cudaError_t e = cudaDeviceSynchronize();
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
    grid::frame_jacobian_dot<T>(g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams,
                                target_jid, reference_frame);
    cudaError_t e = cudaDeviceSynchronize();
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
    T off[3] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
    if (offset) { off[0]=offset[0]; off[1]=offset[1]; off[2]=offset[2]; }
    if (cudaMemcpy(g_data->d_eepose_runtime_offset, off, 3*sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 101;
    grid::end_effector_pose_runtime<T>(g_data, g_robot, batch, g_block_dimms, g_thread_dimms,
                                       g_streams, target_jid);
    cudaError_t e = cudaDeviceSynchronize();
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
    T off[3] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};
    if (offset) { off[0]=offset[0]; off[1]=offset[1]; off[2]=offset[2]; }
    if (cudaMemcpy(g_data->d_eepose_runtime_offset, off, 3*sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 101;
    grid::end_effector_pose_gradient_runtime<T>(g_data, g_robot, batch, g_block_dimms,
                                                g_thread_dimms, g_streams, target_jid);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    std::memcpy(out, g_data->h_eePoseGrad, (size_t)batch * 6 * grid::NUM_VEL * sizeof(T));
    return 0;
#else
    (void)q; (void)out; (void)batch; (void)target_jid; (void)offset;
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
    // d_end_effector_pose is reused as the com (3+3*nv) / ccrba (6*nv+6) device scratch.
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
    size_t smem = grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::ee_pos_cost_kernel<T, 0><<<grid_dim, g_thread_dimms, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose, g_plant.d_end_effector_pose_gradient, g_robot, batch);
    cudaError_t e = cudaDeviceSynchronize();
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::com_cost_kernel<T><<<grid_dim, g_thread_dimms, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose /*reused as com (3+3*NV) scratch*/, g_robot, batch);
    cudaError_t e = cudaDeviceSynchronize();
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::momentum_cost_kernel<T><<<grid_dim, g_thread_dimms, smem, g_streams[0]>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b,
        g_plant.d_in_c, g_plant.d_in_c + (size_t)batch * 6,
        g_plant.d_end_effector_pose /*reused as ccrba (6*NV+6) scratch*/, g_robot, batch);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(out,  g_plant.d_out,  batch * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(grad, g_plant.d_grad, batch * nx * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(hess, g_plant.d_hess, batch * nx * nx * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
#endif  // GRID_PLANT_HAS_MOMENTUM_COST

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
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::plant_step_gradient_kernel<T, IT><<<grid_dim, g_thread_dimms,
        grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T>(), g_streams[0]>>>(
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
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(dAB, g_plant.d_grad, batch * dab * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::plant_step_hessian_kernel<T, IT><<<grid_dim, g_thread_dimms, smem, g_streams[0]>>>(
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
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;
    cudaMemcpy(d2AB, g_plant.d_d2AB, batch * d2ab * sizeof(T), cudaMemcpyDeviceToHost);
    return 0;
}
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

#ifdef GRID_RBD_WITH_JAX

#include "xla/ffi/api/ffi.h"
namespace ffi = xla::ffi;

// inverse_dynamics(q, qd, qdd, f_ext) → c   — fully device-resident path.
//
// qdd and f_ext are ALWAYS passed as explicit device buffers from the Python
// surface (JAX FFI has no optional-buffer support, so the wrapper passes zeros
// when the caller omits them — mirroring idsva_so). qdd flows through the
// separate d_qdd buffer + the USE_QDD overload of the kernel (signature
// (d_c, d_q_qd, stride, d_qdd, d_f_ext, ...)); f_ext is copied D→D into d_f_ext.
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
    grid::inverse_dynamics_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_c, g_data->d_q_qd_u, stride_q_qd, g_data->d_qdd,
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
        .Arg<ffi::Buffer<ffi::F32>>()  // qdd
        .Arg<ffi::Buffer<ffi::F32>>()  // f_ext
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


// forward_dynamics(q, qd, u, f_ext) → qdd  (f_ext always passed; zeros if omitted)
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
    grid::forward_dynamics_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(),
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
        .Arg<ffi::Buffer<ffi::F32>>()  // f_ext
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);


// aba(q, qd, u, f_ext) → qdd  — same kernel signature shape as forward_dynamics
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
        .Arg<ffi::Buffer<ffi::F32>>()  // f_ext
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


// end_effector_pose(q) → end_effector_pose  flat (B, 6*NUM_EES)
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
        grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_end_effector_pose, g_data->d_q_qd_u, stride_q,
            g_robot, batch);

    cudaMemcpyAsync(ee_out->typed_data(), g_data->d_end_effector_pose,
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


// end_effector_pose_gradient(q) → end_effector_pose_gradient d/dv flat (B, 6*NUM_EES*NV).
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
        grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_end_effector_pose_gradient, g_data->d_workspace, g_data->d_q_qd_u, stride_q,
            g_robot, batch);

    cudaMemcpyAsync(dee_out->typed_data(), g_data->d_end_effector_pose_gradient,
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


// end_effector_pose_hessian(q) → end_effector_pose_hessian  flat (B, 6*NUM_EES*NV*NV)
// The kernel also writes d_end_effector_pose_gradient as a byproduct; we only return d2.
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
        grid::END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_end_effector_pose_hessian, g_data->d_end_effector_pose_gradient, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q, g_robot, batch);

    cudaMemcpyAsync(d2ee_out->typed_data(), g_data->d_end_effector_pose_hessian,
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


// inverse_dynamics_gradient(q, qd, qdd) → dc_du  flat (B, 2*NJ*NJ)
// Python reshapes/transposes to (B, NJ, 2*NJ) [dc_dq | dc_dqd].
//
// qdd is ALWAYS passed as an explicit device buffer from the Python surface
// (JAX FFI has no optional-buffer support, so the wrapper passes zeros when the
// caller omits it — mirroring the VALUE inverse_dynamics FFI). ∂c/∂(q,qd)
// depends on qdd via the M·qdd term's derivatives, so qdd flows through the
// separate d_qdd buffer + the USE_QDD overload of the gradient kernel
// (signature adds d_qdd after stride). A zero qdd is byte-identical to the old
// no-qdd behaviour.
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
    grid::inverse_dynamics_gradient_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_dc_du, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q_qd, g_data->d_qdd,
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
        .Arg<ffi::Buffer<ffi::F32>>()  // q
        .Arg<ffi::Buffer<ffi::F32>>()  // qd
        .Arg<ffi::Buffer<ffi::F32>>()  // qdd
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
        grid::FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(),
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


// idsva_so(q, qd, qdd) → packed (B, SECOND_ORDER_TENSOR_SIZE)
// The codegen-time dispatcher picks body- vs world-frame; we dispatch here at
// compile time using the GRID_GENERATES_* macros so a per-robot .so calls
// whichever kernel was emitted. qdd is packed into the acceleration (u) slot of
// d_q_qd_u, which the kernel reads as s_qdd — mirroring the numpy
// pack_q_qd_u(q, qd, qdd). The Python surface passes explicit zeros when the
// caller omits qdd, so the result never depends on a stale device buffer.
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
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
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

// inverse_dynamics_regressor(q, qd, qdd) → Y  flat (B, NV*10*NUM_BODIES).
// qdd is passed explicitly (the bias regressor used by inverse_dynamics's VJP
// passes zeros). The regressor kernel reads q|qd|qdd from d_q_qd_u (stride
// Q_QD_U_STRIDE), the qdd occupying the u-slot — mirroring idsva_so.
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
    grid::inverse_dynamics_regressor_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_Y, g_data->d_q_qd_u, stride_q_qd_qdd,
            g_robot, /*gravity=*/gravity, batch);

    cudaMemcpyAsync(Y_out->typed_data(), g_data->d_Y,
                    (size_t)batch * out_size * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_inverse_dynamics_regressor,
    grid_rbd_jax_inverse_dynamics_regressor_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("gravity")
);


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
        g_block_dimms, g_thread_dimms,
        grid::FORWARD_DYNAMICS_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_dqdd_dpi, g_data->d_workspace, g_data->d_q_qd_u, stride_q_qd_u,
            g_robot, /*gravity=*/gravity, batch);

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
template <bool STATE>
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    if (STATE) {
        grid_plant::quadratic_state_cost_kernel<T><<<grid_dim, g_thread_dimms, 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
    } else {
        grid_plant::quadratic_input_cost_kernel<T><<<grid_dim, g_thread_dimms, 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    if (WHICH == 0)
        grid_plant::joint_position_barrier_kernel<T><<<grid_dim, g_thread_dimms, 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    else if (WHICH == 1)
        grid_plant::joint_velocity_barrier_kernel<T><<<grid_dim, g_thread_dimms, 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    else
        grid_plant::joint_torque_barrier_kernel<T><<<grid_dim, g_thread_dimms, 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
            g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
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
template <grid::IntegratorType IT>
static void launch_plant_step_jax(cudaStream_t stream, int batch, float gravity, float dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::plant_step_kernel<T, IT><<<grid_dim, g_thread_dimms,
        grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, grid::NUM_VEL, g_robot, (T)gravity, (T)dt, batch);
}

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
    GRID_RBD_IT_DISPATCH_FFI((int)it, launch_plant_step_jax, stream, batch, gravity, dt);
    cudaMemcpyAsync(x_kp1->typed_data(), g_plant.d_grad, (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_plant_step,
    grid_rbd_jax_plant_step_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("dt").Attr<int64_t>("it").Attr<float>("gravity")
);
#endif  // GRID_PLANT_HAS_STEP

#ifdef GRID_PLANT_HAS_STEP_GRADIENT
// plant_step_gradient(x, u; dt, it) → dAB  (B, 2*NV*3*NV col-major). Reuses
// g_plant.d_grad as the dAB output (size 2*NV*3*NV), matching the C-ABI.
template <grid::IntegratorType IT>
static void launch_plant_step_gradient_jax(cudaStream_t stream, int batch, float gravity, float dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    const int nv = grid::NUM_VEL;
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::plant_step_gradient_kernel<T, IT><<<grid_dim, g_thread_dimms,
        grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, nv, g_robot, (T)gravity, (T)dt, batch);
}

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
    GRID_RBD_IT_DISPATCH_FFI((int)it, launch_plant_step_gradient_jax, stream, batch, gravity, dt);
    cudaMemcpyAsync(dAB->typed_data(), g_plant.d_grad, (size_t)batch * dab * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_plant_step_gradient,
    grid_rbd_jax_plant_step_gradient_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<float>("dt").Attr<int64_t>("it").Attr<float>("gravity")
);
#endif  // GRID_PLANT_HAS_STEP_GRADIENT

#ifdef GRID_PLANT_HAS_EE_COST
// ee_pos_cost(q, p_des, W) → (value (B,1), grad (B,NX), hess (B,NX*NX)).
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::ee_pos_cost_kernel<T, 0><<<grid_dim, g_thread_dimms, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose, g_plant.d_end_effector_pose_gradient, g_robot, batch);
    cudaMemcpyAsync(out->typed_data(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad->typed_data(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess->typed_data(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_PLANT_COST_BIND(grid_rbd_jax_plant_ee_pos_cost,
                             grid_rbd_jax_plant_ee_pos_cost_impl);
#endif  // GRID_PLANT_HAS_EE_COST

#ifdef GRID_PLANT_HAS_COM_COST
// com_cost(q, p_des, W) → (value (B,1), grad (B,NX), hess (B,NX*NX)).
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::com_cost_kernel<T><<<grid_dim, g_thread_dimms, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose, g_robot, batch);
    cudaMemcpyAsync(out->typed_data(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad->typed_data(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess->typed_data(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

GRID_RBD_JAX_PLANT_COST_BIND(grid_rbd_jax_plant_com_cost,
                             grid_rbd_jax_plant_com_cost_impl);
#endif  // GRID_PLANT_HAS_COM_COST

#ifdef GRID_PLANT_HAS_MOMENTUM_COST
// momentum_cost(q, qd, h_des, W) → (value (B,1), grad (B,NX), hess (B,NX*NX)).
// h_des(6) and W(6) are packed into the two halves of d_in_c, matching the C-ABI.
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::momentum_cost_kernel<T><<<grid_dim, g_thread_dimms, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,
        g_plant.d_in_a, g_plant.d_in_b,
        g_plant.d_in_c, g_plant.d_in_c + (size_t)batch * 6,
        g_plant.d_end_effector_pose, g_robot, batch);
    cudaMemcpyAsync(out->typed_data(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad->typed_data(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess->typed_data(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_plant_momentum_cost,
    grid_rbd_jax_plant_momentum_cost_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
);
#endif  // GRID_PLANT_HAS_MOMENTUM_COST

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

// qdd wiring (torch): when a qdd tensor is provided, copy it D→D into the
// separate d_qdd buffer and launch the USE_QDD overload of the kernel (which
// reads the acceleration from d_qdd; signature adds d_qdd after stride). A null
// qdd keeps the (faster) qdd=0 overload. Mirrors the numpy / JAX ID paths.
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
        grid::inverse_dynamics_kernel<T><<<g_block_dimms, g_thread_dimms, grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_c, g_data->d_q_qd_u, stride, g_data->d_qdd, g_data->d_f_ext, g_robot, (T)gravity, batch);
    } else {
        grid::inverse_dynamics_kernel<T><<<g_block_dimms, g_thread_dimms, grid::INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_c, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    }
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
    grid::forward_dynamics_kernel<T><<<g_block_dimms, g_thread_dimms, grid::FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
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
    grid::end_effector_pose_kernel<T><<<g_block_dimms, g_thread_dimms, grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_end_effector_pose, g_data->d_q_qd_u, stride, g_robot, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_end_effector_pose, batch * 6 * nee * sizeof(T), cudaMemcpyDeviceToDevice, stream);
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
    grid::end_effector_pose_gradient_kernel<T><<<g_block_dimms, g_thread_dimms, grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_end_effector_pose_gradient, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_end_effector_pose_gradient, batch * 6 * nee * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
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
    grid::end_effector_pose_hessian_kernel<T><<<g_block_dimms, g_thread_dimms, grid::END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_end_effector_pose_hessian, g_data->d_end_effector_pose_gradient, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_end_effector_pose_hessian, batch * 6 * nee * nv * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}

// qdd wiring (torch grad): ∂c/∂(q,qd) depends on qdd via the M·qdd term's
// derivatives. When a qdd tensor is provided, copy it D→D into d_qdd and launch
// the USE_QDD overload of the gradient kernel (signature adds d_qdd after
// stride). A null qdd keeps the (faster) qdd=0 overload — byte-identical to the
// prior behaviour. Mirrors the numpy / JAX ID-gradient paths and the order of
// the VALUE torch_inverse_dynamics op (q, qd, gravity, qdd, f_ext).
torch::Tensor torch_inverse_dynamics_gradient(torch::Tensor q, torch::Tensor qd, double gravity,
                              c10::optional<torch::Tensor> qdd,
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
    if (qdd.has_value()) {
        const torch::Tensor& a = qdd.value();
        grid_torch_check(a, "inverse_dynamics_gradient: qdd", nj);
        cudaMemcpyAsync(g_data->d_qdd, a.data_ptr<float>(),
                        (size_t)batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        grid::inverse_dynamics_gradient_kernel<T><<<g_block_dimms, g_thread_dimms, grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_dc_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_qdd, g_data->d_f_ext, g_robot, (T)gravity, batch);
    } else {
        grid::inverse_dynamics_gradient_kernel<T><<<g_block_dimms, g_thread_dimms, grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_data->d_dc_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    }
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
    grid::forward_dynamics_gradient_kernel<T><<<g_block_dimms, g_thread_dimms, grid::FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_df_du, g_data->d_workspace, g_data->d_q_qd_u, stride, g_data->d_f_ext, g_robot, (T)gravity, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_df_du, batch * nj * 2 * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    grid_torch_f_ext_reset(stream, batch, f_ext);
    return out;
}

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

// ── inertial-parameter (sysID) regressor + FD parameter gradient ──
// Mirror the JAX grid_rbd_jax_inverse_dynamics_regressor /
// grid_rbd_jax_forward_dynamics_parameter_gradient handlers: same kernels, same
// d_Y / d_dqdd_dpi / d_workspace scratch, same (B, NV*10*NUM_BODIES) row-major
// output. These back the torch inertial-parameter VJP (tau = Y . pi so
// dtau/dpi = Y, dqdd/dpi = -Minv . Y).

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
    grid::inverse_dynamics_regressor_kernel<T><<<g_block_dimms, g_thread_dimms, grid::INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_Y, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_Y, (size_t)batch * out_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}

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
    grid::forward_dynamics_parameter_gradient_kernel<T><<<g_block_dimms, g_thread_dimms, grid::FORWARD_DYNAMICS_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
        g_data->d_dqdd_dpi, g_data->d_workspace, g_data->d_q_qd_u, stride, g_robot, (T)gravity, batch);
    cudaMemcpyAsync(out.data_ptr<float>(), g_data->d_dqdd_dpi, (size_t)batch * out_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
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
// Non-template (runtime bool) so the torch::Tensor .data_ptr<float>() member-
// template calls parse unambiguously under nvcc's host pass.
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    if (state)
        grid_plant::quadratic_state_cost_kernel<T><<<grid_dim, g_thread_dimms, 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
    else
        grid_plant::quadratic_input_cost_kernel<T><<<grid_dim, g_thread_dimms, 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, batch);
    cudaMemcpyAsync(out.data_ptr<float>(),  g_plant.d_out,  (size_t)batch * sizeof(T),         cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<float>(), g_plant.d_grad, (size_t)batch * N * sizeof(T),     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess.data_ptr<float>(), g_plant.d_hess, (size_t)batch * N * N * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hess};
}

std::vector<torch::Tensor> torch_quadratic_state_cost(torch::Tensor x, torch::Tensor x_des, torch::Tensor Q) {
    return torch_plant_quadratic_cost(x, x_des, Q, true);
}
std::vector<torch::Tensor> torch_quadratic_input_cost(torch::Tensor u, torch::Tensor u_des, torch::Tensor R) {
    return torch_plant_quadratic_cost(u, u_des, R, false);
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    if (which == 0)
        grid_plant::joint_position_barrier_kernel<T><<<grid_dim, g_thread_dimms, 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    else if (which == 1)
        grid_plant::joint_velocity_barrier_kernel<T><<<grid_dim, g_thread_dimms, 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
    else
        grid_plant::joint_torque_barrier_kernel<T><<<grid_dim, g_thread_dimms, 0, stream>>>(
            g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c, (T)mu, batch);
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
template <grid::IntegratorType IT>
static void torch_launch_plant_step(cudaStream_t stream, int batch, double gravity, double dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL;
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::plant_step_kernel<T, IT><<<grid_dim, g_thread_dimms,
        grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, grid::NUM_VEL, g_robot, (T)gravity, (T)dt, batch);
}

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
    GRID_RBD_IT_DISPATCH_TORCH((int)it, torch_launch_plant_step, stream, batch, gravity, dt);
    cudaMemcpyAsync(out.data_ptr<float>(), g_plant.d_grad, (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_PLANT_HAS_STEP

#ifdef GRID_PLANT_HAS_STEP_GRADIENT
template <grid::IntegratorType IT>
static void torch_launch_plant_step_gradient(cudaStream_t stream, int batch, double gravity, double dt) {
    const int nx = grid::NUM_POS + grid::NUM_VEL, nv = grid::NUM_VEL;
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::plant_step_gradient_kernel<T, IT><<<grid_dim, g_thread_dimms,
        grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T>(), stream>>>(
            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,
            nx, nv, g_robot, (T)gravity, (T)dt, batch);
}

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
    GRID_RBD_IT_DISPATCH_TORCH((int)it, torch_launch_plant_step_gradient, stream, batch, gravity, dt);
    cudaMemcpyAsync(out.data_ptr<float>(), g_plant.d_grad, (size_t)batch * dab * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return out;
}
#endif  // GRID_PLANT_HAS_STEP_GRADIENT

#ifdef GRID_PLANT_HAS_EE_COST
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::ee_pos_cost_kernel<T, 0><<<grid_dim, g_thread_dimms, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose, g_plant.d_end_effector_pose_gradient, g_robot, batch);
    cudaMemcpyAsync(out.data_ptr<float>(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<float>(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess.data_ptr<float>(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hess};
}
#endif  // GRID_PLANT_HAS_EE_COST

#ifdef GRID_PLANT_HAS_COM_COST
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::com_cost_kernel<T><<<grid_dim, g_thread_dimms, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,
        g_plant.d_end_effector_pose, g_robot, batch);
    cudaMemcpyAsync(out.data_ptr<float>(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<float>(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess.data_ptr<float>(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hess};
}
#endif  // GRID_PLANT_HAS_COM_COST

#ifdef GRID_PLANT_HAS_MOMENTUM_COST
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
    dim3 grid_dim((unsigned)batch, 1, 1);
    grid_plant::momentum_cost_kernel<T><<<grid_dim, g_thread_dimms, smem, stream>>>(
        g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b,
        g_plant.d_in_c, g_plant.d_in_c + (size_t)batch * 6, g_plant.d_end_effector_pose, g_robot, batch);
    cudaMemcpyAsync(out.data_ptr<float>(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(grad.data_ptr<float>(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(hess.data_ptr<float>(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    return {out, grad, hess};
}
#endif  // GRID_PLANT_HAS_MOMENTUM_COST

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
    m.def("inverse_dynamics(Tensor q, Tensor qd, float gravity, Tensor? qdd=None, Tensor? f_ext=None) -> Tensor");
    m.def("minv(Tensor q) -> Tensor");
    m.def("forward_dynamics(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None) -> Tensor");
    m.def("aba(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None) -> Tensor");
    m.def("crba(Tensor q, float gravity) -> Tensor");
    m.def("end_effector_pose(Tensor q) -> Tensor");
    m.def("end_effector_pose_gradient(Tensor q) -> Tensor");
    m.def("end_effector_pose_hessian(Tensor q) -> Tensor");
    m.def("inverse_dynamics_gradient(Tensor q, Tensor qd, float gravity, Tensor? qdd=None, Tensor? f_ext=None) -> Tensor");
    m.def("forward_dynamics_gradient(Tensor q, Tensor qd, Tensor u, float gravity, Tensor? f_ext=None) -> Tensor");
    m.def("idsva_so(Tensor q, Tensor qd, Tensor qdd, float gravity) -> Tensor");
    m.def("fdsva_so(Tensor q, Tensor qd, Tensor u, float gravity) -> Tensor");
    m.def("inverse_dynamics_regressor(Tensor q, Tensor qd, Tensor qdd, float gravity) -> Tensor");
    m.def("forward_dynamics_parameter_gradient(Tensor q, Tensor qd, Tensor u, float gravity) -> Tensor");
    m.def("integrator(Tensor q, Tensor qd, Tensor u, float dt, int it, float gravity) -> Tensor");
    m.def("integrator_gradient(Tensor q, Tensor qd, Tensor u, float dt, int it, float gravity) -> Tensor");
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
    m.impl("inverse_dynamics_regressor", torch_inverse_dynamics_regressor);
    m.impl("forward_dynamics_parameter_gradient", torch_forward_dynamics_parameter_gradient);
    m.impl("integrator", torch_integrator);
    m.impl("integrator_gradient", torch_integrator_gradient);
    m.impl("quadratic_state_cost", torch_quadratic_state_cost);
    m.impl("quadratic_input_cost", torch_quadratic_input_cost);
    m.impl("joint_position_barrier", torch_joint_position_barrier);
    m.impl("joint_velocity_barrier", torch_joint_velocity_barrier);
    m.impl("joint_torque_barrier", torch_joint_torque_barrier);
#ifdef GRID_PLANT_HAS_STEP
    m.impl("plant_step", torch_plant_step);
#endif
#ifdef GRID_PLANT_HAS_STEP_GRADIENT
    m.impl("plant_step_gradient", torch_plant_step_gradient);
#endif
#ifdef GRID_PLANT_HAS_EE_COST
    m.impl("ee_pos_cost", torch_ee_pos_cost);
#endif
#ifdef GRID_PLANT_HAS_COM_COST
    m.impl("com_cost", torch_com_cost);
#endif
#ifdef GRID_PLANT_HAS_MOMENTUM_COST
    m.impl("momentum_cost", torch_momentum_cost);
#endif
}

#endif  // GRID_RBD_WITH_TORCH

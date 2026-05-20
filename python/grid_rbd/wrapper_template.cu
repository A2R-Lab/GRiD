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
// g_thread_dimms defaults to the codegen-time SUGGESTED_THREADS hint. After
// the v2.0 cuBLASDx removal there is no hard floor — callers can override
// via grid_rbd_set_threads_per_block() before issuing any kernel calls.
static dim3 g_thread_dimms = dim3(grid::SUGGESTED_THREADS, 1, 1);

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
extern "C" int grid_rbd_max_batch()      { return kMaxBatch; }
extern "C" int grid_rbd_suggested_threads() { return grid::SUGGESTED_THREADS; }
extern "C" int grid_rbd_threads_per_block() { return (int)g_thread_dimms.x; }
extern "C" int grid_rbd_set_threads_per_block(int n) {
    // Override the per-block thread count used for all subsequent kernel
    // launches. n must be >= 1; values larger than the per-block max
    // (1024 on current GPUs) will fail at launch time with cudaErrorInvalidConfiguration.
    // The default is grid::SUGGESTED_THREADS; the codegen no longer pins
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

// ─── algorithms ──────────────────────────────────────────────────────────────

// RNEA: c = M(q)·qdd + h(q,qd) − g(q)  (with qdd defaulting to 0 if null)
extern "C" int grid_rbd_rnea(
    const T* q, const T* qd, const T* qdd_opt,
    T* c_out,
    int batch, T gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;  // caller should chunk

    // For now, USE_QDD_FLAG=false; qdd defaults to 0 in the kernel. To support
    // user-provided qdd we'd need USE_QDD_FLAG=true and the qdd buffer pushed
    // separately. Document this in the Python layer.
    (void)qdd_opt;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, nullptr, batch, nj);

    grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/false, /*USE_COMPRESSED_MEM=*/false>(
        g_data, g_robot, gravity, batch, g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
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

    grid::direct_minv<T, /*USE_COMPRESSED_MEM=*/false>(
        g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(minv_out, g_data->h_Minv, batch * nj * nj * sizeof(T));
    return 0;
}

// Forward dynamics: qdd = Minv(q)·(τ − c(q,qd))
extern "C" int grid_rbd_forward_dynamics(
    const T* q, const T* qd, const T* u,
    T* qdd_out,
    int batch, T gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);

    grid::forward_dynamics<T>(
        g_data, g_robot, gravity, batch, g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(qdd_out, g_data->h_qdd, batch * nj * sizeof(T));
    return 0;
}

// Articulated body algorithm: qdd = aba(q, qd, u)
extern "C" int grid_rbd_aba(
    const T* q, const T* qd, const T* u,
    T* qdd_out,
    int batch, T gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);

    grid::aba<T>(g_data, g_robot, gravity, batch,
                 g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
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

// End-effector pose Jacobian: 6×NUM_EES×NUM_JOINTS per timestep.
extern "C" int grid_rbd_end_effector_pose_gradient(
    const T* q,
    T* dee_out,
    int batch)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, q, nullptr, batch, nj);

    grid::end_effector_pose_gradient<T, /*USE_COMPRESSED_MEM=*/false>(
        g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(dee_out, g_data->h_deePos,
                batch * 6 * grid::NUM_EES * nj * sizeof(T));
    return 0;
}

// ∂c/∂(q, qd): output shape (batch, NJ, 2*NJ) — concatenated [dc_dq | dc_dqd].
extern "C" int grid_rbd_rnea_grad(
    const T* q, const T* qd, const T* qdd_opt,
    T* dc_du_out,
    int batch, T gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;
    (void)qdd_opt;  // USE_QDD_FLAG=false currently; future v2

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, nullptr, batch, nj);

    grid::inverse_dynamics_gradient<T, /*USE_QDD_FLAG=*/false,
                                       /*USE_COMPRESSED_MEM=*/false>(
        g_data, g_robot, gravity, batch,
        g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(dc_du_out, g_data->h_dc_du,
                batch * nj * 2 * nj * sizeof(T));
    return 0;
}

// ∂qdd/∂(q, qd): output shape (batch, NJ, 2*NJ).
extern "C" int grid_rbd_forward_dynamics_grad(
    const T* q, const T* qd, const T* u,
    T* df_du_out,
    int batch, T gravity)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }
    if (batch > kMaxBatch) return 2;

    const int nj = grid::NUM_JOINTS;
    pack_q_qd_u(q, qd, u, batch, nj);

    grid::forward_dynamics_gradient<T, /*USE_QDD_MINV_FLAG=*/false>(
        g_data, g_robot, gravity, batch,
        g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(df_du_out, g_data->h_df_du,
                batch * nj * 2 * nj * sizeof(T));
    return 0;
}

// End-effector pose Hessian: 6×NUM_EES×NJ×NJ per timestep.
// Calls grid::end_effector_pose_gradient_hessian which fills BOTH d2eePos AND
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
    pack_q_qd_u(q, q, nullptr, batch, nj);

    grid::end_effector_pose_gradient_hessian<T, /*USE_COMPRESSED_MEM=*/false>(
        g_data, g_robot, batch, g_block_dimms, g_thread_dimms, g_streams);

    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    std::memcpy(d2ee_out, g_data->h_d2eePos,
                batch * 6 * grid::NUM_EES * nj * nj * sizeof(T));
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

// rnea(q, qd) → c   — fully device-resident path.
static ffi::Error grid_rbd_jax_rnea_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,         // shape (B, NJ), device-resident
    ffi::Buffer<ffi::F32> qd,        // shape (B, NJ), device-resident
    ffi::ResultBuffer<ffi::F32> c)   // shape (B, NJ), device-resident
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) {
        return ffi::Error::Internal("grid_rbd_init failed");
    }}
    auto q_shape = q.dimensions();
    if (q_shape.size() != 2) {
        return ffi::Error::InvalidArgument("rnea: q must be 2D (B, NJ)");
    }
    int batch = (int)q_shape[0];
    int nj    = (int)q_shape[1];
    if (nj != grid::NUM_JOINTS) {
        return ffi::Error::InvalidArgument("rnea: last dim != NUM_JOINTS");
    }
    if (batch > kMaxBatch) {
        return ffi::Error::InvalidArgument(
            "rnea: batch exceeds compiled-in max_batch_size; recompile with a larger value");
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
            g_robot, /*gravity=*/9.81f, batch);

    // D→D copy the result into JAX's output buffer on the same stream.
    cudaMemcpyAsync(c->typed_data(), g_data->d_c,
                    batch * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_rnea,
    grid_rbd_jax_rnea_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()  // q
        .Arg<ffi::Buffer<ffi::F32>>()  // qd
        .Ret<ffi::Buffer<ffi::F32>>()  // c
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
    grid::direct_minv_kernel<T><<<
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
    ffi::ResultBuffer<ffi::F32> qdd_out)
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
            g_robot, /*gravity=*/9.81f, batch);

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
);


// aba(q, qd, u) → qdd  — same kernel signature shape as forward_dynamics
static ffi::Error grid_rbd_jax_aba_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> u,
    ffi::ResultBuffer<ffi::F32> qdd_out)
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
            g_robot, /*gravity=*/9.81f, batch);

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
);


// crba(q) → M  (kernel writes the full mass matrix; no symmetrize needed)
static ffi::Error grid_rbd_jax_crba_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> m_out)
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
            g_data->d_M, g_data->d_q_qd_u, stride_q_qd,
            g_robot, /*gravity=*/9.81f, batch);

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


// end_effector_pose_gradient(q) → deePos  flat (B, 6*NUM_EES*NJ)
// Python side reshapes/transposes to the (B, 6*NUM_EES, NJ) row-major
// convention (see _handle.py:end_effector_pose_gradient).
static ffi::Error grid_rbd_jax_end_effector_pose_gradient_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::ResultBuffer<ffi::F32> dee_out)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "end_effector_pose_gradient: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
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
            g_data->d_deePos, g_data->d_q_qd_u, stride_q,
            g_robot, batch);

    cudaMemcpyAsync(dee_out->typed_data(), g_data->d_deePos,
                    batch * 6 * grid::NUM_EES * nj * sizeof(T),
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


// end_effector_pose_hessian(q) → d2eePos  flat (B, 6*NUM_EES*NJ*NJ)
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
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("end_effector_pose_hessian: batch > max_batch");

    const size_t row_bytes = nj * sizeof(T);
    const size_t dst_pitch = 3 * nj * sizeof(T);
    cudaMemcpy2DAsync(&g_data->d_q_qd_u[0], dst_pitch,
                      q.typed_data(),       row_bytes,
                      row_bytes, batch, cudaMemcpyDeviceToDevice, stream);

    constexpr int stride_q = 3 * grid::NUM_JOINTS;
    grid::end_effector_pose_gradient_hessian_kernel<T><<<
        g_block_dimms, g_thread_dimms,
        grid::D2EE_POS_DYNAMIC_SHARED_MEM_BYTES<T>(),
        stream>>>(
            g_data->d_d2eePos, g_data->d_deePos, g_data->d_workspace,
            g_data->d_q_qd_u, stride_q, g_robot, batch);

    cudaMemcpyAsync(d2ee_out->typed_data(), g_data->d_d2eePos,
                    batch * 6 * grid::NUM_EES * nj * nj * sizeof(T),
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


// rnea_grad(q, qd) → dc_du  flat (B, 2*NJ*NJ)
// Python reshapes/transposes to (B, NJ, 2*NJ) [dc_dq | dc_dqd].
// USE_QDD_FLAG=false for now; qdd defaults to 0 in-kernel.
static ffi::Error grid_rbd_jax_rnea_grad_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::ResultBuffer<ffi::F32> dc_du_out)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "rnea_grad: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("rnea_grad: batch > max_batch");

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
            g_robot, /*gravity=*/9.81f, batch);

    cudaMemcpyAsync(dc_du_out->typed_data(), g_data->d_dc_du,
                    batch * nj * 2 * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_rnea_grad,
    grid_rbd_jax_rnea_grad_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);


// forward_dynamics_grad(q, qd, u) → df_du  flat (B, 2*NJ*NJ)
// Python reshapes/transposes to (B, NJ, 2*NJ).
static ffi::Error grid_rbd_jax_forward_dynamics_grad_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> u,
    ffi::ResultBuffer<ffi::F32> df_du_out)
{
    if (!g_data) { int rc = grid_rbd_init(); if (rc) return ffi::Error::Internal("init failed"); }
    GRID_RBD_FFI_VALIDATE_2D(q, "forward_dynamics_grad: q", grid::NUM_JOINTS);
    int batch = (int)q.dimensions()[0];
    int nj    = grid::NUM_JOINTS;
    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("forward_dynamics_grad: batch > max_batch");

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
            g_robot, /*gravity=*/9.81f, batch);

    cudaMemcpyAsync(df_du_out->typed_data(), g_data->d_df_du,
                    batch * nj * 2 * nj * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    grid_rbd_jax_forward_dynamics_grad,
    grid_rbd_jax_forward_dynamics_grad_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
);


// idsva_so(q, qd) → packed (B, SECOND_ORDER_TENSOR_SIZE)
// USE_QDD_FLAG=false. The codegen-time dispatcher picks body- vs world-frame;
// we dispatch here at compile time using the GRID_GENERATES_* macros so a
// per-robot .so calls whichever kernel was emitted.
static ffi::Error grid_rbd_jax_idsva_so_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::ResultBuffer<ffi::F32> out)
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
            g_robot, /*gravity=*/9.81f, batch);

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
);


// fdsva_so(q, qd, u) → packed (B, SECOND_ORDER_TENSOR_SIZE)
// Uses d_idsva_so as scratch — must not run concurrently with idsva_so.
static ffi::Error grid_rbd_jax_fdsva_so_impl(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> qd,
    ffi::Buffer<ffi::F32> u,
    ffi::ResultBuffer<ffi::F32> out)
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
            g_data->d_df2, g_data->d_q_qd_u, stride_q_qd_u,
            g_data->d_workspace, g_data->d_idsva_so,
            g_robot, /*gravity=*/9.81f, batch);

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
);

#endif  // GRID_RBD_WITH_JAX

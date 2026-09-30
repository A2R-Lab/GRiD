// Pure C++/CUDA timing of the generated `grid.cuh` host entry points of the
// SAME artifact the Python wrappers load (the header next to robot.so in the
// build cache), with no Python and no binding layer in the loop.
//
// Two boundaries per cell, both ending in a device synchronization:
//   compute  : the `<op>_compute_only` host function — kernel launch on
//              inputs already resident in the gridData arena, output left on
//              the device. This is the kernel latency GRiD's papers report.
//   with_mem : the `<op>` host function — H2D copies of the packed inputs,
//              the kernel, and the D2H copy of the result (GRiD's own
//              synchronous host call, exactly what the C ABI wraps).
// Differences between the collector's backends then have one meaning each:
//   with_mem - compute        = memory traffic of one call
//   C ABI    - with_mem       = context lookup, staging memcpy, fences
//   NumPy    - C ABI          = Python/pybind
//   JAX/torch resident - compute = framework dispatch
//
// Compiled per (robot, operation) artifact against that artifact's header
// with the wrapper's nvcc flags (same arch, same fast-math settings), so the
// SASS is the artifact's SASS; the worker checks the output bitwise against
// the NumPy wrapper before any timing is kept.
#include <chrono>
#include <cstdio>
#include <cstring>
#include "grid.cuh"

#ifndef GRID_KERNEL_MAX_BATCH
#define GRID_KERNEL_MAX_BATCH 256
#endif
// Poses written per timestep by the artifact's end-effector kernel: one for a
// fixed-target build (grid::NUM_EES still counts every leaf), passed by the
// adapter from the artifact metadata like the wrapper's GRID_RBD_NUM_EES.
#ifndef GRID_KERNEL_NUM_EES
#define GRID_KERNEL_NUM_EES grid::NUM_EES
#endif

using T = float;

namespace {

struct KernelCtx {
    cudaStream_t *streams;
    grid::robotModel<T> *model;
    grid::gridData<T> *data;
};

double now_us() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Operation codes = the collector's OPS order (pin_adapter.OPS / grid_adapter.KERNEL_OPS).
constexpr int OP_ID = 0, OP_ID_GRAD = 1, OP_IDSVA_SO = 2, OP_MINV = 3, OP_FD = 4, OP_FD_GRAD = 5,
              OP_FDSVA_SO = 6, OP_EE_POSE = 7, OP_CRBA = 8, OP_NLE = 9, OP_GRAVITY = 10, OP_CCRBA = 11, OP_CORIOLIS = 12;
#define KB_CAT_(a, b) a##b
#define KB_CAT(a, b) KB_CAT_(a, b)
#if !defined(GRID_HAS_IDSVA_SO)
#define GRID_HAS_IDSVA_SO 0
#endif
#if !defined(GRID_HAS_FDSVA_SO)
#define GRID_HAS_FDSVA_SO 0
#endif

bool op_built(int op) {
    switch (op) {
#if GRID_HAS_INVERSE_DYNAMICS
        case OP_ID: return true;
#endif
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
        case OP_ID_GRAD: return true;
#endif
#if GRID_HAS_IDSVA_SO
        case OP_IDSVA_SO: return true;
#endif
#if GRID_HAS_MINV
        case OP_MINV: return true;
#endif
#if GRID_HAS_FORWARD_DYNAMICS
        case OP_FD: return true;
#endif
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
        case OP_FD_GRAD: return true;
#endif
#if GRID_HAS_FDSVA_SO
        case OP_FDSVA_SO: return true;
#endif
#if GRID_HAS_END_EFFECTOR_POSE && defined(GRID_RBD_EE_POSE_FN)
        case OP_EE_POSE: return true;
#endif
#if GRID_HAS_CRBA
        case OP_CRBA: return true;
#endif
#if GRID_HAS_NONLINEAR_EFFECTS
        case OP_NLE: return true;
#endif
#if GRID_HAS_GENERALIZED_GRAVITY
        case OP_GRAVITY: return true;
#endif
#if GRID_HAS_CCRBA
        case OP_CCRBA: return true;
#endif
#if GRID_HAS_CORIOLIS_MATRIX
        case OP_CORIOLIS: return true;
#endif
        default: return false;
    }
}

int output_size(int op) {
    const int nj = grid::NUM_JOINTS, nv = grid::NUM_VEL;
    switch (op) {
        case OP_ID: case OP_FD: return nj;
        case OP_ID_GRAD: case OP_FD_GRAD: return 2 * nv * nv;
#if GRID_HAS_IDSVA_SO || GRID_HAS_FDSVA_SO
        case OP_IDSVA_SO: case OP_FDSVA_SO: return grid::SECOND_ORDER_TENSOR_SIZE;
#endif
        case OP_MINV: case OP_CRBA: case OP_CORIOLIS: return nv * nv;
        case OP_EE_POSE: return 6 * GRID_KERNEL_NUM_EES;
        case OP_NLE: case OP_GRAVITY: return nv;
        case OP_CCRBA: return 6 * nv + 6;
        default: return 0;
    }
}

int baked_threads(int op) {
    switch (op) {
        case OP_ID: return grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::THREADS;
        case OP_ID_GRAD: return grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::THREADS;
        case OP_IDSVA_SO: return grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO>::THREADS;
        case OP_MINV: return grid::launch_cfg<grid::GRID_ALGO_MINV>::THREADS;
        case OP_FD: return grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::THREADS;
        case OP_FD_GRAD: return grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::THREADS;
        case OP_FDSVA_SO: return grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::THREADS;
        case OP_EE_POSE: return grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE>::THREADS;
        case OP_CRBA: return grid::launch_cfg<grid::GRID_ALGO_CRBA>::THREADS;
        case OP_NLE: return grid::launch_cfg<grid::GRID_ALGO_NONLINEAR_EFFECTS>::THREADS;
        case OP_GRAVITY: return grid::launch_cfg<grid::GRID_ALGO_GENERALIZED_GRAVITY>::THREADS;
        case OP_CCRBA: return grid::launch_cfg<grid::GRID_ALGO_CCRBA>::THREADS;
        case OP_CORIOLIS: return grid::launch_cfg<grid::GRID_ALGO_CORIOLIS_MATRIX>::THREADS;
        default: return 0;
    }
}

// Like the wrapper: clamp to a kernel's register-limited block cap where the
// wrapper does (ccrba); a failed attribute query keeps the request.
template <typename KernelPtr>
dim3 clamp_threads(KernelPtr kernel, dim3 requested) {
    cudaFuncAttributes attr;
    if (cudaFuncGetAttributes(&attr, (const void*)kernel) != cudaSuccess) { cudaGetLastError(); return requested; }
    unsigned cap = attr.maxThreadsPerBlock > 0 ? (unsigned)attr.maxThreadsPerBlock : requested.x;
    if (requested.x > cap) requested.x = cap;
    return requested;
}

// Same template arguments as the wrapper's C ABI bodies (wrapper_template.cu):
// USE_QDD_FLAG=true (explicit acceleration), uncompressed memory, the per-algo
// autotuned tier, and the MUJOCO_OUTPUT=false slot only where the header's
// launcher carries it (-DGRID_RBD_SIG_MJX_* derived from the header itself).
#if GRID_HAS_INVERSE_DYNAMICS
#if defined(GRID_RBD_SIG_MJX_INVERSE_DYNAMICS)
#define KB_ID_ARGS T, true, false, grid::GRID_DATA_ALL, false, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER
#else
#define KB_ID_ARGS T, true, false, grid::GRID_DATA_ALL, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::TIER
#endif
#endif
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
#if defined(GRID_RBD_SIG_MJX_INVERSE_DYNAMICS_GRADIENT)
#define KB_GRAD_ARGS T, true, false, grid::GRID_DATA_ALL, false, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER
#else
#define KB_GRAD_ARGS T, true, false, grid::GRID_DATA_ALL, grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::TIER
#endif
#endif
#if GRID_HAS_IDSVA_SO
#if defined(GRID_RBD_SIG_MJX_IDSVA_SO)
#define KB_SO_ARGS T, grid::GRID_DATA_ALL, false, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO>::TIER
#else
#define KB_SO_ARGS T, grid::GRID_DATA_ALL, grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO>::TIER
#endif
#endif

#if GRID_HAS_MINV
#if defined(GRID_RBD_SIG_MJX_MINV)
#define KB_MINV_ARGS T, false, grid::GRID_DATA_ALL, false, grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER
#else
#define KB_MINV_ARGS T, false, grid::GRID_DATA_ALL, grid::launch_cfg<grid::GRID_ALGO_MINV>::TIER
#endif
#endif
#if GRID_HAS_CRBA
#if defined(GRID_RBD_SIG_MJX_CRBA)
#define KB_CRBA_ARGS T, false, grid::GRID_DATA_ALL, false, grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER
#else
#define KB_CRBA_ARGS T, false, grid::GRID_DATA_ALL, grid::launch_cfg<grid::GRID_ALGO_CRBA>::TIER
#endif
#endif
#if GRID_HAS_FORWARD_DYNAMICS
#if defined(GRID_RBD_SIG_MJX_FORWARD_DYNAMICS)
#define KB_FD_ARGS T, grid::GRID_DATA_ALL, false, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER
#else
#define KB_FD_ARGS T, grid::GRID_DATA_ALL, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS>::TIER
#endif
#endif
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
#if defined(GRID_RBD_SIG_MJX_FORWARD_DYNAMICS_GRADIENT)
#define KB_FDG_ARGS T, false, grid::GRID_DATA_ALL, false, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER
#else
#define KB_FDG_ARGS T, false, grid::GRID_DATA_ALL, grid::launch_cfg<grid::GRID_ALGO_FORWARD_DYNAMICS_GRADIENT>::TIER
#endif
#endif
#if GRID_HAS_FDSVA_SO
#if defined(GRID_RBD_SIG_MJX_FDSVA_SO)
#define KB_FDSO_ARGS T, grid::GRID_DATA_ALL, false, grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER
#else
#define KB_FDSO_ARGS T, grid::GRID_DATA_ALL, grid::launch_cfg<grid::GRID_ALGO_FDSVA_SO>::TIER
#endif
#endif
#if GRID_HAS_END_EFFECTOR_POSE && defined(GRID_RBD_EE_POSE_FN)
#if defined(GRID_RBD_SIG_MJX_EE_POSE)
#define KB_EE_ARGS T, false, grid::GRID_DATA_ALL, false, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE>::TIER
#else
#define KB_EE_ARGS T, false, grid::GRID_DATA_ALL, grid::launch_cfg<grid::GRID_ALGO_END_EFFECTOR_POSE>::TIER
#endif
#define KB_EE_FN grid::GRID_RBD_EE_POSE_FN
#define KB_EE_FN_CO grid::KB_CAT(GRID_RBD_EE_POSE_FN, _compute_only)
#endif

void with_mem(KernelCtx &c, int op, T gravity, int n, dim3 thr) {
    const dim3 blocks((unsigned)n, 1, 1);
    switch (op) {
#if GRID_HAS_INVERSE_DYNAMICS
        case OP_ID: grid::inverse_dynamics<KB_ID_ARGS>(c.data, c.model, gravity, n, blocks, thr, c.streams); break;
#endif
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
        case OP_ID_GRAD: grid::inverse_dynamics_gradient<KB_GRAD_ARGS>(c.data, c.model, gravity, n, blocks, thr, c.streams); break;
#endif
#if GRID_HAS_IDSVA_SO
        case OP_IDSVA_SO: grid::idsva_so<KB_SO_ARGS>(c.data, c.model, gravity, n, blocks, thr, c.streams); break;
#endif
#if GRID_HAS_MINV
        case OP_MINV: grid::minv<KB_MINV_ARGS>(c.data, c.model, n, blocks, thr, c.streams); break;
#endif
#if GRID_HAS_FORWARD_DYNAMICS
        case OP_FD: grid::forward_dynamics<KB_FD_ARGS>(c.data, c.model, gravity, n, blocks, thr, c.streams); break;
#endif
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
        case OP_FD_GRAD: grid::forward_dynamics_gradient<KB_FDG_ARGS>(c.data, c.model, gravity, n, blocks, thr, c.streams); break;
#endif
#if GRID_HAS_FDSVA_SO
        case OP_FDSVA_SO: grid::fdsva_so<KB_FDSO_ARGS>(c.data, c.model, gravity, n, blocks, thr, c.streams); break;
#endif
#if GRID_HAS_END_EFFECTOR_POSE && defined(GRID_RBD_EE_POSE_FN)
        case OP_EE_POSE: KB_EE_FN<KB_EE_ARGS>(c.data, c.model, n, blocks, thr, c.streams); break;
#endif
#if GRID_HAS_CRBA
        case OP_CRBA: grid::crba<KB_CRBA_ARGS>(c.data, c.model, gravity, n, blocks, thr, c.streams); break;
#endif
#if GRID_HAS_NONLINEAR_EFFECTS
        case OP_NLE: grid::nonlinear_effects<T>(c.data, c.model, gravity, n, blocks, thr, c.streams); break;
#endif
#if GRID_HAS_GENERALIZED_GRAVITY
        case OP_GRAVITY: grid::generalized_gravity<T>(c.data, c.model, gravity, n, blocks, thr, c.streams); break;
#endif
#if GRID_HAS_CCRBA
        case OP_CCRBA: grid::ccrba<T>(c.data, c.model, n, blocks, clamp_threads(grid::ccrba_kernel<T>, thr), c.streams); break;
#endif
#if GRID_HAS_CORIOLIS_MATRIX
        case OP_CORIOLIS: grid::coriolis_matrix<T>(c.data, c.model, gravity, n, blocks, thr, c.streams); break;
#endif
        default: break;
    }
}

void compute_only(KernelCtx &c, int op, T gravity, int n, dim3 thr) {
    const dim3 blocks((unsigned)n, 1, 1);
    switch (op) {
#if GRID_HAS_INVERSE_DYNAMICS
        case OP_ID: grid::inverse_dynamics_compute_only<KB_ID_ARGS>(c.data, c.model, gravity, n, blocks, thr); break;
#endif
#if GRID_HAS_INVERSE_DYNAMICS_GRADIENT
        case OP_ID_GRAD: grid::inverse_dynamics_gradient_compute_only<KB_GRAD_ARGS>(c.data, c.model, gravity, n, blocks, thr); break;
#endif
#if GRID_HAS_IDSVA_SO
        case OP_IDSVA_SO: grid::idsva_so_compute_only<KB_SO_ARGS>(c.data, c.model, gravity, n, blocks, thr); break;
#endif
#if GRID_HAS_MINV
        case OP_MINV: grid::minv_compute_only<KB_MINV_ARGS>(c.data, c.model, n, blocks, thr); break;
#endif
#if GRID_HAS_FORWARD_DYNAMICS
        case OP_FD: grid::forward_dynamics_compute_only<KB_FD_ARGS>(c.data, c.model, gravity, n, blocks, thr); break;
#endif
#if GRID_HAS_FORWARD_DYNAMICS_GRADIENT
        case OP_FD_GRAD: grid::forward_dynamics_gradient_compute_only<KB_FDG_ARGS>(c.data, c.model, gravity, n, blocks, thr); break;
#endif
#if GRID_HAS_FDSVA_SO
        case OP_FDSVA_SO: grid::fdsva_so_compute_only<KB_FDSO_ARGS>(c.data, c.model, gravity, n, blocks, thr); break;
#endif
#if GRID_HAS_END_EFFECTOR_POSE && defined(GRID_RBD_EE_POSE_FN)
        case OP_EE_POSE: KB_EE_FN_CO<KB_EE_ARGS>(c.data, c.model, n, blocks, thr); break;
#endif
#if GRID_HAS_CRBA
        case OP_CRBA: grid::crba_compute_only<KB_CRBA_ARGS>(c.data, c.model, gravity, n, blocks, thr); break;
#endif
#if GRID_HAS_NONLINEAR_EFFECTS
        case OP_NLE: grid::nonlinear_effects_compute_only<T>(c.data, c.model, gravity, n, blocks, thr); break;
#endif
#if GRID_HAS_GENERALIZED_GRAVITY
        case OP_GRAVITY: grid::generalized_gravity_compute_only<T>(c.data, c.model, gravity, n, blocks, thr); break;
#endif
#if GRID_HAS_CCRBA
        case OP_CCRBA: grid::ccrba_compute_only<T>(c.data, c.model, n, blocks, clamp_threads(grid::ccrba_kernel<T>, thr)); break;
#endif
#if GRID_HAS_CORIOLIS_MATRIX
        case OP_CORIOLIS: grid::coriolis_matrix_compute_only<T>(c.data, c.model, gravity, n, blocks, thr); break;
#endif
        default: break;
    }
}

const T *host_output(const KernelCtx &c, int op) {
    switch (op) {
        case OP_ID: case OP_NLE: case OP_GRAVITY: return c.data->h_c;
        case OP_ID_GRAD: return c.data->h_dc_du;
#if GRID_HAS_IDSVA_SO
        case OP_IDSVA_SO: return c.data->h_idsva_so;
#endif
#if GRID_HAS_FDSVA_SO
        case OP_FDSVA_SO: return c.data->h_df2;
#endif
        case OP_MINV: return c.data->h_Minv;
        case OP_FD: return c.data->h_qdd;
        case OP_FD_GRAD: return c.data->h_df_du;
        case OP_EE_POSE: return c.data->h_end_effector_pose;
        case OP_CRBA: return c.data->h_M;
        case OP_CCRBA: return c.data->h_ccrba;
        case OP_CORIOLIS: return c.data->h_coriolis;
        default: return nullptr;
    }
}

const T *device_output(const KernelCtx &c, int op) {
    switch (op) {
        case OP_ID: case OP_NLE: case OP_GRAVITY: return c.data->d_c;
        case OP_ID_GRAD: return c.data->d_dc_du;
#if GRID_HAS_IDSVA_SO
        case OP_IDSVA_SO: return c.data->d_idsva_so;
#endif
#if GRID_HAS_FDSVA_SO
        case OP_FDSVA_SO: return c.data->d_df2;
#endif
        case OP_MINV: return c.data->d_Minv;
        case OP_FD: return c.data->d_qdd;
        case OP_FD_GRAD: return c.data->d_df_du;
        case OP_EE_POSE: return c.data->d_end_effector_pose;
        case OP_CRBA: return c.data->d_M;
        case OP_CCRBA: return c.data->d_ccrba;
        case OP_CORIOLIS: return c.data->d_coriolis;
        default: return nullptr;
    }
}

// Pack exactly like the wrapper's pack_q_qd_u: per timestep [q | qd | u] with
// stride 3*NUM_JOINTS. The u slot carries the torque (forward-dynamics family)
// or the acceleration (idsva_so); RNEA / grad RNEA take the acceleration via
// h_qdd (USE_QDD_FLAG); q-only operations mirror q into the qd slot like the
// wrapper does. Then stage the packed inputs on the device once so the
// compute-only path has resident inputs.
int stage_inputs(KernelCtx &c, int op, const T *q, const T *qd, const T *third, int batch) {
    const int nj = grid::NUM_JOINTS, stride = 3 * nj;
    const bool u_slot = (op == OP_IDSVA_SO || op == OP_FD || op == OP_FD_GRAD || op == OP_FDSVA_SO);
    const bool qdd_slot = (op == OP_ID || op == OP_ID_GRAD);
    for (int t = 0; t < batch; ++t) {
        std::memcpy(&c.data->h_q_qd_u[t * stride], &q[t * nj], nj * sizeof(T));
        std::memcpy(&c.data->h_q_qd_u[t * stride + nj], &qd[t * nj], nj * sizeof(T));
        if (u_slot) std::memcpy(&c.data->h_q_qd_u[t * stride + 2 * nj], &third[t * nj], nj * sizeof(T));
        else std::memset(&c.data->h_q_qd_u[t * stride + 2 * nj], 0, nj * sizeof(T));
    }
    if (qdd_slot) std::memcpy(c.data->h_qdd, third, (size_t)batch * nj * sizeof(T));
    cudaError_t e = cudaMemcpy(c.data->d_q_qd_u, c.data->h_q_qd_u, (size_t)batch * stride * sizeof(T), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return 100 + (int)e;
    if (qdd_slot) {
        e = cudaMemcpy(c.data->d_qdd, c.data->h_qdd, (size_t)batch * nj * sizeof(T), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) return 100 + (int)e;
    }
    e = cudaDeviceSynchronize();
    return e == cudaSuccess ? 0 : 100 + (int)e;
}

int download(const KernelCtx &c, int op, int batch, T *out) {
    const T *src = device_output(c, op);
    if (!src) return -2;
    cudaError_t e = cudaMemcpy(out, src, (size_t)batch * output_size(op) * sizeof(T), cudaMemcpyDeviceToHost);
    return e == cudaSuccess ? 0 : 100 + (int)e;
}

}  // namespace

extern "C" int grid_kernel_num_joints() { return grid::NUM_JOINTS; }
extern "C" int grid_kernel_num_vel() { return grid::NUM_VEL; }
extern "C" int grid_kernel_max_batch() { return GRID_KERNEL_MAX_BATCH; }
extern "C" int grid_kernel_op_built(int op) { return op_built(op) ? 1 : 0; }
extern "C" int grid_kernel_output_size(int op) { return output_size(op); }

extern "C" void *grid_kernel_create() {
    KernelCtx *c = new KernelCtx;
    c->streams = grid::init_grid<T>();
    c->model = grid::init_robotModel<T>();
    c->data = grid::init_gridData<T, GRID_KERNEL_MAX_BATCH>();
    if (!c->streams || !c->model || !c->data) { delete c; return nullptr; }
    return c;
}

extern "C" void grid_kernel_close(void *h) {
    if (!h) return;
    KernelCtx *c = static_cast<KernelCtx *>(h);
    grid::close_grid<T>(c->streams, c->model, c->data);
    delete c;
}

// One compute-only call (resident inputs) and the D2H of its output; used for
// the post-timing repeatability / oracle checks of the compute boundary.
extern "C" int grid_kernel_run(void *h, int op, const T *q, const T *qd, const T *third,
                               int batch, int threads, T gravity, T *out_compute) {
    if (!h || !q || !qd || !third || !out_compute || batch < 1 || batch > GRID_KERNEL_MAX_BATCH) return -1;
    if (!op_built(op)) return -2;
    KernelCtx &c = *static_cast<KernelCtx *>(h);
    if (int rc = stage_inputs(c, op, q, qd, third, batch)) return rc;
    const dim3 thr((unsigned)(threads > 0 ? threads : baked_threads(op)), 1, 1);
    compute_only(c, op, gravity, batch, thr);
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) e = cudaGetLastError();
    if (e != cudaSuccess) return 100 + (int)e;
    return download(c, op, batch, out_compute);
}

// Warm the device for at least `warm_seconds` (and `warmups` calls), then time
// `iterations` with_mem calls and `iterations` compute-only calls, each
// bracketed by a device synchronization. Returns the with_mem output (host
// buffer of the last with_mem call) and the compute output (D2H after the
// compute loop) so the caller can check both bitwise against the wrapper.
extern "C" int grid_kernel_time(void *h, int op, const T *q, const T *qd, const T *third,
                                int batch, int threads, T gravity, double warm_seconds,
                                int warmups, int iterations,
                                double *with_mem_us, double *compute_us,
                                T *out_with_mem, T *out_compute, int *threads_used) {
    if (!h || !q || !qd || !third || !with_mem_us || !compute_us || !out_with_mem || !out_compute
        || !threads_used || batch < 1 || batch > GRID_KERNEL_MAX_BATCH || warmups < 1 || iterations < 1
        || warm_seconds < 0) return -1;
    if (!op_built(op)) return -2;
    KernelCtx &c = *static_cast<KernelCtx *>(h);
    if (int rc = stage_inputs(c, op, q, qd, third, batch)) return rc;
    const dim3 thr((unsigned)(threads > 0 ? threads : baked_threads(op)), 1, 1);
    *threads_used = (int)thr.x;

    // Time-based warm-up: a fixed handful of microsecond calls never leaves
    // the idle clock; sustain calls until the device has reached its steady
    // boost state (see timeGRiD_common.h for the measured rationale).
    const double warm_start = now_us();
    int done = 0;
    do {
        with_mem(c, op, gravity, batch, thr);
        ++done;
    } while (done < warmups || now_us() - warm_start < warm_seconds * 1e6);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return 100 + (int)e;

    for (int i = 0; i < iterations; ++i) {
        const double t0 = now_us();
        with_mem(c, op, gravity, batch, thr);
        cudaDeviceSynchronize();
        with_mem_us[i] = now_us() - t0;
    }
    std::memcpy(out_with_mem, host_output(c, op), (size_t)batch * output_size(op) * sizeof(T));

    for (int i = 0; i < warmups; ++i) compute_only(c, op, gravity, batch, thr);
    cudaDeviceSynchronize();
    for (int i = 0; i < iterations; ++i) {
        const double t0 = now_us();
        compute_only(c, op, gravity, batch, thr);
        cudaDeviceSynchronize();
        compute_us[i] = now_us() - t0;
    }
    e = cudaGetLastError();
    if (e != cudaSuccess) return 100 + (int)e;
    return download(c, op, batch, out_compute);
}

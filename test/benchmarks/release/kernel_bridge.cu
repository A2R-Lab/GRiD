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

constexpr int OP_ID = 0, OP_ID_GRAD = 1, OP_IDSVA_SO = 2;

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
        default: return false;
    }
}

int output_size(int op) {
    switch (op) {
        case OP_ID: return grid::NUM_JOINTS;
        case OP_ID_GRAD: return 2 * grid::NUM_VEL * grid::NUM_VEL;
#if GRID_HAS_IDSVA_SO
        case OP_IDSVA_SO: return grid::SECOND_ORDER_TENSOR_SIZE;
#endif
        default: return 0;
    }
}

int baked_threads(int op) {
    switch (op) {
        case OP_ID: return grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS>::THREADS;
        case OP_ID_GRAD: return grid::launch_cfg<grid::GRID_ALGO_INVERSE_DYNAMICS_GRADIENT>::THREADS;
        case OP_IDSVA_SO: return grid::launch_cfg<grid::GRID_ALGO_IDSVA_SO>::THREADS;
        default: return 0;
    }
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
        default: break;
    }
}

const T *host_output(const KernelCtx &c, int op) {
    switch (op) {
        case OP_ID: return c.data->h_c;
        case OP_ID_GRAD: return c.data->h_dc_du;
#if GRID_HAS_IDSVA_SO
        case OP_IDSVA_SO: return c.data->h_idsva_so;
#endif
        default: return nullptr;
    }
}

const T *device_output(const KernelCtx &c, int op) {
    switch (op) {
        case OP_ID: return c.data->d_c;
        case OP_ID_GRAD: return c.data->d_dc_du;
#if GRID_HAS_IDSVA_SO
        case OP_IDSVA_SO: return c.data->d_idsva_so;
#endif
        default: return nullptr;
    }
}

// Pack exactly like the wrapper's pack_q_qd_u: per timestep [q | qd | u] with
// stride 3*NUM_JOINTS; the acceleration goes to the u slot for idsva_so and
// to h_qdd (USE_QDD_FLAG) for RNEA / grad RNEA. Then stage the packed inputs
// on the device once so the compute-only path has resident inputs.
int stage_inputs(KernelCtx &c, int op, const T *q, const T *qd, const T *third, int batch) {
    const int nj = grid::NUM_JOINTS, stride = 3 * nj;
    for (int t = 0; t < batch; ++t) {
        std::memcpy(&c.data->h_q_qd_u[t * stride], &q[t * nj], nj * sizeof(T));
        std::memcpy(&c.data->h_q_qd_u[t * stride + nj], &qd[t * nj], nj * sizeof(T));
        if (op == OP_IDSVA_SO) std::memcpy(&c.data->h_q_qd_u[t * stride + 2 * nj], &third[t * nj], nj * sizeof(T));
        else std::memset(&c.data->h_q_qd_u[t * stride + 2 * nj], 0, nj * sizeof(T));
    }
    if (op != OP_IDSVA_SO) std::memcpy(c.data->h_qdd, third, (size_t)batch * nj * sizeof(T));
    cudaError_t e = cudaMemcpy(c.data->d_q_qd_u, c.data->h_q_qd_u, (size_t)batch * stride * sizeof(T), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return 100 + (int)e;
    if (op != OP_IDSVA_SO) {
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

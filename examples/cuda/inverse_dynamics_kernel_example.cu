// =============================================================================
// GRiD CUDA usage example: writing your OWN kernel against the generated header.
//
// This is the canonical "how do I actually use the codegen output" walkthrough.
// GRiD's whole value is the generated `grid::` CUDA in `grid.cuh`; the high-level
// `grid_rbd` Python wrapper hides it. Here we drive `inverse_dynamics` (the RNEA,
// the simplest dynamics algorithm) the way a controls/MPC author would: from a
// single-block kernel we wrote ourselves, then batched one-block-per-timestep.
//
// -----------------------------------------------------------------------------
// STEP 0. Generate the header for your robot (one line of Python).
// -----------------------------------------------------------------------------
//   from robot_descriptions import iiwa14_description
//   from URDFParser import URDFParser
//   from GRiDCodeGenerator import GRiDCodeGenerator
//   robot = URDFParser().parse(iiwa14_description.URDF_PATH, floating_base=False)
//   GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
//       algorithm_list=["inverse_dynamics"], output_path="grid.cuh")
//
// `algorithm_list` restricts codegen to just the kernels you need (smaller
// header, faster nvcc). The build script for this example does exactly that; see
// examples/cuda/README.md and examples/cuda/gen_iiwa14_header.py.
//
// -----------------------------------------------------------------------------
// STEP 1. Include the generated header. Everything lives in namespace `grid`.
// -----------------------------------------------------------------------------
#include <cstdio>
#include <cmath>
#include <vector>

#include "grid.cuh"   // generated; defines grid::NUM_JOINTS, the kernels, init_*

// gpuErrchk / gpuErrchkKernel are defined inside grid.cuh -- use them after EVERY
// CUDA API call and kernel launch. A silent launch failure (e.g. asking for more
// dynamic shared memory than you registered) otherwise looks like a zeroed
// "answer", which is the single most common GRiD-kernel footgun.

// -----------------------------------------------------------------------------
// Deterministic test inputs (kept identical to examples/cuda/validate.py so the
// numpy oracle and this kernel see the exact same q/qd/qdd).
// -----------------------------------------------------------------------------
template <typename T>
static void make_inputs(std::vector<T> &q, std::vector<T> &qd, std::vector<T> &qdd) {
    const int n = grid::NUM_JOINTS;
    q.resize(n); qd.resize(n); qdd.resize(n);
    for (int i = 0; i < n; ++i) {
        q[i]   = static_cast<T>(0.1 * (i + 1));
        qd[i]  = static_cast<T>(0.01 * (i + 1));
        qdd[i] = static_cast<T>(0.02 * (i + 1));
    }
}

// =============================================================================
// PATH A -- the EASY path: call grid::inverse_dynamics_device<T>(...).
//
// `_device` is the auto-scratch wrapper. It declares the shared-memory arena
// (`extern __shared__`), carves out s_vaf / s_XImats / s_temp / the linalg
// scratch for you, runs load_update_XImats_helpers(), then calls the `_inner`.
// You only hand it inputs + outputs. The launch must reserve exactly
// grid::ID_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>() bytes of dynamic shared memory.
//
// Signature (fixed base, qdd-input variant, from the generated header):
//   inverse_dynamics_device<T>(T *s_c, const T *s_q, const T *s_qd,
//       const T *s_qdd, const robotModel<T> *d_robotModel,
//       T *d_f_ext, const T gravity)
// s_c, s_q, s_qd, s_qdd are SHARED-memory pointers; pass d_f_ext = nullptr for
// no external forces. gravity is signed (-9.81).
// =============================================================================
template <typename T>
__global__ void id_device_kernel(
    T *d_c, const T *d_q, const T *d_qd, const T *d_qdd,
    const grid::robotModel<T> *d_robotModel, const T gravity)
{
    const int n = grid::NUM_JOINTS;
    // Per-block static shared inputs/outputs. (The big algorithm scratch is the
    // *dynamic* arena that _device allocates from the smem we reserve at launch.)
    __shared__ T s_q[grid::NUM_JOINTS];
    __shared__ T s_qd[grid::NUM_JOINTS];
    __shared__ T s_qdd[grid::NUM_JOINTS];
    __shared__ T s_c[grid::NUM_JOINTS];

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        s_q[i] = d_q[i]; s_qd[i] = d_qd[i]; s_qdd[i] = d_qdd[i];
    }
    __syncthreads();

    grid::inverse_dynamics_device<T>(
        s_c, s_q, s_qd, s_qdd, d_robotModel, /*d_f_ext=*/nullptr, gravity);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) d_c[i] = s_c[i];
}

// =============================================================================
// PATH B -- the FULL-CONTROL path: call grid::inverse_dynamics_inner<T>(...).
//
// `_inner` is the fat logic with NO scratch management -- the CALLER places every
// buffer. Use this when you fuse RNEA into a bigger kernel and want to share the
// XImats / linalg scratch with other algorithm calls. You must:
//   (1) lay out the dynamic arena yourself (s_vaf, s_XImats, s_temp, ...),
//   (2) call grid::load_update_XImats_helpers<T>(...) to populate s_XImats,
//   (3) call grid::inverse_dynamics_inner<T>(...).
//
// Buffer sizes for iiwa14 (and the general rule):
//   s_vaf   : 18*NUM_JOINTS  (v/a/f, 6 each per body)
//   s_XImats: 72*NUM_JOINTS  (the 6x6 transform + inertia matrices)
//   s_temp  :  6*NUM_JOINTS  (RNEA helper scratch; see
//                             gen_inverse_dynamics_inner_temp_mem_size)
//   s_topology_helpers: TOPOLOGY_HELPERS_COUNT ints (0 for iiwa14 -> nullptr)
// We size the static shared arrays from these so the example is self-contained.
//
// Signature (fixed base, qdd-input variant, from the generated header):
//   inverse_dynamics_inner<T>(T *s_c, T *s_vaf, const T *s_q, const T *s_qd,
//       const T *s_qdd, T *s_XImats, int *s_topology_helpers, T *s_temp,
//       T *d_f_ext, const T gravity)
// =============================================================================
template <typename T>
__global__ void id_inner_kernel(
    T *d_c, const T *d_q, const T *d_qd, const T *d_qdd,
    const grid::robotModel<T> *d_robotModel, const T gravity)
{
    const int n = grid::NUM_JOINTS;
    __shared__ T s_q[grid::NUM_JOINTS];
    __shared__ T s_qd[grid::NUM_JOINTS];
    __shared__ T s_qdd[grid::NUM_JOINTS];
    __shared__ T s_c[grid::NUM_JOINTS];
    // Caller-owned algorithm scratch.
    __shared__ T s_vaf[18 * grid::NUM_JOINTS];
    __shared__ T s_XImats[72 * grid::NUM_JOINTS];
    __shared__ T s_temp[6 * grid::NUM_JOINTS];

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        s_q[i] = d_q[i]; s_qd[i] = d_qd[i]; s_qdd[i] = d_qdd[i];
    }
    __syncthreads();

    // (2) populate the per-body transform+inertia matrices for this q.
    grid::load_update_XImats_helpers<T>(
        s_XImats, s_q, /*s_topology_helpers=*/nullptr, d_robotModel, s_temp);
    __syncthreads();

    // (3) RNEA. s_topology_helpers is nullptr because TOPOLOGY_HELPERS_COUNT==0
    // for iiwa14; for robots that need it, allocate TOPOLOGY_HELPERS_COUNT ints.
    grid::inverse_dynamics_inner<T>(
        s_c, s_vaf, s_q, s_qd, s_qdd, s_XImats,
        /*s_topology_helpers=*/nullptr, s_temp, /*d_f_ext=*/nullptr, gravity);
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x) d_c[i] = s_c[i];
}

// =============================================================================
// PATH C -- BATCHED: one block per robot/timestep.
//
// The GRiD design is SINGLE-BLOCK per problem (never split one robot across
// blocks). To process a batch of B states, launch B blocks; block k handles
// timestep k. The grid-stride loop makes it robust to B > gridDim.x. Here we
// reuse the easy `_device` wrapper -- the single-block design scales for free.
// Inputs/outputs are [B x NUM_JOINTS], timestep-major.
// =============================================================================
template <typename T>
__global__ void id_batched_kernel(
    T *d_c, const T *d_q, const T *d_qd, const T *d_qdd, int B,
    const grid::robotModel<T> *d_robotModel, const T gravity)
{
    const int n = grid::NUM_JOINTS;
    __shared__ T s_q[grid::NUM_JOINTS];
    __shared__ T s_qd[grid::NUM_JOINTS];
    __shared__ T s_qdd[grid::NUM_JOINTS];
    __shared__ T s_c[grid::NUM_JOINTS];

    for (int k = blockIdx.x; k < B; k += gridDim.x) {
        const T *q_k = &d_q[k * n], *qd_k = &d_qd[k * n], *qdd_k = &d_qdd[k * n];
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            s_q[i] = q_k[i]; s_qd[i] = qd_k[i]; s_qdd[i] = qdd_k[i];
        }
        __syncthreads();
        grid::inverse_dynamics_device<T>(
            s_c, s_q, s_qd, s_qdd, d_robotModel, /*d_f_ext=*/nullptr, gravity);
        __syncthreads();
        for (int i = threadIdx.x; i < n; i += blockDim.x) d_c[k * n + i] = s_c[i];
        __syncthreads();  // reuse smem safely across the grid-stride iterations
    }
}

// -----------------------------------------------------------------------------
// Host driver.
// -----------------------------------------------------------------------------
template <typename T>
static void launch_and_print(const char *label, void (*kernel)(
        T*, const T*, const T*, const T*, const grid::robotModel<T>*, const T),
    const grid::robotModel<T> *d_robotModel, const T *d_q, const T *d_qd,
    const T *d_qdd, T *d_c, int n, int threads, size_t smem, T gravity)
{
    // Opt-in dynamic shared memory MUST be registered before launching, or the
    // launch fails with "invalid argument" (cudaErrorInvalidValue).
    gpuErrchk(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));

    kernel<<<1, threads, smem>>>(d_c, d_q, d_qd, d_qdd, d_robotModel, gravity);
    gpuErrchk(cudaPeekAtLastError());       // catches launch-config failures
    gpuErrchk(cudaDeviceSynchronize());     // catches in-kernel faults

    std::vector<T> h_c(n);
    gpuErrchk(cudaMemcpy(h_c.data(), d_c, n * sizeof(T), cudaMemcpyDeviceToHost));
    printf("BEGIN %s\n", label);
    for (int i = 0; i < n; ++i) printf("%.10g%s", static_cast<double>(h_c[i]),
                                       i + 1 < n ? " " : "\n");
    printf("END %s\n", label);
}

template <typename T>
static void run() {
    const T gravity = static_cast<T>(-9.81);
    const int n = grid::NUM_JOINTS;
    // One warp is plenty for a 7-DoF arm; the kernels carry
    // __launch_bounds__(grid::MAX_PERF_LEVEL_THREADS), so never exceed that.
    int threads = 32;
    if (threads > grid::MAX_PERF_LEVEL_THREADS) threads = grid::MAX_PERF_LEVEL_THREADS;

    // init_robotModel uploads the (compile-time) inertia/topology constants to the
    // GPU; init_grid sets up CUDA streams. We don't use init_gridData here because
    // we manage our own device buffers -- that is the whole point of this example.
    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();

    std::vector<T> h_q, h_qd, h_qdd;
    make_inputs(h_q, h_qd, h_qdd);

    T *d_q, *d_qd, *d_qdd, *d_c;
    gpuErrchk(cudaMalloc(&d_q,   n * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_qd,  n * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_qdd, n * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_c,   n * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_q,   h_q.data(),   n * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_qd,  h_qd.data(),  n * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_qdd, h_qdd.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    // PATH A: _device (auto scratch).
    launch_and_print<T>("inverse_dynamics_device", id_device_kernel<T>,
        d_robotModel, d_q, d_qd, d_qdd, d_c, n, threads,
        grid::ID_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>(), gravity);

    // PATH B: _inner (caller scratch). No dynamic smem registration needed -- all
    // its scratch is the __shared__ arrays declared in the kernel, so launch with
    // smem = 0.
    launch_and_print<T>("inverse_dynamics_inner", id_inner_kernel<T>,
        d_robotModel, d_q, d_qd, d_qdd, d_c, n, threads, /*smem=*/0, gravity);

    // PATH C: batched, one block per timestep.
    const int B = 4;
    std::vector<T> h_qB(B * n), h_qdB(B * n), h_qddB(B * n);
    for (int k = 0; k < B; ++k)
        for (int i = 0; i < n; ++i) {
            // small per-timestep perturbation so each block differs
            h_qB[k * n + i]   = h_q[i]   + static_cast<T>(0.05 * k);
            h_qdB[k * n + i]  = h_qd[i]  + static_cast<T>(0.01 * k);
            h_qddB[k * n + i] = h_qdd[i] + static_cast<T>(0.02 * k);
        }
    T *d_qB, *d_qdB, *d_qddB, *d_cB;
    gpuErrchk(cudaMalloc(&d_qB,   B * n * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_qdB,  B * n * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_qddB, B * n * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_cB,   B * n * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_qB,   h_qB.data(),   B * n * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_qdB,  h_qdB.data(),  B * n * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_qddB, h_qddB.data(), B * n * sizeof(T), cudaMemcpyHostToDevice));

    const size_t smem = grid::ID_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>();
    gpuErrchk(cudaFuncSetAttribute(id_batched_kernel<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    id_batched_kernel<T><<<B, threads, smem>>>(
        d_cB, d_qB, d_qdB, d_qddB, B, d_robotModel, gravity);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    std::vector<T> h_cB(B * n);
    gpuErrchk(cudaMemcpy(h_cB.data(), d_cB, B * n * sizeof(T), cudaMemcpyDeviceToHost));
    for (int k = 0; k < B; ++k) {
        printf("BEGIN inverse_dynamics_batch_%d\n", k);
        for (int i = 0; i < n; ++i)
            printf("%.10g%s", static_cast<double>(h_cB[k * n + i]),
                   i + 1 < n ? " " : "\n");
        printf("END inverse_dynamics_batch_%d\n", k);
    }

    cudaFree(d_q); cudaFree(d_qd); cudaFree(d_qdd); cudaFree(d_c);
    cudaFree(d_qB); cudaFree(d_qdB); cudaFree(d_qddB); cudaFree(d_cB);
    // We own our buffers, so we don't call grid::close_grid (it tears down a
    // gridData we never allocated). Free what init_* gave us directly.
    gpuErrchk(cudaFree(d_robotModel));
    for (int i = 0; i < 3; ++i) gpuErrchk(cudaStreamDestroy(streams[i]));
    free(streams);
}

int main() {
    run<float>();
    return 0;
}

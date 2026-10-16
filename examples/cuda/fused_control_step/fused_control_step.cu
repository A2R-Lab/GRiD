// =============================================================================
// GRiD CUDA example: fuse inverse dynamics into your own control-step kernel.
//
// One computed-torque control step over a batch of B joint states has three
// stages:
//   1. commanded acceleration   qdd_cmd = qdd_des + kp (q_des - q) + kd (qd_des - qd)
//   2. inverse dynamics         tau     = ID(q, qd, qdd_cmd)          (GRiD RNEA)
//   3. one Euler step           q_next  = q + dt qd,  qd_next = qd + dt qdd_cmd
//
// This file computes that step two ways and checks that they agree:
//   fused:   one kernel, one block per state, all three stages in shared memory;
//            the dynamics stage is grid::inverse_dynamics_device<T>, called from
//            inside our kernel. The batch never returns to the host.
//   layered: the generated host function grid::inverse_dynamics<T, ...> for the
//            dynamics stage (it copies the inputs in, launches the generated
//            kernel, copies the torques out), with stages 1 and 3 on the host.
//
// Both paths are timed with cudaEvents; the binary prints the medians and, with
// --json PATH, writes them for the release-measurements page. The timing
// numbers belong to a quiet GPU; the correctness check is valid anywhere.
//
// Header: generated with algorithm_list=["inverse_dynamics"] by
// examples/cuda/gen_iiwa14_header.py (build_and_time.sh runs it). The same
// code works for any fixed-base robot header; a floating base changes the
// input widths (see examples/cuda/README.md, "Adapting to another robot").
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "grid.cuh"

// -----------------------------------------------------------------------------
// Stages 1 and 3 as block-cooperative device helpers. They are the "your own
// math" that fusion keeps next to the dynamics in shared memory.
// -----------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void pd_acceleration(
    T *s_qdd_cmd, const T *s_q, const T *s_qd, const T *s_q_des,
    const T *s_qd_des, const T *s_qdd_des, const T kp, const T kd)
{
    for (int i = threadIdx.x; i < grid::NUM_JOINTS; i += blockDim.x)
        s_qdd_cmd[i] = s_qdd_des[i] + kp * (s_q_des[i] - s_q[i]) + kd * (s_qd_des[i] - s_qd[i]);
}

template <typename T>
__device__ __forceinline__ void euler_step(
    T *s_q_next, T *s_qd_next, const T *s_q, const T *s_qd, const T *s_qdd_cmd, const T dt)
{
    for (int i = threadIdx.x; i < grid::NUM_JOINTS; i += blockDim.x) {
        s_q_next[i]  = s_q[i]  + dt * s_qd[i];
        s_qd_next[i] = s_qd[i] + dt * s_qdd_cmd[i];
    }
}

// -----------------------------------------------------------------------------
// The fused kernel. One block per state (grid-stride over the batch), the three
// stages back to back on shared-memory operands, one global write at the end.
// The launch reserves grid::INVERSE_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>()
// bytes of dynamic shared memory, which is the arena inverse_dynamics_device
// carves its own scratch from; our nine small vectors are static shared memory.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void fused_control_step_kernel(
    T *d_tau, T *d_q_next, T *d_qd_next,
    const T *d_q, const T *d_qd, const T *d_q_des, const T *d_qd_des, const T *d_qdd_des,
    const int B, const grid::robotModel<T> *d_robotModel,
    const T gravity, const T kp, const T kd, const T dt)
{
    constexpr int n = grid::NUM_JOINTS;
    __shared__ T s_q[n], s_qd[n], s_q_des[n], s_qd_des[n], s_qdd_des[n];
    __shared__ T s_qdd_cmd[n], s_tau[n], s_q_next[n], s_qd_next[n];

    for (int k = blockIdx.x; k < B; k += gridDim.x) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            s_q[i] = d_q[k * n + i];           s_qd[i] = d_qd[k * n + i];
            s_q_des[i] = d_q_des[k * n + i];   s_qd_des[i] = d_qd_des[k * n + i];
            s_qdd_des[i] = d_qdd_des[k * n + i];
        }
        __syncthreads();

        pd_acceleration<T>(s_qdd_cmd, s_q, s_qd, s_q_des, s_qd_des, s_qdd_des, kp, kd);
        __syncthreads();

        grid::inverse_dynamics_device<T>(
            s_tau, s_q, s_qd, s_qdd_cmd, d_robotModel, /*d_f_ext=*/nullptr, gravity);
        __syncthreads();

        euler_step<T>(s_q_next, s_qd_next, s_q, s_qd, s_qdd_cmd, dt);
        __syncthreads();

        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            d_tau[k * n + i] = s_tau[i];
            d_q_next[k * n + i] = s_q_next[i];
            d_qd_next[k * n + i] = s_qd_next[i];
        }
        __syncthreads();   // the shared vectors are reused by the next state
    }
}

// -----------------------------------------------------------------------------
// The layered path. Stage 2 is the generated host function; it owns the
// gridData copies and the launch. USE_QDD_FLAG selects the qdd-input RNEA and
// USE_COMPRESSED_MEM the [q | qd] packing of h_q_qd (stride 2*NUM_JOINTS) with
// qdd in its own h_qdd slot (stride NUM_JOINTS).
// -----------------------------------------------------------------------------
template <typename T>
struct Batch {
    int B = 0;
    std::vector<T> q, qd, q_des, qd_des, qdd_des;   // inputs,  [B x n]
    std::vector<T> tau, q_next, qd_next;             // outputs, [B x n]
};

template <typename T>
static void layered_control_step(
    Batch<T> &b, std::vector<T> &qdd_cmd, grid::gridData<T> *hd_data,
    const grid::robotModel<T> *d_robotModel, cudaStream_t *streams,
    const int threads, const T gravity, const T kp, const T kd, const T dt)
{
    constexpr int n = grid::NUM_JOINTS;
    // stage 1 on the host, packed straight into the host function's inputs
    for (int k = 0; k < b.B; ++k)
        for (int i = 0; i < n; ++i) {
            const int e = k * n + i;
            qdd_cmd[e] = b.qdd_des[e] + kp * (b.q_des[e] - b.q[e]) + kd * (b.qd_des[e] - b.qd[e]);
            hd_data->h_q_qd[k * 2 * n + i]     = b.q[e];
            hd_data->h_q_qd[k * 2 * n + n + i] = b.qd[e];
            hd_data->h_qdd[e] = qdd_cmd[e];
        }
    // stage 2: host-to-device copies, the generated kernel, device-to-host copy
    grid::inverse_dynamics<T, /*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/true>(
        hd_data, d_robotModel, gravity, b.B, dim3(b.B), dim3(threads), streams);
    // stage 3 on the host
    for (int e = 0; e < b.B * n; ++e) {
        b.tau[e] = hd_data->h_c[e];
        b.q_next[e] = b.q[e] + dt * b.qd[e];
        b.qd_next[e] = b.qd[e] + dt * qdd_cmd[e];
    }
}

// -----------------------------------------------------------------------------
// Deterministic inputs: B states around a nominal posture, with a small
// tracking error against the desired trajectory so the PD term is non-zero.
// -----------------------------------------------------------------------------
template <typename T>
static Batch<T> make_batch(int B) {
    constexpr int n = grid::NUM_JOINTS;
    Batch<T> b; b.B = B;
    for (auto *v : {&b.q, &b.qd, &b.q_des, &b.qd_des, &b.qdd_des, &b.tau, &b.q_next, &b.qd_next})
        v->assign(static_cast<size_t>(B) * n, T(0));
    for (int k = 0; k < B; ++k)
        for (int i = 0; i < n; ++i) {
            const int e = k * n + i;
            const double phase = 0.7 * k + 0.4 * (i + 1);
            b.q_des[e]   = static_cast<T>(0.8 * std::sin(phase));
            b.qd_des[e]  = static_cast<T>(0.3 * std::cos(phase));
            b.qdd_des[e] = static_cast<T>(0.1 * std::sin(2.0 * phase));
            b.q[e]  = b.q_des[e]  + static_cast<T>(0.05 * std::cos(3.0 * phase));
            b.qd[e] = b.qd_des[e] + static_cast<T>(0.02 * std::sin(5.0 * phase));
        }
    return b;
}

// Largest per-state relative error between two [B x n] results.
template <typename T>
static double worst_rel_err(const std::vector<T> &a, const std::vector<T> &ref, int B) {
    constexpr int n = grid::NUM_JOINTS;
    double worst = 0.0;
    for (int k = 0; k < B; ++k) {
        double num = 0.0, den = 0.0;
        for (int i = 0; i < n; ++i) {
            const double d = double(a[k * n + i]) - double(ref[k * n + i]);
            num += d * d; den += double(ref[k * n + i]) * double(ref[k * n + i]);
        }
        worst = std::max(worst, std::sqrt(num) / std::max(std::sqrt(den), 1e-9));
    }
    return worst;
}

static double median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    const size_t m = v.size() / 2;
    return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

struct Options {
    int batch = 256, repeats = 200, threads = 128, warmup = 20;
    std::string json, arch = "unknown";
};

static Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char *flag) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", flag); exit(2); }
            return argv[++i];
        };
        if (a == "--batch") o.batch = atoi(next("--batch"));
        else if (a == "--repeats") o.repeats = atoi(next("--repeats"));
        else if (a == "--threads") o.threads = atoi(next("--threads"));
        else if (a == "--warmup") o.warmup = atoi(next("--warmup"));
        else if (a == "--json") o.json = next("--json");
        else if (a == "--arch") o.arch = next("--arch");
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); exit(2); }
    }
    if (o.batch < 1 || o.repeats < 1 || o.threads < 1) { fprintf(stderr, "bad option value\n"); exit(2); }
    return o;
}

template <typename T>
static int run(const Options &opt) {
    constexpr int n = grid::NUM_JOINTS;
    const int B = opt.batch;
    const int threads = std::min(opt.threads, grid::MAX_PERF_LEVEL_THREADS);
    const T gravity = T(-9.81), kp = T(100), kd = T(20), dt = T(0.001);
    const double tol = 1e-4;   // float32: both paths run the same RNEA code

    // Initialise the header: streams, then the dynamic shared-memory attribute
    // of the ONE algorithm we launch through the host path. init_grid<T>() would
    // register, and so compile, every kernel in the header.
    cudaStream_t *streams = grid::init_grid_streams<T>();
    gpuErrchk(grid::init_grid_kernel_attr_inverse_dynamics_checked<T>());
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T>(B);   // the layered path's buffers

    // Our own kernel is not in the header, so we register its arena ourselves.
    const size_t smem = grid::INVERSE_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>();
    gpuErrchk(cudaFuncSetAttribute(fused_control_step_kernel<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));

    Batch<T> b = make_batch<T>(B);
    std::vector<T> qdd_cmd(static_cast<size_t>(B) * n);

    // Device-resident inputs and outputs for the fused path.
    const size_t bytes = static_cast<size_t>(B) * n * sizeof(T);
    T *d_q, *d_qd, *d_q_des, *d_qd_des, *d_qdd_des, *d_tau, *d_q_next, *d_qd_next;
    for (T **p : {&d_q, &d_qd, &d_q_des, &d_qd_des, &d_qdd_des, &d_tau, &d_q_next, &d_qd_next})
        gpuErrchk(cudaMalloc(p, bytes));
    gpuErrchk(cudaMemcpy(d_q, b.q.data(), bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_qd, b.qd.data(), bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_q_des, b.q_des.data(), bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_qd_des, b.qd_des.data(), bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_qdd_des, b.qdd_des.data(), bytes, cudaMemcpyHostToDevice));

    auto launch_fused = [&]() {
        fused_control_step_kernel<T><<<B, threads, smem>>>(
            d_tau, d_q_next, d_qd_next, d_q, d_qd, d_q_des, d_qd_des, d_qdd_des,
            B, d_robotModel, gravity, kp, kd, dt);
        gpuErrchk(cudaPeekAtLastError());
    };
    auto run_layered = [&]() {
        layered_control_step<T>(b, qdd_cmd, hd_data, d_robotModel, streams, threads, gravity, kp, kd, dt);
    };

    // --- correctness: the two paths must agree ------------------------------
    launch_fused();
    gpuErrchk(cudaDeviceSynchronize());
    std::vector<T> f_tau(b.tau.size()), f_q_next(b.tau.size()), f_qd_next(b.tau.size());
    gpuErrchk(cudaMemcpy(f_tau.data(), d_tau, bytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(f_q_next.data(), d_q_next, bytes, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(f_qd_next.data(), d_qd_next, bytes, cudaMemcpyDeviceToHost));
    run_layered();
    const double e_tau = worst_rel_err(f_tau, b.tau, B);
    const double e_q = worst_rel_err(f_q_next, b.q_next, B);
    const double e_qd = worst_rel_err(f_qd_next, b.qd_next, B);
    const double worst = std::max({e_tau, e_q, e_qd});
    const bool ok = worst < tol;
    printf("correctness: fused vs layered over %d states, worst rel_err tau=%.3e q_next=%.3e qd_next=%.3e (tol %.0e) %s\n",
           B, e_tau, e_q, e_qd, tol, ok ? "OK" : "FAIL");
    if (!ok) return 1;

    // --- timing: medians of cudaEvent-timed repeats after warm-up -----------
    cudaEvent_t e0, e1;
    gpuErrchk(cudaEventCreate(&e0));
    gpuErrchk(cudaEventCreate(&e1));
    std::vector<float> t_fused, t_layered;
    t_fused.reserve(opt.repeats); t_layered.reserve(opt.repeats);

    for (int r = 0; r < opt.warmup; ++r) launch_fused();
    gpuErrchk(cudaDeviceSynchronize());
    for (int r = 0; r < opt.repeats; ++r) {
        gpuErrchk(cudaEventRecord(e0));
        launch_fused();
        gpuErrchk(cudaEventRecord(e1));
        gpuErrchk(cudaEventSynchronize(e1));
        float ms = 0; gpuErrchk(cudaEventElapsedTime(&ms, e0, e1));
        t_fused.push_back(ms * 1000.0f);
    }

    for (int r = 0; r < opt.warmup; ++r) run_layered();
    gpuErrchk(cudaDeviceSynchronize());
    for (int r = 0; r < opt.repeats; ++r) {
        // The streams are blocking, so events on the legacy default stream
        // order against the host function's copies and launch; the host
        // stages run between the two records and are inside the interval.
        gpuErrchk(cudaEventRecord(e0));
        run_layered();
        gpuErrchk(cudaEventRecord(e1));
        gpuErrchk(cudaEventSynchronize(e1));
        float ms = 0; gpuErrchk(cudaEventElapsedTime(&ms, e0, e1));
        t_layered.push_back(ms * 1000.0f);
    }

    const double fused_us = median(t_fused), layered_us = median(t_layered);
    printf("timing: batch=%d threads=%d repeats=%d  fused=%.2f us  layered=%.2f us  (medians; %s)\n",
           B, threads, opt.repeats, fused_us, layered_us, opt.arch.c_str());

    if (!opt.json.empty()) {
        FILE *f = fopen(opt.json.c_str(), "w");
        if (!f) { fprintf(stderr, "cannot write %s\n", opt.json.c_str()); return 1; }
        fprintf(f, "{\"example\": \"fused_control_step\", \"batch\": %d, \"fused_us\": %.3f, "
                   "\"layered_us\": %.3f, \"repeats\": %d, \"arch\": \"%s\"}\n",
                B, fused_us, layered_us, opt.repeats, opt.arch.c_str());
        fclose(f);
        printf("wrote %s\n", opt.json.c_str());
    }

    gpuErrchk(cudaEventDestroy(e0));
    gpuErrchk(cudaEventDestroy(e1));
    for (T *p : {d_q, d_qd, d_q_des, d_qd_des, d_qdd_des, d_tau, d_q_next, d_qd_next}) gpuErrchk(cudaFree(p));
    grid::close_grid<T>(streams, d_robotModel, hd_data);
    return 0;
}

int main(int argc, char **argv) {
    return run<float>(parse(argc, argv));
}

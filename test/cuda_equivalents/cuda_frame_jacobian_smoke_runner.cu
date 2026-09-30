// CUDA smoke runner for the generated general-frame geometric Jacobian family
// (E2): J (grid::frame_jacobian_device), Jdot (grid::frame_jacobian_dot_device),
// and the operational-space inertia Lambda (grid::osc_inertia_device, which is
// SELF-CONTAINED — it composes Minv on device from q alone, no external feed).
// Drives each from a single-block kernel for the three pinocchio reference frames and
// prints the results in BEGIN/END framed blocks (column-major), to be
// cross-checked against the RBDReference numpy oracle (which matches pinocchio's
// getFrameJacobian / getJointJacobian / computeJointJacobiansTimeVariation and
// inv(J Minv J^T) to ~1e-14).
//
// Input on stdin (whitespace-separated):
//   target_jid (int)        joint id of the frame
//   q  (NUM_POS floats)
//   qd (NUM_VEL floats)     generalized velocity v (Pinocchio order)
//
// Emitted blocks (J/Jdot: 6 x NUM_VEL; Lambda: 6 x 6; all column-major):
//   J_local / Jd_local / L_local     reference_frame = 0  (LOCAL)
//   J_world / Jd_world / L_world      reference_frame = 1  (WORLD)
//   J_lwa   / Jd_lwa   / L_lwa        reference_frame = 2  (LOCAL_WORLD_ALIGNED)
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "grid.cuh"

template <typename T>
void read_vector(T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        double value;
        if (!(std::cin >> value)) { std::cerr << "read fail " << i << "\n"; std::exit(2); }
        dst[i] = static_cast<T>(value);
    }
}

template <typename T>
void print_matrix_col_major(const std::string &name, const T *data, int rows, int cols) {
    std::cout << "BEGIN " << name << " " << rows << " " << cols << "\n";
    std::cout << std::setprecision(10);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (col) std::cout << " ";
            std::cout << static_cast<double>(data[row + rows * col]);
        }
        std::cout << "\n";
    }
    std::cout << "END " << name << "\n";
}

constexpr int NQ = grid::NUM_POS;
constexpr int NV = grid::NUM_VEL;

// Optional block thread count, set from argv[1] in main(). 0 => use the robot's
// MAX_PERF_LEVEL_THREADS (the default the existing frame_jacobian test relies on
// when it passes no thread-count argument). The V6 invariance test passes an
// explicit low/power-of-two count here.
int g_num_threads = 0;

// J kernel: emit J for the three reference frames.
template <typename T>
__global__ void frame_jac_kernel(const T *g_q, const int target_jid,
                                 const grid::robotModel<T> *d_robotModel,
                                 T *o_local, T *o_world, T *o_lwa) {
    __shared__ T s_q[NQ];
    __shared__ T s_J[6 * NV];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    __syncthreads();

    grid::frame_jacobian_device<T>(s_J, target_jid, 0, s_q, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 6 * NV; i += nth) o_local[i] = s_J[i];
    __syncthreads();

    grid::frame_jacobian_device<T>(s_J, target_jid, 1, s_q, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 6 * NV; i += nth) o_world[i] = s_J[i];
    __syncthreads();

    grid::frame_jacobian_device<T>(s_J, target_jid, 2, s_q, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 6 * NV; i += nth) o_lwa[i] = s_J[i];
    __syncthreads();
}

// Jdot kernel: emit Jdot for the three reference frames.
template <typename T>
__global__ void frame_jac_dot_kernel(const T *g_q, const T *g_qd, const int target_jid,
                                      const grid::robotModel<T> *d_robotModel,
                                      T *o_local, T *o_world, T *o_lwa) {
    __shared__ T s_q[NQ];
    __shared__ T s_qd[NV];
    __shared__ T s_Jd[6 * NV];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    for (int i = tid; i < NV; i += nth) s_qd[i] = g_qd[i];
    __syncthreads();

    grid::frame_jacobian_dot_device<T>(s_Jd, target_jid, 0, s_q, s_qd, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 6 * NV; i += nth) o_local[i] = s_Jd[i];
    __syncthreads();

    grid::frame_jacobian_dot_device<T>(s_Jd, target_jid, 1, s_q, s_qd, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 6 * NV; i += nth) o_world[i] = s_Jd[i];
    __syncthreads();

    grid::frame_jacobian_dot_device<T>(s_Jd, target_jid, 2, s_q, s_qd, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 6 * NV; i += nth) o_lwa[i] = s_Jd[i];
    __syncthreads();
}

// Lambda kernel: SELF-CONTAINED. grid::osc_inertia_device composes Minv on
// device (via minv_inner) from q alone — no external Minv feed — then
// emits Lambda for the three reference frames.
//
// osc_inertia is currently ¬mimic in codegen (its on-device mimic-Minv route is
// deferred), so mimic headers do NOT emit it and instead define GRID_FRAME_JAC_MIMIC.
// Guard the whole Lambda path so the runner still builds for mimic robots (e.g.
// fr3); the test detects the missing L_* blocks and skips the Lambda check.
#ifndef GRID_FRAME_JAC_MIMIC
template <typename T>
__global__ void osc_kernel(const T *g_q, const int target_jid,
                           const grid::robotModel<T> *d_robotModel,
                           T *o_local, T *o_world, T *o_lwa) {
    __shared__ T s_q[NQ];
    __shared__ T s_L[36];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    __syncthreads();

    grid::osc_inertia_device<T>(s_L, target_jid, 0, s_q, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 36; i += nth) o_local[i] = s_L[i];
    __syncthreads();

    grid::osc_inertia_device<T>(s_L, target_jid, 1, s_q, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 36; i += nth) o_world[i] = s_L[i];
    __syncthreads();

    grid::osc_inertia_device<T>(s_L, target_jid, 2, s_q, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 36; i += nth) o_lwa[i] = s_L[i];
    __syncthreads();
}
#endif  // GRID_FRAME_JAC_MIMIC

template <typename T>
T *dmalloc(int count) { T *p; cudaMalloc(&p, count * sizeof(T)); return p; }
template <typename T>
void dcopy_out(const std::string &name, T *dptr, int rows, int cols) {
    std::vector<T> h(rows * cols);
    cudaMemcpy(h.data(), dptr, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);
    print_matrix_col_major(name, h.data(), rows, cols);
}

template <typename T>
void run() {
    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    int target_jid;
    if (!(std::cin >> target_jid)) { std::cerr << "read fail target_jid\n"; std::exit(2); }
    std::vector<T> h_q(NQ), h_qd(NV);
    read_vector(h_q.data(), NQ);
    read_vector(h_qd.data(), NV);

    print_matrix_col_major("input_q", h_q.data(), 1, NQ);

    T *g_q = dmalloc<T>(NQ);
    cudaMemcpy(g_q, h_q.data(), NQ * sizeof(T), cudaMemcpyHostToDevice);
    T *g_qd = dmalloc<T>(NV);
    cudaMemcpy(g_qd, h_qd.data(), NV * sizeof(T), cudaMemcpyHostToDevice);

    T *o_jl = dmalloc<T>(6 * NV), *o_jw = dmalloc<T>(6 * NV), *o_jx = dmalloc<T>(6 * NV);
    T *o_dl = dmalloc<T>(6 * NV), *o_dw = dmalloc<T>(6 * NV), *o_dx = dmalloc<T>(6 * NV);
#ifndef GRID_FRAME_JAC_MIMIC
    T *o_ll = dmalloc<T>(36), *o_lw = dmalloc<T>(36), *o_lx = dmalloc<T>(36);
#endif

    // Block thread count for the J / Jdot kernels. Defaults to the robot's
    // MAX_PERF_LEVEL_THREADS but is overridable via g_num_threads (argv[1]) so
    // the thread-count-invariance test (V6) can sweep {1,2,16,32,64,128,256}.
    // Single-block kernels are block-stride loops, so ANY count that fits must
    // produce identical results; a low-count divergence is a missing
    // __syncthreads, not a test artifact.
    int nthreads = g_num_threads;
    if (nthreads <= 0 || nthreads > grid::MAX_PERF_LEVEL_THREADS) {
        nthreads = grid::MAX_PERF_LEVEL_THREADS;
    }

    size_t dyn_j = grid::FRAME_JACOBIAN_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(frame_jac_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_j);
    frame_jac_kernel<T><<<1, nthreads, dyn_j>>>(g_q, target_jid, d_robotModel, o_jl, o_jw, o_jx);
    // Fail loudly on a bad launch (an unchecked failure leaves zeroed outputs that
    // masquerade as a real result). gpuErrchkKernel() (grid.cuh) peeks + syncs + aborts.
    gpuErrchkKernel();

    size_t dyn_d = grid::FRAME_JACOBIAN_DOT_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(frame_jac_dot_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_d);
    frame_jac_dot_kernel<T><<<1, nthreads, dyn_d>>>(g_q, g_qd, target_jid, d_robotModel, o_dl, o_dw, o_dx);
    gpuErrchkKernel();

    // Self-contained Lambda: osc_inertia_device composes Minv on device, so the
    // runner no longer pre-computes/densifies a Minv to feed in. This is emitted
    // for BOTH non-mimic and mimic robots (the mimic path composes Minv via
    // minv_inner -> crba_inner -> invert, which the fr3-fixed CUDA crba/minv
    // tests already prove correct); GRID_FRAME_JAC_MIMIC is only defined when
    // osc_inertia was NOT selected at all.
#ifndef GRID_FRAME_JAC_MIMIC
    size_t dyn_o = grid::OSC_INERTIA_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(osc_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_o);
    // osc_kernel is NOT __launch_bounds__-annotated and inlines the heavy
    // minv_inner / crba_inner / invert_matrix routines (~100+ regs). At
    // MAX_PERF_LEVEL_THREADS (512) the launch overflows the per-block register
    // budget -> "too many resources requested for launch", which is silent unless
    // checked and leaves the (zeroed) outputs untouched. Clamp to the kernel's
    // attribute-reported maxThreadsPerBlock (the register-limited cap) and assert
    // the launch succeeds so a real failure never masquerades as an all-zero Lambda.
    cudaFuncAttributes osc_attr;
    cudaFuncGetAttributes(&osc_attr, osc_kernel<T>);
    int osc_threads = nthreads;
    if (osc_attr.maxThreadsPerBlock > 0 && osc_threads > osc_attr.maxThreadsPerBlock)
        osc_threads = osc_attr.maxThreadsPerBlock;
    osc_kernel<T><<<1, osc_threads, dyn_o>>>(g_q, target_jid, d_robotModel, o_ll, o_lw, o_lx);
    cudaError_t osc_launch_err = cudaGetLastError();
    if (osc_launch_err != cudaSuccess) {
        std::cerr << "osc_kernel launch failed: "
                  << cudaGetErrorString(osc_launch_err)
                  << " (threads=" << osc_threads << ")\n";
        std::exit(3);
    }
#endif
    cudaDeviceSynchronize();

    dcopy_out("J_local", o_jl, 6, NV);
    dcopy_out("J_world", o_jw, 6, NV);
    dcopy_out("J_lwa", o_jx, 6, NV);
    dcopy_out("Jd_local", o_dl, 6, NV);
    dcopy_out("Jd_world", o_dw, 6, NV);
    dcopy_out("Jd_lwa", o_dx, 6, NV);
#ifndef GRID_FRAME_JAC_MIMIC
    dcopy_out("L_local", o_ll, 6, 6);
    dcopy_out("L_world", o_lw, 6, 6);
    dcopy_out("L_lwa", o_lx, 6, 6);
#endif

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main(int argc, char **argv) {
    // Optional argv[1] = block thread count (V6 thread-count-invariance sweep).
    // Omitting it (the existing frame_jacobian test) leaves g_num_threads=0 ->
    // run() falls back to MAX_PERF_LEVEL_THREADS, byte-identical to before.
    if (argc > 1) {
        int requested = std::atoi(argv[1]);
        g_num_threads = requested > 0 ? requested : 0;
    }
    run<float>();
    return 0;
}

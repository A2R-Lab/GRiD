// CUDA smoke runner for the runtime-target end-effector pose + pose-gradient
// family (additive): end_effector_pose_runtime (6-vector [xyz; rpy]) and
// end_effector_pose_gradient_runtime (6 x NUM_VEL = d[xyz; rpy]/dv), both at a
// RUNTIME target_jid and a RUNTIME offset point in the target frame.
//
// Drives each device fn from a single-block kernel for BOTH offset=0 and the
// supplied offset, printing the results in BEGIN/END framed blocks (column-major)
// to be cross-checked against the RBDReference numpy oracle
// (RBDReference.end_effector_pose / .end_effector_pose_gradient with the
// runtime ee_joint_names / ee_offsets list API), which matches pinocchio /
// the analytic geometric Jacobian to ~1e-14.
//
// Input on stdin (whitespace-separated):
//   target_jid (int)        joint id of the EE frame
//   X_tool (16 floats)      4x4 col-major SE(3) tool/tip transform in the target frame
//   q  (NUM_POS floats)
//
// Emitted blocks:
//   pose0 / grad0           X_tool = identity   (frame origin)
//   poseN / gradN           X_tool = supplied
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

int g_num_threads = 0;

// pose kernel: emit the 6-vector pose at the supplied 4x4 col-major tool transform.
template <typename T>
__global__ void pose_kernel(const T *g_q, const int target_jid, const T *g_Xtool,
                            const grid::robotModel<T> *d_robotModel, T *o_pose) {
    __shared__ T s_q[NQ];
    __shared__ T s_Xtool[16];
    __shared__ T s_pose[6];
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    for (int i = tid; i < 16; i += nth) s_Xtool[i] = g_Xtool[i];
    __syncthreads();
    grid::end_effector_pose_runtime_device<T>(s_pose, target_jid, s_Xtool, s_q, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 6; i += nth) o_pose[i] = s_pose[i];
    __syncthreads();
}

// gradient kernel: emit the 6 x NV pose gradient at the supplied 4x4 col-major tool transform.
template <typename T>
__global__ void grad_kernel(const T *g_q, const int target_jid, const T *g_Xtool,
                            const grid::robotModel<T> *d_robotModel, T *o_grad) {
    __shared__ T s_q[NQ];
    __shared__ T s_Xtool[16];
    __shared__ T s_grad[6 * NV];
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    for (int i = tid; i < 16; i += nth) s_Xtool[i] = g_Xtool[i];
    __syncthreads();
    grid::end_effector_pose_gradient_runtime_device<T>(s_grad, target_jid, s_Xtool, s_q, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 6 * NV; i += nth) o_grad[i] = s_grad[i];
    __syncthreads();
}

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
    // 4x4 col-major SE(3) tool transform (16 floats), then q.
    std::vector<T> h_Xtool(16), h_q(NQ);
    read_vector(h_Xtool.data(), 16);
    read_vector(h_q.data(), NQ);
    print_matrix_col_major("input_q", h_q.data(), 1, NQ);

    T *g_q = dmalloc<T>(NQ);
    cudaMemcpy(g_q, h_q.data(), NQ * sizeof(T), cudaMemcpyHostToDevice);
    T h_ident[16] = {static_cast<T>(1),0,0,0, 0,static_cast<T>(1),0,0,
                     0,0,static_cast<T>(1),0, 0,0,0,static_cast<T>(1)};
    T *g_off0 = dmalloc<T>(16); cudaMemcpy(g_off0, h_ident, 16 * sizeof(T), cudaMemcpyHostToDevice);
    T *g_offN = dmalloc<T>(16); cudaMemcpy(g_offN, h_Xtool.data(), 16 * sizeof(T), cudaMemcpyHostToDevice);

    T *o_p0 = dmalloc<T>(6), *o_pN = dmalloc<T>(6);
    T *o_g0 = dmalloc<T>(6 * NV), *o_gN = dmalloc<T>(6 * NV);

    int nthreads = g_num_threads;
    if (nthreads <= 0 || nthreads > grid::MAX_PERF_LEVEL_THREADS) {
        nthreads = grid::MAX_PERF_LEVEL_THREADS;
    }

    size_t dyn_p = grid::END_EFFECTOR_POSE_RUNTIME_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(pose_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_p);
    pose_kernel<T><<<1, nthreads, dyn_p>>>(g_q, target_jid, g_off0, d_robotModel, o_p0);
    gpuErrchkKernel();
    pose_kernel<T><<<1, nthreads, dyn_p>>>(g_q, target_jid, g_offN, d_robotModel, o_pN);
    gpuErrchkKernel();

    size_t dyn_g = grid::END_EFFECTOR_POSE_GRADIENT_RUNTIME_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(grad_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn_g);
    grad_kernel<T><<<1, nthreads, dyn_g>>>(g_q, target_jid, g_off0, d_robotModel, o_g0);
    gpuErrchkKernel();
    grad_kernel<T><<<1, nthreads, dyn_g>>>(g_q, target_jid, g_offN, d_robotModel, o_gN);
    gpuErrchkKernel();

    cudaDeviceSynchronize();

    dcopy_out("pose0", o_p0, 6, 1);
    dcopy_out("poseN", o_pN, 6, 1);
    dcopy_out("grad0", o_g0, 6, NV);
    dcopy_out("gradN", o_gN, 6, NV);

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main(int argc, char **argv) {
    if (argc > 1) {
        int requested = std::atoi(argv[1]);
        g_num_threads = requested > 0 ? requested : 0;
    }
    run<float>();
    return 0;
}

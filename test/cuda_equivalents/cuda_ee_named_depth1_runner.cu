// CUDA smoke runner for a NAMED fixed-target end-effector pose whose chain
// reaches the root EARLY (depth << n_bfs_levels), e.g. go2's imu_joint (depth 1
// of 4). Regression gate for BUG7 (GATO 2026-08-11): the ping-pong chain walk in
// end_effector_pose_inner left an early-rooted chain's live transform stranded
// in the other buffer half, so the extractor read a stale leaf copy and the pose
// came out BASE-RELATIVE (J == 0 downstream: the whole EE tracking cost was a
// no-op on go2).
//
// The header must be generated with fixed_target_name=<target>; the runner uses
// the target-agnostic grid::end_effector_pose_target_device forwarder.
//
// Input on stdin (whitespace-separated): q (NUM_POS floats)
// Output: BEGIN/END framed "pose" block (1 x 6, [xyz; rpy]).
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "grid.cuh"

#define NQ grid::NUM_POS

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
            std::cout << static_cast<double>(data[col * rows + row]);
        }
        std::cout << "\n";
    }
    std::cout << "END " << name << "\n";
}

template <typename T>
__global__ void pose_kernel(const T *g_q, const grid::robotModel<T> *d_robotModel, T *o_pose) {
    __shared__ T s_q[NQ];
    __shared__ T s_pose[6];
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    __syncthreads();
    grid::end_effector_pose_target_device<T>(s_pose, s_q, d_robotModel);
    __syncthreads();
    for (int i = tid; i < 6; i += nth) o_pose[i] = s_pose[i];
    __syncthreads();
}

template <typename T>
void run() {
    grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();

    std::vector<T> h_q(NQ);
    read_vector(h_q.data(), NQ);

    T *g_q; cudaMalloc(&g_q, NQ * sizeof(T));
    cudaMemcpy(g_q, h_q.data(), NQ * sizeof(T), cudaMemcpyHostToDevice);
    T *o_pose; cudaMalloc(&o_pose, 6 * sizeof(T));

    size_t dyn = grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(pose_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dyn);
    pose_kernel<T><<<1, 128, dyn>>>(g_q, d_robotModel, o_pose);
    gpuErrchkKernel();

    std::vector<T> h_pose(6);
    cudaMemcpy(h_pose.data(), o_pose, 6 * sizeof(T), cudaMemcpyDeviceToHost);
    print_matrix_col_major("pose", h_pose.data(), 1, 6);
}

int main() {
    run<float>();
    return 0;
}

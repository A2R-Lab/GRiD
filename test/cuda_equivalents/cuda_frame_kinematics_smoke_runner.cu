// CUDA smoke runner for the generated frame velocity / acceleration device
// functions (item C of 0.8, 2026-11-03). Mirrors cuda_frame_jacobian_smoke_runner:
// stdin = target_jid, q (NUM_POS), qd (NUM_VEL), qdd (NUM_VEL); stdout = the
// six-entry twist and acceleration of the frame in LOCAL, WORLD and
// LOCAL_WORLD_ALIGNED (blocks V_local, V_world, V_lwa, A_local, A_world, A_lwa)
// and their derivatives GV_<frame> = [dv/dq | dv/dqd] (6 x 2 NV) and
// GA_<frame> = [da/dq | da/dqd | da/dqdd] (6 x 3 NV).
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

template <typename T>
__global__ void frame_kin_kernel(const T *g_q, const T *g_qd, const T *g_qdd, const int target_jid,
                                 const grid::robotModel<T> *d_robotModel,
                                 T *o_v, T *o_a, T *o_gv, T *o_ga) {
    __shared__ T s_q[NQ];
    __shared__ T s_qd[NV];
    __shared__ T s_qdd[NV];
    __shared__ T s_out[6];
    __shared__ T s_grad[18 * NV];
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    for (int i = tid; i < NV; i += nth) { s_qd[i] = g_qd[i]; s_qdd[i] = g_qdd[i]; }
    __syncthreads();
    for (int ref = 0; ref < 3; ++ref) {
        grid::frame_velocity_device<T>(s_out, target_jid, ref, s_q, s_qd, d_robotModel);
        __syncthreads();
        for (int i = tid; i < 6; i += nth) o_v[6 * ref + i] = s_out[i];
        __syncthreads();
        grid::frame_acceleration_device<T>(s_out, target_jid, ref, s_q, s_qd, s_qdd, d_robotModel);
        __syncthreads();
        for (int i = tid; i < 6; i += nth) o_a[6 * ref + i] = s_out[i];
        __syncthreads();
        grid::frame_velocity_gradient_device<T>(s_grad, target_jid, ref, s_q, s_qd, d_robotModel);
        __syncthreads();
        for (int i = tid; i < 12 * NV; i += nth) o_gv[12 * NV * ref + i] = s_grad[i];
        __syncthreads();
        grid::frame_acceleration_gradient_device<T>(s_grad, target_jid, ref, s_q, s_qd, s_qdd, d_robotModel);
        __syncthreads();
        for (int i = tid; i < 18 * NV; i += nth) o_ga[18 * NV * ref + i] = s_grad[i];
        __syncthreads();
    }
}

template <typename T>
T *dmalloc(int count) { T *p; cudaMalloc(&p, count * sizeof(T)); return p; }

template <typename T>
void run(int num_threads) {
    cudaStream_t *streams = grid::init_grid_streams<T>();
    (void)streams;
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    int target_jid;
    if (!(std::cin >> target_jid)) { std::cerr << "read fail target_jid\n"; std::exit(2); }
    std::vector<T> h_q(NQ), h_qd(NV), h_qdd(NV);
    read_vector(h_q.data(), NQ);
    read_vector(h_qd.data(), NV);
    read_vector(h_qdd.data(), NV);
    T *g_q = dmalloc<T>(NQ); cudaMemcpy(g_q, h_q.data(), NQ * sizeof(T), cudaMemcpyHostToDevice);
    T *g_qd = dmalloc<T>(NV); cudaMemcpy(g_qd, h_qd.data(), NV * sizeof(T), cudaMemcpyHostToDevice);
    T *g_qdd = dmalloc<T>(NV); cudaMemcpy(g_qdd, h_qdd.data(), NV * sizeof(T), cudaMemcpyHostToDevice);
    T *o_v = dmalloc<T>(18), *o_a = dmalloc<T>(18);
    T *o_gv = dmalloc<T>(36 * NV), *o_ga = dmalloc<T>(54 * NV);
    // the device wrappers own the dynamic arena: reserve the largest
    size_t smem = grid::FRAME_ACCELERATION_DYNAMIC_SHARED_MEM_BYTES<T>();
    if (grid::FRAME_ACCELERATION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() > smem) smem = grid::FRAME_ACCELERATION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    if (grid::FRAME_VELOCITY_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>() > smem) smem = grid::FRAME_VELOCITY_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(frame_kin_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    frame_kin_kernel<T><<<1, num_threads, smem>>>(g_q, g_qd, g_qdd, target_jid, d_robotModel, o_v, o_a, o_gv, o_ga);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n"; std::exit(3); }
    std::vector<T> h_v(18), h_a(18), h_gv(36 * NV), h_ga(54 * NV);
    cudaMemcpy(h_v.data(), o_v, 18 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a.data(), o_a, 18 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gv.data(), o_gv, 36 * NV * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ga.data(), o_ga, 54 * NV * sizeof(T), cudaMemcpyDeviceToHost);
    const char *names[3] = {"local", "world", "lwa"};
    for (int ref = 0; ref < 3; ++ref) {
        print_matrix_col_major(std::string("V_") + names[ref], &h_v[6 * ref], 6, 1);
        print_matrix_col_major(std::string("A_") + names[ref], &h_a[6 * ref], 6, 1);
        print_matrix_col_major(std::string("GV_") + names[ref], &h_gv[12 * NV * ref], 6, 2 * NV);
        print_matrix_col_major(std::string("GA_") + names[ref], &h_ga[18 * NV * ref], 6, 3 * NV);
    }
}

int main(int argc, char **argv) {
    int num_threads = 128;
    if (argc > 1) num_threads = std::atoi(argv[1]);
    run<float>(num_threads);
    return 0;
}

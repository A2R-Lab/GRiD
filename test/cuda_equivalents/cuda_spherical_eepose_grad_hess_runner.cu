// CUDA equivalence runner for SPHERICAL (ball) joint end_effector_pose_gradient
// + end_effector_pose_hessian (Tier-C kinematic derivatives).
//
// Convention (matches RBDReference + pinocchio): outputs are TANGENT-space
// derivatives d/dv, so a spherical joint contributes THREE columns (its
// get_joint_index_v slots; body-frame omega ordering, local/right tangent
// q ⊗ exp(½·omega)) — NOT four quaternion-component columns. Gradient is
// 6 x NUM_VEL per ee, column-major (index = row + 6*vi); hessian is
// (6, nv, nv) C-order per ee (index = c*nv*nv + vi*nv + vj), rpy rows through
// E(rpy)^-1 exactly like the fixed/floating robots.
//
// Surfaces exercised (mirrors cuda_spherical_kinematics_runner.cu):
//   (1a) end_effector_pose_gradient_device (explicit nq-wide s_q buffer)
//   (1b) end_effector_pose_hessian_device (also co-computes the gradient)
//   (2)  the HOST batch wrappers end_effector_pose_gradient<T> /
//        end_effector_pose_hessian<T> over B identical timesteps (nq-stride path)
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "grid.cuh"

int g_num_threads = 32;

template <typename T>
void read_vector(T *dst, int count) {
    for (int i = 0; i < count; ++i) {
        double value;
        if (!(std::cin >> value)) {
            std::cerr << "Failed to read input value " << i << std::endl;
            std::exit(2);
        }
        dst[i] = static_cast<T>(value);
    }
}

template <typename T>
void print_vector(const std::string &name, const T *data, int count) {
    std::cout << "BEGIN " << name << " 1 " << count << "\n";
    std::cout << std::setprecision(10);
    for (int i = 0; i < count; ++i) {
        if (i) std::cout << " ";
        std::cout << static_cast<double>(data[i]);
    }
    std::cout << "\nEND " << name << "\n";
}

// (1a) Device-function runner for end_effector_pose_gradient. s_q is NQ-wide;
// the 6*NUM_VEL*NUM_EES output is a caller param (static shared memory here,
// NOT the device function's dynamic arena).
template <typename T>
__global__ void spherical_deepose_device_runner(
    T *d_grad, const T *d_q, const grid::robotModel<T> *d_robot_model
) {
    __shared__ T s_grad[6 * grid::NUM_VEL * grid::NUM_EES];
    __shared__ T s_q[grid::NUM_JOINTS];      // NUM_JOINTS == nq for this codegen
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_JOINTS; ind += blockDim.x * blockDim.y) {
        s_q[ind] = d_q[ind];
    }
    __syncthreads();
    grid::end_effector_pose_gradient_device<T>(s_grad, s_q, d_robot_model);
    __syncthreads();
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < 6 * grid::NUM_VEL * grid::NUM_EES; ind += blockDim.x * blockDim.y) {
        d_grad[ind] = s_grad[ind];
    }
}

// (1b) Device-function runner for end_effector_pose_hessian (TIER_SHARED: the
// nv^2 output stays in shared memory; d_workspace unused -> nullptr).
template <typename T>
__global__ void spherical_d2eepose_device_runner(
    T *d_hess, T *d_grad, const T *d_q, const grid::robotModel<T> *d_robot_model
) {
    __shared__ T s_hess[6 * grid::NUM_VEL * grid::NUM_VEL * grid::NUM_EES];
    __shared__ T s_grad[6 * grid::NUM_VEL * grid::NUM_EES];
    __shared__ T s_q[grid::NUM_JOINTS];
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_JOINTS; ind += blockDim.x * blockDim.y) {
        s_q[ind] = d_q[ind];
    }
    __syncthreads();
    grid::end_effector_pose_hessian_device<T>(s_hess, s_grad, s_q, d_robot_model);
    __syncthreads();
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < 6 * grid::NUM_VEL * grid::NUM_VEL * grid::NUM_EES;
         ind += blockDim.x * blockDim.y) {
        d_hess[ind] = s_hess[ind];
    }
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < 6 * grid::NUM_VEL * grid::NUM_EES; ind += blockDim.x * blockDim.y) {
        d_grad[ind] = s_grad[ind];
    }
}

template <typename T>
void run() {
    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();

    const int nq = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    const int nee = grid::NUM_EES;
    const int grad_count = 6 * nv * nee;
    const int hess_count = 6 * nv * nv * nee;

    // ----- read inputs (q is nq-wide) -----
    std::vector<T> h_q(nq);
    read_vector(h_q.data(), nq);
    print_vector("input_q", h_q.data(), nq);

    T *d_q;
    gpuErrchk(cudaMalloc((void **)&d_q, nq * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_q, h_q.data(), nq * sizeof(T), cudaMemcpyHostToDevice));

    // ----- (1a) gradient device-function path -----
    T *d_grad;
    gpuErrchk(cudaMalloc((void **)&d_grad, grad_count * sizeof(T)));
    std::vector<T> h_grad(grad_count);
    const size_t dg_smem = grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    gpuErrchk(cudaFuncSetAttribute(spherical_deepose_device_runner<T>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(dg_smem)));
    spherical_deepose_device_runner<T><<<1, g_num_threads, dg_smem>>>(d_grad, d_q, d_robot_model);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_grad.data(), d_grad, grad_count * sizeof(T), cudaMemcpyDeviceToHost));
    print_vector("end_effector_pose_gradient", h_grad.data(), grad_count);

    // ----- (1b) hessian device-function path (+ co-computed gradient) -----
    T *d_hess, *d_grad2;
    gpuErrchk(cudaMalloc((void **)&d_hess, hess_count * sizeof(T)));
    gpuErrchk(cudaMalloc((void **)&d_grad2, grad_count * sizeof(T)));
    std::vector<T> h_hess(hess_count), h_grad2(grad_count);
    const size_t d2_smem = grid::END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>();
    gpuErrchk(cudaFuncSetAttribute(spherical_d2eepose_device_runner<T>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(d2_smem)));
    spherical_d2eepose_device_runner<T><<<1, g_num_threads, d2_smem>>>(d_hess, d_grad2, d_q, d_robot_model);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_hess.data(), d_hess, hess_count * sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_grad2.data(), d_grad2, grad_count * sizeof(T), cudaMemcpyDeviceToHost));
    print_vector("end_effector_pose_hessian", h_hess.data(), hess_count);
    print_vector("end_effector_pose_hessian_gradient", h_grad2.data(), grad_count);

    // ----- (2) host batch wrappers over B IDENTICAL timesteps -----
    const int B = 4;
    grid::gridData<T> *hd_data = grid::init_gridData<T, B>();
    for (int k = 0; k < B; ++k) {
        for (int i = 0; i < nq; ++i) {
            hd_data->h_q_qd_u[k * 3 * nq + i] = h_q[i];   // q slot [0, nq)
        }
        for (int i = 0; i < 2 * nq; ++i) {
            hd_data->h_q_qd_u[k * 3 * nq + nq + i] = static_cast<T>(0);  // qd|u unused
        }
    }
    const dim3 block_dimms(1, 1, 1);
    const dim3 thread_dimms(g_num_threads, 1, 1);

    grid::end_effector_pose_gradient<T, false>(hd_data, d_robot_model, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k) {
        std::vector<T> row(grad_count);
        for (int i = 0; i < grad_count; ++i) row[i] = hd_data->h_end_effector_pose_gradient[k * grad_count + i];
        print_vector("end_effector_pose_gradient_batch_" + std::to_string(k), row.data(), grad_count);
    }

    grid::end_effector_pose_hessian<T, false>(hd_data, d_robot_model, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k) {
        std::vector<T> blk(hess_count);
        for (int i = 0; i < hess_count; ++i) blk[i] = hd_data->h_end_effector_pose_hessian[k * hess_count + i];
        print_vector("end_effector_pose_hessian_batch_" + std::to_string(k), blk.data(), hess_count);
    }

    gpuErrchk(cudaFree(d_grad2));
    gpuErrchk(cudaFree(d_hess));
    gpuErrchk(cudaFree(d_grad));
    gpuErrchk(cudaFree(d_q));
    grid::close_grid<T>(streams, d_robot_model, hd_data);
}

int main(int argc, char **argv) {
    if (argc > 1) {
        int requested = std::atoi(argv[1]);
        g_num_threads = requested > 0 ? requested : 0;
    }
    if (g_num_threads <= 0 || g_num_threads > grid::MAX_PERF_LEVEL_THREADS) {
        g_num_threads = grid::MAX_PERF_LEVEL_THREADS;
    }
    const char *equiv_t = std::getenv("GRID_EQUIV_T");
    if (equiv_t != nullptr && std::string(equiv_t) == "double") {
        run<double>();
    } else {
        run<float>();
    }
    return 0;
}

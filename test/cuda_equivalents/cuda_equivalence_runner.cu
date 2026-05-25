#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "grid.cuh"


// Block thread count for all kernel launches. Defaults to 32 (one warp) and is
// overridable via argv[1] so the test harness can sweep warp counts to catch
// thread-count-dependent races.
int g_num_threads = 32;

template <typename T>
__global__ void runtime_probe_kernel(T *dst) {
    for (int ind = threadIdx.x; ind < grid::NUM_VEL; ind += blockDim.x) {
        dst[ind] = static_cast<T>(10 + ind);
    }
}

#if GRID_CUDA_FLOATING_BASE
template <typename T>
__device__ void load_floating_inputs(
    T *s_q, T *s_qd, T *s_u, const T *d_q, const T *d_qd, const T *d_u
) {
    for (int ind = threadIdx.x; ind < grid::NUM_JOINTS; ind += blockDim.x) {
        s_q[ind] = d_q[ind];
    }
    for (int ind = threadIdx.x; ind < grid::NUM_VEL; ind += blockDim.x) {
        s_qd[ind] = d_qd[ind];
        s_u[ind] = d_u[ind];
    }
    __syncthreads();
}

template <typename T>
__global__ void floating_inverse_dynamics_runner(
    T *d_out, const T *d_q, const T *d_qd, const T *d_u,
    const grid::robotModel<T> *d_robot_model, const T gravity
) {
    __shared__ T s_q[grid::NUM_JOINTS];
    __shared__ T s_qd[grid::NUM_VEL];
    __shared__ T s_u[grid::NUM_VEL];
    __shared__ T s_out[grid::NUM_VEL];
    load_floating_inputs(s_q, s_qd, s_u, d_q, d_qd, d_u);
    grid::inverse_dynamics_device<T>(s_out, s_q, s_qd, s_u, d_robot_model, gravity);
    __syncthreads();
    for (int ind = threadIdx.x; ind < grid::NUM_VEL; ind += blockDim.x) {
        d_out[ind] = s_out[ind];
    }
}

template <typename T>
__global__ void floating_direct_minv_runner(
    T *d_out, const T *d_q, const grid::robotModel<T> *d_robot_model
) {
    __shared__ T s_q[grid::NUM_JOINTS];
    __shared__ T s_out[grid::NUM_VEL * grid::NUM_VEL];
    for (int ind = threadIdx.x; ind < grid::NUM_JOINTS; ind += blockDim.x) {
        s_q[ind] = d_q[ind];
    }
    __syncthreads();
    grid::direct_minv_device<T>(s_out, s_q, d_robot_model);
    __syncthreads();
    for (int ind = threadIdx.x; ind < grid::NUM_VEL * grid::NUM_VEL; ind += blockDim.x) {
        d_out[ind] = s_out[ind];
    }
}

template <typename T>
__global__ void floating_forward_dynamics_runner(
    T *d_out, const T *d_q, const T *d_qd, const T *d_u,
    const grid::robotModel<T> *d_robot_model, const T gravity
) {
    __shared__ T s_q[grid::NUM_JOINTS];
    __shared__ T s_qd[grid::NUM_VEL];
    __shared__ T s_u[grid::NUM_VEL];
    __shared__ T s_out[grid::NUM_VEL];
    load_floating_inputs(s_q, s_qd, s_u, d_q, d_qd, d_u);
    grid::forward_dynamics_device<T>(s_out, s_q, s_qd, s_u, d_robot_model, gravity);
    __syncthreads();
    for (int ind = threadIdx.x; ind < grid::NUM_VEL; ind += blockDim.x) {
        d_out[ind] = s_out[ind];
    }
}

template <typename T>
__global__ void floating_inverse_dynamics_gradient_runner(
    T *d_out, const T *d_q, const T *d_qd,
    const grid::robotModel<T> *d_robot_model, const T gravity
) {
    __shared__ T s_q[grid::NUM_JOINTS];
    __shared__ T s_qd[grid::NUM_VEL];
    __shared__ T s_out[grid::NUM_VEL * 2 * grid::NUM_VEL];
    for (int ind = threadIdx.x; ind < grid::NUM_JOINTS; ind += blockDim.x) {
        s_q[ind] = d_q[ind];
    }
    for (int ind = threadIdx.x; ind < grid::NUM_VEL; ind += blockDim.x) {
        s_qd[ind] = d_qd[ind];
    }
    __syncthreads();
    grid::inverse_dynamics_gradient_device<T>(s_out, s_q, s_qd, d_robot_model, gravity);
    __syncthreads();
    for (int ind = threadIdx.x; ind < grid::NUM_VEL * 2 * grid::NUM_VEL; ind += blockDim.x) {
        d_out[ind] = s_out[ind];
    }
}

template <typename T>
__global__ void floating_forward_dynamics_gradient_runner(
    T *d_out, const T *d_q, const T *d_qd, const T *d_u,
    const grid::robotModel<T> *d_robot_model, const T gravity
) {
    __shared__ T s_q[grid::NUM_JOINTS];
    __shared__ T s_qd[grid::NUM_VEL];
    __shared__ T s_u[grid::NUM_VEL];
    __shared__ T s_out[grid::NUM_VEL * 2 * grid::NUM_VEL];
    load_floating_inputs(s_q, s_qd, s_u, d_q, d_qd, d_u);
    grid::forward_dynamics_gradient_device<T>(s_out, s_q, s_qd, s_u, d_robot_model, gravity);
    __syncthreads();
    for (int ind = threadIdx.x; ind < grid::NUM_VEL * 2 * grid::NUM_VEL; ind += blockDim.x) {
        d_out[ind] = s_out[ind];
    }
}
#endif

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
void print_matrix_col_major(
    const std::string &name, const T *data, int rows, int cols
) {
    std::cout << "BEGIN " << name << " " << rows << " " << cols << "\n";
    std::cout << std::setprecision(10);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            if (col) {
                std::cout << " ";
            }
            std::cout << static_cast<double>(data[row + rows * col]);
        }
        std::cout << "\n";
    }
    std::cout << "END " << name << "\n";
}

template <typename T>
void print_vector(const std::string &name, const T *data, int count) {
    print_matrix_col_major(name, data, 1, count);
}

#if GRID_CUDA_FLOATING_BASE
bool floating_algorithm_requested(const std::string &name) {
    const char *raw = std::getenv("GRID_CUDA_FLOATING_ALGORITHMS");
    if (raw == nullptr || std::string(raw).empty()) {
        return name == "inverse_dynamics" ||
               name == "direct_minv" ||
               name == "forward_dynamics" ||
               name == "inverse_dynamics_gradient_q" ||
               name == "inverse_dynamics_gradient_qd" ||
               name == "forward_dynamics_gradient_q" ||
               name == "forward_dynamics_gradient_qd" ||
               name == "aba" ||
               name == "crba" ||
               name == "end_effector_pose" ||
               name == "end_effector_pose_gradient" ||
               name == "end_effector_pose_hessian";
    }
    const std::string selected(raw);
    if (selected == "all") {
        return true;
    }
    size_t start = 0;
    while (start <= selected.size()) {
        size_t comma = selected.find(',', start);
        std::string item = selected.substr(
            start,
            comma == std::string::npos ? std::string::npos : comma - start
        );
        size_t first = item.find_first_not_of(" \t\n\r");
        size_t last = item.find_last_not_of(" \t\n\r");
        if (first != std::string::npos &&
            item.substr(first, last - first + 1) == name) {
            return true;
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return false;
}
#endif

template <typename T>
void run() {
    const T gravity = static_cast<T>(9.81);
    const dim3 block_dimms(1, 1, 1);
    const dim3 thread_dimms(g_num_threads, 1, 1);

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

#if GRID_CUDA_FLOATING_BASE
    std::vector<T> h_q(grid::NUM_JOINTS);
    std::vector<T> h_qd(grid::NUM_VEL);
    std::vector<T> h_u(grid::NUM_VEL);
    std::vector<T> h_q_qd(grid::NUM_JOINTS + grid::NUM_VEL);
    std::vector<T> h_q_qd_u(grid::NUM_JOINTS + 2 * grid::NUM_VEL);
    std::vector<T> h_vec(grid::NUM_VEL);
    std::vector<T> h_mat(grid::NUM_VEL * grid::NUM_VEL);
    std::vector<T> h_grad(grid::NUM_VEL * 2 * grid::NUM_VEL);
    std::vector<T> h_ee(6 * grid::NUM_EES);
    std::vector<T> h_dee(6 * grid::NUM_JOINTS * grid::NUM_EES);
    std::vector<T> h_d2ee(6 * grid::NUM_JOINTS * grid::NUM_JOINTS * grid::NUM_EES);

    read_vector(h_q.data(), grid::NUM_JOINTS);
    read_vector(h_qd.data(), grid::NUM_VEL);
    read_vector(h_u.data(), grid::NUM_VEL);

    print_vector("input_q", h_q.data(), grid::NUM_JOINTS);
    print_vector("input_qd", h_qd.data(), grid::NUM_VEL);
    print_vector("input_u", h_u.data(), grid::NUM_VEL);

    T *d_q;
    T *d_qd;
    T *d_u;
    T *d_q_qd;
    T *d_q_qd_u;
    T *d_zero;
    T *d_vec;
    T *d_mat;
    T *d_grad;
    T *d_ee;
    T *d_dee;
    T *d_d2ee;
    for (int i = 0; i < grid::NUM_JOINTS; ++i) {
        h_q_qd[i] = h_q[i];
        h_q_qd_u[i] = h_q[i];
    }
    for (int i = 0; i < grid::NUM_VEL; ++i) {
        h_q_qd[grid::NUM_JOINTS + i] = h_qd[i];
        h_q_qd_u[grid::NUM_JOINTS + i] = h_qd[i];
        h_q_qd_u[grid::NUM_JOINTS + grid::NUM_VEL + i] = h_u[i];
    }
    gpuErrchk(cudaMalloc((void**)&d_q, grid::NUM_JOINTS * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_qd, grid::NUM_VEL * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_u, grid::NUM_VEL * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_q_qd, (grid::NUM_JOINTS + grid::NUM_VEL) * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_q_qd_u, (grid::NUM_JOINTS + 2 * grid::NUM_VEL) * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_zero, grid::NUM_VEL * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_vec, grid::NUM_VEL * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_mat, grid::NUM_JOINTS * grid::NUM_JOINTS * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_grad, grid::NUM_VEL * 2 * grid::NUM_VEL * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_ee, 6 * grid::NUM_EES * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_dee, 6 * grid::NUM_JOINTS * grid::NUM_EES * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_d2ee, 6 * grid::NUM_JOINTS * grid::NUM_JOINTS * grid::NUM_EES * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_q, h_q.data(), grid::NUM_JOINTS * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_qd, h_qd.data(), grid::NUM_VEL * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, h_u.data(), grid::NUM_VEL * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_q_qd, h_q_qd.data(), h_q_qd.size() * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_q_qd_u, h_q_qd_u.data(), h_q_qd_u.size() * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_zero, 0, grid::NUM_VEL * sizeof(T)));

    runtime_probe_kernel<T><<<1, g_num_threads>>>(d_vec);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_vec.data(), d_vec, grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
    print_vector("runtime_probe", h_vec.data(), grid::NUM_VEL);

    gpuErrchk(cudaFuncSetAttribute(
        floating_inverse_dynamics_runner<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(grid::ID_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>())
    ));
    gpuErrchk(cudaFuncSetAttribute(
        grid::direct_minv_kernel<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(grid::MINV_DYNAMIC_SHARED_MEM_BYTES<T>())
    ));
    gpuErrchk(cudaFuncSetAttribute(
        floating_forward_dynamics_runner<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(grid::FD_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>())
    ));
    gpuErrchk(cudaFuncSetAttribute(
        grid::aba_kernel<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(grid::ABA_DYNAMIC_SHARED_MEM_BYTES<T>())
    ));
    gpuErrchk(cudaFuncSetAttribute(
        grid::crba_kernel<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(grid::CRBA_DYNAMIC_SHARED_MEM_BYTES<T>())
    ));
    gpuErrchk(cudaFuncSetAttribute(
        grid::end_effector_pose_kernel<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(grid::EE_POS_DYNAMIC_SHARED_MEM_BYTES<T>())
    ));
    if (floating_algorithm_requested("end_effector_pose_gradient")) {
        gpuErrchk(cudaFuncSetAttribute(
            grid::end_effector_pose_gradient_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(grid::DEE_POS_DYNAMIC_SHARED_MEM_BYTES<T>())
        ));
    }
    if (floating_algorithm_requested("end_effector_pose_hessian")) {
        gpuErrchk(cudaFuncSetAttribute(
            grid::end_effector_pose_gradient_hessian_kernel<T>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(grid::D2EE_POS_DYNAMIC_SHARED_MEM_BYTES<T>())
        ));
    }

    if (floating_algorithm_requested("inverse_dynamics")) {
        floating_inverse_dynamics_runner<T><<<1, g_num_threads, grid::ID_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_vec, d_q, d_qd, d_zero, d_robot_model, gravity
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_vec.data(), d_vec, grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
        print_vector("inverse_dynamics", h_vec.data(), grid::NUM_VEL);
    }

    if (floating_algorithm_requested("direct_minv")) {
        grid::direct_minv_kernel<T><<<1, g_num_threads, grid::MINV_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_mat, hd_data->d_workspace, d_q, grid::NUM_JOINTS, d_robot_model, 1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_mat.data(), d_mat, grid::NUM_VEL * grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
        print_matrix_col_major("direct_minv", h_mat.data(), grid::NUM_VEL, grid::NUM_VEL);
    }

    if (floating_algorithm_requested("forward_dynamics")) {
        floating_forward_dynamics_runner<T><<<1, g_num_threads, grid::FD_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_vec, d_q, d_qd, d_u, d_robot_model, gravity
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_vec.data(), d_vec, grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
        print_vector("forward_dynamics", h_vec.data(), grid::NUM_VEL);
    }

    if (floating_algorithm_requested("aba")) {
        grid::aba_kernel<T><<<1, g_num_threads, grid::ABA_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_vec,
            hd_data->d_workspace,
            d_q_qd_u,
            grid::NUM_JOINTS + 2 * grid::NUM_VEL,
            d_robot_model,
            gravity,
            1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_vec.data(), d_vec, grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
        print_vector("aba", h_vec.data(), grid::NUM_VEL);
    }

    if (floating_algorithm_requested("crba")) {
        grid::crba_kernel<T><<<1, g_num_threads, grid::CRBA_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_mat,
            d_q_qd,
            grid::NUM_JOINTS + grid::NUM_VEL,
            d_robot_model,
            gravity,
            1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_mat.data(), d_mat, grid::NUM_VEL * grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
        print_matrix_col_major("crba", h_mat.data(), grid::NUM_VEL, grid::NUM_VEL);
    }

    if (floating_algorithm_requested("end_effector_pose")) {
        grid::end_effector_pose_kernel<T><<<1, g_num_threads, grid::EE_POS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_ee,
            d_q,
            grid::NUM_JOINTS,
            d_robot_model,
            1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_ee.data(), d_ee, 6 * grid::NUM_EES * sizeof(T), cudaMemcpyDeviceToHost));
        print_vector("end_effector_pose", h_ee.data(), 6 * grid::NUM_EES);
    }

    if (floating_algorithm_requested("end_effector_pose_gradient")) {
        grid::end_effector_pose_gradient_kernel<T><<<1, g_num_threads, grid::DEE_POS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_dee,
            hd_data->d_workspace,
            d_q,
            grid::NUM_JOINTS,
            d_robot_model,
            1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_dee.data(), d_dee, 6 * grid::NUM_JOINTS * grid::NUM_EES * sizeof(T), cudaMemcpyDeviceToHost));
        print_vector("end_effector_pose_gradient", h_dee.data(), 6 * grid::NUM_JOINTS * grid::NUM_EES);
    }

    if (floating_algorithm_requested("end_effector_pose_hessian")) {
        if (grid::GRID_D2EE_USES_WORKSPACE_TEMP) {
            gpuErrchk(grid::grid_begin_l2_persisting(
                0, hd_data->d_workspace, grid::GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()
            ));
        }
        grid::end_effector_pose_gradient_hessian_kernel<T><<<1, g_num_threads, grid::D2EE_POS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_d2ee,
            d_dee,
            hd_data->d_workspace,
            d_q,
            grid::NUM_JOINTS,
            d_robot_model,
            1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        if (grid::GRID_D2EE_USES_WORKSPACE_TEMP) {
            gpuErrchk(grid::grid_end_l2_persisting(0));
        }
        gpuErrchk(cudaMemcpy(h_d2ee.data(), d_d2ee, 6 * grid::NUM_JOINTS * grid::NUM_JOINTS * grid::NUM_EES * sizeof(T), cudaMemcpyDeviceToHost));
        print_vector("end_effector_pose_hessian", h_d2ee.data(), 6 * grid::NUM_JOINTS * grid::NUM_JOINTS * grid::NUM_EES);
    }

    if (floating_algorithm_requested("inverse_dynamics_gradient_q") ||
        floating_algorithm_requested("inverse_dynamics_gradient_qd")) {
        grid::inverse_dynamics_gradient_kernel<T><<<1, g_num_threads, grid::ID_DU_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_grad,
            hd_data->d_workspace,
            d_q_qd,
            grid::NUM_JOINTS + grid::NUM_VEL,
            d_robot_model,
            gravity,
            1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_grad.data(), d_grad, grid::NUM_VEL * 2 * grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
        print_matrix_col_major("inverse_dynamics_gradient_q", h_grad.data(), grid::NUM_VEL, grid::NUM_VEL);
        print_matrix_col_major(
            "inverse_dynamics_gradient_qd",
            &h_grad[grid::NUM_VEL * grid::NUM_VEL],
            grid::NUM_VEL,
            grid::NUM_VEL
        );
    }

    if (floating_algorithm_requested("forward_dynamics_gradient_q") ||
        floating_algorithm_requested("forward_dynamics_gradient_qd")) {
        grid::forward_dynamics_gradient_kernel<T><<<1, g_num_threads, grid::FD_DU_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_grad,
            hd_data->d_workspace,
            d_q_qd_u,
            grid::NUM_JOINTS + 2 * grid::NUM_VEL,
            d_robot_model,
            gravity,
            1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_grad.data(), d_grad, grid::NUM_VEL * 2 * grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
        print_matrix_col_major("forward_dynamics_gradient_q", h_grad.data(), grid::NUM_VEL, grid::NUM_VEL);
        print_matrix_col_major(
            "forward_dynamics_gradient_qd",
            &h_grad[grid::NUM_VEL * grid::NUM_VEL],
            grid::NUM_VEL,
            grid::NUM_VEL
        );
    }

    gpuErrchk(cudaFree(d_q));
    gpuErrchk(cudaFree(d_qd));
    gpuErrchk(cudaFree(d_u));
    gpuErrchk(cudaFree(d_q_qd));
    gpuErrchk(cudaFree(d_q_qd_u));
    gpuErrchk(cudaFree(d_zero));
    gpuErrchk(cudaFree(d_vec));
    gpuErrchk(cudaFree(d_mat));
    gpuErrchk(cudaFree(d_grad));
    gpuErrchk(cudaFree(d_ee));
    gpuErrchk(cudaFree(d_dee));
    gpuErrchk(cudaFree(d_d2ee));
#else
    read_vector(hd_data->h_q, grid::NUM_JOINTS);
    read_vector(&hd_data->h_q_qd[grid::NUM_JOINTS], grid::NUM_JOINTS);
    read_vector(&hd_data->h_q_qd_u[2 * grid::NUM_JOINTS], grid::NUM_JOINTS);

    for (int i = 0; i < grid::NUM_JOINTS; ++i) {
        hd_data->h_q_qd[i] = hd_data->h_q[i];
        hd_data->h_q_qd_u[i] = hd_data->h_q[i];
        hd_data->h_q_qd_u[i + grid::NUM_JOINTS] =
            hd_data->h_q_qd[i + grid::NUM_JOINTS];
        hd_data->h_qdd[i] = static_cast<T>(0);
    }

    print_vector("input_q", hd_data->h_q, grid::NUM_JOINTS);
    print_vector("input_qd", &hd_data->h_q_qd[grid::NUM_JOINTS], grid::NUM_JOINTS);
    print_vector("input_u", &hd_data->h_q_qd_u[2 * grid::NUM_JOINTS], grid::NUM_JOINTS);

    runtime_probe_kernel<T><<<1, g_num_threads>>>(hd_data->d_c);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(
        hd_data->h_c, hd_data->d_c, grid::NUM_JOINTS * sizeof(T),
        cudaMemcpyDeviceToHost
    ));
    print_vector("runtime_probe", hd_data->h_c, grid::NUM_JOINTS);

    grid::inverse_dynamics<T, false, true>(
        hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    print_vector("inverse_dynamics", hd_data->h_c, grid::NUM_JOINTS);

    grid::direct_minv<T, true>(
        hd_data, d_robot_model, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    print_matrix_col_major(
        "direct_minv", hd_data->h_Minv, grid::NUM_JOINTS, grid::NUM_JOINTS
    );

    grid::forward_dynamics<T>(
        hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    print_vector("forward_dynamics", hd_data->h_qdd, grid::NUM_JOINTS);

    grid::inverse_dynamics_gradient<T, false, true>(
        hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    print_matrix_col_major(
        "inverse_dynamics_gradient_q",
        hd_data->h_dc_du,
        grid::NUM_JOINTS,
        grid::NUM_JOINTS
    );
    print_matrix_col_major(
        "inverse_dynamics_gradient_qd",
        &hd_data->h_dc_du[grid::NUM_JOINTS * grid::NUM_JOINTS],
        grid::NUM_JOINTS,
        grid::NUM_JOINTS
    );

    grid::forward_dynamics_gradient<T, false>(
        hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    print_matrix_col_major(
        "forward_dynamics_gradient_q",
        hd_data->h_df_du,
        grid::NUM_JOINTS,
        grid::NUM_JOINTS
    );
    print_matrix_col_major(
        "forward_dynamics_gradient_qd",
        &hd_data->h_df_du[grid::NUM_JOINTS * grid::NUM_JOINTS],
        grid::NUM_JOINTS,
        grid::NUM_JOINTS
    );

    grid::aba<T>(
        hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    print_vector("aba", hd_data->h_qdd, grid::NUM_JOINTS);

    grid::crba<T, true>(
        hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    print_matrix_col_major(
        "crba", hd_data->h_M, grid::NUM_JOINTS, grid::NUM_JOINTS
    );
#endif

    grid::close_grid<T>(streams, d_robot_model, hd_data);
}

int main(int argc, char **argv) {
    // Optional argv[1] = block thread count (default 32). Lets the test sweep
    // launches across warp counts to catch thread-count-dependent races (e.g.
    // a missing __syncthreads between a write phase and a read/accumulate phase
    // that is correct only within a single warp).
    if (argc > 1) {
        int requested = std::atoi(argv[1]);
        // 0 (or non-positive) is the sentinel for "launch at the robot's
        // MAX_PERF_LEVEL_THREADS" — see the clamp below. >0 requests an explicit count.
        g_num_threads = requested > 0 ? requested : 0;
    }
    // Dynamic clamp: the kernels carry __launch_bounds__(tier_max_threads<TIER>())
    // (= grid::MAX_PERF_LEVEL_THREADS at the default tier), so launching with MORE
    // threads is an invalid configuration. Derive the cap from the generated
    // header per-robot rather than hardcoding it; 0 => use SUGGESTED exactly.
    if (g_num_threads <= 0 || g_num_threads > grid::MAX_PERF_LEVEL_THREADS) {
        g_num_threads = grid::MAX_PERF_LEVEL_THREADS;
    }
    run<float>();
    return 0;
}

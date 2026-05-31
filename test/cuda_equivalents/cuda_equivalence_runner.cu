#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "grid.cuh"


// Mimic robots (fr3, h1_2) do NOT emit gradient algorithms (the G0 footgun guard
// refuses mimic-gradient codegen — deferred to T3-finisher). The test passes
// -DGRID_RUNNER_SKIP_GRADIENTS=1 for those robots so this runner compiles against
// their gradient-free header. Default 0 for non-mimic robots (full surface).
#ifndef GRID_RUNNER_SKIP_GRADIENTS
#define GRID_RUNNER_SKIP_GRADIENTS 0
#endif

// Finer-grained: ee_pose gradient/hessian (kinematic 1st/2nd order) land in a
// LATER mimic phase (P4) than the dynamics gradients id_du/fd_du (P3). Fixed-base
// mimic emits id_du/fd_du (so GRID_RUNNER_SKIP_GRADIENTS=0) but NOT the ee_pose
// gradients yet, so the test sets -DGRID_RUNNER_SKIP_EEPOSE_GRADIENTS=1 for mimic
// robots to skip only those. Defaults to GRID_RUNNER_SKIP_GRADIENTS so non-mimic
// robots and the floating-skip path behave exactly as before.
#ifndef GRID_RUNNER_SKIP_EEPOSE_GRADIENTS
#define GRID_RUNNER_SKIP_EEPOSE_GRADIENTS GRID_RUNNER_SKIP_GRADIENTS
#endif


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
    const grid::robotModel<T> *d_robot_model, const T gravity, T *d_f_ext = nullptr
) {
    __shared__ T s_q[grid::NUM_JOINTS];
    __shared__ T s_qd[grid::NUM_VEL];
    __shared__ T s_u[grid::NUM_VEL];
    __shared__ T s_out[grid::NUM_VEL];
    load_floating_inputs(s_q, s_qd, s_u, d_q, d_qd, d_u);
    grid::inverse_dynamics_device<T>(s_out, s_q, s_qd, s_u, d_robot_model, d_f_ext, gravity);
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

// forward_dynamics_device is tier-aware (mirrors idsva_so_device / d2ee_device).
// We pick the smallest-smem rung that's still safe: TIER_MINIMAL routes the whole
// FD inner s_temp arena (incl. the Minv-F band at its tail) to L2-pinned
// d_workspace, freeing ~120 KB of smem on humanoid-scale robots. For smaller
// robots TIER_MINIMAL is byte-identical to TIER_SHARED in numerical output (only
// the pointer routing changes). The d_workspace pointer comes from the
// already-allocated hd_data->d_workspace (size GRID_WORKSPACE_BYTES_PER_TIMESTEP).
template <typename T>
__global__ void floating_forward_dynamics_runner(
    T *d_out, const T *d_q, const T *d_qd, const T *d_u,
    const grid::robotModel<T> *d_robot_model, const T gravity,
    unsigned char *d_workspace, T *d_f_ext = nullptr
) {
    __shared__ T s_q[grid::NUM_JOINTS];
    __shared__ T s_qd[grid::NUM_VEL];
    __shared__ T s_u[grid::NUM_VEL];
    __shared__ T s_out[grid::NUM_VEL];
    load_floating_inputs(s_q, s_qd, s_u, d_q, d_qd, d_u);
    grid::forward_dynamics_device<T, grid::TIER_MINIMAL>(
        s_out, s_q, s_qd, s_u, d_robot_model, d_f_ext, gravity,
        reinterpret_cast<T *>(d_workspace));
    __syncthreads();
    for (int ind = threadIdx.x; ind < grid::NUM_VEL; ind += blockDim.x) {
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

// Register a kernel's opt-in dynamic shared memory, but if the request exceeds
// this device's per-block cap, emit the standard GRID message and exit cleanly
// (rc=2) instead of letting gpuErrchk hard-abort with "invalid argument". The
// Python harness (_run_runner) treats that message as a SKIP, so a robot whose
// PERF-tier kernel doesn't fit this GPU (e.g. h1_2-floating direct_minv) skips
// honestly rather than failing. Unlike the generated init_grid_kernel_attrs,
// this floating runner block registers a few RUNNER-LOCAL kernels too, so it
// needs its own guard. (A spilled tier would fit; default tier is PERF.)
template <typename FuncT>
static void grid_runner_set_smem_or_skip(FuncT func, const char *name, size_t bytes) {
    int dev = 0;
    gpuErrchk(cudaGetDevice(&dev));
    int smem_max = 0;
    gpuErrchk(cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    if (bytes > static_cast<size_t>(smem_max)) {
        fprintf(stderr,
                "GRID shared-memory request for %s is %zu bytes, but this device "
                "supports %d bytes per block\n",
                name, bytes, smem_max);
        std::exit(2);
    }
    gpuErrchk(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(bytes)));
}

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
    std::vector<T> h_dee(6 * grid::NUM_VEL * grid::NUM_EES);
    std::vector<T> h_d2ee(6 * grid::NUM_VEL * grid::NUM_VEL * grid::NUM_EES);

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
    gpuErrchk(cudaMalloc((void**)&d_dee, 6 * grid::NUM_VEL * grid::NUM_EES * sizeof(T)));
    gpuErrchk(cudaMalloc((void**)&d_d2ee, 6 * grid::NUM_VEL * grid::NUM_VEL * grid::NUM_EES * sizeof(T)));
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

    // External forces (opt-in via GRID_RUNNER_FEXT=1): read 6*NUM_BODIES values
    // (body-major, local-frame [angular; linear]) into hd_data->d_f_ext and run
    // the fext-aware launches below. d_f_ext_active is nullptr in the default
    // (no-fext) path, so existing behavior is byte-identical.
    const bool g_use_fext = (std::getenv("GRID_RUNNER_FEXT") != nullptr);
    T *d_f_ext_active = nullptr;
    if (g_use_fext) {
        read_vector(hd_data->h_f_ext, 6 * grid::NUM_BODIES);
        gpuErrchk(cudaMemcpy(hd_data->d_f_ext, hd_data->h_f_ext,
                             6 * grid::NUM_BODIES * sizeof(T), cudaMemcpyHostToDevice));
        d_f_ext_active = hd_data->d_f_ext;
        print_vector("input_f_ext", hd_data->h_f_ext, 6 * grid::NUM_BODIES);
    }

    grid_runner_set_smem_or_skip(floating_inverse_dynamics_runner<T>,
        "inverse_dynamics", grid::ID_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>());
    grid_runner_set_smem_or_skip(grid::direct_minv_kernel<T>,
        "direct_minv", grid::MINV_DYNAMIC_SHARED_MEM_BYTES<T>());
    grid_runner_set_smem_or_skip(floating_forward_dynamics_runner<T>,
        "forward_dynamics", grid::FD_DEVICE_INLINE_SMEM_BYTES<T, grid::TIER_MINIMAL>());
    grid_runner_set_smem_or_skip(grid::aba_kernel<T>,
        "aba", grid::ABA_DYNAMIC_SHARED_MEM_BYTES<T>());
    grid_runner_set_smem_or_skip(grid::crba_kernel<T>,
        "crba", grid::CRBA_DYNAMIC_SHARED_MEM_BYTES<T>());
    grid_runner_set_smem_or_skip(grid::end_effector_pose_kernel<T>,
        "end_effector_pose", grid::EE_POS_DYNAMIC_SHARED_MEM_BYTES<T>());
    // ee-pose grad/hessian SMEM registration: gate on SKIP_EEPOSE so floating
    // mimic (id_du/fd_du emitted, ee derivatives not) skips only the ee kernels.
#if !GRID_RUNNER_SKIP_EEPOSE_GRADIENTS
    if (floating_algorithm_requested("end_effector_pose_gradient")) {
        grid_runner_set_smem_or_skip(grid::end_effector_pose_gradient_kernel<T>,
            "end_effector_pose_gradient", grid::DEE_POS_DYNAMIC_SHARED_MEM_BYTES<T>());
    }
    if (floating_algorithm_requested("end_effector_pose_hessian")) {
        grid_runner_set_smem_or_skip(grid::end_effector_pose_gradient_hessian_kernel<T>,
            "end_effector_pose_hessian", grid::D2EE_POS_DYNAMIC_SHARED_MEM_BYTES<T>());
    }
#endif  // !GRID_RUNNER_SKIP_EEPOSE_GRADIENTS

    if (floating_algorithm_requested("inverse_dynamics")) {
        floating_inverse_dynamics_runner<T><<<1, g_num_threads, grid::ID_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_vec, d_q, d_qd, d_zero, d_robot_model, gravity, /*d_f_ext=*/nullptr
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
        floating_forward_dynamics_runner<T><<<1, g_num_threads, grid::FD_DEVICE_INLINE_SMEM_BYTES<T, grid::TIER_MINIMAL>()>>>(
            d_vec, d_q, d_qd, d_u, d_robot_model, gravity, hd_data->d_workspace, /*d_f_ext=*/nullptr
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
            /*d_f_ext=*/nullptr,
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
            hd_data->d_workspace,
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

#if !GRID_RUNNER_SKIP_GRADIENTS
    // ee_pose grad/hessian for FLOATING base. Gated separately from the dynamics
    // gradients so a floating MIMIC robot (which emits id_du/fd_du but NOT the
    // ee-pose derivatives — their floating-root subspace fold is deferred) can
    // compile the id_du/fd_du block while skipping the un-emitted ee kernels.
    // GRID_RUNNER_SKIP_EEPOSE_GRADIENTS defaults to GRID_RUNNER_SKIP_GRADIENTS,
    // so non-mimic floating (both 0) still compiles the ee blocks as before.
#if !GRID_RUNNER_SKIP_EEPOSE_GRADIENTS
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
        gpuErrchk(cudaMemcpy(h_dee.data(), d_dee, 6 * grid::NUM_VEL * grid::NUM_EES * sizeof(T), cudaMemcpyDeviceToHost));
        print_vector("end_effector_pose_gradient", h_dee.data(), 6 * grid::NUM_VEL * grid::NUM_EES);
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
        gpuErrchk(cudaMemcpy(h_d2ee.data(), d_d2ee, 6 * grid::NUM_VEL * grid::NUM_VEL * grid::NUM_EES * sizeof(T), cudaMemcpyDeviceToHost));
        print_vector("end_effector_pose_hessian", h_d2ee.data(), 6 * grid::NUM_VEL * grid::NUM_VEL * grid::NUM_EES);
    }
#endif  // !GRID_RUNNER_SKIP_EEPOSE_GRADIENTS

    if (floating_algorithm_requested("inverse_dynamics_gradient_q") ||
        floating_algorithm_requested("inverse_dynamics_gradient_qd")) {
        grid::inverse_dynamics_gradient_kernel<T><<<1, g_num_threads, grid::ID_DU_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_grad,
            hd_data->d_workspace,
            d_q_qd,
            grid::NUM_JOINTS + grid::NUM_VEL,
            /*d_f_ext=*/nullptr,
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
            /*d_f_ext=*/nullptr,
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
#endif  // !GRID_RUNNER_SKIP_GRADIENTS

    // External forces (opt-in via GRID_RUNNER_FEXT=1): re-run the dynamics that
    // thread d_f_ext, emitting *_fext-labeled outputs. The default outputs above
    // used nullptr (byte-identical to no-fext). d_f_ext_active was populated from
    // stdin earlier in this block.
    if (g_use_fext) {
        floating_inverse_dynamics_runner<T><<<1, g_num_threads, grid::ID_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_vec, d_q, d_qd, d_zero, d_robot_model, gravity, d_f_ext_active
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_vec.data(), d_vec, grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
        print_vector("inverse_dynamics_fext", h_vec.data(), grid::NUM_VEL);

        floating_forward_dynamics_runner<T><<<1, g_num_threads, grid::FD_DEVICE_INLINE_SMEM_BYTES<T, grid::TIER_MINIMAL>()>>>(
            d_vec, d_q, d_qd, d_u, d_robot_model, gravity, hd_data->d_workspace, d_f_ext_active
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_vec.data(), d_vec, grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
        print_vector("forward_dynamics_fext", h_vec.data(), grid::NUM_VEL);

        grid::aba_kernel<T><<<1, g_num_threads, grid::ABA_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_vec, hd_data->d_workspace, d_q_qd_u, grid::NUM_JOINTS + 2 * grid::NUM_VEL,
            d_f_ext_active, d_robot_model, gravity, 1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_vec.data(), d_vec, grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
        print_vector("aba_fext", h_vec.data(), grid::NUM_VEL);

#if !GRID_RUNNER_SKIP_GRADIENTS
        grid::inverse_dynamics_gradient_kernel<T><<<1, g_num_threads, grid::ID_DU_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_grad, hd_data->d_workspace, d_q_qd, grid::NUM_JOINTS + grid::NUM_VEL,
            d_f_ext_active, d_robot_model, gravity, 1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_grad.data(), d_grad, grid::NUM_VEL * 2 * grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
        print_matrix_col_major("inverse_dynamics_gradient_q_fext", h_grad.data(), grid::NUM_VEL, grid::NUM_VEL);
        print_matrix_col_major("inverse_dynamics_gradient_qd_fext",
            &h_grad[grid::NUM_VEL * grid::NUM_VEL], grid::NUM_VEL, grid::NUM_VEL);

        grid::forward_dynamics_gradient_kernel<T><<<1, g_num_threads, grid::FD_DU_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(
            d_grad, hd_data->d_workspace, d_q_qd_u, grid::NUM_JOINTS + 2 * grid::NUM_VEL,
            d_f_ext_active, d_robot_model, gravity, 1
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(h_grad.data(), d_grad, grid::NUM_VEL * 2 * grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
        print_matrix_col_major("forward_dynamics_gradient_q_fext", h_grad.data(), grid::NUM_VEL, grid::NUM_VEL);
        print_matrix_col_major("forward_dynamics_gradient_qd_fext",
            &h_grad[grid::NUM_VEL * grid::NUM_VEL], grid::NUM_VEL, grid::NUM_VEL);
#endif  // !GRID_RUNNER_SKIP_GRADIENTS
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

    // Gradient algorithms are NOT emitted for mimic robots (the G0 footgun guard
    // refuses mimic-gradient codegen — deferred to T3-finisher). The test passes
    // -DGRID_RUNNER_SKIP_GRADIENTS=1 for mimic robots so this runner compiles
    // against their (gradient-free) header; the suite already skips comparing
    // gradient algorithms for mimic robots (MIMIC_SUPPORTED_ALGORITHMS).
#if !GRID_RUNNER_SKIP_GRADIENTS
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
#endif

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

    grid::end_effector_pose<T>(
        hd_data, d_robot_model, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    print_vector("end_effector_pose", hd_data->h_eePos, 6 * grid::NUM_EES);

#if !GRID_RUNNER_SKIP_EEPOSE_GRADIENTS
    grid::end_effector_pose_gradient<T>(
        hd_data, d_robot_model, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    print_vector(
        "end_effector_pose_gradient", hd_data->h_deePos,
        6 * grid::NUM_VEL * grid::NUM_EES
    );

    grid::end_effector_pose_gradient_hessian<T>(
        hd_data, d_robot_model, 1, block_dimms, thread_dimms, streams
    );
    gpuErrchk(cudaPeekAtLastError());
    print_vector(
        "end_effector_pose_hessian", hd_data->h_d2eePos,
        6 * grid::NUM_VEL * grid::NUM_VEL * grid::NUM_EES
    );
#endif

    // External forces (opt-in via GRID_RUNNER_FEXT=1). The host wrappers read
    // hd_data->d_f_ext (body-major 6*NUM_BODIES local-frame); it is zeroed at
    // init so all the outputs above are byte-identical to the no-fext path.
    // Here we read a nonzero f_ext, copy it into d_f_ext, and re-run the
    // dynamics that thread external forces, emitting *_fext-labeled outputs.
    if (std::getenv("GRID_RUNNER_FEXT") != nullptr) {
        read_vector(hd_data->h_f_ext, 6 * grid::NUM_BODIES);
        gpuErrchk(cudaMemcpy(hd_data->d_f_ext, hd_data->h_f_ext,
                             6 * grid::NUM_BODIES * sizeof(T), cudaMemcpyHostToDevice));
        print_vector("input_f_ext", hd_data->h_f_ext, 6 * grid::NUM_BODIES);

        grid::inverse_dynamics<T, false, true>(
            hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
        );
        gpuErrchk(cudaPeekAtLastError());
        print_vector("inverse_dynamics_fext", hd_data->h_c, grid::NUM_JOINTS);

        grid::forward_dynamics<T>(
            hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
        );
        gpuErrchk(cudaPeekAtLastError());
        print_vector("forward_dynamics_fext", hd_data->h_qdd, grid::NUM_JOINTS);

        grid::aba<T>(
            hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
        );
        gpuErrchk(cudaPeekAtLastError());
        print_vector("aba_fext", hd_data->h_qdd, grid::NUM_JOINTS);

#if !GRID_RUNNER_SKIP_GRADIENTS
        grid::inverse_dynamics_gradient<T, false, true>(
            hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
        );
        gpuErrchk(cudaPeekAtLastError());
        print_matrix_col_major("inverse_dynamics_gradient_q_fext",
            hd_data->h_dc_du, grid::NUM_JOINTS, grid::NUM_JOINTS);
        print_matrix_col_major("inverse_dynamics_gradient_qd_fext",
            &hd_data->h_dc_du[grid::NUM_JOINTS * grid::NUM_JOINTS],
            grid::NUM_JOINTS, grid::NUM_JOINTS);

        grid::forward_dynamics_gradient<T, false>(
            hd_data, d_robot_model, gravity, 1, block_dimms, thread_dimms, streams
        );
        gpuErrchk(cudaPeekAtLastError());
        print_matrix_col_major("forward_dynamics_gradient_q_fext",
            hd_data->h_df_du, grid::NUM_JOINTS, grid::NUM_JOINTS);
        print_matrix_col_major("forward_dynamics_gradient_qd_fext",
            &hd_data->h_df_du[grid::NUM_JOINTS * grid::NUM_JOINTS],
            grid::NUM_JOINTS, grid::NUM_JOINTS);
#endif
    }
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

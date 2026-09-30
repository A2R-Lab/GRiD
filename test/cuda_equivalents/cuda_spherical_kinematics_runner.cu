// CUDA equivalence runner for SPHERICAL (ball) joint end_effector_pose +
// frame_jacobian (Tier-C kinematics). Spherical robots have NQ != NV (a 4-wide
// unit-quaternion q-block per ball joint, 3 v-slots), so the per-timestep INPUT
// slot is NQ-wide (= grid::NUM_JOINTS here, == get_num_pos()=nq). The FK chain-up
// consumes the joint's HOMOGENEOUS quaternion transform (keeps R, not R^T), built
// via the shared quaternion XmatsHom substitution on the joint's own 4-wide q-block.
//
// This runner exercises BOTH surfaces for each ported kinematics algorithm:
//   (1) the device functions end_effector_pose_device / frame_jacobian_device
//       (explicit nq-wide s_q buffer), at a caller-chosen thread count (argv[1])
//       so the harness can sweep thread counts for invariance; and
//   (2) the HOST batch wrappers end_effector_pose<T,false> / frame_jacobian<T>
//       over a B-timestep trajectory (the §1e per-timestep nq-stride path).
//
// end_effector_pose writes 6*NUM_EE (= 6 for the single-leaf fixtures), laid out
// [xyz; rpy] per EE. frame_jacobian writes a 6 x NUM_VEL geometric Jacobian
// (column-major, [linear; angular]) at the leaf-EE joint in LOCAL_WORLD_ALIGNED.
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

// (1a) Device-function runner for end_effector_pose. s_q is NQ(=nq)-wide,
// s_eepose is 6*NUM_EE-wide ([xyz; rpy] per EE). end_effector_pose_device
// self-allocates its working set (s_XmatsHom/s_temp/...) from the DYNAMIC
// extern shared arena, so s_end_effector_pose (a caller param) lives in STATIC
// shared memory here (it must NOT alias the device function's dynamic arena).
template <typename T>
__global__ void spherical_eepose_device_runner(
    T *d_eepose, const T *d_q, const grid::robotModel<T> *d_robot_model
) {
    __shared__ T s_eepose[6 * grid::NUM_EES];
    __shared__ T s_q[grid::NUM_JOINTS];      // NUM_JOINTS == nq for this codegen
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_JOINTS; ind += blockDim.x * blockDim.y) {
        s_q[ind] = d_q[ind];
    }
    __syncthreads();
    grid::end_effector_pose_device<T>(s_eepose, s_q, d_robot_model);
    __syncthreads();
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < 6 * grid::NUM_EES; ind += blockDim.x * blockDim.y) {
        d_eepose[ind] = s_eepose[ind];
    }
}

// (1b) Device-function runner for frame_jacobian at the leaf-EE joint in
// LOCAL_WORLD_ALIGNED (reference_frame=2). s_q is NQ-wide, s_J is 6 x NV
// (column-major, [linear; angular]). s_J is a function param (NOT in the
// auto-arena), so allocate it in static shared memory here.
template <typename T>
__global__ void spherical_frame_jacobian_device_runner(
    T *d_J, const T *d_q, const int target_jid,
    const grid::robotModel<T> *d_robot_model
) {
    __shared__ T s_J[6 * grid::NUM_VEL];
    __shared__ T s_q[grid::NUM_JOINTS];      // NUM_JOINTS == nq for this codegen
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_JOINTS; ind += blockDim.x * blockDim.y) {
        s_q[ind] = d_q[ind];
    }
    __syncthreads();
    grid::frame_jacobian_device<T>(s_J, target_jid, /*reference_frame=*/2,
                                   s_q, d_robot_model);
    __syncthreads();
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < 6 * grid::NUM_VEL; ind += blockDim.x * blockDim.y) {
        d_J[ind] = s_J[ind];
    }
}

template <typename T>
void run() {
    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();

    const int nq = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    const int nee = grid::NUM_EES;

    // ----- read inputs (q is nq-wide; the leaf target jid follows on stdin) -----
    std::vector<T> h_q(nq);
    read_vector(h_q.data(), nq);
    int target_jid = 0;
    { double v; if (!(std::cin >> v)) { std::cerr << "missing target_jid\n"; std::exit(2); } target_jid = (int)v; }
    print_vector("input_q", h_q.data(), nq);

    T *d_q;
    gpuErrchk(cudaMalloc((void **)&d_q, nq * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_q, h_q.data(), nq * sizeof(T), cudaMemcpyHostToDevice));

    // ----- (1a) end_effector_pose device-function path -----
    T *d_eepose;
    gpuErrchk(cudaMalloc((void **)&d_eepose, 6 * nee * sizeof(T)));
    std::vector<T> h_eepose(6 * nee);
    const size_t ee_smem = grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>();
    gpuErrchk(cudaFuncSetAttribute(spherical_eepose_device_runner<T>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(ee_smem)));
    spherical_eepose_device_runner<T><<<1, g_num_threads, ee_smem>>>(d_eepose, d_q, d_robot_model);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_eepose.data(), d_eepose, 6 * nee * sizeof(T), cudaMemcpyDeviceToHost));
    print_vector("end_effector_pose", h_eepose.data(), 6 * nee);

    // ----- (1b) frame_jacobian device-function path -----
    T *d_J;
    gpuErrchk(cudaMalloc((void **)&d_J, 6 * nv * sizeof(T)));
    std::vector<T> h_J(6 * nv);
    const size_t fj_smem = grid::FRAME_JACOBIAN_DYNAMIC_SHARED_MEM_BYTES<T>();
    gpuErrchk(cudaFuncSetAttribute(spherical_frame_jacobian_device_runner<T>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(fj_smem)));
    spherical_frame_jacobian_device_runner<T><<<1, g_num_threads, fj_smem>>>(d_J, d_q, target_jid, d_robot_model);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_J.data(), d_J, 6 * nv * sizeof(T), cudaMemcpyDeviceToHost));
    print_vector("frame_jacobian", h_J.data(), 6 * nv);

    // ----- (2) host batch wrappers over B IDENTICAL timesteps -----
    // The per-timestep slot is NQ-wide for q (the canonical 3*NUM_JOINTS pack).
    // We fill B identical timesteps; every output row must match the device
    // single-call result above (catches the §1e nq-stride bug).
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

    // (2a) end_effector_pose host batch wrapper.
    grid::end_effector_pose<T, false>(hd_data, d_robot_model, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k) {
        std::vector<T> row(6 * nee);
        for (int i = 0; i < 6 * nee; ++i) row[i] = hd_data->h_end_effector_pose[k * 6 * nee + i];
        print_vector("end_effector_pose_batch_" + std::to_string(k), row.data(), 6 * nee);
    }

    // (2b) frame_jacobian host batch wrapper (bakes the leaf-EE joint + LWA).
    grid::frame_jacobian<T>(hd_data, d_robot_model, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k) {
        std::vector<T> blk(6 * nv);
        for (int i = 0; i < 6 * nv; ++i) blk[i] = hd_data->h_frame_jacobian[k * 6 * nv + i];
        print_vector("frame_jacobian_batch_" + std::to_string(k), blk.data(), 6 * nv);
    }

    gpuErrchk(cudaFree(d_J));
    gpuErrchk(cudaFree(d_eepose));
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

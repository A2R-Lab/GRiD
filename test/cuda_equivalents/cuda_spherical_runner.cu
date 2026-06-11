// CUDA equivalence runner for SPHERICAL (ball) joint inverse_dynamics + crba
// (Tier-C). Spherical robots have NQ != NV (a 4-wide unit-quaternion q-block
// per ball joint, 3 v-slots), so the per-timestep INPUT slot is NQ-wide
// (= grid::NUM_JOINTS here, which the codegen defines as get_num_pos()=nq). This
// runner exercises BOTH surfaces for each ported algorithm:
//   (1) the device functions inverse_dynamics_device / crba_device (explicit
//       nq-wide s_q / nv-wide s_qd buffers), at a caller-chosen thread count
//       (argv[1]) so the harness can sweep thread counts for invariance; and
//   (2) the HOST batch wrappers inverse_dynamics<T,false,true> /
//       crba<T,false,GRID_DATA_ALL> over a B-timestep trajectory (the §1e
//       per-timestep nq-stride path the bindings use).
//
// crba writes a NUM_VEL x NUM_VEL mass matrix (column-major). The ported
// spherical algorithms are inverse_dynamics + crba + minv + forward_dynamics +
// inverse_dynamics_gradient (the last = dc_du [dc_dq | dc_dqd], 2*NV*NV, via the
// dense reduced-space inner); the other algorithms are follow-on slices.
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

// (1) Device-function runner. s_q is NQ(=nq)-wide, s_qd / s_out are NV-wide.
template <typename T>
__global__ void spherical_id_device_runner(
    T *d_out, const T *d_q, const T *d_qd,
    const grid::robotModel<T> *d_robot_model, const T gravity
) {
    __shared__ T s_q[grid::NUM_JOINTS];      // NUM_JOINTS == nq for this codegen
    __shared__ T s_qd[grid::NUM_VEL];
    __shared__ T s_zero[grid::NUM_VEL];      // qdd = 0
    __shared__ T s_out[grid::NUM_VEL];
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_JOINTS; ind += blockDim.x * blockDim.y) {
        s_q[ind] = d_q[ind];
    }
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_VEL; ind += blockDim.x * blockDim.y) {
        s_qd[ind] = d_qd[ind];
        s_zero[ind] = static_cast<T>(0);
    }
    __syncthreads();
    grid::inverse_dynamics_device<T>(s_out, s_q, s_qd, s_zero, d_robot_model,
                                     /*d_f_ext=*/nullptr, gravity);
    __syncthreads();
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_VEL; ind += blockDim.x * blockDim.y) {
        d_out[ind] = s_out[ind];
    }
}

// (1b) Device-function runner for crba. s_q is NQ(=nq)-wide, s_qd is NV-wide,
// s_M is NV x NV (column-major).
template <typename T>
__global__ void spherical_crba_device_runner(
    T *d_M, const T *d_q, const T *d_qd,
    const grid::robotModel<T> *d_robot_model, const T gravity
) {
    __shared__ T s_q[grid::NUM_JOINTS];      // NUM_JOINTS == nq for this codegen
    __shared__ T s_qd[grid::NUM_VEL];
    __shared__ T s_M[grid::NUM_VEL * grid::NUM_VEL];
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_JOINTS; ind += blockDim.x * blockDim.y) {
        s_q[ind] = d_q[ind];
    }
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_VEL; ind += blockDim.x * blockDim.y) {
        s_qd[ind] = d_qd[ind];
    }
    __syncthreads();
    grid::crba_device<T>(s_M, s_q, s_qd, d_robot_model, gravity);
    __syncthreads();
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_VEL * grid::NUM_VEL; ind += blockDim.x * blockDim.y) {
        d_M[ind] = s_M[ind];
    }
}

// (1c) Device-function runner for minv. s_q is NQ(=nq)-wide, s_Minv is NV x NV
// (column-major, SYMMETRIC_UPPER storage from the inv(CRBA) Tier-C path).
template <typename T>
__global__ void spherical_minv_device_runner(
    T *d_Minv, const T *d_q, const grid::robotModel<T> *d_robot_model
) {
    __shared__ T s_q[grid::NUM_JOINTS];      // NUM_JOINTS == nq for this codegen
    __shared__ T s_Minv[grid::NUM_VEL * grid::NUM_VEL];
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_JOINTS; ind += blockDim.x * blockDim.y) {
        s_q[ind] = d_q[ind];
    }
    __syncthreads();
    grid::minv_device<T>(s_Minv, s_q, d_robot_model);
    __syncthreads();
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_VEL * grid::NUM_VEL; ind += blockDim.x * blockDim.y) {
        d_Minv[ind] = s_Minv[ind];
    }
}

// (1d) Device-function runner for forward_dynamics. s_q is NQ-wide; s_qd / s_u /
// s_qdd are NV-wide. qdd = inv(CRBA(q)) * (u - c).
template <typename T>
__global__ void spherical_fd_device_runner(
    T *d_qdd, const T *d_q, const T *d_qd, const T *d_u,
    const grid::robotModel<T> *d_robot_model, const T gravity
) {
    __shared__ T s_q[grid::NUM_JOINTS];      // NUM_JOINTS == nq for this codegen
    __shared__ T s_qd[grid::NUM_VEL];
    __shared__ T s_u[grid::NUM_VEL];
    __shared__ T s_qdd[grid::NUM_VEL];
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_JOINTS; ind += blockDim.x * blockDim.y) {
        s_q[ind] = d_q[ind];
    }
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_VEL; ind += blockDim.x * blockDim.y) {
        s_qd[ind] = d_qd[ind];
        s_u[ind] = d_u[ind];
    }
    __syncthreads();
    grid::forward_dynamics_device<T>(s_qdd, s_q, s_qd, s_u, d_robot_model,
                                     /*d_f_ext=*/nullptr, gravity);
    __syncthreads();
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_VEL; ind += blockDim.x * blockDim.y) {
        d_qdd[ind] = s_qdd[ind];
    }
}

// (1e) Device-function runner for the STANDALONE aba. s_q is NQ-wide; s_qd /
// s_tau / s_qdd are NV-wide. Direct 3x3-D ABA recursion (NOT the Minv-compose
// forward_dynamics path), so it must independently match ref.aba AND the cuda
// forward_dynamics value.
template <typename T>
__global__ void spherical_aba_device_runner(
    T *d_qdd, const T *d_q, const T *d_qd, const T *d_tau,
    const grid::robotModel<T> *d_robot_model, const T gravity
) {
    __shared__ T s_q[grid::NUM_JOINTS];      // NUM_JOINTS == nq for this codegen
    __shared__ T s_qd[grid::NUM_VEL];
    __shared__ T s_tau[grid::NUM_VEL];
    __shared__ T s_qdd[grid::NUM_VEL];
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_JOINTS; ind += blockDim.x * blockDim.y) {
        s_q[ind] = d_q[ind];
    }
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_VEL; ind += blockDim.x * blockDim.y) {
        s_qd[ind] = d_qd[ind];
        s_tau[ind] = d_tau[ind];
    }
    __syncthreads();
    grid::aba_device<T>(s_qdd, s_q, s_qd, s_tau, d_robot_model,
                        /*d_f_ext=*/nullptr, gravity);
    __syncthreads();
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_VEL; ind += blockDim.x * blockDim.y) {
        d_qdd[ind] = s_qdd[ind];
    }
}

template <typename T>
void run() {
    const T gravity = static_cast<T>(-9.81);

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();

    // ----- read inputs (q is nq-wide, qd / u are nv-wide) -----
    std::vector<T> h_q(grid::NUM_JOINTS);   // nq
    std::vector<T> h_qd(grid::NUM_VEL);     // nv
    std::vector<T> h_u(grid::NUM_VEL);      // nv  (joint torques, for forward_dynamics)
    read_vector(h_q.data(), grid::NUM_JOINTS);
    read_vector(h_qd.data(), grid::NUM_VEL);
    read_vector(h_u.data(), grid::NUM_VEL);
    print_vector("input_q", h_q.data(), grid::NUM_JOINTS);
    print_vector("input_qd", h_qd.data(), grid::NUM_VEL);
    print_vector("input_u", h_u.data(), grid::NUM_VEL);

    // ----- (1) device-function path -----
    T *d_q;
    T *d_qd;
    T *d_u;
    T *d_out;
    gpuErrchk(cudaMalloc((void **)&d_q, grid::NUM_JOINTS * sizeof(T)));
    gpuErrchk(cudaMalloc((void **)&d_qd, grid::NUM_VEL * sizeof(T)));
    gpuErrchk(cudaMalloc((void **)&d_u, grid::NUM_VEL * sizeof(T)));
    gpuErrchk(cudaMalloc((void **)&d_out, grid::NUM_VEL * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_q, h_q.data(), grid::NUM_JOINTS * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_qd, h_qd.data(), grid::NUM_VEL * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, h_u.data(), grid::NUM_VEL * sizeof(T), cudaMemcpyHostToDevice));

    std::vector<T> h_out(grid::NUM_VEL);
    const size_t dev_smem = grid::INVERSE_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>();
    gpuErrchk(cudaFuncSetAttribute(spherical_id_device_runner<T>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(dev_smem)));
    spherical_id_device_runner<T><<<1, g_num_threads, dev_smem>>>(d_out, d_q, d_qd, d_robot_model, gravity);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_out.data(), d_out, grid::NUM_VEL * sizeof(T), cudaMemcpyDeviceToHost));
    print_vector("inverse_dynamics", h_out.data(), grid::NUM_VEL);

    // ----- (2) host batch wrapper path over B IDENTICAL timesteps -----
    // The per-timestep slot is NQ-wide for q AND for qd / u (the canonical
    // 3*NUM_JOINTS pack). We fill B identical timesteps; every output row must
    // match the device single-call result above (catches the §1e nq-stride bug).
    const int B = 4;
    grid::gridData<T> *hd_data = grid::init_gridData<T, B>();
    const int nq = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    for (int k = 0; k < B; ++k) {
        for (int i = 0; i < nq; ++i) {
            hd_data->h_q_qd_u[k * 3 * nq + i] = h_q[i];                 // q slot [0, nq)
            hd_data->h_q_qd[k * 2 * nq + i] = h_q[i];
            hd_data->h_q[k * nq + i] = h_q[i];   // nq-wide compressed-q pack (minv<T,true>)
        }
        // qd slot [nq, 2nq): first nv carry qd, the (nq-nv) tail is padding (0).
        for (int i = 0; i < nq; ++i) {
            T qd_i = (i < nv) ? h_qd[i] : static_cast<T>(0);
            hd_data->h_q_qd_u[k * 3 * nq + nq + i] = qd_i;
            hd_data->h_q_qd[k * 2 * nq + nq + i] = qd_i;
        }
        // u slot [2nq, 3nq): first nv carry torques (forward_dynamics), tail padding.
        // inverse_dynamics ignores u (qdd=0 path), so this is harmless for that cell.
        for (int i = 0; i < nq; ++i) {
            hd_data->h_q_qd_u[k * 3 * nq + 2 * nq + i] =
                (i < nv) ? h_u[i] : static_cast<T>(0);
        }
    }
    const dim3 block_dimms(1, 1, 1);
    const dim3 thread_dimms(g_num_threads, 1, 1);
    grid::inverse_dynamics<T, false, true>(
        hd_data, d_robot_model, gravity, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    // Output c is NUM_JOINTS(=nq)-wide per timestep; the first nv entries carry
    // the meaningful generalized forces. Emit every batch row's first nv.
    for (int k = 0; k < B; ++k) {
        std::vector<T> row(nv);
        for (int i = 0; i < nv; ++i) row[i] = hd_data->h_c[k * nq + i];
        print_vector("inverse_dynamics_batch_" + std::to_string(k), row.data(), nv);
    }

    // ----- (3) crba device-function path: NV x NV mass matrix M -----
    T *d_M;
    gpuErrchk(cudaMalloc((void **)&d_M, nv * nv * sizeof(T)));
    std::vector<T> h_M(nv * nv);
    // crba_device packs its OWN arena: s_XImats(72*NUM_BODIES) + an s_temp band
    // (the crba inner scratch + topology helpers), which exceeds the kernel-level
    // CRBA_DYNAMIC_SHARED_MEM_BYTES, so that helper would OOB the device path.
    // Over-allocate generously (a +512 T-buffer pad dwarfs the inner crba scratch
    // for these small fixtures); extra dynamic smem is harmless (kernel uses a
    // prefix), and over-allocation keeps the runner robot-independent.
    const size_t crba_smem = grid::grid_shared_arena_bytes<T>(
        72 * grid::NUM_BODIES + 512, grid::TOPOLOGY_HELPERS_COUNT,
        grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
    gpuErrchk(cudaFuncSetAttribute(spherical_crba_device_runner<T>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(crba_smem)));
    spherical_crba_device_runner<T><<<1, g_num_threads, crba_smem>>>(d_M, d_q, d_qd, d_robot_model, gravity);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_M.data(), d_M, nv * nv * sizeof(T), cudaMemcpyDeviceToHost));
    print_vector("crba", h_M.data(), nv * nv);

    // ----- (4) crba host batch wrapper over the same B IDENTICAL timesteps -----
    // (the h_q_qd_u / h_q_qd packs filled above are reused). Each timestep writes
    // an NV x NV block at h_M[k*nv*nv]; every block must match the device M.
    grid::crba<T, false, grid::GRID_DATA_ALL>(
        hd_data, d_robot_model, gravity, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k) {
        std::vector<T> blk(nv * nv);
        for (int i = 0; i < nv * nv; ++i) blk[i] = hd_data->h_M[k * nv * nv + i];
        print_vector("crba_batch_" + std::to_string(k), blk.data(), nv * nv);
    }

    // ----- (5) minv device-function path: NV x NV Minv = inv(CRBA(q)) -----
    // Tier-C spherical Minv routes through crba_inner + invert_matrix (the
    // mimic-style inv(M) path), since the ABA-recursion minv does not generalize
    // to a 3-DoF ball joint. The device wrapper packs IA/U/.../F + invert scratch,
    // so over-allocate (mirrors the crba_device pad above) to keep it robust.
    T *d_Minv;
    gpuErrchk(cudaMalloc((void **)&d_Minv, nv * nv * sizeof(T)));
    std::vector<T> h_Minv_dev(nv * nv);
    const size_t minv_smem = grid::grid_shared_arena_bytes<T>(
        72 * grid::NUM_BODIES + 6 * nv * nv + 512, grid::TOPOLOGY_HELPERS_COUNT,
        grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
    gpuErrchk(cudaFuncSetAttribute(spherical_minv_device_runner<T>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(minv_smem)));
    spherical_minv_device_runner<T><<<1, g_num_threads, minv_smem>>>(d_Minv, d_q, d_robot_model);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_Minv_dev.data(), d_Minv, nv * nv * sizeof(T), cudaMemcpyDeviceToHost));
    print_vector("minv", h_Minv_dev.data(), nv * nv);

    // ----- (6) minv host batch wrapper (USE_COMPRESSED_MEM: nq-wide h_q pack) ---
    grid::minv<T, true>(hd_data, d_robot_model, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k) {
        std::vector<T> blk(nv * nv);
        for (int i = 0; i < nv * nv; ++i) blk[i] = hd_data->h_Minv[k * nv * nv + i];
        print_vector("minv_batch_" + std::to_string(k), blk.data(), nv * nv);
    }

    // ----- (7) forward_dynamics device-function path: qdd (NV-wide) -----
    T *d_qdd;
    gpuErrchk(cudaMalloc((void **)&d_qdd, grid::NUM_VEL * sizeof(T)));
    std::vector<T> h_qdd_dev(nv);
    const size_t fd_smem = grid::FORWARD_DYNAMICS_DEVICE_DYNAMIC_SHARED_MEM_BYTES<T>();
    gpuErrchk(cudaFuncSetAttribute(spherical_fd_device_runner<T>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(fd_smem)));
    spherical_fd_device_runner<T><<<1, g_num_threads, fd_smem>>>(d_qdd, d_q, d_qd, d_u, d_robot_model, gravity);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_qdd_dev.data(), d_qdd, nv * sizeof(T), cudaMemcpyDeviceToHost));
    print_vector("forward_dynamics", h_qdd_dev.data(), nv);

    // ----- (8) forward_dynamics host batch wrapper over B IDENTICAL timesteps ---
    // Output qdd slot is NUM_JOINTS(=nq)-wide per timestep; first nv carry accel.
    grid::forward_dynamics<T>(hd_data, d_robot_model, gravity, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k) {
        std::vector<T> row(nv);
        for (int i = 0; i < nv; ++i) row[i] = hd_data->h_qdd[k * nq + i];
        print_vector("forward_dynamics_batch_" + std::to_string(k), row.data(), nv);
    }

    // ----- (8b) standalone aba device-function path: qdd (NV-wide) -----
    // The Tier-C spherical standalone ABA does the DIRECT 3x3-D recursion (NOT
    // the Minv-compose forward_dynamics path), so it independently validates the
    // 3x3 matrix-inverse U/D/Ia/pa/qdd machinery. The aba_device packs s_va(12*n)
    // + the inner recursion band; over-allocate generously (mirrors crba/minv
    // device pads) since there is no dedicated ABA_DEVICE_*_BYTES helper.
    T *d_qdd_aba;
    gpuErrchk(cudaMalloc((void **)&d_qdd_aba, grid::NUM_VEL * sizeof(T)));
    std::vector<T> h_qdd_aba_dev(nv);
    const size_t aba_smem = grid::grid_shared_arena_bytes<T>(
        12 * grid::NUM_BODIES + grid::ABA_DYNAMIC_SHARED_MEM_BYTES<T>() / sizeof(T) + 512,
        grid::TOPOLOGY_HELPERS_COUNT, grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>());
    gpuErrchk(cudaFuncSetAttribute(spherical_aba_device_runner<T>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(aba_smem)));
    spherical_aba_device_runner<T><<<1, g_num_threads, aba_smem>>>(d_qdd_aba, d_q, d_qd, d_u, d_robot_model, gravity);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_qdd_aba_dev.data(), d_qdd_aba, nv * sizeof(T), cudaMemcpyDeviceToHost));
    print_vector("aba", h_qdd_aba_dev.data(), nv);

    // ----- (8c) aba host batch wrapper over B IDENTICAL timesteps -----
    // Reuses the canonical 3*nq h_q_qd_u pack (q@[0,nq), qd@[nq,2nq), tau@[2nq,3nq))
    // filled above. Output qdd slot is NUM_JOINTS(=nq)-wide per timestep; first nv
    // carry accel. Every batch row must match the device single-call result (§1e).
    grid::aba<T, grid::GRID_DATA_ALL>(
        hd_data, d_robot_model, gravity, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k) {
        std::vector<T> row(nv);
        for (int i = 0; i < nv; ++i) row[i] = hd_data->h_qdd[k * nq + i];
        print_vector("aba_batch_" + std::to_string(k), row.data(), nv);
    }

    // ----- (9) inverse_dynamics_gradient: dc_du = [dc_dq | dc_dqd], 2*NV*NV -----
    // The Tier-C spherical id-gradient routes through the DENSE serial reduced-
    // space inner (a mid-chain 3-DoF ball joint owns a 3-wide v-block, which the
    // sparse single-DoF band cannot represent). dc_du = [dc_dq (NV x NV) | dc_dqd
    // (NV x NV)], column-major per half. Exercises BOTH surfaces:
    //   * the with-qdd kernel single-call (d_q_qd stride nq+nv, d_qdd nq-wide), and
    //   * the host batch wrapper inverse_dynamics_gradient<T,true,false> over B
    //     IDENTICAL timesteps (the canonical 3*nq pack + nq-wide qdd slot — the
    //     §1e nq-stride path the bindings use).
    const int idg_len = 2 * nv * nv;
    T *d_dc_du;
    gpuErrchk(cudaMalloc((void **)&d_dc_du, idg_len * sizeof(T)));
    std::vector<T> h_dc_du(idg_len);
    // Single-call pack: d_q_qd = [q(nq) | qd(nv)] (stride nq+nv); d_qdd nq-wide.
    T *d_q_qd_idg;
    T *d_qdd_idg;
    gpuErrchk(cudaMalloc((void **)&d_q_qd_idg, (nq + nv) * sizeof(T)));
    gpuErrchk(cudaMalloc((void **)&d_qdd_idg, nq * sizeof(T)));
    {
        std::vector<T> pack(nq + nv);
        for (int i = 0; i < nq; ++i) pack[i] = h_q[i];
        for (int i = 0; i < nv; ++i) pack[nq + i] = h_qd[i];
        gpuErrchk(cudaMemcpy(d_q_qd_idg, pack.data(), (nq + nv) * sizeof(T), cudaMemcpyHostToDevice));
        std::vector<T> qddp(nq, static_cast<T>(0));
        for (int i = 0; i < nv; ++i) qddp[i] = h_u[i];  // reuse h_u as a nonzero qdd
        gpuErrchk(cudaMemcpy(d_qdd_idg, qddp.data(), nq * sizeof(T), cudaMemcpyHostToDevice));
    }
    {
        const size_t idg_smem = grid::INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
        // Disambiguate the with-qdd overload (id-grad kernel has a with- and a
        // without-qdd overload, both templated <T,TIER,MUJOCO>) by casting to the
        // with-qdd pointer type before cudaFuncSetAttribute / the launch.
        using idg_qdd_kernel_t = void (*)(
            T *, unsigned char *, const T *, const int, const T *, T *,
            const grid::robotModel<T> *, const T, const int);
        idg_qdd_kernel_t idg_kernel = &grid::inverse_dynamics_gradient_kernel<T>;
        gpuErrchk(cudaFuncSetAttribute(
            idg_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(idg_smem)));
        if (grid::GRID_INVERSE_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {
            gpuErrchk(grid::grid_begin_l2_persisting(0, hd_data->d_workspace,
                grid::GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()));
        }
        grid::inverse_dynamics_gradient_kernel<T><<<1, g_num_threads, idg_smem>>>(
            d_dc_du, hd_data->d_workspace, d_q_qd_idg, nq + nv, d_qdd_idg,
            /*d_f_ext=*/nullptr, d_robot_model, gravity, 1);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        if (grid::GRID_INVERSE_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {
            gpuErrchk(grid::grid_end_l2_persisting(0));
        }
        gpuErrchk(cudaMemcpy(h_dc_du.data(), d_dc_du, idg_len * sizeof(T), cudaMemcpyDeviceToHost));
        print_vector("inverse_dynamics_gradient", h_dc_du.data(), idg_len);
    }
    // Host batch wrapper over B IDENTICAL timesteps. The 3*nq h_q_qd_u pack filled
    // above carries q@[0,nq), qd@[nq,2nq), u@[2nq,3nq); the gradient needs qdd in
    // the nq-wide h_qdd slot, so fill it (first nv = the same nonzero accel).
    for (int k = 0; k < B; ++k) {
        for (int i = 0; i < nq; ++i)
            hd_data->h_qdd[k * nq + i] = (i < nv) ? h_u[i] : static_cast<T>(0);
    }
    grid::inverse_dynamics_gradient<T, true, false>(
        hd_data, d_robot_model, gravity, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k) {
        std::vector<T> blk(idg_len);
        for (int i = 0; i < idg_len; ++i) blk[i] = hd_data->h_dc_du[k * idg_len + i];
        print_vector("inverse_dynamics_gradient_batch_" + std::to_string(k), blk.data(), idg_len);
    }

    // ----- (10) forward_dynamics_gradient: df_du = -Minv * dc_du, 2*NV*NV -----
    // Pure orchestrator: fd_gradient = -Minv(q) * id_gradient(q,qd,qdd=forward_
    // dynamics(q,qd,u)). For spherical it composes the already-spherical-aware
    // sub-inners (minv via crba, the dense id-gradient) and finishes with a
    // dimension-agnostic nv x 2nv tangent-space matmul (NO joint-id indexing,
    // NO nq dependence). Exercises BOTH surfaces:
    //   * the u-input kernel single-call (d_q_qd_u canonical 3*nq pack; the kernel
    //     computes qdd/Minv internally), and
    //   * the host batch wrapper forward_dynamics_gradient<T,false> over B
    //     IDENTICAL timesteps (the canonical 3*nq pack — the §1e nq-stride path).
    const int fdg_len = 2 * nv * nv;
    T *d_df_du;
    gpuErrchk(cudaMalloc((void **)&d_df_du, fdg_len * sizeof(T)));
    std::vector<T> h_df_du(fdg_len);
    // Single-call pack: the u-input fd-grad kernel reads s_q_qd_u as the CANONICAL
    // 3*nq pack (q@[0,nq), qd@[nq,2nq), u@[2nq,3nq); the qd/u slots are nq-wide,
    // first nv meaningful + (nq-nv) zero padding -- the §1e nq-stride layout, NOT
    // a tight nq+2*nv pack). Stride = 3*nq.
    T *d_q_qd_u_fdg;
    gpuErrchk(cudaMalloc((void **)&d_q_qd_u_fdg, (3 * nq) * sizeof(T)));
    {
        std::vector<T> pack(3 * nq, static_cast<T>(0));
        for (int i = 0; i < nq; ++i) pack[i] = h_q[i];
        for (int i = 0; i < nv; ++i) pack[nq + i] = h_qd[i];
        for (int i = 0; i < nv; ++i) pack[2 * nq + i] = h_u[i];
        gpuErrchk(cudaMemcpy(d_q_qd_u_fdg, pack.data(), (3 * nq) * sizeof(T), cudaMemcpyHostToDevice));
    }
    {
        const size_t fdg_smem = grid::FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
        // Disambiguate the u-input overload (fd-grad kernel has a u-input and a
        // qdd-Minv-input overload) by casting to the u-input pointer type before
        // cudaFuncSetAttribute / the launch.
        using fdg_u_kernel_t = void (*)(
            T *, unsigned char *, const T *, const int,
            T *, const grid::robotModel<T> *, const T, const int);  // d_f_ext is T*
        fdg_u_kernel_t fdg_kernel = &grid::forward_dynamics_gradient_kernel<T>;
        gpuErrchk(cudaFuncSetAttribute(
            fdg_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(fdg_smem)));
        if (grid::GRID_FORWARD_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {
            gpuErrchk(grid::grid_begin_l2_persisting(0, hd_data->d_workspace,
                grid::GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()));
        }
        grid::forward_dynamics_gradient_kernel<T><<<1, g_num_threads, fdg_smem>>>(
            d_df_du, hd_data->d_workspace, d_q_qd_u_fdg, 3 * nq,
            /*d_f_ext=*/nullptr, d_robot_model, gravity, 1);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        if (grid::GRID_FORWARD_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {
            gpuErrchk(grid::grid_end_l2_persisting(0));
        }
        gpuErrchk(cudaMemcpy(h_df_du.data(), d_df_du, fdg_len * sizeof(T), cudaMemcpyDeviceToHost));
        print_vector("forward_dynamics_gradient", h_df_du.data(), fdg_len);
    }
    // Host batch wrapper over B IDENTICAL timesteps. The 3*nq h_q_qd_u pack filled
    // above carries q@[0,nq), qd@[nq,2nq), u@[2nq,3nq) — exactly what the u-input
    // forward_dynamics_gradient<T,false> wrapper consumes (it computes qdd/Minv).
    grid::forward_dynamics_gradient<T, false>(
        hd_data, d_robot_model, gravity, B, block_dimms, thread_dimms, streams);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k) {
        std::vector<T> blk(fdg_len);
        for (int i = 0; i < fdg_len; ++i) blk[i] = hd_data->h_df_du[k * fdg_len + i];
        print_vector("forward_dynamics_gradient_batch_" + std::to_string(k), blk.data(), fdg_len);
    }

    gpuErrchk(cudaFree(d_df_du));
    gpuErrchk(cudaFree(d_q_qd_u_fdg));
    gpuErrchk(cudaFree(d_dc_du));
    gpuErrchk(cudaFree(d_q_qd_idg));
    gpuErrchk(cudaFree(d_qdd_idg));
    gpuErrchk(cudaFree(d_Minv));
    gpuErrchk(cudaFree(d_qdd_aba));
    gpuErrchk(cudaFree(d_qdd));
    gpuErrchk(cudaFree(d_M));
    gpuErrchk(cudaFree(d_q));
    gpuErrchk(cudaFree(d_qd));
    gpuErrchk(cudaFree(d_u));
    gpuErrchk(cudaFree(d_out));
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

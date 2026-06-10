// CUDA equivalence runner for SPHERICAL (ball) joint inverse_dynamics (Tier-C,
// first slice). Spherical robots have NQ != NV (a 4-wide unit-quaternion q-block
// per ball joint, 3 v-slots), so the per-timestep INPUT slot is NQ-wide
// (= grid::NUM_JOINTS here, which the codegen defines as get_num_pos()=nq). This
// runner exercises BOTH surfaces:
//   (1) the device function inverse_dynamics_device (explicit nq-wide s_q /
//       nv-wide s_qd / nv-wide s_out buffers), at a caller-chosen thread count
//       (argv[1]) so the harness can sweep thread counts for invariance; and
//   (2) the HOST batch wrapper inverse_dynamics<T,false,true> over a B-timestep
//       trajectory (the §1e per-timestep nq-stride path the bindings use).
//
// Only inverse_dynamics is emitted for spherical robots (the other algorithms
// are follow-on slices), so this runner deliberately calls nothing else.
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

template <typename T>
void run() {
    const T gravity = static_cast<T>(-9.81);

    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();

    // ----- read inputs (q is nq-wide, qd is nv-wide) -----
    std::vector<T> h_q(grid::NUM_JOINTS);   // nq
    std::vector<T> h_qd(grid::NUM_VEL);     // nv
    read_vector(h_q.data(), grid::NUM_JOINTS);
    read_vector(h_qd.data(), grid::NUM_VEL);
    print_vector("input_q", h_q.data(), grid::NUM_JOINTS);
    print_vector("input_qd", h_qd.data(), grid::NUM_VEL);

    // ----- (1) device-function path -----
    T *d_q;
    T *d_qd;
    T *d_out;
    gpuErrchk(cudaMalloc((void **)&d_q, grid::NUM_JOINTS * sizeof(T)));
    gpuErrchk(cudaMalloc((void **)&d_qd, grid::NUM_VEL * sizeof(T)));
    gpuErrchk(cudaMalloc((void **)&d_out, grid::NUM_VEL * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_q, h_q.data(), grid::NUM_JOINTS * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_qd, h_qd.data(), grid::NUM_VEL * sizeof(T), cudaMemcpyHostToDevice));

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
        }
        // qd slot [nq, 2nq): first nv carry qd, the (nq-nv) tail is padding (0).
        for (int i = 0; i < nq; ++i) {
            T qd_i = (i < nv) ? h_qd[i] : static_cast<T>(0);
            hd_data->h_q_qd_u[k * 3 * nq + nq + i] = qd_i;
            hd_data->h_q_qd[k * 2 * nq + nq + i] = qd_i;
        }
        // u slot [2nq, 3nq): unused by inverse_dynamics (qdd=0 path), zero it.
        for (int i = 0; i < nq; ++i) {
            hd_data->h_q_qd_u[k * 3 * nq + 2 * nq + i] = static_cast<T>(0);
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

    gpuErrchk(cudaFree(d_q));
    gpuErrchk(cudaFree(d_qd));
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

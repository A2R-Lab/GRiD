// CUDA equivalence runner for the SPHERICAL (ball) joint time INTEGRATOR.
// Mirrors cuda_spherical_runner.cu (the dynamics-value runner) but drives the
// integrator value surface: x_{k+1} = integrator(x_k, u_k, dt). Spherical robots
// have NQ != NV (a 4-wide unit-quaternion q-block per ball joint, 3 v-slots), so
// the state x = [q(nq); qd(nv)] is (nq+nv)-wide and the per-timestep INPUT slot is
// NQ-wide (= grid::NUM_JOINTS == get_num_pos() = nq, the canonical 3*NUM_JOINTS
// pack). The q-side of the step is the SO(3) quaternion retract per ball joint +
// a plain vector add for the downstream-shifted revolute slots (the §1e case);
// the qd-side is the unchanged Euler vector add qd + dt*qdd.
//
// It exercises BOTH surfaces for ALL FIVE IntegratorTypes:
//   (1) the device function integrator_device<T, IT> (explicit nq-wide s_q /
//       nv-wide s_qd / s_u buffers), at a caller-chosen thread count (argv[1])
//       so the harness can sweep thread counts for invariance; and
//   (2) the HOST batch wrapper integrator<T, IT, GRID_DATA_ALL> over a
//       B-timestep trajectory (the §1e per-timestep nq-stride path the bindings
//       use). Each batch row's (nq+nv) state must equal the single-call device
//       state (catches the §1e nq-stride bug).
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "grid.cuh"

int g_num_threads = 32;

// streams handle shared by the host-batch calls (init once in run()).
cudaStream_t *streams_global = nullptr;

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
    std::cout << std::setprecision(12);
    for (int i = 0; i < count; ++i) {
        if (i) std::cout << " ";
        std::cout << static_cast<double>(data[i]);
    }
    std::cout << "\nEND " << name << "\n";
}

// Device-function runner for integrator_device<T, IT>. s_q is NQ-wide; s_qd /
// s_u are NV-wide; s_x_kp1 is (NQ+NV)-wide (q part NQ, qd part NV).
template <typename T, grid::IntegratorType IT>
__global__ void spherical_integrator_device_runner(
    T *d_x_kp1, const T *d_q, const T *d_qd, const T *d_u,
    const grid::robotModel<T> *d_robot_model, const T gravity, const T dt
) {
    __shared__ T s_q[grid::NUM_JOINTS];                  // NUM_JOINTS == nq
    __shared__ T s_qd[grid::NUM_VEL];
    __shared__ T s_u[grid::NUM_VEL];
    __shared__ T s_x_kp1[grid::NUM_POS + grid::NUM_VEL];
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
    grid::integrator_device<T, IT>(s_x_kp1, s_q, s_qd, s_u, d_robot_model, gravity, dt);
    __syncthreads();
    for (int ind = threadIdx.x + threadIdx.y * blockDim.x;
         ind < grid::NUM_POS + grid::NUM_VEL; ind += blockDim.x * blockDim.y) {
        d_x_kp1[ind] = s_x_kp1[ind];
    }
}

// Run a single IntegratorType: device single-call + host batch over B timesteps.
template <typename T, grid::IntegratorType IT>
void run_one(const std::string &tag, const std::vector<T> &h_q,
             const std::vector<T> &h_qd, const std::vector<T> &h_u, const T dt,
             const grid::robotModel<T> *d_robot_model,
             grid::gridData<T> *hd_data, int B, const T gravity) {
    const int nq = grid::NUM_JOINTS;
    const int nv = grid::NUM_VEL;
    const int nx = nq + nv;

    // ----- (1) device-function path -----
    T *d_q, *d_qd, *d_u, *d_x_kp1;
    gpuErrchk(cudaMalloc((void **)&d_q, nq * sizeof(T)));
    gpuErrchk(cudaMalloc((void **)&d_qd, nv * sizeof(T)));
    gpuErrchk(cudaMalloc((void **)&d_u, nv * sizeof(T)));
    gpuErrchk(cudaMalloc((void **)&d_x_kp1, nx * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_q, h_q.data(), nq * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_qd, h_qd.data(), nv * sizeof(T), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_u, h_u.data(), nv * sizeof(T), cudaMemcpyHostToDevice));

    std::vector<T> h_x(nx);
    const size_t dev_smem = grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>();
    gpuErrchk(cudaFuncSetAttribute(spherical_integrator_device_runner<T, IT>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(dev_smem)));
    spherical_integrator_device_runner<T, IT><<<1, g_num_threads, dev_smem>>>(
        d_x_kp1, d_q, d_qd, d_u, d_robot_model, gravity, dt);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_x.data(), d_x_kp1, nx * sizeof(T), cudaMemcpyDeviceToHost));
    print_vector(tag + "_x_kp1", h_x.data(), nx);

    // ----- (2) host batch wrapper over B IDENTICAL timesteps (§1e nq-stride) ---
    for (int k = 0; k < B; ++k) {
        for (int i = 0; i < nq; ++i)
            hd_data->h_q_qd_u[k * 3 * nq + i] = h_q[i];                 // q slot [0, nq)
        for (int i = 0; i < nq; ++i)                                    // qd slot [nq, 2nq)
            hd_data->h_q_qd_u[k * 3 * nq + nq + i] =
                (i < nv) ? h_qd[i] : static_cast<T>(0);
        for (int i = 0; i < nq; ++i)                                    // u slot [2nq, 3nq)
            hd_data->h_q_qd_u[k * 3 * nq + 2 * nq + i] =
                (i < nv) ? h_u[i] : static_cast<T>(0);
    }
    const dim3 block_dimms(1, 1, 1);
    const dim3 thread_dimms(g_num_threads, 1, 1);
    grid::integrator<T, IT, grid::GRID_DATA_ALL>(
        hd_data, d_robot_model, gravity, dt, B, block_dimms, thread_dimms, streams_global);
    gpuErrchk(cudaPeekAtLastError());
    for (int k = 0; k < B; ++k) {
        std::vector<T> row(nx);
        for (int i = 0; i < nx; ++i) row[i] = hd_data->h_x_kp1[k * nx + i];
        print_vector(tag + "_x_kp1_batch_" + std::to_string(k), row.data(), nx);
    }

    gpuErrchk(cudaFree(d_q));
    gpuErrchk(cudaFree(d_qd));
    gpuErrchk(cudaFree(d_u));
    gpuErrchk(cudaFree(d_x_kp1));
}

template <typename T>
void run() {
    const T gravity = static_cast<T>(-9.81);
    const T dt = static_cast<T>(0.01);

    streams_global = grid::init_grid<T>();
    grid::robotModel<T> *d_robot_model = grid::init_robotModel<T>();

    std::vector<T> h_q(grid::NUM_JOINTS);   // nq (unit-quaternion in ball block)
    std::vector<T> h_qd(grid::NUM_VEL);     // nv
    std::vector<T> h_u(grid::NUM_VEL);      // nv (control torque)
    read_vector(h_q.data(), grid::NUM_JOINTS);
    read_vector(h_qd.data(), grid::NUM_VEL);
    read_vector(h_u.data(), grid::NUM_VEL);
    print_vector("input_q", h_q.data(), grid::NUM_JOINTS);
    print_vector("input_qd", h_qd.data(), grid::NUM_VEL);
    print_vector("input_u", h_u.data(), grid::NUM_VEL);

    const int B = 4;
    grid::gridData<T> *hd_data = grid::init_gridData<T, B>();

    run_one<T, grid::IntegratorType::EULER>("integrator_euler", h_q, h_qd, h_u, dt, d_robot_model, hd_data, B, gravity);
    run_one<T, grid::IntegratorType::SEMI_IMPLICIT_EULER>("integrator_si_euler", h_q, h_qd, h_u, dt, d_robot_model, hd_data, B, gravity);
    run_one<T, grid::IntegratorType::MIDPOINT>("integrator_midpoint", h_q, h_qd, h_u, dt, d_robot_model, hd_data, B, gravity);
    run_one<T, grid::IntegratorType::RK3>("integrator_rk3", h_q, h_qd, h_u, dt, d_robot_model, hd_data, B, gravity);
    run_one<T, grid::IntegratorType::RK4>("integrator_rk4", h_q, h_qd, h_u, dt, d_robot_model, hd_data, B, gravity);

    grid::close_grid<T>(streams_global, d_robot_model, hd_data);
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

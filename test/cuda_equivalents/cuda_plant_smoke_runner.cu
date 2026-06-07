// CUDA smoke runner for the generated grid_plant cost/constraint/step kernels.
//
// Input on stdin (whitespace-separated floats), matching _sample_to_stdin:
//   q (NUM_POS) qd (NUM_VEL) u (NUM_VEL) dt
//
// The runner builds DETERMINISTIC desired vectors / weights / bounds / mu as
// fixed functions of the DOF index (mirrored exactly in the Python test), then
// drives the grid_plant primitives from a single-block kernel and prints the
// results in BEGIN/END framed blocks (column-major).
//
// Emitted blocks:
//   state_cost_value             1 x 1
//   state_cost_grad              1 x NX        (NX = NUM_POS + NUM_VEL)
//   state_cost_hess              NX x NX
//   input_cost_value             1 x 1
//   input_cost_grad              1 x NU        (NU = NUM_VEL)
//   input_cost_hess              NU x NU
//   ee_cost_value                1 x 1
//   ee_cost_grad                 1 x NX        (qd-block must be zero)
//   ee_cost_hess                 NX x NX       (only top-left NV x NV non-zero)
//   pos_barrier_value            1 x 1
//   pos_barrier_grad             1 x NX        (q-block)
//   pos_barrier_hess_diag        1 x NUM_POS
//   vel_barrier_value            1 x 1
//   vel_barrier_grad             1 x NX        (qd-block)
//   ctrl_barrier_value           1 x 1
//   ctrl_barrier_grad            1 x NU
//   plant_dAB                    (2*NV) x (3*NV)   (plant_step_gradient output)
//   integrator_dAB              (2*NV) x (3*NV)   (grid::integrator_gradient — pass-through oracle)
//   plant_x_kp1                  1 x (NUM_POS + NUM_VEL)
//   integrator_x_kp1            1 x (NUM_POS + NUM_VEL)
//   ee_pos                       1 x 3             (the EE position p(q), for the Python FD oracle)
// The centroidal blocks below are emitted ONLY when GRID_PLANT_HAS_COM_COST &&
// GRID_PLANT_HAS_MOMENTUM_COST are defined (com/ccrba present); otherwise a single
// `com_cost_skipped` sentinel is emitted in their place:
//   com_cost_value               1 x 1             (grid_plant::com_cost)
//   com_cost_grad                1 x NX            (q-block = J_com^T W r; qd-block zero)
//   com_cost_hess                NX x NX           (top-left NV x NV q-block = J_com^T W J_com)
//   momentum_cost_value          1 x 1             (grid_plant::momentum_cost)
//   momentum_cost_grad           1 x NX            (qd-block = A^T W r; q-block zero)
//   momentum_cost_hess           NX x NX           (bottom-right NV x NV qd-block = A^T W A)
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "grid.cuh"

#ifndef PLANT_EE
#define PLANT_EE 0
#endif

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
template <typename T>
void print_vector(const std::string &name, const T *data, int count) {
    print_matrix_col_major(name, data, 1, count);
}

constexpr int NQ = grid::NUM_POS;
constexpr int NV = grid::NUM_VEL;
constexpr int NX = NQ + NV;
constexpr int NU = NV;

// ---- deterministic problem setup (must match the Python test exactly) ----
// Use a large finite infinity sentinel detected by isfinite()? No — use real
// HUGE_VALF so isfinite() returns false and the barrier skips that side. DOF 0's
// position barrier is made fully unbounded to exercise the isfinite-skip path.
template <typename T> __host__ __device__ T x_des_val(int i)  { return static_cast<T>(0.1) * i; }
template <typename T> __host__ __device__ T Qw_val(int i)     { return static_cast<T>(1.0) + static_cast<T>(0.5) * i; }
template <typename T> __host__ __device__ T u_des_val(int i)  { return static_cast<T>(-0.05) * i; }
template <typename T> __host__ __device__ T Rw_val(int i)     { return static_cast<T>(2.0) + static_cast<T>(0.1) * i; }
template <typename T> __host__ __device__ T Ww_val(int r)     { return static_cast<T>(10.0) + r; }
// Centroidal CoM-cost setup (3 axes) and momentum-cost setup (6 components).
template <typename T> __host__ __device__ T com_pdes_val(int r) { return static_cast<T>(0.2) + static_cast<T>(0.1) * r; }
template <typename T> __host__ __device__ T com_W_val(int r)    { return static_cast<T>(3.0) + static_cast<T>(0.5) * r; }
template <typename T> __host__ __device__ T mom_hdes_val(int r) { return static_cast<T>(-0.3) + static_cast<T>(0.15) * r; }
template <typename T> __host__ __device__ T mom_W_val(int r)    { return static_cast<T>(2.0) + static_cast<T>(0.25) * r; }

// A single-block kernel: fill the deterministic setup, then call every primitive.
template <typename T>
__global__ void plant_kernel(const T *g_q, const T *g_qd, const T *g_u, T dt,
                             const grid::robotModel<T> *d_robotModel, T gravity,
                             // outputs (global)
                             T *o_state_val, T *o_state_grad, T *o_state_hess,
                             T *o_input_val, T *o_input_grad, T *o_input_hess,
                             T *o_ee_val, T *o_ee_grad, T *o_ee_hess,
                             T *o_posb_val, T *o_posb_grad, T *o_posb_hess_diag,
                             T *o_velb_val, T *o_velb_grad,
                             T *o_ctrlb_val, T *o_ctrlb_grad,
                             T *o_plant_dAB, T *o_int_dAB,
                             T *o_plant_xkp1, T *o_int_xkp1, T *o_eepos) {
    __shared__ T s_x[NX], s_u[NU], s_xdes[NX], s_Q[NX], s_udes[NU], s_R[NU];
    __shared__ T s_pdes[3], s_W[3];
    __shared__ T s_scratch[NX];
    __shared__ T s_out[1];
    __shared__ T s_grad[NX], s_hess[NX * NX];
    __shared__ T s_eePos[6 * grid::NUM_EES], s_deePos[6 * NV * grid::NUM_EES];
    // barrier bounds (interior: [val-1, val+1]); DOF 0 position barrier unbounded.
    __shared__ T s_lo_q[NQ], s_hi_q[NQ], s_lo_v[NV], s_hi_v[NV], s_lo_u[NU], s_hi_u[NU];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NX; i += nth) {
        s_x[i] = (i < NQ) ? g_q[i] : g_qd[i - NQ];
        s_xdes[i] = x_des_val<T>(i);
        s_Q[i] = Qw_val<T>(i);
    }
    for (int i = tid; i < NU; i += nth) {
        s_u[i] = g_u[i];
        s_udes[i] = u_des_val<T>(i);
        s_R[i] = Rw_val<T>(i);
    }
    for (int r = tid; r < 3; r += nth) { s_pdes[r] = static_cast<T>(0); s_W[r] = Ww_val<T>(r); }
    for (int i = tid; i < NQ; i += nth) {
        s_lo_q[i] = g_q[i] - static_cast<T>(1);
        s_hi_q[i] = g_q[i] + static_cast<T>(1);
        if (i == 0) { s_lo_q[i] = -HUGE_VALF; s_hi_q[i] = HUGE_VALF; }  // unbounded DOF: isfinite-skip
    }
    for (int i = tid; i < NV; i += nth) {
        s_lo_v[i] = g_qd[i] - static_cast<T>(1);
        s_hi_v[i] = g_qd[i] + static_cast<T>(1);
    }
    for (int i = tid; i < NU; i += nth) {
        s_lo_u[i] = g_u[i] - static_cast<T>(1);
        s_hi_u[i] = g_u[i] + static_cast<T>(1);
    }
    __syncthreads();

    // ---- quadratic state cost ----
    grid_plant::quadratic_state_cost<T>(s_out, s_x, s_xdes, s_Q, s_scratch);
    __syncthreads(); if (tid == 0) o_state_val[0] = s_out[0]; __syncthreads();
    grid_plant::quadratic_state_cost_gradient<T, false>(s_grad, s_x, s_xdes, s_Q);
    grid_plant::quadratic_state_cost_hessian<T, false>(s_hess, s_Q);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_state_grad[i] = s_grad[i];
    for (int i = tid; i < NX * NX; i += nth) o_state_hess[i] = s_hess[i];
    __syncthreads();

    // ---- quadratic input cost ----
    grid_plant::quadratic_input_cost<T>(s_out, s_u, s_udes, s_R, s_scratch);
    __syncthreads(); if (tid == 0) o_input_val[0] = s_out[0]; __syncthreads();
    grid_plant::quadratic_input_cost_gradient<T, false>(s_grad, s_u, s_udes, s_R);
    grid_plant::quadratic_input_cost_hessian<T, false>(s_hess, s_R);
    __syncthreads();
    for (int i = tid; i < NU; i += nth) o_input_grad[i] = s_grad[i];
    for (int i = tid; i < NU * NU; i += nth) o_input_hess[i] = s_hess[i];
    __syncthreads();

    // ---- ee position cost ---- (s_x[:NQ] is q)
    grid_plant::ee_pos_cost<T, PLANT_EE>(s_out, s_x, s_pdes, s_W, s_eePos, d_robotModel);
    __syncthreads(); if (tid == 0) { o_ee_val[0] = s_out[0]; for (int r = 0; r < 3; ++r) o_eepos[r] = s_eePos[6 * PLANT_EE + r]; } __syncthreads();
    grid_plant::ee_pos_cost_gradient<T, PLANT_EE, false>(s_grad, s_x, s_pdes, s_W, s_eePos, s_deePos, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_ee_grad[i] = s_grad[i];
    __syncthreads();
    grid_plant::ee_pos_cost_hessian<T, PLANT_EE, false>(s_hess, s_x, s_W, s_deePos, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX * NX; i += nth) o_ee_hess[i] = s_hess[i];
    __syncthreads();

    // ---- joint position barrier (q block of x) ----
    if (tid == 0) s_out[0] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_position_barrier<T>(s_out, s_x, s_lo_q, s_hi_q, static_cast<T>(0.1), s_scratch);
    __syncthreads(); if (tid == 0) o_posb_val[0] = s_out[0]; __syncthreads();
    for (int i = tid; i < NX; i += nth) s_grad[i] = static_cast<T>(0);
    for (int i = tid; i < NX * NX; i += nth) s_hess[i] = static_cast<T>(0);
    __syncthreads();
    grid_plant::joint_position_barrier_gradient<T, 0, 0>(s_grad, s_x, s_lo_q, s_hi_q, static_cast<T>(0.1));
    grid_plant::joint_position_barrier_hessian<T, NX, 0, 0>(s_hess, s_x, s_lo_q, s_hi_q, static_cast<T>(0.1));
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_posb_grad[i] = s_grad[i];
    for (int i = tid; i < NQ; i += nth) o_posb_hess_diag[i] = s_hess[i * NX + i];
    __syncthreads();

    // ---- joint velocity barrier (qd block of x; VAR_OFFSET=NQ, GRAD_OFFSET=NQ) ----
    if (tid == 0) s_out[0] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_velocity_barrier<T>(s_out, s_x, s_lo_v, s_hi_v, static_cast<T>(0.1), s_scratch);
    __syncthreads(); if (tid == 0) o_velb_val[0] = s_out[0]; __syncthreads();
    for (int i = tid; i < NX; i += nth) s_grad[i] = static_cast<T>(0);
    __syncthreads();
    grid_plant::joint_velocity_barrier_gradient<T, NQ, NQ>(s_grad, s_x, s_lo_v, s_hi_v, static_cast<T>(0.1));
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_velb_grad[i] = s_grad[i];
    __syncthreads();

    // ---- joint torque barrier (standalone u; VAR_OFFSET=0, GRAD_OFFSET=0) ----
    if (tid == 0) s_out[0] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_torque_barrier<T>(s_out, s_u, s_lo_u, s_hi_u, static_cast<T>(0.1), s_scratch);
    __syncthreads(); if (tid == 0) o_ctrlb_val[0] = s_out[0]; __syncthreads();
    for (int i = tid; i < NU; i += nth) s_grad[i] = static_cast<T>(0);
    __syncthreads();
    grid_plant::joint_torque_barrier_gradient<T, 0, 0>(s_grad, s_u, s_lo_u, s_hi_u, static_cast<T>(0.1));
    __syncthreads();
    for (int i = tid; i < NU; i += nth) o_ctrlb_grad[i] = s_grad[i];
    __syncthreads();
}

// ---- pass-through check: plant_step / plant_step_gradient vs grid:: integrator ----
// Done in a separate kernel that needs the integrator-gradient scratch buffers.
template <typename T>
__global__ void plant_step_kernel(const T *g_q, const T *g_qd, const T *g_u, T dt,
                                  const grid::robotModel<T> *d_robotModel, T gravity,
                                  T *o_plant_xkp1, T *o_plant_dAB) {
    __shared__ T s_x[NX], s_u[NU], s_xkp1[NX], s_dAB[2 * NV * 3 * NV];
    // integrator-gradient scratch (caller-placed; mirrors the integrator-gradient kernel)
    __shared__ T s_df_du[NV * 2 * NV], s_dc_du[NV * 2 * NV], s_vaf[18 * NV], s_Minv[NV * NV], s_qdd[NV];
    __shared__ T s_q_orig[NQ], s_qd_orig[NV], s_stage_grad_qdd[4 * NV], s_D_qdd_stage[4 * NV * 3 * NV];
    __shared__ T s_dInt_q_6x6[36], s_dInt_v_6x6[36];
    // The shared XImats table + the FD-grad inner pool. s_XImats is loaded INTO
    // by the integrator-gradient device's internal load_update_XImats; s_temp is
    // generously sized (>= the 66*NUM_JOINTS+... inner full temp; 1722 on iiwa14).
    __shared__ T s_XImats[grid::DYNAMICS_XI_T_COUNT];
    __shared__ T s_temp[4096];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NX; i += nth) s_x[i] = (i < NQ) ? g_q[i] : g_qd[i - NQ];
    for (int i = tid; i < NU; i += nth) s_u[i] = g_u[i];
    __syncthreads();

    grid_plant::plant_step<T, grid::IntegratorType::EULER>(s_xkp1, s_x, s_u, d_robotModel, gravity, dt);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_plant_xkp1[i] = s_xkp1[i];
    __syncthreads();

    grid_plant::plant_step_gradient<T, grid::IntegratorType::EULER, true, false>(
        s_dAB, s_x, s_u, s_df_du, s_dc_du, s_vaf, s_Minv, s_qdd,
        s_q_orig, s_qd_orig, s_stage_grad_qdd, s_D_qdd_stage,
        s_dInt_q_6x6, s_dInt_v_6x6, s_XImats, /*s_topology_helpers*/ nullptr,
        s_temp, /*d_workspace*/ nullptr, /*d_temp_spill*/ nullptr,
        d_robotModel, gravity, dt);
    __syncthreads();
    for (int i = tid; i < 2 * NV * 3 * NV; i += nth) o_plant_dAB[i] = s_dAB[i];
    __syncthreads();
}

// ---- plant_step_hessian (s_d2AB) ----
// Drives the GENERATED grid_plant::plant_step_hessian_kernel (the true 2nd-order
// integrator sensitivity, composing grid::integrator_hessian_device ->
// fdsva_so_device) with a DYNAMIC-smem arena + the tier-spill d_workspace, so big
// fixed-base robots (g1/h1_2) whose SHARED-tier arena overflows the smem cap fall
// back to the spilled tier (d2AB output + fdsva tensors + pool -> d_workspace).
// The kernel handles all staging/scatter itself; this runner only supplies the
// dynamic smem byte count + the workspace allocation (mirrors the binding launch).
// Fixed-base, EULER / SI-EULER. Output is row-major (2*NV x 3*NV x 3*NV).

// ---- centroidal plant costs: com_cost / momentum_cost ----
// These compose grid::com_device / grid::ccrba_device, which use an `extern
// __shared__` dynamic arena (COM/CCRBA_DYNAMIC_SHARED_MEM_BYTES). The launch
// must size dynamic smem to max(COM, CCRBA) and raise the opt-in attribute.
// Emitted ONLY when grid::com_device + grid::ccrba_device are present (the
// GRID_PLANT_HAS_COM_COST / GRID_PLANT_HAS_MOMENTUM_COST macros, emitted by
// _plant.py). For a robot/config that lacks them (e.g. a mimic robot whose ccrba
// is gated off), the whole centroidal block (kernel + alloc + launch + print) is
// #if-compiled out and a parseable `*_skipped` sentinel is emitted instead.
// Validated on iiwa14:fixed / go2:floating (both non-mimic, macros present).
#if defined(GRID_PLANT_HAS_COM_COST) && defined(GRID_PLANT_HAS_MOMENTUM_COST)
template <typename T>
__global__ void plant_centroidal_kernel(const T *g_q, const T *g_qd,
                                        const grid::robotModel<T> *d_robotModel,
                                        T *o_com_val, T *o_com_grad, T *o_com_hess,
                                        T *o_mom_val, T *o_mom_grad, T *o_mom_hess) {
    __shared__ T s_q[NQ], s_qd[NV];
    __shared__ T s_com[3 + 3 * NV];        // grid::com_device output [p_com(3); J_com(3 x NV)]
    __shared__ T s_ccrba[6 * NV + 6];      // grid::ccrba_device output [A(6 x NV); h(6)]
    __shared__ T s_pdes[3], s_cW[3];       // CoM desired + per-axis weight
    __shared__ T s_hdes[6], s_mW[6];       // momentum desired + per-component weight
    __shared__ T s_out[1];
    __shared__ T s_grad[NX], s_hess[NX * NX];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    for (int i = tid; i < NV; i += nth) s_qd[i] = g_qd[i];
    for (int r = tid; r < 3; r += nth) { s_pdes[r] = com_pdes_val<T>(r); s_cW[r] = com_W_val<T>(r); }
    for (int r = tid; r < 6; r += nth) { s_hdes[r] = mom_hdes_val<T>(r); s_mW[r] = mom_W_val<T>(r); }
    __syncthreads();

    // ---- CoM-tracking cost (value + grad over x=[q;qd] + GN hess) ----
    grid_plant::com_cost<T>(s_out, s_q, s_pdes, s_cW, s_com, d_robotModel);
    __syncthreads(); if (tid == 0) o_com_val[0] = s_out[0]; __syncthreads();
    grid_plant::com_cost_gradient<T, false>(s_grad, s_q, s_pdes, s_cW, s_com, d_robotModel);
    grid_plant::com_cost_hessian<T, false>(s_hess, s_q, s_cW, s_com, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_com_grad[i] = s_grad[i];
    for (int i = tid; i < NX * NX; i += nth) o_com_hess[i] = s_hess[i];
    __syncthreads();

    // ---- centroidal-momentum-tracking cost (value + grad + GN hess) ----
    grid_plant::momentum_cost<T>(s_out, s_q, s_qd, s_hdes, s_mW, s_ccrba, d_robotModel);
    __syncthreads(); if (tid == 0) o_mom_val[0] = s_out[0]; __syncthreads();
    grid_plant::momentum_cost_gradient<T, false>(s_grad, s_q, s_qd, s_hdes, s_mW, s_ccrba, d_robotModel);
    grid_plant::momentum_cost_hessian<T, false>(s_hess, s_q, s_qd, s_mW, s_ccrba, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_mom_grad[i] = s_grad[i];
    for (int i = tid; i < NX * NX; i += nth) o_mom_hess[i] = s_hess[i];
    __syncthreads();
}
#endif  // GRID_PLANT_HAS_COM_COST && GRID_PLANT_HAS_MOMENTUM_COST

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
    const T gravity = static_cast<T>(-9.81);
    cudaStream_t *streams = grid::init_grid<T>();
    grid::robotModel<T> *d_robotModel = grid::init_robotModel<T>();
    grid::gridData<T> *hd_data = grid::init_gridData<T, 1>();

    std::vector<T> h_q(NQ), h_qd(NV), h_u(NU);
    read_vector(h_q.data(), NQ);
    read_vector(h_qd.data(), NV);
    read_vector(h_u.data(), NU);
    double dt_d; if (!(std::cin >> dt_d)) { std::cerr << "dt fail\n"; std::exit(2); }
    const T dt = static_cast<T>(dt_d);

    print_vector("input_q", h_q.data(), NQ);
    print_vector("input_qd", h_qd.data(), NV);
    print_vector("input_u", h_u.data(), NU);

    T *g_q = dmalloc<T>(NQ), *g_qd = dmalloc<T>(NV), *g_u = dmalloc<T>(NU);
    cudaMemcpy(g_q, h_q.data(), NQ * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_qd, h_qd.data(), NV * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(g_u, h_u.data(), NU * sizeof(T), cudaMemcpyHostToDevice);

    T *o_sv = dmalloc<T>(1), *o_sg = dmalloc<T>(NX), *o_sh = dmalloc<T>(NX * NX);
    T *o_iv = dmalloc<T>(1), *o_ig = dmalloc<T>(NU), *o_ih = dmalloc<T>(NU * NU);
    T *o_ev = dmalloc<T>(1), *o_eg = dmalloc<T>(NX), *o_eh = dmalloc<T>(NX * NX);
    T *o_pbv = dmalloc<T>(1), *o_pbg = dmalloc<T>(NX), *o_pbh = dmalloc<T>(NQ);
    T *o_vbv = dmalloc<T>(1), *o_vbg = dmalloc<T>(NX);
    T *o_cbv = dmalloc<T>(1), *o_cbg = dmalloc<T>(NU);
    T *o_pdab = dmalloc<T>(2 * NV * 3 * NV), *o_idab = dmalloc<T>(2 * NV * 3 * NV);
    T *o_pxk = dmalloc<T>(NX), *o_ixk = dmalloc<T>(NX), *o_eepos = dmalloc<T>(3);
#if defined(GRID_PLANT_HAS_COM_COST) && defined(GRID_PLANT_HAS_MOMENTUM_COST)
    T *o_comv = dmalloc<T>(1), *o_comg = dmalloc<T>(NX), *o_comh = dmalloc<T>(NX * NX);
    T *o_momv = dmalloc<T>(1), *o_momg = dmalloc<T>(NX), *o_momh = dmalloc<T>(NX * NX);
#endif
#ifdef GRID_PLANT_HAS_STEP_HESSIAN
    const int D2AB_CNT = 2 * NV * (3 * NV) * (3 * NV);
    T *o_h2_eu = dmalloc<T>(D2AB_CNT), *o_h2_si = dmalloc<T>(D2AB_CNT);
#endif

    // Thread count is env-overridable (GRID_CUDA_PLANT_THREADS) so robots whose
    // plant_kernel exceeds this GPU's per-block register budget at the default
    // MAX_PERF_LEVEL_THREADS (e.g. fr3) can still be validated at a lower count.
    const char *_nt_s = std::getenv("GRID_CUDA_PLANT_THREADS");
    const int _nt_env = _nt_s ? std::atoi(_nt_s) : 0;
    const int nthreads = (_nt_env > 0) ? _nt_env : grid::MAX_PERF_LEVEL_THREADS;
    // The grid_plant primitives that compose an auto-allocating grid:: _device
    // wrapper (ee_pos_cost -> end_effector_pose[_gradient]_device;
    // plant_step -> integrator_device) need that device's dynamic-shared arena
    // allocated at launch (the wrappers use `extern __shared__`). Size each
    // kernel's dynamic smem to the max device requirement it composes and raise
    // the opt-in attribute so big arenas are allowed.
    size_t plant_dyn = grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>();
    if (grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>() > plant_dyn) plant_dyn = grid::END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>();
    size_t step_dyn = grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(plant_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)plant_dyn);
    cudaFuncSetAttribute(plant_step_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)step_dyn);
#if defined(GRID_PLANT_HAS_COM_COST) && defined(GRID_PLANT_HAS_MOMENTUM_COST)
    // com_cost composes grid::com_device, momentum_cost composes grid::ccrba_device;
    // both use an extern __shared__ dynamic arena, so size to the max of the two.
    size_t cent_dyn = grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>();
    if (grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>() > cent_dyn) cent_dyn = grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES<T>();
    cudaFuncSetAttribute(plant_centroidal_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cent_dyn);
#endif

    plant_kernel<T><<<1, nthreads, plant_dyn>>>(g_q, g_qd, g_u, dt, d_robotModel, gravity,
        o_sv, o_sg, o_sh, o_iv, o_ig, o_ih, o_ev, o_eg, o_eh,
        o_pbv, o_pbg, o_pbh, o_vbv, o_vbg, o_cbv, o_cbg,
        o_pdab, o_idab, o_pxk, o_ixk, o_eepos);
    // Fail loudly on a bad launch: an unchecked launch failure leaves the (zeroed)
    // outputs untouched and masquerades as a real (wrong) result. gpuErrchkKernel()
    // (from grid.cuh) does cudaPeekAtLastError() + cudaDeviceSynchronize() + abort.
    gpuErrchkKernel();

    // The centroidal cost kernel (com_cost / momentum_cost) is independent of the
    // plant_step / integrator pass-through path, so drive it first — that way a
    // robot whose plant_step_kernel static scratch overflows the device smem cap
    // (e.g. go2:floating, where the big integrator-gradient static buffers exceed
    // the 48 KB default) still produces valid centroidal output. Compiled in only
    // when the centroidal macros are present; otherwise a `*_skipped` sentinel is
    // emitted below so the Python parser detects the absence cleanly.
#if defined(GRID_PLANT_HAS_COM_COST) && defined(GRID_PLANT_HAS_MOMENTUM_COST)
    plant_centroidal_kernel<T><<<1, nthreads, cent_dyn>>>(g_q, g_qd, d_robotModel,
        o_comv, o_comg, o_comh, o_momv, o_momg, o_momh);
    gpuErrchkKernel();
#endif

#ifdef GRID_PLANT_HAS_STEP_HESSIAN
    // plant_step_hessian (s_d2AB), EULER + SI-EULER, via the generated tier-aware
    // kernel. Pack x=[q;qd] contiguous (the kernel reads d_x[k*stride_x+ind]); u is
    // already contiguous. Size dynamic smem to the (tier-selected) macro + raise the
    // opt-in attribute; allocate the per-block spill workspace when any tier spills.
    {
        T *g_x = dmalloc<T>(NX);
        std::vector<T> h_x(NX);
        for (int i = 0; i < NQ; ++i) h_x[i] = h_q[i];
        for (int i = 0; i < NV; ++i) h_x[NQ + i] = h_qd[i];
        cudaMemcpy(g_x, h_x.data(), NX * sizeof(T), cudaMemcpyHostToDevice);

        const size_t h2_smem = grid_plant::INTEGRATOR_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>();
        unsigned char *g_h2_ws = nullptr;
        if (grid_plant::GRID_PLANT_HESSIAN_USES_WORKSPACE_ANY_TIER) {
            cudaMalloc(&g_h2_ws, grid_plant::PLANT_HESSIAN_WORKSPACE_BYTES_PER_TIMESTEP<T>());
        }
        cudaFuncSetAttribute(grid_plant::plant_step_hessian_kernel<T, grid::IntegratorType::EULER>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)h2_smem);
        cudaFuncSetAttribute(grid_plant::plant_step_hessian_kernel<T, grid::IntegratorType::SEMI_IMPLICIT_EULER>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)h2_smem);
        grid_plant::plant_step_hessian_kernel<T, grid::IntegratorType::EULER><<<1, nthreads, h2_smem>>>(
            o_h2_eu, g_h2_ws, g_x, g_u, NX, NU, d_robotModel, gravity, dt, 1);
        gpuErrchkKernel();
        grid_plant::plant_step_hessian_kernel<T, grid::IntegratorType::SEMI_IMPLICIT_EULER><<<1, nthreads, h2_smem>>>(
            o_h2_si, g_h2_ws, g_x, g_u, NX, NU, d_robotModel, gravity, dt, 1);
        gpuErrchkKernel();
        if (g_h2_ws) cudaFree(g_h2_ws);
        cudaFree(g_x);
    }
#endif

    // plant_step_kernel inlines the integrator-gradient with a FIXED-SIZE caller
    // scratch pool (s_temp[4096]). That inner needs FD_DU_MAX_SHARED_MEM_COUNT
    // floats of temp; on big floating-base robots (e.g. go2: 12040 > 4096) the
    // pool overflows -> out-of-bounds. The plant_step / integrator pass-through is
    // only meaningful / sized for robots where it fits (iiwa14:fixed = 2535), so
    // SKIP it (printing a parseable sentinel) rather than corrupting memory. The
    // centroidal validation above is independent and unaffected.
    constexpr int PLANT_STEP_TEMP_FLOATS = 4096;  // == s_temp[4096] in plant_step_kernel
    const bool plant_step_fits = (grid::FD_DU_MAX_SHARED_MEM_COUNT <= PLANT_STEP_TEMP_FLOATS);

    if (plant_step_fits) {
        plant_step_kernel<T><<<1, nthreads, step_dyn>>>(g_q, g_qd, g_u, dt, d_robotModel, gravity, o_pxk, o_pdab);
        gpuErrchkKernel();

        // grid:: integrator pass-through oracle, via the host wrappers.
        const int input_count = NQ + 2 * NV;
        std::vector<T> packed(input_count);
        for (int i = 0; i < NQ; ++i) packed[i] = h_q[i];
        for (int i = 0; i < NV; ++i) { packed[NQ + i] = h_qd[i]; packed[NQ + NV + i] = h_u[i]; }
        const dim3 bd(1, 1, 1), td(nthreads, 1, 1);
        std::memcpy(hd_data->h_q_qd_u, packed.data(), input_count * sizeof(T));
        grid::integrator<T, grid::IntegratorType::EULER>(hd_data, d_robotModel, gravity, dt, 1, bd, td, streams);
        print_vector("integrator_x_kp1", hd_data->h_x_kp1, NX);
        std::memcpy(hd_data->h_q_qd_u, packed.data(), input_count * sizeof(T));
        grid::integrator_gradient<T, grid::IntegratorType::EULER>(hd_data, d_robotModel, gravity, dt, 1, bd, td, streams);
        print_matrix_col_major("integrator_dAB", hd_data->h_dAB, 2 * NV, 3 * NV);
    } else {
        std::cout << "BEGIN plant_step_skipped 1 1\n1\nEND plant_step_skipped\n";
    }

    // print everything
    dcopy_out("state_cost_value", o_sv, 1, 1);
    dcopy_out("state_cost_grad", o_sg, 1, NX);
    dcopy_out("state_cost_hess", o_sh, NX, NX);
    dcopy_out("input_cost_value", o_iv, 1, 1);
    dcopy_out("input_cost_grad", o_ig, 1, NU);
    dcopy_out("input_cost_hess", o_ih, NU, NU);
    dcopy_out("ee_cost_value", o_ev, 1, 1);
    dcopy_out("ee_cost_grad", o_eg, 1, NX);
    dcopy_out("ee_cost_hess", o_eh, NX, NX);
    dcopy_out("pos_barrier_value", o_pbv, 1, 1);
    dcopy_out("pos_barrier_grad", o_pbg, 1, NX);
    dcopy_out("pos_barrier_hess_diag", o_pbh, 1, NQ);
    dcopy_out("vel_barrier_value", o_vbv, 1, 1);
    dcopy_out("vel_barrier_grad", o_vbg, 1, NX);
    dcopy_out("ctrl_barrier_value", o_cbv, 1, 1);
    dcopy_out("ctrl_barrier_grad", o_cbg, 1, NU);
    if (plant_step_fits) {
        dcopy_out("plant_dAB", o_pdab, 2 * NV, 3 * NV);
        dcopy_out("plant_x_kp1", o_pxk, 1, NX);
    }
    dcopy_out("ee_pos", o_eepos, 1, 3);
#if defined(GRID_PLANT_HAS_COM_COST) && defined(GRID_PLANT_HAS_MOMENTUM_COST)
    dcopy_out("com_cost_value", o_comv, 1, 1);
    dcopy_out("com_cost_grad", o_comg, 1, NX);
    dcopy_out("com_cost_hess", o_comh, NX, NX);
    dcopy_out("momentum_cost_value", o_momv, 1, 1);
    dcopy_out("momentum_cost_grad", o_momg, 1, NX);
    dcopy_out("momentum_cost_hess", o_momh, NX, NX);
#else
    // Centroidal costs are not emitted for this robot/config (com/ccrba absent,
    // e.g. a mimic robot). Emit a parseable sentinel so the Python centroidal test
    // pytest.skips this cell (mirrors the plant_step_skipped sentinel above).
    std::cout << "BEGIN com_cost_skipped 1 1\n1\nEND com_cost_skipped\n";
#endif
#ifdef GRID_PLANT_HAS_STEP_HESSIAN
    // Row-major flat (1 x D2AB_CNT); reshaped to (2*NV, 3*NV, 3*NV) C-order in Python.
    dcopy_out("plant_d2AB_euler", o_h2_eu, 1, D2AB_CNT);
    dcopy_out("plant_d2AB_si_euler", o_h2_si, 1, D2AB_CNT);
#endif

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main() {
    run<float>();
    return 0;
}

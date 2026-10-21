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
//   (momentum_cost: the full tangent-state GN cost is covered by
//    test_cuda_momentum_contract.py / cuda_momentum_contract_runner.cu, not here)
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
constexpr int NB = grid::NUM_BODIES;   // bodies; exceeds NV on mimic robots (fr3: 9 vs 8)

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
    // Dynamic arena for the caller-scratch EE-cost inners (the launch reserves
    // END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES; s_scratch[NX] above is the
    // tiny reduction buffer for the quadratic/barrier costs and is far too small here).
    extern __shared__ __align__(16) T s_ee_arena[];
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
    // FLOATING base: the STATE cost grad/hess gain a trailing `s_q` (xyzw config,
    // used ONLY to build R under MUJOCO_OUTPUT). On the pin path (<T,false>) it is
    // unused, so the config prefix s_x is a valid pass. Fixed-base has no such arg
    // (the signatures differ by ARG COUNT), so gate compile-time on NUM_POS!=NUM_VEL.
    if constexpr (grid::NUM_POS != grid::NUM_VEL) {
        grid_plant::quadratic_state_cost_gradient<T, false>(s_grad, s_x, s_xdes, s_Q, s_x);
        grid_plant::quadratic_state_cost_hessian<T, false>(s_hess, s_Q, s_x);
    } else {
        grid_plant::quadratic_state_cost_gradient<T, false>(s_grad, s_x, s_xdes, s_Q);
        grid_plant::quadratic_state_cost_hessian<T, false>(s_hess, s_Q);
    }
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
    grid_plant::ee_pos_cost<T, PLANT_EE>(s_out, s_x, s_pdes, s_W, s_eePos, s_ee_arena, d_robotModel);
    __syncthreads(); if (tid == 0) { o_ee_val[0] = s_out[0]; for (int r = 0; r < 3; ++r) o_eepos[r] = s_eePos[6 * PLANT_EE + r]; } __syncthreads();
    grid_plant::ee_pos_cost_gradient<T, PLANT_EE, false>(s_grad, s_x, s_pdes, s_W, s_eePos, s_deePos, s_ee_arena, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_ee_grad[i] = s_grad[i];
    __syncthreads();
    // GAUSS_NEWTON=true pinned (template <T, EE, ACCUMULATE, GAUSS_NEWTON>): this leg
    // checks the ratified GN surface the solver composites use; the full-Newton default
    // is validated separately (newton-vs-FD of the analytic gradient).
    grid_plant::ee_pos_cost_hessian<T, PLANT_EE, false, true>(s_hess, s_x, nullptr, s_W, nullptr, s_deePos, nullptr, s_ee_arena, d_robotModel);
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
    // integrator-gradient scratch (caller-placed; mirrors the integrator-gradient kernel).
    // s_vaf is body-indexed (three 6-vectors per body): the generated kernels size it
    // 18*NUM_JOINTS on mimic robots (NB > NV) and 18*NV otherwise (guide 1a), so the
    // caller-placed copy takes the larger of the two; 18*NV alone overflowed into
    // s_Minv on fr3 (9 bodies, 8 velocities) and corrupted the qd rows of dAB.
    __shared__ T s_df_du[NV * 2 * NV], s_dc_du[NV * 2 * NV], s_vaf[18 * (NB > NV ? NB : NV)], s_Minv[NV * NV], s_qdd[NV];
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

// ---- centroidal plant cost: com_cost ----
// These compose grid::com_device / grid::ccrba_device, which use an `extern
// __shared__` dynamic arena (COM_DYNAMIC_SHARED_MEM_BYTES). The launch
// must size dynamic smem to it and raise the opt-in attribute.
// Emitted ONLY when grid::com_device + grid::ccrba_device are present (the
// GRID_PLANT_HAS_COM_COST / GRID_PLANT_HAS_MOMENTUM_COST macros, emitted by
// _plant.py). For a robot/config that lacks them (e.g. a mimic robot whose ccrba
// is gated off), the whole centroidal block (kernel + alloc + launch + print) is
// #if-compiled out and a parseable `*_skipped` sentinel is emitted instead.
// Validated on iiwa14:fixed / go2:floating (both non-mimic, macros present).
#if defined(GRID_PLANT_HAS_COM_COST)
template <typename T>
__global__ void plant_centroidal_kernel(const T *g_q, const T *g_qd,
                                        const grid::robotModel<T> *d_robotModel,
                                        T *o_com_val, T *o_com_grad, T *o_com_hess) {
    __shared__ T s_q[NQ], s_qd[NV];
    // Caller-scratch centroidal arena for the com cost inners (the launch reserves
    // COM_DYNAMIC_SHARED_MEM_BYTES; the cost inner lays out s_com/s_extra here).
    extern __shared__ __align__(16) T s_cent_arena[];
    __shared__ T s_pdes[3], s_cW[3];       // CoM desired + per-axis weight
    __shared__ T s_out[1];
    __shared__ T s_grad[NX], s_hess[NX * NX];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NQ; i += nth) s_q[i] = g_q[i];
    for (int i = tid; i < NV; i += nth) s_qd[i] = g_qd[i];
    for (int r = tid; r < 3; r += nth) { s_pdes[r] = com_pdes_val<T>(r); s_cW[r] = com_W_val<T>(r); }
    __syncthreads();

    // ---- CoM-tracking cost (value + grad over x=[q;qd] + GN hess) ----
    grid_plant::com_cost<T>(s_out, s_q, s_pdes, s_cW, s_cent_arena, d_robotModel);
    __syncthreads(); if (tid == 0) o_com_val[0] = s_out[0]; __syncthreads();
    grid_plant::com_cost_gradient<T, false>(s_grad, s_q, s_pdes, s_cW, s_cent_arena, d_robotModel);
    __syncthreads();
    grid_plant::com_cost_hessian<T, false>(s_hess, s_q, s_cW, s_cent_arena, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_com_grad[i] = s_grad[i];
    for (int i = tid; i < NX * NX; i += nth) o_com_hess[i] = s_hess[i];
    __syncthreads();

}
#endif  // GRID_PLANT_HAS_COM_COST

// ---- tracking_cost PRESET check (fixed-base only; GRID_PLANT_HAS_TRACKING_COST) ----
// Drives grid_plant::tracking_cost[_gradient/_hessian] (the GATO BSQP recipe, a chained
// ACCUMULATE composition) AND an INDEPENDENT reference that recomputes each term standalone
// (ACCUMULATE=false into a temp) and sums them explicitly — a different code path, so it
// catches ACCUMULATE-chain / block-offset bugs in the preset. The Python test asserts
// preset == reference for all five blocks (value, s_qk, s_rk, s_Qk, s_Rk). The per-term
// inners themselves are oracle-validated by the main plant equivalence test.
#if defined(GRID_PLANT_HAS_TRACKING_COST)
template <typename T>
__global__ void tracking_preset_kernel(const T *g_q, const T *g_qd, const T *g_u,
                                       const grid::robotModel<T> *d_robotModel,
                                       T *o_pv, T *o_pqk, T *o_prk, T *o_pQk, T *o_pRk,
                                       T *o_rv, T *o_rqk, T *o_rrk, T *o_rQk, T *o_rRk) {
    __shared__ T s_x[NX], s_u[NU], s_xdes[NX], s_udes[NU], s_eedes[3], s_Q[NX], s_R[NU], s_W[3];
    __shared__ T s_lo_q[NQ], s_hi_q[NQ], s_lo_v[NV], s_hi_v[NV], s_lo_u[NU], s_hi_u[NU];
    __shared__ T s_eePos[6 * grid::NUM_EES], s_deePos[6 * NV * grid::NUM_EES];
    extern __shared__ __align__(16) T s_arena[];
    __shared__ T s_pv[1], s_pqk[NX], s_prk[NU], s_pQk[NX * NX], s_pRk[NU * NU];
    __shared__ T s_rv[1], s_rqk[NX], s_rrk[NU], s_rQk[NX * NX], s_rRk[NU * NU];
    __shared__ T s_tv[1], s_tg[NX], s_th[NX * NX];

    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    const T mu = static_cast<T>(0.1);

    // deterministic setup (same functions as plant_kernel / the Python test)
    for (int i = tid; i < NX; i += nth) { s_x[i] = (i < NQ) ? g_q[i] : g_qd[i - NQ]; s_xdes[i] = x_des_val<T>(i); s_Q[i] = Qw_val<T>(i); }
    for (int i = tid; i < NU; i += nth) { s_u[i] = g_u[i]; s_udes[i] = u_des_val<T>(i); s_R[i] = Rw_val<T>(i); }
    for (int r = tid; r < 3; r += nth) { s_eedes[r] = static_cast<T>(0); s_W[r] = Ww_val<T>(r); }
    for (int i = tid; i < NQ; i += nth) { s_lo_q[i] = g_q[i] - static_cast<T>(1); s_hi_q[i] = g_q[i] + static_cast<T>(1); if (i == 0) { s_lo_q[i] = -HUGE_VALF; s_hi_q[i] = HUGE_VALF; } }
    for (int i = tid; i < NV; i += nth) { s_lo_v[i] = g_qd[i] - static_cast<T>(1); s_hi_v[i] = g_qd[i] + static_cast<T>(1); }
    for (int i = tid; i < NU; i += nth) { s_lo_u[i] = g_u[i] - static_cast<T>(1); s_hi_u[i] = g_u[i] + static_cast<T>(1); }
    __syncthreads();

    // ===== PRESET (chained ACCUMULATE) =====
    grid_plant::tracking_cost<T, PLANT_EE>(s_pv, s_x, s_u, s_xdes, s_udes, s_eedes, s_Q, s_R, s_W,
        s_lo_q, s_hi_q, mu, s_lo_v, s_hi_v, mu, s_lo_u, s_hi_u, mu, s_eePos, s_arena, d_robotModel);
    __syncthreads();
    grid_plant::tracking_cost_gradient<T, PLANT_EE>(s_pqk, s_prk, s_x, s_u, s_xdes, s_udes, s_eedes, s_Q, s_R, s_W,
        s_lo_q, s_hi_q, mu, s_lo_v, s_hi_v, mu, s_lo_u, s_hi_u, mu, s_eePos, s_deePos, s_arena, d_robotModel);
    __syncthreads();
    grid_plant::tracking_cost_hessian<T, PLANT_EE>(s_pQk, s_pRk, s_x, s_u, s_Q, s_R, s_W,
        s_lo_q, s_hi_q, mu, s_lo_v, s_hi_v, mu, s_lo_u, s_hi_u, mu, s_deePos, s_arena, d_robotModel);
    __syncthreads();

    // ===== INDEPENDENT REFERENCE (each term standalone, summed explicitly) =====
    if (tid == 0) s_rv[0] = static_cast<T>(0);
    for (int i = tid; i < NX; i += nth) s_rqk[i] = static_cast<T>(0);
    for (int i = tid; i < NU; i += nth) s_rrk[i] = static_cast<T>(0);
    for (int i = tid; i < NX * NX; i += nth) s_rQk[i] = static_cast<T>(0);
    for (int i = tid; i < NU * NU; i += nth) s_rRk[i] = static_cast<T>(0);
    __syncthreads();

    // value = sum of the 6 standalone term values (barriers always +=, so pre-zero s_tv)
    grid_plant::ee_pos_cost<T, PLANT_EE>(s_tv, s_x, s_eedes, s_W, s_eePos, s_arena, d_robotModel);
    __syncthreads(); if (tid == 0) s_rv[0] += s_tv[0]; __syncthreads();
    grid_plant::quadratic_state_cost<T>(s_tv, s_x, s_xdes, s_Q, s_th);
    __syncthreads(); if (tid == 0) s_rv[0] += s_tv[0]; __syncthreads();
    grid_plant::quadratic_input_cost<T>(s_tv, s_u, s_udes, s_R, s_th);
    __syncthreads(); if (tid == 0) s_rv[0] += s_tv[0]; __syncthreads();
    if (tid == 0) s_tv[0] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_position_barrier<T>(s_tv, s_x, s_lo_q, s_hi_q, mu, s_th);
    __syncthreads(); if (tid == 0) s_rv[0] += s_tv[0]; __syncthreads();
    if (tid == 0) s_tv[0] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_velocity_barrier<T>(s_tv, s_x, s_lo_v, s_hi_v, mu, s_th);
    __syncthreads(); if (tid == 0) s_rv[0] += s_tv[0]; __syncthreads();
    if (tid == 0) s_tv[0] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_torque_barrier<T>(s_tv, s_u, s_lo_u, s_hi_u, mu, s_th);
    __syncthreads(); if (tid == 0) s_rv[0] += s_tv[0]; __syncthreads();

    // s_qk = ee_grad + state_grad + posb_grad + velb_grad
    grid_plant::ee_pos_cost_gradient<T, PLANT_EE, false>(s_tg, s_x, s_eedes, s_W, s_eePos, s_deePos, s_arena, d_robotModel);
    __syncthreads(); for (int i = tid; i < NX; i += nth) s_rqk[i] += s_tg[i]; __syncthreads();
    grid_plant::quadratic_state_cost_gradient<T, false>(s_tg, s_x, s_xdes, s_Q);
    __syncthreads(); for (int i = tid; i < NX; i += nth) s_rqk[i] += s_tg[i]; __syncthreads();
    for (int i = tid; i < NX; i += nth) s_tg[i] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_position_barrier_gradient<T, 0, 0>(s_tg, s_x, s_lo_q, s_hi_q, mu);
    __syncthreads(); for (int i = tid; i < NX; i += nth) s_rqk[i] += s_tg[i]; __syncthreads();
    for (int i = tid; i < NX; i += nth) s_tg[i] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_velocity_barrier_gradient<T, NQ, NQ>(s_tg, s_x, s_lo_v, s_hi_v, mu);
    __syncthreads(); for (int i = tid; i < NX; i += nth) s_rqk[i] += s_tg[i]; __syncthreads();

    // s_rk = input_grad + ctrlb_grad
    grid_plant::quadratic_input_cost_gradient<T, false>(s_tg, s_u, s_udes, s_R);
    __syncthreads(); for (int i = tid; i < NU; i += nth) s_rrk[i] += s_tg[i]; __syncthreads();
    for (int i = tid; i < NU; i += nth) s_tg[i] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_torque_barrier_gradient<T, 0, 0>(s_tg, s_u, s_lo_u, s_hi_u, mu);
    __syncthreads(); for (int i = tid; i < NU; i += nth) s_rrk[i] += s_tg[i]; __syncthreads();

    // s_Qk = ee_hess + state_hess + posb_hess + velb_hess
    grid_plant::ee_pos_cost_hessian<T, PLANT_EE, false, true>(s_th, s_x, nullptr, s_W, nullptr, s_deePos, nullptr, s_arena, d_robotModel);
    __syncthreads(); for (int i = tid; i < NX * NX; i += nth) s_rQk[i] += s_th[i]; __syncthreads();
    grid_plant::quadratic_state_cost_hessian<T, false>(s_th, s_Q);
    __syncthreads(); for (int i = tid; i < NX * NX; i += nth) s_rQk[i] += s_th[i]; __syncthreads();
    for (int i = tid; i < NX * NX; i += nth) s_th[i] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_position_barrier_hessian<T, NX, 0, 0>(s_th, s_x, s_lo_q, s_hi_q, mu);
    __syncthreads(); for (int i = tid; i < NX * NX; i += nth) s_rQk[i] += s_th[i]; __syncthreads();
    for (int i = tid; i < NX * NX; i += nth) s_th[i] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_velocity_barrier_hessian<T, NX, NQ, NQ>(s_th, s_x, s_lo_v, s_hi_v, mu);
    __syncthreads(); for (int i = tid; i < NX * NX; i += nth) s_rQk[i] += s_th[i]; __syncthreads();

    // s_Rk = input_hess + ctrlb_hess (reuse s_th; NX*NX >= NU*NU)
    grid_plant::quadratic_input_cost_hessian<T, false>(s_th, s_R);
    __syncthreads(); for (int i = tid; i < NU * NU; i += nth) s_rRk[i] += s_th[i]; __syncthreads();
    for (int i = tid; i < NU * NU; i += nth) s_th[i] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_torque_barrier_hessian<T, NU, 0, 0>(s_th, s_u, s_lo_u, s_hi_u, mu);
    __syncthreads(); for (int i = tid; i < NU * NU; i += nth) s_rRk[i] += s_th[i]; __syncthreads();

    // write out preset + reference
    if (tid == 0) { o_pv[0] = s_pv[0]; o_rv[0] = s_rv[0]; }
    for (int i = tid; i < NX; i += nth) { o_pqk[i] = s_pqk[i]; o_rqk[i] = s_rqk[i]; }
    for (int i = tid; i < NU; i += nth) { o_prk[i] = s_prk[i]; o_rrk[i] = s_rrk[i]; }
    for (int i = tid; i < NX * NX; i += nth) { o_pQk[i] = s_pQk[i]; o_rQk[i] = s_rQk[i]; }
    for (int i = tid; i < NU * NU; i += nth) { o_pRk[i] = s_pRk[i]; o_rRk[i] = s_rRk[i]; }
}
#endif  // GRID_PLANT_HAS_TRACKING_COST

// ---- cost menu: activated terms (pseudo-Huber), the linear feature, the cost-frame residuals,
// the barrier activations and the menu preset. Deterministic recipes mirrored by the Python test.
// Outputs are packed per term as [value(1) | grad(N) | hess(N*N)]; o_quad = [state value, frame-0
// position value] under the quadratic activation; o_menu = [preset value, qk, rk, Qk, Rk | reference ...].
#if defined(GRID_PLANT_HAS_COST_FRAMES)
template <typename T> __host__ __device__ T menu_eps_val(int t)       { return static_cast<T>(0.02) + static_cast<T>(0.01) * t; }
template <typename T> __host__ __device__ T lin_Ax_val(int m, int n)  { return static_cast<T>(0.1) * (m + 1) + static_cast<T>(0.05) * n * ((m == 0) ? 1 : -1); }
template <typename T> __host__ __device__ T lin_bx_val(int m)         { return static_cast<T>(0.3) * m; }
template <typename T> __host__ __device__ T lin_tx_val(int m)         { return static_cast<T>(0.2) + static_cast<T>(0.1) * m; }
template <typename T> __host__ __device__ T lin_Wx_val(int m)         { return static_cast<T>(4.0) + m; }
template <typename T> __host__ __device__ T lin_Au_val(int n)         { return static_cast<T>(0.2) - static_cast<T>(0.03) * n; }
template <typename T> __host__ __device__ T vdes_val(int r)           { return static_cast<T>(0.1) * (r % 3) - static_cast<T>(0.05) * (r / 3); }
template <typename T> __host__ __device__ T Wv_val(int r)             { return static_cast<T>(1.0) + r; }
template <typename T> __host__ __device__ T Wcf_val(int r)            { return static_cast<T>(5.0) + static_cast<T>(0.5) * r; }
template <typename T> __host__ __device__ T cfdes_val(int r)          { return static_cast<T>(0.05) * (r + 1); }
// R_des = Rz(0.3) Ry(0.2) Rx(0.1), column-major
template <typename T> __host__ __device__ T Rdes_val(int i) {
    const double cy = cos(0.3), sy = sin(0.3), cp = cos(0.2), sp = sin(0.2), cr = cos(0.1), sr = sin(0.1);
    const double R[9] = { cy*cp, sy*cp, -sp,  cy*sp*sr - sy*cr, sy*sp*sr + cy*cr, cp*sr,  cy*sp*cr + sy*sr, sy*sp*cr - cy*sr, cp*cr };
    return static_cast<T>(R[i]);
}
#define MENU_MX 2
#define MENU_MU 1
#define MENU_STATE_CNT (1 + NX + NX * NX)
#define MENU_INPUT_CNT (1 + NU + NU * NU)
#define MENU_PRESET_CNT (1 + NX + NU + NX * NX + NU * NU)
template <typename T>
__global__ void cost_menu_kernel(const T *g_q, const T *g_qd, const T *g_u, const grid::robotModel<T> *d_robotModel, T gravity,
                                 T *o_as, T *o_ai, T *o_ae, T *o_lx, T *o_lu, T *o_quad, T *o_menu, T *o_grav,
                                 T *o_or, T *o_vel, T *o_f2, T *o_rlb, T *o_qb) {
    __shared__ T s_x[NX], s_u[NU], s_xdes[NX], s_udes[NU], s_eedes[3], s_Q[NX], s_R[NU], s_W[3];
    __shared__ T s_Ax[MENU_MX * NX], s_bx[MENU_MX], s_tx[MENU_MX], s_Wx[MENU_MX];
    __shared__ T s_Au[MENU_MU * NU], s_bu[MENU_MU], s_tu[MENU_MU], s_Wu[MENU_MU];
    __shared__ T s_Rdes[9], s_vdes[6], s_Wv[6], s_cfdes[3], s_Wcf[3], s_xlo[NX], s_xhi[NX];
    __shared__ int s_act[grid_plant::GRID_PLANT_MENU_TERMS]; __shared__ T s_eps[grid_plant::GRID_PLANT_MENU_TERMS];
    __shared__ T s_frame[grid_plant::GRID_PLANT_FRAME_SCRATCH_COUNT];
    extern __shared__ __align__(16) T s_arena[];
    __shared__ T s_out[1], s_grad[NX], s_hess[NX * NX];
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nth = blockDim.x * blockDim.y;
    for (int i = tid; i < NX; i += nth) { s_x[i] = (i < NQ) ? g_q[i] : g_qd[i - NQ]; s_xdes[i] = x_des_val<T>(i); s_Q[i] = Qw_val<T>(i);
                                           s_xlo[i] = x_des_val<T>(i) - static_cast<T>(0.45); s_xhi[i] = x_des_val<T>(i) + static_cast<T>(0.45); }
    for (int i = tid; i < NU; i += nth) { s_u[i] = g_u[i]; s_udes[i] = u_des_val<T>(i); s_R[i] = Rw_val<T>(i); }
    for (int r = tid; r < 3; r += nth) { s_eedes[r] = static_cast<T>(0); s_W[r] = Ww_val<T>(r); s_cfdes[r] = cfdes_val<T>(r); s_Wcf[r] = Wcf_val<T>(r); }
    for (int r = tid; r < 6; r += nth) { s_vdes[r] = vdes_val<T>(r); s_Wv[r] = Wv_val<T>(r); }
    for (int i = tid; i < 9; i += nth) { s_Rdes[i] = Rdes_val<T>(i); }
    for (int i = tid; i < MENU_MX * NX; i += nth) { s_Ax[i] = lin_Ax_val<T>(i % MENU_MX, i / MENU_MX); }
    for (int m = tid; m < MENU_MX; m += nth) { s_bx[m] = lin_bx_val<T>(m); s_tx[m] = lin_tx_val<T>(m); s_Wx[m] = lin_Wx_val<T>(m); }
    for (int n = tid; n < MENU_MU * NU; n += nth) { s_Au[n] = lin_Au_val<T>(n); }
    for (int m = tid; m < MENU_MU; m += nth) { s_bu[m] = static_cast<T>(0.1); s_tu[m] = static_cast<T>(-0.1); s_Wu[m] = static_cast<T>(5.0); }
    for (int t = tid; t < grid_plant::GRID_PLANT_MENU_TERMS; t += nth) { s_act[t] = grid_plant::GRID_PLANT_ACTIVATION_PSEUDO_HUBER; s_eps[t] = menu_eps_val<T>(t); }
    __syncthreads();
    const int PH = grid_plant::GRID_PLANT_ACTIVATION_PSEUDO_HUBER;

#if defined(GRID_PLANT_HAS_GRAVITY_TORQUE)
    grid_plant::gravity_torque<T>(s_grad, s_x, s_arena, d_robotModel, gravity);
    __syncthreads();
    for (int i = tid; i < NV; i += nth) o_grav[i] = s_grad[i];
    __syncthreads();
#endif
    // activated state cost (pseudo-Huber), then the two barrier activations with bounds x_des +- 0.45
    // (0.45, not 0.5: the corner samples hold round values, and a bound that lands exactly on a
    // sample value flips the quadratic barrier's activity between fp32 and the fp64 reference)
    grid_plant::state_cost<T>(s_out, s_x, s_xdes, s_Q, PH, s_eps[1], nullptr, nullptr, s_arena);
    __syncthreads(); if (tid == 0) o_as[0] = s_out[0]; __syncthreads();
    grid_plant::state_cost_gradient<T, false>(s_grad, s_x, s_xdes, s_Q, PH, s_eps[1], nullptr, nullptr);
    grid_plant::state_cost_hessian<T, false>(s_hess, s_x, s_xdes, s_Q, PH, s_eps[1], nullptr, nullptr);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_as[1 + i] = s_grad[i];
    for (int i = tid; i < NX * NX; i += nth) o_as[1 + NX + i] = s_hess[i];
    __syncthreads();
    grid_plant::state_cost<T>(s_out, s_x, s_xdes, s_Q, grid_plant::GRID_PLANT_ACTIVATION_RELAXED_LOG_BARRIER, static_cast<T>(0.2), s_xlo, s_xhi, s_arena);
    __syncthreads(); if (tid == 0) o_rlb[0] = s_out[0]; __syncthreads();
    grid_plant::state_cost_gradient<T, false>(s_grad, s_x, s_xdes, s_Q, grid_plant::GRID_PLANT_ACTIVATION_RELAXED_LOG_BARRIER, static_cast<T>(0.2), s_xlo, s_xhi);
    grid_plant::state_cost_hessian<T, false>(s_hess, s_x, s_xdes, s_Q, grid_plant::GRID_PLANT_ACTIVATION_RELAXED_LOG_BARRIER, static_cast<T>(0.2), s_xlo, s_xhi);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_rlb[1 + i] = s_grad[i];
    for (int i = tid; i < NX * NX; i += nth) o_rlb[1 + NX + i] = s_hess[i];
    __syncthreads();
    grid_plant::state_cost<T>(s_out, s_x, s_xdes, s_Q, grid_plant::GRID_PLANT_ACTIVATION_QUADRATIC_BARRIER, static_cast<T>(0), s_xlo, s_xhi, s_arena);
    __syncthreads(); if (tid == 0) o_qb[0] = s_out[0]; __syncthreads();
    grid_plant::state_cost_gradient<T, false>(s_grad, s_x, s_xdes, s_Q, grid_plant::GRID_PLANT_ACTIVATION_QUADRATIC_BARRIER, static_cast<T>(0), s_xlo, s_xhi);
    grid_plant::state_cost_hessian<T, false>(s_hess, s_x, s_xdes, s_Q, grid_plant::GRID_PLANT_ACTIVATION_QUADRATIC_BARRIER, static_cast<T>(0), s_xlo, s_xhi);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_qb[1 + i] = s_grad[i];
    for (int i = tid; i < NX * NX; i += nth) o_qb[1 + NX + i] = s_hess[i];
    __syncthreads();
    // activated input cost
    grid_plant::input_cost<T>(s_out, s_u, s_udes, s_R, PH, s_eps[2], nullptr, nullptr, s_arena);
    __syncthreads(); if (tid == 0) o_ai[0] = s_out[0]; __syncthreads();
    grid_plant::input_cost_gradient<T, false>(s_grad, s_u, s_udes, s_R, PH, s_eps[2], nullptr, nullptr);
    grid_plant::input_cost_hessian<T, false>(s_hess, s_u, s_udes, s_R, PH, s_eps[2], nullptr, nullptr);
    __syncthreads();
    for (int i = tid; i < NU; i += nth) o_ai[1 + i] = s_grad[i];
    for (int i = tid; i < NU * NU; i += nth) o_ai[1 + NU + i] = s_hess[i];
    __syncthreads();
    // frame 0 (the end-effector target) position cost (s_x[:NQ] is q)
    grid_plant::frame_position_cost<T>(s_out, s_x, 0, s_eedes, s_W, PH, s_eps[0], nullptr, nullptr, s_frame, s_arena, d_robotModel);
    __syncthreads(); if (tid == 0) o_ae[0] = s_out[0]; __syncthreads();
    grid_plant::frame_position_cost_gradient<T, false>(s_grad, s_x, 0, s_eedes, s_W, PH, s_eps[0], nullptr, nullptr, s_frame, s_arena, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_ae[1 + i] = s_grad[i];
    __syncthreads();
    grid_plant::frame_position_cost_hessian<T, false>(s_hess, s_x, 0, s_eedes, s_W, PH, s_eps[0], nullptr, nullptr, s_frame, s_arena, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX * NX; i += nth) o_ae[1 + NX + i] = s_hess[i];
    __syncthreads();
    // frame 0 orientation (log map)
    grid_plant::frame_orientation_cost<T>(s_out, s_x, 0, s_Rdes, s_W, PH, s_eps[5], s_frame, s_arena, d_robotModel);
    __syncthreads(); if (tid == 0) o_or[0] = s_out[0]; __syncthreads();
    grid_plant::frame_orientation_cost_gradient<T, false>(s_grad, s_x, 0, s_Rdes, s_W, PH, s_eps[5], s_frame, s_arena, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_or[1 + i] = s_grad[i];
    __syncthreads();
    grid_plant::frame_orientation_cost_hessian<T, false>(s_hess, s_x, 0, s_Rdes, s_W, PH, s_eps[5], s_frame, s_arena, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX * NX; i += nth) o_or[1 + NX + i] = s_hess[i];
    __syncthreads();
    // frame 1 position
    grid_plant::frame_position_cost<T>(s_out, s_x, 1, s_cfdes, s_Wcf, PH, s_eps[7], nullptr, nullptr, s_frame, s_arena, d_robotModel);
    __syncthreads(); if (tid == 0) o_f2[0] = s_out[0]; __syncthreads();
    grid_plant::frame_position_cost_gradient<T, false>(s_grad, s_x, 1, s_cfdes, s_Wcf, PH, s_eps[7], nullptr, nullptr, s_frame, s_arena, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_f2[1 + i] = s_grad[i];
    __syncthreads();
    grid_plant::frame_position_cost_hessian<T, false>(s_hess, s_x, 1, s_cfdes, s_Wcf, PH, s_eps[7], nullptr, nullptr, s_frame, s_arena, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX * NX; i += nth) o_f2[1 + NX + i] = s_hess[i];
    __syncthreads();
#if defined(GRID_PLANT_HAS_FRAME_VELOCITY_COST)
    grid_plant::frame_velocity_cost<T>(s_out, s_x, s_x + NQ, 0, s_vdes, s_Wv, PH, s_eps[6], nullptr, nullptr, s_frame, s_arena, d_robotModel);
    __syncthreads(); if (tid == 0) o_vel[0] = s_out[0]; __syncthreads();
    grid_plant::frame_velocity_cost_gradient<T, false>(s_grad, s_x, s_x + NQ, 0, s_vdes, s_Wv, PH, s_eps[6], nullptr, nullptr, s_frame, s_arena, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_vel[1 + i] = s_grad[i];
    __syncthreads();
    grid_plant::frame_velocity_cost_hessian<T, false>(s_hess, s_x, s_x + NQ, 0, s_vdes, s_Wv, PH, s_eps[6], nullptr, nullptr, s_frame, s_arena, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX * NX; i += nth) o_vel[1 + NX + i] = s_hess[i];
    __syncthreads();
#endif
    // linear feature on x
    grid_plant::linear_feature_cost<T, MENU_MX, NX>(s_out, s_x, s_Ax, s_bx, s_tx, s_Wx, PH, s_eps[3], nullptr, nullptr, s_arena);
    __syncthreads(); if (tid == 0) o_lx[0] = s_out[0]; __syncthreads();
    grid_plant::linear_feature_cost_gradient<T, MENU_MX, NX, false>(s_grad, s_x, s_Ax, s_bx, s_tx, s_Wx, PH, s_eps[3], nullptr, nullptr, s_arena);
    __syncthreads();
    grid_plant::linear_feature_cost_hessian<T, MENU_MX, NX, false>(s_hess, s_x, s_Ax, s_bx, s_tx, s_Wx, PH, s_eps[3], nullptr, nullptr, s_arena);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_lx[1 + i] = s_grad[i];
    for (int i = tid; i < NX * NX; i += nth) o_lx[1 + NX + i] = s_hess[i];
    __syncthreads();
    // linear feature on u
    grid_plant::linear_feature_cost<T, MENU_MU, NU>(s_out, s_u, s_Au, s_bu, s_tu, s_Wu, PH, s_eps[4], nullptr, nullptr, s_arena);
    __syncthreads(); if (tid == 0) o_lu[0] = s_out[0]; __syncthreads();
    grid_plant::linear_feature_cost_gradient<T, MENU_MU, NU, false>(s_grad, s_u, s_Au, s_bu, s_tu, s_Wu, PH, s_eps[4], nullptr, nullptr, s_arena);
    __syncthreads();
    grid_plant::linear_feature_cost_hessian<T, MENU_MU, NU, false>(s_hess, s_u, s_Au, s_bu, s_tu, s_Wu, PH, s_eps[4], nullptr, nullptr, s_arena);
    __syncthreads();
    for (int i = tid; i < NU; i += nth) o_lu[1 + i] = s_grad[i];
    for (int i = tid; i < NU * NU; i += nth) o_lu[1 + NU + i] = s_hess[i];
    __syncthreads();
    // the quadratic activation reproduces the quadratic terms
    grid_plant::state_cost<T>(s_out, s_x, s_xdes, s_Q, grid_plant::GRID_PLANT_ACTIVATION_QUADRATIC, s_eps[1], nullptr, nullptr, s_arena);
    __syncthreads(); if (tid == 0) o_quad[0] = s_out[0]; __syncthreads();
    grid_plant::frame_position_cost<T>(s_out, s_x, 0, s_eedes, s_W, grid_plant::GRID_PLANT_ACTIVATION_QUADRATIC, s_eps[0], nullptr, nullptr, s_frame, s_arena, d_robotModel);
    __syncthreads(); if (tid == 0) o_quad[1] = s_out[0]; __syncthreads();

#if defined(GRID_PLANT_HAS_COST_MENU)
    // ===== menu PRESET (every term pseudo-Huber, both linear features, frame 1 as the second position) vs the composition of the blocks above =====
    __shared__ T s_lo_q[NQ], s_hi_q[NQ], s_lo_v[NV], s_hi_v[NV], s_lo_u[NU], s_hi_u[NU];
    __shared__ T s_qk[NX], s_rk[NU], s_Qk[NX * NX], s_Rk[NU * NU];
    __shared__ T s_rv[1], s_tv[1], s_tg[NX], s_th[NX * NX];
    __shared__ grid_plant::cost_menu_params<T> s_p;
    const T mu = static_cast<T>(0.1);
    for (int i = tid; i < NQ; i += nth) { s_lo_q[i] = g_q[i] - static_cast<T>(1); s_hi_q[i] = g_q[i] + static_cast<T>(1); if (i == 0) { s_lo_q[i] = -HUGE_VALF; s_hi_q[i] = HUGE_VALF; } }
    for (int i = tid; i < NV; i += nth) { s_lo_v[i] = g_qd[i] - static_cast<T>(1); s_hi_v[i] = g_qd[i] + static_cast<T>(1); }
    for (int i = tid; i < NU; i += nth) { s_lo_u[i] = g_u[i] - static_cast<T>(1); s_hi_u[i] = g_u[i] + static_cast<T>(1); }
    if (tid == 0) {
        s_p.x_des = s_xdes; s_p.u_des = s_udes; s_p.ee_des = s_eedes; s_p.Q = s_Q; s_p.R = s_R; s_p.W = s_W;
        s_p.q_lower = s_lo_q; s_p.q_upper = s_hi_q; s_p.mu_q = mu; s_p.qd_lower = s_lo_v; s_p.qd_upper = s_hi_v; s_p.mu_qd = mu; s_p.u_lower = s_lo_u; s_p.u_upper = s_hi_u; s_p.mu_u = mu;
        s_p.activation = s_act; s_p.eps = s_eps;
        s_p.x_lower = nullptr; s_p.x_upper = nullptr; s_p.ub_lower = nullptr; s_p.ub_upper = nullptr;
        s_p.Ax = s_Ax; s_p.bx = s_bx; s_p.tx = s_tx; s_p.Wx = s_Wx; s_p.lx_lower = nullptr; s_p.lx_upper = nullptr;
        s_p.Au = s_Au; s_p.bu = s_bu; s_p.tu = s_tu; s_p.Wu = s_Wu; s_p.lu_lower = nullptr; s_p.lu_upper = nullptr;
        s_p.R_des = s_Rdes; s_p.Wo = s_W;
        s_p.v_des = s_vdes; s_p.Wv = s_Wv; s_p.v_lower = nullptr; s_p.v_upper = nullptr;
        s_p.cf_des = s_cfdes; s_p.Wcf = s_Wcf; s_p.cf_lower = nullptr; s_p.cf_upper = nullptr;
    }
    __syncthreads();
    grid_plant::tracking_cost_menu<T, MENU_MX, MENU_MU, 0, 1>(s_out, s_x, s_u, s_p, s_frame, s_arena, d_robotModel);
    __syncthreads(); if (tid == 0) o_menu[0] = s_out[0]; __syncthreads();
    grid_plant::tracking_cost_menu_gradient<T, MENU_MX, MENU_MU, 0, 1>(s_qk, s_rk, s_x, s_u, s_p, s_frame, s_arena, d_robotModel);
    __syncthreads();
    grid_plant::tracking_cost_menu_hessian<T, MENU_MX, MENU_MU, 0, 1>(s_Qk, s_Rk, s_x, s_u, s_p, s_frame, s_arena, d_robotModel);
    __syncthreads();
    for (int i = tid; i < NX; i += nth) o_menu[1 + i] = s_qk[i];
    for (int i = tid; i < NU; i += nth) o_menu[1 + NX + i] = s_rk[i];
    for (int i = tid; i < NX * NX; i += nth) o_menu[1 + NX + NU + i] = s_Qk[i];
    for (int i = tid; i < NU * NU; i += nth) o_menu[1 + NX + NU + NX * NX + i] = s_Rk[i];
    __syncthreads();
    // reference: the standalone blocks (written by this block), plus the barriers
    if (tid == 0) s_rv[0] = o_ae[0] + o_as[0] + o_ai[0] + o_lx[0] + o_lu[0] + o_or[0] + o_vel[0] + o_f2[0];
    for (int i = tid; i < NX; i += nth) s_qk[i] = o_ae[1 + i] + o_as[1 + i] + o_lx[1 + i] + o_or[1 + i] + o_vel[1 + i] + o_f2[1 + i];
    for (int i = tid; i < NU; i += nth) s_rk[i] = o_ai[1 + i] + o_lu[1 + i];
    for (int i = tid; i < NX * NX; i += nth) s_Qk[i] = o_ae[1 + NX + i] + o_as[1 + NX + i] + o_lx[1 + NX + i] + o_or[1 + NX + i] + o_vel[1 + NX + i] + o_f2[1 + NX + i];
    for (int i = tid; i < NU * NU; i += nth) s_Rk[i] = o_ai[1 + NU + i] + o_lu[1 + NU + i];
    if (tid == 0) s_tv[0] = static_cast<T>(0);
    __syncthreads();
    grid_plant::joint_position_barrier<T>(s_tv, s_x, s_lo_q, s_hi_q, mu, s_th);
    __syncthreads(); if (tid == 0) { s_rv[0] += s_tv[0]; s_tv[0] = static_cast<T>(0); } __syncthreads();
    grid_plant::joint_velocity_barrier<T>(s_tv, s_x, s_lo_v, s_hi_v, mu, s_th);
    __syncthreads(); if (tid == 0) { s_rv[0] += s_tv[0]; s_tv[0] = static_cast<T>(0); } __syncthreads();
    grid_plant::joint_torque_barrier<T>(s_tv, s_u, s_lo_u, s_hi_u, mu, s_th);
    __syncthreads(); if (tid == 0) { s_rv[0] += s_tv[0]; } __syncthreads();
    for (int i = tid; i < NX; i += nth) s_tg[i] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_position_barrier_gradient<T, 0, 0>(s_tg, s_x, s_lo_q, s_hi_q, mu);
    __syncthreads(); for (int i = tid; i < NX; i += nth) { s_qk[i] += s_tg[i]; s_tg[i] = static_cast<T>(0); } __syncthreads();
    grid_plant::joint_velocity_barrier_gradient<T, NQ, NQ>(s_tg, s_x, s_lo_v, s_hi_v, mu);
    __syncthreads(); for (int i = tid; i < NX; i += nth) { s_qk[i] += s_tg[i]; s_tg[i] = static_cast<T>(0); } __syncthreads();
    grid_plant::joint_torque_barrier_gradient<T, 0, 0>(s_tg, s_u, s_lo_u, s_hi_u, mu);
    __syncthreads(); for (int i = tid; i < NU; i += nth) { s_rk[i] += s_tg[i]; } __syncthreads();
    for (int i = tid; i < NX * NX; i += nth) s_th[i] = static_cast<T>(0); __syncthreads();
    grid_plant::joint_position_barrier_hessian<T, NX, 0, 0>(s_th, s_x, s_lo_q, s_hi_q, mu);
    __syncthreads(); for (int i = tid; i < NX * NX; i += nth) { s_Qk[i] += s_th[i]; s_th[i] = static_cast<T>(0); } __syncthreads();
    grid_plant::joint_velocity_barrier_hessian<T, NX, NQ, NQ>(s_th, s_x, s_lo_v, s_hi_v, mu);
    __syncthreads(); for (int i = tid; i < NX * NX; i += nth) { s_Qk[i] += s_th[i]; s_th[i] = static_cast<T>(0); } __syncthreads();
    grid_plant::joint_torque_barrier_hessian<T, NU, 0, 0>(s_th, s_u, s_lo_u, s_hi_u, mu);
    __syncthreads(); for (int i = tid; i < NU * NU; i += nth) { s_Rk[i] += s_th[i]; } __syncthreads();
    const int ro = MENU_PRESET_CNT;
    if (tid == 0) o_menu[ro] = s_rv[0];
    for (int i = tid; i < NX; i += nth) o_menu[ro + 1 + i] = s_qk[i];
    for (int i = tid; i < NU; i += nth) o_menu[ro + 1 + NX + i] = s_rk[i];
    for (int i = tid; i < NX * NX; i += nth) o_menu[ro + 1 + NX + NU + i] = s_Qk[i];
    for (int i = tid; i < NU * NU; i += nth) o_menu[ro + 1 + NX + NU + NX * NX + i] = s_Rk[i];
#endif  // GRID_PLANT_HAS_COST_MENU
}
#endif  // GRID_PLANT_HAS_COST_FRAMES

// ---- tangent (log-map) state cost check (floating-base preset; GATO ASK3) ----
// x_des is built ON DEVICE from the input state: q_des = integrate(q, delta) with a
// fixed deterministic tangent delta (guaranteed-valid quaternion by construction),
// qd_des = qd + pattern. x_des is ALSO dumped so the Python side evaluates the oracle
// at the exact same (float-rounded) reference state — the integrate chart itself is
// already covered by the integrator equivalence suite.
#ifdef GRID_PLANT_HAS_TANGENT_STATE_COST
template <typename T>
__global__ void tangent_cost_kernel(const T *g_q, const T *g_qd,
                                    T *o_val, T *o_grad, T *o_hess_gn, T *o_hess_newton,
                                    T *o_xdes) {
    constexpr int TN = 2 * NV;
    constexpr int SCR = (NV + 294 > 3 * NV) ? (NV + 294) : (3 * NV);
    __shared__ T s_x[NX]; __shared__ T s_xdes[NX];
    __shared__ T s_Q2[TN]; __shared__ T s_delta[NV];
    __shared__ T s_scratch[SCR];
    __shared__ T s_out[1]; __shared__ T s_grad[TN];
    __shared__ T s_hgn[TN * TN]; __shared__ T s_hnw[TN * TN];
    for (int i = threadIdx.x; i < NQ; i += blockDim.x) s_x[i] = g_q[i];
    for (int i = threadIdx.x; i < NV; i += blockDim.x) s_x[NQ + i] = g_qd[i];
    for (int i = threadIdx.x; i < NV; i += blockDim.x)
        s_delta[i] = static_cast<T>(0.05) * static_cast<T>((i % 3) + 1) * ((i % 2) ? static_cast<T>(-1) : static_cast<T>(1));
    for (int i = threadIdx.x; i < TN; i += blockDim.x)
        s_Q2[i] = static_cast<T>(1) + static_cast<T>(0.5) * static_cast<T>(i);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        grid::grid_integrate_floating_q<T, NQ>(s_x, s_delta, s_xdes);
        for (int i = 0; i < NV; ++i) s_xdes[NQ + i] = s_x[NQ + i] + static_cast<T>(0.03) * static_cast<T>(i + 1);
    }
    __syncthreads();
    grid_plant::quadratic_state_cost_tangent<T>(s_out, s_x, s_xdes, s_Q2, s_scratch);
    __syncthreads();
    grid_plant::quadratic_state_cost_tangent_gradient<T>(s_grad, s_x, s_xdes, s_Q2, s_scratch);
    __syncthreads();
    grid_plant::quadratic_state_cost_tangent_hessian<T, false, true>(s_hgn, s_x, s_xdes, s_Q2, s_scratch);
    __syncthreads();
    grid_plant::quadratic_state_cost_tangent_hessian<T, false, false>(s_hnw, s_x, s_xdes, s_Q2, s_scratch);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) o_val[0] = s_out[0];
    for (int i = threadIdx.x; i < TN; i += blockDim.x) o_grad[i] = s_grad[i];
    for (int i = threadIdx.x; i < TN * TN; i += blockDim.x) { o_hess_gn[i] = s_hgn[i]; o_hess_newton[i] = s_hnw[i]; }
    for (int i = threadIdx.x; i < NX; i += blockDim.x) o_xdes[i] = s_xdes[i];
}
#endif  // GRID_PLANT_HAS_TANGENT_STATE_COST

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
    // R1 (2026-10-10): streams + only the generated host wrappers this runner
    // calls (integrator / integrator_gradient); the plant kernels below are
    // runner-local and register themselves. The "all" header's aggregate
    // init_grid<T>() compiled every kernel of the robot (go2: ~10 min).
    cudaStream_t *streams = grid::init_grid_streams<T>();
    gpuErrchk(grid::init_grid_kernel_attr_integrator_checked<T>());
    gpuErrchk(grid::init_grid_kernel_attr_integrator_gradient_checked<T>());
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
#if defined(GRID_PLANT_HAS_COM_COST)
    T *o_comv = dmalloc<T>(1), *o_comg = dmalloc<T>(NX), *o_comh = dmalloc<T>(NX * NX);
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
#if defined(GRID_PLANT_HAS_COM_COST)
    // com_cost composes grid::com_device, which uses an extern __shared__ dynamic arena.
    size_t cent_dyn = grid::COM_DYNAMIC_SHARED_MEM_BYTES<T>();
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

    // The centroidal cost kernel (com_cost) is independent of the
    // plant_step / integrator pass-through path, so drive it first — that way a
    // robot whose plant_step_kernel static scratch overflows the device smem cap
    // (e.g. go2:floating, where the big integrator-gradient static buffers exceed
    // the 48 KB default) still produces valid centroidal output. Compiled in only
    // when the centroidal macros are present; otherwise a `*_skipped` sentinel is
    // emitted below so the Python parser detects the absence cleanly.
#if defined(GRID_PLANT_HAS_COM_COST)
    plant_centroidal_kernel<T><<<1, nthreads, cent_dyn>>>(g_q, g_qd, d_robotModel,
        o_comv, o_comg, o_comh);
    gpuErrchkKernel();
#endif

#if defined(GRID_PLANT_HAS_TRACKING_COST)
    // tracking_cost preset == independent per-term composition (fixed-base recipe).
    // Reuses the EE-pose-gradient dynamic arena (same as plant_kernel).
    T *o_tpv = dmalloc<T>(1), *o_tpqk = dmalloc<T>(NX), *o_tprk = dmalloc<T>(NU);
    T *o_tpQk = dmalloc<T>(NX * NX), *o_tpRk = dmalloc<T>(NU * NU);
    T *o_trv = dmalloc<T>(1), *o_trqk = dmalloc<T>(NX), *o_trrk = dmalloc<T>(NU);
    T *o_trQk = dmalloc<T>(NX * NX), *o_trRk = dmalloc<T>(NU * NU);
    cudaFuncSetAttribute(tracking_preset_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)plant_dyn);
    tracking_preset_kernel<T><<<1, nthreads, plant_dyn>>>(g_q, g_qd, g_u, d_robotModel,
        o_tpv, o_tpqk, o_tprk, o_tpQk, o_tpRk, o_trv, o_trqk, o_trrk, o_trQk, o_trRk);
    gpuErrchkKernel();
#endif

#if defined(GRID_PLANT_HAS_COST_FRAMES)
    // Cost menu: activated terms, barrier activations, linear features, frame residuals (+ the preset on fixed base).
    T *o_mas = dmalloc<T>(MENU_STATE_CNT), *o_mai = dmalloc<T>(MENU_INPUT_CNT), *o_mae = dmalloc<T>(MENU_STATE_CNT);
    T *o_mlx = dmalloc<T>(MENU_STATE_CNT), *o_mlu = dmalloc<T>(MENU_INPUT_CNT), *o_mq = dmalloc<T>(2);
    T *o_mm = dmalloc<T>(2 * MENU_PRESET_CNT), *o_mg = dmalloc<T>(NV);
    T *o_mor = dmalloc<T>(MENU_STATE_CNT), *o_mvel = dmalloc<T>(MENU_STATE_CNT), *o_mf2 = dmalloc<T>(MENU_STATE_CNT);
    T *o_mrlb = dmalloc<T>(MENU_STATE_CNT), *o_mqb = dmalloc<T>(MENU_STATE_CNT);
    cudaMemset(o_mvel, 0, MENU_STATE_CNT * sizeof(T));
    size_t menu_dyn = plant_dyn;
    if (grid_plant::FRAME_EVAL_DYNAMIC_SHARED_MEM_BYTES<T>() > menu_dyn) menu_dyn = grid_plant::FRAME_EVAL_DYNAMIC_SHARED_MEM_BYTES<T>();
#if defined(GRID_PLANT_HAS_GRAVITY_TORQUE)
    if (grid_plant::GRAVITY_TORQUE_DYNAMIC_SHARED_MEM_BYTES<T>() > menu_dyn) menu_dyn = grid_plant::GRAVITY_TORQUE_DYNAMIC_SHARED_MEM_BYTES<T>();
#endif
    cudaFuncSetAttribute(cost_menu_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)menu_dyn);
    cost_menu_kernel<T><<<1, nthreads, menu_dyn>>>(g_q, g_qd, g_u, d_robotModel, gravity, o_mas, o_mai, o_mae, o_mlx, o_mlu, o_mq, o_mm, o_mg,
                                                  o_mor, o_mvel, o_mf2, o_mrlb, o_mqb);
    gpuErrchkKernel();
#endif

#ifdef GRID_PLANT_HAS_TANGENT_STATE_COST
    // Tangent (log-map) state cost — static __shared__ only, no dynamic arena.
    T *o_tcv = dmalloc<T>(1), *o_tcg = dmalloc<T>(2 * NV);
    T *o_tch_gn = dmalloc<T>(4 * NV * NV), *o_tch_nw = dmalloc<T>(4 * NV * NV);
    T *o_tc_xdes = dmalloc<T>(NX);
    tangent_cost_kernel<T><<<1, nthreads>>>(g_q, g_qd, o_tcv, o_tcg, o_tch_gn, o_tch_nw, o_tc_xdes);
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
#if defined(GRID_PLANT_HAS_COM_COST)
    dcopy_out("com_cost_value", o_comv, 1, 1);
    dcopy_out("com_cost_grad", o_comg, 1, NX);
    dcopy_out("com_cost_hess", o_comh, NX, NX);
#else
    // Centroidal costs are not emitted for this robot/config (com/ccrba absent,
    // e.g. a mimic robot). Emit a parseable sentinel so the Python centroidal test
    // pytest.skips this cell (mirrors the plant_step_skipped sentinel above).
    std::cout << "BEGIN com_cost_skipped 1 1\n1\nEND com_cost_skipped\n";
#endif
#if defined(GRID_PLANT_HAS_TRACKING_COST)
    dcopy_out("tracking_preset_value", o_tpv, 1, 1);
    dcopy_out("tracking_preset_qk", o_tpqk, 1, NX);
    dcopy_out("tracking_preset_rk", o_tprk, 1, NU);
    dcopy_out("tracking_preset_Qk", o_tpQk, NX, NX);
    dcopy_out("tracking_preset_Rk", o_tpRk, NU, NU);
    dcopy_out("tracking_ref_value", o_trv, 1, 1);
    dcopy_out("tracking_ref_qk", o_trqk, 1, NX);
    dcopy_out("tracking_ref_rk", o_trrk, 1, NU);
    dcopy_out("tracking_ref_Qk", o_trQk, NX, NX);
    dcopy_out("tracking_ref_Rk", o_trRk, NU, NU);
#else
    std::cout << "BEGIN tracking_preset_skipped 1 1\n1\nEND tracking_preset_skipped\n";
#endif
#if defined(GRID_PLANT_HAS_COST_FRAMES)
    dcopy_out("menu_state", o_mas, 1, MENU_STATE_CNT);
    dcopy_out("menu_input", o_mai, 1, MENU_INPUT_CNT);
    dcopy_out("menu_ee", o_mae, 1, MENU_STATE_CNT);
    dcopy_out("menu_linx", o_mlx, 1, MENU_STATE_CNT);
    dcopy_out("menu_linu", o_mlu, 1, MENU_INPUT_CNT);
    dcopy_out("menu_quad", o_mq, 1, 2);
    dcopy_out("menu_orient", o_mor, 1, MENU_STATE_CNT);
    dcopy_out("menu_frame2", o_mf2, 1, MENU_STATE_CNT);
    dcopy_out("menu_rlb_state", o_mrlb, 1, MENU_STATE_CNT);
    dcopy_out("menu_qb_state", o_mqb, 1, MENU_STATE_CNT);
#if defined(GRID_PLANT_HAS_GRAVITY_TORQUE)
    dcopy_out("menu_gravity", o_mg, 1, NV);
#endif
#if defined(GRID_PLANT_HAS_FRAME_VELOCITY_COST)
    dcopy_out("menu_vel", o_mvel, 1, MENU_STATE_CNT);
#endif
#if defined(GRID_PLANT_HAS_COST_MENU)
    dcopy_out("menu_preset", o_mm, 1, 2 * MENU_PRESET_CNT);
#else
    std::cout << "BEGIN menu_preset_skipped 1 1\n1\nEND menu_preset_skipped\n";
#endif
#else
    std::cout << "BEGIN menu_skipped 1 1\n1\nEND menu_skipped\n";
#endif
#ifdef GRID_PLANT_HAS_STEP_HESSIAN
    // Row-major flat (1 x D2AB_CNT); reshaped to (2*NV, 3*NV, 3*NV) C-order in Python.
    dcopy_out("plant_d2AB_euler", o_h2_eu, 1, D2AB_CNT);
    dcopy_out("plant_d2AB_si_euler", o_h2_si, 1, D2AB_CNT);
#endif
#ifdef GRID_PLANT_HAS_TANGENT_STATE_COST
    dcopy_out("tangent_cost_value", o_tcv, 1, 1);
    dcopy_out("tangent_cost_grad", o_tcg, 1, 2 * NV);
    dcopy_out("tangent_cost_hess_gn", o_tch_gn, 2 * NV, 2 * NV);
    dcopy_out("tangent_cost_hess_newton", o_tch_nw, 2 * NV, 2 * NV);
    dcopy_out("tangent_cost_x_des", o_tc_xdes, 1, NX);
#else
    // Fixed-base (or spherical-floating) robots do not emit the tangent preset;
    // parseable sentinel so the Python tangent test pytest.skips the cell.
    std::cout << "BEGIN tangent_cost_skipped 1 1\n1\nEND tangent_cost_skipped\n";
#endif

    grid::close_grid<T>(streams, d_robotModel, hd_data);
}

int main() {
    run<float>();
    return 0;
}

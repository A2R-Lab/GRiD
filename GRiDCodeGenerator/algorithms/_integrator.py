"""Time-integrator value codegen.

Mirrors `_forward_dynamics.py` (inner / device / kernel / host layers) but
emits a single time step `x_{k+1} = integrator(x_k, u_k, dt; f_dyn)` where
`f_dyn` is the existing forward dynamics. State is `x = [q (nq); qd (nv)]`
(fixed-base, so `nq == nv == n`), control `u` is size `n`, output `x_{k+1}`
is size `2n`. dt is a per-call scalar threaded through device/kernel/host.

The integrator type is selected at compile time by an `IntegratorType IT`
template parameter. Only `EULER` is wired up here; the `_dispatch` helper
below is structured so adding semi-implicit Euler / Midpoint / RK3 / RK4
later is purely additive (one extra `if constexpr (IT == ...)` branch).
"""


# integrator name <-> codegen-side string constant
_INTEGRATOR_TYPES = ("EULER", "SEMI_IMPLICIT_EULER", "MIDPOINT", "RK3", "RK4", "TRAPEZOIDAL")

# Number of forward-dynamics evaluations each integrator type requires.
# Used at codegen time to size shared-memory buffers (per-stage qdd) and to
# guide which stage-computation branches are emitted.
# TRAPEZOIDAL is single-stage (1 FD eval) like EULER, so _max_stages_in_use()
# stays == 4 -> per-stage scratch sizing is byte-identical and EULER/SI/RK
# kernels emit unchanged.
_STAGE_COUNT = {
    "EULER": 1,
    "SEMI_IMPLICIT_EULER": 1,
    "MIDPOINT": 2,
    "RK3": 3,
    "RK4": 4,
    "TRAPEZOIDAL": 1,
}


def _max_stages_in_use():
    """Maximum stage count among all currently-emitted integrator types.

    For now, the kernel statically allocates per-stage scratch sized for the
    most-expensive integrator (RK4). This keeps shared-memory layout simple
    and the cost is small (a handful of extra n-sized buffers).
    """
    return max(_STAGE_COUNT.values())


def _integrator_type_token(integrator_type):
    """Either a known enum value (compile-time enum) or a raw template
    parameter passthrough (e.g. "IT" inside a `template <..., IntegratorType IT>`
    scope)."""
    if integrator_type in _INTEGRATOR_TYPES:
        return "IntegratorType::" + integrator_type
    return integrator_type


def gen_integrator_inner_temp_mem_size(self, minv_f_in_smem=True):
    # Integrator's only extra scratch is the FD itself; the assembly step is
    # in-place over a parallel loop with no additional storage. The surgical
    # Minv-F lever is forwarded straight through to the FD inner: when
    # minv_f_in_smem the FD's 6*NV*NV F-region is sized into s_temp here; when
    # spilled it lives in d_workspace and is excluded (callers size per
    # placement via INTEGRATOR_INNER_{SMEM,WORKSPACE}_BYTES<T, MINV_F_IN_SMEM>).
    return self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=minv_f_in_smem)


def gen_lie_group_helpers(self):
    """Emit __device__ helpers for SE(3) Lie-group integration. Only used by
    floating-base codegen. Mirrors the Python implementations in
    RBDReference (`_quat_mul_xyzw`, `_quat_exp_from_half_omega`,
    `_rotation_from_quat_xyzw`, `_so3_skew`, `_so3_V_matrix`,
    `_so3_right_jacobian`, `_so3_exp`, `_se3_Q_block`, `integrate` for the
    free-flyer prefix, and the 6x6 dIntegrate Adjoint / right-Jacobian
    blocks). xyzw quaternion convention; v_dt = [v_lin*dt; omega*dt] in
    Pinocchio order (linear first), body frame.
    """
    self.gen_add_func_doc("Floating-base Lie-group helpers (xyzw quaternion, Pinocchio v order).", [], [], None)
    self.gen_add_code_lines([
        # ---- quaternion multiply (xyzw) ----
        "template <typename T> __device__ inline void grid_quat_mul_xyzw(const T a[4], const T b[4], T out[4]) {",
        "    out[0] = a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1];",
        "    out[1] = a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0];",
        "    out[2] = a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3];",
        "    out[3] = a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2];",
        "}",
        "",
        # ---- quaternion exponential from half-omega ----
        "template <typename T> __device__ inline void grid_quat_exp_half_omega(const T half_omega[3], T out[4]) {",
        "    T theta = sqrt(half_omega[0]*half_omega[0] + half_omega[1]*half_omega[1] + half_omega[2]*half_omega[2]);",
        "    T sinc, cos_t;",
        "    if (theta < static_cast<T>(1e-12)) { sinc = static_cast<T>(1) - theta*theta/static_cast<T>(6); cos_t = static_cast<T>(1) - static_cast<T>(0.5)*theta*theta; }",
        "    else { sinc = sin(theta)/theta; cos_t = cos(theta); }",
        "    out[0] = sinc * half_omega[0]; out[1] = sinc * half_omega[1]; out[2] = sinc * half_omega[2]; out[3] = cos_t;",
        "}",
        "",
        # ---- rotation matrix from xyzw quaternion ----
        "template <typename T> __device__ inline void grid_rot_from_quat_xyzw(const T q[4], T R[9]) {",
        "    T x=q[0], y=q[1], z=q[2], w=q[3];",
        "    T xx=x*x, yy=y*y, zz=z*z;",
        "    T xy=x*y, xz=x*z, yz=y*z;",
        "    T wx=w*x, wy=w*y, wz=w*z;",
        "    // row-major 3x3",
        "    R[0]=static_cast<T>(1)-static_cast<T>(2)*(yy+zz); R[1]=static_cast<T>(2)*(xy-wz);     R[2]=static_cast<T>(2)*(xz+wy);",
        "    R[3]=static_cast<T>(2)*(xy+wz);     R[4]=static_cast<T>(1)-static_cast<T>(2)*(xx+zz); R[5]=static_cast<T>(2)*(yz-wx);",
        "    R[6]=static_cast<T>(2)*(xz-wy);     R[7]=static_cast<T>(2)*(yz+wx);     R[8]=static_cast<T>(1)-static_cast<T>(2)*(xx+yy);",
        "}",
        "",
        # ---- 3x3 skew-symmetric matrix from 3-vector (row-major) ----
        "template <typename T> __device__ inline void grid_so3_skew(const T v[3], T S[9]) {",
        "    S[0]=static_cast<T>(0); S[1]=-v[2];               S[2]=v[1];",
        "    S[3]=v[2];               S[4]=static_cast<T>(0); S[5]=-v[0];",
        "    S[6]=-v[1];              S[7]=v[0];               S[8]=static_cast<T>(0);",
        "}",
        "",
        # ---- 3x3 matmul (row-major) ----
        "template <typename T> __device__ inline void grid_mat3_mul(const T A[9], const T B[9], T C[9]) {",
        "    #pragma unroll",
        "    for (int i = 0; i < 3; ++i) {",
        "        #pragma unroll",
        "        for (int j = 0; j < 3; ++j) {",
        "            T s = static_cast<T>(0);",
        "            #pragma unroll",
        "            for (int k = 0; k < 3; ++k) s += A[3*i+k] * B[3*k+j];",
        "            C[3*i+j] = s;",
        "        }",
        "    }",
        "}",
        "",
        # ---- 3x3 matrix-vector ----
        "template <typename T> __device__ inline void grid_mat3_vec(const T A[9], const T v[3], T out[3]) {",
        "    out[0] = A[0]*v[0] + A[1]*v[1] + A[2]*v[2];",
        "    out[1] = A[3]*v[0] + A[4]*v[1] + A[5]*v[2];",
        "    out[2] = A[6]*v[0] + A[7]*v[1] + A[8]*v[2];",
        "}",
        "",
        # ---- SE(3) V matrix:  p_delta = V(phi) @ rho ----
        "template <typename T> __device__ inline void grid_so3_V_matrix(const T phi[3], T V[9]) {",
        "    T S[9]; grid_so3_skew(phi, S);",
        "    T S2[9]; grid_mat3_mul(S, S, S2);",
        "    T theta = sqrt(phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2]);",
        "    T a, b;",
        "    if (theta < static_cast<T>(1e-8)) { a = static_cast<T>(0.5); b = static_cast<T>(1.0/6.0); }",
        "    else { a = (static_cast<T>(1) - cos(theta)) / (theta*theta); b = (theta - sin(theta)) / (theta*theta*theta); }",
        "    #pragma unroll",
        "    for (int i = 0; i < 9; ++i) V[i] = (i % 4 == 0 ? static_cast<T>(1) : static_cast<T>(0)) + a*S[i] + b*S2[i];",
        "}",
        "",
        # ---- SO(3) right Jacobian J_r(phi) ----
        "template <typename T> __device__ inline void grid_so3_right_jacobian(const T phi[3], T J[9]) {",
        "    T S[9]; grid_so3_skew(phi, S);",
        "    T S2[9]; grid_mat3_mul(S, S, S2);",
        "    T theta = sqrt(phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2]);",
        "    T a, b;",
        "    if (theta < static_cast<T>(1e-8)) { a = static_cast<T>(0.5); b = static_cast<T>(1.0/6.0); }",
        "    else { a = (static_cast<T>(1) - cos(theta)) / (theta*theta); b = (theta - sin(theta)) / (theta*theta*theta); }",
        "    #pragma unroll",
        "    for (int i = 0; i < 9; ++i) J[i] = (i % 4 == 0 ? static_cast<T>(1) : static_cast<T>(0)) - a*S[i] + b*S2[i];",
        "}",
        "",
        # ---- SO(3) exponential: R = exp([phi]_x) ----
        "template <typename T> __device__ inline void grid_so3_exp(const T phi[3], T R[9]) {",
        "    T S[9]; grid_so3_skew(phi, S);",
        "    T S2[9]; grid_mat3_mul(S, S, S2);",
        "    T theta = sqrt(phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2]);",
        "    T a, b;",
        "    if (theta < static_cast<T>(1e-8)) { a = static_cast<T>(1); b = static_cast<T>(0.5); }",
        "    else { a = sin(theta) / theta; b = (static_cast<T>(1) - cos(theta)) / (theta*theta); }",
        "    #pragma unroll",
        "    for (int i = 0; i < 9; ++i) R[i] = (i % 4 == 0 ? static_cast<T>(1) : static_cast<T>(0)) + a*S[i] + b*S2[i];",
        "}",
        "",
        # ---- SE(3) Q coupling block (matches Pinocchio sign convention; see RBDReference._se3_Q_block) ----
        "template <typename T> __device__ inline void grid_se3_Q_block(const T rho[3], const T phi[3], T Q[9]) {",
        "    T phi_neg[3] = {-phi[0], -phi[1], -phi[2]};",
        "    T Px[9]; grid_so3_skew(phi_neg, Px);",
        "    T Rx[9]; grid_so3_skew(rho, Rx);",
        "    T Px2[9]; grid_mat3_mul(Px, Px, Px2);",
        "    T Rx_Px[9]; grid_mat3_mul(Rx, Px, Rx_Px);",
        "    T Px_Rx[9]; grid_mat3_mul(Px, Rx, Px_Rx);",
        "    T Px_Rx_Px[9]; grid_mat3_mul(Px, Rx_Px, Px_Rx_Px);",
        "    T Px2_Rx[9]; grid_mat3_mul(Px2, Rx, Px2_Rx);",
        "    T Rx_Px2[9]; grid_mat3_mul(Rx_Px, Px, Rx_Px2);  // Rx@Px@Px",
        "    T Px_Rx_Px2[9]; grid_mat3_mul(Px_Rx, Px2, Px_Rx_Px2);  // Px@Rx@Px@Px",
        "    T Px2_Rx_Px[9]; grid_mat3_mul(Px2, Rx_Px, Px2_Rx_Px);  // Px@Px@Rx@Px",
        "    T theta_neg = sqrt(phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2]);",
        "    T sola[9];",
        "    if (theta_neg < static_cast<T>(1e-4)) {",
        "        #pragma unroll",
        "        for (int i = 0; i < 9; ++i)",
        "            sola[i] = static_cast<T>(0.5)*Rx[i]",
        "                    + static_cast<T>(1.0/6.0)*(Px_Rx[i] + Rx_Px[i] + Px_Rx_Px[i])",
        "                    - static_cast<T>(1.0/24.0)*(Px2_Rx[i] + Rx_Px2[i] - static_cast<T>(3)*Px_Rx_Px[i]);",
        "    } else {",
        "        T c1 = (theta_neg - sin(theta_neg)) / (theta_neg*theta_neg*theta_neg);",
        "        T c2 = (static_cast<T>(1) - static_cast<T>(0.5)*theta_neg*theta_neg - cos(theta_neg)) / (theta_neg*theta_neg*theta_neg*theta_neg);",
        "        // Negative sign matches Barfoot's SE(3) Q-block coefficient (2θ−3sinθ+θcosθ)/(2θ⁵); verified against pin.dIntegrate(ARG1).",
        "        T c3 = static_cast<T>(-0.5) * (c2 - static_cast<T>(3) * (theta_neg - sin(theta_neg) - theta_neg*theta_neg*theta_neg/static_cast<T>(6)) / (theta_neg*theta_neg*theta_neg*theta_neg*theta_neg));",
        "        #pragma unroll",
        "        for (int i = 0; i < 9; ++i)",
        "            sola[i] = static_cast<T>(0.5)*Rx[i]",
        "                    + c1*(Px_Rx[i] + Rx_Px[i] + Px_Rx_Px[i])",
        "                    - c2*(Px2_Rx[i] + Rx_Px2[i] - static_cast<T>(3)*Px_Rx_Px[i])",
        "                    + c3*(Px_Rx_Px2[i] + Px2_Rx_Px[i]);",
        "    }",
        "    // Pinocchio convention: Q = -Sola(rho, -phi).",
        "    #pragma unroll",
        "    for (int i = 0; i < 9; ++i) Q[i] = -sola[i];",
        "}",
        "",
        # ---- Lie-group q update: q_new <- integrate(q, v_dt) on the floating-base prefix ----
        # Input q is the FULL nq layout [x,y,z, qx,qy,qz,qw, joints...].
        # Input v_dt is in INTERNAL GRiD order [omega(3); v_lin(3); joint_v_dt...]
        # (matches s_qd layout). We swap to Pinocchio order [v_lin; omega]
        # inside the helper before the SE(3) exp, then write the result back
        # into the project's q layout.
        "template <typename T, int NUM_POS> __device__ inline void grid_integrate_floating_q(",
        "    const T *q, const T *v_dt, T *q_new) {",
        "    // Pinocchio user-facing v_dt order: [v_lin; omega; joints].",
        "    // (Matches RBDReference.integrate; same convention used through the",
        "    //  forward_dynamics / aba / etc. CUDA kernels.)",
        "    T rho[3]      = {v_dt[0], v_dt[1], v_dt[2]};",
        "    T omega_dt[3] = {v_dt[3], v_dt[4], v_dt[5]};",
        "    T half[3] = {static_cast<T>(0.5)*omega_dt[0], static_cast<T>(0.5)*omega_dt[1], static_cast<T>(0.5)*omega_dt[2]};",
        "    T dq[4]; grid_quat_exp_half_omega(half, dq);",
        "    T q_old_quat[4] = {q[3], q[4], q[5], q[6]};",
        "    T q_new_quat[4]; grid_quat_mul_xyzw(q_old_quat, dq, q_new_quat);",
        "    // renormalize",
        "    T qn = sqrt(q_new_quat[0]*q_new_quat[0] + q_new_quat[1]*q_new_quat[1] + q_new_quat[2]*q_new_quat[2] + q_new_quat[3]*q_new_quat[3]);",
        "    T inv_qn = static_cast<T>(1) / qn;",
        "    #pragma unroll",
        "    for (int i = 0; i < 4; ++i) q_new[3 + i] = q_new_quat[i] * inv_qn;",
        "    T V[9]; grid_so3_V_matrix(omega_dt, V);",
        "    T p_delta_local[3]; grid_mat3_vec(V, rho, p_delta_local);",
        "    T R_old[9]; grid_rot_from_quat_xyzw(q_old_quat, R_old);",
        "    T p_delta_world[3]; grid_mat3_vec(R_old, p_delta_local, p_delta_world);",
        "    q_new[0] = q[0] + p_delta_world[0];",
        "    q_new[1] = q[1] + p_delta_world[1];",
        "    q_new[2] = q[2] + p_delta_world[2];",
        "    // remaining (revolute joints): Euler add. v_dt[6:nv] -> q[7:nq].",
        "    // (n_joints = NUM_POS - 7 = nv - 6)",
        "    #pragma unroll",
        "    for (int i = 0; i < (NUM_POS - 7); ++i) q_new[7 + i] = q[7 + i] + v_dt[6 + i];",
        "}",
        "",
        # ---- dIntegrate top-left 6x6 block for ARG_q (SE(3) Adjoint of exp(-v_dt)) ----
        # Written in PINOCCHIO order [v_lin; omega] (matches pin.dIntegrate output).
        "template <typename T> __device__ inline void grid_dIntegrate_q_block(const T *v_dt, T J[36]) {",
        "    // v_dt in Pinocchio user-facing order: [v_lin; omega].",
        "    T rho[3]   = {v_dt[0], v_dt[1], v_dt[2]};",
        "    T omega[3] = {v_dt[3], v_dt[4], v_dt[5]};",
        "    T R_inv[9]; T omega_neg[3] = {-omega[0], -omega[1], -omega[2]};",
        "    grid_so3_exp(omega_neg, R_inv);",
        "    T V_neg[9]; grid_so3_V_matrix(omega_neg, V_neg);",
        "    T V_neg_rho[3]; grid_mat3_vec(V_neg, rho, V_neg_rho);",
        "    T p_inv[3] = {-V_neg_rho[0], -V_neg_rho[1], -V_neg_rho[2]};",
        "    T P_inv_x[9]; grid_so3_skew(p_inv, P_inv_x);",
        "    T off[9]; grid_mat3_mul(P_inv_x, R_inv, off);",
        "    // 6x6 block in row-major: [[R_inv, off], [0, R_inv]]  (Pinocchio order [v_lin; omega])",
        "    #pragma unroll",
        "    for (int i = 0; i < 36; ++i) J[i] = static_cast<T>(0);",
        "    #pragma unroll",
        "    for (int i = 0; i < 3; ++i) {",
        "        #pragma unroll",
        "        for (int j = 0; j < 3; ++j) {",
        "            J[6*i + j]     = R_inv[3*i + j];",
        "            J[6*i + 3 + j] = off[3*i + j];",
        "            J[6*(3+i) + 3 + j] = R_inv[3*i + j];",
        "        }",
        "    }",
        "}",
        "",
        # ---- dIntegrate top-left 6x6 block for ARG_v (SE(3) right Jacobian) ----
        "template <typename T> __device__ inline void grid_dIntegrate_v_block(const T *v_dt, T J[36]) {",
        "    // v_dt in Pinocchio user-facing order: [v_lin; omega].",
        "    T rho[3]   = {v_dt[0], v_dt[1], v_dt[2]};",
        "    T omega[3] = {v_dt[3], v_dt[4], v_dt[5]};",
        "    T Jr[9]; grid_so3_right_jacobian(omega, Jr);",
        "    T Q[9]; grid_se3_Q_block(rho, omega, Q);",
        "    #pragma unroll",
        "    for (int i = 0; i < 36; ++i) J[i] = static_cast<T>(0);",
        "    #pragma unroll",
        "    for (int i = 0; i < 3; ++i) {",
        "        #pragma unroll",
        "        for (int j = 0; j < 3; ++j) {",
        "            J[6*i + j]     = Jr[3*i + j];",
        "            J[6*i + 3 + j] = Q[3*i + j];",
        "            J[6*(3+i) + 3 + j] = Jr[3*i + j];",
        "        }",
        "    }",
        "}",
        "",
        # ---- second-order dIntegrate: d2Int[o,j,k] = d(dInt_block[o,j])/d(w[k]) ----
        # 4th-order central finite difference of the 6x6 SE(3) dIntegrate blocks over
        # the 6 free-flyer increment directions k. Done in DOUBLE regardless of the
        # caller's T (the blocks are tiny and double keeps a float32 kernel matching
        # the float64 oracle; a float32 FD here would be too noisy). h matches the
        # RBDReference.d2Integrate stencil (1e-3, near the 4th-order roundoff sweet
        # spot, clear of dIntegrate's small-angle cliff). Output tensor [o*36 + j*6 + k].
        "template <typename T, bool IS_Q> __device__ inline void grid_d2Integrate_block(const T *w, T J2[216]) {",
        "    const double h = 1e-3;",
        "    double wd[6]; for (int m = 0; m < 6; ++m) wd[m] = static_cast<double>(w[m]);",
        "    for (int k = 0; k < 6; ++k) {",
        "        double Jp1[36], Jm1[36], Jp2[36], Jm2[36];",
        "        double wp[6];",
        "        #pragma unroll",
        "        for (int m = 0; m < 6; ++m) wp[m] = wd[m]; wp[k] = wd[k] + h;",
        "        if constexpr (IS_Q) grid_dIntegrate_q_block<double>(wp, Jp1); else grid_dIntegrate_v_block<double>(wp, Jp1);",
        "        #pragma unroll",
        "        for (int m = 0; m < 6; ++m) wp[m] = wd[m]; wp[k] = wd[k] - h;",
        "        if constexpr (IS_Q) grid_dIntegrate_q_block<double>(wp, Jm1); else grid_dIntegrate_v_block<double>(wp, Jm1);",
        "        #pragma unroll",
        "        for (int m = 0; m < 6; ++m) wp[m] = wd[m]; wp[k] = wd[k] + 2.0*h;",
        "        if constexpr (IS_Q) grid_dIntegrate_q_block<double>(wp, Jp2); else grid_dIntegrate_v_block<double>(wp, Jp2);",
        "        #pragma unroll",
        "        for (int m = 0; m < 6; ++m) wp[m] = wd[m]; wp[k] = wd[k] - 2.0*h;",
        "        if constexpr (IS_Q) grid_dIntegrate_q_block<double>(wp, Jm2); else grid_dIntegrate_v_block<double>(wp, Jm2);",
        "        #pragma unroll",
        "        for (int oj = 0; oj < 36; ++oj) {",
        "            int o = oj / 6, j = oj % 6;",
        "            double d = (8.0*(Jp1[oj] - Jm1[oj]) - (Jp2[oj] - Jm2[oj])) / (12.0*h);",
        "            J2[o*36 + j*6 + k] = static_cast<T>(d);",
        "        }",
        "    }",
        "}",
        "",
    ])
    # Spherical (ball) joint SO(3) retract helper — emitted ONLY when the robot
    # has a spherical joint (so pure-floating robots stay byte-identical; the
    # block is absent from their header). Block-pointer form: q_new_blk =
    # normalize(q_blk (x) exp(half)). `half` is 0.5*scale*omega (the caller
    # pre-scales). Reuses grid_quat_exp_half_omega + grid_quat_mul_xyzw + the
    # renorm — the SO(3) half of grid_integrate_floating_q with no SE(3) coupling.
    if self.robot.robot_has_spherical():
        self.gen_add_code_lines([
            "template <typename T> __device__ inline void grid_integrate_spherical_q(",
            "    const T *q_blk, const T *half_omega, T *q_new_blk) {",
            "    T dq[4]; grid_quat_exp_half_omega(half_omega, dq);",
            "    T q_old_quat[4] = {q_blk[0], q_blk[1], q_blk[2], q_blk[3]};",
            "    T q_new_quat[4]; grid_quat_mul_xyzw(q_old_quat, dq, q_new_quat);",
            "    T qn = sqrt(q_new_quat[0]*q_new_quat[0] + q_new_quat[1]*q_new_quat[1] + q_new_quat[2]*q_new_quat[2] + q_new_quat[3]*q_new_quat[3]);",
            "    T inv_qn = static_cast<T>(1) / qn;",
            "    #pragma unroll",
            "    for (int i = 0; i < 4; ++i) q_new_blk[i] = q_new_quat[i] * inv_qn;",
            "}",
            "",
        ])


def gen_integrate_spherical_helper(self):
    """Standalone emitter for the spherical SO(3) retract device helper, used
    when the robot has a spherical joint but is NOT floating (so the floating
    Lie-group helper bundle isn't otherwise emitted). Emits the small quaternion
    primitives it depends on (grid_quat_mul_xyzw, grid_quat_exp_half_omega) plus
    the grid_integrate_spherical_q wrapper. Pure-floating robots emit these via
    gen_lie_group_helpers instead (and never call this)."""
    self.gen_add_func_doc("Spherical (ball) joint SO(3) quaternion retract helper (xyzw).", [], [], None)
    self.gen_add_code_lines([
        "template <typename T> __device__ inline void grid_quat_mul_xyzw(const T a[4], const T b[4], T out[4]) {",
        "    out[0] = a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1];",
        "    out[1] = a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0];",
        "    out[2] = a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3];",
        "    out[3] = a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2];",
        "}",
        "",
        "template <typename T> __device__ inline void grid_quat_exp_half_omega(const T half_omega[3], T out[4]) {",
        "    T theta = sqrt(half_omega[0]*half_omega[0] + half_omega[1]*half_omega[1] + half_omega[2]*half_omega[2]);",
        "    T sinc, cos_t;",
        "    if (theta < static_cast<T>(1e-12)) { sinc = static_cast<T>(1) - theta*theta/static_cast<T>(6); cos_t = static_cast<T>(1) - static_cast<T>(0.5)*theta*theta; }",
        "    else { sinc = sin(theta)/theta; cos_t = cos(theta); }",
        "    out[0] = sinc * half_omega[0]; out[1] = sinc * half_omega[1]; out[2] = sinc * half_omega[2]; out[3] = cos_t;",
        "}",
        "",
        "template <typename T> __device__ inline void grid_integrate_spherical_q(",
        "    const T *q_blk, const T *half_omega, T *q_new_blk) {",
        "    T dq[4]; grid_quat_exp_half_omega(half_omega, dq);",
        "    T q_old_quat[4] = {q_blk[0], q_blk[1], q_blk[2], q_blk[3]};",
        "    T q_new_quat[4]; grid_quat_mul_xyzw(q_old_quat, dq, q_new_quat);",
        "    T qn = sqrt(q_new_quat[0]*q_new_quat[0] + q_new_quat[1]*q_new_quat[1] + q_new_quat[2]*q_new_quat[2] + q_new_quat[3]*q_new_quat[3]);",
        "    T inv_qn = static_cast<T>(1) / qn;",
        "    #pragma unroll",
        "    for (int i = 0; i < 4; ++i) q_new_blk[i] = q_new_quat[i] * inv_qn;",
        "}",
        "",
    ])


def _spherical_retract_index_tables(self):
    """Return (add_q, add_v, spherical_blocks) for the q-update on a robot that
    has spherical joints (fixed-base; spherical robots are not floating here).

    - add_q / add_v : matched index lists for the NON-spherical joint q/v slots
      that retract by plain vector add (s_x_kp1[add_q[i]] = s_q[add_q[i]] +
      scale*s_src_v[add_v[i]]). Built from get_joint_index_q/v so every slot
      DOWNSTREAM of a spherical joint gets the correct shifted q-offset (§1e).
    - spherical_blocks : list of (q4, v3) index lists, one per spherical joint,
      each driving an SO(3) quaternion retract via grid_integrate_spherical_q.
    """
    add_q = []
    add_v = []
    spherical_blocks = []
    for joint in self.robot.get_joints_ordered_by_id():
        jid = joint.get_id()
        jtype = getattr(joint, "jtype", None)
        iq = self.robot.get_joint_index_q(jid)
        iv = self.robot.get_joint_index_v(jid)
        iq = list(iq) if isinstance(iq, (list, tuple)) else [iq]
        iv = list(iv) if isinstance(iv, (list, tuple)) else [iv]
        if jtype == "spherical" and not getattr(joint, "is_mimic", False):
            spherical_blocks.append((iq, iv))
        else:
            # plain vector-add joint(s): pair q-slots with v-slots 1:1.
            for qi, vi in zip(iq, iv):
                add_q.append(qi)
                add_v.append(vi)
    return add_q, add_v, spherical_blocks


def _emit_q_update(self, scale_expr, dst_name, src_q_name="s_q", src_v_name="s_src_v",
                   cardinal_line=None, accel_name=None, accel_scale_expr=None):
    """Emit the q-side update q_new = q (+) scale*src_v for one stage, branching
    on base type:
      - fb            : SE(3) Lie retract (grid_integrate_floating_q), verbatim.
      - spherical     : baked additive index-table parallel loop for the
                        non-spherical joint slots + a serial SO(3) quaternion
                        retract per spherical joint (grid_integrate_spherical_q).
      - else (cardinal): plain parallel Euler add over nq positions, verbatim.
    `scale_expr` is the C++ scalar multiplying src_v (e.g. "dt", "c1 * dt").
    `cardinal_line` (optional) is the EXACT cardinal-branch loop-body line to
    emit; supplied by callers that need to preserve the historical column
    alignment so cardinal-robot codegen stays byte-identical. Defaults to the
    canonical single-space form when not given.

    `accel_name` / `accel_scale_expr` (optional, TRAPEZOIDAL): fold an extra
    `accel_scale_expr * accel_name[v]` term into the retracted tangent at every
    v-index, so the effective tangent is `scale*src_v + accel_scale*accel`. This
    lets TRAPEZOIDAL retract `dt*qd + 0.5*dt^2*qdd` in ONE step — correct for
    fixed-base (collapses to the in-place add), floating-base (single SE(3) Lie
    retract of the combined tangent) AND spherical (the SO(3) half-angle uses the
    combined angular velocity). When `accel_name is None` the emitted code is
    byte-identical to the historical single-term form (EULER/SI/RK unaffected).
    """
    nv = self.robot.get_num_vel()
    nq = self.robot.get_num_pos()
    fb = self.robot.floating_base
    # Per-v-index accel suffix folded into the tangent (empty when no accel term).
    def _acc(vidx):
        if accel_name is None:
            return ""
        return f" + {accel_scale_expr} * {accel_name}[{vidx}]"
    if fb:
        self.gen_add_serial_ops()
        self.gen_add_code_line(f"T v_scaled[{nv}];")
        self.gen_add_code_line(f"for (int i = 0; i < {nv}; ++i) v_scaled[i] = {scale_expr} * {src_v_name}[i]{_acc('i')};")
        self.gen_add_code_line(f"grid_integrate_floating_q<T, {nq}>({src_q_name}, v_scaled, {dst_name});")
        self.gen_add_end_control_flow()
    elif self.robot.robot_has_spherical():
        add_q, add_v, spherical_blocks = self._spherical_retract_index_tables()
        # Non-spherical joint slots: baked additive index tables (downstream-of-
        # spherical q-offsets are already shifted by get_joint_index_q). Parallel.
        n_add = len(add_q)
        if n_add:
            # Wrap in an explicit C++ block so the baked add_q/add_v tables are
            # scoped: integrator_inner emits several q-update sites (stages 2-4 +
            # final assembly) into ONE function scope, so unscoped decls collide.
            self.gen_add_code_line("{", True)
            self.gen_add_code_line(
                "const int add_q[" + str(n_add) + "] = {" + ", ".join(str(i) for i in add_q) + "};")
            self.gen_add_code_line(
                "const int add_v[" + str(n_add) + "] = {" + ", ".join(str(i) for i in add_v) + "};")
            self.gen_add_parallel_loop("ind", str(n_add))
            self.gen_add_code_line(
                f"{dst_name}[add_q[ind]] = {src_q_name}[add_q[ind]] + {scale_expr} * {src_v_name}[add_v[ind]]{_acc('add_v[ind]')};")
            self.gen_add_end_control_flow()
            self.gen_add_end_control_flow()
        # Spherical joints: serial SO(3) quaternion retract (one thread).
        self.gen_add_serial_ops()
        for blk_i, (q4, v3) in enumerate(spherical_blocks):
            self.gen_add_code_line(
                "const int sph_q_" + str(blk_i) + "[4] = {" + ", ".join(str(i) for i in q4) + "};")
            self.gen_add_code_line(
                "const int sph_v_" + str(blk_i) + "[3] = {" + ", ".join(str(i) for i in v3) + "};")
            self.gen_add_code_line(f"T sph_qblk_{blk_i}[4] = {{"
                                   f"{src_q_name}[sph_q_{blk_i}[0]], {src_q_name}[sph_q_{blk_i}[1]], "
                                   f"{src_q_name}[sph_q_{blk_i}[2]], {src_q_name}[sph_q_{blk_i}[3]]}};")
            # half = 0.5 * (scale * omega [+ accel_scale * alpha])  (combined angular tangent)
            def _sph_half(k):
                if accel_name is None:
                    return f"static_cast<T>(0.5)*{scale_expr}*{src_v_name}[sph_v_{blk_i}[{k}]]"
                return (f"static_cast<T>(0.5)*({scale_expr}*{src_v_name}[sph_v_{blk_i}[{k}]]"
                        f" + {accel_scale_expr}*{accel_name}[sph_v_{blk_i}[{k}]])")
            self.gen_add_code_line(f"T sph_half_{blk_i}[3] = {{"
                                   f"{_sph_half(0)}, "
                                   f"{_sph_half(1)}, "
                                   f"{_sph_half(2)}}};")
            self.gen_add_code_line(f"T sph_qnew_{blk_i}[4];")
            self.gen_add_code_line(
                f"grid_integrate_spherical_q<T>(sph_qblk_{blk_i}, sph_half_{blk_i}, sph_qnew_{blk_i});")
            self.gen_add_code_line("#pragma unroll")
            self.gen_add_code_line(
                f"for (int i = 0; i < 4; ++i) {dst_name}[sph_q_{blk_i}[i]] = sph_qnew_{blk_i}[i];")
        self.gen_add_end_control_flow()
    else:
        self.gen_add_parallel_loop("ind", str(nq))
        if cardinal_line is None:
            cardinal_line = f"{dst_name}[ind] = {src_q_name}[ind] + {scale_expr} * {src_v_name}[ind]{_acc('ind')};"
        self.gen_add_code_line(cardinal_line)
        self.gen_add_end_control_flow()


def gen_integrator_finish_function_call(self, integrator_type="IT", updated_var_names=None):
    var_names = dict(
        s_x_kp1_name="s_x_kp1",
        s_q_name="s_q",
        s_qd_name="s_qd",
        s_qdd_name="s_qdd",
        dt_name="dt",
    )
    if updated_var_names is not None:
        for key, value in updated_var_names.items():
            var_names[key] = value
    code = ("integrator_finish<T, " + _integrator_type_token(integrator_type) + ">(" +
            var_names["s_x_kp1_name"] + ", " +
            var_names["s_q_name"] + ", " +
            var_names["s_qd_name"] + ", " +
            var_names["s_qdd_name"] + ", " +
            var_names["dt_name"] + ");")
    self.gen_add_code_line(code)


def gen_integrator_finish(self):
    """Emit a templated `integrator_finish<T, IntegratorType IT>` device function.

    For EULER:
        x_{k+1}[i]    = q[i]  + dt * qd[i]    for i in [0, n)     // q + dt*qd
        x_{k+1}[n+i]  = qd[i] + dt * qdd[i]   for i in [0, n)     // qd + dt*qdd
    Assumes the underlying forward dynamics has already populated s_qdd.
    """
    nv = self.robot.get_num_vel()
    nq = self.robot.get_num_pos()
    fb = self.robot.floating_base
    n_joints = nv - 6 if fb else nv  # revolute joint count (free-flyer adds 6)
    func_params = ["s_x_kp1 is a pointer to memory for the next state (size NUM_POS + NUM_VEL)",
                   "s_q is the vector of joint positions (size NUM_POS)",
                   "s_qd is the vector of joint velocities (size NUM_VEL)",
                   "s_qdd is the vector of joint accelerations (size NUM_VEL, output of forward_dynamics)",
                   "dt is the integration timestep"]
    func_def = "void integrator_finish(T *s_x_kp1, const T *s_q, const T *s_qd, const T *s_qdd, const T dt) {"
    func_notes = ["Assumes s_qdd is already computed for the current (s_q, s_qd, s_u)",
                  "Floating-base: q-update uses an SE(3) Lie-group retract (grid_integrate_floating_q)",
                  "Does not internally sync the thread group, so it should be called after all threads have finished computing their values"]
    self.gen_add_func_doc("Finish the integrator step: write x_{k+1} from (q, qd, qdd) per the integrator type",
                          func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, IntegratorType IT>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    # ---- v_{k+1} part — always Euler-style: v_new = qd + dt*qdd (size nv) ----
    self.gen_add_parallel_loop("ind", str(nv))
    self.gen_add_code_line(f"s_x_kp1[{nq} + ind] = s_qd[ind] + dt * s_qdd[ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # ---- q_{k+1} part — Euler uses qd; SI Euler uses v_new ----
    # For SI-Euler, the v_new computed above is the integration source. Read
    # it back from s_x_kp1[nq:nq+nv] when IT == SEMI_IMPLICIT_EULER.
    # q-update by integrator type. EULER retracts dt*qd; SI-EULER retracts dt*v_new;
    # TRAPEZOIDAL retracts the COMBINED tangent dt*qd + 0.5*dt^2*qdd in ONE step
    # (GATO integrator.cuh:36 -> q_next = q + dt*qd + 0.5*dt^2*qdd, reading OLD qd).
    # Folding the accel into the single retract (via _emit_q_update's accel term) is
    # correct for fixed-base (collapses to the in-place add), floating-base (SE(3)
    # Lie retract of the combined tangent) AND spherical (SO(3) half uses the
    # combined angular velocity) — no in-place add onto manifold/quaternion slots.
    # if constexpr discards the untaken branch, so EULER/SI/RK codegen is byte-identical.
    self.gen_add_code_line("if constexpr (IT == IntegratorType::TRAPEZOIDAL) {", True)
    self._emit_q_update("dt", "s_x_kp1", src_q_name="s_q", src_v_name="s_qd",
                        accel_name="s_qdd", accel_scale_expr="static_cast<T>(0.5) * dt * dt")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("else {", True)
    self.gen_add_code_line(f"const T *s_src_v = (IT == IntegratorType::SEMI_IMPLICIT_EULER) ? &s_x_kp1[{nq}] : s_qd;")
    # q-update: fb -> SE(3) Lie retract; spherical -> SO(3) per-ball retract +
    # additive table for the rest; cardinal -> parallel Euler add. (size nq.)
    self._emit_q_update("dt", "s_x_kp1", src_q_name="s_q", src_v_name="s_src_v")
    self.gen_add_end_control_flow()

    # ---- Multi-stage IT values are not supposed to hit this function ----
    # Compile-time sentinel: emit a static_assert that fires if someone tries
    # to instantiate integrator_finish for MP/RK3/RK4 (they should drive the
    # finish inline from integrator_inner's multi-stage block).
    self.gen_add_code_line(
        "static_assert(IT == IntegratorType::EULER || IT == IntegratorType::SEMI_IMPLICIT_EULER || IT == IntegratorType::TRAPEZOIDAL,")
    self.gen_add_code_line(
        "              \"integrator_finish only handles single-stage IT; multi-stage uses inner directly.\");")
    self.gen_add_end_function()


def gen_integrator_inner_function_call(self, integrator_type="IT", updated_var_names=None,
                                       minv_f_in_smem_expr="true"):
    var_names = dict(
        s_x_kp1_name="s_x_kp1",
        s_q_name="s_q",
        s_qd_name="s_qd",
        s_u_name="s_u",
        s_qdd_name="s_qdd",
        s_stage_qdd_name="s_stage_qdd",
        s_stage_point_name="s_stage_point",
        d_robotModel_name="d_robotModel",
        s_temp_name="s_temp",
        d_workspace_name="nullptr",
        d_f_ext_name="nullptr",
        dt_name="dt",
        gravity_name="gravity",
    )
    if updated_var_names is not None:
        for key, value in updated_var_names.items():
            var_names[key] = value
    code_start = ("integrator_inner<T, " + _integrator_type_token(integrator_type) + ", " + minv_f_in_smem_expr + ">(" +
                  var_names["s_x_kp1_name"] + ", " +
                  var_names["s_q_name"] + ", " +
                  var_names["s_qd_name"] + ", " +
                  var_names["s_u_name"] + ", " +
                  var_names["s_qdd_name"] + ", " +
                  var_names["s_stage_qdd_name"] + ", " +
                  var_names["s_stage_point_name"] + ", ")
    code_end = (var_names["d_robotModel_name"] + ", " +
                var_names["s_temp_name"] + ", " +
                var_names["d_workspace_name"] + ", " +
                var_names["d_f_ext_name"] + ", " +
                var_names["gravity_name"] + ", " +
                var_names["dt_name"] + ");")
    code_middle = self.gen_insert_helpers_function_call()
    self.gen_add_code_line(code_start + code_middle + code_end)


def gen_integrator_inner(self):
    """Templated inner: invokes forward_dynamics_inner(es) then either the
    single-stage integrator_finish or a multi-stage weighted assembly.

    Templated on `<T, IntegratorType IT, bool MINV_F_IN_SMEM>`. The single
    surgical lever (MINV_F_IN_SMEM) is forwarded straight into every FD-inner
    call: when true the FD inner's 6*NV*NV Minv F-region lives in s_temp, when
    false it spills to the L2-pinned d_workspace (the hot FD path stays in
    smem either way). Arenas are sized per placement by the canonical trio in
    GRiDCodeGenerator.py: INTEGRATOR_INNER_SMEM_BYTES<T, MINV_F_IN_SMEM>,
    INTEGRATOR_INNER_WORKSPACE_BYTES<T, MINV_F_IN_SMEM>, and the per-robot
    tier->placement map INTEGRATOR_MINV_F_IN_SMEM<TIER>. There is intentionally
    no whole-arena lever here — the value path's single F lever is sufficient.

    Caller owns:
      - the stage-1 s_XImats load (load_update_XImats_helpers for s_q) — the
        inner re-derives s_XImats internally only for the multi-stage
        intermediate configs (s_p1_q/...); stage 1 is loaded by the
        device/kernel wrappers before the call. (Kept caller-owned so the
        value-path emitted CUDA stays byte-identical; the optional
        helper-inside-inner uniformity move was deliberately skipped.)
      - `s_qdd`: stage-1 qdd output (size n) — always used.
      - `s_stage_qdd`: stages 2..N qdd outputs (size (max_stages-1)*n) — only
        used for multi-stage integrators (Midpoint/RK3/RK4).
      - `s_stage_point`: intermediate state scratch (size (max_stages-1)*2n) —
        only used for multi-stage integrators.
    For Euler/SI-Euler, `s_stage_qdd` / `s_stage_point` are allocated but
    never touched.
    """
    n = self.robot.get_num_vel()
    nq = self.robot.get_num_pos()
    fb = self.robot.floating_base
    slot_size = nq + n  # per-stage intermediate (q, qd) storage = nq + nv
    max_stages = _max_stages_in_use()
    extra_qdd_count = (max_stages - 1) * n
    extra_point_count = (max_stages - 1) * slot_size
    func_params = ["s_x_kp1 is a pointer to memory for the next state (size NUM_POS + NUM_VEL)",
                   "s_q is the vector of joint positions",
                   "s_qd is the vector of joint velocities",
                   "s_u is the vector of joint input torques",
                   "s_qdd is shared memory for the stage-1 joint accelerations (size NUM_VEL)",
                   "s_stage_qdd is shared memory for stages 2..N qdd outputs (size " + str(extra_qdd_count) + ")",
                   "s_stage_point is shared memory for stages 2..N intermediate (q,qd) states (size " + str(extra_point_count) + ")",
                   "s_temp is the pointer to the shared memory needed of size: " +
                       str(self.gen_integrator_inner_temp_mem_size(minv_f_in_smem=True)),
                   "gravity is the gravity constant",
                   "dt is the integration timestep"]
    func_def_start = ("void integrator_inner(T *s_x_kp1, const T *s_q, const T *s_qd, const T *s_u, "
                      "T *s_qdd, T *s_stage_qdd, T *s_stage_point, ")
    # d_robotModel is needed by multi-stage integrators to recompute s_XImats
    # at intermediate states. For single-stage (Euler / SI Euler) it's unused.
    # d_workspace holds the surgically-spilled Minv F-region (6*NV*NV) at the
    # LITE/MINIMAL tiers (MINV_F_IN_SMEM=false); nullptr / unused at PERF.
    func_def_end = "const robotModel<T> *d_robotModel, T *s_temp, T *d_workspace, T *d_f_ext, const T gravity, const T dt) {"
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -3)
    func_params.append("d_workspace is the L2-pinned global scratch for the spilled Minv F-region (LITE/MINIMAL); nullptr at PERF")
    func_params.append("d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr")
    func_notes = ["Assumes s_XImats is updated already for the current s_q",
                  "MINV_F_IN_SMEM selects where the FD inner's Minv 6*NV*NV F-region lives (s_temp vs d_workspace)",
                  "For Midpoint/RK3/RK4, re-runs forward_dynamics at intermediate states and weights stage qdd outputs."]
    self.gen_add_func_doc("Computes a single integrator step (x_{k+1} = integrator(x_k, u_k, dt))",
                          func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, IntegratorType IT, bool MINV_F_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def_start + func_def_end, True)

    # Stage 1: always run forward dynamics on (q, qd, u). Thread the Minv-F
    # placement + global scratch through every FD inner call (stages reuse the
    # same F bytes sequentially).
    self.gen_forward_dynamics_inner_function_call(
        updated_var_names=dict(d_workspace_name="d_workspace", d_f_ext_name="d_f_ext"), minv_f_in_smem_expr="MINV_F_IN_SMEM")
    self.gen_add_sync()

    # Single-stage branch — Euler / Semi-Implicit Euler.
    self.gen_add_code_line("if constexpr (IT == IntegratorType::EULER || IT == IntegratorType::SEMI_IMPLICIT_EULER || IT == IntegratorType::TRAPEZOIDAL) {", True)
    self.gen_integrator_finish_function_call(integrator_type="IT")
    self.gen_add_end_control_flow()

    # Multi-stage branch — emits each subsequent stage in turn, with an
    # if-constexpr to gate which stages actually run for which IT.
    # All multi-stage IT values share the same stage-driver structure with
    # different Butcher coefficients selected at compile time.
    self.gen_add_code_line("else {", True)
    # Aliases for stage-scratch slices.
    # Per-stage slot: nq (q) + nv (qd). For fixed-base nq==nv so slot=2*n.
    self.gen_add_code_line("T *s_qdd_2 = &s_stage_qdd[0];")
    self.gen_add_code_line("T *s_p1_q  = &s_stage_point[0];")
    self.gen_add_code_line("T *s_p1_qd = &s_stage_point[" + str(nq) + "];")
    if max_stages >= 3:
        self.gen_add_code_line("T *s_qdd_3 = &s_stage_qdd[" + str(n) + "];")
        self.gen_add_code_line("T *s_p2_q  = &s_stage_point[" + str(slot_size) + "];")
        self.gen_add_code_line("T *s_p2_qd = &s_stage_point[" + str(slot_size + nq) + "];")
    if max_stages >= 4:
        self.gen_add_code_line("T *s_qdd_4 = &s_stage_qdd[" + str(2 * n) + "];")
        self.gen_add_code_line("T *s_p3_q  = &s_stage_point[" + str(2 * slot_size) + "];")
        self.gen_add_code_line("T *s_p3_qd = &s_stage_point[" + str(2 * slot_size + nq) + "];")

    # ----- Stage 2 (Midpoint / RK3 / RK4): p1 = x + c1*dt*[qd; qdd_1] -----
    # Midpoint: c1 = 0.5. RK3: c1 = 0.5. RK4: c1 = 0.5. (All three use 0.5
    # for stage 2's offset.) For floating-base the q-update is a Lie retract.
    self.gen_add_code_line("constexpr T c1 = static_cast<T>(0.5);")
    self.gen_add_parallel_loop("ind", str(n))
    self.gen_add_code_line("s_p1_qd[ind] = s_qd[ind] + c1 * dt * s_qdd[ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self._emit_q_update("c1 * dt", "s_p1_q", src_q_name="s_q", src_v_name="s_qd",
                        cardinal_line="s_p1_q[ind]  = s_q[ind]  + c1 * dt * s_qd[ind];")
    self.gen_add_sync()
    # IMPORTANT: re-derive s_XImats for the stage-2 configuration before
    # invoking FD — the helper was last populated for s_q (stage 1).
    self.gen_load_update_XImats_helpers_function_call(updated_var_names=dict(s_q_name="s_p1_q"))
    self.gen_add_sync()
    # FD at p1.
    self.gen_forward_dynamics_inner_function_call(updated_var_names=dict(
        s_q_name="s_p1_q", s_qd_name="s_p1_qd", s_qdd_name="s_qdd_2", d_workspace_name="d_workspace", d_f_ext_name="d_f_ext",
    ), minv_f_in_smem_expr="MINV_F_IN_SMEM")
    self.gen_add_sync()

    # ----- Stage 3 (RK3 / RK4) -----
    if max_stages >= 3:
        self.gen_add_code_line("if constexpr (IT == IntegratorType::RK3 || IT == IntegratorType::RK4) {", True)
        # TrajoptPlant convention: xdot_i = [qd; qdd_i] (note: uses original qd,
        # NOT the stage-i velocity). So p_2 = xk + c2*dt*xdot_2 means
        # p_2.q = q + c2*dt*qd, p_2.qd = qd + c2*dt*qdd_2.
        # RK3: c2 = 0.75 (point2 = xk + 0.75*dt*xdot_2)
        # RK4: c2 = 0.5  (point2 = xk + 0.5*dt*xdot_2)
        self.gen_add_code_line("constexpr T c2 = (IT == IntegratorType::RK3) ? static_cast<T>(0.75) : static_cast<T>(0.5);")
        self.gen_add_parallel_loop("ind", str(n))
        self.gen_add_code_line("s_p2_qd[ind] = s_qd[ind] + c2 * dt * s_qdd_2[ind];")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self._emit_q_update("c2 * dt", "s_p2_q", src_q_name="s_q", src_v_name="s_qd",
                            cardinal_line="s_p2_q[ind]  = s_q[ind]  + c2 * dt * s_qd[ind];")
        self.gen_add_sync()
        self.gen_load_update_XImats_helpers_function_call(updated_var_names=dict(s_q_name="s_p2_q"))
        self.gen_add_sync()
        self.gen_forward_dynamics_inner_function_call(updated_var_names=dict(
            s_q_name="s_p2_q", s_qd_name="s_p2_qd", s_qdd_name="s_qdd_3", d_workspace_name="d_workspace", d_f_ext_name="d_f_ext",
        ), minv_f_in_smem_expr="MINV_F_IN_SMEM")
        self.gen_add_sync()
        self.gen_add_end_control_flow()

    # ----- Stage 4 (RK4 only) -----
    if max_stages >= 4:
        self.gen_add_code_line("if constexpr (IT == IntegratorType::RK4) {", True)
        self.gen_add_code_line("constexpr T c3 = static_cast<T>(1.0);")
        self.gen_add_parallel_loop("ind", str(n))
        self.gen_add_code_line("s_p3_qd[ind] = s_qd[ind] + c3 * dt * s_qdd_3[ind];")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self._emit_q_update("c3 * dt", "s_p3_q", src_q_name="s_q", src_v_name="s_qd",
                            cardinal_line="s_p3_q[ind]  = s_q[ind]  + c3 * dt * s_qd[ind];")
        self.gen_add_sync()
        self.gen_load_update_XImats_helpers_function_call(updated_var_names=dict(s_q_name="s_p3_q"))
        self.gen_add_sync()
        self.gen_forward_dynamics_inner_function_call(updated_var_names=dict(
            s_q_name="s_p3_q", s_qd_name="s_p3_qd", s_qdd_name="s_qdd_4", d_workspace_name="d_workspace", d_f_ext_name="d_f_ext",
        ), minv_f_in_smem_expr="MINV_F_IN_SMEM")
        self.gen_add_sync()
        self.gen_add_end_control_flow()

    # ----- Final assembly: x_{k+1} = xk + dt * sum(b_i * xdot_i) -----
    # In TrajoptPlant's convention, xdot_i = [qd; qdd_i] with the SAME qd for
    # every stage. So q_{k+1} is just integrate(q, dt*qd) (Euler-style q
    # update; matches all multi-stage variants), and qd_{k+1} is the
    # weighted sum of the stage qdds.
    self.gen_add_code_line("// final assembly: v_{k+1} part — qd + dt * sum(b_i * qdd_i)")
    self.gen_add_parallel_loop("ind", str(n))
    self.gen_add_code_line("T accel = static_cast<T>(0);")
    self.gen_add_code_line("if constexpr (IT == IntegratorType::MIDPOINT) {")
    self.gen_add_code_line("    accel = s_qdd_2[ind];")
    self.gen_add_code_line("} else if constexpr (IT == IntegratorType::RK3) {")
    self.gen_add_code_line("    constexpr T b1 = static_cast<T>(2.0/9.0);")
    self.gen_add_code_line("    constexpr T b2 = static_cast<T>(3.0/9.0);")
    self.gen_add_code_line("    constexpr T b3 = static_cast<T>(4.0/9.0);")
    self.gen_add_code_line("    accel = b1 * s_qdd[ind] + b2 * s_qdd_2[ind] + b3 * s_qdd_3[ind];")
    self.gen_add_code_line("} else if constexpr (IT == IntegratorType::RK4) {")
    self.gen_add_code_line("    constexpr T b1 = static_cast<T>(1.0/6.0);")
    self.gen_add_code_line("    constexpr T b2 = static_cast<T>(2.0/6.0);")
    self.gen_add_code_line("    constexpr T b3 = static_cast<T>(2.0/6.0);")
    self.gen_add_code_line("    constexpr T b4 = static_cast<T>(1.0/6.0);")
    self.gen_add_code_line("    accel = b1 * s_qdd[ind] + b2 * s_qdd_2[ind] + b3 * s_qdd_3[ind] + b4 * s_qdd_4[ind];")
    self.gen_add_code_line("}")
    self.gen_add_code_line(f"s_x_kp1[{nq} + ind] = s_qd[ind] + dt * accel;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # q_{k+1} part: same as Euler since the q-source is always the original qd
    self.gen_add_code_line("// q_{k+1} part — Euler-style integrate(q, dt*qd) (TrajoptPlant convention)")
    self._emit_q_update("dt", "s_x_kp1", src_q_name="s_q", src_v_name="s_qd")
    self.gen_add_end_control_flow()  # end else (multi-stage)
    self.gen_add_end_function()


def gen_integrator_device(self):
    n = self.robot.get_num_vel()
    nq = self.robot.get_num_pos()
    func_params = ["s_x_kp1 is a pointer to memory for the next state (size NUM_POS + NUM_VEL)",
                   "s_q is the vector of joint positions",
                   "s_qd is the vector of joint velocities",
                   "s_u is the vector of joint input torques",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
                   "d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr",
                   "gravity is the gravity constant",
                   "dt is the integration timestep"]
    func_def_start = "void integrator_device(T *s_x_kp1, const T *s_q, const T *s_qd, const T *s_u, "
    func_def_end = "const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity, const T dt) {"
    # Device wrapper keeps the FD Minv-F region in smem (the default PERF
    # placement); the spill ladder is exercised through the kernel path.
    shared_mem_size = self.gen_integrator_inner_temp_mem_size(minv_f_in_smem=True)
    max_stages = _max_stages_in_use()
    extra_t_buffers = [
        ("s_qdd", n),
        ("s_stage_qdd", (max_stages - 1) * n),
        ("s_stage_point", (max_stages - 1) * (nq + n)),  # per-stage [q (nq); qd (nv)]
    ]
    # shared device-wrapper skeleton (B+C §1.1)
    self.gen_device_wrapper(
        "Computes a single integrator step using the precomputed robotModel",
        func_def_start + func_def_end, shared_mem_size,
        lambda: self.gen_integrator_inner_function_call(integrator_type="IT",
            updated_var_names=dict(d_f_ext_name="d_f_ext")),
        template_line = "template <typename T, IntegratorType IT = IntegratorType::EULER>",
        func_notes = [], func_params = func_params,
        extra_t_buffers = extra_t_buffers, include_linalg_scratch = True)


def _emit_integrator_kernel_body_for_flags(self, nq, nv, spill_minv_F, single_call_timing):
    """Emit integrator_kernel body for one tier's Minv-F spill flag.
    spill_minv_F=False: the FD inner's Minv F-region lives in smem (s_temp);
    spill_minv_F=True:  it lives in the L2-pinned d_workspace (surgical spill,
    keeps the hot FD path in smem)."""
    fb = self.robot.floating_base  # 0 for fixed-base
    max_stages = _max_stages_in_use()
    # Inner-controlled: forward_dynamics_inner slices its own Minv-F from s_temp
    # (smem) or d_workspace (global). The arena size already reflects the choice.
    shared_mem_size = self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=not spill_minv_F)
    # Canonical INPUT packing (mirrors id/crba/aba/forward_dynamics): q, qd, u each
    # occupy a NUM_JOINTS(=nq)-wide slot at stride 3*nq; slice qd at nq, u at 2*nq.
    # For a FIXED base nq==nv so 3*nq == 3*nv+fb byte-identical; for a FLOATING
    # base nq=nv+1 the old nv-strided u offset (2*nv+fb) under-read by nq-nv and
    # mis-sliced u -- the floating B=1 + batch input bug. The OUTPUT state x_kp1 is
    # genuinely nq+nv wide (q is nq, qd is nv), so out_count stays nq+nv.
    input_count = 3 * nq
    extra_t_buffers = [
        ("s_q_qd_u", input_count),
        ("s_qdd", nv),
        ("s_stage_qdd", (max_stages - 1) * nv),
        ("s_stage_point", (max_stages - 1) * (nq + nv)),
        ("s_x_kp1", nq + nv),  # next state [q (nq); qd (nv)]
    ]
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size, extra_t_buffers=extra_t_buffers, include_linalg_scratch=True)
    self.gen_add_code_line(
        "T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[" + str(nq) + "]; T *s_u = &s_q_qd_u[" + str(2 * nq) + "];"
    )
    minv_f_expr = "false" if spill_minv_F else "true"
    out_count = nq + nv  # next state [q (nq); qd (nv)]
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q_qd_u",str(input_count),stride="stride_q_qd_u")
        if spill_minv_F:
            self.gen_add_code_line("T *int_d_workspace = reinterpret_cast<T *>(&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()]);")
        else:
            self.gen_add_code_line("(void)d_workspace;")
        # mjx input convert (RETRACT family): reorder ONLY the input base
        # quaternion wxyz->xyzw so the kernel's SE(3) quaternion integration
        # (grid_integrate_floating_q reads/writes xyzw) and XImats X[0] are
        # built correctly. Do NOT convert qd: the mjx retract needs the RAW mjx
        # GLOBAL base-linear velocity qd[0:3], and the quaternion integration
        # uses qd[3:6] (angular, frame-shared) -> qd stays raw mjx. Must precede
        # the XImats build below.
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_quat_reorder("s_q")
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_integrator_inner_function_call(integrator_type="IT",
            updated_var_names=(dict(d_workspace_name="int_d_workspace", d_f_ext_name="d_f_ext") if spill_minv_F else dict(d_f_ext_name="d_f_ext")),
            minv_f_in_smem_expr=minv_f_expr)
        self.gen_add_sync()
        # mjx output (RETRACT): the kernel integrated q in the PIN convention
        # (SE(3) V(phi) base-position coupling, which is O(dt^2) wrong for mjx).
        # OVERWRITE the base-linear position of s_x_kp1 with the mjx GLOBAL
        # additive step  s_q[0:3] + dt*s_qd[0:3]  (s_q still holds the ORIGINAL
        # pre-integration base position -- the kernel integrates OUT-of-place
        # into the separate s_x_kp1 buffer; s_qd[0:3] is the raw mjx global
        # base-linear velocity). The quaternion + joints the kernel computed are
        # kept. Then convert the output base quaternion xyzw->wxyz back to mjx
        # order: gen_mjx_quat_reorder is a cyclic LEFT-rotate of slots[3..6]
        # (wxyz->xyzw), NOT an involution, so the inverse (xyzw->wxyz) is the
        # cyclic RIGHT-rotate emitted inline here.
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_retract("s_x_kp1", "s_q", "s_qd", "dt")
            self.gen_add_code_lines([
                "// mjx output: base quaternion xyzw->wxyz (inverse of input reorder)",
                "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
                "T qw_out = s_x_kp1[6];",
                "s_x_kp1[6] = s_x_kp1[5]; s_x_kp1[5] = s_x_kp1[4]; s_x_kp1[4] = s_x_kp1[3]; s_x_kp1[3] = qw_out;",
            ])
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            self.gen_add_end_control_flow()
        self.gen_kernel_save_result("x_kp1",str(out_count),stride=str(out_count))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q_qd_u",str(input_count))
        if spill_minv_F:
            self.gen_add_code_line("T *int_d_workspace = reinterpret_cast<T *>(&d_workspace[GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()]);")
        else:
            self.gen_add_code_line("(void)d_workspace;")
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q_qd_u", str(input_count), feedback_from="x_kp1")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_integrator_inner_function_call(integrator_type="IT",
            updated_var_names=(dict(d_workspace_name="int_d_workspace", d_f_ext_name="d_f_ext") if spill_minv_F else dict(d_f_ext_name="d_f_ext")),
            minv_f_in_smem_expr=minv_f_expr)
        self.gen_anti_licm_output_write("x_kp1")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("x_kp1",str(out_count))


def gen_integrator_kernel(self, single_call_timing=False):
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    func_params = ["d_x_kp1 is a pointer to memory for the next state (size 2*NUM_VEL per timestep)",
                   "d_workspace is the L2-pinned global scratch for the spilled Minv F-region (LITE/MINIMAL tiers)",
                   "d_q_qd_u is the packed joint positions, velocities, and input torques",
                   "stride_q_qd_u is the stride between each (q, qd, u) tuple in d_q_qd_u",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
                   "d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr",
                   "gravity is the gravity constant",
                   "dt is the integration timestep",
                   "num_timesteps is the length of the trajectory (or overloaded as test_iters for timing)"]
    func_def_start = "void integrator_kernel(T *d_x_kp1, unsigned char *d_workspace, const T *d_q_qd_u, const int stride_q_qd_u, "
    func_def_end = "const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity, const T dt, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Computes a single integrator step per timestep (Euler by default)",
                          [], func_params, None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag, LAST
    # after RESOURCE_TIER so existing positional <T,IT,TIER> call sites are
    # unaffected; default false if-constexpr-elides the mjx retract epilogue ->
    # byte-identical PTX on the pin path.
    self.gen_add_code_line("template <typename T, IntegratorType IT = IntegratorType::EULER, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    # Pin launch_bounds to MAX_PERF_LEVEL_THREADS (the PERF cap), NOT tier_max_threads:
    # the integrator is register-bound by its RBD callees (load_update_XImats ~86,
    # minv_inner ~92, inverse_dynamics_gradient_inner ~99 regs), so the
    # LITE/MINIMAL thread-count bump (-> fewer regs/thread) starves them and ptxas
    # errors under -rdc=true (callee regcount > caller cap). The integrator's tier
    # behavior is the surgical Minv-F smem spill, which is independent of launch_bounds.
    self.gen_add_code_line("__launch_bounds__(MAX_PERF_LEVEL_THREADS)")
    self.gen_add_code_line(func_def, True)
    # Per-tier Minv-F placement (perf, lite, minimal): 0 = F in smem, 1 = F
    # spilled to d_workspace. When all three agree (robots that fit at PERF),
    # emit a single body; else gate per tier on RESOURCE_TIER (mirrors fd).
    picks = getattr(self, "integrator_spill_tier_3way", (0, 0, 0))
    self.gen_tier_dispatch(picks, lambda pick:
        _emit_integrator_kernel_body_for_flags(self, nq, nv, bool(pick), single_call_timing))
    self.gen_add_end_function()


def gen_integrator_host(self, mode=0):
    single_call_timing = mode == 1
    compute_only = mode == 2
    func_params = ["hd_data is the packaged input and output pointers",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
                   "gravity is the gravity constant",
                   "dt is the integration timestep",
                   "num_timesteps is the length of the trajectory (or overloaded as test_iters for timing)",
                   "streams are pointers to CUDA streams for async memory transfers (if needed)"]
    func_def_start = "void integrator(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const T dt, const int num_timesteps,"
    func_def_end = "                  const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(", 1)
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(", 1)
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    self.gen_add_func_doc("Run a single integrator step (default Euler) per timestep",
                          [], func_params, None)
    # MUJOCO_OUTPUT (floating only) host flag, LAST: forwarded to the kernel
    # launch (naming IT + the tier positionally to reach the trailing flag).
    # Default false -> byte-identical pin codegen.
    mjx_host = self.robot.floating_base
    if mjx_host:
        self.gen_add_code_line("template <typename T, IntegratorType IT = IntegratorType::EULER, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    else:
        self.gen_add_code_line("template <typename T, IntegratorType IT = IntegratorType::EULER, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"integrator requires all-data or dynamics gridData\");")
    integrator_kernel_tmpl = "integrator_kernel<T, IT, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "integrator_kernel<T, IT, RESOURCE_TIER>"
    func_call_start = integrator_kernel_tmpl + "<<<block_dimms,thread_dimms,INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_x_kp1,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd_u,"
    func_call_end = "d_robotModel,hd_data->d_f_ext,gravity,dt,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("integrator_kernel<", "integrator_kernel_single_timing<")
    self.gen_add_code_line("int stride_q_qd_u = 3*NUM_JOINTS;")
    if not compute_only:
        self.gen_add_code_lines([
            "// start code with memory transfer",
            "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_u*" +
                ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));",
            "gpuErrchkKernel();",
        ])
    self.gen_add_code_line("// then call the kernel")
    func_call_code = [func_call_start + func_call_end, "gpuErrchkKernel();"]
    if single_call_timing:
        func_call_code.insert(0, "struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"integrator\", INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    # Pin the spilled Minv-F section in L2 when any tier spills it.
    workspace_bytes = ("GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()" if single_call_timing
                       else "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(num_timesteps)")
    self.gen_add_code_line("if (GRID_INTEGRATOR_USES_WORKSPACE) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, " + workspace_bytes + "));}")
    self.gen_add_code_lines(func_call_code)
    self.gen_add_code_line("if (GRID_INTEGRATOR_USES_WORKSPACE) {gpuErrchk(grid_end_l2_persisting(0));}")
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back",
            "gpuErrchk(cudaMemcpy(hd_data->h_x_kp1,hd_data->d_x_kp1,(NUM_POS + NUM_VEL)*" +
                ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();",
        ])
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("integrator"))
    self.gen_add_end_function()


def gen_integrator(self):
    # Emit finish + inner (templated on IT), then EULER-typed device/kernel/host.
    # For floating-base, also emit SE(3) Lie-group helpers used by the
    # q-update Lie retract (the fixed-base path doesn't reference them).
    # The d2ee kinematic codegen may also emit these helpers; only emit here
    # if they weren't already emitted (avoid C++ redefinition).
    if self.robot.floating_base and not getattr(self, "_lie_helpers_emitted", False):
        # Floating-base: emit the full SE(3) Lie bundle (the q-update Lie retract
        # + dIntegrate/d2Integrate blocks). For a floating robot that ALSO has a
        # spherical joint, gen_lie_group_helpers additionally emits the spherical
        # SO(3) wrapper (gated inside it on robot_has_spherical()).
        self.gen_lie_group_helpers()
        self._lie_helpers_emitted = True
    elif (not self.robot.floating_base) and self.robot.robot_has_spherical() \
            and not getattr(self, "_lie_helpers_emitted", False):
        # Fixed-base spherical robot: it never references the SE(3) bundle, only
        # the SO(3) quaternion retract. Emit JUST that (+ its quaternion deps) so
        # the header stays lean and pure-floating codegen is unaffected.
        self.gen_integrate_spherical_helper()
        self._lie_helpers_emitted = True
    self.gen_integrator_finish()
    self.gen_integrator_inner()
    self.gen_integrator_device()
    self.gen_integrator_kernel(single_call_timing=True)
    self.gen_integrator_kernel(single_call_timing=False)
    self.gen_integrator_host(0)
    self.gen_integrator_host(1)
    self.gen_integrator_host(2)

"""Time-integrator value codegen.

Mirrors `_forward_dynamics.py` (inner / device / kernel / host layers) but
emits a single time step `x_{k+1} = integrator(x_k, u_k, dt; f_dyn)` where
`f_dyn` is the existing forward dynamics. State is `x = [q (nq); qd (nv)]`
(including floating and spherical joints), control `u` is size `nv`, output
`x_{k+1}` is size `nq + nv`. dt is a per-call scalar.

The integrator type is selected at compile time by an `IntegratorType IT`
template parameter. Multi-stage schemes integrate the full state, retracting
each configuration from the initial configuration with the previous stage's
velocity. RK4 has classical fourth order on Euclidean configurations; the
base-point rotational retraction has second order, not Lie-group RK4 order.
"""
from grid_codegen.helpers._code_generation_helpers import gen_workspace_repoint_line, wrap_host_single_call_timing


# integrator name <-> codegen-side string constant
_INTEGRATOR_TYPES = ("EULER", "SEMI_IMPLICIT_EULER", "MIDPOINT", "RK4", "TRAPEZOIDAL", "CONSTANT_ACCELERATION")

# Number of forward-dynamics evaluations each integrator type requires.
# Used at codegen time to size shared-memory buffers (per-stage qdd) and to
# guide which stage-computation branches are emitted.
# CONSTANT_ACCELERATION uses one evaluation; TRAPEZOIDAL is explicit Heun.
_STAGE_COUNT = {
    "EULER": 1,
    "SEMI_IMPLICIT_EULER": 1,
    "MIDPOINT": 2,
    "TRAPEZOIDAL": 2,
    "RK4": 4,
    "CONSTANT_ACCELERATION": 1,
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

    Idempotent via _lie_helpers_emitted (defines the same C++ symbols the
    spherical dIntegrate bundle guards against): safe to call from every
    consumer without an external check.
    """
    if getattr(self, "_lie_helpers_emitted", False):
        return
    self._lie_helpers_emitted = True
    self.gen_add_func_doc("Floating-base Lie-group helpers (xyzw quaternion, Pinocchio v order).", [], [], None)
    self.gen_add_code_lines([
        # Lie primitives live in the vendored GLASS headers (glass::thread::quat_*,
        # so3_*, se3_*), column-major 3x3 and 6x6. The helpers below wire them to the
        # project's q layout and to row-major output; they hold no mathematics of
        # their own.
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
        "    // GLASS Lie ops (glass::thread::, COLUMN-MAJOR 3x3): quat_exp takes the",
        "    // FULL rotation vector (halves internally); so3_left_jacobian IS the SE(3)",
        "    // \"V matrix\" Jl. mat3-vec products below use the column-major convention.",
        "    T rho[3]      = {v_dt[0], v_dt[1], v_dt[2]};",
        "    T omega_dt[3] = {v_dt[3], v_dt[4], v_dt[5]};",
        "    T dq[4]; glass::thread::quat_exp<T>(omega_dt, dq);",
        # The kernels evaluate R(p/|p|): the public function is f o normalize on the
        # ambient quaternion (the contract every q cotangent assumes, zero radial
        # part). The position retract used the RAW quaternion's rotation before
        # 2026-11-03, so a scaled input moved the base by |p|^2 (found by the
        # floating integrator VJP's ambient finite differences).
        "    T q_raw_quat[4] = {q[3], q[4], q[5], q[6]};",
        "    T q_old_quat[4]; glass::thread::quat_normalize<T>(q_raw_quat, q_old_quat);",
        "    T q_new_quat[4]; glass::thread::quat_mul<T>(q_old_quat, dq, q_new_quat);",
        "    T q_norm[4]; glass::thread::quat_normalize<T>(q_new_quat, q_norm);   // renormalize",
        "    #pragma unroll",
        "    for (int i = 0; i < 4; ++i) q_new[3 + i] = q_norm[i];",
        "    T V[9]; glass::thread::so3_left_jacobian<T>(omega_dt, V);            // SE(3) V matrix (col-major)",
        "    T p_delta_local[3];   // p_delta_local = V * rho  (V column-major)",
        "    #pragma unroll",
        "    for (int i = 0; i < 3; ++i) p_delta_local[i] = V[i]*rho[0] + V[3+i]*rho[1] + V[6+i]*rho[2];",
        "    T R_old[9]; glass::thread::quat_to_rot<T>(q_old_quat, R_old);        // col-major",
        "    T p_delta_world[3];   // p_delta_world = R_old * p_delta_local  (R_old column-major)",
        "    #pragma unroll",
        "    for (int i = 0; i < 3; ++i) p_delta_world[i] = R_old[i]*p_delta_local[0] + R_old[3+i]*p_delta_local[1] + R_old[6+i]*p_delta_local[2];",
        "    q_new[0] = q[0] + p_delta_world[0];",
        "    q_new[1] = q[1] + p_delta_world[1];",
        "    q_new[2] = q[2] + p_delta_world[2];",
        "    // remaining (revolute joints): Euler add. v_dt[6:nv] -> q[7:nq].",
        "    // (n_joints = NUM_POS - 7 = nv - 6)",
        "    #pragma unroll",
        "    for (int i = 0; i < (NUM_POS - 7); ++i) q_new[7 + i] = q[7 + i] + v_dt[6 + i];",
        "}",
        "",
        # ---- Lie-group q difference (boxminus): the exact inverse of the retract above ----
        # GATO ASK 4 (2026-08-09): the SQP defect c_k = q_{k+1} [-] integrate(q_k, u_k),
        # initial-state gaps, and merit integrator-error all need this; math half lives in
        # GLASS se3_difference (@78329b6, pinocchio-difference convention, canonical |phi|<=pi
        # branch), this is the wiring half. Tangent output uses the SAME user-facing
        # ordering the retract consumes: [v_lin(3); omega(3); joints...] (nv wide).
        "template <typename T, int NUM_POS> __device__ inline void grid_difference_floating_q(",
        "    const T *q_from, const T *q_to, T *dv) {",
        "    // Pose prefix [x,y,z, qx,qy,qz,qw]: glass thread-serial boxminus (xyzw default",
        "    // layout matches the project q layout; see grid_integrate_floating_q above).",
        "    T rho[3]; T phi[3];",
        "    glass::thread::se3_difference<T>(q_from, q_to, rho, phi);",
        "    #pragma unroll",
        "    for (int i = 0; i < 3; ++i) { dv[i] = rho[i]; dv[3 + i] = phi[i]; }",
        "    // remaining (revolute joints): Euler difference. q[7:nq] -> dv[6:nv].",
        "    #pragma unroll",
        "    for (int i = 0; i < (NUM_POS - 7); ++i) dv[6 + i] = q_to[7 + i] - q_from[7 + i];",
        "}",
        "",
        # ---- dIntegrate top-left 6x6 block for ARG_q (SE(3) Adjoint of exp(-v_dt)) ----
        # Written in PINOCCHIO order [v_lin; omega] (matches pin.dIntegrate output).
        "template <typename T> __device__ inline void grid_dIntegrate_q_block(const T *v_dt, T J[36]) {",
        "    // v_dt in Pinocchio order [v_lin; omega]. GLASS returns the block column-major;",
        "    // J is row-major.",
        "    T G[36]; glass::thread::se3_retract_jacobian_q<T>(v_dt, v_dt + 3, G);",
        "    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) J[6*r + c] = G[6*c + r];",
        "}",
        "",
        # ---- dIntegrate top-left 6x6 block for ARG_v (SE(3) right Jacobian) ----
        "template <typename T> __device__ inline void grid_dIntegrate_v_block(const T *v_dt, T J[36]) {",
        "    // v_dt in Pinocchio order [v_lin; omega]. GLASS returns the block column-major;",
        "    // J is row-major.",
        "    T G[36]; glass::thread::se3_retract_jacobian_v<T>(v_dt, v_dt + 3, G);",
        "    for (int r = 0; r < 6; ++r) for (int c = 0; c < 6; ++c) J[6*r + c] = G[6*c + r];",
        "}",
        "",
        # ---- second-order dIntegrate, analytic ----
        # J2[o*36+j*6+k] = d(dInt_block[o,j])/d(w[k]). The mathematics lives once, in
        # glass::thread::se3_retract_hessian (closed form, series below theta=0.2, validated
        # against RBDReference._d2_se3_*). This wrapper only reorders its output.
        "// Deliberately in double inside GLASS: the one exemption from computing in T (see",
        "// test/test_generated_code_pure_T.py). The second derivatives of SE(3) integration",
        "// cancel badly in single precision, so glass::se3_retract_hessian and its call in",
        "// integrator_hessian_device evaluate in double and return T. It is slow on GPUs",
        "// with reduced double throughput.",
        "template <typename T, bool IS_Q> __device__ inline void grid_d2Integrate_block(const T *w, T J2[216]) {",
        "    T H[216]; glass::thread::se3_retract_hessian<T, IS_Q>(w, w + 3, H);  // H[k*36 + col*6 + row]",
        "    for (int o = 0; o < 6; ++o) for (int j = 0; j < 6; ++j) for (int k = 0; k < 6; ++k)",
        "        J2[o*36 + j*6 + k] = H[k*36 + j*6 + o];",
        "}",
        "",
    ])
    # Spherical (ball) joint SO(3) retract helper — emitted ONLY when the robot
    # has a spherical joint (so pure-floating robots stay byte-identical; the
    # block is absent from their header). Block-pointer form: q_new_blk =
    # normalize(q_blk (x) exp(omega/2)) == glass quat_retract. `omega_vec` is the
    # FULL rotation vector scale*omega (glass halves internally; the caller no
    # longer pre-halves) — the SO(3) half of grid_integrate_floating_q with no
    # SE(3) coupling.
    if self.robot.robot_has_spherical():
        self.gen_add_code_lines([
            "template <typename T> __device__ inline void grid_integrate_spherical_q(",
            "    const T *q_blk, const T *omega_vec, T *q_new_blk) {",
            "    glass::thread::quat_retract<T>(q_blk, omega_vec, q_new_blk);",
            "}",
            "",
        ])


def gen_integrate_spherical_helper(self):
    """Standalone emitter for the spherical SO(3) retract device helper, used
    when the robot has a spherical joint but is NOT floating (so the floating
    Lie-group helper bundle isn't otherwise emitted). Delegates to the vendored
    glass::thread::quat_retract (normalize(q (x) exp(omega/2)), FULL rotation
    vector — glass halves internally). Pure-floating robots emit the same
    wrapper via gen_lie_group_helpers instead (and never call this)."""
    self.gen_add_func_doc("Spherical (ball) joint SO(3) quaternion retract helper (xyzw).", [], [], None)
    self.gen_add_code_lines([
        "template <typename T> __device__ inline void grid_integrate_spherical_q(",
        "    const T *q_blk, const T *omega_vec, T *q_new_blk) {",
        "    glass::thread::quat_retract<T>(q_blk, omega_vec, q_new_blk);",
        "}",
        "",
    ])


def gen_spherical_dintegrate_helpers(self):
    """Emit the SO(3) dIntegrate 3x3 block helpers for a FIXED-base spherical
    robot's integrator gradient (the omega-only restriction of the free-flyer
    blocks; mirrors RBDReference.dIntegrate's spherical branch):
        grid_dIntegrate_q_so3(omega_dt, J9) = exp(-omega_dt)   (ARG_q)
        grid_dIntegrate_v_so3(omega_dt, J9) = J_r(omega_dt)    (ARG_v)
    Output J9 is ROW-major (matches the 6x6 helpers' transpose-read convention,
    so the dAB assembly indexes both the same way). Both call the GLASS SO(3)
    primitives directly.
    Idempotent via _sph_dint_helpers_emitted."""
    if getattr(self, "_sph_dint_helpers_emitted", False):
        return
    self._sph_dint_helpers_emitted = True
    self.gen_add_func_doc("Spherical-joint SO(3) dIntegrate block helpers (fixed-base integrator gradient).", [], [], None)
    self.gen_add_code_lines([
        # ---- spherical dIntegrate ARG_q block: exp(-omega_dt), row-major out ----
        "template <typename T> __device__ inline void grid_dIntegrate_q_so3(const T *omega_dt, T J[9]) {",
        "    T neg[3] = {-omega_dt[0], -omega_dt[1], -omega_dt[2]};",
        "    T R_inv[9]; glass::thread::so3_exp<T>(neg, R_inv);  // col-major",
        "    #pragma unroll",
        "    for (int i = 0; i < 3; ++i) {",
        "        #pragma unroll",
        "        for (int j = 0; j < 3; ++j) J[3*i + j] = R_inv[3*j + i];  // -> row-major",
        "    }",
        "}",
        "",
        # ---- spherical dIntegrate ARG_v block: J_r(omega_dt), row-major out ----
        "template <typename T> __device__ inline void grid_dIntegrate_v_so3(const T *omega_dt, T J[9]) {",
        "    T Jr[9]; glass::thread::so3_right_jacobian<T>(omega_dt, Jr);  // col-major",
        "    #pragma unroll",
        "    for (int i = 0; i < 3; ++i) {",
        "        #pragma unroll",
        "        for (int j = 0; j < 3; ++j) J[3*i + j] = Jr[3*j + i];  // -> row-major",
        "    }",
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

    `accel_name` / `accel_scale_expr` (optional, CONSTANT_ACCELERATION): fold an extra
    `accel_scale_expr * accel_name[v]` term into the retracted tangent at every
    v-index, so the effective tangent is `scale*src_v + accel_scale*accel`. This
    lets CONSTANT_ACCELERATION retract `dt*qd + 0.5*dt^2*qdd` in ONE step — correct for
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
            # off-stack `static const` (§1v); each emit site has its own {} scope above.
            self.gen_bake_const_array("add_q", add_q, "int")
            self.gen_bake_const_array("add_v", add_v, "int")
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
            # omega = scale * omega [+ accel_scale * alpha]  (combined angular tangent;
            # FULL rotation vector — glass quat_retract halves internally)
            def _sph_omega(k):
                if accel_name is None:
                    return f"{scale_expr}*{src_v_name}[sph_v_{blk_i}[{k}]]"
                return (f"({scale_expr}*{src_v_name}[sph_v_{blk_i}[{k}]]"
                        f" + {accel_scale_expr}*{accel_name}[sph_v_{blk_i}[{k}]])")
            self.gen_add_code_line(f"T sph_omega_{blk_i}[3] = {{"
                                   f"{_sph_omega(0)}, "
                                   f"{_sph_omega(1)}, "
                                   f"{_sph_omega(2)}}};")
            self.gen_add_code_line(f"T sph_qnew_{blk_i}[4];")
            self.gen_add_code_line(
                f"grid_integrate_spherical_q<T>(sph_qblk_{blk_i}, sph_omega_{blk_i}, sph_qnew_{blk_i});")
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
    # CONSTANT_ACCELERATION retracts the COMBINED tangent dt*qd + 0.5*dt^2*qdd in ONE step
    # (GATO integrator.cuh:36 -> q_next = q + dt*qd + 0.5*dt^2*qdd, reading OLD qd).
    # Folding the accel into the single retract (via _emit_q_update's accel term) is
    # correct for fixed-base (collapses to the in-place add), floating-base (SE(3)
    # Lie retract of the combined tangent) AND spherical (SO(3) half uses the
    # combined angular velocity) — no in-place add onto manifold/quaternion slots.
    # if constexpr discards the untaken branch, so EULER/SI/RK codegen is byte-identical.
    self.gen_add_code_line("if constexpr (IT == IntegratorType::CONSTANT_ACCELERATION) {", True)
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
    # to instantiate integrator_finish for MP/TRAPEZOIDAL/RK4 (they should drive the
    # finish inline from integrator_inner's multi-stage block).
    self.gen_add_code_line(
        "static_assert(IT == IntegratorType::EULER || IT == IntegratorType::SEMI_IMPLICIT_EULER || IT == IntegratorType::CONSTANT_ACCELERATION,")
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
        used for multi-stage integrators (Midpoint/TRAPEZOIDAL/RK4).
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
                  "For Midpoint/TRAPEZOIDAL/RK4, re-runs forward_dynamics at intermediate states and weights stage qdd outputs."]
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
    self.gen_add_code_line("if constexpr (IT == IntegratorType::EULER || IT == IntegratorType::SEMI_IMPLICIT_EULER || IT == IntegratorType::CONSTANT_ACCELERATION) {", True)
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

    # Stage 2: midpoint/RK4 use a half step; explicit Heun a full step.
    self.gen_add_code_line("constexpr T c1 = (IT == IntegratorType::TRAPEZOIDAL) ? static_cast<T>(1) : static_cast<T>(0.5);")
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

    # ----- Stage 3 (RK4 only) -----
    if max_stages >= 3:
        self.gen_add_code_line("if constexpr (IT == IntegratorType::RK4) {", True)
        self.gen_add_code_line("constexpr T c2 = static_cast<T>(0.5);")
        self.gen_add_parallel_loop("ind", str(n))
        self.gen_add_code_line("s_p2_qd[ind] = s_qd[ind] + c2 * dt * s_qdd_2[ind];")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self._emit_q_update("c2 * dt", "s_p2_q", src_q_name="s_q", src_v_name="s_p1_qd")
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
        self._emit_q_update("c3 * dt", "s_p3_q", src_q_name="s_q", src_v_name="s_p2_qd")
        self.gen_add_sync()
        self.gen_load_update_XImats_helpers_function_call(updated_var_names=dict(s_q_name="s_p3_q"))
        self.gen_add_sync()
        self.gen_forward_dynamics_inner_function_call(updated_var_names=dict(
            s_q_name="s_p3_q", s_qd_name="s_p3_qd", s_qdd_name="s_qdd_4", d_workspace_name="d_workspace", d_f_ext_name="d_f_ext",
        ), minv_f_in_smem_expr="MINV_F_IN_SMEM")
        self.gen_add_sync()
        self.gen_add_end_control_flow()

    # ----- Final assembly: x_{k+1} = xk + dt * sum(b_i * xdot_i) -----
    # Weight velocities as well as accelerations; reuse the dead first
    # stage-position buffer for the nv weighted velocities (nq >= nv).
    self.gen_add_code_line("// final assembly: v_{k+1} part — qd + dt * sum(b_i * qdd_i)")
    self.gen_add_parallel_loop("ind", str(n))
    self.gen_add_code_line("T accel = static_cast<T>(0);")
    self.gen_add_code_line("T velocity = static_cast<T>(0);")
    self.gen_add_code_line("if constexpr (IT == IntegratorType::MIDPOINT) {")
    self.gen_add_code_line("    accel = s_qdd_2[ind];")
    self.gen_add_code_line("    velocity = s_p1_qd[ind];")
    self.gen_add_code_line("} else if constexpr (IT == IntegratorType::TRAPEZOIDAL) {")
    self.gen_add_code_line("    accel = static_cast<T>(0.5) * (s_qdd[ind] + s_qdd_2[ind]);")
    self.gen_add_code_line("    velocity = static_cast<T>(0.5) * (s_qd[ind] + s_p1_qd[ind]);")
    self.gen_add_code_line("} else if constexpr (IT == IntegratorType::RK4) {")
    self.gen_add_code_line("    constexpr T b1 = static_cast<T>(1.0/6.0);")
    self.gen_add_code_line("    constexpr T b2 = static_cast<T>(2.0/6.0);")
    self.gen_add_code_line("    constexpr T b3 = static_cast<T>(2.0/6.0);")
    self.gen_add_code_line("    constexpr T b4 = static_cast<T>(1.0/6.0);")
    self.gen_add_code_line("    accel = b1 * s_qdd[ind] + b2 * s_qdd_2[ind] + b3 * s_qdd_3[ind] + b4 * s_qdd_4[ind];")
    self.gen_add_code_line("    velocity = b1 * s_qd[ind] + b2 * s_p1_qd[ind] + b3 * s_p2_qd[ind] + b4 * s_p3_qd[ind];")
    self.gen_add_code_line("}")
    self.gen_add_code_line(f"s_x_kp1[{nq} + ind] = s_qd[ind] + dt * accel;")
    self.gen_add_code_line("s_p1_q[ind] = velocity;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_code_line("// q_{k+1} = integrate(q, dt * sum(b_i * v_i))")
    self._emit_q_update("dt", "s_x_kp1", src_q_name="s_q", src_v_name="s_p1_q")
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
        extra_t_buffers = extra_t_buffers)


def _integrator_kernel_extra_t_buffers(self, nq, nv):
    """The integrator kernel's shared-arena T-slot list. Single source of truth
    shared by _emit_integrator_kernel_body_for_flags and the integrator_arena
    carve struct (gen_integrator_arena_carve_struct) so the two cannot drift.
    Mirrors GCG's integrator t-count comment block byte-for-byte."""
    max_stages = _max_stages_in_use()
    # Canonical INPUT packing (mirrors id/crba/aba/forward_dynamics): q, qd, u each
    # occupy a NUM_JOINTS(=nq)-wide slot at stride 3*nq; slice qd at nq, u at 2*nq.
    input_count = 3 * nq
    return [
        ("s_q_qd_u", input_count),
        ("s_qdd", nv),
        ("s_stage_qdd", (max_stages - 1) * nv),
        ("s_stage_point", (max_stages - 1) * (nq + nv)),
        ("s_x_kp1", nq + nv),  # next state [q (nq); qd (nv)]
    ]


def gen_integrator_arena_carve_struct(self):
    """Emit the namespace-scope `integrator_arena<T>` carve struct:
    external callers allocate INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T, TIER_SHARED>()
    bytes and carve() the exact TIER_SHARED sub-buffer layout the integrator
    kernel uses."""
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    spill_minv_F = bool(getattr(self, "integrator_spill_tier_3way", (0, 0, 0))[0])
    temp = self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=not spill_minv_F)
    layout = self._resolve_arena_layout(
        _integrator_kernel_extra_t_buffers(self, nq, nv), temp,
        include_topology_helpers = self.gen_topology_helpers_size() > 0,
        ximat_size = self.gen_get_XI_size(False, False),
        apply_runtime_transform_band = getattr(self, "runtime_transform", False))
    self.gen_arena_carve_struct(
        "integrator_arena", layout,
        "INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T, TIER_SHARED>()",
        expected_t_count = self.integrator_t_count_per_tier[0],
        doc = "integrator_arena: carve struct mirroring the integrator kernel's TIER_SHARED "
              "shared-arena layout; allocate INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T, TIER_SHARED>() "
              "bytes (e.g. an external solver's own smem block) and call carve(base).")


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
    extra_t_buffers = _integrator_kernel_extra_t_buffers(self, nq, nv)
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size, extra_t_buffers=extra_t_buffers)
    self.gen_add_code_line(
        "T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[" + str(nq) + "]; T *s_u = &s_q_qd_u[" + str(2 * nq) + "];"
    )
    minv_f_expr = "false" if spill_minv_F else "true"
    out_count = nq + nv  # next state [q (nq); qd (nv)]
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_add_code_line("T *d_f_ext_k = d_f_ext ? d_f_ext + static_cast<size_t>(k) * 6 * NUM_BODIES : nullptr;")
        self.gen_kernel_load_inputs("q_qd_u",str(input_count),stride="stride_q_qd_u")
        if spill_minv_F:
            self.gen_add_code_line(gen_workspace_repoint_line("int_d_workspace", "GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()", batch_indexed=True, declare=True))
        else:
            self.gen_add_code_line("(void)d_workspace;")
        # mjx input convert: the base quaternion wxyz->xyzw (the SE(3) quaternion
        # integration and XImats X[0] read xyzw), the base-linear velocity and
        # force R^T into the pin frame, so the forward dynamics inside the
        # integrate sees pin quantities (until 2026-11-03 the raw world-frame
        # velocity was fed as the local one). The world-frame velocity the mjx
        # position step needs is rebuilt as R v in the output epilogue. Must
        # precede the XImats build below.
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_add_code_line('static_assert(IT == IntegratorType::EULER || IT == IntegratorType::SEMI_IMPLICIT_EULER, "MuJoCo integration currently supports Euler and semi-implicit Euler only.");')
            self.gen_mjx_input_convert(q_name="s_q", qd_name="s_qd", u_name="s_u")
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_integrator_inner_function_call(integrator_type="IT",
            updated_var_names=(dict(d_workspace_name="int_d_workspace", d_f_ext_name="d_f_ext_k") if spill_minv_F else dict(d_f_ext_name="d_f_ext_k")),
            minv_f_in_smem_expr=minv_f_expr)
        self.gen_add_sync()
        # mjx output: the MuJoCo velocity step (world-linear qd_{k+1} = R (qd_{k+1,pin}
        # + dt omega x v), the omega x v term of the MuJoCo-frame acceleration), the
        # GLOBAL additive base-position step and the quaternion xyzw->wxyz. s_q and
        # s_qd still hold step k (the kernel integrates OUT-of-place into s_x_kp1).
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_integrator_output("s_x_kp1", "s_q", "s_qd", "dt", "IT == IntegratorType::SEMI_IMPLICIT_EULER", nq)
            self.gen_add_end_control_flow()
        self.gen_kernel_save_result("x_kp1",str(out_count),stride=str(out_count))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q_qd_u",str(input_count))
        if spill_minv_F:
            self.gen_add_code_line(gen_workspace_repoint_line("int_d_workspace", "GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()", declare=True))
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
        wrap_host_single_call_timing(func_call_code)
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"integrator\", INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    # Pin the spilled Minv-F section in L2 when any tier spills it.
    if not single_call_timing:
        self.gen_add_workspace_slot_count()
    workspace_bytes = ("GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()" if single_call_timing
                       else "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(_grid_ws_n)")
    self.gen_add_code_line("if (GRID_INTEGRATOR_USES_WORKSPACE) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, " + workspace_bytes + "));}")
    if single_call_timing:
        self.gen_add_code_lines(func_call_code)
    else:
        self.gen_add_workspace_clamped_launch(func_call_code, emit_count = False)
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
    # gen_lie_group_helpers is idempotent, so this is safe whether or not the
    # d2ee kinematic codegen already emitted them.
    if self.robot.floating_base:
        # Floating-base: emit the full SE(3) Lie bundle (the q-update Lie retract
        # + dIntegrate/d2Integrate blocks). For a floating robot that ALSO has a
        # spherical joint, gen_lie_group_helpers additionally emits the spherical
        # SO(3) wrapper (gated inside it on robot_has_spherical()).
        self.gen_lie_group_helpers()
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

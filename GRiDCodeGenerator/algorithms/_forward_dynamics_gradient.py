def gen_forward_dynamics_gradient_inner_temp_mem_size(self, use_qdd_Minv_input = False):
    n = self.robot.get_num_vel()
    minv_temp = self.gen_minv_inner_temp_mem_size()
    inverse_dynamics_gradient_temp = self.gen_inverse_dynamics_gradient_inner_temp_mem_size()
    return max(minv_temp,inverse_dynamics_gradient_temp) if not use_qdd_Minv_input else inverse_dynamics_gradient_temp

def gen_forward_dynamics_gradient_inner_python(self, use_qdd_Minv_input = False,
                                               s_df_du_name = "s_df_du",
                                               d_temp_spill_name = "nullptr",
                                               temp_spill_flag_name = "false",
                                               d_f_ext_name = "d_f_ext"):
    n = self.robot.get_num_vel()
    if not use_qdd_Minv_input:
        #
        # TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
        #       but that requires a custom function to be written
        #
        self.gen_add_code_line("//TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed")
        # Inner-controlled placement: minv_inner slices its own F-region
        # from the tail of s_temp (FD_DU keeps Minv-F in smem; its surgical spill
        # is the inverse_dynamics_gradient da_df band, handled separately). After Minv returns, the
        # c+vaf/ID code reuses these bytes (the steps run sequentially).
        self.gen_minv_inner_function_call(f_in_smem_expr = "true")
        # updated_var_names = dict(s_c_name = "s_temp", s_vaf_name = "&s_temp[" + str(n) + "]", s_temp_name = "&s_temp[" + str(19*n) + "]")
        updated_var_names = dict(s_c_name = "s_temp", s_temp_name = "&s_temp[" + str(n) + "]", d_f_ext_name = d_f_ext_name)
        self.gen_inverse_dynamics_inner_function_call(compute_c = True, use_qdd_input = False, updated_var_names = updated_var_names)
        self.gen_forward_dynamics_finish_function_call(updated_var_names)
        self.gen_add_sync()
        self.gen_inverse_dynamics_inner_function_call(compute_c = False, use_qdd_input = True, updated_var_names = dict(d_f_ext_name = d_f_ext_name))
    # else just compute vaf
    else:
        self.gen_inverse_dynamics_inner_function_call(compute_c = False, use_qdd_input = True, updated_var_names = dict(d_f_ext_name = d_f_ext_name))
    # then run the gradient code
    self.gen_inverse_dynamics_gradient_inner_function_call(
        dict(d_temp_spill_name = d_temp_spill_name, temp_spill_flag_name = temp_spill_flag_name)
    )

    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_lines(["printf(\"Minv\\n\");", \
                                 "printMat<T," + str(n) + "," + str(n) + ">(s_Minv," + str(n) + ");", \
                                 "printf(\"qdd\\n\");", \
                                 "printMat<T,1," + str(n) + ">(s_qdd,1);", \
                                 "printf(\"v\\n\");", \
                                 "printMat<T,6," + str(n) + ">(s_vaf,6);", \
                                 "printf(\"a\\n\");", \
                                 "printMat<T,6," + str(n) + ">(&s_vaf[6*" + str(n) + "],6);", \
                                 "printf(\"f\\n\");", \
                                 "printMat<T,6," + str(n) + ">(&s_vaf[12*" + str(n) + "],6);", \
                                 "printf(\"dc/dq\\n\");", \
                                 "printMat<T," + str(n) + "," + str(n) + ">(&s_dc_du[0]," + str(n) + ");", \
                                 "printf(\"dc/dqd\\n\");", \
                                 "printMat<T," + str(n) + "," + str(n) + ">(&s_dc_du[" + str(n*n) + "]," + str(n) + ");"])
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    # and finally finish with df/du = -Minv*dc/du
    self.gen_minv_apply(
        n, s_df_du_name + "[ind]", "s_dc_du[dc_col_offset + col]",
        loop_var = "ind", loop_max = str(n*2*n),
        pre_lines = ["int row = ind % " + str(n) + "; int dc_col_offset = ind - row;"],
        comment_in_loop = False, negate = True)

def gen_forward_dynamics_gradient_device_function_call(self,
                                                           use_qdd_Minv_input = False,
                                                           scratch_in_smem_expr = "true",
                                                           use_da_df_spill_expr = "false",
                                                           s_df_du_name = "s_df_du",
                                                           d_workspace_pool_name = "nullptr",
                                                           d_temp_spill_name = "nullptr",
                                                           d_f_ext_name = "d_f_ext",
                                                           mujoco_output_expr = None):
    """Emit the call to `forward_dynamics_gradient_device`. Arg order MUST
    match the def in gen_forward_dynamics_gradient_device. The caller decides
    where the OUTPUT s_df_du lives (smem buffer or the global d_df_du band) and
    hands in the pool/spill regions; these default to nullptr (unused under the
    matching if-constexpr). The _qdd C++ name variant additionally threads the
    caller-provided s_qdd / s_Minv inputs.
    `mujoco_output_expr` (floating + u-input variant only) appends the trailing
    MUJOCO_OUTPUT template arg; None keeps the legacy 3-arg template (byte-
    identical for non-mjx call sites)."""
    fname = "forward_dynamics_gradient_device_qdd" if use_qdd_Minv_input else "forward_dynamics_gradient_device"
    tmpl = "<T, " + scratch_in_smem_expr + ", " + use_da_df_spill_expr + \
           ((", " + mujoco_output_expr) if mujoco_output_expr is not None else "") + ">"
    start = fname + tmpl + "(" + s_df_du_name + ", s_q, s_qd, "
    if use_qdd_Minv_input:
        start += "s_qdd, s_Minv, "
    else:
        start += "s_u, "
    start += "s_vaf, s_dc_du, s_qdd, s_Minv, "
    middle = self.gen_insert_helpers_function_call()
    end = ("s_temp, " + d_workspace_pool_name + ", " + d_temp_spill_name + ", "
           + "d_robotModel, " + d_f_ext_name + ", gravity);")
    self.gen_add_code_line(start + middle + end)

def _emit_forward_dynamics_gradient_mjx_output(self):
    """Emit the MuJoCo (mjx) output-convention epilogue for the fd-gradient, in
    place on ``s_df_du`` (2*nv*nv col-major: df_dq at [0,nv*nv), df_dqd at
    [nv*nv,2*nv*nv); element [row,col] -> [col*nv + row]). Floating-base only.

    This runs INSIDE forward_dynamics_gradient_device, AFTER the whole
    forward_dynamics_gradient inner (so s_df_du = -Minv*dc/du is already final and
    the in-flight ``s_Minv`` / ``s_qdd`` smem buffers are still live). Unlike the
    id-gradient (which rebuilt M via crba_inner into a dead pool), the fd-gradient
    ALREADY holds the inverse mass matrix ``s_Minv`` (nv x nv col-major, the
    minv_inner output reused for -Minv*dc/du) and the computed acceleration
    ``s_qdd`` (forward_dynamics_finish output) -- so we use both directly with NO
    scratch reuse. Sources: R from the (already xyzw-reordered) base quaternion
    s_q[3..6]; v_lin=s_qd[0:3], omega=s_qd[3:6]; u_lin=s_u[0:3] (applied force,
    pin frame after the input-convert); qdd_lin=s_qdd[0:3] (base-linear of the
    computed forward-dynamics acceleration, generalized/pin ordering).

    The exact linear transform recipe (validated to 3.6e-14 vs fd_gradient_pin_to
    _mjx -- see /tmp/proto_fdgrad_mjx_cindex.py) is reproduced op-for-op below."""
    nv = self.robot.get_num_vel()
    DQ = 0
    DQD = nv * nv
    self.gen_add_code_line("// === mjx output convention (floating-base fd-gradient) ===")
    # PARALLELIZED across the block (was single-thread; ~50% runtime overhead at
    # large batch). Mirrors the validated id-gradient parallelization: the work is
    # split into phases, each a parallel loop over an independent axis, with a sync
    # between any two phases that have a read-after-write dependency. The loop-
    # invariant helpers (R, the base source vectors v_lin/omega/u_lin/qdd_lin, and
    # s_M) are RE-MATERIALIZED register-local at the top of every parallel-loop body
    # via the _LOAD block below — ALL their sources (s_q/s_qd/s_u/s_qdd/s_Minv) stay
    # live in smem through the epilogue, so per-thread recompute is correct with
    # ZERO aliasing risk (no smem staging). The op order within each accumulation is
    # byte-identical to the validated serial version.
    #
    # _LOAD: re-materialize R + the base vectors + s_M at the top of each parallel
    #   body. Build R (row-major R[3*i+j]) from the xyzw base quaternion (s_q already
    #   reordered by the input epilogue) -- mirrors mujoco_convention.rotation_from
    #   _quat_xyzw / the _gen_mjx_build_R_lines helper exactly.
    _LOAD = [
        "T qx = s_q[3], qy = s_q[4], qz = s_q[5], qw = s_q[6];",
        "T xx = qx*qx, yy = qy*qy, zz = qz*qz;",
        "T xy = qx*qy, xz = qx*qz, yz = qy*qz, wx = qw*qx, wy = qw*qy, wz = qw*qz;",
        "T R[9];",
        "R[0] = static_cast<T>(1) - static_cast<T>(2)*(yy+zz); R[1] = static_cast<T>(2)*(xy-wz);                    R[2] = static_cast<T>(2)*(xz+wy);",
        "R[3] = static_cast<T>(2)*(xy+wz);                    R[4] = static_cast<T>(1) - static_cast<T>(2)*(xx+zz); R[5] = static_cast<T>(2)*(yz-wx);",
        "R[6] = static_cast<T>(2)*(xz-wy);                    R[7] = static_cast<T>(2)*(yz+wx);                    R[8] = static_cast<T>(1) - static_cast<T>(2)*(xx+yy);",
        # Minv (nv x nv): in-flight from minv_inner (the -Minv*dc/du step). NOTE
        # minv_inner stores Minv SYMMETRIC_UPPER (only the upper triangle is
        # populated; the -Minv*dc/du apply mirror-indexes it). The Minv[:,0:3]
        # coupling below needs the FULL dense column (all nv rows), so we read it
        # through the same (row<=col) symmetric index used by the apply loop --
        # see the (r<=k) index below. (A naive dense s_Minv[r+nv*k] read would
        # return 0 for the lower-triangle rows r>k, zeroing the joint-row coupling.)
        "const T *s_M = s_Minv;",
        # sources (pin frame, post input-convert)
        "T v_lin[3] = {s_qd[0], s_qd[1], s_qd[2]};",
        "T omega[3] = {s_qd[3], s_qd[4], s_qd[5]};",
        "T u_lin[3] = {s_u[0], s_u[1], s_u[2]};",
        # qdd_lin = base-linear of the computed forward-dynamics accel s_qdd[0:3]
        "T qdd_lin[3] = {s_qdd[0], s_qdd[1], s_qdd[2]};",
    ]
    # ---- df_dq half ----
    # Phase A: df_dq column_reframe (cols 0:3, row r) THEN df_dq base couplings
    #   (cols 3+a, row r). Disjoint df_dq column writes (0:3 vs 3:6), both per-row;
    #   the couplings read df_dqd ORIGINAL + s_M. Parallel over r.
    self.gen_add_code_line("// Phase A: df_dq column_reframe (cols 0:3) + base couplings (cols 3+a), per row")
    self.gen_add_parallel_loop("r", str(nv))
    self.gen_add_code_lines(_LOAD)
    self.gen_add_code_lines([
        "// df_dq: column_reframe cols 0:3 <- cols . R^T",
        "T c0 = s_df_du[" + str(DQ) + " + 0*" + str(nv) + " + r], c1 = s_df_du[" + str(DQ) + " + 1*" + str(nv) + " + r], c2 = s_df_du[" + str(DQ) + " + 2*" + str(nv) + " + r];",
        "s_df_du[" + str(DQ) + " + 0*" + str(nv) + " + r] = c0*R[0] + c1*R[1] + c2*R[2];",
        "s_df_du[" + str(DQ) + " + 1*" + str(nv) + " + r] = c0*R[3] + c1*R[4] + c2*R[5];",
        "s_df_du[" + str(DQ) + " + 2*" + str(nv) + " + r] = c0*R[6] + c1*R[7] + c2*R[8];",
    ])
    self.gen_add_code_line("// df_dq: base-velocity/force couplings into cols 3+a (this row)")
    self.gen_add_code_line("//   df_dq[r,3+a] += df_dqd[r,0:3] @ (-(e_a x v_lin)) + Minv[r,0:3] @ (-(e_a x u_lin))")
    self.gen_add_code_line("for (int a = 0; a < 3; a++) {", True)
    self.gen_add_code_lines([
        # jv = -(e_a x v_lin); ju = -(e_a x u_lin). Basis-vector cross (verified
        # == numpy cross(e_a, .) in /tmp/proto_fdgrad_mjx_cindex.py).
        "T jv[3], ju[3];",
        "jv[0] = -((a==1)*( v_lin[2]) + (a==2)*(-v_lin[1]));",
        "jv[1] = -((a==0)*(-v_lin[2]) + (a==2)*( v_lin[0]));",
        "jv[2] = -((a==0)*( v_lin[1]) + (a==1)*(-v_lin[0]));",
        "ju[0] = -((a==1)*( u_lin[2]) + (a==2)*(-u_lin[1]));",
        "ju[1] = -((a==0)*(-u_lin[2]) + (a==2)*( u_lin[0]));",
        "ju[2] = -((a==0)*( u_lin[1]) + (a==1)*(-u_lin[0]));",
        "T acc = static_cast<T>(0);",
        # s_M is SYMMETRIC_UPPER: Minv[r,k] = s_M[(r<=k)?(k*nv+r):(r*nv+k)].
        "for (int k = 0; k < 3; k++) { T mrk = s_M[(r <= k) ? (k*" + str(nv) + " + r) : (r*" + str(nv) + " + k)]; acc += s_df_du[" + str(DQD) + " + k*" + str(nv) + " + r]*jv[k] + mrk*ju[k]; }",
        "s_df_du[" + str(DQ) + " + (3+a)*" + str(nv) + " + r] += acc;",
    ])
    self.gen_add_end_control_flow()  # for a
    self.gen_add_end_control_flow()  # parallel r
    self.gen_add_sync()
    # Phase B: df_dq base_rotate_rows (rows 0:3 of col c). Reads df_dq rows 0:3 of
    #   ALL cols (cols 0:3 written by A's reframe, cols 3+a by A's couplings) ->
    #   MUST follow A. Parallel over c.
    self.gen_add_code_line("// Phase B: df_dq base_rotate_rows rows 0:3 <- R . rows, per col")
    self.gen_add_parallel_loop("c", str(nv))
    self.gen_add_code_lines(_LOAD)
    self.gen_add_code_lines([
        "T m0 = s_df_du[" + str(DQ) + " + c*" + str(nv) + " + 0], m1 = s_df_du[" + str(DQ) + " + c*" + str(nv) + " + 1], m2 = s_df_du[" + str(DQ) + " + c*" + str(nv) + " + 2];",
        "s_df_du[" + str(DQ) + " + c*" + str(nv) + " + 0] = R[0]*m0 + R[1]*m1 + R[2]*m2;",
        "s_df_du[" + str(DQ) + " + c*" + str(nv) + " + 1] = R[3]*m0 + R[4]*m1 + R[5]*m2;",
        "s_df_du[" + str(DQ) + " + c*" + str(nv) + " + 2] = R[6]*m0 + R[7]*m1 + R[8]*m2;",
    ])
    self.gen_add_end_control_flow()  # parallel c
    self.gen_add_sync()
    # Phase C: df_dq accel output-map (base-linear rows 0:3 of cols 3+a). RMW the
    #   rows B just rotated -> MUST follow B. Parallel over a.
    self.gen_add_code_line("// Phase C: df_dq accel output-map terms (base-linear rows of cols 3+a), AFTER row-rotate")
    self.gen_add_code_line("//   += R@(e_a x (qdd_lin + omega x v_lin)) + R@(omega x (-(e_a x v_lin))), per a")
    self.gen_add_parallel_loop("a", "3")
    self.gen_add_code_lines(_LOAD)
    self.gen_add_code_lines([
        # ov = omega x v_lin ; t = qdd_lin + ov
        "T ov0 = omega[1]*v_lin[2] - omega[2]*v_lin[1];",
        "T ov1 = omega[2]*v_lin[0] - omega[0]*v_lin[2];",
        "T ov2 = omega[0]*v_lin[1] - omega[1]*v_lin[0];",
        "T t0 = qdd_lin[0] + ov0, t1 = qdd_lin[1] + ov1, t2 = qdd_lin[2] + ov2;",
        # out_R = e_a x t
        "T eR0 = (a==1)*( t2) + (a==2)*(-t1);",
        "T eR1 = (a==0)*(-t2) + (a==2)*( t0);",
        "T eR2 = (a==0)*( t1) + (a==1)*(-t0);",
        # ev = e_a x v_lin ; out_vq = omega x (-ev)
        "T ev0 = (a==1)*( v_lin[2]) + (a==2)*(-v_lin[1]);",
        "T ev1 = (a==0)*(-v_lin[2]) + (a==2)*( v_lin[0]);",
        "T ev2 = (a==0)*( v_lin[1]) + (a==1)*(-v_lin[0]);",
        "T ovq0 = omega[1]*(-ev2) - omega[2]*(-ev1);",
        "T ovq1 = omega[2]*(-ev0) - omega[0]*(-ev2);",
        "T ovq2 = omega[0]*(-ev1) - omega[1]*(-ev0);",
        # g = out_R + out_vq ; rows += R @ g
        "T g0 = eR0 + ovq0, g1 = eR1 + ovq1, g2 = eR2 + ovq2;",
        "s_df_du[" + str(DQ) + " + (3+a)*" + str(nv) + " + 0] += R[0]*g0 + R[1]*g1 + R[2]*g2;",
        "s_df_du[" + str(DQ) + " + (3+a)*" + str(nv) + " + 1] += R[3]*g0 + R[4]*g1 + R[5]*g2;",
        "s_df_du[" + str(DQ) + " + (3+a)*" + str(nv) + " + 2] += R[6]*g0 + R[7]*g1 + R[8]*g2;",
    ])
    self.gen_add_end_control_flow()  # parallel a
    self.gen_add_sync()
    # ---- df_dqd half ----
    # Phase D: df_dqd column_reframe (cols 0:3, row r). MUST follow Phase A (which
    #   read df_dqd ORIGINAL cols 0:3); the intervening syncs guarantee it. Parallel
    #   over r.
    self.gen_add_code_line("// Phase D: df_dqd column_reframe cols 0:3 <- cols . R^T, per row")
    self.gen_add_parallel_loop("r", str(nv))
    self.gen_add_code_lines(_LOAD)
    self.gen_add_code_lines([
        "T c0 = s_df_du[" + str(DQD) + " + 0*" + str(nv) + " + r], c1 = s_df_du[" + str(DQD) + " + 1*" + str(nv) + " + r], c2 = s_df_du[" + str(DQD) + " + 2*" + str(nv) + " + r];",
        "s_df_du[" + str(DQD) + " + 0*" + str(nv) + " + r] = c0*R[0] + c1*R[1] + c2*R[2];",
        "s_df_du[" + str(DQD) + " + 1*" + str(nv) + " + r] = c0*R[3] + c1*R[4] + c2*R[5];",
        "s_df_du[" + str(DQD) + " + 2*" + str(nv) + " + r] = c0*R[6] + c1*R[7] + c2*R[8];",
    ])
    self.gen_add_end_control_flow()  # parallel r
    self.gen_add_sync()
    # Phase E: df_dqd base_rotate_rows (rows 0:3 of col c). Reads df_dqd rows 0:3 of
    #   cols 0:3 written by D -> MUST follow D. Parallel over c.
    self.gen_add_code_line("// Phase E: df_dqd base_rotate_rows rows 0:3 <- R . rows, per col")
    self.gen_add_parallel_loop("c", str(nv))
    self.gen_add_code_lines(_LOAD)
    self.gen_add_code_lines([
        "T m0 = s_df_du[" + str(DQD) + " + c*" + str(nv) + " + 0], m1 = s_df_du[" + str(DQD) + " + c*" + str(nv) + " + 1], m2 = s_df_du[" + str(DQD) + " + c*" + str(nv) + " + 2];",
        "s_df_du[" + str(DQD) + " + c*" + str(nv) + " + 0] = R[0]*m0 + R[1]*m1 + R[2]*m2;",
        "s_df_du[" + str(DQD) + " + c*" + str(nv) + " + 1] = R[3]*m0 + R[4]*m1 + R[5]*m2;",
        "s_df_du[" + str(DQD) + " + c*" + str(nv) + " + 2] = R[6]*m0 + R[7]*m1 + R[8]*m2;",
    ])
    self.gen_add_end_control_flow()  # parallel c
    self.gen_add_sync()
    # Phase F: df_dqd out_vd accel-map (base-linear rows 0:3 of cols 0+a and 3+a).
    #   RMW the rows E just rotated -> MUST follow E. Parallel over a.
    self.gen_add_code_line("// Phase F: df_dqd out_vd accel-map terms (base-linear rows), AFTER row-rotate")
    self.gen_add_code_line("//   col 0+a += R@(omega x R^T e_a) ; col 3+a += R@(e_a x v_lin), per a")
    self.gen_add_parallel_loop("a", "3")
    self.gen_add_code_lines(_LOAD)
    self.gen_add_code_lines([
        # R^T e_a = row a of R = (R[3a+0],R[3a+1],R[3a+2])
        "T rte0 = R[3*a + 0], rte1 = R[3*a + 1], rte2 = R[3*a + 2];",
        # d = omega x R^T e_a ; col 0+a rows += R @ d
        "T d0 = omega[1]*rte2 - omega[2]*rte1;",
        "T d1 = omega[2]*rte0 - omega[0]*rte2;",
        "T d2 = omega[0]*rte1 - omega[1]*rte0;",
        "s_df_du[" + str(DQD) + " + (0+a)*" + str(nv) + " + 0] += R[0]*d0 + R[1]*d1 + R[2]*d2;",
        "s_df_du[" + str(DQD) + " + (0+a)*" + str(nv) + " + 1] += R[3]*d0 + R[4]*d1 + R[5]*d2;",
        "s_df_du[" + str(DQD) + " + (0+a)*" + str(nv) + " + 2] += R[6]*d0 + R[7]*d1 + R[8]*d2;",
        # ev = e_a x v_lin ; col 3+a rows += R @ ev
        "T ev0 = (a==1)*( v_lin[2]) + (a==2)*(-v_lin[1]);",
        "T ev1 = (a==0)*(-v_lin[2]) + (a==2)*( v_lin[0]);",
        "T ev2 = (a==0)*( v_lin[1]) + (a==1)*(-v_lin[0]);",
        "s_df_du[" + str(DQD) + " + (3+a)*" + str(nv) + " + 0] += R[0]*ev0 + R[1]*ev1 + R[2]*ev2;",
        "s_df_du[" + str(DQD) + " + (3+a)*" + str(nv) + " + 1] += R[3]*ev0 + R[4]*ev1 + R[5]*ev2;",
        "s_df_du[" + str(DQD) + " + (3+a)*" + str(nv) + " + 2] += R[6]*ev0 + R[7]*ev1 + R[8]*ev2;",
    ])
    self.gen_add_end_control_flow()  # parallel a
    self.gen_add_sync()

def gen_forward_dynamics_gradient_device(self, use_qdd_Minv_input = False):
    """Emit `forward_dynamics_gradient_device` — the whole forward_dynamics_gradient orchestration
    as ONE inner that OWNS its scratch (s_temp) placement (inner-owns-placement;
    mirrors gen_inverse_dynamics_gradient_device / gen_fdsva_so_device). It
    wraps, in order:
      [repoint s_temp] -> load_update_XImats -> minv_inner (f_in_smem=true)
      -> inverse_dynamics_inner (c+vaf) -> forward_dynamics_finish -> id_inner (vaf)
      -> inverse_dynamics_gradient_inner (the inverse_dynamics_gradient BAND sub-inner) -> df/du = -Minv*dc/du.
    Because the s_temp repoint happens at the very top, EVERY consumer below —
    including the XImats helper's sincos scratch and minv's own F-region —
    follows the placement, so the kernel never repoints s_temp from the outside.

    TWO independent template flags:
      SCRATCH_IN_SMEM  : the shared s_temp pool lives in smem (true) or routes the
                         WHOLE pool to d_workspace (false; the rung-2 global-temp
                         path).
      USE_DA_DF_SPILL  : the inverse_dynamics_gradient band selectively spills its da_dq..fxvi band to
                         d_temp_spill (rung 1). Threaded through to the BAND
                         sub-inner's grid_id_du_temp_ptr<T, USE_DA_DF_SPILL> helper.
    The 3-rung menu (see _FD_DU_PICK_FLAGS): pick0=(SMEM=true, SPILL=false) full;
    pick1=(true, true) selective band; pick2=(false, false) whole-pool global.

    Pointer params are caller-supplied. The composed sub-inners (minv_inner,
    inverse_dynamics_inner, inverse_dynamics_gradient_inner) are FROZEN and
    placement-free: after the repoint, s_temp already points at the right pool, so
    passing it through is correct with no sub-inner change. The internal Minv/qdd
    smem buffers (s_Minv, s_qdd) are caller-placed too — in the qdd-input variant
    the caller supplies them as inputs; otherwise they are scratch outputs."""
    n = self.robot.get_num_vel()
    func_params = [
        "s_df_du is the output buffer (caller places); size 2*NUM_VEL*NUM_VEL = " + str(2*n*n),
        "s_q is the vector of joint positions",
        "s_qd is the vector of joint velocities",
    ]
    if use_qdd_Minv_input:
        func_params += [
            "s_qdd is the vector of joint accelerations (input)",
            "s_Minv is the mass matrix (input)",
        ]
    else:
        func_params.append("s_u is the vector of input torques")
    func_params += [
        "s_vaf is the id intermediate band (caller places); size 18*NUM_JOINTS = " + str(18*n),
        "s_dc_du is the inverse_dynamics_gradient output band (caller places); size 2*NUM_VEL*NUM_VEL = " + str(2*n*n),
        "s_qdd is the joint-accel scratch (caller places); size NUM_JOINTS = " + str(n),
        "s_Minv is the mass-matrix scratch (caller places); size NUM_VEL*NUM_VEL = " + str(n*n),
        "s_temp is the shared scratch pool (used when SCRATCH_IN_SMEM)",
        "d_workspace is the global scratch pool (used when !SCRATCH_IN_SMEM)",
        "d_temp_spill is the inverse_dynamics_gradient da_df band spill region (used when USE_DA_DF_SPILL)",
        "d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr",
        "d_robotModel holds XImats/topology; gravity is the gravity constant",
    ]
    fname = "forward_dynamics_gradient_device_qdd" if use_qdd_Minv_input else "forward_dynamics_gradient_device"
    func_def_start = "void " + fname + "(T *s_df_du, const T *s_q, const T *s_qd, "
    if use_qdd_Minv_input:
        func_def_start += "const T *s_qdd, const T *s_Minv, "
        func_def_start += "T *s_vaf, T *s_dc_du, const T *s_qdd_unused, const T *s_Minv_unused, "
    else:
        func_def_start += "const T *s_u, "
        func_def_start += "T *s_vaf, T *s_dc_du, T *s_qdd, T *s_Minv, "
    func_def_end = ("T *s_temp, T *d_workspace, T *d_temp_spill, "
                    "const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity) {")
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -2)
    func_def = func_def_start + func_def_end
    self.gen_add_func_doc("forward_dynamics_gradient orchestration as a single inner-owns-placement device function",
                          ["Uses the fd/du = -Minv*id/du trick (Carpentier & Mansard 'Analytical Derivatives of Rigid Body Dynamics Algorithms')",
                           "Owns the s_temp pool placement; the repoint covers every consumer below (incl. the XImats helper's sincos scratch and minv's F-region)"],
                          func_params, None)
    # MUJOCO_OUTPUT (floating + non-mimic/skew, u-input variant only): compile-time
    # mjx output-convention flag. Added LAST so existing positional <T,SCRATCH,SPILL>
    # call sites are unaffected; default false -> the epilogue if-constexpr-elides to
    # byte-identical PTX. Attached only to the u-input surface (use_qdd_Minv_input=
    # False): mjx fd-grad takes (q,qd,u) and computes qdd/Minv internally -- the
    # epilogue needs the COMPUTED s_qdd, which only that variant produces. The
    # epilogue runs HERE (not in the kernel body) because it needs the in-flight
    # s_Minv (minv_inner) + s_qdd (forward_dynamics_finish) smem buffers.
    mjx_device = (self.robot.floating_base and not use_qdd_Minv_input
                  and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis()))
    if mjx_device:
        self.gen_add_code_line("template <typename T, bool SCRATCH_IN_SMEM = true, bool USE_DA_DF_SPILL = false, bool MUJOCO_OUTPUT = false>")
    else:
        self.gen_add_code_line("template <typename T, bool SCRATCH_IN_SMEM = true, bool USE_DA_DF_SPILL = false>")
    # __forceinline__ so the whole orchestration inlines into the calling kernel.
    # Under -rdc a separate __device__ wrapper keeps its callees as distinct
    # functions whose regcount must fit the kernel's launch_bounds budget -> ptxas
    # regcount error. Inlining folds them into the kernel. Mirrors inverse_dynamics_gradient / fdsva_so.
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line(func_def, True)
    if use_qdd_Minv_input:
        self.gen_add_code_line("(void)s_qdd_unused; (void)s_Minv_unused;")
    # Inner owns the pool placement; the repoint covers every consumer below
    # (incl. the XImats helper's sincos scratch + minv's F-region), so no
    # caller-side repoint. The XImats helper call goes AFTER this repoint so its
    # sincos scratch follows the placement (avoids a null-s_temp sincos crash).
    self.gen_add_code_line("if constexpr(!SCRATCH_IN_SMEM){ s_temp = d_workspace; } else { (void)d_workspace; }")
    self.gen_load_update_XImats_helpers_function_call()
    self.gen_forward_dynamics_gradient_inner_python(
        use_qdd_Minv_input,
        s_df_du_name = "s_df_du",
        d_temp_spill_name = "d_temp_spill",
        temp_spill_flag_name = "USE_DA_DF_SPILL")
    if mjx_device:
        self.gen_add_sync()
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        _emit_forward_dynamics_gradient_mjx_output(self)
        self.gen_add_end_control_flow()
    self.gen_add_end_function()

def gen_forward_dynamics_gradient_kernel_max_temp_mem_size(self):
    n = self.robot.get_num_vel()
    nq = self.robot.get_num_pos()
    vaf_cnt = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else n)
    # Mirror the kernel body's extra_t_buffers (the non-qdd variant is the
    # largest input buffer): s_q_qd_u uses the canonical 3*nq slot (NUM_JOINTS-
    # wide q/qd/u), the gradient bands are tangent-space (nv). For a FIXED base
    # nq==nv so 3*nq == old 3*n byte-identical; for a FLOATING base the +2*(nq-nv)
    # here grows the dynamic-smem launch param to fit the wider input slot (else
    # the kernel's s_q_qd_u load overflows the arena -> illegal __shared__ write).
    base_size = 3*nq + n*2*n + n*2*n + vaf_cnt + n + n*n + n
    temp_mem_size = self.gen_forward_dynamics_gradient_inner_temp_mem_size()
    return base_size + temp_mem_size

_FD_DU_PICK_FLAGS = [
    # (use_selective_spill, use_global_temp, use_output_spill)
    (False, False, False),   # pick 0: full smem
    (True,  False, False),   # pick 1: selective spill (id_du da_df band -> d_temp_spill)
    (False, True,  False),   # pick 2: global temp (whole inner pool -> d_workspace GRAD section)
    (False, True,  True),    # pick 3: output-spill (global temp + s_dc_du/s_Minv -> L2-pinned SO band)
]

def _emit_forward_dynamics_gradient_kernel_body_for_flags(self, nq, nv, use_selective_spill, use_global_temp,
                                      use_output_spill, use_qdd_Minv_input, single_call_timing, mjx_kernel = False):
    """Emit forward_dynamics_gradient kernel body for one tier's spill flags.
    `mjx_kernel` (floating + non-mimic/skew, u-input variant): emit the
    MUJOCO_OUTPUT input-convert (before the device fn builds XImats) and forward
    the flag to the device call (the output epilogue lives inside the device fn)."""
    n = nv
    # Canonical per-timestep packing (mirrors id/crba/aba/forward_dynamics): q, qd,
    # u each occupy a NUM_JOINTS(=nq)-wide slot at stride 3*nq; the qd slot starts
    # at nq, u at 2*nq. The qdd / Minv inputs (qdd-Minv variant) are stored at the
    # canonical nq / nq*nq per-timestep strides too (the host transfers them
    # NUM_JOINTS-wide). The intermediate s_qdd/s_Minv smem buffers hold nv real
    # values. For a FIXED base nq==nv so every offset below is byte-identical to
    # the old nv-strided form; for a FLOATING base nq>nv this reads qd/u from the
    # right slot (was off by nq-nv) and strides batched inputs correctly. The
    # df_du OUTPUT is genuinely tangent-space (nv x 2nv, dense 2*nv*nv-strided; the
    # d_df_du buffer + binding transfer are 2*nv*nv-sized) -- the input-slot fix is
    # the floating bug, the gradient output stride stays 2*nv*nv.
    # s_vaf is body-indexed (NB bodies, stride 6). For a MIMIC robot (fixed base)
    # NB > nv, so size 18*NB to keep the ID inner's writes from overflowing into
    # s_qdd/s_Minv. Non-mimic keeps 18*nv (byte-identical; floating nv > NB).
    _vaf_cnt = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else nv)
    # OUTPUT-spill rung (use_output_spill): the s_dc_du (2*nv*nv id-gradient band) and
    # s_Minv (nv*nv mass matrix) write-once OUTPUT buffers are dropped from the smem
    # arena and repointed to the L2-pinned SO band (crba/fdsva_so style), shrinking the
    # arena by 3*nv*nv. The hot inner pool already lives in d_workspace's GRAD section
    # (use_output_spill implies use_global_temp); the outputs sit in the disjoint SO
    # section. fd_du never runs concurrently with the SO/regressor kernels that also
    # reuse the SO band, so the placement is safe and costs no new allocation.
    extra_t_buffers = [("s_q_qd", 2*nq),
                       ("s_dc_du", nv*2*nv),
                       ("s_vaf", _vaf_cnt),
                       ("s_qdd", nv),
                       ("s_Minv", nv*nv)]
    if not use_qdd_Minv_input:
        extra_t_buffers[0] = ("s_q_qd_u", 3*nq)
    if use_output_spill:
        extra_t_buffers = [b for b in extra_t_buffers if b[0] not in ("s_dc_du", "s_Minv")]
    shared_mem_size = 0 if use_global_temp else (
        max(self.gen_minv_inner_temp_mem_size(), self.gen_inverse_dynamics_gradient_temp_layout()["selective_shared_count"])
        if use_selective_spill else self.gen_forward_dynamics_gradient_inner_temp_mem_size()
    )
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size, extra_t_buffers = extra_t_buffers, include_linalg_scratch=True)
    if use_output_spill:
        self.gen_add_code_line("T *s_dc_du; T *s_Minv;  // repointed to the L2-pinned SO band (output spill) per timing branch")
    self.gen_add_code_line("T *d_temp_spill = nullptr; (void)d_temp_spill;")
    if use_qdd_Minv_input:
        self.gen_add_code_line(f"T *s_q = s_q_qd; T *s_qd = &s_q_qd[{nq}];")
    else:
        self.gen_add_code_line(f"T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[{nq}]; T *s_u = &s_q_qd_u[{2*nq}];")
    if not single_call_timing:
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",block_level = True)
        if use_qdd_Minv_input:
            self.gen_kernel_load_inputs("q_qd",str(2*nq),"qdd",str(nv),"Minv",str(nv*nv),stride="stride_q_qd",stride2=str(nq),stride3=str(nq*nq))
        else:
            self.gen_kernel_load_inputs("q_qd_u",str(3*nq),stride="stride_q_qd_u")
        # mjx input convert (before the device fn builds XImats so X[0] uses the
        # reordered quaternion). Reorders the base quaternion + converts the base
        # velocity/applied-force to the pin frame in place on s_q/s_qd/s_u. qdd is
        # NOT converted here (it is computed inside the device fn).
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert(q_name="s_q", qd_name="s_qd", u_name="s_u")
            self.gen_add_end_control_flow()
        # The kernel only SLICES the workspace band pointers; the device owns
        # the s_temp pool placement (the whole-pool global-temp repoint is its
        # SCRATCH_IN_SMEM=false path). Per-rung flags are passed as literals.
        if use_selective_spill or use_global_temp:
            self.gen_add_code_line("T *d_df_du_k = &d_df_du[k*" + str(nv*2*nv) + "];")
        if use_selective_spill:
            self.gen_add_code_line("d_temp_spill = reinterpret_cast<T *>(&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()]);")
        if use_output_spill:
            # s_dc_du (2*nv*nv) + s_Minv (nv*nv) -> L2-pinned SO band, disjoint from the
            # inner pool which lives in the GRAD section (offset 0) under use_global_temp.
            self.gen_add_code_line("s_dc_du = reinterpret_cast<T *>(&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()]); s_Minv = &s_dc_du[" + str(2*nv*nv) + "];")
        self.gen_add_code_line("// compute — the orchestration inner owns its s_temp pool placement")
        self.gen_forward_dynamics_gradient_device_function_call(
            use_qdd_Minv_input,
            scratch_in_smem_expr = ("false" if use_global_temp else "true"),
            use_da_df_spill_expr = ("true" if use_selective_spill else "false"),
            s_df_du_name = ("d_df_du_k" if (use_global_temp or use_selective_spill) else "s_temp"),
            d_workspace_pool_name = ("reinterpret_cast<T *>(&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()])" if use_global_temp else "nullptr"),
            d_temp_spill_name = ("d_temp_spill" if use_selective_spill else "nullptr"),
            mujoco_output_expr = ("MUJOCO_OUTPUT" if mjx_kernel else None))
        if not (use_global_temp or use_selective_spill):
            self.gen_kernel_save_result("df_du",str(nv*2*nv),"s_temp",stride=str(nv*2*nv))
        self.gen_add_end_control_flow()
    else:
        if use_qdd_Minv_input:
            self.gen_kernel_load_inputs("q_qd",str(2*nq),"qdd",str(nv),"Minv",str(nv*nv))
        else:
            self.gen_kernel_load_inputs("q_qd_u",str(3*nq))
        if use_selective_spill or use_global_temp:
            self.gen_add_code_line("T *d_df_du_k = d_df_du;")
        if use_selective_spill:
            self.gen_add_code_line("d_temp_spill = reinterpret_cast<T *>(d_workspace);")
        if use_output_spill:
            # s_dc_du/s_Minv -> L2-pinned SO band (single-timing: no per-k stride).
            self.gen_add_code_line("s_dc_du = reinterpret_cast<T *>(&d_workspace[GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()]); s_Minv = &s_dc_du[" + str(2*nv*nv) + "];")
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        if use_qdd_Minv_input:
            self.gen_anti_licm_input_reload("q_qd", str(2*nq), "qdd", str(nv), "Minv", str(nv*nv))
        else:
            self.gen_anti_licm_input_reload("q_qd_u", str(3*nq))
        # device owns s_temp placement (whole-pool global path = SCRATCH_IN_SMEM=false).
        # Single-timing forwards MUJOCO_OUTPUT (perf only; the input-convert is
        # omitted here -- the anti-LICM loop reloads raw inputs each rep, so a one-
        # shot convert would not apply; correctness is validated via the full kernel).
        self.gen_forward_dynamics_gradient_device_function_call(
            use_qdd_Minv_input,
            scratch_in_smem_expr = ("false" if use_global_temp else "true"),
            use_da_df_spill_expr = ("true" if use_selective_spill else "false"),
            s_df_du_name = ("d_df_du_k" if (use_global_temp or use_selective_spill) else "s_temp"),
            d_workspace_pool_name = ("reinterpret_cast<T *>(d_workspace)" if use_global_temp else "nullptr"),
            d_temp_spill_name = ("d_temp_spill" if use_selective_spill else "nullptr"),
            mujoco_output_expr = ("MUJOCO_OUTPUT" if mjx_kernel else None))
        self.gen_add_code_line(
            "if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { "
            "reinterpret_cast<volatile T *>(d_df_du)[rep & 63] = "
            + ("d_df_du_k[rep & 63];" if (use_global_temp or use_selective_spill) else "s_temp[rep & 63];")
            + " }"
        )
        self.gen_add_end_control_flow()
        if not (use_global_temp or use_selective_spill):
            self.gen_kernel_save_result("df_du",str(nv*2*nv),"s_temp")


def gen_forward_dynamics_gradient_kernel(self, use_qdd_Minv_input = False, single_call_timing = False):
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    n = nv
    func_params = ["d_df_du is a pointer to memory for the final result of size 2*NUM_VEL*NUM_VEL = " + str(2*n*n), \
                   "d_q_dq is the vector of joint positions and velocities", \
                   "stride_q_qd is the stide between each q, qd", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr", \
                   "gravity is the gravity constant", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    func_def_start = "void forward_dynamics_gradient_kernel(T *d_df_du, unsigned char *d_workspace, const T *d_q_qd, const int stride_q_qd, "
    func_def_end = "T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    if use_qdd_Minv_input:
        func_def_start += "const T *d_qdd, "
        func_params.insert(-2,"d_qdd is the vector of joint accelerations")
        func_def_start += "const T *d_Minv, "
        func_params.insert(-2,"d_Minv is the mass matrix")
    else:
        func_def_start = func_def_start.replace("_q_qd","_q_qd_u")
        func_params[1] = "d_q_dq is the vector of joint positions, velocities, and input torques"
        func_params[2] = "stride_q_qd_u is the stide between each q, qd, u"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Computes the gradient of forward dynamics",func_notes,func_params,None)
    # MUJOCO_OUTPUT (floating + non-mimic/skew, u-input kernel only): compile-time
    # mjx output-convention flag. Added LAST after RESOURCE_TIER so existing
    # positional <T,TIER> call sites are unaffected; default false -> the input
    # convert + output epilogue if-constexpr-elide to byte-identical PTX. The
    # qdd-Minv-input kernel never carries it (mjx computes qdd/Minv internally).
    mjx_kernel = (self.robot.floating_base and not use_qdd_Minv_input
                  and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis()))
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    picks = getattr(self, "forward_dynamics_gradient_spill_tier_3way", (0, 0, 0))
    def _emit_forward_dynamics_gradient_body(pick):
        uss, ugt, uos = _FD_DU_PICK_FLAGS[pick]
        _emit_forward_dynamics_gradient_kernel_body_for_flags(self, nq, nv, uss, ugt, uos, use_qdd_Minv_input, single_call_timing, mjx_kernel)
    self.gen_tier_dispatch(picks, _emit_forward_dynamics_gradient_body)
    self.gen_add_end_function()

def gen_forward_dynamics_gradient_host(self, mode = 0):
    # default is to do the full kernel call -- options are for single timing or compute only kernel wrapper
    single_call_timing = True if mode == 1 else False
    compute_only = True if mode == 2 else False

    # define function def and params
    func_params = ["hd_data is the packaged input and output pointers", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant,", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)", \
                   "streams are pointers to CUDA streams for async memory transfers (if needed)"]
    func_notes = []
    func_def_start = "void forward_dynamics_gradient(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end =   "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    # then generate the code
    self.gen_add_func_doc("Compute the RNEA (Recursive Newton-Euler Algorithm)",\
                          func_notes,func_params,None)
    # MUJOCO_OUTPUT (floating + non-mimic/skew) host template flag: forwarded ONLY
    # to the u-input kernel launch (the mjx-capable overload). Added LAST so
    # existing positional template args are unaffected; default false -> byte-
    # identical. The qdd-Minv-input launch never carries it (mjx computes qdd/Minv
    # internally from u).
    mjx_host = (self.robot.floating_base
                and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis()))
    if mjx_host:
        self.gen_add_code_line("template <typename T, bool USE_QDD_MINV_FLAG = false, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    else:
        self.gen_add_code_line("template <typename T, bool USE_QDD_MINV_FLAG = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"forward_dynamics_gradient requires all-data or dynamics gridData\");")
    # the u-input launch is the mjx-capable overload: name the tier positionally
    # to reach the trailing MUJOCO_OUTPUT flag.
    u_kernel_tmpl = "forward_dynamics_gradient_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "forward_dynamics_gradient_kernel<T, RESOURCE_TIER>"
    func_call_start = u_kernel_tmpl + "<<<block_dimms,thread_dimms,FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_df_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,"
    # the qdd-Minv launch keeps the legacy non-mjx template (never mjx) but is tiered.
    func_call_qdd_start = "forward_dynamics_gradient_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_df_du,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,"
    func_call_end = "hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("forward_dynamics_gradient_kernel<","forward_dynamics_gradient_kernel_single_timing<")
        func_call_qdd_start = func_call_qdd_start.replace("forward_dynamics_gradient_kernel<","forward_dynamics_gradient_kernel_single_timing<")
    self.gen_add_code_line("int stride_q_qd= 3*NUM_JOINTS;")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                 "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));", \
                                 "if (USE_QDD_MINV_FLAG) {" ,\
                                 "    gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*" + \
                                        ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[1]));", \
                                 "    gpuErrchk(cudaMemcpyAsync(hd_data->d_Minv,hd_data->h_Minv,NUM_VEL*NUM_VEL*" + \
                                        ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[2]));", \
                                 "}", \
                                 "gpuErrchkKernel();"])
    # then compute
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    func_call_with_qdd_minv = func_call_qdd_start + "hd_data->d_qdd, hd_data->d_Minv, " + func_call_end
    func_call_code = ["if (USE_QDD_MINV_FLAG) {" + func_call_with_qdd_minv + "}", "else {" + func_call + "}", "gpuErrchkKernel();"]
    # wrap function call in timing (if needed)
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"forward_dynamics_gradient\", FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    workspace_bytes = "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()" if single_call_timing else "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(num_timesteps)"
    self.gen_add_code_line("if (GRID_FORWARD_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, " + workspace_bytes + "));}")
    self.gen_add_code_lines(func_call_code)
    self.gen_add_code_line("if (GRID_FORWARD_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_end_l2_persisting(0));}")
    if not compute_only:
        # then transfer memory back. df_du is the tangent-space gradient
        # (NUM_VEL x 2*NUM_VEL, dense 2*NUM_VEL*NUM_VEL-strided): the kernel writes
        # it 2*NV*NV-strided, so the host copy + h_df_du buffer are 2*NV*NV-strided
        # too. (Previously NUM_JOINTS*2*NUM_JOINTS, which mis-strided h_df_du for a
        # BATCHED floating base, nq>nv; byte-identical for fixed base nq==nv.)
        self.gen_add_code_lines(["// finally transfer the result back", \
                                 "gpuErrchk(cudaMemcpy(hd_data->h_df_du,hd_data->d_df_du,2*NUM_VEL*NUM_VEL*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchkKernel();"])
    # finally report out timing if requested
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("forward_dynamics_gradient"))
    self.gen_add_end_function()

def gen_forward_dynamics_gradient(self):
    # the canonical _device (orchestrator: owns s_temp placement; wraps XImats +
    # minv/id/finish/id + inverse_dynamics_gradient band sub-inner + (-Minv*dc/du); called from kernel
    # and integrator_gradient). Both qdd-Minv-input variants.
    self.gen_forward_dynamics_gradient_device(False)
    self.gen_forward_dynamics_gradient_device(True)
    # then kernels
    self.gen_forward_dynamics_gradient_kernel(True,True)
    self.gen_forward_dynamics_gradient_kernel(True,False)
    self.gen_forward_dynamics_gradient_kernel(False,True)
    self.gen_forward_dynamics_gradient_kernel(False,False)
    # finally host wrappers
    self.gen_forward_dynamics_gradient_host(0)
    self.gen_forward_dynamics_gradient_host(1)
    self.gen_forward_dynamics_gradient_host(2)

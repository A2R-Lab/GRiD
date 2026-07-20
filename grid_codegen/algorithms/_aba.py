def _aba_Svec_cpp(robot, jid):
    """C++ brace-init for body ``jid``'s dense 6-vector motion subspace (Tier-B
    skew emit)."""
    return "{" + ", ".join("static_cast<T>(" + repr(float(c)) + ")" for c in robot._get_flat_S_by_id(jid)) + "}"


# Tier-C spherical (ball) joint support for the standalone fixed-base ABA.
#
# A ball joint has a 3-DoF angular-identity motion subspace S (6x3: cols 0,1,2
# pick rows 0,1,2, sign +1), so the per-joint articulated-inertia D = S^T*IA*S
# becomes a 3x3 matrix that must be matrix-INVERTED (the scalar 1/d path does not
# generalize). It is NOT a skew joint (the signed-index helpers raise on multi-
# DoF), so it gets its own per-joint serial emit inside `level_has_spherical`
# branches; cardinal joints sharing the robot keep the scalar fast path but read
# their qd/tau/qdd through the spherical-shifted v/f maps (downstream of a ball
# joint every single-DoF body is shifted off `jid`; the §1e bug class).
#
# Per spherical joint we carve a contiguous scratch block out of a dedicated
# arena that sits ABOVE the cardinal 140*NJ band (so the cold-spill ladder
# offsets stay byte-identical for cardinal robots). Layout (offsets within the
# per-joint block, `_ABA_SPH_STRIDE` floats wide):
_ABA_SPH_D_OFF      = 0      # D = S^T*U (3x3, col-major)            [9]
_ABA_SPH_DINV_OFF   = 9      # D^-1 (3x3)                            [9]
_ABA_SPH_U_OFF      = 18     # U = IA*S (6x3, col-major: U[6*c + r]) [18]
_ABA_SPH_UDINV_OFF  = 36     # U*D^-1 (6x3, col-major)               [18]
_ABA_SPH_U3_OFF     = 54     # available joint torque u (3-vec)      [3]
_ABA_SPH_TMP3_OFF   = 57     # scratch 3-vec (u - U^T*a, etc.)       [3]
_ABA_SPH_INVTMP_OFF = 60     # invert_matrix scratch (>= 3*dimA=9)   [12]
_ABA_SPH_STRIDE     = 72     # total floats per spherical joint


def _aba_sph_arena_base(self):
    """First float offset of the spherical scratch arena (above the cardinal
    140*NJ hot band). Only meaningful when the robot has a spherical joint."""
    return 140 * self.robot.get_num_joints()


def _aba_sph_block_off(self, jid, sph_order):
    """Float offset of body `jid`'s spherical scratch block. `sph_order` is the
    0-based rank of this joint among the robot's spherical joints (each gets its
    own _ABA_SPH_STRIDE-wide block)."""
    return _aba_sph_arena_base(self) + sph_order * _ABA_SPH_STRIDE


def _aba_sph_orders(self):
    """Map joint id -> 0-based spherical-rank for every spherical joint."""
    order = {}
    for jid in range(self.robot.get_num_joints()):
        if self.robot.joint_is_spherical(jid):
            order[jid] = len(order)
    return order


def _aba_vslot(self, jid):
    """Single scalar reduced v/f-slot for a cardinal body `jid` (shifted past
    any upstream spherical joint). Byte-identical to `jid` on all-cardinal
    fixed-base robots (v-slot == jid there)."""
    v = self.robot.get_joint_index_v(jid)
    if isinstance(v, (list, tuple)):
        v = v[0]
    return v


def _aba_jd_bias_term_cpp(self, jid):
    """C++ subexpression for body ``jid``'s joint-local damping + Coulomb friction
    bias, or "" when there is none to apply.

    The ABA available-torque accumulation is ``u = tau - S^T*pA - U^T*c`` (matches
    RBDReference.aba: ``u[inds_v] = tau - dyn_bias - ...``), where ``dyn_bias`` is
    the SAME joint-local bias that gen_inverse_dynamics_joint_dynamics_bias folds
    into ``s_c``. The articulated inertia ``d``/``IA`` is left untouched (the
    explicit-value path puts damping only in the bias, exactly as the oracle does).
    Returned string is ``alpha*(b*qd + f*sign(qd))`` (reading ``s_qd[vs]``); callers
    subtract it from the per-joint ``s_tau`` read at each ``u`` site.

    EMITTED ONLY when USE_JOINT_DYNAMICS is enabled AND the robot declares nonzero
    damping/friction (both decided at codegen time). With the flag off (the DEFAULT)
    this returns "" for every joint, so the u-sites are byte-identical to the
    historical emit -- damped or not. Mirrors the ID emitter's gate, mimic alpha-fold,
    np.sign((qd>0)-(qd<0)), and literal-baked coefficients exactly.
    """
    if not getattr(self, "USE_JOINT_DYNAMICS", False):
        return ""
    if not (self.robot.robot_has_joint_damping() or self.robot.robot_has_joint_friction()):
        return ""
    if self.robot.floating_base and jid == 0:
        return ""  # floating root carries no damping/friction
    HAS_DAMP = self.robot.robot_has_joint_damping()
    HAS_FRIC = self.robot.robot_has_joint_friction()
    b = float(self.robot.get_damping_by_id(jid)) if HAS_DAMP else 0.0
    fr = float(self.robot.get_friction_by_id(jid)) if HAS_FRIC else 0.0
    if b == 0.0 and fr == 0.0:
        return ""
    if self.robot_has_mimic_joints():
        vs = self._v_slot_cpp(jid)
        alpha = float(self._alpha_for_jid(jid))
    else:
        vs = self.robot.get_joint_index_v(jid)
        alpha = 1.0
    qd = "s_qd[" + str(vs) + "]"
    terms = []
    if getattr(self, "runtime_joint_dynamics", False):
        # runtime_joint_dynamics: read the alpha-FOLDED per-v-slot coefficient from
        # the mutable device table (damping at [vs], friction at [nv+vs]) instead of
        # the baked literal. DO NOT re-apply alpha — the table already holds the
        # folded value. (For non-mimic robots the per-jid u-site and the v-slot are
        # 1:1 so this is exact and bit-identical to the literal path until poked.)
        nv = self.robot.get_num_vel()
        if b != 0.0:
            terms.append("d_robotModel->d_joint_dynamics_params[" + str(vs) + "] * " + qd)
        if fr != 0.0:
            terms.append("d_robotModel->d_joint_dynamics_params[" + str(nv + vs)
                         + "] * static_cast<T>((" + qd + " > static_cast<T>(0)) - ("
                         + qd + " < static_cast<T>(0)))")
        return " + ".join(terms)
    if b != 0.0:
        terms.append("static_cast<T>(" + repr(alpha * b) + ") * " + qd)
    if fr != 0.0:
        # exact sign matching np.sign (0 at qd==0): (qd>0) - (qd<0).
        terms.append("static_cast<T>(" + repr(alpha * fr)
                     + ") * static_cast<T>((" + qd + " > static_cast<T>(0)) - ("
                     + qd + " < static_cast<T>(0)))")
    return " + ".join(terms)


def gen_aba_inner_floating(self):
    NJ = self.robot.get_num_joints()
    nv = self.robot.get_num_vel()
    n_bfs_levels = self.robot.get_max_bfs_level() + 1

    IAOffset = 0
    vcrossOffset = 36 * NJ
    cOffset = 72 * NJ
    pAOffset = 78 * NJ
    UOffset = 84 * NJ
    paOffset = 90 * NJ
    dOffset = 96 * NJ
    uOffset = 97 * NJ
    tempMatOffset = 98 * NJ
    tempVecOffset = 134 * NJ
    fbUOffset = 140 * NJ
    fbDOffset = fbUOffset + 36
    fbDinvOffset = fbDOffset + 36
    fbRhsOffset = fbDinvOffset + 36
    fbInvTempOffset = fbRhsOffset + 6

    func_params = ["s_qdd is the vector of joint accelerations", \
                "s_va is a pointer to shared memory of size 2*6*NUM_BODIES = " + str(12*NJ), \
                "s_q is the vector of joint positions", \
                "s_qd is the vector of joint velocities", \
                "s_tau is the vector of generalized forces", \
                "s_temp is the (shared) scratch; size ABA_INNER_SMEM_BYTES<T, TEMP_IN_SMEM>() (the band when TEMP_IN_SMEM, else 0)", \
                "d_workspace is the global scratch. !TEMP_IN_SMEM: the whole band (ABA_INNER_WORKSPACE_BYTES). TEMP_IN_SMEM && !COLD_IN_SMEM: the cold slab d_cold (ABA_INNER_COLD_BYTES = vcross 36*NJ + fb* root tail 138, packed back-to-back). Pass nullptr at PERF (TEMP_IN_SMEM && COLD_IN_SMEM).", \
                "gravity is the gravity constant"]
    func_def_start = "void aba_inner("
    func_def_middle = "T *s_qdd, T *s_va, const T *s_q, const T *s_qd, const T *s_tau, "
    # runtime_joint_dynamics: the u-site bias reads d_robotModel->d_joint_dynamics_params,
    # so thread d_robotModel in as a trailing defaulted param ONLY under that flag
    # (byte-identical signature when off).
    if getattr(self, "runtime_joint_dynamics", False):
        func_def_end = "T *s_temp, T *d_workspace, T *d_f_ext, const T gravity, const robotModel<T> *d_robotModel = nullptr) {"
    else:
        func_def_end = "T *s_temp, T *d_workspace, T *d_f_ext, const T gravity) {"
    func_params.append("d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr")
    func_notes = ["Assumes the XI matricies have already been updated for the given q",
                  "Floating-base implementation keeps the scalar-joint ABA recursion and solves the 6x6 root block explicitly.",
                  "Inner-controlled placement, two orthogonal levers decided at the top:",
                  "  TEMP_IN_SMEM=false                : whole scratch band -> d_workspace (blunt MINIMAL fallback).",
                  "  TEMP_IN_SMEM=true, COLD_IN_SMEM=true  : PERF, everything in s_temp (byte-identical to the original).",
                  "  TEMP_IN_SMEM=true, COLD_IN_SMEM=false : SURGICAL -- hot recursion stays in s_temp, only the cold vcross slab [36*NJ,72*NJ) and the fb* root tail [140*NJ,140*NJ+138) spill to d_cold (=d_workspace sub-offset), packed back-to-back.",
                  "Caller sizes the arenas from ABA_INNER_{SMEM,WORKSPACE,COLD}_BYTES."]
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(func_def_middle, func_params, -2)
    func_def = func_def_start + func_def_middle + func_def_end
    self.gen_add_func_doc("Computes the Floating-Base Articulated Body Algorithm", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, bool TEMP_IN_SMEM = true, bool COLD_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # Inner-controlled scratch-band placement (two levers):
    #   !TEMP_IN_SMEM           : whole band -> d_workspace (s_temp reassigned).
    #   TEMP_IN_SMEM,!COLD_IN_SMEM: hot band stays in s_temp; the cold vcross
    #     slab [36*NJ,72*NJ) and the fb* root tail [140*NJ,140*NJ+138) are
    #     repointed through s_vcross_cold / s_fb_cold to d_cold (=d_workspace),
    #     packed back-to-back (vcross at d_cold[0..36*NJ), fb tail after it).
    # The biases line up the absolute offsets onto d_cold; when COLD_IN_SMEM
    # both pointers are just s_temp -> byte-identical to the original.
    self.gen_add_code_line("if constexpr (!TEMP_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }")
    self.gen_add_code_line("T *s_vcross_cold = s_temp;")
    self.gen_add_code_line("T *s_fb_cold = s_temp;")
    self.gen_add_code_line("if constexpr (TEMP_IN_SMEM && !COLD_IN_SMEM) { s_vcross_cold = d_workspace - " + str(vcrossOffset) + "; s_fb_cold = d_workspace + " + str(36 * NJ - fbUOffset) + "; }")
    temp_size = self.gen_aba_inner_temp_mem_size()
    self.gen_linalg_smem_setup(temp_size)
    self.gen_add_code_line("// Recursive floating ABA root-port.")

    self.gen_add_code_line("// Initialize IA = I and clear c")
    self.gen_add_parallel_loop("ind", str(36 * NJ + 6 * NJ))
    self.gen_add_code_line("if (ind < " + str(36 * NJ) + ") { s_temp[" + str(IAOffset) + " + ind] = s_XImats[" + str(36 * NJ) + " + ind]; }")
    self.gen_add_code_line("else { s_temp[" + str(cOffset) + " + ind - " + str(36 * NJ) + "] = static_cast<T>(0); }")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    self.gen_add_code_line("//")
    self.gen_add_code_line("// Forward Pass")
    self.gen_add_code_line("//")
    for bfs_level in range(n_bfs_levels):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        self.gen_add_code_line("// forward pass where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))

        if bfs_level == 0:
            self.gen_add_parallel_loop("row", "6")
            self.gen_add_code_line("int fb_col = row < 3 ? row + 3 : row - 3;")
            self.gen_add_code_line("s_va[row] = s_qd[fb_col];")
            self.gen_add_code_line("s_temp[" + str(cOffset) + " + row] = static_cast<T>(0);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            continue

        # Sibling joints at the SAME bfs level are independent (disjoint,
        # already-computed parents), so fuse all of this level's per-joint 6x6
        # row-strided GEMVs into ONE block-cooperative
        # grid_linalg_segmented_row_strided_gemv (mirrors the ID forward-pass
        # fusion in _inverse_dynamics.py). The += S*qd correction folds into the
        # same store via FUSE_SCALED_ADD: a compile-time per-segment selector
        # S_sel holds the joint sign at its S index (0 elsewhere), and
        # scalar[seg] = s_qd[dof]. Then the c[k] = mxS(v[k])*(sign*qd) bias is a
        # single parallel loop over siblings with a multi_threaded_select. This
        # is a pure independent-work reorder -> numerically identical (per-joint
        # GEMV column reduction order unchanged), with ONE sync per level instead
        # of per joint.
        seg = len(inds)
        parents = [self.robot.get_parent_id(jid_val) for jid_val in inds]
        s_inds = [self.robot.get_S_index_by_id(jid_val) for jid_val in inds]
        s_signs = [self.robot.get_S_sign_by_id(jid_val) for jid_val in inds]
        dofs = [jid_val + 5 for jid_val in inds]
        tag = "lvl" + str(bfs_level)
        self.gen_add_code_line("// v[k] = X[k]*v[parent_k] + S[k]*qd[k] for all bfs-level joints at once")
        a_off = ", ".join(str(36 * jid_val) for jid_val in inds)
        v_x_off = ", ".join(str(6 * p) for p in parents)
        v_y_off = ", ".join(str(6 * jid_val) for jid_val in inds)
        self.gen_add_code_line(f"static const int seg_a_off_{tag}[{seg}] = {{{a_off}}};")
        self.gen_add_code_line(f"static const int seg_v_x_off_{tag}[{seg}] = {{{v_x_off}}};")
        self.gen_add_code_line(f"static const int seg_v_y_off_{tag}[{seg}] = {{{v_y_off}}};")
        # per-segment 6-vector selector: sign at the joint S index, 0 elsewhere
        sel_vals = []
        for i in range(seg):
            row_vals = ["static_cast<T>(0)"] * 6
            row_vals[s_inds[i]] = f"static_cast<T>({s_signs[i]})"
            sel_vals.extend(row_vals)
        self.gen_add_code_line(f"static const int seg_s_off_{tag}[{seg}] = {{{', '.join(str(6 * i) for i in range(seg))}}};")
        self.gen_add_code_line(f"static const T S_sel_{tag}[{6 * seg}] = {{{', '.join(sel_vals)}}};")
        # scalar[seg] = s_qd[dof] in tempMat scratch (free during the forward pass)
        self.gen_add_serial_ops()
        for i in range(seg):
            self.gen_add_code_line(f"s_temp[{tempMatOffset + i}] = s_qd[{dofs[i]}];")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_add_code_line(f"grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>({seg}, seg_a_off_{tag}, seg_v_x_off_{tag}, seg_v_y_off_{tag}, s_XImats, s_va, s_va, static_cast<T>(1), static_cast<T>(0), seg_s_off_{tag}, S_sel_{tag}, &s_temp[{tempMatOffset}], s_linalg_smem);")
        # c[k] = mxS(v[k]) * (sign*qd[dof]) for all siblings (parallel over level)
        self.gen_add_code_line("// c[k] = mxS(v[k]) * (S_sign*qd[k])")
        self.gen_add_parallel_loop("ind", str(seg))
        if seg > 1:
            _, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)
            S_sign_cpp = self.gen_topology_S_sign_for_cpp(inds)
            select_var_vals = [("int", "jid", [str(jid_val) for jid_val in inds]),
                               ("int", "dof", [str(d) for d in dofs])]
            self.gen_add_multi_threaded_select("ind", "==", [str(i) for i in range(seg)], select_var_vals)
            dst_name = "&s_temp[" + str(cOffset) + " + 6*jid]"
            src_name = "&s_va[6*jid]"
            scale_name = "(" + S_sign_cpp + ") * s_qd[dof]"
        else:
            jid_val = inds[0]
            S_ind_cpp = str(s_inds[0])
            S_sign_cpp = str(s_signs[0])
            dst_name = "&s_temp[" + str(cOffset + 6 * jid_val) + "]"
            src_name = "&s_va[" + str(6 * jid_val) + "]"
            scale_name = "static_cast<T>(" + str(s_signs[0]) + ") * s_qd[" + str(dofs[0]) + "]"
        self.gen_mx_func_call_for_cpp(inds, updated_var_names = dict(S_ind_name = S_ind_cpp,
                                                                     s_dst_name = dst_name,
                                                                     s_src_name = src_name,
                                                                     s_scale_name = scale_name),
                                          PEQ_FLAG = False, SCALE_FLAG = True)
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    self.gen_add_code_line("// Initialize vcross[k]")
    self.gen_add_parallel_loop("jid", str(NJ))
    self.gen_add_code_line("int jid6 = 6 * jid;")
    self.gen_add_code_line("vcross<T>(&s_vcross_cold[" + str(vcrossOffset) + " + 36*jid], &s_va[jid6]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    self.gen_add_code_line("// temp[k] = -vcross.T*I[k]")
    self.gen_add_parallel_loop("ind", str(36 * NJ))
    self.gen_add_code_line("int row = ind % 6; int col = (ind / 6) % 6; int jid = ind / 36;")
    self.gen_add_code_line("int jid6 = 6 * jid;")
    self.gen_add_code_line("s_temp[" + str(tempMatOffset) + " + jid6*6 + row + col*6] = -dot_prod<T,6,1,1>(&s_vcross_cold[" + str(vcrossOffset) + " + 36*jid + row*6], &s_XImats[" + str(36 * NJ) + " + 36*jid + col*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    self.gen_add_code_line("// pA[k] = temp[k]*v[k]")
    self.gen_add_parallel_loop("ind", str(6 * NJ))
    self.gen_add_code_line("int row = ind % 6; int jid = ind / 6;")
    self.gen_add_code_line("int jid6 = 6 * jid;")
    self.gen_add_code_line("s_temp[" + str(pAOffset) + " + jid6 + row] = dot_prod<T,6,6,1>(&s_temp[" + str(tempMatOffset) + " + 6*jid6 + row], &s_va[jid6]);")
    # External forces (opt-in): subtract the per-body local-frame f_ext from the
    # bias pA (single subtract site). d_f_ext is GLOBAL, body-major 6*NUM_BODIES,
    # ordered [angular; linear]. nullptr -> no-op (dead-code-eliminated).
    self.gen_add_code_line("if (d_f_ext != nullptr) { s_temp[" + str(pAOffset) + " + jid6 + row] -= d_f_ext[jid6 + row]; }")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    self.gen_add_code_line("//")
    self.gen_add_code_line("// Backward Pass")
    self.gen_add_code_line("//")
    for jid in range(NJ - 1, -1, -1):
        jid6 = 6 * jid
        parent = self.robot.get_parent_id(jid)
        if jid == 0:
            self.gen_add_code_line("// floating-base root: U = IA*S, D = S^T*U")
            self.gen_add_parallel_loop("ind", "36")
            self.gen_add_code_line("int row = ind % 6; int col = ind / 6;")
            self.gen_add_code_line("int S_col = col < 3 ? col + 3 : col - 3;")
            self.gen_add_code_line("int S_row = row < 3 ? row + 3 : row - 3;")
            self.gen_add_code_line("s_fb_cold[" + str(fbUOffset) + " + ind] = s_temp[" + str(IAOffset) + " + row + 6*S_col];")
            self.gen_add_code_line("s_fb_cold[" + str(fbDOffset) + " + ind] = s_temp[" + str(IAOffset) + " + S_row + 6*S_col];")
            # No Ainv=I pre-init: glass::inv_dense seeds Ainv internally.
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            self.gen_add_code_line("invert_matrix(6, &s_fb_cold[" + str(fbDOffset) + "], &s_fb_cold[" + str(fbDinvOffset) + "], &s_fb_cold[" + str(fbInvTempOffset) + "]);")
            self.gen_add_parallel_loop("col", "6")
            self.gen_add_code_line("int S_col = col < 3 ? col + 3 : col - 3;")
            self.gen_add_code_line("s_fb_cold[" + str(fbRhsOffset) + " + col] = s_tau[col] - s_temp[" + str(pAOffset) + " + S_col];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            continue

        S_ind = self.robot.get_S_index_by_id(jid)
        S_sign = self.robot.get_S_sign_by_id(jid)
        dof = jid + 5
        self.gen_add_code_line("// scalar joint " + str(jid) + ": U, d, u")
        self.gen_add_parallel_loop("row", "6")
        self.gen_add_code_line("s_temp[" + str(UOffset + jid6) + " + row] = static_cast<T>(" + str(S_sign) + ") * s_temp[" + str(IAOffset + 36 * jid + 6 * S_ind) + " + row];")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_line("s_temp[" + str(dOffset + jid) + "] = static_cast<T>(" + str(S_sign) + ") * s_temp[" + str(UOffset + jid6 + S_ind) + "];")
        _jd_bias = _aba_jd_bias_term_cpp(self, jid)
        _jd_sub = (" - (" + _jd_bias + ")") if _jd_bias else ""
        self.gen_add_code_line("s_temp[" + str(uOffset + jid) + "] = s_tau[" + str(dof) + "]" + _jd_sub + " - static_cast<T>(" + str(S_sign) + ") * s_temp[" + str(pAOffset + jid6 + S_ind) + "] - dot_prod<T,6,1,1>(&s_temp[" + str(UOffset + jid6) + "], &s_temp[" + str(cOffset + jid6) + "]);")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

        if parent != -1:
            parent6 = 6 * parent
            self.gen_add_code_line("// transform U into the parent frame for joint " + str(jid))
            self.gen_add_code_line(f"grid_linalg_gemv<T,6,6,true>(&s_XImats[{36*jid}], &s_temp[{UOffset + jid6}], &s_temp[{tempVecOffset}], static_cast<T>(1), static_cast<T>(0));")
            self.gen_add_parallel_loop("row", "6")
            self.gen_add_code_line("s_temp[" + str(UOffset + jid6) + " + row] = s_temp[" + str(tempVecOffset) + " + row];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()

            self.gen_add_code_line("// temp = X.T*IA*X")
            self.gen_add_code_line(f"grid_linalg_gemm<T,6,6,6,false,true>(&s_XImats[{36*jid}], &s_temp[{IAOffset + 36*jid}], &s_temp[{tempVecOffset}], static_cast<T>(1), static_cast<T>(0), s_linalg_smem);")
            self.gen_add_code_line(f"grid_linalg_gemm<T,6,6,6>(&s_temp[{tempVecOffset}], &s_XImats[{36*jid}], &s_temp[{tempMatOffset}], static_cast<T>(1), static_cast<T>(0), s_linalg_smem);")
            self.gen_add_parallel_loop("ind", "36")
            self.gen_add_code_line("int row = ind % 6; int col = ind / 6;")
            self.gen_add_code_line("s_temp[" + str(tempMatOffset) + " + row + 6*col] -= s_temp[" + str(UOffset + jid6) + " + row] * s_temp[" + str(UOffset + jid6) + " + col] / s_temp[" + str(dOffset + jid) + "];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()

            self.gen_add_code_line("// pa = X.T*(pA + IA*c) + U*u/d")
            self.gen_add_parallel_loop("row", "6")
            self.gen_add_code_line("s_temp[" + str(paOffset + jid6) + " + row] = s_temp[" + str(pAOffset + jid6) + " + row] + dot_prod<T,6,6,1>(&s_temp[" + str(IAOffset + 36 * jid) + " + row], &s_temp[" + str(cOffset + jid6) + "]);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            self.gen_add_code_line(f"grid_linalg_gemv<T,6,6,true>(&s_XImats[{36*jid}], &s_temp[{paOffset + jid6}], &s_temp[{tempVecOffset}], static_cast<T>(1), static_cast<T>(0));")
            self.gen_add_serial_ops()
            for _ind in range(6):
                self.gen_add_code_line(f"s_temp[{tempVecOffset + _ind}] += s_temp[{UOffset + jid6 + _ind}] * s_temp[{uOffset + jid}] / s_temp[{dOffset + jid}];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            self.gen_add_parallel_loop("ind", "42")
            self.gen_add_code_line("int row = ind % 6; int col = ind / 6;")
            self.gen_add_code_line("if (ind < 36) { s_temp[" + str(IAOffset + 36 * parent) + " + row + 6*col] += s_temp[" + str(tempMatOffset) + " + row + 6*col]; }")
            self.gen_add_code_line("else { s_temp[" + str(pAOffset + parent6) + " + row] += s_temp[" + str(tempVecOffset) + " + row]; }")
            self.gen_add_end_control_flow()
            self.gen_add_sync()

    self.gen_add_code_line("//")
    self.gen_add_code_line("// Second Forward Pass")
    self.gen_add_code_line("//")
    for bfs_level in range(n_bfs_levels):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        if bfs_level == 0:
            self.gen_add_code_line("// root acceleration from gravity, then solve root qdd")
            self.gen_add_parallel_loop("ind", "36")
            self.gen_add_code_line("s_temp[" + str(tempMatOffset) + " + ind] = s_XImats[ind];")
            # No Ainv=I pre-init (or row/col): glass::inv_dense seeds Ainv internally.
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            self.gen_add_code_line("invert_matrix(6, &s_temp[" + str(tempMatOffset) + "], &s_temp[" + str(tempVecOffset) + "], &s_fb_cold[" + str(fbInvTempOffset) + "]);")
            self.gen_add_parallel_loop("row", "6")
            self.gen_add_code_line("s_va[" + str(6 * NJ) + " + row] = -s_temp[" + str(tempVecOffset) + " + row + 6*5] * gravity;")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            self.gen_add_parallel_loop("row", "6")
            self.gen_add_code_line("s_fb_cold[" + str(fbRhsOffset) + " + row] -= dot_prod<T,6,1,1>(&s_fb_cold[" + str(fbUOffset) + " + 6*row], &s_va[" + str(6 * NJ) + "]);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            self.gen_add_parallel_loop("row", "6")
            self.gen_add_code_line("s_qdd[row] = dot_prod<T,6,6,1>(&s_fb_cold[" + str(fbDinvOffset) + " + row], &s_fb_cold[" + str(fbRhsOffset) + "]);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            self.gen_add_parallel_loop("row", "6")
            self.gen_add_code_line("int fb_col = row < 3 ? row + 3 : row - 3;")
            self.gen_add_code_line("s_va[" + str(6 * NJ) + " + row] += s_qdd[fb_col];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            continue

        for jid in inds:
            parent = self.robot.get_parent_id(jid)
            S_ind = self.robot.get_S_index_by_id(jid)
            S_sign = self.robot.get_S_sign_by_id(jid)
            dof = jid + 5
            jid6 = 6 * jid
            parent6 = 6 * parent
            self.gen_add_serial_ops()
            self.gen_add_code_line("T tempval = s_temp[" + str(uOffset + jid) + "] - dot_prod<T,6,1,1>(&s_temp[" + str(UOffset + jid6) + "], &s_va[" + str(6 * NJ + parent6) + "]);")
            self.gen_add_code_line("s_qdd[" + str(dof) + "] = tempval / s_temp[" + str(dOffset + jid) + "];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            self.gen_add_code_line(f"grid_linalg_row_strided_gemv<T,6,6,6>(&s_XImats[{36*jid}], &s_va[{6*NJ + parent6}], &s_va[{6*NJ + jid6}], static_cast<T>(1), static_cast<T>(0), s_linalg_smem);")
            self.gen_add_parallel_loop("row", "6")
            self.gen_add_code_line("s_va[" + str(6 * NJ + jid6) + " + row] += s_temp[" + str(cOffset + jid6) + " + row];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            self.gen_add_serial_ops()
            self.gen_add_code_line("s_va[" + str(6 * NJ + jid6 + S_ind) + "] += static_cast<T>(" + str(S_sign) + ") * s_qdd[" + str(dof) + "];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()

    self.gen_add_end_function()


def _gen_aba_inner_mimic_fixed(self, NB):
    """Mimic fixed-base aba_inner = ID(bias) + Minv + qdd = Minv*(tau - bias).

    Composes the already-mimic-aware inverse_dynamics_inner (compute_c) and
    minv_inner (= inv(CRBA)). s_temp layout (sized in
    gen_aba_inner_temp_mem_size):
      [0, NV)                  s_c       (bias / generalized force)
      [NV, NV+18*NB)           s_vaf     (ID intermediate band)
      [NV+18*NB, +NV*NV)       s_Minv
      [..., +work)             s_work    (ID inner temp, then minv inner temp)
    Final qdd[i] = sum_j Minv[i,j] * (tau[j] - c[j]); Minv is symmetric-upper
    from invert_matrix (dense), so we read it symmetric.
    """
    nv = self.robot.get_num_vel()
    c_off = 0
    vaf_off = c_off + nv
    minv_off = vaf_off + 18 * NB
    work_off = minv_off + nv * nv
    self.gen_add_code_line("// mimic ABA: qdd = Minv * (tau - rnea(q,qd,0))")
    self.gen_add_code_line("T *s_aba_c = &s_temp[" + str(c_off) + "];")
    self.gen_add_code_line("T *s_aba_vaf = &s_temp[" + str(vaf_off) + "];")
    self.gen_add_code_line("T *s_aba_Minv = &s_temp[" + str(minv_off) + "];")
    self.gen_add_code_line("T *s_aba_work = &s_temp[" + str(work_off) + "];")
    # bias c = rnea(q, qd, qdd=0): inverse_dynamics_inner(compute_c, no qdd)
    self.gen_inverse_dynamics_inner_function_call(
        compute_c=True, use_qdd_input=False,
        updated_var_names=dict(s_c_name="s_aba_c", s_vaf_name="s_aba_vaf",
                               s_temp_name="s_aba_work"))
    self.gen_add_sync()
    # Minv = inv(CRBA(q)) via the mimic minv_inner path.
    self.gen_minv_inner_function_call(
        updated_var_names=dict(s_Minv_name="s_aba_Minv", s_temp_name="s_aba_work",
                               d_workspace_name="nullptr"),
        f_in_smem_expr="true")
    self.gen_add_sync()
    # qdd[i] = sum_j Minv[i,j] * (tau[j] - c[j]); Minv symmetric (read upper/lower).
    self.gen_add_parallel_loop("i", str(nv))
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int j = 0; j < " + str(nv) + "; j++) {", True)
    self.gen_add_code_line("int r = i <= j ? i : j; int col = i <= j ? j : i;")
    self.gen_add_code_line("acc += s_aba_Minv[col * " + str(nv) + " + r] * (s_tau[j] - s_aba_c[j]);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("s_qdd[i] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def gen_aba_inner(self):
    if self.robot.floating_base:
        return gen_aba_inner_floating(self)
    n = self.robot.get_num_joints()
    n_bfs_levels = self.robot.get_max_bfs_level() + 1 # starts at 0
	# construct the boilerplate and function definition
    func_params = ["s_qdd is the vector of joint accelerations", \
                "s_va is a pointer to shared memory of size 2*6*NUM_JOINTS = " + str(12*n), \
                "s_q is the vector of joint positions", \
                "s_qd is the vector of joint velocities", \
                "s_tau is the vector of joint torques", \
                "s_temp is the (shared) scratch; size ABA_INNER_SMEM_BYTES<T, TEMP_IN_SMEM>() (the 140*NJ+ band when TEMP_IN_SMEM, else 0)", \
                "d_workspace is the global scratch. !TEMP_IN_SMEM: the whole band (ABA_INNER_WORKSPACE_BYTES). TEMP_IN_SMEM && !COLD_IN_SMEM: the cold slab d_cold (ABA_INNER_COLD_BYTES = the [98*NJ,140*NJ) tempMat slab). Pass nullptr at PERF (TEMP_IN_SMEM && COLD_IN_SMEM).", \
                "gravity is the gravity constant"]
    func_def_start = "void aba_inner("
    func_def_middle = "T *s_qdd, T *s_va, const T *s_q, const T *s_qd, const T *s_tau, "
    # runtime_joint_dynamics: the u-site bias reads d_robotModel->d_joint_dynamics_params,
    # so thread d_robotModel in as a trailing defaulted param ONLY under that flag
    # (byte-identical signature when off).
    if getattr(self, "runtime_joint_dynamics", False):
        func_def_end = "T *s_temp, T *d_workspace, T *d_f_ext, const T gravity, const robotModel<T> *d_robotModel = nullptr) {"
    else:
        func_def_end = "T *s_temp, T *d_workspace, T *d_f_ext, const T gravity) {"
    func_params.append("d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr")
    func_notes = ["Assumes the XI matricies have already been updated for the given q",
                  "Inner-controlled placement, two orthogonal levers decided at the top:",
                  "  TEMP_IN_SMEM=false                : whole scratch band -> d_workspace (blunt MINIMAL fallback).",
                  "  TEMP_IN_SMEM=true, COLD_IN_SMEM=true  : PERF, everything in s_temp (byte-identical to the original).",
                  "  TEMP_IN_SMEM=true, COLD_IN_SMEM=false : SURGICAL -- hot recursion stays in s_temp, only the cold tempMat slab [98*NJ,140*NJ) spills to d_cold (=d_workspace sub-offset).",
                  "Caller sizes the arenas from ABA_INNER_{SMEM,WORKSPACE,COLD}_BYTES."]
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(func_def_middle, func_params, -2)
    func_def = func_def_start + func_def_middle + func_def_end
    self.gen_add_func_doc("Computes the Articulated Body Algorithm", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, bool TEMP_IN_SMEM = true, bool COLD_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # Inner-controlled scratch-band placement (two levers):
    #   !TEMP_IN_SMEM           : whole band -> d_workspace (s_temp reassigned).
    #   TEMP_IN_SMEM,!COLD_IN_SMEM: hot band stays in s_temp, the cold tempMat
    #     slab [98*n,140*n) is repointed through s_cold to d_cold (=d_workspace).
    # s_cold is biased by the cold base so the cold s_cold[98*n+...] references
    # land at d_cold[0...]; when COLD_IN_SMEM it is just s_temp -> byte-identical.
    self.gen_add_code_line("if constexpr (!TEMP_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }")
    self.gen_add_code_line("T *s_cold = s_temp;")
    self.gen_add_code_line("if constexpr (TEMP_IN_SMEM && !COLD_IN_SMEM) { s_cold = d_workspace - " + str(98 * n) + "; }")
    temp_size = self.gen_aba_inner_temp_mem_size()
    self.gen_linalg_smem_setup(temp_size)

    if self.robot_has_mimic_joints():
        # Mimic ABA via algebraic decomposition (mirrors RBDReference.aba mimic
        # fast path): qdd = Minv * (tau - rnea(q, qd, 0)). The per-body U/d ABA
        # recursion does not fold for mimic joints (Ia += U U^T / d scales by
        # alpha^2), so compose the already-mimic-aware ID bias + Minv instead.
        _gen_aba_inner_mimic_fixed(self, n)
        self.gen_add_end_function()
        return

    # Tier-C spherical (ball) joint support. When the robot carries a ball joint
    # the per-joint D becomes 3x3 (matrix-inverted), and every single-DoF body
    # downstream of the ball is shifted off `jid` in the reduced q/v/f vectors
    # (the §1e bug class) -- so cardinal joints read their qd/tau/qdd through the
    # parser's v/f maps, not the raw `jid`. All spherical-specific emit lives
    # inside `HAS_SPHERICAL` / `level_has_spherical` branches so cardinal robots
    # stay byte-identical.
    HAS_SPHERICAL = self.robot.robot_has_spherical()
    _sph_order = _aba_sph_orders(self) if HAS_SPHERICAL else {}

    def _qd_idx(jid):
        # reduced qd/qdd v-slot for a single-DoF body (spherical-shifted).
        return _aba_vslot(self, jid) if HAS_SPHERICAL else jid

    def _tau_idx(jid):
        # reduced tau/f-slot for a single-DoF body (== v-slot; see get_joint_index_f).
        return _aba_vslot(self, jid) if HAS_SPHERICAL else jid

    #
    # Initial Debug Prints if Requested
    #
    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_line("printf(\"q\\n\"); printMat<T,1," + str(n) + ">(s_q,1);")
        self.gen_add_code_line("printf(\"qd\\n\"); printMat<T,1," + str(n) + ">(s_qd,1);")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"X[%d]\\n\",i); printMat<T,6,6>(&s_XImats[36*i],6);}")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"I[%d]\\n\",i); printMat<T,6,6>(&s_XImats[36*(i+" + str(n) + ")],6);}")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    #
    # Forward Pass we are going to go in bfs_level waves
    # 
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Forward Pass")
    self.gen_add_code_line("//")
    for bfs_level in range(n_bfs_levels):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)
        # Skew (Tier-B) levels consume the dense S; the signed-index S_sign helper
        # raises on a skew joint, so query it only for cardinal levels.
        S_sign_cpp = None if any(not self.robot.S_is_cardinal_by_id(j) for j in inds) else self.gen_topology_S_sign_for_cpp(inds)

        if bfs_level == 0:
            self.gen_add_code_line("// s_v where parent is base")
            self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
            self.gen_add_code_line("//     links are: " + ", ".join(link_names))
            # compute the initial v which is just S*qd
            self.gen_add_code_line("// s_v[k] = S[k]*qd[k]")
            level_has_spherical = any(self.robot.joint_is_spherical(j) for j in inds)
            level_has_skew = any(
                (not self.robot.joint_is_spherical(j)) and (not self.robot.S_is_cardinal_by_id(j))
                for j in inds)
            if HAS_SPHERICAL:
                # Tier-C spherical at level 0: v[k] = S*qd reads the joint's 3-wide
                # angular v-block (rows 0..2). Cardinal joints (incl. cardinal-only
                # levels on a spherical robot) add their single SHIFTED v-slot.
                # Serial per-joint (spherical robots are rare; thread-invariant
                # single-thread writes).
                self.gen_add_serial_ops()
                for jid_val in inds:
                    jid6 = 6 * jid_val
                    for r in range(6):
                        self.gen_add_code_line("s_va[" + str(jid6 + r) + "] = static_cast<T>(0);")
                    if self.robot.joint_is_spherical(jid_val):
                        vblk = self.robot.get_joint_index_v(jid_val)
                        for k in range(3):  # angular-identity columns -> rows 0,1,2
                            self.gen_add_code_line("s_va[" + str(jid6 + k) + "] += s_qd[" + str(vblk[k]) + "];")
                    else:
                        s_ind_val = self.robot.get_S_index_by_id(jid_val)
                        s_sign_val = self.robot.get_S_sign_by_id(jid_val)
                        self.gen_add_code_line("s_va[" + str(jid6 + s_ind_val) + "] += (" + str(s_sign_val) + ") * s_qd[" + str(_qd_idx(jid_val)) + "];")
                self.gen_add_end_control_flow()
                self.gen_add_sync()
            elif level_has_skew:
                # Tier B (skew): each joint has a dense S column. Parallelize the
                # zero+column-add over the 6*len(inds) rows; per-thread dispatch
                # is by compile-time jid so each jid uses its own S constant.
                self.gen_add_parallel_loop("ind", str(6*len(inds)))
                self.gen_add_code_line("int row = ind % 6;")
                for i, jid_val in enumerate(inds):
                    self.gen_add_code_line(("if " if i == 0 else "else if ") + "(ind < " + str(6*(i+1)) + ") {", True)
                    self.gen_add_code_line("int jid6 = 6*" + str(jid_val) + ";")
                    self.gen_add_code_line("s_va[jid6 + row] = static_cast<T>(0);")
                    if self.robot.S_is_cardinal_by_id(jid_val):
                        s_ind_val = self.robot.get_S_index_by_id(jid_val)
                        s_sign_val = self.robot.get_S_sign_by_id(jid_val)
                        self.gen_add_code_line("if (row == " + str(s_ind_val) + ") { s_va[jid6 + " + str(s_ind_val) + "] += (" + str(s_sign_val) + ") * s_qd[" + str(jid_val) + "]; }")
                    else:
                        Svec = _aba_Svec_cpp(self.robot, jid_val)
                        self.gen_add_code_line("const T S_skew[6] = " + Svec + "; s_va[jid6 + row] += S_skew[row] * s_qd[" + str(jid_val) + "];")
                    self.gen_add_end_control_flow()
                self.gen_add_end_control_flow()
                self.gen_add_sync()
            else:
                if len(inds) > 1:
                    self.gen_add_parallel_loop("ind",str(6*len(inds)))
                    self.gen_add_code_line("int row = ind % 6;")
                    select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                    self.gen_add_multi_threaded_select("ind", "<", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
                    jid = "jid"
                else:
                    self.gen_add_parallel_loop("row",str(6))
                    jid = str(inds[0])
                    self.gen_add_code_line("int jid = " + jid + ";")
                # load in 0 to v
                self.gen_add_code_lines(["int jid6 = 6*jid;", \
                                         "s_va[jid6 + row] = static_cast<T>(0);"])
                # add in qd
                self.gen_add_code_line("if (row == " + S_ind_cpp + "){s_va[jid6 + " + S_ind_cpp + "] += (" + S_sign_cpp + ") * s_qd[" + jid + "];}")
                self.gen_add_end_control_flow()
                self.gen_add_sync()

            # add debug if requested
            if self.DEBUG_MODE:
                self.gen_add_sync()
                self.gen_add_serial_ops()
                for ind in inds:
                    self.gen_add_code_line("printf(\"s_v[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_va[6*" + str(ind) + "],1);")
                self.gen_add_end_control_flow()
                self.gen_add_sync()
        
        else:
            self.gen_add_code_line("// s_v where bfs_level is " + str(bfs_level))
            self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
            self.gen_add_code_line("//     links are: " + ", ".join(link_names))
            self.gen_add_code_line("// s_v[k] = X[k]*v[parent_k] + S[k]*qd[k]")
            # per-jid row_strided_gemv for v
            for jid_val in inds:
                parent_val = self.robot.get_parent_id(jid_val)
                self.gen_add_code_line(f"grid_linalg_row_strided_gemv<T,6,6,6>(&s_XImats[{36*jid_val}], &s_va[{6*parent_val}], &s_va[{6*jid_val}], static_cast<T>(1), static_cast<T>(0), s_linalg_smem);")
                self.gen_add_serial_ops()
                if self.robot.joint_is_spherical(jid_val):
                    # Tier-C spherical: v[jid] += S*qd over the 3 angular columns
                    # (rows 0..2), reading the joint's 3-wide v-block.
                    vblk = self.robot.get_joint_index_v(jid_val)
                    for k in range(3):
                        self.gen_add_code_line(f"s_va[{6*jid_val + k}] += s_qd[{vblk[k]}];")
                elif self.robot.S_is_cardinal_by_id(jid_val):
                    s_ind_val = self.robot.get_S_index_by_id(jid_val)
                    s_sign_val = self.robot.get_S_sign_by_id(jid_val)
                    self.gen_add_code_line(f"s_va[{6*jid_val + s_ind_val}] += ({s_sign_val}) * s_qd[{_qd_idx(jid_val)}];")
                else:
                    # Tier B: += S_col * qd (dense)
                    Svec = _aba_Svec_cpp(self.robot, jid_val)
                    self.gen_add_code_line("{ const T S_skew[6] = " + Svec + "; for (int r = 0; r < 6; r++) { s_va[" + str(6*jid_val) + " + r] += S_skew[r] * s_qd[" + str(_qd_idx(jid_val)) + "]; } }")
                self.gen_add_end_control_flow()
                self.gen_add_sync()

        
            # add debug if requested
            if self.DEBUG_MODE:
                self.gen_add_sync()
                self.gen_add_serial_ops()
                for ind in inds:
                    self.gen_add_code_line("printf(\"s_v[" + str(ind) + "] = X*s_v[" + parent_ind_cpp + "] + S*qd[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_va[6*" + str(ind) + "],1);")
                self.gen_add_end_control_flow()
                self.gen_add_sync()

    # calculate c
    self.gen_add_code_line("// c[k] = mxS(v[k])*qd[k]")
    HAS_SKEW = self.robot.robot_has_skew_axis()
    if HAS_SPHERICAL:
        # Tier-C: c[k] = crm(v[k]) * S[k] * qd[k]. A spherical joint sums its 3
        # angular-identity columns: c = mx0(v)*qd[v0] + mx1(v)*qd[v1] +
        # mx2(v)*qd[v2] (== crm(v)*(S*qd), pure-angular vJ). Cardinal joints use
        # their single mx column with the SHIFTED v-slot. Serial per-joint.
        self.gen_add_serial_ops()
        for jid in range(n):
            jid6 = 6 * jid
            self.gen_add_code_line("for (int r = 0; r < 6; r++) { s_temp[" + str(72*n + jid6) + " + r] = static_cast<T>(0); }")
            if self.robot.joint_is_spherical(jid):
                vblk = self.robot.get_joint_index_v(jid)
                for k in range(3):
                    self.gen_add_code_line("mx" + str(k) + "_peq_scaled<T>(&s_temp[" + str(72*n + jid6) + "], &s_va[" + str(jid6) + "], s_qd[" + str(vblk[k]) + "]);")
            else:
                s_ind = self.robot.get_S_index_by_id(jid); s_sign = self.robot.get_S_sign_by_id(jid)
                self.gen_add_code_line("mx" + str(s_ind) + "_peq_scaled<T>(&s_temp[" + str(72*n + jid6) + "], &s_va[" + str(jid6) + "], static_cast<T>(" + str(s_sign) + ") * s_qd[" + str(_qd_idx(jid)) + "]);")
        self.gen_add_end_control_flow()
    elif HAS_SKEW:
        # Tier B: c[k] = crm(v[k]) * S[k] * qd[k] with a dense S column. Serial
        # per-joint dispatch (skew robots are rare); cardinal joints keep the
        # precomputed mx column, skew joints use the generic crm*S helper.
        self.gen_add_serial_ops()
        for jid in range(n):
            jid6 = 6 * jid
            self.gen_add_code_line("for (int r = 0; r < 6; r++) { s_temp[" + str(72*n + jid6) + " + r] = static_cast<T>(0); }")
            if self.robot.S_is_cardinal_by_id(jid):
                s_ind = self.robot.get_S_index_by_id(jid); s_sign = self.robot.get_S_sign_by_id(jid)
                self.gen_add_code_line("mx" + str(s_ind) + "_peq_scaled<T>(&s_temp[" + str(72*n + jid6) + "], &s_va[" + str(jid6) + "], static_cast<T>(" + str(s_sign) + ") * s_qd[" + str(jid) + "]);")
            else:
                Svec = _aba_Svec_cpp(self.robot, jid)
                self.gen_add_code_line("{ const T S_skew[6] = " + Svec + "; mxS_general_peq_scaled<T>(&s_temp[" + str(72*n + jid6) + "], &s_va[" + str(jid6) + "], S_skew, s_qd[" + str(jid) + "]); }")
        self.gen_add_end_control_flow()
    else:
        self.gen_add_parallel_loop("ind", str(n))
        self.gen_add_code_line("int jid = ind;")
        self.gen_add_code_line("int jid6 = 6 * jid;")
        _, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(NO_GRAD_FLAG = True)
        S_sign_cpp = self.gen_topology_S_sign_for_cpp()
        self.gen_mx_func_call_for_cpp(list(range(n)), updated_var_names = dict(S_ind_name = S_ind_cpp, s_dst_name = "&s_temp[72 * " + str(n) + " + jid6]", s_src_name = "&s_va[jid6]", s_scale_name = "(" + S_sign_cpp + ") * s_qd[jid]"), PEQ_FLAG = False, SCALE_FLAG = True)
        self.gen_add_end_control_flow()
    
    # add debug if requested
    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_line("printf(\"c\\n\"); printMat<T,6,"+str(n)+">(&s_temp[72 * "+str(n)+"], 6);")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    # set IA = I
    self.gen_add_code_line("// Initialize IA = I")
    self.gen_add_parallel_loop("ind",str(36*n))
    self.gen_add_code_line("s_temp[ind] = s_XImats[" + str(36*n) + " + ind];")
    self.gen_add_end_control_flow()
    
    # initialize vcross from v
    self.gen_add_code_line("// Initialize vcross[k]")
    self.gen_add_parallel_loop("ind", str(n))
    self.gen_add_code_line("int jid = ind;")
    self.gen_add_code_line("int jid6 = 6 * jid;")

    self.gen_add_code_line("vcross<T>(&s_temp[36*("+str(n)+"+jid)], &s_va[jid6]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    self.gen_add_code_line("// temp[k] = -vcross.T*I[k]")
    self.gen_add_parallel_loop("ind", str(36*n))
    self.gen_add_code_line("int row = ind % 6; int col = (ind / 6) %6; int jid = ind / 36;")
    self.gen_add_code_line("int jid6 = 6 * jid;")
    self.gen_add_code_line("s_cold[98 * " + str(n) + " + jid6*6 + row+col*6] = -1 * dot_prod<T,6,1,1>(&s_temp[36*("+str(n)+"+jid)+row*6], &s_XImats[36 * ("+str(n)+"+jid) + col*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # calculate pA
    self.gen_add_code_line("// pA[k] = temp[k]*v[k][0]")
    self.gen_add_parallel_loop("ind", str(6*n))
    self.gen_add_code_line("int row = ind % 6; int comp = ind / 6; int jid = comp % " + str(n) + ";")
    self.gen_add_code_line("int jid6 = 6 * jid;")
    self.gen_add_code_line("s_temp[78 * " + str(n) + " + jid6 + row] = dot_prod<T,6,6,1>(&s_cold[98 * " + str(n) + " + 6*jid6+row], &s_va[jid6]);")
    # External forces (opt-in): subtract the per-body local-frame f_ext from the
    # bias pA (single subtract site). d_f_ext is GLOBAL, body-major 6*NUM_BODIES,
    # ordered [angular; linear]. nullptr -> no-op (dead-code-eliminated).
    self.gen_add_code_line("if (d_f_ext != nullptr) { s_temp[78 * " + str(n) + " + jid6 + row] -= d_f_ext[jid6 + row]; }")

    self.gen_add_end_control_flow()

    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"IA[%d]\\n\",i); printMat<T,6,6>(&s_temp[36*(i)],6);}")
        self.gen_add_code_line("printf(\"pA\\n\"); printMat<T,6,"+str(n)+">(&s_temp[78 * "+str(n)+"], 6);")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    #
    # Then compute the Backward Pass again in bfs waves
    #
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Backward Pass")
    self.gen_add_code_line("//")
    for bfs_level in range(n_bfs_levels - 1, -1, -1): 
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)
        S_sign_cpp = None if any(not self.robot.S_is_cardinal_by_id(j) for j in inds) else self.gen_topology_S_sign_for_cpp(inds)
        self.gen_add_code_line("// Backward pass where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))
        # caclulate U, which is just IA*S
        self.gen_add_code_line("// U[k] = IA[k]*S[k]")
        level_has_skew = any(
            (not self.robot.joint_is_spherical(j)) and (not self.robot.S_is_cardinal_by_id(j))
            for j in inds)
        if HAS_SPHERICAL:
            # Tier-C backward pass (U/D/u, Ia, pa) for the whole level, serial per
            # joint. A spherical joint's D = S^T*IA*S is a 3x3 matrix that is
            # MATRIX-INVERTED (invert_matrix dimA=3); cardinal joints keep the
            # scalar-D path with SHIFTED tau/qd slots. The X^T*Ia*X back-prop below
            # is S-agnostic (operates on the full 6x6 Ia / 6-vec pa) -> unchanged.
            for jid in inds:
                jid6 = 6 * jid
                if self.robot.joint_is_spherical(jid):
                    so = _aba_sph_block_off(self, jid, _sph_order[jid])
                    D_off, Dinv_off = so + _ABA_SPH_D_OFF, so + _ABA_SPH_DINV_OFF
                    U_off, UD_off = so + _ABA_SPH_U_OFF, so + _ABA_SPH_UDINV_OFF
                    u3_off, tmp3_off = so + _ABA_SPH_U3_OFF, so + _ABA_SPH_TMP3_OFF
                    invtmp_off = so + _ABA_SPH_INVTMP_OFF
                    fblk = self.robot.get_joint_index_f(jid)
                    vblk = self.robot.get_joint_index_v(jid)
                    # §1j: zero the whole spherical block before any beta=0/read-
                    # before-write use (D, U, UDinv, u3, tmp3, invert scratch).
                    self.gen_add_serial_ops()
                    self.gen_add_code_line("for (int z = 0; z < " + str(_ABA_SPH_STRIDE) + "; z++) { s_temp[" + str(so) + " + z] = static_cast<T>(0); }")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync()
                    # U = IA*S : 6x3, col c = IA[:, c] (S picks cols 0,1,2). Store
                    # col-major U[6*c + r].
                    self.gen_add_parallel_loop("ind", "18")
                    self.gen_add_code_line("int row = ind % 6; int col = ind / 6;")
                    self.gen_add_code_line("s_temp[" + str(U_off) + " + 6*col + row] = s_temp[" + str(36*jid) + " + row + 6*col];")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync()
                    # D = S^T*U : top-left 3x3, D[a,b] = U[6*b + a]
                    self.gen_add_parallel_loop("ind", "9")
                    self.gen_add_code_line("int a = ind % 3; int b = ind / 3;")
                    self.gen_add_code_line("s_temp[" + str(D_off) + " + 3*b + a] = s_temp[" + str(U_off) + " + 6*b + a];")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync()
                    self.gen_add_code_line("invert_matrix(3, &s_temp[" + str(D_off) + "], &s_temp[" + str(Dinv_off) + "], &s_temp[" + str(invtmp_off) + "]);")
                    self.gen_add_sync()
                    # u[k] = tau[fblk[k]] - S^T*pA  (- damping bias). S picks the
                    # angular rows, so S^T*pA = pA[jid6 + k] for k in 0..2. The
                    # Coriolis c enters the pa term below (pa += Ia*c), NOT u
                    # (matches the scalar-D path and the canonical Featherstone ABA).
                    self.gen_add_serial_ops()
                    for k in range(3):
                        _jd_bias = _aba_jd_bias_term_cpp(self, jid)
                        _jd_sub = (" - (" + _jd_bias + ")") if _jd_bias else ""
                        self.gen_add_code_line("s_temp[" + str(u3_off + k) + "] = s_tau[" + str(fblk[k]) + "]" + _jd_sub
                                               + " - s_temp[" + str(78*n + jid6 + k) + "];")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync()
                    # UDinv = U(6x3) * Dinv(3x3) : col-major UDinv[6*c + r]
                    self.gen_add_parallel_loop("ind", "18")
                    self.gen_add_code_line("int row = ind % 6; int c = ind / 6;")
                    self.gen_add_code_line("T acc = static_cast<T>(0);")
                    self.gen_add_code_line("for (int b = 0; b < 3; b++) { acc += s_temp[" + str(U_off) + " + 6*b + row] * s_temp[" + str(Dinv_off) + " + 3*c + b]; }")
                    self.gen_add_code_line("s_temp[" + str(UD_off) + " + 6*c + row] = acc;")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync()
                    # Ia = IA - U*Dinv*U^T : Ia[r,c] = IA[r,c] - sum_b UDinv[6*b+r]*U[6*b+c]
                    self.gen_add_parallel_loop("ind", "36")
                    self.gen_add_code_line("int row = ind % 6; int col = ind / 6;")
                    self.gen_add_code_line("T acc = static_cast<T>(0);")
                    self.gen_add_code_line("for (int b = 0; b < 3; b++) { acc += s_temp[" + str(UD_off) + " + 6*b + row] * s_temp[" + str(U_off) + " + 6*b + col]; }")
                    self.gen_add_code_line("s_temp[" + str(36*(n+jid)) + " + row + 6*col] = s_temp[" + str(36*jid) + " + row + 6*col] - acc;")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync()
                    # pa = pA + Ia*c + U*Dinv*u : pa[r] = pA[r] + Ia[:,r]^T... =
                    #   pA[r] + dot(Ia row r, c) + sum_b UDinv[6*b+r]*u[b]
                    self.gen_add_parallel_loop("row", "6")
                    self.gen_add_code_line("T acc = s_temp[" + str(78*n + jid6) + " + row] + dot_prod<T,6,6,1>(&s_temp[" + str(36*(n+jid)) + " + row], &s_temp[" + str(72*n + jid6) + "]);")
                    self.gen_add_code_line("for (int b = 0; b < 3; b++) { acc += s_temp[" + str(UD_off) + " + 6*b + row] * s_temp[" + str(u3_off) + " + b]; }")
                    self.gen_add_code_line("s_temp[" + str(90*n + jid6) + " + row] = acc;")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync()
                else:
                    # cardinal single-DoF joint (possibly downstream of a spherical):
                    # scalar D path with SHIFTED tau slot. U = IA*S, d = S^T*U,
                    # u = tau - S^T*pA, Ia = IA - U U^T/d, pa = pA + Ia*c + U*u/d.
                    s_ind = self.robot.get_S_index_by_id(jid)
                    s_sign = self.robot.get_S_sign_by_id(jid)
                    _jd_bias = _aba_jd_bias_term_cpp(self, jid)
                    _jd_sub = (" - (" + _jd_bias + ")") if _jd_bias else ""
                    self.gen_add_parallel_loop("row", "6")
                    self.gen_add_code_line("s_temp[" + str(84*n + jid6) + " + row] = (" + str(s_sign) + ") * s_temp[" + str(36*jid) + " + row + 6*" + str(s_ind) + "];")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync()
                    self.gen_add_serial_ops()
                    self.gen_add_code_line("s_temp[" + str(96*n + jid) + "] = (" + str(s_sign) + ") * s_temp[" + str(84*n + jid6 + s_ind) + "];")
                    self.gen_add_code_line("s_temp[" + str(97*n + jid) + "] = s_tau[" + str(_tau_idx(jid)) + "]" + _jd_sub + " - (" + str(s_sign) + ") * s_temp[" + str(78*n + jid6 + s_ind) + "];")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync()
                    self.gen_add_parallel_loop("ind", "36")
                    self.gen_add_code_line("int row = ind % 6; int col = ind / 6;")
                    self.gen_add_code_line("s_temp[" + str(36*(n+jid)) + " + row + 6*col] = s_temp[" + str(36*jid) + " + row + 6*col] - s_temp[" + str(84*n + jid6) + " + row]*s_temp[" + str(84*n + jid6) + " + col]/s_temp[" + str(96*n + jid) + "];")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync()
                    self.gen_add_parallel_loop("row", "6")
                    self.gen_add_code_line("T Uval = s_temp[" + str(84*n + jid6) + " + row]*s_temp[" + str(97*n + jid) + "]/s_temp[" + str(96*n + jid) + "];")
                    self.gen_add_code_line("s_temp[" + str(90*n + jid6) + " + row] = s_temp[" + str(78*n + jid6) + " + row] + dot_prod<T,6,6,1>(&s_temp[" + str(36*(n+jid)) + " + row], &s_temp[" + str(72*n + jid6) + "]) + Uval;")
                    self.gen_add_end_control_flow()
                    self.gen_add_sync()
        elif level_has_skew:
            # Tier B (skew): U = IA*S_dense (full 6x6 * 6 matvec); d = S^T*U;
            # u = tau - S^T*pA. Loop over every joint in the level (branching ok);
            # each joint with its own dense (or unit) S column.
            self.gen_add_serial_ops()
            for jid in inds:
                jid6 = 6 * jid
                Svec = _aba_Svec_cpp(self.robot, jid)
                self.gen_add_code_line("{ const T S_skew[6] = " + Svec + ";")
                # U = IA[jid] (col-major 6x6) * S
                self.gen_add_code_line("  for (int r = 0; r < 6; r++) { T acc = static_cast<T>(0); for (int p = 0; p < 6; p++) { acc += s_temp[36*" + str(jid) + " + r + 6*p] * S_skew[p]; } s_temp[" + str(84*n + jid6) + " + r] = acc; }")
                # d = S^T U ; u = tau - S^T pA
                self.gen_add_code_line("  s_temp[" + str(96*n + jid) + "] = dot_prod<T,6,1,1>(S_skew, &s_temp[" + str(84*n + jid6) + "]);")
                _jd_bias = _aba_jd_bias_term_cpp(self, jid)
                _jd_sub = (" - (" + _jd_bias + ")") if _jd_bias else ""
                self.gen_add_code_line("  s_temp[" + str(97*n + jid) + "] = s_tau[" + str(jid) + "]" + _jd_sub + " - dot_prod<T,6,1,1>(S_skew, &s_temp[" + str(78*n + jid6) + "]); }")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
        else:
            self.gen_add_parallel_loop("ind", str(6*len(inds)))
            self.gen_add_code_line("int row = ind % 6;")
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "<", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
                jid = "jid"
            else:
                jid = str(inds[0])
                self.gen_add_code_line("int jid = " + jid + ";")
            self.gen_add_code_line("int jid6 = 6 * " + jid + ";")

            self.gen_add_code_line("s_temp[84*"+str(n)+"+jid6+row] = (" + S_sign_cpp + ") * s_temp[36*jid+row+6*("+ S_ind_cpp+")];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()

            # caclulate d which is S*U and u which is tau - S*pA
            self.gen_add_code_line("// d[k] = S[k]*U[k], u[k] = tau[k] - S[k].T*pA[k]")
            self.gen_add_parallel_loop("ind", str(len(inds)))
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "<", [str((i+1)) for i in range(len(inds))], select_var_vals)
                jid = "jid"
            else:
                jid = str(inds[0])
                self.gen_add_code_line("int jid = " + jid + ";")
            self.gen_add_code_line("int jid6 = 6 * " + jid + ";")
            self.gen_add_code_line("s_temp[96 * "+ str(n) +" + jid] = (" + S_sign_cpp + ") * s_temp[84 * " + str(n) + " + jid6 + " + S_ind_cpp + "];")

            self.gen_add_code_line("T tempval = (" + S_sign_cpp + ") * s_temp[78 * " + str(n) + " + jid6 + " + S_ind_cpp +"];")
            self.gen_add_code_line("s_temp[97 * " + str(n) + " + jid] = s_tau[jid] - tempval;")
            # joint-local damping + Coulomb friction bias: u -= alpha*(b*qd+f*sign(qd)).
            # Gated/byte-neutral; per-python-jid since the bias coeffs (and mimic
            # v-slot) differ across the joints multiplexed onto this level's `jid`.
            for _pyjid in inds:
                _jd_bias = _aba_jd_bias_term_cpp(self, _pyjid)
                if _jd_bias:
                    self.gen_add_code_line("if (jid == " + str(_pyjid) + ") { s_temp[97 * " + str(n) + " + jid] -= (" + _jd_bias + "); }")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
        
        # calculate Ia from IA, U, and d (cardinal/skew scalar-D path; the
        # spherical branch above already computed Ia + pa with its 3x3 Dinv).
        if not HAS_SPHERICAL:
            self.gen_add_code_line("// Ia[k] = IA[k] - U[k]*U[k].T/d[k]")
            self.gen_add_parallel_loop("ind", str(36 * len(inds)))
            self.gen_add_code_line("int row = ind % 6; int col = (ind / 6) %6;")
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "<", [str(36*(i+1)) for i in range(len(inds))], select_var_vals)
                jid = "jid"
            else:
                jid = str(inds[0])
                self.gen_add_code_line("int jid = " + jid + ";")
            self.gen_add_code_line("int jid6 = 6 * " + jid + ";")

            self.gen_add_code_line("s_temp[36 * "+str(n)+"+6*jid6+row+6*col] = s_temp[84*"+str(n)+"+jid6+row]*s_temp[84*"+str(n)+"+jid6+col]/s_temp[96 *"+str(n)+"+jid];")

            self.gen_add_code_line("s_temp[36 * "+str(n)+"+6*jid6+row+6*col] = s_temp[6*jid6+row+6*col] - s_temp[36 * "+str(n)+"+6*jid6+row+6*col];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()

            # caclulate pa
            self.gen_add_code_line("// pa[k] = pA[k] + Ia[k]*c[k]+U[k]*u[k]/d[k]")
            self.gen_add_parallel_loop("ind", str(6*len(inds)))
            self.gen_add_code_line("int row = ind % 6;")
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "<", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
                jid = "jid"
            else:
                jid = str(inds[0])
                self.gen_add_code_line("int jid = " + jid + ";")
            self.gen_add_code_line("int jid6 = 6 * " + jid + ";")

            self.gen_add_code_line("T Uval = s_temp[84 * "+str(n)+"+jid6+row]*s_temp[97*"+str(n)+"+jid]/s_temp[96*"+str(n)+"+jid];")
            self.gen_add_code_line("s_temp[90 * "+str(n)+" + jid6 + row] = s_temp[78 * "+str(n)+" + jid6+row] + dot_prod<T,6,6,1>(&s_temp[36*("+str(n)+"+jid)+row], &s_temp[72*"+str(n)+"+jid6]) + Uval;")
            self.gen_add_end_control_flow()
        
        if bfs_level != 0:
            if len(inds) > 1 and self.robot.has_repeated_parents(inds):
                # Repeated parents: keep atomic dot_prod loops to avoid write conflicts
                self.gen_add_code_line("// temp[k] = X[k].T*Ia[k]")
                self.gen_add_parallel_loop("ind", str(36 * len(inds)))
                self.gen_add_code_line("int row = ind % 6; int col = (ind / 6) %6;")
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "<", [str(36*(i+1)) for i in range(len(inds))], select_var_vals)
                self.gen_add_code_line("int jid6 = 6 * jid;")
                self.gen_add_code_line("s_cold[98 * " + str(n) + " + 6 * jid6 + row + 6*col] = dot_prod<T,6,1,1>(&s_XImats[6*jid6+6*row], &s_temp[36 * "+str(n)+"+jid6*6+6*col]);")
                self.gen_add_end_control_flow()
                self.gen_add_sync()
                # update IA of the parent
                # NOTE (Inc6): this fixed-base branched path folds shared-parent
                # child IA into the parent via atomicAdd (order-dependent → last-ULP
                # run-to-run nondeterminism, same class as the crba/minv/id_gradient
                # floating folds fixed in Inc6). It is NOT exercised by the current
                # equivalence matrix (go2-fixed's only repeated-parent level is 0,
                # which the `bfs_level != 0` guard skips; fr3's mimic fingers collapse
                # to a single codegen joint; iiwa14 is a chain), so it cannot be
                # GPU-verified yet. Left on atomicAdd + tracked as a same-class
                # follow-up (with idsva_so / dccrba / coriolis-mimic / id_gradient
                # fixed-base sparsity) to convert to the deterministic parent-major
                # fixed-order sum once a robot that exercises it is in the matrix.
                self.gen_add_code_line("// IA[parent] += temp[k]*X[k]")
                self.gen_add_parallel_loop("ind", str(36 * len(inds)))
                self.gen_add_code_line("int row = ind % 6; int col = (ind / 6) %6;")
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "<", [str(36*(i+1)) for i in range(len(inds))], select_var_vals)
                self.gen_add_code_line("int jid6 = 6 * jid;")
                self.gen_add_code_line("T prodtemp = static_cast<T>(0);")
                self.gen_add_code_line("prodtemp =  dot_prod<T,6,6,1>(&s_cold[98 * " + str(n) + " + 6 * jid6 + row], &s_XImats[6*jid6+6*col]);")
                self.gen_add_code_line("atomicAdd(&s_temp[36 * " + parent_ind_cpp +" + row + 6*col], prodtemp);")
                self.gen_add_end_control_flow()
                self.gen_add_sync()
            else:
                # GEMM path: X^T*Ia*X per jid (single jid or all-distinct parents)
                for jid_val in inds:
                    parent_val = self.robot.get_parent_id(jid_val)
                    self.gen_add_code_line("// X[" + str(jid_val) + "].T*Ia[" + str(jid_val) + "]*X[" + str(jid_val) + "] -> IA[" + str(parent_val) + "]")
                    self.gen_add_code_line(f"grid_linalg_gemm<T,6,6,6,false,true>(&s_XImats[{36*jid_val}], &s_temp[{36*(n+jid_val)}], &s_cold[{98*n + 36*jid_val}], static_cast<T>(1), static_cast<T>(0), s_linalg_smem);")
                    self.gen_add_code_line(f"grid_linalg_gemm<T,6,6,6>(&s_cold[{98*n + 36*jid_val}], &s_XImats[{36*jid_val}], &s_temp[{36*parent_val}], static_cast<T>(1), static_cast<T>(1), s_linalg_smem);")

            # update pA of the parent (sequential GEMVs safe even for repeated parents)
            self.gen_add_code_line("// pA[parent] += X[k].T*pa[k]")
            for jid_val in inds:
                parent_val = self.robot.get_parent_id(jid_val)
                self.gen_add_code_line(f"grid_linalg_gemv<T,6,6,true>(&s_XImats[{36*jid_val}], &s_temp[{90*n + 6*jid_val}], &s_temp[{78*n + 6*parent_val}], static_cast<T>(1), static_cast<T>(1));")

    # add debug if requested
    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_line("printf(\"U \\n\"); printMat<T,6,"+str(n)+">(&s_temp[84 * "+str(n)+"], 6);")
        self.gen_add_code_line("printf(\"d \\n\"); printMat<T,1,"+str(n)+">(&s_temp[96 * "+str(n)+"], 1);")
        self.gen_add_code_line("printf(\"u \\n\"); printMat<T,1,"+str(n)+">(&s_temp[97 * "+str(n)+"], 1);")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"Ia[%d]\\n\",i); printMat<T,6,6>(&s_temp[36*("+str(n)+"+i)],6);}")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"IA[%d]\\n\",i); printMat<T,6,6>(&s_temp[36*(i)],6);}")
        self.gen_add_code_line("printf(\"pA\\n\"); printMat<T,6,"+str(n)+">(&s_temp[78 * "+str(n)+"], 6);")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    self.gen_add_code_line("//")
    self.gen_add_code_line("// Second Forward Pass")
    self.gen_add_code_line("//")
    for bfs_level in range(n_bfs_levels):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)
        S_sign_cpp = None if any(not self.robot.S_is_cardinal_by_id(j) for j in inds) else self.gen_topology_S_sign_for_cpp(inds)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        # calculate a where parent is base
        if bfs_level == 0:
            self.gen_add_code_line("// s_a, qdd where parent is base")
            self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
            self.gen_add_code_line("//     links are: " + ", ".join(link_names))
            self.gen_add_code_line("// a[k] = X[k]*gravity_vec + c[k]")
            if len(inds) > 1:
                self.gen_add_parallel_loop("ind",str(6*len(inds)))
                self.gen_add_code_line("int row = ind % 6;")
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "<", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
                jid = "jid"
            else:
                self.gen_add_parallel_loop("row",str(6))
                jid = str(inds[0])
                self.gen_add_code_line("int jid = " + jid + ";")
            self.gen_add_code_line("int jid6 = 6*" + jid + ";")
            self.gen_add_code_line("T gravity_vec[] = {0,0,0,0,0,-gravity}; // -gravity = +9.81 matches RBDReference gravity_vec[5] = -GRAVITY (gravity=-9.81)")
            self.gen_add_code_line("s_va[6*"+str(n)+"+jid6+row] = dot_prod<T,6,6,1>(&s_XImats[36 * jid + row], &gravity_vec[0]) + s_temp[72*"+str(n)+"+jid6+row];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
        # calculate a where parent is not base
        else:
            self.gen_add_code_line("// s_a, s_qdd where bfs_level is " + str(bfs_level))
            self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
            self.gen_add_code_line("//     links are: " + ", ".join(link_names))
            self.gen_add_code_line("// a[k] = X[k]*a[parent] + c[k]")
            # per-jid GEMV: a[jid] = X[jid]*a[parent] + c[jid]
            for jid_val in inds:
                parent_val = self.robot.get_parent_id(jid_val)
                self.gen_add_code_line(f"grid_linalg_row_strided_gemv<T,6,6,6>(&s_XImats[{36*jid_val}], &s_va[{6*n + 6*parent_val}], &s_va[{6*n + 6*jid_val}], static_cast<T>(1), static_cast<T>(0), s_linalg_smem);")
                self.gen_add_parallel_loop("row", "6")
                self.gen_add_code_line(f"s_va[{6*n + 6*jid_val} + row] += s_temp[{72*n + 6*jid_val} + row];")
                self.gen_add_end_control_flow()
                self.gen_add_sync()
        
        # calculate qdd which is (u - U*a)/d
        self.gen_add_code_line("// qdd[k] = (u[k] - U[k].T*a[k])/d[k]")
        if HAS_SPHERICAL:
            # Tier-C: spherical qdd = Dinv*(u - U^T*a) (3-vec, written to the f-block);
            # cardinal joints use the scalar (u - U^T a)/d with the SHIFTED qdd slot.
            self.gen_add_serial_ops()
            for jid in inds:
                jid6 = 6 * jid
                if self.robot.joint_is_spherical(jid):
                    so = _aba_sph_block_off(self, jid, _sph_order[jid])
                    U_off, Dinv_off = so + _ABA_SPH_U_OFF, so + _ABA_SPH_DINV_OFF
                    u3_off, tmp3_off = so + _ABA_SPH_U3_OFF, so + _ABA_SPH_TMP3_OFF
                    fblk = self.robot.get_joint_index_f(jid)
                    # tmp[k] = u[k] - U[:,k]^T * a[jid]
                    for k in range(3):
                        self.gen_add_code_line("s_temp[" + str(tmp3_off + k) + "] = s_temp[" + str(u3_off + k) + "] - dot_prod<T,6,1,1>(&s_temp[" + str(U_off + 6*k) + "], &s_va[" + str(6*n + jid6) + "]);")
                    # qdd_blk = Dinv * tmp  -> write the f-block of s_qdd
                    for k in range(3):
                        self.gen_add_code_line("s_qdd[" + str(fblk[k]) + "] = dot_prod<T,3,3,1>(&s_temp[" + str(Dinv_off + k) + "], &s_temp[" + str(tmp3_off) + "]);")
                else:
                    self.gen_add_code_line("s_qdd[" + str(_qd_idx(jid)) + "] = (s_temp[" + str(97*n + jid) + "] - dot_prod<T,6,1,1>(&s_temp[" + str(84*n + jid6) + "], &s_va[" + str(6*n + jid6) + "])) / s_temp[" + str(96*n + jid) + "];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
        else:
            self.gen_add_parallel_loop("ind",str(len(inds)))
            if len(inds) > 1:
                self.gen_add_code_line("int comp_mod = ind % "+ str(len(inds)) + ";")
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                jid = "jid"
                self.gen_add_multi_threaded_select("comp_mod", "==", [str(i) for i in range(len(inds))], select_var_vals)
            else:
                jid = str(inds[0])
                self.gen_add_code_line("int jid = " + jid + ";")
            self.gen_add_code_line("int jid6 = 6 * " + jid + ";")
            self.gen_add_code_line("T tempval = s_temp[97 * "+str(n)+"+jid] - dot_prod<T,6,1,1>(&s_temp[84*"+str(n)+"+jid6], &s_va[6*"+str(n)+"+jid6]);")
            self.gen_add_code_line("s_qdd[jid] = tempval / s_temp[96*"+str(n)+"+jid];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()

        # update a by adding qdd*S
        self.gen_add_code_line("// a[k] += qdd[k]*S[k]")
        level_has_skew = any(
            (not self.robot.joint_is_spherical(j)) and (not self.robot.S_is_cardinal_by_id(j))
            for j in inds)
        if HAS_SPHERICAL:
            # Tier-C: a[jid] += S*qdd. Spherical adds its 3 angular columns (rows
            # 0..2) from its f-block; cardinal joints add their single SHIFTED slot.
            self.gen_add_serial_ops()
            for jid in inds:
                jid6 = 6 * jid
                if self.robot.joint_is_spherical(jid):
                    fblk = self.robot.get_joint_index_f(jid)
                    for k in range(3):
                        self.gen_add_code_line("s_va[" + str(6*n + jid6 + k) + "] += s_qdd[" + str(fblk[k]) + "];")
                else:
                    s_ind_val = self.robot.get_S_index_by_id(jid)
                    s_sign_val = self.robot.get_S_sign_by_id(jid)
                    self.gen_add_code_line("s_va[" + str(6*n + jid6 + s_ind_val) + "] += (" + str(s_sign_val) + ") * s_qdd[" + str(_qd_idx(jid)) + "];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            continue
        self.gen_add_parallel_loop("ind",str(6*len(inds)))
        if level_has_skew:
            # Tier B (skew): a[:,jid] += S_dense * qdd[jid]. Per-jid dispatch by
            # compile-time index so each joint adds its own dense (or unit) S
            # column; handles multiple (branching) joints per level.
            self.gen_add_code_line("int row = ind % 6;")
            for i, jid_val in enumerate(inds):
                self.gen_add_code_line(("if " if i == 0 else "else if ") + "(ind < " + str(6*(i+1)) + ") {", True)
                jid6 = 6 * jid_val
                if self.robot.S_is_cardinal_by_id(jid_val):
                    s_ind_val = self.robot.get_S_index_by_id(jid_val)
                    s_sign_val = self.robot.get_S_sign_by_id(jid_val)
                    self.gen_add_code_line("s_va[6*"+str(n)+"+"+str(jid6)+"+row] += (row == " + str(s_ind_val) + ") * (" + str(s_sign_val) + ") * s_qdd[" + str(jid_val) + "];")
                else:
                    Svec = _aba_Svec_cpp(self.robot, jid_val)
                    self.gen_add_code_line("const T S_skew[6] = " + Svec + "; s_va[6*"+str(n)+"+"+str(jid6)+"+row] += S_skew[row] * s_qdd[" + str(jid_val) + "];")
                self.gen_add_end_control_flow()
        else:
            if len(inds) > 1:
                self.gen_add_code_line("int row = ind % 6; int comp = ind / 6; int comp_mod = comp % " + str(len(inds)) + ";")
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                jid = "jid"
                self.gen_add_multi_threaded_select("comp_mod", "==", [str(i) for i in range(len(inds))], select_var_vals)
            else:
                self.gen_add_code_line("int row = ind % 6;")
                jid = str(inds[0])
                self.gen_add_code_line("int jid = " + jid + ";")
            self.gen_add_code_line("int jid6 = 6 * " + jid + ";")
            self.gen_add_code_line("T qdd_val = (row == " + S_ind_cpp + ") * (" + S_sign_cpp + ") * (s_qdd[jid]);")
            self.gen_add_code_line("s_va[6*"+str(n)+"+jid6+row] += qdd_val;")

        self.gen_add_end_control_flow()
        self.gen_add_sync()
    
    # add debug if requested
    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_line("printf(\"a\\n\"); printMat<T,6,"+str(n)+">(&s_va[6 * "+str(n)+"], 6);")
        self.gen_add_code_line("printf(\"qdd\\n\"); printMat<T,1," + str(n) + ">(s_qdd,1);")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        

    self.gen_add_end_function()

def gen_aba_inner_temp_mem_size(self):
    n = self.robot.get_num_joints()
    if self.robot_has_mimic_joints():
        # Mimic aba composes ID(bias c) + Minv(=inv CRBA) + qdd=Minv*(tau-c).
        # Layout: s_c[NV] | s_vaf[18*NJ] | s_Minv[NV*NV] | s_work[...] where
        # s_work is shared by ID's inner temp (6*NJ) and minv's full inner band
        # (F = 6*NV*NV + no_F), which run sequentially.
        nv = self.robot.get_num_vel()
        minv_full = self.gen_minv_inner_temp_mem_size()  # 6*NV*NV + no_F
        work = max(self.gen_inverse_dynamics_inner_temp_mem_size(), minv_full)
        return nv + 18 * n + nv * nv + work
    if self.robot.floating_base:
        return max(140 * n + 138, self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=False))
    if self.robot.robot_has_spherical():
        # Tier-C: a dedicated spherical scratch arena (Dinv/U/UDinv/u/tmp/invert
        # per ball joint) sits ABOVE the cardinal 140*n hot band so the cold-spill
        # ladder offsets stay byte-identical for cardinal robots. Gated on
        # robot_has_spherical() so non-spherical sizing is untouched.
        n_sph = sum(1 for j in range(n) if self.robot.joint_is_spherical(j))
        return 140 * n + _ABA_SPH_STRIDE * n_sph
    return 140 * n

def gen_aba_inner_cold_mem_size(self):
    """Float count of the COLD sub-band aba_inner spills under the surgical
    rung (TEMP_IN_SMEM=true, COLD_IN_SMEM=false). The hot recursion stays in
    s_temp; only this cold slab moves to d_cold (a sub-offset of d_workspace).

    FIXED-base : the tempMat/vcross build-scratch slab [98*n, 140*n) -> 42*n.
    FLOATING   : vcross [36*NJ, 72*NJ) (36*NJ) packed contiguously ahead of the
                 fb* root block tail [140*NJ, 140*NJ+138) (138) -> 36*NJ + 138.
    The two floating regions are laid out back-to-back in d_cold so a single
    d_cold pointer covers them (vcross at d_cold[0..36*NJ), fb tail at
    d_cold[36*NJ..36*NJ+138))."""
    n = self.robot.get_num_joints()
    if self.robot.floating_base:
        return 36 * n + 138
    return 42 * n

def gen_aba_inner_function_call(self, updated_var_names = None,
                                temp_in_smem_expr = "true", cold_in_smem_expr = "true"):
    var_names = dict( \
        s_va_name = "s_va", \
        s_q_name = "s_q", \
        s_qd_name = "s_qd", \
        s_qdd_name = "s_qdd", \
        s_tau_name = "s_tau", \
        s_temp_name = "s_temp", \
        d_workspace_name = "nullptr", \
        d_f_ext_name = "d_f_ext", \
        gravity_name = "gravity", \
        d_robotModel_name = "d_robotModel"
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    aba_code_start = "aba_inner<T, " + temp_in_smem_expr + ", " + cold_in_smem_expr + ">(" + var_names["s_qdd_name"] + ", " + var_names["s_va_name"] + ", " + var_names["s_q_name"] + ", " + var_names["s_qd_name"] + ", " + var_names["s_tau_name"] + ", "
    # runtime_joint_dynamics: forward d_robotModel into aba_inner (trailing defaulted
    # param) so the u-site bias can read the mutable table; omitted when off.
    _aba_rt_jd = (", " + var_names["d_robotModel_name"]) if getattr(self, "runtime_joint_dynamics", False) else ""
    aba_code_end = var_names["s_temp_name"] + ", " + var_names["d_workspace_name"] + ", " + var_names["d_f_ext_name"] + ", " + var_names["gravity_name"] + _aba_rt_jd + ");"
    aba_code_middle = self.gen_insert_helpers_function_call()
    aba_code = aba_code_start + aba_code_middle + aba_code_end
    self.gen_add_code_line(aba_code)

def gen_aba_device(self):
    n = self.robot.get_num_joints()
    nv = self.robot.get_num_vel()
    # construct the boilerplate and function definition
    func_params = ["s_qdd is the vector of joint accelerations", \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                    "s_tau is the vector of joint torques", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr", \
                   "gravity is the gravity constant"]
    func_notes = []
    func_def_start = "void aba_device("
    func_def_middle = "T *s_qdd, const T *s_q, const T *s_qd, const T *s_tau, "
    func_def_end = "const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity) {"
    func_def = func_def_start + func_def_middle + func_def_end

    # then generate the code (shared device-wrapper skeleton; B+C §1.1)
    shared_mem_size = self.gen_aba_inner_temp_mem_size()
    self.gen_device_wrapper(
        "Compute the ABA (Articulated Body Algorithm)", func_def, shared_mem_size,
        lambda: self.gen_aba_inner_function_call(),
        func_notes = func_notes, func_params = func_params,
        extra_t_buffers = [("s_va", 12*n)], include_linalg_scratch = True)

def _aba_surgical_inner_smem_size(self):
    """Float count the smem s_temp arena needs at the SURGICAL rung. The cold
    band relocates to d_cold, but the remaining hot references run up to a
    fixed top offset, so the contiguous arena must reach that offset.
      FIXED   : hot ends at the cold base 98*n  -> reclaims the whole 42*n tail.
      FLOATING: hot tempVec ends at 140*NJ; only the 138-float fb* tail above it
                is reclaimed from smem (the interior vcross slot still spills to
                d_cold but its smem hole cannot be compacted byte-identically)."""
    n = self.robot.get_num_joints()
    if self.robot.floating_base:
        return self.gen_aba_inner_temp_mem_size() - 138
    if self.robot.robot_has_spherical():
        # Spherical arena lives ABOVE the cold band [98n,140n); spilling the cold
        # band leaves an interior hole that cannot be compacted byte-identically,
        # so keep the full hot arena in smem at the surgical rung (spherical
        # robots are tiny and never actually reach this spill tier).
        return self.gen_aba_inner_temp_mem_size()
    return 98 * n

def _emit_aba_kernel_body_for_flags(self, nq, nv, n, input_count, level, single_call_timing, mjx_kernel=False):
    """Emit aba_kernel body for one tier's spill level.
    level 0 (full)     : s_temp in smem, whole inner arena in smem (PERF; byte-identical to original).
    level 1 (surgical) : hot recursion stays in smem; only the cold band spills to d_cold
                         (= d_workspace sub-offset GRID_ABA_COLD_OFFSET_BYTES). smem holds the hot arena.
    level 2 (workspace): whole inner arena redirected to L2-pinned workspace; smem holds only extra_t_buffers."""
    use_workspace_temp = (level == 2)
    use_cold_spill     = (level == 1)
    if use_workspace_temp:
        shared_mem_size = 0
    elif use_cold_spill:
        shared_mem_size = _aba_surgical_inner_smem_size(self)
    else:
        shared_mem_size = self.gen_aba_inner_temp_mem_size()
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size, extra_t_buffers = [("s_qdd", nv), ("s_q_qd_tau", input_count), ("s_va", 12*n)], include_linalg_scratch=True)
    # Canonical NUM_JOINTS(=nq)-wide slots: qd at nq, tau/u at 2*nq (mirrors
    # id/forward_dynamics). Using nq+nv here would point s_tau into the qd slot
    # for a floating base (nq>nv) -- the off-by-(nq-nv) tau bug.
    self.gen_add_code_line("T *s_q = s_q_qd_tau; T *s_qd = &s_q_qd_tau[" + str(nq) + "]; T *s_tau = &s_q_qd_tau[" + str(2 * nq) + "];")
    # per-timestep workspace base expr (k-indexed in the batched kernel, slot 0 for single-timing)
    ws_base = "&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()]" if not single_call_timing else "d_workspace"
    if not single_call_timing:
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",block_level = True)
        self.gen_kernel_load_inputs("q_qd_tau",str(input_count),stride="stride_q_qd")
    else:
        self.gen_kernel_load_inputs("q_qd_tau",str(input_count))
    if use_workspace_temp:
        self.gen_add_code_line("T *aba_d_workspace = reinterpret_cast<T *>(" + ws_base + ");")
        # The whole inner arena spilled to global, so the smem s_temp slot is
        # null. Repoint s_temp at the workspace so the XImats helper's sincos
        # scratch (and the inner) have a valid backing store, not nullptr.
        self.gen_add_code_line("s_temp = aba_d_workspace;")
    elif use_cold_spill:
        # Surgical rung: hot band stays in smem s_temp; only the cold sub-band
        # lives in d_cold, a sub-offset of the per-timestep workspace. Reuse the
        # SO/grad band base (ABA never runs concurrently with SO/grad).
        self.gen_add_code_line("T *aba_d_cold = reinterpret_cast<T *>(" + ws_base + " + GRID_ABA_COLD_OFFSET_BYTES<T>());")
    else:
        self.gen_add_code_line("(void)d_workspace;")
    if not single_call_timing:
        # mjx input convert (before XImats so X[0] uses the reordered quaternion):
        # quat wxyz->xyzw on s_q, qd[0:3]=R^T qd[0:3], u(force) s_tau[0:3]=R^T s_tau[0:3].
        # qdd is the OUTPUT (accel_out class) -> no qdd_name here.
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert(q_name="s_q", qd_name="s_qd", u_name="s_tau")
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_aba_inner_function_call(
            updated_var_names = (dict(d_workspace_name = "aba_d_workspace") if use_workspace_temp else
                                 (dict(d_workspace_name = "aba_d_cold") if use_cold_spill else None)),
            temp_in_smem_expr = ("false" if use_workspace_temp else "true"),
            cold_in_smem_expr = ("false" if use_cold_spill else "true"))
        self.gen_add_sync()
        # mjx output: forward-dynamics accel qdd[0:3] = R (qdd[0:3] + omega x v_local),
        # omega=s_qd[3:6], v_local=s_qd[0:3] (PIN qd after the input convert).
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_accel_out(qdd_buf="s_qdd", qd_buf="s_qd")
            self.gen_add_end_control_flow()
        # Output slot is NUM_JOINTS(=nq)-wide (the binding/host read qdd nj-wide,
        # the d_qdd buffer is nj-strided). qdd has nv real accelerations; the
        # nj-wide slot stride keeps per-timestep outputs from overlapping for a
        # FLOATING base (nq>nv). For a FIXED base nq==nv -> byte-identical.
        self.gen_kernel_save_result("qdd",str(nv),stride=str(nq))
        self.gen_add_end_control_flow()
    else:
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q_qd_tau",str(input_count),feedback_from="qdd")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_aba_inner_function_call(
            updated_var_names = (dict(d_workspace_name = "aba_d_workspace") if use_workspace_temp else
                                 (dict(d_workspace_name = "aba_d_cold") if use_cold_spill else None)),
            temp_in_smem_expr = ("false" if use_workspace_temp else "true"),
            cold_in_smem_expr = ("false" if use_cold_spill else "true"))
        self.gen_anti_licm_output_write("qdd")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("qdd",str(nv))


def gen_aba_kernel(self, single_call_timing = False):
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    n = self.robot.get_num_joints()
    # Canonical per-timestep packing (mirrors id/forward_dynamics): q, qd, u/tau
    # each occupy a NUM_JOINTS(=nq)-wide slot at stride 3*NUM_JOINTS. The kernel
    # loads the full 3*nq slot and slices qd at nq, tau at 2*nq. For a FIXED base
    # nq==nv so 3*nq == nq+2*nv (byte-identical); for a FLOATING base nq>nv, so a
    # compressed nq+2*nv stride would read shifted/garbage inputs for k>=1 and a
    # misaligned tau even at slot 0 -- the floating batch>1 (and tau) bug.
    input_count = 3 * nq
    func_params = ["d_qdd is the vector of joint accelerations (output)", \
                    "d_workspace is the L2-pinned global spill buffer (used at LITE/MINIMAL on h1_2-scale)", \
                    "d_q_qd_tau is the vector of joint positions, velocities, torques", \
                    "stride_q_qd is the stride between each q, qd", \
                    "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                    "d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr", \
                    "gravity is the gravity constant", \
                    "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    func_def_start = "void aba_kernel(T *d_qdd, unsigned char *d_workspace, const T *d_q_qd_tau, const int stride_q_qd, "
    func_def_end = "T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Compute the ABA (Articulated Body Algorithm)", func_notes, func_params, None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag. Added
    # LAST so existing positional <T,TIER> call sites are unaffected; the default
    # (false) instantiation if-constexpr-elides the epilogue -> byte-identical PTX.
    # ABA is the accel_out class (same as forward_dynamics): qdd = aba(q, qd, u).
    mjx_kernel = self.robot.floating_base
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # Surgical-spill ladder, 3 rungs. ABA's inner scratch (140*NJ + 138) keeps
    # its hot recursion in smem and spills only the cold sub-band when it can.
    #   level 0 (full)     : whole arena in smem (PERF; byte-identical to original).
    #   level 1 (surgical) : hot band in smem, cold sub-band -> d_cold.
    #   level 2 (workspace): whole arena -> L2-pinned workspace (blunt fallback).
    # picks[tier] IS the level for that tier (see aba_spill_tier_3way).
    picks = getattr(self, "aba_spill_tier_3way", (0, 0, 0))
    self.gen_tier_dispatch(picks, lambda pick:
        _emit_aba_kernel_body_for_flags(self, nq, nv, n, input_count, pick, single_call_timing, mjx_kernel))
    self.gen_add_end_function()

def gen_aba_host(self, mode = 0):
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
    func_def_start = "void aba(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end =   "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    # then generate the code
    self.gen_add_func_doc("Compute the ABA (Articulated Body Algorithm)",\
                          func_notes,func_params,None)
    # MUJOCO_OUTPUT (floating only) host template flag: forwarded to the kernel
    # launch (the only mjx-capable overload). Added LAST so existing positional
    # template args are unaffected; default false -> byte-identical.
    mjx_host = self.robot.floating_base
    if mjx_host:
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    else:
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"aba requires all-data or dynamics gridData\");")

    # mjx-capable launch names the tier positionally to reach the trailing flag.
    aba_kernel_tmpl = "aba_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "aba_kernel<T, RESOURCE_TIER>"
    func_call_start = aba_kernel_tmpl + "<<<block_dimms,thread_dimms,ABA_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_qdd,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,"
    func_call_end = "hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);"
    # Canonical GRiD per-timestep stride: q/qd/u each in a NUM_JOINTS-wide slot
    # (mirrors id/crba/forward_dynamics + the binding's pack_q_qd_u). NUM_JOINTS +
    # 2*NUM_VEL coincides for a FIXED base (NUM_JOINTS==NUM_VEL) but is a
    # compressed stride for a FLOATING base -> shifted inputs for batch>1.
    self.gen_add_code_line("int stride_q_qd = 3*NUM_JOINTS;")
    if single_call_timing:
        # Robust rename (the mjx template appends args after <T>, so match the fn name).
        func_call_start = func_call_start.replace("aba_kernel<","aba_kernel_single_timing<")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                 "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));", \
                                 "gpuErrchkKernel();"])
    # then compute:
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    func_call_code = [func_call, "gpuErrchkKernel();"]
    # wrap function call in timing (if needed)
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"aba\", ABA_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        # then transfer memory back
        self.gen_add_code_lines(["// finally transfer the result back", \
                                "gpuErrchk(cudaMemcpy(hd_data->h_qdd,hd_data->d_qdd,NUM_JOINTS*" + \
                                ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                "gpuErrchkKernel();"])
    # finally report out timing if requested
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("aba"))
    self.gen_add_end_function()

def gen_aba(self):
    # first generate the inner helper
    self.gen_aba_inner()
    # then generate the device wrapper
    self.gen_aba_device()
    # then generate the kernels
    self.gen_aba_kernel(True)
    self.gen_aba_kernel(False)
    # then generate the host wrappers
    self.gen_aba_host(0)
    self.gen_aba_host(1)
    self.gen_aba_host(2)
    

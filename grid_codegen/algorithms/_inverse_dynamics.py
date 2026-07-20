def _id_S_row_coeff(S_desc, row):
    """C++ coefficient string for S[row] of a 1-DoF joint, or None if S[row]==0.

    S_desc is the (tier, ...) tuple from self._id_S_desc(jid): Tier A
    ("A", s_ind, s_sign) contributes sign at s_ind; Tier B ("B", S_vec)
    contributes the dense float at each nonzero row. Used by the skew (Tier-B)
    emit paths in RNEA to fan a dense motion column across rows."""
    if S_desc[0] == "A":
        _, s_ind, s_sign = S_desc
        return str(s_sign) if row == s_ind else None
    S_vec = S_desc[1]
    return ("static_cast<T>(" + repr(float(S_vec[row])) + ")") if S_vec[row] != 0.0 else None


def gen_inverse_dynamics_inner_temp_mem_size(self):
        # The forward f-pass stashes each BODY's I*v product in s_temp indexed by
        # raw body id (6*jid+row, jid in [0, get_num_joints())), so the scratch must
        # hold 6*get_num_joints() floats. Non-mimic keeps the 6*get_num_pos() form
        # (byte-identical: both == NJ fixed; floating's value is never smaller). For
        # mimic robots get_num_joints() > get_num_pos() (mimic joints carry 0 DoF),
        # so 6*get_num_pos() under-sizes s_temp by 6*num_mimic and the I*v writes
        # overflow into the next shared-arena region, corrupting per-body forces.
        if self.robot_has_mimic_joints():
            return 6 * self.robot.get_num_joints()
        n = self.robot.get_num_pos()
        return 6*n

def gen_inverse_dynamics_inner_function_call(self, compute_c = False, use_qdd_input = False, updated_var_names = None):
    var_names = dict( \
        s_c_name = "s_c", \
        s_vaf_name = "s_vaf", \
        s_q_name = "s_q", \
        s_qd_name = "s_qd", \
        s_qdd_name = "s_qdd", \
        s_temp_name = "s_temp", \
        d_f_ext_name = "d_f_ext", \
        gravity_name = "gravity"
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    id_code_start = "inverse_dynamics_inner<T>(" + var_names["s_vaf_name"] + ", " + var_names["s_q_name"] + ", " + var_names["s_qd_name"] + ", "
    # runtime_joint_dynamics: pass d_robotModel as the trailing arg ONLY on the
    # compute_c path (the only one that emits the table-reading bias) AND only when
    # the caller's inner has d_robotModel in scope (it opts in via
    # updated_var_names["d_robotModel_name"]; default "d_robotModel" for callers
    # whose inner already binds that name). Every other call (and every non-runtime
    # build) relies on the trailing `= nullptr` default, so no other call site needs
    # editing and the byte-identity invariant holds.
    _id_rt_jd = ""
    if compute_c and getattr(self, "runtime_joint_dynamics", False):
        _id_rt_jd = ", " + var_names.get("d_robotModel_name", "d_robotModel")
    id_code_end = var_names["s_temp_name"] + ", " + var_names["d_f_ext_name"] + ", " + var_names["gravity_name"] + _id_rt_jd + ");"
    if compute_c:
        id_code_start = id_code_start.replace("(", "(" + var_names["s_c_name"] + ", ")
    else:
        id_code_start = id_code_start.replace("<T>","_vaf<T>")
    # account for thread group and qdd
    if use_qdd_input:
        id_code_start += var_names["s_qdd_name"] + ", "
    id_code_middle = self.gen_insert_helpers_function_call()
    id_code = id_code_start + id_code_middle + id_code_end
    self.gen_add_code_line(id_code)

def gen_inverse_dynamics_joint_dynamics_bias(self):
    """Emit the gated joint-local damping + Coulomb friction bias into s_c.

    tau += damping*qd + friction*sign(qd), per joint, folded into its v-slot.
    EMITTED ONLY when USE_JOINT_DYNAMICS is enabled AND the robot declares
    nonzero damping/friction (both decided at codegen time). With the flag off
    (the DEFAULT) this is a pure no-op, so EVERY robot — damped or not — stays
    byte-identical to the historical emit and consistent with the bare-Pinocchio
    CUDA-equivalence oracle (pin.rnea/pin.aba ignore model.damping/friction).
    Runs serially (thread 0): correctness-first for the rare opt-in robots.
    """
    if not getattr(self, "USE_JOINT_DYNAMICS", False):
        return
    if not (self.robot.robot_has_joint_damping() or self.robot.robot_has_joint_friction()):
        return
    HAS_DAMP = self.robot.robot_has_joint_damping()
    HAS_FRIC = self.robot.robot_has_joint_friction()
    HAS_MIMIC = self.robot_has_mimic_joints()
    fb = self.robot.floating_base
    RUNTIME = getattr(self, "runtime_joint_dynamics", False)
    self.gen_add_code_line("//")
    self.gen_add_code_line("// joint-local viscous damping + Coulomb friction: s_c += b*qd + f*sign(qd)")
    self.gen_add_code_line("//")
    self.gen_add_sync()
    self.gen_add_serial_ops()
    if RUNTIME:
        # runtime_joint_dynamics: read the alpha-FOLDED per-v-slot coefficients from
        # the mutable device table (d_joint_dynamics_params[vs]=damping,
        # [nv+vs]=friction) instead of the baked literal. The table ALREADY holds the
        # folded value (init_joint_dynamics_params == _joint_dynamics_folded_by_vslot),
        # so DO NOT re-apply alpha here — read the cell straight. URDF-initialized, so
        # this is bit-identical to the literal path until set_joint_dynamics is called.
        nv = self.robot.get_num_vel()
        b_vslot, f_vslot = self._joint_dynamics_folded_by_vslot()
        for vs in range(nv):
            b = b_vslot[vs]; fr = f_vslot[vs]
            if b == 0.0 and fr == 0.0:
                continue
            qd = "s_qd[" + str(vs) + "]"
            terms = []
            if b != 0.0:
                terms.append("d_robotModel->d_joint_dynamics_params[" + str(vs) + "] * " + qd)
            if fr != 0.0:
                # exact sign matching np.sign (0 at qd==0): (qd>0) - (qd<0).
                terms.append("d_robotModel->d_joint_dynamics_params[" + str(nv + vs)
                             + "] * static_cast<T>((" + qd + " > static_cast<T>(0)) - ("
                             + qd + " < static_cast<T>(0)))")
            self.gen_add_code_line("s_c[" + str(vs) + "] += " + " + ".join(terms) + ";")
    else:
        for jid in range(self.robot.get_num_joints()):
            b = float(self.robot.get_damping_by_id(jid)) if HAS_DAMP else 0.0
            fr = float(self.robot.get_friction_by_id(jid)) if HAS_FRIC else 0.0
            if b == 0.0 and fr == 0.0:
                continue
            # v-slot this joint folds into (mimic joints share their target's slot;
            # for a floating base the root owns slots 0..5 so actuated joints start
            # at v-slot 6). s_c and s_qd are both indexed by this v-slot, so the
            # bias reads s_qd[vs] and accumulates into s_c[vs].
            if fb and jid == 0:
                continue  # floating root carries no damping/friction
            if HAS_MIMIC:
                vs = self._v_slot_cpp(jid)
                alpha = float(self._alpha_for_jid(jid))
            else:
                vs = self.robot.get_joint_index_v(jid)
                alpha = 1.0
            qd = "s_qd[" + str(vs) + "]"
            terms = []
            if b != 0.0:
                terms.append("static_cast<T>(" + repr(alpha * b) + ") * " + qd)
            if fr != 0.0:
                # exact sign matching np.sign (0 at qd==0): (qd>0) - (qd<0).
                terms.append("static_cast<T>(" + repr(alpha * fr)
                             + ") * static_cast<T>((" + qd + " > static_cast<T>(0)) - ("
                             + qd + " < static_cast<T>(0)))")
            self.gen_add_code_line("s_c[" + str(vs) + "] += " + " + ".join(terms) + ";")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

def gen_inverse_dynamics_inner(self, compute_c = False, use_qdd_input = False):
    n = self.robot.get_num_joints()
    n_bfs_levels = self.robot.get_max_bfs_level() + 1 # starts at 0
    # construct the boilerplate and function definition
    func_params = ["s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = " + str(18*n), \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "s_XI is the pointer to the transformation and inertia matricies ", \
                   "s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = " + \
                            str(self.gen_inverse_dynamics_inner_temp_mem_size()), \
                   "gravity is the gravity constant"]
    func_notes = ["Assumes the XI matricies have already been updated for the given q"]
    func_def_start = "void inverse_dynamics_inner("
    func_def_middle = "T *s_vaf, const T *s_q, const T *s_qd, "
    # d_f_ext: optional GLOBAL per-body external forces (defaults nullptr); the
    # trailing pointer sits just before gravity so the no-fext call is a literal
    # nullptr (dead-code-eliminated) and shared-memory bytes are unchanged.
    # runtime_joint_dynamics: the bias reads the mutable damping/friction table
    # d_robotModel->d_joint_dynamics_params, so thread d_robotModel into the inner
    # as a TRAILING DEFAULTED param (= nullptr) ONLY under that flag. A trailing
    # default keeps every existing call site (id/fd/aba/gradient/so/regressor/...)
    # source-compatible without edits; only the compute_c id-call passes a real
    # pointer (below). The default keeps the signature byte-identical when off.
    if getattr(self, "runtime_joint_dynamics", False):
        func_def_end = "T *s_temp, T *d_f_ext, const T gravity, const robotModel<T> *d_robotModel = nullptr) {"
        func_params.append("d_robotModel is the GPU model pointer (read for the mutable joint-dynamics table; nullptr when unused)")
    else:
        func_def_end = "T *s_temp, T *d_f_ext, const T gravity) {"
    func_params.append("d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr")
    if compute_c:
        func_def_start += "T *s_c,  "
        func_params.insert(0,"s_c is the vector of output torques")
    else:
        func_def_start = func_def_start.replace("(","_vaf(")
        func_notes.append("used to compute vaf as helper values")
    if use_qdd_input:
        func_def_middle += "const T *s_qdd, "
        func_params.insert(-3,"s_qdd is (optional vector of joint accelerations")
    else:
        func_notes.append("optimized for qdd = 0")
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(func_def_middle, func_params, -2)
    func_def = func_def_start + func_def_middle + func_def_end
    # now generate the code
    self.gen_add_func_doc("Compute the RNEA (Recursive Newton-Euler Algorithm)",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    temp_size = self.gen_inverse_dynamics_inner_temp_mem_size()
    self.gen_linalg_smem_setup(temp_size)
    # Mimic-aware velocity/accel reads. Each BODY jid's joint velocity is read
    # from its reduced v-slot scaled by its mimic multiplier (alpha): a mimic
    # joint sees alpha * v_target. For non-mimic robots v_slot(jid)==jid and
    # alpha==1.0, so _id_qd below returns the legacy `s_qd[jid]` verbatim
    # (byte-identical). For mimic robots we emit compile-time per-body lookup
    # tables so the existing parallel fpass structure (with runtime jid select
    # vars on branched levels) folds correctly without restructuring.
    HAS_MIMIC = self.robot_has_mimic_joints()
    if HAS_MIMIC and (compute_c or True):
        # The floating root (jid 0) is a 6-DoF joint whose qd is read through
        # the dedicated fb_col path (never via this single-DoF table), so emit a
        # harmless sentinel (v-slot 0, alpha 1) for it rather than asking the
        # single-DoF _v_slot_cpp helper (which asserts on multi-DoF joints).
        def _vslot_tbl(j):
            if self.robot.floating_base and j == 0:
                return 0
            return self._v_slot_cpp(j)
        vslot_arr = ", ".join(str(_vslot_tbl(j)) for j in range(n))
        alpha_arr = ", ".join(repr(self._alpha_for_jid(j)) for j in range(n))
        self.gen_add_code_line("// mimic per-body v-slot + multiplier tables")
        self.gen_add_code_line("const int s_mimic_vslot[" + str(n) + "] = {" + vslot_arr + "};")
        self.gen_add_code_line("const T s_mimic_alpha[" + str(n) + "] = {" + alpha_arr + "};")
        self.gen_add_code_line("(void)s_mimic_vslot; (void)s_mimic_alpha;")

    HAS_SPHERICAL = self.robot.robot_has_spherical()

    def _v_idx(jid):
        # Scalar reduced v-slot for a single-DoF body `jid` (compile-time int).
        # On a robot with a spherical joint, downstream joints are shifted off
        # `jid` (the spherical consumes 3 v-slots while it is 1 body), so read
        # the parser's dense v-map instead of the raw jid. Byte-identical on
        # all-cardinal robots (v-slot == jid there).
        v = self.robot.get_joint_index_v(jid)
        if isinstance(v, (list, tuple)):
            v = v[0]
        return v

    def _id_qd(jid_expr, qd_name="s_qd"):
        # Read body `jid_expr`'s joint velocity/accel, mimic-folded.
        if HAS_MIMIC:
            return "s_mimic_alpha[" + str(jid_expr) + "] * " + qd_name + "[s_mimic_vslot[" + str(jid_expr) + "]]"
        # Spherical-shifted v-slot for a compile-time-int body id (single-DoF).
        if HAS_SPHERICAL and isinstance(jid_expr, int):
            return qd_name + "[" + str(_v_idx(jid_expr)) + "]"
        return qd_name + "[" + str(jid_expr) + "]"
    #
    # Initial Debug Prints if Requested
    #
    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_line("printf(\"q\\n\"); printMat<T,1," + str(n) + ">(s_q,1);")
        self.gen_add_code_line("printf(\"qd\\n\"); printMat<T,1," + str(n) + ">(s_qd,1);")
        if use_qdd_input:
            self.gen_add_code_line("printf(\"qdd\\n\"); printMat<T,1,6>(s_qdd,1);")
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
        #
        # v and a need to be computed serially by wave
        #
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        # Tier-C spherical (ball) joints: a 3-DoF angular-identity S (cols 0,1,2 =
        # rows 0,1,2). They are NOT skew (the skew path's signed-index helpers
        # raise on multi-DoF), so they get their OWN serial per-joint emit. Each
        # column behaves like a cardinal angular axis reading its v-block slot.
        level_has_spherical = any(self.robot.joint_is_spherical(j) for j in inds)
        # Tier-B (skew axis): any joint at this level whose single-column S is
        # non-cardinal. The signed-index topology helpers raise on such joints,
        # so we route the whole level through the dense-6-vector emit. Cardinal-
        # only levels (every current robot) take the byte-identical fast path.
        # Spherical joints are multi-column (not cardinal) but handled above, so
        # exclude them from the skew classification.
        level_has_skew = any(
            (not self.robot.joint_is_spherical(j)) and (not self.robot.S_is_cardinal_by_id(j))
            for j in inds
        )
        if not level_has_skew and not level_has_spherical:
            parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)
            S_sign_cpp = self.gen_topology_S_sign_for_cpp(inds)
        else:
            parent_ind_cpp, S_ind_cpp, S_sign_cpp = None, None, None

        if bfs_level == 0:
            self.gen_add_code_line("// s_v, s_a where parent is base")
            self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
            self.gen_add_code_line("//     links are: " + ", ".join(link_names))
            # compute the initial v which is just S*qd
            # compute the initial a which is just X*gravity_vec (aka X_last_col*gravity_const) + S*qdd
            comment = "// s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity"
            if use_qdd_input:
                comment += "S[k]*qdd[k]"
            self.gen_add_code_line(comment)
            # load in 0 to v and X*gravity to a in parallel
            # note that depending on S we need to add qd/qdd to one entry
            if level_has_spherical:
                # Tier-C spherical at level 0: v[k]=S*qd (angular rows 0..2 read
                # the joint's 3-wide v-block), a = X*gravity (+ S*qdd). The mxS(v)*v
                # Coriolis term vanishes at level 0 (parent v=0 => v=vJ pure-angular
                # => crm(w)*[w;0]=0), so it is intentionally omitted here. Serial
                # per-joint (correctness-first; spherical robots are rare).
                self.gen_add_serial_ops()
                for jid_val in inds:
                    jid6 = 6 * jid_val
                    if self.robot.joint_is_spherical(jid_val):
                        vblk = self.robot.get_joint_index_v(jid_val)
                        for r in range(6):
                            self.gen_add_code_line("s_vaf[" + str(jid6 + r) + "] = static_cast<T>(0);")
                            self.gen_add_code_line("s_vaf[" + str(n*6 + jid6 + r) + "] = -s_XImats[" + str(6*jid6 + 30 + r) + "]*gravity;")
                        for k in range(3):  # angular-identity columns -> rows 0,1,2
                            self.gen_add_code_line("s_vaf[" + str(jid6 + k) + "] += s_qd[" + str(vblk[k]) + "];")
                            if use_qdd_input:
                                self.gen_add_code_line("s_vaf[" + str(n*6 + jid6 + k) + "] += s_qdd[" + str(vblk[k]) + "];")
                    else:
                        # cardinal single-DoF joint sharing this level with a spherical
                        S_desc = self._id_S_desc(jid_val)
                        qd_term = _id_qd(jid_val)
                        for r in range(6):
                            self.gen_add_code_line("s_vaf[" + str(jid6 + r) + "] = static_cast<T>(0);")
                            self.gen_add_code_line("s_vaf[" + str(n*6 + jid6 + r) + "] = -s_XImats[" + str(6*jid6 + 30 + r) + "]*gravity;")
                        for r in range(6):
                            coeff = _id_S_row_coeff(S_desc, r)
                            if coeff is not None:
                                self.gen_add_code_line("s_vaf[" + str(jid6 + r) + "] += (" + coeff + ") * " + qd_term + ";")
                                if use_qdd_input:
                                    self.gen_add_code_line("s_vaf[" + str(n*6 + jid6 + r) + "] += (" + coeff + ") * " + _id_qd(jid_val, "s_qdd") + ";")
                self.gen_add_end_control_flow()
                self.gen_add_sync()
                continue
            if level_has_skew:
                # Tier B (skew): each joint has a dense S column. Parallelize over
                # the 6*len(inds) rows and dispatch by compile-time jid (mirrors
                # aba/minv) so each joint adds its own dense (or unit) S column;
                # handles multiple (branching) skew joints per BFS level.
                self.gen_add_parallel_loop("ind", str(6*len(inds)))
                self.gen_add_code_line("int row = ind % 6;")
                for i, jid_val in enumerate(inds):
                    self.gen_add_code_line(("if " if i == 0 else "else if ") + "(ind < " + str(6*(i+1)) + ") {", True)
                    jid6 = 6 * jid_val
                    self.gen_add_code_line("s_vaf[" + str(jid6) + " + row] = static_cast<T>(0);")
                    if self.robot.floating_base:
                        self.gen_add_code_line("s_vaf[" + str(n*6 + jid6) + " + row] = (row < 3 ? static_cast<T>(0) : -s_XImats[" + str(6*jid6) + " + 6*row + 5] * gravity);")
                    else:
                        self.gen_add_code_line("s_vaf[" + str(n*6 + jid6) + " + row] = -s_XImats[" + str(6*jid6 + 30) + " + row]*gravity;")
                    S_desc = self._id_S_desc(jid_val)
                    qd_term = _id_qd(jid_val)
                    for r in range(6):
                        coeff = _id_S_row_coeff(S_desc, r)
                        if coeff is not None:
                            ln = "if (row == " + str(r) + "){s_vaf[" + str(jid6 + r) + "] += (" + coeff + ") * " + qd_term + ";"
                            if use_qdd_input:
                                ln += " s_vaf[" + str(n*6 + jid6 + r) + "] += (" + coeff + ") * " + _id_qd(jid_val, "s_qdd") + ";"
                            ln += "}"
                            self.gen_add_code_line(ln)
                    self.gen_add_end_control_flow()
                self.gen_add_end_control_flow()
                self.gen_add_sync()
                continue
            if len(inds) > 1:
                self.gen_add_parallel_loop("ind",str(6*len(inds)))
                self.gen_add_code_line("int row = ind % 6;")
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "<", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
                jid = "jid"
            else:
                self.gen_add_parallel_loop("row",str(6))
                jid = str(inds[0])
            self.gen_add_code_lines(["int jid6 = 6*" + jid + ";", \
                                        "s_vaf[jid6 + row] = static_cast<T>(0);",])
            if self.robot.floating_base:
                root_gravity_code = "(row < 3 ? static_cast<T>(0) : -s_XImats[6*jid6 + 6*row + 5] * gravity)"
                self.gen_add_code_line("s_vaf[" + str(n*6) + " + jid6 + row] = " + root_gravity_code + ";")
            else:
                self.gen_add_code_line("s_vaf[" + str(n*6) + " + jid6 + row] = -s_XImats[6*jid6 + 30 + row]*gravity;")
            # then add in qd and qdd
            if S_ind_cpp == '-1': # floating base uses the root motion subspace, not raw row-wise copies
                qd_qdd_code = "int fb_col = row < 3 ? row + 3 : row - 3; s_vaf[jid6 + row] = s_qd[fb_col];"
            else:
                qd_qdd_code = "if (row == " + S_ind_cpp + "){s_vaf[jid6 + " + S_ind_cpp + "] += (" + S_sign_cpp + ") * " + _id_qd(jid) + ";}"
            if use_qdd_input:
                if S_ind_cpp == '-1':
                    qd_qdd_code += " s_vaf[" + str(n*6) + " + jid6 + row] += s_qdd[fb_col];"
                else:
                    qd_qdd_code = qd_qdd_code.replace("}", " s_vaf[" + str(n*6) + " + jid6 + " + S_ind_cpp + "] += (" + S_sign_cpp + ") * " + _id_qd(jid, "s_qdd") + ";}")
            self.gen_add_code_line(qd_qdd_code)
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            
            # add debug if requested
            if self.DEBUG_MODE:
                self.gen_add_sync()
                self.gen_add_serial_ops()
                for ind in inds:
                    self.gen_add_code_line("printf(\"s_v[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_vaf[6*" + str(ind) + "],1);")
                    self.gen_add_code_line("printf(\"s_a[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_vaf[" + str(6*n) + " + 6*" + str(ind) + "],1);")
                self.gen_add_end_control_flow()
                self.gen_add_sync()

        else:
            self.gen_add_code_line("// s_v and s_a where bfs_level is " + str(bfs_level))
            self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
            self.gen_add_code_line("//     links are: " + ", ".join(link_names))
            # note the update
            comment = "// s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k]"
            comment += " + S[k]*qdd[k] + mxS[k](v[k])*qd[k]" if use_qdd_input else " + mxS[k](v[k])*qd[k]"
            self.gen_add_code_line(comment)
            if level_has_spherical:
                # Tier-C spherical mid-chain: v[jid]=X*v[parent]+S*qd, a[jid]=X*a[parent]
                # (+S*qdd) + crm(v[jid])*vJ, with vJ=S*qd (pure-angular). Serial per
                # joint (correctness-first; thread-invariant via single-thread dot rows).
                self.gen_add_serial_ops()
                for jid_val in inds:
                    jid6 = 6 * jid_val
                    parent6 = 6 * self.robot.get_parent_id(jid_val)
                    xoff = 36 * jid_val
                    # v[jid] = X * v[parent]  (column-major 6x6: X[xoff + 6*c + r])
                    for r in range(6):
                        self.gen_add_code_line("s_vaf[" + str(jid6 + r) + "] = dot_prod<T,6,6,1>(&s_XImats[" + str(xoff + r) + "], &s_vaf[" + str(parent6) + "]);")
                    # a[jid] = X * a[parent]
                    for r in range(6):
                        self.gen_add_code_line("s_vaf[" + str(n*6 + jid6 + r) + "] = dot_prod<T,6,6,1>(&s_XImats[" + str(xoff + r) + "], &s_vaf[" + str(n*6 + parent6) + "]);")
                    if self.robot.joint_is_spherical(jid_val):
                        vblk = self.robot.get_joint_index_v(jid_val)
                        # vJ = S*qd (angular rows 0..2); add to v, build the local vJ
                        self.gen_add_code_line("{ T vJ[6] = {static_cast<T>(0),static_cast<T>(0),static_cast<T>(0),static_cast<T>(0),static_cast<T>(0),static_cast<T>(0)};")
                        for k in range(3):
                            self.gen_add_code_line("  vJ[" + str(k) + "] = s_qd[" + str(vblk[k]) + "];")
                            self.gen_add_code_line("  s_vaf[" + str(jid6 + k) + "] += vJ[" + str(k) + "];")
                            if use_qdd_input:
                                self.gen_add_code_line("  s_vaf[" + str(n*6 + jid6 + k) + "] += s_qdd[" + str(vblk[k]) + "];")
                        # a[jid] += crm(v[jid]) * vJ
                        for r in range(6):
                            close = " }" if r == 5 else ""
                            self.gen_add_code_line("  s_vaf[" + str(n*6 + jid6 + r) + "] += crm_mul<T>(" + str(r) + ", &s_vaf[" + str(jid6) + "], vJ);" + close)
                    else:
                        # cardinal single-DoF joint sharing this level with a spherical
                        S_desc = self._id_S_desc(jid_val)
                        qd_term = _id_qd(jid_val)
                        self.gen_add_code_line("{ T vJ[6] = {static_cast<T>(0),static_cast<T>(0),static_cast<T>(0),static_cast<T>(0),static_cast<T>(0),static_cast<T>(0)};")
                        for r in range(6):
                            coeff = _id_S_row_coeff(S_desc, r)
                            if coeff is not None:
                                self.gen_add_code_line("  vJ[" + str(r) + "] = (" + coeff + ") * " + qd_term + ";")
                                self.gen_add_code_line("  s_vaf[" + str(jid6 + r) + "] += vJ[" + str(r) + "];")
                                if use_qdd_input:
                                    self.gen_add_code_line("  s_vaf[" + str(n*6 + jid6 + r) + "] += (" + coeff + ") * " + _id_qd(jid_val, "s_qdd") + ";")
                        for r in range(6):
                            close = " }" if r == 5 else ""
                            self.gen_add_code_line("  s_vaf[" + str(n*6 + jid6 + r) + "] += crm_mul<T>(" + str(r) + ", &s_vaf[" + str(jid6) + "], vJ);" + close)
                self.gen_add_end_control_flow()
                self.gen_add_sync()
                continue
            # Sibling joints at the SAME bfs level are independent (disjoint, already-computed
            # parents), so we fuse all of this level's per-joint 6x6 row-strided GEMVs into ONE
            # block-cooperative grid_linalg_segmented_row_strided_gemv call instead of a serial
            # Python-unrolled loop. The += S*qd (and += S*qdd) correction folds into the same
            # store via FUSE_SCALED_ADD: a compile-time per-segment selector vector S_sel holds
            # the joint sign at the joint's S index (0 elsewhere), and scalar[seg] = qd[/qdd].
            # This is a pure independent-work reorder -> numerically identical (per-segment GEMV
            # column reduction order is unchanged), but it keeps threads busy across siblings.
            seg = len(inds)
            parents = [self.robot.get_parent_id(jid_val) for jid_val in inds]
            # Tier-A joints expose a signed unit index; Tier-B (skew) joints
            # carry a dense 6-vector S filled directly into the selector below.
            # _id_S_cols returns either ("A", s_ind, s_sign) or ("B", S_vec).
            s_descs = [self._id_S_desc(jid_val) for jid_val in inds]
            # v-slot for each level joint's qd read. Floating: jid+5 (root offset).
            # Spherical robots: the dense v-map (downstream joints shifted off jid).
            # All-cardinal fixed base: jid (byte-identical).
            qd_idxs = [
                str(jid_val + 5) if self.robot.floating_base
                else (str(_v_idx(jid_val)) if HAS_SPHERICAL else str(jid_val))
                for jid_val in inds
            ]
            tag = "lvl" + str(bfs_level)
            # compile-time descriptor / selector arrays for this level (element offsets)
            a_off = ", ".join(str(36*jid_val) for jid_val in inds)
            self.gen_add_code_line(f"static const int seg_a_off_{tag}[{seg}] = {{{a_off}}};")
            v_x_off = ", ".join(str(6*p) for p in parents)
            v_y_off = ", ".join(str(6*jid_val) for jid_val in inds)
            self.gen_add_code_line(f"static const int seg_v_x_off_{tag}[{seg}] = {{{v_x_off}}};")
            self.gen_add_code_line(f"static const int seg_v_y_off_{tag}[{seg}] = {{{v_y_off}}};")
            # per-segment 6-vector selector: sign at the joint S index, 0 elsewhere (seg_s_off = 6*seg)
            sel_vals = []
            for i in range(seg):
                row_vals = ["static_cast<T>(0)"]*6
                desc = s_descs[i]
                if desc[0] == "A":
                    _, s_ind, s_sign = desc
                    row_vals[s_ind] = f"static_cast<T>({s_sign})"
                else:  # Tier B: dense skew column
                    S_vec = desc[1]
                    for r in range(6):
                        if S_vec[r] != 0.0:
                            row_vals[r] = f"static_cast<T>({repr(float(S_vec[r]))})"
                sel_vals.extend(row_vals)
            self.gen_add_code_line(f"static const int seg_s_off_{tag}[{seg}] = {{{', '.join(str(6*i) for i in range(seg))}}};")
            self.gen_add_code_line(f"static const T S_sel_{tag}[{6*seg}] = {{{', '.join(sel_vals)}}};")
            # build scalar[seg] = s_qd[qd_idx] in s_temp (free during the forward pass)
            # Mimic: scalar = alpha_jid * s_qd[v_slot(jid)] (fold the multiplier into
            # the scalar; the S_sel selector keeps just the sign).
            self.gen_add_serial_ops()
            for i in range(seg):
                if HAS_MIMIC:
                    self.gen_add_code_line(f"s_temp[{i}] = {_id_qd(inds[i])};")
                else:
                    self.gen_add_code_line(f"s_temp[{i}] = s_qd[{qd_idxs[i]}];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            # s_v[k] = X[k]*v[parent_k] (+= sign*qd[k] at S index) for all level joints at once
            self.gen_add_code_line(f"grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>({seg}, seg_a_off_{tag}, seg_v_x_off_{tag}, seg_v_y_off_{tag}, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_{tag}, S_sel_{tag}, s_temp, s_linalg_smem);")
            # a[jid] = X[jid]*a[parent] (+= sign*qdd[k] at S index if use_qdd_input)
            a_x_off = ", ".join(str(6*n + 6*p) for p in parents)
            a_y_off = ", ".join(str(6*n + 6*jid_val) for jid_val in inds)
            self.gen_add_code_line(f"static const int seg_a_x_off_{tag}[{seg}] = {{{a_x_off}}};")
            self.gen_add_code_line(f"static const int seg_a_y_off_{tag}[{seg}] = {{{a_y_off}}};")
            if use_qdd_input:
                # rebuild scalar[seg] = s_qdd[qd_idx] in s_temp, then fuse the += S*qdd
                self.gen_add_serial_ops()
                for i in range(seg):
                    if HAS_MIMIC:
                        self.gen_add_code_line(f"s_temp[{i}] = {_id_qd(inds[i], 's_qdd')};")
                    else:
                        self.gen_add_code_line(f"s_temp[{i}] = s_qdd[{qd_idxs[i]}];")
                self.gen_add_end_control_flow()
                self.gen_add_sync()
                self.gen_add_code_line(f"grid_linalg_segmented_row_strided_gemv<T,6,6,6,true>({seg}, seg_a_off_{tag}, seg_a_x_off_{tag}, seg_a_y_off_{tag}, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0), seg_s_off_{tag}, S_sel_{tag}, s_temp, s_linalg_smem);")
            else:
                self.gen_add_code_line(f"grid_linalg_segmented_row_strided_gemv<T,6,6,6>({seg}, seg_a_off_{tag}, seg_a_x_off_{tag}, seg_a_y_off_{tag}, s_XImats, s_vaf, s_vaf, static_cast<T>(1), static_cast<T>(0));")

            # add debug if requested
            if self.DEBUG_MODE:
                self.gen_add_sync()
                self.gen_add_serial_ops()
                for ind in inds:
                    self.gen_add_code_line("printf(\"s_v[" + str(ind) + "] = X*s_v[" + str(self.robot.get_parent_id(ind)) + "] + S*qd[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_vaf[6*" + str(ind) + "],1);")
                    if use_qdd_input:
                        self.gen_add_code_line("printf(\"s_a[" + str(ind) + "] = X*s_a[" + str(self.robot.get_parent_id(ind)) + "] + S*qdd[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_vaf[" + str(6*n) + " + 6*" + str(ind) + "],1);")
                    else:
                        self.gen_add_code_line("printf(\"s_a[" + str(ind) + "] = X*s_a[" + str(self.robot.get_parent_id(ind)) + "]\\n\"); printMat<T,1,6>(&s_vaf[" + str(6*n) + " + 6*" + str(ind) + "],1);")
                self.gen_add_end_control_flow()
                self.gen_add_sync()
            self.gen_add_code_line("// sync before a += MxS(v)*qd[S] ")
            self.gen_add_sync()

            if level_has_skew:
                # Tier B: a += crm(v) * (S * qd). With a dense S column we cannot
                # pick a precomputed mx{col}; emit the generic crm-times-dense-S.
                # Loop over every joint in the level (branching ok); each joint
                # with its own compile-time dense (or unit) S column.
                self.gen_add_serial_ops()
                for jid in inds:
                    S_desc = self._id_S_desc(jid)
                    S_arr = "{" + ", ".join(
                        ((_id_S_row_coeff(S_desc, r) or "static_cast<T>(0)")) for r in range(6)) + "}"
                    qd_term = ("s_qd[" + str(jid + 5) + "]" if self.robot.floating_base
                               else (_id_qd(jid) if HAS_MIMIC else "s_qd[" + str(jid) + "]"))
                    self.gen_add_code_line("{ const T S_skew[6] = " + S_arr + ";")
                    self.gen_add_code_line("  mxS_general_peq_scaled<T>(&s_vaf[" + str(6*n + 6*jid) + "], &s_vaf[" + str(6*jid) + "], S_skew, " + qd_term + "); }")
                self.gen_add_end_control_flow()
                self.gen_add_sync()
                continue

            # attempt to do as much of the Mx in parallel as possible (will branch on different S but that is inevitable)
            self.gen_add_parallel_loop("ind",str(len(inds)))
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "==", [str(i) for i in range(len(inds))], select_var_vals)
                dst_name = "&s_vaf[" + str(6*n) + " + 6*jid]"
                src_name = "&s_vaf[6*jid]"
                if self.robot.floating_base: scale_name = "(" + S_sign_cpp + ") * s_qd[jid + 5]" # dof offset for fb
                elif HAS_MIMIC: scale_name = "(" + S_sign_cpp + ") * " + _id_qd("jid")
                else: scale_name = "(" + S_sign_cpp + ") * s_qd[jid]"
            else:
                jid = inds[0]
                dst_name = "&s_vaf[" + str(6*n + 6*jid) + "]"
                src_name = "&s_vaf[" + str(6*jid) + "]"
                if self.robot.floating_base: scale_name = "(" + S_sign_cpp + ") * s_qd[" + str(jid + 5) + "]" # dof offset due to fb
                elif HAS_MIMIC: scale_name = "(" + S_sign_cpp + ") * " + _id_qd(jid)
                elif HAS_SPHERICAL: scale_name = "(" + S_sign_cpp + ") * s_qd[" + str(_v_idx(jid)) + "]"
                else: scale_name = "(" + S_sign_cpp + ") * s_qd[" + str(jid) + "]"
            updated_var_names = dict(S_ind_name = S_ind_cpp, s_dst_name = dst_name, s_src_name = src_name, s_scale_name = scale_name)
            self.gen_mx_func_call_for_cpp(inds, PEQ_FLAG = True, SCALE_FLAG = True, updated_var_names = updated_var_names)
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            # add debug if requested
            if self.DEBUG_MODE:
                self.gen_add_sync()
                self.gen_add_serial_ops()
                for ind in inds:
                    self.gen_add_code_line("printf(\"s_a[" + str(ind) + "] += MxS(s_v[" + str(ind) + "])\\n\"); printMat<T,1,6>(&s_vaf[" + str(6*n) + " + 6*" + str(ind) + "],1);")
                self.gen_add_end_control_flow()
                self.gen_add_sync()
    #
    # then compute all f in parallel
    #
    inds = list(range(n))
    self.gen_add_code_line("//")
    self.gen_add_code_line("// s_f in parallel given all v, a")
    self.gen_add_code_line("//")
    self.gen_add_code_line("// s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]")
    self.gen_add_code_line("// start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]")
    self.gen_add_parallel_loop("ind",str(6*2*n))
    self.gen_add_code_line("int row = ind % 6; int comp = ind / 6; int jid = comp % " + str(n) + ";")
    self.gen_add_code_line("bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * " + str(6*n) + " + jid6;")
    self.gen_add_code_line("T *dst = IaFlag ? &s_vaf[" + str(12*n) + "] : s_temp;")
    self.gen_add_code_line("// compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync")
    self.gen_add_code_line("dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[" + str(36*n) + " + 6*jid6 + row], &s_vaf[vaOffset]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        for ind in inds:
            self.gen_add_code_line("printf(\"s_f[" + str(ind) + "] = I*s_a[" + str(ind) + "])\\n\"); printMat<T,1,6>(&s_vaf[" + str(12*n) + " + 6*" + str(ind) + "],1);")
            self.gen_add_code_line("printf(\"I*s_v[" + str(ind) + "])\\n\"); printMat<T,1,6>(&s_temp[6*" + str(ind) + "],1);")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
    self.gen_add_code_line("// finish with s_f[k] += fx(v[k])*Iv[k]")
    self.gen_add_parallel_loop("jid",str(len(inds)))
    self.gen_add_code_line("int jid6 = 6*jid;")
    self.gen_add_code_line("fx_times_v_peq<T>(&s_vaf[" + str(12*n) + " + jid6], &s_vaf[jid6], &s_temp[jid6]);")
    # External forces (opt-in): subtract the per-body local-frame f_ext from
    # the just-finished per-body force. d_f_ext is GLOBAL, body-major, length
    # 6*NUM_BODIES, ordered [angular; linear] (same layout as s_vaf force).
    # Reading it here (the single force-finish site) keeps every byte of
    # shared memory unchanged; nullptr -> no-op (dead-code-eliminated).
    self.gen_add_code_line("if (d_f_ext != nullptr) {")
    self.gen_add_code_line("    for (int r = 0; r < 6; r++) { s_vaf[" + str(12*n) + " + jid6 + r] -= d_f_ext[jid6 + r]; }")
    self.gen_add_code_line("}")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        for ind in inds:
            self.gen_add_code_line("printf(\"s_f[" + str(ind) + "] += fx(v[" + str(ind) + "])*I*v[" + str(ind) + "])\\n\"); printMat<T,1,6>(&s_vaf[" + str(12*n) + " + 6*" + str(ind) + "],1);")
        self.gen_add_code_line("printf(\"s_f forward pass\\n\"); printMat<T,6," + str(n) + ">(&s_vaf[" + str(12*n) + "],6);")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
    #
    # Then compute the Backward Pass again in bfs waves
    #
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Backward Pass")
    self.gen_add_code_line("//")
    # backward pass start by updating all f by bfs_level
    for bfs_level in range(n_bfs_levels - 1, 0, -1): # don't consider level 0 as parent is root
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        parent_ind_cpp, _  = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)
        
        self.gen_add_code_line("// s_f update where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))
        # update f parent from f (sequential GEMVs are safe even for repeated parents)
        self.gen_add_code_line("// s_f[parent_k] += X[k]^T*f[k]")
        for jid_val in inds:
            parent_val = self.robot.get_parent_id(jid_val)
            self.gen_add_code_line(f"grid_linalg_gemv<T,6,6,true>(&s_XImats[{36*jid_val}], &s_vaf[{12*n + 6*jid_val}], &s_vaf[{12*n + 6*parent_val}], static_cast<T>(1), static_cast<T>(1));")

        if self.DEBUG_MODE:
            self.gen_add_sync()
            self.gen_add_serial_ops()
            for ind in inds:
                self.gen_add_code_line("printf(\"s_f[" + str(self.robot.get_parent_id(ind)) + "] += X^T*s_f[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_vaf[" + str(12*n) + " + 6*" + str(self.robot.get_parent_id(ind)) + "],1);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()

    if compute_c and self.robot_has_mimic_joints():
        # Mimic-aware c extraction (serial fold). Multiple bodies can share one
        # velocity slot (a mimicked joint and its mimics), so c[v_slot] is an
        # ACCUMULATE over bodies scaled by each body's mimic multiplier alpha
        # (1.0 for non-mimic). This mirrors RBDReference.rnea_bpass:
        #   c[v_slot(jid)] += alpha_jid * (S_sign * f[jid][S_ind]).
        # Serial (thread 0) because v-slots collide; this path runs only for
        # mimic robots where correctness — not perf — is the goal.
        self.gen_add_code_line("//")
        self.gen_add_code_line("// s_c extracted serially (mimic-aware S*f accumulate into v-slot)")
        self.gen_add_code_line("//")
        # The backward pass writes parent forces into s_vaf via block-cooperative
        # GEMVs (all threads). Sync so thread 0's serial fold below sees every
        # thread's f writes — without this the root bodies' c races on stale f.
        self.gen_add_sync()
        self.gen_add_serial_ops()
        for vs in range(self.robot.get_num_vel()):
            self.gen_add_code_line("s_c[" + str(vs) + "] = static_cast<T>(0);")
        for jid in range(n):
            # The floating root (jid 0 on a floating base) is a 6-DoF joint with
            # a full 6x6 motion subspace S; it is NEVER a mimic joint and its
            # six DoFs map directly to v-slots 0..5 via c[k] = sum_row S[row,k]*f.
            # Fold each of its DoFs from the S matrix (alpha == 1). All other
            # bodies are single-DoF, possibly mimic: c[v_slot] += alpha * S_sign*f.
            if self.robot.floating_base and jid == 0:
                import numpy as _np
                S0 = _np.array(self.robot.get_S_by_id(0))
                for k in range(S0.shape[1]):
                    rows = _np.nonzero(S0[:, k])[0]
                    for row in rows:
                        sgn = float(S0[row, k])
                        self.gen_add_code_line(
                            "s_c[" + str(k) + "] += static_cast<T>(" + repr(sgn)
                            + ") * s_vaf[" + str(12*n + 6*jid + int(row)) + "];")
                continue
            vs = self._v_slot_cpp(jid)
            s_ind = self.robot.get_S_index_by_id(jid)
            s_sign = self.robot.get_S_sign_by_id(jid)
            alpha = self._alpha_for_jid(jid)
            coeff = float(s_sign) * float(alpha)
            self.gen_add_code_line(
                "s_c[" + str(vs) + "] += static_cast<T>(" + repr(coeff) + ") * s_vaf["
                + str(12*n + 6*jid + s_ind) + "];")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_inverse_dynamics_joint_dynamics_bias()
        self.gen_add_end_function()
        return
    if compute_c and self.robot.robot_has_spherical():
        # Tier-C spherical (non-mimic): c[inds_v] = S^T f. Spherical's S is the
        # angular identity (cols 0..2 = rows 0..2), so c[v_block[k]] = f[k] for
        # k=0,1,2; cardinal single-DoF bodies use their signed unit row. Serial
        # per joint into the joint's reduced v-slot(s) (correctness-first).
        self.gen_add_code_line("//")
        self.gen_add_code_line("// s_c extracted serially (Tier-C spherical S^T f)")
        self.gen_add_code_line("//")
        self.gen_add_sync()
        self.gen_add_serial_ops()
        for jid in range(n):
            if self.robot.joint_is_spherical(jid):
                vblk = self.robot.get_joint_index_v(jid)
                for k in range(3):  # angular-identity columns -> rows 0,1,2 of f
                    self.gen_add_code_line(
                        "s_c[" + str(vblk[k]) + "] = s_vaf[" + str(12*n + 6*jid + k) + "];")
            else:
                vs = self.robot.get_joint_index_v(jid)
                if isinstance(vs, (list, tuple)):
                    vs = vs[0]
                s_ind = self.robot.get_S_index_by_id(jid)
                s_sign = self.robot.get_S_sign_by_id(jid)
                self.gen_add_code_line(
                    "s_c[" + str(vs) + "] = static_cast<T>(" + repr(float(s_sign))
                    + ") * s_vaf[" + str(12*n + 6*jid + s_ind) + "];")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_inverse_dynamics_joint_dynamics_bias()
        self.gen_add_end_function()
        return
    if compute_c and self.robot.robot_has_skew_axis():
        # Tier B (skew axis, non-mimic): c[k] = S[k]^T f[k] with a dense S.
        # Serial dense dot per joint (correctness-first; skew robots are rare).
        # Non-floating fixed-base: dof_id == jid.
        self.gen_add_code_line("//")
        self.gen_add_code_line("// s_c extracted serially (Tier-B dense S^T f)")
        self.gen_add_code_line("//")
        self.gen_add_sync()
        self.gen_add_serial_ops()
        for jid in range(n):
            S_desc = self._id_S_desc(jid)
            terms = []
            for r in range(6):
                coeff = _id_S_row_coeff(S_desc, r)
                if coeff is not None:
                    terms.append("(" + coeff + ") * s_vaf[" + str(12*n + 6*jid + r) + "]")
            self.gen_add_code_line("s_c[" + str(jid) + "] = " + (" + ".join(terms) if terms else "static_cast<T>(0)") + ";")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_inverse_dynamics_joint_dynamics_bias()
        self.gen_add_end_function()
        return
    if compute_c:
        # then extract all c in parallel
        _, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(NO_GRAD_FLAG = True)
        S_sign_cpp = self.gen_topology_S_sign_for_cpp()
        self.gen_add_code_line("//")
        self.gen_add_code_line("// s_c extracted in parallel (S*f)")
        self.gen_add_code_line("//")
        self.gen_add_parallel_loop("dof_id",str(self.robot.get_num_vel())) # one component for each dof
        if 'jid' in S_ind_cpp: S_ind_cpp = S_ind_cpp.replace('jid', 'dof_id') 
        if 'jid' in S_sign_cpp: S_sign_cpp = S_sign_cpp.replace('jid', 'dof_id')
        if self.robot.floating_base:
            if '+' in S_ind_cpp: S_ind_cpp = S_ind_cpp.replace(']', ' - 5]') # offset back to the beginning of the S_inds
            if '+' in S_sign_cpp: S_sign_cpp = S_sign_cpp.replace(']', ' - 5]') # offset back to the beginning of the S_inds
            self.gen_add_code_line("int fb_offset = (dof_id > 5) * (6 * (dof_id - 5)); // First 6 DOF belong to floating base")
            self.gen_add_code_line("s_c[dof_id] = (" + S_sign_cpp + ") * s_vaf[" + str(12*n) + " + fb_offset + " + S_ind_cpp + "];")
        else: self.gen_add_code_line("s_c[dof_id] = (" + S_sign_cpp + ") * s_vaf[" + str(12*n) + " + 6*dof_id + " + S_ind_cpp + "];")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_inverse_dynamics_joint_dynamics_bias()
    self.gen_add_end_function()

def gen_inverse_dynamics_device_temp_mem_size(self, compute_c = False):
    n = self.robot.get_num_pos()
    # s_vaf and the XImats scratch are indexed by RAW body id (jid in
    # [0, get_num_joints())) inside the inner, so for mimic robots (where
    # get_num_joints() > get_num_pos()) they must be sized by the body count or
    # the f-block writes for the extra mimic bodies overflow into the next arena
    # region (s_XImats), corrupting the low-jid X matrices. Non-mimic robots have
    # get_num_joints() == get_num_pos() (fixed) or the legacy value was already
    # >= the body count (floating), so gate on mimic to stay byte-identical.
    nb = self.robot.get_num_joints() if self.robot_has_mimic_joints() else n
    wrapper_size = (18*nb if compute_c else 0) + self.gen_topology_helpers_size() + 72*nb # for XImats
    return self.gen_inverse_dynamics_inner_temp_mem_size() + wrapper_size

def gen_inverse_dynamics_device(self, compute_c = False, use_qdd_input = False):
    n = self.robot.get_num_pos()
    # construct the boilerplate and function definition
    func_params = ["s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr", \
                   "gravity is the gravity constant"]
    func_notes = []
    func_def_start = "void inverse_dynamics_device("
    func_def_middle = "const T *s_q, const T *s_qd, "
    func_def_end = "const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity) {"
    if compute_c:
        func_def_start += "T *s_c,  "
        func_params.insert(0,"s_c is the vector of output torques")
    else:
        func_def_start = func_def_start.replace("_device(","_vaf_device(")
        func_def_start += "T *s_vaf, "
        func_notes.append("used to compute vaf as helper values")
    if use_qdd_input:
        func_def_middle += "const T *s_qdd, "
        func_params.insert(-2,"s_qdd is the vector of joint accelerations")
    else:
        func_notes.append("optimized for qdd = 0")
    func_def = func_def_start + func_def_middle + func_def_end
    # then generate the code (shared device-wrapper skeleton; B+C §1.1)
    shared_mem_size = self.gen_inverse_dynamics_inner_temp_mem_size()
    # s_vaf is indexed by RAW body id inside the inner (v/a/f blocks each span
    # get_num_joints() bodies), so for mimic robots it must be 18*get_num_joints()
    # or the high-body f writes overflow into the next arena region (s_XImats) and
    # silently corrupt the low-jid X matrices. Non-mimic robots keep the legacy
    # 18*get_num_pos() (== body count for fixed; >= it for floating) byte-identical.
    nb_vaf = self.robot.get_num_joints() if self.robot_has_mimic_joints() else n
    extra_t_buffers = [("s_vaf", 18*nb_vaf)] if compute_c else []
    self.gen_device_wrapper(
        "Compute the RNEA (Recursive Newton-Euler Algorithm)", func_def,
        shared_mem_size,
        lambda: self.gen_inverse_dynamics_inner_function_call(compute_c, use_qdd_input),
        func_notes = func_notes, func_params = func_params,
        extra_t_buffers = extra_t_buffers, include_linalg_scratch = True)

def gen_inverse_dynamics_kernel(self, use_qdd_input = False, single_call_timing = False):
    n = self.robot.get_num_pos()
    compute_c = True
    # define function def and params
    func_params = ["d_c is the vector of output torques", \
                   "d_q_dq is the vector of joint positions and velocities", \
                   "stride_q_qd is the stide between each q, qd", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr", \
                   "gravity is the gravity constant,"
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    func_def_start = "void inverse_dynamics_kernel(T *d_c, const T *d_q_qd, const int stride_q_qd, "
    func_def_end = "T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    if use_qdd_input:
        func_def_start += "const T *d_qdd, "
        func_params.insert(-3,"d_qdd is the vector of joint accelerations")
    else:
        func_notes.append("optimized for qdd = 0")
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    # then generate the code
    self.gen_add_func_doc("Compute the RNEA (Recursive Newton-Euler Algorithm)",\
                          func_notes,func_params,None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag. Added
    # LAST so existing positional <T,TIER> call sites are unaffected; the default
    # (false) instantiation if-constexpr-elides the epilogue -> byte-identical PTX.
    mjx_kernel = self.robot.floating_base and use_qdd_input
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # add shared memory variables. s_vaf is body-indexed (18*NJ); for mimic
    # robots (NJ > n) size it 18*NJ so the inner's body f-writes never overflow
    # into the XImats region. Non-mimic keeps the legacy 18*n byte-identical.
    _kvaf = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else n)
    extra_t_buffers = [("s_q_qd", 2*n), ("s_c", n), ("s_vaf", _kvaf)]
    if use_qdd_input:
        extra_t_buffers.append(("s_qdd", n))
    shared_mem_size = self.gen_inverse_dynamics_inner_temp_mem_size()
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size, extra_t_buffers = extra_t_buffers, include_linalg_scratch=True)
    self.gen_add_code_line("T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(n) + "];")
    if not single_call_timing:
        # load to shared mem and loop over blocks to compute all requested comps
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",block_level = True)
        if use_qdd_input:
            self.gen_kernel_load_inputs("q_qd",str(2*n),"qdd",str(n),stride="stride_q_qd",stride2=str(n))
        else:
            self.gen_kernel_load_inputs("q_qd",str(2*n),stride="stride_q_qd")
        # mjx input convert (before XImats so X[0] uses the reordered quaternion)
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert(qdd_name="s_qdd")
            self.gen_add_end_control_flow()
        # compute
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_inverse_dynamics_inner_function_call(compute_c,use_qdd_input)
        self.gen_add_sync()
        # mjx output: tau is a covector -> base-linear rows rotate by R
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_base_rotate("s_c")
            self.gen_add_end_control_flow()
        # save to global
        self.gen_kernel_save_result("c",str(n),stride=str(n))
        self.gen_add_end_control_flow()
    else:
        #repurpose NUM_TIMESTEPS for number of timing reps
        if use_qdd_input:
            self.gen_kernel_load_inputs("q_qd",str(2*n),"qdd",str(n))
        else:
            self.gen_kernel_load_inputs("q_qd",str(2*n))
        # then compute in loop for timing
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        if use_qdd_input:
            self.gen_anti_licm_input_reload("q_qd",str(2*n),"qdd",str(n),feedback_from="c")
        else:
            self.gen_anti_licm_input_reload("q_qd",str(2*n),feedback_from="c")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_inverse_dynamics_inner_function_call(compute_c,use_qdd_input)
        self.gen_anti_licm_output_write("c")
        self.gen_add_end_control_flow()
        # save to global
        self.gen_kernel_save_result("c",str(n))
    self.gen_add_end_function()

def gen_inverse_dynamics_host(self, mode = 0):
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
    func_def_start = "void inverse_dynamics(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
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
    # MUJOCO_OUTPUT (floating only) host template flag: forwarded to the with-qdd
    # kernel launch (the only mjx-capable overload). Added LAST so existing
    # positional template args are unaffected; default false -> byte-identical.
    mjx_host = self.robot.floating_base
    if mjx_host:
        self.gen_add_code_line("template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    else:
        self.gen_add_code_line("template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"inverse_dynamics requires all-data or dynamics gridData\");")
    func_call_start = "inverse_dynamics_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,"
    # the with-qdd launch is the mjx-capable overload: forward MUJOCO_OUTPUT (and
    # the caller-chosen tier, which it must name positionally to reach the trailing flag).
    qdd_kernel_tmpl = "inverse_dynamics_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "inverse_dynamics_kernel<T, RESOURCE_TIER>"
    func_call_qdd_start = qdd_kernel_tmpl + "<<<block_dimms,thread_dimms,INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,"
    func_call_end = "hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("inverse_dynamics_kernel<","inverse_dynamics_kernel_single_timing<")
        func_call_qdd_start = func_call_qdd_start.replace("inverse_dynamics_kernel<","inverse_dynamics_kernel_single_timing<")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                 "int stride_q_qd;", \
                                 "if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; " + \
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}", \
                                 "else {stride_q_qd = 3*NUM_JOINTS; " + \
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}", \
                                 "if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[1]));}", \
                                 "gpuErrchkKernel();"])
    else:
        self.gen_add_code_line("int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS: 3*NUM_JOINTS;")
    # then compute but adjust for compressed mem and qdd usage
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    func_call_with_qdd = func_call_qdd_start + "hd_data->d_qdd, " + func_call_end
    # add in compressed mem adjusts
    func_call_mem_adjust = "    if (USE_COMPRESSED_MEM) {" + func_call + "}"
    func_call_mem_adjust2 = "    else                    {" + func_call.replace("hd_data->d_q_qd","hd_data->d_q_qd_u") + "}"
    func_call_with_qdd_mem_adjust = "    if (USE_COMPRESSED_MEM) {" + func_call_with_qdd + "}"
    func_call_with_qdd_mem_adjust2 = "    else                    {" + func_call_with_qdd.replace("hd_data->d_q_qd","hd_data->d_q_qd_u") + "}"
    # compule into a set of code
    func_call_code = ["if (USE_QDD_FLAG) {", func_call_with_qdd_mem_adjust, func_call_with_qdd_mem_adjust2, "}", \
                      "else {", func_call_mem_adjust, func_call_mem_adjust2, "}", "gpuErrchkKernel();"]
    # wrap function call in timing (if needed)
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"inverse_dynamics\", INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        # then transfer memory back
        self.gen_add_code_lines(["// finally transfer the result back", \
                                 "gpuErrchk(cudaMemcpy(hd_data->h_c,hd_data->d_c,NUM_JOINTS*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchkKernel();"])
    # finally report out timing if requested
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("inverse_dynamics"))
    self.gen_add_end_function()

def gen_inverse_dynamics(self):
    # first generate the inner helpers
    self.gen_inverse_dynamics_inner(True,True)
    self.gen_inverse_dynamics_inner(True,False)
    self.gen_inverse_dynamics_inner(False,True)
    self.gen_inverse_dynamics_inner(False,False)
    # then generate the device wrappers
    self.gen_inverse_dynamics_device(True,True)
    self.gen_inverse_dynamics_device(True,False)
    self.gen_inverse_dynamics_device(False,True)
    self.gen_inverse_dynamics_device(False,False)
    # then generate the kernels
    self.gen_inverse_dynamics_kernel(True,True)
    self.gen_inverse_dynamics_kernel(True,False)
    self.gen_inverse_dynamics_kernel(False,True)
    self.gen_inverse_dynamics_kernel(False,False)
    # then the host launch wrappers
    self.gen_inverse_dynamics_host(0)
    self.gen_inverse_dynamics_host(1)
    self.gen_inverse_dynamics_host(2)
    # NOTE: `inverse_dynamics` IS the RNEA (Recursive Newton-Euler Algorithm). There is
    # deliberately NO `grid::rnea` alias symbol — the single canonical name is
    # `inverse_dynamics` (clean-break API). The RNEA name is kept greppable via this
    # comment and the per-function docstrings above ("Compute the RNEA ...").

def _idg_Svec_cpp(S_vec):
    """C++ brace-init for a dense 6-vector motion subspace column (Tier-B skew
    emit in the dense serial ID-gradient inner)."""
    return "{" + ", ".join("static_cast<T>(" + repr(float(c)) + ")" for c in S_vec) + "}"


def _idg_damping_diag_cpp(self, n):
    """Emit the gated joint-local DAMPING contribution to the velocity gradient.

    The ID value path folds ``c += damping*qd + friction*sign(qd)`` into s_c
    (gen_inverse_dynamics_joint_dynamics_bias). Differentiating that bias:
      * ``d(damping*qd)/dqd = damping`` -> a DIAGONAL term on the dqd half of
        s_dc_du (the dc_dqd block); damping is q-INDEPENDENT so there is NO
        dc_dq term.
      * ``d(friction*sign(qd))/dqd = 0`` a.e. (subgradient 0 at qd==0) -> NO
        contribution. Hence this is gated on robot_has_joint_damping() ONLY,
        independent of friction.
    s_dc_du is two nv x nv col-major halves (dc_dq at [0,nv*nv), dc_dqd at
    [nv*nv, 2*nv*nv)); the dqd diagonal cell for v-slot vs is
    ``s_dc_du[nv*nv + nv*vs + vs]``. We ``+=`` so multiple joints sharing a
    v-slot (mimic) sum, matching the value path's per-v-slot accumulate.

    Mimic alpha-fold: ONE power of alpha (k^1) -- the value bias scales by
    alpha*damping, so its qd-derivative w.r.t. the SHARED reduced coordinate is
    likewise alpha*damping (d(alpha*b*(alpha*qd_v))/dqd_v would be alpha^2, but
    the bias reads s_qd[vs] (the reduced coord) scaled once by alpha and writes
    once, exactly as gen_inverse_dynamics_joint_dynamics_bias does -> the
    derivative carries a single alpha, matching the value emit's literal
    alpha*b coefficient). The floating root carries no damping (skipped).

    EMITTED ONLY when USE_JOINT_DYNAMICS is enabled AND the robot declares
    nonzero damping. With the flag off (the DEFAULT) this is a pure no-op, so
    every robot stays byte-identical to the historical emit (Gate-A). This sits
    PAST the serial/mimic/spherical/branched extraction forks and operates on
    the FINAL s_dc_du regardless of extraction path.
    """
    if not getattr(self, "USE_JOINT_DYNAMICS", False):
        return
    if not self.robot.robot_has_joint_damping():
        return
    HAS_MIMIC = self.robot_has_mimic_joints()
    fb = self.robot.floating_base
    # collect (v_slot -> summed alpha*damping) so mimic joints sharing a slot
    # fold to a single += into the diagonal cell.
    diag = {}
    for jid in range(self.robot.get_num_joints()):
        if fb and jid == 0:
            continue  # floating root carries no damping
        b = float(self.robot.get_damping_by_id(jid))
        if b == 0.0:
            continue
        if HAS_MIMIC:
            vs = self._v_slot_cpp(jid)
            alpha = float(self._alpha_for_jid(jid))
        else:
            vs = self.robot.get_joint_index_v(jid)
            alpha = 1.0
        diag[vs] = diag.get(vs, 0.0) + alpha * b
    if not diag:
        return
    self.gen_add_code_line("//")
    self.gen_add_code_line("// joint-local viscous damping gradient: dc_dqd diagonal += damping (friction -> 0)")
    self.gen_add_code_line("//")
    self.gen_add_sync()
    # one thread per damped v-slot (parallel; each writes a distinct diagonal cell)
    slots = sorted(diag.keys())
    RUNTIME = getattr(self, "runtime_joint_dynamics", False)
    self.gen_add_parallel_loop("ind", str(len(slots)))
    for k, vs in enumerate(slots):
        # diagonal cell of the dc_dqd half: base nv*nv, col vs, row vs
        cell = n * n + n * vs + vs
        if RUNTIME:
            # runtime_joint_dynamics: read the alpha-FOLDED per-v-slot damping from the
            # mutable device table (damping at [vs]) instead of the baked literal. The
            # table already holds the folded coefficient (same fold as `diag`), so the
            # derivative diagonal is just that cell. Friction -> 0 (subgradient), so only
            # the damping half [vs] is read (never [nv+vs]). Bit-identical until poked.
            rhs = "d_robotModel->d_joint_dynamics_params[" + str(vs) + "]"
        else:
            rhs = "static_cast<T>(" + repr(diag[vs]) + ")"
        self.gen_add_code_line(("if (ind == " + str(k) + ") " if len(slots) > 1 else "")
                               + "s_dc_du[" + str(cell) + "] += " + rhs + ";")
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def _idg_debug_body_buffer(self, jid, base_offset, n, running_sum_cols_per_jid, cols_per_jid):
    """Resolve the (column-block offset, column count) for the per-BODY dv/da/df
    debug printf of body `jid`, on BOTH fixed and floating base.

    DEBUG-ONLY helper. The dv/da/df scratch buffers are laid out per body
    (joint) id, NOT per velocity DOF, so the debug printf must iterate body ids
    (`range(NUM_JOINTS)`), never `range(NUM_VEL)`: on a floating-base robot a
    velocity-DOF index is not a body id (the 6-DoF floating root maps to a single
    body), so indexing a per-jid structure or calling `get_*_by_id` with a
    velocity index returns None / overflows and crashes codegen.

    The buffer layout itself differs by base, mirroring the non-debug emit:
    - FLOATING base: dense per-jid blocks of width `n` (= NUM_VEL) columns at
      `base_offset + 6*n*jid` (matches the `{6*n}*{jid}` addressing the floating
      forward/backward sweeps use).
    - FIXED base: sparsity-compressed blocks at
      `base_offset + 6*running_sum_cols_per_jid[jid]`, each `cols_per_jid[jid]`
      columns wide.
    Returns (offset:int, num_cols:int).
    """
    if self.robot.floating_base:
        return base_offset + 6 * n * jid, n
    return base_offset + 6 * running_sum_cols_per_jid[jid], cols_per_jid[jid]


def _emit_fb_bfs_level_indexing(self, inds, n, dq_flag_line = None):
    """Emit the shared floating-base per-BFS-level index decode used by the
    dv/du, da/du and df/du sweeps, and return the (jid_cpp, parent_jid_cpp)
    C++ handles the caller substitutes downstream.

    Every floating-base sweep opens its level loop the same way: a
    `parallel_loop` over `6*2*n*len(inds)` threads, the `row`/`col` decode
    `int row = ind % 6; int col = (ind / 6) % n;`, and — when a level holds more
    than one joint — an `ind_du` fold plus two `multi_threaded_select`s picking
    this thread's `jid` and `parent_jid` from the level's joint list. Pulling
    this into one helper keeps the three sweeps byte-identical (the emitted CUDA
    is unchanged) while removing the copy-paste.

    The `dq_flag` line's position varies between sweeps: the dv/du and da/du
    sweeps emit it right after the loop opens (pass `dq_flag_line`), while the
    df/du sweep defers it past extra setup (pass None and emit it at the call
    site). Per-sweep extras (e.g. dv/du's `dof_id`) likewise stay at the caller.
    """
    self.gen_add_parallel_loop("ind", str(6*2*n*len(inds)))
    if dq_flag_line is not None:
        self.gen_add_code_line(dq_flag_line)
    self.gen_add_code_line(f"int row = ind % 6; int col = (ind / 6) % {n};")
    if len(inds) > 1:
        self.gen_add_code_line(f'int ind_du = ind % {6*n*len(inds)};')
        breakpoints = [str((idx+1)*n*6) for idx, jid in enumerate(inds)]
        self.gen_add_multi_threaded_select(
            "(ind_du)", "<", breakpoints,
            [("int", "jid", [str(jid) for jid in inds])])
        self.gen_add_multi_threaded_select(
            "(ind_du)", "<", breakpoints,
            [("int", "parent_jid", [str(self.robot.get_parent_id(jid)) for jid in inds])])
        return "jid", "parent_jid"
    return inds[0], self.robot.get_parent_id(inds[0])

def gen_inverse_dynamics_gradient_inner_temp_mem_size(self):
        if self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis() or self.robot.robot_has_spherical():
            # The mimic AND skew (Tier-B) AND spherical (Tier-C) paths emit the DENSE serial fold (6 dense
            # per-body buffers + Iv) rather than the sparse-compressed band, so they
            # need its own (larger) scratch. Big-NB humanoids route this whole pool
            # to d_workspace at the global-temp tier (SCRATCH_IN_SMEM=false).
            return _inverse_dynamics_gradient_mimic_temp_count(self)
        return self.gen_inverse_dynamics_gradient_temp_layout()["full_count"]

def gen_inverse_dynamics_gradient_temp_layout(self):
    n = self.robot.get_num_vel()
    (dva_cols_per_partial, _, _, df_cols_per_partial, _, _, _) = self.gen_topology_sparsity_helpers_python()
    offset_dv_dq = 0
    offset_dv_dqd = offset_dv_dq + 6*dva_cols_per_partial
    offset_da_dq = offset_dv_dqd + 6*dva_cols_per_partial
    offset_da_dqd = offset_da_dq + 6*dva_cols_per_partial
    offset_df_dq = offset_da_dqd + 6*dva_cols_per_partial
    offset_df_dqd = offset_df_dq + 6*df_cols_per_partial
    offset_fxvi = offset_df_dqd + 6*df_cols_per_partial
    offset_mxxv = offset_fxvi + 36*n
    offset_mxxa = offset_mxxv + 6*n
    offset_mxv = offset_mxxa + 6*n
    offset_mxf = offset_mxv + 6*n
    offset_iv = offset_mxf + 6*n
    full_count = offset_iv + 6*n
    spill_start = offset_da_dq
    spill_end = offset_fxvi
    spill_count = spill_end - spill_start
    return {
        "dva_cols_per_partial": dva_cols_per_partial,
        "df_cols_per_partial": df_cols_per_partial,
        "offset_dv_dq": offset_dv_dq,
        "offset_dv_dqd": offset_dv_dqd,
        "offset_da_dq": offset_da_dq,
        "offset_da_dqd": offset_da_dqd,
        "offset_df_dq": offset_df_dq,
        "offset_df_dqd": offset_df_dqd,
        "offset_fxvi": offset_fxvi,
        "offset_mxxv": offset_mxxv,
        "offset_mxxa": offset_mxxa,
        "offset_mxv": offset_mxv,
        "offset_mxf": offset_mxf,
        "offset_iv": offset_iv,
        "full_count": full_count,
        "spill_start": spill_start,
        "spill_end": spill_end,
        "spill_count": spill_count,
        "selective_shared_count": full_count - spill_count,
    }

def _rewrite_inverse_dynamics_gradient_temp_accesses_for_spill(code):
    def replace_accesses(text, address_of):
        token = "&s_temp[" if address_of else "s_temp["
        out = []
        i = 0
        while i < len(text):
            start = text.find(token, i)
            if start < 0:
                out.append(text[i:])
                break
            out.append(text[i:start])
            idx_start = start + len(token)
            depth = 1
            j = idx_start
            while j < len(text) and depth > 0:
                if text[j] == "[":
                    depth += 1
                elif text[j] == "]":
                    depth -= 1
                j += 1
            idx_expr = text[idx_start:j-1]
            ptr_expr = "grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, " + idx_expr + ")"
            out.append(ptr_expr if address_of else "(*" + ptr_expr + ")")
            i = j
        return "".join(out)

    return replace_accesses(replace_accesses(code, True), False)

def gen_inverse_dynamics_gradient_inner_function_call(self, updated_var_names = None):
    var_names = dict( \
        s_dc_du_name = "s_dc_du", \
        s_vaf_name = "s_vaf", \
        s_q_name = "s_q", \
        s_qd_name = "s_qd", \
        s_qdd_name = "s_qdd", \
        s_temp_name = "s_temp", \
        d_temp_spill_name = "nullptr", \
        temp_spill_flag_name = "false", \
        gravity_name = "gravity", \
        d_robotModel_name = "d_robotModel"
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    inverse_dynamics_gradient_code_start = "inverse_dynamics_gradient_inner<T, " + var_names["temp_spill_flag_name"] + ">(" + var_names["s_dc_du_name"] + ", " + var_names["s_q_name"] + ", " + var_names["s_qd_name"] + ", "
    inverse_dynamics_gradient_code_middle = var_names["s_vaf_name"] + ", " + self.gen_insert_helpers_function_call()
    # runtime_joint_dynamics: forward d_robotModel into the inner (trailing defaulted
    # param) so the damping-diag term can read the mutable table; omitted when off.
    _idg_rt_jd = (", " + var_names["d_robotModel_name"]) if getattr(self, "runtime_joint_dynamics", False) else ""
    inverse_dynamics_gradient_code_end = var_names["s_temp_name"] + ", " + var_names["d_temp_spill_name"] + ", " + var_names["gravity_name"] + _idg_rt_jd + ");"
    inverse_dynamics_gradient_code = inverse_dynamics_gradient_code_start + inverse_dynamics_gradient_code_middle + inverse_dynamics_gradient_code_end
    self.gen_add_code_line(inverse_dynamics_gradient_code)

def gen_inverse_dynamics_gradient_inner(self):
    function_start = len(self.code_str)
    n = self.robot.get_num_vel()
    NJ = self.robot.get_num_joints()
    max_bfs_levels = self.robot.get_max_bfs_level()
    n_bfs_levels = max_bfs_levels + 1 # starts at 0

    # construct the boilerplate and function definition
    func_params = ["s_dc_du is a pointer to memory for the final result of size 2*NUM_VEL*NUM_VEL = " + str(2*n*n), \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "s_vaf are the helper intermediate variables computed by inverse_dynamics", \
                   "s_temp is a pointer to helper shared memory of size 66*NUM_JOINTS + 6*sparse_dv,da,df_col_needs = " + \
                            str(self.gen_inverse_dynamics_gradient_inner_temp_mem_size()), \
                   "gravity is the gravity constant"]
    func_def_start = "void inverse_dynamics_gradient_inner(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_vaf, "
    # runtime_joint_dynamics: the damping-diag term reads
    # d_robotModel->d_joint_dynamics_params, so thread d_robotModel in as a trailing
    # defaulted param ONLY under that flag (byte-identical signature when off).
    if getattr(self, "runtime_joint_dynamics", False):
        func_def_end = "T *s_temp, T *d_temp_spill, const T gravity, const robotModel<T> *d_robotModel = nullptr) {"
    else:
        func_def_end = "T *s_temp, T *d_temp_spill, const T gravity) {"
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -2)
    func_notes = ["Assumes s_XImats is updated already for the current s_q",
                  "This is the inverse_dynamics_gradient band sub-inner (the stable surface composed by forward_dynamics_gradient / integrator_gradient). It does NOT own s_temp placement; the USE_DA_DF_SPILL band selectively spills its da_dq..fxvi band to d_temp_spill via grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>. The whole-pool placement is owned by the wrapping inverse_dynamics_gradient_device."]
    func_def = func_def_start + func_def_end
    # then generate the code
    self.gen_add_func_doc("Computes the gradient of inverse dynamics",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T, bool USE_DA_DF_SPILL = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    if self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis() or self.robot.robot_has_spherical():
        # MIMIC (T3-finisher P3): the sparse NJ-indexed gradient assembly below
        # writes s_dc_du by raw body id and assumes NJ == NV, so it can't fold a
        # mimic model (NB > NV, shared v-slots). Emit instead a DENSE serial
        # reduced-space fold that mirrors RBDReference.rnea_grad exactly:
        #
        # SPHERICAL (Tier C): the sparse single-DoF band ALSO cannot represent a
        # mid-chain multi-column (3-DoF) joint (NJ != NV, a body owning a 3-wide
        # v-block), so the dense serial inner carries the spherical case too. A
        # spherical body's columns are CARDINAL (angular-identity S), so it reuses
        # the mx0/mx1/mx2 cardinal helpers per column (col k -> angular row k).
        #
        # SKEW (Tier B): the same dense serial inner ALSO carries the general
        # motion-subspace (skew) case — the scalar (s_ind, s_sign) sites below
        # switch to a dense 6-vector S (mxS_general / S^T dot) per body when the
        # body is non-cardinal. Cardinal+non-mimic robots never enter here, so
        # their byte-identical sparse path is untouched.
        # alpha-scaled velocity/accel reads in the forward pass + per-body
        # v-slot accumulate (+=) in the backward pass. Correctness, not perf, is
        # the goal here (mimic robots are the gripper/hand class). The large
        # dense per-body buffers spill to d_temp_spill / d_workspace via the
        # inner's mimic temp-size; see gen_inverse_dynamics_gradient_inner_temp_mem_size.
        _gen_inverse_dynamics_gradient_mimic_inner(self, n, NJ)
        # gated joint-damping gradient (dc_dqd diagonal += damping); no-op when
        # USE_JOINT_DYNAMICS is off. Operates on the final s_dc_du, so it is
        # orthogonal to the mimic/spherical/skew extraction above.
        _idg_damping_diag_cpp(self, n)
        self.gen_add_end_function()
        return

    #
    # Optimize memory requirements due to sparsity induced by branching
    # requires more complex pointer math but/and saves a lot of space
    #
    (dva_cols_per_partial, dva_cols_per_jid, running_sum_dva_cols_per_jid, \
      df_cols_per_partial, df_cols_per_jid, running_sum_df_cols_per_jid, df_col_that_is_jid) = self.gen_topology_sparsity_helpers_python()
    if not self.robot.floating_base:
        self.gen_add_code_line("//")
        self.gen_add_code_line("// dv and da need " + str(dva_cols_per_partial) + " cols per dq,dqd")
        self.gen_add_code_line("// df needs " + str(df_cols_per_partial) + " cols per dq,dqd")
        self.gen_add_code_line("//    out of a possible " + str(n*n) + " cols per dq,dqd")
        self.gen_add_code_line("// Gradients are stored compactly as dv_i/dq_[0...a], dv_i+1/dq_[0...b], etc")
        self.gen_add_code_line("//    where a and b are the needed number of columns")
        self.gen_add_code_line("//")
    # gen som aditional helpers
    running_sum_delta_df_dva_cols_per_jid = [running_sum_df_cols_per_jid[jid] - running_sum_dva_cols_per_jid[jid] for jid in range(NJ)]

    # add shared memory note
    Offset_dv_dq = 0
    Offset_dv_dqd = Offset_dv_dq + 6*dva_cols_per_partial
    Offset_da_dq = Offset_dv_dqd + 6*dva_cols_per_partial
    Offset_da_dqd = Offset_da_dq + 6*dva_cols_per_partial
    Offset_df_dq = Offset_da_dqd + 6*dva_cols_per_partial
    Offset_df_dqd = Offset_df_dq + 6*df_cols_per_partial
    Offset_FxvI = Offset_df_dqd + 6*df_cols_per_partial
    Offset_MxXv = Offset_FxvI + 36*n
    Offset_MxXa = Offset_MxXv + 6*n
    Offset_Mxv = Offset_MxXa + 6*n
    Offset_Mxf = Offset_Mxv + 6*n
    Offset_Iv = Offset_Mxf + 6*n
    # Offset_dva_cols = Offset_Iv + 6*n
    # Offset_df_cols = Offset_dva_cols + n

    self.gen_add_code_line("// Temp memory offsets are as follows:")
    self.gen_add_code_line("// T *s_dv_dq = &s_temp[" + str(Offset_dv_dq) + "]; " + \
                              "T *s_dv_dqd = &s_temp[" + str(Offset_dv_dqd) + "]; " + \
                              "T *s_da_dq = &s_temp[" + str(Offset_da_dq) + "];")
    self.gen_add_code_line("// T *s_da_dqd = &s_temp[" + str(Offset_da_dqd) + "]; " + \
                              "T *s_df_dq = &s_temp[" + str(Offset_df_dq) + "]; " + \
                              "T *s_df_dqd = &s_temp[" + str(Offset_df_dqd) + "];")
    self.gen_add_code_line("// T *s_FxvI = &s_temp[" + str(Offset_FxvI) + "]; T *s_MxXv = &s_temp[" + str(Offset_MxXv) + "]; " + \
                              "T *s_MxXa = &s_temp[" + str(Offset_MxXa) + "];")
    self.gen_add_code_line("// T *s_Mxv = &s_temp[" + str(Offset_Mxv) + "]; T *s_Mxf = &s_temp[" + str(Offset_Mxf) + "]; " + \
                              "T *s_Iv = &s_temp[" + str(Offset_Iv) + "];")

    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"Validating Function Inputs\\n\");", \
                                 "printf(\"-------------------------\\n\");", \
                                 "printf(\"q\\n\"); printMat<T,1," + str(n) + ">(s_q,1);", \
                                 "printf(\"qd\\n\"); printMat<T,1," + str(n) + ">(s_qd,1);", \
                                 "printf(\"vaf-v\\n\"); printMat<T,6," + str(n) + ">(s_vaf,6);", \
                                 "printf(\"vaf-a\\n\"); printMat<T,6," + str(n) + ">(&s_vaf[6*" + str(n) + "],6);", \
                                 "printf(\"vaf-f\\n\"); printMat<T,6," + str(n) + ">(&s_vaf[12*" + str(n) + "],6);"])
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"X[%d]\\n\",i); printMat<T,6,6>(&s_XImats[36*i],6);}")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"I[%d]\\n\",i); printMat<T,6,6>(&s_XImats[36*(i+" + str(n) + ")],6);}")
        self.gen_add_code_line("printf(\"-------------------------\\n\");")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
    
    #
    # Initial temp comps
    #
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Initial Temp Comps")
    self.gen_add_code_line("//")
    # first compute temporary values by type of operation
    # we can use part of FxvI temp mem for Xv and Xa initial comps also compute Iv
    self.gen_add_code_line("// First compute Imat*v and Xmat*v_parent, Xmat*a_parent (store in FxvI for now)")
    self.gen_add_code_line("// Note that if jid_parent == -1 then v_parent = 0 and a_parent = gravity")
    self.gen_add_parallel_loop("ind",str(6*3*NJ))
    self.gen_add_code_line("int row = ind % 6; int col = ind / 6; int jid = col % " + str(NJ) + "; int jid6 = 6*jid;")
    # get the parent (note that in some cases we have more efficient ways of computing this so add some special cases)
    parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(NO_GRAD_FLAG = True, OFFSET=False)
    S_sign_cpp = self.gen_topology_S_sign_for_cpp(OFFSET=False)
    self.gen_add_code_line("bool parentIsBase = " + parent_ind_cpp + " == -1;")
    # then get the offsets
    self.gen_add_code_lines(["bool comp1 = col < " + str(NJ) + "; bool comp3 = col >= " + str(2*NJ) + ";",
                             "int XIOffset  =  comp1 * " + str(36*NJ) + " + 6*jid6 + row; // rowCol of I (comp1) or X (comp 2 and 3)",
                             "int vaOffset  = comp1 * jid6 + !comp1 * 6*" + parent_ind_cpp + " + comp3 * " + str(6*NJ) + "; // v_i (comp1) or va_parent (comp 2 and 3)",
                             "int dstOffset = comp1 * " + str(Offset_Iv) + " + !comp1 * " + str(Offset_FxvI) + " + comp3 * " + str(6*NJ) + " + jid6 + row; // rowCol of dst"])
    if self.robot.floating_base:
        self.gen_add_code_lines(["s_temp[dstOffset] = (parentIsBase && !comp1) ?",
                                 "                           (comp3 ? (row < 3 ? static_cast<T>(0) : -s_XImats[6*jid6 + 6*row + 5] * gravity) : static_cast<T>(0)) :",
                                 "                           dot_prod<T,6,6,1>(&s_XImats[XIOffset],&s_vaf[vaOffset]);"])
    else:
        self.gen_add_code_lines(["s_temp[dstOffset] = (parentIsBase && !comp1) ? comp3 * -s_XImats[XIOffset + 30] * gravity :",
                                 "                                               dot_prod<T,6,6,1>(&s_XImats[XIOffset],&s_vaf[vaOffset]);"])
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"Temp Comps Part 1\\n\");", \
                                 "printf(\"-------------------------\\n\");", \
                                 "printf(\"Iv\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_Iv) + "],6);", \
                                 "printf(\"Xv\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_FxvI) + "],6);", \
                                 "printf(\"Xa\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_FxvI + 6*n) + "],6);"])
        self.gen_add_code_line("printf(\"-------------------------\\n\");")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    # then do the mx comps
    self.gen_add_code_line("// Then compute Mx(Xv), Mx(Xa), Mx(v), Mx(f)")
    self.gen_add_parallel_loop("col",str(4*n))
    self.gen_add_code_line("int dof_id = col / 4; int selector = col % 4; int dof_id6 = 6*dof_id;")
    if self.robot.floating_base: self.gen_add_code_line("int jid = dof_id < 6 ? 0 : dof_id - 5; int jid6 = jid*6;") # First 6 dof belong to fb
    else: self.gen_add_code_line("int jid6 = dof_id6;")
    select_var_vals = [("int", "dstOffset", [str(Offset_MxXv), str(Offset_MxXa), str(Offset_Mxv), str(Offset_Mxf)]), \
                       ("const T *", "src", ["&s_temp[" + str(Offset_FxvI) + "]", "&s_temp[" + str(Offset_FxvI + 6*NJ) + "]", \
                                       "&s_vaf[0]", "&s_vaf[" + str(12*NJ) + "]"])]
    self.gen_add_multi_threaded_select("selector", "==", [str(i) for i in range(4)], select_var_vals)
    if 'jid' in S_ind_cpp: S_ind_cpp = S_ind_cpp.replace('jid', 'dof_id')
    if 'jid' in S_sign_cpp: S_sign_cpp = S_sign_cpp.replace('jid', 'dof_id')
    updated_var_names = dict(S_ind_name = S_ind_cpp, s_dst_name = "&s_temp[dstOffset + dof_id6]", s_src_name = "&src[jid6]", s_scale_name = S_sign_cpp)
    self.gen_mx_func_call_for_cpp(PEQ_FLAG = False, SCALE_FLAG = True, updated_var_names = updated_var_names)
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    has_linear_axis = any(
        self.robot.get_S_index_by_id(jid) >= 3 for jid in range(n)
    )
    if has_linear_axis:
        # For prismatic axes, the force derivative term needs the force cross-product
        # column. The motion and force columns coincide for the revolute axes covered
        # by the original path, but differ for linear axes.
        self.gen_add_parallel_loop("dof_id", str(n))
        _, S_ind_dof_cpp = self.gen_topology_helpers_pointers_for_cpp(
            list(range(n)),
            updated_var_names=dict(jid_name="dof_id"),
            NO_GRAD_FLAG=True,
            OFFSET=False,
        )
        S_sign_dof_cpp = self.gen_topology_S_sign_for_cpp(
            list(range(n)),
            updated_var_names=dict(jid_name="dof_id"),
            OFFSET=False,
        )
        self.gen_add_code_line("int S_ind = " + S_ind_dof_cpp + ";")
        self.gen_add_code_line("T S_sign = static_cast<T>(" + S_sign_dof_cpp + ");")
        self.gen_add_code_line("if (S_ind >= 3) {", True)
        self.gen_add_code_line(f"T *dst = &s_temp[{Offset_Mxf} + 6*dof_id];")
        if self.robot.floating_base:
            self.gen_add_code_line("int jid = dof_id < 6 ? 0 : dof_id - 5;")
            self.gen_add_code_line(f"const T *src = &s_vaf[{12*NJ} + 6*jid];")
        else:
            self.gen_add_code_line(f"const T *src = &s_vaf[{12*NJ} + 6*dof_id];")
        self.gen_add_code_line("for (int row = 0; row < 6; ++row) dst[row] = static_cast<T>(0);")
        self.gen_add_code_line("if (S_ind == 3) {", True)
        self.gen_add_code_line("dst[1] = S_sign * src[5];")
        self.gen_add_code_line("dst[2] = -S_sign * src[4];")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("else if (S_ind == 4) {", True)
        self.gen_add_code_line("dst[0] = -S_sign * src[5];")
        self.gen_add_code_line("dst[2] = S_sign * src[3];")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("else {", True)
        self.gen_add_code_line("dst[0] = S_sign * src[4];")
        self.gen_add_code_line("dst[1] = -S_sign * src[3];")
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"Temp Comps Part 2\\n\");", \
                                 "printf(\"-------------------------\\n\");", \
                                 "printf(\"Mx(Xv)\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_MxXv) + "],6);", \
                                 "printf(\"Mx(Xa)\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_MxXa) + "],6);", \
                                 "printf(\"Mx(v)\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_Mxv) + "],6);",\
                                 "printf(\"Mx(f)\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_Mxf) + "],6);"])
        self.gen_add_code_line("printf(\"-------------------------\\n\");")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    #
    # FORWARD PASS
    #
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Forward Pass")
    self.gen_add_code_line("//")
    self.gen_add_code_line("// We start with dv/du noting that we only have values")
    self.gen_add_code_line("//    for ancestors and for the current index else 0")
    # then serial dv/du in bfs waves
    for bfs_level in range(n_bfs_levels):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        _, S_ind_cpp, dva_col_offset_for_jid_cpp, _, dva_col_offset_for_parent_cpp, _, _, _ = self.gen_topology_helpers_pointers_for_cpp(inds, OFFSET=False)
        S_sign_cpp = self.gen_topology_S_sign_for_cpp(inds, OFFSET=False)
        self.gen_add_code_line("// dv/du where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))

        # when parent is base dv_dq = 0, dv_dqd = S
        if bfs_level == 0:
            self.gen_add_code_line("// when parent is base dv_dq = 0, dv_dqd = S")
            if self.robot.floating_base:
                self.gen_add_parallel_loop("ind",str(6*2*n))
                self.gen_add_code_line("bool dq_flag = ind < " + str(6*n) + ";")
                self.gen_add_code_line("int row = ind % 6; int col = (!dq_flag * " + str(-n) + ") + (ind / 6);")
                self.gen_add_code_line("int du_offset = dq_flag ? " + str(Offset_dv_dq) + " : " + str(Offset_dv_dqd) + ";")
                self.gen_add_code_line("int fb_col = row < 3 ? row + 3 : row - 3;")
                self.gen_add_code_line("s_temp[du_offset + 6*col + row] = !dq_flag * (fb_col == col);")
            else:
                self.gen_add_parallel_loop("ind",str(6*2*len(inds)))
                if len(inds) > 1:
                    self.gen_add_code_line("int row = ind % 6; int col = ind / 6; int col_du = col % " + str(len(inds)) + "; bool dq_flag = col == col_du;")
                    select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                    self.gen_add_multi_threaded_select("col_du", "<", [str((i+1)) for i in range(len(inds))], select_var_vals)
                else:
                    self.gen_add_code_line("int row = ind % 6; int dq_flag = (ind / 6) == 0;")
                self.gen_add_code_line("int du_offset = dq_flag ? " + str(Offset_dv_dq) + " : " + str(Offset_dv_dqd) + ";")
                self.gen_add_code_line("s_temp[du_offset + 6*" + dva_col_offset_for_jid_cpp + " + row] = " + \
                                    "(!dq_flag && row == " + S_ind_cpp + ") * static_cast<T>(" + S_sign_cpp + ");") 
            self.gen_add_end_control_flow()

        # dv/du = X dv_parent/du + {MxXv or S for col ind}
        # there are 2*(bfs_level + 1) columns per du with 2*bfs mults with X and then the addition in the last col
        else:
            self.gen_add_code_line("// dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}")
            self.gen_add_code_line("// first compute dv/du = Xmat*dv_parent/du")
            if self.robot.floating_base:
                jid, parent_ind_cpp = self._emit_fb_bfs_level_indexing(
                    inds, n, dq_flag_line = f"bool dq_flag = ind < {6*n*len(inds)};")
                self.gen_add_code_line(f"int dof_id = {jid}+5; (void)dof_id;")
                self.gen_add_code_line(f"int du_offset = dq_flag ? {Offset_dv_dq} + {6*n}*{jid} : {Offset_dv_dqd} + {6*n}*{jid};")
                self.gen_add_code_line(f"int parent_du_offset = dq_flag ? {Offset_dv_dq} + {6*n}*{parent_ind_cpp} : {Offset_dv_dqd} + {6*n}*{parent_ind_cpp};")
                self.gen_add_code_line("s_temp[du_offset + 6*col + row] = dot_prod<T,6,6,1>(&s_XImats[36 * " + str(jid) + " + row]," + \
                                        " &s_temp[6*col + parent_du_offset]);")
                # then add in S or Mx(Xv); dof_id is only referenced when the S_ind/S_sign cpp expressions actually substitute it.
                self.gen_add_code_line(f"if (col == {jid} + 5)" + ' {', True)
                if 'jid' in S_ind_cpp: S_ind_cpp = S_ind_cpp.replace('jid', 'dof_id')
                if 'jid' in S_sign_cpp: S_sign_cpp = S_sign_cpp.replace('jid', 'dof_id')
                self.gen_add_code_line(f"s_temp[du_offset + 6*col + row] += !dq_flag * (row == {S_ind_cpp})" + \
                                        f" * ({S_sign_cpp}) + dq_flag * s_temp[{Offset_MxXv} + ({jid}+5)*6 + row];")
                self.gen_add_end_control_flow()
            else:
                self.gen_add_parallel_loop("ind",str(6*2*(bfs_level)*len(inds)))
                self.gen_add_code_line("int row = ind % 6; int col = ind / 6; int col_du = col % " + str(bfs_level*len(inds)) + "; " + \
                                                                            "int col_jid = col_du % " + str(bfs_level) + ";")
                if bfs_level > 1 or len(inds) > 1:
                    self.gen_add_code_line("int dq_flag = col == col_du;")
                else:
                    self.gen_add_code_line("int dq_flag = col < 1;")
                if len(inds) > 1:
                    select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                    jid = "jid"
                    self.gen_add_multi_threaded_select("col_du", "<", [str((i+1)*bfs_level) for i in range(len(inds))], select_var_vals)
                else:
                    jid = str(inds[0])
                self.gen_add_code_line("int du_col_offset = dq_flag * " + str(Offset_dv_dq) + " + !dq_flag * " + str(Offset_dv_dqd) + " + 6 * col_jid;")
                self.gen_add_code_line("s_temp[du_col_offset + 6*" + dva_col_offset_for_jid_cpp + " + row] = ")
                self.gen_add_code_line("    dot_prod<T,6,6,1>(&s_XImats[36*" + jid + " + row]," + \
                                                            "&s_temp[du_col_offset + 6*" + dva_col_offset_for_parent_cpp + "]);")
                self.gen_add_code_line("// then add {Mx(Xv) or S for col ind}")
                # all cols add if bfs_level is 1 so skip the if statement
                if bfs_level > 1:
                    self.gen_add_code_line("if (col_jid == " + str(bfs_level-1) + ") {", True)
                # do the non-branching if/else
                self.gen_add_code_line("s_temp[du_col_offset + 6*" + dva_col_offset_for_jid_cpp + " + 6 + row] = ")
                self.gen_add_code_line("    dq_flag * s_temp[" + str(Offset_MxXv) + " + 6*" + jid + " + row] + " + \
                                        "(!dq_flag && row == " + S_ind_cpp + ") * static_cast<T>(" + S_sign_cpp + ");")
                # all cols add if bfs_level is 1 so skip the if statement
                if bfs_level > 1:
                    self.gen_add_end_control_flow()
            self.gen_add_end_control_flow()
        self.gen_add_sync()

        if self.DEBUG_MODE:
            self.gen_add_sync()
            self.gen_add_serial_ops()
            if bfs_level == 0:
                self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                         "printf(\"dv/du in bfs waves\\n\");", \
                                         "printf(\"-------------------------\\n\");"])
            self.gen_add_code_line("printf(\"dv/du in for bfs wave[%d]\\n\"," + str(bfs_level) + ");")
            for ind in inds:
                self.gen_add_code_lines(["printf(\"dv[%d]/dq\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(bfs_level+1) + ">(&s_temp[" + \
                                                str(Offset_dv_dq + 6*running_sum_dva_cols_per_jid[ind]) + "],6);", \
                                         "printf(\"dv[%d]/dqd\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(bfs_level+1) + ">(&s_temp[" + \
                                                str(Offset_dv_dqd + 6*running_sum_dva_cols_per_jid[ind]) + "],6);"])
            self.gen_add_end_control_flow()
            self.gen_add_sync()

    # Start the da/du comp with da/du = MxS(dv/du)*qd + {MxXa, Mxv}
    self.gen_add_code_line("// start da/du by setting = MxS(dv/du)*qd + {MxXa, Mxv} for all n in parallel")
    self.gen_add_code_line("// start with da/du = MxS(dv/du)*qd")
    _ , S_ind_cpp , _ , _ , _ , _ , dva_col_offset_for_jidp1_cpp, _ = self.gen_topology_helpers_pointers_for_cpp(list(range(n)), OFFSET=False)
    S_sign_cpp = self.gen_topology_S_sign_for_cpp(OFFSET=False)
    add_col_for_jid = "(" + dva_col_offset_for_jidp1_cpp + " - 1)"
    if self.robot.floating_base:
        # set da/du = 0
        self.gen_add_code_line("// First zero da/du")
        self.gen_add_parallel_loop('ind',str(2*n*NJ*6))
        self.gen_add_code_line(f"s_temp[{Offset_da_dq} + ind] = static_cast<T>(0);")
        self.gen_add_end_control_flow()
        # Sync before the += accumulation below: the zeroing loop and the
        # MxS(dv/du)*qd accumulation write the same s_temp[Offset_da_dq] region
        # from different threads. Without this barrier the result is correct
        # only within a single warp (<=32 threads) and races at larger blocks.
        self.gen_add_sync()
        if 'jid' in S_ind_cpp: S_ind_cpp = S_ind_cpp.replace('jid', 'dof_id')
        if 'jid' in S_sign_cpp: S_sign_cpp = S_sign_cpp.replace('jid', 'dof_id')
        # Axis-indexed S helpers for the serialized root accumulation below.
        S_ind_ax = S_ind_cpp.replace('dof_id', 'ax')
        S_sign_ax = S_sign_cpp.replace('dof_id', 'ax')
        self.gen_add_parallel_loop("col",str(2*n*n))
        self.gen_add_code_line(f"int dof = col % {n};") # column within each joint that is being focused
        self.gen_add_code_line(f"int dof_id = (col / {n}) % {n}; int jid = dof_id < 6 ? 0 : dof_id - 5;") # dof_id being applied with S, to jid
        self.gen_add_code_line(f"bool dq_flag = col < {n*n}; int dqd_offset = !dq_flag * {6*dva_cols_per_partial};")
        # Floating root (jid==0): all 6 root axes (dof_id 0..5) accumulate into
        # the SAME da/du column, but mxX_peq_scaled assumes a single writer per
        # destination. Run the whole accumulation for the root on one lane
        # (dof_id==0), summing over the 6 axes, so there is no multi-thread +=
        # race. (Correct only within one warp otherwise -> wrong J_qv at
        # MAX_PERF_LEVEL_THREADS.) Non-root joints have a unique axis per lane.
        self.gen_add_code_line("if (jid == 0 && dof_id == 0) {", True)
        self.gen_add_code_line(f"T *root_dst = &s_temp[{Offset_da_dq} + dof*6 + dqd_offset];")
        self.gen_add_code_line(f"const T *root_src = &s_temp[{Offset_dv_dq} + dof*6 + dqd_offset];")
        self.gen_add_code_line("for (int ax = 0; ax < 6; ax++) {", True)
        self.gen_mx_func_call_for_cpp(PEQ_FLAG = True, SCALE_FLAG = True, updated_var_names = dict(
            S_ind_name = S_ind_ax, s_dst_name = "root_dst", s_src_name = "root_src",
            s_scale_name = "(" + S_sign_ax + ") * s_qd[ax]"))
        self.gen_add_end_control_flow()  # for ax
        # The {MxXa, Mxv} add applies to root columns dof in [0,6) (axis == column).
        self.gen_add_code_line("if (dof < 6) {", True)
        self.gen_add_code_line(f"int src_offset = dq_flag * {Offset_MxXa} + !dq_flag * {Offset_Mxv} + 6*dof;")
        self.gen_add_code_line("for (int row = 0; row < 6; row++) { root_dst[row] += s_temp[src_offset + row]; }")
        self.gen_add_end_control_flow()  # if dof < 6
        self.gen_add_end_control_flow()  # if jid == 0 && dof_id == 0
        # Non-root joints: one lane per (jid, dof), no collision.
        self.gen_add_code_line("if (jid != 0) {", True)
        updated_var_names = dict(S_ind_name = S_ind_cpp, s_dst_name = f"&s_temp[{Offset_da_dq} + jid*{6*n} + dof*6 + dqd_offset]", \
                                 s_src_name = f"&s_temp[{Offset_dv_dq} + jid*{6*n} + dof*6 + dqd_offset]", s_scale_name = "(" + S_sign_cpp + ") * s_qd[dof_id]")
        self.gen_mx_func_call_for_cpp(PEQ_FLAG = True, SCALE_FLAG = True, updated_var_names = updated_var_names)
        self.gen_add_code_line("// then add {MxXa, Mxv} to the appropriate column")
        self.gen_add_code_line("if (dof == dof_id) {", True)
        self.gen_add_code_line(f"int src_offset = dq_flag * {Offset_MxXa} + !dq_flag * {Offset_Mxv} + 6*dof_id;")
        self.gen_add_code_line(f"for (int row = 0; row < 6; row++) {{ s_temp[{Offset_da_dq} + 6*dof + row + jid*{6*n} + dqd_offset] += s_temp[src_offset + row]; }}")
        self.gen_add_end_control_flow()  # if dof == dof_id
        self.gen_add_end_control_flow()  # if jid != 0
        self.gen_add_end_control_flow()  # parallel loop
        self.gen_add_sync()
    else:
        self.gen_add_parallel_loop("col",str(2*dva_cols_per_partial))
        self.gen_add_code_line("int col_du = col % " + str(dva_cols_per_partial) + ";") # signifies col of corresponding du
        select_var_vals = [("int", "jid", [str(jid) for jid in range(NJ)])]
        self.gen_add_multi_threaded_select("col_du", "<", [str(running_sum_dva_cols_per_jid[jid+1]) for jid in range(NJ)], select_var_vals)
        updated_var_names = dict(S_ind_name = S_ind_cpp, s_dst_name = "&s_temp[" + str(Offset_da_dq) + " + 6*col]", \
                                s_src_name = "&s_temp[" + str(Offset_dv_dq) + " + 6*col]", s_scale_name = "(" + S_sign_cpp + ") * s_qd[jid]")
        # call the mx func
        self.gen_mx_func_call_for_cpp(PEQ_FLAG = False, SCALE_FLAG = True, updated_var_names = updated_var_names)
        # then add to the add col
        self.gen_add_code_lines(["// then add {MxXa, Mxv} to the appropriate column", \
                                "int dq_flag = col == col_du; int src_offset = dq_flag * " + str(Offset_MxXa) + " + !dq_flag * " + str(Offset_Mxv) + " + 6*jid;"])
        self.gen_add_code_line("if(col_du == " + add_col_for_jid + "){", True)
        self.gen_add_code_line("for(int row = 0; row < 6; row++){", True)
        self.gen_add_code_line("s_temp[" + str(Offset_da_dq) + " + 6*col + row] += s_temp[src_offset + row];")
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"da/du part 1 = MxS(dv/du)*qd + {MxXa, Mxf}\\n\");", \
                                 "printf(\"-------------------------\\n\");"])
        for ind in range(NJ):
            off_dq, num_cols = _idg_debug_body_buffer(self, ind, Offset_da_dq, n, running_sum_dva_cols_per_jid, dva_cols_per_jid)
            off_dqd, _ = _idg_debug_body_buffer(self, ind, Offset_da_dqd, n, running_sum_dva_cols_per_jid, dva_cols_per_jid)
            self.gen_add_code_lines(["printf(\"da[%d]/dq\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + \
                                            str(off_dq) + "],6);", \
                                     "printf(\"da[%d]/dqd\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + \
                                            str(off_dqd) + "],6);"])
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    # then serial da/du in bfs waves
    self.gen_add_code_line("// Finish da/du with parent updates noting that we only have values")
    self.gen_add_code_line("//    for ancestors and for the current index and nothing for bfs 0")
    for bfs_level in range(1,n_bfs_levels):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        _, _, dva_col_offset_for_jid_cpp, _, dva_col_offset_for_parent_cpp, _, _, _ = self.gen_topology_helpers_pointers_for_cpp(inds, OFFSET=False)
        parent_inds = [self.robot.get_parent_id(ind) for ind in inds]
        self.gen_add_code_line("// da/du where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))
        
        # da/du += X da_parent/du
        # there are 2*(bfs_level + 1) columns per du with 2*bfs mults with X and then the addition in the last col
        self.gen_add_code_line("// da/du += Xmat*da_parent/du")    
        if self.robot.floating_base: 
            jid, parent_ind_cpp = self._emit_fb_bfs_level_indexing(
                inds, n, dq_flag_line = f"bool dq_flag = ind < {6*n*len(inds)};")
            self.gen_add_code_line(f"int du_offset = dq_flag ? {Offset_da_dq} + {6*n}*{jid} : {Offset_da_dqd} + {6*n}*{jid};")
            self.gen_add_code_line(f"int parent_du_offset = dq_flag ? {Offset_da_dq} + {6*n}*{parent_ind_cpp} : {Offset_da_dqd} + {6*n}*{parent_ind_cpp};")
            self.gen_add_code_line("s_temp[du_offset + 6*col + row] += dot_prod<T,6,6,1>(&s_XImats[36 * " + str(jid) + " + row]," + \
                                        " &s_temp[6*col + parent_du_offset]);")
        else:
            self.gen_add_parallel_loop("ind",str(6*2*bfs_level*len(inds)))
            self.gen_add_code_lines(["int row = ind % 6; int col = ind / 6; int col_du = col % " + str(bfs_level*len(inds)) + ";", \
                                    "int dq_flag = col == col_du; int col_jid = col_du % " + str(bfs_level) + ";"])
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                jid = "jid"
                self.gen_add_multi_threaded_select("col_du", "<", [str((i+1)*bfs_level) for i in range(len(inds))], select_var_vals)
            else:
                jid = str(inds[0])
            self.gen_add_code_line("int du_col_offset = dq_flag * " + str(Offset_da_dq) + " + !dq_flag * " + str(Offset_da_dqd) + " + 6 * col_jid;")
            self.gen_add_code_lines(["s_temp[du_col_offset + 6*" + dva_col_offset_for_jid_cpp + " + row] += ", \
                                    "    dot_prod<T,6,6,1>(&s_XImats[36*" + jid + " + row]," + \
                                                            "&s_temp[du_col_offset + 6*" + dva_col_offset_for_parent_cpp + "]);"])
        self.gen_add_end_control_flow()
        self.gen_add_sync()

        if self.DEBUG_MODE:
            self.gen_add_sync()
            self.gen_add_serial_ops()
            if bfs_level == 1:
                self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                         "printf(\"da/du in bfs waves\\n\");", \
                                         "printf(\"-------------------------\\n\");"])
            self.gen_add_code_line("printf(\"da/du for bfs wave[%d]\\n\"," + str(bfs_level) + ");")
            for ind in inds:
                self.gen_add_code_lines(["printf(\"da[%d]/dq\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(bfs_level+1) + ">(&s_temp[" + \
                                                str(Offset_da_dq + 6*running_sum_dva_cols_per_jid[ind]) + "],6);", \
                                         "printf(\"da[%d]/dqd\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(bfs_level+1) + ">(&s_temp[" + \
                                                str(Offset_da_dqd + 6*running_sum_dva_cols_per_jid[ind]) + "],6);"])
            self.gen_add_end_control_flow()
            self.gen_add_sync()

    
    # Intiialize df/du to 0 to make sure we don't have issues with remaining values later when we do +=
    self.gen_add_code_line("// Init df/du to 0")
    self.gen_add_parallel_loop("ind",str(6*2*df_cols_per_partial))
    self.gen_add_code_line("s_temp[" + str(Offset_df_dq) + " + ind] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Start the df/du by setting = fx(dv/du)*Iv and also compute the temp = Fx(v)*I 
    # aka do all of the Fx comps in parallel
    self.gen_add_code_lines(["// Start the df/du by setting = fx(dv/du)*Iv and also compute the temp = Fx(v)*I ", \
                             "//    aka do all of the Fx comps in parallel", \
                             "// note that while df has more cols than dva the dva cols are the first few df cols"])
    _, _, dva_col_offset_for_jid_cpp, df_col_offset_for_jid_cpp, _, _, _, _ = self.gen_topology_helpers_pointers_for_cpp(list(range(n)), OFFSET=False)
    self.gen_add_parallel_loop("col",str(2*dva_cols_per_partial + 6*NJ))
    self.gen_add_code_line("int col_du = col % " + str(dva_cols_per_partial) + ";")
    if self.robot.floating_base: 
        self.gen_add_code_line(f'int jid = col_du / {n};')
        self.gen_add_code_lines(["// Compute Offsets and Pointers", \
                                "int dq_flag = col == col_du;", \
                                "int Offset_col_du_src = dq_flag * " + str(Offset_dv_dq) + " + !dq_flag * " + str(Offset_dv_dqd) + " + 6*col_du;", \
                                "int Offset_col_du_dst = dq_flag * " + str(Offset_df_dq) + " + !dq_flag * " + str(Offset_df_dqd) + " + 6*col_du;"])
    else:
        select_var_vals = [("int", "jid", [str(jid) for jid in range(NJ)])]
        self.gen_add_multi_threaded_select("col_du", "<", [str(running_sum_dva_cols_per_jid[jid+1]) for jid in range(NJ)], select_var_vals)
        self.gen_add_code_lines(["// Compute Offsets and Pointers", \
                                "int dq_flag = col == col_du; int dva_to_df_adjust = " + df_col_offset_for_jid_cpp + " - " + dva_col_offset_for_jid_cpp + ";", \
                                "int Offset_col_du_src = dq_flag * " + str(Offset_dv_dq) + " + !dq_flag * " + str(Offset_dv_dqd) + " + 6*col_du;", \
                                "int Offset_col_du_dst = dq_flag * " + str(Offset_df_dq) + " + !dq_flag * " + str(Offset_df_dqd) + " + 6*(col_du + dva_to_df_adjust);"])
    self.gen_add_code_line("T *dst = &s_temp[Offset_col_du_dst]; " + \
                        "const T *fx_src = &s_temp[Offset_col_du_src]; " + \
                        "const T *mult_src = &s_temp[" + str(Offset_Iv) + " + 6*jid];")
    # do the adjust for the temp comps
    self.gen_add_code_line("// Adjust pointers for temp comps (if applicable)")
    self.gen_add_code_line("if (col >= " + str(2*dva_cols_per_partial) + ") {", True)
    self.gen_add_code_lines(["int comp = col - " + str(2*dva_cols_per_partial) + "; int comp_col = comp % 6; // int jid = comp / 6;", \
                            "int jid6 = comp - comp_col; int jid36_col6 = 6*jid6 + 6*comp_col;"])
    self.gen_add_code_line("dst = &s_temp[" + str(Offset_FxvI) + " + jid36_col6]; " + \
                        "fx_src = &s_vaf[jid6]; " + \
                        "mult_src = &s_XImats[" + str(36*NJ) + " + jid36_col6];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("fx_times_v<T>(dst, fx_src, mult_src);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"df/du part 1 = fx(dv/du)*Iv\\n\");", \
                                 "printf(\"     and Temp = Fx(v)*I\\n\");", \
                                 "printf(\"-------------------------\\n\");"])
        for ind in range(NJ):
            off_dq, num_cols = _idg_debug_body_buffer(self, ind, Offset_df_dq, n, running_sum_df_cols_per_jid, df_cols_per_jid)
            off_dqd, _ = _idg_debug_body_buffer(self, ind, Offset_df_dqd, n, running_sum_df_cols_per_jid, df_cols_per_jid)
            self.gen_add_code_lines(["printf(\"df[%d]/dq\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(off_dq) + "],6);", \
                                     "printf(\"df[%d]/dqd\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(off_dqd) + "],6);"])
            self.gen_add_code_lines(["printf(\"Fx(v)*I[%d]\\n\"," + str(ind) + ");", \
                                     "printMat<T,6,6>(&s_temp[" + str(Offset_FxvI) + " + 36*" + str(ind) + "],6);"])
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    # then in parallel finish df/du += I*da/du + FxvI*dv/du
    self.gen_add_code_line("// Then in parallel finish df/du += I*da/du + (Fx(v)I)*dv/du")
    self.gen_add_parallel_loop("ind",str(6*2*dva_cols_per_partial))
    if self.robot.floating_base: 
        self.gen_add_code_line(f"int row = ind % 6; int col = ind / 6; int jid = (ind / {6*n}) % {NJ};")
        self.gen_add_code_lines(["T *df_row_col = &s_temp[" + str(Offset_df_dq) + " + 6*col + row];",
                                "const T *dv_col = &s_temp[" + str(Offset_dv_dq) + " + 6*col]; " + \
                                        "const T *da_col = &s_temp[" + str(Offset_da_dq) + " + 6*col];",
                                "int jid36 = 36*jid; const T *I_row = &s_XImats[" + str(36*NJ) + " + jid36 + row]; " + \
                                                "const T *FxvI_row = &s_temp[" + str(Offset_FxvI) + " + jid36 + row];"])
    else:
        self.gen_add_code_line("int row = ind % 6; int col = ind / 6; int col6 = ind - row; int col_du = (col % " + str(dva_cols_per_partial) + ");")
        select_var_vals = [("int", "jid", [str(jid) for jid in range(NJ)])]
        self.gen_add_multi_threaded_select("col_du", "<", [str(running_sum_dva_cols_per_jid[jid+1]) for jid in range(NJ)], select_var_vals)
        self.gen_add_code_lines(["// Compute Offsets and Pointers", \
                                "int dva_to_df_adjust = " + df_col_offset_for_jid_cpp + " - " + dva_col_offset_for_jid_cpp + ";", \
                                "if (col >= " + str(dva_cols_per_partial) + "){dva_to_df_adjust += " + str(df_cols_per_partial - dva_cols_per_partial) + ";}", \
                                "T *df_row_col = &s_temp[" + str(Offset_df_dq) + " + 6*dva_to_df_adjust + ind];",
                                "const T *dv_col = &s_temp[" + str(Offset_dv_dq) + " + col6]; " + \
                                        "const T *da_col = &s_temp[" + str(Offset_da_dq) + " + col6];",
                                "int jid36 = 36*jid; const T *I_row = &s_XImats[" + str(36*n) + " + jid36 + row]; " + \
                                                "const T *FxvI_row = &s_temp[" + str(Offset_FxvI) + " + jid36 + row];"])
    self.gen_add_code_lines(["// Compute the values", \
                             "*df_row_col += dot_prod<T,6,6,1>(I_row,da_col) + dot_prod<T,6,6,1>(FxvI_row,dv_col);"])
    self.gen_add_end_control_flow()

    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"df/du += I*da/du + FxvI*dv/du\\n\");", \
                                 "printf(\"-------------------------\\n\");"])
        for ind in range(NJ):
            off_dq, num_cols = _idg_debug_body_buffer(self, ind, Offset_df_dq, n, running_sum_df_cols_per_jid, df_cols_per_jid)
            off_dqd, _ = _idg_debug_body_buffer(self, ind, Offset_df_dqd, n, running_sum_df_cols_per_jid, df_cols_per_jid)
            self.gen_add_code_lines(["printf(\"df[%d]/dq\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(off_dq) + "],6);", \
                                     "printf(\"df[%d]/dqd\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(off_dqd) + "],6);"])
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    # and also at the same time compute the temp var -X^T * mxf
    # since all temps are done re-use one in practice
    self.gen_add_code_line("// At the same time compute the last temp var: -X^T * mx(f)")
    self.gen_add_code_line("// use Mx(Xv) temp memory as those values are no longer needed")
    self.gen_add_parallel_loop("ind",str(6*n))
    if self.robot.floating_base: 
        self.gen_add_code_line("int XTcol = ind % 6; int jid = ind / 6; int dof_id6 = (jid+5)*6; int jid6 = jid*6;")
        self.gen_add_code_line("s_temp[" + str(Offset_MxXv) + " + ind] = -dot_prod<T,6,1,1>(" + \
                                        "&s_XImats[6*(jid6 + XTcol)], &s_temp[" + str(Offset_Mxf) + " + dof_id6]);")
    else:
        self.gen_add_code_line("int XTcol = ind % 6; int jid6 = ind - XTcol;")
        self.gen_add_code_line("s_temp[" + str(Offset_MxXv) + " + ind] = -dot_prod<T,6,1,1>(" + \
                                        "&s_XImats[6*(jid6 + XTcol)], &s_temp[" + str(Offset_Mxf) + " + jid6]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"Temp = -X^T * mx(f)\\n\");", \
                                 "printf(\"-------------------------\\n\");"])
        for ind in range(NJ):
            self.gen_add_code_lines(["printf(\"-X^T*mx(f)[%d]\\n\"," + str(ind) + ");", \
                                     "printMat<T,1,6>(&s_temp[" + str(Offset_MxXv) + " + 6*" + str(ind) + "],1);"])
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    #
    # BACKWARD PASS
    #
    self.gen_add_code_line("//")
    self.gen_add_code_line("// BACKWARD Pass")
    self.gen_add_code_line("//")
    # update df serially (df_lambda/du = X^T * df/du + {Xmx(f), 0})
    for bfs_level in range(max_bfs_levels,0,-1): # STOP AT 1 because updating parent and last is 0 ---- !!!!!
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        _, _, dva_col_offset_for_jid_cpp, df_col_offset_for_jid_cpp, _, df_col_offset_for_parent_cpp, _, df_col_that_is_jid_cpp = self.gen_topology_helpers_pointers_for_cpp(inds, OFFSET=False)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        self.gen_add_code_line("// df/du update where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))

        if self.DEBUG_MODE:
            self.gen_add_sync()
            self.gen_add_serial_ops()
            for ind in self.robot.get_unique_parent_ids(inds):
                self.gen_add_code_lines(["printf(\"df[%d]/dq (parent update) BEFORE UPDATE\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(df_cols_per_jid[ind]) + ">(&s_temp[" + str(Offset_df_dq + \
                                                    6*running_sum_df_cols_per_jid[ind]) + "],6);", \
                                         "printf(\"df[%d]/dqd (parent update) BEFORE UPDATE\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(df_cols_per_jid[ind]) + ">(&s_temp[" + str(Offset_df_dqd + \
                                                    6*running_sum_df_cols_per_jid[ind]) + "],6);"])
            self.gen_add_end_control_flow()
            self.gen_add_sync()

        # df_lambda/du = X^T * df/du + {Xmx(f), 0}
        # there are 2*(bfs_level + 1) columns per du
        df_cols_per_this_bfs = [df_cols_per_jid[ind] for ind in inds]
        curr_cols_per_du = sum(df_cols_per_this_bfs)
        breakpoints = [sum(df_cols_per_this_bfs[0:i+1]) for i in range(len(inds))]
        col_adjusts = [sum(df_cols_per_this_bfs[0:i]) for i in range(len(inds))]
        sparsity_branch_corrector_vals = [str(jid - self.robot.get_parent_id(jid) - 1) for jid in inds]
        sparsity_branch_corrector_needed = any([int(i) for i in sparsity_branch_corrector_vals])
        if not sparsity_branch_corrector_needed:
            sparsity_branch_corrector = str(0)
        self.gen_add_code_line("// df_lambda/du += X^T * df/du + {Xmx(f), 0}")
        if self.robot.floating_base and len(inds) > 1 and self.robot.has_repeated_parents(inds):
            # Shared parents (quadruped legs → floating root): the per-child df
            # updates COLLIDE on the parent's (dense, n-column) df cells. A
            # slot-major atomicAdd would sum them in warp-scheduling order → the
            # result varies in the last ULP run-to-run (single-block kernels must
            # be bit-deterministic, Inc6). Iterate PARENT-column-major instead:
            # each unique parent df cell is owned by exactly one thread that sums
            # its child slots in FIXED ascending slot order (race-free,
            # deterministic, no atomics). (Fixed-base branched robots keep the
            # sparsity-compressed atomicAdd path below for now.)
            unique_parents = sorted(set(self.robot.get_parent_id(j) for j in inds))
            nup = len(unique_parents)
            upar_csv = ", ".join(str(p) for p in unique_parents)
            jids_csv = ", ".join(str(j) for j in inds)
            pars_csv = ", ".join(str(self.robot.get_parent_id(j)) for j in inds)
            # Per-level {} scope so the compile-time tables don't collide across
            # BFS levels emitted into the same function body.
            self.gen_add_code_line("{", True)
            self.gen_add_code_line(f"const int s_jid_lvl[{len(inds)}] = {{{jids_csv}}};")
            self.gen_add_code_line(f"const int s_par_lvl[{len(inds)}] = {{{pars_csv}}};")
            self.gen_add_code_line(f"const int s_upar_lvl[{nup}] = {{{upar_csv}}};")
            self.gen_add_parallel_loop("ind", str(6*2*n*nup))
            self.gen_add_code_line(f"bool dq_flag = ind < {6*n*nup};")
            self.gen_add_code_line(f"int loc = ind % {6*n*nup};")
            self.gen_add_code_line(f"int up = loc / {6*n}; int rc = loc % {6*n};")
            self.gen_add_code_line("int col = rc / 6; int row = rc % 6;")
            self.gen_add_code_line("int par_l = s_upar_lvl[up];")
            self.gen_add_code_line(f"int du_col_offset = dq_flag * {Offset_df_dq} + !dq_flag * {Offset_df_dqd} + 6*col;")
            self.gen_add_code_line(f"T *dst = &s_temp[du_col_offset + par_l*{6*n} + row];")
            self.gen_add_code_line("T acc = static_cast<T>(0);")
            self.gen_add_code_line(f"for (int slot = 0; slot < {len(inds)}; slot++) {{ if (s_par_lvl[slot] != par_l) continue;")
            self.gen_add_code_line("    int jid = s_jid_lvl[slot];")
            self.gen_add_code_line(f"    acc += dot_prod<T,6,1,1>(&s_XImats[36*jid + 6*row],&s_temp[du_col_offset + jid*{6*n}])")
            self.gen_add_code_line(f"          + dq_flag * (col == jid+5) * s_temp[{Offset_MxXv} + 6*jid + row]; }}")
            self.gen_add_code_line("*dst += acc;")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            self.gen_add_end_control_flow()  # close the per-level {} scope
            continue
        if self.robot.floating_base:
            # dq_flag is deferred (emitted below after extra setup), so pass None.
            jid, _ = self._emit_fb_bfs_level_indexing(inds, n)
            if len(inds) > 1:
                parent_jid = f'parent_jid*{6*n}'
        else:
            self.gen_add_parallel_loop("ind",str(6*2*curr_cols_per_du))
            self.gen_add_code_line(f"int row = ind % 6; int col = ind / 6; int col_du = col % {curr_cols_per_du};")
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                select_var_vals.append(("int", "col_adjust", [str(val) for val in col_adjusts]))
                jid = "jid"
                if sparsity_branch_corrector_needed:
                    select_var_vals.append(("int", "sparsity_branch_corrector", sparsity_branch_corrector_vals))
                    sparsity_branch_corrector = "sparsity_branch_corrector"
                self.gen_add_multi_threaded_select("col_du", "<", [str(val) for val in breakpoints], select_var_vals, True)
                adjustments = "col_du -= col_adjust; // adjust for variable number of columns"
            else: adjustments = ''
        if len(inds) <= 1:
            jid = str(inds[0])
            if self.robot.floating_base: parent_jid = self.robot.get_parent_id(inds[0])*6*n
            elif sparsity_branch_corrector_needed:
                sparsity_branch_corrector = str(self.robot.get_parent_id(inds[0]) - inds[0])
                adjustments = ""
        if self.robot.floating_base: self.gen_add_code_line(f"bool dq_flag = ind < {6*n*len(inds)};")
        else: self.gen_add_code_line(f"int dq_flag = col == col_du;")
        if not self.robot.floating_base and adjustments:
            self.gen_add_code_line(adjustments)
        if self.robot.floating_base: 
            self.gen_add_code_line("int du_col_offset = dq_flag * " + str(Offset_df_dq) + " + !dq_flag * " + str(Offset_df_dqd) + " + 6*col;")
            self.gen_add_code_line(f"T *dst = &s_temp[du_col_offset + {parent_jid} + row];")
            self.gen_add_code_lines(["T update_val = dot_prod<T,6,1,1>(&s_XImats[36*" + jid + f" + 6*row],&s_temp[du_col_offset + {jid}*{6*n}])",
                                    f"              + dq_flag * (col == {jid}+5) * s_temp[" + str(Offset_MxXv) + " + 6*" + jid + " + row];"])
        else:
            self.gen_add_code_line("int du_col_offset = dq_flag * " + str(Offset_df_dq) + " + !dq_flag * " + str(Offset_df_dqd) + " + 6*col_du;")
            self.gen_add_code_line("int dst_adjust = (col_du >= " + df_col_that_is_jid_cpp + ") * 6 * " + sparsity_branch_corrector + "; // adjust for sparsity compression offsets")
            self.gen_add_code_line("T *dst = &s_temp[du_col_offset + 6*" + df_col_offset_for_parent_cpp + " + dst_adjust + row];")
            self.gen_add_code_lines(["T update_val = dot_prod<T,6,1,1>(&s_XImats[36*" + jid + " + 6*row],&s_temp[du_col_offset + 6*" + df_col_offset_for_jid_cpp + "])",
                                    "              + dq_flag * (col_du == " + df_col_that_is_jid_cpp + ") * s_temp[" + str(Offset_MxXv) + " + 6*" + jid + " + row];"])
        # check for repeated parent and add atomics
        if self.robot.has_repeated_parents(inds):
            self.gen_add_code_line("// Atomics required for shared parent")
            self.gen_add_code_line("atomicAdd(dst,update_val);")
        else:
            self.gen_add_code_line("*dst += update_val;")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

        if self.DEBUG_MODE:
            self.gen_add_sync()
            self.gen_add_serial_ops()
            if bfs_level == 0:
                self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                         "printf(\"df/du in bfs waves\\n\");", \
                                         "printf(\"-------------------------\\n\");"])
            self.gen_add_code_line("printf(\"df/du for bfs wave[%d]\\n\"," + str(bfs_level) + ");")
            for ind in self.robot.get_unique_parent_ids(inds):
                self.gen_add_code_lines(["printf(\"df[%d]/dq (parent update)\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(df_cols_per_jid[ind]) + ">(&s_temp[" + str(Offset_df_dq + 6*running_sum_df_cols_per_jid[ind]) + "],6);", \
                                         "printf(\"df[%d]/dqd (parent update)\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(df_cols_per_jid[ind]) + ">(&s_temp[" + str(Offset_df_dqd + 6*running_sum_df_cols_per_jid[ind]) + "],6);"])
            self.gen_add_end_control_flow()
            self.gen_add_sync()

    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"Final dvaf/du\\n\");", \
                                 "printf(\"-------------------------\\n\");"])
        for ind in range(NJ):
            off, num_cols = _idg_debug_body_buffer(self, ind, Offset_dv_dq, n, running_sum_dva_cols_per_jid, dva_cols_per_jid)
            self.gen_add_code_lines(["printf(\"dv[%d]/dq\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(off) + "],6);"])
        for ind in range(NJ):
            off, num_cols = _idg_debug_body_buffer(self, ind, Offset_dv_dqd, n, running_sum_dva_cols_per_jid, dva_cols_per_jid)
            self.gen_add_code_lines(["printf(\"dv[%d]/dqd\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(off) + "],6);"])
        for ind in range(NJ):
            off, num_cols = _idg_debug_body_buffer(self, ind, Offset_da_dq, n, running_sum_dva_cols_per_jid, dva_cols_per_jid)
            self.gen_add_code_lines(["printf(\"da[%d]/dq\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(off) + "],6);"])
        for ind in range(NJ):
            off, num_cols = _idg_debug_body_buffer(self, ind, Offset_da_dqd, n, running_sum_dva_cols_per_jid, dva_cols_per_jid)
            self.gen_add_code_lines(["printf(\"da[%d]/dqd\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(off) + "],6);"])
        # df num_cols intentionally uses the dva-width (get_bfs_level_by_id+1) as
        # the original fixed-base emit did; only the offset uses the df running-sum.
        for ind in range(NJ):
            off, _ = _idg_debug_body_buffer(self, ind, Offset_df_dq, n, running_sum_df_cols_per_jid, df_cols_per_jid)
            _, num_cols = _idg_debug_body_buffer(self, ind, 0, n, running_sum_dva_cols_per_jid, dva_cols_per_jid)
            self.gen_add_code_lines(["printf(\"df[%d]/dq\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(off) + "],6);"])
        for ind in range(NJ):
            off, _ = _idg_debug_body_buffer(self, ind, Offset_df_dqd, n, running_sum_df_cols_per_jid, df_cols_per_jid)
            _, num_cols = _idg_debug_body_buffer(self, ind, 0, n, running_sum_dva_cols_per_jid, dva_cols_per_jid)
            self.gen_add_code_lines(["printf(\"df[%d]/dqd\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(off) + "],6);"])
        self.gen_add_end_control_flow()

    # extract dc/du
    self.gen_add_code_line("// Finally dc[i]/du = S[i]^T*df[i]/du")
    _, S_ind_cpp, _, df_col_offset_for_jid_cpp, _, _, _, _ = self.gen_topology_helpers_pointers_for_cpp(list(range(n)), OFFSET=False)
    S_sign_cpp = self.gen_topology_S_sign_for_cpp(OFFSET=False)
    # Note that for a serial chain this is straightforward (all df are size n) but otherwise gets complicated
    if self.robot.is_serial_chain() or self.robot.floating_base:
        self.gen_add_parallel_loop("ind",str(2*n*n))
        if self.robot.floating_base: 
            self.gen_add_code_line(f"bool dq_flag = ind < {n*n}; int row = ind % {n}; int col = (ind / {n}) % {n};")
            self.gen_add_code_line("int jid = row < 6 ? 0 : row - 5;")
            if 'jid' in S_ind_cpp: S_ind_cpp = S_ind_cpp.replace('jid', 'row')
            if 'jid' in S_sign_cpp: S_sign_cpp = S_sign_cpp.replace('jid', 'row')
            self.gen_add_code_line(f"int srcOffset = dq_flag * {Offset_df_dq} + !dq_flag * {Offset_df_dqd} + {6*n}*jid + 6*col + {S_ind_cpp};")
            self.gen_add_code_line(f"s_dc_du[!dq_flag * {n*n} + {n}*col + row] = (" + S_sign_cpp + ") * s_temp[srcOffset];")
        else:
            self.gen_add_code_line("int jid = ind % " + str(n) + "; int jid_dq_qd = ind / " + str(n) + "; " + 
                                "int jid_du = jid_dq_qd % " + str(n) + "; int dq_flag = jid_du == jid_dq_qd;")
            self.gen_add_code_lines(["int Offset_src = dq_flag * " + str(Offset_df_dq) + " + !dq_flag * " + str(Offset_df_dqd) + \
                                        " + 6 * " + str(n) + " * jid + 6 * jid_du + " + S_ind_cpp + ";",
                                    "int Offset_dst = !dq_flag * " + str(n*n) + " + " + str(n) + " * jid_du + jid;"])
            self.gen_add_code_line("s_dc_du[Offset_dst] = (" + S_sign_cpp + ") * s_temp[Offset_src];")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    else:
        # Branched fixed-base dc/du extraction, fanned to one thread per OUTPUT
        # element (2*n*n) instead of 2*NJ lanes each serially marching all NJ
        # du-columns. Each s_dc_du[half, body=jid, du=djid] is INDEPENDENT/DISJOINT,
        # so the fan needs no extra sync and is bit-exact with the old serial march.
        #
        # The old code advanced Offset_src by flag*6 over djid=0..NJ-1, so a coupling
        # (jid, djid) pair's df-source column is df_col_offset_for_jid_cpp[jid] plus
        # the within-block RANK of djid among jid's coupling set. We resolve that
        # rank map at codegen time and select the per-element src column / write-flag
        # from it (no running pointer); the base offset + S_ind stay as the existing
        # runtime topology-helper expressions, so the emitted arithmetic is the same
        # modulo the rank decode.
        # Build the per-(jid, djid) rank + coupling table: march djid=0..NJ-1
        # (mirrors the old Offset_src += flag*6 order), ranking coupling columns.
        rank_table = [[None]*NJ for _ in range(NJ)]
        for jid in range(NJ):
            rank = 0
            for djid in range(NJ):
                couples = self.robot.get_is_in_subtree_of(djid, jid) or self.robot.get_is_ancestor_of(djid, jid)
                if couples:
                    rank_table[jid][djid] = rank
                    rank += 1
        self.gen_add_parallel_loop("ind", str(2*n*n))
        self.gen_add_code_line("// one thread per output element: decode (half, body jid, du-col djid)")
        self.gen_add_code_line(f"int dq_flag = ind < {n*n}; int jid_du_ind = ind % {n*n};")
        self.gen_add_code_line(f"int jid = jid_du_ind % {n}; int djid = jid_du_ind / {n};")
        # Select this element's within-block source-column RANK and write-flag from
        # the codegen table, keyed on the composite (jid*NJ + djid). Emit a per-djid
        # block so each emits a compact in/not-in test over jid (mirrors the old
        # per-column flag) plus a per-jid rank select only where it couples.
        self.gen_add_code_line("int src_rank = 0; bool flag = false;")
        for djid in range(NJ):
            non_zero_inds = [jid for jid in range(NJ) if rank_table[jid][djid] is not None]
            self.gen_add_code_line("// du-col " + str(djid) + " couples to bodies: " + (",".join(str(i) for i in non_zero_inds) if non_zero_inds else "(none)"))
            if not non_zero_inds:
                continue
            self.gen_add_code_line("if (djid == " + str(djid) + ") {", True)
            # write-flag over jid
            zero_inds = [jid for jid in range(NJ) if rank_table[jid][djid] is None]
            if not zero_inds:
                flag_expr = "true"
            elif len(non_zero_inds) > NJ/2:
                flag_expr = self.gen_var_not_in_list("jid", [str(i) for i in zero_inds])
            else:
                flag_expr = self.gen_var_in_list("jid", [str(i) for i in non_zero_inds])
            self.gen_add_code_line("flag = " + flag_expr + ";")
            # per-jid rank: only distinct ranks need a select; group jids by rank
            ranks = sorted(set(rank_table[jid][djid] for jid in non_zero_inds))
            if len(ranks) == 1:
                self.gen_add_code_line("src_rank = " + str(ranks[0]) + ";")
            else:
                # non-branching sum of (jid==j)*rank_j over coupling jids
                terms = " + ".join("(jid == " + str(jid) + ") * " + str(rank_table[jid][djid]) for jid in non_zero_inds)
                self.gen_add_code_line("src_rank = " + terms + ";")
            self.gen_add_end_control_flow()
        # Source offset: same base (df block start + S_ind) as the old march, plus
        # 6*rank for the within-block column. Output index matches the old
        # Offset_dst progression (!dq_flag*n*n + n*djid + jid).
        self.gen_add_code_line("int Offset_src = dq_flag * " + str(Offset_df_dq) + " + !dq_flag * " + str(Offset_df_dqd) + \
                               " + 6*" + df_col_offset_for_jid_cpp + " + 6*src_rank + " + S_ind_cpp + ";")
        self.gen_add_code_line("int Offset_dst = !dq_flag * " + str(n*n) + " + " + str(n) + "*djid + jid;")
        self.gen_add_code_line("s_dc_du[Offset_dst] = flag * (" + S_sign_cpp + ") * s_temp[Offset_src];")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    # gated joint-damping gradient (dc_dqd diagonal += damping); no-op when
    # USE_JOINT_DYNAMICS is off (Gate-A byte-identical). Sits PAST every
    # extraction fork, so it operates on the final s_dc_du regardless of path.
    _idg_damping_diag_cpp(self, n)

    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"Final dc/du\\n\");", \
                                 "printf(\"-------------------------\\n\");"])
        self.gen_add_code_lines(["printf(\"dc/dq\\n\");", \
                                 "printMat<T," + str(n) + "," + str(n) + ">(&s_dc_du[0]," + str(n) + ");", \
                                 "printf(\"dc/dqd\\n\");", \
                                 "printMat<T," + str(n) + "," + str(n) + ">(&s_dc_du[" + str(n*n) + "]," + str(n) + ");"])
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    self.gen_add_end_function()
    function_code = self.code_str[function_start:]
    self.code_str = self.code_str[:function_start] + _rewrite_inverse_dynamics_gradient_temp_accesses_for_spill(function_code)

def gen_inverse_dynamics_gradient_device_function_call(self,
                                                           use_qdd_input = False,
                                                           scratch_in_smem_expr = "true",
                                                           use_da_df_spill_expr = "false",
                                                           d_workspace_pool_name = "nullptr",
                                                           d_temp_spill_name = "nullptr",
                                                           d_f_ext_name = "d_f_ext",
                                                           mujoco_output_expr = None):
    """Emit the call to `inverse_dynamics_gradient_device`. Arg order MUST
    match the def in gen_inverse_dynamics_gradient_device. Pool/spill regions
    default to nullptr (unused under the matching if-constexpr); the kernel passes
    real pointers per tier. The _qdd C++ name variant additionally threads s_qdd.
    `mujoco_output_expr` (floating only) appends the trailing MUJOCO_OUTPUT template
    arg; None keeps the legacy 3-arg template (byte-identical for non-mjx call sites)."""
    fname = "inverse_dynamics_gradient_device_qdd" if use_qdd_input else "inverse_dynamics_gradient_device"
    tmpl = "<T, " + scratch_in_smem_expr + ", " + use_da_df_spill_expr + \
           ((", " + mujoco_output_expr) if mujoco_output_expr is not None else "") + ">"
    start = fname + tmpl + "(s_dc_du, s_q, s_qd, s_vaf, "
    if use_qdd_input:
        start += "s_qdd, "
    middle = self.gen_insert_helpers_function_call()
    end = ("s_temp, " + d_workspace_pool_name + ", " + d_temp_spill_name + ", "
           + "d_robotModel, " + d_f_ext_name + ", gravity);")
    self.gen_add_code_line(start + middle + end)

def _emit_inverse_dynamics_gradient_mjx_output(self, use_qdd_input):
    """Emit the MuJoCo (mjx) output-convention epilogue for the id-gradient, in
    place on ``s_dc_du`` (2*nv*nv col-major: dc_dq at [0,nv*nv), dc_dqd at
    [nv*nv,2*nv*nv); element [row,col] -> [col*nv + row]). Floating-base only.

    This runs INSIDE inverse_dynamics_gradient_device, AFTER the gradient inner
    call, where ``s_XImats`` and ``s_vaf`` are still live (the kernel body has no
    access to s_XImats — it is built inside this device fn). The gradient inner's
    ``s_temp`` pool is DEAD here, so we carve it as [M | crba-scratch | assembly]
    with NO smem growth: crba_inner writes the full nv x nv generalized mass
    matrix M into the head of s_temp (same [lin;ang;joints] ordering the recipe
    needs — validated by the mjx-crba congruence test), using the tail of s_temp
    as its own working scratch. s_temp is whatever pool SCRATCH_IN_SMEM selected
    (smem or d_workspace), so this is tier-agnostic.

    The exact linear transform recipe (validated to 5.7e-14 vs id_gradient_pin_to
    _mjx — see /tmp/proto_idgrad_mjx_cindex.py) is reproduced op-for-op below.
    Sources: R from the (already xyzw-reordered) base quaternion s_q[3..6];
    v_lin=s_qd[0:3], omega=s_qd[3:6]; qdd_lin=s_qdd[0:3] (or 0 in the qdd=0 bias
    case); tau_lin = LINEAR part of base wrench f[0] = s_vaf[12*nv+3 : 12*nv+6]
    (GRiD spatial vectors are [angular(0:3); linear(3:6)])."""
    nv = self.robot.get_num_vel()
    DQ = 0
    DQD = nv * nv
    # Dead gradient pool layout: [ M (nv*nv) | crba working scratch ].
    M_off = 0
    crba_scr_off = M_off + nv * nv
    self.gen_add_code_line("// === mjx output convention (floating-base id-gradient) ===")
    # 1) Mass matrix via crba_inner. crba reads the live s_XImats and writes M
    #    (nv x nv col-major, generalized [lin;ang;joints] ordering); we only need
    #    M[:,0:3] but compute the full M (correctness-first). crba's scratch is the
    #    tail of the dead pool; SCRATCH_IN_SMEM matches the surrounding pool.
    self.gen_add_code_line("// build the generalized mass matrix M into the (dead) gradient pool via crba_inner")
    self.gen_crba_inner_function_call(
        updated_var_names = dict(
            s_M_name = "(s_temp + " + str(M_off) + ")",
            s_temp_name = "(s_temp + " + str(crba_scr_off) + ")",
            d_workspace_name = "nullptr"),
        temp_in_smem_expr = "SCRATCH_IN_SMEM")
    self.gen_add_sync()
    # 2) The assembly itself — PARALLELIZED across the block (was single-thread;
    #    ~50% runtime overhead at batch 4096). Thread 0 precomputes the shared
    #    helpers (R + the base source vectors + tau_lin, which is only valid on
    #    thread 0) into a 21-float scratch carved at the DEAD crba scratch tail;
    #    the transform then runs as phased parallel loops over rows/cols. Mirrors
    #    the validated op order: dq column_reframe -> dq couplings -> dq base_rotate
    #    rows -> dq pref ; dqd column_reframe -> dqd couplings -> dqd base_rotate rows.
    #
    # s_mjxh scratch layout (21 floats): R[0..8], v_lin[9..11], omega[12..14],
    #   qdd_lin[15..17], tau_lin[18..20]. Sits in the crba working scratch, dead
    #   after the crba_inner call above.
    self.gen_add_code_line("// precompute shared helpers (R + base source vectors) into dead crba scratch")
    self.gen_add_code_line("T *s_mjxh = s_temp + " + str(crba_scr_off) + ";")
    self.gen_add_code_line("if (threadIdx.x == 0 && threadIdx.y == 0) {", True)
    # Build R (row-major R[3*i+j]) from the xyzw base quaternion (s_q already
    # reordered by the input epilogue) — mirrors mujoco_convention.rotation_from
    # _quat_xyzw / the _gen_mjx_build_R_lines helper exactly. Write into s_mjxh[0..8].
    self.gen_add_code_lines([
        "T qx = s_q[3], qy = s_q[4], qz = s_q[5], qw = s_q[6];",
        "T xx = qx*qx, yy = qy*qy, zz = qz*qz;",
        "T xy = qx*qy, xz = qx*qz, yz = qy*qz, wx = qw*qx, wy = qw*qy, wz = qw*qz;",
        "s_mjxh[0] = static_cast<T>(1) - static_cast<T>(2)*(yy+zz); s_mjxh[1] = static_cast<T>(2)*(xy-wz);                    s_mjxh[2] = static_cast<T>(2)*(xz+wy);",
        "s_mjxh[3] = static_cast<T>(2)*(xy+wz);                    s_mjxh[4] = static_cast<T>(1) - static_cast<T>(2)*(xx+zz); s_mjxh[5] = static_cast<T>(2)*(yz-wx);",
        "s_mjxh[6] = static_cast<T>(2)*(xz-wy);                    s_mjxh[7] = static_cast<T>(2)*(yz+wx);                    s_mjxh[8] = static_cast<T>(1) - static_cast<T>(2)*(xx+yy);",
    ])
    self.gen_add_code_lines([
        # sources
        "s_mjxh[9] = s_qd[0]; s_mjxh[10] = s_qd[1]; s_mjxh[11] = s_qd[2];",
        "s_mjxh[12] = s_qd[3]; s_mjxh[13] = s_qd[4]; s_mjxh[14] = s_qd[5];",
        ("s_mjxh[15] = s_qdd[0]; s_mjxh[16] = s_qdd[1]; s_mjxh[17] = s_qdd[2];" if use_qdd_input
         else "s_mjxh[15] = static_cast<T>(0); s_mjxh[16] = static_cast<T>(0); s_mjxh[17] = static_cast<T>(0);"),
        # tau_lin = base-linear generalized force tau[0:3] = the value f[0] linear part
        # (spatial [ang;lin], s_vaf[12*nv+3..5]) CAPTURED into mjx_tau_lin* BEFORE the
        # gradient inner ran (it overwrites the s_vaf f-band with df, so reading s_vaf
        # here would give the gradient, not the value — see gen_..._device capture).
        # mjx_tau_lin* are only valid on thread 0, so they MUST be staged here.
        "s_mjxh[18] = mjx_tau_lin0; s_mjxh[19] = mjx_tau_lin1; s_mjxh[20] = mjx_tau_lin2;",
    ])
    self.gen_add_end_control_flow()  # if threadIdx == 0 (precompute)
    self.gen_add_sync()
    # Local-load lines re-materializing R / v_lin / omega / qdd_lin / tau_lin / s_M
    # from s_mjxh, emitted at the top of each parallel-loop body so the verbatim
    # formula lines below keep working unchanged.
    _LOAD = [
        "T R[9]; R[0]=s_mjxh[0];R[1]=s_mjxh[1];R[2]=s_mjxh[2];R[3]=s_mjxh[3];R[4]=s_mjxh[4];R[5]=s_mjxh[5];R[6]=s_mjxh[6];R[7]=s_mjxh[7];R[8]=s_mjxh[8];",
        "T v_lin[3] = {s_mjxh[9], s_mjxh[10], s_mjxh[11]};",
        "T omega[3] = {s_mjxh[12], s_mjxh[13], s_mjxh[14]};",
        "T qdd_lin[3] = {s_mjxh[15], s_mjxh[16], s_mjxh[17]};",
        "T tau_lin[3] = {s_mjxh[18], s_mjxh[19], s_mjxh[20]};",
        "T *s_M = s_temp + " + str(M_off) + ";",
    ]
    # ---- dc_dq half ----
    # Phase A: dc_dq column_reframe (cols 0:3, row r) THEN dc_dq couplings (cols
    #   3:6, row r). Disjoint cols; reads dc_dqd ORIGINAL + s_M. parallel over r.
    self.gen_add_code_line("// Phase A: dc_dq column_reframe (cols 0:3) + couplings (cols 3:6), per row")
    self.gen_add_parallel_loop("r", str(nv))
    self.gen_add_code_lines(_LOAD)
    self.gen_add_code_lines([
        "// dc_dq: column_reframe cols 0:3 <- cols . R^T",
        "T c0 = s_dc_du[" + str(DQ) + " + 0*" + str(nv) + " + r], c1 = s_dc_du[" + str(DQ) + " + 1*" + str(nv) + " + r], c2 = s_dc_du[" + str(DQ) + " + 2*" + str(nv) + " + r];",
        "s_dc_du[" + str(DQ) + " + 0*" + str(nv) + " + r] = c0*R[0] + c1*R[1] + c2*R[2];",
        "s_dc_du[" + str(DQ) + " + 1*" + str(nv) + " + r] = c0*R[3] + c1*R[4] + c2*R[5];",
        "s_dc_du[" + str(DQ) + " + 2*" + str(nv) + " + r] = c0*R[6] + c1*R[7] + c2*R[8];",
    ])
    self.gen_add_code_line("// dc_dq: base-velocity couplings into angular cols 3+a (this row)")
    self.gen_add_code_line("for (int a = 0; a < 3; a++) {", True)
    self.gen_add_code_lines([
        # e_a x w  for a basis vector e_a:  (e x w)[i] = sum_jk eps_ijk e_a_j w_k
        # = derivative columns of the unit-axis cross product.  We form jv and ja.
        "T jv[3], ja[3];",
        # e_a x w (basis-vector cross): [0]=(a==1)w2-(a==2)w1, [1]=-(a==0)w2+(a==2)w0,
        #   [2]=(a==0)w1-(a==1)w0  (verified == numpy cross(e_a,w)).
        # jv = -(e_a x v_lin)
        "jv[0] = -((a==1)*( v_lin[2]) + (a==2)*(-v_lin[1]));",
        "jv[1] = -((a==0)*(-v_lin[2]) + (a==2)*( v_lin[0]));",
        "jv[2] = -((a==0)*( v_lin[1]) + (a==1)*(-v_lin[0]));",
        # ov = omega x v_lin
        "T ov0 = omega[1]*v_lin[2] - omega[2]*v_lin[1];",
        "T ov1 = omega[2]*v_lin[0] - omega[0]*v_lin[2];",
        "T ov2 = omega[0]*v_lin[1] - omega[1]*v_lin[0];",
        # ev = e_a x v_lin
        "T ev0 = (a==1)*( v_lin[2]) + (a==2)*(-v_lin[1]);",
        "T ev1 = (a==0)*(-v_lin[2]) + (a==2)*( v_lin[0]);",
        "T ev2 = (a==0)*( v_lin[1]) + (a==1)*(-v_lin[0]);",
        # ja = -(e_a x qdd_lin) - (e_a x ov) + (omega x ev)
        "T eq0 = (a==1)*( qdd_lin[2]) + (a==2)*(-qdd_lin[1]);",
        "T eq1 = (a==0)*(-qdd_lin[2]) + (a==2)*( qdd_lin[0]);",
        "T eq2 = (a==0)*( qdd_lin[1]) + (a==1)*(-qdd_lin[0]);",
        "T eov0 = (a==1)*( ov2) + (a==2)*(-ov1);",
        "T eov1 = (a==0)*(-ov2) + (a==2)*( ov0);",
        "T eov2 = (a==0)*( ov1) + (a==1)*(-ov0);",
        "T oev0 = omega[1]*ev2 - omega[2]*ev1;",
        "T oev1 = omega[2]*ev0 - omega[0]*ev2;",
        "T oev2 = omega[0]*ev1 - omega[1]*ev0;",
        "ja[0] = -eq0 - eov0 + oev0;",
        "ja[1] = -eq1 - eov1 + oev1;",
        "ja[2] = -eq2 - eov2 + oev2;",
        # dc_dq[r,3+a] += dc_dqd[r,0:3] @ jv + M[r,0:3] @ ja
        "T acc = static_cast<T>(0);",
        "for (int k = 0; k < 3; k++) { acc += s_dc_du[" + str(DQD) + " + k*" + str(nv) + " + r]*jv[k] + s_M[r + " + str(nv) + "*k]*ja[k]; }",
        "s_dc_du[" + str(DQ) + " + (3+a)*" + str(nv) + " + r] += acc;",
    ])
    self.gen_add_end_control_flow()  # for a
    self.gen_add_end_control_flow()  # parallel r
    self.gen_add_sync()
    # Phase B: dc_dq base_rotate_rows (rows 0:3 of col c). Reads dc_dq written by A.
    self.gen_add_code_line("// Phase B: dc_dq base_rotate_rows rows 0:3 <- R . rows, per col")
    self.gen_add_parallel_loop("c", str(nv))
    self.gen_add_code_lines(_LOAD)
    self.gen_add_code_lines([
        "T m0 = s_dc_du[" + str(DQ) + " + c*" + str(nv) + " + 0], m1 = s_dc_du[" + str(DQ) + " + c*" + str(nv) + " + 1], m2 = s_dc_du[" + str(DQ) + " + c*" + str(nv) + " + 2];",
        "s_dc_du[" + str(DQ) + " + c*" + str(nv) + " + 0] = R[0]*m0 + R[1]*m1 + R[2]*m2;",
        "s_dc_du[" + str(DQ) + " + c*" + str(nv) + " + 1] = R[3]*m0 + R[4]*m1 + R[5]*m2;",
        "s_dc_du[" + str(DQ) + " + c*" + str(nv) + " + 2] = R[6]*m0 + R[7]*m1 + R[8]*m2;",
    ])
    self.gen_add_end_control_flow()  # parallel c
    self.gen_add_sync()
    # Phase C: dc_dq pref (e_a x tau_lin block), adds dc_dq rows 0:3 col 3+a.
    self.gen_add_code_line("// Phase C: dc_dq pref  dq[0:3, 3+a] += R @ (e_a x tau_lin), per a")
    self.gen_add_parallel_loop("a", "3")
    self.gen_add_code_lines(_LOAD)
    self.gen_add_code_lines([
        "T et0 = (a==1)*( tau_lin[2]) + (a==2)*(-tau_lin[1]);",
        "T et1 = (a==0)*(-tau_lin[2]) + (a==2)*( tau_lin[0]);",
        "T et2 = (a==0)*( tau_lin[1]) + (a==1)*(-tau_lin[0]);",
        "T w0 = R[0]*et0 + R[1]*et1 + R[2]*et2;",
        "T w1 = R[3]*et0 + R[4]*et1 + R[5]*et2;",
        "T w2 = R[6]*et0 + R[7]*et1 + R[8]*et2;",
        "s_dc_du[" + str(DQ) + " + (3+a)*" + str(nv) + " + 0] += w0;",
        "s_dc_du[" + str(DQ) + " + (3+a)*" + str(nv) + " + 1] += w1;",
        "s_dc_du[" + str(DQ) + " + (3+a)*" + str(nv) + " + 2] += w2;",
    ])
    self.gen_add_end_control_flow()  # parallel a
    self.gen_add_sync()
    # ---- dc_dqd half ----
    # Phase D: dc_dqd column_reframe (cols 0:3, row r). MUST follow Phase A (which
    #   read dc_dqd original); the intervening syncs guarantee it.
    self.gen_add_code_line("// Phase D: dc_dqd column_reframe cols 0:3 <- cols . R^T, per row")
    self.gen_add_parallel_loop("r", str(nv))
    self.gen_add_code_lines(_LOAD)
    self.gen_add_code_lines([
        "T c0 = s_dc_du[" + str(DQD) + " + 0*" + str(nv) + " + r], c1 = s_dc_du[" + str(DQD) + " + 1*" + str(nv) + " + r], c2 = s_dc_du[" + str(DQD) + " + 2*" + str(nv) + " + r];",
        "s_dc_du[" + str(DQD) + " + 0*" + str(nv) + " + r] = c0*R[0] + c1*R[1] + c2*R[2];",
        "s_dc_du[" + str(DQD) + " + 1*" + str(nv) + " + r] = c0*R[3] + c1*R[4] + c2*R[5];",
        "s_dc_du[" + str(DQD) + " + 2*" + str(nv) + " + r] = c0*R[6] + c1*R[7] + c2*R[8];",
    ])
    self.gen_add_end_control_flow()  # parallel r
    self.gen_add_sync()
    # Phase E: dc_dqd couplings (jl,jn), RMW on cols 0:3 (after D) + writes cols 3:6.
    self.gen_add_code_line("// Phase E: dc_dqd couplings  col 0+a += M[:,0:3]@(-(omega x R^T e_a)) ; col 3+a += M[:,0:3]@(-(e_a x v_lin)), per row")
    self.gen_add_parallel_loop("r", str(nv))
    self.gen_add_code_lines(_LOAD)
    self.gen_add_code_lines([
        "T mr0 = s_M[r + " + str(nv) + "*0], mr1 = s_M[r + " + str(nv) + "*1], mr2 = s_M[r + " + str(nv) + "*2];",
    ])
    self.gen_add_code_line("for (int a = 0; a < 3; a++) {", True)
    self.gen_add_code_lines([
        # R^T e_a = column a of R^T = row a of R = (R[3a+0],R[3a+1],R[3a+2])
        "T rte0 = R[3*a + 0], rte1 = R[3*a + 1], rte2 = R[3*a + 2];",
        # jv_lin = -(omega x R^T e_a)
        "T jl0 = -(omega[1]*rte2 - omega[2]*rte1);",
        "T jl1 = -(omega[2]*rte0 - omega[0]*rte2);",
        "T jl2 = -(omega[0]*rte1 - omega[1]*rte0);",
        # jv_ang = -(e_a x v_lin)
        "T jn0 = -((a==1)*( v_lin[2]) + (a==2)*(-v_lin[1]));",
        "T jn1 = -((a==0)*(-v_lin[2]) + (a==2)*( v_lin[0]));",
        "T jn2 = -((a==0)*( v_lin[1]) + (a==1)*(-v_lin[0]));",
        "s_dc_du[" + str(DQD) + " + (0+a)*" + str(nv) + " + r] += mr0*jl0 + mr1*jl1 + mr2*jl2;",
        "s_dc_du[" + str(DQD) + " + (3+a)*" + str(nv) + " + r] += mr0*jn0 + mr1*jn1 + mr2*jn2;",
    ])
    self.gen_add_end_control_flow()  # for a
    self.gen_add_end_control_flow()  # parallel r
    self.gen_add_sync()
    # Phase F: dc_dqd base_rotate_rows (rows 0:3 of col c). Reads dc_dqd from D/E.
    self.gen_add_code_line("// Phase F: dc_dqd base_rotate_rows rows 0:3 <- R . rows, per col")
    self.gen_add_parallel_loop("c", str(nv))
    self.gen_add_code_lines(_LOAD)
    self.gen_add_code_lines([
        "T m0 = s_dc_du[" + str(DQD) + " + c*" + str(nv) + " + 0], m1 = s_dc_du[" + str(DQD) + " + c*" + str(nv) + " + 1], m2 = s_dc_du[" + str(DQD) + " + c*" + str(nv) + " + 2];",
        "s_dc_du[" + str(DQD) + " + c*" + str(nv) + " + 0] = R[0]*m0 + R[1]*m1 + R[2]*m2;",
        "s_dc_du[" + str(DQD) + " + c*" + str(nv) + " + 1] = R[3]*m0 + R[4]*m1 + R[5]*m2;",
        "s_dc_du[" + str(DQD) + " + c*" + str(nv) + " + 2] = R[6]*m0 + R[7]*m1 + R[8]*m2;",
    ])
    self.gen_add_end_control_flow()  # parallel c
    self.gen_add_sync()


def gen_inverse_dynamics_gradient_device(self, use_qdd_input = False):
    """Emit `inverse_dynamics_gradient_device` — the whole inverse_dynamics_gradient orchestration
    as ONE inner that OWNS its scratch (s_temp) placement (inner-owns-placement;
    mirrors gen_fdsva_so_device). It wraps, in order:
      [repoint s_temp] -> load_update_XImats -> inverse_dynamics_inner (vaf) ->
      inverse_dynamics_gradient_inner (the inverse_dynamics_gradient band sub-inner).
    Because the s_temp repoint happens at the very top, EVERY consumer below —
    including the XImats helper's sincos scratch — follows the placement, so the
    kernel never repoints s_temp from the outside.

    TWO independent template flags:
      SCRATCH_IN_SMEM  : the shared s_temp pool lives in smem (true) or routes the
                         WHOLE pool to d_workspace (false; the rung-2 global-temp
                         path). Dominant lever on big floating humanoids.
      USE_DA_DF_SPILL  : the inverse_dynamics_gradient band selectively spills its da_dq..fxvi band to
                         d_temp_spill (rung 1). Threaded through to the band
                         sub-inner's grid_id_du_temp_ptr<T, USE_DA_DF_SPILL> helper.
    The 3-rung menu (see _INVERSE_DYNAMICS_GRADIENT_PICK_FLAGS): pick0=(SMEM=true, SPILL=false) full;
    pick1=(true, true) selective band; pick2=(false, false) whole-pool global.

    Pointer params are caller-supplied (the kernel decides where the OUTPUT s_dc_du
    lives and hands in the spill regions); only the s_temp POOL placement is the
    inner's call. The id inner `inverse_dynamics_inner_vaf` is FROZEN and
    placement-free: after the repoint, s_temp already points at the right pool, so
    passing it through is correct with no id-side change."""
    n = self.robot.get_num_vel()
    func_params = [
        "s_dc_du is the output buffer (caller places); size 2*NUM_VEL*NUM_VEL = " + str(2*n*n),
        "s_q is the vector of joint positions",
        "s_qd is the vector of joint velocities",
        "s_vaf is the id intermediate band (caller places); size 18*NUM_JOINTS = " + str(18*n),
        "s_temp is the shared scratch pool (used when SCRATCH_IN_SMEM)",
        "d_workspace is the global scratch pool (used when !SCRATCH_IN_SMEM)",
        "d_temp_spill is the inverse_dynamics_gradient da_df band spill region (used when USE_DA_DF_SPILL)",
        "d_robotModel holds XImats/topology; gravity is the gravity constant",
    ]
    fname = "inverse_dynamics_gradient_device_qdd" if use_qdd_input else "inverse_dynamics_gradient_device"
    func_def_start = "void " + fname + "(T *s_dc_du, const T *s_q, const T *s_qd, T *s_vaf, "
    if use_qdd_input:
        func_def_start += "const T *s_qdd, "
    func_def_end = ("T *s_temp, T *d_workspace, T *d_temp_spill, "
                    "const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity) {")
    func_params.append("d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr")
    if use_qdd_input:
        func_params.insert(4, "s_qdd is the vector of joint accelerations")
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -2)
    func_def = func_def_start + func_def_end
    self.gen_add_func_doc("inverse_dynamics_gradient orchestration as a single inner-owns-placement device function",
                          ["Owns the s_temp pool placement; the repoint covers every consumer below (incl. the XImats helper's sincos scratch)"],
                          func_params, None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag. Added
    # LAST so existing positional <T,SCRATCH,SPILL> call sites are unaffected; the
    # default (false) instantiation if-constexpr-elides the epilogue -> byte-
    # identical. The epilogue runs HERE (not in the kernel body) because it needs
    # the live s_XImats (crba M) + s_vaf (tau) that only exist inside this fn.
    mjx_device = self.robot.floating_base and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis() or self.robot.robot_has_spherical())
    if mjx_device:
        # Forward-declare crba_inner: the mjx epilogue reuses it to build M, but crba
        # is emitted LATER in grid.cuh than this gradient. A declaration before the
        # use resolves the ordering (the definition follows in the same translation
        # unit). Gated on mjx_device so fixed-base headers stay byte-identical.
        self.gen_add_code_line("template <typename T, bool TEMP_IN_SMEM>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("void crba_inner(T *s_M, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, T *d_workspace, const T gravity);  // fwd decl (default arg lives on the definition)")
    if mjx_device:
        self.gen_add_code_line("template <typename T, bool SCRATCH_IN_SMEM = true, bool USE_DA_DF_SPILL = false, bool MUJOCO_OUTPUT = false>")
    else:
        self.gen_add_code_line("template <typename T, bool SCRATCH_IN_SMEM = true, bool USE_DA_DF_SPILL = false>")
    # __forceinline__ so the whole orchestration inlines into the calling kernel.
    # Under -rdc a separate __device__ wrapper keeps its callees as distinct
    # functions whose regcount must fit the kernel's launch_bounds budget
    # (80 at LITE / 64 at MINIMAL) -> ptxas regcount error. Inlining folds them
    # into the kernel. See _fdsva_so.py:295-300 / HANDOFF.md "Problem 1".
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line(func_def, True)
    # Inner owns the pool placement; the repoint covers every consumer below
    # (incl. the XImats helper's sincos scratch), so no caller-side repoint.
    self.gen_add_code_line("if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }")
    self.gen_load_update_XImats_helpers_function_call()
    self.gen_inverse_dynamics_inner_function_call(False, use_qdd_input)
    if mjx_device:
        # CAPTURE the base-linear generalized force tau[0:3] = the VALUE f[0] linear
        # part (s_vaf[12*nv+3..5]) NOW — the gradient inner below overwrites the s_vaf
        # f-band with df (gradient), so the mjx pref term must use the captured value.
        # Declared unconditionally (floating) so the registers persist into the
        # epilogue; populated only in the mjx instantiation (if constexpr) -> the pin
        # PTX is unchanged (unused-decl DCE).
        # The f-band in the GRADIENT kernel is body-indexed at 12*NJ (NJ=num_joints=
        # num_bodies for non-mimic), NOT 12*nv — the base wrench f[0] is at s_vaf[12*NJ
        # : +6], spatial [ang;lin], so tau[0:3] (base-linear generalized force) = the
        # LINEAR part f[0][3:6] = s_vaf[12*NJ+3 : 12*NJ+6].
        _fb = 12 * self.robot.get_num_joints()
        self.gen_add_code_line("T mjx_tau_lin0 = static_cast<T>(0), mjx_tau_lin1 = static_cast<T>(0), mjx_tau_lin2 = static_cast<T>(0);")
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        self.gen_add_sync()
        self.gen_add_code_line("if (threadIdx.x == 0 && threadIdx.y == 0) {", True)
        self.gen_add_code_line("mjx_tau_lin0 = s_vaf[" + str(_fb + 3) + "]; mjx_tau_lin1 = s_vaf[" + str(_fb + 4) + "]; mjx_tau_lin2 = s_vaf[" + str(_fb + 5) + "];")
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
    self.gen_inverse_dynamics_gradient_inner_function_call(
        dict(d_temp_spill_name = "d_temp_spill", temp_spill_flag_name = "USE_DA_DF_SPILL")
    )
    if mjx_device:
        self.gen_add_sync()
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        _emit_inverse_dynamics_gradient_mjx_output(self, use_qdd_input)
        self.gen_add_end_control_flow()
    self.gen_add_end_function()

def gen_inverse_dynamics_gradient_kernel_max_temp_mem_size(self):
    n = self.robot.get_num_vel()
    # s_vaf is 18*NB (body-indexed) for mimic robots; 18*n otherwise (non-mimic
    # floating has nv > NB so 18*n is the safe/byte-identical size).
    vaf_cnt = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else n)
    base_size = 2*n + n*2*n + vaf_cnt + n
    temp_mem_size = self.gen_inverse_dynamics_gradient_inner_temp_mem_size()
    return base_size + temp_mem_size

_INVERSE_DYNAMICS_GRADIENT_PICK_FLAGS = [
    # (use_selective_spill, use_global_temp)
    (False, False),   # pick 0: full smem
    (True,  False),   # pick 1: selective spill (da_df band to workspace)
    (False, True),    # pick 2: global temp (entire s_temp to workspace)
]

def _emit_inverse_dynamics_gradient_kernel_body_for_flags(self, NUM_POS, n, use_selective_spill, use_global_temp,
                                      use_qdd_input, single_call_timing, mjx_kernel = False):
    """Emit the inverse_dynamics_gradient kernel body for one tier's spill flags.
    `mjx_kernel` (floating + non-mimic/skew, with-qdd): emit the MUJOCO_OUTPUT
    input-convert (before the device fn builds XImats) and forward the flag to the
    device call (the output epilogue lives inside the device fn)."""
    # s_vaf is body-indexed (the ID inner writes NB bodies, stride 6). For a
    # MIMIC robot (fixed base) NB > nv, so size it 18*NB to avoid overflowing
    # into the adjacent arena buffers. Non-mimic keeps 18*n (byte-identical;
    # for floating non-mimic nv > NB so 18*n already covers the body writes).
    _vaf_cnt = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else n)
    extra_t_buffers = [("s_q_qd", n + NUM_POS), ("s_dc_du", n*2*n), ("s_vaf", _vaf_cnt)]
    if use_qdd_input:
        extra_t_buffers.append(("s_qdd", n))
    # Mimic robots use a dense inner with NO sparse-band selective spill, so the
    # selective-spill smem size collapses to the dense full size (the inner
    # ignores d_temp_spill and reads the whole dense pool from s_temp/workspace).
    _selective_shared = (
        self.gen_inverse_dynamics_gradient_inner_temp_mem_size()
        if (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis() or self.robot.robot_has_spherical())
        else self.gen_inverse_dynamics_gradient_temp_layout()["selective_shared_count"]
    )
    shared_mem_size = 0 if use_global_temp else (
        _selective_shared
        if use_selective_spill else self.gen_inverse_dynamics_gradient_inner_temp_mem_size()
    )
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size, extra_t_buffers = extra_t_buffers, include_linalg_scratch=True)
    self.gen_add_code_line("T *d_temp_spill = nullptr; (void)d_temp_spill;")
    self.gen_add_code_line("T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(NUM_POS) + "];")
    if not single_call_timing:
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",block_level = True)
        if use_qdd_input:
            self.gen_kernel_load_inputs("q_qd",str(n + NUM_POS),"qdd",str(n),stride="stride_q_qd",stride2=str(NUM_POS))
        else:
            self.gen_kernel_load_inputs("q_qd",str(n + NUM_POS),stride="stride_q_qd")
        # mjx input convert (before the device fn builds XImats so X[0] uses the
        # reordered quaternion). Reorders the base quaternion + converts the base
        # velocity/accel to the pin frame in place on s_q/s_qd/s_qdd.
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert(q_name="s_q", qd_name="s_qd", qdd_name="s_qdd")
            self.gen_add_end_control_flow()
        # The kernel only SLICES the workspace band pointers; the device owns
        # the s_temp pool placement (the whole-pool global-temp repoint is its
        # SCRATCH_IN_SMEM=false path). Per-rung flags are passed as literals.
        if use_selective_spill:
            self.gen_add_code_line("d_temp_spill = reinterpret_cast<T *>(&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()]);")
        self.gen_add_code_line("// compute — the orchestration inner owns its s_temp pool placement")
        self.gen_inverse_dynamics_gradient_device_function_call(
            use_qdd_input,
            scratch_in_smem_expr = ("false" if use_global_temp else "true"),
            use_da_df_spill_expr = ("true" if use_selective_spill else "false"),
            d_workspace_pool_name = ("reinterpret_cast<T *>(&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()])" if use_global_temp else "nullptr"),
            d_temp_spill_name = ("d_temp_spill" if use_selective_spill else "nullptr"),
            mujoco_output_expr = ("MUJOCO_OUTPUT" if mjx_kernel else None))
        self.gen_add_sync()
        self.gen_kernel_save_result("dc_du",str(n*2*n),stride=str(n*2*n))
        self.gen_add_end_control_flow()
    else:
        if use_qdd_input:
            self.gen_kernel_load_inputs("q_qd",str(n + NUM_POS),"qdd",str(n))
        else:
            self.gen_kernel_load_inputs("q_qd",str(n + NUM_POS))
        if use_selective_spill:
            self.gen_add_code_line("d_temp_spill = reinterpret_cast<T *>(d_workspace);")
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        if use_qdd_input:
            self.gen_anti_licm_input_reload("q_qd",str(n + NUM_POS),"qdd",str(n),feedback_from="dc_du")
        else:
            self.gen_anti_licm_input_reload("q_qd",str(n + NUM_POS),feedback_from="dc_du")
        # device owns s_temp placement (whole-pool global path = SCRATCH_IN_SMEM=false).
        # Single-timing forwards MUJOCO_OUTPUT (perf only; the input-convert is
        # omitted here — the anti-LICM loop reloads raw inputs each rep, so a one-
        # shot convert would not apply; correctness is validated via the full kernel).
        self.gen_inverse_dynamics_gradient_device_function_call(
            use_qdd_input,
            scratch_in_smem_expr = ("false" if use_global_temp else "true"),
            use_da_df_spill_expr = ("true" if use_selective_spill else "false"),
            d_workspace_pool_name = ("reinterpret_cast<T *>(d_workspace)" if use_global_temp else "nullptr"),
            d_temp_spill_name = ("d_temp_spill" if use_selective_spill else "nullptr"),
            mujoco_output_expr = ("MUJOCO_OUTPUT" if mjx_kernel else None))
        self.gen_anti_licm_output_write("dc_du")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("dc_du",str(n*2*n))


def gen_inverse_dynamics_gradient_kernel(self, use_qdd_input = False, single_call_timing = False):
    NUM_POS = self.robot.get_num_pos()
    n = self.robot.get_num_vel()
    func_params = ["d_dc_du is a pointer to memory for the final result of size 2*NUM_VEL*NUM_VEL = " + str(2*n*n), \
                   "d_q_dq is the vector of joint positions and velocities", \
                   "stride_q_qd is the stide between each q, qd", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr", \
                   "gravity is the gravity constant", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    func_def_start = "void inverse_dynamics_gradient_kernel(T *d_dc_du, unsigned char *d_workspace, const T *d_q_qd, const int stride_q_qd, "
    func_def_end = "T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    if use_qdd_input:
        func_def_start += "const T *d_qdd, "
        func_params.insert(-2,"d_qdd is the vector of joint accelerations")
    else:
        func_notes.append("optimized for qdd = 0")
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Computes the gradient of inverse dynamics",func_notes,func_params,None)
    # MUJOCO_OUTPUT (floating + non-mimic/skew, with-qdd kernel only): compile-time
    # mjx output-convention flag. Added LAST after RESOURCE_TIER so existing
    # positional <T,TIER> call sites are unaffected; default false -> the input
    # convert + output epilogue if-constexpr-elide to byte-identical PTX. The
    # qdd=0 (bias) kernel never carries it (mjx needs the with-qdd surface).
    mjx_kernel = (self.robot.floating_base and use_qdd_input
                  and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis() or self.robot.robot_has_spherical()))
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # Tier dispatch: collapsed picks → single body (current behavior);
    # divergent picks → 3 if-constexpr branches.
    picks = getattr(self, "inverse_dynamics_gradient_spill_tier_3way", (0, 0, 0))
    def _emit_inverse_dynamics_gradient_body(pick):
        uss, ugt = _INVERSE_DYNAMICS_GRADIENT_PICK_FLAGS[pick]
        _emit_inverse_dynamics_gradient_kernel_body_for_flags(self, NUM_POS, n, uss, ugt, use_qdd_input, single_call_timing, mjx_kernel)
    self.gen_tier_dispatch(picks, _emit_inverse_dynamics_gradient_body)
    self.gen_add_end_function()

def gen_inverse_dynamics_gradient_host(self, mode = 0):
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
    func_def_start = "void inverse_dynamics_gradient(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end =   "                               const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
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
    # to the with-qdd kernel launch (the mjx-capable overload). Added LAST so
    # existing positional template args are unaffected; default false -> byte-
    # identical. The qdd=0 launch never carries it (mjx needs the with-qdd surface).
    mjx_host = (self.robot.floating_base
                and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis() or self.robot.robot_has_spherical()))
    if mjx_host:
        self.gen_add_code_line("template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    else:
        self.gen_add_code_line("template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"inverse_dynamics_gradient requires all-data or dynamics gridData\");")
    func_call_start = "inverse_dynamics_gradient_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd,stride_q_qd,"
    # the with-qdd launch is the mjx-capable overload: name the tier positionally
    # to reach the trailing MUJOCO_OUTPUT flag.
    qdd_kernel_tmpl = "inverse_dynamics_gradient_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "inverse_dynamics_gradient_kernel<T, RESOURCE_TIER>"
    func_call_qdd_start = qdd_kernel_tmpl + "<<<block_dimms,thread_dimms,INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dc_du,hd_data->d_workspace,hd_data->d_q_qd,stride_q_qd,"
    func_call_end = "hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("inverse_dynamics_gradient_kernel<","inverse_dynamics_gradient_kernel_single_timing<")
        func_call_qdd_start = func_call_qdd_start.replace("inverse_dynamics_gradient_kernel<","inverse_dynamics_gradient_kernel_single_timing<")
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
    # the with-qdd launch uses the mjx-capable template (func_call_qdd_start).
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
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"inverse_dynamics_gradient\", INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    workspace_bytes = "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()" if single_call_timing else "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(num_timesteps)"
    self.gen_add_code_line("if (GRID_INVERSE_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, " + workspace_bytes + "));}")
    self.gen_add_code_lines(func_call_code)
    self.gen_add_code_line("if (GRID_INVERSE_DYNAMICS_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_end_l2_persisting(0));}")
    if not compute_only:
        # then transfer memory back
        self.gen_add_code_lines(["// finally transfer the result back", \
                                 "gpuErrchk(cudaMemcpy(hd_data->h_dc_du,hd_data->d_dc_du,2*NUM_VEL*NUM_VEL*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchkKernel();"])
    # finally report out timing if requested
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("inverse_dynamics_gradient"))
    self.gen_add_end_function()

def gen_inverse_dynamics_gradient(self):
    # first the inverse_dynamics_gradient band sub-inner (internal helper composed by the orchestration
    # _device; also called by forward_dynamics_gradient / integrator_gradient orchestrators).
    self.gen_inverse_dynamics_gradient_inner()
    # then the canonical _device (orchestrator: owns s_temp placement; wraps XImats
    # + id-inner + inverse_dynamics_gradient band sub-inner; called from kernel and forward_dynamics_gradient / integrator).
    # Both qdd variants (qdd-input vs qdd=0 specialization).
    self.gen_inverse_dynamics_gradient_device(True)
    self.gen_inverse_dynamics_gradient_device(False)
    # and the kernels
    self.gen_inverse_dynamics_gradient_kernel(True,True)
    self.gen_inverse_dynamics_gradient_kernel(True,False)
    self.gen_inverse_dynamics_gradient_kernel(False,True)
    self.gen_inverse_dynamics_gradient_kernel(False,False)
    # and host wrapeprs
    self.gen_inverse_dynamics_gradient_host(0)
    self.gen_inverse_dynamics_gradient_host(1)
    self.gen_inverse_dynamics_gradient_host(2)


def _gen_inverse_dynamics_gradient_mimic_inner(self, nv, NB):
    """Dense serial mimic ID-gradient (T3-finisher P3 fixed-base; B1 floating).

    Mirrors RBDReference.rnea_grad exactly, in REDUCED v-space:
      forward pass dq/dqd -> per-body dv_du, da_du, df_du (6 x nv x NB),
      backward pass       -> dc_dq, dc_dqd (nv x nv).
    Every joint-velocity read scales by the body's mimic multiplier alpha and
    every dc_du / df_du write accumulates (+=) into the body's reduced v-slot,
    so a mimic body and its target fold together. Non-root bodies are single
    -DoF (the floating root is the only multi-DoF joint and is never mimic);
    the floating-base root (jid 0) carries a 6-DoF motion subspace S = I_6
    over v-slots 0..5, so its own-DoF contributions LOOP over the 6 root DoFs
    (mirrors RBDReference.rnea_grad_fpass_dq's `for ii in range(len(idx))`).

    Output s_dc_du is 2*nv*nv, column-major nv x nv per half:
      dc_dq  at [0, nv*nv)       element [v_i, c] -> s_dc_du[c*nv + v_i]
      dc_dqd at [nv*nv, 2*nv*nv)  element [v_i, c] -> s_dc_du[nv*nv + c*nv + v_i]
    s_vaf is body-indexed (stride NB): v @ s_vaf[6*ind], a @ s_vaf[6*NB+6*ind],
    f @ s_vaf[12*NB+6*ind]. The big dense buffers live in s_temp (routed to
    workspace at the global-temp tier for humanoid-scale NB)."""
    import numpy as _np
    fb = self.robot.floating_base

    # Floating-base root (jid 0) per-DoF motion-subspace metadata. The root's S
    # is a 6x6 motion subspace (pinocchio free-flyer: the [[0,I3],[I3,0]] block
    # swap), NOT the identity, so each root DoF ii maps to spatial axis k = the
    # nonzero of S column ii (with that entry's sign). mxS(S[:,ii], v) reduces to
    # sign * mx<k>(v) and dv_dqd[:,ii] += S[:,ii] sets entry [k, ii]. Extracted
    # from the actual S matrix (robust to convention), used by the float-root
    # forward/backward emit below.
    root_dof_axes = None
    if fb:
        _S0 = _np.asarray(self.robot.get_S_by_id(0), dtype=float).reshape(6, 6)
        root_dof_axes = []
        for _ii in range(6):
            _col = _S0[:, _ii]
            _nz = _np.nonzero(_np.abs(_col) > 1e-12)[0]
            assert len(_nz) == 1, \
                "floating root S column %d is not a single signed axis: %r" % (_ii, _col)
            _k = int(_nz[0]); _sgn = float(_col[_k])
            root_dof_axes.append((_k, _sgn))

    bw = 6 * nv * NB  # one dense buffer (6 rows x nv cols x NB bodies)
    off_dv_dq  = 0
    off_da_dq  = off_dv_dq  + bw
    off_df_dq  = off_da_dq  + bw
    off_dv_dqd = off_df_dq  + bw
    off_da_dqd = off_dv_dqd + bw
    off_df_dqd = off_da_dqd + bw
    off_iv     = off_df_dqd + bw          # Iv per body: 6*NB
    off_scr    = off_iv + 6 * NB          # scratch 6-vectors (a few)

    def cell(base, ind, c):
        # &buffer[base] element column c of body ind (6-vector)
        return base + ind * (6 * nv) + 6 * c

    self.gen_add_code_line("// === mimic ID-gradient (dense serial reduced-space fold) ===")
    self.gen_add_code_line("// zero the dense fwd buffers + output")
    self.gen_add_parallel_loop("i", str(off_iv))
    self.gen_add_code_line("s_temp[i] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_parallel_loop("i", str(2 * nv * nv))
    self.gen_add_code_line("s_dc_du[i] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # ---- forward pass (serial over bodies, root first) ----
    # WIN B: the body-walk MUST stay serial-ordered (each body reads its parent's
    # dv/da gradient buffers written in a prior iteration), so EVERY lane runs the
    # same `for ind` loop. But the independent work WITHIN a body — the per-column
    # `for c` matvecs (one lane per gradient column) — is fanned via
    # gen_add_parallel_loop("c", nv). The cross-body v-slot folds (`+=` into a
    # single reduced column at slot idx) stay on ONE lane (gen_add_serial_ops) so
    # a single reduced-column accumulation is never split across lanes — keeping
    # the fold bit-exact. __syncthreads() at each subsection boundary makes the
    # parallel writes visible to the next (per-c or single-column) reader.
    # Per-lane scratch (s_iv6/s_mtmp/s_ftmp/s_Svec) becomes per-section stack
    # locals (declared inside each loop/serial block below).
    for ind in range(NB):
        parent = self.robot.get_parent_id(ind)
        # The floating-base root (jid 0) is the only multi-DoF body: 6 v-slots
        # (0..5) and S = I_6, so its own-DoF gradient terms LOOP over the 6 root
        # DoFs instead of using a scalar (idx, s_ind, s_sign). It is never mimic
        # (alpha == 1). Non-root bodies stay on the proven scalar path.
        is_float_root = fb and ind == 0
        alpha = self._alpha_for_jid(ind)
        # SPHERICAL (Tier-C): a mid-chain (or fixed-root) 3-DoF ball joint. Its S
        # is the 6x3 ANGULAR-IDENTITY (rows 0,1,2 = I3), so each of its 3 columns k
        # is a CARDINAL angular axis (mxS(S[:,k], .) == mx<k>(.), sign +1) over the
        # contiguous v-block vblk = get_joint_index_v(ind). It is never mimic
        # (alpha == 1). All single-DoF `idx` folds below become 3-column folds and
        # the da += mxS(dv)*qd term sums over the 3 columns (the multi-column term).
        is_spherical = (not is_float_root) and self.robot.joint_is_spherical(ind)
        is_skew = (not is_float_root) and (not is_spherical) and (not self.robot.S_is_cardinal_by_id(ind))
        S_vec = None if (is_float_root or is_spherical) else [float(v) for v in self.robot._get_flat_S_by_id(ind)]
        vblk = None
        if is_float_root:
            idx = None; s_ind = None; s_sign = None
        elif is_spherical:
            # 3-wide v-block; §1e: ALWAYS via get_joint_index_v (downstream-shifted).
            vblk = self.robot.get_joint_index_v(ind)
            assert isinstance(vblk, (list, tuple)) and len(vblk) == 3, \
                "spherical body %d expected a 3-wide v-block, got %r" % (ind, vblk)
            # Robustness: confirm S is the angular-identity (col k -> axis k, +1).
            _Ssph = _np.asarray(self.robot.get_S_by_id(ind), dtype=float).reshape(6, 3)
            assert _np.allclose(_Ssph, _np.vstack([_np.eye(3), _np.zeros((3, 3))])), \
                "spherical body %d S is not the angular-identity: %r" % (ind, _Ssph)
            idx = None; s_ind = None; s_sign = None
        elif is_skew:
            idx = self._v_slot_cpp(ind); s_ind = None; s_sign = None
        else:
            idx = self._v_slot_cpp(ind)
            s_ind = self.robot.get_S_index_by_id(ind)
            s_sign = float(self.robot.get_S_sign_by_id(ind))
        v_ind = 6 * ind                 # s_vaf v
        a_ind = 6 * NB + 6 * ind        # s_vaf a
        Xoff = 36 * ind                 # X[ind] col-major in s_XImats
        Ioff = 36 * NB + 36 * ind       # I[ind]
        if is_float_root:
            self.gen_add_code_line("// --- body 0 (FLOATING ROOT, v-slots 0..5, S=I6) ---")
        elif is_spherical:
            self.gen_add_code_line("// --- body " + str(ind) + " (SPHERICAL, v-slots " +
                                   str(list(vblk)) + ", S=angular-identity) ---")
        else:
            self.gen_add_code_line("// --- body " + str(ind) + " (v-slot " + str(idx) +
                                   ", alpha=" + repr(alpha) + ", S_ind=" + str(s_ind) +
                                   ", S_sign=" + repr(s_sign) + ") ---")
        self.gen_add_code_line("{", True)  # per-body scope (avoid local redeclare)

        # Iv = I[ind] * v[ind] (one lane writes the per-body 6-vector; read by df below)
        self.gen_add_serial_ops()
        self.gen_add_code_line("for (int r = 0; r < 6; r++) { T iv = static_cast<T>(0);")
        self.gen_add_code_line("  for (int p = 0; p < 6; p++) iv += s_XImats[" + str(Ioff) + " + r + 6*p] * s_vaf[" + str(v_ind) + " + p];")
        self.gen_add_code_line("  s_temp[" + str(off_iv + 6*ind) + " + r] = iv; }")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

        if parent != -1:
            p_dv_dq  = cell(off_dv_dq,  parent, 0)
            p_da_dq  = cell(off_da_dq,  parent, 0)
            p_dv_dqd = cell(off_dv_dqd, parent, 0)
            p_da_dqd = cell(off_da_dqd, parent, 0)
            c_dv_dq  = cell(off_dv_dq,  ind, 0)
            c_da_dq  = cell(off_da_dq,  ind, 0)
            c_dv_dqd = cell(off_dv_dqd, ind, 0)
            c_da_dqd = cell(off_da_dqd, ind, 0)
            # dv_du[ind] = X[ind] * dv_du[parent] ; da_du[ind] = X[ind]*da_du[parent]
            # WIN B: one lane per gradient column c (independent outputs; the per-
            # element 6-wide reduction over p is intact, so bit-exact).
            self.gen_add_parallel_loop("c", str(nv))
            self.gen_add_code_line("for (int r = 0; r < 6; r++) {", True)
            self.gen_add_code_line("T acc_vq=static_cast<T>(0), acc_aq=static_cast<T>(0), acc_vqd=static_cast<T>(0), acc_aqd=static_cast<T>(0);")
            self.gen_add_code_line("for (int p = 0; p < 6; p++) {", True)
            self.gen_add_code_line("T xrp = s_XImats[" + str(Xoff) + " + r + 6*p];")
            self.gen_add_code_line("acc_vq  += xrp * s_temp[" + str(p_dv_dq)  + " + 6*c + p];")
            self.gen_add_code_line("acc_aq  += xrp * s_temp[" + str(p_da_dq)  + " + 6*c + p];")
            self.gen_add_code_line("acc_vqd += xrp * s_temp[" + str(p_dv_dqd) + " + 6*c + p];")
            self.gen_add_code_line("acc_aqd += xrp * s_temp[" + str(p_da_dqd) + " + 6*c + p];")
            self.gen_add_end_control_flow()
            self.gen_add_code_line("s_temp[" + str(c_dv_dq)  + " + 6*c + r] = acc_vq;")
            self.gen_add_code_line("s_temp[" + str(c_da_dq)  + " + 6*c + r] = acc_aq;")
            self.gen_add_code_line("s_temp[" + str(c_dv_dqd) + " + 6*c + r] = acc_vqd;")
            self.gen_add_code_line("s_temp[" + str(c_da_dqd) + " + 6*c + r] = acc_aqd;")
            self.gen_add_end_control_flow()
            self.gen_add_end_control_flow()
            self.gen_add_sync()  # the X*parent cols feed the single-column += folds below

            # dv_dq[:,idx,ind]  += alpha * mxS(S, X*v_parent)  = alpha*s_sign*mx_Sind(X v_parent)
            # X*v_parent: dv contribution uses v[parent]. Single reduced-column (idx)
            # fold -> ONE lane (cross-body v-slot accumulate stays unsplit, bit-exact).
            # SPHERICAL: 3 single-column folds, one per v-block slot vblk[k], using
            # mx<k> (S col k = angular axis k, sign +1). Each is a single-lane write.
            self.gen_add_serial_ops()
            self.gen_add_code_line("// dv_dq[:,idx] += alpha*mxS(S, X*v_parent); dv_dqd[:,idx] += alpha*S")
            self.gen_add_code_line("T s_mtmp[6];")
            self.gen_add_code_line("for (int r = 0; r < 6; r++) { s_mtmp[r] = static_cast<T>(0);")
            self.gen_add_code_line("  for (int p = 0; p < 6; p++) s_mtmp[r] += s_XImats[" + str(Xoff) + " + r + 6*p] * s_vaf[" + str(6*parent) + " + p]; }")
            # mx<s_ind>_peq_scaled into dv_dq[:,idx,ind]. mxS(S,.) carries the
            # joint sign (S = s_sign*e_{s_ind}); mx<ind>_peq_scaled only applies
            # the UNIT-axis column, so fold s_sign into the scale (alpha*s_sign).
            if is_spherical:
                for k in range(3):
                    self.gen_add_code_line("mx" + str(k) + "_peq_scaled<T>(&s_temp[" + str(cell(off_dv_dq, ind, 0)) + " + 6*" + str(vblk[k]) + "], s_mtmp, static_cast<T>(1.0));")
            elif is_skew:
                self.gen_add_code_line("{ const T S_skew[6] = " + _idg_Svec_cpp(S_vec) + "; mxS_general_peq_scaled<T>(&s_temp[" + str(cell(off_dv_dq, ind, 0)) + " + 6*" + str(idx) + "], s_mtmp, S_skew, static_cast<T>(" + repr(alpha) + ")); }")
            else:
                self.gen_add_code_line("mx" + str(s_ind) + "_peq_scaled<T>(&s_temp[" + str(cell(off_dv_dq, ind, 0)) + " + 6*" + str(idx) + "], s_mtmp, static_cast<T>(" + repr(alpha * s_sign) + "));")
            self.gen_add_end_control_flow()
            self.gen_add_sync()  # dv_dq[:,idx] now visible to the da += mxS(dv) reader below

        if is_float_root:
            # ===== FLOATING ROOT own-DoF terms (6-DoF motion subspace S) =====
            # Mirrors rnea_grad_fpass_dq/dqd's `for ii in range(len(idx))` over
            # the 6 root DoFs. S is the free-flyer 6x6 subspace (NOT identity):
            # root DoF ii maps to spatial axis k=root_dof_axes[ii][0] with sign
            # sgn, so mxS(S[:,ii], v) == sgn * mx<k>(v) and dv_dqd[:,ii] += S[:,ii]
            # sets entry [k, ii]. The root is never mimic (alpha == 1). dv_dq
            # stays 0 (no parent + own term parent-gated), so the da += mxS(dv_dq)
            # *qd term is identically 0 for dq.
            # dv_dqd[:,ii,root] += S[:,ii]  (entry [k, ii] = sign). Single-column
            # (per-DoF) folds -> ONE lane (root v-slots, bit-exact accumulate).
            self.gen_add_serial_ops()
            self.gen_add_code_line("// FLOATING ROOT: dv_dqd[:,ii,root] += S[:,ii]")
            for ii in range(6):
                k, sgn = root_dof_axes[ii]
                self.gen_add_code_line("s_temp[" + str(cell(off_dv_dqd, ind, ii)) + " + " + str(k) + "] += static_cast<T>(" + repr(sgn) + ");")
            self.gen_add_end_control_flow()
            self.gen_add_sync()  # dv_dqd[:,ii] visible to the da_dqd += mxS(dv_dqd) reader
            # da_dqd[:,c,root] += sum_ii sgn_ii * mx<k_ii>(dv_dqd[:,c,root]) * qd[ii]
            #   (da_dq term is 0 because dv_dq[:,c,root] == 0). WIN B: one lane per c.
            self.gen_add_code_line("// FLOATING ROOT: da_dqd[:,c] += sum_ii sgn*mx_k(dv_dqd[:,c]) * qd[ii]")
            self.gen_add_parallel_loop("c", str(nv))
            for ii in range(6):
                k, sgn = root_dof_axes[ii]
                self.gen_add_code_line("mx" + str(k) + "_peq_scaled<T>(&s_temp[" + str(cell(off_da_dqd, ind, 0)) + " + 6*c], &s_temp[" + str(cell(off_dv_dqd, ind, 0)) + " + 6*c], static_cast<T>(" + repr(sgn) + ") * s_qd[" + str(ii) + "]);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()  # da_dqd[:,c] for all c before the per-DoF (col ii) folds below
            # da_dq[:,ii,root] += mxS(S[:,ii], root_gravity) = sgn*mx<k>(root_grav)
            #   root_gravity = inv(X)*gravity_vec. inv(X) motion-transform block
            #   form [[R^T,0],[C,R^T]] => col 5 is [0,0,0, R^T[:,2]] = [0,0,0,
            #   X[2,0], X[2,1], X[2,2]] (col-major X[2,k]=s_XImats[Xoff+6*k+2]);
            #   gravity_vec = [0,0,0,0,0,gravity]. Single-column (col ii) -> ONE lane.
            self.gen_add_serial_ops()
            self.gen_add_code_line("T s_mtmp[6];")
            self.gen_add_code_line("// FLOATING ROOT: da_dq[:,ii] += sgn*mx_k(inv(X)*gravity_vec)")
            self.gen_add_code_line("s_mtmp[0] = static_cast<T>(0); s_mtmp[1] = static_cast<T>(0); s_mtmp[2] = static_cast<T>(0);")
            self.gen_add_code_line("s_mtmp[3] = -s_XImats[" + str(Xoff + 2) + "] * gravity;")
            self.gen_add_code_line("s_mtmp[4] = -s_XImats[" + str(Xoff + 8) + "] * gravity;")
            self.gen_add_code_line("s_mtmp[5] = -s_XImats[" + str(Xoff + 14) + "] * gravity;")
            for ii in range(6):
                k, sgn = root_dof_axes[ii]
                self.gen_add_code_line("mx" + str(k) + "_peq_scaled<T>(&s_temp[" + str(cell(off_da_dq, ind, ii)) + "], s_mtmp, static_cast<T>(" + repr(sgn) + "));")
            # da_dqd[:,ii,root] += mxS(S[:,ii], v[root]) = sgn*mx<k>(v[root])
            self.gen_add_code_line("// FLOATING ROOT: da_dqd[:,ii] += sgn*mx_k(v[root])")
            for ii in range(6):
                k, sgn = root_dof_axes[ii]
                self.gen_add_code_line("mx" + str(k) + "_peq_scaled<T>(&s_temp[" + str(cell(off_da_dqd, ind, ii)) + "], &s_vaf[" + str(v_ind) + "], static_cast<T>(" + repr(sgn) + "));")
            self.gen_add_end_control_flow()
            self.gen_add_sync()  # all da cols (incl per-DoF ii) before df reads them
        elif is_spherical:
            # ===== SPHERICAL own-DoF terms (3-DoF angular-identity S) =====
            # Mirrors the oracle's multi-column form over the 3 v-block slots
            # vblk[0..2]: S col k = angular axis k (mx<k>, sign +1), alpha == 1.
            # dv_dqd[:,vblk[k],ind] += S[:,k]  sets entry [k, vblk[k]] = 1 (all
            # bodies incl. a fixed-base root spherical). Single-lane writes.
            self.gen_add_serial_ops()
            self.gen_add_code_line("// SPHERICAL: dv_dqd[:,vblk[k]] += S[:,k] (entry [k, vblk[k]] = 1)")
            for k in range(3):
                self.gen_add_code_line("s_temp[" + str(cell(off_dv_dqd, ind, 0)) + " + 6*" + str(vblk[k]) + " + " + str(k) + "] += static_cast<T>(1.0);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()  # dv_dqd[:,vblk[k]] visible to the da += mxS(dv_dqd) reader

            # da_du[:,c,ind] += sum_k mx<k>(dv_du[:,c,ind]) * qd[vblk[k]]  (BOTH dq
            # and dqd). For a root spherical dv_dq[:,c] is all-zero so its dq term
            # vanishes — emitting it uniformly is correct (no-op for the root).
            # WIN B: one lane per gradient column c.
            self.gen_add_code_line("// SPHERICAL: da_du[:,c] += sum_k mx_k(dv_du[:,c]) * qd[vblk[k]]")
            self.gen_add_parallel_loop("c", str(nv))
            for k in range(3):
                self.gen_add_code_line("mx" + str(k) + "_peq_scaled<T>(&s_temp[" + str(cell(off_da_dq, ind, 0)) + " + 6*c], &s_temp[" + str(cell(off_dv_dq, ind, 0)) + " + 6*c], s_qd[" + str(vblk[k]) + "]);")
                self.gen_add_code_line("mx" + str(k) + "_peq_scaled<T>(&s_temp[" + str(cell(off_da_dqd, ind, 0)) + " + 6*c], &s_temp[" + str(cell(off_dv_dqd, ind, 0)) + " + 6*c], s_qd[" + str(vblk[k]) + "]);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()  # all da cols before the single-column (vblk) da folds below

            # da_dq[:,vblk[k],ind]  += mx<k>(X*a_parent)  (or X*gravity at a fixed root)
            # da_dqd[:,vblk[k],ind] += mx<k>(v[ind]). Single-lane per-slot folds.
            self.gen_add_serial_ops()
            self.gen_add_code_line("T s_mtmp[6];")
            self.gen_add_code_line("// SPHERICAL: da_dq[:,vblk[k]] += mx_k(X*a_parent); da_dqd[:,vblk[k]] += mx_k(v[ind])")
            if parent != -1:
                self.gen_add_code_line("for (int r = 0; r < 6; r++) { s_mtmp[r] = static_cast<T>(0);")
                self.gen_add_code_line("  for (int p = 0; p < 6; p++) s_mtmp[r] += s_XImats[" + str(Xoff) + " + r + 6*p] * s_vaf[" + str(6*NB + 6*parent) + " + p]; }")
            else:
                # fixed-base root spherical: base accel is PURE gravity (X*gravity,
                # col 5 of X scaled by gravity), NOT the body's own a (which would
                # carry S*qdd under use_qdd_input and corrupt fd_gradient).
                self.gen_add_code_line("for (int r = 0; r < 6; r++) s_mtmp[r] = -s_XImats[" + str(Xoff) + " + 30 + r] * gravity;")
            for k in range(3):
                self.gen_add_code_line("mx" + str(k) + "_peq_scaled<T>(&s_temp[" + str(cell(off_da_dq, ind, 0)) + " + 6*" + str(vblk[k]) + "], s_mtmp, static_cast<T>(1.0));")
            for k in range(3):
                self.gen_add_code_line("mx" + str(k) + "_peq_scaled<T>(&s_temp[" + str(cell(off_da_dqd, ind, 0)) + " + 6*" + str(vblk[k]) + "], &s_vaf[" + str(v_ind) + "], static_cast<T>(1.0));")
            self.gen_add_end_control_flow()
            self.gen_add_sync()  # all da cols (incl vblk cols) before df reads them
        else:
            # dv_dqd[:,idx,ind] += alpha*S  (S = s_sign*e_{s_ind}). NOTE: the oracle
            # adds this for EVERY body including the root (it sits OUTSIDE the
            # parent!=-1 guard in rnea_grad_fpass_dqd) — the joint's own velocity
            # subspace contributes to dv/dqd regardless of having a parent.
            # Single reduced-column (idx) fold -> ONE lane (bit-exact accumulate).
            self.gen_add_serial_ops()
            self.gen_add_code_line("// dv_dqd[:,idx] += alpha*S (all bodies incl. root)")
            if is_skew:
                self.gen_add_code_line("{ const T S_skew[6] = " + _idg_Svec_cpp(S_vec) + "; for (int r = 0; r < 6; r++) s_temp[" + str(cell(off_dv_dqd, ind, 0)) + " + 6*" + str(idx) + " + r] += static_cast<T>(" + repr(alpha) + ") * S_skew[r]; }")
            else:
                self.gen_add_code_line("s_temp[" + str(cell(off_dv_dqd, ind, 0)) + " + 6*" + str(idx) + " + " + str(s_ind) + "] += static_cast<T>(" + repr(alpha * s_sign) + ");")
            self.gen_add_end_control_flow()
            self.gen_add_sync()  # dv_dqd[:,idx] visible to the da += mxS(dv_dqd) reader below

            # da_du[:,c,ind] += mxS(S, dv_du[:,c,ind], alpha*qd[idx])   for every column c
            # WIN B: one lane per gradient column c (reads dv_du[:,c] incl col idx
            # written above; each lane owns a distinct column -> no cross-lane race).
            self.gen_add_code_line("// da_du[:,c] += mxS(S, dv_du[:,c], alpha*qd[idx])")
            self.gen_add_parallel_loop("c", str(nv))
            self.gen_add_code_line("T qd_a = static_cast<T>(" + repr(alpha) + ") * s_qd[" + str(idx) + "];")
            if is_skew:
                self.gen_add_code_line("{ const T S_skew[6] = " + _idg_Svec_cpp(S_vec) + ";")
                self.gen_add_code_line("  mxS_general_peq_scaled<T>(&s_temp[" + str(cell(off_da_dq, ind, 0)) + " + 6*c], &s_temp[" + str(cell(off_dv_dq, ind, 0)) + " + 6*c], S_skew, qd_a);")
                self.gen_add_code_line("  mxS_general_peq_scaled<T>(&s_temp[" + str(cell(off_da_dqd, ind, 0)) + " + 6*c], &s_temp[" + str(cell(off_dv_dqd, ind, 0)) + " + 6*c], S_skew, qd_a); }")
            else:
                self.gen_add_code_line("mx" + str(s_ind) + "_peq_scaled<T>(&s_temp[" + str(cell(off_da_dq, ind, 0)) + " + 6*c], &s_temp[" + str(cell(off_dv_dq, ind, 0)) + " + 6*c], static_cast<T>(" + repr(s_sign) + ") * qd_a);")
                self.gen_add_code_line("mx" + str(s_ind) + "_peq_scaled<T>(&s_temp[" + str(cell(off_da_dqd, ind, 0)) + " + 6*c], &s_temp[" + str(cell(off_dv_dqd, ind, 0)) + " + 6*c], static_cast<T>(" + repr(s_sign) + ") * qd_a);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()  # all da cols before the single-column (idx) da folds below

            # da_dq[:,idx,ind] += alpha*mxS(S, X*a_parent or root_gravity)
            # da_dqd[:,idx,ind] += alpha*mxS(S, v[ind]). Single reduced-column (idx)
            # folds -> ONE lane (bit-exact accumulate into one column).
            self.gen_add_serial_ops()
            self.gen_add_code_line("T s_mtmp[6];")
            self.gen_add_code_line("// da_dq[:,idx] += alpha*mxS(S, X*a_parent); da_dqd[:,idx] += alpha*mxS(S, v[ind])")
            if is_skew:
                self.gen_add_code_line("const T S_skew[6] = " + _idg_Svec_cpp(S_vec) + ";")
            if parent != -1:
                self.gen_add_code_line("for (int r = 0; r < 6; r++) { s_mtmp[r] = static_cast<T>(0);")
                self.gen_add_code_line("  for (int p = 0; p < 6; p++) s_mtmp[r] += s_XImats[" + str(Xoff) + " + r + 6*p] * s_vaf[" + str(6*NB + 6*parent) + " + p]; }")
                if is_skew:
                    self.gen_add_code_line("mxS_general_peq_scaled<T>(&s_temp[" + str(cell(off_da_dq, ind, 0)) + " + 6*" + str(idx) + "], s_mtmp, S_skew, static_cast<T>(" + repr(alpha) + "));")
                else:
                    self.gen_add_code_line("mx" + str(s_ind) + "_peq_scaled<T>(&s_temp[" + str(cell(off_da_dq, ind, 0)) + " + 6*" + str(idx) + "], s_mtmp, static_cast<T>(" + repr(alpha * s_sign) + "));")
            else:
                # fixed-base root: the base's accel is PURE gravity (NOT the body's
                # own a, which also carries S*qdd when use_qdd_input — that would
                # corrupt forward_dynamics_gradient). X*gravity is column 5 of X scaled by `gravity`:
                #   (X*gravity)[r] = s_XImats[36*root + 30 + r] * gravity (col5=+30).
                self.gen_add_code_line("for (int r = 0; r < 6; r++) s_mtmp[r] = -s_XImats[" + str(Xoff) + " + 30 + r] * gravity;")
                if is_skew:
                    self.gen_add_code_line("mxS_general_peq_scaled<T>(&s_temp[" + str(cell(off_da_dq, ind, 0)) + " + 6*" + str(idx) + "], s_mtmp, S_skew, static_cast<T>(" + repr(alpha) + "));")
                else:
                    self.gen_add_code_line("mx" + str(s_ind) + "_peq_scaled<T>(&s_temp[" + str(cell(off_da_dq, ind, 0)) + " + 6*" + str(idx) + "], s_mtmp, static_cast<T>(" + repr(alpha * s_sign) + "));")
            # da_dqd[:,idx,ind] += alpha*mxS(S, v[ind])
            if is_skew:
                self.gen_add_code_line("mxS_general_peq_scaled<T>(&s_temp[" + str(cell(off_da_dqd, ind, 0)) + " + 6*" + str(idx) + "], &s_vaf[" + str(v_ind) + "], S_skew, static_cast<T>(" + repr(alpha) + "));")
            else:
                self.gen_add_code_line("mx" + str(s_ind) + "_peq_scaled<T>(&s_temp[" + str(cell(off_da_dqd, ind, 0)) + " + 6*" + str(idx) + "], &s_vaf[" + str(v_ind) + "], static_cast<T>(" + repr(alpha * s_sign) + "));")
            self.gen_add_end_control_flow()
            self.gen_add_sync()  # all da cols (incl col idx) before df reads them

        # df_du[:,:,ind] = I*da_du + fxv(dv_du, Iv) + fxv(v, I*dv_du)
        # WIN B: one lane per gradient column c (df[:,c] reads da[:,c], dv[:,c], Iv
        # — all finalized + synced above; outputs are disjoint per c, bit-exact).
        # Per-lane scratch s_ftmp/s_mtmp are stack locals inside the loop.
        self.gen_add_code_line("// df_du[:,c] = I*da_du[:,c] + fx(dv_du[:,c])*Iv + fx(v)*I*dv_du[:,c]")
        self.gen_add_parallel_loop("c", str(nv))
        self.gen_add_code_line("T s_ftmp[6]; T s_mtmp[6];")
        for (dabuf, dfbuf, dvbuf) in [(off_da_dq, off_df_dq, off_dv_dq), (off_da_dqd, off_df_dqd, off_dv_dqd)]:
            # I*da_du
            self.gen_add_code_line("for (int r = 0; r < 6; r++) { T acc=static_cast<T>(0);")
            self.gen_add_code_line("  for (int p = 0; p < 6; p++) acc += s_XImats[" + str(Ioff) + " + r + 6*p] * s_temp[" + str(cell(dabuf, ind, 0)) + " + 6*c + p];")
            self.gen_add_code_line("  s_temp[" + str(cell(dfbuf, ind, 0)) + " + 6*c + r] = acc; }")
            # fxv(dv_du[:,c], Iv)
            self.gen_add_code_line("fx_times_v<T>(s_ftmp, &s_temp[" + str(cell(dvbuf, ind, 0)) + " + 6*c], &s_temp[" + str(off_iv + 6*ind) + "]);")
            self.gen_add_code_line("for (int r = 0; r < 6; r++) s_temp[" + str(cell(dfbuf, ind, 0)) + " + 6*c + r] += s_ftmp[r];")
            # I*dv_du[:,c]
            self.gen_add_code_line("for (int r = 0; r < 6; r++) { T acc=static_cast<T>(0);")
            self.gen_add_code_line("  for (int p = 0; p < 6; p++) acc += s_XImats[" + str(Ioff) + " + r + 6*p] * s_temp[" + str(cell(dvbuf, ind, 0)) + " + 6*c + p];")
            self.gen_add_code_line("  s_mtmp[r] = acc; }")
            # fxv(v[ind], I*dv_du[:,c])
            self.gen_add_code_line("fx_times_v<T>(s_ftmp, &s_vaf[" + str(v_ind) + "], s_mtmp);")
            self.gen_add_code_line("for (int r = 0; r < 6; r++) s_temp[" + str(cell(dfbuf, ind, 0)) + " + 6*c + r] += s_ftmp[r];")
        self.gen_add_end_control_flow()  # end for c (df_du)
        self.gen_add_sync()  # df[:,:,ind] complete before next body (child) reads it
        self.gen_add_end_control_flow()  # end per-body scope
    self.gen_add_sync()

    # ---- backward pass (serial, deepest first) ----
    # dc_du[idx,:] += alpha * S^T * df_du[:,:,ind]
    # df_du[:,idx,parent] += alpha * (X^T * fxS(S, f[ind]))   [dq only]
    # df_du[:,:,parent]  += X^T * df_du[:,:,ind]
    # WIN B: the body-walk stays serial-ordered (deepest first; a parent reads
    # df_du buffers its children wrote), so EVERY lane runs the same `for ind`.
    # The per-column `for c` work (the dc_du fold + the X^T df propagation) fans
    # one lane per column c; the single-column df_dq[:,idx,parent] fold stays on
    # ONE lane. The dc_du v-slot reduction is a cross-BODY accumulate at a fixed
    # ROW idx, column c — each (c) hit is on one lane and the cross-body sum is
    # serialized by the body-walk + inter-body __syncthreads, so it stays
    # bit-exact (a single reduced entry is never split across lanes).
    for ind in range(NB - 1, -1, -1):
        parent = self.robot.get_parent_id(ind)
        is_float_root = fb and ind == 0
        alpha = self._alpha_for_jid(ind)
        is_spherical = (not is_float_root) and self.robot.joint_is_spherical(ind)
        is_skew = (not is_float_root) and (not is_spherical) and (not self.robot.S_is_cardinal_by_id(ind))
        S_vec = None if (is_float_root or is_spherical) else [float(v) for v in self.robot._get_flat_S_by_id(ind)]
        vblk = None
        if is_float_root:
            idx = None; s_ind = None; s_sign = None
        elif is_spherical:
            vblk = self.robot.get_joint_index_v(ind)  # 3-wide v-block (§1e)
            idx = None; s_ind = None; s_sign = None
        elif is_skew:
            idx = self._v_slot_cpp(ind); s_ind = None; s_sign = None
        else:
            idx = self._v_slot_cpp(ind)
            s_ind = self.robot.get_S_index_by_id(ind)
            s_sign = float(self.robot.get_S_sign_by_id(ind))
        Xoff = 36 * ind
        f_ind = 12 * NB + 6 * ind
        if is_float_root:
            # FLOATING ROOT: dc_du[ii,:] += (S^T df_du)[ii,:]; S^T row ii = S col
            # ii, a single signed axis k=root_dof_axes[ii], so this reduces to
            #   dc_du[ii, c] += sgn * df_du[k, c, root]   for ii in 0..5 (alpha=1)
            self.gen_add_code_line("// --- bpass body 0 (FLOATING ROOT, v-slots 0..5, S^T df_du) ---")
            # WIN B: one lane per gradient column c (each c writes distinct dc_du
            # entries; the cross-body dc_du accumulate is serialized by the walk).
            self.gen_add_parallel_loop("c", str(nv))
            for ii in range(6):
                k, sgn = root_dof_axes[ii]
                self.gen_add_code_line("s_dc_du[c*" + str(nv) + " + " + str(ii) + "] += static_cast<T>(" + repr(sgn) + ") * s_temp[" + str(cell(off_df_dq, ind, 0)) + " + 6*c + " + str(k) + "];")
                self.gen_add_code_line("s_dc_du[" + str(nv*nv) + " + c*" + str(nv) + " + " + str(ii) + "] += static_cast<T>(" + repr(sgn) + ") * s_temp[" + str(cell(off_df_dqd, ind, 0)) + " + 6*c + " + str(k) + "];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()  # finish this body's dc_du fold before the (earlier) walk ends
            # root has no parent -> no df propagation; backward pass done for root.
            continue
        if is_spherical:
            # SPHERICAL backward: dc_du[vblk[k], c] += (S^T df_du)[k, c] = df_du[k,
            # c, ind] (S^T row k = e_k^T, alpha == 1) for k in 0..2; then propagate
            # df to the parent (X^T df, plus the per-column X^T fxS(S[:,k], f) term
            # scattered into the parent's vblk[k] columns).
            self.gen_add_code_line("// --- bpass body " + str(ind) + " (SPHERICAL, v-slots " + str(list(vblk)) + ", S^T df_du) ---")
            # WIN B: one lane per gradient column c (distinct dc_du entries; the
            # cross-body accumulate at rows vblk[k] is serialized by the walk).
            self.gen_add_parallel_loop("c", str(nv))
            for k in range(3):
                self.gen_add_code_line("s_dc_du[c*" + str(nv) + " + " + str(vblk[k]) + "] += s_temp[" + str(cell(off_df_dq, ind, 0)) + " + 6*c + " + str(k) + "];")
                self.gen_add_code_line("s_dc_du[" + str(nv*nv) + " + c*" + str(nv) + " + " + str(vblk[k]) + "] += s_temp[" + str(cell(off_df_dqd, ind, 0)) + " + 6*c + " + str(k) + "];")
            self.gen_add_end_control_flow()
            if parent != -1:
                self.gen_add_sync()  # dc_du fold reads df[ind]; df[parent] += below must wait
                # df_dq[:,vblk[k],parent] += X^T * fxS(S[:,k], f[ind]) for each k.
                # fxS(S[:,k], f) = fx_times_v(e_k, f); S col k = angular axis k.
                # Single-lane per-slot folds (cross-body accumulate stays unsplit).
                self.gen_add_serial_ops()
                self.gen_add_code_line("T s_fxs[6]; T s_xtfxs[6]; T s_Svec[6];")
                for k in range(3):
                    self.gen_add_code_line("for (int r = 0; r < 6; r++) s_Svec[r] = static_cast<T>(0);")
                    self.gen_add_code_line("s_Svec[" + str(k) + "] = static_cast<T>(1.0);")
                    self.gen_add_code_line("fx_times_v<T>(s_fxs, s_Svec, &s_vaf[" + str(f_ind) + "]);")
                    # X^T * s_fxs : (X^T)[r,p] = X[p,r] = s_XImats[Xoff + p + 6*r]
                    self.gen_add_code_line("for (int r = 0; r < 6; r++) { s_xtfxs[r] = static_cast<T>(0);")
                    self.gen_add_code_line("  for (int p = 0; p < 6; p++) s_xtfxs[r] += s_XImats[" + str(Xoff) + " + p + 6*r] * s_fxs[p]; }")
                    self.gen_add_code_line("for (int r = 0; r < 6; r++) s_temp[" + str(cell(off_df_dq, parent, 0)) + " + 6*" + str(vblk[k]) + " + r] += s_xtfxs[r];")
                self.gen_add_end_control_flow()
                # The single-column df_dq[:,vblk[k],parent] += (above) must land
                # BEFORE the X^T propagation below touches those same parent columns.
                self.gen_add_sync()
                # df_du[:,:,parent] += X^T * df_du[:,:,ind]   (both dq and dqd).
                self.gen_add_parallel_loop("c", str(nv))
                self.gen_add_code_line("for (int r = 0; r < 6; r++) {", True)
                self.gen_add_code_line("T acc_q=static_cast<T>(0), acc_qd=static_cast<T>(0);")
                self.gen_add_code_line("for (int p = 0; p < 6; p++) {", True)
                self.gen_add_code_line("T xtr = s_XImats[" + str(Xoff) + " + p + 6*r];")
                self.gen_add_code_line("acc_q  += xtr * s_temp[" + str(cell(off_df_dq,  ind, 0)) + " + 6*c + p];")
                self.gen_add_code_line("acc_qd += xtr * s_temp[" + str(cell(off_df_dqd, ind, 0)) + " + 6*c + p];")
                self.gen_add_end_control_flow()
                self.gen_add_code_line("s_temp[" + str(cell(off_df_dq,  parent, 0)) + " + 6*c + r] += acc_q;")
                self.gen_add_code_line("s_temp[" + str(cell(off_df_dqd, parent, 0)) + " + 6*c + r] += acc_qd;")
                self.gen_add_end_control_flow()
                self.gen_add_end_control_flow()
            self.gen_add_sync()  # this body's writes visible before the next (shallower) body
            continue
        self.gen_add_code_line("// --- bpass body " + str(ind) + " (v-slot " + str(idx) + ") ---")
        # dc_dq[idx, c]  += alpha * s_sign * df_dq[s_ind, c, ind]
        # dc_dqd[idx, c] += alpha * s_sign * df_dqd[s_ind, c, ind]
        # WIN B: one lane per gradient column c. Each c writes a distinct dc_du
        # entry; the cross-body accumulate at row idx is serialized by the walk.
        self.gen_add_parallel_loop("c", str(nv))
        if is_skew:
            # dc[idx,c] += alpha * S^T df[:,c] = alpha * sum_r S[r]*df[r,c]
            dq_terms = " + ".join("static_cast<T>(" + repr(alpha * S_vec[r]) + ") * s_temp[" + str(cell(off_df_dq, ind, 0)) + " + 6*c + " + str(r) + "]" for r in range(6) if S_vec[r] != 0.0)
            qd_terms = " + ".join("static_cast<T>(" + repr(alpha * S_vec[r]) + ") * s_temp[" + str(cell(off_df_dqd, ind, 0)) + " + 6*c + " + str(r) + "]" for r in range(6) if S_vec[r] != 0.0)
            self.gen_add_code_line("s_dc_du[c*" + str(nv) + " + " + str(idx) + "] += " + (dq_terms if dq_terms else "static_cast<T>(0)") + ";")
            self.gen_add_code_line("s_dc_du[" + str(nv*nv) + " + c*" + str(nv) + " + " + str(idx) + "] += " + (qd_terms if qd_terms else "static_cast<T>(0)") + ";")
        else:
            coeff = alpha * s_sign
            self.gen_add_code_line("s_dc_du[c*" + str(nv) + " + " + str(idx) + "] += static_cast<T>(" + repr(coeff) + ") * s_temp[" + str(cell(off_df_dq, ind, 0)) + " + 6*c + " + str(s_ind) + "];")
            self.gen_add_code_line("s_dc_du[" + str(nv*nv) + " + c*" + str(nv) + " + " + str(idx) + "] += static_cast<T>(" + repr(coeff) + ") * s_temp[" + str(cell(off_df_dqd, ind, 0)) + " + 6*c + " + str(s_ind) + "];")
        self.gen_add_end_control_flow()
        if parent != -1:
            self.gen_add_sync()  # dc_du fold reads df[ind]; df[parent] += below must wait
            # df_dq[:,idx,parent] += alpha * X^T * fxS(S, f[ind]). Single reduced-
            # column (idx) fold on the PARENT -> ONE lane (bit-exact accumulate).
            # fxS(S, f) = Fx(S)*f = fx_times_v(S, f); S = s_sign*e_{s_ind}
            self.gen_add_serial_ops()
            self.gen_add_code_line("T s_fxs[6]; T s_xtfxs[6]; T s_Svec[6];")
            if is_skew:
                self.gen_add_code_line("{ const T S_skew[6] = " + _idg_Svec_cpp(S_vec) + "; for (int r = 0; r < 6; r++) s_Svec[r] = S_skew[r]; }")
            else:
                self.gen_add_code_line("for (int r = 0; r < 6; r++) s_Svec[r] = static_cast<T>(0);")
                self.gen_add_code_line("s_Svec[" + str(s_ind) + "] = static_cast<T>(" + repr(s_sign) + ");")
            self.gen_add_code_line("fx_times_v<T>(s_fxs, s_Svec, &s_vaf[" + str(f_ind) + "]);")
            # X^T * s_fxs : (X^T)[r,p] = X[p,r] = s_XImats[Xoff + p + 6*r]
            self.gen_add_code_line("for (int r = 0; r < 6; r++) { s_xtfxs[r] = static_cast<T>(0);")
            self.gen_add_code_line("  for (int p = 0; p < 6; p++) s_xtfxs[r] += s_XImats[" + str(Xoff) + " + p + 6*r] * s_fxs[p]; }")
            self.gen_add_code_line("for (int r = 0; r < 6; r++) s_temp[" + str(cell(off_df_dq, parent, 0)) + " + 6*" + str(idx) + " + r] += static_cast<T>(" + repr(alpha) + ") * s_xtfxs[r];")
            self.gen_add_end_control_flow()
            # The single-column df_dq[:,idx,parent] += (above) must land BEFORE the
            # X^T propagation below touches that same parent column (preserves the
            # original H-then-I float accumulation order -> bit-exact).
            self.gen_add_sync()
            # df_du[:,:,parent] += X^T * df_du[:,:,ind]   (both dq and dqd).
            # WIN B: one lane per gradient column c (distinct parent columns; the
            # 6-wide reduction over p stays intact per (c,r) -> bit-exact).
            self.gen_add_parallel_loop("c", str(nv))
            self.gen_add_code_line("for (int r = 0; r < 6; r++) {", True)
            self.gen_add_code_line("T acc_q=static_cast<T>(0), acc_qd=static_cast<T>(0);")
            self.gen_add_code_line("for (int p = 0; p < 6; p++) {", True)
            self.gen_add_code_line("T xtr = s_XImats[" + str(Xoff) + " + p + 6*r];")
            self.gen_add_code_line("acc_q  += xtr * s_temp[" + str(cell(off_df_dq,  ind, 0)) + " + 6*c + p];")
            self.gen_add_code_line("acc_qd += xtr * s_temp[" + str(cell(off_df_dqd, ind, 0)) + " + 6*c + p];")
            self.gen_add_end_control_flow()
            self.gen_add_code_line("s_temp[" + str(cell(off_df_dq,  parent, 0)) + " + 6*c + r] += acc_q;")
            self.gen_add_code_line("s_temp[" + str(cell(off_df_dqd, parent, 0)) + " + 6*c + r] += acc_qd;")
            self.gen_add_end_control_flow()
            self.gen_add_end_control_flow()
        # Inter-body barrier: this body's dc_du fold (and any df[parent] writes)
        # must be visible before the next (shallower) body reads/accumulates.
        self.gen_add_sync()
    self.gen_add_sync()


def _inverse_dynamics_gradient_mimic_temp_count(self):
    """Dense mimic ID-gradient scratch size: 6 buffers of 6*nv*NB + Iv(6*NB)."""
    nv = self.robot.get_num_vel()
    NB = self.robot.get_num_joints()
    return 6 * (6 * nv * NB) + 6 * NB

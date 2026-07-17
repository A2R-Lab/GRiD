def _minv_Svec_cpp(robot, jid):
    """C++ brace-init for body ``jid``'s dense 6-vector motion subspace (Tier-B
    skew emit)."""
    return "{" + ", ".join("static_cast<T>(" + repr(float(c)) + ")" for c in robot._get_flat_S_by_id(jid)) + "}"


def gen_minv_inner_F_size(self):
    """Size of the F-region buffer used by minv_inner (6 * NV * NV
    floats). Phase 3a splits this out as a separate `s_F` parameter so
    callers can surgically spill it to L2-pinned workspace on humanoid-scale
    robots while keeping IA / U / Dinv / Ia / IaTemp in shared memory."""
    n = self.robot.get_num_vel()
    return 6 * n * n

def gen_minv_inner_no_F_size(self):
    """Size of the s_temp arena passed to minv_inner — IA + U + Dinv
    + Ia + IaTemp (everything except the F-region, which moved to s_F)."""
    n = self.robot.get_num_vel()
    NJ = self.robot.get_num_joints()
    max_bfs_width = self.robot.get_max_bfs_width()
    d_inv_count = NJ + 36 if self.robot.floating_base else n
    return 36*n + 6*n + d_inv_count + 36*2*max_bfs_width

def gen_minv_inner_temp_mem_size(self):
    """Legacy: full inner-temp size = F-region + everything else. Callers
    that pre-date Phase 3a allocate this much smem and pass it as both s_F
    (at offset 0) and s_temp (at offset 6*NV*NV) to minv_inner."""
    return self.gen_minv_inner_F_size() + self.gen_minv_inner_no_F_size()

def gen_minv_inner_function_call(self, updated_var_names = None,
                                        f_in_smem_expr = "true"):
    var_names = dict( \
        s_Minv_name = "s_Minv", \
        s_q_name = "s_q", \
        s_temp_name = "s_temp", \
        d_workspace_name = "nullptr", \
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    # Inner-controlled placement (design rollout): minv_inner is keyed on
    # bool F_IN_SMEM and decides at the top of the fn whether the 6*NV*NV F-region
    # lives in s_temp (smem) or d_workspace (L2-pinned global). The caller just
    # passes both arenas + the placement; sizes come from
    # MINV_INNER_{SMEM,WORKSPACE}_BYTES<T, F_IN_SMEM>().
    minv_code_start = "minv_inner<T, " + f_in_smem_expr + ">(" + var_names["s_Minv_name"] + ", " + var_names["s_q_name"] + ", "
    minv_code_end = var_names["s_temp_name"] + ", " + var_names["d_workspace_name"] + ");"
    minv_code_middle = self.gen_insert_helpers_function_call()
    minv_code = minv_code_start + minv_code_middle + minv_code_end
    self.gen_add_code_line(minv_code)

def _gen_minv_inner_via_crba(self, n, no_F_size, label):
    """Emit the inv(CRBA(q)) Minv path: M = crba_inner(q); Minv = inv(M).

    Shared by the mimic path (per-body (S,U,d) ABA recursion does not superpose:
    M_reduced^{-1} != G^T M_full^{-1} G) and the Tier-C SPHERICAL path (the ABA
    minv recursion's scalar U/Dinv/F are single-DoF; a ball joint is 3-DoF, so the
    only correct general handling is to invert the reduced CRBA M directly). CRBA
    already emits the reduced NV x NV M for both joint structures (mimic: alpha-fold
    +=, spherical: multi-column S^T IC S), so here we just invert it.

    n = NV. s_F (6*NV*NV) is the scratch arena here (F is never built on this
    path). Layout in s_F:
      [0, NV*NV)            M_buf (reduced mass matrix from crba_inner)
      [NV*NV, ...)          invert_matrix scratch (>= NV; 6*NV*NV-NV*NV always fits)
    crba_inner's OWN scratch band reuses s_temp (the no_F region), which is
    otherwise unused on this path and is sized >= crba_inner's temp need (the
    mimic CRBA band is <= 42*NB; the spherical CRBA band is 36*NB+36 — both <=
    no_F = 36*NV + 6*NV + d_inv + 36*2*max_bfs_width for these robots; verified
    in the equivalence test). invert_matrix writes the FULL dense symmetric
    inverse into s_Minv; forward_dynamics_finish / gen_minv_apply read only its
    upper triangle (SYMMETRIC_UPPER), which is correct for a dense symmetric M^-1.
    """
    self.gen_add_code_line("// " + label + " Minv = inv(CRBA(q))")
    # M lives in s_F[0..NV*NV); the rest of s_F (6*NV*NV total) is invert scratch.
    # crba_inner's scratch band reuses s_temp (the no_F region, >= 36*NB) which is
    # otherwise unused on this path. crba is config-only: s_qd + gravity unused.
    self.gen_add_code_line("T *s_M_reduced = &s_F[0];")
    self.gen_add_code_line("T *s_invert_temp = &s_F[" + str(n*n) + "];")
    helpers = self.gen_insert_helpers_function_call()  # "s_XImats, s_topology_helpers, "
    # Signature: crba_inner(s_M, s_q, s_qd, [s_XImats, s_topology_helpers], s_temp, d_workspace, gravity)
    self.gen_add_code_line(
        "crba_inner<T, true>(s_M_reduced, s_q, s_q, " + helpers
        + "s_temp, nullptr, static_cast<T>(0));")
    self.gen_add_sync()
    # Invert the reduced NV x NV mass matrix into s_Minv (dense, block-cooperative).
    self.gen_add_code_line("invert_matrix(" + str(n) + ", s_M_reduced, s_Minv, s_invert_temp);")
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_minv_inner(self):
    NJ = self.robot.get_num_joints()
    n = self.robot.get_num_vel()
    max_bfs_levels = self.robot.get_max_bfs_level()
    max_bfs_width = self.robot.get_max_bfs_width()
    if self.robot_has_mimic_joints() or self.robot.robot_has_spherical():
        # The mimic AND spherical Minv paths call crba_inner, which is emitted
        # AFTER this function in the header. Forward-declare it (only for those
        # robots, so the cardinal non-mimic header stays byte-identical). Signature
        # must match _crba.py.
        self.gen_add_code_line("// forward decl: mimic Minv routes through crba_inner (emitted later)")
        self.gen_add_code_line("// (no default arg here; the definition below carries TEMP_IN_SMEM = true)")
        self.gen_add_code_line("template <typename T, bool TEMP_IN_SMEM>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("void crba_inner(T *s_M, const T *s_q, const T *s_qd, T *s_XImats, int *s_topology_helpers, T *s_temp, T *d_workspace, const T gravity);")
    # construct the boilerplate and function definition
    no_F_size = self.gen_minv_inner_no_F_size()
    F_size = self.gen_minv_inner_F_size()
    func_params = ["s_Minv is a pointer to memory for the final result", \
                   "s_q is the vector of joint positions", \
                   "s_temp is the (shared) scratch; size MINV_INNER_SMEM_BYTES<T, F_IN_SMEM>() (= " + str(no_F_size) + " always, plus the " + str(F_size) + "-float F-region when F_IN_SMEM)", \
                   "d_workspace is the global scratch; size MINV_INNER_WORKSPACE_BYTES<T, F_IN_SMEM>() (= " + str(F_size) + " when !F_IN_SMEM, else 0). Pass nullptr when F_IN_SMEM"]
    func_notes = ["CALLER CONTRACT (direct *_inner callers): s_XImats must ALREADY be populated for the current s_q (READ-only here) via load_update_XImats_helpers(...) + __syncthreads(), and s_temp MUST be MINV_INNER_SMEM_BYTES<T, F_IN_SMEM>() bytes -- at F_IN_SMEM=true the 6*NV*NV F-band lives in the TAIL of s_temp (d_workspace=0/nullptr does NOT mean the band is free). Under-populating XImats or under-sizing s_temp reads never-written shared -> NaN (race-clean, DoF-specific since the band scales as 6*NV*NV). Prefer minv_device, which handles both.", \
                  "Outputs a SYMMETRIC_UPPER triangular matrix for Minv", \
                  "Inner-controlled placement: F_IN_SMEM selects where the 6*NV*NV F-region lives.", \
                  "  true  -> tail of s_temp (shared; fastest, default).  false -> d_workspace (global).", \
                  "The choice is made at the top of this fn so the caller just sizes both arenas from", \
                  "the MINV_INNER_*_BYTES constants and hands both pointers in. Codegen maps each", \
                  "RESOURCE_TIER to an F_IN_SMEM value per robot (MINV_F_IN_SMEM<TIER>())."]
    func_def_start = "void minv_inner(T *s_Minv, const T *s_q, "
    func_def_end = "T *s_temp, T *d_workspace) {"
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -1)
    func_def = func_def_start + func_def_end
    self.gen_add_func_doc("Compute the inverse of the mass matrix",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T, bool F_IN_SMEM = true>")
    # __forceinline__: this inner is the heaviest sub-routine of the dynamics-
    # gradient / SO orchestrators (~108 regs). Under -rdc (single-call/anti-LICM
    # build) a separate __device__ callee's regcount must fit the calling kernel's
    # launch_bounds budget, which at LITE/MINIMAL (more threads) is only ~80/64 ->
    # ptxas regcount error. Inlining folds it into the kernel ENTRY (which may
    # spill to local under launch_bounds) instead of being a budget-checked callee.
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line(func_def, True)
    # linalg_smem_setup expects the temp-region size that the function's emitted
    # body indexes into s_temp (IA/U/Dinv/Ia/IaTemp). The F-region is sliced
    # separately below.
    temp_size = no_F_size
    self.gen_linalg_smem_setup(temp_size)

    # Inner-controlled F placement: F lives at the tail of s_temp (after the
    # no_F region) when F_IN_SMEM, else in d_workspace. Keeping no_F at offset 0
    # leaves every s_temp[...] body reference below unchanged.
    self.gen_add_code_line("T *s_F;")
    self.gen_add_code_line("if constexpr (F_IN_SMEM) { s_F = &s_temp[" + str(no_F_size) + "]; (void)d_workspace; }")
    self.gen_add_code_line("else { s_F = d_workspace; }")

    if self.robot_has_mimic_joints():
        # Mimic-aware Minv = inv(M_reduced). The per-body U/Dinv ABA recursion
        # does NOT superpose for mimic joints (M_reduced^{-1} != G^T M_full^{-1} G),
        # so — exactly like RBDReference.minv's mimic fast path — we form the
        # reduced mass matrix via crba_inner and invert it (NV x NV). s_F is large
        # (6*NV*NV) and unused on this path, so we carve M + the CRBA scratch band
        # out of it; the inverse is written straight into s_Minv.
        _gen_minv_inner_via_crba(self, n, no_F_size, "mimic")
        return
    if self.robot.robot_has_spherical():
        # Tier-C SPHERICAL: the per-body ABA minv recursion (scalar U/Dinv/F) does
        # not generalize to a 3-DoF ball joint, so — like the mimic path — invert
        # the reduced CRBA M directly (crba_inner already emits the multi-column
        # spherical M=S^T IC S). RBDReference.minv's ABA bpass raises on spherical
        # for the same reason, so the verified oracle here is inv(crba(q)).
        _gen_minv_inner_via_crba(self, n, no_F_size, "spherical")
        return
    FOffset = 0   # within s_F
    IAOffset = 0  # within s_temp
    UOffset = IAOffset + 36*n
    DinvOffset = UOffset + 6*n
    IaOffset = DinvOffset + NJ
    if self.robot.floating_base:
        fb_DinvOffset = IaOffset
        IaOffset += 36 # fb_dinv is a 6x6 matrix
    IaTempOffset = IaOffset + 36*max_bfs_width
    self.gen_add_code_line("// T *s_F (param) [size 6*NV*NV]; T *s_IA = &s_temp[" + str(IAOffset) + "]; T *s_U = &s_temp[" + str(UOffset) + "];" + \
                             " T *s_Dinv = &s_temp[" + str(DinvOffset) + "]; T *s_Ia = &s_temp[" + str(IaOffset) + "]; T *s_IaTemp = &s_temp[" + str(IaTempOffset) + "];")

    # set initial IA to I and zero Minv/F
    self.gen_add_code_line("// Initialize IA = I")
    self.gen_add_parallel_loop("ind",str(36*NJ))
    self.gen_add_code_line("s_temp[" + str(IAOffset) + " + ind] = s_XImats[" + str(36*NJ) + " + ind];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("// Zero Minv and F")
    self.gen_add_parallel_loop("ind",str(n*n*7))
    self.gen_add_code_line("if(ind < " + str(6*n*n) + "){s_F[" + str(FOffset) + " + ind] = static_cast<T>(0);}")
    self.gen_add_code_line("else{s_Minv[ind - " + str(6*n*n) + "] = static_cast<T>(0);}")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_line("printf(\"q\\n\"); printMat<T,1," + str(n) + ">(s_q,1);")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"X[%d]\\n\",i); printMat<T,6,6>(&s_XImats[36*i],6);}")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"IA_init = I[%d]\\n\",i); printMat<T,6,6>(&s_temp[" + str(IAOffset) + " + 36*i],6);}")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    #
    # First compute the Backward Pass in bfs waves
    #
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Backward Pass")
    self.gen_add_code_line("//")
    for bfs_level in range(max_bfs_levels, -1, -1):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)
        # Tier-B (skew) levels consume the dense S directly; the signed-index
        # S_sign helper raises on a skew joint, so only query it for cardinal
        # levels (cardinal robots unchanged -> byte-identical).
        level_has_skew_bwd = any(not self.robot.S_is_cardinal_by_id(j) for j in inds)
        S_sign_cpp = None if level_has_skew_bwd else self.gen_topology_S_sign_for_cpp(inds)
        ind_subtree_inds = []
        subree_counts = []
        for ind in inds:
            if self.robot.floating_base: ind_subtree_inds.extend([(ind+5,subInd+5) for subInd in self.robot.get_subtree_by_id(ind)]) # fb dof offset
            else: ind_subtree_inds.extend([(ind,subInd) for subInd in self.robot.get_subtree_by_id(ind)])
            subree_counts.append(len(self.robot.get_subtree_by_id(ind)))
        subtree_adjust = [sum(subree_counts[:idx]) for idx in range(len(inds) + 1)]

        self.gen_add_code_line("// backward pass updates where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))

        if self.robot.floating_base and bfs_level == 0:
            # U = IA because S is identity, so Dinv = IA^{-1}, Top left 6x6 in minv = Dinv
            self.gen_add_code_line("// U = IA*S = IA, D = S^T*U = U = IA => Minv[:6, :6] = IA^{-1}")
            # No Ainv=I pre-init: glass::inv_dense seeds Ainv internally.
            self.gen_add_code_line(f"invert_matrix(6, &s_temp[{str(IAOffset)}], &s_temp[{str(fb_DinvOffset)}], &s_temp[{IaTempOffset}]);")

            # Top left 6x6 in minv <- Dinv
            self.gen_add_parallel_loop("ind", '36')
            self.gen_add_code_line("int row = ind % 6, col = ind / 6;")
            self.gen_add_code_line(f"s_Minv[col*{n}+row] = s_temp[{fb_DinvOffset}+ind];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()


        elif any(not self.robot.S_is_cardinal_by_id(j) for j in inds):
            # Tier B (skew): U = IA*S_dense; D = S^T U; Dinv = 1/D; Minv[i,i] = Dinv.
            # Loop over every joint in the level (branching ok); each with its own
            # dense (or unit) S column.
            self.gen_add_code_line("// U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv (Tier-B dense S)")
            self.gen_add_serial_ops()
            for jid in inds:
                jid6 = 6 * jid
                dof_diag = (jid + 5) if self.robot.floating_base else jid  # fb dof offset
                Svec = _minv_Svec_cpp(self.robot, jid)
                self.gen_add_code_line("{ const T S_skew[6] = " + Svec + ";")
                self.gen_add_code_line("  for (int r = 0; r < 6; r++) { T acc = static_cast<T>(0); for (int p = 0; p < 6; p++) { acc += s_temp[" + str(IAOffset) + " + 6*" + str(jid6) + " + r + 6*p] * S_skew[p]; } s_temp[" + str(UOffset) + " + " + str(jid6) + " + r] = acc; }")
                self.gen_add_code_line("  s_temp[" + str(DinvOffset) + " + " + str(jid) + "] = static_cast<T>(1)/dot_prod<T,6,1,1>(S_skew, &s_temp[" + str(UOffset) + " + " + str(jid6) + "]);")
                self.gen_add_code_line("  s_Minv[" + str(n + 1) + " * " + str(dof_diag) + "] = s_temp[" + str(DinvOffset) + " + " + str(jid) + "]; }")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
        else:
            # U = Scol of IA then D = Srow of U then note that DInv = 1/D, Minv[i,i] = Dinv
            self.gen_add_code_line("// U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv")
            if len(inds) > 1:
                self.gen_add_parallel_loop("ind",str(6*len(inds)))
                self.gen_add_code_line("int row = ind % 6;")
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "<", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
                self.gen_add_code_line("int jid6 = 6*jid;")
                jid = "jid"
                jid6 = "jid6"
            else:
                jid = str(inds[0]) 
                jid6 = str(6*inds[0])
                self.gen_add_parallel_loop("row",str(6))
            self.gen_add_code_line("s_temp[" + str(UOffset) + " + " + jid6 + " + row] = (" + S_sign_cpp + ") * s_temp[" + str(IAOffset) + " + 6*" + jid6 + " + 6*" + S_ind_cpp + " + row];")
            self.gen_add_code_line("if(row == " + S_ind_cpp + "){", True)
            self.gen_add_code_line("s_temp[" + str(DinvOffset) + " + " + jid + "] = static_cast<T>(1)/((" + S_sign_cpp + ") * s_temp[" + str(UOffset) + " + " + jid6 + " + " + S_ind_cpp + "]);")
            # need offset due to floating base matrix in upper left
            if self.robot.floating_base: self.gen_add_code_line(f's_Minv[{str(n + 1)} * ({jid} + 5)]' + " = s_temp[" + str(DinvOffset) + " + " + jid + "];")
            else: self.gen_add_code_line("s_Minv[" + str(n + 1) + " * " + jid + "] = s_temp[" + str(DinvOffset) + " + " + jid + "];")
            self.gen_add_end_control_flow()
            self.gen_add_end_control_flow()
            self.gen_add_sync()

        if self.DEBUG_MODE:
            self.gen_add_sync()
            self.gen_add_serial_ops()
            for ind in inds:
                self.gen_add_code_line("printf(\"U[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_temp[" + str(UOffset) + " + 6*" + str(ind) + "],1);")
                self.gen_add_code_line("printf(\"Dinv[" + str(ind) + "] = %f\\n\",s_temp[" + str(DinvOffset) + " + " + str(ind) + "]);")
            self.gen_add_code_line("printf(\"Minv after Dinv setting before subtree\\n\"); printMat<T," + str(n) + "," + str(n) + ">(s_Minv," + str(n) + ");")
            self.gen_add_end_control_flow()
            self.gen_add_sync()

        # then for the subtrees we know that Minv[i,subTreeInds] -= F[i,Srow,SubTreeInds] scalar -> scalar
        #                                and temp comp F[i,:,subTreeInds] += U*Minv[i,subTreeInds] vector*scalar -> vector (only if parent exists)
        # Note that by not supporting looped URDFs we can ensure that subtrees are independent and safe for parallelism
        self.gen_add_code_line("// Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]")

        if self.robot.floating_base and bfs_level == 0:
            self.gen_add_parallel_loop("ind", str((NJ-1)*6))
            self.gen_add_code_line(f"int row = ind % 6, col = ind / 6;")
            self.gen_add_code_line(f"s_Minv[(6+col)*{n}+row] -= dot_prod<T,6,1,1>(&s_temp[{fb_DinvOffset}+6*row], &s_F[{FOffset+6*n*5+36}+6*col]);") # offset to 7th dof (past fb) in fb F matrix
            self.gen_add_end_control_flow()
        else:
            if bfs_level != 0:
                self.gen_add_code_line("// Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update")
            self.gen_add_parallel_loop("ind",str(len(ind_subtree_inds)))
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds]),
                                ("int", "subTreeAdj", [str(val) for val in subtree_adjust])]
                self.gen_add_multi_threaded_select("ind", "<", [str(subtree_adjust[i+1]) for i in range(len(inds))], select_var_vals, USE_NON_BRANCH_ALWAYS = True)
                if self.robot.floating_base: # dof offset for columns (fb takes up 6 columns instead of 1)
                    self.gen_add_code_line("int jid_subtree = (jid + 5) + (ind - subTreeAdj); " + \
                                    "int jid_subtree6 = 6*jid_subtree; int jid_subtreeN = " + str(n) + "*jid_subtree;")
                else:
                    self.gen_add_code_line("int jid_subtree = jid + (ind - subTreeAdj); " + \
                                    "int jid_subtree6 = 6*jid_subtree; int jid_subtreeN = " + str(n) + "*jid_subtree;")
                jid = "jid"
                jid_subtree6 = "jid_subtree6"
                jid_subtreeN = "jid_subtreeN"
                if self.robot.floating_base: dof_id = '(jid + 5)'
                else: dof_id = jid
            else:
                if self.robot.floating_base: 
                    jid = str(inds[0])
                    dof_id = str(int(jid) + 5)
                else: 
                    jid = str(inds[0])
                    dof_id = jid
                select_var_vals = []
                if len(ind_subtree_inds) > 1:
                    self.gen_add_code_line("int jid_subtree6 = 6*(" + dof_id + " + ind); int jid_subtreeN = " + str(n) + "*(" + dof_id + " + ind);")
                    jid_subtree6 = "jid_subtree6"
                    jid_subtreeN = "jid_subtreeN"
                else: 
                    subId = ind_subtree_inds[0][1]
                    jid_subtree6 = str(subId*6)
                    jid_subtreeN = str(subId*n)

            level_has_skew = any(not self.robot.S_is_cardinal_by_id(j) for j in inds)
            if level_has_skew:
                # Tier B (skew): the S^T*F[:,col] projection is a dense dot over
                # the 6 rows of the F column. For a branching level jid is a
                # runtime var, so pick the per-joint dense S column by jid before
                # the dot (each joint with its own compile-time S constant).
                Fcol = "&s_F[" + str(FOffset) + " + " + str(n*6) + "*" + dof_id + " + " + jid_subtree6 + "]"
                if len(inds) > 1:
                    self.gen_add_code_line("T S_skew[6];")
                    for i, jid_val in enumerate(inds):
                        Svec = _minv_Svec_cpp(self.robot, jid_val)
                        self.gen_add_code_line(("if " if i == 0 else "else if ") + "(" + jid + " == " + str(jid_val) + ") { const T tmp[6] = " + Svec + "; for (int r=0;r<6;r++) S_skew[r]=tmp[r]; }")
                else:
                    Svec = _minv_Svec_cpp(self.robot, inds[0])
                    self.gen_add_code_line("const T S_skew[6] = " + Svec + ";")
                S_proj = "dot_prod<T,6,1,1>(S_skew, " + Fcol + ")"
                self.gen_add_code_line("s_Minv[" + jid_subtreeN + " + " + dof_id + "] -= s_temp[" + str(DinvOffset) + " + " + jid + "] * " + S_proj + ";")
            else:
                self.gen_add_code_line("s_Minv[" + jid_subtreeN + " + " + dof_id + "] -= s_temp[" + str(DinvOffset) + " + " + jid + "] * " + \
                                                        "(" + S_sign_cpp + ") * s_F[" + str(FOffset) + " + " + str(n*6) + "*" + dof_id + " + " + jid_subtree6 + " + " + S_ind_cpp + "];")
            if bfs_level != 0:
                self.gen_add_code_line("for(int row = 0; row < 6; row++) {", True)
                self.gen_add_code_line("s_F[" + str(FOffset) + " + " + str(n*6) + "*" + dof_id + " + " + jid_subtree6 + " + row] += " + \
                                            "s_temp[" + str(UOffset) + " + 6*" + jid + " + row] * s_Minv[" + jid_subtreeN + " + " + dof_id + "];")
                self.gen_add_end_control_flow()
            self.gen_add_end_control_flow()

        if self.DEBUG_MODE:
            self.gen_add_sync()
            self.gen_add_serial_ops()
            self.gen_add_code_line("printf(\"Minv after subtree updates\\n\"); printMat<T," + str(n) + "," + str(n) + ">(s_Minv," + str(n) + ");")
            if bfs_level != 0:
                for ind in inds:
                    self.gen_add_code_line("printf(\"F Temp += U*Minv[" + str(ind) + "]\\n\"); printMat<T,6," + str(n) + ">(&s_F[" + str(FOffset + n*6*ind) + "],6);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()

        # Then start the IA update (if there is a parent) with Ia = IA[ind] - np.outer(U[ind,:],Dinv[ind]*U[ind,:])
        if bfs_level != 0:
            self.gen_add_code_line("// Ia = IA - U^T Dinv U | to start IAparent Update")
            self.gen_add_parallel_loop("ind",str(36*len(inds)))
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "<", [str(36*(i+1)) for i in range(len(inds))], select_var_vals)
                self.gen_add_code_line("int ind36 = (ind % 36); int row = ind36 % 6; int col = ind36 / 6; int jid6 = 6*jid;")
                self.gen_add_code_line("s_temp[" + str(IaOffset) + " + ind] = s_temp[" + str(IAOffset) + " + 6*jid6 + ind36] - " + \
                  "(s_temp[" + str(UOffset) + " + jid6 + row] * s_temp[" + str(DinvOffset) + " + jid] * s_temp[" + str(UOffset) + " + jid6 + col]);")
            else:
                jid = inds[0]
                self.gen_add_code_line("int row = ind % 6; int col = ind / 6;")
                self.gen_add_code_line("s_temp[" + str(IaOffset) + " + ind] = s_temp[" + str(IAOffset + 36*jid) + " + ind] - " + \
                  "(s_temp[" + str(UOffset + 6*jid) + " + row] * s_temp[" + str(DinvOffset + jid) + "] * s_temp[" + str(UOffset + 6*jid) + " + col]);")
            self.gen_add_end_control_flow()
        self.gen_add_sync()

        if self.DEBUG_MODE:
            self.gen_add_sync()
            self.gen_add_serial_ops()
            for i in range(len(inds)):
                self.gen_add_code_lines(["printf(\"Ia[" + str(inds[i]) + "]\\n\");",
                                         "printMat<T,6,6>(&s_temp[" + str(IaOffset + 36*i) + "],6);"])
            self.gen_add_end_control_flow()
            self.gen_add_sync()

        if bfs_level != 0:
            # then for the subtrees we can do (in parallel by both subtree and row)
            #                 F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds] matrix*vector -> vector
            # At the same time also do next step of IA update: IA_Update_Temp = Xmat^T * Ia
            self.gen_add_code_line("// F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]")
            self.gen_add_code_line("// IA_Update_Temp = Xmat^T * Ia | for IAparent Update")
            self.gen_add_parallel_loop("ind",str(6*len(ind_subtree_inds) + 6*6*len(inds)))
            self.gen_add_code_line("int row = ind % 6; int col = ind / 6;")
            if len(ind_subtree_inds) > 1:
                if len(inds) > 1:
                    select_var_vals = [("int", "jid", [str(jid) for jid in inds]),
                                       ("int", "subTreeAdj", [str(val) for val in subtree_adjust])]
                    self.gen_add_multi_threaded_select("col", "<", [str(subtree_adjust[i+1]) for i in range(len(inds))], select_var_vals, USE_NON_BRANCH_ALWAYS = True)
                    if self.robot.floating_base: self.gen_add_code_line("int jid_subtree = (jid + 5) + (col - subTreeAdj);")
                    else: self.gen_add_code_line("int jid_subtree = jid + (col - subTreeAdj);")

                    jid = "jid"
                    jid_subtree = "jid_subtree"
                else:
                    jid = str(inds[0])
                    if self.robot.floating_base: 
                        self.gen_add_code_line(f"int col_offset = 5 * (col < {str(len(ind_subtree_inds))});")
                        jid_subtree = "(" + jid + " + col + col_offset)"
                    else: jid_subtree = "(" + jid + " + col)"
            else:
                if self.robot.floating_base: 
                    jid = str(inds[0])
                    dof_id = str(inds[0] + 5)
                else: 
                    jid = str(inds[0])
                    dof_id = jid
                jid_subtree = str(ind_subtree_inds[0][1])
            # do the standard comps first
            if self.robot.floating_base: parent_ind_cpp_adj = f'(5 + {parent_ind_cpp})'
            else: parent_ind_cpp_adj = parent_ind_cpp
            self.gen_add_code_lines(["T *src = &s_F[" + str(FOffset) + " + " + str(6*n) + "*" + str(dof_id) + " + 6*" + jid_subtree + "]; " + \
                                        "T *dst = &s_F[" + str(FOffset) + " + " + str(6*n) + "*" + parent_ind_cpp_adj + " + 6*" + jid_subtree + "];"])
            # adjust for temp comps
            self.gen_add_code_line("// adjust for temp comps")
            self.gen_add_code_line("if (col >= " + str(len(ind_subtree_inds)) + ") {",True)
            self.gen_add_code_lines(["col -= " + str(len(ind_subtree_inds)) + "; " + \
                                        "src = &s_temp[" + str(IaOffset) + " + 6*col]; " + \
                                        "dst = &s_temp[" + str(IaTempOffset) + " + 6*col];"])
            if len(inds) > 1:
                self.gen_add_code_line("int jid_selector = col / 6;")
                self.gen_add_multi_threaded_select("jid_selector", "==", [str(i) for i in range(len(inds))], [(None, "jid", [str(ind) for ind in inds])])
            self.gen_add_end_control_flow()
            # then do the computation
            self.gen_add_code_line("dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*" + jid + " + 6*row],src);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()

            if self.DEBUG_MODE:
                self.gen_add_sync()
                self.gen_add_serial_ops()
                for i in range(len(inds)):
                    self.gen_add_code_lines(["printf(\"F[" + str(self.robot.get_parent_id(inds[i])) + "] = X^T F[" + str(inds[i]) + "]\\n\");",
                                             "printMat<T,6," + str(n) + ">(&s_F[" + str(FOffset + n*6*self.robot.get_parent_id(inds[i])) + "],6);",
                                             "printf(\"Ia*X[" + str(inds[i]) + "]\\n\");",
                                             "printMat<T,6,6>(&s_temp[" + str(IaTempOffset + 36*i) + "],6);"])
                self.gen_add_end_control_flow()
                self.gen_add_sync()

            # Finally IA[parent_ind] += IA_Update_Temp * Xmat
            self.gen_add_code_line("// IA[parent_ind] += IA_Update_Temp * Xmat")
            if len(inds) > 1 and self.robot.has_repeated_parents(inds):
                # Shared parents (quadruped legs → floating root): the per-child IA
                # updates COLLIDE on the parent's IA cells. A slot-major atomicAdd
                # would sum them in warp-scheduling order → last-ULP run-to-run
                # variation (single-block kernels must be bit-deterministic, Inc6).
                # Iterate PARENT-cell-major instead: each unique parent cell is
                # owned by one thread that sums its child slots in FIXED ascending
                # slot order (race-free, deterministic, no atomics).
                unique_parents = sorted(set(self.robot.get_parent_id(j) for j in inds))
                nup = len(unique_parents)
                upar_csv = ", ".join(str(p) for p in unique_parents)
                jids_csv = ", ".join(str(j) for j in inds)
                pars_csv = ", ".join(str(self.robot.get_parent_id(j)) for j in inds)
                # Per-level {} scope so the compile-time tables don't collide
                # across BFS levels emitted into the same function body.
                self.gen_add_code_line("// deterministic parent-major fixed-order sum (shared parent): IA[parent] += sum_slot IA_Update_Temp[slot] * X[jid_slot]")
                self.gen_add_code_line("{", True)
                self.gen_add_code_line(f"const int s_jid_lvl[{len(inds)}] = {{{jids_csv}}};")
                self.gen_add_code_line(f"const int s_par_lvl[{len(inds)}] = {{{pars_csv}}};")
                self.gen_add_code_line(f"const int s_upar_lvl[{nup}] = {{{upar_csv}}};")
                self.gen_add_parallel_loop("ind",str(6*6*nup))
                self.gen_add_code_line("int up = ind / 36; int rc = ind % 36;")
                self.gen_add_code_line("int col_max6 = rc / 6; int row = rc % 6;")
                self.gen_add_code_line("int par_l = s_upar_lvl[up];")
                self.gen_add_code_line("T acc = static_cast<T>(0);")
                self.gen_add_code_line(f"for (int slot = 0; slot < {len(inds)}; slot++) {{ if (s_par_lvl[slot] != par_l) continue;")
                self.gen_add_code_line(f"    T * src = &s_temp[{IaTempOffset} + 36*slot + row];")
                self.gen_add_code_line("    acc += dot_prod<T,6,6,1>(src,&s_XImats[36*s_jid_lvl[slot] + 6*col_max6]); }")
                self.gen_add_code_line(f"s_temp[{IAOffset} + 36*par_l + 6*col_max6 + row] += acc;")
                self.gen_add_end_control_flow()
                self.gen_add_sync()
                self.gen_add_end_control_flow()  # close the per-level {} scope
            elif len(inds) > 1:
                for i, jid_val in enumerate(inds):
                    parent_val = self.robot.get_parent_id(jid_val)
                    self.gen_add_code_line(f"grid_linalg_gemm<T,6,6,6>(&s_temp[{IaTempOffset + 36*i}], &s_XImats[{36*jid_val}], &s_temp[{IAOffset + 36*parent_val}], static_cast<T>(1), static_cast<T>(1), s_linalg_smem);")
            else:
                jid_val = inds[0]
                parent_val = self.robot.get_parent_id(jid_val)
                self.gen_add_code_line(f"grid_linalg_gemm<T,6,6,6>(&s_temp[{IaTempOffset}], &s_XImats[{36*jid_val}], &s_temp[{IAOffset + 36*parent_val}], static_cast<T>(1), static_cast<T>(1), s_linalg_smem);")

            if self.DEBUG_MODE:
                self.gen_add_sync()
                self.gen_add_serial_ops()
                for ind in inds:
                    self.gen_add_code_lines(["printf(\"IA[" + str(self.robot.get_parent_id(ind)) + "] = X^T*(Ia*X)\\n\");",
                                             "printMat<T,6,6>(&s_temp[" + str(IAOffset + 36*self.robot.get_parent_id(ind)) + "],6);"])
                self.gen_add_end_control_flow()
                self.gen_add_sync()

    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_lines(["printf(\"-------------------\\n\");", \
                                 "printf(\"After Backward Pass\\n\");", \
                                 "printf(\"-------------------\\n\");", \
                                 "printf(\"U\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(UOffset + 6*i) + "],6);", \
                                 "printf(\"Dinv\\n\"); printMat<T,1," + str(n) + ">(&s_temp[" + str(DinvOffset) + "],1);"])
        for i in range(n):
            self.gen_add_code_line("printf(\"F[%d]\\n\"," + str(i) + "); printMat<T,6," + str(n) + ">(&s_F[" + str(FOffset + 6*n*i) + "],6);")
        self.gen_add_code_line("printf(\"Minv\\n\"); printMat<T," + str(n) + "," + str(n) + ">(s_Minv," + str(n) + ");")
        self.gen_add_code_line("printf(\"-------------------\\n\");")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    #
    # Then compute the Forwad Pass
    #    Note that due to the i: operation we need to go serially over all n
    #
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Forward Pass")
    self.gen_add_code_line("//   Note that due to the i: operation we need to go serially over all n")
    self.gen_add_code_line("//")
    # PERF (forward-pass fusion): the outer jid loop is loop-carried-serial (F[jid]
    # feeds children at jid+1..), so it cannot be parallelized across jid. But for a
    # jid WITH a parent the two per-jid parallel loops below were separated by a
    # __syncthreads (the F-column matmul, then the U^T reduction that reads that F
    # column + writes back into it). By making ONE thread own an entire F column we
    # remove the intra-jid sync: the same thread does the 6-row matmul, then reduces
    # ITS OWN just-written column against U, updates Minv, and writes back into its
    # own column. No cross-thread sharing within a jid -> 1 sync per jid instead of
    # 2, halving the ~NV serial syncs that dominate the forward pass at humanoid n.
    # Numerically byte-identical: identical ops in identical reduction order, merely
    # repartitioned (the dot_prod accumulation order is unchanged).
    for jid in range(NJ):
        self.gen_add_code_line("// forward pass for jid: " + str(jid))
        jid_parent = self.robot.get_parent_id(jid)
        jid_cols = list(range(jid,NJ))
        dof_cols = list(range(jid,n))

        jid_is_skew = not (self.robot.floating_base and jid == 0) and not self.robot.S_is_cardinal_by_id(jid)
        if self.robot.floating_base and jid == 0:
            SInd = '-1'
            SSign = '1'
        elif jid_is_skew:
            # Tier B (skew): no signed unit index; the F writeback below uses the
            # dense S column directly (S_skew_fwd) instead of (row==SInd)*SSign.
            SInd = None
            SSign = None
        else:
            SInd = str(self.robot.get_S_index_by_id(jid))
            SSign = str(self.robot.get_S_sign_by_id(jid))

        # Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        # F[i,:,i:] = S^T * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        if jid_parent != -1:
            if self.robot.floating_base:
                dof_id = jid + 5 # dof offset
            else:
                dof_id = jid
            self.gen_add_code_line("// Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N")
            self.gen_add_code_line("// F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N")
            # Fused per-column pass: one thread per F column owns the whole column,
            # so the matmul (writes the column), the U^T reduction (reads it) and the
            # S*Minv writeback (updates it) all happen in-thread with no intra-jid
            # sync. dof_cols (= range(jid,n)) is the matmul's column set (superset);
            # the reduction/writeback runs only for columns at/after the diagonal
            # dof_id (== jid fixed-base; == jid+5 floating, where root-dof columns
            # jid..jid+4 get only the matmul, exactly as before).
            self.gen_add_code_line("//   Per column: F[i,:,col]=Xmat*F[parent,:,col], then")
            self.gen_add_code_line("//   Minv[i,col]-=Dinv*U^T*F[i,:,col] and F[i,Srow,col]+=S*Minv[i,col]")
            diag_offset = dof_id - jid  # 0 fixed-base, 5 floating-base
            self.gen_add_parallel_loop("c",str(len(dof_cols)))
            self.gen_add_code_line("int col_ind = c + " + str(jid) + ";")
            self.gen_add_code_line("T *s_Fcol = &s_F[" + str(FOffset + 6*n*jid) + " + 6*col_ind];")
            self.gen_add_code_line("T *s_Fpcol = &s_F[" + str(FOffset + 6*n*jid_parent) + " + 6*col_ind];")
            self.gen_add_code_line("for (int row = 0; row < 6; row++) {", True)
            self.gen_add_code_line("s_Fcol[row] = dot_prod<T,6,6,1>(&s_XImats[" + str(36*jid) + " + row], s_Fpcol);")
            self.gen_add_end_control_flow()
            # Reduction + Minv update + writeback only for columns at/after the diagonal.
            if diag_offset > 0:
                self.gen_add_code_line("if (c >= " + str(diag_offset) + ") {", True)
            self.gen_add_code_line("s_Minv[" + str(n) + " * col_ind + " + str(dof_id) + "] -= " + \
                                   "s_temp[" + str(DinvOffset + jid) + "] * " + \
                                   "dot_prod<T,6,1,1>(s_Fcol,&s_temp[" + str(UOffset + 6*jid) + "]);")
            if jid < n-1: # skip redundant comp on last loop
                if jid_is_skew:
                    # Tier B: F[:,col] += S_dense * Minv[i,col] (dense column add)
                    Svec = _minv_Svec_cpp(self.robot, jid)
                    self.gen_add_code_line("{ const T S_skew_fwd[6] = " + Svec + "; for (int r = 0; r < 6; r++) { s_Fcol[r] += S_skew_fwd[r] * s_Minv[" + str(n) + " * col_ind + " + str(dof_id) + "]; } }")
                else:
                    self.gen_add_code_line("s_Fcol[" + SInd + "] += (" + SSign + ") * s_Minv[" + str(n) + " * col_ind + " + str(dof_id) + "];")
            if diag_offset > 0:
                self.gen_add_end_control_flow()
            self.gen_add_end_control_flow()
            self.gen_add_sync()

            if self.DEBUG_MODE:
                self.gen_add_sync()
                self.gen_add_serial_ops()
                self.gen_add_code_lines(["printf(\"Minv[i,i:] -= Dinv*U^T*F[i,:,i:] for i = %d\\n\"," + str(jid) + ");", \
                                         "printMat<T," + str(n) + "," + str(n) + ">(s_Minv," + str(n) + ");"])
                if jid < n-1: # redundant comp on last loop
                    self.gen_add_code_lines(["printf(\"F[i,:,i:] += S*Minv[i,i:]\");", \
                                             "printMat<T,6," + str(n) + ">(&s_F[" + str(FOffset + 6*n*jid) + "],6);"])
                self.gen_add_end_control_flow()
                self.gen_add_sync()

        elif jid < n-1: # redundant comp on last loop
            self.gen_add_code_line("// F[i,:,i:] = S * Minv[i,i:] as parent is base so rest is skipped")
            self.gen_add_parallel_loop("ind",str(6*len(dof_cols)))
            self.gen_add_code_line("int row = ind % 6; int col = ind / 6;")
            if self.robot.floating_base: self.gen_add_code_line(f"s_F[ind] = s_Minv[row + {n} * col];") # update F[0] (Phase 3a: F is in s_F)
            elif jid_is_skew:
                # Tier B: F[row,col] = S_dense[row] * Minv[i,col]
                Svec = _minv_Svec_cpp(self.robot, jid)
                self.gen_add_code_line("const T S_skew[6] = " + Svec + ";")
                self.gen_add_code_line("s_F[" + str(FOffset + 6*n*jid + 6*jid) + " + ind] = S_skew[row] * s_Minv[" + str(n*jid + jid) + " + " + str(n) + " * col];")
            else:
                self.gen_add_code_line("s_F[" + str(FOffset + 6*n*jid + 6*jid) + " + ind] = (row == " + SInd + ") * " + \
                                            "(" + SSign + ") * s_Minv[" + str(n*jid + jid) + " + " + str(n) + " * col];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()

            if self.DEBUG_MODE:
                self.gen_add_sync()
                self.gen_add_serial_ops()
                self.gen_add_code_lines(["printf(\"F[i,:,i:] += S*Minv[i,i:] for i = %d\\n\"," + str(jid) + ");", \
                                         "printMat<T,6," + str(n) + ">(&s_F[" + str(FOffset + 6*n*jid) + "],6);"])
                self.gen_add_end_control_flow()
                self.gen_add_sync()
    if self.robot.floating_base:
        self.gen_add_code_line("// Convert floating-base Minv from internal spatial order to public velocity order.")
        self.gen_add_code_line("// Internal root vectors are [angular, linear]; public qd/u/qdd are [linear, angular].")
        self.gen_add_parallel_loop("ind", str(n*n))
        self.gen_add_code_line("int row = ind % " + str(n) + "; int col = ind / " + str(n) + ";")
        self.gen_add_code_line("T val = static_cast<T>(0);")
        self.gen_add_code_line("if (row <= col) {", True)
        self.gen_add_code_line("int src_row = row < 6 ? (row < 3 ? row + 3 : row - 3) : row;")
        self.gen_add_code_line("int src_col = col < 6 ? (col < 3 ? col + 3 : col - 3) : col;")
        self.gen_add_code_line("int read_row = src_row <= src_col ? src_row : src_col;")
        self.gen_add_code_line("int read_col = src_row <= src_col ? src_col : src_row;")
        self.gen_add_code_line("val = s_Minv[read_col * " + str(n) + " + read_row];")
        self.gen_add_end_control_flow()
        # Phase 3a: reuse s_F as permutation scratch — F is no longer needed at
        # this point, and s_F is sized 6*NV*NV which always exceeds NV*NV.
        # (Post-Phase-3a s_temp holds only IA/U/Dinv/Ia/IaTemp, which can be
        # smaller than NV*NV on humanoid-scale floating-base robots.)
        self.gen_add_code_line("s_F[ind] = val;")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_add_parallel_loop("ind", str(n*n))
        self.gen_add_code_line("s_Minv[ind] = s_F[ind];")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
    self.gen_add_end_function()

def gen_minv_device(self):
    # construct the boilerplate and function definition
    func_params = ["s_Minv is a pointer to memory for the final result", \
                   "s_q is the vector of joint positions", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)"]
    func_def = "void minv_device(T *s_Minv, const T *s_q, const robotModel<T> *d_robotModel){"
    func_notes = ["Outputs a SYMMETRIC_UPPER triangular matrix for Minv"]
    # then generate the code (shared device-wrapper skeleton; B+C §1.1).
    # Full smem layout: no_F + F packed contiguously. Inner-callable device path
    # keeps F in smem (F_IN_SMEM=true); the inner slices s_F from the tail of
    # s_temp itself.
    n = self.robot.get_num_vel()
    shared_mem_size = self.gen_minv_inner_temp_mem_size()  # no_F + F
    self.gen_device_wrapper(
        "Compute the inverse of the mass matrix", func_def, shared_mem_size,
        lambda: self.gen_minv_inner_function_call(f_in_smem_expr = "true"),
        func_notes = func_notes, func_params = func_params,
        include_linalg_scratch = True)

def _emit_minv_kernel_body_for_flags(self, n, NV, spill_F, single_call_timing):
    """Emit minv_kernel body for one tier's spill flag.
    spill_F=False: s_F lives at &s_temp[0]; s_temp_inner at &s_temp[6*NV*NV]; smem holds everything.
    spill_F=True:  s_F lives at &d_workspace[...]; s_temp_inner at &s_temp[0]; saves 6*NV*NV from smem."""
    n_pos = self.robot.get_num_pos()  # NUM_POS (= NV for fixed, NV+1 for floating quaternion)
    F_size = self.gen_minv_inner_F_size()
    if spill_F:
        shared_mem_size = self.gen_minv_inner_no_F_size()
    else:
        shared_mem_size = self.gen_minv_inner_temp_mem_size()
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size, extra_t_buffers = [("s_q", n_pos), ("s_Minv", n*n)], include_linalg_scratch=True)
    if not single_call_timing:
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",block_level = True)
        self.gen_kernel_load_inputs("q",str(n_pos),stride="stride_q")
        if spill_F:
            # L2-pinned workspace slot for Minv-F (inner picks it via F_IN_SMEM=false).
            self.gen_add_code_line("T *minv_d_workspace = reinterpret_cast<T *>(&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()]);")
        else:
            self.gen_add_code_line("(void)d_workspace;")
        # mjx input convert (quaternion only -> the congruence epilogue's R; Minv(q)
        # is base-orientation-independent so no qd/accel convert is needed). Must
        # precede the XImats build so X[0] is built from the xyzw quaternion.
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_quat_reorder("s_q")
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_minv_inner_function_call(
            updated_var_names = (dict(d_workspace_name = "minv_d_workspace") if spill_F else None),
            f_in_smem_expr = ("false" if spill_F else "true"))
        self.gen_add_sync()
        # mjx output: Minv is a congruence G Minv G^T. Minv is SYMMETRIC_UPPER
        # storage (lower triangle zero), and the congruence reads FULL base rows/
        # cols, so first mirror upper->lower to fully populate, THEN congruence. The
        # result is fully dense symmetric mjx Minv (host symmetrize becomes a no-op).
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_symmetrize_full("s_Minv", n)
            self.gen_mjx_congruence("s_Minv", n)
            self.gen_add_end_control_flow()
        self.gen_kernel_save_result("Minv",str(n*n),stride=str(n*n))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q",str(n_pos))
        if spill_F:
            self.gen_add_code_line("T *minv_d_workspace = reinterpret_cast<T *>(&d_workspace[GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()]);")
        else:
            self.gen_add_code_line("(void)d_workspace;")
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q",str(n_pos),feedback_from="Minv")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_minv_inner_function_call(
            updated_var_names = (dict(d_workspace_name = "minv_d_workspace") if spill_F else None),
            f_in_smem_expr = ("false" if spill_F else "true"))
        self.gen_anti_licm_output_write("Minv")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("Minv",str(n*n))


def gen_minv_kernel(self, single_call_timing = False):
    n_pos = self.robot.get_num_pos()
    n_vel = self.robot.get_num_vel()
    func_params = ["d_Minv is a pointer to memory for the final result", \
                   "d_workspace is the L2-pinned global spill buffer (used when Minv-F overflows smem)", \
                   "d_q is the vector of joint positions", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_def = "void minv_kernel(T *d_Minv, unsigned char *d_workspace, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS){"
    func_notes = ["Outputs a SYMMETRIC_UPPER triangular matrix for Minv"]
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Compute the inverse of the mass matrix",func_notes,func_params,None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag, LAST
    # after RESOURCE_TIER so existing positional <T,TIER> call sites are unaffected;
    # default false if-constexpr-elides the epilogue -> byte-identical PTX.
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # 3-way pick: (perf_pick, lite_pick, minimal_pick). Each pick is 0 (no spill)
    # or 1 (surgical F-to-workspace). When picks collapse, emit a single body
    # (current behavior); when they diverge, emit if-constexpr branches.
    picks = getattr(self, "minv_spill_tier_3way", (0, 0, 0))
    self.gen_tier_dispatch(picks, lambda pick:
        _emit_minv_kernel_body_for_flags(self, n_vel, n_vel, bool(pick), single_call_timing))
    self.gen_add_end_function()

def gen_minv_host(self, mode = 0):
    # default is to do the full kernel call -- options are for single timing or compute only kernel wrapper
    single_call_timing = True if mode == 1 else False
    compute_only = True if mode == 2 else False

    # define function def and params
    func_params = ["hd_data is the packaged input and output pointers", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)", \
                   "streams are pointers to CUDA streams for async memory transfers (if needed)"]
    func_notes = []
    func_def_start = "void minv(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,"
    func_def_end =   "                 const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    # then generate the code
    self.gen_add_func_doc("Compute the inverse of the mass matrix",func_notes,func_params,None)
    # MUJOCO_OUTPUT (floating only) host flag, LAST: forwarded to the kernel launch
    # (naming the tier positionally to reach the trailing flag). Default false ->
    # byte-identical pin codegen.
    mjx_host = self.robot.floating_base
    if mjx_host:
        self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    else:
        self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"minv requires all-data or dynamics gridData\");")
    minv_kernel_tmpl = "minv_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "minv_kernel<T, RESOURCE_TIER>"
    func_call_start = minv_kernel_tmpl + "<<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_Minv,hd_data->d_workspace,hd_data->d_q,stride_q,"
    func_call_end = "d_robotModel,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("minv_kernel<","minv_kernel_single_timing<")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                 "int stride_q;", \
                                 "if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; " + \
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}", \
                                 "else {stride_q = 3*NUM_JOINTS; " + \
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}", \
                                 "gpuErrchkKernel();"])
    else:
        self.gen_add_code_line("int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;")
    # then compute
    self.gen_add_code_line("// then call the kernel")
    func_call = "if (USE_COMPRESSED_MEM) {" + func_call_start + func_call_end + "}"
    func_call2 = "else                    {" + func_call_start.replace("hd_data->d_q","hd_data->d_q_qd_u") + func_call_end + "}"
    func_call_code = [func_call, func_call2, "gpuErrchkKernel();"]
    # wrap function call in timing (if needed)
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"minv\", MINV_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        # then transfer memory back
        self.gen_add_code_lines(["// finally transfer the result back", \
                                 "gpuErrchk(cudaMemcpy(hd_data->h_Minv,hd_data->d_Minv,NUM_VEL*NUM_VEL*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchkKernel();"])
    # finally report out timing if requested
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("minv"))
    self.gen_add_end_function()

def gen_minv(self):
    # gen inner
    self.gen_minv_inner()
    # and device wrapper
    self.gen_minv_device()
    # and kernel wrappers
    self.gen_minv_kernel(True)
    self.gen_minv_kernel(False)
    # and host function call wrappers
    self.gen_minv_host(0)
    self.gen_minv_host(1)
    self.gen_minv_host(2)

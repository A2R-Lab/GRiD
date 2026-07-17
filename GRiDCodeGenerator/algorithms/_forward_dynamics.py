def gen_forward_dynamics_inner_F_size(self):
    """Phase 3b: size of Minv's F-region used inside forward_dynamics_inner
    (6 * NV * NV floats). Now always a separate `s_minv_F` parameter so the
    caller can keep it in smem (extra t_buffer) or spill it to L2-pinned
    workspace on humanoid-scale robots."""
    n = self.robot.get_num_vel()
    return 6 * n * n

def gen_forward_dynamics_inner_temp_mem_size(self, minv_f_in_smem = True):
        """s_temp arena = s_Minv (n*n, persistent) + max(Minv footprint during
        the Minv call, c+vaf+ID-inner after Minv). Inner-controlled placement:
        when minv_f_in_smem the Minv F-region (6*NV*NV) lives at the tail of the
        Minv sub-arena (so it's included here); when spilled it's in d_workspace
        and excluded. Caller sizes s_temp from FD_INNER_SMEM_BYTES<MINV_F_IN_SMEM>."""
        n = self.robot.get_num_pos()
        nv = self.robot.get_num_vel()
        # The post-Minv ID band is laid out s_c[n] | s_vaf[18*NJ] | ID-inner-temp
        # (see gen_forward_dynamics_inner: s_vaf at n*n+n, ID temp at n*n+n+18*NJ).
        # s_vaf is body-indexed (NJ bodies), so for mimic robots (NJ > n) reserve
        # 18*NJ here too; otherwise the ID inner's body-indexed f writes overflow
        # the band and corrupt the ID temp. Non-mimic keeps the legacy 19*n
        # (n s_c + 18*n s_vaf) byte-identical since NJ == n.
        NJ = self.robot.get_num_joints()
        id_band = (n + 18 * (NJ if self.robot_has_mimic_joints() else n)
                   + self.gen_inverse_dynamics_inner_temp_mem_size())
        minv_footprint = self.gen_minv_inner_no_F_size() + (6*nv*nv if minv_f_in_smem else 0)
        return n*n + max(minv_footprint, id_band)

def gen_forward_dynamics_finish_function_call(self, updated_var_names = None):
    var_names = dict( \
        s_qdd_name = "s_qdd", \
        s_u_name = "s_u", \
        s_c_name = "s_c", \
        s_Minv_name = "s_Minv"
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    code = "forward_dynamics_finish<T>(" + var_names["s_qdd_name"] + ", " + var_names["s_u_name"] + ", " + \
                                           var_names["s_c_name"] + ", " + var_names["s_Minv_name"] + ");"
    self.gen_add_code_line(code)

def gen_forward_dynamics_finish(self):
    n = self.robot.get_num_vel()
    # construct the boilerplate and function definition
    func_params = ["s_qdd is a pointer to memory for the final result", \
                   "s_u is the vector of joint input torques", \
                   "s_c is the bias vector", \
                   "s_Minv is the inverse mass matrix"]
    func_def = "void forward_dynamics_finish(T *s_qdd, const T *s_u, const T *s_c, const T *s_Minv) {"
    func_notes = ["Assumes s_Minv and s_c are already computed", 
                  "Does not internally sync the thread group, so it should be called after all threads have finished computing their values"]
    self.gen_add_func_doc("Finish the forward dynamics computation with qdd = Minv*(u-c)",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    # compute the final answer qdd = Minv * (u - c)
    # remember that Minv is an SYMMETRIC_UPPER triangular matrix
    self.gen_minv_apply(n, "s_qdd[row]", "(s_u[col] - s_c[col])",
                        loop_var = "row", comment_in_loop = True, negate = False)
    self.gen_add_end_function()

def gen_forward_dynamics_inner_function_call(self, updated_var_names = None,
                                             minv_f_in_smem_expr = "true"):
    var_names = dict( \
        s_q_name = "s_q", \
        s_qd_name = "s_qd", \
        s_qdd_name = "s_qdd", \
        s_u_name = "s_u", \
        s_temp_name = "s_temp", \
        d_workspace_name = "nullptr", \
        d_f_ext_name = "d_f_ext", \
        gravity_name = "gravity", \
        d_robotModel_name = "d_robotModel"
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    # Inner-controlled placement: forward_dynamics_inner is keyed on bool
    # MINV_F_IN_SMEM and slices the internal Minv F-region from s_temp (smem) or
    # d_workspace (global) itself. The caller passes both arenas + the placement;
    # sizes come from FD_INNER_{SMEM,WORKSPACE}_BYTES<T, MINV_F_IN_SMEM>().
    # runtime_joint_dynamics: forward d_robotModel (trailing defaulted = nullptr on
    # the inner) so the reused ID inner's u-site bias can read the mutable
    # damping/friction table. Every FD-inner caller (kernel / device / integrator
    # inner / fdsva_so kernels) already has d_robotModel in scope. Off-build relies
    # on the trailing default, keeping the emitted CUDA byte-identical.
    _fd_rt_jd = (", " + var_names["d_robotModel_name"]) if getattr(self, "runtime_joint_dynamics", False) else ""
    fd_code_start = "forward_dynamics_inner<T, " + minv_f_in_smem_expr + ">(" + var_names["s_qdd_name"] + ", " + var_names["s_q_name"] + ", " + \
                                                   var_names["s_qd_name"] + ", " + var_names["s_u_name"] + ", "
    fd_code_end = var_names["s_temp_name"] + ", " + var_names["d_workspace_name"] + ", " + var_names["d_f_ext_name"] + ", " + var_names["gravity_name"] + _fd_rt_jd + ");"
    fd_code_middle = self.gen_insert_helpers_function_call()
    fd_code = fd_code_start + fd_code_middle + fd_code_end
    self.gen_add_code_line(fd_code)

def gen_forward_dynamics_inner(self):
    n = self.robot.get_num_vel()
    NJ = self.robot.get_num_joints()
    # construct the boilerplate and function definition
    func_params = ["s_qdd is a pointer to memory for the final result", \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "s_u is the vector of joint input torques", \
                   "s_temp is the (shared) scratch; size FD_INNER_SMEM_BYTES<T, MINV_F_IN_SMEM>()", \
                   "d_workspace is the global scratch; size FD_INNER_WORKSPACE_BYTES<T, MINV_F_IN_SMEM>() (= 6*NV*NV when !MINV_F_IN_SMEM, else 0). Pass nullptr when MINV_F_IN_SMEM", \
                   "d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr", \
                   "gravity is the gravity constant"]
    func_def_start = "void forward_dynamics_inner(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, "
    # runtime_joint_dynamics: the reused ID inner's u-site bias reads the mutable
    # damping/friction table d_robotModel->d_joint_dynamics_params, so thread
    # d_robotModel in as a trailing defaulted (= nullptr) param and forward it to
    # the internal compute_c ID call. The default keeps the OFF build (and every
    # caller relying on the positional-to-gravity call) byte-identical.
    if getattr(self, "runtime_joint_dynamics", False):
        func_def_end = "T *s_temp, T *d_workspace, T *d_f_ext, const T gravity, const robotModel<T> *d_robotModel = nullptr) {"
    else:
        func_def_end = "T *s_temp, T *d_workspace, T *d_f_ext, const T gravity) {"
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -2)
    func_notes = ["CALLER CONTRACT (direct *_inner callers): s_XImats must ALREADY be populated for the current s_q -- the inner READS but never writes it. Call load_update_XImats_helpers(s_XImats, s_q, s_topology_helpers, d_robotModel, s_temp) then __syncthreads() first, or just call forward_dynamics_device (which does this for you). Skipping it reads uninitialized shared -> NaN (race-clean, initcheck-fixable).",
                  "CALLER CONTRACT (sizing): s_temp MUST be FD_INNER_SMEM_BYTES<T, MINV_F_IN_SMEM>() bytes. At MINV_F_IN_SMEM=true the 6*NV*NV Minv-F band lives in the TAIL of s_temp (the macro includes it); d_workspace is 0/nullptr but that does NOT mean the band is free -- it just moved into s_temp. Under-sizing s_temp (e.g. reusing a fewer-DoF constant) makes the inner read its own never-written band -> NaN, and the failure is DoF-specific because the band scales as 6*NV*NV.",
                  "Does not internally sync the thread group, so it should be called after all threads have finished computing their values",
                  "Inner-controlled placement: MINV_F_IN_SMEM selects where the internal Minv 6*NV*NV F-region lives (s_temp tail vs d_workspace). Decided here; caller sizes both arenas from FD_INNER_*_BYTES and hands both pointers in."]
    func_def = func_def_start + func_def_end
    # then generate the code
    self.gen_add_func_doc("Computes forward dynamics",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T, bool MINV_F_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # s_temp[0..n*n] = s_Minv (persistent). The internal Minv call gets the
    # sub-arena &s_temp[n*n] (its no_F region) + d_workspace; Minv slices its own
    # F-region (tail of the sub-arena when MINV_F_IN_SMEM, else d_workspace).
    # After Minv, &s_temp[n*n..] is reused for c+vaf+ID_inner.
    updated_var_names = dict(s_Minv_name = "s_temp",
                             s_temp_name = "&s_temp[" + str(n*n) + "]",
                             d_workspace_name = "d_workspace")
    self.gen_minv_inner_function_call(updated_var_names, f_in_smem_expr = "MINV_F_IN_SMEM")
    updated_var_names = dict(s_c_name = "&s_temp[" + str(n*n) + "]", s_vaf_name = "&s_temp[" + str(n*n + n) + "]", s_temp_name = "&s_temp[" + str(n*n + n + 18*NJ) + "]")
    # runtime_joint_dynamics: the compute_c ID inner is the FD bias; opt it into the
    # table-reading bias by forwarding d_robotModel (now a param of this inner).
    if getattr(self, "runtime_joint_dynamics", False):
        updated_var_names["d_robotModel_name"] = "d_robotModel"
    self.gen_inverse_dynamics_inner_function_call(compute_c = True, use_qdd_input = False, updated_var_names = updated_var_names)
    
    if self.DEBUG_MODE:
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_lines(["printf(\"Minv\\n\"); printMat<T," + str(n) + "," + str(n) + ">(s_temp," + str(n) + ");",
                                 "printf(\"u\\n\"); printMat<T,1," + str(n) + ">(s_u,1);"
                                 "printf(\"c\\n\"); printMat<T,1," + str(n) + ">(&s_temp[" + str(n*n) + "],1);"])
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    # finally compute the final answer qdd = Minv * (u - c)
    updated_var_names = dict(s_Minv_name = "s_temp", s_c_name = "&s_temp[" + str(n*n) + "]")
    self.gen_forward_dynamics_finish_function_call(updated_var_names)
    self.gen_add_end_function()

def gen_forward_dynamics_device(self):
    n = self.robot.get_num_vel()
    # Inline-CUDA device path. Tier-aware via tier_workspace_expr (mirrors
    # idsva_so_device / d2ee_device / inverse_dynamics_gradient_device): at TIER_SHARED the whole
    # FD inner s_temp arena lives in shared memory (with MINV_F at its tail);
    # at TIER_LITE/TIER_MINIMAL the WHOLE arena is routed to L2-pinned
    # d_workspace, freeing smem for the caller's outer kernel. The inner is
    # called with MINV_F_IN_SMEM=true in both cases because the arena pointer
    # (s_temp or d_workspace) already holds the F tail correctly.
    shared_mem_size = self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=True)
    func_params = ["s_qdd is a pointer to memory for the final result", \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "s_u is the vector of joint input torques", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr", \
                   "gravity is the gravity constant", \
                   "d_workspace is the global scratch buffer; size FORWARD_DYNAMICS_DEVICE_INLINE_WORKSPACE_BYTES<T, RESOURCE_TIER>() bytes (= 0 at TIER_SHARED, " + str(shared_mem_size) + "*sizeof(T) at TIER_LITE+). Pass nullptr at TIER_SHARED"]
    func_def_start = "void forward_dynamics_device(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, "
    func_def_end = "const robotModel<T> *d_robotModel, T *d_f_ext, const T gravity, T *d_workspace = nullptr) {"
    func_notes = ["Inline-CUDA users: at TIER_LITE/TIER_MINIMAL the whole FD inner scratch (~" + str(shared_mem_size) + "*sizeof(T) bytes) moves from shared memory to d_workspace, freeing smem for the caller's outer kernel.",
                  "The inner-temp arena holds the (6*NUM_VEL*NUM_VEL) Minv-F band at its tail, so routing the whole arena to d_workspace also spills the F-band; this is the device-path analog of the kernel's MINV_F_IN_SMEM lever (which surgically spills only F)."]
    func_def = func_def_start + func_def_end
    # then generate the code (shared device-wrapper skeleton; B+C §1.1).
    # Tier-aware arena: at TIER_SHARED s_temp lives in the smem arena; at
    # TIER_LITE/MINIMAL the s_temp slot is sourced from d_workspace and the
    # arena allocation skips it entirely (freeing ~120 KB on humanoid-scale
    # robots so an inline caller can fit FD into a 100 KB box). The XImats
    # helper writes its sincos scratch through s_temp, which has already been
    # repointed at the workspace under non-PERF tiers, so it stays valid.
    # The inner is called with MINV_F_IN_SMEM=true because the s_temp pointer
    # itself already routes per tier (smem at PERF, workspace at LITE+); the
    # inner's MINV_F lever stays "true" in both cases, treating s_temp as the
    # backing arena. d_workspace is not separately dereferenced by the inner
    # under this PERF=smem / non-PERF=workspace whole-arena routing.
    self.gen_device_wrapper(
        "Computes forward dynamics", func_def, shared_mem_size,
        lambda: self.gen_forward_dynamics_inner_function_call(
            minv_f_in_smem_expr = "true",
            updated_var_names = dict(d_workspace_name = "nullptr")),
        template_line = "template <typename T, int RESOURCE_TIER = TIER_SHARED>",
        func_notes = func_notes, func_params = func_params,
        include_linalg_scratch = True, tier_workspace_expr = "d_workspace")

def _emit_fd_kernel_body_for_flags(self, nq, nv, spill_minv_F, single_call_timing):
    """Emit forward_dynamics_kernel body for one tier's Minv-F spill flag.
    spill_minv_F=False: s_minv_F lives in extra smem (at start of s_temp);
    spill_minv_F=True:  s_minv_F lives in L2-pinned workspace."""
    # Inner-controlled: forward_dynamics_inner slices the Minv F itself from
    # s_temp (smem) or d_workspace (global) per MINV_F_IN_SMEM. The arena size
    # already reflects that choice.
    shared_mem_size = self.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem = not spill_minv_F)
    # Canonical per-timestep packing (mirrors id/crba/aba): q, qd, u each occupy a
    # NUM_JOINTS(=nq)-wide slot at stride 3*nq. The kernel loads the full 3*nq slot
    # and slices qd at nq, u at 2*nq. For a FIXED base nq==nv so 3*nq == 3*nv and
    # nq+2*nv (byte-identical); for a FLOATING base nq>nv a compressed nv-strided
    # layout would read a misaligned u (off by nq-nv) even at slot 0 and shifted
    # inputs for k>=1 -- the floating B=1 tau + batch bug.
    input_count = 3 * nq
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size, extra_t_buffers = [("s_q_qd_u", input_count), ("s_qdd", nv)], include_linalg_scratch=True)
    self.gen_add_code_line("T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[" + str(nq) + "]; T *s_u = &s_q_qd_u[" + str(2*nq) + "];")
    minv_f_expr = "false" if spill_minv_F else "true"
    if not single_call_timing:
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",block_level = True)
        self.gen_kernel_load_inputs("q_qd_u",str(input_count),stride="stride_q_qd_u")
        # mjx input convert (before XImats so X[0] uses the reordered quaternion):
        # quat wxyz->xyzw, qd[0:3]=R^T qd[0:3], u[0:3]=R^T u[0:3]. fd has no qdd
        # input (qdd is the OUTPUT) so qdd_name stays None.
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert(q_name="s_q", qd_name="s_qd", u_name="s_u")
            self.gen_add_end_control_flow()
        if spill_minv_F:
            self.gen_add_code_line("T *fd_d_workspace = reinterpret_cast<T *>(&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()]);")
        else:
            self.gen_add_code_line("(void)d_workspace;")
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_forward_dynamics_inner_function_call(
            updated_var_names = (dict(d_workspace_name = "fd_d_workspace") if spill_minv_F else None),
            minv_f_in_smem_expr = minv_f_expr)
        self.gen_add_sync()
        # mjx output: forward-dynamics accel qdd[0:3] = R (qdd[0:3] + omega x v_local);
        # s_qd is the PIN-frame velocity (converted above). The omega x v term is why
        # the acceleration is not a plain base rotation.
        if self.robot.floating_base:
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
        self.gen_kernel_load_inputs("q_qd_u",str(input_count))
        if spill_minv_F:
            self.gen_add_code_line("T *fd_d_workspace = reinterpret_cast<T *>(&d_workspace[GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()]);")
        else:
            self.gen_add_code_line("(void)d_workspace;")
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q_qd_u",str(input_count),feedback_from="qdd")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_forward_dynamics_inner_function_call(
            updated_var_names = (dict(d_workspace_name = "fd_d_workspace") if spill_minv_F else None),
            minv_f_in_smem_expr = minv_f_expr)
        self.gen_anti_licm_output_write("qdd")
        self.gen_add_end_control_flow()
        # save to global
        self.gen_kernel_save_result("qdd",str(nv))


def gen_forward_dynamics_kernel(self, single_call_timing = False):
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    func_params = ["d_qdd is a pointer to memory for the final result", \
                   "d_workspace is the L2-pinned global spill buffer (used when Minv-F overflows smem)", \
                   "d_q_qd_u is the vector of joint positions, velocities, and input torques", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "d_f_ext is the (optional) GLOBAL external forces, body-major 6*NUM_BODIES local-frame, or nullptr", \
                   "gravity is the gravity constant", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_def_start = "void forward_dynamics_kernel(T *d_qdd, unsigned char *d_workspace, const T *d_q_qd_u, const int stride_q_qd_u, "
    func_def_end = "T *d_f_ext, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_notes = []
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Computes forward dynamics",func_notes,func_params,None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag, LAST
    # after RESOURCE_TIER so existing positional <T,TIER> call sites are unaffected;
    # default false if-constexpr-elides the epilogue -> byte-identical PTX. The
    # epilogue (in _emit_fd_kernel_body_for_flags) sees MUJOCO_OUTPUT in scope.
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # Phase 3b: 3-way pick dispatch. Level 0 = Minv-F embedded in s_temp (extra
    # smem block); Level 1 = Minv-F in L2-pinned workspace.
    picks = getattr(self, "fd_spill_tier_3way", (0, 0, 0))
    self.gen_tier_dispatch(picks, lambda pick:
        _emit_fd_kernel_body_for_flags(self, nq, nv, bool(pick), single_call_timing))
    self.gen_add_end_function()


def gen_forward_dynamics_host(self, mode = 0):
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
    func_def_start = "void forward_dynamics(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
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
    # MUJOCO_OUTPUT (floating only) host flag, LAST: forwarded to the kernel launch
    # (naming the tier positionally to reach the trailing flag). Default false ->
    # byte-identical pin codegen.
    mjx_host = self.robot.floating_base
    if mjx_host:
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    else:
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"forward_dynamics requires all-data or dynamics gridData\");")
    fd_kernel_tmpl = "forward_dynamics_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "forward_dynamics_kernel<T, RESOURCE_TIER>"
    func_call_start = fd_kernel_tmpl + "<<<block_dimms,thread_dimms,FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_qdd,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd_u,"
    func_call_end = "hd_data->d_f_ext,d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("forward_dynamics_kernel<","forward_dynamics_kernel_single_timing<")
    self.gen_add_code_line("int stride_q_qd_u = 3*NUM_JOINTS;")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                 "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_u*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));", \
                                 "gpuErrchkKernel();"])
    # then compute
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    func_call_code = [func_call, "gpuErrchkKernel();"]
    # wrap function call in timing (if needed)
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"forward_dynamics\", FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
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
        self.gen_add_code_line(single_call_printf_line("forward_dynamics"))
    self.gen_add_end_function()

def gen_forward_dynamics(self):
    # first helpers
    self.gen_forward_dynamics_finish()
    self.gen_forward_dynamics_inner()
    # then device wrapper
    self.gen_forward_dynamics_device()
    # then kernels
    self.gen_forward_dynamics_kernel(True)
    self.gen_forward_dynamics_kernel(False)
    # then host launch
    self.gen_forward_dynamics_host(0)
    self.gen_forward_dynamics_host(1)
    self.gen_forward_dynamics_host(2)

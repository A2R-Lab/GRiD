"""FD parameter gradient dqdd/dpi = -Minv . Y CUDA emit (split out of _regressor.py 2026-10-03; the
section banner below is the design note). Shares the regressor basis helpers with `_regressor`.
"""

from ._regressor import _emit_mjx_base_rotate_rows_rowmajor
from grid_codegen.helpers._code_generation_helpers import gen_workspace_repoint_line, host_mode_flags, mangle_host_func_defs, wrap_host_single_call_timing
from grid_codegen.helpers._code_generation_helpers import gen_host_wrapper_head

# ===========================================================================
# FD parameter gradient  dqdd/dpi = -Minv . Y   (D.4 / differentiability §B)
# ---------------------------------------------------------------------------
# From M(pi) qdd + c(q,qd,pi) = u (u fixed), d/dpi gives
#     dqdd/dpi = -Minv . Y(q, qd, qdd_actual)
# because ID = M qdd + c is affine in pi with Jacobian Y at the *actual* qdd.
# Composes existing device inners: minv (Minv), inverse_dynamics (bias c
# -> qdd_actual via Minv.(u-c)), then the regressor Y at qdd_actual, then the
# symmetric-upper -Minv . Y apply. Output is nv x 10*NUM_BODIES. R2: it is a
# gridData field (hd_data->d_dqdd_dpi); the host launcher writes it + copies back.
# ===========================================================================

def gen_forward_dynamics_parameter_gradient_inner_temp_mem_size(self):
    n = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    # live footprint = max of the sub-step temps; the regressor inner temp is the
    # RNEA forward scratch (== id inner temp), minv inner temp is its own.
    minv_temp = self.gen_minv_inner_temp_mem_size()
    reg_temp = self.gen_inverse_dynamics_regressor_inner_temp_mem_size()
    id_temp = self.gen_inverse_dynamics_inner_temp_mem_size()
    return max(minv_temp, reg_temp, id_temp)


def gen_forward_dynamics_parameter_gradient_inner_function_call(self, updated_var_names=None):
    var_names = dict(
        s_dqdd_dpi_name="s_dqdd_dpi",
        s_Minv_name="s_Minv",
        s_Y_name="s_Y",
        s_qdd_name="s_qdd",
        s_vaf_name="s_vaf",
        s_c_name="s_c",
        s_q_name="s_q",
        s_qd_name="s_qd",
        s_u_name="s_u",
        s_temp_name="s_temp",
        gravity_name="gravity",
        d_robotModel_name="d_robotModel",
    )
    if updated_var_names is not None:
        for key, value in updated_var_names.items():
            var_names[key] = value
    code_start = "forward_dynamics_parameter_gradient_inner<T>(" + var_names["s_dqdd_dpi_name"] + ", " + \
        var_names["s_Minv_name"] + ", " + var_names["s_Y_name"] + ", " + \
        var_names["s_qdd_name"] + ", " + var_names["s_vaf_name"] + ", " + \
        var_names["s_c_name"] + ", " + var_names["s_q_name"] + ", " + \
        var_names["s_qd_name"] + ", " + var_names["s_u_name"] + ", "
    code_middle = self.gen_insert_helpers_function_call()
    # runtime_joint_dynamics: forward d_robotModel into the inner (trailing defaulted
    # param) so its reused ID bias can read the mutable table; omitted when off.
    _rt_jd = (", " + var_names["d_robotModel_name"]) if getattr(self, "runtime_joint_dynamics", False) else ""
    code_end = var_names["s_temp_name"] + ", " + var_names["gravity_name"] + _rt_jd + ");"
    self.gen_add_code_line(code_start + code_middle + code_end)


def gen_forward_dynamics_parameter_gradient_inner(self):
    n = self.robot.get_num_joints()
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()

    func_params = [
        "s_dqdd_dpi is the output FD param-gradient, row-major nv x 10*NUM_BODIES = " + str(nv * 10 * NB),
        "s_Minv is scratch of size NUM_VEL*NUM_VEL = " + str(nv * nv) + " (symmetric-upper Minv)",
        "s_Y is scratch for the regressor, row-major nv x 10*NUM_BODIES = " + str(nv * 10 * NB),
        "s_qdd is scratch of size NUM_VEL = " + str(nv) + " (recovered actual accelerations)",
        "s_vaf is scratch of size 18*NUM_JOINTS = " + str(18 * n) + " (RNEA v|a|f)",
        "s_c is scratch of size NUM_VEL = " + str(nv) + " (bias term)",
        "s_q is the vector of joint positions",
        "s_qd is the vector of joint velocities",
        "s_u is the vector of joint input torques",
        "s_temp is helper shared memory of size " + str(self.gen_forward_dynamics_parameter_gradient_inner_temp_mem_size()),
        "gravity is the gravity constant",
    ]
    func_notes = [
        "Assumes the XI matricies have already been updated for the given q",
        "dqdd/dpi = -Minv . Y(q,qd,qdd_actual) with qdd_actual = Minv.(u-c)",
    ]
    func_def_start = "void forward_dynamics_parameter_gradient_inner(T *s_dqdd_dpi, T *s_Minv, T *s_Y, T *s_qdd, T *s_vaf, T *s_c, const T *s_q, const T *s_qd, const T *s_u, "
    # runtime_joint_dynamics: the reused inverse_dynamics_inner bias reads
    # d_robotModel->d_joint_dynamics_params, so thread d_robotModel in as a trailing
    # defaulted param ONLY under that flag (byte-identical signature when off).
    if getattr(self, "runtime_joint_dynamics", False):
        func_def_end = "T *s_temp, const T gravity, const robotModel<T> *d_robotModel = nullptr) {"
    else:
        func_def_end = "T *s_temp, const T gravity) {"
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params("", func_params, -1)
    func_def = func_def_start + func_def_middle + func_def_end

    self.gen_add_func_doc("Compute the forward-dynamics param gradient dqdd/dpi = -Minv . Y",
                          func_notes, func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    # 1) Minv (symmetric-upper) via minv inner (F kept in smem).
    self.gen_add_code_line("// Minv = inv(M(q)) (symmetric-upper)")
    self.gen_minv_inner_function_call(f_in_smem_expr="true")
    self.gen_add_sync()

    # 2) bias c = ID(q, qd, qdd=0) via inverse_dynamics inner (compute_c, no qdd).
    self.gen_add_code_line("// bias c = ID(q, qd, 0)")
    # runtime_joint_dynamics: opt the reused ID inner into the table-reading bias by
    # forwarding the d_robotModel threaded into this inner (default name d_robotModel).
    _idcall_vn = dict(d_f_ext_name="nullptr")
    if getattr(self, "runtime_joint_dynamics", False):
        _idcall_vn["d_robotModel_name"] = "d_robotModel"
    self.gen_inverse_dynamics_inner_function_call(
        compute_c=True, use_qdd_input=False,
        updated_var_names=_idcall_vn)
    self.gen_add_sync()

    # 3) qdd_actual = Minv . (u - c)  (symmetric-upper Minv, like forward_dynamics_finish).
    self.gen_add_code_line("// qdd_actual = Minv . (u - c)")
    self.gen_forward_dynamics_finish_function_call(
        updated_var_names=dict(s_qdd_name="s_qdd", s_u_name="s_u", s_c_name="s_c", s_Minv_name="s_Minv"))
    self.gen_add_sync()

    # 4) Y = regressor(q, qd, qdd_actual)  -> s_Y (nv x 10*NB).
    self.gen_add_code_line("// Y = regressor(q, qd, qdd_actual)")
    self.gen_inverse_dynamics_regressor_inner_function_call(
        updated_var_names=dict(s_qdd_name="s_qdd"))
    self.gen_add_sync()

    # 5) dqdd_dpi = -Minv . Y. Minv is SYMMETRIC_UPPER (nv x nv); Y is row-major
    #    nv x 10*NB. One thread per output element (row, col).
    self.gen_add_code_line("// dqdd/dpi = -Minv . Y  (Minv symmetric-upper)")
    self.gen_add_parallel_loop("ind", str(nv * 10 * NB))
    self.gen_add_code_line("int row = ind / " + str(10 * NB) + "; int col = ind % " + str(10 * NB) + ";")
    self.gen_add_code_line("T val = static_cast<T>(0);")
    self.gen_add_code_line("for (int kk = 0; kk < " + str(nv) + "; kk++) {", True)
    self.gen_add_code_line("// account for the fact that Minv is a SYMMETRIC_UPPER triangular matrix")
    self.gen_add_code_line("int index = (row <= kk) * (kk * " + str(nv) + " + row) + (row > kk) * (row * " + str(nv) + " + kk);")
    self.gen_add_code_line("val += s_Minv[index] * s_Y[kk * " + str(10 * NB) + " + col];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("s_dqdd_dpi[ind] = -val;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()




def gen_forward_dynamics_parameter_gradient_device(self):
    n = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    func_params = [
        "s_dqdd_dpi is the output FD param-gradient (nv x 10*NUM_BODIES)",
        "s_q is the vector of joint positions",
        "s_qd is the vector of joint velocities",
        "s_u is the vector of joint input torques",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant",
    ]
    func_def = ("void forward_dynamics_parameter_gradient_device(T *s_dqdd_dpi, const T *s_q, const T *s_qd, const T *s_u, "
                "const robotModel<T> *d_robotModel, const T gravity) {")
    shared_mem_size = self.gen_forward_dynamics_parameter_gradient_inner_temp_mem_size()
    extra_t_buffers = [
        ("s_Minv", nv * nv), ("s_Y", nv * 10 * NB), ("s_qdd", nv),
        ("s_vaf", 18 * n), ("s_c", nv),
    ]
    self.gen_device_wrapper(
        "Compute the FD param gradient dqdd/dpi = -Minv . Y", func_def,
        shared_mem_size,
        lambda: self.gen_forward_dynamics_parameter_gradient_inner_function_call(),
        func_params=func_params,
        extra_t_buffers=extra_t_buffers, include_linalg_scratch=True)


def gen_forward_dynamics_parameter_gradient_kernel(self, single_call_timing=False):
    NUM_POS = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    out_size = nv * 10 * NB
    in_size = 3 * NUM_POS
    func_params = [
        "d_dqdd_dpi is the output FD param-gradient, row-major nv x 10*NUM_BODIES = " + str(out_size),
        "d_q_qd_u is the vector of joint positions, velocities, torques (q|qd|u)",
        "stride_q_qd_u is the stride between each (q, qd, u) triple",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant",
        "num_timesteps is the length of the trajectory points",
    ]
    # g1-spill: the kernel takes d_workspace as its 2nd argument. At a spilled tier
    # (FD_PARAMETER_GRADIENT_Y_IN_SMEM<TIER>()==false) the s_Y regressor scratch
    # lives in the L2-pinned d_workspace SO section instead of smem; at TIER_SHARED
    # (default) it stays in smem and d_workspace is unused. Default TIER keeps the
    # arena byte-identical, but the extra arg changes the signature -- the host
    # wrapper passes hd_data->d_workspace.
    func_def_start = "void forward_dynamics_parameter_gradient_kernel(T *d_dqdd_dpi, unsigned char *d_workspace, const T *d_q_qd_u, const int stride_q_qd_u, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    # MUJOCO_OUTPUT (floating only): dqdd/dpi = -Minv.Y is a covector-row matrix
    # (its base-linear ROWS transform like G.out). Because pi is the differentiation
    # variable (NOT a state), the qacc accel-couple drops -> NO omega x v term: the
    # input convert takes q,qd,u only (no qdd), and the output is a plain base-row
    # rotate. Flag added LAST (after RESOURCE_TIER); default false if-constexpr-
    # elides both -> byte-identical PTX.
    mjx_kernel = self.robot.floating_base
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_func_doc("Compute the FD param gradient dqdd/dpi = -Minv . Y", [], func_params, None)
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # g1-spill: s_Y is the LAST t_buffer; its arena slot is sized out_size at
    # TIER_SHARED and 0 at spilled tiers (the real s_Y is then routed to
    # d_workspace below). Sizing the slot via a per-tier constexpr keeps a single
    # arena declaration (all pointers stay in this scope) while shrinking the
    # smem footprint exactly to match FORWARD_DYNAMICS_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES.
    self.gen_add_code_line("constexpr bool REGRESSOR_Y_OUTPUT_IN_SMEM = FD_PARAMETER_GRADIENT_Y_IN_SMEM<RESOURCE_TIER>();")
    self.gen_add_code_line("constexpr int REGRESSOR_Y_OUTPUT_SLOT = REGRESSOR_Y_OUTPUT_IN_SMEM ? " + str(out_size) + " : 0;")
    extra_t_buffers = [
        ("s_q_qd_u", in_size), ("s_dqdd_dpi", out_size), ("s_Minv", nv * nv),
        ("s_qdd", nv), ("s_vaf", 18 * NUM_POS), ("s_c", nv), ("s_Y", "REGRESSOR_Y_OUTPUT_SLOT"),
    ]
    shared_mem_size = self.gen_forward_dynamics_parameter_gradient_inner_temp_mem_size()
    self.gen_XImats_helpers_temp_shared_memory_code(
        shared_mem_size, extra_t_buffers=extra_t_buffers, include_linalg_scratch=True)
    self.gen_add_code_line("if constexpr (REGRESSOR_Y_OUTPUT_IN_SMEM) { (void)d_workspace; }")
    self.gen_add_code_line("T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[" + str(NUM_POS) +
                           "]; T *s_u = &s_q_qd_u[" + str(2 * NUM_POS) + "];")

    def _repoint_spilled_Y(in_timestep_loop):
        # When spilled, repoint s_Y at the L2-pinned d_workspace SO section
        # (per-timestep slot; reused safely -- fd_param never runs concurrently with
        # the SO kernels). Emitted where `k` is in scope for the batched path.
        self.gen_add_code_line("if constexpr (!REGRESSOR_Y_OUTPUT_IN_SMEM) {", True)
        self.gen_add_code_line(gen_workspace_repoint_line("s_Y", "GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()", batch_indexed=in_timestep_loop))
        self.gen_add_end_control_flow()

    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q_qd_u", str(in_size), stride="stride_q_qd_u")
        # mjx input convert (q,qd,u; NO qdd -- pi-gradient is accel-couple-free)
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert(u_name="s_u")
            self.gen_add_end_control_flow()
        _repoint_spilled_Y(in_timestep_loop=True)
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_forward_dynamics_parameter_gradient_inner_function_call()
        self.gen_add_sync()
        # mjx output: -Minv.Y is a covector-row matrix -> base-linear ROWS rotate by R
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            _emit_mjx_base_rotate_rows_rowmajor(self, "s_dqdd_dpi", nv, 10 * NB)
            self.gen_add_end_control_flow()
        self.gen_kernel_save_result("dqdd_dpi", str(out_size), stride=str(out_size))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q_qd_u", str(in_size))
        _repoint_spilled_Y(in_timestep_loop=False)
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_forward_dynamics_parameter_gradient_inner_function_call()
        self.gen_add_sync()
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("dqdd_dpi", str(out_size))
    self.gen_add_end_function()


def gen_forward_dynamics_parameter_gradient_host(self, mode=0):
    single_call_timing, compute_only = host_mode_flags(mode)
    nv = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    out_size = nv * 10 * NB
    func_params = [
        "hd_data is the packaged input and output pointers (q/qd/u inputs; output written to hd_data->d_dqdd_dpi, 10*NB*nv*num_timesteps floats)",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant",
        "num_timesteps is the length of the trajectory points",
        "streams are pointers to CUDA streams for async memory transfers",
    ]
    func_def_start = "void forward_dynamics_parameter_gradient(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end = "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    func_def_start, func_def_end = mangle_host_func_defs(func_def_start, func_def_end, single_call_timing, compute_only)
    # MUJOCO_OUTPUT (floating only) host template flag, forwarded to the kernel
    # launch (names RESOURCE_TIER positionally to reach the trailing flag). Added
    # LAST so existing positional call sites don't rebind.
    self.gen_add_func_doc("Compute the FD param gradient dqdd/dpi = -Minv . Y", [], func_params, None)
    mjx_host = gen_host_wrapper_head(self, "forward_dynamics_parameter_gradient", func_def_start, func_def_end, kind_rule="dynamics")
    # g1-spill: pass hd_data->d_workspace as the kernel's 2nd arg. At the spilled
    # default tier (s_Y in d_workspace) it is read; at TIER_SHARED it is unused.
    kernel_tmpl = "forward_dynamics_parameter_gradient_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "forward_dynamics_parameter_gradient_kernel<T, RESOURCE_TIER>"
    func_call_start = kernel_tmpl + "<<<block_dimms,thread_dimms,FORWARD_DYNAMICS_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dqdd_dpi,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd_u,"
    func_call_end = "d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("forward_dynamics_parameter_gradient_kernel<", "forward_dynamics_parameter_gradient_kernel_single_timing<")
    self.gen_add_code_line("int stride_q_qd_u = Q_QD_U_STRIDE;")
    if not compute_only:
        self.gen_add_code_lines([
            "// start code with memory transfer",
            "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_u*" +
            ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));",
            "gpuErrchkKernel();",
        ])
    self.gen_add_code_line("// then call the kernel")
    # g1-spill: L2-pin d_workspace when the default tier spills s_Y into it.
    if not single_call_timing:
        self.gen_add_workspace_slot_count()
    ws_bytes = ("GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()" if single_call_timing
                else "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(_grid_ws_n)")
    self.gen_add_code_line("if (!FD_PARAMETER_GRADIENT_Y_IN_SMEM<RESOURCE_TIER>() && hd_data->d_workspace != nullptr) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, " + ws_bytes + "));}")
    func_call_code = [func_call_start + func_call_end]
    if single_call_timing:
        wrap_host_single_call_timing(func_call_code, kernel_errcheck=True)
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"forward_dynamics_parameter_gradient\", FORWARD_DYNAMICS_PARAMETER_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    if single_call_timing:
        self.gen_add_code_lines(func_call_code)
    else:
        self.gen_add_workspace_clamped_launch(func_call_code, emit_count = False)
    self.gen_add_code_line("gpuErrchkKernel();")
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back into the gridData host buffer (hd_data->d_dqdd_dpi -> hd_data->h_dqdd_dpi)",
            "gpuErrchk(cudaMemcpy(hd_data->h_dqdd_dpi,hd_data->d_dqdd_dpi," +
            ("num_timesteps*" if not single_call_timing else "") + str(out_size) + "*sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();",
        ])
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("forward_dynamics_parameter_gradient"))
    self.gen_add_end_function()


def gen_forward_dynamics_parameter_gradient(self):
    # inner -> device -> kernel(s) -> host(s)
    self.gen_forward_dynamics_parameter_gradient_inner()
    self.gen_forward_dynamics_parameter_gradient_device()
    self.gen_forward_dynamics_parameter_gradient_kernel(single_call_timing=False)
    self.gen_forward_dynamics_parameter_gradient_kernel(single_call_timing=True)
    self.gen_forward_dynamics_parameter_gradient_host(0)
    self.gen_forward_dynamics_parameter_gradient_host(1)
    self.gen_forward_dynamics_parameter_gradient_host(2)

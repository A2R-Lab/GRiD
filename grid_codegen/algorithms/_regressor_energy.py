"""Kinetic / potential energy regressor CUDA emit (split out of _regressor.py 2026-10-03; the
section banner below is the design note). Shares `_emit_dI_times_v` with `_regressor`.
"""

from ._frame_jacobian import _emit_world_transform_chainup
from ._regressor import _emit_dI_times_v
from grid_codegen.helpers._code_generation_helpers import host_q_input_transfer_lines, host_mode_flags, host_q_qd_input_transfer_lines, mangle_host_func_defs, wrap_host_single_call_timing
from grid_codegen.helpers._code_generation_helpers import gen_host_wrapper_head

# ===========================================================================
# Energy regressors (sysID): kinetic + potential energy linear in pi.
# ---------------------------------------------------------------------------
# Both regressors are length 10*NUM_BODIES row vectors (per-body 10 inertial
# params pi_i = [m, h(3), I_O(6)]); NOT nv x 10*NB (no backward DoF sweep).
#
#   KE = sum_i 1/2 v_i^T I_i v_i  =>  y_KE[10*i+k] = 1/2 v_i^T (dI_k) v_i
#        (spatial / s_XImats domain; v_i from the RNEA forward sweep)
#   PE = -sum_i g . (m_i p_i + R_i h_i), g = [0,0,GRAVITY]  =>  per body i only
#        4 nonzero cols: y[10i+0] = -(g . p_i) ; y[10i+1:4] = -(R_i^T g);
#        the six inertia cols are identically zero. (kinematics / s_XmatsHom
#        domain; (R_i, p_i) from the world-transform BFS chain-up).
#
# MIMIC: safe case, no gate. s_vaf is sized 18*NUM_POS and every per-body buffer
# + output loop is sized by NUM_BODIES (never nv). PE reuses the already
# mimic-aware s_XmatsHom (effective-angle q-fold baked upstream). Floating-base:
# KE flows through the RNEA floating-root branch; PE's world chain-up handles the
# floating root. Output is <= 10*NB floats -> fits every tier (no spill).
# Matches RBDReference.kinetic_energy_regressor / potential_energy_regressor and
# the identities y_KE.pi == kinetic_energy, y_PE.pi == potential_energy.
# ===========================================================================

def gen_kinetic_energy_regressor_inner_temp_mem_size(self):
    # forward RNEA scratch is the max live footprint (same as the joint-torque
    # regressor); the per-column 1/2 v^T dI v reduction is thread-local registers.
    return self.gen_inverse_dynamics_inner_temp_mem_size()


def gen_kinetic_energy_regressor_inner_function_call(self, updated_var_names=None):
    var_names = dict(
        s_y_name="s_y_ke",
        s_vaf_name="s_vaf",
        s_q_name="s_q",
        s_qd_name="s_qd",
        s_temp_name="s_temp",
        gravity_name="gravity",
        d_robotModel_name="d_robotModel",
    )
    if updated_var_names is not None:
        for key, value in updated_var_names.items():
            var_names[key] = value
    code_start = "kinetic_energy_regressor_inner<T>(" + var_names["s_y_name"] + ", " + \
        var_names["s_vaf_name"] + ", " + var_names["s_q_name"] + ", " + \
        var_names["s_qd_name"] + ", "
    code_middle = self.gen_insert_helpers_function_call()
    # runtime_joint_dynamics: forward d_robotModel into the inner (trailing defaulted
    # param) so its reused ID bias can read the mutable table; omitted when off.
    _rt_jd = (", " + var_names["d_robotModel_name"]) if getattr(self, "runtime_joint_dynamics", False) else ""
    code_end = var_names["s_temp_name"] + ", " + var_names["gravity_name"] + _rt_jd + ");"
    self.gen_add_code_line(code_start + code_middle + code_end)


def gen_kinetic_energy_regressor_inner(self):
    n = self.robot.get_num_joints()
    NB = self.robot.get_num_bodies()

    func_params = [
        "s_y_ke is the output kinetic-energy regressor, length 10*NUM_BODIES = " + str(10 * NB),
        "s_vaf is scratch of size 18*NUM_JOINTS = " + str(18 * n) + " (RNEA v|a|f; only v consumed)",
        "s_q is the vector of joint positions",
        "s_qd is the vector of joint velocities",
        "s_temp is helper shared memory of size " + str(self.gen_kinetic_energy_regressor_inner_temp_mem_size()),
        "gravity is the gravity constant (unused; KE is gravity-independent)",
    ]
    func_notes = [
        "Assumes the XI matricies have already been updated for the given q",
        "KE = y_KE . pi ; y_KE[10*i+k] = 1/2 v_i^T (dI_k) v_i  (pi_i = [m, m*c(3), I_O(6)])",
    ]
    func_def_start = "void kinetic_energy_regressor_inner(T *s_y_ke, T *s_vaf, const T *s_q, const T *s_qd, "
    # runtime_joint_dynamics: the reused inverse_dynamics_inner bias reads
    # d_robotModel->d_joint_dynamics_params, so thread d_robotModel in as a trailing
    # defaulted param ONLY under that flag (byte-identical signature when off). The KE
    # regressor consumes only s_vaf v|a, so the added bias on the scratch torque is
    # discarded, but d_robotModel must be in scope for the read to compile.
    if getattr(self, "runtime_joint_dynamics", False):
        func_def_end = "T *s_temp, const T gravity, const robotModel<T> *d_robotModel = nullptr) {"
    else:
        func_def_end = "T *s_temp, const T gravity) {"
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params("", func_params, -1)
    func_def = func_def_start + func_def_middle + func_def_end

    self.gen_add_func_doc("Compute the kinetic-energy regressor y_KE (KE = y_KE . pi)",
                          func_notes, func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("(void)gravity;")

    # 1) RNEA forward sweep -> s_vaf v|a (we only consume v). qdd is irrelevant to
    #    v, so reuse the bias variant (compute_c=True, use_qdd_input=False).
    self.gen_add_code_line("// forward RNEA sweep: populate s_vaf v (reuse RNEA vaf inner; qdd irrelevant to v)")
    _ke_vn = dict(s_c_name="s_temp", d_f_ext_name="nullptr")
    if getattr(self, "runtime_joint_dynamics", False):
        _ke_vn["d_robotModel_name"] = "d_robotModel"
    self.gen_inverse_dynamics_inner_function_call(
        compute_c=True, use_qdd_input=False,
        updated_var_names=_ke_vn)
    self.gen_add_sync()

    # 2) P2-fan over the 10*NB columns. col -> (link i, param k); each thread writes
    #    one DISTINCT output (no collisions): y[col] = 1/2 v_i^T (dI_k) v_i.
    self.gen_add_code_line("// y_KE[10*i+k] = 1/2 v_i^T (dI_k) v_i over all 10*NUM_BODIES columns")
    self.gen_add_parallel_loop("col", str(10 * NB))
    self.gen_add_code_line("int link_i = col / 10; int param_k = col % 10;")
    self.gen_add_code_line("const T *v_i = &s_vaf[6*link_i];")
    self.gen_add_code_line("T dIv[6];")
    self.gen_add_code_line("switch (param_k) {", True)
    for k in range(10):
        self.gen_add_code_line("case " + str(k) + ": {", True)
        _emit_dI_times_v(self, "dIv", k, "v_i")
        self.gen_add_code_line("break;")
        self.gen_add_end_control_flow()
    self.gen_add_code_line("default: { for (int r=0;r<6;r++){dIv[r]=static_cast<T>(0);} }")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("T dot = static_cast<T>(0); for (int r=0;r<6;r++){ dot += v_i[r]*dIv[r]; }")
    self.gen_add_code_line("s_y_ke[col] = static_cast<T>(0.5) * dot;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_kinetic_energy_regressor_device(self):
    n = self.robot.get_num_pos()
    func_params = [
        "s_y_ke is the output kinetic-energy regressor (length 10*NUM_BODIES)",
        "s_q is the vector of joint positions",
        "s_qd is the vector of joint velocities",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant (unused)",
    ]
    func_def = ("void kinetic_energy_regressor_device(T *s_y_ke, const T *s_q, const T *s_qd, "
                "const robotModel<T> *d_robotModel, const T gravity) {")
    shared_mem_size = self.gen_kinetic_energy_regressor_inner_temp_mem_size()
    extra_t_buffers = [("s_vaf", 18 * n)]
    self.gen_device_wrapper(
        "Compute the kinetic-energy regressor y_KE (KE = y_KE . pi)", func_def,
        shared_mem_size,
        lambda: self.gen_kinetic_energy_regressor_inner_function_call(),
        func_params=func_params,
        extra_t_buffers=extra_t_buffers, include_linalg_scratch=True)


def gen_kinetic_energy_regressor_kernel(self, single_call_timing=False):
    NUM_POS = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    out_size = 10 * NB
    in_size = NUM_POS + nv
    func_params = [
        "d_y_ke is the output kinetic-energy regressor, length 10*NUM_BODIES = " + str(out_size),
        "d_q_qd is the vector of joint positions, velocities (q|qd)",
        "stride_q_qd is the stride between each (q, qd) pair",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant (unused)",
        "num_timesteps is the length of the trajectory points",
    ]
    func_def_start = "void kinetic_energy_regressor_kernel(T *d_y_ke, const T *d_q_qd, const int stride_q_qd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    # MUJOCO_OUTPUT (floating only): the KE regressor is INVARIANT (it lives in
    # param space; KE = y_KE.pi is a frame-invariant scalar) -> NO output epilogue.
    # Only the base velocity input needs converting (KE reads qd via the RNEA v
    # sweep): input_convert q,qd. Flag added LAST so positional call sites are safe;
    # default false if-constexpr-elides the input convert -> byte-identical PTX.
    mjx_kernel = self.robot.floating_base
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_func_doc("Compute the kinetic-energy regressor", [], func_params, None)
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    extra_t_buffers = [("s_q_qd", in_size), ("s_y_ke", out_size), ("s_vaf", 18 * NUM_POS)]
    shared_mem_size = self.gen_kinetic_energy_regressor_inner_temp_mem_size()
    self.gen_XImats_helpers_temp_shared_memory_code(
        shared_mem_size, extra_t_buffers=extra_t_buffers, include_linalg_scratch=True)
    self.gen_add_code_line("T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(NUM_POS) + "];")
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q_qd", str(in_size), stride="stride_q_qd")
        # mjx input convert (before XImats so X[0] uses the reordered quaternion)
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert()
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_kinetic_energy_regressor_inner_function_call()
        self.gen_add_sync()
        self.gen_kernel_save_result("y_ke", str(out_size), stride=str(out_size))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q_qd", str(in_size))
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_kinetic_energy_regressor_inner_function_call()
        self.gen_add_sync()
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("y_ke", str(out_size))
    self.gen_add_end_function()


def gen_kinetic_energy_regressor_host(self, mode=0):
    single_call_timing, compute_only = host_mode_flags(mode)
    NB = self.robot.get_num_bodies()
    out_size = 10 * NB
    func_params = [
        "hd_data is the packaged input and output pointers (q/qd inputs; output written to hd_data->d_ke_regressor, 10*NB*num_timesteps floats)",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant (unused)",
        "num_timesteps is the length of the trajectory points",
        "streams are pointers to CUDA streams for async memory transfers",
    ]
    func_def_start = "void kinetic_energy_regressor(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end = "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    func_def_start, func_def_end = mangle_host_func_defs(func_def_start, func_def_end, single_call_timing, compute_only)
    # MUJOCO_OUTPUT (floating only) host template flag, forwarded to the kernel
    # launch (names RESOURCE_TIER positionally to reach the trailing flag). KE is
    # invariant -> only the kernel's input-convert changes; no host post-process.
    self.gen_add_func_doc("Compute the kinetic-energy regressor y_KE (KE = y_KE . pi)", [], func_params, None)
    mjx_host = gen_host_wrapper_head(self, "kinetic_energy_regressor", func_def_start, func_def_end, kind_rule="dynamics")
    kernel_tmpl = "kinetic_energy_regressor_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "kinetic_energy_regressor_kernel<T, RESOURCE_TIER>"
    func_call_start = kernel_tmpl + "<<<block_dimms,thread_dimms,KINETIC_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_ke_regressor,hd_data->d_q_qd,stride_q_qd,"
    func_call_end = "d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("kinetic_energy_regressor_kernel<", "kinetic_energy_regressor_kernel_single_timing<")
    if not compute_only:
        self.gen_add_code_lines(host_q_qd_input_transfer_lines(single_call_timing))
    else:
        self.gen_add_code_line("int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS : 3*NUM_JOINTS;")
    self.gen_add_code_line("// then call the kernel")
    func_call_mem = "if (USE_COMPRESSED_MEM) {" + func_call_start + func_call_end + "}"
    func_call_mem2 = "else                    {" + (func_call_start + func_call_end).replace("hd_data->d_q_qd", "hd_data->d_q_qd_u") + "}"
    func_call_code = [func_call_mem, func_call_mem2]
    if single_call_timing:
        wrap_host_single_call_timing(func_call_code, kernel_errcheck=True)
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"kinetic_energy_regressor\", KINETIC_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>()));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back into the gridData host buffer (hd_data->d_ke_regressor -> hd_data->h_ke_regressor)",
            "gpuErrchk(cudaMemcpy(hd_data->h_ke_regressor,hd_data->d_ke_regressor," +
            ("num_timesteps*" if not single_call_timing else "") + str(out_size) + "*sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();",
        ])
    else:
        self.gen_add_code_line("gpuErrchkKernel();")
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("kinetic_energy_regressor"))
    self.gen_add_end_function()


def gen_kinetic_energy_regressor(self):
    self.gen_kinetic_energy_regressor_inner()
    self.gen_kinetic_energy_regressor_device()
    self.gen_kinetic_energy_regressor_kernel(single_call_timing=False)
    self.gen_kinetic_energy_regressor_kernel(single_call_timing=True)
    self.gen_kinetic_energy_regressor_host(0)
    self.gen_kinetic_energy_regressor_host(1)
    self.gen_kinetic_energy_regressor_host(2)


# ---------------------------------------------------------------------------
# Potential-energy regressor (kinematics / s_XmatsHom domain).
# ---------------------------------------------------------------------------

def _potential_energy_regressor_inner_temp_mem_size(self):
    # world homogeneous transform per joint (BFS chain-up scratch).
    return 16 * self.robot.get_num_joints()


def gen_potential_energy_regressor_inner_function_call(self, updated_var_names=None):
    var_names = dict(
        s_y_name="s_y_pe",
        s_q_name="s_q",
        s_Xhom_name="s_XmatsHom",
        s_temp_name="s_temp",
        gravity_name="gravity",
    )
    if updated_var_names is not None:
        for key, value in updated_var_names.items():
            var_names[key] = value
    self.gen_add_code_line(
        "potential_energy_regressor_inner<T>(" + var_names["s_y_name"] + ", " +
        var_names["s_q_name"] + ", " + var_names["s_Xhom_name"] +
        ", d_robotModel, " + var_names["s_temp_name"] + ", " + var_names["gravity_name"] + ");")


def gen_potential_energy_regressor_inner(self):
    NB = self.robot.get_num_bodies()
    NJ = self.robot.get_num_joints()
    n_bfs_levels = self.robot.get_max_bfs_level() + 1

    func_params = [
        "s_y_pe is the output potential-energy regressor, length 10*NUM_BODIES = " + str(10 * NB),
        "s_q is the vector of joint positions (unused; q is baked into s_Xhom)",
        "s_Xhom is the per-joint LOCAL homogeneous transforms (mimic-aware)",
        "d_robotModel is the GPU model helpers (unused; PE needs only kinematics)",
        "s_temp is scratch of size " + str(_potential_energy_regressor_inner_temp_mem_size(self)),
        "gravity is the gravity constant (g = [0,0,gravity])",
    ]
    func_notes = [
        "Assumes the homogeneous transforms s_Xhom have been updated for the given q",
        "PE = y_PE . pi ; per body i only the mass + 3 first-moment cols are nonzero:",
        "  y[10i+0] = -(g . p_i) ; y[10i+1:4] = -(R_i^T g) ; inertia cols = 0",
    ]
    func_def = ("void potential_energy_regressor_inner(T *s_y_pe, const T *s_q, const T *s_Xhom, "
                "const robotModel<T> *d_robotModel, T *s_temp, const T gravity) {")
    self.gen_add_func_doc("Compute the potential-energy regressor y_PE (PE = y_PE . pi)",
                          func_notes, func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("(void)s_q; (void)d_robotModel;")
    self.gen_add_code_line("T *s_Xworld = s_temp;   // 16 * NUM_JOINTS world homogeneous transforms")

    # Step 1: world homogeneous transforms by BFS level (chain-up of local s_Xhom).
    # Mirrors centroidal Step-1 -> use s_Xworld, do NOT rebuild from spatial.
    _emit_world_transform_chainup(self, declare_xworld = False)

    # Step 2: P2-fan over 10*NB columns. Zero every col, then fill the 4 nonzeros
    #   per body from (R_i, p_i): g = [0,0,gravity], so
    #     y[10i+0] = -(g.p) = -gravity * p_z
    #     y[10i+1+k] = -(R^T g)[k] = -gravity * R[2,k]  (k=0..2)
    #   World R is column-major in s_Xworld (R[r,c] = s_Xworld[16*i + r + 4*c]);
    #   p = s_Xworld[16*i + 12..14].  -> R[2,k] = s_Xworld[16*i + 2 + 4*k].
    self.gen_add_code_line("// Step 2: zero all cols then fill the 4 nonzero cols per body")
    self.gen_add_parallel_loop("col", str(10 * NB))
    self.gen_add_code_line("int link_i = col / 10; int param_k = col % 10;")
    self.gen_add_code_line("const T *Xw = &s_Xworld[16*link_i];")
    self.gen_add_code_line("if (param_k == 0) { s_y_pe[col] = -gravity * Xw[14]; }")
    self.gen_add_code_line("else if (param_k <= 3) { s_y_pe[col] = -gravity * Xw[2 + 4*(param_k-1)]; }")
    self.gen_add_code_line("else { s_y_pe[col] = static_cast<T>(0); }")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_potential_energy_regressor_device(self):
    func_params = [
        "s_y_pe is the output potential-energy regressor (length 10*NUM_BODIES)",
        "s_q is the vector of joint positions",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant",
    ]
    func_def = ("void potential_energy_regressor_device(T *s_y_pe, const T *s_q, "
                "const robotModel<T> *d_robotModel, const T gravity) {")
    self.gen_add_func_doc("Compute the potential-energy regressor y_PE (PE = y_PE . pi)", [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _potential_energy_regressor_inner_temp_mem_size(self), extra_t_buffers=[],
        include_linalg_scratch=True, linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_potential_energy_regressor_inner_function_call()
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_potential_energy_regressor_kernel(self, single_call_timing=False):
    NUM_POS = self.robot.get_num_pos()
    NB = self.robot.get_num_bodies()
    out_size = 10 * NB
    in_size = NUM_POS
    func_params = [
        "d_y_pe is the output potential-energy regressor, length 10*NUM_BODIES = " + str(out_size),
        "d_q is the vector of joint positions",
        "stride_q is the stride between each q",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant",
        "num_timesteps is the length of the trajectory points",
    ]
    func_def_start = "void potential_energy_regressor_kernel(T *d_y_pe, const T *d_q, const int stride_q, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    # MUJOCO_OUTPUT (floating only): the PE regressor is INVARIANT (param space) ->
    # NO output epilogue. PE reads only q; the only mjx difference is the base
    # quaternion order (wxyz->xyzw) so the world-transform BFS builds the right R.
    # quat_reorder BEFORE the XmatsHom build. Flag added LAST; default false
    # if-constexpr-elides the reorder -> byte-identical PTX.
    mjx_kernel = self.robot.floating_base
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_func_doc("Compute the potential-energy regressor", [], func_params, None)
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    extra_t_buffers = [("s_q", in_size), ("s_y_pe", out_size)]
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _potential_energy_regressor_inner_temp_mem_size(self), extra_t_buffers=extra_t_buffers,
        include_linalg_scratch=True, linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q", str(in_size), stride="stride_q")
        # mjx input convert (quat reorder only; before XmatsHom so R is built right)
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_quat_reorder()
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute")
        self.gen_load_update_XmatsHom_helpers_function_call()
        self.gen_potential_energy_regressor_inner_function_call()
        self.gen_add_sync()
        self.gen_kernel_save_result("y_pe", str(out_size), stride=str(out_size))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q", str(in_size))
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_load_update_XmatsHom_helpers_function_call()
        self.gen_potential_energy_regressor_inner_function_call()
        self.gen_add_sync()
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("y_pe", str(out_size))
    self.gen_add_end_function()


def gen_potential_energy_regressor_host(self, mode=0):
    single_call_timing, compute_only = host_mode_flags(mode)
    NB = self.robot.get_num_bodies()
    out_size = 10 * NB
    func_params = [
        "hd_data is the packaged input and output pointers (q input; output written to hd_data->d_pe_regressor, 10*NB*num_timesteps floats)",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant",
        "num_timesteps is the length of the trajectory points",
        "streams are pointers to CUDA streams for async memory transfers",
    ]
    func_def_start = "void potential_energy_regressor(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end = "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    func_def_start, func_def_end = mangle_host_func_defs(func_def_start, func_def_end, single_call_timing, compute_only)
    # MUJOCO_OUTPUT (floating only) host template flag, forwarded to the kernel
    # launch (names RESOURCE_TIER positionally to reach the trailing flag). PE is
    # invariant -> only the kernel's quat-reorder changes; no host post-process.
    self.gen_add_func_doc("Compute the potential-energy regressor y_PE (PE = y_PE . pi)", [], func_params, None)
    mjx_host = gen_host_wrapper_head(self, "potential_energy_regressor", func_def_start, func_def_end, kind_rule="kinematics")
    kernel_tmpl = "potential_energy_regressor_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "potential_energy_regressor_kernel<T, RESOURCE_TIER>"
    func_call_start = kernel_tmpl + "<<<block_dimms,thread_dimms,POTENTIAL_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>()>>>(hd_data->d_pe_regressor,hd_data->d_q,stride_q,"
    func_call_end = "d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("potential_energy_regressor_kernel<", "potential_energy_regressor_kernel_single_timing<")
    if not compute_only:
        self.gen_add_code_lines(host_q_input_transfer_lines(single_call_timing))
    else:
        self.gen_add_code_line("int stride_q = NUM_JOINTS;")
    self.gen_add_code_line("// then call the kernel")
    func_call_code = [func_call_start + func_call_end]
    if single_call_timing:
        wrap_host_single_call_timing(func_call_code, kernel_errcheck=True)
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"potential_energy_regressor\", POTENTIAL_ENERGY_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>()));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back into the gridData host buffer (hd_data->d_pe_regressor -> hd_data->h_pe_regressor)",
            "gpuErrchk(cudaMemcpy(hd_data->h_pe_regressor,hd_data->d_pe_regressor," +
            ("num_timesteps*" if not single_call_timing else "") + str(out_size) + "*sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();",
        ])
    else:
        self.gen_add_code_line("gpuErrchkKernel();")
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("potential_energy_regressor"))
    self.gen_add_end_function()


def gen_potential_energy_regressor(self):
    self.gen_potential_energy_regressor_inner()
    self.gen_potential_energy_regressor_device()
    self.gen_potential_energy_regressor_kernel(single_call_timing=False)
    self.gen_potential_energy_regressor_kernel(single_call_timing=True)
    self.gen_potential_energy_regressor_host(0)
    self.gen_potential_energy_regressor_host(1)
    self.gen_potential_energy_regressor_host(2)

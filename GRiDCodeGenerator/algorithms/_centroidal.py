"""Centroidal / energy / CoM codegen (R1-R3 of the centroidal-quickwins plan).

NEW, additive algorithm families. Nothing in the existing emit path is touched:
each family is gated on its grid:: deps being generated, mirroring how
`grid_plant` gates its costs.

Two domains:
  * R1 generalized_gravity / nonlinear_effects are RNEA wrappers in the spatial
    (s_XImats) domain — they reuse inverse_dynamics_inner<T>(compute_c, qdd=0).
  * com / jacobian_com / ccrba / energy live in the kinematics (s_XmatsHom)
    domain: one shared inner builds the world homogeneous transform of every
    body (BFS chain-up, like end_effector_pose), then per-body world spatial
    inertias + world body Jacobians, the world momentum map A0 = sum_j Iw_j J_j,
    the CoM (from the world composite inertia's first moment), and shifts/reorders
    to Pinocchio's [linear; angular]-at-CoM-world-aligned convention. This is a
    direct CUDA transcription of the merged RBDReference numpy oracles
    (RBDReference/_centroidal.py, RBDReference/_energy.py), so it agrees to
    float64 rounding with pinocchio's ccrba / centerOfMass / jacobianCenterOfMass
    / computeKineticEnergy / computePotentialEnergy.

Families emitted (each: device + kernel(timing + batch) + host(0/1/2)):
  generalized_gravity   g(q)    = RNEA(q, 0, 0)              (NUM_VEL)
  nonlinear_effects     c(q,qd) = RNEA(q, qd, 0)             (NUM_VEL)
  com                   p_com (3)  +  J_com (3 x NUM_VEL)    -> d_com (3 + 3*NV)
  ccrba                 A (6 x NUM_VEL) + h (6)              -> d_ccrba (6*NV + 6)
  energy                {KE, PE, mechanical}                 -> d_energy (3)
"""

import numpy as np


__all__ = [
    "gen_id_bias_device", "gen_id_bias_kernel", "gen_id_bias_host", "gen_id_bias",
    "gen_centroidal_inner", "gen_com_device", "gen_ccrba_device", "gen_energy_device",
    "_gen_kin_centroidal_kernel", "_gen_kin_centroidal_host",
    "gen_com", "gen_ccrba", "gen_energy",
]


# ===========================================================================
# R1 -- generalized gravity  g(q) = RNEA(q, 0, 0)
#       nonlinear effects    c(q,qd) = RNEA(q, qd, 0)
# Both reuse inverse_dynamics_inner<T>(compute_c=True, use_qdd_input=False).
# gravity pins qd to a zeroed buffer; nonlinear_effects passes the real qd.
# ===========================================================================

def _id_bias_inner_temp_mem_size(self):
    return 6 * self.robot.get_num_pos()


def gen_id_bias_device(self, gravity_only):
    n = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    name = "generalized_gravity" if gravity_only else "nonlinear_effects"
    func_params = ["s_out is the output bias torque (size NUM_VEL)",
                   "s_q is the vector of joint positions",
                   "s_qd is the vector of joint velocities (ignored for gravity)",
                   "d_robotModel is the GPU model helpers",
                   "gravity is the gravity constant"]
    func_def = ("void " + name + "_device(T *s_out, const T *s_q, const T *s_qd, "
                "const robotModel<T> *d_robotModel, const T gravity) {")
    desc = ("Compute the generalized gravity g(q) = RNEA(q, 0, 0)" if gravity_only
            else "Compute the nonlinear (bias) effects c(q,qd) = RNEA(q, qd, 0)")
    # s_vaf is consumed by inverse_dynamics_inner, which writes it body-indexed by
    # RAW body id (v/a/f blocks each span get_num_joints() bodies). For mimic robots
    # get_num_joints() (NB) > get_num_pos() (NV), so size it 18*NB or the high-body
    # f-writes overflow into the next arena region. Non-mimic keeps 18*n byte-identical.
    nb_vaf = self.robot.get_num_joints() if self.robot_has_mimic_joints() else n
    extra = [("s_vaf", 18 * nb_vaf)]
    if gravity_only:
        extra.append(("s_qd0", nv))
    # nonlinear_effects floating-base mjx path: a zeroed qdd buffer (nv). Occupies
    # the SAME arena slack as gravity's s_qd0 band (the host arena macro budgets +nv
    # for the bias kernels either way), so no GRiDCodeGenerator.py arena change.
    use_qdd = self.robot.floating_base and not gravity_only
    if use_qdd:
        extra.append(("s_qdd", nv))

    def _inner():
        if gravity_only:
            self.gen_add_parallel_loop("i", str(nv))
            self.gen_add_code_line("s_qd0[i] = static_cast<T>(0);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
        if use_qdd:
            # s_qdd = 0 so ID(q,qd,0) = nle (value identical to the use_qdd_input=False
            # path). The device wrapper carries no MUJOCO_OUTPUT epilogue (host-template
            # mjx convert is kernel-only), so s_qdd stays all-zero here.
            self.gen_add_parallel_loop("i", str(nv))
            self.gen_add_code_line("s_qdd[i] = static_cast<T>(0);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
        qd_arg = "s_qd0" if gravity_only else "s_qd"
        self.gen_inverse_dynamics_inner_function_call(
            compute_c=True, use_qdd_input=use_qdd,
            updated_var_names=dict(s_c_name="s_out", s_qd_name=qd_arg,
                                   s_qdd_name="s_qdd", d_f_ext_name="nullptr"))

    self.gen_device_wrapper(
        desc, func_def, _id_bias_inner_temp_mem_size(self), _inner,
        func_params=func_params, extra_t_buffers=extra, include_linalg_scratch=True)


def _emit_id_bias_kernel_body(self, gravity_only, single_call_timing, mjx_kernel=False):
    n = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    input_count = 2 * n
    # s_vaf is body-indexed by RAW body id inside inverse_dynamics_inner; size by
    # NB (get_num_joints) for mimic robots (NB > NV) so the high-body f-writes never
    # overflow into the next arena region. Non-mimic keeps 18*n byte-identical.
    nb_vaf = self.robot.get_num_joints() if self.robot_has_mimic_joints() else n
    # nonlinear_effects floating-base mjx path: zeroed qdd buffer (nv) sized into the
    # same arena slack as gravity's s_qd0 band (the host arena macro budgets +nv for
    # the bias kernels regardless), so NO GRiDCodeGenerator.py arena change is needed.
    use_qdd = self.robot.floating_base and not gravity_only
    extra = [("s_q_qd", input_count), ("s_out", nv), ("s_vaf", 18 * nb_vaf)]
    if gravity_only:
        extra.append(("s_qd0", nv))
    if use_qdd:
        extra.append(("s_qdd", nv))
    self.gen_XImats_helpers_temp_shared_memory_code(
        _id_bias_inner_temp_mem_size(self), extra_t_buffers=extra, include_linalg_scratch=True)
    self.gen_add_code_line("T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(n) + "];")
    if gravity_only:
        self.gen_add_code_line("(void)s_qd;")

    def _zero_qdd():
        # Zero the WHOLE s_qdd (nv) FIRST, unconditionally (both pin + mjx
        # instantiations). For the default (pin) instantiation s_qdd stays all-zero
        # -> ID(q,qd,0) = nle, value identical to the historical use_qdd_input=False
        # path. For the mjx instantiation the input-convert (emitted right after this,
        # before the compute) overwrites only s_qdd[0:3] = delta_a = -(omega x v); the
        # non-base entries [3:] stay 0.
        self.gen_add_parallel_loop("i", str(nv))
        self.gen_add_code_line("s_qdd[i] = static_cast<T>(0);")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    def _compute():
        self.gen_load_update_XImats_helpers_function_call()
        if gravity_only:
            self.gen_add_parallel_loop("i", str(nv))
            self.gen_add_code_line("s_qd0[i] = static_cast<T>(0);")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
        qd_arg = "s_qd0" if gravity_only else "s_qd"
        self.gen_inverse_dynamics_inner_function_call(
            compute_c=True, use_qdd_input=use_qdd,
            updated_var_names=dict(s_c_name="s_out", s_qd_name=qd_arg,
                                   s_qdd_name="s_qdd", d_f_ext_name="nullptr"))

    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q_qd", str(input_count), stride="stride_q_qd")
        # nonlinear_effects floating mjx: zero s_qdd (all instantiations) BEFORE the
        # input-convert, which writes s_qdd[0:3] = delta_a for the mjx case.
        if use_qdd:
            _zero_qdd()
        if mjx_kernel:
            # gravity (qd=0): q-only convert (reorder quaternion). nonlinear_effects:
            # FULL input-convert -- reorders quat, qd->pin frame, AND writes
            # s_qdd[0:3] = -(omega x v) = delta_a (mjx input qdd is 0; the helper's
            # accel-couple term produces delta_a). Emitted BEFORE the XImats build so
            # X[0] uses the correctly-ordered quaternion.
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            if use_qdd:
                self.gen_mjx_input_convert(q_name="s_q", qd_name="s_qd", qdd_name="s_qdd")
            else:
                self.gen_mjx_quat_reorder(q_name="s_q")
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute")
        _compute()
        self.gen_add_sync()
        # mjx output: the bias is a covector -> base-linear rows rotate by R
        # (base_rotate). For nonlinear_effects the omega x v acceleration coupling is
        # already folded in via s_qdd (ID is affine in qacc), so the output map is the
        # same trivial base_rotate as gravity. base_rotate(ID(q,qd,delta_a)) = mjx nle.
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_base_rotate("s_out")
            self.gen_add_end_control_flow()
        self.gen_kernel_save_result("out", str(nv), stride=str(nv))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q_qd", str(input_count))
        # single_timing: no mjx epilogue (timing path). Zero s_qdd once so the pin
        # use_qdd_input=True call still evaluates ID(q,qd,0) = nle.
        if use_qdd:
            _zero_qdd()
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q_qd", str(input_count), feedback_from="out")
        _compute()
        self.gen_anti_licm_output_write("out")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("out", str(nv))


def gen_id_bias_kernel(self, gravity_only, single_call_timing=False):
    name = "generalized_gravity" if gravity_only else "nonlinear_effects"
    func_def = ("void " + name + "_kernel(T *d_out, unsigned char *d_workspace, "
                "const T *d_q_qd, const int stride_q_qd, "
                "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {")
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Compute " + name + " (RNEA bias) per timestep", [], [], None)
    # MUJOCO_OUTPUT (floating-base both bias kernels): compile-time mjx
    # output-convention flag. For generalized_gravity (qd=0) there is NO omega x v
    # acceleration coupling -- the transform is the trivial base_rotate (G.g): only
    # the base-linear rows of the covector output g rotate by R. For nonlinear_effects
    # the omega x v coupling is folded in via the input-convert (qdd[0:3] = delta_a;
    # ID is affine in qacc so base_rotate(ID(q,qd,delta_a)) = mjx nle), so the OUTPUT
    # map is the SAME base_rotate. Added LAST so the existing positional <T,TIER> call
    # sites are unaffected; the default (false) instantiation if-constexpr-elides the
    # epilogue -> byte-identical PTX. Never emitted for fixed-base.
    mjx_kernel = self.robot.floating_base
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("(void)d_workspace;")
    _emit_id_bias_kernel_body(self, gravity_only, single_call_timing, mjx_kernel)
    self.gen_add_end_function()


def gen_id_bias_host(self, gravity_only, mode=0):
    name = "generalized_gravity" if gravity_only else "nonlinear_effects"
    macro = "INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES<T>()"
    single_call_timing = (mode == 1)
    compute_only = (mode == 2)
    func_def_start = ("void " + name + "(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, "
                      "const T gravity, const int num_timesteps,")
    func_def_end = "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    self.gen_add_func_doc("Compute " + name + " (RNEA bias)", [], [], None)
    # MUJOCO_OUTPUT (floating-base both bias kernels) host template flag:
    # forwarded to the kernel launch. The kernel template is <T, RESOURCE_TIER, MUJOCO_OUTPUT>
    # so the flag must be named positionally (tier defaulted explicitly). Added LAST so
    # existing positional template args are unaffected; default false -> byte-identical.
    mjx_host = self.robot.floating_base
    if mjx_host:
        self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false>")
    else:
        self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"" + name + " requires all-data or dynamics gridData\");")
    kbase = name + ("_kernel_single_timing" if single_call_timing else "_kernel")
    kname = (kbase + "<T, GRID_DEFAULT_RESOURCE_TIER, MUJOCO_OUTPUT>") if mjx_host else (kbase + "<T>")
    func_call = (kname + "<<<block_dimms,thread_dimms," + macro + ">>>(hd_data->d_c,hd_data->d_workspace,"
                 "hd_data->d_q_qd,stride_q_qd,d_robotModel,gravity,num_timesteps);")
    if not compute_only:
        self.gen_add_code_lines([
            "// start code with memory transfer", "int stride_q_qd;",
            "if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*" +
            ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}",
            "else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*" +
            ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}"])
    else:
        self.gen_add_code_line("int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS : 3*NUM_JOINTS;")
    self.gen_add_code_line("// then call the kernel")
    func_call_mem = "if (USE_COMPRESSED_MEM) {" + func_call + "}"
    func_call_mem2 = "else                    {" + func_call.replace("hd_data->d_q_qd", "hd_data->d_q_qd_u") + "}"
    func_call_code = [func_call_mem, func_call_mem2, "gpuErrchkKernel();"]
    if single_call_timing:
        func_call_code.insert(0, "struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"" + name + "\", " + macro + "));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back",
            "gpuErrchk(cudaMemcpy(hd_data->h_c,hd_data->d_c,NUM_VEL*" +
            ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();"])
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line(name))
    self.gen_add_end_function()


def gen_id_bias(self, gravity_only):
    self.gen_id_bias_device(gravity_only)
    self.gen_id_bias_kernel(gravity_only, True)
    self.gen_id_bias_kernel(gravity_only, False)
    self.gen_id_bias_host(gravity_only, 0)
    self.gen_id_bias_host(gravity_only, 1)
    self.gen_id_bias_host(gravity_only, 2)


# ===========================================================================
# Shared kinematics-domain centroidal inner.
#
# Builds (into the caller-provided scratch slabs) the world momentum map and
# composite inertia, then writes:
#   s_A    (6 x NUM_VEL, column-major)  -- Pinocchio CMM [linear; angular] @ CoM
#   s_com  (3)                          -- world CoM position
#   s_extra[0] = M_total                -- total mass
#   s_extra[1..3] = world first moment IW first-moment (m*c) (unused downstream)
#
# Scratch layout in s_temp:
#   s_Xworld : 16 * NJ   (world homogeneous transform per joint)
#   s_J      : 6 * NV * NB (per-body world spatial Jacobian, angular-first)
#   s_Iw     : 36 * NB   (per-body world spatial inertia, angular-first)
#   s_A0     : 6 * NV     (world momentum map at world origin, angular-first)
#   s_IW     : 36         (world composite inertia)
# Body inertias are CONSTANT — read from d_robotModel->d_XImats[36*NB + 36*jid].
# ===========================================================================

def _centroidal_inner_temp_mem_size(self, j_in_smem=True):
    """centroidal_inner s_temp pool size. When j_in_smem is False the s_J band
    (6*nv*NB) is repointed to the caller's s_J_ext (d_workspace) and the pool
    shrinks by that much (offsets after s_J rebase to drop the hole)."""
    NJ = self.robot.get_num_joints()
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    sJ = (6 * nv * NB) if j_in_smem else 0
    return 16 * NJ + sJ + 36 * NB + 6 * nv + 36


def gen_centroidal_inner(self):
    """Emit centroidal_inner: world momentum map A0, world composite inertia IW,
    CoM, and the Pinocchio-convention CMM A. Serial-correctness focused (new,
    low perf-priority family). s_A / s_com / s_extra are caller outputs."""
    NJ = self.robot.get_num_joints()
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    n_bfs_levels = self.robot.get_max_bfs_level() + 1
    ImatOffset = 36 * NB  # body inertias start here in d_XImats

    func_params = [
        "s_A is the output 6 x NUM_VEL CMM (column-major, [linear; angular] @ CoM, world-aligned)",
        "s_com is the output 3-vector world CoM",
        "s_extra holds [M_total] (s_extra[0])",
        "s_q is the vector of joint positions (unused; q is baked into s_Xhom)",
        "s_Xhom is the per-joint LOCAL homogeneous transforms",
        "d_robotModel is the GPU model helpers (for the constant body inertias)",
        "s_temp is scratch of size " + str(_centroidal_inner_temp_mem_size(self)),
        "s_J_ext is the spilled-J band (used only when !J_IN_SMEM; pass nullptr when J_IN_SMEM)",
        "s_linalg_smem is reserved (unused)"]
    func_def_middle = "T *s_A, T *s_com, T *s_extra, const T *s_q, const T *s_Xhom, const robotModel<T> *d_robotModel, "
    func_def = "void centroidal_inner(" + func_def_middle + "T *s_temp, T *s_J_ext, unsigned char *s_linalg_smem) {"
    self.gen_add_func_doc("Compute the world momentum map, composite inertia, CoM and CMM", [], func_params, None)
    # J_IN_SMEM=true (default) keeps s_J in the s_temp pool (byte-identical to the
    # original com/ccrba/energy emit). When false (DE-GATE #2: dccrba/cmm at the
    # J-spilled tier) the cold/large s_J band (6*nv*NB) is repointed to s_J_ext
    # (the L2-pinned d_workspace SO band) and the s_temp pool shrinks by that much
    # (the offsets after s_J rebase to drop the hole). Pure pointer move; the math,
    # buffer layout (6*nv stride, body-indexed) and mimic alpha-fold are unchanged.
    self.gen_add_code_line("template <typename T, bool J_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("(void)s_q; (void)s_linalg_smem; (void)s_J_ext;")

    # Pointer band layout. When J_IN_SMEM (default; com/ccrba/energy + dccrba/cmm
    # L0/L1) the s_J band lives in s_temp and the trailing offsets are the original
    # baked literals -> BYTE-IDENTICAL emit. When !J_IN_SMEM (dccrba/cmm J-spilled
    # tier) s_J points at s_J_ext (d_workspace) and s_Iw/s_A0/s_IW shift down by
    # 6*nv*NB so the in-smem pool genuinely shrinks. if constexpr keeps the spill
    # branch entirely out of the J_IN_SMEM=true instantiation.
    off_J = 16 * NJ
    off_Iw = off_J + 6 * nv * NB
    off_A0 = off_Iw + 36 * NB
    off_IW = off_A0 + 6 * nv
    # No-J layout: s_Iw sits right after s_Xworld (the s_J band lives in s_J_ext).
    off_Iw_noJ = off_J                       # == 16*NJ (no 6*nv*NB s_J hole)
    off_A0_noJ = off_Iw_noJ + 36 * NB
    off_IW_noJ = off_A0_noJ + 6 * nv
    self.gen_add_code_line("T *s_Xworld = &s_temp[0];")
    self.gen_add_code_line("T *s_J; T *s_Iw; T *s_A0; T *s_IW;")
    self.gen_add_code_line("if constexpr (J_IN_SMEM) {", True)
    self.gen_add_code_line("s_J  = &s_temp[" + str(off_J) + "];   // 6 x NV per body (angular-first)")
    self.gen_add_code_line("s_Iw = &s_temp[" + str(off_Iw) + "];  // 36 per body world inertia")
    self.gen_add_code_line("s_A0 = &s_temp[" + str(off_A0) + "];  // 6 x NV world momentum map")
    self.gen_add_code_line("s_IW = &s_temp[" + str(off_IW) + "];  // 36 world composite inertia")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("else {", True)
    self.gen_add_code_line("s_J  = s_J_ext;   // spilled to d_workspace SO band (6*nv*NB)")
    self.gen_add_code_line("s_Iw = &s_temp[" + str(off_Iw_noJ) + "];  // pool shrinks by 6*nv*NB")
    self.gen_add_code_line("s_A0 = &s_temp[" + str(off_A0_noJ) + "];")
    self.gen_add_code_line("s_IW = &s_temp[" + str(off_IW_noJ) + "];")
    self.gen_add_end_control_flow()

    # ---- Step 1: world homogeneous transforms by BFS level (chain-up) ----
    self.gen_add_code_line("// Step 1: world homogeneous transforms (chain-up of local s_Xhom)")
    for level in range(n_bfs_levels):
        ids_at_level = self.robot.get_ids_by_bfs_level(level)
        if not ids_at_level:
            continue
        njs = len(ids_at_level)
        self.gen_add_parallel_loop("ind", str(16 * njs))
        self.gen_add_code_line("int slot = ind / 16; int ele = ind % 16;")
        self.gen_add_code_line("int row = ele & 3; int col = ele >> 2;")
        jid_list = [str(j) for j in ids_at_level]
        par_list = [str(self.robot.get_parent_id(j)) for j in ids_at_level]
        if njs > 1:
            self.gen_add_multi_threaded_select("slot", "<", [str(i + 1) for i in range(njs)],
                                               [("int", "jid", jid_list), ("int", "par", par_list)])
        else:
            self.gen_add_code_line("const int jid = " + jid_list[0] + "; const int par = " + par_list[0] + ";")
        self.gen_add_code_line("if (par == -1) { s_Xworld[16*jid + ele] = s_Xhom[16*jid + ele]; }")
        self.gen_add_code_line("else { s_Xworld[16*jid + ele] = dot_prod<T,4,4,1>(&s_Xworld[16*par + row], &s_Xhom[16*jid + 4*col]); }")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    # ---- Step 2: per-body world spatial inertia Iw = Xpx^{-T} Ibw Xpx^{-1} ----
    # We assemble the world inertia in angular-first [ang; lin] order directly
    # from the body inertia (constant) and the world (R, p):
    #   I_O_w = R I_O R^T ;  mc_w = R mc ;  S_mc = skew(mc_w) ; S_p = skew(p)
    #   Iw_ang_ang = I_O_w - S_p S_mc^T - S_mc S_p ... assembled via the dual
    #   translation X = [[I,0],[S_p,I]]: Iw = X^{-T} Ibw X^{-1}, Ibw the rotated
    #   body inertia at the body origin. X^{-1} = [[I,0],[-S_p,I]].
    self.gen_add_code_line("// Step 2: per-body world spatial inertia (angular-first)")
    self.gen_add_serial_ops()
    self.gen_add_code_line("for (int jid = 0; jid < " + str(NB) + "; ++jid) {", True)
    # world R (column-major in s_Xworld: R[r,c] = s_Xworld[16*jid + r + 4*c]); p = s_Xworld[16*jid + 12..14]
    self.gen_add_code_line("const T *Xw = &s_Xworld[16*jid];")
    self.gen_add_code_line("T R[9]; for (int c=0;c<3;++c) for (int r=0;r<3;++r) R[r+3*c] = Xw[r + 4*c];")
    self.gen_add_code_line("T px = Xw[12], py = Xw[13], pz = Xw[14];")
    # body inertia (constant): Ib col-major 6x6 at d_XImats[ImatOffset + 36*jid]
    self.gen_add_code_line("const T *Ib = &d_robotModel->d_XImats[" + str(ImatOffset) + " + 36*jid];")
    self.gen_add_code_line("T m = Ib[5 + 6*5];")
    # I_O (top-left 3x3, col-major): I_O[r+3c] = Ib[r + 6*c]
    self.gen_add_code_line("T IO[9]; for (int c=0;c<3;++c) for (int r=0;r<3;++r) IO[r+3*c] = Ib[r + 6*c];")
    # mc from top-right skew block: Ib[:3,3:6] = skew(mc) col-major -> Ib[r + 6*(3+c)]
    self.gen_add_code_line("T mc0 = Ib[2 + 6*4]; T mc1 = Ib[0 + 6*5]; T mc2 = Ib[1 + 6*3];")
    # rotate: I_O_w = R IO R^T ; mc_w = R mc
    self.gen_add_code_line("T tmp[9];")
    self.gen_add_code_line("for (int c=0;c<3;++c) for (int r=0;r<3;++r) { T s=0; for(int k=0;k<3;++k) s += R[r+3*k]*IO[k+3*c]; tmp[r+3*c]=s; }")
    self.gen_add_code_line("T IOw[9];")
    self.gen_add_code_line("for (int c=0;c<3;++c) for (int r=0;r<3;++r) { T s=0; for(int k=0;k<3;++k) s += tmp[r+3*k]*R[c+3*k]; IOw[r+3*c]=s; }")
    self.gen_add_code_line("T mcw0 = R[0]*mc0 + R[3]*mc1 + R[6]*mc2;")
    self.gen_add_code_line("T mcw1 = R[1]*mc0 + R[4]*mc1 + R[7]*mc2;")
    self.gen_add_code_line("T mcw2 = R[2]*mc0 + R[5]*mc1 + R[8]*mc2;")
    # S_mc (skew of mc_w), S_p (skew of p). Build the world inertia (angular-first):
    #   ang-ang = IOw + (S_p S_mc + (S_p S_mc)^T) + S_p (m I) S_p^T ... use the
    #   closed Featherstone translation. We use: Iw = Xt^{-T} Ibw Xt^{-1},
    #   Xt^{-1} = [[I,0],[-S_p,I]]. Let Ibw = [[IOw, S_mc],[S_mc^T, m I]].
    # Compute blocks directly:
    #   ang-lin (top-right)  = S_mc - S_p (m I) ... carefully:
    # Xt^{-1} = [[I,0],[-Sp,I]]; (Xt^{-1})^T = [[I, -Sp^T],[0,I]] and with Sp
    # skew (Sp^T = -Sp) this is [[I, Sp],[0,I]] (verified numerically vs the
    # RBDReference _link_world_spatial_inertia oracle).
    # Iw = Xt^{-T} Ibw Xt^{-1}.  Let A=IOw, B=S_mc, C=S_mc^T, D=mI.
    # M1 = Ibw Xt^{-1} = [[A - B Sp, B],[C - D Sp, D]]
    # Iw = Xt^{-T} M1 = [[A - B Sp + Sp(C - D Sp), B + Sp D],[C - D Sp, D]]
    self.gen_add_code_line("T Sp[9] = {0,pz,-py, -pz,0,px, py,-px,0};   // skew(p) col-major")
    self.gen_add_code_line("T Sm[9] = {0,mcw2,-mcw1, -mcw2,0,mcw0, mcw1,-mcw0,0};  // skew(mc_w) col-major")
    # helper lambdas via explicit loops: matmul 3x3 col-major
    self.gen_add_code_line("auto mm = [](const T*X,const T*Y,T*Z){ for(int c=0;c<3;++c) for(int r=0;r<3;++r){ T s=0; for(int k=0;k<3;++k) s+=X[r+3*k]*Y[k+3*c]; Z[r+3*c]=s; } };")
    self.gen_add_code_line("T BSp[9]; mm(Sm,Sp,BSp);")        # B Sp
    self.gen_add_code_line("T D[9] = {m,0,0, 0,m,0, 0,0,m};")
    self.gen_add_code_line("T DSp[9]; mm(D,Sp,DSp);")          # D Sp
    self.gen_add_code_line("T C[9]; for(int c=0;c<3;++c) for(int r=0;r<3;++r) C[r+3*c]=Sm[c+3*r];")  # S_mc^T
    self.gen_add_code_line("T CmDSp[9]; for(int i=0;i<9;++i) CmDSp[i]=C[i]-DSp[i];")  # C - D Sp (bottom-left)
    self.gen_add_code_line("T SpCmDSp[9]; mm(Sp,CmDSp,SpCmDSp);")   # Sp (C - D Sp)
    self.gen_add_code_line("T SpD[9]; mm(Sp,D,SpD);")               # Sp D
    # blocks
    self.gen_add_code_line("T angang[9]; for(int i=0;i<9;++i) angang[i]=IOw[i]-BSp[i]+SpCmDSp[i];")
    self.gen_add_code_line("T anglin[9]; for(int i=0;i<9;++i) anglin[i]=Sm[i]+SpD[i];")
    # write Iw (col-major 6x6, angular-first) into s_Iw[36*jid]
    self.gen_add_code_line("T *Iw = &s_Iw[36*jid];")
    self.gen_add_code_line("for(int c=0;c<3;++c) for(int r=0;r<3;++r){")
    self.gen_add_code_line("  Iw[r + 6*c] = angang[r+3*c];")          # ang-ang
    self.gen_add_code_line("  Iw[r + 6*(c+3)] = anglin[r+3*c];")      # ang-lin (top-right)
    self.gen_add_code_line("  Iw[(r+3) + 6*c] = CmDSp[r+3*c];")       # lin-ang (bottom-left) = C - D Sp
    self.gen_add_code_line("  Iw[(r+3) + 6*(c+3)] = D[r+3*c];")       # lin-lin = m I
    self.gen_add_code_line("}")
    self.gen_add_end_control_flow()  # end jid loop
    self.gen_add_end_control_flow()  # end serial
    self.gen_add_sync()

    # ---- Step 3: world spatial Jacobian per body (angular-first) ----
    self.gen_add_code_line("// Step 3: per-body world spatial Jacobian J (6 x NV, angular-first)")
    self.gen_add_parallel_loop("ind", str(6 * nv * NB))
    self.gen_add_code_line("s_J[ind] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # Build column fills: for each (body jid, ancestor-or-self jj, S-col c) the
    # world contribution to J[:, vi] is the screw of joint jj at the world origin.
    # We bake the per-body chain jobs at codegen time.
    # MIMIC fold: a mimic joint and its target share ONE velocity coordinate
    # (get_joint_index_v(mimic) == get_joint_index_v(target)), so several chain
    # joints accumulate into the same J column. The mimic body's generalized
    # velocity is alpha * v_target, so its contribution to s_J is alpha-weighted
    # (mirrors RBDReference._body_spatial_jacobian_world's `scale=_mimic_multiplier(j)`
    # and the frame_jacobian Step-3 fold). The serial accumulate already sums the
    # shared-vi columns; the only missing piece is the per-job alpha. For a
    # NON-mimic robot every alpha == 1.0, so the alpha array/multiply is gated on
    # HAS_MIMIC to keep non-mimic emit byte-identical.
    HAS_MIMIC = self.robot_has_mimic_joints()
    jobs = []  # (jid, jj, vi, ang_local[3], lin_local[3], alpha)
    for jid in range(NB):
        chain = sorted(self.robot.get_ancestors_by_id(jid)) + [jid]
        for jj in chain:
            S = np.asarray(self.robot.get_S_by_id(jj), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            vinds = self.robot.get_joint_index_v(jj)
            if not isinstance(vinds, (list, tuple, np.ndarray)):
                vinds = [vinds]
            else:
                vinds = list(vinds)
            alpha = self._alpha_for_jid(jj) if HAS_MIMIC else 1.0
            for c in range(S.shape[1]):
                vi = vinds[c] if c < len(vinds) else vinds[-1]
                ang = [float(S[0, c]), float(S[1, c]), float(S[2, c])]
                lin = [float(S[3, c]), float(S[4, c]), float(S[5, c])]
                jobs.append((jid, jj, vi, ang, lin, alpha))
    njobs = len(jobs)
    if njobs > 0:
        jid_arr = ", ".join(str(j[0]) for j in jobs)
        jj_arr = ", ".join(str(j[1]) for j in jobs)
        vi_arr = ", ".join(str(j[2]) for j in jobs)
        ax_arr = ", ".join("static_cast<T>({:.17g})".format(v) for j in jobs for v in j[3])
        lx_arr = ", ".join("static_cast<T>({:.17g})".format(v) for j in jobs for v in j[4])
        self.gen_add_code_line("const int cj_jid[" + str(njobs) + "] = {" + jid_arr + "};")
        self.gen_add_code_line("const int cj_jj[" + str(njobs) + "] = {" + jj_arr + "};")
        self.gen_add_code_line("const int cj_vi[" + str(njobs) + "] = {" + vi_arr + "};")
        self.gen_add_code_line("const T cj_ang[" + str(3 * njobs) + "] = {" + ax_arr + "};")
        self.gen_add_code_line("const T cj_lin[" + str(3 * njobs) + "] = {" + lx_arr + "};")
        if HAS_MIMIC:
            al_arr = ", ".join("static_cast<T>({:.17g})".format(j[5]) for j in jobs)
            self.gen_add_code_line("const T cj_alpha[" + str(njobs) + "] = {" + al_arr + "};")
        # Serial accumulation (columns within one body can repeat vi across bodies;
        # different bodies write disjoint J slabs, but to keep it simple & correct
        # we accumulate serially). Correctness-first.
        self.gen_add_serial_ops()
        self.gen_add_code_line("for (int t = 0; t < " + str(njobs) + "; ++t) {", True)
        self.gen_add_code_line("int jid = cj_jid[t]; int jj = cj_jj[t]; int vi = cj_vi[t];")
        self.gen_add_code_line("const T *Xj = &s_Xworld[16*jj];")
        self.gen_add_code_line("T a0=cj_ang[3*t], a1=cj_ang[3*t+1], a2=cj_ang[3*t+2];")
        self.gen_add_code_line("T l0=cj_lin[3*t], l1=cj_lin[3*t+1], l2=cj_lin[3*t+2];")
        # world axis: aw = R_jj * ang_local ; lw = R_jj * lin_local
        self.gen_add_code_line("T aw0 = Xj[0]*a0 + Xj[4]*a1 + Xj[8]*a2;")
        self.gen_add_code_line("T aw1 = Xj[1]*a0 + Xj[5]*a1 + Xj[9]*a2;")
        self.gen_add_code_line("T aw2 = Xj[2]*a0 + Xj[6]*a1 + Xj[10]*a2;")
        self.gen_add_code_line("T lw0 = Xj[0]*l0 + Xj[4]*l1 + Xj[8]*l2;")
        self.gen_add_code_line("T lw1 = Xj[1]*l0 + Xj[5]*l1 + Xj[9]*l2;")
        self.gen_add_code_line("T lw2 = Xj[2]*l0 + Xj[6]*l1 + Xj[10]*l2;")
        # p_jj (world origin of joint jj)
        self.gen_add_code_line("T pjx = Xj[12], pjy = Xj[13], pjz = Xj[14];")
        # linear at world origin = lw + p_jj x aw
        self.gen_add_code_line("T linw0 = lw0 + (pjy*aw2 - pjz*aw1);")
        self.gen_add_code_line("T linw1 = lw1 + (pjz*aw0 - pjx*aw2);")
        self.gen_add_code_line("T linw2 = lw2 + (pjx*aw1 - pjy*aw0);")
        # accumulate into J[:, vi] for body jid (angular-first): base = 6*nv*jid + 6*vi
        # MIMIC: scale this chain joint's column contribution by its multiplier
        # alpha (the mimic body moves alpha * v_target; non-mimic alpha == 1.0,
        # and the alpha term is omitted entirely so non-mimic emit is unchanged).
        self.gen_add_code_line("T *Jc = &s_J[" + str(6 * nv) + "*jid + 6*vi];")
        if HAS_MIMIC:
            self.gen_add_code_line("T al = cj_alpha[t];")
            self.gen_add_code_line("Jc[0]+=al*aw0; Jc[1]+=al*aw1; Jc[2]+=al*aw2; Jc[3]+=al*linw0; Jc[4]+=al*linw1; Jc[5]+=al*linw2;")
        else:
            self.gen_add_code_line("Jc[0]+=aw0; Jc[1]+=aw1; Jc[2]+=aw2; Jc[3]+=linw0; Jc[4]+=linw1; Jc[5]+=linw2;")
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()  # serial
        self.gen_add_sync()

    # ---- Step 4: A0 = sum_j Iw_j J_j ; IW = sum_j Iw_j ----
    self.gen_add_code_line("// Step 4: world momentum map A0 = sum_j Iw_j J_j ; composite IW = sum_j Iw_j")
    self.gen_add_serial_ops()
    self.gen_add_code_line("for (int i = 0; i < 36; ++i) s_IW[i] = static_cast<T>(0);")
    self.gen_add_code_line("for (int i = 0; i < " + str(6 * nv) + "; ++i) s_A0[i] = static_cast<T>(0);")
    self.gen_add_code_line("for (int jid = 0; jid < " + str(NB) + "; ++jid) {", True)
    self.gen_add_code_line("const T *Iw = &s_Iw[36*jid];")
    self.gen_add_code_line("for (int i=0;i<36;++i) s_IW[i] += Iw[i];")
    self.gen_add_code_line("const T *Jb = &s_J[" + str(6 * nv) + "*jid];")
    # A0[:, vi] += Iw * J[:, vi]  (6x6 * 6) for each vi
    self.gen_add_code_line("for (int vi = 0; vi < " + str(nv) + "; ++vi) {", True)
    self.gen_add_code_line("const T *Jcol = &Jb[6*vi];")
    self.gen_add_code_line("for (int r = 0; r < 6; ++r) { T s=0; for(int k=0;k<6;++k) s += Iw[r + 6*k]*Jcol[k]; s_A0[r + 6*vi] += s; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    # CoM from IW: mass = IW[5,5]; first moment skew = IW[:3,3:6]
    self.gen_add_code_line("T mass = s_IW[5 + 6*5];")
    self.gen_add_code_line("T mcx = s_IW[2 + 6*4]; T mcy = s_IW[0 + 6*5]; T mcz = s_IW[1 + 6*3];")
    self.gen_add_code_line("s_com[0] = mcx/mass; s_com[1] = mcy/mass; s_com[2] = mcz/mass;")
    self.gen_add_code_line("s_extra[0] = mass;")
    # Shift A0 (angular-first @ world origin) to CoM (world-aligned), reorder to
    # [linear; angular]. n_com = n - com x f ; f unchanged.
    #   Afs_ang[:, vi] = A0_ang[:, vi] - com x A0_lin[:, vi]
    #   Afs_lin[:, vi] = A0_lin[:, vi]
    #   A[:3] = Afs_lin ; A[3:] = Afs_ang
    self.gen_add_code_line("T cx = s_com[0], cy = s_com[1], cz = s_com[2];")
    self.gen_add_code_line("for (int vi = 0; vi < " + str(nv) + "; ++vi) {", True)
    self.gen_add_code_line("T n0=s_A0[0+6*vi], n1=s_A0[1+6*vi], n2=s_A0[2+6*vi];")
    self.gen_add_code_line("T f0=s_A0[3+6*vi], f1=s_A0[4+6*vi], f2=s_A0[5+6*vi];")
    self.gen_add_code_line("T nc0 = n0 - (cy*f2 - cz*f1);")
    self.gen_add_code_line("T nc1 = n1 - (cz*f0 - cx*f2);")
    self.gen_add_code_line("T nc2 = n2 - (cx*f1 - cy*f0);")
    # A column-major 6 x NV: A[row + 6*vi]
    self.gen_add_code_line("s_A[0 + 6*vi] = f0; s_A[1 + 6*vi] = f1; s_A[2 + 6*vi] = f2;")
    self.gen_add_code_line("s_A[3 + 6*vi] = nc0; s_A[4 + 6*vi] = nc1; s_A[5 + 6*vi] = nc2;")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()  # serial
    self.gen_add_sync()
    self.gen_add_end_function()


def _centroidal_device_extra(self):
    NJ = self.robot.get_num_joints()
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    # outputs + inner scratch live in extra_t_buffers for device wrappers.
    return [("s_A", 6 * nv), ("s_com", 3), ("s_extra", 4)]


def _gen_centroidal_call(self, spill=False):
    if spill:
        # kernel path (DE-GATE #2): s_J is carved as a SEPARATE tier-routed buffer
        # (in-smem tail at the J-in-smem tiers, d_workspace at the J-spilled tier), so
        # the inner is ALWAYS called with J external (J_IN_SMEM=false) and s_temp is the
        # SHRUNK no-J pool. s_J already points at the right place (smem tail or repointed).
        self.gen_add_code_line("centroidal_inner<T, false>(s_A, s_com, s_extra, s_q, s_XmatsHom, d_robotModel, s_temp, s_J, s_linalg_smem);")
    else:
        # device-wrapper path: keep s_J inside s_temp (J_IN_SMEM=true) -> s_J_ext nullptr.
        self.gen_add_code_line("centroidal_inner<T, true>(s_A, s_com, s_extra, s_q, s_XmatsHom, d_robotModel, s_temp, nullptr, s_linalg_smem);")


# ----- com (p_com + J_com) device/kernel/host -----

def gen_com_device(self):
    nv = self.robot.get_num_vel()
    func_def = ("void com_device(T *s_out, const T *s_q, const robotModel<T> *d_robotModel) {")
    func_params = ["s_out holds [p_com (3); J_com (3 x NUM_VEL, column-major)]",
                   "s_q is the joint position vector", "d_robotModel is the GPU model helpers"]
    self.gen_add_func_doc("Compute CoM position and CoM Jacobian", [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    extra = _centroidal_device_extra(self)
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _centroidal_inner_temp_mem_size(self), extra_t_buffers=extra, include_linalg_scratch=True,
        linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    self.gen_load_update_XmatsHom_helpers_function_call()
    _gen_centroidal_call(self)
    self.gen_add_sync()
    # write s_out: [com(3); Jcom (3 x nv)] ; Jcom = A[:3,:] / mass
    self.gen_add_serial_ops()
    self.gen_add_code_line("s_out[0]=s_com[0]; s_out[1]=s_com[1]; s_out[2]=s_com[2];")
    self.gen_add_code_line("T inv_m = static_cast<T>(1)/s_extra[0];")
    self.gen_add_code_line("for (int vi=0; vi<" + str(nv) + "; ++vi) for (int r=0;r<3;++r) s_out[3 + 3*vi + r] = s_A[r + 6*vi]*inv_m;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()


# ----- ccrba (A + h) device -----

def gen_ccrba_device(self):
    nv = self.robot.get_num_vel()
    func_def = ("void ccrba_device(T *s_out, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel) {")
    func_params = ["s_out holds [A (6 x NUM_VEL, column-major); h (6)]",
                   "s_q / s_qd are joint position / velocity", "d_robotModel is the GPU model helpers"]
    self.gen_add_func_doc("Compute the centroidal momentum matrix A and momentum h = A qd", [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    extra = _centroidal_device_extra(self)
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _centroidal_inner_temp_mem_size(self), extra_t_buffers=extra, include_linalg_scratch=True,
        linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    self.gen_load_update_XmatsHom_helpers_function_call()
    _gen_centroidal_call(self)
    self.gen_add_sync()
    # write A then h = A qd
    self.gen_add_parallel_loop("ind", str(6 * nv))
    self.gen_add_code_line("s_out[ind] = s_A[ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("for (int r=0;r<6;++r){ T s=0; for(int vi=0;vi<" + str(nv) + ";++vi) s += s_A[r + 6*vi]*s_qd[vi]; s_out[" + str(6 * nv) + " + r] = s; }")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()


# ----- energy (KE, PE, mechanical) device -----

def gen_energy_device(self):
    nv = self.robot.get_num_vel()
    func_def = ("void energy_device(T *s_out, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity) {")
    func_params = ["s_out holds [KE, PE, mechanical]",
                   "s_q / s_qd are joint position / velocity", "d_robotModel is the GPU model helpers",
                   "gravity is the gravity constant"]
    self.gen_add_func_doc("Compute kinetic / potential / mechanical energy", [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    extra = _centroidal_device_extra(self)
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _centroidal_inner_temp_mem_size(self), extra_t_buffers=extra, include_linalg_scratch=True,
        linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    self.gen_load_update_XmatsHom_helpers_function_call()
    _gen_centroidal_call(self)
    self.gen_add_sync()
    # KE = 1/2 sum_j (J_j qd)^T Iw_j (J_j qd). Recompute via the per-body world
    # inertias + Jacobians left in s_temp by the inner.
    NB = self.robot.get_num_bodies()
    off_J = 16 * self.robot.get_num_joints()
    off_Iw = off_J + 6 * nv * NB
    self.gen_add_serial_ops()
    self.gen_add_code_line("T *cJ  = &s_temp[" + str(off_J) + "];")
    self.gen_add_code_line("T *cIw = &s_temp[" + str(off_Iw) + "];")
    self.gen_add_code_line("T ke = static_cast<T>(0);")
    self.gen_add_code_line("for (int jid = 0; jid < " + str(NB) + "; ++jid) {", True)
    self.gen_add_code_line("const T *Jb = &cJ[" + str(6 * nv) + "*jid]; const T *Iw = &cIw[36*jid];")
    # v = J qd (6) ; ke += 1/2 v^T Iw v
    self.gen_add_code_line("T v[6]; for (int r=0;r<6;++r){ T s=0; for(int vi=0;vi<" + str(nv) + ";++vi) s += Jb[6*vi+r]*s_qd[vi]; v[r]=s; }")
    self.gen_add_code_line("for (int r=0;r<6;++r){ T s=0; for(int k=0;k<6;++k) s += Iw[r+6*k]*v[k]; ke += static_cast<T>(0.5)*v[r]*s; }")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("T pe = -s_extra[0] * gravity * s_com[2]; // PE = -M*gravity*com_z (gravity=-9.81, matches RBDReference)")
    self.gen_add_code_line("s_out[0]=ke; s_out[1]=pe; s_out[2]=ke+pe;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()


# ----- generic kinematics-domain kernel/host generator for com / ccrba / energy -----

def _gen_kin_centroidal_kernel(self, name, out_size, has_qd, has_gravity, single_call_timing=False):
    n = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    input_count = (2 * n) if has_qd else n
    in_name = "q_qd" if has_qd else "q"
    grav = ", const T gravity" if has_gravity else ""
    NB = self.robot.get_num_bodies()
    sJ = 6 * nv * NB
    PRED = name.upper() + "_J_IN_SMEM"
    SMEM = name.upper() + "_J_SMEM"
    SLOT = name.upper() + "_J_SLOT"
    func_def = ("void " + name + "_kernel(T *d_out, unsigned char *d_workspace, const T *d_" + in_name + ", const int stride_" + in_name + ", "
                "const robotModel<T> *d_robotModel" + grav + ", const int NUM_TIMESTEPS) {")
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Compute " + name + " per timestep", [], [], None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag, LAST
    # after RESOURCE_TIER so existing positional <T,TIER> call sites are unaffected;
    # default false if-constexpr-elides the epilogues -> byte-identical pin PTX.
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # DE-GATE #2: the Jw band (6*nv*NB) is the LAST t_buffer; sized sJ at the J-in-smem
    # tiers (<ALGO>_J_IN_SMEM<TIER>()==true) and 0 at the J-spilled tier (then repointed
    # to the L2-pinned d_workspace SO band below). s_temp is the SHRUNK no-J pool.
    self.gen_add_code_line("constexpr bool " + SMEM + " = " + PRED + "<RESOURCE_TIER>();")
    self.gen_add_code_line("constexpr int " + SLOT + " = " + SMEM + " ? " + str(sJ) + " : 0;")
    extra = [("s_" + in_name, input_count), ("s_out", out_size)] + _centroidal_device_extra(self) + [("s_J", SLOT)]
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _centroidal_inner_temp_mem_size(self, j_in_smem=False), extra_t_buffers=extra, include_linalg_scratch=True,
        linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    if has_qd:
        self.gen_add_code_line("T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(n) + "];")
    self.gen_add_code_line("if constexpr (" + SMEM + ") { (void)d_workspace; }")

    def _repoint(in_loop):
        # J-spilled tier: point s_J at the L2-pinned d_workspace SO band (shared
        # GRID_DCCRBA_J_OFFSET_BYTES sub-offset with dccrba/cmm; same 6*nv*NB band).
        self.gen_add_code_line("if constexpr (!" + SMEM + ") {", True)
        if in_loop:
            self.gen_add_code_line("s_J = reinterpret_cast<T *>(&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_DCCRBA_J_OFFSET_BYTES<T>()]);")
        else:
            self.gen_add_code_line("s_J = reinterpret_cast<T *>(&d_workspace[GRID_DCCRBA_J_OFFSET_BYTES<T>()]);")
        self.gen_add_end_control_flow()

    def _compute():
        # mjx INPUT convert (floating only): reorder the base quaternion wxyz->xyzw
        # (so XmatsHom builds X[0] correctly + the output R reads xyzw) and, for the
        # qd-reading families (ccrba/energy), convert the base-linear velocity to the
        # pin frame so the internal compute (h = A qd, KE) is correct. com reads q
        # only -> quat reorder suffices. Emitted BEFORE the XmatsHom build.
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            if has_qd:
                self.gen_mjx_input_convert(q_name="s_q", qd_name="s_qd")
            else:
                self.gen_mjx_quat_reorder("s_q")
            self.gen_add_end_control_flow()
        self.gen_load_update_XmatsHom_helpers_function_call()
        _gen_centroidal_call(self, spill=True)
        self.gen_add_sync()
        # finalize per-family from the inner outputs (s_A, s_com, s_extra, s_temp)
        if name == "com":
            self.gen_add_serial_ops()
            self.gen_add_code_line("s_out[0]=s_com[0]; s_out[1]=s_com[1]; s_out[2]=s_com[2];")
            self.gen_add_code_line("T inv_m = static_cast<T>(1)/s_extra[0];")
            self.gen_add_code_line("for (int vi=0; vi<" + str(nv) + "; ++vi) for (int r=0;r<3;++r) s_out[3 + 3*vi + r] = s_A[r + 6*vi]*inv_m;")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            # mjx OUTPUT: p_com (s_out[0:3]) is INVARIANT; J_com (3 x NV col-major at
            # s_out[3]) column-reframes J G^{-1} (base-linear cols . R^T). R reads the
            # already-reordered xyzw quaternion in s_q.
            if self.robot.floating_base:
                self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
                self.gen_mjx_column_reframe("(s_out + 3)", 3, nv)
                self.gen_add_end_control_flow()
        elif name == "ccrba":
            self.gen_add_parallel_loop("ind", str(6 * nv))
            self.gen_add_code_line("s_out[ind] = s_A[ind];")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            self.gen_add_serial_ops()
            self.gen_add_code_line("for (int r=0;r<6;++r){ T s=0; for(int vi=0;vi<" + str(nv) + ";++vi) s += s_A[r + 6*vi]*s_qd[vi]; s_out[" + str(6 * nv) + " + r] = s; }")
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            # mjx OUTPUT: A (6 x NV col-major at s_out[0]) column-reframes A G^{-1}
            # (base-linear cols . R^T); h = A qd (s_out[6*NV:]) is INVARIANT (qd was
            # already input-converted so h is computed correctly above).
            if self.robot.floating_base:
                self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
                self.gen_mjx_column_reframe("s_out", 6, nv)
                self.gen_add_end_control_flow()
        elif name == "energy":
            # KE reach-back. With J carved out of s_temp, the inner used the no-J pool:
            # s_J lives in the external s_J buffer (smem tail or d_workspace) and s_Iw
            # sits right after s_Xworld at 16*NJ (no 6*nv*NB hole). cJ = s_J; cIw @ 16*NJ.
            off_Iw = 16 * self.robot.get_num_joints()
            self.gen_add_serial_ops()
            self.gen_add_code_line("const T *cJ = s_J; const T *cIw = &s_temp[" + str(off_Iw) + "];")
            self.gen_add_code_line("T ke = static_cast<T>(0);")
            self.gen_add_code_line("for (int jid = 0; jid < " + str(NB) + "; ++jid) {", True)
            self.gen_add_code_line("const T *Jb = &cJ[" + str(6 * nv) + "*jid]; const T *Iw = &cIw[36*jid];")
            self.gen_add_code_line("T v[6]; for (int r=0;r<6;++r){ T s=0; for(int vi=0;vi<" + str(nv) + ";++vi) s += Jb[6*vi+r]*s_qd[vi]; v[r]=s; }")
            self.gen_add_code_line("for (int r=0;r<6;++r){ T s=0; for(int k=0;k<6;++k) s += Iw[r+6*k]*v[k]; ke += static_cast<T>(0.5)*v[r]*s; }")
            self.gen_add_end_control_flow()
            self.gen_add_code_line("T pe = -s_extra[0] * gravity * s_com[2]; // PE = -M*gravity*com_z (gravity=-9.81, matches RBDReference)")
            self.gen_add_code_line("s_out[0]=ke; s_out[1]=pe; s_out[2]=ke+pe;")
            self.gen_add_end_control_flow()
        self.gen_add_sync()

    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs(in_name, str(input_count), stride="stride_" + in_name)
        _repoint(in_loop=True)
        self.gen_add_code_line("// compute")
        _compute()
        self.gen_kernel_save_result("out", str(out_size), stride=str(out_size))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs(in_name, str(input_count))
        _repoint(in_loop=False)
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload(in_name, str(input_count), feedback_from="out")
        _compute()
        self.gen_anti_licm_output_write("out")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("out", str(out_size))
    self.gen_add_end_function()


def _gen_kin_centroidal_host(self, name, out_buf, out_size, has_qd, has_gravity, mode=0):
    n = self.robot.get_num_pos()
    macro = name.upper() + "_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()"
    single_call_timing = (mode == 1)
    compute_only = (mode == 2)
    in_name = "q_qd" if has_qd else "q"
    grav_param = "const T gravity, " if has_gravity else ""
    grav_arg = "gravity," if has_gravity else ""
    func_def_start = ("void " + name + "(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, " + grav_param +
                      "const int num_timesteps,")
    func_def_end = "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    self.gen_add_func_doc("Compute " + name, [], [], None)
    # MUJOCO_OUTPUT (floating only) host flag, LAST: forwarded to the kernel launch
    # naming the tier positionally (<T, GRID_DEFAULT_RESOURCE_TIER, MUJOCO_OUTPUT>) to
    # reach the trailing flag. Default false -> byte-identical pin codegen.
    mjx_host = self.robot.floating_base
    if mjx_host:
        self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    else:
        self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, \"" + name + " requires all-data or kinematics gridData\");")
    if mjx_host:
        ktmpl = "<T, RESOURCE_TIER, MUJOCO_OUTPUT>"
        kname = name + ("_kernel_single_timing" if single_call_timing else "_kernel") + ktmpl
    else:
        kname = name + ("_kernel_single_timing<T, RESOURCE_TIER>" if single_call_timing else "_kernel<T, RESOURCE_TIER>")
    func_call = (kname + "<<<block_dimms,thread_dimms," + macro + ">>>(hd_data->" + out_buf + ",hd_data->d_workspace,hd_data->d_" + in_name +
                 ",stride_" + in_name + ",d_robotModel," + grav_arg + "num_timesteps);")
    if not compute_only:
        if has_qd:
            self.gen_add_code_lines([
                "// start code with memory transfer", "int stride_q_qd;",
                "if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*" +
                ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}",
                "else {stride_q_qd = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*" +
                ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}"])
        else:
            self.gen_add_code_lines([
                "// start code with memory transfer", "int stride_q = NUM_JOINTS;",
                "gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*" +
                ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));"])
    else:
        if has_qd:
            self.gen_add_code_line("int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS : 3*NUM_JOINTS;")
        else:
            self.gen_add_code_line("int stride_q = NUM_JOINTS;")
    self.gen_add_code_line("// then call the kernel")
    if has_qd:
        func_call_mem = "if (USE_COMPRESSED_MEM) {" + func_call + "}"
        func_call_mem2 = "else                    {" + func_call.replace("hd_data->d_q_qd", "hd_data->d_q_qd_u") + "}"
        func_call_code = [func_call_mem, func_call_mem2, "gpuErrchkKernel();"]
    else:
        func_call_code = [func_call, "gpuErrchkKernel();"]
    if single_call_timing:
        func_call_code.insert(0, "struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    # DE-GATE #2: L2-pin d_workspace when the chosen tier spills the Jw band into it.
    ws_bytes = ("GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()" if single_call_timing
                else "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(num_timesteps)")
    self.gen_add_code_line("if (!" + name.upper() + "_J_IN_SMEM<RESOURCE_TIER>() && hd_data->d_workspace != nullptr) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, " + ws_bytes + "));}")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"" + name + "\", " + macro + "));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back",
            "gpuErrchk(cudaMemcpy(hd_data->h" + out_buf[1:] + ",hd_data->" + out_buf + "," + str(out_size) + "*" +
            ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();"])
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line(name))
    self.gen_add_end_function()


def gen_com(self):
    nv = self.robot.get_num_vel()
    out_size = 3 + 3 * nv
    self.gen_com_device()
    self._gen_kin_centroidal_kernel("com", out_size, has_qd=False, has_gravity=False, single_call_timing=True)
    self._gen_kin_centroidal_kernel("com", out_size, has_qd=False, has_gravity=False, single_call_timing=False)
    self._gen_kin_centroidal_host("com", "d_com", out_size, has_qd=False, has_gravity=False, mode=0)
    self._gen_kin_centroidal_host("com", "d_com", out_size, has_qd=False, has_gravity=False, mode=1)
    self._gen_kin_centroidal_host("com", "d_com", out_size, has_qd=False, has_gravity=False, mode=2)


def gen_ccrba(self):
    nv = self.robot.get_num_vel()
    out_size = 6 * nv + 6
    self.gen_ccrba_device()
    self._gen_kin_centroidal_kernel("ccrba", out_size, has_qd=True, has_gravity=False, single_call_timing=True)
    self._gen_kin_centroidal_kernel("ccrba", out_size, has_qd=True, has_gravity=False, single_call_timing=False)
    self._gen_kin_centroidal_host("ccrba", "d_ccrba", out_size, has_qd=True, has_gravity=False, mode=0)
    self._gen_kin_centroidal_host("ccrba", "d_ccrba", out_size, has_qd=True, has_gravity=False, mode=1)
    self._gen_kin_centroidal_host("ccrba", "d_ccrba", out_size, has_qd=True, has_gravity=False, mode=2)


def gen_energy(self):
    self.gen_energy_device()
    self._gen_kin_centroidal_kernel("energy", 3, has_qd=True, has_gravity=True, single_call_timing=True)
    self._gen_kin_centroidal_kernel("energy", 3, has_qd=True, has_gravity=True, single_call_timing=False)
    self._gen_kin_centroidal_host("energy", "d_energy", 3, has_qd=True, has_gravity=True, mode=0)
    self._gen_kin_centroidal_host("energy", "d_energy", 3, has_qd=True, has_gravity=True, mode=1)
    self._gen_kin_centroidal_host("energy", "d_energy", 3, has_qd=True, has_gravity=True, mode=2)

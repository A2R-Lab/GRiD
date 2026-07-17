"""Runtime arbitrary multi-EE pose + pose-gradient with target/offset (additive).

NEW, additive algorithm family. Nothing in the existing emit path is touched:
these are emitted only when the ``end_effector_pose_runtime`` /
``end_effector_pose_gradient_runtime`` keys are selected, mirroring how
``frame_jacobian`` gates on its deps. The existing baked-leaf
``end_effector_pose`` / ``end_effector_pose_gradient`` paths are unchanged.

Lives in the kinematics (s_XmatsHom) domain. Both surfaces take a RUNTIME
``target_jid`` (joint id whose frame is the EE; -1 => leaf default) and a runtime
``offset[3]`` point in the target frame (null / {0,0,0} => frame origin). They are
a direct CUDA transcription of the RBDReference numpy oracle
(``RBDReference.end_effector_pose`` / ``.end_effector_pose_gradient`` with the
runtime ``ee_joint_names`` / ``ee_offsets`` list API), which matches pinocchio /
the analytic geometric Jacobian to float64 rounding.

  * ``end_effector_pose_runtime``          : 6-vector [xyz; rpy] of target_jid at
                                             the offset-shifted point.
  * ``end_effector_pose_gradient_runtime`` : 6 x NUM_VEL = d[xyz; rpy]/dv, the
                                             geometric Jacobian at the offset point
                                             then [Jv; E(rpy)^-1 Jw].

Single-target kernels (like frame_jacobian); the Python binding loops them over a
jid list for the multi-EE list API. Output 6 / 6*nv floats -> NO spill ladder.
"""

import numpy as np

from ._frame_jacobian import _emit_world_transform_chainup


__all__ = [
    "_gen_runtime_host",
    "gen_end_effector_pose_runtime_inner",
    "gen_end_effector_pose_runtime_device",
    "gen_end_effector_pose_runtime_kernel",
    "gen_end_effector_pose_runtime_host",
    "gen_end_effector_pose_runtime",
    "gen_end_effector_pose_gradient_runtime_inner",
    "gen_end_effector_pose_gradient_runtime_device",
    "gen_end_effector_pose_gradient_runtime_kernel",
    "gen_end_effector_pose_gradient_runtime_host",
    "gen_end_effector_pose_gradient_runtime",
]


def _runtime_inner_temp_mem_size(self):
    # world homogeneous transform per joint (16 each) -- identical to
    # _frame_jacobian_inner_temp_mem_size.
    return 16 * self.robot.get_num_joints()


# =====================================================================
# end_effector_pose_runtime (POSE)
# =====================================================================

def gen_end_effector_pose_runtime_inner(self):
    """Emit end_effector_pose_runtime_inner: build per-joint world transforms then
    the 6-vector pose [xyz; rpy] of `target_jid` at the offset-shifted point.
    The offset shifts the position by R_target * offset; rpy is unchanged."""
    func_params = [
        "s_eePose is the output 6-vector pose [xyz; rpy] of target_jid",
        "target_jid is the joint id whose frame pose is requested",
        "s_offset is the 3-vector point offset in the target frame (frame origin if {0,0,0})",
        "s_q is the vector of joint positions (unused; baked into s_Xhom)",
        "s_Xhom is the per-joint LOCAL homogeneous transforms",
        "d_robotModel is the GPU model helpers",
        "s_temp is scratch of size " + str(_runtime_inner_temp_mem_size(self))]
    func_def_middle = ("T *s_eePose, const int target_jid, const T *s_offset, "
                       "const T *s_q, const T *s_Xhom, const robotModel<T> *d_robotModel, ")
    func_def = "void end_effector_pose_runtime_inner(" + func_def_middle + "T *s_temp) {"
    self.gen_add_func_doc("Compute a runtime-target end-effector pose [xyz; rpy] at an offset point",
                          [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("(void)s_q;")

    # Step 1: world homogeneous transforms (chain-up).
    _emit_world_transform_chainup(self)

    # Step 2: extract pose from the target's world transform (serial; tiny).
    self.gen_add_code_line("// Step 2: pose = [ (Xworld[target] * [offset,1])[:3] ; rpy(Xworld[target]) ]")
    self.gen_add_serial_ops()
    self.gen_add_code_line("const T *Xf = &s_Xworld[16*target_jid];")
    # column-major: R[r,c] = Xf[r + 4*c], p = Xf[12..14]. pos = R*offset + p.
    self.gen_add_code_line("T ox = s_offset[0], oy = s_offset[1], oz = s_offset[2];")
    self.gen_add_code_line("s_eePose[0] = Xf[0]*ox + Xf[4]*oy + Xf[8]*oz  + Xf[12];")
    self.gen_add_code_line("s_eePose[1] = Xf[1]*ox + Xf[5]*oy + Xf[9]*oz  + Xf[13];")
    self.gen_add_code_line("s_eePose[2] = Xf[2]*ox + Xf[6]*oy + Xf[10]*oz + Xf[14];")
    # rpy from rotation block (offset does NOT change rpy). Matches the baked-leaf
    # extraction (_eepose_gradient_hessian Step-4 / the eePos_from_Xmat_hom oracle):
    #   roll  = atan2(R[2,1], R[2,2])  -> Xf[6], Xf[10]
    #   pitch = -atan2(R[2,0], sqrt(R[2,2]^2 + R[2,1]^2)) -> Xf[2], Xf[10], Xf[6]
    #   yaw   = atan2(R[1,0], R[0,0])  -> Xf[1], Xf[0]
    self.gen_add_code_line("s_eePose[3] = atan2(Xf[6], Xf[10]);")
    self.gen_add_code_line("s_eePose[4] = -atan2(Xf[2], sqrt(Xf[10]*Xf[10] + Xf[6]*Xf[6]));")
    self.gen_add_code_line("s_eePose[5] = atan2(Xf[1], Xf[0]);")
    self.gen_add_end_control_flow()  # serial
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_end_effector_pose_runtime_device(self):
    """Auto-smem device wrapper around end_effector_pose_runtime_inner."""
    func_def = ("void end_effector_pose_runtime_device(T *s_eePose, const int target_jid, "
                "const T *s_offset, const T *s_q, const robotModel<T> *d_robotModel) {")
    func_params = ["s_eePose holds the 6-vector pose [xyz; rpy]",
                   "target_jid is the joint id of the frame",
                   "s_offset is the 3-vector point offset in the target frame",
                   "s_q is the joint position vector",
                   "d_robotModel is the GPU model helpers"]
    self.gen_add_func_doc("Compute a runtime-target end-effector pose at an offset point",
                          [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _runtime_inner_temp_mem_size(self),
        include_linalg_scratch=True, linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_add_code_line("end_effector_pose_runtime_inner<T>(s_eePose, target_jid, s_offset, s_q, s_XmatsHom, d_robotModel, s_temp);")
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_end_effector_pose_runtime_kernel(self, single_call_timing=False):
    """Emit end_effector_pose_runtime_kernel: batched (one block per timestep)
    launcher. target_jid + offset[3] are RUNTIME kernel parameters; the host
    defaults target_jid to the leaf-EE joint and offset to {0,0,0}. Output is a
    6-vector pose per timestep."""
    n = self.robot.get_num_pos()
    func_params = ["d_eePose is the vector of 6-vector poses [xyz; rpy]",
                   "d_q is the vector of joint positions",
                   "stride_q is the stride between each q",
                   "target_jid is the joint id whose pose is requested (runtime)",
                   "d_offset is the 3-vector point offset in the target frame (runtime)",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
                   "num_timesteps is the length of the trajectory points (or overloaded as test_iters for timing)"]
    func_def_start = ("void end_effector_pose_runtime_kernel(T *d_eePose, const T *d_q, const int stride_q, "
                      "const int target_jid, const T *d_offset, ")
    func_def_end = "const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("(", "_single_timing(")
    self.gen_add_func_doc("Compute a runtime-target end-effector pose at an offset point",
                          [], func_params, None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag, LAST
    # after RESOURCE_TIER so existing positional <T,TIER> call sites are unaffected.
    # The world EE pose ([xyz; rpy]) is frame-INVARIANT regardless of the runtime
    # target, so there is NO output epilogue; only the base quaternion is reordered
    # (mjx wxyz -> pin xyzw) so the XmatsHom build forms X[0] from the correct
    # orientation. Default false -> byte-identical pin codegen.
    mjx = self.robot.floating_base
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # cache the runtime offset in shared so every thread/inner reads from smem.
    self.gen_add_code_line("__shared__ T s_offset[3];")
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _runtime_inner_temp_mem_size(self),
        extra_t_buffers=[("s_q", n), ("s_eePose", 6)],
        include_linalg_scratch=True, linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    self.gen_add_code_line("if (threadIdx.x == 0 && threadIdx.y == 0) { s_offset[0]=d_offset[0]; s_offset[1]=d_offset[1]; s_offset[2]=d_offset[2]; }")
    self.gen_add_sync()
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q", str(n), stride="stride_q")
        # mjx input convert (quaternion only): reorder the base quaternion wxyz->xyzw
        # before XmatsHom builds X[0]; the world EE pose is frame-INVARIANT (no output
        # epilogue).
        if mjx:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_quat_reorder("s_q")
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute")
        self.gen_load_update_XmatsHom_helpers_function_call()
        self.gen_add_code_line("end_effector_pose_runtime_inner<T>(s_eePose, target_jid, s_offset, s_q, s_XmatsHom, d_robotModel, s_temp);")
        self.gen_add_sync()
        self.gen_kernel_save_result("eePose", "6", stride="6")
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q", str(n))
        # mjx input convert (quaternion only): see batch branch above. Reorder once
        # before the rep loop so X[0] builds from the correct base orientation.
        if mjx:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_quat_reorder("s_q")
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q", str(n), feedback_from="eePose")
        self.gen_load_update_XmatsHom_helpers_function_call()
        self.gen_add_code_line("end_effector_pose_runtime_inner<T>(s_eePose, target_jid, s_offset, s_q, s_XmatsHom, d_robotModel, s_temp);")
        self.gen_anti_licm_output_write("eePose")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("eePose", "6")
    self.gen_add_end_function()


def _gen_runtime_host(self, base_name, out_field, out_count, single_call_timing=False, compute_only=False):
    """Shared host-launcher emitter for the two runtime pose surfaces (pose:
    out_count='6'; gradient: out_count='6*NUM_VEL'). target_jid defaults to the
    leaf-EE joint (-1 sentinel) and the offset to {0,0,0} (the runtime offset is
    fed from hd_data->d_eepose_runtime_offset, host-initialized to zero)."""
    default_tjid = self.robot.get_leaf_nodes()[0]
    smem = base_name.upper() + "_DYNAMIC_SHARED_MEM_BYTES<T>()"
    func_params = ["hd_data is the packaged input and output pointers",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
                   "num_timesteps is the length of the trajectory points (or overloaded as test_iters for timing)",
                   "streams are pointers to CUDA streams for async memory transfers (if needed)",
                   "target_jid is the joint id of the requested frame (default leaf-EE)"]
    # target_jid is a trailing defaulted param (offset is a fixed device buffer
    # initialized to zero; a runtime offset is set by the binding before the call).
    frame_args = ", int target_jid = " + str(default_tjid)
    func_def_start = ("void " + base_name + "(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, "
                      "const int num_timesteps,")
    func_def_end = "                            const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams" + frame_args + ") {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_end = "                            const dim3 block_dimms, const dim3 thread_dimms" + frame_args + ") {"
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end
    self.gen_add_func_doc("Compute a runtime-target end-effector pose surface", [], func_params, None)
    # MUJOCO_OUTPUT (floating only) host flag, LAST: forwarded to the kernel launch
    # (naming the tier positionally to reach the trailing flag). The value surface is
    # frame-INVARIANT (input quat-reorder only); the gradient surface reframes its
    # base-linear columns by R^T. Default false -> byte-identical pin codegen.
    mjx_host = self.robot.floating_base
    if mjx_host:
        self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    else:
        self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, \"" + base_name + " requires all-data or kinematics gridData\");")
    self.gen_add_code_line("if (target_jid < 0) { target_jid = " + str(default_tjid) + "; }       // -1 => leaf-EE default")
    kernel_tmpl = (base_name + "_kernel"
                   + ("<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "<T, RESOURCE_TIER>"))
    func_call_start = (kernel_tmpl + "<<<block_dimms,thread_dimms," + smem + ">>>"
                       "(hd_data->d_" + out_field + ",hd_data->d_q,stride_q,target_jid,hd_data->d_eepose_runtime_offset,")
    func_call_end = "d_robotModel,num_timesteps);"
    if single_call_timing:
        if mjx_host:
            func_call_start = func_call_start.replace(base_name + "_kernel<", base_name + "_kernel_single_timing<")
        else:
            func_call_start = func_call_start.replace("kernel<T, RESOURCE_TIER>", "kernel_single_timing<T, RESOURCE_TIER>")
    if not compute_only:
        self.gen_add_code_lines(["// start code with memory transfer",
                                 "int stride_q;",
                                 "if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; " +
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*" +
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}",
                                 "else {stride_q = 3*NUM_JOINTS; " +
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*" +
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}",
                                 "gpuErrchkKernel();"])
    else:
        self.gen_add_code_line("int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;")
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    func_call_mem_adjust = "if (USE_COMPRESSED_MEM) {" + func_call + "}"
    func_call_mem_adjust2 = "else                    {" + func_call.replace("hd_data->d_q,", "hd_data->d_q_qd_u,") + "}"
    func_call_code = [func_call_mem_adjust, func_call_mem_adjust2, "gpuErrchkKernel();"]
    if single_call_timing:
        func_call_code.insert(0, "struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"" + base_name + "\", " + smem + "));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        self.gen_add_code_lines(["// finally transfer the result back",
                                 "gpuErrchk(cudaMemcpy(hd_data->h_" + out_field + ",hd_data->d_" + out_field + "," + out_count + "*" +
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchkKernel();"])
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line(base_name))
    self.gen_add_end_function()


def gen_end_effector_pose_runtime_host(self, mode=0):
    self._gen_runtime_host("end_effector_pose_runtime", "eePose", "6",
                           single_call_timing=(mode == 1), compute_only=(mode == 2))


def gen_end_effector_pose_runtime(self):
    self.gen_end_effector_pose_runtime_inner()
    self.gen_end_effector_pose_runtime_device()
    self.gen_end_effector_pose_runtime_kernel(single_call_timing=False)
    self.gen_end_effector_pose_runtime_kernel(single_call_timing=True)
    self.gen_end_effector_pose_runtime_host(mode=0)
    self.gen_end_effector_pose_runtime_host(mode=1)
    self.gen_end_effector_pose_runtime_host(mode=2)


# =====================================================================
# end_effector_pose_gradient_runtime (POSE GRADIENT)
# =====================================================================

def gen_end_effector_pose_gradient_runtime_inner(self):
    """Emit end_effector_pose_gradient_runtime_inner: build per-joint world
    transforms, assemble the geometric Jacobian [Jv; Jw] of `target_jid` with the
    lever arm at the OFFSET-shifted point p_ee = p_target + R_target*offset, then
    write [Jv; E(rpy)^-1 Jw] (6 x NUM_VEL). Mirrors the frame_jacobian job-table
    (runtime target filter) + the eepose_gradient rpy/E^-1 write. The mimic shared
    v-slot alpha-fold is folded into the per-job accumulate (gated on HAS_MIMIC so
    non-mimic emit is byte-identical)."""
    NJ = self.robot.get_num_joints()
    nv = self.robot.get_num_vel()
    HAS_MIMIC = self.robot_has_mimic_joints()

    func_params = [
        "s_grad is the output 6 x NUM_VEL gradient d[xyz; rpy]/dv (column-major)",
        "target_jid is the joint id whose pose gradient is requested",
        "s_offset is the 3-vector point offset in the target frame",
        "s_q is the vector of joint positions (unused; baked into s_Xhom)",
        "s_Xhom is the per-joint LOCAL homogeneous transforms",
        "d_robotModel is the GPU model helpers",
        "s_temp is scratch of size " + str(_runtime_inner_temp_mem_size(self))]
    func_def_middle = ("T *s_grad, const int target_jid, const T *s_offset, "
                       "const T *s_q, const T *s_Xhom, const robotModel<T> *d_robotModel, ")
    func_def = "void end_effector_pose_gradient_runtime_inner(" + func_def_middle + "T *s_temp) {"
    self.gen_add_func_doc("Compute a runtime-target end-effector pose gradient at an offset point",
                          [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("(void)s_q;")

    # Step 1: world homogeneous transforms (chain-up).
    _emit_world_transform_chainup(self)

    # Step 2: zero Jv/Jw (stored in s_grad rows; we accumulate into a temp Jw
    # band, then convert). We assemble Jv directly into rows 0..2 of s_grad and
    # Jw into rows 3..5, then in Step 4 rewrite rows 3..5 = E^-1 * Jw in place.
    self.gen_add_parallel_loop("ind", str(6 * nv))
    self.gen_add_code_line("s_grad[ind] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Step 3: geometric Jacobian at the offset-shifted point p_ee, world axes.
    # Clone of the frame_jacobian job-table (runtime target filter). p_ee =
    # p_target + R_target * offset; the lever arm uses p_ee instead of the frame
    # origin. Rows 0..2 of s_grad <- Jv, rows 3..5 <- Jw (world). The mimic
    # alpha-fold is the per-job multiply (HAS_MIMIC only).
    self.gen_add_code_line("// Step 3: geometric Jacobian at offset point p_ee = p_target + R_target*offset, world axes")
    jobs = []  # (target_jid, jj, vi, ang[3], lin[3], alpha)
    for jid in range(NJ):
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
        tj_arr = ", ".join(str(j[0]) for j in jobs)
        jj_arr = ", ".join(str(j[1]) for j in jobs)
        vi_arr = ", ".join(str(j[2]) for j in jobs)
        ax_arr = ", ".join("static_cast<T>({:.17g})".format(v) for j in jobs for v in j[3])
        lx_arr = ", ".join("static_cast<T>({:.17g})".format(v) for j in jobs for v in j[4])
        self.gen_add_code_line("const int epg_target[" + str(njobs) + "] = {" + tj_arr + "};")
        self.gen_add_code_line("const int epg_jj[" + str(njobs) + "] = {" + jj_arr + "};")
        self.gen_add_code_line("const int epg_vi[" + str(njobs) + "] = {" + vi_arr + "};")
        self.gen_add_code_line("const T epg_ang[" + str(3 * njobs) + "] = {" + ax_arr + "};")
        self.gen_add_code_line("const T epg_lin[" + str(3 * njobs) + "] = {" + lx_arr + "};")
        if HAS_MIMIC:
            al_arr = ", ".join("static_cast<T>({:.17g})".format(j[5]) for j in jobs)
            self.gen_add_code_line("const T epg_alpha[" + str(njobs) + "] = {" + al_arr + "};")
        self.gen_add_serial_ops()
        # p_ee = p_target + R_target * offset (column-major target block).
        self.gen_add_code_line("const T *Xf = &s_Xworld[16*target_jid];")
        self.gen_add_code_line("T ox = s_offset[0], oy = s_offset[1], oz = s_offset[2];")
        self.gen_add_code_line("T pex = Xf[0]*ox + Xf[4]*oy + Xf[8]*oz  + Xf[12];")
        self.gen_add_code_line("T pey = Xf[1]*ox + Xf[5]*oy + Xf[9]*oz  + Xf[13];")
        self.gen_add_code_line("T pez = Xf[2]*ox + Xf[6]*oy + Xf[10]*oz + Xf[14];")
        self.gen_add_code_line("for (int t = 0; t < " + str(njobs) + "; ++t) {", True)
        self.gen_add_code_line("if (epg_target[t] != target_jid) continue;")
        self.gen_add_code_line("int jj = epg_jj[t]; int vi = epg_vi[t];")
        self.gen_add_code_line("const T *Xj = &s_Xworld[16*jj];")
        self.gen_add_code_line("T a0=epg_ang[3*t], a1=epg_ang[3*t+1], a2=epg_ang[3*t+2];")
        self.gen_add_code_line("T l0=epg_lin[3*t], l1=epg_lin[3*t+1], l2=epg_lin[3*t+2];")
        # world axes: aw = R_jj * ang ; lw = R_jj * lin
        self.gen_add_code_line("T aw0 = Xj[0]*a0 + Xj[4]*a1 + Xj[8]*a2;")
        self.gen_add_code_line("T aw1 = Xj[1]*a0 + Xj[5]*a1 + Xj[9]*a2;")
        self.gen_add_code_line("T aw2 = Xj[2]*a0 + Xj[6]*a1 + Xj[10]*a2;")
        self.gen_add_code_line("T lw0 = Xj[0]*l0 + Xj[4]*l1 + Xj[8]*l2;")
        self.gen_add_code_line("T lw1 = Xj[1]*l0 + Xj[5]*l1 + Xj[9]*l2;")
        self.gen_add_code_line("T lw2 = Xj[2]*l0 + Xj[6]*l1 + Xj[10]*l2;")
        self.gen_add_code_line("T pjx = Xj[12], pjy = Xj[13], pjz = Xj[14];")
        # linear at p_ee = lw + aw x (p_ee - p_jj)
        self.gen_add_code_line("T dx = pex - pjx, dy = pey - pjy, dz = pez - pjz;")
        self.gen_add_code_line("T linf0 = lw0 + (aw1*dz - aw2*dy);")
        self.gen_add_code_line("T linf1 = lw1 + (aw2*dx - aw0*dz);")
        self.gen_add_code_line("T linf2 = lw2 + (aw0*dy - aw1*dx);")
        # accumulate into s_grad column vi: rows 0..2 = Jv, rows 3..5 = Jw.
        self.gen_add_code_line("T *Gc = &s_grad[6*vi];")
        if HAS_MIMIC:
            self.gen_add_code_line("T al = epg_alpha[t];")
            self.gen_add_code_line("Gc[0]+=al*linf0; Gc[1]+=al*linf1; Gc[2]+=al*linf2; Gc[3]+=al*aw0; Gc[4]+=al*aw1; Gc[5]+=al*aw2;")
        else:
            self.gen_add_code_line("Gc[0]+=linf0; Gc[1]+=linf1; Gc[2]+=linf2; Gc[3]+=aw0; Gc[4]+=aw1; Gc[5]+=aw2;")
        self.gen_add_end_control_flow()  # for t
        self.gen_add_end_control_flow()  # serial
        self.gen_add_sync()

    # Step 4: rewrite rows 3..5 of each column = E(rpy)^-1 * Jw (in place).
    # E^-1 for R = Rz(yaw)Ry(pitch)Rx(roll); rpy from the target rotation block.
    # Mirrors _eepose_gradient_hessian Step-5 (rows 3..5).
    self.gen_add_code_line("// Step 4: rows 3..5 <- E(rpy)^-1 * Jw (rpy from target world rotation)")
    self.gen_add_serial_ops()
    self.gen_add_code_line("const T *Xf2 = &s_Xworld[16*target_jid];")
    self.gen_add_code_line("T R20 = Xf2[2];  T R21 = Xf2[6];  T R22 = Xf2[10];")
    self.gen_add_code_line("T R10 = Xf2[1];  T R00 = Xf2[0];")
    self.gen_add_code_line("T yaw = atan2(R10, R00);")
    self.gen_add_code_line("T pitch = atan2(-R20, sqrt(R22*R22 + R21*R21));")
    self.gen_add_code_line("T cy = cos(yaw), sy = sin(yaw), cp = cos(pitch), sp = sin(pitch);")
    self.gen_add_code_line("for (int vi = 0; vi < " + str(nv) + "; ++vi) {", True)
    self.gen_add_code_line("T *Gc = &s_grad[6*vi];")
    self.gen_add_code_line("T Jw0 = Gc[3], Jw1 = Gc[4], Jw2 = Gc[5];")
    self.gen_add_code_line("Gc[3] = (cy*Jw0 + sy*Jw1) / cp;")
    self.gen_add_code_line("Gc[4] = -sy*Jw0 + cy*Jw1;")
    self.gen_add_code_line("Gc[5] = (sp / cp) * (cy*Jw0 + sy*Jw1) + Jw2;")
    self.gen_add_end_control_flow()  # for vi
    self.gen_add_end_control_flow()  # serial
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_end_effector_pose_gradient_runtime_device(self):
    """Auto-smem device wrapper around end_effector_pose_gradient_runtime_inner."""
    func_def = ("void end_effector_pose_gradient_runtime_device(T *s_grad, const int target_jid, "
                "const T *s_offset, const T *s_q, const robotModel<T> *d_robotModel) {")
    func_params = ["s_grad holds the 6 x NUM_VEL pose gradient (column-major)",
                   "target_jid is the joint id of the frame",
                   "s_offset is the 3-vector point offset in the target frame",
                   "s_q is the joint position vector",
                   "d_robotModel is the GPU model helpers"]
    self.gen_add_func_doc("Compute a runtime-target end-effector pose gradient at an offset point",
                          [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _runtime_inner_temp_mem_size(self),
        include_linalg_scratch=True, linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_add_code_line("end_effector_pose_gradient_runtime_inner<T>(s_grad, target_jid, s_offset, s_q, s_XmatsHom, d_robotModel, s_temp);")
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_end_effector_pose_gradient_runtime_kernel(self, single_call_timing=False):
    """Emit end_effector_pose_gradient_runtime_kernel: batched launcher. target_jid
    + offset[3] are RUNTIME kernel parameters. Output is 6 x NUM_VEL per timestep."""
    n = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    func_params = ["d_eePoseGrad is the vector of 6 x NUM_VEL pose gradients (column-major)",
                   "d_q is the vector of joint positions",
                   "stride_q is the stride between each q",
                   "target_jid is the joint id whose pose gradient is requested (runtime)",
                   "d_offset is the 3-vector point offset in the target frame (runtime)",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
                   "num_timesteps is the length of the trajectory points (or overloaded as test_iters for timing)"]
    func_def_start = ("void end_effector_pose_gradient_runtime_kernel(T *d_eePoseGrad, const T *d_q, const int stride_q, "
                      "const int target_jid, const T *d_offset, ")
    func_def_end = "const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("(", "_single_timing(")
    self.gen_add_func_doc("Compute a runtime-target end-effector pose gradient at an offset point",
                          [], func_params, None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag, LAST
    # after RESOURCE_TIER so existing positional <T,TIER> call sites are unaffected.
    # The EE-pose Jacobian's base-linear COLUMNS reframe by R^T (J G^{-1}); the
    # 6 x nv output is a single-target block (runtime target => single ee), so the
    # epilogue is one column-reframe. Plus a base quaternion reorder on input.
    # Default false -> byte-identical pin codegen.
    mjx = self.robot.floating_base
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("__shared__ T s_offset[3];")
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _runtime_inner_temp_mem_size(self),
        extra_t_buffers=[("s_q", n), ("s_eePoseGrad", 6 * nv)],
        include_linalg_scratch=True, linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()")
    self.gen_add_code_line("if (threadIdx.x == 0 && threadIdx.y == 0) { s_offset[0]=d_offset[0]; s_offset[1]=d_offset[1]; s_offset[2]=d_offset[2]; }")
    self.gen_add_sync()
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q", str(n), stride="stride_q")
        # mjx input convert (quaternion only): reorder base quaternion wxyz->xyzw
        # before XmatsHom builds X[0]; the Jacobian output gets a column-reframe below.
        if mjx:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_quat_reorder("s_q")
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute")
        self.gen_load_update_XmatsHom_helpers_function_call()
        self.gen_add_code_line("end_effector_pose_gradient_runtime_inner<T>(s_eePoseGrad, target_jid, s_offset, s_q, s_XmatsHom, d_robotModel, s_temp);")
        self.gen_add_sync()
        # mjx output: column reframe J G^{-1} (base-linear cols . R^T) of the single
        # 6 x nv ee block, in place on s_eePoseGrad.
        if mjx:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_column_reframe("s_eePoseGrad", 6, nv)
            self.gen_add_end_control_flow()
        self.gen_kernel_save_result("eePoseGrad", str(6 * nv), stride=str(6 * nv))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q", str(n))
        # mjx input convert (quaternion only): see batch branch. Reorder once before
        # the rep loop so X[0] builds from the correct base orientation.
        if mjx:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_quat_reorder("s_q")
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q", str(n), feedback_from="eePoseGrad")
        self.gen_load_update_XmatsHom_helpers_function_call()
        self.gen_add_code_line("end_effector_pose_gradient_runtime_inner<T>(s_eePoseGrad, target_jid, s_offset, s_q, s_XmatsHom, d_robotModel, s_temp);")
        # mjx output: column reframe J G^{-1} (base-linear cols . R^T) per ee block.
        if mjx:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_column_reframe("s_eePoseGrad", 6, nv)
            self.gen_add_end_control_flow()
        self.gen_anti_licm_output_write("eePoseGrad")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("eePoseGrad", str(6 * nv))
    self.gen_add_end_function()


def gen_end_effector_pose_gradient_runtime_host(self, mode=0):
    self._gen_runtime_host("end_effector_pose_gradient_runtime", "eePoseGrad", "6*NUM_VEL",
                           single_call_timing=(mode == 1), compute_only=(mode == 2))


def gen_end_effector_pose_gradient_runtime(self):
    self.gen_end_effector_pose_gradient_runtime_inner()
    self.gen_end_effector_pose_gradient_runtime_device()
    self.gen_end_effector_pose_gradient_runtime_kernel(single_call_timing=False)
    self.gen_end_effector_pose_gradient_runtime_kernel(single_call_timing=True)
    self.gen_end_effector_pose_gradient_runtime_host(mode=0)
    self.gen_end_effector_pose_gradient_runtime_host(mode=1)
    self.gen_end_effector_pose_gradient_runtime_host(mode=2)

"""General-frame geometric Jacobian + operational-space inertia codegen (E2).

NEW, additive algorithm family. Nothing in the existing emit path is touched:
the family is only emitted when the ``frame_jacobian`` key is selected, mirroring
how the centroidal quick-wins (`_centroidal.py`) gate on their grid:: deps.

Lives in the kinematics (s_XmatsHom) domain: one inner pass builds the world
homogeneous transform of every joint (BFS chain-up, identical to
`end_effector_pose` / `centroidal_inner`), then assembles the geometric Jacobian
of a chosen target joint at runtime for one of the three pinocchio reference
frames:

    LOCAL (0)               -- twist in the frame's own body axes
    WORLD (1)               -- spatial Jacobian at the world origin
    LOCAL_WORLD_ALIGNED (2) -- at the frame origin, world-aligned axes

Output J is 6 x NUM_VEL column-major, rows ordered [linear(3); angular(3)] to
match pinocchio's `getFrameJacobian` / `getJointJacobian`. A second device
function composes Lambda = (J Minv J^T)^{-1}, the 6x6 operational-space inertia.

This is a direct CUDA transcription of the RBDReference numpy oracle
(`RBDReference.frame_jacobian` / `.osc_inertia`), which agrees with pinocchio to
float64 rounding. Correctness-first (single-block, serial inner assembly); the
per-column independence is left for a future perf pass.
"""

import numpy as np

from grid_codegen.helpers._code_generation_helpers import gen_launch_pair, gen_emit_host_result_transfer, gen_workspace_repoint_line, host_mode_flags, mangle_host_func_defs, wrap_host_single_call_timing
from grid_codegen.helpers._code_generation_helpers import gen_host_wrapper_head
from grid_codegen.helpers._code_generation_helpers import host_q_compressed_input_transfer_lines


__all__ = [
    "gen_frame_jacobian_inner",
    "gen_frame_jacobian_device",
    "gen_frame_jacobian_kernel",
    "gen_frame_jacobian_host",
    "gen_frame_jacobian",
    "gen_frame_jacobian_dot_device",
    "gen_frame_jacobian_dot_kernel",
    "gen_frame_jacobian_dot_host",
    "gen_frame_jacobian_dot",
    "frame_kin_gradient_inner_temp_mem_size",
    "gen_frame_kinematics_gradient_inner",
    "gen_frame_velocity_gradient_device",
    "gen_frame_velocity_gradient",
    "gen_frame_acceleration_gradient_device",
    "gen_frame_acceleration_gradient",
    "gen_osc_inertia_device",
    "gen_osc_inertia_kernel",
    "gen_osc_inertia_host",
    "gen_osc_inertia", "gen_frame_velocity_device", "gen_frame_velocity", "gen_frame_acceleration_device", "gen_frame_acceleration"]


# Reference-frame enum (matches pin.ReferenceFrame ordering used by the host).
_REF_LOCAL = 0
_REF_WORLD = 1
_REF_LWA = 2


def _frame_jacobian_inner_temp_mem_size(self):
    NJ = self.robot.get_num_joints()
    # world homogeneous transform per joint (16 each).
    return 16 * NJ


def _emit_world_transform_chainup(self, declare_xworld = True):
    """Step 1: per-joint world homogeneous transforms by BFS level (chain-up of
    local s_Xhom into s_Xworld). The shared kinematics primitive behind
    frame_jacobian, the runtime-target ee-pose surfaces (_eepose_runtime), the
    centroidal inner and the potential-energy regressor (hygiene 12, 2026-09-24:
    the latter two carried verbatim copies). ``declare_xworld`` emits the
    ``s_Xworld = &s_temp[0]`` alias; callers that place s_Xworld themselves pass
    False."""
    n_bfs_levels = self.robot.get_max_bfs_level() + 1
    if declare_xworld:
        self.gen_add_code_line("T *s_Xworld = &s_temp[0];")
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


def frame_jacobian_jobs(self):
    """The baked column table of frame_jacobian_inner: one (target_jid, jj, vi,
    ang_local[3], lin_local[3], alpha) row per (frame joint, ancestor-or-self
    joint, DoF column); alpha is the mimic multiplier (1.0 without mimic joints).
    Shared with the grid_plant frame-velocity derivative so both read one source."""
    NJ = self.robot.get_num_joints()
    HAS_MIMIC = self.robot_has_mimic_joints()
    jobs = []
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
            if alpha == 0.0:
                continue  # Constant bodies own no Jacobian column.
            for c in range(S.shape[1]):
                vi = vinds[c] if c < len(vinds) else vinds[-1]
                ang = [float(S[0, c]), float(S[1, c]), float(S[2, c])]
                lin = [float(S[3, c]), float(S[4, c]), float(S[5, c])]
                jobs.append((jid, jj, vi, ang, lin, alpha))
    return jobs


def gen_frame_jacobian_inner(self):
    """Emit frame_jacobian_inner: build per-joint world transforms then the
    geometric Jacobian (6 x NV, [linear; angular]) of `target_jid` in
    `reference_frame`. Correctness-first single-block assembly."""
    NJ = self.robot.get_num_joints()
    nv = self.robot.get_num_vel()

    func_params = [
        "s_J is the output 6 x NUM_VEL geometric Jacobian (column-major, [linear; angular])",
        "target_jid is the joint id whose frame Jacobian is requested",
        "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED",
        "s_q is the vector of joint positions (unused; baked into s_Xhom)",
        "s_Xhom is the per-joint LOCAL homogeneous transforms",
        "d_robotModel is the GPU model helpers",
        "s_temp is scratch of size " + str(_frame_jacobian_inner_temp_mem_size(self))]
    func_def_middle = ("T *s_J, const int target_jid, const int reference_frame, "
                       "const T *s_q, const T *s_Xhom, const robotModel<T> *d_robotModel, ")
    func_def = "void frame_jacobian_inner(" + func_def_middle + "T *s_temp) {"
    self.gen_add_func_doc("Compute a general-frame geometric Jacobian", [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("(void)s_q;")

    # ---- Step 1: world homogeneous transforms by BFS level (chain-up) ----
    _emit_world_transform_chainup(self)

    # ---- Step 2: zero the Jacobian ----
    self.gen_add_code_line("glass::set_const<T, " + str(6 * nv) + ">(static_cast<T>(0), s_J);")

    # ---- Step 3: assemble J at the frame ORIGIN with WORLD axes ----
    # Bake the per-target-joint chain jobs at codegen time, behind a runtime
    # `target_jid` switch. For each (target jid, ancestor-or-self jj, S-col c)
    # the world contribution to J[:, vi] is the screw of joint jj evaluated at
    # the frame origin p_f (= s_Xworld[16*target_jid + 12..14]).
    self.gen_add_code_line("// Step 3: geometric Jacobian at frame origin, world axes")
    # MIMIC fold: a mimic joint and its target share ONE velocity coordinate
    # (get_joint_index_v(mimic) == get_joint_index_v(target)), so several chain
    # joints accumulate into the same J column. The mimic body's generalized
    # velocity is alpha * v_target, so its geometric-Jacobian column contribution
    # is alpha-weighted (mirrors RBDReference.frame_jacobian's `scale = mimic_scale(j)`
    # applied to Jw/Jv, and the ee_pose_gradient Step 3b alpha-accumulate). The
    # serial accumulate already sums shared-vi columns; the only missing piece is
    # the per-job alpha. For a NON-mimic robot every alpha == 1.0, so the alpha
    # array/multiply is gated on HAS_MIMIC to keep non-mimic emit byte-identical.
    HAS_MIMIC = self.robot_has_mimic_joints()
    jobs = frame_jacobian_jobs(self)  # (target_jid, jj, vi, ang_local[3], lin_local[3], alpha)

    njobs = len(jobs)
    if njobs > 0:
        # Baked topology via gen_bake_const_array -> `static const` (off-stack;
        # agent_debugging_guide §1v). Consumers index [t] with no size dependence.
        self.gen_bake_const_array("fj_target", [j[0] for j in jobs], "int")
        self.gen_bake_const_array("fj_jj", [j[1] for j in jobs], "int")
        self.gen_bake_const_array("fj_vi", [j[2] for j in jobs], "int")
        self.gen_bake_const_array("fj_ang", [v for j in jobs for v in j[3]], "T")
        self.gen_bake_const_array("fj_lin", [v for j in jobs for v in j[4]], "T")
        if HAS_MIMIC:
            self.gen_bake_const_array("fj_alpha", [j[5] for j in jobs], "T")
        # Serial accumulation (correctness-first; columns may repeat vi).
        self.gen_add_serial_ops()
        # frame origin p_f in world.
        self.gen_add_code_line("const T *Xf = &s_Xworld[16*target_jid];")
        self.gen_add_code_line("T pfx = Xf[12], pfy = Xf[13], pfz = Xf[14];")
        self.gen_add_code_line("for (int t = 0; t < " + str(njobs) + "; ++t) {", True)
        self.gen_add_code_line("if (fj_target[t] != target_jid) continue;")
        self.gen_add_code_line("int jj = fj_jj[t]; int vi = fj_vi[t];")
        self.gen_add_code_line("const T *Xj = &s_Xworld[16*jj];")
        self.gen_add_code_line("T a0=fj_ang[3*t], a1=fj_ang[3*t+1], a2=fj_ang[3*t+2];")
        self.gen_add_code_line("T l0=fj_lin[3*t], l1=fj_lin[3*t+1], l2=fj_lin[3*t+2];")
        # world axes: aw = R_jj * ang_local ; lw = R_jj * lin_local
        self.gen_add_code_line("T aw0 = Xj[0]*a0 + Xj[4]*a1 + Xj[8]*a2;")
        self.gen_add_code_line("T aw1 = Xj[1]*a0 + Xj[5]*a1 + Xj[9]*a2;")
        self.gen_add_code_line("T aw2 = Xj[2]*a0 + Xj[6]*a1 + Xj[10]*a2;")
        self.gen_add_code_line("T lw0 = Xj[0]*l0 + Xj[4]*l1 + Xj[8]*l2;")
        self.gen_add_code_line("T lw1 = Xj[1]*l0 + Xj[5]*l1 + Xj[9]*l2;")
        self.gen_add_code_line("T lw2 = Xj[2]*l0 + Xj[6]*l1 + Xj[10]*l2;")
        # p_jj (world origin of joint jj)
        self.gen_add_code_line("T pjx = Xj[12], pjy = Xj[13], pjz = Xj[14];")
        # linear at the FRAME origin = lw + aw x (p_f - p_jj)
        self.gen_add_code_line("T dx = pfx - pjx, dy = pfy - pjy, dz = pfz - pjz;")
        self.gen_add_code_line("T linf0 = lw0 + (aw1*dz - aw2*dy);")
        self.gen_add_code_line("T linf1 = lw1 + (aw2*dx - aw0*dz);")
        self.gen_add_code_line("T linf2 = lw2 + (aw0*dy - aw1*dx);")
        # accumulate into J[:, vi] ([linear; angular] col-major 6 x NV).
        # MIMIC: scale this chain joint's column contribution by its multiplier
        # alpha (the mimic body moves alpha * v_target; non-mimic alpha == 1.0,
        # and the alpha term is omitted entirely so non-mimic emit is unchanged).
        self.gen_add_code_line("T *Jc = &s_J[6*vi];")
        if HAS_MIMIC:
            self.gen_add_code_line("T al = fj_alpha[t];")
            self.gen_add_code_line("Jc[0]+=al*linf0; Jc[1]+=al*linf1; Jc[2]+=al*linf2; Jc[3]+=al*aw0; Jc[4]+=al*aw1; Jc[5]+=al*aw2;")
        else:
            self.gen_add_code_line("Jc[0]+=linf0; Jc[1]+=linf1; Jc[2]+=linf2; Jc[3]+=aw0; Jc[4]+=aw1; Jc[5]+=aw2;")
        self.gen_add_end_control_flow()  # for t
        self.gen_add_end_control_flow()  # serial
        self.gen_add_sync()

    # ---- Step 4: apply the reference-frame transform in place ----
    # s_J currently holds the LOCAL_WORLD_ALIGNED Jacobian (frame origin, world
    # axes). Convert to WORLD or LOCAL when requested.
    self.gen_add_code_line("// Step 4: reference-frame transform (in place per column)")
    self.gen_add_serial_ops()
    self.gen_add_code_line("const T *Xf2 = &s_Xworld[16*target_jid];")
    self.gen_add_code_line("T Rf[9]; for (int c=0;c<3;++c) for (int r=0;r<3;++r) Rf[r+3*c] = Xf2[r + 4*c];")
    self.gen_add_code_line("T pfx2 = Xf2[12], pfy2 = Xf2[13], pfz2 = Xf2[14];")
    self.gen_add_code_line("for (int vi = 0; vi < " + str(nv) + "; ++vi) {", True)
    self.gen_add_code_line("T *Jc = &s_J[6*vi];")
    self.gen_add_code_line("T v0=Jc[0], v1=Jc[1], v2=Jc[2], w0=Jc[3], w1=Jc[4], w2=Jc[5];")
    # if/else-if chain emitted as raw lines (no auto control-flow bookkeeping).
    self.gen_add_code_lines([
        "if (reference_frame == " + str(_REF_WORLD) + ") {",
        # spatial Jacobian at world origin: v_world = v_lwa + p_f x w ; w unchanged.
        "  Jc[0] = v0 + (pfy2*w2 - pfz2*w1);",
        "  Jc[1] = v1 + (pfz2*w0 - pfx2*w2);",
        "  Jc[2] = v2 + (pfx2*w1 - pfy2*w0);",
        "} else if (reference_frame == " + str(_REF_LOCAL) + ") {",
        # rotate both blocks into the frame body axes: Jc <- Rf^T Jc
        "  Jc[0] = Rf[0]*v0 + Rf[1]*v1 + Rf[2]*v2;",
        "  Jc[1] = Rf[3]*v0 + Rf[4]*v1 + Rf[5]*v2;",
        "  Jc[2] = Rf[6]*v0 + Rf[7]*v1 + Rf[8]*v2;",
        "  Jc[3] = Rf[0]*w0 + Rf[1]*w1 + Rf[2]*w2;",
        "  Jc[4] = Rf[3]*w0 + Rf[4]*w1 + Rf[5]*w2;",
        "  Jc[5] = Rf[6]*w0 + Rf[7]*w1 + Rf[8]*w2;",
        "}",
    ])
    self.gen_add_end_control_flow()  # for vi
    self.gen_add_end_control_flow()  # serial
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_frame_jacobian_device(self):
    """Auto-smem device wrapper around frame_jacobian_inner."""
    nv = self.robot.get_num_vel()
    func_def = ("void frame_jacobian_device(T *s_J, const int target_jid, const int reference_frame, "
                "const T *s_q, const robotModel<T> *d_robotModel) {")
    func_params = ["s_J holds the 6 x NUM_VEL geometric Jacobian (column-major, [linear; angular])",
                   "target_jid is the joint id of the frame",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED",
                   "s_q is the joint position vector",
                   "d_robotModel is the GPU model helpers"]
    self.gen_add_func_doc("Compute a general-frame geometric Jacobian", [], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # s_J is caller-provided (function param); the arena only holds the XmatsHom
    # world-transform machinery + inner scratch.
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _frame_jacobian_inner_temp_mem_size(self))
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_add_code_line("frame_jacobian_inner<T>(s_J, target_jid, reference_frame, s_q, s_XmatsHom, d_robotModel, s_temp);")
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_frame_jacobian_kernel(self, single_call_timing=False):
    """Emit frame_jacobian_kernel: batched (one block per timestep) launcher of
    the general-frame geometric Jacobian. target_jid + reference_frame are
    RUNTIME kernel parameters (the inner already supports them); the host
    defaults them to the leaf-EE joint / LOCAL_WORLD_ALIGNED so a binding that
    doesn't pass a frame gets the historical fixed-target behavior, while a
    caller can request any frame at launch time. Output is 6 x NUM_VEL
    column-major per timestep."""
    n = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    func_params = ["d_frame_jacobian is the vector of 6 x NUM_VEL geometric Jacobians (column-major, [linear; angular])",
                   "d_q is the vector of joint positions",
                   "stride_q is the stride between each q",
                   "target_jid is the joint id whose frame Jacobian is requested (runtime)",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED (runtime)",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_def_start = ("void frame_jacobian_kernel(T *d_frame_jacobian, const T *d_q, const int stride_q, "
                      "const int target_jid, const int reference_frame, ")
    func_def_end = "const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("(", "_single_timing(")
    self.gen_add_func_doc("Compute a general-frame geometric Jacobian", [], func_params, None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag, LAST
    # after RESOURCE_TIER so existing positional <T,TIER> call sites are unaffected;
    # default false if-constexpr-elides the epilogue -> byte-identical PTX.
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # arena: world-transform machinery + inner scratch + s_q + s_frame_jacobian.
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _frame_jacobian_inner_temp_mem_size(self),
        extra_t_buffers=[("s_q", n), ("s_frame_jacobian", 6 * nv)])
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q", str(n), stride="stride_q")
        # mjx input convert (quaternion only -> the column-reframe epilogue's R;
        # J depends on q, so reorder before XmatsHom builds X[0] from the quat).
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_quat_reorder("s_q")
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute")
        self.gen_load_update_XmatsHom_helpers_function_call()
        self.gen_add_code_line("frame_jacobian_inner<T>(s_frame_jacobian, target_jid, reference_frame, s_q, s_XmatsHom, d_robotModel, s_temp);")
        self.gen_add_sync()
        # mjx output: geometric Jacobian column reframe J G^{-1} (base-linear cols . R^T)
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_column_reframe("s_frame_jacobian", 6, nv)
            self.gen_add_end_control_flow()
        self.gen_kernel_save_result("frame_jacobian", str(6 * nv), stride=str(6 * nv))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q", str(n))
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q", str(n), feedback_from="frame_jacobian")
        self.gen_load_update_XmatsHom_helpers_function_call()
        self.gen_add_code_line("frame_jacobian_inner<T>(s_frame_jacobian, target_jid, reference_frame, s_q, s_XmatsHom, d_robotModel, s_temp);")
        self.gen_anti_licm_output_write("frame_jacobian")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("frame_jacobian", str(6 * nv))
    self.gen_add_end_function()


def gen_frame_jacobian_host(self, mode=0):
    """Emit frame_jacobian host launcher (3 modes: 0=batch w/ mem, 1=single-call
    timing, 2=batch compute-only). target_jid / reference_frame are trailing
    host params with defaults (leaf-EE joint / LOCAL_WORLD_ALIGNED) so a caller
    that omits them gets the historical fixed-target behavior; both are forwarded
    to the kernel's runtime frame params."""
    single_call_timing, compute_only = host_mode_flags(mode)
    default_tjid = self.robot.get_leaf_nodes()[0]
    func_params = ["hd_data is the packaged input and output pointers",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)",
                   "streams are pointers to CUDA streams for async memory transfers (if needed)",
                   "target_jid is the joint id of the requested frame (default leaf-EE)",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED (default LWA)"]
    # target_jid / reference_frame are trailing defaulted params so existing
    # call sites (which omit them) keep the leaf-EE / LWA behavior.
    # Non-const so the body can resolve a -1 "use default" sentinel independently
    # per arg (a binding may request the default target but an explicit frame).
    frame_args = (", int target_jid = " + str(default_tjid) +
                  ", int reference_frame = " + str(_REF_LWA))
    func_def_start = "void frame_jacobian(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,"
    func_def_end = "                            const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams" + frame_args + ") {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_end = "                            const dim3 block_dimms, const dim3 thread_dimms" + frame_args + ") {"
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end
    self.gen_add_func_doc("Compute a general-frame geometric Jacobian", [], func_params, None)
    # MUJOCO_OUTPUT (floating only) host flag, LAST: forwarded to the kernel launch
    # (naming the tier positionally to reach the trailing flag). Default false ->
    # byte-identical pin codegen.
    mjx_host = gen_host_wrapper_head(self, "frame_jacobian", func_def_start, func_def_end, kind_rule="kinematics")
    self.gen_add_code_line("if (target_jid < 0) { target_jid = " + str(default_tjid) + "; }       // -1 => leaf-EE default (frame still honored)")
    self.gen_add_code_line("if (reference_frame < 0) { reference_frame = " + str(_REF_LWA) + "; }  // -1 => LOCAL_WORLD_ALIGNED default")
    fj_kernel_tmpl = "frame_jacobian_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "frame_jacobian_kernel<T, RESOURCE_TIER>"
    func_call_start = (fj_kernel_tmpl + "<<<block_dimms,thread_dimms,FRAME_JACOBIAN_DYNAMIC_SHARED_MEM_BYTES<T>()>>>"
                       "(hd_data->d_frame_jacobian,hd_data->d_q,stride_q,target_jid,reference_frame,")
    func_call_end = "d_robotModel,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("frame_jacobian_kernel<", "frame_jacobian_kernel_single_timing<")
    if not compute_only:
        self.gen_add_code_lines(host_q_compressed_input_transfer_lines(single_call_timing))
    else:
        self.gen_add_code_line("int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;")
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    func_call_mem_adjust, func_call_mem_adjust2 = gen_launch_pair(func_call, "hd_data->d_q")
    func_call_code = [func_call_mem_adjust, func_call_mem_adjust2, "gpuErrchkKernel();"]
    if single_call_timing:
        wrap_host_single_call_timing(func_call_code)
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"frame_jacobian\", FRAME_JACOBIAN_DYNAMIC_SHARED_MEM_BYTES<T>()));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        gen_emit_host_result_transfer(self, "h_frame_jacobian", "d_frame_jacobian", "6*NUM_VEL*", single_call_timing)
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("frame_jacobian"))
    self.gen_add_end_function()


def gen_frame_jacobian(self):
    self.gen_frame_jacobian_inner()
    self.gen_frame_jacobian_device()
    self.gen_frame_jacobian_kernel(single_call_timing=False)
    self.gen_frame_jacobian_kernel(single_call_timing=True)
    self.gen_frame_jacobian_host(mode=0)
    self.gen_frame_jacobian_host(mode=1)
    self.gen_frame_jacobian_host(mode=2)


def gen_frame_jacobian_dot_device(self):
    """Emit frame_jacobian_dot_device: the ANALYTIC time derivative Jdot of the
    general-frame geometric Jacobian along v = qd (6 x NUM_VEL).

    Direct CUDA transcription of the analytic RBDReference oracle
    (`RBDReference.frame_jacobian_dot`). Differentiates the SAME world-axis
    Jacobian `frame_jacobian_inner` builds (NOT a finite difference). Column i
    owns joint j with world axis a_w = R_j S; the value and its time derivative
    (R_j-dot = w_j x R_j, world point velocities) are, in the LWA basis:

        Jv[:,i]  = l_w + a_w x (p_f - p_j)            Jw[:,i]  = a_w
        Jvd[:,i] = (w_j x a_w) x (p_f - p_j)
                   + a_w x (v_f - v_j) + w_j x l_w     Jwd[:,i] = w_j x a_w

    with l_w = R_j S_lin (0 for a revolute col, so the form is BRANCHLESS), and
    w_j / v_j the world angular / origin-linear velocities of joint j's frame from
    a forward velocity sweep (rigid transport from the parent + each joint's own
    S contribution), consistent with frame_jacobian_inner's own assembly. The
    reference-frame derivative (WORLD / LOCAL) folds in the frame-velocity terms.
    Correctness-first single-block (serial assembly, mirrors frame_jacobian_inner).
    """
    NJ = self.robot.get_num_joints()
    nv = self.robot.get_num_vel()
    HAS_MIMIC = self.robot_has_mimic_joints()

    func_def = ("void frame_jacobian_dot_device(T *s_Jdot, const int target_jid, "
                "const int reference_frame, const T *s_q, const T *s_qd, "
                "const robotModel<T> *d_robotModel) {")
    func_params = ["s_Jdot holds the 6 x NUM_VEL Jacobian time derivative (column-major, [linear; angular])",
                   "target_jid is the joint id of the frame",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED",
                   "s_q is the joint position vector",
                   "s_qd is the joint velocity vector v (Pinocchio order [v_lin; omega; joints] for floating base)",
                   "d_robotModel is the GPU model helpers"]
    func_notes = ["Analytic Jdot (matches pinocchio getFrameJacobianTimeVariation); no finite difference."]
    self.gen_add_func_doc("Compute the time derivative of a general-frame geometric Jacobian",
                          func_notes, func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    # Arena: s_Xworld (16*NJ, chain-up) + world velocity scratch (w,v per joint) +
    # the LWA value Jacobian (needed by the WORLD/LOCAL derivative fold). s_Jdot is
    # the caller-provided output.
    extra = [("s_wvel", 3 * NJ), ("s_vvel", 3 * NJ), ("s_Jval", 6 * nv)]
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _frame_jacobian_inner_temp_mem_size(self), extra_t_buffers=extra)

    # ---- Step 1: build local homogeneous transforms for s_q, then world chain-up ----
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_add_code_line("const T *s_Xhom = s_XmatsHom;")
    _emit_world_transform_chainup(self)

    # ---- Step 2: forward world-velocity sweep -> s_wvel[3*j], s_vvel[3*j] ----
    self.gen_add_code_line("// Step 2: world angular (w) / origin-linear (v) velocity per joint frame")
    self.gen_add_parallel_loop("i", str(3 * NJ))
    self.gen_add_code_line("s_wvel[i] = static_cast<T>(0); s_vvel[i] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Bake the sweep (BFS order, one entry per joint S-column; parent-before-child so
    # the serial recursion reads final parent velocities). first==1 on a joint's first
    # column triggers the rigid transport from its parent.
    s_jid, s_par, s_vi, s_first, s_ang, s_lin, s_alpha = [], [], [], [], [], [], []
    for level in range(self.robot.get_max_bfs_level() + 1):
        for jid in self.robot.get_ids_by_bfs_level(level):
            par = self.robot.get_parent_id(jid)
            S = np.asarray(self.robot.get_S_by_id(jid), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            vinds = self.robot.get_joint_index_v(jid)
            vinds = list(vinds) if isinstance(vinds, (list, tuple, np.ndarray)) else [vinds]
            alpha = self._alpha_for_jid(jid) if HAS_MIMIC else 1.0
            for c in range(S.shape[1]):
                s_jid.append(jid); s_par.append(par)
                s_vi.append(int(vinds[c] if c < len(vinds) else vinds[-1]))
                s_first.append(1 if c == 0 else 0)
                s_ang += [float(S[0, c]), float(S[1, c]), float(S[2, c])]
                s_lin += [float(S[3, c]), float(S[4, c]), float(S[5, c])]
                s_alpha.append(float(alpha))
    nsweep = len(s_jid)
    if nsweep > 0:
        self.gen_bake_const_array("fjd_jid", s_jid, "int")
        self.gen_bake_const_array("fjd_par", s_par, "int")
        self.gen_bake_const_array("fjd_vi", s_vi, "int")
        self.gen_bake_const_array("fjd_first", s_first, "int")
        self.gen_bake_const_array("fjd_ang", s_ang, "T")
        self.gen_bake_const_array("fjd_lin", s_lin, "T")
        if HAS_MIMIC:
            self.gen_bake_const_array("fjd_alpha", s_alpha, "T")
        self.gen_add_serial_ops()
        self.gen_add_code_line("for (int t = 0; t < " + str(nsweep) + "; ++t) {", True)
        self.gen_add_code_line("int jid = fjd_jid[t]; int par = fjd_par[t]; int vi = fjd_vi[t];")
        self.gen_add_code_line("const T *Xj = &s_Xworld[16*jid];")
        self.gen_add_code_line("T *wj = &s_wvel[3*jid]; T *vj = &s_vvel[3*jid];")
        # first column of the joint: rigid transport of the parent's motion to p_jid
        self.gen_add_code_line("if (fjd_first[t] && par >= 0) {", True)
        self.gen_add_code_line("const T *Xp = &s_Xworld[16*par]; const T *wp = &s_wvel[3*par]; const T *vp = &s_vvel[3*par];")
        self.gen_add_code_line("T dx = Xj[12]-Xp[12], dy = Xj[13]-Xp[13], dz = Xj[14]-Xp[14];")
        self.gen_add_code_line("wj[0]=wp[0]; wj[1]=wp[1]; wj[2]=wp[2];")
        self.gen_add_code_line("vj[0]=vp[0]+(wp[1]*dz-wp[2]*dy); vj[1]=vp[1]+(wp[2]*dx-wp[0]*dz); vj[2]=vp[2]+(wp[0]*dy-wp[1]*dx);")
        self.gen_add_end_control_flow()
        # this column's own contribution: w += sc*(R_j S_ang), v += sc*(R_j S_lin)
        if 0.0 in s_alpha:
            # Retain the parent rigid transport above, but never read the
            # constant body's placeholder velocity slot (possibly nonexistent).
            self.gen_add_code_line("if (fjd_alpha[t] == static_cast<T>(0)) continue;")
        self.gen_add_code_line("T a0=fjd_ang[3*t],a1=fjd_ang[3*t+1],a2=fjd_ang[3*t+2];")
        self.gen_add_code_line("T l0=fjd_lin[3*t],l1=fjd_lin[3*t+1],l2=fjd_lin[3*t+2];")
        self.gen_add_code_line("T aw0=Xj[0]*a0+Xj[4]*a1+Xj[8]*a2, aw1=Xj[1]*a0+Xj[5]*a1+Xj[9]*a2, aw2=Xj[2]*a0+Xj[6]*a1+Xj[10]*a2;")
        self.gen_add_code_line("T lw0=Xj[0]*l0+Xj[4]*l1+Xj[8]*l2, lw1=Xj[1]*l0+Xj[5]*l1+Xj[9]*l2, lw2=Xj[2]*l0+Xj[6]*l1+Xj[10]*l2;")
        self.gen_add_code_line("T sc = " + ("fjd_alpha[t]*" if HAS_MIMIC else "") + "s_qd[vi];")
        self.gen_add_code_line("wj[0]+=sc*aw0; wj[1]+=sc*aw1; wj[2]+=sc*aw2;")
        self.gen_add_code_line("vj[0]+=sc*lw0; vj[1]+=sc*lw1; vj[2]+=sc*lw2;")
        self.gen_add_end_control_flow()  # for t
        self.gen_add_end_control_flow()  # serial
        self.gen_add_sync()

    # ---- Step 3: per-column value (LWA) + its time derivative ----
    self.gen_add_code_line("glass::set_const<T, " + str(6 * nv) + ">(static_cast<T>(0), s_Jval);")
    self.gen_add_code_line("glass::set_const<T, " + str(6 * nv) + ">(static_cast<T>(0), s_Jdot);")
    self.gen_add_sync()
    # Column jobs: (target, jj, vi, ang, lin, alpha) -- same chain jobs as the value inner.
    jobs = []
    for jid in range(NJ):
        chain = sorted(self.robot.get_ancestors_by_id(jid)) + [jid]
        for jj in chain:
            S = np.asarray(self.robot.get_S_by_id(jj), dtype=np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            vinds = self.robot.get_joint_index_v(jj)
            vinds = list(vinds) if isinstance(vinds, (list, tuple, np.ndarray)) else [vinds]
            alpha = self._alpha_for_jid(jj) if HAS_MIMIC else 1.0
            if alpha == 0.0:
                continue
            for c in range(S.shape[1]):
                vi = vinds[c] if c < len(vinds) else vinds[-1]
                jobs.append((jid, jj, int(vi),
                             [float(S[0, c]), float(S[1, c]), float(S[2, c])],
                             [float(S[3, c]), float(S[4, c]), float(S[5, c])], float(alpha)))
    njobs = len(jobs)
    if njobs > 0:
        self.gen_bake_const_array("fjc_target", [j[0] for j in jobs], "int")
        self.gen_bake_const_array("fjc_jj", [j[1] for j in jobs], "int")
        self.gen_bake_const_array("fjc_vi", [j[2] for j in jobs], "int")
        self.gen_bake_const_array("fjc_ang", [v for j in jobs for v in j[3]], "T")
        self.gen_bake_const_array("fjc_lin", [v for j in jobs for v in j[4]], "T")
        if HAS_MIMIC:
            self.gen_bake_const_array("fjc_alpha", [j[5] for j in jobs], "T")
        self.gen_add_serial_ops()
        self.gen_add_code_line("const T *Xf = &s_Xworld[16*target_jid];")
        self.gen_add_code_line("T pfx=Xf[12], pfy=Xf[13], pfz=Xf[14];")
        self.gen_add_code_line("const T *wf = &s_wvel[3*target_jid]; const T *vf = &s_vvel[3*target_jid];")
        self.gen_add_code_line("for (int t = 0; t < " + str(njobs) + "; ++t) {", True)
        self.gen_add_code_line("if (fjc_target[t] != target_jid) continue;")
        self.gen_add_code_line("int jj = fjc_jj[t]; int vi = fjc_vi[t];")
        self.gen_add_code_line("const T *Xj = &s_Xworld[16*jj]; const T *wj = &s_wvel[3*jj]; const T *vj = &s_vvel[3*jj];")
        self.gen_add_code_line("T a0=fjc_ang[3*t],a1=fjc_ang[3*t+1],a2=fjc_ang[3*t+2];")
        self.gen_add_code_line("T l0=fjc_lin[3*t],l1=fjc_lin[3*t+1],l2=fjc_lin[3*t+2];")
        self.gen_add_code_line("T aw0=Xj[0]*a0+Xj[4]*a1+Xj[8]*a2, aw1=Xj[1]*a0+Xj[5]*a1+Xj[9]*a2, aw2=Xj[2]*a0+Xj[6]*a1+Xj[10]*a2;")
        self.gen_add_code_line("T lw0=Xj[0]*l0+Xj[4]*l1+Xj[8]*l2, lw1=Xj[1]*l0+Xj[5]*l1+Xj[9]*l2, lw2=Xj[2]*l0+Xj[6]*l1+Xj[10]*l2;")
        self.gen_add_code_line("T dx=pfx-Xj[12], dy=pfy-Xj[13], dz=pfz-Xj[14];")
        # value column (LWA): Jv = lw + aw x d ; Jw = aw
        self.gen_add_code_line("T jv0 = lw0 + (aw1*dz-aw2*dy), jv1 = lw1 + (aw2*dx-aw0*dz), jv2 = lw2 + (aw0*dy-aw1*dx);")
        # derivative: awd = w_j x aw
        self.gen_add_code_line("T awd0 = wj[1]*aw2-wj[2]*aw1, awd1 = wj[2]*aw0-wj[0]*aw2, awd2 = wj[0]*aw1-wj[1]*aw0;")
        self.gen_add_code_line("T dvx=vf[0]-vj[0], dvy=vf[1]-vj[1], dvz=vf[2]-vj[2];")
        # Jvd = awd x d + aw x (v_f - v_j) + w_j x lw
        self.gen_add_code_line("T jvd0 = (awd1*dz-awd2*dy) + (aw1*dvz-aw2*dvy) + (wj[1]*lw2-wj[2]*lw1);")
        self.gen_add_code_line("T jvd1 = (awd2*dx-awd0*dz) + (aw2*dvx-aw0*dvz) + (wj[2]*lw0-wj[0]*lw2);")
        self.gen_add_code_line("T jvd2 = (awd0*dy-awd1*dx) + (aw0*dvy-aw1*dvx) + (wj[0]*lw1-wj[1]*lw0);")
        self.gen_add_code_line("T *Vc = &s_Jval[6*vi]; T *Dc = &s_Jdot[6*vi];")
        if HAS_MIMIC:
            self.gen_add_code_line("T al = fjc_alpha[t];")
            self.gen_add_code_line("Vc[0]+=al*jv0; Vc[1]+=al*jv1; Vc[2]+=al*jv2; Vc[3]+=al*aw0; Vc[4]+=al*aw1; Vc[5]+=al*aw2;")
            self.gen_add_code_line("Dc[0]+=al*jvd0; Dc[1]+=al*jvd1; Dc[2]+=al*jvd2; Dc[3]+=al*awd0; Dc[4]+=al*awd1; Dc[5]+=al*awd2;")
        else:
            self.gen_add_code_line("Vc[0]+=jv0; Vc[1]+=jv1; Vc[2]+=jv2; Vc[3]+=aw0; Vc[4]+=aw1; Vc[5]+=aw2;")
            self.gen_add_code_line("Dc[0]+=jvd0; Dc[1]+=jvd1; Dc[2]+=jvd2; Dc[3]+=awd0; Dc[4]+=awd1; Dc[5]+=awd2;")
        self.gen_add_end_control_flow()  # for t
        self.gen_add_end_control_flow()  # serial
        self.gen_add_sync()

    # ---- Step 4: reference-frame derivative transform (in place per column) ----
    # s_Jdot currently holds the LWA derivative; s_Jval the LWA value. WORLD/LOCAL
    # fold in the frame-velocity terms (d/dt of the value inner's Step-4 transform).
    self.gen_add_code_line("// Step 4: reference-frame transform of the derivative")
    self.gen_add_serial_ops()
    self.gen_add_code_line("const T *Xf2 = &s_Xworld[16*target_jid];")
    self.gen_add_code_line("T Rf[9]; for (int c=0;c<3;++c) for (int r=0;r<3;++r) Rf[r+3*c] = Xf2[r + 4*c];")
    self.gen_add_code_line("T pfx=Xf2[12], pfy=Xf2[13], pfz=Xf2[14];")
    self.gen_add_code_line("const T *wf = &s_wvel[3*target_jid]; const T *vf = &s_vvel[3*target_jid];")
    self.gen_add_code_line("for (int vi = 0; vi < " + str(nv) + "; ++vi) {", True)
    self.gen_add_code_line("T *Dc = &s_Jdot[6*vi]; T *Vc = &s_Jval[6*vi];")
    self.gen_add_code_line("T dv0=Dc[0],dv1=Dc[1],dv2=Dc[2], dw0=Dc[3],dw1=Dc[4],dw2=Dc[5];")
    self.gen_add_code_line("T v0=Vc[0],v1=Vc[1],v2=Vc[2], w0=Vc[3],w1=Vc[4],w2=Vc[5];")
    self.gen_add_code_lines([
        "if (reference_frame == " + str(_REF_WORLD) + ") {",
        # d/dt[ Jv + p_f x Jw ] = Jvd + v_f x Jw + p_f x Jwd ; angular unchanged.
        "  Dc[0] = dv0 + (vf[1]*w2 - vf[2]*w1) + (pfy*dw2 - pfz*dw1);",
        "  Dc[1] = dv1 + (vf[2]*w0 - vf[0]*w2) + (pfz*dw0 - pfx*dw2);",
        "  Dc[2] = dv2 + (vf[0]*w1 - vf[1]*w0) + (pfx*dw1 - pfy*dw0);",
        "} else if (reference_frame == " + str(_REF_LOCAL) + ") {",
        # d/dt[ R_f^T J ] = R_f^T ( Jd - w_f x J ) for each 3-block.
        "  T uv0 = dv0 - (wf[1]*v2 - wf[2]*v1), uv1 = dv1 - (wf[2]*v0 - wf[0]*v2), uv2 = dv2 - (wf[0]*v1 - wf[1]*v0);",
        "  T uw0 = dw0 - (wf[1]*w2 - wf[2]*w1), uw1 = dw1 - (wf[2]*w0 - wf[0]*w2), uw2 = dw2 - (wf[0]*w1 - wf[1]*w0);",
        "  Dc[0] = Rf[0]*uv0 + Rf[1]*uv1 + Rf[2]*uv2;",
        "  Dc[1] = Rf[3]*uv0 + Rf[4]*uv1 + Rf[5]*uv2;",
        "  Dc[2] = Rf[6]*uv0 + Rf[7]*uv1 + Rf[8]*uv2;",
        "  Dc[3] = Rf[0]*uw0 + Rf[1]*uw1 + Rf[2]*uw2;",
        "  Dc[4] = Rf[3]*uw0 + Rf[4]*uw1 + Rf[5]*uw2;",
        "  Dc[5] = Rf[6]*uw0 + Rf[7]*uw1 + Rf[8]*uw2;",
        "}",
    ])
    self.gen_add_end_control_flow()  # for vi
    self.gen_add_end_control_flow()  # serial
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_frame_jacobian_dot_kernel(self, single_call_timing=False):
    """Emit frame_jacobian_dot_kernel: batched launcher of the general-frame
    Jacobian time-derivative Jdot. Reads q (NUM_POS) + qd (NUM_VEL) from the
    packed d_q_qd_u buffer; target_jid + reference_frame are RUNTIME kernel
    params (the inner already supports them), defaulted by the host to leaf-EE /
    LOCAL_WORLD_ALIGNED (mirrors frame_jacobian_kernel). Output 6 x NV."""
    n_vel = self.robot.get_num_vel()
    n_pos = self.robot.get_num_pos()
    fb = 1 if self.robot.floating_base else 0
    func_params = ["d_frame_jacobian_dot is the vector of 6 x NUM_VEL Jacobian time-derivatives (column-major, [linear; angular])",
                   "d_q_qd is the packed [q (NUM_POS); qd (NUM_VEL)] input",
                   "stride_q_qd is the stride between each (q, qd) tuple",
                   "target_jid is the joint id whose frame Jacobian time-derivative is requested (runtime)",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED (runtime)",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_def_start = ("void frame_jacobian_dot_kernel(T *d_frame_jacobian_dot, const T *d_q_qd, const int stride_q_qd, "
                      "const int target_jid, const int reference_frame, ")
    func_def_end = "const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("(", "_single_timing(")
    self.gen_add_func_doc("Compute the time derivative of a general-frame geometric Jacobian", [], func_params, None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag, LAST
    # after RESOURCE_TIER so existing positional <T,TIER> call sites are unaffected;
    # default false if-constexpr-elides the epilogue -> byte-identical PTX.
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # frame_jacobian_dot_device is a full auto-smem wrapper: it owns the ENTIRE
    # dynamic-smem arena (s_XmatsHom + s_qpert + s_Jp + s_Jm + s_temp), sized by
    # FRAME_JACOBIAN_DOT_DYNAMIC_SHARED_MEM_BYTES, and reads s_q/s_qd + writes
    # s_Jdot as caller params. So the kernel keeps its input/output buffers in
    # STATIC __shared__ (NOT the dynamic arena) to avoid aliasing the wrapper's
    # internal scratch (mirrors the cuda_frame_jacobian_smoke_runner static-shared
    # pattern). The dynamic smem the launch reserves is consumed by the wrapper.
    self.gen_add_code_line("__shared__ T s_q_qd[" + str(n_pos + n_vel) + "];")
    self.gen_add_code_line("__shared__ T s_frame_jacobian_dot[" + str(6 * n_vel) + "];")
    self.gen_add_code_line("T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(n_pos) + "];")
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q_qd", str(n_pos + n_vel), stride="stride_q_qd")
        # mjx input convert (q quat reorder + qd base-linear -> pin; Jdot depends on
        # qd, so use the full input_convert before the device rebuilds transforms).
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert("s_q", "s_qd")
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute")
        self.gen_add_code_line("frame_jacobian_dot_device<T>(s_frame_jacobian_dot, target_jid, reference_frame, s_q, s_qd, d_robotModel);")
        self.gen_add_sync()
        # mjx output: Jdot column reframe Jdot G^{-1} (base-linear cols . R^T)
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_column_reframe("s_frame_jacobian_dot", 6, n_vel)
            self.gen_add_end_control_flow()
        self.gen_kernel_save_result("frame_jacobian_dot", str(6 * n_vel), stride=str(6 * n_vel))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q_qd", str(n_pos + n_vel))
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q_qd", str(n_pos + n_vel), feedback_from="frame_jacobian_dot")
        self.gen_add_code_line("frame_jacobian_dot_device<T>(s_frame_jacobian_dot, target_jid, reference_frame, s_q, s_qd, d_robotModel);")
        self.gen_anti_licm_output_write("frame_jacobian_dot")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("frame_jacobian_dot", str(6 * n_vel))
    self.gen_add_end_function()


def gen_frame_jacobian_dot_host(self, mode=0):
    """Emit frame_jacobian_dot host launcher (3 modes). Always uses the full
    q|qd|u memory (Jdot needs qd, which the compressed q-only layout lacks).
    target_jid / reference_frame are trailing defaulted host params (leaf-EE /
    LWA) forwarded to the kernel's runtime frame params."""
    single_call_timing, compute_only = host_mode_flags(mode)
    default_tjid = self.robot.get_leaf_nodes()[0]
    func_params = ["hd_data is the packaged input and output pointers",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)",
                   "streams are pointers to CUDA streams for async memory transfers (if needed)",
                   "target_jid is the joint id of the requested frame (default leaf-EE)",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED (default LWA)"]
    # Non-const so the body can resolve a -1 "use default" sentinel independently
    # per arg (a binding may request the default target but an explicit frame).
    frame_args = (", int target_jid = " + str(default_tjid) +
                  ", int reference_frame = " + str(_REF_LWA))
    func_def_start = "void frame_jacobian_dot(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,"
    func_def_end = "                            const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams" + frame_args + ") {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_end = "                            const dim3 block_dimms, const dim3 thread_dimms" + frame_args + ") {"
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end
    self.gen_add_func_doc("Compute the time derivative of a general-frame geometric Jacobian", [], func_params, None)
    # MUJOCO_OUTPUT (floating only) host flag, LAST: forwarded to the kernel launch
    # (naming the tier positionally to reach the trailing flag). Default false ->
    # byte-identical pin codegen.
    mjx_host = gen_host_wrapper_head(self, "frame_jacobian_dot", func_def_start, func_def_end, kind_rule="kinematics")
    self.gen_add_code_line("if (target_jid < 0) { target_jid = " + str(default_tjid) + "; }       // -1 => leaf-EE default (frame still honored)")
    self.gen_add_code_line("if (reference_frame < 0) { reference_frame = " + str(_REF_LWA) + "; }  // -1 => LOCAL_WORLD_ALIGNED default")
    # Jdot needs qd; always source from the full q|qd|u buffer (stride 3*NUM_JOINTS),
    # the kernel reads the leading [q; qd] slice.
    fjd_kernel_tmpl = "frame_jacobian_dot_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "frame_jacobian_dot_kernel<T, RESOURCE_TIER>"
    func_call = (fjd_kernel_tmpl + "<<<block_dimms,thread_dimms,FRAME_JACOBIAN_DOT_DYNAMIC_SHARED_MEM_BYTES<T>()>>>"
                 "(hd_data->d_frame_jacobian_dot,hd_data->d_q_qd_u,stride_q_qd,target_jid,reference_frame,d_robotModel,num_timesteps);")
    if single_call_timing:
        func_call = func_call.replace("frame_jacobian_dot_kernel<", "frame_jacobian_dot_kernel_single_timing<")
    if not compute_only:
        self.gen_add_code_lines(["// start code with memory transfer",
                                 "int stride_q_qd = 3*NUM_JOINTS;",
                                 "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*" +
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));",
                                 "gpuErrchkKernel();"])
    else:
        self.gen_add_code_line("int stride_q_qd = 3*NUM_JOINTS;")
    self.gen_add_code_line("// then call the kernel")
    func_call_code = [func_call, "gpuErrchkKernel();"]
    if single_call_timing:
        wrap_host_single_call_timing(func_call_code)
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"frame_jacobian_dot\", FRAME_JACOBIAN_DOT_DYNAMIC_SHARED_MEM_BYTES<T>()));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        gen_emit_host_result_transfer(self, "h_frame_jacobian_dot", "d_frame_jacobian_dot", "6*NUM_VEL*", single_call_timing)
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("frame_jacobian_dot"))
    self.gen_add_end_function()


def gen_frame_jacobian_dot(self):
    self.gen_frame_jacobian_dot_device()
    self.gen_frame_jacobian_dot_kernel(single_call_timing=False)
    self.gen_frame_jacobian_dot_kernel(single_call_timing=True)
    self.gen_frame_jacobian_dot_host(mode=0)
    self.gen_frame_jacobian_dot_host(mode=1)
    self.gen_frame_jacobian_dot_host(mode=2)


# ─── frame velocity / acceleration (item C of 0.8, 2026-11-03) ─────────────────
# frame_velocity   = J(q) qd                      (6, [linear; angular], chosen frame)
# frame_acceleration = J(q) qdd + Jdot(q, qd) qd  (the time derivative of that twist:
#   the spatial acceleration in LOCAL / WORLD, the classical acceleration of the
#   frame origin in LOCAL_WORLD_ALIGNED; RBDReference.frame_acceleration, checked
#   against pinocchio's getFrameAcceleration / getFrameClassicalAcceleration).
# Both reuse the frame_jacobian family's inners; the contraction is one parallel
# loop over the six rows.

def _gen_frame_contract_rows(self, out_name, mats_and_vecs, nv):
    """s_out[r] = sum_c A[r + 6c] x[c] (+ B[r + 6c] y[c]) for r < 6, block-parallel."""
    self.gen_add_parallel_loop("r", "6")
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int c = 0; c < " + str(nv) + "; ++c) {", True)
    for mat, vec in mats_and_vecs:
        self.gen_add_code_line("acc += " + mat + "[r + 6*c] * " + vec + "[c];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line(out_name + "[r] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def gen_frame_velocity_device(self):
    """frame_velocity_device: the twist J(q) qd of a frame in the chosen reference
    frame (6 entries, [linear; angular]). Owns the dynamic arena like
    frame_jacobian_device plus a 6 x NUM_VEL J buffer."""
    nv = self.robot.get_num_vel()
    func_def = ("void frame_velocity_device(T *s_v, const int target_jid, const int reference_frame, "
                "const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel) {")
    func_params = ["s_v holds the 6-entry frame twist ([linear; angular]) in the chosen reference frame",
                   "target_jid is the joint id of the frame",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED",
                   "s_q is the joint position vector",
                   "s_qd is the joint velocity vector v (Pinocchio order [v_lin; omega; joints] for floating base)",
                   "d_robotModel is the GPU model helpers"]
    self.gen_add_func_doc("Compute a frame's velocity (twist) in a reference frame",
                          ["Equals pinocchio getFrameVelocity for the three reference frames."], func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _frame_jacobian_inner_temp_mem_size(self), extra_t_buffers=[("s_Jv", 6 * nv)])
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_add_code_line("frame_jacobian_inner<T>(s_Jv, target_jid, reference_frame, s_q, s_XmatsHom, d_robotModel, s_temp);")
    self.gen_add_sync()
    _gen_frame_contract_rows(self, "s_v", [("s_Jv", "s_qd")], nv)
    self.gen_add_end_function()


def gen_frame_acceleration_device(self):
    """frame_acceleration_device: J(q) qdd + Jdot(q, qd) qd, the time derivative of
    the frame twist in the chosen reference frame (6 entries). Jdot comes from
    frame_jacobian_dot_device (which owns the dynamic arena), J from
    frame_jacobian_device afterwards; both land in static shared buffers."""
    nv = self.robot.get_num_vel()
    func_def = ("void frame_acceleration_device(T *s_a, const int target_jid, const int reference_frame, "
                "const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel) {")
    func_params = ["s_a holds the 6-entry frame acceleration ([linear; angular]) in the chosen reference frame",
                   "target_jid is the joint id of the frame",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED",
                   "s_q is the joint position vector",
                   "s_qd is the joint velocity vector v",
                   "s_qdd is the joint acceleration vector",
                   "d_robotModel is the GPU model helpers"]
    self.gen_add_func_doc("Compute a frame's acceleration (the time derivative of its twist) in a reference frame",
                          ["Equals pinocchio getFrameAcceleration (LOCAL, WORLD) and getFrameClassicalAcceleration (LOCAL_WORLD_ALIGNED)."],
                          func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("__shared__ T s_Jd_fa[" + str(6 * nv) + "];")
    self.gen_add_code_line("__shared__ T s_Jv_fa[" + str(6 * nv) + "];")
    self.gen_add_code_line("frame_jacobian_dot_device<T>(s_Jd_fa, target_jid, reference_frame, s_q, s_qd, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("frame_jacobian_device<T>(s_Jv_fa, target_jid, reference_frame, s_q, d_robotModel);")
    self.gen_add_sync()
    _gen_frame_contract_rows(self, "s_a", [("s_Jv_fa", "s_qdd"), ("s_Jd_fa", "s_qd")], nv)
    self.gen_add_end_function()


def _gen_frame_kin_kernel(self, name, single_call_timing, with_qdd, out_count=6, grad_blocks=0):
    """Shared kernel emitter for frame_velocity / frame_acceleration: batched over
    the packed [q; qd; u] buffer (stride 3*NUM_POS), runtime target_jid and
    reference_frame, static shared inputs and output (the device wrapper owns the
    dynamic arena, mirroring frame_jacobian_dot_kernel)."""
    n_pos = self.robot.get_num_pos()
    n_vel = self.robot.get_num_vel()
    devfn = name + "_device"
    func_params = ["d_" + name + " is the vector of 6-entry outputs ([linear; angular])",
                   "d_q_qd is the packed [q (NUM_POS); qd; (qdd)] input, three NUM_POS-wide slots per timestep",
                   "stride_q_qd is the stride between timesteps",
                   "target_jid is the joint id of the frame (runtime)",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED (runtime)",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_def = ("void " + name + "_kernel(T *d_" + name + ", const T *d_q_qd, const int stride_q_qd, "
                "const int target_jid, const int reference_frame, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {")
    if single_call_timing:
        func_def = func_def.replace("(", "_single_timing(")
    self.gen_add_func_doc("Batched " + name.replace("_", " "), [], func_params, None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    load_count = 3 * n_pos if with_qdd else n_pos + n_vel
    self.gen_add_code_line("__shared__ T s_q_qd[" + str(load_count) + "];")
    self.gen_add_code_line("__shared__ T s_" + name + "[" + str(out_count) + "];")
    self.gen_add_code_line("T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(n_pos) + "];"
                           + (" T *s_qdd = &s_q_qd[" + str(2 * n_pos) + "];" if with_qdd else ""))
    call = (devfn + "<T>(s_" + name + ", target_jid, reference_frame, s_q, s_qd, "
            + ("s_qdd, " if with_qdd else "") + "d_robotModel);")
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q_qd", str(load_count), stride="stride_q_qd")
        if self.robot.floating_base:
            # mjx input convert: quaternion order, base-linear velocity and (for
            # the acceleration) the base-linear acceleration, which is not a plain
            # rotation (a_pin = R^T a_mjx - omega x v_local). The outputs are frame
            # quantities (LOCAL / WORLD / LWA twists), so no output reframe.
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert("s_q", "s_qd", qdd_name=("s_qdd" if with_qdd else None))
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute")
        self.gen_add_code_line(call)
        self.gen_add_sync()
        if self.robot.floating_base and grad_blocks:
            # derivative columns against the mjx tangent: base-linear columns of
            # every 6 x NV block are reframed by R^T (the same column reframe as
            # the frame Jacobian twin; the rows are frame quantities).
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            for b in range(grad_blocks):
                self.gen_mjx_column_reframe("(s_" + name + " + " + str(6 * n_vel * b) + ")", 6, n_vel)
            self.gen_add_end_control_flow()
        self.gen_kernel_save_result(name, str(out_count), stride=str(out_count))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q_qd", str(load_count))
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q_qd", str(load_count), feedback_from=name)
        self.gen_add_code_line(call)
        self.gen_anti_licm_output_write(name)
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result(name, str(out_count))
    self.gen_add_end_function()


def _gen_frame_kin_host(self, name, mode, bytes_macro, out_expr="6*"):
    """Shared host emitter (3 modes) for frame_velocity / frame_acceleration; always
    the full q|qd|u memory, target_jid / reference_frame trailing defaulted."""
    single_call_timing, compute_only = host_mode_flags(mode)
    default_tjid = self.robot.get_leaf_nodes()[0]
    func_params = ["hd_data is the packaged input and output pointers",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)",
                   "streams are pointers to CUDA streams for async memory transfers (if needed)",
                   "target_jid is the joint id of the requested frame (default leaf-EE)",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED (default LWA)"]
    frame_args = (", int target_jid = " + str(default_tjid) + ", int reference_frame = " + str(_REF_LWA))
    func_def_start = "void " + name + "(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,"
    func_def_end = "                            const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams" + frame_args + ") {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_end = "                            const dim3 block_dimms, const dim3 thread_dimms" + frame_args + ") {"
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end
    self.gen_add_func_doc("Batched " + name.replace("_", " "), [], func_params, None)
    mjx_host = gen_host_wrapper_head(self, name, func_def_start, func_def_end, kind_rule="kinematics")
    self.gen_add_code_line("if (target_jid < 0) { target_jid = " + str(default_tjid) + "; }       // -1 => leaf-EE default (frame still honored)")
    self.gen_add_code_line("if (reference_frame < 0) { reference_frame = " + str(_REF_LWA) + "; }  // -1 => LOCAL_WORLD_ALIGNED default")
    tmpl = name + "_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else name + "_kernel<T, RESOURCE_TIER>"
    func_call = (tmpl + "<<<block_dimms,thread_dimms," + bytes_macro + "<T>()>>>"
                 "(hd_data->d_" + name + ",hd_data->d_q_qd_u,stride_q_qd,target_jid,reference_frame,d_robotModel,num_timesteps);")
    if single_call_timing:
        func_call = func_call.replace(name + "_kernel<", name + "_kernel_single_timing<")
    if not compute_only:
        self.gen_add_code_lines(["// start code with memory transfer",
                                 "int stride_q_qd = 3*NUM_JOINTS;",
                                 "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*" +
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));",
                                 "gpuErrchkKernel();"])
    else:
        self.gen_add_code_line("int stride_q_qd = 3*NUM_JOINTS;")
    self.gen_add_code_line("// then call the kernel")
    func_call_code = [func_call, "gpuErrchkKernel();"]
    if single_call_timing:
        wrap_host_single_call_timing(func_call_code)
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"" + name + "\", " + bytes_macro + "<T>()));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        gen_emit_host_result_transfer(self, "h_" + name, "d_" + name, out_expr, single_call_timing)
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line(name))
    self.gen_add_end_function()


def gen_frame_velocity(self):
    self.gen_frame_velocity_device()
    _gen_frame_kin_kernel(self, "frame_velocity", False, False)
    _gen_frame_kin_kernel(self, "frame_velocity", True, False)
    for mode in (0, 1, 2):
        _gen_frame_kin_host(self, "frame_velocity", mode, "FRAME_VELOCITY_DYNAMIC_SHARED_MEM_BYTES")


def gen_frame_acceleration(self):
    self.gen_frame_acceleration_device()
    _gen_frame_kin_kernel(self, "frame_acceleration", False, True)
    _gen_frame_kin_kernel(self, "frame_acceleration", True, True)
    for mode in (0, 1, 2):
        _gen_frame_kin_host(self, "frame_acceleration", mode, "FRAME_ACCELERATION_DYNAMIC_SHARED_MEM_BYTES")


def _frame_kin_gradient_tables(self):
    """Baked tables of the frame kinematics gradient sweep: the bodies in BFS
    order (parent first) with their DoF-column range, and one row per DoF
    column (joint, velocity slot, S angular/linear parts, mimic alpha).
    Constant bodies (alpha 0) own no column and inherit their parent's motion."""
    HAS_MIMIC = self.robot_has_mimic_joints()
    b_jid, b_par, b_c0, b_cn = [], [], [], []
    c_jid, c_vi, c_ang, c_lin, c_alpha = [], [], [], [], []
    for level in range(self.robot.get_max_bfs_level() + 1):
        for jid in self.robot.get_ids_by_bfs_level(level):
            par = self.robot.get_parent_id(jid)
            alpha = self._alpha_for_jid(jid) if HAS_MIMIC else 1.0
            b_jid.append(int(jid)); b_par.append(int(par)); b_c0.append(len(c_jid))
            if alpha != 0.0:
                S = np.asarray(self.robot.get_S_by_id(jid), dtype=np.float64)
                if S.ndim == 1:
                    S = S.reshape(-1, 1)
                vinds = self.robot.get_joint_index_v(jid)
                vinds = list(vinds) if isinstance(vinds, (list, tuple, np.ndarray)) else [vinds]
                for c in range(S.shape[1]):
                    c_jid.append(int(jid))
                    c_vi.append(int(vinds[c] if c < len(vinds) else vinds[-1]))
                    c_ang += [float(S[0, c]), float(S[1, c]), float(S[2, c])]
                    c_lin += [float(S[3, c]), float(S[4, c]), float(S[5, c])]
                    c_alpha.append(float(alpha))
            b_cn.append(len(c_jid) - b_c0[-1])
    parent = [int(self.robot.get_parent_id(j)) for j in range(self.robot.get_num_joints())]
    return {"b_jid": b_jid, "b_par": b_par, "b_c0": b_c0, "b_cn": b_cn,
            "c_jid": c_jid, "c_vi": c_vi, "c_ang": c_ang, "c_lin": c_lin, "c_alpha": c_alpha,
            "parent": parent, "has_mimic": HAS_MIMIC}


def frame_kin_gradient_inner_temp_mem_size(self):
    """s_Xworld (16 NJ) + the world-origin columns (6 per DoF column) + the
    per-body twists and spatial accelerations (6 + 6 per joint)."""
    NJ = self.robot.get_num_joints()
    ncol = len(_frame_kin_gradient_tables(self)["c_jid"])
    return 16 * NJ + 6 * ncol + 12 * NJ


def gen_frame_kinematics_gradient_inner(self):
    """frame_kinematics_gradient_inner: dv/dq, J (= dv/dqd = da/dqdd), da/dq and
    da/dqd of a frame's twist v and acceleration a (as frame_velocity_device and
    frame_acceleration_device define them) in the runtime reference frame, each
    6 x NUM_VEL column-major, in the local tangent chart (the floating root and
    spherical joints). Transcribes RBDReference._frame_kinematics_derivatives:
    a serial sweep over the bodies builds the world-origin Plucker columns A of
    every DoF, the body twists ov and spatial accelerations oa; then one thread
    per velocity slot forms, for the slot's columns at or above the target
    body i (parent p of the column's joint j),

        dv_i/dq_j  = A x (ov_i - ov_p)
        da_i/dq_j  = A x (oa_i - oa_p) + (ov_p x A) x (ov_i - ov_p)
        da_i/dqd_j = ov_j x A + A x (ov_i - ov_p)

    and applies the reference-frame map (LOCAL adds the placement derivative
    -fXo(A x w); LOCAL_WORLD_ALIGNED shifts to the moving frame origin and adds
    the classical term). WITH_ACC=false skips the acceleration sweep and rows."""
    NJ = self.robot.get_num_joints()
    nv = self.robot.get_num_vel()
    tabs = _frame_kin_gradient_tables(self)
    HAS_MIMIC = tabs["has_mimic"]
    ncol = len(tabs["c_jid"])
    # helpers: 3-vector cross and the [linear; angular] motion cross product
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line("void frame_kin_cross3(const T *a, const T *b, T *o) {", True)
    self.gen_add_code_line("o[0] = a[1]*b[2] - a[2]*b[1]; o[1] = a[2]*b[0] - a[0]*b[2]; o[2] = a[0]*b[1] - a[1]*b[0];")
    self.gen_add_end_function()
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line("void frame_kin_mcross(const T *a, const T *b, T *o) {", True)
    self.gen_add_code_line("// o = a x b for [linear; angular] spatial motion vectors")
    self.gen_add_code_line("T t1[3], t2[3];")
    self.gen_add_code_line("frame_kin_cross3<T>(&a[3], &b[0], t1); frame_kin_cross3<T>(&a[0], &b[3], t2);")
    self.gen_add_code_line("o[0] = t1[0] + t2[0]; o[1] = t1[1] + t2[1]; o[2] = t1[2] + t2[2];")
    self.gen_add_code_line("frame_kin_cross3<T>(&a[3], &b[3], &o[3]);")
    self.gen_add_end_function()

    func_def = ("void frame_kinematics_gradient_inner(T *s_dvdq, T *s_J, T *s_dadq, T *s_dadqd, "
                "const int target_jid, const int reference_frame, const T *s_q, const T *s_qd, const T *s_qdd, "
                "const T *s_XmatsHom, const robotModel<T> *d_robotModel, T *s_temp) {")
    func_params = ["s_dvdq holds dv/dq (6 x NUM_VEL, column-major, [linear; angular] rows)",
                   "s_J holds the frame Jacobian in the reference frame (= dv/dqd = da/dqdd)",
                   "s_dadq holds da/dq (written only when WITH_ACC)",
                   "s_dadqd holds da/dqd (written only when WITH_ACC)",
                   "target_jid is the joint id of the frame",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED",
                   "s_q is the joint position vector",
                   "s_qd is the joint velocity vector v",
                   "s_qdd is the joint acceleration vector (read only when WITH_ACC)",
                   "s_XmatsHom are the local homogeneous transforms (updated for s_q)",
                   "d_robotModel is the GPU model helpers",
                   "s_temp is the scratch arena (frame_kin_gradient_inner_temp_mem_size)"]
    self.gen_add_func_doc("Derivatives of a frame's twist and acceleration against q, qd (and qdd)",
                          ["Columns follow the local tangent chart; matches pinocchio getFrameVelocityDerivatives / "
                           "getFrameAccelerationDerivatives in LOCAL and WORLD, and differentiates the classical "
                           "acceleration with its moving origin in LOCAL_WORLD_ALIGNED."], func_params, None)
    self.gen_add_code_line("template <typename T, bool WITH_ACC>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("T *s_Xworld = &s_temp[0];")
    self.gen_add_code_line("T *s_A = &s_temp[" + str(16 * NJ) + "];")
    self.gen_add_code_line("T *s_ov = &s_temp[" + str(16 * NJ + 6 * ncol) + "];")
    self.gen_add_code_line("T *s_oa = &s_temp[" + str(16 * NJ + 6 * ncol + 6 * NJ) + "];")
    self.gen_add_code_line("const T *s_Xhom = s_XmatsHom;")
    _emit_world_transform_chainup(self, declare_xworld=False)
    self.gen_bake_const_array("fkg_bjid", tabs["b_jid"], "int")
    self.gen_bake_const_array("fkg_bpar", tabs["b_par"], "int")
    self.gen_bake_const_array("fkg_bc0", tabs["b_c0"], "int")
    self.gen_bake_const_array("fkg_bcn", tabs["b_cn"], "int")
    self.gen_bake_const_array("fkg_cjid", tabs["c_jid"], "int")
    self.gen_bake_const_array("fkg_cvi", tabs["c_vi"], "int")
    self.gen_bake_const_array("fkg_cang", tabs["c_ang"], "T")
    self.gen_bake_const_array("fkg_clin", tabs["c_lin"], "T")
    if HAS_MIMIC:
        self.gen_bake_const_array("fkg_calpha", tabs["c_alpha"], "T")
    self.gen_bake_const_array("fkg_parent", tabs["parent"], "int")
    alpha_expr = "fkg_calpha[t]" if HAS_MIMIC else "static_cast<T>(1)"
    # ---- Step 2: serial world-origin Plucker sweep (bodies parent-first) ----
    self.gen_add_code_line("// Step 2: world-origin columns A per DoF, body twists ov and accelerations oa")
    self.gen_add_serial_ops()
    self.gen_add_code_line("for (int b = 0; b < " + str(NJ) + "; ++b) {", True)
    self.gen_add_code_line("int jid = fkg_bjid[b]; int par = fkg_bpar[b];")
    self.gen_add_code_line("T *ov = &s_ov[6*jid]; T *oa = &s_oa[6*jid];")
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) { ov[k] = (par >= 0) ? s_ov[6*par + k] : static_cast<T>(0); oa[k] = (par >= 0) ? s_oa[6*par + k] : static_cast<T>(0); }")
    self.gen_add_code_line("const T *Xj = &s_Xworld[16*jid];")
    self.gen_add_code_line("T pj[3] = {Xj[12], Xj[13], Xj[14]};")
    self.gen_add_code_line("for (int t = fkg_bc0[b]; t < fkg_bc0[b] + fkg_bcn[b]; ++t) {", True)
    self.gen_add_code_line("T a0=fkg_cang[3*t],a1=fkg_cang[3*t+1],a2=fkg_cang[3*t+2];")
    self.gen_add_code_line("T l0=fkg_clin[3*t],l1=fkg_clin[3*t+1],l2=fkg_clin[3*t+2];")
    self.gen_add_code_line("T *A = &s_A[6*t];")
    self.gen_add_code_line("A[3] = Xj[0]*a0+Xj[4]*a1+Xj[8]*a2; A[4] = Xj[1]*a0+Xj[5]*a1+Xj[9]*a2; A[5] = Xj[2]*a0+Xj[6]*a1+Xj[10]*a2;")
    self.gen_add_code_line("T lw[3]; lw[0] = Xj[0]*l0+Xj[4]*l1+Xj[8]*l2; lw[1] = Xj[1]*l0+Xj[5]*l1+Xj[9]*l2; lw[2] = Xj[2]*l0+Xj[6]*l1+Xj[10]*l2;")
    self.gen_add_code_line("T pa[3]; frame_kin_cross3<T>(pj, &A[3], pa);")
    self.gen_add_code_line("A[0] = pa[0] + lw[0]; A[1] = pa[1] + lw[1]; A[2] = pa[2] + lw[2];")
    self.gen_add_code_line("T sc = " + alpha_expr + " * s_qd[fkg_cvi[t]];")
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) ov[k] += A[k] * sc;")
    self.gen_add_end_control_flow()  # columns (velocity)
    self.gen_add_code_line("if constexpr (WITH_ACC) {", True)
    self.gen_add_code_line("for (int t = fkg_bc0[b]; t < fkg_bc0[b] + fkg_bcn[b]; ++t) {", True)
    self.gen_add_code_line("const T *A = &s_A[6*t]; T mc[6]; frame_kin_mcross<T>(ov, A, mc);")
    self.gen_add_code_line("T sc = " + alpha_expr + " * s_qd[fkg_cvi[t]]; T sa = " + alpha_expr + " * s_qdd[fkg_cvi[t]];")
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) oa[k] += A[k] * sa + mc[k] * sc;")
    self.gen_add_end_control_flow()  # columns (acceleration)
    self.gen_add_end_control_flow()  # if constexpr
    self.gen_add_end_control_flow()  # bodies
    self.gen_add_end_control_flow()  # serial
    self.gen_add_sync()
    # ---- Step 3: one thread per velocity slot ----
    self.gen_add_code_line("// Step 3: per velocity slot, the columns at or above the target body, then the frame map")
    self.gen_add_code_line("const T *ovi = &s_ov[6*target_jid]; const T *oai = &s_oa[6*target_jid];")
    self.gen_add_code_line("const T *Xf = &s_Xworld[16*target_jid];")
    self.gen_add_parallel_loop("vi", str(nv))
    self.gen_add_code_line("T dvq[6], Jc[6], daq[6], dad[6], ovc[6], oac[6];")
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) { dvq[k] = Jc[k] = daq[k] = dad[k] = ovc[k] = oac[k] = static_cast<T>(0); }")
    self.gen_add_code_line("const T zero6[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};")
    self.gen_add_code_line("for (int t = 0; t < " + str(ncol) + "; ++t) {", True)
    self.gen_add_code_line("if (fkg_cvi[t] != vi) continue;")
    self.gen_add_code_line("int j = fkg_cjid[t];")
    self.gen_add_code_line("int anc = target_jid; while (anc != -1 && anc != j) anc = fkg_parent[anc];")
    self.gen_add_code_line("if (anc != j) continue;")
    self.gen_add_code_line("const T *A = &s_A[6*t]; int par = fkg_parent[j];")
    self.gen_add_code_line("const T *ovp = (par >= 0) ? &s_ov[6*par] : zero6; const T *oap = (par >= 0) ? &s_oa[6*par] : zero6;")
    self.gen_add_code_line("T al = " + alpha_expr + ";")
    self.gen_add_code_line("T d1[6], d2[6], tmp[6], tmp2[6];")
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) { d1[k] = ovi[k] - ovp[k]; d2[k] = oai[k] - oap[k]; }")
    self.gen_add_code_line("frame_kin_mcross<T>(A, d1, tmp);")
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) { dvq[k] += al * tmp[k]; Jc[k] += al * A[k]; }")
    self.gen_add_code_line("frame_kin_mcross<T>(A, ovi, tmp2);")
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) ovc[k] += al * tmp2[k];")
    self.gen_add_code_line("if constexpr (WITH_ACC) {", True)
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) dad[k] += al * tmp[k];")
    self.gen_add_code_line("frame_kin_mcross<T>(A, d2, tmp);")
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) daq[k] += al * tmp[k];")
    self.gen_add_code_line("frame_kin_mcross<T>(ovp, A, tmp2); frame_kin_mcross<T>(tmp2, d1, tmp);")
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) daq[k] += al * tmp[k];")
    self.gen_add_code_line("frame_kin_mcross<T>(&s_ov[6*j], A, tmp);")
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) dad[k] += al * tmp[k];")
    self.gen_add_code_line("frame_kin_mcross<T>(A, oai, tmp);")
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) oac[k] += al * tmp[k];")
    self.gen_add_end_control_flow()  # if constexpr
    self.gen_add_end_control_flow()  # for t
    # frame map
    self.gen_add_code_line("T *o_dvq = &s_dvdq[6*vi]; T *o_J = &s_J[6*vi];")
    self.gen_add_code_line("T pf[3] = {Xf[12], Xf[13], Xf[14]};")
    self.gen_add_code_line("if (reference_frame == " + str(_REF_WORLD) + ") {", True)
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) { o_dvq[k] = dvq[k]; o_J[k] = Jc[k]; }")
    self.gen_add_code_line("if constexpr (WITH_ACC) { T *o_daq = &s_dadq[6*vi]; T *o_dad = &s_dadqd[6*vi]; for (int k = 0; k < 6; ++k) { o_daq[k] = daq[k]; o_dad[k] = dad[k]; } }")
    self.gen_add_code_line("} else if (reference_frame == " + str(_REF_LOCAL) + ") {", True)
    # fXo m: lin = R^T (m_lin + m_ang x p_f); ang = R^T m_ang ; dq blocks take m - (A x w) first
    self.gen_add_code_line("auto fXo = [&](const T *m, T *o) { T c[3]; frame_kin_cross3<T>(&m[3], pf, c);")
    self.gen_add_code_line("  T u0 = m[0] + c[0], u1 = m[1] + c[1], u2 = m[2] + c[2];")
    self.gen_add_code_line("  o[0] = Xf[0]*u0 + Xf[1]*u1 + Xf[2]*u2; o[1] = Xf[4]*u0 + Xf[5]*u1 + Xf[6]*u2; o[2] = Xf[8]*u0 + Xf[9]*u1 + Xf[10]*u2;")
    self.gen_add_code_line("  o[3] = Xf[0]*m[3] + Xf[1]*m[4] + Xf[2]*m[5]; o[4] = Xf[4]*m[3] + Xf[5]*m[4] + Xf[6]*m[5]; o[5] = Xf[8]*m[3] + Xf[9]*m[4] + Xf[10]*m[5]; };")
    self.gen_add_code_line("T m[6]; for (int k = 0; k < 6; ++k) m[k] = dvq[k] - ovc[k];")
    self.gen_add_code_line("fXo(m, o_dvq); fXo(Jc, o_J);")
    self.gen_add_code_line("if constexpr (WITH_ACC) { for (int k = 0; k < 6; ++k) m[k] = daq[k] - oac[k]; fXo(m, &s_dadq[6*vi]); fXo(dad, &s_dadqd[6*vi]); }")
    self.gen_add_code_line("} else {", True)
    # LWA: X m: lin = m_lin + m_ang x p_f ; ang = m_ang ; then the moving-origin and classical terms
    self.gen_add_code_line("auto Xl = [&](const T *m, T *o) { T c[3]; frame_kin_cross3<T>(&m[3], pf, c);")
    self.gen_add_code_line("  o[0] = m[0] + c[0]; o[1] = m[1] + c[1]; o[2] = m[2] + c[2]; o[3] = m[3]; o[4] = m[4]; o[5] = m[5]; };")
    self.gen_add_code_line("T Jl[6], dv[6], c[3]; Xl(Jc, Jl); Xl(dvq, dv);")
    self.gen_add_code_line("const T *w = &ovi[3];")
    self.gen_add_code_line("frame_kin_cross3<T>(w, Jl, c); dv[0] += c[0]; dv[1] += c[1]; dv[2] += c[2];")
    self.gen_add_code_line("for (int k = 0; k < 6; ++k) { o_dvq[k] = dv[k]; o_J[k] = Jl[k]; }")
    self.gen_add_code_line("if constexpr (WITH_ACC) {", True)
    self.gen_add_code_line("T da[6], dav[6]; Xl(daq, da); Xl(dad, dav);")
    self.gen_add_code_line("const T *wd = &oai[3];")
    self.gen_add_code_line("frame_kin_cross3<T>(wd, Jl, c); da[0] += c[0]; da[1] += c[1]; da[2] += c[2];")
    self.gen_add_code_line("T vl[3]; frame_kin_cross3<T>(w, pf, vl); vl[0] += ovi[0]; vl[1] += ovi[1]; vl[2] += ovi[2];")
    # classical: a_lin += w x v_lin -> d adds (d ang) x v_lin + w x (d lin)
    self.gen_add_code_line("frame_kin_cross3<T>(&dv[3], vl, c); da[0] += c[0]; da[1] += c[1]; da[2] += c[2];")
    self.gen_add_code_line("frame_kin_cross3<T>(w, dv, c); da[0] += c[0]; da[1] += c[1]; da[2] += c[2];")
    self.gen_add_code_line("frame_kin_cross3<T>(&Jl[3], vl, c); dav[0] += c[0]; dav[1] += c[1]; dav[2] += c[2];")
    self.gen_add_code_line("frame_kin_cross3<T>(w, Jl, c); dav[0] += c[0]; dav[1] += c[1]; dav[2] += c[2];")
    self.gen_add_code_line("T *o_daq = &s_dadq[6*vi]; T *o_dad = &s_dadqd[6*vi]; for (int k = 0; k < 6; ++k) { o_daq[k] = da[k]; o_dad[k] = dav[k]; }")
    self.gen_add_end_control_flow()  # if constexpr
    self.gen_add_end_control_flow()  # else (LWA)
    self.gen_add_end_control_flow()  # for vi
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_frame_velocity_gradient_device(self):
    """frame_velocity_gradient_device: [dv/dq | dv/dqd] (6 x 2 NUM_VEL, column-major)
    of the frame twist in the chosen reference frame; owns the dynamic arena."""
    nv = self.robot.get_num_vel()
    func_def = ("void frame_velocity_gradient_device(T *s_dv, const int target_jid, const int reference_frame, "
                "const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel) {")
    func_params = ["s_dv holds [dv/dq | dv/dqd] (6 x 2*NUM_VEL, column-major, [linear; angular] rows)",
                   "target_jid is the joint id of the frame",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED",
                   "s_q is the joint position vector",
                   "s_qd is the joint velocity vector v",
                   "d_robotModel is the GPU model helpers"]
    self.gen_add_func_doc("Compute the derivatives of a frame's twist against q and qd",
                          ["Equals pinocchio getFrameVelocityDerivatives in LOCAL and WORLD; in LOCAL_WORLD_ALIGNED "
                           "the q-derivative also moves the frame origin (finite differences through the chart)."],
                          func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_XmatsHom_helpers_temp_shared_memory_code(frame_kin_gradient_inner_temp_mem_size(self))
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_add_code_line("frame_kinematics_gradient_inner<T, false>(&s_dv[0], &s_dv[" + str(6 * nv) + "], nullptr, nullptr, "
                           "target_jid, reference_frame, s_q, s_qd, nullptr, s_XmatsHom, d_robotModel, s_temp);")
    self.gen_add_end_function()


def gen_frame_acceleration_gradient_device(self):
    """frame_acceleration_gradient_device: [da/dq | da/dqd | da/dqdd] (6 x 3 NUM_VEL,
    column-major) of the frame acceleration in the chosen reference frame
    (classical in LOCAL_WORLD_ALIGNED); owns the dynamic arena."""
    nv = self.robot.get_num_vel()
    func_def = ("void frame_acceleration_gradient_device(T *s_da, const int target_jid, const int reference_frame, "
                "const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel) {")
    func_params = ["s_da holds [da/dq | da/dqd | da/dqdd] (6 x 3*NUM_VEL, column-major, [linear; angular] rows)",
                   "target_jid is the joint id of the frame",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED",
                   "s_q is the joint position vector",
                   "s_qd is the joint velocity vector v",
                   "s_qdd is the joint acceleration vector",
                   "d_robotModel is the GPU model helpers"]
    self.gen_add_func_doc("Compute the derivatives of a frame's acceleration against q, qd and qdd",
                          ["Equals pinocchio getFrameAccelerationDerivatives in LOCAL and WORLD; in LOCAL_WORLD_ALIGNED "
                           "it differentiates the classical acceleration with its moving origin."],
                          func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_XmatsHom_helpers_temp_shared_memory_code(frame_kin_gradient_inner_temp_mem_size(self),
                                                      extra_t_buffers=[("s_dvdq_fag", 6 * nv)])
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_add_code_line("frame_kinematics_gradient_inner<T, true>(s_dvdq_fag, &s_da[" + str(12 * nv) + "], &s_da[0], &s_da[" + str(6 * nv) + "], "
                           "target_jid, reference_frame, s_q, s_qd, s_qdd, s_XmatsHom, d_robotModel, s_temp);")
    self.gen_add_end_function()


def gen_frame_velocity_gradient(self):
    nv = self.robot.get_num_vel()
    self.gen_frame_velocity_gradient_device()
    _gen_frame_kin_kernel(self, "frame_velocity_gradient", False, False, out_count=12 * nv, grad_blocks=2)
    _gen_frame_kin_kernel(self, "frame_velocity_gradient", True, False, out_count=12 * nv, grad_blocks=2)
    for mode in (0, 1, 2):
        _gen_frame_kin_host(self, "frame_velocity_gradient", mode, "FRAME_VELOCITY_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
                            out_expr="12*NUM_VEL*")


def gen_frame_acceleration_gradient(self):
    nv = self.robot.get_num_vel()
    self.gen_frame_acceleration_gradient_device()
    _gen_frame_kin_kernel(self, "frame_acceleration_gradient", False, True, out_count=18 * nv, grad_blocks=3)
    _gen_frame_kin_kernel(self, "frame_acceleration_gradient", True, True, out_count=18 * nv, grad_blocks=3)
    for mode in (0, 1, 2):
        _gen_frame_kin_host(self, "frame_acceleration_gradient", mode, "FRAME_ACCELERATION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
                            out_expr="18*NUM_VEL*")


def gen_osc_inertia_device(self):
    """Emit osc_inertia_device: the 6x6 operational-space (task) inertia
    Lambda = (J Minv J^T)^{-1}.

    Direct CUDA transcription of the RBDReference numpy oracle
    (`RBDReference.osc_inertia`): compose the frame Jacobian J with the
    joint-space inverse mass matrix Minv, then invert the 6x6 task matrix via
    the block-cooperative GLASS Gauss-Jordan (`invert_matrix`).

    SELF-CONTAINED: Minv is composed ON DEVICE here via `minv_inner`
    (the kernel takes only q, no caller-provided Minv). The arena carries BOTH
    transform families — the spatial `s_XImats` (6x6) that minv consumes
    AND the homogeneous `s_XmatsHom` (4x4) that the Jacobian consumes — loaded
    by their respective `load_update_*` helpers (both already emitted whenever
    the frame_jacobian key is selected, since it pulls in {ee_pose, minv}).

    Plumbing:
      * minv_inner is templated <T, F_IN_SMEM>; we pass F_IN_SMEM=false
        and hand it a dedicated SHARED `s_F` buffer (6*NV*NV) as its
        `d_workspace` arg, so the heavy F-region lives in smem WITHOUT having to
        thread it through the tail of s_temp (avoids the no_F+F contiguity the
        F_IN_SMEM=true path assumes). s_temp is sized to minv's no_F
        region (>= the Jacobian inner's 16*NJ world-transform scratch).
      * minv emits SYMMETRIC_UPPER; we densify to a full symmetric
        s_Minv before the J*Minv*J^T contraction (floating-base output is
        already full-symmetric, but the densify read is symmetric-safe either
        way: read the upper-triangle source for every (r,c))."""
    nv = self.robot.get_num_vel()
    Xhom_size, _, _ = self.gen_get_Xhom_size()             # local homogeneous 4x4 transforms
    no_F_size = self.gen_minv_inner_no_F_size()
    F_size = self.gen_minv_inner_F_size()
    # s_temp serves BOTH the minv inner (no_F region) and the Jacobian inner
    # (16*NJ world-transform scratch); they run sequentially so size by the max.
    temp_size = max(no_F_size, _frame_jacobian_inner_temp_mem_size(self))

    func_def = ("void osc_inertia_device(T *s_Lambda, const int target_jid, "
                "const int reference_frame, const T *s_q, "
                "const robotModel<T> *d_robotModel, unsigned char *d_workspace = nullptr) {")
    func_params = ["s_Lambda holds the 6 x 6 operational-space inertia (column-major)",
                   "target_jid is the joint id of the frame",
                   "reference_frame is 0=LOCAL, 1=WORLD, 2=LOCAL_WORLD_ALIGNED",
                   "s_q is the joint position vector",
                   "d_robotModel is the GPU model helpers",
                   "d_workspace is the L2-pinned spill workspace (used when OSC_INERTIA_F_IN_SMEM<TIER>() is false; pass nullptr at the full tier)"]
    func_notes = ["Lambda = (J Minv J^T)^{-1}; self-contained — Minv is composed on device via minv_inner (no caller Minv)."]
    self.gen_add_func_doc("Compute the operational-space (task) inertia",
                          func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    # Arena: BOTH transform families + Minv + the minv F-region (passed as
    # minv's d_workspace) + the Jacobian J + the J*Minv*J^T compose scratch.
    # 2-rung spill: at the spill-F tier OSC_INERTIA_F_IN_SMEM<TIER>() is false, so
    # s_F drops from the smem arena (slot 0) and routes to the L2-pinned minv-F
    # workspace offset, shrinking the arena by 6*nv*nv (the dominant band). The macro
    # OSC_INERTIA_DYNAMIC_SHARED_MEM_BYTES<T,TIER> sizes the launch from the same predicate.
    self.gen_add_code_line("constexpr bool OSC_F_SMEM = OSC_INERTIA_F_IN_SMEM<RESOURCE_TIER>();")
    self.gen_add_code_line("constexpr int OSC_F_SLOT = OSC_F_SMEM ? " + str(F_size) + " : 0;")
    extra = [("s_XmatsHom", Xhom_size), ("s_Minv", nv * nv), ("s_F", "OSC_F_SLOT"),
             ("s_Jfj", 6 * nv), ("s_MJt", nv * 6), ("s_task", 36), ("s_taskinv", 36)]
    self.gen_XImats_helpers_temp_shared_memory_code(
        temp_size, extra_t_buffers=extra)
    # spill-F: route s_F to the L2-pinned minv-F workspace offset (d_workspace is the
    # per-timestep base, sliced by the kernel). Disjoint from any concurrent kernel —
    # osc_inertia is a standalone kinematics launch.
    self.gen_add_code_line("if constexpr (!OSC_F_SMEM) { " + gen_workspace_repoint_line("s_F", "GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()") + " } else { (void)d_workspace; }")

    # ---- Step 1: spatial transforms -> minv_inner -> SYMMETRIC_UPPER Minv ----
    self.gen_load_update_XImats_helpers_function_call()
    self.gen_add_sync()
    # F_IN_SMEM=false + s_F (shared) as d_workspace: keeps F in smem without the
    # no_F+F s_temp contiguity the smem-tail path requires.
    self.gen_minv_inner_function_call(
        updated_var_names=dict(d_workspace_name="s_F"), f_in_smem_expr="false")
    self.gen_add_sync()

    # ---- Step 2: homogeneous transforms -> frame_jacobian_inner -> J ----
    # (The SYMMETRIC_UPPER -> full densify is folded into the J*Minv*J^T
    # contraction below: the MJt loop reads the upper-triangle Minv entry.)
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_add_sync()
    self.gen_add_code_line("frame_jacobian_inner<T>(s_Jfj, target_jid, reference_frame, s_q, s_XmatsHom, d_robotModel, s_temp);")
    self.gen_add_sync()

    # MJt = Minv @ J^T  (nv x 6, column-major: MJt[k + nv*c]). J is 6 x nv
    # column-major so J^T(m,c) == J[c + 6*m], giving
    #   (Minv J^T)[k,c] = sum_m Minv(k,m) * J[c + 6*m].
    # minv emits SYMMETRIC_UPPER, so read the upper-triangle entry
    # Minv[min(k,m), max(k,m)] to get the full symmetric Minv(k,m).
    self.gen_add_parallel_loop("ind", str(nv * 6))
    self.gen_add_code_line("int k = ind % " + str(nv) + "; int c = ind / " + str(nv) + ";")
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int m = 0; m < " + str(nv) + "; ++m) {", True)
    self.gen_add_code_line("int r0 = (k < m) ? k : m; int c0 = (k < m) ? m : k;")
    self.gen_add_code_line("acc += s_Minv[r0 + " + str(nv) + "*c0] * s_Jfj[c + 6*m];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("s_MJt[ind] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # task = J @ MJt  (6 x 6, column-major: task[i + 6*c] = sum_k J[i + 6*k] * MJt[k + nv*c]).
    self.gen_add_parallel_loop("ind", "36")
    self.gen_add_code_line("int i = ind % 6; int c = ind / 6;")
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int k = 0; k < " + str(nv) + "; ++k) acc += s_Jfj[i + 6*k] * s_MJt[k + " + str(nv) + "*c];")
    self.gen_add_code_line("s_task[ind] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Lambda = task^{-1} (6x6) via block-cooperative GLASS Gauss-Jordan.
    self.gen_add_code_line("invert_matrix<T>(6, s_task, s_taskinv, s_temp);")
    self.gen_add_sync()
    self.gen_add_parallel_loop("ind", "36")
    self.gen_add_code_line("s_Lambda[ind] = s_taskinv[ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_osc_inertia_kernel(self, single_call_timing=False):
    """Emit osc_inertia_kernel: batched launcher of the 6x6 operational-space
    inertia Lambda. SELF-CONTAINED (osc_inertia_device composes Minv on device
    from q alone). target_jid + reference_frame baked as the leaf-EE /
    LOCAL_WORLD_ALIGNED defaults. Output 6x6 (column-major) per timestep.

    Like frame_jacobian_dot_kernel, osc_inertia_device is a full auto-smem
    wrapper that owns the ENTIRE dynamic-smem arena, so the kernel keeps its
    input s_q and output s_osc_inertia in STATIC __shared__ (avoids aliasing the
    wrapper's heavy minv/J/compose scratch). The __launch_bounds__ lets nvcc fit
    the heavy register footprint to the tier thread cap (mirrors the other
    benchmarked kernels; the un-annotated smoke runner needs manual clamping)."""
    n = self.robot.get_num_pos()
    default_tjid = self.robot.get_leaf_nodes()[0]
    func_params = ["d_osc_inertia is the vector of 6 x 6 operational-space inertias Lambda (column-major)",
                   "d_q is the vector of joint positions",
                   "stride_q is the stride between each q",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_def_start = "void osc_inertia_kernel(T *d_osc_inertia, unsigned char *d_workspace, const T *d_q, const int stride_q, "
    func_def_end = "const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("(", "_single_timing(")
    self.gen_add_func_doc("Compute the operational-space (task) inertia", [], func_params, None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag, LAST
    # after RESOURCE_TIER so existing positional <T,TIER> call sites are unaffected.
    # osc_inertia (Lambda = (J Minv J^T)^{-1}) is frame-INVARIANT: the G's cancel,
    # so there is NO output epilogue; only the q quaternion is reordered so the
    # internal J/Minv build from the correct base orientation. Default false ->
    # byte-identical pin codegen.
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("const int target_jid = " + str(default_tjid) + ";")
    self.gen_add_code_line("const int reference_frame = " + str(_REF_LWA) + ";")
    self.gen_add_code_line("__shared__ T s_q[" + str(n) + "];")
    self.gen_add_code_line("__shared__ T s_osc_inertia[36];")
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q", str(n), stride="stride_q")
        # mjx input convert (quaternion only -> the internal J/Minv build from the
        # correct base orientation; Lambda is frame-INVARIANT so NO output epilogue).
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_quat_reorder("s_q")
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute (slice the per-timestep workspace base for the spill-F tier)")
        self.gen_add_code_line("osc_inertia_device<T, RESOURCE_TIER>(s_osc_inertia, target_jid, reference_frame, s_q, d_robotModel, &d_workspace[grid_workspace_slot()*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()]);")
        self.gen_add_sync()
        self.gen_kernel_save_result("osc_inertia", "36", stride="36")
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q", str(n))
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q", str(n), feedback_from="osc_inertia")
        self.gen_add_code_line("osc_inertia_device<T, RESOURCE_TIER>(s_osc_inertia, target_jid, reference_frame, s_q, d_robotModel, d_workspace);")
        self.gen_anti_licm_output_write("osc_inertia")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("osc_inertia", "36")
    self.gen_add_end_function()


def gen_osc_inertia_host(self, mode=0):
    """Emit osc_inertia host launcher (3 modes). Kinematic (q-only input):
    SELF-CONTAINED Minv compose, so the q-only / q|qd|u memory split mirrors
    frame_jacobian."""
    single_call_timing, compute_only = host_mode_flags(mode)
    func_params = ["hd_data is the packaged input and output pointers",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)",
                   "streams are pointers to CUDA streams for async memory transfers (if needed)"]
    func_def_start = "void osc_inertia(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,"
    func_def_end = "                            const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    func_def_start, func_def_end = mangle_host_func_defs(func_def_start, func_def_end, single_call_timing, compute_only)
    self.gen_add_func_doc("Compute the operational-space (task) inertia", [], func_params, None)
    # MUJOCO_OUTPUT (floating only) host flag, LAST: forwarded to the kernel launch
    # (naming the tier positionally to reach the trailing flag). osc_inertia is
    # frame-INVARIANT (no output epilogue) but the kernel still input-converts the
    # quaternion under the flag. Default false -> byte-identical pin codegen.
    mjx_host = gen_host_wrapper_head(self, "osc_inertia", func_def_start, func_def_end, kind_rule="kinematics")
    osc_kernel_tmpl = "osc_inertia_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "osc_inertia_kernel<T, RESOURCE_TIER>"
    func_call_start = (osc_kernel_tmpl + "<<<block_dimms,thread_dimms,OSC_INERTIA_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>"
                       "(hd_data->d_osc_inertia,hd_data->d_workspace,hd_data->d_q,stride_q,")
    func_call_end = "d_robotModel,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("osc_inertia_kernel<", "osc_inertia_kernel_single_timing<")
    if not compute_only:
        self.gen_add_code_lines(host_q_compressed_input_transfer_lines(single_call_timing))
    else:
        self.gen_add_code_line("int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;")
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    func_call_mem_adjust, func_call_mem_adjust2 = gen_launch_pair(func_call, "hd_data->d_q")
    func_call_code = [func_call_mem_adjust, func_call_mem_adjust2, "gpuErrchkKernel();"]
    if single_call_timing:
        wrap_host_single_call_timing(func_call_code)
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"osc_inertia\", OSC_INERTIA_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    if single_call_timing:
        self.gen_add_code_lines(func_call_code)
    else:
        self.gen_add_workspace_clamped_launch(func_call_code)
    if not compute_only:
        gen_emit_host_result_transfer(self, "h_osc_inertia", "d_osc_inertia", "36*", single_call_timing)
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("osc_inertia"))
    self.gen_add_end_function()


def gen_osc_inertia(self):
    self.gen_osc_inertia_device()
    self.gen_osc_inertia_kernel(single_call_timing=False)
    self.gen_osc_inertia_kernel(single_call_timing=True)
    self.gen_osc_inertia_host(mode=0)
    self.gen_osc_inertia_host(mode=1)
    self.gen_osc_inertia_host(mode=2)

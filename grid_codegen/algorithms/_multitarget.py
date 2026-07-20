"""General batched multi-target kinematics emitter (W1b live; W2a gradient prototype).

STATUS:
  * POSITION (build_target_batch + gen_multi_target_position{,_inner,_device,...}) is
    LIVE — wired in algorithms/__init__.py + the GRiDCodeGenerator class import list,
    dispatched from gen_all_code behind the opt-in `multi_target_batch` kwarg (default
    None -> not emitted, all existing robots byte-identical). Validated on iiwa14 +
    baxter (test/cuda_equivalents/test_cuda_multi_target_position.py): world positions
    vs a NumPy FK oracle, offset==0 == end_effector_pose, thread-invariant, sanitizers.
    Kernel/host/gridData-buffer registration + algo_registry descriptor = W1b.3.
  * GRADIENT (gen_multi_target_gradient_phaseB) is still a PROTOTYPE for W2a (the offset
    epilogue dpos = Jv + Jw x (R.r), no FK re-walk). Not wired.

Design: docs/open-tasks/design_W1b_batched_multitarget_position_2026-07-07.md.

A TARGET = (anchor_jid, offset[3]) — a fixed point off a link (named grasp point,
gripper, or collision sphere). A BATCH = an ordered target list + named sub-ranges
("all" + groups). World position pos[t] = X_world[anchor(t)] @ [offset,1], computed
by ONE shared FK (all link world transforms) + a cheap parallel-over-targets
extraction driven by a baked (anchor, offset) table — same table-driven idiom as the
W1a hessian collapse. Subsumes backlog D (multi-named-EE-target).
"""


# ---------------------------------------------------------------------------
# Build-time: assemble the target batch (anchors + baked offsets + named groups)
# ---------------------------------------------------------------------------
def build_target_batch(self, targets):
    """Given `targets` = ordered list of dicts {"anchor_jid": int, "offset": (x,y,z),
    "group": str (optional)}, return a flat batch descriptor:
        {
          "n": N,
          "anchor": [anchor_jid ...]          # len N, GRiD joint/frame index
          "offset": [x,y,z, x,y,z, ...]       # len 3N, LOCAL frame, near-zero snapped
          "groups": {name: (lo, hi)}          # contiguous [lo,hi) index ranges + "all"
        }
    Offsets are snapped |c|<1e-15 -> 0.0 (bit-identical world-axis emission, matches the
    axis-literal snap in _eepose_gradient_hessian). Targets are grouped contiguously so a
    caller can request "all" (whole [0,N)) or a named group's slice.
    NOTE (W3/FLANGE): for foam spheres, `anchor_jid` MUST be resolved through GRiD's own
    URDFParser link->frame id, NOT copied from foam's actuated-joint-count table.
    """
    def _snap(v):
        return [float(c) if abs(c) >= 1e-15 else 0.0 for c in v]

    # stable-sort by group so each group is a contiguous slice; ungrouped -> "_" first.
    ordered = sorted(range(len(targets)), key=lambda i: targets[i].get("group", ""))
    anchor, offset, groups = [], [], {}
    for new_idx, i in enumerate(ordered):
        t = targets[i]
        anchor.append(int(t["anchor_jid"]))
        offset.extend(_snap(t["offset"]))
        g = t.get("group", "")
        if g:
            lo, hi = groups.get(g, (new_idx, new_idx))
            groups[g] = (min(lo, new_idx), new_idx + 1)
    n = len(anchor)
    groups["all"] = (0, n)
    return {"n": n, "anchor": anchor, "offset": offset, "groups": groups}


# NOTE: the shared BFS world-FK chain-up lives in _eepose_gradient_hessian.py as
# emit_world_fk_chainup (committed, byte-identical refactor of the gradient inner's
# Steps 1+1b). The position/gradient emitters below import and call THAT — there is
# no separate copy here (the earlier prototype emit_shared_world_fk was superseded).


# ---------------------------------------------------------------------------
# Batched multi-target POSITION: scratch sizing.
# ---------------------------------------------------------------------------
def gen_multi_target_position_inner_temp_mem_size(self, batch=None):
    """Helper scratch for the position inner = the s_Xworld world-transform arena,
    16 per joint (+ appended fixed-anchor slots when welded targets are baked in).
    Mirrors _eepose_xworld_slot_count in _eepose_gradient_hessian.py so the two
    kinematics paths size the shared FK arena identically. The batch's output buffer
    (s_out_pos = 3*N) is allocated by the kernel/device wrapper, NOT counted here
    (same split as end_effector_pose: s_temp holds only the chain scratch)."""
    from ._eepose_gradient_hessian import _eepose_xworld_slot_count
    return 16 * _eepose_xworld_slot_count(self)


# ---------------------------------------------------------------------------
# Batched multi-target POSITION inner.
# ---------------------------------------------------------------------------
def gen_multi_target_position_inner(self, batch, suffix=""):
    """Emit `multi_target_position<suffix>_inner<T>`: compute ALL targets' world positions in one
    call. s_out_pos is 3*N (xyz per target). One shared FK (s_Xworld) + a parallel
    extraction over the baked (anchor, offset) table. `batch` = build_target_batch(...) output.

    `suffix` (e.g. "_broad") makes the function + macro + NUM_MULTI_TARGETS names unique per
    collision tier so multiple sphere-density batches can coexist in one header; default "" is
    byte-identical to the single-batch emission.

    pos[t][row] = R_world[anchor]·offset + p_world
                = X[row]*o0 + X[row+4]*o1 + X[row+8]*o2 + X[row+12]   (X col-major 4x4)
    """
    n = batch["n"]
    n_joints = self.robot.get_num_joints()  # s_Xworld slot count (extend for fixed anchors as needed)

    # --- function boilerplate (models gen_end_effector_pose_inner) ---
    func_params = [
        "s_out_pos is shared memory of size 3*N_TARGETS (xyz per target), N_TARGETS = " + str(n),
        "s_q is the vector of joint positions",
        "s_Xhom is the per-joint local homogeneous transforms (already updated for q)",
        "s_temp is helper shared memory (holds s_Xworld = 16*NUM_JOINTS)",
        "d_workspace is the global-memory scratch used when !TEMP_IN_SMEM",
    ]
    func_notes = [
        "Computes world positions of a baked batch of fixed-offset targets (grasp points / spheres).",
        "One shared FK (world transforms) + parallel-over-targets offset extraction.",
    ]
    func_def_start = "void multi_target_position" + suffix + "_inner("
    func_def_middle = "T *s_out_pos, const T *s_q, const T *s_Xhom, "
    func_def_end = "T *s_temp, T *d_workspace, unsigned char *s_linalg_smem) {"
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(
        func_def_middle, func_params, -1, NO_XI_FLAG=True)
    self.gen_add_func_doc("Batched multi-target world positions", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, bool TEMP_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def_start + func_def_middle + func_def_end, True)
    self.gen_add_code_line("if constexpr (!TEMP_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }")
    self.gen_add_code_line("(void)s_q; (void)s_linalg_smem;")
    self.gen_add_code_line("T *s_Xworld = s_temp;   // 16 * " + str(n_joints))

    # --- shared FK (the committed emit_world_fk_chainup in _eepose_gradient_hessian) ---
    from ._eepose_gradient_hessian import emit_world_fk_chainup
    emit_world_fk_chainup(
        self,
        header_lines=["//",
                      "// Build world transforms for every joint via BFS-level chain-up",
                      "//"],
        fixed_anchors=None)  # welded (fixed-joint) anchors: pass their (anchor,parent) here (W3)

    # --- baked batch tables ---
    self.gen_add_code_line("// baked target batch: anchor frame id + LOCAL offset per target")
    self.gen_add_code_line("static const int mt_anchor[" + str(n) + "] = {" +
                           ", ".join(str(a) for a in batch["anchor"]) + "};")
    self.gen_add_code_line("const T mt_offset[" + str(3 * n) + "] = {" +
                           ", ".join("static_cast<T>({:.17g})".format(v) for v in batch["offset"]) + "};")

    # --- parallel extraction: one thread per (target, xyz) ---
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Extract each target's world position = R_world[anchor] @ offset + p_world")
    self.gen_add_code_line("//")
    self.gen_add_parallel_loop("ind", str(3 * n))
    self.gen_add_code_line("int row = ind % 3; int t = ind / 3;")
    self.gen_add_code_line("const T *X = &s_Xworld[16 * mt_anchor[t]];")
    self.gen_add_code_line("const T *o = &mt_offset[3 * t];")
    self.gen_add_code_line("s_out_pos[3*t + row] = X[row]*o[0] + X[row + 4]*o[1] + X[row + 8]*o[2] + X[row + 12];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()


# ---------------------------------------------------------------------------
# Inner function-call helper (mirrors gen_end_effector_pose_inner_function_call).
# ---------------------------------------------------------------------------
def gen_multi_target_position_inner_function_call(self, updated_var_names=None, temp_in_smem_expr="true", suffix=""):
    var_names = dict(
        s_Xhom_name="s_XmatsHom",
        s_out_pos_name="s_out_pos",
        s_q_name="s_q",
        s_topology_helpers_name="s_topology_helpers",
        s_temp_name="s_temp",
        d_workspace_name="nullptr",
        s_linalg_smem_name="s_linalg_smem",
    )
    if updated_var_names is not None:
        for key, value in updated_var_names.items():
            var_names[key] = value
    code_start = ("multi_target_position" + suffix + "_inner<T, " + temp_in_smem_expr + ">(" +
                  var_names["s_out_pos_name"] + ", " + var_names["s_q_name"] + ", ")
    code_middle = var_names["s_Xhom_name"] + ", "
    code_end = var_names["s_temp_name"] + ", " + var_names["d_workspace_name"] + ", " + var_names["s_linalg_smem_name"] + ");"
    # NO_XI: this family takes s_Xhom (not s_XImats); mirror the def's NO_XI_FLAG.
    code_middle += self.gen_insert_helpers_function_call(updated_var_names=var_names, NO_XI_FLAG=True)
    self.gen_add_code_line(code_start + code_middle + code_end)


# ---------------------------------------------------------------------------
# Device wrapper (models gen_end_effector_pose_device): allocates the shared
# XmatsHom/temp arena, builds the local per-joint transforms for q, then calls
# the batched inner. Caller supplies the s_out_pos output buffer (3*N shared),
# exactly like end_effector_pose_device(s_pose, d_q, m). No gridData dependency.
# ---------------------------------------------------------------------------
def gen_multi_target_position_device(self, batch, suffix=""):
    n = batch["n"]
    shared_mem_size = self.gen_multi_target_position_inner_temp_mem_size(batch)
    func_params = [
        "s_out_pos is a pointer to memory of size 3*N_TARGETS where N_TARGETS = " + str(n) +
        " (caller chooses smem for small batches or a global buffer for many spheres)",
        "s_q is the vector of joint positions",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
        "d_workspace is the global scratch buffer; size MULTI_TARGET_POSITION" + suffix.upper() + "_DEVICE_INLINE_WORKSPACE_BYTES<T, RESOURCE_TIER>() bytes "
        "(= 0 at TIER_SHARED, " + str(shared_mem_size) + "*sizeof(T) at TIER_LITE+). Pass nullptr at TIER_SHARED"]
    func_notes = [
        "Computes world positions of a baked batch of fixed-offset targets (grasp points / spheres).",
        "Inline-CUDA / grid_collision users: at TIER_LITE/TIER_MINIMAL the shared FK scratch (s_Xworld, ~" +
        str(shared_mem_size) + "*sizeof(T) bytes) moves from smem to d_workspace, freeing smem for the caller's outer kernel.",
        "Output placement is the CALLER's choice (the s_out_pos pointer): smem for small batches, a global buffer when 3*N is large."]
    func_def_start = "void multi_target_position" + suffix + "_device("
    func_def_middle = "T *s_out_pos, const T *s_q, "
    func_def_end = "const robotModel<T> *d_robotModel, T *d_workspace = nullptr) {"
    func_def = func_def_start + func_def_middle + func_def_end
    self.gen_add_func_doc("Computes batched multi-target world positions", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # Tier-aware arena: at TIER_SHARED s_temp (the s_Xworld FK scratch) lives in the smem
    # arena; at TIER_LITE/MINIMAL the whole s_temp slot is sourced from d_workspace and the
    # arena skips it, mirroring forward_dynamics_device. The inner is called with the default
    # TEMP_IN_SMEM=true because the s_temp pointer already routes per tier.
    self.gen_XmatsHom_helpers_temp_shared_memory_code(shared_mem_size, include_linalg_scratch=True,
                                                      linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()",
                                                      tier_workspace_expr="d_workspace")
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_multi_target_position_inner_function_call(suffix=suffix)
    self.gen_add_end_function()


# ---------------------------------------------------------------------------
# Dispatcher — W1b.2 scope: inner + device (validated via a focused runner that
# calls multi_target_position_device directly). Kernel + host + gridData buffer
# registration = W1b.3.
# ---------------------------------------------------------------------------
def gen_multi_target_position(self, batch, suffix="", emit_num_const=True):
    n = batch["n"]
    NUM = "NUM_MULTI_TARGETS" + suffix.upper()
    POS = "MULTI_TARGET_POSITION" + suffix.upper()
    XHom_size, _dXhom, _d2Xhom = self.gen_get_Xhom_size()
    scratch = self.gen_multi_target_position_inner_temp_mem_size(batch)  # s_Xworld FK scratch
    total_t = XHom_size + scratch
    # `emit_num_const=False` when the caller already emitted `const int NUM_MULTI_TARGETS`
    # EARLY (before gen_init_gridData, so the guarded gridData mallocs can size on it) —
    # the public multi_target_batch path. The collision path keeps emitting it here (its
    # suffixed per-tier constants have no early emission and no gridData dependency).
    self.gen_add_code_lines([
        "// W1b batched multi-target world positions (opt-in via multi_target_batch); " + NUM + " = " + str(n)]
        + (["const int " + NUM + " = " + str(n) + ";"] if emit_num_const else []) + [
        # Tier-aware smem arena: at TIER_SHARED the FK scratch (s_Xworld) is in smem; at
        # TIER_LITE/MINIMAL it spills to the device fn's d_workspace, shrinking the arena to
        # s_XmatsHom + linalg only. Default TIER keeps every single-arg call site working.
        "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t " + POS + "_DYNAMIC_SHARED_MEM_BYTES() { "
        "if constexpr (TIER == TIER_SHARED) return grid_shared_arena_bytes<T>(" + str(total_t) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
        "else return grid_shared_arena_bytes<T>(" + str(XHom_size) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }",
        # Companion d_workspace sizing for multi_target_position<suffix>_device at TIER_LITE+.
        "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t " + POS + "_DEVICE_INLINE_WORKSPACE_BYTES() "
        "{ return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(scratch) + "); }",
    ])
    self.gen_multi_target_position_inner(batch, suffix=suffix)
    self.gen_multi_target_position_device(batch, suffix=suffix)


# ---------------------------------------------------------------------------
# Batched multi-target position GRADIENT (W2a): anchor-deduped geometric Jacobian
# (Phase A, shared helper) + per-target offset epilogue (Phase B, no FK re-walk).
# Design: docs/open-tasks/design_W2a_batched_multitarget_gradient_2026-07-07.md
#   dpos[t][:,vi] = Jv[anchor,:,vi] + Jw[anchor,:,vi] x (R_world[anchor] . r_local)
# ---------------------------------------------------------------------------
def _multi_target_anchor_dedup(batch):
    """Order-preserving distinct anchors + per-target index into that set. The
    geometric Jacobian is built ONCE per distinct anchor (bounded by #links); only the
    OUTPUT scales with target count (the anchor-dedup collapse -- same win as W1a)."""
    distinct, idx_of, pos = [], [], {}
    for a in batch["anchor"]:
        if a not in pos:
            pos[a] = len(distinct)
            distinct.append(a)
        idx_of.append(pos[a])
    return distinct, idx_of


def gen_multi_target_position_gradient_inner_temp_mem_size(self, batch):
    """Scratch = s_Xworld (16 * n_xworld) | s_Jv (3*nv*N_anchors) | s_Jw (3*nv*N_anchors)
    | s_ro (3*N_targets). Jv/Jw are deduped over DISTINCT anchors; s_ro holds each
    target's world-rotated offset (Phase-B pre-pass, vi-independent)."""
    from ._eepose_gradient_hessian import _eepose_xworld_slot_count
    nv = self.robot.get_num_vel()
    distinct, _ = _multi_target_anchor_dedup(batch)
    return 16 * _eepose_xworld_slot_count(self) + 2 * 3 * nv * len(distinct) + 3 * batch["n"]


def gen_multi_target_position_gradient_inner(self, batch, suffix=""):
    """Emit multi_target_position_gradient<suffix>_inner<T>: d(world pos)/dv for every target
    (3 x nv per target, row-fastest layout ob = 3*(nv*t+vi)+row). Phase A builds s_Jv/s_Jw
    per DISTINCT anchor via the shared emit_geometric_jacobian_jvjw; Phase B applies the
    offset epilogue. Position gradient only (world-frame LOCAL_WORLD_ALIGNED; no rpy)."""
    from ._eepose_gradient_hessian import (
        emit_world_fk_chainup, _eepose_xworld_slot_count,
        _eepose_grad_chain_metadata, group_jacobian_jobs, emit_geometric_jacobian_jvjw)
    nv = self.robot.get_num_vel()
    n = batch["n"]
    n_xworld = _eepose_xworld_slot_count(self)
    distinct_anchors, anchor_idx_of_target = _multi_target_anchor_dedup(batch)
    n_anchor = len(distinct_anchors)

    func_params = [
        "s_out_grad is shared memory of size 3*NUM_VEL*N_TARGETS (3 x nv per target), N_TARGETS = " + str(n) + ", NUM_VEL = " + str(nv),
        "s_q is the vector of joint positions (unused; kept for signature parity)",
        "s_Xhom is the per-joint LOCAL homogeneous transforms (already updated for q)",
        "s_temp is helper shared memory (Xworld | Jv | Jw | ro)",
        "d_workspace is the global-memory scratch used when !TEMP_IN_SMEM",
    ]
    func_notes = [
        "Position gradient d(world pos)/dv of a baked batch of fixed-offset targets (grasp points / spheres).",
        "Anchor-deduped geometric Jacobian (built once per distinct anchor) + offset epilogue; NO FK re-walk.",
    ]
    func_def_start = "void multi_target_position_gradient" + suffix + "_inner("
    func_def_middle = "T *s_out_grad, const T *s_q, const T *s_Xhom, "
    func_def_end = "T *s_temp, T *d_workspace, unsigned char *s_linalg_smem) {"
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(
        func_def_middle, func_params, -1, NO_XI_FLAG=True)
    self.gen_add_func_doc("Batched multi-target world-position gradient", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, bool TEMP_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def_start + func_def_middle + func_def_end, True)
    self.gen_add_code_line("if constexpr (!TEMP_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }")
    self.gen_add_code_line("(void)s_q; (void)s_linalg_smem;")

    off_Jv = 16 * n_xworld
    off_Jw = off_Jv + 3 * nv * n_anchor
    off_ro = off_Jw + 3 * nv * n_anchor
    self.gen_add_code_line("// scratch layout: Xworld | Jv (3 x nv x anchor) | Jw (3 x nv x anchor) | ro (3 x target)")
    self.gen_add_code_line("T *s_Xworld = &s_temp[0];")
    self.gen_add_code_line("T *s_Jv     = &s_temp[" + str(off_Jv) + "];")
    self.gen_add_code_line("T *s_Jw     = &s_temp[" + str(off_Jw) + "];")
    self.gen_add_code_line("T *s_ro     = &s_temp[" + str(off_ro) + "];")

    # Phase A step 1: shared FK
    emit_world_fk_chainup(
        self,
        header_lines=["//", "// Step 1: build world transforms for every joint via BFS-level chain-up", "//"],
        fixed_anchors=None)
    # Phase A steps 2+3+3b: geometric Jacobian per DISTINCT anchor (shared with ee-pose gradient)
    _chains, anchors, fill_jobs = _eepose_grad_chain_metadata(self, distinct_anchors, "", anchor_override=None)
    single_jobs, multi_groups, has_mimic = group_jacobian_jobs(self, fill_jobs, anchors)
    emit_geometric_jacobian_jvjw(self, nv, n_anchor, single_jobs, multi_groups, has_mimic)

    # Phase B: baked batch tables
    self.gen_add_code_line("// baked batch: target -> anchor world-frame jid, target -> deduped anchor slot, LOCAL offset")
    self.gen_add_code_line("static const int mt_anchor[" + str(n) + "] = {" + ", ".join(str(a) for a in batch["anchor"]) + "};")
    self.gen_add_code_line("static const int mt_anchor_idx[" + str(n) + "] = {" + ", ".join(str(a) for a in anchor_idx_of_target) + "};")
    self.gen_add_code_line("const T mt_offset[" + str(3 * n) + "] = {" + ", ".join("static_cast<T>({:.17g})".format(v) for v in batch["offset"]) + "};")
    # Phase B pre-pass: rotate each target's LOCAL offset into world (vi-independent).
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Phase B pre-pass: ro[t] = R_world[anchor(t)] @ offset(t)  (once per target)")
    self.gen_add_code_line("//")
    self.gen_add_parallel_loop("t", str(n))
    self.gen_add_code_line("const T *X = &s_Xworld[16 * mt_anchor[t]];")
    self.gen_add_code_line("const T *o = &mt_offset[3 * t];")
    self.gen_add_code_line("s_ro[3*t + 0] = X[0]*o[0] + X[4]*o[1] + X[8]*o[2];")
    self.gen_add_code_line("s_ro[3*t + 1] = X[1]*o[0] + X[5]*o[1] + X[9]*o[2];")
    self.gen_add_code_line("s_ro[3*t + 2] = X[2]*o[0] + X[6]*o[1] + X[10]*o[2];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # Phase B main: per (target, vi) offset-corrected column: dpos = Jv + Jw x ro.
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Phase B: dpos[t][:,vi] = Jv[anchor,:,vi] + Jw[anchor,:,vi] x ro[t]  (Jw=0 for prismatic -> no cross)")
    self.gen_add_code_line("//")
    self.gen_add_parallel_loop("ind", str(n * nv))
    self.gen_add_code_line("int vi = ind % " + str(nv) + "; int t = ind / " + str(nv) + ";")
    self.gen_add_code_line("int jb = 3 * (" + str(nv) + " * mt_anchor_idx[t] + vi);")
    self.gen_add_code_line("T Jv0 = s_Jv[jb+0], Jv1 = s_Jv[jb+1], Jv2 = s_Jv[jb+2];")
    self.gen_add_code_line("T Jw0 = s_Jw[jb+0], Jw1 = s_Jw[jb+1], Jw2 = s_Jw[jb+2];")
    self.gen_add_code_line("T r0 = s_ro[3*t+0], r1 = s_ro[3*t+1], r2 = s_ro[3*t+2];")
    self.gen_add_code_line("int ob = 3 * (" + str(nv) + " * t + vi);")
    self.gen_add_code_line("s_out_grad[ob + 0] = Jv0 + (Jw1*r2 - Jw2*r1);")
    self.gen_add_code_line("s_out_grad[ob + 1] = Jv1 + (Jw2*r0 - Jw0*r2);")
    self.gen_add_code_line("s_out_grad[ob + 2] = Jv2 + (Jw0*r1 - Jw1*r0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_multi_target_position_gradient_inner_function_call(self, updated_var_names=None, temp_in_smem_expr="true", suffix=""):
    var_names = dict(
        s_Xhom_name="s_XmatsHom",
        s_out_grad_name="s_out_grad",
        s_q_name="s_q",
        s_topology_helpers_name="s_topology_helpers",
        s_temp_name="s_temp",
        d_workspace_name="nullptr",
        s_linalg_smem_name="s_linalg_smem",
    )
    if updated_var_names is not None:
        for key, value in updated_var_names.items():
            var_names[key] = value
    code_start = ("multi_target_position_gradient" + suffix + "_inner<T, " + temp_in_smem_expr + ">(" +
                  var_names["s_out_grad_name"] + ", " + var_names["s_q_name"] + ", ")
    code_middle = var_names["s_Xhom_name"] + ", "
    code_end = var_names["s_temp_name"] + ", " + var_names["d_workspace_name"] + ", " + var_names["s_linalg_smem_name"] + ");"
    code_middle += self.gen_insert_helpers_function_call(updated_var_names=var_names, NO_XI_FLAG=True)
    self.gen_add_code_line(code_start + code_middle + code_end)


def gen_multi_target_position_gradient_device(self, batch, suffix=""):
    n = batch["n"]
    shared_mem_size = self.gen_multi_target_position_gradient_inner_temp_mem_size(batch)
    func_params = [
        "s_out_grad is a pointer to memory of size 3*NUM_VEL*N_TARGETS where N_TARGETS = " + str(n) +
        " (caller chooses smem for small batches or a global buffer for many spheres)",
        "s_q is the vector of joint positions",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
        "d_workspace is the global scratch buffer; size MULTI_TARGET_POSITION_GRADIENT" + suffix.upper() + "_DEVICE_INLINE_WORKSPACE_BYTES<T, RESOURCE_TIER>() bytes "
        "(= 0 at TIER_SHARED, " + str(shared_mem_size) + "*sizeof(T) at TIER_LITE+). Pass nullptr at TIER_SHARED"]
    func_notes = [
        "Position gradient d(world pos)/dv of a baked batch of fixed-offset targets.",
        "Inline-CUDA / grid_collision users: at TIER_LITE/TIER_MINIMAL the anchor-deduped Jacobian scratch "
        "(s_Xworld|Jv|Jw|ro, ~" + str(shared_mem_size) + "*sizeof(T) bytes; Jv/Jw dominate on big robots) moves from smem to d_workspace.",
        "Output placement is the CALLER's choice (the s_out_grad pointer): smem for small batches, a global buffer when 3*nv*N is large."]
    func_def_start = "void multi_target_position_gradient" + suffix + "_device("
    func_def_middle = "T *s_out_grad, const T *s_q, "
    func_def_end = "const robotModel<T> *d_robotModel, T *d_workspace = nullptr) {"
    func_def = func_def_start + func_def_middle + func_def_end
    self.gen_add_func_doc("Computes batched multi-target world-position gradient", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # Tier-aware arena: at TIER_SHARED s_temp (Xworld|Jv|Jw|ro) lives in smem; at TIER_LITE/MINIMAL
    # the whole s_temp slot is sourced from d_workspace and the arena skips it (the load-bearing
    # spill for many-sphere collision, where 3*nv*n_anchor Jacobian scratch dominates).
    self.gen_XmatsHom_helpers_temp_shared_memory_code(shared_mem_size, include_linalg_scratch=True,
                                                      linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()",
                                                      tier_workspace_expr="d_workspace")
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_multi_target_position_gradient_inner_function_call(suffix=suffix)
    self.gen_add_end_function()


def gen_multi_target_position_gradient(self, batch, suffix=""):
    POSG = "MULTI_TARGET_POSITION_GRADIENT" + suffix.upper()
    XHom_size, _dXhom, _d2Xhom = self.gen_get_Xhom_size()
    scratch = self.gen_multi_target_position_gradient_inner_temp_mem_size(batch)  # Xworld|Jv|Jw|ro
    total_t = XHom_size + scratch
    self.gen_add_code_lines([
        "// W2a batched multi-target world-position GRADIENT (opt-in via multi_target_batch)",
        # Tier-aware smem arena: TIER_SHARED keeps the Jacobian scratch in smem; TIER_LITE/MINIMAL
        # spills it to the device fn's d_workspace, shrinking the arena to s_XmatsHom + linalg.
        "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t " + POSG + "_DYNAMIC_SHARED_MEM_BYTES() { "
        "if constexpr (TIER == TIER_SHARED) return grid_shared_arena_bytes<T>(" + str(total_t) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); "
        "else return grid_shared_arena_bytes<T>(" + str(XHom_size) + ", TOPOLOGY_HELPERS_COUNT, GRID_EE_LINALG_SHARED_BYTES<T>()); }",
        "template <typename T, int TIER = GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ constexpr size_t " + POSG + "_DEVICE_INLINE_WORKSPACE_BYTES() "
        "{ return (TIER == TIER_SHARED) ? static_cast<size_t>(0) : sizeof(T) * static_cast<size_t>(" + str(scratch) + "); }",
    ])
    self.gen_multi_target_position_gradient_inner(batch, suffix=suffix)
    self.gen_multi_target_position_gradient_device(batch, suffix=suffix)


# ===========================================================================
# W1b.3 / W2a.3 — launchable KERNEL + 3-mode HOST wrappers (bench registration).
# Emitted ONLY for the opt-in public multi_target_batch (suffix-free); the per-tier
# collision batches stay device-composite (config_free drives them internally). The
# *_device wrappers own the ENTIRE dynamic-smem arena (s_XmatsHom + FK/Jacobian
# scratch), so the kernel keeps its q input + world-position output in STATIC
# __shared__ (mirrors frame_jacobian_dot_kernel) and launches with the *_device
# DYNAMIC_SHARED_MEM_BYTES macro. multi_target is a world-frame quantity computed by
# manipulators (grasp points / spheres) -> no MUJOCO_OUTPUT convention flag.
# ===========================================================================
def gen_multi_target_position_kernel(self, batch, single_call_timing=False):
    n_pos = self.robot.get_num_pos()
    out_size = 3 * batch["n"]
    func_params = [
        "d_multi_target_position is the vector of 3*NUM_MULTI_TARGETS world positions per timestep",
        "d_q is the vector of joint positions",
        "stride_q is the stride between each q",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
        "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_def_start = "void multi_target_position_kernel(T *d_multi_target_position, const T *d_q, const int stride_q, "
    func_def_end = "const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("(", "_single_timing(")
    self.gen_add_func_doc("Compute batched multi-target world positions", [], func_params, None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # STATIC __shared__ I/O; the *_device wrapper owns the dynamic arena (see header note).
    self.gen_add_code_line("__shared__ T s_q[" + str(n_pos) + "];")
    self.gen_add_code_line("__shared__ T s_multi_target_position[" + str(out_size) + "];")
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q", str(n_pos), stride="stride_q")
        self.gen_add_code_line("// compute")
        self.gen_add_code_line("multi_target_position_device<T>(s_multi_target_position, s_q, d_robotModel);")
        self.gen_add_sync()
        self.gen_kernel_save_result("multi_target_position", str(out_size), stride=str(out_size))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q", str(n_pos))
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q", str(n_pos), feedback_from="multi_target_position")
        self.gen_add_code_line("multi_target_position_device<T>(s_multi_target_position, s_q, d_robotModel);")
        self.gen_anti_licm_output_write("multi_target_position")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("multi_target_position", str(out_size))
    self.gen_add_end_function()


def gen_multi_target_position_host(self, batch, mode=0):
    single_call_timing = True if mode == 1 else False
    compute_only = True if mode == 2 else False
    func_params = ["hd_data is the packaged input and output pointers",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)",
                   "streams are pointers to CUDA streams for async memory transfers (if needed)"]
    func_def_start = "void multi_target_position(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,"
    func_def_end = "                            const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    self.gen_add_func_doc("Compute batched multi-target world positions", [], func_params, None)
    self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, \"multi_target_position requires all-data or kinematics gridData\");")
    func_call_start = ("multi_target_position_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>()>>>"
                       "(hd_data->d_multi_target_position,hd_data->d_q,stride_q,")
    func_call_end = "d_robotModel,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("multi_target_position_kernel<", "multi_target_position_kernel_single_timing<")
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
    func_call_mem_adjust2 = "else                    {" + func_call.replace("hd_data->d_q", "hd_data->d_q_qd_u") + "}"
    func_call_code = [func_call_mem_adjust, func_call_mem_adjust2, "gpuErrchkKernel();"]
    if single_call_timing:
        func_call_code.insert(0, "struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"multi_target_position\", MULTI_TARGET_POSITION_DYNAMIC_SHARED_MEM_BYTES<T>()));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        self.gen_add_code_lines(["// finally transfer the result back",
                                 "gpuErrchk(cudaMemcpy(hd_data->h_multi_target_position,hd_data->d_multi_target_position,3*NUM_MULTI_TARGETS*" +
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchkKernel();"])
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("multi_target_position"))
    self.gen_add_end_function()


def gen_multi_target_position_gradient_kernel(self, batch, single_call_timing=False):
    n_pos = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    out_size = 3 * nv * batch["n"]
    func_params = [
        "d_multi_target_position_gradient is the vector of 3*NUM_VEL*NUM_MULTI_TARGETS position gradients per timestep",
        "d_q is the vector of joint positions",
        "stride_q is the stride between each q",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
        "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_def_start = "void multi_target_position_gradient_kernel(T *d_multi_target_position_gradient, const T *d_q, const int stride_q, "
    func_def_end = "const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("(", "_single_timing(")
    self.gen_add_func_doc("Compute batched multi-target world-position gradient", [], func_params, None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("__shared__ T s_q[" + str(n_pos) + "];")
    self.gen_add_code_line("__shared__ T s_multi_target_position_gradient[" + str(out_size) + "];")
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q", str(n_pos), stride="stride_q")
        self.gen_add_code_line("// compute")
        self.gen_add_code_line("multi_target_position_gradient_device<T>(s_multi_target_position_gradient, s_q, d_robotModel);")
        self.gen_add_sync()
        self.gen_kernel_save_result("multi_target_position_gradient", str(out_size), stride=str(out_size))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q", str(n_pos))
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q", str(n_pos), feedback_from="multi_target_position_gradient")
        self.gen_add_code_line("multi_target_position_gradient_device<T>(s_multi_target_position_gradient, s_q, d_robotModel);")
        self.gen_anti_licm_output_write("multi_target_position_gradient")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("multi_target_position_gradient", str(out_size))
    self.gen_add_end_function()


def gen_multi_target_position_gradient_host(self, batch, mode=0):
    single_call_timing = True if mode == 1 else False
    compute_only = True if mode == 2 else False
    func_params = ["hd_data is the packaged input and output pointers",
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)",
                   "streams are pointers to CUDA streams for async memory transfers (if needed)"]
    func_def_start = "void multi_target_position_gradient(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,"
    func_def_end = "                            const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    self.gen_add_func_doc("Compute batched multi-target world-position gradient", [], func_params, None)
    self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_KINEMATICS, \"multi_target_position_gradient requires all-data or kinematics gridData\");")
    func_call_start = ("multi_target_position_gradient_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()>>>"
                       "(hd_data->d_multi_target_position_gradient,hd_data->d_q,stride_q,")
    func_call_end = "d_robotModel,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("multi_target_position_gradient_kernel<", "multi_target_position_gradient_kernel_single_timing<")
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
    func_call_mem_adjust2 = "else                    {" + func_call.replace("hd_data->d_q", "hd_data->d_q_qd_u") + "}"
    func_call_code = [func_call_mem_adjust, func_call_mem_adjust2, "gpuErrchkKernel();"]
    if single_call_timing:
        func_call_code.insert(0, "struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"multi_target_position_gradient\", MULTI_TARGET_POSITION_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        self.gen_add_code_lines(["// finally transfer the result back",
                                 "gpuErrchk(cudaMemcpy(hd_data->h_multi_target_position_gradient,hd_data->d_multi_target_position_gradient,3*NUM_VEL*NUM_MULTI_TARGETS*" +
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchkKernel();"])
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("multi_target_position_gradient"))
    self.gen_add_end_function()


def gen_multi_target_position_bench(self, batch):
    """Emit the launchable kernel + 3-mode host for the public multi_target batch
    (position + gradient). Called ONLY from the opt-in multi_target_batch dispatch
    (NOT the per-tier collision batches, which stay device-composite)."""
    self.gen_multi_target_position_kernel(batch, single_call_timing=False)
    self.gen_multi_target_position_kernel(batch, single_call_timing=True)
    self.gen_multi_target_position_host(batch, mode=0)
    self.gen_multi_target_position_host(batch, mode=1)
    self.gen_multi_target_position_host(batch, mode=2)
    self.gen_multi_target_position_gradient_kernel(batch, single_call_timing=False)
    self.gen_multi_target_position_gradient_kernel(batch, single_call_timing=True)
    self.gen_multi_target_position_gradient_host(batch, mode=0)
    self.gen_multi_target_position_gradient_host(batch, mode=1)
    self.gen_multi_target_position_gradient_host(batch, mode=2)


# ---------------------------------------------------------------------------
# INTEGRATION STATUS
# ---------------------------------------------------------------------------
# [x] gen_multi_target_position_inner_temp_mem_size -> 16 * n_xworld_slots (+ fixed anchors)
# [x] gen_multi_target_position_device wrapper (models gen_end_effector_pose_device);
#     tier-awareness deferred to W2b.
# [x] shared world-FK: gradient inner Steps 1+1b factored into emit_world_fk_chainup
#     (_eepose_gradient_hessian.py), byte-identical gate PASSED (GCG cb73296); this
#     emitter imports and calls it.
# [x] wired into GRiDCodeGenerator.gen_all_code behind the opt-in multi_target_batch kwarg
#     (default None -> not emitted; existing robots byte-identical).
# [x] test W1b: baxter (multi-anchor) + iiwa14 (offset==0==ee_pose) positions vs NumPy FK
#     oracle; thread-invariance 1/32/256 (bit-identical); synccheck/racecheck/memcheck clean.
# [x] W2a GRADIENT: anchor-deduped geometric Jacobian (Phase A, shared emit_geometric_jacobian_jvjw,
#     byte-identical refactor GCG 44a7014) + offset epilogue (Phase B). Validated: baxter+iiwa14
#     vs central-diff FD oracle; offset==0 == ee_pose_gradient rows 0..2 BIT-IDENTICAL;
#     thread-invariant; sanitizers clean.
# [x] W1b.3 / W2a.3 (Inc4b): gen_multi_target_position_bench() emits _kernel (+_single_timing)
#     + 3-mode _host for BOTH position and gradient (public suffix-free batch only; collision
#     tiers stay device-composite). gridData d_/h_ buffers (unconditional pointers; malloc/free
#     #if GRID_HAS_MULTI_TARGET_POSITION-guarded), NUM_MULTI_TARGETS + GRID_HAS_MULTI_TARGET_POSITION
#     emitted EARLY (before gen_init_gridData), KERNEL_OVERLOADS pins, algo_registry AlgoEntry +
#     AlgoDescriptor(has_kernel_attr=True, gate_attr) rows. Kernel keeps q/out in STATIC __shared__;
#     the *_device wrapper owns the dynamic arena -> launches with the *_device SHARED_MEM_BYTES macro.
# W2b (remaining): spill-tier the batched outputs + <T,TIER> reconciliation (fold registration).

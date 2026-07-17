"""Second-order inverse-dynamics derivatives (∂²τ/∂x²), body- and world-frame.

NOTE: joint damping/friction contribute NOTHING to second-order derivatives, so
this file emits no damping term. The damping bias τ_damp = b·qd is linear in qd
(∂²/∂*∂* = 0); Coulomb friction f·sign(qd) is non-smooth (subgradient 0 a.e.).
They appear only in the FIRST-order inverse_dynamics_gradient output, never here.
"""

SHARED_MEMORY_JOINT_THRESHOLD = 10 # Max shared memory threshold => Write directly to RAM

# EXP-1 (perf_idsva_so_bigrobot.md): high-DOF FIXED-base robots route to the world-frame
# inner too. Measured crossover is between go2 (fixed NV=12, body-frame wins) and g1 (fixed
# NV=29, world-frame wins ~1.27x: 1218us body -> ~958us world @ LITE/224). The body arena
# forces SHARED-tier occupancy collapse past this point; the world-frame inner spills its
# cold trio to d_workspace at LITE tier, recovering occupancy. Threshold sits in the (12,29)
# gap; refine from the A/B crossover if a mid-DOF robot is measured. Routes g1/h1_2/h2_plus
# (NV 29/39/75) to world; iiwa14/go2 (NV 7/12) stay body (byte-identical, predicate false).
NV_FIXED_WORLD_THRESHOLD = 20

def _idsva_so_use_world_frame(self):
    """Codegen-time frame selection for the dispatching `idsva_so` entry points.

    Route to the WORLD-frame inner for floating-base OR spherical-joint robots OR
    high-DOF fixed-base robots (NV >= NV_FIXED_WORLD_THRESHOLD); body-frame for
    low/mid-DOF cardinal fixed-base robots.

    Why spherical -> world: the body-frame inner uses single-DoF contractions that
    are WRONG for a 6x3 spherical motion subspace, while the world-frame inner is
    already per-velocity-column multi-DoF-aware (built for floating/mimic). Its
    metadata helper `_idsva_so_floating_velocity_metadata` iterates each S column,
    maps it to its reduced v-slot via `get_joint_index_v`, and the triple-ancestor
    walk contracts column-agnostically keyed on `wf_body_v_start`/`wf_body_v_index`.
    For a (non-mimic) spherical robot that table is the identity per-column map with
    alpha == 1 and n_int == NV, so the existing non-mimic world path is already
    exactly correct (no internal slab / fold needed). Mirrors how floating/mimic
    already route here. Byte-identical for low/mid-DOF cardinal fixed-base robots
    (predicate false). See docs/idsva_so_inner_refactor_notes.md + the spherical
    progress memo + docs/open-tasks/perf_idsva_so_bigrobot.md (EXP-1 high-DOF route).
    """
    return (self.robot.floating_base
            or self.robot.robot_has_spherical()
            or self.robot.get_num_vel() >= NV_FIXED_WORLD_THRESHOLD)

def _idsva_so_as_index_list(index):
    if isinstance(index, list):
        return [int(v) for v in index]
    if isinstance(index, tuple):
        return [int(v) for v in index]
    if hasattr(index, "flatten"):
        return [int(v) for v in index.flatten()]
    return [int(index)]

def _idsva_so_int_array(values):
    return ", ".join(map(str, values)) if values else "0"

def _idsva_so_unit_axis(column):
    values = column.reshape(-1).tolist() if hasattr(column, "reshape") else list(column)
    for row, value in enumerate(values):
        value = float(value)
        if abs(value) == 1.0:
            return row, (1 if value > 0.0 else -1)
    raise ValueError("Floating IDSVA-SO expected unit joint subspace columns.")

def _idsva_so_floating_velocity_metadata(robot):
    num_bodies = robot.get_num_bodies()
    body_v_start = [0]
    body_v_index = []
    vel_to_body = [0] * robot.get_num_vel()
    vel_to_local_col = [0] * robot.get_num_vel()
    vel_s_index = [0] * robot.get_num_vel()
    vel_s_sign = [1] * robot.get_num_vel()

    # --- Mimic-aware INTERNAL-coordinate tables (mirror RBDReference.idsva_so_world_frame's
    # has_mimic path). When the robot has mimic joints, several bodies (a mimic + its
    # target) SHARE one reduced velocity slot, so writing the per-column forward-sweep /
    # output state keyed on the shared slot would clobber across siblings. We instead give
    # EVERY body-velocity-column a UNIQUE internal slot (n_int = total column count), run
    # the whole assembly in internal coords into a 4*n_int^3 slab, and alpha-fold each axis
    # back to the reduced 4*NV^3 public output at the end. The floating root's 6 DoF need no
    # special handling — its 6 columns simply get 6 distinct internal slots that fold
    # identity (alpha=1) to reduced slots 0..5, so the per-root-DoF treatment is emergent
    # from the per-column internal slotting (no separate 6-DoF root fold needed).
    body_vint_index = []        # internal slot (0..n_int-1) per body-velocity-column
    int_true_vel = []           # internal slot -> reduced (project NV-space) velocity slot
    int_alpha = []              # internal slot -> mimic multiplier (1.0 for non-mimic)
    int_s_index = []            # internal slot -> S unit-axis row
    int_s_sign = []             # internal slot -> S unit-axis sign

    for body_id in range(num_bodies):
        v_inds = _idsva_so_as_index_list(robot.get_joint_index_v(body_id))
        S = robot.get_S_by_id(body_id)
        if len(S.shape) == 1:
            S_cols = [S]
        else:
            S_cols = [S[:, col] for col in range(S.shape[1])]
        if len(v_inds) != len(S_cols):
            raise ValueError(
                "Floating IDSVA-SO velocity metadata expected one S column per velocity index."
            )
        joint = robot.get_joint_by_id(body_id)
        is_mimic_body = getattr(joint, "is_mimic", False)
        alpha = float(joint.get_mimic_multiplier()) if is_mimic_body else 1.0
        for local_col, vel_index in enumerate(v_inds):
            body_v_index.append(vel_index)
            s_index, s_sign = _idsva_so_unit_axis(S_cols[local_col])
            # Internal-coords tables: one UNIQUE slot per column regardless of sharing.
            body_vint_index.append(len(int_true_vel))
            int_true_vel.append(vel_index)
            int_alpha.append(alpha)
            int_s_index.append(s_index)
            int_s_sign.append(s_sign)
            # vel_to_body / vel_to_local_col / vel_s_* map a reduced velocity
            # slot to its CANONICAL owning body. A mimic joint SHARES its
            # target's v-slot, so it must NOT overwrite the target's assignment
            # (the target — a non-mimic joint — is the canonical owner; the
            # mimic's contribution folds in via its alpha multiplier elsewhere).
            # Bodies are visited in id order with the target defined before its
            # mimic, so guarding on is_mimic keeps the target's mapping intact.
            if not is_mimic_body:
                vel_to_body[vel_index] = body_id
                vel_to_local_col[vel_index] = local_col
                vel_s_index[vel_index] = s_index
                vel_s_sign[vel_index] = s_sign
        body_v_start.append(len(body_v_index))
    n_int = len(int_true_vel)

    subtree_v_start = [0]
    subtree_v_index = []
    successor_v_start = [0]
    successor_v_index = []
    ancestor_body_start = [0]
    ancestor_body_index = []
    for body_id in range(num_bodies):
        subtree = list(robot.get_subtree_by_id(body_id))
        successors = [subtree_body for subtree_body in subtree if subtree_body != body_id]
        ancestors = list(robot.get_ancestors_by_id(body_id))
        ancestors.insert(0, body_id)
        ancestors = ancestors[::-1]

        for subtree_body in subtree:
            subtree_v_index.extend(
                body_v_index[body_v_start[subtree_body]:body_v_start[subtree_body + 1]]
            )
        subtree_v_start.append(len(subtree_v_index))

        for successor_body in successors:
            successor_v_index.extend(
                body_v_index[body_v_start[successor_body]:body_v_start[successor_body + 1]]
            )
        successor_v_start.append(len(successor_v_index))

        ancestor_body_index.extend(ancestors)
        ancestor_body_start.append(len(ancestor_body_index))

    return {
        "body_v_start": body_v_start,
        "body_v_index": body_v_index,
        "vel_to_body": vel_to_body,
        "vel_to_local_col": vel_to_local_col,
        "vel_s_index": vel_s_index,
        "vel_s_sign": vel_s_sign,
        "subtree_v_start": subtree_v_start,
        "subtree_v_index": subtree_v_index,
        "successor_v_start": successor_v_start,
        "successor_v_index": successor_v_index,
        "ancestor_body_start": ancestor_body_start,
        "ancestor_body_index": ancestor_body_index,
        "n_int": n_int,
        "body_vint_index": body_vint_index,
        "int_true_vel": int_true_vel,
        "int_alpha": int_alpha,
        "int_s_index": int_s_index,
        "int_s_sign": int_s_sign,
    }

def gen_idsva_so_xdown_plucker_inverse(self, mode):
    """Emit the magic-number Plücker-block spatial-transform inverse Xdown = inv(Xup).

    Single source of truth for the byte-identical Xdown inverse that appears in
    both the fixed body-frame inner (GPU block-parallel) and the (kept,
    non-production) floating-reference inner (single-thread serial). The two call
    sites differ only in their loop scaffolding/index variable, so this helper
    reproduces each site's exact current emission verbatim. The world-frame Step 2
    Xdown uses the structurally-different explicit E^T/-E^T·B·E^T form and is NOT
    routed through here.

    mode="fixed_parallel": block-parallel over XIMAT_SIZE*NUM_BODIES (index `i`),
        followed by a __syncthreads(). Used by gen_idsva_so_body_frame_inner.
    mode="floating_serial": serial for-loops with an explicit zero-init pass
        (index `flat`). Used by gen_idsva_so_body_frame_floating_reference_inner.
    """
    if mode == "fixed_parallel":
        self.gen_add_parallel_loop('i','XIMAT_SIZE*NUM_BODIES')
        self.gen_add_code_line('size_t idx = i % XIMAT_SIZE;')
        self.gen_add_code_line('size_t sub_idx = idx % 18;')
        # TODO fix magic numbers
        self.gen_add_code_line('if (idx % 18 == 1 || idx % 18 == 4 || idx % 18 == 8 || idx % 18 == 11) {', True)
        self.gen_add_code_line(f'Xdown[i] = Xup[i+5];')
        self.gen_add_code_line(f'Xdown[i+5] = Xup[i];')
        self.gen_add_end_control_flow()
        self.gen_add_code_line('else if (idx % 18 == 2 || idx % 18 == 5) {', True)
        self.gen_add_code_line(f'Xdown[i] = Xup[i+10];')
        self.gen_add_code_line('Xdown[i+10] = Xup[i];')
        self.gen_add_end_control_flow()
        self.gen_add_code_line('else if (sub_idx != 6 && sub_idx != 9 && sub_idx != 13 && sub_idx != 16 &&')
        self.gen_add_code_line('            sub_idx != 12 && sub_idx != 15)', True)
        self.gen_add_code_line(f'Xdown[i] = Xup[i];')
        self.gen_add_end_control_flow()
        self.gen_add_sync()
    elif mode == "floating_serial":
        self.gen_add_code_line("for (int flat = 0; flat < 36*NUM_BODIES; ++flat) Xdown[flat] = static_cast<T>(0);")
        self.gen_add_code_line("for (int flat = 0; flat < 36*NUM_BODIES; ++flat) {", True)
        self.gen_add_code_line("int idx = flat % 36;")
        self.gen_add_code_line("int sub_idx = idx % 18;")
        self.gen_add_code_line("if (idx % 18 == 1 || idx % 18 == 4 || idx % 18 == 8 || idx % 18 == 11) {", True)
        self.gen_add_code_line("Xdown[flat] = Xup[flat + 5];")
        self.gen_add_code_line("Xdown[flat + 5] = Xup[flat];")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("else if (idx % 18 == 2 || idx % 18 == 5) {", True)
        self.gen_add_code_line("Xdown[flat] = Xup[flat + 10];")
        self.gen_add_code_line("Xdown[flat + 10] = Xup[flat];")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("else if (sub_idx != 6 && sub_idx != 9 && sub_idx != 13 && sub_idx != 16 && sub_idx != 12 && sub_idx != 15) {", True)
        self.gen_add_code_line("Xdown[flat] = Xup[flat];")
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
    else:
        raise ValueError(f"unknown gen_idsva_so_xdown_plucker_inverse mode: {mode}")

def gen_idsva_so_reference_order_rt_rp_assembly(self, S, psid, psidd, psid_Sd, di, ai, inline_row_col):
    """Emit the rt1..rt9 / rp1..rp6 outer-product + crm/dot assembly shared by the
    reference-order output repair (block-parallel branched-fixed path) and the
    floating-reference inner (single-thread). Both compute the identical algebra
    over a (d, a) coordinate pair; they differ only in the subspace buffer names
    (S vs S_vel, psid vs psid_vel, ...) and the loop index variables (jid/ancestor_j
    vs dd/cc). The emitter reproduces each call site's exact current bytes.

    S/psid/psidd/psid_Sd: subspace buffer base names for this path.
    di/ai: the d-coordinate / ancestor-coordinate index expressions.
    inline_row_col: True emits `int row = ...; int col = ...;` on one line (floating
        path); False emits them on two lines (branched-fixed path).
    """
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) {", True)
    if inline_row_col:
        self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    else:
        self.gen_add_code_line("int row = idx % 6;")
        self.gen_add_code_line("int col = idx / 6;")
    self.gen_add_code_line(f"rt1[idx] = {S}[{di}*6 + row] * {psid}[{ai}*6 + col];")
    self.gen_add_code_line(f"rt2[idx] = {S}[{di}*6 + row] * {S}[{ai}*6 + col];")
    self.gen_add_code_line(f"rt3[idx] = {psid}[{di}*6 + row] * {psid}[{ai}*6 + col];")
    self.gen_add_code_line(f"rt4[idx] = {S}[{di}*6 + row] * {psidd}[{ai}*6 + col];")
    self.gen_add_code_line(f"rt5[idx] = {S}[{di}*6 + row] * {psid_Sd}[{ai}*6 + col];")
    self.gen_add_code_line(f"rt6[idx] = {S}[{ai}*6 + row] * {psid}[{di}*6 + col];")
    self.gen_add_code_line(f"rt7[idx] = {S}[{ai}*6 + row] * {psidd}[{di}*6 + col];")
    self.gen_add_code_line(f"rt8[idx] = {S}[{ai}*6 + row] * {S}[{di}*6 + col];")
    self.gen_add_code_line(f"rt9[idx] = {S}[{ai}*6 + row] * {psid_Sd}[{di}*6 + col];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) {", True)
    self.gen_add_code_line(f"rp1[row] = crm_mul<T>(row, &{psid}[{ai}*6], &{S}[{di}*6]);")
    self.gen_add_code_line(f"rp2[row] = crm_mul<T>(row, &{psidd}[{ai}*6], &{S}[{di}*6]);")
    self.gen_add_code_line(f"rp3[row] = crm_mul<T>(row, &{S}[{ai}*6], &{S}[{di}*6]);")
    self.gen_add_code_line(f"rp4[row] = crm_mul<T>(row, &{psid_Sd}[{ai}*6], &{S}[{di}*6]) - static_cast<T>(2) * crm_mul<T>(row, &{psid}[{di}*6], &{S}[{ai}*6]);")
    self.gen_add_code_line(f"rp5[row] = crm_mul<T>(row, &{S}[{di}*6], &{S}[{ai}*6]);")
    self.gen_add_code_line(f"rp6[row] = dot_prod<T, 6, 1, 1>(&IC_S[{di}*6], &crm_S[{ai}*36 + row*6]) + dot_prod<T, 6, 1, 1>(&{S}[{ai}*6], &crf_S_IC[{di}*36 + row*6]);")
    self.gen_add_end_control_flow()

def gen_idsva_so_body_frame_inner_temp_mem_size(self):
    """
    Returns the total size of the temporary memory required for the
    second order idsva inner function.

    Returns:
        int: The total size of the temporary memory required for the
        second order idsva inner function.
    """
    NV = self.robot.get_num_vel()
    num_bodies = self.robot.get_num_bodies()
    if self.robot.floating_base:
        # 7 (not 8) 36*NB body matrices: icrf_f is a kernel-local 36-float array per
        # body, not a shared buffer.
        body_mat_count = 7 * 36 * num_bodies
        body_vec_count = 7 * 6 * num_bodies
        # B4 FIX: for a mimic robot the floating body-frame inner runs the velocity-indexed
        # sweep in UNIQUE per-column INTERNAL coordinates (n_int = total column count > NV)
        # and assembles into a 4*n_int^3 internal slab before alpha-folding to the reduced
        # 4*NV^3 caller output. So the vel-indexed bands are sized by n_int and the internal
        # slab is appended. Non-mimic: n_int == NV and no slab, so the size is unchanged.
        if self.robot_has_mimic_joints():
            n_int = _idsva_so_floating_velocity_metadata(self.robot)["n_int"]
            vel_n = n_int
            internal_slab = 4 * n_int ** 3
        else:
            vel_n = NV
            internal_slab = 0
        vel_vec_count = 12 * 6 * vel_n
        vel_mat_count = 9 * 36 * vel_n
        base_count = body_mat_count + body_vec_count + vel_vec_count + vel_mat_count + 6 + 64 + internal_slab
        # Gravity-shim shared portion only — d2X / d2a / d2f spill to `d_workspace`
        # (see `gen_floating_gravity_d2tau_dq_spill_count`).
        return int(base_count + gen_floating_gravity_d2tau_dq_shared_count(self))
    jids_a, ancestors = self.robot.get_jid_ancestor_ids(include_joint=True)
    # Fixed-base mimic: the inner runs the whole per-body sweep in unique-per-body
    # INTERNAL coordinates (n_int = NUM_BODIES; internal slot == body id) so that
    # mimic siblings sharing a project v-slot get distinct internal slots (mirrors
    # RBDReference.idsva_so_body_frame's has_mimic path). The vel-indexed scratch
    # bands (the 30*NV vectors) are sized by NB instead of NV, and a 4*NB^3 internal
    # output slab is appended on top; the inner assembles into that slab then folds
    # to the reduced 4*NV^3 public output. Non-mimic robots keep the NV sizing
    # byte-identical (NB == NV for fixed non-mimic single-DoF chains).
    if self.robot_has_mimic_joints():
        NB = num_bodies
        base = 36 * NB * 10 + 30 * NB + 6 + len(jids_a) * 36
        internal_slab = 4 * NB ** 3
        return int(base + internal_slab)
    return int(36 * NV * 10 + 30 * NV + 6 + len(jids_a)*36)

def _floating_gravity_lie_metadata(robot):
    """Per-velocity Lie-generator metadata for the floating-base gravity Hessian.

    For each velocity coordinate `vi` belonging to body `jid`:
      - `body[vi] = jid`
      - `s_index[vi]` is the row in S_local that the unit-axis column points along.
      - `s_sign[vi]` is the unit-axis sign in S_local.
      - `lie_sign_flip[vi] = 1` iff the right-Lie generator `B` such that
        `dXmat/dq = Xmat @ B` is `-crm(S_local)` rather than `+crm(S_local)` in this
        codebase's Xmat convention.

    Empirically (from `_spatial_xmat_derivative_func` for non-root joints), the GRiD
    codebase's `Xmat(q)` is defined such that `dXmat/dq = -Xmat @ crm(S_local)` for
    all non-root single-DoF joints. For the floating-base root, the Python helper at
    `_floating_gravity_d2tau_dq_lie_direct` uses `B = +crm` for rotation columns and
    `B = -crm` for translation columns (Featherstone xlt sign convention).

    Combined rule for `lie_sign_flip`:
      - Root rotation columns (local_col >= 3 for the floating-base joint): 0.
      - Root translation columns (local_col < 3): 1.
      - All non-root joints: 1.
    """
    num_bodies = robot.get_num_bodies()
    num_vel = robot.get_num_vel()
    body = [0] * num_vel
    s_index = [0] * num_vel
    s_sign = [1] * num_vel
    lie_sign_flip = [0] * num_vel

    # B4 FIX: internal-slot (UNIQUE per-column) variants of the same per-velocity Lie
    # tables, mirroring _idsva_so_floating_velocity_metadata. For a mimic robot the
    # gravity shim runs in internal coords so a mimic joint and its target get distinct
    # slots (the reduced `body[vel_index]` would otherwise overwrite the shared slot,
    # dropping the target's gravity contribution). Each internal slot also records its
    # true reduced slot + mimic alpha for the shared fold. Non-mimic: internal == reduced.
    int_body = []
    int_s_index = []
    int_s_sign = []
    int_lie_sign_flip = []
    int_true = []
    int_alpha = []

    for body_id in range(num_bodies):
        v_inds = _idsva_so_as_index_list(robot.get_joint_index_v(body_id))
        S = robot.get_S_by_id(body_id)
        if len(S.shape) == 1:
            S_cols = [S]
        else:
            S_cols = [S[:, col] for col in range(S.shape[1])]
        joint = robot.get_joint_by_id(body_id)
        is_mimic_body = getattr(joint, "is_mimic", False)
        alpha = float(joint.get_mimic_multiplier()) if is_mimic_body else 1.0
        for local_col, vel_index in enumerate(v_inds):
            row_idx, sign = _idsva_so_unit_axis(S_cols[local_col])
            flip = 1 if (body_id != 0 or local_col < 3) else 0
            # Reduced (legacy) tables: guard on is_mimic so the mimic does not overwrite
            # its target's slot (matches the velocity-metadata convention).
            if not is_mimic_body:
                body[vel_index] = body_id
                s_index[vel_index] = row_idx
                s_sign[vel_index] = sign
                # Flip sign for non-root joints (sympy Xmat convention) and for root
                # translation columns (Featherstone xlt sign convention).
                lie_sign_flip[vel_index] = flip
            # Internal tables: one UNIQUE slot per column.
            int_body.append(body_id)
            int_s_index.append(row_idx)
            int_s_sign.append(sign)
            int_lie_sign_flip.append(flip)
            int_true.append(vel_index)
            int_alpha.append(alpha)

    return {
        "body": body,
        "s_index": s_index,
        "s_sign": s_sign,
        # Keep the field name `is_root_translation` for backwards-compat with
        # the existing emission code; it now means "needs Lie-sign flip".
        "is_root_translation": lie_sign_flip,
        # Internal-slot (UNIQUE per-column) variants for the mimic gravity-shim path.
        "n_int": len(int_body),
        "int_body": int_body,
        "int_s_index": int_s_index,
        "int_s_sign": int_s_sign,
        "int_is_root_translation": int_lie_sign_flip,
        "int_true": int_true,
        "int_alpha": int_alpha,
    }

def _gravity_shim_use_full_spill(self):
    """Decide whether to spill the gravity-shim's shared portion to d_workspace.

    Triggered by robot size: when leaving the shared portion in `s_temp` would push
    `idsva_so` total shared bytes over the target, we move dX/a/da/f/df/scratch to
    `d_workspace` (in addition to the d2X/d2a/d2f that always spill). Saves 50-60 KB
    for large floating-base robots. The 4*36 scratch buffers become kernel-local arrays.
    """
    if not self.robot.floating_base:
        return False
    return bool(getattr(self, "idsva_so_body_frame_grav_full_spill", False))


def gen_floating_gravity_d2tau_dq_spill_count(self):
    """Floats of the gravity-Hessian helper that live in the global `d_workspace`
    spill region (per timestep). Always includes the three O(NV²·NB) tensors
    {d2X, d2a, d2f}; when `idsva_so_body_frame_grav_full_spill` is set (large robots), also
    includes the previously-shared {dX, a, da, f, df} arrays.
    """
    NV = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    # B4 FIX: the mimic gravity shim runs its q-perturbation (vi/vj) axes in UNIQUE
    # internal slots (n_int >= NV) so a mimic + its target get distinct columns; the
    # d2X/d2a/d2f/dX/da/df tensors are therefore sized by n_int on those axes. Non-mimic:
    # n_int == NV, byte-identical sizing.
    grav_n = _floating_gravity_lie_metadata(self.robot)["n_int"] if self.robot_has_mimic_joints() else NV
    d2X_count = 36 * grav_n * grav_n
    d2a_count = 6 * grav_n * grav_n * NB
    d2f_count = 6 * grav_n * grav_n * NB
    total = d2X_count + d2a_count + d2f_count
    if _gravity_shim_use_full_spill(self):
        # When fully spilled, dX/a/da/f/df move to d_workspace too. The 4*36
        # scratch buffers become kernel-local arrays (not in workspace).
        NB_ = NB
        total += 36 * grav_n + 6 * NB_ + 6 * grav_n * NB_ + 6 * NB_ + 6 * grav_n * NB_
    return int(total)


def gen_floating_gravity_d2tau_dq_shared_count(self):
    """Floats of the gravity-Hessian helper that stay in shared memory.

    Default: dX (sparse-but-stored-dense), a/da, f/df, and the 4*36 scratch
    buffers. When `idsva_so_body_frame_grav_full_spill` is set, returns 0 (everything moves
    to d_workspace except the 4*36 scratch which becomes kernel-local).
    """
    if _gravity_shim_use_full_spill(self):
        return 0
    NV = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    # B4 FIX: mimic shim sizes the vi-axis (dX/da/df) by internal slot count n_int.
    grav_n = _floating_gravity_lie_metadata(self.robot)["n_int"] if self.robot_has_mimic_joints() else NV
    dX_count = 36 * grav_n
    a_count = 6 * NB
    da_count = 6 * grav_n * NB
    f_count = a_count
    df_count = da_count
    scratch_count = 4 * 36
    return int(dX_count + a_count + da_count + f_count + df_count + scratch_count)


def gen_floating_gravity_d2tau_dq_temp_mem_size(self):
    """Total floats the gravity-Hessian helper needs (shared + spill).

    The kernel emitter splits this into a shared-memory portion (the smaller arrays
    plus scratch) and a `d_workspace` spill portion (d2X / d2a / d2f) so larger
    floating-base robots still fit. See `gen_floating_gravity_d2tau_dq_shared_count`
    and `gen_floating_gravity_d2tau_dq_spill_count`.
    """
    return int(gen_floating_gravity_d2tau_dq_shared_count(self)
               + gen_floating_gravity_d2tau_dq_spill_count(self))

def gen_floating_gravity_d2tau_dq_lie_inline(self):
    """Emit (inline) the floating-base gravity-Hessian addition into `d2tau_dq2`.

    Translates the Python helper `_floating_gravity_d2tau_dq_lie_direct` (see
    `RBDReference/RBDReference.py`) into CUDA emission for use inside
    `gen_idsva_so_body_frame_floating_reference_inner`.

    Preconditions (set up by the caller):
      - `Xup[NB*36]` contains cumulative world-frame joint transforms.
      - `I[NB*36]` is body-frame inertia (the constant inertia tensor for each body).
      - `s_q` holds joint position parameters.
      - `d2tau_dq2[NV*NV*NV]` is the output buffer; we add to it here.

    Memory allocated from the caller's scratch (carved by the caller). Phase A
    assumes the carve fits; Phase D moves the large tensors to `d_workspace`.

    Algorithm (one-pass body-frame propagation):
      1. dX[vi] = X[jid(vi)] @ B_vi   (Lie generator, with translation sign flip
         for the floating-base root).
      2. d2X[vi][vj] = X[jid] @ B_vj @ B_vi when vi, vj share a body; else 0.
      3. Forward: a[jid] = X[jid] @ a[parent] with `a[parent_of_root] = -gravity`;
         carry first/second derivatives via the standard chain rule.
      4. f = I @ a per body (all bodies, batched conceptually).
      5. Backward: project onto each joint's S, propagate f-derivatives to parent.
      6. Output: d2tau_dq2[v_index_of_jid_dof, :, :] += S^T @ d2f[jid, :, :].

    See `RBDReference.py` line ~2371 for the body-major Python reference this mirrors.
    """
    NV = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    lie_meta = _floating_gravity_lie_metadata(self.robot)
    parent_ids = [self.robot.get_parent_id(b) for b in range(NB)]
    # B4 FIX: for a mimic robot this shim runs in UNIQUE per-column INTERNAL slots
    # (grav_n = n_int > NV) and ADDS into the caller's INTERNAL d2tau_dq slab (SO_N_INT
    # stride); the caller's single alpha-fold then reduces sweep + gravity together. The
    # perturbation axes (vi/vj) and the output q-axis use grav_n; the projection writes
    # internal slot finds_c (from the caller's internal body_v_index) at SO_N_INT stride.
    # Non-mimic: grav_n == NV, GN == "NUM_VEL", GO == "NUM_VEL", byte-identical emission.
    grav_is_mimic = self.robot_has_mimic_joints()
    # GN spans the q-perturbation axes AND the internal d2tau_dq output stride. For mimic
    # n_int == SO_N_INT (the caller's internal-slab stride), so a single token suffices.
    GN = "GRAV_N_INT" if grav_is_mimic else "NUM_VEL"

    # MIMIC: every C-level `NUM_VEL` in this shim is a q-perturbation / output q-axis that
    # must become the internal-slot count. Route all C emission through a wrapper that
    # rewrites the token to GRAV_N_INT so the (large, single-threaded) body below stays
    # verbatim. For non-mimic the rewrite is a no-op (gn == "NUM_VEL"), so the emitted code
    # is BYTE-IDENTICAL to before. The wrapper is installed AFTER the GRAV_N_INT constexpr
    # is emitted (so that line keeps its literal value) and restored at function end.
    _orig_gen_add_code_line = self.gen_add_code_line
    _orig_gen_add_code_lines = self.gen_add_code_lines

    # Wrap the entire emission in a single-thread block so the helper is self-contained
    # and safe to call regardless of whether the caller is already in a thread-zero scope.
    self.gen_add_code_line("if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {", True)
    if grav_is_mimic:
        self.gen_add_code_line(f"constexpr int GRAV_N_INT = {lie_meta['n_int']};")
        def _gl_line(line, *a, **k):
            if isinstance(line, str):
                line = line.replace("NUM_VEL", "GRAV_N_INT")
            return _orig_gen_add_code_line(line, *a, **k)
        def _gl_lines(lines, *a, **k):
            lines = [(ln.replace("NUM_VEL", "GRAV_N_INT") if isinstance(ln, str) else ln) for ln in lines]
            return _orig_gen_add_code_lines(lines, *a, **k)
        self.gen_add_code_line = _gl_line
        self.gen_add_code_lines = _gl_lines
    # Compute the shared-memory base offset: existing main-sweep temp size MINUS the
    # gravity-shim's shared portion (we want grav_scratch to point at where the helper's
    # shared arrays live, which is right after the main-sweep allocations).
    main_sweep_count = self.gen_idsva_so_body_frame_inner_temp_mem_size() - gen_floating_gravity_d2tau_dq_shared_count(self)
    full_spill = _gravity_shim_use_full_spill(self)
    layout_comment = (
        "// Full-spill layout (size-triggered): dX/a/da/f/df spill to d_workspace; 4*36 scratch is kernel-local."
        if full_spill else
        "// Shared portion (dX / a / da / f / df / 4x36 scratch) lives in s_temp; the\n"
        "// three O(NV*NV*NB) tensors (d2X / d2a / d2f) spill to d_workspace which the\n"
        "// kernel emitter points into per-timestep."
    )
    self.gen_add_code_lines([
        "// ===== Gravity-Hessian (Lie-tangent) addition into d2tau_dq2 =====",
        *layout_comment.split("\n"),
        f"static const int grav_lie_body[] = {{ {_idsva_so_int_array(lie_meta['int_body'] if grav_is_mimic else lie_meta['body'])} }};",
        f"static const int grav_lie_s_index[] = {{ {_idsva_so_int_array(lie_meta['int_s_index'] if grav_is_mimic else lie_meta['s_index'])} }};",
        f"static const int grav_lie_s_sign[] = {{ {_idsva_so_int_array(lie_meta['int_s_sign'] if grav_is_mimic else lie_meta['s_sign'])} }};",
        f"static const int grav_lie_is_root_translation[] = {{ {_idsva_so_int_array(lie_meta['int_is_root_translation'] if grav_is_mimic else lie_meta['is_root_translation'])} }};",
        f"static const int grav_lie_parent[] = {{ {_idsva_so_int_array(parent_ids)} }};",
        "",
    ])
    if full_spill:
        # Everything that used to be in s_temp moves into d_temp_spill AFTER d2X/d2a/d2f.
        # The 4*36 scratch buffers become kernel-local stack arrays.
        self.gen_add_code_lines([
            "// Global-memory spill carve (large per-timestep tensors + the previously-shared dX/a/da/f/df).",
            "T *grav_d2X     = d_workspace;",
            "T *grav_d2a     = grav_d2X     + 36*NUM_VEL*NUM_VEL;",
            "T *grav_d2f     = grav_d2a     + 6*NUM_VEL*NUM_VEL*NUM_BODIES;",
            "T *grav_dX      = grav_d2f     + 6*NUM_VEL*NUM_VEL*NUM_BODIES;",
            "T *grav_a       = grav_dX      + 36*NUM_VEL;",
            "T *grav_da      = grav_a       + 6*NUM_BODIES;",
            "T *grav_f       = grav_da      + 6*NUM_VEL*NUM_BODIES;",
            "T *grav_df      = grav_f       + 6*NUM_BODIES;",
            "T grav_invX_buf[36];",
            "T grav_tmpA_buf[36];",
            "T grav_tmpB_buf[36];",
            "T grav_tmpC_buf[36];",
            "T *grav_invX    = grav_invX_buf;",
            "T *grav_tmpA    = grav_tmpA_buf;",
            "T *grav_tmpB    = grav_tmpB_buf;",
            "T *grav_tmpC    = grav_tmpC_buf;",
        ])
    else:
        self.gen_add_code_lines([
            "// Shared-memory carve (small arrays).",
            f"T *grav_scratch = s_temp + {main_sweep_count};",
            "T *grav_dX      = grav_scratch;",
            "T *grav_a       = grav_dX      + 36*NUM_VEL;",
            "T *grav_da      = grav_a       + 6*NUM_BODIES;",
            "T *grav_f       = grav_da      + 6*NUM_VEL*NUM_BODIES;",
            "T *grav_df      = grav_f       + 6*NUM_BODIES;",
            "T *grav_invX    = grav_df      + 6*NUM_VEL*NUM_BODIES;",
            "T *grav_tmpA    = grav_invX    + 36;",
            "T *grav_tmpB    = grav_tmpA    + 36;",
            "T *grav_tmpC    = grav_tmpB    + 36;",
            "// Global-memory spill carve (large per-timestep tensors).",
            "T *grav_d2X     = d_workspace;",
            "T *grav_d2a     = grav_d2X     + 36*NUM_VEL*NUM_VEL;",
            "T *grav_d2f     = grav_d2a     + 6*NUM_VEL*NUM_VEL*NUM_BODIES;",
        ])
    self.gen_add_code_lines([
        "",
        "// gravity_vec mirrors Python's `gravity_vec[5] = -GRAVITY`. The CUDA `gravity`",
        "// parameter is the SIGNED gravitational acceleration (= -9.81, the unified GRiD",
        "// convention matching RBDReference's GRAVITY). So `gravity_vec[5] = -GRAVITY = +9.81`",
        "// equals CUDA's `gravity_vec[5] = -gravity`.",
        "T grav_gravity_vec[6] = { static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),",
        "                          static_cast<T>(0), static_cast<T>(0), -gravity };",
        "",
        "// Zero scratch (single-thread; could be parallelised across threadIdx for speed).",
        "for (int idx = 0; idx < 36*NUM_VEL; ++idx) grav_dX[idx] = static_cast<T>(0);",
        "for (int idx = 0; idx < 36*NUM_VEL*NUM_VEL; ++idx) grav_d2X[idx] = static_cast<T>(0);",
        "for (int idx = 0; idx < 6*NUM_BODIES; ++idx) { grav_a[idx] = static_cast<T>(0); grav_f[idx] = static_cast<T>(0); }",
        "for (int idx = 0; idx < 6*NUM_VEL*NUM_BODIES; ++idx) { grav_da[idx] = static_cast<T>(0); grav_df[idx] = static_cast<T>(0); }",
        "for (int idx = 0; idx < 6*NUM_VEL*NUM_VEL*NUM_BODIES; ++idx) { grav_d2a[idx] = static_cast<T>(0); grav_d2f[idx] = static_cast<T>(0); }",
        "",
        "// ---- (1) Build Lie generators B[vi] = +/- crm(unit_axis_si).",
        "//        Root (jid==0):  dX[vi] = X_local[0] @ B[vi]   (Featherstone xlt convention,",
        "//                        sign flip on translation columns baked into B).",
        "//        Non-root joints: dX[vi] = B[vi] @ X_local[jid] (codebase's sympy convention:",
        "//                         dXmat/dq = -crm(S_local) @ Xmat, with B = -crm(S_local)).",
        "for (int vi = 0; vi < NUM_VEL; ++vi) {", True,
        "int jid = grav_lie_body[vi];",
        "int s_row = grav_lie_s_index[vi];",
        "T sign = static_cast<T>(grav_lie_s_sign[vi]);",
        "if (grav_lie_is_root_translation[vi]) sign = -sign;",
        "T B[36];",
        "T e_vec[6] = { static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),",
        "               static_cast<T>(0), static_cast<T>(0), static_cast<T>(0) };",
        "e_vec[s_row] = sign;",
        "for (int idx = 0; idx < 36; ++idx) B[idx] = crm<T>(idx, e_vec);",
        "for (int idx = 0; idx < 36; ++idx) {", True,
        "int row = idx % 6; int col = idx / 6;",
        "T acc = static_cast<T>(0);",
        "if (jid == 0) {",
        "    for (int kk = 0; kk < 6; ++kk) acc += s_XImats[jid*36 + row + 6*kk] * B[kk + 6*col];",
        "} else {",
        "    for (int kk = 0; kk < 6; ++kk) acc += B[row + 6*kk] * s_XImats[jid*36 + kk + 6*col];",
        "}",
        "grav_dX[vi*36 + idx] = acc;",
        "",  # close inner idx loop
        ])
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()

    self.gen_add_code_lines([
        "",
        "// ---- (2) Build d2X[vi][vj] when vi, vj share a body.",
        "//        Root (jid_i==0):  d2X[vi,vj] = X_local[0] @ Bj @ Bi   (right multiplication).",
        "//        Non-root:         d2X[vi,vj] = Bj @ Bi @ X_local[jid] (left multiplication;",
        "//                          for single-DoF non-root vi==vj this is B^2 @ X_local).",
        "for (int vi = 0; vi < NUM_VEL; ++vi) {", True,
        "int jid_i = grav_lie_body[vi];",
        "for (int vj = 0; vj < NUM_VEL; ++vj) {", True,
        "if (grav_lie_body[vj] != jid_i) continue;  // d2X is zero for cross-body pairs.",
        "T sign_i = static_cast<T>(grav_lie_s_sign[vi]);",
        "if (grav_lie_is_root_translation[vi]) sign_i = -sign_i;",
        "T sign_j = static_cast<T>(grav_lie_s_sign[vj]);",
        "if (grav_lie_is_root_translation[vj]) sign_j = -sign_j;",
        "T Bi[36], Bj[36];",
        "T ei[6] = { static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),",
        "            static_cast<T>(0), static_cast<T>(0), static_cast<T>(0) };",
        "T ej[6] = { static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),",
        "            static_cast<T>(0), static_cast<T>(0), static_cast<T>(0) };",
        "ei[grav_lie_s_index[vi]] = sign_i;",
        "ej[grav_lie_s_index[vj]] = sign_j;",
        "for (int idx = 0; idx < 36; ++idx) { Bi[idx] = crm<T>(idx, ei); Bj[idx] = crm<T>(idx, ej); }",
        "// tmpA = Bj @ Bi  (6x6 @ 6x6).",
        "for (int idx = 0; idx < 36; ++idx) {", True,
        "int row = idx % 6; int col = idx / 6;",
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += Bj[row + 6*kk] * Bi[kk + 6*col];",
        "grav_tmpA[idx] = acc;",
        ])
    self.gen_add_end_control_flow()
    self.gen_add_code_line("// Multiply tmpA by X_local on the correct side (right for root, left for non-root).")
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) {", True)
    self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("if (jid_i == 0) {")
    self.gen_add_code_line("    for (int kk = 0; kk < 6; ++kk) acc += s_XImats[jid_i*36 + row + 6*kk] * grav_tmpA[kk + 6*col];")
    self.gen_add_code_line("} else {")
    self.gen_add_code_line("    for (int kk = 0; kk < 6; ++kk) acc += grav_tmpA[row + 6*kk] * s_XImats[jid_i*36 + kk + 6*col];")
    self.gen_add_code_line("}")
    self.gen_add_code_line("grav_d2X[(vi*NUM_VEL + vj)*36 + idx] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()

    self.gen_add_code_lines([
        "",
        "// ---- (3) Forward sweep: propagate a, da, d2a through the tree (body frame).",
        "//        Root (parent < 0): a[0] = inv(X_local[0]) @ gravity_vec. Since X_local[0]",
        "//        equals the cumulative Xup[0] (no parent in the chain), Xdown[0] is its inverse.",
        "//        Non-root: a[jid] = X_local[jid] @ a[parent]; carry first/second derivatives.",
        "for (int jid = 0; jid < NUM_BODIES; ++jid) {", True,
        "int parent = grav_lie_parent[jid];",
        "if (parent < 0) {", True,
        "    // ---- Root: a[0] = inv(X[0]) @ gravity_vec.",
        "    for (int row = 0; row < 6; ++row) {", True,
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += Xdown[jid*36 + row + 6*kk] * grav_gravity_vec[kk];",
        "grav_a[jid*6 + row] = acc;",
        ])
    self.gen_add_end_control_flow()  # close root row loop

    # da[0, vi] = -inv_X @ dX[vi] @ a[0], only for vi whose body is root (else stays 0).
    self.gen_add_code_lines([
        "    // da[0, vi] = -inv_X @ dX[vi] @ a[0] for root coords; zero otherwise.",
        "    for (int vi = 0; vi < NUM_VEL; ++vi) {", True,
        "if (grav_lie_body[vi] != jid) continue;",
        "// tmp = dX[vi] @ a[0]   (6-vector).",
        "T tmp_v[6];",
        "for (int row = 0; row < 6; ++row) {", True,
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += grav_dX[vi*36 + row + 6*kk] * grav_a[jid*6 + kk];",
        "tmp_v[row] = acc;",
        ])
    self.gen_add_end_control_flow()  # close tmp_v row loop
    self.gen_add_code_lines([
        "// grav_da[jid, vi] = -inv_X @ tmp_v = -Xdown[jid] @ tmp_v.",
        "for (int row = 0; row < 6; ++row) {", True,
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += Xdown[jid*36 + row + 6*kk] * tmp_v[kk];",
        "grav_da[(jid*NUM_VEL + vi)*6 + row] = -acc;",
        ])
    self.gen_add_end_control_flow()  # close da row loop
    self.gen_add_end_control_flow()  # close vi loop

    # d2a[0, vi, vj] for root coords only.
    self.gen_add_code_lines([
        "    // d2a[0, vi, vj] = -inv_X @ (dX[vi]@da[0,vj] + dX[vj]@da[0,vi] + d2X[vi,vj]@a[0]).",
        "    // Nonzero only when both vi and vj are coords of the root body.",
        "    for (int vi = 0; vi < NUM_VEL; ++vi) {", True,
        "if (grav_lie_body[vi] != jid) continue;",
        "for (int vj = 0; vj < NUM_VEL; ++vj) {", True,
        "if (grav_lie_body[vj] != jid) continue;",
        "T inner_v[6] = { static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),",
        "                 static_cast<T>(0), static_cast<T>(0), static_cast<T>(0) };",
        "// dX[vi] @ da[0, vj].",
        "for (int row = 0; row < 6; ++row) {", True,
        "for (int kk = 0; kk < 6; ++kk) inner_v[row] += grav_dX[vi*36 + row + 6*kk] * grav_da[(jid*NUM_VEL + vj)*6 + kk];",
        ])
    self.gen_add_end_control_flow()  # close row loop term1
    self.gen_add_code_lines([
        "// dX[vj] @ da[0, vi].",
        "for (int row = 0; row < 6; ++row) {", True,
        "for (int kk = 0; kk < 6; ++kk) inner_v[row] += grav_dX[vj*36 + row + 6*kk] * grav_da[(jid*NUM_VEL + vi)*6 + kk];",
        ])
    self.gen_add_end_control_flow()  # close row loop term2
    self.gen_add_code_lines([
        "// d2X[vi, vj] @ a[0].",
        "for (int row = 0; row < 6; ++row) {", True,
        "for (int kk = 0; kk < 6; ++kk) inner_v[row] += grav_d2X[(vi*NUM_VEL + vj)*36 + row + 6*kk] * grav_a[jid*6 + kk];",
        ])
    self.gen_add_end_control_flow()  # close row loop term3
    self.gen_add_code_lines([
        "// grav_d2a[0, vi, vj] = -Xdown[0] @ inner_v.",
        "for (int row = 0; row < 6; ++row) {", True,
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += Xdown[jid*36 + row + 6*kk] * inner_v[kk];",
        "grav_d2a[((jid*NUM_VEL + vi)*NUM_VEL + vj)*6 + row] = -acc;",
        ])
    self.gen_add_end_control_flow()  # close d2a row loop
    self.gen_add_end_control_flow()  # close vj loop
    self.gen_add_end_control_flow()  # close vi loop

    self.gen_add_end_control_flow()  # close parent < 0 branch
    # Non-root branch.
    self.gen_add_code_lines([
        "else {", True,
        "    // ---- Non-root: a[jid] = X_local[jid] @ a[parent].",
        "    for (int row = 0; row < 6; ++row) {", True,
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += s_XImats[jid*36 + row + 6*kk] * grav_a[parent*6 + kk];",
        "grav_a[jid*6 + row] = acc;",
        ])
    self.gen_add_end_control_flow()  # close a row loop
    self.gen_add_code_lines([
        "    // da[jid, vi] = dX[jid, vi] @ a[parent]   (only when vi is a coord of jid)",
        "    //            + X_local[jid] @ da[parent, vi]   (always for any vi that affects parent).",
        "    for (int vi = 0; vi < NUM_VEL; ++vi) {", True,
        "T row_vals[6] = { static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),",
        "                  static_cast<T>(0), static_cast<T>(0), static_cast<T>(0) };",
        "// First term: dX[jid, vi] @ a[parent], only when vi is a coord of jid.",
        "if (grav_lie_body[vi] == jid) {", True,
        "for (int row = 0; row < 6; ++row) {", True,
        "for (int kk = 0; kk < 6; ++kk) row_vals[row] += grav_dX[vi*36 + row + 6*kk] * grav_a[parent*6 + kk];",
        ])
    self.gen_add_end_control_flow()  # row loop
    self.gen_add_end_control_flow()  # if dX nonzero
    self.gen_add_code_lines([
        "// Second term: X_local[jid] @ da[parent, vi].",
        "for (int row = 0; row < 6; ++row) {", True,
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += s_XImats[jid*36 + row + 6*kk] * grav_da[(parent*NUM_VEL + vi)*6 + kk];",
        "row_vals[row] += acc;",
        ])
    self.gen_add_end_control_flow()  # row loop second term
    self.gen_add_code_lines([
        "for (int row = 0; row < 6; ++row) grav_da[(jid*NUM_VEL + vi)*6 + row] = row_vals[row];",
        ])
    self.gen_add_end_control_flow()  # vi loop
    # d2a non-root: 4 terms.
    self.gen_add_code_lines([
        "    // d2a[jid, vi, vj] = d2X[jid, vi, vj] @ a[parent]   (only when both vi, vj coords of jid)",
        "    //               + dX[jid, vi] @ da[parent, vj]   (only when vi is a coord of jid)",
        "    //               + dX[jid, vj] @ da[parent, vi]   (only when vj is a coord of jid)",
        "    //               + X_local[jid] @ d2a[parent, vi, vj]   (always).",
        "    for (int vi = 0; vi < NUM_VEL; ++vi) {", True,
        "for (int vj = 0; vj < NUM_VEL; ++vj) {", True,
        "T row_vals[6] = { static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),",
        "                  static_cast<T>(0), static_cast<T>(0), static_cast<T>(0) };",
        "// Term 1: d2X[jid, vi, vj] @ a[parent].",
        "if (grav_lie_body[vi] == jid && grav_lie_body[vj] == jid) {", True,
        "for (int row = 0; row < 6; ++row) {", True,
        "for (int kk = 0; kk < 6; ++kk) row_vals[row] += grav_d2X[(vi*NUM_VEL + vj)*36 + row + 6*kk] * grav_a[parent*6 + kk];",
        ])
    self.gen_add_end_control_flow()  # row loop
    self.gen_add_end_control_flow()  # if d2X nonzero
    self.gen_add_code_lines([
        "// Term 2: dX[jid, vi] @ da[parent, vj].",
        "if (grav_lie_body[vi] == jid) {", True,
        "for (int row = 0; row < 6; ++row) {", True,
        "for (int kk = 0; kk < 6; ++kk) row_vals[row] += grav_dX[vi*36 + row + 6*kk] * grav_da[(parent*NUM_VEL + vj)*6 + kk];",
        ])
    self.gen_add_end_control_flow()  # row loop
    self.gen_add_end_control_flow()  # if dX[vi] nonzero
    self.gen_add_code_lines([
        "// Term 3: dX[jid, vj] @ da[parent, vi].",
        "if (grav_lie_body[vj] == jid) {", True,
        "for (int row = 0; row < 6; ++row) {", True,
        "for (int kk = 0; kk < 6; ++kk) row_vals[row] += grav_dX[vj*36 + row + 6*kk] * grav_da[(parent*NUM_VEL + vi)*6 + kk];",
        ])
    self.gen_add_end_control_flow()  # row loop
    self.gen_add_end_control_flow()  # if dX[vj] nonzero
    self.gen_add_code_lines([
        "// Term 4: X_local[jid] @ d2a[parent, vi, vj].",
        "for (int row = 0; row < 6; ++row) {", True,
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += s_XImats[jid*36 + row + 6*kk] * grav_d2a[((parent*NUM_VEL + vi)*NUM_VEL + vj)*6 + kk];",
        "row_vals[row] += acc;",
        ])
    self.gen_add_end_control_flow()  # row loop term 4
    self.gen_add_code_lines([
        "for (int row = 0; row < 6; ++row) grav_d2a[((jid*NUM_VEL + vi)*NUM_VEL + vj)*6 + row] = row_vals[row];",
        ])
    self.gen_add_end_control_flow()  # vj loop
    self.gen_add_end_control_flow()  # vi loop
    self.gen_add_end_control_flow()  # else branch
    self.gen_add_end_control_flow()  # jid loop

    # ---- (4) f = I @ a, df = I @ da, d2f = I @ d2a per body. Body-frame inertia
    # `I` is the per-body Imat from the layout (set up by the caller, see line ~324).
    self.gen_add_code_lines([
        "",
        "// ---- (4) f = I @ a (and derivatives) per body, body-frame inertia.",
        "for (int jid = 0; jid < NUM_BODIES; ++jid) {", True,
        "for (int row = 0; row < 6; ++row) {", True,
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += I[jid*36 + row + 6*kk] * grav_a[jid*6 + kk];",
        "grav_f[jid*6 + row] = acc;",
        ])
    self.gen_add_end_control_flow()  # f row loop
    self.gen_add_code_lines([
        "for (int vi = 0; vi < NUM_VEL; ++vi) {", True,
        "for (int row = 0; row < 6; ++row) {", True,
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += I[jid*36 + row + 6*kk] * grav_da[(jid*NUM_VEL + vi)*6 + kk];",
        "grav_df[(jid*NUM_VEL + vi)*6 + row] = acc;",
        ])
    self.gen_add_end_control_flow()  # df row loop
    self.gen_add_end_control_flow()  # vi loop for df
    self.gen_add_code_lines([
        "for (int vi = 0; vi < NUM_VEL; ++vi) {", True,
        "for (int vj = 0; vj < NUM_VEL; ++vj) {", True,
        "for (int row = 0; row < 6; ++row) {", True,
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += I[jid*36 + row + 6*kk] * grav_d2a[((jid*NUM_VEL + vi)*NUM_VEL + vj)*6 + kk];",
        "grav_d2f[((jid*NUM_VEL + vi)*NUM_VEL + vj)*6 + row] = acc;",
        ])
    self.gen_add_end_control_flow()  # d2f row loop
    self.gen_add_end_control_flow()  # vj loop
    self.gen_add_end_control_flow()  # vi loop for d2f
    self.gen_add_end_control_flow()  # jid loop

    self.gen_add_code_lines([
        "",
        "// ---- (5) Backward sweep: project d2f onto each joint's S (body-frame motion subspace),",
        "//        ADD into d2tau_dq2 (caller's main sweep is expected to have run with gravity=0).",
        "//        Then bubble f, df, d2f to the parent via X_local transposes.",
        "for (int jid = NUM_BODIES - 1; jid >= 0; --jid) {", True,
        "// Projection: for each velocity coord finds_c of jid, d2tau_dq2[finds_c, k, l] += S_body[:, c] . d2f[jid, k, l, :].",
        "// Body-frame S has a single nonzero entry per column: row=grav_lie_s_index[finds_c], value=grav_lie_s_sign[finds_c].",
        "for (int pos = body_v_start[jid]; pos < body_v_start[jid + 1]; ++pos) {", True,
        "int finds_c = body_v_index[pos];",
        "int s_row = grav_lie_s_index[finds_c];",
        "T s_sign = static_cast<T>(grav_lie_s_sign[finds_c]);",
        "for (int k = 0; k < NUM_VEL; ++k) {", True,
        "for (int l = 0; l < NUM_VEL; ++l) {", True,
        "T proj = s_sign * grav_d2f[((jid*NUM_VEL + k)*NUM_VEL + l)*6 + s_row];",
        "d2tau_dq2[(finds_c*NUM_VEL + k)*NUM_VEL + l] += proj;",
        ])
    self.gen_add_end_control_flow()  # l loop
    self.gen_add_end_control_flow()  # k loop
    self.gen_add_end_control_flow()  # pos loop (velocity coords of jid)

    # Parent update: skip when root.
    self.gen_add_code_lines([
        "int parent = grav_lie_parent[jid];",
        "if (parent < 0) continue;",
        "// Build X_local transpose once per body (used in all three parent updates).",
        "T Xt[36];",
        "for (int idx = 0; idx < 36; ++idx) {", True,
        "int row = idx % 6; int col = idx / 6;",
        "Xt[idx] = s_XImats[jid*36 + col + 6*row];",
        ])
    self.gen_add_end_control_flow()  # Xt loop
    self.gen_add_code_lines([
        "// f[parent] += Xt @ f[jid].",
        "for (int row = 0; row < 6; ++row) {", True,
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += Xt[row + 6*kk] * grav_f[jid*6 + kk];",
        "grav_f[parent*6 + row] += acc;",
        ])
    self.gen_add_end_control_flow()  # f update row loop

    # df parent update.
    self.gen_add_code_lines([
        "// df[parent, vi] += dXt[vi] @ f[jid]   (only when vi is a coord of jid)",
        "//                + Xt @ df[jid, vi]   (always).",
        "for (int vi = 0; vi < NUM_VEL; ++vi) {", True,
        "T row_vals[6] = { static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),",
        "                  static_cast<T>(0), static_cast<T>(0), static_cast<T>(0) };",
        "if (grav_lie_body[vi] == jid) {", True,
        "// dXt[vi] uses dX[vi].T, i.e., swap row<->col indices.",
        "for (int row = 0; row < 6; ++row) {", True,
        "for (int kk = 0; kk < 6; ++kk) row_vals[row] += grav_dX[vi*36 + kk + 6*row] * grav_f[jid*6 + kk];",
        ])
    self.gen_add_end_control_flow()  # dXt row loop
    self.gen_add_end_control_flow()  # if dX nonzero
    self.gen_add_code_lines([
        "for (int row = 0; row < 6; ++row) {", True,
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += Xt[row + 6*kk] * grav_df[(jid*NUM_VEL + vi)*6 + kk];",
        "row_vals[row] += acc;",
        ])
    self.gen_add_end_control_flow()  # Xt @ df row loop
    self.gen_add_code_lines([
        "for (int row = 0; row < 6; ++row) grav_df[(parent*NUM_VEL + vi)*6 + row] += row_vals[row];",
        ])
    self.gen_add_end_control_flow()  # vi loop for df

    # d2f parent update.
    self.gen_add_code_lines([
        "// d2f[parent, vi, vj] += d2Xt[vi,vj]@f[jid] + dXt[vi]@df[jid,vj] + dXt[vj]@df[jid,vi] + Xt@d2f[jid,vi,vj].",
        "for (int vi = 0; vi < NUM_VEL; ++vi) {", True,
        "for (int vj = 0; vj < NUM_VEL; ++vj) {", True,
        "T row_vals[6] = { static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),",
        "                  static_cast<T>(0), static_cast<T>(0), static_cast<T>(0) };",
        "// Term 1: d2Xt[vi, vj] @ f[jid] — nonzero only when both vi, vj are coords of jid.",
        "if (grav_lie_body[vi] == jid && grav_lie_body[vj] == jid) {", True,
        "for (int row = 0; row < 6; ++row) {", True,
        "for (int kk = 0; kk < 6; ++kk) row_vals[row] += grav_d2X[(vi*NUM_VEL + vj)*36 + kk + 6*row] * grav_f[jid*6 + kk];",
        ])
    self.gen_add_end_control_flow()  # term 1 row loop
    self.gen_add_end_control_flow()  # if d2X nonzero
    self.gen_add_code_lines([
        "// Term 2: dXt[vi] @ df[jid, vj].",
        "if (grav_lie_body[vi] == jid) {", True,
        "for (int row = 0; row < 6; ++row) {", True,
        "for (int kk = 0; kk < 6; ++kk) row_vals[row] += grav_dX[vi*36 + kk + 6*row] * grav_df[(jid*NUM_VEL + vj)*6 + kk];",
        ])
    self.gen_add_end_control_flow()  # term 2 row loop
    self.gen_add_end_control_flow()  # if dX[vi] nonzero
    self.gen_add_code_lines([
        "// Term 3: dXt[vj] @ df[jid, vi].",
        "if (grav_lie_body[vj] == jid) {", True,
        "for (int row = 0; row < 6; ++row) {", True,
        "for (int kk = 0; kk < 6; ++kk) row_vals[row] += grav_dX[vj*36 + kk + 6*row] * grav_df[(jid*NUM_VEL + vi)*6 + kk];",
        ])
    self.gen_add_end_control_flow()  # term 3 row loop
    self.gen_add_end_control_flow()  # if dX[vj] nonzero
    self.gen_add_code_lines([
        "// Term 4: Xt @ d2f[jid, vi, vj].",
        "for (int row = 0; row < 6; ++row) {", True,
        "T acc = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) acc += Xt[row + 6*kk] * grav_d2f[((jid*NUM_VEL + vi)*NUM_VEL + vj)*6 + kk];",
        "row_vals[row] += acc;",
        ])
    self.gen_add_end_control_flow()  # term 4 row loop
    self.gen_add_code_lines([
        "for (int row = 0; row < 6; ++row) grav_d2f[((parent*NUM_VEL + vi)*NUM_VEL + vj)*6 + row] += row_vals[row];",
        ])
    self.gen_add_end_control_flow()  # vj loop
    self.gen_add_end_control_flow()  # vi loop for d2f
    self.gen_add_end_control_flow()  # jid backward loop
    self.gen_add_end_control_flow()  # close thread-zero wrap
    # Restore the un-wrapped emission methods (mimic NUM_VEL->GRAV_N_INT rewrite is local).
    self.gen_add_code_line = _orig_gen_add_code_line
    self.gen_add_code_lines = _orig_gen_add_code_lines
    self.gen_add_sync()

def gen_idsva_so_body_frame_inner_function_call(self, use_qdd_input = False, updated_var_names = None, bc_in_smem_expr = None, scratch_in_smem_expr = None, tp_in_smem_expr = None):
    var_names = dict( \
        s_idsva_so_name = "s_idsva_so", \
        s_q_name = "s_q", \
        s_qd_name = "s_qd", \
        s_qdd_name = "s_qdd", \
        s_temp_name = "s_temp", \
        d_temp_spill_name = "d_temp_spill", \
        d_robotModel_name = "d_robotModel", \
        gravity_name = "gravity"
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    # Template args: <T> | <T, SCRATCH_IN_SMEM> | <T, SCRATCH_IN_SMEM, BC_IN_SMEM>
    #   | <T, SCRATCH_IN_SMEM, BC_IN_SMEM, TP_IN_SMEM>.
    # SCRATCH_IN_SMEM defaults true; if a caller only passes bc_in_smem_expr / tp_in_smem_expr
    # it must also pass the lower-order exprs (default "true") so positional order stays correct.
    if scratch_in_smem_expr is None and bc_in_smem_expr is None and tp_in_smem_expr is None:
        template_args = "<T>"
    elif bc_in_smem_expr is None and tp_in_smem_expr is None:
        template_args = "<T, " + scratch_in_smem_expr + ">"
    elif tp_in_smem_expr is None:
        scratch_expr = scratch_in_smem_expr if scratch_in_smem_expr is not None else "true"
        template_args = "<T, " + scratch_expr + ", " + bc_in_smem_expr + ">"
    else:
        scratch_expr = scratch_in_smem_expr if scratch_in_smem_expr is not None else "true"
        bc_expr = bc_in_smem_expr if bc_in_smem_expr is not None else "true"
        template_args = "<T, " + scratch_expr + ", " + bc_expr + ", " + tp_in_smem_expr + ">"
    id_so_code_start = "idsva_so_body_frame_inner" + template_args + "(" + var_names["s_idsva_so_name"] + ", " + var_names["s_q_name"] + ", " + var_names["s_qd_name"] + ", " + var_names["s_qdd_name"] + ", "
    id_so_code_middle = self.gen_insert_helpers_function_call()
    # Unified signature: both fixed and floating inners take
    # (s_temp, d_workspace, d_robotModel, gravity). `d_temp_spill` is the kernel-local
    # typed view into d_workspace; the inner loads s_XImats from d_robotModel internally.
    id_so_code_end = var_names["s_temp_name"] + ", " + var_names["d_temp_spill_name"] + ", " + var_names["d_robotModel_name"] + ", " + var_names["gravity_name"] + ");"
    id_so_code = id_so_code_start + id_so_code_middle + id_so_code_end
    self.gen_add_code_line(id_so_code)

def idsva_so_parent_topology_needs_reference_order_output_repair(parent_ids):
    """
    Return true when moving-joint fanout requires the reference-order repair.

    Fanout directly from the fixed world/base parent is treated as independent
    root chains and can keep the optimized assembly path. Fanout below a moving
    joint can create order-sensitive duplicate tensor writes and needs repair.
    """
    direct_child_counts = {}
    for parent_jid in parent_ids:
        if parent_jid < 0:
            continue
        direct_child_counts[parent_jid] = direct_child_counts.get(parent_jid, 0) + 1
    return any(count > 1 for count in direct_child_counts.values())

def idsva_so_needs_reference_order_output_repair(self):
    """
    Returns true for fixed-base topologies where final IDSVA-SO tensor writes
    are order-sensitive because at least one moving joint has multiple direct
    children. Serial chains and base-rooted independent chain forests keep the
    optimized parallel tensor assembly.
    """
    if self.robot.is_serial_chain():
        return False
    parent_ids = [self.robot.get_parent_id(jid) for jid in range(self.robot.get_num_joints())]
    return idsva_so_parent_topology_needs_reference_order_output_repair(parent_ids)

def gen_idsva_so_body_frame_reference_order_output_repair(self):
    """
    Emits the final second-order tensor assembly for branched fixed-base robots,
    block-cooperatively parallelized over the (jid, ancestor) work-pairs.

    The preceding generated code computes all reusable intermediates in
    parallel. The final second-order tensors have many symmetry and
    duplicate-write relationships, so the original implementation replayed the
    reference loop order on thread 0 to respect write ordering.

    That replay-order dependency is unnecessary: each (jid, ancestor_j) pair
    writes a DISJOINT set of destination cells (verified by enumerating every
    write across the full iteration space — g1 fixed, the branched robot that
    actually triggers this repair, has zero cross-pair cell conflicts), so the
    outer (jid, anc) loop can run one work-item per pair with no inter-item
    races. The only same-cell writes are intra-pair (e.g. block-D's `dM[anc,
    jid,succ]` then its mirror `dM[jid,anc,succ]`, which coincide only when
    anc==jid) and stay correctly ordered within a single thread. Each thread
    owns private rt1..rt9 / rp1..rp6 scratch, so there is no shared state to
    sync between pairs — the trailing __syncthreads() is the only barrier
    needed. The output is zeroed first in a separate block-parallel pass (with a
    sync) since every pair only writes the cells it owns and leaves the rest at
    their zeroed value.
    """
    num_bodies = self.robot.get_num_bodies()
    st_start = [0]
    st_values = []
    succ_start = [0]
    succ_values = []
    # Flatten the serial (jid desc, anc in reversed [jid]+ancestors) iteration
    # into a list of disjoint work-pairs; one device thread handles one pair.
    pair_jid = []
    pair_anc = []
    for jid in range(num_bodies):
        subtree = list(self.robot.get_subtree_by_id(jid))
        successors = [st_j for st_j in subtree if st_j != jid]
        st_values.extend(subtree)
        st_start.append(len(st_values))
        succ_values.extend(successors)
        succ_start.append(len(succ_values))
    for jid in range(num_bodies - 1, -1, -1):
        ancestors = list(self.robot.get_ancestors_by_id(jid))
        ancestors.insert(0, jid)
        ancestors = ancestors[::-1]
        for ancestor_j in ancestors:
            pair_jid.append(jid)
            pair_anc.append(ancestor_j)
    num_pairs = len(pair_jid)

    def int_array(values):
        if values:
            return ", ".join(map(str, values))
        return "0"

    self.gen_add_sync()
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line("// Final IDSVA-SO tensor assembly (block-parallel over disjoint (jid, ancestor) work-pairs)")
    self.gen_add_code_line(f"static const int idsva_ref_st_start[] = {{ {int_array(st_start)} }};")
    self.gen_add_code_line(f"static const int idsva_ref_st_values[] = {{ {int_array(st_values)} }};")
    self.gen_add_code_line(f"static const int idsva_ref_succ_start[] = {{ {int_array(succ_start)} }};")
    self.gen_add_code_line(f"static const int idsva_ref_succ_values[] = {{ {int_array(succ_values)} }};")
    self.gen_add_code_line(f"static const int idsva_ref_pair_jid[] = {{ {int_array(pair_jid)} }};")
    self.gen_add_code_line(f"static const int idsva_ref_pair_anc[] = {{ {int_array(pair_anc)} }};")
    # Pass 1: zero the whole output tensor in parallel.
    # MIMIC: the caller shadows SECOND_ORDER_COORDS to NUM_BODIES and repoints
    # s_idsva_so at the 4*NB^3 INTERNAL slab (it is folded to the reduced 4*NV^3
    # public output afterwards). SECOND_ORDER_TENSOR_SIZE stays the NV-based
    # 4*NV^3 macro, so zeroing only that many cells leaves the upper internal
    # slots (incl. the entire dM_dq block + the dvdq tail for NB>NV) stale; the
    # fold then reads garbage. Zero the full internal 4*NB^3 span for mimic
    # (byte-identical for non-mimic, where NB==NV). Mirrors the mimic-aware zero
    # in gen_idsva_so_device's output-init (the `4*n_int^3 if is_mimic` site).
    zero_span = "4*NUM_BODIES*NUM_BODIES*NUM_BODIES" if self.robot_has_mimic_joints() else "SECOND_ORDER_TENSOR_SIZE"
    self.gen_add_parallel_loop("out_idx", zero_span)
    self.gen_add_code_line("s_idsva_so[out_idx] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # Pass 2: one work-item per disjoint (jid, ancestor_j) pair.
    self.gen_add_parallel_loop("pair_idx", str(num_pairs))
    self.gen_add_code_line("T rt1[36], rt2[36], rt3[36], rt4[36], rt5[36], rt6[36], rt7[36], rt8[36], rt9[36];")
    self.gen_add_code_line("T rp1[6], rp2[6], rp3[6], rp4[6], rp5[6], rp6[6];")
    self.gen_add_code_line("int jid = idsva_ref_pair_jid[pair_idx];")
    self.gen_add_code_line("int ancestor_j = idsva_ref_pair_anc[pair_idx];")
    self.gen_add_code_line("int st_begin = idsva_ref_st_start[jid];")
    self.gen_add_code_line("int st_end = idsva_ref_st_start[jid + 1];")
    self.gen_add_code_line("int succ_begin = idsva_ref_succ_start[jid];")
    self.gen_add_code_line("int succ_end = idsva_ref_succ_start[jid + 1];")
    self.gen_idsva_so_reference_order_rt_rp_assembly(
        "S", "psid", "psidd", "psid_Sd", "jid", "ancestor_j", inline_row_col=False)
    self.gen_add_code_line("for (int st_pos = st_begin; st_pos < st_end; ++st_pos) {", True)
    self.gen_add_code_line("int st_j = idsva_ref_st_values[st_pos];")
    self.gen_add_code_line("d2tau_dq2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid*SECOND_ORDER_COORDS + ancestor_j] = -dot_prod<T, 36, 1, 1>(rt3, &D3[st_j*36]) - dot_prod<T, 6, 1, 1>(rp1, &T2[st_j*6]) + dot_prod<T, 6, 1, 1>(rp2, &T1[st_j*6]);")
    self.gen_add_code_line("d2tau_dvdq[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid*SECOND_ORDER_COORDS + ancestor_j] = -dot_prod<T, 36, 1, 1>(rt1, &D3[st_j*36]);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if (ancestor_j < jid) {", True)
    self.gen_add_code_line("for (int st_pos = st_begin; st_pos < st_end; ++st_pos) {", True)
    self.gen_add_code_line("int st_j = idsva_ref_st_values[st_pos];")
    self.gen_add_code_line("d2tau_dq2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j*SECOND_ORDER_COORDS + jid] = d2tau_dq2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid*SECOND_ORDER_COORDS + ancestor_j];")
    self.gen_add_code_line("d2tau_dqd2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j*SECOND_ORDER_COORDS + jid] = -dot_prod<T, 36, 1, 1>(rt2, &D3[st_j*36]);")
    self.gen_add_code_line("d2tau_dqd2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid*SECOND_ORDER_COORDS + ancestor_j] = d2tau_dqd2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j*SECOND_ORDER_COORDS + jid];")
    self.gen_add_code_line("d2tau_dvdq[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j*SECOND_ORDER_COORDS + jid] = -dot_prod<T, 36, 1, 1>(rt6, &D3[st_j*36]) - dot_prod<T, 6, 1, 1>(rp3, &T2[st_j*6]) + dot_prod<T, 6, 1, 1>(rp4, &T1[st_j*6]);")
    self.gen_add_code_line("d2tau_dq2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j*SECOND_ORDER_COORDS + jid] = dot_prod<T, 36, 1, 1>(rt6, &D2[st_j*36]) + dot_prod<T, 36, 1, 1>(rt7, &D1[st_j*36]) - dot_prod<T, 6, 1, 1>(rp5, &T3[st_j*6]);")
    self.gen_add_code_line("d2tau_dvdq[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j*SECOND_ORDER_COORDS + jid] = dot_prod<T, 36, 1, 1>(rt6, &D3[st_j*36]) - dot_prod<T, 6, 1, 1>(rp5, &T4[st_j*6]);")
    self.gen_add_code_line("dM_dq[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j*SECOND_ORDER_COORDS + jid] = dot_prod<T, 36, 1, 1>(rt8, &D4[st_j*36]);")
    self.gen_add_code_line("dM_dq[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j*SECOND_ORDER_COORDS + jid] = dM_dq[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j*SECOND_ORDER_COORDS + jid];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("d2tau_dqd2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid*SECOND_ORDER_COORDS + jid] = dot_prod<T, 6, 1, 1>(rp6, &S[jid*6]);")
    self.gen_add_code_line("for (int succ_pos = succ_begin; succ_pos < succ_end; ++succ_pos) {", True)
    self.gen_add_code_line("int succ_j = idsva_ref_succ_values[succ_pos];")
    self.gen_add_code_line("d2tau_dqd2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + succ_j*SECOND_ORDER_COORDS + jid] = dot_prod<T, 36, 1, 1>(rt8, &D3[succ_j*36]);")
    self.gen_add_code_line("d2tau_dqd2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid*SECOND_ORDER_COORDS + succ_j] = d2tau_dqd2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + succ_j*SECOND_ORDER_COORDS + jid];")
    self.gen_add_code_line("d2tau_dvdq[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid*SECOND_ORDER_COORDS + succ_j] = dot_prod<T, 36, 1, 1>(rt8, &D2[succ_j*36]) + dot_prod<T, 36, 1, 1>(rt9, &D1[succ_j*36]);")
    self.gen_add_code_line("d2tau_dq2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid*SECOND_ORDER_COORDS + succ_j] = d2tau_dq2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + succ_j*SECOND_ORDER_COORDS + jid];")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int succ_pos = succ_begin; succ_pos < succ_end; ++succ_pos) {", True)
    self.gen_add_code_line("int succ_j = idsva_ref_succ_values[succ_pos];")
    self.gen_add_code_line("d2tau_dq2[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j*SECOND_ORDER_COORDS + succ_j] = dot_prod<T, 36, 1, 1>(rt1, &D2[succ_j*36]) + dot_prod<T, 36, 1, 1>(rt4, &D1[succ_j*36]);")
    self.gen_add_code_line("d2tau_dqd2[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j*SECOND_ORDER_COORDS + succ_j] = dot_prod<T, 36, 1, 1>(rt2, &D3[succ_j*36]);")
    self.gen_add_code_line("d2tau_dqd2[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + succ_j*SECOND_ORDER_COORDS + ancestor_j] = d2tau_dqd2[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j*SECOND_ORDER_COORDS + succ_j];")
    self.gen_add_code_line("d2tau_dvdq[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + succ_j*SECOND_ORDER_COORDS + ancestor_j] = dot_prod<T, 36, 1, 1>(rt1, &D3[succ_j*36]);")
    self.gen_add_code_line("d2tau_dq2[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + succ_j*SECOND_ORDER_COORDS + ancestor_j] = d2tau_dq2[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j*SECOND_ORDER_COORDS + succ_j];")
    self.gen_add_code_line("d2tau_dvdq[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j*SECOND_ORDER_COORDS + succ_j] = dot_prod<T, 36, 1, 1>(rt2, &D2[succ_j*36]) + dot_prod<T, 36, 1, 1>(rt5, &D1[succ_j*36]);")
    self.gen_add_code_line("dM_dq[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid*SECOND_ORDER_COORDS + succ_j] = dot_prod<T, 36, 1, 1>(rt8, &D1[succ_j*36]);")
    self.gen_add_code_line("dM_dq[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j*SECOND_ORDER_COORDS + succ_j] = dM_dq[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid*SECOND_ORDER_COORDS + succ_j];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if (ancestor_j == jid) {", True)
    self.gen_add_code_line("for (int st_pos = st_begin; st_pos < st_end; ++st_pos) {", True)
    self.gen_add_code_line("int st_j = idsva_ref_st_values[st_pos];")
    self.gen_add_code_line("d2tau_dqd2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid*SECOND_ORDER_COORDS + ancestor_j] = -dot_prod<T, 36, 1, 1>(rt2, &D1[st_j*36]);")
    self.gen_add_end_control_flow()  # close for st_pos (block E)
    self.gen_add_end_control_flow()  # close if (ancestor_j == jid)
    self.gen_add_end_control_flow()  # close parallel loop over (jid, ancestor) pairs
    self.gen_add_sync()

# ============================================================================
# KEPT NON-PRODUCTION FALLBACK — do NOT delete without the dedicated SO audit
# (docs/open-tasks/so_audit_plan.md). `gen_idsva_so_body_frame_floating_reference_inner`
# (+ its gravity-shim family) is the body-frame floating-base SO path. It is NOT
# emitted in production (the dispatcher routes ALL floating-base SO to world_frame, so
# the floating branch of the body-frame inner is unreachable) and not a test oracle.
# It is ~entirely single-threaded — the largest serial surface in this file, but zero
# production impact. Retained as a reference; the SO audit decides keep-vs-retire.
# ============================================================================
def gen_idsva_so_body_frame_floating_reference_inner(self, use_qdd_input = False):
    """
    Emits a floating-base diagnostic IDSVA-SO path with explicit body/velocity
    split memory. Fixed-base keeps the optimized generator path below.

    DOCUMENTED NON-EMITTED REFERENCE FALLBACK (not dead-by-accident): in
    production the dispatcher routes ALL floating-base second-order to the
    world_frame path, so this body-frame floating branch is never generated /
    benchmarked. It is intentionally retained as a reference (see the header
    block above + docs/open-tasks/so_audit_plan.md); the SO audit decides
    keep-vs-retire. Do NOT remove it as unused.
    """
    NV = self.robot.get_num_vel()
    num_bodies = self.robot.get_num_bodies()
    metadata = _idsva_so_floating_velocity_metadata(self.robot)
    parent_ids = [self.robot.get_parent_id(body_id) for body_id in range(num_bodies)]

    # ---- Floating-base mimic: per-column INTERNAL-coordinate sweep + alpha fold ----
    # B4 FIX (2026-06-02): for a mimic robot NB>NV and several body-velocity columns
    # share one reduced v-slot (a mimic joint + its target). The original emission keyed
    # every per-column buffer AND the output on the REDUCED slot (`body_v_index`, which
    # carries the duplicate, e.g. fr3-floating `...,13,13`) with plain `=` writes, so the
    # mimic sibling's contribution CLOBBERED the target's instead of alpha-folding onto
    # the shared slot — wrong on exactly the mimic column (fr3 col 13). Mirror the (already
    # correct) world-frame inner: run the whole sweep in UNIQUE per-column internal slots
    # (n_int = total column count), assemble into a 4*n_int^3 internal slab, then
    # alpha-fold each axis to the reduced 4*NV^3 public output (R[i,v(i)] += alpha_i).
    # Non-mimic robots keep n_int == NV and the legacy reduced tables, so the emission is
    # byte-identical for them.
    is_mimic = self.robot_has_mimic_joints()
    n_int = metadata["n_int"]
    SO_N = "SO_N_INT" if is_mimic else "NUM_VEL"
    # Internal-slot variants of the per-body / subtree / successor column-index tables.
    # The shared metadata builds these on REDUCED slots (body_v_index); rebuild them here
    # on the UNIQUE internal slots (body_vint_index) so mimic siblings never collide. This
    # is body-frame-local (the world-frame inner has its own per-column loop structure and
    # does not consume subtree/successor lists), so the shared metadata stays untouched.
    body_v_start = metadata["body_v_start"]
    body_vint_index = metadata["body_vint_index"]
    def _internal_cols_for_body(b):
        return body_vint_index[body_v_start[b]:body_v_start[b + 1]]
    if is_mimic:
        int_body_v_index = list(body_vint_index)
        int_subtree_v_index = []
        int_subtree_v_start = [0]
        int_successor_v_index = []
        int_successor_v_start = [0]
        for body_id in range(num_bodies):
            subtree = list(self.robot.get_subtree_by_id(body_id))
            for sb in subtree:
                int_subtree_v_index.extend(_internal_cols_for_body(sb))
            int_subtree_v_start.append(len(int_subtree_v_index))
            for sb in subtree:
                if sb != body_id:
                    int_successor_v_index.extend(_internal_cols_for_body(sb))
            int_successor_v_start.append(len(int_successor_v_index))
        emit_body_v_index = int_body_v_index
        emit_subtree_v_index = int_subtree_v_index
        emit_subtree_v_start = int_subtree_v_start
        emit_successor_v_index = int_successor_v_index
        emit_successor_v_start = int_successor_v_start
        emit_vel_to_body = metadata["vel_to_body"]      # reduced (only used for S build, see below)
        emit_vel_s_index = metadata["int_s_index"]      # internal-slot S unit-axis row
        emit_vel_s_sign = metadata["int_s_sign"]        # internal-slot S unit-axis sign
    else:
        emit_body_v_index = metadata["body_v_index"]
        emit_subtree_v_index = metadata["subtree_v_index"]
        emit_subtree_v_start = metadata["subtree_v_start"]
        emit_successor_v_index = metadata["successor_v_index"]
        emit_successor_v_start = metadata["successor_v_start"]
        emit_vel_to_body = metadata["vel_to_body"]
        emit_vel_s_index = metadata["vel_s_index"]
        emit_vel_s_sign = metadata["vel_s_sign"]

    func_params = ["s_idsva_so is a pointer to memory for the final result of size 4*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS = " + str(4*NV**3), \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "s_qdd is the vector of joint accelerations", \
                   "s_temp is a pointer to helper shared memory of size  = " + \
                            str(self.gen_idsva_so_body_frame_inner_temp_mem_size()), \
                   "gravity is the gravity constant"]
    func_def_start = "void idsva_so_body_frame_inner(T *s_idsva_so, const T *s_q, const T *s_qd, T *s_qdd, "
    func_params.insert(-1, "d_workspace is a pointer to global-memory scratch (per-timestep) of size = " + \
                            str(gen_floating_gravity_d2tau_dq_spill_count(self)) + " floats")
    # Signature parity with the fixed inner: take d_robotModel and load XImats internally.
    func_def_end = "T *s_temp, T *d_workspace, const robotModel<T> *d_robotModel, const T gravity) {"
    func_params.insert(-1, "d_robotModel holds XImats/topology (the inner loads s_XImats internally)")
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -2)
    func_notes = [
        "Floating diagnostic path: body-indexed spatial state plus packed velocity-indexed derivative columns.",
        "d2tau_dq is assembled analytically in velocity-coordinate tensor space.",
    ]
    func_def = func_def_start + func_def_end

    self.gen_add_func_doc("Computes floating-base second-order inverse dynamics diagnostics",func_notes,func_params,None)
    # SCRATCH_IN_SMEM / BC_IN_SMEM are accepted for signature uniformity with the
    # fixed-base inner but inert here (the floating diagnostic path picks (0,0,0): it is
    # never surgically spilled and `d_workspace` carries the gravity-Hessian shim, not
    # the s_temp pool). The repoint below is guarded so it never fires for floating.
    self.gen_add_code_line("template <typename T, bool SCRATCH_IN_SMEM = true, bool BC_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # Inner-owns-placement parity with the fixed inner: load XImats internally (after the
    # repoint). For floating SCRATCH_IN_SMEM is always true so s_temp is untouched and
    # d_workspace keeps its gravity-shim meaning.
    self.gen_add_code_line("if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; } else { (void)0; }")
    self.gen_load_update_XImats_helpers_function_call()

    # int_to_body[s] = owning body of internal slot s (for the per-internal-slot S build).
    int_to_body = [0] * n_int
    for body_id in range(num_bodies):
        for s in _internal_cols_for_body(body_id):
            int_to_body[s] = body_id
    self.gen_add_code_lines(([
        # SO_N_INT = per-column internal coordinate count; drives the vel-band sizes and
        # the internal output-slab strides. Only emitted (and used) for mimic robots.
        f"constexpr int SO_N_INT = {n_int};",
    ] if is_mimic else []) + [
        "// Floating IDSVA-SO split-memory layout.",
        "T *I = s_XImats + XIMAT_SIZE*NUM_BODIES;",
        "T *Xdown = s_temp;",
        "T *Xup = Xdown + 36*NUM_BODIES;",
        "T *IC = Xup + 36*NUM_BODIES;",
        "T *I_Xup = IC + 36*NUM_BODIES;",
        "T *BC = I_Xup + 36*NUM_BODIES;",
        "T *crm_v = BC + 36*NUM_BODIES;",
        "T *crf_v = crm_v + 36*NUM_BODIES;",
        "T *vJ = crf_v + 36*NUM_BODIES;",
        "T *v = vJ + 6*NUM_BODIES;",
        "T *aJ = v + 6*NUM_BODIES;",
        "T *a = aJ + 6*NUM_BODIES;",
        "T *f = a + 6*NUM_BODIES;",
        "T *IC_v = f + 6*NUM_BODIES;",
        "T *a_world = IC_v + 6*NUM_BODIES;",
        "T *S_vel = a_world + 6;",
        f"T *Sd_vel = S_vel + 6*{SO_N};",
        f"T *psid_vel = Sd_vel + 6*{SO_N};",
        f"T *psidd_vel = psid_vel + 6*{SO_N};",
        f"T *psid_Sd_vel = psidd_vel + 6*{SO_N};",
        f"T *IC_S = psid_Sd_vel + 6*{SO_N};",
        f"T *IC_psid = IC_S + 6*{SO_N};",
        f"T *ICT_S = IC_psid + 6*{SO_N};",
        "T *T1 = IC_S;",
        f"T *T2 = ICT_S + 6*{SO_N};",
        f"T *T3 = T2 + 6*{SO_N};",
        f"T *T4 = T3 + 6*{SO_N};",
        f"T *crm_S = T4 + 6*{SO_N};",
        f"T *crf_S = crm_S + 36*{SO_N};",
        f"T *crm_psid = crf_S + 36*{SO_N};",
        f"T *crf_psid = crm_psid + 36*{SO_N};",
        "// icrf_f used to live here (size 36*NUM_BODIES) but is now a kernel-local",
        "// per-body 36-float array; saves NUM_BODIES * 36 floats of shared memory.",
        f"T *B_IC_S = crf_psid + 36*{SO_N};",
        f"T *D1 = B_IC_S + 36*{SO_N};",
        f"T *D2 = D1 + 36*{SO_N};",
        "T *D3 = B_IC_S;",
        f"T *D4 = D2 + 36*{SO_N};",
        f"T *crf_S_IC = D4 + 36*{SO_N};",] + ([
        # Mimic: a 4*n_int^3 INTERNAL slab placed after the vel bands. The sweep assembles
        # into it (SO_N_INT stride); the public reduced caller dest is saved and folded last.
        f"T *so_internal = crf_S_IC + 36*{SO_N};",
        "T *s_idsva_so_public = s_idsva_so;  // reduced 4*NV^3 caller dest (saved before repoint)",
        "T *d2tau_dq2 = so_internal;",
        "T *d2tau_dqd2 = d2tau_dq2 + SO_N_INT*SO_N_INT*SO_N_INT;",
        "T *d2tau_dvdq = d2tau_dqd2 + SO_N_INT*SO_N_INT*SO_N_INT;",
        "T *dM_dq = d2tau_dvdq + SO_N_INT*SO_N_INT*SO_N_INT;",
    ] if is_mimic else [
        "T *d2tau_dq2 = s_idsva_so;",
        "T *d2tau_dqd2 = d2tau_dq2 + SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS;",
        "T *d2tau_dvdq = d2tau_dqd2 + SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS;",
        "T *dM_dq = d2tau_dvdq + SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS;",
    ]) + [
        "",
        f"static const int idsva_float_parent[] = {{ {_idsva_so_int_array(parent_ids)} }};",
        f"static const int body_v_start[] = {{ {_idsva_so_int_array(metadata['body_v_start'])} }};",
        f"static const int body_v_index[] = {{ {_idsva_so_int_array(emit_body_v_index)} }};",
        f"static const int vel_to_body[] = {{ {_idsva_so_int_array(emit_vel_to_body)} }};",
        f"static const int vel_s_index[] = {{ {_idsva_so_int_array(emit_vel_s_index)} }};",
        f"static const int vel_s_sign[] = {{ {_idsva_so_int_array(emit_vel_s_sign)} }};",
        f"static const int subtree_v_start[] = {{ {_idsva_so_int_array(emit_subtree_v_start)} }};",
        f"static const int subtree_v_index[] = {{ {_idsva_so_int_array(emit_subtree_v_index)} }};",
        f"static const int successor_v_start[] = {{ {_idsva_so_int_array(emit_successor_v_start)} }};",
        f"static const int successor_v_index[] = {{ {_idsva_so_int_array(emit_successor_v_index)} }};",
        f"static const int ancestor_body_start[] = {{ {_idsva_so_int_array(metadata['ancestor_body_start'])} }};",
        f"static const int ancestor_body_index[] = {{ {_idsva_so_int_array(metadata['ancestor_body_index'])} }};",] + ([
        # internal slot -> owning body (per-internal-slot S build), -> true reduced v-slot
        # (alpha-scaled qd/qdd reads), -> mimic alpha (qd/qdd reads + the final axis fold).
        f"static const int int_to_body[] = {{ {_idsva_so_int_array(int_to_body)} }};",
        f"static const int int_true_vel[] = {{ {_idsva_so_int_array(metadata['int_true_vel'])} }};",
        "static const T int_alpha[] = { " + ", ".join(
            "static_cast<T>(" + repr(a) + ")" for a in metadata['int_alpha']) + " };",
    ] if is_mimic else []) + [
        "",
    ])

    self.gen_add_code_line("if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {", True)
    if is_mimic:
        # Mimic: zero the 4*n_int^3 INTERNAL slab (d2tau_dq2 points at it). The reduced
        # 4*NV^3 public output is zeroed separately just before the fold.
        self.gen_add_code_line("for (int out_idx = 0; out_idx < 4*SO_N_INT*SO_N_INT*SO_N_INT; ++out_idx) d2tau_dq2[out_idx] = static_cast<T>(0);")
    else:
        self.gen_add_code_line("for (int out_idx = 0; out_idx < SECOND_ORDER_TENSOR_SIZE; ++out_idx) s_idsva_so[out_idx] = static_cast<T>(0);")
    self.gen_add_code_line("// Floating-base: run main sweep with gravity = 0; gravity Hessian added below by")
    self.gen_add_code_line("// `gen_floating_gravity_d2tau_dq_lie_inline` (mirrors Python idsva_gravity = 0.0 + shim).")
    self.gen_add_code_line("a_world[0] = static_cast<T>(0); a_world[1] = static_cast<T>(0); a_world[2] = static_cast<T>(0);")
    self.gen_add_code_line("a_world[3] = static_cast<T>(0); a_world[4] = static_cast<T>(0); a_world[5] = static_cast<T>(0);")

    self.gen_add_code_line("// Compute accumulated Xup transforms.")
    self.gen_add_code_line("for (int jid = 0; jid < NUM_BODIES; ++jid) {", True)
    self.gen_add_code_line("int parent = idsva_float_parent[jid];")
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) {", True)
    self.gen_add_code_line("int row = idx % 6;")
    self.gen_add_code_line("int col = idx / 6;")
    self.gen_add_code_line("if (parent < 0) {", True)
    self.gen_add_code_line("Xup[jid*36 + idx] = s_XImats[jid*36 + idx];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("else {", True)
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int kk = 0; kk < 6; ++kk) acc += s_XImats[jid*36 + row + 6*kk] * Xup[parent*36 + kk + 6*col];")
    self.gen_add_code_line("Xup[jid*36 + idx] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()

    self.gen_add_code_line("// Compute IC = Xup^T * I * Xup.")
    self.gen_add_code_line("for (int jid = 0; jid < NUM_BODIES; ++jid) {", True)
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) {", True)
    self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int kk = 0; kk < 6; ++kk) acc += I[jid*36 + row + 6*kk] * Xup[jid*36 + kk + 6*col];")
    self.gen_add_code_line("I_Xup[jid*36 + idx] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) {", True)
    self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int kk = 0; kk < 6; ++kk) acc += Xup[jid*36 + kk + 6*row] * I_Xup[jid*36 + kk + 6*col];")
    self.gen_add_code_line("IC[jid*36 + idx] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()

    self.gen_add_code_line("// Compute Xdown using the same spatial-transform inverse pattern as the fixed path.")
    self.gen_idsva_so_xdown_plucker_inverse("floating_serial")

    self.gen_add_code_line("// Transform each velocity-coordinate S column.")
    # Mimic: iterate UNIQUE internal slots (n_int), each mapped to its owning body via
    # int_to_body and its internal-slot S-axis via vel_s_index/sign (internal-indexed).
    # Non-mimic: byte-identical legacy reduced-slot loop (NUM_VEL, vel_to_body).
    self.gen_add_code_line(f"for (int vel = 0; vel < {SO_N}; ++vel) {{", True)
    self.gen_add_code_line(("int jid = int_to_body[vel];" if is_mimic else "int jid = vel_to_body[vel];"))
    self.gen_add_code_line("int s_col = vel_s_index[vel];")
    self.gen_add_code_line("T s_sign = static_cast<T>(vel_s_sign[vel]);")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) S_vel[vel*6 + row] = s_sign * Xdown[jid*36 + s_col*6 + row];")
    self.gen_add_end_control_flow()

    self.gen_add_code_line("// Forward pass in body order.")
    self.gen_add_code_line("for (int jid = 0; jid < NUM_BODIES; ++jid) {", True)
    self.gen_add_code_line("int parent = idsva_float_parent[jid];")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) {", True)
    self.gen_add_code_line("vJ[jid*6 + row] = static_cast<T>(0);")
    self.gen_add_code_line("aJ[jid*6 + row] = static_cast<T>(0);")
    self.gen_add_code_line("for (int pos = body_v_start[jid]; pos < body_v_start[jid + 1]; ++pos) {", True)
    self.gen_add_code_line("int vel = body_v_index[pos];")
    if is_mimic:
        # Mimic: `vel` is a unique internal slot; read the (possibly shared) reduced qd/qdd
        # slot and scale by the joint's mimic multiplier (mirrors the oracle's
        # `_qd = alpha_i * qd[inds_v_true]`). The fold below applies the remaining alpha
        # factors on the OTHER axes.
        self.gen_add_code_line("T qd_v = int_alpha[vel] * s_qd[int_true_vel[vel]];")
        self.gen_add_code_line("T qdd_v = int_alpha[vel] * s_qdd[int_true_vel[vel]];")
        self.gen_add_code_line("vJ[jid*6 + row] += S_vel[vel*6 + row] * qd_v;")
        self.gen_add_code_line("aJ[jid*6 + row] += S_vel[vel*6 + row] * qdd_v;")
    else:
        self.gen_add_code_line("vJ[jid*6 + row] += S_vel[vel*6 + row] * s_qd[vel];")
        self.gen_add_code_line("aJ[jid*6 + row] += S_vel[vel*6 + row] * s_qdd[vel];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if (parent < 0) {", True)
    self.gen_add_code_line("v[jid*6 + row] = static_cast<T>(0);")
    self.gen_add_code_line("a[jid*6 + row] = dot_prod<T, 6, 6, 1>(&Xdown[jid*36 + row], a_world);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("else {", True)
    self.gen_add_code_line("v[jid*6 + row] = v[parent*6 + row];")
    self.gen_add_code_line("a[jid*6 + row] = a[parent*6 + row];")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) aJ[jid*6 + row] += crm_mul<T>(row, &v[jid*6], &vJ[jid*6]);")
    # psid_vel and psidd_vel must stay in SEPARATE loops: fusing them makes
    # `crm_mul(row, &v, &psid_vel[vel*6])` read psid_vel rows 1..5 not yet written
    # in the current iteration (stale/zero => wrong psidd_vel rows). Fully populate
    # psid_vel for the current vel BEFORE consuming it in psidd_vel.
    self.gen_add_code_line("for (int pos = body_v_start[jid]; pos < body_v_start[jid + 1]; ++pos) {", True)
    self.gen_add_code_line("int vel = body_v_index[pos];")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) psid_vel[vel*6 + row] = crm_mul<T>(row, &v[jid*6], &S_vel[vel*6]);")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) psidd_vel[vel*6 + row] = crm_mul<T>(row, &a[jid*6], &S_vel[vel*6]) + crm_mul<T>(row, &v[jid*6], &psid_vel[vel*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) {", True)
    self.gen_add_code_line("v[jid*6 + row] += vJ[jid*6 + row];")
    self.gen_add_code_line("a[jid*6 + row] += aJ[jid*6 + row];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int pos = body_v_start[jid]; pos < body_v_start[jid + 1]; ++pos) {", True)
    self.gen_add_code_line("int vel = body_v_index[pos];")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) Sd_vel[vel*6 + row] = crm_mul<T>(row, &v[jid*6], &S_vel[vel*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) {", True)
    self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("crm_v[jid*36 + idx] = crm<T>(idx, &v[jid*6]);")
    self.gen_add_code_line("crf_v[jid*36 + row*6 + col] = -crm<T>(idx, &v[jid*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) IC_v[jid*6 + row] = dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &v[jid*6]);")
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) {", True)
    self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("BC[jid*36 + idx] = dot_prod<T, 6, 6, 1>(&crf_v[jid*36 + row], &IC[jid*36 + col*6]) + icrf<T>(idx, &IC_v[jid*6]) - dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &crm_v[jid*36 + col*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) f[jid*6 + row] = dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &a[jid*6]) + dot_prod<T, 6, 6, 1>(&crf_v[jid*36 + row], &IC_v[jid*6]);")
    self.gen_add_end_control_flow()

    self.gen_add_code_line("// Backward accumulation of body-indexed composite quantities.")
    self.gen_add_code_line("for (int jid = NUM_BODIES - 1; jid >= 0; --jid) {", True)
    self.gen_add_code_line("int parent = idsva_float_parent[jid];")
    self.gen_add_code_line("if (parent >= 0) {", True)
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) { IC[parent*36 + idx] += IC[jid*36 + idx]; BC[parent*36 + idx] += BC[jid*36 + idx]; }")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) f[parent*6 + row] += f[jid*6 + row];")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()

    self.gen_add_code_line("// Build velocity-indexed T and D intermediates.")
    self.gen_add_code_line("for (int jid = NUM_BODIES - 1; jid >= 0; --jid) {", True)
    self.gen_add_code_line("// icrf(f[jid]) is consumed only within this per-jid loop's T3 build, so")
    self.gen_add_code_line("// keep it as a 36-float kernel-local array instead of NUM_BODIES*36 in shared.")
    self.gen_add_code_line("T icrf_f_local[36];")
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) icrf_f_local[idx] = icrf<T>(idx, &f[jid*6]);")
    self.gen_add_code_line("for (int pos = body_v_start[jid]; pos < body_v_start[jid + 1]; ++pos) {", True)
    self.gen_add_code_line("int vel = body_v_index[pos];")
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) {", True)
    self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("crm_S[vel*36 + idx] = crm<T>(idx, &S_vel[vel*6]);")
    self.gen_add_code_line("crf_S[vel*36 + row*6 + col] = -crm<T>(idx, &S_vel[vel*6]);")
    self.gen_add_code_line("crm_psid[vel*36 + idx] = crm<T>(idx, &psid_vel[vel*6]);")
    self.gen_add_code_line("crf_psid[vel*36 + row*6 + col] = -crm<T>(idx, &psid_vel[vel*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) {", True)
    self.gen_add_code_line("IC_S[vel*6 + row] = dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &S_vel[vel*6]);")
    self.gen_add_code_line("IC_psid[vel*6 + row] = dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &psid_vel[vel*6]);")
    self.gen_add_code_line("psid_Sd_vel[vel*6 + row] = psid_vel[vel*6 + row] + Sd_vel[vel*6 + row];")
    self.gen_add_code_line("ICT_S[vel*6 + row] = dot_prod<T, 6, 1, 1>(&IC[jid*36 + row*6], &S_vel[vel*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) {", True)
    self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("B_IC_S[vel*36 + idx] = dot_prod<T, 6, 6, 1>(&crf_S[vel*36 + row], &IC[jid*36 + col*6]) + icrf<T>(idx, &IC_S[vel*6]) - dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &crm_S[vel*36 + col*6]);")
    self.gen_add_code_line("D2[vel*36 + idx] = dot_prod<T, 6, 6, 1>(&crf_psid[vel*36 + row], &IC[jid*36 + col*6]) + icrf<T>(idx, &IC_psid[vel*6]) - dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &crm_psid[vel*36 + col*6]);")
    self.gen_add_code_line("// RBDReference stores D1 with NumPy default flatten() order, unlike D2/D3/D4.")
    self.gen_add_code_line("D1[vel*36 + row*6 + col] = dot_prod<T, 6, 6, 1>(&crf_S[vel*36 + row], &IC[jid*36 + col*6]) - dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &crm_S[vel*36 + col*6]);")
    self.gen_add_code_line("D2[vel*36 + idx] += dot_prod<T, 6, 6, 1>(&crf_S[vel*36 + row], &BC[jid*36 + col*6]) - dot_prod<T, 6, 6, 1>(&BC[jid*36 + row], &crm_S[vel*36 + col*6]);")
    self.gen_add_code_line("D4[vel*36 + idx] = icrf<T>(idx, &ICT_S[vel*6]);")
    self.gen_add_code_line("crf_S_IC[vel*36 + idx] = dot_prod<T, 6, 6, 1>(&crf_S[vel*36 + row], &IC[jid*36 + col*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) {", True)
    self.gen_add_code_line("T2[vel*6 + row] = -dot_prod<T, 6, 1, 1>(&BC[jid*36 + row*6], &S_vel[vel*6]);")
    self.gen_add_code_line("T3[vel*6 + row] = dot_prod<T, 6, 6, 1>(&BC[jid*36 + row], &psid_vel[vel*6]) + dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &psidd_vel[vel*6]) + dot_prod<T, 6, 6, 1>(&icrf_f_local[row], &S_vel[vel*6]);")
    self.gen_add_code_line("T4[vel*6 + row] = dot_prod<T, 6, 6, 1>(&BC[jid*36 + row], &S_vel[vel*6]) + dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &psid_Sd_vel[vel*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()

    # Output-stride token: the internal slab uses the SO_N_INT stride for mimic, else the
    # legacy SECOND_ORDER_COORDS (== NUM_VEL) so non-mimic emission stays byte-identical.
    SO = "SO_N_INT" if is_mimic else "SECOND_ORDER_COORDS"
    self.gen_add_code_line("// Reference-order velocity-indexed tensor assembly.")
    self.gen_add_code_line("T rt1[36], rt2[36], rt3[36], rt4[36], rt5[36], rt6[36], rt7[36], rt8[36], rt9[36];")
    self.gen_add_code_line("T rp1[6], rp2[6], rp3[6], rp4[6], rp5[6], rp6[6];")
    self.gen_add_code_line("for (int jid = NUM_BODIES - 1; jid >= 0; --jid) {", True)
    self.gen_add_code_line("int st_begin = subtree_v_start[jid]; int st_end = subtree_v_start[jid + 1];")
    self.gen_add_code_line("int succ_begin = successor_v_start[jid]; int succ_end = successor_v_start[jid + 1];")
    self.gen_add_code_line("int anc_begin = ancestor_body_start[jid]; int anc_end = ancestor_body_start[jid + 1];")
    self.gen_add_code_line("for (int dpos = body_v_start[jid]; dpos < body_v_start[jid + 1]; ++dpos) {", True)
    self.gen_add_code_line("int dd = body_v_index[dpos];")
    self.gen_add_code_line("for (int anc_pos = anc_begin; anc_pos < anc_end; ++anc_pos) {", True)
    self.gen_add_code_line("int ancestor_body = ancestor_body_index[anc_pos];")
    self.gen_add_code_line("for (int cpos = body_v_start[ancestor_body]; cpos < body_v_start[ancestor_body + 1]; ++cpos) {", True)
    self.gen_add_code_line("int cc = body_v_index[cpos];")
    self.gen_idsva_so_reference_order_rt_rp_assembly(
        "S_vel", "psid_vel", "psidd_vel", "psid_Sd_vel", "dd", "cc", inline_row_col=True)
    self.gen_add_code_line("for (int st_pos = st_begin; st_pos < st_end; ++st_pos) {", True)
    self.gen_add_code_line("int st_vel = subtree_v_index[st_pos];")
    self.gen_add_code_line(f"d2tau_dq2[st_vel*{SO}*{SO} + dd*{SO} + cc] = -dot_prod<T, 36, 1, 1>(rt3, &D3[st_vel*36]) - dot_prod<T, 6, 1, 1>(rp1, &T2[st_vel*6]) + dot_prod<T, 6, 1, 1>(rp2, &T1[st_vel*6]);")
    self.gen_add_code_line(f"d2tau_dvdq[st_vel*{SO}*{SO} + dd*{SO} + cc] = -dot_prod<T, 36, 1, 1>(rt1, &D3[st_vel*36]);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if (ancestor_body < jid) {", True)
    self.gen_add_code_line("for (int st_pos = st_begin; st_pos < st_end; ++st_pos) {", True)
    self.gen_add_code_line("int st_vel = subtree_v_index[st_pos];")
    self.gen_add_code_line(f"d2tau_dq2[st_vel*{SO}*{SO} + cc*{SO} + dd] = d2tau_dq2[st_vel*{SO}*{SO} + dd*{SO} + cc];")
    self.gen_add_code_line("T dqd_val = -dot_prod<T, 36, 1, 1>(rt2, &D3[st_vel*36]);")
    self.gen_add_code_line(f"d2tau_dqd2[st_vel*{SO}*{SO} + cc*{SO} + dd] = dqd_val;")
    self.gen_add_code_line(f"d2tau_dqd2[st_vel*{SO}*{SO} + dd*{SO} + cc] = dqd_val;")
    self.gen_add_code_line(f"d2tau_dvdq[st_vel*{SO}*{SO} + cc*{SO} + dd] = -dot_prod<T, 36, 1, 1>(rt6, &D3[st_vel*36]) - dot_prod<T, 6, 1, 1>(rp3, &T2[st_vel*6]) + dot_prod<T, 6, 1, 1>(rp4, &T1[st_vel*6]);")
    self.gen_add_code_line(f"d2tau_dq2[cc*{SO}*{SO} + st_vel*{SO} + dd] = dot_prod<T, 36, 1, 1>(rt6, &D2[st_vel*36]) + dot_prod<T, 36, 1, 1>(rt7, &D1[st_vel*36]) - dot_prod<T, 6, 1, 1>(rp5, &T3[st_vel*6]);")
    self.gen_add_code_line(f"d2tau_dvdq[cc*{SO}*{SO} + st_vel*{SO} + dd] = dot_prod<T, 36, 1, 1>(rt6, &D3[st_vel*36]) - dot_prod<T, 6, 1, 1>(rp5, &T4[st_vel*6]);")
    self.gen_add_code_line("T dm_val = dot_prod<T, 36, 1, 1>(rt8, &D4[st_vel*36]);")
    self.gen_add_code_line(f"dM_dq[cc*{SO}*{SO} + st_vel*{SO} + dd] = dm_val;")
    self.gen_add_code_line(f"dM_dq[st_vel*{SO}*{SO} + cc*{SO} + dd] = dm_val;")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("T dqd_diag = static_cast<T>(0);")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) {", True)
    self.gen_add_code_line("dqd_diag += S_vel[dd*6 + row] * dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], rp3);")
    self.gen_add_code_line("dqd_diag += S_vel[cc*6 + row] * dot_prod<T, 6, 6, 1>(&crf_S[dd*36 + row], &IC_S[dd*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line(f"d2tau_dqd2[cc*{SO}*{SO} + dd*{SO} + dd] = dqd_diag;")
    self.gen_add_code_line("for (int succ_pos = succ_begin; succ_pos < succ_end; ++succ_pos) {", True)
    self.gen_add_code_line("int succ_vel = successor_v_index[succ_pos];")
    self.gen_add_code_line("T dqd_succ = dot_prod<T, 36, 1, 1>(rt8, &D3[succ_vel*36]);")
    self.gen_add_code_line(f"d2tau_dqd2[cc*{SO}*{SO} + succ_vel*{SO} + dd] = dqd_succ;")
    self.gen_add_code_line(f"d2tau_dqd2[cc*{SO}*{SO} + dd*{SO} + succ_vel] = dqd_succ;")
    self.gen_add_code_line(f"d2tau_dvdq[cc*{SO}*{SO} + dd*{SO} + succ_vel] = dot_prod<T, 36, 1, 1>(rt8, &D2[succ_vel*36]) + dot_prod<T, 36, 1, 1>(rt9, &D1[succ_vel*36]);")
    self.gen_add_code_line(f"d2tau_dq2[cc*{SO}*{SO} + dd*{SO} + succ_vel] = d2tau_dq2[cc*{SO}*{SO} + succ_vel*{SO} + dd];")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int succ_pos = succ_begin; succ_pos < succ_end; ++succ_pos) {", True)
    self.gen_add_code_line("int succ_vel = successor_v_index[succ_pos];")
    self.gen_add_code_line(f"d2tau_dq2[dd*{SO}*{SO} + cc*{SO} + succ_vel] = dot_prod<T, 36, 1, 1>(rt1, &D2[succ_vel*36]) + dot_prod<T, 36, 1, 1>(rt4, &D1[succ_vel*36]);")
    self.gen_add_code_line("T dqd_child = dot_prod<T, 36, 1, 1>(rt2, &D3[succ_vel*36]);")
    self.gen_add_code_line(f"d2tau_dqd2[dd*{SO}*{SO} + cc*{SO} + succ_vel] = dqd_child;")
    self.gen_add_code_line(f"d2tau_dqd2[dd*{SO}*{SO} + succ_vel*{SO} + cc] = dqd_child;")
    self.gen_add_code_line(f"d2tau_dvdq[dd*{SO}*{SO} + succ_vel*{SO} + cc] = dot_prod<T, 36, 1, 1>(rt1, &D3[succ_vel*36]);")
    self.gen_add_code_line(f"d2tau_dq2[dd*{SO}*{SO} + succ_vel*{SO} + cc] = d2tau_dq2[dd*{SO}*{SO} + cc*{SO} + succ_vel];")
    self.gen_add_code_line(f"d2tau_dvdq[dd*{SO}*{SO} + cc*{SO} + succ_vel] = dot_prod<T, 36, 1, 1>(rt2, &D2[succ_vel*36]) + dot_prod<T, 36, 1, 1>(rt5, &D1[succ_vel*36]);")
    self.gen_add_code_line("T dm_child = dot_prod<T, 36, 1, 1>(rt8, &D1[succ_vel*36]);")
    self.gen_add_code_line(f"dM_dq[cc*{SO}*{SO} + dd*{SO} + succ_vel] = dm_child;")
    self.gen_add_code_line(f"dM_dq[dd*{SO}*{SO} + cc*{SO} + succ_vel] = dm_child;")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if (ancestor_body == jid) {", True)
    self.gen_add_code_line("for (int st_pos = st_begin; st_pos < st_end; ++st_pos) {", True)
    self.gen_add_code_line("int st_vel = subtree_v_index[st_pos];")
    self.gen_add_code_line(f"d2tau_dqd2[st_vel*{SO}*{SO} + dd*{SO} + cc] = -dot_prod<T, 36, 1, 1>(rt2, &D1[st_vel*36]);")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()

    if self.robot.floating_base and self.robot.using_quaternion:
        self.gen_add_code_line("// Reduced quaternion-vector q columns differentiate rotation with a factor of two at the identity convention used by GRiD/RBDReference.")
        # The root q-columns 3..5 occupy internal slots 3..5 (alpha=1, identity fold), so
        # the factor-2 is applied on the internal slab (SO stride) BEFORE the fold.
        self.gen_add_code_line(f"for (int q_col = 3; q_col < 6 && q_col < {SO}; ++q_col) {{", True)
        self.gen_add_code_line(f"for (int rc = 0; rc < {SO}*{SO}; ++rc) {{", True)
        self.gen_add_code_line(f"d2tau_dq2[rc*{SO} + q_col] *= static_cast<T>(2);")
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()  # close main thread-0 block
    # Phase B: emit the Lie-tangent gravity-Hessian addition. Main sweep above ran with
    # a_world[5] = 0, so this call provides the missing gravity contribution to d2tau_dq2
    # (mirrors the Python `idsva_so` + `_floating_gravity_d2tau_dq_lie_direct` pattern).
    # For mimic, d2tau_dq2 still points at the INTERNAL slab (internal coords): the shim
    # runs in internal slots and ADDS into the internal d2tau_dq block, so the single fold
    # below reduces sweep + gravity together (matching the oracle, which folds its
    # internal-coord gravity Hessian the same way). For non-mimic d2tau_dq2 == s_idsva_so
    # (the public reduced output) and the shim adds directly there.
    self.gen_floating_gravity_d2tau_dq_lie_inline()
    if is_mimic:
        # ---- Mimic fold: reduce the internal 4*n_int^3 slab (sweep + gravity) to the
        # reduced 4*NV^3 public output. public[v(i),v(j),v(k)] +=
        # alpha_i*alpha_j*alpha_k * internal[i,j,k] (the einsum('ia,ijk,jb,kc->abc',
        # R, T, R, R) reduction the oracle applies, with R[i,v(i)] = alpha_i).
        self.gen_add_code_line("if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {", True)
        self.gen_add_code_line("// Mimic fold: internal n_int^3 sweep+gravity -> reduced NV^3 public output.")
        self.gen_add_code_line("for (int i = 0; i < SECOND_ORDER_TENSOR_SIZE; ++i) s_idsva_so_public[i] = static_cast<T>(0);")
        self.gen_add_code_line("for (int blk = 0; blk < 4; ++blk) {", True)
        self.gen_add_code_line("for (int ii = 0; ii < SO_N_INT; ++ii) {", True)
        self.gen_add_code_line("for (int jj = 0; jj < SO_N_INT; ++jj) {", True)
        self.gen_add_code_line("for (int kk = 0; kk < SO_N_INT; ++kk) {", True)
        self.gen_add_code_line("T val = so_internal[blk*SO_N_INT*SO_N_INT*SO_N_INT + ii*SO_N_INT*SO_N_INT + jj*SO_N_INT + kk];")
        self.gen_add_code_line("if (val != static_cast<T>(0)) {", True)
        self.gen_add_code_line("T w = int_alpha[ii] * int_alpha[jj] * int_alpha[kk];")
        self.gen_add_code_line("s_idsva_so_public[blk*NUM_VEL*NUM_VEL*NUM_VEL + int_true_vel[ii]*NUM_VEL*NUM_VEL + int_true_vel[jj]*NUM_VEL + int_true_vel[kk]] += w * val;")
        self.gen_add_end_control_flow()  # close if (val != 0)
        self.gen_add_end_control_flow()  # close for kk
        self.gen_add_end_control_flow()  # close for jj
        self.gen_add_end_control_flow()  # close for ii
        self.gen_add_end_control_flow()  # close for blk
        self.gen_add_end_control_flow()  # close fold thread-0 block
    self.gen_add_sync()
    self.gen_add_end_function()

def gen_idsva_so_body_frame_inner(self, use_qdd_input = False):
    """
    Generates the inner device function to compute the second order idsva.

    Inner-owns-placement (mirrors fdsva_so_device / crba_inner): two compile-time
    spill levers select where scratch lives, decided at the very top of the function so
    every consumer (including the internal load_update_XImats helper's sincos scratch)
    follows the placement:
      - SCRATCH_IN_SMEM (whole-arena): the s_temp pool is in smem (true) or routed to
        d_workspace (false, the guaranteed-fit fallback rung).
      - BC_IN_SMEM (surgical): only the cold BC slab routes to d_workspace. BC is laid
        out as the LAST 36*NB slab of the arena and B_IC_S/D3 are anchored on the stable
        hot buffer S (NOT on BC), so spilling BC truncates only the arena tail and leaves
        every hot buffer (incl. D3, read by the reference-order repair) in place. This is
        the de-alias that fixes the surgical-rung g1/h1_2 regression.
      - TP_IN_SMEM (surgical): only the ancestor-pair scratch t/p1..p6 (36*len(jids_a)
        floats; 30-45% of the body arena) routes to d_workspace. t/p is anchored on
        tp_anchor (the fixed in-smem hot-chain end) and is dead through the whole forward
        recursion + D-matrix build (live only in the final block-parallel output assembly),
        so spilling it keeps every recursion-hot buffer in smem. BC re-bases off tp_anchor
        too, so it slides down to fill the vacated smem and the arena shrinks by exactly
        36*len(jids_a). Mutually exclusive with BC_IN_SMEM/SCRATCH_IN_SMEM per the tier table.
    The inner loads/updates s_XImats from s_q internally, so it takes d_robotModel.
    """
    if self.robot.floating_base:
        self.gen_idsva_so_body_frame_floating_reference_inner(use_qdd_input)
        return

    NV = self.robot.get_num_vel()
    num_bodies = self.robot.get_num_bodies()
    max_bfs_levels = self.robot.get_max_bfs_level()
    n_bfs_levels = max_bfs_levels + 1 # starts at 0

    # construct the boilerplate and function definition
    func_params = ["s_idsva_so is a pointer to memory for the final result of size 4*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS = " + str(4*NV**3), \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "s_qdd is the vector of joint accelerations", \
                   "s_temp is the shared scratch pool (used when SCRATCH_IN_SMEM) of size  = " + \
                            str(self.gen_idsva_so_body_frame_inner_temp_mem_size()), \
                   "d_workspace is the global scratch pool: routes the whole s_temp arena (when !SCRATCH_IN_SMEM) or just the cold BC slab (when !BC_IN_SMEM)", \
                   "gravity is the gravity constant"]
    func_def_start = "void idsva_so_body_frame_inner(T *s_idsva_so, const T *s_q, const T *s_qd, T *s_qdd, "
    # The inner now loads/updates XImats internally, so it takes d_robotModel (mirrors
    # fdsva_so_device). It still receives s_XImats/s_topology_helpers (the smem dest
    # buffers) via gen_insert_helpers_func_def_params.
    func_def_end = "T *s_temp, T *d_workspace, const robotModel<T> *d_robotModel, const T gravity) {"
    func_params.insert(-1, "d_robotModel holds XImats/topology (the inner loads s_XImats internally)")
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -2)
    # Inner-owns-placement: the whole s_temp pool placement (and the XImats helper's
    # sincos scratch, since the helper now runs INSIDE after the repoint) is the inner's
    # call. Mirrors fdsva_so_device / crba_inner.
    func_notes = ["Loads/updates s_XImats from s_q internally (helper runs after the SCRATCH_IN_SMEM repoint so its scratch follows the placement)"]
    func_def = func_def_start + func_def_end
    # then generate the code
    self.gen_add_func_doc("Computes the second order derivatives of inverse dynamics",func_notes,func_params,None)
    # SCRATCH_IN_SMEM: whole-arena lever (s_temp pool in smem vs routed to d_workspace).
    # BC_IN_SMEM: surgical lever (only the cold BC slab routes to d_workspace).
    # TP_IN_SMEM: surgical lever (only the ancestor-pair scratch t/p1..p6, 36*len(jids_a)
    #   floats, routes to d_workspace). t/p is DEAD through the whole recursion-hot forward
    #   sweep + D-matrix build; it is written/read ONLY in the final block-parallel output
    #   assembly (t1-t9 / p-phase). Spilling it keeps every recursion-hot buffer in smem and
    #   is the single highest-payoff cold sub-band (30-45% of the body arena).
    self.gen_add_code_line("template <typename T, bool SCRATCH_IN_SMEM = true, bool BC_IN_SMEM = true, bool TP_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # Inner owns the pool placement; the repoint goes FIRST, before the offset-derived
    # pointer declarations below, so every consumer (incl. the XImats helper's sincos
    # scratch) follows the placement. Mirrors fdsva_so_device / crba_inner.
    self.gen_add_code_line("if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }")
    # Load/update XImats INSIDE the inner, AFTER the repoint, so its s_temp-backed
    # sincos scratch follows the SCRATCH_IN_SMEM placement (no caller-side repoint).
    self.gen_load_update_XImats_helpers_function_call()


    # MEMORY LAYOUT (s_temp):
        # Xdown(36*NJ)/
            # vJ(6 * NJ)/f(6*NJ)/ICT_S(6*NJ) & 
            # v(6 * NJ)/psid_Sd(6*NJ) & 
            # Sd(6 * NJ) & 
            # aJ(6 * NJ)/IC_v (6 * NJ)/IC_S (6*NJ)/T1 (6*NJ) & 
            # a(6 * NJ)/IC_psid (6 * NJ) & 
            # psid(6 * NJ)
        # IC (36 * NJ)
        # I_Xup (36 * NJ)/
            # S (6*NJ) & 
            # psidd (6 * NJ) & 
            # a_world (6)
            # T2 (6*NJ)
            # T3 (6*NJ)
            # T4 (6*NJ)
        # B_IC_S (36*NJ)/D3 (36*NJ)   <- anchored on S (stable), NOT on BC
        # crm_v (36 * NJ)/crm_S (36*NJ)
        # crf_v (36 * NJ)/crf_S (36*NJ)
        # crm_psid (36 * NJ)/crf_S_IC (36*NJ)
        # crf_psid (36 * NJ)/D4 (36*NJ)
        # icrf_f (36 * NJ)/D1 (36*NJ)
        # D2 (36*NJ)
        # Xup(36*NJ)/t - t1/t2/t3/t4/t5/t6/t7/t8/t9 [(len(jids_a) * 36]/p1 & p2 & p3 & p4 & p5 & p6 ([len(jids_a) * 6]*6)
        # BC (36 * NJ)   <- LAST slab (top of arena); overlays Xup (forward-dead) & t/p
        #                   (backward); surgical BC_IN_SMEM=false truncates exactly this.


    jids_a, ancestors = self.robot.get_jid_ancestor_ids(include_joint=True)
    var_offset = len(jids_a)
    vars = [
        '// Relevant Tensors in the order they appear',
        '// d_workspace: at TIER_SHARED unused; at the surgical BC rung only BC repoints here;',
        '// at the whole-arena rung s_temp was already repointed to it above.',
        'T *I = s_XImats + XIMAT_SIZE*NUM_BODIES;', # Inertia Matrices (6x6 for each joint)
        f'T *Xup = s_temp + 11*XIMAT_SIZE*NUM_BODIES;', # Spatial Transforms from parent to child (6x6 for each joint)
        'T *IC = s_temp + XIMAT_SIZE*NUM_BODIES;', # Centroidal Inertia (6x6 for each joint)
        # Xup, I_Xup done being used
        'T *Xdown = s_temp;\n', # Spatial Transforms from child to parent (6x6 for each joint)
        'T *S = IC + XIMAT_SIZE*NUM_BODIES;', # Transformed Joint Subspace Tensors (6x1 for each joint)
        # Xdown done being used
        'T *vJ = Xdown;', # Non-propogated Joint Spatial velocities (6x1 for each joint),
        'T *v = vJ + 6*NUM_BODIES;', # Joint Spatial velocities (6x1 for each joint),
        'T *Sd = v + 6*NUM_BODIES;', # Time derivative of Joint subspace tensor due to each joint moving (6x1 for each joint),
        'T *aJ = Sd + 6*NUM_BODIES;', # Non-propogated Joint Spatial accelerations (6x1 for each joint),
        'T *a = aJ + 6*NUM_BODIES;', # Joint Spatial accelerations (6x1 for each joint),
        'T *psid = a + 6*NUM_BODIES;', # Time derivative of joint subspace tensor due to each joint's parent moving (6x1 for each joint),
        'T *psidd = S + 6*NUM_BODIES;', # 2nd Time derivative of joint subspace tensor due to each joint's parent moving (6x1 for each joint),
        'T *a_world = psidd + 6*NUM_BODIES;', # Acceleration of the world frame (6x1)',
        'T *f = vJ;', # Joint Spatial forces (6x1 for each joint),
        # De-alias: B_IC_S/D3 anchor on the stable hot buffer S (NOT on BC). This keeps
        # the whole hot matrix chain (B_IC_S..D2, then the t/p backward region) at fixed
        # smem offsets that are independent of where BC lives. BC itself is relocated to
        # the very TOP of the arena (after the t/p region, see below), so the surgical
        # BC_IN_SMEM=false rung can shrink the smem arena by exactly BC's tail slab.
        'T *B_IC_S = S + 30*NUM_BODIES + 6;', # Body coriolis tensor wrt joint subspace (6x6 for each joint)',

        '\n\n',
        '// Temporary Variables for Computations',
        'T *I_Xup = S;', # Temporary to compute IC for I * Xup (6x6 for each joint)
        'T *crm_v = B_IC_S + 36*NUM_BODIES;', # Motion cross product of v (6x6 for each joint)
        'T *crf_v = crm_v + 36*NUM_BODIES;', # Force cross product of v (6x6 for each joint)',
        'T *IC_v = aJ;', # IC @ v (6x1 for each joint)',
        'T *crm_S = crm_v;', # Motion cross product of S (6x6 for each joint),
        'T *crf_S = crf_v;', # Force cross product of S (6x6 for each joint)',
        'T *IC_S = IC_v;', # IC @ S (6x1 for each joint)',
        'T *crm_psid = crf_v + 36*NUM_BODIES;', # Motion cross product of psid (6x6 for each joint)',
        'T *crf_psid = crm_psid + 36*NUM_BODIES;', # Force cross product of psid (6x6 for each joint)',
        'T *IC_psid = a;', # IC @ psid (6x6 for each joint)',
        'T *icrf_f = crf_psid + 36*NUM_BODIES;', # icrf(f) (6x6 for each joint)',
        'T *psid_Sd = v;', # psid + Sd (6x1 for each joint)',
        'T *ICT_S = f;', # IC^T @ S (6x1 for each joint)',

        '\n\n',
        '// Main Temporary Tensors For Backward Pass',
        'T *T1 = IC_S;', # Temporary for IC @ S (6x1 for each joint)',
        'T *T2 = a_world + 6;', # Temporary for -BC.T @ S (6x1 for each joint)',
        'T *T3 = T2 + 6*NUM_BODIES;', # Temporary matrix (6x1 for each joint)',
        'T *T4 = T3 + 6*NUM_BODIES;', # Temporary matrix (6x1 for each joint)',
        'T *D1 = icrf_f;', # Temporary D1 tensor (6x6 for each joint)',
        'T *D2 = D1 + 36*NUM_BODIES;', # Temporary D2 tensor (6x6 for each joint)',
        'T *D3 = B_IC_S;', # Temporary D3 tensor - same as B(IC, S) (6x6 for each joint)',
        'T *D4 = crf_psid;', # Temporary D4 tensor (6x6 for each joint)',
        # tp_anchor is the FIXED in-smem end of the recursion-hot chain (just past D2). The
        # ancestor-pair scratch t/p1..p6 (36*var_offset floats) anchors here when in smem.
        # Holding this anchor stable (independent of where t/p actually lives) lets the
        # TP_IN_SMEM=false rung relocate t/p to d_workspace while BC re-bases off this same
        # in-smem anchor — so the hot chain below is byte-identical regardless of the t/p
        # placement, and the smem arena shrinks by exactly 36*var_offset when t/p spills.
        f'T *tp_anchor = D2 + 36*NUM_BODIES;',
        f'T *t = tp_anchor;', # Temporary outer product tensor for t1-t9 (6x6 for each joint and its ancestors)',
        # Surgical t/p spill: route the ancestor-pair scratch to d_workspace. t/p is DEAD
        # through the whole forward sweep + D-matrix build (written/read ONLY in the final
        # block-parallel t1-t9 / p-phase output assembly), and the t-loop distributes
        # ancestor-pairs across the block on disjoint t_index_map[jid][anc]*36 slices, so a
        # spilled (L2-pinned) access coalesces. Mutually exclusive with BC/whole-arena
        # spills per the body tier table (rung "output_tp": TP=F, BC=T, SCRATCH=T).
        'if constexpr (!TP_IN_SMEM) { t = d_workspace; }',
        'T *p1 = t;', # Temporary cross product vector for p1 (6x1 for each joint and its ancestors)',
        f'T *p2 = p1 + 6*{var_offset};', # Temporary cross product vector for p2 (6x1 for each joint and its ancestors)',
        f'T *p3 = p2 + 6*{var_offset};', # Temporary cross product vector for p3 (6x1 for each joint and its ancestors)',
        f'T *p4 = p3 + 6*{var_offset};', # Temporary cross product vector for p4 (6x1 for each joint and its ancestors)',
        f'T *p5 = p4 + 6*{var_offset};', # Temporary cross product vector for p5 (6x1 for each joint and its ancestors)',
        f'T *p6 = p5 + 6*{var_offset};', # Temporary cross product vector used in computation of d2tau_dqd2[ancestor, joint, joint] (6x1 for each joint and its ancestors)',
        'T *crf_S_IC = crm_psid;', # Cross product of S and IC (6x6 for each joint)',
        # Composite body-Coriolis Bias tensor (6x6 for each joint). It is the LAST 36*NB
        # slab of the SMEM arena, anchored on tp_anchor (the in-smem hot-chain end) plus the
        # in-smem t/p span. When TP_IN_SMEM (default) that span is 36*var_offset, so BC sits
        # exactly where the legacy `p6 + 6*var_offset` put it (byte-identical). When t/p
        # spills (TP_IN_SMEM=false) the in-smem span is 0, so BC slides DOWN to tp_anchor,
        # reclaiming the vacated 36*var_offset smem and shrinking the arena.
        # BC is cold: written in the forward IC/BC propagation and last read by the
        # T2/T3/T4/D2 tensors, then dead before the t/p backward loops. The high arena
        # region it occupies is shared with Xup (forward-dead by the time BC is written)
        # and the in-smem t/p region (backward-only, after BC is dead), so no live-range
        # overlap. Because BC is the literal top smem slab, the surgical BC_IN_SMEM=false
        # rung shrinks the arena by exactly 36*NB and truncates only BC's tail; every hot
        # buffer below keeps its address.
        f'T *BC = tp_anchor + (TP_IN_SMEM ? 36*{var_offset} : 0);',


        '\n\n',
        '// Final Tensors for Output',
        'T *d2tau_dq2 = s_idsva_so;', # Second positional derivative of the joint torques (NJxNJXNJ)',
        'T *d2tau_dqd2 = d2tau_dq2+ SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS;', # Second velocity derivative of the joint torques (NJxNJXNJ)',
        'T *d2tau_dvdq = d2tau_dqd2 + SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS;', # Cross velocity/position derivative of the joint torques (NJxNJXNJ)',
        'T *dM_dq = d2tau_dvdq + SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS;', # Positional Derivative of the mass matrix (NJxNJXNJ)',
    ]
    
    self.gen_add_code_lines(vars)

    # ---- Fixed-base mimic: internal NUM_BODIES-coordinate sweep + alpha fold ----
    # When the robot has mimic joints, NB > NV and multiple bodies share a project
    # v-slot. The whole assembly below is body(jid)-indexed and writes the output
    # with an OUTPUT STRIDE of SECOND_ORDER_COORDS. For mimic we run it in unique-
    # per-body INTERNAL coordinates (internal slot == body id, n_int = NB) into a
    # 4*NB^3 internal slab, then fold each axis to the reduced 4*NV^3 public output
    # with the alpha reduction R[i, v_slot(i)] += alpha_i (mirrors
    # RBDReference.idsva_so_body_frame's has_mimic path). To reuse the entire
    # assembly unchanged we (a) shadow SECOND_ORDER_COORDS = NB inside the inner so
    # every output-stride site uses the NB stride, and (b) repoint s_idsva_so (hence
    # d2tau_dq2/.../dM_dq) at the internal slab. The public NV^3 output pointer is
    # saved first; the fold writes it at the very end.
    is_mimic = self.robot_has_mimic_joints()
    if is_mimic:
        # Public (reduced 4*NV^3) output destination handed in by the caller.
        self.gen_add_code_line("T *s_idsva_so_public = s_idsva_so;")
        # Internal 4*NB^3 slab anchored at the top of the (NB-grown) arena, just
        # past BC (the legacy top slab). BC = tp_anchor + 36*var_offset when in smem.
        self.gen_add_code_line(f"T *s_idsva_so_internal = BC + 36*NUM_BODIES;")
        self.gen_add_code_line("s_idsva_so = s_idsva_so_internal;")
        # Shadow the output stride to NB for the whole assembly. This function-local
        # const shadows the global SECOND_ORDER_COORDS (= NV) inside the inner only.
        self.gen_add_code_line("const int SECOND_ORDER_COORDS = NUM_BODIES;")
        # Re-derive the output tensor pointers off the (now internal) s_idsva_so with
        # the NB stride (the earlier `vars` definitions used the NV-stride global).
        self.gen_add_code_line("d2tau_dq2 = s_idsva_so;")
        self.gen_add_code_line("d2tau_dqd2 = d2tau_dq2 + SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS;")
        self.gen_add_code_line("d2tau_dvdq = d2tau_dqd2 + SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS;")
        self.gen_add_code_line("dM_dq = d2tau_dvdq + SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS;")

    # Surgical spill: BC (36*NB) is a cold buffer (write-once, dead before the t1-t9/
    # p1-p6 hot loops, and not read by reference_order_output_repair), so at a spill
    # tier it can move to global d_workspace while the hot buffers stay in smem. BC is
    # now the LAST (top) 36*NB slab of the arena (anchored on p6, above), so the smem
    # arena can be allocated 36*NB smaller for this rung (see the kernel body's
    # smem_temp computation) and only BC's tail is truncated — the hot chain below is
    # untouched. d_workspace here is the typed BC slab passed by the kernel.
    # Contract: BC_IN_SMEM=false is only used with SCRATCH_IN_SMEM=true (hot stays
    # in smem, BC moves to d_workspace[0]). The deep rung instead uses
    # SCRATCH_IN_SMEM=false (whole arena, incl. BC, to d_workspace) with
    # BC_IN_SMEM=true. The two spill levers are mutually exclusive by construction
    # (see the body tier table in GRiDCodeGenerator.py: rung 2 picks BC=F+SCRATCH=T;
    # rung 3 picks SCRATCH=F+BC=T), so the two BC= writes can never both fire.
    self.gen_add_code_line("if constexpr (!BC_IN_SMEM) { BC = d_workspace; }")

    self.gen_add_code_line("// Initialize output tensor; optimized assembly paths only write structurally nonzero entries.")
    self.gen_add_parallel_loop('i', '4*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS')
    self.gen_add_code_line("s_idsva_so[i] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp([i for i in range(num_bodies)], NO_GRAD_FLAG = True)
    S_sign_cpp = self.gen_topology_S_sign_for_cpp([i for i in range(num_bodies)])
    parent_ind_cpp_for_jid = parent_ind_cpp


    # Compute Xup transformations
    self.gen_add_code_line("\n")
    self.gen_add_code_line("// Compute Xup - parent to child transformation matrices")
    # If parent is base, Copy X to Xup - X matrices always 6x6
    if self.robot.is_serial_chain():
        self.gen_add_code_line('#pragma unroll')
        self.gen_add_code_line('for (int jid = 0; jid < NUM_BODIES; ++jid) {', 1)
        self.gen_add_code_line('// Compute Xup[joint]')
        self.gen_add_code_line('int X_idx = jid*XIMAT_SIZE;')
        self.gen_add_parallel_loop('i','XIMAT_SIZE')
        self.gen_add_code_line(f'if ({parent_ind_cpp } == -1) Xup[X_idx + i] = s_XImats[X_idx + i]; // Parent is base')
        self.gen_add_code_line(f'else matmul<T>(i, &Xup[{parent_ind_cpp} * XIMAT_SIZE], &s_XImats[X_idx], &Xup[X_idx], XIMAT_SIZE, 0);')
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_add_end_control_flow()
    else:
        for bfs_level in range(n_bfs_levels):
            inds = self.robot.get_ids_by_bfs_level(bfs_level)
            self.gen_add_code_line(f'// Compute Xup for bfs_level {bfs_level}')
            self.gen_add_parallel_loop('i', str(36*len(inds)))
            if len(inds) > 1: 
                    select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                    jid_cpp = "jid"
                    level_parent_ind_cpp = parent_ind_cpp_for_jid
                    self.gen_add_multi_threaded_select("(i)", "<", [str((idx+1)*36) for idx, jid in enumerate(inds)], select_var_vals)
            else:
                jid_cpp = str(inds[0])
                level_parent_ind_cpp = str(self.robot.get_parent_id(inds[0]))
            self.gen_add_code_line(f'int X_idx = {jid_cpp}*XIMAT_SIZE;')
            if bfs_level == 0: self.gen_add_code_line(f'Xup[X_idx + i % XIMAT_SIZE] = s_XImats[X_idx + i % XIMAT_SIZE]; // Parent is base')
            else: self.gen_add_code_line(f'matmul<T>(i % 36, &Xup[{level_parent_ind_cpp} * XIMAT_SIZE], &s_XImats[X_idx], &Xup[X_idx], XIMAT_SIZE, 0);')
            self.gen_add_end_control_flow()
            self.gen_add_sync()
            

    # Next compute IC - Centroidal Rigid Body Inertia
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line("// Compute IC - Centroidal Rigid Body Inertia")
    # First I @ Xup
    self.gen_add_code_line('// First I @ Xup')
    self.gen_add_parallel_loop('i','XIMAT_SIZE*NUM_BODIES')
    self.gen_add_code_line('// All involved matrices are 6x6')
    self.gen_add_code_line('matmul<T>(i, Xup, I, I_Xup, 36, false);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # Next Xup.T @ I
    self.gen_add_code_line('// Next Xup.T @ I')
    self.gen_add_parallel_loop('i','XIMAT_SIZE*NUM_BODIES')
    self.gen_add_code_line('// All involved matrices are 6x6')
    self.gen_add_code_line('int mat_idx = (i / 36) * 36;')
    self.gen_add_code_line("matmul_trans<T>(i % 36, &Xup[mat_idx], &I_Xup[mat_idx], &IC[mat_idx], 'a');")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Next compute Xdown transformations
    # Just the transpose of internal 3x3 submatrices
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line("// Compute Xdown - child to parent transformation matrices")
    self.gen_idsva_so_xdown_plucker_inverse("fixed_parallel")

    # Transform S
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Transform S')
    if self.robot.robot_has_skew_axis():
        # Tier B (skew): S_world = Xdown @ S_dense (per-body dense column). Bake a
        # per-body dense S table; cardinal bodies keep their single signed-index
        # column (so a mixed robot still picks the indexed read per body, and an
        # all-cardinal robot never reaches this branch -> byte-identical).
        flat_S = [c for jid in range(num_bodies) for c in self.robot._get_flat_S_by_id(jid)]
        is_skew = [0 if self.robot.S_is_cardinal_by_id(jid) else 1 for jid in range(num_bodies)]
        self.gen_add_code_line("static const T so_S_vec[] = { " + ", ".join("static_cast<T>(" + repr(float(v)) + ")" for v in flat_S) + " };")
        self.gen_add_code_line("static const int so_is_skew[] = { " + ", ".join(str(v) for v in is_skew) + " };")
        self.gen_add_parallel_loop('i','6*NUM_BODIES')
        self.gen_add_code_line('int jid = i / 6; int row = i % 6;')
        self.gen_add_code_line("if (so_is_skew[jid]) {", True)
        self.gen_add_code_line("T a = static_cast<T>(0); for (int k = 0; k < 6; ++k) a += Xdown[jid*XIMAT_SIZE + k*6 + row] * so_S_vec[jid*6 + k]; S[i] = a;")
        self.gen_add_end_control_flow()
        self.gen_add_code_line(f"else {{ S[i] = ({S_sign_cpp}) * Xdown[jid*XIMAT_SIZE + {S_ind_cpp}*6 + row]; }}")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
    else:
        self.gen_add_parallel_loop('i','6*NUM_BODIES')
        self.gen_add_code_line('int jid = i / 6;')
        self.gen_add_code_line(f'S[i] = ({S_sign_cpp}) * Xdown[jid*XIMAT_SIZE + {S_ind_cpp}*6 + (i % 6)];')
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    # Compute vJ = S @ qd & aJ = S @ qdd in parallel
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Compute vJ = S @ qd & aJ = S @ qdd')
    if is_mimic:
        # Mimic-aware velocity read: body `joint`'s spatial joint velocity is
        # alpha_joint * qd[v_slot(joint)]. body_vslot maps each body to its
        # (possibly shared) reduced v-slot; body_alpha is the mimic multiplier
        # (1.0 for non-mimic bodies). This both fixes the OOB s_qd[joint] read
        # (joint == body id can exceed NV-1 for a mimic body) and applies the
        # multiplier exactly as RBDReference does. Mirrors the oracle's
        # `_qd = alpha_i * qd[inds_v_true]`.
        body_vslot = [self._v_slot_cpp(b) for b in range(num_bodies)]
        body_alpha = [self._alpha_for_jid(b) for b in range(num_bodies)]
        self.gen_add_code_line(
            "static const int body_vslot[] = { " + ", ".join(map(str, body_vslot)) + " };")
        self.gen_add_code_line(
            "static const T body_alpha[] = { " + ", ".join(
                "static_cast<T>(" + repr(a) + ")" for a in body_alpha) + " };")
    self.gen_add_parallel_loop('i','2*6*NUM_BODIES')
    self.gen_add_code_line('int joint = i / 6;')
    if is_mimic:
        self.gen_add_code_line('if (joint < NUM_BODIES) vJ[i] = S[i] * (body_alpha[joint] * s_qd[body_vslot[joint]]);')
        self.gen_add_code_line('else { int jj = joint - NUM_BODIES; aJ[i - 6*NUM_BODIES] = S[i - 6*NUM_BODIES] * (body_alpha[jj] * s_qdd[body_vslot[jj]]); }')
    else:
        self.gen_add_code_line('if (joint < NUM_BODIES) vJ[i] = S[i] * s_qd[joint];')
        self.gen_add_code_line('else aJ[i - 6*NUM_BODIES] = S[i - 6*NUM_BODIES] * s_qdd[joint - NUM_BODIES];')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Compute v = v[parent] + vJ
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Compute v = v[parent] + vJ')
    if self.robot.is_serial_chain():
        self.gen_add_code_line('#pragma unroll')
        self.gen_add_code_line('for (int jid = 0; jid < NUM_BODIES; ++jid) {', 1)
        self.gen_add_parallel_loop('i','6')
        self.gen_add_code_line(f'if ({parent_ind_cpp} == -1) v[jid*6 + i] = vJ[jid*6 + i];')
        self.gen_add_code_line(f'else v[jid*6 + i] = v[{parent_ind_cpp}*6 + i] + vJ[jid*6 + i];')
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_add_end_control_flow()
    else:
        for bfs_level in range(n_bfs_levels):
            inds = self.robot.get_ids_by_bfs_level(bfs_level)
            self.gen_add_code_line(f'// Compute v for bfs_level {bfs_level}')
            self.gen_add_parallel_loop('i', str(6*len(inds)))
            if len(inds) > 1: 
                    select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                    jid_cpp = "jid"
                    level_parent_ind_cpp = parent_ind_cpp_for_jid
                    self.gen_add_multi_threaded_select("(i)", "<", [str((idx+1)*6) for idx, jid in enumerate(inds)], select_var_vals)
            else:
                jid_cpp = str(inds[0])
                level_parent_ind_cpp = str(self.robot.get_parent_id(inds[0]))
            self.gen_add_code_line(f'int idx = i % 6;')
            if bfs_level == 0: self.gen_add_code_line(f'v[{jid_cpp}*6 + idx] = vJ[{jid_cpp}*6 + idx]; // Parent is base')
            else: self.gen_add_code_line(f'v[{jid_cpp}*6 + idx] = v[{level_parent_ind_cpp}*6 + idx] + vJ[{jid_cpp}*6 + idx];')
            self.gen_add_end_control_flow()
            self.gen_add_sync()

    # Finish aJ += crm(v[parent])@vJ
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Finish aJ += crm(v[parent])@vJ')
    self.gen_add_code_line('// For base, v[parent] = 0')
    self.gen_add_parallel_loop('i','6*NUM_BODIES')
    self.gen_add_code_line('int jid = i / 6;')
    self.gen_add_code_line('int index = i % 6;')
    self.gen_add_code_line(f'if ({parent_ind_cpp_for_jid} != -1) aJ[i] += crm_mul<T>(index, &v[{parent_ind_cpp_for_jid}*6], &vJ[jid*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Compute Sd = crm(v) @ S & psid = crm(v[parent]) @ S
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Compute Sd = crm(v) @ S & psid = crm(v[parent]) @ S')
    self.gen_add_code_line('// For base, v[parent] = 0')
    self.gen_add_parallel_loop('i','2*6*NUM_BODIES')
    self.gen_add_code_line('int jid = (i / 6) % NUM_BODIES;')
    self.gen_add_code_line('int index = i % 6;')
    self.gen_add_code_line('if (i < 6*NUM_BODIES) Sd[i] = crm_mul<T>(index, &v[jid*6], &S[jid*6]);')
    self.gen_add_code_line('else {', True)
    self.gen_add_code_line(f'if ({parent_ind_cpp_for_jid} == -1) psid[jid*6 + index] = 0;')
    self.gen_add_code_line(f'else psid[i - 6 * NUM_BODIES] = crm_mul<T>(index, &v[{parent_ind_cpp_for_jid}*6], &S[jid*6]);')   
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Compute a = a[parent] + aJ
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Compute a = a[parent] + aJ')
    if self.robot.is_serial_chain():
        self.gen_add_code_line('#pragma unroll')
        self.gen_add_code_line('for (int jid = 0; jid < NUM_BODIES; ++jid) {', 1)
        self.gen_add_parallel_loop('i','6')
        self.gen_add_code_line(f"if ({parent_ind_cpp} == -1) a[jid*6+ i] = aJ[jid*6 + i] - gravity * (i == 5); // Base joint's parent is the world")
        self.gen_add_code_line(f'else a[jid*6 + i] = a[{parent_ind_cpp}*6 + i] + aJ[jid*6 + i];')
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_add_end_control_flow()
    else:
        for bfs_level in range(n_bfs_levels):
            inds = self.robot.get_ids_by_bfs_level(bfs_level)
            self.gen_add_code_line(f'// Compute a for bfs_level {bfs_level}')
            self.gen_add_parallel_loop('i', str(6*len(inds)))
            if len(inds) > 1: 
                    select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                    jid_cpp = "jid"
                    level_parent_ind_cpp = parent_ind_cpp_for_jid
                    self.gen_add_multi_threaded_select("(i)", "<", [str((idx+1)*6) for idx, jid in enumerate(inds)], select_var_vals)
            else:
                jid_cpp = str(inds[0])
                level_parent_ind_cpp = str(self.robot.get_parent_id(inds[0]))
            self.gen_add_code_line(f'int idx = i % 6;')
            if bfs_level == 0: self.gen_add_code_line(f"a[{jid_cpp}*6+ idx] = aJ[{jid_cpp}*6 + idx] - gravity * (idx == 5); // Base joint's parent is the world")
            else: self.gen_add_code_line(f'a[{jid_cpp}*6 + idx] = a[{level_parent_ind_cpp}*6 + idx] + aJ[{jid_cpp}*6 + idx];')
            self.gen_add_end_control_flow()
            self.gen_add_sync()
        

    # Initialize a_world
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Initialize a_world')
    self.gen_add_parallel_loop('i','6')
    self.gen_add_code_line('if (i < 5) a_world[i] = 0;')
    self.gen_add_code_line('else a_world[5] = -gravity; // a_base = gravity_vec[5] = -GRAVITY = +9.81 (gravity=-9.81)')
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    
    # Compute psidd = crm(a[parent])@S + crm(v[parent])@psid & IC_v
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Compute psidd = crm(a[parent])@S + crm(v[:,i])@psid[:,i] & IC @ v (for BC) in parallel')
    self.gen_add_parallel_loop('i','2*6*NUM_BODIES')
    self.gen_add_code_line('int jid = (i / 6) % NUM_BODIES;')
    self.gen_add_code_line('int index = i % 6;')
    self.gen_add_code_line('if (i < 6*NUM_BODIES) {', True)
    self.gen_add_code_line(f'if ({parent_ind_cpp_for_jid} == -1) psidd[i] = crm_mul<T>(index, a_world, &S[jid*6]);')
    self.gen_add_code_line(f'else psidd[i] = crm_mul<T>(index, &a[{parent_ind_cpp_for_jid}*6], &S[jid*6]) + crm_mul<T>(index, &v[{parent_ind_cpp_for_jid}*6], &psid[jid*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_code_line(f'else IC_v[i - 6*NUM_BODIES] = dot_prod<T, 6, 6, 1>(&IC[index + jid*36], &v[jid*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Begin BC Computation
    # First Compute crm(v) & crf(v)
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Need crm(v), crf(v) for BC computation')
    self.gen_add_parallel_loop('i','2*36*NUM_BODIES')
    self.gen_add_code_line('int jid = (i / 36) % NUM_BODIES;')
    self.gen_add_code_line('int col = (i / 6) % 6;')
    self.gen_add_code_line('int row = i % 6;')
    self.gen_add_code_line('if (i < 36*NUM_BODIES) crm_v[i] = crm<T>(i % 36, &v[jid*6]);')
    self.gen_add_code_line('else crf_v[(jid*36) + row*6 + col] = -crm<T>(i % 36, &v[jid*6]); // crf is negative tranpose of crm')
    self.gen_add_end_control_flow()
    self.gen_add_sync()


    # Finish BC = crf(v) @ IC + icrf(IC @ v) - IC @ crm(v)
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Finish BC = crf(v) @ IC + icrf(IC @ v) - IC @ crm(v)')
    self.gen_add_parallel_loop('i','36*NUM_BODIES')
    self.gen_add_code_line('int jid = i / 36;')
    self.gen_add_code_line('int row = i % 6;')
    self.gen_add_code_line('int col_idx = (i / 6) * 6;')
    self.gen_add_code_line('BC[i] = dot_prod<T, 6, 6, 1>(&crf_v[jid*36 + row], &IC[col_idx]) +')
    self.gen_add_code_line('        icrf<T>(i % 36, &IC_v[jid*6]) -')
    self.gen_add_code_line('        dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &crm_v[col_idx]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Next f = IC @ a + crf(v) @ IC @ v
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Compute f = IC @ a + crf(v) @ IC @ v')
    self.gen_add_parallel_loop('i','6*NUM_BODIES')
    self.gen_add_code_line('int jid = i / 6;')
    self.gen_add_code_line('int row = i % 6;')
    self.gen_add_code_line('f[i] = dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &a[jid*6]) +')
    self.gen_add_code_line('        dot_prod<T, 6, 6, 1>(&crf_v[jid*36 + row], &IC_v[jid*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Forward Pass Completed
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Forward Pass Completed')
    self.gen_add_code_line('// Now compute the backward pass')


    # Compute IC[parent] += IC[i], BC[parent] += BC[i], f[parent] += f[i]
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Compute IC[parent] += IC[i], BC[parent] += BC[i], f[parent] += f[i]')
    if self.robot.is_serial_chain():
        self.gen_add_code_line('#pragma unroll')
        self.gen_add_code_line('for (int jid = NUM_BODIES-1; jid > 0; --jid) {', 1)
        self.gen_add_parallel_loop('i','36*2 + 6')
        self.gen_add_code_line(f'if ({parent_ind_cpp} != -1) {{', True)
        self.gen_add_code_line(f'if (i < 36) IC[{parent_ind_cpp}*36 + i] += IC[jid*36 + i];')
        self.gen_add_code_line(f'else if (i < 36*2) BC[{parent_ind_cpp}*36 + i - 36] += BC[jid*36 + i - 36];')
        self.gen_add_code_line(f'else f[{parent_ind_cpp}*6 + i - 36*2] += f[jid*6 + i - 36*2];')
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_add_end_control_flow()
    else:
        for bfs_level in range(n_bfs_levels-1, 0, -1):
            inds = self.robot.get_ids_by_bfs_level(bfs_level)
            self.gen_add_code_line(f'// Compute propogations for bfs_level {bfs_level}')
            for jid in inds:
                parent_ind = self.robot.get_parent_id(jid)
                if parent_ind == -1:
                    continue
                self.gen_add_code_line(
                    f'// Accumulate joint {jid} into parent {parent_ind}'
                )
                self.gen_add_parallel_loop('i','36*2 + 6')
                self.gen_add_code_line('int idx = i;')
                self.gen_add_code_line(f'if (idx < 36) IC[{parent_ind}*36 + idx] += IC[{jid}*36 + idx];')
                self.gen_add_code_line(f'else if (idx < 36*2) BC[{parent_ind}*36 + idx - 36] += BC[{jid}*36 + idx - 36];')
                self.gen_add_code_line(f'else f[{parent_ind}*6 + idx - 36*2] += f[{jid}*6 + idx - 36*2];')
                self.gen_add_end_control_flow()
                self.gen_add_sync()

    # Begin B(IC, S) & B(IC, psid) computation
    # First compute crm(S), crf(S), IC @ S && crm(psid), crf(psid), IC @ psid, icrf(f), psid+Sd
    self.gen_add_code_line("\n\n")
    self.gen_add_code_line('// Need crm(S), crf(S), IC@S, crm(psid), crf(psid), IC@psid for B computations & icrf(f), psid+Sd for T3,T4')
    self.gen_add_parallel_loop('i','5*36*NUM_BODIES + 3*6*NUM_BODIES')
    self.gen_add_code_line('int jid = (i / 36) % NUM_BODIES;')
    self.gen_add_code_line('int jidMatmul = (i / 6) % NUM_BODIES;')
    self.gen_add_code_line('int col = (i / 6) % 6;')
    self.gen_add_code_line('int row = i % 6;')
    self.gen_add_code_line('if (i < 36*NUM_BODIES) crm_S[i] = crm<T>(i % 36, &S[jid*6]);')
    self.gen_add_code_line('else if (i < 2*36*NUM_BODIES) crf_S[(jid*36) + row*6 + col] = -crm<T>(i % 36, &S[jid*6]); // crf is negative tranpose of crm')
    self.gen_add_code_line('else if (i < 3*36*NUM_BODIES) crm_psid[jid*36 + col*6 + row] = crm<T>(i % 36, &psid[jid*6]);')
    self.gen_add_code_line('else if (i < 4*36*NUM_BODIES) crf_psid[(jid*36) + row*6 + col] = -crm<T>(i % 36, &psid[jid*6]); // crf is negative tranpose of crm')
    self.gen_add_code_line('else if (i < 5*36*NUM_BODIES) icrf_f[i - 4*36*NUM_BODIES] = icrf<T>(i % 36, &f[jid*6]);')
    self.gen_add_code_line('else if (i < 5*36*NUM_BODIES + 6*NUM_BODIES) IC_S[i - 5*36*NUM_BODIES] = dot_prod<T, 6, 6, 1>(&IC[row + jidMatmul*36], &S[jidMatmul*6]);')
    self.gen_add_code_line('else if (i < 5*36*NUM_BODIES + 2*6*NUM_BODIES) psid_Sd[i - 5*36*NUM_BODIES - 6*NUM_BODIES] = psid[i - 5*36*NUM_BODIES - 6*NUM_BODIES] + Sd[i - 5*36*NUM_BODIES - 6*NUM_BODIES];')
    self.gen_add_code_line('else IC_psid[i - 5*36*NUM_BODIES - 2*6*NUM_BODIES] = dot_prod<T, 6, 6, 1>(&IC[row + jidMatmul*36], &psid[jidMatmul*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Finish B_IC_S, Start D2
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Finish B_IC_S, Start D2')
    self.gen_add_code_line('// B_IC_S = crf(S) @ IC + icrf(IC @ S) - IC @ crm(S)')
    self.gen_add_code_line('// D2 = crf(psid) @ IC + icrf(IC @ psid) - IC @ crm(psid)')
    self.gen_add_parallel_loop('i','2*36*NUM_BODIES')
    self.gen_add_code_line('int jid = (i / 36) % NUM_BODIES;')
    self.gen_add_code_line('int row = i % 6;')
    self.gen_add_code_line('int col = (i / 6) % 6;')
    self.gen_add_code_line('if (i < 36*NUM_BODIES) {', True)
    self.gen_add_code_line('B_IC_S[i] = dot_prod<T, 6, 6, 1>(&crf_S[jid*36 + row], &IC[jid*36 + col*6]) + ')
    self.gen_add_code_line('            icrf<T>(i % 36, &IC_S[jid*6]) -') 
    self.gen_add_code_line('            dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &crm_S[jid*36 + col*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_code_line('else {', True)
    self.gen_add_code_line('D2[i - 36*NUM_BODIES] = dot_prod<T, 6, 6, 1>(&crf_psid[jid*36 + row], &IC[jid*36 + col*6]) + ')
    self.gen_add_code_line('                                icrf<T>(i % 36, &IC_psid[jid*6]) -') 
    self.gen_add_code_line('                                dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &crm_psid[jid*36 + col*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Compute T2 = -BC.T @ S & T3 = BC @ psid + IC @ psidd + icrf(f) @ S, & T4 = BC @ S + IC @ (psid + Sd), & IC.T @ S for D4
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Compute T2 = -BC.T @ S')
    self.gen_add_code_line('// Compute T3 = BC @ psid + IC @ psidd + icrf(f) @ S')
    self.gen_add_code_line('// Compute T4 = BC @ S + IC @ (psid + Sd)')
    self.gen_add_code_line('// Compute IC.T @ S for D4')
    self.gen_add_parallel_loop('i','4*6*NUM_BODIES')
    self.gen_add_code_line('int jid = (i / 6) % NUM_BODIES;')
    self.gen_add_code_line('int row = i % 6;')
    self.gen_add_code_line('if (i < 6*NUM_BODIES) T2[i] = -dot_prod<T, 6, 1, 1>(&BC[jid*36 + row*6], &S[jid*6]);')
    self.gen_add_code_line('else if (i < 2*6*NUM_BODIES) {', True)
    self.gen_add_code_line('T3[i - 6*NUM_BODIES] = dot_prod<T, 6, 6, 1>(&BC[jid*36 + row], &psid[jid*6]) +')
    self.gen_add_code_line('                    dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &psidd[jid*6]) +')
    self.gen_add_code_line('                    dot_prod<T, 6, 6, 1>(&icrf_f[jid*36 + row], &S[jid*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_code_line('else if (i < 3*6*NUM_BODIES) {', True)
    self.gen_add_code_line('T4[i - 2*6*NUM_BODIES] = dot_prod<T, 6, 6, 1>(&BC[jid*36 + row], &S[jid*6]) +')
    self.gen_add_code_line('                    dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &psid_Sd[jid*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_code_line('else ICT_S[i - 3*6*NUM_BODIES] = dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &S[jid*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Compute D1..D4
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Compute D1, D2, D4, crf_S_IC')
    self.gen_add_parallel_loop('i','4*36*NUM_BODIES')
    self.gen_add_code_line('int jid = (i / 36) % NUM_BODIES;')
    self.gen_add_code_line('int row = i % 6;')
    self.gen_add_code_line('int col = (i / 6) % 6;')
    self.gen_add_code_line('if (i < 36*NUM_BODIES) {', True)
    self.gen_add_code_line('D1[i] = dot_prod<T, 6, 6, 1>(&crf_S[jid*36 + row], &IC[jid*36 + col*6]) -')
    self.gen_add_code_line('        dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &crm_S[jid*36 + col*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_code_line('else if (i < 2*36*NUM_BODIES) {', True)
    self.gen_add_code_line('D2[i - 36*NUM_BODIES] += dot_prod<T, 6, 6, 1>(&crf_S[jid*36 + row], &BC[jid*36 + col*6]) -')
    self.gen_add_code_line('                        dot_prod<T, 6, 6, 1>(&BC[jid*36 + row], &crm_S[jid*36 + col*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_code_line('else if (i < 3*36*NUM_BODIES) D4[i - 2*36*NUM_BODIES] = icrf<T>(i % 36, &ICT_S[jid*6]);')
    self.gen_add_code_line('else crf_S_IC[i - 3*36*NUM_BODIES] = dot_prod<T, 6, 6, 1>(&crf_S[jid*36 + row], &IC[jid*36 + col*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    


    # Compute t1
    self.gen_add_code_line('\n\n')
    jids_a, ancestors = self.robot.get_jid_ancestor_ids(include_joint=True)
    self.gen_add_code_line('// Compute t1 = outer(S[j], psid[ancestor])')
    self.gen_add_code_line('// t1[j][k] is stored at t[((j*(j+1)/2) + k)*36]')
    self.gen_add_code_line(f'static const int jids[] = {{ {", ".join(map(str, jids_a))} }}; // Joints with ancestor at equivalent index of ancestors_j')
    self.gen_add_code_line(f'static const int ancestors_j[] = {{ {", ".join(map(str, ancestors))} }}; // Joint or ancestor of joint at equivalent index of jids_a')

    # Create t indexing map. Sized by NJ (raw joint count), NOT NV (DoF count),
    # because get_jid_ancestor_ids returns joint IDs in range [0, NJ). When the
    # mimic-aware URDFParser keeps fixed/mimic joints (e.g. h1_2 fixed-base:
    # NJ=51 > NV=39), indexing by jid into an NV-sized map raises IndexError.
    # S/psid/etc. are also jid-indexed downstream, so we keep this jid-indexed
    # too rather than rewriting to v-indexed (see Option B in bug notes).
    NJ = self.robot.get_num_joints()
    # Initialize the matrix with -1
    t_index_map = [[-1 for _ in range(NJ)] for _ in range(NJ)]

    # Fill in the map with t_idx
    for t_idx, (j, a) in enumerate(zip(jids_a, ancestors)):
        t_index_map[j][a] = t_idx

    # Emit CUDA code (NJ x NJ to match Python-side sizing above)
    self.gen_add_code_line("const int t_index_map[{}][{}] = {{".format(NJ, NJ))
    for row in t_index_map:
        self.gen_add_code_line("    { " + ", ".join("{:2}".format(x) for x in row) + " },")
    self.gen_add_code_line("};")
    
    self.gen_add_parallel_loop('i',f'{len(jids_a)}*36')
    self.gen_add_code_line('int jid = jids[i / 36];')
    self.gen_add_code_line('int ancestor_j = ancestors_j[i / 36];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line('outerProduct<T>(&S[jid*6], &psid[ancestor_j*6], &t[t_idx], 6, 6, i%36);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Perform all computations with t1
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Perform all computations with t1')
    jids, ancestors, st = self.robot.get_jid_ancestor_st_ids(True) # Generate indices for the joint, ancestor, and subtree
    self.gen_add_code_line(f'static const int jids_compute[] = {{ {", ".join(map(str, jids))} }}; // Joints with ancestor at equivalent index of ancestors_j') 
    self.gen_add_code_line(f'static const int ancestors_j_compute[] = {{ {", ".join(map(str, ancestors))} }}; // Joint or ancestor of joint at equivalent index of jids')
    self.gen_add_code_line(f'static const int st[] = {{ {", ".join(map(str, st))} }}; // Subtree of joint at equivalent index of jids')
    self.gen_add_code_lines(['// d2tau_dvdq[child, joint, ancestor] = -np.dot(t1, D3[:, child])', \
                             '// d2tau_dq[joint, ancestor, child] = np.dot(t1, D2[:, child])', \
                             '// d2tau_dq[joint, child, ancestor] = -np.dot(t1, D2[:, child])', \
                             '// d2tau_dvdq[joint, child, ancestor] = np.dot(t1, D3[:, child])'])
    self.gen_add_parallel_loop('i',f'{4*len(jids)}')
    self.gen_add_code_line(f'int index = i % {len(jids)};')
    self.gen_add_code_line(f'int jid = jids_compute[index];')
    self.gen_add_code_line(f'int ancestor_j = ancestors_j_compute[index];')
    self.gen_add_code_line(f'int st_j = st[index];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line(f'if (i < {len(jids)}) d2tau_dvdq[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j * SECOND_ORDER_COORDS + jid] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);')
    self.gen_add_code_line(f'else if (i < {len(jids)*2} && jid != st_j) d2tau_dq2[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j * SECOND_ORDER_COORDS + ancestor_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D2[st_j*36]);')
    self.gen_add_code_line(f'else if (i < {len(jids)*3} && jid != st_j) d2tau_dq2[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j * SECOND_ORDER_COORDS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D2[st_j*36]);')
    self.gen_add_code_line(f'else if (jid != st_j) d2tau_dvdq[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j * SECOND_ORDER_COORDS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Compute t2
    self.gen_add_code_line('\n\n')
    jids_a, ancestors = self.robot.get_jid_ancestor_ids(include_joint=True)
    self.gen_add_code_line('// Compute t2 = outer(S[j], S[ancestor])')
    self.gen_add_code_line('// t2[j][k] is stored at t[((j*(j+1)/2) + k)*36]')
    self.gen_add_parallel_loop('i',f'{len(jids_a)}*36')
    self.gen_add_code_line('int jid = jids[i / 36];')
    self.gen_add_code_line('int ancestor_j = ancestors_j[i / 36];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line('outerProduct<T>(&S[jid*6], &S[ancestor_j*6], &t[t_idx], 6, 6, i%36);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Perform all computations with t2
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Perform all computations with t2')
    self.gen_add_code_lines(['// for ancestor d2tau_dqd[child, ancestor, joint] = -np.dot(t2, D3[child])', \
                             '// for joint d2tau_dqd[child, joint, joint] = -np.dot(t2, D1[child])', \
                             '// for child d2tau_dqd[joint, ancestor, child] = np.dot(t2, D3[child])', \
                             '// for ancestor d2tau_dqd[child, joint, ancestor] = -np.dot(t2, D3[child])', \
                             '// for child d2tau_dqd[joint, child, ancestor] = np.dot(t2, D3[child])', \
                             '// for child d2tau_dvdq[joint, ancestor, child] = np.dot(t2, D2[child])'])
    self.gen_add_parallel_loop('i',f'{5*len(jids)}')
    self.gen_add_code_line(f'int index = i % {len(jids)};')
    self.gen_add_code_line(f'int jid = jids_compute[index];')
    self.gen_add_code_line(f'int ancestor_j = ancestors_j_compute[index];')
    self.gen_add_code_line(f'int st_j = st[index];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line(f'if (i < {len(jids)} && ancestor_j < jid) d2tau_dqd2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + ancestor_j] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);')
    self.gen_add_code_line(f'else if (i < {len(jids)} && jid == ancestor_j) d2tau_dqd2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j * SECOND_ORDER_COORDS + jid] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);')
    self.gen_add_code_line(f'else if (i < {2*len(jids)} && jid != st_j) d2tau_dqd2[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j * SECOND_ORDER_COORDS + ancestor_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);')
    self.gen_add_code_line(f'else if (i < {3*len(jids)} && ancestor_j < jid) d2tau_dqd2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j * SECOND_ORDER_COORDS + jid] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);')
    self.gen_add_code_line(f'else if (i < {4*len(jids)} && jid != st_j) d2tau_dqd2[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j * SECOND_ORDER_COORDS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);')
    self.gen_add_code_line(f'else if (i >= {4*len(jids)} && jid != st_j) d2tau_dvdq[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j * SECOND_ORDER_COORDS + ancestor_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D2[st_j*36]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()



    # Compute t3
    self.gen_add_code_line('\n\n')
    jids_a, ancestors = self.robot.get_jid_ancestor_ids(include_joint=True)
    self.gen_add_code_line('// Compute t3 = outer(psid[j], psid[ancestor])')
    self.gen_add_code_line('// t3[j][k] is stored at t[((j*(j+1)/2) + k)*36]')
    self.gen_add_parallel_loop('i',f'{len(jids_a)}*36')
    self.gen_add_code_line('int jid = jids[i / 36];')
    self.gen_add_code_line('int ancestor_j = ancestors_j[i / 36];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line('outerProduct<T>(&psid[jid*6], &psid[ancestor_j*6], &t[t_idx], 6, 6, i%36);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Perform all computations with t3
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Perform all computations with t3')
    self.gen_add_code_lines(['// for joint d2tau_dqd[child, joint, ancestor] = -np.dot(t3, D3[:, st_j])', \
                             '// for ancestor d2tau_dqd[child, ancestor, joint] = -np.dot(t3, D3[:, st_j])'])
    self.gen_add_parallel_loop('i',f'{2*len(jids)}')
    self.gen_add_code_line(f'int index = i % {len(jids)};')
    self.gen_add_code_line(f'int jid = jids_compute[index];')
    self.gen_add_code_line(f'int ancestor_j = ancestors_j_compute[index];')
    self.gen_add_code_line(f'int st_j = st[index];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line(f'if (i < {len(jids)}) d2tau_dq2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j * SECOND_ORDER_COORDS + jid] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);')
    self.gen_add_code_line(f'else if (ancestor_j < jid) d2tau_dq2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + ancestor_j] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()


    # Compute t4
    self.gen_add_code_line('\n\n')
    jids_a, ancestors = self.robot.get_jid_ancestor_ids(include_joint=True)
    self.gen_add_code_line('// Compute t4 = outer(S[j], psidd[ancestor])')
    self.gen_add_code_line('// t4[j][k] is stored at t[((j*(j+1)/2) + k)*36]')
    self.gen_add_parallel_loop('i',f'{len(jids_a)}*36')
    self.gen_add_code_line('int jid = jids[i / 36];')
    self.gen_add_code_line('int ancestor_j = ancestors_j[i / 36];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line('outerProduct<T>(&S[jid*6], &psidd[ancestor_j*6], &t[t_idx], 6, 6, i%36);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Perform all computations with t4
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Perform all computations with t4')
    self.gen_add_code_lines(['// for child d2tau_dq[dd, cc, succ_j] += np.dot(t4, D1[:, succ_j])', \
                             '// for child d2tau_dq[dd, succ_j, cc] += np.dot(t4, D1[:, succ_j])'])
    self.gen_add_parallel_loop('i',f'{2*len(jids)}')
    self.gen_add_code_line(f'int index = i % {len(jids)};')
    self.gen_add_code_line(f'int jid = jids_compute[index];')
    self.gen_add_code_line(f'int ancestor_j = ancestors_j_compute[index];')
    self.gen_add_code_line(f'int st_j = st[index];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line(f'if (i < {len(jids)} && jid != st_j) d2tau_dq2[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j * SECOND_ORDER_COORDS + ancestor_j] += dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);')
    self.gen_add_code_line(f'else if (jid != st_j) d2tau_dq2[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j * SECOND_ORDER_COORDS + st_j] += dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()


    # Compute t5
    self.gen_add_code_line('\n\n')
    jids_a, ancestors = self.robot.get_jid_ancestor_ids(include_joint=True)
    self.gen_add_code_line('// Compute t5 = outer(S[j], (Sd+psid)[ancestor])')
    self.gen_add_code_line('// t5[j][k] is stored at t[((j*(j+1)/2) + k)*36]')
    self.gen_add_parallel_loop('i',f'{len(jids_a)}*36')
    self.gen_add_code_line('int jid = jids[i / 36];')
    self.gen_add_code_line('int ancestor_j = ancestors_j[i / 36];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line('outerProduct<T>(&S[jid*6], &psid_Sd[ancestor_j*6], &t[t_idx], 6, 6, i%36);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Perform all computations with t5
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Perform all computations with t5')
    self.gen_add_code_lines(['// for child d2tau_dvdq[dd, cc, succ_j] += np.dot(t5, D1[:, succ_j])'])
    self.gen_add_parallel_loop('i',f'{len(jids)}')
    self.gen_add_code_line(f'int index = i % {len(jids)};')
    self.gen_add_code_line(f'int jid = jids_compute[index];')
    self.gen_add_code_line(f'int ancestor_j = ancestors_j_compute[index];')
    self.gen_add_code_line(f'int st_j = st[index];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line(f'if (st_j != jid) d2tau_dvdq[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j * SECOND_ORDER_COORDS + ancestor_j] += dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()


    # Compute t6
    self.gen_add_code_line('\n\n')
    jids_a, ancestors = self.robot.get_jid_ancestor_ids(include_joint=True)
    self.gen_add_code_line('// Compute t6 = outer(S[ancestor], psid[joint])')
    self.gen_add_code_line('// t6[j][k] is stored at t[((j*(j+1)/2) + k)*36]')
    self.gen_add_parallel_loop('i',f'{len(jids_a)}*36')
    self.gen_add_code_line('int jid = jids[i / 36];')
    self.gen_add_code_line('int ancestor_j = ancestors_j[i / 36];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line('outerProduct<T>(&S[ancestor_j*6], &psid[jid*6], &t[t_idx], 6, 6, i%36);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Perform all computations with t6
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Perform all computations with t6')
    self.gen_add_code_lines(['// for ancestor d2tau_dvdq[st_j, cc, dd] = -np.dot(t6, D3[:, st_j])', \
                             '// for ancestor d2tau_dq[cc, st_j, dd] = np.dot(t6, D2[:, st_j])', \
                             '// for ancestor d2tau_dvdq[cc, st_j, dd] = np.dot(t6, D3[:, st_j])'])
    self.gen_add_parallel_loop('i',f'{3*len(jids)}')
    self.gen_add_code_line(f'int index = i % {len(jids)};')
    self.gen_add_code_line(f'int jid = jids_compute[index];')
    self.gen_add_code_line(f'int ancestor_j = ancestors_j_compute[index];')
    self.gen_add_code_line(f'int st_j = st[index];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line('if (ancestor_j < jid) {', True)
    self.gen_add_code_line(f'if (i < {len(jids)}) d2tau_dvdq[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + ancestor_j] = -dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);')
    self.gen_add_code_line(f'else if (i < {2*len(jids)}) d2tau_dq2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D2[st_j*36]);')
    self.gen_add_code_line('else d2tau_dvdq[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);')
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_sync()


    # Compute t7
    self.gen_add_code_line('\n\n')
    jids_a, ancestors = self.robot.get_jid_ancestor_ids(include_joint=True)
    self.gen_add_code_line('// Compute t7 = outer(S[ancestor], psidd[joint])')
    self.gen_add_code_line('// t7[j][k] is stored at t[((j*(j+1)/2) + k)*36]')
    self.gen_add_parallel_loop('i',f'{len(jids_a)}*36')
    self.gen_add_code_line('int jid = jids[i / 36];')
    self.gen_add_code_line('int ancestor_j = ancestors_j[i / 36];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line('outerProduct<T>(&S[ancestor_j*6], &psidd[jid*6], &t[t_idx], 6, 6, i%36);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Perform all computations with t7
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Perform all computations with t7')
    self.gen_add_code_lines(['// for ancestor d2tau_dq[cc, st_j, dd] += np.dot(t7, D1[:, st_j])'])
    self.gen_add_parallel_loop('i',f'{len(jids)}')
    self.gen_add_code_line(f'int index = i % {len(jids)};')
    self.gen_add_code_line(f'int jid = jids_compute[index];')
    self.gen_add_code_line(f'int ancestor_j = ancestors_j_compute[index];')
    self.gen_add_code_line(f'int st_j = st[index];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line(f'if (ancestor_j < jid) d2tau_dq2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + st_j] += dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()


    # Compute t8
    self.gen_add_code_line('\n\n')
    jids_a, ancestors = self.robot.get_jid_ancestor_ids(include_joint=True)
    self.gen_add_code_line('// Compute t8 = outer(S[ancestor], S[joint])')
    self.gen_add_code_line('// t8[j][k] is stored at t[((j*(j+1)/2) + k)*36]')
    self.gen_add_parallel_loop('i',f'{len(jids_a)}*36')
    self.gen_add_code_line('int jid = jids[i / 36];')
    self.gen_add_code_line('int ancestor_j = ancestors_j[i / 36];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line('outerProduct<T>(&S[ancestor_j*6], &S[jid*6], &t[t_idx], 6, 6, i%36);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Perform all computations with t8
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Perform all computations with t8')
    self.gen_add_code_lines(['// for ancestor dM_dq[cc,st_j,dd] = t8.T @ D4[:, st_j]', \
                             '// for ancestor dM_dq[st_j,cc,dd] = t8.T @ D4[:, st_j]', \
                             '// for child dM_dq[cc, dd, succ_j] = np.dot(t8, D1[:, succ_j])', \
                             '// for child dM_dq[dd, cc, succ_j] = np.dot(t8, D1[:, succ_j])'
                             '// for child & ancestor d2tau_dqd[cc, succ_j, dd] = np.dot(t8, D3[:, succ_j])', \
                             '// for child & ancestor d2tau_dqd[cc, dd, succ_j] = np.dot(t8, D3[:, succ_j])', \
                             '// for child & ancestor d2tau_dvdq[cc, dd, succ_j] = np.dot(t8, D2[:, succ_j])'])
    self.gen_add_parallel_loop('i',f'{7*len(jids)}')
    self.gen_add_code_line(f'int index = i % {len(jids)};')
    self.gen_add_code_line(f'int jid = jids_compute[index];')
    self.gen_add_code_line(f'int ancestor_j = ancestors_j_compute[index];')
    self.gen_add_code_line(f'int st_j = st[index];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line('if (ancestor_j < jid) {', True)
    self.gen_add_code_line(f'if (i < {len(jids)}) dM_dq[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j * SECOND_ORDER_COORDS + jid] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D4[st_j*36]);')
    self.gen_add_code_line(f'else if (i < {2*len(jids)}) dM_dq[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j * SECOND_ORDER_COORDS + jid] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D4[st_j*36]);')
    self.gen_add_code_line('if (st_j != jid) {', True)
    self.gen_add_code_line(f'if (i < {3*len(jids)}) d2tau_dqd2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);')
    self.gen_add_code_line(f'else if (i < {4*len(jids)}) d2tau_dqd2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j * SECOND_ORDER_COORDS + jid] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D3[st_j*36]);')
    self.gen_add_code_line(f'else if (i < {5*len(jids)}) d2tau_dvdq[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j * SECOND_ORDER_COORDS + jid] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D2[st_j*36]);')
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_code_line(f'if (jid != st_j && i < {6*len(jids)}) dM_dq[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);')
    self.gen_add_code_line(f'else if (jid != st_j) dM_dq[jid*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j * SECOND_ORDER_COORDS + st_j] = dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()


    # Compute t9
    self.gen_add_code_line('\n\n')
    jids_a, ancestors = self.robot.get_jid_ancestor_ids(include_joint=True)
    self.gen_add_code_line('// Compute t9 = outer(S[ancestor], (Sd+psid)[joint])')
    self.gen_add_code_line('// t9[j][k] is stored at t[((j*(j+1)/2) + k)*36]')
    self.gen_add_parallel_loop('i',f'{len(jids_a)}*36')
    self.gen_add_code_line('int jid = jids[i / 36];')
    self.gen_add_code_line('int ancestor_j = ancestors_j[i / 36];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line('outerProduct<T>(&S[ancestor_j*6], &psid_Sd[jid*6], &t[t_idx], 6, 6, i%36);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Perform all computations with t9
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Perform all computations with t9')
    self.gen_add_code_lines(['// for ancestor & child d2tau_dvdq[cc, dd, succ_j] += np.dot(t9, D1[:, succ_j])', \
                             '// for ancestor & child d2tau_dq[cc, dd, succ_j] = d2tau_dq[cc, succ_j, dd]'])
    self.gen_add_parallel_loop('i',f'{2*len(jids)}')
    self.gen_add_code_line(f'int index = i % {len(jids)};')
    self.gen_add_code_line(f'int jid = jids_compute[index];')
    self.gen_add_code_line(f'int ancestor_j = ancestors_j_compute[index];')
    self.gen_add_code_line(f'int st_j = st[index];')
    self.gen_add_code_line(f'int t_idx = t_index_map[jid][ancestor_j]*36;')
    self.gen_add_code_line(f'if (i < {len(jids)} && ancestor_j < jid && st_j != jid) d2tau_dvdq[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j * SECOND_ORDER_COORDS + jid] += dot_prod<T, 36, 1, 1>(&t[t_idx], &D1[st_j*36]);')
    self.gen_add_code_line(f'else if (ancestor_j < jid & st_j != jid) d2tau_dq2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j * SECOND_ORDER_COORDS + jid] = d2tau_dq2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + st_j];')
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    
    # Compute p1..p6 in parallel
    jids_a, ancestors = self.robot.get_jid_ancestor_ids(include_joint=True)
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Compute p1..p6 in parallel')
    self.gen_add_code_lines(['// p1 = self.crm(psid_c) @ S_d', \
                             '// p2 = self.crm(psidd[:, k]) @ S_d', \
                             '// p3 = self.crm(S_c) @ S_d', \
                             '// p4 = self.crm(Sd_c + psid_c) @ S_d - 2 * self.crm(psid_d) @ S_c', \
                             '// p5 = self.crm(S_d) @ S_c', \
                             '// p6 = IC_S[joint] @ crm(S[ancestor]) + S[ancestor] @ crf_S_IC[joint]'])
    self.gen_add_parallel_loop('i',f'{6*6*len(jids_a)}')
    self.gen_add_code_line(f'int index = i % {6*len(jids_a)};')
    self.gen_add_code_line(f'int jid = jids[index / 6];')
    self.gen_add_code_line(f'int ancestor_j = ancestors_j[index / 6];')
    self.gen_add_code_line(f'int p_idx = t_index_map[jid][ancestor_j]*6;')
    self.gen_add_code_line(f'if (i < {len(jids_a)*6}) p1[p_idx + i % 6] = crm_mul<T>(i % 6, &psid[ancestor_j*6], &S[jid*6]);')
    self.gen_add_code_line(f'else if (i < {2*len(jids_a)*6}) p2[p_idx + i % 6] = crm_mul<T>(i % 6, &psidd[ancestor_j*6], &S[jid*6]);')
    self.gen_add_code_line(f'else if (i < {3*len(jids_a)*6}) p3[p_idx + i % 6] = crm_mul<T>(i % 6, &S[ancestor_j*6], &S[jid*6]);')
    self.gen_add_code_line(f'else if (i < {4*len(jids_a)*6}) p4[p_idx + i % 6] = crm_mul<T>(i % 6, &psid_Sd[ancestor_j*6], &S[jid*6]) - 2 * crm_mul<T>(i % 6, &psid[jid*6], &S[ancestor_j*6]);')
    self.gen_add_code_line(f'else if (i < {5*len(jids_a)*6}) p5[p_idx + i % 6] = crm_mul<T>(i % 6, &S[jid*6], &S[ancestor_j*6]);')
    self.gen_add_code_line(f'else p6[p_idx + i % 6] = dot_prod<T, 6, 1, 1>(&IC_S[jid*6], &crm_S[ancestor_j*36 + (i % 6)*6]) + dot_prod<T, 6, 1, 1>(&S[ancestor_j*6], &crf_S_IC[jid*36 + (i % 6)*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Finish all computations with p1..p6
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Finish all computations with p1..p5')
    self.gen_add_code_lines(['// for joint d2tau_dq[st_j, dd, cc] += -np.dot(p1, T2[:, st_j]) + np.dot(p2, T1[:, st_j])', \
                             '// for ancestor d2tau_dq[st_j, cc, dd] += -np.dot(p1, T2[:, st_j]) + np.dot(p2, T1[:, st_j])', \
                             '// for ancestor d2tau_dvdq[st_j, cc, dd] += -np.dot(p3, T2[:, st_j]) + np.dot(p4, T1[:, st_j])', \
                             '// for ancestor d2tau_dq[cc, st_j, dd] -= np.dot(p5, T3[:, st_j])', \
                             '// for ancestor && child d2tau_dq[cc, dd, succ_j] -= np.dot(p5, T3[:, st_j])', \
                             '// for ancestor d2tau_dvdq[cc, st_j, dd] -= np.dot(p5, T4[:, st_j])'])
    self.gen_add_parallel_loop('i',f'{6*len(jids)}')
    self.gen_add_code_line(f'int index = i % {len(jids)};')
    self.gen_add_code_line(f'int jid = jids_compute[index];')
    self.gen_add_code_line(f'int ancestor_j = ancestors_j_compute[index];')
    self.gen_add_code_line(f'int st_j = st[index];')
    self.gen_add_code_line(f'int p_idx = t_index_map[jid][ancestor_j]*6;')
    self.gen_add_code_line(f'if (i < {len(jids)}) d2tau_dq2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + ancestor_j * SECOND_ORDER_COORDS + jid] += -dot_prod<T, 6, 1, 1>(&p1[p_idx], &T2[st_j*6]) + dot_prod<T, 6, 1, 1>(&p2[p_idx], &T1[st_j*6]);')
    self.gen_add_code_line('else if (ancestor_j < jid) {', True)
    self.gen_add_code_line(f'if (i < {2*len(jids)}) d2tau_dq2[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + ancestor_j] += -dot_prod<T, 6, 1, 1>(&p1[p_idx], &T2[st_j*6]) + dot_prod<T, 6, 1, 1>(&p2[p_idx], &T1[st_j*6]);')
    self.gen_add_code_line(f'else if (i < {3*len(jids)}) d2tau_dvdq[st_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + ancestor_j] += -dot_prod<T, 6, 1, 1>(&p3[p_idx], &T2[st_j*6]) + dot_prod<T, 6, 1, 1>(&p4[p_idx], &T1[st_j*6]);')
    self.gen_add_code_line(f'else if (i < {4*len(jids)}) d2tau_dq2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + st_j] -= dot_prod<T, 6, 1, 1>(&p5[p_idx], &T3[st_j*6]);')
    self.gen_add_code_line(f'else if (i < {5*len(jids)} && st_j != jid) d2tau_dq2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + st_j * SECOND_ORDER_COORDS + jid] -= dot_prod<T, 6, 1, 1>(&p5[p_idx], &T3[st_j*6]);')
    self.gen_add_code_line(f'else if (i >= {5*len(jids)}) d2tau_dvdq[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + st_j] -= dot_prod<T, 6, 1, 1>(&p5[p_idx], &T4[st_j*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # Finish computation with p6
    self.gen_add_code_line('\n\n')
    self.gen_add_code_line('// Finish computation with p6')
    self.gen_add_code_line('// d2tau_dqd[ancestor, joint, joint] = p6[joint][ancestor] @ S[joint]')
    self.gen_add_parallel_loop('i',f'{len(jids_a)}')
    self.gen_add_code_line(f'int jid = jids[i];')
    self.gen_add_code_line(f'int ancestor_j = ancestors_j[i];')
    self.gen_add_code_line(f'int p_idx = t_index_map[jid][ancestor_j]*6;')
    self.gen_add_code_line(f'if (ancestor_j < jid) d2tau_dqd2[ancestor_j*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + jid * SECOND_ORDER_COORDS + jid] = dot_prod<T, 6, 1, 1>(&p6[p_idx], &S[jid*6]);')
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    if self.idsva_so_needs_reference_order_output_repair():
        self.gen_idsva_so_body_frame_reference_order_output_repair()

    if is_mimic:
        # ---- Fold the internal 4*NB^3 sweep to the reduced 4*NV^3 public output ----
        # public[v(i), v(j), v(k)] += alpha_i*alpha_j*alpha_k * internal[i, j, k],
        # summed over internal slots (i,j,k) that share reduced v-slots (mimic
        # siblings). This is the einsum('ia,ijk,jb,kc->abc', R, T, R, R) reduction
        # the oracle applies, with R[i, v(i)] = alpha_i. Each of the 4 tensor blocks
        # folds independently. We zero the public output first (NV^3 per block) then
        # scatter-accumulate every internal cell into its reduced destination.
        NB = num_bodies
        fold_vslot = [self._v_slot_cpp(b) for b in range(NB)]
        fold_alpha = [self._alpha_for_jid(b) for b in range(NB)]
        self.gen_add_sync()
        self.gen_add_code_line("// Mimic fold: reduce internal NB^3 sweep to public NV^3 output")
        self.gen_add_code_line(
            "static const int so_fold_vslot[] = { " + ", ".join(map(str, fold_vslot)) + " };")
        self.gen_add_code_line(
            "static const T so_fold_alpha[] = { " + ", ".join(
                "static_cast<T>(" + repr(a) + ")" for a in fold_alpha) + " };")
        # Zero the public NV^3 output (4 blocks).
        self.gen_add_parallel_loop('i', f'4*{NV**3}')
        self.gen_add_code_line("s_idsva_so_public[i] = static_cast<T>(0);")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        # Scatter-accumulate. One thread per internal cell per block; the destination
        # public cell is uniquely determined by (v(i),v(j),v(k)). Multiple internal
        # cells can map to the SAME public cell (mimic siblings), so we must use an
        # atomic accumulate to avoid lost updates across the block's threads.
        self.gen_add_parallel_loop('idx', f'4*{NB**3}')
        self.gen_add_code_line(f'int blk = idx / {NB**3};')
        self.gen_add_code_line(f'int rem = idx % {NB**3};')
        self.gen_add_code_line(f'int ii = rem / {NB*NB};')
        self.gen_add_code_line(f'int jj = (rem / {NB}) % {NB};')
        self.gen_add_code_line(f'int kk = rem % {NB};')
        self.gen_add_code_line("T val = s_idsva_so_internal[idx];")
        self.gen_add_code_line("if (val != static_cast<T>(0)) {", True)
        self.gen_add_code_line("T w = so_fold_alpha[ii] * so_fold_alpha[jj] * so_fold_alpha[kk];")
        self.gen_add_code_line(f"int dst = blk*{NV**3} + so_fold_vslot[ii]*{NV*NV} + so_fold_vslot[jj]*{NV} + so_fold_vslot[kk];")
        self.gen_add_code_line("atomicAdd(&s_idsva_so_public[dst], w * val);")
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    self.gen_add_end_function()

        
def gen_idsva_so_body_frame_public_dvdq_layout_repair(self):
    """
    Emit a final public-output repair for the optimized IDSVA-SO assembly path.

    The optimized path stores the d2tau_dvdq block with the last two axes
    transposed relative to RBDReference/public CUDA output. FDSVA consumes the
    inner tensor directly, so callers that consume public-layout tensors should
    run this repair after inner assembly instead of changing the optimized
    assembly order in the first corrective pass.
    """
    if self.robot.floating_base or self.idsva_so_needs_reference_order_output_repair():
        return

    NV = self.robot.get_num_vel()
    block_offset = 2 * NV**3
    self.gen_add_sync()
    self.gen_add_code_line("// Repair public d2tau_dvdq layout for optimized IDSVA-SO output")
    self.gen_add_parallel_loop("dvdq_swap_idx", "SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS")
    self.gen_add_code_line("int dvdq_i = dvdq_swap_idx / (SECOND_ORDER_COORDS*SECOND_ORDER_COORDS);")
    self.gen_add_code_line("int dvdq_j = (dvdq_swap_idx / SECOND_ORDER_COORDS) % SECOND_ORDER_COORDS;")
    self.gen_add_code_line("int dvdq_k = dvdq_swap_idx % SECOND_ORDER_COORDS;")
    self.gen_add_code_line("if (dvdq_j < dvdq_k) {", True)
    self.gen_add_code_line(f"T dvdq_tmp = s_idsva_so[{block_offset} + dvdq_i*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + dvdq_j*SECOND_ORDER_COORDS + dvdq_k];")
    self.gen_add_code_line(f"s_idsva_so[{block_offset} + dvdq_i*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + dvdq_j*SECOND_ORDER_COORDS + dvdq_k] = s_idsva_so[{block_offset} + dvdq_i*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + dvdq_k*SECOND_ORDER_COORDS + dvdq_j];")
    self.gen_add_code_line(f"s_idsva_so[{block_offset} + dvdq_i*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS + dvdq_k*SECOND_ORDER_COORDS + dvdq_j] = dvdq_tmp;")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def _emit_idsva_so_body_frame_kernel_body_for_flags(self, n, NUM_POS, use_qdd_input, single_call_timing,
                                                    use_global_output, s_temp_in_global, bc_in_global, tp_in_global=False):
    """Emit the idsva_so body-frame kernel body for one tier's spill flags.

    Flags (see the per-tier ladder in GRiDCodeGenerator.py):
      - use_global_output: 4*NV^3 output tensor lives in d_idsva_so (global) vs s_idsva_so (smem).
      - s_temp_in_global:  the whole inner s_temp arena routes to d_workspace (guaranteed-fit fallback).
      - bc_in_global:      surgical — only the cold BC buffer routes to d_workspace (inner BC_IN_SMEM=false).
      - tp_in_global:      surgical — only the ancestor-pair scratch t/p1..p6 (36*len(jids_a))
                           routes to d_workspace (inner TP_IN_SMEM=false); BC slides down to fill
                           the vacated smem so the arena shrinks by exactly that span.
    bc_in_global and tp_in_global are mutually exclusive (separate rungs). Floating-base
    (diagnostic) uses the gravity-shim spill at the SO offset regardless of flags.
    """
    extra_t_buffers = [("s_q_qd_u", 3*NUM_POS)]
    if not use_global_output:
        extra_t_buffers.append(("s_idsva_so", 4*n**3))
    if use_qdd_input:
        extra_t_buffers.append(("s_qdd", n))
    inner_temp = self.gen_idsva_so_body_frame_inner_temp_mem_size()
    bc_slab = 36 * self.robot.get_num_bodies()
    jids_a, _ = self.robot.get_jid_ancestor_ids(include_joint=True)
    tp_slab = 36 * len(jids_a)
    # smem s_temp allocation per rung:
    #   - s_temp_in_global (whole-arena rung): 0 (inner repoints s_temp -> d_workspace).
    #   - bc_in_global (surgical BC rung): inner_temp - BC. BC is the LAST (top) 36*NB
    #     slab (see gen_idsva_so_body_frame_inner var layout), so dropping its tail keeps
    #     every hot buffer below in place. This MUST match the per-tier launch smem bytes
    #     (GRiDCodeGenerator.py: _idsva_bf_out - _idsva_bf_BC) or the kernel arena and the
    #     launch disagree and the top slab reads OOB.
    #   - tp_in_global (surgical t/p rung): inner_temp - 36*len(jids_a). t/p sits just below
    #     BC; when it spills, BC slides down to fill it so the smem arena shrinks by exactly
    #     the t/p span. MUST match GRiDCodeGenerator.py: _idsva_bf_out - _idsva_bf_TP.
    #   - otherwise (PERF / global_output rungs): full inner_temp.
    if s_temp_in_global:
        smem_temp = 0
    elif bc_in_global:
        smem_temp = inner_temp - bc_slab
    elif tp_in_global:
        smem_temp = inner_temp - tp_slab
    else:
        smem_temp = inner_temp
    self.gen_XImats_helpers_temp_shared_memory_code(smem_temp, extra_t_buffers = extra_t_buffers)
    # `d_temp_spill` is the typed view into d_workspace handed to the inner as its
    # `d_workspace` arg. The inner does the s_temp/BC repoint itself (inner-owns
    # placement): whole-arena rung -> inner sets s_temp = d_temp_spill; surgical BC/t-p rung
    # -> inner sets BC / t = d_temp_spill; floating shim -> gravity-Hessian uses it directly.
    self.gen_add_code_line("T *d_temp_spill = nullptr; (void)d_temp_spill;")
    needs_workspace = self.robot.floating_base or s_temp_in_global or bc_in_global or tp_in_global
    if not needs_workspace:
        self.gen_add_code_line("(void)d_workspace;")
    if use_qdd_input:
        self.gen_add_code_line(f"T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[{NUM_POS}];")
    else:
        self.gen_add_code_line(f"T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[{NUM_POS}]; T *s_qdd = &s_q_qd_u[{2*NUM_POS}];")
    bc_in_smem_expr = "false" if bc_in_global else "true"
    scratch_in_smem_expr = "false" if s_temp_in_global else "true"
    # Only thread the 4th template arg when t/p actually spills, so every non-tp rung emits
    # the same <T, SCRATCH, BC> instantiation it did before (Gate A: byte-identical default).
    tp_in_smem_expr = "false" if tp_in_global else None
    so_off = "GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()"
    ts_off = ("k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + " + so_off) if not single_call_timing else so_off

    def _emit_spill_ptrs():
        # Whichever spill is active routes through d_temp_spill; the inner consumes it.
        if self.robot.floating_base or bc_in_global or s_temp_in_global or tp_in_global:
            self.gen_add_code_line(f"d_temp_spill = reinterpret_cast<T *>(&d_workspace[{ts_off}]);")

    if not single_call_timing:
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",block_level = True)
        if use_qdd_input: # TODO
            self.gen_kernel_load_inputs("q_qd",str(n + NUM_POS),"qdd",str(n),stride="stride_q_qd",stride2=str(n))
        else:
            self.gen_kernel_load_inputs("q_qd_u",str(3*NUM_POS),stride="stride_q_qd_u")
        _emit_spill_ptrs()
        self.gen_add_code_line("// compute (the inner loads/updates XImats internally, after its scratch repoint)")
        if use_global_output:
            self.gen_add_code_line("// Write directly to RAM due to output tensor size")
            self.gen_add_code_line(f"T *s_idsva_so = &d_idsva_so[k*{4*n**3}];")
        self.gen_idsva_so_body_frame_inner_function_call(bc_in_smem_expr = bc_in_smem_expr, scratch_in_smem_expr = scratch_in_smem_expr, tp_in_smem_expr = tp_in_smem_expr)
        self.gen_idsva_so_body_frame_public_dvdq_layout_repair()
        if not use_global_output: self.gen_kernel_save_result("idsva_so",str(4*n**3),stride=str(4*n**3))
        self.gen_add_end_control_flow()
    else:
        if use_qdd_input: # TODO
            self.gen_kernel_load_inputs("q_qd",str(2*n),"qdd",str(n))
        else:
            self.gen_kernel_load_inputs("q_qd_u",str(3*NUM_POS))
        _emit_spill_ptrs()
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        if use_qdd_input:
            self.gen_anti_licm_input_reload("q_qd",str(2*n),"qdd",str(n))
        else:
            self.gen_anti_licm_input_reload("q_qd_u",str(3*NUM_POS))
        # The inner loads/updates XImats internally each rep (after its scratch repoint).
        if use_global_output:
            self.gen_add_code_line("// Write directly to RAM due to output tensor size")
            self.gen_add_code_line("T *s_idsva_so = d_idsva_so;")
        self.gen_idsva_so_body_frame_inner_function_call(bc_in_smem_expr = bc_in_smem_expr, scratch_in_smem_expr = scratch_in_smem_expr, tp_in_smem_expr = tp_in_smem_expr)
        self.gen_idsva_so_body_frame_public_dvdq_layout_repair()
        self.gen_add_end_control_flow()
        if not use_global_output: self.gen_kernel_save_result("idsva_so",str(4*n**3))


def gen_idsva_so_body_frame_kernel(self, use_qdd_input = False, single_call_timing = False):
    NUM_POS = self.robot.get_num_pos()
    n = self.robot.get_num_vel()
    # define function def and params
    func_params = ["d_idsva_so is a pointer to memory for the final result of size 4*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS = " + str(4*n**3), \
                   "d_q_dq_u is the vector of joint positions, velocities, and accelerations", \
                   "stride_q_qd_u is the stide between each q, qd, u", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    # The kernel takes a per-timestep global-memory workspace pointer. It is used by
    # the floating-base gravity shim and by the LITE/MINIMAL spill rungs (whole-s_temp
    # and surgical BC); at TIER_SHARED for a robot that fits, it is unused.
    func_def_start = "void idsva_so_body_frame_kernel(T *d_idsva_so, unsigned char *d_workspace, const T *d_q_qd_u, const int stride_q_qd_u, "
    func_params.insert(1, "d_workspace is a per-timestep global-memory scratch buffer (gravity shim + LITE/MINIMAL spill rungs)")
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    if use_qdd_input: # TODO
        func_def_start += "const T *d_qdd, "
        func_params.insert(-2,"d_qdd is the vector of joint accelerations")
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Computes the second order derivatives of inverse dynamics",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)

    table = self._idsva_so_body_tier_table  # [(name, t_count, use_global_output, s_temp_in_global, bc_in_global, tp_in_global), ...]
    if not getattr(self, "idsva_so_body_frame_use_ladder", False):
        # Floating-base diagnostic path: single body, legacy gravity-shim spill.
        ugo = getattr(self, "idsva_so_body_frame_use_global_output", False)
        _emit_idsva_so_body_frame_kernel_body_for_flags(self, n, NUM_POS, use_qdd_input, single_call_timing,
                                                             ugo, False, False, False)
    else:
        picks = self.idsva_so_body_frame_spill_tier_3way
        def _emit_idsva_so_body_body(pick):
            _, _, ugo, stg, bcg, tpg = table[pick]
            _emit_idsva_so_body_frame_kernel_body_for_flags(self, n, NUM_POS, use_qdd_input, single_call_timing,
                                                                 ugo, stg, bcg, tpg)
        self.gen_tier_dispatch(picks, _emit_idsva_so_body_body)
    self.gen_add_end_function()

def gen_idsva_so_body_frame_host(self, mode = 0):
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
    func_def_start = "void idsva_so_body_frame(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end =   "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    # then generate the code
    self.gen_add_func_doc("Compute IDSVA-SO (Inverse Dynamics - Spatial Vector Algebra - Second Order)",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"idsva_so_body_frame requires all-data or dynamics gridData\");")
    func_call_start = "idsva_so_body_frame_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_idsva_so," + \
        "hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,"
    func_call_end = "d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("kernel<T, RESOURCE_TIER>","kernel_single_timing<T, RESOURCE_TIER>")

    self.gen_add_code_line("int stride_q_qd = Q_QD_U_STRIDE;")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*" + \
                                ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));", \
                                 "gpuErrchkKernel();"])
    # TODO then compute but adjust for compressed mem and qdd usage
    self.gen_add_code_line("// then call the kernel")
    # TODO - qdd=0 optimization
    
    func_call_code = [f'{func_call_start}{func_call_end}']
    # wrap function call in timing (if needed). The sync between the launch
    # and `clock_gettime(end)` is REQUIRED: kernel launch is async, so
    # without it the timer captures only host-side launch overhead, not
    # actual kernel work. Other algorithms (FD/ABA/etc.) already do this.
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("gpuErrchkKernel();")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"idsva_so\", IDSVA_SO_BODY_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        # then transfer memory back
        self.gen_add_code_lines(["// finally transfer the result back", \
                                 "gpuErrchk(cudaMemcpy(hd_data->h_idsva_so,hd_data->d_idsva_so,SECOND_ORDER_TENSOR_SIZE*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchkKernel();"])
    else:
        # compute_only path needs an explicit sync after the kernel launch
        # so the caller's batch-timing loop captures actual kernel completion
        # time (otherwise the async launch returns ~immediately and we
        # measure ~launch-overhead per call regardless of N).
        self.gen_add_code_line("gpuErrchkKernel();")

    # finally report out timing if requested. Label format matches the
    # bench's `parse_grid_output` parser, which keys on "single call idsva_so_body_frame"
    # (lowercase): emit "IDSVA_SO_BODY_FRAME" so the parser picks it up. The old
    # "ID-SO" label was silently dropped by the parser → null timings.
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("idsva_so_body_frame"))
    self.gen_add_end_function()

def gen_idsva_so_body_frame(self):
    # gen the inner code
    self.gen_idsva_so_body_frame_inner()
    # and the kernels
    self.gen_idsva_so_body_frame_kernel(False,True)
    self.gen_idsva_so_body_frame_kernel(False,False)
    # and host wrapeprs
    self.gen_idsva_so_body_frame_host(0)
    self.gen_idsva_so_body_frame_host(1)
    self.gen_idsva_so_body_frame_host(2)


# =============================================================================
# world-frame IDSVA-SO: a separate, single-pass CUDA emission that mirrors
# `RBDReference.idsva_so_world_frame` (a faithful port of spatial_v2_extended's
# `ID_SO_derivatives.m`). World-frame propagation; gravity baked into the main
# sweep at the floating-base root; no separate gravity-shim. Co-exists with the
# existing shim-based `gen_idsva_so_body_frame_floating_reference_inner` path.
# =============================================================================

def gen_idsva_so_world_frame_temp_mem_size(self):
    """Shared-memory float count for the world-frame inner.

    Layout:
      - Xup, Xdown, IC, BC: 4 * 36 * NB
      - v, a, f:           3 *  6 * NB
      - S, Sd, psid, psidd: 4 *  6 * NV
      - Per-(i, p) scratch (A0..A7, Bic_phi, Bic_psid): 10 * 36
      - Per-(j, t) scratch (u1..u12): 12 * 6
      - a_grav scratch: 6
      - Per-(i, p) helper vectors (ICi_S, ICi_psid, ICi_psidd, BCi_S, BCi_psid,
        BCiT_S, crf_S_f_i, A5_vec, A7_vec): 9 * 6 — moved from thread-0 stack
        to shared so the idx-over-36 parallel loop in phase 5a can read them.
    """
    NV = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    # Mimic-aware: when bodies share reduced velocity slots, the inner runs in
    # per-column INTERNAL coordinates (n_int = total column count >= NV) so the
    # S/Sd/psid/psidd vel-indexed bands and a 4*n_int^3 internal output slab are
    # sized by n_int, and the slab is appended on top. The inner assembles into
    # that slab then alpha-folds to the reduced 4*NV^3 caller output (mirrors
    # RBDReference.idsva_so_world_frame's has_mimic path). Non-mimic: n_int == NV,
    # internal_slab == 0 => byte-identical to the legacy size.
    if self.robot_has_mimic_joints():
        n_int = _idsva_so_floating_velocity_metadata(self.robot)["n_int"]
        internal_slab = 4 * n_int ** 3
    else:
        n_int = NV
        internal_slab = 0
    return int(4 * 36 * NB + 3 * 6 * NB + 4 * 6 * n_int + 10 * 36 + 12 * 6 + 6 + 9 * 6 + internal_slab)


def gen_idsva_so_world_frame_inner(self, use_qdd_input = False):
    """Emit `idsva_so_world_frame_inner` — a clean world-frame IDSVA-SO.

    Mirrors `RBDReference.idsva_so_world_frame`:
      - World-frame quantities: `S[i] = Xdown[i] @ S_local`, `IC[i] = Xup[i].T @ I @ Xup[i]`.
      - Root acceleration `a[:, 0] = -a_grav` (world frame, gravity baked in).
      - Floating-base root has `Xup[0] = inv(X_local[0])` (Featherstone xlt-inverse pattern).
      - Triple ancestor walk `(i over bodies reverse, p over body i's velocity columns,
        j over ancestors-or-self of i, t over body j's velocity columns, k over ancestors-of-j,
        r over body k's velocity columns)` produces d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq.
      - NO gravity-shim. NO `*= 2` quaternion scaling.

    SIMT-parallel: Phases that are inherently parent-dependent (Xup forward pass,
    forward sweep, per-(i,p) and per-(j,t) intermediate builds, IC/BC/f aggregate
    bubble-up) run under a thread-0 guard; phases that are pleasingly parallel
    (Xdown per-body, S_vel per-velocity, output-init, final transpose) use a
    parallel_loop. The dominant triple ancestor walk's innermost (k, rr)
    iteration set is flattened and distributed across threads (each thread owns
    a disjoint vel_k, so output-cell writes are race-free).
    """
    NV = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    metadata = _idsva_so_floating_velocity_metadata(self.robot)
    parent_ids = [self.robot.get_parent_id(body_id) for body_id in range(NB)]

    # Mimic-aware INTERNAL-coordinate sweep (mirrors RBDReference.idsva_so_world_frame's
    # has_mimic path). When bodies share reduced velocity slots, run the whole assembly in
    # per-column internal coords (n_int columns), output into a 4*n_int^3 internal slab,
    # then alpha-fold to the reduced 4*NV^3 public output. The forward sweep / output writes
    # are keyed on the internal slot (SO_N = n_int stride); s_qd/s_qdd reads map internal ->
    # true reduced slot and scale by alpha. Non-mimic: n_int == NV, SO_N == NUM_VEL, no slab,
    # no fold => character-identical CUDA to the legacy emission.
    is_mimic = self.robot_has_mimic_joints()
    n_int = metadata["n_int"]
    # The vel-indexed bands (S/Sd/psid/psidd) and the output strides are sized by SO_N.
    SO_N = "SO_N_INT" if is_mimic else "NUM_VEL"

    func_params = [
        "s_idsva_so is a pointer to memory for the final result of size 4*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS = " + str(4*NV**3),
        "s_q is the vector of joint positions",
        "s_qd is the vector of joint velocities",
        "s_qdd is the vector of joint accelerations",
        "s_temp is a pointer to helper shared memory of size = " + str(self.gen_idsva_so_world_frame_temp_mem_size()),
        "d_workspace is a pointer to per-timestep global-memory scratch (unused at TIER_SHARED; cold buffers spill here at LITE/MINIMAL)",
        "d_robotModel holds XImats/topology; the inner owns the load_update_XImats call (inner-owns-placement)",
        "gravity is the gravity constant",
    ]
    # MUJOCO_OUTPUT (floating + non-mimic/skew): compile-time mjx output-convention
    # flag. Mirrors the id-gradient gate exactly (so the same robots are mjx-capable).
    # When set, an epilogue at the END of this inner (where s_temp is DEAD) transforms
    # the 4 pin SO tensors in s_idsva_so to the mjx convention in place, reusing the
    # id-value / id-gradient / crba inners (M, tau, dtau_dq, dtau_dqd) from a carve of
    # d_mjx_scratch (the SO-temp region of d_workspace; dead post-assembly).
    mjx_inner = self.robot.floating_base and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis())
    # qdd is mutated in place by gen_mjx_input_convert (accel reframe), so s_qdd is
    # non-const here regardless (it already is: T *s_qdd).
    func_def_start = "void idsva_so_world_frame_inner(T *s_idsva_so, const T *s_q, const T *s_qd, T *s_qdd, "
    if mjx_inner:
        func_def_end = "T *s_temp, T *d_workspace, const robotModel<T> *d_robotModel, const T gravity, T *d_mjx_scratch = nullptr) {"
        func_params.append("d_mjx_scratch is the mjx-epilogue scratch arena (SO-temp region of d_workspace; only read when MUJOCO_OUTPUT)")
    else:
        func_def_end = "T *s_temp, T *d_workspace, const robotModel<T> *d_robotModel, const T gravity) {"
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -2)
    func_notes = [
        "world-frame propagation, gravity baked into main sweep.",
        "Mirrors RBDReference.idsva_so_world_frame (port of spatial_v2_extended ID_SO_derivatives.m).",
        "SIMT-parallel: pleasingly parallel phases distribute across threads; sequential phases (Xup, forward sweep, per-(i,p)/(j,t) intermediates) run under a thread-0 guard.",
    ]
    func_def = func_def_start + func_def_end

    self.gen_add_func_doc(
        "Computes IDSVA second-order derivatives via the world-frame single-pass formulation",
        func_notes, func_params, None,
    )
    if mjx_inner:
        # MUJOCO_OUTPUT appended LAST so existing positional <T,SCRATCH,COLD> call
        # sites are unaffected; default false -> the epilogue if-constexpr-elides to
        # byte-identical PTX. Fixed-base / mimic / skew never emit it.
        self.gen_add_code_line("template <typename T, bool SCRATCH_IN_SMEM = true, bool COLD_IN_SMEM = true, bool MUJOCO_OUTPUT = false>")
    else:
        self.gen_add_code_line("template <typename T, bool SCRATCH_IN_SMEM = true, bool COLD_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # Inner owns the XImats load too: the s_temp repoint below covers the helper's
    # sincos scratch, so every consumer (incl. XImats) follows the placement and the
    # kernel never repoints s_temp. Mirrors fdsva_so_device (the canon).
    self.gen_add_code_lines([
        "// world-frame IDSVA-SO shared-memory layout.",
        "// Inner owns scratch placement: SCRATCH_IN_SMEM picks s_temp (shared) vs",
        "// d_workspace (global). Repointing s_temp here keeps every layout line below",
        "// unchanged. See docs/idsva_so_inner_refactor_notes.md (inner-owns-placement).",
        "if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }",
    ])
    # XImats is loaded INSIDE the inner, AFTER the s_temp repoint — so its sincos
    # scratch (which uses s_temp) follows the same placement. Repoint FIRST, then load.
    # (load_update_XImats_helpers ends with its own sync, matching fdsva_so_device.)
    self.gen_load_update_XImats_helpers_function_call()
    self.gen_add_code_lines(([
        # Mimic: SO_N_INT = per-column internal coordinate count (drives the vel-indexed
        # band sizes + the internal output-slab strides). Declared first so every layout
        # line below can reference it. Non-mimic: SO_N == "NUM_VEL" (no SO_N_INT emitted).
        f"constexpr int SO_N_INT = {n_int};",
    ] if is_mimic else []) + [
        "T *Ipool   = s_XImats + XIMAT_SIZE*NUM_BODIES;",
        "// --- HOT region (always smem when SCRATCH_IN_SMEM) ---",
        "// Xup is RELOCATED to the tail cold band (built Step 1, dead after the Step-4 IC",
        "// build; never read by the Step-5 hot triple-walk). The hot chain now starts at IC",
        "// so the recursion-hot arena stays in smem at the surgical (output_cold) rung.",
        "T *IC      = s_temp;",
        "T *BC      = IC      + 36*NUM_BODIES;",
        "T *f_w     = BC      + 36*NUM_BODIES;",
        "T *S_vel   = f_w     +  6*NUM_BODIES;",
        f"T *Sd_vel  = S_vel   +  6*{SO_N};",
        f"T *psid_v  = Sd_vel  +  6*{SO_N};",
        f"T *psidd_v = psid_v  +  6*{SO_N};",
        f"T *scratch = psidd_v +  6*{SO_N};",
        "// Per-(i,p) scratch blocks (each 6x6 column-major, total 10).",
        "T *S_Bphi  = scratch;            // Bic_phi  (Bic(IC[i], S_p))",
        "T *S_Bpsid = S_Bphi    + 36;     // Bic_psid (Bic(IC[i], psid_p))",
        "T *S_A0    = S_Bpsid   + 36;",
        "T *S_A1    = S_A0      + 36;",
        "T *S_A2    = S_A1      + 36;",
        "T *S_A3    = S_A2      + 36;",
        "T *S_A4    = S_A3      + 36;",
        "T *S_A5    = S_A4      + 36;",
        "T *S_A6    = S_A5      + 36;",
        "T *S_A7    = S_A6      + 36;",
        "// Per-(j,t) scratch: u1..u12 (each 6-vector).",
        "T *S_u1    = S_A7      + 36;",
        "T *S_u2    = S_u1      +  6;",
        "T *S_u3    = S_u2      +  6;",
        "T *S_u4    = S_u3      +  6;",
        "T *S_u5    = S_u4      +  6;",
        "T *S_u6    = S_u5      +  6;",
        "T *S_u7    = S_u6      +  6;",
        "T *S_u8    = S_u7      +  6;",
        "T *S_u9    = S_u8      +  6;",
        "T *S_u10   = S_u9      +  6;",
        "T *S_u11   = S_u10     +  6;",
        "T *S_u12   = S_u11     +  6;",
        "T *S_agrav = S_u12     +  6;",
        "// Per-(i,p) vector intermediates (used by phase 5a parallel idx-over-36 build of A0..A7).",
        "T *S_ICi_S      = S_agrav      +  6;",
        "T *S_ICi_psid   = S_ICi_S      +  6;",
        "T *S_ICi_psidd  = S_ICi_psid   +  6;",
        "T *S_BCi_S      = S_ICi_psidd  +  6;",
        "T *S_BCi_psid   = S_BCi_S      +  6;",
        "T *S_BCiT_S     = S_BCi_psid   +  6;",
        "T *S_crf_S_f_i  = S_BCiT_S     +  6;",
        "T *S_A5_vec     = S_crf_S_f_i  +  6;",
        "T *S_A7_vec     = S_A5_vec     +  6;",] + ([
        # --- Mimic: a 4*n_int^3 INTERNAL slab placed in the always-hot region (BEFORE the
        # cold trio) so the surgical COLD spill never disturbs it. The inner runs in internal
        # coords into this slab, then alpha-folds to the reduced 4*NV^3 public caller output.
        "T *wf_so_internal = S_A7_vec + 6;  // 4*SO_N_INT^3 internal slab (always-hot region)",
    ] if is_mimic else []) + [
        "// --- COLD region (end of arena), cold-QUAD: Xup (dead after the Step-4 IC build)",
        "// + Xdown (dead after Step 3) + v_w/a_w (dead after Step 4's f_w build). Placed",
        "// last (contiguous tail) so a surgical sub-region (d_cold) can route JUST these",
        "// 84*NB floats to d_workspace at the output_cold rung while the hot arena stays smem.",
        ("T *Xup     = wf_so_internal + 4*SO_N_INT*SO_N_INT*SO_N_INT;" if is_mimic
         else "T *Xup     = S_A7_vec  +  6;"),
        "T *Xdown   = Xup       + 36*NUM_BODIES;",
        "T *v_w     = Xdown     + 36*NUM_BODIES;",
        "T *a_w     = v_w       +  6*NUM_BODIES;",
        "// COLD_IN_SMEM=false repoints the cold quad to a d_workspace sub-region (d_cold).",
        "// Contract: COLD_IN_SMEM=false is only used with SCRATCH_IN_SMEM=true (hot stays",
        "// in smem), so d_cold = &d_workspace[0] is exclusive — the deep rung instead uses",
        "// SCRATCH_IN_SMEM=false (whole arena, incl. these three, to d_workspace) with",
        "// COLD_IN_SMEM=true. The two spill levers are mutually exclusive by construction.",
        "if constexpr (!COLD_IN_SMEM) { Xup = d_workspace; Xdown = Xup + 36*NUM_BODIES; v_w = Xdown + 36*NUM_BODIES; a_w = v_w + 6*NUM_BODIES; }",] + ([
        # Mimic: assemble into the internal slab; the public caller dest is saved + folded last.
        "T *s_idsva_so_public = s_idsva_so;  // reduced 4*NV^3 caller dest (saved before repoint)",
        "T *s_idsva_so_internal = wf_so_internal;",
        "T *d2tau_dq2  = s_idsva_so_internal;",
        "T *d2tau_dqd2 = d2tau_dq2  + SO_N_INT*SO_N_INT*SO_N_INT;",
        "T *d2tau_dvdq = d2tau_dqd2 + SO_N_INT*SO_N_INT*SO_N_INT;",
        "T *dM_dq      = d2tau_dvdq + SO_N_INT*SO_N_INT*SO_N_INT;",
    ] if is_mimic else [
        "T *d2tau_dq2  = s_idsva_so;",
        "T *d2tau_dqd2 = d2tau_dq2  + SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS;",
        "T *d2tau_dvdq = d2tau_dqd2 + SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS;",
        "T *dM_dq      = d2tau_dvdq + SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS;",
    ]) + [
        "",
        # constexpr (rather than static const) so the values get baked into use
        # sites at compile time and no per-TU linker symbol is emitted. nvcc's
        # nvlink with -rdc=true otherwise complains "Size doesn't match" when
        # the same template instantiation lives in multiple per-algo TUs.
        f"constexpr int wf_parent[] = {{ {_idsva_so_int_array(parent_ids)} }};",
        f"constexpr int wf_body_v_start[] = {{ {_idsva_so_int_array(metadata['body_v_start'])} }};",
        # For mimic, wf_body_v_index holds UNIQUE per-column INTERNAL slots (so mimic
        # siblings get distinct slots) and wf_vel_s_* are indexed by internal slot. For
        # non-mimic these are byte-identical to the legacy reduced-vel tables.
        ("constexpr int wf_body_v_index[] = { " + _idsva_so_int_array(
            metadata['body_vint_index'] if is_mimic else metadata['body_v_index']) + " };"),
        ("constexpr int wf_vel_s_index[]  = { " + _idsva_so_int_array(
            metadata['int_s_index'] if is_mimic else metadata['vel_s_index']) + " };"),
        ("constexpr int wf_vel_s_sign[]   = { " + _idsva_so_int_array(
            metadata['int_s_sign'] if is_mimic else metadata['vel_s_sign']) + " };"),
    ] + ([
        # internal slot -> reduced (project NV-space) velocity slot, and -> mimic alpha.
        # Used by the qd/qdd reads (true reduced read, alpha-scaled) and the final fold.
        f"constexpr int wf_int_true[] = {{ {_idsva_so_int_array(metadata['int_true_vel'])} }};",
        "constexpr T wf_int_alpha[] = { " + ", ".join(
            "static_cast<T>(" + repr(a) + ")" for a in metadata['int_alpha']) + " };",
    ] if is_mimic else []) + [
        "",
    ])

    # ---- Init: zero output tensor in parallel + init S_agrav with thread 0.
    # Mimic: zero the internal 4*n_int^3 slab (d2tau_dq2 points at it); non-mimic: the
    # caller's 4*NV^3 output (SECOND_ORDER_TENSOR_SIZE).
    self.gen_add_parallel_loop("out_idx", f"4*{n_int}*{n_int}*{n_int}" if is_mimic else "SECOND_ORDER_TENSOR_SIZE")
    self.gen_add_code_line("d2tau_dq2[out_idx] = static_cast<T>(0);" if is_mimic else "s_idsva_so[out_idx] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_serial_ops()
    self.gen_add_code_line("// MATLAB convention: a_grav vector with a_grav[5] = GRAVITY (signed, e.g. -9.81).")
    self.gen_add_code_line("// The CUDA `gravity` parameter is the SIGNED gravitational acceleration (-9.81, the")
    self.gen_add_code_line("// unified GRiD convention), so a_grav[5] = gravity matches RBDReference.idsva_so_world_frame's `a_grav[5] = GRAVITY`.")
    self.gen_add_code_line("S_agrav[0] = static_cast<T>(0); S_agrav[1] = static_cast<T>(0); S_agrav[2] = static_cast<T>(0);")
    self.gen_add_code_line("S_agrav[3] = static_cast<T>(0); S_agrav[4] = static_cast<T>(0); S_agrav[5] = gravity;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # ---- Step 1: Build cumulative Xup. For floating-base root, Xup[0] = inv(X_local[0]).
    # BFS parent-dependent — keep sequential under thread-0 guard.
    self.gen_add_code_line("// Build cumulative Xup. Floating-base root: Xup[0] = inv(X_local[0]).")
    floating_base = self.robot.floating_base
    # The body walk is a REQUIRED recursion (Xup[jid] reads Xup[parent]), so the jid
    # loop stays serial; but within each body the work distributes across the block:
    # the root-init builds (rare, one-time) run on thread 0, while the dominant
    # Xup[jid] = X_local[jid] @ Xup[parent] 36-element matmul is element-independent and
    # runs as a block-parallel idx-over-36 loop. A per-body __syncthreads publishes
    # Xup[jid] before any child reads it. All threads execute the loop body.
    self.gen_add_code_line("for (int jid = 0; jid < NUM_BODIES; ++jid) {", True)
    self.gen_add_code_line("int parent = wf_parent[jid];")
    self.gen_add_code_line("if (parent < 0) {", True)
    self.gen_add_serial_ops()
    if floating_base:
        # Spatial Plücker `X = [E 0; B E]` with B = -E*r̂. Inverse:
        # `X^{-1} = [E^T 0; -E^T*B*E^T E^T]`. Both blocks of E transpose; the bottom-left
        # block computes -E^T * B * E^T.
        self.gen_add_code_lines([
            "// Floating-base root: Xup[0] = inv(X_local[0]).",
            "// Plücker form X = [E 0; B E] (column-major) with E orthogonal.",
            "// X^{-1} = [E^T 0; -E^T*B*E^T  E^T].",
            "// Step A: write the four blocks of inv into Xup[jid].",
            "// Top-right block (cols 3..5, rows 0..2) of inv is 0.",
            "for (int idx = 0; idx < 36; ++idx) Xup[jid*36 + idx] = static_cast<T>(0);",
            "// Top-left = E^T:  inv[a, b] = X[b, a] for a,b < 3.",
            "for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) Xup[jid*36 + a + 6*b] = s_XImats[jid*36 + b + 6*a];",
            "// Bottom-right = E^T: inv[a+3, b+3] = X[b+3, a+3] for a,b < 3.",
            "for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) Xup[jid*36 + (a + 3) + 6*(b + 3)] = s_XImats[jid*36 + (b + 3) + 6*(a + 3)];",
            "// Bottom-left = -E^T * B * E^T where B = X[3..6, 0..3].",
            "// Compute tmp1 = E^T * B (3x3 @ 3x3).",
            "T wf_tmp_invX_1[9];",
            "for (int a = 0; a < 3; ++a) {",
            "    for (int b = 0; b < 3; ++b) {",
            "        T acc = static_cast<T>(0);",
            "        for (int kk = 0; kk < 3; ++kk) acc += s_XImats[jid*36 + kk + 6*a] * s_XImats[jid*36 + (kk + 3) + 6*b];",
            "        wf_tmp_invX_1[a + 3*b] = acc;",
            "    }",
            "}",
            "// inv[a+3, b] = -(tmp1 @ E^T)[a, b] = -sum_kk tmp1[a, kk] * E^T[kk, b] = -sum_kk tmp1[a, kk] * X[b, kk].",
            "for (int a = 0; a < 3; ++a) {",
            "    for (int b = 0; b < 3; ++b) {",
            "        T acc = static_cast<T>(0);",
            "        for (int kk = 0; kk < 3; ++kk) acc += wf_tmp_invX_1[a + 3*kk] * s_XImats[jid*36 + b + 6*kk];",
            "        Xup[jid*36 + (a + 3) + 6*b] = -acc;",
            "    }",
            "}",
        ])
    else:
        self.gen_add_code_lines([
            "// Fixed-base root: Xup[0] = X_local[0] (no inversion).",
            "for (int idx = 0; idx < 36; ++idx) Xup[jid*36 + idx] = s_XImats[jid*36 + idx];",
        ])
    self.gen_add_end_control_flow()  # close thread-0 guard for the root-init build
    self.gen_add_end_control_flow()  # close if (parent < 0)
    self.gen_add_code_line("else {", True)
    self.gen_add_code_line("// Xup[jid] = X_local[jid] @ Xup[parent] (column-major matmul, block-parallel over the 36 elements).")
    self.gen_add_parallel_loop("idx", "36")
    self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int kk = 0; kk < 6; ++kk) acc += s_XImats[jid*36 + row + 6*kk] * Xup[parent*36 + kk + 6*col];")
    self.gen_add_code_line("Xup[jid*36 + idx] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()  # close else
    self.gen_add_sync()             # publish Xup[jid] before any child body reads it
    self.gen_add_end_control_flow()  # end Xup forward jid loop

    # ---- Step 2: Xdown[i] = inv(Xup[i]) — parallel over jid.
    self.gen_add_code_line("// Build Xdown[i] = inv(Xup[i]) using Plücker block inverse:")
    self.gen_add_code_line("// Xup = [E 0; B E]  =>  Xdown = [E^T 0; -E^T*B*E^T  E^T].")
    self.gen_add_parallel_loop("jid", "NUM_BODIES")
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) Xdown[jid*36 + idx] = static_cast<T>(0);")
    self.gen_add_code_line("// Top-left = E^T and Bottom-right = E^T.")
    self.gen_add_code_line("for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {", True)
    self.gen_add_code_line("Xdown[jid*36 + a + 6*b] = Xup[jid*36 + b + 6*a];")
    self.gen_add_code_line("Xdown[jid*36 + (a + 3) + 6*(b + 3)] = Xup[jid*36 + (b + 3) + 6*(a + 3)];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("// Bottom-left = -E^T * B * E^T where B = Xup[3..6, 0..3] (column-major).")
    self.gen_add_code_line("T wf_tmpEt_B[9];")
    self.gen_add_code_line("for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {", True)
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int kk = 0; kk < 3; ++kk) acc += Xup[jid*36 + kk + 6*a] * Xup[jid*36 + (kk + 3) + 6*b];")
    self.gen_add_code_line("wf_tmpEt_B[a + 3*b] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {", True)
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int kk = 0; kk < 3; ++kk) acc += wf_tmpEt_B[a + 3*kk] * Xup[jid*36 + b + 6*kk];")
    self.gen_add_code_line("Xdown[jid*36 + (a + 3) + 6*b] = -acc;")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # ---- Step 3: S_vel = Xdown @ S_local — parallel over vel (internal slot when mimic).
    self.gen_add_code_line("// S_vel[vel] = Xdown[body(vel)] @ S_local[vel] = sign * Xdown[body][:, s_index].")
    self.gen_add_parallel_loop("vel", f"{n_int}" if is_mimic else "NUM_VEL")
    # vel_to_body deduce from wf_body_v_index. Easier: pre-compute a vel_to_body table.
    self.gen_add_code_line("// Find body containing this vel.")
    self.gen_add_code_line("int jid = -1;")
    self.gen_add_code_line("for (int b = 0; b < NUM_BODIES && jid < 0; ++b) {", True)
    self.gen_add_code_line("for (int pos = wf_body_v_start[b]; pos < wf_body_v_start[b + 1]; ++pos) {", True)
    self.gen_add_code_line("if (wf_body_v_index[pos] == vel) { jid = b; break; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_code_line("int s_col = wf_vel_s_index[vel];")
    self.gen_add_code_line("T s_sign = static_cast<T>(wf_vel_s_sign[vel]);")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) S_vel[vel*6 + row] = s_sign * Xdown[jid*36 + s_col*6 + row];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # ---- Step 4: Forward sweep — v, a, f, IC, BC, psid, psidd, Sd.
    # Parent-dependent ACROSS bodies (v_w[jid]/a_w[jid] read the parent), so the
    # outer jid loop stays serial. WITHIN each body the work is distributed across
    # the block: the inherently-sequential 6-vector chain (v/a init, vJ/aJ, psid,
    # the v/a update, Sd) runs on thread 0, while the two dominant 36-element
    # matrix builds (IC = Xup^T I Xup, and BC) run as block-parallel idx-over-36
    # loops between syncs. Per-body temporaries that the parallel loops read across
    # threads (vJ, aJ, I_Xup, IC_v) live in the otherwise-idle Step-5 `scratch`
    # region rather than thread-0 stack. All threads execute the jid loop body so
    # every thread reaches each sync; the trailing sync closes the phase.
    self.gen_add_code_line("// Forward sweep: build v, a, f, IC, BC, psid, psidd, Sd.")
    self.gen_add_code_lines([
        "// Per-body forward-sweep temporaries borrowed from the (dead-until-Step-5) scratch region.",
        "T *fs_vJ   = scratch;        // 6",
        "T *fs_aJ   = fs_vJ   + 6;    // 6",
        "T *fs_I_Xup = fs_aJ  + 6;    // 36 (Ipool @ Xup, intermediate for IC)",
        "T *fs_IC_v = fs_I_Xup + 36;  // 6  (IC[jid] @ v[jid])",
    ])
    self.gen_add_code_line("for (int jid = 0; jid < NUM_BODIES; ++jid) {", True)
    self.gen_add_code_line("int parent = wf_parent[jid];")
    self.gen_add_code_line("int vc_begin = wf_body_v_start[jid]; int vc_count = wf_body_v_start[jid + 1] - vc_begin;")
    # --- Thread-0 reduction chain: v/a init + vJ/aJ build (a genuine length-vc serial
    # reduction). v_w[jid]/a_w[jid] hold the PRE-update values needed by psid/psidd.
    self.gen_add_serial_ops()
    # Initialize v[jid], a[jid]
    self.gen_add_code_line("if (parent < 0) {", True)
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) { v_w[jid*6 + row] = static_cast<T>(0); a_w[jid*6 + row] = -S_agrav[row]; }")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("else {", True)
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) { v_w[jid*6 + row] = v_w[parent*6 + row]; a_w[jid*6 + row] = a_w[parent*6 + row]; }")
    self.gen_add_end_control_flow()

    # vJ, aJ (reduction over the body's velocity columns — kept serial on thread 0).
    self.gen_add_code_line("// vJ = sum_p S_vel[p] * qd[p]; aJ = sum_p S_vel[p] * qdd[p].")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) { fs_vJ[row] = static_cast<T>(0); fs_aJ[row] = static_cast<T>(0); }")
    self.gen_add_code_line("for (int pos = vc_begin; pos < vc_begin + vc_count; ++pos) {", True)
    self.gen_add_code_line("int vel = wf_body_v_index[pos];")
    if is_mimic:
        # vel is an INTERNAL slot; read the (shared) reduced qd/qdd slot and scale by the
        # body's mimic multiplier (mirrors oracle's `_qd = alpha_i * qd[inds_v_true]`).
        self.gen_add_code_line("T a_qd = wf_int_alpha[vel] * s_qd[wf_int_true[vel]]; T a_qdd = wf_int_alpha[vel] * s_qdd[wf_int_true[vel]];")
        self.gen_add_code_line("for (int row = 0; row < 6; ++row) { fs_vJ[row] += S_vel[vel*6 + row] * a_qd; fs_aJ[row] += S_vel[vel*6 + row] * a_qdd; }")
    else:
        self.gen_add_code_line("for (int row = 0; row < 6; ++row) { fs_vJ[row] += S_vel[vel*6 + row] * s_qd[vel]; fs_aJ[row] += S_vel[vel*6 + row] * s_qdd[vel]; }")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("// aJ += crm(v[jid]) @ vJ.")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) fs_aJ[row] += crm_mul<T>(row, &v_w[jid*6], fs_vJ);")
    self.gen_add_end_control_flow()  # close thread-0 guard (v/a init + vJ/aJ reduction)
    self.gen_add_sync()             # publish pre-update v_w/a_w + fs_vJ/fs_aJ to all threads

    # psid/psidd are independent across the body's velocity columns: parallelize one
    # column per thread. Reads the PRE-update v_w[jid]/a_w[jid] (broadcast).
    self.gen_add_code_line("// psid[vel] = crm(v[jid]) @ S; psidd[vel] = crm(a[jid]) @ S + crm(v[jid]) @ psid (parallel over body velocity columns).")
    self.gen_add_parallel_loop("lane", "vc_count")
    self.gen_add_code_line("int vel = wf_body_v_index[vc_begin + lane];")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) psid_v[vel*6 + row] = crm_mul<T>(row, &v_w[jid*6], &S_vel[vel*6]);")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) psidd_v[vel*6 + row] = crm_mul<T>(row, &a_w[jid*6], &S_vel[vel*6]) + crm_mul<T>(row, &v_w[jid*6], &psid_v[vel*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()             # all psid/psidd reads of pre-update v_w done before the update below

    # Update v[jid] += vJ, a[jid] += aJ (thread 0; fs_vJ/fs_aJ already in shared).
    self.gen_add_serial_ops()
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) { v_w[jid*6 + row] += fs_vJ[row]; a_w[jid*6 + row] += fs_aJ[row]; }")
    self.gen_add_end_control_flow()  # close thread-0 guard (v/a update)
    self.gen_add_sync()             # publish post-update v_w[jid] to all threads

    # Sd[vel] = crm(v[jid]_new) @ S — independent across the body's velocity columns.
    self.gen_add_code_line("// Sd[vel] = crm(v[jid]_new) @ S (parallel over body velocity columns).")
    self.gen_add_parallel_loop("lane", "vc_count")
    self.gen_add_code_line("int vel = wf_body_v_index[vc_begin + lane];")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) Sd_vel[vel*6 + row] = crm_mul<T>(row, &v_w[jid*6], &S_vel[vel*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # --- IC[jid] = Xup[jid]^T @ I_body @ Xup[jid]: two block-parallel idx-over-36 builds.
    self.gen_add_code_line("// IC[jid] = Xup[jid]^T @ I_body @ Xup[jid] (block-parallel over the 36 elements).")
    self.gen_add_parallel_loop("idx", "36")
    self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int kk = 0; kk < 6; ++kk) acc += Ipool[jid*36 + row + 6*kk] * Xup[jid*36 + kk + 6*col];")
    self.gen_add_code_line("fs_I_Xup[idx] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_parallel_loop("idx", "36")
    self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int kk = 0; kk < 6; ++kk) acc += Xup[jid*36 + kk + 6*row] * fs_I_Xup[kk + 6*col];")
    self.gen_add_code_line("IC[jid*36 + idx] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # --- IC_v (6-vec), then BC[jid] (36, block-parallel), then f[jid] (6-vec).
    # IC_v rows are independent: parallelize one row per thread.
    self.gen_add_parallel_loop("row", "6")
    self.gen_add_code_line("fs_IC_v[row] = dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &v_w[jid*6]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # BC[jid] = crf(v) @ IC + icrf(IC @ v) - IC @ crm(v).
    self.gen_add_code_line("// BC[jid] = crf(v) @ IC + icrf(IC @ v) - IC @ crm(v) (block-parallel over the 36 elements).")
    self.gen_add_parallel_loop("idx", "36")
    self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("T crf_v_row[6];  T crm_v_col[6];")
    self.gen_add_code_line("for (int kk = 0; kk < 6; ++kk) crf_v_row[kk] = -crm<T>(kk + 6*row, &v_w[jid*6]);")
    self.gen_add_code_line("for (int kk = 0; kk < 6; ++kk) crm_v_col[kk] = crm<T>(kk + 6*col, &v_w[jid*6]);")
    self.gen_add_code_line("T t_crfv_IC = dot_prod<T, 6, 1, 1>(crf_v_row, &IC[jid*36 + 6*col]);")
    self.gen_add_code_line("T t_IC_crmv = dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], crm_v_col);")
    self.gen_add_code_line("BC[jid*36 + idx] = t_crfv_IC + icrf<T>(idx, fs_IC_v) - t_IC_crmv;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # f[jid] = IC @ a + crf(v) @ IC @ v. Rows are independent: parallelize one row per thread.
    self.gen_add_code_line("// f[jid] = IC[jid] @ a[jid] + crf(v[jid]) @ (IC[jid] @ v[jid]) (parallel over the 6 rows).")
    self.gen_add_parallel_loop("row", "6")
    self.gen_add_code_line("T crf_v_row2[6];")
    self.gen_add_code_line("for (int kk = 0; kk < 6; ++kk) crf_v_row2[kk] = -crm<T>(kk + 6*row, &v_w[jid*6]);")
    self.gen_add_code_line("f_w[jid*6 + row] = dot_prod<T, 6, 6, 1>(&IC[jid*36 + row], &a_w[jid*6]) + dot_prod<T, 6, 1, 1>(crf_v_row2, fs_IC_v);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_control_flow()  # end forward jid loop
    self.gen_add_sync()

    # ---- Step 5: Triple ancestor walk (reverse over bodies).
    # All threads run the outer (i, pp, j, tt) sequencing; inside, thread-0 builds
    # the (i, p) and (j, t) intermediates in shared mem, then a parallel loop
    # distributes the inner (k, rr) work across threads (each thread handles a
    # disjoint vel_k, so output writes are race-free).
    self.gen_add_code_line("// Triple ancestor walk: i over bodies (reverse), p over body i columns,")
    self.gen_add_code_line("// j over ancestors-or-self of i, t over body j columns, k over ancestors-of-j, r over body k columns.")
    self.gen_add_code_line("for (int i = NUM_BODIES - 1; i >= 0; --i) {", True)
    self.gen_add_code_line("for (int pp = wf_body_v_start[i]; pp < wf_body_v_start[i + 1]; ++pp) {", True)
    self.gen_add_code_line("int vel_i = wf_body_v_index[pp];")
    # Per-(i, p) intermediates: build A0..A7, Bphi, Bpsid in shared mem.
    # Pattern: thread-0 builds the small "global per-(i,p)" helper vectors
    # (ICi_S, ICi_psid, ..., A5_vec, A7_vec) into shared, then a parallel
    # loop over idx ∈ [0, 36) builds each A-matrix element using the shared
    # helpers. This unlocks ~10× wall-time speedup on the per-(i,p) work.
    self.gen_add_code_line("// === Per-(i, p) intermediates: thread-0 helpers, then parallel idx-over-36 build of A0..A7 ===")
    self.gen_add_code_line("T *S_p     = &S_vel[vel_i*6];")
    self.gen_add_code_line("T *Sd_p    = &Sd_vel[vel_i*6];")
    self.gen_add_code_line("T *psid_p  = &psid_v[vel_i*6];")
    self.gen_add_code_line("T *psidd_p = &psidd_v[vel_i*6];")
    # Helpers — small (6-vec each). Parallel over 7 "helper_id" × 6 "r" = 42 elements.
    # Each thread computes one element of one helper. A5_vec/A7_vec depend on
    # other helpers so they're emitted in a second parallel_loop after a sync.
    self.gen_add_parallel_loop("h_idx", "42")
    self.gen_add_code_lines([
        "int helper_id = h_idx / 6;",
        "int r = h_idx % 6;",
        "switch (helper_id) {",
        "  case 0: S_ICi_S[r]   = dot_prod<T, 6, 6, 1>(&IC[i*36 + r], S_p); break;",
        "  case 1: S_ICi_psid[r] = dot_prod<T, 6, 6, 1>(&IC[i*36 + r], psid_p); break;",
        "  case 2: S_ICi_psidd[r] = dot_prod<T, 6, 6, 1>(&IC[i*36 + r], psidd_p); break;",
        "  case 3: S_BCi_S[r]   = dot_prod<T, 6, 6, 1>(&BC[i*36 + r], S_p); break;",
        "  case 4: S_BCi_psid[r] = dot_prod<T, 6, 6, 1>(&BC[i*36 + r], psid_p); break;",
        "  case 5: S_BCiT_S[r]  = dot_prod<T, 6, 1, 1>(&BC[i*36 + 6*r], S_p); break;",
        "  case 6: {",
        "    T crf_S_row[6]; for (int kk = 0; kk < 6; ++kk) crf_S_row[kk] = -crm<T>(kk + 6*r, S_p);",
        "    S_crf_S_f_i[r] = dot_prod<T, 6, 1, 1>(crf_S_row, &f_w[i*6]);",
        "    break;",
        "  }",
        "}",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # A5_vec depends on S_BCi_psid + S_ICi_psidd + S_crf_S_f_i; A7_vec depends on S_BCi_S + IC@(psid+Sd).
    # Parallel over 12 = 6 (A5_vec) + 6 (A7_vec).
    self.gen_add_parallel_loop("v_idx", "12")
    self.gen_add_code_lines([
        "int r = v_idx % 6;",
        "if (v_idx < 6) {",
        "    S_A5_vec[r] = S_BCi_psid[r] + S_ICi_psidd[r] + S_crf_S_f_i[r];",
        "} else {",
        "    T s_sum = static_cast<T>(0);",
        "    for (int kk = 0; kk < 6; ++kk) s_sum += IC[i*36 + r + 6*kk] * (psid_p[kk] + Sd_p[kk]);",
        "    S_A7_vec[r] = S_BCi_S[r] + s_sum;",
        "}",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # Parallel build of A0/A1/Bphi/Bpsid (each over idx ∈ [0, 36)).
    self.gen_add_parallel_loop("idx", "36")
    self.gen_add_code_lines([
        "int row = idx % 6; int col = idx / 6;",
        "T crf_Sp_row[6]; for (int kk = 0; kk < 6; ++kk) crf_Sp_row[kk] = -crm<T>(kk + 6*row, S_p);",
        "T crm_Sp_col[6]; for (int kk = 0; kk < 6; ++kk) crm_Sp_col[kk] = crm<T>(kk + 6*col, S_p);",
        "T crf_psid_row[6]; for (int kk = 0; kk < 6; ++kk) crf_psid_row[kk] = -crm<T>(kk + 6*row, psid_p);",
        "T crm_psid_col[6]; for (int kk = 0; kk < 6; ++kk) crm_psid_col[kk] = crm<T>(kk + 6*col, psid_p);",
        "T t_crfSp_IC = dot_prod<T, 6, 1, 1>(crf_Sp_row, &IC[i*36 + 6*col]);",
        "T t_IC_crmSp = dot_prod<T, 6, 6, 1>(&IC[i*36 + row], crm_Sp_col);",
        "T t_crfpsid_IC = dot_prod<T, 6, 1, 1>(crf_psid_row, &IC[i*36 + 6*col]);",
        "T t_IC_crmpsid = dot_prod<T, 6, 6, 1>(&IC[i*36 + row], crm_psid_col);",
        "S_Bphi[idx]  = t_crfSp_IC  + icrf<T>(idx, S_ICi_S)    - t_IC_crmSp;",
        "S_Bpsid[idx] = t_crfpsid_IC + icrf<T>(idx, S_ICi_psid) - t_IC_crmpsid;",
        "S_A0[idx] = icrf<T>(idx, S_ICi_S);",
        "S_A1[idx] = t_crfSp_IC - t_IC_crmSp;",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # Parallel build of A2/A3/A4/A5/A6/A7 (depends on prior A0/A1/Bphi/Bpsid).
    self.gen_add_parallel_loop("idx", "36")
    self.gen_add_code_lines([
        "int row = idx % 6; int col = idx / 6;",
        "// A2 = 2*A0 - Bphi",
        "S_A2[idx] = static_cast<T>(2) * S_A0[idx] - S_Bphi[idx];",
        "// A3 = Bpsid + dot_matrix(BC[i], S_p)",
        "T crf_Sp_row[6]; for (int kk = 0; kk < 6; ++kk) crf_Sp_row[kk] = -crm<T>(kk + 6*row, S_p);",
        "T crm_Sp_col[6]; for (int kk = 0; kk < 6; ++kk) crm_Sp_col[kk] = crm<T>(kk + 6*col, S_p);",
        "T t_crfSp_BC = dot_prod<T, 6, 1, 1>(crf_Sp_row, &BC[i*36 + 6*col]);",
        "T t_BC_crmSp = dot_prod<T, 6, 6, 1>(&BC[i*36 + row], crm_Sp_col);",
        "S_A3[idx] = S_Bpsid[idx] + t_crfSp_BC - t_BC_crmSp;",
        "// A4 = icrf(BCiT_S)",
        "S_A4[idx] = icrf<T>(idx, S_BCiT_S);",
        "// A5 = icrf(A5_vec)",
        "S_A5[idx] = icrf<T>(idx, S_A5_vec);",
        "// A6 = crf(S_p) @ IC[i][:, col] + A0[idx]",
        "T t_crfSp_IC = dot_prod<T, 6, 1, 1>(crf_Sp_row, &IC[i*36 + 6*col]);",
        "S_A6[idx] = t_crfSp_IC + S_A0[idx];",
        "// A7 = icrf(A7_vec)",
        "S_A7[idx] = icrf<T>(idx, S_A7_vec);",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # j-loop (all threads run sequentially through ancestor chain).
    self.gen_add_code_line("int j = i;")
    self.gen_add_code_line("while (j >= 0) {", True)
    self.gen_add_code_line("for (int tt = wf_body_v_start[j]; tt < wf_body_v_start[j + 1]; ++tt) {", True)
    self.gen_add_code_line("int vel_j = wf_body_v_index[tt];")
    # Per-(j, t) intermediates: parallel build of u1..u12 over 72 element-slots
    # (12 u-vectors × 6 elements each). Each thread builds one element of one u.
    self.gen_add_code_line("T *S_t     = &S_vel[vel_j*6];")
    self.gen_add_code_line("T *Sd_t    = &Sd_vel[vel_j*6];")
    self.gen_add_code_line("T *psid_t  = &psid_v[vel_j*6];")
    self.gen_add_code_line("T *psidd_t = &psidd_v[vel_j*6];")
    self.gen_add_parallel_loop("u_idx", "72")
    self.gen_add_code_lines([
        "int which_u = u_idx / 6;",
        "int r = u_idx % 6;",
        "switch (which_u) {",
        "  case 0:  S_u1[r]  = dot_prod<T, 6, 1, 1>(&S_A3[6*r], S_t); break;",
        "  case 1:  S_u2[r]  = dot_prod<T, 6, 1, 1>(&S_A1[6*r], S_t); break;",
        "  case 2:  S_u3[r]  = dot_prod<T, 6, 6, 1>(&S_A3[r], psid_t) + dot_prod<T, 6, 6, 1>(&S_A1[r], psidd_t) + dot_prod<T, 6, 6, 1>(&S_A5[r], S_t); break;",
        "  case 3:  S_u4[r]  = dot_prod<T, 6, 6, 1>(&S_A6[r], S_t); break;",
        "  case 4:  S_u5[r]  = dot_prod<T, 6, 6, 1>(&S_A2[r], psid_t) + dot_prod<T, 6, 6, 1>(&S_A4[r], S_t); break;",
        "  case 5:  S_u6[r]  = dot_prod<T, 6, 6, 1>(&S_Bphi[r], psid_t) + dot_prod<T, 6, 6, 1>(&S_A7[r], S_t); break;",
        "  case 6: {",
        "    T psd_Sd[6]; for (int kk = 0; kk < 6; ++kk) psd_Sd[kk] = psid_t[kk] + Sd_t[kk];",
        "    S_u7[r] = dot_prod<T, 6, 6, 1>(&S_A3[r], S_t) + dot_prod<T, 6, 6, 1>(&S_A1[r], psd_Sd);",
        "    break;",
        "  }",
        "  case 7:  S_u8[r]  = dot_prod<T, 6, 6, 1>(&S_A4[r], S_t) - dot_prod<T, 6, 1, 1>(&S_Bphi[6*r], psid_t); break;",
        "  case 8:  S_u9[r]  = dot_prod<T, 6, 6, 1>(&S_A0[r], S_t); break;",
        "  case 9:  S_u10[r] = dot_prod<T, 6, 6, 1>(&S_Bphi[r], S_t); break;",
        "  case 10: S_u11[r] = dot_prod<T, 6, 1, 1>(&S_Bphi[6*r], S_t); break;",
        "  case 11: S_u12[r] = dot_prod<T, 6, 6, 1>(&S_A1[r], S_t); break;",
        "}",
    ])
    self.gen_add_end_control_flow()  # end parallel_loop u_idx
    self.gen_add_sync()

    # ---- Parallel inner (k, rr) walk ----
    # Each thread is assigned a unique kr_idx in [0, total_kr). The thread chases
    # the ancestor chain of j to map kr_idx -> (k_local, rr_local, vel_k). All
    # writes to output tensors contain `vel_k` in the index, and vel_k values are
    # disjoint across kr_idx (each velocity belongs to exactly one body in the
    # ancestor chain), so threads never race on output cells.
    self.gen_add_code_line("// Flatten (k, rr) iterations of the ancestor chain of j into a single index space.")
    self.gen_add_code_line("int wf_total_kr = 0;")
    self.gen_add_code_line("for (int _kk = j; _kk >= 0; _kk = wf_parent[_kk]) wf_total_kr += wf_body_v_start[_kk + 1] - wf_body_v_start[_kk];")
    self.gen_add_parallel_loop("kr_idx", "wf_total_kr")
    self.gen_add_code_lines([
        "// Map kr_idx -> (k, vel_k) by walking the chain.",
        "int k = -1; int vel_k = -1; int rr = -1;",
        "{",
        "int _seen = 0;",
        "for (int _kk = j; _kk >= 0; _kk = wf_parent[_kk]) {",
        "    int _nk = wf_body_v_start[_kk + 1] - wf_body_v_start[_kk];",
        "    if (kr_idx < _seen + _nk) {",
        "        rr = wf_body_v_start[_kk] + (kr_idx - _seen);",
        "        vel_k = wf_body_v_index[rr];",
        "        k = _kk;",
        "        break;",
        "    }",
        "    _seen += _nk;",
        "}",
        "}",
        "T *S_r     = &S_vel[vel_k*6];",
        "T *Sd_r    = &Sd_vel[vel_k*6];",
        "T *psid_r  = &psid_v[vel_k*6];",
        "T *psidd_r = &psidd_v[vel_k*6];",
        "T p1 = dot_prod<T, 6, 1, 1>(S_u11, psid_r);",
        "T p2 = dot_prod<T, 6, 1, 1>(S_u8, psid_r) + dot_prod<T, 6, 1, 1>(S_u9, psidd_r);",
        f"d2tau_dq2[(vel_i*{SO_N} + vel_j)*{SO_N} + vel_k] = p2;",
        f"d2tau_dvdq[(vel_i*{SO_N} + vel_k)*{SO_N} + vel_j] = -p1;",
        "",
        "T u1_psid_r  = dot_prod<T, 6, 1, 1>(S_u1, psid_r);",
        "T u2_psidd_r = dot_prod<T, 6, 1, 1>(S_u2, psidd_r);",
        "T u11_S_r    = dot_prod<T, 6, 1, 1>(S_u11, S_r);",
        "T u1_S_r     = dot_prod<T, 6, 1, 1>(S_u1, S_r);",
        "T u2_psd_Sd  = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) u2_psd_Sd += S_u2[kk] * (psid_r[kk] + Sd_r[kk]);",
        "T S_r_u3 = dot_prod<T, 6, 1, 1>(S_r, S_u3);",
        "T S_r_u4 = dot_prod<T, 6, 1, 1>(S_r, S_u4);",
        "T S_r_u5 = dot_prod<T, 6, 1, 1>(S_r, S_u5);",
        "T S_r_u6 = dot_prod<T, 6, 1, 1>(S_r, S_u6);",
        "T S_r_u7 = dot_prod<T, 6, 1, 1>(S_r, S_u7);",
        "T S_r_u9 = dot_prod<T, 6, 1, 1>(S_r, S_u9);",
        "T S_r_u10 = dot_prod<T, 6, 1, 1>(S_r, S_u10);",
        "T S_r_u12 = dot_prod<T, 6, 1, 1>(S_r, S_u12);",
        "T u9_psd_Sd = static_cast<T>(0);",
        "for (int kk = 0; kk < 6; ++kk) u9_psd_Sd += S_u9[kk] * (psid_r[kk] + Sd_r[kk]);",
        "",
        "if (j != i) {", True,
        "T dq_jki = u1_psid_r + u2_psidd_r;",
        f"d2tau_dq2[(vel_j*{SO_N} + vel_k)*{SO_N} + vel_i] = dq_jki;",
        f"d2tau_dq2[(vel_j*{SO_N} + vel_i)*{SO_N} + vel_k] = dq_jki;",
        f"d2tau_dvdq[(vel_j*{SO_N} + vel_k)*{SO_N} + vel_i] = p1;",
        f"d2tau_dvdq[(vel_j*{SO_N} + vel_i)*{SO_N} + vel_k] = u1_S_r + u2_psd_Sd;",
        f"d2tau_dqd2[(vel_j*{SO_N} + vel_k)*{SO_N} + vel_i] = u11_S_r;",
        f"d2tau_dqd2[(vel_j*{SO_N} + vel_i)*{SO_N} + vel_k] = u11_S_r;",
        f"dM_dq[(vel_k*{SO_N} + vel_j)*{SO_N} + vel_i] = S_r_u12;",
        f"dM_dq[(vel_j*{SO_N} + vel_k)*{SO_N} + vel_i] = S_r_u12;",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_code_lines([
        "if (k != j) {", True,
        f"d2tau_dq2[(vel_i*{SO_N} + vel_k)*{SO_N} + vel_j] = p2;",
        f"d2tau_dq2[(vel_k*{SO_N} + vel_i)*{SO_N} + vel_j] = S_r_u3;",
        f"d2tau_dqd2[(vel_i*{SO_N} + vel_j)*{SO_N} + vel_k] = -u11_S_r;",
        f"d2tau_dqd2[(vel_i*{SO_N} + vel_k)*{SO_N} + vel_j] = -u11_S_r;",
        f"d2tau_dvdq[(vel_i*{SO_N} + vel_j)*{SO_N} + vel_k] = S_r_u5 + u9_psd_Sd;",
        f"d2tau_dvdq[(vel_k*{SO_N} + vel_j)*{SO_N} + vel_i] = S_r_u6;",
        f"dM_dq[(vel_k*{SO_N} + vel_i)*{SO_N} + vel_j] = S_r_u9;",
        f"dM_dq[(vel_i*{SO_N} + vel_k)*{SO_N} + vel_j] = S_r_u9;",
        "if (j != i) {", True,
        f"d2tau_dq2[(vel_k*{SO_N} + vel_j)*{SO_N} + vel_i] = S_r_u3;",
        f"d2tau_dqd2[(vel_k*{SO_N} + vel_i)*{SO_N} + vel_j] = S_r_u10;",
        f"d2tau_dqd2[(vel_k*{SO_N} + vel_j)*{SO_N} + vel_i] = S_r_u10;",
        f"d2tau_dvdq[(vel_k*{SO_N} + vel_i)*{SO_N} + vel_j] = S_r_u7;",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_code_line("else {", True)
    self.gen_add_code_line(f"d2tau_dqd2[(vel_k*{SO_N} + vel_j)*{SO_N} + vel_i] = S_r_u4;")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_code_line("else {", True)
    self.gen_add_code_line(f"d2tau_dqd2[(vel_i*{SO_N} + vel_j)*{SO_N} + vel_k] = -dot_prod<T, 6, 1, 1>(S_u2, S_r);")
    self.gen_add_end_control_flow()

    self.gen_add_end_control_flow()  # end parallel_loop over kr_idx
    self.gen_add_sync()

    self.gen_add_end_control_flow()  # end tt loop
    self.gen_add_code_line("j = wf_parent[j];")
    self.gen_add_end_control_flow()  # end while j

    self.gen_add_end_control_flow()  # end pp loop

    # Aggregate IC, BC, f into parent — parallel over 78 elements (36 IC + 36 BC + 6 f).
    self.gen_add_code_line("// Bubble subtree-aggregated IC/BC/f up — parallel over 78 elements per body.")
    self.gen_add_code_line("int parent = wf_parent[i];")
    self.gen_add_code_line("if (parent >= 0) {", True)
    self.gen_add_parallel_loop("agg_idx", "78")
    self.gen_add_code_lines([
        "if (agg_idx < 36) {",
        "    IC[parent*36 + agg_idx] += IC[i*36 + agg_idx];",
        "} else if (agg_idx < 72) {",
        "    int b = agg_idx - 36;",
        "    BC[parent*36 + b] += BC[i*36 + b];",
        "} else {",
        "    int r = agg_idx - 72;",
        "    f_w[parent*6 + r] += f_w[i*6 + r];",
        "}",
    ])
    self.gen_add_end_control_flow()  # end parallel_loop agg_idx
    self.gen_add_end_control_flow()  # end if (parent >= 0)
    self.gen_add_sync()
    self.gen_add_end_control_flow()  # end i loop

    # Final transpose of d2tau_dvdq trailing axes: [τ, q, qd] -> [τ, qd, q].
    # Parallelize over (a_i, b_i) with each thread handling its (b_i, c_i) upper-triangle pairs.
    # Mimic: operate on the internal slab (SO_N stride/bound) BEFORE the fold (mirrors the
    # oracle which transposes the unreduced tensor, then einsum-folds).
    self.gen_add_code_line("// d2tau_dvdq was stored [τ, q, qd]; transpose trailing axes to match RBDReference convention.")
    self.gen_add_parallel_loop("ab_i", f"{SO_N}*{SO_N}")
    self.gen_add_code_line(f"int a_i = ab_i / {SO_N}; int b_i = ab_i % {SO_N};")
    self.gen_add_code_line(f"for (int c_i = b_i + 1; c_i < {SO_N}; ++c_i) {{", True)
    self.gen_add_code_line(f"T x = d2tau_dvdq[(a_i*{SO_N} + b_i)*{SO_N} + c_i];")
    self.gen_add_code_line(f"T y = d2tau_dvdq[(a_i*{SO_N} + c_i)*{SO_N} + b_i];")
    self.gen_add_code_line(f"d2tau_dvdq[(a_i*{SO_N} + b_i)*{SO_N} + c_i] = y;")
    self.gen_add_code_line(f"d2tau_dvdq[(a_i*{SO_N} + c_i)*{SO_N} + b_i] = x;")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    if is_mimic:
        # ---- Fold the internal 4*n_int^3 sweep to the reduced 4*NV^3 public output ----
        # public[true(i), true(j), true(k)] += alpha_i*alpha_j*alpha_k * internal[i,j,k],
        # summed over internal slots that share reduced slots (mimic siblings + the root's
        # 6 distinct columns fold identity). This is the oracle's
        # einsum('ia,ijk,jb,kc->abc', R, T, R, R) with R[i, true(i)] = alpha_i. Each of the
        # 4 tensor blocks folds independently; collisions (siblings) use atomicAdd.
        self.gen_add_code_line("// Mimic fold: reduce internal n_int^3 sweep to public NV^3 output.")
        # Zero the public NV^3 output (4 blocks).
        self.gen_add_parallel_loop("i", f"4*{NV**3}")
        self.gen_add_code_line("s_idsva_so_public[i] = static_cast<T>(0);")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        # Scatter-accumulate one thread per internal cell per block.
        self.gen_add_parallel_loop("idx", f"4*{n_int**3}")
        self.gen_add_code_line(f"int blk = idx / {n_int**3};")
        self.gen_add_code_line(f"int rem = idx % {n_int**3};")
        self.gen_add_code_line(f"int ii = rem / {n_int*n_int};")
        self.gen_add_code_line(f"int jj = (rem / {n_int}) % {n_int};")
        self.gen_add_code_line(f"int kk = rem % {n_int};")
        self.gen_add_code_line("T val = s_idsva_so_internal[idx];")
        self.gen_add_code_line("if (val != static_cast<T>(0)) {", True)
        self.gen_add_code_line("T w = wf_int_alpha[ii] * wf_int_alpha[jj] * wf_int_alpha[kk];")
        self.gen_add_code_line(f"int dst = blk*{NV**3} + wf_int_true[ii]*{NV*NV} + wf_int_true[jj]*{NV} + wf_int_true[kk];")
        self.gen_add_code_line("atomicAdd(&s_idsva_so_public[dst], w * val);")
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    # ---- mjx output-convention epilogue (floating non-mimic/skew only) ----
    # Runs at the very END of the inner, where s_temp (Xup/IC/BC/.../Xdown/v_w/a_w)
    # is DEAD and s_idsva_so holds the finalized pin SO tensors. s_XImats/s_q/s_qd/
    # s_qdd are live. Transforms the 4 pin tensors in s_idsva_so to mjx in place,
    # reusing the id-value / id-gradient / crba inners out of d_mjx_scratch.
    if mjx_inner:
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        _emit_idsva_so_mjx_output(self)
        self.gen_add_end_control_flow()

    self.gen_add_end_function()


def _emit_idsva_so_mjx_locals_lines(nv, nv3, fb):
    """Register-/stack-local recompute of the loop-invariant helpers, emitted at
    the TOP of every parallel-loop body so the verbatim formula lines downstream
    keep working unchanged. SAFE by construction: R is rebuilt from the base
    quaternion in s_q, the base source vectors from s_qd/s_qdd/s_vaf, and the
    tensor pointers are plain offsets into s_idsva_so/s_mjx_out — NOTHING is
    staged into shared scratch (no aliasing risk vs the spilled buffers). A few
    dozen flops per thread. The R-build math is copied verbatim from the
    single-thread original / the id-gradient reference."""
    return [
        # R (row-major R[3*i+j]) from the xyzw base quaternion s_q[3..6] — matches
        # mujoco_convention.rotation_from_quat_xyzw / _gen_mjx_build_R_lines exactly.
        "T qx = s_q[3], qy = s_q[4], qz = s_q[5], qw = s_q[6];",
        "T xx = qx*qx, yy = qy*qy, zz = qz*qz;",
        "T xy = qx*qy, xz = qx*qz, yz = qy*qz, wx = qw*qx, wy = qw*qy, wz = qw*qz;",
        "T R[9];",
        "R[0] = static_cast<T>(1) - static_cast<T>(2)*(yy+zz); R[1] = static_cast<T>(2)*(xy-wz);                    R[2] = static_cast<T>(2)*(xz+wy);",
        "R[3] = static_cast<T>(2)*(xy+wz);                    R[4] = static_cast<T>(1) - static_cast<T>(2)*(xx+zz); R[5] = static_cast<T>(2)*(yz-wx);",
        "R[6] = static_cast<T>(2)*(xz-wy);                    R[7] = static_cast<T>(2)*(yz+wx);                    R[8] = static_cast<T>(1) - static_cast<T>(2)*(xx+yy);",
        "T v_lin[3]   = {s_qd[0], s_qd[1], s_qd[2]};",
        "T omega[3]   = {s_qd[3], s_qd[4], s_qd[5]};",
        "T qdd_lin[3] = {s_qdd[0], s_qdd[1], s_qdd[2]};",
        "T tau_lin[3] = {s_vaf[" + str(fb + 3) + "], s_vaf[" + str(fb + 4) + "], s_vaf[" + str(fb + 5) + "]};",
        "// pin tensor blocks in s_idsva_so (row-major tensors):",
        "T *T_d2q   = s_idsva_so + " + str(0 * nv3) + ";",
        "T *T_d2qd  = s_idsva_so + " + str(1 * nv3) + ";",
        "T *T_cross = s_idsva_so + " + str(2 * nv3) + ";",
        "T *T_dM    = s_idsva_so + " + str(3 * nv3) + ";",
        "// mjx output blocks in s_mjx_out (same layout):",
        "T *O_d2q   = s_mjx_out + " + str(0 * nv3) + ";",
        "T *O_d2qd  = s_mjx_out + " + str(1 * nv3) + ";",
        "T *O_cross = s_mjx_out + " + str(2 * nv3) + ";",
        "T *O_dM    = s_mjx_out + " + str(3 * nv3) + ";",
    ]


def _emit_idsva_so_mjx_output(self):
    """Emit the MuJoCo (mjx) output-convention epilogue for idsva_so, transforming
    the 4 pin second-order tensors held in ``s_idsva_so`` to the mjx convention IN
    PLACE. Floating-base (non-mimic/skew) only; runs at the END of
    ``idsva_so_world_frame_inner`` where ``s_temp`` is dead.

    ``s_idsva_so`` is 4 contiguous NV^3 ROW-major blocks ``[i*NV*NV + j*NV + k]``:
      [0] d2tau_dq2[i,j,k]  [1] d2tau_dqd2[i,j,k]  [2] d2tau_dvdq[i,qd,q] (=cross)
      [3] dM_dq[i,l,m].

    Scratch (``d_mjx_scratch``, the SO-temp region of d_workspace; size 8*NV^3
    floats, dead post-assembly — see the kernel-body carve) is laid out:
      [0,            NV*NV)            s_M     (dense mass matrix, crba_inner)
      [NV*NV,      3*NV*NV)            s_dc_du (pin dtau_dq | dtau_dqd, col-major)
      [3*NV*NV,    3*NV*NV+18*NJ)      s_vaf   (id-value intermediate band)
      [INNERTMP,   INNERTMP+IDG_TEMP)  reused-inner scratch (id/crba/id-grad temps,
                                       used one-at-a-time; size = id-grad full band)
      [OUT,        OUT+4*NV*NV*NV)     mjx output band (4 slabs; copied back at end)

    The closed-form dM block and the explicit per-k form (a) for the other three
    tensors are transcribed verbatim from docs/open-tasks/mjx_proto/
    proto_idsva_so_emit_spec.py (validated <1e-13 vs second_order_id_pin_to_mjx)."""
    nv = self.robot.get_num_vel()
    NJ = self.robot.get_num_joints()
    nv2 = nv * nv
    nv3 = nv * nv * nv
    M_off = 0
    DCDU_off = nv2          # dtau_dq at [DCDU_off + j*nv + i]; dtau_dqd at [DCDU_off + nv2 + j*nv + i]
    DQ = DCDU_off
    DQD = DCDU_off + nv2
    VAF_off = 3 * nv2
    vaf_band = 18 * NJ
    INNERTMP_off = VAF_off + vaf_band
    idgrad_temp = self.gen_inverse_dynamics_gradient_inner_temp_mem_size()
    OUT_off = INNERTMP_off + idgrad_temp
    # tau base-linear = LINEAR part of base wrench f[0] = s_vaf[12*NJ+3..5] (spatial [ang;lin]).
    _fb = 12 * NJ

    self.gen_add_code_line("// === mjx output convention (floating-base idsva_so, form (a)) ===")
    self.gen_add_code_lines([
        "T *s_M     = d_mjx_scratch + " + str(M_off) + ";   // dense mass matrix (crba_inner)",
        "T *s_dc_du = d_mjx_scratch + " + str(DCDU_off) + ";   // pin dtau_dq | dtau_dqd",
        "T *s_vaf   = d_mjx_scratch + " + str(VAF_off) + ";   // id-value band (tau via f[0])",
        "T *s_mjx_tmp = d_mjx_scratch + " + str(INNERTMP_off) + ";   // reused-inner scratch (one at a time)",
        "T *s_mjx_out = d_mjx_scratch + " + str(OUT_off) + ";   // 4*NV^3 mjx output band",
    ])
    # 1) id-value (vaf), 2) crba (dense M), 3) id-gradient (pin dtau_dq|dtau_dqd).
    #    All read the live s_XImats (built for the converted q). Each uses s_mjx_tmp
    #    as its own scratch (TEMP_IN_SMEM=true with a global ptr — the inner just uses
    #    whatever s_temp points to). Sequential so the scratch band is reused.
    self.gen_add_code_line("// reuse the id-value / crba / id-gradient inners for tau, M, dtau_dq|dtau_dqd")
    self.gen_inverse_dynamics_inner_function_call(
        compute_c=False, use_qdd_input=True,
        updated_var_names=dict(s_vaf_name="s_vaf", s_temp_name="s_mjx_tmp", d_f_ext_name="nullptr"))
    self.gen_add_sync()
    self.gen_crba_inner_function_call(
        updated_var_names=dict(s_M_name="s_M", s_temp_name="s_mjx_tmp", d_workspace_name="nullptr"),
        temp_in_smem_expr="true")
    self.gen_add_sync()
    self.gen_inverse_dynamics_gradient_inner_function_call(
        dict(s_dc_du_name="s_dc_du", s_vaf_name="s_vaf", s_temp_name="s_mjx_tmp",
             d_temp_spill_name="nullptr", temp_spill_flag_name="false"))
    self.gen_add_sync()

    # ---- block-parallel assembly (was single-thread; ~50% overhead at batch) ----
    # The natural parallel axis is k, the OUTER tensor-slab index: each k owns its
    # own thread-stack scratch (work1/work2/d_*), reads only shared read-only
    # inputs (T_*, s_dc_du, s_M, R/v/omega/...), and writes ONLY O_*[...,k]. So the
    # per-k assembly (d2q/cross/d2qd) and dM Step1+2 are embarrassingly parallel
    # over k. Loop-invariant helpers (R + base vectors + tensor ptrs) are recomputed
    # REGISTER-/STACK-LOCAL at the top of every loop body (see _emit_idsva_so_mjx
    # _locals_lines) — NOTHING staged to shared scratch, so zero aliasing risk vs
    # the spilled buffers (cf. the fdsva_so spill bug). The dM Step3 base-rot frame
    # term (+= into O_dM[...,3+c]) reads O_dM written by Step1+2, so it follows a
    # sync and is parallelized over c in [0,3).
    _emit_idsva_so_mjx_perk_assembly(self, nv)        # parallel over k
    self.gen_add_sync()
    _emit_idsva_so_mjx_dM_closed_form(self, nv)       # parallel over k, then over c (sync between)
    self.gen_add_sync()
    # ---- copy the mjx output band back over s_idsva_so (block-parallel) ----
    self.gen_add_parallel_loop("ci", str(4 * nv3))
    self.gen_add_code_line("s_idsva_so[ci] = s_mjx_out[ci];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def _emit_idsva_so_mjx_perk_assembly(self, nv):
    """Per-k assembly of d2tau_dq2 / d2tau_dvdq(cross) / d2tau_dqd2 (form (a)).
    Transcribed op-for-op from proto_idsva_so_emit_spec.py. Matrices are flat
    col-major X[c*nv + r]; tensors row-major T[(i*nv + j)*nv + k]."""
    n = nv
    nv3 = n * n * n
    _fb = 12 * self.robot.get_num_joints()
    def t3(i, j, k):  # python ints
        return "((" + i + ")*" + str(n) + " + (" + j + "))*" + str(n) + " + (" + k + ")"
    # PARALLEL over k (the outer tensor-slab index): each k-iteration is fully
    # independent — its scratch is thread-stack-local, it reads only shared
    # read-only inputs, and writes only O_*[...,k]. The strided parallel loop
    # gives each thread a disjoint set of k slabs.
    self.gen_add_parallel_loop("k", str(n))
    # Loop-invariant helpers recomputed register-/stack-local at the top of the
    # body (R + base vectors + tensor ptrs) — nothing staged to shared scratch.
    self.gen_add_code_lines(_emit_idsva_so_mjx_locals_lines(n, nv3, _fb))
    # Per-k thread-stack scratch (col-major nv*nv): sensitivities d_dtdq/d_dtdqd/
    # d_Msens (persisted across the slab build), work1/work2 (A/inner_v/B reuse),
    # d_tau[nv], and the jqk/jvk/jak base-block columns. Declared per-iteration —
    # one private copy per thread, no cross-k sharing.
    self.gen_add_code_lines([
        "T d_dtdq[" + str(n * n) + "], d_dtdqd[" + str(n * n) + "], d_Msens[" + str(n * n) + "];",
        "T work1[" + str(n * n) + "], work2[" + str(n * n) + "];",
        "T d_tau[" + str(n) + "];",
        "T jqk[" + str(n) + "], jvk[" + str(n) + "], jak[" + str(n) + "];",
    ])
    # ---- jqk / jvk / jak (base-block sparse) ----
    self.gen_add_code_lines([
        "for (int q_ = 0; q_ < " + str(n) + "; q_++) { jqk[q_] = static_cast<T>(0); jvk[q_] = static_cast<T>(0); jak[q_] = static_cast<T>(0); }",
        "bool is_rot = (k >= 3 && k < 6);",
        "int a_rot = k - 3;",
        "if (k < 3) { jqk[0] = R[3*k+0]; jqk[1] = R[3*k+1]; jqk[2] = R[3*k+2]; }",
        "else { jqk[k] = static_cast<T>(1); }",
        "if (is_rot) {",
        "  // e_a x w with the exact component encoding (a = a_rot)",
        "  T evx = (a_rot==1)*( v_lin[2]) + (a_rot==2)*(-v_lin[1]);",
        "  T evy = (a_rot==0)*(-v_lin[2]) + (a_rot==2)*( v_lin[0]);",
        "  T evz = (a_rot==0)*( v_lin[1]) + (a_rot==1)*(-v_lin[0]);",
        "  jvk[0] = -evx; jvk[1] = -evy; jvk[2] = -evz;",
        "  T ovx = omega[1]*v_lin[2] - omega[2]*v_lin[1];",
        "  T ovy = omega[2]*v_lin[0] - omega[0]*v_lin[2];",
        "  T ovz = omega[0]*v_lin[1] - omega[1]*v_lin[0];",
        "  T eqx = (a_rot==1)*( qdd_lin[2]) + (a_rot==2)*(-qdd_lin[1]);",
        "  T eqy = (a_rot==0)*(-qdd_lin[2]) + (a_rot==2)*( qdd_lin[0]);",
        "  T eqz = (a_rot==0)*( qdd_lin[1]) + (a_rot==1)*(-qdd_lin[0]);",
        "  T eovx = (a_rot==1)*( ovz) + (a_rot==2)*(-ovy);",
        "  T eovy = (a_rot==0)*(-ovz) + (a_rot==2)*( ovx);",
        "  T eovz = (a_rot==0)*( ovy) + (a_rot==1)*(-ovx);",
        "  T oevx = omega[1]*evz - omega[2]*evy;",
        "  T oevy = omega[2]*evx - omega[0]*evz;",
        "  T oevz = omega[0]*evy - omega[1]*evx;",
        "  jak[0] = -eqx - eovx + oevx; jak[1] = -eqy - eovy + oevy; jak[2] = -eqz - eovz + oevz;",
        "}",
    ])
    # Rd = R @ skew(e_{a_rot}) (row-major), only when is_rot.
    # skew(e_a) columns: skew(e0)=[[0,0,0],[0,0,-1],[0,1,0]], etc. Rd = R@skew(e_a).
    # Rd[r,c] = sum_p R[r,p] skew[p,c]. We just compute Rd via the generic skew.
    self.gen_add_code_lines([
        "T Rd[9];",
        "for (int ii = 0; ii < 9; ii++) Rd[ii] = static_cast<T>(0);",
        "if (is_rot) {",
        "  // skew(e_a): sk[p][c]; Rd[r][c] = sum_p R[3r+p]*sk[p][c]",
        "  T sk[9]; for (int ii=0; ii<9; ii++) sk[ii]=static_cast<T>(0);",
        "  // skew(e_a)[p][c] = -eps(a,p,c): e0->(1,2)=-1,(2,1)=1; e1->(2,0)=-1,(0,2)=1; e2->(0,1)=-1,(1,0)=1",
        "  if (a_rot==0){ sk[1*3+2] = static_cast<T>(-1); sk[2*3+1] = static_cast<T>(1); }",
        "  if (a_rot==1){ sk[2*3+0] = static_cast<T>(-1); sk[0*3+2] = static_cast<T>(1); }",
        "  if (a_rot==2){ sk[0*3+1] = static_cast<T>(-1); sk[1*3+0] = static_cast<T>(1); }",
        "  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) {",
        "    T acc = static_cast<T>(0);",
        "    for (int p = 0; p < 3; p++) acc += R[3*r+p]*sk[3*p+c];",
        "    Rd[3*r+c] = acc;",
        "  }",
        "}",
    ])
    # ---- sensitivities ----
    self.gen_add_code_line("// sensitivities d_dtdq, d_dtdqd, d_Msens (col-major), d_tau")
    self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++) {", True)
    self.gen_add_code_line("for (int j = 0; j < " + str(n) + "; j++) {", True)
    self.gen_add_code_lines([
        "T s = static_cast<T>(0);",
        "for (int m = 0; m < " + str(n) + "; m++) s += T_d2q[" + t3("i", "j", "m") + "]*jqk[m];",
        "for (int nn = 0; nn < " + str(n) + "; nn++) s += T_cross[" + t3("i", "nn", "j") + "]*jvk[nn];",
        "for (int l = 0; l < " + str(n) + "; l++) s += T_dM[" + t3("i", "l", "j") + "]*jak[l];",
        "d_dtdq[j*" + str(n) + " + i] = s;",
        "T s2 = static_cast<T>(0);",
        "for (int m = 0; m < " + str(n) + "; m++) s2 += T_cross[" + t3("i", "j", "m") + "]*jqk[m];",
        "for (int nn = 0; nn < " + str(n) + "; nn++) s2 += T_d2qd[" + t3("i", "j", "nn") + "]*jvk[nn];",
        "d_dtdqd[j*" + str(n) + " + i] = s2;",
        "T s3 = static_cast<T>(0);",
        "for (int m = 0; m < " + str(n) + "; m++) s3 += T_dM[" + t3("i", "j", "m") + "]*jqk[m];",
        "d_Msens[j*" + str(n) + " + i] = s3;",
    ])
    self.gen_add_end_control_flow()  # j
    self.gen_add_code_lines([
        "T st = static_cast<T>(0);",
        "for (int j = 0; j < " + str(n) + "; j++) st += s_dc_du[" + str(0) + " + j*" + str(n) + " + i]*jqk[j]"
        " + s_dc_du[" + str(n * n) + " + j*" + str(n) + " + i]*jvk[j] + s_M[i + " + str(n) + "*j]*jak[j];",
        "d_tau[i] = st;",
    ])
    self.gen_add_end_control_flow()  # i
    # ---- assemble the three slabs ----
    _emit_idsva_so_mjx_slab_d2q(self, n)
    _emit_idsva_so_mjx_slab_cross(self, n)
    _emit_idsva_so_mjx_slab_d2qd(self, n)
    self.gen_add_end_control_flow()  # parallel k


# ----------------------------------------------------------------------------
# Per-tensor slab assemblers (single-thread, one k). All operate on flat
# col-major matrices X[c*nv + r]; the three jqk/jvk/jak vectors + Rd are in
# scope from _emit_idsva_so_mjx_perk_assembly. Output written to O_d2q/O_cross/
# O_d2qd row-major [(i*nv + j)*nv + k]. Transcribed from proto_idsva_so_emit_spec.py.
# ----------------------------------------------------------------------------
def _emit_idsva_so_mjx_slab_d2q(self, n):
    """d2tau_dq2[:,:,k] = MAIN + CORR (form a)."""
    N = str(n)
    self.gen_add_code_line("// --- d2tau_dq2 slab[:,:,k] = MAIN + CORR ---")
    self.gen_add_code_lines([
        "// MAIN: A = reframe_cols(d_dtdq); A[:,3+a] += d_dtdqd[:,0:3]@Jvq_col(a) + d_Msens[:,0:3]@Jaq_col(a)",
        "for (int r = 0; r < " + N + "; r++) {",
        "  T c0 = d_dtdq[0*" + N + "+r], c1 = d_dtdq[1*" + N + "+r], c2 = d_dtdq[2*" + N + "+r];",
        "  work1[0*" + N + "+r] = c0*R[0] + c1*R[1] + c2*R[2];",
        "  work1[1*" + N + "+r] = c0*R[3] + c1*R[4] + c2*R[5];",
        "  work1[2*" + N + "+r] = c0*R[6] + c1*R[7] + c2*R[8];",
        "  for (int c = 3; c < " + N + "; c++) work1[c*" + N + "+r] = d_dtdq[c*" + N + "+r];",
        "}",
    ])
    # couplings into angular cols 3+a (all rows). Jvq_col(a) = -(e_a x v_lin);
    # Jaq_col(a) = -(e_a x qdd_lin) - (e_a x (omega x v_lin)) + (omega x (e_a x v_lin)).
    self.gen_add_code_lines(_emit_jvq_jaq_block(n, "d_dtdqd", "d_Msens", "work1"))
    # g0 = rot_rows(work1) -> work2 ; then pref g0[0:3,3+a] += R(e_a x d_tau[0:3])
    self.gen_add_code_lines(_emit_rot_rows_block(n, "work1", "work2", "R"))
    self.gen_add_code_lines(_emit_pref_block(n, "work2", "d_tau", "R"))
    # CORR into work1: inner_v = reframe_cols(dtau_dq) + couplings ; Gd@inner_v -> O accum
    self.gen_add_code_lines([
        "// CORR: inner_v = reframe_cols(dtau_dq); inner_v[:,3+a] += dtau_dqd[:,0:3]@Jvq + M[:,0:3]@Jaq",
        "for (int r = 0; r < " + N + "; r++) {",
        "  T c0 = s_dc_du[0*" + N + "+r], c1 = s_dc_du[1*" + N + "+r], c2 = s_dc_du[2*" + N + "+r];",
        "  work1[0*" + N + "+r] = c0*R[0] + c1*R[1] + c2*R[2];",
        "  work1[1*" + N + "+r] = c0*R[3] + c1*R[4] + c2*R[5];",
        "  work1[2*" + N + "+r] = c0*R[6] + c1*R[7] + c2*R[8];",
        "  for (int c = 3; c < " + N + "; c++) work1[c*" + N + "+r] = s_dc_du[c*" + N + "+r];",
        "}",
    ])
    # couplings use s_dc_du[DQD..] (dtau_dqd) and s_M
    self.gen_add_code_lines(_emit_jvq_jaq_block(n, "(s_dc_du + " + str(n * n) + ")", "_SM_", "work1"))
    # corr (start) = Gd @ inner_v(work1) rows0:3 ; write directly into work2 accum via +=,
    # but g0 already in work2. We accumulate corr into work2.
    self.gen_add_code_lines(_emit_gd_inner_block(n, "work1", "work2"))
    # B (into work1, fresh) = (dtau_dq cols0:3 @ Rd^T) + dtau_dqd@Jvq_d + M@Jaq_d
    self.gen_add_code_lines([
        "// B = (dtau_dq cols0:3 @ Rd^T) + dtau_dqd@Jvq_d + M@Jaq_d ; then work2 += rot_rows(B)",
        "for (int r = 0; r < " + N + "; r++) {",
        "  T c0 = s_dc_du[0*" + N + "+r], c1 = s_dc_du[1*" + N + "+r], c2 = s_dc_du[2*" + N + "+r];",
        "  work1[0*" + N + "+r] = c0*Rd[0] + c1*Rd[1] + c2*Rd[2];",  # Rd^T[cp,c]=Rd[c,cp]=Rd[3c+cp]
        "  work1[1*" + N + "+r] = c0*Rd[3] + c1*Rd[4] + c2*Rd[5];",
        "  work1[2*" + N + "+r] = c0*Rd[6] + c1*Rd[7] + c2*Rd[8];",
        "  for (int c = 3; c < " + N + "; c++) work1[c*" + N + "+r] = static_cast<T>(0);",
        "}",
    ])
    self.gen_add_code_lines(_emit_jvqd_jaqd_block(n, "(s_dc_du + " + str(n * n) + ")", "_SM_", "work1"))
    self.gen_add_code_lines(_emit_rot_rows_accum_block(n, "work1", "work2", "R"))
    # pref_Rd: work2[0:3,3+a] += Rd(e_a x tau_lin)
    self.gen_add_code_lines(_emit_pref_block(n, "work2", "tau_lin", "Rd"))
    # write slab: O_d2q[i,j,k] = work2[j*nv+i]
    self.gen_add_code_lines([
        "for (int i = 0; i < " + N + "; i++) for (int j = 0; j < " + N + "; j++)",
        "  O_d2q[(i*" + N + " + j)*" + N + " + k] = work2[j*" + N + " + i];",
    ])


def _emit_idsva_so_mjx_slab_cross(self, n):
    """d2tau_dvdq (cross)[:,:,k] = rot_rows(reframe_cols(d_dtdqd) + d_M@Jav) + Gd@inner1_v + rot_rows(B1)."""
    N = str(n)
    self.gen_add_code_line("// --- d2tau_dvdq (cross) slab[:,:,k] ---")
    # A1 = reframe_cols(d_dtdqd); + d_Msens@Jav (lin cols 0+a, ang cols 3+a)
    self.gen_add_code_lines([
        "for (int r = 0; r < " + N + "; r++) {",
        "  T c0 = d_dtdqd[0*" + N + "+r], c1 = d_dtdqd[1*" + N + "+r], c2 = d_dtdqd[2*" + N + "+r];",
        "  work1[0*" + N + "+r] = c0*R[0] + c1*R[1] + c2*R[2];",
        "  work1[1*" + N + "+r] = c0*R[3] + c1*R[4] + c2*R[5];",
        "  work1[2*" + N + "+r] = c0*R[6] + c1*R[7] + c2*R[8];",
        "  for (int c = 3; c < " + N + "; c++) work1[c*" + N + "+r] = d_dtdqd[c*" + N + "+r];",
        "}",
    ])
    self.gen_add_code_lines(_emit_jav_block(n, "d_Msens", "work1", "R"))   # Jav uses R^T e_a
    self.gen_add_code_lines(_emit_rot_rows_block(n, "work1", "work2", "R"))   # g1
    # inner1_v = reframe_cols(dtau_dqd) + M@Jav ; Gd@inner1_v accum into work2
    self.gen_add_code_lines([
        "for (int r = 0; r < " + N + "; r++) {",
        "  T c0 = s_dc_du[" + str(n * n) + "+0*" + N + "+r], c1 = s_dc_du[" + str(n * n) + "+1*" + N + "+r], c2 = s_dc_du[" + str(n * n) + "+2*" + N + "+r];",
        "  work1[0*" + N + "+r] = c0*R[0] + c1*R[1] + c2*R[2];",
        "  work1[1*" + N + "+r] = c0*R[3] + c1*R[4] + c2*R[5];",
        "  work1[2*" + N + "+r] = c0*R[6] + c1*R[7] + c2*R[8];",
        "  for (int c = 3; c < " + N + "; c++) work1[c*" + N + "+r] = s_dc_du[" + str(n * n) + "+c*" + N + "+r];",
        "}",
    ])
    self.gen_add_code_lines(_emit_jav_block(n, "_SM_", "work1", "R"))
    self.gen_add_code_lines(_emit_gd_inner_block(n, "work1", "work2"))
    # B1 = (dtau_dqd cols0:3 @ Rd^T) + M@Jav_d ; work2 += rot_rows(B1)
    self.gen_add_code_lines([
        "for (int r = 0; r < " + N + "; r++) {",
        "  T c0 = s_dc_du[" + str(n * n) + "+0*" + N + "+r], c1 = s_dc_du[" + str(n * n) + "+1*" + N + "+r], c2 = s_dc_du[" + str(n * n) + "+2*" + N + "+r];",
        "  work1[0*" + N + "+r] = c0*Rd[0] + c1*Rd[1] + c2*Rd[2];",
        "  work1[1*" + N + "+r] = c0*Rd[3] + c1*Rd[4] + c2*Rd[5];",
        "  work1[2*" + N + "+r] = c0*Rd[6] + c1*Rd[7] + c2*Rd[8];",
        "  for (int c = 3; c < " + N + "; c++) work1[c*" + N + "+r] = static_cast<T>(0);",
        "}",
    ])
    self.gen_add_code_lines(_emit_javd_block(n, "_SM_", "work1"))
    self.gen_add_code_lines(_emit_rot_rows_accum_block(n, "work1", "work2", "R"))
    self.gen_add_code_lines([
        "for (int i = 0; i < " + N + "; i++) for (int j = 0; j < " + N + "; j++)",
        "  O_cross[(i*" + N + " + j)*" + N + " + k] = work2[j*" + N + " + i];",
    ])


def _emit_idsva_so_mjx_slab_d2qd(self, n):
    """d2tau_dqd2[:,:,k] = rot_rows(reframe_cols(dv_dtdqd)) + rot_rows(M@Jav_dv).
    jvvk = G^T col k (= jqk: base-mixed for k<3, else e_k)."""
    N = str(n)
    self.gen_add_code_line("// --- d2tau_dqd2 slab[:,:,k] (qvel perturbation, R fixed) ---")
    # dv_dtdqd[i,j] = sum_n d2qd[i,j,n]*jvvk[n] ; jvvk == jqk (G^T col k).
    self.gen_add_code_line("for (int i = 0; i < " + N + "; i++) for (int j = 0; j < " + N + "; j++) {", True)
    self.gen_add_code_lines([
        "T s = static_cast<T>(0);",
        "for (int nn = 0; nn < " + N + "; nn++) s += T_d2qd[(i*" + N + " + j)*" + N + " + nn]*jqk[nn];",
        "work1[j*" + N + " + i] = s;",
    ])
    self.gen_add_end_control_flow()
    # A2 = reframe_cols(work1) (in place into work2 then rot) — do reframe into work2 cols
    self.gen_add_code_lines([
        "for (int r = 0; r < " + N + "; r++) {",
        "  T c0 = work1[0*" + N + "+r], c1 = work1[1*" + N + "+r], c2 = work1[2*" + N + "+r];",
        "  work2[0*" + N + "+r] = c0*R[0] + c1*R[1] + c2*R[2];",
        "  work2[1*" + N + "+r] = c0*R[3] + c1*R[4] + c2*R[5];",
        "  work2[2*" + N + "+r] = c0*R[6] + c1*R[7] + c2*R[8];",
        "  for (int c = 3; c < " + N + "; c++) work2[c*" + N + "+r] = work1[c*" + N + "+r];",
        "}",
    ])
    # g1v = rot_rows(work2) -> work1
    self.gen_add_code_lines(_emit_rot_rows_block(n, "work2", "work1", "R"))
    # MJ = M @ Jav_dv with dqv_v = jqk[0:3], dqv_om = jqk[3:6]
    self.gen_add_code_lines([
        "// MJ = M[:,0:3] @ Jav_dv : lin col a = -(dqv_om x R^T e_a), ang col a = -(e_a x dqv_v)",
        "for (int rr = 0; rr < " + N + "*" + N + "; rr++) work2[rr] = static_cast<T>(0);",
        "for (int a = 0; a < 3; a++) {",
        "  T rte0 = R[3*a+0], rte1 = R[3*a+1], rte2 = R[3*a+2];",   # R^T e_a = row a of R
        "  T dvo0 = jqk[3], dvo1 = jqk[4], dvo2 = jqk[5];",
        "  T dvv0 = jqk[0], dvv1 = jqk[1], dvv2 = jqk[2];",
        "  T jl0 = -(dvo1*rte2 - dvo2*rte1);",
        "  T jl1 = -(dvo2*rte0 - dvo0*rte2);",
        "  T jl2 = -(dvo0*rte1 - dvo1*rte0);",
        "  T evx = (a==1)*( dvv2) + (a==2)*(-dvv1);",
        "  T evy = (a==0)*(-dvv2) + (a==2)*( dvv0);",
        "  T evz = (a==0)*( dvv1) + (a==1)*(-dvv0);",
        "  T jn0 = -evx, jn1 = -evy, jn2 = -evz;",
        "  for (int r = 0; r < " + N + "; r++) {",
        "    T mr0 = s_M[r + " + N + "*0], mr1 = s_M[r + " + N + "*1], mr2 = s_M[r + " + N + "*2];",
        "    work2[(0+a)*" + N + "+r] += mr0*jl0 + mr1*jl1 + mr2*jl2;",
        "    work2[(3+a)*" + N + "+r] += mr0*jn0 + mr1*jn1 + mr2*jn2;",
        "  }",
        "}",
    ])
    # work1 += rot_rows(work2)
    self.gen_add_code_lines(_emit_rot_rows_accum_block(n, "work2", "work1", "R"))
    self.gen_add_code_lines([
        "for (int i = 0; i < " + N + "; i++) for (int j = 0; j < " + N + "; j++)",
        "  O_d2qd[(i*" + N + " + j)*" + N + " + k] = work1[j*" + N + " + i];",
    ])


# ---- reusable inline blocks (return list[str]); _SM_ marks the dense s_M source ----
def _emit_jvq_jaq_block(n, src_dtqd, src_M, dest):
    """dest[:,3+a] += src_dtqd[:,0:3]@Jvq_col(a) + src_M[:,0:3]@Jaq_col(a) (all rows)."""
    N = str(n)
    sm = "s_M[r + " + N + "*0]" if src_M == "_SM_" else src_M + "[0*" + N + "+r]"
    sm1 = "s_M[r + " + N + "*1]" if src_M == "_SM_" else src_M + "[1*" + N + "+r]"
    sm2 = "s_M[r + " + N + "*2]" if src_M == "_SM_" else src_M + "[2*" + N + "+r]"
    return [
        "for (int a = 0; a < 3; a++) {",
        "  T evx = (a==1)*( v_lin[2]) + (a==2)*(-v_lin[1]);",
        "  T evy = (a==0)*(-v_lin[2]) + (a==2)*( v_lin[0]);",
        "  T evz = (a==0)*( v_lin[1]) + (a==1)*(-v_lin[0]);",
        "  T jvc0 = -evx, jvc1 = -evy, jvc2 = -evz;",
        "  T ovx = omega[1]*v_lin[2] - omega[2]*v_lin[1];",
        "  T ovy = omega[2]*v_lin[0] - omega[0]*v_lin[2];",
        "  T ovz = omega[0]*v_lin[1] - omega[1]*v_lin[0];",
        "  T eqx = (a==1)*( qdd_lin[2]) + (a==2)*(-qdd_lin[1]);",
        "  T eqy = (a==0)*(-qdd_lin[2]) + (a==2)*( qdd_lin[0]);",
        "  T eqz = (a==0)*( qdd_lin[1]) + (a==1)*(-qdd_lin[0]);",
        "  T eovx = (a==1)*( ovz) + (a==2)*(-ovy);",
        "  T eovy = (a==0)*(-ovz) + (a==2)*( ovx);",
        "  T eovz = (a==0)*( ovy) + (a==1)*(-ovx);",
        "  T oevx = omega[1]*evz - omega[2]*evy;",
        "  T oevy = omega[2]*evx - omega[0]*evz;",
        "  T oevz = omega[0]*evy - omega[1]*evx;",
        "  T jac0 = -eqx - eovx + oevx, jac1 = -eqy - eovy + oevy, jac2 = -eqz - eovz + oevz;",
        "  for (int r = 0; r < " + N + "; r++) {",
        "    T q0 = " + src_dtqd + "[0*" + N + "+r], q1 = " + src_dtqd + "[1*" + N + "+r], q2 = " + src_dtqd + "[2*" + N + "+r];",
        "    T m0 = " + sm + ", m1 = " + sm1 + ", m2 = " + sm2 + ";",
        "    " + dest + "[(3+a)*" + N + "+r] += q0*jvc0 + q1*jvc1 + q2*jvc2 + m0*jac0 + m1*jac1 + m2*jac2;",
        "  }",
        "}",
    ]


def _emit_jvqd_jaqd_block(n, src_dtqd, src_M, dest):
    """dest[:,3+a] += src_dtqd[:,0:3]@Jvq_d_col(a) + src_M[:,0:3]@Jaq_d_col(a).
    d_v=jvk[0:3], d_om=jvk[3:6], d_qdd=jak[0:3]."""
    N = str(n)
    sm = "s_M[r + " + N + "*0]" if src_M == "_SM_" else src_M + "[0*" + N + "+r]"
    sm1 = "s_M[r + " + N + "*1]" if src_M == "_SM_" else src_M + "[1*" + N + "+r]"
    sm2 = "s_M[r + " + N + "*2]" if src_M == "_SM_" else src_M + "[2*" + N + "+r]"
    return [
        "for (int a = 0; a < 3; a++) {",
        "  T dv0 = jvk[0], dv1 = jvk[1], dv2 = jvk[2];",
        "  T dom0 = jvk[3], dom1 = jvk[4], dom2 = jvk[5];",
        "  T dqd0 = jak[0], dqd1 = jak[1], dqd2 = jak[2];",
        # Jvq_d col a = -(e_a x d_v)
        "  T edvx = (a==1)*( dv2) + (a==2)*(-dv1);",
        "  T edvy = (a==0)*(-dv2) + (a==2)*( dv0);",
        "  T edvz = (a==0)*( dv1) + (a==1)*(-dv0);",
        "  T jvc0 = -edvx, jvc1 = -edvy, jvc2 = -edvz;",
        # Jaq_d col a = -(e_a x d_qdd) - (e_a x (d_om x v_lin + omega x d_v)) + (d_om x (e_a x v_lin)) + (omega x (e_a x d_v))
        "  T edqx = (a==1)*( dqd2) + (a==2)*(-dqd1);",
        "  T edqy = (a==0)*(-dqd2) + (a==2)*( dqd0);",
        "  T edqz = (a==0)*( dqd1) + (a==1)*(-dqd0);",
        "  T mid0 = (dom1*v_lin[2]-dom2*v_lin[1]) + (omega[1]*dv2-omega[2]*dv1);",
        "  T mid1 = (dom2*v_lin[0]-dom0*v_lin[2]) + (omega[2]*dv0-omega[0]*dv2);",
        "  T mid2 = (dom0*v_lin[1]-dom1*v_lin[0]) + (omega[0]*dv1-omega[1]*dv0);",
        "  T emx = (a==1)*( mid2) + (a==2)*(-mid1);",
        "  T emy = (a==0)*(-mid2) + (a==2)*( mid0);",
        "  T emz = (a==0)*( mid1) + (a==1)*(-mid0);",
        "  T evx = (a==1)*( v_lin[2]) + (a==2)*(-v_lin[1]);",   # e_a x v_lin
        "  T evy = (a==0)*(-v_lin[2]) + (a==2)*( v_lin[0]);",
        "  T evz = (a==0)*( v_lin[1]) + (a==1)*(-v_lin[0]);",
        "  T edvx2 = (a==1)*( dv2) + (a==2)*(-dv1);",            # e_a x d_v
        "  T edvy2 = (a==0)*(-dv2) + (a==2)*( dv0);",
        "  T edvz2 = (a==0)*( dv1) + (a==1)*(-dv0);",
        "  T t1x = dom1*evz - dom2*evy;",                        # d_om x (e_a x v_lin)
        "  T t1y = dom2*evx - dom0*evz;",
        "  T t1z = dom0*evy - dom1*evx;",
        "  T t2x = omega[1]*edvz2 - omega[2]*edvy2;",            # omega x (e_a x d_v)
        "  T t2y = omega[2]*edvx2 - omega[0]*edvz2;",
        "  T t2z = omega[0]*edvy2 - omega[1]*edvx2;",
        "  T jac0 = -edqx - emx + t1x + t2x;",
        "  T jac1 = -edqy - emy + t1y + t2y;",
        "  T jac2 = -edqz - emz + t1z + t2z;",
        "  for (int r = 0; r < " + N + "; r++) {",
        "    T q0 = " + src_dtqd + "[0*" + N + "+r], q1 = " + src_dtqd + "[1*" + N + "+r], q2 = " + src_dtqd + "[2*" + N + "+r];",
        "    T m0 = " + sm + ", m1 = " + sm1 + ", m2 = " + sm2 + ";",
        "    " + dest + "[(3+a)*" + N + "+r] += q0*jvc0 + q1*jvc1 + q2*jvc2 + m0*jac0 + m1*jac1 + m2*jac2;",
        "  }",
        "}",
    ]


def _emit_jav_block(n, src_M, dest, Rname):
    """dest[:,0+a] += src_M[:,0:3]@(-(omega x R^T e_a)) ; dest[:,3+a] += src_M[:,0:3]@(-(e_a x v_lin))."""
    N = str(n)
    sm = "s_M[r + " + N + "*0]" if src_M == "_SM_" else src_M + "[0*" + N + "+r]"
    sm1 = "s_M[r + " + N + "*1]" if src_M == "_SM_" else src_M + "[1*" + N + "+r]"
    sm2 = "s_M[r + " + N + "*2]" if src_M == "_SM_" else src_M + "[2*" + N + "+r]"
    return [
        "for (int a = 0; a < 3; a++) {",
        "  T rte0 = " + Rname + "[3*a+0], rte1 = " + Rname + "[3*a+1], rte2 = " + Rname + "[3*a+2];",
        "  T jl0 = -(omega[1]*rte2 - omega[2]*rte1);",
        "  T jl1 = -(omega[2]*rte0 - omega[0]*rte2);",
        "  T jl2 = -(omega[0]*rte1 - omega[1]*rte0);",
        "  T evx = (a==1)*( v_lin[2]) + (a==2)*(-v_lin[1]);",
        "  T evy = (a==0)*(-v_lin[2]) + (a==2)*( v_lin[0]);",
        "  T evz = (a==0)*( v_lin[1]) + (a==1)*(-v_lin[0]);",
        "  T jn0 = -evx, jn1 = -evy, jn2 = -evz;",
        "  for (int r = 0; r < " + N + "; r++) {",
        "    T m0 = " + sm + ", m1 = " + sm1 + ", m2 = " + sm2 + ";",
        "    " + dest + "[(0+a)*" + N + "+r] += m0*jl0 + m1*jl1 + m2*jl2;",
        "    " + dest + "[(3+a)*" + N + "+r] += m0*jn0 + m1*jn1 + m2*jn2;",
        "  }",
        "}",
    ]


def _emit_javd_block(n, src_M, dest):
    """dest[:,0+a] += M[:,0:3]@(-(d_om x R^T e_a) - (omega x Rd^T e_a)) ;
       dest[:,3+a] += M[:,0:3]@(-(e_a x d_v)). d_v=jvk[0:3], d_om=jvk[3:6]."""
    N = str(n)
    sm = "s_M[r + " + N + "*0]"
    sm1 = "s_M[r + " + N + "*1]"
    sm2 = "s_M[r + " + N + "*2]"
    return [
        "for (int a = 0; a < 3; a++) {",
        "  T dv0 = jvk[0], dv1 = jvk[1], dv2 = jvk[2];",
        "  T dom0 = jvk[3], dom1 = jvk[4], dom2 = jvk[5];",
        "  T rte0 = R[3*a+0], rte1 = R[3*a+1], rte2 = R[3*a+2];",
        "  T rdte0 = Rd[3*a+0], rdte1 = Rd[3*a+1], rdte2 = Rd[3*a+2];",
        "  T jl0 = -((dom1*rte2 - dom2*rte1) + (omega[1]*rdte2 - omega[2]*rdte1));",
        "  T jl1 = -((dom2*rte0 - dom0*rte2) + (omega[2]*rdte0 - omega[0]*rdte2));",
        "  T jl2 = -((dom0*rte1 - dom1*rte0) + (omega[0]*rdte1 - omega[1]*rdte0));",
        "  T edvx = (a==1)*( dv2) + (a==2)*(-dv1);",
        "  T edvy = (a==0)*(-dv2) + (a==2)*( dv0);",
        "  T edvz = (a==0)*( dv1) + (a==1)*(-dv0);",
        "  T jn0 = -edvx, jn1 = -edvy, jn2 = -edvz;",
        "  for (int r = 0; r < " + N + "; r++) {",
        "    T m0 = " + sm + ", m1 = " + sm1 + ", m2 = " + sm2 + ";",
        "    " + dest + "[(0+a)*" + N + "+r] += m0*jl0 + m1*jl1 + m2*jl2;",
        "    " + dest + "[(3+a)*" + N + "+r] += m0*jn0 + m1*jn1 + m2*jn2;",
        "  }",
        "}",
    ]


def _emit_rot_rows_block(n, srcX, dstX, Rname):
    """dstX = copy(srcX) then rows0:3 <- Rname @ rows."""
    N = str(n)
    return [
        "for (int c = 0; c < " + N + "; c++) {",
        "  T m0 = " + srcX + "[c*" + N + "+0], m1 = " + srcX + "[c*" + N + "+1], m2 = " + srcX + "[c*" + N + "+2];",
        "  " + dstX + "[c*" + N + "+0] = " + Rname + "[0]*m0 + " + Rname + "[1]*m1 + " + Rname + "[2]*m2;",
        "  " + dstX + "[c*" + N + "+1] = " + Rname + "[3]*m0 + " + Rname + "[4]*m1 + " + Rname + "[5]*m2;",
        "  " + dstX + "[c*" + N + "+2] = " + Rname + "[6]*m0 + " + Rname + "[7]*m1 + " + Rname + "[8]*m2;",
        "  for (int r = 3; r < " + N + "; r++) " + dstX + "[c*" + N + "+r] = " + srcX + "[c*" + N + "+r];",
        "}",
    ]


def _emit_rot_rows_accum_block(n, srcX, dstX, Rname):
    """dstX += rot_rows(srcX): rows0:3 += Rname@srcX[0:3,:]; rows3+ += srcX[3+,:]."""
    N = str(n)
    return [
        "for (int c = 0; c < " + N + "; c++) {",
        "  T m0 = " + srcX + "[c*" + N + "+0], m1 = " + srcX + "[c*" + N + "+1], m2 = " + srcX + "[c*" + N + "+2];",
        "  " + dstX + "[c*" + N + "+0] += " + Rname + "[0]*m0 + " + Rname + "[1]*m1 + " + Rname + "[2]*m2;",
        "  " + dstX + "[c*" + N + "+1] += " + Rname + "[3]*m0 + " + Rname + "[4]*m1 + " + Rname + "[5]*m2;",
        "  " + dstX + "[c*" + N + "+2] += " + Rname + "[6]*m0 + " + Rname + "[7]*m1 + " + Rname + "[8]*m2;",
        "  for (int r = 3; r < " + N + "; r++) " + dstX + "[c*" + N + "+r] += " + srcX + "[c*" + N + "+r];",
        "}",
    ]


def _emit_gd_inner_block(n, srcX, dstX):
    """dstX += Gd@srcX: rows0:3 += Rd@srcX[0:3,:] (other rows unchanged)."""
    N = str(n)
    return [
        "for (int c = 0; c < " + N + "; c++) {",
        "  T m0 = " + srcX + "[c*" + N + "+0], m1 = " + srcX + "[c*" + N + "+1], m2 = " + srcX + "[c*" + N + "+2];",
        "  " + dstX + "[c*" + N + "+0] += Rd[0]*m0 + Rd[1]*m1 + Rd[2]*m2;",
        "  " + dstX + "[c*" + N + "+1] += Rd[3]*m0 + Rd[4]*m1 + Rd[5]*m2;",
        "  " + dstX + "[c*" + N + "+2] += Rd[6]*m0 + Rd[7]*m1 + Rd[8]*m2;",
        "}",
    ]


def _emit_pref_block(n, dstX, vec, Rname):
    """dstX[0:3, 3+a] += Rname @ (e_a x vec[0:3])."""
    N = str(n)
    return [
        "for (int a = 0; a < 3; a++) {",
        "  T etx = (a==1)*( " + vec + "[2]) + (a==2)*(-" + vec + "[1]);",
        "  T ety = (a==0)*(-" + vec + "[2]) + (a==2)*( " + vec + "[0]);",
        "  T etz = (a==0)*( " + vec + "[1]) + (a==1)*(-" + vec + "[0]);",
        "  T w0 = " + Rname + "[0]*etx + " + Rname + "[1]*ety + " + Rname + "[2]*etz;",
        "  T w1 = " + Rname + "[3]*etx + " + Rname + "[4]*ety + " + Rname + "[5]*etz;",
        "  T w2 = " + Rname + "[6]*etx + " + Rname + "[7]*ety + " + Rname + "[8]*etz;",
        "  " + dstX + "[(3+a)*" + N + "+0] += w0;",
        "  " + dstX + "[(3+a)*" + N + "+1] += w1;",
        "  " + dstX + "[(3+a)*" + N + "+2] += w2;",
        "}",
    ]


def _emit_idsva_so_mjx_dM_closed_form(self, n):
    """dM_dq (block3) closed form (congruence + base-rot frame), scalar transcription.
    Reads T_dM, writes O_dM. Uses s_M for the frame terms."""
    N = str(n)
    nv3 = n * n * n
    _fb = 12 * self.robot.get_num_joints()
    self.gen_add_code_line("// --- dM_dq (block3) closed form: Step1 reframe q-tangent, Step2 congruence, Step3 frame ---")
    # Step1+2 fused per k — PARALLEL over k (each writes only O_dM[...,k], reads
    # only T_dM/R, uses private work1/work2). Locals + work1/work2 are recomputed/
    # declared register-/stack-local at the top of the body (no shared scratch).
    self.gen_add_parallel_loop("k", N)
    self.gen_add_code_lines(_emit_idsva_so_mjx_locals_lines(n, nv3, _fb))
    self.gen_add_code_line("T work1[" + str(n * n) + "], work2[" + str(n * n) + "];")
    # Build tmp[i*nv+l] (row i, col l) for this k into work1 (col-major: work1[l*nv+i]).
    self.gen_add_code_lines([
        "for (int i = 0; i < " + N + "; i++) for (int l = 0; l < " + N + "; l++) {",
        "  T val;",
        "  if (k < 3) {",
        "    val = static_cast<T>(0);",
        "    for (int m = 0; m < 3; m++) val += T_dM[(i*" + N + " + l)*" + N + " + m]*R[3*k+m];",
        "  } else { val = T_dM[(i*" + N + " + l)*" + N + " + k]; }",
        "  work1[l*" + N + " + i] = val;",   # col-major store (i row, l col)
        "}",
    ])
    # Step2: rows i<3 <- R@rows (over cols l) into work2; then cols l<3 <- cols@R^T.
    self.gen_add_code_lines(_emit_rot_rows_block(n, "work1", "work2", "R"))
    # cols l<3 <- cols . R^T : out[i,l] = sum_lp work2[i,lp] R[l,lp] for l<3
    self.gen_add_code_lines([
        "for (int i = 0; i < " + N + "; i++) {",
        "  T c0 = work2[0*" + N + "+i], c1 = work2[1*" + N + "+i], c2 = work2[2*" + N + "+i];",
        "  work1[0*" + N + "+i] = c0*R[0] + c1*R[1] + c2*R[2];",   # R[l=0,lp]=R[0..2]
        "  work1[1*" + N + "+i] = c0*R[3] + c1*R[4] + c2*R[5];",
        "  work1[2*" + N + "+i] = c0*R[6] + c1*R[7] + c2*R[8];",
        "  for (int l = 3; l < " + N + "; l++) work1[l*" + N + "+i] = work2[l*" + N + "+i];",
        "}",
        # write O_dM[i,l,k] = work1[l*nv+i]
        "for (int i = 0; i < " + N + "; i++) for (int l = 0; l < " + N + "; l++)",
        "  O_dM[(i*" + N + " + l)*" + N + " + k] = work1[l*" + N + " + i];",
    ])
    self.gen_add_end_control_flow()  # parallel k
    # Step3 reads O_dM[...,3+c] WRITTEN by Step1+2 at k=3+c, so it must follow a
    # block sync (the parallel k-loop above may run those k on a DIFFERENT thread).
    self.gen_add_sync()
    # Step3: base-rot frame for c in 0..2, kk=3+c: O_dM[:,:,kk] += Gd@M@G^T + G@M@Gd^T.
    # PARALLEL over c in [0,3): each writes only O_dM[...,3+c], uses private
    # work1/work2 + register-local R. (Only 3 active threads — but it's a tiny tail.)
    self.gen_add_code_line("// Step3: base-rot frame term added to columns kk=3+c")
    self.gen_add_parallel_loop("c", "3")
    self.gen_add_code_lines(_emit_idsva_so_mjx_locals_lines(n, nv3, _fb))
    self.gen_add_code_line("T work1[" + str(n * n) + "], work2[" + str(n * n) + "];")
    self.gen_add_code_lines([
        "int kk = 3 + c;",
        "T Rdc[9]; for (int ii=0; ii<9; ii++) Rdc[ii]=static_cast<T>(0);",
        "T skc[9]; for (int ii=0; ii<9; ii++) skc[ii]=static_cast<T>(0);",
        "if (c==0){ skc[1*3+2]=static_cast<T>(-1); skc[2*3+1]=static_cast<T>(1); }",
        "if (c==1){ skc[2*3+0]=static_cast<T>(-1); skc[0*3+2]=static_cast<T>(1); }",
        "if (c==2){ skc[0*3+1]=static_cast<T>(-1); skc[1*3+0]=static_cast<T>(1); }",
        "for (int r = 0; r < 3; r++) for (int cc = 0; cc < 3; cc++) {",
        "  T acc = static_cast<T>(0);",
        "  for (int p = 0; p < 3; p++) acc += R[3*r+p]*skc[3*p+cc];",
        "  Rdc[3*r+cc] = acc;",
        "}",
        # MGt = M @ G^T : cols0:3 <- M[:,0:3]@R^T (store into work1 col-major work1[b*nv+a])
        "for (int a = 0; a < " + N + "; a++) {",
        "  T m0 = s_M[a + " + N + "*0], m1 = s_M[a + " + N + "*1], m2 = s_M[a + " + N + "*2];",
        "  work1[0*" + N + "+a] = m0*R[0] + m1*R[1] + m2*R[2];",
        "  work1[1*" + N + "+a] = m0*R[3] + m1*R[4] + m2*R[5];",
        "  work1[2*" + N + "+a] = m0*R[6] + m1*R[7] + m2*R[8];",
        "  for (int b = 3; b < " + N + "; b++) work1[b*" + N + "+a] = s_M[a + " + N + "*b];",
        "}",
        # term = Gd@MGt : rows0:3 = Rdc@work1[0:3,:] ; store col-major into work2
        "for (int rr = 0; rr < " + N + "*" + N + "; rr++) work2[rr] = static_cast<T>(0);",
        "for (int b = 0; b < " + N + "; b++) {",
        "  T r0 = work1[b*" + N + "+0], r1 = work1[b*" + N + "+1], r2 = work1[b*" + N + "+2];",
        "  work2[b*" + N + "+0] += Rdc[0]*r0 + Rdc[1]*r1 + Rdc[2]*r2;",
        "  work2[b*" + N + "+1] += Rdc[3]*r0 + Rdc[4]*r1 + Rdc[5]*r2;",
        "  work2[b*" + N + "+2] += Rdc[6]*r0 + Rdc[7]*r1 + Rdc[8]*r2;",
        "}",
        # GM = G@M : rows0:3 <- R@M[0:3,:] (col-major work1[b*nv+a])
        "for (int b = 0; b < " + N + "; b++) {",
        "  T r0 = s_M[0 + " + N + "*b], r1 = s_M[1 + " + N + "*b], r2 = s_M[2 + " + N + "*b];",
        "  work1[b*" + N + "+0] = R[0]*r0 + R[1]*r1 + R[2]*r2;",
        "  work1[b*" + N + "+1] = R[3]*r0 + R[4]*r1 + R[5]*r2;",
        "  work1[b*" + N + "+2] = R[6]*r0 + R[7]*r1 + R[8]*r2;",
        "  for (int a = 3; a < " + N + "; a++) work1[b*" + N + "+a] = s_M[a + " + N + "*b];",
        "}",
        # term += G@M@Gd^T : cols0:3 <- (G@M)[:,0:3]@Rdc^T : out[a,l<3]=sum_lp work1[a,lp]Rdc[l,lp]
        "for (int a = 0; a < " + N + "; a++) {",
        "  T c0 = work1[0*" + N + "+a], c1 = work1[1*" + N + "+a], c2 = work1[2*" + N + "+a];",
        "  work2[0*" + N + "+a] += c0*Rdc[0] + c1*Rdc[1] + c2*Rdc[2];",
        "  work2[1*" + N + "+a] += c0*Rdc[3] + c1*Rdc[4] + c2*Rdc[5];",
        "  work2[2*" + N + "+a] += c0*Rdc[6] + c1*Rdc[7] + c2*Rdc[8];",
        "}",
        # O_dM[a,l,kk] += work2[l*nv+a]
        "for (int a = 0; a < " + N + "; a++) for (int l = 0; l < " + N + "; l++)",
        "  O_dM[(a*" + N + " + l)*" + N + " + kk] += work2[l*" + N + " + a];",
    ])
    self.gen_add_end_control_flow()  # parallel c


def gen_idsva_so_world_frame_inner_function_call(self, scratch_in_smem_expr = "true",
                                                 cold_in_smem_expr = "true",
                                                 mujoco_output_expr = None):
    """Emit the call to `idsva_so_world_frame_inner` mirroring the existing call helper.
    scratch_in_smem_expr selects the inner's scratch placement (s_temp vs d_workspace);
    cold_in_smem_expr selects the surgical cold-trio placement (Xdown/v_w/a_w in smem vs
    d_workspace). Callers spilling the whole inner pass scratch="false" + a valid
    d_temp_spill region; callers doing the surgical spill pass cold="false" + d_temp_spill.
    The inner now OWNS the load_update_XImats call, so d_robotModel is threaded through.

    `mujoco_output_expr` (floating non-mimic/skew only): when not None, appends the
    trailing MUJOCO_OUTPUT template arg + the d_mjx_scratch pointer (the SO-temp
    region of d_workspace); None keeps the legacy template/signature byte-identical
    for the pin path."""
    tmpl = "idsva_so_world_frame_inner<T, " + scratch_in_smem_expr + ", " + cold_in_smem_expr
    if mujoco_output_expr is not None:
        tmpl += ", " + mujoco_output_expr
    tmpl += ">"
    id_so_code_start = tmpl + "(s_idsva_so, s_q, s_qd, s_qdd, "
    id_so_code_middle = self.gen_insert_helpers_function_call()
    # Unified signature: world inner takes (s_temp, d_workspace, d_robotModel, gravity).
    # `d_temp_spill` is the kernel-local typed view into d_workspace (nullptr at the full
    # rung). d_robotModel is forwarded so the inner can own the XImats load.
    if mujoco_output_expr is not None:
        id_so_code_end = "s_temp, d_temp_spill, d_robotModel, gravity, d_mjx_scratch);"
    else:
        id_so_code_end = "s_temp, d_temp_spill, d_robotModel, gravity);"
    self.gen_add_code_line(id_so_code_start + id_so_code_middle + id_so_code_end)


def gen_idsva_so_world_cold_floats(self):
    """Float count of the world-frame idsva_so surgical (output_cold) cold band.

    The cold QUAD {Xup, Xdown, v_w, a_w} = (36+36+6+6)*NB = 84*NB, all provably dead
    before the Step-5 hot triple-walk (Xup's last read is the Step-4 IC build; Xdown
    dead after Step 3; v_w/a_w dead after Step 4's f_w build). SINGLE SOURCE OF TRUTH
    for: the inner cold-band layout, the kernel surgical-rung smem arena sizing
    (cold_floats below), the world ws-floats reservation, and fdsva_so's composed
    idsva_cold rung (all in GRiDCodeGenerator.py)."""
    return 84 * self.robot.get_num_bodies()


def _emit_idsva_so_world_frame_kernel_body_for_flags(self, n, NUM_POS, single_call_timing,
                                                     use_global_output, s_temp_in_global, cold_in_global = False):
    """Emit the idsva_so world-frame kernel body for one tier's spill flags.

    Flags:
      - use_global_output: 4*NV^3 output -> d_idsva_so global.
      - cold_in_global:    surgical — only the cold quad (Xup 36*NB + Xdown 36*NB + v_w/a_w
                           6*NB each = 84*NB) routes to d_workspace (inner COLD_IN_SMEM=false); hot stays smem.
      - s_temp_in_global:  whole world inner s_temp arena -> d_workspace (inner
                           SCRATCH_IN_SMEM=false; guaranteed-fit fallback).
    The inner now OWNS its scratch placement AND the XImats load (inner-owns-placement,
    mirrors fdsva_so_device): the kernel no longer repoints s_temp nor calls
    load_update_XImats — it just forwards the flags + the d_temp_spill region.
    """
    # MUJOCO_OUTPUT (floating non-mimic/skew): the kernel template flag in scope.
    # When set: (1) input-convert q/qd/qdd to the pin frame before the inner builds
    # XImats; (2) carve d_mjx_scratch from the SO-temp region of d_workspace (dead
    # post-assembly) and forward it + the flag to the inner. Default false ->
    # if-constexpr-elided to byte-identical PTX.
    mjx_kernel = self.robot.floating_base and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis())
    extra_t_buffers = [("s_q_qd_u", 3*NUM_POS)]
    if not use_global_output:
        extra_t_buffers.append(("s_idsva_so", 4*n**3))
    inner_temp = gen_idsva_so_world_frame_temp_mem_size(self) if self.robot.floating_base else self.gen_idsva_so_body_frame_inner_temp_mem_size()
    # Surgical rung: hot arena = inner_temp minus the cold QUAD (Xup 36*NB + Xdown 36*NB +
    # v_w/a_w 12*NB = 84*NB). Single source of truth (must stay in lockstep with the inner
    # cold-band layout ~3251-3263 and the shared-file _idsva_wf_cold).
    cold_floats = self.gen_idsva_so_world_cold_floats()
    if s_temp_in_global:
        smem_temp = 0
    elif cold_in_global:
        smem_temp = inner_temp - cold_floats
    else:
        smem_temp = inner_temp
    self.gen_XImats_helpers_temp_shared_memory_code(smem_temp, extra_t_buffers=extra_t_buffers)
    # `d_temp_spill` is the inner's d_workspace param: the whole-arena base (s_temp_in_global)
    # or the cold-trio base / d_cold (cold_in_global). nullptr at the full rung.
    self.gen_add_code_line("T *d_temp_spill = nullptr; (void)d_temp_spill;")
    needs_workspace = s_temp_in_global or cold_in_global
    # mjx ALSO needs d_workspace (for d_mjx_scratch), so don't void it then.
    if not needs_workspace and not mjx_kernel:
        self.gen_add_code_line("(void)d_workspace;")
    if mjx_kernel:
        self.gen_add_code_line("T *d_mjx_scratch = nullptr; (void)d_mjx_scratch;")
    self.gen_add_code_line(f"T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[{NUM_POS}]; T *s_qdd = &s_q_qd_u[{2*NUM_POS}];")
    scratch_in_smem_expr = "false" if s_temp_in_global else "true"
    cold_in_smem_expr = "false" if cold_in_global else "true"
    mjx_expr = "MUJOCO_OUTPUT" if mjx_kernel else None
    so_off = "GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()"
    ts_off = ("k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + " + so_off) if not single_call_timing else so_off
    def _emit_mjx_input_and_scratch(ts_expr):
        # mjx scratch = SO-temp region of d_workspace (dead post-assembly). The
        # input-convert mutates s_q/s_qd/s_qdd in place BEFORE the inner builds XImats.
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_add_code_line(f"d_mjx_scratch = reinterpret_cast<T *>(&d_workspace[{ts_expr}]);")
            self.gen_mjx_input_convert(q_name="s_q", qd_name="s_qd", qdd_name="s_qdd")
            self.gen_add_end_control_flow()
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q_qd_u",str(3*NUM_POS),stride="stride_q_qd_u")
        if needs_workspace:
            self.gen_add_code_line(f"d_temp_spill = reinterpret_cast<T *>(&d_workspace[{ts_off}]);")
        _emit_mjx_input_and_scratch(ts_off)
        if use_global_output:
            self.gen_add_code_line(f"T *s_idsva_so = &d_idsva_so[k*{4*n**3}];")
        self.gen_idsva_so_world_frame_inner_function_call(scratch_in_smem_expr, cold_in_smem_expr, mjx_expr)
        if not use_global_output:
            self.gen_kernel_save_result("idsva_so",str(4*n**3),stride=str(4*n**3))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q_qd_u",str(3*NUM_POS))
        if needs_workspace:
            self.gen_add_code_line(f"d_temp_spill = reinterpret_cast<T *>(&d_workspace[{ts_off}]);")
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) { d_mjx_scratch = reinterpret_cast<T *>(&d_workspace[" + so_off + "]); }")
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q_qd_u", str(3*NUM_POS))
        # timing path: input-convert each rep (anti-LICM reloads raw inputs) before the inner
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert(q_name="s_q", qd_name="s_qd", qdd_name="s_qdd")
            self.gen_add_end_control_flow()
        if use_global_output:
            self.gen_add_code_line("T *s_idsva_so = d_idsva_so;")
        self.gen_idsva_so_world_frame_inner_function_call(scratch_in_smem_expr, cold_in_smem_expr, mjx_expr)
        self.gen_add_end_control_flow()
        if not use_global_output:
            self.gen_kernel_save_result("idsva_so",str(4*n**3))


def gen_idsva_so_world_frame_kernel(self, single_call_timing = False):
    NUM_POS = self.robot.get_num_pos()
    n = self.robot.get_num_vel()
    func_params = [
        "d_idsva_so is a pointer to memory for the final result of size 4*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS = " + str(4*n**3),
        "d_workspace is a per-timestep global-memory scratch buffer (unused at TIER_SHARED; cold buffers spill here at LITE/MINIMAL)",
        "d_q_dq_u is the vector of joint positions, velocities, and accelerations",
        "stride_q_qd_u is the stride between each q, qd, u",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
        "gravity is the gravity constant",
        "num_timesteps is the length of the trajectory points",
    ]
    func_notes = ["world-frame IDSVA-SO kernel: clean single-pass reference path."]
    func_def_start = "void idsva_so_world_frame_kernel(T *d_idsva_so, unsigned char *d_workspace, const T *d_q_qd_u, const int stride_q_qd_u, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Computes IDSVA-SO via the world-frame single-pass formulation", func_notes, func_params, None)
    # MUJOCO_OUTPUT (floating non-mimic/skew): appended LAST after RESOURCE_TIER so
    # existing positional <T,TIER> call sites are unaffected; default false ->
    # byte-identical PTX. Fixed-base / mimic / skew never carry it.
    mjx_kernel = self.robot.floating_base and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis())
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)

    table = self._idsva_so_world_tier_table  # [(name, t_count, use_global_output, s_temp_in_global, cold_in_global), ...]
    picks = self.idsva_so_world_frame_spill_tier_3way
    def _emit_idsva_so_world_body(pick):
        _, _, ugo, stg, cig = table[pick]
        _emit_idsva_so_world_frame_kernel_body_for_flags(self, n, NUM_POS, single_call_timing, ugo, stg, cig)
    self.gen_tier_dispatch(picks, _emit_idsva_so_world_body)
    self.gen_add_end_function()


def gen_idsva_so_world_frame_host(self, mode = 0):
    single_call_timing = True if mode == 1 else False
    compute_only = True if mode == 2 else False
    func_params = [
        "hd_data is the packaged input and output pointers",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant",
        "num_timesteps is the length of the trajectory points",
        "streams are pointers to CUDA streams",
    ]
    func_def_start = "void idsva_so_world_frame(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end =   "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    self.gen_add_func_doc("Compute IDSVA-SO via the world-frame single-pass formulation", [], func_params, None)
    # MUJOCO_OUTPUT (floating non-mimic/skew): host template flag appended LAST;
    # forwarded positionally to the kernel (which names the tier to reach it).
    mjx_host = self.robot.floating_base and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis())
    if mjx_host:
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    else:
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"idsva_so_world_frame requires all-data or dynamics gridData\");")
    kernel_tmpl = "idsva_so_world_frame_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "idsva_so_world_frame_kernel<T, RESOURCE_TIER>"
    func_call_start = kernel_tmpl + "<<<block_dimms,thread_dimms,IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_idsva_so," + \
        "hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd,"
    func_call_end = "d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("kernel<T", "kernel_single_timing<T")
    self.gen_add_code_line("int stride_q_qd = Q_QD_U_STRIDE;")
    if not compute_only:
        self.gen_add_code_lines([
            "// start code with memory transfer",
            "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*" + ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));",
            "gpuErrchkKernel();",
        ])
    self.gen_add_code_line("// then call the kernel")
    func_call_code = [f'{func_call_start}{func_call_end}']
    # See gen_idsva_so_body_frame_host for the same fix: sync between launch and
    # clock_gettime(end) is required for real single-call timing, and
    # compute_only needs a sync so callers' batch timers see actual
    # kernel-completion time (not just async-launch overhead).
    if single_call_timing:
        func_call_code.insert(0, "struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("gpuErrchkKernel();")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"idsva_so_world_frame\", IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back",
            "gpuErrchk(cudaMemcpy(hd_data->h_idsva_so,hd_data->d_idsva_so,SECOND_ORDER_TENSOR_SIZE*" + ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();",
        ])
    else:
        self.gen_add_code_line("gpuErrchkKernel();")
    # Label kept distinct from the original IDSVA_SO so a future bench
    # that times both can keep them separate. The parser doesn't know
    # this label today; if/when the bench wires it up, add a
    # _GRID_SINGLE_LABELS entry.
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("idsva_so_world_frame"))
    self.gen_add_end_function()


def gen_idsva_so_world_frame(self):
    """Emit the complete world-frame IDSVA-SO path: inner, kernel, host wrappers.

    Co-exists with the existing `gen_idsva_so_body_frame` emission. Gated by the
    `enable_idsva_so_world_frame` flag in `gen_all_code`.
    """
    self.gen_idsva_so_world_frame_inner()
    self.gen_idsva_so_world_frame_kernel(single_call_timing=False)
    self.gen_idsva_so_world_frame_kernel(single_call_timing=True)
    self.gen_idsva_so_world_frame_host(0)
    self.gen_idsva_so_world_frame_host(1)
    self.gen_idsva_so_world_frame_host(2)


def gen_idsva_so_device(self, use_qdd_input = True):
    """Emit `idsva_so_device` — a __device__ entry that picks the perf-winning
    frame at codegen time: body_frame_inner for fixed-base, world_frame_inner
    for floating-base (mirrors the host-level idsva_so dispatcher; same body /
    world perf wins documented there).

    Inline-CUDA users call this from their own kernel. The d_workspace ptr
    is required at TIER_LITE/MINIMAL (size IDSVA_SO_DEVICE_INLINE_WORKSPACE_BYTES);
    at TIER_SHARED it is unused and can be nullptr (default).
    """
    NV = self.robot.get_num_vel()
    use_world = _idsva_so_use_world_frame(self)
    inner_temp_size = (self.gen_idsva_so_world_frame_temp_mem_size()
                       if use_world
                       else self.gen_idsva_so_body_frame_inner_temp_mem_size())
    frame_label = "world_frame" if use_world else "body_frame"
    func_params = ["s_idsva_so is a pointer to memory for the final result of size 4*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS*SECOND_ORDER_COORDS = " + str(4*NV**3), \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "s_qdd is the vector of joint accelerations", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant", \
                   "d_workspace is the global scratch buffer; size IDSVA_SO_DEVICE_INLINE_WORKSPACE_BYTES<T, RESOURCE_TIER>() bytes (= 0 at TIER_SHARED, " + str(inner_temp_size) + "*sizeof(T) at TIER_LITE+). Pass nullptr at TIER_SHARED"]
    func_def_start = "void idsva_so_device(T *s_idsva_so, const T *s_q, const T *s_qd, const T *s_qdd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, T *d_workspace = nullptr) {"
    if use_world:
        _frame_reason = "world for floating-base" if self.robot.floating_base else "world for spherical"
    else:
        _frame_reason = "body for fixed-base"
    func_notes = ["Dispatches to " + frame_label + "_inner at codegen time (" + _frame_reason + ").",
                  "Inline-CUDA users: at TIER_LITE/TIER_MINIMAL the inner scratch moves from s_temp to d_workspace, freeing shared memory for the caller's outer kernel"]
    func_def = func_def_start + func_def_end
    # shared device-wrapper skeleton (B+C §1.1); s_temp routes to d_workspace at
    # LITE+ via tier_workspace_expr. idsva_so does NOT reserve linalg scratch.
    def _emit_idsva_so_inner():
        # Inline entry spills the WHOLE s_temp arena via tier_workspace_expr, so the
        # inner's per-buffer spill pointer is unused here (pass nullptr).
        self.gen_add_code_line("T *d_temp_spill = nullptr; (void)d_temp_spill;")
        if use_world:
            self.gen_idsva_so_world_frame_inner_function_call()
        else:
            self.gen_idsva_so_body_frame_inner_function_call()
            self.gen_idsva_so_body_frame_public_dvdq_layout_repair()
    self.gen_device_wrapper(
        "Computes the second order derivatives of inverse dynamics (frame picked at codegen time)",
        func_def, inner_temp_size, _emit_idsva_so_inner,
        template_line = "template <typename T, int RESOURCE_TIER = TIER_SHARED>",
        func_notes = func_notes, func_params = func_params,
        include_linalg_scratch = False, tier_workspace_expr = "d_workspace")


def gen_idsva_so_dispatcher_host(self, mode = 0):
    """Emit `grid::idsva_so` — a host wrapper that calls the perf-winning
    variant for this robot's base type. Picked at codegen time: body_frame
    for fixed-base, world_frame for floating-base. Both inners produce
    numerically equivalent output; this is purely a perf optimization.
    Body/world host wrappers remain individually callable for direct
    comparison.

    Measured on sm_120 / RTX 5090 (2026-05-18 sweep):
      - iiwa14 (NV=7,  fixed):    body 7-9x   faster than world
      - go2    (NV=12, fixed):    body 7-10x  faster than world
      - g1     (NV=29, fixed):    world 6-15% faster than body
      - iiwa14 (NV=6,  floating): world 7x   faster than body
      - go2    (NV=18, floating): world 7x   faster than body
      - g1     (NV=35, floating): world 20x  faster than body

    The "body for fixed, world for floating" rule is the safe choice — it
    preserves the large wins at the common low-DOF fixed-base case
    (iiwa14, go2) and the large wins on every floating-base case. The
    g1_fixed regression is small (~15%) and isolated to a single
    high-DOF data point; refining the dispatcher with a NV threshold is a
    worthwhile follow-up once more high-DOF fixed-base robots exist in
    the manifest.

    Regular and compute_only modes forward to the underlying host wrapper.
    Single-timing mode inlines its own clock_gettime + kernel launch so the
    printf label says "IDSVA_SO" (not "IDSVA_SO_BODY_FRAME"/"IDSVA_SO_WORLD_FRAME"),
    which is what the bench's timing parser keys on for this row.
    """
    single_call_timing = (mode == 1)
    compute_only = (mode == 2)

    frame_suffix = "world_frame" if _idsva_so_use_world_frame(self) else "body_frame"
    smem_macro = f"IDSVA_SO_{frame_suffix.upper()}_DYNAMIC_SHARED_MEM_BYTES<T>()"

    func_def_start = "void idsva_so(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end =   "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")

    self.gen_add_func_doc(
        f"Dispatching IDSVA-SO wrapper (forwards to {frame_suffix} at codegen time)",
        [f"Body wins ~30x on fixed-base; world wins 2-4x on floating-base. Same numbers either way."],
        ["hd_data is the packaged input and output pointers",
         "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
         "gravity is the gravity constant",
         "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)",
         "streams are pointers to CUDA streams for async memory transfers (if needed)"],
        None,
    )
    # MUJOCO_OUTPUT (floating non-mimic/skew only — the dispatcher forwards to
    # world_frame): appended LAST after KIND. The binding calls
    # grid::idsva_so<T, KIND, /*MUJOCO_OUTPUT=*/true>. Fixed-base (body_frame, no
    # mjx) keeps the legacy 2-arg template -> byte-identical.
    mjx_disp = self.robot.floating_base and not (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis())
    if mjx_disp:
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL, bool MUJOCO_OUTPUT = false>")
    else:
        self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)

    if single_call_timing:
        # Inline our own single-timing block so the printf label is "IDSVA_SO"
        # (rather than the underlying body_frame/world_frame label that the
        # delegate's _single_timing wrapper would print). Mirrors the structure
        # of `gen_idsva_so_body_frame_host(mode=1)`.
        # Both body_frame_kernel and world_frame_kernel now take d_workspace
        # (unified signature; cold buffers spill there at LITE/MINIMAL).
        kernel_name = f"idsva_so_{frame_suffix}_kernel_single_timing"
        # world_frame kernel carries the trailing MUJOCO_OUTPUT (named tier to reach it).
        kernel_tmpl = (f"{kernel_name}<T, GRID_DEFAULT_RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_disp
                       else f"{kernel_name}<T>")
        kernel_workspace_arg = "hd_data->d_workspace,"
        self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"idsva_so requires all-data or dynamics gridData\");")
        self.gen_add_code_line("int stride_q_qd = Q_QD_U_STRIDE;")
        self.gen_add_code_lines([
            "// start code with memory transfer",
            "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*sizeof(T),cudaMemcpyHostToDevice,streams[0]));",
            "gpuErrchkKernel();",
        ])
        self.gen_add_code_line("// then call the kernel")
        self.gen_add_code_line(f"gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"idsva_so\", {smem_macro}));")
        self.gen_add_code_line("struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        self.gen_add_code_line(
            f"{kernel_tmpl}<<<block_dimms,thread_dimms,{smem_macro}>>>(hd_data->d_idsva_so,{kernel_workspace_arg}hd_data->d_q_qd_u,stride_q_qd,d_robotModel,gravity,num_timesteps);"
        )
        self.gen_add_code_line("gpuErrchkKernel();")
        self.gen_add_code_line("clock_gettime(CLOCK_MONOTONIC,&end);")
        self.gen_add_code_lines([
            "// finally transfer the result back",
            "gpuErrchk(cudaMemcpy(hd_data->h_idsva_so,hd_data->d_idsva_so,SECOND_ORDER_TENSOR_SIZE*sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();",
        ])
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("idsva_so"))
    else:
        # Regular and compute_only modes forward to the underlying host wrapper.
        # The underlying wrapper handles all memcpy + launch + sync correctly,
        # and neither mode emits a single-call printf label — so the label
        # collision doesn't apply here.
        target = f"idsva_so_{frame_suffix}"
        if compute_only:
            target += "_compute_only"
        forward_args = "hd_data, d_robotModel, gravity, num_timesteps, block_dimms, thread_dimms"
        if not compute_only:
            forward_args += ", streams"
        # forward MUJOCO_OUTPUT to the world_frame host (floating); fixed-base unchanged.
        target_tmpl = "<T, KIND, MUJOCO_OUTPUT>" if mjx_disp else "<T, KIND>"
        self.gen_add_code_line(f"{target}{target_tmpl}({forward_args});")
    self.gen_add_end_function()


def gen_idsva_so_dispatcher(self):
    """Emit the device-level dispatcher + all three host dispatcher modes."""
    self.gen_idsva_so_device()
    self.gen_idsva_so_dispatcher_host(0)
    self.gen_idsva_so_dispatcher_host(1)
    self.gen_idsva_so_dispatcher_host(2)

"""Regressor STATE gradient dY/dx CUDA emit (split out of _regressor.py 2026-10-03; the section
banner below is the design note). Shares `_emit_dI_times_v` with `_regressor`.
"""

from ._regressor import _emit_dI_times_v
from grid_codegen.helpers._code_generation_helpers import gen_workspace_repoint_line, gen_workspace_cast_expr, host_mode_flags, mangle_host_func_defs, wrap_host_single_call_timing

# ===========================================================================
# Regressor STATE gradient  dY/dx  (differentiability plan B.0 "du x pi" cell)
# ---------------------------------------------------------------------------
# d(inverse_dynamics_gradient)/dpi == d(Y)/dx: tau = Y(q,qd,qdd) . pi with
# constant pi, so dY_dx[c] . pi == dtau_dx[:, c] (the B.0 identity). Mirrors
# the verified numpy reference `inverse_dynamics_regressor_gradient`:
#
#   1. staging: the analytic RNEA-gradient forward pass supplies dv_i/dx_c,
#      da_i/dx_c per (body, direction). Rather than re-emitting that pass,
#      the _device calls the PROVEN inverse_dynamics_gradient_inner into a
#      throwaway s_dc_du scratch and reads the dv/da bands it leaves in
#      s_temp (sparse-compressed for the cardinal path, dense cells for the
#      mimic/skew/spherical fold). The df backward work is wasted — accepted
#      v1 cost; a later byte-gated refactor can split the fpass out.
#   2. per-thread walk (the Y inner's idiom, one thread per work item
#      (partial, direction c, link i, param k)): build the body-regressor
#      column derivative by the THREE-term product rule
#         df0 = dI_k da + crf(dv) (dI_k v) + crf(v) (dI_k dv)
#      plus the base column f0 = dI_k a + crf(v)(dI_k v), then walk the
#      ancestor chain projecting df onto each DOF row and propagating
#      f/df with X^T; on the dq side the propagation's own X(q_j)
#      derivative adds alpha_j * X^T (crf(S_col) f) into df exactly when
#      the thread's direction is that joint's DOF (the RNEA-gradient
#      bpass fxS idiom, applied to the per-thread column).
#
# Work items only exist where the contribution is nonzero: direction c must
# be a DOF of link i's ancestor chain (same support as dv_i/dx_c). Every
# thread writes a DISJOINT set of output cells ((partial, c, col=10i+k) is
# unique per thread), so writes are collision-free and thread-count
# invariant; the (sparse) remainder of the output is zero-filled first.
#
# OUTPUT is DIRECT-TO-GLOBAL from day one (d2ee/MT idiom): 2*nv*nv*10*NB
# floats per timestep (1.9 MB fp32 on g1) never touches smem. Layout matches
# the numpy oracle's (nv, nv, 10NB) C-order per partial:
#   d_dY_dx[(dq? 0 : nv*nv*10NB) + c*(nv*10NB) + row*(10NB) + 10*i + k]
# ===========================================================================

def gen_inverse_dynamics_regressor_gradient_inner_temp_mem_size(self):
    # staging provider = the id_du inner (called into scratch); the walk itself
    # is register-only.
    return self.gen_inverse_dynamics_gradient_inner_temp_mem_size()


def _idrg_uses_dense_staging(self):
    """True when the id_du inner is the DENSE serial fold (mimic/skew/spherical),
    whose dv/da staging is dense per (body, direction) cells; False for the
    sparse-compressed cardinal band."""
    return (self.robot_has_mimic_joints() or self.robot.robot_has_skew_axis()
            or self.robot.robot_has_spherical())


def _idrg_walk_plan(self):
    """Emit-time enumeration of the walk. Returns (chains, dirs, vcols_fn):
    chains[i] = [i, parent(i), ..., root] (leaf->root, the walk order);
    dirs[i] = ROOT-FIRST ordered distinct reduced v-slots of the chain's
    joints — exactly the support of dv_i/dx_c, and (for the fixed-base sparse
    band) exactly the band's column order for body i."""
    robot = self.robot
    NB = robot.get_num_bodies()
    chains = [[i] + robot.get_ancestors_by_id(i) for i in range(NB)]

    def vcols(j):
        idx = robot.get_joint_index_v(j)
        return [int(v) for v in idx] if isinstance(idx, (list, tuple)) else [int(idx)]

    dirs = []
    for i in range(NB):
        seen, d = set(), []
        for j in reversed(chains[i]):  # root-first
            for c in vcols(j):
                if c not in seen:
                    seen.add(c)
                    d.append(c)
        dirs.append(d)
    return chains, dirs, vcols


def _emit_fxS_times_f_peq(self, dst_name, S_vec, src_name, scale):
    """Emit dst[r] += scale * (crf(S) . src)[r] for a CONSTANT 6-vector S,
    expanding the force-cross rows (fx_times_v) with S's entries baked in.
    crf(v).f rows (GRiD fx_times_v, [ang; lin] order):
      r0 = -v2 f1 + v1 f2 - v5 f4 + v4 f5
      r1 =  v2 f0 - v0 f2 + v5 f3 - v3 f5
      r2 = -v1 f0 + v0 f1 - v4 f3 + v3 f4
      r3 = -v2 f4 + v1 f5
      r4 =  v2 f3 - v0 f5
      r5 = -v1 f3 + v0 f4
    """
    rows = [
        [(-1, 2, 1), (1, 1, 2), (-1, 5, 4), (1, 4, 5)],
        [(1, 2, 0), (-1, 0, 2), (1, 5, 3), (-1, 3, 5)],
        [(-1, 1, 0), (1, 0, 1), (-1, 4, 3), (1, 3, 4)],
        [(-1, 2, 4), (1, 1, 5)],
        [(1, 2, 3), (-1, 0, 5)],
        [(-1, 1, 3), (1, 0, 4)],
    ]
    for r in range(6):
        terms = []
        for (sgn, vi, fi) in rows[r]:
            coeff = sgn * float(S_vec[vi]) * float(scale)
            if coeff == 0.0:
                continue
            if coeff == 1.0:
                terms.append(("+", src_name + "[" + str(fi) + "]"))
            elif coeff == -1.0:
                terms.append(("-", src_name + "[" + str(fi) + "]"))
            else:
                terms.append(("+", "static_cast<T>(" + repr(coeff) + ") * " + src_name + "[" + str(fi) + "]"))
        if not terms:
            continue
        expr = ""
        for (sgn, t) in terms:
            if not expr:
                expr = ("-" if sgn == "-" else "") + t
            else:
                expr += " " + sgn + " " + t
        self.gen_add_code_line(dst_name + "[" + str(r) + "] += " + expr + ";")


def gen_inverse_dynamics_regressor_gradient_inner(self):
    NJ = self.robot.get_num_joints()
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    W = 10 * NB
    ROWBLK = nv * W            # one direction's (nv x 10NB) block
    HALF_OUT = nv * ROWBLK     # the dq half
    OUT = 2 * HALF_OUT
    dense = _idrg_uses_dense_staging(self)
    chains, dirs, vcols = _idrg_walk_plan(self)
    HAS_MIMIC = self.robot_has_mimic_joints()

    # per-partial item counts and select tables
    items_per_body = [10 * len(dirs[i]) for i in range(NB)]
    HALF_ITEMS = sum(items_per_body)
    cum = [0]
    for i in range(NB):
        cum.append(cum[-1] + items_per_body[i])

    # sparse-band column bases (cardinal path only)
    if not dense:
        (dva_cols_per_partial, _, running_sum_dva_cols_per_jid, _, _, _, _) = \
            self.gen_topology_sparsity_helpers_python()
        layout = self.gen_inverse_dynamics_gradient_temp_layout()
        OFF_DV_DQ, OFF_DV_DQD = layout["offset_dv_dq"], layout["offset_dv_dqd"]
        OFF_DA_DQ, OFF_DA_DQD = layout["offset_da_dq"], layout["offset_da_dqd"]
    else:
        bw = 6 * nv * NB
        OFF_DV_DQ, OFF_DA_DQ = 0, bw
        OFF_DV_DQD, OFF_DA_DQD = 3 * bw, 4 * bw

    func_params = [
        "d_dY_dx is the GLOBAL output slab, size 2*NUM_VEL*NUM_VEL*10*NUM_BODIES = " + str(OUT) +
        " (dq block then dqd block; per direction c a row-major nv x 10*NB matrix)",
        "s_vaf is the RNEA v|a|f band (v,a read; body-indexed stride " + str(NJ) + ")",
        "s_temp is the id_du inner scratch AFTER inverse_dynamics_gradient_inner ran (the dv/da staging bands are read)",
        "d_temp_spill is the id_du da_df spill region (used when USE_DA_DF_SPILL)",
    ]
    func_notes = [
        "Assumes s_XImats is updated for the current s_q AND inverse_dynamics_gradient_inner has populated s_temp's dv/da bands",
        "dY_dx[c] . pi == dtau_dx[:, c] (the B.0 identity); every output cell has exactly one writer (thread-count invariant)",
    ]
    func_def_start = "void inverse_dynamics_regressor_gradient_inner(T *d_dY_dx, const T *s_vaf, "
    func_def_end = "T *s_temp, T *d_temp_spill) {"
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params("", func_params, -1)
    func_def = func_def_start + func_def_middle + func_def_end
    self.gen_add_func_doc("Compute the joint-torque-regressor state derivative dY/dx (walk stage)",
                          func_notes, func_params, None)
    self.gen_add_code_line("template <typename T, bool USE_DA_DF_SPILL = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    # 1) zero the (sparse-support) global output slab
    self.gen_add_code_line("// zero the output slab (the walk only writes the nonzero support)")
    self.gen_add_parallel_loop("ind", str(OUT))
    self.gen_add_code_line("d_dY_dx[ind] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # 2) the walk
    self.gen_add_code_line("// per-thread chain walk: one item per (partial, direction, link, param)")
    self.gen_add_parallel_loop("it", str(2 * HALF_ITEMS))
    self.gen_add_code_line("bool dq_flag = it < " + str(HALF_ITEMS) + ";")
    self.gen_add_code_line("int itp = dq_flag ? it : it - " + str(HALF_ITEMS) + ";")
    select_var_vals = [("int", "link_i", [str(i) for i in range(NB)]),
                      ("int", "item_base", [str(cum[i]) for i in range(NB)])]
    if not dense:
        select_var_vals.append(("int", "stage_base", [str(running_sum_dva_cols_per_jid[i]) for i in range(NB)]))
    self.gen_add_multi_threaded_select("(itp)", "<", [str(cum[i + 1]) for i in range(NB)], select_var_vals)
    self.gen_add_code_line("int loc = itp - item_base; int r = loc / 10; int param_k = loc % 10;")

    # 2a) direction slot c_dir (needed for the output index; for the dense and
    #     floating staging layouts it is also the staging column)
    self.gen_add_code_line("int c_dir;")
    self.gen_add_code_line("switch (link_i) {", True)
    for i in range(NB):
        expr = ""
        d = dirs[i]
        for ridx in range(len(d) - 1):
            expr += "r == " + str(ridx) + " ? " + str(d[ridx]) + " : "
        expr += str(d[len(d) - 1])
        self.gen_add_code_line("case " + str(i) + ": c_dir = " + expr + "; break;")
    self.gen_add_code_line("default: c_dir = 0;")
    self.gen_add_end_control_flow()

    # 2b) staging pointers for this thread's (link_i, c_dir) column
    if not dense:
        if self.robot.floating_base:
            # floating band: dense per-jid columns indexed by direction
            self.gen_add_code_line("int s6 = 6*(link_i*" + str(nv) + " + c_dir);")
        else:
            # fixed sparse band: root-first ancestor columns, rank == r
            self.gen_add_code_line("int s6 = 6*(stage_base + r);")
        self.gen_add_code_line("const T *dv = &s_temp[(dq_flag ? " + str(OFF_DV_DQ) + " : " + str(OFF_DV_DQD) + ") + s6];")
        self.gen_add_code_line("const T *da = grid_id_du_temp_ptr<T, USE_DA_DF_SPILL>(s_temp, d_temp_spill, (dq_flag ? " + str(OFF_DA_DQ) + " : " + str(OFF_DA_DQD) + ") + s6);")
    else:
        self.gen_add_code_line("int s6 = 6*(link_i*" + str(nv) + " + c_dir); (void)d_temp_spill;")
        self.gen_add_code_line("const T *dv = &s_temp[(dq_flag ? " + str(OFF_DV_DQ) + " : " + str(OFF_DV_DQD) + ") + s6];")
        self.gen_add_code_line("const T *da = &s_temp[(dq_flag ? " + str(OFF_DA_DQ) + " : " + str(OFF_DA_DQD) + ") + s6];")
    self.gen_add_code_line("const T *v_i = &s_vaf[6*link_i];")
    self.gen_add_code_line("const T *a_i = &s_vaf[" + str(6 * NJ) + " + 6*link_i];")

    # 2c) dI_k applications + the three-term product rule
    self.gen_add_code_line("T dIv[6]; T dIa[6]; T dIdv[6]; T dIda[6];")
    self.gen_add_code_line("switch (param_k) {", True)
    for k in range(10):
        self.gen_add_code_line("case " + str(k) + ": {", True)
        _emit_dI_times_v(self, "dIv", k, "v_i")
        _emit_dI_times_v(self, "dIa", k, "a_i")
        _emit_dI_times_v(self, "dIdv", k, "dv")
        _emit_dI_times_v(self, "dIda", k, "da")
        self.gen_add_code_line("break;")
        self.gen_add_end_control_flow()
    self.gen_add_code_line("default: { for (int q_=0;q_<6;q_++){dIv[q_]=static_cast<T>(0); dIa[q_]=static_cast<T>(0); dIdv[q_]=static_cast<T>(0); dIda[q_]=static_cast<T>(0);} }")
    self.gen_add_end_control_flow()
    self.gen_add_code_lines([
        "T f[6]; T df[6]; T t6[6];",
        "// f0 = dI_k a + crf(v)(dI_k v)   (the walk's dX-term source)",
        "fx_times_v<T>(t6, v_i, dIv);",
        "for (int q_=0;q_<6;q_++){ f[q_] = dIa[q_] + t6[q_]; }",
        "// df0 = dI_k da + crf(dv)(dI_k v) + crf(v)(dI_k dv)   (three-term product rule)",
        "fx_times_v<T>(t6, dv, dIv);",
        "for (int q_=0;q_<6;q_++){ df[q_] = dIda[q_] + t6[q_]; }",
        "fx_times_v<T>(t6, v_i, dIdv);",
        "for (int q_=0;q_<6;q_++){ df[q_] += t6[q_]; }",
    ])

    # 2d) output base for this thread's column
    self.gen_add_code_line("int out_base = (dq_flag ? 0 : " + str(HALF_OUT) + ") + c_dir*" + str(ROWBLK) + " + 10*link_i + param_k;")

    # 2e) the chain walk (leaf -> root), unrolled per link
    import numpy as _np

    def _mimic_scale(jid):
        return float(self._alpha_for_jid(jid)) if HAS_MIMIC else 1.0

    def _emit_projection(j):
        scale = _mimic_scale(j)
        if self.robot.floating_base and j == 0:
            inds_f = self.robot.get_joint_index_f(0)
            S0 = _np.array(self.robot.get_S_by_id(0))
            for kcol in range(S0.shape[1]):
                nz = _np.nonzero(S0[:, kcol])[0]
                for row in nz:
                    sgn = float(S0[row, kcol]) * scale
                    pref = "" if sgn == 1.0 else ("static_cast<T>(" + repr(sgn) + ") * ")
                    self.gen_add_code_line(
                        "d_dY_dx[out_base + " + str(int(inds_f[kcol])) + "*" + str(W) + "] += " +
                        pref + "df[" + str(int(row)) + "];")
        elif self.robot.joint_is_spherical(j):
            vblk_f = self.robot.get_joint_index_f(j)
            for m in range(3):
                self.gen_add_code_line(
                    "d_dY_dx[out_base + " + str(int(vblk_f[m])) + "*" + str(W) + "] += df[" + str(m) + "];")
        elif not self.robot.S_is_cardinal_by_id(j):
            fidx = self.robot.get_joint_index_f(j)
            if isinstance(fidx, (list, tuple)):
                fidx = fidx[0]
            S_vec = self.robot._get_flat_S_by_id(j)
            terms = [("static_cast<T>(" + repr(float(S_vec[q]) * scale) + ") * df[" + str(q) + "]")
                     for q in range(6) if S_vec[q] != 0.0]
            self.gen_add_code_line(
                "d_dY_dx[out_base + " + str(fidx) + "*" + str(W) + "] += " +
                (" + ".join(terms) if terms else "static_cast<T>(0)") + ";")
        else:
            s_ind = self.robot.get_S_index_by_id(j)
            s_sign = self.robot.get_S_sign_by_id(j)
            coeff = float(s_sign) * scale
            pref = "" if coeff == 1.0 else ("static_cast<T>(" + repr(coeff) + ") * ")
            fidx = self.robot.get_joint_index_f(j)
            if isinstance(fidx, (list, tuple)):
                fidx = fidx[0]
            self.gen_add_code_line(
                "d_dY_dx[out_base + " + str(fidx) + "*" + str(W) + "] += " + pref + "df[" + str(s_ind) + "];")

    def _emit_dX_terms(i, j):
        # dq only: df += alpha_j * crf(S_col) f, gated on this thread's
        # direction being that column's slot (rank compare against dirs[i]).
        scale = _mimic_scale(j)
        vcs = vcols(j)
        if self.robot.joint_is_spherical(j):
            S = _np.vstack([_np.eye(3), _np.zeros((3, 3))])
        elif not self.robot.S_is_cardinal_by_id(j):
            S = _np.array(self.robot._get_flat_S_by_id(j), dtype=float).reshape(6, 1)
        else:
            S = _np.zeros((6, 1))
            S[self.robot.get_S_index_by_id(j), 0] = float(self.robot.get_S_sign_by_id(j))
        for m, vc in enumerate(vcs):
            rank = dirs[i].index(vc)
            self.gen_add_code_line("if (dq_flag && r == " + str(rank) + ") {", True)
            _emit_fxS_times_f_peq(self, "df", [float(x) for x in S[:, m]], "f", scale)
            self.gen_add_end_control_flow()

    def _emit_XT_propagate(j):
        self.gen_add_code_lines([
            "{ T ft[6]; T dft[6];",
            "  for (int q_=0;q_<6;q_++){ T af=static_cast<T>(0); T adf=static_cast<T>(0);",
            "    for (int p_=0;p_<6;p_++){ T x = s_XImats[" + str(36 * j) + " + 6*q_ + p_]; af += x*f[p_]; adf += x*df[p_]; }",
            "    ft[q_]=af; dft[q_]=adf; }",
            "  for (int q_=0;q_<6;q_++){ f[q_]=ft[q_]; df[q_]=dft[q_]; } }",
        ])

    self.gen_add_code_line("switch (link_i) {", True)
    for i in range(NB):
        chain = chains[i]
        self.gen_add_code_line("case " + str(i) + ": {", True)
        for depth, j in enumerate(chain):
            self.gen_add_code_line("// chain joint " + str(j))
            _emit_projection(j)
            if depth < len(chain) - 1:
                if not (self.robot.floating_base and j == 0):
                    _emit_dX_terms(i, j)
                _emit_XT_propagate(j)
        self.gen_add_code_line("break;")
        self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()  # switch link_i
    self.gen_add_end_control_flow()  # parallel loop
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_inverse_dynamics_regressor_gradient_inner_function_call(self, updated_var_names=None):
    var_names = dict(
        d_dY_dx_name="d_dY_dx",
        s_vaf_name="s_vaf",
        s_temp_name="s_temp",
        d_temp_spill_name="nullptr",
        temp_spill_flag_name="false",
    )
    if updated_var_names is not None:
        for key, value in updated_var_names.items():
            var_names[key] = value
    code = ("inverse_dynamics_regressor_gradient_inner<T, " + var_names["temp_spill_flag_name"] + ">(" +
            var_names["d_dY_dx_name"] + ", " + var_names["s_vaf_name"] + ", " +
            self.gen_insert_helpers_function_call() +
            var_names["s_temp_name"] + ", " + var_names["d_temp_spill_name"] + ");")
    self.gen_add_code_line(code)


def gen_inverse_dynamics_regressor_gradient_device(self):
    """The whole dY/dx orchestration as ONE inner-owns-placement device fn
    (mirrors inverse_dynamics_gradient_device): [repoint s_temp] ->
    load_update_XImats -> inverse_dynamics_inner (vaf, qdd) ->
    inverse_dynamics_gradient_inner (into the s_dc_du SCRATCH; leaves the
    dv/da staging in s_temp) -> the dY/dx walk (direct-to-global output)."""
    n = self.robot.get_num_vel()
    func_params = [
        "d_dY_dx is the GLOBAL output slab (2*NUM_VEL*NUM_VEL*10*NUM_BODIES per timestep)",
        "s_dc_du is a 2*NUM_VEL*NUM_VEL SCRATCH the id_du staging call writes into (its value is unused)",
        "s_q is the vector of joint positions",
        "s_qd is the vector of joint velocities",
        "s_qdd is the vector of joint accelerations",
        "s_vaf is the id intermediate band (caller places); body-indexed",
        "s_temp is the shared scratch pool (used when SCRATCH_IN_SMEM)",
        "d_workspace is the global scratch pool (used when !SCRATCH_IN_SMEM)",
        "d_temp_spill is the id_du da_df band spill region (used when USE_DA_DF_SPILL)",
        "d_robotModel holds XImats/topology; gravity is the gravity constant",
    ]
    func_def_start = "void inverse_dynamics_regressor_gradient_device(T *d_dY_dx, T *s_dc_du, const T *s_q, const T *s_qd, const T *s_qdd, T *s_vaf, "
    func_def_end = ("T *s_temp, T *d_workspace, T *d_temp_spill, "
                    "const robotModel<T> *d_robotModel, const T gravity) {")
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -2)
    func_def = func_def_start + func_def_end
    self.gen_add_func_doc("dY/dx orchestration as a single inner-owns-placement device function",
                          ["Owns the s_temp pool placement (the repoint covers every consumer below)"],
                          func_params, None)
    self.gen_add_code_line("template <typename T, bool SCRATCH_IN_SMEM = true, bool USE_DA_DF_SPILL = false>")
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("if constexpr (!SCRATCH_IN_SMEM) { s_temp = d_workspace; } else { (void)d_workspace; }")
    self.gen_load_update_XImats_helpers_function_call()
    self.gen_inverse_dynamics_inner_function_call(
        compute_c=False, use_qdd_input=True,
        updated_var_names=dict(d_f_ext_name="nullptr"))
    self.gen_inverse_dynamics_gradient_inner_function_call(
        dict(d_temp_spill_name="d_temp_spill", temp_spill_flag_name="USE_DA_DF_SPILL"))
    self.gen_add_sync()
    self.gen_inverse_dynamics_regressor_gradient_inner_function_call(
        dict(d_temp_spill_name="d_temp_spill", temp_spill_flag_name="USE_DA_DF_SPILL"))
    self.gen_add_end_function()


def _emit_idrg_kernel_body_for_flags(self, NUM_POS, nv, use_selective_spill, use_global_temp, single_call_timing):
    NB = self.robot.get_num_bodies()
    out_size = 2 * nv * nv * 10 * NB
    in_size = 3 * NUM_POS
    # s_vaf vel-flavour (18*NJ mimic else 18*nv) — matches the ladder's c.grad_vaf
    _vaf_cnt = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else nv)
    extra_t_buffers = [("s_q_qd_qdd", in_size), ("s_dc_du", 2 * nv * nv), ("s_vaf", _vaf_cnt)]
    _selective_shared = (
        self.gen_inverse_dynamics_regressor_gradient_inner_temp_mem_size()
        if _idrg_uses_dense_staging(self)
        else self.gen_inverse_dynamics_gradient_temp_layout()["selective_shared_count"]
    )
    shared_mem_size = 0 if use_global_temp else (
        _selective_shared if use_selective_spill
        else self.gen_inverse_dynamics_regressor_gradient_inner_temp_mem_size()
    )
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size, extra_t_buffers=extra_t_buffers, include_linalg_scratch=True)
    self.gen_add_code_line("T *d_temp_spill = nullptr; (void)d_temp_spill;")
    self.gen_add_code_line("T *s_q = s_q_qd_qdd; T *s_qd = &s_q_qd_qdd[" + str(NUM_POS) +
                           "]; T *s_qdd = &s_q_qd_qdd[" + str(2 * NUM_POS) + "];")

    def _device_call(out_expr):
        self.gen_add_code_line(
            "inverse_dynamics_regressor_gradient_device<T, " +
            ("false" if use_global_temp else "true") + ", " +
            ("true" if use_selective_spill else "false") + ">(" +
            out_expr + ", s_dc_du, s_q, s_qd, s_qdd, s_vaf, " +
            self.gen_insert_helpers_function_call() +
            "s_temp, " +
            (gen_workspace_cast_expr(batch_indexed=not single_call_timing) if use_global_temp else "nullptr") + ", " +
            ("d_temp_spill" if use_selective_spill else "nullptr") +
            ", d_robotModel, gravity);")

    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q_qd_qdd", str(in_size), stride="stride_q_qd_qdd")
        if use_selective_spill:
            self.gen_add_code_line(gen_workspace_repoint_line("d_temp_spill", batch_indexed=True))
        self.gen_add_code_line("// compute (direct-to-global output slab for this timestep)")
        _device_call("&d_dY_dx[k*" + str(out_size) + "]")
        self.gen_add_sync()
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q_qd_qdd", str(in_size))
        if use_selective_spill:
            self.gen_add_code_line(gen_workspace_repoint_line("d_temp_spill"))
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q_qd_qdd", str(in_size), feedback_from="dY_dx")
        _device_call("d_dY_dx")
        self.gen_add_sync()
        self.gen_anti_licm_output_write("dY_dx", load_from_name="d_dY_dx")
        self.gen_add_end_control_flow()


def gen_inverse_dynamics_regressor_gradient_kernel(self, single_call_timing=False):
    NUM_POS = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    out_size = 2 * nv * nv * 10 * NB
    func_params = [
        "d_dY_dx is the output regressor state derivative, per timestep " + str(out_size) +
        " floats: dq block then dqd block, each direction c a row-major nv x 10*NUM_BODIES matrix",
        "d_workspace is the global scratch pool (spilled tiers)",
        "d_q_qd_qdd is the vector of joint positions, velocities, accelerations (q|qd|qdd)",
        "stride_q_qd_qdd is the stride between each (q, qd, qdd) triple",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant",
        "num_timesteps is the length of the trajectory points",
    ]
    func_def_start = "void inverse_dynamics_regressor_gradient_kernel(T *d_dY_dx, unsigned char *d_workspace, const T *d_q_qd_qdd, const int stride_q_qd_qdd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Compute the joint-torque-regressor state derivative dY/dx", [], func_params, None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    picks = getattr(self, "inverse_dynamics_regressor_gradient_spill_tier_3way", (0, 0, 0))
    def _emit_body(pick):
        uss, ugt = _INVERSE_DYNAMICS_REGRESSOR_GRADIENT_PICK_FLAGS[pick]
        _emit_idrg_kernel_body_for_flags(self, NUM_POS, nv, uss, ugt, single_call_timing)
    self.gen_tier_dispatch(picks, _emit_body)
    self.gen_add_end_function()


_INVERSE_DYNAMICS_REGRESSOR_GRADIENT_PICK_FLAGS = [
    # (use_selective_spill, use_global_temp) — same 3-rung menu as id_du
    (False, False),   # pick 0: full smem
    (True,  False),   # pick 1: selective spill (da_df band to workspace)
    (False, True),    # pick 2: global temp (entire s_temp to workspace)
]


def gen_inverse_dynamics_regressor_gradient_host(self, mode=0):
    single_call_timing, compute_only = host_mode_flags(mode)
    nv = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    out_size = 2 * nv * nv * 10 * NB
    func_params = [
        "hd_data is the packaged input and output pointers (q/qd/qdd inputs; output written to hd_data->d_dY_dx, " +
        str(out_size) + "*num_timesteps floats)",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant",
        "num_timesteps is the length of the trajectory points",
        "streams are pointers to CUDA streams for async memory transfers",
    ]
    func_def_start = "void inverse_dynamics_regressor_gradient(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end = "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    func_def_start, func_def_end = mangle_host_func_defs(func_def_start, func_def_end, single_call_timing, compute_only)
    self.gen_add_func_doc("Compute the joint-torque-regressor state derivative dY/dx (tau = Y . pi; dY_dx[c] . pi == dtau_dx[:, c])",
                          [], func_params, None)
    self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"inverse_dynamics_regressor_gradient requires all-data or dynamics gridData\");")
    func_call_start = "inverse_dynamics_regressor_gradient_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,INVERSE_DYNAMICS_REGRESSOR_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_dY_dx,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd_qdd,"
    func_call_end = "d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("inverse_dynamics_regressor_gradient_kernel<", "inverse_dynamics_regressor_gradient_kernel_single_timing<")
    self.gen_add_code_line("int stride_q_qd_qdd = Q_QD_U_STRIDE;")
    if not compute_only:
        self.gen_add_code_lines([
            "// start code with memory transfer",
            "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_qdd*" +
            ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));",
            "gpuErrchkKernel();",
        ])
    self.gen_add_code_line("// then call the kernel")
    func_call_code = [func_call_start + func_call_end]
    if single_call_timing:
        wrap_host_single_call_timing(func_call_code, kernel_errcheck=True)
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"inverse_dynamics_regressor_gradient\", INVERSE_DYNAMICS_REGRESSOR_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    if not single_call_timing:
        self.gen_add_workspace_slot_count()
    ws_bytes = ("GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()" if single_call_timing
                else "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(_grid_ws_n)")
    self.gen_add_code_line("if (GRID_INVERSE_DYNAMICS_REGRESSOR_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, " + ws_bytes + "));}")
    if single_call_timing:
        self.gen_add_code_lines(func_call_code)
    else:
        self.gen_add_workspace_clamped_launch(func_call_code, emit_count=False)
    self.gen_add_code_line("if (GRID_INVERSE_DYNAMICS_REGRESSOR_GRADIENT_USES_WORKSPACE_ANY_TIER) {gpuErrchk(grid_end_l2_persisting(0));}")
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back (hd_data->d_dY_dx -> hd_data->h_dY_dx)",
            "gpuErrchk(cudaMemcpy(hd_data->h_dY_dx,hd_data->d_dY_dx," +
            ("num_timesteps*" if not single_call_timing else "") + str(out_size) + "*sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();",
        ])
    else:
        self.gen_add_code_line("gpuErrchkKernel();")
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("inverse_dynamics_regressor_gradient"))
    self.gen_add_end_function()


def gen_inverse_dynamics_regressor_gradient(self):
    # walk inner -> orchestration device -> kernels -> hosts
    self.gen_inverse_dynamics_regressor_gradient_inner()
    self.gen_inverse_dynamics_regressor_gradient_device()
    self.gen_inverse_dynamics_regressor_gradient_kernel(single_call_timing=False)
    self.gen_inverse_dynamics_regressor_gradient_kernel(single_call_timing=True)
    self.gen_inverse_dynamics_regressor_gradient_host(0)
    self.gen_inverse_dynamics_regressor_gradient_host(1)
    self.gen_inverse_dynamics_regressor_gradient_host(2)

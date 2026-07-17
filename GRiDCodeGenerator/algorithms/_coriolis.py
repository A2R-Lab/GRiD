"""Coriolis matrix C(q, qd) CUDA emit (PS5 oracle 2).

`C(q, qd)` is the full nv x nv Coriolis matrix with `C qd + g = nonlinear
effects`. A direct, closed-form transcription of Pinocchio's
`computeCoriolisMatrix` spatial recursion (the verified numpy reference
`RBDReference._EnergyMixin.coriolis_matrix`) — NOT a finite difference, so it
matches the oracle to the tight value-tolerance bucket (fp32 ~1e-3 / fp64 ~1e-12).

Two-pass world-frame spatial recursion (mirrors the oracle exactly):

  Forward (body order, NB bodies):
    iX0[i]  = X_i . iX0[parent]          (body<-world accumulated transform)
    oXi[i]  = inv(iX0[i])                (Plucker-block spatial inverse, no solve)
    oY[i]   = iX0[i]^T . I_loc . iX0[i]  (world single-body spatial inertia)
    Sw[c]   = oXi[i] . S[:,c]            (world motion subspace, per S-column)
    ov[i]   = ov[parent] + sum_c Sw[c]*(alpha_i*qd[vinds[c]])   (serial chain)
    oh      = oY[i] . ov[i]
    B[i]    = crf(1/2 ov) oY - oY crm(1/2 ov) + icrf(1/2 oh)
    dJ[c]   = crm(ov[i]) . Sw[c]

  Backward (NB-1..0): composite oYc[parent]+=oYc[i], Bc[parent]+=Bc[i];
    dFdv[c] = oYc[i] . dJ[c] + Bc[i] . Sw[c]
    C[v_i, v_d] += alpha_i alpha_d  Sw[i]^T . dFdv[d]            (d in subtree(i))
    C[v_i, v_j] += alpha_i alpha_j ((oYc[i] Sw[i])^T dJ[j]
                                    + (Sw[i]^T Bc[i]) Sw[j])     (j ancestor of i)

Parallelism: the forward pass is SERIAL over bodies (correctness-first landing;
a P1 BFS-level fan is a separate A/B-timed perf task per the guide CAVEAT). The
main P2 lever is the C-assembly: a baked job table of (i_col, target_col, kind)
cells is fanned across threads, each cell writing a DISTINCT C entry (no write
collisions). All per-body/per-column data (S unit-axis, alpha, parent, subtree/
ancestor job table) is baked into `static const` int/T arrays read once — never
function-local const[] materialized inside the parallel loop (guide CAVEAT-2).

MIMIC (guide 1a): per-body buffers (iX0/oXi/oY/B/oYc/Bc/ov) are sized by
NB = get_num_bodies(); per-column buffers (Sw/dJ/dFdv) by n_int = total S-column
count; NEVER nv. The mimic multiplier alpha=_mimic_multiplier folds into Sw's qd
read (ov) and into the alpha_i*alpha_d / alpha_i*alpha_j C products. FLOATING
base: the root uses its real 6-column motion subspace (each column gets a distinct
internal slot, alpha=1), so no manual root permute (differs from CRBA).

Output buffer `d_coriolis` = nv x nv, row-major (C[row*nv + col]).

Emitted surface (mirrors _regressor / id_bias): inner -> device -> kernel(x2,
single_call_timing) -> host(x3 modes). The kernel takes `unsigned char *d_workspace`
as its 2nd arg (manifest convention; reserved for a future big-robot spill — the
nv x nv output fits smem at FULL for every shipped robot so it is currently unused).
"""

import numpy as np


def _coriolis_unit_axis(column):
    """Return (row, sign) of the unit entry of a single CARDINAL S column, or
    (None, None) for a Tier-B (skew/general) column (>=2 nonzero entries). The
    caller bakes the dense column for the skew case and an indexed read for the
    cardinal case (so cardinal robots stay byte-identical)."""
    values = column.reshape(-1).tolist() if hasattr(column, "reshape") else list(column)
    nz = [v for v in values if float(v) != 0.0]
    for row, value in enumerate(values):
        value = float(value)
        if abs(value) == 1.0 and len(nz) == 1:
            return row, (1 if value > 0.0 else -1)
    return None, None


def _coriolis_int_array(values):
    return ", ".join(map(str, values)) if values else "0"


def _coriolis_metadata(self):
    """Build the per-body / per-internal-column topology tables baked into the
    inner. One INTERNAL slot per S-column (n_int), mirroring the idsva_so world
    metadata so the mimic alpha-fold and the floating root fall out of per-column
    slotting (no special root code)."""
    robot = self.robot
    NB = robot.get_num_bodies()

    parent = [robot.get_parent_id(i) for i in range(NB)]
    # per-internal-column tables
    col_body = []     # internal slot -> owning body id
    col_true_vel = [] # internal slot -> reduced velocity slot (qd read)
    col_alpha = []    # internal slot -> mimic multiplier (1.0 for non-mimic)
    col_s_index = []  # internal slot -> S unit-axis row (cardinal); 0 for skew
    col_s_sign = []   # internal slot -> S unit-axis sign (cardinal); 0 for skew
    col_S_vec = []    # internal slot -> dense 6-vector S column (Tier-B skew)
    col_is_skew = []  # internal slot -> 1 if Tier-B skew, else 0
    body_col_start = [0]  # body i's internal columns are [body_col_start[i], body_col_start[i+1])
    for i in range(NB):
        joint = robot.get_joint_by_id(i)
        is_mimic = getattr(joint, "is_mimic", False)
        alpha = float(joint.get_mimic_multiplier()) if is_mimic else 1.0
        vi = robot.get_joint_index_v(i)
        if not isinstance(vi, (list, tuple, np.ndarray)):
            vi = [vi]
        vi = [int(v) for v in np.asarray(vi).reshape(-1)]
        S = np.asarray(robot.get_S_by_id(i), dtype=np.float64)
        if S.ndim == 1:
            S = S.reshape(6, 1)
        for c, vel in enumerate(vi):
            s_row, s_sign = _coriolis_unit_axis(S[:, c])
            col_body.append(i)
            col_true_vel.append(vel)
            col_alpha.append(alpha)
            if s_row is None:
                # Tier-B skew column: bake the dense S, sentinel the index/sign.
                col_s_index.append(0)
                col_s_sign.append(0)
                col_S_vec.append([float(v) for v in S[:, c].reshape(-1)])
                col_is_skew.append(1)
            else:
                col_s_index.append(s_row)
                col_s_sign.append(s_sign)
                col_S_vec.append([0.0] * 6)
                col_is_skew.append(0)
        body_col_start.append(len(col_body))
    n_int = len(col_body)

    # subtree / ancestor body lists (body ids), per body.
    subtree = {i: sorted(robot.get_subtree_by_id(i)) for i in range(NB)}

    def ancestors(i):
        res = []
        j = robot.get_parent_id(i)
        while j != -1:
            res.append(j)
            j = robot.get_parent_id(j)
        return res

    # ---- C-assembly job table: one job per (i_col, target_col) cell ----
    # kind 0 = subtree term  Sw[i_col]^T dFdv[t_col]
    # kind 1 = ancestor term (oYc[i].Sw[i_col])^T dJ[t_col] + (Sw[i_col]^T Bc[i]) Sw[t_col]
    # Each job writes a DISTINCT (row=true_vel[i_col], col=true_vel[t_col]) C cell
    # scaled by alpha[i_col]*alpha[t_col] then atomicAdd-folded (mimic columns can
    # share a reduced v-slot, so accumulate). job = (i_col, t_col, body_i, kind).
    jobs = []  # (i_col, t_col, body_i_for_composite, kind)
    for i in range(NB):
        ci0, ci1 = body_col_start[i], body_col_start[i + 1]
        # subtree: for each body d in subtree(i), every (i_col, d_col)
        for d in subtree[i]:
            cd0, cd1 = body_col_start[d], body_col_start[d + 1]
            for ic in range(ci0, ci1):
                for tc in range(cd0, cd1):
                    jobs.append((ic, tc, i, 0))
        # ancestors: for each ancestor j of i, every (i_col, j_col)
        for j in ancestors(i):
            cj0, cj1 = body_col_start[j], body_col_start[j + 1]
            for ic in range(ci0, ci1):
                for tc in range(cj0, cj1):
                    jobs.append((ic, tc, i, 1))
    job_icol = [jb[0] for jb in jobs]
    job_tcol = [jb[1] for jb in jobs]
    job_body = [jb[2] for jb in jobs]
    job_kind = [jb[3] for jb in jobs]

    return {
        "NB": NB,
        "n_int": n_int,
        "parent": parent,
        "body_col_start": body_col_start,
        "col_body": col_body,
        "col_true_vel": col_true_vel,
        "col_alpha": col_alpha,
        "col_s_index": col_s_index,
        "col_s_sign": col_s_sign,
        "col_S_vec": col_S_vec,
        "col_is_skew": col_is_skew,
        "has_skew": any(col_is_skew),
        "job_icol": job_icol,
        "job_tcol": job_tcol,
        "job_body": job_body,
        "job_kind": job_kind,
        "njobs": len(jobs),
    }


def gen_coriolis_matrix_inner_temp_mem_size(self):
    # Per-body bands (NB-sized, 36 each) + per-column bands (n_int-sized) + small
    # vectors. The XImats load owns s_XImats separately; this is the inner scratch.
    md = _coriolis_metadata(self)
    NB = md["NB"]
    n_int = md["n_int"]
    # iX0, oXi, oY, B, oYc, Bc : 36*NB each ; ov : 6*NB ; Sw, dJ, dFdv : 6*n_int each
    return 6 * (36 * NB) + 6 * NB + 3 * (6 * n_int)


def gen_coriolis_matrix_inner_function_call(self, updated_var_names=None):
    var_names = dict(
        s_coriolis_name="s_coriolis",
        s_q_name="s_q",
        s_qd_name="s_qd",
        s_temp_name="s_temp",
        gravity_name="gravity",
    )
    if updated_var_names is not None:
        for key, value in updated_var_names.items():
            var_names[key] = value
    code_start = "coriolis_matrix_inner<T>(" + var_names["s_coriolis_name"] + ", " + \
        var_names["s_q_name"] + ", " + var_names["s_qd_name"] + ", "
    code_middle = self.gen_insert_helpers_function_call()
    code_end = var_names["s_temp_name"] + ", " + var_names["gravity_name"] + ");"
    self.gen_add_code_line(code_start + code_middle + code_end)


def gen_coriolis_matrix_inner(self):
    md = _coriolis_metadata(self)
    NB = md["NB"]
    n_int = md["n_int"]
    nv = self.robot.get_num_vel()
    # s_XImats stores X for all bodies in [0, 36*NJ), then I_loc for all bodies in
    # [36*NJ, 72*NJ) — keyed by get_num_joints() (== NB), NOT get_num_pos() (which
    # differs for floating/mimic robots). Use NJ for the inertia block offset.
    NJ = self.robot.get_num_joints()
    HAS_MIMIC = self.robot_has_mimic_joints()

    func_params = [
        "s_coriolis is the output Coriolis matrix, row-major nv x nv = " + str(nv * nv),
        "s_q is the vector of joint positions",
        "s_qd is the vector of joint velocities",
        "s_temp is helper shared memory of size " + str(gen_coriolis_matrix_inner_temp_mem_size(self)),
        "gravity is the gravity constant (unused; C is gravity-independent)",
    ]
    func_notes = [
        "Assumes the XI matricies have already been updated for the given q",
        "C qd + g = nonlinear_effects ; port of pin computeCoriolisMatrix (world frame)",
    ]
    func_def_start = "void coriolis_matrix_inner(T *s_coriolis, const T *s_q, const T *s_qd, "
    func_def_end = "T *s_temp, const T gravity) {"
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params("", func_params, -1)
    func_def = func_def_start + func_def_middle + func_def_end

    self.gen_add_func_doc("Compute the Coriolis matrix C(q, qd)", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("(void)gravity;")

    # ---- baked topology tables (read-only; in registers/const, NOT per-thread const[]) ----
    self.gen_add_code_lines([
        f"static const int cor_parent[] = {{ {_coriolis_int_array(md['parent'])} }};",
        f"static const int cor_col_body[] = {{ {_coriolis_int_array(md['col_body'])} }};",
        f"static const int cor_col_true_vel[] = {{ {_coriolis_int_array(md['col_true_vel'])} }};",
        f"static const int cor_col_s_index[] = {{ {_coriolis_int_array(md['col_s_index'])} }};",
        f"static const int cor_col_s_sign[] = {{ {_coriolis_int_array(md['col_s_sign'])} }};",
        f"static const int cor_body_col_start[] = {{ {_coriolis_int_array(md['body_col_start'])} }};",
        "static const T cor_col_alpha[] = { " + ", ".join(
            "static_cast<T>(" + repr(a) + ")" for a in md['col_alpha']) + " };",
        f"static const int cor_job_icol[] = {{ {_coriolis_int_array(md['job_icol'])} }};",
        f"static const int cor_job_tcol[] = {{ {_coriolis_int_array(md['job_tcol'])} }};",
        f"static const int cor_job_body[] = {{ {_coriolis_int_array(md['job_body'])} }};",
        f"static const int cor_job_kind[] = {{ {_coriolis_int_array(md['job_kind'])} }};",
    ])
    if md["has_skew"]:
        # Tier-B (skew) only: dense per-column S table + per-column skew flag.
        # Gated on has_skew so cardinal robots emit no extra table -> byte-identical.
        flat_S = [c for col in md["col_S_vec"] for c in col]
        self.gen_add_code_lines([
            "static const T cor_col_S_vec[] = { " + ", ".join("static_cast<T>(" + repr(v) + ")" for v in flat_S) + " };",
            f"static const int cor_col_is_skew[] = {{ {_coriolis_int_array(md['col_is_skew'])} }};",
        ])

    # ---- scratch band pointers ----
    self.gen_add_code_lines([
        "// per-body bands (NB-sized, column-major 6x6) + per-column bands (n_int).",
        "T *s_iX0 = s_temp;",
        f"T *s_oXi = &s_temp[{36 * NB}];",
        f"T *s_oY  = &s_temp[{2 * 36 * NB}];",
        f"T *s_B   = &s_temp[{3 * 36 * NB}];",
        f"T *s_oYc = &s_temp[{4 * 36 * NB}];",
        f"T *s_Bc  = &s_temp[{5 * 36 * NB}];",
        f"T *s_ov  = &s_temp[{6 * 36 * NB}];",
        f"T *s_Sw  = &s_temp[{6 * 36 * NB + 6 * NB}];",
        f"T *s_dJ  = &s_temp[{6 * 36 * NB + 6 * NB + 6 * n_int}];",
        f"T *s_dFdv= &s_temp[{6 * 36 * NB + 6 * NB + 2 * 6 * n_int}];",
    ])

    # ---- zero the output (parallel) ----
    self.gen_add_code_line("// zero the output Coriolis matrix")
    self.gen_add_parallel_loop("ind", str(nv * nv))
    self.gen_add_code_line("s_coriolis[ind] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # ---- forward pass: accumulate iX0 = X_i . iX0[parent] (serial over bodies) ----
    # X_i (local joint transform) lives in s_XImats[36*jid]; I_loc in s_XImats[36*(jid+n)].
    self.gen_add_code_line("// forward: accumulated iX0[i] = X_i . iX0[parent] (column-major 6x6)")
    self.gen_add_code_line("if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {", True)
    self.gen_add_code_line("for (int jid = 0; jid < " + str(NB) + "; ++jid) {", True)
    self.gen_add_code_line("int par = cor_parent[jid];")
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) {", True)
    self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("if (par < 0) { s_iX0[jid*36 + idx] = s_XImats[jid*36 + idx]; }")
    self.gen_add_code_line("else {", True)
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int kk = 0; kk < 6; ++kk) acc += s_XImats[jid*36 + row + 6*kk] * s_iX0[par*36 + kk + 6*col];")
    self.gen_add_code_line("s_iX0[jid*36 + idx] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()  # body loop
    self.gen_add_end_control_flow()  # thread-0
    self.gen_add_sync()

    # ---- oXi = inv(iX0) (Plucker-block inverse) + oY = iX0^T I_loc iX0 (parallel over bodies) ----
    self.gen_add_code_line("// oXi[i] = inv(iX0[i]) via the Plucker-block spatial inverse (no solve)")
    self.gen_add_parallel_loop("jid", str(NB))
    self.gen_add_code_line("T *Xup = &s_iX0[jid*36]; T *Xdn = &s_oXi[jid*36];")
    self.gen_add_code_line("for (int flat = 0; flat < 36; ++flat) Xdn[flat] = static_cast<T>(0);")
    self.gen_add_code_line("for (int flat = 0; flat < 36; ++flat) {", True)
    self.gen_add_code_line("int sub_idx = flat % 18;")
    self.gen_add_code_line("if (flat % 18 == 1 || flat % 18 == 4 || flat % 18 == 8 || flat % 18 == 11) { Xdn[flat] = Xup[flat + 5]; Xdn[flat + 5] = Xup[flat]; }")
    self.gen_add_code_line("else if (flat % 18 == 2 || flat % 18 == 5) { Xdn[flat] = Xup[flat + 10]; Xdn[flat + 10] = Xup[flat]; }")
    self.gen_add_code_line("else if (sub_idx != 6 && sub_idx != 9 && sub_idx != 13 && sub_idx != 16 && sub_idx != 12 && sub_idx != 15) { Xdn[flat] = Xup[flat]; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()  # parallel jid
    self.gen_add_sync()

    self.gen_add_code_line("// oY[i] = iX0[i]^T . I_loc . iX0[i]  (world single-body spatial inertia)")
    self.gen_add_parallel_loop("ind", str(36 * NB))
    self.gen_add_code_line("int jid = ind / 36; int idx = ind % 36; int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("const T *Iloc = &s_XImats[36*(jid + " + str(NJ) + ")];")
    self.gen_add_code_line("T *X = &s_iX0[jid*36];")
    # tmp = I_loc . X  ; oY = X^T . tmp  -> oY[row+6col] = sum_kk X[kk+6row] * (I_loc . X)[kk+6col]
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int kk = 0; kk < 6; ++kk) {", True)
    self.gen_add_code_line("T ix = static_cast<T>(0);")
    self.gen_add_code_line("for (int mm = 0; mm < 6; ++mm) ix += Iloc[kk + 6*mm] * X[mm + 6*col];")
    self.gen_add_code_line("acc += X[kk + 6*row] * ix;")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("s_oY[ind] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # ---- Sw[c] = oXi[body] . S[:,c] (alpha NOT folded into Sw itself; folded in qd read
    #      for ov and into the alpha products in C). Per the oracle, Sw = oXi @ S (raw S);
    #      the mimic multiplier scales qd (ov) and the C contributions. ----
    self.gen_add_code_line("// Sw[c] = oXi[body(c)] . S_col(c)  (world motion subspace, per internal column)")
    self.gen_add_parallel_loop("c", str(n_int))
    self.gen_add_code_line("int jid = cor_col_body[c]; int s_row = cor_col_s_index[c]; T s_sgn = static_cast<T>(cor_col_s_sign[c]);")
    if md["has_skew"]:
        # Tier B: dense Sw = oXi @ S_col (6x6 * 6 matvec); cardinal stays indexed.
        self.gen_add_code_line("if (cor_col_is_skew[c]) {", True)
        self.gen_add_code_line("for (int row = 0; row < 6; ++row) { T a = static_cast<T>(0); for (int kk = 0; kk < 6; ++kk) a += s_oXi[jid*36 + kk*6 + row] * cor_col_S_vec[c*6 + kk]; s_Sw[c*6 + row] = a; }")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("else { for (int row = 0; row < 6; ++row) s_Sw[c*6 + row] = s_sgn * s_oXi[jid*36 + s_row*6 + row]; }")
    else:
        self.gen_add_code_line("for (int row = 0; row < 6; ++row) s_Sw[c*6 + row] = s_sgn * s_oXi[jid*36 + s_row*6 + row];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # ---- ov[i] = ov[parent] + sum_c Sw[c]*(alpha_i*qd[true_vel]) (serial chain over bodies) ----
    self.gen_add_code_line("// ov[i] = ov[parent] + sum_{c in body i} Sw[c]*(alpha_c * qd[true_vel(c)])")
    self.gen_add_code_line("if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {", True)
    self.gen_add_code_line("for (int jid = 0; jid < " + str(NB) + "; ++jid) {", True)
    self.gen_add_code_line("int par = cor_parent[jid];")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) {", True)
    self.gen_add_code_line("T acc = (par < 0) ? static_cast<T>(0) : s_ov[par*6 + row];")
    self.gen_add_code_line("for (int c = cor_body_col_start[jid]; c < cor_body_col_start[jid + 1]; ++c) {", True)
    self.gen_add_code_line("acc += s_Sw[c*6 + row] * (cor_col_alpha[c] * s_qd[cor_col_true_vel[c]]);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("s_ov[jid*6 + row] = acc;")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()  # body loop
    self.gen_add_end_control_flow()  # thread-0
    self.gen_add_sync()

    # ---- B[i] = crf(1/2 ov) oY - oY crm(1/2 ov) + icrf(1/2 oh) ; dJ[c] = crm(ov) Sw[c] ----
    self.gen_add_code_line("// B[i] (per-body bias) and dJ[c] = crm(ov[i]) Sw[c]")
    self.gen_add_parallel_loop("jid", str(NB))
    self.gen_add_code_line("T *v = &s_ov[jid*6]; T *oY = &s_oY[jid*36]; T *Bi = &s_B[jid*36];")
    # oh = oY . v
    self.gen_add_code_line("T oh[6];")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) { T a = static_cast<T>(0); for (int kk = 0; kk < 6; ++kk) a += oY[row + 6*kk]*v[kk]; oh[row] = a; }")
    self.gen_add_code_line("T hv[6]; for (int r = 0; r < 6; ++r) hv[r] = static_cast<T>(0.5) * v[r];")
    self.gen_add_code_line("T hoh[6]; for (int r = 0; r < 6; ++r) hoh[r] = static_cast<T>(0.5) * oh[r];")
    # crm(hv), crf(hv)=-crm(hv)^T, icrf(hoh) as explicit column-major 6x6 builds
    _emit_crm_cm(self, "Crm", "hv")
    _emit_crf_cm(self, "Crf", "hv")
    _emit_icrf_cm(self, "Icrf", "hoh")
    # Bi = Crf . oY - oY . Crm + Icrf  (all column-major)
    self.gen_add_code_line("for (int idx = 0; idx < 36; ++idx) {", True)
    self.gen_add_code_line("int row = idx % 6; int col = idx / 6;")
    self.gen_add_code_line("T t1 = static_cast<T>(0); T t2 = static_cast<T>(0);")
    self.gen_add_code_line("for (int kk = 0; kk < 6; ++kk) { t1 += Crf[row + 6*kk]*oY[kk + 6*col]; t2 += oY[row + 6*kk]*Crm[kk + 6*col]; }")
    self.gen_add_code_line("Bi[idx] = t1 - t2 + Icrf[idx];")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()  # parallel jid
    self.gen_add_sync()

    self.gen_add_code_line("// dJ[c] = crm(ov[body(c)]) . Sw[c]")
    self.gen_add_parallel_loop("c", str(n_int))
    self.gen_add_code_line("int jid = cor_col_body[c]; T *v = &s_ov[jid*6];")
    _emit_crm_cm(self, "Crm", "v")
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) { T a = static_cast<T>(0); for (int kk = 0; kk < 6; ++kk) a += Crm[row + 6*kk]*s_Sw[c*6 + kk]; s_dJ[c*6 + row] = a; }")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # ---- backward composite: oYc, Bc accumulate root-ward (serial reverse body order) ----
    # dFdv depends on the FINAL composite for body i; children (larger ids) are
    # processed first so oYc[i]/Bc[i] are final when we compute dFdv. Mirror the
    # oracle: init oYc=oY, Bc=B; reverse loop computes dFdv[i]'s columns then adds
    # into parent.
    self.gen_add_code_line("// init composites oYc = oY, Bc = B")
    self.gen_add_parallel_loop("ind", str(36 * NB))
    self.gen_add_code_line("s_oYc[ind] = s_oY[ind]; s_Bc[ind] = s_B[ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    self.gen_add_code_line("// backward composite accumulation + per-column dFdv (serial, reverse body order)")
    self.gen_add_code_line("if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {", True)
    self.gen_add_code_line("for (int jid = " + str(NB - 1) + "; jid >= 0; --jid) {", True)
    # dFdv[c] = oYc[jid] dJ[c] + Bc[jid] Sw[c]  for c in body jid's columns
    self.gen_add_code_line("for (int c = cor_body_col_start[jid]; c < cor_body_col_start[jid + 1]; ++c) {", True)
    self.gen_add_code_line("for (int row = 0; row < 6; ++row) {", True)
    self.gen_add_code_line("T a = static_cast<T>(0);")
    self.gen_add_code_line("for (int kk = 0; kk < 6; ++kk) a += s_oYc[jid*36 + row + 6*kk]*s_dJ[c*6 + kk] + s_Bc[jid*36 + row + 6*kk]*s_Sw[c*6 + kk];")
    self.gen_add_code_line("s_dFdv[c*6 + row] = a;")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()  # column loop
    # add into parent composite
    self.gen_add_code_line("int par = cor_parent[jid];")
    self.gen_add_code_line("if (par >= 0) { for (int idx = 0; idx < 36; ++idx) { s_oYc[par*36 + idx] += s_oYc[jid*36 + idx]; s_Bc[par*36 + idx] += s_Bc[jid*36 + idx]; } }")
    self.gen_add_end_control_flow()  # body loop
    self.gen_add_end_control_flow()  # thread-0
    self.gen_add_sync()

    # ---- C assembly: P2 fan over the baked job table (disjoint i_col contributions;
    #      mimic columns may share a reduced v-slot -> atomicAdd). ----
    self.gen_add_code_line("// C assembly: fan over baked (i_col, t_col, body, kind) jobs")
    self.gen_add_parallel_loop("job", str(md["njobs"]))
    self.gen_add_code_line("int ic = cor_job_icol[job]; int tc = cor_job_tcol[job]; int bi = cor_job_body[job]; int kind = cor_job_kind[job];")
    self.gen_add_code_line("int rrow = cor_col_true_vel[ic]; int ccol = cor_col_true_vel[tc];")
    self.gen_add_code_line("T coeff = cor_col_alpha[ic] * cor_col_alpha[tc];")
    self.gen_add_code_line("T val;")
    self.gen_add_code_line("if (kind == 0) {", True)
    self.gen_add_code_line("// subtree: Sw[ic]^T dFdv[tc]")
    self.gen_add_code_line("val = static_cast<T>(0); for (int r = 0; r < 6; ++r) val += s_Sw[ic*6 + r]*s_dFdv[tc*6 + r];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("else {", True)
    self.gen_add_code_line("// ancestor: (oYc[bi] Sw[ic])^T dJ[tc] + (Sw[ic]^T Bc[bi]) Sw[tc]")
    self.gen_add_code_line("T ag[6]; for (int row = 0; row < 6; ++row) { T a = static_cast<T>(0); for (int kk = 0; kk < 6; ++kk) a += s_oYc[bi*36 + row + 6*kk]*s_Sw[ic*6 + kk]; ag[row] = a; }")
    self.gen_add_code_line("T mt[6]; for (int colk = 0; colk < 6; ++colk) { T a = static_cast<T>(0); for (int kk = 0; kk < 6; ++kk) a += s_Sw[ic*6 + kk]*s_Bc[bi*36 + kk + 6*colk]; mt[colk] = a; }")
    self.gen_add_code_line("T t1 = static_cast<T>(0); T t2 = static_cast<T>(0);")
    self.gen_add_code_line("for (int r = 0; r < 6; ++r) { t1 += ag[r]*s_dJ[tc*6 + r]; t2 += mt[r]*s_Sw[tc*6 + r]; }")
    self.gen_add_code_line("val = t1 + t2;")
    self.gen_add_end_control_flow()
    if HAS_MIMIC:
        # mimic columns can share a reduced v-slot -> accumulate with atomicAdd.
        self.gen_add_code_line("atomicAdd(&s_coriolis[rrow*" + str(nv) + " + ccol], coeff * val);")
    else:
        self.gen_add_code_line("s_coriolis[rrow*" + str(nv) + " + ccol] += coeff * val;")
    self.gen_add_end_control_flow()  # job loop
    self.gen_add_sync()
    self.gen_add_end_function()


# ---- explicit column-major 6x6 cross-operator builders (registers, per-thread) ----
def _emit_crm_cm(self, dst, v):
    """crm(v): motion cross 6x6, column-major dst[row+6*col]. Matches cross_operator."""
    lines = [f"T {dst}[36];"]
    M = [
        [0, "-{v}[2]", "{v}[1]", 0, 0, 0],
        ["{v}[2]", 0, "-{v}[0]", 0, 0, 0],
        ["-{v}[1]", "{v}[0]", 0, 0, 0, 0],
        [0, "-{v}[5]", "{v}[4]", 0, "-{v}[2]", "{v}[1]"],
        ["{v}[5]", 0, "-{v}[3]", "{v}[2]", 0, "-{v}[0]"],
        ["-{v}[4]", "{v}[3]", 0, "-{v}[1]", "{v}[0]", 0],
    ]
    for r in range(6):
        for c in range(6):
            e = M[r][c]
            if e == 0:
                lines.append(f"{dst}[{r} + 6*{c}] = static_cast<T>(0);")
            else:
                lines.append(f"{dst}[{r} + 6*{c}] = {e.format(v=v)};")
    self.gen_add_code_lines(lines)


def _emit_crf_cm(self, dst, v):
    """crf(v) = -crm(v)^T, column-major. crf[r+6c] = -crm[c+6r]."""
    lines = [f"T {dst}[36];"]
    M = [
        [0, "-{v}[2]", "{v}[1]", 0, 0, 0],
        ["{v}[2]", 0, "-{v}[0]", 0, 0, 0],
        ["-{v}[1]", "{v}[0]", 0, 0, 0, 0],
        [0, "-{v}[5]", "{v}[4]", 0, "-{v}[2]", "{v}[1]"],
        ["{v}[5]", 0, "-{v}[3]", "{v}[2]", 0, "-{v}[0]"],
        ["-{v}[4]", "{v}[3]", 0, "-{v}[1]", "{v}[0]", 0],
    ]
    # crf[r][c] = -crm[c][r]
    for r in range(6):
        for c in range(6):
            e = M[c][r]  # crm[c][r]
            if e == 0:
                lines.append(f"{dst}[{r} + 6*{c}] = static_cast<T>(0);")
            else:
                e = e.format(v=v)
                neg = e[1:] if e.startswith("-") else "-" + e
                lines.append(f"{dst}[{r} + 6*{c}] = {neg};")
    self.gen_add_code_lines(lines)


def _emit_icrf_cm(self, dst, v):
    """icrf(v): inverse force cross 6x6, column-major. Matches RBDReference.icrf
    (note the trailing global negation -> entries below are the FINAL signed values)."""
    lines = [f"T {dst}[36];"]
    # RBDReference.icrf returns -res with res given row-major below; bake -res here.
    res = [
        [0, "-{v}[2]", "{v}[1]", 0, "-{v}[5]", "{v}[4]"],
        ["{v}[2]", 0, "-{v}[0]", "{v}[5]", 0, "-{v}[3]"],
        ["-{v}[1]", "{v}[0]", 0, "-{v}[4]", "{v}[3]", 0],
        [0, "-{v}[5]", "{v}[4]", 0, 0, 0],
        ["{v}[5]", 0, "-{v}[3]", 0, 0, 0],
        ["-{v}[4]", "{v}[3]", 0, 0, 0, 0],
    ]
    for r in range(6):
        for c in range(6):
            e = res[r][c]  # final = -res[r][c]
            if e == 0:
                lines.append(f"{dst}[{r} + 6*{c}] = static_cast<T>(0);")
            else:
                e = e.format(v=v)
                neg = e[1:] if e.startswith("-") else "-" + e  # apply global -
                lines.append(f"{dst}[{r} + 6*{c}] = {neg};")
    self.gen_add_code_lines(lines)


def gen_coriolis_matrix_device(self):
    nv = self.robot.get_num_vel()
    func_params = [
        "s_coriolis is the output Coriolis matrix (nv x nv)",
        "s_q is the vector of joint positions",
        "s_qd is the vector of joint velocities",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant (unused)",
    ]
    func_def = ("void coriolis_matrix_device(T *s_coriolis, const T *s_q, const T *s_qd, "
                "const robotModel<T> *d_robotModel, const T gravity) {")
    shared_mem_size = gen_coriolis_matrix_inner_temp_mem_size(self)
    self.gen_device_wrapper(
        "Compute the Coriolis matrix C(q, qd)", func_def,
        shared_mem_size,
        lambda: gen_coriolis_matrix_inner_function_call(self),
        func_params=func_params,
        extra_t_buffers=None, include_linalg_scratch=True)


def _emit_coriolis_matrix_kernel_body_for_flags(self, NUM_POS, nv, in_size, out_size,
                                                use_workspace_temp, coriolis_in_smem, single_call_timing):
    """Emit coriolis_matrix_kernel body for one tier's spill flags (mirror crba).
    coriolis_in_smem=True: s_coriolis output in smem; False: s_coriolis -> L2-pinned SO band.
    use_workspace_temp=False: inner spatial-recursion scratch in smem; True: -> L2-pinned GRAD section."""
    shared_mem_size = 0 if use_workspace_temp else gen_coriolis_matrix_inner_temp_mem_size(self)
    # s_coriolis (nv*nv) is the write-once output. At rung0 (coriolis_in_smem) it lives in
    # smem -> byte-identical to the pre-spill emission. At the spill rungs it drops from the
    # smem arena and repoints to the L2-pinned SO band (crba/dccrba-style), shrinking by nv*nv.
    if coriolis_in_smem:
        extra_t_buffers = [("s_q_qd", in_size), ("s_coriolis", out_size)]
    else:
        extra_t_buffers = [("s_q_qd", in_size)]
    self.gen_XImats_helpers_temp_shared_memory_code(
        shared_mem_size, extra_t_buffers=extra_t_buffers, include_linalg_scratch=True)
    self.gen_add_code_line("T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(NUM_POS) + "];")
    if not coriolis_in_smem:
        self.gen_add_code_line("T *s_coriolis;  // repointed to the L2-pinned SO band (output spill) per timing branch")
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q_qd", str(in_size), stride="stride_q_qd")
        if use_workspace_temp:
            # whole inner band spilled: the smem s_temp slot is null. Repoint s_temp at the
            # GRAD section BEFORE the XImats helper so its sincos scratch + the inner have a
            # valid backing store. Disjoint from the SO band where s_coriolis lives.
            self.gen_add_code_line("T *coriolis_d_workspace = reinterpret_cast<T *>(&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()]);")
            self.gen_add_code_line("s_temp = coriolis_d_workspace;")
        elif coriolis_in_smem:
            self.gen_add_code_line("(void)d_workspace;")
        if not coriolis_in_smem:
            self.gen_add_code_line("s_coriolis = reinterpret_cast<T *>(&d_workspace[k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()]);")
        # mjx input convert: quaternion wxyz->xyzw AND qd[0:3] = R^T qd[0:3] (the
        # Coriolis matrix reads qd, so the base-linear velocity must be in pin frame)
        # before the XImats build (so X[0] is built from the reordered quaternion).
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert(q_name="s_q", qd_name="s_qd")
            self.gen_add_end_control_flow()
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call()
        gen_coriolis_matrix_inner_function_call(self)
        self.gen_add_sync()
        # mjx output: Coriolis matrix is a congruence/similarity G C G^T (base rows
        # then base cols). C is non-symmetric but the similarity code is identical;
        # the row-major C buffer fed to the column-major helper yields G C G^T
        # because the congruence commutes with transposition.
        if self.robot.floating_base:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_congruence("s_coriolis", nv)
            self.gen_add_end_control_flow()
        self.gen_kernel_save_result("coriolis", str(out_size), stride=str(out_size))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q_qd", str(in_size))
        if use_workspace_temp:
            self.gen_add_code_line("T *coriolis_d_workspace = reinterpret_cast<T *>(d_workspace);")
            self.gen_add_code_line("s_temp = coriolis_d_workspace;")
        elif coriolis_in_smem:
            self.gen_add_code_line("(void)d_workspace;")
        if not coriolis_in_smem:
            self.gen_add_code_line("s_coriolis = reinterpret_cast<T *>(&d_workspace[GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()]);")
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_load_update_XImats_helpers_function_call()
        gen_coriolis_matrix_inner_function_call(self)
        self.gen_add_sync()
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("coriolis", str(out_size))


def gen_coriolis_matrix_kernel(self, single_call_timing=False):
    NUM_POS = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    out_size = nv * nv
    in_size = NUM_POS + nv
    func_params = [
        "d_coriolis is the output Coriolis matrix, row-major nv x nv = " + str(out_size),
        "d_workspace is the L2-pinned spill workspace (used at the output-spill / whole-band tiers)",
        "d_q_qd is the vector of joint positions, velocities (q|qd)",
        "stride_q_qd is the stride between each (q, qd) pair",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant (unused)",
        "num_timesteps is the length of the trajectory points",
    ]
    func_def_start = "void coriolis_matrix_kernel(T *d_coriolis, unsigned char *d_workspace, const T *d_q_qd, const int stride_q_qd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    self.gen_add_func_doc("Compute the Coriolis matrix C(q, qd)", [], func_params, None)
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag, LAST
    # after RESOURCE_TIER so existing positional <T,TIER> call sites are unaffected;
    # default false if-constexpr-elides the epilogue -> byte-identical PTX.
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # 3-rung surgical ladder: rung0 full (s_coriolis + inner band in smem); rung1 output-spill
    # (s_coriolis -> L2-pinned SO band, hot inner band stays smem); rung2 whole-band (both
    # output and inner band spilled). gen_tier_dispatch de-dups tiers that share a pick, so
    # rung0-only robots emit one byte-identical body (Gate A).
    picks = getattr(self, "coriolis_matrix_spill_tier_3way", (0, 0, 0))
    self.gen_tier_dispatch(picks, lambda pick:
        _emit_coriolis_matrix_kernel_body_for_flags(self, NUM_POS, nv, in_size, out_size,
            use_workspace_temp=(pick == 2), coriolis_in_smem=(pick == 0), single_call_timing=single_call_timing))
    self.gen_add_end_function()


def gen_coriolis_matrix_host(self, mode=0):
    single_call_timing = True if mode == 1 else False
    compute_only = True if mode == 2 else False
    nv = self.robot.get_num_vel()
    out_size = nv * nv
    func_params = [
        "hd_data is the packaged input and output pointers (q/qd inputs; output written to hd_data->d_coriolis, nv*nv*num_timesteps floats)",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant (unused)",
        "num_timesteps is the length of the trajectory points",
        "streams are pointers to CUDA streams for async memory transfers",
    ]
    func_def_start = "void coriolis_matrix(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end = "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    self.gen_add_func_doc("Compute the Coriolis matrix C(q, qd)", [], func_params, None)
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
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"coriolis_matrix requires all-data or dynamics gridData\");")
    coriolis_kernel_tmpl = "coriolis_matrix_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "coriolis_matrix_kernel<T, RESOURCE_TIER>"
    func_call_start = coriolis_kernel_tmpl + "<<<block_dimms,thread_dimms,CORIOLIS_MATRIX_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_coriolis,hd_data->d_workspace,hd_data->d_q_qd,stride_q_qd,"
    func_call_end = "d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("coriolis_matrix_kernel<", "coriolis_matrix_kernel_single_timing<")
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
    func_call_mem = "if (USE_COMPRESSED_MEM) {" + func_call_start + func_call_end + "}"
    func_call_mem2 = "else                    {" + (func_call_start + func_call_end).replace("hd_data->d_q_qd", "hd_data->d_q_qd_u") + "}"
    func_call_code = [func_call_mem, func_call_mem2]
    if single_call_timing:
        func_call_code.insert(0, "struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("gpuErrchkKernel();")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"coriolis_matrix\", CORIOLIS_MATRIX_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back into the gridData host buffer (hd_data->d_coriolis -> hd_data->h_coriolis)",
            "gpuErrchk(cudaMemcpy(hd_data->h_coriolis,hd_data->d_coriolis," +
            ("num_timesteps*" if not single_call_timing else "") + str(out_size) + "*sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();",
        ])
    else:
        self.gen_add_code_line("gpuErrchkKernel();")
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("coriolis_matrix"))
    self.gen_add_end_function()


def gen_coriolis_matrix(self):
    # inner -> device -> kernel(s) -> host(s)
    gen_coriolis_matrix_inner(self)
    gen_coriolis_matrix_device(self)
    gen_coriolis_matrix_kernel(self, single_call_timing=False)
    gen_coriolis_matrix_kernel(self, single_call_timing=True)
    gen_coriolis_matrix_host(self, 0)
    gen_coriolis_matrix_host(self, 1)
    gen_coriolis_matrix_host(self, 2)

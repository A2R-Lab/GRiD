"""External-force gradient column (section A of the differentiability plan).

Emits the three f_ext-gradient outputs (all q-only, f_ext-value independent —
f_ext enters RNEA additively & linearly so the Jacobian carries no f_ext value):

  dtau/dfext      = -J(q)^T          (section A.1)  stacked body-Jacobian transpose
  dqdd/dfext      =  M^{-1} J^T      (section A.2)  operational-space inverse-inertia
  d(inverse_dynamics_gradient)/dfext  = -dJ^T/dq         (section A.3)  q-derivative of the body Jacobian

J^T is the stacked SPATIAL body-Jacobian transpose in each link's LOCAL frame:

  -J^T[v_j, 6*i + k] = -( S_j^T  P_{i,j} )_k   for j on path(root->i), else 0
        P_{i,j} = X[j+1]^T X[j+2]^T ... X[i]^T   (composed local 6x6 spatial
                                                  transforms, [angular;linear])

This is EXACTLY the matrix the RNEA backward force sweep applies (f[parent] +=
X^T f), so feeding a unit local wrench at link i and running the back-prop yields
column block i of -J^T. We emit it as the explicit per-(body, chain-joint) build
from s_XImats (the same 6x6 local transforms RNEA loads), one running 6x6 product
per body chained root-ward. The convention (LOCAL frame, SUBTRACTED -> sign -)
is locked to the T4 forward path (apply_external_forces: f[:,i] -= f_ext[i]).

The output layout is body-major: s_dtau_dfext is nv x (6*NB), column-major in the
[v_row + nv*col] sense used by the rest of GRiD's dense gradient outputs.

dqdd/dfext is -s_Minv @ s_dtau_dfext (a single nv x nv * nv x 6NB GEMM reusing the
minv s_Minv). dJ^T/dq is central-FD of the analytic -J^T over each
generalized coordinate (the same FD-on-Jacobian strategy the d2ee GPU path uses);
the q-dot block is identically zero (J^T is q-only) and is not stored.
"""


def _f_ext_gradient_chain_jobs(self):
    """Bake the per-(body i, chain-joint j, S-column) fill jobs for -J^T.

    Returns (NB, nv, jobs) where each job is a dict:
      { 'i': body id, 'j': chain joint id, 'vi': velocity slot, 'Scol': the 6-vec
        motion subspace column, 'tf_chain': the ordered joint ids [j+1, ..., i]
        whose local 6x6 motion transforms X[m] are applied (left-fold) to Scol to
        push it from joint j's frame down to body i's frame, 'alpha': the mimic
        multiplier of joint j (1.0 for non-mimic). }

    The body-Jacobian column of body i for chain joint j is
      col = X[i] X[i-1] ... X[j+1] S_j   (Featherstone motion transforms),
    written to row v_j, column-block i of -J^T (negated). Out-of-chain columns are
    absent here (left zero by the inner's init).

    MIMIC fold: a mimic joint j shares its TARGET's reduced v-slot
    (get_joint_index_v(j) == get_joint_index_v(target)) and its body moves
    alpha * the target's rate, so its geometric-Jacobian column folds into the
    shared slot scaled by alpha (== RBDReference.rnea_bpass:
    c[inds_f] += mimic_scale * S^T f). The reduction below already accumulates
    (+=) all jobs sharing one (i, v_j); baking 'alpha' lets each contribution be
    alpha-weighted. For a non-mimic robot every alpha == 1.0, so the emit is
    byte-identical to the legacy path (guarded by robot_has_mimic_joints()).
    """
    import numpy as _np
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    jobs = []
    for i in range(NB):
        chain = sorted(self.robot.get_ancestors_by_id(i)) + [i]
        for j in chain:
            S = _np.asarray(self.robot.get_S_by_id(j), dtype=_np.float64)
            if S.ndim == 1:
                S = S.reshape(-1, 1)
            try:
                vinds = self.robot.get_joint_index_v(j)
            except Exception:
                vinds = self.robot.get_joint_index_q(j)
            if not isinstance(vinds, (list, tuple, _np.ndarray)):
                vinds = [vinds]
            vinds = list(vinds)
            # transform chain: the body-Jacobian column of body i for chain joint
            # j is the motion subspace S_j transformed from joint j's frame DOWN to
            # body i's frame: col = X[i] X[i-1] ... X[j+1] S_j  (Featherstone motion
            # transforms, [angular;linear]). Applying as a left-fold over a running
            # 6-vector means apply X[m] (NO transpose) for m = j+1, j+2, ..., i. We
            # bake the chain in that (root-ward-reversed) order. This is exactly the
            # transpose of the RNEA backward force sweep's P_{i,j} = X[j+1]^T..X[i]^T
            # (verified bit-exact vs the rnea_bpass unit-wrench oracle).
            tf_chain = []
            m = i
            while m != j:
                tf_chain.append(int(m))
                m = self.robot.get_parent_id(m)
            tf_chain = list(reversed(tf_chain))  # j+1, j+2, ..., i
            alpha = self._alpha_for_jid(j)
            for c in range(S.shape[1]):
                vi = vinds[c] if c < len(vinds) else vinds[-1]
                Scol = [float(x) for x in S[:6, c]]
                jobs.append({
                    "i": int(i), "j": int(j), "vi": int(vi),
                    "Scol": Scol, "tf_chain": tf_chain, "alpha": float(alpha),
                })
    return NB, nv, jobs


def gen_f_ext_gradient_inner_temp_mem_size(self):
    # The inner stages ONE 6-vector contribution per (body, chain-joint, S-col) job in
    # s_temp (s_feg_slab, size njobs*6), then lane-0 reduces it into the output. On deep
    # serial chains njobs*6 exceeds the legacy 2*36*NB double-buffer reserve (h2_plus:
    # 7026 > 5472), so take the max -- small robots (njobs*6 <= 2*36*NB) keep the legacy
    # size byte-identical, while big robots get the real slab. At the f_ext full/out-spill
    # rungs the minv scratch dominates this anyway; it only sets the floor at the deep
    # rung (minv-F spilled) and for the dq kernel's in-smem s_jt_temp.
    NB = self.robot.get_num_bodies()
    _, _, jobs = _f_ext_gradient_chain_jobs(self)
    return max(2 * 36 * NB, 6 * len(jobs))


def gen_f_ext_gradient_inner_function_call(self, updated_var_names=None):
    var_names = dict(
        s_dtau_dfext_name="s_dtau_dfext",
        s_q_name="s_q",
        s_temp_name="s_temp",
    )
    if updated_var_names is not None:
        for k, v in updated_var_names.items():
            var_names[k] = v
    code_start = "f_ext_gradient_jacobianT_inner<T>(" + var_names["s_dtau_dfext_name"] + ", " + var_names["s_q_name"] + ", "
    code_mid = self.gen_insert_helpers_function_call()
    code_end = var_names["s_temp_name"] + ");"
    self.gen_add_code_line(code_start + code_mid + code_end)


def gen_f_ext_gradient_jacobianT_inner(self):
    """Emit f_ext_gradient_jacobianT_inner: builds -J^T into s_dtau_dfext.

    s_dtau_dfext is nv x (6*NB), zeroed then filled per chain job. Assumes
    s_XImats holds the per-joint LOCAL 6x6 spatial transforms for the current q.
    """
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    _, _, jobs = _f_ext_gradient_chain_jobs(self)
    HAS_MIMIC = self.robot_has_mimic_joints()

    func_params = [
        "s_dtau_dfext is the output dtau/dfext = -J^T, size NV*(6*NB) = " + str(nv * 6 * NB),
        "s_q is the vector of joint positions (used only via s_XImats)",
        "s_temp is helper shared memory of size " + str(self.gen_f_ext_gradient_inner_temp_mem_size()),
    ]
    func_notes = [
        "Assumes s_XImats is updated already for the current s_q.",
        "Output is the LOCAL-frame stacked body-Jacobian transpose, negated.",
        "Column block i (6 cols) is the joint-torque response to a unit local",
        "wrench on body i; nonzero only on rows v_j for j on path(root->i).",
    ]
    func_def_start = "void f_ext_gradient_jacobianT_inner(T *s_dtau_dfext, const T *s_q, "
    func_def_end = "T *s_temp) {"
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -1)
    func_def = func_def_start + func_def_end

    self.gen_add_func_doc("Computes dtau/dfext = -J(q)^T (stacked local body-Jacobian transpose)",
                          func_notes, func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_add_code_line("(void)s_q;")

    out_size = nv * 6 * NB
    # zero the output
    self.gen_add_code_line("// zero the full nv x 6*NB output (out-of-chain cols stay zero)")
    self.gen_add_parallel_loop("ind", str(out_size))
    self.gen_add_code_line("s_dtau_dfext[ind] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

    # For each job (body i, chain joint j): the -J^T column block is -col, where
    #   col = X[i] X[i-1] ... X[j+1] S_j   (push the motion subspace S_j from joint
    # j's frame down to body i's frame). Built as a left-fold over a running 6-vector:
    # col = S_j, then col := X[m] @ col for m = j+1..i. This is the transpose of the
    # RNEA backward force sweep's S_j^T (X[j+1]^T...X[i]^T).
    # -col writes to output row v_j, column block i. Jobs sharing (i, v_j) accumulate
    # (+=); the final += reduction runs serially on lane 0 (the slab is filled in
    # parallel) to keep it race-free.
    #
    # MIMIC: a mimic joint j shares its TARGET's v_j slot, so its column folds into
    # that shared slot scaled by its multiplier alpha (the mimic body moves alpha*
    # target_rate) -- matching RBDReference.rnea_bpass's c[inds_f] += mimic_scale*S^T f.
    # alpha is baked per-job (1.0 non-mimic) and only emitted for mimic robots, so
    # non-mimic grid.cuh is byte-identical.
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Per chain job: col(v_j, body i) = -(X[i]..X[j+1] S_j) (motion-transform pushdown)")
    self.gen_add_code_line("//")

    if len(jobs) > 0:
        # Bake the per-job chains as a flat const int array with [start,len]
        # offsets, plus the per-job (body i, output row v_j) and the 6-vec S column.
        flat_chain = []
        job_off = []
        job_len = []
        job_i = []
        job_vrow = []
        job_S = []
        job_alpha = []
        for job in jobs:
            job_off.append(len(flat_chain))
            job_len.append(len(job["tf_chain"]))
            flat_chain.extend(job["tf_chain"])
            job_i.append(job["i"])
            job_vrow.append(job["vi"])
            job_S.append(job["Scol"])
            job_alpha.append(job.get("alpha", 1.0))

        def _ints(vals):
            return "{ " + ", ".join(str(v) for v in vals) + " }"

        def _floats(vals):
            return "{ " + ", ".join("static_cast<T>({:.17g})".format(v) for v in vals) + " }"

        if len(flat_chain) == 0:
            flat_chain = [0]  # avoid zero-size array
        self.gen_add_code_line("static const int feg_chain[]   = " + _ints(flat_chain) + ";")
        self.gen_add_code_line("static const int feg_job_off[]  = " + _ints(job_off) + ";")
        self.gen_add_code_line("static const int feg_job_len[]  = " + _ints(job_len) + ";")
        self.gen_add_code_line("static const int feg_job_i[]    = " + _ints(job_i) + ";")
        self.gen_add_code_line("static const int feg_job_vrow[] = " + _ints(job_vrow) + ";")
        flatS = [s for job in job_S for s in job]
        self.gen_add_code_line("const T feg_job_S[] = " + _floats(flatS) + ";")
        # MIMIC fold: per-job mimic multiplier alpha (only emitted for mimic robots
        # so non-mimic grid.cuh stays byte-identical; alpha == 1.0 for non-mimic).
        if HAS_MIMIC:
            self.gen_add_code_line("const T feg_job_alpha[] = " + _floats(job_alpha) + ";")

        njobs = len(jobs)
        # Two-phase to avoid += races on shared (i, v_j) destinations: (1) each job
        # (one work-item) computes its -col 6-vector chain in parallel into a per-job
        # scratch slab (njobs*6 in s_temp); (2) a lane-0 serial reduce sums each
        # job's slab into the output row v_j / column-block i (folding shared v-slots
        # in deterministic order). The chain matvec is a cheap short serial loop.
        slab = njobs * 6
        self.gen_add_code_line("// per-job contribution slab in s_temp (njobs*6)")
        self.gen_add_code_line("T *s_feg_slab = s_temp;   // size " + str(slab))
        self.gen_add_parallel_loop("jb", str(njobs))
        self.gen_add_code_line("int off = feg_job_off[jb]; int len = feg_job_len[jb];")
        self.gen_add_code_line("T col[6];")
        self.gen_add_code_line("#pragma unroll")
        self.gen_add_code_line("for (int r = 0; r < 6; ++r) { col[r] = feg_job_S[6*jb + r]; }")
        # chain: for each m in feg_chain[off..off+len): col := X[m] @ col
        self.gen_add_code_line("for (int s = 0; s < len; ++s) {", True)
        self.gen_add_code_line("int m = feg_chain[off + s];")
        self.gen_add_code_line("const T *X = &s_XImats[36*m];")
        self.gen_add_code_line("T tmp[6];")
        self.gen_add_code_line("#pragma unroll")
        self.gen_add_code_line("for (int r = 0; r < 6; ++r) {", True)
        # col := X @ col (motion transform, NO transpose). X is column-major 6x6:
        # element (row r, col c) = X[6*c + r]. So (X @ col)[r] = sum_c X[6*c + r] col[c]
        # = dot_prod with stride 6 on X starting at r, stride 1 on col.
        self.gen_add_code_line("tmp[r] = dot_prod<T,6,6,1>(&X[r], col);")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("#pragma unroll")
        self.gen_add_code_line("for (int r = 0; r < 6; ++r) { col[r] = tmp[r]; }")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("#pragma unroll")
        if HAS_MIMIC:
            # mimic fold: scale this job's column by its mimic multiplier so jobs
            # sharing a v-slot (mimic + target) accumulate alpha-weighted in reduce.
            self.gen_add_code_line("T a = feg_job_alpha[jb];")
            self.gen_add_code_line("for (int r = 0; r < 6; ++r) { s_feg_slab[6*jb + r] = -a * col[r]; }")
        else:
            self.gen_add_code_line("for (int r = 0; r < 6; ++r) { s_feg_slab[6*jb + r] = -col[r]; }")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

        # reduce slab -> output: for each job add its 6 cells into
        # s_dtau_dfext[vrow + nv*(6*i + r)]  (column-major nv-row layout)
        self.gen_add_code_line("// reduce per-job contributions into the output (serial over jobs to fold shared v-slots)")
        self.gen_add_serial_ops()
        self.gen_add_code_line("for (int jb = 0; jb < " + str(njobs) + "; ++jb) {", True)
        self.gen_add_code_line("int i = feg_job_i[jb]; int vrow = feg_job_vrow[jb];")
        self.gen_add_code_line("#pragma unroll")
        self.gen_add_code_line("for (int r = 0; r < 6; ++r) {", True)
        self.gen_add_code_line("s_dtau_dfext[vrow + " + str(nv) + "*(6*i + r)] += s_feg_slab[6*jb + r];")
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
        self.gen_add_sync()
    self.gen_add_end_function()


def gen_f_ext_gradient_output_size(self):
    """Number of T elements in EACH of the two first-order f_ext-grad outputs:
    dtau_dfext and dqdd_dfext are nv x (6*NB)."""
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    return nv * 6 * NB


def _f_ext_gradient_dq_smem_count(self):
    """T-element shared count for the -dJ^T/dq kernel arena.

    Layout in s_temp: s_qpert[n_pos] | s_JTp[nv*6NB] | s_JTm[nv*6NB] |
    s_jt_temp[jt_inner] | s_xi_scratch[xi]. The XImats buffer + s_q live in their
    own arena regions (declared via gen_XImats_helpers_temp_shared_memory_code).

    Floating base additionally needs an nv-sized velocity-perturbation buffer
    (s_dv) for the SE(3) Lie-group retract that perturbs the root twist (and the
    revolute joints) for the central FD -- the root's 6-DoF tangent cannot be a
    scalar q[i] += h, it must go through grid_integrate_floating_q."""
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    n_pos = self.robot.get_num_pos()
    out6 = 6 * NB
    jt_temp = self.gen_f_ext_gradient_inner_temp_mem_size()
    xi_scratch = self.gen_load_update_XImats_helpers_temp_mem_size()
    dv_extra = nv if self.robot.floating_base else 0
    return n_pos + dv_extra + 2 * nv * out6 + jt_temp + xi_scratch


def _emit_f_ext_gradient_dq_perturb(self, sign):
    """Emit the FD perturbation of s_q into s_qpert by `sign`*fd_h on coordinate qi.

    Fixed base: a plain scalar retract q[qi] += sign*h (re-seeded from s_q each
    time). The position and velocity coordinates coincide, so this is exact.

    Floating base: the root (jid 0) carries a 6-DoF SE(3) twist, so a scalar add
    on the quaternion prefix is NOT the tangent perturbation the oracle uses. We
    instead build a velocity perturbation dv (size nv, all zero except
    dv[qi] = sign*h) and apply the SAME on-device Lie-group retract the integrator
    uses, grid_integrate_floating_q(s_q, dv, s_qpert). For root qi in [0,6) this
    is an SE(3)-exp of the perturbed twist (matching RBDReference.integrate, which
    the numpy/pin oracle calls with dv[i]=h); for revolute qi >= 6 the helper's
    tail does the plain Euler add q[7+...] += dv[6+...]. This makes the FD
    perturbation uniform across root + joints and exactly mirrors the oracle."""
    nv = self.robot.get_num_vel()
    n_pos = self.robot.get_num_pos()
    if not self.robot.floating_base:
        # scalar retract: s_qpert = s_q then s_qpert[qi] += sign*h
        self.gen_add_parallel_loop("ind", str(n_pos))
        self.gen_add_code_line("s_qpert[ind] = s_q[ind];")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_add_serial_ops()
        self.gen_add_code_line("s_qpert[qi] " + ("+= fd_h;" if sign > 0 else "-= fd_h;"))
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        return
    # floating: build dv (nv) then grid_integrate_floating_q(s_q, dv, s_qpert).
    self.gen_add_parallel_loop("ind", str(nv))
    self.gen_add_code_line("s_dv[ind] = (ind == qi) ? (" + ("fd_h" if sign > 0 else "-fd_h") + ") : static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("grid_integrate_floating_q<T, " + str(n_pos) + ">(s_q, s_dv, s_qpert);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def _emit_f_ext_gradient_dq_body(self, out_ptr_expr, in_timestep_loop, jt_smem):
    """Emit the per-timestep -dJ^T/dq FD body. Assumes s_q (smem), s_XImats, and
    s_temp arena are already declared/loaded. Writes into `out_ptr_expr` (a global
    or shared pointer to the nv*6NB*nv output for this timestep).

    The per-coordinate perturbation is a scalar retract on a fixed base and an
    SE(3) Lie-group retract on a floating base (see
    _emit_f_ext_gradient_dq_perturb); both feed the same central FD of -J^T.

    h2_plus-spill: the two nv x (6*NB) -J^T FD buffers (s_JTp, s_JTm) are the dominant
    band (~288 KB on h2_plus). When jt_smem is False (rung 1) they live in the L2-pinned
    d_workspace SO section (s_JTp at SO base, s_JTm at SO base + nv*6NB) and the smem layout
    drops them; s_qpert/s_dv/s_jt_temp/s_xi_scratch stay hot in smem."""
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    n_pos = self.robot.get_num_pos()
    fb = self.robot.floating_base
    out6 = 6 * NB
    jt_temp = self.gen_f_ext_gradient_inner_temp_mem_size()
    self.gen_add_code_line("T *s_f_ext_gradient_dq = " + out_ptr_expr + ";")
    self.gen_add_code_line("const T fd_h = static_cast<T>(1e-3);")
    self.gen_add_code_line("T *s_qpert = s_temp;")
    dv_extra = nv if fb else 0
    if fb:
        # s_dv velocity-perturbation buffer (nv) lives at the head, after s_qpert.
        self.gen_add_code_line("T *s_dv = &s_temp[" + str(n_pos) + "];")
    base = n_pos + dv_extra
    # The s_JTp/s_JTm pair occupies 2*nv*out6 smem slots only when jt_smem; when spilled,
    # the smem layout closes that gap and the pair points at the d_workspace SO section.
    jt_slot = 2 * nv * out6 if jt_smem else 0
    self.gen_add_code_line("T *s_jt_temp = &s_temp[" + str(base + jt_slot) + "];")
    self.gen_add_code_line("T *s_xi_scratch = &s_temp[" + str(base + jt_slot + jt_temp) + "];")
    if jt_smem:
        self.gen_add_code_line("T *s_JTp = &s_temp[" + str(base) + "];")
        self.gen_add_code_line("T *s_JTm = &s_temp[" + str(base + nv * out6) + "];")
        self.gen_add_code_line("(void)d_workspace;")
    else:
        _ws_base = ("k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + " if in_timestep_loop else "") + "GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()"
        self.gen_add_code_line("T *s_JTp = reinterpret_cast<T *>(&d_workspace[" + _ws_base + "]);")
        self.gen_add_code_line("T *s_JTm = reinterpret_cast<T *>(&d_workspace[" + _ws_base + " + " + str(nv * out6) + "*sizeof(T)]);")
    # loop over each q coordinate qi in [0, nv)
    self.gen_add_code_line("for (int qi = 0; qi < " + str(nv) + "; ++qi) {", True)
    # +h perturbation -> s_qpert
    _emit_f_ext_gradient_dq_perturb(self, +1)
    self.gen_load_update_XImats_helpers_function_call(updated_var_names={"s_q_name": "s_qpert", "s_temp_name": "s_xi_scratch"})
    self.gen_add_sync()
    self.gen_f_ext_gradient_inner_function_call(updated_var_names={
        "s_dtau_dfext_name": "s_JTp", "s_q_name": "s_qpert", "s_temp_name": "s_jt_temp"})
    self.gen_add_sync()
    # -h perturbation -> s_qpert (re-seeded from s_q inside the perturb helper)
    _emit_f_ext_gradient_dq_perturb(self, -1)
    self.gen_load_update_XImats_helpers_function_call(updated_var_names={"s_q_name": "s_qpert", "s_temp_name": "s_xi_scratch"})
    self.gen_add_sync()
    self.gen_f_ext_gradient_inner_function_call(updated_var_names={
        "s_dtau_dfext_name": "s_JTm", "s_q_name": "s_qpert", "s_temp_name": "s_jt_temp"})
    self.gen_add_sync()
    # central diff into output column qi: out[...][qi] = (JTp - JTm)/(2h).
    # s_JTp/s_JTm already hold -J^T (the inner emits -J^T), so this is -dJ^T/dq.
    # output layout: [ (row v_j) + nv*(6NB col) + nv*6NB*qi ]
    self.gen_add_parallel_loop("ind", str(nv * out6))
    self.gen_add_code_line("s_f_ext_gradient_dq[ind + " + str(nv * out6) + "*qi] = "
                           "(s_JTp[ind] - s_JTm[ind]) / (static_cast<T>(2)*fd_h);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_control_flow()  # for qi
    # Restore s_XImats / s_q-state for the ORIGINAL q so any later use is correct.
    self.gen_load_update_XImats_helpers_function_call(updated_var_names={"s_temp_name": "s_xi_scratch"})
    self.gen_add_sync()


def gen_f_ext_gradient_dq_kernel(self, single_call_timing=False):
    """Emit f_ext_gradient_dq_kernel: the mixed second-order block
    d(inverse_dynamics_gradient)/dfext = -dJ^T/dq  (section A.3), size nv x (6*NB) x nv.

    Central finite-difference of the analytic A.1 -J^T over each generalized
    coordinate (the same FD-on-Jacobian approach the d2ee GPU path uses for the
    kinematic Hessian). A velocity-coordinate perturbation equals q[i] += h
    directly on a fixed base; on a floating base the root (jid 0) is perturbed
    along its 6-DoF twist via the on-device SE(3) Lie-group retract
    grid_integrate_floating_q (revolute joints keep the scalar add), exactly
    mirroring the numpy + pinocchio oracle's self.integrate(q, dv) FD. The q-dot
    block is identically zero (J^T is q-only) and is not emitted."""
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    n_pos = self.robot.get_num_pos()
    out6 = 6 * NB
    out_each = nv * out6 * nv

    func_params = [
        "d_f_ext_gradient_dq is the output -dJ^T/dq, size NV*(6*NB)*NV = " + str(out_each) + " per timestep",
        "d_q is the joint positions, stride_q the per-timestep stride",
        "d_robotModel is the initialized model helpers on the GPU",
        "NUM_TIMESTEPS is the trajectory length (or timing reps)",
    ]
    # h2_plus-spill: d_workspace 2nd arg backs the spilled s_JTp/s_JTm pair at rung 1.
    func_def_start = ("void f_ext_gradient_dq_kernel(T *d_f_ext_gradient_dq, unsigned char *d_workspace, "
                      "const T *d_q, const int stride_q, ")
    func_def_end = "const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("(", "_single_timing(")
    self.gen_add_func_doc("Compute -dJ^T/dq = d(inverse_dynamics_gradient)/dfext (section A.3, fixed base, batched kernel)",
                          [], func_params, None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    _dq_full = _f_ext_gradient_dq_smem_count(self)

    def _emit_body(pick):
        # pick 0: JT pair in smem (full arena). pick 1: spill the pair to d_workspace.
        jt_smem = (pick == 0)
        scratch = _dq_full if jt_smem else _dq_full - 2 * nv * out6
        self.gen_XImats_helpers_temp_shared_memory_code(
            scratch, extra_t_buffers=[("s_q", n_pos)], include_linalg_scratch=True)
        if jt_smem:
            self.gen_add_code_line("(void)d_workspace;")
        if not single_call_timing:
            self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
            self.gen_kernel_load_inputs("q", str(n_pos), stride="stride_q")
            self.gen_add_code_line("// compute")
            _emit_f_ext_gradient_dq_body(self, "&d_f_ext_gradient_dq[k*" + str(out_each) + "]", in_timestep_loop=True, jt_smem=jt_smem)
            self.gen_add_end_control_flow()
        else:
            self.gen_kernel_load_inputs("q", str(n_pos))
            self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
            self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
            self.gen_anti_licm_input_reload("q", str(n_pos), feedback_from="f_ext_gradient_dq")
            _emit_f_ext_gradient_dq_body(self, "d_f_ext_gradient_dq", in_timestep_loop=False, jt_smem=jt_smem)
            self.gen_add_end_control_flow()

    picks = getattr(self, "f_ext_gradient_dq_spill_tier_3way", (0, 0, 0))
    self.gen_tier_dispatch(picks, _emit_body)
    self.gen_add_end_function()


def gen_f_ext_gradient_dq_host(self, mode=0):
    """Host wrapper for the -dJ^T/dq kernel (fixed + floating base)."""
    single_call_timing = (mode == 1)
    compute_only = (mode == 2)
    func_params = [
        "hd_data is the packaged input and output pointers",
        "d_robotModel is the initialized model helpers on the GPU",
        "num_timesteps is the trajectory length (or timing reps)",
        "streams are CUDA streams for async transfers",
    ]
    func_def_start = ("void f_ext_gradient_dq(gridData<T, KIND> *hd_data, "
                      "const robotModel<T> *d_robotModel, const int num_timesteps,")
    func_def_end = "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    self.gen_add_func_doc("Compute -dJ^T/dq = d(inverse_dynamics_gradient)/dfext (host wrapper, fixed base)", [], func_params, None)
    self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"f_ext_gradient_dq requires all-data or dynamics gridData\");")
    out_each = "NUM_VEL*6*NUM_BODIES*NUM_VEL"
    func_call_start = ("f_ext_gradient_dq_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,F_EXT_GRADIENT_DQ_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>("
                       "hd_data->d_f_ext_gradient_dq,hd_data->d_workspace,hd_data->d_q,stride_q,")
    func_call_end = "d_robotModel,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("kernel<T, RESOURCE_TIER>", "kernel_single_timing<T, RESOURCE_TIER>")
    if not compute_only:
        self.gen_add_code_lines([
            "// start code with memory transfer",
            "int stride_q;",
            "if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*" + ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}",
            "else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*" + ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}",
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
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"f_ext_gradient_dq\", F_EXT_GRADIENT_DQ_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    # h2_plus-spill: L2-pin d_workspace when the tier spills the s_JTp/s_JTm pair into it.
    _feg_dq_ws_bytes = ("GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()" if single_call_timing
                        else "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(num_timesteps)")
    self.gen_add_code_line("if (!F_EXT_GRADIENT_DQ_JT_IN_SMEM<RESOURCE_TIER>() && hd_data->d_workspace != nullptr) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, " + _feg_dq_ws_bytes + "));}")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back",
            "gpuErrchk(cudaMemcpy(hd_data->h_f_ext_gradient_dq,hd_data->d_f_ext_gradient_dq," + out_each + "*" + ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();"])
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("f_ext_gradient_dq"))
    self.gen_add_end_function()


def gen_f_ext_gradient_device(self):
    """Emit f_ext_gradient_device: computes dtau/dfext = -J^T and
    dqdd/dfext = M^{-1} J^T into caller-provided shared buffers.

    Reuses minv_inner for s_Minv (the same inverse-inertia buffer forward_dynamics_gradient
    consumes) and the f_ext_gradient_jacobianT_inner for -J^T, then one
    nv x nv * nv x 6NB GEMM (dqdd = -Minv @ dtau). Both outputs are q-only and
    f_ext-VALUE independent (so this device takes q, not f_ext)."""
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    out6 = 6 * NB

    func_params = [
        "s_dtau_dfext is the output dtau/dfext = -J^T, size NV*6*NB = " + str(nv * out6),
        "s_dqdd_dfext is the output dqdd/dfext = M^{-1} J^T, size NV*6*NB = " + str(nv * out6),
        "s_q is the vector of joint positions",
        "d_robotModel is the initialized model helpers on the GPU",
    ]
    func_notes = [
        "Both outputs are q-only (f_ext enters RNEA additively & linearly).",
        "dqdd = -Minv @ dtau (since dtau = -J^T, M^{-1} J^T = -Minv @ dtau).",
    ]
    func_def = ("void f_ext_gradient_device(T *s_dtau_dfext, T *s_dqdd_dfext, "
                "const T *s_q, const robotModel<T> *d_robotModel) {")
    # scratch: max of the J^T inner temp and the minv inner temp, plus an
    # nv*nv s_Minv buffer.
    jt_temp = self.gen_f_ext_gradient_inner_temp_mem_size()
    minv_temp = self.gen_minv_inner_temp_mem_size()
    shared_extra = nv * nv + max(jt_temp, minv_temp)

    self.gen_add_func_doc("Compute the f_ext gradient (dtau/dfext, dqdd/dfext)",
                          func_notes, func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_XImats_helpers_temp_shared_memory_code(
        shared_extra, extra_t_buffers=None, include_linalg_scratch=True)
    self.gen_load_update_XImats_helpers_function_call()
    # s_Minv lives at head of s_temp; the inner scratch follows.
    self.gen_add_code_line("T *s_Minv = s_temp;")
    self.gen_add_code_line("T *s_fext_temp = &s_temp[" + str(nv * nv) + "];")
    # build -J^T
    self.gen_f_ext_gradient_inner_function_call(
        updated_var_names={"s_temp_name": "s_fext_temp"})
    self.gen_add_sync()
    # Minv into s_Minv (F kept in smem; inner slices it from the tail of its temp)
    self.gen_minv_inner_function_call(
        updated_var_names={"s_Minv_name": "s_Minv", "s_temp_name": "s_fext_temp"},
        f_in_smem_expr="true")
    self.gen_add_sync()
    # densify Minv upper->full (minv outputs SYMMETRIC_UPPER)
    self.gen_add_code_line("// densify Minv (minv emits symmetric-upper)")
    self.gen_add_parallel_loop("ind", str(nv * nv))
    self.gen_add_code_line("int r = ind % " + str(nv) + "; int c = ind / " + str(nv) + ";")
    self.gen_add_code_line("if (c < r) { s_Minv[r + " + str(nv) + "*c] = s_Minv[c + " + str(nv) + "*r]; }")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # dqdd = -Minv @ dtau  (both nv x 6NB; Minv is nv x nv symmetric)
    self.gen_add_code_line("// dqdd/dfext = M^{-1} J^T = -Minv @ (dtau/dfext)")
    self.gen_add_parallel_loop("ind", str(nv * out6))
    self.gen_add_code_line("int row = ind % " + str(nv) + "; int col = ind / " + str(nv) + ";")
    # (Minv @ dtau)[row,col] = sum_k Minv[row,k] dtau[k,col]
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int k = 0; k < " + str(nv) + "; ++k) {", True)
    self.gen_add_code_line("acc += s_Minv[row + " + str(nv) + "*k] * s_dtau_dfext[k + " + str(nv) + "*col];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("s_dqdd_dfext[ind] = -acc;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()


def _emit_f_ext_gradient_kernel_body_for_flags(self, pick, single_call_timing):
    """Emit the f_ext_gradient kernel body for one tier's spill pick.

    pick 0 (full):      both outputs (s_dtau, s_dqdd) + minv-F in smem.
    pick 1 (out-spill): s_dqdd -> d_workspace SO section; s_dtau + minv-F in smem.
    pick 2 (deep):      s_dqdd + s_dtau -> d_workspace SO section; minv's 6*nv*nv
                        F-region -> GRAD-section minv-F offset (F_IN_SMEM=false).
    The pick is static here, so sizes are literal ints (no constexpr SLOTs) and the
    s_temp scratch shrinks to the minv no-F band at the deep rung. select_shared_tier
    picks the least-spill rung that fits, so non-spilling robots collapse to pick 0."""
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    n_pos = self.robot.get_num_pos()
    out_each = nv * 6 * NB
    jt_temp = self.gen_f_ext_gradient_inner_temp_mem_size()
    minv_temp = self.gen_minv_inner_temp_mem_size()       # minv scratch incl. F-region
    minv_no_F = self.gen_minv_inner_no_F_size()            # minv scratch WITHOUT F-region
    dqdd_smem = (pick == 0)
    dtau_smem = (pick <= 1)                                # minv-F placement == DTAU placement
    scratch = nv * nv + max(jt_temp, (minv_temp if dtau_smem else minv_no_F))
    self.gen_XImats_helpers_temp_shared_memory_code(
        scratch, extra_t_buffers=[("s_q", n_pos),
                                  ("s_dtau_dfext", out_each if dtau_smem else 0),
                                  ("s_dqdd_dfext", out_each if dqdd_smem else 0)],
        include_linalg_scratch=True)
    if dqdd_smem and dtau_smem:
        self.gen_add_code_line("(void)d_workspace;")

    def _repoint_spilled_output(in_timestep_loop):
        # repoint spilled output(s) at the L2-pinned d_workspace SO section (per-timestep
        # slot; reused safely -- f_ext_gradient never co-runs with the SO kernels). s_dqdd
        # at the SO base, s_dtau at SO base + out_each (deep rung only).
        base = ("k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + " if in_timestep_loop else "") + "GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()"
        if not dqdd_smem:
            self.gen_add_code_line("s_dqdd_dfext = reinterpret_cast<T *>(&d_workspace[" + base + "]);")
        if not dtau_smem:
            self.gen_add_code_line("s_dtau_dfext = reinterpret_cast<T *>(&d_workspace[" + base + " + " + str(out_each) + "*sizeof(T)]);")

    def _body(in_timestep_loop):
        self.gen_add_code_line("T *s_Minv = s_temp;")
        self.gen_add_code_line("T *s_fext_temp = &s_temp[" + str(nv * nv) + "];")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_f_ext_gradient_inner_function_call(
            updated_var_names={"s_temp_name": "s_fext_temp"})
        self.gen_add_sync()
        if dtau_smem:
            # minv keeps F in the tail of s_fext_temp (smem).
            self.gen_minv_inner_function_call(
                updated_var_names={"s_Minv_name": "s_Minv", "s_temp_name": "s_fext_temp"},
                f_in_smem_expr="true")
        else:
            # deep rung: route minv's 6*nv*nv F-region to the GRAD-section minv-F offset.
            _minv_ws = ("k*GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>() + " if in_timestep_loop else "") + "GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()"
            self.gen_add_code_line("T *minv_d_workspace = reinterpret_cast<T *>(&d_workspace[" + _minv_ws + "]);")
            self.gen_minv_inner_function_call(
                updated_var_names={"s_Minv_name": "s_Minv", "s_temp_name": "s_fext_temp", "d_workspace_name": "minv_d_workspace"},
                f_in_smem_expr="false")
        self.gen_add_sync()
        self.gen_add_code_line("// densify Minv (minv emits symmetric-upper)")
        self.gen_add_parallel_loop("ind", str(nv * nv))
        self.gen_add_code_line("int r = ind % " + str(nv) + "; int c = ind / " + str(nv) + ";")
        self.gen_add_code_line("if (c < r) { s_Minv[r + " + str(nv) + "*c] = s_Minv[c + " + str(nv) + "*r]; }")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_add_code_line("// dqdd/dfext = M^{-1} J^T = -Minv @ (dtau/dfext)")
        self.gen_add_parallel_loop("ind", str(out_each))
        self.gen_add_code_line("int row = ind % " + str(nv) + "; int col = ind / " + str(nv) + ";")
        self.gen_add_code_line("T acc = static_cast<T>(0);")
        self.gen_add_code_line("for (int k = 0; k < " + str(nv) + "; ++k) { acc += s_Minv[row + " + str(nv) + "*k] * s_dtau_dfext[k + " + str(nv) + "*col]; }")
        self.gen_add_code_line("s_dqdd_dfext[ind] = -acc;")
        self.gen_add_end_control_flow()
        self.gen_add_sync()

    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q", str(n_pos), stride="stride_q")
        _repoint_spilled_output(in_timestep_loop=True)
        self.gen_add_code_line("// compute")
        _body(in_timestep_loop=True)
        self.gen_kernel_save_result("dtau_dfext", str(out_each), stride=str(out_each))
        self.gen_kernel_save_result("dqdd_dfext", str(out_each), stride=str(out_each))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q", str(n_pos))
        _repoint_spilled_output(in_timestep_loop=False)
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_anti_licm_input_reload("q", str(n_pos), feedback_from="dtau_dfext")
        _body(in_timestep_loop=False)
        self.gen_anti_licm_output_write("dtau_dfext")
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("dtau_dfext", str(out_each))
        self.gen_kernel_save_result("dqdd_dfext", str(out_each))


def gen_f_ext_gradient_kernel(self, single_call_timing=False):
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()

    func_params = [
        "d_dtau_dfext / d_dqdd_dfext are the two outputs (each NV*6*NB per timestep)",
        "d_workspace is the L2-pinned global spill buffer (outputs / minv-F at spilled tiers)",
        "d_q is the joint positions, stride_q the per-timestep stride",
        "d_robotModel is the initialized model helpers on the GPU",
        "NUM_TIMESTEPS is the trajectory length (or timing reps)",
    ]
    # g1/h2_plus-spill: the kernel takes d_workspace as its 3rd arg. A 3-rung surgical
    # ladder (full / s_dqdd-spill / +s_dtau+minv-F-spill) backs the L2-pinned spill;
    # non-spilling robots collapse to the full body (smem-arena-bytes unchanged).
    func_def_start = ("void f_ext_gradient_kernel(T *d_dtau_dfext, T *d_dqdd_dfext, unsigned char *d_workspace, "
                      "const T *d_q, const int stride_q, ")
    func_def_end = "const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("(", "_single_timing(")
    self.gen_add_func_doc("Compute the f_ext gradient (batched kernel)",
                          [], func_params, None)
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    picks = getattr(self, "f_ext_gradient_spill_tier_3way", (0, 0, 0))
    self.gen_tier_dispatch(picks, lambda pick:
        _emit_f_ext_gradient_kernel_body_for_flags(self, pick, single_call_timing))
    self.gen_add_end_function()


def gen_f_ext_gradient_host(self, mode=0):
    single_call_timing = (mode == 1)
    compute_only = (mode == 2)
    func_params = [
        "hd_data is the packaged input and output pointers",
        "d_robotModel is the initialized model helpers on the GPU",
        "num_timesteps is the trajectory length (or timing reps)",
        "streams are CUDA streams for async transfers",
    ]
    func_def_start = ("void f_ext_gradient(gridData<T, KIND> *hd_data, "
                      "const robotModel<T> *d_robotModel, const int num_timesteps,")
    func_def_end = "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    self.gen_add_func_doc("Compute the f_ext gradient (host wrapper)", [], func_params, None)
    self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false, gridDataKind KIND = GRID_DATA_ALL, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    self.gen_add_code_line("static_assert(KIND == GRID_DATA_ALL || KIND == GRID_DATA_DYNAMICS, \"f_ext_gradient requires all-data or dynamics gridData\");")
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    out_each = "NUM_VEL*6*NUM_BODIES"
    # g1-spill: pass hd_data->d_workspace as the kernel's 3rd arg. At the spilled
    # default tier (s_dqdd_dfext in d_workspace) it is read; at TIER_SHARED unused.
    func_call_start = ("f_ext_gradient_kernel<T, RESOURCE_TIER><<<block_dimms,thread_dimms,F_EXT_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>("
                       "hd_data->d_dtau_dfext,hd_data->d_dqdd_dfext,hd_data->d_workspace,hd_data->d_q,stride_q,")
    func_call_end = "d_robotModel,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("kernel<T, RESOURCE_TIER>", "kernel_single_timing<T, RESOURCE_TIER>")
    if not compute_only:
        self.gen_add_code_lines([
            "// start code with memory transfer",
            "int stride_q;",
            "if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*" + ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}",
            "else {stride_q = 3*NUM_JOINTS; gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*" + ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}",
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
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"f_ext_gradient\", F_EXT_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    # g1-spill: L2-pin d_workspace when the default tier spills s_dqdd_dfext into it.
    _feg_ws_bytes = ("GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()" if single_call_timing
                     else "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(num_timesteps)")
    self.gen_add_code_line("if (!F_EXT_GRADIENT_DQDD_IN_SMEM<RESOURCE_TIER>() && hd_data->d_workspace != nullptr) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, " + _feg_ws_bytes + "));}")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back",
            "gpuErrchk(cudaMemcpy(hd_data->h_dtau_dfext,hd_data->d_dtau_dfext," + out_each + "*" + ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchk(cudaMemcpy(hd_data->h_dqdd_dfext,hd_data->d_dqdd_dfext," + out_each + "*" + ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();"])
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("f_ext_gradient"))
    self.gen_add_end_function()


def gen_f_ext_gradient(self):
    """Emit the full f_ext-gradient family: J^T inner, device, kernels, hosts.

    A.1 (-J^T) and A.2 (M^-1 J^T) are emitted for ALL base modes. A.3 (-dJ^T/dq,
    the mixed second-order block) is now emitted for BOTH base modes: fixed-base
    perturbs each q coordinate by a scalar q[i] += h, floating-base perturbs the
    root (jid 0) along its 6-DoF twist via the on-device SE(3) Lie-group retract
    grid_integrate_floating_q (revolute joints keep the scalar add). The numpy +
    pinocchio oracle ships A.3 for BOTH base modes."""
    # The A.3 floating-base FD calls grid_integrate_floating_q (an SE(3) Lie-group
    # helper). gen_f_ext_gradient runs BEFORE gen_integrator in gen_all_code, and
    # ee_pose_hessian (the only other early emitter) may not be requested, so emit
    # the Lie helpers here if floating and not already emitted (gen_integrator then
    # skips its own emit via the same _lie_helpers_emitted flag, avoiding a C++
    # redefinition).
    if self.robot.floating_base and not getattr(self, "_lie_helpers_emitted", False):
        self.gen_lie_group_helpers()
        self._lie_helpers_emitted = True
    self.gen_f_ext_gradient_jacobianT_inner()
    self.gen_f_ext_gradient_device()
    # A.3 (-dJ^T/dq) GPU device emit: the mixed second-order block, now emitted for
    # both base modes. The kernel/host wire it as the third output
    # (s_f_ext_gradient_dq, size nv*6NB*nv); the first-order kernel/host are unchanged.
    self.gen_f_ext_gradient_kernel(single_call_timing=False)
    self.gen_f_ext_gradient_kernel(single_call_timing=True)
    self.gen_f_ext_gradient_host(mode=0)
    self.gen_f_ext_gradient_host(mode=1)
    self.gen_f_ext_gradient_host(mode=2)
    # A.3 (-dJ^T/dq): own kernel + host (both base modes); separate output buffer
    # d_f_ext_gradient_dq so the first-order kernel/host stay byte-identical. The
    # _f_ext_gradient_dq_emitted gate keys the KERNEL_ATTR_MANIFEST registration.
    self._f_ext_gradient_dq_emitted = True
    self.gen_f_ext_gradient_dq_kernel(single_call_timing=False)
    self.gen_f_ext_gradient_dq_kernel(single_call_timing=True)
    self.gen_f_ext_gradient_dq_host(mode=0)
    self.gen_f_ext_gradient_dq_host(mode=1)
    self.gen_f_ext_gradient_dq_host(mode=2)

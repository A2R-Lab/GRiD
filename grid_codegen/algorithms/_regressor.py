"""Joint-torque regressor CUDA emit (D.4 / G.x of the sysID plan).

`tau = Y(q, qd, qdd) . pi`, where `pi = [pi_1; ...; pi_NB]` stacks each link's
10 standard inertial parameters and `Y` is the joint-torque regressor
(nv x 10*NB). Inverse dynamics is exactly affine in each link's spatial inertia,
so `Y = d(tau)/d(pi)` is exact (no finite differences).

This mirrors the verified numpy reference `RBDReference._RegressorMixin`:

  - Parameter basis (per link, GRiD/URDF convention):
        pi_i = [ m, h(3)=m*c, I_O(6)=[Ixx, Ixy, Ixz, Iyy, Iyz, Izz] ]
    with I_O the inertia about the link-frame ORIGIN.
  - Spatial convention: internal [angular; linear] (Featherstone), matching
    `RBDReference.rnea` / `dual_cross_operator` and GRiD's `s_vaf` layout.
  - The body regressor `Y_body,i` (6x10) satisfies `f_i = Y_body,i . pi_i` with
    `f_i = I_i a_i + v_i x* (I_i v_i)`; column k is `dI_k a_i + crf(v_i) dI_k v_i`
    where `dI_k` is the k-th basis spatial inertia and `crf == fx_times_v`
    (`crf = -crm^T`, the dual/force cross — see `fx_times_v` in the spatial
    algebra helpers).
  - The joint regressor is the RNEA backward force sweep run with a 6x10
    right-hand side instead of a 6x1 force: each link's 6x10 block is propagated
    toward the root with X^T and projected onto each ancestor DOF's subspace
    with the same +/-S selection RNEA uses for `s_c`.

Layout: the inner reuses the RNEA forward sweep (`inverse_dynamics_inner_vaf`)
to populate `s_vaf` with per-link (v, a), then builds `s_Y` (row-major nv x
10*NB: row = DOF, column block i = link i's 10 params).

Emitted surface (mirrors the simple first-order algorithms: thin host/kernel,
fat inner):
  inverse_dynamics_regressor_inner   (__device__, placement-free; assumes XImats)
  inverse_dynamics_regressor_device  (__device__; owns XImats load + scratch)
  inverse_dynamics_regressor_kernel  (__global__; batched, writes d_Y)
  inverse_dynamics_regressor         (__host__ launcher; writes hd_data->d_Y)

The output is shaped nv x 10*NB. R2: it is a gridData field (hd_data->d_Y, sized
10*NB*nv*num_timesteps floats); the host launcher writes it and copies the result
back into hd_data->h_Y (uniform `(hd_data, model, ...)` host signature).
"""

from ._frame_jacobian import _emit_world_transform_chainup
from grid_codegen.helpers._code_generation_helpers import host_q_input_transfer_lines, _gen_mjx_build_R_lines, gen_workspace_repoint_line, gen_workspace_cast_expr, host_mode_flags, host_q_qd_input_transfer_lines, mangle_host_func_defs, wrap_host_single_call_timing
from grid_codegen.helpers._code_generation_helpers import gen_host_wrapper_head

# The 10 basis spatial-inertia derivatives dI/dpi_k in GRiD [angular; linear]
# 6x6 order, for pi = [m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz].
#   I(pi) = [[ I_O,        skew(h) ],
#            [ skew(h)^T,  m * I3  ]]
# This is the SAME `_BASIS_I` the numpy reference builds; we hard-code the
# nonzero entries per column so the device assembles `dI_k @ x` as a short fixed
# expression (no 6x6 dense multiply). Returned as a list (per param k) of
# (row, col, value) nonzeros of the 6x6 basis matrix.
def _regressor_basis_nonzeros():
    bases = []
    # k=0  m: lower-right 3x3 = I3 -> (3,3),(4,4),(5,5)
    bases.append([(3, 3, 1.0), (4, 4, 1.0), (5, 5, 1.0)])
    # k=1..3  h (first moment): top-right skew(h) and bottom-left skew(h)^T
    #   hx -> S[1,2]=-1, S[2,1]=1   (top-right block at rows 0..2, cols 3..5)
    bases.append([(1, 5, -1.0), (2, 4, 1.0), (5, 1, -1.0), (4, 2, 1.0)])
    #   hy -> S[0,2]=1,  S[2,0]=-1
    bases.append([(0, 5, 1.0), (2, 3, -1.0), (5, 0, 1.0), (3, 2, -1.0)])
    #   hz -> S[0,1]=-1, S[1,0]=1
    bases.append([(0, 4, -1.0), (1, 3, 1.0), (4, 0, -1.0), (3, 1, 1.0)])
    # k=4..9  I_O [Ixx,Ixy,Ixz,Iyy,Iyz,Izz] -> symmetric top-left 3x3
    bases.append([(0, 0, 1.0)])              # Ixx
    bases.append([(0, 1, 1.0), (1, 0, 1.0)])  # Ixy
    bases.append([(0, 2, 1.0), (2, 0, 1.0)])  # Ixz
    bases.append([(1, 1, 1.0)])              # Iyy
    bases.append([(1, 2, 1.0), (2, 1, 1.0)])  # Iyz
    bases.append([(2, 2, 1.0)])              # Izz
    return bases


_REGRESSOR_BASIS = _regressor_basis_nonzeros()


def _emit_mjx_base_rotate_rows_rowmajor(self, mat, n_rows, n_cols, q_name="s_q"):
    """ROW-MAJOR base-linear row rotate for the regressors.

    The column-major siblings rotate the base-linear ROWS 0:3 of a matrix
    (`mat[r + n_rows*c]`). The regressor `s_Y` is ROW-MAJOR (`mat[row*n_cols + c]`,
    row = DOF, col = param), so its base-linear rows 0,1,2 live at offsets
    `0*n_cols`, `1*n_cols`, `2*n_cols` with the column index stepping by 1 -- a
    layout the column-major helper cannot express. This emits the same
    transform (rows0:3 <- R . rows, R from the xyzw quaternion, matching
    `mujoco_convention.rotation_from_quat_xyzw`) for the row-major storage.

    Single thread + sync (mirrors the shared helpers; ~zero work, <=10*NB cols)."""
    q = q_name
    self.gen_add_code_lines([
        "// mjx output: base-linear rows of " + mat + " <- R . rows (row-major nv x " + str(n_cols) + ")",
        "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
        *_gen_mjx_build_R_lines(q),
        "for (int c = 0; c < " + str(n_cols) + "; c++) {"
        " T m0 = " + mat + "[c], m1 = " + mat + "[" + str(n_cols) + " + c], m2 = " + mat + "[" + str(2 * n_cols) + " + c];"
        " " + mat + "[c] = R[0]*m0 + R[1]*m1 + R[2]*m2;"
        " " + mat + "[" + str(n_cols) + " + c] = R[3]*m0 + R[4]*m1 + R[5]*m2;"
        " " + mat + "[" + str(2 * n_cols) + " + c] = R[6]*m0 + R[7]*m1 + R[8]*m2; }",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def _emit_dI_times_v(self, dst_name, k, vec_name):
    """Emit `dst[0..5] = dI_k @ vec` as a fixed 6-line expression."""
    # accumulate per-row terms
    rows = {r: [] for r in range(6)}
    for (r, c, val) in _REGRESSOR_BASIS[k]:
        sign = "+" if val > 0 else "-"
        rows[r].append((sign, c))
    for r in range(6):
        terms = rows[r]
        if not terms:
            self.gen_add_code_line(dst_name + "[" + str(r) + "] = static_cast<T>(0);")
            continue
        expr = ""
        for (sign, c) in terms:
            expr += (" " + sign + " " if expr or sign == "-" else "") + vec_name + "[" + str(c) + "]"
        self.gen_add_code_line(dst_name + "[" + str(r) + "] = " + expr + ";")


def gen_inverse_dynamics_regressor_inner_temp_mem_size(self):
    n = self.robot.get_num_pos()
    # forward RNEA needs 6*n; the backward block sweep needs a small per-thread
    # staging of two 6-vectors (dIv, dIa) + a 6-vector crf product, but those are
    # thread-local registers, not shared. We only need the RNEA forward scratch
    # to be the max live footprint.
    return self.gen_inverse_dynamics_inner_temp_mem_size()


def gen_inverse_dynamics_regressor_inner_function_call(self, updated_var_names=None):
    var_names = dict(
        s_Y_name="s_Y",
        s_vaf_name="s_vaf",
        s_q_name="s_q",
        s_qd_name="s_qd",
        s_qdd_name="s_qdd",
        s_temp_name="s_temp",
        gravity_name="gravity",
    )
    if updated_var_names is not None:
        for key, value in updated_var_names.items():
            var_names[key] = value
    code_start = "inverse_dynamics_regressor_inner<T>(" + var_names["s_Y_name"] + ", " + \
        var_names["s_vaf_name"] + ", " + var_names["s_q_name"] + ", " + \
        var_names["s_qd_name"] + ", " + var_names["s_qdd_name"] + ", "
    code_middle = self.gen_insert_helpers_function_call()
    code_end = var_names["s_temp_name"] + ", " + var_names["gravity_name"] + ");"
    self.gen_add_code_line(code_start + code_middle + code_end)


def gen_inverse_dynamics_regressor_inner(self):
    n = self.robot.get_num_joints()
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    HAS_MIMIC = self.robot_has_mimic_joints()

    func_params = [
        "s_Y is the output joint-torque regressor, row-major nv x 10*NUM_BODIES = " + str(nv * 10 * NB),
        "s_vaf is scratch of size 18*NUM_JOINTS = " + str(18 * n) + " (RNEA v|a|f)",
        "s_q is the vector of joint positions",
        "s_qd is the vector of joint velocities",
        "s_qdd is the vector of joint accelerations",
        "s_temp is helper shared memory of size " + str(self.gen_inverse_dynamics_regressor_inner_temp_mem_size()),
        "gravity is the gravity constant",
    ]
    func_notes = [
        "Assumes the XI matricies have already been updated for the given q",
        "tau = Y . pi with pi_i = [m, m*c(3), I_O(6)=[Ixx,Ixy,Ixz,Iyy,Iyz,Izz]] per link",
    ]
    func_def_start = "void inverse_dynamics_regressor_inner(T *s_Y, T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, "
    func_def_end = "T *s_temp, const T gravity) {"
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(
        "", func_params, -1)
    func_def = func_def_start + func_def_middle + func_def_end

    self.gen_add_func_doc("Compute the joint-torque regressor Y (tau = Y . pi)",
                          func_notes, func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    # 1) RNEA forward sweep to populate s_vaf with per-link (v, a). We reuse the
    #    existing `inverse_dynamics_inner_vaf` (compute_c=False, use_qdd_input=True)
    #    which fills s_vaf = [v | a | f]; we only consume the v|a blocks.
    self.gen_add_code_line("// forward RNEA sweep: populate s_vaf v|a (reuse RNEA vaf inner)")
    self.gen_inverse_dynamics_inner_function_call(
        compute_c=False, use_qdd_input=True,
        updated_var_names=dict(d_f_ext_name="nullptr"))
    self.gen_add_sync()

    # 2) Zero the whole regressor.
    self.gen_add_code_line("// zero the regressor")
    self.gen_add_code_line("glass::set_const<T, " + str(nv * 10 * NB) + ">(static_cast<T>(0), s_Y);")

    # 3) Backward block sweep. One thread owns one (link i, basis-column k) pair
    #    -> a single 6-vector "force regressor column" that we propagate from
    #    link i up to the root, projecting onto each ancestor DOF along the way.
    #    Each thread writes a DISTINCT set of Y columns (column 10*i+k), so there
    #    are no inter-thread write collisions.
    #
    #    The body-regressor column f0 = dI_k a_i + crf(v_i) (dI_k v_i) is built
    #    from v_i,a_i (s_vaf), then walked up: for each ancestor j on the path
    #    i->root, project f onto j's DOF rows (mimic-scaled), then f = X[j]^T f.
    #
    # Precompute, per body i, the ancestor chain [i, parent(i), ...] until root.
    chains = [[i] + self.robot.get_ancestors_by_id(i) for i in range(NB)]

    # mimic multiplier + S projection tables, indexed by body id.
    def _mimic_scale(jid):
        return float(self._alpha_for_jid(jid)) if HAS_MIMIC else 1.0

    self.gen_add_code_line("// build Y: per (link, param) propagate the 6x10 body regressor up the tree")
    self.gen_add_parallel_loop("col", str(10 * NB))
    self.gen_add_code_line("int link_i = col / 10; int param_k = col % 10;")
    self.gen_add_code_line("T f[6]; T dIv[6]; T dIa[6]; T crfv[6];")
    # build f0 = dI_k a_i + crf(v_i) (dI_k v_i) via a per-link switch on link_i,
    # but the dI_k application is param-k dependent and uniform across links, so
    # we branch on param_k for the dI build and read v_i,a_i from s_vaf[6*link_i].
    self.gen_add_code_line("const T *v_i = &s_vaf[6*link_i];")
    self.gen_add_code_line("const T *a_i = &s_vaf[" + str(6 * n) + " + 6*link_i];")
    # dIv = dI_k @ v_i ; dIa = dI_k @ a_i  (switch on param_k)
    self.gen_add_code_line("switch (param_k) {", True)
    for k in range(10):
        self.gen_add_code_line("case " + str(k) + ": {", True)
        _emit_dI_times_v(self, "dIv", k, "v_i")
        _emit_dI_times_v(self, "dIa", k, "a_i")
        self.gen_add_code_line("break;")
        self.gen_add_end_control_flow()
    self.gen_add_code_line("default: { for (int r=0;r<6;r++){dIv[r]=static_cast<T>(0); dIa[r]=static_cast<T>(0);} }")
    self.gen_add_end_control_flow()
    # crfv = crf(v_i) @ dIv  (== fx_times_v(crfv, v_i, dIv))
    self.gen_add_code_line("fx_times_v<T>(crfv, v_i, dIv);")
    self.gen_add_code_line("for (int r=0;r<6;r++){ f[r] = dIa[r] + crfv[r]; }")

    # walk up the tree per link_i. Emit a switch on link_i; each case unrolls the
    # ancestor chain projections + X^T propagations.
    self.gen_add_code_line("switch (link_i) {", True)
    for i in range(NB):
        chain = chains[i]
        self.gen_add_code_line("case " + str(i) + ": {", True)
        for depth, j in enumerate(chain):
            scale = _mimic_scale(j)
            scale_pref = "" if scale == 1.0 else ("static_cast<T>(" + repr(scale) + ") * ")
            # project f onto body j's DOF rows
            if self.robot.floating_base and j == 0:
                # free-flyer root: S is identity over rows 0..5 -> Y[r, col] += f[r]
                inds_f = self.robot.get_joint_index_f(0)
                import numpy as _np
                S0 = _np.array(self.robot.get_S_by_id(0))
                for kcol in range(S0.shape[1]):
                    rows = _np.nonzero(S0[:, kcol])[0]
                    for row in rows:
                        sgn = float(S0[row, kcol])
                        sgn_pref = "" if sgn == 1.0 else ("static_cast<T>(" + repr(sgn) + ") * ")
                        fidx = inds_f[kcol]
                        self.gen_add_code_line(
                            "s_Y[" + str(fidx) + "*" + str(10 * NB) + " + col] += " +
                            scale_pref + sgn_pref + "f[" + str(int(row)) + "];")
            elif not self.robot.S_is_cardinal_by_id(j):
                # Tier B (skew): project f onto the dense S column: Y[fidx,col] +=
                # scale * (S^T f) = scale * sum_r S[r]*f[r].
                fidx = self.robot.get_joint_index_f(j)
                if isinstance(fidx, (list, tuple)):
                    fidx = fidx[0]
                S_vec = self.robot._get_flat_S_by_id(j)
                terms = [("static_cast<T>(" + repr(float(S_vec[r]) * scale) + ") * f[" + str(r) + "]")
                         for r in range(6) if S_vec[r] != 0.0]
                self.gen_add_code_line(
                    "s_Y[" + str(fidx) + "*" + str(10 * NB) + " + col] += " +
                    (" + ".join(terms) if terms else "static_cast<T>(0)") + ";")
            else:
                s_ind = self.robot.get_S_index_by_id(j)
                s_sign = self.robot.get_S_sign_by_id(j)
                coeff = float(s_sign) * scale
                coeff_pref = "" if coeff == 1.0 else ("static_cast<T>(" + repr(coeff) + ") * ")
                fidx = self.robot.get_joint_index_f(j)
                if isinstance(fidx, (list, tuple)):
                    fidx = fidx[0]
                self.gen_add_code_line(
                    "s_Y[" + str(fidx) + "*" + str(10 * NB) + " + col] += " +
                    coeff_pref + "f[" + str(s_ind) + "];")
            # propagate to parent: f = X[j]^T @ f  (unless j is the last on chain).
            # PER-THREAD matvec (each thread owns its own f); X is column-major
            # 6x6 at s_XImats[36*j], so (X^T f)[r] = sum_c X[r,c]*... = dot of
            # column r of X with f = sum_c s_XImats[36*j + 6*r + c] * f[c].
            # NB: must NOT use the block-cooperative grid_linalg_gemv here.
            if depth < len(chain) - 1:
                self.gen_add_code_line("{ T ft[6];")
                self.gen_add_code_line("  for (int r=0;r<6;r++){ T acc=static_cast<T>(0); for(int c=0;c<6;c++){ acc += s_XImats[" +
                                       str(36 * j) + " + 6*r + c] * f[c]; } ft[r]=acc; }")
                self.gen_add_code_line("  for (int r=0;r<6;r++){f[r]=ft[r];} }")
        self.gen_add_code_line("break;")
        self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()  # end switch
    self.gen_add_end_control_flow()  # end parallel loop
    self.gen_add_sync()
    self.gen_add_end_function()




def gen_inverse_dynamics_regressor_device(self):
    n = self.robot.get_num_pos()
    func_params = [
        "s_Y is the output regressor (nv x 10*NUM_BODIES)",
        "s_q is the vector of joint positions",
        "s_qd is the vector of joint velocities",
        "s_qdd is the vector of joint accelerations",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant",
    ]
    func_def = ("void inverse_dynamics_regressor_device(T *s_Y, const T *s_q, const T *s_qd, const T *s_qdd, "
                "const robotModel<T> *d_robotModel, const T gravity) {")
    shared_mem_size = self.gen_inverse_dynamics_regressor_inner_temp_mem_size()
    extra_t_buffers = [("s_vaf", 18 * n)]
    self.gen_device_wrapper(
        "Compute the joint-torque regressor Y (tau = Y . pi)", func_def,
        shared_mem_size,
        lambda: self.gen_inverse_dynamics_regressor_inner_function_call(),
        func_params=func_params,
        extra_t_buffers=extra_t_buffers, include_linalg_scratch=True)


def gen_inverse_dynamics_regressor_kernel(self, single_call_timing=False):
    # Floating-aware input layout (mirrors the idsva_so / id kernels):
    #   NUM_POS = get_num_pos()  (== quaternion-form NUM_JOINTS for floating base)
    #   nv      = get_num_vel()
    #   input block per timestep is THREE NUM_POS-WIDE SLOTS (stride Q_QD_U_STRIDE ==
    #   3*NUM_POS): q at 0, qd at NUM_POS, qdd at 2*NUM_POS. On a floating base qd/qdd
    #   carry nv values in their LEADING slots plus one trailing pad -- they are NOT
    #   packed tightly at nv width (see test_cuda_input_abi.py).
    # s_vaf is sized 18*NUM_POS to match the ID kernel convention (oversized but
    # consistent; the inner indexes it with n=get_num_joints()).
    NUM_POS = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    NB = self.robot.get_num_bodies()
    out_size = nv * 10 * NB
    in_size = 3 * NUM_POS
    func_params = [
        "d_Y is the output regressor, row-major nv x 10*NUM_BODIES = " + str(out_size),
        "d_q_qd_qdd is the vector of joint positions, velocities, accelerations (q|qd|qdd)",
        "stride_q_qd_qdd is the stride between each (q, qd, qdd) triple",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant",
        "num_timesteps is the length of the trajectory points",
    ]
    # g1-spill: the kernel takes d_workspace as its 2nd arg. At a spilled tier
    # (INVERSE_DYNAMICS_REGRESSOR_Y_IN_SMEM<TIER>()==false) the s_Y output (nv*10*NB,
    # ~277KB on h2_plus -> unlaunchable) lives in the L2-pinned d_workspace SO section
    # instead of smem, making the kernel launchable; at TIER_SHARED it stays in smem and
    # d_workspace is unused. Verbatim clone of the in-file fdpg s_Y spill.
    func_def_start = "void inverse_dynamics_regressor_kernel(T *d_Y, unsigned char *d_workspace, const T *d_q_qd_qdd, const int stride_q_qd_qdd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    # MUJOCO_OUTPUT (floating only): compile-time mjx output-convention flag. Added
    # LAST (after RESOURCE_TIER) so existing positional <T,TIER> call sites are
    # unaffected; the default (false) instantiation if-constexpr-elides the
    # epilogue -> byte-identical PTX. Y is a covector matrix (Y.pi = tau), so its
    # base-linear ROWS rotate like the ID torque covector (G.Y); inputs q,qd,qdd
    # are mjx -> convert in (qdd: regressor is the full id regressor with a qdd input).
    mjx_kernel = self.robot.floating_base
    self.gen_add_code_line("template <typename T, int RESOURCE_TIER = GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_func_doc("Compute the joint-torque regressor", [], func_params, None)
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line(func_def, True)
    # g1-spill: s_Y is the LAST t_buffer (tail-carve); its arena slot is sized out_size
    # at an in-smem tier and 0 when spilled (the real s_Y is then routed to d_workspace),
    # shrinking the smem footprint exactly to match the tiered DYNAMIC_SHARED_MEM_BYTES.
    self.gen_add_code_line("constexpr bool REGRESSOR_Y_IN_SMEM = INVERSE_DYNAMICS_REGRESSOR_Y_IN_SMEM<RESOURCE_TIER>();")
    self.gen_add_code_line("constexpr int REGRESSOR_Y_SLOT = REGRESSOR_Y_IN_SMEM ? " + str(out_size) + " : 0;")
    extra_t_buffers = [("s_q_qd_qdd", in_size), ("s_vaf", 18 * NUM_POS), ("s_Y", "REGRESSOR_Y_SLOT")]
    shared_mem_size = self.gen_inverse_dynamics_regressor_inner_temp_mem_size()
    self.gen_XImats_helpers_temp_shared_memory_code(
        shared_mem_size, extra_t_buffers=extra_t_buffers, include_linalg_scratch=True)
    self.gen_add_code_line("if constexpr (REGRESSOR_Y_IN_SMEM) { (void)d_workspace; }")
    self.gen_add_code_line("T *s_q = s_q_qd_qdd; T *s_qd = &s_q_qd_qdd[" + str(NUM_POS) +
                           "]; T *s_qdd = &s_q_qd_qdd[" + str(2 * NUM_POS) + "];")
    def _repoint_spilled_Y(in_timestep_loop):
        # When spilled, repoint s_Y at the L2-pinned d_workspace SO section (per-timestep
        # slot; the regressor never runs concurrently with the SO kernels). Emitted where
        # `k` is in scope for the batched path. Verbatim clone of fdpg's _repoint_spilled_Y.
        self.gen_add_code_line("if constexpr (!REGRESSOR_Y_IN_SMEM) {", True)
        self.gen_add_code_line(gen_workspace_repoint_line("s_Y", "GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()", batch_indexed=in_timestep_loop))
        self.gen_add_end_control_flow()
    if not single_call_timing:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_kernel_load_inputs("q_qd_qdd", str(in_size), stride="stride_q_qd_qdd")
        # mjx input convert (before XImats so X[0] uses the reordered quaternion)
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            self.gen_mjx_input_convert(qdd_name="s_qdd")
            self.gen_add_end_control_flow()
        _repoint_spilled_Y(in_timestep_loop=True)
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_inverse_dynamics_regressor_inner_function_call()
        self.gen_add_sync()
        # mjx output: Y.pi = tau is a covector -> base-linear ROWS rotate by R
        if mjx_kernel:
            self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
            _emit_mjx_base_rotate_rows_rowmajor(self, "s_Y", nv, 10 * NB)
            self.gen_add_end_control_flow()
        self.gen_kernel_save_result("Y", str(out_size), stride=str(out_size))
        self.gen_add_end_control_flow()
    else:
        self.gen_kernel_load_inputs("q_qd_qdd", str(in_size))
        _repoint_spilled_Y(in_timestep_loop=False)
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_load_update_XImats_helpers_function_call()
        self.gen_inverse_dynamics_regressor_inner_function_call()
        self.gen_add_sync()
        self.gen_add_end_control_flow()
        self.gen_kernel_save_result("Y", str(out_size))
    self.gen_add_end_function()


def gen_inverse_dynamics_regressor_host(self, mode=0):
    single_call_timing, compute_only = host_mode_flags(mode)
    n = self.robot.get_num_pos()
    NB = self.robot.get_num_bodies()
    nv = self.robot.get_num_vel()
    out_size = nv * 10 * NB
    func_params = [
        "hd_data is the packaged input and output pointers (q/qd/qdd inputs; output regressor written to hd_data->d_Y, 10*NB*nv*num_timesteps floats)",
        "d_robotModel is the pointer to the initialized model specific helpers on the GPU",
        "gravity is the gravity constant",
        "num_timesteps is the length of the trajectory points",
        "streams are pointers to CUDA streams for async memory transfers",
    ]
    func_def_start = "void inverse_dynamics_regressor(gridData<T, KIND> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end = "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    func_def_start, func_def_end = mangle_host_func_defs(func_def_start, func_def_end, single_call_timing, compute_only)
    # MUJOCO_OUTPUT (floating only) host template flag, forwarded to the kernel
    # launch (names RESOURCE_TIER positionally to reach the trailing flag). Added
    # LAST so existing positional call sites don't rebind.
    self.gen_add_func_doc("Compute the joint-torque regressor Y (tau = Y . pi)", [], func_params, None)
    mjx_host = gen_host_wrapper_head(self, "inverse_dynamics_regressor", func_def_start, func_def_end, kind_rule="dynamics")
    kernel_tmpl = "inverse_dynamics_regressor_kernel<T, RESOURCE_TIER, MUJOCO_OUTPUT>" if mjx_host else "inverse_dynamics_regressor_kernel<T, RESOURCE_TIER>"
    func_call_start = kernel_tmpl + "<<<block_dimms,thread_dimms,INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()>>>(hd_data->d_Y,hd_data->d_workspace,hd_data->d_q_qd_u,stride_q_qd_qdd,"
    func_call_end = "d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("inverse_dynamics_regressor_kernel<", "inverse_dynamics_regressor_kernel_single_timing<")
    # q|qd|qdd block layout (floating-aware): stride is the standard Q_QD_U_STRIDE
    # (== NUM_POS + 2*NUM_VEL). The host reuses the d_q_qd_u buffer for q|qd|qdd.
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
    # g1-spill: L2-pin d_workspace when the tier spills s_Y into it.
    if not single_call_timing:
        self.gen_add_workspace_slot_count()
    ws_bytes = ("GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()" if single_call_timing
                else "GRID_WORKSPACE_BYTES_PER_TIMESTEP<T>()*static_cast<size_t>(_grid_ws_n)")
    self.gen_add_code_line("if (!INVERSE_DYNAMICS_REGRESSOR_Y_IN_SMEM<RESOURCE_TIER>() && hd_data->d_workspace != nullptr) {gpuErrchk(grid_begin_l2_persisting(0, hd_data->d_workspace, " + ws_bytes + "));}")
    self.gen_add_code_line("gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"inverse_dynamics_regressor\", INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T, RESOURCE_TIER>()));")
    if single_call_timing:
        self.gen_add_code_lines(func_call_code)
    else:
        self.gen_add_workspace_clamped_launch(func_call_code, emit_count = False)
    if not compute_only:
        self.gen_add_code_lines([
            "// finally transfer the result back into the gridData host buffer (hd_data->d_Y -> hd_data->h_Y)",
            "gpuErrchk(cudaMemcpy(hd_data->h_Y,hd_data->d_Y," +
            ("num_timesteps*" if not single_call_timing else "") + str(out_size) + "*sizeof(T),cudaMemcpyDeviceToHost));",
            "gpuErrchkKernel();",
        ])
    else:
        self.gen_add_code_line("gpuErrchkKernel();")
    if single_call_timing:
        from ..algo_registry import single_call_printf_line
        self.gen_add_code_line(single_call_printf_line("inverse_dynamics_regressor"))
    self.gen_add_end_function()


def gen_inverse_dynamics_regressor(self):
    # inner -> device -> kernel(s) -> host(s)
    self.gen_inverse_dynamics_regressor_inner()
    self.gen_inverse_dynamics_regressor_device()
    self.gen_inverse_dynamics_regressor_kernel(single_call_timing=False)
    self.gen_inverse_dynamics_regressor_kernel(single_call_timing=True)
    self.gen_inverse_dynamics_regressor_host(0)
    self.gen_inverse_dynamics_regressor_host(1)
    self.gen_inverse_dynamics_regressor_host(2)

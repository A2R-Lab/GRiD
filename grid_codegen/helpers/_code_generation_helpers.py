import os

# GRID_NO_LICM_BARRIER=1 suppresses the anti-LICM machinery (volatile input
# reload, rep-stomp, output->input feedback, syncthreads-protected output write)
# emitted by gen_anti_licm_input_reload + gen_anti_licm_output_write. Use this
# when ptxas hangs on floating-base kernels on older toolkits/GPUs (we've seen
# this on sm_86 with stock CUDA). Batch timings (N=16..256) are unaffected;
# single-call timings on _single_timing kernels may LICM-elide and report
# sub-microsecond nonsense values for ABA/FD/MINV/CRBA/ID/EE_POSE.
# Read at call time, not import time, so CLI flags in grid/run.py can set it
# after this module is already imported.
def _no_licm_barrier() -> bool:
    return os.environ.get("GRID_NO_LICM_BARRIER", "0") == "1"


def robot_has_mimic_joints(self):
    """True if the model has any URDF ``<mimic>`` joint.

    This is the master gate for every mimic-aware codegen branch: when False,
    every emit site below MUST produce character-identical CUDA to the legacy
    (pre-mimic) codegen, so non-mimic robots' generated headers are byte-
    identical (=> identical PTX => no perf regression). When True, the codegen
    folds mimic joints into their target's velocity slot exactly as the
    mimic-aware RBDReference does.
    """
    return any(getattr(j, "is_mimic", False) for j in self.robot.joints)

def _v_slot_cpp(self, jid):
    """Return the (single) reduced velocity-space slot index for body ``jid``.

    For a non-mimic joint this is its own dense v-offset, which equals ``jid``
    for fixed-base single-DoF chains and ``jid + 5`` for floating-base ones —
    i.e. byte-identical to the legacy hardcoded index. For a mimic joint it is
    the mimicked target's v-slot (mimic + target share one generalized
    coordinate). Asserts single-DoF: multi-DoF mimic joints don't exist in the
    URDF spec, and the floating root (the only multi-DoF joint) is never mimic.
    """
    inds = self.robot.get_joint_index_v(jid)
    if isinstance(inds, (list, tuple)):
        assert len(inds) == 1, (
            "_v_slot_cpp assumes a single-DoF joint; got multi-DoF v-index "
            + str(inds) + " for jid " + str(jid)
        )
        return inds[0]
    return inds

def _alpha_for_jid(self, jid):
    """Return body ``jid``'s mimic multiplier (1.0 for non-mimic joints).

    Mirrors RBDReference._mimic_multiplier: a mimic joint's generalized
    velocity is ``multiplier * v_target``, so every ``qd``/``qdd`` read and
    every ``c``/``M`` write for the body is scaled by this factor.
    """
    j = self.robot.get_joint_by_id(jid)
    if getattr(j, "is_mimic", False):
        return float(j.get_mimic_multiplier())
    return 1.0

def _id_S_desc(self, jid):
    """Classify body ``jid``'s 1-DoF motion subspace for codegen dispatch.

    Returns one of:
      ("A", s_ind, s_sign) -- Tier A cardinal axis (single signed unit index);
                              the byte-identical fast path every current robot
                              takes.
      ("B", S_vec)         -- Tier B skew/general axis (dense 6-vector column,
                              >=2 nonzero entries). Consumers emit dense
                              dot/axpy/GEMV ops instead of an indexed scale.

    Only single-column (1-DoF) joints are classified here; multi-column
    planar/spherical (Tier C) is rejected upstream by _assert_single_axis_S.
    """
    if self.robot.S_is_cardinal_by_id(jid):
        return ("A", self.robot.get_S_index_by_id(jid), self.robot.get_S_sign_by_id(jid))
    return ("B", [float(v) for v in self.robot._get_flat_S_by_id(jid)])

def _alpha_prefix_cpp(self, jid):
    """Return a C++ multiplicative prefix ``"<alpha> * "`` for body ``jid``,
    or the empty string when alpha == 1.0 (non-mimic). Used so non-mimic emit
    is byte-identical (no spurious ``1.0 *``)."""
    alpha = self._alpha_for_jid(jid)
    if alpha == 1.0:
        return ""
    return "static_cast<T>(" + repr(alpha) + ") * "

def gen_add_code_line(self, new_code_line, add_indent_after = False):
    self.code_str += self.indent_level * "    " + new_code_line + "\n"
    if add_indent_after:
        self.indent_level += 1

def gen_add_code_lines(self, new_code_lines, add_indent_after = False):
    """Emit a list of code lines.

    Items are emitted in order via `gen_add_code_line`. A literal `True`
    immediately following a line is consumed and treated as that line's
    `add_indent_after` flag (i.e., opens a new indented block). This lets
    callers write `[\"if (cond) {\", True, ...]` inline rather than splitting
    out a separate `gen_add_code_line(line, True)` call.

    The trailing `add_indent_after` parameter still controls whether the
    overall block indents one more level after all lines are emitted.
    """
    i = 0
    while i < len(new_code_lines):
        line = new_code_lines[i]
        opens = (i + 1 < len(new_code_lines)) and (new_code_lines[i + 1] is True)
        if opens:
            self.gen_add_code_line(line, True)
            i += 2
        else:
            self.gen_add_code_line(line)
            i += 1
    if add_indent_after:
        self.indent_level += 1

def gen_add_end_control_flow(self):
    self.indent_level -= 1
    self.gen_add_code_line("}")

def gen_add_end_function(self):
    self.indent_level -= 1
    self.gen_add_code_line("}\n")

def gen_add_func_doc(self, func_desc, notes = [], params = [], return_val = None):
    self.gen_add_code_line("/**")
    self.gen_add_code_line(" * " + func_desc)
    self.gen_add_code_line(" *")
    if len(notes) > 0:
        self.gen_add_code_line(" * Notes:")
        for note in notes:
            self.gen_add_code_line(" *   " + note)
        self.gen_add_code_line(" *")
    for param in params:
        self.gen_add_code_line(" * @param " + param)
    if return_val is not None:
        self.gen_add_code_line(" * @return " + return_val)
    self.gen_add_code_line(" */")

def gen_add_serial_ops(self):
    self.gen_add_code_line("if(threadIdx.x == 0 && threadIdx.y == 0){", True)

def gen_add_parallel_loop(self, var_name, max_val, block_level = False):
    if block_level:
        code = "for(int " + var_name + " = blockIdx.x + blockIdx.y*gridDim.x; " + \
                    var_name + " < " + max_val + "; " + var_name + " += gridDim.x*gridDim.y){"
    else:
        code = "for(int " + var_name + " = threadIdx.x + threadIdx.y*blockDim.x; " + \
                    var_name + " < " + max_val + "; " + var_name + " += blockDim.x*blockDim.y){"
    self.gen_add_code_line(code, True)

def gen_minv_apply(self, n, out_name, rhs_expr, loop_var = "row", loop_max = None,
                   pre_lines = None, comment_in_loop = True, negate = False):
    """Emit `out = (+/-) Minv @ rhs` over a parallel loop, exploiting that Minv
    is stored SYMMETRIC_UPPER (only the upper triangle is materialized; the
    lower triangle is read transposed via the `(row<=col)`/`(row>col)` index).

    Shared core of forward_dynamics_finish (`qdd = Minv*(u-c)`, n*1, not negated)
    and forward_dynamics_gradient's df/du finish (`df_du = -Minv*dc_du`, n*2n,
    negated). The reduction over `col`, the symmetric-upper `int index = ...`
    line, and the `val +=` accumulation are identical; the call differs only in
    the loop bound, the per-iteration index decode (`pre_lines`), the rhs slice
    expression, the output target, and the sign — all parameters here.

    Parameters
    ----------
    n            : reduced velocity dim (matrix is n x n).
    out_name     : C++ lvalue (indexed by `loop_var`) receiving the result.
    rhs_expr     : C++ expression for the rhs column `col` (a function of `col`).
    loop_var     : parallel-loop induction var (default "row").
    loop_max     : parallel-loop bound expr (default str(n) => one column).
    pre_lines    : extra C++ lines emitted at the top of the loop body, before
                   `T val` (e.g. row/offset decode for the n x 2n case).
    comment_in_loop : True  -> the SYMMETRIC_UPPER comment sits inside the `col`
                              loop, just above `int index` (forward_dynamics).
                      False -> the comment sits in the loop body before `T val`
                              (forward_dynamics_gradient).
    negate       : negate the accumulated result on the output write.
    """
    if loop_max is None:
        loop_max = str(n)
    comment = "// account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix"
    index_line = "int index = (row <= col) * (col * " + str(n) + " + row) + (row > col) * (row * " + str(n) + " + col);"
    self.gen_add_parallel_loop(loop_var, loop_max)
    if pre_lines is not None:
        for line in pre_lines:
            self.gen_add_code_line(line)
    if not comment_in_loop:
        self.gen_add_code_line(comment)
    self.gen_add_code_line("T val = static_cast<T>(0);")
    self.gen_add_code_line("for(int col = 0; col < " + str(n) + "; col++) {", True)
    if comment_in_loop:
        self.gen_add_code_line(comment)
    self.gen_add_code_line(index_line)
    self.gen_add_code_line("val += s_Minv[index] * " + rhs_expr + ";")
    self.gen_add_end_control_flow()
    self.gen_add_code_line(out_name + " = " + ("-val;" if negate else "val;"))
    self.gen_add_end_control_flow()

def gen_static_array_ind_2d(self, col, row, col_stride = 6):
    return col_stride*col + row

def gen_static_array_ind_3d(self, ind, col, row, ind_stride = 36, col_stride = 6):
    return ind_stride*ind + col_stride*col + row

def gen_add_sync(self):
    self.gen_add_code_line("__syncthreads();")


# ---------------------------------------------------------------------------
# MuJoCo / mjx output-convention emit helpers (shared across floating algorithms)
# ---------------------------------------------------------------------------
#
# These emit the small base-block corner math that converts a GRiD/pinocchio
# floating-base result to the MuJoCo/mjx convention (quaternion order wxyz<->xyzw
# and the free-joint base-LINEAR velocity frame G = blockdiag(R, I); R = base
# orientation). They are the CUDA mirror of `RBDReference/equivalents/
# mujoco_convention.py` (the validated oracle). Each is emitted ONLY when the
# robot is floating AND inside a compile-time `if constexpr (MUJOCO_OUTPUT)` block,
# so the default pin PTX is byte-identical and fixed-base headers never contain it.
#
# Design: every transform runs on a single thread over the <=6x6 base block (the
# work is trivially small), building the 3x3 rotation `R` from the configuration
# quaternion in registers (so NO new shared memory is needed and we never depend
# on the internal s_XImats[0] spatial-transform layout, which CRBA leaves stale).
# Bracket with `gen_add_sync()` so the rest of the block sees the converted data.
# Matrices are COLUMN-MAJOR: `mat[r + n_rows*c]` is element (row r, col c).
#
# `q_name` defaults to the per-kernel configuration buffer (`s_q`), whose layout
# is [pos(3), quat_xyzw(4), joints]; the tangent buffers (`s_qd`/`s_qdd`/`s_c`/...)
# are [lin(3), ang(3), joints]. Pass custom names for kernels that buffer elsewhere.

def _gen_mjx_build_R_lines(q_name="s_q"):
    """Emit (as a list) the register declaration of the 3x3 row-major rotation
    ``R`` (``R[3*i+j]``) from the xyzw quaternion at ``q_name[3..6]`` -- matching
    ``mujoco_convention.rotation_from_quat_xyzw`` exactly. Caller must already be
    on a single thread (these are plain register writes)."""
    q = q_name
    return [
        f"T qx = {q}[3], qy = {q}[4], qz = {q}[5], qw = {q}[6];",
        "T xx = qx*qx, yy = qy*qy, zz = qz*qz;",
        "T xy = qx*qy, xz = qx*qz, yz = qy*qz, wx = qw*qx, wy = qw*qy, wz = qw*qz;",
        "T R[9];",
        "R[0] = static_cast<T>(1) - static_cast<T>(2)*(yy+zz); R[1] = static_cast<T>(2)*(xy-wz);                    R[2] = static_cast<T>(2)*(xz+wy);",
        "R[3] = static_cast<T>(2)*(xy+wz);                    R[4] = static_cast<T>(1) - static_cast<T>(2)*(xx+zz); R[5] = static_cast<T>(2)*(yz-wx);",
        "R[6] = static_cast<T>(2)*(xz-wy);                    R[7] = static_cast<T>(2)*(yz+wx);                    R[8] = static_cast<T>(1) - static_cast<T>(2)*(xx+yy);",
    ]


def gen_mjx_input_convert(self, q_name="s_q", qd_name="s_qd", qdd_name=None, u_name=None):
    """Convert the mjx-frame INPUTS to the pin frame, in place, before the kernel
    body runs. Emitted right after `gen_kernel_load_inputs` + sync, BEFORE the
    XImats build (so X[0] is built from the correctly-ordered quaternion).

    Steps (single thread): (1) reorder the base quaternion wxyz->xyzw in
    ``q_name[3..6]``; (2) build R; (3) ``qd[0:3] = R^T qd[0:3]`` (mjx global base
    velocity -> pin local); (4) if ``qdd_name``: ``qdd[0:3] = R^T qdd[0:3] -
    omega x v_local`` (the acceleration is NOT a plain rotation -- see the oracle);
    (5) if ``u_name``: ``u[0:3] = R^T u[0:3]`` (covector force). Steps 3-5 use the
    base-angular block (``qd[3:6]``, frame-invariant) and the just-converted local
    linear velocity."""
    self.gen_add_code_lines([
        "// mjx input convert: quaternion wxyz->xyzw, base-linear velocity/accel/force -> pin frame",
        "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
        # quaternion wxyz (mjx, scalar-first) -> xyzw (pin, scalar-last)
        f"T qw_in = {q_name}[3];",
        f"{q_name}[3] = {q_name}[4]; {q_name}[4] = {q_name}[5]; {q_name}[5] = {q_name}[6]; {q_name}[6] = qw_in;",
    ])
    self.gen_add_code_lines(_gen_mjx_build_R_lines(q_name))
    self.gen_add_code_lines([
        # v_pin_lin = R^T v_mjx_lin
        f"T vlx = {qd_name}[0], vly = {qd_name}[1], vlz = {qd_name}[2];",
        f"{qd_name}[0] = R[0]*vlx + R[3]*vly + R[6]*vlz;",
        f"{qd_name}[1] = R[1]*vlx + R[4]*vly + R[7]*vlz;",
        f"{qd_name}[2] = R[2]*vlx + R[5]*vly + R[8]*vlz;",
    ])
    if qdd_name is not None:
        # a_pin_lin = R^T a_mjx_lin - omega x v_local  (v_local = converted qd[0:3])
        self.gen_add_code_lines([
            f"T alx = {qdd_name}[0], aly = {qdd_name}[1], alz = {qdd_name}[2];",
            f"T wx_ = {qd_name}[3], wy_ = {qd_name}[4], wz_ = {qd_name}[5];",
            f"T vx_ = {qd_name}[0], vy_ = {qd_name}[1], vz_ = {qd_name}[2];",
            f"{qdd_name}[0] = (R[0]*alx + R[3]*aly + R[6]*alz) - (wy_*vz_ - wz_*vy_);",
            f"{qdd_name}[1] = (R[1]*alx + R[4]*aly + R[7]*alz) - (wz_*vx_ - wx_*vz_);",
            f"{qdd_name}[2] = (R[2]*alx + R[5]*aly + R[8]*alz) - (wx_*vy_ - wy_*vx_);",
        ])
    if u_name is not None:
        self.gen_add_code_lines([
            f"T ufx = {u_name}[0], ufy = {u_name}[1], ufz = {u_name}[2];",
            f"{u_name}[0] = R[0]*ufx + R[3]*ufy + R[6]*ufz;",
            f"{u_name}[1] = R[1]*ufx + R[4]*ufy + R[7]*ufz;",
            f"{u_name}[2] = R[2]*ufx + R[5]*ufy + R[8]*ufz;",
        ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def gen_mjx_quat_reorder(self, q_name="s_q"):
    """Reorder ONLY the base quaternion mjx wxyz (scalar-first) -> pin xyzw
    (scalar-last) in place at ``q_name[3..6]``, then sync. This is the q-only
    subset of :func:`gen_mjx_input_convert` (no velocity/accel/force conversion) for
    functions that read just ``q`` (crba, minv, com, end_effector_pose,
    generalized_gravity): the value is base-orientation-independent in body frame,
    so only the OUTPUT epilogue's R (built from the xyzw quaternion) needs it.
    Emitted right after `gen_kernel_load_inputs` + sync, BEFORE the XImats build so
    a non-`skip_floating_base_X` kernel builds X[0] from the correct quaternion."""
    self.gen_add_code_lines([
        "// mjx input convert: base quaternion wxyz->xyzw (q-only; value frame-independent)",
        "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
        f"T qw_in = {q_name}[3];",
        f"{q_name}[3] = {q_name}[4]; {q_name}[4] = {q_name}[5]; {q_name}[5] = {q_name}[6]; {q_name}[6] = qw_in;",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def gen_mjx_base_rotate(self, buf, q_name="s_q"):
    """Covector/contravector OUTPUT row map ``out[0:3] = R out[0:3]`` (the
    ``base_rotate`` family: generalized_gravity, inverse_dynamics tau, and -- via
    the matrix overload, see :func:`gen_mjx_base_rotate_rows` -- the regressors).
    Single thread + sync."""
    self.gen_add_code_lines([
        f"// mjx output: base-linear rows of {buf} <- R * rows",
        "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
    ])
    self.gen_add_code_lines(_gen_mjx_build_R_lines(q_name))
    self.gen_add_code_lines([
        f"T b0 = {buf}[0], b1 = {buf}[1], b2 = {buf}[2];",
        f"{buf}[0] = R[0]*b0 + R[1]*b1 + R[2]*b2;",
        f"{buf}[1] = R[3]*b0 + R[4]*b1 + R[5]*b2;",
        f"{buf}[2] = R[6]*b0 + R[7]*b1 + R[8]*b2;",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def gen_mjx_base_rotate_rows(self, mat, n_rows, n_cols, q_name="s_q"):
    """Base-linear ROW rotation ``rows0:3 <- R . rows`` for an ``n_rows x n_cols``
    COLUMN-MAJOR matrix (the row-step of :func:`gen_mjx_congruence`, standalone). For
    matrix covector outputs whose base-linear rows transform like the ID torque:
    inverse_dynamics_regressor (``G.Y``) and forward_dynamics_parameter_gradient.
    Single thread + sync."""
    self.gen_add_code_lines([
        f"// mjx output: base-linear rows of {mat} <- R . rows (all {n_cols} cols)",
        "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
    ])
    self.gen_add_code_lines(_gen_mjx_build_R_lines(q_name))
    self.gen_add_code_lines([
        f"for (int c = 0; c < {n_cols}; c++) {{ T m0 = {mat}[0 + {n_rows}*c], m1 = {mat}[1 + {n_rows}*c], m2 = {mat}[2 + {n_rows}*c];"
        f" {mat}[0 + {n_rows}*c] = R[0]*m0 + R[1]*m1 + R[2]*m2; {mat}[1 + {n_rows}*c] = R[3]*m0 + R[4]*m1 + R[5]*m2; {mat}[2 + {n_rows}*c] = R[6]*m0 + R[7]*m1 + R[8]*m2; }}",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def gen_mjx_symmetrize_full(self, mat, n):
    """Fully populate a SYMMETRIC_UPPER-stored ``n x n`` COLUMN-MAJOR matrix by
    mirroring the upper triangle into the lower (``mat[r,c]=mat[c,r]`` for ``r>c``),
    so a subsequent :func:`gen_mjx_congruence` reads complete base rows/cols. Used by
    minv (stored upper-triangular). After the congruence the result is fully
    symmetric, so the host symmetrize step becomes a no-op. Single thread + sync.
    NOTE: assumes the UPPER triangle (``mat[i + n*j]`` for ``i<=j``) holds the data;
    if the kernel stores the LOWER triangle instead, swap the copy direction."""
    self.gen_add_code_lines([
        f"// mjx: mirror upper->lower of SYMMETRIC_UPPER {mat} before the congruence",
        "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
        f"for (int c = 0; c < {n}; c++) {{ for (int r = c + 1; r < {n}; r++) {{ {mat}[r + {n}*c] = {mat}[c + {n}*r]; }} }}",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def gen_mjx_accel_out(self, qdd_buf, qd_buf, q_name="s_q"):
    """Forward-dynamics acceleration OUTPUT pin->mjx:
    ``qdd[0:3] = R (qdd[0:3] + omega x v_local)`` where ``omega = qd[3:6]`` and
    ``v_local = qd[0:3]`` are the PIN-frame velocity (the kernel's own buffers,
    pre-output). The ``omega x v`` term is what makes the acceleration not a plain
    rotation. Single thread + sync."""
    self.gen_add_code_lines([
        f"// mjx output: forward-dynamics accel {qdd_buf}[0:3] <- R (qdd + omega x v_local)",
        "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
    ])
    self.gen_add_code_lines(_gen_mjx_build_R_lines(q_name))
    self.gen_add_code_lines([
        f"T wx_ = {qd_buf}[3], wy_ = {qd_buf}[4], wz_ = {qd_buf}[5];",
        f"T vx_ = {qd_buf}[0], vy_ = {qd_buf}[1], vz_ = {qd_buf}[2];",
        f"T c0 = {qdd_buf}[0] + (wy_*vz_ - wz_*vy_);",
        f"T c1 = {qdd_buf}[1] + (wz_*vx_ - wx_*vz_);",
        f"T c2 = {qdd_buf}[2] + (wx_*vy_ - wy_*vx_);",
        f"{qdd_buf}[0] = R[0]*c0 + R[1]*c1 + R[2]*c2;",
        f"{qdd_buf}[1] = R[3]*c0 + R[4]*c1 + R[5]*c2;",
        f"{qdd_buf}[2] = R[6]*c0 + R[7]*c1 + R[8]*c2;",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def gen_mjx_congruence(self, mat, n, q_name="s_q"):
    """Congruence / similarity ``X_mjx = G X G^T`` for an ``n x n`` COLUMN-MAJOR
    matrix (mass matrix, Minv, coriolis): base-linear ROWS 0:3 <- R . rows, then
    base-linear COLS 0:3 <- cols . R^T (the corner becomes ``R X00 R^T``). Works
    for non-symmetric C (it is a true similarity, identical code). Single thread +
    sync. NOTE for SYMMETRIC_UPPER storage (Minv): re-mirror the base block after."""
    self.gen_add_code_lines([
        f"// mjx output: congruence G {mat} G^T (base rows then base cols)",
        "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
    ])
    self.gen_add_code_lines(_gen_mjx_build_R_lines(q_name))
    self.gen_add_code_lines([
        # rows 0:3 <- R . rows, for every column c
        f"for (int c = 0; c < {n}; c++) {{ T m0 = {mat}[0 + {n}*c], m1 = {mat}[1 + {n}*c], m2 = {mat}[2 + {n}*c];"
        f" {mat}[0 + {n}*c] = R[0]*m0 + R[1]*m1 + R[2]*m2; {mat}[1 + {n}*c] = R[3]*m0 + R[4]*m1 + R[5]*m2; {mat}[2 + {n}*c] = R[6]*m0 + R[7]*m1 + R[8]*m2; }}",
        # cols 0:3 <- cols . R^T, for every row r:  newcol[j] = sum_k M[r,k] R[j][k]
        f"for (int r = 0; r < {n}; r++) {{ T m0 = {mat}[r + {n}*0], m1 = {mat}[r + {n}*1], m2 = {mat}[r + {n}*2];"
        f" {mat}[r + {n}*0] = m0*R[0] + m1*R[1] + m2*R[2]; {mat}[r + {n}*1] = m0*R[3] + m1*R[4] + m2*R[5]; {mat}[r + {n}*2] = m0*R[6] + m1*R[7] + m2*R[8]; }}",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def gen_mjx_column_reframe(self, mat, n_rows, n_cols, q_name="s_q"):
    """Jacobian column reframe ``J_mjx = J G^{-1}`` for an ``n_rows x n_cols``
    COLUMN-MAJOR matrix: base-linear COLS 0:3 <- cols . R^T (right-multiply only;
    the output rows are frame-invariant). Covers frame_jacobian/_dot, jacobian_com,
    the CCRBA matrix A, cmm_time_variation, end_effector_pose_gradient and dh/dqd.
    Single thread + sync."""
    self.gen_add_code_lines([
        f"// mjx output: column reframe {mat} G^{{-1}} (base-linear cols . R^T)",
        "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
    ])
    self.gen_add_code_lines(_gen_mjx_build_R_lines(q_name))
    self.gen_add_code_lines([
        f"for (int r = 0; r < {n_rows}; r++) {{ T j0 = {mat}[r + {n_rows}*0], j1 = {mat}[r + {n_rows}*1], j2 = {mat}[r + {n_rows}*2];"
        f" {mat}[r + {n_rows}*0] = j0*R[0] + j1*R[1] + j2*R[2]; {mat}[r + {n_rows}*1] = j0*R[3] + j1*R[4] + j2*R[5]; {mat}[r + {n_rows}*2] = j0*R[6] + j1*R[7] + j2*R[8]; }}",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()


def gen_mjx_retract(self, q_out, q_in, qd, dt_expr, q_name=None):
    """mjx free-joint retract for the integrator: the base POSITION takes a GLOBAL
    additive step ``q_out[0:3] = q_in[0:3] + dt * qd[0:3]`` (vs pin's SE(3) V(phi)
    coupling, which is O(dt^2) wrong for MuJoCo). The base quaternion and all
    internal joints integrate exactly as pin -- the caller emits those normally and
    this helper OVERWRITES only the base-linear position block. ``qd[0:3]`` is the
    mjx (global) base-linear velocity. Single thread + sync."""
    self.gen_add_code_lines([
        f"// mjx retract: base position global additive step (q_out[0:3] = q_in[0:3] + dt*qd[0:3])",
        "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
        f"{q_out}[0] = {q_in}[0] + ({dt_expr}) * {qd}[0];",
        f"{q_out}[1] = {q_in}[1] + ({dt_expr}) * {qd}[1];",
        f"{q_out}[2] = {q_in}[2] + ({dt_expr}) * {qd}[2];",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()

def gen_add_debug_print_code_lines(self, print_code_string_arr):
    self.gen_add_sync()
    self.gen_add_serial_ops()
    for code_string in print_code_string_arr:
        self.gen_add_code_line(code_string)
    self.gen_add_end_control_flow()
    self.gen_add_sync()

def gen_var_in_list(self, var_name, option_list):
    if len(option_list) == 1:
        return "(" + var_name + " == " + option_list[0] + ")"
    else:
        return "(" + " || ".join(["(" + var_name + " == " + option + ")" for option in option_list]) + ")"

def gen_var_not_in_list(self, var_name, option_list):
    if len(option_list) == 1:
        return "(" + var_name + " != " + option_list[0] + ")"
    else:
        return "(" + " && ".join(["(" + var_name + " != " + option + ")" for option in option_list]) + ")"

def gen_add_multi_threaded_select(self, loop_counter, comparator, counts, select_tuples, USE_NON_BRANCH_ALWAYS = False):
    # first find the resulting type and variable name
    dst_code = []
    for (dst_type, dst_var, select_list) in select_tuples:
        if dst_type is None:
            dst_code.append(dst_var)
        elif "|" in dst_type:
            dst_type_parts = dst_type.split("|")
            dst_code.append(dst_type_parts[0] + dst_var + ")" + dst_type_parts[1])
        else:
            dst_code.append(dst_type + " " + dst_var)
    # then if many things to select gen it and branch
    if len(select_tuples) > 1 and not USE_NON_BRANCH_ALWAYS:
        self.gen_add_code_line("// branch to get pointer locations")
        # init pointers outside of select
        self.gen_add_code_line("; ".join(dst_code)  + ";")
        # if / else if / else to select pointers
        n = len(counts)
        code_end = "}"
        for ind in range(n):
            if ind == 0:
                code_start = "     if (" + loop_counter + " " + comparator + " " + counts[ind] + "){ "
            elif ind < n-1:
                code_start = "else if (" + loop_counter + " " + comparator + " " + counts[ind] + "){ "
            else:
                code_start = "else              { "
            code_middle = ""
            for (dst_type, dst_var, select_list) in select_tuples:
                code_middle += dst_var + " = " + select_list[ind] + "; "
            self.gen_add_code_line(code_start + code_middle + code_end)
    # else use a non-branching selector
    else:
        self.gen_add_code_line("// non-branching pointer selector")
        # get the inverse comparator
        n = len(counts)
        inverse_comparator = comparator.replace("<",">") if "<" in comparator else comparator.replace(">","<")
        inverse_comparator = inverse_comparator + "=" if len(inverse_comparator) == 1 else (inverse_comparator[0] if inverse_comparator != "==" else inverse_comparator)
        for tuple_i in range(len(select_tuples)):
            branch_code = ""
            for ind in range(n):
                if ind == 0 or comparator == "==":
                    if comparator == "==" and ind > 0:
                        branch_code += " + "
                    branch_code += "(" + loop_counter + " " + comparator + " " + counts[ind] + ")" 
                elif ind < n-1:
                    branch_code += " + (" + loop_counter + " " + comparator + " " + counts[ind] + " && " + loop_counter + " " + inverse_comparator + " " + counts[ind-1] + ")"
                else:
                    branch_code += " + (" + loop_counter + " " + inverse_comparator + " " + counts[ind-1] + ")"
                branch_code += " * " + select_tuples[tuple_i][2][ind]
            self.gen_add_code_line(dst_code[tuple_i] + " = " + branch_code + ";")

def gen_kernel_load_inputs(self, name, amount, name2=None, amount2=1, name3=None, amount3=1,
                                 stride=None, stride2=None, stride3=None):
    """Emit a load-inputs-to-shared block for up to 3 (name, amount) pairs.

    Batched-k kernels pass `stride{,2,3}` to address each timestep's slot via
    `&d_<name>[k*stride]`. Single-timing kernels omit strides; the load reads
    `d_<name>` directly.
    """
    def _emit(nm, amt, st):
        if st is None:
            src = "d_" + nm
        else:
            self.gen_add_code_line("const T *d_" + nm + "_k = &d_" + nm + "[k*" + st + "];")
            src = "d_" + nm + "_k"
        self.gen_add_parallel_loop("ind", amt)
        self.gen_add_code_line("s_" + nm + "[ind] = " + src + "[ind];")
        self.gen_add_end_control_flow()
    self.gen_add_code_line("// load to shared mem")
    _emit(name, amount, stride)
    if name2 is not None:
        _emit(name2, amount2, stride2)
    if name3 is not None:
        _emit(name3, amount3, stride3)
    self.gen_add_sync()

def gen_kernel_save_result(self, store_to_name, amount, load_from_name=None, stride=None):
    """Emit a save-result-from-shared block. Batched-k kernels pass `stride`
    to address each timestep's slot via `&d_<store_to_name>[k*stride]`;
    single-timing kernels omit it."""
    if load_from_name is None:
        load_from_name = "s_" + store_to_name
    self.gen_add_code_line("// save down to global")
    if stride is None:
        dst = "d_" + store_to_name
    else:
        self.gen_add_code_line("T *d_" + store_to_name + "_k = &d_" + store_to_name + "[k*" + stride + "];")
        dst = "d_" + store_to_name + "_k"
    self.gen_add_parallel_loop("ind", amount)
    self.gen_add_code_line(dst + "[ind] = " + load_from_name + "[ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()

def gen_anti_licm_input_reload(self, name, amount, \
                                     name2 = None, amount2 = 1, name3 = None, amount3 = 1, \
                                     feedback_from = None):
    """Inside a `for (rep ...)` single_timing loop, reload all inputs from
    device memory via a `const volatile T *` cast, stomp one slot of each
    input with `static_cast<T>(rep)`, and (when `feedback_from` is given)
    inject the previous rep's output back into the input. This creates a
    true loop-carried data dependency that no LICM pass can hoist.

    The feedback chain is the strongest defense: input[N] := f(d_input,
    rep, d_output[(rep-k) & 0x3FF])  with output[N] written by the rep N
    body. Because input N depends on output N-1, output N depends on input
    N, and so on, ptxas would need to symbolically execute every iteration
    to find a fixed point — far beyond any LICM pass's budget.

    Earlier defenses tried in order of failure:
      1. `__noinline__ grid_licm_barrier()` — defeated when ptxas stripped
         the function as no-op self-stores.
      2. Rep-stomp alone (`s_input[rep % N] = static_cast<T>(rep)`) — kept
         in this helper as the first line of defense, but ptxas can still
         prove subsets of work loop-invariant via partial value-range analysis
         (observed for end_effector_pose_gradient on sm_86 / CUDA 12.6).

    Joint-position values are safe across all GRiD algos: sin/cos are
    well-defined for any float; the algos don't assert ranges on q/qd/u.
    `feedback_from` is the name of the output buffer
    (matching `gen_anti_licm_output_write`'s `store_to_name`), so the read
    is `d_<feedback_from>[(rep - k) & 0x3FF]`. The output buffer is sized
    NUM_TIMESTEPS * output_per_step which is comfortably > 1024.
    """
    if _no_licm_barrier():
        # Opt-out path: emit nothing. Inputs were loaded before the rep loop;
        # the rep body runs against stable data and nvcc is free to LICM-elide.
        # Trade-off documented at the top of this module.
        self.gen_add_code_line(
            "// anti-LICM suppressed (GRID_NO_LICM_BARRIER=1); single-call may elide"
        )
        return
    # Both sides are volatile: read forces re-load from global; write to shared
    # via `volatile T *` cast forces nvcc to emit each store and prevents CSE
    # across iterations. The volatile reload alone is insufficient (compiler
    # can prove d_* contents are loop-invariant when no kernel writes them) —
    # the rep-stomp below provides the actual LICM defense.
    self.gen_add_code_line("// anti-LICM: volatile reload of inputs each rep")
    self.gen_add_parallel_loop("_aopt_i", amount)
    self.gen_add_code_line(
        "reinterpret_cast<volatile T *>(s_" + name + ")[_aopt_i] = "
        "reinterpret_cast<const volatile T *>(d_" + name + ")[_aopt_i];"
    )
    self.gen_add_end_control_flow()
    if name2 is not None:
        self.gen_add_parallel_loop("_aopt_i", amount2)
        self.gen_add_code_line(
            "reinterpret_cast<volatile T *>(s_" + name2 + ")[_aopt_i] = "
            "reinterpret_cast<const volatile T *>(d_" + name2 + ")[_aopt_i];"
        )
        self.gen_add_end_control_flow()
    if name3 is not None:
        self.gen_add_parallel_loop("_aopt_i", amount3)
        self.gen_add_code_line(
            "reinterpret_cast<volatile T *>(s_" + name3 + ")[_aopt_i] = "
            "reinterpret_cast<const volatile T *>(d_" + name3 + ")[_aopt_i];"
        )
        self.gen_add_end_control_flow()
    self.gen_add_sync()
    # anti-LICM defense (1/2): stomp one input slot with `rep`. First line
    # of defense. The compiler cannot fold the loop induction variable, so
    # at minimum one input slot provably varies per rep.
    self.gen_add_code_line("// anti-LICM (1/2): stomp one input slot with `rep`")
    self.gen_add_code_line("if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {")
    self.gen_add_code_line(
        "    reinterpret_cast<volatile T *>(s_" + name + ")[rep % (" + str(amount) + ")] = "
        "static_cast<T>(rep);"
    )
    if name2 is not None:
        self.gen_add_code_line(
            "    reinterpret_cast<volatile T *>(s_" + name2 + ")[rep % (" + str(amount2) + ")] = "
            "static_cast<T>(rep);"
        )
    if name3 is not None:
        self.gen_add_code_line(
            "    reinterpret_cast<volatile T *>(s_" + name3 + ")[rep % (" + str(amount3) + ")] = "
            "static_cast<T>(rep);"
        )
    self.gen_add_code_line("}")
    # anti-LICM defense (2/2): output→input feedback. Reads previous reps'
    # output values from d_<feedback_from> and adds them into input slots.
    # Combined with the output write at end of rep, this creates a closed
    # loop-carried dependency cycle: ptxas would need to symbolically
    # execute all NUM_TIMESTEPS iterations to find any fixed point, far
    # beyond any LICM budget. Reading 3 staggered slots ensures even
    # aggressive cycle analysis can't collapse the chain.
    if feedback_from is not None:
        self.gen_add_code_line(
            "// anti-LICM (2/2): feedback prev rep's d_" + feedback_from
            + " into s_" + name + " (true loop-carried dep)"
        )
        self.gen_add_code_line("if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) {")
        self.gen_add_code_line(
            "    T _aopt_fb1 = reinterpret_cast<const volatile T *>(d_"
            + feedback_from + ")[(rep + 0x3FF) & 0x3FF];"
        )
        self.gen_add_code_line(
            "    T _aopt_fb2 = reinterpret_cast<const volatile T *>(d_"
            + feedback_from + ")[(rep + 0x3FE) & 0x3FF];"
        )
        self.gen_add_code_line(
            "    T _aopt_fb3 = reinterpret_cast<const volatile T *>(d_"
            + feedback_from + ")[(rep + 0x3FD) & 0x3FF];"
        )
        self.gen_add_code_line(
            "    reinterpret_cast<volatile T *>(s_" + name
            + ")[(rep + 1) % (" + str(amount) + ")] += _aopt_fb1;"
        )
        self.gen_add_code_line(
            "    reinterpret_cast<volatile T *>(s_" + name
            + ")[(rep + 2) % (" + str(amount) + ")] += _aopt_fb2;"
        )
        self.gen_add_code_line(
            "    reinterpret_cast<volatile T *>(s_" + name
            + ")[(rep + 3) % (" + str(amount) + ")] += _aopt_fb3;"
        )
        self.gen_add_code_line("}")
    self.gen_add_sync()

def gen_anti_licm_output_write(self, store_to_name, load_from_name = None):
    """Inside a `for (rep ...)` single_timing loop, write the per-iter output's
    first element to a varying global address. Companion to
    `gen_anti_licm_input_reload`: the input reload prevents LICM in the SIMT
    path, but cuBLASDx-templated paths can still prove invariance unless
    we also force a per-iter side effect that depends on the iter's work.

    The destination cycles through 1024 slots of d_<store_to_name>, well
    within the MAX_TIMESTEPS=256 × output_size_per_step allocation any
    kernel's output buffer gets in init_gridData.
    """
    if load_from_name is None:
        load_from_name = "s_" + store_to_name
    if _no_licm_barrier():
        self.gen_add_code_line(
            "// anti-LICM output write suppressed (GRID_NO_LICM_BARRIER=1)"
        )
        return
    # __syncthreads() so thread 0 sees other threads' writes to s_<output>.
    # Without this, thread 0 only sees its own writes (or stale init-load values),
    # and if the inner algo doesn't have thread 0 personally write s_<output>[rep & 7],
    # the value is invariant across reps and LICM elides the entire algo.
    self.gen_add_code_line("__syncthreads();")
    self.gen_add_code_line(
        "if ((threadIdx.x | threadIdx.y | threadIdx.z) == 0) { "
        "reinterpret_cast<volatile T *>(d_" + store_to_name + ")[rep & 1023] = "
        "reinterpret_cast<const volatile T *>(" + load_from_name + ")[rep & 7]; }"
    )


def gen_add_shared_memory_helpers(self):
    self.gen_add_code_lines([
        "__host__ __device__ constexpr size_t grid_align_up(size_t offset, size_t alignment) {",
        "    return (offset + alignment - 1) / alignment * alignment;",
        "}",
        "",
        "template <typename U>",
        "__device__ U *grid_arena_ptr(unsigned char *arena, size_t byte_offset) {",
        "    return reinterpret_cast<U *>(arena + byte_offset);",
        "}",
        "",
        "template <typename T>",
        "__host__ __device__ constexpr size_t grid_shared_arena_bytes(size_t t_count, size_t int_count = 0, size_t extra_byte_count = 0) {",
        "    size_t offset = 0;",
        "    offset = grid_align_up(offset, alignof(T));",
        "    offset += sizeof(T) * t_count;",
        "    if (int_count > 0) {",
        "        offset = grid_align_up(offset, alignof(int));",
        "        offset += sizeof(int) * int_count;",
        "    }",
        "    if (extra_byte_count > 0) {",
        "        offset = grid_align_up(offset, static_cast<size_t>(16));",
        "        offset += extra_byte_count;",
        "    }",
        "    return grid_align_up(offset, static_cast<size_t>(16));",
        "}",
        "",
        "#ifndef GRID_CUDA_TARGET_SHARED_MEM_BYTES",
        "#define GRID_CUDA_TARGET_SHARED_MEM_BYTES 98304",
        "#endif",
        "",
        "#ifndef GRID_WORKSPACE_SLOTS",
        "#define GRID_WORKSPACE_SLOTS 1",
        "#endif",
        "",
        "enum gridDataKind { GRID_DATA_ALL = 0, GRID_DATA_DYNAMICS = 1, GRID_DATA_KINEMATICS = 2 };",
        "enum gridSharedTier { GRID_SHARED_FULL = 0, GRID_SPILL_DA_DF_OUTPUT = 1, GRID_SPILL_DV_DA_DF_OUTPUT = 2 };",
        "// Time integrator family selected by integrator kernels at compile time.",
        "// EULER / SEMI_IMPLICIT_EULER / TRAPEZOIDAL are single-stage; MIDPOINT / RK3 / RK4",
        "// are multi-stage (driven inline from integrator_inner). TRAPEZOIDAL = 5 (NOT MIDPOINT=2).",
        "enum class IntegratorType { EULER = 0, SEMI_IMPLICIT_EULER = 1, MIDPOINT = 2, RK3 = 3, RK4 = 4, TRAPEZOIDAL = 5 };",
        "",
        "#ifndef GRID_CUDA_ENABLE_L2_PERSISTING",
        # Phase 3a/b/c spill design: workspace bytes are HOT (recursion-internal
        # buffers like Minv-F, ABA's interleaved scratch, FDSVA_SO's df_du/Minv)
        # touched many times per kernel. L2 pinning narrows the smem→HBM penalty
        # to smem→L2 (~few-cycle hit instead of 100s of cycles). Default-ON
        # because every codegen target we emit either doesn't spill (= no-op)
        # or spills hot data (= L2 is the right cache). Niche concurrent-kernel
        # workloads can override with -DGRID_CUDA_ENABLE_L2_PERSISTING=0.
        "#define GRID_CUDA_ENABLE_L2_PERSISTING 1",
        "#endif",
        "",
        "__host__ inline cudaError_t grid_get_max_dynamic_shared_memory_bytes(size_t *bytes) {",
        "    int device = 0;",
        "    cudaError_t err = cudaGetDevice(&device);",
        "    if (err != cudaSuccess) { return err; }",
        "    int max_per_block = 0;",
        "    err = cudaDeviceGetAttribute(&max_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device);",
        "    if (err != cudaSuccess) { return err; }",
        "    int max_optin = 0;",
        "#if CUDART_VERSION >= 9000",
        "    err = cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);",
        "    if (err != cudaSuccess) { cudaGetLastError(); max_optin = 0; }",
        "#endif",
        "    *bytes = static_cast<size_t>(max_optin > max_per_block ? max_optin : max_per_block);",
        "    return cudaSuccess;",
        "}",
        "",
        "__host__ inline cudaError_t grid_check_dynamic_shared_memory_bytes(const char *kernel_name, size_t bytes) {",
        "    size_t max_bytes = 0;",
        "    cudaError_t err = grid_get_max_dynamic_shared_memory_bytes(&max_bytes);",
        "    if (err != cudaSuccess) { return err; }",
        "    if (bytes > max_bytes) {",
        "        fprintf(stderr, \"GRID shared-memory request for %s is %zu bytes, but this device supports %zu bytes per block\\n\",",
        "                kernel_name, bytes, max_bytes);",
        "        return cudaErrorInvalidConfiguration;",
        "    }",
        "    return cudaSuccess;",
        "}",
        "",
        "__host__ inline cudaError_t grid_begin_l2_persisting(cudaStream_t stream, void *ptr, size_t bytes) {",
        "#if GRID_CUDA_ENABLE_L2_PERSISTING && CUDART_VERSION >= 11000",
        "    if (ptr == nullptr || bytes == 0) { return cudaSuccess; }",
        "    int device = 0;",
        "    cudaError_t err = cudaGetDevice(&device);",
        "    if (err != cudaSuccess) { return err; }",
        "    int max_window = 0;",
        "    err = cudaDeviceGetAttribute(&max_window, cudaDevAttrMaxAccessPolicyWindowSize, device);",
        "    if (err != cudaSuccess || max_window <= 0) { cudaGetLastError(); return cudaSuccess; }",
        "    int max_persisting_l2 = 0;",
        "    err = cudaDeviceGetAttribute(&max_persisting_l2, cudaDevAttrMaxPersistingL2CacheSize, device);",
        "    if (err == cudaSuccess && max_persisting_l2 > 0) {",
        "        size_t l2_bytes = bytes < static_cast<size_t>(max_persisting_l2) ? bytes : static_cast<size_t>(max_persisting_l2);",
        "        cudaError_t limit_err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_bytes);",
        "        if (limit_err != cudaSuccess) { cudaGetLastError(); }",
        "    }",
        "    else { cudaGetLastError(); }",
        "    cudaStreamAttrValue attr;",
        "    memset(&attr, 0, sizeof(attr));",
        "    attr.accessPolicyWindow.base_ptr = ptr;",
        "    attr.accessPolicyWindow.num_bytes = bytes < static_cast<size_t>(max_window) ? bytes : static_cast<size_t>(max_window);",
        "    attr.accessPolicyWindow.hitRatio = 0.60;",
        "    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;",
        "    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;",
        "    return cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);",
        "#else",
        "    (void)stream; (void)ptr; (void)bytes;",
        "    return cudaSuccess;",
        "#endif",
        "}",
        "",
        "__host__ inline cudaError_t grid_end_l2_persisting(cudaStream_t stream) {",
        "#if GRID_CUDA_ENABLE_L2_PERSISTING && CUDART_VERSION >= 11000",
        "    cudaStreamAttrValue attr;",
        "    memset(&attr, 0, sizeof(attr));",
        "    attr.accessPolicyWindow.num_bytes = 0;",
        "    return cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);",
        "#else",
        "    (void)stream;",
        "    return cudaSuccess;",
        "#endif",
        "}",
        ""
    ])

def gen_declare_shared_arena(self, t_buffers, temp_mem_size, include_topology_helpers = True,
                             ximat_name = "s_XImats", ximat_size = 0,
                             temp_name = "s_temp", topology_name = "s_topology_helpers",
                             extra_byte_regions = None,
                             tier_workspace_expr = None,
                             arena_base_expr = None):
    """Emit the shared-memory arena layout.

    When ``tier_workspace_expr`` is non-None, the ``s_temp`` slot becomes
    tier-aware: at TIER_SHARED the slot is allocated from the arena as usual;
    at TIER_LITE+/MINIMAL the slot is sourced from the supplied workspace
    pointer expression (e.g. ``"d_workspace"``) and the arena allocation
    skips the temp slot entirely, freeing that smem for the caller's outer
    kernel. The arena_offset variable accumulates conditionally so trailing
    slots (topology, linalg, etc.) shift up at LITE+/MINIMAL.

    The caller must declare ``RESOURCE_TIER`` as a template parameter and
    expose ``tier_workspace_expr`` as a function argument.

    When ``arena_base_expr`` is non-None the arena is sourced from a CALLER
    pointer (e.g. ``"s_scratch"``) instead of the kernel's ``extern __shared__``
    dynamic-smem block. This lets an `_inner` lay out the same sub-buffers from
    memory the caller already owns -- so it can be invoked from another kernel
    (e.g. GATO's BSQP) without the ``extern __shared__`` aliasing the caller's
    live arena. The base must be at least 16-byte aligned (it is reinterpreted
    to ``unsigned char *`` and the per-slot grid_align_up handles the rest).
    Default None preserves the ``extern __shared__`` declaration verbatim, so
    every existing caller is byte-identical.
    """
    if extra_byte_regions is None:
        extra_byte_regions = []
    topology_count = self.gen_topology_helpers_size() if include_topology_helpers else 0
    # A t_buffer count may be a C++ constexpr expression string (e.g. a per-tier
    # slot size like "REGRESSOR_Y_OUTPUT_SLOT") rather than a Python int; such slots are
    # tier-routed and excluded from the (debug-only) Python fixed-size accounting.
    def _int_or_zero(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0
    fixed_t_region_count = sum(_int_or_zero(count) for _, count in t_buffers)
    if ximat_size:
        fixed_t_region_count += int(ximat_size)
    temp_size_int = int(temp_mem_size) if temp_mem_size is not None else 0
    t_region_count = fixed_t_region_count + temp_size_int
    # If any t_buffer count is a C++ constexpr-string (tier-routed slot like a
    # per-tier spilled buffer), the Python int sum above undercounts it. Build a
    # C++ sum expression so the (debug-only) layout assert stays exact at every
    # tier. has_expr_count => the t_region_count int is incomplete; use the expr.
    has_expr_count = any(not isinstance(count, int) for _, count in t_buffers)
    t_region_count_expr = " + ".join(["0"] + [str(count) for _, count in t_buffers]
                                     + ([str(int(ximat_size))] if ximat_size else [])
                                     + ([str(temp_size_int)] if temp_size_int else []))
    extra_byte_expr = " + ".join(str(count) for _, count in extra_byte_regions) if extra_byte_regions else "0"
    self.gen_add_code_line("// GRID shared arena layout")
    for name, count in t_buffers:
        self.gen_add_code_line("//   T " + name + "[" + str(count) + "]")
    if ximat_size:
        self.gen_add_code_line("//   T " + ximat_name + "[" + str(ximat_size) + "]")
    if temp_size_int != 0:
        if tier_workspace_expr is not None:
            self.gen_add_code_line("//   T " + temp_name + "[" + str(temp_mem_size) + "] (TIER_SHARED only; LITE/MINIMAL route to " + tier_workspace_expr + ")")
        else:
            self.gen_add_code_line("//   T " + temp_name + "[" + str(temp_mem_size) + "]")
    if topology_count > 0:
        self.gen_add_code_line("//   int " + topology_name + "[" + str(topology_count) + "]")
    for name, count in extra_byte_regions:
        self.gen_add_code_line("//   bytes " + name + "[" + str(count) + "]")
    if arena_base_expr is None:
        self.gen_add_code_line("extern __shared__ __align__(16) unsigned char s_arena[];")
    else:
        self.gen_add_code_line("unsigned char *s_arena = reinterpret_cast<unsigned char *>(" + arena_base_expr + ");")
    self.gen_add_code_line("size_t s_arena_offset = 0;")
    for name, count in t_buffers:
        self.gen_add_code_line("s_arena_offset = grid_align_up(s_arena_offset, alignof(T));")
        self.gen_add_code_line("T *" + name + " = grid_arena_ptr<T>(s_arena, s_arena_offset);")
        self.gen_add_code_line("s_arena_offset += sizeof(T) * static_cast<size_t>(" + str(count) + ");")
    if ximat_size:
        self.gen_add_code_line("s_arena_offset = grid_align_up(s_arena_offset, alignof(T));")
        self.gen_add_code_line("T *" + ximat_name + " = grid_arena_ptr<T>(s_arena, s_arena_offset);")
        self.gen_add_code_line("s_arena_offset += sizeof(T) * static_cast<size_t>(" + str(ximat_size) + ");")
    if temp_size_int != 0:
        if tier_workspace_expr is not None:
            self.gen_add_code_line("T *" + temp_name + ";")
            self.gen_add_code_line("if constexpr (RESOURCE_TIER == TIER_SHARED) {", True)
            self.gen_add_code_line("(void)" + tier_workspace_expr + ";")
            self.gen_add_code_line("s_arena_offset = grid_align_up(s_arena_offset, alignof(T));")
            self.gen_add_code_line(temp_name + " = grid_arena_ptr<T>(s_arena, s_arena_offset);")
            self.gen_add_code_line("s_arena_offset += sizeof(T) * static_cast<size_t>(" + str(temp_mem_size) + ");")
            self.gen_add_end_control_flow()
            self.gen_add_code_line("else {", True)
            self.gen_add_code_line(temp_name + " = " + tier_workspace_expr + ";")
            self.gen_add_end_control_flow()
        else:
            self.gen_add_code_line("s_arena_offset = grid_align_up(s_arena_offset, alignof(T));")
            self.gen_add_code_line("T *" + temp_name + " = grid_arena_ptr<T>(s_arena, s_arena_offset);")
            self.gen_add_code_line("s_arena_offset += sizeof(T) * static_cast<size_t>(" + str(temp_mem_size) + ");")
    else:
        self.gen_add_code_line("T *" + temp_name + " = nullptr;")
    if topology_count > 0:
        self.gen_add_code_line("s_arena_offset = grid_align_up(s_arena_offset, alignof(int));")
        self.gen_add_code_line("int *" + topology_name + " = grid_arena_ptr<int>(s_arena, s_arena_offset);")
        self.gen_add_code_line("s_arena_offset += sizeof(int) * static_cast<size_t>(" + str(topology_count) + ");")
    else:
        # Always declare the pointer (nullptr) so inner-function call sites can pass
        # it uniformly even when this robot allocates no topology helpers.
        self.gen_add_code_line("int *" + topology_name + " = nullptr;")
    for name, count in extra_byte_regions:
        self.gen_add_code_line("unsigned char *" + name + " = nullptr;")
        self.gen_add_code_line("if (static_cast<size_t>(" + str(count) + ") > 0) {", True)
        self.gen_add_code_line("s_arena_offset = grid_align_up(s_arena_offset, static_cast<size_t>(16));")
        self.gen_add_code_line(name + " = grid_arena_ptr<unsigned char>(s_arena, s_arena_offset);")
        self.gen_add_code_line("s_arena_offset += static_cast<size_t>(" + str(count) + ");")
        self.gen_add_end_control_flow()
    self.gen_add_code_line("#ifdef GRID_CUDA_DEBUG_LAYOUT")
    if tier_workspace_expr is not None and temp_size_int != 0:
        # Different t_region_count per tier: PERF includes temp; LITE+ excludes it
        self.gen_add_code_line("if constexpr (RESOURCE_TIER == TIER_SHARED) {", True)
        self.gen_add_code_line("assert(s_arena_offset == grid_shared_arena_bytes<T>(" + str(t_region_count) + ", " + str(topology_count) + ", " + extra_byte_expr + "));")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("else {", True)
        self.gen_add_code_line("assert(s_arena_offset == grid_shared_arena_bytes<T>(" + str(fixed_t_region_count) + ", " + str(topology_count) + ", " + extra_byte_expr + "));")
        self.gen_add_end_control_flow()
    elif has_expr_count:
        # Tier-routed slot present: the C++ slot-size expression is exact per tier.
        self.gen_add_code_line("assert(s_arena_offset == grid_shared_arena_bytes<T>(static_cast<size_t>(" + t_region_count_expr + "), " + str(topology_count) + ", " + extra_byte_expr + "));")
    else:
        self.gen_add_code_line("assert(s_arena_offset == grid_shared_arena_bytes<T>(" + str(t_region_count) + ", " + str(topology_count) + ", " + extra_byte_expr + "));")
    self.gen_add_code_line("#endif")
    self.gen_add_code_line("(void)s_arena_offset;")

def gen_shared_arena_t_count(self, t_buffers, temp_mem_size, helper_size):
    count = helper_size
    if temp_mem_size is not None:
        count += int(temp_mem_size)
    for _, region_count in t_buffers:
        count += int(region_count)
    return count

def gen_device_wrapper(self, func_desc, func_def, shared_mem_size, inner_call_fn,
                       template_line = "template <typename T>",
                       func_notes = None, func_params = None,
                       extra_t_buffers = None, include_linalg_scratch = True,
                       tier_workspace_expr = None, skip_floating_base_X = False):
    """Emit the shared `__device__` wrapper skeleton common to the simple
    inline-CUDA device entry points (id / fd / aba / crba / minv /
    idsva_so / integrator). Each of these repeats the identical sequence:

        gen_add_func_doc(...)
        gen_add_code_line(<template line>)
        gen_add_code_line("__device__")
        gen_add_code_line(func_def, True)
        gen_XImats_helpers_temp_shared_memory_code(shared_mem_size, ...)
        gen_load_update_XImats_helpers_function_call(...)
        <per-algo inner call(s)>
        gen_add_end_function()

    The genuinely per-algo parts are kept as parameters, NOT erased:
      - `func_def`        : the full signature string (built by the caller).
      - `template_line`   : `<typename T>` vs the tier-aware /
                            IntegratorType variants.
      - `extra_t_buffers` : the algo's extra smem t-regions (default none).
      - `include_linalg_scratch` : whether the algo reserves linalg scratch
                            (True for all but idsva_so).
      - `tier_workspace_expr` : the LITE/MINIMAL whole-arena repoint target
                            (only the tier-aware device paths set this).
      - `skip_floating_base_X` : CRBA-only surgical lever on the XImats load.
      - `inner_call_fn`   : a 0-arg closure that emits the per-algo inner
                            call(s) between the XImats load and the function
                            end (the irreducibly per-algo body).

    The (B+C §1.1) consolidation: every matching `gen_*_device` shrinks to
    building its `func_def`/params then ONE call here. Output is byte-identical
    to the pre-collapse hand-rolled wrappers."""
    self.gen_add_func_doc(func_desc, func_notes if func_notes is not None else [],
                          func_params if func_params is not None else [], None)
    self.gen_add_code_line(template_line)
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    self.gen_XImats_helpers_temp_shared_memory_code(
        shared_mem_size, extra_t_buffers = extra_t_buffers,
        include_linalg_scratch = include_linalg_scratch,
        tier_workspace_expr = tier_workspace_expr)
    self.gen_load_update_XImats_helpers_function_call(skip_floating_base_X = skip_floating_base_X)
    inner_call_fn()
    self.gen_add_end_function()

def gen_tier_dispatch(self, picks, emit_body_fn):
    """Emit the per-tier spill-pick dispatch scaffolding shared by 12 kernel
    emitters (crba / fd / forward_dynamics_gradient / aba / fdsva_so / integrator / minv / inverse_dynamics_gradient /
    end_effector_pose_gradient / d2ee / idsva_so body+world). Every site repeated the identical
    scaffolding (B+C §1.2):

        picks = getattr(self, "<algo>_spill_tier_3way", (...))
        if picks[0] == picks[1] == picks[2]:
            <emit body for picks[0]>
        else:
            tier_names = ("TIER_SHARED", "TIER_LITE", "TIER_MINIMAL")
            for tier_idx, (tier_name, pick) in enumerate(zip(tier_names, picks)):
                head = "if constexpr (RESOURCE_TIER == "+tier_name+") {" if tier_idx==0
                       else "else if constexpr (RESOURCE_TIER == "+tier_name+") {"
                self.gen_add_code_line(head, True)
                <emit body for pick>
                self.gen_add_end_control_flow()

    `picks` is the (perf, lite, minimal) 3-tuple. `emit_body_fn(pick)` is the
    per-algo closure that unpacks `pick` (e.g. `bool(pick)`, or
    `_<ALGO>_PICK_FLAGS[pick]`) and emits that tier's
    `_emit_<algo>_kernel_body_for_flags(...)`. When all three picks agree the
    body is emitted once with no if-constexpr guard (byte-identical to the
    collapsed-single-body original); otherwise the three-branch constexpr
    ladder is emitted. The tier-name tuple is single-sourced here (T5's
    TIER_SHARED/TIER_LITE/TIER_MINIMAL symbols)."""
    if picks[0] == picks[1] == picks[2]:
        emit_body_fn(picks[0])
    else:
        tier_names = ("TIER_SHARED", "TIER_LITE", "TIER_MINIMAL")
        for tier_idx, (tier_name, pick) in enumerate(zip(tier_names, picks)):
            head = "if constexpr (RESOURCE_TIER == " + tier_name + ") {" if tier_idx == 0 else \
                   "else if constexpr (RESOURCE_TIER == " + tier_name + ") {"
            self.gen_add_code_line(head, True)
            emit_body_fn(pick)
            self.gen_add_end_control_flow()

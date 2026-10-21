"""Cost menu (2026-10-21): activations, activated terms, linear feature, named-frame
residuals, barrier activations and the menu preset.

Every cost term in grid_plant factors as an activation applied per entry to a
residual r with Jacobian J: value = sum_i a(r_i), gradient = J^T a'(r), and the
Gauss-Newton Hessian J^T diag(a''(r)) J. The quadratic terms the plant namespace
already carries are the quadratic activation over their residuals and keep their
bytes; everything here is emitted after them.

Activations a(r) with weight w per entry (eps is the activation parameter):
  quadratic            a = 1/2 w r^2            a' = w r           a'' = w
  pseudo-Huber         a = w (sqrt(r^2+e^2)-e)  a' = w r / s       a'' = w e^2 / s^3,  s = sqrt(r^2+e^2), e > 0
  relaxed log barrier  on the feature value v in [lo, hi]: B(v - lo) + B(hi - v), w = mu, e = delta,
                       B(z) = -mu log z above delta and its quadratic extension below (Grandia et al.)
  quadratic barrier    1/2 w (max(0, lo - v)^2 + max(0, v - hi)^2)
An infinite bound contributes nothing. The barrier kinds read the per-entry bound
buffers a term receives (nullptr when unused).

Residuals: the state x - x_des, the input u - u_des, a linear feature A v + b -
target of the state or the input, and, on a baked cost frame (an articulated or
fixed joint, index 0 is the end-effector target): the position p_f - p_des, the
log-map orientation log(R_des^T R_f) and the twist J(q) qd - v_des. The frame
terms sit on grid::frame_jacobian_inner through a caller-scratch evaluator.
`tracking_cost_menu` composes them with the existing barriers behind
`cost_menu_params<T>`; a term whose activation entry is GRID_PLANT_ACTIVATION_NONE
is skipped.
"""
import numpy as np


ACTIVATIONS = (("GRID_PLANT_ACTIVATION_NONE", -1), ("GRID_PLANT_ACTIVATION_QUADRATIC", 0),
               ("GRID_PLANT_ACTIVATION_PSEUDO_HUBER", 1), ("GRID_PLANT_ACTIVATION_RELAXED_LOG_BARRIER", 2),
               ("GRID_PLANT_ACTIVATION_QUADRATIC_BARRIER", 3))
MENU_TERMS = ("FRAME_POS", "STATE", "INPUT", "LINEAR_X", "LINEAR_U", "ORIENTATION", "VELOCITY", "FRAME_POS_2")


def resolve_cost_frames(self, names):
    """[(name, jid, X_offset[16] column-major)] for the baked cost frames. An
    articulated joint resolves to its own id with an identity offset; a fixed
    joint to its movable parent with the fixed transform. None selects the
    end-effector target alone (the named fixed target when one is baked, else
    the first leaf joint), which is also always frame 0."""
    robot = self.robot
    default = getattr(self, "_ee_target_name", "") or robot.get_joint_by_id(robot.get_leaf_nodes()[0]).get_name()
    requested = list(names or [])
    names = [default]
    for n in requested:
        if n not in names:
            names.append(n)
    frames = []
    for nm in names:
        joint = robot.get_joint_by_name(nm)
        if joint is not None:
            frames.append((nm, int(joint.get_id()), [float(v) for v in np.eye(4).flatten(order="F")]))
            continue
        fixed = robot.get_fixed_joint_by_name(nm)
        if fixed is None:
            raise ValueError(f"cost frame '{nm}' is neither a joint nor a fixed joint of this URDF")
        parent_name = fixed.get_parent()
        parent = robot.get_joint_by_name(parent_name) if parent_name else None
        if parent is None:
            raise ValueError(f"cost frame '{nm}' hangs off the world root; a cost frame must move with a joint")
        Xh = np.asarray(fixed.get_transformation_matrix_hom(), dtype=np.float64)
        frames.append((nm, int(parent.get_id()), [float(v) for v in Xh.flatten(order="F")]))
    return frames


# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------

def gen_cost_activations(self):
    self.gen_add_code_line("")
    self.gen_add_code_line("// ---- cost activations: value, slope and curvature per residual entry ----")
    self.gen_add_code_line("// a(r; w, eps): quadratic 1/2 w r^2; pseudo-Huber w (sqrt(r^2 + eps^2) - eps), eps > 0; the barrier kinds act on")
    self.gen_add_code_line("// the feature value v with bounds [lo, hi] (w = mu, eps = delta for the relaxed log barrier; an infinite bound is 0).")
    self.gen_add_code_line("// Every activated term is value = sum_i a(r_i), gradient = J^T a'(r), GN Hessian = J^T diag(a''(r)) J.")
    for name, val in ACTIVATIONS:
        self.gen_add_code_line("const int " + name + " = " + str(val) + ";")
    self.gen_add_code_line("// menu term order in cost_menu_params::activation / eps (tracking_cost_menu)")
    for i, t in enumerate(MENU_TERMS):
        self.gen_add_code_line("const int GRID_PLANT_MENU_" + t + " = " + str(i) + ";")
    self.gen_add_code_line("const int GRID_PLANT_MENU_TERMS = " + str(len(MENU_TERMS)) + ";")
    # relaxed log barrier on z >= 0
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line("T relaxed_log_barrier_value(const T z, const T mu, const T delta) {", True)
    self.gen_add_code_line("if (z > delta) { return -mu * log(z); }")
    self.gen_add_code_line("T t = (z - static_cast<T>(2) * delta) / delta;")
    self.gen_add_code_line("return mu * (static_cast<T>(0.5) * t * t - static_cast<T>(0.5) - log(delta));")
    self.gen_add_end_function()
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line("T relaxed_log_barrier_slope(const T z, const T mu, const T delta) {", True)
    self.gen_add_code_line("if (z > delta) { return -mu / z; }")
    self.gen_add_code_line("return mu * (z - static_cast<T>(2) * delta) / (delta * delta);")
    self.gen_add_end_function()
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line("T relaxed_log_barrier_curvature(const T z, const T mu, const T delta) {", True)
    self.gen_add_code_line("if (z > delta) { return mu / (z * z); }")
    self.gen_add_code_line("return mu / (delta * delta);")
    self.gen_add_end_function()
    for which, body in (
        ("value", [
            "if (kind == GRID_PLANT_ACTIVATION_PSEUDO_HUBER) { T s = sqrt(r * r + eps * eps); return w * (s - eps); }",
            "if (kind == GRID_PLANT_ACTIVATION_RELAXED_LOG_BARRIER) {",
            "    T acc = static_cast<T>(0);",
            "    if (!isinf(lo)) { acc += relaxed_log_barrier_value<T>(v - lo, w, eps); }",
            "    if (!isinf(hi)) { acc += relaxed_log_barrier_value<T>(hi - v, w, eps); }",
            "    return acc;",
            "}",
            "if (kind == GRID_PLANT_ACTIVATION_QUADRATIC_BARRIER) {",
            "    T below = (!isinf(lo) && v < lo) ? (lo - v) : static_cast<T>(0);",
            "    T above = (!isinf(hi) && v > hi) ? (v - hi) : static_cast<T>(0);",
            "    return static_cast<T>(0.5) * w * (below * below + above * above);",
            "}",
            "if (kind == GRID_PLANT_ACTIVATION_NONE) { return static_cast<T>(0); }",
            "return static_cast<T>(0.5) * w * r * r;"]),
        ("slope", [
            "if (kind == GRID_PLANT_ACTIVATION_PSEUDO_HUBER) { T s = sqrt(r * r + eps * eps); return w * r / s; }",
            "if (kind == GRID_PLANT_ACTIVATION_RELAXED_LOG_BARRIER) {",
            "    T acc = static_cast<T>(0);",
            "    if (!isinf(lo)) { acc += relaxed_log_barrier_slope<T>(v - lo, w, eps); }",
            "    if (!isinf(hi)) { acc -= relaxed_log_barrier_slope<T>(hi - v, w, eps); }",
            "    return acc;",
            "}",
            "if (kind == GRID_PLANT_ACTIVATION_QUADRATIC_BARRIER) {",
            "    T below = (!isinf(lo) && v < lo) ? (lo - v) : static_cast<T>(0);",
            "    T above = (!isinf(hi) && v > hi) ? (v - hi) : static_cast<T>(0);",
            "    return w * (above - below);",
            "}",
            "if (kind == GRID_PLANT_ACTIVATION_NONE) { return static_cast<T>(0); }",
            "return w * r;"]),
        ("curvature", [
            "if (kind == GRID_PLANT_ACTIVATION_PSEUDO_HUBER) { T s = sqrt(r * r + eps * eps); return w * eps * eps / (s * s * s); }",
            "if (kind == GRID_PLANT_ACTIVATION_RELAXED_LOG_BARRIER) {",
            "    T acc = static_cast<T>(0);",
            "    if (!isinf(lo)) { acc += relaxed_log_barrier_curvature<T>(v - lo, w, eps); }",
            "    if (!isinf(hi)) { acc += relaxed_log_barrier_curvature<T>(hi - v, w, eps); }",
            "    return acc;",
            "}",
            "if (kind == GRID_PLANT_ACTIVATION_QUADRATIC_BARRIER) {",
            "    bool active = (!isinf(lo) && v < lo) || (!isinf(hi) && v > hi);",
            "    return active ? w : static_cast<T>(0);",
            "}",
            "if (kind == GRID_PLANT_ACTIVATION_NONE) { return static_cast<T>(0); }",
            "return w;"])):
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__device__ __forceinline__")
        self.gen_add_code_line("T activation_" + which + "(const int kind, const T w, const T r, const T eps, const T v, const T lo, const T hi) {", True)
        for line in body:
            self.gen_add_code_line(line)
        self.gen_add_end_function()
    # the two-argument spellings for residual-only kinds keep call sites short
    for which in ("value", "slope", "curvature"):
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__device__ __forceinline__")
        self.gen_add_code_line("T activation_" + which + "(const int kind, const T w, const T r, const T eps) {", True)
        self.gen_add_code_line("return activation_" + which + "<T>(kind, w, r, eps, r, -INFINITY, INFINITY);")
        self.gen_add_end_function()


_BOUNDS_DOC = "s_lower / s_upper are per-entry bounds for the barrier activations (nullptr when the activation is not a barrier)"


def _bound_reads(i):
    return ("T lo = s_lower ? s_lower[" + i + "] : -INFINITY; T hi = s_upper ? s_upper[" + i + "] : INFINITY;")


# ---------------------------------------------------------------------------
# state / input (activated quadratic family)
# ---------------------------------------------------------------------------

def gen_activated_cost_family(self, which):
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    if which == "state":
        size, var, des, w, base = nq + nv, "s_x", "s_x_des", "s_Q", "state_cost"
        size_doc = "NUM_POS + NUM_VEL = " + str(size)
    else:
        size, var, des, w, base = nv, "s_u", "s_u_des", "s_R", "input_cost"
        size_doc = "NUM_VEL = " + str(size)
    N = str(size)
    common = [var + " is the current value (size " + size_doc + ")",
              des + " is the target (size " + size_doc + ")",
              w + " is the per-entry weight (size " + size_doc + ")",
              "activation is a GRID_PLANT_ACTIVATION_* id; eps is its parameter", _BOUNDS_DOC]
    args = "const int activation, const T eps, const T *s_lower, const T *s_upper"
    self.gen_add_func_doc(
        base + ": value = sum_i a(" + var + "[i] - " + des + "[i]; " + w + "[i], eps) under the chosen activation",
        ["Block-cooperative: strided per-entry activations into s_scratch, then a serial reduction into s_out[0].",
         "ACCUMULATE=false overwrites s_out[0]; ACCUMULATE=true adds into it.",
         "Quadratic activation reproduces quadratic_" + base + " exactly; s_scratch must hold >= " + N + "."],
        ["s_out is the scalar cost output (s_out[0])"] + common + ["s_scratch is shared scratch of size >= " + N], None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void " + base + "(T *s_out, const T *" + var + ", const T *" + des + ", const T *" + w + ", " + args + ", T *s_scratch) {", True)
    self.gen_add_parallel_loop("i", N)
    self.gen_add_code_line(_bound_reads("i"))
    self.gen_add_code_line("s_scratch[i] = activation_value<T>(activation, " + w + "[i], " + var + "[i] - " + des + "[i], eps, " + var + "[i], lo, hi);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int i = 0; i < " + N + "; ++i) acc += s_scratch[i];")
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc(base + "_gradient: g[i] = a'(" + var + "[i] - " + des + "[i]; " + w + "[i], eps)",
                          ["ACCUMULATE=false overwrites s_grad; ACCUMULATE=true adds into it."],
                          ["s_grad is the gradient output (size " + size_doc + ")"] + common, None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void " + base + "_gradient(T *s_grad, const T *" + var + ", const T *" + des + ", const T *" + w + ", " + args + ") {", True)
    self.gen_add_parallel_loop("i", N)
    self.gen_add_code_line(_bound_reads("i"))
    self.gen_add_code_line("T g = activation_slope<T>(activation, " + w + "[i], " + var + "[i] - " + des + "[i], eps, " + var + "[i], lo, hi);")
    self.gen_add_code_line("if (ACCUMULATE) { s_grad[i] += g; } else { s_grad[i] = g; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc(base + "_hessian: Gauss-Newton Hessian = diag(a''(" + var + " - " + des + "; " + w + ", eps)), column-major " + N + " x " + N,
                          ["Off-diagonal entries are zero. ACCUMULATE=false overwrites; ACCUMULATE=true adds."],
                          ["s_hess is the dense Hessian output (size " + N + "*" + N + ", column-major)"] + common, None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void " + base + "_hessian(T *s_hess, const T *" + var + ", const T *" + des + ", const T *" + w + ", " + args + ") {", True)
    self.gen_add_parallel_loop("ind", str(size * size))
    self.gen_add_code_line("int row = ind % " + N + ";")
    self.gen_add_code_line("int col = ind / " + N + ";")
    self.gen_add_code_line("T h = static_cast<T>(0);")
    self.gen_add_code_line("if (row == col) { " + _bound_reads("row") + " h = activation_curvature<T>(activation, " + w + "[row], " + var + "[row] - " + des + "[row], eps, " + var + "[row], lo, hi); }")
    self.gen_add_code_line("if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


# ---------------------------------------------------------------------------
# linear feature A v + b - target
# ---------------------------------------------------------------------------

def gen_linear_feature_cost(self):
    doc = ["s_v is the variable (size N: NX for the state, NU for the input)",
           "s_A is the feature matrix, column-major M x N (A[m + M*n]); s_b is the offset (M)",
           "s_target is the per-knot target (M); s_W is the per-entry weight (M)",
           "activation / eps as for state_cost; " + _BOUNDS_DOC + " (bounds on the feature value A v + b)",
           "s_scratch must hold >= M"]
    args = "const int activation, const T eps, const T *s_lower, const T *s_upper, T *s_scratch"
    residual = ["T f = s_b[m];", "for (int n = 0; n < N; ++n) f += s_A[m + M * n] * s_v[n];", "T r = f - s_target[m];", _bound_reads("m")]
    self.gen_add_func_doc("linear_feature_cost: value = sum_m a(r_m; W[m], eps), r = A v + b - target (a linear feature of the state or the input)",
                          ["Template M is the feature row count, N the variable size; both compile-time (loop bounds).",
                           "ACCUMULATE=false overwrites s_out[0]; ACCUMULATE=true adds into it."],
                          ["s_out is the scalar cost output (s_out[0])"] + doc, None)
    self.gen_add_code_line("template <typename T, int M, int N, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void linear_feature_cost(T *s_out, const T *s_v, const T *s_A, const T *s_b, const T *s_target, const T *s_W, " + args + ") {", True)
    self.gen_add_parallel_loop("m", "M")
    for l in residual:
        self.gen_add_code_line(l)
    self.gen_add_code_line("s_scratch[m] = activation_value<T>(activation, s_W[m], r, eps, f, lo, hi);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int m = 0; m < M; ++m) acc += s_scratch[m];")
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc("linear_feature_cost_gradient: g = A^T a'(r)  (size N)",
                          ["s_scratch (>= M) holds the per-row slopes between the two passes.",
                           "ACCUMULATE=false overwrites s_grad; ACCUMULATE=true adds into it."],
                          ["s_grad is the gradient output (size N)"] + doc, None)
    self.gen_add_code_line("template <typename T, int M, int N, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void linear_feature_cost_gradient(T *s_grad, const T *s_v, const T *s_A, const T *s_b, const T *s_target, const T *s_W, " + args + ") {", True)
    self.gen_add_parallel_loop("m", "M")
    for l in residual:
        self.gen_add_code_line(l)
    self.gen_add_code_line("s_scratch[m] = activation_slope<T>(activation, s_W[m], r, eps, f, lo, hi);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_parallel_loop("n", "N")
    self.gen_add_code_line("T g = static_cast<T>(0);")
    self.gen_add_code_line("for (int m = 0; m < M; ++m) g += s_A[m + M * n] * s_scratch[m];")
    self.gen_add_code_line("if (ACCUMULATE) { s_grad[n] += g; } else { s_grad[n] = g; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc("linear_feature_cost_hessian: Gauss-Newton Hessian = A^T diag(a''(r)) A  (column-major N x N)",
                          ["s_scratch (>= M) holds the per-row curvatures between the two passes.",
                           "ACCUMULATE=false overwrites s_hess; ACCUMULATE=true adds into it."],
                          ["s_hess is the dense Hessian output (size N*N, column-major)"] + doc, None)
    self.gen_add_code_line("template <typename T, int M, int N, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void linear_feature_cost_hessian(T *s_hess, const T *s_v, const T *s_A, const T *s_b, const T *s_target, const T *s_W, " + args + ") {", True)
    self.gen_add_parallel_loop("m", "M")
    for l in residual:
        self.gen_add_code_line(l)
    self.gen_add_code_line("s_scratch[m] = activation_curvature<T>(activation, s_W[m], r, eps, f, lo, hi);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_parallel_loop("ind", "N * N")
    self.gen_add_code_line("int row = ind % N;")
    self.gen_add_code_line("int col = ind / N;")
    self.gen_add_code_line("T h = static_cast<T>(0);")
    self.gen_add_code_line("for (int m = 0; m < M; ++m) h += s_A[m + M * row] * s_scratch[m] * s_A[m + M * col];")
    self.gen_add_code_line("if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


# ---------------------------------------------------------------------------
# gravity torque (caller-scratch RNEA at zero velocity)
# ---------------------------------------------------------------------------

def gen_gravity_torque(self):
    from ._centroidal import _id_bias_inner_temp_mem_size
    n = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    nb_vaf = self.robot.get_num_joints() if self.robot_has_mimic_joints() else n
    extra = [("s_vaf", 18 * nb_vaf), ("s_qd0", nv)]
    self.gen_add_code_line("#define GRID_PLANT_HAS_GRAVITY_TORQUE 1")
    self.gen_add_func_doc(
        "gravity_torque: RAW caller-scratch generalized gravity g(q) = RNEA(q, 0, 0) (size NUM_VEL)",
        ["Caller-scratch INNER (the ee_pos pattern): lays out the RNEA arena from s_scratch and calls "
         "grid::inverse_dynamics_inner at zero velocity and acceleration, so it is callable from another "
         "kernel's block. Writes g(q) into s_g; a consumer makes it a knot's torque target (u - g(q_ref)).",
         "s_scratch must hold GRAVITY_TORQUE_DYNAMIC_SHARED_MEM_BYTES<T>() bytes, 16B aligned."],
        ["s_g is the generalized gravity output (size NUM_VEL)",
         "s_q is the joint position vector (size NUM_POS)",
         "s_scratch is caller shared scratch",
         "d_robotModel is the GPU model helpers",
         "gravity is the gravity constant"], None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void gravity_torque(T *s_g, const T *s_q, T *s_scratch, const grid::robotModel<T> *d_robotModel, const T gravity) {", True)
    self.gen_add_code_line("using namespace grid;")
    layout = self.gen_XImats_helpers_temp_shared_memory_code(
        _id_bias_inner_temp_mem_size(self), extra_t_buffers=extra, include_linalg_scratch=True, arena_base_expr="s_scratch")
    if layout.topology_count > 0:
        self.gen_add_code_line("load_topology_helpers<T>(s_topology_helpers, d_robotModel);")
    self.gen_load_update_XImats_helpers_function_call()
    self.gen_add_code_line("glass::set_const<T, " + str(nv) + ">(static_cast<T>(0), s_qd0);")
    self.gen_add_sync()
    self.gen_inverse_dynamics_inner_function_call(compute_c=True, use_qdd_input=False,
                                                  updated_var_names=dict(s_c_name="s_g", s_qd_name="s_qd0", d_f_ext_name="nullptr"))
    self.gen_add_sync()
    self.gen_add_end_function()
    t_count = sum(int(sz) for _, sz in layout.t_buffers) + int(layout.ximat_size) + int(layout.temp_mem_size)
    self.gen_add_code_line("template <typename T> __host__ __device__ constexpr size_t GRAVITY_TORQUE_DYNAMIC_SHARED_MEM_BYTES() "
                           "{ return grid::grid_shared_arena_bytes<T>(" + str(t_count) + ", grid::TOPOLOGY_HELPERS_COUNT, grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); }")


# ---------------------------------------------------------------------------
# named cost frames: evaluator, SO(3) helpers, position / orientation / velocity
# ---------------------------------------------------------------------------

def _gen_so3_helpers(self):
    """log(R) on the canonical branch and Jr(phi)^-1, serial 3x3 helpers (column-major)."""
    self.gen_add_code_line("// SO(3) helpers for the orientation residual (serial, column-major 3x3)")
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line("void plant_so3_log(const T *R, T *phi) {", True)
    self.gen_add_code_line("T c = (R[0] + R[4] + R[8] - static_cast<T>(1)) * static_cast<T>(0.5);")
    self.gen_add_code_line("c = c > static_cast<T>(1) ? static_cast<T>(1) : (c < static_cast<T>(-1) ? static_cast<T>(-1) : c);")
    self.gen_add_code_line("T theta = acos(c);")
    self.gen_add_code_line("T w0 = R[5] - R[7], w1 = R[6] - R[2], w2 = R[1] - R[3];")
    self.gen_add_code_line("if (theta < static_cast<T>(1e-6)) { phi[0] = static_cast<T>(0.5) * w0; phi[1] = static_cast<T>(0.5) * w1; phi[2] = static_cast<T>(0.5) * w2; return; }")
    self.gen_add_code_line("if (theta < static_cast<T>(3.14159265358979323846) - static_cast<T>(1e-4)) {", True)
    self.gen_add_code_line("T k = theta / (static_cast<T>(2) * sin(theta));")
    self.gen_add_code_line("phi[0] = k * w0; phi[1] = k * w1; phi[2] = k * w2; return;")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("// near pi: axis from the symmetric part (R + I)/2, sign from the antisymmetric part")
    self.gen_add_code_line("T S0 = static_cast<T>(0.5) * (R[0] + static_cast<T>(1)), S4 = static_cast<T>(0.5) * (R[4] + static_cast<T>(1)), S8 = static_cast<T>(0.5) * (R[8] + static_cast<T>(1));")
    self.gen_add_code_line("T a0, a1, a2;")
    self.gen_add_code_line("if (S0 >= S4 && S0 >= S8) { T s = sqrt(S0 > static_cast<T>(1e-18) ? S0 : static_cast<T>(1e-18)); a0 = S0 / s; a1 = static_cast<T>(0.5) * (R[1] + R[3]) / s; a2 = static_cast<T>(0.5) * (R[2] + R[6]) / s; }")
    self.gen_add_code_line("else if (S4 >= S8) { T s = sqrt(S4 > static_cast<T>(1e-18) ? S4 : static_cast<T>(1e-18)); a0 = static_cast<T>(0.5) * (R[1] + R[3]) / s; a1 = S4 / s; a2 = static_cast<T>(0.5) * (R[5] + R[7]) / s; }")
    self.gen_add_code_line("else { T s = sqrt(S8 > static_cast<T>(1e-18) ? S8 : static_cast<T>(1e-18)); a0 = static_cast<T>(0.5) * (R[2] + R[6]) / s; a1 = static_cast<T>(0.5) * (R[5] + R[7]) / s; a2 = S8 / s; }")
    self.gen_add_code_line("T n = sqrt(a0 * a0 + a1 * a1 + a2 * a2); a0 /= n; a1 /= n; a2 /= n;")
    self.gen_add_code_line("if (a0 * w0 + a1 * w1 + a2 * w2 < static_cast<T>(0)) { a0 = -a0; a1 = -a1; a2 = -a2; }")
    self.gen_add_code_line("phi[0] = theta * a0; phi[1] = theta * a1; phi[2] = theta * a2;")
    self.gen_add_end_function()
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line("void plant_so3_right_jacobian_inv(const T *phi, T *J) {", True)
    self.gen_add_code_line("T theta = sqrt(phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]);")
    self.gen_add_code_line("T coef = (theta < static_cast<T>(1e-5)) ? (static_cast<T>(1) / static_cast<T>(12))")
    self.gen_add_code_line("    : (static_cast<T>(1) / (theta * theta) - (static_cast<T>(1) + cos(theta)) / (static_cast<T>(2) * theta * sin(theta)));")
    self.gen_add_code_line("// S = [phi]x column-major; J = I + 1/2 S + coef S^2")
    self.gen_add_code_line("T S[9] = { static_cast<T>(0), phi[2], -phi[1], -phi[2], static_cast<T>(0), phi[0], phi[1], -phi[0], static_cast<T>(0) };")
    self.gen_add_code_line("for (int c = 0; c < 3; ++c) for (int r = 0; r < 3; ++r) {", True)
    self.gen_add_code_line("T s2 = S[r] * S[3 * c] + S[r + 3] * S[3 * c + 1] + S[r + 6] * S[3 * c + 2];")
    self.gen_add_code_line("J[r + 3 * c] = ((r == c) ? static_cast<T>(1) : static_cast<T>(0)) + static_cast<T>(0.5) * S[r + 3 * c] + coef * s2;")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


def gen_frame_eval(self):
    """`frame_eval`: caller-scratch evaluator of a baked cost frame: its world
    placement X_f (16, column-major, the fixed-joint offset applied) and the
    LOCAL_WORLD_ALIGNED geometric Jacobian at the frame point (6 x NUM_VEL)."""
    from ._frame_jacobian import _frame_jacobian_inner_temp_mem_size
    nv = self.robot.get_num_vel()
    NJ = self.robot.get_num_joints()
    frames = self._cost_frames
    self.gen_add_code_line("#define GRID_PLANT_HAS_COST_FRAMES 1")
    self.gen_add_code_line("const int GRID_PLANT_NUM_COST_FRAMES = " + str(len(frames)) + ";")
    self.gen_add_code_line("// cost frames (index -> name): " + ", ".join(str(i) + " = " + nm for i, (nm, _, _) in enumerate(frames)))
    self.gen_add_code_line("const int GRID_PLANT_FRAME_SCRATCH_COUNT = " + str(16 + 12 * nv) + ";  // [X_f(16) | J(6*NV) | dJqd(6*NV)]")
    self.gen_add_func_doc(
        "frame_eval: RAW caller-scratch cost-frame evaluator (world placement + LOCAL_WORLD_ALIGNED Jacobian at the frame point)",
        ["Lays out the XmatsHom arena from s_scratch, loads the transforms once, runs grid::frame_jacobian_inner at the frame's joint and "
         "shifts the linear rows to the frame point (a fixed joint's offset), so a named frame such as an elbow or a tool is addressed by its baked index.",
         "s_X receives the 4x4 world placement (column-major), s_J the 6 x NUM_VEL Jacobian [linear; angular] with world axes at the frame point.",
         "s_scratch must hold >= FRAME_EVAL_DYNAMIC_SHARED_MEM_BYTES<T>() bytes, 16B aligned; after the call s_scratch[0 .. 16*NUM_JOINTS) holds every joint's world transform (frame_velocity_cost reads it)."],
        ["s_J is the 6 x NUM_VEL Jacobian output; s_X is the 16-element placement output",
         "cf is the cost-frame index (0 .. GRID_PLANT_NUM_COST_FRAMES - 1)",
         "s_q is the joint position vector (size NUM_POS)",
         "s_scratch is caller shared scratch; d_robotModel is the GPU model helpers"], None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void frame_eval(T *s_J, T *s_X, const int cf, const T *s_q, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    self.gen_add_code_line("using namespace grid;")
    self.gen_bake_const_array("cf_jid", [jid for _, jid, _ in frames], "int")
    self.gen_bake_const_array("cf_offset", [v for _, _, off in frames for v in off], "T")
    layout = self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _frame_jacobian_inner_temp_mem_size(self), include_linalg_scratch=True,
        linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()", arena_base_expr="s_scratch")
    self._frame_eval_t_count = sum(int(sz) for _, sz in layout.t_buffers) + int(layout.ximat_size) + int(layout.temp_mem_size)
    # s_temp (the world transforms) follows the T buffers and the transform block in the carve
    self._frame_eval_xworld_offset = sum(int(sz) for _, sz in layout.t_buffers) + int(layout.ximat_size)
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_add_code_line("const int jid = cf_jid[cf];")
    self.gen_add_code_line("frame_jacobian_inner<T>(s_J, jid, 2, s_q, s_XmatsHom, d_robotModel, s_temp);")
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("// X_f = Xworld[jid] * X_offset; then shift the linear rows from the joint origin to the frame point")
    self.gen_add_code_line("const T *Xj = &s_temp[16 * jid]; const T *O = &cf_offset[16 * cf];")
    self.gen_add_code_line("for (int c = 0; c < 4; ++c) for (int r = 0; r < 4; ++r) { s_X[r + 4 * c] = Xj[r] * O[4 * c] + Xj[r + 4] * O[4 * c + 1] + Xj[r + 8] * O[4 * c + 2] + Xj[r + 12] * O[4 * c + 3]; }")
    self.gen_add_code_line("T dx = s_X[12] - Xj[12], dy = s_X[13] - Xj[13], dz = s_X[14] - Xj[14];")
    self.gen_add_code_line("for (int vi = 0; vi < " + str(nv) + "; ++vi) { T *Jc = &s_J[6 * vi]; T w0 = Jc[3], w1 = Jc[4], w2 = Jc[5]; Jc[0] += w1 * dz - w2 * dy; Jc[1] += w2 * dx - w0 * dz; Jc[2] += w0 * dy - w1 * dx; }")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()
    self.gen_add_code_line("template <typename T> __host__ __device__ constexpr size_t FRAME_EVAL_DYNAMIC_SHARED_MEM_BYTES() "
                           "{ return grid::grid_shared_arena_bytes<T>(" + str(self._frame_eval_t_count) + ", grid::TOPOLOGY_HELPERS_COUNT, grid::GRID_EE_LINALG_SHARED_BYTES<T>()); }")


def gen_frame_position_cost(self):
    nq = self.robot.get_num_pos(); nv = self.robot.get_num_vel(); nx = nq + nv
    NV, NX = str(nv), str(nx)
    doc = ["cf is the cost-frame index; s_q is the joint position vector (size NUM_POS)",
           "s_p_des is the desired frame position (3); s_W is the per-axis weight (3)",
           "activation / eps as for state_cost; " + _BOUNDS_DOC + " (bounds on the position)",
           "s_frame is scratch of GRID_PLANT_FRAME_SCRATCH_COUNT (holds X_f and the Jacobian after the call); s_scratch is the frame_eval arena",
           "d_robotModel is the GPU model"]
    args = "const int cf, const T *s_p_des, const T *s_W, const int activation, const T eps, const T *s_lower, const T *s_upper, T *s_frame, T *s_scratch, const grid::robotModel<T> *d_robotModel"
    head = ["T *s_X = s_frame; T *s_J = s_frame + 16;", "frame_eval<T>(s_J, s_X, cf, s_q, s_scratch, d_robotModel);"]
    self.gen_add_func_doc("frame_position_cost: value = sum_r a(p_f,r(q) - p_des_r; W[r], eps) on a baked cost frame",
                          ["The end-effector frame (cf = 0) under the quadratic activation reproduces ee_pos_cost. ACCUMULATE=false overwrites s_out[0]; true adds."],
                          ["s_out is the scalar cost output (s_out[0])"] + doc, None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void frame_position_cost(T *s_out, const T *s_q, " + args + ") {", True)
    for l in head:
        self.gen_add_code_line(l)
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int r = 0; r < 3; ++r) { " + _bound_reads("r") + " T p = s_X[12 + r]; acc += activation_value<T>(activation, s_W[r], p - s_p_des[r], eps, p, lo, hi); }")
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc("frame_position_cost_gradient: grad_x = [J_p^T a'(r) ; 0] over x = [q; qd]",
                          ["The qd-block (entries NUM_VEL.." + str(nx - 1) + ") is set to zero when !ACCUMULATE. ACCUMULATE=false overwrites s_grad; true adds."],
                          ["s_grad is the gradient over x (size " + NX + ")"] + doc, None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void frame_position_cost_gradient(T *s_grad, const T *s_q, " + args + ") {", True)
    for l in head:
        self.gen_add_code_line(l)
    self.gen_add_parallel_loop("i", NV)
    self.gen_add_code_line("T g = static_cast<T>(0);")
    self.gen_add_code_line("for (int r = 0; r < 3; ++r) { " + _bound_reads("r") + " T p = s_X[12 + r]; g += s_J[6 * i + r] * activation_slope<T>(activation, s_W[r], p - s_p_des[r], eps, p, lo, hi); }")
    self.gen_add_code_line("if (ACCUMULATE) { s_grad[i] += g; } else { s_grad[i] = g; }")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if (!ACCUMULATE) {", True)
    self.gen_add_parallel_loop("i", str(nx - nv))
    self.gen_add_code_line("s_grad[" + NV + " + i] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc("frame_position_cost_hessian: Gauss-Newton Hessian = J_p^T diag(a''(r)) J_p in the q-block of an NX x NX column-major matrix",
                          ["The qd rows and columns are zero. ACCUMULATE=false overwrites s_hess; true adds."],
                          ["s_hess is the dense Hessian output (size " + NX + "*" + NX + ", column-major)"] + doc, None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void frame_position_cost_hessian(T *s_hess, const T *s_q, " + args + ") {", True)
    for l in head:
        self.gen_add_code_line(l)
    self.gen_add_parallel_loop("ind", str(nx * nx))
    self.gen_add_code_line("int row = ind % " + NX + "; int col = ind / " + NX + ";")
    self.gen_add_code_line("T h = static_cast<T>(0);")
    self.gen_add_code_line("if (row < " + NV + " && col < " + NV + ") { for (int r = 0; r < 3; ++r) { " + _bound_reads("r") + " T p = s_X[12 + r]; h += s_J[6 * row + r] * activation_curvature<T>(activation, s_W[r], p - s_p_des[r], eps, p, lo, hi) * s_J[6 * col + r]; } }")
    self.gen_add_code_line("if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


def gen_frame_orientation_cost(self):
    nq = self.robot.get_num_pos(); nv = self.robot.get_num_vel(); nx = nq + nv
    NV, NX = str(nv), str(nx)
    doc = ["cf is the cost-frame index; s_q is the joint position vector (size NUM_POS)",
           "s_R_des is the desired rotation (3x3 column-major, 9); s_W is the per-axis weight (3)",
           "activation / eps as for state_cost (no bounds: the residual is a rotation vector)",
           "s_frame is scratch of GRID_PLANT_FRAME_SCRATCH_COUNT; after the call its angular rows hold J_e and s_frame[0..2] the residual e; s_scratch is the frame_eval arena",
           "d_robotModel is the GPU model"]
    args = "const int cf, const T *s_R_des, const T *s_W, const int activation, const T eps, T *s_frame, T *s_scratch, const grid::robotModel<T> *d_robotModel"
    # evaluate: X_f, J; then serially e = log(R_des^T R_f), J_e = Jr(e)^-1 R_f^T J_w  (written over the angular rows), e kept in s_frame[0..2]
    head = ["T *s_X = s_frame; T *s_J = s_frame + 16;", "frame_eval<T>(s_J, s_X, cf, s_q, s_scratch, d_robotModel);"]
    serial = [
        "T R[9]; for (int c = 0; c < 3; ++c) for (int r = 0; r < 3; ++r) R[r + 3 * c] = s_X[r + 4 * c];",
        "T E[9]; for (int c = 0; c < 3; ++c) for (int r = 0; r < 3; ++r) E[r + 3 * c] = s_R_des[3 * r] * R[3 * c] + s_R_des[3 * r + 1] * R[3 * c + 1] + s_R_des[3 * r + 2] * R[3 * c + 2];  // (R_des^T R_f)[r][c]",
        "T e[3]; plant_so3_log<T>(E, e);",
        "T Ji[9]; plant_so3_right_jacobian_inv<T>(e, Ji);",
        "for (int vi = 0; vi < " + NV + "; ++vi) {",
        "    T *Jc = &s_J[6 * vi];",
        "    T wl0 = R[0] * Jc[3] + R[1] * Jc[4] + R[2] * Jc[5], wl1 = R[3] * Jc[3] + R[4] * Jc[4] + R[5] * Jc[5], wl2 = R[6] * Jc[3] + R[7] * Jc[4] + R[8] * Jc[5];",
        "    Jc[3] = Ji[0] * wl0 + Ji[3] * wl1 + Ji[6] * wl2; Jc[4] = Ji[1] * wl0 + Ji[4] * wl1 + Ji[7] * wl2; Jc[5] = Ji[2] * wl0 + Ji[5] * wl1 + Ji[8] * wl2;",
        "}",
        "s_X[0] = e[0]; s_X[1] = e[1]; s_X[2] = e[2];",
    ]
    self.gen_add_func_doc("frame_orientation_cost: value = sum_r a(e_r; W[r], eps), e = log(R_des^T R_f(q)) on a baked cost frame",
                          ["Log-map residual on the canonical branch |e| <= pi (no Euler chart); its Jacobian is Jr(e)^-1 times the frame's body-axes angular Jacobian.",
                           "ACCUMULATE=false overwrites s_out[0]; true adds."],
                          ["s_out is the scalar cost output (s_out[0])"] + doc, None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void frame_orientation_cost(T *s_out, const T *s_q, " + args + ") {", True)
    for l in head:
        self.gen_add_code_line(l)
    self.gen_add_serial_ops()
    for l in serial:
        self.gen_add_code_line(l)
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int r = 0; r < 3; ++r) acc += activation_value<T>(activation, s_W[r], e[r], eps);")
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    for kind in ("gradient", "hessian"):
        if kind == "gradient":
            self.gen_add_func_doc("frame_orientation_cost_gradient: grad_x = [J_e^T a'(e) ; 0] over x = [q; qd]",
                                  ["The qd-block is set to zero when !ACCUMULATE. ACCUMULATE=false overwrites s_grad; true adds."],
                                  ["s_grad is the gradient over x (size " + NX + ")"] + doc, None)
            self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
            self.gen_add_code_line("__device__")
            self.gen_add_code_line("void frame_orientation_cost_gradient(T *s_grad, const T *s_q, " + args + ") {", True)
        else:
            self.gen_add_func_doc("frame_orientation_cost_hessian: Gauss-Newton Hessian = J_e^T diag(a''(e)) J_e in the q-block of an NX x NX column-major matrix",
                                  ["The qd rows and columns are zero. ACCUMULATE=false overwrites s_hess; true adds."],
                                  ["s_hess is the dense Hessian output (size " + NX + "*" + NX + ", column-major)"] + doc, None)
            self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
            self.gen_add_code_line("__device__")
            self.gen_add_code_line("void frame_orientation_cost_hessian(T *s_hess, const T *s_q, " + args + ") {", True)
        for l in head:
            self.gen_add_code_line(l)
        self.gen_add_serial_ops()
        for l in serial:
            self.gen_add_code_line(l)
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        if kind == "gradient":
            self.gen_add_parallel_loop("i", NV)
            self.gen_add_code_line("T g = static_cast<T>(0);")
            self.gen_add_code_line("for (int r = 0; r < 3; ++r) g += s_J[6 * i + 3 + r] * activation_slope<T>(activation, s_W[r], s_X[r], eps);")
            self.gen_add_code_line("if (ACCUMULATE) { s_grad[i] += g; } else { s_grad[i] = g; }")
            self.gen_add_end_control_flow()
            self.gen_add_code_line("if (!ACCUMULATE) {", True)
            self.gen_add_parallel_loop("i", str(nx - nv))
            self.gen_add_code_line("s_grad[" + NV + " + i] = static_cast<T>(0);")
            self.gen_add_end_control_flow()
            self.gen_add_end_control_flow()
        else:
            self.gen_add_parallel_loop("ind", str(nx * nx))
            self.gen_add_code_line("int row = ind % " + NX + "; int col = ind / " + NX + ";")
            self.gen_add_code_line("T h = static_cast<T>(0);")
            self.gen_add_code_line("if (row < " + NV + " && col < " + NV + ") { for (int r = 0; r < 3; ++r) h += s_J[6 * row + 3 + r] * activation_curvature<T>(activation, s_W[r], s_X[r], eps) * s_J[6 * col + 3 + r]; }")
            self.gen_add_code_line("if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }")
            self.gen_add_end_control_flow()
        self.gen_add_end_function()


def gen_frame_velocity_cost(self):
    """Frame twist residual r = J(q) qd - v_des (LOCAL_WORLD_ALIGNED, 6) with the
    residual Jacobian [d(J qd)/dq | J] over x = [q; qd]. Fixed base: the
    configuration derivative is the rigid-rotation form over the baked column
    table of frame_jacobian_inner (see RBDReference._frame_velocity_q_derivative)."""
    from ._frame_jacobian import frame_jacobian_jobs
    nq = self.robot.get_num_pos(); nv = self.robot.get_num_vel(); nx = nq + nv
    NV, NX = str(nv), str(nx)
    jobs = frame_jacobian_jobs(self)
    self.gen_add_code_line("#define GRID_PLANT_HAS_FRAME_VELOCITY_COST 1")
    doc = ["cf is the cost-frame index; s_q / s_qd are the joint positions and velocities",
           "s_v_des is the desired twist [linear; angular] (6) with world axes at the frame point; s_W is the per-entry weight (6)",
           "activation / eps as for state_cost; " + _BOUNDS_DOC + " (bounds on the twist entries)",
           "s_frame is scratch of GRID_PLANT_FRAME_SCRATCH_COUNT ([X_f | J | dJqd]); s_scratch is the frame_eval arena",
           "d_robotModel is the GPU model"]
    args = "const int cf, const T *s_v_des, const T *s_W, const int activation, const T eps, const T *s_lower, const T *s_upper, T *s_frame, T *s_scratch, const grid::robotModel<T> *d_robotModel"
    # the derivative pass: thread 0 walks the (column j, mover m) pairs of the frame's chain
    self.gen_add_func_doc("frame_velocity_eval: frame_eval plus the twist J qd (into s_frame[0..5]) and its configuration derivative d(J qd)/dq (6 x NUM_VEL, after the Jacobian)",
                          ["Column j of the world-axis Jacobian is a_j x (p_f - p_j) (revolute) or l_j (prismatic). A joint m moves everything downstream of it rigidly: "
                           "for m an ancestor-or-self of j the axis, origin and frame point, for m downstream of j the frame point alone. Serial over the baked chain pairs.",
                           "Fixed base only (the floating root has no single axis); the world transforms are read back from the frame_eval arena."],
                          ["s_frame / s_scratch / cf / s_q / s_qd / d_robotModel as for frame_velocity_cost"], None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void frame_velocity_eval(const int cf, const T *s_q, const T *s_qd, T *s_frame, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    self.gen_add_code_line("T *s_X = s_frame; T *s_J = s_frame + 16; T *s_dJ = s_frame + 16 + 6 * " + NV + ";")
    self.gen_add_code_line("frame_eval<T>(s_J, s_X, cf, s_q, s_scratch, d_robotModel);")
    self.gen_bake_const_array("fv_target", [j[0] for j in jobs], "int")
    self.gen_bake_const_array("fv_jj", [j[1] for j in jobs], "int")
    self.gen_bake_const_array("fv_vi", [j[2] for j in jobs], "int")
    self.gen_bake_const_array("fv_ang", [v for j in jobs for v in j[3]], "T")
    self.gen_bake_const_array("fv_lin", [v for j in jobs for v in j[4]], "T")
    self.gen_bake_const_array("fv_alpha", [j[5] for j in jobs], "T")
    self.gen_bake_const_array("fv_cf_jid", [jid for _, jid, _ in self._cost_frames], "int")
    self.gen_add_code_line("const T *s_Xworld = reinterpret_cast<const T *>(s_scratch) + " + str(self._frame_eval_xworld_offset) + ";  // world transforms left by frame_eval")
    self.gen_add_parallel_loop("ind", str(6 * nv))
    self.gen_add_code_line("s_dJ[ind] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("const int tj = fv_cf_jid[cf];")
    self.gen_add_code_line("const T pfx = s_X[12], pfy = s_X[13], pfz = s_X[14];")
    self.gen_add_code_line("for (int r = 0; r < 6; ++r) { T acc = static_cast<T>(0); for (int vi = 0; vi < " + NV + "; ++vi) acc += s_J[6 * vi + r] * s_qd[vi]; s_X[r] = acc; }  // the twist J qd into s_X[0..5] (the placement's rotation block is no longer needed)")
    self.gen_add_code_line("for (int a = 0; a < " + str(len(jobs)) + "; ++a) {", True)
    self.gen_add_code_line("if (fv_target[a] != tj) continue;")
    self.gen_add_code_line("const int jj = fv_jj[a]; const int vi = fv_vi[a]; const T *Xj = &s_Xworld[16 * jj];")
    self.gen_add_code_line("const bool rot_j = (fv_ang[3 * a] * fv_ang[3 * a] + fv_ang[3 * a + 1] * fv_ang[3 * a + 1] + fv_ang[3 * a + 2] * fv_ang[3 * a + 2]) > static_cast<T>(0.25);")
    self.gen_add_code_line("const T *axl = rot_j ? &fv_ang[3 * a] : &fv_lin[3 * a];")
    self.gen_add_code_line("T aj0 = Xj[0] * axl[0] + Xj[4] * axl[1] + Xj[8] * axl[2], aj1 = Xj[1] * axl[0] + Xj[5] * axl[1] + Xj[9] * axl[2], aj2 = Xj[2] * axl[0] + Xj[6] * axl[1] + Xj[10] * axl[2];")
    self.gen_add_code_line("T dj0 = pfx - Xj[12], dj1 = pfy - Xj[13], dj2 = pfz - Xj[14];")
    self.gen_add_code_line("const T w = fv_alpha[a] * s_qd[vi];")
    self.gen_add_code_line("for (int b = 0; b < " + str(len(jobs)) + "; ++b) {", True)
    self.gen_add_code_line("if (fv_target[b] != tj) continue;")
    self.gen_add_code_line("const int mm = fv_jj[b]; const int vm = fv_vi[b]; const T *Xm = &s_Xworld[16 * mm];")
    self.gen_add_code_line("const bool rot_m = (fv_ang[3 * b] * fv_ang[3 * b] + fv_ang[3 * b + 1] * fv_ang[3 * b + 1] + fv_ang[3 * b + 2] * fv_ang[3 * b + 2]) > static_cast<T>(0.25);")
    self.gen_add_code_line("const T *axm = rot_m ? &fv_ang[3 * b] : &fv_lin[3 * b];")
    self.gen_add_code_line("T am0 = Xm[0] * axm[0] + Xm[4] * axm[1] + Xm[8] * axm[2], am1 = Xm[1] * axm[0] + Xm[5] * axm[1] + Xm[9] * axm[2], am2 = Xm[2] * axm[0] + Xm[6] * axm[1] + Xm[10] * axm[2];")
    self.gen_add_code_line("const T sw = fv_alpha[b] * w;")
    self.gen_add_code_line("T l0 = static_cast<T>(0), l1 = static_cast<T>(0), l2 = static_cast<T>(0), g0 = static_cast<T>(0), g1 = static_cast<T>(0), g2 = static_cast<T>(0);")
    self.gen_add_code_line("if (mm <= jj) {", True)
    self.gen_add_code_line("if (rot_j && rot_m) {", True)
    self.gen_add_code_line("// d_axis = am x aj; d_lin = d_axis x d + aj x (am x d)")
    self.gen_add_code_line("T x0 = am1 * aj2 - am2 * aj1, x1 = am2 * aj0 - am0 * aj2, x2 = am0 * aj1 - am1 * aj0;")
    self.gen_add_code_line("T y0 = am1 * dj2 - am2 * dj1, y1 = am2 * dj0 - am0 * dj2, y2 = am0 * dj1 - am1 * dj0;")
    self.gen_add_code_line("l0 = (x1 * dj2 - x2 * dj1) + (aj1 * y2 - aj2 * y1); l1 = (x2 * dj0 - x0 * dj2) + (aj2 * y0 - aj0 * y2); l2 = (x0 * dj1 - x1 * dj0) + (aj0 * y1 - aj1 * y0);")
    self.gen_add_code_line("g0 = x0; g1 = x1; g2 = x2;")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("else if (!rot_j && rot_m) { l0 = am1 * aj2 - am2 * aj1; l1 = am2 * aj0 - am0 * aj2; l2 = am0 * aj1 - am1 * aj0; }")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("else if (rot_j) {", True)
    self.gen_add_code_line("// m downstream of j: only the frame point moves, by am x (p_f - p_m) (revolute m) or by the prismatic axis")
    self.gen_add_code_line("T dm0 = pfx - Xm[12], dm1 = pfy - Xm[13], dm2 = pfz - Xm[14];")
    self.gen_add_code_line("T p0 = rot_m ? (am1 * dm2 - am2 * dm1) : am0, p1 = rot_m ? (am2 * dm0 - am0 * dm2) : am1, p2 = rot_m ? (am0 * dm1 - am1 * dm0) : am2;")
    self.gen_add_code_line("l0 = aj1 * p2 - aj2 * p1; l1 = aj2 * p0 - aj0 * p2; l2 = aj0 * p1 - aj1 * p0;")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("T *D = &s_dJ[6 * vm]; D[0] += sw * l0; D[1] += sw * l1; D[2] += sw * l2; D[3] += sw * g0; D[4] += sw * g1; D[5] += sw * g2;")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()

    head = ["frame_velocity_eval<T>(cf, s_q, s_qd, s_frame, s_scratch, d_robotModel);",
            "const T *s_v = s_frame; const T *s_J = s_frame + 16; const T *s_dJ = s_frame + 16 + 6 * " + NV + ";"]
    self.gen_add_func_doc("frame_velocity_cost: value = sum_r a(v_r - v_des_r; W[r], eps), v = J(q) qd the frame twist [linear; angular] with world axes at the frame point",
                          ["ACCUMULATE=false overwrites s_out[0]; true adds. Fixed base."],
                          ["s_out is the scalar cost output (s_out[0])"] + doc, None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void frame_velocity_cost(T *s_out, const T *s_q, const T *s_qd, " + args + ") {", True)
    for l in head:
        self.gen_add_code_line(l)
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int r = 0; r < 6; ++r) { " + _bound_reads("r") + " acc += activation_value<T>(activation, s_W[r], s_v[r] - s_v_des[r], eps, s_v[r], lo, hi); }")
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    jx = "((c < " + NV + ") ? s_dJ[6 * c + r] : s_J[6 * (c - " + NV + ") + r])"
    self.gen_add_func_doc("frame_velocity_cost_gradient: grad_x = [d(J qd)/dq^T a'(r) ; J^T a'(r)] over x = [q; qd]",
                          ["ACCUMULATE=false overwrites s_grad; true adds."],
                          ["s_grad is the gradient over x (size " + NX + ")"] + doc, None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void frame_velocity_cost_gradient(T *s_grad, const T *s_q, const T *s_qd, " + args + ") {", True)
    for l in head:
        self.gen_add_code_line(l)
    self.gen_add_parallel_loop("c", NX)
    self.gen_add_code_line("T g = static_cast<T>(0);")
    self.gen_add_code_line("for (int r = 0; r < 6; ++r) { " + _bound_reads("r") + " g += " + jx + " * activation_slope<T>(activation, s_W[r], s_v[r] - s_v_des[r], eps, s_v[r], lo, hi); }")
    self.gen_add_code_line("if (ACCUMULATE) { s_grad[c] += g; } else { s_grad[c] = g; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc("frame_velocity_cost_hessian: Gauss-Newton Hessian = J_x^T diag(a''(r)) J_x, J_x = [d(J qd)/dq | J], NX x NX column-major",
                          ["ACCUMULATE=false overwrites s_hess; true adds."],
                          ["s_hess is the dense Hessian output (size " + NX + "*" + NX + ", column-major)"] + doc, None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void frame_velocity_cost_hessian(T *s_hess, const T *s_q, const T *s_qd, " + args + ") {", True)
    for l in head:
        self.gen_add_code_line(l)
    self.gen_add_parallel_loop("ind", str(nx * nx))
    self.gen_add_code_line("int row = ind % " + NX + "; int col = ind / " + NX + ";")
    self.gen_add_code_line("T h = static_cast<T>(0);")
    self.gen_add_code_line("for (int r = 0; r < 6; ++r) { " + _bound_reads("r") + " T cu = activation_curvature<T>(activation, s_W[r], s_v[r] - s_v_des[r], eps, s_v[r], lo, hi); int c = row; T jr = " + jx + "; c = col; T jc = " + jx + "; h += jr * cu * jc; }")
    self.gen_add_code_line("if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


# ---------------------------------------------------------------------------
# the menu preset
# ---------------------------------------------------------------------------

def gen_tracking_cost_menu(self):
    nq = self.robot.get_num_pos(); nv = self.robot.get_num_vel(); nu = nv; nx = nq + nv
    NQ, NU, NX = str(nq), str(nu), str(nx)
    self.gen_add_code_line("#define GRID_PLANT_HAS_COST_MENU 1")
    self.gen_add_func_doc("cost_menu_params: the per-knot buffers of tracking_cost_menu (every pointer a runtime buffer; a term whose activation is GRID_PLANT_ACTIVATION_NONE is skipped)",
                          ["x_des / u_des / Q / R and the barrier bound-and-mu triples as tracking_cost; ee_des / W are the cost-frame CF position target and weight.",
                           "activation (GRID_PLANT_MENU_TERMS ints) and eps (same length) select and parameterize every term, order GRID_PLANT_MENU_*.",
                           "x_lower / x_upper, ub_lower / ub_upper, lx_lower / lx_upper, lu_lower / lu_upper, v_lower / v_upper, cf_lower / cf_upper are the barrier-activation bounds of the matching term (nullptr when unused).",
                           "Ax / bx / tx / Wx (MX rows over NX) and Au / bu / tu / Wu (MU rows over NU) are the linear features; R_des / Wo the orientation of frame CF; v_des / Wv its twist; cf_des / Wcf the position of frame CF2."],
                          [], None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("struct cost_menu_params {", True)
    for line in ["const T *x_des; const T *u_des; const T *ee_des; const T *Q; const T *R; const T *W;",
                 "const T *q_lower; const T *q_upper; T mu_q; const T *qd_lower; const T *qd_upper; T mu_qd; const T *u_lower; const T *u_upper; T mu_u;",
                 "const int *activation; const T *eps;",
                 "const T *x_lower; const T *x_upper; const T *ub_lower; const T *ub_upper;",
                 "const T *Ax; const T *bx; const T *tx; const T *Wx; const T *lx_lower; const T *lx_upper;",
                 "const T *Au; const T *bu; const T *tu; const T *Wu; const T *lu_lower; const T *lu_upper;",
                 "const T *R_des; const T *Wo;",
                 "const T *v_des; const T *Wv; const T *v_lower; const T *v_upper;",
                 "const T *cf_des; const T *Wcf; const T *cf_lower; const T *cf_upper;"]:
        self.gen_add_code_line(line)
    self.indent_level -= 1
    self.gen_add_code_line("};")
    doc = ["s_x / s_u are the current state [q; qd] and control (sizes " + NX + " / " + NU + "); p holds every per-knot buffer",
           "MX / MU are the linear feature row counts (0 omits the term); CF is the cost frame of the position, orientation and twist terms (0 = the end-effector target); CF2 (-1 = none) the frame of the second position term",
           "s_frame is GRID_PLANT_FRAME_SCRATCH_COUNT scratch; s_scratch must hold the frame_eval arena (FRAME_EVAL_DYNAMIC_SHARED_MEM_BYTES) and >= max(NX, MX, MU) elements; d_robotModel is the GPU model"]
    tmpl = "template <typename T, int MX = 0, int MU = 0, int CF = 0, int CF2 = -1, bool ACCUMULATE = false>"
    act = lambda t: "p.activation[GRID_PLANT_MENU_" + t + "]"
    eps = lambda t: "p.eps[GRID_PLANT_MENU_" + t + "]"
    on = lambda t: "if (" + act(t) + " != GRID_PLANT_ACTIVATION_NONE) {"

    # ---- value ----
    self.gen_add_func_doc("tracking_cost_menu: frame position (CF) + state + input + the three joint barriers + linear features on x and u + frame orientation and twist (CF) + a second frame position (CF2)",
                          ["PRESET over the activated terms, chained through ACCUMULATE with a sync between terms; the first term carries the preset's ACCUMULATE.",
                           "With quadratic activations on the first three terms, the others GRID_PLANT_ACTIVATION_NONE and MX = MU = 0, the value equals tracking_cost. Fixed base."],
                          ["s_out is the scalar total-cost output (s_out[0])"] + doc, None)
    self.gen_add_code_line(tmpl)
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void tracking_cost_menu(T *s_out, const T *s_x, const T *s_u, const cost_menu_params<T> &p, T *s_frame, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    self.gen_add_code_line("if (!ACCUMULATE) { if (threadIdx.x == 0 && threadIdx.y == 0) { s_out[0] = static_cast<T>(0); } }")
    self.gen_add_sync()
    terms = [
        ("FRAME_POS", "frame_position_cost<T, true>(s_out, s_x, CF, p.ee_des, p.W, " + act("FRAME_POS") + ", " + eps("FRAME_POS") + ", p.cf_lower == nullptr ? nullptr : nullptr, nullptr, s_frame, s_scratch, d_robotModel);"),
        ("STATE", "state_cost<T, true>(s_out, s_x, p.x_des, p.Q, " + act("STATE") + ", " + eps("STATE") + ", p.x_lower, p.x_upper, s_scratch);"),
        ("INPUT", "input_cost<T, true>(s_out, s_u, p.u_des, p.R, " + act("INPUT") + ", " + eps("INPUT") + ", p.ub_lower, p.ub_upper, s_scratch);"),
    ]
    for t, call in terms:
        self.gen_add_code_line(on(t), True)
        self.gen_add_code_line(call)
        self.gen_add_sync()
        self.gen_add_end_control_flow()
    for call in ["joint_position_barrier<T>(s_out, s_x, p.q_lower, p.q_upper, p.mu_q, s_scratch);",
                 "joint_velocity_barrier<T>(s_out, s_x, p.qd_lower, p.qd_upper, p.mu_qd, s_scratch);",
                 "joint_torque_barrier<T>(s_out, s_u, p.u_lower, p.u_upper, p.mu_u, s_scratch);"]:
        self.gen_add_code_line(call)
        self.gen_add_sync()
    self.gen_add_code_line("if constexpr (MX > 0) { " + on("LINEAR_X"), True)
    self.gen_add_code_line("linear_feature_cost<T, MX, " + NX + ", true>(s_out, s_x, p.Ax, p.bx, p.tx, p.Wx, " + act("LINEAR_X") + ", " + eps("LINEAR_X") + ", p.lx_lower, p.lx_upper, s_scratch);")
    self.gen_add_sync()
    self.gen_add_code_line("} }")
    self.indent_level -= 1
    self.gen_add_code_line("if constexpr (MU > 0) { " + on("LINEAR_U"), True)
    self.gen_add_code_line("linear_feature_cost<T, MU, " + NU + ", true>(s_out, s_u, p.Au, p.bu, p.tu, p.Wu, " + act("LINEAR_U") + ", " + eps("LINEAR_U") + ", p.lu_lower, p.lu_upper, s_scratch);")
    self.gen_add_sync()
    self.gen_add_code_line("} }")
    self.indent_level -= 1
    self.gen_add_code_line(on("ORIENTATION"), True)
    self.gen_add_code_line("frame_orientation_cost<T, true>(s_out, s_x, CF, p.R_des, p.Wo, " + act("ORIENTATION") + ", " + eps("ORIENTATION") + ", s_frame, s_scratch, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_end_control_flow()
    self.gen_add_code_line(on("VELOCITY"), True)
    self.gen_add_code_line("frame_velocity_cost<T, true>(s_out, s_x, s_x + " + NQ + ", CF, p.v_des, p.Wv, " + act("VELOCITY") + ", " + eps("VELOCITY") + ", p.v_lower, p.v_upper, s_frame, s_scratch, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if constexpr (CF2 >= 0) { " + on("FRAME_POS_2"), True)
    self.gen_add_code_line("frame_position_cost<T, true>(s_out, s_x, CF2, p.cf_des, p.Wcf, " + act("FRAME_POS_2") + ", " + eps("FRAME_POS_2") + ", p.cf_lower, p.cf_upper, s_frame, s_scratch, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("} }")
    self.indent_level -= 1
    self.gen_add_end_function()

    # ---- gradient and hessian share the structure: zero the blocks, then every term adds ----
    for kind in ("gradient", "hessian"):
        if kind == "gradient":
            self.gen_add_func_doc("tracking_cost_menu_gradient: s_qk (state gradient, NX) + s_rk (input gradient, NU) of tracking_cost_menu",
                                  ["Every term adds into the blocks after a sync; ACCUMULATE=false zeroes them first."],
                                  ["s_qk is the state-block gradient output (size " + NX + ")", "s_rk is the input-block gradient output (size " + NU + ")"] + doc, None)
            self.gen_add_code_line(tmpl)
            self.gen_add_code_line("__device__")
            self.gen_add_code_line("void tracking_cost_menu_gradient(T *s_qk, T *s_rk, const T *s_x, const T *s_u, const cost_menu_params<T> &p, T *s_frame, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
            nqk, nrk = NX, NU
            fn = lambda base: base + "_gradient"
            sizes = {"q": NX, "u": NU}
        else:
            self.gen_add_func_doc("tracking_cost_menu_hessian: s_Qk (state GN Hessian, NX*NX) + s_Rk (input Hessian, NU*NU) of tracking_cost_menu",
                                  ["Gauss-Newton throughout; every term adds after a sync; ACCUMULATE=false zeroes the blocks first."],
                                  ["s_Qk is the state GN Hessian output (size " + str(nx * nx) + ", column-major)", "s_Rk is the input Hessian output (size " + str(nu * nu) + ", column-major)"] + doc, None)
            self.gen_add_code_line(tmpl)
            self.gen_add_code_line("__device__")
            self.gen_add_code_line("void tracking_cost_menu_hessian(T *s_qk, T *s_rk, const T *s_x, const T *s_u, const cost_menu_params<T> &p, T *s_frame, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
            nqk, nrk = str(nx * nx), str(nu * nu)
            fn = lambda base: base + "_hessian"
        self.gen_add_code_line("if (!ACCUMULATE) {", True)
        self.gen_add_parallel_loop("i", nqk)
        self.gen_add_code_line("s_qk[i] = static_cast<T>(0);")
        self.gen_add_end_control_flow()
        self.gen_add_parallel_loop("i", nrk)
        self.gen_add_code_line("s_rk[i] = static_cast<T>(0);")
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_add_code_line(on("FRAME_POS"), True)
        self.gen_add_code_line(fn("frame_position_cost") + "<T, true>(s_qk, s_x, CF, p.ee_des, p.W, " + act("FRAME_POS") + ", " + eps("FRAME_POS") + ", nullptr, nullptr, s_frame, s_scratch, d_robotModel);")
        self.gen_add_sync()
        self.gen_add_end_control_flow()
        self.gen_add_code_line(on("STATE"), True)
        self.gen_add_code_line(fn("state_cost") + "<T, true>(s_qk, s_x, p.x_des, p.Q, " + act("STATE") + ", " + eps("STATE") + ", p.x_lower, p.x_upper);")
        self.gen_add_sync()
        self.gen_add_end_control_flow()
        self.gen_add_code_line(on("INPUT"), True)
        self.gen_add_code_line(fn("input_cost") + "<T, true>(s_rk, s_u, p.u_des, p.R, " + act("INPUT") + ", " + eps("INPUT") + ", p.ub_lower, p.ub_upper);")
        self.gen_add_sync()
        self.gen_add_end_control_flow()
        if kind == "gradient":
            barriers = ["joint_position_barrier_gradient<T, 0, 0>(s_qk, s_x, p.q_lower, p.q_upper, p.mu_q);",
                        "joint_velocity_barrier_gradient<T, " + NQ + ", " + NQ + ">(s_qk, s_x, p.qd_lower, p.qd_upper, p.mu_qd);",
                        "joint_torque_barrier_gradient<T, 0, 0>(s_rk, s_u, p.u_lower, p.u_upper, p.mu_u);"]
        else:
            barriers = ["joint_position_barrier_hessian<T, " + NX + ", 0, 0>(s_qk, s_x, p.q_lower, p.q_upper, p.mu_q);",
                        "joint_velocity_barrier_hessian<T, " + NX + ", " + NQ + ", " + NQ + ">(s_qk, s_x, p.qd_lower, p.qd_upper, p.mu_qd);",
                        "joint_torque_barrier_hessian<T, " + NU + ", 0, 0>(s_rk, s_u, p.u_lower, p.u_upper, p.mu_u);"]
        for call in barriers:
            self.gen_add_code_line(call)
            self.gen_add_sync()
        self.gen_add_code_line("if constexpr (MX > 0) { " + on("LINEAR_X"), True)
        self.gen_add_code_line(fn("linear_feature_cost") + "<T, MX, " + NX + ", true>(s_qk, s_x, p.Ax, p.bx, p.tx, p.Wx, " + act("LINEAR_X") + ", " + eps("LINEAR_X") + ", p.lx_lower, p.lx_upper, s_scratch);")
        self.gen_add_sync()
        self.gen_add_code_line("} }")
        self.indent_level -= 1
        self.gen_add_code_line("if constexpr (MU > 0) { " + on("LINEAR_U"), True)
        self.gen_add_code_line(fn("linear_feature_cost") + "<T, MU, " + NU + ", true>(s_rk, s_u, p.Au, p.bu, p.tu, p.Wu, " + act("LINEAR_U") + ", " + eps("LINEAR_U") + ", p.lu_lower, p.lu_upper, s_scratch);")
        self.gen_add_sync()
        self.gen_add_code_line("} }")
        self.indent_level -= 1
        self.gen_add_code_line(on("ORIENTATION"), True)
        self.gen_add_code_line(fn("frame_orientation_cost") + "<T, true>(s_qk, s_x, CF, p.R_des, p.Wo, " + act("ORIENTATION") + ", " + eps("ORIENTATION") + ", s_frame, s_scratch, d_robotModel);")
        self.gen_add_sync()
        self.gen_add_end_control_flow()
        self.gen_add_code_line(on("VELOCITY"), True)
        self.gen_add_code_line(fn("frame_velocity_cost") + "<T, true>(s_qk, s_x, s_x + " + NQ + ", CF, p.v_des, p.Wv, " + act("VELOCITY") + ", " + eps("VELOCITY") + ", p.v_lower, p.v_upper, s_frame, s_scratch, d_robotModel);")
        self.gen_add_sync()
        self.gen_add_end_control_flow()
        self.gen_add_code_line("if constexpr (CF2 >= 0) { " + on("FRAME_POS_2"), True)
        self.gen_add_code_line(fn("frame_position_cost") + "<T, true>(s_qk, s_x, CF2, p.cf_des, p.Wcf, " + act("FRAME_POS_2") + ", " + eps("FRAME_POS_2") + ", p.cf_lower, p.cf_upper, s_frame, s_scratch, d_robotModel);")
        self.gen_add_sync()
        self.gen_add_code_line("} }")
        self.indent_level -= 1
        self.gen_add_end_function()


def gen_cost_menu(self, algorithms, ee_cost_ok):
    """Emit the cost menu after the existing terms and the preset."""
    if "inverse_dynamics" in algorithms:
        gen_gravity_torque(self)
    else:
        self.gen_add_code_line("// [grid_plant] gravity_torque skipped: requires 'inverse_dynamics' (grid::inverse_dynamics_inner).")
    gen_cost_activations(self)
    gen_activated_cost_family(self, "state")
    gen_activated_cost_family(self, "input")
    gen_linear_feature_cost(self)
    if "frame_jacobian" in algorithms:
        _gen_so3_helpers(self)
        gen_frame_eval(self)
        gen_frame_position_cost(self)
        gen_frame_orientation_cost(self)
        if not self.robot.floating_base:
            gen_frame_velocity_cost(self)
        else:
            self.gen_add_code_line("// [grid_plant] frame_velocity_cost skipped: fixed base only (the floating root has no single axis).")
        if not self.robot.floating_base:
            gen_tracking_cost_menu(self)
        else:
            self.gen_add_code_line("// [grid_plant] tracking_cost_menu skipped: fixed-base only (NUM_POS == NUM_VEL), like tracking_cost.")
    else:
        self.gen_add_code_line("// [grid_plant] frame_eval / frame_*_cost / tracking_cost_menu skipped: require 'frame_jacobian' (pass cost_frames= or add it to the algorithm list).")

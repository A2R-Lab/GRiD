"""Cost menu, phase one (2026-10-21): activations and the linear feature residual.

Every cost term in grid_plant factors as an activation applied per entry to a
residual r with Jacobian J: value = sum_i a(r_i), gradient = J^T a'(r), and the
Gauss-Newton Hessian J^T diag(a''(r)) J. The existing quadratic terms are the
quadratic activation over their residuals and keep their bytes; this module adds
the activation layer (quadratic and pseudo-Huber), activated forms of the state,
input and end-effector position terms, the linear feature residual A v + b over
the state or the input, and `tracking_cost_menu`, the preset that takes a
per-term activation choice and parameter per knot. Every target, weight and
activation parameter is a runtime buffer; the residual shape (sizes, the end
effector, the feature row count) is the only generation-time structure.

Activations a(r) with weight w per entry:
  quadratic     a = 1/2 w r^2            a' = w r              a'' = w
  pseudo-Huber  a = w (sqrt(r^2+e^2)-e)  a' = w r / s          a'' = w e^2 / s^3,  s = sqrt(r^2+e^2)
The pseudo-Huber parameter e must be positive; at e -> 0 it is the weighted
absolute value, at |r| << e it is the quadratic w r^2 / (2e).
"""

ACTIVATIONS = (("GRID_PLANT_ACTIVATION_QUADRATIC", 0), ("GRID_PLANT_ACTIVATION_PSEUDO_HUBER", 1))


def gen_cost_activations(self):
    """The activation layer: ids and the three inline device evaluators."""
    self.gen_add_code_line("")
    self.gen_add_code_line("// ---- cost activations (menu phase one): value, slope and curvature per residual entry ----")
    self.gen_add_code_line("// a(r; w, eps): quadratic 1/2 w r^2, or pseudo-Huber w (sqrt(r^2 + eps^2) - eps) with eps > 0.")
    self.gen_add_code_line("// Every activated term is value = sum_i a(r_i), gradient = J^T a'(r), GN Hessian = J^T diag(a''(r)) J.")
    for name, val in ACTIVATIONS:
        self.gen_add_code_line("const int " + name + " = " + str(val) + ";")
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line("T activation_value(const int kind, const T w, const T r, const T eps) {", True)
    self.gen_add_code_line("if (kind == GRID_PLANT_ACTIVATION_PSEUDO_HUBER) { T s = sqrt(r * r + eps * eps); return w * (s - eps); }")
    self.gen_add_code_line("return static_cast<T>(0.5) * w * r * r;")
    self.gen_add_end_function()
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line("T activation_slope(const int kind, const T w, const T r, const T eps) {", True)
    self.gen_add_code_line("if (kind == GRID_PLANT_ACTIVATION_PSEUDO_HUBER) { T s = sqrt(r * r + eps * eps); return w * r / s; }")
    self.gen_add_code_line("return w * r;")
    self.gen_add_end_function()
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__ __forceinline__")
    self.gen_add_code_line("T activation_curvature(const int kind, const T w, const T r, const T eps) {", True)
    self.gen_add_code_line("if (kind == GRID_PLANT_ACTIVATION_PSEUDO_HUBER) { T s = sqrt(r * r + eps * eps); return w * eps * eps / (s * s * s); }")
    self.gen_add_code_line("return w;")
    self.gen_add_end_function()


def gen_activated_cost_family(self, which):
    """`state_cost` / `input_cost`: the quadratic_* terms with an activation choice.

    state: r = x - x_des (NX), weights s_Q; input: r = u - u_des (NU), weights s_R.
    With activation = GRID_PLANT_ACTIVATION_QUADRATIC the value, gradient and
    Hessian are those of quadratic_state_cost / quadratic_input_cost.
    """
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
              "activation is GRID_PLANT_ACTIVATION_QUADRATIC or GRID_PLANT_ACTIVATION_PSEUDO_HUBER; eps is the pseudo-Huber parameter (> 0)"]
    self.gen_add_func_doc(
        base + ": value = sum_i a(" + var + "[i] - " + des + "[i]; " + w + "[i], eps) under the chosen activation",
        ["Block-cooperative: strided per-entry activations into s_scratch, then a serial reduction into s_out[0].",
         "ACCUMULATE=false overwrites s_out[0]; ACCUMULATE=true adds into it.",
         "Quadratic activation reproduces quadratic_" + base + " exactly; s_scratch must hold >= " + N + "."],
        ["s_out is the scalar cost output (s_out[0])"] + common + ["s_scratch is shared scratch of size >= " + N],
        None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void " + base + "(T *s_out, const T *" + var + ", const T *" + des + ", const T *" + w
                           + ", const int activation, const T eps, T *s_scratch) {", True)
    self.gen_add_parallel_loop("i", N)
    self.gen_add_code_line("s_scratch[i] = activation_value<T>(activation, " + w + "[i], " + var + "[i] - " + des + "[i], eps);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int i = 0; i < " + N + "; ++i) acc += s_scratch[i];")
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc(
        base + "_gradient: g[i] = a'(" + var + "[i] - " + des + "[i]; " + w + "[i], eps)",
        ["ACCUMULATE=false overwrites s_grad; ACCUMULATE=true adds into it."],
        ["s_grad is the gradient output (size " + size_doc + ")"] + common, None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void " + base + "_gradient(T *s_grad, const T *" + var + ", const T *" + des + ", const T *" + w
                           + ", const int activation, const T eps) {", True)
    self.gen_add_parallel_loop("i", N)
    self.gen_add_code_line("T g = activation_slope<T>(activation, " + w + "[i], " + var + "[i] - " + des + "[i], eps);")
    self.gen_add_code_line("if (ACCUMULATE) { s_grad[i] += g; } else { s_grad[i] = g; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc(
        base + "_hessian: Gauss-Newton Hessian = diag(a''(" + var + " - " + des + "; " + w + ", eps)), column-major " + N + " x " + N,
        ["Off-diagonal entries are zero. ACCUMULATE=false overwrites; ACCUMULATE=true adds."],
        ["s_hess is the dense Hessian output (size " + N + "*" + N + ", column-major)"] + common, None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void " + base + "_hessian(T *s_hess, const T *" + var + ", const T *" + des + ", const T *" + w
                           + ", const int activation, const T eps) {", True)
    self.gen_add_parallel_loop("ind", str(size * size))
    self.gen_add_code_line("int row = ind % " + N + ";")
    self.gen_add_code_line("int col = ind / " + N + ";")
    self.gen_add_code_line("T h = (row == col) ? activation_curvature<T>(activation, " + w + "[row], " + var + "[row] - " + des + "[row], eps) : static_cast<T>(0);")
    self.gen_add_code_line("if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


def gen_linear_feature_cost(self):
    """`linear_feature_cost<T, M, N>`: r = A v + b - target over a linear feature
    of v (the state, N = NX, or the input, N = NU), A column-major M x N."""
    doc = ["s_v is the variable (size N: NX for the state, NU for the input)",
           "s_A is the feature matrix, column-major M x N (A[m + M*n]); s_b is the offset (M)",
           "s_target is the per-knot target (M); s_W is the per-entry weight (M)",
           "activation / eps as for state_cost; s_scratch must hold >= M"]
    self.gen_add_func_doc(
        "linear_feature_cost: value = sum_m a(r_m; W[m], eps), r = A v + b - target (a linear feature of the state or the input)",
        ["Template M is the feature row count, N the variable size; both compile-time (loop bounds).",
         "ACCUMULATE=false overwrites s_out[0]; ACCUMULATE=true adds into it."],
        ["s_out is the scalar cost output (s_out[0])"] + doc, None)
    self.gen_add_code_line("template <typename T, int M, int N, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void linear_feature_cost(T *s_out, const T *s_v, const T *s_A, const T *s_b, const T *s_target, "
                           "const T *s_W, const int activation, const T eps, T *s_scratch) {", True)
    self.gen_add_parallel_loop("m", "M")
    self.gen_add_code_line("T r = s_b[m] - s_target[m];")
    self.gen_add_code_line("for (int n = 0; n < N; ++n) r += s_A[m + M * n] * s_v[n];")
    self.gen_add_code_line("s_scratch[m] = activation_value<T>(activation, s_W[m], r, eps);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int m = 0; m < M; ++m) acc += s_scratch[m];")
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc(
        "linear_feature_cost_gradient: g = A^T a'(r)  (size N)",
        ["s_scratch (>= M) holds the per-row slopes between the two passes.",
         "ACCUMULATE=false overwrites s_grad; ACCUMULATE=true adds into it."],
        ["s_grad is the gradient output (size N)"] + doc, None)
    self.gen_add_code_line("template <typename T, int M, int N, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void linear_feature_cost_gradient(T *s_grad, const T *s_v, const T *s_A, const T *s_b, const T *s_target, "
                           "const T *s_W, const int activation, const T eps, T *s_scratch) {", True)
    self.gen_add_parallel_loop("m", "M")
    self.gen_add_code_line("T r = s_b[m] - s_target[m];")
    self.gen_add_code_line("for (int n = 0; n < N; ++n) r += s_A[m + M * n] * s_v[n];")
    self.gen_add_code_line("s_scratch[m] = activation_slope<T>(activation, s_W[m], r, eps);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_parallel_loop("n", "N")
    self.gen_add_code_line("T g = static_cast<T>(0);")
    self.gen_add_code_line("for (int m = 0; m < M; ++m) g += s_A[m + M * n] * s_scratch[m];")
    self.gen_add_code_line("if (ACCUMULATE) { s_grad[n] += g; } else { s_grad[n] = g; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc(
        "linear_feature_cost_hessian: Gauss-Newton Hessian = A^T diag(a''(r)) A  (column-major N x N)",
        ["s_scratch (>= M) holds the per-row curvatures between the two passes.",
         "ACCUMULATE=false overwrites s_hess; ACCUMULATE=true adds into it."],
        ["s_hess is the dense Hessian output (size N*N, column-major)"] + doc, None)
    self.gen_add_code_line("template <typename T, int M, int N, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void linear_feature_cost_hessian(T *s_hess, const T *s_v, const T *s_A, const T *s_b, const T *s_target, "
                           "const T *s_W, const int activation, const T eps, T *s_scratch) {", True)
    self.gen_add_parallel_loop("m", "M")
    self.gen_add_code_line("T r = s_b[m] - s_target[m];")
    self.gen_add_code_line("for (int n = 0; n < N; ++n) r += s_A[m + M * n] * s_v[n];")
    self.gen_add_code_line("s_scratch[m] = activation_curvature<T>(activation, s_W[m], r, eps);")
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


def gen_ee_pos_cost_activated(self):
    """`ee_pos_cost_activated`: the end-effector position residual r = p(q) - p_des
    under an activation, through the raw ee_pos / ee_pos_gradient evaluators."""
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    nx = nq + nv
    NV, NX = str(nv), str(nx)
    doc = ["s_q is the joint position vector (size NUM_POS)",
           "s_p_des is the desired EE position (3); s_W is the per-axis weight (3)",
           "activation / eps as for state_cost",
           "s_end_effector_pose (6*NUM_EE) and s_scratch (>= END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT, 16B aligned) are caller scratch",
           "d_robotModel is the GPU model"]
    self.gen_add_func_doc(
        "ee_pos_cost_activated: value = sum_r a(p_r(q) - p_des_r; W[r], eps) over the 3 position axes",
        ["Evaluates the pose with ee_pos (caller-scratch inner). Quadratic activation reproduces ee_pos_cost.",
         "ACCUMULATE=false overwrites s_out[0]; ACCUMULATE=true adds into it."],
        ["s_out is the scalar cost output (s_out[0])"] + doc, None)
    self.gen_add_code_line("template <typename T, int EE = 0, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void ee_pos_cost_activated(T *s_out, const T *s_q, const T *s_p_des, const T *s_W, const int activation, const T eps, "
                           "T *s_end_effector_pose, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    self.gen_add_code_line("ee_pos<T>(s_end_effector_pose, s_q, s_scratch, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("#pragma unroll")
    self.gen_add_code_line("for (int r = 0; r < 3; ++r) { acc += activation_value<T>(activation, s_W[r], s_end_effector_pose[6*EE + r] - s_p_des[r], eps); }")
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc(
        "ee_pos_cost_activated_gradient: grad_x = [J_p^T a'(r) ; 0] over x = [q; qd]",
        ["Evaluates pose and Jacobian with ee_pos_gradient. The qd-block (entries NUM_VEL.." + str(nx - 1) + ") is set to zero when !ACCUMULATE.",
         "ACCUMULATE=false overwrites s_grad; ACCUMULATE=true adds into it."],
        ["s_grad is the gradient over x (size " + NX + ")",
         "s_end_effector_pose_gradient (6*NUM_VEL*NUM_EE) is caller Jacobian scratch"] + doc, None)
    self.gen_add_code_line("template <typename T, int EE = 0, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void ee_pos_cost_activated_gradient(T *s_grad, const T *s_q, const T *s_p_des, const T *s_W, const int activation, const T eps, "
                           "T *s_end_effector_pose, T *s_end_effector_pose_gradient, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    self.gen_add_code_line("ee_pos_gradient<T>(s_end_effector_pose, s_end_effector_pose_gradient, s_q, s_scratch, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_parallel_loop("i", NV)
    self.gen_add_code_line("T g = static_cast<T>(0);")
    self.gen_add_code_line("#pragma unroll")
    self.gen_add_code_line("for (int r = 0; r < 3; ++r) {")
    self.gen_add_code_line("    T Jri = s_end_effector_pose_gradient[6*" + NV + "*EE + 6*i + r];")
    self.gen_add_code_line("    g += Jri * activation_slope<T>(activation, s_W[r], s_end_effector_pose[6*EE + r] - s_p_des[r], eps);")
    self.gen_add_code_line("}")
    self.gen_add_code_line("if (ACCUMULATE) { s_grad[i] += g; } else { s_grad[i] = g; }")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if (!ACCUMULATE) {", True)
    self.gen_add_parallel_loop("i", str(nq))
    self.gen_add_code_line("s_grad[" + NV + " + i] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc(
        "ee_pos_cost_activated_hessian: Gauss-Newton Hessian = J_p^T diag(a''(r)) J_p in the q-block of an NX x NX column-major matrix",
        ["Evaluates pose and Jacobian with ee_pos_gradient; the qd rows and columns are zero.",
         "ACCUMULATE=false overwrites s_hess; ACCUMULATE=true adds into it."],
        ["s_hess is the dense Hessian output (size " + NX + "*" + NX + ", column-major)",
         "s_end_effector_pose_gradient (6*NUM_VEL*NUM_EE) is caller Jacobian scratch"] + doc, None)
    self.gen_add_code_line("template <typename T, int EE = 0, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void ee_pos_cost_activated_hessian(T *s_hess, const T *s_q, const T *s_p_des, const T *s_W, const int activation, const T eps, "
                           "T *s_end_effector_pose, T *s_end_effector_pose_gradient, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    self.gen_add_code_line("ee_pos_gradient<T>(s_end_effector_pose, s_end_effector_pose_gradient, s_q, s_scratch, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_parallel_loop("ind", str(nx * nx))
    self.gen_add_code_line("int row = ind % " + NX + ";")
    self.gen_add_code_line("int col = ind / " + NX + ";")
    self.gen_add_code_line("T h = static_cast<T>(0);")
    self.gen_add_code_line("if (row < " + NV + " && col < " + NV + ") {")
    self.gen_add_code_line("    #pragma unroll")
    self.gen_add_code_line("    for (int r = 0; r < 3; ++r) {")
    self.gen_add_code_line("        T Jri = s_end_effector_pose_gradient[6*" + NV + "*EE + 6*row + r];")
    self.gen_add_code_line("        T Jrj = s_end_effector_pose_gradient[6*" + NV + "*EE + 6*col + r];")
    self.gen_add_code_line("        h += Jri * activation_curvature<T>(activation, s_W[r], s_end_effector_pose[6*EE + r] - s_p_des[r], eps) * Jrj;")
    self.gen_add_code_line("    }")
    self.gen_add_code_line("}")
    self.gen_add_code_line("if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


def gen_tracking_cost_menu(self):
    """`tracking_cost_menu`: the tracking preset with a per-term activation choice
    and parameter, plus optional linear feature terms on the state (MX rows) and
    the input (MU rows). Term order in s_activation / s_eps: [ee, state, input,
    linear_x, linear_u]. Fixed base, like tracking_cost."""
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    nu = nv
    nx = nq + nv
    NQ, NU, NX = str(nq), str(nu), str(nx)
    params = ("const T *s_x_des, const T *s_u_des, const T *s_ee_des, "
              "const T *s_Q, const T *s_R, const T *s_W, "
              "const T *s_q_lower, const T *s_q_upper, const T mu_q, "
              "const T *s_qd_lower, const T *s_qd_upper, const T mu_qd, "
              "const T *s_u_lower, const T *s_u_upper, const T mu_u, "
              "const int *s_activation, const T *s_eps, "
              "const T *s_Ax, const T *s_bx, const T *s_tx, const T *s_Wx, "
              "const T *s_Au, const T *s_bu, const T *s_tu, const T *s_Wu, ")
    doc = ["s_x / s_u are the current state [q; qd] and control (sizes " + NX + " / " + NU + ")",
           "s_x_des / s_u_des / s_ee_des / s_Q / s_R / s_W and the three barrier bound-and-mu triples exactly as tracking_cost",
           "s_activation (5 ints) and s_eps (5) are the per-term activation and pseudo-Huber parameter, order [ee, state, input, linear_x, linear_u]",
           "s_Ax (column-major MX x NX), s_bx, s_tx, s_Wx (MX) define the linear state feature; MX = 0 omits the term and the pointers may be nullptr",
           "s_Au (column-major MU x NU), s_bu, s_tu, s_Wu (MU) define the linear input feature; MU = 0 omits the term",
           "s_scratch must hold >= END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT and >= max(NX, MX, MU)"]
    self.gen_add_code_line("#define GRID_PLANT_HAS_COST_MENU 1")
    self.gen_add_func_doc(
        "tracking_cost_menu: tracking_cost with a per-term activation choice (ee, state, input) plus optional linear feature terms on x and u",
        ["PRESET: ee_pos_cost_activated + state_cost + input_cost + the three barriers + linear_feature_cost on x (MX rows) and on u (MU rows).",
         "ACCUMULATE=false overwrites s_out[0]; true adds. All quadratic activations and MX = MU = 0 reproduce tracking_cost.",
         "Fixed-base only (NUM_POS == NUM_VEL)."],
        ["s_out is the scalar total-cost output (s_out[0])"] + doc +
        ["s_end_effector_pose / s_scratch are caller EE scratch; d_robotModel is the GPU model"], None)
    self.gen_add_code_line("template <typename T, int EE = 0, int MX = 0, int MU = 0, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void tracking_cost_menu(T *s_out, const T *s_x, const T *s_u, " + params +
                           "T *s_end_effector_pose, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    self.gen_add_code_line("ee_pos_cost_activated<T, EE, ACCUMULATE>(s_out, s_x, s_ee_des, s_W, s_activation[0], s_eps[0], s_end_effector_pose, s_scratch, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("state_cost<T, true>(s_out, s_x, s_x_des, s_Q, s_activation[1], s_eps[1], s_scratch);")
    self.gen_add_sync()
    self.gen_add_code_line("input_cost<T, true>(s_out, s_u, s_u_des, s_R, s_activation[2], s_eps[2], s_scratch);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_position_barrier<T>(s_out, s_x, s_q_lower, s_q_upper, mu_q, s_scratch);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_velocity_barrier<T>(s_out, s_x, s_qd_lower, s_qd_upper, mu_qd, s_scratch);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_torque_barrier<T>(s_out, s_u, s_u_lower, s_u_upper, mu_u, s_scratch);")
    self.gen_add_code_line("if constexpr (MX > 0) {", True)
    self.gen_add_sync()
    self.gen_add_code_line("linear_feature_cost<T, MX, " + NX + ", true>(s_out, s_x, s_Ax, s_bx, s_tx, s_Wx, s_activation[3], s_eps[3], s_scratch);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if constexpr (MU > 0) {", True)
    self.gen_add_sync()
    self.gen_add_code_line("linear_feature_cost<T, MU, " + NU + ", true>(s_out, s_u, s_Au, s_bu, s_tu, s_Wu, s_activation[4], s_eps[4], s_scratch);")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc(
        "tracking_cost_menu_gradient: s_qk (state gradient, NX) + s_rk (input gradient, NU) of tracking_cost_menu",
        ["State block = ee_pos_cost_activated_gradient + state_cost_gradient + position and velocity barriers + linear state feature.",
         "Input block = input_cost_gradient + torque barrier + linear input feature.",
         "ACCUMULATE=false overwrites the blocks; true adds into them. Sync between every accumulating term."],
        ["s_qk is the state-block gradient output (size " + NX + ")", "s_rk is the input-block gradient output (size " + NU + ")"] + doc +
        ["s_end_effector_pose / s_end_effector_pose_gradient / s_scratch are caller EE scratch; d_robotModel is the GPU model"], None)
    self.gen_add_code_line("template <typename T, int EE = 0, int MX = 0, int MU = 0, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void tracking_cost_menu_gradient(T *s_qk, T *s_rk, const T *s_x, const T *s_u, " + params +
                           "T *s_end_effector_pose, T *s_end_effector_pose_gradient, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    self.gen_add_code_line("ee_pos_cost_activated_gradient<T, EE, ACCUMULATE>(s_qk, s_x, s_ee_des, s_W, s_activation[0], s_eps[0], s_end_effector_pose, s_end_effector_pose_gradient, s_scratch, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("state_cost_gradient<T, true>(s_qk, s_x, s_x_des, s_Q, s_activation[1], s_eps[1]);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_position_barrier_gradient<T, 0, 0>(s_qk, s_x, s_q_lower, s_q_upper, mu_q);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_velocity_barrier_gradient<T, " + NQ + ", " + NQ + ">(s_qk, s_x, s_qd_lower, s_qd_upper, mu_qd);")
    self.gen_add_code_line("if constexpr (MX > 0) {", True)
    self.gen_add_sync()
    self.gen_add_code_line("linear_feature_cost_gradient<T, MX, " + NX + ", true>(s_qk, s_x, s_Ax, s_bx, s_tx, s_Wx, s_activation[3], s_eps[3], s_scratch);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_code_line("input_cost_gradient<T, ACCUMULATE>(s_rk, s_u, s_u_des, s_R, s_activation[2], s_eps[2]);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_torque_barrier_gradient<T, 0, 0>(s_rk, s_u, s_u_lower, s_u_upper, mu_u);")
    self.gen_add_code_line("if constexpr (MU > 0) {", True)
    self.gen_add_sync()
    self.gen_add_code_line("linear_feature_cost_gradient<T, MU, " + NU + ", true>(s_rk, s_u, s_Au, s_bu, s_tu, s_Wu, s_activation[4], s_eps[4], s_scratch);")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    self.gen_add_func_doc(
        "tracking_cost_menu_hessian: s_Qk (state GN Hessian, NX*NX) + s_Rk (input Hessian, NU*NU) of tracking_cost_menu",
        ["State block = ee_pos_cost_activated_hessian + state_cost_hessian + position and velocity barrier Hessians + linear state feature.",
         "Input block = input_cost_hessian + torque barrier Hessian + linear input feature. Gauss-Newton throughout.",
         "ACCUMULATE=false overwrites; true adds."],
        ["s_Qk is the state GN Hessian output (size " + str(nx * nx) + ", column-major)", "s_Rk is the input Hessian output (size " + str(nu * nu) + ", column-major)"] + doc +
        ["s_end_effector_pose / s_end_effector_pose_gradient / s_scratch are caller EE scratch; d_robotModel is the GPU model"], None)
    self.gen_add_code_line("template <typename T, int EE = 0, int MX = 0, int MU = 0, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void tracking_cost_menu_hessian(T *s_Qk, T *s_Rk, const T *s_x, const T *s_u, " + params +
                           "T *s_end_effector_pose, T *s_end_effector_pose_gradient, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    self.gen_add_code_line("ee_pos_cost_activated_hessian<T, EE, ACCUMULATE>(s_Qk, s_x, s_ee_des, s_W, s_activation[0], s_eps[0], s_end_effector_pose, s_end_effector_pose_gradient, s_scratch, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("state_cost_hessian<T, true>(s_Qk, s_x, s_x_des, s_Q, s_activation[1], s_eps[1]);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_position_barrier_hessian<T, " + NX + ", 0, 0>(s_Qk, s_x, s_q_lower, s_q_upper, mu_q);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_velocity_barrier_hessian<T, " + NX + ", " + NQ + ", " + NQ + ">(s_Qk, s_x, s_qd_lower, s_qd_upper, mu_qd);")
    self.gen_add_code_line("if constexpr (MX > 0) {", True)
    self.gen_add_sync()
    self.gen_add_code_line("linear_feature_cost_hessian<T, MX, " + NX + ", true>(s_Qk, s_x, s_Ax, s_bx, s_tx, s_Wx, s_activation[3], s_eps[3], s_scratch);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_code_line("input_cost_hessian<T, ACCUMULATE>(s_Rk, s_u, s_u_des, s_R, s_activation[2], s_eps[2]);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_torque_barrier_hessian<T, " + NU + ", 0, 0>(s_Rk, s_u, s_u_lower, s_u_upper, mu_u);")
    self.gen_add_code_line("if constexpr (MU > 0) {", True)
    self.gen_add_sync()
    self.gen_add_code_line("linear_feature_cost_hessian<T, MU, " + NU + ", true>(s_Rk, s_u, s_Au, s_bu, s_tu, s_Wu, s_activation[4], s_eps[4], s_scratch);")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


def gen_cost_menu(self, ee_cost_ok):
    """Emit the phase-one cost menu after the existing terms and the preset."""
    gen_cost_activations(self)
    gen_activated_cost_family(self, "state")
    gen_activated_cost_family(self, "input")
    gen_linear_feature_cost(self)
    if ee_cost_ok:
        self.gen_add_code_line("#define GRID_PLANT_HAS_ACTIVATED_EE_COST 1")
        gen_ee_pos_cost_activated(self)
        if not self.robot.floating_base:
            gen_tracking_cost_menu(self)
        else:
            self.gen_add_code_line("// [grid_plant] tracking_cost_menu skipped: fixed-base only (NUM_POS == NUM_VEL), like tracking_cost.")
    else:
        self.gen_add_code_line("// [grid_plant] ee_pos_cost_activated / tracking_cost_menu skipped: require ee_pos_cost (both 'end_effector_pose' and 'end_effector_pose_gradient').")

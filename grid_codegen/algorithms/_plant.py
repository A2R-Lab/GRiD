"""Plant namespace codegen (T6).

Emits a SIBLING `namespace grid_plant { ... }` block AFTER the `grid`
namespace closes. Everything here is ADDITIVE: it composes the already-emitted
`grid::` device functions (integrator value/gradient, end-effector pose +
Jacobian) and the vendored GLASS `grid_linalg_*` primitives. ZERO edits are made
to any existing `grid::` emit path.

State convention (matches the integrator + GATO/PDDP references):
    x = [q (NUM_POS); qd (NUM_VEL)]   (fixed-base: NUM_POS == NUM_VEL == n)
    u = control torques (NUM_VEL)
The plant-gradient block order is [A | B] = [dx_{k+1}/dx | dx_{k+1}/du], the
same `s_dAB` surface `grid::integrator_gradient` already produces (2n x 3n,
column-major).

Cost convention (matches PDDP TrajoptCost / GATO trackingcost):
    quadratic cost  = 1/2 * r^T diag(W) r           (the 1/2 scaling)
    gradient        = diag(W) r
    Gauss-Newton hessian (the RATIFIED choice):
        quadratic   -> diag(W)
        ee-position -> J_p^T W J_p
The true analytic 2nd-order hessian (cost curvature folded with the integrator
Hessian) is intentionally NOT folded into the COST layer — Gauss-Newton is the
ratified choice here. The analytic integrator Hessian itself DOES exist as the
separate opt-in `integrator_hessian` / `plant_step_hessian` surface (s_d2AB,
both bases); a consumer wanting the exact plant Hessian composes that surface.

Barriers (log-barrier, per GATO jointBarrier):
    b(x)  = -mu * ( log(x - lower) + log(upper - x) )
    b'(x) = -mu * ( 1/(x - lower) - 1/(upper - x) )
    b''(x)=  mu * ( 1/(x - lower)^2 + 1/(upper - x)^2 )
Bounds are passed in as explicit `s_lower` / `s_upper` pointers (the URDF parser
only carries position limits today, so velocity/torque bounds must be supplied
by the caller). An `isfinite` guard skips any side whose bound is +/-inf, so an
unbounded joint contributes EXACTLY zero to value/gradient/hessian.
"""

from grid_codegen.helpers._code_generation_helpers import _gen_mjx_build_R_lines
from grid_codegen.algorithms._centroidal import (
    _gen_centroidal_call, _centroidal_inner_temp_mem_size, _centroidal_device_extra,
)


def _emit_centroidal_caller_scratch(self):
    """Lay out the centroidal arena from a caller-provided s_scratch + load XmatsHom +
    run centroidal_inner -> fills s_A (CMM 6 x NUM_VEL, col-major), s_com (CoM pos 3),
    s_extra (mass at [0]). Mirrors com_device/ccrba_device but caller-scratch (arena
    sourced from s_scratch, NOT extern __shared__), so the cost fns are true inners
    callable from another kernel's block without aliasing its dynamic-smem arena."""
    self.gen_add_code_line("using namespace grid;")
    self.gen_XmatsHom_helpers_temp_shared_memory_code(
        _centroidal_inner_temp_mem_size(self), extra_t_buffers=_centroidal_device_extra(self),
        include_linalg_scratch=True, linalg_scratch_bytes="GRID_EE_LINALG_SHARED_BYTES<T>()",
        arena_base_expr="s_scratch")
    self.gen_load_update_XmatsHom_helpers_function_call()
    _gen_centroidal_call(self)
    self.gen_add_sync()


def _emit_momentum_residual_precompute(self, nv):
    """Compute the centroidal-momentum residual e[6] = A*qd - h_des ONCE into a
    block-shared s_e[6] (parallel over the 6 components), so the value reduction and
    the per-DOF gradient both READ it instead of each recomputing h = A*qd. The old
    gradient recomputed the full h (6*NUM_VEL work) inside every per-DOF thread -> an
    O(NUM_VEL^2) redundancy; this makes it O(NUM_VEL). Numerically identical (the h
    sum runs over vi = 0..NUM_VEL in the same order). Requires s_A (the CMM) already
    filled by centroidal_inner; ends with a sync so all threads see s_e."""
    self.gen_add_code_line("__shared__ T s_e[6];")
    self.gen_add_parallel_loop("r", "6")
    self.gen_add_code_line("T h = static_cast<T>(0);")
    self.gen_add_code_line("#pragma unroll")
    self.gen_add_code_line("for (int vi = 0; vi < " + str(nv) + "; ++vi) h += s_A[r + 6*vi] * s_qd[vi];")
    self.gen_add_code_line("s_e[r] = h - s_h_des[r];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()


# ---------------------------------------------------------------------------
# mjx (MuJoCo output-convention) helper for the tracking-cost epilogues.
#
# The Gauss-Newton tracking costs (ee_pos_cost / com_cost q-block at offset 0;
# momentum_cost qd-block at offset nq) have a single nonzero nv-wide tangent
# block. Its base-LINEAR 3 entries reframe by the value-Jacobian column map
# (J_mjx = J_pin G^{-1}): grad reframes as a covector (G·) and the GN hessian by
# congruence (G·Gᵀ). For ee_pos/com the block sits at offset 0 so the shared
# `gen_mjx_base_rotate(s_grad)` / `gen_mjx_congruence(s_hess, nx)` apply directly
# (the zero qd rows/cols are untouched). For momentum the block is at offset nq,
# so the congruence base rows/cols are at nq:nq+3 (NOT 0:3): the shared
# `gen_mjx_congruence` hardcodes 0,1,2 and cannot be used. `_gen_cost_congruence_at_offset`
# emits the identical R-rotation on rows/cols off:off+3 (transcribing the helper's
# exact row-major R-indexing, validated in numpy vs `quadratic_tracking_cost_pin_to_mjx`).
# The GN hessian DROPS the value-curvature, so there is NO frame-correction term.
# ---------------------------------------------------------------------------

def _gen_cost_mjx_kernel_input(self, nq, convert_qd=False):
    """Emit the mjx INPUT conversion for a tracking-cost kernel, into per-block
    ``__shared__`` buffers, so the underlying grid:: device fns (which consume the
    base quaternion to place the world-frame EE/CoM/centroidal quantities) see the
    correct PIN-frame config + velocity. Returns the names of the mutable buffers to
    feed the device calls: ``("s_q_mjx",)`` or ``("s_q_mjx", "s_qd_mjx")``.

    q: reorder the base quaternion wxyz (mjx, scalar-first) -> xyzw (pin, scalar-last).
    qd (momentum only): ``qd[0:3] = R^T qd[0:3]`` (mjx GLOBAL base-linear velocity ->
    pin LOCAL), R from the reordered xyzw quaternion. Mirrors gen_mjx_input_convert.
    Single thread + sync; emitted inside the per-timestep block loop, BEFORE the
    value/grad/hess device calls. Only meaningful for a floating base (caller gates)."""
    nv = self.robot.get_num_vel()
    self.gen_add_code_line("__shared__ T s_q_mjx[" + str(nq) + "];")
    if convert_qd:
        self.gen_add_code_line("__shared__ T s_qd_mjx[" + str(nv) + "];")
    # copy q (and qd) in parallel, then reorder/convert the base block on one thread.
    self.gen_add_parallel_loop("i", str(nq))
    self.gen_add_code_line("s_q_mjx[i] = s_q[i];")
    self.gen_add_end_control_flow()
    if convert_qd:
        self.gen_add_parallel_loop("i", str(nv))
        self.gen_add_code_line("s_qd_mjx[i] = s_qd[i];")
        self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_code_lines([
        "// mjx input convert: base quaternion wxyz->xyzw" + (" + base-linear velocity -> pin frame" if convert_qd else ""),
        "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
        "T qw_in = s_q_mjx[3];",
        "s_q_mjx[3] = s_q_mjx[4]; s_q_mjx[4] = s_q_mjx[5]; s_q_mjx[5] = s_q_mjx[6]; s_q_mjx[6] = qw_in;",
    ])
    if convert_qd:
        self.gen_add_code_lines(_gen_mjx_build_R_lines("s_q_mjx"))
        self.gen_add_code_lines([
            # v_pin_lin = R^T v_mjx_lin
            "T vlx = s_qd_mjx[0], vly = s_qd_mjx[1], vlz = s_qd_mjx[2];",
            "s_qd_mjx[0] = R[0]*vlx + R[3]*vly + R[6]*vlz;",
            "s_qd_mjx[1] = R[1]*vlx + R[4]*vly + R[7]*vlz;",
            "s_qd_mjx[2] = R[2]*vlx + R[5]*vly + R[8]*vlz;",
        ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    return ("s_q_mjx", "s_qd_mjx") if convert_qd else ("s_q_mjx",)


def _gen_state_cost_mjx_kernel_input(self, nq, nv):
    """Emit the mjx INPUT conversion for the STATE cost kernel. Unlike the geometric
    tracking costs the value is convention-DEPENDENT: the kernel evaluates the cost on
    the PIN-frame state, so the velocity block of ``x`` is converted mjx->pin while the
    config block stays in mjx layout (it is differenced against the user's mjx
    ``x_des``). Two buffers result:

      * ``s_x_use`` (size NX): a copy of ``x`` with ``qd[0:3] = R^T qd[0:3]`` (the
        base-linear velocity, at x[NQ:NQ+3]); the q-block is UNCHANGED (still wxyz, to
        match the mjx-frame ``x_des``/``Q`` in the residual ``r = x_use - x_des``).
      * ``s_q_xyzw`` (size NQ): the config with the base quaternion reordered
        wxyz->xyzw, used ONLY to build R in the grad/hess epilogues.

    R is built from the reordered xyzw quaternion. Single thread + sync. Returns
    ``("s_x_use", "s_q_xyzw")``. Floating-base only (caller gates)."""
    nx = nq + nv
    self.gen_add_code_line("__shared__ T s_x_use[" + str(nx) + "];")
    self.gen_add_code_line("__shared__ T s_q_xyzw[" + str(nq) + "];")
    self.gen_add_parallel_loop("i", str(nx))
    self.gen_add_code_line("s_x_use[i] = s_x[i];")
    self.gen_add_end_control_flow()
    self.gen_add_parallel_loop("i", str(nq))
    self.gen_add_code_line("s_q_xyzw[i] = s_x[i];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_code_lines([
        "// mjx input convert: build R from xyzw quaternion, then qd[0:3] -> pin frame",
        "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
        # reorder the base quaternion wxyz->xyzw in s_q_xyzw[3..6] (R source only)
        "T qw_in = s_q_xyzw[3];",
        "s_q_xyzw[3] = s_q_xyzw[4]; s_q_xyzw[4] = s_q_xyzw[5]; s_q_xyzw[5] = s_q_xyzw[6]; s_q_xyzw[6] = qw_in;",
    ])
    self.gen_add_code_lines(_gen_mjx_build_R_lines("s_q_xyzw"))
    self.gen_add_code_lines([
        # qd[0:3] is at s_x_use[NQ:NQ+3]: v_pin = R^T v_mjx
        "T vlx = s_x_use[" + str(nq) + "], vly = s_x_use[" + str(nq + 1) + "], vlz = s_x_use[" + str(nq + 2) + "];",
        "s_x_use[" + str(nq) + "] = R[0]*vlx + R[3]*vly + R[6]*vlz;",
        "s_x_use[" + str(nq + 1) + "] = R[1]*vlx + R[4]*vly + R[7]*vlz;",
        "s_x_use[" + str(nq + 2) + "] = R[2]*vlx + R[5]*vly + R[8]*vlz;",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    return ("s_x_use", "s_q_xyzw")


def _gen_cost_congruence_at_offset(self, mat, n, off, q_name="s_q"):
    """Emit a congruence ``X_mjx = G X G^T`` on the base-linear block at rows/cols
    ``off:off+3`` of an ``n x n`` COLUMN-MAJOR matrix ``mat`` (element (r,c) at
    ``mat[r + n*c]``). Mirrors :func:`gen_mjx_congruence` EXACTLY except the base
    block is at ``off`` rather than 0 (for the momentum-cost qd-block at offset nq).
    R is built row-major from the xyzw quaternion at ``q_name[3..6]``.

    BLOCK-PARALLEL (the two sweeps each touch all ``n`` columns/rows — O(nv) serial
    work). Phase 1 reframes the base ROWS over every column ``c`` (independent across
    c); phase 2 reframes the base COLS over every row ``r`` (independent across r).
    A sync separates them: phase 2 reads the ``off:off+3`` x ``off:off+3`` corner
    that phase 1 wrote. R is recomputed register-local per thread from the read-only
    smem quaternion (no shared scratch) — math is byte-identical to the serial form."""
    self.gen_add_code_line(
        "// mjx output: congruence G " + mat + " G^T on the base block at offset " + str(off) + " (rows then cols)")
    # Phase 1: rows off:off+3 <- R . rows, one thread per column c.
    self.gen_add_parallel_loop("c", str(n))
    self.gen_add_code_lines(_gen_mjx_build_R_lines(q_name))
    self.gen_add_code_lines([
        "T m0 = {m}[{o0} + {n}*c], m1 = {m}[{o1} + {n}*c], m2 = {m}[{o2} + {n}*c];".format(
            m=mat, n=n, o0=off + 0, o1=off + 1, o2=off + 2),
        "{m}[{o0} + {n}*c] = R[0]*m0 + R[1]*m1 + R[2]*m2;".format(m=mat, n=n, o0=off + 0),
        "{m}[{o1} + {n}*c] = R[3]*m0 + R[4]*m1 + R[5]*m2;".format(m=mat, n=n, o1=off + 1),
        "{m}[{o2} + {n}*c] = R[6]*m0 + R[7]*m1 + R[8]*m2;".format(m=mat, n=n, o2=off + 2),
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # Phase 2: cols off:off+3 <- cols . R^T, one thread per row r.
    self.gen_add_parallel_loop("r", str(n))
    self.gen_add_code_lines(_gen_mjx_build_R_lines(q_name))
    self.gen_add_code_lines([
        "T m0 = {m}[r + {n}*{o0}], m1 = {m}[r + {n}*{o1}], m2 = {m}[r + {n}*{o2}];".format(
            m=mat, n=n, o0=off + 0, o1=off + 1, o2=off + 2),
        "{m}[r + {n}*{o0}] = m0*R[0] + m1*R[1] + m2*R[2];".format(m=mat, n=n, o0=off + 0),
        "{m}[r + {n}*{o1}] = m0*R[3] + m1*R[4] + m2*R[5];".format(m=mat, n=n, o1=off + 1),
        "{m}[r + {n}*{o2}] = m0*R[6] + m1*R[7] + m2*R[8];".format(m=mat, n=n, o2=off + 2),
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()


# ---------------------------------------------------------------------------
# Plant step (value) + plant step gradient — thin wrappers over the integrator.
# ---------------------------------------------------------------------------

def _gen_plant_step_gradient_mjx_kernel_input(self, x_name="s_x", u_name="s_u"):
    """Emit the GRADIENT-family mjx INPUT conversion for plant_step_gradient_kernel,
    in place on the mutable smem state ``x_name = [q (NQ); qd (NV)]`` and ``u_name``,
    so the underlying grid::integrator_gradient device fn + its dAB epilogue see
    pin-frame inputs. Mirrors the integrator-gradient kernel's gen_mjx_input_convert
    exactly, but operates on the STACKED state: q = x[0:NQ], qd = x[NQ:NQ+NV].

    Full convert (single thread + sync inside the helper): base quat wxyz->xyzw +
    qd[0:3]=R^T qd[0:3] (mjx GLOBAL base-linear velocity -> pin LOCAL) + u[0:3]=R^T
    u[0:3] (covector force). Emitted AFTER staging x/u into smem, BEFORE the device
    call (so XImats X[0] is built from the reordered quaternion). Wrapped in
    `if constexpr (MUJOCO_OUTPUT)`; no-op on the pin path. Floating base only (caller
    gates). The VALUE kernel uses a q-ONLY reorder inline instead (qd stays raw mjx
    for the global-add retract), so it does not call this."""
    nq = self.robot.get_num_pos()
    self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
    # full input convert on the stacked state: q = x[0:NQ], qd = x[NQ:]. The qd
    # sub-pointer is parenthesized so the helper's [i] indexing binds to the offset
    # pointer, not the literal (operator precedence).
    self.gen_mjx_input_convert(q_name=x_name, qd_name="(" + x_name + " + " + str(nq) + ")", u_name=u_name)
    self.gen_add_end_control_flow()


def gen_plant_step(self):
    """`plant_step` — thin wrapper over `grid::integrator_device` (value).

    x_{k+1} = integrator(x_k, u_k, dt). s_x is [q; qd]; we slice q/qd and call
    the integrator's auto-allocating device wrapper (which owns its own scratch).

    mjx (MuJoCo output-convention): MUJOCO_OUTPUT is threaded LAST (floating-base
    only). The integrator VALUE mjx convention lives in the RETRACT epilogue (the
    base-linear position takes a GLOBAL additive step + the output quaternion is
    reordered xyzw->wxyz), NOT in grid::integrator_device (which is pin-only). The
    kernel does the q-only INPUT convert (quat wxyz->xyzw) into mutable smem BEFORE
    the call; here, after the pin integrate, we OVERWRITE s_x_kp1's base-linear
    position with the mjx global add (s_q[0:3] still holds the ORIGINAL base
    position — integrator wrote OUT-of-place — and s_qd[0:3] is the RAW mjx global
    base-linear velocity, NOT converted, exactly like grid::integrator_kernel's
    RETRACT family). EULER/SI-EULER only (multistage static_asserts out in the
    integrator device); MUJOCO_OUTPUT on a multistage IT would silently mis-retract,
    but the value path has no analytic guard so callers stay on single-stage.
    """
    nq = self.robot.get_num_pos()
    fb = self.robot.floating_base
    func_params = [
        "s_x_kp1 is the next state output (size NUM_POS + NUM_VEL)",
        "s_x is the current state [q (NUM_POS); qd (NUM_VEL)]" +
            (" (mjx: q already quat-reordered xyzw by the kernel)" if fb else ""),
        "s_u is the control torque vector (size NUM_VEL)",
        "d_robotModel is the GPU model helpers (XImats, topology, ...)",
        "gravity is the gravity constant",
        "dt is the integration timestep",
        "d_f_ext is the (optional) external forces, body-major 6*NUM_BODIES joint-LOCAL Featherstone "
        "[angular;linear], or nullptr for the force-free step. grid_plant::f_ext_body builds this "
        "from contact-frame forces.",
    ]
    self.gen_add_func_doc("Plant step: x_{k+1} = integrator(x_k, u_k, dt) (thin wrapper over grid::integrator_device)",
                          [], func_params, None)
    # MUJOCO_OUTPUT (floating only): LAST template param so existing <T, IT> call
    # sites are unaffected; default false if-constexpr-elides the mjx retract ->
    # byte-identical pin codegen. Fixed-base never emits it (no base block).
    self.gen_add_code_line("template <typename T, grid::IntegratorType IT = grid::IntegratorType::EULER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__device__")
    # d_f_ext is threaded straight through to the integrator (which has always accepted it) and
    # defaults to nullptr, so every existing call site is unchanged and the force-free path is
    # behaviorally identical. Contact forces reach this as JOINT-LOCAL body wrenches; build them
    # from contact-frame forces with grid_plant::f_ext_body (GATO ask 1).
    self.gen_add_code_line("void plant_step(T *s_x_kp1, const T *s_x, const T *s_u, "
                           "const grid::robotModel<T> *d_robotModel, const T gravity, const T dt, "
                           "T *d_f_ext = nullptr) {", True)
    self.gen_add_code_line("const T *s_q  = s_x;")
    self.gen_add_code_line("const T *s_qd = &s_x[" + str(nq) + "];")
    self.gen_add_code_line("grid::integrator_device<T, IT>(s_x_kp1, s_q, s_qd, s_u, d_robotModel, d_f_ext, gravity, dt);")
    if fb:
        # mjx output (RETRACT): the integrator integrated the base in the pin
        # convention (SE(3) V(phi) base-position coupling, O(dt^2) wrong for mjx).
        # OVERWRITE the base-linear position with the mjx GLOBAL additive step
        # s_q[0:3] + dt*s_qd[0:3] (s_q still holds the ORIGINAL pre-integration base
        # position — integrator wrote OUT-of-place into s_x_kp1; s_qd[0:3] is the raw
        # mjx global base-linear velocity). Then reorder the output base quaternion
        # xyzw->wxyz back to mjx order (inverse of the kernel's input reorder).
        self.gen_add_sync()
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        self.gen_mjx_retract("s_x_kp1", "s_q", "s_qd", "dt")
        self.gen_add_code_lines([
            "// mjx output: base quaternion xyzw->wxyz (inverse of input reorder)",
            "if (threadIdx.x == 0 && threadIdx.y == 0) {", True,
            "T qw_out = s_x_kp1[6];",
            "s_x_kp1[6] = s_x_kp1[5]; s_x_kp1[5] = s_x_kp1[4]; s_x_kp1[4] = s_x_kp1[3]; s_x_kp1[3] = qw_out;",
        ])
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_add_end_control_flow()
    self.gen_add_end_function()


def gen_plant_step_gradient(self, with_value=False):
    """`plant_step_gradient[_and_value]` — thin wrapper over
    `grid::integrator_gradient_device` (the [A|B] = s_dAB surface).

    Pass-through: the emitted s_dAB IS grid's integrator gradient (this is what
    the equivalence test asserts). When `with_value`, also returns x_{k+1} via
    the both-at-once integrator gradient device (no extra RBD work).

    The caller supplies the FD-grad scratch buffers the integrator-gradient
    device needs (it is an inner-owns-placement orchestrator); we forward them
    straight through. s_q / s_qd are NON-const because the multi-stage RK path
    mutates them in place across stages (see grid::integrator_gradient_device).

    mjx (MuJoCo output-convention): MUJOCO_OUTPUT is threaded LAST (floating-base
    only) and forwarded straight to grid::integrator_gradient[_with_value]_device,
    whose state-transition-Jacobian epilogue (_emit_integrator_gradient_mjx_output,
    plus the x_{k+1} retract for the with_value surface) is ALREADY validated. The
    kernel does the INPUT convert (quat wxyz->xyzw, base-linear velocity + force ->
    pin frame) into the mutable smem s_x/s_u BEFORE the call. EULER/SI-EULER only
    (multistage RK mjx static_asserts out in the integrator-gradient device).

    Plant HESSIAN: per the ratified decision the plant-step hessian used by the
    cost layer is the Gauss-Newton outer product of the COST gradient, assembled
    in the cost hessian functions below. The true second-order integrator
    Hessian (d^2 x_{k+1} / d(x,u)^2, a 2n x 3n x 3n tensor) is NOT emitted here.
    // NOTE(plant-2nd-order): the analytic integrator Hessian exists as the
    // opt-in plant_step_hessian surface (s_d2AB); the cost layer deliberately
    // stays Gauss-Newton (ratified) rather than composing it here.
    """
    suffix = "_and_value" if with_value else ""
    fname = "plant_step_gradient" + suffix
    func_params = [
        "s_dAB is the [A | B] output (2*NUM_VEL x 3*NUM_VEL, column-major)",
    ]
    if with_value:
        func_params.append("s_x_kp1 is the next-state output (size NUM_POS + NUM_VEL)")
    func_params += [
        "s_x is the current state [q; qd] (NON-const: mutated across RK stages)",
        "s_u is the control torque vector (size NUM_VEL)",
        "s_df_du / s_dc_du / s_vaf / s_Minv / s_qdd are FD-grad in/out scratch (caller-placed)",
        "s_q_orig / s_qd_orig / s_stage_grad_qdd / s_D_qdd_stage are multi-stage scratch (caller-placed)",
        "s_dInt_q_6x6 / s_dInt_v_6x6 are floating-base SE(3) dIntegrate blocks (unused fixed-base)",
        "s_temp / d_workspace / d_temp_spill are the integrator-gradient scratch arenas (caller-placed)",
        "d_robotModel / gravity / dt as for plant_step",
        "d_f_ext is the (optional) external forces, body-major 6*NUM_BODIES joint-LOCAL Featherstone "
        "[angular;linear], or nullptr. NOTE this returns d(x_kp1)/d(x,u) AT the given f_ext; the "
        "sensitivity to f_ext itself is grid::f_ext_gradient_device (dqdd/dfext = M^-1 J^T).",
    ]
    nq = self.robot.get_num_pos()
    self.gen_add_func_doc("Plant step gradient [A|B]" + (" + value" if with_value else "") +
                          " (thin wrapper over grid::" + ("integrator_with_gradient" if with_value else "integrator_gradient") +
                          "_device — pass-through)",
                          [], func_params, None)
    # MUJOCO_OUTPUT (floating only): LAST template param so existing
    # <T, IT, SCRATCH_IN_SMEM, USE_DA_DF_SPILL> call sites are unaffected; forwarded
    # straight to the integrator-gradient device (which owns the validated mjx dAB
    # epilogue). Fixed-base never emits it -> byte-identical pin codegen.
    mjx_device = self.robot.floating_base
    if mjx_device:
        self.gen_add_code_line("template <typename T, grid::IntegratorType IT = grid::IntegratorType::EULER, "
                               "bool SCRATCH_IN_SMEM = true, bool USE_DA_DF_SPILL = false, bool MUJOCO_OUTPUT = false>")
    else:
        self.gen_add_code_line("template <typename T, grid::IntegratorType IT = grid::IntegratorType::EULER, "
                               "bool SCRATCH_IN_SMEM = true, bool USE_DA_DF_SPILL = false>")
    self.gen_add_code_line("__device__")
    sig = "void " + fname + "(T *s_dAB, "
    if with_value:
        sig += "T *s_x_kp1, "
    # The middle params end with the SE(3) dInt blocks; the shared XImats /
    # topology helpers are injected here (same mechanism the integrator-gradient
    # device uses) so this stays a faithful pass-through. The s_temp / workspace
    # / spill arenas + (model, gravity, dt) close the signature.
    sig_middle = ("T *s_x, const T *s_u, T *s_df_du, T *s_dc_du, T *s_vaf, T *s_Minv, T *s_qdd, "
                  "T *s_q_orig, T *s_qd_orig, T *s_stage_grad_qdd, T *s_D_qdd_stage, "
                  "T *s_dInt_q_6x6, T *s_dInt_v_6x6, ")
    sig_middle, func_params = self.gen_insert_helpers_func_def_params(sig_middle, func_params, -1)
    sig_end = ("T *s_temp, T *d_workspace, T *d_temp_spill, "
               "const grid::robotModel<T> *d_robotModel, const T gravity, const T dt, "
               "T *d_f_ext = nullptr) {")
    self.gen_add_code_line(sig + sig_middle + sig_end, True)
    self.gen_add_code_line("T *s_q  = s_x;")
    self.gen_add_code_line("T *s_qd = &s_x[" + str(nq) + "];")
    inner_tmpl = "<T, IT, SCRATCH_IN_SMEM, USE_DA_DF_SPILL" + (", MUJOCO_OUTPUT>" if mjx_device else ">")
    inner = "grid::" + ("integrator_with_gradient" if with_value else "integrator_gradient") + "_device" \
            + inner_tmpl + "(s_dAB, "
    if with_value:
        inner += "s_x_kp1, "
    inner_middle = ("s_q, s_qd, s_u, s_df_du, s_dc_du, s_vaf, s_Minv, s_qdd, "
                    "s_q_orig, s_qd_orig, s_stage_grad_qdd, s_D_qdd_stage, "
                    "s_dInt_q_6x6, s_dInt_v_6x6, ")
    inner_helpers = self.gen_insert_helpers_function_call()
    # d_f_ext threaded straight through (the integrator gradient has always accepted it); defaults to
    # nullptr so existing call sites and the force-free path are unchanged.
    inner_end = ("s_temp, d_workspace, d_temp_spill, d_robotModel, d_f_ext, gravity, dt);")
    self.gen_add_code_line(inner + inner_middle + inner_helpers + inner_end)
    self.gen_add_end_function()


def gen_plant_step_hessian(self):
    """`plant_step_hessian` — thin wrapper over `grid::integrator_hessian_device`
    (the s_d2AB surface: the true 2nd-order sensitivity of the integrator step).

    H has shape (2*NUM_VEL, 3*NUM_VEL, 3*NUM_VEL), row-major flat:
        H[o*nz*nz + a*nz + b] = d^2 x_{k+1}[o] / dz[a] dz[b],
    z = [dq(nv); dqd(nv); du(nv)], output rows = [position-tangent(nv); velocity(nv)].
    The emitted s_d2AB IS grid::integrator_hessian_device's output (this is what
    the equivalence test asserts vs the RBDReference oracle). Scope: EULER /
    SI-EULER on BOTH fixed and floating base (floating via the SE(3)-retract Hessian
    in integrator_hessian_device); only multi-stage RK static_asserts out in the
    composed device fn (clean-break). See f1_plant_step_hessian_plan.md.

    The caller supplies the fdsva_so scratch buffers (s_df2/s_idsva_so/s_Minv/
    s_df_du/s_qdd) the inner needs (it is an inner-owns-placement orchestrator);
    we forward them straight through, splitting s_x into s_q / s_qd.
    """
    func_params = [
        "s_d2AB is the Hessian output (2*NUM_VEL x 3*NUM_VEL x 3*NUM_VEL, row-major)",
        "s_x is the current state [q; qd]",
        "s_u is the control torque vector (size NUM_VEL)",
        "s_df2 / s_idsva_so / s_Minv / s_df_du / s_qdd are fdsva_so in/out scratch (caller-placed)",
        "s_temp / d_workspace / d_fd_grad_spill / s_fdsva_temp are the fdsva_so scratch arenas (caller-placed)",
        "d_robotModel / gravity / dt as for plant_step",
    ]
    nq = self.robot.get_num_pos()
    self.gen_add_func_doc("Plant step hessian s_d2AB (thin wrapper over grid::integrator_hessian_device — pass-through)",
                          [], func_params, None)
    # MUJOCO_OUTPUT (floating only): LAST template param so existing
    # <T, IT, SCRATCH, SPILL, CONTRACT> call sites are unaffected; forwarded straight
    # to grid::integrator_hessian_device (which owns the validated mjx d2AB epilogue).
    # Fixed-base never emits it -> byte-identical pin codegen. The kernel does the
    # INPUT convert into the mutable s_x/s_u smem BEFORE the call (mirroring
    # plant_step_gradient), so s_q/s_qd/s_u here are already pin-frame (const OK).
    mjx_wrapper = self.robot.floating_base
    if mjx_wrapper:
        self.gen_add_code_line("template <typename T, grid::IntegratorType IT = grid::IntegratorType::EULER, "
                               "bool SCRATCH_IN_SMEM = true, bool FD_GRAD_USE_SPILL = false, bool CONTRACT_IN_SMEM = true, bool MUJOCO_OUTPUT = false>")
    else:
        self.gen_add_code_line("template <typename T, grid::IntegratorType IT = grid::IntegratorType::EULER, "
                               "bool SCRATCH_IN_SMEM = true, bool FD_GRAD_USE_SPILL = false, bool CONTRACT_IN_SMEM = true>")
    self.gen_add_code_line("__device__")
    sig = "void plant_step_hessian(T *s_d2AB, "
    sig_middle = ("const T *s_x, const T *s_u, T *s_df2, T *s_idsva_so, T *s_Minv, T *s_df_du, T *s_qdd, ")
    sig_middle, func_params = self.gen_insert_helpers_func_def_params(sig_middle, func_params, -1)
    # d_mjx_ws: dedicated mjx-epilogue scratch band (floating only); forwarded
    # straight to the device fn. FIXED-BASE OMITS it -> byte-identical pin codegen.
    mjx_ws_param = ("T *d_mjx_ws, " if mjx_wrapper else "")
    sig_end = ("T *s_temp, T *d_workspace, T *d_fd_grad_spill, T *s_fdsva_temp, " + mjx_ws_param
               + "const grid::robotModel<T> *d_robotModel, const T gravity, const T dt) {")
    self.gen_add_code_line(sig + sig_middle + sig_end, True)
    self.gen_add_code_line("const T *s_q  = s_x;")
    self.gen_add_code_line("const T *s_qd = &s_x[" + str(nq) + "];")
    inner_tmpl = ("<T, IT, SCRATCH_IN_SMEM, FD_GRAD_USE_SPILL, CONTRACT_IN_SMEM"
                  + (", MUJOCO_OUTPUT>" if mjx_wrapper else ">"))
    inner = ("grid::integrator_hessian_device" + inner_tmpl
             + "(s_d2AB, s_df2, s_idsva_so, s_Minv, s_df_du, s_qdd, s_q, s_qd, s_u, ")
    inner_helpers = self.gen_insert_helpers_function_call()
    mjx_ws_arg = ("d_mjx_ws, " if mjx_wrapper else "")
    inner_end = ("s_temp, d_workspace, d_fd_grad_spill, s_fdsva_temp, " + mjx_ws_arg + "d_robotModel, gravity, dt);")
    self.gen_add_code_line(inner + inner_helpers + inner_end)
    self.gen_add_end_function()


# ---------------------------------------------------------------------------
# Quadratic state / input cost (value, gradient, GN-diag hessian, fused).
# ---------------------------------------------------------------------------

def _gen_quadratic_cost_family(self, which):
    """Emit the quadratic cost family for `which` in {"state", "input"}.

    state cost: r = x - x_des (size NX = NUM_POS + NUM_VEL), weights s_Q (NX).
    input cost: r = u - u_des (size NU = NUM_VEL),           weights s_R (NU).
    Cost = 1/2 * sum_i W_i r_i^2. Gradient_i = W_i r_i. GN hessian = diag(W).
    """
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    if which == "state":
        size = nq + nv
        var, des, w = "s_x", "s_x_des", "s_Q"
        base = "quadratic_state_cost"
        size_doc = "NUM_POS + NUM_VEL = " + str(size)
    else:
        size = nv
        var, des, w = "s_u", "s_u_des", "s_R"
        base = "quadratic_input_cost"
        size_doc = "NUM_VEL = " + str(size)
    N = str(size)
    # mjx (MuJoCo output-convention) epilogue applies ONLY to the STATE cost on a
    # FLOATING base: the velocity (qd) block of x reframes by G (base-linear GLOBAL
    # vs LOCAL). The q-block is convention-invariant raw coords; input cost has no
    # base block. The grad/hess device fns take an extra `s_q` (xyzw quaternion, the
    # kernel-reordered config) used ONLY to build R for the epilogue.
    mjx = (which == "state" and self.robot.floating_base)
    mjx_tmpl = ", bool MUJOCO_OUTPUT = false"
    mjx_qarg = ", const T *s_q" if mjx else ""

    # ---- value: cost = 1/2 sum W_i (var_i - des_i)^2, accumulated into s_out[0] ----
    notes = ["Block-cooperative: each thread accumulates its strided terms into s_scratch, then a serial reduction writes s_out[0].",
             "ACCUMULATE=false overwrites s_out[0]; ACCUMULATE=true ADDS into it (for summing cost terms into one scalar).",
             "s_scratch must hold at least " + N + " elements."]
    if which == "state" and self.robot.floating_base:
        # GATO ASK3: raw x-space subtraction on a FLOATING base differences the
        # quaternion COMPONENTS — not a chart/tangent error; its grad/hess are
        # not the derivatives a manifold Newton/SQP consumes.
        notes.append("FLOATING BASE WARNING: r = x - x_des subtracts raw quaternion components — NOT a "
                     "tangent-space error. For manifold-correct tracking derivatives use "
                     "quadratic_state_cost_tangent (log-map error, exact J_diff, tangent-sized outputs).")
    self.gen_add_func_doc(
        base + ": value = 1/2 * sum_i " + w + "[i] * (" + var + "[i] - " + des + "[i])^2",
        notes,
        ["s_out is the scalar cost output (s_out[0])",
         var + " is the current value (size " + size_doc + ")",
         des + " is the desired/target value (size " + size_doc + ")",
         w + " is the diagonal weight vector (size " + size_doc + ")",
         "s_scratch is shared scratch of size >= " + N],
        None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void " + base + "(T *s_out, const T *" + var + ", const T *" + des +
                           ", const T *" + w + ", T *s_scratch) {", True)
    self.gen_add_parallel_loop("i", N)
    self.gen_add_code_line("T r = " + var + "[i] - " + des + "[i];")
    self.gen_add_code_line("s_scratch[i] = static_cast<T>(0.5) * " + w + "[i] * r * r;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int i = 0; i < " + N + "; ++i) acc += s_scratch[i];")
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    # ---- gradient: g_i = W_i (var_i - des_i), written with `mode` (set or add) ----
    self.gen_add_func_doc(
        base + "_gradient: g[i] = " + w + "[i] * (" + var + "[i] - " + des + "[i])",
        ["ACCUMULATE=false overwrites s_grad; ACCUMULATE=true adds into it (for fusing into a packed [x;u] gradient)."],
        ["s_grad is the gradient output (size " + size_doc + ")",
         var + " / " + des + " / " + w + " as in the value function"],
        None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false" + mjx_tmpl + ">")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void " + base + "_gradient(T *s_grad, const T *" + var + ", const T *" + des +
                           ", const T *" + w + mjx_qarg + ") {", True)
    self.gen_add_parallel_loop("i", N)
    self.gen_add_code_line("T g = " + w + "[i] * (" + var + "[i] - " + des + "[i]);")
    self.gen_add_code_line("if (ACCUMULATE) { s_grad[i] += g; } else { s_grad[i] = g; }")
    self.gen_add_end_control_flow()
    if mjx:
        # mjx output: the velocity (qd) tangent block sits at offset NQ of the NX
        # gradient; its base-LINEAR 3 entries are at s_grad[NQ:NQ+3]. Reframe as a
        # covector via base_rotate on the SUB-POINTER (s_grad + NQ), NEVER &s_grad[NQ].
        # s_q is the xyzw-reordered config (kernel-converted). The q-block is untouched.
        self.gen_add_sync()
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        self.gen_mjx_base_rotate("(s_grad + " + str(nq) + ")", q_name="s_q")
        self.gen_add_end_control_flow()
    self.gen_add_end_function()

    # ---- GN-diag hessian: H = diag(W), column-major n x n ----
    self.gen_add_func_doc(
        base + "_hessian: Gauss-Newton hessian = diag(" + w + ") (RATIFIED: GN outer product; for a quadratic cost this is exactly diag(W))",
        ["Writes a dense column-major " + N + " x " + N + " matrix; off-diagonal entries are zero.",
         "ACCUMULATE=false overwrites; ACCUMULATE=true adds into the diagonal of an existing block."],
        ["s_hess is the dense hessian output (size " + N + "*" + N + ", column-major)",
         w + " is the diagonal weight vector (size " + size_doc + ")"],
        None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false" + mjx_tmpl + ">")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void " + base + "_hessian(T *s_hess, const T *" + w + mjx_qarg + ") {", True)
    self.gen_add_parallel_loop("ind", str(size * size))
    self.gen_add_code_line("int row = ind % " + N + ";")
    self.gen_add_code_line("int col = ind / " + N + ";")
    self.gen_add_code_line("T h = (row == col) ? " + w + "[row] : static_cast<T>(0);")
    self.gen_add_code_line("if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }")
    self.gen_add_end_control_flow()
    if mjx:
        # mjx output: the velocity (qd) block is at offset NQ of the NX x NX col-major
        # hessian; its base-LINEAR rows/cols are at NQ:NQ+3 (NOT 0:3), so the shared
        # gen_mjx_congruence (hardcoded 0,1,2) cannot be used — emit the identical
        # R-congruence on rows/cols NQ:NQ+3. The cross blocks (q-qd) are exactly zero
        # (hess = diag(Q)) so there is NO cross-reframe. s_q is the xyzw config.
        self.gen_add_sync()
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        _gen_cost_congruence_at_offset(self, "s_hess", size, nq, q_name="s_q")
        self.gen_add_end_control_flow()
    self.gen_add_end_function()

    # ---- fused value + grad + hess ----
    self.gen_add_func_doc(
        base + "_value_grad_hess: fused value + gradient + GN-diag hessian in one pass",
        ["Convenience fusion of the three functions above; same conventions and ACCUMULATE semantics for grad/hess."],
        ["s_out / s_grad / s_hess are the three outputs",
         var + " / " + des + " / " + w + " / s_scratch as above"],
        None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false" + mjx_tmpl + ">")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void " + base + "_value_grad_hess(T *s_out, T *s_grad, T *s_hess, "
                           "const T *" + var + ", const T *" + des + ", const T *" + w + ", T *s_scratch" + mjx_qarg + ") {", True)
    gh_tmpl = ", ACCUMULATE, MUJOCO_OUTPUT" if mjx else ", ACCUMULATE"
    gh_qarg = ", s_q" if mjx else ""
    self.gen_add_code_line(base + "<T>(s_out, " + var + ", " + des + ", " + w + ", s_scratch);")
    self.gen_add_code_line(base + "_gradient<T" + gh_tmpl + ">(s_grad, " + var + ", " + des + ", " + w + gh_qarg + ");")
    self.gen_add_code_line(base + "_hessian<T" + gh_tmpl + ">(s_hess, " + w + gh_qarg + ");")
    self.gen_add_end_function()


def gen_quadratic_state_cost_tangent(self):
    """GATO ASK3: tangent-space (log-map) quadratic state cost for FLOATING-base
    tracking — the CUDA twin of RBDReference.quadratic_state_cost_tangent.

    Error is TANGENT-width 2*NV: e = [difference(q_des, q); qd - qd_des], with
    difference = the SE(3) boxminus (grid_difference_floating_q). Outputs are
    TANGENT-sized (grad 2*NV, hess 2*NV x 2*NV col-major) — the chart
    derivatives of delta -> cost(integrate(q, delta_q), qd + delta_v) at
    delta = 0, i.e. exactly the KKT blocks a manifold Newton/SQP consumes.
    The q-block uses the EXACT J_diff = inv(dIntegrate(q_des, e, 'v')):
    blockdiag(inv(J6), I) with J6 = [[Jr, Q],[0, Jr]] row-major inverted in
    closed form ([[Ai, -Ai Q Ai],[0, Ai]]).

    Hessian is templated GAUSS_NEWTON (default TRUE — the ratified PSD choice
    for this preset; the sealed ASK3 decision). GAUSS_NEWTON=false adds the
    EXACT curvature, which has TWO terms (guide §7.z2 — dropping the second
    gives an O(1)-wrong hessian with a perfect gradient):
      -J (dM) J   via d2Integrate at (q_des, e), contracted with grad_q, PLUS
      the chart-slope term grad_q · d2Integrate(q, 0) (N(0)=I but dN(0)!=0).
    Pin convention only (no MUJOCO_OUTPUT twin)."""
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    tn = 2 * nv
    NV, TN = str(nv), str(tn)
    setup_scratch = nv + 72          # eq (nv) + J6 (36) + Jinv6 (36)
    newton_scratch = nv + 72 + 6 + 216   # + g6 (6) + one 6x6x6 tensor buffer

    # ---- shared serial setup: eq + J6 + Jinv6 into caller scratch ----
    self.gen_add_func_doc(
        "quadratic_state_cost_tangent_setup: serial helper — fills s_scratch with "
        "[eq (NV) | J6 (36, row-major) | Jinv6 (36, row-major)] and syncs",
        ["eq = difference(q_des, q) (SE(3) boxminus prefix + Euler joints); J6 = the "
         "dIntegrate ARG_v 6x6 at eq; Jinv6 = its exact block-triangular inverse.",
         "Serial (thread 0) + one sync; callers read the buffers afterwards."],
        ["s_x / s_x_des are [q (NUM_POS); qd (NUM_VEL)] states",
         "s_scratch holds >= " + str(setup_scratch) + " elements of T"],
        None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void quadratic_state_cost_tangent_setup(const T *s_x, const T *s_x_des, T *s_scratch) {", True)
    self.gen_add_code_line("T *s_eq = s_scratch; T *s_J = &s_scratch[" + NV + "]; T *s_Ji = &s_scratch[" + NV + " + 36];")
    self.gen_add_serial_ops()
    self.gen_add_code_line("grid::grid_difference_floating_q<T, " + str(nq) + ">(s_x_des, s_x, s_eq);")
    self.gen_add_code_line("grid::grid_dIntegrate_v_block<T>(s_eq, s_J);")
    self.gen_add_code_lines([
        "// invert J6 = [[A, B], [0, A]] (row-major): Jinv = [[Ai, -Ai B Ai], [0, Ai]]",
        "T a00=s_J[0], a01=s_J[1], a02=s_J[2], a10=s_J[6], a11=s_J[7], a12=s_J[8], a20=s_J[12], a21=s_J[13], a22=s_J[14];",
        "T det = a00*(a11*a22 - a12*a21) - a01*(a10*a22 - a12*a20) + a02*(a10*a21 - a11*a20);",
        "T idet = static_cast<T>(1) / det;",
        "T Ai[9];",
        "Ai[0] = (a11*a22 - a12*a21)*idet; Ai[1] = (a02*a21 - a01*a22)*idet; Ai[2] = (a01*a12 - a02*a11)*idet;",
        "Ai[3] = (a12*a20 - a10*a22)*idet; Ai[4] = (a00*a22 - a02*a20)*idet; Ai[5] = (a02*a10 - a00*a12)*idet;",
        "Ai[6] = (a10*a21 - a11*a20)*idet; Ai[7] = (a01*a20 - a00*a21)*idet; Ai[8] = (a00*a11 - a01*a10)*idet;",
        "T B[9], AiB[9], C[9];",
        "#pragma unroll",
        "for (int i = 0; i < 3; ++i) { B[3*i] = s_J[6*i + 3]; B[3*i+1] = s_J[6*i + 4]; B[3*i+2] = s_J[6*i + 5]; }",
        "#pragma unroll",
        "for (int i = 0; i < 9; ++i) { int r = i / 3, cc = i % 3; T acc = static_cast<T>(0);",
        "    for (int k = 0; k < 3; ++k) acc += Ai[3*r+k]*B[3*k+cc]; AiB[i] = acc; }",
        "#pragma unroll",
        "for (int i = 0; i < 9; ++i) { int r = i / 3, cc = i % 3; T acc = static_cast<T>(0);",
        "    for (int k = 0; k < 3; ++k) acc += AiB[3*r+k]*Ai[3*k+cc]; C[i] = acc; }",
        "#pragma unroll",
        "for (int i = 0; i < 9; ++i) { int r = i / 3, cc = i % 3;",
        "    s_Ji[6*r + cc] = Ai[i]; s_Ji[6*r + 3 + cc] = -C[i];",
        "    s_Ji[6*(3+r) + cc] = static_cast<T>(0); s_Ji[6*(3+r) + 3 + cc] = Ai[i]; }",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()

    # ---- value ----
    self.gen_add_func_doc(
        "quadratic_state_cost_tangent: value = 1/2 * (eq^T diag(Qq) eq + ev^T diag(Qv) ev), "
        "eq = difference(q_des, q) (log-map), ev = qd - qd_des",
        ["TANGENT error (2*NV wide); s_Q is the DIAGONAL weight vector of size 2*NV (NOT NUM_POS+NUM_VEL).",
         "ACCUMULATE=false overwrites s_out[0]; true adds.",
         "s_scratch must hold >= " + str(3 * nv) + " elements of T."],
        ["s_out is the scalar cost output (s_out[0])",
         "s_x / s_x_des are [q (NUM_POS); qd (NUM_VEL)] states",
         "s_Q is the diagonal tangent weight vector (size 2*NV = " + TN + ")",
         "s_scratch is shared scratch"],
        None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void quadratic_state_cost_tangent(T *s_out, const T *s_x, const T *s_x_des, const T *s_Q, T *s_scratch) {", True)
    self.gen_add_serial_ops()
    self.gen_add_code_line("grid::grid_difference_floating_q<T, " + str(nq) + ">(s_x_des, s_x, s_scratch);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_parallel_loop("i", TN)
    self.gen_add_code_line("T e = (i < " + NV + ") ? s_scratch[i] : (s_x[" + str(nq) + " + i - " + NV + "] - s_x_des[" + str(nq) + " + i - " + NV + "]);")
    self.gen_add_code_line("s_scratch[" + NV + " + i] = static_cast<T>(0.5) * s_Q[i] * e * e;")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int i = 0; i < " + TN + "; ++i) acc += s_scratch[" + NV + " + i];")
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    # ---- gradient ----
    self.gen_add_func_doc(
        "quadratic_state_cost_tangent_gradient: g = [Jq^T (Qq .* eq) ; Qv .* ev] (TANGENT-sized, 2*NV)",
        ["EXACT J_diff: the top-left 6x6 of Jq is inv(dIntegrate(q_des, eq, 'v')); joints are identity.",
         "ACCUMULATE=false overwrites s_grad; true adds.",
         "s_scratch must hold >= " + str(setup_scratch) + " elements of T."],
        ["s_grad is the tangent gradient output (size 2*NV = " + TN + ")",
         "s_x / s_x_des / s_Q / s_scratch as in quadratic_state_cost_tangent"],
        None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void quadratic_state_cost_tangent_gradient(T *s_grad, const T *s_x, const T *s_x_des, const T *s_Q, T *s_scratch) {", True)
    self.gen_add_code_line("quadratic_state_cost_tangent_setup<T>(s_x, s_x_des, s_scratch);")
    self.gen_add_code_line("const T *s_eq = s_scratch; const T *s_Ji = &s_scratch[" + NV + " + 36];")
    self.gen_add_parallel_loop("i", TN)
    self.gen_add_code_line("T g;")
    self.gen_add_code_line("if (i < 6) { g = static_cast<T>(0);")
    self.gen_add_code_line("    for (int r = 0; r < 6; ++r) g += s_Ji[6*r + i] * s_Q[r] * s_eq[r]; }")
    self.gen_add_code_line("else if (i < " + NV + ") { g = s_Q[i] * s_eq[i]; }")
    self.gen_add_code_line("else { g = s_Q[i] * (s_x[" + str(nq) + " + i - " + NV + "] - s_x_des[" + str(nq) + " + i - " + NV + "]); }")
    self.gen_add_code_line("if (ACCUMULATE) { s_grad[i] += g; } else { s_grad[i] = g; }")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_function()

    # ---- hessian ----
    self.gen_add_func_doc(
        "quadratic_state_cost_tangent_hessian: H = [[Jq^T diag(Qq) Jq, 0], [0, diag(Qv)]] "
        "(+ exact curvature when GAUSS_NEWTON=false); TANGENT-sized 2*NV x 2*NV, column-major",
        ["GAUSS_NEWTON default TRUE (the ratified PSD preset choice). GAUSS_NEWTON=false adds the "
         "EXACT curvature — TWO terms (guide §7.z2): -J(dM)J via d2Integrate(q_des, eq) contracted "
         "with grad_q, PLUS the chart-slope grad_q . d2Integrate(q, 0). Both live in the top-left "
         "6x6 (joint charts are linear). Not necessarily PSD away from the solution.",
         "ACCUMULATE=false overwrites the whole 2*NV x 2*NV block; true adds.",
         "s_scratch must hold >= " + str(setup_scratch) + " (GN) / " + str(newton_scratch) + " (Newton) elements of T."],
        ["s_hess is the tangent hessian output (size " + TN + "*" + TN + ", column-major)",
         "s_x / s_x_des / s_Q / s_scratch as in quadratic_state_cost_tangent"],
        None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false, bool GAUSS_NEWTON = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void quadratic_state_cost_tangent_hessian(T *s_hess, const T *s_x, const T *s_x_des, const T *s_Q, T *s_scratch) {", True)
    self.gen_add_code_line("quadratic_state_cost_tangent_setup<T>(s_x, s_x_des, s_scratch);")
    self.gen_add_code_line("const T *s_eq = s_scratch; const T *s_Ji = &s_scratch[" + NV + " + 36];")
    self.gen_add_parallel_loop("ind", str(tn * tn))
    self.gen_add_code_line("int row = ind % " + TN + ";")
    self.gen_add_code_line("int col = ind / " + TN + ";")
    self.gen_add_code_line("T h = static_cast<T>(0);")
    self.gen_add_code_line("if (row < 6 && col < 6) {")
    self.gen_add_code_line("    for (int b = 0; b < 6; ++b) h += s_Ji[6*b + row] * s_Q[b] * s_Ji[6*b + col]; }")
    self.gen_add_code_line("else if (row == col) { h = s_Q[row]; }")
    self.gen_add_code_line("if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_code_line("if constexpr (!GAUSS_NEWTON) {", True)
    self.gen_add_serial_ops()
    self.gen_add_code_lines([
        "T *s_g6 = &s_scratch[" + NV + " + 72]; T *s_T2 = &s_scratch[" + NV + " + 78];",
        "#pragma unroll",
        "for (int b = 0; b < 6; ++b) { T acc = static_cast<T>(0);",
        "    for (int r = 0; r < 6; ++r) acc += s_Ji[6*r + b] * s_Q[r] * s_eq[r]; s_g6[b] = acc; }",
        "// term 1: -J (dM) J through e — d2Integrate at (q_des, eq), contracted with grad_q.",
        "grid::grid_d2Integrate_block<T, false>(s_eq, s_T2);",
        "for (int i = 0; i < 6; ++i) { for (int k = 0; k < 6; ++k) {",
        "    T acc = static_cast<T>(0);",
        "    for (int b = 0; b < 6; ++b) for (int c = 0; c < 6; ++c) for (int m = 0; m < 6; ++m)",
        "        acc += s_g6[b] * s_T2[36*b + 6*c + m] * s_Ji[6*c + i] * s_Ji[6*m + k];",
        "    s_hess[k*" + TN + " + i] -= acc; } }",
        "// term 2 (guide §7.z2 chart slope): + grad_q . d2Integrate(q, 0) — N(0)=I but dN(0)!=0.",
        "T zero6[6] = {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)};",
        "grid::grid_d2Integrate_block<T, false>(zero6, s_T2);",
        "for (int i = 0; i < 6; ++i) { for (int k = 0; k < 6; ++k) {",
        "    T acc = static_cast<T>(0);",
        "    for (int b = 0; b < 6; ++b) acc += s_g6[b] * s_T2[36*b + 6*i + k];",
        "    s_hess[k*" + TN + " + i] += acc; } }",
    ])
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    # ---- fused ----
    self.gen_add_func_doc(
        "quadratic_state_cost_tangent_value_grad_hess: fused value + gradient + hessian",
        ["Convenience fusion; same conventions and ACCUMULATE/GAUSS_NEWTON semantics as the three functions above."],
        ["s_out / s_grad / s_hess are the three outputs",
         "s_x / s_x_des / s_Q / s_scratch as above (scratch sized for the hessian path)"],
        None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false, bool GAUSS_NEWTON = true>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void quadratic_state_cost_tangent_value_grad_hess(T *s_out, T *s_grad, T *s_hess, "
                           "const T *s_x, const T *s_x_des, const T *s_Q, T *s_scratch) {", True)
    self.gen_add_code_line("quadratic_state_cost_tangent<T, ACCUMULATE>(s_out, s_x, s_x_des, s_Q, s_scratch);")
    self.gen_add_code_line("quadratic_state_cost_tangent_gradient<T, ACCUMULATE>(s_grad, s_x, s_x_des, s_Q, s_scratch);")
    self.gen_add_code_line("quadratic_state_cost_tangent_hessian<T, ACCUMULATE, GAUSS_NEWTON>(s_hess, s_x, s_x_des, s_Q, s_scratch);")
    self.gen_add_end_function()


def gen_quadratic_state_cost(self):
    _gen_quadratic_cost_family(self, "state")


def gen_quadratic_input_cost(self):
    _gen_quadratic_cost_family(self, "input")


# ---------------------------------------------------------------------------
# End-effector position cost (value, gradient wrt x=[q;qd], GN hessian J_p^T W J_p).
# ---------------------------------------------------------------------------

def gen_ee_raw_evaluators(self, with_gradient = True):
    """GATO ASK2: raw caller-scratch EE-pose evaluators, NO cost coupling.

    Consumers (constraint row-group layers: EE-position rows, cone frames) need
    pose/Jacobian from another kernel's block WITHOUT the cost math and WITHOUT
    hand-carving the cost internals' arena. These are exactly the cost family's
    evaluation halves: lay out the XmatsHom arena from caller s_scratch, load
    transforms once, run the *_inner(s), sync. Same target resolution as the
    cost family (named fixed target when baked, else the generic family)."""
    nv = self.robot.get_num_vel()
    _tgt = getattr(self, "_ee_target_name", "")
    num_ees = 1 if _tgt else self.robot.get_total_leaf_nodes()

    # ---- ee_pos: the 6*NUM_EE pose (position = rows 0..2 of each EE block) ----
    self.gen_add_func_doc(
        "ee_pos: RAW end-effector pose evaluator (no cost coupling; GATO ASK2)",
        ["Caller-scratch INNER: lays out the EE-pose scratch from s_scratch and calls "
         "grid::end_effector_pose_inner directly, so it is callable from another kernel's "
         "block without aliasing that kernel's dynamic-smem arena.",
         "Fills ALL " + str(num_ees) + " EE block(s); position is rows 0..2 of each 6-row block.",
         "s_scratch must hold >= END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT elements of T, 16B aligned."],
        ["s_end_effector_pose is the 6*NUM_EE pose output",
         "s_q is the joint position vector (size NUM_POS)",
         "s_scratch is caller shared scratch",
         "d_robotModel is the GPU model helpers"],
        None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void ee_pos(T *s_end_effector_pose, const T *s_q, "
                           "T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    self.gen_add_code_line("using namespace grid;")
    _ee_scratch = self.gen_end_effector_pose_inner_temp_mem_size(_tgt)
    self.gen_XmatsHom_helpers_temp_shared_memory_code(_ee_scratch, include_linalg_scratch = True,
                                                      linalg_scratch_bytes = "GRID_EE_LINALG_SHARED_BYTES<T>()",
                                                      arena_base_expr = "s_scratch")
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_end_effector_pose_inner_function_call(fixed_target_name = _tgt)
    self.gen_add_sync()
    self.gen_add_end_function()

    if not with_gradient:
        self.gen_add_code_line("// [grid_plant] ee_pos_gradient skipped: requires 'end_effector_pose_gradient' — not generated.")
        return

    # ---- ee_pos_gradient: pose + full 6 x NV geometric Jacobian per EE ----
    self.gen_add_func_doc(
        "ee_pos_gradient: RAW end-effector pose + Jacobian evaluator (no cost coupling; GATO ASK2)",
        ["Caller-scratch INNER: ONE XmatsHom load feeds BOTH end_effector_pose_inner and "
         "end_effector_pose_gradient_inner (const s_Xhom shared; geometric-Jacobian path, "
         "s_dXhom = nullptr) — same single-load structure as ee_pos_cost_gradient.",
         "Jacobian layout: s_end_effector_pose_gradient[6*" + str(nv) + "*ee + 6*vi + row] "
         "(position rows 0..2, orientation rows 3..5; tangent d/dv convention).",
         "s_scratch must hold >= END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT elements of T, 16B aligned."],
        ["s_end_effector_pose is the 6*NUM_EE pose output",
         "s_end_effector_pose_gradient is the 6*NUM_VEL*NUM_EE Jacobian output",
         "s_q is the joint position vector (size NUM_POS)",
         "s_scratch is caller shared scratch",
         "d_robotModel is the GPU model helpers"],
        None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void ee_pos_gradient(T *s_end_effector_pose, T *s_end_effector_pose_gradient, "
                           "const T *s_q, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    self.gen_add_code_line("using namespace grid;")
    _ee_scratch = max(self.gen_end_effector_pose_inner_temp_mem_size(_tgt),
                      self.gen_end_effector_pose_gradient_inner_temp_mem_size(_tgt))
    self.gen_XmatsHom_helpers_temp_shared_memory_code(_ee_scratch, include_linalg_scratch = True,
                                                      linalg_scratch_bytes = "GRID_EE_LINALG_SHARED_BYTES<T>()",
                                                      arena_base_expr = "s_scratch")
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_end_effector_pose_inner_function_call(fixed_target_name = _tgt)
    self.gen_add_sync()
    self.gen_end_effector_pose_gradient_inner_function_call(fixed_target_name = _tgt,
                                                            updated_var_names = {"s_dXhom_name": "nullptr"})
    self.gen_add_sync()
    self.gen_add_end_function()


def gen_ee_pos_cost(self, with_d2ee = False):
    """ee_pos_cost family. p(q) = grid::end_effector_pose (rows 0..2 of the 6-pose);
    J_p = rows 0..2 of grid::end_effector_pose_gradient (layout
    s_end_effector_pose_gradient[6*NV*ee + 6*vi + row]). Templated on `int EE = 0`.

        r       = p(q) - p_des                              (3-vector)
        value   = 1/2 * sum_{r} W[r] * r[r]^2
        grad_q  = J_p^T W r        (size NV)
        grad_x  = [grad_q ; 0]     (the qd-block is exactly zero)
        GN hess = J_p^T W J_p      (NV x NV block; the qd rows/cols are zero)

    W is a 3-vector of per-axis position weights. ee_pos_cost_hessian is
    templated on GAUSS_NEWTON (default false): the DEFAULT is the TRUE
    (full-Newton) hessian — GN term J_p^T W J_p PLUS the residual-weighted
    curvature term sum_r W[r] (p_r - p_des_r) d^2 p_r/dv^2 via the analytic
    d2ee (grid::end_effector_pose_hessian_inner) — matching the house rule
    that "hessian" means the true analytical object everywhere in GRiD.
    GAUSS_NEWTON=true opts into the PSD J_p^T W J_p approximation (the
    ratified-GN choice existing solver consumers rely on for Cholesky).
    The Newton path requires the analytic d2ee: when with_d2ee is False
    ('end_effector_pose_hessian' not in the algorithm set) the default
    instantiation static_asserts with an actionable message instead of
    silently falling back to GN.
    """
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    # GATO Ask-4: track the TRUE ee_frame. The generic end_effector_pose* family evaluates
    # the last MOVING joint and drops the terminal fixed joint's <origin> (indy7 "EE": 6cm z;
    # iiwa14: 4cm), so every EE cost here was tracking a frame offset from the real TCP. When
    # the robot is generated with a named fixed target we now bottom out at that target's
    # `_<name>` inners. "" / "all" keep the generic family (resolved in gen_all_code).
    # NOTE this changes num_ees to 1 -> it MUST also drive the *_temp_mem_size calls below,
    # not just the call sites: a named target sizes its EE scratch for 1 EE while the generic
    # sizes for one per LEAF NODE, and those differ on a multi-leaf robot (go2: 4 feet).
    _tgt = getattr(self, "_ee_target_name", "")
    num_ees = 1 if _tgt else self.robot.get_total_leaf_nodes()
    nx = nq + nv

    # ---- value ----
    self.gen_add_func_doc(
        "ee_pos_cost: value = 1/2 * sum_r W[r] * (p_r(q) - p_des_r)^2 over the 3 position axes",
        ["Caller-scratch INNER: lays out the EE-pose scratch from s_scratch and calls "
         "grid::end_effector_pose_inner directly (NOT the auto-allocating _device), so it is "
         "callable from another kernel's block without aliasing that kernel's dynamic-smem arena.",
         "EE selects which end-effector (0.." + str(num_ees - 1) + ").",
         "s_end_effector_pose must hold 6*NUM_EE; s_scratch must hold >= "
         "END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT elements of T."],
        ["s_out is the scalar cost output (s_out[0])",
         "s_q is the joint position vector (size NUM_POS)",
         "s_p_des is the desired EE position (3-vector)",
         "s_W is the per-axis position weight (3-vector)",
         "s_end_effector_pose is scratch for the 6*NUM_EE pose (the position is rows 0..2 of EE block)",
         "s_scratch is caller shared scratch for the EE-pose helper (>= END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT, 16B aligned)",
         "d_robotModel is the GPU model helpers"],
        None)
    self.gen_add_code_line("template <typename T, int EE = 0, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void ee_pos_cost(T *s_out, const T *s_q, const T *s_p_des, const T *s_W, "
                           "T *s_end_effector_pose, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    # EE pose via the caller-scratch inner path (NOT the auto-allocating _device, whose
    # extern __shared__ would alias an outer kernel's arena). using namespace grid lets the
    # shared XmatsHom/load/inner emit helpers resolve unqualified (same pattern as plant_step_gradient_kernel).
    self.gen_add_code_line("using namespace grid;")
    _ee_scratch = self.gen_end_effector_pose_inner_temp_mem_size(_tgt)
    self.gen_XmatsHom_helpers_temp_shared_memory_code(_ee_scratch, include_linalg_scratch = True,
                                                      linalg_scratch_bytes = "GRID_EE_LINALG_SHARED_BYTES<T>()",
                                                      arena_base_expr = "s_scratch")
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_end_effector_pose_inner_function_call(fixed_target_name = _tgt)
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("#pragma unroll")
    self.gen_add_code_line("for (int r = 0; r < 3; ++r) { T e = s_end_effector_pose[6*EE + r] - s_p_des[r]; acc += static_cast<T>(0.5) * s_W[r] * e * e; }")
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    # ---- gradient wrt x = [q; qd] (qd block is zero) ----
    self.gen_add_func_doc(
        "ee_pos_cost_gradient: grad_x = [J_p^T W (p - p_des) ; 0], over x = [q; qd]",
        ["Caller-scratch INNER: ONE XmatsHom load feeds BOTH end_effector_pose_inner (for p) and "
         "end_effector_pose_gradient_inner (for J_p) -- s_Xhom is const in both inners, so the local "
         "homogeneous transforms are loaded once and shared (vs the old double-load through two "
         "auto-allocating _device calls). The geometric-Jacobian inner uses only s_Xhom (s_dXhom = "
         "nullptr), so no per-joint d-transform load is needed. Callable from another kernel's block "
         "without aliasing that kernel's dynamic-smem arena.",
         "J_p = rows 0..2 of s_end_effector_pose_gradient, layout s_end_effector_pose_gradient[6*NUM_VEL*ee + 6*vi + row].",
         "The qd-block of the gradient (entries NUM_VEL.." + str(nx - 1) + ") is set to exactly zero.",
         "ACCUMULATE=false overwrites s_grad; true adds (for fusing with a state-cost gradient)."],
        ["s_grad is the gradient over x (size NUM_POS + NUM_VEL = " + str(nx) + ")",
         "s_q / s_p_des / s_W / d_robotModel as above",
         "s_end_effector_pose is 6*NUM_EE pose scratch; s_end_effector_pose_gradient is 6*NUM_VEL*NUM_EE Jacobian scratch",
         "s_scratch is caller shared scratch for the EE-pose+gradient helper "
         "(>= END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT, 16B aligned)"],
        None)
    self.gen_add_code_line("template <typename T, int EE = 0, bool ACCUMULATE = false" + ", bool MUJOCO_OUTPUT = false" + ">")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void ee_pos_cost_gradient(T *s_grad, const T *s_q, const T *s_p_des, const T *s_W, "
                           "T *s_end_effector_pose, T *s_end_effector_pose_gradient, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    # Caller-scratch INNER path: lay out the (shared) XmatsHom + temp + linalg arena from s_scratch,
    # load the local homogeneous transforms ONCE, then call end_effector_pose_inner (p) and
    # end_effector_pose_gradient_inner (J_p) -- both read the same const s_Xhom. using namespace grid
    # lets the unqualified XmatsHom/load/inner emit helpers resolve (same pattern as the value fn).
    self.gen_add_code_line("using namespace grid;")
    _ee_scratch = max(self.gen_end_effector_pose_inner_temp_mem_size(_tgt),
                      self.gen_end_effector_pose_gradient_inner_temp_mem_size(_tgt))
    self.gen_XmatsHom_helpers_temp_shared_memory_code(_ee_scratch, include_linalg_scratch = True,
                                                      linalg_scratch_bytes = "GRID_EE_LINALG_SHARED_BYTES<T>()",
                                                      arena_base_expr = "s_scratch")
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_end_effector_pose_inner_function_call(fixed_target_name = _tgt)
    self.gen_add_sync()
    self.gen_end_effector_pose_gradient_inner_function_call(fixed_target_name = _tgt,
                                                            updated_var_names = {"s_dXhom_name": "nullptr"})
    self.gen_add_sync()
    # grad_q[i] = sum_r J_p[r,i] * W[r] * (p_r - p_des_r)
    self.gen_add_parallel_loop("i", str(nv))
    self.gen_add_code_line("T g = static_cast<T>(0);")
    self.gen_add_code_line("#pragma unroll")
    self.gen_add_code_line("for (int r = 0; r < 3; ++r) {")
    self.gen_add_code_line("    T Jri = s_end_effector_pose_gradient[6*" + str(nv) + "*EE + 6*i + r];")
    self.gen_add_code_line("    T e   = s_end_effector_pose[6*EE + r] - s_p_des[r];")
    self.gen_add_code_line("    g += Jri * s_W[r] * e;")
    self.gen_add_code_line("}")
    self.gen_add_code_line("if (ACCUMULATE) { s_grad[i] += g; } else { s_grad[i] = g; }")
    self.gen_add_end_control_flow()
    # Zero the entire non-q-gradient tail [nv, nx): the meaningful gradient occupies
    # [0, nv); everything after must be zero. Zeroing [nq, nq+nv) left [nv, nq)
    # UNINITIALIZED for floating-base robots (nq>nv) -> stale shared mem (go2 nq=19,
    # nv=18 left s_grad[18] stale). Mirrors com_cost_gradient. Byte-identical fixed-base (nq==nv).
    self.gen_add_code_line("if (!ACCUMULATE) {", True)
    self.gen_add_parallel_loop("i", str(nq))
    self.gen_add_code_line("s_grad[" + str(nv) + " + i] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    if self.robot.floating_base:
        # mjx output: the q-tangent block is at offset 0; its base-LINEAR 3 entries
        # reframe as a covector grad_mjx[0:3] = R . grad_pin[0:3] (G·, G^{-T}=G).
        # s_q is already xyzw (the kernel reorders the wxyz mjx quaternion once).
        self.gen_add_sync()
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        self.gen_mjx_base_rotate("s_grad", q_name="s_q")
        self.gen_add_end_control_flow()
    self.gen_add_end_function()

    # ---- cost hessian: DEFAULT = full Newton (GN + residual-weighted analytic d2ee
    # curvature); GAUSS_NEWTON=true opts into the PSD J_p^T W J_p approximation.
    # ONE function ("hessian" means the true analytical object everywhere in GRiD);
    # the Newton path exists only when the analytic d2ee is in the algorithm set
    # (with_d2ee) — otherwise the default instantiation static_asserts (actionable
    # compile error, never a silent GN fallback).
    self.gen_add_func_doc(
        "ee_pos_cost_hessian: EE-position cost hessian. DEFAULT = full Newton = J_p^T W J_p + sum_r W[r] (p_r - p_des_r) d2p_r/dv2; GAUSS_NEWTON=true = ratified PSD GN term only",
        ["DEFAULT (GAUSS_NEWTON=false) folds the exact residual-weighted EE-curvature term via the "
         "analytic d2ee (grid::end_effector_pose_hessian_inner). Not necessarily PSD away from the "
         "solution -- callers must regularize (e.g. the solver's rho schedule).",
         "GAUSS_NEWTON=true keeps the ratified PSD choice H = J_p^T diag(W) J_p (curvature dropped); "
         "s_p_des, s_end_effector_pose and s_end_effector_pose_hessian may be nullptr in that case.",
         "PSD_CLAMP=true (opt-in, default false) eigen-clamps the NV x NV q-block to >= psd_reg_eps "
         "(glass::eig_clamp) so the returned hessian is SPD and directly factorable even when the Newton "
         "curvature is indefinite -- a guaranteed-PSD alternative to a caller-side rho schedule. Costs one "
         "block-cooperative Jacobi eigensolve; s_scratch must hold NV*NV + eig_clamp_scratch when set.",
         "Caller-scratch INNER: ONE XmatsHom load feeds the needed inners (GN: gradient_inner only, "
         "s_dXhom = nullptr; Newton: pose_inner for the residual + hessian_inner, which also fills "
         "the gradient buffer), so it is callable from another kernel's block without aliasing that "
         "kernel's dynamic-smem arena.",
         "d2p layout: s_end_effector_pose_hessian[6*NV*NV*ee + r*NV*NV + vi*NV + vj] (pose row r, "
         "joint pair (vi, vj); tangent d/dv convention, position rows 0..2 symmetric in (vi, vj)).",
         "Dense column-major NX x NX (NX = NUM_POS + NUM_VEL = " + str(nx) + "); only the top-left NUM_VEL x NUM_VEL q-block is non-zero.",
         "ACCUMULATE=false overwrites the whole NX x NX block; true adds the q-block into an existing hessian."],
        ["s_hess is the dense x-hessian output (size " + str(nx) + "*" + str(nx) + ", column-major)",
         "s_q / s_p_des / s_W / d_robotModel as above (s_p_des is unused under GAUSS_NEWTON)",
         "s_end_effector_pose is 6*NUM_EE pose scratch (Newton only); s_end_effector_pose_gradient is "
         "6*NUM_VEL*NUM_EE Jacobian scratch; s_end_effector_pose_hessian is 6*NUM_VEL*NUM_VEL*NUM_EE "
         "d2ee scratch (Newton only)",
         "s_scratch is caller shared scratch for the EE helpers (GN: >= END_EFFECTOR_POSE_GRADIENT_"
         "DYNAMIC_SHARED_MEM_COUNT; Newton: >= END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_COUNT "
         "covers it; 16B aligned)"],
        None)
    self.gen_add_code_line("template <typename T, int EE = 0, bool ACCUMULATE = false, bool GAUSS_NEWTON = false" + ", bool MUJOCO_OUTPUT = false" + ", bool PSD_CLAMP = false" + ">")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void ee_pos_cost_hessian(T *s_hess, const T *s_q, const T *s_p_des, const T *s_W, "
                           "T *s_end_effector_pose, T *s_end_effector_pose_gradient, T *s_end_effector_pose_hessian, "
                           "T *s_scratch, const grid::robotModel<T> *d_robotModel, T psd_reg_eps = static_cast<T>(1e-6)) {", True)
    self.gen_add_code_line("using namespace grid;")
    if not with_d2ee:
        self.gen_add_code_line("static_assert(GAUSS_NEWTON, \"full-Newton ee_pos_cost_hessian requires 'end_effector_pose_hessian' in the algorithm set; pass GAUSS_NEWTON=true for the J_p^T W J_p approximation\");")
    if with_d2ee:
        # Newton branch: ONE XmatsHom load feeds pose_inner (residual p) then hessian_inner
        # (J_p + d2p, out-in-smem to the caller buffer). Arena sized for the larger inner
        # (they run sequentially and share the temp). Branch-scoped so the GN instantiation
        # keeps its smaller gradient-only arena contract.
        self.gen_add_code_line("if constexpr (!GAUSS_NEWTON) {", True)
        _ee_scratch_newton = max(self.gen_end_effector_pose_inner_temp_mem_size(_tgt),
                                 self.gen_end_effector_pose_hessian_inner_temp_mem_size(_tgt))
        self.gen_XmatsHom_helpers_temp_shared_memory_code(_ee_scratch_newton, include_linalg_scratch = True,
                                                          linalg_scratch_bytes = "GRID_EE_LINALG_SHARED_BYTES<T>()",
                                                          arena_base_expr = "s_scratch")
        self.gen_load_update_XmatsHom_helpers_function_call()
        self.gen_end_effector_pose_inner_function_call(fixed_target_name = _tgt)
        self.gen_add_sync()
        self.gen_end_effector_pose_hessian_inner_function_call(fixed_target_name = _tgt,
                                                               out_in_smem_expr = "true")
        self.gen_add_sync()
        self.gen_add_end_control_flow()
        self.gen_add_code_line("else {", True)
    # GN path (the ONLY path when with_d2ee is False): lay out the EE-pose-gradient arena
    # from s_scratch, load XmatsHom, call the geometric-Jacobian inner (s_dXhom = nullptr).
    _ee_scratch = self.gen_end_effector_pose_gradient_inner_temp_mem_size(_tgt)
    self.gen_XmatsHom_helpers_temp_shared_memory_code(_ee_scratch, include_linalg_scratch = True,
                                                      linalg_scratch_bytes = "GRID_EE_LINALG_SHARED_BYTES<T>()",
                                                      arena_base_expr = "s_scratch")
    self.gen_load_update_XmatsHom_helpers_function_call()
    self.gen_end_effector_pose_gradient_inner_function_call(fixed_target_name = _tgt,
                                                            updated_var_names = {"s_dXhom_name": "nullptr"})
    self.gen_add_sync()
    if with_d2ee:
        self.gen_add_end_control_flow()
    # H[i,j] = sum_r J_p[r,i] W[r] J_p[r,j] (+ W[r] (p_r - p_des_r) d2p[r,i,j] under Newton),
    # column-major over the full NX x NX block (zero outside the NUM_VEL x NUM_VEL q-block).
    self.gen_add_parallel_loop("ind", str(nx * nx))
    self.gen_add_code_line("int row = ind % " + str(nx) + ";")
    self.gen_add_code_line("int col = ind / " + str(nx) + ";")
    self.gen_add_code_line("T h = static_cast<T>(0);")
    self.gen_add_code_line("if (row < " + str(nv) + " && col < " + str(nv) + ") {")
    self.gen_add_code_line("    #pragma unroll")
    self.gen_add_code_line("    for (int r = 0; r < 3; ++r) {")
    self.gen_add_code_line("        T Jri = s_end_effector_pose_gradient[6*" + str(nv) + "*EE + 6*row + r];")
    self.gen_add_code_line("        T Jrj = s_end_effector_pose_gradient[6*" + str(nv) + "*EE + 6*col + r];")
    self.gen_add_code_line("        h += Jri * s_W[r] * Jrj;")
    if with_d2ee:
        self.gen_add_code_line("        if constexpr (!GAUSS_NEWTON) {")
        self.gen_add_code_line("            T e   = s_end_effector_pose[6*EE + r] - s_p_des[r];")
        self.gen_add_code_line("            T Hij = s_end_effector_pose_hessian[6*" + str(nv * nv) + "*EE + r*" + str(nv * nv) + " + row*" + str(nv) + " + col];")
        self.gen_add_code_line("            h += s_W[r] * e * Hij;")
        self.gen_add_code_line("        }")
    self.gen_add_code_line("    }")
    self.gen_add_code_line("}")
    self.gen_add_code_line("if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }")
    self.gen_add_end_control_flow()
    # PSD projection (opt-in): floor the q-block eigenvalues at psd_reg_eps so a direct
    # solver can factor the Newton hessian even where the residual-weighted curvature makes
    # it indefinite. Applied in the pin output frame BEFORE the mjx congruence (G=blockdiag(R,I)
    # is a congruence, which preserves PSD). The NV x NV q-block is column-major with stride NX
    # inside s_hess, so gather it contiguous, eig_clamp in place, scatter back. The EE-inner
    # arena (s_scratch) is dead here, so it doubles as the eig_clamp workspace
    # (NV*NV gathered block + 2*NV*NV+2*NV+4 syev scratch); callers that set PSD_CLAMP=true must
    # size s_scratch to at least that (>= END_EFFECTOR_POSE_HESSIAN inner need covers small robots).
    self.gen_add_code_line("if constexpr (PSD_CLAMP) {", True)
    self.gen_add_sync()
    self.gen_add_code_line("T *s_psd_qb = s_scratch; T *s_psd_eig = &s_scratch[" + str(nv * nv) + "];")
    self.gen_add_parallel_loop("ind", str(nv * nv))
    self.gen_add_code_line("int r = ind % " + str(nv) + "; int c = ind / " + str(nv) + ";")
    self.gen_add_code_line("s_psd_qb[r + " + str(nv) + "*c] = s_hess[r + " + str(nx) + "*c];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_code_line("glass::eig_clamp<T, " + str(nv) + ">(s_psd_qb, psd_reg_eps, s_psd_eig);")
    self.gen_add_sync()
    self.gen_add_parallel_loop("ind", str(nv * nv))
    self.gen_add_code_line("int r = ind % " + str(nv) + "; int c = ind / " + str(nv) + ";")
    self.gen_add_code_line("s_hess[r + " + str(nx) + "*c] = s_psd_qb[r + " + str(nv) + "*c];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_control_flow()
    if self.robot.floating_base:
        # mjx output: the active q-block is at offset 0 of the NX x NX col-major
        # hessian; congruence reframes its base-LINEAR rows/cols 0:3 (the zero qd
        # rows/cols >= NV are untouched). NO frame-correction term in EITHER mode
        # (exact in the pin convention; the mjx-frame Newton curvature correction
        # is a follow-up). s_q is already xyzw (kernel reordered once).
        self.gen_add_sync()
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        self.gen_mjx_congruence("s_hess", str(nx), q_name="s_q")
        self.gen_add_end_control_flow()
    self.gen_add_end_function()



# ---------------------------------------------------------------------------
# CoM-tracking cost (R3 plant hook). Direct clone of ee_pos_cost with the EE
# position/Jacobian replaced by the CoM position / CoM Jacobian from
# grid::com_device (which writes [p_com(3); J_com(3 x NUM_VEL, column-major)]).
#   r       = p_com(q) - p_des                          (3-vector)
#   value   = 1/2 sum_r W[r] r[r]^2
#   grad_x  = [J_com^T W r ; 0]
#   GN hess = J_com^T W J_com  (q-block of the NX x NX hessian; qd rows/cols 0)
# ---------------------------------------------------------------------------

def gen_com_cost(self):
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    nx = nq + nv
    # ---- value ----
    self.gen_add_func_doc(
        "com_cost: value = 1/2 sum_r W[r] (p_com_r(q) - p_des_r)^2 over the 3 CoM axes",
        ["Caller-scratch INNER: lays out the centroidal arena from s_scratch and runs centroidal_inner "
         "(fills s_com = CoM pos, s_A = CMM, s_extra = mass), NOT the auto-allocating grid::com_device, "
         "so it is callable from another kernel's block without aliasing that kernel's dynamic-smem arena.",
         "s_scratch must hold >= COM_DYNAMIC_SHARED_MEM_COUNT elements of T (16B aligned)."],
        ["s_out scalar cost", "s_q joint positions", "s_p_des desired CoM (3)",
         "s_W per-axis weight (3)", "s_scratch caller centroidal-arena scratch (>= COM_DYNAMIC_SHARED_MEM_COUNT)",
         "d_robotModel GPU model helpers"],
        None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void com_cost(T *s_out, const T *s_q, const T *s_p_des, const T *s_W, "
                           "T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    # Caller-scratch INNER: lay out the centroidal arena from s_scratch + run centroidal_inner
    # (fills s_com = CoM pos, s_A = CMM, s_extra = mass) instead of the auto-allocating com_device.
    _emit_centroidal_caller_scratch(self)
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("#pragma unroll")
    self.gen_add_code_line("for (int r = 0; r < 3; ++r) { T e = s_com[r] - s_p_des[r]; acc += static_cast<T>(0.5) * s_W[r] * e * e; }")
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    # ---- gradient wrt x = [q; qd] (qd block zero) ----
    self.gen_add_func_doc(
        "com_cost_gradient: grad_x = [J_com^T W (p_com - p_des) ; 0]",
        ["Caller-scratch INNER (centroidal_inner from s_scratch): J_com[r,vi] = s_A[r + 6*vi] / mass "
         "(top-3 rows of the CMM s_A / mass)."],
        ["s_grad gradient over x (" + str(nx) + ")", "s_q / s_p_des / s_W / d_robotModel as above",
         "s_scratch caller centroidal-arena scratch (>= COM_DYNAMIC_SHARED_MEM_COUNT)"], None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false" + ", bool MUJOCO_OUTPUT = false" + ">")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void com_cost_gradient(T *s_grad, const T *s_q, const T *s_p_des, const T *s_W, "
                           "T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    # Caller-scratch INNER: s_A = CMM (6 x NUM_VEL col-major), s_com = CoM pos, s_extra[0] = mass.
    # J_com[r, vi] = s_A[r + 6*vi] / mass (the CoM Jacobian = top-3 rows of the CMM / mass).
    _emit_centroidal_caller_scratch(self)
    self.gen_add_code_line("T inv_m = static_cast<T>(1) / s_extra[0];")
    self.gen_add_parallel_loop("i", str(nv))
    self.gen_add_code_line("T g = static_cast<T>(0);")
    self.gen_add_code_line("#pragma unroll")
    self.gen_add_code_line("for (int r = 0; r < 3; ++r) { T Jri = s_A[r + 6*i] * inv_m; T e = s_com[r] - s_p_des[r]; g += Jri * s_W[r] * e; }")
    self.gen_add_code_line("if (ACCUMULATE) { s_grad[i] += g; } else { s_grad[i] = g; }")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("if (!ACCUMULATE) {", True)
    # Zero the entire non-q-gradient tail [nv, nx): the meaningful gradient occupies the
    # first nv slots [0, nv), everything after must be zero (matches the GN hessian
    # convention below, nonzero only on [0,nv)x[0,nv)). Zeroing [nq, nq+nv) left [nv, nq)
    # UNINITIALIZED for floating-base robots (nq>nv) -> stale shared mem (go2 nq=19,nv=18
    # left s_grad[18] stale). For fixed-base (nq==nv) this is byte-identical to the old loop.
    self.gen_add_parallel_loop("i", str(nq))
    self.gen_add_code_line("s_grad[" + str(nv) + " + i] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    if self.robot.floating_base:
        # mjx output: the q-tangent block is at offset 0; reframe its base-LINEAR 3
        # entries as a covector grad_mjx[0:3] = R . grad_pin[0:3]. s_q is xyzw (kernel).
        self.gen_add_sync()
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        self.gen_mjx_base_rotate("s_grad", q_name="s_q")
        self.gen_add_end_control_flow()
    self.gen_add_end_function()

    # ---- GN hessian J_com^T W J_com over the q-block of x ----
    self.gen_add_func_doc(
        "com_cost_hessian: Gauss-Newton hessian = J_com^T diag(W) J_com in the q-block of the x-hessian",
        ["Caller-scratch INNER (centroidal_inner from s_scratch); J_com[r,vi] = s_A[r + 6*vi] / mass.",
         "Dense column-major NX x NX; only the top-left NUM_VEL x NUM_VEL q-block is non-zero."],
        ["s_hess dense x-hessian (" + str(nx*nx) + ")", "s_q / s_W / d_robotModel as above",
         "s_scratch caller centroidal-arena scratch (>= COM_DYNAMIC_SHARED_MEM_COUNT)"], None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false" + ", bool MUJOCO_OUTPUT = false" + ">")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void com_cost_hessian(T *s_hess, const T *s_q, const T *s_W, "
                           "T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    # Caller-scratch INNER: J_com[r, vi] = s_A[r + 6*vi] / mass (top-3 CMM rows / mass).
    _emit_centroidal_caller_scratch(self)
    self.gen_add_code_line("T inv_m = static_cast<T>(1) / s_extra[0];")
    self.gen_add_parallel_loop("ind", str(nx * nx))
    self.gen_add_code_line("int row = ind % " + str(nx) + "; int col = ind / " + str(nx) + ";")
    self.gen_add_code_line("T h = static_cast<T>(0);")
    self.gen_add_code_line("if (row < " + str(nv) + " && col < " + str(nv) + ") {")
    self.gen_add_code_line("    #pragma unroll")
    self.gen_add_code_line("    for (int r = 0; r < 3; ++r) { T Jri = s_A[r + 6*row] * inv_m; T Jrj = s_A[r + 6*col] * inv_m; h += Jri * s_W[r] * Jrj; }")
    self.gen_add_code_line("}")
    self.gen_add_code_line("if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }")
    self.gen_add_end_control_flow()
    if self.robot.floating_base:
        # mjx output: q-block at offset 0; congruence reframes base rows/cols 0:3
        # of the NX x NX col-major hessian (zero qd rows/cols untouched). s_q xyzw.
        self.gen_add_sync()
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        self.gen_mjx_congruence("s_hess", str(nx), q_name="s_q")
        self.gen_add_end_control_flow()
    self.gen_add_end_function()


# ---------------------------------------------------------------------------
# Centroidal-momentum-tracking cost (R2 plant hook). The momentum h = A(q) qd
# has velocity-Jacobian J_h = A (the CMM), so this is the same value/grad/GN-hess
# pattern with p->h, J->A but the variable is x=[q;qd] and h depends on qd
# linearly: grad_qd = A^T W r, GN-hess qd-block = A^T W A (the q-dependence of A
# is dropped, Gauss-Newton style, matching the ee_pos_cost ratified choice).
# grid::ccrba_device writes [A (6 x NUM_VEL, column-major); h (6)].
# ---------------------------------------------------------------------------

def gen_momentum_cost(self):
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    nx = nq + nv
    self.gen_add_func_doc(
        "momentum_cost: value = 1/2 sum_r W[r] (h_r(q,qd) - h_des_r)^2 over the 6 centroidal components",
        ["Caller-scratch INNER: lays out the centroidal arena from s_scratch and runs centroidal_inner "
         "(fills s_A = CMM 6 x NUM_VEL col-major), NOT the auto-allocating grid::ccrba_device. "
         "The momentum h = A*qd is computed here from s_A (ccrba's separate h-output is not in the arena).",
         "s_scratch must hold >= CCRBA_DYNAMIC_SHARED_MEM_COUNT elements of T (16B aligned)."],
        ["s_out scalar cost", "s_q / s_qd joint position/velocity", "s_h_des desired momentum (6)",
         "s_W per-component weight (6)", "s_scratch caller centroidal-arena scratch (>= CCRBA_DYNAMIC_SHARED_MEM_COUNT)",
         "d_robotModel GPU model helpers"],
        None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void momentum_cost(T *s_out, const T *s_q, const T *s_qd, const T *s_h_des, const T *s_W, "
                           "T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    # Caller-scratch INNER: s_A = CMM (6 x NUM_VEL col-major). The residual
    # e = A*qd - h_des is precomputed once into s_e[6] (ccrba_device's separate
    # h-output is not in the centroidal arena), then reduced into the value.
    _emit_centroidal_caller_scratch(self)
    _emit_momentum_residual_precompute(self, nv)
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("#pragma unroll")
    self.gen_add_code_line("for (int r = 0; r < 6; ++r) { T e = s_e[r]; acc += static_cast<T>(0.5) * s_W[r] * e * e; }")
    self.gen_add_code_line("if (ACCUMULATE) { s_out[0] += acc; } else { s_out[0] = acc; }")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    # gradient wrt x = [q; qd]; the q-block is dropped (GN on A), qd-block = A^T W r
    self.gen_add_func_doc(
        "momentum_cost_gradient: grad_x = [0 ; A^T W (h - h_des)] (q-block dropped, GN on A)",
        ["Caller-scratch INNER (centroidal_inner from s_scratch): A = s_A[r + 6*vi] (6 x NUM_VEL "
         "column-major, the CMM); h = A*qd; J_h = A (GN drops dA/dq)."],
        ["s_grad gradient over x (" + str(nx) + ")", "s_q / s_qd / s_h_des / s_W / d_robotModel as above",
         "s_scratch caller centroidal-arena scratch (>= CCRBA_DYNAMIC_SHARED_MEM_COUNT)"], None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false" + ", bool MUJOCO_OUTPUT = false" + ">")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void momentum_cost_gradient(T *s_grad, const T *s_q, const T *s_qd, const T *s_h_des, const T *s_W, "
                           "T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    # Caller-scratch INNER: s_A = CMM. J_h = A (the qd-Jacobian of h = A*qd, GN drops
    # dA/dq). The residual e = A*qd - h_des is precomputed once into s_e[6], so each
    # per-DOF thread reads s_e[r] instead of recomputing the full h (O(NV) not O(NV^2)).
    _emit_centroidal_caller_scratch(self)
    _emit_momentum_residual_precompute(self, nv)
    self.gen_add_code_line("if (!ACCUMULATE) {", True)
    self.gen_add_parallel_loop("i", str(nq))
    self.gen_add_code_line("s_grad[i] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_parallel_loop("i", str(nv))
    self.gen_add_code_line("T g = static_cast<T>(0);")
    self.gen_add_code_line("#pragma unroll")
    self.gen_add_code_line("for (int r = 0; r < 6; ++r) { g += s_A[r + 6*i] * s_W[r] * s_e[r]; }")
    self.gen_add_code_line("if (ACCUMULATE) { s_grad[" + str(nq) + " + i] += g; } else { s_grad[" + str(nq) + " + i] = g; }")
    self.gen_add_end_control_flow()
    if self.robot.floating_base:
        # mjx output: the active qd-tangent block is at offset NQ; its base-LINEAR 3
        # entries are at s_grad[NQ:NQ+3]. Reframe as a covector via base_rotate on the
        # SUB-POINTER (s_grad + NQ), NEVER &s_grad[NQ]. s_q is xyzw (kernel reordered).
        self.gen_add_sync()
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        self.gen_mjx_base_rotate("(s_grad + " + str(nq) + ")", q_name="s_q")
        self.gen_add_end_control_flow()
    self.gen_add_end_function()

    # GN hessian A^T W A in the qd-block of the NX x NX hessian
    self.gen_add_func_doc(
        "momentum_cost_hessian: Gauss-Newton hessian = A^T diag(W) A in the qd-block of the x-hessian",
        ["Caller-scratch INNER (centroidal_inner from s_scratch): A = s_A[r + 6*vi] (the CMM).",
         "Dense column-major NX x NX; only the bottom-right NUM_VEL x NUM_VEL qd-block is non-zero."],
        ["s_hess dense x-hessian (" + str(nx*nx) + ")", "s_q / s_qd / s_W / d_robotModel as above",
         "s_scratch caller centroidal-arena scratch (>= CCRBA_DYNAMIC_SHARED_MEM_COUNT)"], None)
    self.gen_add_code_line("template <typename T, bool ACCUMULATE = false" + ", bool MUJOCO_OUTPUT = false" + ">")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void momentum_cost_hessian(T *s_hess, const T *s_q, const T *s_qd, const T *s_W, "
                           "T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    # Caller-scratch INNER: GN hessian A^T diag(W) A in the qd-block (A = s_A = CMM).
    _emit_centroidal_caller_scratch(self)
    self.gen_add_parallel_loop("ind", str(nx * nx))
    self.gen_add_code_line("int row = ind % " + str(nx) + "; int col = ind / " + str(nx) + ";")
    self.gen_add_code_line("T h = static_cast<T>(0);")
    self.gen_add_code_line("if (row >= " + str(nq) + " && col >= " + str(nq) + ") {")
    self.gen_add_code_line("    int vi = row - " + str(nq) + "; int vj = col - " + str(nq) + ";")
    self.gen_add_code_line("    #pragma unroll")
    self.gen_add_code_line("    for (int r = 0; r < 6; ++r) { T Ari = s_A[r + 6*vi]; T Arj = s_A[r + 6*vj]; h += Ari * s_W[r] * Arj; }")
    self.gen_add_code_line("}")
    self.gen_add_code_line("if (ACCUMULATE) { s_hess[ind] += h; } else { s_hess[ind] = h; }")
    self.gen_add_end_control_flow()
    if self.robot.floating_base:
        # mjx output: the active qd-block is at offset NQ of the NX x NX col-major
        # hessian; its base-LINEAR rows/cols are at NQ:NQ+3 (NOT 0:3), so the shared
        # gen_mjx_congruence (hardcoded 0,1,2) cannot be used — emit the identical
        # R-congruence on rows/cols NQ:NQ+3. NO frame-correction term (GN). s_q xyzw.
        self.gen_add_sync()
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        _gen_cost_congruence_at_offset(self, "s_hess", nx, nq, q_name="s_q")
        self.gen_add_end_control_flow()
    self.gen_add_end_function()


# ---------------------------------------------------------------------------
# tracking_cost PRESET: one composition of the per-term cost inners above into
# GATO's batched-SQP recipe (EE-pose tracking + qd quadratic + u quadratic +
# joint-position/velocity/torque log-barriers), emitting the SEPARATE state /
# input blocks BSQP wants (value scalar; gradient s_qk(NX)+s_rk(NU); hessian
# s_Qk(NX*NX)+s_Rk(NU*NU)).
#
# This is SUGAR, not a ceiling: the per-term inners (ee_pos_cost,
# quadratic_state/input_cost, joint_*_barrier) are the public API — a user
# composes any mix (joint-space vs EE vs CoM tracking, +/- barriers, different
# regularization) by calling them directly with the uniform ACCUMULATE contract.
# The preset just wires up the common case. Running-vs-terminal is the caller's:
# write terminal weights into the weight buffers at the terminal knot (contract,
# NOT a baked KNOT_POINTS-1 branch). A zero weight / +/-inf bound disables a term
# cleanly (barrier mu=0 => 0 value/grad/hess; EE/quadratic zero weight => 0), so
# the single preset covers the whole family by weight selection.
#
# FIXED-BASE ONLY (NUM_POS == NUM_VEL): the mjx-output reframe must be applied
# ONCE to the composed blocks (per-term reframing would double-rotate), so the
# floating-base preset is a follow-up. gen_grid_plant gates the emit on the
# fixed-base profile; the per-term inners still carry their own MUJOCO_OUTPUT
# path for users who compose floating costs manually with a single final reframe.
# ---------------------------------------------------------------------------

def gen_tracking_cost_preset(self):
    """Emit `tracking_cost` / `tracking_cost_gradient` / `tracking_cost_hessian`
    (fixed-base): the GATO BSQP tracking recipe composed from the per-term cost
    inners via the uniform ACCUMULATE contract. Caller-scratch throughout."""
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    nu = nv
    nx = nq + nv
    NQ, NV, NU, NX = str(nq), str(nv), str(nu), str(nx)

    # Shared param-doc fragments (the weight/bound/target buffers are the
    # composition contract; the caller packs running OR terminal weights).
    weight_doc = [
        "s_x / s_u are the current state [q; qd] and control (sizes " + NX + " / " + NU + ")",
        "s_x_des / s_u_des / s_ee_des are the targets (state " + NX + ", input " + NU + ", EE position 3)",
        "s_Q / s_R are the quadratic state/input diagonal weights (sizes " + NX + " / " + NU + "); "
        "s_W is the per-axis EE position weight (3). Zero a weight to disable that term.",
        "s_q_lower/upper + mu_q, s_qd_lower/upper + mu_qd, s_u_lower/upper + mu_u are the "
        "position / velocity / torque log-barrier bounds + weights (mu=0 or +/-inf bound disables).",
        "EE selects the end-effector; running-vs-terminal weighting is caller-supplied (write terminal "
        "weights at the terminal knot — contract, not a baked branch).",
    ]

    # ---- value: total scalar cost (EE + qd-quad + u-quad + 3 barriers) ----
    self.gen_add_func_doc(
        "tracking_cost: total scalar cost = ee_pos_cost + quadratic_state_cost + quadratic_input_cost "
        "+ joint_{position,velocity,torque}_barrier (GATO BSQP recipe, composed from the per-term inners)",
        ["PRESET (sugar): one composition of the public per-term cost inners; users compose other mixes directly.",
         "ACCUMULATE=false overwrites s_out[0]; true adds (the first term carries ACCUMULATE, the rest add).",
         "s_end_effector_pose holds 6*NUM_EE; s_scratch must hold >= END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_COUNT (and >= NX).",
         "Fixed-base only (NUM_POS == NUM_VEL); floating-base composition is a follow-up."],
        ["s_out is the scalar total-cost output (s_out[0])"] + weight_doc +
        ["s_end_effector_pose / s_scratch are caller EE-pose scratch; d_robotModel is the GPU model"],
        None)
    self.gen_add_code_line("template <typename T, int EE = 0, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(
        "void tracking_cost(T *s_out, const T *s_x, const T *s_u, "
        "const T *s_x_des, const T *s_u_des, const T *s_ee_des, "
        "const T *s_Q, const T *s_R, const T *s_W, "
        "const T *s_q_lower, const T *s_q_upper, const T mu_q, "
        "const T *s_qd_lower, const T *s_qd_upper, const T mu_qd, "
        "const T *s_u_lower, const T *s_u_upper, const T mu_u, "
        "T *s_end_effector_pose, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    # Each term reuses s_scratch (EE arena, then tiny reduction buffers) and the
    # s_out[0] accumulator, so sync between every term. The first (EE) carries the
    # preset ACCUMULATE; the rest always add.
    self.gen_add_code_line("ee_pos_cost<T, EE, ACCUMULATE>(s_out, s_x, s_ee_des, s_W, s_end_effector_pose, s_scratch, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("quadratic_state_cost<T, true>(s_out, s_x, s_x_des, s_Q, s_scratch);")
    self.gen_add_sync()
    self.gen_add_code_line("quadratic_input_cost<T, true>(s_out, s_u, s_u_des, s_R, s_scratch);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_position_barrier<T>(s_out, s_x, s_q_lower, s_q_upper, mu_q, s_scratch);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_velocity_barrier<T>(s_out, s_x, s_qd_lower, s_qd_upper, mu_qd, s_scratch);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_torque_barrier<T>(s_out, s_u, s_u_lower, s_u_upper, mu_u, s_scratch);")
    self.gen_add_end_function()

    # ---- gradient: s_qk (NX, state block) + s_rk (NU, input block) ----
    self.gen_add_func_doc(
        "tracking_cost_gradient: s_qk (state gradient, NX) + s_rk (input gradient, NU), composed from the per-term gradient inners",
        ["State block s_qk = ee_pos_cost_gradient (J^T W r in q-block, 0 in qd) + quadratic_state_cost_gradient "
         "+ joint_position_barrier_gradient (q-block) + joint_velocity_barrier_gradient (qd-block).",
         "Input block s_rk = quadratic_input_cost_gradient + joint_torque_barrier_gradient.",
         "ACCUMULATE=false overwrites the blocks; true adds into them (the EE / input-quadratic terms carry ACCUMULATE, the rest add).",
         "Sync between every accumulating term (different inners own different threads per index).",
         "s_scratch must hold >= END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT. Fixed-base only."],
        ["s_qk is the state-block gradient output (size NX = " + NX + ")",
         "s_rk is the input-block gradient output (size NU = " + NU + ")"] + weight_doc +
        ["s_end_effector_pose (6*NUM_EE) / s_end_effector_pose_gradient (6*NUM_VEL*NUM_EE) / s_scratch are caller EE scratch",
         "d_robotModel is the GPU model"],
        None)
    self.gen_add_code_line("template <typename T, int EE = 0, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(
        "void tracking_cost_gradient(T *s_qk, T *s_rk, const T *s_x, const T *s_u, "
        "const T *s_x_des, const T *s_u_des, const T *s_ee_des, "
        "const T *s_Q, const T *s_R, const T *s_W, "
        "const T *s_q_lower, const T *s_q_upper, const T mu_q, "
        "const T *s_qd_lower, const T *s_qd_upper, const T mu_qd, "
        "const T *s_u_lower, const T *s_u_upper, const T mu_u, "
        "T *s_end_effector_pose, T *s_end_effector_pose_gradient, T *s_scratch, "
        "const grid::robotModel<T> *d_robotModel) {", True)
    self.gen_add_code_line("// ---- state-block gradient s_qk (NX) ----")
    self.gen_add_code_line("ee_pos_cost_gradient<T, EE, ACCUMULATE>(s_qk, s_x, s_ee_des, s_W, "
                           "s_end_effector_pose, s_end_effector_pose_gradient, s_scratch, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("quadratic_state_cost_gradient<T, true>(s_qk, s_x, s_x_des, s_Q);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_position_barrier_gradient<T, 0, 0>(s_qk, s_x, s_q_lower, s_q_upper, mu_q);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_velocity_barrier_gradient<T, " + NQ + ", " + NQ + ">(s_qk, s_x, s_qd_lower, s_qd_upper, mu_qd);")
    self.gen_add_sync()
    self.gen_add_code_line("// ---- input-block gradient s_rk (NU) ----")
    self.gen_add_code_line("quadratic_input_cost_gradient<T, ACCUMULATE>(s_rk, s_u, s_u_des, s_R);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_torque_barrier_gradient<T, 0, 0>(s_rk, s_u, s_u_lower, s_u_upper, mu_u);")
    self.gen_add_end_function()

    # ---- hessian: s_Qk (NX*NX, GN state block) + s_Rk (NU*NU, input block) ----
    self.gen_add_func_doc(
        "tracking_cost_hessian: s_Qk (state GN hessian, NX*NX col-major) + s_Rk (input hessian, NU*NU), composed from the per-term hessian inners",
        ["State block s_Qk = ee_pos_cost_hessian (J^T W J in q-block) + quadratic_state_cost_hessian (diag) "
         "+ joint_position_barrier_hessian (q diagonal) + joint_velocity_barrier_hessian (qd diagonal).",
         "Input block s_Rk = quadratic_input_cost_hessian (diag) + joint_torque_barrier_hessian (diagonal).",
         "Gauss-Newton: the EE value-curvature is dropped (matches the per-term GN choice).",
         "ACCUMULATE=false overwrites; true adds (the EE / input-quadratic terms carry ACCUMULATE, the rest add).",
         "s_scratch must hold >= END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_COUNT. Fixed-base only."],
        ["s_Qk is the state GN hessian output (size NX*NX = " + str(nx * nx) + ", column-major)",
         "s_Rk is the input hessian output (size NU*NU = " + str(nu * nu) + ", column-major)",
         "s_x / s_u are the current state and control",
         "s_Q / s_R / s_W are the quadratic state/input + EE weights",
         "s_q_lower/upper + mu_q, s_qd_lower/upper + mu_qd, s_u_lower/upper + mu_u are the barrier bounds + weights",
         "s_end_effector_pose_gradient (6*NUM_VEL*NUM_EE) / s_scratch are caller EE scratch; d_robotModel is the GPU model"],
        None)
    self.gen_add_code_line("template <typename T, int EE = 0, bool ACCUMULATE = false>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(
        "void tracking_cost_hessian(T *s_Qk, T *s_Rk, const T *s_x, const T *s_u, "
        "const T *s_Q, const T *s_R, const T *s_W, "
        "const T *s_q_lower, const T *s_q_upper, const T mu_q, "
        "const T *s_qd_lower, const T *s_qd_upper, const T mu_qd, "
        "const T *s_u_lower, const T *s_u_upper, const T mu_u, "
        "T *s_end_effector_pose_gradient, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
    self.gen_add_code_line("// ---- state-block GN hessian s_Qk (NX*NX) ----")
    # GAUSS_NEWTON=true pinned: tracking_cost_hessian is the solver-facing composite and
    # its consumers (GATO/MPCGPU rho schedules) rely on the PSD GN term; it also has no
    # p_des/d2ee buffers in its signature. Newton tracking cost = follow-up (needs s_p_des
    # param + d2ee scratch plumbed through).
    self.gen_add_code_line("ee_pos_cost_hessian<T, EE, ACCUMULATE, true>(s_Qk, s_x, nullptr, s_W, nullptr, s_end_effector_pose_gradient, nullptr, s_scratch, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("quadratic_state_cost_hessian<T, true>(s_Qk, s_Q);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_position_barrier_hessian<T, " + NX + ", 0, 0>(s_Qk, s_x, s_q_lower, s_q_upper, mu_q);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_velocity_barrier_hessian<T, " + NX + ", " + NQ + ", " + NQ + ">(s_Qk, s_x, s_qd_lower, s_qd_upper, mu_qd);")
    self.gen_add_sync()
    self.gen_add_code_line("// ---- input-block hessian s_Rk (NU*NU) ----")
    self.gen_add_code_line("quadratic_input_cost_hessian<T, ACCUMULATE>(s_Rk, s_R);")
    self.gen_add_sync()
    self.gen_add_code_line("joint_torque_barrier_hessian<T, " + NU + ", 0, 0>(s_Rk, s_u, s_u_lower, s_u_upper, mu_u);")
    self.gen_add_end_function()

    # ---- fc-aware overloads (GATO ASK 1, 2026-08-09): emitted ONLY when contact
    # frames are baked, so FC_SIZE==0 builds stay preprocessor/bitwise identical.
    # Forces-as-controls: the control vector widens to CONTROL_SIZE = NU+FC with
    # the contact wrenches [n; f] per baked frame in the TAIL of s_u. The fc term
    # is 0.5*fc_cost*|fc - fc_ref|^2 (s_fc_ref == nullptr -> zero reference, the
    # pure regularization, bitwise). Terminal knots: pass fc_cost = 0 (the same
    # weight-selection contract as every other preset term).
    n_contacts = getattr(self, "_contact_frames_n", 0)
    if n_contacts:
        FC = str(6 * n_contacts)
        NUFC = str(nu + 6 * n_contacts)
        self.gen_add_code_line("#define GRID_PLANT_HAS_TRACKING_COST_FC 1")
        self.gen_add_code_line("const int GRID_PLANT_CONTROL_SIZE = " + NUFC + ";  // NU + 6*NUM_CONTACT_FRAMES")

        self.gen_add_func_doc(
            "tracking_cost_fc: tracking_cost + the forces-as-controls regularization "
            "0.5*fc_cost*|fc - fc_ref|^2 over the fc tail of a CONTROL_SIZE-wide s_u",
            ["s_u is CONTROL_SIZE = NU+" + FC + " wide: [actuated(NU); fc(" + FC + ")], wrench order [n; f] per baked frame.",
             "s_fc_ref (" + FC + ") may be nullptr -> zero reference (pure regularization, bitwise).",
             "Terminal knot: pass fc_cost = 0 (weight-selection contract; matches the u-reg drop).",
             "All other parameters exactly as tracking_cost."],
            [], None)
        self.gen_add_code_line("template <typename T, int EE = 0, bool ACCUMULATE = false>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line(
            "void tracking_cost_fc(T *s_out, const T *s_x, const T *s_u, "
            "const T *s_x_des, const T *s_u_des, const T *s_ee_des, "
            "const T *s_Q, const T *s_R, const T *s_W, "
            "const T *s_q_lower, const T *s_q_upper, const T mu_q, "
            "const T *s_qd_lower, const T *s_qd_upper, const T mu_qd, "
            "const T *s_u_lower, const T *s_u_upper, const T mu_u, "
            "const T fc_cost, const T *s_fc_ref, "
            "T *s_end_effector_pose, T *s_scratch, const grid::robotModel<T> *d_robotModel) {", True)
        self.gen_add_code_line("tracking_cost<T, EE, ACCUMULATE>(s_out, s_x, s_u, s_x_des, s_u_des, s_ee_des, "
                               "s_Q, s_R, s_W, s_q_lower, s_q_upper, mu_q, s_qd_lower, s_qd_upper, mu_qd, "
                               "s_u_lower, s_u_upper, mu_u, s_end_effector_pose, s_scratch, d_robotModel);")
        self.gen_add_sync()
        self.gen_add_code_line("if (threadIdx.x == 0 && threadIdx.y == 0) {", True)
        self.gen_add_code_line("T _fc_acc = static_cast<T>(0);")
        self.gen_add_code_line("for (int j = 0; j < " + FC + "; ++j) {", True)
        self.gen_add_code_line("T _e = s_u[" + NU + " + j] - (s_fc_ref ? s_fc_ref[j] : static_cast<T>(0));")
        self.gen_add_code_line("_fc_acc += _e * _e;")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("s_out[0] += static_cast<T>(0.5) * fc_cost * _fc_acc;")
        self.gen_add_end_control_flow()
        self.gen_add_end_function()

        self.gen_add_func_doc(
            "tracking_cost_gradient_fc: tracking_cost_gradient with a CONTROL_SIZE-wide input block",
            ["s_rk is CONTROL_SIZE = NU+" + FC + " wide: actuated rows exactly as tracking_cost_gradient, "
             "fc rows = fc_cost*(fc - fc_ref).",
             "ACCUMULATE=false overwrites the fc rows; true adds. s_fc_ref nullptr -> zero reference."],
            [], None)
        self.gen_add_code_line("template <typename T, int EE = 0, bool ACCUMULATE = false>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line(
            "void tracking_cost_gradient_fc(T *s_qk, T *s_rk, const T *s_x, const T *s_u, "
            "const T *s_x_des, const T *s_u_des, const T *s_ee_des, "
            "const T *s_Q, const T *s_R, const T *s_W, "
            "const T *s_q_lower, const T *s_q_upper, const T mu_q, "
            "const T *s_qd_lower, const T *s_qd_upper, const T mu_qd, "
            "const T *s_u_lower, const T *s_u_upper, const T mu_u, "
            "const T fc_cost, const T *s_fc_ref, "
            "T *s_end_effector_pose, T *s_end_effector_pose_gradient, T *s_scratch, "
            "const grid::robotModel<T> *d_robotModel) {", True)
        self.gen_add_code_line("tracking_cost_gradient<T, EE, ACCUMULATE>(s_qk, s_rk, s_x, s_u, s_x_des, s_u_des, s_ee_des, "
                               "s_Q, s_R, s_W, s_q_lower, s_q_upper, mu_q, s_qd_lower, s_qd_upper, mu_qd, "
                               "s_u_lower, s_u_upper, mu_u, s_end_effector_pose, s_end_effector_pose_gradient, s_scratch, d_robotModel);")
        self.gen_add_sync()
        self.gen_add_code_line("for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < " + FC + "; ind += blockDim.x*blockDim.y) {", True)
        self.gen_add_code_line("T _g = fc_cost * (s_u[" + NU + " + ind] - (s_fc_ref ? s_fc_ref[ind] : static_cast<T>(0)));")
        self.gen_add_code_line("if (ACCUMULATE) { s_rk[" + NU + " + ind] += _g; } else { s_rk[" + NU + " + ind] = _g; }")
        self.gen_add_end_control_flow()
        self.gen_add_end_function()

        self.gen_add_func_doc(
            "tracking_cost_hessian_fc: tracking_cost_hessian with a CONTROL_SIZE-wide input block",
            ["s_Rk is CONTROL_SIZE*CONTROL_SIZE (" + NUFC + "x" + NUFC + ", col-major): actuated NUxNU block "
             "exactly as tracking_cost_hessian, fc diagonal = fc_cost, all cross terms 0.",
             "s_R_nu_scratch must hold NU*NU (" + NU + "*" + NU + "): the base input-hessian is composed there "
             "then scattered to the widened strides (value/grad/hess single-source contract).",
             "ACCUMULATE=false overwrites s_Rk; true adds into it."],
            [], None)
        self.gen_add_code_line("template <typename T, int EE = 0, bool ACCUMULATE = false>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line(
            "void tracking_cost_hessian_fc(T *s_Qk, T *s_Rk, const T *s_x, const T *s_u, "
            "const T *s_Q, const T *s_R, const T *s_W, "
            "const T *s_q_lower, const T *s_q_upper, const T mu_q, "
            "const T *s_qd_lower, const T *s_qd_upper, const T mu_qd, "
            "const T *s_u_lower, const T *s_u_upper, const T mu_u, "
            "const T fc_cost, "
            "T *s_R_nu_scratch, T *s_end_effector_pose_gradient, T *s_scratch, "
            "const grid::robotModel<T> *d_robotModel) {", True)
        self.gen_add_code_line("// base composite: state block direct (honors ACCUMULATE), input block")
        self.gen_add_code_line("// into the NU-wide scratch — which must compose FRESH regardless (the")
        self.gen_add_code_line("// caller's ACCUMULATE is applied by the scatter below, not by the scratch).")
        self.gen_add_code_line("if (ACCUMULATE) {", True)
        self.gen_add_code_line("for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < " + NU + "*" + NU + "; ind += blockDim.x*blockDim.y) { s_R_nu_scratch[ind] = static_cast<T>(0); }")
        self.gen_add_code_line("__syncthreads();")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("tracking_cost_hessian<T, EE, ACCUMULATE>(s_Qk, s_R_nu_scratch, s_x, s_u, "
                               "s_Q, s_R, s_W, s_q_lower, s_q_upper, mu_q, s_qd_lower, s_qd_upper, mu_qd, "
                               "s_u_lower, s_u_upper, mu_u, s_end_effector_pose_gradient, s_scratch, d_robotModel);")
        self.gen_add_sync()
        self.gen_add_code_line("// scatter to CONTROL_SIZE strides: actuated block verbatim, fc diag = fc_cost, cross 0")
        self.gen_add_code_line("for (int ind = threadIdx.x + threadIdx.y*blockDim.x; ind < " + NUFC + "*" + NUFC + "; ind += blockDim.x*blockDim.y) {", True)
        self.gen_add_code_line("int _r = ind % " + NUFC + "; int _c = ind / " + NUFC + ";")
        self.gen_add_code_line("T _v;")
        self.gen_add_code_line("if (_r < " + NU + " && _c < " + NU + ") { _v = s_R_nu_scratch[_r + " + NU + "*_c]; }")
        self.gen_add_code_line("else if (_r == _c) { _v = fc_cost; }")
        self.gen_add_code_line("else { _v = static_cast<T>(0); }")
        self.gen_add_code_line("if (ACCUMULATE) { s_Rk[ind] += _v; } else { s_Rk[ind] = _v; }")
        self.gen_add_end_control_flow()
        self.gen_add_end_function()


# ---------------------------------------------------------------------------
# Log-barriers (joint position / velocity / torque). Explicit bound pointers.
# ---------------------------------------------------------------------------

def _gen_barrier_helpers(self):
    """Emit the scalar one-sided-safe log-barrier helpers (value/grad/hess).

    Each `isfinite`-guards both sides so a +/-inf bound contributes zero. A tiny
    margin floor keeps the log/reciprocal finite at (and just outside) the
    boundary without changing the interior value materially.
    """
    self.gen_add_func_doc(
        "Scalar log-barrier helpers: b = -mu*(log(x-lo)+log(hi-x)); isfinite-guarded so an inf bound contributes zero",
        ["Margin floored at 1e-10 (value) / 1e-6 (grad,hess) to stay finite at the boundary."])
    self.gen_add_code_lines([
        "template <typename T> __device__ inline T grid_plant_log_barrier(T x, T lo, T hi, T mu) {",
        "    T b = static_cast<T>(0);",
        "    if (isfinite(lo)) { T d = x - lo; d = (d <= static_cast<T>(1e-10)) ? static_cast<T>(1e-10) : d; b -= log(d); }",
        "    if (isfinite(hi)) { T d = hi - x; d = (d <= static_cast<T>(1e-10)) ? static_cast<T>(1e-10) : d; b -= log(d); }",
        "    return mu * b;",
        "}",
        "",
        "template <typename T> __device__ inline T grid_plant_log_barrier_grad(T x, T lo, T hi, T mu) {",
        "    T g = static_cast<T>(0);",
        "    const T eps = static_cast<T>(1e-6);",
        "    if (isfinite(lo)) { T d = x - lo; T a = (d < static_cast<T>(0)) ? -d : d; if (a < eps) a = eps; d = (d < static_cast<T>(0)) ? -a : a; g -= static_cast<T>(1) / d; }",
        "    if (isfinite(hi)) { T d = hi - x; T a = (d < static_cast<T>(0)) ? -d : d; if (a < eps) a = eps; d = (d < static_cast<T>(0)) ? -a : a; g += static_cast<T>(1) / d; }",
        "    return mu * g;",
        "}",
        "",
        "template <typename T> __device__ inline T grid_plant_log_barrier_hess(T x, T lo, T hi, T mu) {",
        "    T h = static_cast<T>(0);",
        "    const T eps = static_cast<T>(1e-6);",
        "    if (isfinite(lo)) { T d = x - lo; T a = (d < static_cast<T>(0)) ? -d : d; if (a < eps) a = eps; h += static_cast<T>(1) / (a * a); }",
        "    if (isfinite(hi)) { T d = hi - x; T a = (d < static_cast<T>(0)) ? -d : d; if (a < eps) a = eps; h += static_cast<T>(1) / (a * a); }",
        "    return mu * h;",
        "}",
        "",
    ])


def _gen_one_barrier(self, base, doc_target, count, slice_offset):
    """Emit `<base>` (value), `<base>_gradient`, `<base>_hessian` for `count`
    bounded DOFs reading `s_var[slice_offset + i]`, writing grad/hess into the
    matching slice of a packed buffer (so torque/velocity barriers add into the
    right rows of a [x;u] gradient/hessian). All barriers ADD (+=) into outputs.

    `slice_offset` is the row/col offset of this barrier's DOFs inside the
    packed gradient/hessian (0 for a standalone buffer; NUM_POS for the qd block
    of an x-gradient; etc.). The hessian is written into a dense column-major
    `stride x stride` block — caller passes the block leading dimension.
    """
    N = str(count)
    self.gen_add_func_doc(
        base + ": value += -mu * sum_i ( log(x_i - lo_i) + log(hi_i - x_i) ) over " + doc_target,
        ["Block-cooperative reduction into s_scratch then a serial add into s_out[0].",
         "s_lower / s_upper are explicit bound vectors (size " + N + "); an inf entry skips that side (isfinite guard).",
         "s_scratch must hold at least " + N + " elements."],
        ["s_out is the scalar barrier cost (added into s_out[0])",
         "s_var is the variable vector (this barrier reads s_var[" + str(slice_offset) + " + i])",
         "s_lower / s_upper are the per-DOF bounds (size " + N + ")",
         "mu is the barrier weight",
         "s_scratch is shared scratch of size >= " + N],
        None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void " + base + "(T *s_out, const T *s_var, const T *s_lower, const T *s_upper, "
                           "const T mu, T *s_scratch) {", True)
    self.gen_add_parallel_loop("i", N)
    self.gen_add_code_line("s_scratch[i] = grid_plant_log_barrier<T>(s_var[" + str(slice_offset) + " + i], s_lower[i], s_upper[i], mu);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("T acc = static_cast<T>(0);")
    self.gen_add_code_line("for (int i = 0; i < " + N + "; ++i) acc += s_scratch[i];")
    self.gen_add_code_line("s_out[0] += acc;")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    # gradient (always adds into the right slice)
    self.gen_add_func_doc(
        base + "_gradient: s_grad[GRAD_OFFSET + i] += -mu*(1/(x_i-lo_i) - 1/(hi_i-x_i))",
        ["Adds into the packed gradient at GRAD_OFFSET (a template arg so velocity/torque barriers land in the qd / u rows).",
         "isfinite-guarded per side; an unbounded DOF adds exactly zero."],
        ["s_grad is the packed gradient output (added into)",
         "s_var / s_lower / s_upper / mu as in the value function"],
        None)
    self.gen_add_code_line("template <typename T, int VAR_OFFSET = " + str(slice_offset) + ", int GRAD_OFFSET = 0>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void " + base + "_gradient(T *s_grad, const T *s_var, const T *s_lower, const T *s_upper, const T mu) {", True)
    self.gen_add_parallel_loop("i", N)
    self.gen_add_code_line("s_grad[GRAD_OFFSET + i] += grid_plant_log_barrier_grad<T>(s_var[VAR_OFFSET + i], s_lower[i], s_upper[i], mu);")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

    # hessian (adds onto the diagonal of a dense column-major block)
    self.gen_add_func_doc(
        base + "_hessian: s_hess[(HESS_OFFSET+i)*HESS_STRIDE + (HESS_OFFSET+i)] += mu*(1/(x_i-lo_i)^2 + 1/(hi_i-x_i)^2)",
        ["Adds the barrier curvature onto the DIAGONAL of a dense column-major HESS_STRIDE x HESS_STRIDE block.",
         "HESS_OFFSET places it in the qd / u block; HESS_STRIDE is the block leading dimension.",
         "isfinite-guarded per side; an unbounded DOF adds exactly zero."],
        ["s_hess is the dense column-major hessian (added into)",
         "s_var / s_lower / s_upper / mu as above"],
        None)
    self.gen_add_code_line("template <typename T, int HESS_STRIDE, int VAR_OFFSET = " + str(slice_offset) + ", int HESS_OFFSET = 0>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void " + base + "_hessian(T *s_hess, const T *s_var, const T *s_lower, const T *s_upper, const T mu) {", True)
    self.gen_add_parallel_loop("i", N)
    self.gen_add_code_line("int d = HESS_OFFSET + i;")
    self.gen_add_code_line("s_hess[d * HESS_STRIDE + d] += grid_plant_log_barrier_hess<T>(s_var[VAR_OFFSET + i], s_lower[i], s_upper[i], mu);")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


def gen_plant_barriers(self):
    """Emit the three log-barriers. URDF parses only position limits today, so
    velocity/torque barriers read caller-supplied explicit bound pointers — no
    URDFParser change. Default VAR_OFFSETs place each on the right block of the
    packed state x = [q; qd]:
      joint_position_barrier reads x[0..NUM_POS)           (q block)
      joint_velocity_barrier reads x[NUM_POS..NUM_POS+NUM_VEL)  (qd block)
      joint_torque_barrier   reads u[0..NUM_VEL)           (standalone u buffer)
    """
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    _gen_barrier_helpers(self)
    _gen_one_barrier(self, "joint_position_barrier", "the NUM_POS q-DOFs", nq, 0)
    _gen_one_barrier(self, "joint_velocity_barrier", "the NUM_VEL qd-DOFs", nv, nq)
    _gen_one_barrier(self, "joint_torque_barrier",   "the NUM_VEL u-DOFs",  nv, 0)


# ---------------------------------------------------------------------------
# Kernels + host wrappers (the binding layer: G1 grid_plant Python surface).
#
# Each plant kernel runs ONE BLOCK PER TIMESTEP (block-level grid-stride loop)
# and calls the auto-allocating `grid_plant::`/`grid::` device functions, which
# own the WHOLE `extern __shared__` arena. The kernel therefore passes GLOBAL
# device pointers for state / input / output straight through to the device
# function — the device fn reads/writes through whatever pointer it is given,
# so the heavy RBD scratch stays internal and we never collide with it.
#
# Reduction-style cost / barrier device functions additionally take an
# `s_scratch` shared buffer; the kernel declares a tiny `__shared__` array for
# it (these kernels do no RBD arena work, so a static shared array is fine).
#
# The host wrappers take raw device pointers for the plant-specific in/out
# buffers (desired states, weights, bounds, scalar outputs). The grid_rbd C ABI
# (wrapper_template.cu) allocates those device buffers and stages H<->D copies.
# ---------------------------------------------------------------------------

def gen_plant_step_kernel(self):
    """`plant_step_kernel` — one block/timestep, calls grid_plant::plant_step.

    Inputs/outputs are global; the device fn (-> grid::integrator_device) owns
    the shared arena. Reuses grid::INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES.

    mjx (MuJoCo output-convention, floating-base only): MUJOCO_OUTPUT is threaded
    LAST. On the pin path (default) the kernel calls plant_step straight on the
    global pointers (byte-identical to before). On the mjx path the per-timestep x
    is staged into a small static-smem buffer so the q-only INPUT convert (base quat
    wxyz->xyzw) can mutate it in place before the device call; plant_step then does
    the pin integrate + the mjx RETRACT epilogue (base-position global add + output
    quat xyzw->wxyz). The integrator_device owns the (separate, dynamic) RBD arena.
    """
    nq = self.robot.get_num_pos()
    nx = self.robot.get_num_pos() + self.robot.get_num_vel()
    nv = self.robot.get_num_vel()
    mjx_kernel = self.robot.floating_base
    self.gen_add_func_doc("plant_step kernel: x_{k+1} = integrator(x_k, u_k, dt) per timestep",
                          [],
                          ["d_x_kp1 is the next-state output (NUM_POS+NUM_VEL per timestep)",
                           "d_x is the packed current state [q; qd] (NUM_POS+NUM_VEL per timestep)",
                           "d_u is the packed control torque (NUM_VEL per timestep)",
                           "stride_x / stride_u are the per-timestep strides",
                           "d_robotModel / gravity / dt as for plant_step",
                           "NUM_TIMESTEPS is the batch size"], None)
    # MUJOCO_OUTPUT (floating only): LAST template param so existing <T, IT> call
    # sites are unaffected; default false -> byte-identical pin codegen.
    self.gen_add_code_line("template <typename T, grid::IntegratorType IT = grid::IntegratorType::EULER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("void plant_step_kernel(T *d_x_kp1, const T *d_x, const T *d_u, "
                           "const int stride_x, const int stride_u, "
                           "const grid::robotModel<T> *d_robotModel, const T gravity, const T dt, const int NUM_TIMESTEPS) {", True)
    if mjx_kernel:
        # Small static-smem state buffer for the in-place q-only input convert (the
        # integrator_device owns the separate dynamic RBD arena). On the pin path we
        # bypass staging and call straight on global, keeping that path byte-identical.
        self.gen_add_code_line("__shared__ T s_x_mjx[" + str(nx) + "];")
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        # stage x into mutable smem, q-only reorder (RETRACT value family: qd stays
        # RAW mjx for the global-add retract), then plant_step (pin integrate + mjx
        # retract). u stays global (value path doesn't convert u). gen_mjx_quat_reorder
        # is inlined here (the surrounding if-constexpr already gates MUJOCO_OUTPUT).
        self.gen_add_parallel_loop("ind", str(nx))
        self.gen_add_code_line("s_x_mjx[ind] = d_x[k*stride_x + ind];")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_mjx_quat_reorder("s_x_mjx")
        self.gen_add_code_line("plant_step<T, IT, true>(&d_x_kp1[k*" + str(nx) + "], s_x_mjx, &d_u[k*stride_u], d_robotModel, gravity, dt);")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("else {", True)
        self.gen_add_code_line("plant_step<T, IT, false>(&d_x_kp1[k*" + str(nx) + "], &d_x[k*stride_x], &d_u[k*stride_u], d_robotModel, gravity, dt);")
        self.gen_add_end_control_flow()
        self.gen_add_sync()
        self.gen_add_end_control_flow()
    else:
        self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
        self.gen_add_code_line("plant_step<T, IT>(&d_x_kp1[k*" + str(nx) + "], &d_x[k*stride_x], &d_u[k*stride_u], d_robotModel, gravity, dt);")
        self.gen_add_sync()
        self.gen_add_end_control_flow()
    self.gen_add_end_function()


def _plant_step_gradient_extra_t_buffers(self):
    """The plant_step_gradient kernel's shared-arena T-slot list (PERF/full-smem
    shape; every band in smem). Single source of truth shared by
    gen_plant_step_gradient_kernel and the plant_step_gradient_arena carve
    struct so the two cannot drift."""
    n = self.robot.get_num_vel()
    fb = 1 if self.robot.floating_base else 0
    nx = self.robot.get_num_pos() + self.robot.get_num_vel()
    from ._integrator import _max_stages_in_use
    max_stages = _max_stages_in_use()
    d_qdd_count = max_stages * n * 3 * n
    vaf_cnt = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else n)
    return [
        ("s_x", nx),
        ("s_u", n),
        ("s_dAB", 2 * n * 3 * n),
        ("s_df_du", n * 2 * n),
        ("s_dc_du", n * 2 * n),
        ("s_vaf", vaf_cnt),
        ("s_Minv", n * n),
        ("s_qdd", n),
        ("s_q_orig", n + fb),
        ("s_qd_orig", n),
        ("s_stage_grad_qdd", max_stages * n),
        ("s_D_qdd_stage", d_qdd_count),
        ("s_dInt_q_6x6", 36),
        ("s_dInt_v_6x6", 36),
    ]


def gen_plant_step_gradient_arena_carve_struct(self):
    """Emit the namespace-scope `plant_step_gradient_arena<T>` carve struct
    (GATO ASK6): the full-smem sub-buffer layout plant_step_gradient_kernel
    carves — the struct's members ARE plant_step_gradient's caller-placed
    buffer arguments, so an external kernel can allocate
    grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T, TIER_SHARED>() bytes (the
    same reservation the kernel launches with), carve(), and forward. No exact
    t-count tie exists for this surface (it reserves the du sizer, which may be
    a spilled rung on big robots while this layout is always full-smem), so the
    carve assert is the <= allocation-safety form only."""
    layout = self._resolve_arena_layout(
        _plant_step_gradient_extra_t_buffers(self),
        self.gen_integrator_gradient_inner_temp_mem_size(),
        include_topology_helpers = (not self.robot.is_serial_chain()
                                    or not self.robot.are_Ss_identical(list(range(self.robot.get_num_pos())))),
        ximat_size = self.gen_get_XI_size(False, False),
        include_linalg_scratch = True,
        linalg_scratch_bytes = "grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()",
        apply_runtime_transform_band = getattr(self, "runtime_transform", False))
    self.gen_arena_carve_struct(
        "plant_step_gradient_arena", layout,
        "grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T, grid::TIER_SHARED>()",
        expected_t_count = None,
        helper_ns = "grid::",  # emitted in namespace grid_plant; the arena helpers live in grid::
        doc = "plant_step_gradient_arena: carve struct mirroring plant_step_gradient_kernel's "
              "full-smem scratch layout; members map 1:1 onto plant_step_gradient's caller-placed "
              "buffer arguments. Allocate INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T, TIER_SHARED>() "
              "bytes and call carve(base).")


def gen_plant_step_gradient_kernel(self):
    """`plant_step_gradient_kernel` — one block/timestep [A|B] = s_dAB.

    plant_step_gradient is an inner-owns-placement orchestrator (it forwards to
    grid::integrator_gradient_device, which OWNS its FD-grad s_temp pool
    placement). The kernel therefore sets up the WHOLE scratch arena in shared
    memory itself — mirroring grid::integrator_gradient_kernel's PERF/full-smem
    body (every band in smem: SCRATCH_IN_SMEM=true, no workspace/spill) — then
    calls grid_plant::plant_step_gradient (the thin pass-through). The emitted
    s_dAB is byte-identical to grid::integrator_gradient's. Inputs/outputs are
    global; the launch reserves grid::INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES.

    x = [q (NUM_POS); qd (NUM_VEL)] (NON-const: the multi-stage RK path mutates
    q/qd across stages in smem); u = control torque (NUM_VEL)."""
    n = self.robot.get_num_vel()
    fb = 1 if self.robot.floating_base else 0
    nx = self.robot.get_num_pos() + self.robot.get_num_vel()
    from ._integrator import _max_stages_in_use
    max_stages = _max_stages_in_use()
    d_qdd_count = max_stages * n * 3 * n
    inner_temp_full = self.gen_integrator_gradient_inner_temp_mem_size()
    # s_vaf is body-indexed (NB bodies, stride 6); size by NB for mimic robots
    # (NB>nv) so the composed FD-grad ID sub-inner's 18*NB writes don't overflow
    # the adjacent buffers (mirrors integrator_gradient_kernel's vaf sizing).
    vaf_cnt = 18 * (self.robot.get_num_joints() if self.robot_has_mimic_joints() else n)
    self.gen_add_func_doc("plant_step_gradient kernel: [A|B] = integrator_gradient([q;qd], u, dt) per timestep "
                          "(full-smem scratch arena; pass-through to grid::integrator_gradient_device)",
                          [],
                          ["d_dAB is the [A|B] output (2*NUM_VEL*3*NUM_VEL per timestep, column-major)",
                           "d_x is the packed current state [q; qd] (NUM_POS+NUM_VEL per timestep)",
                           "d_u is the packed control torque (NUM_VEL per timestep)",
                           "stride_x / stride_u are the per-timestep strides",
                           "d_robotModel / gravity / dt as for plant_step",
                           "NUM_TIMESTEPS is the batch size"], None)
    # MUJOCO_OUTPUT (floating only): LAST template param so existing <T, IT> call
    # sites are unaffected; default false -> byte-identical pin codegen. Threaded to
    # the input convert + forwarded to plant_step_gradient (whose dAB epilogue is the
    # validated grid::integrator_gradient_device mjx transform).
    mjx_kernel = self.robot.floating_base
    self.gen_add_code_line("template <typename T, grid::IntegratorType IT = grid::IntegratorType::EULER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(grid::MAX_PERF_LEVEL_THREADS)")
    self.gen_add_code_line("void plant_step_gradient_kernel(T *d_dAB, const T *d_x, const T *d_u, "
                           "const int stride_x, const int stride_u, "
                           "const grid::robotModel<T> *d_robotModel, const T gravity, const T dt, const int NUM_TIMESTEPS) {", True)
    # The shared-arena machinery (grid_align_up / grid_arena_ptr / the *_BYTES
    # helpers) + IntegratorType / robotModel live in `namespace grid`; this
    # kernel is in the sibling `grid_plant`, so pull them into scope. The only
    # grid_plant symbol the body references (plant_step_gradient) is found by
    # enclosing-namespace lookup.
    self.gen_add_code_line("using namespace grid;")
    # Whole scratch arena in shared memory (PERF/full-smem; mirrors the
    # integrator_gradient_kernel _emit_body(dqdd_in_smem=True, dab_in_smem=True,
    # inner_level=0) layout). s_x (= s_q;s_qd) + s_u are staged from global; the
    # FD-grad bands, the dAB output, and the multi-stage scratch all live here.
    extra_t_buffers = _plant_step_gradient_extra_t_buffers(self)
    self.gen_XImats_helpers_temp_shared_memory_code(
        inner_temp_full, extra_t_buffers=extra_t_buffers, include_linalg_scratch=True)
    self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
    # stage x (q;qd) + u into smem (plant_step_gradient mutates q/qd across stages).
    self.gen_add_parallel_loop("ind", str(nx))
    self.gen_add_code_line("s_x[ind] = d_x[k*stride_x + ind];")
    self.gen_add_end_control_flow()
    self.gen_add_parallel_loop("ind", str(n))
    self.gen_add_code_line("s_u[ind] = d_u[k*stride_u + ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # mjx input convert (GRADIENT family): quat wxyz->xyzw + base-linear velocity +
    # force -> pin frame, in place on the staged s_x/s_u, BEFORE the device call (so
    # the RBD callees + the dAB epilogue see pin quantities). No-op on the pin path.
    if mjx_kernel:
        _gen_plant_step_gradient_mjx_kernel_input(self)
    # All scratch is smem (SCRATCH_IN_SMEM=true, no spill): pass nullptr for the
    # global workspace + spill regions. The orchestrator owns its s_temp pool.
    mjx_flag = ", MUJOCO_OUTPUT" if mjx_kernel else ""
    self.gen_add_code_line("plant_step_gradient<T, IT, true, false" + mjx_flag + ">("
                           "s_dAB, s_x, s_u, s_df_du, s_dc_du, s_vaf, s_Minv, s_qdd, "
                           "s_q_orig, s_qd_orig, s_stage_grad_qdd, s_D_qdd_stage, "
                           "s_dInt_q_6x6, s_dInt_v_6x6, "
                           + self.gen_insert_helpers_function_call()
                           + "s_temp, nullptr, nullptr, d_robotModel, gravity, dt);")
    self.gen_add_sync()
    self.gen_add_serial_ops()
    self.gen_add_code_line("for (int ind = 0; ind < " + str(2 * n * 3 * n) + "; ++ind) d_dAB[k*" + str(2 * n * 3 * n) + " + ind] = s_dAB[ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


# Per-tier spill flags for plant_step_hessian_kernel, ordered least-spill first.
# (spill_d2AB, spill_tensors, spill_fdsva_pool):
#   spill_d2AB     : the 18*nv^3 COLD output band s_d2AB -> the kernel's global
#                    d_workspace hessian section (the device fn just scatters into
#                    the pointer it is handed — global is fine).
#   spill_tensors  : s_df2 + s_idsva_so (4*nv^3 each, fdsva_so outputs) -> global
#                    workspace bands. These are write-once-consumed-once.
#   spill_fdsva_pool: route the WHOLE fdsva_so s_temp pool + 4*nv^3 contraction
#                    scratch to d_workspace (SCRATCH_IN_SMEM=false in the composed
#                    device fn). This is the dominant smem consumer on big robots.
# Tier 0 keeps everything in smem (SHARED/PERF, byte-identical to the pre-spill
# kernel for robots that fit). Tier 1 is the deep all-large-bands-to-global spill
# that lets g1/h1_2 fit the ~99 KB cap (smem then holds only the small base:
# inputs + s_Minv + s_df_du + s_qdd + XI). Surgical-spill rule: the small hot
# df_du/Minv/qdd stay in smem; only the cold/large d2AB + fdsva outputs + pool spill.
_PLANT_HESSIAN_PICK_FLAGS = [
    (False, False, False),   # tier 0: full smem (SHARED/PERF)
    (True,  True,  True),    # tier 1: deep spill (d2AB + fdsva tensors + pool -> global)
]


def _emit_plant_step_hessian_kernel_body_for_flags(self, n, nx, d2ab_count,
                                                   spill_d2AB, spill_tensors, spill_fdsva_pool):
    """Emit the plant_step_hessian_kernel body for one tier's spill flags.

    Mirrors _emit_fdsva_so_kernel_body_for_flags: declare the shared arena (only
    the buffers that stay in smem for this tier), then declare the workspace bands
    for the spilled buffers, then call grid_plant::plant_step_hessian with the
    matching SCRATCH_IN_SMEM/CONTRACT_IN_SMEM flags, then (if d2AB is in smem)
    scatter it to the global output."""
    # SHARED-tier fdsva_so inner pool: max(idsva_so inner, 4*nv^3 contract,
    # fd_grad inline) -- the same sizing GCG.py's _temp_full uses for fdsva_so.
    inner_idsva = self.gen_idsva_so_body_frame_inner_temp_mem_size()
    inner_temp_full = max(inner_idsva,
                          self.gen_fdsva_so_contract_temp_mem_size(),
                          self.gen_fdsva_so_fd_gradient_inline_temp_mem_size())
    if self.robot.floating_base:
        from ._integrator_gradient import FLOATING_HESSIAN_SE3_SCRATCH
        inner_temp_full = max(inner_temp_full, FLOATING_HESSIAN_SE3_SCRATCH)
    # When the pool spills, the smem s_temp slot is unused (size 0); the whole
    # pool (incl. the contraction scratch) routes to d_workspace.
    shared_temp_size = 0 if spill_fdsva_pool else inner_temp_full
    extra_t_buffers = [("s_x", nx), ("s_u", n)]
    if not spill_d2AB:
        extra_t_buffers.append(("s_d2AB", d2ab_count))
    if not spill_tensors:
        extra_t_buffers.append(("s_df2", 4 * n * n * n))
        extra_t_buffers.append(("s_idsva_so", 4 * n * n * n))
    # df_du / Minv / qdd are small + hot — always smem (surgical-spill rule).
    extra_t_buffers.append(("s_Minv", n * n))
    extra_t_buffers.append(("s_df_du", 2 * n * n))
    extra_t_buffers.append(("s_qdd", n))
    self.gen_XImats_helpers_temp_shared_memory_code(
        shared_temp_size, extra_t_buffers=extra_t_buffers, include_linalg_scratch=True)
    # Floating mjx always needs d_workspace (the dedicated d_mjx_ws band), even at the
    # SHARED tier where no other band spills.
    needs_workspace = spill_d2AB or spill_tensors or spill_fdsva_pool or self.robot.floating_base
    if not needs_workspace:
        self.gen_add_code_line("(void)d_workspace;")
    self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
    self.gen_add_parallel_loop("ind", str(nx))
    self.gen_add_code_line("s_x[ind] = d_x[k*stride_x + ind];")
    self.gen_add_end_control_flow()
    self.gen_add_parallel_loop("ind", str(n))
    self.gen_add_code_line("s_u[ind] = d_u[k*stride_u + ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    # mjx input convert (floating only): quat wxyz->xyzw + base-linear velocity + force
    # -> pin frame, in place on the staged s_x/s_u, BEFORE the device call (so the RBD
    # callees + the d2AB epilogue see pin quantities). No-op on the pin path. Reuses
    # the gradient-family helper (same stacked-state convert).
    mjx_kernel = self.robot.floating_base
    if mjx_kernel:
        _gen_plant_step_gradient_mjx_kernel_input(self)
    # Per-timestep workspace bands (carved past the fdsva_so sections so they never
    # collide with the fdsva pool/contraction spill). Layout (per timestep slot):
    #   [0 .. d2AB)        : s_d2AB output band (18*nv^3) when spill_d2AB
    #   [d2AB .. +4nv^3)   : s_df2     when spill_tensors
    #   [.. +4nv^3)        : s_idsva_so when spill_tensors
    #   [.. + pool)        : s_fdsva_temp (the fdsva pool/contraction) when spill_fdsva_pool
    #   [.. + mjx_ws)      : d_mjx_ws (floating mjx epilogue scratch) — ALWAYS carved
    #                        for floating so the SHARED tier (no other spill) still has it
    if needs_workspace:
        self.gen_add_code_line("T *d_ws = reinterpret_cast<T *>(&d_workspace[k*PLANT_HESSIAN_WORKSPACE_BYTES_PER_TIMESTEP<T>()]);")
        self.gen_add_code_line("size_t ws_off = 0;")
    if spill_d2AB:
        self.gen_add_code_line("T *s_d2AB = &d_ws[ws_off]; ws_off += " + str(d2ab_count) + ";")
    if spill_tensors:
        self.gen_add_code_line("T *s_df2 = &d_ws[ws_off]; ws_off += " + str(4 * n * n * n) + ";")
        self.gen_add_code_line("T *s_idsva_so = &d_ws[ws_off]; ws_off += " + str(4 * n * n * n) + ";")
    if spill_fdsva_pool:
        self.gen_add_code_line("T *s_fdsva_pool = &d_ws[ws_off]; ws_off += " + str(inner_temp_full) + ";")
    if mjx_kernel:
        from ._integrator_gradient import floating_hessian_mjx_ws_count
        self.gen_add_code_line("T *d_mjx_ws = &d_ws[ws_off]; ws_off += " + str(floating_hessian_mjx_ws_count(n)) + ";")
    if needs_workspace:
        self.gen_add_code_line("(void)ws_off;")
    # Compose flags: SCRATCH_IN_SMEM=false routes the fdsva pool to d_workspace
    # (the device fn hands it to the body-frame idsva inner + contraction).
    scratch_in_smem = "false" if spill_fdsva_pool else "true"
    contract_in_smem = "false" if spill_fdsva_pool else "true"
    pool_arg = "s_fdsva_pool" if spill_fdsva_pool else "nullptr"
    # Fixed-base OMITS the d_mjx_ws arg (the wrapper has no such param) -> byte-identical.
    mjx_ws_arg = ("d_mjx_ws, " if mjx_kernel else "")
    mjx_flag = ", MUJOCO_OUTPUT" if mjx_kernel else ""
    self.gen_add_code_line("plant_step_hessian<T, IT, " + scratch_in_smem + ", false, " + contract_in_smem + mjx_flag + ">("
                           "s_d2AB, s_x, s_u, s_df2, s_idsva_so, s_Minv, s_df_du, s_qdd, "
                           + self.gen_insert_helpers_function_call()
                           + "s_temp, " + pool_arg + ", nullptr, " + pool_arg + ", " + mjx_ws_arg + "d_robotModel, gravity, dt);")
    self.gen_add_sync()
    # Scatter s_d2AB to the global output. When d2AB is already in workspace (spill)
    # the device fn wrote straight into the global band, but the public d_d2AB output
    # is a separate contiguous array, so copy in BOTH cases (the workspace slot is a
    # scratch slice, not the user's output array).
    src = "s_d2AB"
    self.gen_add_parallel_loop("ind", str(d2ab_count))
    self.gen_add_code_line("d_d2AB[k*" + str(d2ab_count) + " + ind] = " + src + "[ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_control_flow()


def gen_plant_step_hessian_kernel(self):
    """`plant_step_hessian_kernel` — one block/timestep, s_d2AB -> d_d2AB.

    Tier-aware (mirrors fdsva_so_kernel): TIER_SHARED keeps the whole fdsva_so
    scratch arena + the 18*nv^3 s_d2AB output band in shared memory; TIER_LITE /
    TIER_MINIMAL spill the cold/large bands (s_d2AB output, the fdsva_so output
    tensors s_df2/s_idsva_so, and the fdsva_so s_temp pool + contraction scratch)
    to the L2-pinned d_workspace so big fixed-base robots (g1/h1_2, nv>=35) fit
    the ~99 KB dynamic-smem cap. The small hot buffers (s_Minv/s_df_du/s_qdd) stay
    in smem (surgical-spill rule). Calls grid_plant::plant_step_hessian (the thin
    pass-through over grid::integrator_hessian_device) with the matching
    SCRATCH_IN_SMEM / CONTRACT_IN_SMEM flags. Inputs/outputs are global.

    Scope: EULER / SI-EULER, fixed AND floating base (floating routes to the
    SE(3)-retract hessian device fn; only multi-stage RK static_asserts out).
    x = [q (NUM_POS); qd (NUM_VEL)]; u = control torque
    (NUM_VEL). d_d2AB is the (2*NUM_VEL x 3*NUM_VEL x 3*NUM_VEL) row-major output;
    d_workspace is the generated global spill workspace (nullptr at TIER_SHARED)."""
    n = self.robot.get_num_vel()
    nx = self.robot.get_num_pos() + self.robot.get_num_vel()
    nz = 3 * n
    d2ab_count = 2 * n * nz * nz
    xi_size = self.gen_get_XI_size()
    inner_idsva = self.gen_idsva_so_body_frame_inner_temp_mem_size()
    inner_temp_full = max(inner_idsva,
                          self.gen_fdsva_so_contract_temp_mem_size(),
                          self.gen_fdsva_so_fd_gradient_inline_temp_mem_size())
    if self.robot.floating_base:
        from ._integrator_gradient import FLOATING_HESSIAN_SE3_SCRATCH
        inner_temp_full = max(inner_temp_full, FLOATING_HESSIAN_SE3_SCRATCH)
    # Per-timestep workspace band for the spilled tiers: s_d2AB (18*nv^3) +
    # s_df2 + s_idsva_so (8*nv^3) + the fdsva_so pool/contraction scratch. Sized
    # for the MAX a tier might spill, so the allocation always covers any tier the
    # kernel template is instantiated with. Dedicated to the hessian kernel (its
    # own d_workspace arg), so it does NOT perturb the shared gridData
    # GRID_WORKSPACE_BYTES_PER_TIMESTEP used by every other algorithm. Emitted
    # BEFORE the kernel template so the kernel body can reference it.
    ws_t_count = d2ab_count + 8 * n * n * n + inner_temp_full
    # Floating mjx: add the dedicated d_mjx_ws band (pin-d2AB copy + dInt_q + pin dAB)
    # so the per-timestep workspace always covers it (carved AFTER the spill sections).
    if self.robot.floating_base:
        from ._integrator_gradient import floating_hessian_mjx_ws_count
        ws_t_count += floating_hessian_mjx_ws_count(n)
    self.gen_add_code_line(
        "template <typename T> __host__ __device__ inline size_t "
        "PLANT_HESSIAN_WORKSPACE_BYTES_PER_TIMESTEP() { return sizeof(T) * static_cast<size_t>("
        + str(ws_t_count) + "); }")
    self.gen_add_func_doc("plant_step_hessian kernel: s_d2AB = d^2 integrator([q;qd], u, dt) per timestep "
                          "(tier-aware scratch; pass-through to grid::integrator_hessian_device)",
                          [],
                          ["d_d2AB is the Hessian output (2*NUM_VEL*3*NUM_VEL*3*NUM_VEL per timestep, row-major)",
                           "d_workspace is the generated global spill workspace (nullptr at TIER_SHARED)",
                           "d_x is the packed current state [q; qd] (NUM_POS+NUM_VEL per timestep)",
                           "d_u is the packed control torque (NUM_VEL per timestep)",
                           "stride_x / stride_u are the per-timestep strides",
                           "d_robotModel / gravity / dt as for plant_step",
                           "NUM_TIMESTEPS is the batch size"], None)
    # MUJOCO_OUTPUT (floating only): LAST template param so existing
    # <T, IT, RESOURCE_TIER> launches are unaffected; default false -> byte-identical
    # pin codegen. Threaded to the input convert + forwarded to plant_step_hessian
    # (whose d2AB epilogue is the validated grid::integrator_hessian_device mjx transform).
    self.gen_add_code_line("template <typename T, grid::IntegratorType IT = grid::IntegratorType::EULER, "
                           "int RESOURCE_TIER = grid::GRID_DEFAULT_RESOURCE_TIER, bool MUJOCO_OUTPUT = false>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("__launch_bounds__(grid::tier_max_threads<RESOURCE_TIER>())")
    self.gen_add_code_line("void plant_step_hessian_kernel(T *d_d2AB, unsigned char *d_workspace, const T *d_x, const T *d_u, "
                           "const int stride_x, const int stride_u, "
                           "const grid::robotModel<T> *d_robotModel, const T gravity, const T dt, const int NUM_TIMESTEPS) {", True)
    self.gen_add_code_line("using namespace grid;")
    picks = getattr(self, "plant_step_hessian_spill_tier_3way", (0, 0, 0))

    def _emit_body(pick):
        spill_d2AB, spill_tensors, spill_fdsva_pool = _PLANT_HESSIAN_PICK_FLAGS[pick]
        _emit_plant_step_hessian_kernel_body_for_flags(
            self, n, nx, d2ab_count, spill_d2AB, spill_tensors, spill_fdsva_pool)

    self.gen_tier_dispatch(picks, _emit_body)
    self.gen_add_end_function()
    # Per-tier launch shared-mem byte count for plant_step_hessian_kernel. The
    # binding launch MUST size the dynamic smem with this macro AND raise the
    # per-kernel max via cudaFuncSetAttribute. The t_count mirrors each tier's
    # arena exactly (the in-smem extra_t_buffers + s_XImats + s_temp pool).
    base_t = nx + n + n * n + 2 * n * n + n + xi_size

    def _tier_t_count(pick):
        spill_d2AB, spill_tensors, spill_fdsva_pool = _PLANT_HESSIAN_PICK_FLAGS[pick]
        t = base_t
        if not spill_d2AB:
            t += d2ab_count
        if not spill_tensors:
            t += 8 * n * n * n
        if not spill_fdsva_pool:
            t += inner_temp_full
        return t

    t0 = _tier_t_count(picks[0])
    t1 = _tier_t_count(picks[1])
    t2 = _tier_t_count(picks[2])
    self.gen_add_code_line(
        "template <typename T, int TIER = grid::GRID_DEFAULT_RESOURCE_TIER> __host__ __device__ inline size_t "
        "INTEGRATOR_HESSIAN_DYNAMIC_SHARED_MEM_BYTES() { "
        "if constexpr (TIER == grid::TIER_SHARED)    return grid::grid_shared_arena_bytes<T>(" + str(t0) + ", grid::TOPOLOGY_HELPERS_COUNT, grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
        "else if constexpr (TIER == grid::TIER_LITE) return grid::grid_shared_arena_bytes<T>(" + str(t1) + ", grid::TOPOLOGY_HELPERS_COUNT, grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
        "else                                        return grid::grid_shared_arena_bytes<T>(" + str(t2) + ", grid::TOPOLOGY_HELPERS_COUNT, grid::GRID_LINALG_NVIDIA_MAX_HELPER_BYTES<T>()); "
        "}")
    # 1 when any tier spills (so the launcher knows it must allocate d_workspace).
    spills_any = 1 if any(p >= 1 for p in picks) else 0
    self.gen_add_code_line(
        "static const int GRID_PLANT_HESSIAN_USES_WORKSPACE_ANY_TIER = " + str(spills_any) + ";")


def gen_com_cost_kernel(self):
    """`com_cost_kernel` — one block/timestep value+grad_x+GN-hess_x.

    Calls grid_plant::com_cost[_gradient/_hessian] (caller-scratch inners over
    centroidal_inner). The kernel reserves grid::COM_DYNAMIC_SHARED_MEM_BYTES as a
    dynamic-smem arena and passes it as the cost fns' s_scratch. Global in/out.
    Mirrors gen_ee_pos_cost_kernel (CoM position/Jacobian in place of EE)."""
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    nx = nq + nv
    self.gen_add_func_doc("com_cost_kernel: value + grad_x + GN hess_x per timestep",
                          [], ["d_out scalar cost (1 per timestep)",
                               "d_grad grad over x (" + str(nx) + " per timestep)",
                               "d_hess dense col-major x-hessian (" + str(nx*nx) + " per timestep)",
                               "d_q joint positions (NUM_POS per timestep)",
                               "d_p_des desired CoM position (3 per timestep)",
                               "d_W per-axis weight (3 per timestep)",
                               "NUM_TIMESTEPS is the batch size"], None)
    mjx_tmpl = ", bool MUJOCO_OUTPUT = false"
    self.gen_add_code_line("template <typename T" + mjx_tmpl + ">")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("void com_cost_kernel(T *d_out, T *d_grad, T *d_hess, "
                           "const T *d_q, const T *d_p_des, const T *d_W, "
                           "const grid::robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {", True)
    # Dynamic arena for the caller-scratch centroidal cost inners (the launch reserves
    # COM_DYNAMIC_SHARED_MEM_BYTES); the cost fns lay their scratch out from it.
    self.gen_add_code_line("extern __shared__ __align__(16) T s_com_arena[];")
    self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
    self.gen_add_code_line("const T *s_q = &d_q[k*" + str(nq) + "]; const T *s_p_des = &d_p_des[k*3]; const T *s_W = &d_W[k*3];")
    if self.robot.floating_base:
        # mjx INPUT convert (q wxyz->xyzw) so the world-frame CoM/J_com are computed
        # from the correct PIN base orientation; grad/hess epilogues reframe the output.
        self.gen_add_code_line("const T *s_q_use = s_q;")
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        bufs = _gen_cost_mjx_kernel_input(self, nq, convert_qd=False)
        self.gen_add_code_line("s_q_use = " + bufs[0] + ";")
        self.gen_add_end_control_flow()
        qname = "s_q_use"
        gh_tmpl = ", false, MUJOCO_OUTPUT"
    else:
        qname = "s_q"
        gh_tmpl = ""
    self.gen_add_code_line("com_cost<T>(&d_out[k], " + qname + ", s_p_des, s_W, s_com_arena, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("com_cost_gradient<T" + gh_tmpl + ">(&d_grad[k*" + str(nx) + "], " + qname + ", s_p_des, s_W, s_com_arena, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("com_cost_hessian<T" + gh_tmpl + ">(&d_hess[k*" + str(nx*nx) + "], " + qname + ", s_W, s_com_arena, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


def gen_momentum_cost_kernel(self):
    """`momentum_cost_kernel` — one block/timestep value+grad_x+GN-hess_x.

    Calls grid_plant::momentum_cost[_gradient/_hessian] (caller-scratch inners over
    centroidal_inner). The kernel reserves grid::CCRBA_DYNAMIC_SHARED_MEM_BYTES as a
    dynamic-smem arena and passes it as the cost fns' s_scratch. Global in/out.
    Reads q + qd (the momentum h = A qd depends on qd)."""
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    nx = nq + nv
    self.gen_add_func_doc("momentum_cost_kernel: value + grad_x + GN hess_x per timestep",
                          [], ["d_out scalar cost (1 per timestep)",
                               "d_grad grad over x (" + str(nx) + " per timestep)",
                               "d_hess dense col-major x-hessian (" + str(nx*nx) + " per timestep)",
                               "d_q / d_qd joint positions/velocities (NUM_POS / NUM_VEL per timestep)",
                               "d_h_des desired centroidal momentum (6 per timestep)",
                               "d_W per-component weight (6 per timestep)",
                               "NUM_TIMESTEPS is the batch size"], None)
    mjx_tmpl = ", bool MUJOCO_OUTPUT = false"
    self.gen_add_code_line("template <typename T" + mjx_tmpl + ">")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("void momentum_cost_kernel(T *d_out, T *d_grad, T *d_hess, "
                           "const T *d_q, const T *d_qd, const T *d_h_des, const T *d_W, "
                           "const grid::robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {", True)
    # Dynamic arena for the caller-scratch centroidal cost inners (launch reserves
    # CCRBA_DYNAMIC_SHARED_MEM_BYTES); the cost fns lay their scratch out from it.
    self.gen_add_code_line("extern __shared__ __align__(16) T s_ccrba_arena[];")
    self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
    self.gen_add_code_line("const T *s_q = &d_q[k*" + str(nq) + "]; const T *s_qd = &d_qd[k*" + str(nv) + "];")
    self.gen_add_code_line("const T *s_h_des = &d_h_des[k*6]; const T *s_W = &d_W[k*6];")
    if self.robot.floating_base:
        # mjx INPUT convert: q wxyz->xyzw AND qd[0:3]=R^T qd[0:3] (mjx GLOBAL base
        # velocity -> pin LOCAL), so the centroidal momentum h = A qd is computed in
        # the pin frame (the cost VALUE + residual r = h - h_des need the pin h). The
        # grad/hess qd-block epilogues then reframe the OUTPUT (gated MUJOCO_OUTPUT).
        self.gen_add_code_line("const T *s_q_use = s_q; const T *s_qd_use = s_qd;")
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        bufs = _gen_cost_mjx_kernel_input(self, nq, convert_qd=True)
        self.gen_add_code_line("s_q_use = " + bufs[0] + "; s_qd_use = " + bufs[1] + ";")
        self.gen_add_end_control_flow()
        qn, qdn = "s_q_use", "s_qd_use"
        gh_tmpl = ", false, MUJOCO_OUTPUT"
    else:
        qn, qdn = "s_q", "s_qd"
        gh_tmpl = ""
    self.gen_add_code_line("momentum_cost<T>(&d_out[k], " + qn + ", " + qdn + ", s_h_des, s_W, s_ccrba_arena, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("momentum_cost_gradient<T" + gh_tmpl + ">(&d_grad[k*" + str(nx) + "], " + qn + ", " + qdn + ", s_h_des, s_W, s_ccrba_arena, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("momentum_cost_hessian<T" + gh_tmpl + ">(&d_hess[k*" + str(nx*nx) + "], " + qn + ", " + qdn + ", s_W, s_ccrba_arena, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


def gen_quadratic_cost_kernel(self, which):
    """`<base>_kernel` — one block/timestep value+grad+GN-diag-hess.

    No RBD arena needed; a small static `__shared__` reduction scratch suffices.
    """
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    if which == "state":
        size = nq + nv
        var, des, w = "x", "x_des", "Q"
        base = "quadratic_state_cost"
    else:
        size = nv
        var, des, w = "u", "u_des", "R"
        base = "quadratic_input_cost"
    N = str(size)
    self.gen_add_func_doc(base + "_kernel: value + gradient + GN-diag hessian per timestep",
                          [], ["d_out scalar cost (1 per timestep)",
                               "d_grad gradient (" + N + " per timestep)",
                               "d_hess dense col-major hessian (" + N + "*" + N + " per timestep)",
                               "d_" + var + " / d_" + des + " / d_" + w + " inputs (" + N + " per timestep)",
                               "NUM_TIMESTEPS is the batch size"], None)
    mjx = (which == "state" and self.robot.floating_base)
    mjx_tmpl = ", bool MUJOCO_OUTPUT = false"
    self.gen_add_code_line("template <typename T" + mjx_tmpl + ">")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("void " + base + "_kernel(T *d_out, T *d_grad, T *d_hess, "
                           "const T *d_" + var + ", const T *d_" + des + ", const T *d_" + w + ", const int NUM_TIMESTEPS) {", True)
    self.gen_add_code_line("__shared__ T s_scratch[" + N + "];")
    self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
    if mjx:
        # mjx INPUT convert: x's velocity block -> pin frame (value is convention-
        # dependent), plus an xyzw config for the grad/hess output epilogues. The
        # q-block of x_use stays mjx (differenced vs the user's mjx x_des).
        self.gen_add_code_line("const T *s_x = &d_x[k*" + N + "];")
        self.gen_add_code_line("const T *s_x_arg = s_x; const T *s_q_arg = s_x;")
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        bufs = _gen_state_cost_mjx_kernel_input(self, nq, nv)
        self.gen_add_code_line("s_x_arg = " + bufs[0] + "; s_q_arg = " + bufs[1] + ";")
        self.gen_add_end_control_flow()
        vgh_tmpl = ", false, MUJOCO_OUTPUT"
        xarg, qarg = "s_x_arg", ", s_q_arg"
    else:
        vgh_tmpl = ""
        xarg, qarg = "&d_" + var + "[k*" + N + "]", ""
    self.gen_add_code_line(base + "_value_grad_hess<T" + vgh_tmpl + ">(&d_out[k], &d_grad[k*" + N + "], &d_hess[k*" + str(size*size) + "], "
                           + xarg + ", &d_" + des + "[k*" + N + "], &d_" + w + "[k*" + N + "], s_scratch" + qarg + ");")
    self.gen_add_sync()
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


def gen_ee_pos_cost_kernel(self):
    """`ee_pos_cost_kernel` — one block/timestep value+grad_x+GN-hess_x.

    Calls grid_plant::ee_pos_cost[_gradient/_hessian], which are caller-scratch
    INNERS (they bottom out at grid::end_effector_pose_inner /
    end_effector_pose_gradient_inner). The kernel grabs the dynamic
    `extern __shared__` arena and threads it as s_scratch into all three (reused
    serially with a __syncthreads between). Global in/out; the launch reserves
    grid::END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES (the gradient arena
    dominates the value arena).
    """
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    # GATO Ask-4: MUST mirror gen_ee_pos_cost's num_ees -- this kernel slices its global
    # d_end_effector_pose[_gradient] buffers at 6*num_ees / 6*nv*num_ees per timestep, and the
    # cost fns it calls now write 6*NUM_EE for the RESOLVED target (1 when a named target is
    # baked). A stale all-leaf num_ees here would stride past what was written.
    _tgt = getattr(self, "_ee_target_name", "")
    num_ees = 1 if _tgt else self.robot.get_total_leaf_nodes()
    nx = nq + nv
    self.gen_add_func_doc("ee_pos_cost_kernel: value + grad_x + GN hess_x per timestep (EE=0)",
                          [], ["d_out scalar cost (1 per timestep)",
                               "d_grad grad over x (" + str(nx) + " per timestep)",
                               "d_hess dense col-major x-hessian (" + str(nx*nx) + " per timestep)",
                               "d_q joint positions (NUM_POS per timestep)",
                               "d_p_des desired EE position (3 per timestep)",
                               "d_W per-axis weight (3 per timestep)",
                               "d_end_effector_pose / d_end_effector_pose_gradient global scratch (6*NUM_EES / 6*NUM_VEL*NUM_EES per timestep)",
                               "NUM_TIMESTEPS is the batch size"], None)
    mjx_tmpl = ", bool MUJOCO_OUTPUT = false"
    self.gen_add_code_line("template <typename T, int EE = 0" + mjx_tmpl + ">")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("void ee_pos_cost_kernel(T *d_out, T *d_grad, T *d_hess, "
                           "const T *d_q, const T *d_p_des, const T *d_W, T *d_end_effector_pose, T *d_end_effector_pose_gradient, "
                           "const grid::robotModel<T> *d_robotModel, const int NUM_TIMESTEPS) {", True)
    # Dynamic arena threaded as s_scratch into the caller-scratch cost inners (reused
    # serially across the value/gradient/hessian calls; the launch reserves the
    # gradient arena, which dominates). 16B aligned for the homogeneous-transform loads.
    self.gen_add_code_line("extern __shared__ __align__(16) T s_ee_arena[];")
    self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
    self.gen_add_code_line("const T *s_q = &d_q[k*" + str(nq) + "]; const T *s_p_des = &d_p_des[k*3]; const T *s_W = &d_W[k*3];")
    self.gen_add_code_line("T *s_end_effector_pose = &d_end_effector_pose[k*" + str(6*num_ees) + "]; T *s_end_effector_pose_gradient = &d_end_effector_pose_gradient[k*" + str(6*nv*num_ees) + "];")
    if self.robot.floating_base:
        # mjx INPUT convert (q wxyz->xyzw) into a per-block buffer, so the world-frame
        # EE pose/Jacobian are computed from the correct PIN-frame base orientation;
        # the grad/hess device epilogues then reframe the OUTPUT (gated MUJOCO_OUTPUT).
        self.gen_add_code_line("const T *s_q_use = s_q;")
        self.gen_add_code_line("if constexpr (MUJOCO_OUTPUT) {", True)
        bufs = _gen_cost_mjx_kernel_input(self, nq, convert_qd=False)
        self.gen_add_code_line("s_q_use = " + bufs[0] + ";")
        self.gen_add_end_control_flow()
        qname = "s_q_use"
        gh_tmpl = ", EE, false, MUJOCO_OUTPUT"
        # hessian template order is <T, EE, ACCUMULATE, GAUSS_NEWTON, MUJOCO_OUTPUT>.
        # GAUSS_NEWTON=true pinned: this kernel's launch reserves only the GRADIENT
        # arena and has no d2ee global scratch — the Newton variant through the C-ABI
        # is a follow-up (needs arena resize + a d2ee scratch param).
        hess_tmpl = ", EE, false, true, MUJOCO_OUTPUT"
    else:
        qname = "s_q"
        gh_tmpl = ", EE"
        hess_tmpl = ", EE, false, true"
    self.gen_add_code_line("ee_pos_cost<T, EE>(&d_out[k], " + qname + ", s_p_des, s_W, s_end_effector_pose, s_ee_arena, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("ee_pos_cost_gradient<T" + gh_tmpl + ">(&d_grad[k*" + str(nx) + "], " + qname + ", s_p_des, s_W, s_end_effector_pose, s_end_effector_pose_gradient, s_ee_arena, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_code_line("ee_pos_cost_hessian<T" + hess_tmpl + ">(&d_hess[k*" + str(nx*nx) + "], " + qname + ", s_p_des, s_W, s_end_effector_pose, s_end_effector_pose_gradient, nullptr, s_ee_arena, d_robotModel);")
    self.gen_add_sync()
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


def gen_barrier_kernel(self, base, count, var_offset):
    """`<base>_kernel` — one block/timestep value+grad+hess-diag for a barrier.

    `count` bounded DOFs, reading the var slice at `var_offset`. Grad/hess are
    written into standalone packed buffers (offset 0) since the Python surface
    returns the per-DOF gradient/hessian-diagonal directly.
    """
    N = str(count)
    self.gen_add_func_doc(base + "_kernel: value + grad + hess-diagonal per timestep",
                          [], ["d_out scalar barrier cost (1 per timestep)",
                               "d_grad per-DOF gradient (" + N + " per timestep)",
                               "d_hess_diag per-DOF hessian diagonal (" + N + " per timestep)",
                               "d_var variable vector (" + N + " per timestep)",
                               "d_lower / d_upper per-DOF bounds (" + N + " per timestep)",
                               "mu barrier weight; NUM_TIMESTEPS is the batch size"], None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line("void " + base + "_kernel(T *d_out, T *d_grad, T *d_hess_diag, "
                           "const T *d_var, const T *d_lower, const T *d_upper, const T mu, const int NUM_TIMESTEPS) {", True)
    self.gen_add_code_line("__shared__ T s_scratch[" + N + "];")
    self.gen_add_parallel_loop("k", "NUM_TIMESTEPS", block_level=True)
    # zero the scalar out (the value fn ADDS into it), then run the three fns.
    self.gen_add_serial_ops()
    self.gen_add_code_line("d_out[k] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_code_line("const T *s_var = &d_var[k*" + N + "]; const T *s_lo = &d_lower[k*" + N + "]; const T *s_hi = &d_upper[k*" + N + "];")
    # value reads s_var[VAR_OFFSET + i]; we pass an offset-0 view, so call with VAR_OFFSET=0 semantics.
    self.gen_add_code_line(base + "<T>(&d_out[k], s_var, s_lo, s_hi, mu, s_scratch);")
    self.gen_add_sync()
    # grad/hess templates default VAR_OFFSET to the packed slice offset; we pass an
    # offset-0 view of s_var and want offset-0 writes, so force both offsets to 0.
    self.gen_add_parallel_loop("i", N)
    self.gen_add_code_line("d_grad[k*" + N + " + i] = grid_plant_log_barrier_grad<T>(s_var[i], s_lo[i], s_hi[i], mu);")
    self.gen_add_code_line("d_hess_diag[k*" + N + " + i] = grid_plant_log_barrier_hess<T>(s_var[i], s_lo[i], s_hi[i], mu);")
    self.gen_add_end_control_flow()
    self.gen_add_sync()
    self.gen_add_end_control_flow()
    self.gen_add_end_function()


def gen_plant_kernels(self, algorithms):
    """Emit the binding-layer kernels for the plant value surface."""
    self.gen_quadratic_cost_kernel("state")
    self.gen_quadratic_cost_kernel("input")
    nq = self.robot.get_num_pos()
    nv = self.robot.get_num_vel()
    gen_barrier_kernel(self, "joint_position_barrier", nq, 0)
    gen_barrier_kernel(self, "joint_velocity_barrier", nv, nq)
    gen_barrier_kernel(self, "joint_torque_barrier",   nv, 0)
    if "integrator" in algorithms:
        gen_plant_step_kernel(self)
        # Signal to the binding layer (wrapper_template.cu) that plant_step exists.
        self.gen_add_code_line("#define GRID_PLANT_HAS_STEP 1")
    if ("integrator_gradient" in algorithms) or ("integrator_with_gradient" in algorithms):
        gen_plant_step_gradient_kernel(self)
        # GATO ASK6: carve struct for external callers of plant_step_gradient
        # (members map 1:1 onto its caller-placed buffer args).
        gen_plant_step_gradient_arena_carve_struct(self)
        self.gen_add_code_line("#define GRID_PLANT_HAS_STEP_GRADIENT 1")
    # F1: plant_step_hessian composes grid::integrator_hessian_device (needs
    # `fdsva_so`) AND is templated on `grid::IntegratorType` (emitted only by
    # `integrator`), so it requires BOTH. Gating on fdsva_so alone breaks any
    # profile that has fdsva_so without integrator (e.g. the benchmark's per-algo
    # fdsva_so TU): the kernel emits but `grid::IntegratorType` is undefined.
    if ("fdsva_so" in algorithms) and ("integrator" in algorithms):
        gen_plant_step_hessian_kernel(self)
        self.gen_add_code_line("#define GRID_PLANT_HAS_STEP_HESSIAN 1")
    if ("end_effector_pose" in algorithms) and ("end_effector_pose_gradient" in algorithms):
        gen_ee_pos_cost_kernel(self)
        self.gen_add_code_line("#define GRID_PLANT_HAS_EE_COST 1")
        # tracking_cost preset (fixed-base only) is emitted alongside the EE cost in
        # gen_grid_plant; flag it so the smoke runner can guard its preset==composition
        # check (the preset references grid_plant::tracking_cost, absent on floating).
        if not self.robot.floating_base:
            self.gen_add_code_line("#define GRID_PLANT_HAS_TRACKING_COST 1")
    # CoM / centroidal-momentum cost kernels emit whenever their device fns do
    # (gated identically to gen_com_cost/gen_momentum_cost in gen_grid_plant:
    # require grid::com_device + grid::ccrba_device). MIMIC-OK (de-gate #3): the
    # cost emit consumes only the NV-sized com/ccrba output (J_com 3xNV, A 6xNV),
    # which is already mimic-correct (alpha-fold lives inside the centroidal inner,
    # de-gate #1); no body-indexed (NB-stride) scratch lives here, so no NB-vs-NV
    # sizing is needed at the cost layer.
    centroidal_ok = ("com" in algorithms and "ccrba" in algorithms)
    if centroidal_ok:
        gen_com_cost_kernel(self)
        self.gen_add_code_line("#define GRID_PLANT_HAS_COM_COST 1")
        gen_momentum_cost_kernel(self)
        self.gen_add_code_line("#define GRID_PLANT_HAS_MOMENTUM_COST 1")


# ---------------------------------------------------------------------------
# Top-level emit: open the sibling namespace and gate each sub-emit on deps.
# ---------------------------------------------------------------------------

def gen_grid_plant(self, algorithms):
    """Emit the sibling `namespace grid_plant { ... }` block. Called AFTER the
    `grid` namespace closes. Each sub-emit is gated on the `grid::` deps it
    composes being present in `algorithms`; when a dep is missing we emit a
    `#warning`-style comment instead of an undefined call.
    """
    self.gen_add_code_line("")
    self.gen_add_func_doc("Plant namespace: cost / constraint / plant-step primitives composed over grid::")
    self.gen_add_code_line("namespace " + self.file_namespace + "_plant {", True)

    # Quadratic costs have no grid:: dep — always emit.
    self.gen_quadratic_state_cost()
    self.gen_quadratic_input_cost()

    # GATO ASK3: tangent-space (log-map) state cost preset — floating base only
    # (fixed base reduces exactly to quadratic_state_cost). Needs the grid:: Lie
    # helper bundle (difference / dIntegrate_v / d2Integrate blocks); a floating
    # robot with spherical joints additionally needs per-joint SO(3) chart blocks
    # in J_diff — a follow-up, so skip with a note rather than emit wrong math.
    if self.robot.floating_base:
        if getattr(self, "_lie_helpers_emitted", False) and not self.robot.robot_has_spherical():
            gen_quadratic_state_cost_tangent(self)
            self.gen_add_code_line("#define GRID_PLANT_HAS_TANGENT_STATE_COST 1")
        elif self.robot.robot_has_spherical():
            self.gen_add_code_line("// [grid_plant] quadratic_state_cost_tangent skipped: floating base WITH spherical joints needs per-joint SO(3) chart blocks in J_diff (follow-up).")
        else:
            self.gen_add_code_line("// [grid_plant] quadratic_state_cost_tangent skipped: requires the grid:: Lie helper bundle (emit an algorithm that pulls in gen_lie_group_helpers, e.g. 'integrator').")

    # Barriers have no grid:: dep — always emit.
    self.gen_plant_barriers()

    # Plant step needs the integrator value.
    if "integrator" in algorithms:
        self.gen_plant_step()
    else:
        self.gen_add_code_line("// [grid_plant] plant_step skipped: requires the 'integrator' algorithm (grid::integrator_device) — not generated.")

    # Plant step gradient needs the integrator gradient.
    if ("integrator_gradient" in algorithms) or ("integrator_with_gradient" in algorithms):
        self.gen_plant_step_gradient(with_value=False)
        self.gen_plant_step_gradient(with_value=True)
    else:
        self.gen_add_code_line("// [grid_plant] plant_step_gradient[_and_value] skipped: requires 'integrator_gradient' (grid::integrator_gradient_device) — not generated.")

    # Plant step hessian (s_d2AB) needs grid::integrator_hessian_device (`fdsva_so`)
    # AND `grid::IntegratorType` (`integrator`); requires BOTH (see the kernel gate
    # above — fdsva_so-without-integrator profiles otherwise emit an undefined-type ref).
    if ("fdsva_so" in algorithms) and ("integrator" in algorithms):
        self.gen_plant_step_hessian()
    else:
        self.gen_add_code_line("// [grid_plant] plant_step_hessian skipped: requires 'fdsva_so' (grid::integrator_hessian_device) — not generated.")

    # EE position cost needs both ee_pose and ee_pose_gradient. The default (full-
    # Newton) hessian path additionally needs the analytic d2ee
    # ('end_effector_pose_hessian'); without it only GAUSS_NEWTON=true instantiates.
    ee_cost_ok = ("end_effector_pose" in algorithms) and ("end_effector_pose_gradient" in algorithms)
    # GATO ASK2: raw caller-scratch evaluators (no cost coupling). ee_pos needs only
    # the pose family; ee_pos_gradient additionally needs the gradient family.
    if "end_effector_pose" in algorithms:
        gen_ee_raw_evaluators(self, with_gradient = ("end_effector_pose_gradient" in algorithms))
    else:
        self.gen_add_code_line("// [grid_plant] ee_pos / ee_pos_gradient (raw) skipped: requires 'end_effector_pose' — not generated.")
    if ee_cost_ok:
        self.gen_ee_pos_cost(with_d2ee = ("end_effector_pose_hessian" in algorithms))
    else:
        self.gen_add_code_line("// [grid_plant] ee_pos_cost skipped: requires both 'end_effector_pose' and 'end_effector_pose_gradient' (grid::end_effector_pose[_gradient]_device) — not generated.")

    # tracking_cost PRESET (GATO BSQP recipe): composes the EE-pose cost + the
    # always-present quadratic state/input costs + the three barriers. Needs the EE
    # cost (=> the ee_pose deps) and is FIXED-BASE only (the mjx single-reframe
    # composition is a follow-up); floating profiles emit a skip note.
    if ee_cost_ok and not self.robot.floating_base:
        gen_tracking_cost_preset(self)
    elif ee_cost_ok:
        self.gen_add_code_line("// [grid_plant] tracking_cost preset skipped: fixed-base only (NUM_POS == NUM_VEL); floating-base composition is a follow-up.")
    else:
        self.gen_add_code_line("// [grid_plant] tracking_cost preset skipped: requires ee_pos_cost (both 'end_effector_pose' and 'end_effector_pose_gradient').")

    # CoM-tracking / centroidal-momentum-tracking costs need the centroidal
    # kinematics-domain device fns (grid::com_device / grid::ccrba_device), which
    # are emitted ONLY when their `com` / `ccrba` keys are selected. Gating on
    # `end_effector_pose` alone was wrong: a profile that pulls in ee_pose for some
    # OTHER reason (e.g. the frame_jacobian family, whose normalization adds
    # end_effector_pose) but does not request com/ccrba would emit
    # com_cost/momentum_cost referencing undefined grid::com_device/ccrba_device.
    # MIMIC-OK (de-gate #3): com_cost/momentum_cost consume only the NV-sized
    # com/ccrba output (J_com 3xNV, A 6xNV, h 6); the mimic alpha-fold is already
    # baked into that output by the centroidal inner (de-gate #1). The cost emit
    # carries NO body-indexed (NB-stride) scratch — every loop is over NV columns
    # of A/J_com or the NX state dims — so it needs no NB-vs-NV sizing and rides
    # the mimic-correct centroidal output directly (matches the RBDReference oracle,
    # which uses the same NV-sized self.com/jacobian_com/ccrba).
    centroidal_ok = ("com" in algorithms and "ccrba" in algorithms)
    if centroidal_ok:
        gen_com_cost(self)
        gen_momentum_cost(self)
    else:
        self.gen_add_code_line("// [grid_plant] com_cost/momentum_cost skipped: require grid::com_device/ccrba_device (need 'com'+'ccrba').")

    # Binding layer (G1): emit the per-timestep kernels that wrap the device
    # functions above, so the grid_rbd Python/C-ABI surface can launch them.
    self.gen_plant_kernels(algorithms)

    self.gen_add_end_control_flow()  # close namespace grid_plant

"""Generate C-ABI wrapper bodies from ABI_SPECS (P1, increment 1).

Emits the TIGHT-STYLE family of numpy C-ABI bodies in
bindings/grid_rbd/wrapper_template.cu from their AbiSpec rows. The generated
text lives CHECKED-IN between BEGIN/END markers in the template (same pattern
as the baked launch configs): regenerate with

    .venv/bin/python -m grid_codegen.wrapper_body_gen        # rewrites the block
    .venv/bin/python -m grid_codegen.wrapper_body_gen --check # drift check (CI)

test/test_wrapper_generated_block.py runs --check so the block can never
drift from the table. Emission is CANONICAL (uniform formatting; stub voids
every param — two hand bodies were less consistent, reviewed 2026-08-29);
semantic equivalence to the replaced hand bodies was proven by the offline
diff harness before the swap (see auto_dispatch_design_2026-08-28.md).

Emitter rules worth knowing:
- out_size_expr is ALWAYS emitted parenthesized — additive expressions are
  precedence-load-bearing inside `batch * <expr> * sizeof(T)` (a stripped
  paren here was the bug class the offline harness caught).
- Three generated regions (REGIONS): the C-ABI bodies, the kernel_max_threads
  branch table (incr-4a), and the mjx twin bodies (incr-4b — 30 twins from the
  same spec rows; the 4 plant cost twins and tool_fext/fk_batched stay
  hand-written by design).
"""
from __future__ import annotations

import sys
from pathlib import Path

from .abi_specs import ABI_SPECS, AbiSpec


def _vjp_roles() -> tuple[frozenset[str], frozenset[str]]:
    """W04-B B2 (K4): the ops whose native handlers get a model-version STAMP.
    Forward-role = every op with a vjp row (the custom_vjp / autograd forward);
    gradient-role = every op a vjp row's backward calls (grad_op, param_grad_op,
    minv when u_via_minv). Derived from the one vjp table so a new differentiable
    op picks up its stamps by construction."""
    fwd = {k for k, s in ABI_SPECS.items() if s.vjp is not None}
    grad: set[str] = set()
    for k in fwd:
        v = ABI_SPECS[k].vjp
        grad.add(v.grad_op)
        if v.param_grad_op:
            grad.add(v.param_grad_op)
        if v.u_via_minv:
            grad.add("minv")
    return frozenset(fwd), frozenset(grad)


VJP_FORWARD_KEYS, VJP_GRAD_KEYS = _vjp_roles()

# Increment-1 scope: the tight-style family (no sig-fork, no qdd fork, no
# IT dispatch, no body_override).
GENERATED_KEYS: tuple[str, ...] = (
    # increment 1: tight style
    "nonlinear_effects", "generalized_gravity", "coriolis_matrix",
    "energy", "com", "ccrba", "dccrba", "cmm_time_variation",
    "kinetic_energy_regressor", "potential_energy_regressor",
    "frame_jacobian", "frame_jacobian_dot", "osc_inertia",
    # increment 2a: expanded style (sig-fork; std5/so4 template shapes).
    # Canonicalized vs the hand originals: no nj/nv locals (grid:: names
    # inline) and a uniform (size_t)batch cast — both meaning-preserving,
    # verified by the relaxed semantic harness before the swap.
    "minv", "crba", "end_effector_pose", "end_effector_pose_gradient",
    "end_effector_pose_hessian", "idsva_so", "fdsva_so",
    # increment 2b: f_ext-epilogue class (apply/reset around the sync) and the
    # qdd flag-fork pair; two more template shapes (qdd6, fdgrad5). The sig
    # comment is emitted ONCE above a qdd fork (hand originals repeated it in
    # both branches — comment-only canonicalization).
    "forward_dynamics", "aba", "inverse_dynamics",
    "inverse_dynamics_gradient", "forward_dynamics_gradient",
    # increment 3: IT-dispatch bodies (integrator family — launch via the
    # GRID_RBD_IT_DISPATCH macro into the template <IntegratorType> launchers,
    # post-dispatch 200+e consume), the regressor (pure tight style), and the
    # runtime-EE pair (XTOOL_STAGING feature). Only tool_fext + fk_batched
    # remain hand-written (bespoke by design).
    "integrator", "integrator_gradient", "inverse_dynamics_regressor",
    "end_effector_pose_runtime", "end_effector_pose_gradient_runtime",
)

# IT-dispatch rows: the C-ABI body forwards to a hand-written host launcher
# (above the block) that owns the <IntegratorType> template switch.
IT_LAUNCHER: dict[str, str] = {
    "integrator": "launch_integrator_host",
    "integrator_gradient": "launch_integrator_grad_host",
}

# Runtime-EE rows stage the 4x4 col-major SE(3) tool transform to the device
# before launch (identity when offset==nullptr => frame origin).
XTOOL_STAGING: frozenset[str] = frozenset(
    {"end_effector_pose_runtime", "end_effector_pose_gradient_runtime"})
XTOOL_BLOCK = (
    "    // stage the runtime offset (frame origin when offset==nullptr):\n"
    "    // offset is the 4x4 col-major SE(3) tool/tip transform (16 floats); identity => frame origin.\n"
    "    T Xtool[16] = {static_cast<T>(1),0,0,0, 0,static_cast<T>(1),0,0,\n"
    "                   0,0,static_cast<T>(1),0, 0,0,0,static_cast<T>(1)};\n"
    "    if (offset) { for (int i = 0; i < 16; ++i) Xtool[i] = offset[i]; }\n"
    "    if (cudaMemcpy(g_data->d_eepose_runtime_offset, Xtool, 16*sizeof(T),\n"
    "                   cudaMemcpyHostToDevice) != cudaSuccess) return 101;")

BEGIN = "// ── BEGIN GENERATED C-ABI BODIES (grid_codegen/wrapper_body_gen.py — do not hand-edit) ──"
END = "// ── END GENERATED C-ABI BODIES ──"

# ── kernel_max_threads branch table (P1 incr-4a) ─────────────────────────────
# The grid_rbd_kernel_max_threads switch is generated between these markers.
# Row data: (autotune_key, algo_short) in the emitted (= historical) order;
# gate/kernel/enum derive uniformly from algo_short, and the overload cast
# comes from _kernel_attrs.KERNEL_OVERLOADS (the LAST non-single_timing entry
# — for the qdd-forked kernels that is the no-qdd overload; both share
# __launch_bounds__ so either reports the same ceiling), reformatted to the
# switch's compact spelling. The idsva_so frame dispatcher and the batch-2
# divider comment are positioned literals.
CEIL_BEGIN = ("// ── BEGIN GENERATED KERNEL_MAX_THREADS BRANCHES "
              "(grid_codegen/wrapper_body_gen.py — do not hand-edit) ──")
CEIL_END = "// ── END GENERATED KERNEL_MAX_THREADS BRANCHES ──"

_CEIL_DISPATCH = "__idsva_so_dispatch__"
_CEIL_DIVIDER = "__divider__"
CEIL_ROWS: tuple[tuple[str, str], ...] = (
    ("id", "inverse_dynamics"),
    ("minv", "minv"),
    ("fd", "forward_dynamics"),
    ("aba", "aba"),
    ("crba", "crba"),
    ("id_du", "inverse_dynamics_gradient"),
    ("fd_du", "forward_dynamics_gradient"),
    ("ee_pose", "end_effector_pose"),
    ("ee_pose_gradient", "end_effector_pose_gradient"),
    ("ee_pose_hessian", "end_effector_pose_hessian"),
    ("idsva_so", _CEIL_DISPATCH),
    ("fdsva_so", "fdsva_so"),
    ("", _CEIL_DIVIDER),
    ("f_ext_gradient", "f_ext_gradient"),
    ("f_ext_gradient_dq", "f_ext_gradient_dq"),
    ("inverse_dynamics_regressor", "inverse_dynamics_regressor"),
    ("forward_dynamics_parameter_gradient", "forward_dynamics_parameter_gradient"),
    ("kinetic_energy_regressor", "kinetic_energy_regressor"),
    ("potential_energy_regressor", "potential_energy_regressor"),
    ("frame_jacobian", "frame_jacobian"),
    ("frame_jacobian_dot", "frame_jacobian_dot"),
    ("osc_inertia", "osc_inertia"),
    ("generalized_gravity", "generalized_gravity"),
    ("nonlinear_effects", "nonlinear_effects"),
    ("energy", "energy"),
    ("com", "com"),
    ("ccrba", "ccrba"),
    ("coriolis_matrix", "coriolis_matrix"),
    ("dccrba", "dccrba"),
    ("cmm_time_variation", "cmm_time_variation"),
)
# The EE kernels are baked behind static/runtime selector macros — the branch
# takes the macro's address so it queries whichever kernel this .so baked.
CEIL_KERNEL_OVERRIDE: dict[str, str] = {
    "end_effector_pose": "GRID_RBD_EE_POSE_KERNEL",
    "end_effector_pose_gradient": "GRID_RBD_EE_POSE_GRADIENT_KERNEL",
    "end_effector_pose_hessian": "GRID_RBD_EE_POSE_HESSIAN_KERNEL",
}
_CEIL_DISPATCH_BLOCK = """\
    if (std::strcmp(algo, "idsva_so") == 0) {
        // Dispatcher: the codegen emits EXACTLY ONE concrete frame kernel per robot
        // (world for floating/spherical, body for cardinal fixed). Query whichever
        // variant is present, at its frame-specific tier. Frame-specific ceilings can
        // differ (different register footprints) — correct, we want the one that runs.
#if GRID_HAS_IDSVA_SO_WORLD_FRAME
        return GRID_KERNEL_CEIL(idsva_so_world_frame_kernel, GRID_ALGO_IDSVA_SO_WORLD_FRAME,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const T, const int));
#elif GRID_HAS_IDSVA_SO_BODY_FRAME
        return GRID_KERNEL_CEIL(idsva_so_body_frame_kernel, GRID_ALGO_IDSVA_SO_BODY_FRAME,
                                void(*)(T*, unsigned char*, const T*, const int, RM, const T, const int));
#else
        return -1;
#endif
    }"""
_CEIL_DIVIDER_LINE = ("    // ─ batch-2 coverage extension (2026-08-26): "
                      "keys are the FULL symbol names ─")


def _ceil_sig(short: str) -> str:
    """The branch's compact cast: the LAST non-single_timing overload for this
    kernel from KERNEL_OVERLOADS, reformatted (robotModel first — it contains
    'T *'; then the pointer-spacing collapses)."""
    from .kernel_attrs import KERNEL_OVERLOADS
    cands = [sig for name, sig in KERNEL_OVERLOADS[short]
             if not name.startswith(short + "_kernel_single_timing")]
    sig = cands[-1]
    sig = sig.replace("void (*)", "void(*)")
    sig = sig.replace("const robotModel<T> *", "RM")
    sig = sig.replace("unsigned char *", "unsigned char*")
    return sig.replace("T *", "T*")


def gen_ceil_block() -> str:
    parts = [CEIL_BEGIN,
             "// Regenerate: .venv/bin/python -m grid_codegen.wrapper_body_gen",
             "// Rows: CEIL_ROWS (keys crosschecked against the descriptor table's",
             "// autotune_keys by test/test_abi_spec_crosscheck.py)."]
    for key, short in CEIL_ROWS:
        if short == _CEIL_DISPATCH:
            parts.append(_CEIL_DISPATCH_BLOCK)
            continue
        if short == _CEIL_DIVIDER:
            parts.append(_CEIL_DIVIDER_LINE)
            continue
        kern = CEIL_KERNEL_OVERRIDE.get(short, short + "_kernel")
        up = short.upper()
        parts.append(f"""\
#if GRID_HAS_{up}
    if (std::strcmp(algo, "{key}") == 0)
        return GRID_KERNEL_CEIL({kern}, GRID_ALGO_{up},
                                {_ceil_sig(short)});
#endif""")
    parts.append(CEIL_END)
    return "\n".join(parts) + "\n"

# Load-bearing per-body comments preserved from the hand-written originals.
# The standard signature-switch comment (verbatim from the hand originals).
SIG_COMMENT = (
    "// signature switch: the host template carries MUJOCO_OUTPUT on floating\n"
    "// builds regardless of enable_mujoco_kernels — keyed on the per-fn\n"
    "// GRID_RBD_SIG_MJX_* flag _compile.py derives from the generated header\n"
    "// (NOT on GRID_RBD_WITH_MUJOCO, the mjx-KERNELS gate).")

# Load-bearing per-row comments preserved from the hand originals.
PRE_PACK_COMMENTS: dict[str, str] = {
    "idsva_so": (
        "    // idsva_so reads the joint acceleration from the u-slot of d_q_qd_u (s_qdd);\n"
        "    // pack qdd there so the second-order tensors use the requested acceleration."),
}
PRE_LAUNCH_COMMENTS: dict[str, str] = {
    "idsva_so": (
        "// RESOURCE_TIER must match the tier the autotuned thread count was picked for —\n"
        "// the default-tier instantiation with a LITE-tuned count exceeded the default\n"
        "// kernel's register-limited thread cap and failed the launch (invalid argument)."),
    "fdsva_so": (
        "// RESOURCE_TIER must match the autotuned tier (see idsva_so note)."),
}
PRE_COPY_COMMENTS: dict[str, str] = {
    "crba": (
        "    // Device-direct copy at the nv*nv kernel stride: the generated host wrapper's\n"
        "    // h_M staging used the nj*nj stride (over-read for floating; see the nj-stride\n"
        "    // host-wrapper item). FIXED base has nv == nj so this is byte-identical."),
    "minv": (
        "    // Device-direct copy at the nv*nv kernel stride (same nj-stride staging issue\n"
        "    // as crba)."),
}

NO_EXIT_COMMENT = (
    "    // NO_EXIT builds: a LAUNCH-time failure inside a generated host wrapper is\n"
    "    // recorded in the sticky slot (the stream stays empty, so the sync above\n"
    "    // returns success — the silent stale-buffer class). Consume it here so the\n"
    "    // caller gets a loud rc instead of plausible garbage.")

BODY_COMMENTS: dict[str, str] = {
    "frame_jacobian": (
        "    // -1 per arg => \"use default\" (leaf-EE / LWA); the host resolves each\n"
        "    // INDEPENDENTLY, so a default target with an explicit frame is honored."),
    "frame_jacobian_dot": (
        "    // -1 per arg => \"use default\" (leaf-EE / LWA); the host resolves each\n"
        "    // INDEPENDENTLY, so a default target with an explicit frame is honored."),
}

_PACK = {
    "q_qd_null": "    pack_q_qd_u(g_ctx, q, qd, nullptr, batch, grid::NUM_JOINTS);",
    "q_q_null": "    pack_q_qd_u(g_ctx, q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);",
    "q_qd_u": "    pack_q_qd_u(g_ctx, q, qd, u, batch, grid::NUM_JOINTS);",
    "qdd_u_slot": "    pack_q_qd_u(g_ctx, q, qd, qdd, batch, grid::NUM_JOINTS);",
    "pack_q": "    pack_q(g_ctx, q, batch, grid::NUM_JOINTS);",
}

_STUB_MSG = {
    "subset": "not built into this .so (subset profile)",
    "reduced": "not generated for this robot (reduced codegen profile)",
    "bare": "not built",
}


def _paren(expr: str) -> str:
    e = expr.strip()
    return e if (e.startswith("(") and e.endswith(")")) or ("+" not in e and "-" not in e) else f"({e})"


def gen_body(spec: AbiSpec) -> str:
    stem = spec.abi_stem or spec.key
    gate = spec.gate_macro or ("GRID_HAS_" + spec.key.upper())
    launch = spec.launch_algo or ("GRID_ALGO_" + spec.key.upper())
    sym = spec.grid_symbol or ("grid::" + spec.key)
    pnames = [n for n, _t in spec.inputs]
    out_name = next(n for n in pnames if n.endswith("out") or n == "out")
    sig = ", ".join(f"{t} {n}" for (n, t) in spec.inputs)
    L = [f'extern "C" int grid_rbd_{stem}(long long ctx_id, {sig}) {{',
         f"#{spec.gate_form} {gate}",
         "    GRID_RBD_CTX_OR_RETURN(ctx_id);",
         "    if (batch > kMaxBatch) return 2;"]
    if spec.key in PRE_PACK_COMMENTS:
        L.append(PRE_PACK_COMMENTS[spec.key])
    L.append(_PACK[spec.pack_mode])
    if spec.key in XTOOL_STAGING:
        L.append(XTOOL_BLOCK)
    if spec.template_shape != "plain":
        return _gen_expanded(spec, L)
    if spec.key in BODY_COMMENTS:
        L.append(BODY_COMMENTS[spec.key])
    if spec.it_dispatch:
        L.append(f"    GRID_RBD_IT_DISPATCH(it, {IT_LAUNCHER[spec.key]}, batch, gravity, dt);")
    else:
        grav = "gravity, " if spec.takes_gravity else ""
        trail = "".join(", " + a for a in spec.trailing_runtime_args)
        dims = f"grid_rbd_launch_threads_n<grid::{launch}>(g_ctx, batch)"
        if spec.clamp_kernel:
            dims = f"grid_clamp_threads_for({spec.clamp_kernel}, {dims})"
        L.append(f"    {sym}<T>(g_data, g_robot, {grav}batch, "
                 f"dim3((unsigned)batch, 1, 1), {dims}, g_streams{trail});")
    if spec.pre_launch_check:
        L.append("    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }")
    L.append("    if (int rc = grid_rbd_sync_consume()) return rc;")
    size = _paren(spec.out_size_expr)
    if spec.out_copy == "memcpy_h":
        L.append(f"    std::memcpy({out_name}, g_data->{spec.out_buffer}, "
                 f"(size_t)batch * {size} * sizeof(T));")
    else:
        L.append(f"    gpuErrchk(cudaMemcpy({out_name}, g_data->{spec.out_buffer}, "
                 f"(size_t)batch * {size} * sizeof(T), cudaMemcpyDeviceToHost));")
    msg = _STUB_MSG.get(spec.not_built_msg, spec.not_built_msg)
    L += ["    return 0;",
          "#else",
          "    " + " ".join(f"(void){n};" for n in pnames),
          f"    return 3;  // {stem} {msg}",
          "#endif",
          "}"]
    return "\n".join(L) + "\n"


def _size_c(expr: str) -> str:
    """Spell an out_size_expr with fully-qualified names (canonical style)."""
    import re as _re
    e = _re.sub(r"\bnv\b", "grid::NUM_VEL", expr)
    e = _re.sub(r"\bnj\b", "grid::NUM_JOINTS", e)
    return _paren(e)


def _gen_expanded(spec: AbiSpec, L: list[str]) -> str:
    stem = spec.abi_stem or spec.key
    launch = spec.launch_algo or ("GRID_ALGO_" + spec.key.upper())
    sym = spec.grid_symbol or ("grid::" + spec.key)
    pnames = [n for n, _t in spec.inputs]
    out_name = next(n for n in pnames if n.endswith("out") or n == "out")
    tier = f"/*RESOURCE_TIER=*/grid::launch_cfg<grid::{launch}>::TIER"

    def mid(qdd_flag=None):
        if spec.template_shape == "std5":
            return "/*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL"
        if spec.template_shape == "qdd6":
            return (f"/*USE_QDD_FLAG=*/{qdd_flag}, /*USE_COMPRESSED_MEM=*/false, "
                    "/*KIND=*/grid::GRID_DATA_ALL")
        if spec.template_shape == "fdgrad5":
            return "/*USE_QDD_MINV_FLAG=*/false, /*KIND=*/grid::GRID_DATA_ALL"
        return "/*KIND=*/grid::GRID_DATA_ALL"  # so4

    grav = "gravity, " if spec.takes_gravity else ""

    def launch_lines(qdd_flag=None, indent="    "):
        out = [f"#if defined({spec.sig_mjx_macro})",
               f"{indent}{sym}<T, {mid(qdd_flag)}, /*MUJOCO_OUTPUT=*/false, {tier}>(",
               "#else",
               f"{indent}{sym}<T, {mid(qdd_flag)}, {tier}>(",
               "#endif",
               f"{indent}    g_data, g_robot, {grav}batch, dim3((unsigned)batch, 1, 1), "
               f"grid_rbd_launch_threads_n<grid::{launch}>(g_ctx, batch), g_streams);"]
        return out

    if spec.f_ext_mode == "optional":
        L.append("    if (int rc = apply_f_ext(g_ctx, f_ext, batch)) return rc;")
    L.append("")
    L.append(SIG_COMMENT)
    if spec.key in PRE_LAUNCH_COMMENTS:
        L.append(PRE_LAUNCH_COMMENTS[spec.key])
    if spec.qdd_route == "flag_fork":
        L.append("    if (qdd_opt) {")
        L.append("        // Host wrapper copies h_qdd->d_qdd (NUM_JOINTS per timestep, contiguous).")
        L.append("        std::memcpy(g_data->h_qdd, qdd_opt, (size_t)batch * grid::NUM_JOINTS * sizeof(T));")
        L.extend(launch_lines("true", indent="        "))
        L.append("    } else {")
        L.extend(launch_lines("false", indent="        "))
        L.append("    }")
    else:
        L.extend(launch_lines())
    L.append("")
    if spec.f_ext_mode == "optional":
        # reset_f_ext must run BETWEEN the sync and the return — the epilogue
        # sync_consume cannot express (the 11-site class from the audit).
        L.append("    cudaError_t e = cudaDeviceSynchronize();")
        L.append(NO_EXIT_COMMENT)
        L.append("    if (e == cudaSuccess) e = grid_consume_last_error();")
        L.append("    reset_f_ext(g_ctx, f_ext, batch);")
        L.append("    if (e != cudaSuccess) return 100 + (int)e;")
    else:
        L.append("    if (int rc = grid_rbd_sync_consume()) return rc;")
    L.append("")
    if spec.key in PRE_COPY_COMMENTS:
        L.append(PRE_COPY_COMMENTS[spec.key])
    size = _size_c(spec.out_size_expr)
    if spec.out_copy == "memcpy_h":
        L.append(f"    std::memcpy({out_name}, g_data->{spec.out_buffer}, "
                 f"(size_t)batch * {size} * sizeof(T));")
    else:
        L.append(f"    cudaMemcpy({out_name}, g_data->{spec.out_buffer}, "
                 f"(size_t)batch * {size} * sizeof(T), cudaMemcpyDeviceToHost);")
    msg = _STUB_MSG.get(spec.not_built_msg, spec.not_built_msg)
    L += ["    return 0;",
          "#else",
          "    " + " ".join(f"(void){n};" for n in pnames),
          f"    return 3;  // {stem} {msg}",
          "#endif",
          "}"]
    return "\n".join(L) + "\n"


def gen_block() -> str:
    parts = [BEGIN,
             "// Regenerate: .venv/bin/python -m grid_codegen.wrapper_body_gen",
             "// Table: grid_codegen/abi_specs.py (ABI_SPECS); drift-gated by",
             "// test/test_wrapper_generated_block.py.",
             ""]
    for key in GENERATED_KEYS:
        parts.append(gen_body(ABI_SPECS[key]))
    parts.append(END)
    return "\n".join(parts) + "\n"


# ── mjx twin bodies (P1 incr-4b) ─────────────────────────────────────────────
# The grid_rbd_<stem>_mujoco C-ABI twins, generated from the SAME spec rows as
# their pin siblings plus the mjx_* fields. Twin-vs-pin deltas the emitter
# models: the GRID_RBD_WITH_MUJOCO gate is OUTSIDE the function (symbol absent
# on a non-mjx build — the python surface probes by symbol presence), the host
# template always carries /*MUJOCO_OUTPUT=*/true (no SIG_MJX fork — the twin
# only compiles where the mjx signature exists), qdd-required rows replace the
# pin flag-fork with an rc=4 prologue + unconditional qdd path, mjx_omits_tier
# rows launch the DEFAULT-tier instantiation (ported hand behavior — such
# twins never pass launch_cfg tier), and the runtime-EE pair keeps its inner
# HAS-gate + rc=3 stub so the symbol exists on every mjx build.
MJX_BEGIN = ("// ── BEGIN GENERATED MJX TWIN BODIES "
             "(grid_codegen/wrapper_body_gen.py — do not hand-edit) ──")
MJX_END = "// ── END GENERATED MJX TWIN BODIES ──"

MJX_KEYS: tuple[str, ...] = tuple(
    k for k in GENERATED_KEYS if ABI_SPECS[k].has_mjx_twin)

# Twins whose function keeps an INNER gate + rc=3 stub (symbol present on any
# mjx build). Everyone else gates the whole function away.
MJX_INNER_GATE: frozenset[str] = XTOOL_STAGING

_MJX_QDD_REQ = "    if (!qdd_opt) return 4;  // mjx requires an explicit qdd"
_MJX_QDD_COPY = (
    "    // Host wrapper copies h_qdd->d_qdd (NUM_JOINTS per timestep, contiguous).\n"
    "    std::memcpy(g_data->h_qdd, qdd_opt, (size_t)batch * grid::NUM_JOINTS * sizeof(T));")
_POST_LAUNCH = "    { cudaError_t _le = cudaGetLastError(); if (_le != cudaSuccess) return 200 + (int)_le; }"


def _mjx_mid(spec: AbiSpec) -> str:
    if spec.template_shape == "qdd6":
        return "/*USE_QDD_FLAG=*/true, /*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL"
    if spec.template_shape == "fdgrad5":
        return "/*USE_QDD_MINV_FLAG=*/false, /*KIND=*/grid::GRID_DATA_ALL"
    if spec.template_shape == "so4":
        return "/*KIND=*/grid::GRID_DATA_ALL"
    # std5 — and every "plain" pin row's twin, which calls the expanded
    # template form (the tight <T> spelling has no MUJOCO_OUTPUT slot).
    return "/*USE_COMPRESSED_MEM=*/false, /*KIND=*/grid::GRID_DATA_ALL"


def gen_mjx_body(spec: AbiSpec) -> str:
    from .wrapper_mjx_docs import MJX_DOC
    stem = spec.abi_stem or spec.key
    gate = spec.gate_macro or ("GRID_HAS_" + spec.key.upper())
    launch = spec.launch_algo or ("GRID_ALGO_" + spec.key.upper())
    sym = spec.grid_symbol or ("grid::" + spec.key)
    pnames = [n for n, _t in spec.inputs]
    out_name = next(n for n in pnames if n.endswith("out") or n == "out")
    sig = ", ".join(f"{t} {n}" for (n, t) in spec.inputs)
    inner = spec.key in MJX_INNER_GATE

    L = []
    if inner:
        L.append("#ifdef GRID_RBD_WITH_MUJOCO")
    else:
        L.append(f"#if defined(GRID_RBD_WITH_MUJOCO) && {gate}")
    if stem in MJX_DOC:
        L.append(MJX_DOC[stem])
    L.append(f'extern "C" int grid_rbd_{stem}_mujoco({sig}) {{')
    if inner:
        L.append(f"#{spec.gate_form} {gate}")
    if spec.mjx_requires_qdd:
        L.append(_MJX_QDD_REQ)
    L.append("    GRID_RBD_CTX_OR_RETURN(ctx_id);")
    L.append("    if (batch > kMaxBatch) return 2;")
    L.append(_PACK[spec.pack_mode])
    if spec.key in XTOOL_STAGING:
        L.append(XTOOL_BLOCK)
    if spec.f_ext_mode == "optional":
        L.append("    if (int rc = apply_f_ext(g_ctx, f_ext, batch)) return rc;")
    if spec.mjx_requires_qdd:
        L.append(_MJX_QDD_COPY)

    if spec.it_dispatch or spec.mjx_it_dispatch:
        macro = ("GRID_RBD_IT_DISPATCH_HESSIAN" if spec.mjx_it_dispatch == "HESSIAN"
                 else "GRID_RBD_IT_DISPATCH")
        L.append(f"    {macro}(it, {IT_LAUNCHER[spec.key]}_mujoco, batch, gravity, dt);")
    else:
        grav = "gravity, " if spec.takes_gravity else ""
        trail = "".join(", " + a for a in spec.trailing_runtime_args)
        tier = ("" if spec.mjx_omits_tier
                else f", /*RESOURCE_TIER=*/grid::launch_cfg<grid::{launch}>::TIER")
        L.append(f"    {sym}<T, {_mjx_mid(spec)}, /*MUJOCO_OUTPUT=*/true{tier}>(")
        L.append(f"        g_data, g_robot, {grav}batch, dim3((unsigned)batch, 1, 1), "
                 f"grid_rbd_launch_threads_n<grid::{launch}>(g_ctx, batch), g_streams{trail});")
    if spec.pre_launch_check or spec.mjx_post_launch_check:
        L.append(_POST_LAUNCH)

    if spec.f_ext_mode == "optional":
        L.append("    cudaError_t e = cudaDeviceSynchronize();")
        L.append(NO_EXIT_COMMENT)
        L.append("    if (e == cudaSuccess) e = grid_consume_last_error();")
        L.append("    reset_f_ext(g_ctx, f_ext, batch);")
        L.append("    if (e != cudaSuccess) return 100 + (int)e;")
    else:
        L.append("    if (int rc = grid_rbd_sync_consume()) return rc;")

    size = _size_c(spec.out_size_expr)
    if spec.out_copy == "memcpy_h":
        L.append(f"    std::memcpy({out_name}, g_data->{spec.out_buffer}, "
                 f"(size_t)batch * {size} * sizeof(T));")
    else:
        L.append(f"    cudaMemcpy({out_name}, g_data->{spec.out_buffer}, "
                 f"(size_t)batch * {size} * sizeof(T), cudaMemcpyDeviceToHost);")
    L.append("    return 0;")
    if inner:
        L += ["#else",
              "    " + " ".join(f"(void){n};" for n in pnames),
              "    return 3;",
              "#endif"]
    L.append("}")
    if inner:
        L.append("#endif  // GRID_RBD_WITH_MUJOCO")
    else:
        L.append(f"#endif  // GRID_RBD_WITH_MUJOCO && {gate}")
    return "\n".join(L) + "\n"


def gen_mjx_block() -> str:
    parts = [MJX_BEGIN,
             "// Regenerate: .venv/bin/python -m grid_codegen.wrapper_body_gen",
             "// Table: grid_codegen/abi_specs.py (mjx_* fields); docs verbatim from",
             "// grid_codegen/wrapper_mjx_docs.py. The 4 plant cost twins stay hand-written.",
             ""]
    for key in MJX_KEYS:
        parts.append(gen_mjx_body(ABI_SPECS[key]))
    parts.append(MJX_END)
    return "\n".join(parts) + "\n"


# ── torch/jax surface regions (#4-6, A2 2026-09-11) ─────────────────────────
# The torch op bodies, the jax FFI handlers (+ their BIND registrations), and
# the torch X-macro op table, generated from the same ABI_SPECS rows (A1's
# kernel_args / kernel_symbol / jax_buffer_inputs / hoist_out_size fields plus
# the registry's tier_blind_bytes for the smem-call tier spelling, A3).
# Emitters were proven offline 2026-09-10 against the hand-written units
# (torch 24/24 byte-identical, jax 26/26 semantically equal under the
# documented canonicalization); the region landing normalizes the enumerated
# hand quirks (per-op local-decl layout, one misaligned wrap column, jax
# per-op formatting/local-naming variance, the id handler's tautological
# last-dim check) — every delta reviewed at the swap, receipts re-proven by
# the wrapper-domain refresh. Bespoke surface bodies (torch id/id_grad qdd
# forks, both integrators, idsva_so, runtime-EE pair; jax idsva_so, both
# integrators, runtime-EE pair; all plant ops) stay hand-written OUTSIDE the
# regions.
from .abi_specs import (
    jax_buffer_inputs_for, jax_substitution_keys,
    kernel_launch_args, kernel_symbol_for, smem_bytes_call,
    torch_substitution_keys, torch_tensor_args)
from .wrapper_surface_docs import (
    JAX_OP_DOCS, JAX_TWIN_BIND_DOC_OPS, SECTION_BANNERS, TORCH_OP_DOCS)

TORCH_BODIES_BEGIN = ("// ── BEGIN GENERATED TORCH OP BODIES "
                      "(grid_codegen/wrapper_body_gen.py — do not hand-edit) ──")
TORCH_BODIES_END = "// ── END GENERATED TORCH OP BODIES ──"
JAX_HANDLERS_BEGIN = ("// ── BEGIN GENERATED JAX FFI HANDLERS "
                      "(grid_codegen/wrapper_body_gen.py — do not hand-edit) ──")
JAX_HANDLERS_END = "// ── END GENERATED JAX FFI HANDLERS ──"
TORCH_OPS_BEGIN = ("// ── BEGIN GENERATED TORCH OP TABLE "
                   "(grid_codegen/wrapper_body_gen.py — do not hand-edit) ──")
TORCH_OPS_END = "// ── END GENERATED TORCH OP TABLE ──"

# Region order = the hand-written file order at the swap (banners keyed to the
# op they preceded). jax additionally emits id/id_grad (substitution there,
# bespoke qdd-fork bodies on torch).
TORCH_SURFACE_KEYS: tuple[str, ...] = (
    "minv", "forward_dynamics", "aba", "crba", "end_effector_pose",
    "end_effector_pose_gradient", "end_effector_pose_hessian",
    "forward_dynamics_gradient", "fdsva_so", "inverse_dynamics_regressor",
    "forward_dynamics_parameter_gradient", "generalized_gravity",
    "nonlinear_effects", "coriolis_matrix", "kinetic_energy_regressor",
    "potential_energy_regressor", "energy", "com", "ccrba",
    "cmm_time_variation", "dccrba", "frame_jacobian", "frame_jacobian_dot",
    "osc_inertia")
JAX_SURFACE_KEYS: tuple[str, ...] = (
    "inverse_dynamics", "minv", "forward_dynamics", "aba", "crba",
    "end_effector_pose", "end_effector_pose_gradient",
    "end_effector_pose_hessian", "inverse_dynamics_gradient",
    "forward_dynamics_gradient", "fdsva_so", "inverse_dynamics_regressor",
    "forward_dynamics_parameter_gradient", "generalized_gravity",
    "nonlinear_effects", "coriolis_matrix", "kinetic_energy_regressor",
    "potential_energy_regressor", "energy", "com", "ccrba",
    "cmm_time_variation", "dccrba", "frame_jacobian", "frame_jacobian_dot",
    "osc_inertia")
_TORCH_REGION_BANNERS = {"inverse_dynamics_regressor": "torch_sysid",
                         "generalized_gravity": "torch_ptier1"}
_JAX_REGION_BANNERS = {"inverse_dynamics_regressor": "jax_regressor",
                       "generalized_gravity": "jax_ptier1",
                       "energy": "jax_wave2",
                       "frame_jacobian": "jax_wave3"}

# Load-bearing per-op comments inside the torch bodies (verbatim from the hand
# originals; same preserved-comment pattern as the C-ABI emitters above).
TORCH_PRE_PACK_COMMENTS: dict[str, str] = {
    "inverse_dynamics_regressor":
        "// qdd occupies the u-slot (read as the acceleration; mirrors the JAX handler).",
}
TORCH_PRE_ALLOC_COMMENTS: dict[str, str] = {
    "minv": (
        "// Minv is nv x nv (tangent-space); the kernel writes d_Minv nv*nv-strided.\n"
        "    // Size the output + copy at nv*nv (unified with numpy/JAX). FIXED base: nv == nj."),
    "crba": (
        "// M is nv x nv (tangent-space); the kernel writes d_M nv*nv-strided. Size the\n"
        "    // output + copy at nv*nv (unified with numpy/JAX). FIXED base: nv == nj."),
    "forward_dynamics_gradient": (
        "// df_du is nv x 2nv (tangent-space); the kernel writes d_df_du 2*nv*nv-strided.\n"
        "    // Size + copy at 2*nv*nv (unified with numpy/JAX). FIXED base: nv == nj."),
}

_SURF_DIM_ORDER = (("grid::NUM_JOINTS", "nj"), ("grid::NUM_VEL", "nv"),
                   ("grid::NUM_BODIES", "nb"), ("GRID_RBD_NUM_EES", "nee"))


def _surface_gate(spec: AbiSpec) -> tuple[str, str]:
    """(opener, closer) preprocessor lines of one surface unit. gate_requires
    flattens the old nested frame-family blocks into composite gates."""
    gate = spec.gate_macro or ("GRID_HAS_" + spec.key.upper())
    if spec.gate_requires:
        macros = (*spec.gate_requires, gate)
        cond = " && ".join(f"defined({m})" for m in macros)
        return f"#if {cond}", f"#endif  // {' && '.join(macros)}"
    if spec.gate_form == "ifdef":
        return f"#ifdef {gate}", f"#endif  // {gate}"
    return f"#if {gate}", f"#endif  // {gate}"


def _surface_size(spec: AbiSpec) -> tuple[str, str, list[str]]:
    """(alloc_size, copy_size, local dims used): out_size_expr with dims as
    locals — memcpy keeps the original parenthesization, alloc strips an outer
    paren pair. SECOND_ORDER-sized ops keep the constant inline; hoisted ops
    route through `out_size` (handled by the callers)."""
    e = spec.out_size_expr
    if "SECOND_ORDER" in e:
        return e, e, []
    dims = []
    for full, loc in _SURF_DIM_ORDER:
        if full in e:
            e = e.replace(full, loc)
            dims.append(loc)
    e = " ".join(e.replace("*", " * ").split())
    alloc = e[1:-1].strip() if e.startswith("(") and e.endswith(")") else e
    return alloc, e, dims


def emit_torch_body(key: str) -> str:
    spec = ABI_SPECS[key]
    tensors = list(torch_tensor_args(spec))
    templated = spec.has_mjx_twin
    args = [f"torch::Tensor {n}" for n in tensors]
    args += [f"int64_t {n}" for n in ("target_jid", "reference_frame")
             if n in spec.trailing_runtime_args]
    if spec.takes_gravity:
        args.append("double gravity")
    if spec.f_ext_mode == "optional":
        args.append("c10::optional<torch::Tensor> f_ext")
    args.append("int64_t ctx_id")  # W04-B B1: the context id (after it: only the B2 stamp)
    role = ("fwd" if key in VJP_FORWARD_KEYS else "grad" if key in VJP_GRAD_KEYS else None)
    if role == "fwd":
        args.append("c10::optional<torch::Tensor> stamp_out")       # B2 (K4): version stamp OUT
    elif role == "grad":
        args.append("c10::optional<torch::Tensor> stamp_expect")    # B2 (K4): version stamp CHECK
    prefix = ("template <bool MUJOCO>\n" if templated else "") + \
        f"torch::Tensor torch_{key}("
    if any(a.startswith("c10::optional") for a in args):
        # the optional f_ext alone wraps, aligned to the open paren
        col = len(prefix.rsplit("\n", 1)[-1])
        sig = prefix + ", ".join(args[:-1]) + ",\n" + " " * col + args[-1] + ") {"
    else:
        sig = prefix + ", ".join(args) + ") {"
    alloc_size, copy_size, size_dims = _surface_size(spec)
    if spec.hoist_out_size:
        alloc_size = copy_size = "out_size"
        size_dims = []
    dims = [("grid::NUM_JOINTS", "nj")] + [
        (f, l) for f, l in _SURF_DIM_ORDER if l in size_dims and l != "nj"]
    decls = "    const int " + ", ".join(f"{l} = {f}" for f, l in dims) + ";"
    checks = [f'grid_torch_check({n}, "{key}: {n}", nj);' for n in tensors]
    joined = "    " + " ".join(checks)
    check_lines = [joined] if len(joined) <= 140 else ["    " + c for c in checks]
    pack = ["&" + n for n in tensors[:3]] + ["nullptr"] * (3 - len(tensors))
    algo = spec.launch_algo or ("GRID_ALGO_" + key.upper())
    ksym = kernel_symbol_for(spec)
    targs = (f"T, grid::launch_cfg<grid::{algo}>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO"
             if templated else "T")
    L = [sig, "    GRID_RBD_CTX_OR_THROW(ctx_id);", decls, *check_lines,
         "    int batch = grid_torch_batch(q);",
         "    cudaStream_t stream = at::cuda::getCurrentCUDAStream();"]
    if role == "grad":
        L.append("    grid_torch_stamp_check(g_ctx, stream, stamp_expect);")
    if key in TORCH_PRE_PACK_COMMENTS:
        L.append("    " + TORCH_PRE_PACK_COMMENTS[key])
    L.append(f"    grid_torch_pack(g_ctx, stream, batch, nj, {', '.join(pack)});")
    if spec.f_ext_mode == "optional":
        L.append("    grid_torch_f_ext_apply(g_ctx, stream, batch, f_ext);")
    if key in TORCH_PRE_ALLOC_COMMENTS:
        L.append("    " + TORCH_PRE_ALLOC_COMMENTS[key])
    if spec.hoist_out_size:
        L.append(f"    const int out_size = {spec.out_size_expr};")
    kargs = ", ".join(kernel_launch_args(spec, "torch"))
    L += [f"    auto out = grid_torch_empty(batch, {alloc_size}, q);",
          "    constexpr int stride = 3 * grid::NUM_JOINTS;",
          f"    grid::{ksym}<{targs}><<<grid_rbd_grid_for(g_ctx, batch), "
          f"grid_rbd_launch_threads_n<grid::{algo}>(g_ctx, batch), {smem_bytes_call(spec)}, stream>>>(",
          f"        {kargs});",
          f'    grid_torch_check_launch("{ksym}");']
    batch_sz = "(size_t)batch" if spec.hoist_out_size else "batch"
    out_arg = kernel_launch_args(spec, "torch")[0]
    L.append(f"    cudaMemcpyAsync(out.data_ptr<T>(), {out_arg}, "
             f"{batch_sz} * {copy_size} * sizeof(T), cudaMemcpyDeviceToDevice, stream);")
    if spec.f_ext_mode == "optional":
        L.append("    grid_torch_f_ext_reset(g_ctx, stream, batch, f_ext);")
    if role == "fwd":
        L.append("    grid_torch_stamp_write(g_ctx, stream, stamp_out);")
    L += ["    return out;", "}"]
    return "\n".join(L) + "\n"


def gen_torch_bodies_block() -> str:
    parts = [TORCH_BODIES_BEGIN,
             "// Regenerate: .venv/bin/python -m grid_codegen.wrapper_body_gen",
             "// Table: grid_codegen/abi_specs.py (kernel_args et al.); docs verbatim",
             "// from grid_codegen/wrapper_surface_docs.py.",
             ""]
    assert set(TORCH_SURFACE_KEYS) == set(torch_substitution_keys())
    for key in TORCH_SURFACE_KEYS:
        if key in _TORCH_REGION_BANNERS:
            parts.append(SECTION_BANNERS[_TORCH_REGION_BANNERS[key]])
        opener, closer = _surface_gate(ABI_SPECS[key])
        doc = TORCH_OP_DOCS.get(key)
        parts.append(opener + "\n" + (doc if doc else "")
                     + emit_torch_body(key) + closer + "\n")
    parts.append(TORCH_BODIES_END)
    return "\n".join(parts) + "\n"


def emit_jax_handler(key: str) -> str:
    spec = ABI_SPECS[key]
    bufs = list(jax_buffer_inputs_for(spec))
    kargs_tokens = spec.jax_kernel_args or spec.kernel_args
    staged_qdd = "QDD" in kargs_tokens
    packed = [b for b in bufs if b != "f_ext" and not (b == "qdd" and staged_qdd)]
    templated = spec.has_mjx_twin
    sig = ["    cudaStream_t stream"]
    sig += [f"    ffi::Buffer<GRID_FFI_T> {b}" for b in bufs]
    sig.append("    ffi::ResultBuffer<GRID_FFI_T> out")
    sig += [f"    int64_t {a}" for a in spec.trailing_runtime_args]
    if spec.takes_gravity:
        sig.append("    T gravity")
    role = ("fwd" if key in VJP_FORWARD_KEYS else "grad" if key in VJP_GRAD_KEYS else None)
    _alloc, copy_size, size_dims = _surface_size(spec)
    dims = [("grid::NUM_JOINTS", "nj")] + [
        (f, l) for f, l in _SURF_DIM_ORDER if l in size_dims and l != "nj"]
    L = []
    if role is None:
        sig.append("    int64_t ctx_id")  # W04-B B1: the Bind chain's LAST attr
        if templated:
            L.append("template <bool MUJOCO>")
        L.append(f"static ffi::Error grid_rbd_jax_{key}_impl(")
        L.append(",\n".join(sig) + ")")
        L.append("{")
        L.append("    GRID_RBD_CTX_OR_FFI(ctx_id);")
    else:
        # B2 (K4): the role keys split into ONE body (context passed in) and the
        # plain / `_stamped` (forward) or `_checked` (gradient) entry shims, so the
        # stamp write / check sits inside the SAME admission scope as the launch.
        body_sig = ["    GridCtx *ctx, cudaStream_t stream"] + sig[1:]
        if templated:
            L.append("template <bool MUJOCO>")
        L.append(f"static ffi::Error grid_rbd_jax_{key}_body(")
        L.append(",\n".join(body_sig) + ")")
        L.append("{")
        L.append("    GRID_RBD_CTX_LOCALS(ctx);")
    L.append(f'    GRID_RBD_FFI_VALIDATE_2D(q, "{key}: q", grid::NUM_JOINTS);')
    L.append("    int batch = (int)q.dimensions()[0];")
    L.append("    int " + ", ".join(f"{l} = {f}" for f, l in dims) + ";")
    L.append(f'    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("{key}: batch > max_batch");')
    L.append("    const size_t row_bytes = nj * sizeof(T);")
    L.append("    const size_t dst_pitch = 3 * nj * sizeof(T);")
    slots = ["0", "nj", "2*nj"]
    for i, b in enumerate(packed):
        L.append(f"    cudaMemcpy2DAsync(&g_data->d_q_qd_u[{slots[i]}], dst_pitch, "
                 f"{b}.typed_data(), row_bytes, row_bytes, batch, cudaMemcpyDeviceToDevice, stream);")
    if staged_qdd:
        L.append("    cudaMemcpyAsync(g_data->d_qdd, qdd.typed_data(), "
                 "batch * nj * sizeof(T), cudaMemcpyDeviceToDevice, stream);")
    if "f_ext" in bufs:
        # Native buffer contract (audit W03, 2026-09-19): the copy below is sized by
        # the STATE batch, so the force operand must physically be (batch, 6*NB) —
        # a broadcast-shaped (1, 6*NB) buffer would be read 6*NB*(batch-1) elements
        # past its end. The Python surface materializes broadcasts; this is the
        # boundary check for traced programs and direct handler use.
        L.append(f'    GRID_RBD_FFI_VALIDATE_2D(f_ext, "{key}: f_ext", 6 * grid::NUM_BODIES);')
        L.append(f'    if ((int)f_ext.dimensions()[0] != batch) return ffi::Error::InvalidArgument("{key}: f_ext batch must equal the q batch");')
        L.append("    cudaMemcpyAsync(g_data->d_f_ext, f_ext.typed_data(), "
                 "(size_t)batch * 6 * grid::NUM_BODIES * sizeof(T), cudaMemcpyDeviceToDevice, stream);")
    L.append("    constexpr int stride = 3 * grid::NUM_JOINTS;")
    algo = spec.launch_algo or ("GRID_ALGO_" + key.upper())
    ksym = kernel_symbol_for(spec)
    targs = (f"T, grid::launch_cfg<grid::{algo}>::TIER, /*MUJOCO_OUTPUT=*/MUJOCO"
             if templated else "T")
    kargs = ", ".join(kernel_launch_args(spec, "jax"))
    L.append(f"    grid::{ksym}<{targs}><<<")
    L.append(f"        grid_rbd_grid_for(g_ctx, batch), grid_rbd_launch_threads_n<grid::{algo}>(g_ctx, batch), {smem_bytes_call(spec)}, stream>>>(")
    L.append(f"            {kargs});")
    L.append(f'    GRID_RBD_FFI_CHECK_LAUNCH("{ksym}");')
    out_arg = kernel_launch_args(spec, "jax")[0]
    if spec.hoist_out_size:
        L.append(f"    const int out_size = {spec.out_size_expr};")
        L.append(f"    cudaMemcpyAsync(out->typed_data(), {out_arg}, "
                 f"batch * out_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);")
    else:
        L.append(f"    cudaMemcpyAsync(out->typed_data(), {out_arg}, "
                 f"batch * {copy_size} * sizeof(T), cudaMemcpyDeviceToDevice, stream);")
    if "f_ext" in bufs:
        L.append("    cudaMemsetAsync(g_data->d_f_ext, 0, "
                 "(size_t)batch * 6 * grid::NUM_BODIES * sizeof(T), stream);")
    L.append("    return ffi::Error::Success();")
    L.append("}")
    if role is not None:
        names = bufs + ["out"] + list(spec.trailing_runtime_args) + (["gravity"] if spec.takes_gravity else [])
        tmpl = "template <bool MUJOCO>\n" if templated else ""
        call = f"grid_rbd_jax_{key}_body{'<MUJOCO>' if templated else ''}(g_ctx, stream, {', '.join(names)})"
        plain = sig + ["    int64_t ctx_id"]
        L.append(f"{tmpl}static ffi::Error grid_rbd_jax_{key}_impl(")
        L.append(",\n".join(plain) + ")")
        L.append("{")
        L.append("    GRID_RBD_CTX_OR_FFI(ctx_id);")
        L.append(f"    return {call};")
        L.append("}")
        if role == "fwd":
            out_i = sig.index("    ffi::ResultBuffer<GRID_FFI_T> out")
            stamped = sig[:out_i + 1] + ["    ffi::ResultBuffer<ffi::S32> stamp"] + sig[out_i + 1:] + ["    int64_t ctx_id"]
            L.append(f"// B2 `_stamped` twin (the custom_vjp / autograd forward): value + int32 version stamp.")
            L.append(f"{tmpl}static ffi::Error grid_rbd_jax_{key}_stamped_impl(")
            L.append(",\n".join(stamped) + ")")
            L.append("{")
            L.append("    GRID_RBD_CTX_OR_FFI(ctx_id);")
            L.append(f"    ffi::Error e = {call};")
            L.append("    if (e.failure()) return e;")
            L.append("    grid_rbd_stamp_write(g_ctx, stream, stamp->typed_data());")
            L.append("    return ffi::Error::Success();")
            L.append("}")
        else:
            checked = sig[:1] + ["    ffi::Buffer<ffi::S32> stamp"] + sig[1:] + ["    int64_t ctx_id"]
            L.append(f"// B2 `_checked` twin (the custom_vjp / autograd backward): refuses a stale forward stamp.")
            L.append(f"{tmpl}static ffi::Error grid_rbd_jax_{key}_checked_impl(")
            L.append(",\n".join(checked) + ")")
            L.append("{")
            L.append("    GRID_RBD_CTX_OR_FFI(ctx_id);")
            L.append("    GRID_RBD_FFI_STAMP_CHECK(stamp);")
            L.append(f"    return {call};")
            L.append("}")
    return "\n".join(L) + "\n"


def _jax_bind(name: str, impl: str, spec: AbiSpec, stamp: str | None = None) -> str:
    if spec.trailing_runtime_args or stamp is not None:
        # raw registration: int64 attrs / the B2 stamp slot have no BIND_* macro shape
        args = "".join("        .Arg<ffi::Buffer<GRID_FFI_T>>()\n"
                       for _ in jax_buffer_inputs_for(spec))
        if stamp == "arg":
            args = "        .Arg<ffi::Buffer<ffi::S32>>()\n" + args
        attrs = "".join(f'        .Attr<int64_t>("{a}")\n' for a in spec.trailing_runtime_args)
        if spec.takes_dt_it:
            attrs = '        .Attr<T>("dt").Attr<int64_t>("it")\n' + attrs
        if spec.takes_gravity:
            attrs += '        .Attr<T>("gravity")\n'
        return ("XLA_FFI_DEFINE_HANDLER_SYMBOL(\n"
                f"    {name},\n"
                f"    {impl},\n"
                "    ffi::Ffi::Bind()\n"
                "        .Ctx<ffi::PlatformStream<cudaStream_t>>()\n"
                f"{args}"
                "        .Ret<ffi::Buffer<GRID_FFI_T>>()\n"
                + ("        .Ret<ffi::Buffer<ffi::S32>>()\n" if stamp == "ret" else "")
                + attrs
                + '        .Attr<int64_t>("ctx_id")\n'
                ");")
    macro = (f"GRID_RBD_JAX_BIND_{len(jax_buffer_inputs_for(spec))}IN"
             + ("_DT_IT" if spec.takes_dt_it else "")
             + ("_GRAV" if spec.takes_gravity else ""))
    return f"{macro}({name}, {impl});"


def _jax_unit(key: str) -> str:
    spec = ABI_SPECS[key]
    opener, closer = _surface_gate(spec)
    name = f"grid_rbd_jax_{key}"
    impl = name + "_impl"
    doc = JAX_OP_DOCS.get(key)
    parts = [opener + "\n" + (doc if doc else "") + emit_jax_handler(key)]
    # B2 (K4): the role keys also register a `_stamped` (forward) / `_checked`
    # (gradient) symbol next to the plain one (and `_mujoco_stamped` / `_mujoco_checked`).
    role = ("ret", "_stamped") if key in VJP_FORWARD_KEYS else ("arg", "_checked") if key in VJP_GRAD_KEYS else None

    def binds(suffix: str, targs: str) -> list[str]:
        out = [_jax_bind(name + suffix, impl + targs, spec)]
        if role is not None:
            out.append(_jax_bind(name + suffix + role[1],
                                 f"grid_rbd_jax_{key}{role[1]}_impl" + targs, spec, stamp=role[0]))
        return out

    if spec.has_mjx_twin:
        parts.append("\n" + "\n".join(binds("", "<false>")) + "\n")
        twin = ["\n#ifdef GRID_RBD_WITH_MUJOCO"]
        if key in JAX_TWIN_BIND_DOC_OPS:
            twin.append(f"// MuJoCo-convention {key} (floating only): identical "
                        "plumbing, kernel launched with MUJOCO_OUTPUT=true.")
        twin += binds("_mujoco", "<true>")
        twin.append("#endif  // GRID_RBD_WITH_MUJOCO")
        parts.append("\n".join(twin) + "\n")
    else:
        parts.append("\n" + "\n".join(binds("", "")) + "\n")
    parts.append(closer + "\n")
    return "".join(parts)


def gen_jax_handlers_block() -> str:
    parts = [JAX_HANDLERS_BEGIN,
             "// Regenerate: .venv/bin/python -m grid_codegen.wrapper_body_gen",
             "// Table: grid_codegen/abi_specs.py (kernel_args / jax_buffer_inputs",
             "// et al.); docs verbatim from grid_codegen/wrapper_surface_docs.py.",
             ""]
    assert set(JAX_SURFACE_KEYS) == set(jax_substitution_keys())
    for key in JAX_SURFACE_KEYS:
        if key in _JAX_REGION_BANNERS:
            parts.append(SECTION_BANNERS[_JAX_REGION_BANNERS[key]])
        parts.append(_jax_unit(key))
    parts.append(JAX_HANDLERS_END)
    return "\n".join(parts) + "\n"


# ── torch X-macro op table (region #4) ───────────────────────────────────────
# Row order is the checked-in order; the coverage referee in
# test_abi_spec_crosscheck.py asserts the row SET equals the spec-derived one.
# The four cost rows have no spec rows yet (Wave D) — side table below.
TORCH_TABLE_ORDER: tuple[str, ...] = (
    "inverse_dynamics", "minv", "forward_dynamics", "aba", "crba",
    "end_effector_pose", "end_effector_pose_gradient",
    "end_effector_pose_hessian", "inverse_dynamics_gradient",
    "forward_dynamics_gradient", "idsva_so", "fdsva_so",
    "inverse_dynamics_regressor", "integrator", "integrator_gradient",
    "generalized_gravity", "nonlinear_effects", "coriolis_matrix",
    "kinetic_energy_regressor", "potential_energy_regressor", "energy",
    "com", "ccrba", "cmm_time_variation", "dccrba", "frame_jacobian",
    "frame_jacobian_dot", "osc_inertia", "end_effector_pose_runtime",
    "end_effector_pose_gradient_runtime",
    # Wave D 2026-09-12: the plant cost quartet are REAL spec rows now (the
    # old COST side table dissolved); their table names de-prefix in
    # _torch_table_row (historical torch op naming).
    "plant_quadratic_state_cost",
    "plant_step", "plant_step_gradient",
    "plant_ee_pos_cost", "plant_com_cost", "plant_momentum_cost",
)

_TORCH_TABLE_HEADER = """\
// ── def/impl op table (X-macro) ──────────────────────────────────────────────
// One row per op that has a <bool MUJOCO> impl template AND (on floating
// builds) a _mujoco twin — i.e. every torch op except the pin-only stragglers
// def'd/impl'd by hand after each table expansion below
// (forward_dynamics_parameter_gradient, quadratic_input_cost, the three
// barriers). Each row macro expands to X(name, sig) when its algorithm gate is
// on and to NOTHING otherwise, so the def / impl / mjx-def / mjx-impl blocks
// share ONE gate per op by construction (a schema def'd without its impl
// throws a confusing error at call time; gating the def makes the op absent
// instead, matching the numpy rc=3 / missing-symbol subset pattern — this also
// fixes the formerly UNGATED energy/com/ccrba/cmm/dccrba/frame-family/
// ee-runtime schema defs). The gate FORM is load-bearing: core-algo GRID_HAS_*
// macros are always defined (to 1/0) -> #if; opt-in ones are defined-or-absent
// -> #ifdef / defined().
"""

_TORCH_SCALAR_FMT = {"gravity": "float gravity", "dt": "float dt", "it": "int it",
                     "target_jid": "int target_jid",
                     "reference_frame": "int reference_frame",
                     "offset": "Tensor offset"}


def _torch_table_gate(key: str, spec: AbiSpec) -> str:
    if spec.gate_macro:
        return spec.gate_macro
    if spec.surface_class == "plant":
        return "GRID_PLANT_HAS_" + key.removeprefix("plant_").upper()
    return "GRID_HAS_" + key.upper()


def _torch_schema(spec: AbiSpec) -> str:
    """Torch schema from the C-ABI input row. Uniform rule (reproduces both
    historical 'exceptions'): tensors before the out slot in order (minus the
    flag-fork qdd_opt), then dt/it if takes_dt_it, then gravity, then the
    non-scalar trailing args, then `Tensor? qdd=None` (flag_fork) and
    `Tensor? f_ext=None` (optional f_ext)."""
    names = [n for n, _t in spec.inputs]
    bi = names.index("batch")
    tensors = [n for n in names[:bi - 1]
               if not (spec.qdd_route == "flag_fork" and n == "qdd_opt")]
    post = [n for n in names[bi + 1:] if n not in ("gravity", "f_ext", "dt", "it")]
    args = [f"Tensor {n}" for n in tensors]
    if spec.takes_dt_it:
        args += ["float dt", "int it"]
    if spec.takes_gravity:
        args.append("float gravity")
    args += [_TORCH_SCALAR_FMT[n] for n in post]
    if spec.qdd_route == "flag_fork":
        args.append("Tensor? qdd=None")
    if spec.f_ext_mode == "optional":
        args.append("Tensor? f_ext=None")
    args.append("int ctx_id=0")
    # B2 (K4): the vjp-role ops take an optional int32 stamp (write / check).
    key = next(k for k, s in ABI_SPECS.items() if s is spec)
    if key in VJP_FORWARD_KEYS:
        args.append("Tensor? stamp_out=None")
    elif key in VJP_GRAD_KEYS:
        args.append("Tensor? stamp_expect=None")
    return '"(' + ", ".join(args) + ') -> Tensor"'


def _torch_table_row(key: str) -> str:
    spec = ABI_SPECS[key]
    # Wave D: the plant COST rows (plant_returns) register their torch ops
    # under the historical de-prefixed names; schema = the const T* tensor
    # inputs -> Tensor[] (multi-output out/grad/hess). quadratic_state_cost
    # is the one deliberately ALWAYS-emitted row (no plant gate).
    if spec.plant_returns:
        name = key.removeprefix("plant_")
        macro = "GRID_TORCH_ROW_" + name.upper()
        tensors = [n for n, t in spec.inputs if t == "const T*"]
        schema = '"(' + ", ".join(f"Tensor {n}" for n in tensors) + ', int ctx_id=0) -> Tensor[]"'
        if spec.gate_macro is None:
            return f"#define {macro}(X) X({name}, {schema})  // always emitted\n"
        return (f"#ifdef {spec.gate_macro}\n"
                f"#define {macro}(X) X({name}, {schema})\n"
                f"#else\n#define {macro}(X)\n#endif\n")
    macro = "GRID_TORCH_ROW_" + key.upper()
    gate = _torch_table_gate(key, spec)
    if spec.gate_requires:
        cond = " && ".join(f"defined({m})" for m in (*spec.gate_requires, gate))
        opener = f"#if {cond}"
    elif spec.gate_form == "ifdef" or spec.surface_class == "plant":
        opener = f"#ifdef {gate}"
    else:
        opener = f"#if {gate}"
    return (f"{opener}\n"
            f"#define {macro}(X) X({key}, {_torch_schema(spec)})\n"
            f"#else\n#define {macro}(X)\n#endif\n")


def _torch_table_name(key: str) -> str:
    """Registered torch-op name of a table row: plant COST rows (the ones
    with plant_returns) drop the plant_ prefix (historical naming); the
    plant_step family and every rbd op keep their key."""
    return key.removeprefix("plant_") if ABI_SPECS[key].plant_returns else key


def gen_torch_ops_table() -> str:
    parts = [TORCH_OPS_BEGIN + "\n",
             "// Regenerate: .venv/bin/python -m grid_codegen.wrapper_body_gen\n",
             _TORCH_TABLE_HEADER]
    parts += [_torch_table_row(k) for k in TORCH_TABLE_ORDER]
    parts.append("\n#define GRID_RBD_TORCH_OPS(X) \\\n")
    parts.append(" \\\n".join(
        f"    GRID_TORCH_ROW_{_torch_table_name(k).upper()}(X)"
        for k in TORCH_TABLE_ORDER))
    parts.append("\n" + TORCH_OPS_END + "\n")
    return "".join(parts)


def template_path() -> Path:
    return Path(__file__).resolve().parents[1] / "bindings" / "grid_rbd" / "wrapper_template.cu"


# Every generated region of the template: (begin marker, end marker, generator).
REGIONS = (
    (BEGIN, END, gen_block),
    (CEIL_BEGIN, CEIL_END, gen_ceil_block),
    (MJX_BEGIN, MJX_END, gen_mjx_block),
    (TORCH_OPS_BEGIN, TORCH_OPS_END, gen_torch_ops_table),
    (TORCH_BODIES_BEGIN, TORCH_BODIES_END, gen_torch_bodies_block),
    (JAX_HANDLERS_BEGIN, JAX_HANDLERS_END, gen_jax_handlers_block),
)


def main() -> int:
    check = "--check" in sys.argv
    p = template_path()
    src = p.read_text()
    new = src
    for begin, end, gen in REGIONS:
        if begin not in new or end not in new:
            print(f"markers not found in wrapper_template.cu: {begin[:60]}...",
                  file=sys.stderr)
            return 2
        head, rest = new.split(begin, 1)
        _old, tail = rest.split(end, 1)
        new = head + gen().rstrip("\n") + tail
    if check:
        if new != src:
            print("GENERATED BLOCK DRIFT: rerun python -m grid_codegen.wrapper_body_gen",
                  file=sys.stderr)
            return 1
        print("generated blocks up to date")
        return 0
    p.write_text(new)
    print(f"rewrote generated regions ({len(GENERATED_KEYS)} bodies + "
          f"{len(CEIL_ROWS)} ceiling branches + {len(TORCH_SURFACE_KEYS)} torch ops + "
          f"{len(JAX_SURFACE_KEYS)} jax handlers + {len(TORCH_TABLE_ORDER)}-row op table)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

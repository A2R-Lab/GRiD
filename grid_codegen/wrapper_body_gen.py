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
- mjx twins are NOT generated in this increment (their tier/gate variance is
  D2/D6 in abi_specs.py) — twins stay hand-written below the block.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .abi_specs import ABI_SPECS, AbiSpec

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
    "q_qd_null": "    pack_q_qd_u(q, qd, nullptr, batch, grid::NUM_JOINTS);",
    "q_q_null": "    pack_q_qd_u(q, /*qd=*/q, /*u=*/nullptr, batch, grid::NUM_JOINTS);",
    "q_qd_u": "    pack_q_qd_u(q, qd, u, batch, grid::NUM_JOINTS);",
    "qdd_u_slot": "    pack_q_qd_u(q, qd, qdd, batch, grid::NUM_JOINTS);",
    "pack_q": "    pack_q(q, batch, grid::NUM_JOINTS);",
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
    L = [f'extern "C" int grid_rbd_{stem}({sig}) {{',
         f"#{spec.gate_form} {gate}",
         "    if (!g_data) { int rc = grid_rbd_init(); if (rc) return rc; }",
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
        dims = f"grid_rbd_launch_threads_n<grid::{launch}>(batch)"
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
               f"grid_rbd_launch_threads_n<grid::{launch}>(batch), g_streams);"]
        return out

    if spec.f_ext_mode == "optional":
        L.append("    if (int rc = apply_f_ext(f_ext, batch)) return rc;")
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
        L.append("    reset_f_ext(f_ext, batch);")
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
             "// test/test_wrapper_generated_block.py. Mjx twins stay hand-written.",
             ""]
    for key in GENERATED_KEYS:
        parts.append(gen_body(ABI_SPECS[key]))
    parts.append(END)
    return "\n".join(parts) + "\n"


def template_path() -> Path:
    return Path(__file__).resolve().parents[1] / "bindings" / "grid_rbd" / "wrapper_template.cu"


def main() -> int:
    check = "--check" in sys.argv
    p = template_path()
    src = p.read_text()
    if BEGIN not in src or END not in src:
        print("markers not found in wrapper_template.cu", file=sys.stderr)
        return 2
    head, rest = src.split(BEGIN, 1)
    _old, tail = rest.split(END, 1)
    new = head + gen_block().rstrip("\n") + tail
    if check:
        if new != src:
            print("GENERATED BLOCK DRIFT: rerun python -m grid_codegen.wrapper_body_gen",
                  file=sys.stderr)
            return 1
        print("generated block up to date")
        return 0
    p.write_text(new)
    print(f"rewrote generated block ({len(GENERATED_KEYS)} bodies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    "nonlinear_effects", "generalized_gravity", "coriolis_matrix",
    "energy", "com", "ccrba", "dccrba", "cmm_time_variation",
    "kinetic_energy_regressor", "potential_energy_regressor",
    "frame_jacobian", "frame_jacobian_dot", "osc_inertia",
)

BEGIN = "// ── BEGIN GENERATED C-ABI BODIES (grid_codegen/wrapper_body_gen.py — do not hand-edit) ──"
END = "// ── END GENERATED C-ABI BODIES ──"

# Load-bearing per-body comments preserved from the hand-written originals.
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
         "    if (batch > kMaxBatch) return 2;",
         _PACK[spec.pack_mode]]
    if spec.key in BODY_COMMENTS:
        L.append(BODY_COMMENTS[spec.key])
    grav = "gravity, " if spec.takes_gravity else ""
    trail = "".join(", " + a for a in spec.trailing_runtime_args)
    dims = f"grid_rbd_launch_threads_n<grid::{launch}>(batch)"
    if spec.clamp_kernel:
        dims = f"grid_clamp_threads_for({spec.clamp_kernel}, {dims})"
    L.append(f"    {sym}<T>(g_data, g_robot, {grav}batch, "
             f"dim3((unsigned)batch, 1, 1), {dims}, g_streams{trail});")
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

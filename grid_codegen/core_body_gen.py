"""Generate the pybind method bodies of bindings/src/_core.cpp from ABI_SPECS.

C4 arc, slice 2 (2026-09-08). Mirrors grid_codegen/wrapper_body_gen.py: the
same AbiSpec table that drives the wrapper's generated C-ABI regions also
drives the numpy Runner's pybind methods — one row, every surface. The
generated region covers the 30 spec-backed value/gradient/SO methods, their
mjx twins, and the twins' has_* accessors (61 methods + 29 accessors). The
hand-written remainder of _core.cpp: metadata/overlay surface, plant family
(helper-folded), tool_fext, the check_*/f_ext_ptr helpers, load_so, pybind
defs (slice 3).

Regenerate:  .venv/bin/python -m grid_codegen.core_body_gen
Check drift: .venv/bin/python -m grid_codegen.core_body_gen --check
(the CI gate lives in test/test_core_generated_block.py).

Emission contract (shared input-stride semantics, documented once here and in
the region banner): q/qd/qdd/u are (batch, NUM_JOINTS) — for a FLOATING base
the 6-dof base velocity occupies the leading slots and the +1 quaternion
offset is a padded slot, so qd/qdd stay nq-wide on this surface. Value vector
outputs are likewise nq-wide; matrix/Jacobian outputs are tangent-space
(nv-dimensioned).
"""
from __future__ import annotations

import sys
from pathlib import Path

from .abi_specs import ABI_SPECS

CORE_PATH = Path(__file__).resolve().parents[1] / "bindings" / "src" / "_core.cpp"

BEGIN = "    // ── BEGIN GENERATED PYBIND METHOD BODIES (grid_codegen/core_body_gen.py — do not hand-edit) ──"
END = "    // ── END GENERATED PYBIND METHOD BODIES ──"

# Rows with a bespoke hand-written python surface (kept out of the region).
EXCLUDE = frozenset({"f_ext_contact"})  # -> the tool_fext method (custom staging)

# Runner fn-pointer members (transcribed from the load block; a rename there
# shows up as a compile error here, which is the drift gate we want).
FN_MEMBER = {
    "crba": "fn_crba_", "inverse_dynamics": "fn_inverse_dynamics_",
    "integrator": "fn_integrator_", "minv": "fn_minv_",
    "forward_dynamics": "fn_fd_", "aba": "fn_aba_",
    "inverse_dynamics_gradient": "fn_inverse_dynamics_gradient_",
    "forward_dynamics_gradient": "fn_fd_grad_",
    "idsva_so": "fn_idsva_so_", "fdsva_so": "fn_fdsva_so_",
    "inverse_dynamics_regressor": "fn_id_regressor_",
    "integrator_gradient": "fn_integrator_grad_",
    "kinetic_energy_regressor": "fn_kinetic_energy_regressor_",
    "potential_energy_regressor": "fn_potential_energy_regressor_",
    "energy": "fn_energy_", "end_effector_pose": "fn_ee_pose_",
    "end_effector_pose_gradient": "fn_ee_pose_grad_",
    "end_effector_pose_hessian": "fn_ee_pose_hessian_",
    "fk_batched": "fn_fk_batched_", "frame_jacobian": "fn_frame_jacobian_",
    "frame_jacobian_dot": "fn_frame_jacobian_dot_",
    "osc_inertia": "fn_osc_inertia_",
    "generalized_gravity": "fn_generalized_gravity_",
    "nonlinear_effects": "fn_nonlinear_effects_",
    "coriolis_matrix": "fn_coriolis_matrix_", "com": "fn_com_",
    "ccrba": "fn_ccrba_", "dccrba": "fn_dccrba_",
    "cmm_time_variation": "fn_cmm_time_variation_",
    "end_effector_pose_runtime": "fn_ee_pose_runtime_",
    "end_effector_pose_gradient_runtime": "fn_ee_pose_grad_runtime_",
}

# Twin-side rc==3 messages. The integrator twins map it to an integrator-type
# message; the opt-in runtime-EE twins keep the "not generated" phrasing (their
# family is opt-in codegen, so rc=3 is reachable in the twin too). Every other
# twin has no reachable rc==3 (the twin symbol only exists when the algo was
# generated — the C0 dead-branch audit).
MJX_RC3 = {
    "integrator": "integrator_mujoco: unsupported integrator_type for this build",
    "integrator_gradient": "integrator_gradient_mujoco: only EULER/SI-EULER supported",
    "end_effector_pose_runtime":
        "end_effector_pose_runtime_mujoco not generated for this robot .so",
    "end_effector_pose_gradient_runtime":
        "end_effector_pose_gradient_runtime_mujoco not generated for this robot .so",
}


def _cxx_str(text: str, indent: str) -> str:
    """A C++ string literal (split across lines like the hand code when long)."""
    esc = text.replace("\\", "\\\\").replace('"', '\\"')
    if len(esc) <= 70:
        return f'"{esc}"'
    words = esc.split(" ")
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w) if cur else w
        if len(cand) > 66 and cur:
            lines.append(cur + " ")
            cur = w
        else:
            cur = cand
    lines.append(cur)
    return ('\n' + indent).join(f'"{ln}"' for ln in lines)


def _signature_and_call(spec, mjx: bool):
    """Derive (pybind params, prelude lines, C-ABI call args) from spec.inputs."""
    params, prelude, call = [], [], []
    has_qd = any(n == "qd" for n, _ in spec.inputs)
    so_sized = spec.py_out_dims and "second_order_tensor_size" in spec.py_out_dims
    name = spec.key + ("_mujoco" if mjx else "")
    for n, _t in spec.inputs:
        if n in ("q", "qd", "u"):
            params.append(f"arr_t {n}")
            call.append(f"{n}.data()")
            if n == "u":
                prelude.append('check_array_2d(u, batch, num_joints_, "u");')
        elif n in ("qdd_opt", "qdd"):  # nullable C-ABI qdd -> py::object surface
            if mjx and spec.mjx_requires_qdd:
                params.append("arr_t qdd")
                prelude.append('check_array_2d(qdd, batch, num_joints_, "qdd");')
                call.append("qdd.data()")
            else:
                params.append("py::object qdd_opt")
                prelude += [
                    "const CT* qdd_ptr = nullptr;",
                    "if (!qdd_opt.is_none()) {",
                    "    auto qdd = qdd_opt.cast<arr_t>();",
                    '    check_array_2d(qdd, batch, num_joints_, "qdd");',
                    "    qdd_ptr = qdd.data();",
                    "}",
                ]
                call.append("qdd_ptr")
        elif n.endswith("_out") or n == "out":
            call.append("out.mutable_data()")
        elif n == "batch":
            call.append("batch")
        elif n == "gravity":
            if so_sized:
                params.append("int second_order_tensor_size")
                so_sized = False  # emit once, right before gravity (hand order)
            params.append("CT gravity")
            call.append("gravity")
        elif n == "dt":
            params.append("CT dt, int it")
            call.append("gravity, dt, it")  # dt row always follows gravity in the C ABI
        elif n == "it":
            continue  # folded into the dt entry above
        elif n == "f_ext":
            params.append("py::object f_ext_opt")
            prelude += ["arr_t fe_hold;",
                        "const CT* fe_ptr = f_ext_ptr(f_ext_opt, fe_hold, batch);"]
            call.append("fe_ptr")
        elif n == "target_jid":
            params.append("int target_jid")
            call.append("target_jid")
        elif n == "reference_frame":
            params.append("int reference_frame")
            call.append("reference_frame")
        elif n == "offset":
            params.append("arr_t offset")
            prelude += [
                "const CT* off_ptr = nullptr;",
                "if (offset.size() == 16) off_ptr = offset.data();",
                f'else if (offset.size() != 0) throw std::invalid_argument("{name}: offset must be length-16 (4x4 col-major) or empty");',
            ]
            call.append("off_ptr")
        elif n == "use_warp":
            params.append("bool use_warp")
            call.append("use_warp ? 1 : 0")
        else:
            raise ValueError(f"{spec.key}: unhandled C-ABI input {n!r}")
    # dt/gravity ordering: the C ABI is (..., gravity, dt, it) but the pybind
    # signature exposes (..., dt, it, gravity) — the joint call entry above
    # already covers the C order; drop the separate gravity call entry and move
    # the gravity PARAM after dt/it.
    if any(c == "gravity, dt, it" for c in call):
        call = [c for c in call if c != "gravity"]
        params.remove("CT gravity")
        params.insert(params.index("CT dt, int it") + 1, "CT gravity")
    return params, prelude, call


def gen_method(spec, mjx: bool) -> str:
    key = spec.key
    name = key + ("_mujoco" if mjx else "")
    fn = FN_MEMBER[key] + "mujoco_" if mjx else FN_MEMBER[key]
    params, prelude, call = _signature_and_call(spec, mjx)
    has_qd = any(n == "qd" for n, _ in spec.inputs)
    dims = ", ".join(spec.py_out_dims)
    L = []
    sig = ", ".join(params)
    L.append(f"    // {name}({', '.join(p.split()[-1].rstrip(',') for p in params)}) -> (batch, {dims})")
    L.append(f"    py::array_t<CT> {name}({sig})")
    L.append("    {")
    if mjx:
        guard = spec.py_twin_guard or f"{name} unavailable: floating-base .so only"
        L.append(f"        if (!{fn}) throw std::runtime_error(")
        L.append(f"            {_cxx_str(guard, '            ')});")
    if has_qd:
        L.append("        int batch = check_inputs_2d(q, qd, num_joints_);")
    else:
        L.append(f'        int batch = check_q(q, "{name}");')
    for p in prelude:
        L.append("        " + p)
    L.append(f"        py::array_t<CT> out({{batch, {dims}}});")
    L.append(f"        int rc = {fn}({', '.join(call)});")
    rc3 = MJX_RC3.get(key) if mjx else spec.py_rc3_msg
    if rc3:
        L.append("        if (rc == 3) throw std::runtime_error(")
        L.append(f"            {_cxx_str(rc3, '            ')});")
    stem = (spec.abi_stem or key) + ("_mujoco" if mjx else "")
    L.append(f'        if (rc != 0) throw std::runtime_error("grid_rbd_{stem} failed: rc=" + std::to_string(rc));')
    L.append("        return out;")
    L.append("    }")
    return "\n".join(L)


def gen_accessor(spec) -> str:
    fn = FN_MEMBER[spec.key] + "mujoco_"
    return (f"    bool has_{spec.key}_mujoco() const {{ return {fn} != nullptr; }}")


def generated_keys():
    return [k for k in ABI_SPECS if k not in EXCLUDE and ABI_SPECS[k].py_out_dims]


def gen_region() -> str:
    L = [BEGIN,
         "    // Regenerate: .venv/bin/python -m grid_codegen.core_body_gen",
         "    // Table: grid_codegen/abi_specs.py (ABI_SPECS: inputs/py_out_dims/",
         "    // py_rc3_msg/py_twin_guard); drift-gated by test/test_core_generated_block.py.",
         "    //",
         "    // Shared input contract: q/qd/qdd/u are (batch, NUM_JOINTS) float32,",
         "    // C-contiguous. For a FLOATING base the 6-dof base velocity lives in the",
         "    // leading slots and the +1 quaternion offset is a padded slot (qd/qdd stay",
         "    // nq-wide, NOT nv-wide, on this surface). Value vector outputs (c, qdd) are",
         "    // likewise nq-wide; matrix/Jacobian outputs are tangent-space (nv-sized).",
         ""]
    for key in generated_keys():
        spec = ABI_SPECS[key]
        L.append(gen_method(spec, mjx=False))
        L.append("")
        if spec.has_mjx_twin:
            L.append(gen_accessor(spec))
            L.append(gen_method(spec, mjx=True))
            L.append("")
    L.append(END)
    return "\n".join(L)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    text = CORE_PATH.read_text()
    if BEGIN not in text or END not in text:
        print("core_body_gen: no generated-region markers in _core.cpp yet — "
              "printing the region to stdout (offline-proof mode)")
        print(gen_region())
        return 0
    start = text.index(BEGIN)
    end = text.index(END) + len(END)
    new = text[:start] + gen_region() + text[end:]
    if "--check" in argv:
        if new != text:
            print("core_body_gen --check: DRIFT — regenerate with "
                  ".venv/bin/python -m grid_codegen.core_body_gen")
            return 1
        print("core_body_gen --check: clean")
        return 0
    CORE_PATH.write_text(new)
    print(f"core_body_gen: region rewritten ({len(gen_region().splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

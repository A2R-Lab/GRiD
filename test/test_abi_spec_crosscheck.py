"""Cross-check ABI_SPECS (grid_codegen/abi_specs.py) against wrapper_template.cu.

P0 of the table-driven wrapper collapse transcribes each hand-written C-ABI
body into declarative AbiSpec fields. Nothing consumes the rows at emission
time yet, so THIS test is the contract: every field that names something in
the wrapper (function, gate, param, buffer, size expression, launch enum) is
validated against the actual template text. A transcription typo fails here
today instead of surfacing as a miscompiled body when P1 starts generating.

CPU-only; string-level checks by design (tolerant of formatting, strict on
identifiers and expressions).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from grid_codegen.abi_specs import ABI_SPECS, AbiSpec

_WRAPPER = Path(__file__).resolve().parents[1] / "bindings" / "grid_rbd" / "wrapper_template.cu"
_SRC = _WRAPPER.read_text()

# Infrastructure C-ABI functions that are NOT per-algorithm bodies. Any other
# grid_rbd_* extern "C" without an ABI_SPECS row fails the coverage test, so a
# new algorithm surface cannot land untranscribed.
_INFRA = {
    "init", "close", "num_joints", "num_vel", "num_ees", "num_bodies",
    "max_batch", "max_perf_level_threads", "threads_per_block",
    "set_threads_per_block", "algo_count", "set_threads_for",
    "set_threads_for_n", "get_batch_switch", "kernel_max_threads",
    "set_inertia_params", "get_inertia_params", "has_runtime_inertia",
    "set_transform_params", "get_transform_params", "has_runtime_transform",
    "set_joint_dynamics_params", "get_joint_dynamics_params",
    "has_runtime_joint_dynamics", "attach_tool", "detach_tool", "tool_info",
    # plant surface: PlantBuffers-based, out of AbiSpec scope until the
    # descriptor-table refactor reaches the plant layer
    "plant_alloc", "plant_free", "plant_quadratic_cost", "plant_step",
    "plant_step_gradient", "plant_step_hessian", "plant_ee_cost",
    "plant_com_cost", "plant_momentum_cost", "plant_quadratic_state_cost",
    # device-pool (slab) framework-allocator integration
    "device_pool_bytes", "set_device_pool", "device_pool_used",
}


def _stem(spec: AbiSpec) -> str:
    return spec.abi_stem or spec.key


def _fn_def(stem: str) -> tuple[int, str]:
    """(start index, full signature text) of extern "C" grid_rbd_<stem>."""
    m = re.search(r'extern "C" int grid_rbd_' + re.escape(stem) + r"\(", _SRC)
    assert m, f"no extern C grid_rbd_{stem}( in wrapper_template.cu"
    close = _SRC.index(")", m.end())
    # tolerate multi-line signatures: scan to the matching close paren
    depth, i = 1, m.end()
    while depth:
        c = _SRC[i]
        depth += c == "("
        depth -= c == ")"
        i += 1
    return m.start(), _SRC[m.end():i - 1]


def _body(stem: str) -> str:
    start, _ = _fn_def(stem)
    nxt = _SRC.find('extern "C"', start + 1)
    return _SRC[start:nxt if nxt != -1 else len(_SRC)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


_SPEC_IDS = sorted(ABI_SPECS)


def _fn_def_prefixed(stem: str, prefix: str) -> tuple[int, str]:
    """(start index, signature text) of extern "C" <prefix><stem> (the plant
    section uses the grid_plant_ prefix; everything else grid_rbd_)."""
    m = re.search(r'extern "C" int ' + re.escape(prefix + stem) + r"\(", _SRC)
    assert m, f"no extern C {prefix}{stem}( in wrapper_template.cu"
    depth, i = 1, m.end()
    while depth:
        c = _SRC[i]
        depth += c == "("
        depth -= c == ")"
        i += 1
    return m.start(), _SRC[m.end():i - 1]


@pytest.mark.parametrize("key", _SPEC_IDS)
def test_function_and_twin_exist(key):
    spec = ABI_SPECS[key]
    stem = _stem(spec)
    if spec.surface_class == "plant":
        # hand-written PlantBuffers body under the grid_plant_ prefix
        _fn_def_prefixed(stem[len("plant_"):], "grid_plant_")
        has_twin = f"grid_{stem}_mujoco(" in _SRC
    elif spec.surface_class == "ffi_only":
        # no C-ABI body: the jax FFI handler is the ground truth
        assert f"grid_rbd_jax_{stem}_impl(" in _SRC, (
            f"{key}: ffi_only row but no jax FFI handler in wrapper")
        has_twin = f"grid_rbd_jax_{stem}_mujoco" in _SRC
    elif spec.surface_class == "kernel_only":
        # no binding surface at all — the kernel ceiling entry is the anchor
        assert f'strcmp(algo, "{stem}")' in _SRC, (
            f"{key}: kernel_only row but no kernel_max_threads ceiling entry")
        has_twin = False
    else:
        _fn_def(stem)
        has_twin = f"grid_rbd_{stem}_mujoco(" in _SRC
    assert has_twin == spec.has_mjx_twin, (
        f"{key}: has_mjx_twin={spec.has_mjx_twin} but twin "
        f"{'exists' if has_twin else 'missing'} in wrapper")


@pytest.mark.parametrize("key", _SPEC_IDS)
def test_signature_params(key):
    spec = ABI_SPECS[key]
    if spec.surface_class in ("ffi_only", "kernel_only"):
        assert spec.inputs == (), f"{key}: {spec.surface_class} rows carry no C params"
        return
    if spec.surface_class == "plant":
        _, sig = _fn_def_prefixed(_stem(spec)[len("plant_"):], "grid_plant_")
    else:
        _, sig = _fn_def(_stem(spec))
    names = [p.strip().split()[-1].lstrip("*") for p in sig.split(",") if p.strip()]
    want = [n for (n, _t) in spec.inputs]
    assert names == want, f"{key}: params {names} != spec.inputs {want}"


@pytest.mark.parametrize("key", _SPEC_IDS)
def test_gate(key):
    spec = ABI_SPECS[key]
    if spec.surface_class != "cabi":
        return  # plant/ffi/kernel rows: no generated gate topology to check
    start, _ = _fn_def(_stem(spec))
    macro = spec.gate_macro or ("GRID_HAS_" + spec.key.upper())
    # Convention A (most bodies): the gate wraps the body INSIDE the function
    # (`{ #if GRID_HAS_X ... #else stub ... #endif }`) — look there first.
    body = _body(_stem(spec))
    found = None
    for line in body.splitlines():
        s = line.strip()
        if s.startswith(("#if ", "#ifdef ", "#ifndef ")) and macro in s:
            found = s
            break
    if found is None:
        # Convention B (whole-function wrap, e.g. tool_fext): scan BACKWARDS
        # above the def with #endif balancing so closed inner pairs can't
        # shadow the enclosing gate.
        depth = 0
        for line in reversed(_SRC[:start].splitlines()):
            s = line.strip()
            if s.startswith("#endif"):
                depth += 1
            elif s.startswith(("#if ", "#ifdef ", "#ifndef ")):
                if depth == 0:
                    found = s
                    break
                depth -= 1
    assert found, f"{key}: no gate directive containing {macro} found"
    form = "ifdef" if found.startswith("#ifdef") else "if"
    assert macro in found, f"{key}: gate '{found}' lacks {macro}"
    assert form == spec.gate_form, (
        f"{key}: gate form #{form} != spec.gate_form #{spec.gate_form} ({found})")


@pytest.mark.parametrize("key", _SPEC_IDS)
def test_body_fields(key):
    spec = ABI_SPECS[key]
    if spec.surface_class != "cabi":
        return  # hand-written / FFI-only / surface-less: no generated body
    body = _body(_stem(spec))
    if spec.sig_mjx_macro:
        # IT-dispatch bodies forward to a hand-written launcher that owns the
        # sig fork — the macro then lives in the launcher, not the body.
        where = _SRC if spec.it_dispatch else body
        assert spec.sig_mjx_macro in where, f"{key}: sig_mjx_macro not found"
    if spec.body_override:
        return  # bespoke body: identity checks only
    nb = _norm(body)
    if spec.out_buffer:
        assert spec.out_buffer in body, f"{key}: out_buffer {spec.out_buffer} not in body"
    if spec.out_size_expr:
        assert _norm(spec.out_size_expr) in nb, (
            f"{key}: out_size_expr {spec.out_size_expr!r} not found (normalized)")
    launch = spec.launch_algo or ("GRID_ALGO_" + spec.key.upper())
    needle = _norm(f"grid_rbd_launch_threads_n<grid::{launch}>")
    if spec.it_dispatch:
        # IT-dispatch bodies launch through their template <IntegratorType>
        # launcher above the C-ABI fn — the enum lives there, not in the body.
        assert needle in _norm(_SRC), (
            f"{key}: launch enum {launch} not found anywhere in wrapper")
    else:
        assert needle in nb, f"{key}: launch enum {launch} not found in body"
    if spec.takes_dt_it:
        pnames = [n for n, _ in spec.inputs]
        assert "dt" in pnames and "it" in pnames, f"{key}: takes_dt_it but no dt/it params"
    if spec.pre_launch_check:
        assert "cudaError_t _le" in body or "cudaGetLastError()" in body, (
            f"{key}: pre_launch_check set but no pre-launch consume in body")
    if spec.clamp_kernel:
        assert _norm(spec.clamp_kernel) in nb, f"{key}: clamp_kernel not found"
    if spec.f_ext_mode == "optional":
        assert "apply_f_ext" in body, f"{key}: f_ext_mode=optional but no apply_f_ext"


def test_coverage_no_untranscribed_algo_fns():
    """Every per-algo extern "C" body must have a spec row (or be infra)."""
    fns = set(re.findall(r'extern "C" int grid_rbd_([a-z0-9_]+)\(', _SRC))
    fns = {f for f in fns if not f.endswith("_mujoco")}
    covered = {_stem(s) for s in ABI_SPECS.values()} | _INFRA
    missing = sorted(fns - covered)
    assert not missing, f"untranscribed C-ABI functions: {missing}"


def test_specs_join_registry():
    """Every "cabi" spec key must be a registry key (fk_batched is the known
    extra); non-cabi rows (plant_step family) may sit outside the registry —
    the registry describes grid.cuh kernels, not the PlantBuffers layer."""
    from grid_codegen.algo_registry import ALGO_DESCRIPTORS
    reg = {d.key for d in ALGO_DESCRIPTORS}
    cabi = {k for k, s in ABI_SPECS.items() if s.surface_class == "cabi"}
    extras = sorted(cabi - reg - {"fk_batched"})
    assert not extras, f"spec keys not in registry: {extras}"


def test_sig_mjx_macros_bidirectional():
    """REVERSE direction (H6): every GRID_RBD_SIG_MJX_* the template consumes
    must be carried by exactly one spec row — a macro used in C but absent
    from the table is how the _compile.py fns dict and ABI_SPECS silently
    disagreed about the integrator until 2026-09-06."""
    used = set(re.findall(r"GRID_RBD_SIG_MJX_[A-Z_0-9]+", _SRC))
    carried = {s.sig_mjx_macro for s in ABI_SPECS.values() if s.sig_mjx_macro}
    missing = sorted(used - carried)
    assert not missing, f"template uses sig macros with no spec row: {missing}"
    orphaned = sorted(carried - used)
    assert not orphaned, f"spec rows carry unused sig macros: {orphaned}"


def test_mjx_rejects_f_ext_invariant():
    """The mjx kernel twins do not reframe external wrenches (2026-09-09 layout
    audit): every f_ext-taking row WITH a twin must carry mjx_rejects_f_ext
    (the jax/torch surfaces refuse f_ext under the active mjx convention via
    BaseDelegateMixin._refuse_mjx_f_ext) — and ONLY those rows. If a future
    twin learns to reframe, flip its row and delete it from this derivation."""
    for key, spec in ABI_SPECS.items():
        expect = spec.f_ext_mode == "optional" and spec.has_mjx_twin
        assert spec.mjx_rejects_f_ext == expect, (
            f"{key}: mjx_rejects_f_ext={spec.mjx_rejects_f_ext} but "
            f"f_ext_mode={spec.f_ext_mode!r}, has_mjx_twin={spec.has_mjx_twin}")

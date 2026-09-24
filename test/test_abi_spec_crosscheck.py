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
    # NOTE (Wave D 2026-09-12): the former plant_* entries here were dead
    # weight — every plant body is grid_plant_-prefixed (or a *_mujoco twin),
    # neither of which this grid_rbd_ coverage scan matches. The plant layer
    # now has real spec rows; its referees are the plant-specific tests below.
    # device-pool (slab) framework-allocator integration
    "device_pool_bytes", "set_device_pool", "device_pool_used",
    "ctx_create", "ctx_close", "ctx_default_id", "ctx_profile", "ctx_count",  # W04-B B1 runtime contexts
    "ctx_version",  # W04-B B2 model version
    "graph_begin", "graph_end",  # codex R5 replay admission
    # Multi-contact f_ext (2026-09-17): `grid_rbd_contact_fext` + `grid_rbd_num_contact_frames`
    # are HAND-WRITTEN in wrapper_template.cu under GRID_HAS_CONTACT_FRAMES (they call the
    # baked grid::f_ext_body_device and mirror tool_fext's d_f_ext handling); they are a
    # host helper surface, not a generated per-algo body, so they carry no ABI_SPECS row.
    # Converting them to a spec row (so wrapper_body_gen owns the body) is the tracked
    # follow-up in the canonical plan's register; until then this is the recorded exemption.
    "contact_fext", "num_contact_frames",
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
        # hand-written PlantBuffers body under the grid_plant_ prefix; the
        # cost twins are grid_rbd_<name>_mujoco (mjx_twin_symbol override)
        _fn_def_prefixed(stem[len("plant_"):], "grid_plant_")
        twin_sym = spec.mjx_twin_symbol or f"grid_{stem}_mujoco"
        has_twin = f"{twin_sym}(" in _SRC
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
    elif spec.surface_class == "python_only":
        # pure python-surface construct (the *_wrt_params sysID linearizations):
        # nothing C-side to anchor; the vjp validity test is its referee.
        assert spec.vjp is not None, f"{key}: python_only row carries no vjp"
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
    if spec.surface_class == "ffi_only":
        # A1 2026-09-11: the row spells the inputs the C-ABI body WOULD take
        # so the surface emitters derive uniformly; validate the tensor args
        # against the actual jax handler buffers + torch op tensor params.
        from grid_codegen.abi_specs import jax_buffer_inputs_for, torch_tensor_args
        assert spec.inputs, f"{key}: ffi_only row now carries emitter inputs"
        stem = _stem(spec)
        m = re.search(r"static ffi::Error grid_rbd_jax_" + re.escape(stem)
                      + r"_impl\(", _SRC)
        assert m, f"{key}: no jax handler"
        depth, i = 1, m.end()
        while depth:
            depth += _SRC[i] == "("
            depth -= _SRC[i] == ")"
            i += 1
        bufs = tuple(re.findall(r"ffi::Buffer<GRID_FFI_T>\s+(\w+)", _SRC[m.end():i - 1]))
        assert bufs == jax_buffer_inputs_for(spec), (
            f"{key}: jax buffers {bufs} != derived {jax_buffer_inputs_for(spec)}")
        tm = re.search(r"torch::Tensor torch_" + re.escape(stem) + r"\(([^)]*)\)", _SRC)
        assert tm, f"{key}: no torch op"
        tensors = tuple(re.findall(r"torch::Tensor\s+(\w+)", tm.group(1)))
        assert tensors == torch_tensor_args(spec), (
            f"{key}: torch tensors {tensors} != derived {torch_tensor_args(spec)}")
        return
    if spec.surface_class in ("kernel_only", "python_only"):
        assert spec.inputs == (), f"{key}: {spec.surface_class} rows carry no C params"
        return
    if spec.surface_class == "plant":
        _, sig = _fn_def_prefixed(_stem(spec)[len("plant_"):], "grid_plant_")
    else:
        _, sig = _fn_def(_stem(spec))
    names = [p.strip().split()[-1].lstrip("*") for p in sig.split(",") if p.strip()]
    # W04-B B1 (2026-09-24): every C-ABI compute entry point names its runtime
    # context first (`long long ctx_id`); the per-op inputs the spec spells follow.
    assert names and names[0] == "ctx_id", f"{key}: C-ABI entry point must take the context id first, got {names}"
    want = [n for (n, _t) in spec.inputs]
    assert names[1:] == want, f"{key}: params {names[1:]} != spec.inputs {want}"


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


def test_vjp_recipes_valid():
    """A4-1: every VJP recipe must reference real spec rows whose shaped
    outputs the driver can contract — grad_op/param_grad_op rows carry an
    out_layout, wrt names come from the residual set (plus the minv-backed u),
    and the differentiable-op set is exactly the seven approved recipes."""
    vjps = {k: s.vjp for k, s in ABI_SPECS.items() if s.vjp is not None}
    assert sorted(vjps) == [
        "aba", "end_effector_pose", "forward_dynamics",
        "forward_dynamics_wrt_params", "integrator", "inverse_dynamics",
        "inverse_dynamics_wrt_params"]
    for key, v in vjps.items():
        for op in (v.grad_op, v.param_grad_op):
            if op is None:
                continue
            assert op in ABI_SPECS, f"{key}: vjp op {op!r} is not a spec row"
            assert ABI_SPECS[op].out_layout is not None, (
                f"{key}: vjp op {op!r} has no out_layout to shape by")
        assert set(v.wrt) <= set(v.residuals), f"{key}: wrt not in residuals"
        if v.u_via_minv:
            assert "u" in v.residuals, f"{key}: u_via_minv without a saved u"
            assert "minv" in ABI_SPECS
        assert not (set(v.nondiff) & set(v.wrt)), f"{key}: nondiff ∩ wrt"


# ── A1 surface-emitter field referees (2026-09-11) ──────────────────────────
# The kernel_args / kernel_symbol / smem / jax-buffer fields drive the torch
# op-body and jax FFI-handler emitters (wrapper_body_gen REGIONS #4-6). Until
# those regions land, these checks validate every field against the
# HAND-WRITTEN surface bodies — the same transcribe-then-consume pattern as
# the C-ABI half above. After the regions land they keep running against the
# generated text (double coverage with the byte-gate, both cheap).

from grid_codegen.abi_specs import (  # noqa: E402
    jax_buffer_inputs_for, jax_substitution_keys,
    kernel_launch_args, kernel_symbol_for, smem_bytes_call,
    torch_substitution_keys, torch_tensor_args,
)

_TORCH_SUB = torch_substitution_keys()
_JAX_SUB = jax_substitution_keys()
# Surface bodies that stay hand-written (bespoke qdd forks / IT dispatch /
# frame fork / offset staging) — the emitters keep them literal.
_TORCH_BESPOKE = {
    "inverse_dynamics", "inverse_dynamics_gradient", "integrator",
    "integrator_gradient", "idsva_so", "end_effector_pose_runtime",
    "end_effector_pose_gradient_runtime",
}
_JAX_BESPOKE = {
    "idsva_so", "integrator", "integrator_gradient",
    "end_effector_pose_runtime", "end_effector_pose_gradient_runtime",
}


def _surface_body(name_re: str, key: str) -> str:
    m = re.search(name_re, _SRC)
    assert m, f"{key}: no {name_re} in wrapper_template.cu"
    end = re.compile(r"^\}$", re.M).search(_SRC, m.start())
    return _SRC[m.start():end.end()]


def _torch_body(key):
    return _surface_body(r"torch::Tensor torch_" + re.escape(key) + r"\(", key)


def _jax_body(key):
    # W04-B B2: the vjp-role handlers split into `_body` (the launch, context
    # passed in) + the plain / `_stamped` / `_checked` entry shims.
    name = r"static ffi::Error grid_rbd_jax_" + re.escape(key)
    if re.search(name + r"_body\(", _SRC):
        return _surface_body(name + r"_body\(", key)
    return _surface_body(name + r"_impl\(", key)


def _launch_args(body: str, key: str) -> list[str]:
    i = body.find(">>>(")
    assert i != -1, f"{key}: no kernel launch in body"
    depth, j = 1, i + 4
    while depth:
        depth += body[j] == "("
        depth -= body[j] == ")"
        j += 1
    return [a.strip() for a in body[i + 4:j - 1].split(",")]


def _jax_norm(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)    # /*arg=*/ annotations
    s = re.sub(r"\bstride\w*\b", "stride", s)      # local stride_* names vary
    s = s.replace("static_cast<T>(gravity)", "gravity").replace("(T)gravity", "gravity")
    return re.sub(r"\s+", "", s)


@pytest.mark.parametrize("key", sorted(_TORCH_SUB))
def test_kernel_args_torch(key):
    spec = ABI_SPECS[key]
    body = _torch_body(key)
    got = [re.sub(r"\s+", "", a) for a in _launch_args(body, key)]
    want = [re.sub(r"\s+", "", a) for a in kernel_launch_args(spec, "torch")]
    assert got == want, f"{key}: torch launch args {got} != spec {want}"
    ksym = kernel_symbol_for(spec)
    assert f"grid::{ksym}<" in body, f"{key}: kernel symbol {ksym} not launched"
    assert re.sub(r"\s+", "", smem_bytes_call(spec)) in re.sub(r"\s+", "", body), (
        f"{key}: smem call {smem_bytes_call(spec)} not in torch body")


@pytest.mark.parametrize("key", sorted(_JAX_SUB))
def test_kernel_args_jax(key):
    spec = ABI_SPECS[key]
    body = _jax_body(key)
    got = [_jax_norm(a) for a in _launch_args(body, key)]
    want = [_jax_norm(a) for a in kernel_launch_args(spec, "jax")]
    assert got == want, f"{key}: jax launch args {got} != spec {want}"
    ksym = kernel_symbol_for(spec)
    assert f"grid::{ksym}<" in body, f"{key}: kernel symbol {ksym} not launched"
    assert _jax_norm(smem_bytes_call(spec)) in _jax_norm(body), (
        f"{key}: smem call {smem_bytes_call(spec)} not in jax handler")


@pytest.mark.parametrize("key", sorted(_JAX_SUB))
def test_jax_buffer_inputs(key):
    spec = ABI_SPECS[key]
    body = _jax_body(key)
    sig = body[:body.index(")\n{") if ")\n{" in body else body.index("{")]
    bufs = tuple(re.findall(r"ffi::Buffer<GRID_FFI_T>\s+(\w+)", sig))
    assert bufs == jax_buffer_inputs_for(spec), (
        f"{key}: handler buffers {bufs} != jax_buffer_inputs {jax_buffer_inputs_for(spec)}")


def test_mujoco_twins_declare_their_context():
    """codex R1 (2026-09-24): every generated `_mujoco` C-ABI twin takes the same
    leading `long long ctx_id` as its primary (the guard inside names it; 30 twins
    once compiled only because fixed-base smokes #ifdef them out)."""
    twins = re.findall(r'extern "C" int grid_rbd_([a-z0-9_]+_mujoco)\(([^)]*)\)', _SRC)
    assert len(twins) >= 30, f"only {len(twins)} mujoco twins found"
    bad = [name for name, params in twins if not params.strip().startswith("long long ctx_id")]
    assert not bad, f"mujoco twins without a leading ctx_id: {bad}"


def test_substitution_partitions():
    """The substitution sets derive from kernel_args[_jax] presence and must
    partition the surface inventory: every torch_<x>/jax impl body that is
    NOT substitution-emitted is on the known-bespoke list (or plant/infra)."""
    assert len(_TORCH_SUB) == 24 and len(_JAX_SUB) == 26
    assert set(_JAX_SUB) - set(_TORCH_SUB) == {
        "inverse_dynamics", "inverse_dynamics_gradient"}
    torch_fns = set(re.findall(r"torch::Tensor torch_([a-z0-9_]+)\(", _SRC))
    torch_fns -= {k for k in torch_fns if k.startswith("plant_")}
    unaccounted = torch_fns - set(_TORCH_SUB) - _TORCH_BESPOKE
    assert not unaccounted, f"torch bodies neither specced nor bespoke: {unaccounted}"
    jax_fns = set(re.findall(r"grid_rbd_jax_([a-z0-9_]+)_impl\(", _SRC))
    # B2: `<key>_stamped` / `<key>_checked` shims are twins of a specced key.
    jax_fns = {re.sub(r"_(stamped|checked)$", "", f) for f in jax_fns
               if not f.startswith("plant_") and not f.endswith("_mujoco")}
    unaccounted = jax_fns - set(_JAX_SUB) - _JAX_BESPOKE
    assert not unaccounted, f"jax handlers neither specced nor bespoke: {unaccounted}"


def test_err_prefixes_canonical():
    """User decision 2026-09-11: error prefixes are the FULL op key — the
    abbreviated fd/fd_grad spellings are gone and every substitution torch
    body checks its tensors under its own key."""
    assert '"fd: ' not in _SRC and '"fd_grad: ' not in _SRC
    for key in _TORCH_SUB:
        body = _torch_body(key)
        for msg in re.findall(r'grid_torch_check\(\w+, "([^"]+)"', body):
            assert msg.startswith(key + ": "), (
                f"{key}: non-canonical check message {msg!r}")
    for key in _JAX_SUB:
        body = _jax_body(key)
        for msg in re.findall(r'GRID_RBD_FFI_VALIDATE_2D\(\w+, "([^"]+)"', body):
            assert msg.startswith(key + ": "), (
                f"{key}: non-canonical validate message {msg!r}")


def test_hoist_out_size():
    for key, spec in ABI_SPECS.items():
        if spec.kernel_args is None and spec.jax_kernel_args is None:
            continue
        want = spec.hoist_out_size
        if key in _TORCH_SUB:
            assert ("const int out_size" in _torch_body(key)) == want, (
                f"{key}: hoist_out_size={want} mismatch in torch body")
        assert ("const int out_size" in _jax_body(key)) == want, (
            f"{key}: hoist_out_size={want} mismatch in jax handler")


def test_no_default_tier_smem_on_tier_tuned_launches():
    """A3 (finding #2): wherever a kernel is LAUNCHED at a launch_cfg tier
    (its template args carry launch_cfg<...>::TIER), the dynamic-smem
    argument must not be a default-tier (<T>()) instantiation of a
    tier-AWARE bytes macro — that requests the DEFAULT tier's byte count
    under a TIER-tuned thread count (the idsva_so launch-failure class).
    Sweeps every launch in the template, bespoke bodies included. Sites that
    use a bytes macro merely to SIZE scratch for an untuned default-tier
    kernel (tool_fext, the plant momentum_cost CCRBA proxy) launch WITHOUT
    launch_cfg tier template args and are correctly out of scope."""
    from grid_codegen.algo_registry import ALGO_DESCRIPTORS
    aware_stems = {d.bytes_macro_stem or (d.key.upper() + "_DYNAMIC_SHARED_MEM_BYTES")
                   for d in ALGO_DESCRIPTORS if not d.tier_blind_bytes}
    bad = []
    for m in re.finditer(r"grid::(\w+)<([^;]*?)><<<(.*?)>>>", _SRC, re.S):
        kern, targs, cfg = m.groups()
        if "launch_cfg<" not in targs or "::TIER" not in targs:
            continue
        for sm in re.findall(r"(\w+_DYNAMIC_SHARED_MEM_BYTES)<T>\(\)", cfg):
            if sm in aware_stems:
                bad.append((kern, sm))
    assert not bad, ("tier-tuned launches with default-tier smem bytes "
                     f"(pass the launch tier to the macro): {bad}")


def test_torch_ops_table_coverage():
    """X-macro row coverage: the GRID_RBD_TORCH_OPS table rows must be exactly
    the torch-surface ops with a <bool MUJOCO> impl/_mujoco twin. Wave D: the
    plant cost rows are real specs now; their TABLE names drop the plant_
    prefix (historical torch op naming — plant_returns rows de-prefix, the
    plant_step family keeps its prefix), so quadratic_input_cost/barriers
    (no twin) correctly stay hand-registered outside the table."""
    rows = {m.lower() for m in re.findall(r"#define GRID_TORCH_ROW_([A-Z0-9_]+)\(X\)", _SRC)}
    want = {(k.removeprefix("plant_") if s.plant_returns else k)
            for k, s in ABI_SPECS.items()
            if s.has_mjx_twin and (s.py_surfaces is None or "torch" in s.py_surfaces)}
    assert rows == want, (f"op-table rows != expected: extra={sorted(rows - want)} "
                          f"missing={sorted(want - rows)}")


def test_gate_requires():
    """Composite row gates: each gate_requires macro must appear AND-ed in the
    #if line right above the op-table row macro definition."""
    for key, spec in ABI_SPECS.items():
        if not spec.gate_requires:
            continue
        macro = f"#define GRID_TORCH_ROW_{key.upper()}(X)"
        i = _SRC.index(macro)
        opener = _SRC[:i].rstrip().rsplit("\n", 1)[-1]
        assert opener.startswith("#if "), f"{key}: row gate is not an #if ({opener!r})"
        for req in (*spec.gate_requires, spec.gate_macro or "GRID_HAS_" + key.upper()):
            assert f"defined({req})" in opener, (
                f"{key}: gate_requires macro {req} not in row gate {opener!r}")


# ── Wave D plant referees (2026-09-12) ──────────────────────────────────────

def test_plant_gate_coverage():
    """Every GRID_PLANT_HAS_* macro the template consumes must be carried by
    exactly one plant spec row's gate_macro — these gates were UNCHECKED
    before Wave D (a renamed/added plant gate could silently orphan its op).
    """
    used = set(re.findall(r"GRID_PLANT_HAS_[A-Z_0-9]+", _SRC))
    carried = {s.gate_macro for s in ABI_SPECS.values()
               if s.surface_class == "plant" and s.gate_macro}
    missing = sorted(used - carried)
    assert not missing, f"plant gates with no spec row: {missing}"
    orphaned = sorted(carried - used)
    assert not orphaned, f"spec rows carry unused plant gates: {orphaned}"


@pytest.mark.parametrize("key", sorted(
    k for k, s in ABI_SPECS.items() if s.plant_returns))
def test_plant_returns_match_out_params(key):
    """plant_returns buffer names must be exactly the C out-params (the T*
    params after the last const input, before batch), in order."""
    spec = ABI_SPECS[key]
    outs = [n for (n, t) in spec.inputs if t == "T*"]
    want = [b for b, _dims in spec.plant_returns]
    assert outs == want, f"{key}: out params {outs} != plant_returns {want}"
    for _b, dims in spec.plant_returns:
        for d in dims:
            assert d in ("1", "3", "6", "nq", "nv", "nx"), (
                f"{key}: unknown plant_returns dim token {d!r}")


def test_plant_twin_symbols_resolve():
    """Each mjx_twin_symbol must name a real extern C fn; rows WITHOUT a twin
    must not have one under either naming convention."""
    for key, spec in ABI_SPECS.items():
        if spec.surface_class != "plant":
            continue
        stem = spec.abi_stem or spec.key
        if spec.has_mjx_twin:
            sym = spec.mjx_twin_symbol or f"grid_{stem}_mujoco"
            assert f'extern "C" int {sym}(' in _SRC, (
                f"{key}: twin symbol {sym} not found")
        else:
            for sym in (f"grid_{stem}_mujoco",
                        f"grid_rbd_{stem.removeprefix('plant_')}_mujoco"):
                assert f'extern "C" int {sym}(' not in _SRC, (
                    f"{key}: has_mjx_twin=False but {sym} exists")

"""Kernel-attribute manifests (the irregular per-kernel overload SIGNATURE
payload) + init/close emission (init_grid_kernel_attrs, per-algo split-compile
attr fns, init_grid/close_grid/streams). H4 move from GRiDCodeGenerator.py
(2026-08-27, verbatim). The names are class-bound on GRiDCodeGenerator (tests
read GRiDCodeGenerator.KERNEL_ATTR_MANIFEST etc.)."""
from .algo_registry import ALGO_DESCRIPTORS, descriptor_for
from .launch_config import baked_launch_cfg


# ── B2: divergent-tier registration support ──────────────────────────────────
# cudaFuncSetAttribute is PER TEMPLATE INSTANTIATION. The manifest below spells
# kernels at the DEFAULT tier (`<T>` / mjx `<T, GRID_DEFAULT_RESOURCE_TIER,
# true>`), but the bindings/host launchers instantiate at
# launch_cfg<GRID_ALGO_*>::TIER — a DIFFERENT kernel whenever the baked tier
# diverges from the default. That instantiation was silently unregistered: a
# >48KB divergent-tier kernel failed at launch (cudaErrorInvalidValue) while
# its default-tier sibling worked. Fix: for every algo whose BAKED entry
# (same load path as the launch_cfg<> specializations) carries a non-default
# tier, ALSO register the divergent-tier instantiation, sized by the
# tier-correct bytes. Robots with no divergent bake emit byte-identical text.
#
# These smem macros are `template <typename T>` only — their algos have ONE
# arena, so smem is tier-INVARIANT and the `<T>()` spelling stays exact for
# any tier (audited 2026-09-05; the kernels themselves are still
# tier-templated — the tier only moves launch_bounds/registers there).
# Derived from the descriptor table's tier_blind_bytes field (C3 swap
# 2026-09-08, assert-equal-verified against the audited literal set).
from .algo_registry import ALGO_DESCRIPTORS as _ALGO_DESCRIPTORS

TIER_BLIND_BYTES_MACROS = frozenset(
    d.bytes_macro for d in _ALGO_DESCRIPTORS if d.tier_blind_bytes)


def _tier_variant_kernel(kernel_name: str, tier_sym: str) -> str:
    """The divergent-tier spelling of a manifest kernel name. mjx names carry
    an explicit GRID_DEFAULT_RESOURCE_TIER token (replaced); pin names rely on
    the defaulted tier param, which is always the NEXT template arg after the
    ones spelled (T, or T + IntegratorType — audited: every manifest kernel
    takes RESOURCE_TIER immediately after T / after IT)."""
    if "GRID_DEFAULT_RESOURCE_TIER" in kernel_name:
        return kernel_name.replace("GRID_DEFAULT_RESOURCE_TIER", tier_sym)
    assert kernel_name.endswith(">"), kernel_name
    return kernel_name[:-1] + ", " + tier_sym + ">"


def _tier_variant_bytes(bytes_macro: str, tier_sym: str) -> str:
    """The tier-correct smem request for the divergent-tier registration."""
    if bytes_macro in TIER_BLIND_BYTES_MACROS:
        return bytes_macro
    assert bytes_macro.endswith("<T>()"), bytes_macro
    return bytes_macro[:-3] + ", " + tier_sym + ">()"


def _tier_variant_attr_lines(label, bytes_macro, kernels, tier_sym, alias_start):
    """The extra registration block for one divergent-tier algo entry.
    Returns (lines, next_alias). Mirrors the default-tier block's shape."""
    vb = _tier_variant_bytes(bytes_macro, tier_sym)
    lines = [f"// {label}: baked launch_cfg tier {tier_sym} != default — the launchers",
             "// instantiate THAT kernel, so it needs its own dynamic-smem opt-in.",
             f"if ({vb} <= _grid_smem_max) {{",
             f"    gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"{label}@{tier_sym}\", {vb}));"]
    alias_counter = alias_start
    for kernel_name, signature in kernels:
        alias = f"_grid_kern_alias_{alias_counter}"
        alias_counter += 1
        lines.append(f"    auto {alias} = static_cast<{signature}>(&{_tier_variant_kernel(kernel_name, tier_sym)});")
        lines.append(f"    gpuErrchk(cudaFuncSetAttribute({alias}, cudaFuncAttributeMaxDynamicSharedMemorySize, {vb}));")
    lines.append("}")
    return lines, alias_counter



# Manifest of every algorithm kernel that needs cudaFuncSetAttribute
# (MaxDynamicSharedMemorySize). Each entry:
#   (algo_label, algo_short_name, gate_attr, bytes_macro, [(kernel_name, signature), ...])
#
# algo_short_name matches the keys used in self.generated_algorithms (see
# _normalize_codegen_algorithms). gate_attr is the `self.generate_*` bool —
# honored when present, else we rely on algo_short_name membership.
#
# Applied to EVERY kernel, not just the historically-large ones: without
# cudaFuncSetAttribute(MaxDynamicSharedMemorySize, BYTES), a kernel whose
# runtime dynamic smem exceeds the device default per-block limit (48 KB on
# most consumer NVIDIA GPUs incl. sm_8x) fails to launch silently with
# cudaErrorInvalidConfiguration — the error doesn't propagate through
# cudaDeviceSynchronize() reliably, so timings come back as bogus ~0 us (hit
# on g1 floating where ABA/FD/MINV need 52-57 KB). No-op when BYTES is already
# under the device default.
# Default False: the f_ext A.3 (-dJ^T/dq) kernel is registered only when
# gen_f_ext_gradient actually emitted it (set True whenever f_ext_gradient runs).
_f_ext_gradient_dq_emitted = False

# ── Step 2: per-algo kernel-overload SIGNATURES (the irregular C++ payload).
# Keyed by algo_short, INSERTION ORDER == the historical KERNEL_ATTR_MANIFEST
# order (byte-identity: the emitted cudaFuncSetAttribute block + its alias
# counter follow this order). The per-entry HEAD (algo_label, gate_attr,
# bytes_macro) is NO LONGER restated here — it is derived from the descriptor
# table (algo_registry.ALGO_DESCRIPTORS) below, so gate/bytes live in exactly
# one place. algo_label == algo_short for every pin entry (registry invariant).
KERNEL_OVERLOADS = {
    "inverse_dynamics": [
        ("inverse_dynamics_kernel<T>",
         "void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)"),
        ("inverse_dynamics_kernel<T>",
         "void (*)(T *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
        ("inverse_dynamics_kernel_single_timing<T>",
         "void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)"),
        ("inverse_dynamics_kernel_single_timing<T>",
         "void (*)(T *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
    ],
    "minv": [
        ("minv_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ("minv_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
    ],
    "forward_dynamics": [
        ("forward_dynamics_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
        ("forward_dynamics_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
    ],
    "aba": [
        ("aba_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
        ("aba_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
    ],
    "crba": [
        ("crba_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ("crba_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
    ],
    "end_effector_pose": [
        ("end_effector_pose_kernel<T>",
         "void (*)(T *, const T *, const int, const robotModel<T> *, const int)"),
        ("end_effector_pose_kernel_single_timing<T>",
         "void (*)(T *, const T *, const int, const robotModel<T> *, const int)"),
    ],
    "end_effector_pose_gradient": [
        ("end_effector_pose_gradient_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ("end_effector_pose_gradient_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
    ],
    "inverse_dynamics_gradient": [
        ("inverse_dynamics_gradient_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)"),
        ("inverse_dynamics_gradient_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
        ("inverse_dynamics_gradient_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)"),
        ("inverse_dynamics_gradient_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
    ],
    "forward_dynamics_gradient": [
        ("forward_dynamics_gradient_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const T *, const T *, T *, const robotModel<T> *, const T, const int)"),
        ("forward_dynamics_gradient_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
        ("forward_dynamics_gradient_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const T *, const T *, T *, const robotModel<T> *, const T, const int)"),
        ("forward_dynamics_gradient_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
    ],
    # g1-spill: f_ext_gradient_kernel gained `unsigned char *d_workspace` as its
    # 3rd arg (after the two outputs) so s_dqdd_dfext can spill there at LITE/MINIMAL.
    "f_ext_gradient": [
        ("f_ext_gradient_kernel<T>",
         "void (*)(T *, T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ("f_ext_gradient_kernel_single_timing<T>",
         "void (*)(T *, T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
    ],
    # A.3 (-dJ^T/dq): own kernel + smem macro (both base modes). Gated (descriptor
    # gate_attr=_f_ext_gradient_dq_emitted) so any header lacking the kernel/macro
    # never references them.
    # mimic-spill: f_ext_gradient_dq_kernel gained `unsigned char *d_workspace` as
    # its 2nd arg (after the output) so the MIMIC per-sub slab can spill there at rung 1.
    "f_ext_gradient_dq": [
        ("f_ext_gradient_dq_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ("f_ext_gradient_dq_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
    ],
    # E1 joint-torque regressor: Y is nv x 10*NUM_BODIES, can exceed the 48 KB
    # default dynamic-smem cap on big robots (g1: ~55 KB), so it MUST opt in.
    "inverse_dynamics_regressor": [
        ("inverse_dynamics_regressor_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ("inverse_dynamics_regressor_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
    ],
    # PS5 energy regressors (each output 10*NUM_BODIES; opt-in like the joint-torque
    # regressor so big-robot launches set the dynamic-smem attr).
    "kinetic_energy_regressor": [
        ("kinetic_energy_regressor_kernel<T>",
         "void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ("kinetic_energy_regressor_kernel_single_timing<T>",
         "void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)"),
    ],
    "potential_energy_regressor": [
        ("potential_energy_regressor_kernel<T>",
         "void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ("potential_energy_regressor_kernel_single_timing<T>",
         "void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)"),
    ],
    # FD param gradient dqdd/dpi = -Minv . Y: output is nv x 10*NUM_BODIES (same
    # size class as the regressor), can exceed the 48 KB default cap; opt in.
    # g1-spill: forward_dynamics_parameter_gradient_kernel gained `unsigned char *d_workspace`
    # as its 2nd arg (after d_dqdd_dpi) so s_Y can spill there at LITE/MINIMAL.
    "forward_dynamics_parameter_gradient": [
        ("forward_dynamics_parameter_gradient_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ("forward_dynamics_parameter_gradient_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
    ],
    "idsva_so_body_frame": [
        ("idsva_so_body_frame_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ("idsva_so_body_frame_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
    ],
    # world-frame single-pass alternative (opt-in via enable_idsva_so_world_frame).
    # Now takes d_workspace (unified signature; cold buffers spill there at LITE/MINIMAL).
    # Uses its own shared-mem macro (~25 KB for g1 vs shim's ~162 KB).
    "idsva_so_world_frame": [
        ("idsva_so_world_frame_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ("idsva_so_world_frame_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
    ],
    "fdsva_so": [
        ("fdsva_so_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
        ("fdsva_so_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)"),
    ],
    # Integrator kernels are templated on IntegratorType IT (a non-type param
    # that does not change the function signature). Each IT is a distinct
    # __global__ instantiation, so cudaFuncSetAttribute must run for ALL of
    # them — otherwise a non-Euler IT whose floating-base arena exceeds the
    # 48 KB device default launches with cudaErrorInvalidValue while Euler
    # (the only one historically registered) succeeds.
    "integrator": [
        (f"integrator_kernel{suffix}<T, IntegratorType::{it}>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, T *, const T, const T, const int)")
        for suffix in ("", "_single_timing")
        for it in ("EULER", "SEMI_IMPLICIT_EULER", "MIDPOINT", "RK3", "RK4", "TRAPEZOIDAL")
    ],
"integrator_gradient": [
        (f"integrator_gradient_kernel{suffix}<T, IntegratorType::{it}>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, T *, const T, const T, const int)")
        for suffix in ("", "_single_timing")
        for it in ("EULER", "SEMI_IMPLICIT_EULER", "MIDPOINT", "RK3", "RK4", "TRAPEZOIDAL")
    ],
    "integrator_with_gradient": [
        (f"integrator_with_gradient_kernel{suffix}<T, IntegratorType::{it}>",
         "void (*)(T *, T *, unsigned char *, const T *, const int, const robotModel<T> *, T *, const T, const T, const int)")
        for suffix in ("", "_single_timing")
        for it in ("EULER", "SEMI_IMPLICIT_EULER", "MIDPOINT", "RK3", "RK4", "TRAPEZOIDAL")
    ],
    # ee_pose_hessian is special: only emitted when its shared-mem fits the
    # GRID_CUDA_TARGET_SHARED_MEM_BYTES budget at compile time. The runtime
    # guard wraps the cudaFuncSetAttribute call.
    "end_effector_pose_hessian": [
        ("end_effector_pose_hessian_kernel<T>",
         "void (*)(T *, T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ("end_effector_pose_hessian_kernel_single_timing<T>",
         "void (*)(T *, T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
    ],
    # G2 centroidal quick-wins. R6: each registers on its OWN key (matching
    # gen_centroidal_quickwins' per-key emit) — algo_short keys an entry that
    # is in generated_algorithms exactly when THAT centroidal fn was emitted.
    "generalized_gravity": [
        ("generalized_gravity_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ("generalized_gravity_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
    ],
    "nonlinear_effects": [
        ("nonlinear_effects_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ("nonlinear_effects_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
    ],
    # PS5 Coriolis matrix C(q,qd): nv x nv output can exceed the 48 KB default
    # dynamic-smem cap on big robots, so it MUST opt in. Kernel takes d_workspace
    # as its 2nd arg (reserved for a future big-robot spill; unused at FULL).
    "coriolis_matrix": [
        ("coriolis_matrix_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ("coriolis_matrix_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
    ],
    # PS5 dCCRBA: cmm_time_variation (Adot 6*nv; no spill, no d_workspace arg) and
    # dccrba (6*nv*nv tensor; spill -> d_workspace as the 2nd arg). REQUIRED so
    # init_grid runs cudaFuncSetAttribute (the 6*nv*nv output blows the 48 KB cap).
    "cmm_time_variation": [
        ("cmm_time_variation_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ("cmm_time_variation_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
    ],
    "dccrba": [
        ("dccrba_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ("dccrba_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
    ],
    # E2/S1 general-frame Jacobian family (opt-in; gated on membership in
    # generated_algorithms via algo_short). The kernels take the target frame
    # at RUNTIME: (T *out, const T *q, const int stride_q, const int target_jid,
    # const int reference_frame, robotModel, int N).
    "frame_jacobian": [
        ("frame_jacobian_kernel<T>",
         "void (*)(T *, const T *, const int, const int, const int, const robotModel<T> *, const int)"),
        ("frame_jacobian_kernel_single_timing<T>",
         "void (*)(T *, const T *, const int, const int, const int, const robotModel<T> *, const int)"),
    ],
    "frame_jacobian_dot": [
        ("frame_jacobian_dot_kernel<T>",
         "void (*)(T *, const T *, const int, const int, const int, const robotModel<T> *, const int)"),
        ("frame_jacobian_dot_kernel_single_timing<T>",
         "void (*)(T *, const T *, const int, const int, const int, const robotModel<T> *, const int)"),
    ],
    "osc_inertia": [
        ("osc_inertia_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ("osc_inertia_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
    ],
    # Runtime-target pose / pose-gradient (opt-in). Kernels take target_jid +
    # the runtime offset pointer: (T *out, const T *q, const int stride_q,
    # const int target_jid, const T *offset, robotModel, int N).
    "end_effector_pose_runtime": [
        ("end_effector_pose_runtime_kernel<T>",
         "void (*)(T *, const T *, const int, const int, const T *, const robotModel<T> *, const int)"),
        ("end_effector_pose_runtime_kernel_single_timing<T>",
         "void (*)(T *, const T *, const int, const int, const T *, const robotModel<T> *, const int)"),
    ],
    "end_effector_pose_gradient_runtime": [
        ("end_effector_pose_gradient_runtime_kernel<T>",
         "void (*)(T *, const T *, const int, const int, const T *, const robotModel<T> *, const int)"),
        ("end_effector_pose_gradient_runtime_kernel_single_timing<T>",
         "void (*)(T *, const T *, const int, const int, const T *, const robotModel<T> *, const int)"),
    ],
    # W1b.3 batched multi-target world positions / position-gradient (opt-in via
    # multi_target_batch; gated on _has_multi_target_position). Same launch shape as
    # end_effector_pose: (T *out, const T *q, const int stride_q, robotModel, int N).
    "multi_target_position": [
        ("multi_target_position_kernel<T>",
         "void (*)(T *, const T *, const int, const robotModel<T> *, const int)"),
        ("multi_target_position_kernel_single_timing<T>",
         "void (*)(T *, const T *, const int, const robotModel<T> *, const int)"),
    ],
    "multi_target_position_gradient": [
        ("multi_target_position_gradient_kernel<T>",
         "void (*)(T *, const T *, const int, const robotModel<T> *, const int)"),
        ("multi_target_position_gradient_kernel_single_timing<T>",
         "void (*)(T *, const T *, const int, const robotModel<T> *, const int)"),
    ],
    "com": [
        ("com_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ("com_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
    ],
    "ccrba": [
        ("ccrba_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
        ("ccrba_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)"),
    ],
    "energy": [
        ("energy_kernel<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
        ("energy_kernel_single_timing<T>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)"),
    ],
}

# KERNEL_ATTR_MANIFEST head (algo_label, algo_short, gate_attr, bytes_macro)
# DERIVED from the descriptor rows — single source of truth for gate/bytes. The
# `kernels` payload comes from KERNEL_OVERLOADS. Order follows KERNEL_OVERLOADS
# (== the historical manifest order) so the emitted attr block is byte-identical.
KERNEL_ATTR_MANIFEST = [
    (short, short, descriptor_for(short).gate_attr, descriptor_for(short).bytes_macro, kernels)
    for short, kernels in KERNEL_OVERLOADS.items()
]

# ── Step 2: floating mjx-twin (MUJOCO_OUTPUT=true) kernel-overload SIGNATURES.
# The mjx twin is a SUBSET of the pin kernels with the trailing `, GRID_DEFAULT_
# RESOURCE_TIER, true` template flag; it has NO single_timing twin. Payload only;
# the mujoco_manifest head (label=short+"(mjx)", gate_attr, bytes_macro) is derived
# from the descriptor rows in gen_init_close_grid. Insertion order == historical
# mujoco_manifest order (byte-identity of the mjx portion of the attr block). Some
# sigs differ from the pin base: id / id-grad use the qdd overload; fd-grad uses the
# single-output overload; integrator families fan out over their mjx IntegratorType set.
MJX_KERNEL_OVERLOADS = {
    "inverse_dynamics": [("inverse_dynamics_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)")],
    "minv": [("minv_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)")],
    "forward_dynamics": [("forward_dynamics_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)")],
    "aba": [("aba_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)")],
    "crba": [("crba_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)")],
    "end_effector_pose": [("end_effector_pose_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, const T *, const int, const robotModel<T> *, const int)")],
    # osc_inertia(mjx): Lambda is frame-invariant; quat reorder on input. Needs its
    # >48KB dynamic-smem opt-in (passes on go2 where arena<48KB; fails on big floating).
    "osc_inertia": [("osc_inertia_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)")],
    "end_effector_pose_gradient": [("end_effector_pose_gradient_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)")],
    "end_effector_pose_hessian": [("end_effector_pose_hessian_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, T *, unsigned char *, const T *, const int, const robotModel<T> *, const int)")],
    "inverse_dynamics_gradient": [("inverse_dynamics_gradient_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, unsigned char *, const T *, const int, const T *, T *, const robotModel<T> *, const T, const int)")],
    "forward_dynamics_gradient": [("forward_dynamics_gradient_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)")],
    "inverse_dynamics_regressor": [("inverse_dynamics_regressor_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)")],
    "idsva_so_world_frame": [("idsva_so_world_frame_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, const T, const int)")],
    "fdsva_so": [("fdsva_so_kernel<T, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, unsigned char *, const T *, const int, T *, const robotModel<T> *, const T, const int)")],
    "integrator": [(f"integrator_kernel<T, IntegratorType::{it}, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, T *, const T, const T, const int)")
         for it in ("EULER", "SEMI_IMPLICIT_EULER", "MIDPOINT", "RK3", "RK4")],
    # integrator_gradient mjx is single-stage (EULER / SI-Euler) only.
    "integrator_gradient": [(f"integrator_gradient_kernel<T, IntegratorType::{it}, GRID_DEFAULT_RESOURCE_TIER, true>",
         "void (*)(T *, unsigned char *, const T *, const int, const robotModel<T> *, T *, const T, const T, const int)")
         for it in ("EULER", "SEMI_IMPLICIT_EULER")],
}

def gen_init_close_grid(self):
    # set the max shared mem to account for large robots and allocate streams
    MAX_STREAMS = 3 # max needed in any of our functions
    # ----- init_grid_kernel_attrs<T>(): cudaFuncSetAttribute for every kernel --
    # Split out from init_grid so per-algo TU callers can invoke ONLY the
    # attribute-setting part (without stream allocation), needed because
    # cudaFuncSetAttribute operates on the TU-local host stub. The per-algo
    # TU split (P6-7b) calls this from a static initializer in each
    # measure_X_*_entry so its stubs get the attribute set.
    self.gen_add_func_doc("Set MaxDynamicSharedMemorySize for every algorithm kernel "
                          "(callable from any TU; idempotent). __forceinline__ is "
                          "REQUIRED so the &kernel<T> expressions resolve to the "
                          "CALLING TU's host stubs — otherwise the linker merges this "
                          "function across TUs and we set the attribute on one TU's "
                          "stubs while the launch goes through a different TU's.",
                          [], [], None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__ __forceinline__")
    self.gen_add_code_line("void init_grid_kernel_attrs(){", True)
    attr_lines = ["// enable opt-in dynamic shared memory for every algorithm kernel",
                  "// Gate registration on the DEVICE opt-in max (not the codegen target):",
                  "// grid_check_dynamic_shared_memory_bytes and the bench's",
                  "// grid_kernel_fits_device both use the device cap, so registering only up",
                  "// to the smaller GRID_CUDA_TARGET_SHARED_MEM_BYTES left kernels in",
                  "// (target, device-max] checkable+launchable but UNregistered -> launching",
                  "// them failed with cudaErrorInvalidValue (e.g. the floating idsva_so",
                  "// body-frame diagnostic ~101 KB on g1). Keying on the device max keeps",
                  "// registration, the fit-check, and the launch-skip in lockstep.",
                  "size_t _grid_smem_max = 0; gpuErrchk(grid_get_max_dynamic_shared_memory_bytes(&_grid_smem_max));"]
    generated_set = getattr(self, "generated_algorithms", None)
    alias_counter = 0
    # B2: algos whose BAKED launch_cfg tier diverges from the default get a
    # second registration at that tier (see the module-top note). Same load
    # path as the launch_cfg<> specializations, so the two can never disagree.
    # GRID_DEFAULT_RESOURCE_TIER is TIER_SHARED by policy (_constants_arena).
    divergent_tiers = {sym: e["tier"] for sym, e in baked_launch_cfg(self).items()
                       if e["tier"] != "TIER_SHARED"}
    # mjx (floating MUJOCO_OUTPUT) emits a SECOND __global__ instantiation of every
    # mjx-capable kernel with the trailing MUJOCO_OUTPUT=true template flag. Those are
    # DISTINCT device functions, so cudaFuncSetAttribute must run for them too — else a
    # big mjx kernel (fdsva_so / idsva_so-world / integrator_gradient / id-grad on a
    # humanoid) whose dynamic smem exceeds the 48 KB default launches with
    # cudaErrorInvalidValue while its pin twin succeeds. The signature is identical
    # (MUJOCO_OUTPUT is a non-type param). For the qdd-overloaded kernels (inverse_dynamics
    # / inverse_dynamics_gradient) only the qdd overload carries MUJOCO_OUTPUT, so the
    # qdd-overload signature is used; integrator_gradient mjx is single-stage (EULER/SI) only.
    # Gate on floating_base, NOT self.MUJOCO_OUTPUT: the mjx kernel template flag is
    # emitted for EVERY floating-base robot (the C-ABI/jax/torch mjx wrappers are
    # #ifdef GRID_FLOATING_BASE), independent of the MUJOCO_OUTPUT constructor arg
    # (which the .so build does not set). So the MUJOCO_OUTPUT=true instantiations
    # exist for any floating .so and need their dynamic-smem attribute registered.
    # mujoco_manifest head derived from the descriptor rows (label=short+"(mjx)",
    # gate_attr/bytes_macro from the descriptor — single source of truth); the mjx
    # kernel-overload signatures come from MJX_KERNEL_OVERLOADS. Order follows that
    # dict (== the historical mujoco_manifest order) for a byte-identical attr block.
    mujoco_manifest = []
    # enable_mujoco_kernels=False skips the aggregate's mjx registration -> the
    # address-take that ODR-uses (and therefore instantiates) every mjx twin never
    # happens (trigger 1 of 2; see gen_all_code).
    if getattr(self, "enable_mujoco_kernels", True) and self.robot.floating_base:
        mujoco_manifest = [
            (short + "(mjx)", short, descriptor_for(short).gate_attr,
             descriptor_for(short).bytes_macro, kernels)
            for short, kernels in self.MJX_KERNEL_OVERLOADS.items()
        ]
    for entry in self.KERNEL_ATTR_MANIFEST + mujoco_manifest:
        algo_label, algo_short, gate_attr, bytes_macro, kernels = entry
        # Honor the legacy generate_* gate when present; otherwise fall
        # back to membership in generated_algorithms; if neither is set
        # (legacy callers), assume the algo is generated.
        if gate_attr is not None and not getattr(self, gate_attr, True):
            continue
        if gate_attr is None and generated_set is not None and algo_short not in generated_set:
            continue
        # Spherical robots: the multi-stage RK integrator GRADIENTS
        # static_assert (follow-on slice) — address-taking their kernel
        # instantiations here would trip that assert from init_grid. Drop
        # just the RK-typed gradient rows; every other robot keeps the full
        # set (byte-identical). The VALUE integrator rows stay: spherical
        # supports all five value ITs.
        if (self.robot.robot_has_spherical()
                and algo_short in ("integrator_gradient", "integrator_with_gradient")):
            kernels = [
                (kname, sig) for (kname, sig) in kernels
                if not any(rk in kname for rk in ("MIDPOINT", "RK3", "RK4"))
            ]
        # Wrap EVERY kernel's attribute registration in a compile-time-
        # resolvable size guard so init_grid never hard-aborts when a kernel
        # literally can't fit a device even with cudaFuncSetAttribute (e.g.
        # the floating-base idsva_so_body_frame *diagnostic* frame on h1_2 at
        # ~168 KB, or the integrator value/gradient kernels at ~103-228 KB on
        # big floating-base robots). Such kernels simply go unregistered;
        # they aren't the dispatched production path, and any code that DOES
        # launch them still runs `grid_check_dynamic_shared_memory_bytes` at
        # the host wrapper, so the fit check + attribute setup stay in
        # lockstep at the actual use site. Small kernels are always under the
        # target, so the guard is a no-op for them.
        attr_lines.append(f"if ({bytes_macro} <= _grid_smem_max) {{")
        attr_lines.append(f"    gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"{algo_label}\", {bytes_macro}));")
        for kernel_name, signature in kernels:
            alias = f"_grid_kern_alias_{alias_counter}"
            alias_counter += 1
            attr_lines.append(f"    auto {alias} = static_cast<{signature}>(&{kernel_name});")
            attr_lines.append(f"    gpuErrchk(cudaFuncSetAttribute({alias}, cudaFuncAttributeMaxDynamicSharedMemorySize, {bytes_macro}));")
        attr_lines.append("}")
        if algo_short in divergent_tiers:
            extra, alias_counter = _tier_variant_attr_lines(
                algo_label, bytes_macro, kernels, divergent_tiers[algo_short],
                alias_counter)
            attr_lines.extend(extra)
    self.gen_add_code_lines(attr_lines)
    self.gen_add_end_function()

    # ----- Per-algo init_grid_kernel_attr_<short><T>() (P0: split-compile) ----
    # Each registers ONLY its own kernel's attribute(s) (pin + mjx twin). A TU
    # that calls just one of these ODR-uses only that kernel, so only that
    # kernel instantiates + hits ptxas -- instead of the whole ~35-kernel set
    # the aggregate init_grid_kernel_attrs above forces (the monolith that OOMs
    # big-humanoid compiles). Consumed by the per-algo bench (Fix #1) and the
    # Model X launcher TUs. Additive: the aggregate is unchanged, so existing
    # emission stays byte-identical. Same guard / alias / gate logic as above;
    # per-function local alias counter. __forceinline__ (as on the aggregate) so
    # &kernel<T> binds to the CALLING TU's host stubs.
    pin_by_short = {short: (label, gate_attr, bytes_macro, kernels)
                    for (label, short, gate_attr, bytes_macro, kernels) in self.KERNEL_ATTR_MANIFEST}
    mjx_by_short = {}
    if getattr(self, "enable_mujoco_kernels", True) and self.robot.floating_base:
        for short, kernels in self.MJX_KERNEL_OVERLOADS.items():
            d = descriptor_for(short)
            mjx_by_short[short] = (short + "(mjx)", d.bytes_macro, kernels)
    for short, (label, gate_attr, bytes_macro, kernels) in pin_by_short.items():
        # Mirror the aggregate's gate: skip un-generated algos so the function
        # never references a non-emitted kernel.
        if gate_attr is not None and not getattr(self, gate_attr, True):
            continue
        if gate_attr is None and generated_set is not None and short not in generated_set:
            continue
        # PIN and MJX are emitted as SEPARATE functions on purpose. The
        # MUJOCO_OUTPUT=true twin is a DISTINCT device function, and on the
        # derivative/second-order kernels it is larger than its pin counterpart
        # (idsva_so_world_frame was 28.1x pin raw; block-parallelizing the epilogue
        # brought it to 2.42x, fdsva_so to 1.41x on go2-floating -- agent_debugging
        # _guide 1u -- still the largest kernels). Splitting them lets a
        # consumer that never launches mjx kernels (the benchmark; any pin-only
        # user such as GATO/PDDP) pay NOTHING for them, while the floating-base
        # mjx wrappers still register theirs by calling the _mjx variant.
        def _emit_attr_fn(fn_suffix, doc_what, entries):
            self.gen_add_func_doc("Set MaxDynamicSharedMemorySize for the %s kernel(s) only "
                                  "(callable from any TU; idempotent). Split-compile entry "
                                  "point: registers just this algo so a solo TU instantiates "
                                  "only its kernel." % doc_what, [], [], None)
            self.gen_add_code_line("template <typename T>")
            self.gen_add_code_line("__host__ __forceinline__")
            self.gen_add_code_line("void init_grid_kernel_attr_%s(){" % fn_suffix, True)
            per_lines = ["size_t _grid_smem_max = 0; gpuErrchk(grid_get_max_dynamic_shared_memory_bytes(&_grid_smem_max));"]
            per_alias = 0
            for entry_label, entry_bytes, entry_kernels in entries:
                per_lines.append(f"if ({entry_bytes} <= _grid_smem_max) {{")
                per_lines.append(f"    gpuErrchk(grid_check_dynamic_shared_memory_bytes(\"{entry_label}\", {entry_bytes}));")
                for kernel_name, signature in entry_kernels:
                    alias = f"_grid_kern_alias_{per_alias}"
                    per_alias += 1
                    per_lines.append(f"    auto {alias} = static_cast<{signature}>(&{kernel_name});")
                    per_lines.append(f"    gpuErrchk(cudaFuncSetAttribute({alias}, cudaFuncAttributeMaxDynamicSharedMemorySize, {entry_bytes}));")
                per_lines.append("}")
                if short in divergent_tiers:  # B2: mirror the aggregate
                    extra, per_alias = _tier_variant_attr_lines(
                        entry_label, entry_bytes, entry_kernels,
                        divergent_tiers[short], per_alias)
                    per_lines.extend(extra)
            self.gen_add_code_lines(per_lines)
            self.gen_add_end_function()

        _emit_attr_fn(short, label, [(label, bytes_macro, kernels)])
        if short in mjx_by_short:
            _emit_attr_fn(short + "_mjx", label + "(mjx)", [mjx_by_short[short]])

    # ----- init_grid_streams<T>(): streams only, no attr registration -------
    # The stream-allocation half of init_grid, WITHOUT init_grid_kernel_attrs
    # (so it instantiates zero kernels). A split-compile / per-algo TU pairs
    # this with a single init_grid_kernel_attr_<algo> to avoid pulling in the
    # whole kernel set (P0/P1). Mirrors the init_grid stream block below.
    self.gen_add_func_doc("Allocates streams for host functions WITHOUT registering any kernel "
                          "attributes (pair with an init_grid_kernel_attr_<algo> for split "
                          "compiles).", [], [], "A pointer to the array of streams")
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("cudaStream_t *init_grid_streams(){", True)
    self.gen_add_code_lines(["gpuErrchk(cudaDeviceSynchronize());",
                  "// allocate streams",
                  "cudaStream_t *streams = (cudaStream_t *)malloc(" + str(MAX_STREAMS) + "*sizeof(cudaStream_t));",
                  "int priority, minPriority, maxPriority;",
                  "gpuErrchk(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority));",
                  "for(int i=0; i<" + str(MAX_STREAMS) + "; i++){",
                  "    int adjusted_max = maxPriority - i; priority = adjusted_max > minPriority ? adjusted_max : minPriority;",
                  "    gpuErrchk(cudaStreamCreateWithPriority(&(streams[i]),cudaStreamDefault,priority));  // BLOCKING streams: every generated host wrapper copies inputs on streams[0] and launches kernels on the DEFAULT stream — legacy default-stream sync is the ordering guarantee (nonblocking streams made that copy->launch pair a data race; ps5 fr3 first-call repro 2026-08-11)",
                  "}", "return streams;"])
    self.gen_add_end_function()

    # ----- init_grid<T>(): full init = attrs + streams (the original API) ----
    self.gen_add_func_doc("Sets MaxDynamicSharedMemorySize for every algorithm kernel and initializes streams for host functions", \
                          [], [], "A pointer to the array of streams")
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("cudaStream_t *init_grid(){", True)
    init_lines = ["init_grid_kernel_attrs<T>();",
                  "gpuErrchk(cudaDeviceSynchronize());",
                  "// allocate streams",
                  "cudaStream_t *streams = (cudaStream_t *)malloc(" + str(MAX_STREAMS) + "*sizeof(cudaStream_t));",
                  "int priority, minPriority, maxPriority;",
                  "gpuErrchk(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority));",
                  "for(int i=0; i<" + str(MAX_STREAMS) + "; i++){",
                  "    int adjusted_max = maxPriority - i; priority = adjusted_max > minPriority ? adjusted_max : minPriority;",
                  "    gpuErrchk(cudaStreamCreateWithPriority(&(streams[i]),cudaStreamDefault,priority));  // BLOCKING streams: every generated host wrapper copies inputs on streams[0] and launches kernels on the DEFAULT stream — legacy default-stream sync is the ordering guarantee (nonblocking streams made that copy->launch pair a data race; ps5 fr3 first-call repro 2026-08-11)",
                  "}", "return streams;"]
    self.gen_add_code_lines(init_lines)
    self.gen_add_end_function()
    # free the streams and all allocated data
    self.gen_add_func_doc("Frees the memory used by grid", [], ["streams allocated by init_grid", "robotModel allocated by init_robotModel", "data allocated by init_gridData"], None)
    self.gen_add_code_line("template <typename T, gridDataKind KIND = GRID_DATA_ALL>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("void close_grid(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T, KIND> *hd_data){", True)
    close_lines = (["free_robotModel(d_robotModel); // frees nested d_XImats/d_topology_helpers(+runtime tables)+struct (bare cudaFree would leak the nested arrays)", \
                             "gpuErrchk(cudaFree(hd_data->d_q_qd_u)); gpuErrchk(cudaFree(hd_data->d_q_qd)); gpuErrchk(cudaFree(hd_data->d_q));", \
                             "gpuErrchk(cudaFree(hd_data->d_f_ext)); grid_host_free(hd_data->h_f_ext);", \
                             "gpuErrchk(cudaFree(hd_data->d_c)); gpuErrchk(cudaFree(hd_data->d_Minv)); gpuErrchk(cudaFree(hd_data->d_qdd)); gpuErrchk(cudaFree(hd_data->d_M));", \
                             "gpuErrchk(cudaFree(hd_data->d_dc_du)); gpuErrchk(cudaFree(hd_data->d_df_du));", \
                             "gpuErrchk(cudaFree(hd_data->d_dtau_dfext)); gpuErrchk(cudaFree(hd_data->d_dqdd_dfext));",
                             "grid_host_free(hd_data->h_dtau_dfext); grid_host_free(hd_data->h_dqdd_dfext);",
                             "gpuErrchk(cudaFree(hd_data->d_f_ext_gradient_dq)); grid_host_free(hd_data->h_f_ext_gradient_dq);",
                             # R2: regressor Y + FD param-gradient dqdd/dpi outputs
                             "gpuErrchk(cudaFree(hd_data->d_Y)); gpuErrchk(cudaFree(hd_data->d_dqdd_dpi));",
                             "grid_host_free(hd_data->h_Y); grid_host_free(hd_data->h_dqdd_dpi);",
                             # PS5 energy regressors
                             "gpuErrchk(cudaFree(hd_data->d_ke_regressor)); gpuErrchk(cudaFree(hd_data->d_pe_regressor));",
                             "grid_host_free(hd_data->h_ke_regressor); grid_host_free(hd_data->h_pe_regressor);",
                             # PS5 Coriolis matrix
                             "gpuErrchk(cudaFree(hd_data->d_coriolis)); grid_host_free(hd_data->h_coriolis);",
                             # PS5 dCCRBA
                             "gpuErrchk(cudaFree(hd_data->d_dccrba)); gpuErrchk(cudaFree(hd_data->d_cmm_time_variation));",
                             "grid_host_free(hd_data->h_dccrba); grid_host_free(hd_data->h_cmm_time_variation);"]
                             + [
                             "gpuErrchk(cudaFree(hd_data->d_end_effector_pose)); gpuErrchk(cudaFree(hd_data->d_end_effector_pose_gradient)); gpuErrchk(cudaFree(hd_data->d_end_effector_pose_hessian));", \
                             # Phase 3a/b/c/e: end the L2 persisting window opened at init.
                             "gpuErrchk(grid_end_l2_persisting(0));", \
                             "gpuErrchk(cudaFree(hd_data->d_workspace));", \
                            # idsva_so - d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq
                             "gpuErrchk(cudaFree(hd_data->d_idsva_so));", \
                             # fdsva_so - d2fd_dq2, d2fd_cross, d2fd_dqd2, d2fd_dtaudq
                             "gpuErrchk(cudaFree(hd_data->d_df2));", \
                             "grid_host_free(hd_data->h_idsva_so); grid_host_free(hd_data->h_df2);", \
                             "grid_host_free(hd_data->h_q_qd_u); grid_host_free(hd_data->h_q_qd); grid_host_free(hd_data->h_q);", \
                             "grid_host_free(hd_data->h_c); grid_host_free(hd_data->h_Minv); grid_host_free(hd_data->h_qdd); grid_host_free(hd_data->h_M);", \
                             "grid_host_free(hd_data->h_dc_du); grid_host_free(hd_data->h_df_du);",\
                             "grid_host_free(hd_data->h_end_effector_pose); grid_host_free(hd_data->h_end_effector_pose_gradient); grid_host_free(hd_data->h_end_effector_pose_hessian);", \
                             # E2/S1: general-frame Jacobian outputs (frame_jacobian / frame_jacobian_dot / osc_inertia)
                             "gpuErrchk(cudaFree(hd_data->d_frame_jacobian)); gpuErrchk(cudaFree(hd_data->d_frame_jacobian_dot)); gpuErrchk(cudaFree(hd_data->d_osc_inertia));", \
                             "grid_host_free(hd_data->h_frame_jacobian); grid_host_free(hd_data->h_frame_jacobian_dot); grid_host_free(hd_data->h_osc_inertia);", \
                             "gpuErrchk(cudaFree(hd_data->d_eePose)); gpuErrchk(cudaFree(hd_data->d_eePoseGrad)); gpuErrchk(cudaFree(hd_data->d_eepose_runtime_offset));", \
                             "grid_host_free(hd_data->h_eePose); grid_host_free(hd_data->h_eePoseGrad);"] \
                             # W1b.3 batched multi-target (opt-in; Python-conditional, mirrors the alloc)
                             + ([
                             "gpuErrchk(cudaFree(hd_data->d_multi_target_position)); gpuErrchk(cudaFree(hd_data->d_multi_target_position_gradient));",
                             "grid_host_free(hd_data->h_multi_target_position); grid_host_free(hd_data->h_multi_target_position_gradient);",
                             ] if getattr(self, "_has_multi_target_position", False) else []) \
                             + [
                             "gpuErrchk(cudaFree(hd_data->d_x_kp1)); gpuErrchk(cudaFree(hd_data->d_dAB));", \
                             "grid_host_free(hd_data->h_x_kp1); grid_host_free(hd_data->h_dAB);", \
                             "for(int i=0; i<" + str(MAX_STREAMS) + "; i++){gpuErrchk(cudaStreamDestroy(streams[i]));} free(streams);"])
    # Device-pool mode (2026-09-09): gridData device frees route through
    # grid_device_free (no-op for slab-carved pointers, cudaFree otherwise),
    # and the consumed pool is rewound so a close/re-init cycle re-carves from
    # the top of the caller-owned slab.
    close_lines = [l.replace("gpuErrchk(cudaFree(hd_data->",
                             "gpuErrchk(grid_device_free(hd_data->")
                   for l in close_lines]
    close_lines.insert(len(close_lines) - 1, "grid_device_pool().used = 0;")
    self.gen_add_code_lines(close_lines)
    self.gen_add_end_function()

"""Single source of truth for GRiD algorithm metadata.

ONE verbose canonical name per algorithm, used IDENTICALLY as: registry `key` ==
emitted `grid::` symbol == bench key == algorithm_list token == printf label. There
are NO short aliases or legacy labels here — this is a deliberate clean break.

Each entry binds:
  - `key`: the JSON output key (lowercase) — also the parser's lowercased label match
  - `display`: the verbose label rendered in benchmark reports
  - `section`: the section heading in the report (Core Dynamics / Gradients / ...)

The codegen's printf label is `key.upper()` unless `printf_label` is overridden.
The parser then lowercases the matched line and looks it up against `key`.

Consumers:
  - `test/benchmarks/timing_parser.py` builds its label→key maps from this list.
  - `test/benchmarks/generate_report.py` builds ALGO_DISPLAY and ALGO_SECTIONS.
  - (Future) per-algo `gen_*_host` codegen functions read `printf_label` here.

Adding a new algo touches exactly this file (plus the corresponding codegen
emission of the kernel + the bench's `#if GRID_HAS_X` measure wrappers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AlgoEntry:
    key: str
    display: str
    section: str
    printf_label_override: str | None = None

    @property
    def printf_label(self) -> str:
        return self.printf_label_override or self.key.upper()


ALGO_REGISTRY: tuple[AlgoEntry, ...] = (
    # Core Dynamics
    AlgoEntry("inverse_dynamics",     "Inverse Dynamics (RNEA / Recursive Newton-Euler Algorithm)",
              "Core Dynamics"),
    AlgoEntry("minv",                 "Minv (M⁻¹, computed directly)",
              "Core Dynamics"),
    AlgoEntry("forward_dynamics",     "Forward Dynamics (Minv+RNEA)",
              "Core Dynamics"),
    AlgoEntry("aba",                  "ABA (Articulated Body Algorithm)",
              "Core Dynamics"),
    AlgoEntry("crba",                 "CRBA",
              "Core Dynamics"),

    # Gradients
    AlgoEntry("inverse_dynamics_gradient", "Inverse Dynamics Gradient (∂ID/∂q,v)", "Gradients"),
    AlgoEntry("forward_dynamics_gradient", "Forward Dynamics Gradient (∂FD/∂q,v)", "Gradients"),
    AlgoEntry("f_ext_gradient",       "F_EXT_GRAD (∂tau/∂fext=-Jᵀ, ∂q̈/∂fext=M⁻¹Jᵀ)",
              "Gradients"),
    AlgoEntry("f_ext_gradient_dq",    "F_EXT_GRADIENT_DQ (∂(inverse_dynamics_gradient)/∂fext=-∂Jᵀ/∂q, fixed base)",
              "Gradients"),
    AlgoEntry("inverse_dynamics_regressor", "Inverse Dynamics Regressor (Joint-torque Y; tau=Y·π, ∂tau/∂π)",
              "Gradients"),
    AlgoEntry("forward_dynamics_parameter_gradient", "Forward Dynamics Parameter Gradient (∂q̈/∂π = -M⁻¹·Y)",
              "Gradients"),
    AlgoEntry("kinetic_energy_regressor", "Kinetic Energy Regressor (KE = y_KE·π, length 10·NB)",
              "Gradients"),
    AlgoEntry("potential_energy_regressor", "Potential Energy Regressor (PE = y_PE·π, length 10·NB)",
              "Gradients"),

    # Integrators
    AlgoEntry("integrator",           "Integrator (x_{k+1})",              "Integrators"),
    AlgoEntry("integrator_gradient",  "Integrator_Gradient (∂x_{k+1}/∂x,u)",
              "Integrators"),
    AlgoEntry("integrator_with_gradient",
              "Integrator_With_Gradient (x_{k+1} + ∂x_{k+1}/∂x,u)",
              "Integrators"),
    AlgoEntry("integrator_hessian", "Integrator_Hessian (∂²x_{k+1}/∂z², z=[q,qd,u]; plant_step_hessian s_d2AB)",
              "Integrators"),

    # Kinematics
    AlgoEntry("end_effector_pose",              "END_EFFECTOR_POSE",                 "Kinematics"),
    AlgoEntry("end_effector_pose_gradient",     "END_EFFECTOR_POSE_GRADIENT (Jacobian)", "Kinematics"),
    AlgoEntry("end_effector_pose_hessian",      "END_EFFECTOR_POSE_HESSIAN (2nd-order EE Jacobian)", "Kinematics"),
    AlgoEntry("frame_jacobian",       "FRAME_JACOBIAN (general-frame J: LOCAL/WORLD/LWA)", "Kinematics"),
    AlgoEntry("frame_jacobian_dot",   "FRAME_JACOBIAN_DOT (time derivative Jdot of the general-frame J)", "Kinematics"),
    AlgoEntry("osc_inertia",          "OSC_INERTIA (operational-space inertia Lambda = (J Minv J^T)^-1)", "Kinematics"),
    AlgoEntry("end_effector_pose_runtime",          "END_EFFECTOR_POSE_RUNTIME (runtime target/offset pose [xyz;rpy])", "Kinematics"),
    AlgoEntry("end_effector_pose_gradient_runtime", "END_EFFECTOR_POSE_GRADIENT_RUNTIME (runtime target/offset pose Jacobian)", "Kinematics"),
    AlgoEntry("multi_target_position",          "MULTI_TARGET_POSITION (batched world positions of baked fixed-offset targets)", "Kinematics"),
    AlgoEntry("multi_target_position_gradient", "MULTI_TARGET_POSITION_GRADIENT (∂world pos/∂v of baked targets)", "Kinematics"),

    # Second-Order
    AlgoEntry("idsva_so",             "IDSVA_SO (dispatched: body for fixed, world for floating)",
              "Second-Order"),
    AlgoEntry("idsva_so_body_frame",  "IDSVA_SO_BODY_FRAME (2nd-order ID, body-frame)",
              "Second-Order"),
    AlgoEntry("idsva_so_world_frame", "IDSVA_SO_WORLD_FRAME (2nd-order ID, world-frame)",
              "Second-Order"),
    AlgoEntry("fdsva_so",             "FDSVA_SO (2nd-order FD)",
              "Second-Order"),

    # Centroidal / Energy / CoM (G2 quick-wins, R1-R3)
    AlgoEntry("generalized_gravity", "Generalized Gravity g(q)=RNEA(q,0,0)",
              "Centroidal"),
    AlgoEntry("nonlinear_effects",   "Nonlinear Effects c(q,qd)=RNEA(q,qd,0)",
              "Centroidal"),
    AlgoEntry("energy",              "Energy (KE/PE/mechanical)",
              "Centroidal"),
    AlgoEntry("com",                 "CoM + CoM Jacobian",
              "Centroidal"),
    AlgoEntry("ccrba",               "CCRBA (A, h)",
              "Centroidal"),
    AlgoEntry("coriolis_matrix",     "Coriolis Matrix C(q,q̇)",
              "Centroidal"),
    AlgoEntry("dccrba",              "dCCRBA (∂A/∂q tensor, 6×NV×NV)",
              "Centroidal"),
    AlgoEntry("cmm_time_variation",  "CMM Time Variation (Ȧ, 6×NV)",
              "Centroidal"),

    # Plant (T6): cost / constraint / plant-step primitives emitted in the
    # sibling `grid_plant` namespace. No standalone benchmarked kernel — this
    # entry exists so the family has a registry key (display / sectioning) and
    # so future per-primitive bench wrappers can reference it.
    AlgoEntry("plant",                "Plant (cost/constraint/step primitives)",
              "Plant"),
    # Collision (W3): baked sphere data + config_free emitted in the sibling
    # `grid_collision` namespace (composed over grid::multi_target_position + the
    # static SDF geometry header). No standalone benchmarked kernel — registry key
    # only, like `plant`.
    AlgoEntry("collision",            "Collision (config_free / spherized-URDF)",
              "Collision"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-algo DESCRIPTOR table (item M, Step 0). One row per algorithm capturing the
# launch-config + kernel-attribute METADATA that is otherwise scattered across
# GRiDCodeGenerator.py. This step generates NOTHING — it is the parity safety net
# (test/test_algo_descriptor_parity.py asserts the table reproduces the live
# LAUNCH_CONFIG_ALGO_TO_SYMBOL dict and the KERNEL_ATTR_MANIFEST metadata exactly).
# Later steps (per docs/open-tasks/design_descriptor_table_spec.md) extend the
# schema with the arena/spill closures and DRIVE those sites from these rows.
#
# Only the IRREGULAR fields are stored per row; everything regular is derived:
#   - bytes_macro defaults to "<KEY.upper()>_DYNAMIC_SHARED_MEM_BYTES" — overridden
#     for the 4 algos that share a sibling's macro (integrator_gradient /
#     integrator_with_gradient -> INTEGRATOR_DU; generalized_gravity /
#     nonlinear_effects -> INVERSE_DYNAMICS_BIAS).
#   - gate_attr defaults to None (registration falls back to membership in
#     generated_algorithms) — overridden for the 7 opt-in / conditionally-emitted
#     kernels that carry an explicit `generate_*` / `_*_emitted` flag.
#   - has_kernel_attr defaults True — False for the 3 registry keys with no own
#     cudaFuncSetAttribute entry: `idsva_so` (a dispatch alias for the body/world
#     kernels), `integrator_hessian` (no standalone benchmarked kernel yet), and
#     `plant` (cost/constraint primitives, no __global__).
#   - autotune_keys defaults to () — set for the 17 algos that carry a baked
#     launch_cfg<> (the bench-abbreviated JSON key(s) that map to this grid symbol).


@dataclass(frozen=True)
class AlgoDescriptor:
    key: str                              # == AlgoEntry.key (the join)
    autotune_keys: tuple[str, ...] = ()   # bench JSON keys -> LAUNCH_CONFIG_ALGO_TO_SYMBOL
    has_kernel_attr: bool = True          # has its own KERNEL_ATTR_MANIFEST entry
    gate_attr: str | None = None          # explicit generate_*/_*_emitted gate, else None
    bytes_macro_stem: str | None = None   # override the default *_DYNAMIC_SHARED_MEM_BYTES stem

    @property
    def carries_launch_cfg(self) -> bool:
        return bool(self.autotune_keys)

    @property
    def enum_name(self) -> str:
        return "GRID_ALGO_" + self.key.upper()

    @property
    def bytes_macro(self) -> str:
        """Full `NAME<T>()` string as it appears in KERNEL_ATTR_MANIFEST."""
        stem = self.bytes_macro_stem or (self.key.upper() + "_DYNAMIC_SHARED_MEM_BYTES")
        return stem + "<T>()"


ALGO_DESCRIPTORS: tuple[AlgoDescriptor, ...] = (
    # Core Dynamics
    AlgoDescriptor("inverse_dynamics", autotune_keys=("id",)),
    AlgoDescriptor("minv", autotune_keys=("minv",)),
    AlgoDescriptor("forward_dynamics", autotune_keys=("fd",)),
    AlgoDescriptor("aba", autotune_keys=("aba",)),
    AlgoDescriptor("crba", autotune_keys=("crba",)),

    # Gradients
    AlgoDescriptor("inverse_dynamics_gradient", autotune_keys=("id_du",),
                   gate_attr="generate_inverse_dynamics_gradient"),
    AlgoDescriptor("forward_dynamics_gradient", autotune_keys=("fd_du",),
                   gate_attr="generate_forward_dynamics_gradient"),
    AlgoDescriptor("f_ext_gradient"),
    AlgoDescriptor("f_ext_gradient_dq", gate_attr="_f_ext_gradient_dq_emitted"),
    AlgoDescriptor("inverse_dynamics_regressor"),
    AlgoDescriptor("forward_dynamics_parameter_gradient"),
    AlgoDescriptor("kinetic_energy_regressor"),
    AlgoDescriptor("potential_energy_regressor"),

    # Kinematics
    AlgoDescriptor("end_effector_pose", autotune_keys=("ee_pose",)),
    AlgoDescriptor("end_effector_pose_gradient", autotune_keys=("ee_pose_gradient",)),
    AlgoDescriptor("end_effector_pose_hessian", autotune_keys=("ee_pose_hessian",),
                   gate_attr="generate_end_effector_pose_hessian"),
    AlgoDescriptor("frame_jacobian"),
    AlgoDescriptor("frame_jacobian_dot"),
    AlgoDescriptor("osc_inertia"),
    AlgoDescriptor("end_effector_pose_runtime"),
    AlgoDescriptor("end_effector_pose_gradient_runtime"),
    # W1b.3 batched multi-target (opt-in via multi_target_batch). Real benchmarked
    # kernels (has_kernel_attr=True) gated on the generator's _has_multi_target_position
    # flag. bytes_macro stems default from KEY.upper() (MULTI_TARGET_POSITION[_GRADIENT]
    # _DYNAMIC_SHARED_MEM_BYTES) — no override needed. No autotune_keys (no baked launch_cfg).
    AlgoDescriptor("multi_target_position", gate_attr="_has_multi_target_position"),
    AlgoDescriptor("multi_target_position_gradient", gate_attr="_has_multi_target_position"),

    # Second-Order
    AlgoDescriptor("idsva_so", autotune_keys=("idsva_so",), has_kernel_attr=False),
    AlgoDescriptor("idsva_so_body_frame", autotune_keys=("idsva_so_body_frame",),
                   gate_attr="generate_idsva_so_body_frame"),
    AlgoDescriptor("idsva_so_world_frame", autotune_keys=("idsva_so_world_frame",),
                   gate_attr="generate_idsva_so_world_frame"),
    AlgoDescriptor("fdsva_so", autotune_keys=("fdsva_so",), gate_attr="generate_fdsva_so"),

    # Integrators — kept AFTER Second-Order so `[d for d in ALGO_DESCRIPTORS if
    # d.carries_launch_cfg]` reproduces the launch-config enum order (integrator
    # group LAST), which Step 1 relies on for a byte-identical grid.cuh. (The
    # ALGO_REGISTRY order above is the report/section order and is unaffected;
    # the descriptor test asserts key-equality as a set/dict, order-independent.)
    AlgoDescriptor("integrator", autotune_keys=("integrator",)),
    AlgoDescriptor("integrator_gradient", autotune_keys=("integrator_gradient",),
                   bytes_macro_stem="INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES"),
    AlgoDescriptor("integrator_with_gradient", autotune_keys=("integrator_with_gradient",),
                   bytes_macro_stem="INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES"),
    AlgoDescriptor("integrator_hessian", has_kernel_attr=False),

    # Centroidal / Energy / CoM
    AlgoDescriptor("generalized_gravity",
                   bytes_macro_stem="INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES"),
    AlgoDescriptor("nonlinear_effects",
                   bytes_macro_stem="INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES"),
    AlgoDescriptor("energy"),
    AlgoDescriptor("com"),
    AlgoDescriptor("ccrba"),
    AlgoDescriptor("coriolis_matrix"),
    AlgoDescriptor("dccrba"),
    AlgoDescriptor("cmm_time_variation"),

    # Plant (no standalone kernel)
    AlgoDescriptor("plant", has_kernel_attr=False),
    # Collision (no standalone benchmarked kernel; config_free lives in grid_collision)
    AlgoDescriptor("collision", has_kernel_attr=False),
)


_BY_KEY_DESCRIPTOR = {d.key: d for d in ALGO_DESCRIPTORS}


def descriptor_for(key: str) -> AlgoDescriptor:
    try:
        return _BY_KEY_DESCRIPTOR[key]
    except KeyError:
        raise KeyError(f"{key!r} not in ALGO_DESCRIPTORS — add it to algo_registry.py") from None


def build_launch_config_algo_to_symbol() -> dict[str, str]:
    """Reconstruct LAUNCH_CONFIG_ALGO_TO_SYMBOL (bench JSON key -> grid symbol)
    from the descriptor rows. Parity-checked against the live dict in
    test/test_algo_descriptor_parity.py; the eventual Step-1 generation will
    consume this directly."""
    return {k: d.key for d in ALGO_DESCRIPTORS for k in d.autotune_keys}


def build_single_label_map() -> dict[str, str]:
    """e.g. {'single call inverse_dynamics': 'inverse_dynamics', 'single call idsva_so_body_frame': 'idsva_so_body_frame', ...}."""
    return {f"single call {e.printf_label.lower()}": e.key for e in ALGO_REGISTRY}


def build_batch_with_mem_label_map() -> dict[str, str]:
    """e.g. {'inverse_dynamics with memory': 'inverse_dynamics', 'fdsva_so with memory': 'fdsva_so', ...}."""
    return {f"{e.printf_label.lower()} with memory": e.key for e in ALGO_REGISTRY}


def build_batch_compute_only_label_map() -> dict[str, str]:
    """e.g. {'inverse_dynamics compute only': 'inverse_dynamics', 'fdsva_so compute only': 'fdsva_so', ...}."""
    return {f"{e.printf_label.lower()} compute only": e.key for e in ALGO_REGISTRY}


def build_display_map() -> dict[str, str]:
    """key → display string, for generate_report.ALGO_DISPLAY."""
    return {e.key: e.display for e in ALGO_REGISTRY}


def build_sections_map() -> dict[str, list[str]]:
    """section → [keys], preserving registry order, for generate_report.ALGO_SECTIONS."""
    out: dict[str, list[str]] = {}
    for entry in ALGO_REGISTRY:
        out.setdefault(entry.section, []).append(entry.key)
    return out


_BY_KEY = {e.key: e for e in ALGO_REGISTRY}


def printf_label_for(key: str) -> str:
    """Return the codegen printf label for an algorithm key.

    Used by per-algo `gen_*_host` codegen functions so the emitted
    `printf("Single Call X ...")` string is derived from this registry
    instead of hardcoded in each algorithm file.
    """
    try:
        return _BY_KEY[key].printf_label
    except KeyError:
        raise KeyError(f"{key!r} not in ALGO_REGISTRY — add it to algo_registry.py") from None


def single_call_printf_line(key: str) -> str:
    """Return the full C++ printf statement codegen should emit for single-call timing."""
    return (
        f'printf("Single Call {printf_label_for(key)} %fus\\n",'
        'time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));'
    )


# ─────────────────────────────────────────────────────────────────────────────
# DESCRIPTOR TABLE STEP 3 — arena / spill schema + composer (item M).
#
# Step 3 folds the ~272 hand-written `*_t_count` arena expressions + spill ladders
# in GRiDCodeGenerator.gen_add_constants_helpers into the descriptor table
# (design: docs/open-tasks/design_descriptor_table_spec.md §3-4). This is the
# RISKIEST step (a wrong arena silently under-sizes shared memory), so it lands
# strictly one algo per commit behind a byte-diff + CUDA-equivalence gate.
#
# Step 3.0 (this landing) GENERATES NOTHING. It lands:
#   - the `ArenaRegion` / `SpillRung` schema the per-algo folds will populate,
#   - an `ArenaCtx` snapshot of the sizing primitives + inner-temp helper results,
#   - a per-algo `arena_full_fn` closure that recomputes the FULL (least-spill /
#     rung-0) arena t_count from the ArenaCtx, DECOUPLED from the imperative math,
#   - `compose_arena_full(key, ctx)` + `ARENA_COMPOSED_KEYS`.
# test/test_algo_descriptor_arena_parity.py asserts every closure reproduces the
# imperative `GRiDCodeGenerator._arena_full_t_counts[key]` on the matrix robots —
# the Step-0-style safety net that de-risks driving the arena sites from the table.
#
# The 3 SO-DISPATCH algos (idsva_so_body_frame / idsva_so_world_frame / fdsva_so)
# have per_base_override + workspace_bytes_fn + dispatch aliases that are inseparable
# from their multi-rung emission, so their closures land WITH their generation flip
# in commits 3.4/3.5 — they are captured in `_arena_full_t_counts` but intentionally
# excluded from `ARENA_COMPOSED_KEYS` here.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArenaRegion:
    """One named buffer band in an algo's shared-memory arena. The per-algo folds
    (3.1+) populate these; the full arena is `sum(count_fn(ctx))` + the auto-injected
    cross-cutting rt_xfixed region for s_temp-domain arenas (the §2 bug class fix)."""
    name: str
    count_fn: Callable[[ArenaCtx], int]
    domain: str = "s_temp"          # "s_temp" | "XmatsHom"
    spillable: bool = False         # can move to d_workspace at a spilled rung
    cross_cutting: bool = False     # contributed by a shared helper (rt_xfixed)


@dataclass(frozen=True)
class SpillRung:
    """One rung of an algo's spill ladder, least-spill first. `in_smem` names the
    regions KEPT in smem at this rung (the rest route to d_workspace)."""
    label: str
    in_smem: tuple[str, ...]


@dataclass(frozen=True)
class ArenaCtx:
    """Immutable snapshot of the arena-sizing primitives + inner-temp helper results
    for ONE robot, as read by gen_add_constants_helpers. Built by
    `arena_ctx_from_codegen(gen)` AFTER gen_add_constants_helpers has run (so all
    spill flags are final). The `arena_full_fn` closures below read only these
    fields, never the generator — so the composer is decoupled from the imperative
    math and a drift in either is caught by the parity test."""
    # sizing scalars
    n: int            # num_pos
    nv: int           # num_vel
    NB: int           # num_bodies
    NJ: int           # num_joints
    XI: int           # DYNAMICS_XI_T_COUNT (gen_get_XI_size)
    XHom: int         # XHOM_T_COUNT
    rt: int           # rt_xfixed_reserve (36*NJ under runtime_transform, else 0)
    floating: bool
    has_mimic: bool
    n_leaf: int       # total_leaf_nodes (num EEs)
    osc_XI: int       # gen_get_XI_size(False, False) — usually == XI
    # inner-temp helper results (pure size functions of the robot)
    id_inner: int
    idr_inner: int
    ker_inner: int
    coriolis_inner: int
    dccrba_inner: int
    dccrba_sJ: int
    fpg_inner: int
    feg_inner: int
    minv_inner: int
    minv_F: int
    minv_noF: int
    ximats_helper_temp: int
    fd_inner_Fsmem: int
    fd_inner_noFsmem: int
    fdgrad_inner: int
    idgrad_inner: int
    aba_inner: int
    crba_inner: int
    ee_inner: int
    eeg_inner: int
    d2ee_inner: int
    d2ee_out: int
    # second-order (fdsva_so) inner-temp helpers
    idsva_body_inner: int      # gen_idsva_so_body_frame_inner_temp_mem_size()
    idsva_world_inner: int     # world (floating) else body — matches gen dispatch
    fdsva_fdg_inline: int      # gen_fdsva_so_fd_gradient_inline_temp_mem_size()
    fdsva_fdg_inline_spilled: int  # ..._spilled()
    idsva_world_cold: int      # gen_idsva_so_world_cold_floats()
    idsva_bf_jids_a: int       # len(get_jid_ancestor_ids(include_joint=True)[0]) — body t/p scratch span
    idgrad_selective_shared: int   # id_gradient_temp_layout()["selective_shared_count"] (sparse da_df band)
    has_spherical: bool

    # ── derived buffer counts (thin helpers so the closures read like the source) ──
    @property
    def id_vaf(self) -> int:          # s_vaf body-band, pos-flavour (18*NJ mimic else 18*n)
        return 18 * (self.NJ if self.has_mimic else self.n)

    @property
    def grad_vaf(self) -> int:        # s_vaf body-band, vel-flavour (18*NJ mimic else 18*nv)
        return 18 * (self.NJ if self.has_mimic else self.nv)

    @property
    def nb_vaf(self) -> int:          # id-bias s_vaf body-count (NJ mimic else n)
        return self.NJ if self.has_mimic else self.n

    @property
    def centroidal_inner_noJ(self) -> int:
        return 16 * self.NJ + 36 * self.NB + 6 * self.nv + 36

    @property
    def osc_temp(self) -> int:
        return max(self.minv_noF, 16 * self.NJ)

    # ── fdsva_so ladder primitives (mirror _fdsva_so_tiers in the generator) ──
    @property
    def fdsva_base(self) -> int:      # inputs + s_qdd + s_Minv + s_df_du + XI
        return 4*self.nv + self.nv*self.nv + self.nv + 2*self.nv*self.nv + self.XI

    @property
    def fdsva_inner_idsva(self) -> int:   # world (floating) else body — the dispatched idsva inner
        return self.idsva_world_inner

    @property
    def fdsva_cold_floats(self) -> int:   # world cold-quad spill, only when the composed inner is world
        return self.idsva_world_cold if (self.floating or self.has_spherical) else 0

    @property
    def fdsva_temp_full(self) -> int:     # inner | contraction(4nv³) | fd_grad_inline, + rt
        return max(self.fdsva_inner_idsva, 4*self.nv**3, self.fdsva_fdg_inline) + self.rt

    @property
    def fdsva_temp_idsva_cold(self) -> int:
        """Pool for the `idsva_cold` rung: the idsva world inner spills its cold quad to
        GLOBAL, so only the IDSVA term shrinks — by fdsva_cold_floats.

        ⚠ The reduction MUST be applied INSIDE the max, not subtracted from the total.
        The pool is a max over three INDEPENDENT consumers (idsva inner | contraction 4nv³
        | fd_grad inline), and on a floating quadruped the CONTRACTION dominates
        (go2-floating: idsva=3030, contraction=23328, fdg=10494). Shrinking the idsva term
        3030 -> 1938 therefore changes the max by NOTHING, yet the old formula
        `fdsva_base + fdsva_temp_full - fdsva_cold_floats` still cut 1092 elements off the
        arena. The kernel kept carving the full pool, so the launch under-reserved by 1077
        elements (4308 B) and fdsva_so_kernel wrote past the end of shared memory ->
        "illegal memory access" on go2-floating @ TIER_SHARED, at EVERY thread count
        (memcheck: invalid __shared__ write at 0x190fc, 252 B past the 100 KiB cap).
        See docs/agent_debugging_guide.md §1t.
        """
        return max(self.fdsva_inner_idsva - self.fdsva_cold_floats,
                   4*self.nv**3, self.fdsva_fdg_inline) + self.rt

    @property
    def fdsva_temp_no_contract(self) -> int:
        return max(self.fdsva_inner_idsva, self.fdsva_fdg_inline) + self.rt

    @property
    def fdsva_temp_spilled(self) -> int:
        return max(self.fdsva_inner_idsva, self.fdsva_fdg_inline_spilled) + self.rt

    @property
    def ig_inner_selective(self) -> int:
        # integrator_gradient rung-2 selective FD-grad inner: the MIMIC dense inner
        # can't shrink (ignores USE_DA_DF_SPILL) so it stays at the full inner; the
        # SPARSE inner shrinks to max(minv inner, the da_df selective_shared_count).
        return self.fdgrad_inner if self.has_mimic else max(self.minv_inner, self.idgrad_selective_shared)

    @property
    def idgrad_selective(self) -> int:   # id_gradient selective inner (mimic dense can't shrink)
        return self.idgrad_inner if self.has_mimic else self.idgrad_selective_shared

    @property
    def fdgrad_selective(self) -> int:   # fd_gradient selective inner = max(minv, id_grad selective)
        return max(self.minv_inner, self.idgrad_selective)

    @property
    def aba_surgical_inner(self) -> int:   # aba surgical rung hot-band inner (cold sub-band -> d_cold)
        return (self.aba_inner - 138) if self.floating else (98 * self.NJ)


def arena_ctx_from_codegen(gen, xi=None, xhom=None, rt=None) -> ArenaCtx:
    """Build an ArenaCtx from a GRiDCodeGenerator. Reads the same sizing primitives +
    inner-temp helpers the imperative arena math uses. `xi`/`xhom`/`rt` override the
    derived values so gen_add_constants_helpers can pass its EXACT locals (matters when
    called with include_base_inertia=True, where the derived XI would differ). Kept in
    this module (not the generator) as the table's adapter."""
    robot = gen.robot
    n = robot.get_num_pos()
    return ArenaCtx(
        n=n,
        nv=robot.get_num_vel(),
        NB=robot.get_num_bodies(),
        NJ=robot.get_num_joints(),
        XI=gen.gen_get_XI_size(False, include_homogenous_transforms=False) if xi is None else xi,
        XHom=gen.gen_get_Xhom_size()[0] if xhom is None else xhom,
        rt=((36 * robot.get_num_joints()) if getattr(gen, "runtime_transform", False) else 0) if rt is None else rt,
        floating=bool(robot.floating_base),
        has_mimic=bool(gen.robot_has_mimic_joints()),
        n_leaf=robot.get_total_leaf_nodes(),
        osc_XI=gen.gen_get_XI_size(False, False),
        id_inner=gen.gen_inverse_dynamics_inner_temp_mem_size(),
        idr_inner=gen.gen_inverse_dynamics_regressor_inner_temp_mem_size(),
        ker_inner=gen.gen_kinetic_energy_regressor_inner_temp_mem_size(),
        coriolis_inner=gen.gen_coriolis_matrix_inner_temp_mem_size(),
        dccrba_inner=gen._dccrba_inner_temp_mem_size(),
        dccrba_sJ=gen._dccrba_sweep_J_count(),
        fpg_inner=gen.gen_forward_dynamics_parameter_gradient_inner_temp_mem_size(),
        feg_inner=gen.gen_f_ext_gradient_inner_temp_mem_size(),
        minv_inner=gen.gen_minv_inner_temp_mem_size(),
        minv_F=gen.gen_minv_inner_F_size(),
        minv_noF=gen.gen_minv_inner_no_F_size(),
        ximats_helper_temp=gen.gen_load_update_XImats_helpers_temp_mem_size(),
        fd_inner_Fsmem=gen.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=True),
        fd_inner_noFsmem=gen.gen_forward_dynamics_inner_temp_mem_size(minv_f_in_smem=False),
        fdgrad_inner=gen.gen_forward_dynamics_gradient_inner_temp_mem_size(),
        idgrad_inner=gen.gen_inverse_dynamics_gradient_inner_temp_mem_size(),
        aba_inner=gen.gen_aba_inner_temp_mem_size(),
        crba_inner=gen.gen_crba_inner_temp_mem_size(),
        ee_inner=gen.gen_end_effector_pose_inner_temp_mem_size(),
        eeg_inner=gen.gen_end_effector_pose_gradient_inner_temp_mem_size(),
        d2ee_inner=gen.gen_end_effector_pose_hessian_inner_temp_mem_size(),
        d2ee_out=gen.gen_end_effector_pose_hessian_output_count(),
        idsva_body_inner=gen.gen_idsva_so_body_frame_inner_temp_mem_size(),
        idsva_world_inner=(gen.gen_idsva_so_world_frame_temp_mem_size() if robot.floating_base
                           else gen.gen_idsva_so_body_frame_inner_temp_mem_size()),
        fdsva_fdg_inline=gen.gen_fdsva_so_fd_gradient_inline_temp_mem_size(),
        fdsva_fdg_inline_spilled=gen.gen_fdsva_so_fd_gradient_inline_temp_mem_size_spilled(),
        idsva_world_cold=gen.gen_idsva_so_world_cold_floats(),
        idsva_bf_jids_a=len(robot.get_jid_ancestor_ids(include_joint=True)[0]),
        idgrad_selective_shared=gen.gen_inverse_dynamics_gradient_temp_layout()["selective_shared_count"],
        has_spherical=bool(gen.robot.robot_has_spherical()),
    )


# Per-algo FULL (least-spill / rung-0) arena t_count closures, decoupled from the
# imperative gen_add_constants_helpers math. Each mirrors the buffer list the kernel
# slices; the parity test proves they agree on every matrix robot. The `+ c.rt`
# terms encode the rt_xfixed reservation on the s_temp-domain arenas (§2); commit 3.1
# refactors these into `ArenaRegion`s with domain-driven auto-injection.
_ARENA_FULL_FNS: dict[str, Callable[[ArenaCtx], int]] = {
    # ── Core dynamics (s_temp domain) ──
    "inverse_dynamics":
        lambda c: 2*c.n + c.n + c.id_vaf + c.n + c.id_inner + c.XI + c.rt,
    "minv":
        lambda c: c.n + c.n*c.n + c.minv_F + c.minv_noF + c.XI + c.rt,
    "forward_dynamics":
        lambda c: 3*c.n + c.nv + c.XI + c.rt + c.fd_inner_Fsmem,
    "aba":
        lambda c: c.nv + 3*c.n + 12*c.NJ + c.XI + c.rt + c.aba_inner,
    "crba":
        lambda c: c.nv*c.nv + (c.n + c.nv) + c.XI + c.rt + c.crba_inner,
    # ── Gradients / regressors (s_temp domain) ──
    "inverse_dynamics_gradient":
        lambda c: (c.nv + c.n) + 2*c.nv*c.nv + c.grad_vaf + c.nv + c.idgrad_inner + c.XI + c.rt,
    "forward_dynamics_gradient":
        lambda c: 3*c.n + 2*c.nv*c.nv + c.grad_vaf + c.nv + c.nv*c.nv + c.fdgrad_inner + c.XI + c.rt,
    "f_ext_gradient":
        lambda c: c.n + 2*(c.nv*6*c.NB) + c.nv*c.nv + max(c.feg_inner, c.minv_inner) + c.XI + c.rt,
    "f_ext_gradient_dq":
        # + c.rt: FD loop recomputes XImats per perturbation -> rt_xfixed reserved under
        # runtime_transform (matches the rt-aware kernel carve; §2 fix 2026-07-14). See the rung closure.
        lambda c: c.n + (c.n + (c.nv if c.floating else 0) + 2*c.nv*6*c.NB
                         + c.feg_inner + c.ximats_helper_temp) + c.XI + c.rt,
    "inverse_dynamics_regressor":
        lambda c: (c.n + 2*c.nv) + c.nv*10*c.NB + 18*c.n + c.idr_inner + c.XI + c.rt,
    "forward_dynamics_parameter_gradient":
        lambda c: (c.n + 2*c.nv) + c.nv*10*c.NB + c.nv*c.nv + c.nv*10*c.NB + c.nv + 18*c.n + c.nv
                  + c.fpg_inner + c.XI + c.rt,
    "kinetic_energy_regressor":
        lambda c: (c.n + c.nv) + 10*c.NB + 18*c.n + c.ker_inner + c.XI + c.rt,
    # ── Kinematics (XmatsHom domain — no rt) ──
    "potential_energy_regressor":
        lambda c: c.n + 10*c.NB + 16*c.NJ + c.XHom,
    "end_effector_pose":
        lambda c: c.n + 6*c.n_leaf + c.ee_inner + c.XHom,
    "end_effector_pose_gradient":
        lambda c: c.n + 6*c.n*c.n_leaf + c.eeg_inner + c.XHom,
    "end_effector_pose_hessian":
        lambda c: c.n + 6*c.nv*c.n_leaf + c.d2ee_out + c.d2ee_inner + c.XHom,
    "osc_inertia":
        lambda c: c.osc_XI + c.XHom + c.nv*c.nv + c.minv_F + 6*c.nv + c.nv*6 + 72 + c.osc_temp,
    # ── Integrators (s_temp domain) ──
    "integrator":
        lambda c: (3*c.n + c.nv + 3*c.nv + 3*(c.n + c.nv) + (c.n + c.nv) + c.XI + c.rt
                   + c.fd_inner_Fsmem),
    "integrator_gradient":
        lambda c: _integrator_gradient_full(c),
    "integrator_with_gradient":
        lambda c: _integrator_gradient_full(c),
    # ── Centroidal / energy (XmatsHom domain — no rt) ──
    "generalized_gravity":
        lambda c: 2*c.n + c.nv + 18*c.nb_vaf + c.nv + 6*c.n + c.XI + c.rt,
    "nonlinear_effects":
        lambda c: 2*c.n + c.nv + 18*c.nb_vaf + c.nv + 6*c.n + c.XI + c.rt,
    "coriolis_matrix":
        lambda c: c.nv*c.nv + (c.n + c.nv) + c.XI + c.rt + c.coriolis_inner,
    "dccrba":
        lambda c: (c.n + 6*c.nv + 3 + 4 + c.dccrba_inner + c.XHom) + 6*c.nv*c.nv + c.dccrba_sJ,
    "cmm_time_variation":
        lambda c: (2*c.n + 6*c.nv + 6*c.nv + 3 + 4 + c.dccrba_inner + c.XHom) + c.dccrba_sJ,
    "com":
        lambda c: (c.n + (3 + 3*c.nv) + 6*c.nv + 3 + 4 + c.centroidal_inner_noJ + c.XHom)
                  + 6*c.nv*c.NB,
    "ccrba":
        lambda c: (2*c.n + (6*c.nv + 6) + 6*c.nv + 3 + 4 + c.centroidal_inner_noJ + c.XHom)
                  + 6*c.nv*c.NB,
    "energy":
        lambda c: (2*c.n + 3 + 6*c.nv + 3 + 4 + c.centroidal_inner_noJ + c.XHom)
                  + 6*c.nv*c.NB,
    # ── Second-order (s_temp domain — full == rung-0) ──
    "fdsva_so":
        lambda c: c.fdsva_base + 8*c.nv**3 + c.fdsva_temp_full,
    # idsva_so world frame: full arena is ctx-pure (no picker). Body-frame full is
    # NOT composed — its floating path is a picker-dependent single-value override
    # (grav_full_spill) inseparable from the smem budget, so it stays in-gen (see 3.5).
    "idsva_so_world_frame":
        lambda c: (3*c.n) + c.idsva_world_inner + c.XI + c.rt + 4*c.nv**3,
    # plant_step_hessian (integrator_hessian) s_d2AB surface. Pool == fdsva_temp_full.
    "integrator_hessian":
        lambda c: (_psh_base(c) + 2*c.nv*(3*c.nv)*(3*c.nv) + 8*c.nv**3 + c.fdsva_temp_full),
}


def _psh_base(c: ArenaCtx) -> int:
    # plant_step_hessian smem base: s_x(2nv) + s_u(nv) + s_qdd(nv) + s_Minv(nv²) + s_df_du(2nv²) + XI.
    return (c.nv + c.nv) + c.nv + c.nv*c.nv + 2*c.nv*c.nv + c.nv + c.XI


def _integrator_gradient_full(c: ArenaCtx) -> int:
    # max(integrator_gradient_t_count, +s_x_kp1) == the with_x_kp1 arena (n+nv > 0).
    _max_stages = 4
    ig = (3*c.n + 2*c.nv*3*c.nv + 2*(c.nv*2*c.nv)
          + c.grad_vaf + c.nv*c.nv + c.nv
          + (c.n + c.nv) + _max_stages*c.nv + _max_stages*c.nv*3*c.nv
          + 72
          + c.fdgrad_inner + c.XI + c.rt)
    return ig + (c.n + c.nv)


ARENA_COMPOSED_KEYS: frozenset[str] = frozenset(_ARENA_FULL_FNS)


def compose_arena_full(key: str, ctx: ArenaCtx) -> int:
    """FULL (least-spill / rung-0) arena t_count for `key`, composed from the ArenaCtx.
    Parity-checked against GRiDCodeGenerator._arena_full_t_counts[key]. Raises KeyError
    for SO-dispatch algos not yet composed (see ARENA_COMPOSED_KEYS)."""
    return _ARENA_FULL_FNS[key](ctx)


# Per-algo SPILL-LADDER rung arenas (least-spill first), composed from the ArenaCtx.
# The generator still runs `select_shared_tier_3way(*rungs)` (the picker needs the
# robot's smem budget, not pure ctx) and builds the per-tier tuple that drives the
# TIER-templated `*_DYNAMIC_SHARED_MEM_BYTES` macro. This folds the RUNG SIZING (the
# §2-relevant buffer lists) into the table; rung[0] MUST equal `arena_full_fn`. Most
# rungs are subtractive (full − buffer) but FD/MINV re-call the inner helper with
# minv_f_in_smem=False, so they read the dedicated ctx.fd_inner_noFsmem / minv_noF.
_ARENA_RUNG_FNS: dict[str, tuple[Callable[[ArenaCtx], int], ...]] = {
    "crba": (
        lambda c: c.nv*c.nv + (c.n + c.nv) + c.XI + c.rt + c.crba_inner,   # full: s_M + inner + input/XI
        lambda c: (c.n + c.nv) + c.XI + c.rt + c.crba_inner,               # output_spill: s_M -> ws
        lambda c: (c.n + c.nv) + c.XI + c.rt,                              # workspace: s_M + inner -> ws
    ),
    "minv": (
        lambda c: c.n + c.n*c.n + c.minv_F + c.minv_noF + c.XI + c.rt,     # full: F in smem
        lambda c: c.n + c.n*c.n + c.minv_noF + c.XI + c.rt,                # surgical: F -> ws
    ),
    "forward_dynamics": (
        lambda c: 3*c.n + c.nv + c.XI + c.rt + c.fd_inner_Fsmem,           # full: minv-F in smem
        lambda c: 3*c.n + c.nv + c.XI + c.rt + c.fd_inner_noFsmem,         # surgical: minv-F -> ws
    ),
    # ── 3.4 fdsva_so 8-rung ladder (least-spill first). Only t_count here; the 9-tuple
    # STATE FLAGS (use_global_tensors, use_workspace_temp, ...) stay in the generator. ──
    "fdsva_so": (
        lambda c: c.fdsva_base + 8*c.nv**3 + c.fdsva_temp_full,             # full
        lambda c: c.fdsva_base + c.fdsva_temp_full,                         # global_tensors
        lambda c: c.fdsva_base + c.fdsva_temp_idsva_cold,                   # idsva_cold (see property: reduction goes INSIDE the max)
        lambda c: c.fdsva_base + c.fdsva_temp_no_contract,                  # workspace_temp
        lambda c: c.fdsva_base + c.fdsva_temp_spilled,                      # workspace_temp_spill
        lambda c: (c.fdsva_base - 2*c.nv*c.nv) + c.fdsva_temp_spilled,      # spill_df_du
        lambda c: (c.fdsva_base - 2*c.nv*c.nv - c.nv*c.nv) + c.fdsva_temp_spilled,  # spill_Minv
        lambda c: c.fdsva_base,                                             # pool_global
    ),
    # ── 3.5 idsva_so dispatch ladders. Body ladder (5-rung) is FIXED-BASE only —
    # the generator composes/asserts it just on fixed base (floating body uses a
    # picker-dependent single-value override kept in-gen). World ladder (4-rung) is
    # composed on all bases. rung[0] == the respective full arena. ──
    "idsva_so_body_frame": (
        lambda c: (3*c.n) + c.idsva_body_inner + c.XI + 4*c.nv**3 + c.rt,   # full: output + s_temp + BC in smem
        lambda c: (3*c.n) + c.idsva_body_inner + c.XI + c.rt,               # global_output: 4nv³ output -> ws
        lambda c: (3*c.n) + c.idsva_body_inner + c.XI + c.rt - 36*c.NB,     # output_bc: + BC(36NB) -> ws
        lambda c: (3*c.n) + c.idsva_body_inner + c.XI + c.rt - 36*c.idsva_bf_jids_a,  # output_tp: + t/p scratch -> ws
        lambda c: (3*c.n) + c.XI,                                           # output_temp: whole s_temp -> ws (no rt)
    ),
    "idsva_so_world_frame": (
        lambda c: (3*c.n) + c.idsva_world_inner + c.XI + c.rt + 4*c.nv**3,  # full: output + s_temp in smem
        lambda c: (3*c.n) + c.idsva_world_inner + c.XI + c.rt,              # global_output: 4nv³ output -> ws
        lambda c: (3*c.n) + c.idsva_world_inner + c.XI + c.rt - c.idsva_world_cold,  # output_cold: + cold quad -> ws
        lambda c: (3*c.n) + c.XI,                                           # output_temp: whole s_temp -> ws (no rt)
    ),
    # ── 3.5c core-gradient + aba ladders ──
    "inverse_dynamics_gradient": (
        lambda c: (c.nv + c.n) + 2*c.nv*c.nv + c.grad_vaf + c.nv + c.idgrad_inner + c.XI + c.rt,      # full
        lambda c: (c.nv + c.n) + 2*c.nv*c.nv + c.grad_vaf + c.nv + c.idgrad_selective + c.XI + c.rt,  # selective inner
        lambda c: (c.nv + c.n) + 2*c.nv*c.nv + c.grad_vaf + c.nv + c.XI + c.rt,                       # emergency: inner -> ws
    ),
    "forward_dynamics_gradient": (
        lambda c: 3*c.n + 2*c.nv*c.nv + c.grad_vaf + c.nv + c.nv*c.nv + c.fdgrad_inner + c.XI + c.rt,      # full
        lambda c: 3*c.n + 2*c.nv*c.nv + c.grad_vaf + c.nv + c.nv*c.nv + c.fdgrad_selective + c.XI + c.rt,  # selective inner
        lambda c: 3*c.n + 2*c.nv*c.nv + c.grad_vaf + c.nv + c.nv*c.nv + c.XI + c.rt,                       # emergency: inner -> ws
        lambda c: (3*c.n + 2*c.nv*c.nv + c.grad_vaf + c.nv + c.nv*c.nv + c.XI + c.rt
                   - (2*c.nv*c.nv + c.nv*c.nv)),                                                           # output_spill: + s_dc_du/s_Minv -> SO band
    ),
    "aba": (
        lambda c: c.nv + 3*c.n + 12*c.NJ + c.XI + c.rt + c.aba_inner,           # full: whole inner in smem
        lambda c: c.nv + 3*c.n + 12*c.NJ + c.XI + c.rt + c.aba_surgical_inner,  # surgical: cold sub-band -> d_cold
        lambda c: c.nv + 3*c.n + 12*c.NJ + c.XI + c.rt,                         # workspace: whole inner -> ws
    ),
    # ── 3.5d parameter/regressor + f_ext ladders ──
    "inverse_dynamics_regressor": (
        lambda c: (c.n + 2*c.nv) + c.nv*10*c.NB + 18*c.n + c.idr_inner + c.XI + c.rt,   # full: s_Y in smem
        lambda c: (c.n + 2*c.nv) + 18*c.n + c.idr_inner + c.XI + c.rt,                  # surgical: s_Y -> ws
    ),
    "forward_dynamics_parameter_gradient": (
        lambda c: ((c.n + 2*c.nv) + c.nv*10*c.NB + c.nv*c.nv + c.nv*10*c.NB + c.nv + 18*c.n + c.nv
                   + c.fpg_inner + c.XI + c.rt),                                        # full: s_Y in smem
        lambda c: ((c.n + 2*c.nv) + c.nv*c.nv + c.nv*10*c.NB + c.nv + 18*c.n + c.nv
                   + c.fpg_inner + c.XI + c.rt),                                        # surgical: first s_Y -> ws
    ),
    "f_ext_gradient": (
        lambda c: c.n + 2*(c.nv*6*c.NB) + c.nv*c.nv + max(c.feg_inner, c.minv_inner) + c.XI + c.rt,  # full: both outs + minv-F
        lambda c: c.n + (c.nv*6*c.NB) + c.nv*c.nv + max(c.feg_inner, c.minv_inner) + c.XI + c.rt,     # out_spill: 2nd out -> ws
        lambda c: c.n + c.nv*c.nv + max(c.feg_inner, c.minv_noF) + c.XI + c.rt,                        # deep: both outs -> ws, minv no-F
    ),
    "f_ext_gradient_dq": (
        # + c.rt: the FD loop RECOMPUTES XImats per perturbation, so under runtime_transform its
        # load_update_XImats scratch reserves the rt_xfixed region (36*NJ) exactly like every other
        # s_temp-domain arena. The imperative kernel carve (_f_ext_gradient_dq_smem_count -> the rt-aware
        # gen_load_update_XImats_helpers_temp_mem_size) already includes it; this closure did NOT, so the
        # launch macro under-counted by 36*NJ under rt (a §2 silent under-size, caught by
        # test_shared_arena_covers_carve's rt cells 2026-07-14).
        lambda c: c.n + (c.n + (c.nv if c.floating else 0) + 2*c.nv*6*c.NB
                         + c.feg_inner + c.ximats_helper_temp) + c.XI + c.rt,  # full: both JT bufs in smem
        lambda c: c.n + (c.n + (c.nv if c.floating else 0)
                         + c.feg_inner + c.ximats_helper_temp) + c.XI + c.rt,  # spill: JT pair -> ws
    ),
    # ── 3.5e kinematics ladders (XmatsHom domain; degenerate 3rd rung == 2nd) ──
    "osc_inertia": (
        lambda c: c.osc_XI + c.XHom + c.nv*c.nv + c.minv_F + 6*c.nv + c.nv*6 + 72 + c.osc_temp,  # full: s_F in smem
        lambda c: c.osc_XI + c.XHom + c.nv*c.nv + 6*c.nv + c.nv*6 + 72 + c.osc_temp,             # spill_F: s_F -> ws
    ),
    "end_effector_pose_gradient": (
        lambda c: c.n + 6*c.n*c.n_leaf + c.eeg_inner + c.XHom,   # full: inner + output in smem
        lambda c: c.n + c.XHom,                                 # spill_temp: inner + output -> ws
        lambda c: c.n + c.XHom,                                 # MINIMAL (== spill_temp)
    ),
    "end_effector_pose_hessian": (
        lambda c: c.n + 6*c.nv*c.n_leaf + c.d2ee_out + c.d2ee_inner + c.XHom,  # full: grad + output + inner
        lambda c: c.n + 6*c.nv*c.n_leaf + c.d2ee_inner + c.XHom,               # spill: output -> ws
        lambda c: c.n + 6*c.nv*c.n_leaf + c.d2ee_inner + c.XHom,               # MINIMAL (== spill)
    ),
    "integrator_hessian": (
        lambda c: _psh_base(c) + 2*c.nv*(3*c.nv)*(3*c.nv) + 8*c.nv**3 + c.fdsva_temp_full,  # full: d2AB + fdsva tensors + pool
        lambda c: _psh_base(c),                                                             # deep spill: base only in smem
    ),
    # integrator_gradient 4-rung ladder. rung[0] == _integrator_gradient_full. Dqdd =
    # max_stages(4)*nv*3nv, dAB = 2*nv*3nv. rung-2 selective-inner shrink is mimic-conditional.
    "integrator_gradient": (
        lambda c: _integrator_gradient_full(c),                                          # 0 full
        lambda c: _integrator_gradient_full(c) - 4*c.nv*3*c.nv,                           # 1 +Dqdd -> ws
        lambda c: (_integrator_gradient_full(c) - 4*c.nv*3*c.nv - 2*c.nv*3*c.nv           # 2 +dAB + selective inner
                   - (c.fdgrad_inner - c.ig_inner_selective)),
        lambda c: _integrator_gradient_full(c) - 4*c.nv*3*c.nv - 2*c.nv*3*c.nv - c.fdgrad_inner,  # 3 +whole inner -> ws
    ),
    # ── 3.3 clean ladders ──
    "coriolis_matrix": (
        lambda c: c.nv*c.nv + (c.n + c.nv) + c.XI + c.rt + c.coriolis_inner,  # full: s_coriolis + inner + input/XI
        lambda c: (c.n + c.nv) + c.XI + c.rt + c.coriolis_inner,              # output_spill: s_coriolis -> SO band
        lambda c: (c.n + c.nv) + c.XI + c.rt,                                 # workspace: output + inner -> ws
    ),
    "integrator": (
        lambda c: (3*c.n + c.nv + 3*c.nv + 3*(c.n + c.nv) + (c.n + c.nv)
                   + c.XI + c.rt + c.fd_inner_Fsmem),                         # full: minv-F in smem
        lambda c: (3*c.n + c.nv + 3*c.nv + 3*(c.n + c.nv) + (c.n + c.nv)
                   + c.XI + c.rt + c.fd_inner_noFsmem),                       # Fspill: minv-F -> ws
    ),
    # dccrba family (XmatsHom domain, s_J band tier-routed). rung[0] == full closure.
    "dccrba": (
        lambda c: (c.n + 6*c.nv + 3 + 4 + c.dccrba_inner + c.XHom) + 6*c.nv*c.nv + c.dccrba_sJ,  # L0: out + s_J in smem
        lambda c: (c.n + 6*c.nv + 3 + 4 + c.dccrba_inner + c.XHom) + c.dccrba_sJ,                 # L1: out -> ws, s_J smem
        lambda c: (c.n + 6*c.nv + 3 + 4 + c.dccrba_inner + c.XHom),                                # L2: out + s_J -> ws
    ),
    "cmm_time_variation": (
        lambda c: (2*c.n + 6*c.nv + 6*c.nv + 3 + 4 + c.dccrba_inner + c.XHom) + c.dccrba_sJ,  # L0: s_J in smem
        lambda c: (2*c.n + 6*c.nv + 6*c.nv + 3 + 4 + c.dccrba_inner + c.XHom),                # L1: s_J -> ws
    ),
    # com/ccrba/energy (centroidal, s_J band tier-routed). Differ only in the input+output band.
    "com": (
        lambda c: (c.n + (3 + 3*c.nv) + 6*c.nv + 3 + 4 + c.centroidal_inner_noJ + c.XHom) + 6*c.nv*c.NB,  # L0: s_J smem
        lambda c: (c.n + (3 + 3*c.nv) + 6*c.nv + 3 + 4 + c.centroidal_inner_noJ + c.XHom),                # L1: s_J -> ws
    ),
    "ccrba": (
        lambda c: (2*c.n + (6*c.nv + 6) + 6*c.nv + 3 + 4 + c.centroidal_inner_noJ + c.XHom) + 6*c.nv*c.NB,
        lambda c: (2*c.n + (6*c.nv + 6) + 6*c.nv + 3 + 4 + c.centroidal_inner_noJ + c.XHom),
    ),
    "energy": (
        lambda c: (2*c.n + 3 + 6*c.nv + 3 + 4 + c.centroidal_inner_noJ + c.XHom) + 6*c.nv*c.NB,
        lambda c: (2*c.n + 3 + 6*c.nv + 3 + 4 + c.centroidal_inner_noJ + c.XHom),
    ),
}

ARENA_RUNG_KEYS: frozenset[str] = frozenset(_ARENA_RUNG_FNS)


def compose_arena_rungs(key: str, ctx: ArenaCtx) -> tuple[int, ...]:
    """Spill-ladder rung arenas (least-spill first) for `key`, composed from the
    ArenaCtx. Fed to the generator's `select_shared_tier_3way`. rung[0] == the FULL
    arena (asserted consistent with `compose_arena_full` by the parity test)."""
    return tuple(fn(ctx) for fn in _ARENA_RUNG_FNS[key])

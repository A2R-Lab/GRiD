"""Step-0 parity net for the per-algo DESCRIPTOR table (item M).

The descriptor table (`ALGO_DESCRIPTORS` in `algo_registry.py`) is being grown to
collapse the ~10 scattered per-algo edit sites in `GRiDCodeGenerator.py` into ONE
row per algorithm (design: docs/open-tasks/design_descriptor_table_spec.md). Step 0
GENERATES NOTHING — it just lands the rows and asserts they REPRODUCE the live
literal sites exactly, so later steps can drive those sites from the table behind a
byte-identical gate.

These are pure-Python introspection tests (no codegen run, no nvcc, no GPU). They
fail the instant the descriptor table and the hand-written sites disagree — which is
the whole point: the table becomes the single source of truth, and drift is caught
in ordinary CI instead of as a silent tier/smem mismatch on one robot.

Covered sites this step:
  #1  LAUNCH_CONFIG_ALGO_TO_SYMBOL  (bench JSON key -> grid symbol)
  #8  KERNEL_ATTR_MANIFEST metadata (algo_label, algo_short, gate_attr, bytes_macro)
  #9  mujoco_manifest metadata      (the floating mjx twin's short/gate/bytes)
The kernel SIGNATURES (#8 5th tuple element) and the arena/spill closures (#3-#7)
are intentionally NOT yet in the schema; they move in Steps 2-3 with their own
parity gate.
"""

from __future__ import annotations

from GRiDCodeGenerator.GRiDCodeGenerator import GRiDCodeGenerator
from GRiDCodeGenerator.algo_registry import (
    ALGO_DESCRIPTORS,
    ALGO_REGISTRY,
    build_launch_config_algo_to_symbol,
    descriptor_for,
)

# ─── FROZEN GOLDEN (Step 1) ──────────────────────────────────────────────────
# Was the live `LAUNCH_CONFIG_ALGO_TO_SYMBOL` dict in GRiDCodeGenerator.py; Step 1
# deleted it and now DERIVES it from the descriptor table. This frozen copy keeps
# the table regression-guarded against silent drift (a wrong autotune_key or a lost
# row would change the rebuilt map / enum order but this literal never moves).
_GOLDEN_ALGO_TO_SYMBOL = {
    "id":                    "inverse_dynamics",
    "minv":                  "minv",
    "fd":                    "forward_dynamics",
    "aba":                   "aba",
    "crba":                  "crba",
    "id_du":                 "inverse_dynamics_gradient",
    "fd_du":                 "forward_dynamics_gradient",
    "ee_pose":               "end_effector_pose",
    "ee_pose_gradient":      "end_effector_pose_gradient",
    "ee_pose_hessian":       "end_effector_pose_hessian",
    "idsva_so":              "idsva_so",
    "idsva_so_body_frame":   "idsva_so_body_frame",
    "idsva_so_world_frame":  "idsva_so_world_frame",
    "fdsva_so":              "fdsva_so",
    "integrator":            "integrator",
    "integrator_gradient":   "integrator_gradient",
    "integrator_with_gradient": "integrator_with_gradient",
}

# The emitted `enum GridAlgo` order (byte-identity anchor). Equals the launch-cfg
# symbols in descriptor order — integrators LAST (after Second-Order), which is why
# the Integrators AlgoDescriptor block is placed after fdsva_so in algo_registry.py.
_GOLDEN_LAUNCH_ORDER = (
    "inverse_dynamics", "minv", "forward_dynamics", "aba", "crba",
    "inverse_dynamics_gradient", "forward_dynamics_gradient",
    "end_effector_pose", "end_effector_pose_gradient", "end_effector_pose_hessian",
    "idsva_so", "idsva_so_body_frame", "idsva_so_world_frame", "fdsva_so",
    "integrator", "integrator_gradient", "integrator_with_gradient",
)


def test_descriptor_keys_are_registry_keys_one_to_one():
    """Every descriptor joins to exactly one registry entry and vice-versa — the
    table is TOTAL over the registry (no algo without a row, no orphan row). Order
    may DIFFER from ALGO_REGISTRY: the descriptor order is launch-config order
    (integrators last) so Step-1 enum emission is byte-identical; registry order is
    the report/section order. Both list the same keys with no dupes."""
    registry_keys = [e.key for e in ALGO_REGISTRY]
    descriptor_keys = [d.key for d in ALGO_DESCRIPTORS]
    assert len(descriptor_keys) == len(set(descriptor_keys)), (
        f"duplicate keys in ALGO_DESCRIPTORS: "
        f"{sorted({k for k in descriptor_keys if descriptor_keys.count(k) > 1})}"
    )
    assert set(descriptor_keys) == set(registry_keys), (
        "ALGO_DESCRIPTORS must cover the SAME keys as ALGO_REGISTRY (bijection).\n"
        f"  registry-only:   {sorted(set(registry_keys) - set(descriptor_keys))}\n"
        f"  descriptor-only: {sorted(set(descriptor_keys) - set(registry_keys))}"
    )


def test_descriptors_reproduce_launch_config_algo_to_symbol():
    """Site #1: the descriptors' autotune_keys reconstruct the frozen-golden
    LAUNCH_CONFIG_ALGO_TO_SYMBOL map (bench JSON key -> grid symbol) exactly."""
    rebuilt = build_launch_config_algo_to_symbol()
    assert rebuilt == _GOLDEN_ALGO_TO_SYMBOL, (
        "descriptor autotune_keys do not reproduce the golden ALGO_TO_SYMBOL map.\n"
        f"  missing from rebuilt: {set(_GOLDEN_ALGO_TO_SYMBOL.items()) - set(rebuilt.items())}\n"
        f"  extra in rebuilt:     {set(rebuilt.items()) - set(_GOLDEN_ALGO_TO_SYMBOL.items())}"
    )


def test_descriptor_launch_order_is_byte_identity_enum_order():
    """Step-1 byte-identity anchor: the launch-cfg descriptors, IN ORDER, reproduce
    the emitted `enum GridAlgo` order exactly. Guards the Integrators-block placement
    (must stay after Second-Order) so grid.cuh never silently reorders the enum."""
    launch_order = tuple(d.key for d in ALGO_DESCRIPTORS if d.carries_launch_cfg)
    assert launch_order == _GOLDEN_LAUNCH_ORDER, (
        "descriptor launch order drifted from the emitted GridAlgo enum order.\n"
        f"  got:    {launch_order}\n"
        f"  golden: {_GOLDEN_LAUNCH_ORDER}"
    )


def test_carries_launch_cfg_matches_symbol_set():
    """The set of descriptors that carry a launch_cfg equals the set of distinct
    grid symbols in the golden ALGO_TO_SYMBOL map (which algos get a baked enum)."""
    descriptor_syms = {d.key for d in ALGO_DESCRIPTORS if d.carries_launch_cfg}
    live_syms = set(_GOLDEN_ALGO_TO_SYMBOL.values())
    assert descriptor_syms == live_syms, (
        f"carries_launch_cfg set {sorted(descriptor_syms)} != symbol set {sorted(live_syms)}"
    )


def _manifest_metadata(entries):
    """(algo_label, algo_short, gate_attr, bytes_macro) per manifest entry."""
    return [(e[0], e[1], e[2], e[3]) for e in entries]


def test_descriptors_reproduce_kernel_attr_manifest_metadata():
    """Site #8: every KERNEL_ATTR_MANIFEST entry's (label, short, gate_attr,
    bytes_macro) is reproduced by the descriptor keyed on algo_short. Also asserts
    the label==short invariant the registry relies on, and that exactly the
    has_kernel_attr descriptors appear in the manifest (no missing / no orphan)."""
    manifest_meta = _manifest_metadata(GRiDCodeGenerator.KERNEL_ATTR_MANIFEST)

    manifest_shorts = [short for (_label, short, _gate, _bytes) in manifest_meta]
    assert len(manifest_shorts) == len(set(manifest_shorts)), (
        f"duplicate algo_short in KERNEL_ATTR_MANIFEST: "
        f"{sorted({s for s in manifest_shorts if manifest_shorts.count(s) > 1})}"
    )

    # every manifest entry matches its descriptor
    for label, short, gate, bytes_macro in manifest_meta:
        assert label == short, f"KERNEL_ATTR_MANIFEST label {label!r} != short {short!r}"
        d = descriptor_for(short)
        assert d.has_kernel_attr, f"{short}: manifest entry exists but descriptor.has_kernel_attr is False"
        assert d.gate_attr == gate, f"{short}: descriptor gate_attr {d.gate_attr!r} != manifest {gate!r}"
        assert d.bytes_macro == bytes_macro, (
            f"{short}: descriptor bytes_macro {d.bytes_macro!r} != manifest {bytes_macro!r}"
        )

    # exactly the has_kernel_attr descriptors appear in the manifest
    descriptor_attr_keys = {d.key for d in ALGO_DESCRIPTORS if d.has_kernel_attr}
    assert descriptor_attr_keys == set(manifest_shorts), (
        "has_kernel_attr descriptors must match the KERNEL_ATTR_MANIFEST short set.\n"
        f"  descriptor-only (claim attr, absent from manifest): {sorted(descriptor_attr_keys - set(manifest_shorts))}\n"
        f"  manifest-only (present, but descriptor.has_kernel_attr False): {sorted(set(manifest_shorts) - descriptor_attr_keys)}"
    )


# Frozen golden for the floating mjx twins (Step 2). The mujoco_manifest is now
# DERIVED (gen_init_close_grid: head = short+"(mjx)" + descriptor gate/bytes; payload =
# GRiDCodeGenerator.MJX_KERNEL_OVERLOADS). This golden pins which algos have an mjx twin
# and the (gate_attr, bytes_macro) each must resolve to — so a descriptor edit that would
# silently change an mjx twin's macro is caught here, same single-source-of-truth invariant.
_GOLDEN_MJX_HEADS = {
    "inverse_dynamics":            (None, "INVERSE_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "minv":                        (None, "MINV_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "forward_dynamics":            (None, "FORWARD_DYNAMICS_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "aba":                         (None, "ABA_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "crba":                        (None, "CRBA_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "end_effector_pose":           (None, "END_EFFECTOR_POSE_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "osc_inertia":                 (None, "OSC_INERTIA_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "end_effector_pose_gradient":  (None, "END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "end_effector_pose_hessian":   ("generate_end_effector_pose_hessian", "END_EFFECTOR_POSE_HESSIAN_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "inverse_dynamics_gradient":   ("generate_inverse_dynamics_gradient", "INVERSE_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "forward_dynamics_gradient":   ("generate_forward_dynamics_gradient", "FORWARD_DYNAMICS_GRADIENT_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "inverse_dynamics_regressor":  (None, "INVERSE_DYNAMICS_REGRESSOR_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "idsva_so_world_frame":        ("generate_idsva_so_world_frame", "IDSVA_SO_WORLD_FRAME_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "fdsva_so":                    ("generate_fdsva_so", "FDSVA_SO_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "integrator":                  (None, "INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES<T>()"),
    "integrator_gradient":         (None, "INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES<T>()"),
}


def test_descriptors_cover_mujoco_manifest_metadata():
    """Site #9: the floating mjx twins register a SUBSET of the kernel-attr algos.
    The manifest is DERIVED from MJX_KERNEL_OVERLOADS + descriptor rows; assert the
    mjx set matches the golden and every twin's descriptor gate_attr/bytes_macro is
    what the golden expects — so an mjx twin can never reference a macro spelled
    differently from its pin twin, and no twin is silently added/dropped."""
    mjx_shorts = list(GRiDCodeGenerator.MJX_KERNEL_OVERLOADS.keys())
    assert set(mjx_shorts) == set(_GOLDEN_MJX_HEADS), (
        "MJX_KERNEL_OVERLOADS set drifted from the golden mjx-twin set.\n"
        f"  added:   {sorted(set(mjx_shorts) - set(_GOLDEN_MJX_HEADS))}\n"
        f"  dropped: {sorted(set(_GOLDEN_MJX_HEADS) - set(mjx_shorts))}"
    )
    for short in mjx_shorts:
        golden_gate, golden_bytes = _GOLDEN_MJX_HEADS[short]
        d = descriptor_for(short)
        assert d.has_kernel_attr, f"{short}(mjx): descriptor.has_kernel_attr is False"
        assert d.gate_attr == golden_gate, (
            f"{short}(mjx): descriptor gate_attr {d.gate_attr!r} != golden {golden_gate!r}"
        )
        assert d.bytes_macro == golden_bytes, (
            f"{short}(mjx): descriptor bytes_macro {d.bytes_macro!r} != golden {golden_bytes!r}"
        )

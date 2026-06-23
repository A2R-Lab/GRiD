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
from GRiDCodeGenerator.GRiDCodeGenerator import LAUNCH_CONFIG_ALGO_TO_SYMBOL


def test_descriptor_keys_are_registry_keys_one_to_one():
    """Every descriptor joins to exactly one registry entry and vice-versa — the
    table is TOTAL over the registry (no algo without a row, no orphan row)."""
    registry_keys = [e.key for e in ALGO_REGISTRY]
    descriptor_keys = [d.key for d in ALGO_DESCRIPTORS]
    assert descriptor_keys == registry_keys, (
        "ALGO_DESCRIPTORS must list the SAME keys in the SAME order as ALGO_REGISTRY.\n"
        f"  registry-only:   {sorted(set(registry_keys) - set(descriptor_keys))}\n"
        f"  descriptor-only: {sorted(set(descriptor_keys) - set(registry_keys))}\n"
        f"  order mismatch:  {[ (i,r,d) for i,(r,d) in enumerate(zip(registry_keys, descriptor_keys)) if r != d ][:3]}"
    )


def test_descriptors_reproduce_launch_config_algo_to_symbol():
    """Site #1: the descriptors' autotune_keys reconstruct the live
    LAUNCH_CONFIG_ALGO_TO_SYMBOL dict (bench JSON key -> grid symbol) exactly."""
    rebuilt = build_launch_config_algo_to_symbol()
    assert rebuilt == LAUNCH_CONFIG_ALGO_TO_SYMBOL, (
        "descriptor autotune_keys do not reproduce LAUNCH_CONFIG_ALGO_TO_SYMBOL.\n"
        f"  missing from rebuilt: {set(LAUNCH_CONFIG_ALGO_TO_SYMBOL.items()) - set(rebuilt.items())}\n"
        f"  extra in rebuilt:     {set(rebuilt.items()) - set(LAUNCH_CONFIG_ALGO_TO_SYMBOL.items())}"
    )


def test_carries_launch_cfg_matches_symbol_set():
    """The set of descriptors that carry a launch_cfg equals the set of distinct
    grid symbols in LAUNCH_CONFIG_ALGO_TO_SYMBOL (which algos get a baked enum)."""
    descriptor_syms = {d.key for d in ALGO_DESCRIPTORS if d.carries_launch_cfg}
    live_syms = set(LAUNCH_CONFIG_ALGO_TO_SYMBOL.values())
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


def test_descriptors_cover_mujoco_manifest_metadata():
    """Site #9: the floating mjx twin (built inside gen_init_close_grid for a
    floating robot) registers a SUBSET of the kernel-attr algos. Every mjx-twin
    entry's (short, gate_attr, bytes_macro) must match its descriptor — same
    single-source-of-truth invariant, so an mjx twin can never reference a macro
    spelled differently from its pin twin."""
    import re

    src = GRiDCodeGenerator.gen_init_close_grid.__code__
    # The mjx manifest is a local literal; introspect via the source rather than
    # running codegen. Parse the (label, short, gate, bytes) head of each tuple in
    # the `mujoco_manifest = [ ... ]` block.
    import inspect
    text = inspect.getsource(GRiDCodeGenerator.gen_init_close_grid)
    block = text.split("mujoco_manifest = [", 1)
    assert len(block) == 2, "could not locate the mujoco_manifest literal block"
    body = block[1].split("\n        ]", 1)[0]
    # match: ("label(mjx)", "short", None|"gate", "BYTES<T>()",
    pat = re.compile(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*(None|"[^"]+")\s*,\s*"([^"]+)"\s*,')
    # Keep only the manifest-head tuples (label ends in "(mjx)"); the regex also
    # catches the inner `for it in ("EULER", "SEMI_IMPLICIT_EULER", ...)` literals.
    found = [m for m in pat.findall(body) if m[0].endswith("(mjx)")]
    assert found, "no mjx manifest tuples parsed — the literal shape changed; update this test"

    for label, short, gate_raw, bytes_macro in found:
        assert label == short + "(mjx)", f"mjx label {label!r} != {short}(mjx)"
        gate = None if gate_raw == "None" else gate_raw.strip('"')
        d = descriptor_for(short)
        assert d.has_kernel_attr, f"{short}(mjx): descriptor.has_kernel_attr is False"
        assert d.gate_attr == gate, f"{short}(mjx): descriptor gate_attr {d.gate_attr!r} != mjx {gate!r}"
        assert d.bytes_macro == bytes_macro, (
            f"{short}(mjx): descriptor bytes_macro {d.bytes_macro!r} != mjx {bytes_macro!r}"
        )

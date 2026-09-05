"""Drift gate for the generated C-ABI block in wrapper_template.cu.

The block between the BEGIN/END markers is CHECKED-IN generated text
(grid_codegen/wrapper_body_gen.py, driven by ABI_SPECS). This test fails
whenever the checked-in block differs from what the emitter produces —
so editing either the table or the emitter without regenerating, or
hand-editing the block, is caught in CI. CPU-only.
"""
from __future__ import annotations

from grid_codegen import wrapper_body_gen as g


def test_generated_block_matches_emitter():
    src = g.template_path().read_text()
    for begin, end, gen in g.REGIONS:
        assert begin in src and end in src, f"markers missing: {begin[:50]}"
        checked_in = src.split(begin, 1)[1].split(end, 1)[0]
        expected = gen()
        expected_inner = expected.split(begin, 1)[1].split(end, 1)[0]
        assert checked_in == expected_inner, (
            "generated region drifted — rerun: "
            ".venv/bin/python -m grid_codegen.wrapper_body_gen")


def test_ceil_rows_match_registry():
    """Referee for the kernel_max_threads branch table: each row's autotune key
    is the descriptor table's key for that algo, the gate/enum derive from the
    short name (the emission invariant), and every row's overload list exists.
    The idsva_so dispatcher and the divider are the only non-uniform rows."""
    from grid_codegen.algo_registry import descriptor_for, launch_config_descriptors
    from grid_codegen._kernel_attrs import KERNEL_OVERLOADS
    enum_keys = {d.key for d in launch_config_descriptors()}
    specials = 0
    for key, short in g.CEIL_ROWS:
        if short in (g._CEIL_DISPATCH, g._CEIL_DIVIDER):
            specials += 1
            continue
        assert short in KERNEL_OVERLOADS, short
        assert short in enum_keys, f"{short}: no GRID_ALGO enum row"
        assert key in descriptor_for(short).autotune_keys, (
            f"{short}: branch key {key!r} not in descriptor autotune_keys")
        assert g._ceil_sig(short), short
    assert specials == 2
    # kernel overrides only name rows that exist
    row_shorts = {s for _, s in g.CEIL_ROWS}
    assert set(g.CEIL_KERNEL_OVERRIDE) <= row_shorts


def test_generated_keys_within_emitter_scope():
    """Invariant: no generated row may carry the features the emitter does
    not model yet (bespoke bodies; f_ext "produces"). Sig-forks (2a), qdd
    forks + f_ext epilogues (2b), and IT dispatch + Xtool staging (3) ARE
    modeled — an IT-dispatch row must name its launcher in IT_LAUNCHER."""
    from grid_codegen.abi_specs import ABI_SPECS
    for key in g.GENERATED_KEYS:
        s = ABI_SPECS[key]
        assert not s.body_override, key
        assert s.qdd_route in ("none", "u_slot", "flag_fork"), key
        if s.takes_dt_it:
            assert s.it_dispatch and key in g.IT_LAUNCHER, key
        assert s.f_ext_mode in ("none", "optional"), key
        assert s.template_shape in ("plain", "std5", "so4", "qdd6", "fdgrad5"), key
        if s.template_shape == "plain":
            assert s.sig_mjx_macro is None, key
        else:
            assert s.sig_mjx_macro, key

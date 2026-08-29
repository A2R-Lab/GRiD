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
    assert g.BEGIN in src and g.END in src, "markers missing from wrapper_template.cu"
    checked_in = src.split(g.BEGIN, 1)[1].split(g.END, 1)[0]
    expected = g.gen_block()
    expected_inner = expected.split(g.BEGIN, 1)[1].split(g.END, 1)[0]
    assert checked_in == expected_inner, (
        "generated block drifted — rerun: "
        ".venv/bin/python -m grid_codegen.wrapper_body_gen")


def test_generated_keys_are_tight_style():
    """Increment-1 invariant: no generated row may carry the features the
    emitter does not model yet (forks, IT dispatch, bespoke bodies)."""
    from grid_codegen.abi_specs import ABI_SPECS
    for key in g.GENERATED_KEYS:
        s = ABI_SPECS[key]
        assert not s.body_override, key
        assert s.qdd_route == "none", key
        assert not s.takes_dt_it, key
        assert s.sig_mjx_macro is None, key
        assert s.f_ext_mode == "none", key

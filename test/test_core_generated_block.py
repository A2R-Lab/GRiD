"""Drift gate for the generated pybind-method region of bindings/src/_core.cpp.

The region between the BEGIN/END markers is emitted by
grid_codegen/core_body_gen.py from ABI_SPECS (C4 arc). This asserts the
checked-in text matches a fresh regeneration, so an abi_specs/core_body_gen
edit that forgets to regenerate (or a hand-edit inside the markers) fails CI.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))


def test_core_generated_region_matches_table():
    from grid_codegen.core_body_gen import BEGIN, END, CORE_PATH, gen_region

    text = CORE_PATH.read_text()
    assert BEGIN in text and END in text, "generated-region markers missing from _core.cpp"
    start = text.index(BEGIN)
    end = text.index(END) + len(END)
    checked_in = text[start:end]
    assert checked_in == gen_region(), (
        "generated pybind region drifted — regenerate with "
        ".venv/bin/python -m grid_codegen.core_body_gen"
    )


def test_generated_methods_cover_all_specced_rows():
    from grid_codegen.abi_specs import ABI_SPECS
    from grid_codegen.core_body_gen import EXCLUDE, generated_keys

    keys = set(generated_keys())
    expected = {k for k, s in ABI_SPECS.items() if s.py_out_dims and k not in EXCLUDE}
    assert keys == expected

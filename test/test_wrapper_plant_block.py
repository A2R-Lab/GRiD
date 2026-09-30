"""Drift referee for the GENERATED grid_plant op tails of wrapper_template.cu.

M2 (canonical plan 2026-09-13): the five gated plant ops (plant_step,
plant_step_gradient, ee_pos_cost, com_cost, momentum_cost) are emitted for
BOTH the jax-FFI and torch surfaces by grid_codegen/wrapper_plant_gen.py —
one PLANT_STEP_OPS / COST_OPS row per op. Since the 2026-09-15 wrapper
window the regions are delimited by explicit BEGIN/END GENERATED marker
comments (house style; the markers themselves are hand-written and sit
OUTSIDE the emitted region). This referee enforces the do-not-hand-edit
contract: the emitter output must equal the checked-in text between the
markers byte-for-byte. On a legitimate change, edit the table / emitter and
run `.venv/bin/python -m grid_codegen.wrapper_plant_gen`.
"""
from pathlib import Path

from grid_codegen.wrapper_plant_gen import (
    _TEMPLATE,
    gen_jax_plant_tail,
    gen_torch_plant_tail,
    plant_tail_spans,
)


def _file_slice(surface: str) -> str:
    text = Path(_TEMPLATE).read_text()
    b, e = plant_tail_spans(text)[surface]
    return "\n".join(text.split("\n")[b:e + 1]) + "\n"


def test_jax_plant_tail_matches_emitter():
    assert _file_slice("jax") == gen_jax_plant_tail(), (
        "jax plant tail drifted from wrapper_plant_gen — regenerate with "
        ".venv/bin/python -m grid_codegen.wrapper_plant_gen (or fix the table)"
    )


def test_torch_plant_tail_matches_emitter():
    assert _file_slice("torch") == gen_torch_plant_tail(), (
        "torch plant tail drifted from wrapper_plant_gen — regenerate with "
        ".venv/bin/python -m grid_codegen.wrapper_plant_gen (or fix the table)"
    )


def test_spans_are_disjoint_and_ordered():
    text = Path(_TEMPLATE).read_text()
    spans = plant_tail_spans(text)
    jb, je = spans["jax"]
    tb, te = spans["torch"]
    assert jb < je < tb < te, "plant tail marker regions out of order — markers moved?"

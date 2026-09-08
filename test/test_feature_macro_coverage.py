"""No-drift gate for the wrapper's feature-macro wall (CPU-only).

The binding wrapper's C-ABI bodies are gated with plain ``#if GRID_HAS_X`` /
``#ifdef GRID_PLANT_HAS_X``; an UNDEFINED macro reads as 0 there, silently
turning a real body into an rc=3 stub. The codegen therefore emits every core
macro unconditionally (grid_codegen/_feature_macros.py) — but until this test
nothing cross-checked the two lists, so a new wrapper gate with no emitter (or
a renamed macro) would drift silently. This asserts: every GRID_HAS_* /
GRID_PLANT_HAS_* macro the wrapper conditions on has an emitter somewhere in
grid_codegen (a literal ``#define`` fragment or the CORE_HAS_MACROS table).
"""
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_WRAPPER = _REPO / "bindings" / "grid_rbd" / "wrapper_template.cu"
_CODEGEN = _REPO / "grid_codegen"

# Macros the wrapper conditions on that are NOT algorithm-availability gates
# emitted by _feature_macros/the algorithm emitters: build-mode defines injected
# by the compile command line (-D...) or emitted by dedicated codegen paths.
_NON_ALGO_GATES = {
    "GRID_RBD_WITH_MUJOCO",       # mjx-twins gate (emitted by _feature_macros, floating-only)
    "GRID_RBD_WITH_JAX",          # -D from _compile.py (jax installed at build time)
    "GRID_RBD_WITH_TORCH",        # -D from _compile.py (torch installed at build time)
    "GRID_RBD_RUNTIME_INERTIA",   # -D from _compile.py (runtime_inertia builds)
    "GRID_RBD_RUNTIME_TRANSFORM",
    "GRID_RBD_RUNTIME_JOINT_DYNAMICS",
}


def _wrapper_gated_macros():
    text = _WRAPPER.read_text()
    used = set()
    for m in re.finditer(r"#\s*(?:if|elif|ifdef|ifndef)\b(.*)", text):
        used.update(re.findall(r"\b(GRID_(?:PLANT_)?HAS_[A-Z0-9_]+)\b", m.group(1)))
        used.update(re.findall(r"\b(GRID_RBD_(?:WITH|RUNTIME)_[A-Z0-9_]+)\b", m.group(1)))
    return used


def _codegen_emitted_macros():
    from grid_codegen._feature_macros import CORE_HAS_MACROS

    emitted = {"GRID_HAS_" + suffix for suffix in CORE_HAS_MACROS}
    for py in _CODEGEN.rglob("*.py"):
        text = py.read_text()
        # literal "#define GRID_HAS_X"-style fragments inside emitted strings
        emitted.update(re.findall(r"#define (GRID_(?:PLANT_)?HAS_[A-Z0-9_]+)", text))
        emitted.update(re.findall(r"#define (GRID_RBD_WITH_[A-Z0-9_]+)", text))
    return emitted


@pytest.mark.developer_only
def test_every_wrapper_gate_has_an_emitter():
    used = _wrapper_gated_macros()
    emitted = _codegen_emitted_macros()
    missing = sorted(used - emitted - _NON_ALGO_GATES)
    assert used, "no gated macros found in wrapper_template.cu (parse broke?)"
    assert not missing, (
        "wrapper_template.cu conditions on macros no codegen emitter defines "
        f"(would silently read 0 -> rc=3 stub): {missing}"
    )


@pytest.mark.developer_only
def test_core_has_macros_wall_reaches_the_wrapper():
    """Every CORE_HAS_MACROS row must actually gate something in the wrapper —
    a stale row (algo renamed/removed) would emit a dead #define forever."""
    from grid_codegen._feature_macros import CORE_HAS_MACROS

    used = _wrapper_gated_macros()
    stale = sorted("GRID_HAS_" + s for s in CORE_HAS_MACROS
                   if "GRID_HAS_" + s not in used)
    assert not stale, f"CORE_HAS_MACROS rows unused by wrapper_template.cu: {stale}"

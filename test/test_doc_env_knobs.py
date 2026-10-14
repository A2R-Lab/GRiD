"""The Python-wrappers page's env-knob list must name every cache-key knob in
grid_codegen/env_knobs.py (2026-10-13 audit: the page said "the one list" and
omitted GRID_ENABLE_MUJOCO_KERNELS and the GRID_RBD_* key knobs)."""
import re
from pathlib import Path

from grid_codegen import env_knobs

DOC = Path(__file__).resolve().parents[1] / "docs/source/user_guide/tutorials/python_wrappers.rst"


def test_python_wrappers_page_lists_every_cache_key_knob():
    text = DOC.read_text()
    named = set(re.findall(r"GRID_[A-Z0-9_]+", text))
    expected = set(env_knobs.GENERATION_ENV_KNOBS) | set(env_knobs.BINDINGS_ENV_KNOBS)
    assert expected <= named, sorted(expected - named)

"""Generation-time env knobs have ONE list (grid_codegen/env_knobs.py) and every header
cache folds it (2026-09-24). Guards the bug class where a cache hand-listed a subset of
the knobs and served a stale header across an A/B toggle."""
from __future__ import annotations

import re
from pathlib import Path

from grid_codegen.env_knobs import GENERATION_ENV_KNOBS, generation_env

REPO = Path(__file__).resolve().parents[1]
_READ = re.compile(r"""(?:environ\.get|environ\[|getenv)\(?\s*['"](GRID_[A-Z0-9_]+)['"]""")


def _env_reads(root: Path) -> set[str]:
    names = set()
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts or p.name == "env_knobs.py":
            continue
        names.update(_READ.findall(p.read_text()))
    return names


def test_knob_list_matches_the_reads_in_grid_codegen():
    reads = _env_reads(REPO / "grid_codegen")
    assert reads == set(GENERATION_ENV_KNOBS), (
        f"grid_codegen reads {sorted(reads - set(GENERATION_ENV_KNOBS))} that env_knobs.py does not list, "
        f"or lists {sorted(set(GENERATION_ENV_KNOBS) - reads)} that nothing reads")


def test_every_header_cache_folds_the_shared_list():
    for rel in ("bindings/grid_rbd/_cache.py", "test/benchmarks/baselines/grid/run.py", "test/cuda_equivalents/cuda_harness.py"):
        src = (REPO / rel).read_text()
        assert "generation_env" in src or "GENERATION_ENV_KNOBS" in src, f"{rel} keys its header cache without the shared knob list"


def test_generation_env_reports_every_knob(monkeypatch):
    monkeypatch.setenv("GRID_FDSVA_SO_MINV_TILE", "1")
    env = generation_env()
    assert set(env) == set(GENERATION_ENV_KNOBS) and env["GRID_FDSVA_SO_MINV_TILE"] == "1"

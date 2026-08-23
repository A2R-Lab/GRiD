#!/usr/bin/env python3
"""Bijection guard: PER_ALGO_SPECS <-> benchmarkable registry keys.

The per-algo bench wrapper dispatches off ``PER_ALGO_SPECS`` (in
``test/benchmarks/baselines/grid/run.py``). If a registry algo has a real
benchmarkable kernel but no spec row, the wrapper silently skips it ("skipping
algos missing a PER_ALGO_SPECS row") and it never gets timed. If a spec row
names a key with no kernel, the bench TU fails to build.

This pins the exact set equality so neither can drift unnoticed. It is the
first cross-check of the "ONE Algo record" -- and permanently closes the
HAS-vs-emission drift class (the same class that produced the f_ext_gradient_dq
null-alloc bug: a key that exists in one table but not the other).

Run: pytest test/benchmarks/test_per_algo_specs_bijection.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from grid_codegen.algo_registry import ALGO_DESCRIPTORS, ALGO_REGISTRY

# run.py lives in a non-package dir; load it by path (same idiom as test_autotune_picker).
_RUN_PY = REPO_ROOT / "test" / "benchmarks" / "baselines" / "grid" / "run.py"
_spec = importlib.util.spec_from_file_location("grid_bench_run", _RUN_PY)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


# Registry keys whose descriptor declares NO dedicated kernel attrs
# (has_kernel_attr=False) but which ARE still benchmarked: idsva_so is a
# dispatch ALIAS that routes to the body/world-frame kernels (those carry the
# attrs), so it has a spec row and is timed as the production SO entry point.
BENCHMARKED_ATTRLESS = {"idsva_so"}

# The ONLY registry keys legitimately excluded from PER_ALGO_SPECS: has_kernel_attr=False
# composites with no standalone benchmarkable host. Sourced from run.py so there is ONE
# canonical exclusion list (that module also filters them from its sweep-time warning).
EXPECTED_SPEC_EXCLUSIONS = set(run.BENCH_EXCLUDED_ALGOS)

_REQUIRED_FIELDS = {
    "single_call", "batch_with_mem", "batch_compute_only",
    "batch_label", "gate", "shared_mem_skip",
}


def _benchmarkable_keys() -> set[str]:
    attr_true = {d.key for d in ALGO_DESCRIPTORS if d.has_kernel_attr}
    return attr_true | BENCHMARKED_ATTRLESS


def _variant_base(spec_key: str) -> str | None:
    """Timing-VARIANT spec keys (B4, @68fedbd): '<base>_mjx' rows time an
    EXISTING registry algo's kernel through its MUJOCO_OUTPUT template twin —
    same kernel, different output convention, gated on
    `GRID_HAS_<BASE> && GRID_RBD_WITH_MUJOCO`. They are deliberately NOT
    separate registry algorithms (no own KERNEL_ATTR_MANIFEST identity), so
    the bijection exempts them — but ONLY when their base key is itself a
    benchmarkable registry key with a spec row (checked below), so an
    arbitrary orphan spec key still fails loudly."""
    return spec_key.removesuffix("_mjx") if spec_key.endswith("_mjx") else None


def test_per_algo_specs_is_bijective_with_benchmarkable_registry_keys():
    benchmarkable = _benchmarkable_keys()
    spec_keys = set(run.PER_ALGO_SPECS)
    variants = {k for k in spec_keys if _variant_base(k) is not None}

    missing = benchmarkable - spec_keys   # benchmarkable but no row -> silently un-timed
    extra = (spec_keys - variants) - benchmarkable  # non-variant row w/o kernel -> build break

    assert not missing, (
        "registry keys with a benchmarkable kernel but NO PER_ALGO_SPECS row "
        f"(they would be silently skipped from timing): {sorted(missing)}")
    assert not extra, (
        "PER_ALGO_SPECS rows for non-benchmarkable registry keys "
        f"(their bench TU would fail to build): {sorted(extra)}")


def test_variant_spec_rows_are_anchored_to_benchmarkable_bases():
    """Every '<base>_mjx' variant row must anchor to a benchmarkable registry
    key that ALSO has its own spec row, and must gate on GRID_RBD_WITH_MUJOCO
    (mjx twins only exist in mujoco-enabled builds)."""
    benchmarkable = _benchmarkable_keys()
    for key, spec in run.PER_ALGO_SPECS.items():
        base = _variant_base(key)
        if base is None:
            continue
        assert base in benchmarkable, (
            f"variant spec row {key!r}: base {base!r} is not a benchmarkable "
            f"registry key — the variant times nothing real")
        assert base in run.PER_ALGO_SPECS, (
            f"variant spec row {key!r}: base {base!r} has no spec row of its own")
        assert "GRID_RBD_WITH_MUJOCO" in (spec.get("gate") or ""), (
            f"variant spec row {key!r} must gate on GRID_RBD_WITH_MUJOCO")


def test_spec_exclusions_are_exactly_the_documented_composites():
    reg_keys = {e.key for e in ALGO_REGISTRY}
    not_benchmarked = reg_keys - _benchmarkable_keys()
    assert not_benchmarked == EXPECTED_SPEC_EXCLUSIONS, (
        f"un-benchmarked registry keys changed: {sorted(not_benchmarked)} "
        f"(expected {sorted(EXPECTED_SPEC_EXCLUSIONS)}). A new algo landed -- either add "
        "a PER_ALGO_SPECS row (if it has a benchmarkable kernel) or list it in "
        "EXPECTED_SPEC_EXCLUSIONS with justification.")


def test_every_spec_key_is_a_registry_key():
    reg_keys = {e.key for e in ALGO_REGISTRY}
    # variant rows anchor to a registry key via their base (checked above)
    orphan = {k for k in set(run.PER_ALGO_SPECS) - reg_keys
              if _variant_base(k) is None}
    assert not orphan, f"PER_ALGO_SPECS rows with no ALGO_REGISTRY entry: {sorted(orphan)}"


def test_every_spec_row_has_required_fields():
    for key, spec in run.PER_ALGO_SPECS.items():
        assert _REQUIRED_FIELDS <= set(spec), (
            f"PER_ALGO_SPECS['{key}'] missing fields: {sorted(_REQUIRED_FIELDS - set(spec))}")

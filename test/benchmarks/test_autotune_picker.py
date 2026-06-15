#!/usr/bin/env python3
"""Unit tests for the joint (tier × threads) autotune picker (T5, Part 2).

These exercise the picker logic on SYNTHETIC per-tier timing data — the heavy
N=256 GPU timing sweep is deferred to the main agent's serialized perf run, so
the correctness of the argmin / cap-clipping / schema-2 emission is validated
here without touching nvcc or the GPU.

Run: pytest test/benchmarks/test_autotune_picker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# run.py lives in a non-package dir; load it by path.
import importlib.util

_RUN_PY = REPO_ROOT / "test" / "benchmarks" / "baselines" / "grid" / "run.py"
_spec = importlib.util.spec_from_file_location("grid_bench_run", _RUN_PY)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


# ---------------------------------------------------------------------------
# Pure helpers: tier caps + grid clipping
# ---------------------------------------------------------------------------
def test_tier_thread_cap_matches_codegen_formula():
    # SHARED = MAX_PERF_LEVEL_THREADS; LITE = min(2x, 768); MINIMAL = 1024.
    assert run._tier_thread_cap("shared", 512) == 512
    assert run._tier_thread_cap("lite", 512) == 768       # min(1024, 768)
    assert run._tier_thread_cap("lite", 288) == 576       # min(576, 768)
    assert run._tier_thread_cap("minimal", 512) == 1024
    # Unknown max_perf → fall back to 1024 hardware cap.
    assert run._tier_thread_cap("shared", None) == 1024


def test_clip_grid_drops_above_cap_but_keeps_one():
    grid = (32, 64, 128, 256, 512, 768)
    assert run._clip_grid_to_cap(grid, 256) == (32, 64, 128, 256)
    # Degenerate: cap below every grid point → fall back to the cap itself.
    assert run._clip_grid_to_cap((512, 768), 256) == (256,)


def test_argmin_tier_threads_picks_global_min():
    by_tier = {
        "shared":  {128: 10.0, 256: 9.0},
        "lite":    {128: 12.0},
        "minimal": {256: 8.5, 512: 11.0},
    }
    tier, threads, us = run._argmin_tier_threads(by_tier)
    assert (tier, threads, us) == ("minimal", 256, 8.5)


def test_argmin_returns_none_on_empty():
    assert run._argmin_tier_threads({}) is None
    assert run._argmin_tier_threads({"shared": {}}) is None


# ---------------------------------------------------------------------------
# Joint picker on synthetic per-tier sweeps (monkeypatch the binary sweep)
# ---------------------------------------------------------------------------
def _install_synthetic_sweep(monkeypatch, table: dict[str, dict[str, dict[int, float]]]):
    """table[tier][algo][threads] = us. The monkeypatched _sweep_one_binary
    returns the rows for the tier identified by the (sentinel) binary path,
    restricted to the threads actually requested (so cap-clipping is honored)."""
    # Map sentinel Path -> tier via the dict we hand to the picker.
    def fake_sweep(binary, base, thread_grid, target_key):
        tier = str(binary)  # we pass the tier name as the "binary path"
        out: dict[str, dict[int, float]] = {}
        for algo, by_threads in table.get(tier, {}).items():
            for th in thread_grid:
                if th in by_threads:
                    out.setdefault(algo, {})[int(th)] = by_threads[th]
        return out
    monkeypatch.setattr(run, "_sweep_one_binary", fake_sweep)


def test_joint_pick_schema2_and_winner(monkeypatch):
    # algo "fd": minimal@256 is the global best (7.0) even though shared@256=9.0.
    table = {
        "shared":  {"fd": {128: 11.0, 256: 9.0}},
        "lite":    {"fd": {128: 12.0, 256: 10.0}},
        "minimal": {"fd": {128: 8.0,  256: 7.0}},
    }
    _install_synthetic_sweep(monkeypatch, table)
    tier_binaries = {"shared": Path("shared"), "lite": Path("lite"), "minimal": Path("minimal")}
    picks = run._autotune_pick_winners(
        tier_binaries, "fixed", thread_grid=(128, 256),
        autotune_N=256, max_perf_level_threads=512, mode="batch",
    )
    assert "fd" in picks
    info = picks["fd"]
    assert info["schema"] == 2
    assert info["tier_optimal"] == "minimal"
    assert info["threads_optimal"] == 256
    assert info["us_at_optimal"] == 7.0
    # sweep carries every tier we measured
    assert set(info["sweep"]) == {"shared", "lite", "minimal"}
    assert info["sweep"]["minimal"]["256"] == 7.0
    # flat back-compat view points at the WINNING tier's per-threads sweep
    assert info["sweep_us"] == {"128": 8.0, "256": 7.0}


def test_cap_clipping_excludes_oversized_probes_for_shared(monkeypatch):
    # shared cap = MAX_PERF_LEVEL_THREADS = 256, so a 512 probe must never be
    # offered to the shared binary. We assert the picker never records shared@512
    # even though the table "would" have a (fast) reading there.
    table = {
        "shared":  {"id": {256: 10.0, 512: 1.0}},   # 512 is a trap: above shared cap
        "minimal": {"id": {256: 9.0, 512: 9.5}},
    }
    _install_synthetic_sweep(monkeypatch, table)
    tier_binaries = {"shared": Path("shared"), "minimal": Path("minimal")}
    picks = run._autotune_pick_winners(
        tier_binaries, "fixed", thread_grid=(256, 512),
        autotune_N=256, max_perf_level_threads=256, mode="batch",
    )
    info = picks["id"]
    # shared@512 must be clipped out, so the (fake) 1.0us trap can't win.
    assert "512" not in info["sweep"].get("shared", {})
    assert info["tier_optimal"] == "minimal" and info["threads_optimal"] == 256


def test_algo_with_no_readings_is_omitted(monkeypatch):
    _install_synthetic_sweep(monkeypatch, {"shared": {}, "minimal": {}})
    picks = run._autotune_pick_winners(
        {"shared": Path("shared"), "minimal": Path("minimal")},
        "fixed", thread_grid=(128, 256), autotune_N=256,
        max_perf_level_threads=512, mode="batch",
    )
    assert picks == {}


# ---------------------------------------------------------------------------
# A.7: h2_plus.fixed `id` must NEVER tune to LITE.
#
# `inverse_dynamics` has no smem spill, so SHARED and LITE share a body and
# differ only in __launch_bounds__; LITE's looser bound starves registers, so
# LITE is slower. The joint picker must land on shared or minimal. We encode
# that with a synthetic table where LITE is uniformly the slowest tier (the
# empirical signature of the bug) and assert the property.
# ---------------------------------------------------------------------------
def test_h2_plus_fixed_id_never_picks_lite(monkeypatch):
    # LITE is slowest everywhere (register starvation); SHARED and MINIMAL are
    # close and either could legitimately win — but never LITE.
    table = {
        "shared":  {"id": {256: 20.0, 384: 19.5}},
        "lite":    {"id": {256: 28.0, 384: 27.0, 512: 26.0}},  # always worst
        "minimal": {"id": {256: 19.8, 512: 19.0, 1024: 21.0}},
    }
    _install_synthetic_sweep(monkeypatch, table)
    tier_binaries = {"shared": Path("shared"), "lite": Path("lite"), "minimal": Path("minimal")}
    picks = run._autotune_pick_winners(
        tier_binaries, "fixed",
        thread_grid=(256, 384, 512, 768, 1024),
        autotune_N=256, max_perf_level_threads=384, mode="batch",
    )
    assert "id" in picks
    assert picks["id"]["tier_optimal"] in {"shared", "minimal"}, (
        f"A.7 violated: id tuned to {picks['id']['tier_optimal']!r}, expected shared/minimal"
    )


def test_single_mode_targets_single_us_key(monkeypatch):
    # In single mode the target key is single_us; the picker still works the same
    # over the synthetic sweep (which is keyed by tier, not by metric).
    captured = {}

    def fake_sweep(binary, base, thread_grid, target_key):
        captured["target_key"] = target_key
        return {"fd": {th: float(th) for th in thread_grid}}  # smaller threads win

    monkeypatch.setattr(run, "_sweep_one_binary", fake_sweep)
    picks = run._autotune_pick_winners(
        {"shared": Path("shared")}, "fixed", thread_grid=(64, 128),
        autotune_N=256, max_perf_level_threads=512, mode="single",
    )
    assert captured["target_key"] == "single_us"
    assert picks["fd"]["threads_optimal"] == 64


# ---------------------------------------------------------------------------
# Provable tier-equivalence dedup.
#
# The picker fingerprints each algo's kernel SASS per tier and, when two tiers
# emit a BYTE-IDENTICAL kernel for an algo, times one and copies the other's
# numbers (marking `tier_equiv_to`). These tests monkeypatch the two structural
# probes (`_present_algos_in_binary` + `_tier_algo_signature`) with synthetic
# signatures so the dedup decision logic is validated without nvcc/cuobjdump.
# ---------------------------------------------------------------------------
def _install_synthetic_signatures(monkeypatch, present, sig_table):
    """present: list[algo] in every binary. sig_table[tier][algo] = signature
    string (equal strings ⇒ byte-identical kernels ⇒ eligible to dedup)."""
    monkeypatch.setattr(run, "_present_algos_in_binary",
                        lambda binary, elf_text=None: list(present))

    def fake_sig(binary, algo, elf_text=None):
        return sig_table.get(str(binary), {}).get(algo)
    monkeypatch.setattr(run, "_tier_algo_signature", fake_sig)
    # The picker dumps the ELF once and threads it through; the monkeypatched
    # probes ignore it, so stub the dump to a non-None sentinel so the dedup
    # path is exercised (a None dump would treat the binary as unreadable).
    monkeypatch.setattr(run, "_cuobjdump_elf", lambda binary: "stub-elf")


def test_dedup_collapses_identical_tier_and_copies_sweep(monkeypatch):
    # Big-robot collapse: lite's `fd` kernel is byte-identical to shared's, so
    # lite is NOT re-swept — it copies shared's numbers and is marked equiv.
    # minimal differs (distinct signature) and is swept normally.
    sweep_calls = []

    def fake_sweep(binary, base, thread_grid, target_key):
        sweep_calls.append(str(binary))
        rows = {
            "shared":  {"fd": {128: 11.0, 256: 9.0}},
            "minimal": {"fd": {128: 8.5,  256: 8.0}},
        }
        out = {}
        for algo, by in rows.get(str(binary), {}).items():
            for th in thread_grid:
                if th in by:
                    out.setdefault(algo, {})[int(th)] = by[th]
        return out

    monkeypatch.setattr(run, "_sweep_one_binary", fake_sweep)
    _install_synthetic_signatures(
        monkeypatch, present=["fd"],
        sig_table={
            "shared":  {"fd": "SIG_A"},
            "lite":    {"fd": "SIG_A"},   # identical to shared
            "minimal": {"fd": "SIG_B"},   # distinct
        },
    )
    tier_binaries = {"shared": Path("shared"), "lite": Path("lite"), "minimal": Path("minimal")}
    picks = run._autotune_pick_winners(
        tier_binaries, "fixed", thread_grid=(128, 256),
        autotune_N=256, max_perf_level_threads=512, mode="batch",
    )
    # lite's binary was NEVER swept (deduped to shared); shared + minimal were.
    assert "lite" not in sweep_calls, f"lite should be deduped, but was swept: {sweep_calls}"
    assert "shared" in sweep_calls and "minimal" in sweep_calls
    info = picks["fd"]
    # All three tier columns are still populated (report stays complete).
    assert set(info["sweep"]) == {"shared", "lite", "minimal"}
    # lite's copied numbers exactly equal shared's (byte-identical kernel).
    assert info["sweep"]["lite"] == info["sweep"]["shared"]
    # The equivalence is recorded.
    assert info.get("tier_equiv_to") == {"lite": "shared"}
    # Global winner unaffected: minimal@256 (8.0) is fastest.
    assert info["tier_optimal"] == "minimal" and info["us_at_optimal"] == 8.0


def test_dedup_skips_whole_binary_when_all_algos_collapse(monkeypatch):
    # If EVERY present algo in lite is byte-identical to shared, lite's binary is
    # skipped entirely (the big-robot all-collapse case).
    swept = []

    def fake_sweep(binary, base, thread_grid, target_key):
        swept.append(str(binary))
        return {a: {int(th): 10.0 for th in thread_grid} for a in ("id", "fd")}

    monkeypatch.setattr(run, "_sweep_one_binary", fake_sweep)
    _install_synthetic_signatures(
        monkeypatch, present=["id", "fd"],
        sig_table={
            "shared": {"id": "X", "fd": "Y"},
            "lite":   {"id": "X", "fd": "Y"},   # both identical to shared
        },
    )
    picks = run._autotune_pick_winners(
        {"shared": Path("shared"), "lite": Path("lite")}, "fixed",
        thread_grid=(128, 256), autotune_N=256, max_perf_level_threads=512, mode="batch",
    )
    # lite's binary is never run (main sweep OR refinement); shared is.
    assert "lite" not in swept, f"lite binary should be skipped wholesale; swept={swept}"
    assert "shared" in swept
    for algo in ("id", "fd"):
        assert picks[algo]["sweep"]["lite"] == picks[algo]["sweep"]["shared"]
        assert picks[algo].get("tier_equiv_to") == {"lite": "shared"}


def test_dedup_does_not_collapse_distinct_signatures(monkeypatch):
    # Small-robot case: every tier has a DISTINCT signature → no dedup, all three
    # binaries swept, no tier_equiv_to recorded.
    swept = []

    def fake_sweep(binary, base, thread_grid, target_key):
        swept.append(str(binary))
        base_us = {"shared": 9.0, "lite": 10.0, "minimal": 8.0}[str(binary)]
        return {"fd": {int(th): base_us for th in thread_grid}}

    monkeypatch.setattr(run, "_sweep_one_binary", fake_sweep)
    _install_synthetic_signatures(
        monkeypatch, present=["fd"],
        sig_table={
            "shared":  {"fd": "S0"},
            "lite":    {"fd": "S1"},
            "minimal": {"fd": "S2"},
        },
    )
    picks = run._autotune_pick_winners(
        {"shared": Path("shared"), "lite": Path("lite"), "minimal": Path("minimal")},
        "fixed", thread_grid=(128, 256), autotune_N=256,
        max_perf_level_threads=512, mode="batch",
    )
    # Every tier is swept at least once (refinement may add extra calls); the
    # key property is that no tier is deduped away.
    assert set(swept) == {"shared", "lite", "minimal"}, (
        f"all tiers must be swept when signatures differ; swept={swept}"
    )
    assert "tier_equiv_to" not in picks["fd"]


def test_dedup_blind_spot_forces_sweep(monkeypatch):
    # If a present algo cannot be fingerprinted (sig None) in a tier, that tier
    # is NEVER skipped wholesale even if the fingerprintable algos all collapse.
    swept = []

    def fake_sweep(binary, base, thread_grid, target_key):
        swept.append(str(binary))
        return {a: {int(th): 10.0 for th in thread_grid} for a in ("id", "fd")}

    monkeypatch.setattr(run, "_sweep_one_binary", fake_sweep)
    _install_synthetic_signatures(
        monkeypatch, present=["id", "fd"],
        sig_table={
            "shared": {"id": "X", "fd": "Y"},
            "lite":   {"id": "X"},   # fd has NO signature (blind spot) in lite
        },
    )
    picks = run._autotune_pick_winners(
        {"shared": Path("shared"), "lite": Path("lite")}, "fixed",
        thread_grid=(128, 256), autotune_N=256, max_perf_level_threads=512, mode="batch",
    )
    # lite must still be swept (fd couldn't be proven identical).
    assert "lite" in swept
    # id WAS proven identical, so it is marked equiv even though the binary ran.
    assert picks["id"].get("tier_equiv_to") == {"lite": "shared"}
    # fd was swept fresh in lite (no equiv).
    assert "lite" not in picks["fd"].get("tier_equiv_to", {})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

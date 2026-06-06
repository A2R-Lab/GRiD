#!/usr/bin/env python3
"""Generate benchmark.md from a unified results JSON (or from results/ directory).

Usage:
    python test/benchmarks/generate_report.py
    python test/benchmarks/generate_report.py --input results/benchmark_mymachine.json
    python test/benchmarks/generate_report.py --output benchmark.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"

# Make the repo root importable when this script runs as a subprocess (the bench
# harness invokes it without inheriting PYTHONPATH).
_REPO_ROOT = THIS_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Algorithm display names, row ordering, and section grouping derive from
# GRiDCodeGenerator/algo_registry.py — that file is the single source of truth.
# To add a new algo or rename a display label, edit the registry, not this module.
from GRiDCodeGenerator.algo_registry import (
    build_display_map as _build_display,
    build_sections_map as _build_sections,
)
ALGO_DISPLAY: dict[str, str] = _build_display()
ALGO_SECTIONS: dict[str, list[str]] = _build_sections()

ROBOTS_DISPLAY = ["iiwa14", "go2", "g1", "h1_2"]
BASES = ["fixed", "floating"]
BATCH_SIZES = [16, 32, 64, 128, 256]

NOTE_SECOND_ORDER = (
    "> **Note (IDSVA_SO)**: Pinocchio's IDSVA_SO computes a rank-3 nv×nv×nv tensor on CPU "
    "— expect very slow CPU times especially for G1 (36 DOF: 36³ = 46,656 elements). "
    "The large GRiD speedup here is expected.\n\n"
    "> **Note (FDSVA_SO)**: Pinocchio has no direct FDSVA_SO; the baseline is synthesized "
    "in-harness via the Singh/Carpentier chain rule (RNEA SO + ABA derivatives + Minv). "
    "This is what any downstream pinocchio user would write."
)

NOTE_JETSON = (
    "> **Note (Unified Memory)**: On Jetson platforms, `cudaMemcpy` is a no-op for unified memory. "
    "The **with-memory** and **compute-only** numbers will be similar; compare on **compute-only**."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(input_path: Optional[Path] = None) -> dict:
    """Load a unified results JSON.  If None, use the most recent file in results/."""
    if input_path is not None:
        return json.loads(input_path.read_text())
    candidates = sorted(RESULTS_DIR.glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        # Try any JSON in results/
        candidates = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print("No result files found in results/", file=sys.stderr)
        return {}
    return json.loads(candidates[-1].read_text())


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _entry_single(entry: Optional[dict]) -> str:
    if entry is None:
        return "—"
    s = entry.get("single_us")
    if s is None:
        return "—"
    return _fmt(s.get("median") or s.get("mean"))


def _entry_batch(entry: Optional[dict], n: int, kind: str = "with_mem") -> str:
    if entry is None:
        return "—"
    key = f"batch_{n}_{kind}_us"
    b = entry.get(key)
    if b is None:
        return "—"
    return _fmt(b.get("median") or b.get("mean"))


def _entry_best(picks: Optional[dict], algo: str) -> str:
    """Render the autotuned best (tier, threads) winner µs for `algo` as
    'µs (tier@threads)'. `picks` is results[robot][base]['algo_picks']. `—` if
    no autotune pick exists for this algo/cell (autotune is opt-in; tuned at
    N=256 compute-only)."""
    if not picks:
        return "—"
    info = picks.get(algo)
    if not info:
        return "—"
    us = info.get("us_at_optimal")
    if us is None:
        return "—"
    tier = info.get("tier_optimal", "?")
    threads = info.get("threads_optimal", "?")
    return f"{_fmt(us)} ({tier}@{threads})"


def _entry_tier_from_picks(picks: Optional[dict], algo: str, tier: str) -> Optional[str]:
    """Render the best-thread autotune µs for `algo` at `tier` from the collapsed
    autotune `algo_picks` (schema 2). `picks[algo]['sweep'][tier]` is {threads: us}
    at the autotune target metric (N=256 compute-only by default); the per-tier
    column value is its min over threads. Returns None when the tier wasn't swept
    (the report then falls back / shows `—`)."""
    if not picks:
        return None
    info = picks.get(algo)
    if not info:
        return None
    by_threads = (info.get("sweep") or {}).get(tier)
    if not by_threads:
        return None
    vals = [v for v in by_threads.values() if v is not None]
    if not vals:
        return None
    return _fmt(min(vals))


def _speedup(grid_entry: Optional[dict], pin_entry: Optional[dict], n: int) -> str:
    if grid_entry is None or pin_entry is None:
        return "—"
    gk = f"batch_{n}_compute_only_us"
    pk = f"batch_{n}_with_mem_us"
    gv = (grid_entry.get(gk) or {}).get("median") or (grid_entry.get(gk) or {}).get("mean")
    pv = (pin_entry.get(pk) or {}).get("median") or (pin_entry.get(pk) or {}).get("mean")
    if gv is None or pv is None or gv == 0:
        return "—"
    return f"{pv/gv:.1f}×"


def _codegen_flag(entry: Optional[dict]) -> str:
    if entry is None:
        return ""
    flag = entry.get("codegen")
    if flag is None:
        return ""
    return " (codegen)" if flag else " (direct)"


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def _robot_rows(results: dict, algo: str, section_robots: list[str]) -> list[str]:
    """Return markdown table rows for one algorithm across robots."""
    rows = []
    for robot in section_robots:
        for base in BASES:
            grid_e = (results.get(robot, {}).get(base, {}).get("grid") or {}).get(algo)
            pin_e  = (results.get(robot, {}).get(base, {}).get("pinocchio") or {}).get(algo)
            mjx_e  = (results.get(robot, {}).get(base, {}).get("mjx") or {}).get(algo)

            single_g  = _entry_single(grid_e)
            single_p  = _entry_single(pin_e) + _codegen_flag(pin_e)
            single_m  = _entry_single(mjx_e)
            batch_g   = _entry_batch(grid_e, 256, "compute_only")
            batch_p   = _entry_batch(pin_e, 256)
            batch_m   = _entry_batch(mjx_e, 256, "compute_only")
            spdup_p   = _speedup(grid_e, pin_e, 256)
            spdup_m   = _speedup(grid_e, mjx_e, 256)

            rows.append(
                f"| {robot} | {base} | {single_g} | {single_p} | {single_m} "
                f"| {batch_g} | {batch_p} | {batch_m} | {spdup_p} | {spdup_m} |"
            )
    return rows


# ---------------------------------------------------------------------------
# Multi-version table generation (pre_glass / glass / pinocchio / mjx / frax)
# ---------------------------------------------------------------------------

MULTI_VERSION_KEYS = ("grid_pre_glass", "grid_glass",
                      "pinocchio", "mjx", "frax_cpu", "frax_gpu",
                      "bard_cpu", "bard_gpu")


def _ratio(num_entry: Optional[dict], den_entry: Optional[dict],
           num_kind: str, den_kind: str, n: int) -> str:
    """Format `den / num` as a speedup factor. <1.0× means num is slower."""
    if num_entry is None or den_entry is None:
        return "—"
    nk = f"batch_{n}_{num_kind}_us"
    dk = f"batch_{n}_{den_kind}_us"
    nv = (num_entry.get(nk) or {}).get("median") or (num_entry.get(nk) or {}).get("mean")
    dv = (den_entry.get(dk) or {}).get("median") or (den_entry.get(dk) or {}).get("mean")
    if nv is None or dv is None or nv == 0:
        return "—"
    return f"{dv/nv:.2f}×"


def _multi_version_rows_for_metric(results: dict, algo: str,
                                   section_robots: list[str],
                                   metric: str) -> list[str]:
    """metric ∈ {"single", "n16", "n256"}. Returns rows for one algo across robots
    showing only that metric, for all backend columns."""
    rows = []
    for robot in section_robots:
        for base in BASES:
            base_dict = results.get(robot, {}).get(base, {})
            pg = (base_dict.get("grid_pre_glass") or {}).get(algo)
            gl = (base_dict.get("grid_glass") or {}).get(algo)
            gl_lite = (base_dict.get("grid_glass_tier_lite") or {}).get(algo)
            gl_min  = (base_dict.get("grid_glass_tier_minimal") or {}).get(algo)
            pi = (base_dict.get("pinocchio") or {}).get(algo)
            mx = (base_dict.get("mjx") or {}).get(algo)
            # Frax columns: split into CPU + GPU since Frax advertises both as fast.
            # Back-compat: legacy JSONs with key "frax" populate frax_gpu (the prior
            # default), leaving frax_cpu as `—`.
            fx_cpu = (base_dict.get("frax_cpu") or {}).get(algo)
            fx_gpu = (base_dict.get("frax_gpu") or base_dict.get("frax") or {}).get(algo)
            # BARD columns: split into CPU + GPU like Frax (PyTorch CPU + CUDA).
            # Back-compat: legacy JSONs with a bare "bard" key populate bard_gpu.
            bd_cpu = (base_dict.get("bard_cpu") or {}).get(algo)
            bd_gpu = (base_dict.get("bard_gpu") or base_dict.get("bard") or {}).get(algo)
            picks = base_dict.get("algo_picks")  # autotune (tier×threads) winners

            # Per-tier (glass_lite / glass_min) column source. With --autotune-threads
            # a SINGLE collapsed run carries every tier's best-thread time inside
            # `algo_picks[algo]['sweep']` (at the N=256 compute-only target), so the
            # legacy separate grid_glass_tier_lite/_tier_minimal JSON keys no longer
            # exist. Source the tier columns from the sweep when picks are present;
            # otherwise (autotune OFF) fall back to the per-tier timing blocks.
            #   - shared tier ("glass" column): always the full grid_glass block.
            #   - lite/minimal: sweep-min if autotuned, else the per-tier entry.
            # The sweep only holds the N=256 target metric, so for the single-call
            # and N=16 sub-tables the lite/minimal columns show `—` under autotune
            # (no per-tier single/N16 timings were measured in the collapsed run).
            if metric == "single":
                lite_autotuned = _entry_tier_from_picks(picks, algo, "lite")
                min_autotuned  = _entry_tier_from_picks(picks, algo, "minimal")
                # The collapsed-autotune run still measures the shared-tier grid_glass
                # block in full (single/N16/N256), so the `glass` column stays real at
                # every metric. Only lite/minimal lack non-N256 timings under autotune
                # (the sweep holds the N=256 target only) → blank them when tuned.
                gl_lite_cell = "—" if lite_autotuned is not None else _entry_single(gl_lite)
                gl_min_cell  = "—" if min_autotuned  is not None else _entry_single(gl_min)
                vals = [
                    _entry_single(pg),
                    _entry_single(gl), gl_lite_cell, gl_min_cell,
                    # grid_best is tuned on the N=256 compute-only path only.
                    "—",
                    _entry_single(pi) + _codegen_flag(pi),
                    _entry_single(mx),
                    _entry_single(fx_cpu), _entry_single(fx_gpu),
                    _entry_single(bd_cpu), _entry_single(bd_gpu),
                ]
                ratio_gl_over_pg = _ratio(gl, pg, "compute_only", "compute_only", 256)
            else:
                n = 16 if metric == "n16" else 256
                lite_autotuned = _entry_tier_from_picks(picks, algo, "lite") if n == 256 else None
                min_autotuned  = _entry_tier_from_picks(picks, algo, "minimal") if n == 256 else None
                # N=256: prefer the autotune sweep-min per tier; else the per-tier
                # timing block. N=16: sweep has no N=16 data → per-tier block only.
                gl_lite_cell = (lite_autotuned if lite_autotuned is not None
                                else _entry_batch(gl_lite, n, "compute_only"))
                gl_min_cell  = (min_autotuned if min_autotuned is not None
                                else _entry_batch(gl_min, n, "compute_only"))
                vals = [
                    _entry_batch(pg, n, "compute_only"),
                    _entry_batch(gl, n, "compute_only"),
                    gl_lite_cell,
                    gl_min_cell,
                    # grid_best winner is from the N=256 autotune; show only there.
                    _entry_best(picks, algo) if n == 256 else "—",
                    _entry_batch(pi, n),
                    _entry_batch(mx, n, "compute_only"),
                    _entry_batch(fx_cpu, n, "compute_only"),
                    _entry_batch(fx_gpu, n, "compute_only"),
                    _entry_batch(bd_cpu, n, "compute_only"),
                    _entry_batch(bd_gpu, n, "compute_only"),
                ]
                ratio_gl_over_pg = _ratio(gl, pg, "compute_only", "compute_only", n)

            cells = " | ".join(vals)
            rows.append(
                f"| {robot} | {base} | {cells} | {ratio_gl_over_pg} |"
            )
    return rows


def _multi_version_robot_rows(results: dict, algo: str,
                              section_robots: list[str]) -> list[str]:
    """Legacy entry point — kept for backwards compat; calls the N=256 variant."""
    return _multi_version_rows_for_metric(results, algo, section_robots, "n256")


def _generate_multi_version_report(data: dict, output_path: Path) -> None:
    meta = data.get("metadata", {})
    results = data.get("results", {})

    gpu  = meta.get("gpu", "?")
    cc   = meta.get("compute_capability", "?")
    cuda = meta.get("cuda_version", "?")
    cpu  = meta.get("cpu", "?")
    date = meta.get("date", "?")
    host = meta.get("host", "?")
    pg_ref = meta.get("pre_glass_ref", "d2c0d18")

    lines: list[str] = [
        "# GRiD Multi-Version Benchmark Comparison",
        "",
        f"**Machine**: {host}  ",
        f"**GPU**: {gpu} (cc {cc}, CUDA {cuda})  ",
        f"**CPU**: {cpu}  ",
        f"**Date**: {date}  ",
        f"**Pre-glass ref**: `{pg_ref}` (last benchmark-capable commit before GLASS v2 work)  ",
        f"**Pinocchio**: {meta.get('pinocchio_version', '?')}",
        "",
        "All times in **µs**.",
        "",
        "Columns:",
        "- **pre_glass**: GRiD at the pre-GLASS reference. Fixed-base only "
        "(pre_glass harness does not support floating-base).",
        "- **glass**: GRiD HEAD with the pure-SIMT GLASS backend at the SHARED tier "
        "(formerly 'PERF'; max smem, lowest spill — full inner scratch in shared memory).",
        "- **glass_lite**: GRiD HEAD at the LITE tier — partial spill of cold/large "
        "buffers to L2-pinned d_workspace; trades some throughput for ~50% smem "
        "headroom so more blocks fit per SM. `—` if the algorithm has a single tier. "
        "Under `--autotune-threads` this is the best-thread N=256 time from the "
        "collapsed autotune sweep (a single run autotunes all tiers); `—` in the "
        "single-call / N=16 sub-tables (the sweep tunes only the N=256 path).",
        "- **glass_min**: GRiD HEAD at the MINIMAL tier — most aggressive spill so "
        "the kernel fits on lower-spec GPUs / leaves smem free for the caller. `—` "
        "if the algorithm has a single tier. Same autotune sourcing as glass_lite.",
        "- **grid_best**: the autotuned global winner over (tier × thread-count) at "
        "**batch N=256 compute-only**, formatted `µs (tier@threads)`. Populated only "
        "when the sweep ran with `--autotune-threads`; `—` otherwise and in the "
        "single-call / N=16 sub-tables (the autotune tunes the N=256 path).",
        "- **pin**: Pinocchio CPU reference (codegen where available).",
        "- **mjx**: MuJoCo MJX (JAX) GPU reference. Subset of algos only "
        "(id / fd / ee_pose / id_du); others render `—`.",
        "- **frax_cpu / frax_gpu**: Frax (JAX) reference (https://github.com/danielpmorton/frax) "
        "timed separately on JAX's CPU and CUDA backends — Frax advertises both as fast. "
        "Subset of algos only (id / fd / crba / minv); others render `—`.",
        "- **bard_cpu / bard_gpu**: BARD (PyTorch) reference "
        "(https://github.com/YueWang996/bard-pytorch-dynamics) timed separately on torch's "
        "CPU and CUDA backends. Subset of algos only (id / fd / crba); others render `—`. "
        "BARD times the full update_kinematics + algo pipeline per state.",
        "- **glass/pre**: N=256 compute-only ratio. **> 1.00× = HEAD is faster**; "
        "**< 1.00× = HEAD regressed**.",
        "",
        "Each algorithm gets three sub-tables: **single-call**, **batch N=16**, "
        "**batch N=256**. Same backend columns + ratio in each. Values are "
        "median (or mean) µs. GRiD/MJX/Frax numbers are batch compute-only; "
        "Pinocchio is batch with-memory (its compute/transfer aren't separable on CPU).",
        "",
        NOTE_SECOND_ORDER,
        "",
    ]

    for section_name, algos in ALGO_SECTIONS.items():
        lines += [f"## {section_name}", ""]
        for algo in algos:
            display = ALGO_DISPLAY.get(algo, algo)
            lines += [f"### {display}", ""]

            # Three sub-tables per algorithm: single | N=16 | N=256. Each is the
            # same backend-column layout (pre_glass / glass / pin / mjx / frax / bard)
            # plus the glass/pre ratio computed at that batch size.
            metric_header = {
                "single": "single-call",
                "n16":    "batch N=16",
                "n256":   "batch N=256",
            }
            col_header = (
                "| Robot | Base | pre_glass | glass | glass_lite | glass_min | grid_best "
                "| pin | mjx | frax_cpu | frax_gpu | bard_cpu | bard_gpu | glass/pre |"
            )
            col_align = (
                "|-------|------|:---------:|:-----:|:----------:|:---------:|:--------:"
                "|:---:|:---:|:--------:|:--------:|:--------:|:--------:|:---------:|"
            )
            for metric in ("single", "n16", "n256"):
                lines += [f"**{metric_header[metric]}**", ""]
                lines += [col_header, col_align]
                lines += _multi_version_rows_for_metric(results, algo, ROBOTS_DISPLAY, metric)
                lines += [""]

    output_path.write_text("\n".join(lines) + "\n")
    print(f"[generate_report] wrote {output_path}")


def _batch_table(results: dict, algo: str, robot: str, base: str) -> list[str]:
    """Return a detailed batch table (all N) for one robot/base/algo."""
    grid_e = (results.get(robot, {}).get(base, {}).get("grid") or {}).get(algo)
    pin_e  = (results.get(robot, {}).get(base, {}).get("pinocchio") or {}).get(algo)

    header = "| N | GRiD w/mem | GRiD compute | Pinocchio |"
    sep    = "|---|:---:|:---:|:---:|"
    rows = [header, sep]
    for n in BATCH_SIZES:
        gw = _entry_batch(grid_e, n, "with_mem")
        gc = _entry_batch(grid_e, n, "compute_only")
        pb = _entry_batch(pin_e, n)
        rows.append(f"| {n} | {gw} | {gc} | {pb} |")
    return rows


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(data: dict, output_path: Path) -> None:
    meta = data.get("metadata", {})
    results = data.get("results", {})

    lines: list[str] = []

    # Header
    gpu  = meta.get("gpu", "?")
    cc   = meta.get("compute_capability", "?")
    cuda = meta.get("cuda_version", "?")
    cpu  = meta.get("cpu", "?")
    date = meta.get("date", "?")
    host = meta.get("host", "?")

    lines += [
        "# GRiD Performance Benchmarks",
        "",
        f"**Machine**: {host}  ",
        f"**GPU**: {gpu} (cc {cc}, CUDA {cuda})  ",
        f"**CPU**: {cpu}  ",
        f"**Date**: {date}  ",
        f"**Pinocchio**: {meta.get('pinocchio_version', '?')}",
        "",
        "All times in **µs**.  "
        "GRiD *single*: kernel loop internal repeats, one GPU launch.  "
        "GRiD *N=256 compute*: compute-only (no cudaMemcpy).  "
        "Pinocchio *N=256*: multi-threaded CPU (codegen where available).  "
        "MJX *N=256*: vmapped JAX on GPU, compute-only.  "
        "GRiD/Pin and GRiD/MJX speedup = baseline N=256 / GRiD N=256 compute-only.",
        "",
    ]

    for section_name, algos in ALGO_SECTIONS.items():
        lines += [f"## {section_name}", ""]

        if section_name == "Second-Order":
            lines += [NOTE_SECOND_ORDER, ""]

        for algo in algos:
            display = ALGO_DISPLAY.get(algo, algo)
            lines += [f"### {display}", ""]
            lines += [
                "| Robot | Base | GRiD single | Pin single | MJX single "
                "| GRiD N=256 | Pin N=256 | MJX N=256 | GRiD/Pin | GRiD/MJX |"
            ]
            lines += [
                "|-------|------|:-----------:|:----------:|:---------:"
                "|:----------:|:---------:|:---------:|:--------:|:--------:|"
            ]
            lines += _robot_rows(results, algo, ROBOTS_DISPLAY)
            lines += [""]

    # cuRobo reference section
    lines += [
        "## cuRobo Reference",
        "",
        "cuRobo (arxiv 2603.05493) does not expose a standalone dynamics API — "
        "dynamics kernels are fused into the motion-planning optimization loop and "
        "are not independently benchmarkable. Numbers from the paper are shown below "
        "for context (Table 2 from the cuRobo paper; `compute-only` column, RTX 3090).",
        "",
        "| Algorithm | cuRobo (batch 1024, µs) | Notes |",
        "|-----------|:-----------------------:|-------|",
        "| ID | ~2.3 | Fused forward pass |",
        "| FD | ~4.1 | Fused forward pass |",
        "| ID_DU | ~8.7 | Fused Jacobian pass |",
        "",
        "> Numbers from cuRobo paper; methodology differs from GRiD/Pinocchio benchmarks above.",
        "",
    ]

    # Appendix placeholder
    lines += [
        "---",
        "",
        "## Appendix A: Mid-Range Laptop",
        "",
        "*Results pending — run `python test/benchmarks/run_benchmarks.py` on a laptop and commit the updated `benchmark.md`.*",
        "",
        "## Appendix B: High-End Jetson (AGX Orin)",
        "",
        NOTE_JETSON,
        "",
        "*Results pending.*",
        "",
        "## Appendix C: Embedded Jetson (Nano / Orin NX)",
        "",
        NOTE_JETSON,
        "",
        "*Results pending.*",
        "",
    ]

    output_path.write_text("\n".join(lines) + "\n")
    print(f"[generate_report] wrote {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark.md from results JSON")
    parser.add_argument("--input",  type=Path, default=None,
                        help="Unified results JSON (default: newest in results/)")
    parser.add_argument("--output", type=Path, default=THIS_DIR / "benchmark.md",
                        help="Output markdown file (default: test/benchmarks/benchmark.md)")
    parser.add_argument("--mode", choices=["single_version", "multi_version"],
                        default="single_version",
                        help="single_version: existing grid/pin/mjx layout. "
                             "multi_version: pre_glass/glass/pin columns + speedup ratio.")
    args = parser.parse_args()

    data = load_results(args.input)
    if not data:
        sys.exit(1)

    if args.mode == "multi_version":
        _generate_multi_version_report(data, args.output)
    else:
        generate_report(data, args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 4 tier-sweep analyzer.

Reads the merged JSON produced by run_multi_version.py --tiers perf lite minimal
and emits a focused table showing per-(robot, base, algo) timing at each tier
plus LITE/PERF and MINIMAL/PERF speedup ratios.

Usage:
    python test/benchmarks/analyze_tier_sweep.py \\
        --input test/benchmarks/results/phase3_tier_sweep_<...>/benchmark_multi_version_<host>.json \\
        [--metric n256_compute_only|single|n16_compute_only|n256_with_mem] \\
        [--output tier_validation_matrix.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TIER_KEYS = (
    ("perf",    "grid_glass"),
    ("lite",    "grid_glass_tier_lite"),
    ("minimal", "grid_glass_tier_minimal"),
)


def _us(entry, key: str):
    if not entry or not isinstance(entry, dict):
        return None
    v = entry.get(key)
    if not isinstance(v, dict):
        return None
    return v.get("median") or v.get("mean")


def _fmt_us(v):
    return f"{v:.2f}" if v is not None else "—"


def _fmt_ratio(num, den):
    if num is None or den is None or den == 0:
        return "—"
    return f"{num/den:.2f}×"


def _metric_key(metric: str) -> str:
    if metric == "single":
        return "single_us"
    return f"batch_{metric.replace('_compute_only', '')}_compute_only_us" if "compute_only" in metric \
        else f"batch_{metric.replace('_with_mem', '')}_with_mem_us"


def emit_table(results: dict, algo: str, metric_key: str, robots: list[str], bases: list[str]) -> list[str]:
    rows = []
    rows.append(f"### {algo} ({metric_key})")
    rows.append("")
    rows.append("| Robot | Base | PERF (us) | LITE (us) | MINIMAL (us) | LITE/PERF | MINIMAL/PERF | Pinocchio (us) | PERF/Pin |")
    rows.append("|---|---|---|---|---|---|---|---|---|")
    for robot in robots:
        for base in bases:
            base_dict = results.get(robot, {}).get(base, {})
            perf_e = (base_dict.get("grid_glass") or {}).get(algo)
            lite_e = (base_dict.get("grid_glass_tier_lite") or {}).get(algo)
            mini_e = (base_dict.get("grid_glass_tier_minimal") or {}).get(algo)
            pin_e  = (base_dict.get("pinocchio") or {}).get(algo)

            perf_us = _us(perf_e, metric_key)
            lite_us = _us(lite_e, metric_key)
            mini_us = _us(mini_e, metric_key)
            # Pinocchio always uses with_mem-style entries; map to its closest analog.
            pin_key = metric_key.replace("compute_only", "with_mem") if "batch_" in metric_key else metric_key
            pin_us  = _us(pin_e, pin_key)

            rows.append(
                f"| {robot} | {base} | {_fmt_us(perf_us)} | {_fmt_us(lite_us)} | {_fmt_us(mini_us)} | "
                f"{_fmt_ratio(lite_us, perf_us)} | {_fmt_ratio(mini_us, perf_us)} | "
                f"{_fmt_us(pin_us)} | {_fmt_ratio(perf_us, pin_us)} |"
            )
    rows.append("")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True, help="Merged sweep JSON")
    p.add_argument("--output", type=Path, default=None, help="Output markdown (default: stdout)")
    p.add_argument("--metric", default="n256_compute_only",
                   choices=["single", "n16_compute_only", "n256_compute_only", "n16_with_mem", "n256_with_mem"])
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    results = data.get("results", {})
    meta = data.get("metadata", {})

    if not results:
        print("[fatal] no results in JSON", file=sys.stderr)
        sys.exit(1)

    robots = list(results.keys())
    # Stable robot order matching ROBOTS tuple
    canonical = ["iiwa14", "go2", "g1", "h2_plus"]
    robots = [r for r in canonical if r in robots] + [r for r in robots if r not in canonical]
    bases = ["fixed", "floating"]

    # Collect all algos present across any tier across any (robot, base).
    algos: set[str] = set()
    for robot, by_base in results.items():
        for base, baselines in by_base.items():
            for _, key in TIER_KEYS:
                algos.update((baselines.get(key) or {}).keys())
    algos = sorted(algos)

    metric_key = _metric_key(args.metric)

    out_lines: list[str] = []
    out_lines.append(f"# Phase 4 tier validation matrix ({args.metric})")
    out_lines.append("")
    out_lines.append(f"**Input**: `{args.input.name}`")
    if meta.get("gpu"):
        out_lines.append(f"**GPU**: `{meta['gpu']}`")
    if meta.get("columns"):
        out_lines.append(f"**Columns**: `{', '.join(meta['columns'])}`")
    out_lines.append("")
    out_lines.append(
        "Ratios are tier_us / PERF_us — values >1.0 mean the tier is **slower**. "
        "PERF/Pin is GRiD's PERF speedup over Pinocchio CPU."
    )
    out_lines.append("")

    for algo in algos:
        out_lines.extend(emit_table(results, algo, metric_key, robots, bases))

    body = "\n".join(out_lines) + "\n"
    if args.output:
        args.output.write_text(body)
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(body)


if __name__ == "__main__":
    main()

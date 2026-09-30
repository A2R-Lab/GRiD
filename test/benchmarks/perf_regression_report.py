#!/usr/bin/env python3
"""Non-failing GRiD performance regression reporter.

This intentionally reports timing deltas without failing the process.  It is
meant for developer feedback after running the generated timing kernels with
GPU warmup/internal repeats.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from pathlib import Path
from statistics import median


TIMING_RE = re.compile(r"^(?P<name>.+?)\s+(?P<value>[0-9]+(?:\.[0-9]+)?)\s*us\b", re.IGNORECASE)
STATS_RE = re.compile(
    r"^(?P<name>.+?):\s+Average\[(?P<avg>[0-9]+(?:\.[0-9]+)?)us\]\s+"
    r"Std Dev \[(?P<std>[0-9]+(?:\.[0-9]+)?)us\]\s+"
    r"Min \[(?P<min>[0-9]+(?:\.[0-9]+)?)us\]\s+"
    r"Max \[(?P<max>[0-9]+(?:\.[0-9]+)?)us\]",
    re.IGNORECASE,
)
PTXAS_RE = re.compile(r"ptxas info\s*:.*", re.IGNORECASE)


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def gpu_metadata() -> dict[str, str]:
    result = _run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"])
    if result.returncode != 0:
        return {"gpu": "unknown", "compute_capability": "unknown", "driver": "unknown"}
    first = result.stdout.strip().splitlines()[0].split(",")
    return {
        "gpu": first[0].strip() if len(first) > 0 else "unknown",
        "compute_capability": first[1].strip() if len(first) > 1 else "unknown",
        "driver": first[2].strip() if len(first) > 2 else "unknown",
    }


def cuda_metadata() -> dict[str, str]:
    result = _run(["nvcc", "--version"])
    version = "unknown"
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "release" in line:
                version = line.strip()
                break
    return {"cuda": version}


def _summary_from_samples(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    med = median(values)
    min_value = values[0]
    max_value = values[-1]
    spread_pct = 0.0 if med == 0 else 100.0 * (max_value - min_value) / med
    return {
        "min_us": min_value,
        "median_us": med,
        "max_us": max_value,
        "spread_pct": spread_pct,
    }


def parse_timings(output: str) -> dict[str, dict[str, float]]:
    samples: dict[str, list[float]] = {}
    summaries: dict[str, dict[str, float]] = {}
    for line in output.splitlines():
        stripped = line.strip()
        stats_match = STATS_RE.search(stripped)
        if stats_match:
            name = stats_match.group("name").strip()
            min_value = float(stats_match.group("min"))
            max_value = float(stats_match.group("max"))
            avg_value = float(stats_match.group("avg"))
            spread_pct = 0.0 if avg_value == 0 else 100.0 * (max_value - min_value) / avg_value
            summaries[name] = {
                "min_us": min_value,
                "median_us": avg_value,
                "max_us": max_value,
                "spread_pct": spread_pct,
                "mean_us": avg_value,
                "std_us": float(stats_match.group("std")),
            }
            continue
        match = TIMING_RE.search(stripped)
        if match:
            samples.setdefault(match.group("name").strip(), []).append(float(match.group("value")))
    for name, values in samples.items():
        summaries[name] = _summary_from_samples(values)
    return summaries


def extract_ptxas(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if PTXAS_RE.search(line)]


def _baseline_median(entry) -> float | None:
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict):
        value = entry.get("median_us", entry.get("mean_us"))
        if isinstance(value, (int, float)):
            return float(value)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a timing command and report non-failing deltas against GPU-class baselines.")
    parser.add_argument("--baseline", type=Path, default=Path("test/benchmarks/perf_baselines.json"))
    parser.add_argument("--robot", default="unknown")
    parser.add_argument("--base-mode", choices=["fixed", "floating"], default="fixed")
    parser.add_argument("--profile", default="all")
    parser.add_argument("--fallback-tier", default="auto")
    parser.add_argument("--precision", default="float")
    parser.add_argument("--save", action="store_true", help="Save current timings as the baseline for this key.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Timing command to run after --, e.g. -- ./timeGRiD")
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("provide a timing command after --")

    meta = {**gpu_metadata(), **cuda_metadata(), "host": platform.node()}
    key = "|".join([
        meta["gpu"],
        meta["compute_capability"],
        args.robot,
        args.base_mode,
        args.profile,
        args.fallback_tier,
        args.precision,
    ])

    run = _run(args.command)
    print(run.stdout, end="")
    if run.stderr:
        print(run.stderr, end="")

    combined_output = run.stdout + "\n" + run.stderr
    timings = parse_timings(combined_output)
    if not timings:
        print("No timing lines matched '<name> <value>us'; report only, no failure.")
        return 0

    baselines = {}
    if args.baseline.exists():
        baselines = json.loads(args.baseline.read_text())

    old = baselines.get(key, {}).get("timings", {})
    ptxas_lines = extract_ptxas(combined_output)
    print("\nPerformance report")
    print(f"key: {key}")
    for name, summary in sorted(timings.items()):
        value = summary["median_us"]
        baseline = _baseline_median(old.get(name))
        line = (
            f"{name}: median {value:.3f} us, min {summary['min_us']:.3f} us, "
            f"max {summary['max_us']:.3f} us, spread {summary['spread_pct']:.1f}%"
        )
        if baseline and baseline > 0:
            delta = 100.0 * (value - baseline) / baseline
            print(f"{line} ({delta:+.1f}% vs baseline {baseline:.3f} us)")
        else:
            print(f"{line} (no baseline)")
    if ptxas_lines:
        print("\nptxas summary")
        for line in ptxas_lines[-20:]:
            print(line)

    if args.save:
        baselines[key] = {"metadata": meta, "timings": timings}
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(baselines, indent=2, sort_keys=True) + "\n")
        print(f"saved baseline: {args.baseline}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

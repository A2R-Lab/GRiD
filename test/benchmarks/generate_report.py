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

# Algorithm display names and row ordering
ALGO_DISPLAY: dict[str, str] = {
    "id":               "ID (Inverse Dynamics)",
    "minv":             "Minv (M⁻¹)",
    "fd":               "FD (Minv+RNEA)",
    "aba":              "ABA (Articulated Body)",
    "crba":             "CRBA",
    "id_du":            "ID_DU (∂ID/∂q,v)",
    "fd_du":            "FD_DU (∂FD/∂q,v)",
    "ee_pose":          "EE_POSE",
    "ee_pose_gradient": "EE_POSE_GRADIENT (Jacobian)",
    "idsva_so":         "IDSVA_SO (2nd-order ID)",
    "fdsva_so":         "FDSVA_SO (2nd-order FD)",
}

ALGO_SECTIONS: dict[str, list[str]] = {
    "Core Dynamics": ["id", "minv", "fd", "aba", "crba"],
    "Gradients":     ["id_du", "fd_du"],
    "Kinematics":    ["ee_pose", "ee_pose_gradient"],
    "Second-Order":  ["idsva_so", "fdsva_so"],
}

ROBOTS_DISPLAY = ["iiwa14", "go2", "g1"]
BASES = ["fixed", "floating"]
BATCH_SIZES = [16, 32, 64, 128, 256]

NOTE_SECOND_ORDER = (
    "> **Note (IDSVA_SO)**: Pinocchio's IDSVA_SO computes a rank-3 nv×nv×nv tensor on CPU "
    "— expect very slow CPU times especially for G1 (36 DOF: 36³ = 46,656 elements). "
    "The large GRiD speedup here is expected.\n\n"
    "> **Note (FDSVA_SO)**: No Pinocchio equivalent — GRiD numbers only."
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

            single_g = _entry_single(grid_e)
            single_p = _entry_single(pin_e) + _codegen_flag(pin_e)
            batch_g  = _entry_batch(grid_e, 256, "compute_only")
            batch_p  = _entry_batch(pin_e, 256)
            spdup    = _speedup(grid_e, pin_e, 256)

            rows.append(f"| {robot} | {base} | {single_g} | {single_p} | {batch_g} | {batch_p} | {spdup} |")
    return rows


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
        "Speedup = Pinocchio N=256 / GRiD N=256 compute-only.",
        "",
    ]

    for section_name, algos in ALGO_SECTIONS.items():
        lines += [f"## {section_name}", ""]

        if section_name == "Second-Order":
            lines += [NOTE_SECOND_ORDER, ""]

        for algo in algos:
            display = ALGO_DISPLAY.get(algo, algo)
            lines += [f"### {display}", ""]
            lines += ["| Robot | Base | GRiD single (µs) | Pin single (µs) | GRiD N=256 compute | Pin N=256 | Speedup |"]
            lines += ["|-------|------|:-----------------:|:---------------:|:------------------:|:---------:|:-------:|"]
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
    args = parser.parse_args()

    data = load_results(args.input)
    if not data:
        sys.exit(1)

    generate_report(data, args.output)


if __name__ == "__main__":
    main()

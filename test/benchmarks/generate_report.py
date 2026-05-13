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
# Multi-version table generation (pre_glass / glass / glass_nvidia / pinocchio)
# ---------------------------------------------------------------------------

MULTI_VERSION_KEYS = ("grid_pre_glass", "grid_glass", "grid_glass_nvidia", "pinocchio", "mjx", "frax")


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
    showing only that metric, for all 6 columns."""
    rows = []
    for robot in section_robots:
        for base in BASES:
            pg = (results.get(robot, {}).get(base, {}).get("grid_pre_glass") or {}).get(algo)
            gl = (results.get(robot, {}).get(base, {}).get("grid_glass") or {}).get(algo)
            gn = (results.get(robot, {}).get(base, {}).get("grid_glass_nvidia") or {}).get(algo)
            pi = (results.get(robot, {}).get(base, {}).get("pinocchio") or {}).get(algo)
            mx = (results.get(robot, {}).get(base, {}).get("mjx") or {}).get(algo)
            fx = (results.get(robot, {}).get(base, {}).get("frax") or {}).get(algo)

            if metric == "single":
                vals = [
                    _entry_single(pg), _entry_single(gl), _entry_single(gn),
                    _entry_single(pi) + _codegen_flag(pi),
                    _entry_single(mx), _entry_single(fx),
                ]
                ratio_gl_over_pg = _ratio(gl, pg, "compute_only", "compute_only", 256)
                ratio_gn_over_gl = _ratio(gn, gl, "compute_only", "compute_only", 256)
            else:
                n = 16 if metric == "n16" else 256
                vals = [
                    _entry_batch(pg, n, "compute_only"),
                    _entry_batch(gl, n, "compute_only"),
                    _entry_batch(gn, n, "compute_only"),
                    _entry_batch(pi, n),
                    _entry_batch(mx, n, "compute_only"),
                    _entry_batch(fx, n, "compute_only"),
                ]
                ratio_gl_over_pg = _ratio(gl, pg, "compute_only", "compute_only", n)
                ratio_gn_over_gl = _ratio(gn, gl, "compute_only", "compute_only", n)

            cells = " | ".join(vals)
            rows.append(
                f"| {robot} | {base} | {cells} | {ratio_gl_over_pg} | {ratio_gn_over_gl} |"
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
        "- **glass**: GRiD HEAD with the pure-SIMT GLASS v2 backend.",
        "- **glass_nv**: GRiD HEAD with the cuBLASDx-backed GLASS v2 backend.",
        "- **pin**: Pinocchio CPU reference (codegen where available).",
        "- **mjx**: MuJoCo MJX (JAX) GPU reference. Subset of algos only "
        "(id / fd / ee_pose / id_du); others render `—`.",
        "- **frax**: Frax (JAX) GPU reference (https://github.com/danielpmorton/frax). "
        "Subset of algos only (id / fd / crba / minv); others render `—`.",
        "- **glass/pre**: N=256 compute-only ratio. **> 1.00× = HEAD is faster**; "
        "**< 1.00× = HEAD regressed**.",
        "- **glass_nv/glass**: N=256 compute-only ratio. **> 1.00× = cuBLASDx is faster**.",
        "",
        "Each algorithm gets three sub-tables: **single-call**, **batch N=16**, "
        "**batch N=256**. Same 6 backend columns + ratios in each. Values are "
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
            # same 6-column layout (pre_glass / glass / glass_nv / pin / mjx / frax)
            # plus the glass/pre and glass_nv/glass ratios computed at that batch size.
            metric_header = {
                "single": "single-call",
                "n16":    "batch N=16",
                "n256":   "batch N=256",
            }
            col_header = (
                "| Robot | Base | pre_glass | glass | glass_nv | pin | mjx | frax "
                "| glass/pre | glass_nv/glass |"
            )
            col_align = (
                "|-------|------|:---------:|:-----:|:--------:|:---:|:---:|:----:"
                "|:---------:|:-------------:|"
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
                             "multi_version: pre_glass/glass/glass_nv/pin columns + speedup ratios.")
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

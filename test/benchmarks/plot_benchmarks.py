#!/usr/bin/env python3
"""Benchmark figures for GRiD competitive results — matplotlib, styled after the
hand-made Excel figure docs/imgs/benchmark_multi_fd_grad.png.

Two figure types:

  latency  : per-robot CPU-baseline-vs-GRiD-GPU latency across the batch sweep
             (N=16..256). GRiD GPU bars are STACKED blue=Compute + gray=I/O Overhead
             (with_mem - compute). A dashed line marks the CPU N=256 level; orange
             (compute-only) and green (with-mem) arrows annotate GRiD's speedup vs
             the CPU baseline at each N. One algorithm per figure (matches the example).

  compete   : per-robot grouped bars at a single N (default 256) — GRiD vs every
             competitor for one algorithm, log-y, ×speedup labels. The "GRiD wins
             everywhere" summary figure.

Input = a unified benchmark_multi_version_<host>.json (run_multi_version.py output:
results[robot][base][column][algo] = {batch_N_with_mem_us, batch_N_compute_only_us, ...}).
GRiD column key defaults to 'grid_glass'; CPU baseline defaults to 'pinocchio'.

Usage:
  python test/benchmarks/plot_benchmarks.py latency --input <unified.json> \
      --algo forward_dynamics_gradient --base fixed --robots iiwa14 go2 g1 \
      --out docs/imgs/grid_vs_pin_fd_grad.png
  python test/benchmarks/plot_benchmarks.py compete --input <unified.json> \
      --algo inverse_dynamics --base fixed --robots iiwa14 go2 g1 \
      --out docs/imgs/grid_compete_id.png
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NS = [16, 32, 64, 128, 256, 1024]
# Palette matched to the example figure.
C_COMPUTE = "#4472C4"   # GRiD compute (blue)
C_IO      = "#BFBFBF"   # I/O overhead (gray)
C_BASE    = "#4472C4"   # CPU baseline bars (same blue, no IO split)
ORANGE    = "#ED7D31"
GREEN     = "#548235"
# Competitor bar colors for the grouped 'compete' figure.
COMP_COLORS = {
    "grid":        C_COMPUTE,
    "pinocchio":   "#A6A6A6",
    "mjx":         "#ED7D31",
    "frax_gpu":    "#FFC000",
    "frax_cpu":    "#FFE699",
    "mujoco_warp": "#70AD47",
    "curobo":      "#C00000",
}

def _num(v):
    if isinstance(v, dict):
        v = v.get("median", v.get("mean"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def _series(cell, kind):  # kind in {"with_mem","compute_only"}
    if not isinstance(cell, dict):
        return [None]*len(NS)
    return [_num(cell.get(f"batch_{n}_{kind}_us")) for n in NS]

def _get(data, robot, base, col, algo):
    try:
        return data["results"][robot][base][col][algo]
    except (KeyError, TypeError):
        return None

def plot_latency(data, algo, base, robots, grid_col, base_col, title, out):
    fig, axes = plt.subplots(1, len(robots), figsize=(4.3*len(robots), 5.0), squeeze=False)
    axes = axes[0]
    x = np.arange(len(NS))
    for ax, robot in zip(axes, robots):
        gcell = _get(data, robot, base, grid_col, algo)
        bcell = _get(data, robot, base, base_col, algo)
        g_co = _series(gcell, "compute_only")
        g_wm = _series(gcell, "with_mem")
        b_wm = _series(bcell, "with_mem")  # CPU baseline (pin has only with_mem)
        # CPU baseline bars (left group) + GRiD GPU bars (right group), like the example.
        w = 0.38
        xb = x - 0.21
        xg = x + 0.21
        ax.bar(xb, [v or 0 for v in b_wm], w, color=C_BASE, label="CPU baseline")
        ax.bar(xg, [v or 0 for v in g_co], w, color=C_COMPUTE, label="Compute")
        io = [max(0,(wm or 0)-(co or 0)) for wm,co in zip(g_wm,g_co)]
        ax.bar(xg, io, w, bottom=[v or 0 for v in g_co], color=C_IO, label="I/O Overhead")
        # dashed CPU-N256 reference + speedup arrows at each N.
        ref = b_wm[-1]
        if ref:
            ax.axhline(ref, ls="--", lw=0.8, color="k", alpha=0.6, xmin=0.02, xmax=0.98)
            for i in range(len(NS)):
                co, wm, b = g_co[i], g_wm[i], b_wm[i]
                if co and b:
                    ax.annotate(f"{b/co:.1f}x", (xg[i], co), color=ORANGE, fontsize=8,
                                ha="center", va="bottom", fontweight="bold")
                if wm and b:
                    ax.annotate(f"{b/wm:.1f}x", (xg[i], (wm or 0)+ref*0.02), color=GREEN,
                                fontsize=8, ha="center", va="bottom", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(NS)
        ax.set_xlabel(f"N =\n{robot} (CPU | GRiD GPU)")
        ax.set_title(robot, fontsize=11)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Mean Computation Time (µs)")
    h,l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(title, y=1.04, fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.95])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[wrote {out}]")

def plot_compete(data, algo, base, robots, n, title, out):
    cols = ["grid", "pinocchio", "mjx", "frax_gpu", "mujoco_warp", "curobo"]
    grid_col = "grid_glass"
    fig, ax = plt.subplots(figsize=(1.7*len(robots)+2, 5.0))
    x = np.arange(len(robots)); w = 0.13
    present = []
    for ci, col in enumerate(cols):
        key = grid_col if col == "grid" else col
        vals = []
        for robot in robots:
            cell = _get(data, robot, base, key, algo)
            kind = "compute_only" if col == "grid" else "with_mem"
            v = _num((cell or {}).get(f"batch_{n}_{kind}_us")) if cell else None
            vals.append(v)
        if not any(vals):
            continue
        present.append(col)
        off = (len(present)-1 - 2.5) * w
        bars = ax.bar(x + off, [v or np.nan for v in vals], w,
                      color=COMP_COLORS.get(col, "#888"), label=("GRiD" if col=="grid" else col))
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(robots)
    ax.set_ylabel(f"Mean Computation Time (µs), N={n}  [log]")
    ax.set_title(title, fontsize=13)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[wrote {out}]")

def plot_summary(autotune, results_dirs, algos, base, robots, title, out):
    """N=256 grouped bars: GRiD (autotuned compute-only) vs EVERY competitor that has
    the cell (pinocchio/mjx/frax/mujoco_warp/curobo), per robot, one subplot per algo,
    log-y, ×speedup label over each competitor bar. Reads the SAME inputs as
    analyze_competitive.py so the data + canonicalization match exactly."""
    import importlib.util, sys
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("ac", here/"analyze_competitive.py")
    ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
    grid = ac.load_grid(autotune)                       # {(robot,base,algo): compute_us}
    comps = ac.load_competitors(results_dirs)           # {col: {(robot,base,algo): withmem}}
    order = ["grid","pinocchio","curobo","mujoco_warp","mjx","frax_gpu","frax_cpu"]
    cols = [c for c in order if c == "grid" or c in comps]
    fig, axes = plt.subplots(1, len(algos), figsize=(4.2*len(algos), 4.6), squeeze=False)
    axes = axes[0]
    for ax, algo in zip(axes, algos):
        x = np.arange(len(robots)); w = 0.8/len(cols); plotted = []
        for ci, col in enumerate(cols):
            vals = []
            for robot in robots:
                key = (robot, base, ac.CANON.get(algo, algo))
                vals.append(grid.get(key) if col == "grid" else comps.get(col, {}).get(key))
            if not any(vals):
                continue
            plotted.append(col)
            off = (len(plotted)-1 - (len(cols)-1)/2)*w
            ax.bar(x+off, [v or np.nan for v in vals], w,
                   color=COMP_COLORS.get(col,"#888"), label="GRiD" if col=="grid" else col)
            # speedup label (competitor / GRiD) over each competitor bar
            if col != "grid":
                for xi, robot in enumerate(robots):
                    key=(robot,base,ac.CANON.get(algo,algo)); g=grid.get(key); c=vals[xi]
                    if g and c: ax.annotate(f"{c/g:.0f}x",(x[xi]+off,c),ha="center",
                                            va="bottom",fontsize=7,rotation=90,color="#333")
        ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(robots)
        ax.set_title(algo, fontsize=11); ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", ls=":", alpha=0.4)
    axes[0].set_ylabel("Per-batch time @ N=256 (µs, log)\nGRiD=compute-only · competitors=with-mem")
    h,l = axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc="upper center",ncol=len(cols),frameon=False,bbox_to_anchor=(0.5,1.0))
    fig.suptitle(title, y=1.07, fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.93])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[wrote {out}]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["latency", "compete", "summary"])
    ap.add_argument("--input", help="unified multi_version json (latency/compete modes)")
    ap.add_argument("--autotune", help="autotune_best json (summary mode = GRiD)")
    ap.add_argument("--results", nargs="+", default=[], help="competitor result dirs (summary mode)")
    ap.add_argument("--algo")
    ap.add_argument("--algos", nargs="+", default=["inverse_dynamics","forward_dynamics","crba"])
    ap.add_argument("--base", default="fixed")
    ap.add_argument("--robots", nargs="+", default=["iiwa14","go2","g1"])
    ap.add_argument("--grid-col", default="grid_glass")
    ap.add_argument("--base-col", default="pinocchio")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "summary":
        title = a.title or f"GRiD vs competitors @ N=256 ({a.base})"
        plot_summary(a.autotune, a.results, a.algos, a.base, a.robots, title, a.out)
        return
    data = json.load(open(a.input))
    title = a.title or f"{a.algo} — GRiD vs baselines ({a.base})"
    if a.mode == "latency":
        plot_latency(data, a.algo, a.base, a.robots, a.grid_col, a.base_col, title, a.out)
    else:
        plot_compete(data, a.algo, a.base, a.robots, a.n, title, a.out)

if __name__ == "__main__":
    main()

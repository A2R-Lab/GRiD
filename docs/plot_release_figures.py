#!/usr/bin/env python3
"""Website figures from an audited release report. CPU only; no GRiD import, no GPU.

Input: a directory written by ``python -m test.benchmarks.release.report`` (its
``table.json`` carries every cell with provenance). Output, in --output:

  stacked_core        RNEA, grad RNEA, Hessian RNEA — GRiD (JAX API) as CUDA compute-only
                      call plus transfer/API increments; GPU baselines as resident +
                      full-call increment and CPU API modes as separate full calls
  stacked_all         the same for every operation that has at least one competitor
  speedup_core        the homepage summary: RNEA, grad RNEA and Hessian RNEA against every
                      baseline, one column per library — CPU libraries against GRiD's CUDA
                      host call with the copies, GPU libraries at the resident boundary
  speedup_pinocchio   GRiD kernel (compute-only) and CUDA host call (with memory)
                      against Pinocchio's code-generated and standard C++ APIs
  speedup_gpu_*       GRiD against MJX, MuJoCo Warp, BARD and Frax at three matched
                      boundaries: kernel (compute-only) vs the library's resident call,
                      JAX resident vs resident (no memory traffic on either side, the
                      framework dispatch on both), JAX full call vs full call
  table.csv, decomposition.csv   copies of the report tables
  manifest.json       report identity (table.json hash, commits, capture order,
                      accepted source drift, status counts) and output hashes

Homepage figures omit overall titles/banners; their context lives on the page.
This presentation choice does NOT approve the data. Without --approve, --output
must not be the published asset directory. ``--approve`` writes the figures
into docs/source/_static/release and marks the manifest approved; use it only
after the collection has been audited (docs/open-tasks/release_timing_handoff).
"""
import argparse
import collections
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/source/_static/release"
sys.path.insert(0, str(ROOT))

from test.benchmarks.release.report import (  # noqa: E402
    LABELS, SHORT_OP, _ratio_heatmap, banner, cell_marks, plot_stacked_comparison,
    plot_grid_composition)
from test.benchmarks.release.protocol import CORE, EXTRA, ROBOTS, WRAPPER_OPS  # noqa: E402

PINOCCHIO = ("pinocchio", "pinocchio_plain")
GPU_LIBRARIES = ("mjx", "mujoco_warp", "bard", "frax")
GRID_SIDE = {"kernel": ("grid_cuda", "resident_us", "GRiD CUDA compute-only call (launch + sync)"),
             "host": ("grid_cuda", "host_us", "GRiD CUDA host call (with memory)"),
             "jax_resident": ("grid_jax", "resident_us", "GRiD JAX resident call (device in, device out)"),
             "jax_full": ("grid_jax", "host_us", "GRiD JAX full call (host in, host out)")}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(report):
    table = json.loads((report / "table.json").read_text())
    rows = [r for r in table["cells"] if r["status"] in ("validated", "accuracy_warning")]
    return table, rows


def speedup_grid(rows, out, name, purpose, sides, comps, comp_field, title, note):
    """Heatmap grid: one row per GRiD side, one column per competitor; cells are
    competitor time / GRiD time for every (robot, operation) that both measured."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {(r["robot"], r["operation"], r["backend"], r["batch"]): r for r in rows}
    batches = sorted({r["batch"] for r in rows})
    comps = [c for c in comps if any(r["backend"] == c and r.get(comp_field) for r in rows)]
    cells = [(op, ro) for op in CORE + EXTRA for ro in ROBOTS
             if any(lookup.get((ro, op, c, b), {}).get(comp_field) for c in comps for b in batches)
             and any(lookup.get((ro, op, gb, b), {}).get(gf) for gb, gf, _ in sides for b in batches)]
    if not comps or not cells:
        return None
    labels = [f"{ro} · {SHORT_OP.get(op, op)}" for op, ro in cells]
    fig, axes = plt.subplots(len(sides), len(comps), figsize=(2.7 * len(comps) + 1.8, (.26 * len(labels) + 1.3) * len(sides)),
                             squeeze=False, sharey=True)
    for si, (gb, gf, side_label) in enumerate(sides):
        for ci, comp in enumerate(comps):
            matrix = [[(lambda g, c: c / g if (g and c) else np.nan)(
                lookup.get((ro, op, gb, b), {}).get(gf), lookup.get((ro, op, comp, b), {}).get(comp_field))
                for b in batches] for op, ro in cells]
            marks = [[cell_marks(lookup.get((ro, op, gb, b)), gf, lookup.get((ro, op, comp, b)), comp_field) for b in batches] for op, ro in cells]
            _ratio_heatmap(axes[si, ci], matrix, labels, batches, f"{side_label}\nvs {LABELS[comp]}", marks=marks)
            axes[si, ci].set_xlabel("batch")
    if purpose != "release":
        fig.suptitle(banner(purpose, title), fontsize=12)
    fig.text(.5, .015,
             "Baseline / GRiD. Blue: GRiD faster; red: baseline faster; white: 1×. Colors clipped at 100×.\n"
             "– no matched cell · * fp64 on one side · † retained fp32 accuracy warning\n"
             "~ process means span more than 1.5×. Timing boundaries and protocol are described on the page.",
             ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout(rect=(0, .13 if len(sides) == 1 else .09, 1,
                           1 if purpose == "release" else .96))
    fig.savefig(out / f"{name}.svg"); fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)
    return out / f"{name}.svg"


def speedup_core(rows, out, purpose):
    """Homepage summary: the three core operations against every baseline. CPU
    libraries (no resident boundary) are compared with GRiD's CUDA host call
    including the copies; GPU libraries at the resident boundary against GRiD's
    kernel. Each column title states its boundary."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {(r["robot"], r["operation"], r["backend"], r["batch"]): r for r in rows}
    batches = sorted({r["batch"] for r in rows})
    columns = [("pinocchio", "grid_cuda", "host_us", "host_us", "vs Pinocchio codegen\nGRiD Host Call Including I/O"),
               ("pinocchio_plain", "grid_cuda", "host_us", "host_us", "vs Pinocchio standard API\nGRiD Host Call Including I/O"),
               ("mujoco_cpu", "grid_cuda", "host_us", "host_us", "vs MuJoCo CPU\nGRiD Host Call Including I/O"),
               ("mjx", "grid_cuda", "resident_us", "resident_us", "vs MJX resident\nGRiD compute-only call"),
               ("mujoco_warp", "grid_cuda", "resident_us", "resident_us", "vs MuJoCo Warp resident\nGRiD compute-only call"),
               ("bard", "grid_cuda", "resident_us", "resident_us", "vs BARD resident\nGRiD compute-only call"),
               ("frax", "grid_cuda", "resident_us", "resident_us", "vs Frax resident\nGRiD compute-only call")]
    columns = [c for c in columns if any(r["backend"] == c[0] and r["operation"] in CORE and r.get(c[3]) for r in rows)]
    cells = [(op, ro) for op in CORE for ro in ROBOTS if any(lookup.get((ro, op, "grid_cuda", b), {}).get("host_us") for b in batches)]
    labels = [f"{ro} · {SHORT_OP.get(op, op)}" for op, ro in cells]
    fig, axes = plt.subplots(1, len(columns), figsize=(2.5 * len(columns) + 1.8, .34 * len(labels) + 2.4), squeeze=False, sharey=True)
    for ci, (comp, gb, gf, cf, title) in enumerate(columns):
        matrix = [[(lambda g, c: c / g if (g and c) else np.nan)(lookup.get((ro, op, gb, b), {}).get(gf), lookup.get((ro, op, comp, b), {}).get(cf))
                   for b in batches] for op, ro in cells]
        marks = [[cell_marks(lookup.get((ro, op, gb, b)), gf, lookup.get((ro, op, comp, b)), cf) for b in batches] for op, ro in cells]
        im = _ratio_heatmap(axes[0, ci], matrix, labels, batches, title, marks=marks)
        axes[0, ci].set_xlabel("batch")
    fig.tight_layout(rect=(0, .23, 1, 1))
    cax = fig.add_axes((.28, .145, .44, .03))
    ticks = [.01, .1, .5, 1., 2., 10., 100.]
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=np.log10(ticks))
    cb.ax.set_xticklabels(["0.01×", "0.1×", "0.5×", "1.0×", "2×", "10×", "100×"])
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Baseline faster  ←  Speedup (baseline / GRiD)  →  GRiD faster", fontsize=9)
    fig.text(.5, .015, "* fp64 · ~ variable repeats · – N/A · Log color scale clipped at 100×", ha="center", fontsize=8)
    fig.savefig(out / "speedup_core.svg"); fig.savefig(out / "speedup_core.png", dpi=150)
    plt.close(fig)
    return out / "speedup_core.svg"


API_BOUNDARIES = (
    ("CUDA Device", "grid_cuda", "resident_us", "#00693e"),
    ("C++ Host", "grid_cuda", "host_us", "#c4dd88"),
    ("NumPy", "grid_numpy", "host_us", "#267aba"),
    ("PyTorch", "grid_torch", "host_us", "#d94415"),
    ("JAX", "grid_jax", "host_us", "#8a6996"),
)


def api_boundaries(rows, out):
    """Five directly measured call totals; no C ABI or inferred overhead stacks.

    CUDA Device is the synchronized native compute-only call with data resident,
    NOT a new CUDA-event measurement. All other bars use the full host call.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    lookup = {(r["robot"], r["operation"], r["backend"], r["batch"]): r for r in rows}
    robots = [ro for ro in ROBOTS if any(r["robot"] == ro for r in rows)]
    batches = sorted({r["batch"] for r in rows})
    fig, axes = plt.subplots(len(WRAPPER_OPS), len(robots),
                             figsize=(5.3*len(robots), 3.3*len(WRAPPER_OPS)), squeeze=False)
    width = .8/len(API_BOUNDARIES)
    for oi, op in enumerate(WRAPPER_OPS):
        limits = []
        for ri, robot in enumerate(robots):
            ax = axes[oi, ri]
            for xi, batch in enumerate(batches):
                for bi, (_, backend, field, color) in enumerate(API_BOUNDARIES):
                    row = lookup.get((robot, op, backend, batch), {})
                    value = row.get(field)
                    if value is None:
                        continue
                    lo, hi = row[field.replace("_us", "_min_us")], row[field.replace("_us", "_max_us")]
                    x = xi-.4+width*(bi+.5)
                    ax.bar(x, value, width*.9, color=color)
                    limits += [lo, hi]
            ax.set(title=f"{robot} · {'RNEA' if oi == 0 else 'grad RNEA'}", xlabel="Batch size",
                   ylabel="Call wall time (µs / batch)", xticks=range(len(batches)), xticklabels=batches)
            ax.set_xlim(-.5, len(batches)-.5)
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=.15)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        if limits:
            for ax in axes[oi]:
                ax.set_ylim(min(limits)*.7, max(limits)*1.3)
    fig.legend(handles=[Patch(color=color, label=label) for label, _, _, color in API_BOUNDARIES],
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .015), fontsize=11)
    fig.tight_layout(rect=(0, .085, 1, 1))
    fig.savefig(out / "wrappers.svg")
    fig.savefig(out / "wrappers.png", dpi=150)
    plt.close(fig)
    return out / "wrappers.svg"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path, help="directory written by test.benchmarks.release.report")
    ap.add_argument("--output", type=Path, help="destination (default: <report>/website-preview; --approve forces the tracked asset directory)")
    ap.add_argument("--approve", action="store_true", help="write banner-free figures into docs/source/_static/release (after the audit)")
    args = ap.parse_args()
    report = args.report.resolve()
    out = ASSETS if args.approve else (args.output or report / "website-preview").resolve()
    if not args.approve and out == ASSETS.resolve():
        ap.error("the tracked asset directory is written only with --approve")
    purpose = "release" if args.approve else "collection"
    table, rows = load(report)
    if table.get("purpose") != "collection":
        ap.error(f"not a collection report (purpose={table.get('purpose')!r}); smoke captures are never published")
    out.mkdir(parents=True, exist_ok=True)
    outputs = []
    stacked_core = plot_stacked_comparison(rows, out, purpose, "grid_jax", CORE, "stacked_core", show_title=False, homepage_style=True)
    outputs += [stacked_core, out / "stacked_core.png"]
    stacked_all = plot_stacked_comparison(rows, out, purpose, "grid_jax", CORE + EXTRA, "stacked_all")
    outputs += [stacked_all, out / "stacked_all.png"]
    outputs += [plot_grid_composition(rows, out, purpose), out / "grid_composition.png"]
    outputs += [speedup_core(rows, out, purpose), out / "speedup_core.png"]
    outputs += [api_boundaries(rows, out), out / "wrappers.png"]
    outputs.append(speedup_grid(rows, out, "speedup_pinocchio", purpose,
        [GRID_SIDE["kernel"], GRID_SIDE["host"]], PINOCCHIO, "host_us",
        "GRiD on the GPU against Pinocchio on the CPU (medians of run means, same inputs)",
        "Pinocchio: warmed batch through a persistent C++ thread pool, best of the recorded thread counts, host arrays in and out. "
        "GRiD kernel row: data already resident on the GPU; host-call row: includes the host↔device copies."))
    outputs.append(out / "speedup_pinocchio.png")
    outputs.append(speedup_grid(rows, out, "speedup_gpu_resident", purpose,
        [GRID_SIDE["kernel"]], GPU_LIBRARIES, "resident_us",
        "GRiD kernel against GPU libraries, inputs and outputs resident on the device",
        "Competitor resident call = its warmed device-to-device evaluation including the framework's dispatch; GRiD = CUDA host call compute-only."))
    outputs.append(out / "speedup_gpu_resident.png")
    outputs.append(speedup_grid(rows, out, "speedup_gpu_jax_resident", purpose,
        [GRID_SIDE["jax_resident"]], GPU_LIBRARIES, "resident_us",
        "GRiD JAX API against GPU libraries, both resident: no memory traffic, each framework's own dispatch",
        "Both sides: inputs already on the device, outputs left on the device, synchronised; the strictest like-for-like comparison."))
    outputs.append(out / "speedup_gpu_jax_resident.png")
    outputs.append(speedup_grid(rows, out, "speedup_gpu_full", purpose,
        [GRID_SIDE["jax_full"]], GPU_LIBRARIES, "host_us",
        "GRiD JAX API against GPU libraries, complete call from host arrays to host arrays",
        "Both sides: warmed full call including transfers, dispatch and synchronisation."))
    outputs.append(out / "speedup_gpu_full.png")
    for name in ("table.csv", "decomposition.csv"):
        if (report / name).exists():
            shutil.copyfile(report / name, out / name); outputs.append(out / name)
    outputs = [o for o in outputs if o and o.exists()]
    raw = table.get("raw_records", [])
    manifest = {
        "approved": args.approve, "purpose": purpose, "report": str(report.relative_to(ROOT)) if report.is_relative_to(ROOT) else str(report),
        "table_json_sha256": sha256(report / "table.json"),
        "commits": sorted({r.get("commit") for r in raw if r.get("commit")}),
        "capture_order": table.get("capture_order"), "accepted_source_drift": table.get("accepted_source_drift"),
        "superseded_cells": len(table.get("superseded_cells", [])),
        "status_counts": dict(collections.Counter(r["status"] for r in table["cells"])),
        "outputs": {o.name: sha256(o) for o in outputs}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()

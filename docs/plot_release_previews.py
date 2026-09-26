#!/usr/bin/env python3
"""CPU-only layout previews. Historical native bars are not release comparisons.

Default regenerates from the tracked excerpt. --import-existing refreshes only
the six named local native captures. No GRiD imports or GPU work are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/source/_static/release-preview"
BATCHES = [16, 32, 64, 128, 256]
ROBOTS = {"iiwa14": "fixed", "go2": "floating", "g1": "floating"}
OPS = {"inverse_dynamics": "RNEA", "inverse_dynamics_gradient": "grad RNEA",
       "idsva_so": "Hessian RNEA"}
METHODS = {"inverse_dynamics": ["GRiD", "Pinocchio", "MJX", "MuJoCo Warp"],
           "inverse_dynamics_gradient": ["GRiD", "Pinocchio", "MJX"],
           "idsva_so": ["GRiD", "Pinocchio"]}
COLORS = {"GRiD": "#087c58", "Pinocchio": "#416ca6", "MJX": "#bb7027",
          "MuJoCo Warp": "#925da2"}
CAPTURE_DIR = "test/benchmarks/results/tier_sweep_phased_20260824_1125"


def extract_existing():
    data = {"status": "HISTORICAL NATIVE STYLE PREVIEW — NOT RELEASE COMPARISONS",
            "provenance_limits": [
                "Source metadata does not fully pin revision or dtype.",
                "Recorded means, not true medians; no uncertainty invented.",
                "Native compute/with_mem boundaries differ from proposed API wall boundaries.",
                "Competitor and interface positions are pending, not measured zeros.",
                "Read grid_glass blocks, not algo_picks autotune minima."],
            "sources": {}, "core": []}
    for robot, base in ROBOTS.items():
        for second_order in [False, True]:
            directory = ("ov2_SO_g1_floating" if robot == "g1" else f"ov1_SO_{base}") \
                if second_order else f"ov1_noSO_{base}"
            filename = f"{CAPTURE_DIR}/{directory}/{robot}_{base}_grid_glass.json"
            raw = (ROOT / filename).read_bytes()
            source = json.loads(raw)
            key = f"{robot}_{'SO' if second_order else 'first_order'}"
            data["sources"][key] = {"path": filename,
                "sha256": hashlib.sha256(raw).hexdigest(), "metadata": source["metadata"]}
            cells = source["results"][robot][base]["grid_glass"]
            for op in (["idsva_so"] if second_order else list(OPS)[:2]):
                for batch in BATCHES:
                    for boundary in ["compute_only", "with_mem"]:
                        field = f"batch_{batch}_{boundary}_us"
                        stats = (cells.get(op) or {}).get(field)
                        data["core"].append({"robot": robot, "base": base, "operation": op,
                            "batch": batch, "boundary": boundary,
                            "mean_us": stats.get("mean") if stats else None, "source": key,
                            "source_key": f"results/{robot}/{base}/grid_glass/{op}/{field}/mean"})
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "historical-excerpt.json").write_text(json.dumps(data, indent=2) + "\n")
    return data


def overhead(core, total):
    """A missing or negative delta is not zero overhead."""
    if core is None or total is None or core <= 0 or total < core:
        return None
    return total - core


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.hashsalt": "grid-release-layout", "svg.fonttype": "none"})
    outputs = []

    def save(fig, name, title, subtitle, note):
        fig.suptitle(title, x=.07, y=.985, ha="left", fontsize=17, weight="bold")
        fig.text(.07, .94, subtitle, fontsize=10, color="#925e11", weight="bold")
        fig.text(.07, .02, note, fontsize=9, color="#535d66")
        for ext in ["svg", "png"]:
            path = ASSETS / f"{name}.{ext}"
            fig.savefig(path, dpi=150, facecolor="white",
                        metadata={"Date": None} if ext == "svg" else {})
            outputs.append(path.name)
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharey="row")
    fig.subplots_adjust(left=.07, right=.98, top=.87, bottom=.16, hspace=.52, wspace=.18)
    lookup = {(c["robot"], c["operation"], c["batch"], c["boundary"]): c["mean_us"]
              for c in data["core"]}
    for row, (op, title) in enumerate(OPS.items()):
        methods = METHODS[op]
        width = .8 / len(methods)
        row_values = [c["mean_us"] for c in data["core"]
                      if c["operation"] == op and c["mean_us"] is not None and c["mean_us"] > 0]
        row_limits = (min(row_values) * .5, max(row_values) * 1.8) if row_values else (1, 10)
        for col, (robot, base) in enumerate(ROBOTS.items()):
            ax = axes[row, col]
            for k, batch in enumerate(BATCHES):
                x = k - .4 + width / 2
                core = lookup.get((robot, op, batch, "compute_only"))
                total = lookup.get((robot, op, batch, "with_mem"))
                delta = overhead(core, total)
                if core is not None and core > 0:
                    ax.bar(x, core, width * .9, color=COLORS["GRiD"], zorder=3)
                if delta is not None:
                    ax.bar(x, delta, width * .9, bottom=core, color="#c4c8cc",
                           edgecolor="#646970", hatch="////", linewidth=.5, zorder=3)
                elif total is not None and total > 0:
                    ax.plot(x, total, marker="x", color="#b33d35", zorder=4)
                for method_idx in range(1, len(methods)):
                    # Axis-coordinate text has no quantitative height or implied timing.
                    ax.text(x + method_idx * width, .025, "P", ha="center", fontsize=7,
                            color=COLORS[methods[method_idx]], transform=ax.get_xaxis_transform())
            ax.set(yscale="log", ylim=row_limits, xlim=(-.5, 4.5),
                   xticks=np.arange(5), xticklabels=BATCHES,
                   title=f"{title} · {robot} ({base})", xlabel="Batch size")
            if col == 0:
                ax.set_ylabel("Historical native mean µs / batch")
            ax.grid(axis="y", alpha=.2, zorder=0)
            ax.text(.02, .96, " | ".join(methods), transform=ax.transAxes, va="top",
                    fontsize=8, color="#535d66")
    fig.legend(handles=[Patch(color=COLORS["GRiD"], label="Historical GRiD compute"),
        Patch(facecolor="#c4c8cc", edgecolor="#646970", hatch="////", label="Historical with_mem − compute")],
        loc="lower center", bbox_to_anchor=(.5, .085), ncol=2, frameon=False)
    save(fig, "core", "Core comparison layout · three operations, three robots",
         "HISTORICAL GRiD BARS ONLY · P = COMPARATOR PENDING · NO RELEASE SPEEDUPS",
         "Five clusters per panel; positions follow the method order above each panel. Shared log scales; segment heights are not cost fractions.\n"
         "Native archive illustrates bar styling only. New collection will use matched API wall-time boundaries and dtype.\n"
         "No mixed-date competitor bars. Missing measurements stay missing; a red × marks an unstackable recorded total.")

    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))
    fig.subplots_adjust(left=.07, right=.98, top=.82, bottom=.16, hspace=.65, wspace=.18)
    for row, title in enumerate(["RNEA", "grad RNEA"]):
        for col, (robot, base) in enumerate(ROBOTS.items()):
            ax = axes[row, col]
            ax.set(xticks=np.arange(5), xticklabels=BATCHES, yticks=[], ylim=(0, 1),
                   xlim=(-.5, 4.5), title=f"{title} · {robot} ({base})", xlabel="Batch size")
            if col == 0:
                ax.set_ylabel("Host-to-host µs / batch · pending")
            ax.text(.5, .57, "CUDA C++  |  NumPy/pybind  |  JAX  |  PyTorch\n\n"
                    "Native reference + gray hatched interface delta\n\nNo matched timings collected",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#535d66")
    save(fig, "wrappers-layout", "Interface comparison layout · same generated kernels",
         "LAYOUT ONLY · NO SYNTHETIC TIMINGS OR IMPLIED SPEEDUPS",
         "Four bars per batch after collection. Match inputs, outputs, precision, launch configuration and allocation policy.\n"
         "Gray hatching = wrapper total minus matched native host-to-host reference, not pure Python time.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    fig.subplots_adjust(left=.08, right=.97, top=.76, bottom=.2, wspace=.25)
    axes[0].set(title="Timing panel", xlabel="Batch size · 16 / 32 / 64 / 128 / 256",
                ylabel="Synchronized µs / batch", xticks=[], yticks=[])
    axes[0].text(.5, .5, "Single fine tier vs coarse-to-fine\n\nNo timing data supplied",
                 transform=axes[0].transAxes, ha="center", va="center", color="#535d66")
    axes[1].axis("off")
    axes[1].set_title("Coverage and agreement")
    axes[1].text(.05, .9, "Pinned robot / scene / geometry hashes\n\nFree / colliding / near-contact cases\n\n"
                 "Fine-tier verdict agreement\n\nUnresolved geometry count", va="top",
                 transform=axes[1].transAxes)
    save(fig, "collisions-layout", "Collision pilot · deferred follow-up",
         "LAYOUT ONLY · CURRENT TIMING WRAPPER STILL NEEDED",
         "No synthetic timings. Fine-representation agreement is not exact-mesh validation.")
    manifest = {"status": data["status"],
        "input_sha256": hashlib.sha256((ASSETS / "historical-excerpt.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "outputs": {name: hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() for name in outputs}}
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {len(outputs)} CPU-only preview assets")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-existing", action="store_true")
    args = parser.parse_args()
    data = extract_existing() if args.import_existing else json.loads((ASSETS / "historical-excerpt.json").read_text())
    render(data)


if __name__ == "__main__":
    main()

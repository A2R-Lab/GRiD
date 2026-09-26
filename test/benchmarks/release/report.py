"""Export validated captures to a full table and DRAFT clustered-bar figures.

No capture is approved for publication by this tool. Missing, failed, mixed-
contract, or incomplete repeat groups never receive an invented timing.
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import html
import json
from pathlib import Path
import statistics

from .protocol import CORE, PRIMARY, ROBOTS, WRAPPER_OPS, WRAPPERS, digest, overhead, write_json
from .protocol import TIMED_STATUSES, cell_accuracy_status, ACCURACY_FOOTNOTE

LABELS = {"grid_cuda": "GRiD CUDA host call", "grid_native": "GRiD C ABI", "grid_numpy": "GRiD NumPy", "grid_jax": "GRiD JAX",
          "grid_torch": "GRiD PyTorch", "pinocchio": "Pinocchio CPU", "mjx": "MJX",
          "mujoco_warp": "MuJoCo Warp", "mujoco_cpu": "MuJoCo CPU", "bard": "BARD", "frax": "Frax"}
OP_LABELS = dict(zip(CORE, ("RNEA", "grad RNEA", "Hessian RNEA")))


def records(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    if plan["purpose"] == "preparation":
        raise ValueError("Preparation is not a validated timing capture; run smoke or collection before reporting")
    manifest = directory / "manifest.json"
    if manifest.exists():
        for name, expected_hash in json.loads(manifest.read_text()).items():
            path = directory / name
            if path.parent.resolve() != directory.resolve() or digest(path) != expected_hash:
                raise ValueError(f"Manifest path/hash mismatch: {path}")
    result_path = directory / "results.json"
    completed = json.loads(result_path.read_text())["jobs"] if result_path.exists() else []
    by_key = {(j["robot"], j["backend"], j["operation"], j["repeat"]): j for j in completed}
    for job in plan["jobs"]:
        for repeat in range(plan["repeats"]):
            key = (job["robot"], job["backend"], job["operation"], repeat)
            done = by_key.get(key, {})
            capture = {}
            if done.get("capture"):
                path = directory / done["capture"]
                if path.parent.resolve() != directory.resolve() or digest(path) != done["sha256"]:
                    raise ValueError(f"Capture path/hash mismatch: {path}")
                capture = json.loads(path.read_text())
            cells = {c["batch"]: c for c in capture.get("cells", [])}
            for batch in plan["batches"]:
                c = cells.get(batch, {})
                adapter = c.get("adapter", capture.get("adapter", {}))
                available = job.get("unavailable")
                status = c.get("status", done.get("status", "not_collected"))
                reason = c.get("reason", capture.get("setup_error", done.get("reason", "")))
                if not c and available:
                    status, reason = available.split(": ", 1)
                if status == "completed":
                    status, reason = "error", "completed worker omitted this batch"
                valid = status == "validated" and c.get("comparison_eligible") and c.get("oracle_agreement", {}).get("passed")
                policy = plan.get("accuracy_policy", "strict")
                version = plan.get("accuracy_policy_version", 1)
                if status == "accuracy_warning" or version >= 2:
                    valid = (capture.get("accuracy_policy", "strict") == policy
                        and capture.get("accuracy_policy_version", 1) == version
                        and c.get("comparison_eligible") and status in TIMED_STATUSES
                        and cell_accuracy_status(c, policy, job["operation"], adapter.get("dtype"), version=version) == status)
                if status == "accuracy_warning":
                    reason = c.get("accuracy_warning", "Retained with accuracy warning")
                checks = [c.get(k, {}) for k in ("oracle_agreement", "post_timing_agreement",
                    "resident_oracle_agreement", "resident_post_timing_agreement")]
                variation = [c.get(k, {}) for k in ("repeatability_agreement", "boundary_agreement",
                    "resident_repeatability_agreement", "post_boundary_agreement")]
                provenance = plan.get("provenance", {})
                yield {"robot": job["robot"], "operation": job["operation"], "backend": job["backend"],
                    "batch": batch, "repeat": repeat, "expected_repeats": plan["repeats"],
                    "status": status, "reason": reason, "purpose": plan["purpose"],
                    "dtype": adapter.get("dtype", "unknown"), "method": adapter.get("method", "unknown"),
                    "accuracy_policy": policy,
                    "accuracy_policy_version": version,
                    "warning_checks": c.get("warning_checks", []),
                    "variation_max_abs_error": max((b["max_abs"] for ck in variation for b in ck.get("blocks", [])), default=None),
                    "variation_relative_l2_error": max((b["relative_l2"] for ck in variation for b in ck.get("blocks", [])), default=None),
                    "host_us": c.get("host_to_host", {}).get("mean_us") if valid else None,
                    "resident_us": c.get("resident", {}).get("mean_us") if valid else None,
                    "resident_eager_us": c.get("resident_eager", {}).get("mean_us") if valid else None,
                    "threads": adapter.get("active_cpu_threads", adapter.get("threads_per_block")),
                    "urdf_sha256": capture.get("fixture", {}).get("urdf_sha256"),
                    "input_values_sha256": capture.get("input_values_sha256"),
                    "contract": json.dumps({"commit": provenance.get("commit"),
                        "sources": capture.get("collector_sources", provenance.get("collector_sources")), "packages": provenance.get("packages"),
                        "arithmetic_policy": plan.get("arithmetic_policy"),
                        "accuracy_policy": policy, "fd_warning_max_relative_l2": plan.get("fd_warning_max_relative_l2"),
                        "accuracy_policy_version": version,
                        "code_diff": provenance.get("diff_sha256"), "submodules": provenance.get("submodules"),
                        "gpu": provenance.get("gpu"), "cpu": provenance.get("cpu"),
                        "iterations": plan["iterations"], "warmups": plan["warmups"],
                        "cpu_threads": plan.get("cpu_threads")}, sort_keys=True),
                    "max_abs_error": max((b["max_abs"] for ck in checks for b in ck.get("blocks", [])), default=None),
                    "relative_l2_error": max((b["relative_l2"] for ck in checks for b in ck.get("blocks", [])), default=None),
                    "bad_entries": max(sum(b["bad_entries"] for b in ck.get("blocks", [])) for ck in checks),
                    "entries": max(sum(b.get("entries", 0) for b in ck.get("blocks", [])) for ck in checks),
                    "capture": str(directory / done["capture"]) if done.get("capture") else str(directory / "plan.json")}


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("robot", "operation", "backend", "batch"))].append(row)
    output = []
    for key, group in sorted(groups.items()):
        first = group[0]
        if len({r["repeat"] for r in group}) != len(group):
            raise ValueError(f"Duplicate repeat for {key}; do not merge reruns as independent repeats")
        row = {k: first[k] for k in ("robot", "operation", "backend", "batch", "purpose", "dtype", "method")}
        row.update(repeats=len(group), status="validated", reason="", host_us=None, resident_us=None,
                   resident_eager_us=None,
                   threads="/".join(sorted({str(r.get("threads")) for r in group if r.get("threads") is not None})) or None,
                   accuracy_policy=first.get("accuracy_policy", "strict"),
                   accuracy_policy_version=first.get("accuracy_policy_version", 1),
                   variation_max_abs_error=max((r["variation_max_abs_error"] for r in group if r.get("variation_max_abs_error") is not None), default=None),
                   variation_relative_l2_error=max((r["variation_relative_l2_error"] for r in group if r.get("variation_relative_l2_error") is not None), default=None),
                   max_bad_entries=max(r.get("bad_entries", 0) for r in group),
                   entries=max(r.get("entries", 0) for r in group),
                   host_min_us=None, host_max_us=None, overhead_us=None, boundary_flag="",
                   max_abs_error=max((r["max_abs_error"] for r in group if r["max_abs_error"] is not None), default=None),
                   relative_l2_error=max((r["relative_l2_error"] for r in group if r["relative_l2_error"] is not None), default=None))
        contracts = {(r["contract"], r["urdf_sha256"], r["input_values_sha256"], r["dtype"], r["method"], r["purpose"], r["expected_repeats"]) for r in group}
        if len(contracts) != 1:
            row.update(status="contract_mismatch", reason="repeat hardware/software/input/precision contracts differ")
        elif len(group) != first["expected_repeats"]:
            row.update(status="incomplete", reason="not all requested repeats are present")
        elif any(r["status"] not in TIMED_STATUSES or r["host_us"] is None for r in group):
            row.update(status="; ".join(sorted({r["status"] for r in group})),
                       reason="; ".join(sorted({r["reason"] for r in group if r["reason"]})))
        else:
            if any(r["status"] == "accuracy_warning" for r in group):
                row.update(status="accuracy_warning", reason="Entrywise oracle/variation gate exceeded; explicitly retained with errors reported")
            total = [r["host_us"] for r in group]
            row.update(host_us=statistics.median(total), host_min_us=min(total), host_max_us=max(total))
            if all(r.get("resident_eager_us") is not None for r in group):
                row["resident_eager_us"] = statistics.median(r["resident_eager_us"] for r in group)
            if all(r["resident_us"] is not None for r in group):
                row["resident_us"] = statistics.median(r["resident_us"] for r in group)
                if any(overhead(r["host_us"], r["resident_us"]) is None for r in group):
                    row["boundary_flag"] = "negative total-minus-resident in at least one repeat; recollect"
                else:
                    row["overhead_us"] = overhead(row["host_us"], row["resident_us"])
        output.append(row)
    # Cross-backend comparisons also need matched hardware/software/fixtures.
    # Never place bars from unrelated captures beside each other silently.
    comparisons = defaultdict(list)
    for r in rows:
        if r["status"] in TIMED_STATUSES:
            comparisons[(r["robot"],r["operation"],r["batch"])].append(r)
    for r in output:
        peers = comparisons[(r["robot"],r["operation"],r["batch"])]
        if len({(p["contract"],p["urdf_sha256"],p["input_values_sha256"]) for p in peers}) > 1:
            r.update(status="contract_mismatch", reason="cross-backend hardware/software/input contracts differ",
                host_us=None,resident_us=None,host_min_us=None,host_max_us=None,overhead_us=None)
    return output


# Overhead decomposition of GRiD's own surfaces around the CUDA host call.
# Each term is a difference of two measured means of the same cell; a negative
# difference is reported as None with a flag, never clamped.
DECOMPOSITION = (
    ("kernel_compute_us", "grid_cuda", "resident_us", None, None),
    ("memory_traffic_us", "grid_cuda", "host_us", "grid_cuda", "resident_us"),
    ("c_abi_staging_us", "grid_native", "host_us", "grid_cuda", "host_us"),
    ("numpy_python_us", "grid_numpy", "host_us", "grid_native", "host_us"),
    ("jax_dispatch_us", "grid_jax", "resident_us", "grid_cuda", "resident_us"),
    ("jax_round_trip_us", "grid_jax", "host_us", "grid_jax", "resident_us"),
    ("torch_dispatch_us", "grid_torch", "resident_us", "grid_cuda", "resident_us"),
    ("torch_round_trip_us", "grid_torch", "host_us", "grid_torch", "resident_us"),
)


def decompose(rows):
    lookup = {(r["robot"], r["operation"], r["backend"], r["batch"]): r for r in rows}
    keys = sorted({(r["robot"], r["operation"], r["batch"]) for r in rows if r["backend"] == "grid_cuda"})
    output = []
    for robot, op, batch in keys:
        row = {"robot": robot, "operation": op, "batch": batch, "flags": []}
        for name, backend, field, base_backend, base_field in DECOMPOSITION:
            value = lookup.get((robot, op, backend, batch), {}).get(field)
            if base_backend is None:
                row[name] = value
                continue
            base = lookup.get((robot, op, base_backend, batch), {}).get(base_field)
            row[name] = overhead(value, base)
            if value is not None and base is not None and row[name] is None:
                row["flags"].append(f"{name}: negative difference ({value:.1f} < {base:.1f}); recollect")
        row["flags"] = "; ".join(row["flags"])
        output.append(row)
    return output


def plot(rows, directory, kind, purpose):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ops = CORE if kind == "core" else WRAPPER_OPS
    selected = lambda op: PRIMARY[op] if kind == "core" else WRAPPERS
    robots = [r for r in ROBOTS if any(x["robot"] == r for x in rows)]
    batches = sorted({r["batch"] for r in rows})
    lookup = {(r["robot"], r["operation"], r["backend"], r["batch"]): r for r in rows}
    fig, axes = plt.subplots(len(ops), len(robots), figsize=(max(10,5.3*len(robots)), 3.3*len(ops)), squeeze=False)
    colors = {b: plt.get_cmap("tab10")(i) for i,b in enumerate(LABELS)}
    for oi, op in enumerate(ops):
        row_values = [v for r in rows if r["operation"] == op and r["backend"] in selected(op)
                      for v in (r.get("resident_us"),r.get("host_min_us"),r.get("host_max_us")) if v is not None and v > 0]
        for ri, robot in enumerate(robots):
            ax = axes[oi,ri]
            backends = selected(op)
            width = .8/len(backends)
            for bi, backend in enumerate(backends):
                for xi, batch in enumerate(batches):
                    x = xi-.4+width*(bi+.5)
                    row = lookup.get((robot,op,backend,batch), {})
                    total, resident, cap = (row.get(k) for k in ("host_us", "resident_us", "overhead_us"))
                    if total is None:
                        ax.text(x, .025, "N/C" if not row else row.get("status", "N/A").replace("_", " "),
                                rotation=90, ha="center", va="bottom", fontsize=5.5, transform=ax.get_xaxis_transform())
                        continue
                    stacked = cap is not None and resident is not None
                    ax.bar(x, resident if stacked else total, width*.9, color=colors[backend])
                    if stacked:
                        ax.bar(x, cap, width*.9, bottom=resident, facecolor=".86", edgecolor=".4", hatch="////", linewidth=.4)
                    ax.errorbar(x, total, yerr=[[total-row["host_min_us"]],[row["host_max_us"]-total]], color="black", capsize=2, linewidth=.7)
                    if row.get("boundary_flag"):
                        ax.plot(x, total, "v", color="red", markersize=5)
                    if row.get("dtype") == "float64":
                        ax.annotate("*", (x,total), xytext=(0,3), textcoords="offset points", ha="center")
                    if row.get("status") == "accuracy_warning":
                        ax.annotate("†", (x,total), xytext=(0,3), textcoords="offset points", ha="center")
            ax.set(title=f"{robot} · {OP_LABELS[op]}", xticks=range(len(batches)), xticklabels=batches, xlabel="Batch size", ylabel="µs / complete batch")
            ax.set_xlim(-.5,len(batches)-.5)
            if any(r["host_us"] for r in rows if r["robot"] == robot and r["operation"] == op):
                ax.set_yscale("log")
                if row_values:
                    ax.set_ylim(min(row_values)*.5,max(row_values)*1.5)
            ax.grid(axis="y", alpha=.15)
    used = list(dict.fromkeys(b for op in ops for b in selected(op)))
    handles = [Patch(color=colors[b], label=LABELS[b]) for b in used]
    handles.append(Patch(facecolor=".86", edgecolor=".4", hatch="////", label="Full-call minus resident API wall time"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(.5,.015))
    fig.suptitle(f"{'SMOKE TEST — NOT PERFORMANCE EVIDENCE' if purpose == 'smoke' else 'DRAFT — UNREVIEWED COLLECTION'}\n{kind.title()} comparison · median of run means; whiskers show run-mean range", fontsize=12)
    fig.text(.5,.095,"* fp64 arithmetic exception. Red triangle: negative timing delta, not stacked. N/C: not collected.\nGRiD CUDA host call: base = compute-only kernel launch, cap = H2D/D2H of one call. Other stacked bases include resident API dispatch. Unstacked bars are full-call only.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.15,1,.93))
    fig.savefig(directory / f"{kind}.svg")
    fig.savefig(directory / f"{kind}.png", dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("captures", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    raw = [r for directory in args.captures for r in records(directory)]
    if not raw:
        ap.error("No planned cells")
    purposes = {r["purpose"] for r in raw}
    if len(purposes) != 1:
        ap.error("Do not mix smoke and collection captures")
    rows = aggregate(raw)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "table.json", {"publication_approved": False, "purpose": raw[0]["purpose"], "cells": rows, "raw_records": raw})
    with (args.output / "table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for kind in ("core", "wrappers"):
        plot(rows,args.output,kind,raw[0]["purpose"])
    decomposition = decompose(rows)
    write_json(args.output / "decomposition.json", {"publication_approved": False, "cells": decomposition,
        "definition": {name: (f"{LABELS[b]} {f}" if bb is None else f"{LABELS[b]} {f} minus {LABELS[bb]} {bf}")
                       for name, b, f, bb, bf in DECOMPOSITION}})
    if decomposition:
        with (args.output / "decomposition.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(decomposition[0]))
            writer.writeheader(); writer.writerows(decomposition)
    cols = ["robot", "operation", "backend", "batch", "dtype", "status", "accuracy_policy", "accuracy_policy_version", "host_us", "resident_us", "resident_eager_us", "overhead_us", "threads", "max_abs_error", "relative_l2_error", "variation_max_abs_error", "variation_relative_l2_error", "max_bad_entries", "entries", "boundary_flag", "reason"]
    esc = lambda v: html.escape(str(v)) if v is not None else "—"
    table = "<tr>"+"".join(f"<th>{c}</th>" for c in cols)+"</tr>"
    table += "".join("<tr>"+"".join(f"<td>{esc(r[c])}</td>" for c in cols)+"</tr>" for r in rows)
    dcols = ["robot", "operation", "batch"] + [d[0] for d in DECOMPOSITION] + ["flags"]
    dtable = "<tr>"+"".join(f"<th>{c}</th>" for c in dcols)+"</tr>"
    dtable += "".join("<tr>"+"".join(f"<td>{esc(round(r[c], 1) if isinstance(r[c], float) else r[c])}</td>" for c in dcols)+"</tr>" for r in decomposition)
    (args.output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>GRiD benchmark draft</title><style>body{font:14px system-ui;margin:2rem}td,th{padding:.5rem;border:1px solid #ddd}table{border-collapse:collapse}img{max-width:100%}</style><h1>DRAFT benchmark audit</h1><p>Not publication-approved. Smoke captures are functional checks, not performance evidence. All times are microseconds per batch; summary is median of run means. Gray caps are paired boundary differences, not isolated transfer timings. Precision exceptions are explicit. No speedup claims are generated.</p><a href="table.csv">CSV</a> · <a href="table.json">Full provenance and error metrics</a> · <a href="decomposition.csv">Overhead decomposition CSV</a><h2>Core</h2><img src="core.svg"><h2>Wrappers</h2><img src="wrappers.svg"><h2>GRiD overhead decomposition (µs per batch, differences of medians of run means)</h2><p>kernel_compute = CUDA host call compute-only; memory_traffic = with-memory host call minus compute-only; c_abi_staging = C ABI minus CUDA host call; numpy_python = NumPy minus C ABI; *_dispatch = framework resident minus compute-only; *_round_trip = framework full-call minus resident. Pinocchio rows show the selected thread count in the main table (threads column, best of the recorded variants).</p><table>'+dtable+'</table><h2>All planned cells</h2><table>'+table+'</table>\n')
    print(args.output / "index.html")
    with (args.output / "index.html").open("a") as stream:
        stream.write('<h2>Accuracy disclosure</h2><p>'+html.escape(ACCURACY_FOOTNOTE)+
            '</p><p>Warning policy: fp32 Minv/FD operations only; each output block must have relative L2 error ≤ 0.001. '
            'Policy v2 checks both API outputs against the oracle before/after timing and also bounds inter-call/cross-API variation. '
            'This samples numerical variation; it is not a guarantee about every timed call or proof of absence of state mutation. '
            'The original entrywise gate remains atol=0.001, rtol=0.0002. '
            'max_bad_entries is the largest exceedance count across pre/post checks and repeats, not their sum.</p>\n')


if __name__ == "__main__":
    main()

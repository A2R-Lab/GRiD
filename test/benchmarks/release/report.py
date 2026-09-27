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

from .protocol import CORE, EXTRA, PRIMARY, ROBOTS, WRAPPER_OPS, WRAPPERS, TABLE_BACKENDS, digest, overhead, write_json
from .protocol import TIMED_STATUSES, cell_accuracy_status, ACCURACY_FOOTNOTE

LABELS = {"grid_cuda": "GRiD CUDA host call", "grid_native": "GRiD C ABI", "grid_numpy": "GRiD NumPy", "grid_jax": "GRiD JAX",
          "grid_torch": "GRiD PyTorch", "pinocchio": "Pinocchio CPU (codegen)", "pinocchio_plain": "Pinocchio CPU (standard API)", "mjx": "MJX",
          "mujoco_warp": "MuJoCo Warp", "mujoco_cpu": "MuJoCo CPU", "bard": "BARD", "frax": "Frax"}
OP_LABELS = {**dict(zip(CORE, ("RNEA", "grad RNEA", "Hessian RNEA"))),
             "minv": "M⁻¹", "forward_dynamics": "FD", "forward_dynamics_gradient": "grad FD", "fdsva_so": "Hessian FD",
             "end_effector_pose": "EE pose", "end_effector_pose_gradient": "grad EE pose", "end_effector_pose_hessian": "Hessian EE pose",
             "crba": "CRBA (M)", "nonlinear_effects": "bias (C·qd + g)", "generalized_gravity": "gravity g",
             "ccrba": "centroidal momentum matrix", "coriolis_matrix": "Coriolis matrix"}


BACKEND_HUE = {"grid_cuda": "#2a78d6", "grid_native": "#2a78d6", "grid_numpy": "#2a78d6",
               "grid_jax": "#2a78d6", "grid_torch": "#2a78d6", "pinocchio": "#eb6834", "pinocchio_plain": "#c94d1f",
               "mjx": "#1baf7a", "mujoco_warp": "#eda100", "mujoco_cpu": "#e87ba4",
               "bard": "#008300", "frax": "#4a3aa7"}


def stable_provenance(provenance):
    """The hardware identity that a cross-capture comparison must share, with
    the values that legitimately change between captures on one box removed:
    the momentary SM clock in the nvidia-smi line and lscpu's current scaling
    percentage. Driver, memory, power limit and the CPU model stay."""
    gpu = provenance.get("gpu") or ""
    gpu = ", ".join(t for t in gpu.split(", ") if not t.strip().endswith("MHz"))
    cpu = "\n".join(l for l in (provenance.get("cpu") or "").splitlines() if "scaling MHz" not in l)
    return gpu, cpu


ACCEPTED_SOURCE_DRIFT = set()   # filled from --accept-source-drift; always recorded in table.json


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
                    "input_storage_dtype": adapter.get("input_storage_dtype", "unknown"),
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
                    # The contract is the measurement protocol, the inputs and the
                    # hardware; the code under test is versioned per capture
                    # (commit and diff in provenance, library hashes in the adapter
                    # metadata) so a narrow re-collection at a later commit can
                    # supersede cells beside their unchanged neighbours.
                    "commit": provenance.get("commit"), "code_diff": provenance.get("diff_sha256"),
                    "submodules": provenance.get("submodules"),
                    "contract": json.dumps({
                        # the timing code only: the oracle files (RBDReference, URDFParser) are
                        # provenance — every cell is validated against the oracle at collection
                        # time, so an oracle fix later does not change what a timing means.
                        "sources": {k: v for k, v in (capture.get("collector_sources") or provenance.get("collector_sources") or {}).items()
                                    if k.startswith("test/benchmarks/") and not k.endswith("/report.py")
                                    and k not in ACCEPTED_SOURCE_DRIFT},
                        "packages": provenance.get("packages"),
                        "arithmetic_policy": plan.get("arithmetic_policy"),
                        "accuracy_policy": policy, "fd_warning_max_relative_l2": plan.get("fd_warning_max_relative_l2"),
                        "accuracy_policy_version": version,
                        "gpu": stable_provenance(provenance)[0], "cpu": stable_provenance(provenance)[1],
                        # CPU power management (governor, energy preference, limits, affinity):
                        # captures taken under different settings are never combined silently.
                        "cpu_power": provenance.get("cpu_power"),
                        "iterations": plan["iterations"], "warmups": plan["warmups"],
                        "warm_seconds": capture.get("warm_seconds", plan.get("warm_seconds")),
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
                   host_min_us=None, host_max_us=None, resident_min_us=None, resident_max_us=None, overhead_us=None, boundary_flag="",
                   max_abs_error=max((r["max_abs_error"] for r in group if r["max_abs_error"] is not None), default=None),
                   relative_l2_error=max((r["relative_l2_error"] for r in group if r["relative_l2_error"] is not None), default=None))
        contracts = {(r["contract"], r["urdf_sha256"], r["input_values_sha256"], r["dtype"], r.get("input_storage_dtype", "unknown"), r["method"], r["purpose"], r["expected_repeats"]) for r in group}
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
                res = [r["resident_us"] for r in group]
                row.update(resident_us=statistics.median(res), resident_min_us=min(res), resident_max_us=max(res))
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
                host_us=None,resident_us=None,host_min_us=None,host_max_us=None,resident_min_us=None,resident_max_us=None,overhead_us=None)
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
    ("pinocchio_codegen_us", "pinocchio", "host_us", None, None),
    ("pinocchio_standard_api_overhead_us", "pinocchio_plain", "host_us", "pinocchio", "host_us"),
)


def decompose(rows):
    lookup = {(r["robot"], r["operation"], r["backend"], r["batch"]): r for r in rows}
    keys = sorted({(r["robot"], r["operation"], r["batch"]) for r in rows if r["backend"] in {"grid_cuda", "pinocchio_plain"}})
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
    if kind == "table":
        ops = [op for op in EXTRA if any(r["operation"] == op and r["host_us"] for r in rows)]
        if not ops:
            return None
    else:
        ops = CORE if kind == "core" else WRAPPER_OPS
    selected = lambda op: PRIMARY[op] if kind == "core" else WRAPPERS if kind == "wrappers" else TABLE_BACKENDS
    robots = [r for r in ROBOTS if any(x["robot"] == r for x in rows)]
    batches = sorted({r["batch"] for r in rows})
    lookup = {(r["robot"], r["operation"], r["backend"], r["batch"]): r for r in rows}
    fig, axes = plt.subplots(len(ops), len(robots), figsize=(max(10,5.3*len(robots)), 3.3*len(ops)), squeeze=False)
    colors = {b: BACKEND_HUE[b] for b in LABELS}   # fixed validated palette, shared with the stacked figure
    colors.update({"grid_cuda": "#0d366b", "grid_native": "#184f95", "grid_numpy": "#256abf", "grid_jax": "#3987e5", "grid_torch": "#86b6ef"})
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
    fig.suptitle(banner(purpose, f"{kind.title()} comparison · median of run means; whiskers show run-mean range"), fontsize=12)
    if kind == "table":
        fig.set_size_inches(max(10, 5.3*len(robots)), 3.3*len(ops))
    fig.text(.5,.095,"* fp64 arithmetic exception. Red triangle: negative timing delta, not stacked. N/C: not collected.\nGRiD CUDA host call: base = compute-only kernel launch, cap = H2D/D2H of one call. Other stacked bases include resident API dispatch. Unstacked bars are full-call only.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.15,1,.93))
    fig.savefig(directory / f"{kind}.svg")
    fig.savefig(directory / f"{kind}.png", dpi=140)
    plt.close(fig)

# ── Stacked comparison and GRiD composition figures ─────────────────────────
# Categorical hue per backend in a fixed, validated order (dataviz reference
# palette, light mode); every GRiD surface shares the blue slot. The GRiD bar
# is stacked from the CUDA host-call boundaries: compute (kernel), memory
# (with-memory minus compute) and wrapper (API full call minus with-memory),
# in three ordinal steps of the same hue. Competitors with a resident boundary
# show resident (solid) plus host round trip (hatched, same hue, lighter).
# (BACKEND_HUE is defined near the top of the module.)
SEGMENT_HUE = {"compute": "#184f95", "memory": "#3987e5", "wrapper": "#86b6ef"}
COMPETITOR_ORDER = ("pinocchio", "mjx", "mujoco_warp", "mujoco_cpu", "bard", "frax")
PINOCCHIO_CODEGEN_OPS = {"inverse_dynamics", "inverse_dynamics_gradient", "minv", "forward_dynamics",
                         "forward_dynamics_gradient", "crba", "nonlinear_effects", "generalized_gravity"}
SURFACE_LABELS = {"grid_native": "C ABI", "grid_numpy": "NumPy", "grid_jax": "JAX", "grid_torch": "PyTorch"}


def grid_stack(lookup, robot, op, batch, api):
    """(compute, memory, wrapper, flags) for one cell from grid_cuda + the API row;
    a missing or negative term is None (never clamped)."""
    cuda = lookup.get((robot, op, "grid_cuda", batch), {})
    surface = lookup.get((robot, op, api, batch), {})
    compute, with_mem, full = cuda.get("resident_us"), cuda.get("host_us"), surface.get("host_us")
    memory = overhead(with_mem, compute)
    wrapper = overhead(full, with_mem)
    flags = []
    if with_mem is not None and compute is not None and memory is None:
        flags.append("memory")
    if full is not None and with_mem is not None and wrapper is None:
        flags.append("wrapper")
    return compute, memory, wrapper, flags


BANNER = {"smoke": "SMOKE TEST — NOT PERFORMANCE EVIDENCE", "collection": "DRAFT — UNREVIEWED COLLECTION"}


def banner(purpose, title):
    """Figure title with the purpose banner. Only the explicit publication
    purpose ("release", set by docs/plot_release_figures.py --approve after the
    audit) drops the banner; every report this tool writes keeps it."""
    prefix = BANNER.get(purpose, BANNER["collection"] if purpose != "release" else "")
    return f"{prefix}\n{title}" if prefix else title


def _panel_grid(ops, robots, purpose, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(ops), len(robots), figsize=(max(10, 5.3*len(robots)), 3.4*len(ops)), squeeze=False)
    fig.suptitle(banner(purpose, title), fontsize=12)
    return plt, fig, axes


def plot_stacked_comparison(rows, directory, purpose, api="grid_jax", ops=CORE, stem="comparison_stacked"):
    """The given operations (core by default): GRiD (one API surface) as a
    compute/memory/wrapper stack beside every competitor that has data, per
    batch size, log axis."""
    from matplotlib.patches import Patch
    # a panel needs the GRiD stack (the CUDA host call) to exist for that operation
    ops = [op for op in ops if any(r["operation"] == op and r["backend"] == "grid_cuda" and r["host_us"] for r in rows)]
    robots = [r for r in ROBOTS if any(x["robot"] == r for x in rows)]
    if not ops or not robots:
        return None
    batches = sorted({r["batch"] for r in rows})
    lookup = {(r["robot"], r["operation"], r["backend"], r["batch"]): r for r in rows}
    plt, fig, axes = _panel_grid(ops, robots, purpose,
        f"GRiD ({LABELS[api]}) stacked as kernel compute + memory traffic + wrapper overhead, competitors as resident + host round trip · medians of run means, log scale")
    for oi, op in enumerate(ops):
        # Pinocchio has a bar when either API measured the operation (the standard
        # API alone for the operations without a codegen class)
        has = lambda b: any(r["operation"] == op and r["backend"] == b and r["host_us"] for r in rows)
        competitors = [b for b in COMPETITOR_ORDER if has(b) or (b == "pinocchio" and has("pinocchio_plain"))]
        backends = ["grid"] + competitors
        values = []
        for ri, robot in enumerate(robots):
            ax = axes[oi, ri]
            width = .8/len(backends)
            for xi, batch in enumerate(batches):
                for bi, backend in enumerate(backends):
                    x = xi - .4 + width*(bi + .5)
                    if backend == "grid":
                        compute, memory, wrapper, flags = grid_stack(lookup, robot, op, batch, api)
                        if compute is None:
                            ax.text(x, .025, "N/C", rotation=90, ha="center", va="bottom", fontsize=5.5, transform=ax.get_xaxis_transform())
                            continue
                        bottom = 0.
                        for name, value in (("compute", compute), ("memory", memory), ("wrapper", wrapper)):
                            if value is None:
                                break
                            ax.bar(x, value, width*.85, bottom=bottom or None, color=SEGMENT_HUE[name], edgecolor="white", linewidth=.6)
                            bottom += value
                            values.append(bottom)
                        if flags:
                            ax.plot(x, bottom, "v", color="#d03b3b", markersize=5)
                        api_row = lookup.get((robot, op, api, batch), {})
                        if wrapper is not None and api_row.get("host_min_us") and api_row.get("host_max_us"):
                            ax.errorbar(x, api_row["host_us"], yerr=[[api_row["host_us"] - api_row["host_min_us"]], [api_row["host_max_us"] - api_row["host_us"]]], color="black", capsize=2, linewidth=.7)
                        marks = "".join(m for m, hit in (("*", any(lookup.get((robot, op, b, batch), {}).get("dtype") == "float64" for b in ("grid_cuda", api))),
                                                          ("†", any(lookup.get((robot, op, b, batch), {}).get("status") == "accuracy_warning" for b in ("grid_cuda", api)))) if hit)
                        if marks:
                            ax.annotate(marks, (x, bottom), xytext=(0, 3), textcoords="offset points", ha="center")
                        continue
                    row = lookup.get((robot, op, backend, batch), {})
                    total, resident = row.get("host_us"), row.get("resident_us")
                    standard_only = False
                    if backend == "pinocchio":
                        # codegen full call as the base, the standard API's extra
                        # time as the cap (same robot, inputs and thread policy).
                        # Where no codegen path exists (second-order, kinematics,
                        # centroidal) both modes run the same standard analytical
                        # code: the whole bar is the standard API and is hatched.
                        plain = lookup.get((robot, op, "pinocchio_plain", batch), {}).get("host_us")
                        codegen_op = op in PINOCCHIO_CODEGEN_OPS
                        if not codegen_op:
                            total, resident, standard_only = (plain if plain is not None else total), None, True
                        elif total is None and plain is not None:
                            total, resident, standard_only = plain, None, True
                        elif total is not None and plain is not None:
                            resident, total = total, plain
                    if total is None:
                        ax.text(x, .025, "N/C" if not row else row.get("status", "N/A").replace("_", " "),
                                rotation=90, ha="center", va="bottom", fontsize=5.5, transform=ax.get_xaxis_transform())
                        continue
                    cap = overhead(total, resident) if resident is not None else None
                    if cap is not None:
                        ax.bar(x, resident, width*.85, color=BACKEND_HUE[backend], edgecolor="white", linewidth=.6)
                        ax.bar(x, cap, width*.85, bottom=resident, facecolor=BACKEND_HUE[backend], alpha=.45, hatch="////", edgecolor="white", linewidth=.6)
                    elif standard_only:
                        ax.bar(x, total, width*.85, facecolor=BACKEND_HUE[backend], alpha=.45, hatch="////", edgecolor="white", linewidth=.6)
                    else:
                        ax.bar(x, total, width*.85, color=BACKEND_HUE[backend], edgecolor="white", linewidth=.6)
                    lo, hi = row.get("host_min_us"), row.get("host_max_us")
                    if backend == "pinocchio" and (cap is not None or standard_only) and lookup.get((robot, op, "pinocchio_plain", batch), {}).get("host_us") is not None:
                        plain_row = lookup.get((robot, op, "pinocchio_plain", batch), {})
                        lo, hi = plain_row.get("host_min_us"), plain_row.get("host_max_us")
                    if lo and hi:
                        ax.errorbar(x, total, yerr=[[max(total - lo, 0.)], [max(hi - total, 0.)]], color="black", capsize=2, linewidth=.7)
                    if resident is not None and cap is None:
                        ax.plot(x, total, "v", color="#d03b3b", markersize=5)
                    if row.get("dtype") == "float64":
                        ax.annotate("*", (x, total), xytext=(0, 3), textcoords="offset points", ha="center")
                    if row.get("status") == "accuracy_warning":
                        ax.annotate("†", (x, total), xytext=(0, 3), textcoords="offset points", ha="center")
                    values.append(total)
            ax.set(title=f"{robot} · {OP_LABELS[op]}", xticks=range(len(batches)), xticklabels=batches, xlabel="Batch size", ylabel="µs / complete batch")
            ax.set_xlim(-.5, len(batches) - .5)
            ax.grid(axis="y", alpha=.15)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        finite = [v for v in values if v and v > 0]
        if finite:
            for ri in range(len(robots)):
                axes[oi, ri].set_yscale("log")
                axes[oi, ri].set_ylim(min(finite)*.5, max(finite)*1.6)
    handles = [Patch(color=SEGMENT_HUE["compute"], label="GRiD kernel compute (CUDA host call, compute-only)"),
               Patch(color=SEGMENT_HUE["memory"], label="GRiD memory traffic (with-memory host call − compute)"),
               Patch(color=SEGMENT_HUE["wrapper"], label=f"GRiD wrapper overhead ({LABELS[api]} full call − with-memory)")]
    handles += [Patch(color=BACKEND_HUE[b], label=f"{LABELS[b]} (resident or full call)" if b != "pinocchio" else "Pinocchio CPU codegen full call")
                for b in COMPETITOR_ORDER if any(r["backend"] == b and r["host_us"] for r in rows)]
    handles.append(Patch(facecolor=".7", alpha=.45, hatch="////", edgecolor="white",
                         label="Competitor host round trip (full call − resident); for Pinocchio: the standard API's time over codegen, or the whole bar where no codegen path exists"))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(.5, .01))
    tall = len(ops) > 3
    fig.text(.5, .045 if tall else .13, "Log axis: stacked segment heights are not proportional; read the composition figure or the decomposition table for shares. "
             "Whiskers: range of the three run means of the full call. * fp64 arithmetic exception. † accuracy warning. Red triangle: a negative difference, segment omitted. N/C: not collected.", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, .06 if tall else .19, 1, .97 if tall else .93))
    fig.savefig(directory / f"{stem}.svg")
    fig.savefig(directory / f"{stem}.png", dpi=140)
    plt.close(fig)
    return directory / f"{stem}.svg"


def plot_grid_composition(rows, directory, purpose):
    """Wrapper operations: each GRiD surface as compute + memory + its own
    wrapper overhead on a linear axis (same compute and memory in every bar)."""
    from matplotlib.patches import Patch
    surfaces = [b for b in SURFACE_LABELS if any(r["backend"] == b and r["host_us"] for r in rows)]
    ops = [op for op in WRAPPER_OPS if any(r["operation"] == op and r["backend"] == "grid_cuda" for r in rows)]
    robots = [r for r in ROBOTS if any(x["robot"] == r for x in rows)]
    if not surfaces or not ops or not robots:
        return None
    batches = sorted({r["batch"] for r in rows})
    lookup = {(r["robot"], r["operation"], r["backend"], r["batch"]): r for r in rows}
    plt, fig, axes = _panel_grid(ops, robots, purpose,
        "GRiD surfaces: kernel compute + memory traffic + wrapper overhead per full call · medians of run means, linear")
    width = .8/len(surfaces)
    for oi, op in enumerate(ops):
        for ri, robot in enumerate(robots):
            ax = axes[oi, ri]
            for xi, batch in enumerate(batches):
                for si, surface in enumerate(surfaces):
                    x = xi - .4 + width*(si + .5)
                    compute, memory, wrapper, flags = grid_stack(lookup, robot, op, batch, surface)
                    if compute is None:
                        ax.text(x, .025, "N/C", rotation=90, ha="center", va="bottom", fontsize=5.5, transform=ax.get_xaxis_transform())
                        continue
                    bottom = 0.
                    for name, value in (("compute", compute), ("memory", memory), ("wrapper", wrapper)):
                        if value is None:
                            break
                        ax.bar(x, value, width*.85, bottom=bottom, color=SEGMENT_HUE[name], edgecolor="white", linewidth=.6)
                        bottom += value
                    ax.text(x, bottom, SURFACE_LABELS[surface], rotation=90, ha="center", va="bottom", fontsize=5.5, color="#52514e")
                    if flags:
                        ax.plot(x, bottom, "v", color="#d03b3b", markersize=5)
            ax.set(title=f"{robot} · {OP_LABELS[op]}", xticks=range(len(batches)), xticklabels=batches, xlabel="Batch size", ylabel="µs / complete batch")
            ax.set_xlim(-.5, len(batches) - .5)
            ax.set_ylim(0, None)
            ax.margins(y=.18)
            ax.grid(axis="y", alpha=.15)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    handles = [Patch(color=SEGMENT_HUE["compute"], label="Kernel compute (CUDA host call, compute-only)"),
               Patch(color=SEGMENT_HUE["memory"], label="Memory traffic (with-memory host call − compute)"),
               Patch(color=SEGMENT_HUE["wrapper"], label="Wrapper overhead (surface full call − with-memory)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(.5, .02))
    fig.text(.5, .095, "Bars per batch: " + ", ".join(SURFACE_LABELS[s] for s in surfaces) + ". Red triangle: a negative difference, segment omitted.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, .93))
    fig.savefig(directory / "grid_composition.svg")
    fig.savefig(directory / "grid_composition.png", dpi=140)
    plt.close(fig)
    return directory / "grid_composition.svg"


# ── Presentation figures: speedup heatmaps, best competitor, throughput ──────
# Every ratio is a ratio of medians of run means from matched cells of the same
# report, same boundary on both sides; a missing side is blank, never zero.
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
import math  # noqa: E402

COMPETITOR_ORDER_ALL = ("pinocchio", "pinocchio_plain", "mjx", "mujoco_warp", "mujoco_cpu", "bard", "frax")
SHORT_OP = {"inverse_dynamics": "RNEA", "inverse_dynamics_gradient": "∇RNEA", "idsva_so": "∇²RNEA", "minv": "M⁻¹",
            "forward_dynamics": "FD", "forward_dynamics_gradient": "∇FD", "fdsva_so": "∇²FD",
            "end_effector_pose": "EE pose", "end_effector_pose_gradient": "∇EE", "end_effector_pose_hessian": "∇²EE",
            "crba": "M", "nonlinear_effects": "C·q̇+g", "generalized_gravity": "g", "ccrba": "A_G", "coriolis_matrix": "C"}
# diverging blue <-> gray <-> red (reference palette poles), centred on 1x
SPEEDUP_CMAP = LinearSegmentedColormap.from_list("grid_speedup", ["#d03b3b", "#ec835a", "#f0efec", "#86b6ef", "#2a78d6", "#0d366b"])


UNSTABLE_SPREAD = 1.5   # largest / smallest run mean across the repeats


def unstable(row, field):
    """True when the repeats of this boundary disagree by more than UNSTABLE_SPREAD."""
    if not row:
        return False
    lo, hi = row.get(field.replace("_us", "_min_us")), row.get(field.replace("_us", "_max_us"))
    return bool(lo and hi and hi / lo > UNSTABLE_SPREAD)


def cell_marks(grid_row, grid_field, comp_row, comp_field):
    """'*' when a side is fp64, '†' when a side is a retained accuracy warning,
    '~' when a side's repeats of the compared boundary spread by more than 1.5×."""
    rows = [r for r in (grid_row, comp_row) if r]
    return ("*" if any(r.get("dtype") == "float64" for r in rows) else "") + \
           ("†" if any(r.get("status") == "accuracy_warning" for r in rows) else "") + \
           ("~" if unstable(grid_row, grid_field) or unstable(comp_row, comp_field) else "")


def _ratio_heatmap(ax, matrix, row_labels, col_labels, title, vmax=100., marks=None):
    import numpy as np
    m = np.array(matrix, float)
    ax.imshow(np.log10(np.where(np.isfinite(m), m, np.nan)), cmap=SPEEDUP_CMAP,
              norm=TwoSlopeNorm(vmin=-math.log10(vmax), vcenter=0., vmax=math.log10(vmax)), aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if np.isfinite(v):
                ax.text(j, i, (f"{v:.0f}×" if v >= 10 else f"{v:.1f}×") + (marks[i][j] if marks else ""), ha="center", va="center", fontsize=7,
                        color="white" if abs(math.log10(v)) > 0.9 else "#0b0b0b")
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="#9a9994")
    ax.set_title(title, fontsize=9, wrap=True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)


def _ratio_rows(rows, grid_backend):
    ops = [op for op in CORE + EXTRA if any(r["operation"] == op and r["backend"] == grid_backend and r["host_us"] for r in rows)]
    robots = [ro for ro in ROBOTS if any(r["robot"] == ro for r in rows)]
    return [(op, ro) for op in ops for ro in robots]


def plot_speedup(rows, directory, purpose, grid_backend, grid_field, comp_field, title, name):
    import numpy as np
    lookup = {(r["robot"], r["operation"], r["backend"], r["batch"]): r for r in rows}
    batches = sorted({r["batch"] for r in rows})
    comps = [c for c in COMPETITOR_ORDER_ALL if any(r["backend"] == c and r.get(comp_field) for r in rows)]
    cells = [(op, ro) for op, ro in _ratio_rows(rows, grid_backend)
             if any(lookup.get((ro, op, c, b), {}).get(comp_field) for c in comps for b in batches)]
    if not comps or not cells:
        return None
    labels = [f"{ro} · {SHORT_OP.get(op, op)}" for op, ro in cells]
    plt, fig, axes = _panel_grid([None], [None], purpose, title)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(comps), figsize=(2.6*len(comps) + 1.6, .28*len(labels) + 1.8), squeeze=False, sharey=True)
    for ci, comp in enumerate(comps):
        matrix = [[(lambda g, c: c / g if (g and c) else np.nan)(
            lookup.get((ro, op, grid_backend, b), {}).get(grid_field), lookup.get((ro, op, comp, b), {}).get(comp_field))
            for b in batches] for op, ro in cells]
        marks = [[cell_marks(lookup.get((ro, op, grid_backend, b)), grid_field, lookup.get((ro, op, comp, b)), comp_field) for b in batches] for op, ro in cells]
        _ratio_heatmap(axes[0, ci], matrix, labels, batches, LABELS[comp], marks=marks)
        axes[0, ci].set_xlabel("batch")
    fig.suptitle(banner(purpose, title), fontsize=11)
    fig.text(.5, .01, "Ratio > 1: GRiD faster. Diverging scale centred on 1×, log spaced, clipped at 100×. '–': no matched cell (adapter pending, excluded, or failed validation). * a side computes in fp64. † a side is a retained fp32 accuracy warning. ~ a side's three run means spread by more than 1.5×.", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, .03, 1, .94))
    fig.savefig(directory / f"{name}.svg"); fig.savefig(directory / f"{name}.png", dpi=150)
    plt.close(fig)
    return directory / f"{name}.svg"


def plot_best_competitor(rows, directory, purpose):
    import numpy as np
    lookup = {(r["robot"], r["operation"], r["backend"], r["batch"]): r for r in rows}
    batches = sorted({r["batch"] for r in rows})
    panels = (("grid_jax", "host_us", "host_us", "GRiD JAX full call vs the fastest competitor full call"),
              ("grid_jax", "resident_us", "resident_us", "GRiD JAX resident vs the fastest GPU competitor resident"),
              ("grid_cuda", "host_us", "host_us", "GRiD CUDA host call (with memory) vs the fastest competitor full call"))
    cells = [(op, ro) for op, ro in _ratio_rows(rows, "grid_jax")
             if any(lookup.get((ro, op, c, b), {}).get("host_us") for c in COMPETITOR_ORDER_ALL for b in batches)]
    if not cells:
        return None
    labels = [f"{ro} · {SHORT_OP.get(op, op)}" for op, ro in cells]
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, .28*len(labels) + 1.8), squeeze=False, sharey=True)
    for pi, (gb, gf, cf, ttl) in enumerate(panels):
        matrix, marks = [], []
        for op, ro in cells:
            line, mline = [], []
            for b in batches:
                g = lookup.get((ro, op, gb, b), {}).get(gf)
                cands = [lookup[(ro, op, c, b)] for c in COMPETITOR_ORDER_ALL if (ro, op, c, b) in lookup and lookup[(ro, op, c, b)].get(cf)]
                best = min(cands, key=lambda r: r[cf]) if cands else None
                line.append(best[cf] / g if (g and best) else np.nan)
                mline.append(cell_marks(lookup.get((ro, op, gb, b)), gf, best, cf))
            matrix.append(line); marks.append(mline)
        _ratio_heatmap(axes[0, pi], matrix, labels, batches, ttl, marks=marks)
        axes[0, pi].set_xlabel("batch")
    fig.suptitle(banner(purpose, "GRiD against the fastest competitor measured for each cell, same boundary on both sides"), fontsize=11)
    fig.tight_layout(rect=(0, .02, 1, .94))
    fig.savefig(directory / "best_competitor.svg"); fig.savefig(directory / "best_competitor.png", dpi=150)
    plt.close(fig)
    return directory / "best_competitor.svg"


def plot_throughput(rows, directory, purpose):
    import matplotlib.pyplot as plt
    lookup = {(r["robot"], r["operation"], r["backend"], r["batch"]): r for r in rows}
    ops = [op for op in CORE + EXTRA if any(r["operation"] == op and r["host_us"] for r in rows)]
    robots = [ro for ro in ROBOTS if any(r["robot"] == ro for r in rows)]
    if not ops or not robots:
        return None
    batches = sorted({r["batch"] for r in rows})
    backends = ["grid_cuda", "grid_jax", "grid_torch", "grid_numpy"] + list(COMPETITOR_ORDER_ALL)
    styles = {"grid_cuda": "-", "grid_jax": "--", "grid_torch": "-.", "grid_numpy": ":", "pinocchio_plain": ":"}
    fig, axes = plt.subplots(len(ops), len(robots), figsize=(5*len(robots), 3.2*len(ops)), squeeze=False)
    for oi, op in enumerate(ops):
        for ri, ro in enumerate(robots):
            ax = axes[oi, ri]
            for be in backends:
                pts = [(b, b / lookup[(ro, op, be, b)]["host_us"] * 1e6) for b in batches
                       if (ro, op, be, b) in lookup and lookup[(ro, op, be, b)]["host_us"]]
                if pts:
                    ax.plot(*zip(*pts), label=LABELS[be], color=BACKEND_HUE[be], linestyle=styles.get(be, "-"), linewidth=2, marker="o", markersize=4)
            ax.set(xscale="log", yscale="log", title=f"{ro} · {SHORT_OP.get(op, op)}", xlabel="batch", ylabel="samples / s (full call)")
            ax.set_xticks(batches); ax.set_xticklabels(batches)
            ax.grid(alpha=.15)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    handles, names = [], []
    for a in axes.flat:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in names:
                handles.append(h); names.append(l)
    fig.legend(handles, names, loc="lower center", ncol=6, fontsize=8)
    fig.suptitle(banner(purpose, "Throughput per full call (host in, host out) · higher is better"), fontsize=12)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(directory / "throughput.svg"); fig.savefig(directory / "throughput.png", dpi=110)
    plt.close(fig)
    return directory / "throughput.svg"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("captures", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--accept-source-drift", action="append", default=[], metavar="PATH",
                    help="collector source file whose version may differ between the captures being combined "
                         "(an adapter edited for another backend); every accepted path is recorded in table.json")
    args = ap.parse_args()
    ACCEPTED_SOURCE_DRIFT.update(args.accept_source_drift)
    # Captures are read in order; a cell (robot, operation, backend, batch, repeat)
    # already produced by an earlier capture supersedes the same cell in a later
    # one — so a narrow re-collection goes FIRST, and the core and wrappers
    # captures (which both plan the GRiD CUDA/JAX RNEA cells) can be reported
    # together without being mistaken for extra repeats.
    # A planned cell that an earlier capture never collected (a chain that died,
    # a worker that was never reached) is only a placeholder: it must not shadow
    # the same cell collected later, so a later collected record replaces it and
    # the placeholder is recorded as superseded instead.
    chosen, superseded = {}, []
    for directory in args.captures:
        for r in records(directory):
            key = (r["robot"], r["operation"], r["backend"], r["batch"], r["repeat"])
            named = dict(zip(("robot", "operation", "backend", "batch", "repeat"), key))
            held = chosen.get(key)
            if held is None:
                chosen[key] = r
            elif held["status"] == "not_collected" and r["status"] != "not_collected":
                superseded.append({"capture": held["capture"], "reason": "planned but never collected; a later capture holds the cell", **named})
                chosen[key] = r
            else:
                superseded.append({"capture": str(directory), **named})
    raw = list(chosen.values())
    if not raw:
        ap.error("No planned cells")
    purposes = {r["purpose"] for r in raw}
    if len(purposes) != 1:
        ap.error("Do not mix smoke and collection captures")
    rows = aggregate(raw)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "table.json", {"publication_approved": False, "purpose": raw[0]["purpose"], "cells": rows, "raw_records": raw,
        "superseded_cells": superseded, "capture_order": [str(d) for d in args.captures],
        "accepted_source_drift": sorted(ACCEPTED_SOURCE_DRIFT)})
    with (args.output / "table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for kind in ("core", "wrappers", "table"):
        plot(rows,args.output,kind,raw[0]["purpose"])
    stacked = plot_stacked_comparison(rows, args.output, raw[0]["purpose"])
    composition = plot_grid_composition(rows, args.output, raw[0]["purpose"])
    purpose = raw[0]["purpose"]
    extra_figures = [f for f in (
        plot_speedup(rows, args.output, purpose, "grid_jax", "host_us", "host_us", "Speedup of GRiD (JAX API, full call) over each competitor's full call", "speedup_full"),
        plot_speedup(rows, args.output, purpose, "grid_jax", "resident_us", "resident_us", "Speedup of GRiD (JAX API, resident) over each GPU competitor's resident call", "speedup_resident"),
        plot_speedup(rows, args.output, purpose, "grid_cuda", "host_us", "host_us", "Speedup of GRiD (CUDA host call with memory) over each competitor's full call", "speedup_kernel"),
        plot_best_competitor(rows, args.output, purpose),
        plot_throughput(rows, args.output, purpose)) if f]
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
    (args.output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>GRiD benchmark draft</title><style>body{font:14px system-ui;margin:2rem}td,th{padding:.5rem;border:1px solid #ddd}table{border-collapse:collapse}img{max-width:100%}</style><h1>DRAFT benchmark audit</h1><p>Not publication-approved. Smoke captures are functional checks, not performance evidence. All times are microseconds per batch; summary is median of run means. Gray caps are paired boundary differences, not isolated transfer timings. Precision exceptions are explicit. No speedup claims are generated.</p><a href="table.csv">CSV</a> · <a href="table.json">Full provenance and error metrics</a> · <a href="decomposition.csv">Overhead decomposition CSV</a><h2>Core</h2><img src="core.svg">'+('<h2>Stacked comparison (GRiD compute + memory + wrapper vs competitors)</h2><img src="comparison_stacked.svg">' if stacked else '')+'<h2>Wrappers</h2><img src="wrappers.svg">'+('<h2>Table operations</h2><img src="table.svg">' if (args.output / "table.svg").exists() else '')+('<h2>GRiD surface composition</h2><img src="grid_composition.svg">' if composition else '')+'<h2>Speedup and throughput views</h2>'+''.join(f'<h3>{f.stem}</h3><img src="{f.name}">' for f in extra_figures)+'<h2>GRiD overhead decomposition (µs per batch, differences of medians of run means)</h2><p>kernel_compute = CUDA host call compute-only; memory_traffic = with-memory host call minus compute-only; c_abi_staging = C ABI minus CUDA host call; numpy_python = NumPy minus C ABI; *_dispatch = framework resident minus compute-only; *_round_trip = framework full-call minus resident. Pinocchio rows show the selected thread count in the main table (threads column, best of the recorded variants).</p><table>'+dtable+'</table><h2>All planned cells</h2><table>'+table+'</table>\n')
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

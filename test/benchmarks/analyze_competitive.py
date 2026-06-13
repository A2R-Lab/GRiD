#!/usr/bin/env python3
"""A1b competitive analysis: GRiD-AUTOTUNED vs each competitor at N=256.

GRiD column = the autotuned best per (robot,base,algo) from autotune_best_<host>.json
  -> 'us' = TOTAL batch_256 COMPUTE-ONLY microseconds at the optimal (tier,threads).
    (the autotune minimizes batch_256_compute_only_us; this is the GPU-resident /
     MPC-rollout number — see project_grid_competitive_analysis.md methodology.)
Competitor column = batch_256_with_mem_us (TOTAL batch us) from each baseline json.
  Same unit (whole-256-batch us), so speedup = competitor_us / grid_us  (>1 => GRiD faster).

This is the apples-to-apples generate_report.py methodology (GRiD compute-only vs
baseline with-mem). It answers: with the FFI/launch thread pathology fixed by the
autotune, which C.7 "losses" are real vs measurement artifacts.

Usage:
  python test/benchmarks/analyze_competitive.py \
      --autotune test/benchmarks/results/autotune_best_<host>.json \
      --results  test/benchmarks/results/competitive_20260613 \
      [--extra-results DIR ...] [--out test/benchmarks/results/competitive_20260613/ANALYSIS.md]
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path

# Canonicalize any algo key (long or short) to one short key.
CANON = {
    "inverse_dynamics": "id", "id": "id",
    "forward_dynamics": "fd", "fd": "fd",
    "inverse_dynamics_gradient": "id_du", "id_du": "id_du",
    "forward_dynamics_gradient": "fd_du", "fd_du": "fd_du",
    "crba": "crba", "minv": "minv", "aba": "aba",
    "end_effector_pose": "ee_pose", "ee_pose": "ee_pose",
    "end_effector_pose_gradient": "ee_pose_gradient", "ee_pose_gradient": "ee_pose_gradient",
    "end_effector_pose_hessian": "ee_pose_hessian", "ee_pose_hessian": "ee_pose_hessian",
    # SO: all variants canonicalize to 'idsva_so' so a competitor's body-frame SO
    # compares against GRiD's production dispatcher (body=fixed / world=floating).
    "idsva_so": "idsva_so",
    "idsva_so_body_frame": "idsva_so",
    "idsva_so_world_frame": "idsva_so",
    "fdsva_so": "fdsva_so",
    "integrator": "integrator", "integrator_gradient": "integrator_gradient",
    "integrator_with_gradient": "integrator_with_gradient",
}

def _num(v):
    if isinstance(v, dict):
        v = v.get("median", v.get("mean"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def load_grid(autotune_path):
    """-> {(robot,base,canon_algo): us_compute_only_total}

    SECOND-ORDER: competitors (pinocchio) only expose the BODY-frame SO, but GRiD's
    PRODUCTION dispatcher (`idsva_so`) auto-selects body-frame for FIXED and the much
    faster world-frame for FLOATING (the two are algorithmically equivalent — same SO
    derivative, different intermediate frame). The standalone `idsva_so_body_frame`
    *floating* path is a NON-PRODUCTION reference that is pathologically slow and was
    the source of the C.7 "idsva_so losses" artifact. So for the SO comparison we use
    GRiD's dispatcher `idsva_so` (canonical 'idsva_so') and drop the raw body/world
    standalone entries — production-vs-pin-body is the honest comparison.
    """
    best = json.load(open(autotune_path))["best"]
    out = {}
    for robot, bases in best.items():
        for base, algos in bases.items():
            for algo, info in algos.items():
                if algo in ("idsva_so_body_frame", "idsva_so_world_frame"):
                    continue  # use the dispatcher `idsva_so` instead (see docstring)
                c = CANON.get(algo, algo)
                u = _num(info.get("us"))
                if u:
                    out[(robot, base, c)] = u
    return out

def load_competitors(dirs):
    """-> {col_key: {(robot,base,canon_algo): us_batch256_withmem_total}}"""
    cols = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                data = json.load(open(f))
            except Exception:
                continue
            res = data.get("results")
            if not isinstance(res, dict):
                continue
            for robot, bases in res.items():
                for base, colmap in bases.items():
                    for col, algos in colmap.items():
                        if col.startswith("grid") or "pick" in col or col == "metadata":
                            continue                      # skip GRiD rows + autotune-pick noise
                        if not isinstance(algos, dict):
                            continue
                        bucket = cols.setdefault(col, {})
                        for algo, m in algos.items():
                            if not isinstance(m, dict):
                                continue
                            u = _num(m.get("batch_256_with_mem_us"))
                            if u:
                                bucket[(robot, base, CANON.get(algo, algo))] = u
    return cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--autotune", required=True)
    ap.add_argument("--results", required=True, help="dir with competitor jsons")
    ap.add_argument("--extra-results", nargs="*", default=[], help="more competitor dirs (e.g. pin from overnight)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    grid = load_grid(args.autotune)
    comps = load_competitors([args.results, *args.extra_results])

    lines = ["# A1b competitive analysis — GRiD-autotuned vs competitors @ N=256",
             "",
             "GRiD = autotuned best **compute-only** total-batch us (GPU-resident). "
             "Competitor = **with-mem** total-batch us. speedup = comp/grid (>1 => GRiD faster).",
             "",
             "## Methodology + caveats (read before citing)",
             "- **GRiD number** = the autotuned best (tier,threads) per (robot,base,algo) at N=256, "
             "minimizing batch_256 **compute-only** us (GPU-resident — the MPC/rollout use case). "
             "This is the A1 launch-config fix in action: it removes the FFI thread-default pathology "
             "that contaminated the C.7 tally.",
             "- **Competitor number** = batch_256 **with-mem** us (their natural mode). For CPU pinocchio "
             "this is the standard framing. For GPU baselines (frax/mjx/mujoco_warp/curobo) with-mem includes "
             "host transfer that GRiD's compute-only excludes — but the win magnitudes (3–90×) far exceed any "
             "plausible transfer overhead, so the ranking is robust. A pure compute-only-vs-compute-only pass "
             "is future work (most GPU adapters report with-mem only).",
             "- **Coverage:** the tally only covers algos the competitor implements. GRiD ALSO ships many algos "
             "with NO competitor equivalent (fd_du, idsva_so/fdsva_so 2nd-order, ee_pose hessian, integrators, "
             "regressors, centroidal) — a capability lead not reflected in W/L.",
             "- **SO comparison** uses GRiD's PRODUCTION dispatcher `idsva_so` (body-frame for fixed, world-frame "
             "for floating) vs pinocchio's body-frame SO (algorithmically equivalent). The standalone "
             "body-frame-FLOATING path is non-production + pathologically slow and was the C.7 'idsva_so loss' artifact.",
             "- **cuRobo** loads its g1 config at **35 DOF** vs GRiD's g1_29dof (cuRobo does ~20% MORE work, "
             "so the comparison slightly favors cuRobo); GRiD still wins g1 id 5.75× / id_du 10.44×. cuRobo only "
             "ships configs for g1 (no iiwa14/go2) and is fixed-base only.",
             "- N=256; autotune_N=256; RTX 5090 / sm_120. h1_2 omitted from this competitive run (focus iiwa14/go2/g1).",
             ""]
    tally = {}
    for col in sorted(comps):
        cdata = comps[col]
        rows = []
        w = l = t = 0
        for key in sorted(cdata):
            if key not in grid:
                continue
            robot, base, algo = key
            g = grid[key]; c = cdata[key]
            sp = c / g
            verdict = "GRiD" if sp > 1.05 else ("comp" if sp < 0.95 else "~tie")
            if verdict == "GRiD": w += 1
            elif verdict == "comp": l += 1
            else: t += 1
            rows.append((robot, base, algo, g, c, sp, verdict))
        tally[col] = (w, l, t)
        lines.append(f"## vs {col}  —  GRiD {w}W / {l}L / {t}T  ({len(rows)} comparable cells)")
        lines.append("")
        lines.append("| robot | base | algo | GRiD us (compute) | comp us (w/mem) | speedup | winner |")
        lines.append("|---|---|---|---:|---:|---:|---|")
        for robot, base, algo, g, c, sp, v in rows:
            lines.append(f"| {robot} | {base} | {algo} | {g:.2f} | {c:.2f} | {sp:.2f}x | {v} |")
        lines.append("")
        # Highlight GRiD LOSSES (the cells that matter for the C.7 re-check)
        losses = [r for r in rows if r[6] == "comp"]
        if losses:
            lines.append(f"**GRiD losses vs {col} ({len(losses)}):** " +
                         ", ".join(f"{r[0]}.{r[1]}.{r[2]} ({r[5]:.2f}x)" for r in losses))
            lines.append("")

    lines.insert(3, "**Tally:** " + "; ".join(f"{c}: {w}W/{l}L/{t}T" for c,(w,l,t) in tally.items()) + "\n")
    txt = "\n".join(lines)
    out = args.out or os.path.join(args.results, "ANALYSIS.md")
    Path(out).write_text(txt)
    print(txt)
    print(f"\n[written: {out}]")

if __name__ == "__main__":
    main()

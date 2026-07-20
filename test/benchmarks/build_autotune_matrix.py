#!/usr/bin/env python3
"""Merge swept thread-picks + kernel limits into the codegen-ready autotune
matrix (A1a Phase 1). MEASUREMENT-ONLY: pure JSON join + the §1c guard. Reads
nothing but JSON the sweep / limits-collector already produced; writes one file.

Inputs (auto-discovered under results/ unless overridden):
  * Swept picks — any GRiD run.py / run_multi_version.py output JSON carrying
    ``results[robot][base].algo_picks[algo]`` (schema-2 from
    ``_autotune_pick_winners``): {tier_optimal, threads_optimal, us_at_optimal,
    sweep:{tier:{threads:us}}, tier_equiv_to?}.
  * Kernel limits — results/kernel_limits_<host>.json from
    collect_kernel_limits.py: limits[robot][base][algo][tier] ->
    {max_threads, min_smem, num_regs}.

The §1c (agent_debugging_guide) BOGUS-FAST guard is applied on the join:
  * suggested_threads = min(swept winner, max_threads). A swept winner can never
    legitimately exceed the kernel's maxThreadsPerBlock; if it does (stale pick,
    or a reading at threads > cap), we clamp DOWN to max_threads AND drop the
    individual sweep readings taken above max_threads (a failed launch reads as
    fastest — never trust it).
  * A cell whose min_smem exceeds the device dynamic-smem cap at EVERY tier is
    emitted as an ``unfit`` sentinel entry, NOT a bogus matrix row.

Output schema (schema 3):
  metadata: {hostname, gpu_name, cuda_arch, grid_rbd_version, glass_commit,
             autotune_N, thread_grid, schema:3, generated_utc}
  matrix[robot][base][algo][tier] -> {suggested_threads, max_threads, min_smem,
      num_regs, us_at_optimal, batch_N, sweep:{threads->us}, tier_equiv_to?}
  unfit: [ {robot, base, algo, min_smem_by_tier, device_smem_cap} ]

batch is fixed at N=DEFAULT_AUTOTUNE_N (256) for Phase 1; batch_N is stamped per
leaf + autotune_N in metadata so a later batch dimension is non-breaking.

Usage:
    python test/benchmarks/build_autotune_matrix.py
    python test/benchmarks/build_autotune_matrix.py \
        --picks results/iiwa14_fixed_grid_<host>.json \
        --kernel-limits results/kernel_limits_<host>.json \
        --robot iiwa14 --base fixed --algo crba
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from test.benchmarks.baselines.grid.run import (  # noqa: E402
    DEFAULT_AUTOTUNE_N,
    DEFAULT_AUTOTUNE_THREAD_GRID,
)

RESULTS_DIR = REPO_ROOT / "test" / "benchmarks" / "results"
SCHEMA = 3


def _host() -> str:
    return platform.node().replace(" ", "_")


def _grid_rbd_version() -> str | None:
    try:
        sys.path.insert(0, str(REPO_ROOT / "bindings"))
        from grid_rbd import __version__  # type: ignore
        return __version__
    except Exception:  # noqa: BLE001
        return None


def _glass_commit() -> str | None:
    import subprocess
    glass_dir = REPO_ROOT / "external" / "GLASS"
    if not glass_dir.exists():
        # GLASS may be vendored under GRiDCodeGenerator's submodule path.
        for cand in REPO_ROOT.glob("**/GLASS"):
            if (cand / ".git").exists() or cand.is_dir():
                glass_dir = cand
                break
    try:
        r = subprocess.run(["git", "-C", str(glass_dir), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    except OSError:
        pass
    return None


def _gpu_name() -> str | None:
    import shutil
    import subprocess
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        r = subprocess.run([smi, "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            line = r.stdout.strip().splitlines()
            return line[0].strip() if line else None
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def discover_picks(results_dir: Path) -> list[Path]:
    """Every JSON under results_dir whose body carries `algo_picks`."""
    out: list[Path] = []
    for p in sorted(results_dir.rglob("*.json")):
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        results = doc.get("results")
        if not isinstance(results, dict):
            continue
        found = False
        for _robot, bases in results.items():
            if not isinstance(bases, dict):
                continue
            for _base, block in bases.items():
                if isinstance(block, dict) and block.get("algo_picks"):
                    found = True
                    break
            if found:
                break
        if found:
            out.append(p)
    return out


def iter_picks(doc: dict):
    """Yield (robot, base, algo, pick_dict, autotune_meta) from a grid output JSON."""
    meta = doc.get("metadata", {})
    at_meta = meta.get("autotune_threads", {}) if isinstance(meta, dict) else {}
    results = doc.get("results", {})
    if not isinstance(results, dict):
        return
    for robot, bases in results.items():
        if not isinstance(bases, dict):
            continue
        for base, block in bases.items():
            if not isinstance(block, dict):
                continue
            picks = block.get("algo_picks")
            if not isinstance(picks, dict):
                continue
            for algo, pick in picks.items():
                if isinstance(pick, dict):
                    yield robot, base, algo, pick, at_meta


# ---------------------------------------------------------------------------
# §1c guard + join
# ---------------------------------------------------------------------------
def _guarded_sweep(tier_sweep: dict, max_threads: int | None) -> dict[str, float]:
    """Drop any swept reading taken at threads > max_threads (a launch above the
    register cap reads as 'fastest' — never trust it). Keys are str thread counts."""
    out: dict[str, float] = {}
    for th_str, us in tier_sweep.items():
        try:
            th = int(th_str)
        except (TypeError, ValueError):
            continue
        if max_threads is not None and th > max_threads:
            continue
        out[str(th)] = float(us)
    return out


def build_matrix(
    pick_docs: list[dict],
    limits: dict,
    *,
    device_smem_cap: int | None,
    autotune_N: int,
    thread_grid: list[int],
    only_robot: str | None = None,
    only_base: str | None = None,
    only_algo: str | None = None,
) -> tuple[dict, list[dict]]:
    """Returns (matrix, unfit). matrix[robot][base][algo][tier]; unfit list."""
    matrix: dict = {}
    unfit: list[dict] = []
    limit_table = limits.get("limits", {})

    for doc in pick_docs:
        for robot, base, algo, pick, _at_meta in iter_picks(doc):
            if only_robot and robot != only_robot:
                continue
            if only_base and base != only_base:
                continue
            if only_algo and algo != only_algo:
                continue

            algo_limits = (
                limit_table.get(robot, {}).get(base, {}).get(algo, {})
            )

            # Per-tier sweep from the pick (schema-2 `sweep`). Fall back to the
            # flat `sweep_us` under the winning tier if `sweep` is absent.
            tier_sweeps: dict[str, dict] = pick.get("sweep") or {}
            if not tier_sweeps and pick.get("sweep_us"):
                tier_sweeps = {pick.get("tier_optimal", "shared"): pick["sweep_us"]}

            # §1c unfit check: does smem exceed the device cap at EVERY tier?
            min_smem_by_tier = {
                t: algo_limits.get(t, {}).get("min_smem") for t in algo_limits
            }
            if device_smem_cap is not None and min_smem_by_tier:
                known = {t: s for t, s in min_smem_by_tier.items() if s is not None}
                if known and all(s > device_smem_cap for s in known.values()):
                    unfit.append({
                        "robot": robot, "base": base, "algo": algo,
                        "min_smem_by_tier": min_smem_by_tier,
                        "device_smem_cap": device_smem_cap,
                    })
                    continue

            tier_equiv = pick.get("tier_equiv_to", {})
            win_tier = pick.get("tier_optimal")
            win_threads = pick.get("threads_optimal")
            win_us = pick.get("us_at_optimal")

            algo_cell = (
                matrix.setdefault(robot, {})
                      .setdefault(base, {})
                      .setdefault(algo, {})
            )

            # Emit a leaf per tier that has either a limit entry OR a sweep.
            tiers = sorted(set(tier_sweeps) | set(algo_limits))
            for tier in tiers:
                lim = algo_limits.get(tier, {})
                max_threads = lim.get("max_threads")
                guarded = _guarded_sweep(tier_sweeps.get(tier, {}), max_threads)

                # suggested_threads: the winning-tier winner clamped to this
                # tier's max_threads. For non-winning tiers we suggest that
                # tier's own argmin over its guarded sweep (if any), else None.
                if tier == win_tier and win_threads is not None:
                    sug = int(win_threads)
                    if max_threads is not None:
                        sug = min(sug, int(max_threads))
                    us_opt = float(win_us) if win_us is not None else None
                elif guarded:
                    best_th = min(guarded, key=lambda k: guarded[k])
                    sug = int(best_th)
                    us_opt = guarded[best_th]
                else:
                    sug = int(max_threads) if max_threads is not None else None
                    us_opt = None

                leaf = {
                    "suggested_threads": sug,
                    "max_threads": max_threads,
                    "min_smem": lim.get("min_smem"),
                    "num_regs": lim.get("num_regs"),
                    "us_at_optimal": us_opt,
                    "batch_N": autotune_N,
                    "sweep": guarded,
                }
                if tier in tier_equiv:
                    leaf["tier_equiv_to"] = tier_equiv[tier]
                algo_cell[tier] = leaf

    return matrix, unfit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--picks", type=Path, nargs="*", default=None,
                    help="Explicit grid output JSON(s) carrying algo_picks "
                         "(default: auto-discover under results/).")
    ap.add_argument("--kernel-limits", type=Path, default=None,
                    help="kernel_limits_<host>.json (default: results/kernel_limits_<host>.json).")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--output", type=Path, default=None,
                    help="Output matrix path (default: results/autotune_matrix_<host>.json).")
    ap.add_argument("--device-smem-cap", type=int, default=None,
                    help="Device dynamic-smem cap (bytes) for the unfit check. "
                         "Default: read from kernel_limits metadata, else 101376 "
                         "(sm_120 ~99KB optin).")
    ap.add_argument("--robot", default=None)
    ap.add_argument("--base", default=None, choices=["fixed", "floating"])
    ap.add_argument("--algo", default=None)
    ap.add_argument("--stamp", default=None,
                    help="Override generated_utc (ISO8601). Default: now (UTC).")
    args = ap.parse_args()

    results_dir = args.results_dir.resolve()

    # Kernel limits.
    kl_path = args.kernel_limits or (results_dir / f"kernel_limits_{_host()}.json")
    if not kl_path.exists():
        print(f"ERROR: kernel limits not found at {kl_path}. "
              f"Run collect_kernel_limits.py first.", file=sys.stderr)
        sys.exit(1)
    limits = _load_json(kl_path)

    device_smem_cap = args.device_smem_cap
    if device_smem_cap is None:
        device_smem_cap = limits.get("metadata", {}).get("device_smem_cap_bytes")
    if device_smem_cap is None:
        device_smem_cap = 101376  # sm_120 ~99 KB dynamic optin (fallback)

    # Picks.
    if args.picks:
        pick_paths = [p.resolve() for p in args.picks]
    else:
        pick_paths = discover_picks(results_dir)
    if not pick_paths:
        print(f"ERROR: no JSON with algo_picks found under {results_dir}. "
              f"Run the sweep (run.py --autotune-threads) first.", file=sys.stderr)
        sys.exit(1)
    print(f"[matrix] picks from {len(pick_paths)} file(s):")
    for p in pick_paths:
        print(f"    {p}")
    pick_docs = [_load_json(p) for p in pick_paths]

    # Determine autotune_N / thread_grid from the picks metadata (fall back to
    # codegen defaults). Phase-1 fixes batch at DEFAULT_AUTOTUNE_N.
    autotune_N = DEFAULT_AUTOTUNE_N
    thread_grid = list(DEFAULT_AUTOTUNE_THREAD_GRID)
    for doc in pick_docs:
        at = doc.get("metadata", {}).get("autotune_threads", {})
        if at.get("autotune_N"):
            autotune_N = int(at["autotune_N"])
        if at.get("thread_grid"):
            thread_grid = list(at["thread_grid"])

    matrix, unfit = build_matrix(
        pick_docs, limits,
        device_smem_cap=device_smem_cap,
        autotune_N=autotune_N, thread_grid=thread_grid,
        only_robot=args.robot, only_base=args.base, only_algo=args.algo,
    )

    generated_utc = args.stamp or datetime.datetime.now(
        datetime.timezone.utc).isoformat()
    metadata = {
        "hostname": _host(),
        "gpu_name": _gpu_name(),
        "cuda_arch": limits.get("metadata", {}).get("cuda_arch"),
        "grid_rbd_version": _grid_rbd_version(),
        "glass_commit": _glass_commit(),
        "autotune_N": autotune_N,
        "thread_grid": thread_grid,
        "device_smem_cap_bytes": device_smem_cap,
        "schema": SCHEMA,
        "generated_utc": generated_utc,
    }

    out_path = args.output or (results_dir / f"autotune_matrix_{_host()}.json")
    doc = {"metadata": metadata, "matrix": matrix, "unfit": unfit}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    n_cells = sum(
        len(algos)
        for bases in matrix.values()
        for algos in bases.values()
    )
    print(f"[matrix] wrote {out_path}")
    print(f"[matrix] {n_cells} (robot,base,algo) cells; {len(unfit)} unfit")


if __name__ == "__main__":
    main()

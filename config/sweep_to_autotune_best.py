#!/usr/bin/env python3
"""Build a CLEAN ``autotune_best_<host>.json`` from one tier-sweep output dir.

Why this exists: the incremental autotune producers (today
``test/benchmarks/autotune_ffi.py`` and the tier-sweep tooling, e.g.
``per_algo_bench.py --mode autotune`` / ``run_tier_sweep_phased.sh``; the retired
``run.py --autotune-threads`` before them) update a shared
``results/autotune_best_<host>.json`` (read-modify-write), so over many runs it
accumulates STALE per-robot entries and mixed key conventions. For a trustworthy
launch-config bake you want exactly ONE run's picks. This reads every
``*_grid_glass.json`` under a sweep dir, pulls ``results[robot][base].algo_picks``,
and emits a fresh autotune_best containing ONLY that run's valid (non-null) picks.

Algo keys are kept as emitted by the sweep (the long GRiD symbol, e.g.
``forward_dynamics``); ``config/autotune_to_launch_config.py`` maps them to the
short launch-config key at bake time and accepts either convention.

Pairs with the bake step:
    python config/sweep_to_autotune_best.py --sweep-dir <dir> --out /tmp/best.json
    python config/autotune_to_launch_config.py --robot go2 --bases floating \
        --gpu-key rtx5090_sm120 --cuda-arch sm_120 \
        --gpu-name "NVIDIA GeForce RTX 5090" --best /tmp/best.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _host() -> str:
    return platform.node().replace(" ", "_")


def build_best(sweep_dir: Path) -> tuple[dict, list[str]]:
    """Return ({best, metadata}, warnings) from a sweep dir's grid_glass JSONs."""
    best: dict = {}
    meta: dict = {}
    warns: list[str] = []
    jsons = sorted(sweep_dir.rglob("*_grid_glass.json"))
    if not jsons:
        warns.append(f"no *_grid_glass.json under {sweep_dir}")
    for j in jsons:
        try:
            doc = json.loads(j.read_text())
        except (OSError, json.JSONDecodeError) as e:
            warns.append(f"skip unreadable {j.name}: {e}")
            continue
        md = doc.get("metadata", {})
        for k in ("gpu_name", "cuda_arch", "compute_capability", "cpu"):
            if k in md and k not in meta:
                meta[k] = md[k]
        for robot, bases in (doc.get("results") or {}).items():
            for base, node in (bases or {}).items():
                picks = (node or {}).get("algo_picks") or {}
                for algo, info in picks.items():
                    if not isinstance(info, dict):
                        continue
                    us = info.get("us_at_optimal")
                    tier = info.get("tier_optimal")
                    threads = info.get("threads_optimal")
                    if not us or tier is None or threads is None:
                        continue  # null / OOM pick (e.g. h2_plus) -> omit
                    cell = best.setdefault(robot, {}).setdefault(base, {})
                    # last writer wins; sweeps don't double-time a (robot,base,algo)
                    cell[algo] = {
                        "tier": tier,
                        "threads": int(threads),
                        "us": round(float(us), 5),
                    }
    return {"best": best, "metadata": meta}, warns


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, required=True,
                    help="tier_sweep_phased_* output dir to harvest picks from.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output autotune_best path "
                         "(default: <sweep-dir>/autotune_best_<host>.json).")
    args = ap.parse_args()

    if not args.sweep_dir.is_dir():
        print(f"ERROR: not a dir: {args.sweep_dir}", file=sys.stderr)
        sys.exit(1)

    doc, warns = build_best(args.sweep_dir)
    for w in warns:
        print(f"WARN: {w}", file=sys.stderr)

    best = doc["best"]
    if not best:
        print("ERROR: no valid picks harvested.", file=sys.stderr)
        sys.exit(1)

    out = args.out or (args.sweep_dir / f"autotune_best_{_host()}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print(f"[autotune_best] wrote {out}")
    for robot in sorted(best):
        for base in sorted(best[robot]):
            print(f"  {robot}/{base}: {len(best[robot][base])} algos")


if __name__ == "__main__":
    main()

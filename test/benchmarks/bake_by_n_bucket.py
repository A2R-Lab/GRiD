#!/usr/bin/env python
"""Bake a small-batch autotune bucket into the launch-config JSONs (E6 batch-switch).

Reads a winners dataset (parse_n16_winners.py output: {robots: {robot: {base:
{algo: {threads, tier, us, max}}}}}) and writes each robot's
``<profile>_bases_by_n[str(n)][base]`` block as ``{algo: {tier, threads}}`` in
``config/launch_configs/<robot>/<gpu>.json``.

Entries are written even when the bucket's tier differs from the baked ffi
tier — the runtime overlay (RobotHandle.apply_batch_overlay) SKIPS those, and
keeping them in the config documents the sweep winner for the next bake.

Usage:
  .venv/bin/python test/benchmarks/bake_by_n_bucket.py \
      --winners test/benchmarks/results/night4_20260827/n16_winners_20260828.json \
      --n 16 [--profile ffi] [--gpu rtx5090_sm120] [--dry-run]
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winners", required=True)
    ap.add_argument("--n", type=int, required=True, help="bucket batch size (the sweep's --n)")
    ap.add_argument("--profile", default="ffi")
    ap.add_argument("--gpu", default="rtx5090_sm120")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    winners = json.loads(Path(args.winners).read_text())
    src = winners.get("meta", {}).get("source", args.winners)
    key = args.profile + "_bases_by_n"
    for robot, bases in winners["robots"].items():
        cfg_path = REPO / "config" / "launch_configs" / robot / (args.gpu + ".json")
        if not cfg_path.exists():
            print(f"[skip] {robot}: no {cfg_path}")
            continue
        doc = json.loads(cfg_path.read_text())
        block = doc.setdefault(key, {}).setdefault(str(args.n), {})
        wrote = 0
        for base, algos in bases.items():
            dst = block.setdefault(base, {})
            for algo, w in algos.items():
                dst[algo] = {"tier": w["tier"], "threads": int(w["threads"])}
                wrote += 1
        meta = doc.setdefault(args.profile + "_by_n_meta", {})
        meta[str(args.n)] = {"source": src}
        print(f"[{'dry' if args.dry_run else 'bake'}] {robot}: {wrote} entries -> {key}['{args.n}']")
        if not args.dry_run:
            cfg_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

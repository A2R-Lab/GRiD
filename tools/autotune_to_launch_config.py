#!/usr/bin/env python3
"""Convert an ``autotune_best_<host>.json`` slice into a
``launch_configs/<robot>/<gpu_key>.json`` override (the A1 launch-config schema).

The autotune sweep (``run.py --autotune-threads``, driven by
``tools/autotune_robot.sh``) writes the canonical per-host artifact
``test/benchmarks/results/autotune_best_<host>.json``::

    {"metadata": {hostname, gpu_name, cuda_arch}, "best": {robot: {base: {algo: {tier, threads, us}}}}}

``launch_configs/<robot>/<gpu>.json`` (consumed by codegen to emit
``grid_launch_config.cuh``) wants the documented schema::

    {gpu, cuda_arch, gpu_name, autotune_N, source,
     bases: {fixed|floating: {algo: {tier, threads, us_at_optimal}}}}

This script reads ONE robot's slice out of the autotune_best file and writes the
launch_configs override. Pure JSON transform — no GPU, no build. Called by
``tools/autotune_robot.sh`` after the sweep, but usable standalone to (re)convert
an existing autotune_best file.

Usage::

    python tools/autotune_to_launch_config.py \
        --robot iiwa14 --bases fixed floating \
        --gpu-key rtx5090_sm120 --cuda-arch sm_120 \
        --gpu-name "NVIDIA GeForce RTX 5090" --autotune-N 256 \
        [--best test/benchmarks/results/autotune_best_<host>.json] \
        [--source "GRiD autotune sweep 2026-06-13"] \
        [--out launch_configs/iiwa14/rtx5090_sm120.json]
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _host() -> str:
    return platform.node().replace(" ", "_")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--robot", required=True,
                    help="Robot id (matches the codegen/URDF name, e.g. iiwa14).")
    ap.add_argument("--bases", nargs="+", default=["fixed", "floating"],
                    choices=["fixed", "floating"],
                    help="Bases to include (default: fixed floating).")
    ap.add_argument("--gpu-key", required=True,
                    help="GPU key <model>_<arch> lowercased, e.g. rtx5090_sm120. "
                         "Becomes both the `gpu` field and the output filename.")
    ap.add_argument("--cuda-arch", required=True,
                    help="sm_XX (e.g. sm_120).")
    ap.add_argument("--gpu-name", required=True,
                    help="Human GPU name from nvidia-smi (e.g. 'NVIDIA GeForce RTX 5090').")
    ap.add_argument("--autotune-N", type=int, default=256,
                    help="Batch N the sweep timed at (default 256).")
    ap.add_argument("--best", type=Path, default=None,
                    help="autotune_best_<host>.json (default: "
                         "results/autotune_best_<host>.json).")
    ap.add_argument("--source", default=None,
                    help="`source` provenance string (default: "
                         "'GRiD autotune sweep <today>').")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (default: launch_configs/<robot>/<gpu_key>.json).")
    args = ap.parse_args()

    best_path = args.best or (
        REPO_ROOT / "test" / "benchmarks" / "results"
        / f"autotune_best_{_host()}.json")
    if not best_path.exists():
        print(f"ERROR: autotune_best file not found: {best_path}\n"
              f"       Run the autotune sweep first (tools/autotune_robot.sh).",
              file=sys.stderr)
        sys.exit(1)

    try:
        doc = json.loads(best_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"ERROR: could not read {best_path}: {e}", file=sys.stderr)
        sys.exit(1)

    best = doc.get("best", {})
    robot_slice = best.get(args.robot)
    if not robot_slice:
        have = ", ".join(sorted(best)) or "(none)"
        print(f"ERROR: robot '{args.robot}' not present in {best_path}.\n"
              f"       Robots in the file: {have}", file=sys.stderr)
        sys.exit(1)

    bases_out: dict[str, dict] = {}
    for base in args.bases:
        base_slice = robot_slice.get(base)
        if not base_slice:
            print(f"WARN: base '{base}' has no autotune picks for "
                  f"'{args.robot}' — skipping it.", file=sys.stderr)
            continue
        algos_out: dict[str, dict] = {}
        for algo, info in base_slice.items():
            # autotune_best uses {tier, threads, us}; launch_configs wants
            # {tier, threads, us_at_optimal}.
            algos_out[algo] = {
                "tier": info["tier"],
                "threads": int(info["threads"]),
                "us_at_optimal": round(float(info["us"]), 2),
            }
        bases_out[base] = dict(sorted(algos_out.items()))

    if not bases_out:
        print(f"ERROR: no usable bases for '{args.robot}' in {best_path}.",
              file=sys.stderr)
        sys.exit(1)

    today = datetime.date.today().isoformat()
    source = args.source or f"GRiD autotune sweep {today}"

    out_doc = {
        "gpu": args.gpu_key,
        "cuda_arch": args.cuda_arch,
        "gpu_name": args.gpu_name,
        "autotune_N": int(args.autotune_N),
        "source": source,
        "bases": dict(sorted(bases_out.items())),
    }

    out_path = args.out or (
        REPO_ROOT / "launch_configs" / args.robot / f"{args.gpu_key}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_doc, indent=2, sort_keys=True) + "\n")

    n_algos = sum(len(v) for v in bases_out.values())
    print(f"[launch_config] wrote {out_path}")
    print(f"[launch_config] {args.robot} {args.gpu_key}: "
          f"{', '.join(sorted(bases_out))} ({n_algos} algo picks)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run BARD timing benchmark for one robot/base combination.

BARD ("Batched Articulated Robot Dynamics") is a batched PyTorch rigid-body
dynamics library (https://github.com/YueWang996/bard-pytorch-dynamics).
Exposes rnea (id), aba (fd), crba. Others (minv, gradients, SO algorithms,
ee_pose) are mapped to null. Mirrors the Frax adapter (a Python lib timed on
its own backend), but BARD runs on PyTorch CPU/CUDA instead of JAX.

Usage:
    python test/benchmarks/baselines/bard/run.py \
        --robot iiwa14 --base fixed [--output results/iiwa14_fixed_bard_<host>.json] \
        [--device both]
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
THIS_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from test.benchmarks.timing_parser import (  # noqa: E402
    parse_grid_output, fill_nulls, build_metadata,
)

# ---------------------------------------------------------------------------
# BARD-available algorithms (others will be null after fill_nulls)
# ---------------------------------------------------------------------------
BARD_ALGOS = ["inverse_dynamics", "forward_dynamics", "crba"]

# BARD loads from URDFs (like Pinocchio + Frax). Reuse the same
# robot_descriptions modules; same URDF_PATH attribute + same joint order.
ROBOT_DESCRIPTION_MODULE: dict[str, str] = {
    "iiwa14": "robot_descriptions.iiwa14_description",
    "go2":    "robot_descriptions.go2_description",
    "g1":     "robot_descriptions.g1_description",
    "h1_2":   "robot_descriptions.h1_2_description",
}


def get_urdf_path(robot: str) -> str:
    mod_name = ROBOT_DESCRIPTION_MODULE.get(robot)
    if mod_name is None:
        raise ValueError(f"Unknown robot '{robot}'. Known: {list(ROBOT_DESCRIPTION_MODULE)}")
    import importlib
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        raise RuntimeError(
            f"robot_descriptions module '{mod_name}' not found. "
            "Install with: pip install robot_descriptions"
        )
    path = getattr(mod, "URDF_PATH", None)
    if path is None:
        raise RuntimeError(f"{mod_name} has no URDF_PATH attribute")
    return str(path)


# ---------------------------------------------------------------------------
# Run timeBARD.py and parse its output
# ---------------------------------------------------------------------------
TIMING_SCRIPT = THIS_DIR / "timeBARD.py"


def run_timing(urdf_path: str, base: str,
               device: str | None = None,
               test_iters: int | None = None) -> str:
    """Invoke timeBARD.py as a subprocess. `device` selects the torch backend
    via the BARD_DEVICE env ('cpu' or 'gpu')."""
    floating_arg = "T" if base == "floating" else "F"
    cmd = [sys.executable, str(TIMING_SCRIPT), urdf_path, floating_arg]
    env = os.environ.copy()
    if test_iters is not None:
        env["BENCH_TEST_ITERS"] = str(int(test_iters))
    if device is not None:
        env["BARD_DEVICE"] = device
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"timeBARD.py [device={device or 'auto'}] exited with code {result.returncode}:\n{result.stderr}"
        )
    return result.stdout + "\n" + result.stderr


def _parse_bard_metadata(stdout: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    in_block = False
    for line in stdout.splitlines():
        if "=== BEGIN BARD METADATA ===" in line:
            in_block = True
            continue
        if "=== END BARD METADATA ===" in line:
            break
        if in_block and ":" in line:
            k, _, v = line.partition(":")
            key = k.strip().lower().replace(" ", "_")
            if key in ("bard_version", "torch_version", "bard_device", "nq", "nv"):
                meta[key] = v.strip()
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run BARD benchmark for one robot/base")
    parser.add_argument("--robot", required=True, choices=list(ROBOT_DESCRIPTION_MODULE))
    parser.add_argument("--base", required=True, choices=["fixed", "floating"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ee-frame", default=None,
                        help="Unused for BARD (kept for harness CLI parity)")
    parser.add_argument("--test-iters", type=int, default=None,
                        help="Override TEST_ITERS (default 500). Number of timed reps; "
                             "bump for more stable medians.")
    parser.add_argument("--device", choices=["cpu", "gpu", "both"], default="both",
                        help="Which torch backend(s) to time. 'both' runs the timing "
                             "subprocess twice and writes JSON with both `bard_cpu` "
                             "and `bard_gpu` sub-keys. (BARD advertises CPU + CUDA.) "
                             "Default: both. A missing CUDA device makes the gpu run "
                             "fail gracefully and only bard_cpu is written.")
    args = parser.parse_args()

    build_dir = REPO_ROOT / "test" / "benchmarks" / "results"
    build_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        host = platform.node().replace(" ", "_")
        args.output = build_dir / f"{args.robot}_{args.base}_bard_{host}.json"

    try:
        urdf_path = get_urdf_path(args.robot)
    except Exception as e:
        print(f"  [bard] ERROR resolving URDF: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[bard] {args.robot} {args.base} — URDF: {urdf_path}")

    devices = ["cpu", "gpu"] if args.device == "both" else [args.device]

    results_section: dict[str, dict] = {}
    meta = build_metadata(include_gpu=True, include_pinocchio=False)
    meta["robot"] = args.robot
    meta["base"]  = args.base

    for dev in devices:
        col_key = f"bard_{dev}"
        print(f"  [bard] running timeBARD.py (device={dev})...")
        try:
            output = run_timing(urdf_path, args.base, device=dev, test_iters=args.test_iters)
        except Exception as e:
            print(f"  [bard] ERROR (device={dev}): {e}", file=sys.stderr)
            # Continue to the other device — partial results better than none
            # (e.g. no CUDA GPU available → only bard_cpu).
            continue

        timings = parse_grid_output(output)
        for algo in list(timings.keys()):
            if algo not in BARD_ALGOS:
                timings[algo] = None
        filled = fill_nulls(timings)
        results_section[col_key] = filled

        per_dev_meta = _parse_bard_metadata(output)
        for k, v in per_dev_meta.items():
            meta[f"{col_key}_{k}"] = v

    if not results_section:
        print("  [bard] all device runs failed", file=sys.stderr)
        sys.exit(1)

    result = {"metadata": meta, "results": {args.robot: {args.base: results_section}}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"  [bard] results saved: {args.output}")

    for dev, filled in results_section.items():
        print(f"  [{dev}]")
        for algo, entry in sorted(filled.items()):
            if entry is None:
                continue
            if "single_us" in entry:
                v = entry["single_us"]["mean"]
                print(f"    {algo}: {v:.2f}us (single)")


if __name__ == "__main__":
    main()

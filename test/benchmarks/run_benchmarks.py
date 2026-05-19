#!/usr/bin/env python3
"""Coordinator: run all GRiD and Pinocchio benchmarks across robots, bases, and batch sizes.

Usage:
    python test/benchmarks/run_benchmarks.py
    python test/benchmarks/run_benchmarks.py --robots iiwa14 go2 --base fixed
    python test/benchmarks/run_benchmarks.py --baselines grid --no-recompile
    python test/benchmarks/run_benchmarks.py --save-as-regression-baseline
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from test.benchmarks.timing_parser import build_metadata  # noqa: E402

RESULTS_DIR = THIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ROBOTS    = ["iiwa14", "go2", "g1"]
BASES     = ["fixed", "floating"]
BASELINES = ["grid", "pinocchio", "mjx"]

# GRiD EE frames must be fixed-joint names (zero-DOF joints in the URDF).
# Pinocchio/MJX EE frames are link/body frame names (can be any named frame).
EE_FRAMES_GRID: dict[str, str] = {
    "iiwa14": "iiwa_joint_ee",
    "go2":    "FR_foot_joint",
    "g1":     "right_hand_palm_joint",
}
EE_FRAMES_PIN_MJX: dict[str, str] = {
    "iiwa14": "iiwa_link_ee",
    "go2":    "FR_foot",
    "g1":     "right_rubber_hand",
}

# g1 locomotion EE — only available as a Pinocchio/MJX frame (no fixed ankle joint in URDF)
G1_FOOT_EE_FRAME = "right_ankle_roll_link"


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def run_baseline(
    baseline: str,
    robot: str,
    base: str,
    output: Path,
    no_recompile: bool,
    ee_frame: str,
) -> dict | None:
    script = REPO_ROOT / "test" / "benchmarks" / "baselines" / baseline / "run.py"
    cmd = [
        sys.executable, str(script),
        "--robot", robot,
        "--base", base,
        "--output", str(output),
        "--ee-frame", ee_frame,
    ]
    if no_recompile:
        cmd.append("--no-recompile")
    if baseline == "pinocchio":
        cmd.append("--no-cpu-lock")  # coordinator manages locking externally
    # mjx has no extra flags needed

    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"  [{baseline}] exited with code {result.returncode}", file=sys.stderr)
            return None
        if output.exists():
            return json.loads(output.read_text())
        return None
    except Exception as e:
        print(f"  [{baseline}] exception: {e}", file=sys.stderr)
        return None


def merge_results(all_results: list[dict]) -> dict:
    """Merge per-run JSON results into one unified result dict."""
    merged: dict = {}
    for r in all_results:
        if r is None:
            continue
        for robot, bases in r.get("results", {}).items():
            merged.setdefault(robot, {})
            for base, baselines in bases.items():
                merged[robot].setdefault(base, {})
                for baseline, data in baselines.items():
                    merged[robot][base][baseline] = data
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GRiD benchmarking suite")
    parser.add_argument("--robots",    nargs="+", default=ROBOTS,    choices=ROBOTS)
    parser.add_argument("--base",      nargs="+", default=BASES,     choices=BASES, dest="bases")
    parser.add_argument("--baselines", nargs="+", default=["grid", "pinocchio"], choices=BASELINES,
                        help="Baselines to run (default: grid pinocchio). Add 'mjx' explicitly.")
    parser.add_argument("--no-recompile",  action="store_true")
    parser.add_argument("--save-as-regression-baseline", action="store_true",
                        help="Save results to test/benchmarks/perf_baselines.json")
    args = parser.parse_args()

    host = platform.node().replace(" ", "_")
    total = len(args.robots) * len(args.bases) * len(args.baselines)
    # g1-foot is an extra entry (same robot, different EE frame) for pinocchio and mjx
    for bl in ("pinocchio", "mjx"):
        if bl in args.baselines and "g1" in args.robots:
            total += len(args.bases)

    i = 0
    all_results: list[dict] = []

    for robot in args.robots:
        for base in args.bases:
            for baseline in args.baselines:
                i += 1
                ee = (EE_FRAMES_GRID if baseline == "grid" else EE_FRAMES_PIN_MJX).get(robot, "")
                output = RESULTS_DIR / f"{robot}_{base}_{baseline}_{host}.json"
                print(f"[{ts()}] [{i}/{total}] {robot} {base} → {baseline} (EE: {ee or 'none'})...")
                try:
                    r = run_baseline(baseline, robot, base, output, args.no_recompile, ee)
                    all_results.append(r)
                    if r is not None:
                        print(f"  [{baseline}] ✓ done")
                    else:
                        print(f"  [{baseline}] ✗ failed (check output above)", file=sys.stderr)
                except Exception as e:
                    print(f"  [{baseline}] ✗ exception: {e}", file=sys.stderr)

            # g1: extra run for foot EE (pinocchio and mjx)
            for bl in ("pinocchio", "mjx"):
                if robot == "g1" and bl in args.baselines:
                    i += 1
                    output = RESULTS_DIR / f"g1_foot_{base}_{bl}_{host}.json"
                    print(f"[{ts()}] [{i}/{total}] g1-foot {base} → {bl} (EE: {G1_FOOT_EE_FRAME})...")
                    try:
                        r = run_baseline(bl, "g1", base, output, args.no_recompile, G1_FOOT_EE_FRAME)
                        all_results.append(r)
                        if r is not None:
                            print(f"  [{bl}-g1-foot] ✓ done")
                        else:
                            print(f"  [{bl}-g1-foot] ✗ failed", file=sys.stderr)
                    except Exception as e:
                        print(f"  [{bl}-g1-foot] ✗ exception: {e}", file=sys.stderr)

    # Merge into unified JSON
    merged_results = merge_results(all_results)
    meta = build_metadata(
        include_gpu="grid" in args.baselines or "mjx" in args.baselines,
        include_pinocchio="pinocchio" in args.baselines,
    )
    unified = {
        "metadata": meta,
        "results": merged_results,
    }
    unified_path = RESULTS_DIR / f"benchmark_{host}.json"
    unified_path.write_text(json.dumps(unified, indent=2, sort_keys=True) + "\n")
    print(f"\n[{ts()}] Unified results: {unified_path}")

    if args.save_as_regression_baseline:
        baseline_dest = REPO_ROOT / "test" / "benchmarks" / "perf_baselines.json"
        baseline_dest.parent.mkdir(parents=True, exist_ok=True)
        # Merge into existing baseline file if present
        existing = {}
        if baseline_dest.exists():
            existing = json.loads(baseline_dest.read_text())
        existing[host] = unified
        baseline_dest.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        print(f"[{ts()}] Regression baseline saved: {baseline_dest}")

    # Generate report
    report_script = THIS_DIR / "generate_report.py"
    print(f"[{ts()}] Generating benchmark.md...")
    subprocess.run([sys.executable, str(report_script), "--input", str(unified_path)], check=False)


if __name__ == "__main__":
    main()

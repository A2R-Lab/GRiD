#!/usr/bin/env python3
"""Run MJX timing benchmark for one robot/base combination.

Usage:
    python test/benchmarks/baselines/mjx/run.py \
        --robot iiwa14 --base fixed [--output results/iiwa14_fixed_mjx_<host>.json] \
        [--ee-frame iiwa_link_ee]
"""
from __future__ import annotations

import argparse
import json
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
# MJX-available algorithms (others will be null after fill_nulls)
# ---------------------------------------------------------------------------
MJX_ALGOS = ["id", "fd", "ee_pose", "id_du"]

# Canonical EE body names per robot in MuJoCo MJCF from robot_descriptions
DEFAULT_EE_FRAMES: dict[str, str] = {
    "iiwa14": "iiwa_link_ee",
    "go2":    "FR_foot",
    "g1":     "right_rubber_hand",
}

# ---------------------------------------------------------------------------
# MJCF path resolution via robot_descriptions
# ---------------------------------------------------------------------------
ROBOT_MJCF_MODULE: dict[str, str] = {
    "iiwa14": "robot_descriptions.iiwa14_mj_description",
    "go2":    "robot_descriptions.go2_mj_description",
    "g1":     "robot_descriptions.g1_mj_description",
}


def get_mjcf_path(robot: str) -> str:
    mod_name = ROBOT_MJCF_MODULE.get(robot)
    if mod_name is None:
        raise ValueError(f"Unknown robot '{robot}'. Known: {list(ROBOT_MJCF_MODULE)}")
    import importlib
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        raise RuntimeError(
            f"robot_descriptions module '{mod_name}' not found. "
            "Install with: pip install robot_descriptions"
        )
    for attr in ("MJCF_PATH", "PACKAGE_PATH", "XML_PATH"):
        path = getattr(mod, attr, None)
        if path is not None:
            return str(path)
    raise RuntimeError(
        f"robot_descriptions module {mod_name} has no MJCF_PATH/PACKAGE_PATH/XML_PATH attribute"
    )


# ---------------------------------------------------------------------------
# Run timeMJX.py and parse its output
# ---------------------------------------------------------------------------
TIMING_SCRIPT = THIS_DIR / "timeMJX.py"


def run_timing(mjcf_path: str, base: str, ee_frame: str) -> str:
    floating_arg = "T" if base == "floating" else "F"
    cmd = [sys.executable, str(TIMING_SCRIPT), mjcf_path, floating_arg]
    if ee_frame:
        cmd.append(ee_frame)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"timeMJX.py exited with code {result.returncode}:\n{result.stderr}"
        )
    return result.stdout + "\n" + result.stderr


# ---------------------------------------------------------------------------
# MJX-specific metadata from timeMJX stdout
# ---------------------------------------------------------------------------
def _parse_mjx_metadata(stdout: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    in_block = False
    for line in stdout.splitlines():
        if "=== BEGIN MJX METADATA ===" in line:
            in_block = True
            continue
        if "=== END MJX METADATA ===" in line:
            break
        if in_block and ":" in line:
            k, _, v = line.partition(":")
            key = k.strip().lower().replace(" ", "_")
            if key in ("mujoco_version", "jax_version", "jax_backend"):
                meta[key] = v.strip()
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run MJX benchmark for one robot/base")
    parser.add_argument("--robot", required=True, choices=list(ROBOT_MJCF_MODULE))
    parser.add_argument("--base", required=True, choices=["fixed", "floating"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ee-frame", default=None,
                        help="EE body name in MJCF (default: per-robot canonical)")
    args = parser.parse_args()

    ee_frame  = args.ee_frame or DEFAULT_EE_FRAMES.get(args.robot, "")
    build_dir = REPO_ROOT / "test" / "benchmarks" / "results"
    build_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        host = platform.node().replace(" ", "_")
        args.output = build_dir / f"{args.robot}_{args.base}_mjx_{host}.json"

    try:
        mjcf_path = get_mjcf_path(args.robot)
    except Exception as e:
        print(f"  [mjx] ERROR resolving MJCF: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[mjx] {args.robot} {args.base} — MJCF: {mjcf_path}")
    print(f"  [mjx] running timeMJX.py (EE frame: {ee_frame or 'none'})...")

    try:
        output = run_timing(mjcf_path, args.base, ee_frame)
    except Exception as e:
        print(f"  [mjx] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # parse_grid_output handles the same label format that timeMJX.py emits
    timings = parse_grid_output(output)
    # Zero out algos that MJX doesn't support (so they appear as null, not absent)
    for algo in list(timings.keys()):
        if algo not in MJX_ALGOS:
            timings[algo] = None
    filled = fill_nulls(timings)

    meta = build_metadata(include_gpu=True, include_pinocchio=False)
    meta.update(_parse_mjx_metadata(output))
    meta["robot"]    = args.robot
    meta["base"]     = args.base
    meta["ee_frame"] = ee_frame

    result = {"metadata": meta, "results": {args.robot: {args.base: {"mjx": filled}}}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"  [mjx] results saved: {args.output}")

    for algo, entry in sorted(filled.items()):
        if entry is None:
            print(f"    {algo}: null")
        elif "single_us" in entry:
            v = entry["single_us"]["mean"]
            print(f"    {algo}: {v:.2f}us (single)")


if __name__ == "__main__":
    main()

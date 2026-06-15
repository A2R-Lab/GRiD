#!/usr/bin/env python3
"""Run MuJoCo Warp (MJWarp) timing benchmark for one robot/base combination.

UNTESTED: there is no `mujoco_warp` package installed on this machine yet. This
adapter is a first draft modeled byte-for-byte on baselines/mjx/run.py (the
closest analog — also MuJoCo MJCF + GPU). It shells out to `timeMujocoWarp.py`
and reuses the SAME shared helpers (parse_grid_output / fill_nulls /
build_metadata), so its JSON is drop-in compatible with generate_report.py under
the baseline key "mujoco_warp".

MJWarp (https://github.com/google-deepmind/mujoco_warp) is the NVIDIA-Warp-based
GPU successor to MJX, advertised as faster than MJX. It is GPU-only (Warp targets
NVIDIA hardware), so unlike frax there is no cpu/gpu split — a single column.

First-run things to verify (see docs/open-tasks/mujoco_warp_baseline_plan.md):
  * wp.synchronize() actually brackets each timed launch (no async leakage).
  * the batched-state assign API (d.qpos.assign(...)) matches the installed
    MJWarp Data layout (1D flattened vs 2D (nworld, n)).
  * CRBA-via-(crb + factor_m) semantics — it's a factorization, not a dense M;
    decide whether to keep or null it after seeing real numbers.
  * single-call µs are sane vs mjx (MJWarp should be in the same ballpark or
    faster at large batch; much slower at nworld=1 is expected for a GPU lib).

Usage:
    python test/benchmarks/baselines/mujoco_warp/run.py \
        --robot iiwa14 --base fixed [--output results/iiwa14_fixed_mujoco_warp_<host>.json] \
        [--ee-frame iiwa_link_ee]
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
# MJWarp-available algorithms (others will be null after fill_nulls).
#
# CONFIRMED top-level exports: mjw.inverse, mjw.forward, mjw.kinematics, mjw.crb,
# mjw.factor_m. `crba` is UNCERTAIN — crb+factor_m yields a factorization (qLD),
# not a dense mass matrix like GRiD/Frax CRBA; included as the closest analog.
# Derivative/Jacobian algos (inverse_dynamics_gradient / forward_dynamics_gradient)
# use a GPU FINITE-DIFFERENCE Jacobian (wp.autograd.jacobian_fd) — mujoco_warp
# ships enable_backward=False so the autodiff Jacobian is unavailable, but the FD
# path relaunches the forward kernel with perturbed inputs and works unpatched.
# ---------------------------------------------------------------------------
MUJOCO_WARP_ALGOS = ["inverse_dynamics", "forward_dynamics", "end_effector_pose", "crba",
                     "inverse_dynamics_gradient", "forward_dynamics_gradient"]

# Canonical EE body names per robot in MuJoCo MJCF from robot_descriptions
# (same MJCF models + body names as the mjx adapter).
DEFAULT_EE_FRAMES: dict[str, str] = {
    "iiwa14": "iiwa_link_ee",
    "go2":    "FR_foot",
    "g1":     "right_rubber_hand",
    "h1_2":   "R_hand_base_link",
}

# ---------------------------------------------------------------------------
# MJCF path resolution via robot_descriptions (same modules as the mjx adapter)
# ---------------------------------------------------------------------------
ROBOT_MJCF_MODULE: dict[str, str] = {
    "iiwa14": "robot_descriptions.iiwa14_mj_description",
    "go2":    "robot_descriptions.go2_mj_description",
    "g1":     "robot_descriptions.g1_mj_description",
    "h1_2":   "robot_descriptions.h1_2_mj_description",
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
# Run timeMujocoWarp.py and parse its output
# ---------------------------------------------------------------------------
TIMING_SCRIPT = THIS_DIR / "timeMujocoWarp.py"


def run_timing(mjcf_path: str, base: str, ee_frame: str,
               test_iters: int | None = None) -> str:
    floating_arg = "T" if base == "floating" else "F"
    cmd = [sys.executable, str(TIMING_SCRIPT), mjcf_path, floating_arg]
    if ee_frame:
        cmd.append(ee_frame)
    env = os.environ.copy()
    if test_iters is not None:
        env["BENCH_TEST_ITERS"] = str(int(test_iters))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"timeMujocoWarp.py exited with code {result.returncode}:\n{result.stderr}"
        )
    return result.stdout + "\n" + result.stderr


# ---------------------------------------------------------------------------
# MJWarp-specific metadata from timeMujocoWarp stdout
# ---------------------------------------------------------------------------
def _parse_mujoco_warp_metadata(stdout: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    in_block = False
    for line in stdout.splitlines():
        if "=== BEGIN MUJOCO_WARP METADATA ===" in line:
            in_block = True
            continue
        if "=== END MUJOCO_WARP METADATA ===" in line:
            break
        if in_block and ":" in line:
            k, _, v = line.partition(":")
            key = k.strip().lower().replace(" ", "_")
            if key in ("mujoco_version", "warp_version", "warp_device",
                       "mujoco_warp_version"):
                meta[key] = v.strip()
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run MuJoCo Warp benchmark for one robot/base")
    parser.add_argument("--robot", required=True, choices=list(ROBOT_MJCF_MODULE))
    parser.add_argument("--base", required=True, choices=["fixed", "floating"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ee-frame", default=None,
                        help="EE body name in MJCF (default: per-robot canonical)")
    parser.add_argument("--device", choices=["gpu"], default="gpu",
                        help="MJWarp is GPU-only (NVIDIA Warp). Kept for harness "
                             "CLI parity; only 'gpu' is valid.")
    parser.add_argument("--test-iters", type=int, default=None,
                        help="Override TEST_ITERS (default 500). Number of timed reps per "
                             "single-call or per batch size; bump for more stable medians.")
    args = parser.parse_args()

    ee_frame  = args.ee_frame or DEFAULT_EE_FRAMES.get(args.robot, "")
    build_dir = REPO_ROOT / "test" / "benchmarks" / "results"
    build_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        host = platform.node().replace(" ", "_")
        args.output = build_dir / f"{args.robot}_{args.base}_mujoco_warp_{host}.json"

    try:
        mjcf_path = get_mjcf_path(args.robot)
    except Exception as e:
        print(f"  [mujoco_warp] ERROR resolving MJCF: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[mujoco_warp] {args.robot} {args.base} — MJCF: {mjcf_path}")
    print(f"  [mujoco_warp] running timeMujocoWarp.py (EE frame: {ee_frame or 'none'})...")

    try:
        output = run_timing(mjcf_path, args.base, ee_frame, test_iters=args.test_iters)
    except Exception as e:
        print(f"  [mujoco_warp] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # parse_grid_output handles the same label format that timeMujocoWarp.py emits.
    timings = parse_grid_output(output)
    # Zero out algos that MJWarp doesn't support (so they appear as null, not absent).
    for algo in list(timings.keys()):
        if algo not in MUJOCO_WARP_ALGOS:
            timings[algo] = None
    filled = fill_nulls(timings)

    meta = build_metadata(include_gpu=True, include_pinocchio=False)
    meta.update(_parse_mujoco_warp_metadata(output))
    meta["robot"]    = args.robot
    meta["base"]     = args.base
    meta["ee_frame"] = ee_frame

    result = {"metadata": meta, "results": {args.robot: {args.base: {"mujoco_warp": filled}}}}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"  [mujoco_warp] results saved: {args.output}")

    for algo, entry in sorted(filled.items()):
        if entry is None:
            print(f"    {algo}: null")
        elif "single_us" in entry:
            v = entry["single_us"]["mean"]
            print(f"    {algo}: {v:.2f}us (single)")


if __name__ == "__main__":
    main()

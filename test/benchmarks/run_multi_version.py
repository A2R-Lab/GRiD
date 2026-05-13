#!/usr/bin/env python3
"""Run a multi-version GRiD benchmark sweep against Pinocchio and MJX.

Columns produced (per robot/base):
  - grid_pre_glass:    GRiD at git ref d2c0d18 (last commit before the GLASS v2 work).
                       Fixed-base only — d2c0d18 harness doesn't support floating-base.
  - grid_glass:        GRiD HEAD with --linalg-backend=glass (pure-SIMT GLASS v2).
  - grid_glass_nvidia: GRiD HEAD with --linalg-backend=glass-nvidia (cuBLASDx-backed).
  - pinocchio:         CPU reference, HEAD harness with --algo parallel fan-out.
  - mjx:               MuJoCo MJX GPU reference (JAX). Requires mujoco-mjx + jax[cuda12].

Usage (single robot, fastest):
    python test/benchmarks/run_multi_version.py \
        --robots iiwa14 --bases fixed --mathdx-root /opt/nvidia/mathdx/25.12

Full sweep:
    python test/benchmarks/run_multi_version.py \
        --mathdx-root /opt/nvidia/mathdx/25.12

Worktree for the pre-glass column is created at $GRID_PRE_GLASS_WORKTREE
(default: ../GRiD-A2R-pre-glass/ relative to this repo's root).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from test.benchmarks.timing_parser import build_metadata  # noqa: E402

PRE_GLASS_REF = "d2c0d18"
RESULTS_DIR   = THIS_DIR / "results" / "comparison"
DEFAULT_WORKTREE_PATH = REPO_ROOT.parent / "GRiD-A2R-pre-glass"

ROBOTS = ("iiwa14", "go2", "g1")
BASES  = ("fixed", "floating")
COLUMNS = ("pre_glass", "glass", "glass_nvidia", "pinocchio", "mjx")

# Maps the column identifier to the baseline key used in the merged JSON
# (so generate_report.py / generate_multi_version_report.py can find them).
COLUMN_TO_BASELINE_KEY = {
    "pre_glass":    "grid_pre_glass",
    "glass":        "grid_glass",
    "glass_nvidia": "grid_glass_nvidia",
    "pinocchio":    "pinocchio",
    "mjx":          "mjx",
}

EE_FRAMES_GRID = {
    "iiwa14": "iiwa_joint_ee",
    "go2":    "FR_foot_joint",
    "g1":     "right_hand_palm_joint",
}
EE_FRAMES_PIN = {
    "iiwa14": "iiwa_link_ee",
    "go2":    "FR_foot",
    "g1":     "right_rubber_hand",
}
# MJX uses MuJoCo body names (same names as Pinocchio link names for these robots).
EE_FRAMES_MJX = EE_FRAMES_PIN


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Worktree setup
# ---------------------------------------------------------------------------
def setup_pre_glass_worktree(path: Path) -> Path:
    """Ensure a git worktree at PRE_GLASS_REF exists at `path`, with submodules.

    Idempotent: re-running with an existing worktree just verifies the HEAD.
    """
    if path.exists():
        try:
            head = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            target = subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", PRE_GLASS_REF],
                text=True,
            ).strip()
            if head == target:
                print(f"  [worktree] reusing existing worktree at {path}")
                return path
            print(f"  [worktree] WARNING: {path} HEAD ({head[:8]}) != {PRE_GLASS_REF} ({target[:8]}). "
                  f"Pass --skip-setup to use as-is or remove the directory to recreate.")
            return path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to inspect existing worktree at {path}: {e}")

    print(f"  [worktree] creating worktree at {path} (ref: {PRE_GLASS_REF})...")
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(path), PRE_GLASS_REF],
        check=True,
    )
    print(f"  [worktree] initializing submodules in {path}...")
    subprocess.run(
        ["git", "-C", str(path), "submodule", "update", "--init", "--recursive"],
        check=True,
    )
    grid_run_py = path / "test" / "benchmarks" / "baselines" / "grid" / "run.py"
    if not grid_run_py.exists():
        raise RuntimeError(f"Worktree set up but {grid_run_py} is missing")
    return path


# ---------------------------------------------------------------------------
# Per-column runners
# ---------------------------------------------------------------------------
def _grid_run_cmd(harness_repo_root: Path, robot: str, base: str,
                  output: Path, ee_frame: str, linalg_backend: str | None,
                  mathdx_root: str | None, no_recompile: bool,
                  no_rdc: bool = False) -> list[str]:
    cmd = [
        sys.executable,
        str(harness_repo_root / "test" / "benchmarks" / "baselines" / "grid" / "run.py"),
        "--robot", robot, "--base", base, "--output", str(output),
        "--ee-frame", ee_frame,
    ]
    if no_recompile:
        cmd.append("--no-recompile")
    if linalg_backend is not None:
        cmd += ["--linalg-backend", linalg_backend]
    if mathdx_root is not None:
        cmd += ["--mathdx-root", mathdx_root]
    if no_rdc:
        cmd.append("--no-rdc")
    return cmd


def run_grid_column(column: str, robot: str, base: str, *,
                    output_dir: Path, worktree_path: Path,
                    mathdx_root: str | None, no_recompile: bool,
                    no_rdc: bool = False) -> Path | None:
    """Run the appropriate GRiD harness for `column`. Returns output JSON path or None."""
    ee_frame = EE_FRAMES_GRID.get(robot, "")
    baseline_key = COLUMN_TO_BASELINE_KEY[column]
    output = output_dir / f"{robot}_{base}_{baseline_key}.json"

    if column == "pre_glass":
        if base != "fixed":
            print(f"  [{column}] skipping {robot}/{base}: pre-glass harness doesn't support floating-base")
            return None
        # pre_glass harness predates --no-rdc; don't pass it.
        cmd = _grid_run_cmd(worktree_path, robot, base, output, ee_frame,
                            linalg_backend=None, mathdx_root=None, no_recompile=no_recompile)
    elif column == "glass":
        cmd = _grid_run_cmd(REPO_ROOT, robot, base, output, ee_frame,
                            linalg_backend="glass", mathdx_root=None,
                            no_recompile=no_recompile, no_rdc=no_rdc)
    elif column == "glass_nvidia":
        cmd = _grid_run_cmd(REPO_ROOT, robot, base, output, ee_frame,
                            linalg_backend="glass-nvidia", mathdx_root=mathdx_root,
                            no_recompile=no_recompile, no_rdc=no_rdc)
    else:
        raise ValueError(f"Unknown grid column: {column}")

    print(f"[{ts()}] [{column}] {robot} {base} → {output.name}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0 or not output.exists():
        print(f"  [{column}] FAILED for {robot}/{base}", file=sys.stderr)
        return None

    # Rewrite the JSON so the baseline key is column-specific (e.g. "grid_glass")
    # instead of the generic "grid" the inner harness emits.
    _rename_grid_key(output, baseline_key)
    return output


def run_pinocchio_column(robot: str, base: str, *,
                         output_dir: Path, no_recompile: bool) -> Path | None:
    ee_frame = EE_FRAMES_PIN.get(robot, "")
    output = output_dir / f"{robot}_{base}_pinocchio.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "test" / "benchmarks" / "baselines" / "pinocchio" / "run.py"),
        "--robot", robot, "--base", base, "--output", str(output),
        "--ee-frame", ee_frame,
        "--no-cpu-lock",
    ]
    if no_recompile:
        cmd.append("--no-recompile")
    print(f"[{ts()}] [pinocchio] {robot} {base} → {output.name}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0 or not output.exists():
        print(f"  [pinocchio] FAILED for {robot}/{base}", file=sys.stderr)
        return None
    return output


def run_mjx_column(robot: str, base: str, *,
                   output_dir: Path) -> Path | None:
    ee_frame = EE_FRAMES_MJX.get(robot, "")
    output = output_dir / f"{robot}_{base}_mjx.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "test" / "benchmarks" / "baselines" / "mjx" / "run.py"),
        "--robot", robot, "--base", base, "--output", str(output),
        "--ee-frame", ee_frame,
    ]
    print(f"[{ts()}] [mjx] {robot} {base} → {output.name}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0 or not output.exists():
        print(f"  [mjx] FAILED for {robot}/{base}", file=sys.stderr)
        return None
    return output


def _rename_grid_key(json_path: Path, new_key: str) -> None:
    """Rewrite a grid run.py JSON output so the baseline key is `new_key` instead of 'grid'."""
    data = json.loads(json_path.read_text())
    results = data.get("results", {})
    for robot, bases in results.items():
        for base, baselines in bases.items():
            if "grid" in baselines and new_key != "grid":
                baselines[new_key] = baselines.pop("grid")
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Merge + report
# ---------------------------------------------------------------------------
def merge_to_unified(json_paths: list[Path]) -> dict:
    """Combine per-column per-robot JSONs into one unified results dict."""
    merged: dict = {}
    for p in json_paths:
        if p is None or not p.exists():
            continue
        data = json.loads(p.read_text())
        for robot, bases in data.get("results", {}).items():
            merged.setdefault(robot, {})
            for base, baselines in bases.items():
                merged[robot].setdefault(base, {})
                for baseline_key, algo_dict in baselines.items():
                    merged[robot][base][baseline_key] = algo_dict
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-version GRiD benchmark sweep (pre-glass + glass + glass-nvidia vs pinocchio)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--robots", nargs="+", default=list(ROBOTS), choices=list(ROBOTS))
    parser.add_argument("--bases",  nargs="+", default=list(BASES),  choices=list(BASES))
    parser.add_argument("--columns", nargs="+", default=list(COLUMNS), choices=list(COLUMNS),
                        help="Subset of columns to run (default: all five)")
    parser.add_argument("--skip", nargs="+", default=[], metavar="ROBOT_BASE",
                        help="Exclude specific robot/base combinations, e.g. "
                             "'--skip iiwa14_floating g1_fixed'. Useful when one "
                             "combination hangs the compiler.")
    parser.add_argument("--mathdx-root", default=os.environ.get("MATHDX_ROOT"),
                        help="Required for the glass_nvidia column (or set MATHDX_ROOT)")
    parser.add_argument("--worktree-path", type=Path,
                        default=Path(os.environ.get("GRID_PRE_GLASS_WORKTREE", str(DEFAULT_WORKTREE_PATH))),
                        help=f"Pre-glass worktree path (default: {DEFAULT_WORKTREE_PATH})")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Assume the pre-glass worktree already exists at --worktree-path")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR,
                        help=f"Where per-column JSONs land (default: {RESULTS_DIR})")
    parser.add_argument("--no-recompile", action="store_true",
                        help="Forward --no-recompile to inner harnesses")
    parser.add_argument("--no-rdc", action="store_true",
                        help="Drop -rdc=true from the GRiD glass / glass-nvidia compile line. "
                             "Use when ptxas hangs on floating-base kernels (older toolkits). "
                             "Batch timings unaffected; single-call timings may LICM-elide.")
    parser.add_argument("--report", type=Path,
                        default=THIS_DIR / "benchmark_multi_version.md",
                        help="Markdown report output path")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Worktree setup (only if pre_glass column is requested)
    if "pre_glass" in args.columns and not args.skip_setup:
        try:
            setup_pre_glass_worktree(args.worktree_path)
        except Exception as e:
            print(f"[fatal] worktree setup failed: {e}", file=sys.stderr)
            sys.exit(1)

    # 2) Run all (column, robot, base) combinations sequentially. GPU work is
    #    inherently serial; running in parallel would cause cache races and
    #    contend for the single GPU.
    produced: list[Path] = []
    skipped: list[tuple[str, str, str]] = []

    skip_set = {s.strip() for s in args.skip}
    for column in args.columns:
        for robot in args.robots:
            for base in args.bases:
                if f"{robot}_{base}" in skip_set:
                    print(f"  [{column}] SKIP {robot}/{base}: excluded via --skip")
                    skipped.append((column, robot, base))
                    continue
                if column == "glass_nvidia" and args.mathdx_root is None:
                    print(f"  [{column}] SKIP {robot}/{base}: --mathdx-root not provided")
                    skipped.append((column, robot, base))
                    continue
                if column == "pinocchio":
                    p = run_pinocchio_column(
                        robot, base,
                        output_dir=args.output_dir, no_recompile=args.no_recompile,
                    )
                elif column == "mjx":
                    p = run_mjx_column(
                        robot, base, output_dir=args.output_dir,
                    )
                else:
                    p = run_grid_column(
                        column, robot, base,
                        output_dir=args.output_dir, worktree_path=args.worktree_path,
                        mathdx_root=args.mathdx_root, no_recompile=args.no_recompile,
                        no_rdc=args.no_rdc,
                    )
                if p is not None:
                    produced.append(p)
                else:
                    skipped.append((column, robot, base))

    # 3) Merge + dump unified JSON
    merged = merge_to_unified(produced)
    meta = build_metadata(
        include_gpu=any(c.startswith("glass") or c == "pre_glass" for c in args.columns),
        include_pinocchio="pinocchio" in args.columns,
    )
    meta["multi_version_sweep"] = True
    meta["pre_glass_ref"]       = PRE_GLASS_REF
    meta["columns"]             = list(args.columns)
    host = platform.node().replace(" ", "_")
    unified = {"metadata": meta, "results": merged}
    unified_path = args.output_dir / f"benchmark_multi_version_{host}.json"
    unified_path.write_text(json.dumps(unified, indent=2, sort_keys=True) + "\n")
    print(f"\n[{ts()}] Unified results: {unified_path}")

    # 4) Generate markdown report
    report_script = THIS_DIR / "generate_report.py"
    print(f"[{ts()}] Generating {args.report.name}...")
    rc = subprocess.run(
        [sys.executable, str(report_script),
         "--input", str(unified_path),
         "--output", str(args.report),
         "--mode", "multi_version"],
        check=False,
    ).returncode
    if rc != 0:
        print(f"  [report] generate_report.py exited {rc}", file=sys.stderr)

    # 5) Summary
    print(f"\n[{ts()}] === Summary ===")
    print(f"  produced: {len(produced)} JSON files in {args.output_dir}")
    for p in produced:
        print(f"    ✓ {p.name}")
    if skipped:
        print(f"  skipped: {len(skipped)}")
        for col, r, b in skipped:
            print(f"    ✗ {col} {r}/{b}")
    print(f"  report:   {args.report}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a multi-version GRiD benchmark sweep against Pinocchio, MJX, and Frax.

Columns produced (per robot/base):
  - grid_pre_glass:    GRiD at git ref d2c0d18 (last commit before the GLASS v2 work).
                       Fixed-base only — d2c0d18 harness doesn't support floating-base.
  - grid_glass:        GRiD HEAD (pure-SIMT GLASS).
  - pinocchio:         CPU reference, HEAD harness with --algo parallel fan-out.
  - mjx:               MuJoCo MJX GPU reference (JAX). Requires mujoco-mjx + jax[cuda12].
  - frax:              Frax GPU reference (JAX, https://github.com/danielpmorton/frax).
                       Covers id/fd/crba/minv. Requires frax + jax[cuda12].

Usage (single robot, fastest):
    python test/benchmarks/run_multi_version.py --robots iiwa14 --bases fixed

Full sweep:
    python test/benchmarks/run_multi_version.py

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

ROBOTS = ("iiwa14", "go2", "g1", "h1_2")
BASES  = ("fixed", "floating")
# Columns the sweep knows how to run. cuBLASDx (glass_nvidia) was removed in
# v2.0 — the 2026-05-18 sweep + per-host autotune showed it loses to SIMT at
# every GEMM shape GRiD calls (notably 4×4×4 in eepose_gradient_hessian, where
# SIMT wins by 2.6×). The historical data is preserved at the
# `archive/last-cublasdx` git tag; see
# docs/source/user_guide/concepts/cublasdx_removal_design.rst.
COLUMNS = ("pre_glass", "glass", "pinocchio", "mjx", "frax")
DEFAULT_COLUMNS = ("pre_glass", "glass", "pinocchio", "mjx", "frax")

# Maps the column identifier to the baseline key used in the merged JSON
# (so generate_report.py / generate_multi_version_report.py can find them).
COLUMN_TO_BASELINE_KEY = {
    "pre_glass":    "grid_pre_glass",
    "glass":        "grid_glass",
    "pinocchio":    "pinocchio",
    "mjx":          "mjx",
    "frax":         "frax",
}

EE_FRAMES_GRID = {
    "iiwa14": "iiwa_joint_ee",
    "go2":    "FR_foot_joint",
    "g1":     "right_hand_palm_joint",
    "h1_2":   "R_base_link_joint",       # fixed joint at base of right hand (before fingers)
}
EE_FRAMES_PIN = {
    "iiwa14": "iiwa_link_ee",
    "go2":    "FR_foot",
    "g1":     "right_rubber_hand",
    "h1_2":   "R_hand_base_link",
}
# MJX uses MuJoCo body names (same names as Pinocchio link names for these robots).
EE_FRAMES_MJX = EE_FRAMES_PIN


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Pre-flight dependency checks
# ---------------------------------------------------------------------------
def _check_column_deps(column: str, worktree_path: Path) -> tuple[bool, str]:
    """Return (ok, reason_if_not_ok). Used to short-circuit columns whose
    runtime/build dependencies aren't installed on the target machine."""
    if column in ("glass", "pre_glass"):
        # Both need nvcc. The pre_glass path additionally needs the worktree.
        nvcc = subprocess.run(["which", "nvcc"], capture_output=True).returncode == 0
        if not nvcc:
            return False, "nvcc not on PATH (install CUDA Toolkit)"
        if column == "pre_glass" and not worktree_path.exists():
            return False, (f"pre_glass worktree {worktree_path} does not exist; "
                           f"orchestrator will create it on demand or pass --skip-setup")
        return True, ""
    if column == "pinocchio":
        # The pinocchio column compiles a C++ binary; needs pinocchio headers +
        # libpinocchio.so accessible via pkg-config OR cmeel.prefix. Mirror the
        # logic in baselines/pinocchio/run.py::pinocchio_cflags().
        gxx = subprocess.run(["which", "g++"], capture_output=True).returncode == 0
        if not gxx:
            return False, "g++ not on PATH (install build-essential)"
        pkg = subprocess.run(["pkg-config", "--exists", "pinocchio"],
                             capture_output=True).returncode == 0
        cmeel = (Path(sys.prefix) / "lib"
                 / f"python{sys.version_info.major}.{sys.version_info.minor}"
                 / "site-packages" / "cmeel.prefix" / "include" / "pinocchio")
        if not (pkg or cmeel.exists()):
            return False, ("pinocchio C++ headers not found "
                           "(`pip install pin` or follow README's pkg-config setup)")
        return True, ""
    if column == "mjx":
        rc = subprocess.run(
            [sys.executable, "-c", "import jax, mujoco, mujoco.mjx"],
            capture_output=True,
        ).returncode
        if rc != 0:
            return False, ("jax+mujoco+mujoco-mjx not installed "
                           "(`pip install mujoco mujoco-mjx 'jax[cuda12]'`)")
        return True, ""
    if column == "frax":
        rc = subprocess.run(
            [sys.executable, "-c", "import frax, jax"],
            capture_output=True,
        ).returncode
        if rc != 0:
            return False, "frax+jax not installed (`pip install frax 'jax[cuda12]'`)"
        return True, ""
    return False, f"unknown column '{column}'"


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
                  output: Path, ee_frame: str, no_recompile: bool,
                  no_rdc: bool = False, no_licm_barrier: bool = False,
                  single_call_iters: int | None = None,
                  batch_iters: int | None = None,
                  ptxas_opt_level: int | None = None,
                  split_compile: int | None = None,
                  ofast_compile: str | None = None) -> list[str]:
    cmd = [
        sys.executable,
        str(harness_repo_root / "test" / "benchmarks" / "baselines" / "grid" / "run.py"),
        "--robot", robot, "--base", base, "--output", str(output),
        "--ee-frame", ee_frame,
    ]
    if no_recompile:
        cmd.append("--no-recompile")
    if no_rdc:
        cmd.append("--no-rdc")
    if no_licm_barrier:
        cmd.append("--no-licm-barrier")
    if single_call_iters is not None:
        cmd += ["--single-call-iters", str(single_call_iters)]
    if batch_iters is not None:
        cmd += ["--batch-iters", str(batch_iters)]
    if ptxas_opt_level is not None:
        cmd += ["--ptxas-opt-level", str(ptxas_opt_level)]
    if split_compile is not None:
        cmd += ["--split-compile", str(split_compile)]
    if ofast_compile is not None:
        cmd += ["--ofast-compile", ofast_compile]
    return cmd


def run_grid_column(column: str, robot: str, base: str, *,
                    output_dir: Path, worktree_path: Path,
                    no_recompile: bool,
                    no_rdc: bool = False, no_licm_barrier: bool = False,
                    single_call_iters: int | None = None,
                    batch_iters: int | None = None,
                    ptxas_opt_level: int | None = None,
                    split_compile: int | None = None,
                    ofast_compile: str | None = None) -> Path | None:
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
                            no_recompile=no_recompile)
    elif column == "glass":
        effective_ptxas = ptxas_opt_level if base == "floating" else None
        cmd = _grid_run_cmd(REPO_ROOT, robot, base, output, ee_frame,
                            no_recompile=no_recompile, no_rdc=no_rdc,
                            no_licm_barrier=no_licm_barrier,
                            single_call_iters=single_call_iters, batch_iters=batch_iters,
                            ptxas_opt_level=effective_ptxas,
                            split_compile=split_compile, ofast_compile=ofast_compile)
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

    # The pre_glass worktree's grid/run.py (frozen at d2c0d18) predates the
    # single/N=16/N=256 batch-summary print added in HEAD. Re-emit it here from
    # the JSON so the stdout looks consistent across all columns.
    if column == "pre_glass":
        _print_batch_summary_from_json(output, baseline_key)
    return output


def run_pinocchio_column(robot: str, base: str, *,
                         output_dir: Path, no_recompile: bool,
                         single_call_iters: int | None = None,
                         batch_iters: int | None = None,
                         pin_num_threads: int | None = None) -> Path | None:
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
    if single_call_iters is not None:
        cmd += ["--single-call-iters", str(single_call_iters)]
    if batch_iters is not None:
        cmd += ["--batch-iters", str(batch_iters)]
    if pin_num_threads is not None:
        cmd += ["--num-threads", str(pin_num_threads)]
    print(f"[{ts()}] [pinocchio] {robot} {base} → {output.name}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0 or not output.exists():
        print(f"  [pinocchio] FAILED for {robot}/{base}", file=sys.stderr)
        return None
    return output


def run_mjx_column(robot: str, base: str, *,
                   output_dir: Path,
                   batch_iters: int | None = None) -> Path | None:
    ee_frame = EE_FRAMES_MJX.get(robot, "")
    output = output_dir / f"{robot}_{base}_mjx.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "test" / "benchmarks" / "baselines" / "mjx" / "run.py"),
        "--robot", robot, "--base", base, "--output", str(output),
        "--ee-frame", ee_frame,
    ]
    if batch_iters is not None:
        cmd += ["--test-iters", str(batch_iters)]
    print(f"[{ts()}] [mjx] {robot} {base} → {output.name}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0 or not output.exists():
        print(f"  [mjx] FAILED for {robot}/{base}", file=sys.stderr)
        return None
    return output


def run_frax_column(robot: str, base: str, *,
                    output_dir: Path,
                    batch_iters: int | None = None) -> Path | None:
    output = output_dir / f"{robot}_{base}_frax.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "test" / "benchmarks" / "baselines" / "frax" / "run.py"),
        "--robot", robot, "--base", base, "--output", str(output),
    ]
    if batch_iters is not None:
        cmd += ["--test-iters", str(batch_iters)]
    print(f"[{ts()}] [frax] {robot} {base} → {output.name}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0 or not output.exists():
        print(f"  [frax] FAILED for {robot}/{base}", file=sys.stderr)
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


def _print_batch_summary_from_json(json_path: Path, baseline_key: str) -> None:
    """Read a per-column JSON and print a single/N=16/N=256 summary table. Used
    when the inner harness doesn't print one itself (e.g., the pre_glass worktree
    at d2c0d18 predates the batch-summary update in HEAD's grid/run.py)."""
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text())
    except Exception as e:
        print(f"  [{baseline_key}] WARN: could not parse {json_path.name} for summary: {e}")
        return

    def _us(entry, key):
        v = (entry.get(key) or {}).get("median") or (entry.get(key) or {}).get("mean")
        return f"{v:.2f}" if v is not None else "—"

    is_pinocchio = baseline_key == "pinocchio"
    batch_key_16  = "batch_16_with_mem_us"  if is_pinocchio else "batch_16_compute_only_us"
    batch_key_256 = "batch_256_with_mem_us" if is_pinocchio else "batch_256_compute_only_us"

    for robot, bases in data.get("results", {}).items():
        for base, baselines in bases.items():
            algos = baselines.get(baseline_key) or {}
            if not algos:
                continue
            print(f"  [{baseline_key}] summary for {robot}/{base}:")
            for algo, entry in sorted(algos.items()):
                if entry is None:
                    print(f"      {algo}: null")
                    continue
                single = _us(entry, "single_us")
                n16    = _us(entry, batch_key_16)
                n256   = _us(entry, batch_key_256)
                label  = "compute" if not is_pinocchio else "w/mem"
                print(f"      {algo:18s} single={single:>8} us   N=16({label})={n16:>7} us   N=256({label})={n256:>7} us")


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
        description="Multi-version GRiD benchmark sweep (pre-glass + glass vs pinocchio + mjx + frax)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--robots", nargs="+", default=list(ROBOTS), choices=list(ROBOTS))
    parser.add_argument("--bases",  nargs="+", default=list(BASES),  choices=list(BASES))
    parser.add_argument("--fixed-only", action="store_true",
                        help="Shortcut for `--bases fixed`. Skips every floating-base combo "
                             "(useful when floating compile hangs and you want fixed data first). "
                             "Equivalent to --bases fixed; overrides --bases if both are set.")
    parser.add_argument("--columns", nargs="+", default=list(DEFAULT_COLUMNS), choices=list(COLUMNS),
                        help="Subset of columns to run (default: all five)")
    parser.add_argument("--skip", nargs="+", default=[], metavar="ROBOT_BASE",
                        help="Exclude specific robot/base combinations, e.g. "
                             "'--skip iiwa14_floating g1_fixed'. Useful when one "
                             "combination hangs the compiler.")
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
                        help="Drop -rdc=true from the GRiD compile line. Use when ptxas hangs "
                             "on floating-base kernels (older toolkits). Batch timings unaffected; "
                             "single-call timings may LICM-elide.")
    parser.add_argument("--no-licm-barrier", action="store_true",
                        help="Strongest hammer for ptxas hangs: suppress the anti-LICM "
                             "machinery in codegen (volatile reload + __noinline__ barrier). "
                             "Try this if --no-rdc alone doesn't fix the hang. "
                             "Batch timings unaffected; single-call may LICM-elide.")
    parser.add_argument("--ptxas-opt-level", type=int, default=None,
                        choices=[0, 1, 2, 3],
                        help="Pass `-Xptxas -O<n>` to GRiD floating-base compiles only. "
                             "SM_86-SPECIFIC WORKAROUND: on sm_86 / CUDA 12.6, ptxas -O3 "
                             "wedges at 100%% CPU on heavy floating-base kernels. -O2 typically "
                             "completes in a few minutes. Not needed on Blackwell (sm_120). "
                             "Default: nvcc default (-O3 to ptxas).")
    parser.add_argument("--split-compile", type=int, default=None,
                        help="Pass `--split-compile=N` to nvcc (12.x). Parallelizes cicc "
                             "optimization passes (0 = all CPU cores). ~2× faster compile.\n"
                             "*** DEFEATS ANTI-LICM at all N>=2 *** — single-call and batch "
                             "compute-only timings collapse to ~0us / launch overhead. Use "
                             "ONLY for non-timing dev iteration, never for measurement runs.")
    parser.add_argument("--ofast-compile", choices=["min", "mid", "max"], default=None,
                        help="Pass `-Ofc=<level>` to nvcc (12.x). Fast-compile mode for "
                             "device code. Trades device-code runtime perf for compile "
                             "time — opt-in dev knob, NOT for perf measurement runs.")
    parser.add_argument("--single-call-iters", type=int, default=None,
                        help="Override SINGLE_CALL_ITERS_GLOBAL for GRiD/Pinocchio "
                             "(default 10000). Inner-kernel rep count for single-call timings.")
    parser.add_argument("--batch-iters", type=int, default=None,
                        help="Override TEST_ITERS_GLOBAL for GRiD/Pinocchio (default 100) "
                             "and BENCH_TEST_ITERS for MJX/Frax (default 500). Outer batch "
                             "rep count at each N. Bump for more stable medians.")
    parser.add_argument("--pin-num-threads", type=int, default=None,
                        help="Override Pinocchio CPU_THREADS_GLOBAL (default: physical "
                             "cores). Logical/SMT siblings are skipped because every "
                             "thread runs the same JIT'd code; HT hurts.")
    parser.add_argument("--report", type=Path,
                        default=THIS_DIR / "benchmark_multi_version.md",
                        help="Markdown report output path")
    args = parser.parse_args()

    if args.fixed_only:
        args.bases = ["fixed"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Worktree setup (only if pre_glass column is requested)
    if "pre_glass" in args.columns and not args.skip_setup:
        try:
            setup_pre_glass_worktree(args.worktree_path)
        except Exception as e:
            print(f"[fatal] worktree setup failed: {e}", file=sys.stderr)
            sys.exit(1)

    # 1b) Pre-flight dep check per column. Drop columns whose deps are missing
    #     so we don't waste time spawning subprocesses that will ImportError.
    print(f"[{ts()}] === Pre-flight dependency check ===")
    requested = list(args.columns)
    runnable_columns: list[str] = []
    for col in requested:
        ok, reason = _check_column_deps(col, args.worktree_path)
        if ok:
            print(f"  [{col}] ✓ deps OK")
            runnable_columns.append(col)
        else:
            print(f"  [{col}] ✗ SKIP (missing): {reason}")
    if not runnable_columns:
        print(f"[fatal] no columns have their dependencies installed.", file=sys.stderr)
        sys.exit(1)
    args.columns = runnable_columns

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
                if column == "pinocchio":
                    p = run_pinocchio_column(
                        robot, base,
                        output_dir=args.output_dir, no_recompile=args.no_recompile,
                        single_call_iters=args.single_call_iters,
                        batch_iters=args.batch_iters,
                        pin_num_threads=args.pin_num_threads,
                    )
                elif column == "mjx":
                    p = run_mjx_column(
                        robot, base, output_dir=args.output_dir,
                        batch_iters=args.batch_iters,
                    )
                elif column == "frax":
                    p = run_frax_column(
                        robot, base, output_dir=args.output_dir,
                        batch_iters=args.batch_iters,
                    )
                else:
                    p = run_grid_column(
                        column, robot, base,
                        output_dir=args.output_dir, worktree_path=args.worktree_path,
                        no_recompile=args.no_recompile,
                        no_rdc=args.no_rdc, no_licm_barrier=args.no_licm_barrier,
                        single_call_iters=args.single_call_iters,
                        batch_iters=args.batch_iters,
                        ptxas_opt_level=args.ptxas_opt_level,
                        split_compile=args.split_compile,
                        ofast_compile=args.ofast_compile,
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

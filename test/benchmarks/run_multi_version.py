#!/usr/bin/env python3
"""Run a multi-version GRiD benchmark sweep against Pinocchio, MJX, Frax, and BARD.

Columns produced (per robot/base):
  - grid_pre_glass:    GRiD at git ref d2c0d18 (last commit before the GLASS v2 work).
                       Fixed-base only — d2c0d18 harness doesn't support floating-base.
  - grid_glass:        GRiD HEAD (pure-SIMT GLASS).
  - pinocchio:         CPU reference, HEAD harness with --algo parallel fan-out.
  - mjx:               MuJoCo MJX GPU reference (JAX). Requires mujoco-mjx + jax[cuda12].
  - frax:              Frax GPU reference (JAX, https://github.com/danielpmorton/frax).
                       Covers id/fd/crba/minv. Requires frax + jax[cuda12].
  - bard:              BARD reference (PyTorch, https://github.com/YueWang996/bard-pytorch-dynamics).
                       Covers id/fd/crba. Timed on torch CPU + CUDA (bard_cpu/bard_gpu).
                       Requires the `bard` package (pip install --no-deps git+...).

Usage (single robot, fastest):
    python test/benchmarks/run_multi_version.py --robots iiwa14 --bases fixed

Full sweep:
    python test/benchmarks/run_multi_version.py

Worktree for the pre-glass column is created at $GRID_PRE_GLASS_WORKTREE
(default: ../GRiD-A2R-pre-glass/ relative to this repo's root).

Per-(robot, base, algo) JOINT (tier × thread-count) autotune (C.4 + T5):
    --autotune-threads opt-in flag (off by default; default behavior unchanged).
    Forwarded to the GRiD glass column only. For each algo, sweeps a small grid
    of per-block thread counts on EACH per-tier batch binary (shared/lite/minimal,
    reused from the content-keyed binary cache — no new compiles), clipping the
    grid per tier to its launch_bounds cap, and picks the global min-µs/sample
    winner (tier, threads). Picks land in the per-column JSON under
    results[robot][base]["algo_picks"][algo] (schema 2) =
    {"tier_optimal", "threads_optimal", "us_at_optimal",
     "sweep": {tier: {threads: us}}, "sweep_us": {threads: us}}.
    Implementation: the binaries' grid_timing_dimms() honors the
    GRID_AUTOTUNE_THREAD_COUNT env var; the per-tier body is selected by the
    -DGRID_DEFAULT_RESOURCE_TIER macro at compile time.
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

ROBOTS = ("iiwa14", "go2", "g1", "h2_plus", "baxter")
BASES  = ("fixed", "floating")

# Robots that exist ONLY inside GRiD: vendored URDF with no robot_descriptions
# module, no MuJoCo MJCF, and no cuRobo config. The competitor columns all
# resolve their model from robot_descriptions (pinocchio/frax/bard) or an MJCF
# (mjx/mujoco_warp), so they cannot load these robots — we SKIP those columns
# gracefully instead of crashing. H2+ = Unitree H2+ (nv=75 fixed / 81 floating),
# the large-robot SCALING target that retired the redundant h1_2.
GRID_ONLY_ROBOTS = frozenset({"h2_plus", "baxter"})
# Columns that need a non-GRiD model (robot_descriptions URDF or MuJoCo MJCF).
# Skipped for any robot in GRID_ONLY_ROBOTS.
NON_GRID_COLUMNS = frozenset({"pinocchio", "mjx", "mujoco_warp", "frax", "bard"})
# Columns the sweep knows how to run. cuBLASDx (glass_nvidia) was removed in
# v2.0 — the 2026-05-18 sweep + per-host autotune showed it loses to SIMT at
# every GEMM shape GRiD calls (notably 4×4×4 in end_effector_pose_hessian, where
# SIMT wins by 2.6×). The historical data is preserved at the
# `archive/last-cublasdx` git tag; see
# docs/source/user_guide/concepts/cublasdx_removal_design.rst.
COLUMNS = ("pre_glass", "glass", "pinocchio", "mjx", "mujoco_warp", "frax", "bard")
DEFAULT_COLUMNS = ("pre_glass", "glass", "pinocchio", "mjx", "mujoco_warp", "frax", "bard")

# Maps the column identifier to the baseline key used in the merged JSON
# (so generate_report.py / generate_multi_version_report.py can find them).
COLUMN_TO_BASELINE_KEY = {
    "pre_glass":    "grid_pre_glass",
    "glass":        "grid_glass",
    "pinocchio":    "pinocchio",
    "mjx":          "mjx",
    "mujoco_warp":  "mujoco_warp",
    "frax":         "frax",
    "bard":         "bard",
}

EE_FRAMES_GRID = {
    "iiwa14": "iiwa_joint_ee",
    "go2":    "FR_foot_joint",
    "g1":     "right_hand_palm_joint",
    "h2_plus": "right_hand_joint",       # fixed joint at right hand (H2+ large-robot scaling target)
    "baxter":  "left_endpoint",          # fixed joint at left gripper (dual-arm; single-EE convention)
}
# Competitor frames: only robots with a robot_descriptions / MJCF model. h2_plus
# and baxter are GRiD-internal (GRID_ONLY_ROBOTS) so they intentionally have no entry here.
EE_FRAMES_PIN = {
    "iiwa14": "iiwa_link_ee",
    "go2":    "FR_foot",
    "g1":     "right_rubber_hand",
}
# MJX uses MuJoCo body names (same names as Pinocchio link names for these robots).
EE_FRAMES_MJX = EE_FRAMES_PIN
# mujoco_warp loads the same MJCF as MJX, so it uses the same body names.
EE_FRAMES_MUJOCO_WARP = EE_FRAMES_MJX


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
    if column == "mujoco_warp":
        rc = subprocess.run(
            [sys.executable, "-c", "import mujoco, mujoco_warp"],
            capture_output=True,
        ).returncode
        if rc != 0:
            return False, ("mujoco-warp not installed "
                           "(`pip install mujoco mujoco-warp`)")
        return True, ""
    if column == "frax":
        rc = subprocess.run(
            [sys.executable, "-c", "import frax, jax"],
            capture_output=True,
        ).returncode
        if rc != 0:
            return False, "frax+jax not installed (`pip install frax 'jax[cuda12]'`)"
        return True, ""
    if column == "bard":
        rc = subprocess.run(
            [sys.executable, "-c", "import bard, torch"],
            capture_output=True,
        ).returncode
        if rc != 0:
            return False, ("bard+torch not installed (`pip install --no-deps "
                           "git+https://github.com/YueWang996/bard-pytorch-dynamics.git`)")
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
                  ofast_compile: str | None = None,
                  tier: str | None = None,
                  build_dir: Path | None = None,
                  compile_only: bool = False,
                  compile_workers: int | None = None,
                  autotune_threads: bool = False,
                  autotune_thread_grid: str | None = None,
                  autotune_N: int | None = None,
                  single_timing: bool = False) -> list[str]:
    cmd = [
        sys.executable,
        str(harness_repo_root / "test" / "benchmarks" / "baselines" / "grid" / "run.py"),
        "--robot", robot, "--base", base, "--output", str(output),
        "--ee-frame", ee_frame,
    ]
    if build_dir is not None:
        cmd += ["--build-dir", str(build_dir)]
    if compile_only:
        cmd.append("--compile-only")
    if compile_workers is not None:
        cmd += ["--compile-workers", str(compile_workers)]
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
    if tier is not None:
        cmd += ["--tier", tier]
    if autotune_threads:
        cmd.append("--autotune-threads")
        if autotune_thread_grid is not None:
            cmd += ["--autotune-thread-grid", autotune_thread_grid]
        if autotune_N is not None:
            cmd += ["--autotune-N", str(autotune_N)]
    # Single-CALL timing is opt-in/default-off (B8): the -rdc=true single build
    # fatally errors (ptxas regcount) on big floating robots and wastes ~50
    # min/tier. Only pass --single-timing when the orchestrator was asked for it;
    # the autotune matrix uses batch-only timing, unaffected.
    if single_timing:
        cmd.append("--single-timing")
    return cmd


def run_grid_column(column: str, robot: str, base: str, *,
                    output_dir: Path, worktree_path: Path,
                    no_recompile: bool,
                    no_rdc: bool = False, no_licm_barrier: bool = False,
                    single_call_iters: int | None = None,
                    batch_iters: int | None = None,
                    ptxas_opt_level: int | None = None,
                    split_compile: int | None = None,
                    ofast_compile: str | None = None,
                    tier: str | None = None,
                    autotune_threads: bool = False,
                    autotune_thread_grid: str | None = None,
                    autotune_N: int | None = None,
                    single_timing: bool = False) -> Path | None:
    """Run the appropriate GRiD harness for `column`. Returns output JSON path or None."""
    ee_frame = EE_FRAMES_GRID.get(robot, "")
    baseline_key = COLUMN_TO_BASELINE_KEY[column]
    # Tier-tagged output filename so SHARED/LITE/MINIMAL runs don't overwrite
    # each other. SHARED (and its deprecated alias "perf") keeps the legacy name
    # (no _tier_ suffix) so historical filenames stay stable when no tier sweep
    # is requested.
    tier_suffix = "" if (tier is None or tier in ("shared", "perf")) else f"_tier_{tier}"
    output = output_dir / f"{robot}_{base}_{baseline_key}{tier_suffix}.json"

    if column == "pre_glass":
        if base != "fixed":
            print(f"  [{column}] skipping {robot}/{base}: pre-glass harness doesn't support floating-base")
            return None
        if tier is not None and tier not in ("shared", "perf"):
            print(f"  [{column}] skipping {robot}/{base}: pre-glass harness predates tier system")
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
                            split_compile=split_compile, ofast_compile=ofast_compile,
                            tier=tier,
                            autotune_threads=autotune_threads,
                            autotune_thread_grid=autotune_thread_grid,
                            autotune_N=autotune_N,
                            single_timing=single_timing)
    else:
        raise ValueError(f"Unknown grid column: {column}")

    tier_label = f" tier={tier}" if tier else ""
    print(f"[{ts()}] [{column}] {robot} {base}{tier_label} → {output.name}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0 or not output.exists():
        print(f"  [{column}] FAILED for {robot}/{base}{tier_label}", file=sys.stderr)
        return None

    # Rewrite the JSON so the baseline key is column-specific (e.g. "grid_glass")
    # instead of the generic "grid" the inner harness emits. For tier sweeps,
    # also append the tier suffix so PERF/LITE/MINIMAL results live as distinct
    # keys in the merged output.
    keyed = baseline_key + (f"_tier_{tier}" if (tier is not None and tier not in ("shared", "perf")) else "")
    _rename_grid_key(output, keyed)

    # The pre_glass worktree's grid/run.py (frozen at d2c0d18) predates the
    # single/N=16/N=256 batch-summary print added in HEAD. Re-emit it here from
    # the JSON so the stdout looks consistent across all columns.
    if column == "pre_glass":
        _print_batch_summary_from_json(output, keyed)
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
    # Don't let JAX preallocate ~the whole GPU (its default) — keeps a shared box
    # safe and the comparison fair (MJX is JIT-bound, not memory-bound here).
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0 or not output.exists():
        print(f"  [mjx] FAILED for {robot}/{base}", file=sys.stderr)
        return None
    return output


def run_mujoco_warp_column(robot: str, base: str, *,
                          output_dir: Path,
                          batch_iters: int | None = None) -> Path | None:
    ee_frame = EE_FRAMES_MUJOCO_WARP.get(robot, "")
    output = output_dir / f"{robot}_{base}_mujoco_warp.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "test" / "benchmarks" / "baselines" / "mujoco_warp" / "run.py"),
        "--robot", robot, "--base", base, "--output", str(output),
        "--ee-frame", ee_frame,
    ]
    if batch_iters is not None:
        cmd += ["--test-iters", str(batch_iters)]
    print(f"[{ts()}] [mujoco_warp] {robot} {base} → {output.name}")
    # mujoco_warp can pull in JAX/XLA transitively; same no-preallocate guard so it
    # doesn't grab the whole GPU on a shared box.
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)
    if result.returncode != 0 or not output.exists():
        print(f"  [mujoco_warp] FAILED for {robot}/{base}", file=sys.stderr)
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


def run_bard_column(robot: str, base: str, *,
                    output_dir: Path,
                    batch_iters: int | None = None) -> Path | None:
    output = output_dir / f"{robot}_{base}_bard.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "test" / "benchmarks" / "baselines" / "bard" / "run.py"),
        "--robot", robot, "--base", base, "--output", str(output),
    ]
    if batch_iters is not None:
        cmd += ["--test-iters", str(batch_iters)]
    print(f"[{ts()}] [bard] {robot} {base} → {output.name}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0 or not output.exists():
        print(f"  [bard] FAILED for {robot}/{base}", file=sys.stderr)
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
# Parallel build phase
# ---------------------------------------------------------------------------
def _auto_build_jobs() -> int:
    """Default BUILD-phase parallelism from cores + free RAM. Compiles are
    CPU-bound and ~6GB RSS each (the SO kernels dominate), so cap on whichever
    of cores/RAM is tighter. Timing is unaffected (measure phase is serial)."""
    cores = os.cpu_count() or 4
    free_gb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    free_gb = int(line.split()[1]) / (1024 * 1024)
                    break
    except OSError:
        pass
    by_cores = max(1, cores // 3)            # each case drives ~2-3 cicc/ptxas
    by_ram = max(1, int(free_gb // 6)) if free_gb else by_cores
    return max(1, min(by_cores, by_ram, 8))


def _build_grid_binaries(grid_columns, robots, bases, tiers, *, build_jobs,
                         worktree_path, output_dir, skip_set, no_rdc,
                         no_licm_barrier, single_call_iters, batch_iters,
                         ptxas_opt_level, split_compile, ofast_compile,
                         single_timing=False) -> None:
    """Parallel BUILD phase: compile + cache every GRiD (column,robot,base,tier)
    binary across `build_jobs` workers, WITHOUT timing. The serial measure phase
    re-runs each with --no-recompile (instant content-keyed cache hit), so timing
    stays isolated on the GPU. Each task gets its own --build-dir so working files
    don't collide; the binary cache is shared + content-keyed, so the keys (and
    thus the cache hits) match the measure phase as long as compile flags match."""
    tasks = []
    for column in grid_columns:
        for robot in robots:
            for base in bases:
                if f"{robot}_{base}" in skip_set:
                    continue
                for tier in tiers:
                    # mirror run_grid_column's pre_glass limitations
                    if column == "pre_glass" and (base != "fixed" or (tier and tier not in ("shared", "perf"))):
                        continue
                    tasks.append((column, robot, base, tier))
    if not tasks:
        return
    cores = os.cpu_count() or 4
    # GRID_COMPILE_WORKERS overrides the per-cell TU-compile parallelism. The big
    # monolithic single_main + batch_main TUs for large robots (g1/h2_plus SO) use
    # ~24-36 GB of cicc EACH; compiling them concurrently exhausts a 62 GB box.
    # Set GRID_COMPILE_WORKERS=1 (with --build-jobs 1) to serialize to ONE big
    # compile at a time (~36 GB peak = safe). See [[feedback_build_ram_so_compiles]].
    _cw_env = os.environ.get("GRID_COMPILE_WORKERS", "")
    per_task_workers = int(_cw_env) if _cw_env.strip() else max(1, cores // max(1, build_jobs))
    print(f"[{ts()}] === BUILD phase: {len(tasks)} GRiD binaries, {build_jobs} parallel "
          f"(compile-workers={per_task_workers} each); timing stays serial ===")

    def _one(task: tuple[str, str, str, str]) -> bool:
        column, robot, base, tier = task
        ee_frame = EE_FRAMES_GRID.get(robot, "")
        harness_root = worktree_path if column == "pre_glass" else REPO_ROOT
        bdir = output_dir / "_build" / f"{column}_{robot}_{base}_{tier}"
        bdir.mkdir(parents=True, exist_ok=True)
        scratch = bdir / "scratch.json"
        if column == "pre_glass":
            cmd = _grid_run_cmd(harness_root, robot, base, scratch, ee_frame,
                                no_recompile=False, build_dir=bdir, compile_only=True,
                                compile_workers=per_task_workers)
        else:
            effective_ptxas = ptxas_opt_level if base == "floating" else None
            cmd = _grid_run_cmd(harness_root, robot, base, scratch, ee_frame,
                                no_recompile=False, no_rdc=no_rdc,
                                no_licm_barrier=no_licm_barrier,
                                single_call_iters=single_call_iters, batch_iters=batch_iters,
                                ptxas_opt_level=effective_ptxas,
                                split_compile=split_compile, ofast_compile=ofast_compile,
                                tier=tier, build_dir=bdir, compile_only=True,
                                compile_workers=per_task_workers,
                                single_timing=single_timing)
        t0 = datetime.now()
        r = subprocess.run(cmd, capture_output=True, text=True)
        dur = (datetime.now() - t0).total_seconds()
        tag = f"{column} {robot}/{base} tier={tier}"
        if r.returncode == 0:
            print(f"  [build ✓] {tag} ({dur:.0f}s)")
            return True
        print(f"  [build ✗] {tag} ({dur:.0f}s)\n{r.stdout[-800:]}\n{r.stderr[-800:]}",
              file=sys.stderr)
        return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=build_jobs) as ex:
        results = list(ex.map(_one, tasks))
    ok = sum(1 for x in results if x)
    print(f"[{ts()}] === BUILD phase done: {ok}/{len(tasks)} compiled "
          f"(failures will recompile or surface in the measure phase) ===")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-version GRiD benchmark sweep (pre-glass + glass vs pinocchio + mjx + frax + bard)",
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
                        help="Subset of columns to run (default: all six)")
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
    parser.add_argument("--tiers", nargs="+", default=["shared"],
                        choices=["shared", "perf", "lite", "minimal"],
                        help="Resource tiers to sweep for the GRiD columns. Default: "
                             "['shared'] (legacy single-tier behavior; 'perf' is a "
                             "deprecated alias for 'shared'). Pass "
                             "'--tiers shared lite minimal' for full Phase 4 sweep. WITHOUT "
                             "--autotune-threads each GRiD column gets one full run per tier, "
                             "tagged grid_glass / grid_glass_tier_lite / grid_glass_tier_minimal "
                             "in the merged JSON. WITH --autotune-threads the measure phase "
                             "COLLAPSES to a SINGLE glass run per (robot, base): that one run "
                             "already builds every tier binary and autotunes across all tiers, "
                             "and the report sources its per-tier columns from the run's "
                             "'algo_picks' sweep (so --tiers only affects the non-autotune "
                             "path here). Non-GRiD columns (pinocchio/mjx/frax) are tier-agnostic "
                             "and run only once.")
    parser.add_argument("--report", type=Path,
                        default=THIS_DIR / "benchmark_multi_version.md",
                        help="Markdown report output path")
    parser.add_argument("--build-jobs", type=int, default=None,
                        help="Parallelism for the GRiD compile (BUILD) phase. The compile is "
                             "CPU-bound (nvcc/cicc/ptxas), so it fans across cores; the timing "
                             "(MEASURE) phase always stays SERIAL on the isolated GPU, so this "
                             "never affects the numbers. Default: auto (from cores + free RAM, "
                             "~6GB/compile). Pass 1 for the legacy fully-serial behavior.")
    parser.add_argument("--build-only", action="store_true",
                        help="PRE-BUILD only: run the parallel BUILD phase (warm the content-"
                             "addressed binary cache for every requested robot/base/tier of the "
                             "GRiD columns) and STOP before the serial MEASURE phase. No timing, "
                             "no report. Lets you compile all sweep versions ahead of time (e.g. "
                             "while the box is busy) so the later real sweep is pure cache-hit "
                             "timing on a quiet GPU. Cache keys match the timed run exactly because "
                             "it is the SAME build call. Forces the build phase even at "
                             "--build-jobs 1.")
    parser.add_argument("--autotune-threads", action="store_true",
                        help="Forward --autotune-threads to the GRiD glass column. For each "
                             "(robot, base, algo) tuple, do the JOINT (tier × thread-count) "
                             "autotune: sweep a per-tier-cap-clipped thread grid (default: "
                             "96,128,192,256,320,384 + one-level refinement) on each per-tier "
                             "batch binary (shared/lite/minimal, reused from the binary cache) and "
                             "pick the global min-µs/sample winner (tier, threads). The picks land "
                             "in the per-column JSON under 'algo_picks[algo]' (schema 2) = "
                             "{'tier_optimal','threads_optimal','us_at_optimal',"
                             "'sweep':{tier:{threads:us}},'sweep_us':{threads:us}}. Opt-in; default OFF.")
    parser.add_argument("--autotune-thread-grid", type=str, default=None,
                        help="Override the autotune thread grid (comma-separated). "
                             "Default: '96,128,192,256,320,384'.")
    parser.add_argument("--autotune-N", type=int, default=None,
                        help="Batch size to autotune on (default: 256). The winner is the thread "
                             "count that minimizes batch_<N>_compute_only µs/sample.")
    parser.add_argument("--single-timing", action="store_true",
                        default=(os.environ.get("GRID_BENCH_SINGLE_TIMING", "0") not in ("0", "", "false", "False")),
                        help="OPT-IN (default OFF): forward --single-timing to the GRiD glass "
                             "column so it ALSO builds + runs the single-CALL latency binary. Off "
                             "by default because the -rdc=true single build fatally errors (ptxas "
                             "regcount) on big floating robots (g1/h2_plus) and wastes ~50 min/tier; "
                             "batch throughput timing (the autotune matrix) is unaffected and "
                             "always runs. Also settable via env GRID_BENCH_SINGLE_TIMING=1.")
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

    skip_set = {s.strip() for s in args.skip}

    # 1c) BUILD phase (parallel, CPU-bound). Compiles are nvcc/cicc/ptxas work, NOT
    #     GPU work, so they fan across cores; the per-case binary cache is content-
    #     keyed, so the serial measure phase below finds the same binaries. This
    #     collapses the dominant cost (e.g. iiwa14-fixed alone is a ~460s compile)
    #     from sequential to parallel without touching the (still-serial) timing.
    grid_columns = [c for c in args.columns if c in ("glass", "pre_glass")]
    build_jobs = args.build_jobs if args.build_jobs is not None else _auto_build_jobs()
    measure_no_recompile = False
    # --build-only forces the build phase even at build_jobs==1 (the usual >1 guard
    # is a perf optimization for the timed path; here the build IS the deliverable).
    if grid_columns and (build_jobs > 1 or args.build_only):
        _build_grid_binaries(
            grid_columns, args.robots, args.bases, args.tiers,
            build_jobs=build_jobs, worktree_path=args.worktree_path,
            output_dir=args.output_dir, skip_set=skip_set,
            no_rdc=args.no_rdc, no_licm_barrier=args.no_licm_barrier,
            single_call_iters=args.single_call_iters, batch_iters=args.batch_iters,
            ptxas_opt_level=args.ptxas_opt_level, split_compile=args.split_compile,
            ofast_compile=args.ofast_compile,
            single_timing=args.single_timing,
        )
        # Measure phase pulls from the warm cache; never compile during timing.
        measure_no_recompile = True

    # --build-only: the binary cache is now warm for every requested cell. Stop here;
    # the real (timed) sweep re-runs the same command WITHOUT --build-only and hits
    # the cache for the build, leaving only serial GPU timing.
    if args.build_only:
        print(f"[{ts()}] === --build-only: binary cache warmed for "
              f"robots={args.robots} bases={args.bases} tiers={args.tiers}; "
              f"skipping MEASURE + report. ===")
        return

    # 2) MEASURE phase: run all (column, robot, base) combinations sequentially.
    #    The GPU is the shared serial resource; timing must not contend. GRiD
    #    columns hit the cache built above (measure_no_recompile), so this loop
    #    is pure timing for them.
    produced: list[Path] = []
    skipped: list[tuple[str, str, str]] = []

    for column in args.columns:
        for robot in args.robots:
            for base in args.bases:
                if f"{robot}_{base}" in skip_set:
                    print(f"  [{column}] SKIP {robot}/{base}: excluded via --skip")
                    skipped.append((column, robot, base))
                    continue
                # GRiD-internal-only robots (e.g. H2+) have no robot_descriptions
                # URDF / MJCF / cuRobo model, so the competitor columns can't load
                # them. Skip those columns gracefully (the GRiD columns still run).
                if robot in GRID_ONLY_ROBOTS and column in NON_GRID_COLUMNS:
                    print(f"  [{column}] SKIP {robot}/{base}: GRiD-internal robot "
                          f"(no {column} model)")
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
                elif column == "mujoco_warp":
                    p = run_mujoco_warp_column(
                        robot, base, output_dir=args.output_dir,
                        batch_iters=args.batch_iters,
                    )
                elif column == "frax":
                    p = run_frax_column(
                        robot, base, output_dir=args.output_dir,
                        batch_iters=args.batch_iters,
                    )
                elif column == "bard":
                    p = run_bard_column(
                        robot, base, output_dir=args.output_dir,
                        batch_iters=args.batch_iters,
                    )
                else:
                    # GRiD columns. Two regimes:
                    #  * autotune ON (glass): a SINGLE run already builds every tier
                    #    binary and autotunes ACROSS all tiers, emitting per-algo
                    #    `algo_picks` whose `sweep` carries every tier's best-thread
                    #    times + the global `tier_optimal` winner. So we measure ONCE
                    #    per (robot, base) — running per-tier here would re-do that
                    #    full all-tier autotune N times for N tiers (pure redundancy;
                    #    the outputs differ only by measurement noise). The report
                    #    sources its per-tier columns from `algo_picks[...]['sweep']`.
                    #  * autotune OFF: per-tier runs are legitimate (each tier's
                    #    binary cache is keyed by -DGRID_DEFAULT_RESOURCE_TIER and
                    #    produces a distinct grid_glass[_tier_*] timing block).
                    do_autotune = (args.autotune_threads and column == "glass")
                    measure_tiers = [None] if do_autotune else args.tiers
                    for tier in measure_tiers:
                        p = run_grid_column(
                            column, robot, base,
                            output_dir=args.output_dir, worktree_path=args.worktree_path,
                            no_recompile=args.no_recompile or measure_no_recompile,
                            no_rdc=args.no_rdc, no_licm_barrier=args.no_licm_barrier,
                            single_call_iters=args.single_call_iters,
                            batch_iters=args.batch_iters,
                            ptxas_opt_level=args.ptxas_opt_level,
                            split_compile=args.split_compile,
                            ofast_compile=args.ofast_compile,
                            tier=tier,
                            autotune_threads=do_autotune,
                            autotune_thread_grid=args.autotune_thread_grid,
                            autotune_N=args.autotune_N,
                            single_timing=args.single_timing,
                        )
                        if p is not None:
                            produced.append(p)
                        else:
                            skipped.append((f"{column}/{tier}", robot, base))
                    continue
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

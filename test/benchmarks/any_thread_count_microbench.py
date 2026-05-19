#!/usr/bin/env python3
"""Microbench: GRiD kernel timing across per-block thread counts.

After the v2.0 cuBLASDx removal, every kernel emission dropped
``__launch_bounds__(SUGGESTED_THREADS)``. The SIMT GLASS helpers use
block-stride loops, so any block size is correct. This microbench
characterizes the perf trajectory across the sweep:

    {64, 128, 256, SUGGESTED_THREADS, 512}

for a fixed batch on iiwa14, go2, g1, h1_2 (all fixed-base) using the
``grid_rbd`` Python wrapper. Output is a small markdown table for each
robot showing single-call median µs at each block size.

Run with:
    .venv/bin/python test/benchmarks/any_thread_count_microbench.py

Or for one robot:
    .venv/bin/python test/benchmarks/any_thread_count_microbench.py --robot iiwa14

Saved here (not /tmp) so future-self can re-run after codegen changes
and compare the trajectory.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


URDFS = {
    "iiwa14": (
        Path.home()
        / ".cache/robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
    ),
    "go2": (
        Path.home()
        / ".cache/robot_descriptions/unitree_ros/robots/go2_description/urdf/go2_description.urdf"
    ),
    "g1": (
        Path.home()
        / ".cache/robot_descriptions/unitree_ros/robots/g1_description/g1_29dof.urdf"
    ),
    "h1_2": (
        Path.home()
        / ".cache/robot_descriptions/unitree_ros/robots/h1_2_description/h1_2.urdf"
    ),
}


def _check_preconditions():
    if shutil.which("nvcc") is None:
        print("FATAL: nvcc not on PATH; install CUDA Toolkit.", file=sys.stderr)
        sys.exit(1)
    try:
        import grid_rbd  # noqa: F401
    except ImportError:
        print("FATAL: grid_rbd not importable; run `pip install -e python/`.", file=sys.stderr)
        sys.exit(1)


def _bench_rnea(handle, q, qd, threads: int, iters: int = 500) -> float:
    """Return median per-call µs for handle.rnea at the given block size."""
    handle.set_threads_per_block(threads)
    # Warmup (touches caches; first call after set may also re-init).
    for _ in range(10):
        handle.rnea(q, qd)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        handle.rnea(q, qd)
        times.append((time.perf_counter() - t0) * 1e6)  # µs
    return median(times)


def bench_robot(robot: str, batch: int = 16, iters: int = 500) -> dict:
    """Bench one robot across the standard block-size sweep."""
    import grid_rbd

    urdf = URDFS.get(robot)
    if urdf is None or not urdf.exists():
        raise FileNotFoundError(
            f"URDF for {robot!r} not found at {urdf}. "
            f"Try `pip install robot-descriptions[<robot>_description]` or "
            f"point the script at a custom path."
        )

    handle = grid_rbd.register_robot(
        name=f"{robot}_any_thread_count_microbench",
        urdf_path=str(urdf),
        floating_base=False,
        max_batch_size=max(batch, 32),
    )
    sug = handle.suggested_threads

    # Sweep: 64, 128, 256, SUGGESTED, 512 (deduplicated, sorted).
    block_sizes = sorted({64, 128, 256, sug, 512})

    rng = np.random.default_rng(0)
    q = rng.standard_normal((batch, handle.num_joints)).astype(np.float32)
    qd = rng.standard_normal((batch, handle.num_joints)).astype(np.float32)

    print(f"\n=== {robot}_fixed (NJ={handle.num_joints}, SUGGESTED_THREADS={sug}) ===")
    print(f"  batch={batch}, iters={iters}")
    results = {"robot": robot, "num_joints": handle.num_joints,
               "suggested_threads": sug, "batch": batch, "iters": iters,
               "per_block_us": {}}
    for n in block_sizes:
        us = _bench_rnea(handle, q, qd, n, iters=iters)
        label = f"{n} (default)" if n == sug else str(n)
        print(f"    threads={label:>16s}: {us:8.3f} µs")
        results["per_block_us"][n] = us
    # Reset to default so subsequent runs see a clean state.
    handle.set_threads_per_block(sug)
    return results


def _format_markdown(results: list[dict]) -> str:
    """Render results as a single markdown table per robot."""
    lines = ["# GRiD any-thread-count microbench — RNEA median µs", ""]
    for r in results:
        lines.append(f"## {r['robot']}_fixed (NJ={r['num_joints']}, "
                     f"SUGGESTED_THREADS={r['suggested_threads']})")
        lines.append("")
        lines.append("| threads/block | median µs |")
        lines.append("|---|---:|")
        for n in sorted(r["per_block_us"]):
            us = r["per_block_us"][n]
            label = f"{n} *(default)*" if n == r["suggested_threads"] else str(n)
            lines.append(f"| {label} | {us:.3f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", choices=list(URDFS), default=None,
                   help="Run one robot (default: all available).")
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size for the rnea timings (default 16).")
    p.add_argument("--iters", type=int, default=500,
                   help="Number of timed iterations per cell (default 500).")
    p.add_argument("--output", type=Path, default=None,
                   help="Write a markdown summary to this path.")
    args = p.parse_args()

    _check_preconditions()

    robots = [args.robot] if args.robot else list(URDFS)
    results = []
    for robot in robots:
        try:
            results.append(bench_robot(robot, batch=args.batch, iters=args.iters))
        except FileNotFoundError as e:
            print(f"\n[skip] {robot}: {e}", file=sys.stderr)

    if args.output:
        args.output.write_text(_format_markdown(results))
        print(f"\nSummary written: {args.output}")


if __name__ == "__main__":
    main()

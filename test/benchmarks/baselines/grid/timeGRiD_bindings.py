#!/usr/bin/env python3
"""Layer-3 (wrapper-inclusive) timing for GRiD through the REAL grid_rbd python FFI.

This is the "extreme" / most-conservative GRiD number: what an adopter actually pays
calling GRiD from python — pack + host->device + kernel + device-sync + dispatch through
the `grid_rbd` jax surface, end-to-end. It complements the two C++ layers reported by the
grid harness (layer 1 = compute-only GPU-resident; layer 2 = C++ with-mem H2D/D2H).

SYMMETRIC JIT RULE (BENCHMARK_METHODOLOGY.md): the grid_rbd jax surface wraps the FFI call
in jax.jit, so timing it cold would charge layer-3 the first-call trace/compile (the exact
mjx artifact we fixed for competitors). We therefore give GRiD the SAME treatment as every
JIT competitor: warm the EXACT jitted closure (block_until_ready, same shapes/dtypes incl.
the numpy->device path) in warmup, then time pure execution. This is symmetric fairness.

Emits the competitor json schema so analyze_competitive.py / plot_benchmarks.py consume it:
    results[robot][base]["grid_bindings"][algo][batch_<N>_{compute_only,with_mem}_us]
                                                = {mean, median, min, max, std}
- compute_only : inputs already device-resident (jnp), jitted call + block_until_ready.
- with_mem     : inputs numpy (regenerated per iter, outside the timer), jitted call
                 converts/transfers them (H2D) + block_until_ready — symmetric with how
                 mjx/frax time their with_mem (sync, not explicit D2H), so the comparison
                 is apples-to-apples with the competitor adapters.

Usage:
    python test/benchmarks/baselines/grid/timeGRiD_bindings.py \
        --robot iiwa14 --base fixed --output results/.../iiwa14_fixed_grid_bindings.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the harness's robot/EE resolution so layer-3 uses the same URDF + EE frame
# as the C++ layers (import-safe: grid/run.py guards main() under __main__).
from test.benchmarks.baselines.grid.run import (  # noqa: E402
    get_urdf_path, DEFAULT_EE_FRAMES,
)

BATCH_SIZES = [16, 32, 64, 128, 256, 1024]
TEST_ITERS = int(os.environ.get("BENCH_TEST_ITERS", "500"))
N_WARMUP_PASSES = 5

# Core algos shared with the competitor set, plus GRiD's gradient/SO headline kernels.
# (qpos, qvel, qacc/qfrc) arg arity is encoded so we build the right inputs per algo.
ALGOS = [
    ("inverse_dynamics",          ("q", "v", "a")),
    ("forward_dynamics",          ("q", "v", "u")),
    ("inverse_dynamics_gradient", ("q", "v", "a")),
    ("forward_dynamics_gradient", ("q", "v", "u")),
    ("crba",                      ("q",)),
    ("minv",                      ("q",)),
    ("aba",                       ("q", "v", "u")),
    ("idsva_so",                  ("q", "v", "a")),
    ("fdsva_so",                  ("q", "v", "u")),
    ("end_effector_pose",         ("q",)),
    ("end_effector_pose_gradient", ("q",)),
]


def _stats(times_us: np.ndarray) -> dict:
    return {
        "mean":   float(np.mean(times_us)),
        "median": float(np.median(times_us)),
        "min":    float(np.min(times_us)),
        "max":    float(np.max(times_us)),
        "std":    float(np.std(times_us)),
    }


def _make_np(arity, n, nq, nv, rng):
    """Random batched numpy inputs (host) for the requested arg arity."""
    widths = {"q": nq, "v": nv, "a": nv, "u": nv}
    return tuple(rng.standard_normal((n, widths[k])).astype(np.float32) for k in arity)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--base", required=True, choices=["fixed", "floating"])
    ap.add_argument("--urdf", default=None, help="override URDF path")
    ap.add_argument("--ee-frame", default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--iters", type=int, default=TEST_ITERS)
    ap.add_argument("--max-batch", type=int, default=1024)
    ap.add_argument("--algos", nargs="+", default=None,
                    help="subset of algo names (default: all)")
    args = ap.parse_args()

    floating = args.base == "floating"
    urdf = args.urdf or get_urdf_path(args.robot)
    ee_frame = args.ee_frame or DEFAULT_EE_FRAMES.get(args.robot)

    try:
        import jax
        import jax.numpy as jnp
        import grid_rbd
        import grid_rbd.jax as grid_jax
    except ImportError as e:
        print(f"grid_rbd/jax import failed: {e}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build / load the binding (.so) with N=1024 batch capacity baked in.
    # precompile is idempotent: a cached tier is an instant no-op (no nvcc).
    # ------------------------------------------------------------------
    name = f"bench_{args.robot}_{args.base}"
    t0 = time.perf_counter()
    grid_rbd.precompile(
        name, urdf,
        floating_base=floating,
        ee_joint_names=[ee_frame] if ee_frame else None,
        max_batch_size=args.max_batch,
        backends=("jax",),
    )
    handle = grid_jax.get_robot(name)   # JaxRobotHandle (device-resident, jittable)
    print(f"  [grid_bindings] {name} ready ({time.perf_counter()-t0:.1f}s), "
          f"nq={handle.num_joints} nv={handle.num_vel} max_batch={handle.max_batch}")

    nq, nv = handle.num_joints, handle.num_vel
    want = set(args.algos) if args.algos else None
    rng = np.random.default_rng(0)

    metadata = {
        "robot": args.robot, "base": args.base, "urdf": str(urdf),
        "ee_frame": ee_frame, "layer": "grid_bindings (wrapper-inclusive jax FFI e2e)",
        "jax_version": jax.__version__,
        "jax_backend": (jax.default_backend() if hasattr(jax, "default_backend") else "?"),
        "iters": args.iters, "nq": nq, "nv": nv,
        "note": "SYMMETRIC JIT: jitted FFI closure warmed before timing, like competitors.",
    }

    algo_results: dict = {}
    for algo, arity in ALGOS:
        if want is not None and algo not in want:
            continue
        method = getattr(handle, algo, None)
        if method is None:
            print(f"  [grid_bindings] skip {algo}: not on jax surface", file=sys.stderr)
            continue
        fn = jax.jit(method)
        per_n: dict = {}
        ok = True
        for n in BATCH_SIZES:
            if n > args.max_batch:
                continue
            try:
                # ---- compute_only: device-resident args ----
                np_args = _make_np(arity, n, nq, nv, rng)
                dev_args = tuple(jnp.asarray(a) for a in np_args)
                jax.block_until_ready(fn(*dev_args))               # JIT compile (discard)
                for _ in range(N_WARMUP_PASSES):
                    jax.block_until_ready(fn(*dev_args))
                co = np.empty(args.iters)
                for i in range(args.iters):
                    t = time.perf_counter()
                    jax.block_until_ready(fn(*dev_args))
                    co[i] = (time.perf_counter() - t) * 1e6

                # ---- with_mem: numpy args (H2D inside the timed call) ----
                jax.block_until_ready(fn(*np_args))                # warm the numpy-input path
                for _ in range(N_WARMUP_PASSES):
                    jax.block_until_ready(fn(*np_args))
                wm = np.empty(args.iters)
                for i in range(args.iters):
                    fresh = _make_np(arity, n, nq, nv, rng)        # outside the timer
                    t = time.perf_counter()
                    jax.block_until_ready(fn(*fresh))
                    wm[i] = (time.perf_counter() - t) * 1e6

                per_n[f"batch_{n}_compute_only_us"] = _stats(co)
                per_n[f"batch_{n}_with_mem_us"] = _stats(wm)
                print(f"  [grid_bindings] {algo:28s} N={n:5d}  "
                      f"compute={np.mean(co):9.2f}us  with_mem={np.mean(wm):9.2f}us")
            except Exception as e:  # one bad cell shouldn't sink the whole algo
                print(f"  [grid_bindings] {algo} N={n} FAILED: {e}", file=sys.stderr)
                ok = False
                break
        if per_n:
            algo_results[algo] = per_n
        if not ok:
            continue

    out = {
        "metadata": metadata,
        "results": {args.robot: {args.base: {"grid_bindings": algo_results}}},
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2)
        print(f"  [grid_bindings] wrote {args.output}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

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

# XLA's default allocator PREALLOCATES 75% of GPU memory at first backend init
# (test/conftest.py carries the same guard for pytest runs; standalone drivers
# import THIS module before jax, so this is their earliest hook). Without it,
# big-humanoid kernels fail AT LAUNCH with cudaErrorMemoryAllocation — the
# runtime can't allocate the local-memory pool once XLA holds 24+ GiB
# (h1_2 forward_dynamics "launch failed", autotune leg 2026-09-05).
# setdefault so an explicit caller choice still wins.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the harness's robot/EE resolution so layer-3 uses the same URDF + EE frame
# as the C++ layers (import-safe: grid/run.py guards main() under __main__).
from test.benchmarks.baselines.grid.run import get_urdf_path  # noqa: E402

BATCH_SIZES = [16, 32, 64, 128, 256, 1024]
TEST_ITERS = int(os.environ.get("BENCH_TEST_ITERS", "500"))
N_WARMUP_PASSES = 5

# Core algos shared with the competitor set, plus GRiD's gradient/SO headline
# kernels. The bench ROSTER (which algos to time) is policy and stays literal;
# each algo's (qpos, qvel, qacc/qfrc) arg arity derives from its AbiSpec row
# (H6 — the same table that generates the wrapper bodies), so the bench can
# never build inputs the C-ABI doesn't take. Golden-pinned by
# test/test_bench_algo_arity.py.
_BENCH_ROSTER = [
    "inverse_dynamics", "forward_dynamics", "inverse_dynamics_gradient",
    "forward_dynamics_gradient", "crba", "minv", "aba", "idsva_so",
    "fdsva_so", "end_effector_pose", "end_effector_pose_gradient",
]
_ARITY_OF = {"q": "q", "qd": "v", "u": "u", "qdd": "a", "qdd_opt": "a"}


def _spec_arity(key):
    """Leading DOF-buffer params of the algo's C-ABI, as bench arity codes."""
    from grid_codegen.abi_specs import ABI_SPECS
    out = []
    for name, ctype in ABI_SPECS[key].inputs:
        if ctype != "const T*" or name == "f_ext":
            break
        out.append(_ARITY_OF[name])
    return tuple(out)


ALGOS = [(k, _spec_arity(k)) for k in _BENCH_ROSTER]


def _stats(times_us: np.ndarray) -> dict:
    return {
        "mean":   float(np.mean(times_us)),
        "median": float(np.median(times_us)),
        "min":    float(np.min(times_us)),
        "max":    float(np.max(times_us)),
        "std":    float(np.std(times_us)),
    }


def _make_np(arity, n, nq, nv, rng, floating=False):
    """Random batched numpy inputs (host) for the requested arg arity.

    Floating base: GRiD packs every per-timestep input (q, qd, qacc/qfrc) into nq-wide
    slots (stride = 3*NUM_JOINTS, NUM_JOINTS=nq), so qd/qacc/qfrc are nq-wide too — NOT nv.
    Fixed base: nq==nv so this is identical."""
    vw = nq if floating else nv
    widths = {"q": nq, "v": vw, "a": vw, "u": vw}
    return tuple(rng.standard_normal((n, widths[k])).astype(np.float32) for k in arity)


# FFI-regime thread candidates. The binding bakes the C++-autotuned per-algo threads
# (register-clamped, often <=352), but the jax/torch FFI launch path is ~30x SLOWER in that
# low-thread regime and only reaches GRiD's true batched throughput at higher thread counts
# (robot/algo-dependent: iiwa14 fd needs >=512). GRiD kernels are thread-count-INVARIANT in
# result (single-block design; verified max|err|=0 across 128..1024), so sweeping threads only
# trades speed, never correctness. We autotune threads in the FFI regime — symmetric with how
# the C++ layer is autotuned — so layer-3 reports GRiD-through-the-wrapper at ITS best, fairly.
THREAD_CANDIDATES = [128, 256, 384, 512, 640, 768, 896, 1024]


def _pick_threads(handle, fn, dev_args, candidates):
    """Return (best_threads, best_us) for this jitted op at the probe batch by a quick sweep.
    Skips candidates that raise (e.g. too-high thread/smem for a heavy algo)."""
    import jax
    best_t, best_us = None, float("inf")
    for n in candidates:
        try:
            handle.set_threads_per_block(n)
            for _ in range(3):
                jax.block_until_ready(fn(*dev_args))      # warm at this thread count
            t = time.perf_counter()
            for _ in range(30):
                jax.block_until_ready(fn(*dev_args))
            us = (time.perf_counter() - t) / 30 * 1e6
            if us < best_us:
                best_t, best_us = n, us
        except Exception:
            continue
    return best_t, best_us


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--base", required=True, choices=["fixed", "floating"])
    ap.add_argument("--urdf", default=None, help="override URDF path")
    ap.add_argument("--ee-frame", default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--iters", type=int, default=TEST_ITERS)
    ap.add_argument("--max-batch", type=int, default=1024)
    ap.add_argument("--ffi-thread-sweep", action="store_true",
                    help="re-autotune threads for the FFI launch path (default off: use the "
                         "baked per-algo autotuned config = 'call it the way it was tuned')")
    ap.add_argument("--algos", nargs="+", default=None,
                    help="subset of algo names (default: all)")
    args = ap.parse_args()

    floating = args.base == "floating"
    urdf = args.urdf or get_urdf_path(args.robot)
    # Default to the DEFAULT (unnamed) EE target: passing a NAMED ee_joint_names trips
    # gen_end_effector_pose_gradient_inner ("named fixed_target not yet supported by the
    # shared-chain geometric-Jacobian rewrite"). EE-frame choice doesn't affect the dynamics
    # algos we benchmark, so only pass a named frame if the user explicitly asks for one.
    ee_frame = args.ee_frame

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

    nq, nv = handle.num_joints, handle.num_vel   # num_joints IS nq (nv+1 for floating)
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
        # DEFAULT (P1.1b): the binding bakes each algo's AUTOTUNED per-algo {tier, threads}
        # (launch_cfg<ALGO>) via the stem-resolved launch_configs, so the default launch IS the
        # autotuned config — "the binding calls each kernel the way it was autotuned". We report
        # THAT. (The jax FFI launch path has a slightly different thread optimum than the C++ path
        # it was tuned on; a binding-path FFI re-autotune could squeeze further — opt-in with
        # --ffi-thread-sweep, tracked as the FFI-autotune backlog.)
        # handle defaults to threads_per_block=-1 (the baked per-algo autotuned config); no reset needed.
        if args.ffi_thread_sweep:
            probe_n = min(256, args.max_batch)
            probe_args = tuple(jnp.asarray(a) for a in _make_np(arity, probe_n, nq, nv, rng, floating))
            jax.block_until_ready(fn(*probe_args))                 # JIT compile once
            best_threads, best_us = _pick_threads(handle, fn, probe_args, THREAD_CANDIDATES)
            if best_threads is not None:
                handle.set_threads_per_block(best_threads)
                print(f"  [grid_bindings] {algo:28s} FFI-swept threads={best_threads} "
                      f"(probe N={probe_n}: {best_us:.1f}us)")
            per_n["ffi_threads"] = best_threads
        else:
            per_n["ffi_threads"] = handle.threads_per_block        # -1 = baked autotuned default
        for n in BATCH_SIZES:
            if n > args.max_batch:
                continue
            try:
                # ---- compute_only: device-resident args ----
                np_args = _make_np(arity, n, nq, nv, rng, floating)
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
                    fresh = _make_np(arity, n, nq, nv, rng, floating)        # outside the timer
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

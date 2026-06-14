"""FFI-regime autotune for the grid_rbd python bindings.

WHY THIS EXISTS (read first): the optimal CUDA block size for a GRiD kernel is
NOT a property of the kernel alone — it depends on the LAUNCH PATH and the
MEASUREMENT (use case). The C++ run.py autotune (-> launch_configs `bases`)
measures the host launch path and bakes a throughput-optimal thread count. The
python bindings launch the SAME kernel through the jax/torch FFI custom-call
path, and for the "batch-to-land" use case (fire ONE batched launch of N, wait
for all N to land — the control/optimization use case) that path has a DIFFERENT
optimum. Measured on iiwa14/fixed/fd/N=256 (lite tier, identical kernel):

    threads   C++ host(us)   jax FFI(us)
      128        15.4           49.7      <- host-optimal; FFI inherits this -> SLOW
      768        (n/a)          30.2      <- FFI-optimal (= lite launch_bounds max)

So a binding that inherits the host's 128 runs fd ~1.6x slow. This tool sweeps
the FFI launch path for the batch-to-land metric and writes the winners into
launch_configs/<robot>/<gpu>.json under `ffi_bases` (the host `bases` block is
left untouched). The binding's codegen reads `ffi_bases` by default (profile=
"ffi"; see GRiDCodeGenerator.load_launch_config + bindings/grid_rbd/_compile.py),
falling back per-algo to host `bases` for any algo this tool didn't tune.

V1 re-tunes THREADS only (runtime-settable, no rebuild), keeping each algo's
host-autotuned TIER. The mechanism behind the host-vs-FFI thread flip is not yet
explained (sharp FFI cliff at the launch_bounds ceiling) — Nsight profiling is
backlogged. Run this AFTER a C++ autotune (so `bases`/tiers exist) and on the
SAME GPU you deploy on.

Usage:
    python test/benchmarks/autotune_ffi.py --robot iiwa14 --base fixed
    python test/benchmarks/autotune_ffi.py --robot go2 --base both --n 256
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the layer-3 harness machinery so we tune EXACTLY what it times.
from test.benchmarks.baselines.grid.timeGRiD_bindings import (  # noqa: E402
    ALGOS, THREAD_CANDIDATES, _make_np,
)
from test.benchmarks.baselines.grid.run import get_urdf_path  # noqa: E402
from GRiDCodeGenerator.GRiDCodeGenerator import (  # noqa: E402
    LAUNCH_CONFIG_ALGO_TO_SYMBOL, LAUNCH_CONFIG_DEFAULT_GPU, _launch_configs_dir,
)

# symbol (handle method name) -> short launch_configs json key (fd, id, ...).
SYMBOL_TO_KEY = {sym: key for key, sym in LAUNCH_CONFIG_ALGO_TO_SYMBOL.items()}


def _median_batch_to_land_us(fn, dev_args, iters):
    """Median wall-clock us for ONE batched launch of N to fully land (sync)."""
    import jax
    t = np.empty(iters)
    for i in range(iters):
        s = time.perf_counter()
        jax.block_until_ready(fn(*dev_args))
        t[i] = (time.perf_counter() - s) * 1e6
    return float(np.median(t))


def _sweep_algo(handle, fn, dev_args, candidates, iters, warmup):
    """Return {threads: median_us} over candidates + the winning thread count."""
    import jax
    curve, best_t, best_us = {}, None, float("inf")
    for thr in candidates:
        try:
            handle.set_threads_per_block(thr)
            for _ in range(warmup):
                jax.block_until_ready(fn(*dev_args))   # warm/compile at this size
            us = _median_batch_to_land_us(fn, dev_args, iters)
        except Exception as e:                          # too-high smem/threads for a heavy algo
            print(f"      threads={thr:5d}  SKIP ({type(e).__name__})")
            continue
        curve[thr] = us
        flag = ""
        if us < best_us:
            best_t, best_us, flag = thr, us, "  <- best"
        print(f"      threads={thr:5d}  {us:8.2f} us{flag}")
    return curve, best_t, best_us


def _host_tier_for(doc, base, key):
    """The host-autotuned tier for this algo (FFI v1 keeps it; threads-only retune)."""
    entry = ((doc.get("bases") or {}).get(base) or {}).get(key) or {}
    return entry.get("tier", "shared")


def autotune_base(robot, base, n, iters, warmup, want_algos):
    import grid_rbd
    import grid_rbd.jax as grid_jax

    floating = base == "floating"
    urdf = get_urdf_path(robot)
    name = f"autotune_ffi_{robot}_{base}"
    grid_rbd.precompile(name, urdf, floating_base=floating,
                        max_batch_size=max(256, n), backends=("jax",))
    handle = grid_jax.get_robot(name)
    nq, nv = handle.num_joints, handle.num_vel
    rng = np.random.default_rng(0)
    print(f"\n=== {robot}/{base}  nq={nq} nv={nv}  N={n}  (FFI batch-to-land) ===")

    import jax
    import jax.numpy as jnp
    picks = {}
    for algo, arity in ALGOS:
        if want_algos and algo not in want_algos:
            continue
        key = SYMBOL_TO_KEY.get(algo)
        if key is None:
            continue
        method = getattr(handle, algo, None)
        if method is None:
            print(f"    {algo:28s} skip (not on jax surface)")
            continue
        fn = jax.jit(method)
        dev = tuple(jnp.asarray(a) for a in _make_np(arity, n, nq, nv, rng, floating))
        jax.block_until_ready(fn(*dev))                 # one-time JIT
        print(f"    {algo}:")
        curve, best_t, best_us = _sweep_algo(handle, fn, dev, THREAD_CANDIDATES, iters, warmup)
        # (no reset needed: the next algo's sweep sets its own thread count; the
        #  python handle guards n>=1 so we can't pass 0 to reset to the baked -1.)
        if best_t is None:
            continue
        # Tie-break: the fast regime is a flat top plateau (e.g. fd 768/896/1024 all
        # ~equal); argmin jitters run-to-run. Pick the SMALLEST thread count within
        # TOL of the best — deterministic + leaner on registers/occupancy, same speed.
        tol = 1.05
        thr = min(t for t, us in curve.items() if us <= best_us * tol)
        picks[key] = {"threads": int(thr), "us": round(curve[thr], 2),
                      "argmin_threads": int(best_t), "argmin_us": round(best_us, 2)}
        if thr != best_t:
            print(f"      -> pick {thr} (within {int((tol-1)*100)}% of best {best_t})")
    return picks


def write_ffi_config(robot, gpu, base_picks, n):
    """Merge {base: {key: {threads, us}}} into launch_configs/<robot>/<gpu>.json
    under `ffi_bases`, keeping each algo's host tier and leaving `bases` intact."""
    path = Path(_launch_configs_dir()) / robot / f"{gpu}.json"
    doc = json.loads(path.read_text()) if path.exists() else {}
    ffi = doc.setdefault("ffi_bases", {})
    for base, picks in base_picks.items():
        blk = ffi.setdefault(base, {})
        for key, pk in picks.items():
            blk[key] = {"tier": _host_tier_for(doc, base, key), "threads": pk["threads"]}
    meta = doc.setdefault("ffi_meta", {})
    meta["metric"] = "batch_to_land_median_us"
    meta["autotune_N"] = n
    meta["note"] = ("FFI/jax launch-path optimum (threads-only retune of the host tier); "
                    "regenerated by test/benchmarks/autotune_ffi.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"\n  wrote ffi_bases -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--base", default="fixed", choices=["fixed", "floating", "both"])
    ap.add_argument("--n", type=int, default=256, help="batch size to tune for (default 256)")
    ap.add_argument("--iters", type=int, default=100, help="timed iters per thread count")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--gpu", default=LAUNCH_CONFIG_DEFAULT_GPU)
    ap.add_argument("--algos", nargs="+", default=None, help="subset of algo symbols")
    ap.add_argument("--dry-run", action="store_true", help="sweep + print, do not write")
    args = ap.parse_args()

    bases = ["fixed", "floating"] if args.base == "both" else [args.base]
    want = set(args.algos) if args.algos else None
    base_picks = {}
    for base in bases:
        base_picks[base] = autotune_base(args.robot, base, args.n, args.iters, args.warmup, want)

    print("\n=== FFI picks (batch-to-land) ===")
    for base, picks in base_picks.items():
        for key, pk in sorted(picks.items()):
            print(f"  {base:8s} {key:24s} threads={pk['threads']:5d}  {pk['us']:8.2f} us")
    if args.dry_run:
        print("\n  --dry-run: not writing ffi_bases")
        return
    write_ffi_config(args.robot, args.gpu, base_picks, args.n)


if __name__ == "__main__":
    main()

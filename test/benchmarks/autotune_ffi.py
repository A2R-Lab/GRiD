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
config/launch_configs/<robot>/<gpu>.json under `ffi_bases` (the host `bases` block is
left untouched). The binding's codegen reads `ffi_bases` by default (profile=
"ffi"; see GRiDCodeGenerator.load_launch_config + bindings/grid_rbd/_compile.py),
falling back per-algo to host `bases` for any algo this tool didn't tune.

V1 re-tunes THREADS only (runtime-settable, no rebuild), keeping each algo's
host-autotuned TIER. The mechanism behind the host-vs-FFI thread flip is not yet
explained (sharp FFI cliff at the launch_bounds ceiling) — Nsight profiling is
backlogged. Run this AFTER a C++ autotune (so `bases`/tiers exist) and on the
SAME GPU you deploy on.

--surface picks WHICH python launch path to sweep (each is a different use case
with its own optimum, written to its own launch_configs block):
    jax   -> `ffi_bases`    (jax FFI custom-call, batch-to-land via block_until_ready)
    torch -> `torch_bases`  (torch op path, batch-to-land via torch.cuda.synchronize)
    numpy -> `pybind_bases` (synchronous C-ABI host wrapper, H2D+launch+D2H included
                             — that copy round-trip IS the numpy use case)
The bindings consume these via codegen profile overlays ("ffi"/"torch"/"pybind" ->
<profile>_bases, GRiDCodeGenerator.load_launch_config) and the E6 runtime overlay
(RobotHandle.apply_profile_overlay).

Usage:
    python test/benchmarks/autotune_ffi.py --robot iiwa14 --base fixed
    python test/benchmarks/autotune_ffi.py --robot go2 --base both --n 256
    python test/benchmarks/autotune_ffi.py --robot iiwa14 --surface torch
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
from grid_codegen.GRiDCodeGenerator import (  # noqa: E402
    LAUNCH_CONFIG_DEFAULT_GPU, _launch_configs_dir,
)
from grid_codegen.algo_registry import build_launch_config_algo_to_symbol  # noqa: E402

# symbol (handle method name) -> short launch_configs json key (fd, id, ...).
# Built from the descriptor table (single source of truth for {json key -> symbol}).
SYMBOL_TO_KEY = {sym: key for key, sym in build_launch_config_algo_to_symbol().items()}

# surface -> codegen profile name (= launch_configs block prefix: <profile>_bases).
SURFACE_PROFILE = {"jax": "ffi", "torch": "torch", "numpy": "pybind"}


def _surface_adapter(surface):
    """(to_dev, wrap, call) for one launch surface. `call(fn, dev_args)` runs ONE
    batched invocation to completion (sync included) — the batch-to-land unit."""
    if surface == "jax":
        import jax
        import jax.numpy as jnp

        def to_dev(args):
            return tuple(jnp.asarray(a) for a in args)

        def wrap(method):
            return jax.jit(method)

        def call(fn, dev_args):
            jax.block_until_ready(fn(*dev_args))
    elif surface == "torch":
        import torch

        def to_dev(args):
            return tuple(torch.as_tensor(a, device="cuda") for a in args)

        def wrap(method):
            return method

        def call(fn, dev_args):
            fn(*dev_args)
            torch.cuda.synchronize()
    else:  # numpy: the C-ABI host wrapper is synchronous (returns host arrays)
        def to_dev(args):
            return args

        def wrap(method):
            return method

        def call(fn, dev_args):
            fn(*dev_args)
    return to_dev, wrap, call


def _median_batch_to_land_us(call, fn, dev_args, iters):
    """Median wall-clock us for ONE batched launch of N to fully land (sync)."""
    t = np.empty(iters)
    for i in range(iters):
        s = time.perf_counter()
        call(fn, dev_args)
        t[i] = (time.perf_counter() - s) * 1e6
    return float(np.median(t))


def _kernel_real_ceiling(handle, algo_key):
    """The kernel's REAL compiled __launch_bounds__ ceiling (= the baked tier's
    maxThreadsPerBlock) via the grid_rbd_kernel_max_threads introspection C-ABI.

    This is the E1 tier-contract fix: instead of GUESSING the tier (and defaulting
    to "shared"/512), query cudaFuncGetAttributes(...).maxThreadsPerBlock for the
    EXACT kernel + tier the binding bakes for this algo. Returns the int ceiling, or
    None if the ABI is absent (old .so), the algo is unknown, or it reports <1 — in
    which case the caller falls back to inferring the tier from the swept ceiling."""
    fn = getattr(handle, "kernel_max_threads", None)
    if fn is None:
        return None
    try:
        cap = int(fn(algo_key))
    except Exception:
        return None
    return cap if cap >= 1 else None


def _sweep_algo(handle, fn, dev_args, candidates, iters, warmup, call, real_ceiling=None):
    """Sweep block sizes for the batch-to-land metric.

    Returns (curve, best_actual, best_us) where `curve` is keyed on the ACTUAL
    launched thread count (= min(requested, real_ceiling)) — NOT the requested
    count. De-duplicating on the actual count closes the Case-A reporting hole: if
    640/768/1024 all clamp to the same real ceiling they collapse to ONE curve point
    and cannot masquerade as a faster higher-thread regime (design_autotune_matrix.md
    §1.2). When `real_ceiling` is known we also SKIP requests above it (they only
    clamp-collapse onto the ceiling point), and we keep the SMALLEST request that maps
    to each actual count (deterministic)."""
    curve, best_t, best_us = {}, None, float("inf")
    seen_actual = set()
    for thr in candidates:
        actual = thr if real_ceiling is None else min(thr, real_ceiling)
        if actual in seen_actual:
            # already timed this actual launch (a higher request clamps to it) — skip
            # the redundant point so the curve can't double-count a clamped regime.
            continue
        try:
            # Request the CLAMPED count, not the raw candidate: the jax FFI path
            # silently clamps an over-ceiling request, but the torch/numpy launch
            # paths hard-error on it — requesting `actual` times the at-ceiling
            # regime on every surface instead of SKIPping it (and is identical to
            # the old behavior on jax, where thr clamped to actual anyway).
            handle.set_threads_per_block(actual)
            for _ in range(warmup):
                call(fn, dev_args)                     # warm/compile at this size
            us = _median_batch_to_land_us(call, fn, dev_args, iters)
        except Exception as e:                          # too-high smem/threads for a heavy algo
            print(f"      threads={thr:5d}  SKIP ({type(e).__name__})")
            continue
        seen_actual.add(actual)
        curve[actual] = us           # KEY ON ACTUAL LAUNCHED COUNT, not the request
        clamp = f" (->{actual})" if actual != thr else ""
        flag = ""
        if us < best_us:
            best_t, best_us, flag = actual, us, "  <- best"
        print(f"      threads={thr:5d}{clamp}  {us:8.2f} us{flag}")
    return curve, best_t, best_us


def _tier_for_ceiling(ceiling, max_perf):
    """Invert grid::tier_max_threads<TIER>() : a real maxThreadsPerBlock -> tier label.

      tier_max_threads<TIER_SHARED>()  = max_perf             (capped at 512)
      tier_max_threads<TIER_LITE>()    = min(2*max_perf, 768)
      tier_max_threads<TIER_MINIMAL>() = 1024

    Returns the LOWEST tier whose launch_bounds equals `ceiling` (conservative on
    ties: a kernel at exactly `max_perf` is "shared", never a higher tier). Returns
    None if no tier matches (the introspected ceiling is inconsistent with this
    robot's max_perf — surfaced as a hard error by the caller, never silently
    recorded)."""
    if not max_perf:
        return None
    shared = max_perf
    lite = min(max_perf * 2, 768)
    minimal = 1024
    if ceiling == shared:
        return "shared"
    if ceiling == lite:
        return "lite"
    if ceiling == minimal:
        return "minimal"
    return None


def _host_tier_for(doc, base, key):
    """LAST-RESORT tier guess: the host-autotuned tier for this algo, defaulting to
    "shared". Used ONLY when both the kernel introspection (Option 1) and the
    swept-ceiling inference (Option 3) are unavailable — i.e. a pre-E1 .so with no
    host `bases` entry. The E1 fix exists precisely because this guess is wrong for
    un-tuned big robots (it clamps the real regime to shared/512)."""
    entry = ((doc.get("bases") or {}).get(base) or {}).get(key) or {}
    return entry.get("tier", "shared")


def autotune_base(robot, base, n, iters, warmup, want_algos, build_algos=None,
                  surface="jax"):
    import grid_rbd

    floating = base == "floating"
    urdf = get_urdf_path(robot)
    name = f"autotune_ffi_{robot}_{base}"
    # RAM-safe subset build: big robots (g1 nv=35, h2_plus nv=75/81) blow a SINGLE
    # nvcc process to 24-36GB when the SO kernels (idsva_so/fdsva_so) are in the
    # .so. Pass --build-algos to compile ONLY a chosen set (deps pulled in by the
    # codegen profile) so the build fits — e.g. tune the core/non-SO algos first
    # and defer SO. When set we also only sweep what's built. None = full build.
    precompile_kw = {}
    if build_algos:
        # algorithm_list (passed via a single tier override) re-keys the cache to its
        # own subset .so (won't collide with the full build) and limits the nvcc memory
        # footprint. Methods outside the subset simply won't exist on the handle and are
        # skipped by the sweep loop below.
        precompile_kw["tiers"] = [{"algorithm_list": list(build_algos)}]
    grid_rbd.precompile(name, urdf, floating_base=floating,
                        max_batch_size=max(256, n), backends=(surface,), **precompile_kw)
    if surface == "jax":
        import grid_rbd.jax as grid_jax
        handle = grid_jax.get_robot(name)
    elif surface == "torch":
        import grid_rbd.torch as grid_torch
        handle = grid_torch.get_robot(name)
    else:
        handle = grid_rbd.get_robot(name)
    to_dev, wrap, call = _surface_adapter(surface)
    nq, nv = handle.num_joints, handle.num_vel
    max_perf = handle.max_perf_level_threads   # MPLT, for the tier inverse map (E1)
    rng = np.random.default_rng(0)
    print(f"\n=== {robot}/{base}  nq={nq} nv={nv}  N={n}  ({surface} batch-to-land) ===")

    picks = {}
    for algo, arity in ALGOS:
        if want_algos and algo not in want_algos:
            continue
        key = SYMBOL_TO_KEY.get(algo)
        if key is None:
            continue
        method = getattr(handle, algo, None)
        if method is None:
            print(f"    {algo:28s} skip (not on {surface} surface)")
            continue
        fn = wrap(method)
        dev = to_dev(_make_np(arity, n, nq, nv, rng, floating))
        # The one-time JIT launches BEFORE this algo's sweep sets any thread
        # count, so it inherits the PREVIOUS algo's last-swept count — which
        # can exceed THIS kernel's compiled launch_bounds ceiling ("launch
        # failed" at the JIT probe; first hit: baxter forward_dynamics, N=16
        # leg, 2026-08-25). 32 threads is legal for every kernel.
        handle.set_threads_per_block(32)
        try:
            call(fn, dev)                               # one-time JIT / first launch
        except (RuntimeError, AttributeError) as e:
            # --build-algos subset .so: the jax method exists on the handle but its
            # compiled symbol (grid_rbd_jax_<algo>) was excluded from this build, so
            # the FIRST call raises "not built into this robot .so" / undefined symbol.
            # Skip it (this is the documented subset-build behavior — e.g. defer the
            # SO kernels idsva_so/fdsva_so that OOM a single nvcc on big robots).
            if "not built" in str(e) or "undefined symbol" in str(e):
                print(f"    {algo:28s} skip (not in subset .so)")
                continue
            raise
        # E1 tier-contract fix: read the kernel's REAL compiled launch_bounds ceiling
        # (cudaFuncGetAttributes maxThreadsPerBlock) BEFORE the sweep, so we (a) never
        # request a count that only clamp-collapses and (b) record the tier the kernel
        # was actually compiled at — not a "shared" guess. None => ABI absent (old .so).
        real_ceiling = _kernel_real_ceiling(handle, key)
        if real_ceiling is not None:
            print(f"    {algo}:  (kernel max_threads={real_ceiling})")
        else:
            print(f"    {algo}:  (kernel max_threads UNKNOWN -> infer from sweep)")
        curve, best_t, best_us = _sweep_algo(handle, fn, dev, THREAD_CANDIDATES,
                                             iters, warmup, call, real_ceiling=real_ceiling)
        # (no reset needed: the next algo's sweep sets its own thread count; the
        #  python handle guards n>=1 so we can't pass 0 to reset to the baked -1.)
        if best_t is None:
            continue
        # Tie-break: the fast regime is a flat top plateau (e.g. fd 768/896/1024 all
        # ~equal); argmin jitters run-to-run. Pick the SMALLEST thread count within
        # TOL of the best — deterministic + leaner on registers/occupancy, same speed.
        # NOTE: curve is keyed on ACTUAL launched counts (clamped), so a clamped fast
        # "regime" has already collapsed onto its real ceiling and can't win here.
        tol = 1.05
        thr = min(t for t, us in curve.items() if us <= best_us * tol)
        # Derive the recorded tier from the kernel's real ceiling (Option 1); fall back
        # to the swept ceiling (the max count that demonstrably launched) (Option 3).
        ceiling_for_tier = real_ceiling if real_ceiling is not None else max(curve)
        tier_source = "kernel" if real_ceiling is not None else "swept_ceiling"
        recorded_tier = _tier_for_ceiling(ceiling_for_tier, max_perf)
        picks[key] = {"threads": int(thr), "us": round(curve[thr], 2),
                      "argmin_threads": int(best_t), "argmin_us": round(best_us, 2),
                      "kernel_max_threads": int(ceiling_for_tier),
                      "tier": recorded_tier, "tier_source": tier_source}
        if recorded_tier is None:
            # Introspected ceiling doesn't match any tier for this robot's max_perf —
            # a real inconsistency (e.g. a kernel not actually launch_bounds-limited).
            # Surface it loudly rather than silently recording a bad {tier, threads}.
            print(f"      !! WARNING: ceiling {ceiling_for_tier} maps to NO tier "
                  f"(max_perf={max_perf}); tier left null, FIX before baking")
        if thr != best_t:
            print(f"      -> pick {thr} (within {int((tol-1)*100)}% of best {best_t})")
    return picks, max_perf


def _tier_max_threads(tier, max_perf):
    """Mirror C++ grid::tier_max_threads<TIER>(): the tier's __launch_bounds__ ceiling.
    A pick above this is silently clamped at launch (jax path) and would crash the raw
    numpy/pybind launch — the codegen bakes the clamp; we clamp the JSON to match."""
    t = str(tier).lower()
    if t == "minimal":
        return 1024
    if t == "lite":
        return min(max_perf * 2, 768)
    return max_perf  # shared


def write_ffi_config(robot, gpu, base_picks, n, base_maxperf, surface="jax"):
    """Merge {base: {key: pick}} into config/launch_configs/<robot>/<gpu>.json under
    the surface's `<profile>_bases` block (ffi_bases / torch_bases / pybind_bases),
    leaving `bases` and every other surface's block intact.

    E1 tier contract: the recorded tier is the KERNEL's real compiled tier (from
    cudaFuncGetAttributes, `pick["tier"]`/`tier_source`), NOT a "shared" guess. The
    swept `threads` is <= that tier's real launch_bounds BY CONSTRUCTION (the sweep
    only kept counts that launched on the real kernel and de-dups on the clamped
    count), so the old blanket clamp is now an ASSERTION — it can no longer silently
    throw a valid pick away. Falls back to the host-tier guess only for a pre-E1 .so
    that lacked the introspection ABI (tier == None)."""
    profile = SURFACE_PROFILE[surface]
    path = Path(_launch_configs_dir()) / robot / f"{gpu}.json"
    doc = json.loads(path.read_text()) if path.exists() else {}
    ffi = doc.setdefault(f"{profile}_bases", {})
    tier_sources = set()
    for base, picks in base_picks.items():
        blk = ffi.setdefault(base, {})
        mp = base_maxperf.get(base)
        for key, pk in picks.items():
            tier = pk.get("tier")
            src = pk.get("tier_source")
            if tier is None:
                # No kernel/swept ceiling available (old .so) -> last-resort host guess.
                tier = _host_tier_for(doc, base, key)
                src = "host_guess"
            tier_sources.add(src)
            thr = int(pk["threads"])
            if mp:
                ceiling = _tier_max_threads(tier, mp)
                if thr > ceiling:
                    # Should be impossible under the contract; if it fires the tier and
                    # the pick disagree -> do NOT silently clamp (that re-introduces the
                    # bug). Surface it and clamp defensively so the JSON stays launchable.
                    print(f"  !! {base}/{key}: pick {thr} > tier {tier} ceiling {ceiling} "
                          f"(src={src}) — tier/threads INCONSISTENT, clamping + flag")
                    thr = ceiling
            blk[key] = {"tier": tier, "threads": thr}
    meta = doc.setdefault(f"{profile}_meta", {})
    meta["metric"] = "batch_to_land_median_us"
    meta["autotune_N"] = n
    meta["tier_source"] = sorted(tier_sources)   # how each tier was derived (audit)
    meta["note"] = (f"{surface} launch-path optimum; tier read from the kernel's REAL "
                    "compiled launch_bounds (cudaFuncGetAttributes, E1 tier contract). "
                    "Regenerated by test/benchmarks/autotune_ffi.py --surface " + surface)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"\n  wrote {profile}_bases -> {path}  (tier_source={sorted(tier_sources)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--base", default="fixed", choices=["fixed", "floating", "both"])
    ap.add_argument("--n", type=int, default=256, help="batch size to tune for (default 256)")
    ap.add_argument("--iters", type=int, default=100, help="timed iters per thread count")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--gpu", default=LAUNCH_CONFIG_DEFAULT_GPU)
    ap.add_argument("--surface", default="jax", choices=sorted(SURFACE_PROFILE),
                    help="launch path to sweep: jax -> ffi_bases, torch -> torch_bases, "
                         "numpy -> pybind_bases (default jax)")
    ap.add_argument("--algos", nargs="+", default=None, help="subset of algo symbols to SWEEP")
    ap.add_argument("--build-algos", nargs="+", default=None,
                    help="RAM-safe subset to BUILD into the .so (deps auto-pulled). Big robots "
                         "(g1/h2_plus) OOM a single nvcc when SO kernels are in the .so — build the "
                         "core/non-SO set first and defer idsva_so/fdsva_so. Default = full build.")
    ap.add_argument("--dry-run", action="store_true", help="sweep + print, do not write")
    args = ap.parse_args()

    bases = ["fixed", "floating"] if args.base == "both" else [args.base]
    want = set(args.algos) if args.algos else None
    base_picks, base_maxperf = {}, {}
    for base in bases:
        base_picks[base], base_maxperf[base] = autotune_base(
            args.robot, base, args.n, args.iters, args.warmup, want,
            build_algos=args.build_algos, surface=args.surface)

    print(f"\n=== {args.surface} picks (batch-to-land) ===")
    for base, picks in base_picks.items():
        for key, pk in sorted(picks.items()):
            tier = pk.get("tier") or "?"
            src = pk.get("tier_source") or "?"
            print(f"  {base:8s} {key:24s} threads={pk['threads']:5d}  {pk['us']:8.2f} us"
                  f"  tier={tier:8s} (max={pk.get('kernel_max_threads','?')}, src={src})")
    if args.dry_run:
        print(f"\n  --dry-run: not writing {SURFACE_PROFILE[args.surface]}_bases")
        return
    write_ffi_config(args.robot, args.gpu, base_picks, args.n, base_maxperf,
                     surface=args.surface)


if __name__ == "__main__":
    main()

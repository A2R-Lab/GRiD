#!/usr/bin/env python
"""Per-TIER autotune sweep -> the E5 `matrix` block (Phase B, item B1).

WHAT THIS ADDS over autotune_ffi.py: that tool sweeps THREADS at whatever tier
each algo happens to be baked at (the host/ffi-autotuned pick). This driver
obtains, per tier in {shared, lite, minimal}, a probe .so whose EVERY swept algo
is compiled AT THAT TIER, re-runs autotune_ffi's batch-to-land thread sweep on
it, and records the per-(base, algo, tier) results in the E5 matrix schema
(docs/open-tasks/E5_E6_autotune_matrix_design.md):

    matrix.<base>.<algo>.<tier> = {max_threads, min_smem,
                                   suggested_threads.<profile>, ...}

HOW THE PER-TIER PROBE .so IS BUILT (no changes to any existing file):
The wrapper instantiates every kernel/host launcher at
`grid::launch_cfg<GRID_ALGO_*>::TIER` (bindings/grid_rbd/wrapper_template.cu),
and that table is baked by gen_add_launch_config_helpers from
`config/launch_configs/<resolved-robot>/<DEFAULT_GPU>.json`
(grid_codegen/_launch_config.py:load_launch_config). The robot is resolved from
the URDF FILENAME STEM, exact-directory-match first
(bindings/grid_rbd/_compile.py:_resolve_launch_config_robot). So per tier we:
  1. write a TEMPORARY probe config dir
     config/launch_configs/tierprobe_<robot>_<tier>/<DEFAULT_GPU>.json whose
     host `bases` force EVERY launch-config algo to that tier (both bases);
  2. copy the robot's URDF to <tmp>/tierprobe_<robot>_<tier>.urdf so the stem
     resolves to that probe dir (verified by a pre-flight load_launch_config
     call before any build — never assumed);
  3. run autotune_ffi.autotune_base on the probe name (its module-level
     get_urdf_path is redirected to the probe URDF for the probe name only).
warm_robot folds the RESOLVED launch config into the .so cache key
(bindings/grid_rbd/__init__.py, code_options["launch_config"]), so the three
tier builds land in three distinct content-keyed cache entries and are REUSED
on rerun (3 builds/robot/base total). Probe config dirs are deleted afterwards
(--keep-probes to inspect them); recreating them identically cache-hits.

INTEGRITY GUARDS:
  * pre-flight: the probe stem must resolve to the probe dir and
    load_launch_config must return every algo at the forced tier, else abort
    before building anything.
  * post-build: each algo's introspected kernel ceiling
    (handle.kernel_max_threads, the E1 ABI) must be <= the forced tier's
    launch-bounds cap; a HIGHER ceiling proves the probe was not applied ->
    that tier's cells are reported and withheld from any write, exit 1.
  * a tier/base whose probe .so fails to build (e.g. smem misfit on a big
    robot) is recorded as build_failed and reported; the run continues with
    the remaining tiers. Per-algo launch failures are likewise recorded
    per-cell, never fatal.

`min_smem` is joined READ-ONLY from results/kernel_limits_<host>.json
(collect_kernel_limits.py) when present; cells without it carry null.

DEFAULT IS DRY-RUN-LIKE: the matrix is printed; nothing is written unless
--write is passed, which merges the `matrix` + `matrix_meta` blocks into
config/launch_configs/<robot>/<gpu>.json preserving the existing
2-space-indent sort_keys format (existing suggested_threads profiles in a
cell are preserved and merged, never clobbered).

Usage (tonight's pilot):
    .venv/bin/python test/benchmarks/autotune_tier_matrix.py --robot iiwa14 --base fixed --smoke
    .venv/bin/python test/benchmarks/autotune_tier_matrix.py --robot iiwa14 --base fixed --write
    .venv/bin/python test/benchmarks/autotune_tier_matrix.py --robot go2 --base floating --write

NOTE: the codegen bake always reads the LAUNCH_CONFIG_DEFAULT_GPU-named probe
file (load_launch_config's default gpu), so probes work regardless of --gpu;
--gpu only selects which <robot>/<gpu>.json the results are written into.
"""
import argparse
import contextlib
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Reuse autotune_ffi's measurement machinery wholesale (module import so the
# probe URDF redirect below can patch ITS get_urdf_path binding, not ours).
from test.benchmarks import autotune_ffi as af  # noqa: E402
from grid_codegen.launch_config import (  # noqa: E402
    LAUNCH_CONFIG_DEFAULT_GPU,
    LAUNCH_CONFIG_TIER_SYMBOL,
    _launch_configs_dir,
    load_launch_config,
)
from grid_codegen.algo_registry import build_launch_config_algo_to_symbol  # noqa: E402

TIERS = ("shared", "lite", "minimal")
PROBE_PREFIX = "tierprobe_"
# The surface this driver sweeps; its picks land under suggested_threads[PROFILE].
SURFACE = "jax"
PROFILE = af.SURFACE_PROFILE[SURFACE]          # "ffi"
# --smoke: 2 cheap algos, tiny iters — pilot-validates probe + schema fast.
SMOKE_ALGOS = ("inverse_dynamics", "forward_dynamics")

RESULTS_DIR = REPO_ROOT / "test" / "benchmarks" / "results"


def _host() -> str:
    return platform.node().replace(" ", "_")


# ---------------------------------------------------------------------------
# Probe config + URDF (the per-tier .so mechanism)
# ---------------------------------------------------------------------------
def _probe_config_doc(tier):
    """A launch-config doc forcing EVERY launch-config algo to `tier`, both
    bases. Baked threads=32 is a deterministic don't-care: the sweep sets the
    runtime thread count per candidate via set_threads_per_block; only the
    TIER (-> __launch_bounds__ + smem constants) is baked from here. Keeping
    the doc deterministic keeps the probe .so cache key stable across reruns."""
    keys = sorted(build_launch_config_algo_to_symbol().keys())
    block = {k: {"tier": tier, "threads": 32} for k in keys}
    return {
        "bases": {"fixed": dict(block), "floating": dict(block)},
        "probe_meta": {
            "note": "TEMPORARY per-tier probe config written by "
                    "test/benchmarks/autotune_tier_matrix.py — safe to delete.",
            "tier": tier,
        },
    }


@contextlib.contextmanager
def _probe_environment(robot, tier, keep=False):
    """Create the probe config dir + probe-stem URDF copy; yield (stem, urdf).

    The probe file MUST be named <LAUNCH_CONFIG_DEFAULT_GPU>.json: the codegen
    bake calls load_launch_config without a gpu argument
    (grid_codegen/_launch_config.py, gen_add_launch_config_helpers)."""
    stem = f"{PROBE_PREFIX}{robot}_{tier}"
    probe_dir = Path(_launch_configs_dir()) / stem
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / f"{LAUNCH_CONFIG_DEFAULT_GPU}.json").write_text(
        json.dumps(_probe_config_doc(tier), indent=2, sort_keys=True) + "\n")
    tmpdir = tempfile.mkdtemp(prefix="grid_tierprobe_")
    try:
        urdf_src = af.get_urdf_path(robot)   # real resolver (patch not active here)
        probe_urdf = Path(tmpdir) / f"{stem}.urdf"
        shutil.copyfile(urdf_src, probe_urdf)
        yield stem, probe_urdf
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if not keep:
            shutil.rmtree(probe_dir, ignore_errors=True)


@contextlib.contextmanager
def _patched_urdf(stem, probe_urdf):
    """Redirect autotune_ffi's get_urdf_path for the probe name ONLY (smallest
    wrapper over its internals; autotune_ffi.py itself is untouched)."""
    orig = af.get_urdf_path

    def patched(robot):
        return str(probe_urdf) if robot == stem else orig(robot)

    af.get_urdf_path = patched
    try:
        yield
    finally:
        af.get_urdf_path = orig


def _preflight(stem, probe_urdf, tier, bases):
    """Prove (never assume) the loader resolves the probe: the stem must map to
    the probe dir, and load_launch_config must return EVERY algo at the forced
    tier for each requested base. Raises RuntimeError before any build."""
    from grid_rbd._compile import _resolve_launch_config_robot
    resolved = _resolve_launch_config_robot(str(probe_urdf))
    if resolved != stem:
        raise RuntimeError(
            f"probe pre-flight: URDF stem {stem!r} resolved to launch-config "
            f"robot {resolved!r} — the probe dir was not picked up")
    want_sym = LAUNCH_CONFIG_TIER_SYMBOL[tier]
    n_algos = len(build_launch_config_algo_to_symbol())
    for base in bases:
        cfg = load_launch_config(stem, base == "floating", profile=PROFILE)
        if len(cfg) != n_algos:
            raise RuntimeError(
                f"probe pre-flight: load_launch_config({stem}, {base}) returned "
                f"{len(cfg)}/{n_algos} algos — probe config incomplete")
        bad = {s: c for s, c in cfg.items() if c["tier"] != want_sym}
        if bad:
            raise RuntimeError(
                f"probe pre-flight: {len(bad)} algos not at {want_sym} for "
                f"{base}: {sorted(bad)[:5]} ...")


# ---------------------------------------------------------------------------
# min_smem join (read-only, from collect_kernel_limits.py output)
# ---------------------------------------------------------------------------
def _load_kernel_limits(path):
    if path is None:
        path = RESULTS_DIR / f"kernel_limits_{_host()}.json"
    p = Path(path)
    if not p.exists():
        return None, str(p)
    try:
        return json.loads(p.read_text()).get("limits", {}), str(p)
    except (OSError, ValueError):
        return None, str(p)


def _min_smem(limits, robot, base, algo_key, tier):
    if not limits:
        return None
    cell = (((limits.get(robot) or {}).get(base) or {}).get(algo_key) or {})
    return (cell.get(tier) or {}).get("min_smem")


# ---------------------------------------------------------------------------
# The per-(tier, base) sweep
# ---------------------------------------------------------------------------
def _looks_like_build_failure(err_text):
    """A precompile/nvcc failure repeats identically for every algo — detect it
    on the first hit so we don't re-attempt the same build 11 times."""
    t = err_text.lower()
    return ("nvcc failed" in t) or ("build log" in t) or ("gen_all_code" in t)


def sweep_tier(robot, base, tier, n, iters, warmup, want_algos, build_algos,
               keep_probes, limits):
    """Returns (cells, problems, status). cells: {algo_key: leaf} for good
    sweeps; problems: [{algo, error}]; status: ok|build_failed|probe_not_applied."""
    cells, problems = {}, []
    status = "ok"
    with _probe_environment(robot, tier, keep=keep_probes) as (stem, probe_urdf):
        _preflight(stem, probe_urdf, tier, [base])
        with _patched_urdf(stem, probe_urdf):
            for algo_sym, _arity in af.ALGOS:
                if want_algos and algo_sym not in want_algos:
                    continue
                key = af.SYMBOL_TO_KEY.get(algo_sym)
                if key is None:
                    continue
                try:
                    # One algo per call = per-algo fault isolation (a heavy
                    # algo's launch failure can't eat the rest of the tier).
                    # The probe .so build happens on the FIRST call; every
                    # later call is a content-keyed cache hit.
                    picks, max_perf = af.autotune_base(
                        stem, base, n, iters, warmup, {algo_sym},
                        build_algos=build_algos, surface=SURFACE)
                except Exception as e:  # noqa: BLE001 — report, don't crash (task contract)
                    err = f"{type(e).__name__}: {e}"
                    problems.append({"algo": key, "error": err[:500]})
                    if _looks_like_build_failure(err):
                        status = "build_failed"
                        print(f"  !! {robot}/{base}/{tier}: probe .so build failed — "
                              f"skipping the rest of this tier\n     {err[:200]}")
                        break
                    print(f"  !! {robot}/{base}/{tier}/{key}: sweep failed ({err[:160]})")
                    continue
                pk = picks.get(key)
                if pk is None:
                    problems.append({"algo": key, "error": "no pick (all thread "
                                     "candidates failed to launch, or algo absent "
                                     "from this surface/subset .so)"})
                    continue
                ceiling = af._tier_max_threads(tier, max_perf)
                kmax = int(pk["kernel_max_threads"])
                if kmax > ceiling:
                    # A ceiling ABOVE the forced tier's launch-bounds cap proves
                    # the probe config was not applied to this kernel.
                    status = "probe_not_applied"
                    problems.append({"algo": key, "error":
                                     f"kernel ceiling {kmax} > forced {tier} "
                                     f"cap {ceiling} — probe NOT applied"})
                    print(f"  !! {robot}/{base}/{tier}/{key}: kernel ceiling {kmax} "
                          f"exceeds forced-{tier} cap {ceiling} — PROBE NOT APPLIED")
                    continue
                cells[key] = {
                    "max_threads": kmax,                        # E1 introspection
                    "max_threads_source": pk.get("tier_source"),
                    "min_smem": _min_smem(limits, robot, base, key, tier),
                    "suggested_threads": {PROFILE: int(pk["threads"])},
                    "us_at_optimal": {PROFILE: float(pk["us"])},
                    "batch_N": int(n),
                }
    return cells, problems, status


# ---------------------------------------------------------------------------
# Reporting + write
# ---------------------------------------------------------------------------
def print_matrix(robot, matrix, statuses, all_problems):
    print(f"\n=== per-tier matrix — {robot} (profile={PROFILE}, "
          f"metric=batch_to_land_median_us) ===")
    for base in sorted(matrix):
        for key in sorted(matrix[base]):
            for tier in TIERS:
                leaf = matrix[base][key].get(tier)
                if leaf is None:
                    continue
                sug = leaf["suggested_threads"][PROFILE]
                us = leaf["us_at_optimal"][PROFILE]
                smem = leaf["min_smem"]
                print(f"  {base:8s} {key:22s} {tier:8s} "
                      f"max_threads={leaf['max_threads']:5d}  "
                      f"suggested[{PROFILE}]={sug:5d}  {us:9.2f} us  "
                      f"min_smem={smem if smem is not None else '?'}")
    for (base, tier), st in sorted(statuses.items()):
        if st != "ok":
            print(f"  !! {base}/{tier}: {st}")
    for (base, tier), probs in sorted(all_problems.items()):
        for p in probs:
            print(f"  -- {base}/{tier}/{p['algo']}: {p['error'][:160]}")


def write_matrix(robot, gpu, matrix, n, statuses):
    """Merge the E5 `matrix` block into config/launch_configs/<robot>/<gpu>.json.
    Format mirrors bake_by_n_bucket.py / write_ffi_config: 2-space indent,
    sort_keys, trailing newline. Existing per-profile entries inside a cell are
    merged, not clobbered; cells from a non-ok (base, tier) are never written."""
    path = Path(_launch_configs_dir()) / robot / f"{gpu}.json"
    doc = json.loads(path.read_text()) if path.exists() else {}
    mx = doc.setdefault("matrix", {})
    wrote = 0
    for base, algos in matrix.items():
        for key, tiers in algos.items():
            for tier, leaf in tiers.items():
                if statuses.get((base, tier)) != "ok":
                    continue
                cell = mx.setdefault(base, {}).setdefault(key, {})
                old = cell.get(tier) or {}
                merged = dict(old)
                merged.update(leaf)
                for prof_field in ("suggested_threads", "us_at_optimal"):
                    d = dict(old.get(prof_field) or {})
                    d.update(leaf.get(prof_field) or {})
                    merged[prof_field] = d
                cell[tier] = merged
                wrote += 1
    meta = doc.setdefault("matrix_meta", {})
    meta["metric"] = "batch_to_land_median_us"
    meta["autotune_N"] = int(n)
    meta["note"] = ("per-tier probe sweep (E5 schema); regenerated by "
                    "test/benchmarks/autotune_tier_matrix.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"\n  wrote {wrote} matrix cells -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--base", default="fixed", choices=["fixed", "floating", "both"])
    ap.add_argument("--gpu", default=LAUNCH_CONFIG_DEFAULT_GPU,
                    help="results file to write into (probe bake always reads the "
                         f"{LAUNCH_CONFIG_DEFAULT_GPU}-named probe; see module doc)")
    ap.add_argument("--n", type=int, default=256, help="batch size to tune for (default 256)")
    ap.add_argument("--iters", type=int, default=100, help="timed iters per thread count")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tiers", nargs="+", default=list(TIERS), choices=list(TIERS),
                    help="tiers to probe (default: all three)")
    ap.add_argument("--algos", nargs="+", default=None,
                    help="subset of algo symbols to SWEEP (default: all bound algos)")
    ap.add_argument("--build-algos", nargs="+", default=None,
                    help="RAM-safe subset to BUILD into each probe .so (passed to "
                         "autotune_ffi.autotune_base; needed on g1/h2_plus)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"pilot mode: sweep only {list(SMOKE_ALGOS)} at iters=5 "
                         "warmup=1 and build only those algos (fast subset .so)")
    ap.add_argument("--kernel-limits", type=Path, default=None,
                    help="kernel_limits_<host>.json for the min_smem join "
                         "(default: results/kernel_limits_<host>.json if present)")
    ap.add_argument("--write", action="store_true",
                    help="merge the matrix into config/launch_configs/<robot>/<gpu>.json "
                         "(default: print only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="force print-only even if --write was given")
    ap.add_argument("--keep-probes", action="store_true",
                    help="keep the temporary tierprobe_* config dirs for inspection")
    args = ap.parse_args()

    iters, warmup = args.iters, args.warmup
    want = set(args.algos) if args.algos else None
    build_algos = list(args.build_algos) if args.build_algos else None
    if args.smoke:
        want = set(SMOKE_ALGOS) if want is None else (want & set(SMOKE_ALGOS))
        iters, warmup = min(iters, 5), min(warmup, 1)
        if build_algos is None:
            build_algos = list(SMOKE_ALGOS)
        print(f"[smoke] algos={sorted(want)} iters={iters} warmup={warmup} "
              f"build_algos={build_algos} (subset .so — its own cache entry)")

    valid_syms = {a for a, _ in af.ALGOS}
    if want is not None:
        unknown = want - valid_syms
        if unknown:
            ap.error(f"unknown --algos {sorted(unknown)}; known: {sorted(valid_syms)}")

    limits, limits_path = _load_kernel_limits(args.kernel_limits)
    if limits is None:
        print(f"[note] no kernel-limits file at {limits_path} — min_smem will be "
              "null (run collect_kernel_limits.py to fill it)")

    bases = ["fixed", "floating"] if args.base == "both" else [args.base]
    matrix = {}                  # base -> key -> tier -> leaf
    statuses = {}                # (base, tier) -> ok|build_failed|probe_not_applied
    all_problems = {}            # (base, tier) -> [{algo, error}]
    for base in bases:
        for tier in args.tiers:
            print(f"\n### probe: {args.robot}/{base} @ tier={tier} ###")
            cells, problems, status = sweep_tier(
                args.robot, base, tier, args.n, iters, warmup, want,
                build_algos, args.keep_probes, limits)
            statuses[(base, tier)] = status
            if problems:
                all_problems[(base, tier)] = problems
            for key, leaf in cells.items():
                matrix.setdefault(base, {}).setdefault(key, {})[tier] = leaf

    print_matrix(args.robot, matrix, statuses, all_problems)

    bad = sorted(bt for bt, st in statuses.items() if st == "probe_not_applied")
    if args.write and not args.dry_run:
        write_matrix(args.robot, args.gpu, matrix, args.n, statuses)
        skipped = sorted(bt for bt, st in statuses.items() if st != "ok")
        if skipped:
            print(f"  (skipped non-ok tiers: {skipped})")
    else:
        print(f"\n  (dry-run: not writing matrix; pass --write to merge into "
              f"config/launch_configs/{args.robot}/{args.gpu}.json)")
    if bad:
        print(f"\nFATAL-ISH: probe not applied for {bad} — those cells were "
              "withheld; investigate before trusting any numbers from this run.")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-algo isolated benchmark orchestrator (2026-07-15).

WHY THIS EXISTS. The two pre-existing bench paths both funnel every algorithm through ONE process:
  * monolithic timeGRiD_{batch,single}.cu -- one giant TU (24-36 GB cicc on big robots) AND one process,
    so a single kernel's gpuErrchk exit() kills the whole sweep and loses every algo after it;
  * the "per-algo TU" path split the COMPILE into small TUs (good) but still LINKED them into one
    dispatcher binary, so a crash still takes down everything after it (observed: exit 188 on iiwa14 lost
    9 of 17 algos).

This orchestrator makes each algorithm a SELF-CONTAINED .cu -> its OWN .exe -> its OWN process:
  * COMPILE: small per-algo TUs, RAM-guarded parallelism (never OOM the box), ccache-friendly (edit one
    algo -> recompile one TU).
  * RUN: each exe in isolation. A crash (nonzero exit / timeout / signal) is CONTAINED to that one algo
    and ATTRIBUTED to it -- every other algo still produces its numbers. No more "one kernel nukes the run".
  * SOURCE OF TRUTH: dispatch AND collection are driven by PER_ALGO_SPECS (reused from run.py) + the
    generated-header GRID_HAS_* gate. Adding a benchable algo is one PER_ALGO_SPECS row; nothing here is
    hand-maintained per-algo.

Timings are process-independent because timeGRiD_common.h now warms the GPU to its sustained boost clock
(time-based, ~1.5s) before timing -- proven to match the monolithic numbers from a cold process start
(2026-07-15). See that header.

Usage:
    python test/benchmarks/per_algo_bench.py --robot iiwa14 --base fixed [--output out.json]
                                             [--compile-jobs N] [--per-exe-timeout S]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(THIS_DIR))

# Reuse the SINGLE source of truth + header generation from the existing harness -- do NOT duplicate.
from baselines.grid import run as gridrun  # noqa: E402
from timing_parser import parse_grid_output, build_metadata, fill_nulls, ALL_ALGOS  # noqa: E402

# Autotune mode reuses run.py's picker VERBATIM (schema-2 algo_picks, tie-breaks, SASS tier-dedup,
# one-level refinement) so the wrapper's picks are structurally identical to the monolithic path by
# construction -- that is exactly what the Phase-2 gate checks. We only feed it a per-algo {tier: exe}
# dict instead of one linked binary per tier; each solo exe emits just its own algo, so the picker's
# _present_algos_in_binary / _sweep_one_binary / dedup all operate correctly on it.
_AUTOTUNE_TIERS = gridrun.AUTOTUNE_TIERS                       # ("shared", "lite", "minimal")
_TIER_MACRO = {"shared": None, "lite": "TIER_LITE", "minimal": "TIER_MINIMAL"}  # shared == default (no -D)


# --------------------------------------------------------------------------- self-contained per-algo TU
def _solo_batch_tu_source(algo_key: str) -> str:
    """A COMPLETE .cu for one algo: its measure entry + a main that runs only it, sweeping N.

    Reuses `gridrun._per_algo_batch_tu_source` for the entry body (the measure_<algo>_batch_entry that
    already encodes the exact PER_ALGO_SPECS call), then appends a self-contained main -- so each algo
    links to its own binary with no shared dispatcher.
    """
    spec = gridrun.PER_ALGO_SPECS[algo_key]
    gate = spec.get("gate")
    open_g = f"#if {gate}\n" if gate else ""
    close_g = "#endif\n" if gate else ""
    entry = gridrun._per_algo_batch_tu_source(algo_key)  # includes timeGRiD_common.h + the entry
    # Per-algo kernel-attr registration functor (P1 / Fix #1): the solo TU calls the
    # run_all_tests overload that takes this functor + init_grid_streams, so it NEVER
    # pulls in init_grid's monolith init_grid_kernel_attrs (which address-takes ALL ~35
    # kernels -> whole-set instantiation -> ptxas OOM on big humanoids). Registers just
    # this algo (guarded, since it sits OUTSIDE the measure entry's own gate). Same
    # per-algo registration the measure entry does internally -- idempotent.
    attr_init = f"[](){{ {gridrun._attr_init_call(algo_key, spec, guarded=True)} }}"
    # main: run_all_tests provides init/load/warmup/close; call only THIS algo's entry per N.
    # If the algo is gated out for this header (GRID_HAS_* == 0), the entry does not exist, so the main
    # must also be gated -- and we still emit a valid, do-nothing main so the exe builds and exits clean
    # (the wrapper then records the algo as "gated out", not as a failure).
    main = (
        "\n"
        "int main(int argc, const char **argv){\n"
        "    bool floating_base = parse_floating_base_arg(argc, argv);\n"
        "    run_all_tests<float, 1024>(floating_base, [&](cudaStream_t *streams, grid::robotModel<float> *m, grid::gridData<float> *d){\n"
        "#if !TEST_FOR_EQUIVALENCE\n"
        f"{open_g}"
        "        for (int N : {16, 32, 64, 128, 256, 1024}) {\n"
        f"            measure_{algo_key}_batch_entry(N, streams, m, d);\n"
        "        }\n"
        f"{close_g}"
        "        (void)streams; (void)m; (void)d;\n"
        "#else\n"
        "        (void)streams; (void)m; (void)d;\n"
        "#endif\n"
        f"    }}, {attr_init});\n"
        "    (void)floating_base;\n"
        "    return 0;\n"
        "}\n"
    )
    return entry + main


# --------------------------------------------------------------------------- RAM guard
def _ram_avail_gb() -> float:
    try:
        out = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Mem:"):
                return float(line.split()[6])   # "available" column
    except Exception:
        pass
    return 0.0


def _wait_for_ram(min_gb: float, label: str) -> None:
    """Block until at least `min_gb` is available, so parallel compiles never OOM the box."""
    waited = 0
    while _ram_avail_gb() < min_gb:
        if waited == 0:
            print(f"  [ram-guard] {label}: waiting for {min_gb:.0f} GB free (have {_ram_avail_gb():.0f})...")
        time.sleep(5)
        waited += 5
        if waited > 1800:   # 30 min: something is wrong, stop waiting silently
            print(f"  [ram-guard] {label}: still short after 30 min; proceeding anyway")
            return


# --------------------------------------------------------------------------- compile + run one algo
def _tier_suffix(tier: str | None) -> str:
    """Exe/src filename suffix for a resource tier. 'shared' (== the TIER_SHARED default) and None both
    map to NO suffix, so the shared-tier autotune exe IS the timing exe -- the expensive SO-monster
    compile is paid once and serves both the timing pass and the autotune 'shared' tier."""
    return "" if tier in (None, "shared") else f"__tier_{tier}"


# Compile-heavy SO / second-order family. Their cicc/ptxas can balloon to tens of GB
# (measured overnight: cicc 21.5 GB on a g1-floating SO twin), and the `_wait_for_ram`
# guard is PRE-ADMISSION ONLY — it checks free RAM before launch but cannot throttle a
# compile that grows AFTER admission. Two of these in parallel OOM-killed the box (and
# the VSCode scope, and Claude Code with it). So: force these SERIAL and put a HARD
# cgroup ceiling on each (see _cgroup_wrap / _compile_algos).
_SO_FAMILY_ALGOS = frozenset({
    "idsva_so", "idsva_so_body_frame", "idsva_so_world_frame", "fdsva_so",
    "end_effector_pose_hessian",
    # mjx twins of the second-order kernels are just as heavy -> serial + cgroup-capped too.
    "idsva_so_world_frame_mjx", "fdsva_so_mjx",
})


def _cgroup_available() -> bool:
    """True iff `systemd-run --user --scope` works here (a user systemd manager is up).

    Headless/cron runs may lack a user manager; then we fall back to the pre-admission
    RAM guard alone. Cached on first call."""
    if getattr(_cgroup_available, "_cached", None) is None:
        ok = False
        if shutil.which("systemd-run"):
            try:
                r = subprocess.run(
                    ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=256M",
                     "--collect", "/bin/true"],
                    capture_output=True, text=True, timeout=30)
                ok = r.returncode == 0
            except Exception:
                ok = False
        _cgroup_available._cached = ok
    return _cgroup_available._cached


def _cgroup_wrap(cmd: list[str], cap_gb: float) -> list[str]:
    """Prefix a HARD memory ceiling onto `cmd` via a transient user cgroup scope.

    The cap is enforced on the scope, and nvcc's children (cudafe++/cicc/ptxas — the
    actual RAM hogs) inherit it, so a runaway device compile is OOM-killed inside its
    own scope instead of taking down the box. MemorySwapMax=0 makes the ceiling real
    (no swap-thrash past it). No-op when a user manager isn't available."""
    if cap_gb <= 0 or not _cgroup_available():
        return cmd
    return [
        "systemd-run", "--user", "--scope", "-q", "--collect",
        "-p", f"MemoryMax={cap_gb:.0f}G", "-p", "MemorySwapMax=0",
        *cmd,
    ]


def _nvcc_cmd(src: Path, exe: Path, header_file: Path, arch: str, tier: str | None = None) -> list[str]:
    nvcc = shutil.which("nvcc") or "nvcc"
    cmd = [
        nvcc, "-std=c++17", "-O3", f"-arch=sm_{arch}",
        "-I", str(header_file.parent), "-I", str(REPO_ROOT), "-I", str(THIS_DIR / "baselines" / "grid"),
        "-DGRID_HEADER_FILE=" + f'"{header_file}"',   # generate_header names it <robot>_<base>.cuh, not grid.cuh
    ]
    macro = _TIER_MACRO.get(tier)
    if macro is not None:   # shared == default => no flag (byte-identical to run.py's TIER_SHARED)
        cmd.append(f"-DGRID_DEFAULT_RESOURCE_TIER={macro}")
    cmd += ["-o", str(exe), str(src)]
    return cmd


def _compile_one(algo: str, build_dir: Path, header_file: Path, arch: str,
                 ram_per_compile_gb: float, tier: str | None = None,
                 cgroup_cap_gb: float = 0.0) -> tuple[str, Path | None, str]:
    """Write the algo's self-contained .cu and compile it to an .exe. Returns (algo, exe|None, log).

    The .cu is tier-independent (the resource tier is a compile-time -D flag, not source), so the source
    text is shared across tiers -- only the .exe differs. `tier` selects the -DGRID_DEFAULT_RESOURCE_TIER
    macro + the exe suffix; None/'shared' = the default tier (timing path)."""
    sfx = _tier_suffix(tier)
    src = build_dir / f"solo_batch_{algo}.cu"
    src_txt = _solo_batch_tu_source(algo)
    gridrun._write_if_changed(src, src_txt)
    exe = build_dir / f"solo_batch_{algo}{sfx}.exe"
    stamp = build_dir / f"solo_batch_{algo}{sfx}.stamp"
    # CONTENT-keyed compile cache. generate_header rewrites the .cuh (fresh mtime) on EVERY run -- even
    # on a content cache-hit -- so mtime is unreliable. Key on (source text + tier flag + header bytes)
    # instead. This makes the autotune 'shared' tier reuse the timing pass's suffix-less exe (no double
    # ~355s compile), makes a resumed sweep cheap, and can't be fooled by mtime churn. The exe path
    # encodes the tier via `sfx`, so tiers never alias.
    key = hashlib.sha1(
        (src_txt + "\0" + str(_TIER_MACRO.get(tier)) + "\0" + header_file.read_text()).encode()
    ).hexdigest()
    if exe.exists() and stamp.exists() and stamp.read_text().strip() == key:
        return algo, exe, "cache hit (content stamp match)"
    _wait_for_ram(ram_per_compile_gb, f"compile {algo}{sfx}")
    t0 = time.monotonic()
    cmd = _cgroup_wrap(_nvcc_cmd(src, exe, header_file, arch, tier), cgroup_cap_gb)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        # 137 = SIGKILL, the OOM-killer's signature when the cgroup ceiling is hit.
        oom = " (likely OOM-killed at the cgroup MemoryMax ceiling)" if proc.returncode == 137 else ""
        return algo, None, f"COMPILE FAILED rc={proc.returncode}{oom} ({dt:.0f}s):\n{proc.stderr[-2000:]}"
    stamp.write_text(key)
    return algo, exe, f"compiled ({dt:.0f}s)"


def _run_one(algo: str, exe: Path, base: str, timeout_s: float) -> tuple[str, dict, str, str]:
    """Run one algo's exe in isolation. Returns (algo, data, status, log).

    status is one of:
      "ok"     -- produced timing rows
      "gated"  -- clean exit, NO rows: the algo is GATED OUT of this header (GRID_HAS_* == 0), which is
                  NORMAL (e.g. an opt-in family the robot wasn't generated with). NOT a failure.
      "crash"  -- nonzero exit (e.g. 188). CONTAINED to this algo + attributed; the sweep continues.
      "timeout"/"error" -- likewise contained.
    """
    try:
        proc = subprocess.run([str(exe), base], capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return algo, {}, "timeout", f"TIMEOUT after {timeout_s:.0f}s (isolated -- other algos unaffected)"
    except Exception as e:  # noqa: BLE001
        return algo, {}, "error", f"RUN ERROR {e!r}"
    if proc.returncode != 0:
        return algo, {}, "crash", (f"CRASH rc={proc.returncode} (isolated -- other algos unaffected)\n"
                                   f"stderr tail:\n{proc.stderr[-800:]}")
    got = {k: v for k, v in parse_grid_output(proc.stdout).items() if v}
    if got:
        return algo, got, "ok", f"ok ({len(got)} row group(s))"
    return algo, {}, "gated", "gated out of this header (no rows) -- normal, not a failure"


# --------------------------------------------------------------------------- compile / run fan-out
def _compile_algos(algos: list[str], build_dir: Path, header: Path, arch: str,
                   ram_per_compile_gb: float, jobs: int, tier: str | None = None,
                   cgroup_cap_gb: float = 0.0) -> dict[str, Path]:
    """Compile each algo's solo exe for `tier`. Returns {algo: exe}.

    Two-phase so the compile-heavy SO family can't OOM the box:
      1. the SO family (_SO_FAMILY_ALGOS) compiles SERIALLY, each under a hard cgroup
         MemoryMax ceiling (the `_wait_for_ram` pre-admission guard can't throttle a
         compile that balloons after launch; two SO compiles in parallel is exactly
         what killed the box overnight);
      2. everything else compiles RAM-guarded parallel as before.
    The cgroup ceiling still applies to the light phase (cheap, harmless) so a
    surprise heavyweight is contained too."""
    exes: dict[str, Path] = {}
    label = _tier_suffix(tier) or "(shared)"
    so_algos = [a for a in algos if a in _SO_FAMILY_ALGOS]
    light_algos = [a for a in algos if a not in _SO_FAMILY_ALGOS]

    def _record(algo: str, exe: Path | None, log: str) -> None:
        print(f"  [compile{label}] {algo}: {log.splitlines()[0]}")
        if exe is not None:
            exes[algo] = exe
        else:
            print(log)

    if so_algos:
        cap_note = (f"cgroup MemoryMax={cgroup_cap_gb:.0f}G" if cgroup_cap_gb and _cgroup_available()
                    else "no cgroup cap (systemd --user unavailable — RAM guard only)")
        print(f"  [compile{label}] {len(so_algos)} SO-family algo(s) SERIAL, {cap_note}: "
              f"{', '.join(so_algos)}")
        for a in so_algos:
            _record(*_compile_one(a, build_dir, header, arch, ram_per_compile_gb, tier, cgroup_cap_gb))

    if light_algos:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_compile_one, a, build_dir, header, arch,
                                ram_per_compile_gb, tier, cgroup_cap_gb): a
                    for a in light_algos}
            for fut in concurrent.futures.as_completed(futs):
                _record(*fut.result())
    return exes


def _run_isolated(algos: list[str], exes: dict[str, Path], base: str,
                  timeout_s: float) -> tuple[dict, list[str], list[str]]:
    """Run each algo's exe ISOLATED, serially (timing must not overlap on the GPU). Crashes are
    contained + attributed. Returns (results, gated, crashed)."""
    results: dict[str, dict] = {}
    crashed: list[str] = []
    gated: list[str] = []
    for algo in algos:
        if algo not in exes:
            crashed.append(f"{algo}(compile)")
            print(f"  [run] {algo}: SKIPPED (compile failed)")
            continue
        a, got, status, log = _run_one(algo, exes[algo], base, timeout_s)
        print(f"  [run] {a}: {log.splitlines()[0]}")
        if status == "ok":
            results.update(got)
        elif status == "gated":
            gated.append(a)
        else:
            crashed.append(f"{a}({status})")
    return results, gated, crashed


# --------------------------------------------------------------------------- autotune (tier x threads)
def _run_autotune(algos: list[str], build_dir: Path, header: Path, arch: str, base: str,
                  ram_per_compile_gb: float, jobs: int, *, thread_grid: tuple[int, ...],
                  autotune_N: int, tiers: tuple[str, ...], cgroup_cap_gb: float = 0.0) -> dict[str, dict]:
    """Build each algo's {tier: solo_exe} set and run run.py's picker VERBATIM on it -> schema-2
    algo_picks[algo] (tier_optimal/threads_optimal/us_at_optimal/sweep/sweep_us[/tier_equiv_to]).

    The 'shared' tier reuses the suffix-less timing exe (no -D flag), so the expensive SO-monster
    compile is paid once. Each solo exe emits only its own algo, so the picker's SASS tier-dedup,
    thread sweep + one-level refinement all operate correctly per-algo. The picker runs the exes
    SERIALLY within an algo, and we loop algos serially -- so no two timing launches overlap."""
    max_perf = gridrun._read_max_perf_level_threads(header)
    print(f"[autotune] MAX_PERF_LEVEL_THREADS={max_perf} | tiers={list(tiers)} | N={autotune_N} "
          f"| thread grid={list(thread_grid)}")
    # Compile every (algo, tier) exe up front, one tier at a time so a tier's SO-monster compiles
    # finish + free RAM before the next tier starts (RAM-guarded within each tier too).
    tier_exes: dict[str, dict[str, Path]] = {}
    for tier in tiers:
        print(f"[autotune] compiling tier={tier} for {len(algos)} algos...")
        tier_exes[tier] = _compile_algos(algos, build_dir, header, arch, ram_per_compile_gb, jobs, tier,
                                         cgroup_cap_gb=cgroup_cap_gb)

    algo_picks: dict[str, dict] = {}
    for algo in algos:
        binaries = {t: tier_exes[t][algo] for t in tiers if algo in tier_exes.get(t, {})}
        if not binaries:
            print(f"  [autotune] {algo}: no tier exe built -- skipped")
            continue
        picks = gridrun._autotune_pick_winners(
            binaries, base, thread_grid=thread_grid, autotune_N=autotune_N,
            max_perf_level_threads=max_perf, mode="batch")
        if algo in picks:
            algo_picks[algo] = picks[algo]
            p = picks[algo]
            print(f"  [autotune] {algo}: tier={p['tier_optimal']} threads={p['threads_optimal']} "
                  f"us={p['us_at_optimal']:.3f}"
                  + (f" (tier_equiv {p['tier_equiv_to']})" if "tier_equiv_to" in p else ""))
        else:
            print(f"  [autotune] {algo}: no readings at any (tier,threads) -- omitted")
    return algo_picks


def _repick_from_sweep(pick: dict) -> dict:
    """Re-derive (tier_optimal, threads_optimal, us_at_optimal, sweep_us) from a saved pick['sweep']
    grid WITHOUT re-timing -- the cheap --stage analyze path. Uses run.py's argmin so the tie-break
    (first-encountered on ties) matches the sweep-time pick exactly."""
    by_tier = {t: {int(th): float(us) for th, us in s.items()}
               for t, s in pick.get("sweep", {}).items()}
    best = gridrun._argmin_tier_threads(by_tier)
    if best is None:
        return pick
    wtier, wthreads, wus = best
    pick["tier_optimal"] = wtier
    pick["threads_optimal"] = int(wthreads)
    pick["us_at_optimal"] = float(wus)
    pick["sweep_us"] = {str(int(th)): float(us) for th, us in sorted(by_tier.get(wtier, {}).items())}
    return pick


def _emit_tier_analysis(robot: str, base: str, algo_picks: dict[str, dict]) -> str:
    """Markdown tier-comparison table from the autotune sweeps: best-achievable us per tier (min over
    the thread sweep) + lite/minimal-vs-shared ratios. Salvages analyze_tier_sweep.py's ratio table
    onto the autotune data so that orphan analyzer can be retired."""
    def best(by_tier: dict, tier: str):
        s = by_tier.get(tier)
        return min(s.values()) if s else None

    def ratio(num, den):
        return f"{num / den:.2f}x" if (num is not None and den not in (None, 0)) else "—"

    def fmt(v):
        return f"{v:.2f}" if v is not None else "—"

    lines = [
        f"# Autotune tier analysis — {robot}-{base}", "",
        f"Best-achievable us per tier (min over the thread sweep at N={gridrun.DEFAULT_AUTOTUNE_N}); "
        "ratios are tier/shared (>1.0 = tier slower). `opt` = the joint (tier, threads) argmin.", "",
        "| Algo | opt tier | opt thr | opt us | shared | lite | minimal | lite/shd | min/shd |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for algo in sorted(algo_picks):
        p = algo_picks[algo]
        by_tier = {t: {int(th): float(us) for th, us in s.items()}
                   for t, s in p.get("sweep", {}).items()}
        sh, li, mi = best(by_tier, "shared"), best(by_tier, "lite"), best(by_tier, "minimal")
        lines.append(
            f"| {algo} | {p.get('tier_optimal', '—')} | {p.get('threads_optimal', '—')} | "
            f"{fmt(p.get('us_at_optimal'))} | {fmt(sh)} | {fmt(li)} | {fmt(mi)} | "
            f"{ratio(li, sh)} | {ratio(mi, sh)} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- orchestrate
def _report_run(results: dict, gated: list[str], crashed: list[str], out: Path) -> None:
    print(f"\n[per-algo] {len(results)} timed | {len(gated)} gated-out | {len(crashed)} FAILED")
    if gated:
        print(f"[per-algo] gated out of this header (normal): {gated}")
    print(f"[per-algo] wrote {out}")
    # A failure does NOT fail the whole run (that is the point) -- but surface it loudly for triage.
    if crashed:
        print(f"[per-algo] ⚠ {len(crashed)} algo(s) FAILED (isolated, attributed): {crashed}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--base", required=True, choices=["fixed", "floating"])
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--mode", choices=["timing", "autotune"], default="timing",
                    help="timing = plain N-sweep (default, unchanged); autotune = tier x thread sweep "
                         "producing schema-2 algo_picks (+ the grid timing block)")
    ap.add_argument("--stage", choices=["sweep", "analyze", "both"], default="both",
                    help="autotune only: sweep = build+time+write (needs a quiet GPU); analyze = re-pick "
                         "from an EXISTING sweep JSON + emit the tier table, NO timing (CPU-only); "
                         "both = sweep then analyze")
    ap.add_argument("--autotune-N", type=int, default=gridrun.DEFAULT_AUTOTUNE_N,
                    help="batch size the autotune minimizes over (default 256)")
    ap.add_argument("--tiers", type=str, default=None,
                    help="comma-separated resource tiers for autotune (default: shared,lite,minimal)")
    ap.add_argument("--tier", type=str, default=None, choices=["shared", "lite", "minimal"],
                    help="timing mode only: compile+time at this resource tier (default: shared). Lets "
                         "run_multi_version drive per-tier timing columns through the wrapper.")
    ap.add_argument("--thread-grid", type=str, default=None,
                    help="comma-separated thread counts for the autotune sweep (default: run.py's grid)")
    ap.add_argument("--analysis-output", type=Path, default=None, help="tier-analysis markdown path")
    ap.add_argument("--compile-jobs", type=int, default=0,
                    help="max concurrent compiles (0 = auto from RAM headroom)")
    ap.add_argument("--ram-per-compile-gb", type=float, default=8.0,
                    help="assumed RAM per nvcc; the RAM guard blocks a new compile below this free")
    ap.add_argument("--cgroup-cap-gb", type=float, default=45.0,
                    help="HARD per-compile memory ceiling via a transient systemd --user cgroup scope "
                         "(0 = off). The SO family compiles serially under this cap so a device compile "
                         "that balloons after admission is OOM-killed in its own scope, not on the box. "
                         "Default 45 on a 62 GB box; no-op where a user systemd manager isn't available.")
    ap.add_argument("--per-exe-timeout", type=float, default=900.0)
    ap.add_argument("--compile-only", action="store_true",
                    help="build the per-(algo[,tier]) exes and exit WITHOUT timing (the hub's build "
                         "phase). A later run cache-hits every exe (content stamp), so measurement is "
                         "pure timing on a quiet GPU -- the 'build != time' methodology.")
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--algos", type=str, default=None,
                    help="comma-separated subset of algos to build/run (default: all in scope)")
    # Header-variant flags (passed straight to generate_header, which owns the codegen). Each CHANGES
    # the generated header -> distinct content stamp -> correct rebuild, no cache collision with baked.
    ap.add_argument("--runtime-inertia", action="store_true",
                    help="source link inertias from a mutable device table (sparsity stays baked)")
    ap.add_argument("--runtime-transform", action="store_true",
                    help="source joint transforms from a mutable device table (sparsity stays baked)")
    ap.add_argument("--runtime-joint-dynamics", action="store_true",
                    help="source joint damping/friction from a mutable device table")
    ap.add_argument("--multi-target-from-collision", action="store_true",
                    help="bake the robot's collision spherization as the multi_target batch, so "
                         "multi_target_position{,_gradient} time against the real (collision-sized) batch")
    args = ap.parse_args()

    build_dir = args.build_dir or (THIS_DIR / "results" / f"per_algo_{args.robot}_{args.base}")
    build_dir.mkdir(parents=True, exist_ok=True)
    tiers = tuple(t.strip() for t in args.tiers.split(",")) if args.tiers else _AUTOTUNE_TIERS
    thread_grid = (tuple(int(t) for t in args.thread_grid.split(","))
                   if args.thread_grid else gridrun.DEFAULT_AUTOTUNE_THREAD_GRID)

    # ---- analyze-only fast path: re-pick from a saved sweep + emit the tier table. NO nvcc/GPU/header
    # (the whole point of --stage analyze: re-analyze a completed overnight sweep cheaply). ----
    if args.mode == "autotune" and args.stage == "analyze":
        out = args.output or (build_dir / f"{args.robot}_{args.base}_grid_glass.json")
        if not out.exists():
            sys.exit(f"[analyze] no sweep JSON at {out} -- run --stage sweep first")
        payload = json.loads(out.read_text())
        block = payload["results"][args.robot][args.base]
        algo_picks = block.get("algo_picks", {})
        for a in algo_picks:
            _repick_from_sweep(algo_picks[a])
        out.write_text(json.dumps(payload, indent=1))
        print(f"[analyze] re-picked {len(algo_picks)} algo(s) from saved sweeps -> {out}")
        md_path = args.analysis_output or (build_dir / f"{args.robot}_{args.base}_tier_analysis.md")
        md_path.write_text(_emit_tier_analysis(args.robot, args.base, algo_picks))
        print(f"[analyze] wrote {md_path}")
        return

    floating = args.base == "floating"

    # 1. Generate the header ONCE (reuse the harness path -> same header the monolithic bench uses).
    urdf = gridrun.get_urdf_path(args.robot)
    ee_frame = gridrun.DEFAULT_EE_FRAMES.get(args.robot, "")
    print(f"[per-algo] generating header for {args.robot}-{args.base}...")
    header = gridrun.generate_header(
        urdf, args.robot, args.base, ee_frame, build_dir,
        runtime_inertia=args.runtime_inertia,
        runtime_transform=args.runtime_transform,
        runtime_joint_dynamics=args.runtime_joint_dynamics,
        multi_target_from_collision=args.multi_target_from_collision)

    # Which algos are in scope for this robot/base (drops non-production + mimic-unsupported).
    has_mimic = gridrun.robot_is_mimic(urdf)
    # When the user names algos explicitly, don't dedup the dispatcher-redundant SO
    # row away — they may want to build exactly `idsva_so_world_frame` for an A/B.
    algos = gridrun._algo_keys_in_registry_order(
        floating, has_mimic, dedup_dispatcher_redundant=not args.algos)
    if args.algos:
        want = {a.strip() for a in args.algos.split(",") if a.strip()}
        # Explicit --algos may name PER_ALGO_SPECS keys that are NOT in the registry-order
        # list (e.g. the mjx timing twins `<algo>_mjx`, which have no registry/descriptor row
        # -- they are bench-only). Keep registry order for the ones that are, then append any
        # remaining requested keys that exist in PER_ALGO_SPECS (their per-header GRID_HAS_* /
        # GRID_RBD_WITH_MUJOCO gate still decides whether they actually build).
        algos = [a for a in algos if a in want]
        algos += [a for a in want if a not in algos and a in gridrun.PER_ALGO_SPECS]
    arch = gridrun.detect_cuda_arch()
    print(f"[per-algo] {len(algos)} algos in scope: {', '.join(algos)}")

    jobs = args.compile_jobs or max(1, int(_ram_avail_gb() / args.ram_per_compile_gb))
    print(f"[per-algo] mode={args.mode} | compiling with up to {jobs} parallel job(s) "
          f"(RAM guard {args.ram_per_compile_gb:.0f} GB/compile)")

    # --compile-only: the hub's BUILD phase. Build the needed exes and exit; a later run cache-hits
    # every one (content stamp), so measurement is pure timing on a quiet GPU.
    if args.compile_only:
        if args.mode == "timing":
            _compile_algos(algos, build_dir, header, arch, args.ram_per_compile_gb, jobs, tier=args.tier,
                           cgroup_cap_gb=args.cgroup_cap_gb)
            what = f"tier {args.tier or 'shared'}"
        else:  # autotune: pre-build every tier so the measure run compiles nothing
            for tier in tiers:
                _compile_algos(algos, build_dir, header, arch, args.ram_per_compile_gb, jobs, tier,
                               cgroup_cap_gb=args.cgroup_cap_gb)
            what = f"tiers {','.join(tiers)}"
        print(f"[per-algo] --compile-only: exes built for {what}; skipping timing")
        return

    # === TIMING MODE (default) ==============================================================
    # --tier selects the resource tier (default None == shared == no -D flag, the original path).
    if args.mode == "timing":
        exes = _compile_algos(algos, build_dir, header, arch, args.ram_per_compile_gb, jobs, tier=args.tier,
                           cgroup_cap_gb=args.cgroup_cap_gb)
        results, gated, crashed = _run_isolated(algos, exes, args.base, args.per_exe_timeout)
        out = args.output or (build_dir / f"{args.robot}_{args.base}_grid_per_algo.json")
        payload = {
            "metadata": {**build_metadata(include_gpu=True), "robot": args.robot, "base": args.base,
                         "bench_path": "per_algo_isolated", "resource_tier": args.tier or "shared"},
            "results": {args.robot: {args.base: {"grid": results}}},
        }
        out.write_text(json.dumps(payload, indent=1))
        _report_run(results, gated, crashed, out)
        return

    # === AUTOTUNE MODE (stage sweep|both) ===================================================
    # Timing pass first (the 'grid' block) -- the SHARED-tier exes double as the autotune 'shared'
    # tier (suffix-less, no -D flag), so the expensive SO-monster compile is paid once.
    print("[autotune] --- timing pass (grid block) ---")
    exes = _compile_algos(algos, build_dir, header, arch, args.ram_per_compile_gb, jobs,
                          cgroup_cap_gb=args.cgroup_cap_gb)
    results, gated, crashed = _run_isolated(algos, exes, args.base, args.per_exe_timeout)
    filled = fill_nulls(dict(results))   # ensure the ALL_ALGOS core keys exist as null when un-run

    print("[autotune] --- tier x thread pass (algo_picks) ---")
    algo_picks = _run_autotune(algos, build_dir, header, arch, args.base,
                               args.ram_per_compile_gb, jobs, thread_grid=thread_grid,
                               cgroup_cap_gb=args.cgroup_cap_gb,
                               autotune_N=args.autotune_N, tiers=tiers)

    # Assemble the run.py-faithful autotune payload: results[robot][base] = {"grid": filled,
    # "algo_picks": {...}} with a schema-2 autotune_threads metadata block. Column key stays "grid"
    # exactly as run.py emits it; the grid->grid_glass RENAME is run_multi_version._rename_grid_key's
    # job when this is wired into the hub (Phase 3). Default filename is the grid_glass name the
    # autotune consumers glob (build_autotune_matrix / sweep_to_autotune_best).
    meta = {**build_metadata(include_gpu=True), "robot": args.robot, "base": args.base,
            "bench_path": "per_algo_isolated",
            "autotune_threads": {"thread_grid": list(thread_grid), "autotune_N": int(args.autotune_N),
                                 "mode": "batch", "tiers": list(tiers), "schema": 2}}
    out = args.output or (build_dir / f"{args.robot}_{args.base}_grid_glass.json")
    payload = {"metadata": meta,
               "results": {args.robot: {args.base: {"grid": filled, "algo_picks": algo_picks}}}}
    out.write_text(json.dumps(payload, indent=1))
    _report_run(results, gated, crashed, out)
    print(f"[autotune] {len(algo_picks)} algo_picks written")

    if args.stage == "both":
        md_path = args.analysis_output or (build_dir / f"{args.robot}_{args.base}_tier_analysis.md")
        md_path.write_text(_emit_tier_analysis(args.robot, args.base, algo_picks))
        print(f"[autotune] wrote {md_path}")


if __name__ == "__main__":
    main()

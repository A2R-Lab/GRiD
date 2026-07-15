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
from timing_parser import parse_grid_output, build_metadata  # noqa: E402


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
        "    });\n"
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
def _nvcc_cmd(src: Path, exe: Path, header_file: Path, arch: str) -> list[str]:
    nvcc = shutil.which("nvcc") or "nvcc"
    return [
        nvcc, "-std=c++17", "-O3", f"-arch=sm_{arch}",
        "-I", str(header_file.parent), "-I", str(REPO_ROOT), "-I", str(THIS_DIR / "baselines" / "grid"),
        "-DGRID_HEADER_FILE=" + f'"{header_file}"',   # generate_header names it <robot>_<base>.cuh, not grid.cuh
        "-o", str(exe), str(src),
    ]


def _compile_one(algo: str, build_dir: Path, header_file: Path, arch: str,
                 ram_per_compile_gb: float) -> tuple[str, Path | None, str]:
    """Write the algo's self-contained .cu and compile it to an .exe. Returns (algo, exe|None, log)."""
    src = build_dir / f"solo_batch_{algo}.cu"
    gridrun._write_if_changed(src, _solo_batch_tu_source(algo))
    exe = build_dir / f"solo_batch_{algo}.exe"
    _wait_for_ram(ram_per_compile_gb, f"compile {algo}")
    t0 = time.monotonic()
    proc = subprocess.run(_nvcc_cmd(src, exe, header_file, arch), capture_output=True, text=True)
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        return algo, None, f"COMPILE FAILED rc={proc.returncode} ({dt:.0f}s):\n{proc.stderr[-2000:]}"
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


# --------------------------------------------------------------------------- orchestrate
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--base", required=True, choices=["fixed", "floating"])
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--compile-jobs", type=int, default=0,
                    help="max concurrent compiles (0 = auto from RAM headroom)")
    ap.add_argument("--ram-per-compile-gb", type=float, default=8.0,
                    help="assumed RAM per nvcc; the RAM guard blocks a new compile below this free")
    ap.add_argument("--per-exe-timeout", type=float, default=900.0)
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--algos", type=str, default=None,
                    help="comma-separated subset of algos to build/run (default: all in scope)")
    args = ap.parse_args()

    build_dir = args.build_dir or (THIS_DIR / "results" / f"per_algo_{args.robot}_{args.base}")
    build_dir.mkdir(parents=True, exist_ok=True)
    floating = args.base == "floating"

    # 1. Generate the header ONCE (reuse the harness path -> same header the monolithic bench uses).
    urdf = gridrun.get_urdf_path(args.robot)
    ee_frame = gridrun.DEFAULT_EE_FRAMES.get(args.robot, "")
    print(f"[per-algo] generating header for {args.robot}-{args.base}...")
    header = gridrun.generate_header(urdf, args.robot, args.base, ee_frame, build_dir)

    # Which algos are in scope for this robot/base (drops non-production + mimic-unsupported).
    has_mimic = gridrun.robot_is_mimic(urdf)
    algos = gridrun._algo_keys_in_registry_order(floating, has_mimic)
    if args.algos:
        want = {a.strip() for a in args.algos.split(",") if a.strip()}
        algos = [a for a in algos if a in want]
    arch = gridrun.detect_cuda_arch()
    print(f"[per-algo] {len(algos)} algos in scope: {', '.join(algos)}")

    jobs = args.compile_jobs or max(1, int(_ram_avail_gb() / args.ram_per_compile_gb))
    print(f"[per-algo] compiling with up to {jobs} parallel job(s) (RAM guard {args.ram_per_compile_gb:.0f} GB/compile)")

    # 2. COMPILE each algo to its own exe, RAM-guarded parallel.
    exes: dict[str, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futs = {pool.submit(_compile_one, a, build_dir, header, arch, args.ram_per_compile_gb): a
                for a in algos}
        for fut in concurrent.futures.as_completed(futs):
            algo, exe, log = fut.result()
            print(f"  [compile] {algo}: {log.splitlines()[0]}")
            if exe is not None:
                exes[algo] = exe
            else:
                print(log)

    # 3. RUN each exe ISOLATED, serially (timing must not overlap on the GPU). Crashes are contained.
    results: dict[str, dict] = {}
    crashed: list[str] = []   # real failures (compile/crash/timeout) -- attributed, sweep continues
    gated: list[str] = []     # gated out of this header -- normal
    for algo in algos:
        if algo not in exes:
            crashed.append(f"{algo}(compile)")
            print(f"  [run] {algo}: SKIPPED (compile failed)")
            continue
        a, got, status, log = _run_one(algo, exes[algo], args.base, args.per_exe_timeout)
        print(f"  [run] {a}: {log.splitlines()[0]}")
        if status == "ok":
            results.update(got)
        elif status == "gated":
            gated.append(a)
        else:
            crashed.append(f"{a}({status})")

    # 4. Assemble the results JSON in the SAME schema run.py writes.
    out = args.output or (build_dir / f"{args.robot}_{args.base}_grid_per_algo.json")
    payload = {
        "metadata": {**build_metadata(include_gpu=True), "robot": args.robot, "base": args.base,
                     "bench_path": "per_algo_isolated"},
        "results": {args.robot: {args.base: {"grid": results}}},
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"\n[per-algo] {len(results)} timed | {len(gated)} gated-out | {len(crashed)} FAILED")
    if gated:
        print(f"[per-algo] gated out of this header (normal): {gated}")
    print(f"[per-algo] wrote {out}")
    # A failure does NOT fail the whole run (that is the point) -- but surface it loudly for triage.
    if crashed:
        print(f"[per-algo] ⚠ {len(crashed)} algo(s) FAILED (isolated, attributed): {crashed}")


if __name__ == "__main__":
    main()

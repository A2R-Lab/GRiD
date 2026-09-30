#!/usr/bin/env python3
"""Deterministic per-(robot, base, algo, tier) kernel-limit collector (A1a Phase 1).

MEASUREMENT-ONLY / READ-ONLY. This script does NOT modify any codegen, kernel,
binding, or grid.cuh emit. It only READS values that GRiD already emits or that
the CUDA driver already reports:

  * ``max_threads`` — the register/launch-bounds-limited maxThreadsPerBlock for
    that (algo, tier) kernel. This is the SAME quantity the production runtime
    clamp reads via ``cudaFuncGetAttributes(kernel).maxThreadsPerBlock``
    (bindings/grid_rbd/wrapper_template.cu :: grid_clamp_threads_for). Every
    benchmarked kernel carries ``__launch_bounds__(tier_max_threads<TIER>())``,
    so ptxas guarantees
        maxThreadsPerBlock == min(tier_max_threads<TIER>(), register_cap)
    where register_cap = floor(65536 / num_regs) rounded DOWN to a warp (32).
    We mirror exactly that: the per-tier launch-bounds cap comes from run.py's
    ``_tier_thread_cap`` (which mirrors grid.cuh's ``tier_max_threads<TIER>()``),
    and ``num_regs`` comes from cuobjdump of the already-compiled bench binary.

  * ``min_smem`` — the dynamic shared-memory bytes the tier's kernel requires,
    read straight from the EXISTING emitted ``constexpr ...
    <ALGO>_DYNAMIC_SHARED_MEM_BYTES<float, TIER>()`` macros (we compile a tiny
    host TU that #includes the robot's generated grid.cuh and prints each macro).
    We only READ these macros — they already exist.

  * ``num_regs`` — cross-check, from ``cuobjdump -elf`` of the bench binary (the
    same cuobjdump path run.py already uses in ``_kernel_sass_hash``).

Output: results/kernel_limits_<host>.json keyed
    limits[robot][base][algo][tier] -> {max_threads, min_smem, num_regs}
plus a metadata block. Cells whose smem macro / kernel symbol is absent get
nulls (never a bogus value) — exactly mirroring the harness's
"symbol absent -> sweep normally" graceful-skip discipline.

Usage:
    python test/benchmarks/collect_kernel_limits.py --robot iiwa14 --base fixed
    python test/benchmarks/collect_kernel_limits.py \
        --robot iiwa14 --base fixed --algo crba --tier shared
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Reuse the mature harness machinery verbatim — do NOT re-implement.
from test.benchmarks.baselines.grid.run import (  # noqa: E402
    AUTOTUNE_TIERS,
    PER_ALGO_SPECS,
    _algo_kernel_symbol_base,
    _algo_keys_in_registry_order,
    _read_max_perf_level_threads,
    _tier_thread_cap,
    detect_cuda_arch,
    generate_header,
    get_urdf_path,
    robot_is_mimic,
)
# per_algo_bench builds a self-contained solo exe per (algo, tier). Each solo exe LINKS ALL kernels
# (via init_grid_kernel_attrs in run_all_tests), so cuobjdump -res-usage on ONE solo exe per tier yields
# every algo's register count -- exactly what collect_regs_and_max_threads needs. This replaces the old
# run.py per-algo-tus dispatcher (build_tier_binaries), deleted in the per-exe bench cutover.
from test.benchmarks import per_algo_bench as pab  # noqa: E402

# Tier name -> grid.cuh enum int (grid::TIER_SHARED=0, TIER_LITE=1, TIER_MINIMAL=2).
TIER_ENUM: dict[str, str] = {
    "shared": "grid::TIER_SHARED",
    "lite": "grid::TIER_LITE",
    "minimal": "grid::TIER_MINIMAL",
}

RESULTS_DIR = REPO_ROOT / "test" / "benchmarks" / "results"


def _host() -> str:
    return platform.node().replace(" ", "_")


# ---------------------------------------------------------------------------
# min_smem — read the existing *_DYNAMIC_SHARED_MEM_BYTES<float, TIER>() macros.
# ---------------------------------------------------------------------------
def _smem_macro_for(algo: str) -> str | None:
    """The dynamic-smem macro name for an algo, reusing PER_ALGO_SPECS'
    `shared_mem_skip` (the exact macro the bench's runtime skip already reads)."""
    spec = PER_ALGO_SPECS.get(algo)
    if not spec:
        return None
    return spec.get("shared_mem_skip")


def _macro_present_and_takes_tier(header_text: str, macro: str) -> tuple[bool, bool]:
    """(present, takes_TIER) for a *_DYNAMIC_SHARED_MEM_BYTES macro in the header.

    Some macros are `<typename T>` (tier-invariant smem) and some are
    `<typename T, int TIER = ...>`. We must instantiate each with the right arity.
    """
    # Match the template line immediately preceding the macro's `inline size_t NAME(`.
    m = re.search(
        r"template <([^>]*)>\s*__host__ __device__ inline size_t "
        + re.escape(macro) + r"\(",
        header_text,
    )
    if not m:
        return (False, False)
    takes_tier = "TIER" in m.group(1)
    return (True, takes_tier)


def collect_min_smem(
    header_path: Path,
    algos: list[str],
    tiers: tuple[str, ...],
    *,
    arch: str,
    build_dir: Path,
    nvcc: str,
) -> dict[str, dict[str, int | None]]:
    """Compile + run a tiny host TU that prints each
    `<ALGO>_DYNAMIC_SHARED_MEM_BYTES<float, TIER>()`. Returns
    {algo: {tier: bytes|None}}. Host-only TU — no kernel codegen, fast compile."""
    header_text = header_path.read_text()

    # Build (macro -> (present, takes_tier)) once; dedup macros (bias/ID share one).
    macro_info: dict[str, tuple[bool, bool]] = {}
    algo_macro: dict[str, str | None] = {}
    for algo in algos:
        macro = _smem_macro_for(algo)
        algo_macro[algo] = macro
        if macro and macro not in macro_info:
            macro_info[macro] = _macro_present_and_takes_tier(header_text, macro)

    # Emit print lines. We tag each line `SMEM <MACRO> <TIER> <bytes>` so the
    # parser keys by (macro, tier) — algos sharing a macro read the same number.
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for macro, (present, takes_tier) in macro_info.items():
        if not present:
            continue
        for tier in tiers:
            key = (macro, tier)
            if key in seen:
                continue
            seen.add(key)
            if takes_tier:
                call = f"grid::{macro}<float, {TIER_ENUM[tier]}>()"
            else:
                call = f"grid::{macro}<float>()"
            lines.append(
                f'    printf("SMEM {macro} {tier} %zu\\n", (size_t){call});'
            )

    src = (
        "// AUTO-GENERATED by collect_kernel_limits.py — measurement-only,\n"
        "// reads existing *_DYNAMIC_SHARED_MEM_BYTES macros. Do not hand-edit.\n"
        "#include <cstdio>\n"
        "#include <cstddef>\n"
        f'#include "{header_path}"\n'
        "int main(){\n"
        + "\n".join(lines) + "\n"
        "    return 0;\n"
        "}\n"
    )
    src_path = build_dir / "collect_smem.cu"
    src_path.write_text(src)
    exe_path = build_dir / "collect_smem.exe"

    compile_cmd = [
        nvcc, "-std=c++17", "-o", str(exe_path), str(src_path),
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
    ]
    res = subprocess.run(compile_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"min_smem TU compile failed:\n{res.stdout}\n{res.stderr}"
        )
    run = subprocess.run([str(exe_path)], capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"min_smem TU run failed:\n{run.stdout}\n{run.stderr}")

    # Parse: (macro, tier) -> bytes
    by_macro_tier: dict[tuple[str, str], int] = {}
    for ln in run.stdout.splitlines():
        m = re.match(r"SMEM (\S+) (\S+) (\d+)", ln.strip())
        if m:
            by_macro_tier[(m.group(1), m.group(2))] = int(m.group(3))

    out: dict[str, dict[str, int | None]] = {}
    for algo in algos:
        macro = algo_macro[algo]
        out[algo] = {}
        for tier in tiers:
            out[algo][tier] = by_macro_tier.get((macro, tier)) if macro else None
    return out


# ---------------------------------------------------------------------------
# num_regs + max_threads — from cuobjdump of the already-compiled bench binary.
# ---------------------------------------------------------------------------
def _kernel_regs_in_binary(binary: Path) -> dict[str, int]:
    """Map kernel-symbol-base ('crba_kernel') -> register count, via
    `cuobjdump -res-usage`. Mirrors the cuobjdump usage in run.py's
    _kernel_sass_hash. Returns {} on failure."""
    try:
        res = subprocess.run(
            ["cuobjdump", "-res-usage", str(binary)],
            capture_output=True, text=True,
        )
    except OSError:
        return {}
    if res.returncode != 0 or not res.stdout:
        return {}
    out: dict[str, int] = {}
    # Blocks look like:
    #   Function _ZN4grid12crba_kernelIfLi0ELb0EEEv...:
    #   REG:128 STACK:0 SHARED:0 LOCAL:0 ...
    cur_base: str | None = None
    for ln in res.stdout.splitlines():
        fm = re.search(r"Function\s+(\S+)", ln)
        if fm:
            cur_base = _demangle_kernel_base(fm.group(1))
            continue
        rm = re.search(r"\bREG:(\d+)", ln)
        if rm and cur_base is not None:
            # First REG line after a Function line wins; keep the max if a base
            # recurs (distinct tier instantiations in one binary share the base).
            regs = int(rm.group(1))
            out[cur_base] = max(out.get(cur_base, 0), regs)
            cur_base = None
    return out


def _demangle_kernel_base(symbol: str) -> str | None:
    """Extract the '<name>_kernel' base from a mangled grid symbol, excluding the
    `_single_timing` variant. e.g. `_ZN4grid12crba_kernelIfLi0E...` -> crba_kernel."""
    # `_ZN4grid<len><name>I...` — the <len> is the identifier byte length.
    m = re.search(r"_ZN4grid\d+([A-Za-z0-9_]+?_kernel)I", symbol)
    if not m:
        return None
    base = m.group(1)
    if base.endswith("_single_timing") or "_single_timing" in symbol.split("I")[0]:
        return None
    return base


def _reg_limited_max_threads(num_regs: int) -> int:
    """floor(65536 / num_regs) rounded DOWN to a warp (32). 65536 = registers
    per SM block on all supported archs (sm_70+); this is exactly the ceiling
    ptxas honours when satisfying __launch_bounds__."""
    if num_regs <= 0:
        return 1024
    raw = 65536 // num_regs
    return max(32, (raw // 32) * 32)


def collect_regs_and_max_threads(
    header_path: Path,
    algos: list[str],
    tiers: tuple[str, ...],
    tier_binaries: dict[str, Path],
    *,
    max_perf: int | None,
) -> tuple[dict[str, dict[str, int | None]], dict[str, dict[str, int | None]]]:
    """Returns (num_regs, max_threads), each {algo: {tier: value|None}}.

    max_threads mirrors cudaFuncGetAttributes(kernel).maxThreadsPerBlock exactly:
        min(tier_max_threads<TIER>(), reg_limited_max_threads(num_regs)).
    """
    # Per-tier register tables (cuobjdump once per tier binary).
    regs_by_tier: dict[str, dict[str, int]] = {}
    for tier, binary in tier_binaries.items():
        regs_by_tier[tier] = _kernel_regs_in_binary(binary)

    num_regs: dict[str, dict[str, int | None]] = {}
    max_threads: dict[str, dict[str, int | None]] = {}
    for algo in algos:
        kbase = _algo_kernel_symbol_base(algo)
        num_regs[algo] = {}
        max_threads[algo] = {}
        for tier in tiers:
            regs = regs_by_tier.get(tier, {}).get(kbase) if kbase else None
            num_regs[algo][tier] = regs
            launch_cap = _tier_thread_cap(tier, max_perf)
            if regs:
                max_threads[algo][tier] = min(launch_cap, _reg_limited_max_threads(regs))
            else:
                # Kernel symbol not resolvable in this tier binary (opt-in algo
                # absent, dispatch-only, or cuobjdump miss): we cannot prove the
                # register cap, so fall back to the static launch-bounds cap (the
                # tighter, always-valid bound the runtime clamp would also honour).
                max_threads[algo][tier] = launch_cap
    return num_regs, max_threads


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--base", required=True, choices=["fixed", "floating"])
    ap.add_argument("--algo", default=None,
                    help="Restrict to one algo (default: all in registry order).")
    ap.add_argument("--tier", default=None, choices=list(AUTOTUNE_TIERS),
                    help="Restrict to one tier (default: shared,lite,minimal).")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output JSON path (default: results/kernel_limits_<host>.json).")
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--no-recompile", action="store_true",
                    help="Reuse cached bench binaries even if the header changed.")
    args = ap.parse_args()

    tiers = (args.tier,) if args.tier else AUTOTUNE_TIERS

    build_dir = (args.build_dir if args.build_dir is not None else RESULTS_DIR).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        print("ERROR: nvcc not found", file=sys.stderr)
        sys.exit(1)

    arch = detect_cuda_arch()
    urdf_path = get_urdf_path(args.robot)
    has_mimic = robot_is_mimic(urdf_path)
    floating_base = (args.base == "floating")

    algos_all = _algo_keys_in_registry_order(floating_base, has_mimic)
    if args.algo:
        if args.algo not in algos_all:
            print(f"ERROR: algo {args.algo!r} not in the active set for "
                  f"{args.robot}/{args.base} (mimic={has_mimic}). "
                  f"Available: {algos_all}", file=sys.stderr)
            sys.exit(1)
        algos = [args.algo]
    else:
        algos = algos_all

    print(f"[limits] {args.robot} {args.base} (mimic={has_mimic}) "
          f"algos={len(algos)} tiers={list(tiers)}")

    # 1) Generate / reuse the bench header (same path the sweep uses).
    header_path = generate_header(
        urdf_path, args.robot, args.base,
        ee_frame="", build_dir=build_dir, no_recompile=args.no_recompile,
    )

    # 2) min_smem — read the emitted macros (host-only TU).
    print("[limits] reading *_DYNAMIC_SHARED_MEM_BYTES macros ...")
    min_smem = collect_min_smem(
        header_path, algos, tiers, arch=arch, build_dir=build_dir, nvcc=nvcc,
    )

    # 3) num_regs + max_threads — from one per-tier solo exe. Each solo exe links every kernel, so a
    #    single cuobjdump per tier reads all algos' registers (content-stamp cache -> hits the sweep's
    #    exes if per_algo_bench already built them for this robot/base/tier).
    print("[limits] building/cache-hitting one per-tier solo exe (all kernels linked) ...")
    ram_gb = float(os.environ.get("GRID_RAM_PER_COMPILE_GB", "8"))
    rep_algo = algos[0]   # any in-scope algo: its solo exe links the full kernel set regardless
    tier_binaries: dict[str, Path] = {}
    for tier in tiers:
        _, exe, log = pab._compile_one(rep_algo, build_dir, header_path, arch, ram_gb, tier=tier)
        if exe is not None:
            tier_binaries[tier] = exe
        else:
            print(f"[limits] WARN: tier {tier} solo exe build failed: {log}", file=sys.stderr)
    if not tier_binaries:
        print("[limits] WARN: no per-tier bench binary available; "
              "num_regs/max_threads will fall back to launch-bounds caps only",
              file=sys.stderr)

    max_perf = _read_max_perf_level_threads(header_path)
    num_regs, max_threads = collect_regs_and_max_threads(
        header_path, algos, tiers, tier_binaries, max_perf=max_perf,
    )

    # 4) Assemble limits[robot][base][algo][tier].
    out_path = args.output or (RESULTS_DIR / f"kernel_limits_{_host()}.json")
    doc: dict = {}
    if out_path.exists():
        try:
            doc = json.loads(out_path.read_text())
        except (OSError, json.JSONDecodeError):
            doc = {}
    doc.setdefault("metadata", {})
    doc["metadata"].update({
        "hostname": _host(),
        "cuda_arch": arch,
        "max_perf_level_threads": max_perf,
        "tier_launch_bounds_caps": {t: _tier_thread_cap(t, max_perf) for t in tiers},
        "schema": 1,
    })
    limits = doc.setdefault("limits", {})
    cell = limits.setdefault(args.robot, {}).setdefault(args.base, {})
    for algo in algos:
        cell[algo] = {
            tier: {
                "max_threads": max_threads[algo][tier],
                "min_smem": min_smem[algo][tier],
                "num_regs": num_regs[algo][tier],
            }
            for tier in tiers
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"[limits] wrote {out_path}")

    # Human-readable summary.
    for algo in algos:
        for tier in tiers:
            c = cell[algo][tier]
            print(f"    {algo:30s} {tier:8s} "
                  f"max_threads={str(c['max_threads']):>5s} "
                  f"min_smem={str(c['min_smem']):>7s} "
                  f"num_regs={str(c['num_regs']):>5s}")


if __name__ == "__main__":
    main()

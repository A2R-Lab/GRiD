"""Bake autotune_ffi sweep LOGS into config/launch_configs <profile>_bases blocks.

The overnight collection pattern (see test/benchmarks/autotune_ffi.py) runs the
sweeps --dry-run and captures stdout to one log per robot+surface leg; the
winners then live only in those logs. This tool re-parses the logs' final
"=== <surface> picks (batch-to-land) ===" summary blocks and replays them
through autotune_ffi.write_ffi_config — the SAME writer the live tool uses —
so a bake from logs is byte-equivalent to a bake the sweep would have written
itself.

Log filename contract (night-of-2026-09-11 shapes, both supported):
    <robot>_<surface>_n16.log             one leg, --base both in one process
    <robot>_<surface>_n16_<base>.log      per-base split legs (VRAM class, guide
                                          7.z9: one process per base)
Robot names may contain underscores (h1_2, h2_plus); the surface token is the
segment just before _n16. Archived failures (*.failed-*) are skipped.

base_maxperf (write_ffi_config's tier-consistency assertion input) is derived
from the picks rows themselves: a tier=shared row's `max=` IS the robot/base's
max_perf_level_threads (tier_max_threads<TIER_SHARED>() == max_perf — the same
inversion autotune_ffi._tier_for_ceiling uses). Legs whose picks contain no
shared row get base_maxperf None for that base, which skips the defensive
clamp (never wrong, just unchecked) — loudly noted in the output.

Default is a DRY RUN printing the planned JSON merges; --apply writes.
⚠ A bake edits config/launch_configs/** = a CODEGEN INPUT -> stales the cuda
receipt domain per-robot. Sequencing per docs/open-tasks TRIAGE: bake only
after cuda shards carry header-key records, and only on explicit say-so.

Usage:
    .venv/bin/python test/benchmarks/autotune_bake_from_logs.py --logs <dir>
    .venv/bin/python test/benchmarks/autotune_bake_from_logs.py --logs <dir> --apply
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[2]
sys.path.insert(0, str(REPO_ROOT))

from test.benchmarks.autotune_ffi import (  # noqa: E402
    SURFACE_PROFILE, write_ffi_config,
)
from grid_codegen.GRiDCodeGenerator import LAUNCH_CONFIG_DEFAULT_GPU  # noqa: E402

SURFACES = sorted(SURFACE_PROFILE)  # jax / numpy / torch

# `  fixed    fd_du     threads=  256    599.93 us  tier=shared   (max=512, src=kernel)`
PICK_ROW = re.compile(
    r"^\s+(fixed|floating)\s+(\S+)\s+threads=\s*(\d+)\s+([\d.]+)\s*us"
    r"\s+tier=(\S+)\s+\(max=(\d+|\?), src=(\S+)\)\s*$")
PICKS_HEADER = re.compile(r"^=== (\w+) picks \(batch-to-land\) ===$")
LEG_HEADER = re.compile(r"^=== (\S+)/(fixed|floating)\s+.*N=(\d+)\s")


def parse_log_name(path):
    """<robot>_<surface>_n16[_<base>].log -> (robot, surface, base|None)."""
    stem = path.name
    m = re.match(r"^(.*)_(%s)_n16(?:_(fixed|floating))?\.log$" % "|".join(SURFACES), stem)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def parse_log(path):
    """-> (surface_in_log|None, autotune_N|None, {base: {key: pick_dict}})."""
    picks = defaultdict(dict)
    surface_in_log = None
    autotune_n = None
    in_picks = False
    for line in path.read_text().splitlines():
        lh = LEG_HEADER.match(line)
        if lh:
            autotune_n = int(lh.group(3))
        ph = PICKS_HEADER.match(line)
        if ph:
            surface_in_log = ph.group(1)
            in_picks = True
            continue
        if not in_picks:
            continue
        m = PICK_ROW.match(line)
        if not m:
            if line.strip() and not line.startswith("  --dry-run") \
                    and not line.startswith("LEG DONE"):
                in_picks = False  # left the summary block
            continue
        base, key, thr, us, tier, kmax, src = m.groups()
        picks[base][key] = {
            "threads": int(thr), "us": float(us),
            "tier": None if tier == "?" else tier,
            "tier_source": None if src == "?" else src,
            "kernel_max_threads": None if kmax == "?" else int(kmax),
        }
    return surface_in_log, autotune_n, dict(picks)


def derive_base_maxperf(base_picks):
    """{base: max_perf|None} from the shared-tier rows (see module doc)."""
    out = {}
    for base, picks in base_picks.items():
        shared_maxes = {pk["kernel_max_threads"] for pk in picks.values()
                        if pk["tier"] == "shared" and pk["kernel_max_threads"]}
        if len(shared_maxes) > 1:
            raise SystemExit(f"ERROR: inconsistent shared-tier max= values {shared_maxes} "
                             f"for base {base} — logs disagree, refusing to bake")
        out[base] = shared_maxes.pop() if shared_maxes else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", required=True, help="directory of *_n16*.log sweep logs")
    ap.add_argument("--gpu", default=LAUNCH_CONFIG_DEFAULT_GPU)
    ap.add_argument("--robots", nargs="+", default=None, help="bake only these robots")
    ap.add_argument("--surfaces", nargs="+", default=None, choices=SURFACES,
                    help="bake only these surfaces")
    ap.add_argument("--apply", action="store_true",
                    help="write config/launch_configs (default: dry-run print)")
    args = ap.parse_args()

    log_dir = Path(args.logs)
    logs = sorted(p for p in log_dir.glob("*_n16*.log")
                  if ".failed" not in p.name)
    if not logs:
        raise SystemExit(f"ERROR: no *_n16*.log files in {log_dir}")

    # (robot, surface) -> {base: {key: pick}}; split per-base legs merge here.
    merged = defaultdict(dict)
    merged_n = {}
    for path in logs:
        parsed_name = parse_log_name(path)
        if parsed_name is None:
            print(f"  skip (unrecognized name): {path.name}")
            continue
        robot, surface, base_in_name = parsed_name
        if args.robots and robot not in args.robots:
            continue
        if args.surfaces and surface not in args.surfaces:
            continue
        surface_in_log, autotune_n, base_picks = parse_log(path)
        if not base_picks:
            raise SystemExit(f"ERROR: zero pick rows parsed from {path.name} — "
                             "zero-parse is FATAL, fix the parser or the log")
        if surface_in_log and surface_in_log != surface:
            raise SystemExit(f"ERROR: {path.name}: filename says surface {surface} "
                             f"but log summary says {surface_in_log}")
        if base_in_name and set(base_picks) != {base_in_name}:
            raise SystemExit(f"ERROR: {path.name}: filename says base {base_in_name} "
                             f"but log has picks for {sorted(base_picks)}")
        for base, picks in base_picks.items():
            if base in merged[(robot, surface)]:
                raise SystemExit(f"ERROR: duplicate picks for {robot}/{surface}/{base} "
                                 f"(second source: {path.name})")
            merged[(robot, surface)][base] = picks
        merged_n[(robot, surface)] = autotune_n
        print(f"  parsed {path.name}: {', '.join(f'{b}:{len(p)} picks' for b, p in base_picks.items())}")

    if not merged:
        raise SystemExit("ERROR: nothing to bake after filters")

    print()
    for (robot, surface), base_picks in sorted(merged.items()):
        base_maxperf = derive_base_maxperf(base_picks)
        for base, mp in base_maxperf.items():
            if mp is None:
                print(f"  NOTE {robot}/{surface}/{base}: no shared-tier row -> "
                      "base_maxperf unknown, tier-consistency clamp skipped")
        n = merged_n[(robot, surface)]
        profile = SURFACE_PROFILE[surface]
        if args.apply:
            write_ffi_config(robot, args.gpu, base_picks, n, base_maxperf,
                             surface=surface)
        else:
            plan = {base: {k: {"tier": pk["tier"], "threads": pk["threads"]}
                           for k, pk in sorted(picks.items())}
                    for base, picks in base_picks.items()}
            print(f"  DRY {robot} {profile}_bases (N={n}, maxperf={base_maxperf}):")
            print("    " + json.dumps(plan, sort_keys=True))
    if not args.apply:
        print("\n  dry-run only — re-run with --apply to write (⚠ codegen input; "
              "bake sequencing per TRIAGE.md)")


if __name__ == "__main__":
    main()

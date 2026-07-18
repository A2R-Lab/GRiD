#!/usr/bin/env python3
"""Merge a phased sweep's per-cell unified JSONs into ONE report.

WHY: run_tier_sweep_phased.sh runs run_multi_version.py once per CELL
(noSO-fixed, noSO-floating, SO-fixed, ...), and EACH invocation regenerates
benchmark_multi_version.md from only ITS OWN unified JSON — last writer wins,
so the final report reflects only the last cell (historically h2_plus = all
nulls -> an all-dashes report despite full data in the JSONs).

This tool deep-merges every benchmark_multi_version_*.json found under a sweep
root (results[robot][base][column][algo] level; later files win leaf conflicts,
which cells never produce in practice since no (robot,base,column,algo) is timed
twice) and renders the report ONCE from the merged doc via generate_report.py.

Usage:
    python test/benchmarks/merge_sweep_report.py --sweep-dir test/benchmarks/results/tier_sweep_phased_<TS> \
        [--output test/benchmarks/benchmark_multi_version.md]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# This script lives in test/benchmarks/ next to generate_report.py and the
# canonical report; keep all paths sibling-relative so it survives moves.
THIS_DIR = Path(__file__).resolve().parent
GENERATE_REPORT = THIS_DIR / "generate_report.py"


def deep_merge(dst: dict, src: dict) -> dict:
    """Recursively merge src into dst; dict-vs-dict recurses, otherwise src wins."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", type=Path, required=True,
                    help="Sweep root; all benchmark_multi_version_*.json below it are merged.")
    ap.add_argument("--output", type=Path,
                    default=THIS_DIR / "benchmark_multi_version.md",
                    help="Markdown report path (default: the canonical report).")
    ap.add_argument("--merged-json", type=Path, default=None,
                    help="Where to write the merged unified JSON "
                         "(default: <sweep-dir>/benchmark_multi_version_MERGED.json).")
    args = ap.parse_args()

    cells = sorted(p for p in args.sweep_dir.rglob("benchmark_multi_version_*.json")
                   if "MERGED" not in p.name)
    if not cells:
        print(f"ERROR: no benchmark_multi_version_*.json under {args.sweep_dir}",
              file=sys.stderr)
        sys.exit(1)

    merged: dict = {"metadata": {}, "results": {}}
    for p in cells:
        try:
            doc = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARN: skipping unreadable {p}: {e}", file=sys.stderr)
            continue
        deep_merge(merged["metadata"], doc.get("metadata") or {})
        deep_merge(merged["results"], doc.get("results") or {})
        print(f"  merged {p.relative_to(args.sweep_dir)}")

    out_json = args.merged_json or (args.sweep_dir / "benchmark_multi_version_MERGED.json")
    out_json.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    n_cells = sum(len(b) for r in merged["results"].values() for b in [r])
    print(f"[merge] {len(cells)} cell JSONs -> {out_json} "
          f"({len(merged['results'])} robots)")

    rc = subprocess.run(
        [sys.executable, str(GENERATE_REPORT),
         "--input", str(out_json), "--output", str(args.output),
         "--mode", "multi_version"],
        check=False,
    ).returncode
    if rc != 0:
        print(f"ERROR: generate_report.py exited {rc}", file=sys.stderr)
        sys.exit(rc)
    print(f"[merge] report -> {args.output}")


if __name__ == "__main__":
    main()

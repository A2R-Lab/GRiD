#!/usr/bin/env python3
"""Prune old timing-sweep output dirs under test/benchmarks/results/.

Retention housekeeping (N3, 2026-09-09). results/ accumulates multi-GB sweep
dirs; old sweeps are USUALLY dead weight, but some feed the rolling
launch-config bakes (refresh_launch_configs.sh picks the NEWEST
tier_sweep_phased_* by default), so this tool is deliberately conservative:

- Groups timestamped dirs by prefix family (everything before the trailing
  _YYYYMMDD[_HHMM] stamp), keeps the newest --keep N of EACH family.
- DRY-RUN by default: prints what it would delete; pass --yes to delete.
- Never touches loose files (autotune JSONs, .cuh caches) — dirs only.

Usage:
  .venv/bin/python test/benchmarks/prune_results.py            # preview, keep 2/family
  .venv/bin/python test/benchmarks/prune_results.py --keep 1 --yes
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
STAMP = re.compile(r"^(?P<family>.+?)_(?P<stamp>\d{8}(?:_\d{4,6})?)$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", type=int, default=2,
                    help="newest N dirs to keep per prefix family (default 2)")
    ap.add_argument("--yes", action="store_true",
                    help="actually delete (default is a dry-run preview)")
    args = ap.parse_args()

    if not RESULTS.is_dir():
        print(f"no results dir at {RESULTS}")
        return 0

    families: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for p in RESULTS.iterdir():
        if not p.is_dir():
            continue
        m = STAMP.match(p.name)
        if m:
            families[m.group("family")].append((m.group("stamp"), p))

    doomed: list[Path] = []
    for family, entries in sorted(families.items()):
        entries.sort()  # stamp-sorted, oldest first
        cut = entries[:-args.keep] if args.keep > 0 else entries
        for _, p in cut:
            doomed.append(p)
        kept = len(entries) - len(cut)
        print(f"{family}: {len(entries)} dir(s), keeping {kept}, pruning {len(cut)}")

    total = sum(sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                for p in doomed) if doomed else 0
    for p in doomed:
        if args.yes:
            print(f"delete: {p}")
            shutil.rmtree(p)
        else:
            print(f"would delete: {p}")
    print(f"{'freed' if args.yes else 'would free'} ~{total / 1e9:.2f} GB "
          f"({len(doomed)} dir(s)){'' if args.yes else ' — pass --yes to delete'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

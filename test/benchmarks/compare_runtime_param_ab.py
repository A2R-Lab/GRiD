#!/usr/bin/env python3
"""Runtime-param A/B: what does hardware-co-design mutability actually COST?

Each `--runtime-*` variant sources one class of model parameters from a MUTABLE device table instead of
baking it as a compile-time literal. The SPARSITY PATTERN stays baked in every case (that is the big
structural win and it is never given up), and every variant is BIT-IDENTICAL to the baked path until you
call its `set_*_params()` mutator. So the ONLY thing this measures is the loss of the compiler's
VALUE-folding (constant cells that used to fold ×0/×1 away become loads).

HYPOTHESIS: inertia and joint-dynamics should be ~free -- their tables are read once per body and hoist
out of the hot path. `runtime_transform` is the one expected to BITE, because its loads land per-cell
INSIDE the hot X-recompute. (Codegen diff on iiwa14-fixed: transform touches 1440 lines vs inertia 378
and joint-dynamics 216.)

WHAT THE ANSWER DECIDES: whether runtime params stay an OPT-IN codegen variant (separate artifact per
robot; zero cost when off) or can become the DEFAULT (one artifact, pay the table-read everywhere).

Usage: compare_runtime_param_ab.py --dir <results/runtime_params_ab>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANTS = ["baked", "runtime_inertia", "runtime_transform", "runtime_joint_dynamics"]
# Below this, a delta is indistinguishable from run-to-run jitter on this box. Anything inside the band
# is reported as a WASH, not as a win or a loss -- quoting a 0.4% "regression" as real is how noise gets
# promoted to a finding.
NOISE_FLOOR_PCT = 1.0


def _algo_us(path: Path) -> dict[str, float]:
    """algo -> compute-only batch microseconds, from one grid run.py JSON."""
    try:
        blob = json.loads(path.read_text())
    except Exception:
        return {}
    out: dict[str, float] = {}

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, dict):
                    for field in ("batch_256_compute_only_us", "compute_only_us", "us"):
                        if isinstance(v.get(field), (int, float)):
                            out.setdefault(k, float(v[field]))
                            break
                    walk(v)
                elif isinstance(v, list):
                    for x in v:
                        walk(x)
    walk(blob)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()

    cells: dict[str, dict[str, dict[str, float]]] = {}
    for f in sorted(args.dir.glob("*.json")):
        for v in sorted(VARIANTS, key=len, reverse=True):   # longest first: runtime_joint_dynamics
            if f.stem.endswith("_" + v):
                cells.setdefault(f.stem[: -len(v) - 1], {})[v] = _algo_us(f)
                break

    if not cells:
        print("  !! no runtime-param A/B JSONs found in", args.dir)
        return

    for cell, byvar in sorted(cells.items()):
        base = byvar.get("baked") or {}
        if not base:
            print(f"\n  {cell}: !! no BAKED baseline -- cannot A/B")
            continue
        print(f"\n  === {cell} (compute-only µs; baked = 1.00x baseline) ===")
        print(f"  {'variant':<24} {'algos':>6} {'geomean':>9} {'worst algo':>11}  worst")
        for v in VARIANTS[1:]:
            cur = byvar.get(v)
            if not cur:
                print(f"  {v:<24} {'-':>6} {'MISSING':>9}")
                continue
            ratios = [(cur[a] / base[a], a) for a in base
                      if a in cur and base[a] > 0 and cur[a] > 0]
            if not ratios:
                print(f"  {v:<24} {'-':>6} {'NO OVERLAP':>9}")
                continue
            geo = 1.0
            for r, _ in ratios:
                geo *= r
            geo **= 1.0 / len(ratios)
            worst_r, worst_a = max(ratios)
            tag = "WASH" if abs(geo - 1.0) * 100 < NOISE_FLOOR_PCT else ("COSTS" if geo > 1 else "FASTER?")
            print(f"  {v:<24} {len(ratios):>6} {geo:>8.3f}x {worst_r:>10.3f}x  {worst_a}  [{tag}]")

    print(f"\n  (WASH = within the ±{NOISE_FLOOR_PCT:.0f}% noise floor; do NOT quote a wash as a regression.)")
    print("  A variant that is a WASH can safely become the DEFAULT. One that COSTS stays OPT-IN.")


if __name__ == "__main__":
    main()

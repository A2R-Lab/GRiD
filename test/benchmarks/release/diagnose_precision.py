"""Validation-only matched-input fp32/fp64 replay; NEVER exports timings.

python -m test.benchmarks.release.diagnose_precision --robot iiwa14 \
  --output /tmp/grid-precision-replay

The release collector remains fp32. This diagnostic keeps its inputs and gate
fixed, changing only GRiD arithmetic precision to test numerical sensitivity.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .protocol import ROBOTS, agreement, digest, leaves, source_fingerprints, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", choices=list(ROBOTS), required=True)
    ap.add_argument("--operations", nargs="+", default=["forward_dynamics_gradient", "fdsva_so"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--dtypes", nargs="+", choices=["float32", "float64"], default=["float32", "float64"])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from .fixtures import Fixture
    from .grid_adapter import GridAdapter
    f = Fixture(args.robot, args.batch)
    inputs = args.output / "inputs.npz"
    np.savez(inputs, q=f.q, v=f.v, a=f.a, u=f.u)
    report = {"purpose": "precision diagnostic; no timing", "robot": args.robot,
              "fixture": f.metadata, "input_sha256": digest(inputs),
              "sources": source_fingerprints(), "results": []}
    for op in args.operations:
        expected = f.expected(op, args.batch)
        for dtype in args.dtypes:
            print(f"{args.robot} {op} {dtype}: preparing", flush=True)
            adapter = GridAdapter("grid_numpy", op, f, args.batch, args.output, dtype=dtype)
            adapter.prepare(args.batch)
            actual = adapter.normalize(adapter.host())
            check = agreement(actual, expected)
            arrays = args.output / f"{op}-{dtype}.npz"
            np.savez(arrays, **{f"actual_{i}": x for i, x in enumerate(leaves(actual))},
                     **{f"oracle_{i}": x for i, x in enumerate(leaves(expected))})
            entry = {"operation": op, "dtype": dtype, "agreement": check,
                     "adapter": adapter.metadata, "outputs_sha256": digest(arrays)}
            report["results"].append(entry)
            write_json(args.output / "report.json", report)
            print(check, flush=True)


if __name__ == "__main__":
    main()

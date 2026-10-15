#!/usr/bin/env python3
"""Generate the inverse_dynamics-only GRiD header used by the CUDA examples.

This is the literal one-line codegen invocation the examples document, wrapped in
a tiny CLI. Run from the repo root:

    .venv/bin/python examples/cuda/gen_iiwa14_header.py \
        --output examples/cuda/grid.cuh
"""
from __future__ import annotations

import argparse
from pathlib import Path


from URDFParser import URDFParser
from grid_codegen import GRiDCodeGenerator


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate iiwa14 inverse_dynamics header.")
    ap.add_argument("--output", default="grid.cuh", help="Path for the generated header.")
    args = ap.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    robot = URDFParser().parse(str(Path(__file__).resolve().parents[2] / "grid_assets" / "robots" / "iiwa14.urdf"), floating_base=False)
    # The one line that matters: restrict codegen to inverse_dynamics.
    GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
        algorithm_list=["inverse_dynamics"], output_path=str(out)
    )
    print(f"Wrote {out}  (robot={robot.name}, DOF={robot.get_num_joints()})")


if __name__ == "__main__":
    main()

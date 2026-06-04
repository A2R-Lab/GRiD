#!/usr/bin/env python3
"""Quickstart: generate GRiD CUDA code for the Unitree Go2 quadruped (floating base).

Requires: pip install robot_descriptions  (included in dev dependencies)

Run:
    python examples/codegen/generate_go2_floating.py --output /tmp/grid_go2.cuh

Generates a floating-base dynamics header by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from robot_descriptions import go2_description

from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator

URDF_PATH = go2_description.URDF_PATH


def main():
    parser_args = argparse.ArgumentParser(description="Generate floating-base Go2 GRiD CUDA code.")
    parser_args.add_argument("--output", default="grid.cuh", help="Path for the generated CUDA header.")
    parser_args.add_argument(
        "--profile",
        default="dynamics",
        choices=[
            "dynamics",
            "dynamics-core",
            "dynamics-gradients",
            "kinematics",
            "kinematics-derivatives",
            "all",
        ],
        help="Generated code profile. Use kinematics-derivatives or all for floating end-effector-pose gradients/Hessians.",
    )
    args = parser_args.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing URDF: {URDF_PATH}")
    parser = URDFParser()
    robot = parser.parse(URDF_PATH, floating_base=True)

    print(f"Robot: {robot.name}  |  DOF: {robot.get_num_joints()} + floating root")
    print(f"Generating floating-base GRiD CUDA code with profile '{args.profile}'...")

    codegen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid")
    codegen.gen_all_code(
        include_homogenous_transforms=args.profile in {"kinematics", "kinematics-derivatives", "all"},
        output_path=str(output_path),
        codegen_profile=args.profile,
    )

    print(f"Done: {output_path} written.")
    print()
    print("Next steps:")
    print("  1. Compile a CUDA program against the generated header")
    print("  2. Use --profile kinematics-derivatives to include floating end-effector-pose derivatives")
    print("  3. Use test/benchmarks/baselines/grid/run.py for benchmark slices")


if __name__ == "__main__":
    main()

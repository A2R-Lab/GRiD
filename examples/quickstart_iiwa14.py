#!/usr/bin/env python3
"""Quickstart: generate GRiD CUDA code for the KUKA iiwa14 arm (fixed base).

Requires: pip install robot_descriptions  (included in dev dependencies)

Run:
    python examples/quickstart_iiwa14.py --output /tmp/grid_iiwa14.cuh

Generates grid.cuh in the current directory unless --output is provided.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from robot_descriptions import iiwa14_description

from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator

URDF_PATH = iiwa14_description.URDF_PATH


def main():
    parser_args = argparse.ArgumentParser(description="Generate fixed-base iiwa14 GRiD CUDA code.")
    parser_args.add_argument("--output", default="grid.cuh", help="Path for the generated CUDA header.")
    args = parser_args.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing URDF: {URDF_PATH}")
    parser = URDFParser()
    robot = parser.parse(URDF_PATH, floating_base=False)

    print(f"Robot: {robot.name}  |  DOF: {robot.get_num_joints()}")
    print("Generating fixed-base GRiD CUDA code with retained end-effector kinematics...")

    codegen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid")
    codegen.gen_all_code(
        include_homogenous_transforms=True,
        fixed_target_name="iiwa_joint_ee",
        output_path=str(output_path),
    )

    print(f"Done: {output_path} written.")
    print()
    print("Next steps:")
    print("  1. Compile a CUDA program against the generated header")
    print("  2. Use examples/print_grid.py to compile and run the built-in print kernel")
    print("  3. Use test/cuda_equivalents/run_staged_cuda_checks.py for CUDA equivalence")


if __name__ == "__main__":
    main()

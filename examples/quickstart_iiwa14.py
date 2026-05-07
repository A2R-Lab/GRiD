#!/usr/bin/env python3
"""Quickstart: generate GRiD CUDA code for the KUKA iiwa14 arm (fixed base).

Requires: pip install robot_descriptions  (included in dev dependencies)

Run:
    python examples/quickstart_iiwa14.py

Generates grid.cuh in the current directory.
"""
from __future__ import annotations

from robot_descriptions import iiwa14_description

from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator

URDF_PATH = iiwa14_description.URDF_PATH

print(f"Parsing URDF: {URDF_PATH}")
parser = URDFParser()
robot = parser.parse(URDF_PATH, floating_base=False)

print(f"Robot: {robot.name}  |  DOF: {robot.get_num_joints()}")
print("Generating GRiD CUDA code...")

codegen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid")
codegen.gen_all_code(
    include_homogenous_transforms=True,
    fixed_target_name="iiwa_joint_ee",
    output_path="grid.cuh",
)

print("Done — grid.cuh written to current directory.")
print()
print("Next steps:")
print("  1. Compile a CUDA program against grid.cuh")
print("  2. See examples/print_grid.py to compile and run the built-in print kernel")
print("  3. See test/benchmarks/ for performance benchmarking")

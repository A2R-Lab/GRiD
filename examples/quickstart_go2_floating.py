#!/usr/bin/env python3
"""Quickstart: generate GRiD CUDA code for the Unitree Go2 quadruped (floating base).

Requires: pip install robot_descriptions  (included in dev dependencies)

Run:
    python examples/quickstart_go2_floating.py

Generates grid.cuh in the current directory.
"""
from __future__ import annotations

from robot_descriptions import go2_description

from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator

URDF_PATH = go2_description.URDF_PATH

print(f"Parsing URDF: {URDF_PATH}")
parser = URDFParser()
robot = parser.parse(URDF_PATH, floating_base=True)

print(f"Robot: {robot.name}  |  DOF: {robot.get_num_joints()} + 6 (floating base)")
print("Generating GRiD CUDA code (floating base — homogeneous transforms excluded)...")

codegen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid")
codegen.gen_all_code(
    include_homogenous_transforms=False,
    output_path="grid.cuh",
)

print("Done — grid.cuh written to current directory.")
print()
print("Next steps:")
print("  1. Compile a CUDA program against grid.cuh")
print("  2. See test/benchmarks/ for performance benchmarking")
print("  3. Try examples/quickstart_iiwa14.py for a fixed-base example")

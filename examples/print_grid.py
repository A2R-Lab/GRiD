#!/usr/bin/env python3
"""Compile and run the printGRiD CUDA executable to display generated kernel outputs.

Usage:
    python examples/print_grid.py PATH_TO_URDF [-n NAMESPACE] [-d] [-f]
    python examples/print_grid.py           # if grid.cuh already exists
"""
from __future__ import annotations

import subprocess
import sys

from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator
from GRiDCodeGenerator.cli import parseInputs, validateRobot


def main():
    inputs = parseInputs(NO_ARG_OPTION=True)
    if inputs is not None:
        URDF_PATH, DEBUG_MODE, FILE_NAMESPACE_NAME, FLOATING_BASE, FIXED_TARGET_NAMES = inputs
        parser = URDFParser()
        robot = parser.parse(URDF_PATH, floating_base=FLOATING_BASE)

        validateRobot(robot, NO_ARG_OPTION=True)

        print("-----------------")
        print("Generating GRiD.cuh")
        print("-----------------")
        codegen = GRiDCodeGenerator(robot, DEBUG_MODE, True, FILE_NAMESPACE=FILE_NAMESPACE_NAME)
        include_homogenous_transforms = not FLOATING_BASE
        codegen.gen_all_code(
            include_homogenous_transforms=include_homogenous_transforms,
            fixed_target_name=FIXED_TARGET_NAMES,
        )
        print("New code generated and saved to grid.cuh!")

    print("-----------------")
    print("Compiling printGRiD")
    print("-----------------")
    result = subprocess.run(
        ["nvcc", "-std=c++11", "-o", "printGRiD.exe", "printGRiD.cu",
         "-gencode", "arch=compute_120,code=sm_120"],
        capture_output=True, text=True,
    )
    if result.stderr:
        print("Compilation errors follow:")
        print(result.stderr)
        sys.exit(1)

    print("-----------------")
    print("Running printGRiD")
    print("-----------------")
    result = subprocess.run(["./printGRiD.exe"], capture_output=True, text=True)
    if result.stderr:
        print("Runtime errors follow:")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)


if __name__ == "__main__":
    main()

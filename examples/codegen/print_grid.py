#!/usr/bin/env python3
"""Compile and run the printGRiD CUDA executable to display generated kernel outputs.

Usage:
    python examples/codegen/print_grid.py PATH_TO_URDF [-n NAMESPACE] [-d] [-f]
    python examples/codegen/print_grid.py           # if grid.cuh already exists
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator
from GRiDCodeGenerator.cli import parseInputs, validateRobot


def detect_cuda_arch() -> str:
    env_arch = os.environ.get("GRID_CUDA_ARCH")
    if env_arch:
        return env_arch.replace("sm_", "").replace("compute_", "").replace(".", "")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                compute_cap = line.strip()
                if compute_cap:
                    return compute_cap.replace(".", "")

    return "86"


def main():
    inputs = parseInputs(NO_ARG_OPTION=True)
    arch = detect_cuda_arch()

    with tempfile.TemporaryDirectory(prefix="grid_print_") as tmpdir:
        build_dir = Path(tmpdir)
        build_header = build_dir / "grid.cuh"

        if inputs is not None:
            URDF_PATH, DEBUG_MODE, FILE_NAMESPACE_NAME, FLOATING_BASE, FIXED_TARGET_NAMES, _, _ = inputs
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
                output_path=str(build_header),
            )
            print(f"New code generated in temporary build directory: {build_header}")
        else:
            grid_header = Path("grid.cuh").resolve()
            if not grid_header.exists():
                print("grid.cuh does not exist. Generate it first or pass a URDF path.")
                sys.exit(1)
            shutil.copyfile(grid_header, build_header)

        shutil.copyfile(Path(__file__).resolve().parent / "printGRiD.cu", build_dir / "printGRiD.cu")

        print("-----------------")
        print("Compiling printGRiD")
        print("-----------------")
        result = subprocess.run(
            [
                "nvcc",
                "-std=c++11",
                "-o",
                "printGRiD.exe",
                "printGRiD.cu",
                "-gencode",
                f"arch=compute_{arch},code=sm_{arch}",
            ],
            cwd=build_dir,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            print("Compilation failed.")
            sys.exit(result.returncode)

        print("-----------------")
        print("Running printGRiD")
        print("-----------------")
        result = subprocess.run(
            ["./printGRiD.exe"],
            cwd=build_dir,
            capture_output=True,
            text=True,
        )
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            print("Runtime failed.")
            sys.exit(result.returncode)

        print(result.stdout)


if __name__ == "__main__":
    main()

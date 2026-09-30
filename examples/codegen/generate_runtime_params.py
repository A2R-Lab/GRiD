#!/usr/bin/env python3
"""Runtime-mutable physics params codegen (hardware co-design / sysID / domain randomization).

By default GRiD bakes both the sparsity PATTERN and the literal constant VALUES of a
robot's inertial parameters and fixed joint transforms into the generated code. The
runtime-param variants keep the (perf-critical) sparsity pattern baked but read the
nonzero VALUES from a mutable on-device table, so you can change link masses/inertias
(`runtime_inertia`) or fixed joint origins (`runtime_transform`) with NO recompile.
The baked default path stays byte-identical until a mutator is called.

  - runtime_inertia   -> emits a mutable inertia table + `set_inertia_params(...)`.
  - runtime_transform -> emits a mutable Xfixed table + `set_transform_params(...)`.

From Python these are reachable on the grid-rbd handle (numpy / jax / torch), e.g.:

    import grid_rbd
    h = grid_rbd.register_robot("iiwa14", urdf_path="iiwa.urdf", runtime_inertia=True)
    h.set_inertia_params(perturbed)      # domain randomization, no rebuild
    qdd = h.forward_dynamics(q, qd, u)   # uses the poked values

Requires: pip install robot_descriptions  (dev dependency)

Run:
    python examples/codegen/generate_runtime_params.py --output /tmp/grid_iiwa14_rt.cuh
"""
from __future__ import annotations

import argparse
from pathlib import Path

from robot_descriptions import iiwa14_description

from URDFParser import URDFParser
from grid_codegen import GRiDCodeGenerator

URDF_PATH = iiwa14_description.URDF_PATH


def main():
    ap = argparse.ArgumentParser(description="Generate iiwa14 GRiD code with runtime-mutable params.")
    ap.add_argument("--output", default="grid.cuh", help="Path for the generated CUDA header.")
    ap.add_argument("--inertia", action="store_true", help="Emit the runtime-mutable inertia table.")
    ap.add_argument("--transform", action="store_true", help="Emit the runtime-mutable Xfixed table.")
    args = ap.parse_args()
    if not (args.inertia or args.transform):
        args.inertia = args.transform = True  # default: demonstrate both
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing URDF: {URDF_PATH}")
    parser = URDFParser()
    robot = parser.parse(URDF_PATH, floating_base=False)

    print(f"Generating GRiD CUDA code (runtime_inertia={args.inertia}, "
          f"runtime_transform={args.transform})...")
    codegen = GRiDCodeGenerator(robot, FILE_NAMESPACE="grid")
    codegen.gen_all_code(output_path=str(output_path),
                         runtime_inertia=args.inertia,
                         runtime_transform=args.transform)

    print(f"Done: {output_path} written.")
    if args.inertia:
        print("  -> device entry: set_inertia_params(...)   (mutate inertias, no recompile)")
    if args.transform:
        print("  -> device entry: set_transform_params(...) (mutate fixed joint origins)")
    print("Validate on-GPU with: pytest test/python_wrappers/test_runtime_inertia.py "
          "test/python_wrappers/test_runtime_transform.py")


if __name__ == "__main__":
    main()

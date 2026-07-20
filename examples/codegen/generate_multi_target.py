#!/usr/bin/env python3
"""Multi-target kinematics codegen: batched `multi_target_position{,_gradient}` for iiwa14.

`multi_target_batch` bakes a set of end-effector targets — each an anchor joint id
plus a body-frame offset — into the generated header, emitting batched
`multi_target_position` and `multi_target_position_gradient` kernels + host wrappers.
One compiled robot then evaluates FK (and its Jacobian) at every baked target in a
single launch. Targets are given as a list of {"anchor_jid": int, "offset": (x,y,z)}.

Requires: pip install robot_descriptions  (dev dependency)

Run:
    python examples/codegen/generate_multi_target.py --output /tmp/grid_iiwa14_mt.cuh
"""
from __future__ import annotations

import argparse
from pathlib import Path

from robot_descriptions import iiwa14_description

from URDFParser import URDFParser
from grid_codegen import GRiDCodeGenerator

URDF_PATH = iiwa14_description.URDF_PATH


def build_batch(robot):
    """One zero-offset control target per leaf, plus one nonzero-offset target per leaf."""
    leaves = robot.get_leaf_nodes()
    targets = [{"anchor_jid": int(jid), "offset": (0.0, 0.0, 0.0)} for jid in leaves]
    for e, jid in enumerate(leaves):
        targets.append({"anchor_jid": int(jid), "offset": (0.03 + 0.01 * e, -0.02, 0.05)})
    return targets


def main():
    ap = argparse.ArgumentParser(description="Generate iiwa14 GRiD code with a multi-target batch.")
    ap.add_argument("--output", default="grid.cuh", help="Path for the generated CUDA header.")
    args = ap.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing URDF: {URDF_PATH}")
    parser = URDFParser()
    robot = parser.parse(URDF_PATH, floating_base=False)

    targets = build_batch(robot)
    print(f"Multi-target batch: {len(targets)} targets across {len(robot.get_leaf_nodes())} leaf frame(s).")

    print("Generating GRiD CUDA code with multi_target_position{,_gradient} kernels...")
    codegen = GRiDCodeGenerator(robot, FILE_NAMESPACE="grid")
    codegen.gen_all_code(output_path=str(output_path), multi_target_batch=targets)

    print(f"Done: {output_path} written (NUM_MULTI_TARGETS = {len(targets)}).")
    print("Validate on-GPU with: pytest test/cuda_equivalents/test_cuda_multi_target_position.py")


if __name__ == "__main__":
    main()

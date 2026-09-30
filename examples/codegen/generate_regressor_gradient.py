#!/usr/bin/env python3
"""Generate the joint-torque-regressor STATE derivative (dY/dx) surface.

`tau = Y(q, qd, qdd) . pi` is exactly linear in the per-link inertial
parameters pi, and `inverse_dynamics_regressor_gradient` emits the analytic
STATE derivative of that regressor:

    dY_dx[c] . pi == d(tau)/dx[:, c]        (the "du x pi" identity)

i.e. the mixed second derivative d(inverse_dynamics_gradient)/dpi — the
missing block for differentiable system identification (how the TORQUE
GRADIENTS an optimizer consumes change with the inertial parameters).

The emitted surface (all in grid.cuh):
    inverse_dynamics_regressor_gradient_inner / _device      (composable)
    inverse_dynamics_regressor_gradient_kernel[_single_timing]
    inverse_dynamics_regressor_gradient(hd_data, ...)        (host launcher)
Output: hd_data->d_dY_dx / h_dY_dx, per timestep 2*NV*NV*10*NUM_BODIES floats
(the dq block then the dqd block; each direction c an NV x 10*NB row-major
matrix — matching `RBDReference.inverse_dynamics_regressor_gradient`).

There is deliberately NO Python-binding op for this yet (the output is large:
1.9 MB fp32 per timestep on g1) — the CUDA host surface is the consumer story
today, and `test/cuda_equivalents/test_cuda_regressor_gradient.py` is the
end-to-end usage reference (numpy-oracle comparison + the pi-identity gate).

Requires: pip install robot_descriptions  (included in dev dependencies)

Run:
    python examples/codegen/generate_regressor_gradient.py --output /tmp/grid_dydx.cuh
"""
from __future__ import annotations

import argparse
from pathlib import Path

from robot_descriptions import iiwa14_description

from URDFParser import URDFParser
from grid_codegen import GRiDCodeGenerator

URDF_PATH = iiwa14_description.URDF_PATH


def main():
    parser_args = argparse.ArgumentParser(description="Generate the dY/dx (regressor state-derivative) GRiD CUDA code.")
    parser_args.add_argument("--output", default="grid.cuh", help="Path for the generated CUDA header.")
    args = parser_args.parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Parsing URDF: {URDF_PATH}")
    robot = URDFParser().parse(URDF_PATH, floating_base=False)
    nv, nb = robot.get_num_vel(), robot.get_num_bodies()
    print(f"Robot: {robot.name}  |  DOF: {nv}  |  bodies: {nb}")
    print(f"dY/dx output per timestep: 2*{nv}*{nv}*10*{nb} = {2 * nv * nv * 10 * nb} floats")

    # The subset request pulls the transitive deps automatically
    # (inverse_dynamics + inverse_dynamics_gradient, whose dv/da staging the
    # dY/dx walk reads). Use "all" instead to get the whole library.
    codegen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid")
    codegen.gen_all_code(
        include_homogenous_transforms=True,
        output_path=str(output_path),
        algorithm_list=["inverse_dynamics_regressor_gradient"],
    )

    print(f"Done: {output_path} written.")
    print()
    print("Next steps:")
    print("  1. Call grid::inverse_dynamics_regressor_gradient(hd_data, ...) and read hd_data->h_dY_dx")
    print("  2. Check the identity: dY_dx[c] @ pi == d(tau)/dx[:, c] against inverse_dynamics_gradient")
    print("  3. See test/cuda_equivalents/test_cuda_regressor_gradient.py for the full oracle-checked flow")


if __name__ == "__main__":
    main()

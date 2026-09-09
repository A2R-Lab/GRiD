#!/usr/bin/env python3
"""Print CPU reference values for all RBD algorithms for a given URDF.

Useful for debugging and validating GRiD CUDA output against ground truth.

Usage:
    python examples/codegen/print_reference_values.py PATH_TO_URDF [-t FIXED_TARGET_NAMES] [-n NAMESPACE] [-d] [-f]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from URDFParser import URDFParser
from RBDReference import RBDReference
from grid_codegen import GRiDCodeGenerator
from grid_codegen.cli import parseInputs, validateRobot

# The repo's test/ package must shadow the stdlib `test` package regardless of
# how this script is invoked (test/helpers.py is not part of the installed dist).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from test.helpers import initializeValues


def main():
    args = parseInputs()  # argparse Namespace (2026-09-08; was a drift-prone tuple)

    parser = URDFParser()
    robot = parser.parse(args.urdf_path, floating_base=args.floating_base)

    validateRobot(robot)

    reference = RBDReference(robot)
    q, qd, u, n = initializeValues(robot, MATCH_CPP_RANDOM=True)

    print("q\n", q)
    print("qd\n", qd)
    print("u\n", u)

    (c, v, a, f) = reference.inverse_dynamics(q, qd)
    print("c\n", c)

    Minv = reference.minv(q)
    print("Minv\n", Minv)

    qdd = np.matmul(Minv, (u - c))
    print("qdd\n", qdd)

    if not args.floating_base:
        qdd_aba = reference.aba(q, qd, u)
        print("aba\n", qdd_aba)

        crba = reference.crba(q)
        print("crba\n", crba)

    dc_du = reference.inverse_dynamics_gradient(q, qd, qdd)
    print("dc/dq with qdd\n", dc_du)
    print("dc/dqd with qdd\n", dc_du)

    df_du = np.matmul(-Minv, dc_du)
    print("df/dq\n", df_du)
    print("df/dqd\n", df_du)

    dqdd_dq, dqdd_dqd = reference.forward_dynamics_gradient(q, qd, u)
    print("dqdd_dq")
    print(dqdd_dq)
    print("dqdd_dqd")
    print(dqdd_dqd)

    if not args.floating_base:
        ee_pos = reference.end_effector_pose(q)
        print("end_effector_pose\n", ee_pos)

        if args.fixed_target_names != "":
            ee_pos2 = reference.end_effector_pose(q, ee_joint_names=args.fixed_target_names)
            print("end_effector_pose-" + args.fixed_target_names + "\n", ee_pos2)

        dee_pos = reference.end_effector_pose_gradient(q)
        print("end_effector_pose_gradient\n", dee_pos)

        if args.fixed_target_names != "":
            dee_pos2 = reference.end_effector_pose_gradient(q, ee_joint_names=args.fixed_target_names)
            print("end_effector_pose_gradient-" + args.fixed_target_names + "\n", dee_pos2)

        d2ee_pos = reference.end_effector_pose_hessian(q)
        print("end_effector_pose_hessian\n", d2ee_pos)

    # Second-order inverse dynamics — auto-dispatched (body-frame for
    # fixed-base, world-frame for floating-base). Both variants are
    # mathematically equivalent; the dispatcher picks the faster one
    # for the robot's base type.
    d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq = reference.idsva_so(
        q, qd, np.zeros(len(qd))
    )
    print(f'\nd2tau_dq:\n{d2tau_dq}')
    print(f'\nd2tau_dqd:\n{d2tau_dqd}')
    print(f'\nd2tau_cross:\n{d2tau_cross}')
    print(f'\ndM_dq:\n{dM_dq}')

    # Second-order forward dynamics
    fdsva_so_out = reference.fdsva_so(q, qd, u)
    print(f'\nfdsva_so (rank-3 partials of qdd):\n{fdsva_so_out}')

    if args.debug:
        print("-------------------")
        print("printing intermediate outputs from refactorings")
        print("-------------------")
        codegen = GRiDCodeGenerator(robot, args.debug, FILE_NAMESPACE=args.namespace)
        (c, v, a, f) = codegen.test_rnea(q, qd)
        print("v\n", v)
        print("a\n", a)
        print("f\n", f)
        print("c\n", c)

        Minv = codegen.test_minv(q)
        print("Minv\n", Minv)

        umc = u - c
        print("u-c\n", umc)

        qdd = np.matmul(Minv, umc)
        print("qdd\n", qdd)

        dc_du = codegen.test_rnea_grad(q, qd, qdd)
        print("dc/dq with qdd\n", dc_du[:, :n])
        print("dc/dqd with qdd\n", dc_du[:, n:])

        df_du = np.matmul(-Minv, dc_du)
        print("df/dq\n", df_du[:, :n])
        print("df/dqd\n", df_du[:, n:])


if __name__ == "__main__":
    main()

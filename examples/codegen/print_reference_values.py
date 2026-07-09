#!/usr/bin/env python3
"""Print CPU reference values for all RBD algorithms for a given URDF.

Useful for debugging and validating GRiD CUDA output against ground truth.

Usage:
    python examples/codegen/print_reference_values.py PATH_TO_URDF [-t FIXED_TARGET_NAMES] [-n NAMESPACE] [-d] [-f]
"""
from __future__ import annotations

import numpy as np

from URDFParser import URDFParser
from RBDReference import RBDReference
from GRiDCodeGenerator import GRiDCodeGenerator
from GRiDCodeGenerator.cli import parseInputs, validateRobot
from test.helpers import initializeValues


def main():
    URDF_PATH, DEBUG_MODE, FILE_NAMESPACE_NAME, FLOATING_BASE, FIXED_TARGET_NAMES, _, _ = parseInputs()

    parser = URDFParser()
    robot = parser.parse(URDF_PATH, floating_base=FLOATING_BASE)

    validateRobot(robot)

    reference = RBDReference(robot)
    q, qd, u, n = initializeValues(robot, MATCH_CPP_RANDOM=True)

    print("q\n", q)
    print("qd\n", qd)
    print("u\n", u)

    (c, v, a, f) = reference.rnea(q, qd)
    print("c\n", c)

    Minv = reference.minv(q)
    print("Minv\n", Minv)

    qdd = np.matmul(Minv, (u - c))
    print("qdd\n", qdd)

    if not FLOATING_BASE:
        qdd_aba = reference.aba(q, qd, u)
        print("aba\n", qdd_aba)

        crba = reference.crba(q)
        print("crba\n", crba)

    dc_du = reference.rnea_grad(q, qd, qdd)
    print("dc/dq with qdd\n", dc_du)
    print("dc/dqd with qdd\n", dc_du)

    df_du = np.matmul(-Minv, dc_du)
    print("df/dq\n", df_du)
    print("df/dqd\n", df_du)

    dqdd_dq, dqdd_dqd = reference.forward_dynamics_grad(q, qd, c)
    print("dqdd_dq")
    print(dqdd_dq)
    print("dqdd_dqd")
    print(dqdd_dqd)

    if not FLOATING_BASE:
        ee_pos = reference.end_effector_pose(q)
        print("end_effector_pose\n", ee_pos)

        if FIXED_TARGET_NAMES != "":
            ee_pos2 = reference.end_effector_pose(q, ee_joint_names=FIXED_TARGET_NAMES)
            print("end_effector_pose-" + FIXED_TARGET_NAMES + "\n", ee_pos2)

        dee_pos = reference.end_effector_pose_gradient(q)
        print("end_effector_pose_gradient\n", dee_pos)

        if FIXED_TARGET_NAMES != "":
            dee_pos2 = reference.end_effector_pose_gradient(q, ee_joint_names=FIXED_TARGET_NAMES)
            print("end_effector_pose_gradient-" + FIXED_TARGET_NAMES + "\n", dee_pos2)

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

    if DEBUG_MODE:
        print("-------------------")
        print("printing intermediate outputs from refactorings")
        print("-------------------")
        codegen = GRiDCodeGenerator(robot, DEBUG_MODE, FILE_NAMESPACE=FILE_NAMESPACE_NAME)
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

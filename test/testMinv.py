#! /usr/bin/env python3

from URDFParser import URDFParser
from RBDReference import RBDReference
from GRiDCodeGenerator import GRiDCodeGenerator
from util import parseInputs, printUsage, validateRobot, initializeValues, printErr
import numpy as np

"""
This script runs the Minv algorithm from RNEA reference and outputs the resulting
inverse of the mass matrix, Minv.
"""

def test_direct_minv():
    URDF_PATH, DEBUG_MODE, FLOATING_BASE = parseInputs()

    parser = URDFParser()
    robot = parser.parse(URDF_PATH, floating_base = FLOATING_BASE, using_quaternion = True)

    validateRobot(robot)

    reference = RBDReference(robot)
    q, qd, qdd, _ = initializeValues(robot, MATCH_CPP_RANDOM = False)

    qdd = np.zeros(len(qdd))

    for i in range(len(qdd)):
        qdd[i] = 0


    print("T q[NUM_DOF+FLOATING_BASE] = {" + ", ".join(str(x) for x in q) + "};")
    print("T qd[NUM_DOF] = {" + ", ".join(str(x) for x in qd) + "};")
    print("T qdd[NUM_DOF] = {" + ", ".join(str(x) for x in qdd) + "};")


    minv = reference.minv(q, output_dense=False)
    print(f'\nMinv:\n{minv}')

if __name__ == "__main__":
    test_direct_minv()
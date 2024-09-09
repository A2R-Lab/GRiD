#! /usr/bin/env python3

from URDFParser import URDFParser
from RBDReference import RBDReference
from GRiDCodeGenerator import GRiDCodeGenerator
from util import parseInputs, printUsage, validateRobot, initializeValues, printErr
import numpy as np

"""
Script allowing for easy comparison of our RNEA implementation with spatial_v2_extended (ID.m).
Usage: python compare_rnea_fb_spatial.py [PATH_TO_URDF_FILE] [-f]

-f option => use floating base

The script prints randomized input entries for q, qd and qdd in a format that allows them to be easily transferred to MATLAB.
We generate quaternions when using floating base.

We also print the results of our RNEA algorithm from RBDReference.

- Kwamena & Naren
"""

def test_with_spatial():
    URDF_PATH, DEBUG_MODE, FLOATING_BASE = parseInputs()

    parser = URDFParser()
    robot = parser.parse(URDF_PATH, floating_base = FLOATING_BASE, using_quaternion = True)

    validateRobot(robot)

    reference = RBDReference(robot)
    q, qd, qdd, _ = initializeValues(robot, MATCH_CPP_RANDOM = False)

    qdd = np.zeros(len(qdd))

    for i in range(len(qdd)):
        qdd[i] = 0

    # if comparing cuda
    #if not FLOATING_BASE:
    print("T q[NUM_DOF+FLOATING_BASE] = {" + ", ".join(str(x) for x in q) + "};")
    print("T qd[NUM_DOF] = {" + ", ".join(str(x) for x in qd) + "};")
    print("T qdd[NUM_DOF] = {" + ", ".join(str(x) for x in qdd) + "};")

    # if comparing spatial:
    # else:
    #     print("T q[NUM_JOINTS] = {", end="")

    #     # Getting the floating base position vals
    #     fb_q = [0] * 7
    #     # Get the quaternion
    #     fb_q[:4] = q[3:7]
    #     # Cycle the quaternion from wxyz to xyzw (for spatial)
    #     w = fb_q[0]
    #     fb_q[:3] = fb_q[1:4]
    #     fb_q[3] = w
    #     # Get the xyz for the floating base positioning
    #     fb_q[4:] = q[:3]

    #     # print("; ".join(str(x) for x in fb_q) + "]; ", end = "")
    #     # print("; ".join(str(x) for x in q[7:]), end = "}\n")

    
    #     print("qd = {[" + "; ".join(str(x) for x in qd[:6]) + "]; " + "; ".join(str(x) for x in qd[6:]) + "}")
    #     print("qdd = {[" + "; ".join(str(x) for x in qdd[:6]) + "]; " + "; ".join(str(x) for x in qdd[6:]) + "}")

    (c, v, a, f) = reference.rnea(q, qd, qdd)
    print('c\n{}\nv\n{}\na\n{}\nf\n{}'.format(c, v, a, f))

if __name__ == "__main__":
    test_with_spatial()


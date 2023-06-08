#!/usr/bin/python3
from URDFParser import URDFParser
from RBDReference import RBDReference
from GRiDCodeGenerator import GRiDCodeGenerator
from util import parseInputs, printUsage, validateRobot, initializeValues, printErr
import numpy as np

def main():
    URDF_PATH, DEBUG_MODE, FILE_NAMESPACE_NAME = parseInputs()

    parser = URDFParser()
    robot = parser.parse(URDF_PATH)

    validateRobot(robot)

    reference = RBDReference(robot)
    q, qd, u, n = initializeValues(robot, MATCH_CPP_RANDOM = True)

    print("q\n",q)
    print("qd\n",qd)
    print("u\n",u)

    print("\nNote: the variables u, q, and qd are hardcoded even though they are computed above\n")
    u = np.zeros(n)
    q =[1,2,3,1,2,3,1]
    qd = np.zeros(n)

    (c, v, a, f) = reference.rnea(q,qd)

    Minv = reference.minv(q)
    print("Minv\n", Minv)

    qdd = np.matmul(Minv,(u-c))
    print("qdd\n",qdd)

    crba=reference.crba(q,qd,u)
    print("crba\n",crba)

    qdd_aba = reference.aba(q,qd,u)
    print("aba\n",qdd_aba)
    
    dc_du = reference.rnea_grad(q, qd, qdd)
    print("dc/dq with qdd\n",dc_du[:,:n])
    print("dc/dqd with qdd\n",dc_du[:,n:])

    df_du = np.matmul(-Minv,dc_du)
    print("df/dq\n",df_du[:,:n])
    print("df/dqd\n",df_du[:,n:])

    if DEBUG_MODE:
        print("-------------------")
        print("printing intermediate outputs from refactorings")
        print("-------------------")
        codegen = GRiDCodeGenerator(robot, DEBUG_MODE, FILE_NAMESPACE = FILE_NAMESPACE_NAME)
        (c, v, a, f) = codegen.test_rnea(q,qd)
        print("v\n",v)
        print("a\n",a)
        print("f\n",f)
        print("c\n",c)
        
        Minv = codegen.test_minv(q)
        print("Minv\n",Minv)

        umc = u-c
        print("u-c\n",umc)

        qdd = np.matmul(Minv,umc)
        print("qdd\n",qdd)
        
        dc_du = codegen.test_rnea_grad(q, qd, qdd)
        print("dc/dq with qdd\n",dc_du[:,:n])
        print("dc/dqd with qdd\n",dc_du[:,n:])
        
        df_du = np.matmul(-Minv,dc_du)
        print("df/dq\n",df_du[:,:n])
        print("df/dqd\n",df_du[:,n:])

if __name__ == "__main__":
    main()
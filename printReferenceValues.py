#!/usr/bin/python3
from URDFParser import URDFParser
from RBDReference import RBDReference
from GRiDCodeGenerator import GRiDCodeGenerator
from util import parseInputs, printUsage, validateRobot, initializeValues, printErr
import numpy as np

def main():
    URDF_PATH, DEBUG_MODE, FILE_NAMESPACE_NAME, FLOATING_BASE, FIXED_TARGET_NAMES = parseInputs()

    parser = URDFParser()
    robot = parser.parse(URDF_PATH, floating_base=FLOATING_BASE)

    validateRobot(robot)

    reference = RBDReference(robot)
    q, qd, u, n = initializeValues(robot, MATCH_CPP_RANDOM = True)
    #for (idx, _) in enumerate(u): u[idx] = 0

    print("q\n",q)
    print("qd\n",qd)
    print("u\n",u)

    (c, v, a, f) = reference.rnea(q,qd)
    print("c\n",c)

    Minv = reference.minv(q)
    print("Minv\n", Minv)

    qdd = np.matmul(Minv,(u-c))
    print("qdd\n",qdd)

    if not FLOATING_BASE:
        qdd_aba = reference.aba(q,qd,u)
        print("aba\n",qdd_aba)

        crba=reference.crba(q)
        print("crba\n",crba)

    dc_du = reference.rnea_grad(q, qd, qdd)
    print("dc/dq with qdd\n",dc_du)
    print("dc/dqd with qdd\n",dc_du)

    df_du = np.matmul(-Minv,dc_du)
    print("df/dq\n",df_du)
    print("df/dqd\n",df_du)

    dqdd_dq, dqdd_dqd = reference.forward_dynamics_grad(q,qd,c)
    print("dqdd_dq")
    print(dqdd_dq)
    print("dqdd_dqd")
    print(dqdd_dqd)

    if not FLOATING_BASE:
        ee_pos = reference.end_effector_pose(q)
        print("eepos\n",ee_pos)
        
        if FIXED_TARGET_NAMES != "":
            ee_pos2 = reference.end_effector_pose(q, ee_joint_names=FIXED_TARGET_NAMES)
            print("eepos-" + FIXED_TARGET_NAMES + "\n",ee_pos2)

        dee_pos = reference.end_effector_pose_gradient(q)
        print("deepos\n",dee_pos)
        
        if FIXED_TARGET_NAMES != "":
            dee_pos2 = reference.end_effector_pose_gradient(q, ee_joint_names=FIXED_TARGET_NAMES)
            print("deepos-" + FIXED_TARGET_NAMES + "\n",dee_pos2)

        d2ee_pos = reference.end_effector_pose_hessian(q)
        print("d2eepos\n", d2ee_pos)

    if not FLOATING_BASE:
        d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq = reference.second_order_idsva_parallel(q,qd,np.zeros(len(qd)))
        print(f'\nd2tau_dq:\n{d2tau_dq}')
        print(f'\nd2tau_dqd:\n{d2tau_dqd}')
        print(f'\nd2tau_cross:\n{d2tau_cross}')
        print(f'\ndM_dq:\n{dM_dq}')

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
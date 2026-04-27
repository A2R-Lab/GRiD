#!/usr/bin/python3
from URDFParser import URDFParser
from RBDReference import RBDReference
from util import parseInputs, printUsage, validateRobot, initializeValues, printErr
import numpy as np
import copy

def finiteDiffGrad(q, diff_func, eps = 1e-3):
    pos = diff_func(q)
    npos = len(pos)
    nq = len(q)
    grads = np.zeros((npos,nq))
    for i in range(nq):
        q_p = copy.deepcopy(q)
        q_m = copy.deepcopy(q)
        q_p[i] += eps
        q_m[i] -= eps
        pos_p = diff_func(q_p)
        pos_m = diff_func(q_m)
        delta = pos_p - pos_m
        delta /= (2*eps)
        grads[:,i] = np.squeeze(delta)
    return grads

def finiteDiffHessian(q, diff_func, eps = 1e-3):
    pos = diff_func(q)
    npos = len(pos)
    nq = len(q)
    hessians = np.zeros((npos,nq,nq))
    for i in range(nq):
        for j in range(nq):
            q_pi = copy.deepcopy(q)
            q_mi = copy.deepcopy(q)
            q_pj = copy.deepcopy(q)
            q_mj = copy.deepcopy(q)
            q_pi[i] += eps
            q_mi[i] -= eps
            q_pj[j] += eps
            q_mj[j] -= eps
            pos_pi = diff_func(q_pi)
            pos_mi = diff_func(q_mi)
            pos_pj = diff_func(q_pj)
            pos_mj = diff_func(q_mj)
            delta = (pos_pj[:,i] - pos_mj[:,i]) + (pos_pi[:,j] - pos_mi[:,j])
            delta /= (4*eps)
            hessians[:,i,j] = np.squeeze(delta)
    return hessians

def main():
    URDF_PATH, DEBUG_MODE, FILE_NAMESPACE_NAME, FLOATING_BASE = parseInputs()

    parser = URDFParser()
    robot = parser.parse(URDF_PATH)

    validateRobot(robot)

    reference = RBDReference(robot)
    q, qd, u, n = initializeValues(robot)

    grads = reference.end_effector_pose_gradient(q)
    test_grads = finiteDiffGrad(q,lambda a : reference.end_effector_pose(a)[0])
    print("Finite Diff Grads")
    print(test_grads)
    print("Analytical Grads")
    print(grads)

    hessians = reference.end_effector_pose_hessian(q)
    test_hessians = finiteDiffHessian(q,lambda a : reference.end_effector_pose_gradient(a)[0])
    print("Finite Diff Hessian")
    print(test_hessians)
    print("Analytical Hessian")
    print(hessians)

if __name__ == "__main__":
    main()
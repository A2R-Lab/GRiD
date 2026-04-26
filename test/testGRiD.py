#!/usr/bin/python3
from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator
from RBDReference import RBDReference
from util import parseInputs, printUsage, validateRobot, initializeValues
import subprocess
import numpy as np
import re

DIFF = 1e-2 # maximum allowable difference per value in outputs

def testGRiD(URDF_PATH, FLOATING_BASE):
    """
    Tests all implemented algorithms in GRiD, and
    compares them with RBDReference. Returns a
    boolean that signifies whether all tests
    have been passed.
    """
    parser = URDFParser()
    robot = parser.parse(URDF_PATH, floating_base=FLOATING_BASE)
    q,qd,u,num_pos = initializeValues(robot, MATCH_CPP_RANDOM = True)

    nv = num_pos - 1 * FLOATING_BASE

    validateRobot(robot, NO_ARG_OPTION = True)

    codegen = GRiDCodeGenerator(robot, False, True, FILE_NAMESPACE = 'grid')
    print("-----------------")
    print("Generating GRiD.cuh")
    print("-----------------")
    if FLOATING_BASE: include_homogenous_transforms = False
    else: include_homogenous_transforms = True
    codegen.gen_all_code(include_homogenous_transforms = include_homogenous_transforms)
    print("New code generated and saved to grid.cuh!")

    if FLOATING_BASE: filename = 'TestGRiD/testGRiDFB.cu'
    else: filename = 'TestGRiD/testGRiD.cu'
    print("-----------------")
    print("Compiling testGRiD")
    print("-----------------")
    result = subprocess.run( \
        ["nvcc", "-o", "testGRiD.exe", filename], \
        capture_output=True, text=True \
    )
    if result.stderr:
        print("Compilation errors follow:")
        print(result.stderr)
        exit()

    print("-----------------")
    print("Running testGRiD")
    print("-----------------")
    result = subprocess.run(["./testGRiD.exe"], capture_output=True, text=True)
    if result.stderr:
        print("Runtime errors follow:")
        print(result.stderr)
        exit()

    # collect output
    grid_output = result.stdout.split('\n') # split output into separate strings
    grid_output = [data.strip() for data in grid_output if data]
    q = grid_output[0].strip().split(' ')
    q = np.array([float(i) for i in q])
    qd = np.array(grid_output[1].strip().split(' '))
    qd = np.array([float(i) for i in qd])
    u = np.array(grid_output[2].strip().split(' '))
    u = np.array([float(i) for i in u])
    grid_output_gradients = grid_output[len(grid_output)-5:] # rnea, & gradients
    grid_output = grid_output[3:] 

    # assign functions
    r = RBDReference(robot)
    passed_all = True


    # inverse dynamics gradient
    dcdu_ref = r.rnea_grad(q,qd,np.zeros(nv))
    dcdq_ref, dcdqd_ref = np.hsplit(dcdu_ref, [nv])
    dcdq_grid = grid_output[:nv]
    dcdqd_grid = grid_output[nv:2*nv]
    grid_output = grid_output[2*nv:]
    passed, differences, equal, sum_diff = compare_matrix(dcdq_ref, dcdq_grid)
    print(f'dc_dq: {equal}\n')
    if 'Failed' in equal: passed_all = False
    passed, differences, equal, sum_diff = compare_matrix(dcdqd_ref, dcdqd_grid)
    print(f'dc_dqd: {equal}\n')
    if 'Failed' in equal: passed_all = False


    # minv
    minv_ref = np.array(r.minv(q))
    minv_grid = grid_output[:nv]
    grid_output = grid_output[nv:]
    passed, differences, equal, sum_diff = compare_matrix(minv_ref, minv_grid, True)
    print(f'minv: {equal}\n')
    if 'Failed' in equal: passed_all = False


    # forward dynamics
    fd_ref = np.array(r.forward_dynamics(q,qd,u))
    qdd = fd_ref
    fd_grid = grid_output[0].strip().split(' ')
    grid_output = grid_output[1:]
    passed, differences, equal, sum_diff = compare_array(fd_ref,fd_grid)
    print(f'forward dynamics: {equal}\n')
    if 'Failed' in equal: passed_all = False
    

    # inverse dynamics
    rnea_ref = r.rnea(q,qd,u)[0]
    rnea_grid = grid_output[0].strip().split(' ')
    grid_output = grid_output[1:]
    passed, differences, equal, sum_diff = compare_array(rnea_ref,rnea_grid)
    print(f'inverse dynamics: {equal}\n')
    if 'Failed' in equal: passed_all = False


    # forward dynamics gradient
    (dfdq_ref, dfdqd_ref) = r.forward_dynamics_grad(q,qd,np.zeros_like(fd_ref))
    dfdq_grid = grid_output[:nv]
    dfdqd_grid = grid_output[nv:2*nv]
    grid_output = grid_output[2*nv:]
    passed, differences, equal, sum_diff = compare_matrix(dfdq_ref, dfdq_grid)
    print(f'df_dq: {equal}\n')
    if 'Failed' in equal: passed_all = False
    passed, differences, equal, sum_diff = compare_matrix(dfdqd_ref, dfdqd_grid)
    print(f'df_dqd: {equal}\n')
    if 'Failed' in equal: passed_all = False


    if not FLOATING_BASE:
        # eepos
        eepos_ref = r.end_effector_pose(q)
        for i in range(len(eepos_ref)): eepos_ref[i] = eepos_ref[i].flatten()
        eepos_ref = np.concatenate(eepos_ref).flatten().tolist()[0]
        eepos_grid = grid_output[0].strip().split(' ')
        eepos_grid = np.array([float(i) for i in eepos_grid])
        passed, differences, equal, sum_diff = compare_array(eepos_ref,eepos_grid)
        print(f'eepos: {equal}\n')
        if 'Failed' in equal: passed_all = False

        # eepos grad
        eeposgrad_ref = np.array(r.end_effector_pose_gradient(q))[0]
        eeposgrad_grid = grid_output[1:7]
        grid_output = grid_output[7:]
        passed, differences, equal, sum_diff = compare_matrix(eeposgrad_ref, eeposgrad_grid)
        print(f'eeposgrad: {equal}\n')
        if 'Failed' in equal: passed_all = False

        # aba
        aba_grid = grid_output[0].strip().split(' ')
        grid_output = grid_output[1:]
        passed, differences, equal, sum_diff = compare_array(fd_ref,aba_grid)
        print(f'aba: {equal}\n')
        if 'Failed' in equal: passed_all = False

        # crba
        crba_ref = np.array(r.crba(q))
        crba_grid = grid_output[:nv]
        grid_output = grid_output[nv:]
        passed, differences, equal, sum_diff = compare_matrix(crba_ref,crba_grid)
        print(f'crba: {equal}\n')
        if 'Failed' in equal: passed_all = False


        return passed_all


def compare_matrix(ref, grid, minv=False):
    passed = []
    differences = []
    equal = True
    for (i, row) in enumerate(ref):
        passed.append([])
        differences.append([])
        grid_row = grid[i].strip().split(' ')
        for (j, data) in enumerate(row):
            if j < i and minv: # cuda lower triangle not filled in
                if float(grid_row[j]) == 0: 
                    passed[i].append(True)
                    differences[i].append(0)
                elif abs(data - float(grid_row[j])) < DIFF: 
                    passed[i].append(True)
                    differences[i].append(abs(data - float(grid_row[j])))
                else: 
                    passed[i].append(False)
                    equal = False
                    differences[i].append(abs(data - float(grid_row[j])))
            else:
                if abs(data - float(grid_row[j])) < DIFF: passed[i].append(True)
                else: 
                    passed[i].append(False)
                    equal = False
                differences[i].append(abs(data - float(grid_row[j])))
    sum_diff = 0
    for i in differences: sum_diff += sum(i)
    if sum_diff < DIFF * len(differences[0]) * len(differences): equal = True
    else: equal = False
    if equal: equal = "\033[92mPassed\033[0m" # Green 
    else: 
        equal = "\033[91mFailed\033[0m" # Red
    return np.array(passed), np.array(differences), equal, sum_diff


def compare_array(ref, grid):
    passed = []
    differences = []
    for (idx, val) in enumerate(grid):
        if abs(float(val) - ref[idx]) < DIFF: passed.append(True)
        else: passed.append(False)
        differences.append(abs(float(val) - ref[idx]))
    if sum(differences) < len(differences) * DIFF: equal = "\033[92mPassed\033[0m" # Green
    else: equal = "\033[91mFailed\033[0m" # Red
    return passed, differences, equal, sum(differences)




if __name__ == "__main__":
    inputs = parseInputs(NO_ARG_OPTION = True)
    if not inputs is None:
        URDF_PATH, DEBUG_MODE, FILE_NAMESPACE_NAME, FLOATING_BASE = parseInputs()
    else: 
        print(f'Usage: printGRiD.py URDF_PATH FILE_NAMESPACE_NAME (-f) (-d)')
        exit()
    testGRiD(URDF_PATH, FLOATING_BASE)
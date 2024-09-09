import sys
import pathlib
import random
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth = 100)

def printUsage(NO_ARG_OPTION = False):
    print("Usage is: script.py PATH_TO_URDF (FILE_NAMESPACE_NAME) (-d) (-f)")
    print("                    where -D indicates full debug mode")
    print("                    where -f indicates floating base")
    if NO_ARG_OPTION:
        print("Alternative usage assuming grid.cuh is already generated: script.py")

def fileExists(FILE_PATH):
    return pathlib.Path(FILE_PATH).is_file()

def validateFile(FILE_PATH, NO_ARG_OPTION = False):
    if not fileExists(FILE_PATH):
        print("[!Error] grid.cuh does not exist")
        printUsage(NO_ARG_OPTION)
        exit()

def parseInputs(NO_ARG_OPTION = False):
    args = sys.argv[1:]
    
    if len(args) == 0:
        if NO_ARG_OPTION:
            validateFile("grid.cuh", NO_ARG_OPTION)
            print("Using generated grid.cuh")
            return None
        print("[!Error] No URDF filepath specified")
        printUsage(NO_ARG_OPTION)
        exit()
    
    URDF_PATH = args[0]
    validateFile(URDF_PATH, NO_ARG_OPTION)

    args = args[1:]

    DEBUG_MODE = False
    FLOATING_BASE = False
    FILE_NAMESPACE_NAME = 'grid'
    
    for arg in args:
        if arg.lower() == '-d': DEBUG_MODE = True
        elif arg.lower() == '-f': FLOATING_BASE = True
        else: FILE_NAMESPACE_NAME = arg
    
    if FLOATING_BASE: DEBUG_MODE = False

    print("Running with: DEBUG_MODE = " + str(DEBUG_MODE))
    print("           FLOATING_BASE = " + str(FLOATING_BASE))
    print("                    URDF = " + URDF_PATH)
    print("                    NAME = " + FILE_NAMESPACE_NAME)

    return (URDF_PATH, DEBUG_MODE, FILE_NAMESPACE_NAME, FLOATING_BASE)

def validateRobot(robot, NO_ARG_OPTION = False):
    if robot == None:
        print("[!Error] URDF parsing failed. Please make sure you input a valid URDF file.")
        printUsage(NO_ARG_OPTION)
        exit()

def printErr(a, b, FULL_DEBUG = False, TOLERANCE = 1e-10):
    err = a - b
    err = abs(err) > TOLERANCE
    if err.any():
        print(err)
        if (FULL_DEBUG):
            print("Inputs were:")
            print(a)
            print(b)
    else:
        print("  passed")

def rand3_to_quat(u, v, w):
        u = abs(u)
        v = abs(v)
        w = abs(w)
        while u > 1:
            u -= 1
        while v > 1:
            v -= 1
        while w > 1:
            w -= 1
        sqrtomu = np.sqrt(1-u)
        sqrtu = np.sqrt(u)
        PI = 3.14159
        a = sqrtomu * np.sin(2*PI*v)
        b = sqrtomu * np.cos(2*PI*v)
        c = sqrtu * np.sin(2*PI*w)
        d = sqrtu * np.cos(2*PI*w)
        return (a, b, c, d)

def initializeValues(robot, MATCH_CPP_RANDOM = False):
    # allocate memory
    n = robot.get_num_pos()
    m = robot.get_num_vel()
    q = np.zeros((n))
    qd = np.zeros((m))
    u = np.zeros((m))

    if (MATCH_CPP_RANDOM):
        # load CPP rand point
        if n > 0:
            q[0] = -0.336899
            if m > 0:
                qd[0] = 0.43302
                u[0] = 0.741788
        if n > 1:
            q[1] = 1.29662
            if m > 1:
                qd[1] = -0.421561
                u[1] = 1.92844
        if n > 2:
            q[2] = -0.677475 
            if m > 2:
                qd[2] = -0.645439
                u[2] = -0.903882
        if n > 3:
            q[3] = -1.42182
            if m > 3:
                qd[3] = -1.86055
                u[3] = 0.0333959
        if n > 4:
            q[4] = -0.706676
            if m > 4:
                qd[4] = -0.0130938
                u[4] = 1.17986
        if n > 5:
            q[5] = -0.134981 
            if m > 5:
                qd[5] = -0.458284
                u[5] = -1.94599
        if n > 6:
            q[6] = -1.14953
            if m > 6:
                qd[6] = 0.741174
                u[6] = 0.32869
        if n > 7:
            q[7] = -0.296646
            if m > 7:
                qd[7] = 1.76642
                u[7] = -0.139457
        if n > 8:
            q[8] = 2.13845
            if m > 8:
                qd[8] = 0.898011
                u[8] = 2.00667
        if n > 9:
            q[9] = 2.00956
            if m > 9:
                qd[9] = -1.85675
                u[9] = -0.519292
        if n > 10:
            q[10] = 1.55163
            if m > 10:
                qd[10] = 1.62223
                u[10] = -0.711198
        if n > 11:
            q[11] = 2.2893
            if m > 11:
                qd[11] = 0.709379
                u[11] = 0.376638
        if n > 12:
            q[12] = 0.0418005
            if m > 12:
                qd[12] = -0.382885
                u[12] = -0.209225
        if n > 13:
            q[13] = -0.125271
            if m > 13:
                qd[13] = -0.239602
                u[13] = -0.816928
        if n > 14:
            q[14] = -1.35512
            if m > 14:
                qd[14] = 1.88499
                u[14] = -0.943019
        if n > 15:
            q[15] = -0.606463
            if m > 15:
                qd[15] = -2.20784
                u[15] = -2.16433
        if n > 16:
            q[16] = -2.13552
            if m > 16:
                qd[16] = -0.921183
                u[16] = 1.37954
        if n > 17:
            q[17] = 0.229695
            if m > 17:
                qd[17] = -0.110463
                u[17] = 0.456738
        if n > 18:
            q[18] = 0.229592
            if m > 18:
                qd[18] = -1.64542
                u[18] = -0.702506
        if n > 19:
            q[19] = -0.197398
            if m > 19:
                qd[19] = -1.7481
                u[19] = 0.159814
        if n > 20:
            q[20] = -0.221438
            if m > 20:
                qd[20] = -0.562579
                u[20] = 0.944469
        if n > 21:
            q[21] = 1.02441
            if m > 21:
                qd[21] = 1.02289
                u[21] = 0.100297
        if n > 22:
            q[22] = -0.9309
            if m > 22:
                qd[22] = 0.21233
                u[22] = -0.1311
        if n > 23:
            q[23] = 1.12961
            if m > 23:
                qd[23] = 1.30624
                u[23] = 0.750389
        if n > 24:
            q[24] = 0.864741
            if m > 24:
                qd[24] = 1.31059
                u[24] = -0.666778
        if n > 25:
            q[25] = 0.705222
            if m > 25:
                qd[25] = -0.0383565
                u[25] = 0.486885
        if n > 26:
            q[26] = 0.0810176
            if m >260:
                qd[26] = 0.317353
                u[26] = 0.513445
        if n > 27:
            q[27] = 0.541962
            if m > 27:
                qd[27] = 0.479234
                u[27] = 0.0573834
        if n > 28:
            q[28] = 1.01213
            if m > 28:
                qd[28] = 0.55686
                u[28] = 0.425883
        if n > 29:
            q[29] = 2.213
            if m > 29:
                qd[29] = 0.541122
                u[29] = 0.293804
        if n > 30:
            print("[!ERROR] CPP Random Match only implemented up to n = 30. Please use a lower dof URDF.")
            exit()
    else:
        for i in range(n):
            q[i] = random.random()
            if i < m:
                qd[i] = random.random()
                u[i] = random.random()
    
    if robot.floating_base and robot.using_quaternion:
        (a, b, c, d) = rand3_to_quat(q[3],q[4],q[5])
        q[3] = a
        q[4] = b
        q[5] = c
        q[6] = d

    return q, qd, u, n
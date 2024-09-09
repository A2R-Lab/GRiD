#!/usr/bin/python3
from URDFParser import URDFParser
from GRiDCodeGenerator import GRiDCodeGenerator
from util import parseInputs, printUsage, validateRobot
from numpy import identity, zeros
import sys

J_TYPE = ["'Rx'", "'Ry'", "'Rz'", "'Px'", "'Py'", "Pz'"]
IDENTITY_MATRIX = identity(6)
ZERO_MATRIX = zeros(6)

#### Matlab Model Generator
## Parses NB, parent array, jtype array, Xtree and I matrices from URDF and create .m file compatible with Featherstone source code in working directory
def generate_matlab_model(robot, floating_base):
    original_stdout = sys.stdout
    f = open(f"{robot.name}.m",  "w")
    sys.stdout = f
    print(f"function robot = {robot.name}()")
    #robot NB
    if floating_base:
        print(f"\trobot.NB = {robot.get_num_joints()+5};")
    else:
        print(f"\trobot.NB = {robot.get_num_joints()};")
    parent_array = robot.get_parent_id_array()
    #robot parent array
    if floating_base:
        parent_array = list(range(0,len(parent_array)+5))
    else:
        for i in range(len(parent_array)):
            parent_array[i] = parent_array[i]
    print(f"\trobot.parent = [", end="")
    print(*parent_array, "];")
    if floating_base:
        print("\trobot.jtype = {'Px', 'Py', 'Pz', 'Rx', 'Ry', 'Rz',", end="")
    else:
        print("\trobot.jtype = {", end="")
    joints = robot.get_joints_ordered_by_id()
    jtype_array = []
    for i in range(robot.get_num_joints()):
        s = joints[i].S 
        for index in range(len(s)):
            if s[index] == 1:
                jtype_array.append(J_TYPE[index])
                if i != robot.get_num_joints()-1:
                    jtype_array.append(",")
                break
    print(*jtype_array, "};")
    Imats = robot.get_Imats_ordered_by_id()
    for x in range(len(joints)):
        # prints the Xtree
        joint = joints[x].origin.Xmat_sp_fixed
        print("\trobot.Xtree{" + str(x) + "} =", end=" ")
        print("[", end="")
        for i in range(6):
            for j in range(6):
                if j == 0 and i != 0:
                    print("\t", end="")
                print(joint[i,j], end=" ")
            if(i != 5):
                print(";")
            else:
                print("]" + ";")
        print()
        # print the I
        print("\trobot.I{" + str(x) + "} =", end=" ")
        print("[", end="")
        for a in range(6):
            for b in range(6):
                if b == 0 and a != 0:
                    print("\t", end="")
                print(Imats[x][a,b], end=" ")
            if(a != 5):
                print(";")
            else:
                print("]" + ";")
        print()
    sys.stdout = original_stdout
    f.close()

def main():
    URDF_PATH, DEBUG_MODE, FILE_NAMESPACE_NAME, FLOATING_BASE = parseInputs()

    parser = URDFParser()
    robot = parser.parse(URDF_PATH, floating_base=FLOATING_BASE)

    validateRobot(robot)

    # FLOATINGBASE = False
    # generate_matlab_model(robot, FLOATINGBASE)
    # print(f"m file genereated and saved to {robot.name}.m!")

    codegen = GRiDCodeGenerator(robot, DEBUG_MODE, True, FILE_NAMESPACE = FILE_NAMESPACE_NAME)
    if FLOATING_BASE: include_homogenous_transforms = False
    else: include_homogenous_transforms = True
    codegen.gen_all_code(include_homogenous_transforms = include_homogenous_transforms)
    print("New code generated and saved to grid.cuh!")

if __name__ == "__main__":
    main()
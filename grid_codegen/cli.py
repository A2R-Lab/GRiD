"""CLI helpers and entry point for the grid-generate command."""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=100)


# ---------------------------------------------------------------------------
# CLI helpers (formerly in util/util.py)
# ---------------------------------------------------------------------------

def printUsage(NO_ARG_OPTION=False):
    print("Usage is: script.py PATH_TO_URDF (-t FIXED_TARGET_NAMES) (-n FILE_NAMESPACE_NAME) (-d) (-f) (-c) (--collision-res RES) (--collision-native) (--algorithm-list LIST) (--no-mujoco-kernels) (-o OUTPUT)")
    print("                    where -d indicates full debug mode")
    print("                    where -f indicates floating base")
    print("                    where -c spherizes collision geometry and emits grid_collision (config_free)")
    print("                    where --collision-res sets the sphere spacing (comma-separated for a broad->fine cascade)")
    print("                    where --collision-native uses the URDF's native collision primitives (implies -c)")
    print("                    where --algorithm-list subsets the emitted algorithms/profiles (comma-separated; shrinks big-robot builds)")
    print("                    where --no-mujoco-kernels skips the mjx twins (the dominant humanoid build cost)")
    print("                    where -o sets the output header path (default grid.cuh)")
    if NO_ARG_OPTION:
        print("Alternative usage assuming grid.cuh is already generated: script.py")


def fileExists(FILE_PATH):
    return pathlib.Path(FILE_PATH).is_file()


def validateFile(FILE_PATH, NO_ARG_OPTION=False):
    if not fileExists(FILE_PATH):
        print("[!Error] file not found: %s" % FILE_PATH)
        printUsage(NO_ARG_OPTION)
        sys.exit(1)


def parseInputs(NO_ARG_OPTION=False):
    parser = argparse.ArgumentParser(
        description="Process a URDF file and Generate Optimized CUDA Kinematics and Dynamics Code."
    )
    parser.add_argument("urdf_path", nargs="?" if NO_ARG_OPTION else None,
                        help="The path to the URDF file")
    parser.add_argument("-t", "--fixed-target-names", default="", type=str,
                        help="Fixed joint kinematic target names")
    parser.add_argument("-n", "--namespace", default="grid", type=str,
                        help="File namespace name")
    parser.add_argument("-d", "--debug", default=False, action="store_true",
                        help="Enable debug mode")
    parser.add_argument("-f", "--floating-base", default=False, action="store_true",
                        help="Add a floating base")
    parser.add_argument("-c", "--collision", default=False, action="store_true",
                        help="Spherize the URDF collision geometry and emit the grid_collision "
                             "namespace (config_free)")
    parser.add_argument("--collision-res", default="0.05", type=str,
                        help="Collision sphere spacing in meters (smaller = finer/more spheres). "
                             "Comma-separate multiple densities for a broad->fine cascade, e.g. "
                             "'0.10,0.05' (config_free uses coarsest to reject + finest to confirm). "
                             "Default 0.05")
    parser.add_argument("--collision-native", default=False, action="store_true",
                        help="Use the URDF's own collision primitives as NATIVE capsule rows "
                             "(sphere -> degenerate row; cylinder -> containing capsule, conservative; "
                             "box/mesh links keep spherized rows at --collision-res). Emits a "
                             "broad->fine cascade with covering spheres derived from the rows. "
                             "Implies -c.")
    # N2.8 (2026-09-08): the two knobs that decide whether a humanoid build
    # fits in RAM, plus the output path — previously gen_all_code-only.
    parser.add_argument("--algorithm-list", default=None, type=str,
                        help="Comma-separated algorithm/profile subset to emit (e.g. "
                             "'dynamics' or 'inverse_dynamics,minv,forward_dynamics'). "
                             "Default: the full 'all' profile. Subsetting is how "
                             "big-robot (humanoid) builds stay inside RAM.")
    parser.add_argument("--no-mujoco-kernels", default=False, action="store_true",
                        help="Skip instantiating the MuJoCo-convention (mjx) kernel twins "
                             "(floating-base only; they dominate humanoid build cost). "
                             "The pin-convention kernels are unaffected.")
    parser.add_argument("-o", "--output", default="grid.cuh", type=str,
                        help="Output header path (default grid.cuh in the current directory)")
    args = parser.parse_args()

    if args.urdf_path is None:
        if NO_ARG_OPTION:
            validateFile("grid.cuh", NO_ARG_OPTION)
            print("Using generated grid.cuh")
            return None
        print("[!Error] No URDF filepath specified")
        printUsage(NO_ARG_OPTION)
        sys.exit(1)

    validateFile(args.urdf_path, NO_ARG_OPTION)

    # Derived/normalized fields, set back on the namespace so every consumer
    # agrees (parseInputs returns the argparse Namespace as of 2026-09-08 —
    # the old grow-forever tuple had already drifted out of sync with one
    # consumer's unpack).
    args.collision = args.collision or args.collision_native
    args.collision_res = [float(x) for x in str(args.collision_res).split(",") if x.strip()]
    if args.floating_base:
        args.debug = False
    if args.algorithm_list is not None:
        args.algorithm_list = [a.strip() for a in args.algorithm_list.replace(";", ",").split(",")
                               if a.strip()]

    print("Running with: DEBUG_MODE = " + str(args.debug))
    print("           FLOATING_BASE = " + str(args.floating_base))
    print("                    URDF = " + args.urdf_path)
    print("      FIXED_TARGET_NAMES = " + args.fixed_target_names)
    print("               FILE_NAME = " + args.namespace)
    print("               COLLISION = " + str(args.collision) +
          ((" (NATIVE rows, res=%s)" % ",".join("%g" % r for r in args.collision_res)) if args.collision_native else
           (" (res=%s)" % ",".join("%g" % r for r in args.collision_res)) if args.collision else ""))
    if args.algorithm_list is not None:
        print("          ALGORITHM_LIST = " + ",".join(args.algorithm_list))
    if args.no_mujoco_kernels:
        print("          MUJOCO_KERNELS = disabled")
    if args.output != "grid.cuh":
        print("                  OUTPUT = " + args.output)

    return args


def validateRobot(robot, NO_ARG_OPTION=False):
    if robot is None:
        print("[!Error] URDF parsing failed. Please make sure you input a valid URDF file.")
        printUsage(NO_ARG_OPTION)
        sys.exit(1)


# ---------------------------------------------------------------------------
# grid-generate entry point
# ---------------------------------------------------------------------------

def main():
    """Entry point for the ``grid-generate`` CLI command."""
    from URDFParser import URDFParser
    from grid_codegen import GRiDCodeGenerator

    args = parseInputs()
    parser = URDFParser()
    robot = parser.parse(args.urdf_path, floating_base=args.floating_base)

    validateRobot(robot)

    collision_spec = None
    if args.collision_native:
        from grid_codegen.algorithms._collision import native_collision_spec_from_urdf
        collision_spec = native_collision_spec_from_urdf(robot, args.urdf_path, args.collision_res[-1])
        print("      collision rows = " + ", ".join(
            "%s:%d" % (t["name"], len(t["anchor"])) for t in collision_spec["tiers"]))
    elif args.collision:
        from grid_codegen.algorithms._collision import multi_tier_collision_spec_from_urdf
        collision_spec = multi_tier_collision_spec_from_urdf(robot, args.urdf_path, args.collision_res)
        if "tiers" in collision_spec:
            print("      collision spheres = " + ", ".join(
                "%s:%d" % (t["name"], len(t["anchor"])) for t in collision_spec["tiers"]))
        else:
            print("      collision spheres = " + str(len(collision_spec["anchor"])))

    codegen = GRiDCodeGenerator(robot, args.debug, True, FILE_NAMESPACE=args.namespace)
    include_homogenous_transforms = not args.floating_base
    codegen.gen_all_code(
        include_homogenous_transforms=include_homogenous_transforms,
        fixed_target_name=args.fixed_target_names,
        collision_spec=collision_spec,
        algorithm_list=args.algorithm_list,
        enable_mujoco_kernels=(False if args.no_mujoco_kernels else None),
        output_path=args.output,
    )
    print("New code generated and saved to %s!" % args.output)


if __name__ == "__main__":
    main()

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
        print("[!Error] URDF file not found: %s" % FILE_PATH)
        print("         Check the path (a relative path is resolved against the current "
              "directory) and run grid-generate again.")
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
    parser.add_argument("--plant", default=False, action="store_true",
                        help="Emit the grid_plant block even when the request carries no plant algorithm, "
                             "cost frames or contact frames (a kinematics-only header drops it by default)")
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
    parser.add_argument('--spherized-urdf', help='Existing sphere URDF path, or the optional foam preset; implies -c')
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
        print("[!Error] No URDF path given. Pass the robot's URDF file as the first "
              "argument, for example: grid-generate robot.urdf")
        printUsage(NO_ARG_OPTION)
        sys.exit(1)

    validateFile(args.urdf_path, NO_ARG_OPTION)

    # Derived/normalized fields, set back on the namespace so every consumer
    # agrees (parseInputs returns the argparse Namespace as of 2026-09-08 —
    # the old grow-forever tuple had already drifted out of sync with one
    # consumer's unpack).
    if args.spherized_urdf and args.collision_native:
        parser.error('--spherized-urdf and --collision-native select different geometry models')
    args.collision = args.collision or args.collision_native or bool(args.spherized_urdf)
    try:
        args.collision_res = [float(x) for x in str(args.collision_res).split(",") if x.strip()]
    except ValueError:
        print("[!Error] --collision-res expects a sphere spacing in meters, or a comma-separated "
              "list such as 0.10,0.05 for a broad-to-fine cascade; got %r." % args.collision_res)
        sys.exit(1)
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
        print("[!Error] URDF parsing failed: the parser returned no robot model. "
              "Check that the file is a valid URDF (one <robot> root with <link> and "
              "<joint> elements) and run grid-generate again.")
        printUsage(NO_ARG_OPTION)
        sys.exit(1)


def reportParseError(exc):
    """A rejected URDF: say why and what to change, then exit 1 (same code as the
    other input errors; the parser's typed exception carries the file and reason)."""
    print("[!Error] URDF parsing failed: %s" % exc)
    print("         Fix the model file and run grid-generate again (a floating root link "
          "needs -f; a joint type the parser does not model must be rewritten as one "
          "of the supported types it lists).")
    sys.exit(1)


# ---------------------------------------------------------------------------
# grid-generate entry point
# ---------------------------------------------------------------------------

def main():
    """Entry point for the ``grid-generate`` CLI command."""
    from URDFParser import URDFParser, URDFParseError
    from grid_codegen import GRiDCodeGenerator
    from grid_codegen._algo_profiles import normalize_codegen_algorithms

    args = parseInputs()
    parser = URDFParser()
    try:
        robot = parser.parse(args.urdf_path, floating_base=args.floating_base)
    except URDFParseError as exc:
        reportParseError(exc)

    validateRobot(robot)

    collision_spec = None
    try:
        collision_spec = _collision_spec(args, robot)
    except ValueError as exc:
        print("[!Error] Collision geometry could not be built: %s" % exc)
        print("         Make every <collision> mesh resolvable (set ROS_PACKAGE_PATH for "
              "package:// URIs, or keep the asset next to the URDF), or pass "
              "--spherized-urdf with a sphere model you already have.")
        sys.exit(1)

    codegen = GRiDCodeGenerator(robot, args.debug, True, FILE_NAMESPACE=args.namespace)
    if args.algorithm_list is not None:
        # The same check gen_all_code runs, done first so a typo exits with a
        # message instead of a traceback from inside the generator.
        try:
            normalize_codegen_algorithms(codegen, algorithm_list=args.algorithm_list)
        except ValueError as exc:
            print("[!Error] --algorithm-list was not understood: %s" % exc)
            print("         Use canonical algorithm names such as inverse_dynamics, minv, "
                  "forward_dynamics, end_effector_pose, or a profile such as all, dynamics, "
                  "kinematics, second-order (comma-separated).")
            sys.exit(1)
    include_homogenous_transforms = not args.floating_base
    codegen.gen_all_code(
        include_homogenous_transforms=include_homogenous_transforms,
        fixed_target_name=args.fixed_target_names,
        collision_spec=collision_spec,
        algorithm_list=args.algorithm_list,
        enable_mujoco_kernels=(False if args.no_mujoco_kernels else None),
        output_path=args.output,
        emit_plant=(True if args.plant else None),
    )
    print("New code generated and saved to %s!" % args.output)


def _collision_spec(args, robot):
    """Resolve the -c / --collision-native / --spherized-urdf geometry into the
    collision_spec gen_all_code takes (None when no collision flag is set)."""
    collision_spec = None
    if args.spherized_urdf:
        from grid_codegen.algorithms._collision import resolve_spherized_urdf, build_sphere_tiers
        path = resolve_spherized_urdf(args.spherized_urdf, args.urdf_path)
        tier = build_sphere_tiers(robot, {'fine': path})['fine']
        collision_spec = {key: tier[key] for key in ('anchor', 'offset', 'radius', 'self_cc_ranges')}
    elif args.collision_native:
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
    return collision_spec


if __name__ == "__main__":
    main()

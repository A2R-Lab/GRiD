#!/usr/bin/env python3
"""Codegen byte-gate: generate grid.cuh for 7 representative cells (iiwa14
fixed ×4 incl. full/multi-target/collision, go2 floating ×2, fr3 mimic) into
<outdir> and print sha256 per cell. Run BEFORE and AFTER a codegen change and
diff the hashes — the byte-identical-codegen discipline's harness (CLAUDE.md).

Usage: .venv/bin/python tools/byte_gate.py <outdir>
(promoted from docs/open-tasks/tools_byte_gate.py, 2026-09-09; the repo root
is derived from this file's location — no hardcoded paths)
"""
import hashlib, shutil, subprocess, sys, os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bindings"))
os.chdir(REPO)
# never let stale codegen bytecode leak between A and B
shutil.rmtree(REPO / "grid_codegen" / "__pycache__", ignore_errors=True)

from config import robot_urdf  # noqa: E402
from URDFParser import URDFParser  # noqa: E402
from grid_codegen import GRiDCodeGenerator  # noqa: E402

FULL_ALGOS = ["all", "frame_jacobian", "frame_jacobian_dot", "osc_inertia",
              "end_effector_pose_runtime", "end_effector_pose_gradient_runtime",
              "integrator_hessian"]

# (name, urdf, floating, gen_all_code extra kwargs builder)
CELLS = [
    ("iiwa14", robot_urdf("iiwa14"), False, lambda r, u: {}),
    ("go2", robot_urdf("go2"), True, lambda r, u: {}),
    ("fr3", robot_urdf("fr3"), True, lambda r, u: {}),
    # bindings-style full surface + runtime tables (covers the opt-in emitters)
    ("iiwa14_full", robot_urdf("iiwa14"), False,
     lambda r, u: dict(algorithm_list=FULL_ALGOS, runtime_inertia=True,
                       runtime_transform=True, runtime_joint_dynamics=True)),
    ("go2_full", robot_urdf("go2"), True,
     lambda r, u: dict(algorithm_list=FULL_ALGOS, runtime_inertia=True,
                       runtime_transform=True, runtime_joint_dynamics=True)),
    # multi-target + collision emission paths (H2 lesson: default baselines miss them)
    ("iiwa14_mt", robot_urdf("iiwa14"), False,
     lambda r, u: dict(multi_target_batch=(
         [{"anchor_jid": int(j), "offset": (0.0, 0.0, 0.0)} for j in r.get_leaf_nodes()]
         + [{"anchor_jid": int(j), "offset": (0.03, -0.02, 0.05)} for j in r.get_leaf_nodes()]))),
    ("iiwa14_coll", robot_urdf("iiwa14"), False,
     lambda r, u: dict(codegen_profile="kinematics", collision_spec=_coll_spec(r, u))),
]


def _coll_spec(robot, urdf):
    from grid_codegen.algorithms._collision import collision_spec_from_urdf
    return collision_spec_from_urdf(robot, str(urdf), resolution=0.06)


outdir = Path(sys.argv[1]); outdir.mkdir(parents=True, exist_ok=True)
for name, urdf, floating, extra in CELLS:
    parser = URDFParser()
    robot = parser.parse(str(urdf), floating_base=floating)
    dest = outdir / f"{name}.cuh"
    codegen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid")
    kwargs = extra(robot, urdf)
    if "codegen_profile" not in kwargs:
        kwargs["include_homogenous_transforms"] = True
    codegen.gen_all_code(output_path=str(dest), **kwargs)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(f"{name} {h}")

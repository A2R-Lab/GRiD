"""grid-rbd named end-effector targets: pick the EE frame by NAME.

By default GRiD's end_effector kernels target the robot's leaf link(s). When you want a
SPECIFIC frame (a tool flange, a gripper TCP, a sensor mount), select it by joint name.
Two routes:

  1. BAKED target (codegen) — register with ee_joint_names=[...]. The compiled .so targets
     exactly those frame(s), and end_effector_pose / _gradient / _hessian all report THAT
     frame. ee_joint_names is part of the cache key, so different target choices land in
     separate cache entries. This is the fast, jittable path (numpy + jax + torch).
     The just-landed gradient/hessian codegen support means the NAMED target flows through
     end_effector_pose_gradient() and end_effector_pose_hessian(), not only the value.

  2. RUNTIME target (numpy only) — one compiled robot, choose the frame + an offset point at
     CALL time:
        end_effector_pose_runtime(q, ee_joint_names=..., ee_offsets=...)          -> (B, NUM_EE, 6)
        end_effector_pose_gradient_runtime(q, ee_joint_names=..., ee_offsets=...) -> (B, NUM_EE, 6, NV)
     ee_joint_names: None (all leaves) | a name | a list of names.
     ee_offsets:     None (frame origin) | one [x,y,z] (or [x,y,z,1]) per selected frame.
     Great for OSC / task-space control where the target/offset changes online without a rebuild.

Run:  python bindings/examples/ee_named_targets.py --urdf PATH --ee-joint JOINT_NAME [--batch 8]
Needs: pip install -e .   ·   nvcc on PATH   ·   a URDF + a fixed-joint EE/tool name

NOTE: the EE joint name is robot-specific (it is a <joint> name in YOUR URDF). Pass it with
--ee-joint; there is no universal default. For iiwa14 the flange joint is often "iiwa_joint_ee".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_DEFAULT_URDF = (
    Path.home()
    / ".cache/robot_descriptions/drake/manipulation/models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=str(_DEFAULT_URDF))
    ap.add_argument("--ee-joint", default=None,
                    help="Fixed-joint EE/tool name in the URDF (e.g. iiwa_joint_ee). "
                         "Default None => all leaf frames.")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    urdf = Path(args.urdf).expanduser()
    if not urdf.exists():
        sys.exit(f"URDF not found: {urdf} (pass --urdf)")

    import grid_rbd

    B = args.batch
    ee_names = [args.ee_joint] if args.ee_joint else None

    # ── 1. BAKED named target: end_effector_pose/_gradient/_hessian see it ────
    # ee_joint_names is in the cache key; this compiles a .so whose EE kernels
    # target the named frame. (None => all leaf nodes, the default codegen.)
    h = grid_rbd.register_robot(
        "iiwa14_ee_named", str(urdf),
        ee_joint_names=ee_names, max_batch_size=max(B, 8))
    nq, nv = h.num_joints, h.num_vel
    tgt = ee_names[0] if ee_names else "<all leaf frames>"
    print(f"baked EE target = {tgt}   num_ees={h.num_ees}  nq={nq} nv={nv}")

    rng = np.random.default_rng(0)
    q = rng.standard_normal((B, nq)).astype(np.float32)

    pose = h.end_effector_pose(q)                  # (B, 6*num_ees) = [xyz; rpy] per EE
    grad = h.end_effector_pose_gradient(q)         # (B, 6*num_ees, NV)  <- named target flows here
    hess = h.end_effector_pose_hessian(q)          # (B, 6*num_ees, NV, NV)
    print(f"  end_effector_pose          -> {pose.shape}  row0 xyz={pose.reshape(B, -1)[0, :3]}")
    print(f"  end_effector_pose_gradient -> {grad.shape}  (named target, not just leaf)")
    print(f"  end_effector_pose_hessian  -> {hess.shape}")

    # ── 2. RUNTIME named target + offset: choose frame/offset per call ───────
    # One compiled robot serves any target frame + measurement-point offset,
    # picked at call time (numpy backend). Useful for task-space / OSC control.
    if ee_names:
        # measure 5 cm out along +z from the named frame (a TCP offset)
        tcp_offset = [np.array([0.0, 0.0, 0.05], dtype=np.float32)]
        pose_rt = h.end_effector_pose_runtime(q, ee_joint_names=ee_names, ee_offsets=tcp_offset)
        jac_rt  = h.end_effector_pose_gradient_runtime(q, ee_joint_names=ee_names, ee_offsets=tcp_offset)
        print(f"\n  runtime target={ee_names} offset=+5cm z:")
        print(f"    end_effector_pose_runtime          -> {pose_rt.shape}  (B, NUM_EE, 6)")
        print(f"    end_effector_pose_gradient_runtime -> {jac_rt.shape}  (B, NUM_EE, 6, NV)")
    else:
        # default: all leaf frames at their origins
        pose_rt = h.end_effector_pose_runtime(q)
        print(f"\n  runtime (all leaf frames) -> {pose_rt.shape}  (B, NUM_EE, 6)")

    print("\nTakeaway: bake ee_joint_names=[...] for a fast, jittable fixed target (value + "
          "gradient + hessian all honor it); use *_runtime for online frame/offset changes "
          "without a rebuild.")


if __name__ == "__main__":
    main()

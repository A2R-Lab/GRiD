"""Multi-contact external forces through the Python bindings (both-feet pattern).

Registers go2 (floating base) with its four feet as baked contact frames, maps
per-foot world-aligned contact wrenches to joint-local f_ext in one kernel call,
and feeds the result to inverse dynamics / forward dynamics. The same pattern
covers a humanoid's two feet — pass those fixed-joint names instead.

Conventions:
  * contact_frames are URDF FIXED-JOINT names; registration order fixes the
    f_c column order.
  * f_c is (B, 6*num_contact_frames): per frame [n_w; f_w] with WORLD-ALIGNED
    axes and the moment taken about the contact-frame origin (pinocchio
    LOCAL_WORLD_ALIGNED). A pure 3D point force is just n_w = 0.
  * the returned f_ext is (B, 6*num_bodies) body-local [angular; linear] —
    exactly what every dynamics op's f_ext= argument expects.
  * FLOATING-BASE width discipline: q is nq-wide ([pos(3), quat_xyzw(4),
    joints]); qd/qdd/u are ALSO passed nq-wide with the trailing quaternion-
    padding slot zeroed (leading nv entries = tangent data). An nv-wide qd is
    the classic footgun and the handle rejects it with a precise error.

Run:  .venv/bin/python bindings/examples/multi_contact_fext.py [--urdf PATH]
"""
import argparse
from pathlib import Path

import numpy as np

import grid_rbd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_URDF = _REPO_ROOT / "config" / "robot_assets" / "go2.urdf"
GO2_FEET = ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]


def main(urdf: str):
    h = grid_rbd.register_robot(
        "go2_multi_contact", urdf, floating_base=True,
        contact_frames=GO2_FEET,
        # Subset build: this example only calls inverse/forward dynamics (+ the
        # contact helper, which is baked by contact_frames= regardless of the
        # algorithm list). Without this a floating go2 compiles the whole
        # ~35-algorithm surface plus its mjx twins — minutes of nvcc for nothing.
        algorithm_list=["inverse_dynamics", "forward_dynamics"],
    )
    print("registered contact frames:", [f["name"] for f in h.contact_frames])

    B = 4
    nq, nv = h.num_joints, h.num_vel
    rng = np.random.default_rng(0)

    # q = [pos(3), quat_xyzw(4) identity, joints]; qd/qdd/u nq-wide + zero pad.
    q = np.zeros((B, nq), dtype=np.float32)
    q[:, 6] = 1.0  # identity quaternion (xyzw: w at index 6)
    q[:, 7:] = rng.uniform(-0.3, 0.3, size=(B, nq - 7)).astype(np.float32)
    pad = np.zeros((B, nq - nv), dtype=np.float32)
    qd = np.concatenate([np.zeros((B, nv), dtype=np.float32), pad], axis=1)
    qdd = np.concatenate([np.zeros((B, nv), dtype=np.float32), pad], axis=1)
    u = np.concatenate([np.zeros((B, nv), dtype=np.float32), pad], axis=1)

    # Stance: each foot pushes up with ~1/4 of the robot's weight (world +z),
    # no contact moments (point feet).
    f_c = np.zeros((B, 6 * len(GO2_FEET)), dtype=np.float32)
    for c in range(len(GO2_FEET)):
        f_c[:, 6 * c + 5] = 40.0  # f_w = [0, 0, +40 N]

    f_ext = h.contact_fext(q, f_c)                       # (B, 6*num_bodies)
    tau_free = np.asarray(h.inverse_dynamics(q, qd, qdd))
    tau_stance = np.asarray(h.inverse_dynamics(q, qd, qdd, f_ext=f_ext))
    print("|tau| free   :", float(np.max(np.abs(tau_free))))
    print("|tau| stance :", float(np.max(np.abs(tau_stance))))

    qdd_free = np.asarray(h.forward_dynamics(q, qd, u))
    qdd_stance = np.asarray(h.forward_dynamics(q, qd, u, f_ext=f_ext))
    print("max |qdd| free/stance:", float(np.max(np.abs(qdd_free))),
          float(np.max(np.abs(qdd_stance))))
    h.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=str(_DEFAULT_URDF))
    args = ap.parse_args()
    main(args.urdf)

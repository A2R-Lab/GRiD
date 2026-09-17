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

Run:  .venv/bin/python bindings/examples/multi_contact_fext.py
"""
import numpy as np

import grid_rbd

GO2_URDF = "config/robot_assets/go2.urdf"
GO2_FEET = ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]


def main():
    h = grid_rbd.register_robot(
        "go2_multi_contact", GO2_URDF, floating_base=True,
        contact_frames=GO2_FEET,
    )
    print("registered contact frames:", [f["name"] for f in h.contact_frames])

    B = 4
    rng = np.random.default_rng(0)
    q = np.tile(h.neutral_q(), (B, 1)) if hasattr(h, "neutral_q") else None
    if q is None:
        # floating base: [x y z, quat xyzw] + joint angles
        q = np.zeros((B, h.num_joints), dtype=np.float32)
        q[:, 6] = 1.0  # identity quaternion (xyzw)
        q[:, 7:] += rng.uniform(-0.3, 0.3, size=(B, h.num_joints - 7)).astype(np.float32)
    qd = np.zeros((B, h.num_vel), dtype=np.float32)
    qdd = np.zeros((B, h.num_vel), dtype=np.float32)

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

    qdd_free = np.asarray(h.forward_dynamics(q, qd, np.zeros_like(qd)))
    qdd_stance = np.asarray(h.forward_dynamics(q, qd, np.zeros_like(qd), f_ext=f_ext))
    print("max |qdd| free/stance:", float(np.max(np.abs(qdd_free))),
          float(np.max(np.abs(qdd_stance))))
    h.close()


if __name__ == "__main__":
    main()

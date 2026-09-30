"""grid-rbd tool / payload welding: attach a rigid tool at RUNTIME, no recompile.

A rigidly-welded tool (a gripper, drill, or carried object) has exactly two effects, and
`handle.attach_tool(...)` handles both with zero recompile on a robot registered with
`enable_tool=True`:

  1. its MASS/INERTIA folds into the attach link's spatial inertia (so inverse/forward
     dynamics + gradients see the composite rigid body), and
  2. its TIP becomes a new SE(3) frame hanging off the attach joint (so the runtime
     end-effector pose / Jacobian report the tool tip).

`detach_tool()` restores the baked robot. A tool can be attached ANYWHERE in the chain
(mid-chain or a leaf), and only a tool that ADDS a joint (an articulated gripper) or a
two-finger CLOSED-LOOP grasp needs special handling (see E5) -- a rigid tool never does.

Run:  python bindings/examples/tool_use.py [--urdf PATH]
Needs: pip install -e .  .  nvcc on PATH  .  an iiwa14 URDF
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

_LEAF = "iiwa_joint_7"     # iiwa14 wrist joint (the tool anchor)
_MID = "iiwa_joint_4"      # a mid-chain (forearm) joint


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _se3(R, p):
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = p
    return X


def e1_pure_payload(h, q):
    """E1 -- PURE PAYLOAD (no tip frame): a carried 2 kg mass changes the dynamics only."""
    print("\n== E1: pure payload (carried mass) ==")
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    tau0 = np.asarray(h.inverse_dynamics(q, qd, qdd)).reshape(-1)
    h.attach_tool(_LEAF, mass=2.0, com=[0.0, 0.0, 0.08])   # no tip_transform -> inertia only
    tau1 = np.asarray(h.inverse_dynamics(q, qd, qdd)).reshape(-1)
    print(f"  gravity torque delta from payload (max |Δτ|): {np.max(np.abs(tau1 - tau0)):.4f} N·m")
    h.detach_tool()


def e2_se3_tip(h, q):
    """E2 -- TOOL WITH AN SE(3) TIP FRAME: an angled drill tip has its own orientation."""
    print("\n== E2: SE(3) tool tip frame ==")
    X_tool = _se3(_rot_x(np.deg2rad(35)), [0.0, 0.0, 0.15])   # 35deg tilt + 15cm reach
    bare = np.asarray(h.end_effector_pose_runtime(q, ee_joint_names=_LEAF)).reshape(-1)
    h.attach_tool(_LEAF, mass=1.0, com=[0, 0, 0.075], tip_transform=X_tool)
    tip = np.asarray(h.end_effector_pose_runtime(q)).reshape(-1)   # defaults to the tool tip
    print(f"  bare wrist pose [xyz;rpy]: {np.round(bare, 4)}")
    print(f"  tool tip  pose [xyz;rpy]: {np.round(tip, 4)}   (position reaches out, rpy rotated)")
    h.detach_tool()


def e3_mid_chain(h, q):
    """E3 -- ATTACH ANYWHERE: weld a sensor to a FOREARM link; upstream torques change,
    the wrist tip is unaffected."""
    print("\n== E3: mid-chain attach ==")
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    tau0 = np.asarray(h.inverse_dynamics(q, qd, qdd)).reshape(-1)
    tip0 = np.asarray(h.end_effector_pose_runtime(q, ee_joint_names=_LEAF)).reshape(-1)
    h.attach_tool(_MID, mass=1.5, com=[0.0, 0.05, 0.0])
    tau1 = np.asarray(h.inverse_dynamics(q, qd, qdd)).reshape(-1)
    tip1 = np.asarray(h.end_effector_pose_runtime(q, ee_joint_names=_LEAF)).reshape(-1)
    print(f"  upstream torque delta (max |Δτ|):  {np.max(np.abs(tau1 - tau0)):.4f} N·m")
    print(f"  wrist tip pose unchanged (max Δ):  {np.max(np.abs(tip1 - tip0)):.2e}")
    h.detach_tool()


def e4_gripping(h, q):
    """E4 -- GRIPPING A TOOL: model a firmly grasped tool as a rigid attach on the wrist
    (palm) + soft-lock the finger joints (hold q fixed). No recompile; the arm carries the
    tool correctly, and the finger torques are just the inverse-dynamics torques.

    (iiwa14 has no fingers, so the 'soft lock' here is conceptual -- on a hand robot you would
    hold the finger joints at their closed angle with qd=0, optionally with high stiffness via
    runtime_joint_dynamics. The KEY point is that holding a tool needs no joint removal.)"""
    print("\n== E4: gripping a tool (rigid attach + soft-locked fingers) ==")
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    h.attach_tool(_LEAF, mass=2.5, com=[0, 0, 0.10],
                  inertia=np.diag([0.02, 0.02, 0.008]),
                  tip_transform=_se3(np.eye(3), [0, 0, 0.18]))
    tau = np.asarray(h.inverse_dynamics(q, qd, qdd)).reshape(-1)
    print(f"  holding torques with tool grasped (|τ|): {np.round(tau, 3)}")
    print("  -> the arm carries the tool with NO recompile; fingers stay real DOFs held at q.")
    h.detach_tool()


def e5_closed_loop(h, q):
    """E5 -- TWO-FINGER / CLOSED-LOOP grasp + TIP FORCES: a tool bridging two fingertips is a
    closed kinematic loop, which GRiD's tree codegen cannot represent. Reduce it to an OPEN tree:
    attach the tool rigidly to ONE contact (palm / one fingertip) and model the OTHER finger's
    grip force (or any environment reaction: grinding, pushing) as a world-aligned wrench on the
    tool tip via `tool_fext`, then feed it to the dynamics as `f_ext=`."""
    print("\n== E5: two-finger closed-loop grasp + tip forces ==")
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    h.attach_tool(_LEAF, mass=1.0, com=[0, 0, 0.08],
                  tip_transform=_se3(np.eye(3), [0, 0, 0.15]))
    tau_free = np.asarray(h.inverse_dynamics(q, qd, qdd)).reshape(-1)
    # a 6D world-aligned wrench at the tool tip (e.g. the second finger pressing + a moment):
    wrench = np.array([[0.0, 0.0, 0.2, 5.0, 0.0, -3.0]], dtype=np.float32)  # [n_w; f_w]
    fext = h.tool_fext(q, wrench)               # (B, 6*num_bodies) joint-local f_ext
    tau = np.asarray(h.inverse_dynamics(q, qd, qdd, f_ext=fext)).reshape(-1)
    print(f"  tip wrench -> joint-torque contribution (max |Δτ|): {np.max(np.abs(tau - tau_free)):.4f} N·m")
    print("  (attach to ONE fingertip + tool_fext for the other finger = open tree, no closed loop.)")
    h.detach_tool()


def e6_roundtrip(h, q):
    """E6 -- ATTACH -> USE -> DETACH round-trip + a payload-hypothesis sweep in ONE module
    (no per-hypothesis recompile)."""
    print("\n== E6: attach/detach round-trip + payload-hypothesis sweep ==")
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    tau_baked = np.asarray(h.inverse_dynamics(q, qd, qdd)).reshape(-1)
    for m in (0.0, 1.2, 3.0, 6.0, 11.2):
        h.attach_tool(_LEAF, mass=m, com=[0, 0, 0.09])
        tau = np.asarray(h.inverse_dynamics(q, qd, qdd)).reshape(-1)
        print(f"  payload {m:5.1f} kg -> |τ|_max = {np.max(np.abs(tau)):7.3f} N·m")
        h.detach_tool()
    tau_after = np.asarray(h.inverse_dynamics(q, qd, qdd)).reshape(-1)
    ok = np.allclose(tau_after, tau_baked, atol=1e-5)
    print(f"  round-trip restores the baked robot: {'OK' if ok else 'MISMATCH'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", default=str(_DEFAULT_URDF))
    args = ap.parse_args()
    urdf = Path(args.urdf).expanduser()
    if not urdf.exists():
        sys.exit(f"URDF not found: {urdf} (pass --urdf)")

    import grid_rbd
    h = grid_rbd.register_robot("iiwa14_tool", urdf_path=str(urdf), enable_tool=True)
    q = np.linspace(0.1, 0.6, h.num_joints).astype(np.float32)[None, :]

    e1_pure_payload(h, q)
    e2_se3_tip(h, q)
    e3_mid_chain(h, q)
    e4_gripping(h, q)
    e5_closed_loop(h, q)
    e6_roundtrip(h, q)
    print("\nAll tool-use examples ran.")


if __name__ == "__main__":
    main()

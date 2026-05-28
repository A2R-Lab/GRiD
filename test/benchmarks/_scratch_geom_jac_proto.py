"""Prototype: shared-chain geometric (spatial) Jacobian for ee_pose_gradient.

Goal: prove correctness vs the existing per-column analytic end_effector_pose_gradient
(fixed base: nq==nv, must match exactly), and show the shared-chain O(nq+depth) structure.

Geometric Jacobian (world frame, at the EE origin):
  column for DOF with world axis â at joint world-origin p_j:
    revolute : J_w = â,  J_v = â x (p_ee - p_j)
    prismatic: J_w = 0,  J_v = â
Then analytic pose gradient = [ J_v ; E(rpy)^-1 J_w ], rpy from the EE world transform,
E maps rpy-rates -> WORLD angular velocity for R = Rz(yaw)Ry(pitch)Rx(roll).
"""
import os, sys
import numpy as np
sys.path.insert(0, "/home/plancher/Desktop/GRiD")
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter

def model(rid, base):
    for c in iter_robot_cases(MANIFEST_PATH, base_mode=base):
        if c["spec"].robot_id == rid:
            return build_project_adapter(c["spec"], resolve_robot_spec(c["spec"]), base_mode=base)
    raise SystemExit(rid)

def qeval(R, jid, q):
    inds = R.get_joint_index_q(jid)
    inds = inds if isinstance(inds, (list, tuple, np.ndarray)) else [inds]
    blk = np.asarray(q)[list(inds)].astype(np.float64)
    return float(blk[0]) if blk.size == 1 else blk

def world_transforms(R, q):
    Xw = {}
    for jid in range(R.get_num_joints()):
        Xloc = np.asarray(R.get_Xmat_hom_Func_by_id(jid)(qeval(R, jid, q)), dtype=np.float64)
        par = R.get_parent_id(jid)
        Xw[jid] = Xloc if par == -1 else Xw[par] @ Xloc
    return Xw

def rpy_from_R(M):
    roll = np.arctan2(M[2,1], M[2,2])
    pitch = np.arctan2(-M[2,0], np.sqrt(M[2,2]**2 + M[2,1]**2))
    yaw = np.arctan2(M[1,0], M[0,0])
    return roll, pitch, yaw

def E_world(roll, pitch, yaw):
    # omega_world = E * [roll_dot, pitch_dot, yaw_dot] for R=Rz(yaw)Ry(pitch)Rx(roll)
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    return np.array([[cy*cp, -sy, 0.0],
                     [sy*cp,  cy, 0.0],
                     [-sp,    0.0, 1.0]])

def geom_ee_grad(model, q):
    R = model.robot
    Xw = world_transforms(R, q)
    # default EE = leaf joint(s); offset = origin [0,0,0,1]
    leaves = R.get_leaf_nodes()
    cols_per_dof = []   # list of (vidx, Jv(3), Jw(3))
    out = []
    for ee in leaves:
        Xee = Xw[ee]
        p_ee = Xee[:3, 3]
        # ancestor chain (incl ee)
        chain = sorted(R.get_ancestors_by_id(ee)) + [ee]
        nv = R.get_num_vel()
        Jv = np.zeros((3, nv)); Jw = np.zeros((3, nv))
        # velocity-index assignment: base 6 (if floating) then joints; derive via S columns.
        # We map each joint's DOF to a velocity index using get_joint_index_v if present,
        # else fall back to q-index (fixed base: v==q).
        for j in chain:
            S = np.asarray(R.get_S_by_id(j), dtype=np.float64)
            if S.ndim == 1: S = S.reshape(-1, 1)
            Rj = Xw[j][:3, :3]; p_j = Xw[j][:3, 3]
            # velocity indices for this joint
            try:
                vinds = R.get_joint_index_v(j)
            except Exception:
                vinds = R.get_joint_index_q(j)
            vinds = vinds if isinstance(vinds, (list, tuple, np.ndarray)) else [vinds]
            vinds = list(vinds)
            ncols = S.shape[1]
            for c in range(ncols):
                col = S[:, c]
                vi = vinds[c] if c < len(vinds) else vinds[-1]
                ax_local = col[:3]; lin_local = col[3:6]
                if np.linalg.norm(ax_local) > 0.5:   # revolute
                    aw = Rj @ ax_local
                    Jw[:, vi] = aw
                    Jv[:, vi] = np.cross(aw, p_ee - p_j)
                else:                                 # prismatic
                    vw = Rj @ lin_local
                    Jv[:, vi] = vw
        roll, pitch, yaw = rpy_from_R(Xee[:3, :3])
        Einv = np.linalg.inv(E_world(roll, pitch, yaw))
        Jrpy = Einv @ Jw
        out.append(np.vstack([Jv, Jrpy]))   # 6 x nv
    return out

if __name__ == "__main__":
    rid = sys.argv[1] if len(sys.argv) > 1 else "iiwa14"
    base = sys.argv[2] if len(sys.argv) > 2 else "fixed"
    m = model(rid, base)
    rng = np.random.default_rng(0)
    nq = m.robot.get_num_pos()
    print(f"=== {rid}-{base} nq={nq} nv={m.robot.get_num_vel()} ===")
    for scale in [0.3, 0.8]:
        q = rng.uniform(-scale, scale, size=nq)
        if base == "floating":  # normalize quaternion part if present (q[3:7] convention)
            pass
        ref = m.reference.end_effector_pose_gradient(q)[0]   # 6 x nq
        geo = geom_ee_grad(m, q)[0]                            # 6 x nv
        # compare on shared columns: fixed base nq==nv -> full compare
        ncmp = min(ref.shape[1], geo.shape[1])
        if ref.shape == geo.shape:
            err = np.max(np.abs(ref - geo))
            print(f"  scale={scale}: shapes match {ref.shape}, max_abs_err={err:.3e}")
        else:
            print(f"  scale={scale}: ref{ref.shape} vs geo{geo.shape} (d/dq vs d/dv mismatch for floating)")

import contextlib
import copy
import io
from dataclasses import dataclass
from typing import List

import numpy as np
from bs4 import BeautifulSoup

from RBDReference import RBDReference
from URDFParser.Joint import Joint
from URDFParser.Robot import Robot
from URDFParser.URDFParser import URDFParser

from test.pinocchio_equivalents.utils.normalization import (
    ConventionMismatch,
    movable_joint_names_excluding_floating_root,
    normalize_matrix,
    normalize_vector,
)


class ProjectParseError(RuntimeError):
    """Raised when the strict GRiD-side parser flow fails."""


@dataclass
class ProjectModelAdapter:
    spec: object
    base_mode: str
    robot: Robot
    reference: RBDReference
    parse_output: str
    mismatches: List[ConventionMismatch]

    @property
    def nq(self) -> int:
        return self.robot.get_num_pos()

    @property
    def nv(self) -> int:
        return self.robot.get_num_vel()

    @property
    def joint_names(self) -> List[str]:
        return [joint.get_name() for joint in self.robot.get_joints_ordered_by_id()]

    @property
    def actuated_joint_names(self) -> List[str]:
        return movable_joint_names_excluding_floating_root(self.base_mode, self.joint_names)

    @property
    def fixed_joint_names(self) -> List[str]:
        return self.robot.get_fixed_joint_names()

    @property
    def joint_types_by_id(self):
        return self.robot.get_joint_types_by_id()

    @property
    def joint_types_by_name(self):
        return self.robot.get_joint_types_by_name()

    def rnea(self, q, qd, qdd):
        c, _v, _a, _f = self.reference.rnea(q, qd, qdd)
        return normalize_vector(c)

    def aba(self, q, qd, tau):
        return normalize_vector(self.reference.aba(q, qd, tau))

    def forward_dynamics(self, q, qd, u):
        return normalize_vector(self.reference.forward_dynamics(q, qd, u))

    def minv(self, q):
        return normalize_matrix(self.reference.minv(q))

    def crba(self, q):
        return normalize_matrix(self.reference.crba(q))

    def rnea_grad(self, q, qd, qdd):
        dc_du = normalize_matrix(self.reference.rnea_grad(q, qd, qdd))
        return dc_du[:, : self.nv], dc_du[:, self.nv :]

    def forward_dynamics_grad(self, q, qd, u):
        dqdd_dq, dqdd_dqd = self.reference.forward_dynamics_grad(q, qd, u)
        return normalize_matrix(dqdd_dq), normalize_matrix(dqdd_dqd)

    def idsva_so_body_frame(self, q, qd, qdd):
        d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq = self.reference.idsva_so_body_frame(q, qd, qdd)
        return (
            np.asarray(d2tau_dq, dtype=np.float64),
            np.asarray(d2tau_dqd, dtype=np.float64),
            np.asarray(d2tau_dvdq, dtype=np.float64),
            np.asarray(dM_dq, dtype=np.float64),
        )

    def fdsva_so(self, q, qd, u):
        daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq = self.reference.fdsva_so(q, qd, u)
        return (
            np.asarray(daba_dqdq, dtype=np.float64),
            np.asarray(daba_dvdq, dtype=np.float64),
            np.asarray(daba_dvdv, dtype=np.float64),
            np.asarray(daba_dtdq, dtype=np.float64),
        )

    # ----- Time integrators -----
    # Mirrors TrajoptPlant.integrator math: composes the Python reference
    # forward_dynamics / forward_dynamics_grad / minv into a single
    # `x_{k+1} = integrator(x_k, u_k, dt)` step and its [A | B] gradient.
    # Only fixed-base (nq == nv) supported here; floating-base would need a
    # Lie-group retract on the configuration part.

    def integrator(self, q, qd, u, dt, integrator_type: str = "euler"):
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        qdd1 = np.asarray(self.reference.forward_dynamics(q, qd, u)).reshape(-1)
        n = self.nv
        if integrator_type == "euler":
            x_kp1 = np.concatenate([q + dt * qd, qd + dt * qdd1])
        elif integrator_type in ("semi_implicit_euler", "si_euler"):
            v_kp1 = qd + dt * qdd1
            q_kp1 = q + dt * v_kp1
            x_kp1 = np.concatenate([q_kp1, v_kp1])
        elif integrator_type in ("midpoint", "rk3", "rk4"):
            # TrajoptPlant convention: qd-component of xdot_i is always the
            # original qd (not the stage-i velocity). So q_{k+1} = q + dt*qd
            # for every multi-stage variant, and only qd_{k+1} benefits from
            # the higher-order qdd refinement.
            def fd(pq, pqd):
                return np.asarray(self.reference.forward_dynamics(pq, pqd, u)).reshape(-1)
            if integrator_type == "midpoint":
                c1 = 0.5
                p1q  = q  + c1 * dt * qd
                p1qd = qd + c1 * dt * qdd1
                qdd2 = fd(p1q, p1qd)
                accel = qdd2
            elif integrator_type == "rk3":
                c1, c2 = 0.5, 0.75
                p1q, p1qd = q + c1 * dt * qd, qd + c1 * dt * qdd1
                qdd2 = fd(p1q, p1qd)
                p2q, p2qd = q + c2 * dt * qd, qd + c2 * dt * qdd2
                qdd3 = fd(p2q, p2qd)
                accel = (2.0 / 9.0) * qdd1 + (3.0 / 9.0) * qdd2 + (4.0 / 9.0) * qdd3
            else:  # rk4
                c1, c2, c3 = 0.5, 0.5, 1.0
                p1q, p1qd = q + c1 * dt * qd, qd + c1 * dt * qdd1
                qdd2 = fd(p1q, p1qd)
                p2q, p2qd = q + c2 * dt * qd, qd + c2 * dt * qdd2
                qdd3 = fd(p2q, p2qd)
                p3q, p3qd = q + c3 * dt * qd, qd + c3 * dt * qdd3
                qdd4 = fd(p3q, p3qd)
                accel = (1.0 / 6.0) * qdd1 + (2.0 / 6.0) * qdd2 + (2.0 / 6.0) * qdd3 + (1.0 / 6.0) * qdd4
            x_kp1 = np.concatenate([q + dt * qd, qd + dt * accel])
        else:
            raise ValueError(f"Unknown integrator_type: {integrator_type}")
        return normalize_vector(x_kp1)

    def integrator_gradient(self, q, qd, u, dt, integrator_type: str = "euler"):
        """Return [A | B] of shape (2*nv, 3*nv) — column 0..nv is dq, nv..2nv is dqd, 2nv..3nv is du."""
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        n = self.nv
        I_n = np.eye(n)
        Z_n = np.zeros((n, n))

        def fd_grad_at(pq, pqd):
            """Returns (J_qq, J_qv, Minv_dense) at the given state."""
            J_qq, J_qv = self.reference.forward_dynamics_grad(pq, pqd, u)
            return (np.asarray(J_qq, dtype=np.float64),
                    np.asarray(J_qv, dtype=np.float64),
                    np.asarray(self.reference.minv(pq), dtype=np.float64))

        def fd_at(pq, pqd):
            return np.asarray(self.reference.forward_dynamics(pq, pqd, u), dtype=np.float64).reshape(-1)

        if integrator_type == "euler":
            J_qq, J_qv, Minv = fd_grad_at(q, qd)
            A = np.block([[I_n,           dt * I_n],
                          [dt * J_qq,     I_n + dt * J_qv]])
            B = np.block([[Z_n],
                          [dt * Minv]])
        elif integrator_type in ("semi_implicit_euler", "si_euler"):
            J_qq, J_qv, Minv = fd_grad_at(q, qd)
            dt2 = dt * dt
            A = np.block([[I_n + dt2 * J_qq,  dt * I_n + dt2 * J_qv],
                          [dt * J_qq,         I_n + dt * J_qv]])
            B = np.block([[dt2 * Minv],
                          [dt * Minv]])
        elif integrator_type in ("midpoint", "rk3", "rk4"):
            # Stage-wise chain rule. For each stage i:
            #   p_i.q = q + c_{i-1}*dt*qd, p_i.qd = qd + c_{i-1}*dt*qdd_{i-1}
            #   D_qdd_i = base_i + c_{i-1}*dt * J_qv_i @ D_qdd_{i-1}
            # where base_i[r, c] is J_qq_i[:, c] for c < n,
            #                       c_{i-1}*dt*J_qq_i[:, c-n] + J_qv_i[:, c-n] for c in [n, 2n),
            #                       Minv_i[:, c-2n] for c >= 2n.
            if integrator_type == "midpoint":
                c_list = [0.5]; b_list = [0.0, 1.0]
            elif integrator_type == "rk3":
                c_list = [0.5, 0.75]; b_list = [2.0/9.0, 3.0/9.0, 4.0/9.0]
            else:  # rk4
                c_list = [0.5, 0.5, 1.0]; b_list = [1.0/6.0, 2.0/6.0, 2.0/6.0, 1.0/6.0]
            N = len(b_list)

            # Stage qdd's and per-stage gradients (np arrays).
            qdd_list = []
            D_qdd_list = []
            J_qq_prev, J_qv_prev, Minv_prev = fd_grad_at(q, qd)
            qdd_list.append(fd_at(q, qd))
            D_qdd_1 = np.hstack([J_qq_prev, J_qv_prev, Minv_prev])  # (n, 3n)
            D_qdd_list.append(D_qdd_1)

            for stage_idx in range(1, N):
                c_prev = c_list[stage_idx - 1]
                prev_qdd = qdd_list[-1]
                p_q  = q  + c_prev * dt * qd
                p_qd = qd + c_prev * dt * prev_qdd  # TrajoptPlant: uses prior qdd
                J_qq, J_qv, Minv = fd_grad_at(p_q, p_qd)
                qdd_list.append(fd_at(p_q, p_qd))
                base = np.hstack([J_qq, c_prev * dt * J_qq + J_qv, Minv])
                chain = c_prev * dt * (J_qv @ D_qdd_list[-1])
                D_qdd_list.append(base + chain)

            # Bottom n rows of [A|B] = [0, I, 0] + dt * sum(b_i * D_qdd_i)
            sum_b_D = sum(b * D for b, D in zip(b_list, D_qdd_list))
            bottom = np.hstack([Z_n, I_n, Z_n]) + dt * sum_b_D  # (n, 3n)
            top    = np.hstack([I_n, dt * I_n, Z_n])              # (n, 3n) — Euler-style q update
            return np.vstack([top, bottom])
        else:
            raise ValueError(f"Unknown integrator_type: {integrator_type}")
        return np.hstack([A, B])

    def end_effector_pose(self, q, target_name: str, offset=None):
        if offset is None:
            offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        ee_pose = self.reference.end_effector_pose(
            q,
            ee_joint_names=target_name,
            ee_offsets=[offset],
        )[0]
        return normalize_vector(np.asarray(ee_pose).reshape(-1))

    def end_effector_rotation_matrix(self, q, target_name: str):
        if target_name in self.joint_names:
            joint = self.robot.get_joint_by_name(target_name)
            target_id = joint.get_id()
            xmat_hom = np.eye(4)
            curr_id = target_id
            while curr_id != -1:
                inds_q = self.robot.get_joint_index_q(curr_id)
                curr_x = self.robot.get_Xmat_hom_Func_by_id(curr_id)(q[inds_q])
                xmat_hom = np.matmul(curr_x, xmat_hom)
                curr_id = self.robot.get_parent_id(curr_id)
            return normalize_matrix(np.asarray(xmat_hom[:3, :3], dtype=np.float64))

        fixed_joint = self.robot.get_fixed_joint_by_name(target_name)
        if fixed_joint is None:
            raise ValueError(f"Could not find joint or fixed joint named: {target_name}")
        if fixed_joint.parent_name == -1:
            xmat_hom = fixed_joint.get_transformation_matrix_hom()
        else:
            parent = self.robot.get_joint_by_name(fixed_joint.parent_name)
            xmat_hom = fixed_joint.get_transformation_matrix_hom()
            curr_id = parent.get_id()
            while curr_id != -1:
                inds_q = self.robot.get_joint_index_q(curr_id)
                curr_x = self.robot.get_Xmat_hom_Func_by_id(curr_id)(q[inds_q])
                xmat_hom = np.matmul(curr_x, xmat_hom)
                curr_id = self.robot.get_parent_id(curr_id)
        return normalize_matrix(np.asarray(xmat_hom[:3, :3], dtype=np.float64))

    def end_effector_pose_gradient(self, q, target_name: str, offset=None):
        if offset is None:
            offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        dee_pose = self.reference.end_effector_pose_gradient(
            q,
            ee_joint_names=target_name,
            ee_offsets=[offset],
        )[0]
        return normalize_matrix(dee_pose)

    def end_effector_pose_hessian(self, q, target_name: str, offset=None):
        if offset is None:
            offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        if self.base_mode == "floating":
            d2ee_pose = self.reference.end_effector_pose_hessian(
                q,
                offsets=[offset],
                ee_joint_names=target_name,
            )[0]
            return np.asarray(d2ee_pose, dtype=np.float64)

        target_joint = self.robot.get_joint_by_name(target_name)
        if target_joint is None:
            raise ValueError(f"Hessian helper only supports articulated joint targets, got {target_name}.")
        leaf_ids = self.robot.get_leaf_nodes()
        if target_joint.get_id() not in leaf_ids:
            raise ValueError(
                f"Hessian helper analytic path currently expects a leaf joint target, got {target_name}."
            )
        leaf_index = leaf_ids.index(target_joint.get_id())
        d2ee_pose = self.reference.end_effector_pose_hessian(
            q,
            offsets=[offset],
        )[leaf_index]
        return np.asarray(d2ee_pose, dtype=np.float64)


def build_project_adapter(
    spec,
    resolved_model,
    base_mode: str,
    floating_base_convention: str = "pinocchio",
) -> ProjectModelAdapter:
    floating_base = base_mode == "floating"
    robot, parse_output = strict_parse_robot(
        resolved_model.urdf_path,
        floating_base=floating_base,
        floating_base_convention=floating_base_convention,
    )
    mismatches = [
        ConventionMismatch(
            category="parse_behavior",
            detail="URDFParser.parse() suppresses exceptions and returns None instead of surfacing structured errors.",
        ),
        ConventionMismatch(
            category="joint_order",
            detail="Joint order follows parser-defined DFS order with Pinocchio-style sibling sorting by child subtree name.",
        ),
    ]
    if floating_base:
        mismatches.append(
            ConventionMismatch(
                category="floating_base_quaternion",
                detail=(
                    "GRiD floating-base configurations default to Pinocchio-compatible "
                    "xyzw / [vx, vy, vz, wx, wy, wz] input ordering, with optional legacy parsing."
                ),
            )
        )
    return ProjectModelAdapter(
        spec=spec,
        base_mode=base_mode,
        robot=robot,
        reference=RBDReference(robot),
        parse_output=parse_output,
        mismatches=mismatches,
    )


def strict_parse_robot(
    urdf_path: str,
    floating_base: bool,
    floating_base_convention: str = "pinocchio",
):
    parser = URDFParser()
    output_capture = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_capture):
            Joint.floating_base = floating_base
            with open(urdf_path, "r", encoding="utf-8") as urdf_file:
                parser.soup = BeautifulSoup(urdf_file.read(), "xml").find("robot")
            if parser.soup is None:
                raise ValueError("URDF file did not contain a <robot> root element.")
            parser.robot = Robot(
                parser.soup["name"],
                floating_base,
                True,
                floating_base_convention=floating_base_convention,
            )
            parser.parse_links()
            parser.parse_joints()
            parser.renumber_linksJoints(using_quaternion=True, joint_ordering="pinocchio_order")
            parser.print_joint_order()
            robot = copy.deepcopy(parser.robot)
    except Exception as exc:
        raise ProjectParseError(str(exc)) from exc

    return robot, output_capture.getvalue()

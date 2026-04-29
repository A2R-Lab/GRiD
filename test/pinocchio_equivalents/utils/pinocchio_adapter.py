from dataclasses import dataclass
from typing import List

import numpy as np
from bs4 import BeautifulSoup

from test.pinocchio_equivalents.utils.normalization import (
    ConventionMismatch,
    movable_joint_names_excluding_floating_root,
    normalize_matrix,
    normalize_pin_compatible_quaternion,
    normalize_project_q_for_pin,
    reduce_pinocchio_q_jacobian_to_project,
    normalize_vector,
)


@dataclass
class PinocchioModelAdapter:
    spec: object
    base_mode: str
    model: object
    data: object
    mismatches: List[ConventionMismatch]
    urdf_joint_types_by_name: dict
    urdf_mimic_joint_names: set

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    @property
    def joint_names(self) -> List[str]:
        return [str(name) for name in list(self.model.names)[1:]]

    @property
    def actuated_joint_names(self) -> List[str]:
        return movable_joint_names_excluding_floating_root(self.base_mode, self.joint_names)

    @property
    def frame_names(self) -> List[str]:
        return [frame.name for frame in self.model.frames]

    @property
    def scalar_joint_names(self) -> List[str]:
        return [
            name
            for name in self.actuated_joint_names
            if self.urdf_joint_types_by_name.get(name) != "floating"
        ]

    @property
    def continuous_joint_names(self) -> List[str]:
        return [
            name
            for name in self.actuated_joint_names
            if self.urdf_joint_types_by_name.get(name) == "continuous"
        ]

    @property
    def mimic_joint_names(self) -> List[str]:
        return sorted(self.urdf_mimic_joint_names)

    def has_invertible_mass_matrix(self, q, min_singular_value: float = 1e-12) -> bool:
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        mass = pin.crba(self.model, self.data, q_pin)
        mass = np.asarray(mass, dtype=np.float64)
        mass = 0.5 * (mass + mass.T)
        singular_values = np.linalg.svd(mass, compute_uv=False)
        if singular_values.size == 0:
            return False
        return bool(np.isfinite(singular_values).all() and singular_values[-1] > min_singular_value)

    def rnea(self, q, qd, qdd):
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        qd_pin = np.asarray(qd, dtype=np.float64)
        qdd_pin = np.asarray(qdd, dtype=np.float64)
        tau = pin.rnea(self.model, self.data, q_pin, qd_pin, qdd_pin)
        return normalize_vector(tau)

    def aba(self, q, qd, tau):
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        qd_pin = np.asarray(qd, dtype=np.float64)
        tau_pin = np.asarray(tau, dtype=np.float64)
        qdd = pin.aba(self.model, self.data, q_pin, qd_pin, tau_pin)
        return normalize_vector(qdd)

    def forward_dynamics(self, q, qd, u):
        return self.aba(q, qd, u)

    def minv(self, q):
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        mass = pin.crba(self.model, self.data, q_pin)
        mass = np.asarray(mass, dtype=np.float64)
        mass = 0.5 * (mass + mass.T)
        minv = np.linalg.inv(mass)
        return normalize_matrix(minv)

    def crba(self, q):
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        mass = pin.crba(self.model, self.data, q_pin)
        mass = np.asarray(mass, dtype=np.float64)
        mass = 0.5 * (mass + mass.T)
        return normalize_matrix(mass)

    def rnea_grad(self, q, qd, qdd):
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        qd_pin = np.asarray(qd, dtype=np.float64)
        qdd_pin = np.asarray(qdd, dtype=np.float64)
        pin.computeRNEADerivatives(self.model, self.data, q_pin, qd_pin, qdd_pin)
        return (
            reduce_pinocchio_q_jacobian_to_project(
                np.asarray(self.data.dtau_dq, dtype=np.float64),
                self.base_mode,
                q,
                joint_names=self.scalar_joint_names,
                joint_types_by_name=self.urdf_joint_types_by_name,
            ),
            normalize_matrix(np.asarray(self.data.dtau_dv, dtype=np.float64)),
        )

    def forward_dynamics_grad(self, q, qd, u):
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        qd_pin = np.asarray(qd, dtype=np.float64)
        u_pin = np.asarray(u, dtype=np.float64)
        pin.computeABADerivatives(self.model, self.data, q_pin, qd_pin, u_pin)
        return (
            reduce_pinocchio_q_jacobian_to_project(
                np.asarray(self.data.ddq_dq, dtype=np.float64),
                self.base_mode,
                q,
                joint_names=self.scalar_joint_names,
                joint_types_by_name=self.urdf_joint_types_by_name,
            ),
            normalize_matrix(np.asarray(self.data.ddq_dv, dtype=np.float64)),
        )

    def end_effector_pose(self, q, target_name: str, offset=None):
        import pinocchio as pin

        if offset is None:
            offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        q_pin = normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        pin.forwardKinematics(self.model, self.data, q_pin)
        pin.updateFramePlacements(self.model, self.data)

        frame_id = self.model.getFrameId(target_name)
        placement = self.data.oMf[frame_id]
        point_local = np.asarray(offset[:3], dtype=np.float64)
        point_world = placement.translation + placement.rotation @ point_local
        rot = placement.rotation
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch_temp = np.sqrt(rot[2, 2] * rot[2, 2] + rot[2, 1] * rot[2, 1])
        pitch = np.arctan2(-rot[2, 0], pitch_temp)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        return normalize_vector(np.concatenate((point_world, np.array([roll, pitch, yaw]))))

    def end_effector_rotation_matrix(self, q, target_name: str):
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        pin.forwardKinematics(self.model, self.data, q_pin)
        pin.updateFramePlacements(self.model, self.data)

        if target_name in self.joint_names:
            joint_id = self.model.getJointId(target_name)
            return normalize_matrix(np.asarray(self.data.oMi[joint_id].rotation, dtype=np.float64))

        frame_id = self.model.getFrameId(target_name)
        return normalize_matrix(np.asarray(self.data.oMf[frame_id].rotation, dtype=np.float64))

    def _normalize_project_q_for_pose_differences(self, q):
        q = np.asarray(q, dtype=np.float64).copy()
        if self.base_mode == "floating":
            q[:7] = normalize_pin_compatible_quaternion(q[:7])
        return q

    def end_effector_pose_gradient(self, q, target_name: str, offset=None, step: float = 1e-6):
        if offset is None:
            offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        q = self._normalize_project_q_for_pose_differences(q)
        nq = len(q)
        gradient = np.zeros((6, nq), dtype=np.float64)

        for q_ind in range(nq):
            q_pos = q.copy()
            q_neg = q.copy()
            q_pos[q_ind] += step
            q_neg[q_ind] -= step
            q_pos = self._normalize_project_q_for_pose_differences(q_pos)
            q_neg = self._normalize_project_q_for_pose_differences(q_neg)
            pose_pos = self.end_effector_pose(q_pos, target_name, offset=offset)
            pose_neg = self.end_effector_pose(q_neg, target_name, offset=offset)
            gradient[:, q_ind] = (pose_pos - pose_neg) / (2.0 * step)
        return gradient

    def end_effector_pose_hessian(self, q, target_name: str, offset=None, step: float = 1e-5):
        if offset is None:
            offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        q = self._normalize_project_q_for_pose_differences(q)
        nq = len(q)
        hessian = np.zeros((6, nq, nq), dtype=np.float64)

        base_pose = self.end_effector_pose(q, target_name, offset=offset)
        for q_ind_i in range(nq):
            for q_ind_j in range(q_ind_i, nq):
                if q_ind_i == q_ind_j:
                    q_pos = q.copy()
                    q_neg = q.copy()
                    q_pos[q_ind_i] += step
                    q_neg[q_ind_i] -= step
                    q_pos = self._normalize_project_q_for_pose_differences(q_pos)
                    q_neg = self._normalize_project_q_for_pose_differences(q_neg)
                    pose_pos = self.end_effector_pose(q_pos, target_name, offset=offset)
                    pose_neg = self.end_effector_pose(q_neg, target_name, offset=offset)
                    value = (pose_pos - 2.0 * base_pose + pose_neg) / (step * step)
                else:
                    q_pp = q.copy()
                    q_pm = q.copy()
                    q_mp = q.copy()
                    q_mm = q.copy()
                    q_pp[q_ind_i] += step
                    q_pp[q_ind_j] += step
                    q_pm[q_ind_i] += step
                    q_pm[q_ind_j] -= step
                    q_mp[q_ind_i] -= step
                    q_mp[q_ind_j] += step
                    q_mm[q_ind_i] -= step
                    q_mm[q_ind_j] -= step
                    q_pp = self._normalize_project_q_for_pose_differences(q_pp)
                    q_pm = self._normalize_project_q_for_pose_differences(q_pm)
                    q_mp = self._normalize_project_q_for_pose_differences(q_mp)
                    q_mm = self._normalize_project_q_for_pose_differences(q_mm)
                    pose_pp = self.end_effector_pose(q_pp, target_name, offset=offset)
                    pose_pm = self.end_effector_pose(q_pm, target_name, offset=offset)
                    pose_mp = self.end_effector_pose(q_mp, target_name, offset=offset)
                    pose_mm = self.end_effector_pose(q_mm, target_name, offset=offset)
                    value = (pose_pp - pose_pm - pose_mp + pose_mm) / (4.0 * step * step)
                hessian[:, q_ind_i, q_ind_j] = value
                hessian[:, q_ind_j, q_ind_i] = value
        return hessian


def build_pinocchio_adapter(spec, resolved_model, base_mode: str) -> PinocchioModelAdapter:
    import pinocchio as pin

    with open(resolved_model.urdf_path, "r", encoding="utf-8") as urdf_file:
        soup = BeautifulSoup(urdf_file.read(), "xml").find("robot")
    urdf_joint_types_by_name = {
        joint["name"]: joint["type"]
        for joint in soup.find_all("joint", recursive=False)
    }
    urdf_mimic_joint_names = {
        joint["name"]
        for joint in soup.find_all("joint", recursive=False)
        if joint.find("mimic") is not None
    }

    if base_mode == "floating":
        model = pin.buildModelFromUrdf(
            resolved_model.urdf_path,
            pin.JointModelFreeFlyer(),
        )
        mismatches = [
            ConventionMismatch(
                category="floating_base_quaternion",
                detail="Pinocchio free-flyer uses the same xyzw quaternion ordering as the current GRiD floating-base convention.",
            )
        ]
    else:
        model = pin.buildModelFromUrdf(resolved_model.urdf_path)
        mismatches = []

    data = model.createData()
    return PinocchioModelAdapter(
        spec=spec,
        base_mode=base_mode,
        model=model,
        data=data,
        mismatches=mismatches,
        urdf_joint_types_by_name=urdf_joint_types_by_name,
        urdf_mimic_joint_names=urdf_mimic_joint_names,
    )

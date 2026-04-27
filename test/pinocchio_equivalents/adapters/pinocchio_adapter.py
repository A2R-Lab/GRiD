from dataclasses import dataclass
from typing import List

import numpy as np

from test.pinocchio_equivalents.adapters.normalization import (
    ConventionMismatch,
    movable_joint_names_excluding_floating_root,
    normalize_matrix,
    normalize_project_q_for_pin,
    normalize_vector,
)


@dataclass
class PinocchioModelAdapter:
    spec: object
    base_mode: str
    model: object
    data: object
    mismatches: List[ConventionMismatch]

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

    def rnea(self, q, qd, qdd):
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(self.base_mode, q)
        qd_pin = np.asarray(qd, dtype=np.float64)
        qdd_pin = np.asarray(qdd, dtype=np.float64)
        tau = pin.rnea(self.model, self.data, q_pin, qd_pin, qdd_pin)
        return normalize_vector(tau)

    def minv(self, q):
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(self.base_mode, q)
        mass = pin.crba(self.model, self.data, q_pin)
        mass = np.asarray(mass, dtype=np.float64)
        mass = 0.5 * (mass + mass.T)
        return normalize_matrix(np.linalg.inv(mass))

    def crba(self, q):
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(self.base_mode, q)
        mass = pin.crba(self.model, self.data, q_pin)
        mass = np.asarray(mass, dtype=np.float64)
        return normalize_matrix(0.5 * (mass + mass.T))

    def rnea_grad(self, q, qd, qdd):
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(self.base_mode, q)
        qd_pin = np.asarray(qd, dtype=np.float64)
        qdd_pin = np.asarray(qdd, dtype=np.float64)
        pin.computeRNEADerivatives(self.model, self.data, q_pin, qd_pin, qdd_pin)
        return (
            normalize_matrix(np.asarray(self.data.dtau_dq, dtype=np.float64)),
            normalize_matrix(np.asarray(self.data.dtau_dv, dtype=np.float64)),
        )

    def end_effector_pose(self, q, target_name: str, offset=None):
        import pinocchio as pin

        if offset is None:
            offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        q_pin = normalize_project_q_for_pin(self.base_mode, q)
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


def build_pinocchio_adapter(spec, resolved_model, base_mode: str) -> PinocchioModelAdapter:
    import pinocchio as pin

    if base_mode == "floating":
        model = pin.buildModelFromUrdf(
            resolved_model.urdf_path,
            pin.JointModelFreeFlyer(),
        )
        mismatches = [
            ConventionMismatch(
                category="floating_base_quaternion",
                detail="Pinocchio free-flyer uses xyzw quaternion ordering, so GRiD states must be reordered before comparison.",
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
    )

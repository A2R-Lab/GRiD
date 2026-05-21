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
    urdf_path: str = ""

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

    def idsva_so_body_frame(self, q, qd, qdd):
        """Return (d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq) from Pinocchio's
        C++ `ComputeRNEASecondOrderDerivatives`, in our nv-indexed Lie-tangent
        convention.

        Pinocchio's mixed tensor `d2tau_dqdv[i,j,k] = d2tau_i/(dq_j dv_k)` is
        transposed on axes (1,2) to match our `d2tau_dvdq[i,j,k] = d2tau_i/(dv_j dq_k)`.
        """
        from test.pinocchio_equivalents.pin_so_ext import load as _load_so

        ext = _load_so()
        q_pin = normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        qd_arr = np.asarray(qd, dtype=np.float64)
        qdd_arr = np.asarray(qdd, dtype=np.float64)
        d2tau_dqdq, d2tau_dvdv, d2tau_dqdv, d2tau_dadq = ext.compute_rnea_second_order(
            self.urdf_path,
            self.base_mode == "floating",
            np.asarray(q_pin, dtype=np.float64),
            qd_arr,
            qdd_arr,
        )
        d2tau_dvdq = np.asarray(d2tau_dqdv, dtype=np.float64).transpose(0, 2, 1)
        return (
            np.asarray(d2tau_dqdq, dtype=np.float64),
            np.asarray(d2tau_dvdv, dtype=np.float64),
            d2tau_dvdq,
            np.asarray(d2tau_dadq, dtype=np.float64),
        )

    def fdsva_so(self, q, qd, u):
        """Return (daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq) composed from
        Pinocchio's second-order RNEA + first-order Minv/ABA derivatives.

        Uses the same composition formula as `RBDReference.fdsva_so`, but with
        every input grounded in Pinocchio's bound C++ implementations so the
        result is independent of our analytic code path.
        """
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        qd_arr = np.asarray(qd, dtype=np.float64)
        u_arr = np.asarray(u, dtype=np.float64)
        # Pinocchio ABA gives qdd and fills first-order derivatives.
        qdd = np.asarray(pin.aba(self.model, self.data, q_pin, qd_arr, u_arr), dtype=np.float64)
        d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq = self.idsva_so_body_frame(q, qd, qdd)
        pin.computeMinverse(self.model, self.data, q_pin)
        Minv = normalize_matrix(np.asarray(self.data.Minv, dtype=np.float64))
        fd_dq, fd_dqd = self.forward_dynamics_grad(q, qd, u)
        daba_dqdq = -np.einsum(
            "il,ljk->ijk",
            Minv,
            d2tau_dq
            + np.einsum("ilk,lj->ijk", dM_dq, fd_dq)
            + np.einsum("ilk,lj->ikj", dM_dq, fd_dq),
        )
        daba_dvdq = -np.einsum(
            "il,ljk->ijk",
            Minv,
            d2tau_dvdq + np.einsum("ilk,lj->ijk", dM_dq, fd_dqd),
        )
        daba_dvdv = -np.einsum("il,ljk->ijk", Minv, d2tau_dqd)
        daba_dtdq = -np.einsum(
            "il,ljk->ijk",
            Minv,
            np.einsum("ilk,lj->ijk", dM_dq, Minv),
        )
        return daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq

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

    # ----- Time integrators (canonical via pinocchio.integrate / dIntegrate) -----

    def _to_pin_q(self, q):
        return normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )

    def _pin_integrate(self, q, v_dt):
        """`pin.integrate(model, q, v_dt)` mapped through the project's q
        convention. v_dt is in tangent space (size nv) and passes through
        directly (no conversion needed). Output is in project nq layout for
        the floating-base prefix (xyzw quaternion + joints with no
        continuous joints in our manifest).
        """
        import pinocchio as pin

        q_pin = self._to_pin_q(q)
        q_new_pin = np.asarray(
            pin.integrate(self.model, q_pin, np.asarray(v_dt, dtype=np.float64)),
            dtype=np.float64,
        )
        # The pinocchio q layout matches the project q layout for revolute +
        # free-flyer robots (no continuous joints in the manifest), so we can
        # pass through. Renormalize the quaternion prefix for safety.
        if self.base_mode == "floating":
            # normalize_pin_compatible_quaternion expects the full pose prefix,
            # so renormalize in place via that helper.
            q_new_pin = normalize_pin_compatible_quaternion(q_new_pin)
        return q_new_pin

    def _pin_dIntegrate(self, q, v_dt, with_respect_to):
        """Wrap `pin.dIntegrate` and return the (nv, nv) Jacobian."""
        import pinocchio as pin

        q_pin = self._to_pin_q(q)
        arg = pin.ArgumentPosition.ARG0 if with_respect_to == "q" else pin.ArgumentPosition.ARG1
        J = np.asarray(
            pin.dIntegrate(self.model, q_pin, np.asarray(v_dt, dtype=np.float64), arg),
            dtype=np.float64,
        )
        return J

    @staticmethod
    def _butcher(integrator_type: str):
        if integrator_type == "euler":
            return [], [1.0]
        if integrator_type in ("semi_implicit_euler", "si_euler"):
            return [], [1.0]
        if integrator_type == "midpoint":
            return [0.5], [0.0, 1.0]
        if integrator_type == "rk3":
            return [0.5, 0.75], [2.0 / 9.0, 3.0 / 9.0, 4.0 / 9.0]
        if integrator_type == "rk4":
            return [0.5, 0.5, 1.0], [1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0]
        raise ValueError(f"Unknown integrator_type: {integrator_type}")

    def integrator(self, q, qd, u, dt, integrator_type: str = "euler"):
        """Pinocchio-backed integrator step. Uses `pin.integrate` for the
        Lie-group q update and `pin.aba` per stage for the qdd refinement.
        Output shape: nq + nv concatenated as [q_new, v_new]."""
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        qdd1 = self.aba(q, qd, u)
        if integrator_type == "euler":
            q_new = self._pin_integrate(q, dt * qd)
            v_new = qd + dt * qdd1
            return np.concatenate([q_new, v_new])
        if integrator_type in ("semi_implicit_euler", "si_euler"):
            v_new = qd + dt * qdd1
            q_new = self._pin_integrate(q, dt * v_new)
            return np.concatenate([q_new, v_new])
        c_list, b_list = self._butcher(integrator_type)
        N = len(b_list)
        qdd_list = [qdd1]
        prev_qdd = qdd1
        for stage_idx in range(1, N):
            c_prev = c_list[stage_idx - 1]
            p_q = self._pin_integrate(q, c_prev * dt * qd)
            p_qd = qd + c_prev * dt * prev_qdd
            stage_qdd = self.aba(p_q, p_qd, u)
            qdd_list.append(stage_qdd)
            prev_qdd = stage_qdd
        accel = sum(b * qdd for b, qdd in zip(b_list, qdd_list))
        q_new = self._pin_integrate(q, dt * qd)
        v_new = qd + dt * accel
        return np.concatenate([q_new, v_new])

    def integrator_gradient(self, q, qd, u, dt, integrator_type: str = "euler"):
        """Pinocchio-backed integrator gradient [A|B] of shape (2*nv, 3*nv).

        Uses `pin.dIntegrate` for the q-side blocks and `pin.computeABADerivatives`
        for the qdd partials. The chain rule across multi-stage variants
        mirrors the form in `RBDReference.integrator_grad`.
        """
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        nv = self.nv
        I_n = np.eye(nv)
        Z_n = np.zeros((nv, nv))

        def fd_grad_at(pq, pqd):
            J_qq, J_qv = self.forward_dynamics_grad(pq, pqd, u)
            return np.asarray(J_qq, dtype=np.float64), np.asarray(J_qv, dtype=np.float64), self.minv(pq)

        def q_top_blocks(v_dt_arg):
            return (self._pin_dIntegrate(q, v_dt_arg, "q"),
                    self._pin_dIntegrate(q, v_dt_arg, "v"))

        if integrator_type == "euler":
            J_qq, J_qv, Minv = fd_grad_at(q, qd)
            dInt_q, dInt_v = q_top_blocks(dt * qd)
            top = np.hstack([dInt_q, dt * dInt_v, Z_n])
            bottom = np.hstack([dt * J_qq, I_n + dt * J_qv, dt * Minv])
            return np.vstack([top, bottom])
        if integrator_type in ("semi_implicit_euler", "si_euler"):
            J_qq, J_qv, Minv = fd_grad_at(q, qd)
            v_new = qd + dt * self.aba(q, qd, u)
            dInt_q, dInt_v = q_top_blocks(dt * v_new)
            dvdq = dt * J_qq
            dvdv = I_n + dt * J_qv
            dvdu = dt * Minv
            dt_dInt_v = dt * dInt_v
            top = np.hstack([dInt_q + dt_dInt_v @ dvdq,
                              dt_dInt_v @ dvdv,
                              dt_dInt_v @ dvdu])
            bottom = np.hstack([dvdq, dvdv, dvdu])
            return np.vstack([top, bottom])
        c_list, b_list = self._butcher(integrator_type)
        N = len(b_list)
        qdd_list = []
        D_qdd_list = []
        qdd_list.append(self.aba(q, qd, u))
        J_qq, J_qv, Minv = fd_grad_at(q, qd)
        D_qdd_list.append(np.hstack([J_qq, J_qv, Minv]))
        for stage_idx in range(1, N):
            c_prev = c_list[stage_idx - 1]
            prev_qdd = qdd_list[-1]
            p_q = self._pin_integrate(q, c_prev * dt * qd)
            p_qd = qd + c_prev * dt * prev_qdd
            qdd_list.append(self.aba(p_q, p_qd, u))
            J_qq_i, J_qv_i, Minv_i = fd_grad_at(p_q, p_qd)
            v_dt_stage = c_prev * dt * qd
            dInt_q_stage, dInt_v_stage = q_top_blocks(v_dt_stage)
            dp_q_block = np.hstack([dInt_q_stage, c_prev * dt * dInt_v_stage, Z_n])
            dp_qd_block = np.hstack([Z_n, I_n, Z_n]) + c_prev * dt * D_qdd_list[-1]
            d_u_block = np.hstack([Z_n, Z_n, Minv_i])
            D_qdd_list.append(J_qq_i @ dp_q_block + J_qv_i @ dp_qd_block + d_u_block)
        sum_b_D = sum(b * D for b, D in zip(b_list, D_qdd_list))
        dInt_q_final, dInt_v_final = q_top_blocks(dt * qd)
        top = np.hstack([dInt_q_final, dt * dInt_v_final, Z_n])
        bottom = np.hstack([Z_n, I_n, Z_n]) + dt * sum_b_D
        return np.vstack([top, bottom])

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
        urdf_path=str(resolved_model.urdf_path),
    )

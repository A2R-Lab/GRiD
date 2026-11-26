"""Convert states between GRiD's coordinates and those of other libraries.

GRiD, Pinocchio and MuJoCo store the same physical state differently. The
differences and the reasons for them are in the documentation page
"Coordinate conventions across libraries". This module holds the maps as
numpy functions. It needs no GPU and no compiler, and it does not import
Pinocchio or MuJoCo.

Pinocchio
    ``PinocchioMap(descriptor, pin_model)`` pairs a GRiD model descriptor
    (``grid_rbd.describe_model`` or ``handle.model_descriptor``) with a
    Pinocchio model of the same robot. Joints are paired by name, so the two
    libraries may order them differently. The root is paired by structure: a
    GRiD floating base with a Pinocchio free-flyer root joint, and a GRiD
    planar base with a Pinocchio planar root joint.

MuJoCo
    ``q_to_mujoco``, ``v_to_mujoco`` and their inverses convert the floating
    base block. Every other entry is the same in both libraries when the
    joints are in the same order, which these functions assume.

All functions take and return one state: 1-D float64 arrays.
"""
from __future__ import annotations

import numpy as np

from . import _mujoco

__all__ = [
    "PinocchioMap",
    "q_to_mujoco",
    "q_from_mujoco",
    "v_to_mujoco",
    "v_from_mujoco",
]

_SCALAR_KINDS = ("revolute", "continuous", "prismatic", "helical", "screw")


def _vector(value, size, name):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {out.shape}")
    return out


def _planar_base(descriptor):
    # grid_codegen is a dependency of describe_model already; import it late so
    # importing this module stays cheap.
    from grid_codegen.model_descriptor import planar_base_block  # noqa: PLC0415

    return planar_base_block(descriptor) is not None


class PinocchioMap:
    """The coordinate maps between one GRiD model and one Pinocchio model.

    ``q`` and ``v`` are GRiD's configuration and velocity, ``q_pin`` and
    ``v_pin`` Pinocchio's. The velocity map is linear, ``v_pin = P(q) v``, and
    ``velocity_matrix`` returns ``P``. It is the identity up to a permutation
    except for a planar base, where it rotates the two translation rates from
    the parent frame into the body frame, and for mimic joints, where a
    follower's column is folded into its leader's.

    A mimic follower owns no GRiD coordinate. Converting to Pinocchio writes
    ``multiplier * leader + offset`` into the follower's slot. Converting from
    Pinocchio reads the leaders and ignores the followers' slots.

    A continuous joint is an angle in GRiD and ``(cos, sin)`` in a Pinocchio
    model built from a URDF. ``q_from_pinocchio`` returns the angle in
    ``(-pi, pi]``: Pinocchio's storage does not keep the number of turns.
    """

    def __init__(self, descriptor, pin_model):
        if not isinstance(descriptor, dict) or "joints" not in descriptor:
            raise TypeError("descriptor must be a GRiD model descriptor (grid_rbd.describe_model)")
        self.nq, self.nv = int(descriptor["nq"]), int(descriptor["nv"])
        self.nq_pin, self.nv_pin = int(pin_model.nq), int(pin_model.nv)
        names = [str(name) for name in pin_model.names]
        joints = descriptor["joints"]
        planar = _planar_base(descriptor)
        # Each entry: (kind, GRiD q start, GRiD v start, Pinocchio q start,
        # Pinocchio v start, wraps, multiplier, offset).
        self._entries = []
        used_q, used_v = set(), set()

        def claim(joint_id, nq, nv, what):
            joint = pin_model.joints[joint_id]
            if (int(joint.nq), int(joint.nv)) not in nq_nv_allowed[what]:
                raise ValueError(
                    f"Pinocchio joint '{names[joint_id]}' has nq={joint.nq}, nv={joint.nv}, "
                    f"which does not match GRiD's {what}")
            used_q.update(range(int(joint.idx_q), int(joint.idx_q) + int(joint.nq)))
            used_v.update(range(int(joint.idx_v), int(joint.idx_v) + int(joint.nv)))
            return joint

        nq_nv_allowed = {
            "floating base": {(7, 6)}, "planar base": {(4, 3)},
            "scalar joint": {(1, 1), (2, 1)}, "spherical joint": {(4, 3)},
        }
        rest = joints
        if planar:
            if len(pin_model.joints) < 2:
                raise ValueError("the Pinocchio model has no root joint for GRiD's planar base")
            root = claim(1, 4, 3, "planar base")
            self._entries.append(("planar", 0, 0, int(root.idx_q), int(root.idx_v), True, 1.0, 0.0))
            rest = joints[3:]
        for joint in rest:
            kind, name = joint["kind"], joint["name"]
            if kind == "floating":
                if joint["parent_body"] != -1:
                    raise ValueError(f"joint '{name}': only a floating root joint can be converted")
                if len(pin_model.joints) < 2:
                    raise ValueError("the Pinocchio model has no root joint for GRiD's floating base")
                root = claim(1, 7, 6, "floating base")
                self._entries.append(("floating", joint["q_start"], joint["v_start"],
                                      int(root.idx_q), int(root.idx_v), False, 1.0, 0.0))
                continue
            if name not in names:
                raise ValueError(
                    f"GRiD joint '{name}' has no joint of that name in the Pinocchio model. A planar "
                    "or floating joint that is not the root is split into scalar joints by GRiD and "
                    "cannot be paired.")
            joint_id = names.index(name)
            if kind == "spherical":
                target = claim(joint_id, 4, 3, "spherical joint")
                self._entries.append(("spherical", joint["q_start"], joint["v_start"],
                                      int(target.idx_q), int(target.idx_v), False, 1.0, 0.0))
                continue
            if kind not in _SCALAR_KINDS:
                raise ValueError(f"joint '{name}': kind '{kind}' cannot be converted")
            target = claim(joint_id, 1, 1, "scalar joint")
            wraps = int(target.nq) == 2
            mimic = joint.get("mimic")
            if mimic is None:
                q_start, v_start, multiplier, offset = joint["q_start"], joint["v_start"], 1.0, 0.0
            elif mimic["leader_joint"] is None:  # a constant joint reads no input
                q_start, v_start, multiplier, offset = None, None, 0.0, float(mimic["offset"])
            else:
                leader = joints[mimic["leader_joint"]]
                q_start, v_start = leader["q_start"], leader["v_start"]
                multiplier, offset = float(mimic["multiplier"]), float(mimic["offset"])
            self._entries.append(("follower" if mimic is not None else "scalar", q_start, v_start,
                                  int(target.idx_q), int(target.idx_v), wraps, multiplier, offset))
        if len(used_q) != self.nq_pin or len(used_v) != self.nv_pin:
            raise ValueError(
                f"the Pinocchio model has nq={self.nq_pin}, nv={self.nv_pin}, of which the GRiD model "
                f"accounts for {len(used_q)} and {len(used_v)}. The two models are not the same robot.")
        self.has_mimic = any(entry[0] == "follower" for entry in self._entries)

    # ---- configuration --------------------------------------------------

    def q_to_pinocchio(self, q):
        q = _vector(q, self.nq, "q")
        out = np.zeros(self.nq_pin)
        for kind, iq, _iv, jq, _jv, wraps, multiplier, offset in self._entries:
            if kind == "floating":
                out[jq:jq + 7] = q[iq:iq + 7]
            elif kind == "spherical":
                out[jq:jq + 4] = q[iq:iq + 4]
            elif kind == "planar":
                out[jq:jq + 4] = [q[iq], q[iq + 1], np.cos(q[iq + 2]), np.sin(q[iq + 2])]
            else:
                value = offset if iq is None else multiplier * q[iq] + offset
                out[jq:jq + (2 if wraps else 1)] = [np.cos(value), np.sin(value)] if wraps else [value]
        return out

    def q_from_pinocchio(self, q_pin):
        q_pin = _vector(q_pin, self.nq_pin, "q_pin")
        out = np.zeros(self.nq)
        for kind, iq, _iv, jq, _jv, wraps, _multiplier, _offset in self._entries:
            if kind == "floating":
                out[iq:iq + 7] = q_pin[jq:jq + 7]
            elif kind == "spherical":
                out[iq:iq + 4] = q_pin[jq:jq + 4]
            elif kind == "planar":
                out[iq:iq + 3] = [q_pin[jq], q_pin[jq + 1], np.arctan2(q_pin[jq + 3], q_pin[jq + 2])]
            elif kind == "scalar":
                out[iq] = np.arctan2(q_pin[jq + 1], q_pin[jq]) if wraps else q_pin[jq]
        return out

    def configuration_jacobian(self, q):
        """``d q_pin / d q`` as a coordinate Jacobian, shape ``(nq_pin, nq)``.

        It chains a derivative taken with respect to Pinocchio's stored
        configuration back to GRiD's. The quaternion blocks are identities."""
        q = _vector(q, self.nq, "q")
        out = np.zeros((self.nq_pin, self.nq))
        for kind, iq, _iv, jq, _jv, wraps, multiplier, offset in self._entries:
            if kind in ("floating", "spherical"):
                width = 7 if kind == "floating" else 4
                out[jq:jq + width, iq:iq + width] = np.eye(width)
            elif kind == "planar":
                out[jq, iq] = out[jq + 1, iq + 1] = 1.0
                out[jq + 2, iq + 2], out[jq + 3, iq + 2] = -np.sin(q[iq + 2]), np.cos(q[iq + 2])
            elif iq is not None:
                value = multiplier * q[iq] + offset
                if wraps:
                    out[jq, iq], out[jq + 1, iq] = -multiplier * np.sin(value), multiplier * np.cos(value)
                else:
                    out[jq, iq] = multiplier
        return out

    # ---- velocity, acceleration, force ----------------------------------

    def velocity_matrix(self, q):
        """``P(q)`` with ``v_pin = P v``, shape ``(nv_pin, nv)``."""
        q = _vector(q, self.nq, "q")
        out = np.zeros((self.nv_pin, self.nv))
        for kind, iq, iv, _jq, jv, _wraps, multiplier, _offset in self._entries:
            if kind == "floating":
                out[jv:jv + 6, iv:iv + 6] = np.eye(6)
            elif kind == "spherical":
                out[jv:jv + 3, iv:iv + 3] = np.eye(3)
            elif kind == "planar":
                c, s = np.cos(q[iq + 2]), np.sin(q[iq + 2])
                out[jv:jv + 3, iv:iv + 3] = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
            elif iv is not None:
                out[jv, iv] = multiplier
        return out

    def velocity_matrix_rate(self, q, v):
        """The time derivative of ``P`` along ``v``. Zero without a planar base."""
        q, v = _vector(q, self.nq, "q"), _vector(v, self.nv, "v")
        out = np.zeros((self.nv_pin, self.nv))
        for kind, iq, iv, _jq, jv, _wraps, _multiplier, _offset in self._entries:
            if kind == "planar":
                c, s = np.cos(q[iq + 2]), np.sin(q[iq + 2])
                out[jv:jv + 2, iv:iv + 2] = v[iv + 2] * np.array([[-s, c], [-c, -s]])
        return out

    def _from_pinocchio_tangent(self, q, tangent, name):
        q, tangent = _vector(q, self.nq, "q"), _vector(tangent, self.nv_pin, name)
        out = np.zeros(self.nv)
        for kind, iq, iv, _jq, jv, _wraps, _multiplier, _offset in self._entries:
            if kind == "floating":
                out[iv:iv + 6] = tangent[jv:jv + 6]
            elif kind == "spherical":
                out[iv:iv + 3] = tangent[jv:jv + 3]
            elif kind == "planar":
                c, s = np.cos(q[iq + 2]), np.sin(q[iq + 2])
                out[iv:iv + 3] = [c * tangent[jv] - s * tangent[jv + 1],
                                  s * tangent[jv] + c * tangent[jv + 1], tangent[jv + 2]]
            elif kind == "scalar":
                out[iv] = tangent[jv]
        return out

    def v_to_pinocchio(self, q, v):
        return self.velocity_matrix(q) @ _vector(v, self.nv, "v")

    def v_from_pinocchio(self, q, v_pin):
        """GRiD's velocity for a Pinocchio velocity; ``q`` is GRiD's configuration."""
        return self._from_pinocchio_tangent(q, v_pin, "v_pin")

    def a_to_pinocchio(self, q, v, a):
        """Pinocchio's acceleration, ``P a + (dP/dt) v``, from GRiD's ``q, v, a``."""
        return (self.velocity_matrix(q) @ _vector(a, self.nv, "a")
                + self.velocity_matrix_rate(q, v) @ _vector(v, self.nv, "v"))

    def a_from_pinocchio(self, q, v, a_pin):
        """GRiD's acceleration; ``q`` and ``v`` are GRiD's configuration and velocity."""
        a_pin = _vector(a_pin, self.nv_pin, "a_pin")
        return self._from_pinocchio_tangent(
            q, a_pin - self.velocity_matrix_rate(q, v) @ _vector(v, self.nv, "v"), "a_pin")

    def tau_from_pinocchio(self, q, tau_pin):
        """GRiD's generalized force, ``P^T tau_pin``. With mimic joints this
        sums each follower's force into its leader."""
        return self.velocity_matrix(q).T @ _vector(tau_pin, self.nv_pin, "tau_pin")

    def tau_to_pinocchio(self, q, tau):
        """Pinocchio's generalized force for GRiD's. Not defined with mimic
        joints, where one leader force has no unique split over its followers."""
        if self.has_mimic:
            raise ValueError("tau_to_pinocchio is not defined for a model with mimic joints")
        # Without mimic joints P is orthogonal, so its inverse transpose is P.
        return self.velocity_matrix(q) @ _vector(tau, self.nv, "tau")


def _row(value, name):
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be one state (a 1-D array), got shape {out.shape}")
    return out[None, :]


def q_to_mujoco(q, floating_base=True):
    """GRiD's configuration with the base quaternion reordered to ``wxyz``."""
    return _mujoco.q_pin_to_mjx(_row(q, "q"), floating_base)[0]


def q_from_mujoco(q_mujoco, floating_base=True):
    """A MuJoCo configuration with the base quaternion reordered to ``xyzw``."""
    return _mujoco.q_mjx_to_pin(_row(q_mujoco, "q_mujoco"), floating_base)[0]


def v_to_mujoco(q, v, floating_base=True):
    """GRiD's velocity with the base linear part rotated into the world frame.

    ``q`` is GRiD's configuration (``xyzw``)."""
    q, v = _row(q, "q"), _row(v, "v")
    rotation = _mujoco.base_rotation(q) if floating_base else None
    return _mujoco.v_pin_to_mjx(v, rotation, floating_base)[0]


def v_from_mujoco(q, v_mujoco, floating_base=True):
    """A MuJoCo velocity with the base linear part rotated into the body frame.

    ``q`` is GRiD's configuration (``xyzw``)."""
    q, v_mujoco = _row(q, "q"), _row(v_mujoco, "v_mujoco")
    rotation = _mujoco.base_rotation(q) if floating_base else None
    return _mujoco.v_mjx_to_pin(v_mujoco, rotation, floating_base)[0]

"""Shared construction and mutation rules for reduced-coordinate armature."""
import numpy as np


def supported_slots(robot):
    slots = []
    for jid in range(robot.get_num_joints()):
        if ((robot.floating_base and jid == 0) or robot.joint_is_spherical(jid)
                or robot.get_joint_by_id(jid).is_mimic):
            continue
        slot = robot.get_joint_index_v(jid)
        if not 0 <= int(slot) < robot.get_num_vel():
            raise ValueError(f"Independent joint {jid} has invalid armature slot {slot}.")
        slots.append(int(slot))
    return sorted(set(slots))


def validated_armature(values, nv, slots, dtype):
    raw = np.asarray(values)
    if raw.shape != (nv,) or raw.dtype.kind not in "biuf":
        raise ValueError(f"armature must be a real vector of shape ({nv},); got {raw.shape} ({raw.dtype}).")
    if not np.all(np.isfinite(raw)) or np.any(raw < 0):
        raise ValueError("armature must be finite and nonnegative in every slot.")
    allowed = np.zeros(nv, dtype=bool)
    allowed[list(slots)] = True
    if np.any(raw[~allowed] != 0):
        raise ValueError("armature must be zero on floating-root and spherical v-slots.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=dtype, order="C", copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("armature must remain finite in the selected precision.")
    result[result == 0] = 0
    return result


def option_vector(values):
    """JSON-safe input, without narrowing to kernel precision before slot checks."""
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("armature must be a one-dimensional vector.")
    result = validated_armature(raw, len(raw), range(len(raw)), np.float64)
    if np.any((raw != 0) & (result == 0)):
        raise ValueError("armature input is too small to preserve in float64 construction metadata.")
    return result.tolist()

"""Armature boundary validation, before and after precision conversion."""
import numpy as np


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
    result[result == 0] = 0  # canonical positive zero for identity/initializers
    return result

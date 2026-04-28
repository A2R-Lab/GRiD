from dataclasses import dataclass
from typing import List

import numpy as np


RNG_SEED = 7


@dataclass(frozen=True)
class DynamicsSample:
    name: str
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray


def _joint_ranges(
    robot, count: int, default_low: float, default_high: float, skip_joint_ids: int = 0
) -> np.ndarray:
    bounds = np.zeros((count, 2), dtype=np.float64)
    default_span = float(default_high - default_low)
    for local_index, jid in enumerate(range(skip_joint_ids, skip_joint_ids + count)):
        limits = robot.get_joint_by_id(jid).get_joint_limits()
        low = default_low
        high = default_high
        if limits:
            raw_low, raw_high = limits
            finite_low = np.isfinite(raw_low)
            finite_high = np.isfinite(raw_high)

            if finite_low and finite_high:
                clipped_low = max(default_low, raw_low)
                clipped_high = min(default_high, raw_high)
                if clipped_low <= clipped_high:
                    low = clipped_low
                    high = clipped_high
                else:
                    span = min(default_span, raw_high - raw_low)
                    midpoint = 0.5 * (raw_low + raw_high)
                    low = midpoint - 0.5 * span
                    high = midpoint + 0.5 * span
            else:
                if finite_low:
                    low = max(default_low, raw_low)
                if finite_high:
                    high = min(default_high, raw_high)
        bounds[local_index, 0] = low
        bounds[local_index, 1] = high
    return bounds


def _make_zero_state(adapter) -> DynamicsSample:
    q = np.zeros(adapter.nq, dtype=np.float64)
    qd = np.zeros(adapter.nv, dtype=np.float64)
    qdd = np.zeros(adapter.nv, dtype=np.float64)
    if adapter.base_mode == "floating":
        q[3] = 1.0
    return DynamicsSample(name="zero", q=q, qd=qd, qdd=qdd)


def _make_conservative_state(adapter, rng: np.random.Generator) -> DynamicsSample:
    q = np.zeros(adapter.nq, dtype=np.float64)
    qd = rng.uniform(-1.0, 1.0, size=adapter.nv).astype(np.float64)
    qdd = rng.uniform(-2.0, 2.0, size=adapter.nv).astype(np.float64)

    if adapter.base_mode == "floating":
        q[0:3] = rng.uniform(-0.25, 0.25, size=3)
        quat_xyzw = rng.uniform(-1.0, 1.0, size=4)
        quat_xyzw /= np.linalg.norm(quat_xyzw)
        q[3:7] = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
        )
        joint_offset = 7
        joint_count = adapter.nq - joint_offset
        skip_joint_ids = 1
    else:
        joint_offset = 0
        joint_count = adapter.nq
        skip_joint_ids = 0

    if joint_count:
        bounds = _joint_ranges(
            adapter.robot,
            joint_count,
            -0.5,
            0.5,
            skip_joint_ids=skip_joint_ids,
        )
        q[joint_offset:] = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float64)

    return DynamicsSample(name="conservative", q=q, qd=qd, qdd=qdd)


def build_dynamics_samples(adapter) -> List[DynamicsSample]:
    rng = np.random.default_rng(RNG_SEED)
    return [_make_zero_state(adapter), _make_conservative_state(adapter, rng)]

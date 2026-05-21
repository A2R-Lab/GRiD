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
        q[6] = 1.0
    return DynamicsSample(name="zero", q=q, qd=qd, qdd=qdd)


def _make_conservative_state(adapter, rng: np.random.Generator) -> DynamicsSample:
    q = np.zeros(adapter.nq, dtype=np.float64)
    qd = rng.uniform(-1.0, 1.0, size=adapter.nv).astype(np.float64)
    qdd = rng.uniform(-2.0, 2.0, size=adapter.nv).astype(np.float64)

    if adapter.base_mode == "floating":
        q[0:3] = rng.uniform(-0.25, 0.25, size=3)
        quat_xyzw = rng.uniform(-1.0, 1.0, size=4)
        quat_xyzw /= np.linalg.norm(quat_xyzw)
        q[3:7] = quat_xyzw.astype(np.float64)
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


def _make_energetic_state(
    adapter, rng: np.random.Generator, name: str, qd_scale: float, qdd_scale: float,
    q_span: float = 1.0,
) -> DynamicsSample:
    """A configuration within joint limits but with large velocity and/or
    acceleration. High |qd| / |qdd| exercise the velocity- and acceleration-
    coupling terms (crm/crf spatial products and their gradients) that low-
    energy samples leave near zero — exactly the regime where coupling bugs
    hide (e.g. the floating-base J_qv root angular↔linear coupling found via
    the integrator). Floating-base base velocity (qd[0:6] = [omega, v_lin]) is
    scaled too, so the free-flyer gyroscopic coupling is exercised."""
    q = np.zeros(adapter.nq, dtype=np.float64)
    qd = rng.uniform(-qd_scale, qd_scale, size=adapter.nv).astype(np.float64)
    qdd = rng.uniform(-qdd_scale, qdd_scale, size=adapter.nv).astype(np.float64)

    if adapter.base_mode == "floating":
        q[0:3] = rng.uniform(-0.25, 0.25, size=3)
        quat_xyzw = rng.uniform(-1.0, 1.0, size=4)
        quat_xyzw /= np.linalg.norm(quat_xyzw)
        q[3:7] = quat_xyzw.astype(np.float64)
        joint_offset = 7
        joint_count = adapter.nq - joint_offset
        skip_joint_ids = 1
    else:
        joint_offset = 0
        joint_count = adapter.nq
        skip_joint_ids = 0

    if joint_count:
        bounds = _joint_ranges(
            adapter.robot, joint_count, -q_span, q_span, skip_joint_ids=skip_joint_ids,
        )
        q[joint_offset:] = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float64)

    return DynamicsSample(name=name, q=q, qd=qd, qdd=qdd)


def build_dynamics_samples(adapter) -> List[DynamicsSample]:
    rng = np.random.default_rng(RNG_SEED)
    samples = [_make_zero_state(adapter), _make_conservative_state(adapter, rng)]
    # High-energy samples. These expose velocity-/acceleration-scaled coupling
    # bugs that the low-energy "conservative" sample (|qd|<=1, |qdd|<=2) misses.
    # float64-vs-float64 comparisons stay exact for correct algorithms, so any
    # mismatch here is structural, not numerical noise.
    samples.append(_make_energetic_state(adapter, rng, "high_velocity",       qd_scale=10.0, qdd_scale=2.0))
    samples.append(_make_energetic_state(adapter, rng, "high_acceleration",   qd_scale=1.0,  qdd_scale=50.0))
    samples.append(_make_energetic_state(adapter, rng, "high_velocity_accel", qd_scale=10.0, qdd_scale=50.0))
    samples.append(_make_energetic_state(adapter, rng, "energetic_random_0",  qd_scale=6.0,  qdd_scale=20.0))
    samples.append(_make_energetic_state(adapter, rng, "energetic_random_1",  qd_scale=6.0,  qdd_scale=20.0))
    return samples

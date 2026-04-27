from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ConventionMismatch:
    category: str
    detail: str


def as_float64(array_like) -> np.ndarray:
    return np.asarray(array_like, dtype=np.float64)


def grid_quaternion_wxyz_to_pin_xyzw(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    if q.shape[0] < 7:
        raise ValueError("Floating-base configuration must have at least 7 position entries.")
    q[3:7] = np.array([q[4], q[5], q[6], q[3]], dtype=np.float64)
    norm = np.linalg.norm(q[3:7])
    if norm == 0.0:
        raise ValueError("Floating-base quaternion norm was zero during normalization.")
    q[3:7] /= norm
    return q


def pin_quaternion_xyzw_to_grid_wxyz(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).copy()
    if q.shape[0] < 7:
        raise ValueError("Floating-base configuration must have at least 7 position entries.")
    q[3:7] = np.array([q[6], q[3], q[4], q[5]], dtype=np.float64)
    norm = np.linalg.norm(q[3:7])
    if norm == 0.0:
        raise ValueError("Floating-base quaternion norm was zero during normalization.")
    q[3:7] /= norm
    return q


def normalize_project_q_for_pin(base_mode: str, q: Sequence[float]) -> np.ndarray:
    q = as_float64(q)
    if base_mode == "floating":
        return grid_quaternion_wxyz_to_pin_xyzw(q)
    return q


def normalize_vector(vector: Sequence[float]) -> np.ndarray:
    return np.atleast_1d(as_float64(vector)).reshape(-1)


def normalize_matrix(matrix) -> np.ndarray:
    return np.atleast_2d(as_float64(matrix))


def movable_joint_names_excluding_floating_root(
    base_mode: str, joint_names: Iterable[str]
) -> list:
    joint_names = list(joint_names)
    if base_mode == "floating" and joint_names:
        return [name for name in joint_names if name != "floating_base_joint"]
    return joint_names

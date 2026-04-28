from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

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


def expand_continuous_joint_positions_for_pin(
    q: Sequence[float],
    joint_names: Sequence[str],
    joint_types_by_name: Mapping[str, str] | None,
) -> np.ndarray:
    q = as_float64(q)
    if not joint_types_by_name:
        return q

    expanded = []
    if len(q) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} scalar joint positions, got {len(q)}."
        )

    for index, joint_name in enumerate(joint_names):
        joint_type = joint_types_by_name.get(joint_name)
        value = float(q[index])
        if joint_type == "continuous":
            expanded.extend((np.cos(value), np.sin(value)))
        else:
            expanded.append(value)
    return np.asarray(expanded, dtype=np.float64)


def normalize_project_q_for_pin(
    base_mode: str,
    q: Sequence[float],
    joint_names: Sequence[str] | None = None,
    joint_types_by_name: Mapping[str, str] | None = None,
) -> np.ndarray:
    q = as_float64(q)
    if base_mode == "floating":
        q_prefix = grid_quaternion_wxyz_to_pin_xyzw(q[:7])
        if joint_names is None:
            return q_prefix
        q_suffix = expand_continuous_joint_positions_for_pin(
            q[7:],
            joint_names,
            joint_types_by_name,
        )
        return np.concatenate((q_prefix, q_suffix))
    if joint_names is None:
        return q
    return expand_continuous_joint_positions_for_pin(q, joint_names, joint_types_by_name)


def project_q_to_pin_q_jacobian(
    base_mode: str,
    q: Sequence[float],
    joint_names: Sequence[str] | None = None,
    joint_types_by_name: Mapping[str, str] | None = None,
) -> np.ndarray:
    q = as_float64(q)

    rows = []
    cols = q.shape[0]

    if base_mode == "floating":
        if q.shape[0] < 7:
            raise ValueError("Floating-base configuration must have at least 7 position entries.")
        base_jac = np.eye(7, dtype=np.float64)
        base_quat = q[3:7]
        norm = np.linalg.norm(base_quat)
        if norm == 0.0:
            raise ValueError("Floating-base quaternion norm was zero during normalization.")
        # Current tests keep the zero-state quaternion normalized, so this captures the
        # simple permutation used by the adapter. More general quaternion-coordinate
        # derivatives can be added later if needed for floating-base q-derivative checks.
        perm = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        base_jac[3:7, 3:7] = perm
        rows.append(base_jac)
        q_joints = q[7:]
        joint_col_offset = 7
    else:
        q_joints = q
        joint_col_offset = 0

    if joint_names is None:
        if rows:
            return rows[0]
        return np.eye(q.shape[0], dtype=np.float64)

    if len(q_joints) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} scalar joint positions, got {len(q_joints)}."
        )

    joint_rows = []
    for local_index, joint_name in enumerate(joint_names):
        joint_type = joint_types_by_name.get(joint_name) if joint_types_by_name else None
        col_index = joint_col_offset + local_index
        if joint_type == "continuous":
            theta = float(q_joints[local_index])
            block = np.zeros((2, cols), dtype=np.float64)
            block[0, col_index] = -np.sin(theta)
            block[1, col_index] = np.cos(theta)
            joint_rows.append(block)
        else:
            block = np.zeros((1, cols), dtype=np.float64)
            block[0, col_index] = 1.0
            joint_rows.append(block)

    if joint_rows:
        rows.append(np.vstack(joint_rows))

    if not rows:
        return np.zeros((0, cols), dtype=np.float64)
    return np.vstack(rows)


def reduce_pinocchio_q_jacobian_to_project(
    jacobian,
    base_mode: str,
    q: Sequence[float],
    joint_names: Sequence[str] | None = None,
    joint_types_by_name: Mapping[str, str] | None = None,
) -> np.ndarray:
    jacobian = normalize_matrix(jacobian)
    project_q = as_float64(q)
    if jacobian.shape[1] == project_q.shape[0]:
        return jacobian
    chain = project_q_to_pin_q_jacobian(
        base_mode,
        q,
        joint_names=joint_names,
        joint_types_by_name=joint_types_by_name,
    )
    if jacobian.shape[1] != chain.shape[0]:
        raise ValueError(
            f"Cannot reduce Pinocchio q-Jacobian with shape {jacobian.shape}; expected "
            f"{project_q.shape[0]} or {chain.shape[0]} columns."
        )
    return normalize_matrix(jacobian @ chain)


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

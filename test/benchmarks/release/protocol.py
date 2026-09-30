"""CPU-only collection policy and strict, JSON-safe measurement helpers."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ROBOTS = {"iiwa14": "fixed", "go2": "floating", "g1": "floating"}
BATCHES = (16, 32, 64, 128, 256, 1024)
CORE = ("inverse_dynamics", "inverse_dynamics_gradient", "idsva_so")
EXTRA = ("minv", "forward_dynamics", "forward_dynamics_gradient", "fdsva_so",
         "end_effector_pose", "end_effector_pose_gradient", "end_effector_pose_hessian",
         "crba", "nonlinear_effects", "generalized_gravity", "ccrba", "coriolis_matrix")
Q_ONLY_OPS = {"minv", "end_effector_pose", "end_effector_pose_gradient", "end_effector_pose_hessian",
              "crba", "generalized_gravity"}
Q_QD_OPS = {"nonlinear_effects", "ccrba", "coriolis_matrix"}
VECTOR_OPS = {"inverse_dynamics", "forward_dynamics", "nonlinear_effects", "generalized_gravity"}
OPERATIONS = CORE + EXTRA
WRAPPER_OPS = CORE[:2]
PRIMARY = {CORE[0]: ("grid_cuda", "grid_jax", "pinocchio", "pinocchio_plain", "mjx", "mujoco_warp"),
           CORE[1]: ("grid_cuda", "grid_jax", "pinocchio", "pinocchio_plain", "mjx"),
           CORE[2]: ("grid_cuda", "grid_jax", "pinocchio", "pinocchio_plain")}
WRAPPERS = ("grid_cuda", "grid_native", "grid_numpy", "grid_jax", "grid_torch")
TABLE_BACKENDS = ("grid_cuda", "grid_jax", "pinocchio", "pinocchio_plain", "mjx", "mujoco_warp", "mujoco_cpu", "bard", "frax")
BACKENDS = WRAPPERS + ("pinocchio", "pinocchio_plain", "mjx", "mujoco_warp", "mujoco_cpu", "bard", "frax")
# Sustained warm-up before sampling: a handful of microsecond calls never
# leaves the idle clock (this box idles far below its sustained boost and
# cannot lock clocks without root), so every backend, CPU or GPU, is driven
# for at least this long before its samples are taken.
WARM_SECONDS = 1.5
ACCURACY_POLICIES = ("strict", "fp32-fd-warnings")
ACCURACY_POLICY_VERSION = 2
TIMED_STATUSES = {"validated", "accuracy_warning"}
FD_WARNING_OPS = {"minv", "forward_dynamics", "forward_dynamics_gradient", "fdsva_so"}
# Gross-error backstop, not a componentwise or application accuracy guarantee.
FD_WARNING_MAX_RELATIVE_L2 = 1e-3
ACCURACY_FOOTNOTE = (
    "FP32 forward-dynamics computations can amplify rounding and cancellation, "
    "particularly in high-velocity cases. Selected entries show percent-level "
    "discrepancies from the fp64 reference; relative errors can be larger near zero. "
    "GPU reduction order can also cause small run-to-run differences. "
    "Accuracy-warning timings are retained with measured errors and entrywise "
    "exceedance counts, not labeled strict validation passes.")


def accuracy_status(check, policy, operation, dtype):
    if policy not in ACCURACY_POLICIES:
        raise ValueError(f"Unknown accuracy policy: {policy}")
    if check.get("passed"):
        return "validated"
    blocks = check.get("blocks", [])
    if (policy == "fp32-fd-warnings" and dtype == "float32"
            and operation in FD_WARNING_OPS and not check.get("reason") and blocks
            and all(math.isfinite(b.get("relative_l2", math.inf))
                    and b["relative_l2"] <= FD_WARNING_MAX_RELATIVE_L2 for b in blocks)):
        return "accuracy_warning"
    return "validation_failed"


def cell_accuracy_status(cell, policy, operation, dtype, *, version=ACCURACY_POLICY_VERSION):
    """Same decision for collection and export; v1 retains its original gates."""
    if version not in (1, ACCURACY_POLICY_VERSION):
        return "validation_failed"
    oracle_keys = ["oracle_agreement", "post_timing_agreement"]
    variation_keys = ["repeatability_agreement"]
    if any(k in cell for k in ("resident", "boundary_agreement", "resident_oracle_agreement",
                               "resident_post_timing_agreement", "resident_repeatability_agreement")):
        variation_keys.append("boundary_agreement")
        if version >= 2:
            oracle_keys += ["resident_oracle_agreement", "resident_post_timing_agreement"]
            variation_keys += ["resident_repeatability_agreement", "post_boundary_agreement"]
    statuses = [accuracy_status(cell.get(k, {}), policy, operation, dtype) for k in oracle_keys]
    statuses += [accuracy_status(cell.get(k, {}), policy if version >= 2 else "strict", operation, dtype)
                 for k in variation_keys]
    # The native/NumPy bridge must remain bitwise identical, never warning-only.
    if not cell.get("native_wrapper_agreement", {"passed": True}).get("passed"):
        return "validation_failed"
    if "validation_failed" in statuses:
        return "validation_failed"
    return "accuracy_warning" if "accuracy_warning" in statuses else "validated"


def capability(backend, operation, robot):
    """Availability of OUR adapters, never inferred library-wide incapability."""
    floating = ROBOTS[robot] == "floating"
    hessian = operation in {"idsva_so", "fdsva_so", "end_effector_pose_hessian"}
    if backend == "grid_native":
        return None if operation in WRAPPER_OPS else "adapter_pending: native C-ABI timing bridge covers RNEA and its gradient"
    if backend == "grid_cuda":
        if operation in {"end_effector_pose_gradient", "end_effector_pose_hessian"}:
            return "adapter_pending: CUDA host-call timing bridge covers every operation but the end-effector derivatives"
        return None
    if backend.startswith("grid_"):
        return None
    if backend in {"pinocchio", "pinocchio_plain"}:
        if operation in {"end_effector_pose_gradient", "end_effector_pose_hessian"}:
            return "adapter_pending: Pinocchio spatial kinematic derivatives are not the requested RPY pose-coordinate derivatives"
        if backend == "pinocchio" and operation in {"ccrba", "coriolis_matrix"}:
            return "adapter_pending: no CppADCodeGen class for this operation; see pinocchio_plain"
        return None
    if hessian:
        return "excluded_method: analytical Hessian study; no finite-difference or nested-autodiff Hessian sweep"
    if operation in {"ccrba", "coriolis_matrix"}:
        return "adapter_pending: no matched centroidal-momentum / Coriolis-matrix output in this library's public API"
    if backend in {"mujoco_warp", "mujoco_cpu"}:
        if backend == "mujoco_warp" and operation == "minv":
            return "adapter_pending: no dense inverse-inertia output path"
        if operation in {"inverse_dynamics", "forward_dynamics", "end_effector_pose", "minv", "crba",
                         "nonlinear_effects", "generalized_gravity"}:
            return None
        return "adapter_pending: no matched full tangent-space derivative adapter"
    if backend == "mjx":
        if operation in {"inverse_dynamics", "forward_dynamics", "inverse_dynamics_gradient", "forward_dynamics_gradient",
                         "end_effector_pose", "crba", "nonlinear_effects", "generalized_gravity"}:
            return None
        return "adapter_pending: selected operation not wired"
    if backend == "bard":
        if operation in {"inverse_dynamics", "forward_dynamics", "crba", "nonlinear_effects", "generalized_gravity"}:
            return None
        return "adapter_pending: selected operation not wired; not a library capability claim"
    if backend == "frax":
        if floating:
            return "model_mismatch: existing Frax adapter uses a six-coordinate floating base; shared quaternion fixture needs a validated conversion"
        if operation in {"inverse_dynamics", "forward_dynamics", "minv", "crba", "nonlinear_effects", "generalized_gravity"}:
            return None
        return "adapter_pending: selected operation not wired"
    raise ValueError(backend)


def jobs(stage, robots, backends=None, operations=None):
    ops = operations or (CORE if stage == "core" else WRAPPER_OPS if stage == "wrappers" else OPERATIONS)
    for robot in robots:
        if robot not in ROBOTS:
            raise ValueError(f"Unknown robot: {robot}")
        for op in ops:
            if op not in OPERATIONS:
                raise ValueError(f"Unknown operation: {op}")
            if stage == "core" and op not in PRIMARY and backends is None:
                raise ValueError(f"{op} is a table operation; use --stage table or explicit --backends")
            selected = backends or (PRIMARY[op] if stage == "core" else
                WRAPPERS if stage == "wrappers" else
                TABLE_BACKENDS)
            for backend in selected:
                if backend not in BACKENDS:
                    raise ValueError(f"Unknown backend: {backend}")
                yield {"robot": robot, "base": ROBOTS[robot], "operation": op,
                       "backend": backend, "unavailable": capability(backend, op, robot)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprints():
    paths = [p for pattern in ("*.py", "*.cpp", "*.cu", "*.h") for p in Path(__file__).parent.glob(pattern)]
    paths += [ROOT / "test/benchmarks/baselines/pinocchio/timePinocchio.cpp"]
    paths += list((ROOT / "test/benchmarks/baselines/util").rglob("*.h*"))
    paths += [ROOT / "external/RBDReference/equivalents/pinocchio_backend.py"]
    paths += [ROOT / "external/URDFParser/Joint.py"]
    paths += list((ROOT / "external/RBDReference/equivalents/pin_so_ext").glob("*.py"))
    paths += list((ROOT / "external/RBDReference/equivalents/pin_so_ext").glob("*.cpp"))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def write_json(path, payload):
    """Atomic within the explicitly selected capture directory; refuse NaN JSON."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def leaves(value):
    if isinstance(value, (list, tuple)):
        return [a for v in value for a in leaves(v)]
    if isinstance(value, dict):
        return [a for k in sorted(value) for a in leaves(value[k])]
    return [np.asarray(value)]


def agreement(actual, expected, *, rtol=2e-4, atol=1e-3):
    aa, bb = leaves(actual), leaves(expected)
    if len(aa) != len(bb):
        return {"passed": False, "reason": "output block count mismatch"}
    blocks = []
    for a, b in zip(aa, bb):
        if a.shape != b.shape:
            return {"passed": False, "reason": f"shape mismatch {a.shape} != {b.shape}"}
        a, b = a.astype(np.float64), b.astype(np.float64)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return {"passed": False, "reason": "non-finite output"}
        diff = np.abs(a - b)
        blocks.append({"shape": list(a.shape), "max_abs": float(diff.max(initial=0)),
            "relative_l2": float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-30)),
            "bad_entries": int(np.sum(diff > atol + rtol * np.abs(b))),
            "entries": int(a.size),
            "max_abs_reference_below_atol": float(diff[np.abs(b) < atol].max(initial=0)),
            "max_relative_reference_ge_atol": float(
                (diff[np.abs(b) >= max(atol, 1e-30)] / np.abs(b[np.abs(b) >= max(atol, 1e-30)])).max(initial=0))})
    return {"passed": all(b["bad_entries"] == 0 for b in blocks),
            "rtol": rtol, "atol": atol, "blocks": blocks}


def timed(call, sync, warmups, iterations, warm_seconds=0.0):
    """`warmups` synchronized calls, continued until `warm_seconds` of wall time
    have elapsed, then `iterations` synchronized samples."""
    if warmups < 1 or iterations < 1:
        raise ValueError("At least one warmup and one timed iteration are required")
    if warm_seconds < 0:
        raise ValueError("warm_seconds must be non-negative")
    start, done = time.perf_counter(), 0
    while done < warmups or time.perf_counter() - start < warm_seconds:
        out = call()
        sync(out)
        done += 1
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        out = call()
        sync(out)
        samples.append((time.perf_counter_ns() - start) / 1000)
    return {"samples_us": samples, "mean_us": float(np.mean(samples)),
            "median_us": float(np.median(samples)), "min_us": min(samples), "max_us": max(samples)}


def overhead(total, resident):
    if total is None or resident is None:
        return None
    if not all(math.isfinite(v) for v in (total, resident)) or total < resident:
        return None
    return total - resident

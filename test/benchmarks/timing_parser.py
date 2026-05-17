"""Parse raw timing output from timeGRiD and timePinocchio into the unified JSON schema."""

from __future__ import annotations

import platform
import re
import subprocess
from statistics import median
from typing import Optional

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Single-call lines: "Single Call ID 2.34us" or "ID codegen 2.34us"
_SINGLE_RE = re.compile(
    r"^(?P<name>.+?)\s+(?P<value>[0-9]+(?:\.[0-9]+)?)\s*us\s*$",
    re.IGNORECASE,
)

# Batch lines: "[N:16]: ID WITH MEMORY: Average[2.34us] Std Dev [0.05us] Min [2.30us] Max [2.40us]"
_BATCH_RE = re.compile(
    r"^\[N:(?P<n>\d+)\]:\s+(?P<name>.+?):\s+"
    r"Average\[(?P<avg>[0-9]+(?:\.[0-9]+)?)us\]\s+"
    r"Std Dev \[(?P<std>[0-9]+(?:\.[0-9]+)?)us\]\s+"
    r"Min \[(?P<min>[0-9]+(?:\.[0-9]+)?)us\]\s+"
    r"Max \[(?P<max>[0-9]+(?:\.[0-9]+)?)us\]",
    re.IGNORECASE,
)

# Pinocchio metadata lines
_PIN_META_START = re.compile(r"=== BEGIN PINOCCHIO METADATA ===")
_PIN_META_END   = re.compile(r"=== END PINOCCHIO METADATA ===")
_PIN_META_LINE  = re.compile(r"^(?P<algo>\S+)\s+codegen:\s+(?P<val>true|false|null)$", re.IGNORECASE)

# Batch sizes used in timeGRiD and timePinocchio
BATCH_SIZES = [16, 32, 64, 128, 256]

# ---------------------------------------------------------------------------
# Label → JSON key mapping
# ---------------------------------------------------------------------------

# GRiD single-call labels (from printf in _single_timing functions)
_GRID_SINGLE_LABELS: dict[str, str] = {
    "single call id":              "id",
    "single call minv":            "minv",
    "single call fd":              "fd",
    "single call aba":             "aba",
    "single call crba":            "crba",
    "single call id_du":           "id_du",
    "single call fd_du":           "fd_du",
    "single call eepos":           "ee_pose",
    "single call deepos":          "ee_pose_gradient",
    "single call idsva_so_body_frame":        "idsva_so_body_frame",
    "single call idsva_so_world_frame":    "idsva_so_world_frame",
    "single call fdsva_so":        "fdsva_so",
    # aliases for variants in generated code
    "single call inverse dynamics":         "id",
    "single call forward dynamics":         "fd",
    "single call minv (direct)":            "minv",
    "single call aba (articulated body)":   "aba",
}

# GRiD batch labels (WITH MEMORY / COMPUTE ONLY suffixes)
_GRID_BATCH_WITH_MEM_LABELS: dict[str, str] = {
    "id with memory":              "id",
    "minv with memory":            "minv",
    "fd with memory":              "fd",
    "aba with memory":             "aba",
    "crba with memory":            "crba",
    "id_du with memory":           "id_du",
    "fd_du with memory":           "fd_du",
    "ee_pose with memory":         "ee_pose",
    "ee_pose_gradient with memory": "ee_pose_gradient",
    "idsva_so_body_frame with memory":        "idsva_so_body_frame",
    "idsva_so_world_frame with memory":    "idsva_so_world_frame",
    "fdsva_so with memory":        "fdsva_so",
    "id_so with memory":           "idsva_so_body_frame",   # legacy label
    "fd_so with memory":           "fdsva_so",   # legacy label
}

_GRID_BATCH_COMPUTE_ONLY_LABELS: dict[str, str] = {
    "id compute only":              "id",
    "minv compute only":            "minv",
    "fd compute only":              "fd",
    "aba compute only":             "aba",
    "crba compute only":            "crba",
    "id_du compute only":           "id_du",
    "fd_du compute only":           "fd_du",
    "ee_pose compute only":         "ee_pose",
    "ee_pose_gradient compute only": "ee_pose_gradient",
    "idsva_so_body_frame compute only":        "idsva_so_body_frame",
    "idsva_so_world_frame compute only":    "idsva_so_world_frame",
    "fdsva_so compute only":        "fdsva_so",
    "id_so compute only":           "idsva_so_body_frame",   # legacy label
    "fd_so compute only":           "fdsva_so",   # legacy label
}

# Pinocchio single-call labels
_PIN_SINGLE_LABELS: dict[str, str] = {
    "id codegen":               "id",
    "minv codegen":             "minv",
    "aba codegen":              "aba",
    "fd codegen":               "fd",
    "crba codegen":             "crba",
    "id_du codegen":            "id_du",
    "fd_du codegen":            "fd_du",
    "id direct":                "id",
    "minv direct":              "minv",
    "aba direct":               "aba",
    "fd direct":                "fd",
    "crba direct":              "crba",
    "id_du direct":             "id_du",
    "fd_du direct":             "fd_du",
    "ee_pose direct":           "ee_pose",
    "ee_pose_gradient direct":  "ee_pose_gradient",
    "idsva_so_body_frame direct":          "idsva_so_body_frame",
}

# Pinocchio batch labels
_PIN_BATCH_LABELS: dict[str, str] = {
    "id codegen":               "id",
    "minv codegen":             "minv",
    "aba codegen":              "aba",
    "fd codegen":               "fd",
    "crba codegen":             "crba",
    "id_du codegen":            "id_du",
    "fd_du codegen":            "fd_du",
    "id direct":                "id",
    "minv direct":              "minv",
    "aba direct":               "aba",
    "fd direct":                "fd",
    "crba direct":              "crba",
    "id_du direct":             "id_du",
    "fd_du direct":             "fd_du",
    "ee_pose direct":           "ee_pose",
    "ee_pose_gradient direct":  "ee_pose_gradient",
    "idsva_so_body_frame direct":          "idsva_so_body_frame",
}


def _stats(avg: float, std: float, mn: float, mx: float) -> dict[str, float]:
    return {"min": mn, "mean": avg, "max": mx, "std": std, "median": avg}


def _single_stats(value: float) -> dict[str, float]:
    return {"min": value, "mean": value, "max": value, "std": 0.0, "median": value}


def _batch_key(n: int, kind: str) -> str:
    """Return the JSON key for a batch timing, e.g. 'batch_16_with_mem_us'."""
    return f"batch_{n}_{kind}_us"


# ---------------------------------------------------------------------------
# GRiD parser
# ---------------------------------------------------------------------------

def parse_grid_output(stdout: str) -> dict[str, Optional[dict]]:
    """Parse timeGRiD stdout into algo→timing dict.

    Returns a dict keyed by algo name (e.g. 'id', 'aba') with values:
        {
          "single_us": {...},
          "batch_16_with_mem_us": {...}, "batch_16_compute_only_us": {...},
          ...
        }
    Missing entries are left absent (caller fills with None for the JSON).
    """
    results: dict[str, dict] = {}

    for raw_line in stdout.splitlines():
        line = raw_line.strip()

        # Batch line
        m = _BATCH_RE.match(line)
        if m:
            n = int(m.group("n"))
            label = m.group("name").strip().lower()
            avg = float(m.group("avg"))
            std = float(m.group("std"))
            mn  = float(m.group("min"))
            mx  = float(m.group("max"))

            algo = _GRID_BATCH_WITH_MEM_LABELS.get(label)
            if algo is not None:
                results.setdefault(algo, {})[_batch_key(n, "with_mem")] = _stats(avg, std, mn, mx)
                continue
            algo = _GRID_BATCH_COMPUTE_ONLY_LABELS.get(label)
            if algo is not None:
                results.setdefault(algo, {})[_batch_key(n, "compute_only")] = _stats(avg, std, mn, mx)
            continue

        # Single-call line
        m = _SINGLE_RE.match(line)
        if m:
            label = m.group("name").strip().lower()
            value = float(m.group("value"))
            algo = _GRID_SINGLE_LABELS.get(label)
            if algo is not None:
                results.setdefault(algo, {})["single_us"] = _single_stats(value)

    return results


# ---------------------------------------------------------------------------
# Pinocchio parser
# ---------------------------------------------------------------------------

def parse_pinocchio_output(stdout: str) -> dict[str, Optional[dict]]:
    """Parse timePinocchio stdout into algo→timing dict.

    Returns same schema as parse_grid_output but with a 'codegen' field per algo
    and only batch_N_with_mem_us (no compute_only for CPU).
    """
    results: dict[str, dict] = {}
    codegen_flags: dict[str, Optional[bool]] = {}

    in_meta = False
    for raw_line in stdout.splitlines():
        line = raw_line.strip()

        # Metadata block
        if _PIN_META_START.search(line):
            in_meta = True
            continue
        if _PIN_META_END.search(line):
            in_meta = False
            continue
        if in_meta:
            m = _PIN_META_LINE.match(line)
            if m:
                algo_raw = m.group("algo").lower().replace(" ", "_")
                val_raw  = m.group("val").lower()
                val: Optional[bool] = None if val_raw == "null" else (val_raw == "true")
                codegen_flags[algo_raw] = val
            continue

        # Batch line
        m = _BATCH_RE.match(line)
        if m:
            n = int(m.group("n"))
            label = m.group("name").strip().lower()
            avg = float(m.group("avg"))
            std = float(m.group("std"))
            mn  = float(m.group("min"))
            mx  = float(m.group("max"))
            algo = _PIN_BATCH_LABELS.get(label)
            if algo is not None:
                if not isinstance(results.get(algo), dict):
                    results[algo] = {}
                results[algo][_batch_key(n, "with_mem")] = _stats(avg, std, mn, mx)
            continue

        # Single-call line — but skip "FDSVA_SO direct null"
        if "null" in line.lower():
            parts = line.lower().split()
            if len(parts) >= 2:
                label = " ".join(parts[:-1])
                algo = _PIN_SINGLE_LABELS.get(label)
                if algo is None:
                    # try the full label without "null"
                    for k, v in _PIN_SINGLE_LABELS.items():
                        if label.startswith(k):
                            algo = v
                            break
                if algo is not None:
                    results[algo] = None  # explicit null
            continue

        m = _SINGLE_RE.match(line)
        if m:
            label = m.group("name").strip().lower()
            value = float(m.group("value"))
            algo = _PIN_SINGLE_LABELS.get(label)
            if algo is not None:
                if not isinstance(results.get(algo), dict):
                    results[algo] = {}
                results[algo]["single_us"] = _single_stats(value)

    # Attach codegen flags
    for algo, entry in results.items():
        if entry is not None and algo in codegen_flags:
            entry["codegen"] = codegen_flags[algo]

    return results


# ---------------------------------------------------------------------------
# Hardware metadata
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def gpu_metadata() -> dict[str, str]:
    result = _run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"])
    if result.returncode != 0:
        return {"gpu": "unknown", "compute_capability": "unknown", "driver": "unknown"}
    first = result.stdout.strip().splitlines()[0].split(",")
    return {
        "gpu":                 first[0].strip() if len(first) > 0 else "unknown",
        "compute_capability":  first[1].strip() if len(first) > 1 else "unknown",
        "driver":              first[2].strip() if len(first) > 2 else "unknown",
    }


def cuda_metadata() -> dict[str, str]:
    result = _run(["nvcc", "--version"])
    version = "unknown"
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "release" in line.lower():
                m = re.search(r"release\s+([0-9]+\.[0-9]+)", line)
                version = m.group(1) if m else line.strip()
                break
    return {"cuda_version": version}


def cpu_metadata() -> dict[str, object]:
    import os
    thread_count = os.cpu_count() or 0
    cpu_name = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"cpu": cpu_name, "cpu_threads": thread_count}


def pinocchio_metadata() -> dict[str, str]:
    try:
        import pinocchio
        return {"pinocchio_version": pinocchio.__version__}
    except Exception:
        return {"pinocchio_version": "unknown"}


def build_metadata(include_gpu: bool = True, include_pinocchio: bool = False) -> dict:
    """Collect machine metadata into a single dict."""
    import datetime
    meta: dict = {"host": platform.node(), "date": datetime.date.today().isoformat()}
    if include_gpu:
        meta.update(gpu_metadata())
        meta.update(cuda_metadata())
    meta.update(cpu_metadata())
    if include_pinocchio:
        meta.update(pinocchio_metadata())
    return meta


# ---------------------------------------------------------------------------
# Result merging
# ---------------------------------------------------------------------------

ALL_ALGOS = ["id", "minv", "fd", "aba", "crba", "id_du", "fd_du",
             "ee_pose", "ee_pose_gradient", "idsva_so_body_frame", "idsva_so_world_frame", "fdsva_so"]


def fill_nulls(result: dict[str, Optional[dict]], algos: list[str] = ALL_ALGOS) -> dict[str, Optional[dict]]:
    """Ensure all expected algo keys exist, filling missing ones with None."""
    for algo in algos:
        if algo not in result:
            result[algo] = None
    return result

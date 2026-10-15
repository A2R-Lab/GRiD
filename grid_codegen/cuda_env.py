"""Toolchain probes shared by the test harness, the benchmark baselines and the
examples (2026-10-14 audit: four near-identical nvidia-smi detectors and three
nvcc readers had drifted in defaults and normalisation).

NOT used by the ``grid_rbd`` bindings' cache keys (``grid_rbd/_cache.py`` keeps
its own int-valued detector and nvcc tag): those are cache-key inputs for every
user's compiled robots, so their semantics stay frozen on purpose.
"""
from __future__ import annotations

import shutil
import subprocess

DEFAULT_CUDA_ARCH = "86"


def detect_cuda_arch(override: str | None = None, default: str = DEFAULT_CUDA_ARCH) -> str:
    """Compute capability digits of the first GPU ("120" for sm_120).

    ``override`` wins when given (accepts "120", "12.0", "sm_120", "compute_120");
    the test/bench harness passes the GRID_CUDA_ARCH environment value —
    this package itself reads no environment (every ``GRID_*`` read inside
    grid_codegen is an emission knob folded into cache keys, see env_knobs.py).
    Otherwise ``nvidia-smi --query-gpu=compute_cap``; otherwise ``default``.
    """
    if override:
        return override.replace("sm_", "").replace("compute_", "").replace(".", "")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                compute_cap = line.strip()
                if compute_cap:
                    return compute_cap.replace(".", "")
    return default


def nvcc_version_text() -> str:
    """``nvcc --version`` stdout (stripped); "missing" without nvcc on PATH,
    "error:<rc>:<stderr>" on a failing invocation."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return "missing"
    result = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        return f"error:{result.returncode}:{result.stderr.strip()}"
    return result.stdout.strip()

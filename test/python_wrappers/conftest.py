"""python_wrappers suite lifecycle conftest (Wave 2b, 2026-08-11).

Two jobs, both stem from the in-process accumulation bug class (a ~15-module
in-process run dlopens ~40 robot .so's + torch + jax into one interpreter and
NOTHING ever called RobotHandle.close() — the historical late-suite SIGABRT and
the ps5 ordering-dependent failure live here):

1. **Handle close tracker** (default ON): wraps grid_rbd.register_robot for the
   duration of each test module and closes every handle the module produced at
   module teardown — covering fixture registrations AND bare in-test calls
   without editing ~24 modules. close() is idempotent and only drops the Runner
   ref (numpy handles actually dlclose; torch/jax keep process-global CDLL/op
   refs by design). Disable with GRID_TEST_NO_CLOSE=1 to A/B the old behavior.

2. **Per-test VRAM watermark probe** (opt-in, GRID_PROBE_VRAM=1): after each
   test, append `module,test,pid_mib,total_mib` for THIS pid to the CSV at
   GRID_PROBE_VRAM_OUT (default .split_suite/vram_probe.csv). Per-PID via
   nvidia-smi --query-compute-apps so concurrent GPU work (e.g. a sanitizer
   arm) cannot contaminate the trajectory.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True, scope="module")
def _close_module_handles():
    if os.environ.get("GRID_TEST_NO_CLOSE") == "1":
        yield
        return
    try:
        import grid_rbd
    except Exception:
        yield
        return
    produced = []
    real = grid_rbd.register_robot

    def _tracking_register(*args, **kwargs):
        handle = real(*args, **kwargs)
        produced.append(handle)
        return handle

    grid_rbd.register_robot = _tracking_register
    try:
        yield
    finally:
        grid_rbd.register_robot = real
        for handle in produced:
            try:
                handle.close()
            except Exception:
                pass


def _pid_vram_mib() -> tuple[int, int]:
    """(this pid's MiB, total MiB) from nvidia-smi; (-1, -1) if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        mine, total = 0, 0
        for line in out.strip().splitlines():
            pid_s, mem_s = [p.strip() for p in line.split(",")[:2]]
            total += int(mem_s)
            if int(pid_s) == os.getpid():
                mine += int(mem_s)
        return mine, total
    except Exception:
        return -1, -1


@pytest.fixture(autouse=True)
def _vram_watermark(request):
    yield
    if os.environ.get("GRID_PROBE_VRAM") != "1":
        return
    mine, total = _pid_vram_mib()
    out = Path(os.environ.get(
        "GRID_PROBE_VRAM_OUT",
        str(Path(__file__).resolve().parents[1] / ".split_suite" / "vram_probe.csv")))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(f"{request.node.module.__name__},{request.node.name},{mine},{total}\n")

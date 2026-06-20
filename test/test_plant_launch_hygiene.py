"""Structural guard: every register-heavy grid_plant kernel launch in the bindings
wrapper must derive its block dimension from ``grid_clamp_threads_for(...)`` (a
``thr`` dim3), NOT a bare ``grid_rbd_launch_threads<...>()`` in the launch-config
slot.

WHY THIS EXISTS (the silent register-OOR launch class, 2026-06-19). The plant
cost/step kernels (com_cost, ee_pos_cost, momentum_cost, plant_step,
plant_step_gradient) are register-heavy. Launched at the unclamped global default
thread count, the launch is silently rejected with
``cudaErrorLaunchOutOfResources`` (or ``cudaErrorInvalidValue`` for a >48 KB smem
arena), the kernel never runs, and the device output buffer keeps its stale/zero
contents -- so the binding returns plausible-looking garbage with no error. This
masqueraded as a dozen mjx-vs-oracle failures (com/ee returned 0, plant_step
returned the stale mjx buffer) until ``grid_clamp_threads_for`` + a post-launch
``cudaGetLastError`` were added at every site.

The invariant: for every launch of a register-heavy plant kernel, the thread
argument of the ``<<<grid_dim, THREADS, smem, stream>>>`` config is a clamped
``thr`` (the result of ``grid_clamp_threads_for``), never a bare
``grid_rbd_launch_threads<...>()``. Pure-Python source introspection (no codegen,
no nvcc, no GPU) so it runs in ordinary CI and fails the instant a new launch site
(or a regenerated wrapper) drops the clamp.
"""

from __future__ import annotations

import re
from pathlib import Path

_WRAPPER = Path(__file__).resolve().parents[1] / "bindings" / "grid_rbd" / "wrapper_template.cu"

# Register-heavy plant kernels whose launches must be clamped.
_GUARDED = ("com_cost_kernel", "ee_pos_cost_kernel", "momentum_cost_kernel",
            "plant_step_kernel", "plant_step_gradient_kernel")

# A kernel launch: grid_plant::<NAME><...><<< grid_dim , <THREADS> , ...
_LAUNCH_RE = re.compile(
    r"grid_plant::(" + "|".join(_GUARDED) + r")\s*<[^<>]*(?:<[^<>]*>[^<>]*)*>\s*"
    r"<<<\s*[^,]+,\s*(?P<threads>[^,]+),",
    re.DOTALL,
)


def test_plant_launches_are_clamped():
    assert _WRAPPER.exists(), f"wrapper template not found: {_WRAPPER}"
    src = _WRAPPER.read_text()
    offenders = []
    for m in _LAUNCH_RE.finditer(src):
        threads = m.group("threads").strip()
        # OK: a clamped dim3 (named `thr`, the grid_clamp_threads_for result).
        if "grid_clamp_threads_for" in threads or re.fullmatch(r"thr\b", threads):
            continue
        # NOT OK: a bare launch-thread helper in the launch-config thread slot.
        if "grid_rbd_launch_threads" in threads:
            line = src.count("\n", 0, m.start()) + 1
            offenders.append((line, m.group(1), threads))
    assert not offenders, (
        "register-heavy plant kernel launch(es) missing grid_clamp_threads_for "
        "(silent register-OOR risk):\n"
        + "\n".join(f"  line {ln}: grid_plant::{name} launched with `{thr}`" for ln, name, thr in offenders)
    )

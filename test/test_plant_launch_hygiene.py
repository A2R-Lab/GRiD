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


# ---------------------------------------------------------------------------
# GRID-slot hygiene (the workspace-slot aliasing class, 2026-08-09). Kernels
# index the per-block workspace arena by blockIdx and grid-stride over
# timesteps; init_gridData may fit FEWER slots than kMaxBatch under memory
# pressure. A direct launch whose grid dim is a raw batch (not clamped through
# grid_rbd_grid_for) aliases live workspace slots across blocks -- silent
# wrong results on exactly the big robots that shrink. And because the
# jax/torch paths are deliberately sync-free, a rejected launch without a
# post-launch cudaGetLastError check surfaces as stale buffer contents.
# ---------------------------------------------------------------------------

# Launchers whose kernel launches are checked at their (macro) call sites, not
# inside their void bodies -- see the GRID_RBD_IT_DISPATCH* assertion below.
_DISPATCHED_VOID_LAUNCHERS = (
    "launch_integrator_host", "launch_integrator_host_mujoco",
    "launch_integrator_grad_host", "launch_integrator_grad_host_mujoco",
    "launch_plant_step", "launch_plant_step_mujoco",
    "launch_plant_step_gradient", "launch_plant_step_gradient_mujoco",
    "launch_plant_step_hessian", "launch_plant_step_hessian_mujoco",
    "launch_integrator_kernel_jax", "launch_integrator_grad_kernel_jax",
    "launch_plant_step_jax", "launch_plant_step_gradient_jax",
    "torch_launch_integrator", "torch_launch_integrator_grad",
    "torch_launch_plant_step", "torch_launch_plant_step_gradient",
)

_CHECK_MARKERS = ("cudaGetLastError", "gpuErrchk")


def _void_launcher_ranges(lines):
    ranges = []
    for name in _DISPATCHED_VOID_LAUNCHERS:
        for i, l in enumerate(lines):
            if re.search(rf"static void {name}(_mujoco)?\s*\(", l) and name in l:
                j = i
                while j < len(lines) and lines[j].rstrip() != "}":
                    j += 1
                ranges.append((i, j))
                break
    return ranges


def test_every_launch_grid_is_slot_clamped():
    """No raw-batch grid dims: every <<<>>> grid expression must be
    grid_rbd_grid_for(...) itself or a grid_dim built from it."""
    src = _WRAPPER.read_text()
    assert "<<<dim3(" not in src, "raw dim3(...) in a launch-config grid slot"
    assert "<<<batch" not in src, "raw batch in a launch-config grid slot"
    for m in re.finditer(r"dim3 grid_dim[^;\n]*;", src):
        decl = m.group(0)
        assert "grid_rbd_grid_for" in decl, f"unclamped grid_dim decl: {decl}"


def test_every_launch_is_error_checked():
    """Every kernel launch statement is followed by a cudaGetLastError-class
    check within 4 lines, except launches inside the dispatched void launchers
    (whose GRID_RBD_IT_DISPATCH* call sites carry the check instead)."""
    lines = _WRAPPER.read_text().split("\n")
    ranges = _void_launcher_ranges(lines)
    assert len(ranges) == len(_DISPATCHED_VOID_LAUNCHERS), (
        "void-launcher inventory drifted -- update _DISPATCHED_VOID_LAUNCHERS")
    offenders = []
    i = 0
    while i < len(lines):
        if "<<<" in lines[i] and not lines[i].lstrip().startswith("//"):
            j = i
            seen_close = ">>>" in lines[j]
            while not (seen_close and lines[j].rstrip().rstrip("\\").rstrip().endswith(";")):
                j += 1
                if ">>>" in lines[j]:
                    seen_close = True
            in_void = any(a <= i <= b for a, b in ranges)
            # 12 lines: covers the consolidated-check pattern where one check
            # follows a 3-branch if/else-if or switch/case launch chain
            # (branches ~3-4 lines each).
            lookahead = "\n".join(lines[j + 1:j + 13])
            if not in_void and not any(mk in lookahead for mk in _CHECK_MARKERS):
                offenders.append(i + 1)
            i = j + 1
        else:
            i += 1
    assert not offenders, f"launches missing a post-launch error check at lines: {offenders}"


def test_dispatch_macro_calls_are_error_checked():
    """Every GRID_RBD_IT_DISPATCH* statement (which expands to a void launcher
    containing an unchecked kernel launch) must be followed by a check within
    2 lines -- that check is what covers the launcher's launches."""
    lines = _WRAPPER.read_text().split("\n")
    offenders = []
    for i, l in enumerate(lines):
        if re.match(r"\s*GRID_RBD_IT_DISPATCH\w*\(", l) and l.rstrip().endswith(";"):
            lookahead = "\n".join(lines[i + 1:i + 5])
            if not any(mk in lookahead for mk in _CHECK_MARKERS + ("TORCH_CHECK",)):
                offenders.append(i + 1)
    assert not offenders, f"unchecked dispatch-macro call(s) at lines: {offenders}"

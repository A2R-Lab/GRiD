"""CPU referee for abi_specs.cabi_direct (2026-10-01).

A cabi_direct row makes the C-ABI body aim the generated host wrapper's D2H copy at the
caller's buffer (GridMirrorRetarget) instead of copying a second time (a memcpy of the
pinned mirror, or — the device-direct rows — a second download of the device buffer). That
is only memory-safe when the wrapper copies EXACTLY batch * out_size_expr elements into the
mirror.
Three wrappers copy NUM_JOINTS-strided rows (more than the NUM_VEL public row on a floating
base) and five copy by another pattern; this test regenerates fixed and floating headers
and evaluates every cabi_direct wrapper's D2H size against the spec, so flagging a row that
does not qualify — or a wrapper edit that breaks a flagged one — fails here, not in a user's
heap.

It also pins the delivery invariant the first cut of this feature broke (2026-10-02): the
MuJoCo-twin bodies dropped their copy-out but never gained the retarget, so a twin returned
the caller's buffer UNWRITTEN — and no receipt-path GPU test compared a retargeted numpy
twin against anything. Every generated C-ABI body, primary and twin, must deliver its
output: by a retarget guard or by an explicit copy into the out pointer.
"""
from __future__ import annotations

import contextlib
import os
import re
import tempfile
from pathlib import Path

import pytest

from grid_codegen.abi_specs import ABI_SPECS
from grid_codegen.GRiDCodeGenerator import GRiDCodeGenerator
from grid_codegen.wrapper_body_gen import (GENERATED_KEYS, MJX_KEYS, _mirror_name, _mirror_swap,
                                           gen_body, gen_mjx_body)
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec

_CASES = (("iiwa14", "fixed"), ("go2", "floating"))


def _header(robot, base, monkeypatch):
    monkeypatch.setenv("GRID_ENABLE_MUJOCO_KERNELS", "1")   # the mjx-twin bodies retarget too
    spec = next(c["spec"] for c in iter_robot_cases(MANIFEST_PATH, base_mode=base) if c["spec"].robot_id == robot)
    pm = build_project_adapter(spec, resolve_robot_spec(spec), base_mode=base)
    out = Path(tempfile.mkdtemp()) / "grid.cuh"
    gen = GRiDCodeGenerator(pm.robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        gen.gen_all_code(output_path=str(out), codegen_profile="all", include_homogenous_transforms=True)
    return out.read_text()


def _consts(src):
    c = {m.group(1): int(m.group(2)) for m in re.finditer(r"const int (\w+) = (\d+);", src)}
    for m in re.finditer(r"#define (GRID_RBD_NUM_EES|NUM_EES)\s+(\d+)", src):
        c[m.group(1)] = int(m.group(2))
    c.setdefault("GRID_RBD_NUM_EES", c.get("NUM_EES", 0))
    return c


def _ev(expr, c, batch):
    env = dict(c, num_timesteps=batch)
    return eval(expr.replace("grid::", "").replace("sizeof(T)", "1"), {"__builtins__": {}}, env)


@pytest.mark.parametrize("robot,base", _CASES, ids=[f"{r}-{b}" for r, b in _CASES])
def test_every_cabi_direct_wrapper_copies_exactly_batch_times_out_size(robot, base, monkeypatch):
    src = _header(robot, base, monkeypatch)
    c = _consts(src)
    batch = 5
    direct = {k: s for k, s in ABI_SPECS.items() if s.cabi_direct}
    assert direct and all(_mirror_swap(s) for s in direct.values())
    # The comparison below evaluates both sides with THIS header's constants, so it is only
    # a proof if the spec's size names nothing the wrapper can define differently from the
    # header. GRID_RBD_NUM_EES is such a name (1 on a named-target build, grid::NUM_EES
    # leaves in the host function's download).
    wrapper_sized = [k for k, s in direct.items() if "GRID_RBD_" in s.out_size_expr]
    assert not wrapper_sized, f"direct rows sized by a wrapper-side macro: {wrapper_sized}"
    problems = []
    for key, s in direct.items():
        want = _ev(s.out_size_expr, c, batch) * batch
        mirror = _mirror_name(s)
        pat = re.compile(r"cudaMemcpy(?:Async)?\(\s*hd_data->" + re.escape(mirror)
                         + r"\s*,\s*[^,]+,\s*(.+?),\s*cudaMemcpyDeviceToHost")
        lines = [m for line in src.splitlines() for m in [pat.search(line)] if m]
        if not lines:
            problems.append(f"{key}: no D2H into {mirror} (copy pattern not recognised)")
            continue
        for m in lines:
            size = m.group(1)
            got = _ev(size, c, batch)
            if "num_timesteps" not in size:     # the single-call timing variant copies one item
                got *= batch
            if got != want:
                problems.append(f"{key}: wrapper copies {got} elements, batch*out_size is {want} [{size.strip()}]")
    assert not problems, "\n".join(problems)


def test_known_over_copying_wrappers_are_not_direct():
    for key in ("generalized_gravity", "nonlinear_effects", "integrator_gradient",
                "frame_jacobian", "frame_jacobian_dot", "osc_inertia",
                "end_effector_pose_runtime", "end_effector_pose_gradient_runtime",
                # named-target builds: wrapper-side NUM_EES == 1, host download == all leaves
                "end_effector_pose", "end_effector_pose_gradient", "end_effector_pose_hessian"):
        assert not ABI_SPECS[key].cabi_direct, key


def _out_name(spec):
    names = [n for n, _t in spec.inputs]
    return next(n for n in names if n.endswith("out") or n == "out")


def test_every_generated_cabi_body_delivers_its_output():
    """Primary AND MuJoCo-twin bodies: a retarget guard aimed at the out pointer, or an
    explicit copy into it. A body with neither returns the caller's buffer unwritten."""
    bodies = [(k, gen_body(ABI_SPECS[k])) for k in GENERATED_KEYS]
    bodies += [(k + "_mujoco", gen_mjx_body(ABI_SPECS[k])) for k in MJX_KEYS]
    problems = []
    for name, body in bodies:
        spec = ABI_SPECS[name.removesuffix("_mujoco")]
        out = re.escape(_out_name(spec))
        guard = re.search(r"GridMirrorRetarget \w+\(&g_data->" + re.escape(_mirror_name(spec)) + r", " + out + r"\);", body)
        copy = re.search(r"(?:std::memcpy|unpack_rows|cudaMemcpy(?:2D)?)\(" + out + r",", body)
        if _mirror_swap(spec) and not guard:
            problems.append(f"{name}: cabi_direct body has no retarget guard")
        if not _mirror_swap(spec) and guard:
            problems.append(f"{name}: retarget guard on a row that is not cabi_direct")
        if bool(guard) == bool(copy):
            problems.append(f"{name}: output must be delivered exactly once (guard={bool(guard)}, copy={bool(copy)})")
    assert not problems, "\n".join(problems)

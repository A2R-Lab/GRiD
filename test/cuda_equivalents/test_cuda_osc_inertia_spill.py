"""osc_inertia spill-F rung CUDA regression (h2_plus de-gate).

osc_inertia (Lambda = (J Minv J^T)^-1) composes Minv on device with the heavy minv
F-region kept in a dedicated shared s_F buffer (6*nv*nv). On h2_plus (floating nv=81)
that F-region is ~153KB and the full arena ~228KB (UNLAUNCHABLE). A 2-rung ladder adds
a spill-F tier that routes s_F to the L2-pinned minv-F workspace offset, shrinking the
arena by 6*nv*nv (h2_plus ~228KB -> ~74KB, PERF-launchable). All other robots fit at full.

osc_inertia is INHERENTLY non-deterministic (the block-cooperative 6x6 GLASS invert_matrix
+ the atomic J*Minv*J^T contraction reorder float32 adds; the 6x6 inversion amplifies the
~1e-7 epsilon to ~1e-5 run-to-run). So this validates the spill-F rung against the
pinocchio/RBDReference ORACLE (deterministic) at the same float32 tolerance the standard
osc_inertia test uses (5e-3) -- NOT bit-identity vs the full rung. The host wrapper path is
used (it allocates + L2-pins d_workspace through gridData and threads it kernel->device); a
forced-low GRID_CUDA_TARGET_SHARED_MEM_BYTES + an osc-only codegen subset selects the spill-F
rung without pulling the SO kernels into the header.

Override robots with GRID_CUDA_OSC_SPILL_ROBOTS="g1:floating,iiwa14:fixed".
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from grid_codegen import GRiDCodeGenerator
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter
from test.cuda_equivalents.test_cuda_executable_equivalence import _detect_cuda_arch, _build_cuda_samples

_RUNNER = Path(__file__).with_name("cuda_frame_jacobian_host_runner.cu")
_REF_FRAME = "LOCAL_WORLD_ALIGNED"  # the osc_inertia kernel bakes leaf[0] + LWA


def _robot_modes():
    raw = os.environ.get("GRID_CUDA_OSC_SPILL_ROBOTS", "g1:floating,iiwa14:fixed")
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        rid, _, mode = tok.partition(":")
        out.append((rid.strip(), (mode.strip() or "fixed")))
    return out


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not in manifest")


def _py_arena_bytes(t, topo, tbytes=4):
    off = tbytes * int(t)
    if topo > 0:
        off = ((off + 3) // 4) * 4 + 4 * int(topo)
    return ((off + 15) // 16) * 16


def _gen(robot, build_dir, target):
    prev = os.environ.get("GRID_CUDA_TARGET_SHARED_MEM_BYTES")
    os.environ["GRID_CUDA_TARGET_SHARED_MEM_BYTES"] = str(target)
    try:
        cg = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid")
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            cg.gen_all_code(algorithm_list=["osc_inertia"], output_path=str(build_dir / "grid.cuh"))
    finally:
        if prev is None:
            os.environ.pop("GRID_CUDA_TARGET_SHARED_MEM_BYTES", None)
        else:
            os.environ["GRID_CUDA_TARGET_SHARED_MEM_BYTES"] = prev
    return cg.osc_inertia_spill_tier_3way[0], cg


def _compile(build_dir, arch):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if shutil.which("nvcc") is None and not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    shutil.copyfile(_RUNNER, build_dir / "runner.cu")
    glass = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    exe = build_dir / "runner.exe"
    cmd = [nvcc, "-std=c++17", "-O0", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           f"-I{glass}", f"-I{build_dir}", "-o", str(exe), str(build_dir / "runner.cu")]
    res = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(f"osc_inertia spill runner compile failed.\n{' '.join(cmd)}\n{res.stderr[-3000:]}")
    return exe


def _run_lambda(exe, q, qd):
    stdin = " ".join(f"{x:.9g}" for x in q) + "\n" + " ".join(f"{x:.9g}" for x in qd) + "\n"
    res = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True)
    m = re.search(r"BEGIN LAM \d+ \d+\n(.*?)\nEND LAM", res.stdout, re.S)
    if not m:
        pytest.fail(f"osc runner produced no LAM.\nstderr:\n{res.stderr[-2000:]}")
    assert res.returncode == 0, f"runner exited {res.returncode}: {res.stderr[-1500:]}"
    return np.array([float(x) for x in m.group(1).split()], dtype=np.float64).reshape(6, 6, order="F")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_osc_inertia_spill_matches_reference(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    pm = build_project_adapter(spec, resolved, base_mode=base_mode)
    robot = pm.robot
    leaf_name = robot.get_joint_by_id(robot.get_leaf_nodes()[0]).get_name()
    arch = _detect_cuda_arch()

    # Force the spill-F rung: codegen one byte under the full arena (full > spill-F).
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    full_pick, cg_full = _gen(robot, probe_dir, 98304)
    assert full_pick == 0, f"expected osc_inertia full rung (pick 0) at the default target, got {full_pick}"
    topo = cg_full.gen_topology_helpers_size()
    full_t = cg_full.osc_inertia_t_count_per_tier[0]

    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    spill_pick, _ = _gen(robot, spill_dir, _py_arena_bytes(full_t, topo) - 1)
    assert spill_pick == 1, f"forced target did not land the osc_inertia spill-F rung (got {spill_pick})"
    htxt = (spill_dir / "grid.cuh").read_text()
    assert "GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()" in htxt and "!OSC_F_SMEM" in htxt, \
        "spill-F repoint not emitted in the forced header"
    assert "GRID_OSC_INERTIA_USES_WORKSPACE = 1" in htxt, "kinematics workspace gate not set at the spill tier"
    spill_exe = _compile(spill_dir, arch)

    samples = _build_cuda_samples(pm, random_count=4, include_corner_samples=False)
    checked = 0
    for sample in samples:
        q = np.asarray(sample.q, np.float64)
        qd = np.asarray(sample.qd, np.float64)
        # skip rank-deficient task matrices (the 6x6 inverse is ill-defined for both
        # the oracle and the device at those configs).
        J = np.asarray(pm.frame_jacobian(q, leaf_name, _REF_FRAME), dtype=np.float64)
        task = J @ np.asarray(pm.minv(q), dtype=np.float64) @ J.T
        if np.linalg.cond(task) >= 1e8:
            continue
        L_ref = np.asarray(pm.osc_inertia(q, leaf_name, _REF_FRAME), dtype=np.float64)
        L_cuda = _run_lambda(spill_exe, q, qd)
        scale = float(np.max(np.abs(L_ref))) or 1.0
        np.testing.assert_allclose(
            L_cuda, L_ref, rtol=5e-3, atol=max(5e-3, 5e-3 * scale),
            err_msg=f"{robot_id}-{base_mode} @ {sample.name}: osc_inertia spill-F Lambda vs oracle")
        checked += 1
    assert checked > 0, "no well-conditioned samples were checked"

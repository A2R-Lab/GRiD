"""forward_dynamics_gradient (fd_du) OUTPUT-spill rung CUDA regression.

The fd_du kernel grew a 4th tier rung (output-spill, the new MINIMAL): on top of
the emergency rung's whole-inner-pool -> d_workspace (GRAD section), it additionally
repoints the two OUTPUT bands s_dc_du (2*nv*nv id-gradient band) + s_Minv (nv*nv mass
matrix) to the L2-pinned SO band, shrinking the smem arena by 3*nv*nv. This is what
makes h2_plus (floating nv=81) launch (its emergency arena was ~105KB > sm_120's
~99KB; the output-spill arena is ~28KB).

The spill RELOCATES buffers; it must not change the computed df_du. This test forces
the output-spill rung at codegen (a low GRID_CUDA_TARGET_SHARED_MEM_BYTES on a small
robot, with a no-second-order codegen subset so the forced-low target does not trigger
the pathological idsva_so/fdsva_so deep-spill compile wall) and asserts the host-wrapper
df_du is BIT-IDENTICAL to the unspilled full-smem rung (the validated reference path,
checked against pinocchio elsewhere) AND thread-count invariant. GRiD-vs-GRiD bit-
equality is a stronger, convention-free check of the spill than an oracle comparison.

Override robots with GRID_CUDA_FD_DU_SPILL_ROBOTS="iiwa14:fixed,go2:floating".
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

from GRiDCodeGenerator import GRiDCodeGenerator
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter
from test.cuda_equivalents.test_cuda_executable_equivalence import _detect_cuda_arch

# fd_du needs these (+ its id/minv/fd composition); crba/aba/end_effector_pose are
# launched unconditionally by cuda_equivalence_runner.cu's value block. NONE are
# second-order, so a forced-low target spills only the first-order ladders (cheap
# compile) and never the 4*NV^3 SO contract.
_SUBSET = ["inverse_dynamics", "minv", "forward_dynamics", "crba", "aba",
           "end_effector_pose", "inverse_dynamics_gradient", "forward_dynamics_gradient"]
_RUNNER = Path(__file__).with_name("cuda_equivalence_runner.cu")


def _robot_modes():
    raw = os.environ.get("GRID_CUDA_FD_DU_SPILL_ROBOTS", "iiwa14:fixed,go2:floating")
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


def _gen_header(robot, build_dir, target):
    """Codegen the no-SO subset at `target` shared-mem bytes (read at __init__).
    Returns (perf_pick, codegen) so callers can read the per-tier metadata."""
    prev = os.environ.get("GRID_CUDA_TARGET_SHARED_MEM_BYTES")
    os.environ["GRID_CUDA_TARGET_SHARED_MEM_BYTES"] = str(target)
    try:
        cg = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid")
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            cg.gen_all_code(algorithm_list=_SUBSET, output_path=str(build_dir / "grid.cuh"))
    finally:
        if prev is None:
            os.environ.pop("GRID_CUDA_TARGET_SHARED_MEM_BYTES", None)
        else:
            os.environ["GRID_CUDA_TARGET_SHARED_MEM_BYTES"] = prev
    return cg.forward_dynamics_gradient_spill_tier_3way[0], cg


def _compile(build_dir, arch):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if shutil.which("nvcc") is None and not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    shutil.copyfile(_RUNNER, build_dir / "runner.cu")
    glass = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    exe = build_dir / "runner.exe"
    cmd = [nvcc, "-std=c++17", "-O0", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           "-DGRID_RUNNER_SKIP_GRADIENTS=0", "-DGRID_RUNNER_SKIP_EEPOSE_GRADIENTS=1",
           f"-I{glass}", f"-I{build_dir}", "-o", str(exe), str(build_dir / "runner.cu")]
    res = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(f"fd_du spill runner compile failed.\n{' '.join(cmd)}\n{res.stderr[-3000:]}")
    return exe


def _run(exe, q, qd, u, nthreads):
    stdin = ("\n".join(" ".join(f"{x:.9g}" for x in v) for v in (q, qd, u))
             + "\n" + ("0 " * max(6 * len(q), 400)) + "\n")
    env = dict(os.environ)
    env["GRID_NTHREADS"] = str(nthreads)
    res = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True, env=env)

    def parse(name):
        m = re.search(rf"BEGIN {name} (\d+) (\d+)\n(.*?)\nEND {name}", res.stdout, re.S)
        if not m:
            pytest.fail(f"runner produced no {name}.\nstderr:\n{res.stderr[-2000:]}")
        return np.array([float(x) for x in m.group(3).split()], dtype=np.float64)

    assert res.returncode == 0, f"runner exited {res.returncode}: {res.stderr[-1500:]}"
    return parse("forward_dynamics_gradient_q"), parse("forward_dynamics_gradient_qd")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_fd_du_output_spill_matches_full(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    pm = build_project_adapter(spec, resolved, base_mode=base_mode)
    robot, nq, nv = pm.robot, pm.nq, pm.nv
    arch = _detect_cuda_arch()

    # FULL (unspilled) reference build at the production target. fd_du's minimal rung
    # is ALWAYS the output-spill one, so per_tier[2] is its arena t-count and the
    # emergency arena is that + 3*nv*nv (the s_dc_du(2nv^2)+s_Minv(nv^2) bands).
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full_pick, cg_full = _gen_header(robot, full_dir, 98304)
    assert full_pick == 0, f"expected fd_du full rung (pick 0) at the default target, got {full_pick}"
    full_exe = _compile(full_dir, arch)

    # OUTPUT-SPILL build: force the rung by codegen'ing one byte below the emergency
    # arena (the 4 arenas decrease with spill: full > selective > emergency > output).
    topo = cg_full.gen_topology_helpers_size()
    output_t = cg_full.forward_dynamics_gradient_t_count_per_tier[2]
    emergency_t = output_t + 3 * nv * nv
    target = _py_arena_bytes(emergency_t, topo) - 1
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    final_pick, _ = _gen_header(robot, spill_dir, target)
    assert final_pick == 3, f"forced target {target} gave fd_du pick {final_pick}, expected 3 (output-spill)"
    htxt = (spill_dir / "grid.cuh").read_text()
    assert "GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()]); s_Minv" in htxt, \
        "output-spill repoint not emitted in the forced header"
    spill_exe = _compile(spill_dir, arch)

    rng = np.random.default_rng(_stable_seed(robot_id))
    for trial in range(3):
        q = rng.uniform(-1.0, 1.0, nq)
        qd = rng.uniform(-1.0, 1.0, nv)
        u = rng.uniform(-1.0, 1.0, nv)
        f_dq, f_dqd = _run(full_exe, q, qd, u, 64)
        s_dq, s_dqd = _run(spill_exe, q, qd, u, 64)
        tag = f"{robot_id}-{base_mode} trial {trial}"
        np.testing.assert_array_equal(s_dq, f_dq, err_msg=f"{tag}: output-spill df_dq != full df_dq")
        np.testing.assert_array_equal(s_dqd, f_dqd, err_msg=f"{tag}: output-spill df_dqd != full df_dqd")

    # thread-count invariance at the output-spill rung (1 thread, a warp, many).
    q = rng.uniform(-1.0, 1.0, nq)
    qd = rng.uniform(-1.0, 1.0, nv)
    u = rng.uniform(-1.0, 1.0, nv)
    base_dq, base_dqd = _run(spill_exe, q, qd, u, 64)
    for nt in (1, 32):
        dq, dqd = _run(spill_exe, q, qd, u, nt)
        np.testing.assert_array_equal(dq, base_dq, err_msg=f"{robot_id} fd_du output-spill df_dq not thread-invariant (nt={nt})")
        np.testing.assert_array_equal(dqd, base_dqd, err_msg=f"{robot_id} fd_du output-spill df_dqd not thread-invariant (nt={nt})")


def _stable_seed(robot_id: str) -> int:
    return 1000 + sum((i + 1) * ord(c) for i, c in enumerate(robot_id))

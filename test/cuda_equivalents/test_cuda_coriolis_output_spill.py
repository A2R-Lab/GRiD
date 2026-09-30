"""coriolis_matrix spill-ladder CUDA regression (h2_plus de-gate).

coriolis_matrix gained a 3-rung surgical tier ladder (mirror crba): rung0 full
(s_coriolis output + inner spatial-recursion scratch in smem); rung1 output-spill
(s_coriolis nv*nv -> L2-pinned SO band, hot inner band stays smem); rung2 whole-band
(both the output and the inner band -> d_workspace). On h2_plus (floating nv=81) the
full arena is ~121KB > sm_120's ~99KB (UNLAUNCHABLE); the output-spill rung is ~94KB
(PERF-launchable) and the whole-band rung ~24KB (LITE/MINIMAL).

The spill RELOCATES buffers; it must not change C(q,qd). This test forces each spill
rung at codegen (a low GRID_CUDA_TARGET_SHARED_MEM_BYTES) and asserts the host-wrapper
output is BIT-IDENTICAL to the unspilled full-smem rung (the oracle-validated path,
checked vs pinocchio in test_cuda_coriolis.py) AND thread-count invariant. A coriolis-
only codegen subset keeps the SO kernels out of the header so the forced-low target does
not trigger their pathological deep-spill compile wall.

Override robots with GRID_CUDA_CORIOLIS_SPILL_ROBOTS="g1:floating,iiwa14:fixed".
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
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch

_RUNNER = Path(__file__).with_name("cuda_coriolis_smoke_runner.cu")


def _robot_modes():
    raw = os.environ.get("GRID_CUDA_CORIOLIS_SPILL_ROBOTS", "g1:floating,iiwa14:fixed")
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
            cg.gen_all_code(algorithm_list=["coriolis_matrix"], output_path=str(build_dir / "grid.cuh"))
    finally:
        if prev is None:
            os.environ.pop("GRID_CUDA_TARGET_SHARED_MEM_BYTES", None)
        else:
            os.environ["GRID_CUDA_TARGET_SHARED_MEM_BYTES"] = prev
    return cg.coriolis_matrix_spill_tier_3way[0], cg


def _compile(build_dir, arch):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if shutil.which("nvcc") is None and not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    shutil.copyfile(_RUNNER, build_dir / "runner.cu")
    glass = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    exe = build_dir / "runner.exe"
    cmd = [nvcc, "-std=c++17", "-O0", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           "-DGRID_CUDA_CORIOLIS_TEST_THREADS=64", f"-I{glass}", f"-I{build_dir}",
           "-o", str(exe), str(build_dir / "runner.cu")]
    res = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(f"coriolis spill runner compile failed.\n{' '.join(cmd)}\n{res.stderr[-3000:]}")
    return exe


def _run(exe, q, qd, nthreads):
    stdin = (" ".join(f"{x:.9g}" for x in q) + "\n" + " ".join(f"{x:.9g}" for x in qd)
             + "\n" + ("0 " * len(qd)) + "\n")
    res = subprocess.run([str(exe), str(nthreads)], input=stdin, capture_output=True, text=True)
    m = re.search(r"BEGIN coriolis \d+ \d+\n(.*?)\nEND coriolis", res.stdout, re.S)
    if not m:
        pytest.fail(f"coriolis runner produced no output (nt={nthreads}).\nstderr:\n{res.stderr[-2000:]}")
    assert res.returncode == 0, f"runner exited {res.returncode}: {res.stderr[-1500:]}"
    return np.array([float(x) for x in m.group(1).split()], dtype=np.float64)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_coriolis_spill_matches_full(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    pm = build_project_adapter(spec, resolved, base_mode=base_mode)
    robot, nq, nv = pm.robot, pm.nq, pm.nv
    arch = _detect_cuda_arch()

    # FULL reference build. The arenas decrease with spill: full > output-spill
    # (full - nv*nv) > whole-band. Force each spill rung one byte under the next-larger arena.
    full_dir = tmp_path / "full"
    full_dir.mkdir()
    full_pick, cg_full = _gen(robot, full_dir, 98304)
    assert full_pick == 0, f"expected coriolis full rung (pick 0) at the default target, got {full_pick}"
    topo = cg_full.gen_topology_helpers_size()
    full_t = cg_full.coriolis_matrix_t_count_per_tier[0]
    out_t = full_t - nv * nv  # output-spill arena
    full_exe = _compile(full_dir, arch)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_pick, _ = _gen(robot, out_dir, _py_arena_bytes(full_t, topo) - 1)
    assert out_pick == 1, f"forced target did not land coriolis output-spill rung (got {out_pick})"
    out_exe = _compile(out_dir, arch)

    whole_dir = tmp_path / "whole"
    whole_dir.mkdir()
    whole_pick, _ = _gen(robot, whole_dir, _py_arena_bytes(out_t, topo) - 1)
    assert whole_pick == 2, f"forced target did not land coriolis whole-band rung (got {whole_pick})"
    whole_exe = _compile(whole_dir, arch)

    rng = np.random.default_rng(_stable_seed(robot_id))
    for trial in range(3):
        q = rng.uniform(-1.0, 1.0, nq)
        qd = rng.uniform(-1.0, 1.0, nv)
        ref = _run(full_exe, q, qd, 64)
        tag = f"{robot_id}-{base_mode} trial {trial}"
        np.testing.assert_array_equal(_run(out_exe, q, qd, 64), ref,
                                      err_msg=f"{tag}: output-spill C != full C")
        np.testing.assert_array_equal(_run(whole_exe, q, qd, 64), ref,
                                      err_msg=f"{tag}: whole-band C != full C")

    # thread-count invariance at each spill rung.
    q = rng.uniform(-1.0, 1.0, nq)
    qd = rng.uniform(-1.0, 1.0, nv)
    for label, exe in (("output-spill", out_exe), ("whole-band", whole_exe)):
        base = _run(exe, q, qd, 64)
        for nt in (1, 32):
            np.testing.assert_array_equal(_run(exe, q, qd, nt), base,
                                          err_msg=f"{robot_id} coriolis {label} not thread-invariant (nt={nt})")


def _stable_seed(robot_id: str) -> int:
    return 1000 + sum((i + 1) * ord(c) for i, c in enumerate(robot_id))

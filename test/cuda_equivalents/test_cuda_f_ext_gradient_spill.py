"""f_ext_gradient + f_ext_gradient_dq spill-ladder CUDA regression (h2_plus de-gate).

The f_ext-gradient family blows the sm_120 ~99 KB smem cap on big robots. On h2_plus
(floating nv=81) the FIRST-order kernel needs ~361 KB (two nv x 6NB outputs + minv's
6*nv*nv F-region) and the dq kernel ~311 KB (the s_JTp/s_JTm -J^T FD pair) -- both
UNLAUNCHABLE. Two surgical ladders fix this:

  * first-order: 3-rung (rung 0 full; rung 1 spill s_dqdd; rung 2 ALSO spill s_dtau +
    route minv's F-region to the GRAD-section minv-F workspace offset). h2_plus lands
    rung 2 at ~74 KB; mid robots stop at rung 1; small robots keep rung 0.
  * dq: 2-rung (rung 0 full; rung 1 spill the s_JTp/s_JTm pair to the SO band).
    h2_plus lands rung 1 at ~50 KB.

The spill RELOCATES buffers (smem -> L2-pinned d_workspace); it must not change the
result. f_ext is fully DETERMINISTIC (RNEA-backprop -J^T + minv + a dense GEMM + central
FD), so this forces the deepest rung at codegen (a low GRID_CUDA_TARGET_SHARED_MEM_BYTES)
and asserts the host-wrapper outputs are BIT-IDENTICAL to the unspilled full-smem rung
(the oracle-validated path, checked vs pinocchio in test_cuda_f_ext_gradient_equivalence.py).
A minimal id/minv/f_ext_gradient codegen subset keeps the SO kernels out of the header so
the forced-low target does not trigger their pathological deep-spill compile wall.

h2_plus itself is launch-only: its full rung never fits, so it is generated at the deep
rung and checked for a finite, non-crashing launch (the de-gate proof).

Override the bit-equality robots with GRID_CUDA_FEG_SPILL_ROBOTS="g1:floating,iiwa14:fixed".
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from grid_codegen import GRiDCodeGenerator
from URDFParser import URDFParser
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.tests.state_sampling import build_dynamics_samples
from test.cuda_equivalents.test_cuda_executable_equivalence import _detect_cuda_arch
from test.cuda_equivalents.test_cuda_f_ext_gradient_equivalence import _parse_runner_output

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from config import robot_urdf

_RUNNER = Path(__file__).with_name("cuda_f_ext_gradient_runner.cu")
_SUBSET = ["inverse_dynamics", "minv", "f_ext_gradient"]  # no SO kernels -> no deep-spill compile wall
_OUTPUTS = ("f_ext_gradient_dtau_dfext", "f_ext_gradient_dqdd_dfext",
            "f_ext_gradient_did_du_dfext_dq")


def _bit_equal_robots():
    raw = os.environ.get("GRID_CUDA_FEG_SPILL_ROBOTS", "g1:floating,iiwa14:fixed")
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


def _gen(robot, build_dir, target):
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
    return cg


def _compile(build_dir, arch, floating):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if shutil.which("nvcc") is None and not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    shutil.copyfile(_RUNNER, build_dir / "runner.cu")
    repo = Path(__file__).resolve().parents[2]
    exe = build_dir / "runner.exe"
    cmd = [nvcc, "-std=c++17", "-O0", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           f"-DGRID_CUDA_FLOATING_BASE={1 if floating else 0}",
           "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
           f"-I{repo}", f"-I{build_dir}", "-o", str(exe), str(build_dir / "runner.cu")]
    res = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(f"f_ext spill runner compile failed.\n{' '.join(cmd)}\n{res.stderr[-3000:]}")
    return exe


def _run(exe, q):
    stdin = " ".join(f"{x:.9g}" for x in q) + "\n"
    res = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True)
    combined = (res.stdout + res.stderr).lower()
    if res.returncode != 0:
        if "shared-memory request" in combined and "this device supports" in combined:
            pytest.skip("Kernel smem request exceeds this GPU's per-block cap.")
        pytest.fail(f"f_ext spill runner exited {res.returncode}: {res.stderr[-2000:]}")
    return _parse_runner_output(res.stdout)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _bit_equal_robots(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_f_ext_gradient_spill_matches_full(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    pm = build_project_adapter(spec, resolved, base_mode=base_mode)
    robot = pm.robot
    floating = base_mode == "floating"
    arch = _detect_cuda_arch()
    nq = robot.get_num_pos()  # runner reads NUM_JOINTS == NUM_POS values

    # REFERENCE build at the real sm_120 target -> the perf rung that ACTUALLY runs
    # on this GPU (rung 0 full for small robots; rung 1 out-spill for g1, whose full
    # ~99 KB arena exceeds the cap). This rung is the one the default-tier equivalence
    # test validates vs the pinocchio oracle, so deep == reference transitively proves
    # deep == oracle. (g1's true full rung 0 is un-runnable, so it cannot be the ref.)
    full_dir = tmp_path / "ref"
    full_dir.mkdir()
    cg_full = _gen(robot, full_dir, 98304)
    assert cg_full.f_ext_gradient_spill_tier_3way[0] <= 1, \
        f"reference build must be a runnable (<=rung 1) perf rung, got {cg_full.f_ext_gradient_spill_tier_3way}"
    full_exe = _compile(full_dir, arch, floating)

    # DEEP build: a tiny target forces the first-order deep rung (2) AND the dq spill rung (1).
    deep_dir = tmp_path / "deep"
    deep_dir.mkdir()
    cg_deep = _gen(robot, deep_dir, 2048)
    assert cg_deep.f_ext_gradient_spill_tier_3way[0] == 2, \
        f"forced target did not land f_ext deep rung (got {cg_deep.f_ext_gradient_spill_tier_3way})"
    assert cg_deep.f_ext_gradient_dq_spill_tier_3way[0] == 1, \
        f"forced target did not land dq spill rung (got {cg_deep.f_ext_gradient_dq_spill_tier_3way})"
    htxt = (deep_dir / "grid.cuh").read_text()
    assert "GRID_SO_WORKSPACE_TEMP_OFFSET_BYTES<T>()" in htxt and "GRID_MINV_F_WORKSPACE_OFFSET_BYTES<T>()" in htxt, \
        "spill repoints (output SO band + minv-F) not emitted in the forced header"
    deep_exe = _compile(deep_dir, arch, floating)

    sample = build_dynamics_samples(pm)[1]
    q = np.asarray(sample.q, dtype=np.float64)[:nq]
    full_out = _run(full_exe, q)
    deep_out = _run(deep_exe, q)
    # dtau (-J^T, lane-0 serial reduce) and did_du (central FD of -J^T) are q-only and
    # fully DETERMINISTIC -> assert BIT-IDENTICAL (the strongest output-spill relocation
    # check). dqdd = -Minv @ dtau depends on minv, whose GLASS articulated-body reduction
    # reorders fp32 adds (the same inversion/atomic non-determinism osc_inertia hit, §14):
    # it varies ~1e-4 run-to-run independent of spill, so it is checked vs the reference
    # at a float32 tolerance, NOT bit-equality. (The reference rung is itself oracle-
    # validated by test_cuda_f_ext_gradient_equivalence.py, so deep ~= reference ~= oracle.)
    for name in ("f_ext_gradient_dtau_dfext", "f_ext_gradient_did_du_dfext_dq"):
        assert name in full_out and name in deep_out, f"missing output {name}"
        np.testing.assert_array_equal(
            np.asarray(deep_out[name], dtype=np.float64),
            np.asarray(full_out[name], dtype=np.float64),
            err_msg=f"{robot_id}-{base_mode}: f_ext deep-rung output {name} != reference rung")
    # dqdd is checked against the ANALYTIC oracle directly (not the reference rung): this
    # proves the deep rung is correct in ABSOLUTE terms and is robust to minv's float
    # non-determinism (deep != reference by ~1e-4 is just two independent minv evals; a
    # real relocation bug would diverge far more than the float32 oracle tolerance).
    oracle_dqdd = np.asarray(pm.f_ext_gradient(q)[1], dtype=np.float64).reshape(-1)
    got = np.asarray(deep_out["f_ext_gradient_dqdd_dfext"], dtype=np.float64).reshape(-1)
    scale = max(1.0, float(np.max(np.abs(oracle_dqdd))))
    np.testing.assert_allclose(
        got, oracle_dqdd, rtol=5e-3, atol=5e-3 * scale,
        err_msg=f"{robot_id}-{base_mode}: f_ext deep-rung dqdd_dfext diverged from the analytic oracle")


def _h2plus_robot():
    urdf = robot_urdf("h2_plus")
    if not urdf.exists():
        pytest.skip("h2_plus.urdf not vendored")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return URDFParser().parse(str(urdf), floating_base=True)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_cuda_f_ext_gradient_h2plus_launches(tmp_path):
    """h2_plus's full rung never fits; generate the deep rung and prove a finite launch."""
    robot = _h2plus_robot()
    arch = _detect_cuda_arch()
    nq = robot.get_num_pos()
    cg = _gen(robot, tmp_path, 98304)  # at the real sm_120 target, perf MUST already spill
    assert cg.f_ext_gradient_spill_tier_3way[0] == 2, \
        f"h2_plus first-order did not land the deep rung at the sm_120 target (got {cg.f_ext_gradient_spill_tier_3way})"
    assert cg.f_ext_gradient_dq_spill_tier_3way[0] == 1, \
        f"h2_plus dq did not land the spill rung at the sm_120 target (got {cg.f_ext_gradient_dq_spill_tier_3way})"
    exe = _compile(tmp_path, arch, floating=True)
    # identity floating config: xyz=0, quat xyzw=(0,0,0,1), joints=0 -> finite, valid.
    q = np.zeros(nq, dtype=np.float64)
    q[6] = 1.0
    out = _run(exe, q)
    for name in _OUTPUTS:
        assert name in out, f"h2_plus missing output {name}"
        arr = np.asarray(out[name], dtype=np.float64)
        assert np.all(np.isfinite(arr)), f"h2_plus {name} has non-finite entries"

"""CUDA equivalence gate for W1b `grid::multi_target_position` (batched world positions).

A TARGET is (anchor_jid, local offset). The batched inner builds ONE shared FK pass
(emit_world_fk_chainup, shared with the ee-pose gradient) then extracts each target's
world position = Xw[anchor] @ [offset, 1] in parallel over targets. This gate certifies:

  1. ORACLE: every target's GPU world position matches a NumPy world-FK oracle
     (RBDReference._frame_world_placement_and_chain -> Xw[anchor] @ [offset,1]).
  2. offset==0 == end_effector_pose: a target anchored at a leaf with a ZERO offset
     equals that end-effector's end_effector_pose position (rows 0..2) -- so the batch
     path subsumes the single-named-EE path (backlog D).
  3. THREAD-INVARIANCE: identical output at 1 / 32 / 256 threads (checked in-runner).

baxter (leaves [0,7,14]) exercises the MULTI-ANCHOR branched-tree path; iiwa14 (one
leaf, serial chain) is the offset==0 single-EE control. Fixed-base, correctness only.
Override robots with GRID_CUDA_MULTITARGET_ROBOTS.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from GRiDCodeGenerator import GRiDCodeGenerator
from RBDReference import RBDReference
from test.cuda_equivalents.test_cuda_executable_equivalence import _detect_cuda_arch
from RBDReference.tests.model_sources import resolve_robot_spec, iter_robot_cases
from RBDReference.tests import MANIFEST_PATH
from RBDReference.equivalents.reference_backend import build_project_adapter


RUNNER_SOURCE = Path(__file__).with_name("cuda_multi_target_position_runner.cu")


def _robot_cells():
    override = os.environ.get("GRID_CUDA_MULTITARGET_ROBOTS")
    if override:
        return [(r.strip(), "fixed") for r in override.split(",") if r.strip()]
    return [("iiwa14", "fixed"), ("baxter", "fixed")]


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not in manifest")


def _build_batch(robot):
    """One (leaf, offset=0) control target per leaf (leaf order == ee order), then one
    (leaf, nonzero offset) target per leaf. Returns (targets, n_leaves)."""
    leaves = robot.get_leaf_nodes()
    targets = []
    for e, jid in enumerate(leaves):                 # offset==0 controls, ee order
        targets.append({"anchor_jid": int(jid), "offset": (0.0, 0.0, 0.0)})
    for e, jid in enumerate(leaves):                 # nonzero offsets exercise the epilogue
        targets.append({"anchor_jid": int(jid),
                        "offset": (0.03 + 0.01 * e, -0.02, 0.05 + 0.02 * e)})
    return targets, len(leaves)


def _generate_header(robot, targets, build_dir):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(robot, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(codegen_profile="all", output_path=str(header),
                             multi_target_batch=targets)
    return header


def _compile_runner(build_dir):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_multi_target_position_runner.exe"
    cmd = [
        nvcc, "-std=c++17", "-O2",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-I", str(build_dir), "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "cuda_multi_target_position_runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable


def _parse(stdout):
    mt, ee = {}, {}
    for line in stdout.splitlines():
        p = line.split()
        if len(p) == 5 and p[0] == "MT":
            mt[int(p[1])] = np.array([float(p[2]), float(p[3]), float(p[4])])
        elif len(p) == 5 and p[0] == "EE":
            ee[int(p[1])] = np.array([float(p[2]), float(p[3]), float(p[4])])
    return mt, ee


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("robot_id,base_mode", _robot_cells(), ids=lambda v: str(v))
def test_multi_target_position(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    robot = project_model.robot

    targets, n_leaves = _build_batch(robot)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_multitarget"
    build_dir.mkdir()
    _generate_header(robot, targets, build_dir)
    executable = _compile_runner(build_dir)

    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, (
        "multi_target_position runner FAILED (thread-variance or CUDA error).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # forced-spill gate: TIER_MINIMAL routes the FK scratch to d_workspace; the batched
    # output must be BIT-identical to TIER_SHARED (whole-arena spill only relocates memory).
    spill = next((l for l in result.stdout.splitlines() if l.startswith("SPILLDIFF")), None)
    assert spill is not None, f"runner emitted no SPILLDIFF line:\n{result.stdout}"
    assert float(spill.split("maxdiff=")[1].split()[0]) == 0.0, f"spill not bit-identical: {spill}"

    mt, ee = _parse(result.stdout)
    assert len(mt) == len(targets), f"expected {len(targets)} MT rows, got {len(mt)}"

    # NumPy world-FK oracle: same q the runner used (q[i] = 0.2 sin(0.7 i) + 0.1).
    nq = robot.get_num_pos()
    q = np.array([0.2 * np.sin(0.7 * i) + 0.1 for i in range(nq)], dtype=np.float64)
    Xw, _ = RBDReference(robot)._frame_world_placement_and_chain(q)

    for t, tgt in enumerate(targets):
        anc = tgt["anchor_jid"]
        off = np.array([*tgt["offset"], 1.0], dtype=np.float64)
        oracle = (np.asarray(Xw[anc], dtype=np.float64) @ off)[:3]
        np.testing.assert_allclose(
            mt[t], oracle, rtol=1e-10, atol=1e-10,
            err_msg=f"{robot_id} target {t} (anchor {anc}, offset {tgt['offset']}) "
                    f"world position != oracle")

    # offset==0 controls (targets 0..n_leaves-1, leaf order) must equal end_effector_pose.
    for e in range(n_leaves):
        np.testing.assert_allclose(
            mt[e], ee[e], rtol=1e-10, atol=1e-10,
            err_msg=f"{robot_id} offset==0 target {e} != end_effector_pose position")

    print(f"{robot_id}: {len(targets)} targets vs oracle + {n_leaves} offset==0==ee_pose OK\n{result.stdout}")

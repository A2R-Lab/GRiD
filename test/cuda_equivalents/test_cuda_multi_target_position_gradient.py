"""CUDA equivalence gate for W2a `grid::multi_target_position_gradient` (batched d pos/dv).

The gradient is an anchor-deduped geometric Jacobian (built ONCE per distinct anchor via the
shared `emit_geometric_jacobian_jvjw`, reused from the ee-pose gradient) plus a per-target
offset epilogue `dpos[t][:,vi] = Jv[anchor,:,vi] + Jw[anchor,:,vi] x (R_world[anchor]·r)` --
no FK re-walk. This gate certifies:

  1. FD ORACLE: every target's GPU gradient matches a central-difference of the W1b world-
     position oracle (Xw[anchor] @ [offset,1]) w.r.t. q (fixed-base: d/dv == d/dq).
  2. offset==0 == end_effector_pose_gradient: a target anchored at a leaf with ZERO offset
     equals that ee's ee_pose_gradient rows 0..2 BIT-IDENTICALLY (same Jv fill code) -- so the
     batch gradient subsumes the single-EE Jacobian.
  3. THREAD-INVARIANCE: identical output at 1 / 32 / 256 threads (checked in-runner).

baxter (leaves [0,7,14]) exercises the MULTI-ANCHOR dedup; iiwa14 is the single-anchor control.
Fixed-base non-mimic (so vi == qi), correctness only. Override with GRID_CUDA_MULTITARGET_ROBOTS.
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


RUNNER_SOURCE = Path(__file__).with_name("cuda_multi_target_position_gradient_runner.cu")


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
    leaves = robot.get_leaf_nodes()
    targets = [{"anchor_jid": int(j), "offset": (0.0, 0.0, 0.0)} for j in leaves]
    targets += [{"anchor_jid": int(j), "offset": (0.03 + 0.01 * e, -0.02, 0.05 + 0.02 * e)}
                for e, j in enumerate(leaves)]
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
    executable = build_dir / "cuda_multi_target_position_gradient_runner.exe"
    cmd = [
        nvcc, "-std=c++17", "-O2",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-I", str(build_dir), "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "cuda_multi_target_position_gradient_runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable


def _parse(stdout):
    mtg, eeg = {}, {}
    for line in stdout.splitlines():
        p = line.split()
        if len(p) == 6 and p[0] == "MTG":
            mtg[(int(p[1]), int(p[2]))] = np.array([float(p[3]), float(p[4]), float(p[5])])
        elif len(p) == 6 and p[0] == "EEG":
            eeg[(int(p[1]), int(p[2]))] = np.array([float(p[3]), float(p[4]), float(p[5])])
    return mtg, eeg


def _pos(Xw, anchor, offset):
    off = np.array([*offset, 1.0], dtype=np.float64)
    return (np.asarray(Xw[anchor], dtype=np.float64) @ off)[:3]


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("robot_id,base_mode", _robot_cells(), ids=lambda v: str(v))
def test_multi_target_position_gradient(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    robot = project_model.robot

    targets, n_leaves = _build_batch(robot)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_multitarget_grad"
    build_dir.mkdir()
    _generate_header(robot, targets, build_dir)
    executable = _compile_runner(build_dir)

    result = subprocess.run([str(executable)], capture_output=True, text=True)
    assert result.returncode == 0, (
        "multi_target_position_gradient runner FAILED.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # forced-spill gate: TIER_MINIMAL routes the Jacobian scratch to d_workspace; the batched
    # gradient must be BIT-identical to TIER_SHARED (whole-arena spill only relocates memory).
    spill = next((l for l in result.stdout.splitlines() if l.startswith("SPILLDIFF")), None)
    assert spill is not None, f"runner emitted no SPILLDIFF line:\n{result.stdout}"
    assert float(spill.split("maxdiff=")[1].split()[0]) == 0.0, f"spill not bit-identical: {spill}"

    mtg, eeg = _parse(result.stdout)
    nv = robot.get_num_vel()
    nq = robot.get_num_pos()
    assert nq == nv, "gate assumes fixed-base (vi == qi)"
    assert len(mtg) == len(targets) * nv, f"expected {len(targets)*nv} MTG rows, got {len(mtg)}"

    ref = RBDReference(robot)
    q = np.array([0.2 * np.sin(0.7 * i) + 0.1 for i in range(nq)], dtype=np.float64)

    # central-difference FD oracle of the W1b world-position oracle w.r.t. q_vi
    eps = 1e-6
    for t, tgt in enumerate(targets):
        anc, off = tgt["anchor_jid"], tgt["offset"]
        for vi in range(nv):
            qp = q.copy(); qp[vi] += eps
            qm = q.copy(); qm[vi] -= eps
            Xp, _ = ref._frame_world_placement_and_chain(qp)
            Xm, _ = ref._frame_world_placement_and_chain(qm)
            fd = (_pos(Xp, anc, off) - _pos(Xm, anc, off)) / (2 * eps)
            np.testing.assert_allclose(
                mtg[(t, vi)], fd, rtol=1e-5, atol=1e-6,
                err_msg=f"{robot_id} grad target {t} (anchor {anc}) col vi={vi} != FD oracle")

    # offset==0 controls (targets 0..n_leaves-1, leaf order) must equal ee_pose_gradient
    # rows 0..2 BIT-IDENTICALLY (same Jv fill on the same q).
    for e in range(n_leaves):
        for vi in range(nv):
            np.testing.assert_allclose(
                mtg[(e, vi)], eeg[(e, vi)], rtol=1e-9, atol=1e-11,
                err_msg=f"{robot_id} offset==0 target {e} col {vi} != ee_pose_gradient")

    print(f"{robot_id}: {len(targets)} targets x {nv} cols vs FD + {n_leaves} offset==0==ee_grad OK")

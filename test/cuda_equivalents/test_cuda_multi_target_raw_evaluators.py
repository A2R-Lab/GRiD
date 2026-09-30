"""GATO nit 2 (2026-09-24): grid_plant::multi_target_position[_gradient] — the
caller-scratch RAW evaluators over the DEFAULT multi-target batch (the
multi_target_batch option / collision spheres), emitted by the same emitter as
the contact-frame pair. GATO's hand-carved `ee_carve` FK composed exactly this
(load XmatsHom, run the multi-target inner) and can now call these instead.

Checks (cuda_multi_target_raw_runner.cu, derived from the contact-frame runner):
PLANTDIFF == 0 (plant wrapper == generated *_device fn, positions AND
Jacobians), THREADINV == 0 (thread-count invariance), SPILLDIFF == 0
(TIER_MINIMAL spill path == TIER_SHARED). The device fns themselves are
oracle-checked in test_cuda_multi_target_position[_gradient].py; here the
question is only that the raw wrappers compose them identically.
Local gate policy: iiwa14 fixed (serial) + go2 floating (branched).
"""
from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from grid_codegen import GRiDCodeGenerator
from URDFParser import URDFParser
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch
from test.cuda_equivalents.test_cuda_contact_frame_positions import _q_for
from test.cuda_equivalents.test_cuda_multi_target_position import _build_batch

REPO = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = Path(__file__).with_name("cuda_multi_target_raw_runner.cu")
CELLS = [("iiwa14", False), ("go2", True)]


def _robot(name, floating):
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(str(REPO / "config" / "robot_assets" / f"{name}.urdf"), floating_base=floating)


def _build(robot, targets, build_dir):
    gen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid")
    with contextlib.redirect_stdout(io.StringIO()):
        gen.gen_all_code(algorithm_list=["end_effector_pose"], multi_target_batch=targets,
                         output_path=str(build_dir / "grid.cuh"), enable_mujoco_kernels=False)
    header = (build_dir / "grid.cuh").read_text()
    assert "void multi_target_position(T *s_pos, const T *s_q, T *s_scratch, " in header
    assert "void multi_target_position_gradient(T *s_pos, T *s_dpos, const T *s_q, T *s_scratch, " in header
    runner = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner)
    exe = build_dir / "runner.exe"
    arch = _detect_cuda_arch()
    r = subprocess.run(["nvcc", "-std=c++17", "-O2", f"-arch=sm_{arch}", "-I", str(build_dir),
                        "-o", str(exe), str(runner)], capture_output=True, text=True)
    assert r.returncode == 0, f"runner compilation failed:\n{r.stderr[-4000:]}"
    return exe


def _checks(stdout):
    out = {}
    for line in stdout.splitlines():
        p = line.split()
        if p and p[0] in ("THREADINV", "SPILLDIFF", "PLANTDIFF"):
            out[p[0]] = float(p[1].split("=")[1])
    return out


def test_header_without_a_multi_target_batch_emits_no_raw_evaluators(tmp_path):
    robot = _robot("iiwa14", False)
    gen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid")
    with contextlib.redirect_stdout(io.StringIO()):
        gen.gen_all_code(algorithm_list=["end_effector_pose"], output_path=str(tmp_path / "g.cuh"),
                         enable_mujoco_kernels=False)
    h = (tmp_path / "g.cuh").read_text()
    assert "void multi_target_position(" not in h and "GATO nit 2" not in h


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("name,floating", CELLS, ids=[f"{c[0]}-{'floating' if c[1] else 'fixed'}" for c in CELLS])
def test_raw_multi_target_evaluators_equal_the_device_fns(tmp_path, name, floating):
    if shutil.which("nvcc") is None:
        pytest.skip("nvcc not on PATH")
    robot = _robot(name, floating)
    targets, _ = _build_batch(robot)
    build_dir = tmp_path / f"{name}_mt_raw"; build_dir.mkdir()
    exe = _build(robot, targets, build_dir)
    q = _q_for(robot, floating)
    (build_dir / "q.txt").write_text("\n".join(f"{v:.17g}" for v in q) + "\n")
    r = subprocess.run([str(exe)], capture_output=True, text=True, cwd=build_dir)
    assert r.returncode == 0, f"runner self-checks failed:\n{r.stdout}\n{r.stderr}"
    checks = _checks(r.stdout)
    assert checks == {"THREADINV": 0.0, "SPILLDIFF": 0.0, "PLANTDIFF": 0.0}, checks
    P = np.array([[float(x) for x in l.split()[2:5]] for l in r.stdout.splitlines() if l.startswith("P ")])
    assert P.shape == (len(targets), 3) and np.isfinite(P).all()

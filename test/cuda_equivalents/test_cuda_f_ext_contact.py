"""CUDA FD-oracle gate for the contact-FRAME f_ext map (GATO ask 1, C.2). Robot: go2-FLOATING.

GRiD already had the f_ext DERIVATIVE stack (dtau/dfext = -J^T, dqdd/dfext = M^-1 J^T, -dJ^T/dq), but all
of it speaks the f_ext SLOT convention: a per-body wrench in that body's JOINT-LOCAL Featherstone frame.
A solver's decision variable is a contact force at a DESIGNATED FRAME with WORLD-ALIGNED axes. This module
emits that map and BOTH its derivatives, so the chain rule closes:

    f_ext[b] += [ R^T n_w + r_c x (R^T f_w) ; R^T f_w ]        (moment transported to the joint origin)
    d(f_ext)/d(f_c) = [[R^T, skew(r_c) R^T], [0, R^T]]         (f_c-INDEPENDENT -- the map is linear)
    d(f_ext)/dq_v   = [ -w_v x g - r_c x (w_v x h) ; -w_v x h ]  (the term solvers drop)

Gated on go2-FLOATING deliberately: this touches the world-FK chain-up + tier arena, the exact surface
where BOTH of this month's silent bugs lived (§1s uninit-smem read on BRANCHED trees; §1t arena
under-count on FLOATING base). Both compiled clean and passed every fixed-base test. An arm cannot fail
this test in the ways that matter.

Checks: value; f_c-linearity (bit-exact); d/d(f_c) vs central FD + bit-exact f_c-independence;
d/dq vs central FD with an SE(3) RETRACT perturbation (a componentwise q[i]+=h leaves the manifold and
would yield a wrong oracle); and that uncontacted bodies have exactly-zero rows.
See cuda_f_ext_contact_runner.cu.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from GRiDCodeGenerator import GRiDCodeGenerator
from GRiDCodeGenerator.algorithms._f_ext_contact import contact_frames_from_urdf
from test.cuda_equivalents.test_cuda_executable_equivalence import _detect_cuda_arch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import robot_urdf
RUNNER_SOURCE = Path(__file__).with_name("cuda_f_ext_contact_runner.cu")
GO2_FEET = ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_f_ext_contact_frame_map_fd(tmp_path):
    from URDFParser import URDFParser
    urdf = robot_urdf("go2")
    if not urdf.exists():
        pytest.skip("go2.urdf not found")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(urdf), floating_base=True)
        if robot is None:
            pytest.skip("go2 URDF parse failed")
        frames = contact_frames_from_urdf(robot, GO2_FEET)
        build_dir = tmp_path / "f_ext_contact"
        build_dir.mkdir()
        header = build_dir / "grid.cuh"
        GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
            codegen_profile="all", output_path=str(header), contact_frames=frames)

    assert len(frames) == 4, f"expected 4 go2 foot frames, resolved {len(frames)}"

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_f_ext_contact_runner.exe"
    cmd = [nvcc, "-std=c++17", "-O2", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           "-I", str(build_dir), "-I", str(REPO_ROOT), "-o", str(exe), str(runner_copy)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, f"nvcc build failed:\n{build.stderr}"

    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=900)
    assert "RESULT: PASS" in run.stdout, f"runner failed:\n{run.stdout}\n{run.stderr}"
    assert run.returncode == 0, f"runner exit {run.returncode}:\n{run.stdout}"

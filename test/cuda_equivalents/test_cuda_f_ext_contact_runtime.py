"""CUDA FD-oracle gate for the RUNTIME single-contact f_ext map (welded-tool tip). Robot: go2-FLOATING.

The runtime path shares the baked family's map/derivative math, but the contact body `b` and local
offset `r_c` are RUNTIME arguments. This gate drives the *_runtime device fns with a real go2 foot's
(jid, offset) resolved at build time (passed to the runner via -D defines) and checks value / d/df_c /
d/dq against finite differences (SE(3) retract in q), exactly as the baked gate does. go2-FLOATING is
deliberate (branched + floating -> the §1s/§1t surface).
"""
from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from grid_codegen import GRiDCodeGenerator
from grid_codegen.algorithms._f_ext_contact import contact_frames_from_urdf
from test.cuda_equivalents.test_cuda_executable_equivalence import _detect_cuda_arch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import robot_urdf
RUNNER_SOURCE = Path(__file__).with_name("cuda_f_ext_contact_runtime_runner.cu")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
def test_f_ext_contact_runtime_map_fd(tmp_path):
    from URDFParser import URDFParser
    urdf = robot_urdf("go2")
    if not urdf.exists():
        pytest.skip("go2.urdf not found")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        robot = URDFParser().parse(str(urdf), floating_base=True)
        if robot is None:
            pytest.skip("go2 URDF parse failed")
        # resolve one real foot frame -> a runtime (jid, offset) for the tool tip.
        frames = contact_frames_from_urdf(robot, ["FR_foot_joint"])
        build_dir = tmp_path / "f_ext_contact_runtime"
        build_dir.mkdir()
        header = build_dir / "grid.cuh"
        # SPLIT codegen: runtime-contact emission is enable_contact_runtime-driven
        # inside the kinematics region (needs one ee key); the runner also calls
        # f_ext_gradient_device and grid_integrate_floating_q ("integrator" pulls
        # the SE(3) lie helpers on this floating base).
        GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
            algorithm_list=["f_ext_gradient", "end_effector_pose", "integrator"],
            output_path=str(header), enable_contact_runtime=True)

    tjid = frames[0]["jid"]
    rc = frames[0]["offset"]

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_f_ext_contact_runtime_runner.exe"
    cmd = [nvcc, "-std=c++17", "-O2", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
           f"-DTOOL_JID={int(tjid)}",
           f"-DTOOL_RC0={rc[0]:.10g}", f"-DTOOL_RC1={rc[1]:.10g}", f"-DTOOL_RC2={rc[2]:.10g}",
           "-I", str(build_dir), "-I", str(REPO_ROOT), "-o", str(exe), str(runner_copy)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, f"nvcc build failed:\n{build.stderr}"

    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=900)
    assert "RESULT: PASS" in run.stdout, f"runner failed:\n{run.stdout}\n{run.stderr}"
    assert run.returncode == 0, f"runner exit {run.returncode}:\n{run.stdout}"

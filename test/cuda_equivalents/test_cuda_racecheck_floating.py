"""compute-sanitizer racecheck gate on the FLOATING-base composition (GATO nit 3,
2026-09-24). GATO reported tens of thousands of warp-level "Potential WAW hazard"
warnings inside its merit kernel on go2 floating (loaders + integrator_inner).
GRiD's own composition — forward_dynamics_device, end_effector_pose_device
(XmatsHom loader) and integrator_device — runs at 0 hazards at 32/64/128/256/512
threads (verified 2026-09-24 at tip 9d6ef86). This test keeps it that way: a
regression here is a real intra-block race (or a loader rewrite that needs a
__syncwarp), never something to allow-list. Local gate policy: go2 floating.
"""
from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
from pathlib import Path

import pytest

from grid_codegen import GRiDCodeGenerator
from URDFParser import URDFParser
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch

REPO = Path(__file__).resolve().parents[2]
RUNNER_SOURCE = Path(__file__).with_name("cuda_racecheck_floating_runner.cu")


def _sanitizer():
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not on PATH")
    cs = Path(nvcc).resolve().parent / "compute-sanitizer"
    assert cs.exists(), f"compute-sanitizer ships with the CUDA toolkit but is missing at {cs}"
    return str(cs)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.floating_base
def test_go2_floating_composition_is_racecheck_clean(tmp_path):
    cs = _sanitizer()
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(REPO / "config" / "robot_assets" / "go2.urdf"), floating_base=True)
        gen = GRiDCodeGenerator(robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid")
        gen.gen_all_code(algorithm_list=["forward_dynamics", "end_effector_pose", "integrator"],
                         output_path=str(tmp_path / "grid.cuh"), enable_mujoco_kernels=False)
    runner = tmp_path / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner)
    exe = tmp_path / "runner.exe"
    r = subprocess.run(["nvcc", "-std=c++17", "-O2", "-lineinfo", f"-arch=sm_{_detect_cuda_arch()}",
                        "-I", str(tmp_path), "-o", str(exe), str(runner)], capture_output=True, text=True)
    assert r.returncode == 0, f"runner compilation failed:\n{r.stderr[-4000:]}"
    for threads in (32, 256):
        r = subprocess.run([cs, "--tool", "racecheck", "--print-limit", "50", str(exe), str(threads)],
                           capture_output=True, text=True, cwd=tmp_path)
        assert r.returncode == 0 and "OK" in r.stdout, f"runner failed at {threads} threads:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
        assert "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)" in r.stdout, \
            f"racecheck hazards at {threads} threads:\n{r.stdout[-6000:]}"

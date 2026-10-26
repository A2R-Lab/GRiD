"""G13 numerical gate: all-prismatic fixed tool, block and warp/thread FK."""
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from URDFParser import URDFParser
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]


@pytest.mark.parametrize("mimic", [False, True])
def test_prismatic_fixed_tool_values_and_thread_invariance(tmp_path, mimic):
    robot = URDFParser().parse(str(robot_urdf("slider_mimic_fixed_tool" if mimic else "slider_fixed_tool")))
    gen = GRiDCodeGenerator(robot, False, False, "grid")
    gen.gen_all_code(output_path=str(tmp_path / "grid.cuh"),
                     algorithm_list=["end_effector_pose", "end_effector_pose_gradient"],
                     include_homogenous_transforms=True, fixed_target_name="tool",
                     enable_mujoco_kernels=False)
    runner = Path(__file__).with_name("cuda_prismatic_fixed_target_runner.cu")
    exe = tmp_path / "slider.exe"
    arch = _detect_cuda_arch()
    nvcc = shutil.which("nvcc")
    assert nvcc, "CUDA toolkit required for this numerical gate"
    compiled = subprocess.run([nvcc, "-std=c++17", "-O2", f"-arch=sm_{arch}",
                               f"-I{tmp_path}", str(runner), "-o", str(exe)],
                              capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    r = Rotation.from_euler("xyz", [0.3, 0.2, 0.1])
    p, axis = np.array([0.04, 0.01, 0.02]), np.array([0.6, 0., 0.8])
    for q in (0., 0.13, 0.5):
        displacement = 0.5 * q + 0.17 if mimic else q
        expected = np.r_[p + r.apply(axis * displacement + p), (r * r).as_euler("xyz"),
                         (0.5 if mimic else 1.) * r.apply(axis), np.zeros(3)]
        quat = r.as_quat()  # scipy's xyzw order, including older supported versions
        frame = np.r_[p + r.apply(axis * displacement), quat[3], quat[:3]]
        baseline = None
        for threads in (1, 32, 64, 128):
            run = subprocess.run([str(exe), str(threads), str(q)], capture_output=True, text=True)
            assert run.returncode == 0, run.stdout + run.stderr
            values = np.fromstring(run.stdout.strip().splitlines()[-1], sep=" ")
            assert values.size == (26 if threads >= 32 else 19)
            np.testing.assert_allclose(values[:12], expected, atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(values[12:19], frame, atol=2e-5, rtol=2e-5)
            if threads >= 32:
                np.testing.assert_allclose(values[19:26], frame, atol=2e-5, rtol=2e-5)
            if baseline is not None:
                np.testing.assert_array_equal(values[:19], baseline)
            baseline = values[:19]

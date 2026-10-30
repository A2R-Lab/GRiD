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


@pytest.mark.parametrize("mimic", [False, True, "constant"])
@pytest.mark.parametrize("axis", ["cardinal", "skew"])
def test_prismatic_dynamics_against_independent_mechanics(tmp_path, mimic, axis, monkeypatch):
    """ID/FD/ABA/M/Minv and cardinal first derivatives, not only FK.

    Pure translation gives a constant scalar mass. The follower's absolute
    displacement is q/2 + offset, so its generalized mass is .7/4 and its
    gravity contribution is .7/2. These expectations do not use either generator
    or RBDReference spatial algebra. Dense-axis derivative coverage is separate.
    """
    from RBDReference import RBDReference
    from test.cuda_equivalents.cuda_harness import _compile_runner, _parse_runner_output, _run_runner

    source = robot_urdf("slider_mimic_fixed_tool" if mimic else "slider_fixed_tool")
    urdf = tmp_path / "slider.urdf"
    urdf.write_text(source.read_text().replace("0.6 0 0.8", "0 0 1")
                    if axis == "cardinal" else source.read_text())
    if mimic == "constant":
        urdf.write_text(urdf.read_text().replace('mimic joint="slide"', 'mimic joint="tool"'))
    robot = URDFParser().parse(str(urdf))
    algos = ["inverse_dynamics", "forward_dynamics", "aba", "crba", "minv"]
    if axis == "cardinal":
        algos += ["inverse_dynamics_gradient", "forward_dynamics_gradient"]
    gen = GRiDCodeGenerator(robot, False, True, "grid")
    gen.gen_all_code(output_path=str(tmp_path / "grid.cuh"), algorithm_list=algos,
                     enable_mujoco_kernels=False)
    monkeypatch.setenv("GRID_CUDA_CACHE_DIR", str(tmp_path / "executables"))
    exe, command = _compile_runner(tmp_path, run_tokens={"RUN_" + a.upper() for a in algos},
                                   skip_gradients=axis != "cardinal", skip_eepose_gradients=True)
    direction = np.array([0., 0., 1.]) if axis == "cardinal" else np.array([.6, 0., .8])
    world_axis = Rotation.from_euler("xyz", [.3, .2, .1]).apply(direction)
    # A constant follower still inherits the ancestor's full translation.
    ratio = 1. if mimic == "constant" else .5
    mass = 1. + (.7 * ratio**2 if mimic else 0.)
    bias = (1. + (.7 * ratio if mimic else 0.)) * 9.81 * world_axis[2]
    reference = RBDReference(robot)
    for q, qd, torque in ((0., 0., 0.), (.13, -.4, 2.), (.5, .7, -3.)):
        expected = {"inverse_dynamics": bias, "forward_dynamics": (torque - bias) / mass,
                    "aba": (torque - bias) / mass, "crba": mass, "minv": 1. / mass}
        np.testing.assert_allclose(reference.crba(np.array([q])), [[mass]], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(reference.inverse_dynamics(np.array([q]), np.array([qd]), np.zeros(1))[0],
                                   [bias], atol=1e-12, rtol=1e-12)
        if axis == "cardinal":
            expected.update({name + suffix: 0. for name in
                             ("inverse_dynamics_gradient", "forward_dynamics_gradient")
                             for suffix in ("_q", "_qd")})
        baseline = None
        for threads in (1, 32, 64, 128):
            stdout = _run_runner(exe, f"{q}\n{qd}\n{torque}\n", command, num_threads=threads)
            outputs = _parse_runner_output(stdout)
            actual = []
            for name, value in expected.items():
                assert name in outputs, (name, stdout)
                result = np.asarray(outputs[name]).reshape(-1)
                assert result.size == 1, (name, result)
                np.testing.assert_allclose(result, [value], atol=2e-4, rtol=2e-4, err_msg=name)
                actual.extend(result)
            if baseline is not None:
                np.testing.assert_array_equal(actual, baseline)
            baseline = actual

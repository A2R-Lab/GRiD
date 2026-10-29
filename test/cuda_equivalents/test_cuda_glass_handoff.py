"""Independent numerical checks of embedded cooperative handoffs, not timing."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from grid_assets import robot_urdf
from grid_codegen import GRiDCodeGenerator
from URDFParser import URDFParser
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch


@pytest.mark.cuda_equivalence
@pytest.mark.parametrize("scalar,atol,rtol", [("float", 3e-5, 3e-5), ("double", 2e-11, 2e-11)])
def test_embedded_glass_restricted_handoffs(tmp_path, scalar, atol, rtol):
    nvcc = shutil.which("nvcc")
    assert nvcc, "CUDA toolkit required for this numerical gate"
    robot = URDFParser().parse(str(robot_urdf("iiwa14")))
    GRiDCodeGenerator(robot, FILE_NAMESPACE="grid").gen_all_code(
        codegen_profile="kinematics", output_path=str(tmp_path / "grid.cuh"))
    exe = tmp_path / "handoff.exe"
    command = [nvcc, "-std=c++17", "-O3", "-Xptxas=-v", f"-DTEST_SCALAR={scalar}",
               f"-arch=sm_{_detect_cuda_arch()}", f"-I{tmp_path}",
               str(Path(__file__).with_name("cuda_glass_handoff_runner.cu")), "-o", str(exe)]
    compiled = subprocess.run(command, capture_output=True, text=True)
    (tmp_path / "compile.log").write_text(" ".join(command) + "\n" + compiled.stdout + compiled.stderr)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    rng = np.random.default_rng(729)
    inputs, expected = [], []
    for _ in range(2):
        m = rng.normal(size=(7, 7))
        spd = m @ m.T + 2*np.eye(7)
        indefinite = (m + m.T)/2
        # Force pivot selection away from the original diagonal.
        pivoted = np.roll(spd, 1, axis=0)
        rhs = rng.normal(size=7)
        dtype = np.float32 if scalar == "float" else np.float64
        spd, indefinite, pivoted, rhs = [a.astype(dtype).astype(np.float64)
                                       for a in (spd, indefinite, pivoted, rhs)]
        w, v = np.linalg.eigh(indefinite)
        projected = (v * np.maximum(w, .125)) @ v.T
        inputs.extend(np.r_[spd.ravel(order="F"), indefinite.ravel(order="F"),
                            pivoted.ravel(order="F"), rhs])
        expected.extend(np.r_[np.linalg.solve(spd, rhs), projected.ravel(order="F"),
                              np.linalg.inv(pivoted).ravel(order="F"), [1234]*3])
    stdin = " ".join(format(x, ".17g") for x in inputs) + "\n"
    (tmp_path / "input.txt").write_text(stdin)
    for storage in (0, 1):
        baseline = None
        for threads in (1, 17, 32, 64, 128, 256):
            result = subprocess.run([str(exe), str(threads), str(storage)], input=stdin,
                                    capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stdout + result.stderr
            values = np.array([np.fromstring(line, sep=" ") for line in result.stdout.splitlines()])
            assert values.shape == (3, len(expected)), result.stdout
            np.testing.assert_allclose(values, np.broadcast_to(expected, values.shape), atol=atol, rtol=rtol)
            np.testing.assert_array_equal(values[0], values[1])
            np.testing.assert_array_equal(values[0], values[2])
            if baseline is not None:
                np.testing.assert_array_equal(values[0], baseline)
            baseline = values[0]

"""Full [dq|dv] momentum cost, including floating coordinate cross terms."""
import contextlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from grid_codegen import GRiDCodeGenerator
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.tests.model_sources import resolve_robot_spec
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch, _build_cuda_samples
from test.cuda_equivalents.test_cuda_integrator_equivalence import _robot_spec

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.cuda_equivalence
@pytest.mark.robot_smoke
@pytest.mark.parametrize("robot,base", [("iiwa14", "fixed"), ("go2", "floating"), ("fr3", "fixed")])
@pytest.mark.parametrize("dtype,tier", [("double", "TIER_MINIMAL"), ("float", "TIER_SHARED")])
def test_full_momentum_contract(tmp_path, robot, base, dtype, tier):
    nvcc = shutil.which("nvcc")
    if not nvcc:
        pytest.skip("nvcc is unavailable")
    spec = _robot_spec(robot, base)
    model = build_project_adapter(spec, resolve_robot_spec(spec), base_mode=base)
    ref, nv = model.reference, model.nv
    codegen = GRiDCodeGenerator(model.robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(output_path=str(tmp_path / "grid.cuh"), algorithm_list=["dccrba"])
    source = Path(__file__).with_name("cuda_momentum_contract_runner.cu")
    exe = tmp_path / "momentum.exe"
    arch = _detect_cuda_arch()
    cmd = [nvcc, "-std=c++17", "-O0", f"-arch=sm_{arch}", f"-DTEST_SCALAR={dtype}",
           f"-DTEST_TIER=grid::{tier}", f"-I{tmp_path}", f"-I{ROOT / 'external/GLASS/include'}",
           str(source), "-o", str(exe)]
    compiled = subprocess.run(cmd, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    h_des = np.array([0.2, -0.3, 0.7, -0.4, 0.6, -0.1])
    weights = np.array([0.5, 1.0, 2.0, 0.8, 1.7, 1.2])
    samples = _build_cuda_samples(model, random_count=2, include_corner_samples=False)
    for sample in samples:
        q, v = np.asarray(sample.q), np.asarray(sample.qd)
        value, grad, hess = ref.momentum_cost(q, v, h_des, weights)
        for mjx in ([False, True] if base == "floating" else [False]):
            qi, vi = q.copy(), v.copy()
            expected_g, expected_h = grad, hess
            if mjx:
                rotation = ref._rotation_from_quat_xyzw(q[3:7])
                qi[3:7] = q[[6, 3, 4, 5]]
                vi[:3] = rotation @ v[:3]
                # Full state-coordinate Jacobian, not block-diagonal G alone.
                transform = np.eye(2*nv)
                transform[:3, :3] = rotation.T
                transform[nv:nv+3, nv:nv+3] = rotation.T
                for axis in range(3):
                    transform[nv:nv+3, 3+axis] = -np.cross(np.eye(3)[axis], v[:3])
                expected_g = transform.T @ grad
                expected_h = transform.T @ hess @ transform
            data = " ".join(map(str, np.r_[qi, vi, h_des, weights])) + "\n"
            for threads in (32, 128):
                result = subprocess.run([str(exe), str(int(mjx)), str(threads)],
                                        input=data, text=True, capture_output=True)
                assert result.returncode == 0, result.stderr
                out = np.fromstring(result.stdout, sep=" ")
                assert out.size == 1 + 2*nv + 4*nv*nv, result.stdout
                tol = 3e-4 if dtype == "float" else 2e-9
                tag = f"{robot}/{base}/{dtype}/{tier}/{sample.name}/mjx={mjx}/{threads}"
                # Scale each output independently: a large Hessian must not
                # mask a missing configuration-gradient contribution.
                for actual, expected in [(out[0], value), (out[1:1+2*nv], expected_g),
                                         (out[1+2*nv:], expected_h.ravel(order="F"))]:
                    np.testing.assert_allclose(actual, expected, rtol=tol,
                                               atol=tol * max(1., np.max(np.abs(expected))),
                                               err_msg=tag)
                actual_h = out[1+2*nv:].reshape((2*nv, 2*nv), order="F")
                np.testing.assert_allclose(actual_h, actual_h.T, rtol=tol, atol=tol)

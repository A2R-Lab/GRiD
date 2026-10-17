"""Per-sample forces through all batched integrator value/gradient kernels.

Batch four with fewer blocks also catches offsetting by blockIdx instead of k.
These bounded, torque-conditioned states use the existing integrator tolerances.
"""
from pathlib import Path
import subprocess

import numpy as np
import pytest

from RBDReference.tests.model_sources import resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter
from test.cuda_equivalents.cuda_harness import _detect_cuda_arch, _parse_runner_output
from test.cuda_equivalents.executable_cache import cached_nvcc_executable
from test.cuda_equivalents.test_cuda_integrator_equivalence import (
    _robot_spec, _generate_header, _INTEGRATORS, _assert_close_scaled,
)

pytestmark = pytest.mark.cuda_equivalence
RUNNER_SOURCE = Path(__file__).with_name("cuda_integrator_batch_forces_runner.cu")


@pytest.mark.parametrize("robot,base", [("iiwa14", "fixed"), ("go2", "floating")])
def test_integrator_batch_forces(tmp_path, robot, base):
    spec = _robot_spec(robot, base)
    model = build_project_adapter(spec, resolve_robot_spec(spec), base_mode=base)
    _generate_header(model, tmp_path)
    arch = _detect_cuda_arch()
    exe, cmd = cached_nvcc_executable(
        [RUNNER_SOURCE, tmp_path / "grid.cuh"],
        ["-std=c++17", "-O0", "-gencode", f"arch=compute_{arch},code=sm_{arch}"],
        exe_name="integrator_batch_forces.exe", fallback_dir=tmp_path,
        include_dirs=[Path(__file__).resolve().parents[2] / "external/GLASS/include"],
    )
    rng = np.random.default_rng(17053)
    q = rng.uniform(-.2, .2, (4, model.nq)).astype(np.float32)
    if base == "floating":
        q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    v = rng.uniform(-.1, .1, (4, model.nv)).astype(np.float32)
    a = rng.uniform(-.2, .2, (4, model.nv)).astype(np.float32)
    u = np.stack([model.inverse_dynamics(q[k], v[k], a[k]) for k in range(4)]).astype(np.float32)
    nb = len(model.robot.get_parent_id_array())
    forces = rng.uniform(-.1, .1, (4, nb, 6)).astype(np.float32)
    dt = float(np.float32(.01))
    for pattern in ("distinct", "repeated", "zero", "null", "permuted"):
        f = forces.copy()
        order = [3, 1, 0, 2] if pattern == "permuted" else list(range(4))
        if pattern == "repeated":
            f[:] = f[0]
        elif pattern in ("zero", "null"):
            f[:] = 0
        rows = np.concatenate([q[order], v[order], u[order]], axis=1)
        stdin = " ".join(map(str, np.concatenate([rows.ravel(), [dt], f[order].ravel()])))
        expected = {}
        for _, scheme, _, _ in _INTEGRATORS:
            for k, src in enumerate(order):
                kwargs = dict(integrator_type=scheme, f_ext=None if pattern == "null" else f[src])
                expected[scheme, k] = (
                    model.integrator(q[src].astype(float), v[src].astype(float), u[src].astype(float), dt, **kwargs),
                    model.integrator_gradient(q[src].astype(float), v[src].astype(float), u[src].astype(float), dt, **kwargs),
                )
        for threads, blocks in ((1, 1), (32, 2)):
            result = subprocess.run([str(exe), str(threads), str(blocks), str(int(pattern == "null"))],
                                    input=stdin, text=True, capture_output=True, timeout=120)
            assert result.returncode == 0, (cmd, result.stdout, result.stderr)
            actual = _parse_runner_output(result.stdout)
            for (scheme, k), (x, grad) in expected.items():
                for suffix, oracle in (("value", x), ("both_value", x),
                                       ("gradient", grad), ("both_gradient", grad)):
                    got = np.asarray(actual[f"{scheme}_{suffix}_{k}"])
                    if suffix.endswith("value"):
                        got = got.reshape(-1)
                    _assert_close_scaled(got, oracle, 5e-4, 5e-4,
                        f"{robot}/{pattern}/{threads}/{blocks}/{scheme}/{suffix}/{k}")

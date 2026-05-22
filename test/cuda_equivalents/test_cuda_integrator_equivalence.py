"""CUDA equivalence test for the generated time-integrator kernels.

Validates `grid::integrator<EULER|SEMI_IMPLICIT_EULER>` and the matching
`integrator_gradient<...>` / `integrator_gradient_with_x_kp1<...>` host
wrappers against the Python reference composed in
`ProjectModelAdapter.integrator` / `integrator_gradient`. Mirrors the
world-frame IDSVA-SO smoke-runner pattern: codegen iiwa14 with the
``integrators`` profile, compile a small CUDA driver that exercises both
integrator types over each sample (q, qd, u, dt), then diff the printed
matrices block-by-block.

Default robot is iiwa14-fixed; pass GRID_CUDA_INTEGRATOR_ROBOTS to widen
the sweep. Set GRID_CUDA_INTEGRATOR_DT to override the integration
timestep (default 0.01).
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
from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _build_cuda_samples,
    _detect_cuda_arch,
    _parse_runner_output,
    _run_runner,
    _sample_to_stdin,
    _thread_counts,
)
from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from test.pinocchio_equivalents.utils.project_adapter import build_project_adapter


RUNNER_SOURCE = Path(__file__).with_name("cuda_integrator_smoke_runner.cu")

# (prefix, python-side integrator name, has_gradient)
# Fixed-base: all five integrators emit value + gradient + both-at-once kernels.
# Floating-base: all five emit the value kernel; only Euler emits the gradient
# (SI-Euler / Midpoint / RK3 / RK4 floating gradients static_assert in-kernel
# pending dIntegrate chain-rule wiring). The per-integrator emission is gated in
# the test by `_gradient_emitted`.
_INTEGRATORS = (
    ("integrator_euler",    "euler",                True),
    ("integrator_si_euler", "semi_implicit_euler",  True),
    ("integrator_midpoint", "midpoint",             True),
    ("integrator_rk3",      "rk3",                  True),
    ("integrator_rk4",      "rk4",                  True),
)


def _comma_separated_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _robot_ids() -> tuple[str, ...]:
    return _comma_separated_env("GRID_CUDA_INTEGRATOR_ROBOTS", "iiwa14,go2")


def _dts() -> tuple[float, ...]:
    """Set of dt values to exercise. Override with comma-separated env var."""
    raw = os.environ.get("GRID_CUDA_INTEGRATOR_DT", "0.001,0.01,0.1")
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def _robot_spec(robot_id: str, base_mode: str):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not in manifest")


def _samples(project_model):
    return _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)


def _generate_header(project_model, build_dir: Path) -> Path:
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        project_model.robot,
        DEBUG_MODE=False,
        NEED_PRINT_MAT=False,
        FILE_NAMESPACE="grid",
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(
            codegen_profile="integrators",
            output_path=str(header),
        )
    return header


def _compile_runner(build_dir: Path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc was not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_integrator_smoke_runner.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "GLASS" / "include"
    cmd = [
        nvcc,
        "-std=c++17",
        "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-I{glass_inc}",
        "-o", str(executable),
        str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA integrator smoke runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return executable, cmd


def _build_case(project_model, tmp_path, label):
    build_dir = tmp_path / label
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    return _compile_runner(build_dir)


def _sample_stdin_with_dt(sample, dt: float) -> str:
    base = _sample_to_stdin(sample)
    return base + f" {dt}\n"


def _run_sample(executable, compile_cmd, sample, dt: float, num_threads=None):
    stdout = _run_runner(executable, _sample_stdin_with_dt(sample, dt), compile_cmd, num_threads=num_threads)
    return _parse_runner_output(stdout)


def _assert_close_scaled(actual, expected, rtol, atol, err_msg):
    """assert_allclose with the absolute floor raised to rtol*max|expected|.

    A structurally-zero entry (e.g. a coupling term that vanishes at this
    operating point) carries float32 round-off ~ rtol*scale; comparing it with
    a fixed tiny atol trips at high velocity/dt even though the kernel is
    correct. Flooring atol at the array's overall scale lets "small relative to
    the matrix" count as close, while a genuine error stays O(scale) and fails."""
    expected_arr = np.asarray(expected, dtype=np.float64)
    scale = float(np.max(np.abs(expected_arr))) if expected_arr.size else 0.0
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=max(atol, rtol * scale), err_msg=err_msg,
    )


def _base_modes() -> tuple[str, ...]:
    return _comma_separated_env("GRID_CUDA_INTEGRATOR_BASE_MODES", "fixed,floating")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("base_mode", _base_modes())
@pytest.mark.parametrize(
    "robot_id",
    _robot_ids(),
    ids=lambda robot_id: f"{robot_id}-integrator",
)
def test_cuda_integrator_matches_python_reference(tmp_path, robot_id, base_mode):
    """CUDA integrator kernels must match the Python reference composed via FD + Minv.

    Fixed-base exercises value + gradient + both-at-once for all 5 integrators.
    Floating-base exercises value for all 5 integrators, plus the Euler gradient.
    SI-Euler / Midpoint / RK3 / RK4 floating gradients are not emitted yet
    (kernel static_assert), so only Euler is checked for floating.
    """
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./developer_install.sh "
            f"before executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    executable, compile_cmd = _build_case(project_model, tmp_path, f"{robot_id}_{base_mode}_cuda_integrator")
    samples = _samples(project_model)
    dts = _dts()
    nv = project_model.nv
    nq = project_model.nq

    rtol = 5e-4
    atol = 5e-4

    # Floating-base emits the gradient kernel for Euler only; fixed-base for all
    # five.
    def _gradient_emitted(integrator_type: str) -> bool:
        if base_mode == "fixed":
            return True
        return integrator_type == "euler"

    # Sweep block thread counts (one warp + multi-warp + a session-random count)
    # to catch thread-count-dependent races; the kernel is compiled once and the
    # thread count is passed to the runner via argv.
    for num_threads in _thread_counts():
      for dt in dts:
        for sample in samples:
            # The shared `DynamicsSample` carries q, qd, qdd — for the integrator
            # the third vector serves as the control torque u.
            actual = _run_sample(executable, compile_cmd, sample, dt, num_threads=num_threads)
            u = sample.qdd
            for prefix, integrator_type, has_gradient in _INTEGRATORS:
                expected_x_kp1 = project_model.integrator(
                    sample.q, sample.qd, u, dt, integrator_type=integrator_type,
                )
                x_kp1_block = np.asarray(actual[prefix + "_x_kp1"], dtype=np.float64).reshape(-1)
                assert x_kp1_block.shape == (nq + nv,), (
                    f"{prefix} x_kp1 shape {x_kp1_block.shape} (expected {(nq + nv,)})"
                )
                _assert_close_scaled(
                    x_kp1_block, expected_x_kp1, rtol, atol,
                    err_msg=f"{robot_id}-{base_mode} {prefix} x_kp1 @ {sample.name} dt={dt} threads={num_threads}",
                )

                if not (has_gradient and _gradient_emitted(integrator_type)):
                    continue

                expected_dAB = project_model.integrator_gradient(
                    sample.q, sample.qd, u, dt, integrator_type=integrator_type,
                )
                dAB_block = np.asarray(actual[prefix + "_dAB"], dtype=np.float64)
                x_kp1_with_block = np.asarray(actual[prefix + "_x_kp1_with_dAB"], dtype=np.float64).reshape(-1)
                dAB_with_block = np.asarray(actual[prefix + "_dAB_with_x_kp1"], dtype=np.float64)

                assert dAB_block.shape == (2 * nv, 3 * nv), (
                    f"{prefix} dAB shape {dAB_block.shape} (expected {(2*nv, 3*nv)})"
                )

                _assert_close_scaled(
                    dAB_block, expected_dAB, rtol, atol,
                    err_msg=f"{robot_id}-{base_mode} {prefix} dAB @ {sample.name} dt={dt} threads={num_threads}",
                )
                _assert_close_scaled(
                    x_kp1_with_block, expected_x_kp1, rtol, atol,
                    err_msg=f"{robot_id}-{base_mode} {prefix} x_kp1_with_dAB @ {sample.name} dt={dt} threads={num_threads}",
                )
                _assert_close_scaled(
                    dAB_with_block, expected_dAB, rtol, atol,
                    err_msg=f"{robot_id}-{base_mode} {prefix} dAB_with_x_kp1 @ {sample.name} dt={dt} threads={num_threads}",
                )

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from GRiDCodeGenerator import GRiDCodeGenerator
from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _detect_cuda_arch,
    _parse_runner_output,
    _run_runner,
    _sample_to_stdin,
)
from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from test.pinocchio_equivalents.utils.project_adapter import build_project_adapter
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples


RUNNER_SOURCE = Path(__file__).with_name("cuda_second_order_smoke_runner.cu")


@contextlib.contextmanager
def _temporary_env(updates):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _fixed_robot_spec(robot_id: str):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="fixed"):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-fixed was not found in the robot manifest.")


def _generate_second_order_header(project_model, build_dir: Path, target_shared_bytes):
    header_path = build_dir / "grid.cuh"
    env_updates = {"GRID_CUDA_TARGET_SHARED_MEM_BYTES": target_shared_bytes}
    with _temporary_env(env_updates):
        codegen = GRiDCodeGenerator(
            project_model.robot,
            DEBUG_MODE=False,
            NEED_PRINT_MAT=False,
            FILE_NAMESPACE="grid",
        )
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            codegen.gen_all_code(
                include_homogenous_transforms=True,
                codegen_profile="all",
                output_path=str(header_path),
            )
    return header_path


def _compile_second_order_runner(build_dir: Path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc was not found; install CUDA Toolkit to run CUDA tests.")

    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_second_order_smoke_runner.exe"
    cmd = [
        nvcc,
        "-std=c++11",
        "-O0",
        "-gencode",
        f"arch=compute_{arch},code=sm_{arch}",
        "-gencode",
        f"arch=compute_{arch},code=compute_{arch}",
        "-o",
        str(executable),
        str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA second-order smoke runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return executable, cmd


def _run_second_order_case(project_model, sample, tmp_path, label, target_shared_bytes):
    build_dir = tmp_path / label
    build_dir.mkdir()
    _generate_second_order_header(project_model, build_dir, target_shared_bytes)
    executable, compile_cmd = _compile_second_order_runner(build_dir)
    stdout = _run_runner(executable, _sample_to_stdin(sample), compile_cmd)
    return _parse_runner_output(stdout)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def _flatten_second_order_tensors(tensors):
    return np.concatenate([np.asarray(tensor, dtype=np.float64).reshape(-1) for tensor in tensors]).reshape(1, -1)


def test_fixed_second_order_forced_fallback_matches_python_reference(tmp_path):
    if os.environ.get("GRID_CUDA_RUN_SECOND_ORDER_FALLBACK_SMOKE") != "1":
        pytest.skip(
            "Second-order CUDA fallback smoke is quarantined while IDSVA-SO/FDSVA-SO "
            "resource pressure and thread-count assumptions are investigated. Set "
            "GRID_CUDA_RUN_SECOND_ORDER_FALLBACK_SMOKE=1 to run this diagnostic."
        )
    robot_id = os.environ.get("GRID_CUDA_SECOND_ORDER_SMOKE_ROBOT", "iiwa14")
    spec = _fixed_robot_spec(robot_id)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./developer_install.sh "
            f"before executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode="fixed")
    sample = next(item for item in build_dynamics_samples(project_model) if item.name == "zero")

    forced_fallback = _run_second_order_case(
        project_model,
        sample,
        tmp_path,
        "second_order_forced_fallback",
        target_shared_bytes=10000,
    )

    np.testing.assert_allclose(
        forced_fallback["second_order_config"][0, 2:5],
        np.asarray([1.0, 1.0, 1.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert np.all(forced_fallback["second_order_config"][0, 0:2] > 0.0)
    np.testing.assert_allclose(
        forced_fallback["idsva_so"],
        _flatten_second_order_tensors(project_model.idsva_so(sample.q, sample.qd, sample.qdd)),
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_allclose(
        forced_fallback["fdsva_so"],
        _flatten_second_order_tensors(project_model.fdsva_so(sample.q, sample.qd, sample.qdd)),
        rtol=2e-4,
        atol=2e-4,
    )

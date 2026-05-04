import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from GRiDCodeGenerator import GRiDCodeGenerator
from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from test.pinocchio_equivalents.utils.project_adapter import build_project_adapter
from test.pinocchio_equivalents.utils.state_sampling import build_dynamics_samples


RUNNER_SOURCE = Path(__file__).with_name("cuda_equivalence_runner.cu")
GPU_UNAVAILABLE_PATTERNS = (
    "no cuda-capable device",
    "cuda driver version is insufficient",
    "cuda driver version is insufficient for cuda runtime version",
    "cudaerrorinsufficientdriver",
    "cudaerrornodevice",
    "failed to initialize nvml",
    "gpu access blocked",
    "operation not permitted",
    "all cuda-capable devices are busy or unavailable",
)


def _detect_cuda_arch() -> str:
    env_arch = os.environ.get("GRID_CUDA_ARCH")
    if env_arch:
        return env_arch.replace(".", "")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                compute_cap = line.strip()
                if compute_cap:
                    return compute_cap.replace(".", "")

    return "86"


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda_equivalence")
    config.addinivalue_line("markers", "developer_only")


def _iiwa14_fixed_spec():
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="fixed"):
        spec = case["spec"]
        if spec.robot_id == "iiwa14":
            return spec
    raise RuntimeError("Could not find fixed-base iiwa14 in the robot manifest.")


def _generate_grid_header(project_model, build_dir: Path) -> Path:
    header_path = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        project_model.robot,
        DEBUG_MODE=False,
        NEED_PRINT_MAT=True,
        FILE_NAMESPACE="grid",
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header_path),
        )
    return header_path


def _compile_runner(build_dir: Path) -> tuple[Path, list[str]]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc was not found; install CUDA Toolkit to run CUDA equivalence tests.")

    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)

    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_equivalence_runner.exe"
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
            "CUDA equivalence runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return executable, cmd


def _run_runner(executable: Path, sample_input: str, compile_cmd: list[str]) -> str:
    result = subprocess.run(
        [str(executable)],
        input=sample_input,
        cwd=executable.parent,
        capture_output=True,
        text=True,
    )
    combined_output = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0:
        if any(pattern in combined_output for pattern in GPU_UNAVAILABLE_PATTERNS):
            pytest.skip(
                "CUDA runtime is unavailable in this environment. "
                "Run on a GPU-enabled machine with:\n"
                "  nvcc --version\n"
                "  nvidia-smi\n"
                "  .venv/bin/python -m pytest test/cuda_equivalents "
                "-m cuda_equivalence -vv\n"
                "If it still fails, report the pytest output plus this compile command:\n"
                f"  {' '.join(compile_cmd)}"
            )
        pytest.fail(
            "CUDA equivalence runner failed at runtime.\n"
            f"Command: {executable}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout


def _sample_to_stdin(sample) -> str:
    values = [
        np.asarray(sample.q, dtype=np.float32),
        np.asarray(sample.qd, dtype=np.float32),
        np.asarray(sample.qdd, dtype=np.float32),
    ]
    serialized_rows = (" ".join(f"{value:.9g}" for value in vec) for vec in values)
    return "\n".join(serialized_rows) + "\n"


def _parse_runner_output(stdout: str) -> dict[str, np.ndarray]:
    outputs = {}
    lines = iter(stdout.splitlines())
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("BEGIN "):
            raise AssertionError(f"Unexpected CUDA runner output line: {stripped}")
        _begin, name, rows, cols = stripped.split()
        rows = int(rows)
        cols = int(cols)
        matrix_rows = []
        for _ in range(rows):
            matrix_rows.append([float(value) for value in next(lines).split()])
        end = next(lines).strip()
        if end != f"END {name}":
            raise AssertionError(f"Expected END {name}, got: {end}")
        outputs[name] = np.asarray(matrix_rows, dtype=np.float64).reshape(rows, cols)
    return outputs


def _normalize_cuda_minv(matrix: np.ndarray) -> np.ndarray:
    normalized = matrix.copy()
    lower = np.tril(normalized, k=-1)
    upper = np.triu(normalized, k=1)
    if np.count_nonzero(np.abs(lower) > 1e-7) == 0:
        normalized = normalized + upper.T
    return normalized


def _assert_close(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    try:
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    except AssertionError as exc:
        diff = np.abs(actual - expected)
        rel = diff / np.maximum(np.abs(expected), 1e-12)
        flat_index = int(np.argmax(diff))
        index = np.unravel_index(flat_index, diff.shape)
        actual_at_index = actual[index]
        expected_at_index = expected[index]
        raise AssertionError(
            f"{name} CUDA mismatch: max_abs={diff[index]}, "
            f"max_rel={rel[index]}, first_worst_index={index}, "
            f"actual={actual_at_index}, expected={expected_at_index}"
        ) from exc


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def test_fixed_iiwa14_generated_cuda_matches_python_reference(tmp_path):
    spec = _iiwa14_fixed_spec()
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            "Could not resolve manifest iiwa14. Run ./developer_install.sh before "
            f"executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode="fixed")
    sample = build_dynamics_samples(project_model)[-1]

    build_dir = tmp_path / "cuda_iiwa14"
    build_dir.mkdir()
    _generate_grid_header(project_model, build_dir)
    executable, compile_cmd = _compile_runner(build_dir)
    stdout = _run_runner(executable, _sample_to_stdin(sample), compile_cmd)
    cuda = _parse_runner_output(stdout)

    np.testing.assert_allclose(
        cuda["input_q"],
        np.asarray(sample.q, dtype=np.float32).reshape(1, -1),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        cuda["input_qd"],
        np.asarray(sample.qd, dtype=np.float32).reshape(1, -1),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        cuda["input_u"],
        np.asarray(sample.qdd, dtype=np.float32).reshape(1, -1),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        cuda["runtime_probe"],
        np.arange(10, 10 + project_model.nv, dtype=np.float32).reshape(1, -1),
        rtol=0.0,
        atol=1e-7,
    )

    zeros = np.zeros(project_model.nv, dtype=np.float64)
    expected = {
        "inverse_dynamics": project_model.rnea(sample.q, sample.qd, zeros).reshape(
            1, -1
        ),
        "direct_minv": project_model.minv(sample.q),
        "forward_dynamics": project_model.forward_dynamics(
            sample.q, sample.qd, sample.qdd
        ).reshape(1, -1),
        "inverse_dynamics_gradient_q": project_model.rnea_grad(
            sample.q, sample.qd, zeros
        )[0],
        "inverse_dynamics_gradient_qd": project_model.rnea_grad(
            sample.q, sample.qd, zeros
        )[1],
        "forward_dynamics_gradient_q": project_model.forward_dynamics_grad(
            sample.q, sample.qd, sample.qdd
        )[0],
        "forward_dynamics_gradient_qd": project_model.forward_dynamics_grad(
            sample.q, sample.qd, sample.qdd
        )[1],
        "aba": project_model.aba(sample.q, sample.qd, sample.qdd).reshape(1, -1),
        "crba": project_model.crba(sample.q),
    }

    cuda["direct_minv"] = _normalize_cuda_minv(cuda["direct_minv"])
    failures = []
    for name, expected_value in expected.items():
        try:
            _assert_close(name, cuda[name], expected_value)
        except AssertionError as exc:
            failures.append(str(exc))
    if failures:
        pytest.fail("\n".join(failures))

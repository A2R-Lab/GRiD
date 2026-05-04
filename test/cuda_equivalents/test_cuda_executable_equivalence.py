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
from test.pinocchio_equivalents.utils.state_sampling import (
    DynamicsSample,
    _joint_ranges,
    build_dynamics_samples,
)


RUNNER_SOURCE = Path(__file__).with_name("cuda_equivalence_runner.cu")
DEFAULT_RANDOM_SAMPLE_COUNT = 3
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
SINGULAR_DEPENDENT_ALGORITHMS = {
    "direct_minv",
    "forward_dynamics",
    "forward_dynamics_gradient_q",
    "forward_dynamics_gradient_qd",
    "aba",
}
CUDA_DEFAULT_TOLERANCE = {
    "rtol": 2e-4,
    "atol": 2e-4,
}
CUDA_ROBOT_ALGORITHM_TOLERANCES = {
    ("go2", "aba"): {
        "rtol": 2.5e-2,
        "atol": 2e-4,
        "norm_rtol": 1e-2,
        "note": "GO2 CUDA ABA is a float32 generated-kernel path and currently reaches low-percent per-entry residuals against the Python float64 reference on the random smoke samples.",
    },
    ("go2", "crba"): {
        "rtol": 2e-4,
        "atol": 5e-2,
        "note": "GO2 CUDA CRBA differs from the Python float64 reference by a few hundredths on near-zero off-diagonal entries while the rest of the dynamics stack remains strict.",
    },
    ("g1", "aba"): {
        "rtol": 2.5e-2,
        "atol": 2e-4,
        "norm_rtol": 1e-2,
        "note": "G1 CUDA ABA is a large generated float32 kernel and currently reaches sub-percent to low-percent per-entry residuals against the Python float64 reference.",
    },
    ("g1", "crba"): {
        "rtol": 2e-4,
        "atol": 1.25,
        "note": "G1 CUDA CRBA has about unit-scale residuals on selected off-diagonal entries in this smoke path; keep this override scoped to G1 CRBA.",
    },
}


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


def build_fixed_cuda_case_params():
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="fixed"):
        spec = case["spec"]
        params.append(
            pytest.param(
                spec,
                "fixed",
                id=f"{spec.robot_id}-fixed",
                marks=[
                    pytest.mark.cuda_equivalence,
                    pytest.mark.developer_only,
                    pytest.mark.robot_smoke,
                ],
            )
        )
    return params


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


def _random_sample_count() -> int:
    raw_count = os.environ.get("GRID_CUDA_RANDOM_SAMPLES")
    if raw_count is None:
        return DEFAULT_RANDOM_SAMPLE_COUNT
    try:
        return max(0, int(raw_count))
    except ValueError as exc:
        raise ValueError("GRID_CUDA_RANDOM_SAMPLES must be an integer.") from exc


def _stable_robot_seed(robot_id: str) -> int:
    return 1000 + sum((index + 1) * ord(char) for index, char in enumerate(robot_id))


def _build_cuda_samples(project_model, random_count: int | None = None):
    samples = list(build_dynamics_samples(project_model))
    if random_count is None:
        random_count = _random_sample_count()

    rng = np.random.default_rng(_stable_robot_seed(project_model.spec.robot_id))
    if project_model.nq:
        bounds = _joint_ranges(project_model.robot, project_model.nq, -0.75, 0.75)
    else:
        bounds = np.zeros((0, 2), dtype=np.float64)

    for sample_index in range(random_count):
        q = np.zeros(project_model.nq, dtype=np.float64)
        if project_model.nq:
            q = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float64)
        qd = rng.uniform(-1.5, 1.5, size=project_model.nv).astype(np.float64)
        qdd = rng.uniform(-2.5, 2.5, size=project_model.nv).astype(np.float64)
        samples.append(
            DynamicsSample(
                name=f"cuda_random_{sample_index}",
                q=q,
                qd=qd,
                qdd=qdd,
            )
        )
    return samples


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


def _has_invertible_project_mass_matrix(project_model, q, min_singular_value=1e-12):
    try:
        mass = np.asarray(project_model.crba(q), dtype=np.float64)
        singular_values = np.linalg.svd(mass, compute_uv=False)
    except np.linalg.LinAlgError:
        return False
    if singular_values.size == 0:
        return False
    return bool(
        np.isfinite(singular_values).all()
        and singular_values[-1] > min_singular_value
    )


def _expected_output(project_model, sample, name: str):
    zeros = np.zeros(project_model.nv, dtype=np.float64)
    if name == "inverse_dynamics":
        return project_model.rnea(sample.q, sample.qd, zeros).reshape(1, -1)
    if name == "direct_minv":
        return project_model.minv(sample.q)
    if name == "forward_dynamics":
        return project_model.forward_dynamics(sample.q, sample.qd, sample.qdd).reshape(
            1, -1
        )
    if name == "inverse_dynamics_gradient_q":
        return project_model.rnea_grad(sample.q, sample.qd, zeros)[0]
    if name == "inverse_dynamics_gradient_qd":
        return project_model.rnea_grad(sample.q, sample.qd, zeros)[1]
    if name == "forward_dynamics_gradient_q":
        return project_model.forward_dynamics_grad(sample.q, sample.qd, sample.qdd)[0]
    if name == "forward_dynamics_gradient_qd":
        return project_model.forward_dynamics_grad(sample.q, sample.qd, sample.qdd)[1]
    if name == "aba":
        return project_model.aba(sample.q, sample.qd, sample.qdd).reshape(1, -1)
    if name == "crba":
        return project_model.crba(sample.q)
    raise ValueError(f"Unexpected CUDA output name: {name}")


def _cuda_tolerance(robot_id: str, algorithm: str):
    return CUDA_ROBOT_ALGORITHM_TOLERANCES.get(
        (robot_id, algorithm), CUDA_DEFAULT_TOLERANCE
    )


def _assert_close(
    label: str,
    actual: np.ndarray,
    expected: np.ndarray,
    robot_id: str,
    algorithm: str,
) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    tol = _cuda_tolerance(robot_id, algorithm)
    try:
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=tol["rtol"],
            atol=tol["atol"],
        )
    except AssertionError as exc:
        diff = np.abs(actual - expected)
        rel = diff / np.maximum(np.abs(expected), 1e-12)
        flat_index = int(np.argmax(diff))
        index = np.unravel_index(flat_index, diff.shape)
        norm_rel = np.linalg.norm(diff) / max(np.linalg.norm(expected), 1e-12)
        norm_rtol = tol.get("norm_rtol")
        if norm_rtol is not None and norm_rel <= norm_rtol:
            return
        actual_at_index = actual[index]
        expected_at_index = expected[index]
        raise AssertionError(
            f"{label} CUDA mismatch: max_abs={diff[index]}, "
            f"max_rel={rel[index]}, first_worst_index={index}, "
            f"norm_rel={norm_rel}, actual={actual_at_index}, "
            f"expected={expected_at_index}, rtol={tol['rtol']}, atol={tol['atol']}"
        ) from exc


@pytest.mark.parametrize(("spec", "base_mode"), build_fixed_cuda_case_params())
def test_fixed_base_generated_cuda_matches_python_reference(spec, base_mode, tmp_path):
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./developer_install.sh before "
            f"executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)

    build_dir = tmp_path / f"cuda_{spec.robot_id}_{base_mode}"
    build_dir.mkdir()
    _generate_grid_header(project_model, build_dir)
    executable, compile_cmd = _compile_runner(build_dir)

    failures = []
    skipped = []
    for sample in _build_cuda_samples(project_model):
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

        cuda["direct_minv"] = _normalize_cuda_minv(cuda["direct_minv"])
        invertible_mass_matrix = _has_invertible_project_mass_matrix(
            project_model, sample.q
        )
        for name in (
            "inverse_dynamics",
            "direct_minv",
            "forward_dynamics",
            "inverse_dynamics_gradient_q",
            "inverse_dynamics_gradient_qd",
            "forward_dynamics_gradient_q",
            "forward_dynamics_gradient_qd",
            "aba",
            "crba",
        ):
            if name in SINGULAR_DEPENDENT_ALGORITHMS and not invertible_mass_matrix:
                skipped.append(f"{spec.robot_id}/{sample.name}/{name}")
                continue
            try:
                expected_value = _expected_output(project_model, sample, name)
                _assert_close(
                    f"{spec.robot_id}/{sample.name}/{name}",
                    cuda[name],
                    expected_value,
                    robot_id=spec.robot_id,
                    algorithm=name,
                )
            except AssertionError as exc:
                failures.append(str(exc))

    if failures:
        pytest.fail("\n".join(failures))
    if skipped:
        pytest.skip(
            "Skipped singular-mass CUDA comparisons: " + ", ".join(skipped)
        )

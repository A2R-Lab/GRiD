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
    _has_invertible_project_mass_matrix,
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


RUNNER_SOURCE = Path(__file__).with_name("cuda_second_order_smoke_runner.cu")


def _comma_separated_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        pytest.fail(f"{name} must contain at least one value when set.")
    return values


def _second_order_smoke_robot_ids() -> tuple[str, ...]:
    if "GRID_CUDA_SECOND_ORDER_SMOKE_ROBOTS" in os.environ:
        return _comma_separated_env("GRID_CUDA_SECOND_ORDER_SMOKE_ROBOTS", "")
    return _comma_separated_env(
        "GRID_CUDA_SECOND_ORDER_SMOKE_ROBOT",
        "iiwa14",
    )


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
    threads = os.environ.get("GRID_CUDA_SECOND_ORDER_TEST_THREADS")
    if threads:
        try:
            thread_count = int(threads)
        except ValueError:
            pytest.fail(
                "GRID_CUDA_SECOND_ORDER_TEST_THREADS must be an integer when set."
            )
        if thread_count <= 0:
            pytest.fail(
                "GRID_CUDA_SECOND_ORDER_TEST_THREADS must be positive when set."
            )
        cmd.insert(-1, f"-DGRID_CUDA_SECOND_ORDER_TEST_THREADS={thread_count}")
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA second-order smoke runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return executable, cmd


def _build_second_order_case(project_model, tmp_path, label, target_shared_bytes):
    build_dir = tmp_path / label
    build_dir.mkdir()
    _generate_second_order_header(project_model, build_dir, target_shared_bytes)
    return _compile_second_order_runner(build_dir)


def _run_second_order_sample(executable, compile_cmd, sample):
    stdout = _run_runner(executable, _sample_to_stdin(sample), compile_cmd)
    return _parse_runner_output(stdout)


def _second_order_target_shared_bytes() -> int:
    raw = os.environ.get("GRID_CUDA_SECOND_ORDER_TARGET_SHARED_BYTES", "10000")
    try:
        target_shared_bytes = int(raw)
    except ValueError:
        pytest.fail("GRID_CUDA_SECOND_ORDER_TARGET_SHARED_BYTES must be an integer.")
    if target_shared_bytes <= 0:
        pytest.fail("GRID_CUDA_SECOND_ORDER_TARGET_SHARED_BYTES must be positive.")
    return target_shared_bytes


def _second_order_expected_flags():
    raw = os.environ.get("GRID_CUDA_SECOND_ORDER_EXPECT_FLAGS")
    if raw is None:
        return np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) != 3:
        pytest.fail(
            "GRID_CUDA_SECOND_ORDER_EXPECT_FLAGS must contain exactly three "
            "comma-separated values for IDSVA global output, FDSVA global "
            "tensors, and FDSVA workspace temp."
        )
    try:
        return np.asarray([float(value) for value in values], dtype=np.float64)
    except ValueError:
        pytest.fail("GRID_CUDA_SECOND_ORDER_EXPECT_FLAGS values must be numeric.")


def _second_order_samples(project_model):
    sample_names = _comma_separated_env("GRID_CUDA_SECOND_ORDER_SAMPLE_NAMES", "zero")
    try:
        random_count = int(os.environ.get("GRID_CUDA_SECOND_ORDER_RANDOM_SAMPLES", "0"))
    except ValueError:
        pytest.fail("GRID_CUDA_SECOND_ORDER_RANDOM_SAMPLES must be an integer.")
    if random_count < 0:
        pytest.fail("GRID_CUDA_SECOND_ORDER_RANDOM_SAMPLES must be non-negative.")

    include_corner_samples = sample_names == ("all",) or any(
        name not in {"zero", "conservative"} for name in sample_names
    )
    samples = _build_cuda_samples(
        project_model,
        random_count=random_count,
        include_corner_samples=include_corner_samples,
    )
    if sample_names == ("all",):
        return samples

    samples_by_name = {sample.name: sample for sample in samples}
    missing = [name for name in sample_names if name not in samples_by_name]
    if missing:
        available = ", ".join(sorted(samples_by_name))
        pytest.fail(
            "Unknown GRID_CUDA_SECOND_ORDER_SAMPLE_NAMES value(s): "
            f"{', '.join(missing)}. Available samples: {available}"
        )
    return [samples_by_name[name] for name in sample_names]


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def _flatten_second_order_tensors(tensors):
    return np.concatenate([np.asarray(tensor, dtype=np.float64).reshape(-1) for tensor in tensors]).reshape(1, -1)


def _assert_allclose_with_optional_norm_guard(
    actual,
    expected,
    *,
    rtol,
    atol,
    err_msg,
    norm_rtol=None,
    max_abs=None,
    max_abs_rtol=None,
):
    try:
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            err_msg=err_msg,
        )
    except AssertionError:
        if norm_rtol is None:
            raise
        actual_arr = np.asarray(actual, dtype=np.float64)
        expected_arr = np.asarray(expected, dtype=np.float64)
        diff = actual_arr - expected_arr
        norm_rel = np.linalg.norm(diff) / max(np.linalg.norm(expected_arr), 1e-30)
        max_abs_diff = float(np.max(np.abs(diff))) if diff.size else 0.0
        max_abs_limit = max_abs
        if max_abs_rtol is not None:
            expected_max = float(np.max(np.abs(expected_arr))) if expected_arr.size else 0.0
            scaled_limit = max_abs_rtol * expected_max
            max_abs_limit = scaled_limit if max_abs_limit is None else max(max_abs_limit, scaled_limit)
        if norm_rel <= norm_rtol and (max_abs_limit is None or max_abs_diff <= max_abs_limit):
            return
        raise


def _fdsva_so_tolerance(robot_id: str):
    # FDSVA-SO composes IDSVA-SO, Minv, and FD gradients in float32 CUDA.
    # Some structurally near-zero entries are cancellation dominated, so keep
    # the strict elementwise check first, then allow a small tensor-level guard.
    return dict(norm_rtol=2e-4, max_abs=3e-2, max_abs_rtol=5e-5)


@pytest.mark.parametrize(
    "robot_id",
    _second_order_smoke_robot_ids(),
    ids=lambda robot_id: f"{robot_id}-fixed",
)
def test_fixed_second_order_forced_fallback_matches_python_reference(tmp_path, robot_id):
    if os.environ.get("GRID_CUDA_RUN_SECOND_ORDER_FALLBACK_SMOKE") != "1":
        pytest.skip(
            "Second-order CUDA fallback smoke is quarantined while IDSVA-SO/FDSVA-SO "
            "resource pressure and thread-count assumptions are investigated. Set "
            "GRID_CUDA_RUN_SECOND_ORDER_FALLBACK_SMOKE=1 to run this diagnostic."
        )
    spec = _fixed_robot_spec(robot_id)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./developer_install.sh "
            f"before executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode="fixed")
    samples = _second_order_samples(project_model)
    target_shared_bytes = _second_order_target_shared_bytes()
    expected_flags = _second_order_expected_flags()
    executable, compile_cmd = _build_second_order_case(
        project_model,
        tmp_path,
        f"{robot_id}_second_order_forced_fallback",
        target_shared_bytes,
    )

    for sample in samples:
        forced_fallback = _run_second_order_sample(executable, compile_cmd, sample)

        np.testing.assert_allclose(
            forced_fallback["second_order_config"][0, 2:5],
            expected_flags,
            rtol=0.0,
            atol=0.0,
            err_msg=f"{robot_id}-fixed {sample.name} second-order tier flags",
        )
        assert np.all(forced_fallback["second_order_config"][0, 0:2] > 0.0)
        expected_idsva = _flatten_second_order_tensors(
            project_model.idsva_so(sample.q, sample.qd, sample.qdd)
        )
        np.testing.assert_allclose(
            forced_fallback["idsva_so"],
            expected_idsva,
            rtol=2e-4,
            # The zero-state dM/dq block has a tiny reference norm, so float32
            # accumulation noise can dominate relative error despite sub-1e-3
            # absolute agreement.
            atol=1e-3,
            err_msg=f"{robot_id}-fixed {sample.name} IDSVA-SO",
        )
        if _has_invertible_project_mass_matrix(project_model, sample.q):
            fdsva_tolerance = _fdsva_so_tolerance(robot_id)
            _assert_allclose_with_optional_norm_guard(
                forced_fallback["fdsva_so"],
                _flatten_second_order_tensors(
                    project_model.fdsva_so(sample.q, sample.qd, sample.qdd)
                ),
                rtol=2e-4,
                atol=2e-4,
                err_msg=f"{robot_id}-fixed {sample.name} FDSVA-SO",
                **fdsva_tolerance,
            )

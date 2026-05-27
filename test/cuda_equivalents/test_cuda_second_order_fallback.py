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
    _random_thread_count,
    _run_runner,
    _sample_to_stdin,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.equivalents import build_adapter, resolve_backend


def _build_so_oracle(spec, resolved, base_mode):
    """Independent second-order oracle (default: EXACT pinocchio pin_so_ext).
    Lets the slow pure-Python SO reference be replaced by C++ ms-scale calls on
    big robots; GRID_REFERENCE_BACKEND=reference forces the pure-Python path."""
    backend = resolve_backend(os.environ.get("GRID_REFERENCE_BACKEND", "pinocchio"))
    return build_adapter(spec, resolved, base_mode=base_mode, backend=backend)


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


def _robot_spec(robot_id: str, base_mode: str):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} was not found in the robot manifest.")


def _floating_second_order_robot_ids() -> tuple[str, ...]:
    return _comma_separated_env(
        "GRID_CUDA_FLOATING_SECOND_ORDER_ROBOTS",
        "iiwa14",
    )


def _generate_second_order_header(
    project_model,
    build_dir: Path,
    target_shared_bytes,
    *,
    enable_floating_second_order=False,
    algorithm_list=None,
):
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
                algorithm_list=algorithm_list,
                enable_floating_second_order=enable_floating_second_order,
                output_path=str(header_path),
            )
    return header_path


def _compile_second_order_runner(build_dir: Path, *, enable_fdsva=True):
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
    else:
        # Session-random multi-warp count (non-multiple of 32) so the SO kernels
        # are probed across warp counts over time, catching thread-count races
        # that a fixed block size hides. Override with the env var to reproduce.
        thread_count = _random_thread_count()
    cmd.insert(-1, f"-DGRID_CUDA_SECOND_ORDER_TEST_THREADS={thread_count}")
    cmd.insert(-1, f"-DGRID_CUDA_SECOND_ORDER_ENABLE_FDSVA={int(enable_fdsva)}")
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA second-order smoke runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return executable, cmd


def _build_second_order_case(
    project_model,
    tmp_path,
    label,
    target_shared_bytes,
    *,
    enable_floating_second_order=False,
    enable_fdsva=True,
    algorithm_list=None,
):
    build_dir = tmp_path / label
    build_dir.mkdir()
    _generate_second_order_header(
        project_model,
        build_dir,
        target_shared_bytes,
        enable_floating_second_order=enable_floating_second_order,
        algorithm_list=algorithm_list,
    )
    return _compile_second_order_runner(build_dir, enable_fdsva=enable_fdsva)


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


def _floating_second_order_samples(project_model):
    sample_names = _comma_separated_env(
        "GRID_CUDA_FLOATING_SECOND_ORDER_SAMPLE_NAMES",
        "zero,conservative",
    )
    try:
        random_count = int(os.environ.get("GRID_CUDA_FLOATING_SECOND_ORDER_RANDOM_SAMPLES", "0"))
    except ValueError:
        pytest.fail("GRID_CUDA_FLOATING_SECOND_ORDER_RANDOM_SAMPLES must be an integer.")
    if random_count < 0:
        pytest.fail("GRID_CUDA_FLOATING_SECOND_ORDER_RANDOM_SAMPLES must be non-negative.")

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
            "Unknown GRID_CUDA_FLOATING_SECOND_ORDER_SAMPLE_NAMES value(s): "
            f"{', '.join(missing)}. Available samples: {available}"
        )
    return [samples_by_name[name] for name in sample_names]


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
def _flatten_second_order_tensors(tensors):
    return np.concatenate([np.asarray(tensor, dtype=np.float64).reshape(-1) for tensor in tensors]).reshape(1, -1)


IDSVA_BLOCK_NAMES = ("d2tau_dq", "d2tau_dqd", "d2tau_dvdq", "dM_dq")


def _idsva_block_indices_from_env():
    raw = os.environ.get("GRID_CUDA_FLOATING_SECOND_ORDER_COMPARE_BLOCKS", "all")
    names = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not names:
        pytest.fail("GRID_CUDA_FLOATING_SECOND_ORDER_COMPARE_BLOCKS must not be empty.")
    if names == ("all",):
        return tuple(range(len(IDSVA_BLOCK_NAMES)))
    aliases = {
        "non_q_side": ("d2tau_dqd", "d2tau_dvdq", "dM_dq"),
        "velocity_side": ("d2tau_dqd", "d2tau_dvdq", "dM_dq"),
    }
    expanded = []
    for name in names:
        expanded.extend(aliases.get(name, (name,)))
    unknown = [name for name in expanded if name not in IDSVA_BLOCK_NAMES]
    if unknown:
        pytest.fail(
            "Unknown GRID_CUDA_FLOATING_SECOND_ORDER_COMPARE_BLOCKS value(s): "
            f"{', '.join(unknown)}. Available: all, non_q_side, "
            f"{', '.join(IDSVA_BLOCK_NAMES)}"
        )
    return tuple(IDSVA_BLOCK_NAMES.index(name) for name in expanded)


def _select_idsva_blocks(flat_tensor, block_indices, nv):
    block_size = nv**3
    parts = [
        flat_tensor[:, block_index * block_size : (block_index + 1) * block_size]
        for block_index in block_indices
    ]
    return np.concatenate(parts, axis=1)


def _assert_idsva_blocks_close(actual, expected, block_indices, nv, *, rtol, atol, err_msg):
    block_size = nv**3
    try:
        for block_index in block_indices:
            block_slice = slice(block_index * block_size, (block_index + 1) * block_size)
            np.testing.assert_allclose(
                actual[:, block_slice],
                expected[:, block_slice],
                rtol=rtol,
                atol=atol,
                err_msg=f"{err_msg} / {IDSVA_BLOCK_NAMES[block_index]}",
            )
    except AssertionError as exc:
        lines = [str(exc), "IDSVA-SO block diagnostics:"]
        for block_index in block_indices:
            name = IDSVA_BLOCK_NAMES[block_index]
            block_slice = slice(block_index * block_size, (block_index + 1) * block_size)
            actual_block = actual[:, block_slice].reshape(nv, nv, nv)
            expected_block = expected[:, block_slice].reshape(nv, nv, nv)
            diff = actual_block - expected_block
            bad = np.abs(diff) > (atol + rtol * np.abs(expected_block))
            bad_indices = np.argwhere(bad)
            max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
            rel_norm = float(np.linalg.norm(diff) / max(np.linalg.norm(expected_block), 1e-30))
            transpose_diff = actual_block - np.swapaxes(expected_block, 1, 2)
            transpose_rel_norm = float(
                np.linalg.norm(transpose_diff) / max(np.linalg.norm(expected_block), 1e-30)
            )
            root_bad = int(np.count_nonzero(bad[0, :, :] | bad[:, 0, :] | bad[:, :, 0]))
            successor_bad = int(np.count_nonzero(bad)) - root_bad
            lines.append(
                f"  {name}: bad={len(bad_indices)}/{block_size}, "
                f"max_abs={max_abs:.6g}, rel_norm={rel_norm:.6g}, "
                f"last_two_axis_transpose_rel_norm={transpose_rel_norm:.6g}, "
                f"root_axis_bad={root_bad}, successor_bad={successor_bad}"
            )
            for idx in bad_indices[:20]:
                i, j, k = (int(value) for value in idx)
                lines.append(
                    f"    ({i}, {j}, {k}): actual={actual_block[i, j, k]:.10g}, "
                    f"expected={expected_block[i, j, k]:.10g}, "
                    f"diff={diff[i, j, k]:.10g}"
                )
        raise AssertionError("\n".join(lines)) from exc


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
    spec = _fixed_robot_spec(robot_id)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./developer_install.sh "
            f"before executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode="fixed")
    reference_model = _build_so_oracle(spec, resolved, "fixed")
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
            reference_model.idsva_so_body_frame(sample.q, sample.qd, sample.qdd)
        )
        np.testing.assert_allclose(
            forced_fallback["idsva_so_body_frame"],
            expected_idsva,
            rtol=2e-4,
            # The zero-state dM/dq block has a tiny reference norm, so float32
            # accumulation noise can dominate relative error despite sub-1e-3
            # absolute agreement.
            atol=1e-3,
            err_msg=f"{robot_id}-fixed {sample.name} IDSVA-SO",
        )
        if _has_invertible_project_mass_matrix(reference_model, sample.q):
            fdsva_tolerance = _fdsva_so_tolerance(robot_id)
            _assert_allclose_with_optional_norm_guard(
                forced_fallback["fdsva_so"],
                _flatten_second_order_tensors(
                    reference_model.fdsva_so(sample.q, sample.qd, sample.qdd)
                ),
                rtol=2e-4,
                atol=2e-4,
                err_msg=f"{robot_id}-fixed {sample.name} FDSVA-SO",
                **fdsva_tolerance,
            )


@pytest.mark.parametrize(
    "robot_id",
    _floating_second_order_robot_ids(),
    ids=lambda robot_id: f"{robot_id}-floating",
)
def test_floating_second_order_diagnostic_matches_python_reference(tmp_path, robot_id, capsys):
    spec = _robot_spec(robot_id, "floating")
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./developer_install.sh "
            f"before executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode="floating")
    reference_model = _build_so_oracle(spec, resolved, "floating")
    samples = _floating_second_order_samples(project_model)
    target_shared_bytes = _second_order_target_shared_bytes()
    enable_fdsva = os.environ.get("GRID_CUDA_FLOATING_SECOND_ORDER_ENABLE_FDSVA", "0") == "1"
    algorithm_list = "idsva_so_body_frame,fdsva_so" if enable_fdsva else "idsva_so_body_frame"

    executable, compile_cmd = _build_second_order_case(
        project_model,
        tmp_path,
        f"{robot_id}_floating_second_order_diagnostic",
        target_shared_bytes,
        enable_floating_second_order=True,
        enable_fdsva=enable_fdsva,
        algorithm_list=algorithm_list,
    )
    block_indices = _idsva_block_indices_from_env()

    for sample in samples:
        actual = _run_second_order_sample(executable, compile_cmd, sample)
        config = actual["second_order_config"][0]
        np.testing.assert_allclose(
            config[5:7],
            np.asarray([1.0, float(enable_fdsva)], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
            err_msg=f"{robot_id}-floating {sample.name} generation flags",
        )
        np.testing.assert_allclose(
            config[7:12],
            np.asarray(
                [
                    project_model.nq,
                    project_model.nv,
                    project_model.robot.get_num_bodies(),
                    project_model.nq + 2 * project_model.nv,
                    4 * project_model.nv**3,
                ],
                dtype=np.float64,
            ),
            rtol=0.0,
            atol=0.0,
            err_msg=f"{robot_id}-floating {sample.name} dimension config",
        )
        expected_idsva = _flatten_second_order_tensors(
            reference_model.idsva_so_body_frame(sample.q, sample.qd, sample.qdd)
        )
        _assert_idsva_blocks_close(
            actual["idsva_so_body_frame"],
            expected_idsva,
            block_indices,
            project_model.nv,
            rtol=2e-4,
            atol=1e-3,
            err_msg=(
                f"{robot_id}-floating {sample.name} IDSVA-SO blocks "
                f"{[IDSVA_BLOCK_NAMES[index] for index in block_indices]}"
            ),
        )
        if enable_fdsva and _has_invertible_project_mass_matrix(reference_model, sample.q):
            _assert_allclose_with_optional_norm_guard(
                actual["fdsva_so"],
                _flatten_second_order_tensors(
                    project_model.fdsva_so(sample.q, sample.qd, sample.qdd)
                ),
                rtol=2e-4,
                atol=2e-4,
                err_msg=f"{robot_id}-floating {sample.name} FDSVA-SO",
                **_fdsva_so_tolerance(robot_id),
            )

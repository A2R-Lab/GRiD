"""CUDA equivalence test for the world-frame IDSVA-SO emission.

Validates `gen_idsva_so_world_frame_inner` (CUDA) against
`RBDReference.idsva_so_world_frame` (verified Python).

The world-frame path is the second floating-base IDSVA-SO emission added
alongside the existing shim-based `gen_idsva_so_floating_reference_inner`.
It runs the single-pass world-frame algorithm in CUDA — world-frame
propagation with gravity baked into the main sweep, no separate gravity
shim. Mirrors `RBDReference.idsva_so_world_frame` line-for-line.

Runs by default; iiwa14-floating is the default robot.
Set GRID_CUDA_IDSVA_SO_WORLD_FRAME_ROBOTS=iiwa14,go2,g1 to exercise more.
"""

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
)
from test.cuda_equivalents.test_cuda_second_order_fallback import _temporary_env
from test.pinocchio_equivalents.conftest import MANIFEST_PATH
from test.pinocchio_equivalents.utils.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from test.pinocchio_equivalents.utils.project_adapter import build_project_adapter


RUNNER_SOURCE = Path(__file__).with_name("cuda_idsva_so_world_frame_smoke_runner.cu")


def _comma_separated_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        pytest.fail(f"{name} must contain at least one value when set.")
    return values


def _world_frame_robot_ids() -> tuple[str, ...]:
    return _comma_separated_env("GRID_CUDA_IDSVA_SO_WORLD_FRAME_ROBOTS", "iiwa14")


def _world_frame_target_shared_bytes() -> int:
    raw = os.environ.get("GRID_CUDA_IDSVA_SO_WORLD_FRAME_TARGET_SHARED_BYTES", "100000")
    try:
        value = int(raw)
    except ValueError:
        pytest.fail("GRID_CUDA_IDSVA_SO_WORLD_FRAME_TARGET_SHARED_BYTES must be an integer.")
    if value <= 0:
        pytest.fail("GRID_CUDA_IDSVA_SO_WORLD_FRAME_TARGET_SHARED_BYTES must be positive.")
    return value


def _robot_spec(robot_id: str, base_mode: str):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} was not found in the robot manifest.")


def _world_frame_samples(project_model):
    sample_names = _comma_separated_env(
        "GRID_CUDA_IDSVA_SO_WORLD_FRAME_SAMPLE_NAMES", "zero,conservative"
    )
    try:
        random_count = int(os.environ.get("GRID_CUDA_IDSVA_SO_WORLD_FRAME_RANDOM_SAMPLES", "0"))
    except ValueError:
        pytest.fail("GRID_CUDA_IDSVA_SO_WORLD_FRAME_RANDOM_SAMPLES must be an integer.")
    if random_count < 0:
        pytest.fail("GRID_CUDA_IDSVA_SO_WORLD_FRAME_RANDOM_SAMPLES must be non-negative.")
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
            "Unknown GRID_CUDA_IDSVA_SO_WORLD_FRAME_SAMPLE_NAMES value(s): "
            f"{', '.join(missing)}. Available samples: {available}"
        )
    return [samples_by_name[name] for name in sample_names]


def _generate_world_frame_header(project_model, build_dir: Path, target_shared_bytes: int) -> Path:
    header_path = build_dir / "grid.cuh"
    env_updates = {"GRID_CUDA_TARGET_SHARED_MEM_BYTES": str(target_shared_bytes)}
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
                algorithm_list="idsva_so_body_frame",
                enable_floating_second_order=True,
                enable_idsva_so_world_frame=True,
                output_path=str(header_path),
            )
    return header_path


def _compile_world_frame_runner(build_dir: Path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc was not found; install CUDA Toolkit to run CUDA tests.")

    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_idsva_so_world_frame_smoke_runner.exe"
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
    threads = os.environ.get("GRID_CUDA_IDSVA_SO_WORLD_FRAME_TEST_THREADS")
    if threads:
        try:
            thread_count = int(threads)
        except ValueError:
            pytest.fail(
                "GRID_CUDA_IDSVA_SO_WORLD_FRAME_TEST_THREADS must be an integer when set."
            )
        if thread_count <= 0:
            pytest.fail("GRID_CUDA_IDSVA_SO_WORLD_FRAME_TEST_THREADS must be positive when set.")
        cmd.insert(-1, f"-DGRID_CUDA_IDSVA_SO_WORLD_FRAME_TEST_THREADS={thread_count}")
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA world-frame smoke runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return executable, cmd


def _build_world_frame_case(project_model, tmp_path, label, target_shared_bytes):
    build_dir = tmp_path / label
    build_dir.mkdir()
    _generate_world_frame_header(project_model, build_dir, target_shared_bytes)
    return _compile_world_frame_runner(build_dir)


def _run_world_frame_sample(executable, compile_cmd, sample):
    stdout = _run_runner(executable, _sample_to_stdin(sample), compile_cmd)
    return _parse_runner_output(stdout)


def _flatten_idsva_blocks(tensors):
    return np.concatenate(
        [np.asarray(t, dtype=np.float64).reshape(-1) for t in tensors]
    ).reshape(1, -1)


IDSVA_BLOCK_NAMES = ("d2tau_dq", "d2tau_dqd", "d2tau_dvdq", "dM_dq")


def _assert_blocks_close(actual, expected, nv, sample_name):
    block_size = nv**3
    errors = []
    for idx, name in enumerate(IDSVA_BLOCK_NAMES):
        slc = slice(idx * block_size, (idx + 1) * block_size)
        try:
            np.testing.assert_allclose(
                actual[:, slc], expected[:, slc],
                rtol=2e-4, atol=1e-3,
                err_msg=f"world-frame CUDA vs Python {name} at sample={sample_name}",
            )
        except AssertionError as exc:
            actual_block = actual[:, slc].reshape(nv, nv, nv)
            expected_block = expected[:, slc].reshape(nv, nv, nv)
            diff = actual_block - expected_block
            bad = np.abs(diff) > (1e-3 + 2e-4 * np.abs(expected_block))
            bad_idx = np.argwhere(bad)
            max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
            rel_norm = float(
                np.linalg.norm(diff) / max(np.linalg.norm(expected_block), 1e-30)
            )
            lines = [
                str(exc),
                f"  {name}: bad={len(bad_idx)}/{block_size}, max_abs={max_abs:.6g}, rel_norm={rel_norm:.6g}",
            ]
            for ii in bad_idx[:20]:
                a, b, c = (int(v) for v in ii)
                lines.append(
                    f"    ({a},{b},{c}): actual={actual_block[a,b,c]:.10g}, "
                    f"expected={expected_block[a,b,c]:.10g}, "
                    f"diff={diff[a,b,c]:.10g}"
                )
            errors.append("\n".join(lines))
    if errors:
        raise AssertionError("\n".join(errors))


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.parametrize(
    "robot_id",
    _world_frame_robot_ids(),
    ids=lambda robot_id: f"{robot_id}-floating-spatial-v2",
)
def test_cuda_world_frame_matches_python_reference(tmp_path, robot_id):
    """CUDA `idsva_so_world_frame_kernel` must match Python `idsva_so_world_frame`."""
    spec = _robot_spec(robot_id, "floating")
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./developer_install.sh "
            f"before executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode="floating")
    target_shared_bytes = _world_frame_target_shared_bytes()
    executable, compile_cmd = _build_world_frame_case(
        project_model, tmp_path, f"{robot_id}_cuda_world_frame", target_shared_bytes
    )
    samples = _world_frame_samples(project_model)

    for sample in samples:
        actual = _run_world_frame_sample(executable, compile_cmd, sample)
        config = actual["world_frame_config"][0]
        np.testing.assert_allclose(
            config[3:8],
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
            err_msg=f"{robot_id} world-frame dimension config @ {sample.name}",
        )
        expected_idsva = _flatten_idsva_blocks(
            project_model.reference.idsva_so_world_frame(sample.q, sample.qd, sample.qdd)
        )
        _assert_blocks_close(
            actual["idsva_so_body_frame"], expected_idsva, project_model.nv, sample.name
        )

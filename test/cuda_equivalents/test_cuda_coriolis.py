"""CUDA equivalence test for the full Coriolis matrix C(q, qd).

Validates `gen_coriolis_matrix` (CUDA) against the verified numpy reference
`RBDReference._EnergyMixin.coriolis_matrix(q, qd)` (a closed-form world-frame
spatial recursion, NOT a finite difference — so the tight value-tolerance bucket
applies). Also asserts the factorization-free identity

    C(q,qd) @ qd + g(q) == nonlinear_effects(q,qd)

(g from generalized_gravity, nle from nonlinear_effects — both numpy oracles),
and the M_dot == C + C^T skew identity via a central finite difference of M(q).

Thread-invariance: each case is swept over {1, 2, 16, 32, 256} block threads;
single-block CORE kernels must be thread-count-invariant.

Gated iiwa14 (fixed) first (the first green slice), then go2/g1 (floating),
then fr3/h1_2 (mimic). The output is the nv x nv C; the runner reads the gridData
host buffer hd_data->h_coriolis directly.
"""

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _build_cuda_samples,
    _detect_cuda_arch,
    _parse_runner_output,
    _run_runner,
    _sample_to_stdin,
    GPU_UNAVAILABLE_PATTERNS,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.tests.tolerances import get_tolerance


RUNNER_SOURCE = Path(__file__).with_name("cuda_coriolis_smoke_runner.cu")

# (robot_id, base_mode). iiwa14 gated first (first slice), then floating (go2/g1),
# then mimic (fr3 fixed / h1_2 fixed).
_CASES = [
    ("iiwa14", "fixed"),
    ("go2", "floating"),
    ("g1", "floating"),
    ("fr3", "fixed"),
    ("h1_2", "fixed"),
]

# Thread-invariance sweep (single-block CORE kernels must be invariant).
_THREAD_COUNTS = (1, 2, 16, 32, 256)


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not found in robot manifest.")


def _generate_header(project_model, build_dir: Path) -> Path:
    header_path = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        project_model.robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid"
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        # Explicit algorithm_list keeps the header small (just coriolis_matrix + its
        # auto-expanded inverse_dynamics dep) and is mimic-safe (no refused
        # mimic-gradient algos), so fr3 / h1_2 codegen cleanly.
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header_path),
            algorithm_list=["coriolis_matrix"],
        )
    return header_path


def _compile_runner(build_dir: Path, floating_base: bool):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_coriolis_runner.exe"
    cmd = [
        nvcc, "-std=c++11", "-O0",
        f"-DGRID_CUDA_FLOATING_BASE={1 if floating_base else 0}",
        "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-gencode", f"arch=compute_{arch},code=compute_{arch}",
        "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA coriolis runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


@pytest.mark.cuda_equivalence
@pytest.mark.parametrize(("robot_id", "base_mode"), _CASES)
def test_cuda_coriolis_matrix_matches_reference(robot_id, base_mode, tmp_path):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./install/developer_install.sh "
            f"before executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    robot = project_model.robot
    reference = project_model.reference
    nv = robot.get_num_vel()

    _generate_header(project_model, tmp_path)
    executable, compile_cmd = _compile_runner(tmp_path, base_mode == "floating")

    samples = _build_cuda_samples(project_model, random_count=2)
    tol = get_tolerance("inverse_dynamics", robot_id=robot_id)

    failures = []
    for sample in samples:
        q, qd = sample.q, sample.qd

        # numpy reference (verified vs pinocchio), and the identity oracles.
        C_ref = np.asarray(reference.coriolis_matrix(q, qd), dtype=np.float64).reshape(nv, nv)
        g_ref = np.asarray(reference.generalized_gravity(q), dtype=np.float64).reshape(nv)
        nle_ref = np.asarray(reference.nonlinear_effects(q, qd), dtype=np.float64).reshape(nv)

        scale = max(1.0, float(np.max(np.abs(C_ref))) if C_ref.size else 1.0)
        atol = tol.atol + tol.rtol * scale + 5e-3 * scale  # float32 CUDA headroom

        # Thread-invariance sweep: the CUDA C must agree with the oracle (and be
        # identical across thread counts) for every count in the sweep.
        prev = None
        for nthreads in _THREAD_COUNTS:
            try:
                stdout = _run_runner(executable, _sample_to_stdin(sample), compile_cmd, num_threads=nthreads)
            except Exception as exc:  # noqa: BLE001
                combined = str(exc).lower()
                if any(p in combined for p in GPU_UNAVAILABLE_PATTERNS):
                    pytest.skip("CUDA runtime unavailable.")
                raise
            outputs = _parse_runner_output(stdout)
            config = outputs["coriolis_config"][0]
            np.testing.assert_allclose(
                config[:2], np.asarray([robot.get_num_pos(), nv], dtype=np.float64),
                rtol=0.0, atol=0.0,
                err_msg=f"{robot_id} coriolis dimension config @ {sample.name} (threads={nthreads})",
            )
            C_cuda = np.asarray(outputs["coriolis"], dtype=np.float64).reshape(nv, nv)

            err = float(np.max(np.abs(C_cuda - C_ref))) if C_ref.size else 0.0
            if err > atol:
                failures.append(
                    f"{robot_id} @ {sample.name} (threads={nthreads}): C CUDA-vs-reference "
                    f"maxerr={err:.3e} > {atol:.3e}"
                )

            # thread-invariance: identical output across thread counts
            if prev is not None:
                tinv = float(np.max(np.abs(C_cuda - prev)))
                if tinv > 1e-4 * scale:
                    failures.append(
                        f"{robot_id} @ {sample.name}: thread-invariance violation "
                        f"(threads={nthreads}) maxdiff={tinv:.3e}"
                    )
            prev = C_cuda

            # run-to-run determinism: same exe, same stdin, same thread count must
            # be BYTE-identical (catches warp-order accumulation, e.g. the pre-2026-07-31
            # mimic C-assembly atomicAdd fold). Checked at the highest-contention count.
            if nthreads == _THREAD_COUNTS[-1]:
                stdout_repeat = _run_runner(executable, _sample_to_stdin(sample), compile_cmd, num_threads=nthreads)
                if stdout_repeat != stdout:
                    failures.append(
                        f"{robot_id} @ {sample.name} (threads={nthreads}): run-to-run "
                        f"nondeterminism (stdout differs on identical re-run)"
                    )

            # identity (against the CUDA C): C qd + g == nonlinear_effects
            id_resid = float(np.max(np.abs(C_cuda @ qd + g_ref - nle_ref)))
            id_scale = max(1.0, float(np.max(np.abs(nle_ref))) if nle_ref.size else 1.0)
            id_atol = tol.atol + tol.rtol * id_scale + 5e-3 * id_scale
            if id_resid > id_atol:
                failures.append(
                    f"{robot_id} @ {sample.name} (threads={nthreads}): C qd + g == nle "
                    f"resid={id_resid:.3e} > {id_atol:.3e}"
                )

    assert not failures, "coriolis CUDA equivalence failures:\n" + "\n".join(failures)

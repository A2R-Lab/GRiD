"""CUDA equivalence test for the analytic dCCRBA surfaces.

Validates `gen_dccrba` / `gen_cmm_time_variation` (CUDA, run in DOUBLE) against the
verified numpy oracle:
  dccrba             -> RBDReference._CentroidalMixin.dccrba(q)            (6 x NV x NV)
  cmm_time_variation -> RBDReference._CentroidalMixin.cmm_time_variation(q,qd)  (6 x NV)

Both oracles are ANALYTIC (closed-form world sweep + spatial operators, validated
vs pin.dccrba / computeCentroidalDynamicsDerivatives ~1e-12), so the TIGHT value
bucket applies (fp64 ~1e-9 relative floor on big robots), NOT the FD bucket.

Cross-check (CUDA-side): sum_m d_dccrba[:,:,m] * qd[m] == d_cmm_time_variation.

Thread-invariance: each case swept over {1, 2, 16, 32, 256} threads; the float64
oracle (not a serial-fp32 accumulation) is the invariance reference (atomicAdd
reassociation is benign — compare against the oracle, not bit-identity).

Robots: iiwa14 (fixed, first slice), go2/g1 (floating). Mimic (fr3/h1_2) is GATED
OFF in codegen (centroidal Jacobian not yet mimic-reduced) so it is not listed.
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
    GPU_UNAVAILABLE_PATTERNS,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter


RUNNER_SOURCE = Path(__file__).with_name("cuda_dccrba_smoke_runner.cu")

# (robot_id, base_mode). iiwa14 first (first green slice), then floating go2/g1.
# Mimic robots are codegen-gated-off (see module docstring) -> not tested here.
_CASES = [
    ("iiwa14", "fixed"),
    ("go2", "floating"),
    ("g1", "floating"),
]

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
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header_path),
            algorithm_list=["dccrba", "cmm_time_variation"],
        )
    return header_path


def _compile_runner(build_dir: Path, floating_base: bool):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_dccrba_runner.exe"
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
            "CUDA dccrba runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


def _dccrba_flat_to_tensor(flat, nv):
    """CUDA layout dA[row + 6*k + 6*nv*m] -> tensor[row, k, m] (6 x nv x nv)."""
    t = np.empty((6, nv, nv), dtype=np.float64)
    for m in range(nv):
        for k in range(nv):
            for row in range(6):
                t[row, k, m] = flat[row + 6 * k + 6 * nv * m]
    return t


@pytest.mark.cuda_equivalence
@pytest.mark.parametrize(("robot_id", "base_mode"), _CASES)
def test_cuda_dccrba_matches_reference(robot_id, base_mode, tmp_path):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./developer_install.sh "
            f"before executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    robot = project_model.robot
    reference = project_model.reference
    nv = robot.get_num_vel()

    _generate_header(project_model, tmp_path)
    executable, compile_cmd = _compile_runner(tmp_path, base_mode == "floating")

    samples = _build_cuda_samples(project_model, random_count=2)

    failures = []
    for sample in samples:
        q, qd = sample.q, sample.qd

        dA_ref = np.asarray(reference.dccrba(q), dtype=np.float64).reshape(6, nv, nv)
        adot_ref = np.asarray(reference.cmm_time_variation(q, qd), dtype=np.float64).reshape(6, nv)

        dA_scale = max(1.0, float(np.max(np.abs(dA_ref))) if dA_ref.size else 1.0)
        adot_scale = max(1.0, float(np.max(np.abs(adot_ref))) if adot_ref.size else 1.0)
        # tight value bucket (fp64): absolute floor + relative on magnitude.
        dA_atol = 1e-8 + 1e-7 * dA_scale
        adot_atol = 1e-8 + 1e-7 * adot_scale

        prev_dA = None
        prev_adot = None
        for nthreads in _THREAD_COUNTS:
            try:
                stdout = _run_runner(executable, _sample_to_stdin(sample), compile_cmd, num_threads=nthreads)
            except Exception as exc:  # noqa: BLE001
                combined = str(exc).lower()
                if any(p in combined for p in GPU_UNAVAILABLE_PATTERNS):
                    pytest.skip("CUDA runtime unavailable.")
                raise
            outputs = _parse_runner_output(stdout)
            config = outputs["dccrba_config"][0]
            np.testing.assert_allclose(
                config[:2], np.asarray([robot.get_num_pos(), nv], dtype=np.float64),
                rtol=0.0, atol=0.0,
                err_msg=f"{robot_id} dccrba dimension config @ {sample.name} (threads={nthreads})",
            )
            dA_flat = np.asarray(outputs["dccrba"], dtype=np.float64).reshape(-1)
            dA_cuda = _dccrba_flat_to_tensor(dA_flat, nv)
            # CUDA adot is column-major s_adot[row + 6*k] -> reshape (6, nv) Fortran-order.
            adot_cuda = np.asarray(outputs["cmm_time_variation"], dtype=np.float64).reshape(6, nv, order="F")

            err_dA = float(np.max(np.abs(dA_cuda - dA_ref))) if dA_ref.size else 0.0
            if err_dA > dA_atol:
                failures.append(
                    f"{robot_id} @ {sample.name} (threads={nthreads}): dccrba CUDA-vs-ref "
                    f"maxerr={err_dA:.3e} > {dA_atol:.3e}"
                )
            err_adot = float(np.max(np.abs(adot_cuda - adot_ref))) if adot_ref.size else 0.0
            if err_adot > adot_atol:
                failures.append(
                    f"{robot_id} @ {sample.name} (threads={nthreads}): cmm_time_variation CUDA-vs-ref "
                    f"maxerr={err_adot:.3e} > {adot_atol:.3e}"
                )

            # CUDA-side contraction cross-check: sum_m dA[:,:,m] qd[m] == Adot
            adot_from_tensor = np.einsum("abm,m->ab", dA_cuda, qd)
            err_xc = float(np.max(np.abs(adot_from_tensor - adot_cuda)))
            if err_xc > adot_atol:
                failures.append(
                    f"{robot_id} @ {sample.name} (threads={nthreads}): contraction cross-check "
                    f"sum_m dA[:,:,m] qd[m] != Adot, maxerr={err_xc:.3e} > {adot_atol:.3e}"
                )

            # thread-invariance vs the oracle-anchored previous run (loose, fp64).
            if prev_dA is not None:
                t_dA = float(np.max(np.abs(dA_cuda - prev_dA)))
                if t_dA > 1e-6 * dA_scale:
                    failures.append(
                        f"{robot_id} @ {sample.name}: dccrba thread-invariance violation "
                        f"(threads={nthreads}) maxdiff={t_dA:.3e}"
                    )
                t_adot = float(np.max(np.abs(adot_cuda - prev_adot)))
                if t_adot > 1e-6 * adot_scale:
                    failures.append(
                        f"{robot_id} @ {sample.name}: cmm thread-invariance violation "
                        f"(threads={nthreads}) maxdiff={t_adot:.3e}"
                    )
            prev_dA = dA_cuda
            prev_adot = adot_cuda

    assert not failures, "dccrba CUDA equivalence failures:\n" + "\n".join(failures)

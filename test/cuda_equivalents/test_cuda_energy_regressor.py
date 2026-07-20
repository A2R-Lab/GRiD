"""CUDA equivalence test for the energy regressors (kinetic + potential).

Validates `gen_kinetic_energy_regressor` / `gen_potential_energy_regressor` (CUDA)
against the verified numpy references
`RBDReference.kinetic_energy_regressor(q, qd)` /
`RBDReference.potential_energy_regressor(q, GRAVITY=-9.81)` and the structural
identities `y_KE @ pi == kinetic_energy` and `y_PE @ pi == potential_energy`.

Both regressors are length 10*NUM_BODIES row vectors (per-body 10 standard
inertial params pi_i = [m, m*c(3), I_O(6)=[Ixx,Ixy,Ixz,Iyy,Iyz,Izz]]). KE reuses
the RNEA forward sweep (per-link spatial velocity v_i); PE reuses the ee_pose
world-transform machinery (link-origin world R_i, p_i).

Gated iiwa14 (fixed) first, then fr3 (mimic), then g1 (floating). The outputs are
NOT a DoF sweep (no nv dimension); the runner reads the gridData host buffers
hd_data->h_ke_regressor / h_pe_regressor directly.
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
    _random_thread_count,
    _run_runner,
    _sample_to_stdin,
    GPU_UNAVAILABLE_PATTERNS,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.tests.tolerances import get_tolerance


RUNNER_SOURCE = Path(__file__).with_name("cuda_energy_regressor_smoke_runner.cu")

# (robot_id, base_mode). iiwa14 gated first (first slice), then fr3 (mimic) +
# g1 (floating) for both regressors.
_CASES = [("iiwa14", "fixed"), ("fr3", "fixed"), ("g1", "floating")]


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not found in robot manifest.")


def _project_pi(robot):
    """Stack each body's 10 standard inertial params, GRiD/URDF basis
    [m, h(3), Ixx, Ixy, Ixz, Iyy, Iyz, Izz]. Mirrors test_cuda_regressor."""
    nb = robot.get_num_bodies()
    pi = np.zeros(10 * nb, dtype=np.float64)
    for b in range(nb):
        Ib = np.asarray(robot.get_Imat_by_id(b), dtype=np.float64)
        m = Ib[5, 5]
        mc_skew = Ib[:3, 3:6]
        h = np.array([mc_skew[2, 1], mc_skew[0, 2], mc_skew[1, 0]], dtype=np.float64)
        I_O = Ib[:3, :3]
        pi[10 * b:10 * b + 10] = [
            m, h[0], h[1], h[2],
            I_O[0, 0], I_O[0, 1], I_O[0, 2], I_O[1, 1], I_O[1, 2], I_O[2, 2],
        ]
    return pi


def _generate_header(project_model, build_dir: Path) -> Path:
    header_path = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        project_model.robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid"
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        # "regressor" profile is mimic-safe (no refused mimic-gradient algos) and
        # pulls in both energy regressors + their deps (id RNEA sweep, ee_pose
        # world-transform machinery), so fr3 (mimic) codegens cleanly.
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header_path),
            codegen_profile="regressor",
        )
    return header_path


def _compile_runner(build_dir: Path, floating_base: bool):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_energy_regressor_runner.exe"
    thread_count = _random_thread_count()
    cmd = [
        nvcc, "-std=c++11", "-O0",
        f"-DGRID_CUDA_FLOATING_BASE={1 if floating_base else 0}",
        "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
        f"-DGRID_CUDA_REGRESSOR_TEST_THREADS={thread_count}",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-gencode", f"arch=compute_{arch},code=compute_{arch}",
        "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA energy-regressor runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


@pytest.mark.cuda_equivalence
@pytest.mark.parametrize(("robot_id", "base_mode"), _CASES)
def test_cuda_energy_regressor_matches_reference(robot_id, base_mode, tmp_path):
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
    nb = robot.get_num_bodies()

    _generate_header(project_model, tmp_path)
    executable, compile_cmd = _compile_runner(tmp_path, base_mode == "floating")
    pi = _project_pi(robot)

    samples = _build_cuda_samples(project_model, random_count=2)

    failures = []
    for sample in samples:
        q, qd = sample.q, sample.qd
        try:
            stdout = _run_runner(executable, _sample_to_stdin(sample), compile_cmd)
        except Exception as exc:  # noqa: BLE001
            combined = str(exc).lower()
            if any(p in combined for p in GPU_UNAVAILABLE_PATTERNS):
                pytest.skip("CUDA runtime unavailable.")
            raise
        outputs = _parse_runner_output(stdout)
        config = outputs["regressor_config"][0]
        np.testing.assert_allclose(
            config[:3],
            np.asarray([robot.get_num_pos(), nv, nb], dtype=np.float64),
            rtol=0.0, atol=0.0,
            err_msg=f"{robot_id} energy regressor dimension config @ {sample.name}",
        )

        y_ke_cuda = np.asarray(outputs["ke_regressor"], dtype=np.float64).reshape(10 * nb)
        y_pe_cuda = np.asarray(outputs["pe_regressor"], dtype=np.float64).reshape(10 * nb)

        # numpy references (verified vs pinocchio); shared -9.81 gravity convention.
        y_ke_ref = np.asarray(reference.kinetic_energy_regressor(q, qd), dtype=np.float64)
        y_pe_ref = np.asarray(reference.potential_energy_regressor(q, GRAVITY=-9.81), dtype=np.float64)
        assert y_ke_cuda.shape == y_ke_ref.shape, (
            f"{robot_id}: CUDA y_KE shape {y_ke_cuda.shape} != ref {y_ke_ref.shape}"
        )

        # tight tolerance bucket (no FD / no loose tol).
        tol = get_tolerance("inverse_dynamics", robot_id=robot_id)

        def _check(name, cuda, ref):
            scale = max(1.0, float(np.max(np.abs(ref))) if ref.size else 1.0)
            atol = tol.atol + tol.rtol * scale + 5e-3 * scale  # float32 CUDA headroom
            err = float(np.max(np.abs(cuda - ref))) if ref.size else 0.0
            if err > atol:
                failures.append(
                    f"{robot_id} @ {sample.name}: {name} CUDA-vs-reference maxerr={err:.3e} > {atol:.3e}"
                )

        _check("y_KE", y_ke_cuda, y_ke_ref)
        _check("y_PE", y_pe_cuda, y_pe_ref)

        # structural identities (exact): y_KE @ pi == kinetic_energy, y_PE @ pi == potential_energy
        ke = float(project_model.kinetic_energy(q, qd))
        pe = float(project_model.potential_energy(q))
        for name, val, target in (("y_KE@pi", float(y_ke_cuda @ pi), ke),
                                  ("y_PE@pi", float(y_pe_cuda @ pi), pe)):
            tscale = max(1.0, abs(target))
            atol_id = tol.atol + tol.rtol * tscale + 5e-3 * tscale
            err_id = abs(val - target)
            if err_id > atol_id:
                failures.append(
                    f"{robot_id} @ {sample.name}: {name}-vs-energy maxerr={err_id:.3e} > {atol_id:.3e}"
                )

    assert not failures, "energy-regressor CUDA equivalence failures:\n" + "\n".join(failures)

"""CUDA equivalence test for the joint-torque regressor emission (E1).

Validates `gen_inverse_dynamics_regressor` (CUDA) against
`RBDReference.inverse_dynamics_regressor` (the verified numpy reference,
`_RegressorMixin`) and the structural identity `Y @ pi == inverse_dynamics(q,qd,qdd)`.

`tau = Y(q,qd,qdd) . pi` with `pi_i = [m, m*c(3), I_O(6)=[Ixx,Ixy,Ixz,Iyy,Iyz,Izz]]`
per link (GRiD/URDF basis). The CUDA kernel reuses the RNEA forward sweep to get
per-link (v, a), builds each link's 6x10 body regressor, and back-propagates the
6x10 blocks up the tree (X^T + S^T projection) exactly like the numpy reference.

Gated iiwa14 (fixed) first, then g1 (floating). The regressor output is
nv x 10*NUM_BODIES and is NOT a gridData field, so the runner allocates the
output buffer itself.
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


RUNNER_SOURCE = Path(__file__).with_name("cuda_regressor_smoke_runner.cu")

# (robot_id, base_mode). iiwa14 gated first, then fr3 (mimic), then one floating
# robot. fr3 is a MIMIC robot — validates the mimic-aware regressor emit
# (alpha*s_sign projection + NUM_BODIES sizing) against the mimic numpy oracle.
_CASES = [("iiwa14", "fixed"), ("fr3", "fixed"), ("g1", "floating")]


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not found in robot manifest.")


def _project_pi(robot):
    """Stack each body's 10 standard inertial params, GRiD/URDF basis
    [m, h(3), Ixx, Ixy, Ixz, Iyy, Iyz, Izz]. Mirrors the RBDReference test
    `_project_pi`."""
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
        # Use the focused "regressor" profile ({id, regressor}) rather than the
        # default "all": this suite only compares the regressor, and "all" would
        # pull in the whole gradient/SO surface for nothing but compile time.
        # (Historically "all" was also refused for mimic fr3 by the old G0 guard;
        # mimic gradients are fully supported now, so the narrow profile is purely
        # a speed choice.) The regressor + its RNEA forward dep ("id") are here.
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
    executable = build_dir / "cuda_regressor_runner.exe"
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
            "CUDA regressor runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


@pytest.mark.cuda_equivalence
@pytest.mark.parametrize(("robot_id", "base_mode"), _CASES)
def test_cuda_regressor_matches_reference(robot_id, base_mode, tmp_path):
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
        q, qd, qdd = sample.q, sample.qd, sample.qdd
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
            err_msg=f"{robot_id} regressor dimension config @ {sample.name}",
        )

        Y_cuda = np.asarray(outputs["regressor"], dtype=np.float64).reshape(nv, 10 * nb)

        # numpy reference (verified _RegressorMixin); GRiD and the reference now
        # share one gravity convention (-9.81), so the runner passes -9.81 too.
        Y_ref = np.asarray(
            reference.inverse_dynamics_regressor(q, qd, qdd, GRAVITY=-9.81), dtype=np.float64
        )
        assert Y_cuda.shape == Y_ref.shape, (
            f"{robot_id}: CUDA Y shape {Y_cuda.shape} != ref {Y_ref.shape}"
        )

        # structural identity Y @ pi == inverse_dynamics(q,qd,qdd)
        tau = np.asarray(reference.inverse_dynamics(q, qd, qdd, GRAVITY=-9.81)[0], dtype=np.float64)

        tol = get_tolerance("inverse_dynamics", robot_id=robot_id)
        scale = max(1.0, float(np.max(np.abs(Y_ref))) if Y_ref.size else 1.0)
        atol = tol.atol + tol.rtol * scale + 5e-3 * scale  # float32 CUDA headroom

        err_Y = float(np.max(np.abs(Y_cuda - Y_ref))) if Y_ref.size else 0.0
        if err_Y > atol:
            failures.append(
                f"{robot_id} @ {sample.name}: CUDA-vs-reference Y maxerr={err_Y:.3e} > {atol:.3e}"
            )
        err_id = float(np.max(np.abs(Y_cuda @ pi - tau)))
        tau_scale = max(1.0, float(np.max(np.abs(tau))))
        atol_id = tol.atol + tol.rtol * tau_scale + 5e-3 * tau_scale
        if err_id > atol_id:
            failures.append(
                f"{robot_id} @ {sample.name}: Y@pi-vs-inverse_dynamics maxerr={err_id:.3e} > {atol_id:.3e}"
            )

    assert not failures, "regressor CUDA equivalence failures:\n" + "\n".join(failures)

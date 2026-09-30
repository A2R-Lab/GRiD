"""CUDA equivalence test for the joint-torque-regressor STATE derivative (dY/dx, B.0).

Validates `gen_inverse_dynamics_regressor_gradient` (CUDA) against
`RBDReference.inverse_dynamics_regressor_gradient` (the verified numpy oracle,
itself pinned against pinocchio's analytic RNEA gradient via the pi-identity)
and the structural B.0 identity `dY_dx[c] @ pi == dtau_dx[:, c]` against the
oracle's own `inverse_dynamics_gradient`.

CUDA layout (matches the oracle's (nv, nv, 10NB) C-order per partial):
    h_dY_dx[(dq? 0 : nv*nv*10NB) + c*(nv*10NB) + row*10NB + 10*i + k]

Covering set: iiwa14 (fixed cardinal, sparse id_du staging band), fr3 (mimic ->
dense staging fold), go2 (floating -> dense-per-jid staging band).
"""

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import (
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

from test.cuda_equivalents.test_cuda_regressor import _project_pi, _robot_spec


RUNNER_SOURCE = Path(__file__).with_name("cuda_regressor_gradient_runner.cu")

# (robot_id, base_mode): sparse fixed band, mimic dense fold, floating band.
_CASES = [("iiwa14", "fixed"), ("fr3", "fixed"), ("go2", "floating")]


def _generate_header(project_model, build_dir: Path) -> Path:
    header_path = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        project_model.robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid"
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        # Focused subset: the new algo + its transitive deps (id + id_du) only —
        # "all" would pull the whole SO surface for nothing but compile time.
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header_path),
            algorithm_list=["inverse_dynamics_regressor_gradient"],
        )
    return header_path


def _compile_runner(build_dir: Path, floating_base: bool):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_regressor_gradient_runner.exe"
    thread_count = _random_thread_count()
    cmd = [
        nvcc, "-std=c++11", "-O0",
        f"-DGRID_CUDA_FLOATING_BASE={1 if floating_base else 0}",
        "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
        f"-DGRID_CUDA_REGRESSOR_GRADIENT_TEST_THREADS={thread_count}",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-gencode", f"arch=compute_{arch},code=compute_{arch}",
        "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA regressor-gradient runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


@pytest.mark.cuda_equivalence
@pytest.mark.parametrize(("robot_id", "base_mode"), _CASES)
def test_cuda_regressor_gradient_matches_reference(robot_id, base_mode, tmp_path):
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
    W = 10 * nb

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
        config = outputs["regressor_gradient_config"][0]
        np.testing.assert_allclose(
            config[:3],
            np.asarray([robot.get_num_pos(), nv, nb], dtype=np.float64),
            rtol=0.0, atol=0.0,
            err_msg=f"{robot_id} dY/dx dimension config @ {sample.name}",
        )

        flat = np.asarray(outputs["regressor_gradient"], dtype=np.float64).reshape(2 * nv * nv, W)
        dY_dq_cuda = flat[: nv * nv].reshape(nv, nv, W)
        dY_dqd_cuda = flat[nv * nv:].reshape(nv, nv, W)

        dY_dq_ref, dY_dqd_ref = reference.inverse_dynamics_regressor_gradient(
            q, qd, qdd, GRAVITY=-9.81
        )
        dY_dq_ref = np.asarray(dY_dq_ref, dtype=np.float64)
        dY_dqd_ref = np.asarray(dY_dqd_ref, dtype=np.float64)
        assert dY_dq_cuda.shape == dY_dq_ref.shape, (
            f"{robot_id}: CUDA dY/dq shape {dY_dq_cuda.shape} != ref {dY_dq_ref.shape}"
        )

        tol = get_tolerance("inverse_dynamics_gradient", robot_id=robot_id)
        scale = max(1.0, float(np.max(np.abs(dY_dq_ref))), float(np.max(np.abs(dY_dqd_ref))))
        atol = tol.atol + tol.rtol * scale + 5e-3 * scale  # float32 CUDA headroom

        for name, cuda_t, ref_t in (
            ("dY/dq", dY_dq_cuda, dY_dq_ref),
            ("dY/dqd", dY_dqd_cuda, dY_dqd_ref),
        ):
            err = float(np.max(np.abs(cuda_t - ref_t))) if ref_t.size else 0.0
            if err > atol:
                failures.append(
                    f"{robot_id} @ {sample.name}: CUDA-vs-reference {name} maxerr={err:.3e} > {atol:.3e}"
                )

        # B.0 structural identity vs the oracle's analytic RNEA gradient.
        # public output is the hstacked (nv, 2*nv) [dc_dq | dc_dqd] matrix
        grad = np.asarray(reference.inverse_dynamics_gradient(q, qd, qdd, GRAVITY=-9.81),
                          dtype=np.float64)
        dtau_dq, dtau_dqd = grad[:, :nv], grad[:, nv:]
        proj_q = np.stack([dY_dq_cuda[c] @ pi for c in range(nv)], axis=1)
        proj_qd = np.stack([dY_dqd_cuda[c] @ pi for c in range(nv)], axis=1)
        tau_scale = max(1.0, float(np.max(np.abs(dtau_dq))), float(np.max(np.abs(dtau_dqd))))
        atol_id = tol.atol + tol.rtol * tau_scale + 5e-3 * tau_scale
        for name, proj, ref_g in (("dq", proj_q, dtau_dq), ("dqd", proj_qd, dtau_dqd)):
            err = float(np.max(np.abs(proj - ref_g)))
            if err > atol_id:
                failures.append(
                    f"{robot_id} @ {sample.name}: dY_{name}@pi-vs-id_du maxerr={err:.3e} > {atol_id:.3e}"
                )

    assert not failures, "dY/dx CUDA equivalence failures:\n" + "\n".join(failures)

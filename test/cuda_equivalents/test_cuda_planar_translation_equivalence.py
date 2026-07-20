"""End-to-end CUDA verification of PLANAR + TRANSLATION joints (Phase-6 STAGE 3).

Planar / translation joints are DECOMPOSED at parse time into a chain of
cardinal 1-DOF sub-joints + zero-mass dummy links, so the generated CUDA sees
only ordinary cardinal joints (Tier A). This test codegens the decomposed
fixtures, drives the shared ``cuda_equivalence_runner.cu``, and asserts the
CUDA ``inverse_dynamics`` (gravity+coriolis, qdd=0) and ``crba`` (M) outputs
match the verified RBDReference numpy reference -- which is in turn validated
against pinocchio's native JointModelPlanar / JointModelTranslation in
``URDFParser/tests/test_planar_translation_decomposition.py``.

This closes the CUDA gap for the decomposition: it proves the dummy-link chain
emits and runs correctly through the full GRiD kernel pipeline.
"""
import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from URDFParser import URDFParser
from RBDReference import RBDReference
from RBDReference.equivalents.reference_backend import ProjectModelAdapter
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _detect_cuda_arch,
    _parse_runner_output,
    GPU_UNAVAILABLE_PATTERNS,
)

RUNNER_SOURCE = Path(__file__).with_name("cuda_equivalence_runner.cu")
FIXDIR = Path(__file__).resolve().parents[2] / "external" / "URDFParser" / "tests" / "fixtures"


def _parse(name):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return URDFParser().parse(str(FIXDIR / name), floating_base=False)


def _generate_header(robot, build_dir):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid"
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header),
            algorithm_list="inverse_dynamics,minv,forward_dynamics,aba,crba,end_effector_pose",
        )
    return header


def _compile_runner(build_dir):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_planar_translation_runner.exe"
    cmd = [
        nvcc, "-std=c++11", "-O0",
        "-DGRID_CUDA_FLOATING_BASE=0",
        "-DGRID_RUNNER_SKIP_GRADIENTS=1",
        "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-gencode", f"arch=compute_{arch},code=compute_{arch}",
        "-o", str(exe), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA planar/translation runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return exe


def _run(exe, q, qd, u):
    def row(v):
        return " ".join(f"{x:.9g}" for x in np.asarray(v, dtype=np.float32))
    stdin = "\n".join([row(q), row(qd), row(u)]) + "\n"
    result = subprocess.run(
        [str(exe)], input=stdin, cwd=exe.parent, capture_output=True, text=True
    )
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0:
        if any(p in combined for p in GPU_UNAVAILABLE_PATTERNS):
            pytest.skip("CUDA runtime unavailable.")
        pytest.fail(f"runner failed:\n{result.stdout}\n{result.stderr}")
    return _parse_runner_output(result.stdout)


def _check(failures, label, cuda, ref, atol=2e-4, rtol=2e-4):
    cuda = np.asarray(cuda, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    if cuda.shape != ref.shape:
        failures.append(f"{label}: shape {cuda.shape} != {ref.shape}")
        return
    if not np.allclose(cuda, ref, atol=atol, rtol=rtol):
        failures.append(
            f"{label}: max|d|={np.max(np.abs(cuda - ref)):.3e}\n  cuda={cuda}\n  ref ={ref}")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("fixture", ["planar_arm.urdf", "translation_arm.urdf"])
def test_cuda_decomposed_joint_matches_reference(tmp_path, fixture):
    """CUDA id / crba for a decomposed planar/translation robot must match the
    RBDReference numpy reference (itself pin-validated). Proves the parse-time
    dummy-link chain emits + runs correctly end-to-end."""
    robot = _parse(fixture)
    assert robot is not None
    ref = RBDReference(robot)
    nv = robot.get_num_vel()

    exe = _compile_runner(tmp_path) if _generate_header(robot, tmp_path) else None

    rng = np.random.default_rng(3)
    zeros = np.zeros(nv, dtype=np.float64)
    failures = []
    for trial in range(4):
        q = rng.uniform(-0.5, 0.5, nv)
        qd = rng.uniform(-0.8, 0.8, nv)
        out = _run(exe, q, qd, zeros)
        tag = f"{fixture} trial {trial}"

        cuda_id = np.asarray(out["inverse_dynamics"], dtype=np.float64).reshape(-1)
        ref_id = np.asarray(ref.inverse_dynamics(q, qd, zeros)[0], dtype=np.float64).reshape(-1)
        _check(failures, f"{tag} inverse_dynamics", cuda_id, ref_id)

        cuda_m = np.asarray(out["crba"], dtype=np.float64).reshape(nv, nv, order="F")
        ref_m = np.asarray(ref.crba(q), dtype=np.float64)
        _check(failures, f"{tag} crba", cuda_m, ref_m)

        # end_effector_pose: exercises the runtime-FK hom-transform emit through
        # the decomposed prismatic chain (the bare-theta prismatic-translation
        # substitution the codegen fix corrected). One 6-vector pose per leaf EE.
        cuda_ee = np.asarray(out["end_effector_pose"], dtype=np.float64).reshape(-1)
        ref_ee = _ee_pose_reference(robot, q)
        _check(failures, f"{tag} end_effector_pose", cuda_ee, ref_ee)

    assert not failures, "decomposed-joint CUDA equivalence failures:\n" + "\n".join(failures)


def _ee_pose_reference(robot, q):
    """Concatenated 6-vector [xyz; rpy] pose over every leaf EE, matching the
    runner's end_effector_pose output layout 6*NUM_EES."""
    adapter = ProjectModelAdapter(
        spec=None, base_mode="fixed", robot=robot,
        reference=RBDReference(robot), parse_output="", mismatches=[])
    poses = []
    for jid in robot.get_leaf_nodes():
        target = robot.get_joint_by_id(jid).get_name()
        poses.append(np.asarray(adapter.end_effector_pose(q, target), dtype=np.float64).reshape(-1))
    return np.concatenate(poses)

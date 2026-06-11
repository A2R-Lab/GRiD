"""End-to-end CUDA verification of the SPHERICAL (ball) joint KINEMATICS suite
(Phase-6 Tier-C): end_effector_pose + frame_jacobian (geometric Jacobian).

A spherical joint is a 3-DoF manifold joint: NV=3 (body-frame angular velocity),
NQ=4 (unit quaternion xyzw, pinocchio JointModelSpherical convention), so NQ!=NV.
The forward-kinematics chain-up consumes the joint's HOMOGENEOUS quaternion
transform (which keeps R, not the spatial R^T), built via the shared quaternion
XmatsHom substitution on the joint's own 4-wide q-block. A mid-chain spherical
joint shifts every downstream joint's q-offset (nq>nv); the §1e q-slot fix routes
each downstream revolute joint's sin/cos to its own q-slot.

This test codegens ``end_effector_pose`` + ``frame_jacobian`` for the spherical
fixtures, drives ``cuda_spherical_kinematics_runner.cu``, and asserts the CUDA
result matches the verified RBDReference numpy oracle (itself validated against
pinocchio). It exercises BOTH CUDA surfaces:
  * the device functions ``end_effector_pose_device`` / ``frame_jacobian_device``
    (explicit nq-wide s_q buffer), at thread counts {1, 32, 256} for invariance; and
  * the HOST batch wrappers ``end_effector_pose<T,false>`` / ``frame_jacobian<T>``
    over a 4-timestep trajectory (the §1e nq-stride check).

Fixtures: spherical_arm (root spherical + revolute) and mixed_spherical_arm
(revolute -> spherical -> revolute, the mid-chain shift case).

EE-pose comparison: position rows (xyz) compared tightly; rpy rows looser (and a
near-gimbal |pitch|->pi/2 configuration is rejected at sample time, since rpy is
ill-conditioned there). A benign rest-ish q with a non-degenerate quaternion is
used; an fp64 spot-check runs alongside the fp32 default.
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
from GRiDCodeGenerator import GRiDCodeGenerator
from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _detect_cuda_arch,
    _parse_runner_output,
    GPU_UNAVAILABLE_PATTERNS,
)

RUNNER_SOURCE = Path(__file__).with_name("cuda_spherical_kinematics_runner.cu")
FIXDIR = Path(__file__).resolve().parents[2] / "URDFParser" / "tests" / "fixtures"

# Each fixture's quaternion-block start index in q (the spherical joint's first
# q slot). spherical_arm: ball at jid0 -> q[0:4]. mixed_spherical_arm: revolute
# then ball at jid1 -> q[1:5].
QUAT_START = {"spherical_arm.urdf": 0, "mixed_spherical_arm.urdf": 1}


def _parse(name):
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        return URDFParser().parse(str(FIXDIR / name), floating_base=False)


def _generate_header(robot, build_dir):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid"
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        # end_effector_pose + frame_jacobian are the ported kinematics algorithms
        # for spherical (Tier-C); frame_jacobian additionally pulls in
        # end_effector_pose + minv as deps (handled by the codegen normalizer).
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header),
            algorithm_list=["end_effector_pose", "frame_jacobian"],
        )
    return header


def _compile_runner(build_dir, gridT="float"):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_spherical_kinematics_runner.exe"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-DGRID_CUDA_FLOATING_BASE=0",
        "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-gencode", f"arch=compute_{arch},code=compute_{arch}",
        "-o", str(exe), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA spherical kinematics runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return exe


def _run(exe, q, target_jid, threads=32, gridT="float"):
    def row(v):
        return " ".join(f"{x:.9g}" for x in np.asarray(v, dtype=np.float64))
    stdin = "\n".join([row(q), str(int(target_jid))]) + "\n"
    env = dict(os.environ)
    env["GRID_EQUIV_T"] = gridT
    result = subprocess.run(
        [str(exe), str(threads)], input=stdin, cwd=exe.parent,
        capture_output=True, text=True, env=env,
    )
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0:
        if any(p in combined for p in GPU_UNAVAILABLE_PATTERNS):
            pytest.skip("CUDA runtime unavailable.")
        pytest.fail(f"runner failed:\n{result.stdout}\n{result.stderr}")
    return _parse_runner_output(result.stdout)


def _benign_q(robot, fixture):
    """A rest-ish configuration with a non-degenerate unit quaternion in the
    spherical q-block (chosen to avoid the rpy gimbal-lock |pitch|->pi/2)."""
    nq = robot.get_num_pos()
    q = np.full(nq, 0.2, dtype=np.float64)
    qs = QUAT_START[fixture]
    # A generic, clearly non-identity quaternion (xyzw) -> normalized.
    quat = np.array([0.1, 0.3, -0.2, 0.9], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    q[qs:qs + 4] = quat
    return q


def _oracle_ee_pose(ref, leaf_name, q):
    poses = ref.end_effector_pose(q, ee_joint_names=[leaf_name])
    return np.asarray(poses[0], dtype=np.float64).reshape(-1)  # [xyz; rpy]


def _oracle_frame_jacobian(ref, leaf_name, q):
    return np.asarray(
        ref.frame_jacobian(q, frame_name=leaf_name,
                           reference_frame="LOCAL_WORLD_ALIGNED"),
        dtype=np.float64,
    )


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
@pytest.mark.parametrize("gridT", ["float", "double"])
def test_cuda_spherical_kinematics_matches_reference(tmp_path, fixture, gridT):
    """CUDA spherical end_effector_pose + frame_jacobian must match the
    RBDReference numpy oracle on both the device function and the host batch
    wrapper, and be batch self-consistent (§1e)."""
    robot = _parse(fixture)
    assert robot is not None
    ref = RBDReference(robot)
    nv = robot.get_num_vel()
    leaf = robot.get_leaf_nodes()[0]
    leaf_name = robot.get_joint_by_id(leaf).name

    exe = _compile_runner(tmp_path, gridT=gridT) if _generate_header(robot, tmp_path) else None

    q = _benign_q(robot, fixture)
    out = _run(exe, q, leaf, gridT=gridT)

    # fp32 tols are loose-ish; fp64 should be tight. rpy is looser than xyz.
    if gridT == "double":
        pos_atol, rpy_atol, jac_atol = 1e-9, 1e-9, 1e-9
    else:
        pos_atol, rpy_atol, jac_atol = 1e-4, 1e-3, 1e-4

    failures = []

    # --- end_effector_pose ([xyz; rpy]) ---
    ref_ee = _oracle_ee_pose(ref, leaf_name, q)
    cuda_ee = np.asarray(out["end_effector_pose"], dtype=np.float64).reshape(-1)
    if cuda_ee.shape != ref_ee.shape:
        failures.append(f"ee device shape {cuda_ee.shape} != {ref_ee.shape}")
    else:
        # guard near gimbal: |pitch| -> pi/2 makes rpy ill-conditioned.
        pitch = ref_ee[4]
        near_gimbal = abs(abs(pitch) - np.pi / 2) < 1e-2
        if not np.allclose(cuda_ee[:3], ref_ee[:3], atol=pos_atol, rtol=pos_atol):
            failures.append(
                f"ee device xyz: max|d|={np.max(np.abs(cuda_ee[:3] - ref_ee[:3])):.3e}\n"
                f"  cuda={cuda_ee[:3]}\n  ref ={ref_ee[:3]}")
        if not near_gimbal and not np.allclose(cuda_ee[3:], ref_ee[3:], atol=rpy_atol, rtol=rpy_atol):
            failures.append(
                f"ee device rpy: max|d|={np.max(np.abs(cuda_ee[3:] - ref_ee[3:])):.3e}\n"
                f"  cuda={cuda_ee[3:]}\n  ref ={ref_ee[3:]}")
        # host batch rows: every timestep must equal the device single-call.
        for k in range(4):
            row = np.asarray(out[f"end_effector_pose_batch_{k}"], dtype=np.float64).reshape(-1)
            if not np.allclose(row, cuda_ee, atol=1e-5, rtol=1e-5):
                failures.append(
                    f"ee batch[{k}] vs device: max|d|={np.max(np.abs(row - cuda_ee)):.3e}")

    # --- frame_jacobian (6 x NV, column-major [linear; angular]) ---
    ref_J = _oracle_frame_jacobian(ref, leaf_name, q)
    assert ref_J.shape == (6, nv), f"oracle J shape {ref_J.shape} != {(6, nv)}"
    cuda_J = np.asarray(out["frame_jacobian"], dtype=np.float64)
    if cuda_J.size != 6 * nv:
        failures.append(f"frame_jacobian device size {cuda_J.size} != {6*nv}")
    else:
        cuda_J = cuda_J.reshape(6, nv, order="F")
        if not np.allclose(cuda_J, ref_J, atol=jac_atol, rtol=jac_atol):
            failures.append(
                f"frame_jacobian device: max|d|={np.max(np.abs(cuda_J - ref_J)):.3e}\n"
                f"  cuda=\n{cuda_J}\n  ref =\n{ref_J}")
        for k in range(4):
            blk = np.asarray(out[f"frame_jacobian_batch_{k}"], dtype=np.float64).reshape(6, nv, order="F")
            if not np.allclose(blk, cuda_J, atol=1e-5, rtol=1e-5):
                failures.append(
                    f"frame_jacobian batch[{k}] vs device: max|d|={np.max(np.abs(blk - cuda_J)):.3e}")

    assert not failures, "spherical kinematics CUDA equivalence failures:\n" + "\n".join(failures)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_kinematics_thread_invariant(tmp_path, fixture):
    """The single-block spherical end_effector_pose / frame_jacobian kernels must
    be exactly thread-count invariant (bit-identical at 1/32/256)."""
    robot = _parse(fixture)
    assert robot is not None
    leaf = robot.get_leaf_nodes()[0]
    exe = _compile_runner(tmp_path) if _generate_header(robot, tmp_path) else None

    q = _benign_q(robot, fixture)
    base_out = _run(exe, q, leaf, threads=32)
    base_ee = np.asarray(base_out["end_effector_pose"], dtype=np.float64)
    base_J = np.asarray(base_out["frame_jacobian"], dtype=np.float64)
    failures = []
    cells = [
        ("end_effector_pose", base_ee, "end_effector_pose_batch_"),
        ("frame_jacobian", base_J, "frame_jacobian_batch_"),
    ]
    for threads in (1, 32, 256):
        out = _run(exe, q, leaf, threads=threads)
        for key, baseline, batch_prefix in cells:
            dev = np.asarray(out[key], dtype=np.float64)
            if not np.array_equal(dev, baseline):
                failures.append(
                    f"{fixture} threads={threads}: {key} device differs from threads=32 "
                    f"(max|d|={np.max(np.abs(dev - baseline)):.3e})")
            for k in range(4):
                row = np.asarray(out[f"{batch_prefix}{k}"], dtype=np.float64)
                if not np.array_equal(row, dev):
                    failures.append(
                        f"{fixture} threads={threads}: {key} batch[{k}] != device single-call")

    assert not failures, "spherical kinematics thread-invariance failures:\n" + "\n".join(failures)

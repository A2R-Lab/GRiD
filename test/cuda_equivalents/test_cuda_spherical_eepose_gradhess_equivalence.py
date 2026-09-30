"""End-to-end CUDA verification of the SPHERICAL (ball) joint EE-pose
DERIVATIVES (Tier-C): end_effector_pose_gradient + end_effector_pose_hessian.

Convention (matches RBDReference + pinocchio, and the fixed/floating robots):
outputs are TANGENT-space derivatives d/dv and d^2/dv^2 — a spherical joint
contributes THREE columns at its ``get_joint_index_v`` slots (body-frame omega
ordering, local/right tangent q ⊗ exp(½·omega·dt)), NOT four quaternion-
component columns. Column k's world axis is R_world(joint) @ e_k (R_world
INCLUDES the joint's own quaternion rotation) with lever arm p_ee - p_joint;
the intra-ball second-derivative block uses the SYMMETRIZED SO(3) curvature
½([e_k]x [e_l]x + [e_l]x [e_k]x). Output shapes/layout are identical to every
other robot: gradient 6 x nv per ee COLUMN-major (idx = row + 6*vi); hessian
(6, nv, nv) C-order per ee (idx = c*nv*nv + vi*nv + vj). So for
spherical_arm (ball + revolute Z): nv = 4 -> 6x4 gradient / (6,4,4) hessian;
mixed_spherical_arm (rev Z + ball + rev X): nv = 5 -> 6x5 / (6,5,5).

The oracle side (RBDReference end_effector_pose_gradient /
end_effector_pose_hessian_analytic) is itself validated against pinocchio
getJointJacobian AND central differences ON THE QUATERNION MANIFOLD in
external/RBDReference/tests/test_spherical_ee_pose_derivatives_equivalence.py.

Surfaces exercised (mirrors test_cuda_spherical_kinematics_equivalence.py):
  * end_effector_pose_gradient_device / end_effector_pose_hessian_device
    (explicit nq-wide s_q buffer), at thread counts {1, 32, 256} (bitwise
    thread-count invariance);
  * the HOST batch wrappers end_effector_pose_gradient<T> /
    end_effector_pose_hessian<T> over a 4-timestep trajectory (nq-stride path);
  * the hessian device's co-computed gradient must equal the gradient device's.
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
from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import (
    _detect_cuda_arch,
    _parse_runner_output,
    GPU_UNAVAILABLE_PATTERNS,
)

RUNNER_SOURCE = Path(__file__).with_name("cuda_spherical_eepose_grad_hess_runner.cu")
FIXDIR = Path(__file__).resolve().parents[2] / "external" / "URDFParser" / "tests" / "fixtures"

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
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header),
            algorithm_list=["end_effector_pose", "end_effector_pose_gradient",
                            "end_effector_pose_hessian"],
        )
    return header


def _compile_runner(build_dir, gridT="float"):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_spherical_eepose_grad_hess_runner.exe"
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
            "CUDA spherical ee grad/hess runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return exe


def _run(exe, q, threads=32, gridT="float"):
    def row(v):
        return " ".join(f"{x:.9g}" for x in np.asarray(v, dtype=np.float64))
    stdin = row(q) + "\n"
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
    quat = np.array([0.1, 0.3, -0.2, 0.9], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    q[qs:qs + 4] = quat
    return q


def _oracle_gradient(ref, leaf_name, q):
    grads = ref.end_effector_pose_gradient(q, ee_joint_names=[leaf_name])
    return np.asarray(grads[0], dtype=np.float64)          # (6, nv)


def _oracle_hessian(ref, leaf_name, q):
    hess = ref.end_effector_pose_hessian_analytic(q, ee_joint_names=[leaf_name])
    return np.asarray(hess[0], dtype=np.float64)           # (6, nv, nv)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
@pytest.mark.parametrize("gridT", ["float", "double"])
def test_cuda_spherical_eepose_gradhess_matches_reference(tmp_path, fixture, gridT):
    """CUDA spherical end_effector_pose_gradient/_hessian must match the
    RBDReference tangent-space oracle on the device functions AND the host
    batch wrappers, and be batch self-consistent."""
    robot = _parse(fixture)
    assert robot is not None
    ref = RBDReference(robot)
    nv = robot.get_num_vel()
    leaf = robot.get_leaf_nodes()[0]
    leaf_name = robot.get_joint_by_id(leaf).name

    exe = _compile_runner(tmp_path, gridT=gridT) if _generate_header(robot, tmp_path) else None

    q = _benign_q(robot, fixture)
    out = _run(exe, q, gridT=gridT)

    if gridT == "double":
        grad_atol, hess_atol = 1e-9, 1e-8
    else:
        grad_atol, hess_atol = 1e-4, 1e-3

    failures = []

    # --- gradient (6 x nv, column-major) ---
    ref_J = _oracle_gradient(ref, leaf_name, q)
    assert ref_J.shape == (6, nv)
    cuda_J = np.asarray(out["end_effector_pose_gradient"], dtype=np.float64)
    if cuda_J.size != 6 * nv:
        failures.append(f"gradient device size {cuda_J.size} != {6*nv}")
    else:
        cuda_J = cuda_J.reshape(6, nv, order="F")
        if not np.allclose(cuda_J, ref_J, atol=grad_atol, rtol=grad_atol):
            failures.append(
                f"gradient device: max|d|={np.max(np.abs(cuda_J - ref_J)):.3e}\n"
                f"  cuda=\n{cuda_J}\n  ref =\n{ref_J}")
        # hessian device co-computes the gradient: must agree bit-for-bit-ish
        cuda_J2 = np.asarray(out["end_effector_pose_hessian_gradient"],
                             dtype=np.float64).reshape(6, nv, order="F")
        if not np.allclose(cuda_J2, cuda_J, atol=1e-6, rtol=1e-6):
            failures.append(
                f"hessian-device gradient vs gradient device: "
                f"max|d|={np.max(np.abs(cuda_J2 - cuda_J)):.3e}")
        for k in range(4):
            row = np.asarray(out[f"end_effector_pose_gradient_batch_{k}"],
                             dtype=np.float64).reshape(6, nv, order="F")
            if not np.allclose(row, cuda_J, atol=1e-5, rtol=1e-5):
                failures.append(
                    f"gradient batch[{k}] vs device: max|d|={np.max(np.abs(row - cuda_J)):.3e}")

    # --- hessian ((6, nv, nv) C-order) ---
    ref_H = _oracle_hessian(ref, leaf_name, q)
    assert ref_H.shape == (6, nv, nv)
    cuda_H = np.asarray(out["end_effector_pose_hessian"], dtype=np.float64)
    if cuda_H.size != 6 * nv * nv:
        failures.append(f"hessian device size {cuda_H.size} != {6*nv*nv}")
    else:
        cuda_H = cuda_H.reshape(6, nv, nv)
        if not np.allclose(cuda_H, ref_H, atol=hess_atol, rtol=hess_atol):
            failures.append(
                f"hessian device: max|d|={np.max(np.abs(cuda_H - ref_H)):.3e}")
        # symmetry of the (vi, vj) pair axes
        if not np.allclose(cuda_H, np.transpose(cuda_H, (0, 2, 1)), atol=1e-5):
            failures.append("hessian device not (vi, vj)-symmetric")
        for k in range(4):
            blk = np.asarray(out[f"end_effector_pose_hessian_batch_{k}"],
                             dtype=np.float64).reshape(6, nv, nv)
            if not np.allclose(blk, cuda_H, atol=1e-5, rtol=1e-5):
                failures.append(
                    f"hessian batch[{k}] vs device: max|d|={np.max(np.abs(blk - cuda_H)):.3e}")

    assert not failures, "spherical ee grad/hess CUDA equivalence failures:\n" + "\n".join(failures)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("fixture", ["spherical_arm.urdf", "mixed_spherical_arm.urdf"])
def test_cuda_spherical_eepose_gradhess_thread_invariant(tmp_path, fixture):
    """The single-block spherical ee-pose gradient/hessian kernels must be
    exactly thread-count invariant (bit-identical at 1/32/256)."""
    robot = _parse(fixture)
    assert robot is not None
    exe = _compile_runner(tmp_path) if _generate_header(robot, tmp_path) else None

    q = _benign_q(robot, fixture)
    base_out = _run(exe, q, threads=32)
    cells = [
        ("end_effector_pose_gradient", "end_effector_pose_gradient_batch_"),
        ("end_effector_pose_hessian", "end_effector_pose_hessian_batch_"),
    ]
    baselines = {key: np.asarray(base_out[key], dtype=np.float64) for key, _ in cells}
    failures = []
    for threads in (1, 32, 256):
        out = _run(exe, q, threads=threads)
        for key, batch_prefix in cells:
            dev = np.asarray(out[key], dtype=np.float64)
            if not np.array_equal(dev, baselines[key]):
                failures.append(
                    f"{fixture} threads={threads}: {key} device differs from threads=32 "
                    f"(max|d|={np.max(np.abs(dev - baselines[key])):.3e})")
            for k in range(4):
                row = np.asarray(out[f"{batch_prefix}{k}"], dtype=np.float64)
                if not np.array_equal(row, dev):
                    failures.append(
                        f"{fixture} threads={threads}: {key} batch[{k}] != device single-call")

    assert not failures, "spherical ee grad/hess thread-invariance failures:\n" + "\n".join(failures)

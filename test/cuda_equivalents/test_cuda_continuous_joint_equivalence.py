"""End-to-end CUDA verification of continuous (unbounded revolute) joints.

The CUDA counterpart to ``RBDReference/tests/test_continuous_joint_equivalence.py``.
That Python test pins the numpy-vs-pinocchio contract for the gen3 robot (4
continuous + 3 revolute joints): GRiD models a URDF ``continuous`` joint as a
RAW SCALAR ANGLE (NQ=NV=1) whereas pinocchio uses an SO(2) ``(cos, sin)`` pair,
yet all dynamics/kinematics OUTPUTS agree because they depend on the angle only
through ``cos``/``sin`` -- even at LARGE wrapped angles (theta = 5pi + delta),
where the raw-q representations differ most.

This module closes the matching CUDA gap (test_coverage_matrix.md item 8 / G2):
the generated CUDA kernels for continuous joints had NO end-to-end coverage. It
codegens gen3-fixed, drives the shared ``cuda_equivalence_runner.cu`` over the
SAME large-wrapped-angle states the Python test uses, and asserts the CUDA
``inverse_dynamics`` (gravity+coriolis, qdd=0), ``crba`` (M), and
``end_effector_pose`` outputs match the verified RBDReference numpy reference.

If the CUDA codegen ever mishandled a continuous joint (e.g. wrapped its scalar
angle through an SO(2) layout, or got the transform's cos/sin wrong), this test
fails at the large-wrapped-angle samples while a small-angle smoke would pass.

Gravity convention: unified at -9.81. The runner passes gravity = -9.81 to GRiD,
matching the RBDReference adapter's inverse_dynamics default -9.81 -- both sides use one
convention, the pairing every CUDA dynamics equivalence test relies on.
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
    _detect_cuda_arch,
    _parse_runner_output,
    GPU_UNAVAILABLE_PATTERNS,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter

RUNNER_SOURCE = Path(__file__).with_name("cuda_equivalence_runner.cu")

# The continuous-joint robot. gen3: 4 continuous + 3 revolute joints; the only
# robot in the manifest that exercises the continuous-joint codegen path. The
# fixed-base case has NQ==NV (raw scalar angle); the floating-base case prepends
# the 7-dim free-flyer (translation + quaternion) to the same joint chain and so
# exercises the wraparound-safe continuous path on top of the floating root.
_ROBOT_ID = "gen3"


def _gen3_spec(base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == _ROBOT_ID:
            return case["spec"]
    return None


GEN3_SPEC = _gen3_spec("fixed")

pytestmark = pytest.mark.skipif(
    GEN3_SPEC is None,
    reason="gen3 (continuous-joint robot) not in manifest",
)


def _continuous_joint_index_q(robot):
    return [
        robot.get_joint_index_q(j.get_id())
        for j in robot.get_joints_ordered_by_id()
        if j.jtype == "continuous"
    ]


def _wrapped_angle_states(project_model, n_trials=4):
    """States with the continuous-joint coordinates pushed many full turns past
    [-pi, pi] -- exactly the comparator guard the Python twin uses.

    Base-aware: for a floating base the q-vector is [translation(3); quat(4,
    xyzw); joints], so we seed a *normalized* quaternion (an off-manifold quat
    would make the reference invalid) and push the continuous-joint coordinates
    -- which live in the joint segment -- past +-pi. The continuous-joint
    q-indices come from get_joint_index_q, which already accounts for the
    floating-root offset, so the same wrap loop works for both bases.
    """
    robot = project_model.robot
    cont_iq = _continuous_joint_index_q(robot)
    assert cont_iq, "gen3 should expose continuous joints"
    rng = np.random.default_rng(11)
    nq, nv = project_model.nq, project_model.nv
    floating = project_model.base_mode == "floating"
    states = []
    for _ in range(n_trials):
        q = rng.uniform(-1.0, 1.0, size=nq)
        if floating:
            q[0:3] = rng.uniform(-0.35, 0.35, size=3)          # translation
            quat = rng.uniform(-1.0, 1.0, size=4)
            q[3:7] = quat / np.linalg.norm(quat)               # normalized xyzw
        for iq in cont_iq:
            q[iq] = q[iq] + 2.0 * np.pi * rng.integers(-3, 4)
        qd = rng.uniform(-0.8, 0.8, size=nv)
        states.append((q.astype(np.float64), qd.astype(np.float64)))
    return states


def _generate_header(project_model, build_dir):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        project_model.robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid"
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        # Emit only the first-order kernels the shared runner references
        # unconditionally (id / minv / fd / aba / crba / ee_pose); we compile the
        # runner with GRID_RUNNER_SKIP_GRADIENTS=1 so it does not reference the
        # heavy gradient / second-order kernels, keeping the gen3 nvcc compile
        # cheap. continuous-joint handling lives in the per-joint transform
        # codegen shared by all of these, so this subset fully exercises the path.
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header),
            algorithm_list="inverse_dynamics,minv,forward_dynamics,aba,crba,end_effector_pose",
        )
    return header


def _compile_runner(build_dir, floating=False):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    exe = build_dir / "cuda_continuous_joint_runner.exe"
    cmd = [
        nvcc, "-std=c++11", "-O0",
        f"-DGRID_CUDA_FLOATING_BASE={1 if floating else 0}",
        # gen3 exercises id / crba / ee_pose only; skip the gradient and
        # ee-pose-gradient runner sections so we don't need the heavy gradient /
        # second-order kernels in the (lean) header. (SKIP_GRADIENTS also defaults
        # SKIP_EEPOSE_GRADIENTS, so the floating runner block drops its
        # id_du/fd_du/ee-derivative launches too.)
        "-DGRID_RUNNER_SKIP_GRADIENTS=1",
        "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-gencode", f"arch=compute_{arch},code=compute_{arch}",
        "-o", str(exe), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA continuous-joint runner compilation failed.\n"
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


def _ee_pose_reference(project_model, q):
    """Concatenated 6-vector pose over every leaf EE (matches the runner's
    end_effector_pose output layout 6*NUM_EES)."""
    robot = project_model.robot
    poses = []
    for jid in robot.get_leaf_nodes():
        target = robot.get_joint_by_id(jid).get_name()
        poses.append(
            np.asarray(project_model.end_effector_pose(q, target), dtype=np.float64).reshape(-1)
        )
    return np.concatenate(poses)


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("base_mode", ["fixed", "floating"], ids=lambda b: f"gen3-{b}")
def test_cuda_continuous_joint_matches_reference_at_wrapped_angles(tmp_path, base_mode):
    """CUDA id / crba / ee_pose for gen3 must match the RBDReference numpy
    reference even when the continuous-joint angles are wrapped many turns.

    Both bases are covered: gen3-fixed (NQ==NV raw scalar angle) and gen3-floating
    (the same continuous chain on top of the 7-dim free-flyer root, NQ==NV+1). The
    floating cell exercises the wraparound-safe continuous path through the
    floating-base code path -- the numpy reference already covers gen3-floating, so
    this closes the matching CUDA gap.
    """
    spec = _gen3_spec(base_mode)
    if spec is None:
        pytest.skip(f"gen3-{base_mode} not in manifest")
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./developer_install.sh "
            f"before executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    nq, nv = project_model.nq, project_model.nv
    if base_mode == "fixed":
        # Raw-scalar-angle representation: continuous joints do not expand q.
        assert nq == nv, "gen3-fixed should have NQ == NV"
    else:
        # Floating root adds the 7-dim free-flyer config but only 6 velocity DoF.
        assert nq == nv + 1, "gen3-floating should have NQ == NV + 1"

    floating = base_mode == "floating"
    exe = _compile_runner_and_header(project_model, tmp_path, floating=floating)

    zeros = np.zeros(nv, dtype=np.float64)
    failures = []
    for trial, (q, qd) in enumerate(_wrapped_angle_states(project_model)):
        out = _run(exe, q, qd, zeros)
        tag = f"gen3-{base_mode} wrapped-angle trial {trial}"

        # ---- inverse_dynamics (qdd=0 -> gravity + coriolis); runner and the
        # adapter inverse_dynamics both use -9.81 (unified convention). The adapter
        # surface returns the normalized c vector (the raw reference returns a
        # 4-tuple), matching the runner output layout.
        cuda_id = np.asarray(out["inverse_dynamics"], dtype=np.float64).reshape(-1)
        ref_id = np.asarray(project_model.inverse_dynamics(q, qd, zeros), dtype=np.float64).reshape(-1)
        _check(failures, f"{tag} inverse_dynamics", cuda_id, ref_id)

        # ---- crba (mass matrix M)
        cuda_m = np.asarray(out["crba"], dtype=np.float64).reshape(nv, nv, order="F")
        ref_m = np.asarray(project_model.crba(q), dtype=np.float64)
        _check(failures, f"{tag} crba", cuda_m, ref_m)

        # ---- end_effector_pose (depends on continuous angle only via cos/sin)
        cuda_ee = np.asarray(out["end_effector_pose"], dtype=np.float64).reshape(-1)
        ref_ee = _ee_pose_reference(project_model, q)
        _check(failures, f"{tag} end_effector_pose", cuda_ee, ref_ee)

    assert not failures, "continuous-joint CUDA equivalence failures:\n" + "\n".join(failures)


def _compile_runner_and_header(project_model, tmp_path, floating=False):
    _generate_header(project_model, tmp_path)
    return _compile_runner(tmp_path, floating=floating)


def _check(failures, label, cuda, ref):
    cuda = np.asarray(cuda, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    if cuda.shape != ref.shape:
        failures.append(f"{label}: CUDA shape {cuda.shape} != reference {ref.shape}")
        return
    scale = max(1.0, float(np.max(np.abs(ref))) if ref.size else 1.0)
    # float32 CUDA path: scale the absolute floor by the value magnitude (the
    # same headroom the other CUDA smoke tests use).
    atol = 2e-3 * scale + 2e-3
    err = float(np.max(np.abs(cuda - ref))) if ref.size else 0.0
    if err > atol:
        failures.append(f"{label}: CUDA-vs-reference maxerr={err:.3e} > {atol:.3e}")

"""Polynomial couplings on the GPU against the reference's projection of the
full tree (item E of 0.8, 2026-11-03).

The chain of polynomial_coupling_model has two followers whose fold
multiplier alpha(q_l) = dq_f/dq_l varies with the leader; the generated
inverse dynamics, CRBA, Minv, forward dynamics and ABA read alpha and alpha'
at runtime (the acceleration gains alpha' qd_l^2). Both bases; the outputs
must be thread-count invariant.
"""
import contextlib
import io

import numpy as np
import pytest

from grid_codegen import GRiDCodeGenerator
from RBDReference import RBDReference
from URDFParser import URDFParser
from test.cuda_equivalents.polynomial_coupling_model import write_polynomial_chain
from test.cuda_equivalents.thread_invariance import thread_sweep

pytestmark = [pytest.mark.cuda_equivalence, pytest.mark.robot_smoke]

ALGOS = ["inverse_dynamics", "crba", "minv", "forward_dynamics", "aba",
         "inverse_dynamics_gradient", "forward_dynamics_gradient",
         "end_effector_pose", "end_effector_pose_gradient"]


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
def test_polynomial_coupling_dynamics_match_projection(tmp_path, monkeypatch, floating):
    from test.cuda_equivalents.cuda_harness import _compile_runner, _parse_runner_output, _run_runner

    urdf = write_polynomial_chain(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(urdf), floating_base=floating)
    gen = GRiDCodeGenerator(robot, False, True, "grid")
    with contextlib.redirect_stdout(io.StringIO()):
        gen.gen_all_code(output_path=str(tmp_path / "grid.cuh"), algorithm_list=ALGOS,
                         enable_mujoco_kernels=False)
    monkeypatch.setenv("GRID_CUDA_CACHE_DIR", str(tmp_path / "executables"))
    exe, command = _compile_runner(tmp_path, floating_base=floating,
                                   run_tokens={"RUN_" + a.upper() for a in ALGOS},
                                   skip_gradients=False, skip_eepose_gradients=False)
    ref = RBDReference(robot)
    assert ref._embedding is not None
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    leaf = robot.get_joint_by_id(int(robot.get_leaf_nodes()[0])).get_name()
    rng = np.random.default_rng(7)
    for sample in range(3):
        q = rng.standard_normal(nq)
        if floating:
            q[3:7] /= np.linalg.norm(q[3:7])
        qd = rng.standard_normal(nv)
        u = rng.standard_normal(nv)
        expected = {
            "inverse_dynamics": ref.inverse_dynamics(q, qd, None)[0],      # the runner's ID has qdd = 0
            "crba": ref.crba(q),
            "minv": ref.minv(q),
            "forward_dynamics": ref.forward_dynamics(q, qd, u),
            "aba": ref.aba(q, qd, u),
            # the runner's gradients: dc/du at qdd = 0 (rows = output, cols = variable), dqdd/du at torque u
            "inverse_dynamics_gradient_q": ref.inverse_dynamics_gradient(q, qd, None)[:, :nv],
            "inverse_dynamics_gradient_qd": ref.inverse_dynamics_gradient(q, qd, None)[:, nv:],
            "forward_dynamics_gradient_q": ref.forward_dynamics_gradient(q, qd, u)[0],
            "forward_dynamics_gradient_qd": ref.forward_dynamics_gradient(q, qd, u)[1],
            # the leaf end effector: pose (6) and its gradient (6 x nv, column-major flatten)
            "end_effector_pose": np.asarray(ref.end_effector_pose(q, leaf), np.float64).reshape(-1),
            "end_effector_pose_gradient": np.asarray(ref.end_effector_pose_gradient(q, leaf), np.float64).reshape(-1, order="F"),
        }
        stdin = "\n".join(" ".join(f"{v:.9g}" for v in np.asarray(x, np.float32)) for x in (q, qd, u)) + "\n"

        def run(threads):
            out = _parse_runner_output(_run_runner(exe, stdin, command, num_threads=threads))
            flat = []
            for name, exp in expected.items():
                assert name in out, (name, sorted(out))
                got = np.asarray(out[name], np.float64)
                exp = np.asarray(exp, np.float64)
                if got.shape != exp.shape:
                    got = got.reshape(exp.shape) if exp.ndim == 2 else got.reshape(-1)
                scale = max(1.0, float(np.max(np.abs(exp))))
                np.testing.assert_allclose(got, exp, rtol=0, atol=3e-4 * scale,
                                           err_msg=f"{name} sample {sample} floating={floating} threads={threads}")
                flat.append(got.reshape(-1))
            return np.concatenate(flat)

        thread_sweep((1, 32, 64, 128), run)


_REF_FRAMES = (("local", "LOCAL"), ("world", "WORLD"), ("lwa", "LOCAL_WORLD_ALIGNED"))
_FRAME_ALGOS = ["frame_jacobian", "frame_jacobian_dot", "osc_inertia", "frame_velocity", "frame_acceleration",
                "frame_velocity_gradient", "frame_acceleration_gradient"]


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
def test_polynomial_coupling_frame_kinematics_match_projection(tmp_path, floating):
    """The frame Jacobian family on the polynomial chain: J, Jdot (with the
    alpha' qd_l term), Lambda, the frame velocity and acceleration and their
    derivatives, against the reference's projection, three frames, thread sweep."""
    from pathlib import Path
    from test.cuda_equivalents.cuda_harness import _detect_cuda_arch, _parse_runner_output, _run_runner
    from test.cuda_equivalents.executable_cache import cached_nvcc_executable

    urdf = write_polynomial_chain(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(urdf), floating_base=floating)
    gen = GRiDCodeGenerator(robot, False, True, "grid")
    with contextlib.redirect_stdout(io.StringIO()):
        gen.gen_all_code(output_path=str(tmp_path / "grid.cuh"), algorithm_list=_FRAME_ALGOS,
                         enable_mujoco_kernels=False)
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    arch = _detect_cuda_arch()
    flags = ["-std=c++17", "-O0", "-gencode", f"arch=compute_{arch},code=sm_{arch}", f"-I{glass_inc}"]
    here = Path(__file__).resolve().parent
    kin_exe, kin_cmd = cached_nvcc_executable(
        sources=[here / "cuda_frame_kinematics_smoke_runner.cu", tmp_path / "grid.cuh"], flags=flags,
        exe_name="cuda_frame_kinematics_smoke_runner", fallback_dir=tmp_path, include_dirs=[tmp_path],
        what=f"frame kinematics runner, polynomial chain floating={floating}")
    jac_exe, jac_cmd = cached_nvcc_executable(
        sources=[here / "cuda_frame_jacobian_smoke_runner.cu", tmp_path / "grid.cuh"], flags=flags,
        exe_name="cuda_frame_jacobian_smoke_runner", fallback_dir=tmp_path, include_dirs=[tmp_path],
        what=f"frame jacobian runner, polynomial chain floating={floating}")
    ref = RBDReference(robot)
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    leaf_id = int(robot.get_leaf_nodes()[0])
    leaf = robot.get_joint_by_id(leaf_id).get_name()
    rng = np.random.default_rng(11)
    for sample in range(2):
        q = rng.standard_normal(nq)
        if floating:
            q[3:7] /= np.linalg.norm(q[3:7])
        qd = rng.standard_normal(nv)
        qdd = rng.standard_normal(nv)
        expected = {}
        for blk, rf in _REF_FRAMES:
            J = np.asarray(ref.frame_jacobian(q, leaf, rf), np.float64)
            expected["J_" + blk] = J
            expected["Jd_" + blk] = np.asarray(ref.frame_jacobian_dot(q, qd, leaf, rf), np.float64)
            if nv >= 6:   # the task inertia exists only with six independent coordinates upstream
                expected["L_" + blk] = np.asarray(ref.osc_inertia(q, leaf, rf), np.float64)
            expected["V_" + blk] = np.asarray(ref.frame_velocity(q, qd, leaf, rf), np.float64)
            expected["A_" + blk] = np.asarray(ref.frame_acceleration(q, qd, qdd, leaf, rf), np.float64)
            dv_dq, dv_dqd = ref.frame_velocity_derivatives(q, qd, leaf, rf)
            _, da_dq, da_dqd, da_dqdd = ref.frame_acceleration_derivatives(q, qd, qdd, leaf, rf)
            expected["GV_" + blk] = np.hstack([dv_dq, dv_dqd])
            expected["GA_" + blk] = np.hstack([da_dq, da_dqd, da_dqdd])
        rows = [str(leaf_id)] + [" ".join(f"{v:.9g}" for v in np.asarray(x, np.float32)) for x in (q, qd, qdd)]
        kin_stdin = "\n".join(rows) + "\n"
        jac_stdin = "\n".join(rows[:3]) + "\n"

        def run(threads):
            out = _parse_runner_output(_run_runner(kin_exe, kin_stdin, kin_cmd, num_threads=threads))
            out.update(_parse_runner_output(_run_runner(jac_exe, jac_stdin, jac_cmd, num_threads=threads)))
            flat = []
            for name, exp in expected.items():
                assert name in out, (name, sorted(out))
                got = np.asarray(out[name], np.float64)
                if name.startswith(("J_", "Jd_", "L_")):
                    got = got.reshape(exp.shape, order="F")   # the jacobian runner prints column-major flats
                elif exp.ndim == 1:
                    got = got.reshape(-1)
                scale = max(1.0, float(np.max(np.abs(exp))))
                tol = 2e-3 if name.startswith(("J_", "V_")) else 5e-3
                np.testing.assert_allclose(got, exp, rtol=0, atol=tol * scale,
                                           err_msg=f"{name} sample {sample} floating={floating} threads={threads}")
                flat.append(got.reshape(-1))
            return np.concatenate(flat)

        thread_sweep((1, 32, 64, 128), run)

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

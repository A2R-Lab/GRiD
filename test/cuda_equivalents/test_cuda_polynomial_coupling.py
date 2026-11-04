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


_INTEGRATORS = (("integrator_euler", "euler"), ("integrator_si_euler", "semi_implicit_euler"),
                ("integrator_midpoint", "midpoint"), ("integrator_trapezoidal", "trapezoidal"),
                ("integrator_rk4", "rk4"), ("integrator_constant_acceleration", "constant_acceleration"))


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
def test_polynomial_coupling_integrators_match_projection(tmp_path, floating):
    """The six integrators and their [A | B] gradients on the polynomial chain:
    they compose the converted forward dynamics and its gradient with the
    reduced-coordinate retract, so they carry no fold of their own; checked
    against the reference (which integrates the projected dynamics)."""
    from pathlib import Path
    from test.cuda_equivalents.cuda_harness import _detect_cuda_arch, _parse_runner_output, _run_runner
    from test.cuda_equivalents.executable_cache import cached_nvcc_executable

    urdf = write_polynomial_chain(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(urdf), floating_base=floating)
    gen = GRiDCodeGenerator(robot, False, True, "grid")
    with contextlib.redirect_stdout(io.StringIO()):
        gen.gen_all_code(output_path=str(tmp_path / "grid.cuh"), codegen_profile="integrators",
                         enable_mujoco_kernels=False)
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    arch = _detect_cuda_arch()
    exe, cmd = cached_nvcc_executable(
        [Path(__file__).with_name("cuda_integrator_smoke_runner.cu"), tmp_path / "grid.cuh"],
        ["-std=c++17", "-O0", "-gencode", f"arch=compute_{arch},code=sm_{arch}"],
        exe_name="cuda_integrator_smoke_runner.exe", fallback_dir=tmp_path, include_dirs=[glass_inc],
        what=f"integrator runner, polynomial chain floating={floating}")
    ref = RBDReference(robot)
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    rng = np.random.default_rng(13)
    for sample in range(2):
        q = rng.standard_normal(nq)
        if floating:
            q[3:7] /= np.linalg.norm(q[3:7])
        qd = 0.5 * rng.standard_normal(nv)
        u = rng.standard_normal(nv)
        dt = 0.01
        stdin = "\n".join(" ".join(f"{v:.9g}" for v in np.asarray(x, np.float32)) for x in (q, qd, u)) + f"\n{dt:.9g}\n"
        expected = {}
        for prefix, kind in _INTEGRATORS:
            expected[prefix + "_x_kp1"] = np.asarray(ref.integrator(q, qd, u, dt, integrator_type=kind), np.float64).reshape(-1)
            expected[prefix + "_dAB"] = np.asarray(ref.integrator_gradient(q, qd, u, dt, integrator_type=kind), np.float64)

        def run(threads):
            out = _parse_runner_output(_run_runner(exe, stdin, cmd, num_threads=threads))
            flat = []
            for name, exp in expected.items():
                assert name in out, (name, sorted(out))
                got = np.asarray(out[name], np.float64)
                got = got.reshape(-1) if exp.ndim == 1 else got.reshape(exp.shape)
                scale = max(1.0, float(np.max(np.abs(exp))))
                np.testing.assert_allclose(got, exp, rtol=0, atol=2e-3 * scale,
                                           err_msg=f"{name} sample {sample} floating={floating} threads={threads}")
                flat.append(got.reshape(-1))
            return np.concatenate(flat)

        thread_sweep((1, 32, 128), run)


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
def test_polynomial_coupling_regressor_and_centroidal_match_projection(tmp_path, floating):
    """The joint-torque regressor (Y_y = G^T Y_f), the generalized gravity, the
    bias, the CoM and its Jacobian, the centroidal momentum matrix and the
    energies on the polynomial chain, against the projected reference."""
    from pathlib import Path
    from test.cuda_equivalents.cuda_harness import _detect_cuda_arch, _parse_runner_output, _run_runner
    from test.cuda_equivalents.executable_cache import cached_nvcc_executable

    urdf = write_polynomial_chain(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(urdf), floating_base=floating)
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    arch = _detect_cuda_arch()
    here = Path(__file__).resolve().parent
    # regressor header + runner
    reg_dir = tmp_path / "regressor"; reg_dir.mkdir()
    gen = GRiDCodeGenerator(robot, False, True, "grid")
    with contextlib.redirect_stdout(io.StringIO()):
        gen.gen_all_code(output_path=str(reg_dir / "grid.cuh"), codegen_profile="regressor",
                         include_homogenous_transforms=True, enable_mujoco_kernels=False)
    reg_exe, reg_cmd = cached_nvcc_executable(
        [here / "cuda_regressor_smoke_runner.cu", reg_dir / "grid.cuh"],
        ["-std=c++17", "-O0", f"-DGRID_CUDA_FLOATING_BASE={1 if floating else 0}",
         "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS", "-DGRID_CUDA_REGRESSOR_TEST_THREADS=64",
         "-gencode", f"arch=compute_{arch},code=sm_{arch}"],
        exe_name="cuda_regressor_smoke_runner.exe", fallback_dir=reg_dir, include_dirs=[glass_inc, reg_dir],
        what=f"regressor runner, polynomial chain floating={floating}")
    # centroidal header + the mimic runner (no plant costs)
    cen_dir = tmp_path / "centroidal"; cen_dir.mkdir()
    gen = GRiDCodeGenerator(robot, False, True, "grid")
    with contextlib.redirect_stdout(io.StringIO()):
        gen.gen_all_code(output_path=str(cen_dir / "grid.cuh"),
                         algorithm_list=["generalized_gravity", "nonlinear_effects", "com", "ccrba", "energy"],
                         enable_mujoco_kernels=False)
    cen_exe, cen_cmd = cached_nvcc_executable(
        [here / "cuda_centroidal_mimic_smoke_runner.cu", cen_dir / "grid.cuh"],
        ["-std=c++17", "-O0", "-gencode", f"arch=compute_{arch},code=sm_{arch}", f"-I{glass_inc}"],
        exe_name="cuda_centroidal_mimic_smoke_runner.exe", fallback_dir=cen_dir, include_dirs=[cen_dir],
        what=f"centroidal runner, polynomial chain floating={floating}")
    ref = RBDReference(robot)
    nq, nv, nb = robot.get_num_pos(), robot.get_num_vel(), robot.get_num_bodies()
    rng = np.random.default_rng(17)
    for sample in range(2):
        q = rng.standard_normal(nq)
        if floating:
            q[3:7] /= np.linalg.norm(q[3:7])
        qd = rng.standard_normal(nv)
        qdd = rng.standard_normal(nv)
        line = lambda x: " ".join(f"{v:.9g}" for v in np.asarray(x, np.float32))
        reg_stdin = "\n".join(line(x) for x in (q, qd, qdd)) + "\n"
        cen_stdin = "\n".join(line(x) for x in (q, qd)) + "\n"
        Y = np.asarray(ref.inverse_dynamics_regressor(q, qd, qdd), np.float64)
        A, h = ref.ccrba(q, qd)
        expected = {
            "regressor": Y,
            "gen_gravity": np.asarray(ref.generalized_gravity(q), np.float64).reshape(-1),
            "nonlinear": np.asarray(ref.nonlinear_effects(q, qd), np.float64).reshape(-1),
            "com": np.asarray(ref.com(q), np.float64).reshape(-1),
            "jcom": np.asarray(ref.jacobian_com(q), np.float64),
            "ccrba_A": np.asarray(A, np.float64),
            "ccrba_h": np.asarray(h, np.float64).reshape(-1),
            "energy": np.array([ref.kinetic_energy(q, qd), ref.potential_energy(q), ref.mechanical_energy(q, qd)]),
        }
        out = _parse_runner_output(_run_runner(reg_exe, reg_stdin, reg_cmd))
        out.update(_parse_runner_output(_run_runner(cen_exe, cen_stdin, cen_cmd)))
        for name, exp in expected.items():
            assert name in out, (name, sorted(out))
            got = np.asarray(out[name], np.float64)
            if name == "regressor":
                got = got.reshape(nv, 10 * nb)
            elif exp.ndim == 2:
                got = got.reshape(exp.shape, order="F")
            else:
                got = got.reshape(-1)
            scale = max(1.0, float(np.max(np.abs(exp))))
            np.testing.assert_allclose(got, exp, rtol=0, atol=3e-4 * scale,
                                       err_msg=f"{name} sample {sample} floating={floating}")


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
def test_polynomial_coupling_runtime_pose_match_projection(tmp_path, floating):
    """The runtime-target pose and pose gradient (with and without a tool
    transform) on the polynomial chain, against the projected reference."""
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    from test.cuda_equivalents.cuda_harness import _detect_cuda_arch, _parse_runner_output, _run_runner
    from test.cuda_equivalents.executable_cache import cached_nvcc_executable

    urdf = write_polynomial_chain(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(urdf), floating_base=floating)
    gen = GRiDCodeGenerator(robot, False, True, "grid")
    with contextlib.redirect_stdout(io.StringIO()):
        gen.gen_all_code(output_path=str(tmp_path / "grid.cuh"),
                         algorithm_list=["end_effector_pose_runtime", "end_effector_pose_gradient_runtime"],
                         enable_mujoco_kernels=False)
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    arch = _detect_cuda_arch()
    exe, cmd = cached_nvcc_executable(
        [Path(__file__).with_name("cuda_eepose_runtime_smoke_runner.cu"), tmp_path / "grid.cuh"],
        ["-std=c++17", "-O0", "-DGRID_CUDA_EEPOSE_RT_SCALAR=float", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
         f"-I{glass_inc}"],
        exe_name="cuda_eepose_runtime_smoke_runner.exe", fallback_dir=tmp_path, include_dirs=[tmp_path],
        what=f"runtime pose runner, polynomial chain floating={floating}")
    ref = RBDReference(robot)
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    leaf_id = int(robot.get_leaf_nodes()[0])
    leaf = robot.get_joint_by_id(leaf_id).get_name()
    Xtool = np.eye(4)
    Xtool[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.5]).as_matrix()
    Xtool[:3, 3] = [0.05, -0.02, 0.08]
    rng = np.random.default_rng(23)
    checked = 0
    for sample in range(4):
        q = rng.standard_normal(nq)
        if floating:
            q[3:7] /= np.linalg.norm(q[3:7])
        ref0 = np.asarray(ref.end_effector_pose(q, [leaf], None), np.float64).reshape(-1)
        refN = np.asarray(ref.end_effector_pose(q, [leaf], [Xtool]), np.float64).reshape(-1)
        if min(abs(abs(ref0[4]) - np.pi / 2), abs(abs(refN[4]) - np.pi / 2)) < 0.15:
            continue   # the rpy rows are singular near +-pi/2 pitch (the oracle and the device alike)
        grad0 = np.asarray(ref.end_effector_pose_gradient(q, [leaf], None), np.float64).reshape(6, nv)
        gradN = np.asarray(ref.end_effector_pose_gradient(q, [leaf], [Xtool]), np.float64).reshape(6, nv)
        rows = [str(leaf_id), " ".join(f"{v:.9g}" for v in np.asarray(Xtool, np.float32).reshape(-1, order="F")),
                " ".join(f"{v:.9g}" for v in np.asarray(q, np.float32))]
        out = _parse_runner_output(_run_runner(exe, "\n".join(rows) + "\n", cmd))
        for name, exp in (("pose0", ref0), ("poseN", refN)):
            got = np.asarray(out[name], np.float64).reshape(-1)
            np.testing.assert_allclose(got[:3], exp[:3], rtol=0, atol=3e-4 * max(1.0, float(np.max(np.abs(exp[:3])))), err_msg=f"{name} xyz sample {sample}")
            np.testing.assert_allclose((got[3:] - exp[3:] + np.pi) % (2 * np.pi) - np.pi, np.zeros(3), rtol=0, atol=3e-4, err_msg=f"{name} rpy sample {sample}")
        for name, exp in (("grad0", grad0), ("gradN", gradN)):
            got = np.asarray(out[name], np.float64).reshape(6, nv, order="F")
            scale = max(1.0, float(np.max(np.abs(exp))))
            np.testing.assert_allclose(got, exp, rtol=0, atol=5e-4 * scale, err_msg=f"{name} sample {sample}")
        checked += 1
    assert checked >= 2


_JD_ALGOS = ["inverse_dynamics", "forward_dynamics", "aba", "inverse_dynamics_gradient", "forward_dynamics_gradient"]


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
@pytest.mark.parametrize("model", ["sign", "tanh"])
@pytest.mark.parametrize("kind", ["linear", "polynomial"])
def test_coupled_joint_dynamics_match_reference(tmp_path, monkeypatch, kind, model, floating):
    """Damping and Coulomb friction on coupled joints: alpha^2 b and |alpha| f
    in the leader's slot (user ruling 2026-11-03), on a chain with ratios 2
    and -1.5 (linear) and on the polynomial chain, whose fold multiplier and
    therefore its damping coefficient vary with the leader's q (a dc_dq term
    the gradient carries). The reference is checked against the projection
    of a full tree that carries the damping itself and against finite
    differences in RBDReference/tests/test_joint_dynamics_coupling_fold.py."""
    from test.cuda_equivalents.cuda_harness import _compile_runner, _parse_runner_output, _run_runner

    urdf = write_polynomial_chain(tmp_path, kind=kind, joint_dynamics=True)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(urdf), floating_base=floating)
    assert robot.robot_has_joint_damping() and robot.robot_has_joint_friction()
    width = 0.05
    gen = GRiDCodeGenerator(robot, False, True, "grid", USE_JOINT_DYNAMICS=True,
                            FRICTION_MODEL=model, FRICTION_WIDTH=width)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.gen_all_code(output_path=str(tmp_path / "grid.cuh"), algorithm_list=_JD_ALGOS,
                         enable_mujoco_kernels=False)
    header = (tmp_path / "grid.cuh").read_text()
    if kind == "polynomial":
        assert "grid_coupling_alpha_" in header and "fabs(" in header, "runtime alpha fold missing"
    monkeypatch.setenv("GRID_CUDA_CACHE_DIR", str(tmp_path / "executables"))
    exe, command = _compile_runner(tmp_path, floating_base=floating,
                                   run_tokens={"RUN_" + a.upper() for a in _JD_ALGOS},
                                   skip_gradients=False, skip_eepose_gradients=True)
    ref = RBDReference(robot, use_joint_dynamics=True, friction_model=model, friction_width=width)
    bare = RBDReference(robot)
    assert (ref._embedding is None) == (kind == "linear")
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    rng = np.random.default_rng(13)
    for sample in range(3):
        q = rng.standard_normal(nq)
        if floating:
            q[3:7] /= np.linalg.norm(q[3:7])
        qd = rng.uniform(0.3, 1.5, nv) * rng.choice([-1.0, 1.0], nv)   # away from the Coulomb switch
        u = rng.standard_normal(nv)
        c = ref.inverse_dynamics(q, qd, None)[0]
        # the fold is active: the damped bias differs from the undamped one by the joint torques
        assert float(np.max(np.abs(c - bare.inverse_dynamics(q, qd, None)[0]))) > 1e-2
        expected = {
            "inverse_dynamics": c,
            "forward_dynamics": ref.forward_dynamics(q, qd, u),
            "aba": ref.aba(q, qd, u),
            "inverse_dynamics_gradient_q": ref.inverse_dynamics_gradient(q, qd, None)[:, :nv],
            "inverse_dynamics_gradient_qd": ref.inverse_dynamics_gradient(q, qd, None)[:, nv:],
            "forward_dynamics_gradient_q": ref.forward_dynamics_gradient(q, qd, u)[0],
            "forward_dynamics_gradient_qd": ref.forward_dynamics_gradient(q, qd, u)[1],
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
                                           err_msg=f"{name} {kind} {model} sample {sample} floating={floating} threads={threads}")
                flat.append(got.reshape(-1))
            return np.concatenate(flat)

        thread_sweep((1, 32, 64, 128), run)


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
def test_polynomial_coupling_coriolis_matches_projection(tmp_path, floating):
    """The Coriolis matrix of a polynomial robot: C_y = G^T C_f G + G^T M_f Gdot,
    whose inertia term the kernel assembles from the composite inertias of
    the backward pass (kind 2 jobs) with alpha(q) and alpha'(q) read at
    runtime. The reference projection is pinned by C yd + g = bias and
    C + C^T = dM/dt in RBDReference/tests/test_embedding_projection.py."""
    from test.cuda_equivalents.cuda_harness import _parse_runner_output, _run_runner
    from test.cuda_equivalents.test_cuda_coriolis import _compile_runner as _compile_coriolis_runner

    urdf = write_polynomial_chain(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(urdf), floating_base=floating)
    gen = GRiDCodeGenerator(robot, False, True, "grid")
    with contextlib.redirect_stdout(io.StringIO()):
        gen.gen_all_code(output_path=str(tmp_path / "grid.cuh"), algorithm_list=["coriolis_matrix"],
                         enable_mujoco_kernels=False)
    header = (tmp_path / "grid.cuh").read_text()
    assert "cor_col_alphad" in header and "kind == 2" in header, "the embedding's inertia term is missing"
    exe, command = _compile_coriolis_runner(tmp_path, floating)
    ref = RBDReference(robot)
    assert ref._embedding is not None
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    rng = np.random.default_rng(17)
    for sample in range(3):
        q = rng.standard_normal(nq)
        if floating:
            q[3:7] /= np.linalg.norm(q[3:7])
        qd = rng.standard_normal(nv)
        u = rng.standard_normal(nv)
        C_ref = np.asarray(ref.coriolis_matrix(q, qd), np.float64).reshape(nv, nv)
        # the inertia term is active on this chain
        G = ref._embedding.G(q)
        C_plain = G.T @ ref._embedding.full.coriolis_matrix(ref._embedding.expand_q(q), G @ qd) @ G
        assert float(np.max(np.abs(C_ref - C_plain))) > 1e-3
        stdin = "\n".join(" ".join(f"{v:.9g}" for v in np.asarray(x, np.float32)) for x in (q, qd, u)) + "\n"
        scale = max(1.0, float(np.max(np.abs(C_ref))))

        def run(threads):
            out = _parse_runner_output(_run_runner(exe, stdin, command, num_threads=threads))
            got = np.asarray(out["coriolis"], np.float64).reshape(nv, nv)
            np.testing.assert_allclose(got, C_ref, rtol=0, atol=3e-4 * scale,
                                       err_msg=f"coriolis sample {sample} floating={floating} threads={threads}")
            return got.reshape(-1)

        thread_sweep((1, 32, 64, 128), run)


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
def test_polynomial_coupling_dccrba_matches_projection(tmp_path, floating):
    """The momentum-matrix derivatives of a polynomial robot, A_y = A_f G:
    the tensor dA_y[:, k, i] and Adot_y carry, beyond the alpha-scaled units,
    the embedding term A_f[:, fv] alpha'(q_l) (times qd_l when contracted) at
    the leader's cell. The projection is pinned against the chart and the
    flow in RBDReference/tests/test_embedding_projection.py."""
    from test.cuda_equivalents.cuda_harness import _parse_runner_output, _run_runner
    from test.cuda_equivalents.test_cuda_dccrba import _compile_runner as _compile_dccrba_runner, _dccrba_flat_to_tensor

    urdf = write_polynomial_chain(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(urdf), floating_base=floating)
    gen = GRiDCodeGenerator(robot, False, True, "grid")
    with contextlib.redirect_stdout(io.StringIO()):
        gen.gen_all_code(output_path=str(tmp_path / "grid.cuh"), algorithm_list=["dccrba", "cmm_time_variation"],
                         enable_mujoco_kernels=False)
    header = (tmp_path / "grid.cuh").read_text()
    assert "dc_unit_alphad" in header, "the embedding term is missing"
    exe, command = _compile_dccrba_runner(tmp_path, floating)
    ref = RBDReference(robot)
    assert ref._embedding is not None
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    rng = np.random.default_rng(19)
    for sample in range(3):
        q = rng.standard_normal(nq)
        if floating:
            q[3:7] /= np.linalg.norm(q[3:7])
        qd = rng.standard_normal(nv)
        u = rng.standard_normal(nv)
        dA_ref = np.asarray(ref.dccrba(q), np.float64).reshape(6, nv, nv)
        adot_ref = np.asarray(ref.cmm_time_variation(q, qd), np.float64).reshape(6, nv)
        # the embedding term is active on this chain
        G = ref._embedding.G(q)
        plain = np.einsum("ajm,jk,mi->aki", np.asarray(ref._embedding.full.dccrba(ref._embedding.expand_q(q))), G, G)
        assert float(np.max(np.abs(dA_ref - plain))) > 1e-3
        stdin = "\n".join(" ".join(f"{v:.9g}" for v in np.asarray(x, np.float32)) for x in (q, qd, u)) + "\n"
        dA_scale = max(1.0, float(np.max(np.abs(dA_ref))))
        adot_scale = max(1.0, float(np.max(np.abs(adot_ref))))

        def run(threads):
            out = _parse_runner_output(_run_runner(exe, stdin, command, num_threads=threads))
            dA = _dccrba_flat_to_tensor(np.asarray(out["dccrba"], np.float64).reshape(-1), nv)
            adot = np.asarray(out["cmm_time_variation"], np.float64).reshape(6, nv, order="F")
            np.testing.assert_allclose(dA, dA_ref, rtol=0, atol=3e-4 * dA_scale,
                                       err_msg=f"dccrba sample {sample} floating={floating} threads={threads}")
            np.testing.assert_allclose(adot, adot_ref, rtol=0, atol=3e-4 * adot_scale,
                                       err_msg=f"cmm_time_variation sample {sample} floating={floating} threads={threads}")
            return np.concatenate([dA.reshape(-1), adot.reshape(-1)])

        thread_sweep((1, 32, 64, 128), run)

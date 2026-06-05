"""CUDA equivalence test for the generated `grid_plant` primitives (T6).

Validates the sibling `grid_plant::` namespace emitted after `grid::` closes:
  - quadratic_state_cost / quadratic_input_cost (value + gradient + GN-diag hessian)
  - ee_pos_cost (value + gradient over x=[q;qd] + GN hessian J_p^T W J_p)
  - joint_{position,velocity,torque}_barrier (log-barrier value/grad/hess)
  - plant_step / plant_step_gradient (thin wrappers over grid::integrator[_gradient])

Strategy (correctness only; mirrors the integrator smoke-test pattern):
  * codegen iiwa14-fixed with the full profile, compile cuda_plant_smoke_runner.cu;
  * the runner builds a DETERMINISTIC cost/bound setup (mirrored here exactly);
  * checks:
      - GN diagonal hessians vs a NumPy diag(Q)/diag(R) recompute;
      - ee_pos_cost value/grad/hess vs a NumPy J_p^T W (...) recompute using the
        DOUBLE-PRECISION Python reference Jacobian (the FD oracle is run in double
        to avoid float32 cancellation), plus a central-difference FD check of the
        ee cost gradient against the Python double EE pose;
      - barrier value/grad/hess vs a NumPy log-barrier recompute, AND that the
        deliberately-unbounded DOF 0 contributes EXACTLY zero (isfinite-skip);
      - plant_step_gradient == grid::integrator_gradient (pass-through), and
        plant_step == grid::integrator.

Default robot is iiwa14-fixed (cheap, gate here first). Override the robot set
with GRID_CUDA_PLANT_ROBOTS.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from GRiDCodeGenerator import GRiDCodeGenerator
from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _build_cuda_samples,
    _detect_cuda_arch,
    _parse_runner_output,
    _run_runner,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter


RUNNER_SOURCE = Path(__file__).with_name("cuda_plant_smoke_runner.cu")
_DT = float(os.environ.get("GRID_CUDA_PLANT_DT", "0.01"))
_MU = 0.1  # barrier weight (must match the runner)
_PLANT_EE = 0  # which end-effector the runner exercises (PLANT_EE)


# ---- deterministic problem setup (MUST match cuda_plant_smoke_runner.cu) ----
def _x_des(nx):  return np.array([0.1 * i for i in range(nx)], dtype=np.float64)
def _Qw(nx):     return np.array([1.0 + 0.5 * i for i in range(nx)], dtype=np.float64)
def _u_des(nu):  return np.array([-0.05 * i for i in range(nu)], dtype=np.float64)
def _Rw(nu):     return np.array([2.0 + 0.1 * i for i in range(nu)], dtype=np.float64)
def _Ww():       return np.array([10.0 + r for r in range(3)], dtype=np.float64)
# Centroidal CoM-cost setup (3 axes) and momentum-cost setup (6 components),
# mirroring com_pdes_val/com_W_val/mom_hdes_val/mom_W_val in the runner exactly.
def _com_pdes(): return np.array([0.2 + 0.1 * r for r in range(3)], dtype=np.float64)
def _com_W():    return np.array([3.0 + 0.5 * r for r in range(3)], dtype=np.float64)
def _mom_hdes(): return np.array([-0.3 + 0.15 * r for r in range(6)], dtype=np.float64)
def _mom_W():    return np.array([2.0 + 0.25 * r for r in range(6)], dtype=np.float64)


def _comma_env(name, default):
    raw = os.environ.get(name, default)
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def _robot_ids():
    return _comma_env("GRID_CUDA_PLANT_ROBOTS", "iiwa14")


# (robot_id, base_mode) cells for the centroidal (com/momentum) plant-cost check.
# com_cost/momentum_cost are emitted NON-MIMIC ONLY (see GRiDCodeGenerator/
# algorithms/_plant.py ~line 907), so validate on non-mimic robots only:
# iiwa14:fixed (cheap fixed-base) and go2:floating (floating-base, non-mimic).
def _centroidal_cells():
    raw = os.environ.get("GRID_CUDA_PLANT_CENTROIDAL_CELLS", "iiwa14:fixed,go2:floating")
    cells = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        robot_id, _, base = tok.partition(":")
        cells.append((robot_id, base or "fixed"))
    return cells


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not in manifest")


def _generate_header(project_model, build_dir):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(project_model.robot, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(codegen_profile="all", output_path=str(header))
    return header


def _compile_runner(build_dir):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_plant_smoke_runner.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "GLASS" / "include"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-I{glass_inc}", "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA plant smoke runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


def _stdin(q, qd, u, dt):
    rows = [" ".join(f"{v:.9g}" for v in np.asarray(vec, dtype=np.float32))
            for vec in (q, qd, u)]
    return "\n".join(rows) + f"\n{dt}\n"


def _run(executable, cmd, q, qd, u, dt):
    out = _run_runner(executable, _stdin(q, qd, u, dt), cmd)
    return _parse_runner_output(out)


def _ee_jacobian_and_pos(project_model, q):
    """Position rows (0..2) of the Python double EE pose + its Jacobian for EE 0."""
    leaf = project_model.robot.get_leaf_nodes()[_PLANT_EE]
    target = project_model.robot.get_joint_by_id(leaf).get_name()
    pose = np.asarray(project_model.end_effector_pose(q, target), dtype=np.float64).reshape(-1)
    J = np.asarray(project_model.end_effector_pose_gradient(q, target), dtype=np.float64)
    # J is 6 x nv (pose-deriv); position rows are 0..2.
    return pose[:3], J[:3, :]


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("robot_id", _robot_ids(), ids=lambda r: f"{r}-plant")
def test_cuda_plant_matches_reference(tmp_path, robot_id):
    base_mode = "fixed"
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_plant"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    executable, cmd = _compile_runner(build_dir)

    nq, nv = project_model.nq, project_model.nv
    nx, nu = nq + nv, nv
    samples = _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)

    rtol, atol = 2e-3, 2e-3

    def close(actual, expected, msg):
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64), expected,
            rtol=rtol, atol=max(atol, rtol * scale), err_msg=msg,
        )

    for sample in samples:
        q, qd = np.asarray(sample.q, np.float64), np.asarray(sample.qd, np.float64)
        u = np.asarray(sample.qdd, np.float64)
        out = _run(executable, cmd, q, qd, u, _DT)
        x = np.concatenate([q, qd])
        tag = f"{robot_id} @ {sample.name}"

        # ---------- quadratic state cost ----------
        Qw, x_des = _Qw(nx), _x_des(nx)
        r = x - x_des
        close(out["state_cost_value"].reshape(-1)[0], 0.5 * np.sum(Qw * r * r), f"{tag} state value")
        close(out["state_cost_grad"].reshape(-1), Qw * r, f"{tag} state grad")
        close(out["state_cost_hess"].reshape(nx, nx, order="F"), np.diag(Qw), f"{tag} state GN hess (diag(Q))")

        # ---------- quadratic input cost ----------
        Rw, u_des = _Rw(nu), _u_des(nu)
        ru = u - u_des
        close(out["input_cost_value"].reshape(-1)[0], 0.5 * np.sum(Rw * ru * ru), f"{tag} input value")
        close(out["input_cost_grad"].reshape(-1), Rw * ru, f"{tag} input grad")
        close(out["input_cost_hess"].reshape(nu, nu, order="F"), np.diag(Rw), f"{tag} input GN hess (diag(R))")

        # ---------- ee position cost (double-precision oracle) ----------
        p, J = _ee_jacobian_and_pos(project_model, q)   # double
        W = _Ww()
        rp = p  # p_des = 0
        # value / grad / hess analytic recompute
        ee_val = 0.5 * np.sum(W * rp * rp)
        grad_q = J.T @ (W * rp)
        grad_x = np.concatenate([grad_q, np.zeros(nv)])
        H_q = J.T @ np.diag(W) @ J
        H_x = np.zeros((nx, nx)); H_x[:nv, :nv] = H_q
        # the runner prints p(q) it actually used — sanity-check it matches the double oracle
        close(out["ee_pos"].reshape(-1), p, f"{tag} ee_pos vs double Python pose")
        close(out["ee_cost_value"].reshape(-1)[0], ee_val, f"{tag} ee value")
        close(out["ee_cost_grad"].reshape(-1), grad_x, f"{tag} ee grad (J^T W r; qd-block zero)")
        # qd-block must be EXACTLY zero
        assert np.all(np.asarray(out["ee_cost_grad"]).reshape(-1)[nv:] == 0.0), f"{tag} ee grad qd-block not exactly zero"
        close(out["ee_cost_hess"].reshape(nx, nx, order="F"), H_x, f"{tag} ee GN hess (J^T W J)")

        # ---------- FD check (in DOUBLE) of the ee cost gradient via the Python pose ----------
        # central difference of the double-precision ee cost value over each q DOF.
        h = 1e-6
        fd = np.zeros(nv)
        for j in range(nv):
            qp, qm = q.copy(), q.copy()
            qp[j] += h; qm[j] -= h
            pp, _ = _ee_jacobian_and_pos(project_model, qp)
            pm, _ = _ee_jacobian_and_pos(project_model, qm)
            vp = 0.5 * np.sum(W * pp * pp)
            vm = 0.5 * np.sum(W * pm * pm)
            fd[j] = (vp - vm) / (2 * h)
        close(grad_q, fd, f"{tag} ee grad vs double central-difference FD")

        # ---------- barriers (deterministic interior bounds; DOF 0 position unbounded) ----------
        def barrier_terms(vals, los, his):
            v = 0.0
            g = np.zeros(len(vals))
            hd = np.zeros(len(vals))
            for i in range(len(vals)):
                if np.isfinite(los[i]):
                    d = max(vals[i] - los[i], 1e-10); v -= np.log(d)
                    g[i] -= 1.0 / (vals[i] - los[i]); hd[i] += 1.0 / (vals[i] - los[i]) ** 2
                if np.isfinite(his[i]):
                    d = max(his[i] - vals[i], 1e-10); v -= np.log(d)
                    g[i] += 1.0 / (his[i] - vals[i]); hd[i] += 1.0 / (his[i] - vals[i]) ** 2
            return _MU * v, _MU * g, _MU * hd

        # position barrier (q block); DOF 0 unbounded
        lo_q = q - 1.0; hi_q = q + 1.0
        lo_q[0] = -np.inf; hi_q[0] = np.inf
        bv, bg, bh = barrier_terms(q, lo_q, hi_q)
        close(out["pos_barrier_value"].reshape(-1)[0], bv, f"{tag} pos barrier value")
        close(out["pos_barrier_grad"].reshape(-1)[:nq], bg, f"{tag} pos barrier grad")
        close(out["pos_barrier_hess_diag"].reshape(-1), bh, f"{tag} pos barrier hess diag")
        # isfinite-skip: unbounded DOF 0 contributes EXACTLY zero
        assert out["pos_barrier_grad"].reshape(-1)[0] == 0.0, f"{tag} unbounded DOF grad not exactly zero"
        assert out["pos_barrier_hess_diag"].reshape(-1)[0] == 0.0, f"{tag} unbounded DOF hess not exactly zero"

        # velocity barrier (qd block of x)
        bv, bg, _ = barrier_terms(qd, qd - 1.0, qd + 1.0)
        close(out["vel_barrier_value"].reshape(-1)[0], bv, f"{tag} vel barrier value")
        close(out["vel_barrier_grad"].reshape(-1)[nq:nq + nv], bg, f"{tag} vel barrier grad (qd block)")

        # torque barrier (standalone u)
        bv, bg, _ = barrier_terms(u, u - 1.0, u + 1.0)
        close(out["ctrl_barrier_value"].reshape(-1)[0], bv, f"{tag} ctrl barrier value")
        close(out["ctrl_barrier_grad"].reshape(-1), bg, f"{tag} ctrl barrier grad")

        # ---------- plant pass-through: plant == grid::integrator ----------
        close(out["plant_x_kp1"].reshape(-1), out["integrator_x_kp1"].reshape(-1),
              f"{tag} plant_step == grid::integrator (value pass-through)")
        close(out["plant_dAB"].reshape(2 * nv, 3 * nv, order="F"),
              out["integrator_dAB"].reshape(2 * nv, 3 * nv, order="F"),
              f"{tag} plant_step_gradient == grid::integrator_gradient (dAB pass-through)")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("robot_id", _robot_ids(), ids=lambda r: f"{r}-plant-hessian")
def test_cuda_plant_step_hessian_matches_reference(tmp_path, robot_id):
    """CUDA `grid_plant::plant_step_hessian` (s_d2AB) vs the RBDReference oracle.

    The Hessian H[o,a,b] = d^2 x_{k+1}[o] / dz[a] dz[b] composes
    grid::integrator_hessian_device -> fdsva_so_device, dt-scaled per integrator
    (EULER / SI-EULER), fixed-base. The numpy oracle is
    `RBDReference.plant_step_hessian` (the same method the FD-sanity suite
    validates). Per-robot float32 bucket; high-energy samples agree to ~1e-6
    relative, the per-robot atol absorbs static-sample conditioning (never loosen
    a global tolerance).
    """
    base_mode = "fixed"
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    ref = project_model.reference
    build_dir = tmp_path / f"{robot_id}_plant_hessian"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    executable, cmd = _compile_runner(build_dir)

    nv = project_model.nv
    nz = 3 * nv
    samples = _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)

    # Per-robot float32 bucket. The Hessian carries dt^2 (~1e-4) scaling and the
    # M^{-1}-coupled fdsva_so blocks; iiwa14 is well-conditioned so a tight bucket
    # holds. Conditioning-driven static-sample residuals are absorbed by the
    # magnitude-relative atol (rtol * max|expected|), not a global loosen.
    rtol, atol = 2e-3, 2e-3

    def close(actual, expected, msg):
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64), expected,
            rtol=rtol, atol=max(atol, rtol * scale), err_msg=msg,
        )

    integ_blocks = {"euler": "plant_d2AB_euler", "semi_implicit_euler": "plant_d2AB_si_euler"}

    for sample in samples:
        q, qd = np.asarray(sample.q, np.float64), np.asarray(sample.qd, np.float64)
        u = np.asarray(sample.qdd, np.float64)
        out = _run(executable, cmd, q, qd, u, _DT)
        for integrator_type, block in integ_blocks.items():
            tag = f"{robot_id} {integrator_type} @ {sample.name}"
            H_ref = np.asarray(ref.plant_step_hessian(q, qd, u, _DT, integrator_type=integrator_type),
                               dtype=np.float64)
            assert H_ref.shape == (2 * nv, nz, nz)
            # Runner prints a row-major flat 1 x (2nv*nz*nz) vector -> C-order reshape.
            H_cuda = np.asarray(out[block]).reshape(2 * nv, nz, nz)
            close(H_cuda, H_ref, f"{tag} plant_step_hessian vs RBDReference oracle")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(
    "robot_id,base_mode", _centroidal_cells(),
    ids=lambda v: str(v),
)
def test_cuda_plant_centroidal_costs_match_reference(tmp_path, robot_id, base_mode):
    """CUDA `grid_plant::com_cost` / `momentum_cost` vs the RBDReference oracle.

    These centroidal plant costs compose grid::com_device / grid::ccrba_device
    and are emitted NON-MIMIC ONLY, so we validate on iiwa14:fixed and
    go2:floating. The runner drives the device cost kernels (value + gradient +
    GN hessian) with a DETERMINISTIC p_des/h_des/W setup (mirrored here); the
    oracle is the numpy `RBDReference` plant reference (`reference.com_cost` /
    `reference.momentum_cost`), the same path the numpy plant suite uses.
    """
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    ref = project_model.reference
    build_dir = tmp_path / f"{robot_id}_{base_mode}_plant_centroidal"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    executable, cmd = _compile_runner(build_dir)

    nq, nv = project_model.nq, project_model.nv
    nx = nq + nv
    samples = _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)

    rtol, atol = 2e-3, 2e-3

    def close(actual, expected, msg):
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64), expected,
            rtol=rtol, atol=max(atol, rtol * scale), err_msg=msg,
        )

    p_des = _com_pdes(); cW = _com_W()
    h_des = _mom_hdes(); mW = _mom_W()

    for sample in samples:
        q, qd = np.asarray(sample.q, np.float64), np.asarray(sample.qd, np.float64)
        u = np.asarray(sample.qdd, np.float64)
        tag = f"{robot_id}:{base_mode} @ {sample.name}"

        # Degenerate / zero-inertia models (M_total==0) give NaN CoM/CMM; skip
        # those samples (same guard the numpy plant + energy/centroidal suites
        # use). iiwa14/go2 are physical, so this never triggers for them.
        m_total, _com_chk = ref._total_mass_and_com(q)
        if not (np.isfinite(m_total) and m_total != 0.0):
            continue

        out = _run(executable, cmd, q, qd, u, _DT)

        # ---------- CoM-tracking cost (value + grad_x + GN hess_x) ----------
        com_val, com_grad, com_hess = ref.com_cost(q, p_des, cW)
        close(out["com_cost_value"].reshape(-1)[0], com_val, f"{tag} com value")
        close(out["com_cost_grad"].reshape(-1), com_grad, f"{tag} com grad (J_com^T W r; qd-block zero)")
        # qd-block of the CoM-cost gradient must be EXACTLY zero.
        assert np.all(np.asarray(out["com_cost_grad"]).reshape(-1)[nv:] == 0.0), \
            f"{tag} com grad qd-block not exactly zero"
        close(out["com_cost_hess"].reshape(nx, nx, order="F"), com_hess,
              f"{tag} com GN hess (J_com^T W J_com; top-left q-block)")

        # ---------- centroidal-momentum-tracking cost (value + grad_x + GN hess_x) ----------
        mom_val, mom_grad, mom_hess = ref.momentum_cost(q, qd, h_des, mW)
        close(out["momentum_cost_value"].reshape(-1)[0], mom_val, f"{tag} momentum value")
        close(out["momentum_cost_grad"].reshape(-1), mom_grad, f"{tag} momentum grad (A^T W r; q-block zero)")
        # q-block of the momentum-cost gradient must be EXACTLY zero (GN drop).
        assert np.all(np.asarray(out["momentum_cost_grad"]).reshape(-1)[:nq] == 0.0), \
            f"{tag} momentum grad q-block not exactly zero"
        close(out["momentum_cost_hess"].reshape(nx, nx, order="F"), mom_hess,
              f"{tag} momentum GN hess (A^T W A; bottom-right qd-block)")

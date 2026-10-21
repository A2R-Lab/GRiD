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

from grid_codegen import GRiDCodeGenerator
from test.cuda_equivalents.cuda_harness import (
    _build_cuda_samples,
    _detect_cuda_arch,
    _parse_runner_output,
    _run_runner,
)
from test.cuda_equivalents.executable_cache import cached_nvcc_executable
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
# Centroidal CoM-cost setup (3 axes), mirroring com_pdes_val/com_W_val in the runner
# exactly. (The full tangent-state momentum cost is covered by
# test_cuda_momentum_contract.py, which drives the fused dccrba kernel directly.)
def _com_pdes(): return np.array([0.2 + 0.1 * r for r in range(3)], dtype=np.float64)
def _com_W():    return np.array([3.0 + 0.5 * r for r in range(3)], dtype=np.float64)


def _comma_env(name, default):
    raw = os.environ.get(name, default)
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def _robot_ids():
    return _comma_env("GRID_CUDA_PLANT_ROBOTS", "iiwa14")


# (robot_id, base_mode) cells for the MAIN plant-primitive equivalence
# (quadratic costs / ee_pos cost / barriers / plant pass-through).
# Default is iiwa14:fixed + go2:floating + fr3:fixed. The runner now `#if`-gates
# its centroidal block (com/momentum costs) behind GRID_PLANT_HAS_COM_COST /
# GRID_PLANT_HAS_MOMENTUM_COST (mirroring the GRID_PLANT_HAS_STEP_HESSIAN gate),
# so a robot/config that lacks those costs (e.g. fr3 mimic, whose ccrba is gated
# off → centroidal macros absent) compiles cleanly and emits a `*_skipped`
# sentinel for the centroidal blocks. The MAIN test below never reads the
# centroidal blocks, so it passes regardless; the SEPARATE centroidal test skips
# the cell when the sentinel is present.
#
# One pre-existing static-smem limit remains (not a plant correctness bug; the
# runner reports it with a sentinel rather than emitting a wrong number):
#   * go2:floating COMPILES and runs the cost/barrier/centroidal checks, but the
#     plant_step/integrator PASS-THROUGH self-gates off (the fixed-size s_temp[4096]
#     caller pool overflows: FD_DU_MAX_SHARED_MEM_COUNT 12040 > 4096), so the runner
#     emits `plant_step_skipped` and the test checks everything but that block.
#
# Launch thread count (2026-10-15). The runner launches every kernel at
# grid::MAX_PERF_LEVEL_THREADS unless GRID_CUDA_PLANT_THREADS is set. At that
# count two of the default cells could not launch in the -O0 runner build: the
# go2:floating tangent_cost_kernel at 288 threads and the fr3:fixed
# plant_step_kernel at 1024 threads both exceeded the 64K per-block register
# budget ("too many resources requested for launch", receipt evidence: refresh
# 2026-10-12), and because the runner aborts at the first failed launch every
# cell of those two robots was voided, including the plant_kernel and centroidal
# checks that had already launched. The runner's thread count is a test choice,
# not a production launch point (the flagship suite sweeps the generated kernels
# at MAX_PERF_LEVEL_THREADS), so _run launches the plant runner at
# _PLANT_RUNNER_THREADS for every cell: 256 threads fit the register budget by
# construction (255 registers per thread at most, times 256 threads, is within
# 64K) and every plant kernel strides its work over blockDim. Set
# GRID_CUDA_PLANT_THREADS to launch at another count.
# Bigger robots (g1/h1_2, nv~29) additionally overflow the 48 KB static cap in
# plant_kernel/plant_step_kernel (guide §7); the honest big-robot plant path is the
# BINDINGS (dynamic-smem + cudaFuncSetAttribute TU). Override with GRID_CUDA_PLANT_CELLS.
def _plant_cells():
    raw = os.environ.get("GRID_CUDA_PLANT_CELLS", None)
    if raw is None:
        return [("iiwa14", "fixed"), ("go2", "floating"), ("fr3", "fixed")]
    cells = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        robot_id, _, base = tok.partition(":")
        cells.append((robot_id, base or "fixed"))
    return cells


# (robot_id, base_mode) cells for the centroidal (com/momentum) plant-cost check.
# com_cost/momentum_cost emit whenever grid::com_device + grid::ccrba_device are
# present (de-gate #3: the cost emit rides the already-mimic-correct NV-sized
# com/ccrba output, so MIMIC robots are now covered too). Defaults exercise:
# iiwa14:fixed (cheap fixed-base), go2:floating (floating-base), fr3:fixed (mimic).
def _centroidal_cells():
    raw = os.environ.get("GRID_CUDA_PLANT_CENTROIDAL_CELLS", "iiwa14:fixed,go2:floating,fr3:fixed")
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
        # Full profile is JUSTIFIED here (unlike the other aux suites, which use
        # split codegen): plant_step_hessian consumes fdsva_so + integrator, the
        # cost families consume the full ee/centroidal surfaces — restricting
        # would drop the very compositions this suite validates.
        codegen.gen_all_code(codegen_profile="all", output_path=str(header))
    return header


def _compile_runner(build_dir):
    arch = _detect_cuda_arch()
    glass_inc = Path(__file__).resolve().parents[2] / "external" / "GLASS" / "include"
    return cached_nvcc_executable(
        [RUNNER_SOURCE, build_dir / "grid.cuh"],
        ["-std=c++17", "-O0", "-gencode", f"arch=compute_{arch},code=sm_{arch}"],
        exe_name="cuda_plant_smoke_runner.exe", fallback_dir=build_dir,
        include_dirs=[glass_inc], what="CUDA plant smoke runner",
    )


def _stdin(q, qd, u, dt):
    rows = [" ".join(f"{v:.9g}" for v in np.asarray(vec, dtype=np.float32))
            for vec in (q, qd, u)]
    return "\n".join(rows) + f"\n{dt}\n"


# Register-safe launch count for cuda_plant_smoke_runner.cu (see the note above
# _plant_cells). GRID_CUDA_PLANT_THREADS, when set, takes precedence.
_PLANT_RUNNER_THREADS = "256"


@contextlib.contextmanager
def _plant_runner_threads():
    key = "GRID_CUDA_PLANT_THREADS"
    previous = os.environ.get(key)
    if previous is None:
        os.environ[key] = _PLANT_RUNNER_THREADS
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _run(executable, cmd, q, qd, u, dt):
    with _plant_runner_threads():
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
@pytest.mark.parametrize(
    "robot_id,base_mode", _plant_cells(),
    ids=lambda v: str(v),
)
def test_cuda_plant_matches_reference(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_plant"
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
        tag = f"{robot_id}:{base_mode} @ {sample.name}"

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
        grad_q = J.T @ (W * rp)            # velocity-tangent gradient (nv entries)
        H_q = J.T @ np.diag(W) @ J
        H_x = np.zeros((nx, nx)); H_x[:nv, :nv] = H_q
        # the runner prints p(q) it actually used — sanity-check it matches the double oracle
        close(out["ee_pos"].reshape(-1), p, f"{tag} ee_pos vs double Python pose")
        close(out["ee_cost_value"].reshape(-1)[0], ee_val, f"{tag} ee value")
        # The CUDA grad is laid out over x = [q(NQ); qd(NV)] but the EE-position
        # term is a function of the velocity-tangent Jacobian (NV columns), so the
        # leading NV entries hold J^T W r and the qd-block [NQ, NX) is exactly zero.
        # For a floating base (NQ != NV) entries [NV, NQ) are a representation gap
        # (uninitialized in the kernel) and are NOT checked.
        ee_grad = np.asarray(out["ee_cost_grad"]).reshape(-1)
        close(ee_grad[:nv], grad_q, f"{tag} ee grad (J^T W r; velocity-tangent block)")
        # qd-block [NQ, NX) must be EXACTLY zero
        assert np.all(ee_grad[nq:nx] == 0.0), f"{tag} ee grad qd-block not exactly zero"
        close(out["ee_cost_hess"].reshape(nx, nx, order="F"), H_x, f"{tag} ee GN hess (J^T W J)")

        # ---------- FD check (in DOUBLE) of the ee cost gradient via the Python pose ----------
        # central difference of the double-precision ee cost value over each q DOF.
        # Only valid where q is Euclidean (NQ == NV): a raw-q perturbation on a
        # floating base would step off the configuration manifold (quaternion) and
        # would not match the velocity-tangent Jacobian, so skip the FD there (the
        # analytic J^T W r check above already pins the tangent gradient).
        if nq == nv:
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

        # ---------- tracking_cost PRESET == independent per-term composition ----------
        # Fixed-base only (the preset is emitted only there; the runner emits a
        # `tracking_preset_skipped` sentinel otherwise). The runner computes the
        # preset (chained ACCUMULATE) AND an independent reference (each term standalone,
        # summed explicitly) for value / s_qk(NX) / s_rk(NU) / s_Qk(NX*NX) / s_Rk(NU*NU);
        # assert the two agree (the per-term inners are oracle-validated above, so this
        # pins the composition: ACCUMULATE chaining + block offsets + race-free syncs).
        if "tracking_preset_skipped" not in out:
            close(out["tracking_preset_value"].reshape(-1)[0],
                  out["tracking_ref_value"].reshape(-1)[0], f"{tag} tracking preset value == composition")
            close(out["tracking_preset_qk"].reshape(-1), out["tracking_ref_qk"].reshape(-1),
                  f"{tag} tracking preset s_qk == composition")
            close(out["tracking_preset_rk"].reshape(-1), out["tracking_ref_rk"].reshape(-1),
                  f"{tag} tracking preset s_rk == composition")
            close(out["tracking_preset_Qk"].reshape(nx, nx, order="F"),
                  out["tracking_ref_Qk"].reshape(nx, nx, order="F"), f"{tag} tracking preset s_Qk == composition")
            close(out["tracking_preset_Rk"].reshape(nu, nu, order="F"),
                  out["tracking_ref_Rk"].reshape(nu, nu, order="F"), f"{tag} tracking preset s_Rk == composition")

        # ---------- plant pass-through: plant == grid::integrator ----------
        # The runner self-gates the plant_step/integrator pass-through behind
        # `plant_step_fits` (the fixed-size s_temp[4096] caller pool overflows for
        # big floating-base robots, e.g. go2:floating where FD_DU_MAX_SHARED_MEM_COUNT
        # 12040 > 4096) and emits a `plant_step_skipped` sentinel instead. That is a
        # pre-existing static-smem infra limit (the honest big-robot plant path is the
        # bindings' dynamic-smem TU), not a plant correctness bug — so honor the
        # sentinel and skip ONLY the pass-through checks; the cost/barrier checks above
        # already validated for this cell.
        if "plant_step_skipped" in out:
            continue
        close(out["plant_x_kp1"].reshape(-1), out["integrator_x_kp1"].reshape(-1),
              f"{tag} plant_step == grid::integrator (value pass-through)")
        close(out["plant_dAB"].reshape(2 * nv, 3 * nv, order="F"),
              out["integrator_dAB"].reshape(2 * nv, 3 * nv, order="F"),
              f"{tag} plant_step_gradient == grid::integrator_gradient (dAB pass-through)")


# (robot_id, base_mode) cells for the plant_step_hessian (s_d2AB) check. Fixed-base
# is the cheap default gate; go2:floating exercises the SE(3) retract Hessian path
# (the F1 floating emit). Override with GRID_CUDA_PLANT_HESSIAN_CELLS.
def _hessian_cells():
    raw = os.environ.get("GRID_CUDA_PLANT_HESSIAN_CELLS", None)
    if raw is None:
        # default: each robot in _robot_ids() fixed, plus go2 floating.
        return [(r, "fixed") for r in _robot_ids()] + [("go2", "floating")]
    cells = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        robot_id, _, base = tok.partition(":")
        cells.append((robot_id, base or "fixed"))
    return cells


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(
    "robot_id,base_mode", _hessian_cells(),
    ids=lambda v: f"{v}",
)
def test_cuda_plant_step_hessian_matches_reference(tmp_path, robot_id, base_mode):
    """CUDA `grid_plant::plant_step_hessian` (s_d2AB) vs the RBDReference oracle.

    The Hessian H[o,a,b] = d^2 x_{k+1}[o] / dz[a] dz[b] composes
    grid::integrator_hessian_device -> fdsva_so_device per integrator (EULER /
    SI-EULER). Fixed-base is the dt-scaled fdsva_so assembly; floating-base adds
    the SE(3) retract second derivative (position rows) and transposes the
    velocity-row D2qdd axes. The numpy oracle is `RBDReference.plant_step_hessian`
    (the same method the FD-sanity suite validates). Per-(robot,base) float32
    bucket; the per-cell atol absorbs static-sample conditioning (never loosen a
    global tolerance).
    """
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    ref = project_model.reference
    build_dir = tmp_path / f"{robot_id}_{base_mode}_plant_hessian"
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
    # Big robots (g1/h1_2) run the SPILLED tier (s_d2AB + fdsva scratch -> global
    # workspace) and have a much larger |M^{-1}| dynamic range, so the float32
    # round-off floor is higher — give them their own (still magnitude-relative)
    # bucket. go2:floating adds the SE(3) retract second derivative (double-precision
    # FD of the dIntegrate blocks) on top of the spilled fdsva_so path; it sits at
    # ~1e-3 relative (the double FD keeps the float32 kernel matching the float64
    # oracle), so it gets the same 2e-3 default. NEVER loosen a global tolerance;
    # these are per-(robot,base) buckets.
    _hessian_buckets = {
        ("g1", "fixed"):      (5e-3, 5e-3),
        ("h1_2", "fixed"):    (1e-2, 1e-2),
        ("go2", "floating"):  (2e-3, 2e-3),
    }
    rtol, atol = _hessian_buckets.get((robot_id, base_mode), (2e-3, 2e-3))

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
            tag = f"{robot_id}:{base_mode} {integrator_type} @ {sample.name}"
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
    """CUDA `grid_plant::com_cost` vs the RBDReference oracle.

    The CoM cost composes grid::com_device; validated on iiwa14:fixed and
    go2:floating. The runner drives the device cost kernel (value + gradient +
    GN hessian) with a DETERMINISTIC p_des/W setup (mirrored here); the oracle
    is the numpy `RBDReference` plant reference (`reference.com_cost`), the same
    path the numpy plant suite uses. The momentum cost (full tangent-state GN,
    fused dccrba) is covered by `test_cuda_momentum_contract.py`.
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

        # The runner `#if`-gates the centroidal block behind the com/momentum
        # macros and emits a `com_cost_skipped` sentinel when they are absent
        # (e.g. a mimic robot whose ccrba is gated off). Defensive: the default
        # cells here are non-mimic so this normally never triggers, but honor the
        # sentinel cleanly if a mimic cell is passed via the override.
        if "com_cost_skipped" in out:
            pytest.skip(f"{tag}: centroidal costs not emitted (com/ccrba absent)")

        # ---------- CoM-tracking cost (value + grad_x + GN hess_x) ----------
        com_val, com_grad, com_hess = ref.com_cost(q, p_des, cW)
        close(out["com_cost_value"].reshape(-1)[0], com_val, f"{tag} com value")
        close(out["com_cost_grad"].reshape(-1), com_grad, f"{tag} com grad (J_com^T W r; qd-block zero)")
        # qd-block of the CoM-cost gradient must be EXACTLY zero.
        assert np.all(np.asarray(out["com_cost_grad"]).reshape(-1)[nv:] == 0.0), \
            f"{tag} com grad qd-block not exactly zero"
        close(out["com_cost_hess"].reshape(nx, nx, order="F"), com_hess,
              f"{tag} com GN hess (J_com^T W J_com; top-left q-block)")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.floating_base
def test_cuda_tangent_state_cost_matches_reference(tmp_path):
    """GATO ASK3: quadratic_state_cost_tangent (value / gradient / GN hessian /
    full-Newton hessian) vs the RBDReference oracle twin.

    The runner builds x_des ON DEVICE (q_des = integrate(q, fixed delta) — a
    valid quaternion by construction) and dumps it, so the oracle is evaluated
    at the exact same float-rounded reference state; the integrate chart itself
    is covered by the integrator equivalence suite. The Newton hessian carries
    BOTH curvature terms (guide §7.z2) — the value-FD-gated oracle is the
    authority."""
    robot_id, base_mode = "go2", "floating"
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    ref = project_model.reference
    build_dir = tmp_path / f"{robot_id}_{base_mode}_tangent"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    executable, cmd = _compile_runner(build_dir)

    nq, nv = project_model.nq, project_model.nv
    tn = 2 * nv
    samples = _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)
    rtol, atol = 2e-3, 2e-3

    def close(actual, expected, msg):
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64), expected,
            rtol=rtol, atol=max(atol, rtol * scale), err_msg=msg,
        )

    Q2 = 1.0 + 0.5 * np.arange(tn, dtype=np.float64)
    for sample in samples:
        q, qd = np.asarray(sample.q, np.float64), np.asarray(sample.qd, np.float64)
        u = np.asarray(sample.qdd, np.float64)
        out = _run(executable, cmd, q, qd, u, _DT)
        tag = f"{robot_id}:{base_mode} @ {sample.name} (tangent)"
        if "tangent_cost_skipped" in out:
            pytest.skip("tangent preset not emitted for this robot/config")
        x = np.concatenate([q, qd])
        x_des = np.asarray(out["tangent_cost_x_des"], dtype=np.float64).reshape(-1)
        v_gn, g_ref, h_gn = ref.quadratic_state_cost_tangent(x, x_des, Q2, gauss_newton=True)
        _, _, h_nw = ref.quadratic_state_cost_tangent(x, x_des, Q2, gauss_newton=False)
        close(out["tangent_cost_value"].reshape(-1)[0], v_gn, f"{tag} value")
        close(out["tangent_cost_grad"].reshape(-1), g_ref, f"{tag} grad (exact J_diff)")
        close(out["tangent_cost_hess_gn"].reshape(tn, tn, order="F"), h_gn, f"{tag} GN hess")
        close(out["tangent_cost_hess_newton"].reshape(tn, tn, order="F"), h_nw,
              f"{tag} Newton hess (BOTH curvature terms, guide 7.z2)")
        # Newton != GN away from e = 0 — guard against a silently-dead curvature path.
        assert np.max(np.abs(h_nw - h_gn)) > 1e-6, f"{tag}: Newton == GN (curvature path dead?)"


# ---- cost menu, phase one: activated terms, the linear feature and the menu preset ----
_MENU_MX, _MENU_MU = 2, 1
_PSEUDO_HUBER = 1


def _menu_eps(t):      return 0.02 + 0.01 * t
def _lin_Ax(nx):       return np.array([[0.1 * (m + 1) + 0.05 * n * (1 if m == 0 else -1) for n in range(nx)] for m in range(_MENU_MX)], dtype=np.float64)
def _lin_bx():         return np.array([0.3 * m for m in range(_MENU_MX)], dtype=np.float64)
def _lin_tx():         return np.array([0.2 + 0.1 * m for m in range(_MENU_MX)], dtype=np.float64)
def _lin_Wx():         return np.array([4.0 + m for m in range(_MENU_MX)], dtype=np.float64)
def _lin_Au(nu):       return np.array([[0.2 - 0.03 * n for n in range(nu)]], dtype=np.float64)


def _unpack_term(blk, n):
    blk = np.asarray(blk, dtype=np.float64).reshape(-1)
    return blk[0], blk[1:1 + n], blk[1 + n:1 + n + n * n].reshape(n, n, order="F")


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(
    "robot_id,base_mode", _plant_cells(),
    ids=[f"{r}-{b}" for r, b in _plant_cells()],
)
def test_cuda_cost_menu_matches_reference(tmp_path, robot_id, base_mode):
    """state_cost / input_cost / ee_pos_cost_activated / linear_feature_cost under the
    pseudo-Huber activation match RBDReference (value, gradient, GN Hessian); the
    quadratic activation reproduces the quadratic terms bit for bit; and on fixed base
    the tracking_cost_menu preset equals the independent composition of its terms."""
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_menu"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    executable, cmd = _compile_runner(build_dir)
    ref = project_model.reference

    nq, nv = project_model.nq, project_model.nv
    nx, nu = nq + nv, nv
    samples = _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)
    rtol, atol = 2e-3, 2e-3

    def close(actual, expected, msg):
        expected = np.asarray(expected, dtype=np.float64)
        scale = float(np.max(np.abs(expected))) if expected.size else 0.0
        np.testing.assert_allclose(np.asarray(actual, dtype=np.float64), expected,
                                   rtol=rtol, atol=max(atol, rtol * scale), err_msg=msg)

    checked = 0
    for sample in samples:
        q, qd = np.asarray(sample.q, np.float64), np.asarray(sample.qd, np.float64)
        u = np.asarray(sample.qdd, np.float64)
        out = _run(executable, cmd, q, qd, u, _DT)
        if "menu_skipped" in out:
            pytest.skip(f"{robot_id}:{base_mode}: the activated EE cost is not emitted (no ee_pos_cost)")
        x = np.concatenate([q, qd])
        tag = f"{robot_id}:{base_mode} @ {sample.name}"

        val, grad, hess = _unpack_term(out["menu_state"], nx)
        rv, rg, rh = ref.state_cost(x, _x_des(nx), _Qw(nx), _PSEUDO_HUBER, _menu_eps(1))
        close(val, rv, f"{tag} state value"); close(grad, rg, f"{tag} state grad"); close(hess, rh, f"{tag} state hess")

        val, grad, hess = _unpack_term(out["menu_input"], nu)
        rv, rg, rh = ref.input_cost(u, _u_des(nu), _Rw(nu), _PSEUDO_HUBER, _menu_eps(2))
        close(val, rv, f"{tag} input value"); close(grad, rg, f"{tag} input grad"); close(hess, rh, f"{tag} input hess")

        val, grad, hess = _unpack_term(out["menu_ee"], nx)
        rv, rg, rh = ref.ee_pos_cost_activated(q, np.zeros(3), _Ww(), _PSEUDO_HUBER, _menu_eps(0), ee=_PLANT_EE)
        close(val, rv, f"{tag} ee value"); close(grad, rg, f"{tag} ee grad"); close(hess, rh, f"{tag} ee hess")
        assert np.count_nonzero(grad[nv:]) == 0, f"{tag} ee grad qd-block must be zero"

        val, grad, hess = _unpack_term(out["menu_linx"], nx)
        rv, rg, rh = ref.linear_feature_cost(x, _lin_Ax(nx), _lin_bx(), _lin_tx(), _lin_Wx(), _PSEUDO_HUBER, _menu_eps(3))
        close(val, rv, f"{tag} linear x value"); close(grad, rg, f"{tag} linear x grad"); close(hess, rh, f"{tag} linear x hess")

        val, grad, hess = _unpack_term(out["menu_linu"], nu)
        rv, rg, rh = ref.linear_feature_cost(u, _lin_Au(nu), [0.1], [-0.1], [5.0], _PSEUDO_HUBER, _menu_eps(4))
        close(val, rv, f"{tag} linear u value"); close(grad, rg, f"{tag} linear u grad"); close(hess, rh, f"{tag} linear u hess")

        # the quadratic activation is the quadratic term: bit for bit for the state cost
        # (identical arithmetic); to fp32 rounding for the EE cost, whose pose is evaluated
        # in a separate function body (nvcc contracts its multiplies into FMAs differently,
        # one ULP on go2 floating).
        quad = np.asarray(out["menu_quad"], dtype=np.float64).reshape(-1)
        assert quad[0] == np.asarray(out["state_cost_value"], dtype=np.float64).reshape(-1)[0], f"{tag} quadratic activation != quadratic_state_cost"
        ee_val = np.asarray(out["ee_cost_value"], dtype=np.float64).reshape(-1)[0]
        assert abs(quad[1] - ee_val) <= 4e-7 * max(1.0, abs(ee_val)), f"{tag} quadratic activation != ee_pos_cost ({quad[1]} vs {ee_val})"

        if "menu_preset_skipped" not in out:
            blk = np.asarray(out["menu_preset"], dtype=np.float64).reshape(-1)
            cnt = 1 + nx + nu + nx * nx + nu * nu
            preset, reference = blk[:cnt], blk[cnt:]
            close(preset[0], reference[0], f"{tag} menu preset value == composition")
            close(preset[1:1 + nx], reference[1:1 + nx], f"{tag} menu preset s_qk == composition")
            close(preset[1 + nx:1 + nx + nu], reference[1 + nx:1 + nx + nu], f"{tag} menu preset s_rk == composition")
            close(preset[1 + nx + nu:1 + nx + nu + nx * nx], reference[1 + nx + nu:1 + nx + nu + nx * nx], f"{tag} menu preset s_Qk == composition")
            close(preset[1 + nx + nu + nx * nx:], reference[1 + nx + nu + nx * nx:], f"{tag} menu preset s_Rk == composition")
        checked += 1
    assert checked >= 3

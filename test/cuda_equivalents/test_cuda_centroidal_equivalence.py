"""CUDA equivalence test for the generated centroidal / energy device kernels
(D1b) and the grid_plant CoM / centroidal-momentum costs (D1c).

D1b validates the 5 centroidal device functions:
  - generalized_gravity_device  vs RBDReference.generalized_gravity
  - nonlinear_effects_device     vs RBDReference.nonlinear_effects
  - com_device                   vs RBDReference.com / jacobian_com
  - ccrba_device                 vs RBDReference.ccrba (A, h)
  - energy_device                vs RBDReference.{kinetic,potential,mechanical}_energy
The RBDReference numpy oracles match Pinocchio to ~1e-14, so they are the
double-precision ground truth here (the CUDA path is float32, so the comparison
uses a float32-scale tolerance, like the other CUDA smoke tests).

D1c validates the grid_plant CoM / centroidal-momentum tracking costs
(value / gradient over x=[q;qd] / Gauss-Newton hessian) against a NumPy
recompute from the double-precision reference CoM-Jacobian / CMM.

Gravity convention: unified at -9.81. The runner passes gravity = -9.81 to GRiD,
matching the RBDReference oracles' default GRAVITY = -9.81 — both sides now use
one convention (the same pairing the main inverse_dynamics CUDA-equivalence test uses).

Robots: iiwa14-fixed (cheap, gate first) + a floating robot (default go2, an
18-DoF quadruped that compiles quickly; g1/h1_2 also work but their large
all-profile headers take many minutes to nvcc-compile). NOTE: this runner is
NON-MIMIC ONLY — it drives grid_plant com_cost/momentum_cost (and com/ccrba
device fns) which codegen emits for non-mimic robots only (_plant.py). Mimic
validation of the centroidal id-bias `s_vaf` (NB-sized) path needs a dedicated
generalized_gravity/nonlinear_effects-only runner (backlog: f2_audit_findings).
Override with GRID_CUDA_CENTROIDAL_ROBOTS="iiwa14:fixed,g1:floating".
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


RUNNER_SOURCE = Path(__file__).with_name("cuda_centroidal_smoke_runner.cu")

# Mimic-SAFE runner: drives ONLY generalized_gravity / nonlinear_effects (the two
# centroidal primitives codegen emits for mimic robots). The full runner above
# also drives com/ccrba/energy + plant com_cost/momentum_cost, which are non-mimic
# only, so it cannot link against a mimic robot's header. See its module docstring.
MIMIC_RUNNER_SOURCE = Path(__file__).with_name("cuda_centroidal_mimic_smoke_runner.cu")


# ---- deterministic cost setup (MUST match cuda_centroidal_smoke_runner.cu) ----
def _comW():           return np.array([5.0 + r for r in range(3)], dtype=np.float64)
def _momW():           return np.array([2.0 + 0.5 * r for r in range(6)], dtype=np.float64)
def _com_des_off():    return np.array([0.05 * (r + 1) for r in range(3)], dtype=np.float64)
def _mom_des_off():    return np.array([0.1 * (r + 1) for r in range(6)], dtype=np.float64)


def _robot_modes():
    raw = os.environ.get("GRID_CUDA_CENTROIDAL_ROBOTS", "iiwa14:fixed,go2:floating")
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        rid, _, mode = tok.partition(":")
        out.append((rid.strip(), (mode.strip() or "fixed")))
    return out


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
    executable = build_dir / "cuda_centroidal_smoke_runner.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "GLASS" / "include"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-I{glass_inc}", "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA centroidal smoke runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


# ============================================================================
# Mimic-safe centroidal path (generalized_gravity / nonlinear_effects only).
# ============================================================================
def _mimic_robot_modes():
    # fr3:fixed is the canonical small MIMIC robot (NB=NJ=9 > NV=NPOS=8): it exercises
    # the _centroidal.py device-wrapper s_vaf=18*NB NB-sizing on a fixed base (the bias
    # inner writes s_vaf body-indexed by RAW body id; under-sizing by NV overflows into
    # the next arena region). iiwa14:fixed is a NON-mimic CONTROL (NB==NV) so a pass on
    # both proves the mimic path specifically.
    #
    # h1_2:fixed is the BIG mimic (NB=NJ=51 > NV=NPOS=39) and is the regression guard for
    # a real emitter bug this test EXPOSED: the device wrapper lays out s_vaf at 18*NB=918
    # floats, but the HOST arena macro INVERSE_DYNAMICS_BIAS_DYNAMIC_SHARED_MEM_BYTES (GRiDCodeGenerator.py
    # `id_bias_t_count`) used to budget only 18*NV=702, so the wrapper overflowed the
    # dynamic-smem arena by 18*(NB-NV) floats → illegal __shared__ write (fr3 NB-NV=1
    # survived on slack; h1_2 NB-NV=12 crashed). FIXED: the macro now sizes s_vaf by
    # 18*NB for mimic (mirroring the device-wrapper fix; non-mimic byte-identical). h1_2:fixed
    # is kept in the default list so a regression re-trips here.
    # Override the whole list with GRID_CUDA_CENTROIDAL_MIMIC_ROBOTS.
    raw = os.environ.get("GRID_CUDA_CENTROIDAL_MIMIC_ROBOTS",
                         "fr3:fixed,h1_2:fixed,iiwa14:fixed")
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        rid, _, mode = tok.partition(":")
        out.append((rid.strip(), (mode.strip() or "fixed")))
    return out


# Restricted algorithm list: request the two RNEA bias wrappers by their OWN keys
# (R6) — generalized_gravity / nonlinear_effects auto-pull `inverse_dynamics` as
# their dep — and SKIP com/ccrba/energy (kin-domain, not requested + mimic-refused)
# AND skip the grid_plant com_cost/momentum_cost (gen_grid_plant: centroidal_ok
# requires ee_pose + non-mimic). So the resulting header defines ONLY the mimic-
# supported centroidal symbols this runner references — it links for mimic AND
# non-mimic robots alike.
_MIMIC_SAFE_ALGORITHMS = ["generalized_gravity", "nonlinear_effects"]


def _generate_mimic_header(project_model, build_dir):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(project_model.robot, FILE_NAMESPACE="grid")
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(algorithm_list=list(_MIMIC_SAFE_ALGORITHMS),
                             output_path=str(header))
    return header


def _compile_mimic_runner(build_dir):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / MIMIC_RUNNER_SOURCE.name
    shutil.copyfile(MIMIC_RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_centroidal_mimic_smoke_runner.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "GLASS" / "include"
    cmd = [
        nvcc, "-std=c++17", "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-I{glass_inc}", "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA centroidal mimic smoke runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


def _robot_has_mimic(project_model) -> bool:
    return any(
        getattr(j, "is_mimic", False)
        for j in project_model.robot.get_joints_ordered_by_id()
    )


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _mimic_robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_centroidal_mimic_safe_matches_reference(tmp_path, robot_id, base_mode):
    """Mimic-safe centroidal CUDA equivalence: generalized_gravity / nonlinear_effects.

    Drives ONLY the two mimic-supported centroidal id-bias device fns against the
    numpy/pin RBDReference oracle, via a restricted (`id`-only) codegen header so a
    MIMIC robot (fr3:fixed, h1_2:fixed) compiles. iiwa14:fixed is the non-mimic
    control. A PASS on the mimic robot is the runtime confirmation that the
    _centroidal.py s_vaf=18*NB NB-sizing path runs without buffer overflow/corruption.
    """
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_centroidal_mimic"
    build_dir.mkdir()
    _generate_mimic_header(project_model, build_dir)
    executable, cmd = _compile_mimic_runner(build_dir)

    ref = project_model.reference
    nv = project_model.nv
    samples = _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)
    has_mimic = _robot_has_mimic(project_model)

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
        m_total, _ = ref._total_mass_and_com(q)
        if not (np.isfinite(m_total) and m_total != 0.0):
            continue
        out = _run(executable, cmd, q, qd)
        mtag = "mimic" if has_mimic else "non-mimic"
        tag = f"{robot_id}-{base_mode}[{mtag}] @ {sample.name}"

        close(out["gen_gravity"].reshape(-1), ref.generalized_gravity(q),
              f"{tag} generalized_gravity")
        close(out["nonlinear"].reshape(-1), ref.nonlinear_effects(q, qd),
              f"{tag} nonlinear_effects")
        assert nv == out["gen_gravity"].reshape(-1).shape[0], (
            f"{tag} gen_gravity width {out['gen_gravity'].reshape(-1).shape[0]} != nv {nv}"
        )


def _stdin(q, qd):
    rows = [" ".join(f"{v:.9g}" for v in np.asarray(vec, dtype=np.float32)) for vec in (q, qd)]
    return "\n".join(rows) + "\n"


def _run(executable, cmd, q, qd):
    return _parse_runner_output(_run_runner(executable, _stdin(q, qd), cmd))


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize(("robot_id", "base_mode"), _robot_modes(),
                         ids=lambda v: v if isinstance(v, str) else None)
def test_cuda_centroidal_matches_reference(tmp_path, robot_id, base_mode):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(f"Could not resolve manifest {spec.robot_id}: {exc}")
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    build_dir = tmp_path / f"{robot_id}_{base_mode}_centroidal"
    build_dir.mkdir()
    _generate_header(project_model, build_dir)
    executable, cmd = _compile_runner(build_dir)

    ref = project_model.reference
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

    comW, momW = _comW(), _momW()
    com_off, mom_off = _com_des_off(), _mom_des_off()

    for sample in samples:
        q, qd = np.asarray(sample.q, np.float64), np.asarray(sample.qd, np.float64)
        # skip degenerate / non-physical configs (e.g. zero-inertia URDFs)
        m_total, _ = ref._total_mass_and_com(q)
        if not (np.isfinite(m_total) and m_total != 0.0):
            continue
        out = _run(executable, cmd, q, qd)
        tag = f"{robot_id}-{base_mode} @ {sample.name}"

        # ================= D1b: centroidal device kernels =================
        close(out["gen_gravity"].reshape(-1), ref.generalized_gravity(q),
              f"{tag} generalized_gravity")
        close(out["nonlinear"].reshape(-1), ref.nonlinear_effects(q, qd),
              f"{tag} nonlinear_effects")

        p_com = np.asarray(ref.com(q), dtype=np.float64).reshape(-1)
        Jcom = np.asarray(ref.jacobian_com(q), dtype=np.float64)        # 3 x nv
        close(out["com"].reshape(-1), p_com, f"{tag} com")
        close(out["jcom"].reshape(3, nv, order="F"), Jcom, f"{tag} jacobian_com")

        A_ref, h_ref = ref.ccrba(q, qd)
        A_ref = np.asarray(A_ref, dtype=np.float64)                     # 6 x nv
        h_ref = np.asarray(h_ref, dtype=np.float64).reshape(-1)
        close(out["ccrba_A"].reshape(6, nv, order="F"), A_ref, f"{tag} ccrba A")
        close(out["ccrba_h"].reshape(-1), h_ref, f"{tag} ccrba h")

        ke = ref.kinetic_energy(q, qd)
        pe = ref.potential_energy(q)
        me = ref.mechanical_energy(q, qd)
        close(out["energy"].reshape(-1), np.array([ke, pe, me]),
              f"{tag} energy [KE, PE, mechanical]")

        # ================= D1c: grid_plant CoM / momentum costs =================
        # The emitted gradient/hessian layout is over x = [q (nq); qd (nv)],
        # size NX = nq + nv. For a floating base nq > nv (the quaternion uses 4
        # position slots for 3 velocity DOFs), so the gradient's tangent ("q")
        # block occupies the FIRST nv entries (d/dv), entries [nv:nq] are zero
        # padding for the surplus position slots, and the qd block [nq:nx] is
        # zero (com) or carries A^T W r (momentum). The GN hessian is NX x NX
        # with the active block in [0:nv, 0:nv] (com) / [nq:nx, nq:nx] (momentum).
        #
        # p_des / h_des are realized value + fixed offset (mirrors the runner).
        p_des = p_com + com_off
        rc = p_com - p_des
        com_val = 0.5 * np.sum(comW * rc * rc)
        com_grad = np.zeros(nx); com_grad[:nv] = Jcom.T @ (comW * rc)   # [d/dv ; 0 pad ; qd=0]
        H_com = np.zeros((nx, nx)); H_com[:nv, :nv] = Jcom.T @ np.diag(comW) @ Jcom
        close(out["com_cost_value"].reshape(-1)[0], com_val, f"{tag} com_cost value")
        close(out["com_cost_grad"].reshape(-1), com_grad, f"{tag} com_cost grad")
        # the qd block (entries [nq:nx]) must be EXACTLY zero
        assert np.all(np.asarray(out["com_cost_grad"]).reshape(-1)[nq:] == 0.0), \
            f"{tag} com_cost grad qd-block not exactly zero"
        close(out["com_cost_hess"].reshape(nx, nx, order="F"), H_com, f"{tag} com_cost GN hess")

        h_des = h_ref + mom_off
        rm = h_ref - h_des
        mom_val = 0.5 * np.sum(momW * rm * rm)
        mom_grad = np.zeros(nx); mom_grad[nq:] = A_ref.T @ (momW * rm)   # [q-block=0 ; A^T W r]
        H_mom = np.zeros((nx, nx)); H_mom[nq:, nq:] = A_ref.T @ np.diag(momW) @ A_ref
        close(out["mom_cost_value"].reshape(-1)[0], mom_val, f"{tag} momentum_cost value")
        close(out["mom_cost_grad"].reshape(-1), mom_grad, f"{tag} momentum_cost grad")
        # the q block (entries [0:nq]) must be EXACTLY zero (GN drops dA/dq)
        assert np.all(np.asarray(out["mom_cost_grad"]).reshape(-1)[:nq] == 0.0), \
            f"{tag} momentum_cost grad q-block not exactly zero"
        close(out["mom_cost_hess"].reshape(nx, nx, order="F"), H_mom, f"{tag} momentum_cost GN hess")

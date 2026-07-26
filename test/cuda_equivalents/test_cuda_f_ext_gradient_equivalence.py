"""CUDA equivalence for the DEDICATED f_ext_gradient algorithm family (section A
of the differentiability extensions plan), as distinct from
``test_cuda_fext_equivalence.py`` (which checks f_ext as a *parameter* threaded
through inverse_dynamics / fd / aba / their gradients).

This test drives the emitted f_ext-gradient host wrappers and compares their
outputs against the RBDReference + pinocchio oracle:

  dtau_dfext      = -J^T          (nv x 6*NB)            [A.1]  -- exact
  dqdd_dfext      =  M^{-1} J^T   (nv x 6*NB)            [A.2]  -- exact
  did_du_dfext_dq = -dJ^T/dq      (nv x 6*NB x nv)       [A.3]  -- ANALYTIC
                                  (RBDReference.f_ext_jacobian_transpose_dq,
                                  closed form) on BOTH fixed and floating base

All three are q-only (f_ext enters RNEA additively & linearly), so the runner
reads only q. The A.3 block (-dJ^T/dq) is emitted for BOTH base modes as the
ANALYTIC closed form -- both the EMITTED CUDA kernel and the numpy RBDReference
oracle use RBDReference.f_ext_jacobian_transpose_dq's d col_{i,j}/d q_m =
-X_{m->i}(S_m x col_{m,j}) (Featherstone dX[m]/dq_m = -crm(S_m)X[m]); the floating
root's 6 motion-subspace columns slot into the same pushdown with no SE(3) FD. Since
both sides are the same closed form, A.3 holds to a TIGHT f32 floor (was central-FD
vs analytic at ~5e-3). The pinocchio cross-check stays self.integrate(q, dv) FD-of-exact.

Gated iiwa14 (fixed, all three) plus go2 / g1 (floating, all three).
"""
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.equivalents.pinocchio_backend import build_pinocchio_adapter
from RBDReference.tests.state_sampling import build_dynamics_samples
from RBDReference.tests.tolerances import get_tolerance

from grid_codegen import GRiDCodeGenerator

from test.cuda_equivalents.test_cuda_executable_equivalence import (
    _detect_cuda_arch,
    _parse_runner_output,
    GPU_UNAVAILABLE_PATTERNS,
)

RUNNER_SOURCE = Path(__file__).with_name("cuda_f_ext_gradient_runner.cu")

# (robot_id, base_mode). The A.3 -dJ^T/dq block is the ANALYTIC closed form for all.
# iiwa14 (fixed) + go2 / g1 (floating) are NON-MIMIC: each sub-job writes its unique
# output cell directly (no slab). fr3 (fixed + floating) and h1_2 (floating) are MIMIC:
# a mimic joint shares its target's reduced v-slot, so its column folds (alpha-weighted)
# into that shared slot -- the per-sub SLAB + deterministic serial reduce path. h1_2's
# slab (nsub ~ 3.4k -> ~81 KB) exceeds the smem cap, so it exercises the MIMIC-SLAB
# SPILL to the L2-pinned d_workspace SO band (the only case that does); fr3's fits in
# smem. (The full default profile pulls in the still-refused integrator gradients for
# mimic robots, so fr3 / h1_2 are codegen'd with the 'f-ext-gradient' profile
# {id, minv, f_ext_grad}; non-mimic cases use 'all'.)
_CASES = [("iiwa14", "fixed"), ("go2", "floating"), ("g1", "floating"),
          ("fr3", "fixed"), ("fr3", "floating"), ("h1_2", "floating")]

# Robots whose full default codegen profile would hit a still-refused mimic
# gradient (integrator gradients); generate them with the f-ext-gradient profile.
_MIMIC_FEG_PROFILE = {"fr3", "h1_2"}


def _build_adapters(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            spec = case["spec"]
            resolved = resolve_robot_spec(spec)
            proj = build_project_adapter(spec, resolved, base_mode=base_mode)
            pin = build_pinocchio_adapter(spec, resolved, base_mode=base_mode)
            return spec, proj, pin
    pytest.skip(f"case {robot_id}/{base_mode} not in manifest")


def _gen_and_compile(proj, build_dir, floating_base, codegen_profile="all"):
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        proj.robot, DEBUG_MODE=False, NEED_PRINT_MAT=True, FILE_NAMESPACE="grid"
    )
    import contextlib
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(include_homogenous_transforms=True, output_path=str(header),
                             codegen_profile=codegen_profile)
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    arch = _detect_cuda_arch()
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    exe = build_dir / "cuda_f_ext_gradient_runner.exe"
    cmd = [
        nvcc, "-std=c++11", "-O0",
        f"-DGRID_CUDA_FLOATING_BASE={1 if floating_base else 0}",
        "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
        "-I", str(Path(__file__).resolve().parents[2]),
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-gencode", f"arch=compute_{arch},code=compute_{arch}",
        "-o", str(exe), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"compile failed:\n{' '.join(cmd)}\n{result.stdout}\n{result.stderr}")
    return exe


def _run(exe, stdin_text):
    result = subprocess.run(
        [str(exe)], input=stdin_text, cwd=exe.parent, capture_output=True, text=True,
    )
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0:
        if any(p in combined for p in GPU_UNAVAILABLE_PATTERNS):
            pytest.skip("CUDA runtime unavailable.")
        if "shared-memory request" in combined and "this device supports" in combined:
            pytest.skip("Kernel smem request exceeds this GPU's per-block cap.")
        pytest.fail(f"runner failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


@pytest.mark.cuda_equivalence
@pytest.mark.parametrize(("robot_id", "base_mode"), _CASES)
def test_cuda_f_ext_gradient_equivalence(robot_id, base_mode, tmp_path):
    spec, proj, pin = _build_adapters(robot_id, base_mode)
    ref = proj.reference
    nb = ref.robot.get_num_bodies()
    nv = ref.robot.get_num_vel()
    floating = base_mode == "floating"

    sample = build_dynamics_samples(proj)[1]
    q = sample.q

    profile = "f-ext-gradient" if robot_id in _MIMIC_FEG_PROFILE else "all"
    exe = _gen_and_compile(proj, tmp_path, floating, codegen_profile=profile)

    # Mimic robots: GRiD/RBDReference expose a per-BODY f_ext column for ALL NB
    # bodies (the mimic body is a real physical link that can receive an external
    # wrench), so the f_ext-gradient is nv x 6*NB. Pinocchio's reduced model
    # collapses the mimic body, so its f_ext_gradient adapter only exposes the
    # NB-1 actuated bodies (nv x 6*(NB-1)) — a different, incomplete column layout.
    # The two cannot be element-compared, so the pinocchio cross-check is skipped
    # for mimic robots; RBDReference (which the CUDA matches exactly) is the
    # authoritative oracle here.
    has_mimic = any(getattr(j, "is_mimic", False) for j in ref.robot.joints)

    def row(v):
        return " ".join(f"{x:.9g}" for x in np.asarray(v, dtype=np.float32))
    outputs = _parse_runner_output(_run(exe, row(q) + "\n"))

    # oracle (project RBDReference + pinocchio). First-order pair is exact; the
    # A.3 -dJ^T/dq block is the ANALYTIC closed form on the project side for BOTH
    # fixed and floating base (RBDReference.f_ext_jacobian_transpose_dq), while the
    # pinocchio cross-check stays FD-of-exact. (The emitted CUDA kernel FD's
    # on-device for both modes; it is checked against the analytic numpy oracle.)
    a_dtau, a_dqdd, a_djt = proj.f_ext_gradient(q)
    if has_mimic:
        e_dtau = e_dqdd = e_djt = None
    else:
        e_dtau, e_dqdd, e_djt = pin.f_ext_gradient(q)

    def _cuda(name):
        assert name in outputs, f"missing CUDA output {name}; have {list(outputs)}"
        return np.asarray(outputs[name], dtype=np.float64)

    failures = []

    def _check(label, cuda_flat, ref_arr, pin_arr, tol_algo, f32_floor=5e-3):
        ref_arr = np.asarray(ref_arr, dtype=np.float64).reshape(-1)
        cuda_flat = np.asarray(cuda_flat, dtype=np.float64).reshape(-1)
        tol = get_tolerance(tol_algo, robot_id=robot_id)
        scale = max(1.0, float(np.max(np.abs(ref_arr))) if ref_arr.size else 1.0)
        # float32 CUDA path: widen the absolute floor by the value magnitude. The
        # -dJ^T/dq block is now ANALYTIC (was central-FD): both sides are the exact
        # closed form, so it holds to a TIGHT f32 floor (was 5e-3 for the FD step),
        # proving the de-FD. The first-order pair keeps the wider f32 floor.
        atol = tol.atol + tol.rtol * scale + f32_floor * scale
        err_ref = float(np.max(np.abs(cuda_flat - ref_arr))) if ref_arr.size else 0.0
        if err_ref > atol:
            failures.append(f"{label}: CUDA-vs-RBDReference maxerr={err_ref:.3e} > {atol:.3e}")
        # RBDReference == pinocchio (the convention itself); honors the per-robot
        # tolerance (e.g. gen3's RNEA-difference round-off override). Skipped for
        # mimic robots (pinocchio's reduced model omits the mimic body's f_ext
        # column, so the layouts differ — see has_mimic note above).
        if pin_arr is None:
            return
        pin_arr = np.asarray(pin_arr, dtype=np.float64).reshape(-1)
        err_pin = float(np.max(np.abs(ref_arr - pin_arr)))
        ptol = tol.atol + tol.rtol * scale
        if err_pin > ptol:
            failures.append(f"{label}: RBDReference-vs-pinocchio maxerr={err_pin:.3e} > {ptol:.3e}")

    # First-order pair (exact). CUDA layout is nv x 6NB column-major == oracle.
    _check("dtau_dfext", _cuda("f_ext_gradient_dtau_dfext"), a_dtau, e_dtau, "f_ext_grad")
    _check("dqdd_dfext", _cuda("f_ext_gradient_dqdd_dfext"), a_dqdd, e_dqdd, "f_ext_grad")

    # A.3 -dJ^T/dq (both base modes; analytic numpy oracle, on-device-FD CUDA).
    # The runner prints it as a (nv*6NB) x nv
    # matrix with the q-coordinate as the column and the flattened -J^T (row
    # v_j + nv*col, column-major) as the row; rebuild to the oracle's
    # (nv, 6NB, nv) = [v_j, col, qi] layout before comparing.
    cuda_djt = _cuda("f_ext_gradient_did_du_dfext_dq")  # (nv*6NB) x nv
    cuda3 = np.empty((nv, 6 * nb, nv), dtype=np.float64)
    for vj in range(nv):
        for col in range(6 * nb):
            cuda3[vj, col, :] = cuda_djt[vj + nv * col, :]
    _check("did_du_dfext_dq", cuda3, a_djt, e_djt, "f_ext_grad_so", f32_floor=2e-4)

    assert not failures, "f_ext_gradient CUDA equivalence failures:\n" + "\n".join(failures)

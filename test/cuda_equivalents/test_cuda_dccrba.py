"""CUDA equivalence test for the analytic dCCRBA surfaces.

Validates `gen_dccrba` / `gen_cmm_time_variation` (CUDA, run in FLOAT by default;
GRID_EQUIV_T=double selects the fp64 alpha path) against the verified numpy oracle:
  dccrba             -> RBDReference._CentroidalMixin.dccrba(q)            (6 x NV x NV)
  cmm_time_variation -> RBDReference._CentroidalMixin.cmm_time_variation(q,qd)  (6 x NV)

Both oracles are ANALYTIC (closed-form world sweep + spatial operators, validated
vs pin.dccrba / computeCentroidalDynamicsDerivatives ~1e-12). The float runner is
compared at the harness's float default class (rtol = atol = 2e-4, scale-relative);
under GRID_EQUIV_T=double the TIGHT fp64 value bucket applies (1e-8 + 1e-7 * scale).

Cross-check (CUDA-side): sum_m d_dccrba[:,:,m] * qd[m] == d_cmm_time_variation.

Thread-invariance: each case swept over {1, 2, 16, 32, 256} threads; the float64
oracle (not a serial-fp32 accumulation) is the invariance reference (atomicAdd
reassociation is benign; compare against the oracle, not bit-identity). The
run-to-run check between thread counts is held at the same class as the oracle
comparison (float class in float, 1e-6 * scale in double).

Robots: every manifest robot on both bases (widened 2026-10-15), iiwa14 fixed
first as the gate slice. fr3 / h1_2 are the MIMIC robots (NB>NV). Mimic is
SUPPORTED: centroidal_inner's Jw and the per-unit phi here are alpha-folded
(dc_unit_alpha), mirroring the mimic-aware RBDReference oracle. The oracle is
authoritative for mimic (pin omits mimic bodies, so skip the pin cross-check;
the CUDA-vs-RBDReference diff stands). The big robots fit because the runner is
float: the largest arenas are h1_2-floating at 75440 (dccrba) and 76704
(cmm_time_variation) bytes against the 101376-byte per-block cap (see _cases).
"""

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
    _sample_to_stdin,
    CUDA_DEFAULT_TOLERANCE,
    GPU_UNAVAILABLE_PATTERNS,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter


RUNNER_SOURCE = Path(__file__).with_name("cuda_dccrba_smoke_runner.cu")

# Precision of the runner and of the generated header. "float" (default) is the
# validated fp32 path; GRID_EQUIV_T=double builds the header with the
# dtype="double" codegen knob (spill tiers sized for 8-byte values) and the
# runner takes the same variable to run<double> (the fp64 alpha tier), exactly
# as cuda_equivalence_runner.cu does.
_EQUIV_T = "double" if os.environ.get("GRID_EQUIV_T") == "double" else "float"

# (robot_id, base_mode): every manifest robot on both bases, iiwa14 fixed first
# (the gate slice). Override with GRID_CUDA_DCCRBA_ROBOTS="iiwa14:fixed,fr3:fixed".
#
# g1:floating and h1_2:fixed were de-collected on 2026-10-15 in the belief that
# the limit was hardware. It was the double launch at the full tier: in double,
# g1:floating requested 132064 bytes of shared memory for dccrba and h1_2:fixed
# 133488 bytes for cmm_time_variation, against the 101376 bytes this GPU offers
# per block. The same arenas in float are about 66-67 KB and fit, so with the
# float runner (same day) both cells are back, and the rest of the manifest came
# in with them (largest float arenas: h1_2-floating 75440 / 76704 bytes). The
# double path stays reachable under GRID_EQUIV_T=double, where g1-floating
# (132064), h1_2-fixed (131296 / 133488) and h1_2-floating (149632) need a
# complete double spill ladder.
_ALL_ROBOTS = ("iiwa14", "go2", "g1", "h1_2", "fr3", "rizon4", "gen3", "fetch", "baxter")
_DEFAULT_ROBOTS = ",".join(f"{r}:{b}" for b in ("fixed", "floating") for r in _ALL_ROBOTS)


def _cases():
    raw = os.environ.get("GRID_CUDA_DCCRBA_ROBOTS", _DEFAULT_ROBOTS)
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        rid, _, mode = tok.partition(":")
        out.append((rid.strip(), (mode.strip() or "fixed")))
    return out


_CASES = _cases()

_THREAD_COUNTS = (1, 2, 16, 32, 256)


def _robot_spec(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not found in robot manifest.")


def _generate_header(project_model, build_dir: Path) -> Path:
    header_path = build_dir / "grid.cuh"
    # dtype is the codegen knob that sizes the spill tiers (float by default;
    # "double" under GRID_EQUIV_T=double, matching the runner's run<T>).
    codegen = GRiDCodeGenerator(
        project_model.robot, DEBUG_MODE=False, NEED_PRINT_MAT=False, FILE_NAMESPACE="grid",
        dtype=_EQUIV_T,
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(
            include_homogenous_transforms=True,
            output_path=str(header_path),
            algorithm_list=["dccrba", "cmm_time_variation"],
        )
    return header_path


def _compile_runner(build_dir: Path, floating_base: bool):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        pytest.skip("nvcc not found; install CUDA Toolkit to run CUDA equivalence tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_dccrba_runner.exe"
    cmd = [
        nvcc, "-std=c++11", "-O0",
        f"-DGRID_CUDA_FLOATING_BASE={1 if floating_base else 0}",
        "-DGRID_CUDA_LINALG_BACKEND=GRID_LINALG_GLASS",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        "-gencode", f"arch=compute_{arch},code=compute_{arch}",
        "-o", str(executable), str(runner_copy),
    ]
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA dccrba runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return executable, cmd


def _dccrba_flat_to_tensor(flat, nv):
    """CUDA layout dA[row + 6*k + 6*nv*m] -> tensor[row, k, m] (6 x nv x nv)."""
    t = np.empty((6, nv, nv), dtype=np.float64)
    for m in range(nv):
        for k in range(nv):
            for row in range(6):
                t[row, k, m] = flat[row + 6 * k + 6 * nv * m]
    return t


@pytest.mark.cuda_equivalence
@pytest.mark.parametrize(("robot_id", "base_mode"), _CASES)
def test_cuda_dccrba_matches_reference(robot_id, base_mode, tmp_path):
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./install/developer_install.sh "
            f"before executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    robot = project_model.robot
    reference = project_model.reference
    nv = robot.get_num_vel()

    _generate_header(project_model, tmp_path)
    executable, compile_cmd = _compile_runner(tmp_path, base_mode == "floating")

    samples = _build_cuda_samples(project_model, random_count=2)

    failures = []
    for sample in samples:
        q, qd = sample.q, sample.qd

        dA_ref = np.asarray(reference.dccrba(q), dtype=np.float64).reshape(6, nv, nv)
        adot_ref = np.asarray(reference.cmm_time_variation(q, qd), dtype=np.float64).reshape(6, nv)

        dA_scale = max(1.0, float(np.max(np.abs(dA_ref))) if dA_ref.size else 1.0)
        adot_scale = max(1.0, float(np.max(np.abs(adot_ref))) if adot_ref.size else 1.0)
        if _EQUIV_T == "double":
            # tight value bucket (fp64 alpha path): absolute floor + relative on magnitude.
            dA_atol = 1e-8 + 1e-7 * dA_scale
            adot_atol = 1e-8 + 1e-7 * adot_scale
            inv_dA_tol = 1e-6 * dA_scale
            inv_adot_tol = 1e-6 * adot_scale
        else:
            # Float runner since 2026-10-15 (the double path remains under
            # GRID_EQUIV_T=double). Basis: the harness's float default class,
            # CUDA_DEFAULT_TOLERANCE rtol = atol = 2e-4, in its scale-relative
            # form atol_eff = max(atol, rtol * max|expected|), which with
            # scale = max(1, max|expected|) is atol = 2e-4 * max(1, scale).
            # Never looser than that class.
            dA_atol = max(CUDA_DEFAULT_TOLERANCE["atol"], CUDA_DEFAULT_TOLERANCE["rtol"] * dA_scale)
            adot_atol = max(CUDA_DEFAULT_TOLERANCE["atol"], CUDA_DEFAULT_TOLERANCE["rtol"] * adot_scale)
            # Run-to-run thread invariance is held at the same float class: each
            # run is already within that class of the oracle, and fp32 atomicAdd
            # reassociation between thread counts is benign below it.
            inv_dA_tol = dA_atol
            inv_adot_tol = adot_atol

        prev_dA = None
        prev_adot = None
        for nthreads in _THREAD_COUNTS:
            try:
                stdout = _run_runner(executable, _sample_to_stdin(sample), compile_cmd, num_threads=nthreads)
            except Exception as exc:  # noqa: BLE001
                combined = str(exc).lower()
                if any(p in combined for p in GPU_UNAVAILABLE_PATTERNS):
                    pytest.skip("CUDA runtime unavailable.")
                raise
            outputs = _parse_runner_output(stdout)
            config = outputs["dccrba_config"][0]
            np.testing.assert_allclose(
                config[:2], np.asarray([robot.get_num_pos(), nv], dtype=np.float64),
                rtol=0.0, atol=0.0,
                err_msg=f"{robot_id} dccrba dimension config @ {sample.name} (threads={nthreads})",
            )
            dA_flat = np.asarray(outputs["dccrba"], dtype=np.float64).reshape(-1)
            dA_cuda = _dccrba_flat_to_tensor(dA_flat, nv)
            # CUDA adot is column-major s_adot[row + 6*k] -> reshape (6, nv) Fortran-order.
            adot_cuda = np.asarray(outputs["cmm_time_variation"], dtype=np.float64).reshape(6, nv, order="F")

            err_dA = float(np.max(np.abs(dA_cuda - dA_ref))) if dA_ref.size else 0.0
            if err_dA > dA_atol:
                failures.append(
                    f"{robot_id} @ {sample.name} (threads={nthreads}): dccrba CUDA-vs-ref "
                    f"maxerr={err_dA:.3e} > {dA_atol:.3e}"
                )
            err_adot = float(np.max(np.abs(adot_cuda - adot_ref))) if adot_ref.size else 0.0
            if err_adot > adot_atol:
                failures.append(
                    f"{robot_id} @ {sample.name} (threads={nthreads}): cmm_time_variation CUDA-vs-ref "
                    f"maxerr={err_adot:.3e} > {adot_atol:.3e}"
                )

            # CUDA-side contraction cross-check: sum_m dA[:,:,m] qd[m] == Adot
            adot_from_tensor = np.einsum("abm,m->ab", dA_cuda, qd)
            err_xc = float(np.max(np.abs(adot_from_tensor - adot_cuda)))
            if err_xc > adot_atol:
                failures.append(
                    f"{robot_id} @ {sample.name} (threads={nthreads}): contraction cross-check "
                    f"sum_m dA[:,:,m] qd[m] != Adot, maxerr={err_xc:.3e} > {adot_atol:.3e}"
                )

            # thread-invariance vs the oracle-anchored previous run (same class as above).
            if prev_dA is not None:
                t_dA = float(np.max(np.abs(dA_cuda - prev_dA)))
                if t_dA > inv_dA_tol:
                    failures.append(
                        f"{robot_id} @ {sample.name}: dccrba thread-invariance violation "
                        f"(threads={nthreads}) maxdiff={t_dA:.3e}"
                    )
                t_adot = float(np.max(np.abs(adot_cuda - prev_adot)))
                if t_adot > inv_adot_tol:
                    failures.append(
                        f"{robot_id} @ {sample.name}: cmm thread-invariance violation "
                        f"(threads={nthreads}) maxdiff={t_adot:.3e}"
                    )
            prev_dA = dA_cuda
            prev_adot = adot_cuda

    assert not failures, "dccrba CUDA equivalence failures:\n" + "\n".join(failures)

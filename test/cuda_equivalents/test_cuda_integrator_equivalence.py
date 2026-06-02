"""CUDA equivalence test for the generated time-integrator kernels.

Validates `grid::integrator<EULER|SEMI_IMPLICIT_EULER>` and the matching
`integrator_gradient<...>` / `integrator_gradient_with_x_kp1<...>` host
wrappers against the Python reference composed in
`ProjectModelAdapter.integrator` / `integrator_gradient`. Mirrors the
world-frame IDSVA-SO smoke-runner pattern: codegen iiwa14 with the
``integrators`` profile, compile a small CUDA driver that exercises both
integrator types over each sample (q, qd, u, dt), then diff the printed
matrices block-by-block.

Default robot is iiwa14-fixed; pass GRID_CUDA_INTEGRATOR_ROBOTS to widen
the sweep. Set GRID_CUDA_INTEGRATOR_DT to override the integration
timestep (default 0.01).
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
    _sample_to_stdin,
    _thread_counts,
)
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import (
    iter_robot_cases,
    resolve_robot_spec,
)
from RBDReference.equivalents.reference_backend import build_project_adapter


RUNNER_SOURCE = Path(__file__).with_name("cuda_integrator_smoke_runner.cu")

# (prefix, python-side integrator name, has_gradient)
# Both fixed- and floating-base emit value + gradient + both-at-once kernels for
# all five integrators (the floating SI-Euler / Midpoint / RK3 / RK4 gradients
# carry the SE(3) dIntegrate chain-rule wiring). EXCEPTION: floating-base MIMIC
# robots refuse the gradient (the floating multi-stage mimic gradient is deferred;
# see the has_mimic skip in the test body) — those cases are skipped.
_INTEGRATORS = (
    ("integrator_euler",    "euler",                True),
    ("integrator_si_euler", "semi_implicit_euler",  True),
    ("integrator_midpoint", "midpoint",             True),
    ("integrator_rk3",      "rk3",                  True),
    ("integrator_rk4",      "rk4",                  True),
)


def _comma_separated_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _robot_ids() -> tuple[str, ...]:
    # SMALL robots iiwa14/go2 fit at PERF; fr3 is the small MIMIC case (fixed +
    # floating): its integrator gradient COMPOSES the mimic-reduced FD gradient and
    # assembles dAB in reduced NV space. The mimic path needs s_vaf sized 18*NB
    # (NB>NV for fr3 fixed) so the composed FD-grad inner's body-indexed writes
    # don't overflow into s_Minv/s_qdd.
    #
    # BIG robots g1/h1_2 exercise the resource-tier SPILL paths (the integrator
    # gradient's FD-grad inner s_temp / s_D_qdd_stage band routes to d_workspace /
    # d_temp_spill under TIER_LITE/MINIMAL). g1 is non-mimic (NB==NV); h1_2 is the
    # BIG MIMIC case (NB=51>NV=39 fixed, NB=52>NV=45 floating) — its per-body
    # s_vaf/scratch MUST size by NB, not NV, or the composed FD-grad inner overflows
    # (the recurring mimic-overflow bug class). h1_2-floating's multi-stage RK
    # gradient is a KNOWN codegen-refused case (B3 backlog), so for that one cell we
    # validate the value-only integrator path (see _is_value_only_cell).
    return _comma_separated_env("GRID_CUDA_INTEGRATOR_ROBOTS", "iiwa14,go2,fr3,g1,h1_2")


def _dts() -> tuple[float, ...]:
    """Set of dt values to exercise. Override with comma-separated env var."""
    raw = os.environ.get("GRID_CUDA_INTEGRATOR_DT", "0.001,0.01,0.1")
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def _robot_spec(robot_id: str, base_mode: str):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not in manifest")


def _samples(project_model):
    return _build_cuda_samples(project_model, random_count=3, include_corner_samples=True)


# Value-only integrator algorithm list (no integrator_gradient): used for the
# floating-base MIMIC cell (h1_2-floating) whose multi-stage RK gradient is
# codegen-refused (B3 backlog). The value integrator only needs id/minv/fd; with
# integrator_gradient ABSENT, codegen defines GRID_HAS_INTEGRATOR_GRADIENT=0 and
# the runner's #if-guarded gradient block is dropped — so the same runner compiles
# and exercises only the value (x_kp1) path.
_VALUE_ONLY_ALGORITHMS = ["id", "minv", "fd", "integrator"]


def _generate_header(project_model, build_dir: Path, value_only: bool = False) -> Path:
    header = build_dir / "grid.cuh"
    codegen = GRiDCodeGenerator(
        project_model.robot,
        DEBUG_MODE=False,
        NEED_PRINT_MAT=False,
        FILE_NAMESPACE="grid",
    )
    kwargs = (
        dict(algorithm_list=list(_VALUE_ONLY_ALGORITHMS))
        if value_only
        else dict(codegen_profile="integrators")
    )
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        codegen.gen_all_code(output_path=str(header), **kwargs)
    return header


def _compile_runner(build_dir: Path, tier: str | None = None):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc was not found; install CUDA Toolkit to run CUDA tests.")
    runner_copy = build_dir / RUNNER_SOURCE.name
    shutil.copyfile(RUNNER_SOURCE, runner_copy)
    arch = _detect_cuda_arch()
    executable = build_dir / "cuda_integrator_smoke_runner.exe"
    glass_inc = Path(__file__).resolve().parents[2] / "GLASS" / "include"
    cmd = [
        nvcc,
        "-std=c++17",
        "-O0",
        "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        f"-I{glass_inc}",
        "-o", str(executable),
        str(runner_copy),
    ]
    # Tier override so the suite exercises the LITE/MINIMAL SPILL path (the
    # integrator gradient's FD-grad inner s_temp / s_D_qdd_stage band routes to
    # d_workspace / d_temp_spill) in addition to PERF. The math is tier-independent,
    # so a tier sweep must still match the reference. The default `tier` arg comes
    # from the parametrized `tier` fixture (TIER_SHARED + TIER_LITE); the legacy
    # GRID_CUDA_INTEGRATOR_TIER env still overrides it for ad-hoc single-tier runs.
    tier = os.environ.get("GRID_CUDA_INTEGRATOR_TIER", tier)
    if tier and tier != "TIER_SHARED":
        if tier not in ("TIER_PERF", "TIER_LITE", "TIER_MINIMAL"):
            pytest.fail("GRID_CUDA_INTEGRATOR_TIER must be TIER_SHARED (a.k.a. legacy TIER_PERF), TIER_LITE, or TIER_MINIMAL.")
        cmd.insert(-1, f"-DGRID_DEFAULT_RESOURCE_TIER={tier}")
    result = subprocess.run(cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "CUDA integrator smoke runner compilation failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return executable, cmd


def _build_case(project_model, tmp_path, label, tier=None, value_only=False):
    build_dir = tmp_path / label
    build_dir.mkdir()
    _generate_header(project_model, build_dir, value_only=value_only)
    return _compile_runner(build_dir, tier=tier)


def _sample_stdin_with_dt(sample, dt: float) -> str:
    base = _sample_to_stdin(sample)
    return base + f" {dt}\n"


def _run_sample(executable, compile_cmd, sample, dt: float, num_threads=None):
    stdout = _run_runner(executable, _sample_stdin_with_dt(sample, dt), compile_cmd, num_threads=num_threads)
    return _parse_runner_output(stdout)


def _assert_close_scaled(actual, expected, rtol, atol, err_msg):
    """assert_allclose with the absolute floor raised to rtol*max|expected|.

    A structurally-zero entry (e.g. a coupling term that vanishes at this
    operating point) carries float32 round-off ~ rtol*scale; comparing it with
    a fixed tiny atol trips at high velocity/dt even though the kernel is
    correct. Flooring atol at the array's overall scale lets "small relative to
    the matrix" count as close, while a genuine error stays O(scale) and fails."""
    expected_arr = np.asarray(expected, dtype=np.float64)
    scale = float(np.max(np.abs(expected_arr))) if expected_arr.size else 0.0
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=max(atol, rtol * scale), err_msg=err_msg,
    )


def _base_modes() -> tuple[str, ...]:
    return _comma_separated_env("GRID_CUDA_INTEGRATOR_BASE_MODES", "fixed,floating")


def _tiers() -> tuple[str, ...]:
    """Resource tiers to compile+run each cell at. Defaults to PERF (TIER_SHARED)
    AND a spilled tier (TIER_LITE) so the big-robot SPILL path (FD-grad inner
    s_temp / s_D_qdd_stage -> d_workspace / d_temp_spill) is exercised, not just
    the all-in-smem PERF arena. Override with GRID_CUDA_INTEGRATOR_TIERS."""
    return _comma_separated_env("GRID_CUDA_INTEGRATOR_TIERS", "TIER_SHARED,TIER_LITE")


def _robot_has_mimic(project_model) -> bool:
    return any(
        getattr(j, "is_mimic", False)
        for j in project_model.robot.get_joints_ordered_by_id()
    )


@pytest.mark.cuda_equivalence
@pytest.mark.developer_only
@pytest.mark.robot_smoke
@pytest.mark.parametrize("tier", _tiers())
@pytest.mark.parametrize("base_mode", _base_modes())
@pytest.mark.parametrize(
    "robot_id",
    _robot_ids(),
    ids=lambda robot_id: f"{robot_id}-integrator",
)
def test_cuda_integrator_matches_python_reference(tmp_path, robot_id, base_mode, tier):
    """CUDA integrator kernels must match the Python reference composed via FD + Minv.

    Both fixed- and floating-base exercise value + gradient + both-at-once for
    all 5 integrators (Euler / SI-Euler / Midpoint / RK3 / RK4), at PERF
    (TIER_SHARED) AND at a spilled tier (TIER_LITE) so the big-robot (g1/h1_2)
    resource-tier SPILL path (FD-grad inner s_temp / s_D_qdd_stage band -> global
    d_workspace / d_temp_spill) is covered, not just the all-in-smem PERF arena.
    """
    spec = _robot_spec(robot_id, base_mode)
    try:
        resolved = resolve_robot_spec(spec)
    except RuntimeError as exc:
        pytest.skip(
            f"Could not resolve manifest {spec.robot_id}. Run ./developer_install.sh "
            f"before executing CUDA equivalence tests. Resolution error: {exc}"
        )
    project_model = build_project_adapter(spec, resolved, base_mode=base_mode)
    # Floating-base mimic robots (fr3-floating, h1_2-floating): the integrator
    # GRADIENT is still refused by codegen. The fixed-base mimic integrator gradient
    # is fully supported and exact (all 5 integrators); the floating single-stage
    # gradient is correct too, but the floating MULTI-stage (Midpoint/RK3/RK4)
    # gradient is wrong for mimic only and is deferred (B3 backlog) — so the
    # `integrators` profile refuses to emit the floating-mimic gradient rather than
    # ship a silently-wrong RK value path. We do NOT skip the whole cell: instead we
    # codegen the VALUE-ONLY header (algorithm_list w/o integrator_gradient, so
    # GRID_HAS_INTEGRATOR_GRADIENT=0 and the runner drops its #if-guarded gradient
    # block) and validate the value (x_kp1) path for all 5 integrators. Only the
    # refused gradient cells are skipped.
    has_mimic = _robot_has_mimic(project_model)
    value_only = has_mimic and base_mode == "floating"
    label = f"{robot_id}_{base_mode}_{tier}_cuda_integrator"
    executable, compile_cmd = _build_case(
        project_model, tmp_path, label, tier=tier, value_only=value_only
    )
    samples = _samples(project_model)
    dts = _dts()
    nv = project_model.nv
    nq = project_model.nq

    rtol = 5e-4
    atol = 5e-4

    # Gradient kernels are emitted (and thus comparable) only when the header was
    # NOT generated value-only. For the floating-mimic value-only cell the gradient
    # block is absent from the binary, so we assert the value path alone.
    def _gradient_emitted(integrator_type: str) -> bool:
        return not value_only

    # Sweep block thread counts (one warp + multi-warp + a session-random count)
    # to catch thread-count-dependent races; the kernel is compiled once and the
    # thread count is passed to the runner via argv.
    for num_threads in _thread_counts():
      for dt in dts:
        for sample in samples:
            # The shared `DynamicsSample` carries q, qd, qdd — for the integrator
            # the third vector serves as the control torque u.
            actual = _run_sample(executable, compile_cmd, sample, dt, num_threads=num_threads)
            u = sample.qdd
            for prefix, integrator_type, has_gradient in _INTEGRATORS:
                expected_x_kp1 = project_model.integrator(
                    sample.q, sample.qd, u, dt, integrator_type=integrator_type,
                )
                x_kp1_block = np.asarray(actual[prefix + "_x_kp1"], dtype=np.float64).reshape(-1)
                assert x_kp1_block.shape == (nq + nv,), (
                    f"{prefix} x_kp1 shape {x_kp1_block.shape} (expected {(nq + nv,)})"
                )
                _assert_close_scaled(
                    x_kp1_block, expected_x_kp1, rtol, atol,
                    err_msg=f"{robot_id}-{base_mode} {prefix} x_kp1 @ {sample.name} dt={dt} threads={num_threads}",
                )

                if not (has_gradient and _gradient_emitted(integrator_type)):
                    continue

                expected_dAB = project_model.integrator_gradient(
                    sample.q, sample.qd, u, dt, integrator_type=integrator_type,
                )
                dAB_block = np.asarray(actual[prefix + "_dAB"], dtype=np.float64)
                x_kp1_with_block = np.asarray(actual[prefix + "_x_kp1_with_dAB"], dtype=np.float64).reshape(-1)
                dAB_with_block = np.asarray(actual[prefix + "_dAB_with_x_kp1"], dtype=np.float64)

                assert dAB_block.shape == (2 * nv, 3 * nv), (
                    f"{prefix} dAB shape {dAB_block.shape} (expected {(2*nv, 3*nv)})"
                )

                _assert_close_scaled(
                    dAB_block, expected_dAB, rtol, atol,
                    err_msg=f"{robot_id}-{base_mode} {prefix} dAB @ {sample.name} dt={dt} threads={num_threads}",
                )
                _assert_close_scaled(
                    x_kp1_with_block, expected_x_kp1, rtol, atol,
                    err_msg=f"{robot_id}-{base_mode} {prefix} x_kp1_with_dAB @ {sample.name} dt={dt} threads={num_threads}",
                )
                _assert_close_scaled(
                    dAB_with_block, expected_dAB, rtol, atol,
                    err_msg=f"{robot_id}-{base_mode} {prefix} dAB_with_x_kp1 @ {sample.name} dt={dt} threads={num_threads}",
                )
